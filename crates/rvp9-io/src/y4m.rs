//! Minimal Y4M (YUV4MPEG2) 4:2:0 reader and writer.
//!
//! Read a y4m source from disk, decode the header, optionally
//! centre-crop each frame to a target resolution. Supports the common
//! 4:2:0 chroma variants (`C420`, `C420jpeg`, `C420mpeg2`, `C420paldv`,
//! and unstated default). Higher chroma is rejected.
//!
//! Returned frames are `(Y, U, V)` triples in tight stride matching the
//! crop dimensions — directly usable as `rvp9_encoder::YuvFrameRef`
//! contents.
//!
//! The writer ([`write_header`] + [`write_frame`]) emits the
//! format-correct chroma tag for the decoded stream
//! (`C420jpeg` / `C422` / `C444`); 4:4:0 has no representable y4m tag
//! and is rejected upstream by the CLI. It is the round-trip partner of
//! the reader (on the shared 4:2:0 format), used by `rvp9dec --y4m`.

use rvp9_core::bitstream::ChromaFormat;
use std::io::{self, Read, Write};
use std::path::Path;

/// Match the encoder's supported coded-dimension envelope.
const MAX_Y4M_DIMENSION: u32 = 16_384;

/// Keep malformed or hostile header lines from growing memory without bound.
const MAX_Y4M_HEADER_BYTES: usize = 4 * 1024;

/// Largest supported 8-bit 4:2:0 frame: 16384 × 16384 × 3/2.
const MAX_Y4M_FRAME_BYTES: usize = 384 * 1024 * 1024;

/// Owned tight-stride YUV planes: `(Y, U, V)`.
pub type YuvFrameBuf = (Vec<u8>, Vec<u8>, Vec<u8>);

#[derive(Clone, Copy)]
struct Yuv420Layout {
    width: usize,
    height: usize,
    y_bytes: usize,
    uv_bytes: usize,
    frame_bytes: usize,
}

fn y4m_error(kind: io::ErrorKind, message: impl Into<String>) -> io::Error {
    io::Error::new(kind, message.into())
}

fn checked_yuv420_layout(
    width: u32,
    height: u32,
    kind: io::ErrorKind,
    label: &str,
) -> io::Result<Yuv420Layout> {
    if width == 0 || height == 0 {
        return Err(y4m_error(
            kind,
            format!("{label} dimensions must be non-zero"),
        ));
    }
    if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(y4m_error(
            kind,
            format!("{label} dimensions must be even for 4:2:0"),
        ));
    }
    if width > MAX_Y4M_DIMENSION || height > MAX_Y4M_DIMENSION {
        return Err(y4m_error(
            kind,
            format!(
                "{label} dimensions {width}x{height} exceed the supported \
                 {MAX_Y4M_DIMENSION}x{MAX_Y4M_DIMENSION} envelope"
            ),
        ));
    }

    let width = usize::try_from(width)
        .map_err(|_| y4m_error(kind, format!("{label} width does not fit usize")))?;
    let height = usize::try_from(height)
        .map_err(|_| y4m_error(kind, format!("{label} height does not fit usize")))?;
    let y_bytes = width
        .checked_mul(height)
        .ok_or_else(|| y4m_error(kind, format!("{label} luma size overflows usize")))?;
    let uv_bytes = (width / 2)
        .checked_mul(height / 2)
        .ok_or_else(|| y4m_error(kind, format!("{label} chroma size overflows usize")))?;
    let frame_bytes = uv_bytes
        .checked_mul(2)
        .and_then(|chroma| y_bytes.checked_add(chroma))
        .ok_or_else(|| y4m_error(kind, format!("{label} frame size overflows usize")))?;
    if frame_bytes > MAX_Y4M_FRAME_BYTES {
        return Err(y4m_error(
            kind,
            format!(
                "{label} frame needs {frame_bytes} bytes, above the \
                 {MAX_Y4M_FRAME_BYTES}-byte limit"
            ),
        ));
    }

    Ok(Yuv420Layout {
        width,
        height,
        y_bytes,
        uv_bytes,
        frame_bytes,
    })
}

fn zeroed_buffer(len: usize, label: &str) -> io::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(len).map_err(|error| {
        y4m_error(
            io::ErrorKind::OutOfMemory,
            format!("cannot allocate {label} ({len} bytes): {error}"),
        )
    })?;
    buffer.resize(len, 0);
    Ok(buffer)
}

/// Y4M header fields (only what we use).
#[derive(Debug, Clone, Copy)]
pub struct Y4mHeader {
    /// Source width.
    pub w: u32,
    /// Source height.
    pub h: u32,
    /// Framerate numerator.
    pub fps_num: u32,
    /// Framerate denominator.
    pub fps_den: u32,
}

/// Incremental centre-cropping Y4M reader.
///
/// The reader retains one source-sized frame buffer and returns one owned
/// cropped frame at a time, so callers need not keep the full sequence in
/// memory.
pub struct Y4mReader {
    file: std::fs::File,
    header: Y4mHeader,
    source: Yuv420Layout,
    crop: Yuv420Layout,
    off_x: usize,
    off_y: usize,
    body: Vec<u8>,
    remaining: usize,
}

impl Y4mHeader {
    /// Framerate as a float (`fps_num / fps_den`).
    #[inline]
    #[must_use]
    pub fn framerate(&self) -> f64 {
        f64::from(self.fps_num) / f64::from(self.fps_den)
    }
}

/// Parse the y4m header from `f`'s start position. Leaves `f` positioned
/// just after the header newline.
///
/// Only 4:2:0 chroma variants are accepted; everything else returns
/// `Err`. The format string is fixed at the file's start ("YUV4MPEG2 ...
/// \n").
///
/// # Errors
///
/// Returns [`io::ErrorKind::InvalidData`] if the magic is missing, the
/// chroma is not a 4:2:0 variant, a required `W`/`H`/`F` field is absent
/// or zero, the header exceeds 4 KiB, source geometry is odd, or the source
/// exceeds the supported frame envelope. Returns
/// [`io::ErrorKind::UnexpectedEof`] if the header lacks its terminating
/// newline. Otherwise forwards any underlying read/seek error.
pub fn parse_header(f: &mut std::fs::File) -> std::io::Result<Y4mHeader> {
    use std::io::{BufRead, BufReader, Seek, SeekFrom};
    let mut reader = BufReader::new(&mut *f);
    let mut line = String::new();
    reader
        .by_ref()
        .take((MAX_Y4M_HEADER_BYTES + 1) as u64)
        .read_line(&mut line)?;
    if line.len() > MAX_Y4M_HEADER_BYTES {
        return Err(y4m_error(
            io::ErrorKind::InvalidData,
            format!("y4m header exceeds {MAX_Y4M_HEADER_BYTES} bytes"),
        ));
    }
    if !line.ends_with('\n') {
        return Err(y4m_error(
            io::ErrorKind::UnexpectedEof,
            "EOF before y4m header newline",
        ));
    }
    let bytes_consumed = line.len() as u64;
    if !line.starts_with("YUV4MPEG2 ") {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "missing YUV4MPEG2 magic",
        ));
    }
    drop(reader);
    f.seek(SeekFrom::Start(bytes_consumed))?;

    let mut w = None;
    let mut h = None;
    let mut fps_num = None;
    let mut fps_den = None;
    let mut chroma_ok = true;
    for tok in line.trim_end().split(' ').skip(1) {
        if let Some(v) = tok.strip_prefix('W') {
            w = v.parse().ok();
        } else if let Some(v) = tok.strip_prefix('H') {
            h = v.parse().ok();
        } else if let Some(v) = tok.strip_prefix('F') {
            let mut sp = v.split(':');
            fps_num = sp.next().and_then(|n| n.parse().ok());
            fps_den = sp.next().and_then(|d| d.parse().ok());
        } else if let Some(v) = tok.strip_prefix('C') {
            chroma_ok = matches!(v, "420" | "420jpeg" | "420mpeg2" | "420paldv");
            // W6.2(vpxenc parity, HBD scaffold): recognise the 10/12-bit
            // 4:2:0 y4m chroma tags (`C420p10`, `C420p12`) and reject
            // them with a clear "not yet implemented" error rather than
            // the generic chroma rejection. libvpx ingests these via
            // `vpx_img_alloc` with a `VPX_IMG_FMT_HIGHBITDEPTH` flag
            // (`y4minput.c::y4m_input_open`); the rvp9 HBD pixel pipeline
            // (u16 plane storage + kernel pairs) lands in the W6.4
            // follow-up, so for now we surface a precise diagnostic.
            if matches!(v, "420p10" | "420p12" | "420p16") {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "HBD y4m ingest (C420p10/p12) not yet implemented — \
                     coming in the W6.4 HBD kernel-pair follow-up",
                ));
            }
        }
    }
    if !chroma_ok {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported y4m chroma (need 4:2:0 variant)",
        ));
    }
    match (w, h, fps_num, fps_den) {
        (Some(w), Some(h), Some(fps_num), Some(fps_den)) => {
            if fps_num == 0 || fps_den == 0 {
                return Err(y4m_error(
                    io::ErrorKind::InvalidData,
                    "y4m framerate numerator and denominator must be non-zero",
                ));
            }
            checked_yuv420_layout(w, h, io::ErrorKind::InvalidData, "y4m source")?;
            Ok(Y4mHeader {
                w,
                h,
                fps_num,
                fps_den,
            })
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "missing required y4m field (W/H/F)",
        )),
    }
}

impl Y4mReader {
    /// Open a Y4M input and prepare to return at most `frame_limit` frames,
    /// centre-cropped to `crop_w` × `crop_h`.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] for invalid headers or crop geometry,
    /// allocation failures, and filesystem errors.
    pub fn open(path: &Path, crop_w: u32, crop_h: u32, frame_limit: usize) -> io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let header = parse_header(&mut file)?;
        if crop_w > header.w || crop_h > header.h {
            return Err(y4m_error(
                io::ErrorKind::InvalidInput,
                format!(
                    "crop {crop_w}x{crop_h} exceeds y4m {}x{}",
                    header.w, header.h
                ),
            ));
        }
        if crop_w == 0 || crop_h == 0 {
            return Err(y4m_error(
                io::ErrorKind::InvalidInput,
                "crop dims must be non-zero",
            ));
        }
        if !crop_w.is_multiple_of(2) || !crop_h.is_multiple_of(2) {
            return Err(y4m_error(
                io::ErrorKind::InvalidInput,
                "crop dims must be even",
            ));
        }
        let source =
            checked_yuv420_layout(header.w, header.h, io::ErrorKind::InvalidData, "y4m source")?;
        let crop = checked_yuv420_layout(crop_w, crop_h, io::ErrorKind::InvalidInput, "y4m crop")?;
        let off_x = (source.width - crop.width) / 2;
        let off_y = (source.height - crop.height) / 2;
        let body = zeroed_buffer(source.frame_bytes, "y4m frame body")?;
        Ok(Self {
            file,
            header,
            source,
            crop,
            off_x,
            off_y,
            body,
            remaining: frame_limit,
        })
    }

    /// Parsed source header.
    #[must_use]
    pub const fn header(&self) -> Y4mHeader {
        self.header
    }

    /// Read and crop the next frame, or return `None` at EOF or the limit.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] for malformed frame tags, truncated frame
    /// payloads, allocation failures, and filesystem errors.
    pub fn next_frame(&mut self) -> io::Result<Option<YuvFrameBuf>> {
        if self.remaining == 0 {
            return Ok(None);
        }

        let mut hdr_buf = [0u8; 6];
        let mut header_len = 0;
        while header_len < hdr_buf.len() {
            let read = self.file.read(&mut hdr_buf[header_len..])?;
            if read == 0 {
                if header_len == 0 {
                    return Ok(None);
                }
                return Err(y4m_error(
                    io::ErrorKind::UnexpectedEof,
                    "EOF in FRAME header",
                ));
            }
            header_len += read;
        }
        if &hdr_buf[..5] != b"FRAME" {
            return Err(y4m_error(io::ErrorKind::InvalidData, "missing FRAME tag"));
        }
        if hdr_buf[5] != b'\n' {
            let mut byte = [0u8; 1];
            loop {
                self.file.read_exact(&mut byte).map_err(|error| {
                    if error.kind() == io::ErrorKind::UnexpectedEof {
                        y4m_error(io::ErrorKind::UnexpectedEof, "EOF in FRAME params")
                    } else {
                        error
                    }
                })?;
                if byte[0] == b'\n' {
                    break;
                }
            }
        }
        self.file.read_exact(&mut self.body)?;

        let mut y_crop = zeroed_buffer(self.crop.y_bytes, "cropped luma plane")?;
        for row in 0..self.crop.height {
            let start = (self.off_y + row) * self.source.width + self.off_x;
            y_crop[row * self.crop.width..(row + 1) * self.crop.width]
                .copy_from_slice(&self.body[start..start + self.crop.width]);
        }
        let chroma_off_x = self.off_x / 2;
        let chroma_off_y = self.off_y / 2;
        let chroma_src_w = self.source.width / 2;
        let chroma_crop_w = self.crop.width / 2;
        let chroma_crop_h = self.crop.height / 2;
        let mut u_crop = zeroed_buffer(self.crop.uv_bytes, "cropped U plane")?;
        let mut v_crop = zeroed_buffer(self.crop.uv_bytes, "cropped V plane")?;
        for row in 0..chroma_crop_h {
            let source_offset = (chroma_off_y + row) * chroma_src_w + chroma_off_x;
            let target = row * chroma_crop_w..(row + 1) * chroma_crop_w;
            let u_start = self.source.y_bytes + source_offset;
            u_crop[target.clone()].copy_from_slice(&self.body[u_start..u_start + chroma_crop_w]);
            let v_start = self.source.y_bytes + self.source.uv_bytes + source_offset;
            v_crop[target].copy_from_slice(&self.body[v_start..v_start + chroma_crop_w]);
        }

        self.remaining -= 1;
        Ok(Some((y_crop, u_crop, v_crop)))
    }
}

/// Read up to `n` frames from a y4m file, cropping each frame's centre
/// to `(crop_w, crop_h)`. `crop_w` and `crop_h` MUST be even (chroma).
/// Returns `(frames, framerate)`.
///
/// On EOF before `n` frames are read, returns however many were read
/// (caller should check `frames.len()`).
///
/// # Errors
///
/// Returns [`io::Error`] if the crop exceeds the source dimensions, the
/// crop dims are zero or odd, a required buffer cannot be reserved, a
/// `FRAME` tag is missing, or the stream is truncated mid-frame;
/// forwards any underlying read error.
pub fn read_cropped(
    path: &Path,
    crop_w: u32,
    crop_h: u32,
    n: usize,
) -> std::io::Result<(Vec<YuvFrameBuf>, f64)> {
    let mut reader = Y4mReader::open(path, crop_w, crop_h, n)?;
    let framerate = reader.header().framerate();
    let mut frames = Vec::new();
    frames.try_reserve(n).map_err(|error| {
        y4m_error(
            io::ErrorKind::InvalidInput,
            format!("cannot reserve space for {n} y4m frames: {error}"),
        )
    })?;
    while let Some(frame) = reader.next_frame()? {
        frames.push(frame);
    }

    Ok((frames, framerate))
}

// ============================================================================
// Y4M writer
// ============================================================================
//
// Mirror of the reader in the write direction. Emits a progressive
// stream tagged with the decoded `chroma` format's y4m C-tag
// (`C420jpeg` / `C422` / `C444`). Used by `rvp9dec --y4m`.

/// Write a Y4M stream header for a progressive stream at
/// `(width, height)`, `fps_num / fps_den` fps, with the chroma tag
/// derived from `chroma`.
///
/// Emits `YUV4MPEG2 W<w> H<h> F<num>:<den> Ip A1:1 <Ctag>\n` — every
/// field [`parse_header`] reads. The chroma tag is format-correct:
/// `Yuv420 → C420jpeg`, `Yuv422 → C422`, `Yuv444 → C444`.
///
/// Note the writer/reader pair only round-trips on 4:2:0 — the reader is
/// 4:2:0-only (encoder input path); the writer speaks all three
/// representable formats (decoder output path, which may be Profile-1).
///
/// # Errors
///
/// Returns [`io::ErrorKind::InvalidInput`] for `Yuv440`, which has no
/// standard Y4M chroma tag. Otherwise forwards any [`io::Error`] from the
/// writer.
pub fn write_header<W: Write>(
    out: &mut W,
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    chroma: ChromaFormat,
) -> io::Result<()> {
    // 8-bit y4m C-tags. Yuv440 has no standard tag.
    let tag = match chroma {
        ChromaFormat::Yuv420 => "C420jpeg",
        ChromaFormat::Yuv422 => "C422",
        ChromaFormat::Yuv444 => "C444",
        ChromaFormat::Yuv440 => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Yuv440 has no standard Y4M chroma tag",
            ));
        }
    };
    writeln!(
        out,
        "YUV4MPEG2 W{width} H{height} F{fps_num}:{fps_den} Ip A1:1 {tag}"
    )
}

/// Write one frame: the `FRAME\n` tag followed by the Y, U, V planes
/// (tight stride, no per-row padding) in that order.
///
/// Plane lengths are the caller's responsibility — for a `width ×
/// height` 4:2:0 frame, `y.len() == width * height` and
/// `u.len() == v.len() == (width / 2) * (height / 2)`.
///
/// # Errors
///
/// Forwards any [`io::Error`] from the writer.
pub fn write_frame<W: Write>(out: &mut W, y: &[u8], u: &[u8], v: &[u8]) -> io::Result<()> {
    out.write_all(b"FRAME\n")?;
    out.write_all(y)?;
    out.write_all(u)?;
    out.write_all(v)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writer/reader round-trip: a stream written by [`write_header`] +
    /// [`write_frame`] is parsed back identically by [`parse_header`] +
    /// [`read_cropped`] (crop == full frame). This is the oracle that
    /// keeps the two halves in lock-step.
    #[test]
    fn y4m_write_read_roundtrip() {
        let (w, h) = (16u32, 8u32);
        let y_len = (w * h) as usize;
        let uv_len = ((w / 2) * (h / 2)) as usize;

        // Two distinct frames so we also prove ordering survives.
        let f0: YuvFrameBuf = (
            (0..y_len).map(|i| i as u8).collect(),
            vec![0x40u8; uv_len],
            vec![0x80u8; uv_len],
        );
        let f1: YuvFrameBuf = (
            (0..y_len).map(|i| (255 - i) as u8).collect(),
            vec![0x41u8; uv_len],
            vec![0x81u8; uv_len],
        );

        let mut buf: Vec<u8> = Vec::new();
        write_header(&mut buf, w, h, 30, 1, ChromaFormat::Yuv420).expect("write header");
        write_frame(&mut buf, &f0.0, &f0.1, &f0.2).expect("write f0");
        write_frame(&mut buf, &f1.0, &f1.1, &f1.2).expect("write f1");

        // Round-trip through a temp file (read_cropped takes a Path).
        let path =
            std::env::temp_dir().join(format!("rvp9_io_y4m_roundtrip_{}.y4m", std::process::id()));
        std::fs::write(&path, &buf).expect("write temp");

        let mut file = std::fs::File::open(&path).expect("open");
        let hdr = parse_header(&mut file).expect("parse header");
        assert_eq!((hdr.w, hdr.h, hdr.fps_num, hdr.fps_den), (w, h, 30, 1));

        let (frames, fr) = read_cropped(&path, w, h, 8).expect("read");
        let _ = std::fs::remove_file(&path);

        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], f0);
        assert_eq!(frames[1], f1);
        assert!((fr - 30.0).abs() < 1e-9);
    }

    /// Oracle for the G-B fix: [`write_header`] emits the y4m chroma
    /// tag that matches the decoded stream's format. If the tag were a
    /// constant again (the bug), the `C422` / `C444` cases fail.
    #[test]
    fn write_header_emits_correct_chroma_tag() {
        use rvp9_core::bitstream::ChromaFormat::{Yuv420, Yuv422, Yuv444};
        for (fmt, want) in [(Yuv420, "C420jpeg"), (Yuv422, "C422"), (Yuv444, "C444")] {
            let mut buf = Vec::new();
            write_header(&mut buf, 352, 288, 30, 1, fmt).unwrap();
            let line = String::from_utf8(buf).unwrap();
            assert!(line.contains(&format!(" {want}\n")), "{fmt:?} → {line:?}");
            assert!(line.starts_with("YUV4MPEG2 W352 H288 F30:1 "));
        }
    }

    #[test]
    fn write_header_rejects_unrepresentable_yuv440_without_output() {
        let mut buf = Vec::new();
        let err = write_header(&mut buf, 352, 288, 30, 1, ChromaFormat::Yuv440).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(buf.is_empty());
    }

    // -----------------------------------------------------------------
    // Stage 11 coverage push — header parse + read_cropped error
    // contracts. Each test writes a tiny synthetic y4m to a temp
    // file, then asserts the specific `io::ErrorKind` / payload.
    // -----------------------------------------------------------------

    fn write_temp(content: &[u8], suffix: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "rvp9_io_y4m_test_{}_{}.y4m",
            std::process::id(),
            suffix,
        ));
        std::fs::write(&path, content).expect("write temp");
        path
    }

    #[test]
    fn parse_header_rejects_missing_magic() {
        let path = write_temp(b"NOTYUV4MPEG2 W16 H16 F30:1\n", "no_magic");
        let mut f = std::fs::File::open(&path).expect("open");
        let err = parse_header(&mut f).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(format!("{err}").contains("magic"));
    }

    #[test]
    fn parse_header_rejects_unsupported_chroma() {
        // C444 is not a 4:2:0 variant — rejected.
        let path = write_temp(b"YUV4MPEG2 W16 H16 F30:1 C444\n", "c444");
        let mut f = std::fs::File::open(&path).expect("open");
        let err = parse_header(&mut f).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(format!("{err}").contains("chroma"));
    }

    /// W6.2 PROOF — 10/12-bit y4m chroma tags are rejected with a
    /// precise HBD-not-implemented diagnostic (libvpx
    /// `y4minput.c::y4m_input_open` ingests them via the HBD image
    /// flag; rvp9's HBD pixel pipeline lands in W6.4). If the HBD
    /// recognition is stripped, these tags fall through to the generic
    /// "unsupported chroma" error and the `.kind()` assertion fails.
    #[test]
    fn w6_2_parse_header_rejects_hbd_chroma_with_precise_error() {
        for variant in ["420p10", "420p12", "420p16"] {
            let line = format!("YUV4MPEG2 W16 H16 F25:1 C{variant}\n");
            let path = write_temp(line.as_bytes(), variant);
            let mut f = std::fs::File::open(&path).expect("open");
            let err = parse_header(&mut f).unwrap_err();
            let _ = std::fs::remove_file(&path);
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::Unsupported,
                "variant {variant} must reject as Unsupported"
            );
            assert!(
                format!("{err}").contains("HBD"),
                "variant {variant}: error must name HBD: {err}"
            );
        }
    }

    #[test]
    fn parse_header_accepts_c420jpeg_c420paldv_c420mpeg2() {
        for variant in ["420", "420jpeg", "420mpeg2", "420paldv"] {
            let line = format!("YUV4MPEG2 W16 H16 F25:1 C{variant}\n");
            let path = write_temp(line.as_bytes(), variant);
            let mut f = std::fs::File::open(&path).expect("open");
            let hdr = parse_header(&mut f).unwrap_or_else(|e| panic!("variant {variant}: {e}"));
            let _ = std::fs::remove_file(&path);
            assert_eq!(hdr.w, 16);
            assert_eq!(hdr.h, 16);
            assert_eq!(hdr.fps_num, 25);
            assert_eq!(hdr.fps_den, 1);
        }
    }

    #[test]
    fn parse_header_rejects_missing_w_h_f() {
        // No W, H, or F fields.
        let path = write_temp(b"YUV4MPEG2 C420\n", "no_whf");
        let mut f = std::fs::File::open(&path).expect("open");
        let err = parse_header(&mut f).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(format!("{err}").contains("required"));
    }

    #[test]
    fn parse_header_rejects_zero_geometry_and_framerate() {
        for (suffix, line) in [
            ("zero_w", "YUV4MPEG2 W0 H16 F30:1 C420\n"),
            ("zero_h", "YUV4MPEG2 W16 H0 F30:1 C420\n"),
            ("zero_fps_num", "YUV4MPEG2 W16 H16 F0:1 C420\n"),
            ("zero_fps_den", "YUV4MPEG2 W16 H16 F30:0 C420\n"),
        ] {
            let path = write_temp(line.as_bytes(), suffix);
            let mut f = std::fs::File::open(&path).expect("open");
            let err = parse_header(&mut f).expect_err("zero-valued header field was accepted");
            let _ = std::fs::remove_file(&path);
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{suffix}");
        }
    }

    #[test]
    fn parse_header_rejects_odd_or_oversized_420_geometry() {
        for (suffix, line) in [
            ("odd_w", "YUV4MPEG2 W15 H16 F30:1 C420\n"),
            ("odd_h", "YUV4MPEG2 W16 H15 F30:1 C420\n"),
            (
                "oversized",
                "YUV4MPEG2 W4294967294 H4294967294 F30:1 C420\n",
            ),
        ] {
            let path = write_temp(line.as_bytes(), suffix);
            let mut f = std::fs::File::open(&path).expect("open");
            let err = parse_header(&mut f).expect_err("unsafe 4:2:0 geometry was accepted");
            let _ = std::fs::remove_file(&path);
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{suffix}");
        }
    }

    #[test]
    fn header_framerate_returns_ratio() {
        let h = Y4mHeader {
            w: 100,
            h: 100,
            fps_num: 60,
            fps_den: 2,
        };
        let fr = h.framerate();
        assert!((fr - 30.0).abs() < 1e-9, "framerate computed wrong: {fr}");
    }

    #[test]
    fn read_cropped_rejects_oversize_crop() {
        // Build a minimal y4m header (no frames) and ask for a crop
        // larger than the source.
        let path = write_temp(b"YUV4MPEG2 W16 H16 F30:1 C420\n", "oversize");
        let err = read_cropped(&path, 32, 32, 1).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(format!("{err}").contains("exceeds"));
    }

    #[test]
    fn read_cropped_rejects_unallocatable_frame_count_without_panicking() {
        let path = write_temp(
            b"YUV4MPEG2 W16 H16 F30:1 C420\n",
            "unallocatable_frame_count",
        );
        let outcome = std::panic::catch_unwind(|| read_cropped(&path, 16, 16, usize::MAX));
        let _ = std::fs::remove_file(&path);

        match outcome {
            Ok(Err(error)) => assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput),
            Ok(Ok(_)) => panic!("unallocatable frame count was accepted"),
            Err(_) => panic!("unallocatable frame count panicked"),
        }
    }

    #[test]
    fn read_cropped_rejects_odd_crop() {
        let path = write_temp(b"YUV4MPEG2 W32 H32 F30:1 C420\n", "odd_crop");
        let err = read_cropped(&path, 5, 4, 1).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(format!("{err}").contains("even"));
    }

    #[test]
    fn read_cropped_zero_n_returns_empty_immediately() {
        let path = write_temp(b"YUV4MPEG2 W16 H16 F30:1 C420\n", "n0");
        let (frames, fr) = read_cropped(&path, 16, 16, 0).expect("read");
        let _ = std::fs::remove_file(&path);
        assert!(frames.is_empty(), "n=0 returned {} frames", frames.len());
        assert!((fr - 30.0).abs() < 1e-9);
    }

    #[test]
    fn incremental_reader_stops_before_unread_bad_frame_tag() {
        let mut bytes = b"YUV4MPEG2 W16 H16 F30:1 C420\nFRAME\n".to_vec();
        bytes.extend(std::iter::repeat_n(128, 16 * 16 * 3 / 2));
        bytes.extend_from_slice(b"BROKEN\n");
        let path = write_temp(&bytes, "stream_limit");

        let mut limited = Y4mReader::open(&path, 16, 16, 1).expect("open limited reader");
        assert!(limited.next_frame().expect("read first frame").is_some());
        assert!(limited.next_frame().expect("honor limit").is_none());

        let mut unlimited =
            Y4mReader::open(&path, 16, 16, usize::MAX).expect("open unlimited reader");
        assert!(unlimited.next_frame().expect("read first frame").is_some());
        let error = unlimited.next_frame().expect_err("bad tail accepted");
        let _ = std::fs::remove_file(&path);
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("FRAME tag"));
    }

    #[test]
    fn read_cropped_eof_before_frame_breaks_loop() {
        // Valid header, no FRAME body → loop iter 0's read_exact
        // fails → break.
        let path = write_temp(b"YUV4MPEG2 W16 H16 F30:1 C420\n", "eof");
        let (frames, fr) = read_cropped(&path, 16, 16, 5).expect("read");
        let _ = std::fs::remove_file(&path);
        assert!(frames.is_empty());
        assert!((fr - 30.0).abs() < 1e-9);
    }

    #[test]
    fn read_cropped_missing_frame_tag_errors() {
        // 6-byte body that is NOT b"FRAME\n" (and not b"FRAME?…\n").
        let mut buf = b"YUV4MPEG2 W16 H16 F30:1 C420\n".to_vec();
        buf.extend_from_slice(b"NOTAGZ"); // 6 bytes not starting with FRAME
        let path = write_temp(&buf, "no_frame_tag");
        let err = read_cropped(&path, 16, 16, 1).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(format!("{err}").contains("FRAME"));
    }

    #[test]
    fn read_cropped_frame_with_params_consumes_until_newline() {
        // FRAME tag followed by params (e.g. " I0") that consume the
        // 6th byte and then more bytes until a newline. Body of
        // 16*16 + 2*(8*8) = 384 bytes follows.
        let mut buf = b"YUV4MPEG2 W16 H16 F30:1 C420\n".to_vec();
        buf.extend_from_slice(b"FRAME"); // 5 bytes
        buf.push(b' '); // 6th byte is space — not newline → enters slow-path loop
        buf.extend_from_slice(b"I0\n"); // params terminated by newline
        // Body: 16*16 y + 2*8*8 uv = 384 bytes of zeros.
        buf.extend_from_slice(&vec![0u8; 256 + 2 * 64]);
        let path = write_temp(&buf, "frame_params");
        let (frames, _) = read_cropped(&path, 16, 16, 1).expect("read with params");
        let _ = std::fs::remove_file(&path);
        assert_eq!(frames.len(), 1);
        // All zeros in → all zeros out (no offset, full-frame crop).
        assert!(frames[0].0.iter().all(|&b| b == 0));
    }

    #[test]
    fn read_cropped_frame_params_eof_errors() {
        // FRAME + non-newline byte, then EOF inside the params loop.
        let mut buf = b"YUV4MPEG2 W16 H16 F30:1 C420\n".to_vec();
        buf.extend_from_slice(b"FRAME"); // 5 bytes
        buf.push(b' '); // 6th byte
        // No newline before EOF → triggers
        // "EOF in FRAME params" error.
        let path = write_temp(&buf, "frame_eof");
        let err = read_cropped(&path, 16, 16, 1).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
        assert!(format!("{err}").contains("FRAME params"));
    }

    #[test]
    fn read_cropped_centre_crop_offset_is_correct() {
        // 16x16 source, crop 8x8 → off_x = (16-8)/2 = 4, off_y = 4.
        // Fill the Y plane with a known pattern so we can verify the
        // crop window.
        let mut buf = b"YUV4MPEG2 W16 H16 F30:1 C420\n".to_vec();
        buf.extend_from_slice(b"FRAME\n");
        // Y: each byte = row*16 + col. So Y[row=4..12][col=4..12].
        let mut y = vec![0u8; 16 * 16];
        for r in 0..16 {
            for c in 0..16 {
                y[r * 16 + c] = (r * 16 + c) as u8;
            }
        }
        buf.extend_from_slice(&y);
        // U/V planes — constant, irrelevant.
        buf.extend_from_slice(&[64u8; 8 * 8]);
        buf.extend_from_slice(&[192u8; 8 * 8]);
        let path = write_temp(&buf, "centre_crop");
        let (frames, _) = read_cropped(&path, 8, 8, 1).expect("read centre");
        let _ = std::fs::remove_file(&path);
        assert_eq!(frames.len(), 1);
        let (y_crop, u_crop, v_crop) = &frames[0];
        // First row of crop is source row 4, cols 4..12.
        assert_eq!(y_crop[0], (4u32 * 16 + 4) as u8);
        assert_eq!(y_crop[7], (4u32 * 16 + 11) as u8);
        // Last row of crop is source row 11, cols 4..12.
        assert_eq!(y_crop[7 * 8 + 7], (11u32 * 16 + 11) as u8);
        // U/V planes are constants → entire crop is that constant.
        assert!(u_crop.iter().all(|&b| b == 64));
        assert!(v_crop.iter().all(|&b| b == 192));
    }
}
