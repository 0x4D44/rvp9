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
//! The writer ([`write_header`] + [`write_frame`]) emits a 4:2:0
//! `C420jpeg` stream — the round-trip partner of the reader, used by
//! `rvp9dec --y4m`.

use std::io::{self, Read, Write};
use std::path::Path;

/// Owned tight-stride YUV planes: `(Y, U, V)`.
pub type YuvFrameBuf = (Vec<u8>, Vec<u8>, Vec<u8>);

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
/// Returns [`io::Error`] (`InvalidData`) if the magic is missing, the
/// chroma is not a 4:2:0 variant, or a required `W`/`H`/`F` field is
/// absent; forwards any underlying read/seek error.
pub fn parse_header(f: &mut std::fs::File) -> std::io::Result<Y4mHeader> {
    use std::io::{BufRead, BufReader, Seek, SeekFrom};
    let mut reader = BufReader::new(&mut *f);
    let mut line = String::new();
    reader.read_line(&mut line)?;
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
        (Some(w), Some(h), Some(fps_num), Some(fps_den)) => Ok(Y4mHeader {
            w,
            h,
            fps_num,
            fps_den,
        }),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "missing required y4m field (W/H/F)",
        )),
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
/// crop dims are odd, a `FRAME` tag is missing, or the stream is
/// truncated mid-frame; forwards any underlying read error.
pub fn read_cropped(
    path: &Path,
    crop_w: u32,
    crop_h: u32,
    n: usize,
) -> std::io::Result<(Vec<YuvFrameBuf>, f64)> {
    let mut f = std::fs::File::open(path)?;
    let hdr = parse_header(&mut f)?;

    if crop_w > hdr.w || crop_h > hdr.h {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("crop {crop_w}x{crop_h} exceeds y4m {}x{}", hdr.w, hdr.h),
        ));
    }
    if !crop_w.is_multiple_of(2) || !crop_h.is_multiple_of(2) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "crop dims must be even",
        ));
    }

    let src_w = hdr.w as usize;
    let src_h = hdr.h as usize;
    let cw = crop_w as usize;
    let ch = crop_h as usize;
    let off_x = (src_w - cw) / 2;
    let off_y = (src_h - ch) / 2;
    let chroma_off_x = off_x / 2;
    let chroma_off_y = off_y / 2;

    let y_size = src_w * src_h;
    let uv_size = (src_w / 2) * (src_h / 2);
    let frame_body = y_size + 2 * uv_size;

    let mut frames = Vec::with_capacity(n);
    let mut hdr_buf = [0u8; 6];
    let mut body = vec![0u8; frame_body];

    for _ in 0..n {
        if f.read_exact(&mut hdr_buf).is_err() {
            break;
        }
        if &hdr_buf[..5] != b"FRAME" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "missing FRAME tag",
            ));
        }
        if hdr_buf[5] != b'\n' {
            let mut b = [0u8; 1];
            loop {
                if f.read_exact(&mut b).is_err() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "EOF in FRAME params",
                    ));
                }
                if b[0] == b'\n' {
                    break;
                }
            }
        }
        f.read_exact(&mut body)?;

        let mut y_crop = vec![0u8; cw * ch];
        for r in 0..ch {
            let src_row = &body[(off_y + r) * src_w + off_x..(off_y + r) * src_w + off_x + cw];
            y_crop[r * cw..(r + 1) * cw].copy_from_slice(src_row);
        }
        let u_off = y_size;
        let v_off = y_size + uv_size;
        let chroma_src_w = src_w / 2;
        let chroma_cw = cw / 2;
        let chroma_ch = ch / 2;
        let mut u_crop = vec![0u8; chroma_cw * chroma_ch];
        let mut v_crop = vec![0u8; chroma_cw * chroma_ch];
        for r in 0..chroma_ch {
            let u_row_off = u_off + (chroma_off_y + r) * chroma_src_w + chroma_off_x;
            u_crop[r * chroma_cw..(r + 1) * chroma_cw]
                .copy_from_slice(&body[u_row_off..u_row_off + chroma_cw]);
            let v_row_off = v_off + (chroma_off_y + r) * chroma_src_w + chroma_off_x;
            v_crop[r * chroma_cw..(r + 1) * chroma_cw]
                .copy_from_slice(&body[v_row_off..v_row_off + chroma_cw]);
        }

        frames.push((y_crop, u_crop, v_crop));
    }

    Ok((frames, hdr.framerate()))
}

// ============================================================================
// Y4M writer
// ============================================================================
//
// Mirror of the reader in the write direction. Emits a `C420jpeg`
// 4:2:0 progressive stream. Used by `rvp9dec --y4m`.

/// Write a Y4M stream header for a 4:2:0 progressive stream at
/// `(width, height)`, `fps_num / fps_den` fps.
///
/// Emits `YUV4MPEG2 W<w> H<h> F<num>:<den> Ip A1:1 C420jpeg\n` — every
/// field [`parse_header`] reads, with the chroma tag set to a variant
/// the reader accepts (so the writer/reader pair round-trips).
///
/// # Errors
///
/// Forwards any [`io::Error`] from the writer.
pub fn write_header<W: Write>(
    out: &mut W,
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
) -> io::Result<()> {
    writeln!(out, "YUV4MPEG2 W{width} H{height} F{fps_num}:{fps_den} Ip A1:1 C420jpeg")
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
        write_header(&mut buf, w, h, 30, 1).expect("write header");
        write_frame(&mut buf, &f0.0, &f0.1, &f0.2).expect("write f0");
        write_frame(&mut buf, &f1.0, &f1.1, &f1.2).expect("write f1");

        // Round-trip through a temp file (read_cropped takes a Path).
        let path = std::env::temp_dir().join(format!(
            "rvp9_io_y4m_roundtrip_{}.y4m",
            std::process::id()
        ));
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
