//! Minimal Y4M (YUV4MPEG2) reader for harness tests.
//!
//! Read a y4m source from disk, decode the header, optionally
//! centre-crop each frame to a target resolution. Supports the common
//! 4:2:0 chroma variants (`C420`, `C420jpeg`, `C420mpeg2`, `C420paldv`,
//! and unstated default). Higher chroma is rejected.
//!
//! Returned frames are `(Y, U, V)` triples in tight stride matching the
//! crop dimensions — directly usable as `rvp9_encoder::YuvFrameRef`
//! contents.

use std::io::Read;
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

/// Probe whether a known testsrc y4m corpus file is available at one of
/// the canonical paths. Returns `Some(path)` for the first hit.
///
/// Used by harness tests to opt into real-content runs without making
/// the path-finding logic the test's concern.
#[must_use]
pub fn probe_testsrc_720p() -> Option<std::path::PathBuf> {
    let candidates = [
        std::path::PathBuf::from("tests/corpus/testsrc_720p30.y4m"),
        std::path::PathBuf::from(r"D:\language\rvp9-2\tests\corpus\testsrc_720p30.y4m"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

/// Probe whether the classic Foreman CIF (352×288, ~30 fps, 300 frames)
/// y4m corpus file is available at one of the canonical paths. Returns
/// `Some(path)` for the first hit.
///
/// Foreman is a real-content clip with substantial motion (head turn,
/// camera pan to construction site) and texture that stresses every
/// component (DCT, MV search, MC). It complements the saturating
/// testsrc 720p clip — see `tests/corpus/sources/README.md` for the
/// acquisition steps.
#[must_use]
pub fn probe_foreman_cif() -> Option<std::path::PathBuf> {
    let candidates = [
        std::path::PathBuf::from("tests/corpus/foreman_cif.y4m"),
        std::path::PathBuf::from(r"D:\language\rvp9-2\tests\corpus\foreman_cif.y4m"),
        // Sibling-tree cache populated by the legacy rvp9 repo's fixtures.
        std::path::PathBuf::from(r"D:\language\rvp9\crates\rvp9\tests\data\cache\foreman_cif.y4m"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_some_when_testsrc_exists() {
        // Skip if testsrc isn't present (CI without corpus).
        let Some(p) = probe_testsrc_720p() else {
            return;
        };
        let mut f = std::fs::File::open(&p).expect("open testsrc");
        let hdr = parse_header(&mut f).expect("parse header");
        assert_eq!(hdr.w, 1280);
        assert_eq!(hdr.h, 720);
        assert!(hdr.framerate() > 0.0);
    }

    #[test]
    fn read_cropped_512x512_8_frames_from_testsrc() {
        let Some(p) = probe_testsrc_720p() else {
            return;
        };
        let (frames, framerate) =
            read_cropped(&p, 512, 512, 8).expect("read 8 frames at 512x512");
        assert_eq!(frames.len(), 8);
        assert!(framerate > 0.0);
        // Y plane is 512*512 = 262144 bytes per frame.
        assert_eq!(frames[0].0.len(), 512 * 512);
        // Chroma at half-rate: 256*256 = 65536.
        assert_eq!(frames[0].1.len(), 256 * 256);
        assert_eq!(frames[0].2.len(), 256 * 256);
        // Frames should differ across the sequence (testsrc has motion).
        assert_ne!(frames[0].0, frames[4].0, "testsrc should have motion");
    }

    #[test]
    fn read_cropped_rejects_odd_crop() {
        let Some(p) = probe_testsrc_720p() else {
            return;
        };
        let err = read_cropped(&p, 511, 511, 1).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}
