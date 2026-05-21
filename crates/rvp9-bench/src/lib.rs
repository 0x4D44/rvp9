//! rvp9-bench: shared utilities for the benchmark binaries.
//!
//! See `src/bin/realtime.rs` and `src/bin/bdrate.rs`.

#![allow(missing_docs)] // benchmark binaries; per-fn rustdoc not required

use std::path::Path;

/// Count frames in an IVF file by walking its frame headers.
///
/// IVF format (libvpx output):
///   - 32-byte file header (signature, codec, width, height, framerate)
///   - For each frame: 12-byte sub-header (4 bytes LE size, 8 bytes PTS),
///     then `size` bytes of frame data.
///
/// Returns 0 on malformed/short files.
pub fn count_ivf_frames(path: &Path) -> u32 {
    let Ok(data) = std::fs::read(path) else {
        return 0;
    };
    if data.len() < 32 {
        return 0;
    }
    let mut count: u32 = 0;
    let mut pos: usize = 32;
    while pos + 12 <= data.len() {
        let size =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 12 + size;
        if pos > data.len() {
            // Truncated last frame — don't count it.
            break;
        }
        count = count.saturating_add(1);
    }
    count
}

/// Simple stats over a slice of `f64` samples.
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub n: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

impl Stats {
    /// Compute percentile/mean stats from a pre-sorted slice.
    /// Returns `None` for empty input.
    ///
    /// Percentile uses the nearest-rank method: rank = ceil(p * n), then
    /// `sorted[rank - 1]` (clamped to `[0, n - 1]`). For n=100 sorted
    /// `[1, 2, ..., 100]`, p50=50, p90=90, p95=95, p99=99.
    pub fn from_sorted(sorted: &[f64]) -> Option<Self> {
        if sorted.is_empty() {
            return None;
        }
        let n = sorted.len();
        let nf = n as f64;
        let pct = |p: f64| -> f64 {
            let rank = (p * nf).ceil() as usize;
            let idx = rank.saturating_sub(1).min(n - 1);
            sorted[idx]
        };
        let mean = sorted.iter().sum::<f64>() / nf;
        Some(Stats {
            n,
            min: sorted[0],
            max: sorted[n - 1],
            mean,
            p50: pct(0.50),
            p90: pct(0.90),
            p95: pct(0.95),
            p99: pct(0.99),
        })
    }
}

pub mod throughput;

//
// Y4M (YUV4MPEG2) reader + Y-PSNR
//
// Used by bdrate.rs to compute mean luma PSNR between a source y4m and a
// decoded y4m (the latter produced by `vpxdec -o foo.y4m ivf`).
//
// Y4M format (per spec):
//   Header: "YUV4MPEG2" + space-separated tags + 0x0A
//     Tags: W<w> H<h> F<num>:<den> I<interlace> A<aspect> C<csp> [X...]
//   Frame: "FRAME" [+ optional tags] + 0x0A, then raw planar YUV.
//   For I420 (4:2:0): Y is w*h bytes, U and V are each (w/2)*(h/2).

/// Subset of the Y4M header we actually use.
#[derive(Debug, Clone, Copy)]
pub struct Y4mHeader {
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
}

impl Y4mHeader {
    /// Bytes per frame for I420 (the only colorspace bdrate.rs uses).
    #[inline]
    pub fn frame_bytes_i420(&self) -> usize {
        let w = self.width as usize;
        let h = self.height as usize;
        let y = w * h;
        let chroma = (w / 2) * (h / 2);
        y + 2 * chroma
    }

    #[inline]
    pub fn luma_bytes(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    #[inline]
    pub fn fps(&self) -> f64 {
        f64::from(self.fps_num) / f64::from(self.fps_den.max(1))
    }
}

/// Parse a Y4M header from the start of `data`. Returns the parsed header
/// and the byte offset just past the header's terminating `\n`. The header
/// must start with `YUV4MPEG2 ` (uppercase). Returns `None` on any
/// malformed input, including a missing `W`/`H`/`F` tag.
pub fn parse_y4m_header(data: &[u8]) -> Option<(Y4mHeader, usize)> {
    let nl = data.iter().position(|&b| b == 0x0A)?;
    let line = std::str::from_utf8(&data[..nl]).ok()?;
    if !line.starts_with("YUV4MPEG2 ") && line != "YUV4MPEG2" {
        return None;
    }
    let mut width: Option<u32> = None;
    let mut height: Option<u32> = None;
    let mut fps_num: u32 = 30;
    let mut fps_den: u32 = 1;
    for tag in line.split_ascii_whitespace().skip(1) {
        let (k, v) = tag.split_at(1);
        match k {
            "W" => width = v.parse().ok(),
            "H" => height = v.parse().ok(),
            "F" => {
                let (n, d) = v.split_once(':')?;
                fps_num = n.parse().ok()?;
                fps_den = d.parse().ok()?;
            }
            _ => { /* I, A, C, X: ignored for PSNR purposes */ }
        }
    }
    Some((
        Y4mHeader {
            width: width?,
            height: height?,
            fps_num,
            fps_den,
        },
        nl + 1,
    ))
}

/// Sum of squared differences over two equal-length byte slices.
/// Returned as u64 so an entire HD frame (~2 MB) cannot overflow
/// (max per-pixel = 255*255 = 65025; 2.1M pixels → 1.4e11 < 2^63).
#[inline]
fn sum_squared_diff(a: &[u8], b: &[u8]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: u64 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x as i32) - (y as i32);
        acc += (d * d) as u64;
    }
    acc
}

/// Mean luma PSNR (dB) across all frames between `reference_y4m` and
/// `decoded_y4m`. If a frame is bit-identical, that frame is reported as
/// `MAX_PSNR_DB = 100.0` (libvpx convention) so the mean stays finite.
///
/// Returns `None` if either file cannot be read, headers disagree on
/// (width, height), or the frame walks desync (different frame count).
///
/// Per-plane sample count is taken from the *reference* header.
pub fn mean_y_psnr(reference_y4m: &Path, decoded_y4m: &Path) -> Option<f64> {
    const MAX_PSNR_DB: f64 = 100.0;

    let ref_data = std::fs::read(reference_y4m).ok()?;
    let dec_data = std::fs::read(decoded_y4m).ok()?;
    let (ref_h, mut ref_pos) = parse_y4m_header(&ref_data)?;
    let (dec_h, mut dec_pos) = parse_y4m_header(&dec_data)?;
    if (ref_h.width, ref_h.height) != (dec_h.width, dec_h.height) {
        return None;
    }
    let frame_bytes = ref_h.frame_bytes_i420();
    let luma_bytes = ref_h.luma_bytes();
    let max_squared = 255.0 * 255.0;

    // Helper: read one "FRAME...\n" tag, returning the byte offset just
    // past the trailing newline. `None` at clean end of file.
    let skip_frame_tag = |data: &[u8], pos: usize| -> Option<usize> {
        if pos >= data.len() {
            return None;
        }
        if !data[pos..].starts_with(b"FRAME") {
            return None;
        }
        let rel_nl = data[pos..].iter().position(|&b| b == 0x0A)?;
        Some(pos + rel_nl + 1)
    };

    let mut psnr_sum = 0.0_f64;
    let mut frame_count: u32 = 0;
    loop {
        // Both must have a next frame, or both EOF.
        let r_next = skip_frame_tag(&ref_data, ref_pos);
        let d_next = skip_frame_tag(&dec_data, dec_pos);
        match (r_next, d_next) {
            (None, None) => break,
            (Some(rn), Some(dn)) => {
                if rn + frame_bytes > ref_data.len() || dn + frame_bytes > dec_data.len() {
                    return None; // truncated frame
                }
                let ssd = sum_squared_diff(
                    &ref_data[rn..rn + luma_bytes],
                    &dec_data[dn..dn + luma_bytes],
                );
                let psnr = if ssd == 0 {
                    MAX_PSNR_DB
                } else {
                    let mse = (ssd as f64) / (luma_bytes as f64);
                    10.0 * (max_squared / mse).log10()
                };
                psnr_sum += psnr;
                frame_count += 1;
                ref_pos = rn + frame_bytes;
                dec_pos = dn + frame_bytes;
            }
            _ => return None, // desync
        }
    }
    if frame_count == 0 {
        return None;
    }
    Some(psnr_sum / f64::from(frame_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_empty() {
        assert!(Stats::from_sorted(&[]).is_none());
    }

    #[test]
    fn stats_single() {
        let s = Stats::from_sorted(&[42.0]).unwrap();
        assert_eq!(s.n, 1);
        assert_eq!(s.min, 42.0);
        assert_eq!(s.p50, 42.0);
        assert_eq!(s.p95, 42.0);
        assert_eq!(s.p99, 42.0);
    }

    #[test]
    fn stats_percentiles() {
        let v: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let s = Stats::from_sorted(&v).unwrap();
        assert_eq!(s.n, 100);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 100.0);
        assert_eq!(s.p50, 50.0);
        assert_eq!(s.p90, 90.0);
        assert_eq!(s.p95, 95.0);
        assert_eq!(s.p99, 99.0);
    }

    #[test]
    fn y4m_header_minimal() {
        let data = b"YUV4MPEG2 W1280 H720 F30:1 Ip A1:1 C420jpeg\nFRAME\n\x00\x00\x00";
        let (h, off) = parse_y4m_header(data).unwrap();
        assert_eq!(h.width, 1280);
        assert_eq!(h.height, 720);
        assert_eq!(h.fps_num, 30);
        assert_eq!(h.fps_den, 1);
        assert_eq!(off, 44); // byte just past first '\n'
        assert_eq!(h.luma_bytes(), 1280 * 720);
        assert_eq!(h.frame_bytes_i420(), 1280 * 720 + 2 * 640 * 360);
    }

    #[test]
    fn y4m_header_rejects_garbage() {
        assert!(parse_y4m_header(b"not-y4m\n").is_none());
        assert!(parse_y4m_header(b"YUV4MPEG2 W1280\n").is_none()); // no H
        assert!(parse_y4m_header(b"YUV4MPEG2 H720\n").is_none()); // no W
    }

    #[test]
    fn psnr_identical_frames_caps_at_100() {
        // Synthesize a 2-frame 2x2 Y4M, identical in ref and "decoded".
        // 2x2 I420 = 4 (Y) + 1 (U) + 1 (V) = 6 bytes per frame.
        let mut ref_data = Vec::new();
        ref_data.extend_from_slice(b"YUV4MPEG2 W2 H2 F30:1 Ip A1:1 C420jpeg\n");
        for _ in 0..2 {
            ref_data.extend_from_slice(b"FRAME\n");
            ref_data.extend_from_slice(&[10, 20, 30, 40, 128, 128]);
        }
        let dir = std::env::temp_dir();
        let r = dir.join("rvp9_test_ref.y4m");
        let d = dir.join("rvp9_test_dec.y4m");
        std::fs::write(&r, &ref_data).unwrap();
        std::fs::write(&d, &ref_data).unwrap();
        let psnr = mean_y_psnr(&r, &d).unwrap();
        assert_eq!(psnr, 100.0);
    }

    #[test]
    fn psnr_off_by_one_y_plane() {
        // 1x1 Y4M with single luma value 100 vs 101. MSE = 1, PSNR = 10*log10(255^2) ≈ 48.13.
        let mut ref_data = Vec::new();
        ref_data.extend_from_slice(b"YUV4MPEG2 W2 H2 F30:1 Ip A1:1 C420jpeg\n");
        ref_data.extend_from_slice(b"FRAME\n");
        ref_data.extend_from_slice(&[100, 100, 100, 100, 128, 128]);
        let mut dec_data = Vec::new();
        dec_data.extend_from_slice(b"YUV4MPEG2 W2 H2 F30:1 Ip A1:1 C420jpeg\n");
        dec_data.extend_from_slice(b"FRAME\n");
        dec_data.extend_from_slice(&[101, 101, 101, 101, 128, 128]);
        let dir = std::env::temp_dir();
        let r = dir.join("rvp9_test_ref2.y4m");
        let d = dir.join("rvp9_test_dec2.y4m");
        std::fs::write(&r, &ref_data).unwrap();
        std::fs::write(&d, &dec_data).unwrap();
        let psnr = mean_y_psnr(&r, &d).unwrap();
        // Per-pixel diff = 1; 4 pixels; SSD = 4; MSE = 1; PSNR = 10*log10(65025) ≈ 48.1308
        assert!((psnr - 48.1308).abs() < 0.001, "psnr={psnr}");
    }

    #[test]
    fn psnr_desync_returns_none() {
        let mut ref_data = Vec::new();
        ref_data.extend_from_slice(b"YUV4MPEG2 W2 H2 F30:1 Ip A1:1 C420jpeg\n");
        for _ in 0..2 {
            ref_data.extend_from_slice(b"FRAME\n");
            ref_data.extend_from_slice(&[0u8; 6]);
        }
        // Decoded has only 1 frame.
        let mut dec_data = Vec::new();
        dec_data.extend_from_slice(b"YUV4MPEG2 W2 H2 F30:1 Ip A1:1 C420jpeg\n");
        dec_data.extend_from_slice(b"FRAME\n");
        dec_data.extend_from_slice(&[0u8; 6]);
        let dir = std::env::temp_dir();
        let r = dir.join("rvp9_test_ref3.y4m");
        let d = dir.join("rvp9_test_dec3.y4m");
        std::fs::write(&r, &ref_data).unwrap();
        std::fs::write(&d, &dec_data).unwrap();
        assert!(mean_y_psnr(&r, &d).is_none());
    }
}
