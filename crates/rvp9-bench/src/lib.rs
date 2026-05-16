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
    let Ok(data) = std::fs::read(path) else { return 0 };
    if data.len() < 32 {
        return 0;
    }
    let mut count: u32 = 0;
    let mut pos: usize = 32;
    while pos + 12 <= data.len() {
        let size = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
            as usize;
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
    pub p99: f64,
}

impl Stats {
    /// Compute percentile/mean stats from a pre-sorted slice.
    /// Returns `None` for empty input.
    ///
    /// Percentile uses the nearest-rank method: rank = ceil(p * n), then
    /// `sorted[rank - 1]` (clamped to `[0, n - 1]`). For n=100 sorted
    /// `[1, 2, ..., 100]`, p50=50, p90=90, p99=99.
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
            p99: pct(0.99),
        })
    }
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
        assert_eq!(s.p99, 42.0);
    }

    #[test]
    fn stats_percentiles() {
        let v: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let s = Stats::from_sorted(&v).unwrap();
        assert_eq!(s.n, 100);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 100.0);
        // p50 of 1..=100 = element at index 49 (0-based) after rounding = 50
        assert_eq!(s.p50, 50.0);
        assert_eq!(s.p90, 90.0);
        assert_eq!(s.p99, 99.0);
    }
}
