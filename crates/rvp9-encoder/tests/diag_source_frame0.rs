#![allow(clippy::print_stderr)]

//! Check what frame 0 actually looks like (variance, range).

use rvp9_harness::y4m::{probe_testsrc_720p, read_cropped as y4m_read_cropped};

#[test]
#[ignore = "diagnostic"]
fn source_frame0_stats() {
    const W: u32 = 256;
    const H: u32 = 256;
    let path = probe_testsrc_720p().expect("testsrc");
    let (frames, _fr) = y4m_read_cropped(&path, W, H, 5).expect("read");
    for (i, frame) in frames.iter().enumerate() {
        let y = &frame.0;
        let min = *y.iter().min().unwrap();
        let max = *y.iter().max().unwrap();
        let sum: u64 = y.iter().map(|&b| u64::from(b)).sum();
        let sq: u64 = y.iter().map(|&b| u64::from(b) * u64::from(b)).sum();
        let n = y.len() as u64;
        let mean = sum as f64 / n as f64;
        let var = (sq as f64 / n as f64) - mean * mean;
        eprintln!(
            "  frame {i}: Y min={min}  max={max}  mean={:.2}  var={:.2}",
            mean, var
        );
    }
}
