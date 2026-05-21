#![allow(clippy::print_stderr)]

//! Per-SB motion analysis: where in the frame is the testsrc motion?

use rvp9_harness::y4m::{probe_testsrc_720p, read_cropped as y4m_read_cropped};

#[test]
#[ignore = "diagnostic"]
fn per_sb_motion() {
    const W: u32 = 256;
    const H: u32 = 256;
    let path = probe_testsrc_720p().expect("testsrc");
    let (frames, _fr) = y4m_read_cropped(&path, W, H, 8).expect("read");

    let w_us = W as usize;
    let sb_cols = w_us / 64;
    let sb_rows = (H as usize) / 64;
    for i in 1..frames.len() {
        let prev = &frames[i - 1].0;
        let cur = &frames[i].0;
        eprintln!("frame {i} vs {}: per-SB SSE", i - 1);
        for sr in 0..sb_rows {
            for sc in 0..sb_cols {
                let mut sse: u64 = 0;
                for r in 0..64 {
                    for c in 0..64 {
                        let off = (sr * 64 + r) * w_us + (sc * 64 + c);
                        let d = i32::from(prev[off]) - i32::from(cur[off]);
                        sse += (d * d) as u64;
                    }
                }
                eprint!(" {:>9}", sse);
            }
            eprintln!();
        }
    }
}
