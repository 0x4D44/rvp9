#![allow(clippy::print_stderr)]

//! Find where in SB(0,3) the motion actually is between frames 0 and 1.

use rvp9_harness::y4m::{probe_testsrc_720p, read_cropped as y4m_read_cropped};

#[test]
#[ignore = "diagnostic"]
fn sb03_motion_row_dump() {
    const W: u32 = 256;
    const H: u32 = 256;
    let path = probe_testsrc_720p().expect("testsrc");
    let (frames, _fr) = y4m_read_cropped(&path, W, H, 2).expect("read");

    let w_us = W as usize;
    eprintln!("SB(0,3) = px (192..256, 0..64) frame 1 vs frame 0: per-row SSE");
    let prev = &frames[0].0;
    let cur = &frames[1].0;
    for r in 0..64 {
        let mut sse: u64 = 0;
        for c in 192..256 {
            let d = i32::from(prev[r * w_us + c]) - i32::from(cur[r * w_us + c]);
            sse += (d * d) as u64;
        }
        eprintln!("  row {r:>2}: SSE_row={sse}");
    }
}
