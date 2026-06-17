#![allow(clippy::print_stderr)]

//! Auxiliary: measure inter-frame source SAD/SSE on testsrc 256x256 to
//! check whether the high-frame-index PSNR-100 phenomenon is "input is
//! literally static after frame N" vs "encoder produced bit-exact recon".

use rvp9_harness::y4m::{probe_testsrc_720p, read_cropped as y4m_read_cropped};

#[test]
#[ignore = "Diagnostic; run with --ignored"]
fn source_static_check() {
    const W: u32 = 256;
    const H: u32 = 256;
    const N: u32 = 64;
    let path = probe_testsrc_720p().expect("testsrc");
    let (frames, _fr) = y4m_read_cropped(&path, W, H, N as usize).expect("read");
    eprintln!("source inter-frame Y deltas (vs previous frame):");
    for i in 1..frames.len() {
        let prev = &frames[i - 1].0;
        let cur = &frames[i].0;
        let mut sse: u64 = 0;
        let mut sad: u64 = 0;
        for (a, b) in prev.iter().zip(cur.iter()) {
            let d = i32::from(*a) - i32::from(*b);
            sse += (d * d) as u64;
            sad += d.unsigned_abs() as u64;
        }
        eprintln!(
            "  frame {i:>2} vs {:>2}: SSE={sse:>12}  SAD={sad:>12}  pixels_changed (any) approx",
            i - 1
        );
    }
}
