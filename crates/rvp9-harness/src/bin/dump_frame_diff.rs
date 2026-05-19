//! Stage 3 stretch M.9 — per-pixel diff between rvp9's decoded frame
//! and libvpx's `vpxdec --i420` output.
//!
//! Given a vector + frame index, decodes via rvp9 and reads a
//! pre-generated libvpx I420 dump for the same frame; prints
//! (plane, x, y) of the first mismatch and a small surrounding patch.
//!
//! Useful for localizing WHICH block of a failing keyframe diverges
//! first — i.e. the input to the next M.9-style oracle snapshot.
//!
//! Usage:
//! 1. Build a libvpx I420 dump:
//!    `vpxdec.exe --i420 --limit=1 -o expected.yuv vector.webm`
//! 2. Run this binary:
//!    `cargo run --release --bin dump-frame-diff -- --vector
//!    <path-to-vector> --expected expected.yuv --frame 1`

use rvp9_decoder::Decoder;
use rvp9_harness::container::read_packets;
use std::path::PathBuf;

struct Args {
    vector: PathBuf,
    expected: PathBuf,
    frame: usize,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let value = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1).cloned())
    };
    Args {
        vector: PathBuf::from(value("--vector").expect("--vector required")),
        expected: PathBuf::from(value("--expected").expect("--expected required")),
        frame: value("--frame")
            .map(|s| s.parse().expect("--frame usize"))
            .unwrap_or(1),
    }
}

fn main() {
    let args = parse_args();

    let (data, packets) = read_packets(&args.vector).expect("container parse");
    let expected = std::fs::read(&args.expected).expect("read expected.yuv");

    let mut decoder = Decoder::new();
    let mut shown_idx: usize = 0;
    let mut got_buf = None;

    'outer: for range in &packets {
        let parsed = decoder
            .parse_packet(range.slice(&data))
            .expect("parse_packet");
        for frame in parsed {
            let shown = frame.header.show_frame
                || matches!(
                    frame.header.kind,
                    rvp9_core::bitstream::FrameKind::ShowExistingFrame { .. }
                );
            // D.19 — always decode every frame (including non-shown
            // altrefs). Skipping altrefs leaves the ref-pool slot
            // uninitialised, so any subsequent inter frame that uses
            // ALTREF reads stale data and the diff is meaningless.
            let buf_opt = decoder.decode_frame(&frame).ok();
            if !shown {
                continue;
            }
            shown_idx += 1;
            if shown_idx == args.frame {
                got_buf = buf_opt;
                break 'outer;
            }
        }
    }

    let buf = got_buf.expect("frame not found");
    let crop_w = buf.width() as usize;
    let crop_h = buf.height() as usize;
    let uv_w = crop_w.div_ceil(2);
    let uv_h = crop_h.div_ceil(2);

    let y_size = crop_w * crop_h;
    let uv_size = uv_w * uv_h;
    assert!(
        expected.len() >= y_size + 2 * uv_size,
        "expected.yuv too short: have {} need {}",
        expected.len(),
        y_size + 2 * uv_size
    );

    // Extract planes from rvp9.
    let mut y_got = Vec::with_capacity(y_size);
    let y_plane = buf.y_plane();
    for r in 0..crop_h {
        let start = r * y_plane.stride();
        y_got.extend_from_slice(&y_plane.data()[start..start + crop_w]);
    }
    let mut u_got = Vec::with_capacity(uv_size);
    let u_plane = buf.u_plane();
    for r in 0..uv_h {
        let start = r * u_plane.stride();
        u_got.extend_from_slice(&u_plane.data()[start..start + uv_w]);
    }
    let mut v_got = Vec::with_capacity(uv_size);
    let v_plane = buf.v_plane();
    for r in 0..uv_h {
        let start = r * v_plane.stride();
        v_got.extend_from_slice(&v_plane.data()[start..start + uv_w]);
    }

    let y_exp = &expected[..y_size];
    let u_exp = &expected[y_size..y_size + uv_size];
    let v_exp = &expected[y_size + uv_size..y_size + 2 * uv_size];

    fn diff_plane(name: &str, got: &[u8], exp: &[u8], w: usize, h: usize) -> Option<(usize, usize)> {
        let mut count = 0usize;
        let mut first: Option<(usize, usize)> = None;
        let mut sum_abs = 0u64;
        let mut max_abs = 0i32;
        for r in 0..h {
            for c in 0..w {
                let i = r * w + c;
                if got[i] != exp[i] {
                    if first.is_none() {
                        first = Some((c, r));
                    }
                    count += 1;
                    let d = got[i] as i32 - exp[i] as i32;
                    sum_abs += d.unsigned_abs() as u64;
                    max_abs = max_abs.max(d.abs());
                }
            }
        }
        if let Some((c, r)) = first {
            println!(
                "first {} diff @ ({}, {}): got={:3}  expected={:3}  | total {} mismatched pixels of {} (sum_abs={}, max_abs={})",
                name, c, r, got[r*w+c], exp[r*w+c], count, w*h, sum_abs, max_abs
            );
        } else {
            println!("{name}: MATCH ({} pixels)", w*h);
        }
        first
    }

    // Show a per-4x4-block summary of Y diffs.
    let block_w = w_blocks(crop_w);
    let block_h = w_blocks(crop_h);
    let mut block_diff = vec![0u32; block_w * block_h];
    for r in 0..crop_h {
        for c in 0..crop_w {
            let i = r * crop_w + c;
            if y_got[i] != y_exp[i] {
                let bc = c / 4;
                let br = r / 4;
                block_diff[br * block_w + bc] += 1;
            }
        }
    }
    println!("---- Y per-4x4-block mismatch count grid ({}x{} blocks) ----", block_w, block_h);
    print!("       ");
    for bc in 0..block_w {
        print!("{:3}c{:1} ", bc * 4, bc);
    }
    println!();
    for br in 0..block_h {
        print!("r{:1}({:3}):", br, br * 4);
        for bc in 0..block_w {
            let n = block_diff[br * block_w + bc];
            if n == 0 {
                print!("  .  ");
            } else {
                print!(" {:3} ", n);
            }
        }
        println!();
    }

    println!("vector: {}", args.vector.display());
    println!("frame: {} ({}x{})", args.frame, crop_w, crop_h);
    println!("---- per-plane diff ----");
    let y_first = diff_plane("Y", &y_got, y_exp, crop_w, crop_h);
    let u_first = diff_plane("U", &u_got, u_exp, uv_w, uv_h);
    let v_first = diff_plane("V", &v_got, v_exp, uv_w, uv_h);

    // Find first 4x4 block with ANY diff and dump that block in full.
    let mut found_block: Option<(usize, usize)> = None;
    for br in 0..block_h {
        for bc in 0..block_w {
            if block_diff[br * block_w + bc] > 0 {
                found_block = Some((bc, br));
                break;
            }
        }
        if found_block.is_some() {
            break;
        }
    }
    // Also dump the FULL FIRST 8x8 region of expected and got (Y plane).
    println!("---- Y first 12 cols × 8 rows: got | expected ----");
    for r in 0..8 {
        print!("  r{r:2} got: ");
        for c in 0..12 {
            print!("{:4} ", y_got[r * crop_w + c]);
        }
        print!("  | exp: ");
        for c in 0..12 {
            print!("{:4} ", y_exp[r * crop_w + c]);
        }
        println!();
    }

    // M.17 probe: dump the (px 276..295, row 0..7) window — straddles the
    // last-passing 4×4 (px 276..279), our failing Block8x8 (px 280..287),
    // and the start of the next Block8x8 (px 288..295). Lets us read off
    // which column is the smoking-gun pre-LF divergence.
    if let Ok(s) = std::env::var("RVP9_DEBUG_DUMP_WINDOW") {
        // Format: "x_lo,x_hi,y_lo,y_hi" — half-open ranges.
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() == 4 {
            let c_lo: usize = parts[0].parse().unwrap_or(0);
            let c_hi: usize = parts[1].parse::<usize>().unwrap_or(0).min(crop_w);
            let r_lo: usize = parts[2].parse().unwrap_or(0);
            let r_hi: usize = parts[3].parse::<usize>().unwrap_or(0).min(crop_h);
            println!("---- Y cols {c_lo}..{c_hi} rows {r_lo}..{r_hi}: got | exp | diff ----");
            for r in r_lo..r_hi {
                print!("  r{r:3} got: ");
                for c in c_lo..c_hi {
                    print!("{:4} ", y_got[r * crop_w + c]);
                }
                print!("  | exp: ");
                for c in c_lo..c_hi {
                    print!("{:4} ", y_exp[r * crop_w + c]);
                }
                print!("  | diff: ");
                for c in c_lo..c_hi {
                    let g = y_got[r * crop_w + c] as i32;
                    let e = y_exp[r * crop_w + c] as i32;
                    print!("{:+4} ", g - e);
                }
                println!();
            }
        }
    }
    // M.13 probe — dump rows 8..24 cols 22..32 to inspect upstream cascade.
    if std::env::var("RVP9_DUMP_PATCH").is_ok() {
        let r_lo: usize = 8;
        let r_hi: usize = 24.min(crop_h);
        let c_lo: usize = 22.min(crop_w);
        let c_hi: usize = 32.min(crop_w);
        println!("---- Y rows {r_lo}..{r_hi} cols {c_lo}..{c_hi}: got | exp | diff ----");
        for r in r_lo..r_hi {
            print!("  r{r:2} got: ");
            for c in c_lo..c_hi {
                print!("{:4} ", y_got[r * crop_w + c]);
            }
            print!("  | exp: ");
            for c in c_lo..c_hi {
                print!("{:4} ", y_exp[r * crop_w + c]);
            }
            print!("  | diff: ");
            for c in c_lo..c_hi {
                let g = y_got[r * crop_w + c] as i32;
                let e = y_exp[r * crop_w + c] as i32;
                print!("{:+4} ", g - e);
            }
            println!();
        }
    }
    if let Some((bc, br)) = found_block {
        let bx0 = bc * 4;
        let by0 = br * 4;
        println!(
            "---- first divergent Y 4x4 block @ (col {}, row {}) — {} diffs ----",
            bx0, by0, block_diff[br * block_w + bc]
        );
        println!("  got:");
        for r in 0..4 {
            print!("    ");
            for c in 0..4 {
                print!("{:4} ", y_got[(by0 + r) * crop_w + bx0 + c]);
            }
            println!();
        }
        println!("  expected:");
        for r in 0..4 {
            print!("    ");
            for c in 0..4 {
                print!("{:4} ", y_exp[(by0 + r) * crop_w + bx0 + c]);
            }
            println!();
        }
        println!("  diff (got - expected):");
        for r in 0..4 {
            print!("    ");
            for c in 0..4 {
                let g = y_got[(by0 + r) * crop_w + bx0 + c] as i32;
                let e = y_exp[(by0 + r) * crop_w + bx0 + c] as i32;
                print!("{:+4} ", g - e);
            }
            println!();
        }
    }

    if let Some((c, r)) = y_first {
        dump_patch("Y", &y_got, y_exp, crop_w, crop_h, c, r);
    }
    if let Some((c, r)) = u_first {
        dump_patch("U", &u_got, u_exp, uv_w, uv_h, c, r);
    }
    if let Some((c, r)) = v_first {
        dump_patch("V", &v_got, v_exp, uv_w, uv_h, c, r);
    }

    let any_diff = y_first.is_some() || u_first.is_some() || v_first.is_some();
    if any_diff {
        std::process::exit(1);
    }
}

fn w_blocks(w: usize) -> usize {
    w.div_ceil(4)
}

fn dump_patch(name: &str, got: &[u8], exp: &[u8], w: usize, h: usize, cx: usize, cy: usize) {
    let r0 = cy.saturating_sub(2);
    let r1 = (cy + 6).min(h);
    let c0 = cx.saturating_sub(2);
    let c1 = (cx + 10).min(w);
    println!("---- {} got patch @ rows {}..{}, cols {}..{} ----", name, r0, r1, c0, c1);
    for r in r0..r1 {
        print!("  r{:3}: ", r);
        for c in c0..c1 {
            print!("{:3} ", got[r * w + c]);
        }
        println!();
    }
    println!("---- {} expected patch @ same range ----", name);
    for r in r0..r1 {
        print!("  r{:3}: ", r);
        for c in c0..c1 {
            print!("{:3} ", exp[r * w + c]);
        }
        println!();
    }
    println!("---- {} diff (got - expected) ----", name);
    for r in r0..r1 {
        print!("  r{:3}: ", r);
        for c in c0..c1 {
            let g = got[r * w + c] as i32;
            let e = exp[r * w + c] as i32;
            print!("{:+4} ", g - e);
        }
        println!();
    }
}
