//! Bitstream differential diff tool.
//!
//! Takes two `.ivf` files and reports:
//!
//! 1. First byte of divergence in the raw IVF byte stream.
//! 2. First frame (1-based index) where the frames differ in size or content.
//! 3. (Stage 4+) First decoded pixel of divergence in YUV output, when both
//!    `--decoder` flag and the source resolution are supplied.
//!
//! Invaluable in Stage 4+ when chasing byte-parity divergence between rvp9
//! and libvpx. Stage 0 just validates the byte-level path against two copies
//! of the same input.
//!
//! Usage:
//!   cargo run --release --bin diff-bitstream -- \
//!       --a path/to/a.ivf \
//!       --b path/to/b.ivf

use std::path::PathBuf;

struct Args {
    a: PathBuf,
    b: PathBuf,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1).cloned())
    };
    Args {
        a: get("--a").map(PathBuf::from).expect("--a required"),
        b: get("--b").map(PathBuf::from).expect("--b required"),
    }
}

/// Walk IVF frames in parallel and report the first divergence.
/// Returns (frame_index_1based, byte_offset_within_frame) on divergence.
fn diff_frames(a: &[u8], b: &[u8]) -> Option<(u32, usize)> {
    let mut ai = 32usize;
    let mut bi = 32usize;
    let mut frame_idx = 0u32;
    while ai + 12 <= a.len() && bi + 12 <= b.len() {
        frame_idx += 1;
        let a_size = u32::from_le_bytes([a[ai], a[ai + 1], a[ai + 2], a[ai + 3]]) as usize;
        let b_size = u32::from_le_bytes([b[bi], b[bi + 1], b[bi + 2], b[bi + 3]]) as usize;
        if a_size != b_size {
            return Some((frame_idx, 0));
        }
        let a_payload = &a[ai + 12..ai + 12 + a_size];
        let b_payload = &b[bi + 12..bi + 12 + b_size];
        for (k, (&x, &y)) in a_payload.iter().zip(b_payload.iter()).enumerate() {
            if x != y {
                return Some((frame_idx, k));
            }
        }
        ai += 12 + a_size;
        bi += 12 + b_size;
    }
    if ai < a.len() || bi < b.len() {
        return Some((frame_idx + 1, 0));
    }
    None
}

fn main() {
    let args = parse_args();
    let a = std::fs::read(&args.a).expect("read --a");
    let b = std::fs::read(&args.b).expect("read --b");

    println!("a: {}  ({} bytes)", args.a.display(), a.len());
    println!("b: {}  ({} bytes)", args.b.display(), b.len());

    // 1. Raw byte diff.
    let first_byte_diff = a.iter().zip(b.iter()).position(|(x, y)| x != y);
    match first_byte_diff {
        None if a.len() == b.len() => {
            println!("\nIdentical: every byte matches.");
            return;
        }
        None => println!("\nOne is a prefix of the other (lengths differ)."),
        Some(off) => println!("\nFirst byte divergence at offset {off}"),
    }

    // 2. Frame-level diff.
    if a.len() < 32 || b.len() < 32 {
        eprintln!("file too short to contain IVF header");
        std::process::exit(2);
    }
    if &a[0..4] != b"DKIF" || &b[0..4] != b"DKIF" {
        eprintln!("not an IVF file (bad signature)");
        std::process::exit(2);
    }
    match diff_frames(&a, &b) {
        Some((idx, off)) => println!("First frame divergence: frame {idx}, byte +{off} into payload"),
        None => println!("Frames match (only IVF headers differ)"),
    }

    // 3. TODO(Stage 4+): decode both and report first pixel of divergence in
    //    YUV output. Requires both --decoder and source resolution.
}
