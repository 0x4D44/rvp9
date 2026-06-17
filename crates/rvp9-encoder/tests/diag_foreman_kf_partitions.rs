#![allow(clippy::print_stderr)]

//! A0' — Foreman KF partition distribution + per-leaf intra mode dump.
//!
//! We instrument the partition picker + intra picker by calling them
//! directly on the foreman KF source. This shows:
//!   1. Block-size leaf distribution (does the partition picker descend
//!      on the textured face?)
//!   2. Per-leaf picked intra mode (do leaves on the face pick V/H/TM
//!      rather than DC?)
//!   3. Per-leaf SSE between source and reconstruction (where does the
//!      pixel error live?)

use rvp9_core::block::BlockSize;
use rvp9_core::mode_decision::partition::{
    PartitionDecision, PartitionSearchParams, choose_partitioning,
};
use rvp9_core::mv_search::PlaneRef;
use rvp9_harness::y4m::{probe_foreman_cif, read_cropped as y4m_read_cropped};

#[test]
#[ignore = "A0' KF partition dump"]
fn diag_foreman_kf_partitions() {
    const W: usize = 256;
    const H: usize = 256;

    let Some(src) = probe_foreman_cif() else {
        eprintln!("    [skip] foreman_cif.y4m not found");
        return;
    };
    let (frames, _fr) = y4m_read_cropped(&src, W as u32, H as u32, 1).expect("read");
    let (y, _u, _v) = &frames[0];

    // Try a range of q-indices to see how partition decisions change.
    for &q in &[34u8, 90, 124, 150, 200] {
        eprintln!("\n=== q={q} partition pick ===");
        let sb_cols = W / 64;
        let sb_rows = H / 64;
        let src_plane = PlaneRef {
            data: y,
            stride: W,
            width: W,
            height: H,
        };
        let ref_plane = PlaneRef {
            data: y,
            stride: W,
            width: W,
            height: H,
        }; // ignored for KF
        let mut bsize_counts: [u32; 5] = [0; 5]; // 8/16/32/64/other
        for sb_row in 0..sb_rows {
            for sb_col in 0..sb_cols {
                let params = PartitionSearchParams {
                    src_plane: &src_plane,
                    src_offset_xy: (sb_col * 64, sb_row * 64),
                    ref_plane: &ref_plane,
                    ref_offset_xy: (sb_col * 64, sb_row * 64),
                    is_keyframe: true,
                    q_index: q,
                    // KF path ignores these inter-only fields.
                    frame_width: W,
                    frame_height: H,
                    avg_inter_qindex: q,
                    noise_threshold_scale: 1,
                };
                let decision = choose_partitioning(&params);
                count_decision(&decision, &mut bsize_counts);
            }
        }
        eprintln!(
            "  Block8x8={} Block16x16={} Block32x32={} Block64x64={} other={}",
            bsize_counts[0], bsize_counts[1], bsize_counts[2], bsize_counts[3], bsize_counts[4],
        );
    }
}

fn count_decision(decision: &PartitionDecision, counts: &mut [u32; 5]) {
    let mut visited = [[false; 8]; 8];
    for r in 0..8 {
        for c in 0..8 {
            if visited[r][c] {
                continue;
            }
            let bs = decision.cells[r][c];
            let span = match bs {
                BlockSize::Block8x8 => 1usize,
                BlockSize::Block16x16 => 2,
                BlockSize::Block32x32 => 4,
                BlockSize::Block64x64 => 8,
                _ => 0,
            };
            for dr in 0..span {
                for dc in 0..span {
                    visited[r + dr][c + dc] = true;
                }
            }
            let idx = match bs {
                BlockSize::Block8x8 => 0,
                BlockSize::Block16x16 => 1,
                BlockSize::Block32x32 => 2,
                BlockSize::Block64x64 => 3,
                _ => 4,
            };
            counts[idx] += 1;
        }
    }
}
