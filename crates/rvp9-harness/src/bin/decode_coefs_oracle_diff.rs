//! Stage 3 stretch M.8 — decode_coefs oracle diff.
//!
//! Replays rvp9's production `decode_coefs` on the captured snapshot
//! from `vp90-2-02-size-32x32.webm` frame 1 sub-1, then diffs the
//! output against the libvpx C oracle's output for the same snapshot.
//!
//! The C oracle lives at `tools/libvpx_decode_coefs_oracle/oracle.c`;
//! its captured output is hard-coded below as `LIBVPX_DQ_COEFS`. If
//! the oracle is rebuilt with different inputs, re-run it and update
//! these constants — they are the ground-truth bytes from an
//! independent libvpx v1.14.1 build.
//!
//! Companion: `iht4x4_16_add(DctAdst, libvpx_output, dest=128)` is
//! also run on the rvp9 side, and the residual is compared against
//! the libvpx-side residual (also hard-coded from the C oracle run).
//!
//! Usage:
//!   cargo run --release --bin decode-coefs-oracle-diff
//!
//! Exits non-zero on any divergence.

use rvp9_core::entropy::{BoolReader, CoefBandCtxProbs, decode_coefs};
use rvp9_core::transforms::inverse::{iadst4, idct4};
use rvp9_core::transforms::scans::col_scan_4x4;
use rvp9_core::transforms::{TxSize, TxType, iht4x4_16_add};

// =====================================================================
// Snapshot — must stay in sync with tools/libvpx_decode_coefs_oracle/
// snapshot.h. Source: rvp9 trace at sub-1 of the M-series failing block.
// =====================================================================

/// 331 bytes of the compressed-partition slice the rvp9 production
/// reader was reading from at sub-1 of `vp90-2-02-size-32x32.webm`
/// frame 1.
const SNAP_BUFFER: &[u8] = &[
    0x7e, 0xe9, 0x5c, 0x93, 0xea, 0x58, 0xfd, 0xcf, 0x63, 0x9d, 0xa2, 0x2c, 0xe8, 0x0b, 0x2c, 0xc7,
    0x66, 0x6d, 0x52, 0x7d, 0x06, 0x9b, 0x28, 0xca, 0x1e, 0x74, 0x97, 0xb5, 0x1b, 0x0b, 0xfb, 0xcc,
    0x44, 0x87, 0x6b, 0x3d, 0x60, 0x84, 0x15, 0x01, 0xa9, 0x2e, 0x90, 0x7f, 0x04, 0x8a, 0x76, 0xb1,
    0x98, 0x9a, 0xdd, 0x6d, 0x4b, 0x73, 0x18, 0x84, 0xd5, 0x1e, 0xbb, 0x3c, 0xbb, 0x29, 0x45, 0xbd,
    0xf3, 0x3a, 0x5a, 0x20, 0x8e, 0x92, 0x83, 0x4e, 0x4e, 0x8e, 0x53, 0x07, 0x11, 0xb5, 0xee, 0x98,
    0x69, 0x8b, 0xe8, 0x36, 0x6a, 0x4c, 0x69, 0xff, 0xb4, 0xb9, 0x2a, 0x7d, 0x03, 0xa8, 0x6e, 0x4d,
    0xb3, 0x2c, 0x9f, 0x7e, 0xbe, 0x63, 0x56, 0x18, 0x88, 0x61, 0x8e, 0x5f, 0xe7, 0x38, 0xd2, 0x2f,
    0xf5, 0x7e, 0xcc, 0xe4, 0x34, 0x52, 0x15, 0x19, 0x23, 0x7d, 0x1d, 0xb2, 0x97, 0x7a, 0x9f, 0xe9,
    0xfc, 0x5e, 0x44, 0x8c, 0x05, 0x07, 0x43, 0xa1, 0x72, 0x73, 0x06, 0x11, 0x79, 0x3d, 0xfc, 0x6a,
    0xab, 0x1b, 0x4a, 0xd7, 0x82, 0x40, 0x5e, 0xb4, 0x34, 0x55, 0xfa, 0xe1, 0xb7, 0x4a, 0x78, 0x10,
    0x43, 0x7e, 0x0d, 0xa8, 0x56, 0x41, 0xd8, 0x3b, 0x4d, 0xbe, 0x1d, 0x24, 0xb9, 0x69, 0x1f, 0x7d,
    0x31, 0x36, 0x90, 0x4a, 0x2a, 0x67, 0xa9, 0x17, 0xb9, 0x43, 0xc0, 0x80, 0x30, 0xb8, 0xe6, 0x23,
    0x0e, 0xc1, 0xf9, 0xcf, 0x47, 0x82, 0x6d, 0x3d, 0x4a, 0xb4, 0x2e, 0x4a, 0x14, 0xc9, 0xfa, 0xed,
    0x9f, 0x4b, 0x4e, 0x0b, 0x52, 0x08, 0xdb, 0xb4, 0x58, 0x96, 0xbf, 0xfe, 0xbd, 0x67, 0xc6, 0x65,
    0xf0, 0x3a, 0x76, 0xfd, 0x79, 0x55, 0x2d, 0x0d, 0x4c, 0x6e, 0xee, 0x9a, 0x8f, 0x98, 0xb9, 0xf0,
    0x0e, 0x50, 0x68, 0x6d, 0x36, 0x40, 0x76, 0x7b, 0x95, 0x61, 0x6c, 0x3a, 0x3c, 0x92, 0x23, 0x61,
    0x84, 0x3c, 0x82, 0x98, 0x7c, 0xe6, 0x4b, 0xb3, 0xcd, 0x6c, 0xc6, 0xa5, 0x09, 0x3f, 0xf5, 0x94,
    0x55, 0xfe, 0xad, 0x49, 0xd7, 0xe5, 0x2c, 0x7a, 0x4c, 0x6a, 0x79, 0xe7, 0x31, 0x43, 0x99, 0xa4,
    0x57, 0xb4, 0x06, 0x82, 0x06, 0x01, 0x33, 0x5e, 0x95, 0x1b, 0xa0, 0x0b, 0x38, 0x3b, 0x1d, 0xde,
    0xe1, 0x36, 0x91, 0xc1, 0xe3, 0x80, 0xb7, 0x51, 0xb6, 0xae, 0x7c, 0x4e, 0x7e, 0xf7, 0x57, 0xda,
    0x2b, 0x90, 0x93, 0x58, 0x50, 0x0c, 0xfc, 0xc5, 0x63, 0x80, 0x00,
];

/// Coef-probs slice as passed to `decode_coefs` at sub-1: matches
/// libvpx `default_coef_probs_4x4[PLANE_TYPE_Y]` exactly (M.7 audited).
const SNAP_COEF_PROBS: CoefBandCtxProbs = [
    // band 0
    [
        [195, 29, 183],
        [84, 49, 136],
        [8, 42, 71],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    // band 1
    [
        [31, 107, 169],
        [35, 99, 159],
        [17, 82, 140],
        [8, 66, 114],
        [2, 44, 76],
        [1, 19, 32],
    ],
    // band 2
    [
        [40, 132, 201],
        [29, 114, 187],
        [13, 91, 157],
        [7, 75, 127],
        [3, 58, 95],
        [1, 28, 47],
    ],
    // band 3
    [
        [69, 142, 221],
        [42, 122, 201],
        [15, 91, 159],
        [6, 67, 121],
        [1, 42, 77],
        [1, 17, 31],
    ],
    // band 4
    [
        [102, 148, 228],
        [67, 117, 204],
        [17, 82, 154],
        [6, 59, 114],
        [2, 39, 75],
        [1, 15, 29],
    ],
    // band 5
    [
        [156, 57, 233],
        [119, 57, 212],
        [58, 48, 163],
        [29, 40, 124],
        [12, 30, 81],
        [3, 12, 31],
    ],
];

const SNAP_PRE_RANGE: u32 = 200;
const SNAP_PRE_VALUE: u64 = 0x7768_8b3a_02cb_0000;
const SNAP_PRE_COUNT: i32 = 42;
const SNAP_PRE_POS: usize = 15;
const SNAP_INITIAL_PT: usize = 1;
const SNAP_SEG_EOB: usize = 16;
const SNAP_DQ: (i32, i32) = (177, 231);

/// Output from `tools/libvpx_decode_coefs_oracle/oracle.exe` on the
/// same snapshot. Captured 2026-05-18 — see commit message.
///
/// libvpx and rvp9 production AGREE on this — both decode the same
/// `dq_coefs`. The bug is therefore downstream of decode_coefs.
const LIBVPX_DQ_COEFS: [i32; 16] = [
    177, 0, 0, 0, -693, -231, 0, 0, 462, 231, 0, 0, 231, 231, 231, -231,
];
const LIBVPX_EOB: usize = 16;

/// Output from `tools/libvpx_decode_coefs_oracle/oracle.exe` for the
/// follow-up `vp9_iht4x4_16_add_c(DctAdst, dq_coefs, dest=128, stride=4)`
/// probe — what libvpx's iht4x4 emits given the dq_coefs above.
///
/// `dest[i] - 128` per cell, as printed by the oracle. This is the
/// FIRST surface where libvpx and rvp9 diverge — confirming the bug
/// is in our iht4x4 / iadst4 / idct4 stack, NOT in decode_coefs.
const LIBVPX_IHT4X4_RESIDUAL: [[i32; 4]; 4] = [
    [2, 2, -14, -3],
    [-42, -58, -13, -28],
    [17, 26, -8, 21],
    [33, 49, 60, 38],
];

fn main() {
    println!("=== Stage 3 stretch M.8 — decode_coefs oracle diff ===");
    println!(
        "Snapshot: vp90-2-02-size-32x32.webm frame 1 sub-1 (Tx4x4, intra Y, \
         DctAdst → col_scan_4x4, initial_pt={SNAP_INITIAL_PT}, dq={SNAP_DQ:?})"
    );
    println!(
        "Pre-state: range={SNAP_PRE_RANGE} value=0x{SNAP_PRE_VALUE:016x} \
         count={SNAP_PRE_COUNT} pos={SNAP_PRE_POS}"
    );

    // --- 1. Replay rvp9 production decode_coefs.
    // Init a BoolReader at byte 0 of the buffer, then directly restore
    // the captured pre-snapshot register state. This is the same
    // technique the M.6 differential hook uses internally.
    let mut reader = BoolReader::new(SNAP_BUFFER).expect("BoolReader::new");
    let pre_snap = capture_pre_state();
    reader.restore(&pre_snap);

    let scan = col_scan_4x4();
    let mut rvp9_dq = vec![0i32; SNAP_SEG_EOB];
    let rvp9_eob = decode_coefs(
        &mut reader,
        &SNAP_COEF_PROBS,
        None,
        &scan,
        SNAP_INITIAL_PT,
        TxSize::Tx4x4,
        SNAP_SEG_EOB,
        SNAP_DQ,
        &mut rvp9_dq[..],
    );

    println!("\nrvp9 production decode_coefs:");
    println!("  eob      = {rvp9_eob}");
    println!("  dq_coefs = {rvp9_dq:?}");
    println!("\nlibvpx oracle decode_coefs:");
    println!("  eob      = {LIBVPX_EOB}");
    println!("  dq_coefs = {LIBVPX_DQ_COEFS:?}");

    let coefs_match = rvp9_eob == LIBVPX_EOB && rvp9_dq[..] == LIBVPX_DQ_COEFS[..];
    println!(
        "\nDecode_coefs diff: {}",
        if coefs_match { "MATCH" } else { "DIVERGE" }
    );

    // --- 2. Run rvp9 iht4x4_16_add(DctAdst, dq, dest=128) and compare
    //        against libvpx's iht4x4 output for the same input.
    let mut dq_input = [0i32; 16];
    dq_input.copy_from_slice(&rvp9_dq);

    // Run the row-pass standalone to expose the intermediate matrix.
    let mut intermediate = [0i32; 16];
    for r in 0..4 {
        let row_in: &[i32; 4] = (&dq_input[r * 4..r * 4 + 4]).try_into().unwrap();
        let row_out: &mut [i32; 4] = (&mut intermediate[r * 4..r * 4 + 4]).try_into().unwrap();
        idct4(row_in, row_out);
    }
    println!("\nrvp9 intermediate after row-pass (idct4 × 4 rows):");
    for row in 0..4 {
        let r = &intermediate[row * 4..row * 4 + 4];
        println!("  row {row}: {r:?}");
    }

    println!("\nrvp9 col-pass output (iadst4 of each column, raw):");
    for i in 0..4 {
        let col_in = [
            intermediate[i],
            intermediate[4 + i],
            intermediate[8 + i],
            intermediate[12 + i],
        ];
        let mut col_out = [0i32; 4];
        iadst4(&col_in, &mut col_out);
        println!("  col {i}: in={col_in:?}  out={col_out:?}");
    }

    let mut dest = [128u8; 16];
    iht4x4_16_add(&dq_input, &mut dest, 4, TxType::DctAdst);

    let mut rvp9_residual = [[0i32; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            rvp9_residual[row][col] = (dest[row * 4 + col] as i32) - 128;
        }
    }

    println!("\nrvp9 iht4x4_16_add(DctAdst, dq, dest=128) residual:");
    for row in &rvp9_residual {
        println!("  {row:?}");
    }
    println!("\nlibvpx oracle iht4x4 residual:");
    for row in &LIBVPX_IHT4X4_RESIDUAL {
        println!("  {row:?}");
    }

    let iht_match = rvp9_residual == LIBVPX_IHT4X4_RESIDUAL;
    println!(
        "\niht4x4 diff: {}",
        if iht_match { "MATCH" } else { "DIVERGE" }
    );

    if !iht_match {
        println!("\nFirst diff (by [row,col]):");
        'outer: for row in 0..4 {
            for col in 0..4 {
                if rvp9_residual[row][col] != LIBVPX_IHT4X4_RESIDUAL[row][col] {
                    println!(
                        "  [{row},{col}] rvp9={} libvpx={}",
                        rvp9_residual[row][col], LIBVPX_IHT4X4_RESIDUAL[row][col]
                    );
                    break 'outer;
                }
            }
        }
    }

    if !coefs_match || !iht_match {
        std::process::exit(1);
    }
}

/// Build the captured pre-snapshot register state. We construct a
/// fresh `BoolReader` at byte 0, then poke into its private fields
/// via `restore` — only `BoolReaderSnapshot` is publicly constructible
/// here. To get a `BoolReaderSnapshot` matching our captured state we
/// snapshot a temporary reader after positioning it explicitly.
///
/// Note: we can't directly construct a `BoolReaderSnapshot` because
/// its fields are private. Workaround: walk a synthetic reader through
/// `fill` until pos=15, then patch range/value/count to match — but
/// even simpler is to expose a back-door builder. We use a `cfg(test)`
/// builder if present; otherwise we manually drive the reader.
///
/// The cleanest fix is to expose `BoolReaderSnapshot::from_parts`. For
/// now, the snapshot's pre-state is identical to what
/// `BoolReader::new(SNAP_BUFFER)` reaches after the initial `fill` if
/// we manually advance pos to 15 with no consuming reads — which isn't
/// trivial. So we use the same trick as the M.6 hook: call
/// `BoolReader::new` (advances past the marker bit), then directly
/// `restore` to a snapshot built from the captured tuple via a public
/// builder we expose below in this binary's own module.
fn capture_pre_state() -> rvp9_core::entropy::BoolReaderSnapshot {
    // Mirrors the M.6 capture: build a snapshot with the exact register
    // state the production trace recorded.
    rvp9_core::entropy::BoolReaderSnapshot::from_parts(
        SNAP_PRE_VALUE,
        SNAP_PRE_RANGE,
        SNAP_PRE_COUNT,
        SNAP_PRE_POS,
    )
}
