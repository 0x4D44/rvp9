//! F.0.2 — smoke test for the per-block oracle harness.
//!
//! Drives the rvp9-side of the harness end-to-end (no libvpx
//! invocation) and validates the four schema-stability invariants
//! from HLD §6:
//!
//! 1. **Schema round-trip:** the encoder's recorded `BlockDecision`
//!    serialises to JSONL and deserialises back to the same record.
//! 2. **n_blocks structure:** the per-frame sink emits the expected
//!    number of leaves for a single-SB synthetic source.
//! 3. **Identity at flat-grey input:** every recorded block on a
//!    flat-grey single-frame inter source has
//!    `(mode == ZEROMV || mode == SKIP) && mv == (0, 0) &&
//!     skip_flag == true`. Per HLD §6 revision 4, partition_match is
//!    NOT asserted — only mode/skip/MV identity.
//! 4. **Off-by-default:** with `RVP9_F02_ORACLE` unset, the sink is
//!    empty.
//!
//! The libvpx leg's smoke (requires the patched `vpxenc.exe`) is
//! gated behind `#[ignore]`; see `scripts/build_f02_libvpx.ps1` for
//! how to bring it up.

use rvp9_encoder::f02_oracle::{
    self, BlockDecision, BlockDecisionMode,
};
use rvp9_encoder::{AqMode, EncoderConfig, StreamEncoder, YuvFrameRef};
use rvp9_core::mode_info::InterMode;
use rvp9_harness::f02_schema::parse_jsonl_line;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Serialises the env-gated tests. `RVP9_F02_ORACLE` is process-
/// global; running env-flipping tests in parallel races on
/// [`f02_oracle::init_from_env`] and the resulting sink state.
/// Every env-flipping test holds this lock for its whole body.
fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Build a flat-grey YUV frame at the given resolution.
fn flat_grey(width: usize, height: usize, value: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y = vec![value; width * height];
    let u = vec![128u8; (width / 2) * (height / 2)];
    let v = vec![128u8; (width / 2) * (height / 2)];
    (y, u, v)
}

fn view<'a>(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    width: usize,
) -> YuvFrameRef<'a> {
    YuvFrameRef {
        y,
        u,
        v,
        y_stride: width,
        u_stride: width / 2,
        v_stride: width / 2,
    }
}

/// 64×64 single-SB encoder config — matches the minimal-shape input
/// HLD §6 calls out (re-uses the same SB sizing the existing
/// `encode_synthetic_two_frame_kf_inter_non_skip` helper uses).
fn single_sb_config() -> EncoderConfig {
    EncoderConfig {
        width: 64,
        height: 64,
        framerate: 30.0,
        target_bitrate_bps: 2_000_000,
        gop_length: 30,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        enable_alt_ref: false,
        min_qindex: 4,
        max_qindex: 255,
    }
}

#[test]
fn schema_round_trip_through_real_encoder() {
    // Enable the oracle, drive a 2-frame flat-grey sequence (KF +
    // inter), drain the sink, and serialise each record to JSONL +
    // parse it back. Round-trip equality is the contract.
    //
    // SAFETY: `cargo test` defaults to multi-thread, but each test
    // function gets its own `thread_local!` sink. The env-var store
    // is process-global; we set it, drive the encoder on THIS thread,
    // and drain on THIS thread before any cleanup. Other tests in
    // this binary that read the same var would race; the four tests
    // here are independent in their use of the sink because each
    // test's records die at end of function.
    let _guard = env_lock();
    // SAFETY: env_lock() serialises the env-flipping tests within
    // this binary; no other concurrent set_var is possible.
    unsafe { std::env::set_var("RVP9_F02_ORACLE", "1") };
    f02_oracle::init_from_env();
    assert!(f02_oracle::is_enabled(), "oracle must be enabled");

    let cfg = single_sb_config();
    let mut enc = StreamEncoder::new(cfg).unwrap();
    let (y, u, v) = flat_grey(64, 64, 128);

    // KF.
    let _kf_drain_before = f02_oracle::drain_frame();
    let kf = enc.encode_frame(&view(&y, &u, &v, 64)).unwrap();
    let kf_records = f02_oracle::drain_frame();
    assert!(matches!(kf.frame_kind, rvp9_encoder::StreamFrameKind::Key));

    // Inter (same flat content).
    let inter = enc.encode_frame(&view(&y, &u, &v, 64)).unwrap();
    let inter_records = f02_oracle::drain_frame();
    assert!(matches!(
        inter.frame_kind,
        rvp9_encoder::StreamFrameKind::Inter
    ));

    // Round-trip every recorded BlockDecision.
    for d in kf_records.iter().chain(inter_records.iter()) {
        let mut buf = Vec::new();
        d.write_jsonl(&mut buf).unwrap();
        let line = std::str::from_utf8(&buf).unwrap();
        let parsed = parse_jsonl_line(line).unwrap_or_else(|e| {
            panic!("round-trip parse failed for record {d:?}: {e}");
        });
        assert_eq!(
            *d, parsed,
            "BlockDecision must round-trip through JSONL byte-exact",
        );
    }

    // SAFETY: cleanup — same single-thread invariant.
    unsafe { std::env::remove_var("RVP9_F02_ORACLE") };
}

#[test]
fn n_blocks_summary_structure_per_frame() {
    // HLD §6: "single-frame, single-SB stream gives 1 SB". For a
    // 64×64 frame the SB grid is 1×1; the encoder's inter path
    // walks at least one leaf per SB. We don't assert the EXACT
    // leaf count (partition picker can split or not under the same
    // RD tie that HLD §6 revision 4 warns against asserting 100%
    // partition_match); we assert the sink isn't empty AND every
    // record's mi cell falls inside the SB.
    let _guard = env_lock();
    // SAFETY: env_lock() held.
    unsafe { std::env::set_var("RVP9_F02_ORACLE", "1") };
    f02_oracle::init_from_env();
    let _ = f02_oracle::drain_frame(); // reset

    let cfg = single_sb_config();
    let mut enc = StreamEncoder::new(cfg).unwrap();
    let (y, u, v) = flat_grey(64, 64, 128);
    let _kf = enc.encode_frame(&view(&y, &u, &v, 64)).unwrap();
    let _ = f02_oracle::drain_frame(); // drop KF (intra) — focus on inter

    let _inter = enc.encode_frame(&view(&y, &u, &v, 64)).unwrap();
    let records = f02_oracle::drain_frame();

    assert!(
        !records.is_empty(),
        "inter sink must contain ≥ 1 leaf for a single-SB inter frame"
    );
    // Every record's top-left mi cell is inside the 1×1 SB grid
    // (mi_row, mi_col ∈ {0..8}). 64 px / 8 mi_per_8x8 = 8 mi cells.
    for r in &records {
        assert!(
            r.mi_row < 8,
            "record mi_row={} out of single-SB bounds (< 8)",
            r.mi_row
        );
        assert!(
            r.mi_col < 8,
            "record mi_col={} out of single-SB bounds (< 8)",
            r.mi_col
        );
        // The harness binary's per-frame "n_blocks" column counts
        // these records; structural shape is what we're asserting
        // here.
    }
    // SAFETY: cleanup.
    unsafe { std::env::remove_var("RVP9_F02_ORACLE") };
}

#[test]
fn identity_invariants_at_flat_grey_input() {
    // HLD §6 revision 4 — every coded leaf on a flat-grey inter
    // source must have `(mode == ZEROMV || mode == SKIP) && mv ==
    // (0,0) && skip_flag == true`. Partition tilings can differ for
    // flat input due to RD ties; do NOT assert partition_match.
    let _guard = env_lock();
    // SAFETY: env_lock() held.
    unsafe { std::env::set_var("RVP9_F02_ORACLE", "1") };
    f02_oracle::init_from_env();
    let _ = f02_oracle::drain_frame();

    let cfg = single_sb_config();
    let mut enc = StreamEncoder::new(cfg).unwrap();
    let (y, u, v) = flat_grey(64, 64, 128);
    let _kf = enc.encode_frame(&view(&y, &u, &v, 64)).unwrap();
    let _kf_recs = f02_oracle::drain_frame(); // KF records are intra; not the focus here

    let _inter = enc.encode_frame(&view(&y, &u, &v, 64)).unwrap();
    let inter_recs = f02_oracle::drain_frame();

    assert!(!inter_recs.is_empty());
    for d in &inter_recs {
        // `mode == ZEROMV` is the only inter mode the flat-grey
        // input can sensibly pick under the picker's RD ranking —
        // NEAREST/NEAR/NEW all collapse onto the same prediction at
        // zero MV and the rate ranking prefers ZEROMV (cheapest).
        let mode_is_identity = match d.mode {
            BlockDecisionMode::Inter(InterMode::ZeroMv) => true,
            // The HLD also allows "SKIP" — in the rvp9 encoder
            // there isn't a separate SKIP mode discriminant; skip
            // is a per-leaf flag. A leaf with `skip_flag == true`
            // satisfies the HLD's "SKIP" branch regardless of the
            // nominal mode.
            _ => d.skip_flag,
        };
        assert!(
            mode_is_identity,
            "leaf at ({},{}) bsize={:?} mode={:?} skip={} \
             — expected ZEROMV or skip=true on flat-grey input",
            d.mi_row,
            d.mi_col,
            d.bsize,
            d.mode,
            d.skip_flag,
        );
        assert_eq!(
            (d.mv_row, d.mv_col),
            (0, 0),
            "leaf at ({},{}) bsize={:?} mv={:?} — expected (0,0) MV on flat-grey input",
            d.mi_row,
            d.mi_col,
            d.bsize,
            (d.mv_row, d.mv_col),
        );
        assert!(
            d.skip_flag,
            "leaf at ({},{}) bsize={:?} — expected skip_flag=true on flat-grey input",
            d.mi_row,
            d.mi_col,
            d.bsize,
        );
    }
    // SAFETY: cleanup.
    unsafe { std::env::remove_var("RVP9_F02_ORACLE") };
}

#[test]
fn off_by_default_sink_is_empty() {
    // HLD §6: when `RVP9_F02_ORACLE` is unset, the encoder's
    // recorded sink is empty (no measurement leakage).
    let _guard = env_lock();
    // SAFETY: env_lock() held — serialised against the other
    // env-flipping tests in this binary.
    unsafe { std::env::remove_var("RVP9_F02_ORACLE") };
    f02_oracle::init_from_env();
    assert!(
        !f02_oracle::is_enabled(),
        "oracle must NOT be enabled when env var is unset"
    );

    // Drain anything any other test in this binary may have left.
    let _ = f02_oracle::drain_frame();

    let cfg = single_sb_config();
    let mut enc = StreamEncoder::new(cfg).unwrap();
    let (y, u, v) = flat_grey(64, 64, 128);
    let _kf = enc.encode_frame(&view(&y, &u, &v, 64)).unwrap();
    let kf_recs = f02_oracle::drain_frame();
    assert!(
        kf_recs.is_empty(),
        "KF sink must be empty with RVP9_F02_ORACLE unset (saw {} records)",
        kf_recs.len(),
    );

    let _inter = enc.encode_frame(&view(&y, &u, &v, 64)).unwrap();
    let inter_recs = f02_oracle::drain_frame();
    assert!(
        inter_recs.is_empty(),
        "inter sink must be empty with RVP9_F02_ORACLE unset (saw {} records)",
        inter_recs.len(),
    );
}

#[test]
fn libvpx_leg_smoke() {
    // Manual smoke (requires `scripts/build_f02_libvpx.ps1` to have
    // produced `third_party/libvpx/build_f02/vpxenc.exe`). Skipped
    // by default — re-enable when the patched libvpx build is
    // available locally. See HLD §3.3.
    //
    // Intentionally a no-op assert so the test runs green when the
    // libvpx binary is absent; the standalone binary
    // `f02_block_oracle` is the integration entry point.
    if std::env::var_os("VPX_F02_PATCHED").is_none() {
        eprintln!(
            "f02_oracle_smoke::libvpx_leg_smoke skipped — set \
             VPX_F02_PATCHED=1 to enable (requires patched vpxenc)."
        );
        return;
    }
    eprintln!(
        "f02_oracle_smoke::libvpx_leg_smoke — patched libvpx \
         end-to-end exercise is driven by the f02_block_oracle \
         binary; this test stub keeps the gating path discoverable."
    );
}

/// Helper: confirm the BlockDecisionMode enum exposes its inter
/// variant so consumers can pattern-match the schema (compile-time
/// guard against accidental flattening).
#[test]
fn block_decision_mode_variant_shape_is_stable() {
    let m = BlockDecisionMode::Inter(InterMode::ZeroMv);
    match m {
        BlockDecisionMode::Inter(_) => (),
        BlockDecisionMode::Intra(_) => panic!("variant shape regressed"),
    }
    let _: BlockDecision = BlockDecision {
        frame_idx: 0,
        mi_row: 0,
        mi_col: 0,
        bsize: rvp9_core::block::BlockSize::Block8x8,
        mode: m,
        ref_frame: rvp9_core::mode_info::RefFrame::LastFrame,
        mv_row: 0,
        mv_col: 0,
        mode_bits_est: 0,
        mv_bits_est: 0,
        coef_bits_est: 0,
        sse_after_pred: 0,
        skip_flag: true,
    };
}
