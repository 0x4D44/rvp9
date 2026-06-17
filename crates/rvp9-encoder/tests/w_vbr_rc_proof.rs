//! W_VBR PROOF — one-pass VBR rate-control completes a 30-frame run and
//! produces per-frame Q-indices that respect the libvpx-faithful
//! algorithm.
//!
//! libvpx reference: `third_party/libvpx/vp9/encoder/vp9_ratectrl.c`
//! `rc_pick_q_and_bounds_one_pass_vbr` (line 1138-1303),
//! `calc_active_worst_quality_one_pass_vbr` (905-928),
//! `vp9_calc_iframe_target_size_one_pass_vbr` (2039-2048),
//! `vp9_calc_pframe_target_size_one_pass_vbr` (2019-2037).
//!
//! ## What this asserts
//!
//! 1. **The encode COMPLETES.** Before W_VBR, the VBR arm of
//!    `RateControl::pick_q_and_bounds` panicked with "1-pass VBR ... not
//!    yet implemented". A panic-free 30-frame run is the primary
//!    behavioural proof that the port replaced the stub.
//! 2. **Q varies across frame types.** KF and inter frames pick
//!    different Q in steady state (different active_best floors,
//!    different per-frame targets — the keyframe target is 25× avg
//!    while the inter target is 0.625× avg in the libvpx VBR model).
//! 3. **Bounds enforcement.** Every per-frame
//!    `(q, active_best, active_worst)` triple satisfies
//!    `active_best <= q <= active_worst` (libvpx
//!    `vp9_ratectrl.c:1287-1295` regulate_q + clamp), and the bounds
//!    themselves stay within the controller's
//!    `[best_quality, worst_quality]` budget.
//! 4. **Bounds are libvpx-style — not the CBR shape.** VBR's
//!    active_worst is ambient-Q-driven (`avg_inter * 150/100`); CBR's
//!    is buffer-driven. The two RC modes must therefore produce
//!    different `this_frame_target` and (typically) different picked Q
//!    on the same input at the same starting state. We assert
//!    `this_frame_target` divergence — the cleanest oracle since both
//!    sides are deterministic given the controller's state.
//!
//! This test drives the rate controller directly (no bitstream
//! encoding) — the encoder front-end doesn't plumb `rc_mode` through
//! `EncoderConfig`. That keeps the cross-decoder gate byte-exact by
//! construction: this proof never touches the CBR-default encode path.

use rvp9_encoder::rate_control::{FrameKind, RateControl, RateControlConfig, RcMode};

const FPS: f64 = 30.0;
const N_FRAMES: u32 = 30;
const GOP: u32 = 30; // One KF at frame 0; rest are inter.
const TARGET_BPS: u32 = 1_000_000;

fn vbr_config() -> RateControlConfig {
    RateControlConfig {
        rc_mode: RcMode::Vbr,
        target_bitrate_bps: TARGET_BPS,
        framerate: FPS,
        gop_length: GOP,
        ..RateControlConfig::default()
    }
}

fn cbr_config() -> RateControlConfig {
    RateControlConfig {
        rc_mode: RcMode::Cbr,
        target_bitrate_bps: TARGET_BPS,
        framerate: FPS,
        gop_length: GOP,
        ..RateControlConfig::default()
    }
}

/// Drive `n` frames through the rate controller, simulating an encoder
/// that hits the per-frame target exactly. Returns the per-frame
/// `(frame_kind, q, active_best, active_worst, this_frame_target)`.
///
/// Hitting the target exactly is the cleanest oracle: the controller's
/// buffer model stays at its starting state, and any per-frame Q
/// variation comes from the rate model itself (not from buffer
/// pressure). This isolates VBR's structural Q decisions from
/// post-encode buffer feedback noise.
fn run(rc: &mut RateControl, n: u32) -> Vec<(FrameKind, u8, u8, u8, u32)> {
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let frame_kind = if i == 0 {
            FrameKind::Key
        } else {
            FrameKind::Inter
        };
        let (q, lo, hi) = rc.pick_q_and_bounds(frame_kind);
        let target = rc.this_frame_target;
        // Pretend the encoder spent exactly the per-frame target.
        rc.postencode_update(target, frame_kind);
        rc.advance_frame();
        out.push((frame_kind, q, lo, hi, target));
    }
    out
}

/// PROOF #1 — the VBR dispatcher does not panic for 30 frames.
///
/// Before this wave, `RateControl::pick_q_and_bounds` panicked on
/// `RcMode::Vbr`. Reaching the end of `run(&mut rc, 30)` without
/// unwinding is the primary "panic arm is gone" proof.
#[test]
fn vbr_completes_30_frames_without_panic() {
    let mut rc = RateControl::new(vbr_config());
    let trace = run(&mut rc, N_FRAMES);
    assert_eq!(
        trace.len(),
        N_FRAMES as usize,
        "VBR PROOF #1: pick_q_and_bounds must return for all {N_FRAMES} frames"
    );
}

/// PROOF #2 — Q-bounds are enforced per frame.
///
/// libvpx's `rc_pick_q_and_bounds_one_pass_vbr` returns `q` from
/// `regulate_q`, which is clamped to `[active_best, active_worst]`
/// (`vp9_ratectrl.c:1287-1295`). We verify this contract holds for
/// every emitted frame. Additionally the bounds themselves must lie
/// within the controller's `[best_quality, worst_quality]` budget
/// (libvpx's `clamp(...)` at lines 1252-1255).
#[test]
fn vbr_q_respects_active_best_and_active_worst_bounds_every_frame() {
    let mut rc = RateControl::new(vbr_config());
    let best_q = rc.config().min_qindex;
    let worst_q = rc.config().max_qindex;
    let trace = run(&mut rc, N_FRAMES);
    for (idx, (kind, q, lo, hi, target)) in trace.iter().enumerate() {
        assert!(
            *lo <= *q && *q <= *hi,
            "VBR PROOF #2 frame {idx} ({kind:?}): q={q} must lie in \
             [active_best={lo}, active_worst={hi}], target={target}"
        );
        assert!(
            *lo >= best_q,
            "VBR PROOF #2 frame {idx}: active_best ({lo}) must be >= \
             best_quality ({best_q})"
        );
        assert!(
            *hi <= worst_q,
            "VBR PROOF #2 frame {idx}: active_worst ({hi}) must be <= \
             worst_quality ({worst_q})"
        );
    }
}

/// PROOF #3 — KF and steady-state inter pick different Q.
///
/// VBR's per-frame target differs structurally by frame type:
/// - KF target = `avg_bw * kf_ratio (= 25)` — libvpx `vp9_ratectrl.c:2039-2048`.
/// - Inter target = `avg_bw * baseline_gf_interval / (baseline_gf_interval
///   + af_ratio - 1)` ≈ `0.625 * avg_bw` — libvpx `vp9_ratectrl.c:
///   2019-2037`.
/// Combined with the different `active_best` floors
/// (`get_kf_active_quality` vs `inter_minq`), the picked Q must differ
/// between the keyframe and subsequent inter frames. We compare frame 0
/// (KF) against frame 5 (a steady-state inter — far enough past the
/// first-inter fallback at `vp9_ratectrl.c:1241` that the
/// `avg_inter * 150/100` ambient model is active).
#[test]
fn vbr_kf_q_differs_from_steady_state_inter_q() {
    let mut rc = RateControl::new(vbr_config());
    let trace = run(&mut rc, N_FRAMES);

    let (kf_kind, kf_q, _, _, kf_target) = trace[0];
    assert_eq!(kf_kind, FrameKind::Key);

    // Pick an inter from well past the first-inter fallback.
    let (inter_kind, inter_q, _, _, inter_target) = trace[10];
    assert_eq!(inter_kind, FrameKind::Inter);

    assert_ne!(
        kf_q, inter_q,
        "VBR PROOF #3: KF Q ({kf_q}, target={kf_target}) must differ \
         from steady-state inter Q ({inter_q}, target={inter_target}) \
         under the libvpx VBR rate model"
    );

    // Strengthen: the targets themselves must differ.
    assert_ne!(
        kf_target, inter_target,
        "VBR PROOF #3 corollary: per-frame targets must differ by frame \
         type (KF: avg*25; inter: avg*15/24)"
    );
}

/// PROOF #4 — VBR diverges from CBR at the same starting state.
///
/// CBR's `active_worst_quality` is buffer-driven (`calc_active_worst_
/// quality_one_pass_cbr`, `vp9_ratectrl.c:931-1000`); VBR's is ambient-
/// Q-driven (`calc_active_worst_quality_one_pass_vbr`, lines 905-928).
/// CBR's KF target is `starting_buffer_level / 2` (~2 Mbits at 1 Mbps /
/// 4 s buffer); VBR's is `avg_bw * 25` (~833 kbits at 1 Mbps / 30 fps).
/// Running both controllers from a fresh `RateControl::new` state must
/// produce a different `this_frame_target` on frame 0, proving the
/// dispatcher routes through VBR's path (not CBR's).
#[test]
fn vbr_targets_diverge_from_cbr_on_first_frame() {
    let mut rc_cbr = RateControl::new(cbr_config());
    let mut rc_vbr = RateControl::new(vbr_config());

    let (_, _, _) = rc_cbr.pick_q_and_bounds(FrameKind::Key);
    let (_, _, _) = rc_vbr.pick_q_and_bounds(FrameKind::Key);

    assert_ne!(
        rc_cbr.this_frame_target, rc_vbr.this_frame_target,
        "VBR PROOF #4: CBR (target={}) and VBR (target={}) must pick \
         different first-frame targets — CBR uses starting_buffer_level/2, \
         VBR uses avg_bw * kf_ratio. If these match, the VBR dispatcher \
         is silently routing through CBR.",
        rc_cbr.this_frame_target, rc_vbr.this_frame_target,
    );
}

/// PROOF #5 — Q-indices stay in a reasonable range across the full run.
///
/// Sanity gate: no degenerate output. `q` must lie strictly within
/// `(best_quality, worst_quality)` — picking q=0 (lossless edge) or
/// q=255 (worst) for every frame would indicate the regulator is
/// broken. We allow the absolute bounds inclusive at the edges (a
/// brief brush is fine) but require at least one frame to land in the
/// "interesting" middle band [20, 230] — confirming the algorithm
/// is actually choosing Q, not just clamping.
#[test]
fn vbr_q_stays_in_reasonable_range() {
    let mut rc = RateControl::new(vbr_config());
    let best_q = rc.config().min_qindex;
    let worst_q = rc.config().max_qindex;
    let trace = run(&mut rc, N_FRAMES);
    let mut saw_middle_band = false;
    for (idx, (kind, q, _, _, _)) in trace.iter().enumerate() {
        assert!(
            *q >= best_q && *q <= worst_q,
            "VBR PROOF #5 frame {idx} ({kind:?}): q={q} must lie in \
             [{best_q}, {worst_q}]"
        );
        if (20..=230).contains(q) {
            saw_middle_band = true;
        }
    }
    assert!(
        saw_middle_band,
        "VBR PROOF #5: at least one frame must pick Q in the working \
         range [20, 230] — if every frame is at the absolute floor or \
         ceiling, the regulator isn't engaging"
    );
}
