# RC lever — first-inter active_best starvation (2026-05-30)

## Baseline (HEAD 8e288c7, quality_gate foreman_cif cpu7)
rvp9 trails libvpx **+3.00 dB mean** matched-bitrate (worst +3.86 @ ~83 kbps).
rvp9 overshoots low targets (25k→69k vs libvpx 25k→46k).

## Mode/skip ground-truth (e1_block_stats HEAD)
- ZEROMV 22-26% vs libvpx 3-4% (over-use); skip 65-82% vs 81-97% (under-use).
- Differential read (libvpx vp9_pickmode.c vs inter.rs): picker is FAITHFUL except the
  RD coef-domain-rate vs pixel-SSE-dist mismatch — that is Lever-3b, already EXHAUSTED
  (helps foreman, regresses synthetic; only ~0.13 dB beyond shipped S2). **Mode cost-model
  lever is mined out.**

## The real lever — GOP bit allocation (per-frame data, 100k)
libvpx invests in a high-quality FIRST inter (frame 1: q114, 8752 bits) then coasts
(frames cost 144-688 bits later). rvp9 codes FLAT (q pinned ~227, ~2000-3000 bits/frame).
libvpx inter-q swings 114-255; rvp9 218-242.

## Root cause (CONFIRMED in code)
`rate_control/cbr.rs::pick_q_and_bounds` Inter branch (lines 761-772) always uses
`avg_frame_qindex_inter` as the active_best basis. That field SEEDS to max_qindex (255)
(`state.rs:240`). So the first inter frame computes:
  active_best = rtc_minq_approx(255) ≈ 0.75*255 = 191  →  q floored at 191  →  q=242.

libvpx (`vp9_ratectrl.c:1058-1068`) has a `current_video_frame > 1` special case: the
FIRST inter of the stream seeds active_best from `avg_frame_qindex[KEY]` (≈57 after the
KF) → active_best ≈ rtc_minq[57] ≈ 42 → q can drop to ~114 → invests in a good reference
the whole GOP coasts on. rvp9's `rate_control/mod.rs:48-51` notes the GF/ARF Q paths were
skipped in v1; this first-inter case was skipped with them.

## Fix (gated RVP9_RCFI, default ON; =0 = byte-identical baseline)
Port the `current_video_frame <= 1` arm: first inter frame uses `avg_frame_qindex_key`
basis. Add a `current_video_frame` counter to RateControl (incr in advance_frame),
mirroring libvpx exactly (single-GOP test fires it once on frame 1).

## Validation plan
- quality_gate foreman (expect mean Δ drop) + testsrc (no regression).
- cross_decoder 4/4 byte-exact; goldens re-capture if needed.
- e1 per-frame: frame-1 q should drop ~242→~114, bits rise.
