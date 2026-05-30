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

## RESULTS
### RCFI (commit 56764d5, default-ON) — SMALL
foreman mean +3.00 → +2.96 (+0.16 @250k only; 25/50/100k byte-identical —
buffer/q-saturation prevents the first-inter investment from binding at low br).
testsrc byte-identical. Faithful libvpx port; goldens w_buffer_pool + w_gop_topology
re-captured.

### KFLB (low-bitrate KF-share cap, default-ON) — BIG, the real low-br trunk
Diagnosis: KFCAP (40%-of-GOP-budget KF target cap) only fires in C1's *affordable*
branch. At LOW bitrate the C1 gate takes the *cap* branch (active_best≈q57 floor)
and leaves the KF target un-shaved → KF over-allocates (foreman 25k KF 46344 bits vs
libvpx 22824; 50k 75432 vs 41168). The 40% ceiling aligns with libvpx's KF sizes.
Fix: apply the same target cap in the cap branch (cbr.rs else-branch, gated RVP9_KFLB).
- foreman mean **+2.96 → +2.51 dB** (−0.45): 25k +3.08→+2.35 (br 68.9→57.5k),
  50k +3.86→+2.80 (br 83.6→68.2k). 100k/250k inert (KF already fits the cap).
- testsrc neutral-to-+0.31 (no regression). cross_decoder 4/4. goldens unchanged
  (inert at fixture bitrates — KF target clamps to max_frame_bw == cap there).
- 456/456 lib tests single-threaded (one parallel run hit a pre-existing env-race
  flake on aq_cyclic_refresh; green on re-run + single-threaded).
- Scoped to the FIRST keyframe (`is_first_frame`). Subsequent-KF share-cap = future.

### LPFQ (loop-filter level = libvpx LPF_PICK_FROM_Q, default-ON) — universal win
Diagnosis: rvp9's `pick_filter_level(q) = (q>>1).min(63)` over-filters at mid/low q.
libvpx (cpu7 `sf->lpf_pick = LPF_PICK_FROM_Q`) uses an ac_quant-based calibrated
formula: `filt_guess = ROUND_POWER_OF_TWO(ac_quant(q)*20723 + 1015158, 18)`, KF -4.
At the KF q≈57: rvp9 proxy = 28 vs libvpx ≈5 → rvp9 over-smooths the anchor reference;
every inter inherits the lost detail. Fix: port the formula (gated RVP9_LPFQ).
- foreman mean +2.51 → **+2.43 dB** (250k +0.27 — q low enough for the over-filter to
  bite; 25/100k small KF-propagated gains; 50k +0.03 noise).
- testsrc STRICTLY better (sharp synthetic edges the over-filter blurred): 25k +1.83,
  50k +1.03, 100k +3.19 dB, all at equal-or-fewer bits.
- cross_decoder 4/4. goldens w_buffer_pool 6778→6625 + w_gop_topology 24727→23722
  re-captured. 456/456 lib tests. f11 tests rewritten as property oracles.

## CUMULATIVE: foreman_cif cpu7 mean matched-bitrate Δ +3.00 → +2.43 dB this session
(RCFI + KFLB + LPFQ; testsrc strongly improved by LPFQ — no regression anywhere).

## Remaining gap (honest)
- Mode/skip cost-model: EXHAUSTED (domain mismatch = Lever-3b, journal-confirmed NULL
  in-scope; differential read found picker otherwise faithful).
- Biggest unexploited lever = ALT-REF / temporal-filtering architecture (libvpx builds
  periodic high-quality refs + coasts; rvp9 flat). Multi-week, architectural → Arthur.
