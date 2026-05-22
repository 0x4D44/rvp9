# E12 — pinning the prediction-quality nature: the decisive oracle-MV test

**VERDICT: SEARCH FAILURE — and specifically the FULL-PEL diamond
search does not converge.** An exhaustive MV oracle finds a prediction
**+8.4 dB** better than what rvp9 ships, and **+7.6 dB** of that is
recoverable with FULL-PEL MVs alone (sub-pel adds only +0.3 dB). This
is a TRACTABLE next fix, and it overturns the E3/E9 "NEWMV finds good
MVs but loses on RD" story: NEWMV's *raw search output* is only +0.5 dB
better than (0,0). The search itself is stuck near its seed.

Diagnostic-only. foreman_cif 256×256, 64 frames, AQ off, fixed q=120
(`RVP9_FIXED_Q`). Oracle env-gated `RVP9_E12_ORACLE` (off by default;
94/94 encoder cross-decode tests still green — byte-exact output
unchanged). Harness:
`cargo test -p rvp9-encoder --release --test e12_oracle -- --ignored --nocapture`.
Sampled ~1/7 of picker-supported inter leaves (n=7589 over frames 2-63).

## Method

At the LAST-ref picker tap (`lib.rs:2944`) the oracle computes, for each
sampled leaf, the **prediction PSNR** (source vs MC prediction, BEFORE
transform — pure MC quality) for four MVs, all built with the SAME
`inter_predictor` (identity scale, EightTap) the encoder reconstruction
uses:
1. **ZERO** (0,0);
2. **shipped** — the MV rvp9's RD argmin actually ships (`pick_last`);
3. **NEWMVraw** — NEWMV's diamond+sub-pel search output BEFORE the RD
   argmin can reject it (`pick_inter_mode_luma_all_costs[3].best_mv`);
4. **oracle** — exhaustive: dense ±16 full-pel grid, then ±half/quarter-
   pel around the best full-pel. Also split out **oracle_fullpel**
   (phase-1 only).

## 1. Per-frame prediction PSNR — cascade test (Q1)

| group | n | ZERO | shipped | NEWMVraw | oracle |
|-------|---|------|---------|----------|--------|
| FIRST inter frame (predicts fresh KF) | 113 | 27.54 | 27.54 | 27.64 | **36.98** |
| LATER inter frames (mid-GOP) | 7589 | 26.18 | 26.18 | 26.70 | **34.56** |

The first inter frame, predicting the freshly-coded keyframe, is ALREADY
poor at shipped 27.5 dB — **NOT a pure reference cascade**. The oracle
reaches 37 dB on that very frame, so the reference is good enough; rvp9
just doesn't use it. (libvpx anchor, E9: 39.23 dB recon / 15.6 kbit at
this Q — consistent with low-30s prediction.)

## 2. The decisive split (LATER inter frames, area-weighted)

| comparison | Δ dB |
|------------|------|
| oracle − shipped | **+8.37** |
| oracle − NEWMVraw | +7.86 |
| **NEWMVraw − shipped** | **+0.52** |
| oracle_fullpel − NEWMVraw | **+7.58** |
| sub-pel adds (oracle − oracle_fullpel) | **+0.27** |

- oracle beats shipped by ≥2 dB on **82%** of leaves (6204/7589).
- oracle-best MV is sub-pel on **84%** (foreman's pan is sub-pixel).

## 3. Localisation — it's the FULL-PEL diamond, not sub-pel

NEWMVraw is barely above (0,0): **+0.52 dB**. The full-pel-only oracle
already recovers **+7.58 dB** over NEWMVraw; adding sub-pel contributes
only **+0.27 dB** more. So although 84% of true motions are sub-pel,
the dominant deficit is finding the right **full-pel** location — the
diamond search converges back to near its seed instead of the reachable
good MV. Sample leaves show oracle MVs at full-pel magnitudes the
diamond never reaches (e.g. col −80, +108, row +100 in Q3) while NEWMV
returns ≈(0,0).

## 4. Sub-pel interpolation faithfulness

Verified faithful by construction: the oracle's prediction reuses the
production `rvp9_core::inter::inter_predictor` (EightTap, identity
scale) — the exact kernel the encoder recon and decoder MC use. If the
8-tap filters were wrong, the oracle's sub-pel candidates would not
out-predict full-pel ones — but they consistently add a small positive
gain, and sub-pel candidates win 84% of oracle picks. Sub-pel is NOT
the bug.

## Root cause map → libvpx

The full-pel diamond (`vp9_diamond_search_sad_c`, our
`mv_search/full_pixel.rs::diamond_search_sad`) is seeded at the int_pro
MV or `nearest=(0,0)` and its `mvsad_err_cost` penalises distance from
`ref_mv` (frame-wide (0,0), E9 §4), dragging the step search back toward
the seed. With a near-(0,0) seed and a (0,0) ref-MV attractor, the
4-site descent terminates without reaching foreman's larger true
motions. libvpx avoids this via (a) a better SB-level `pred_mv` seed
propagated into `x->pred_mv[LAST]`, and (b) the NSTEP/8-site search and
`use_prev_frame_mvs` candidate seeding feeding `vp9_full_pixel_search`
(`vp9_mcomp.c`). **Next tractable fix: full-pel search seed + descent
convergence** (seed quality / `sad_per_bit` attractor / search pattern),
NOT the sub-pel stage and NOT a multi-week reference-quality rebuild.
