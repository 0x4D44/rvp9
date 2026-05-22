# A7.inter-cap — Inter-frame Block16x16 cap experiment

Date: 2026-05-22.
Goal: extend A1.foreman's KF Block16x16 partition cap to inter frames
and measure foreman_cif PSNR + testsrc PSNR.

## Hypothesis (from A6.lambda + brief)

A6's diagnosis: foreman_cif PSNR is locked at 16 dB because:
1. RC pins qindex at 255 across all bitrate targets
2. Per-leaf overhead (mode/skip/MV symbols) creates a 0.3-0.6 Mbps
   floor at 30 fps with ~1024 Block8x8 leaves
3. Bitrate overshoots target 2-5× even with empty residuals
4. q=255 quantizes residuals to zero → recon = LASTFRAME → 16 dB floor

The hypothesis was: capping inter at Block16x16 (same as KF) reduces
per-leaf count from 1024 to ~256 → fewer mode bits → RC can lower q →
non-empty residuals → PSNR climbs.

## What I did

In `crates/rvp9-core/src/mode_decision/partition.rs::descend`, made
the Block16x16 → Block8x8 cap unconditional (was gated on
`is_keyframe`). Updated the two `tests` that asserted "chaotic SBs
split deeply to Block8x8" to instead assert split to Block16x16.

## Measurements

### testsrc (256x256, 64f, F12 corpus)

| Target (kbps) | Baseline kbps / PSNR | Capped kbps / PSNR |
|---|---|---|
|  25 | 68.4 / 35.93 | 58.5 / 35.75 |
|  50 | 112.7 / 38.68 | 101.4 / 38.75 |
| 100 | 174.7 / 43.08 | 165.6 / 42.83 |
| 250 | 262.3 / 59.69 | 265.8 / 59.42 |

testsrc: slight PSNR regression (-0.18 dB at 25 kbps, -0.27 dB at 250
kbps); bitrate dropped ~6% at low rates.

### foreman_cif (256x256 centre-crop, 64f, F12 corpus)

| Target (kbps) | Baseline kbps / PSNR | Capped kbps / PSNR |
|---|---|---|
|  25 | 127.0 / 16.36 | 85.0 / 16.33 |
|  50 | 159.7 / 16.21 | 129.1 / 16.19 |
| 100 | 249.0 / 16.28 | 232.0 / 16.26 |
| 250 | 514.6 / 16.55 | 502.1 / 16.53 |

**foreman PSNR: UNCHANGED. Still locked at 16 dB across all targets.**
Bitrate dropped 33% at 25 kbps target and 19-7% at higher targets.

### Cross-decoder gate

PASS — rvp9-decoder, vpxdec, ffmpeg all BYTE-EXACT on the capped
encoding.

### RT gate

PASS — median 5.71 ms (well under 33 ms budget).

## Interpretation

The experiment **disproves** the bit-budget hypothesis:
- Capping at Block16x16 reduces foreman bitrate from 514 → 502 kbps
  at 250 kbps target (~2% drop, ~17 kbps savings)
- But PSNR did NOT climb. q is STILL pinned at 255.
- Fewer leaves → fewer mode/skip symbols → fewer bits → RC is
  *closer* to target but still over because the per-leaf overhead is
  still dominant.
- Even if every byte saved went to residuals, at q=255 the dequant
  step is so coarse that residuals quantize to zero anyway.

The real blocker is one or more of:
1. **q=255 is just too coarse for foreman content.** At q=255 the
   AC dequant is in the thousands — any real motion residual is
   smaller than that step and rounds to zero. RC needs to STOP pinning
   q at 255 and accept buffer overflow.
2. **The picker never picks NEWMV.** From A6 diag: 86,115 non-skip
   leaves total, ZERO NEWMV picks. ZEROMV/NEARESTMV both have MV=(0,0)
   → recon = LASTFRAME-at-same-position. Without NEWMV there's no
   motion compensation.
3. **The RC target-bitrate units may be wrong.** A 250 kbps target
   producing a 500+ kbps actual bitrate at q=255 means the RC budget
   model is mis-calibrated for foreman content.

## Recommendation for next experiment

Per the brief's "Alternative experiments":

**Try A: Force NEWMV picks heuristically.** The picker has all the
int_pro seed wiring in place but the RD math rejects NEWMV at q=255
because lambda × mv_rate swamps the distortion gain. Try a heuristic
override: if int_pro produces an MV magnitude > 2 full-pels AND
NEWMV's MV-cost is below some threshold, force-pick NEWMV regardless
of RD. This breaks libvpx-faithfulness but tests whether motion
compensation alone would lift PSNR above 16 dB.

**Or Try B: Loosen RC q-ceiling.** Currently q saturates at 255. Try
clamping q to ≤ 200 (or whatever quant step lets a typical foreman
residual survive) and let the buffer overshoot. If foreman PSNR
climbs from 16 → 25+ dB at the cost of bitrate overshoot, the real
blocker is identified and we can engineer a proper fix.

Of the two, B is cheaper to try (one-line RC change) and more
diagnostically informative.

## Files touched (then reverted)

- `crates/rvp9-core/src/mode_decision/partition.rs` — unconditional
  cap experiment + test updates. **Reverted.** Only the post-mortem
  comment block stays in tree.

## Status

Reverted, not committed (per brief: "if experiment doesn't pan out,
revert and document"). Comment block added to partition.rs at the
descent cap to record the negative result.
