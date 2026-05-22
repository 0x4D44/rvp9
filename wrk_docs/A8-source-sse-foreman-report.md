# A8.source-SSE — F8.1 picker source-domain SSE experiment

Date: 2026-05-22.
Worktree: `D:\language\rvp9-2\.claude\worktrees\agent-a5a832dd7221fec94`.
Goal: replace the F8.1 inter picker's post-recon SSE with
source-domain SSE (SSE(src, pred)) and measure foreman_cif PSNR.

## Hypothesis (from brief)

The current F8.1 picker uses post-recon SSE: `D = sum((src - recon)^2)`
where `recon = pred + IDCT(Q^-1(Q(DCT(src - pred))))`. The brief
asserted that source-domain SSE `D = sum((src - pred)^2)` would
activate NEWMV picks on foreman_cif (textured motion content) and
lift PSNR.

Mechanism: at high q, the picker's residual quantises to zero so
recon = pred and post-recon SSE = source-domain SSE — same metric.
But when residuals SURVIVE quant, post-recon SSE underestimates the
"true badness" of high-residual modes because dequant partially
fixes them. The picker doesn't charge for the residual bits the
encoder will emit downstream (only mode + MV bits), so high-residual
modes (NEARESTMV with stale ref) look artificially cheap.

## What I did

Replaced the distortion metric in
`crates/rvp9-core/src/mode_decision/inter.rs::evaluate_candidate`
to compute SSE against `pred` (before round-trip) instead of `recon`.
Reconstruction round-trip still runs (downstream consumers need
recon) but is no longer the RD distortion source.

## Measurements

### NEWMV pick activation (foreman_cif, 64f, 4 bitrates)

| Metric | BEFORE | AFTER |
|---|---|---|
| NEWMV picks | 0 | 3788 |
| NEARESTMV picks | 199429 | 192984 |
| ZEROMV picks | 26972 | 27028 |
| Total leaves | 226401 | 223800 |

**Source-domain SSE activates NEWMV picks (0 → 3788).** Confirms
the brief's hypothesis on the mechanism.

NEWMV picks concentrated at q=135-200 (1740 of 3788 at q∈[135,175]).
At q=255 (31% of leaves) NEWMV still never wins — residuals quant
to zero so both source-domain and post-recon SSE are identical, and
the rate cost still dominates.

### foreman_cif PSNR (256x256 centre-crop, 64f)

| Target | BEFORE kbps/PSNR | AFTER kbps/PSNR |
|---|---|---|
|  25 kbps | 127.0 / 16.36 | 126.9 / 16.36 |
|  50 kbps | 159.7 / 16.21 | 158.9 / 16.21 |
| 100 kbps | 249.0 / 16.28 | 247.1 / 16.26 |
| 250 kbps | 514.6 / 16.55 | 482.7 / 16.54 |

**foreman PSNR essentially unchanged.** The 16 dB floor is NOT
broken. Tiny bitrate savings (6.2% at 250 kbps) but no PSNR
movement. RC q=255 saturation remains the dominant blocker (A6
finding).

### testsrc PSNR (256x256, 64f)

| Target | BEFORE kbps/PSNR | AFTER kbps/PSNR |
|---|---|---|
|  25 kbps | 68.4 / 35.93 | 66.0 / 35.79 (-0.14) |
|  50 kbps | 112.7 / 38.68 | 100.8 / 38.90 (+0.22) |
| 100 kbps | 174.7 / 43.08 | 144.4 / 43.39 (+0.31) |
| 250 kbps | 262.3 / 59.69 | 253.0 / 57.77 (-1.92) |

**testsrc 250 kbps regresses by 1.92 dB.** Mid-bitrate slightly
improved, lowest-bitrate slight drop. The high-bitrate regression
violates the brief's "don't ship a regression" constraint.

Mechanism: at low q on testsrc, residuals SURVIVE quant. With
source-domain SSE the picker chooses NEWMV based on pred quality
alone, ignoring the residual cost the encoder will emit. NEWMV's
extra MV bits + the unaccounted residual bits combine to push the
encoding off the RD curve. Post-recon SSE implicitly accounts for
the residual cost (via the dequant-fixup magnitude), keeping the
NEWMV-vs-NEAREST tradeoff balanced.

### Cross-decoder gate

PASS — rvp9-decoder, vpxdec, ffmpeg all BYTE-EXACT on
source-domain SSE encoding. The pick changes the bitstream
contents but every decoder agrees the bitstream is valid.

### RT gate

PASS — median 5.74 ms (well under 33 ms budget).

### Workspace clippy

Pre-existing errors in Wave F kernel files
(`lpf_vertical_4_avx2.rs`, etc.) — not affected by this change.
No new warnings in `mode_decision/inter.rs`.

### Inter mode_decision unit tests

All 6 tests pass (zero-mv-on-identical-ref, finds-new-mv-on-shifted-ref,
finite-rd-cost, walks-3-candidates, block-size-smoke,
mode-ctx-affects-rate).

## Interpretation

The brief's mechanism is partially confirmed: source-domain SSE
**does** activate NEWMV picks (0 → 3788), which was the stated
goal. But:

1. **foreman doesn't climb.** The 16 dB floor on foreman is set by
   RC pinning q=255 (A6, A7 findings, reconfirmed here). At q=255
   residuals quant to zero so source-domain SSE and post-recon SSE
   are identical; NEWMV picks don't change reconstruction quality
   because reconstruction = pred regardless. The NEWMV picks that
   DO activate (q=135-200) happen on KF / first-few-inter frames
   where the picker already had some room to differentiate; they
   don't shift the 64-frame PSNR average.

2. **testsrc regresses at 250 kbps.** -1.92 dB at high bitrate is
   a real regression. Source-domain SSE ignores the downstream
   residual rate cost; the encoder emits more bits for NEWMV
   residuals than the picker accounted for, blowing the high-q
   RD optimum.

## Decision: revert

Source-domain SSE is the WRONG fix for the foreman PSNR floor.
The real blocker is RC q=255 saturation, not the picker's
distortion metric. Shipping source-domain SSE would trade a
testsrc regression for a non-improvement on foreman.

## Files touched (then reverted)

- `crates/rvp9-core/src/mode_decision/inter.rs::evaluate_candidate`
  — switched distortion to SSE(src, pred). Reverted to SSE(src, recon)
  with a post-mortem comment block documenting this experiment.

## Best next hypothesis

Per A7's recommendation (and A6's diagnostic): the real lever is
**RC q-ceiling**. RC saturates qindex at 255 on foreman_cif, which
quantises every residual to zero and locks recon = LASTFRAME.

Two cheaper experiments to try next:

A. **Clamp inter q-ceiling at ≤200.** One-line change in the RC
   that says `q = min(q, 200)` on inter frames. If foreman PSNR
   climbs from 16 → 22+ dB at the cost of bitrate overshoot, the
   real blocker is identified. If overshoot is small and PSNR
   climbs significantly, ship it (or a less aggressive clamp).

B. **Reduce per-leaf overhead.** At q=255 each leaf emits ~3-7
   bits of mode/skip symbols. Foreman has ~1024 Block8x8 leaves
   per frame; that's 3-7 kbits/frame floor at zero residual. At
   25 kbps × 30 fps = 833 bits/frame budget, the per-leaf
   overhead alone exceeds the target. A7 tried capping at
   Block16x16 (256 leaves) and it dropped bitrate ~12% but PSNR
   stayed flat. A more aggressive cap (Block32x32 on truly
   uniform motion regions) or a SB-level skip-all-leaves
   shortcut might help.

Of the two, A is cheaper and more diagnostic. B requires
detection logic for "uniform motion region" which is itself a
design problem.

## Status

Reverted. Worktree branch `worktree-agent-a5a832dd7221fec94` carries
only the post-mortem comment block in `inter.rs` + this report +
divergence ledger note.
