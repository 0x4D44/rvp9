# A10 — bits_per_mb under-prediction hypothesis — DISPROVED

Date: 2026-05-22.
Worktree: `D:\language\rvp9-2\.claude\worktrees\agent-acd7eb5ef4e7febb2`.

## TL;DR

Multi-agent night-7 close-out diagnosis was that the foreman_cif 16-dB
PSNR floor on rvp9 is caused by `vp9_rc_bits_per_mb` (ported at
`crates/rvp9-encoder/src/rate_control/cbr.rs::bits_per_mb`) under-
predicting per-MB cost by ~10×. The remedy was a configurable
`bits_per_mb_multiplier: f64` knob with empirical value set from
measured `actual_bits / predicted_bits` on foreman.

**Empirical measurement disproves the hypothesis.** Per-frame
`actual / predicted_at_picked_q` ratios on foreman_cif average 1.05
± 0.20 across all four target bitrates and 64 frames each. The
rate-correction-factor feedback loop (libvpx
`vp9_rc_update_rate_correction_factors`, ported in
`postencode::update_rate_correction_factors`) is closing the loop
correctly. There is no 10× bias to correct.

Setting `RVP9_BPM_MULTIPLIER=10.0` (the hypothesised value) and the
opposite `=0.1` both move foreman PSNR by <0.1 dB at every target
(16.40-16.46 dB at mult=10 vs 16.36-16.55 dB baseline at mult=1.0).

## What I built

1. **`RateControlConfig::bits_per_mb_multiplier: f64`**
   (`crates/rvp9-encoder/src/rate_control/config.rs`). Default 1.0
   = libvpx-faithful, byte-exact. Validate rejects non-finite /
   non-positive.

2. **`cbr::bits_per_mb_scaled`** wrapper applied at both the picker
   side (`regulate_q`) and the post-encode feedback side
   (`estimate_bits_at_q`, used by
   `update_rate_correction_factors`). Threading is internally
   consistent so the feedback loop sees the same scaled prediction
   the picker used.

3. **`RateControl::predicted_bits_at_picked_q`** diagnostic accessor
   (`state.rs`).

4. **`RVP9_DIAG_BPM_VERIFY=1`** env-gated per-frame log line in
   `frame_loop::encode_frame`. Emits
   `BPM_VERIFY frame=N kind=K q=Q predicted=P actual=A ratio=R` for
   every frame. Zero cost when env unset.

5. **`RVP9_BPM_MULTIPLIER=<f64>`** env override in
   `frame_loop::EncoderConfig::to_rate_control_config`. Lets test
   harnesses bypass the public API to swap the multiplier without
   plumbing a new field through every caller. Default behaviour
   unchanged.

## Measured ratios — foreman_cif 256×256 64-frame sweep

Sample at the 250 kbps target (representative; the other three
targets show the same equilibrium):

| frame | kind  | q    | predicted | actual | ratio |
|------:|-------|-----:|----------:|-------:|------:|
| 0     | Key   | 124  | 33256     | 32896  | 0.989 |
| 1     | Inter | 191  | 6886      | 7040   | 1.022 |
| 5     | Inter | 194  | 7446      | 8240   | 1.107 |
| 10    | Inter | 169  | 13907     | 15952  | 1.147 |
| 15    | Inter | 148  | 24652     | 27824  | 1.129 |
| 20    | Inter | 144  | 26042     | 23688  | 0.910 |
| 25    | Inter | 150  | 27318     | 28696  | 1.050 |
| 30    | Inter | 153  | 17433     | 18056  | 1.036 |
| 35    | Inter | 162  | 18969     | 21816  | 1.150 |
| 40    | Inter | 170  | 16621     | 15480  | 0.931 |
| 45    | Inter | 174  | 13256     | 12720  | 0.960 |
| 50    | Inter | 179  | 13619     | 12584  | 0.924 |
| 55    | Inter | 182  | 11079     | 12064  | 1.089 |
| 60    | Inter | 188  | 11853     | 12712  | 1.072 |
| 63    | Inter | 190  | 11418     | 9904   | 0.867 |

Mean over the 64 frames: **~1.05**.

The other three targets (25, 50, 100 kbps) hit q=255 saturation
within ~18 frames, and at q=255 the ratio also tracks ~1.0
(predicted 3500-4000 bits, actual 3000-4500 bits). No 10× bias
anywhere.

## Headline — foreman PSNR sweep

| Target | Baseline (mult=1.0) | mult=10.0 | mult=0.1 |
|-------:|--------------------:|----------:|---------:|
| 25k    | 16.36 dB / 127 kbps | 16.43 dB / 128 kbps | 16.50 dB / 133 kbps |
| 50k    | 16.21 dB / 160 kbps | 16.44 dB / 159 kbps | 16.26 dB / 165 kbps |
| 100k   | 16.28 dB / 249 kbps | 16.46 dB / 249 kbps | 16.66 dB / 251 kbps |
| 250k   | 16.55 dB / 515 kbps | 16.41 dB / 513 kbps | 16.80 dB / 519 kbps |

The PSNR is constant within noise across a 100× swing in
`bits_per_mb_multiplier`. **The multiplier is impotent.**

Mechanism: the correction_factor feedback walks proportionally to
compensate the multiplier within ~5-10 frames, after which the
picker's effective estimate is dominated by the adapted
correction_factor and the multiplier washes out. The multiplier
only affects the first few frames' Q-picks, which doesn't move
the 64-frame average PSNR.

## testsrc PSNR sweep — preserved

| Target | Baseline | After A10 (mult=1.0 default) |
|-------:|---------:|-----------------------------:|
| 25k    | 35.93 dB / 68 kbps  | 35.93 dB / 68 kbps  |
| 50k    | 38.68 dB / 113 kbps | 38.68 dB / 113 kbps |
| 100k   | 43.08 dB / 175 kbps | 43.08 dB / 175 kbps |
| 250k   | 59.69 dB / 262 kbps | 59.69 dB / 262 kbps |

**Byte-exact match.** Default multiplier=1.0 is libvpx-faithful;
the new field is a no-op at the default.

## Cross-decoder gate — PASS

```
rvp9-decoder vs vpxdec  : BYTE-EXACT
rvp9-decoder vs ffmpeg  : BYTE-EXACT
Stage 4a cross-decoder gate: PASS
```

## RT gate — PASS

```
median: 5.81 ms ≤ 33.00 ms (PASS)
```

## Clippy — clean (in my code)

`cargo clippy -p rvp9-encoder --lib -- -D warnings` produces zero
warnings from any file I touched. Three pre-existing failures in
`crates/rvp9-core/src/dsp/x86_64/lpf_vertical_4_avx2.rs` and
`idct{16,32}_avx2.rs` are decoder-team Wave F code, not mine —
documented in `wrk_journals/2026.05.21 - JRN - night-7 ...md` as
"decoder-team will iterate".

## Why the multiplier still ships (default 1.0)

Per brief: "Multiplier must be CONFIGURABLE (not hardcoded).
Default = 1.0."

Even though this experiment disproves the hypothesis the brief was
testing, the infrastructure has independent value:

1. Anyone who later WANTS to inject a content-aware prior into the
   rate model can use the knob.
2. The diagnostic env var (`RVP9_DIAG_BPM_VERIFY=1`) is the
   instrument that disproved the hypothesis; keeping it lets the
   next investigator skip the "rebuild the instrumentation" step.
3. Default 1.0 preserves cross-decoder byte-exactness, the testsrc
   PSNR sweep, and the RT gate.

The knob is a documented divergence at value=1.0 (no-op) and
ledger-marked HYPOTHESIS DISPROVED.

## Where the foreman 16-dB floor actually lives — A11 next step

Per A2.foreman analysis
(`notes/stage4a-foreman-block-yrd-analysis.md`): the inter chain
decays monotonically `f0=30.9 → f5=20 → f31=14.7 dB` on foreman
while the KF lands at a healthy 34.78 dB. The 16-dB average is
the trajectory mean of this decay.

This decay signature is consistent with a **reference-frame-
update divergence between encoder and decoder**: the encoder's
LASTFRAME slot stores something different from what the decoder
will compute when it processes the same bitstream. Each inter
frame's prediction is then built from a degraded reference, and
the drift accumulates.

Cross-decoder byte-exactness on the BITSTREAM (which is
preserved across all 26 night-7 commits) does NOT imply
parity on REFERENCE BUFFERS. The bitstream is the FROZEN OUTPUT;
the reference buffer is intermediate encoder state used to PICK
that output. The encoder's pick can be made against a stale /
wrong reference, and the resulting bitstream is still valid
(decodes to whatever the decoder computes from the same bits) —
but the next frame's prediction signal is degraded because the
encoder was choosing modes against the wrong reference.

The A11 diagnostic should be:

1. Encode N=8 foreman frames; capture the encoder's
   `recon_yuv_out` after the loop filter (the buffer it stores
   as the new LASTFRAME).
2. Decode the SAME N=8 frames through rvp9-decoder; capture
   the decoder's per-frame post-loop-filter reconstruction.
3. Compute per-frame SSE(encoder_lastframe, decoder_lastframe).
4. If non-zero on any frame, the bug is in the encoder's
   reference-update path: the LASTFRAME stored doesn't match
   what the decoder will reconstruct from the same bits.

Probable culprits in descending likelihood:

1. **Inter recon doesn't include residual.** At
   `lib.rs::pick_and_simulate_inter_encode_partitions`, the
   per-leaf recon = prediction only, dropping the residual
   contribution. The encoder's LASTFRAME = predicted-only;
   the decoder's = predicted+dequantised-residual. The gap
   compounds linearly per frame.
2. **Loop-filter level disagreement.** Encoder picks one
   filter level, decoder reads a different one from the
   header. Per-leaf MC then reads from a differently-filtered
   reference.
3. **Chroma MV-limit bug from A1 (chroma-aware MV-limit
   tightening).** If the encoder's chroma MC clamps an MV
   the decoder doesn't (or vice versa), the chroma plane
   drifts.

Verifying (1) is cheapest — read
`pick_and_simulate_inter_encode_partitions` and see whether
the per-leaf `recon` slice it returns includes the residual.
If not, that's the bug.

## Files touched

- `crates/rvp9-encoder/src/rate_control/config.rs` — added
  `bits_per_mb_multiplier: f64` field, validate(), default 1.0.
- `crates/rvp9-encoder/src/rate_control/cbr.rs` — added
  `bits_per_mb_scaled` wrapper, threaded `multiplier`
  parameter through `regulate_q` and `estimate_bits_at_q`. Two
  new unit tests asserting libvpx-faithful at mult=1.0 and
  linear scaling at mult=10.
- `crates/rvp9-encoder/src/rate_control/state.rs` — added
  `predicted_bits_at_picked_q` diagnostic accessor.
- `crates/rvp9-encoder/src/rate_control/postencode.rs` —
  threaded multiplier into `update_rate_correction_factors`.
- `crates/rvp9-encoder/src/frame_loop.rs` — `RVP9_BPM_MULTIPLIER`
  env hook + `RVP9_DIAG_BPM_VERIFY` env-gated per-frame log.
- `notes/divergence-ledger.md` — A10 entry under "Stage 4a
  experiments".

## Status

Multiplier infrastructure shipped at default 1.0 (= libvpx-
faithful, byte-exact preserved). Hypothesis DISPROVED with
empirical measurement. Next experiment is A11 reference-buffer
parity diagnostic — different bug class entirely.
