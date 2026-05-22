# A19 — loop-filter picker level sweep — DISCARD (no foreman win)

Date: 2026-05-22.
Worktree: `D:\language\rvp9-2\.claude\worktrees\agent-a6d67086009c6cfd8`.

## TL;DR

Tested 3 alternate `pick_filter_level` formulas against the baseline
`(q_index >> 1).min(63)` on foreman_cif (256×256 centre crop, 64
frames) and testsrc (256×256, 64 frames). **No formula improves
foreman BD-rate by the ≥1% bar.** The picker has no leverage on
foreman in the rate-control regime that the bdrate_gate harness
operates in: 133 of 256 measured frames have q ≥ 220, where every
candidate formula caps at `MAX_LOOP_FILTER = 63` anyway.

Verdict: **discard**. Loop-filter picker tuning is impotent on
foreman at the current rate-control operating point. The night-7
journal's a priori estimate of "<0.5% gain from real
`vp9_pick_filter_level`" stands confirmed.

No code changes committed. The picker is reverted to baseline.

## Hypothesis

`crates/rvp9-encoder/src/loopfilter.rs:75 pick_filter_level` uses a
coarse libvpx-divergent approximation `filter_level = (q_index >>
1).min(MAX_LOOP_FILTER)`. libvpx's `vp9_pick_filter_level`
(`vp9_picklpf.c:191-294`) does a quality-driven SSE search around the
previous frame's filter_level. The doc comment in
`loopfilter.rs:34-45` flags this as a divergence to re-evaluate. If
the picker is over- or under-filtering foreman, that costs PSNR
directly.

## Method

Added an env-gated formula switch to `pick_filter_level` so a single
release build could be swept across formulas without rebuilding:

```rust
match std::env::var("RVP9_LPF_FORMULA").ok().as_deref() {
    Some("1") => /* weaker:  (q * 7 / 16).min(63) */,
    Some("2") => /* stronger:(q * 9 / 16).min(63) */,
    Some("3") => /* piecewise libvpx-table proxy */,
    _         => /* baseline:(q >> 1).min(63)     */,
}
```

The env-knob was removed after measurement (no formula won — see
results below). The file is byte-equal to the pre-A19 baseline.

Test surface:
- `cargo test -p rvp9-encoder --test bdrate_gate --release -- --ignored --nocapture`
- Run both `bdrate_gate_stage4a` (testsrc) and
  `bdrate_gate_stage4a_realcontent` (foreman).
- Each run encodes at 4 target bitrates (25/50/100/250 kbps), 64
  frames, computes Y-PSNR vs source and BD-rate vs libvpx-stock
  curve.

## Results

### Foreman_cif (256×256, 64 frames)

| Formula           | 25k PSNR | 50k PSNR | 100k PSNR | 250k PSNR | BD-rate vs libvpx |
|-------------------|---------:|---------:|----------:|----------:|------------------:|
| Baseline (q>>1)   |  24.01   |  24.16   |   25.42   |   28.96   |       +395.92 %   |
| C1 weaker (7/16)  |  24.01   |  24.16   |   25.42   |   28.96   |       +398.22 %   |
| C2 stronger (9/16)|  24.01   |  24.16   |   25.42   |   28.96   |       +396.47 %   |
| C3 piecewise      |  24.01   |  24.16   |   25.42   |   28.96   |       +395.92 %   |

PSNR is identical to 2 decimal places at all four targets across all
four formulas. BD-rate moves by at most ±2.3 pp; **none reaches the
≥1% improvement bar**. C1 is mildly *worse* than baseline. C3 is
indistinguishable from baseline.

### testsrc (256×256, 64 frames)

| Formula           | 25k PSNR | 50k PSNR | 100k PSNR | 250k PSNR | BD-rate vs libvpx |
|-------------------|---------:|---------:|----------:|----------:|------------------:|
| Baseline (q>>1)   |  33.51   |  42.92   |   48.28   |   56.86   |       +630.68 %   |
| C1 weaker (7/16)  |  33.51   |  42.92   |   48.32   |   57.59   |       +626.61 %   |
| C2 stronger (9/16)|  33.51   |  42.92   |   48.29   |   56.94   |       +628.82 %   |
| C3 piecewise      |  33.51   |  42.92   |   50.03   |   57.00   |       +576.94 %   |

C3 shows a sizable testsrc BD-rate gain (-53.74 pp) and a +1.75 dB
PSNR jump at 100k. **But foreman is unmoved** — so the C3 win is
testsrc-only and the picker tuning lever doesn't transfer to real
content.

## Why is foreman unmoved? — Q saturation

Captured the per-frame q distribution with `RVP9_DIAG_BPM_VERIFY=1`
on the foreman baseline run (256 inter+key frames across 4
targets):

| Q bucket        | Frames |
|-----------------|-------:|
| q ≤ 80          |   0    |
| 80 < q ≤ 130    |   1    |
| 130 < q ≤ 180   |  54    |
| 180 < q < 220   |  68    |
| 220 ≤ q ≤ 255   | 133    |

At **q ≥ 126** every candidate formula caps at
`MAX_LOOP_FILTER = 63`:
- baseline: q=126→63, q=255→63
- weaker (7/16): q=144→63, q=255→63
- stronger (9/16): q=112→63, q=255→63
- piecewise: q ∈ [161,255] → q*4/9 → 71+ → cap 63; q ∈ [81,160] →
  q/2 → 40-80 → may cap at high end

133/256 frames (52%) are in pure cap regime where all formulas are
byte-identical in the bitstream. The only zone with cross-formula
variance is q ∈ [130,180] (54/256 = 21%), and the BD-rate movement
above shows that variance is in the PSNR noise.

The rate controller pushing Q to the cap on foreman is the
underlying bug — not the loop-filter picker. A19's lever is on the
wrong side of the saturation knee.

## Sanity check — encoder/decoder parity

The 94 encoder lib unit tests (including the F11.0
`f11_0_lf_at_q_zero_is_zero` and cross-decoder byte-exact gates) all
pass with the env var unset; with `RVP9_LPF_FORMULA` set the same
formula is used on both the encoder's reference-reconstruction LF
pass and the decoded bitstream's LF pass (the `filter_level` is
written into the uncompressed header and the decoder reads it), so
byte-equality is preserved by construction — the picker formula
flows through the header, not the kernels.

## Conclusion

Loop-filter picker tuning provides ≤2.3 pp of BD-rate movement on
foreman and no PSNR movement at all in the working-range targets
(25-100 kbps). The picker is a real divergence from libvpx but it
is not a meaningful lever on the foreman gap at this rate-control
operating point.

This corroborates the night-7 close-out note that "real
`vp9_pick_filter_level` < 0.5%" of the +395% gap.

A19 is the 19th and last credible bench item in the night-7 Track A
sweep and the result is an empirical disprove of the hypothesis.

### Recommendation

**Discard.** Do not implement real `vp9_pick_filter_level` for the
foreman gap. The foreman BD-rate gap is dominated by the rate
controller saturating Q at the cap, not by loop-filter mistuning.
The team-lead should close /goal as documented.

The only environment in which the picker has measurable leverage is
testsrc at 100k, where C3 (piecewise) gains +1.75 dB. testsrc is
already known to plateau pathologically (libvpx hits 78.8 dB at
68 kbps on it; see F12.0 journal) so this is unlikely to be a
real-content signal.

## File pointers

- Picker (reverted to baseline):
  `D:\language\rvp9-2\.claude\worktrees\agent-a6d67086009c6cfd8\crates\rvp9-encoder\src\loopfilter.rs:75`
- Test harness:
  `D:\language\rvp9-2\.claude\worktrees\agent-a6d67086009c6cfd8\crates\rvp9-encoder\tests\bdrate_gate.rs:591`
  (`bdrate_gate_stage4a_realcontent`)
- Q-saturation diagnostic source:
  `D:\language\rvp9-2\.claude\worktrees\agent-a6d67086009c6cfd8\crates\rvp9-encoder\src\frame_loop.rs:520`
  (`RVP9_DIAG_BPM_VERIFY` env knob)
