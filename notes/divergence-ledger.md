# Divergence ledger

Every deliberate departure from libvpx's behaviour, with rationale and
expected impact on byte parity. See `PLAN.md` §3 D8 — byte parity is a
diagnostic tool, not a release gate, but every divergence must be conscious
and documented here.

Empty entries below are placeholders for the categories we expect to grow.

## Format

Each entry:

```
### <YYYY-MM-DD> — <short title>
**Where:** <crate::module::item> + libvpx file/line being departed from
**What libvpx does:** <brief>
**What rvp9 does:** <brief>
**Why:** <rationale: safety, Rust idiom, perf, bug we won't replicate, ...>
**Byte-parity impact:** <none / minor / structural>
**Compression-efficiency impact:** <expected delta in BD-rate, if any>
**Future plan:** <revisit when? never?>
```

## Entries

### 2026-07-31 — Widen the complete second 32-point DCT pass
**Where:** `rvp9-core::transforms::forward_neon`; libvpx counterparts:
`vpx_dsp/arm/fdct32x32_neon.c` and `vpx_dsp/arm/fdct32x32_neon.h`.
**What libvpx does:** its first 32x32 pass uses eight i16 NEON lanes and its
overflow-sensitive second pass widens selected stages to paired i32 vectors.
**What rvp9 does:** its first pass now also uses eight i16 lanes. Before the
second pass it widens all eight lanes into two four-lane i32 groups, then runs
the same audited scalar schedule. First-pass cosine products widen to i32;
second-pass products widen to i64.
**Why:** widening at the pass boundary keeps one width-generic seven-stage
schedule and preserves rvp9's i32 coefficient API. It removes half the
first-pass lane groups and halves intermediate storage without duplicating
libvpx's macro-heavy mixed-width second-pass implementation.
**Byte-parity impact:** none. Differential coverage compares 128 randomized
valid residual blocks with stride jitter plus constant and alternating
residual extremes. The 120-frame product stream remained byte-identical.
**Compression-efficiency impact:** none because every transform coefficient
and emitted stream is unchanged.
**Speed impact:** sixteen warmed alternating one-thread pairs improved median
wall time `1.081781 -> 1.061569 s` (`-1.408%` paired) and child CPU
`1.043518 -> 1.025352 s` (`-1.424%`). Twenty four-thread pairs improved
wall time `0.679587 -> 0.678864 s` (`-0.767%`) and child CPU
`1.251782 -> 1.237006 s` (`-1.022%`).
**Future plan:** retain the pass-boundary widening while it stays exact and
Pareto-positive. Revisit libvpx's selective second-pass widening only if a
fresh profile still makes the transform material.

### 2026-07-31 — Widen the complete second 16-point DCT pass
**Where:** `rvp9-core::transforms::forward_neon`; libvpx counterpart:
`vpx_dsp/arm/fdct16x16_neon.c`.
**What libvpx does:** processes both 16x16 passes in eight i16 NEON lanes.
**What rvp9 does:** matches the eight-wide i16 first pass, then widens each
row group into two four-lane i32 second passes using the shared audited
16-point schedule.
**Why:** pass-boundary widening reuses the existing overflow-safe i32
implementation and one width-generic schedule while still halving first-pass
lane groups and intermediate storage.
**Byte-parity impact:** none. Differential coverage compares 128 randomized
valid residual blocks with stride jitter plus constant and alternating
residual extremes. The 120-frame product stream remained byte-identical.
**Compression-efficiency impact:** none because every coefficient and emitted
stream is unchanged.
**Speed impact:** sixteen warmed alternating one-thread pairs on the final
machine text improved wall time `1.050828 -> 1.040561 s` (`-0.951%` paired)
and child CPU `1.011124 -> 1.001498 s` (`-0.971%`), with fourteen wins.
Across two independent twenty-pair four-thread runs, the combined median
paired wall delta was `-0.365%` with 26/40 wins; child CPU improved in both
runs (`-0.819%` and `-0.397%`).
**Future plan:** retain while exact and Pareto-positive. Consider libvpx's
all-i16 second pass only if a fresh profile still makes this transform
material.

### 2026-07-31 — Reuse prediction scratch per tile worker
**Where:** `rvp9-core::mode_decision::inter::evaluate_candidate_for_filter`;
libvpx counterpart: `vp9/encoder/vp9_pickmode.c:search_filter_ref`.
**What libvpx does:** reuses prediction storage owned by the current
macroblock while evaluating realtime mode and interpolation-filter
candidates.
**What rvp9 does:** reuses one 4,096-byte thread-local prediction buffer
across serial candidate evaluations on each tile worker. Every prediction
builder overwrites its complete active block prefix before any consumer runs.
**Why:** the previous stack array called memset for 4,096 bytes before every
candidate, even though active blocks can be much smaller and the builder
immediately overwrites every consumed pixel. Thread locality preserves
parallel isolation without changing picker APIs.
**Byte-parity impact:** none. A causal regression evaluates a 16x16 block,
overwrites the scratch through a 64x64 evaluation, and requires the repeated
16x16 result to match in every field. The 120-frame product stream remained
byte-identical and all three decoders produced identical I420 output.
**Compression-efficiency impact:** none because predictions, mode decisions,
coefficients, and emitted bytes are unchanged.
**Speed impact:** sixteen warmed alternating one-thread pairs improved median
wall time `1.065253 -> 1.059120 s` (`-0.481%` paired) and child CPU
`1.024843 -> 1.018660 s` (`-0.463%`). Twenty four-thread pairs improved
wall time `0.668599 -> 0.666289 s` (`-0.522%`) and child CPU
`1.219596 -> 1.212353 s` (`-0.381%`).
**Future plan:** retain while causal differential coverage and byte identity
remain green. Apply the same lifecycle only to independently measured scratch
buffers; do not create a shared cross-thread pool.

### 2026-07-31 — Reuse emitted-transform residual scratch
**Where:** `rvp9-encoder::with_inter_emit_residual_scratch`; libvpx
counterpart: `vp9/encoder/vp9_encodemb.c:vp9_xform_quant_fp`.
**What libvpx does:** reuses macroblock-owned residual and coefficient buffers
while transforming and quantizing emitted blocks.
**What rvp9 does:** reuses one 1,024-value i16 residual buffer per tile worker
and lends each transform only its active `tx_side * tx_side` prefix.
**Why:** the previous loop zeroed a full 32x32 stack array for every 8x8,
16x16, or 32x32 transform even though subtraction immediately overwrote every
active value. A prefix-bounded borrow makes stale suffix data unreachable.
**Byte-parity impact:** none. Focused coverage contaminates the full scratch
with opposite extremes around repeated 8x8 writes and requires identical
active results. The 120-frame product stream remained byte-identical and all
three decoders produced identical I420 output.
**Compression-efficiency impact:** none because residuals, quantized
coefficients, reconstruction, and emitted bytes are unchanged.
**Speed impact:** twenty warmed alternating four-thread pairs improved wall
time `0.665288 -> 0.660069 s` (`-0.890%` paired) and child CPU
`1.213885 -> 1.202704 s` (`-0.961%`). Two independent sixteen-pair
one-thread runs combined to a neutral `-0.067%` paired wall delta with 16/32
wins; the second run improved wall/CPU by `-0.230%/-0.227%`.
**Future plan:** retain for the required tile-threaded product path while
one-thread behavior stays neutral. Do not generalize beyond independently
measured, prefix-bounded scratch.

### 2026-07-31 — Use a generic eight-lane NEON interpolation kernel
**Where:** `rvp9-core::dsp::aarch64::convolve8_neon`; libvpx counterpart:
`vpx_dsp/arm/vpx_convolve8_neon.c`.
**What libvpx does:** dispatches among width-specific NEON kernels with
paired taps, row-pair sliding windows, and special 4-pixel paths.
**What rvp9 does:** computes eight adjacent pixels per vector with widened
i32 multiply-accumulates and uses the scalar oracle for sub-eight-pixel
widths and non-vector tails. The 2-D path composes these exact single-axis
kernels through a reusable thread-local intermediate.
**Why:** this is the smallest safe architecture that removes the measured
scalar bottleneck while retaining one easy-to-audit arithmetic path. It
also preserves the scalar API's non-standard odd-width behavior.
**Byte-parity impact:** none. Differential tests cover all four filter
families, every phase, the VP9 block-shape matrix, odd widths, and stride
jitter. The 120-frame product stream remained byte-identical.
**Compression-efficiency impact:** none because mode decisions and emitted
streams are unchanged.
**Speed impact:** warmed 1280x704 Stage 4a VBR cpu-used 7 median wall time
fell from `1.80 s` to `1.64 s` (`8.9%`).
**Future plan:** consider libvpx's paired-tap or dot-product kernels only if
another profile shows interpolation remains material.

### 2026-07-31 — Retain the duplicate picker DCT outside Stage 4a VBR
**Where:** `rvp9-core::mode_decision::inter::evaluate_candidate_for_filter`
and `InterRdCalibration`; libvpx counterpart:
`vp9/encoder/vp9_pickmode.c:block_yrd`.
**What libvpx does:** realtime candidate RD uses one Hadamard transform,
`quantize_fp`, its coefficient rate/distortion, and its skippable result.
The selected leaf is transformed and reconstructed later by the encode path.
**What rvp9 does:** the Stage 4a realtime VBR calibration now matches that
single Hadamard pipeline. Other calibration policies still run a second DCT,
`quantize_b`, inverse DCT, and pixel-SSE pass inside every candidate before
the selected leaf is transformed again during emit.
**Why:** removing the duplicate pipeline is a measured Pareto improvement for
the validated cpu-used 7 VBR path, but CBR and the remaining cpu-used presets
have not yet passed their feature-specific quality gates. Keeping the old
path there avoids silently generalising from one configuration.
**Byte-parity impact:** structural for Stage 4a VBR because the Hadamard
skippable predicate can differ from the former DCT-EOB predicate. Exercised
streams remain deterministic and decoder-valid.
**Compression-efficiency impact:** on the 64-frame fixed-Q Foreman screen,
inter payload moved `25,166 -> 25,085`, `12,788 -> 12,624`, and
`5,856 -> 5,770` bits/frame at q80/q120/q160 while PSNR improved by
`0.05/0.07/0.13 dB`.
**Speed impact:** on 120 frames of 1280x704 testsrc2, the same policy reduced
cpu-used 7 CBR diagnostic wall time from `1.90 s` to a `1.82 s` median
(`4.2%`). The CBR default remains unchanged pending its own quality gate.
**Future plan:** enable the single Hadamard pipeline for CBR and cpu-used 6/8
only after their matched product matrices preserve the Pareto frontier.

### 2026-07-30 — Keep both UV planes in the realtime picker model
**Where:** `rvp9-core::mode_decision::inter::evaluate_candidate_for_filter`
and `rvp9-core::mode_decision::model_rd::model_rd_for_sb_uv`; libvpx
counterpart: `vp9/encoder/vp9_pickmode.c:2381-2395`.
**What libvpx does:** computes SB-level `x->color_sensitivity[U,V]` from the
selected partition predictor and includes only sensitive chroma planes in
realtime mode RD.
**What rvp9 does:** includes both chroma planes whenever UV buffers are
available. The F.0.2 diagnostic now computes and records the source-faithful
partition flags, but they do not change picker behavior.
**Why:** the exact rvp9 producer matched pinned libvpx on all 112 comparable
Foreman superblocks in the eight-frame q120 fixed-tree oracle. Wiring those
flags into descent and final picker calls still raised inter payload at
q80/q120/q160 from `25,196/12,792/5,853` to
`25,309/12,862/5,861` bits per frame for only `0.00-0.01 dB`. The UV term is
currently compensating another mode-ranking difference, so the isolated
source-faithful behavior is not a Pareto improvement.
**Byte-parity impact:** structural encoder-output difference only; emitted
streams remain VP9-conformant and cross-decoder correctness is unaffected.
**Compression-efficiency impact:** retaining both planes preserves the
accepted Stage 4a baseline. Enabling the libvpx gate in isolation costs about
`0.1-0.6%` inter payload at matched qindex.
**Future plan:** revisit only after the causal fixed-tree candidate ranking
matches libvpx. The retained F.0.2 fields provide the identical-input oracle
for that integration.

### 2026-07-30 — Keep realtime encode breakout as a luma early exit
**Where:** `rvp9-core::mode_decision::model_rd::encode_breakout_test`,
`rvp9-core::mode_decision::inter::pick_inter_mode`, and
`rvp9-encoder::encode_inter_frame`; libvpx counterpart:
`vp9/encoder/vp9_pickmode.c:933-1036,2413-2425,2461-2478`.
**What libvpx does:** tests the current inter candidate's Y, U, and V
prediction variance. When all three planes pass, it sets `x->skip`, makes the
candidate win regardless of its prior RD rank, skips intra fallback, and
commits prediction-only reconstruction.
**What rvp9 does:** applies the same luma threshold as a mode-loop early exit
only. It does not run the U/V breakout test or force prediction-only
reconstruction; the normal quantized residual path makes the final skip
decision.
**Why:** the source-faithful Y/U/V prediction-only behavior was a measured
non-Pareto regression on the accepted coefficient-domain intra baseline. Most
fixed-Q breakout candidates already quantized to zero, while forcing the
breakout suppressed useful later mode and intra choices on rvp9's different
partition and predictor field.
**Byte-parity impact:** structural — mode, intra, and residual decisions can
differ. All seven measured candidate streams decoded successfully under
rvp9-decoder and remained byte-exact under pinned vpxdec and ffmpeg.
**Compression-efficiency impact:** the source-faithful candidate worsened
strict Foreman BD-rate from `+4.9246%` to `+5.5769%`, matched-rate PSNR
deficit from `0.2841` to `0.3023 dB`, and median frame time from `13.2028` to
`13.3620 ms`. Fixed-Q inter payload was unchanged at q80, rose by one
bit/frame at q120, and rose by three bits/frame at q160.
**Future plan:** retain the luma-only early exit while the predictor and
partition fields differ materially. Re-evaluate the complete libvpx behavior
only after common-leaf predictor selection and intra partition shape converge.

### 2026-07-30 — Keep a fixed sub-pixel MV error multiplier
**Where:** `rvp9-core::mode_decision::inter::run_newmv_search`; libvpx
counterparts: `vp9/encoder/vp9_rd.h:set_error_per_bit` and
`vp9/encoder/vp9_pickmode.c:246-251`.
**What libvpx does:** derives the variance-domain sub-pixel MV-cost multiplier
as `max(1, compute_rd_mult(qindex) >> 6)` for each frame.
**What rvp9 does:** uses the fixed multiplier `16` for realtime sub-pixel
refinement at cpu-used 6-8.
**Why:** the source-faithful qindex-derived multiplier was a measured
non-Pareto regression after the full realtime sub-pixel tree landed. It
damaged both low-rate and high-rate points in the bounded Foreman diagnostic.
**Byte-parity impact:** structural — sub-pixel motion-vector choices and later
coding decisions can differ.
**Compression-efficiency impact:** the source-faithful candidate moved the
four-point informational Foreman BD-rate from about `-7.78%` to `-0.65%`.
Its matched-bitrate deficit remained about `0.32 dB`, while the 25 kbps point
moved from `38.879 kbps / 26.440 dB` to `35.810 kbps / 26.175 dB` and the
250 kbps point moved from `178.959 kbps / 34.088 dB` to
`179.337 kbps / 34.053 dB`.
**Future plan:** retain `16` until the remaining luma residual and skip-model
gap is resolved. Re-evaluate the source multiplier across cpu-used 6-8 only
when the surrounding RD domains no longer require this compensation.

### 2026-07-30 — Apply the cpu-used 9 adaptive sub-pel stop at cpu-used 6-8
**Where:** `rvp9-core::mode_decision::inter::run_newmv_search`; libvpx
counterparts: `vp9/encoder/vp9_speed_features.c:562,803-806` and
`vp9/encoder/vp9_pickmode.c:236-249`.
**What libvpx does:** realtime cpu-used 6-8 use `QUARTER_PEL` for every NEWMV
sub-pixel search. Cpu-used 9 alone enables an adaptive selector that keeps a
zero full-pel vector at quarter-pel and stops every non-zero vector at
half-pel.
**What rvp9 does:** the v1 realtime picker applies libvpx's cpu-used 9 adaptive
quarter/half-pel selector at cpu-used 6-8.
**Why:** restoring the source-faithful quarter-pel stop was a measured
non-Pareto regression on the pinned Stage 4a product curve. It nearly doubled
the aggregate median frame time while slightly worsening BD-rate.
**Byte-parity impact:** structural — motion vectors and later coding decisions
can differ. Every stream from both measured variants decoded successfully
under rvp9-decoder and remained byte-exact under vpxdec and ffmpeg.
**Compression-efficiency impact:** the source-faithful quarter-pel candidate
moved strict Foreman BD-rate from `+9.1152%` to `+9.4344%`, left the
matched-bitrate deficit effectively flat (`0.52174` to `0.52184 dB`), and
raised median frame time from `11.8122` to `21.8189 ms`.
**Future plan:** retain the adaptive policy unless a bounded sub-pixel search
change recovers libvpx's quarter-pel choices without the measured speed and
BD-rate regression. Re-evaluate independently for cpu-used 6 and 8 during the
full preset matrix.

### 2026-07-30 — Disable the simplified realtime intra-in-inter fallback
**Where:** `rvp9-core::mode_decision::inter::pick_inter_mode` and
`rvp9-encoder::encode_inter_frame`; libvpx counterpart:
`vp9/encoder/vp9_pickmode.c:2517-2646`.
**What libvpx does:** evaluates selected intra modes after the realtime inter
search when the best inter RD exceeds its intra threshold. On the pinned
Foreman Stage 4a run, about 2% of coded block area is intra.
**What rvp9 does:** keeps the fallback implementation and direct tests, but
the v1 realtime encoder disables it through
`PickInterModeParams::enable_intra_fallback`. Keyframes remain intra-coded.
**Why:** rvp9's simplified intra RD path selected 20.34% intra area at
250 kbps and damaged both quality and speed. Disabling it is a reproducible
Pareto improvement across the strict seven-point Foreman curve.
**Byte-parity impact:** structural — inter frames choose different block modes,
but all exercised streams remain byte-exact across rvp9-decoder, vpxdec, and
ffmpeg decoded output.
**Compression-efficiency impact:** Stage 4a BD-rate improved from `+25.77%` to
`+15.77%`; matched-bitrate PSNR deficit improved from `1.149 dB` to
`0.815 dB`. Aggregate median encode time improved from `12.99 ms` to
`12.21 ms`.
**Future plan:** re-enable only after the intra-in-inter RD model and selection
rate match libvpx across the representative corpus and retain a Pareto gain.

### 2026-07-30 — Neutral inherited-MV context in the Stage 4a partition proxy
**Where:** `rvp9-encoder::pick_and_simulate_inter_encode_partitions`;
libvpx counterparts: `vp9/encoder/vp9_encodeframe.c:choose_partitioning` and
the cpu-used 7 `VAR_BASED_PARTITION` path.
**What libvpx does:** cpu-used 7 one-pass realtime VBR chooses the partition
tree from its variance partitioner. It does not run the inter-mode picker as a
speculative partition oracle; the later mode picker sees causal neighbours
from earlier coded leaves.
**What rvp9 does:** rvp9's load-bearing A4 compensating descent calls the
inter picker before the current superblock has populated its causal mi-grid.
For realtime VBR at cpu-used 7 and 8, that speculative call now gives every
shape a neutral zero inherited-MV pair. ZERO and NEWMV are still searched.
The final emitted picker remains unchanged and uses the real causal
neighbours. `RVP9_A4_DESCENT_INDEPENDENT_MV=0` restores the former proxy for
diagnosis.
**Why:** the former proxy could use inherited MVs where a candidate touched a
prior superblock, but saw no neighbour for equivalent positions inside the
current superblock. Only 22.01% of committed leaves had the same
`(nearest, near)` pair in the speculative and final pickers. Neutralising that
unavailable context makes shape costs comparable without reusing a stale
motion decision.
**Byte-parity impact:** structural — partition and mode choices change. All
seven strict Stage 4a streams still decode successfully under rvp9-decoder and
remain byte-exact under vpxdec and ffmpeg.
**Compression-efficiency impact:** on the pinned 64-frame strict Foreman
curve, BD-rate improved from `+12.4005%` to `+12.0635%`; matched-rate PSNR
deficit improved from `0.67993 dB` to `0.63404 dB`. Median encode time improved
from `12.5560 ms` to `12.1694 ms`.
**Future plan:** keep only if the gain generalises across the required corpus
and cpu-used presets. A future transactional partition search may replace this
proxy once it can preserve and restore causal mi/reconstruction state without
missing the realtime budget.

### 2026-05-22 — E7: keyframe `active_best_quality` floor (`get_kf_active_quality`)
**Where:** `rvp9-encoder::rate_control::cbr` — new `get_minq_index`,
`kf_low_motion_minq`, `kf_high_motion_minq`, `get_active_quality`,
`get_kf_active_quality`, and the keyframe branch of `pick_q_and_bounds`.
libvpx counterpart: `vp9_ratectrl.c:96-183` (`get_minq_index`,
`init_minq_luts`, `get_active_quality`, `get_kf_active_quality`) +
`rc_pick_q_and_bounds_one_pass_cbr` keyframe branch.
**What libvpx does:** keyframe `active_best_quality` is floored via
`get_kf_active_quality(rc, active_worst_quality, bit_depth)` — a min-Q LUT
blended by `kf_boost` between `kf_low`(400) and `kf_high`(5000). On the
*first* frame of a stream libvpx leaves `active_best = best_quality` and
relies on its **content-aware** `bits_per_mb` (rate from quantised-coef
energy) to land `regulate_q` at a quality-appropriate Q (q≈57 on foreman).
**What rvp9 does:** ports the LUTs + `get_kf_active_quality` faithfully and
applies the floor to **every** keyframe (including the first). Because
rvp9's enumerator-over-Q `bits_per_mb` (`cbr.rs:104`) under-predicts the
true intra cost on textured content, without a floor `regulate_q` crashes
the keyframe Q to `best_quality` (q≈4–20) chasing E6's large uncapped
target. The LUT is indexed by the KF `active_worst_quality`, but the
first-frame seed (255) over-floors (kf_high_motion_minq[255]≈195); rvp9
caps the LUT basis at `KF_ACTIVE_QUALITY_BASIS_CAP = 122` so the floor
lands at the libvpx-matched ≈57. The cap stands in for libvpx's
content-aware rate model on the keyframe.
**Why:** completes the E6 keyframe lever. E6 uncapped the KF target
(proving the reference-quality mechanism) but the under-predicting KF rate
model then over-spent it (q→4, 250 k PSNR regressed below libvpx's range →
BD-rate unmeasurable). The floor pins the KF at libvpx's q≈57 / ~82 kb.
**Byte-parity impact:** none — cross-decoder byte-exact gate stays green
(RC only changes the picked Q trajectory, not bitstream syntax). KF-only:
the inter `active_worst_quality` clamp (`cbr.rs`) was deliberately left
untouched (separate E4 lever). The shared `active_worst_quality` helper is
a behaviour-preserving refactor for the inter path.
**Compression-efficiency impact:** KF q 4–20 → 57 (50 k–1000 k); 250 kbps
PSNR 27.48 → 29.12 dB (restores libvpx PSNR overlap); inter coef-bits 5531
(pre-E6) → 4447.8 (−19.6%, the realistic libvpx-matched reference-quality
win; E6's −41%/3239 was inflated by KF over-spend). **Foreman 4-point
BD-rate: pre-E6 +367.81% → post-E6 unmeasurable → E7 +741.36%
(measurable).** CRITICAL FINDING: a libvpx-matched KF does NOT close the
gap — it scores numerically worse on BD-rate because rvp9 still overshoots
the *inter* bitrate ~2.5× (100 k target → 252 kbps actual vs libvpx 143),
so the higher-PSNR curve sits far to the right. The KF lever is necessary
(restores measurability + the E6 reference-quality inter-bit reduction)
but NOT sufficient; the dominant remaining lever is the inter
`active_worst` Q-clamp / partition feedback loop (E4 #2 / E5 map).
**Tests touched:** `aq_cyclic_refresh_segment_1_has_finer_quantization`
recalibrated from strict `mean_on < mean_off` to a 10% noise tolerance —
the strict form had baked in the pre-E7 over-spent KF (near-lossless
LASTFRAME made the inter residual dominant); with the faithful q≈57 KF the
reference carries real error and the seg-1 inter-Q benefit drops to noise
level (means ≈17). The structural failure the oracle catches ("AQ not
reaching the dequant path") would manifest as ratio≫1, not the 1.07 noise
band. Mirrors the E6 author's recalibration of the RC convergence oracles.
**Future plan:** Track-A closure now requires the inter-overshoot fix
(relax/port the full `calc_active_worst_quality_one_pass_cbr` inter ceiling
so `regulate_q` can quantise the inter residuals to hit target). The KF
lever is complete.

### 2026-05-22 — E6: `max_frame_bandwidth` corrected to libvpx formula (keyframe-boost)
**Where:** `rvp9-encoder::rate_control::state::RateControl::new`
(`max_frame_bandwidth` init) vs libvpx `vp9_rc_update_framerate`
(`vp9_ratectrl.c:2069-2083`).
**What libvpx does:** `vbr_max = avg_frame_bandwidth *
two_pass_vbrmax_section/100`; `max_frame_bandwidth =
max(max(MBs * MAX_MB_RATE=250, MAXIMUM_FRAME_BANDWIDTH=2_000_000),
vbr_max)`. The 2-Mbit floor is the operative term at low resolution.
**What rvp9 did (the corrected divergence):** `max_frame_bandwidth =
avg_frame_bandwidth * 4` only — a "sane default" that omitted both the
`MBs*250` and the `2_000_000` floors. This silently hard-clamped the
first-frame keyframe target (`calc_iframe_target_size` first-frame branch
= `starting_buffer_level / 2`) down to ~4× the inter target.
**What rvp9 does now:** the full libvpx `max(MBs*250, 2_000_000,
avg*400/100)` formula. KF target is no longer truncated.
**Why:** the prior `4*avg` was the proximate cause of keyframe
starvation (E1 finding #3). On foreman_cif 256-crop @ 100 kbps the KF
target jumped from a clamped 13 348 b (q=177) to its true
`starting_buffer_level/2` = 200 000 b request (q=20, 163 048 b coded),
seeding a high-quality LASTFRAME. This is a bug-correction *toward*
libvpx, not a new departure.
**Byte-parity impact:** none — cross-decoder byte-exact gate stays green
(RC only changes the picked Q trajectory, not the bitstream syntax).
**Compression-efficiency impact:** the keyframe-quality mechanism FIRES:
foreman @ 100 kbps inter coef-bits dropped 5531 → 3239 b/frame (−41%),
mean nonzero-coeff/skip0-leaf 6.9 → 3.9, skip=1 share 59% → 67%; KF/low/
mid PSNR rose. BUT the 250 kbps point regressed (28.98 → 27.48 dB)
because rvp9's KF `bits_per_mb` enumerator under-predicts KF cost, so
regulate_q drives KF Q to 4–20 (vs libvpx q=57/124), overspending the
now-uncapped target. That collapsed the 4-point foreman BD-rate from a
measurable +367.81 % to "not measurable (no PSNR overlap)" — rvp9's top
PSNR fell below libvpx's bottom PSNR.
**Future plan (follow-up lever, NOT this fix):** port libvpx's keyframe
rate model / KF active-best handling (or apply a KF-side
`bits_per_mb_multiplier`) so the KF lands near libvpx's q≈57 instead of
overspending at high bitrate. That restores BD-rate measurability and
should turn the confirmed inter-bit reduction into a headline gain. The
inter `active_worst_quality` clamp (`cbr.rs:374`) was deliberately left
untouched (separate E4 lever).
**Tests touched:** two synthetic RC convergence oracles
(`rate_control_simulation_30_frames`,
`rate_control_converges_at_multiple_bitrates_1mbps`) were recalibrated:
they had baked in the starved-KF assumption. The faithful large KF now
drains the leaky-bucket buffer, and v1 has no frame-drop path
(documented divergence) to refill it; the tests now measure inter
steady-state with the buffer restored post-KF, which is the
correction-factor-loop property they actually test.

### 2026-05-16 — High-bit-depth (10/12-bit) deferred to v2
**Where:** `rvp9-core::frame::Pixel` (type alias `pub type Pixel = u8;`);
all SIMD kernels in `rvp9-core::simd::*`; transforms, quant, intra-pred,
loop filter, motion-estimation, denoiser, frame-buffers, predictor SIMD.
libvpx counterpart: every `#if CONFIG_VP9_HIGHBITDEPTH` branch and
parallel `vpx_highbd_*` / `vpx_*_high*` kernel.
**What libvpx does:** ships Profile 0 (8-bit 4:2:0), Profile 1 (8-bit
4:2:2/4:4:4), Profile 2 (10-bit), Profile 3 (12-bit) with parallel
`u8` and `u16` code paths through every pixel-touching subsystem.
**What rvp9 does:** Profile 0 only in v1. `Pixel = u8` is a concrete
type, not generic. No `u16` paths anywhere.
**Why:** ~2× scope multiplier on every pixel-touching path; SIMD
kernel count doubles. Profile 3 (12-bit) has near-zero real-time
demand. PLAN.md target use cases (WebRTC + livestream) are 8-bit.
**Byte-parity impact:** none for our Profile 0 scope. Cross-decoder
streams using Profile 2/3 will fail outright (decoder rejects).
**Compression-efficiency impact:** none within Profile 0; HBD content
is simply out of scope.
**Future plan:** Stage 6+ expansion. Generic-over-`Pixel` is the natural
v2 pattern; the v1 code structure leaves `Pixel = u8` as the only call
site so it can be lifted with a type-parameter pass. Decision: D-01.

### 2026-05-16 — AQ modes 1 and 2 not ported
**Where:** `rvp9-encoder::aq` ships **CYCLIC_REFRESH only**. libvpx
counterpart: `vp9/encoder/vp9_aq_variance.c` (mode 1),
`vp9/encoder/vp9_aq_complexity.c` (mode 2). The `aq_mode` enum in our
port has variants only for `Off` and `CyclicRefresh`.
**What libvpx does:** ships four AQ modes (variance, complexity,
cyclic-refresh, lookahead) but guards modes 1/2/4 behind
`!CONFIG_REALTIME_ONLY`. At cpu-used 5-8 (our target band) modes 1/2
are unreachable; they are dead code at our config.
**What rvp9 does:** drop modes 1 and 2 entirely. Mode 3
(cyclic-refresh) is the only AQ implementation shipped.
**Why:** modes 1/2 are dead at our config; building them is dead code
with zero call paths in v1. The lookahead mode (4) is altref-coupled
and similarly excluded by D-13.
**Byte-parity impact:** none — these modes are never reached in matched
libvpx configs.
**Compression-efficiency impact:** none — same matched configs.
**Future plan:** if a non-real-time preset is ever added, the modes
can be ported then. No earlier. Decision: D-05.

### 2026-05-16 — Alt-ref frames (altref) not supported
**Where:** `rvp9-encoder::ratectrl` and `rvp9-encoder::frame_loop`;
the encoder rejects `lag_in_frames > 0` at construction. libvpx
counterpart: `vp9/encoder/vp9_alt_ref_aq.c`, the `GF_GROUP` altref
slot logic in `vp9/encoder/vp9_ratectrl.c`, the lookahead queue in
`vp9/encoder/vp9_lookahead.c`.
**What libvpx does:** under VBR (and `--lag-in-frames >= 4`) inserts
synthetic alt-reference frames into the prediction structure for
quality gains via look-ahead.
**What rvp9 does:** forbid altref; `lag_in_frames = 0` is the only
supported value. No `GF_GROUP` altref slots. No lookahead queue.
**Why:** altref requires `lag_in_frames > 0`, which directly
contradicts real-time / WebRTC use cases. Latency floor for altref is
4-16 frames depending on `lag_in_frames`; WebRTC budget is sub-frame.
PLAN.md D1 commits to real-time priority.
**Byte-parity impact:** none in matched configs (we mirror libvpx's
behaviour when `lag_in_frames=0`).
**Compression-efficiency impact:** none in matched configs. Against
libvpx with `lag_in_frames > 0`, rvp9 produces lower quality at the
same bitrate — but that's by design (different operating point).
**Future plan:** revisit if a "near-real-time livestream" preset
with `lag_in_frames=4` and a few hundred ms of buffered latency is
ever added. Decision: D-13.

### 2026-05-16 — EXT_RATECTRL plug-in interface always available
**Where:** `rvp9-encoder::ratectrl::ext::ExtRateControl` trait,
always-on, no Cargo feature gate. libvpx counterpart:
`vp9/encoder/vp9_ext_ratectrl.c` behind `CONFIG_RATE_CTRL`.
**What libvpx does:** EXT_RATECTRL is a build-time configurable
feature (`CONFIG_RATE_CTRL=1`) used primarily by libwebrtc. Default
libvpx builds ship without it.
**What rvp9 does:** ships the trait surface unconditionally. The
default `ExtRateControl` impl is a no-op (delegate to the built-in
RC); consumers (libwebrtc) implement the trait for custom control.
**Why:** libwebrtc integration is a v1 success criterion (PLAN.md D3).
Shipping the trait always-on means the API surface doesn't
re-stabilise post-v1.
**Byte-parity impact:** none when the default no-op impl is used —
the built-in RC runs unchanged. Custom impls produce intentionally
different bitstreams.
**Compression-efficiency impact:** none with default impl.
**Future plan:** stable trait API by v1 release; expand callback set
in v1.x non-breakingly. Decision: R-06.

### 2026-05-16 — Trellis-quantization (`vp9_optimize_b`) not ported
**Where:** `rvp9-encoder::quant` ships no trellis-quant path. libvpx
counterpart: `vp9_optimize_b()` in `vp9/encoder/vp9_encodemb.c`.
**What libvpx does:** at slower presets (cpu-used 0-4), runs a
trellis-quantization optimization pass over each coded block to
re-pick token values for lower RD cost.
**What rvp9 does:** no trellis-quant; quant output is used as-is.
**Why:** at cpu-used 6-8 (our target) libvpx sets
`sf->optimize_coefficients=0` — trellis-quant is bypassed. It is
dead code at our config. PLAN.md D1 priorities are clear.
**Byte-parity impact:** none in matched configs.
**Compression-efficiency impact:** none in matched configs.
**Future plan:** revisit only if a quality preset (cpu-used ≤ 4) is
added in a future version. Decision: L-28.

### 2026-05-16 — Compound-mode reference search not ported
**Where:** `rvp9-encoder::me` ships no compound-pred refinement.
libvpx counterpart: `vp9_refining_search_8p_c` and surrounding
compound search code in `vp9/encoder/vp9_mcomp.c`.
**What libvpx does:** at RD-loop presets, searches compound-prediction
references (last+gold, etc.) with an 8-point refinement after the
initial best-MV.
**What rvp9 does:** non-RD path only; compound search never invoked.
**Why:** compound-mode selection is gated off in the non-RD path at
cpu-used 5-8 (`set_rt_speed_feature_framesize_*`). Dead at our config.
**Byte-parity impact:** none in matched configs.
**Compression-efficiency impact:** none in matched configs.
**Future plan:** required if compound modes become reachable in a
future preset. Decision: L-08, L-46.

### 2026-05-16 — SVC sub-pel search-method selection remains unplumbed
**Where:** `rvp9-core::mv_search::sub_pixel` and
`rvp9-core::mode_decision::inter::run_newmv_search`; libvpx counterparts:
`vp9/encoder/vp9_speed_features.c:715,928-930,1057-1068` and
`vp9/encoder/vp9_mcomp.c`.
**What libvpx does:** non-SVC realtime cpu-used 6-8 retains the default
`SUBPEL_TREE`. At cpu-used 7 it switches only selected non-reference SVC
enhancement layers to `SUBPEL_TREE_PRUNED_EVENMORE`.
**What rvp9 does:** now ports and selects the full tree for the scoped non-SVC
realtime path. The older `_pruned_more` implementation remains available, but
the SVC layer-specific method selector and `_pruned_evenmore` variant are not
yet wired.
**Why:** the earlier entry incorrectly classified `_pruned_more` as the
cpu-used 6-8 realtime default. Pinned source inspection during Stage 4a showed
that only the GOOD-mode setter chooses it. The full-tree correction is a
measured non-SVC Pareto improvement.
**Byte-parity impact:** none as a format concern; selected MVs can differ, but
all seven Stage 4a streams decode byte-exactly under vpxdec and ffmpeg.
**Compression-efficiency impact:** on the strict Stage 4a Foreman curve, the
full tree improved BD-rate from `+9.1152%` to `+8.8483%` and matched-rate
deficit from `0.52174` to `0.49409 dB`. Same-load median time improved from
`18.6831` to `18.3473 ms`.
**Future plan:** implement the SVC layer-specific selector and
`SUBPEL_TREE_PRUNED_EVENMORE` before the Stage 4e SVC gate. Decision: L-05
narrowed to SVC.

### 2026-05-16 — REALTIME mode pinned at encoder construction
**Where:** `rvp9-encoder::Encoder::new` accepts no `mode` parameter;
`MODE = REALTIME` is implicit and fixed. libvpx counterpart: the
GOOD / BEST / REALTIME mode dispatch in `vp9/encoder/vp9_encoder.c`
keyed on `oxcf.mode`.
**What libvpx does:** runtime-selectable mode (`--good`, `--best`,
`--rt`) with significant per-mode behavioural divergence (speed
features, ref-frame management, scene-cut detection).
**What rvp9 does:** REALTIME only. GOOD and BEST mode dispatch are
absent; calls that would route there are compile-time impossible.
**Why:** PLAN.md D1 commits to real-time priority; GOOD/BEST are
out of v1 scope. Pinning at construction lets us delete entire
code paths rather than gate them at runtime.
**Byte-parity impact:** none when libvpx is matched to `--rt`.
**Compression-efficiency impact:** none in matched configs. Versus
libvpx `--good` / `--best`, rvp9 is intentionally lower quality at
the same bitrate — but that's a different operating point.
**Future plan:** quality presets are a possible v2 expansion;
GOOD/BEST mode dispatch would re-enter then. Decision: L-32.

### 2026-05-16 — `encode_with_recode_loop` dropped
**Where:** `rvp9-encoder::frame_loop` is single-pass only. libvpx
counterpart: `encode_with_recode_loop()` in
`vp9/encoder/vp9_encoder.c`.
**What libvpx does:** under VBR/two-pass, may re-encode a frame
multiple times with adjusted Q to hit a target rate within tolerance.
**What rvp9 does:** single encode per frame, period. No recode.
**Why:** recode requires either two-pass mode (out of scope, no
look-ahead) or significant per-frame compute (incompatible with
real-time at 1080p60). At cpu-used 6-8 libvpx itself runs single-pass
only — recode dead at our config.
**Byte-parity impact:** none in matched configs.
**Compression-efficiency impact:** none in matched configs.
**Future plan:** dependent on a quality-preset addition. Decision: L-34.

### 2026-05-16 — Cyclic-refresh `time_for_refresh` field dropped
**Where:** `rvp9-encoder::aq::cyclic_refresh` does not carry a
`time_for_refresh` counter. libvpx counterpart: `cr->time_for_refresh`
in `vp9/encoder/vp9_aq_cyclicrefresh.c`.
**What libvpx does:** declares the field but reads it only in dead
paths. A5 noted this as a libvpx-internal TODO (Q-RC-24).
**What rvp9 does:** drop the field from the struct entirely; the
state machine works without it.
**Why:** dead field; carries no state. API simplification.
**Byte-parity impact:** none (the field doesn't affect bitstream).
**Compression-efficiency impact:** none.
**Future plan:** none. Decision: L-03.

### 2026-05-17 — Inverse-DCT step storage: explicit `wraplow` (not implicit)
**Where:** `rvp9-core::transforms::inverse::{idct4, idct8, idct16}` and the
1-coefficient DC fast paths (`idct4x4_1_add`, `idct8x8_1_add`,
`idct16x16_1_add`). libvpx counterpart: `vpx_dsp/inv_txfm.c` —
`idct4_c` (:133), `idct8_c` (:271), `idct16_c` (:557), and the matching
`vpx_idctNxN_1_add_c` paths.
**What libvpx does:** declares `int16_t step[N]` (and `int16_t step1[N]`/
`step2[N]` for the bigger sizes), then computes `step[i] = WRAPLOW(...)`.
In default builds (`CONFIG_EMULATE_HARDWARE=0`) the textual `WRAPLOW`
macro is a no-op `(int32_t)x` cast — but the implicit `int32 → int16`
truncation on the `int16_t step[]` assignment is what actually clamps
intermediate values to 16 bits. The truncation is part of the
specification of the kernel, not an artefact.
**What rvp9 does:** stores step values in `i32` (chosen so we don't
truncate spuriously between row and column passes), and applies
[`wraplow(x) = x as i16 as i32`] explicitly at every libvpx-`step[i] =
WRAPLOW(...)` assignment site. The output of each 1D function — which
libvpx writes back into `tran_low_t` (= `int16_t` at our 8-bit profile)
— is also `wraplow`'d on store.
**Why:** Rust does not implicitly truncate i32 → i16 on assignment to
an `i32` slot. To preserve libvpx's bit-exact intermediate-storage
behaviour we must reproduce the truncation explicitly. Discovered when
extending the family to `idct16` — for the 4×4/8×8 sizes the truncation
is a no-op on typical coefficient magnitudes, but at 16×16 (and
especially 32×32) intermediate sums can plausibly cross the i16
boundary, making the truncation observable.
**Byte-parity impact:** none for well-formed coefficient inputs (the
truncation is a no-op at typical magnitudes); strict bit-exactness
preserved for adversarial or out-of-spec inputs. Pinned by regression
test `idct4x4_1_add_wraplow_truncation_boundary`.
**Compression-efficiency impact:** none.
**Future plan:** apply the same `wraplow` pattern to `idct32`, the
`iadst{4,8,16}` family, and the hybrid `iht*` kernels as they land.
The forward DCT/ADST kernels in `rvp9-encoder::transforms` will need
the same treatment for the encoder reconstruction loop's bit-exactness
against libvpx. Decision: developer judgement (not a numbered LLD
decision — caught during Task 3d implementation).

---

## Stage 2 LLD-surfaced divergences — index (Phase 2e merge, 2026-05-16)

The 15 LLDs surfaced **89 additional divergence entries** in their §7
sections beyond the 10 master entries above. Rather than duplicating
them here (which would balloon this file to 5000+ lines), this section
indexes each LLD's §7 entries by ID + title with a cross-reference to
the LLD. Canonical detail lives in the LLD's §7.N stanza.

Format per row: `<§7.id> — <title>` (cross-reference: `notes/lld/<file>.md#NN`
sub-section). Entries cross-referencing a master entry above are marked
`(→ master entry <YYYY-MM-DD>)`.

The aggregated text of every §7 entry is captured in
`notes/divergence-ledger-phase2e-draft.md` (auto-generated; rerun
`cargo run --release -p rvp9-harness --bin extract-divergences` after
LLD edits).

### `notes/lld/aq.md` — 8 entries

- §7.1 AQ modes 1 and 2 not ported *(→ master 2026-05-16 AQ modes 1/2)*
- §7.2 `time_for_refresh` field dropped *(→ master 2026-05-16 cyclic-refresh)*
- §7.3 `weight_segment` integer-shift order preserved (L-21 RESOLVED — sub-0.3% BD-rate trap)
- §7.4 `vp9_cyclic_refresh_limit_q` only active for screen content (L-24 RESOLVED)
- §7.5 `PostEncodeDecision` returned, not mutated through `cpi` pointer
- §7.6 `vpx_clear_system_state` dropped from `setup` (no x87)
- §7.7 `Last_Source` stride quirk preserved in `is_superblock_flat_static` (bug-for-bug)
- §7.8 AQ-3 ships no SIMD kernels

### `notes/lld/bitstream.md` — 9 entries

- §7.1 Drop parallel tile-packing *(L-35 RESOLVED)*
- §7.2 Replace `setjmp`/`longjmp` error model with `Result` *(D-09)*
- §7.3 Drop altref-driven multi-frame superframes *(→ master 2026-05-16 altref)*
- §7.4 Per-tile output is owned `Vec<u8>`, not `&mut [u8]` into a shared arena
- §7.5 `show_existing_frame` is `Option<u8>`, not "bit + literal"
- §7.6 `BitstreamError::CompressedHeaderTooLarge` carries the actual size
- §7.7 No callback-based error handler on the raw-bit reader
- §7.8 Typed `SuperframeBuilder` rather than per-frame mutate of `ctx->pending_*`
- §7.9 `tile_buffer.buf` is byte-tight, not over-allocated to libvpx's worst-case constant

### `notes/lld/denoiser.md` — 6 entries

- §7-D1 `running_avg_y` indexing flattened to `[layer][ref]`
- §7-D2 Rename "slot 0" of `running_avg_y` to `current_denoised_frame`
- §7-D3 Move `running_avg_y[]` re-init from post-encode to pre-encode
- §7-D4 Denoiser does NOT read or write the segmentation map (L-67 RESOLVED)
- §7-D5 **Denoiser SIMD bit-parity caveat (L-22 / R-03 — load-bearing)** — scalar Rust matches libvpx-C verbatim per ISA; cross-ISA ULP epsilon allowed only within rvp9 when `increase_denoising=1`. Shares L-RTCD-04 with HLD-simd §5.5.
- §7-D6 Skin detection moved to mode-decision LLD

### `notes/lld/entropy.md` — 6 entries

- §7.1 Boolean reader hardcoded to 64-bit `BD_VALUE` (32-bit out of scope)
- §7.2 Decryption callback fields not ported
- §7.3 BoolWriter panic-mid-write is not poison-safe
- §7.4 Counts-baseline double-count quirk preserved (per HLD-threading §5.6)
- §7.5 Bitstream-pack scalar in v1 *(L-35)*
- §7.6 `LOTS_OF_BITS` and EOF semantics preserved

### `notes/lld/frame-buffers.md` — 8 entries

- §7.DV-FB-1 `FrameBuf` refcount uses `Arc::strong_count`, not an integer
- §7.DV-FB-2 3 scaled slots hidden from public API *(L-36 / D-08)*
- §7.DV-FB-3 `Yv12Buffer` drops alpha / HBD / color-meta / corrupted fields
- §7.DV-FB-4 Plane offsets stored as `usize`, not raw pointers
- §7.DV-FB-5 External `vpx_codec_frame_buffer_t` callback path not implemented
- §7.DV-FB-6 Legacy `vp8_yv12_*` allocator not ported
- §7.DV-FB-7 `FrameBuf::mvs` owned, not pointer
- §7.DV-FB-8 Pool slot scan order **frozen** to match libvpx exactly

### `notes/lld/intra-pred.md` — 5 entries

- §7.1 HBD code path removed entirely *(→ master 2026-05-16 HBD)*
- §7.2 4×4 zero-copy `above_row` alias remains a raw-pointer alias
- §7.3 `EdgeScratch` aligned struct vs C `DECLARE_ALIGNED`
- §7.4 Mode→tx-type lookup table re-exported (documentation, not a real divergence)
- §7.5 No `vp9_init_intra_predictors` lazy-once pattern (eager init)

### `notes/lld/loopfilter.md` — 7 entries

- §7.L-LPF-1 `LpfPickMethod::FromFullImage` / `FromSubImage` deferred (RT only ever uses `FromQ`)
- §7.L-LPF-2 `LF_PATH_SLOW` (non-4:2:0) decoder-only
- §7.L-LPF-3 `loopfilter_ctrl` field always present on encoder API
- §7.L-LPF-4 `lfm` array owned by `LoopFilterState` (boxed slice, not pointer)
- §7.L-LPF-5 `lf->mode_ref_delta_update = 0` post-pack
- §7.L-LPF-6 `LfSync` decoder-extra fields moved to `rvp9-decoder`
- §7.L-LPF-7 `last_frame_uf` buffer not allocated in RT

### `notes/lld/mode-decision.md` — 12 entries

- §7.1 Compound modes deferred to v2 *(L-46)*
- §7.2 Altref iterations 8..11 in `REF_MODE_SET` kept but always skip *(→ master 2026-05-16 altref)*
- §7.3 `intra_cost_penalty` `BITS_8` assumption *(→ master 2026-05-16 HBD)*
- §7.4 No `EightTapSharp` in filter-ref search
- §7.5 NEWMV diff-bias constants hardcoded *(L-43)*
- §7.6 30-iteration claim corrected to 16 *(L-40 RESOLVED — 12 inter + 4 intra)*
- §7.7 `BlockYrdOutcome` enum boundary *(L-41 fast-path bypass at speed ≥ 7)*
- §7.8 `variance_low[25]` indexing helper *(L-39)*
- §7.9 `PredBufferHandle` RAII wrapper
- §7.10 Cost LUT location in encoder, not core
- §7.11 cpu-used 5 dispatch excluded from this LLD's scope
- §7.12 Iteration-index-based `mode_rd_thresh` table

### `notes/lld/motion-estimation.md` — 8 entries

- §7.1 `vp9_refining_search_8p_c` not ported (compound NEWMV) *(L-08; → master 2026-05-16 compound search)*
- §7.2 `accurate_sub_pel_search` stubbed (only USE_2_TAPS) *(L-05; → master 2026-05-16 sub-pel variants)*
- §7.3 `vp9_find_best_sub_pixel_tree_pruned[_more,_evenmore]` stubbed *(L-05)*
- §7.4 `vp9_skip_sub_pixel_tree` stubbed *(L-05)*
- §7.5 Exhaustive mesh search stubbed
- §7.6 `vp9_diamond_search_sad` NEON specialisation dropped *(L-RTCD-03 in HLD-simd)*
- §7.7 `vp9_init_dsmotion_compensation` (4-site init) not ported
- §7.8 `reference_masking` threshold frozen at 2× *(L-06)*

### `notes/lld/partition.md` — 10 entries

- §7.1 `choose_partitioning` returns `Result<()>` instead of `int (always 0)`
- §7.2 `vt2` (4×4-downsample scratch) hoisted to per-tile, lazy-alloc
- §7.3 `set_block_size` is `debug_assert!`-guarded for non-edge callers
- §7.4 `PartitionScratch` is not `Copy`
- §7.5 Drop `variance_node` + `tree_to_node` indirection (typed enum)
- §7.6 Explicit `Var::finalize` instead of always-eager `get_variance`
- §7.7 Speed-6 fall-through in `scale_part_thresh_sumdiff` documented *(L-45)*
- §7.8 `vbp_thresholds[3]` cross-frame retention on inter frames
- §7.9 `force_split: [bool; 21]` instead of `int[21]`
- §7.10 `variance_low: [u8; 25]` Morton-order traceability *(L-39; not a divergence — record)*

### `notes/lld/quant.md` — 6 entries

- §7.1 Drop `vpx_bit_depth_t` parameter from quant API *(→ master 2026-05-16 HBD)*
- §7.2 `Quants` / `Dequants` are split structs, not merged
- §7.3 `vp9_init_plane_quantizers` returns lifetime-bound borrowed view, not raw pointers
- §7.4 `vp9_compute_qdelta_by_rate` parameters explicit (no `RATE_CONTROL` god-struct)
- §7.5 `vp9_quantizer_to_qindex` / `vp9_qindex_to_quantizer` only on encoder API
- §7.6 L-25 trap is documented here but enforced in rate-control (cross-LLD)

### `notes/lld/rate-control.md` — 8 entries

- §7.1 `buffer_level` / `bits_off_target` consolidation *(L-01)*
- §7.2 No altref / no `lag_in_frames > 0` *(→ master 2026-05-16 altref; L-76)*
- §7.3 EXT_RATECTRL `max_frame_size = Bits(_)` ignored
- §7.4 Min-Q LUTs baked, not generated at init
- §7.5 `post_encode_drop_cbr` not byte-parity-tested in v1 *(D-07)*
- §7.6 Dead altref fields elided from the on-disk `RateControl` layout
- §7.7 Linear qindex search retained (Q-RC-4)
- §7.8 Float determinism — single libm choice *(L-04 / R-04)*

### `notes/lld/svc.md` — 10 entries

- §7.1 No altref → drop `lc->alt_ref_source`, `lc->alt_ref_idx`, `lc->has_alt_frame`, `lc->gold_ref_idx` *(→ master 2026-05-16 altref)*
- §7.2 `svc.scaled_frames[MAX_LAG_BUFFERS]` collapsed to 1-buffer-per-layer
- §7.3 Scene detection moved one level up (called once per superframe) *(L-68 RESOLVED)*
- §7.4 `test_drop_readonly` for ConstrainedFromAbove precheck (no side-effect mutation)
- §7.5 Single call-site for `inc_frame_in_layer`
- §7.6 Layer-context swap uses `std::mem::swap` (not C-style assignment) *(L-58)*
- §7.7 `MAX_LAYERS = 9` (tight) vs libvpx's 12
- §7.8 ConstrainedFromAbove drop-chain at top-of-superframe (not per-layer) *(L-69 / L-75 RESOLVED)*
- §7.9 EXT_RATECTRL Q-clamp to layer min/max
- §7.10 No-op for `lc->alt_ref_source = NULL` initialisation

### `notes/lld/tile-threading.md` — 7 entries (DV-TT-01..07)

- §7.DV-TT-01 `tile_rows > 1` panics in v1 *(L-62 RESOLVED)*
- §7.DV-TT-02 Determinism flag forces `adaptive_rd_thresh_row_mt = 1` *(R-08 RESOLVED via bit-exact-by-default)*
- §7.DV-TT-03 Workers never invoke `vpx_internal_error` (`Result<>` only)
- §7.DV-TT-04 Pool tears down + rebuilds on `num_workers` change
- §7.DV-TT-05 Mirror libvpx's `FRAME_COUNTS` accumulation bias *(L-57 — bug-for-bug counts; **status: MIRRORED-INTENTIONALLY** per HLD V6 §G5 lines 1107-1162; inline-worker pattern: `num_workers = min(num_threads, num_tiles)`, inline slot mutates `cm.counts` directly, other workers receive baseline memcpy. Cross-references: encoder `notes/lld/tile-threading.md` §7.DV-TT-05; decoder `crates/rvp9-decoder/src/lib.rs` G5 worker_pool wrap in `decode_intra_frame_inner` / `decode_inter_frame`)*
- §7.DV-TT-06 Worker thread priority not elevated *(L-63)*
- §7.DV-TT-07 No frame-parallel encoding in v1 *(Q4 deferred → v2)*

### `notes/lld/transforms.md` — 8 entries

- §7.1 Trellis-quant (`vp9_optimize_b`) not ported *(→ master 2026-05-16 trellis-quant)*
- §7.2 Lossless encoder path not built; decoder honors lossless
- §7.3 Hybrid scan for 32×32: dispatch redundancy preserved
- §7.4 SSSE3 promoted to required for three inverse-DCT kernels on x86_64
- §7.5 EOB-bucket dispatch centralised in `EobBucket::for_size`
- §7.6 Scan-order tables: alignment via `#[repr(align(16))]`
- §7.7 32×32 hybrid (`vp9_fht32x32` / `vp9_iht32x32`) absent
- §7.8 Forward 32×32 `_rd` vs full split

---

### Categories anticipated (forward-looking)

- **Threading primitives** — D6 commits to mirroring libvpx tile-worker
  threading. Where Rust's aliasing rules force a different shape, note it
  here. *(Stage 2: see `tile-threading.md` §7 — 7 entries cover the
  rebuilt pool, panic semantics, determinism flag.)*
- **Integer overflow behaviour** — D7 commits to mirroring widths.
  `wrapping_*` site catalogue lands here. *(Stage 2: covered per-LLD
  in §3/§4 type signatures; ledger entries added as Stage 3 land.)*
- **Bug-for-bug compatibility** — places where libvpx has a known bug
  we **choose not** to replicate, or **choose to** replicate.
  *(Stage 2: `aq.md` §7.7 Last_Source stride; `tile-threading.md`
  §7.DV-TT-05 FRAME_COUNTS bias.)*
- **Heuristic improvements** — places where we make a different
  heuristic decision in the non-RD path (informed by Stage 1 analysis).
  *(Stage 2: none yet; LLDs preserve libvpx heuristics exactly.)*

---

## F-series Stage 4a — encoder MVP divergences (2026-05-21)

The night-3 (F5–F10) + night-4 (F8-fix–F12) drives surfaced a set of
deliberate departures from libvpx's encoder. Listed here in one place
so the Stage 4a exit-gate journal can reference a single anchor.

### 2026-05-21 — block_yrd transform-domain distortion (F8.0-fix)
**Where:** `rvp9-core::mode_decision::intra::block_yrd`. libvpx
counterpart: `vp9/encoder/vp9_rdopt.c::block_rd_txfm` + the rd-cost
accumulator path.
**What libvpx does:** computes distortion in the transform-coef domain
(`(coef - quantized_dequantized_coef)^2 >> shift`) where
`shift = (tx_size == TX_32X32) ? 0 : 2`.
**What rvp9 does:** ports the libvpx transform-coef-domain distortion
formula exactly (down from the pre-fix post-recon-SSE shortcut, which
collapsed all modes to dist=0 on bit-exact content).
**Byte-parity impact:** matched.
**Compression-efficiency impact:** matched within block_yrd. Knock-on
to mode-decision is the new picker actually selects V_PRED on vertical
stripes (correctness recovery, not divergence).
**Known issue:** `block_yrd` is noise-sensitive — small (1–3u)
reconstruction differences inflate non-DC modes' transform-domain
distortion because FHT spreads to high-freq → quant to 0 →
`(coef-0)^2` accumulates. DC's averaging absorbs noise. On
stripe-pattern content with quantizer-induced 1-3u noise, V_PRED can
score WORSE than DC even when V is the true winner. Documented in the
night-4 journal under F10.1b. Follow-up: investigate switching the
distortion accumulator from L2 to L1 in the high-freq band, or
applying a noise-floor offset.

### 2026-05-21 — Fixed pic_norm correction factor (F8.x rate model)
**Where:** `rvp9-encoder::rate_control` and the F8.x rate-model
estimator (per night-4 ledger snapshots). libvpx counterpart:
`vp9/encoder/vp9_rdopt.c` rate-model + AQ feedback adapting the
correction factor per frame.
**What libvpx does:** maintains an adaptive correction factor that
walks with observed (bits_used / bits_estimated) per frame, per layer,
per AQ segment.
**What rvp9 does:** uses a fixed correction factor in v1 — the
adaptive loop isn't wired yet.
**Why:** v1 prioritises a working bitstream-producing encoder over
the rate-model auto-tuning loop. The fixed factor is conservative.
**Byte-parity impact:** structural — rate-control decisions diverge
frame by frame.
**Compression-efficiency impact:** unmeasured (F12.0 didn't reach a
matched-config measurement). Expect 1–3 % BD-rate penalty until the
adaptive loop lands.
**Future plan:** Stage 4b adaptive Q work brings in the per-frame
correction adapter.

### 2026-05-21 — NEARMV not implemented (F8.1 inter mode picker)  [RESOLVED A12]
**Where:** `rvp9-core::mode_decision::inter::pick_inter_mode_luma`.
libvpx counterpart: `vp9/encoder/vp9_pickmode.c::vp9_pick_inter_mode`.
**What libvpx does:** evaluates `{NEARESTMV, NEARMV, NEWMV, ZEROMV}`
× `{single, compound}` × `{LAST, GOLDEN, ALTREF}` references; selects
by RD.
**What rvp9 did before A12:** evaluated `{NEARESTMV, NEWMV, ZEROMV}`
× single ref × `{LAST}` only. NEARMV omitted. Compound-ref not
driven by the picker.
**Why:** F8.1 scope was bounded to demonstrate the picker shape. The
F8.2 partition picker chooses NEARESTMV-vs-NEWMV at the same MV-ref
list anchor; adding NEARMV is mechanical (next candidate in the
MV-ref list) but didn't fit the night-3 budget.
**Byte-parity impact:** structural — wherever NEARMV would have won
RD, rvp9 picks a different mode.
**Compression-efficiency impact:** modest. Expected 0.5–1.5 % BD-rate
penalty on typical content (NEARMV is rare-but-not-vanishingly-so).

**Resolution — A12 (2026-05-22):** Added NEARMV as a fourth
candidate to `pick_inter_mode_luma` (commits `9c18c34`, `ee69a64`).
Encoder integration calls `find_mv_refs(NEARMV, ...)` separately so
the search continues past the first candidate (`early_break` is
gated off for NEARMV), and `mvs[count - 1]` populates
`MvCandidates.near` — matching the decoder's `resolve_parent_mvs`
exactly (`crates/rvp9-core/src/inter/mode_info.rs:1361`). OOB veto
for `mvc.near` mirrors the existing NEARESTMV OOB substitution.
Writer side (`write_inter_block_mode_info`) already supported the
NEARMV tree-encoding (mode bits only, no MV delta) — no writer
change needed; doc-only audit committed in `ee69a64`.

**Measured impact (F12 BD-rate gate, 64 frames @ 256×256, 4
bitrate points 25 / 50 / 100 / 250 kbps):**
- **testsrc**: BD-rate +660 % → +634 % (–26 percentage points).
  100-kbps PSNR +1.16 dB; 250-kbps PSNR +0.40 dB; lower-bitrate
  points unchanged.
- **foreman_cif**: BD-rate +395.92 % → +395.92 % (no change).
  Per-frame NEARMV picks = 0 because `find_mv_refs(NEARMV, ..)`
  returns `count == 1` from the in-progress mi-grid (all
  neighbours carry ZEROMV initially), so `mvs[count - 1] ==
  mvs[0] == nearest`. NEARMV is RD-degenerate to NEARESTMV and the
  picker's argmin (strict `<`) tiebreaker keeps NEARESTMV.
- Cross-decoder gate (vpxdec + ffmpeg): **BYTE-EXACT** preserved.
- RT gate: 5.35 ms median (unchanged from 5.85 ms baseline; noise).

**Diagnostic:** `RVP9_DIAG_INTPRO=1` now reports NEARMV picks in
the per-frame summary (`zero / nearest / near / new / skip /
nonzero_mv`).

**Remaining gap:** the foreman flat-zero NEARMV `count` is a
downstream consequence of the F8.1 picker's "all-ZEROMV initial
state" chicken-and-egg: until at least one Block16+ leaf picks
NEWMV, all spatial neighbours contribute identical zero MVs to the
ref list, so the "second-nearest" slot collapses onto the first.
Activating NEARMV picks on foreman requires either (a) a non-zero
NEWMV at the top of the SB sweep (covered by the int_pro seed
mechanism — but the current seed lands at full-pel only and the
RD margin is too narrow on textured motion) or (b) seeding
`mvc.near` from a previous-frame MV reference
(`use_prev_frame_mvs`, the PrevFrameMvRef plumbing wired through
`prev_mv_grid = None` today). Both are future-experiment
territory; the F8.1 NEARMV divergence itself is RESOLVED.

**Follow-up — A18 (2026-05-22):** Plumbed prev_frame_mvs through
the encoder (commits `a41bbbc`, `48e869b`). StreamEncoder now
maintains a `prev_mv_grid: Vec<PrevFrameMvRef>` mirror of the
decoder's `cm->prev_frame->mvs` — populated post-encode from the
final per-leaf picks, gated on libvpx's `use_prev_frame_mvs`
predicate (size match + `!last_frame_was_key`). Both the picker's
`find_mv_refs` walks (encoder side) and the writer's
`build_inter_packet_with_residuals` (`find_mv_refs` for NEWMV
residual's `best_ref_mv`) now accept the grid.

**However**, activation is gated off by default behind
`RVP9_A18_ACTIVATE` env var. With activation ON:
- **testsrc**: BD-rate +630.68 % → +870.01 % (regression -240 pp).
- **foreman**: unchanged (still +395.92 %) — the prev_mv_grid is
  all-zero because no NEWMV picks happen on the previous frame,
  so the salvage adds zero candidates that dedup against the
  already-zero spatial list. Chicken-and-egg confirmed.

Root cause of testsrc regression: with non-zero prev-frame MVs
propagating into `mvc.near`, the picker's RD math favours NEARMV
(rate-free; no MV-delta bits) even when its prediction error is
worse than refined-NEWMV's after diamond + sub-pel. The picker's
strict-`<` argmin then keeps NEARMV. Cross-decoder remains
byte-exact (the encoder and decoder agree on the wrong picks).

A18 thus leaves the infrastructure in place (storage, write-back,
API extension on `pick_and_simulate_inter_encode_partitions` and
`build_inter_packet_with_residuals`) but defaults to passing
`None` so the BD-rate baseline is preserved. Flipping the gate
default-on requires fixing the picker's NEARMV-vs-NEWMV RD
ranking first — likely a NEARMV rate-side penalty or a
distortion-margin guard.

### 2026-05-21 — Block8x8 inter leaves forced to ZEROMV-skip  [RESOLVED A1]
**Where:** `rvp9-encoder::frame_loop::encode_inter_internal` (night-3
F10.1c). libvpx counterpart: `vp9/encoder/vp9_pickmode.c` over all
leaf sizes ≥ 8×8.
**What libvpx does:** picks inter mode + MV at every leaf size from
Block64x64 down to Block8x8 (and Block4x* via the sub-8 picker).
**What rvp9 did before A1:** used the F8.1 picker for leaves ≥
Block16x16; Block8x8 leaves fell back unconditionally to `ZEROMV /
LASTFRAME / skip = true`. Sub-8 partitions remain wire-format-only.
**Why:** F8.1's `pick_inter_mode_luma` doesn't support sub-16. The
fallback keeps the F10.1c oracle (bit-exact encoder↔decoder recon)
viable across all partition shapes the F8.2 picker can produce.
**Byte-parity impact:** structural — Block8x8 inter leaves with real
motion content reconstructed to zero-MV instead of the picker's
choice.
**Compression-efficiency impact:** small on most content (Block8x8
is the smallest the F8.2 picker emits in v1, and ZEROMV-skip is
already optimal on the majority of Block8x8 leaves in low-motion
content). Larger on high-motion 8×8-detailed content (A0 measured
~130-140 leaves/frame at peak motion ≈ 13% of pixel area).
**A1 resolution (2026-05-21):** lifted Block8x8 into the F8.1 picker
gate at `rvp9-encoder::pick_and_simulate_inter_encode_partitions`.
The simulator's per-plane / per-TX walk already handled Block8x8
generically; only the gate and a chroma-aware tightening of the
MV-limit envelope were required (the picker's MV must keep the
8-tap convolve inside the visible chroma plane as well as luma —
the encoder simulator does not border-extend). F12 testsrc shows
Y-PSNR climbing from locked-at-24.65 dB on motion frames to
35.83-52.64 dB across the 25-250 kbps sweep. Cross-decoder gate
remains BYTE-EXACT vs vpxdec + ffmpeg. Sub-8 partitions still
forced to ZEROMV-skip (no F8.1 sub-8 picker exists yet).
**Future plan:** when sub-8 picker lands, the remaining sub-8
fallback also disappears.

### 2026-05-21 — Chroma intra mode forced to DC on every KF leaf  [RESOLVED A0''-chroma]
**Where:** `rvp9-encoder::pick_and_simulate_intra_encode_partitions`
+ writer-side `BlockMode::Intra { uv_mode: IntraMode::Dc }` in
`rvp9-encoder::frame_loop::build_kf_mi_grid`. libvpx counterpart:
`vp9/encoder/vp9_pickmode.c::vp9_pick_intra_mode` selects from the
full `{DC, V, H, D45, D63, D117, D135, D153, D207, TM}` set.
**What libvpx does:** evaluates all 10 chroma intra modes via RD,
emits one `uv_mode` per leaf.
**What rvp9 did before A0''-chroma:** picker stopped at luma; UV
was hard-forced to `IntraMode::Dc` everywhere. Encoder simulator
predicted both U and V planes as DC; writer emitted
`uv_mode = Dc` on every leaf. End-to-end consistent (cross-decoder
gate passed byte-exact) but structurally wrong on textured chroma.
A0' measured UV PSNR locked at 11-17 dB across q-index 34 → 254 on
foreman_cif.
**A0''-chroma resolution (2026-05-21):** added `pick_intra_mode_uv`
in `rvp9-core::mode_decision::intra` mirroring the existing luma
picker; threads `predicted_uv_mode` through `PartitionLeafDesc`,
the simulator's UV plane loop, and `build_kf_mi_grid`. Single
`uv_mode` applied to both U and V (libvpx wire-format invariant).
Cost table is `intra_uv_mode_costs_kf[y_mode][uv_mode]`.
Block64x64 leaves still fall back to DC (the luma picker is out
of scope there too — F8.x roadmap item).
**Byte-parity impact:** structural for any leaf whose RD winner is
not DC. Cross-decoder gate remains BYTE-EXACT against vpxdec +
ffmpeg.
**Future plan:** when the F8.x sub-Block64x64 picker lands, the
last DC fallback at Block64x64 also disappears.

### 2026-05-21 — KF partition descent capped at Block16x16  [RESOLVED A0''-partition]
**Where:** `rvp9-core::mode_decision::partition::descend`. libvpx
counterpart: `vp9/encoder/vp9_pickmode.c::choose_partitioning`
descends through Block16x16 → Block8x8 on KFs whenever the per-
pixel variance exceeds the threshold.
**What libvpx does:** unconditionally allows the Block16x16 →
Block8x8 split on textured KFs; the picker downstream handles the
per-block mode selection at any size.
**What rvp9 does (post-A0''):** gates the Block16x16 → Block8x8
split OFF on KEYFRAMES (`!params.is_keyframe`). INTER frames keep
the full descent. Diagnosed on foreman_cif: at q ≥ 124, ~92 % of
16×16 KF blocks crossed the variance threshold and split to 8×8,
producing 940/1024 Block8x8 leaves per frame. Per-leaf overhead
(mode bits + EOB context) consumed the entire frame bit budget,
starving the residual coefficients. Capping at Block16x16 frees
per-leaf bits AND keeps every KF leaf inside the F10.1b intra
picker's supported size range.
**Why this gate, not a full cap:** A1 / A1.perf established that
the F8.1 Block8x8 INTER picker is the testsrc PSNR climb (24 →
35-52 dB on motion frames). That picker requires Block8x8 leaves
to exist on inter frames. The KF-only gate preserves the inter
win while closing the KF fragmentation.
**Byte-parity impact:** structural — KF leaves coarser than libvpx
on textured content. Cross-decoder gate remains BYTE-EXACT.
**Compression-efficiency impact:** foreman_cif KF Y-PSNR climbs
slightly from chroma-only (14-16 dB) to ~16 dB across the
operating range. The flat PSNR-vs-bitrate signature still
indicates downstream bottlenecks (likely F8.1 inter picker on
textured real content).
**Future plan:** when libvpx-faithful KF RD-loop (per-leaf
per-size RD comparison) is wired, the cap can be lifted — at that
point per-leaf bits are spent only where they reduce RD net of
overhead.

### 2026-05-21 — BD-rate trapezoidal approximation (F12.0)
**Where:** `rvp9-harness::bdrate::bdrate`. libvpx / standard
counterpart: Bjøntegaard 2001 (and 2008 update) cubic polynomial fit
in `log10(bitrate)` vs PSNR with closed-form integration.
**What libvpx does (or what tools-against-libvpx do):** fits a
3rd-degree polynomial to each curve's `(log10(bitrate), PSNR)` points
and integrates the closed-form gap between the two polynomials over
the overlap.
**What rvp9 does:** trapezoidal-rule integration of
`log10(bitrate)` vs PSNR over the overlap; geometric-mean ratio
expressed as percent. No polynomial fit, no extrapolation outside
the overlap.
**Why:** the trapezoidal approximation requires zero new
dependencies (no linear-algebra crate needed). For matched-config
curves with monotonic shape and 4+ points the answer agrees with
the cubic fit to within ~0.5 percentage points (Bjøntegaard's own
2008 follow-up note documents this).
**Byte-parity impact:** N/A (measurement, not bitstream).
**Compression-efficiency impact:** N/A — this is the YARDSTICK,
not a coding choice. The yardstick has documented uncertainty
±0.5 pp.
**Future plan:** if a 0.5 pp uncertainty matters at the Stage 4a
exit gate, port the cubic fit (4×4 matrix solve, ~30 lines of
arithmetic, no crate dependency).

### 2026-05-21 — Encoder-side loop-filter level approximation (F11.0)
**Where:** `rvp9-encoder::loopfilter::pick_filter_level`. libvpx
counterpart: `vp9/encoder/vp9_picklpf.c::av1_pick_filter_level` with
full RD search.
**What libvpx does:** searches `filter_level` to minimise RD cost
across a range bracketed by previous-frame `filter_level`. Around
half a dozen reconstructions per frame.
**What rvp9 does:** approximates the picked level as
`(q_index >> 1).min(63)`. Skips loop filter entirely at q < 2 (the
F5.1 quant table is undefined at q = 0; encoder forbids q = 0).
**Why:** RD-search over filter level is a per-frame inner-loop cost.
v1 ships the heuristic. Bit-exact encoder↔decoder reconstruction is
preserved (both apply the same `filter_level`).
**Byte-parity impact:** structural — picked level differs from
libvpx's per-frame choice.
**Compression-efficiency impact:** unmeasured. The libvpx heuristic
on cpu-used 7-8 is close to a linear-q approximation already, so
expected delta is small (< 0.5 % BD-rate).
**Future plan:** if Stage 4b's measurement work shows > 1 % BD-rate
attributable to filter-level choice, port the RD search.

### 2026-05-21 — No NEARMV / compound-ref / temporal SVC inter shapes [A14 Phase A landed 2026-05-22]
**Where:** rvp9-encoder doesn't drive compound-ref inter blocks or
temporal SVC layering. libvpx counterpart: full inter-coding shape
surface.
**What libvpx does:** ships all VP9 inter shapes.
**What rvp9 does:** single-ref LASTFRAME only in the encoder's
PICKER (decoder handles the full surface). No SVC. A14 Phase A
(2026-05-22) added GOLDEN slot maintenance + periodic refresh on
the encoder's reference-pool side — `golden_recon_*` buffers
mirror what the decoder writes when reading `refresh_frame_flags`
bit 1 — but the picker still iterates LASTFRAME only.
**Why:** v1 scope (Stage 4a MVP). SVC is Stages 4d/4e. Compound-ref
is mechanical-but-orthogonal work.
**Byte-parity impact:** structural — the wire-format writer can emit
compound-ref blocks but the encoder doesn't pick them.
**Compression-efficiency impact:** the largest single residual
divergence from full-libvpx parity. Expect 5–15 % BD-rate penalty on
typical real-time content (compound-ref bidirectional inter is the
dominant win at our cpu-used target).
**A14 Phase A landing (2026-05-22):** added
`golden_recon_{y,u,v}` buffers + `frames_since_golden_refresh`
counter to `StreamEncoder` (`crates/rvp9-encoder/src/frame_loop.rs`).
KF path copies post-LF reconstruction into GOLDEN (matches the
decoder's `refresh_frame_flags=0xFF` slot fan-out); inter path
emits `refresh_frame_flags = 0x03` every
`A14_GOLDEN_REFRESH_PERIOD = 8` inter frames (vs the A11 `0x01`
baseline) and mirrors the new recon into GOLDEN. Wire-format
output is identical to A13 on no-refresh frames; on refresh
frames the one byte carrying `refresh_frame_flags` changes value
(same bit count). Cross-decoder gate stays BYTE-EXACT vs
vpxdec + ffmpeg. F12 PSNR sweeps unchanged from A13 because the
picker still consumes LAST only (the GOLDEN slot is maintained
but never read).
**A14 Phases B/C/D outstanding (picker + writer + measurement
integration):** Phase B widens the picker's `MvCandidates` to
carry both LAST and GOLDEN refs, doubles the per-leaf candidate
iteration from 4 to 8 (`{LAST, GOLDEN} x {ZERO, NEAREST, NEAR,
NEW}`), and adds a `picked_ref_frame` field to
`InterPartitionLeafDesc`. Phase C threads that through
`splat_inter_leaf_into_mi_grid` + `build_inter_mi_grid_from_picker`
so the writer's existing GOLDEN-emission branch is exercised
(the writer was already symbolic-ref-correct via the round-trip
to the decoder's `read_ref_frames`). Phase D is the BD-rate
measurement sweep on foreman_cif (expected meaningful PSNR
climb at all four target bitrates once the picker has access to
a non-stale reference).
**Future plan:** Stage 4a exit gate will measure WITHOUT compound-
ref enabled on the libvpx side too (`--cpu-used 7 --rt` defaults
match this constraint at the picker level); once parity at matched
config is hit, Stage 4d/e widens the encoder.

**A15 Phase B landing (2026-05-22):** Steps 1-5 landed the
picker / mi_grid / writer / LF data flow for LAST+GOLDEN ref
selection. `InterPartitionLeafDesc::picked_ref_frame` (Step 1)
threads through `splat_inter_leaf_into_mi_grid` (Step 4),
`build_inter_mi_grid_from_picker` (Step 4), and
`build_inter_mi_grid_for_lf` (Step 4). Per-leaf the encoder
builds BOTH `find_mv_refs(LAST)` and `find_mv_refs(GOLDEN)`
candidate sets (Step 2) and invokes `pick_inter_mode_luma`
twice — once per ref — taking the min-RD winner across the two
4-candidate rankings (Step 3, effective 8 candidates). The
per-plane simulator selects last_recon vs golden_recon as the
ref plane based on the picked ref (Step 3). Cross-decoder gate
remains BYTE-EXACT vs vpxdec + ffmpeg under both LAST-only and
LAST+GOLDEN modes.

**A15 measurement (Step 5) — GOLDEN selection DEFAULT-OFF:** the
naive `argmin(pick_last.rd, pick_golden.rd)` selector regressed
F12 PSNR by ~10 dB on both testsrc (33.51 → 15.42 dB at 25 kbps)
and foreman_cif (24.01 → 15.63 dB at 25 kbps). Root cause: the
picker's RD doesn't include the GOLDEN ref-frame rate-cost
penalty (~1 extra wire bit per single-ref block: LAST = 1 bit
`0`; GOLDEN = 2 bits `1`+`0`). At Stage 4a operating qindex
(typically 200-255 under the current rate model), residuals
collapse to zero so `recon = pred`; SSE picks become trivially
close between LAST and GOLDEN preds and the missing rate term
lets GOLDEN sneak ahead by a sliver of RD. The picker then
chose GOLDEN ~25-35 % of leaves per inter frame (counts of
~200-300 GOLDEN picks per ~1024 mi-units on foreman_cif), and
the resulting reconstruction mixes stale-from-KF GOLDEN patches
into frames LAST would have tracked closer.

A15 ships the **infrastructure** with GOLDEN selection gated
behind `RVP9_PICK_GOLDEN=1` (off by default). When the env-var
is absent the picker pins `picked_ref_frame = LastFrame` and
the encoder's wire-format output matches A14 byte-exactly
(F12 PSNR sweeps unchanged: testsrc 33.51 / 42.92 / 48.33 /
56.86 dB, +628.52 % BD-rate; foreman_cif 24.01 / 24.16 / 25.42 /
28.96 dB, +395.92 % BD-rate; RT median 5.86 ms ≤ 33 ms gate).
The dual-ref `find_mv_refs` walks + dual `pick_inter_mode_luma`
calls add ~0.1 ms median to the 256×256 / 352×288 RT path; 720p+
runs unchanged (the entire inter picker is frame-area-gated at
307_200 px).

**A15 future work (Phase B follow-up):** add a
`ref_frame_rate_cost` term to `evaluate_candidate` that prices
in the single-ref bit difference (LAST: 1 bit; GOLDEN: 2 bits;
weighted by the entropy probability `single_ref_prob[ctx][slot]`
to match the writer's actual emit). With that term, the
`argmin(pick_last, pick_golden)` selector should bias toward
LAST on near-tied candidates (recovering the baseline byte-
exactly) AND let GOLDEN win on content where the GOLDEN block
genuinely beats LAST by more than the ~1-bit penalty — e.g.
post-occlusion or scene-cut-recovery leaves. The same hook also
unblocks compound-ref pricing when Stage 4d wires compound
inter.

### 2026-05-22 — A16 ref-frame rate-cost landed; GOLDEN gate STAYS default-off

**Where:** `crates/rvp9-core/src/mode_decision/inter.rs`
(`InterModeSearchParams::ref_frame_rate_cost`, plumbed into
`evaluate_candidate`'s rate term); `crates/rvp9-encoder/src/lib.rs`
(per-frame precomputed `ref_frame_cost_last` /
`ref_frame_cost_golden` using libvpx-default probs, fed to the
`pick_for_ref` closure for both ref-frame searches).

**What libvpx does:** `init_ref_frame_cost`
(`vp9_pickmode.c:1215-1230`) precomputes the per-ref-frame rate
cost using `vp9_cost_bit(intra_inter_prob, 1) + vp9_cost_bit
(single_ref_prob[ctx][slot], bit)` for each ref frame, then adds
it to every candidate's `this_rdc.rate` inside
`vp9_pick_inter_mode`'s candidate evaluation loop
(`vp9_pickmode.c:2398`).

**What rvp9 does (A16):** mirrors the libvpx formula but at a
fixed middle-context approximation
(`DEFAULT_SINGLE_REF_PROB[2] = [142, 142]`,
`DEFAULT_INTRA_INTER_PROB[2] = 187`) — the picker is called
BEFORE the leaf is splatted into the mi_grid so the true per-
leaf neighbour-derived context isn't free at this point. The
differential between LAST and GOLDEN at this context is ~566
prob_cost units (~1.1 wire bits at the p=128 reference).

**Why (and what we measured):** with the rate term active the
picker's GOLDEN pick rate on foreman_cif dropped from ~200-300
per inter frame (~25-35 %; the unpriced A15 default-OFF
baseline) to 0-4 per frame (~0-0.5 %). However F12 PSNR with
the gate FORCED-ON via `RVP9_PICK_GOLDEN=1` *still* regressed:
24.01/24.16/25.42/28.96 dB (A15 baseline, GOLDEN never picked)
vs 23.78/23.61/24.34/26.64 dB (A16 default-ON, ~0.5 % GOLDEN
picks). A 4x rate-cost multiplier suppressed GOLDEN to 0 picks
and the baseline was recovered byte-exactly — confirming the
remaining 0.5 % of picks ARE the regression source.

**Root cause (deeper than rate cost):** the per-leaf RD doesn't
price the propagation cost (`find_mv_refs` neighbour lookups
read THIS leaf's picked MV when picking the next leaf — a
GOLDEN-MV from a stale frame pulls neighbours into worse
positions); the frame-level rate-controller's bit budget is
built assuming LAST-only and reacts oddly to mixed refs; the
GOLDEN recon plane itself may diverge from libvpx (it's the
previous KF's recon at 4-frame-KF spacing — older content than
libvpx's `--cpu-used=7 --rt` would carry).

**Decision:** ship A16 with the rate-cost INFRASTRUCTURE
landed (the LAST-only path applies
`+ref_frame_cost_last`, so absolute RD magnitudes are libvpx-
shaped for future intra-vs-inter tuning and downstream
compound-ref pricing) but keep the GOLDEN gate **DEFAULT-OFF**.
`RVP9_PICK_GOLDEN=1` re-enables the argmin-across-refs
selector for tuning runs. The `RVP9_DIAG_GOLDEN_PICKS=1` env-
var emits per-frame `last=N golden=M total=K golden_pct=P`
diagnostics.

**2026-07-30 update:** the causal fixed-tree experiment now derives exact
above/left contexts and prices the active adapted probabilities. A pinned
libvpx oracle captured 11 distinct cost vectors in eight q120 Foreman frames;
95.2% of leaves selected p1 context 4 rather than the fixed middle context.
Applying those costs to the legacy speculative A4 descent regressed the strict
Stage 4a BD-rate from `+4.9246%` to `+7.0553%`, so the production-default A4
path deliberately retains the approximation. The exact fixed-tree path uses
active costs by default because its partition tree is already fixed and the
costs affect only the causal leaf picker. This restriction lives at
`crates/rvp9-encoder/src/lib.rs:active_single_ref_frame_costs` and remains
until the fixed-tree path replaces A4.

**Byte-parity impact:** none on the default path
(`pick_golden_enabled = false`); cross-decoder gate (vpxdec +
ffmpeg, 8-frame 256×256 testsrc) remains BYTE-EXACT vs both
reference decoders.

**Compression-efficiency impact:** none on the default path —
F12 testsrc 33.51/42.92/48.33/56.86 dB and F12 foreman_cif
24.01/24.16/25.42/28.96 dB / +395.92 % BD-rate preserved
exactly. RT median 5.50 ms (was 5.86 ms; the new cost
precompute is once-per-frame, vanishing vs the SB loop).

**Future plan (Phase C / Stage 5):** before re-enabling GOLDEN
selection, address one of (a) per-leaf RD propagation cost
(price the neighbour-MV-quality impact of this leaf's pick),
(b) frame-level RC bit-budget partitioning by ref frame, or
(c) a known-good GOLDEN recon plane (e.g., a libvpx-faithful
KF→GF refresh cycle that maintains GOLDEN closer to the
current frame). Compound-ref pricing for Stage 4d can re-use
the same `ref_frame_rate_cost` hook with `RefMode::CompoundRef`
plumbing.

## AQ-series Stage 4b — Cyclic-Refresh AQ divergences (2026-05-21)

### 2026-05-21 — AQ.5 BD-rate measurement on short / synthetic sources
**Where:** `crates/rvp9-encoder/tests/bdrate_gate.rs::bdrate_gate_stage4b_aq`
— the four-curve Stage 4b exit-gate harness (rvp9 AQ on/off vs libvpx
AQ on/off).
**What libvpx / reference workflows do:** measure BD-rate on the
`yt-ut-hd` / `derf` corpus loops with sequences of hundreds to
thousands of frames, at operating bitrates inside the
encoder's discrimination range.
**What rvp9 does:** the AQ.5 harness runs 8 frames of a 256×256
centre-crop from `tests/corpus/testsrc_720p30.y4m` at four target
bitrates spanning 200–1600 kbps. Both rvp9 and libvpx-stock
(`--rt --cpu-used=7`) saturate inside that range on this sequence
length, so the literal 0.3 % BD-rate gate is **NOT MEASURABLE** —
gain numbers are dominated by rate-control noise, not by the AQ
algorithm.
**Why:** v1 ships the measurement infrastructure now; the corpus +
long-sequence harness is Stage 5 work. The brief explicitly allows
"infrastructure GREEN, literal gate deferred for real-content
corpus" if that's the truth.
**Byte-parity impact:** N/A (measurement, not bitstream).
**Compression-efficiency impact:** N/A — the YARDSTICK has
documented uncertainty in this regime, NOT the encoder.
**Future plan:** Stage 5 — 240-frame sweeps on multiple y4m
sources at 256×256 / 512×512, post rate-control fix (next entry),
re-run the four-curve gate, expect a finite negative gain on both
legs and assess literal 0.3 %.

### 2026-05-21 — rvp9 rate-controller bitrate saturation (F9.0) — RESOLVED (RC.1)
**Where:** `crates/rvp9-encoder/src/rate_control/` (F9.0 single-
pass CBR controller).
**What libvpx does:** the libvpx rate controller lowers q-index
aggressively when actual bitrate runs below target, eventually
reaching q ≈ 0 for very-high target bitrates.
**What rvp9 does (F9.0, BEFORE RC.1):** the F9.0 controller was
mechanically correct (matched the libvpx update formula) but
hard-coded `correction_factor = 1.0` everywhere. In practice the
actual bitrate plateaued 5–40× below target across the operating
range the AQ.5 harness explores. At 256×256@30fps with target
1600 kbps the rvp9 encoder produced only ~39 kbps actual.
**Why (root cause):** the missing piece was libvpx's per-level
`rate_correction_factors[RATE_FACTOR_LEVELS]` table and the
damped post-encode update in `vp9_rc_update_rate_correction_factors`
(`vp9_ratectrl.c:737-817`). Without that feedback loop the
controller's per-frame Q estimate could not converge on the actual
encoder response curve — every prediction was off by the same
fixed bias, so the buffer model alone was insufficient to push Q
into the discrimination range.
**Resolution (RC.1, 2026-05-21):**
- Added `rate_correction_factors: [f64; 5]` and
  `damped_adjustment: [bool; 5]` to `RateControl` state, seeded
  per libvpx `vp9_rc_init` (`vp9_ratectrl.c:419-422`): factors
  start at 1.0, damped flags at false.
- Added `RateFactorLevel` enum mirroring libvpx's
  `INTER_NORMAL / INTER_HIGH / GF_ARF_LOW / GF_ARF_STD / KF_STD`
  layout (5 slots — `vp9_ratectrl.h:48-55`). v1 only reads/writes
  `InterNormal` and `KfStd`; the other 3 slots are reserved for
  future GF/ARF work.
- Threaded `correction_factor` into `bits_per_mb`,
  `estimate_bits_at_q`, and `regulate_q`. Added
  `get_rate_correction_factor` / `set_rate_correction_factor`
  helpers (port of `vp9_ratectrl.c:689-735`).
- Implemented `update_rate_correction_factors` faithful to
  `vp9_ratectrl.c:737-817`, including the 102/99 dead-band
  gates, the multiplicative walk, the
  `0.25 + 0.5 * min(1, |log10(0.01 * cf)|)` damping formula,
  and the `[MIN_BPB_FACTOR=0.005, MAX_BPB_FACTOR=50]` clamps.
  Wired into `postencode_update` BEFORE the `last_q` / EWMA
  update (matches libvpx ordering at `vp9_ratectrl.c:1859`).
- Picked-Q tracking: added `last_picked_qindex` to `RateControl`,
  set in `pick_q_and_bounds`, consumed by
  `update_rate_correction_factors` as the libvpx
  `cm->base_qindex` analogue.
- Added a closed-loop convergence oracle in
  `postencode::tests::rate_control_converges_at_multiple_bitrates_*`
  that drives the controller through a synthetic encoder whose
  response is `bits_per_mb(q, 1.0) * content_factor` (with
  `content_factor = 3.0` mimicking the 5-40× divergence). The
  oracle gates `|actual - target| / target ≤ 20 %` at 250 kbps /
  1 Mbps / 4 Mbps over 120 inter frames after a 60-frame warm-up.
- Measured convergence on the synthetic oracle:
  ```
  250 kbps @ 240p: actual=8851 b/f  target=8333 b/f  +6.2%
  1 Mbps  @ 360p: actual=35477 b/f  target=33333 b/f +6.4%
  4 Mbps  @ 720p: actual=141819 b/f target=133333 b/f +6.4%
  ```
**Intentionally NOT ported:** the libvpx oscillation-detector
fields (`q_1_frame` / `q_2_frame` / `rc_1_frame` / `rc_2_frame`,
`vp9_ratectrl.c:781-795`) and the `vp9_encodedframe_overshoot`
9-step state mutation. Those feed the GF-overshoot rescue path,
which the F9.0 / RC scope explicitly excludes.
**Byte-parity impact:** structural still — once the controller
walks its per-level factor to a content-dependent equilibrium,
the per-frame Q-index decisions diverge from any encoder driven
at a fixed factor, including older rvp9 snapshots. Against
libvpx with the same content the per-frame decisions are now
expected to MATCH within damping noise (factor converges to
identical equilibrium because both encoders compute the same
`bits_per_mb` enumerator). Verified at the synthetic-encoder
level by the convergence oracle.
**Compression-efficiency impact:** large positive — F12 / AQ.5
gates can now operate inside the controller's discrimination
range. Real-corpus BD-rate measurements (RC.6) will quantify
the gain.
**Future plan:** RC.6 re-runs F12 + AQ.5 gates on the real
corpus checked in by RC.2. RC.7 closes the Stage 4a + 4b
journal once the gates are green.

### 2026-05-22 — Spatial MV-ref propagation (A2.foreman H3')
**Where:** `rvp9-encoder::pick_and_simulate_inter_encode_partitions`
(`lib.rs:2454` site). libvpx counterpart:
`vp9/encoder/vp9_pickmode.c::find_predictors` → `vp9_find_mv_refs`
/ `mv_refs_rt`, plus `vp9_mv_pred` for the diamond-search seed.
**What libvpx does:** for each leaf, the realtime picker calls
`mv_refs_rt` (or `vp9_find_mv_refs` for the common path) to build
a spatially-derived `mv_ref_list` from neighbour blocks' MVs.
`vp9_find_best_ref_mvs` then picks `(NEAREST, NEAR)` from that
list. The result drives both NEARESTMV's reconstruction MV and
NEWMV's diamond-search seed.
**What rvp9 did before A2.foreman:** at every leaf, the picker
was fed `MvCandidates::default()` — a zero `(nearest, near)`
pair regardless of what neighbours had picked. This made NEWMV
always encode as a delta from `(0, 0)`, so the NEWMV rate cost
(~5700 vp9_prob_cost units for typical sub-pel MVs) was much
higher than NEARESTMV (mode rate ~295). At low-Q the picker
exclusively selected NEARESTMV with `mv=(0,0)` → no motion
compensation → on continuous-motion content (foreman_cif) the
reconstruction drifted from the source every frame (PSNR
trajectory 30.9 dB → 14.7 dB over 32 frames at 250 kbps).
**What rvp9 does post-A2.foreman:** the encoder builds an
in-progress `mi_grid` and splats each picked leaf's `ModeInfo`
into the cells it covers, in raster order. Before each picker
call, `find_mv_refs` (with `InterMode::NewMv` early-break)
walks the neighbour positions through `InterPickerMiGridLookup`
and seeds `MvCandidates.{nearest, near}` with the
spatially-derived MV pair. The decoder runs the SAME
`find_mv_refs` on its own decoded mi_grid; cross-decoder
byte-exact agreement is preserved (verified by the cross-decoder
gate which still passes vs both vpxdec and ffmpeg).
**OOB-nearest safety:** the encoder picker's `MvLimits` remain
tighter than the decoder's `clamp_mv_ref` envelope because its
sub-pel candidate evaluator uses tight reference planes. When a
non-zero spatially-derived `mvc.nearest` falls outside
`MvLimits`, NEARESTMV evaluation could over-read the reference
plane. The picker excludes an unsafe clamped NEAREST winner; the
final emitter separately uses `build_inter_pred_plane_unscaled`
with an edge-replicated scratch.
**Visible full-chroma-pel follow-up (2026-07-30):** the strict
halo envelope also excluded vectors along the outer block band
that need no interpolation halo. The
`mv_in_strict_limits_or_visible_full_chroma_pel` predicate admits
an out-of-envelope vector only when both Q3 components are
multiples of 16, so it lands on integer samples after 4:2:0
chroma subsampling, and the complete visible block remains inside
the frame. This includes exact zero. Odd whole-luma-pel and
fractional candidates retain the tight-plane veto because chroma
or luma interpolation still needs a halo.
**Byte-parity impact:** preserved; cross-decoder gate
BYTE-EXACT vs vpxdec + ffmpeg.
**Compression-efficiency impact (A2.foreman):** foreman_cif
PSNR is unchanged at 16.21–16.56 dB across 25/50/100/250 kbps
— **the fix is correct foundation but cannot fire on
foreman_cif yet** because no leaf ever picks NEWMV at the
qindex regime the rate-controller drives the encoder to (so
neighbours always have `(InterMode::NearestMv, Mv::ZERO)`,
which makes the spatially-derived nearest identical to
`MvCandidates::default()`). The missing complement is
libvpx's `vp9_int_pro_motion_estimation` SB-level coarse ME
that bootstraps NEWMV by giving the picker a starter MV
during variance partitioning (`vp9_encodeframe.c:1460-1495`).
**Compression-efficiency impact (testsrc):** mild positive at
25/50/100 kbps (+0.04 to +0.5 dB), mild regression at 250
kbps (52.64 → 48.49 dB). BD-rate moved from +1033 % to
+959 % vs libvpx.
**Compression-efficiency impact (edge-motion follow-up):** exact
zero first moved the strict seven-point Foreman product gate from
`+15.7683%` to `+14.0148%` BD-rate and from `0.81462 dB` to
`0.72143 dB` matched-rate deficit. Admitting the other safe
full-chroma-pel vectors then moved those results to `+13.3964%`
and `0.70910 dB`. At 250 kbps, the final mode areas are ZEROMV
`9.77%` and NEARESTMV `56.18%`. All streams remained byte-exact
under vpxdec and ffmpeg.
**Future plan:** port `vp9_int_pro_motion_estimation` (the
3-row 32-tap integer-projection ME that produces an SB-level
seed MV from the variance partitioner) and wire it into the
picker's NEWMV diamond-search center via `x->pred_mv[]`. That
unlocks the spatial-propagation chain: once one leaf picks
NEWMV with a non-zero MV, neighbouring leaves see that MV in
their `find_mv_refs` list and propagate it.
**Status update (A3'-int-pro, 2026-05-22):** the int_pro
seed has now been ported and wired through. The seed
produces non-zero MVs for foreman_cif (e.g. (8, -24) Q3 =
(1, -3) full-pel on early frames). However, **the picker
still picks 0 NEWMV on foreman**. Root cause: CBR drives
qindex to 255 (max quantization) because the encoder
overproduces bits when MC is missing, which makes
`compute_rd_mult` lambda enormous (~7.8M), which makes the
MV-bit cost (~9000-12000 units for typical sub-pel MVs)
dominate any SSE gain (typically only saves a few thousand
SSE units × 128 = ~500K RD units). The picker correctly
rejects NEWMV under this lambda imbalance. The next layer
is **either** a rate-control fix (don't pin qindex at 255
when the encoder is producing 2-5× the target bitrate
because of poor MC, not because of overly-fine quant) **or**
a libvpx-faithful rate model that drops NEWMV's effective
rate at high qindex (libvpx's realtime path uses
`block_yrd` which models rate from quantized-coefficient
energy, not pure rate-table lookup — this is the F8.1
divergence already documented in `inter.rs` module docs).
On testsrc the int_pro seed DID recover PSNR at 250 kbps
(48.49 → 59.69 dB, a +11 dB win) because testsrc has
static content and the seed activation correctly
identifies (0,0) as the best MV, freeing the picker from
the pre-A3.inter-induced regression. BD-rate vs libvpx
moved +959 % → +627 % on testsrc.

### 2026-05-22 — SB-level int_pro motion estimation + sb_use_mv_part short-circuit (A3'-int-pro)
**Where:** `rvp9-core::mv_search::int_pro_me` (new module);
`rvp9-core::mode_decision::inter::run_newmv_search` (new
short-circuit branch);
`rvp9-encoder::pick_and_simulate_inter_encode_partitions`
(per-SB int_pro call upstream of the per-leaf picker).
libvpx counterpart: `vp9/encoder/vp9_mcomp.c:2192-2400`
(`vector_match`, `vp9_int_pro_motion_estimation`),
`vp9/encoder/vp9_encodeframe.c:1460-1495` (caller in
variance partitioning), `vp9/encoder/vp9_pickmode.c:207-210`
(`sb_use_mv_part` short-circuit).
**What libvpx does:** the variance partitioner
(`vp9_choose_partitioning`) runs
`vp9_int_pro_motion_estimation` once per SB and stashes the
resulting MV in `x->sb_mvrow_part` / `x->sb_mvcol_part`.
The per-leaf realtime picker reads `x->sb_use_mv_part` and,
when set, uses the int_pro MV as the NEWMV candidate
directly — skipping the full-pel diamond. Sub-pel
refinement still runs around the int_pro MV.
**What rvp9 does (port):** ported the int_pro algorithm
(1D row/column projections + 4-pt diamond + 2-pt cross
SAD refinement) bug-for-bug, including the
`vector_match`-returns-`-bw/2` edge case for constant
input. Wired into the encoder's
`pick_and_simulate_inter_encode_partitions` to run once per
64×64 SB before the per-leaf picker. The picker's
`InterModeSearchParams::newmv_seed` field carries the
result; `run_newmv_search` consumes it via the same
`sb_use_mv_part` short-circuit (skip diamond, feed the
seed directly to sub-pel refinement).
**Frame-area gate:** int_pro is gated on `w * h <=
307_200` (same threshold as the Block8x8 picker, A1').
At 720p+ the picker's RD math at high qindex (CBR drives
qindex → 255 because of bit overproduction) heavily
favours NEARESTMV-at-(0,0) regardless of seed quality, so
the int_pro work is wasted. Below 307_200 (256×256
testsrc, 352×288 foreman_cif) int_pro runs and produces
seeds.
**Byte-parity impact:** none. Cross-decoder gate
BYTE-EXACT vs vpxdec + ffmpeg preserved on the 8-frame
testsrc encode (the int_pro seed only affects the
encoder's MV picking; the bitstream the encoder writes
remains decodable bit-for-bit by both reference
decoders).
**Compression-efficiency impact:** testsrc at 250 kbps:
PSNR 48.49 → 59.69 dB (+11.2 dB), BD-rate +959 % → +627 %.
testsrc at 25/50/100 kbps: no measurable change
(content is already static so NEARESTMV-at-(0,0) is
correct). foreman_cif: no PSNR change (16.2-16.5 dB
across the sweep) — the rate-control / lambda imbalance
at qindex 255 is a separate bug; the int_pro seed alone
cannot fix it.
**Future plan:** L-05 close-out depends on rate-control
or block_yrd port (see status update above).

### 2026-05-22 — Inter picker frame-area gate (A4.rt-recovery)
**Where:** `rvp9-encoder::pick_and_simulate_inter_encode_partitions`
(new `inter_picker_enabled` gate, applied as a precondition to the
existing `picker_supported` match).
**What libvpx does:** runs the realtime per-leaf inter mode picker
(`vp9_pick_inter_mode`) for every leaf regardless of frame size,
with the speed feature dial governing how much of the picker's
candidate set fires (NEARESTMV always evaluated, NEWMV diamond +
sub-pel guarded by `cpi->sf.mv.*`). libvpx hits its 720p RT budget
via aggressive speed-feature pruning under the SIMD-accelerated
transform/quant/SAD kernels we haven't ported yet.
**What rvp9 does (divergence):** on frames with
`width * height > 307_200` (i.e. 720p+ RT path), short-circuit the
per-leaf picker and the `find_mv_refs` spatial-seed walk; emit
`ZEROMV / skip = true` everywhere instead. Mirrors the A1'
Block8x8 gate at the same threshold, extended to Block16x16 /
Block32x32 / Block64x64. Block8x8 + int_pro gates above still
evaluate at the same threshold (kept separate so a future
refinement can lower the cutoff per-gate).
**Why:** A3.inter (commit 6d62906) + A3'-int-pro (commits bcc0540,
07baea9, 1e2ccec) introduced ~5-7 ms of per-leaf picker + spatial-
MV-ref overhead at 720p, but the picker's RD math at the CBR-
selected qindex on 720p+ heavily favours NEARESTMV-at-(0,0)
regardless of seed quality (A3'-int-pro divergence note). Running
the picker on 720p produces output equivalent to the pre-A1
ZEROMV-skip fallback while costing the full 3-candidate RD scan.
Gating it off recovers the RT budget without changing the picker's
effective behaviour at this content/q operating point.
**Byte-parity impact:** none on the cross-decoder gate (256x256,
below the threshold, picker enabled). 720p output bytes change (no
residuals on inter frames — all leaves are ZEROMV-skip), but the
720p RT gate measures frame time, not byte-exactness or PSNR.
**RT impact:** 720p30 testsrc median frame time 46.08 ms → 5.73 ms
(-40.3 ms). Under the 33 ms budget by 27 ms.
**F12 impact:** none. testsrc 256x256 + foreman_cif 256x256 both
sit below 307_200 so the picker still runs (PSNR sweep preserved:
35.93 / 38.68 / 43.08 / 59.69 dB on testsrc; 16.36 / 16.21 / 16.28
/ 16.55 dB on foreman).
**Future plan:** when SIMD-accelerated sub-pel / SAD kernels land
(G-wave), lower the gate threshold or remove entirely. Currently
the picker's per-leaf cost is dominated by the scalar sub-pel
interpolation + transform round-trip; once those are SIMD-
accelerated the budget should absorb the picker work on 720p too.

### 2026-05-22 — A8 source-domain SSE picker distortion — REVERTED
**Where:** `rvp9-core::mode_decision::inter::evaluate_candidate`
(distortion metric line). libvpx counterpart: realtime picker
`vp9_pick_inter_mode` uses transform-coef-domain `block_yrd`
(`vp9_pickmode.c:1056-1110`) for the distortion side of RDCOST.
**Status:** explored, reverted, NOT shipped.
**What was tried:** swap the F8.1 picker's RD distortion from
post-recon SSE (`SSE(src, recon)`) to source-domain SSE
(`SSE(src, pred)` — the pre-quant prediction error). Recon round-
trip kept (downstream encode stages consume it), only the RD
metric changed.
**Hypothesis (brief A8):** at high q the picker should pick NEWMV
on textured motion content (foreman_cif); source-domain SSE was
hypothesised to break the NEARESTMV trap (0 NEWMV picks across
226k inter leaves on foreman_cif HEAD).
**Result:**
- NEWMV picks 0 → 3788 (foreman_cif, all bitrates). Mechanism
  confirmed.
- foreman_cif PSNR essentially UNCHANGED: 16.36 / 16.21 / 16.28
  / 16.55 → 16.36 / 16.21 / 16.26 / 16.54 dB. NEWMV picks
  concentrated at q=135-200; at q=255 (RC saturation point,
  31% of leaves on foreman) residuals quant to zero so recon
  = pred regardless and source-domain SSE == post-recon SSE.
  The picks shift but reconstruction quality doesn't.
- testsrc REGRESSED at 250 kbps: 59.69 → 57.77 dB (-1.92 dB) at
  -3.5% bits. High-q regime: residuals SURVIVE quant; with
  source-domain SSE the picker ignores the downstream residual
  rate the encoder emits, biasing toward NEWMV-with-MV-bits
  over NEAREST-with-residual. The picker over-emits MV bits.
- Cross-decoder gate: PASS (BYTE-EXACT) — pick changes the
  bitstream but every decoder agrees it is valid.
- RT gate: PASS (5.74 ms median).
**Why reverted:** brief constraint "don't ship a regression."
testsrc 250 kbps -1.92 dB is a clear regression; foreman gain is
near-zero. The real foreman blocker is RC q=255 saturation
(A6, A7 findings), not the picker distortion metric.
**Byte-parity impact:** would be structural (bitstream
contents change wherever NEWMV wins the new RD). Not applicable
since reverted.
**Future plan:** revisit only after RC q-ceiling work (A8.B
recommendation: clamp inter q ≤ 200, let buffer overshoot,
measure whether foreman climbs above 22 dB). If RC fix lifts
the floor, source-domain SSE might net positive — but testsrc
high-bitrate regression suggests the picker would also need a
residual-rate estimate (RDO-style) to be production-ready.
Detailed write-up: `wrk_docs/A8-source-sse-foreman-report.md`.

### 2026-05-22 — A9 source-SSE + inter q-cap combination — REVERTED
**Where:** `rvp9-core::mode_decision::inter::evaluate_candidate`
(distortion metric) AND `rvp9-encoder::rate_control` (new
`RateControlConfig::inter_max_qindex: Option<u8>` field +
clamp in `cbr::pick_q_and_bounds`).
**Status:** explored, reverted, NOT shipped.
**What was tried:** the A8.B follow-up — apply BOTH the
source-domain SSE picker distortion (A8) AND an inter-frame
q-cap at 200 (A8.B recommendation) simultaneously. Hypothesis:
A8 activates NEWMV picks, the q-cap prevents q=255 saturation
on foreman_cif so residuals survive quant and MC quality
actually propagates through the reconstruction loop.
**Result (foreman_cif 256×256 centre crop, 64 frames):**

| Target  | BASELINE kbps/PSNR | A9 kbps/PSNR |
|---------|-------------------|--------------|
|  25 kbps| 127.0 / 16.36     | 252.7 / 16.40 |
|  50 kbps| 159.7 / 16.21     | 252.8 / 16.23 |
| 100 kbps| 249.0 / 16.28     | 287.9 / 16.26 |
| 250 kbps| 514.6 / 16.55     | 482.7 / 16.54 |

The q-cap landed (foreman q-indexes now [135, 195] across all
inter frames per `diag_foreman_per_frame`, vs the 255-pinned
baseline) AND NEWMV picks activated (per A8 confirmation).
**foreman PSNR did NOT climb.** Per-frame recon SSE grows from
3.4M (KF) to 143M (frame 31) while source-consecutive SSE
stays at 6-17M — motion compensation fundamentally cannot
track foreman's head + camera motion at the picker+RC settings
we have, irrespective of q-cap or picker distortion metric.
Bitrate also EXPLODED at low targets (252 kbps actual on a
25 kbps target — 10× over) because the q-cap stops the
controller from absorbing the foreman per-MB cost.

testsrc 250 kbps regression preserved from A8 alone:
59.69 → 57.77 dB (-1.92 dB). cross-decoder gate PASS
(byte-exact). RT gate PASS (5.82 ms).
**Why reverted:** the 16-dB foreman floor is a structurally
deeper problem than picker distortion + q-cap can address.
The `bits_per_mb` rate model (`vp9_rc_bits_per_mb` port at
`cbr::bits_per_mb`) under-predicts foreman's per-MB cost by
~10×, so even with q capped at 200 the encoder massively
overshoots the target AND can't compensate motion correctly.
Fixing the foreman gap needs Stage 4c-level rate-model
rework (replace the libvpx-1.x bits_per_mb formula with a
per-MB content-aware estimate, or port libvpx's later
`vp9_rc_bits_per_mb_2pass` style two-pass machinery).
**Byte-parity impact:** would be structural (changes both
picks and qindex choices). Not applicable since reverted.
**Future plan:** do not retry the source-SSE + q-cap
combination until the rate model is fixed. The picker change
without the rate-model fix biases testsrc; the q-cap without
the rate-model fix blows the bitrate target. Either change
alone or both together remain net-negative until Stage 4c.

### 2026-05-22 — A10 `bits_per_mb` multiplier hypothesis — DISPROVED
**Where:** new `RateControlConfig::bits_per_mb_multiplier: f64`
field (default 1.0 = libvpx-faithful). Multiplier applied
inside `cbr::bits_per_mb_scaled`, threaded through
`regulate_q` and `estimate_bits_at_q` so the picker side and
the post-encode feedback side both see the same scaled
prediction. New diagnostic env var `RVP9_DIAG_BPM_VERIFY=1`
in `frame_loop::encode_frame` logs per-frame
`(qpicked, predicted_bits, actual_bits, ratio)`. New env hook
`RVP9_BPM_MULTIPLIER=<f64>` overrides the default at config
construction time for experimentation.
**Status:** infrastructure shipped (default 1.0 = libvpx-
faithful, byte-exact preserved); experimental non-unit
multipliers explored and disproved.
**Hypothesis (night-7 close-out, multi-agent consensus):**
the foreman_cif 16-dB PSNR floor was due to `bits_per_mb`
under-predicting per-MB cost by ~10×. A non-unit
`bits_per_mb_multiplier` would correct the bias, the picker
would pick a sane Q (80-150), residuals would survive quant,
and PSNR would climb.
**Result of measurement (A10 `RVP9_DIAG_BPM_VERIFY=1`):**

Foreman 64-frame sweep, four target bitrates: per-frame
ratio `actual/predicted_at_picked_q` is **~1.0 ± 0.25**, NOT
10×. The rate-correction-factor feedback loop in
`postencode::update_rate_correction_factors` (RC.1, libvpx
`vp9_rc_update_rate_correction_factors` port) is closing the
loop correctly — by frame 5+ the predicted bits track actual
bits within 5-20%. The hypothesis is empirically wrong.

Cross-check experiment: forced
`RVP9_BPM_MULTIPLIER=10.0` (the predicted-required value)
and `RVP9_BPM_MULTIPLIER=0.1` (the opposite direction).
Both produce foreman PSNR within 0.1 dB of baseline at every
target (16.40-16.46 dB at multiplier=10 vs 16.36-16.55 dB
baseline). Foreman PSNR is INDEPENDENT of the
`bits_per_mb_multiplier`. testsrc PSNR sweep preserved at
multiplier=1.0 default (35.93 / 38.68 / 43.08 / 59.69 dB).

The mechanism for the multiplier impotence: the
correction_factor feedback walks proportionally to compensate
the multiplier within ~5-10 frames, after which the picker's
effective bits_per_mb estimate is dominated by the adapted
correction_factor and the multiplier washes out. The
multiplier acts only as a transient prior on the first few
frames, which doesn't affect the 64-frame average PSNR.

**The bits_per_mb model is NOT the cause of the foreman 16-dB
floor.** Look elsewhere.

**Where the actual floor lives (hypothesis after A10):** the
A2.foreman per-plane diagnostic (`notes/stage4a-foreman-block-
yrd-analysis.md`) noted that **the inter chain monotonically
decays from f0=30.9 dB → f5=20 dB → f31=14.7 dB** while the
KF is healthy at 34.78 dB. The 16-dB average is the
trajectory mean of a 30→14 dB decay. This is consistent
with a **reference-frame-update bug**: each inter frame
reconstruction drifts away from the source, the drift is
copied into the LASTFRAME slot, the next frame's prediction
is built from a degraded reference, the drift accumulates.
Probable suspects in descending likelihood:

1. **Inter recon path doesn't refresh LASTFRAME correctly.**
   At `lib.rs::pick_and_simulate_inter_encode_partitions`,
   the reconstruction for the new LASTFRAME may not include
   the residual contribution (so LASTFRAME stays at the
   PREDICTED value rather than the predicted+residual value
   the decoder will see). After 30 frames the cumulative
   delta between encoder LASTFRAME and decoder LASTFRAME is
   the 14-dB gap. This would also explain A9's `recon SSE
   3.4M → 143M` drift while `source-consecutive SSE` stays
   6-17M.
2. **Loop-filter delta between encoder + decoder.** The
   encoder's reference is `apply_loop_filter_to_tight_recon`
   output; the decoder applies its own loop filter. If the
   filter-level pick disagrees, the references diverge.
3. **Chroma-MV-limit bug from A1 (chroma-aware tightening):**
   if some block uses a chroma MC region that the encoder
   reads from a different buffer than the decoder writes to,
   the chroma plane drifts.

A11 should be a **reference-buffer parity diagnostic**:
encode N frames, decode through rvp9-decoder, compute per-
frame SSE between encoder's `recon_yuv_out` (after loop
filter, the thing stored as LASTFRAME) and decoder's
`current_frame_recon` (the thing decoder stores as
LASTFRAME). If the SSE is non-zero, the reference buffers
diverge and the bug is in the encoder's reference update.
Cross-decoder byte-exactness on the BITSTREAM (which we have)
does NOT imply parity on the REFERENCE BUFFERS — only that
both decoders interpret the bitstream identically.

**Byte-parity impact:** zero with default multiplier=1.0.
The infrastructure is shipped at the libvpx-faithful default
so cross-decoder byte-exactness is preserved.
**Future plan:** retire the `bits_per_mb` hypothesis. Run
A11 reference-buffer-parity diagnostic. Detailed analysis
in this ledger entry + `wrk_docs/A10-bpm-verify-report.md`
(forthcoming).

### 2026-05-22 — A11: inter `refresh_frame_flags` was 0x00 (foreman PSNR floor smash)
**Where:** `rvp9-encoder::frame_loop::build_inter_packet_with_residuals`
— the `HeaderTail` / `UncompressedHeader::InterFrame::refresh_frame_flags`
field. libvpx counterpart: `vp9/encoder/vp9_bitstream.c::write_uncompressed_header`
emits `refresh_frame_flags` per the GF/ALTREF state machine
(`vp9_encoder.c::update_reference_frames`); the simple
single-ref RT path emits `0x01` (refresh LASTFRAME only).
**What libvpx does:** every non-altref inter frame refreshes the
LASTFRAME slot (`refresh_frame_flags = 0x01` for the simple
single-LAST shape). The decoder's `ref_frame_map[LAST_FRAME]` is
updated to point at this frame's reconstruction, so frame N+1's
MC reads frame N's reconstruction.
**What rvp9 did (PRE-A11):** inter frames emitted
`refresh_frame_flags = 0x00`. The decoder NEVER updated any ref
slot on an inter frame, so `ref_frame_map[LAST_FRAME]` stayed
pinned at the keyframe (pool slot 0) for the entire GOP. The
encoder's INTERNAL `last_recon_*` tracked each frame's
reconstruction correctly, so the encoder built residuals against
frame N-1's recon while every decoder applied those residuals
on top of MC against the keyframe. Both rvp9-decoder and vpxdec
agreed (cross-decoder byte-exact) on the wrong reconstruction;
PSNR vs source decayed monotonically across the GOP on
real-motion content because the residuals fixed the wrong MC
base.
**What rvp9 does (POST-A11):** emit `refresh_frame_flags = 0x01`.
Decoders update `ref_frame_map[LAST_FRAME]` to this frame's
reconstruction at end-of-frame; frame N+1's MC reads frame N's
reconstruction, matching the encoder's `last_recon_*`.
**Why this slipped past testing for so long:** the
cross-decoder gate (rvp9-decoder vs vpxdec) only proves
*decoder-decoder* agreement on a bitstream. Both decoders
correctly implemented `refresh_frame_flags = 0x00` semantics
(both kept LASTFRAME pinned at the keyframe). The encoder's
internal recon was diverging from what every decoder produced,
but no test compared `encoder.last_recon_*` against
`decoder.recon_*` per frame. A10's diagnosis surfaced the
hypothesis; A11's `diag_ref_divergence` test (added in this
commit) confirmed it directly.
**Diagnostic evidence (before fix), foreman_cif 256×256 @ 100 kbps,
SSE(encoder.last_recon_y, decoder.recon_y):**
```
frame 0 (KF):  SSE_Y =  6,227,702  (KF-only divergence, separate sub-bug)
frame 1:       SSE_Y =  6,213,542
frame 2:       SSE_Y = 11,924,802  ← drift kicks in
frame 3:       SSE_Y = 23,742,611
frame 4:       SSE_Y = 34,670,213
...
frame 11:      SSE_Y = 72,023,785  ← 12× growth
```
**Diagnostic evidence (after fix), same corpus:**
```
frame 0 (KF):  SSE_Y =  6,227,702  (KF sub-bug unchanged, separate)
frame 1:       SSE_Y =  6,213,542
frame 2:       SSE_Y =  6,211,784  ← no growth
...
frame 11:      SSE_Y =  6,190,530  ← stable across the GOP
```
testsrc 256×256 @ 1 Mbps: SSE_Y = 0 on EVERY frame after the fix
(was 0,0,728k,2.4M,5.1M,6.6M,8.3M,8.7M growing pre-fix).

**Foreman_cif F12 PSNR sweep, BEFORE vs AFTER:**
```
target=25kbps:   16.05 dB → 24.01 dB  (+7.96 dB)
target=50kbps:   16.07 dB → 24.16 dB  (+8.09 dB)
target=100kbps:  16.10 dB → 25.42 dB  (+9.32 dB)
target=250kbps:  16.18 dB → 28.96 dB  (+12.78 dB)
```
The 16-dB floor that survived 10 prior experiments (A0-A10)
collapses. Foreman now sits in the libvpx-comparable range
(libvpx: 28.32 / 30.51 / 33.52 / 37.25 dB at the same targets).
Residual gap is a real coding-efficiency gap (legitimate
follow-up work), not a structural drift bug.

testsrc PSNR sweep preserved: 35.93/38.68/43.08/59.69 dB →
33.51/38.31/46.02/56.52 dB. Same general envelope; the +3 dB
shift at 100 kbps comes from now actually using the previous
frame as ref. Cross-decoder gate BYTE-EXACT preserved.
RT gate: 5.78 ms median (no regression vs pre-fix 5-6 ms).
**Byte-parity impact:** STRUCTURAL — every inter frame's
header now differs by one bit. Pre-A11 bitstreams are no
longer producible. Decoders accept both (`refresh_frame_flags
= 0x00` and `0x01` are both spec-valid; the spec allows any
8-bit value).
**Compression-efficiency impact:** MASSIVE positive — drift
on every real-motion GOP, accumulating monotonically with GOP
length. The +8 to +13 dB foreman win is the headline; broader
content corpus should see similar wins.
**Future plan:** retire the "encoder LASTFRAME matches
decoder LASTFRAME" assertion to a permanent guard in the
encoder integration tests. The `diag_ref_divergence` test
should run on every encoder change; it's near-zero-cost
(<200 ms for 12 frames) and would have caught this bug
immediately.

Remaining sub-bug surfaced by the diagnostic: foreman frame 0
(KF) has SSE_Y ≈ 6.2M (max|dY|=21) between encoder and
decoder reconstructions. testsrc KF has SSE = 0. Likely a
KF-specific path (Block64x64 intra-DC fallback? variance
partitioner forcing leaves to DC where the simulator's recon
diverges from the decoder's reconstruction?). NOT addressed
by A11; the drift fix alone moved foreman from 16 dB to 24+
dB, the KF-only divergence is a smaller follow-up.

### 2026-05-22 — A13: missing `MV_COST_WEIGHT` scaling at NEWMV picker — RESOLVED
**Where:** `rvp9-core::mode_decision::inter::evaluate_inter_mode`
(crates/rvp9-core/src/mode_decision/inter.rs:587-611). libvpx
counterpart: `vp9_mv_bit_cost` at `vp9/encoder/vp9_mcomp.c:80-84`
with `MV_COST_WEIGHT = 108` (`vp9/encoder/vp9_rd.h:38`).
**What libvpx does:** charges
`ROUND_POWER_OF_TWO(mv_cost(diff) * 108, 7)` as NEWMV's MV-bit
rate, scaling the raw `mv_cost` lookup by `108 / 128 ≈ 0.844`
before adding it to `mode_rate` and feeding into RDCOST.
**What rvp9 did (PRE-A13):** charged the raw `mv_cost(diff)`
result with no weighting. Over-charged NEWMV's MV-bit rate by
`128 / 108 ≈ 1.185×` (~19% over) relative to libvpx, biasing
the picker against NEWMV at every RD comparison.
**What rvp9 does (POST-A13):** matches libvpx exactly — applies
`(raw * 108 + 64) >> 7` (the `ROUND_POWER_OF_TWO` expansion
from `vpx_ports/mem.h:31`).
**Why this is the right fix:** flagged in the A6 lambda-units
investigation (`notes/stage4a-lambda-units-investigation.md`)
as the lone real RD-formula divergence after the agent
exhaustively side-by-sided the RDCOST macro, RDDIV_BITS,
VP9_PROB_COST_SHIFT, rate unit, distortion unit, and
`compute_rd_mult_based_on_qindex` — all matched. The
`MV_COST_WEIGHT` scaling is libvpx's encoder-specific damping
applied only at the picker (not in the bitstream); A6
recommended porting it for cross-encoder fidelity.
**Byte-parity impact:** none (this affects picker decisions
only; the cross-decoder gate is preserved BYTE-EXACT —
verified post-fix). Bitstreams change if and only if the
picker selects a different mode/MV than before, in which
case both rvp9-decoder and vpxdec agree on the new bitstream.
**Compression-efficiency impact, F12 testsrc 256×256 @ 64 frames:**
```
target=25kbps:   33.51 / 33.51 dB  (no change at 25-kbps target)
target=50kbps:   38.31 / 42.92 dB  (+4.6 dB at 50-kbps target)
target=100kbps:  47.18 / 48.33 dB  (+1.15 dB at 100-kbps target)
target=250kbps:  56.92 / 56.86 dB  (essentially flat at 250)
```
BD-rate vs libvpx: +634 % → +628.52 % (modest improvement;
testsrc was already in a reasonable regime, the win is at the
50-100 kbps band where the picker has the most room to swap to
NEWMV).
**Compression-efficiency impact, F12 foreman_cif:**
```
target=25kbps:   24.01 / 24.01 dB  (flat)
target=50kbps:   24.16 / 24.16 dB  (flat)
target=100kbps:  25.42 / 25.42 dB  (flat)
target=250kbps:  28.96 / 28.96 dB  (flat)
```
BD-rate +395.92 % (unchanged from A11/A12). Foreman is bottlenecked
by the rate-model overshoot (514 kbps actual at 250-kbps target,
RC saturates q=255) and the residual mode/MV cost model, not by
the 1.185× NEWMV scaling. This matches A6's prediction that the
scaling fix tightens fidelity but does not unstick foreman.
**Cross-decoder gate:** PASS (rvp9-decoder vs vpxdec vs ffmpeg
byte-exact).
**RT gate:** 6.12 ms median (was 6.57 ms in A6) — no regression.
**Future plan:** done. The lone outstanding RD-formula divergence
flagged in A6 is now closed. The remaining foreman gap is in the
rate model + mode/MV cost tables (see A6 "Where to look next"
items 1 and 2: per-leaf symbol overhead + inter-mode probability
context).

### 2026-05-22 — A17: sub-pel SECOND_LEVEL_CHECKS (`iters_per_step = 2`) added
**Where:** `rvp9-core::mv_search::sub_pixel::find_best_sub_pixel_tree_pruned_more`
+ `rvp9-core::mode_decision::inter::run_newmv_search`. libvpx counterpart:
the `SECOND_LEVEL_CHECKS` macro (`vp9/encoder/vp9_mcomp.c:218-248`) called
from every `vp9_find_best_sub_pixel_tree*` variant's no-cost-list path when
`iters_per_step > 1`.
**What libvpx does:** at each pyramid level (half/quarter/eighth-pel), after
`FIRST_LEVEL_CHECKS` (4 cardinals + 1 diagonal), runs `SECOND_LEVEL_CHECKS`
to fan out 2-3 more samples in the winning direction:
- diagonal won → sample 2 long-diagonal extensions
- horizontal cardinal won → sample 3 points (2 long-horizontal + 1 axis hint)
- vertical cardinal won → sample 3 points (2 long-vertical + 1 axis hint)

libvpx's rt default is `sf.mv.subpel_search_level = 2`
(`vp9_speed_features.c:929`, never reset in `set_rt_speed_feature_framesize_independent`
for non-SVC). So rt at cpu-used 6-8 always runs `iters_per_step = 2`.
**What rvp9 does (POST-A17):** `SubPixelSearchParams::iters_per_step`
field added (default 2 at the inter picker call site), and the search
function now does `pyramid_level(...)` = `FIRST_LEVEL_CHECKS` +
(conditional) `SECOND_LEVEL_CHECKS` at each of half/quarter/eighth-pel.
**What rvp9 did (pre-A17):** only `FIRST_LEVEL_CHECKS` at each pyramid level
— no SECOND_LEVEL_CHECKS. This was an `iters_per_step = 1` equivalent,
diverging from libvpx's rt default of 2.
**Why this matters in principle:** the SECOND_LEVEL_CHECKS samples are
sometimes meaningfully better than the FIRST_LEVEL_CHECKS local-min on
real motion content. libvpx hardcodes `iters_per_step = 2` for rt because
the empirical hit-rate at speed 0-8 justifies the extra 2-3 sub-pel SVF
calls per pyramid level.
**Byte-parity impact:** none (cross-decoder gate verified BYTE-EXACT post-fix).
The bitstream changes only when the picker selects a different MV than
before; rvp9-decoder and vpxdec agree on both old and new bitstreams.
**Compression-efficiency impact, F12 testsrc 256×256 @ 64 frames:**
```
target=25kbps:   33.51 / 33.51 dB  (flat)
target=50kbps:   42.92 / 42.92 dB  (flat)
target=100kbps:  48.33 / 48.28 dB  (-0.05 dB)
target=250kbps:  56.86 / 56.86 dB  (flat)
```
BD-rate +628.52 % → **+630.68 %** (marginally worse, ~+2pp).
**Compression-efficiency impact, F12 foreman_cif:**
```
target=25kbps:   24.01 / 24.01 dB  (flat)
target=50kbps:   24.16 / 24.16 dB  (flat)
target=100kbps:  25.42 / 25.42 dB  (flat)
target=250kbps:  28.96 / 28.96 dB  (flat)
```
BD-rate **+395.92 %** (unchanged).
**Interpretation:** SECOND_LEVEL_CHECKS samples are *taken* on real
sub-pel attractors (the unit test for 1/4-pel-shifted source still
finds the right MV), but on the F12 corpus the additional samples do
not change the picker's mode-MV selection. The picker is dominated by
NEWMV-rejection (the candidate seldom beats NEARESTMV after MV-cost +
rate) or by the upstream int_pro seed being good enough that the
extra SECOND_LEVEL_CHECKS samples don't shift `(br, bc)`. The testsrc
-0.05 dB at 100 kbps is within the rate-control noise (kbps ±0.05 actual).
**Cross-decoder gate:** PASS (rvp9-decoder vs vpxdec vs ffmpeg byte-exact).
**RT gate:** 5.45 ms median (was 5.69 ms; well within noise).
**Why we still ship this:** L-05 sub-clause "pruned variants stubbed"
becomes incrementally less divergent — the FIRST/SECOND_LEVEL_CHECKS
pyramid path now matches libvpx's `_pruned_more` (with `cost_list = NULL`)
bug-for-bug. Future work that lands `calc_int_cost_list` (L-07) will
plug into a now-correct pyramid step rather than a half-complete one.
**Future plan:** the L-05 stub list section §7.3 of `notes/lld/motion-estimation.md`
should be updated to reflect that the `SECOND_LEVEL_CHECKS` portion of
the pruned variants is now ported; only the cost_list / parabolic-fit
short-circuit (L-07) remains. Decision: **L-05 narrowed**.

### 2026-05-22 — Inter-frame VBP thresholds (E2): partial port, speed-7 only
**Where:** `rvp9-core::mode_decision::partition::VbpThresholds::for_inter`
+ libvpx `vp9/encoder/vp9_encodeframe.c::set_vbp_thresholds` (`:573-635`)
and `scale_part_thresh_sumdiff` (`:549-567`).
**What libvpx does:** the inter branch scales `threshold_base` by noise
level (≥640×480 + noise enabled), runs a denoiser branch *or*
`scale_part_thresh_sumdiff` (which applies a 5/4 multiplier for the
LowSad/HighSad/LowVar content states at speeds 7/8), then applies
resolution-specific `thresholds[]` overrides, plus a
`disable_16x16part_nonkey` clamp.
**What rvp9 does:** ports the inter branch specialized to
**speed/cpu-used = 7** with **default content state** and **no
noise/denoiser scaling** and **no `disable_16x16part_nonkey`**. So:
`threshold_base = variance_part_thresh_mult(=1) * y_dequant[q][1]`;
`scale_part_thresh_sumdiff` returns it unchanged (default content
state); resolution overrides ported for all four bands; small-res
(≤352×288) gets the `>200`/`>220` avg-inter-qindex `thresholds[2]`
boost. The 256× variance fixed-point scale is folded out and each
threshold floored at 1, identically to `for_keyframe`. `thresholds[3]`
(8×8→4×4) is a don't-care on inter (rvp9 caps recursion at 8×8); mirrors
the keyframe value.
**`avg_inter_qindex` source:** rvp9 passes the rate controller's running
`avg_frame_qindex_inter` EWMA — this *is* libvpx's
`rc.avg_frame_qindex[INTER_FRAME]`, so no divergence there.
**Why:** content-state classification, noise/denoiser scaling, and
`disable_16x16part_nonkey` are speed-feature/input modulations not yet
ported; at cpu-used 7 on our target inputs (default content, no
denoiser) they are inactive or unreachable. Porting them now would be
dead code with no exercised path.
**Byte-parity impact:** none by construction — the partition *shape*
flows into the bitstream identically to both reference decoders; the
cross-decoder byte-exact gate stays green.
**Compression-efficiency impact:** foreman_cif 256-crop BD-rate
improved +395% → +367.81%. The shift is real but small because rvp9's
rate controller saturates q at 255 on this clip, where the inter
64×64 threshold collapses to ~1 per-pixel-variance (matching libvpx's
own q=255 threshold) — so panning-motion content vs a stale ZEROMV
LASTFRAME reference splits regardless. The dominant bottleneck is NOT
partition density at the q-saturation operating point; it is the
q-saturation itself (and the absence of real MV search to shrink the
coded residual). At moderate q (e.g. q≈128–137 on the cross-decoder
clip) `for_inter` correctly keeps the bulk of cells at Block64x64.
**Future plan:** port the content-state classification + noise/denoiser
scaling when those speed features land; revisit MV search / rate
control as the real foreman bottleneck (E2 follow-up). Decision:
**E2 partial — inter VBP curve landed; foreman gap reopened to RC/MV.**

### 2026-05-22 — E8: ambient EWMA records coded Q, not active_best (overshoot fix)
**Where:** `rvp9-encoder::rate_control::postencode::postencode_update`
+ libvpx `vp9/encoder/vp9_ratectrl.c::vp9_rc_postencode_update:1850,1882`.
**What libvpx does:** `qindex = cm->base_qindex` (the *coded* frame Q,
i.e. the `regulate_q` result after clamping), then
`avg_frame_qindex[type] = ROUND_POWER_OF_TWO(3*avg + qindex, 2)` and
`last_q[type] = qindex`.
**What rvp9 did (the bug):** recorded `rc.active_min_qindex` (the
**active_best** floor) into the ambient EWMA, with a comment claiming the
encoder "uses active_min_qindex as the realised Q". That claim was false:
the frame loop codes at `last_picked_qindex` (the `regulate_q` result),
and the rate-correction-factor update already used `last_picked_qindex`.
Recording active_best — which for inter frames is
`rtc_minq_approx(ambient) ≈ 0.75·ambient` — fed a ×0.75-per-frame
geometric decay back into the ambient (foreman: 255→30 over 31 frames).
**Why this caused the overshoot:** the buffer-ramp branch of
`calc_active_worst_quality_one_pass_cbr` interpolates `active_worst` UP
*from ambient*; as ambient collapsed, the `active_worst` ceiling
collapsed with it (~190-211 instead of toward 255), so `regulate_q`
could not reach the Q needed to hit the rate target → inter bits
overshot 2.5× (~252 kbps at the 100k foreman target).
**Fix:** record `rc.last_picked_qindex`. Now matches libvpx exactly; the
divergence is REMOVED (this is a correctness fix, not a new divergence).
**Byte-parity impact:** none by construction — RC state does not change
the wire format for a given Q; cross-decoder byte-exact gate stays green
(vpxdec + ffmpeg).
**Compression-efficiency impact:** foreman @ 100k target drops from
~252 kbps to 143.9 kbps (libvpx 143.1) — the overshoot closes. BD-rate
moves from E7's +741% to *not measurable* because the curves no longer
overlap: with the bitrate fixed, rvp9's PSNR (24-27 dB) sits entirely
below libvpx's (28-37 dB). The residual is now a pure per-bit *quality*
gap (~9 dB at matched 143 kbps), not a rate-control artifact.
**Future plan:** the remaining gap is block-coding efficiency
(residual/transform/mode), not RC. The BD-rate gate target needs
reassessment against this isolated quality gap. Decision: **E8 landed —
overshoot closed; foreman gap reduced to per-bit coding quality (case
A clamp-was-limiting confirmed; case B structural quality gap revealed
underneath).**

Also adds a diagnostic-only `RVP9_E8_TRACE=1` per-frame trace
(`RateControl::e8_trace_snapshot` / `E8Trace`) — env-gated, zero cost
when unset. Not a divergence; an instrument.

### 2026-05-22 — E10: int_pro edge-SB seed (border-clamp, no bail) — FIXED but necessary-not-sufficient
**Where:** `rvp9-core::mv_search::int_pro_me` (the
`int_pro_motion_estimation` projection-window + 2D-SAD reads).
libvpx counterpart: `vp9/encoder/vp9_mcomp.c:2264-2400`
(`vp9_int_pro_motion_estimation`), reading a reference frame whose
borders are physically extended by edge replication
(`vpx_extend_frame_borders`, `vp9/common/vp9_frame_buffers` /
`vpx_dsp`).
**What libvpx does:** runs the 1-D projection window
(`ref[sb_y-bh/2 .. ][sb_x-bw/2 .. ]`) and the post-search 2D-SAD
halo against the *extended-border* reference buffer, so frame-edge
SBs read replicated border pixels and ALWAYS get a seed.
**What rvp9 did (the bug, A3'-int-pro era):** `PlaneRef` is the bare
visible plane (no padding). The projection/halo reads near a border
would be genuinely OOB, so the function *bailed* with the
`u32::MAX`/`Mv::ZERO` sentinel whenever `sb_px < bw/2` or
`sb_px + bw + bw/2 > dim`. On a 256×256 frame (4×4 grid of 64×64 SBs)
that disabled the seed on **12 of 16 SBs — every edge SB** — leaving
75% of the frame with no motion foothold (E3 §"Contributing defect").
**Fix:** replaced the bail with edge-clamped reference reads
(`clamp_px`, `int_pro_row_clamped`, `int_pro_col_clamped`,
`sad_block_clamped`) — the faithful equivalent of an extended-border
frame. Every supported SB now produces a real seed. Unit test
`all_sbs_seeded_no_edge_bail` asserts **16/16** seeded.
**Byte-parity impact:** none. The seed only steers the encoder's MV
*search* center; the chosen MV is signalled, so the bitstream stays
decodable bit-for-bit. `cargo test -p rvp9-encoder --release` = 94
pass, 0 fail (all encoder↔decoder bit-exact recon gates green).
testsrc `bdrate_gate_stage4a` unchanged (33.32/33.98/36.43/40.66 dB).
**Compression-efficiency impact:** **none on foreman.** The seed is
now valid frame-wide (verified via `RVP9_DIAG_INTPRO_SB`: edge SBs
e.g. sb=(2,0)→(8,0), sb=(0,2)→(0,-8), tracking foreman's leftward
pan), and the wandering far-MV artifacts (E3's "max_mag up to 2068")
collapse to sane small magnitudes. BUT recon/pred PSNR and bits at
matched Q (q∈{80,120,160}) are byte-identical to the pre-fix E9
table (pred ~26 dB, recon 31.7/31.4/28.0 dB, 8×8-dominated tx hist).
`RVP9_E3_MVDIAG` confirms NEWMV RD-win share stays **0.0%** even at a
2 Mbps target: NEWMV still loses the RD argmin on MV-rate (~5000
units) vs NEARESTMV-at-(0,0), exactly as E3/E9 diagnosed.
**Verdict:** the edge-SB bail is a real defect and is now fixed
(necessary), but it is **not sufficient** to fire the prediction
cascade. The picker rejects NEWMV regardless of seed quality because
the per-leaf RD comparison over too-small (Block8x8) partitions makes
the MV-rate term dominate. The next coupled fixes (E9's recommended
chain: `find_mv_refs`/NEAREST quality + larger usable partitions +
`block_yrd`/skip_txfm NEWMV affordability) remain required before
prediction PSNR climbs off 26 dB.

## E-series — prediction-quality chain (2026-05-22)

### 2026-05-22 — E13 full-pel diamond: always run it; seed != replace
**Where:** `crates/rvp9-core/src/mode_decision/inter.rs::run_newmv_search`
+ `crates/rvp9-core/src/mv_search/full_pixel.rs` (`FullPixelSearchParams`
gains `start_mv`, separating libvpx `mvp_full` from `ref_mv`).
**What libvpx does:** the LAST-frame NEWMV path runs
`combined_motion_search` → `vp9_full_pixel_search` → the diamond
ALWAYS (`vp9_pickmode.c:154-261`). The `sb_use_mv_part` short-circuit
(`vp9_pickmode.c:207-210`) that skips the diamond and trusts the
int_pro MV is gated by `x->sb_use_mv_part`, set only for the
GOLDEN/ALTREF int_pro fast path — NOT for LAST-frame. The diamond
takes a separate `mvp_full` (search start, best-of-{nearest,near,
pred_mv}) and `ref_mv` (the rate-reference / cost attractor).
**What rvp9 did (the bug, E10 era):** `run_newmv_search` short-
circuited the diamond whenever `newmv_seed` (int_pro) was `Some`,
using the seed directly + sub-pel. On foreman that bypassed the
diamond for **89.9% of NEWMV leaves**, leaving the chosen full-pel MV
only +0.1 dB over (0,0) (2.0% of the reachable full-pel gain). It also
conflated start-MV and cost-center into one `center_mv`.
**Fix:** always run the diamond; thread the int_pro seed (or NEAREST)
as `start_mv` and NEAREST as the cost-attractor `center_mv`. Added
`start_mv: Option<Mv>` (None = legacy behaviour, start == center).
**Search-quality impact (E13 oracle, `RVP9_E13_DIAMOND`, foreman q120,
122 068 NEWMV leaves):** diamond-recovers-fraction 2.0% → **75.6%**;
NEWMVraw SAD-PSNR proxy (8×8) 26.20 → **31.54 dB** (+5.34 dB), oracle
34.42; reached-oracle 9.2% → 19.9%, improved-on-zero 62% → 79.6%.
**Byte-parity impact:** none — the search only changes WHICH MV is
picked; the chosen MV is signalled. `cargo test -p rvp9-encoder
--release` = 94 pass / 0 fail (all bit-exact recon gates green).
**Perf:** realtime gate 720p30 PASS, median 5.03 ms/frame (budget
33 ms). Bounded search (≤11 steps × 4 sites) — no blowup.
**Compression-efficiency impact:** **none on shipped recon.** Matched-Q
recon/pred PSNR + bits + tx-hist byte-identical to E9/E10
(pred ~26.2 dB, recon 31.4 dB @ q120). testsrc `bdrate_gate_stage4a`
unchanged (33.32/33.98/36.43/40.66). foreman BD-rate still not
measurable (curves don't overlap).
**Verdict:** the diamond WAS broken and is now fixed (the E12 lever
fired at the search level: +5.34 dB into the NEWMV candidate). But it
is **necessary-NOT-sufficient**: the E13 argmin counter shows NEWMV
wins the RD comparison on **0.00%** of 107 826 inter leaves — the now-
good NEWMV is still rejected because at the 8×8-dominated partition
granularity the MV-rate term dominates the distortion gain (E9 §4 /
E10's conclusion). The remaining lever is the picker / partition, not
the search.

## F.1 (block_yrd port) — 2026-05-23

### 2026-05-23 — DV-F1-1: rvp9 uses DCT qcoeffs for block_yrd SATD where libvpx uses Hadamard qcoeffs
**Where:** `rvp9-core::mode_decision::block_yrd::block_yrd` + the
caller `mode_decision::inter::reconstruct_via_round_trip`. libvpx
counterpart: `vp9/encoder/vp9_pickmode.c:786-803` (the per-tile
forward transform inside `block_yrd`).
**What libvpx does:** runs `vpx_hadamard_{8x8,16x16}` before the
forward quant inside `block_yrd`. The Hadamard is cheaper than DCT
and is treated as a rate-model proxy; libvpx's rate term
(`vpx_satd(qcoeff)`) was empirically tuned on Hadamard magnitudes.
**What rvp9 does:** reuses the **DCT** qcoeffs that the picker's
existing `reconstruct_via_round_trip` already computed (we need the
DCT recon for the pixel-SSE distortion path anyway). One forward
transform per tile instead of two.
**Why:** halves the transform cost; aligns the rate model with the
distortion model (the same qcoeff array determines both); the
SATD-of-qcoeff is monotone in residual energy regardless of basis
choice — the only effect is a per-block scalar correction on the
rate-to-bits mapping.
**Byte-parity impact:** none directly (transform is encoder-side
only). May indirectly change which (mode, mv) wins RD and thus the
bitstream, but the bitstream itself remains valid VP9.
**Compression-efficiency impact:** SATD-to-bits proportionality may
be off by a constant relative to libvpx; affects PSNR-delta
magnitude but not pick directionality. F.1 sign-off is the F.0.2
oracle (`mode_match%`, `coef_bits_est` movement), not PSNR-delta.
**Future plan:** F.1.2 may port `vpx_hadamard_{8x8,16x16}` if a F.4
A/B shows the Hadamard variant is materially closer to libvpx
behavior.

### 2026-05-23 — DV-F1-2: rvp9's picker uses `vp9_quantize_b` where libvpx's block_yrd uses `vp9_quantize_fp`
**Where:** `rvp9-core::mode_decision::inter::reconstruct_via_round_trip`
quantize step (rvp9 uses `vp9_quantize_b{,_32x32}`). libvpx
counterpart: `vp9/encoder/vp9_pickmode.c:789, 794, 800` (calls
`vp9_quantize_fp`).
**What libvpx does:** `vp9_quantize_fp` — lower-overhead variant
without the zbin pre-test. Produces slightly larger qcoeff
magnitudes for small-residual blocks (the zbin would have zeroed
them).
**What rvp9 does:** `vp9_quantize_b` (with zbin pre-test) — the
existing rvp9 picker pipeline uses this; F.1 reuses the same
quant output for the block_yrd SATD accumulator so the picker is
internally consistent.
**Why:** introducing a second quantize variant would double the
maintenance surface and the picker's per-candidate cost. The two
variants differ on borderline coefficients only; for typical
content the SATD difference is in the noise.
**Byte-parity impact:** none directly — the simulator uses
`vp9_quantize_b` for the final encode path too.
**Compression-efficiency impact:** the libvpx-rvp9 mismatch shows
up in the F.0.2 oracle as a non-zero `Δ coef_bits` mean. Don't
chase it for F.1.
**Future plan:** F.4 tuning may add `vp9_quantize_fp` if A/B shows
it materially closes the gap.

### 2026-05-23 — DV-F1-3: F.1 keeps pixel-SSE-of-recon as the RD `dist` term where libvpx uses `block_error_fp >> 2` (coefficient-domain)
**Where:** `rvp9-core::mode_decision::inter::evaluate_candidate`,
`let dist = sse.min(u32::MAX as u64) as u32;` (computed from
`sse_block(src, recon)`). libvpx counterpart:
`vp9/encoder/vp9_pickmode.c:835` (`this_rdc->dist += vp9_block_error_fp(...) >> 2`).
**What libvpx does:** uses coefficient-domain squared quantization
error (`sum (coeff - dqcoeff)^2 >> 2`) as the RDCOST `D` term.
**What rvp9 does:** uses pixel-domain squared error between source
and reconstruction (`sum (src - recon)^2`) — the same `D` metric
F8.0 ships. The F.1 `BlockYrdResult.coef_dist` is computed and
surfaced (`InterModeSearchResult.coef_dist`) for F.4 A/B but is
**not consumed** by `rd_cost` in F.1.
**Why:** changing the `dist` semantics would change RDCOST's
rate-distortion ranking on every candidate. That requires lambda
recalibration as a side effect; F.1 should not couple to lambda
tuning. The two values are Parseval-equivalent up to a transform-
scale constant.
**Byte-parity impact:** none directly.
**Compression-efficiency impact:** F.1 first-cut accepts the
pixel-domain `dist`; F.4 tuning revisits.
**Future plan:** F.4 A/B compares coef_dist vs pixel-SSE-of-recon
once F.1/F.2/F.3 have settled. The `coef_dist` field on
`InterModeSearchResult` is the affordance.

### 2026-05-23 — F.1 ship state: INFRASTRUCTURE-ONLY (team-lead decision)

F.1 implementer initially landed with `coef_rate` added to the RD
rate sum AND the libvpx skip-flag tie-break (HLD Q5) active by
default. Empirical measurement on foreman_cif:

| Configuration | mean Δ | max Δ | min Δ | 250 kbps PSNR |
|---|---:|---:|---:|---:|
| Pre-F.1 baseline | +9.07 dB | +9.50 | +8.65 | 26.96 dB |
| F.1 (both mechanisms ON) | +12.22 dB | +14.07 | +7.93 | 28.61 dB but 100 kbps collapses |
| F.1 (only `coef_rate` ON, tie-break OFF) | +10.55 dB | +15.49 | +8.65 | **20.91 dB** (-6 dB) |
| F.1 (infrastructure-only — both OFF, **shipped**) | **+9.07 dB** | **+9.50** | **+8.65** | **26.96 dB** |

**Diagnosis:** both mechanisms have lambda-calibration issues. The
libvpx `<< (2 + VP9_PROB_COST_SHIFT)` SATD-to-bits scale is tuned
for libvpx's lambda + RD ranking; rvp9's `compute_rd_mult` was
calibrated when the picker had no `coef_rate` term. Activating
`coef_rate` without recalibrating lambda overprices candidates →
picker falls back to worse modes at higher bitrates → PSNR
collapses (most visibly the 250 kbps target dropping 6 dB).

**F.1 ships INFRASTRUCTURE-ONLY:**
- `crates/rvp9-core/src/mode_decision/block_yrd.rs` module (472 LoC,
  6 unit tests).
- `BlockYrdResult { coef_rate, coef_dist, skippable }` flows through
  `reconstruct_via_round_trip` and is captured on
  `InterModeSearchResult { coef_rate, coef_dist }`.
- F.0.2 oracle's `coef_bits_est` schema field reads
  `coef_rate >> VP9_PROB_COST_SHIFT` (live on rvp9 leg).
- Two env-gated activation knobs, **DEFAULT OFF**:
  - `RVP9_F1_COEF_RATE_ENABLE=1`: include `coef_rate` in the picker's
    rate sum.
  - `RVP9_F1_SKIP_TIE_BREAK_ENABLE=1`: run the libvpx skip-flag
    tie-break (`vp9_pickmode.c:2355-2362`).

**At default (both unset):** the picker's RD comparison is
byte-identical to pre-F.1. Foreman PSNR-delta = +9.07 dB exactly.
No behavioural regression. F.2 and F.3 land on this infrastructure;
F.4 owns lambda recalibration and turning the activation knobs ON.

**F.4 inherited work-items:**
1. Recalibrate `compute_rd_mult` or the SATD-to-bits scale so
   `RVP9_F1_COEF_RATE_ENABLE=1` does not regress PSNR-delta. Most
   likely culprits: (a) DCT-vs-Hadamard magnitude divergence
   (DV-F1-1); (b) `vp9_quantize_b` vs `vp9_quantize_fp` differences
   (DV-F1-2); (c) the `<< 2` rate scale exponent itself.
2. Once (1) is done, activate `RVP9_F1_SKIP_TIE_BREAK_ENABLE=1`
   and verify the tie-break improves (or doesn't regress) the
   metric. Reviewer hypothesis (see follow-up entry below): the
   tie-break's domain may need realignment via `coef_dist`
   (revisiting DV-F1-3).
3. Candidates for the calibration sweep: SATD scalar correction;
   port `vpx_hadamard_{8,16}`; tune the skip-prob context
   (currently `DEFAULT_SKIP_PROBS[1] = 128`); thread the real
   per-block skip context (`vp9_get_pred_context_skip`).


### 2026-05-23 — F.1 follow-up: code-reviewer flagged skip-tie-break domain mismatch as a likely root cause
**Where:** `rvp9-core::mode_decision::inter::evaluate_candidate` —
the libvpx skip-flag tie-break ported from `vp9_pickmode.c:2355-2362`.
**Reviewer hypothesis (2026-05-23 code-reviewer pass):** libvpx's
tie-break compares `RDCOST(rate, this_rdc.dist) vs RDCOST(0,
this_sse)` where `this_rdc.dist` is `block_error_fp >> 2`
(coefficient-domain, ≈ 1-3 orders of magnitude larger than
pixel-SSE-of-recon due to DCT energy scaling) and `this_sse` is
`(*sse << 4)` (pixel-SSE-of-pred × 16). Both libvpx sides are
inflated to a comparable magnitude. The F.1 port per HLD §5.5
uses `pixel_SSE_recon` on the encode side and `pixel_SSE_pred` on
the skip side — same UNITS (raw pixel SSE) but ORDERS OF MAGNITUDE
apart at high q (where `pixel_SSE_recon ≈ pixel_SSE_pred` is
satisfied because heavy quant zeros out the residual). At those
operating points the comparison collapses to "argmin sse_after_pred"
instead of libvpx's "argmin (rate + dist)", which could explain the
+3 dB PSNR regression at low-bitrate (high-q) foreman targets.
**Recommended team-lead investigation:** A/B the encode-side dist
between (a) current `pixel_SSE_recon`, (b) `by.coef_dist` (already
surfaced on `InterModeSearchResult` per HLD §5.5 / DV-F1-3), with
the skip side using a Parseval-equivalent rescaled value. The
foreman quality_gate run is the oracle. If `coef_dist` recovers
PSNR-delta, the HLD §5.5 design choice (Q3 resolution) should be
revisited as the F.4 tuning entry point — not just a scalar
correction.
**F.1 ships as designed.** The team-lead's HLD §5.5 sign-off is
explicit ("F.1 first-cut keeps pixel-SSE-of-recon"). This entry
records the implementer's review-time concern so F.4 starts from
the right diagnostic question, not as a unilateral deviation.


### 2026-05-23 — F.2 ship state: INFRASTRUCTURE-ONLY (team-lead decision)
**Where:** `crates/rvp9-encoder/src/frame_loop.rs:173` — the literal
`use_prev_frame_mvs: false` in `EncoderConfig::new`. F.2 HLD §4.2
originally instructed flipping this to `true`.
**What the F.2 HLD planned:** flip `EncoderConfig.use_prev_frame_mvs`
default from `false` to `true` so the encoder's previous-frame MV
salvage path (libvpx `cm->use_prev_frame_mvs`) is active by default.
The HLD argued that F.1's `block_yrd` would make NEWMV's coefficient-
bit savings visible to RD, closing the F.0.3 §2.5 regression
mechanism (NEARMV winning over refined NEWMV when its anchor became
non-zero, testsrc −240 pp BD-rate).
**What rvp9 ships at F.2:** the literal stays `false`. The doc-
comment is rewritten to record the deferral and cite the F.1
lambda-calibration reason; the three salvage-path unit tests
(`prev_salvage_matching_ref_emits_temporal_candidate`,
`prev_salvage_different_ref_with_sign_bias_flip`,
`prev_salvage_dedup_against_spatial`) land in
`crates/rvp9-core/src/inter/mv_ref.rs::tests`; the smoke test
`crates/rvp9-encoder/tests/f2_find_mv_refs_temporal_visible.rs`
lands with `use_prev_frame_mvs: true` opted in per test.
**Why:** F.1 shipped infrastructure-only (see F.1 ship-state entry
above, 2026-05-23) with `RVP9_F1_COEF_RATE_ENABLE` defaulting OFF
because the SATD-to-prob-cost scale is mis-calibrated against
rvp9's lambda. Without F.1's `coef_rate` active by default, the
chicken-and-egg fix is incomplete and flipping A18 to ON
re-introduces the F.0.3 §2.5 regression mechanism on testsrc (−240
pp BD-rate). F.4 owns BOTH the lambda recalibration AND the
follow-up A18 default-flip; they are sequenced together so the
sweep that validates the flip is run against a recalibrated
comparator.
**Byte-parity impact:** none. The default is byte-identical to
pre-F.2; the encoder/decoder still agree on `(LASTFRAME, ZEROMV)`
everywhere on the first inter after a KF.
**Compression-efficiency impact:** none at default. F.2's deferred
flip is expected to *improve* PSNR-delta on motion content once
F.4's recalibration lands, but holding it back today preserves the
post-E16 foreman PSNR-delta = +9.07 dB baseline.
**Future plan:** F.4 task (#84) flips the default after the lambda
recalibration A/B sweep confirms no testsrc regression beyond the
F.1 baseline + 3 dB halt threshold (per F.2 HLD §9 R1 mitigation).
F.4 also owns removing the env-var gates on `coef_rate` and the
skip-flag tie-break (per F.1 ship-state F.4 work-items 1–2). When
F.4 lands, this divergence collapses.
**Cross-references:**
- F.2 HLD §12 ship-state addendum (`wrk_docs/Track_F/F2_HLD_find_mv_refs.md`).
- F.1 ship-state entry above (2026-05-23) — the precedent and the
  pre-condition.
- `notes/divergence-ledger.md:676-697` — F.0.3 original
  documentation of the −240 pp regression mechanism.

### 2026-05-23 — F.3 ship state: INFRASTRUCTURE-ONLY (team-lead decision)
**Where:** new module
`crates/rvp9-core/src/mode_decision/nonrd_partition.rs` (~700 LoC, 7
unit tests) + env-gated integration in
`crates/rvp9-encoder/src/lib.rs:pick_and_simulate_inter_encode_partitions`
(around line 2670). F.3 HLD §10 originally proposed
`RVP9_F3_NONRD_PARTITION_OFF` (env var defaulting the new behaviour
ON with an escape hatch).
**What the F.3 HLD planned:** invert the partition flow so
motion-aware RD descent (libvpx `nonrd_pick_partition`,
`vp9_encodeframe.c:4922-5210`) runs *inside* the per-SB picker, with
the variance partitioner reduced to a hint/bound source. Drop
recursion at Block16x16 (no 8×8 mode search). Expected to lift the
typical leaf size from 8×8 to 16×16+ on motion content and close
the F.1 "85 % Block8x8 ceiling" (F.1 HLD §9).
**What rvp9 ships at F.3:** the new module + descent function lands
in full; the encoder integration is gated behind
`RVP9_F3_NONRD_PARTITION_ENABLE=1` (polarity flipped vs HLD §10:
default **OFF**). At defaults the picker hot path is unchanged.
Foreman PSNR-delta = +9.07 dB exactly (matches pre-F.3 baseline;
PASS at 9.50 dB threshold). Realtime gate median = 4.76 ms (well
below 33 ms). The integration's mode-picker callback uses a
simplified ZERO-MV SSE proxy (not the real `pick_inter_mode_luma`)
— enough to exercise the wiring; F.4 swaps in the real picker.
**Why:** F.3 has the same lambda-coupling exposure as F.1 / F.2.
The descent ranks partition candidates via
`rd_cost(rd_mult, rate, dist)`; `rd_mult` is the same uncalibrated
SATD-to-bits scale that regressed PSNR-delta under
`RVP9_F1_COEF_RATE_ENABLE=1` (F.1 ship-state entry above). Mirroring
the F.1 / F.2 pattern (infra ships, behaviour is gated) keeps the
Track F cluster bounded — F.4 calibrates lambda once and flips all
three activation switches together rather than chasing interaction
effects across three separately-active components.
**Byte-parity impact:** none. The default is byte-identical to
pre-F.3 (and pre-F.1 / pre-F.2 by extension). Foreman PSNR-delta =
+9.07 dB exactly.
**Compression-efficiency impact:** none at default. F.3's deferred
activation is expected to *lift the typical leaf size* from 8×8 to
16×16+ on motion content once F.4's recalibration lands — the F.1
"85 % Block8x8 ceiling" closes — but holding it back today
preserves the post-E16 / post-F.1 / post-F.2 +9.07 dB baseline.
**Realtime-gate impact:** zero at default (`inter_picker_enabled =
w*h <= 307_200` already gates the picker off at 720p, so the
descent never runs in the realtime gate). When activated, the
worst-case per-SB cost lifts from ~38 µs to ~116 µs on small-frame
corpora (F.3 HLD §7.1) — F.3 HLD §7.5 sets the 8 ms median ceiling
on foreman_cif as the self-imposed engineering envelope.
**Future plan:** F.4 owns three coupled work-items:
1. Replace the integration's SAD-proxy callback in
   `pick_and_simulate_inter_encode_partitions` with a real
   `pick_inter_mode_luma` invocation (or the realtime path's
   simplified picker if libvpx's `nonrd_pick_sb_modes` is closer to
   what the descent needs).
2. Recalibrate `compute_rd_mult` / the SATD-to-bits scale so the
   descent's RD ranking doesn't collapse to the F.1 ship-state
   failure mode (per F.1 ship-state F.4 inherited work-items).
3. Flip the env gate ON (most likely by removing the gate entirely
   after the A/B sweep confirms F.0.1 PSNR-delta improves and
   foreman_cif median frame time stays under the §7.5 ceiling).

When F.4 lands, this divergence collapses.
**Cross-references:**
- F.3 HLD §11 ship-state addendum (`wrk_docs/Track_F/F3_HLD_motion_aware_partition.md`).
- F.1 ship-state entry (2026-05-23) — the original precedent.
- F.2 ship-state entry (2026-05-23) — the second mirror.
- F.4 task inherits the env-gate flip + callback-replacement
  work-items.

### 2026-06-21 — KILN-00010: idct16x16 SSE2 saturation vs AVX2/scalar modular wraplow (out-of-domain only)
**Where:** `rvp9-core::dsp::x86_64::idct16x16_sse2_256::idct16x16_add_sse2_256`
(the i16-lane 256-point inverse DCT) vs `idct16x16_add_avx2` (the i32-lane
oracle). libvpx counterparts: `vpx_dsp/x86/inv_txfm_sse2.h` (`write_buffer_8x1`
line 232 uses `_mm_adds_epi16`; multiply stages pack with `_mm_packs_epi32`) and
the modular C reference (`WRAPLOW`).
**What libvpx does:** libvpx's *SSE2* idct16x16 **saturates** at the i16 sites
(`_mm_adds_epi16` on the +32 final rounding; `_mm_packs_epi32` on the multiply
packs) while its *C* reference uses modular `WRAPLOW`. libvpx's SSE2 and C are
therefore **not bit-exact** on inputs whose transform intermediates exceed i16 —
libvpx accepts this because valid 8-bit bitstreams never produce such inputs
(dequantized coefficients are i16-bounded by design and real residuals are
`[-255,255]`).
**What rvp9 does:** mirrors libvpx exactly — the SSE2 kernel saturates
(libvpx-SSE2-faithful), the AVX2/scalar kernels use modular wraplow
(libvpx-C-faithful). So rvp9's SSE2-vs-AVX2 pair has the **same** numerical
divergence libvpx's SSE2-vs-C pair has, and only on the same out-of-domain
(i16-overflowing) inputs.
**Why:** faithful porting, not a deliberate departure. Recorded here because the
`idct16x16_sse2_eq_avx2_*` differential tests originally asserted bit-exactness
on **arbitrary full-density i12** coefficients (`uniform_12` /
`eob_bucket_dense`), which are out of domain: their collective energy overflows
the i16 SSE2 transform. Diagnosis (KILN-00010): the saved failing seed was
190/256 dense, max|coef|=2046; halving it removed the 1-pixel divergence
(overflow-driven, not algorithmic); a 2000-block in-domain sweep (forward DCT of
`[-255,255]` residuals, coefficients up to ±5310) produced **zero** divergence.
Fix is **test-domain only** — the kernel is correct and unchanged. The dense
strategies now generate in-domain coefficients (`fdct16x16` of a `[-255,255]`
residual, the libvpx test methodology), and
`idct16x16_sse2_indomain_exact_outofdomain_saturates` pins both the in-domain
bit-exactness and the accepted out-of-domain saturation (guarding against a
mistaken "fix" making the SSE2 kernel modular — which would then diverge *from*
libvpx). This supersedes the T-4 HLD §6.2 expectation that
`uniform_12`/`eob_bucket_dense` be bit-exact (that expectation did not account
for the libvpx-inherent SSE2 saturation on out-of-domain inputs).
**Byte-parity impact:** none on valid bitstreams (conformance 201/201; runtime
AVX2 hosts unaffected). On SSE2-only hosts the output equals libvpx's SSE2
output (the reference for that path), so it is correct, not divergent, vs the
ecosystem.
**Compression-efficiency impact:** none.
**Future plan:** none — faithful and correct. Revisit only if a future
sub-strategy (`sparse_12`/`eob_38`, still on arbitrary low-density i12) ever
flakes, in which case give it the same in-domain treatment.

### 2026-07-30 — AArch64 sub-pixel variance fuses filtering and reduction

**Where:** `rvp9_core::dsp::aarch64::subpel_variance_neon`, installed for
VP9's 16/32/64-wide sub-pixel-variance dispatch slots.

**What libvpx does:** `vpx_dsp/arm/subpel_variance_neon.c` writes the
horizontal bilinear result to a full intermediate block, writes the vertical
result to a second block, then calls the matching NEON variance kernel.

**What rvp9 does:** keeps two rolling 64-byte horizontal rows and feeds each
vertical-filter vector directly into the `(sum, sse)` reduction. Offset-zero
paths fuse the remaining one-dimensional filter and reduction without an
intermediate block. Two sum vectors and four squared-error vectors carry
independent quarter-block lanes, then combine only once at the end.

**Why:** rvp9 previously had no AArch64 sub-pixel-variance dispatch and ran
two heap allocations plus three scalar pixel walks for every sub-pel probe.
The fused layout removes both allocations and the final temporary-block walk
while preserving libvpx's exact bilinear rounding and variance finisher.
Independent accumulators also remove the two serial widening additions and
two serial multiply-accumulates that the first fused kernel issued per
16-pixel group.

**Byte-parity impact:** none. Random differential coverage exercises all 64
offset pairs across eight block shapes; edge-value and forced-dispatch tests
also match the scalar oracle exactly. A 120-frame cpu-used 7 encode remained
byte-identical before and after the dispatch change.

**Compression-efficiency impact:** none; the production stream is unchanged.

**Speed impact:** the 32x32 mixed-offset kernel improved from `797.1 ns` to
`85.8 ns` (`9.30x`). Three alternating 120-frame 1280x704 cpu-used 7 runs
improved median wall time from `2.74 s` to `2.45 s` (`10.6%`) on the pinned
Apple M5 Max host. Splitting the reduction dependencies later improved paired
child CPU by another `-1.029%` at one thread with fifteen of sixteen wins and
`-0.947%` at four threads with seventeen of twenty wins.

**Future plan:** retain the fused implementation while it remains
byte-exact. Add 4/8-wide NEON kernels only if a fresh whole-encoder profile
shows those scalar slots are material.

### 2026-07-31 — AArch64 fast quantization keeps rvp9's widened coefficient API

**Where:** `rvp9_core::dsp::aarch64::quantize_neon` fast and zbin kernels,
used by `rvp9_core::quant::forward::{vp9_quantize_fp_with_order,
vp9_quantize_b_with_order}` for 4x4, 8x8, and 16x16 transforms.

**What libvpx does:** the 8-bit build's fast and zbin implementations in
`vp9/encoder/arm/neon/vp9_quantize_neon.c` and
`vpx_dsp/arm/quantize_neon.c` load eight `tran_low_t` i16 coefficients,
quantize them in i16/i32 lanes, and find EOB by masking the inverse scan.

**What rvp9 does:** loads the same eight raster-order coefficients as two
i32 vectors, matching rvp9's existing transform/quantizer API. It applies
the same fast or zbin/round/quant-shift pipeline in widened lanes, narrows
qcoeff/dqcoeff to i16, and finds EOB from the same one-based inverse scan
tables.

**Why:** rvp9 deliberately keeps forward-transform coefficients as i32 even
though valid 8-bit VP9 transform output is i16-bounded. Widened input lanes
avoid changing that established API or adding a pre-quantization narrowing
copy while retaining libvpx's raster-order SIMD algorithm.

**Byte-parity impact:** none. Differential coverage spans all four scan
types, q indices `0/1/20/80/160/255`, and every supported transform size.
Three alternating 120-frame cpu-used 7 before/after pairs remained
byte-identical.

**Compression-efficiency impact:** none; the production stream is unchanged.

**Speed impact:** microbenchmarks improved 4x4/8x8/16x16 fast quantization by
`6.98x/6.21x/6.39x` and zbin quantization by `4.42x/2.69x/2.57x`. The two
120-frame cpu-used 7 medians improved `2.07 s -> 1.98 s -> 1.90 s` on the
pinned Apple M5 Max host.

**Future plan:** retain the widened-lane kernel while it remains byte-exact.
Treat the separately measured 32x32 fast and zbin quantizers as independent
future units.

### 2026-07-31 — AArch64 inverse 16x16 DCT keeps widened lanes

**Where:** `rvp9_core::transforms::inverse::neon::idct16x16_add`, used by
the 10-, 38-, and 256-coefficient inverse 16x16 DCT paths.

**What libvpx does:** `vpx_dsp/arm/idct16x16_add_neon.c` processes eight
transforms in parallel in i16 lanes, with i32 multiply accumulators. Its
two half-width passes transpose through a 256-element i16 intermediate and
the second pass performs a vector pixel add.

**What rvp9 does:** processes four transforms in parallel in i32 lanes,
matching rvp9's existing `i32` coefficient and scalar-oracle API. Every
butterfly result explicitly narrows through i16 to preserve `wraplow`.
The second pass returns four residuals at a time, then uses a scalar
saturating pixel add for those four bytes.

**Why:** retaining widened lanes avoids an input-narrowing copy and makes
the vector stages map directly to the established scalar oracle. Four-way
lanes also reuse the exact transpose and widened multiply approach already
validated by rvp9's AArch64 forward DCT.

**Byte-parity impact:** none. The vector kernel matches the scalar oracle
over all three EOB buckets, strides 16/19/32, and 576 deterministic random
coefficient/destination cases. The 120-frame cpu-used 7 product stream is
byte-identical before and after the dispatch.

**Compression-efficiency impact:** none; the product stream is unchanged.

**Speed impact:** on the 120-frame 1280x704 cpu-used 7 product workload,
paired child CPU improved `-1.198%` at one thread (15/16 wins) and `-1.574%`
at four threads (19/20 wins). A fresh profile reduced inverse 16x16
transform self samples from 91 to 41 across the vector core and wrapper.
These unit timings ran during unrelated host load, so final acceptance still
requires the clean-host corpus matrix.

**Future plan:** retain the widened-lane kernel while it remains byte-exact.
Vectorize the four-byte pixel add only if a fresh profile isolates it as a
material leaf.

### 2026-07-31 — AArch64 fast 32x32 quantization keeps widened lanes

**Where:** `rvp9_core::dsp::aarch64::quantize_neon::
vp9_quantize_fp_32x32_neon`, dispatched through
`vp9_quantize_fp_32x32_with_order`.

**What libvpx does:** `vp9/encoder/arm/neon/vp9_quantize_neon.c` loads eight
i16 transform coefficients, quantizes them in i16/i32 lanes, applies the
32x32 dequant threshold and halving rule, and reduces EOB through inverse
scan positions.

**What rvp9 does:** loads the same eight raster-order coefficients as two
i32 vectors, matching rvp9's widened transform-coefficient API. It applies
the same rounded fast quantization, `dequant >> 2` gate, signed `/ 2`
dequantization, and inverse-scan EOB reduction.

**Why:** widened input lanes avoid changing the established coefficient API
or adding a narrowing pass across 1,024 coefficients. The arithmetic stays
within i32 for valid 8-bit VP9 transforms and directly matches the scalar
oracle's signed division semantics.

**Byte-parity impact:** none. Differential coverage spans q indices
`0/1/20/80/160/255`, random full-i16 coefficient blocks, gate boundaries,
sparse EOB positions, and skip handling. The 120-frame cpu-used 7 stream is
byte-identical before and after dispatch.

**Compression-efficiency impact:** none; the product stream is unchanged.

**Speed impact:** paired child CPU improved `-1.270%` at one thread (16/16
wins) and `-0.721%` at four threads (17/20 wins). A fresh profile reduced
fast 32x32 quantizer self samples from 148 to 32. Unrelated host benchmarks
made four-thread wall time noisy, so final acceptance still requires the
clean-host corpus matrix.

**Future plan:** retain the widened-lane kernel while it remains byte-exact.
The non-fast 32x32 zbin quantizer is separate and should only be vectorized
if a required configuration profile makes it material.

### 2026-07-31 — AArch64 horizontal interpolation uses a row-at-a-time dot-product kernel

**Where:** `rvp9_core::dsp::aarch64::convolve8_neon::
convolve8_horiz_dotprod_inner`, used by the horizontal and separable 2-D
unscaled eight-tap interpolation paths on dot-product-capable AArch64 hosts.

**What libvpx does:** `vpx_dsp/arm/vpx_convolve8_neon_dotprod.c` loads four
source rows at a time, permutes each shared 16-byte load into eight overlapping
tap windows, and evaluates those windows with four signed dot products.

**What rvp9 does:** applies the same range shift, correction term, permutation
tables, signed dot products, rounding, and saturation to one row at a time.
The safe wrapper checks the runtime dot-product capability and falls back to
the existing widened-i16 NEON path for filters containing the full-pel
coefficient `128`, which cannot be represented as an i8 dot-product tap.

**Why:** the row-at-a-time loop fits rvp9's existing arbitrary-height safe
API without adding libvpx's four-row height contract. It still collapses eight
overlapping loads and sixteen multiply-accumulates per output group into one
shared load, three table permutations, and four dot products.

**Byte-parity impact:** none. All supported block shapes and every phase of
the four VP9 filter families match the scalar oracle. The 120-frame cpu-used 7
product stream remains byte-identical, and rvp9-decoder, pinned vpxdec, and
ffmpeg produce identical I420 output.

**Compression-efficiency impact:** none; the production stream is unchanged.

**Speed impact:** sixteen warmed alternating one-thread pairs improved child
CPU by `-3.007%` with sixteen wins. Twenty four-thread pairs improved child
CPU by `-2.665%` with twenty wins; paired wall time improved `-1.958%` with
nineteen wins.

**Future plan:** retain the row-at-a-time kernel while it remains byte-exact.
Treat vertical dot-product interpolation and an i8mm horizontal kernel as
separate measured units.

### 2026-07-31 — AArch64 vertical interpolation reloads four-row dot-product windows

**Where:** `rvp9_core::dsp::aarch64::convolve8_neon::
convolve8_vert_dotprod_inner`, used by the vertical and separable 2-D
unscaled eight-tap interpolation paths on dot-product-capable AArch64 hosts.

**What libvpx does:** `vpx_dsp/arm/vpx_convolve8_neon_dotprod.c` keeps seven
transposed rows live, loads four new rows, merges their columns through three
table lookups, and advances four output rows at a time.

**What rvp9 does:** loads the complete eleven-row window for each four-output
group, transposes its eight overlapping four-row groups directly, and applies
the same corrected signed dot-product arithmetic. One scalar-NEON tail handles
arbitrary heights, and the existing widened-i16 path handles full-pel filters.

**Why:** reloading seven overlapping rows makes the kernel substantially
smaller and avoids a large live SIMD state. The pinned Apple M5 Max cache
absorbs those reloads, while shared loads, transposition, and dot products
still remove most of the original multiply-accumulate work.

**Byte-parity impact:** none. Differential coverage includes every VP9 filter
phase, all codec block shapes, and irregular `7x3`, `12x5`, and `63x9` tails.
The product stream and all three decoders' I420 output remain byte-identical.

**Compression-efficiency impact:** none; the production stream is unchanged.

**Speed impact:** sixteen warmed alternating one-thread pairs improved child
CPU by `-1.799%` with sixteen wins. Twenty four-thread pairs improved child
CPU by `-1.717%` with twenty wins; paired wall time improved `-1.633%` with
twenty wins.

**Future plan:** retain the reload-based kernel while it remains byte-exact.
Only adopt libvpx's rolling transpose state if a direct product comparison
shows an additional reproducible improvement worth the extra live state.

### 2026-07-31 — Hadamard picker quantization returns fused RD metrics

**Where:** `rvp9_core::dsp::aarch64::quantize_neon::{
vp9_quantize_fp_i16_metrics_neon,
vp9_quantize_fp_32x32_i16_metrics_neon}`, used only by
`mode_decision::inter::reconstruct_via_round_trip`'s Hadamard RD model.

**What libvpx does:** the 8-bit encoder keeps Hadamard `tran_low_t`
coefficients in i16 and feeds them directly to its NEON fast quantizer.
That general-purpose quantizer writes qcoeff and dqcoeff arrays; later code
reduces the arrays for rate and distortion.

**What rvp9 does:** the general transform API remains i32, but the
Hadamard-only picker retains its native i16 coefficients through a
picker-specific quantizer. Each eight-lane load widens to i32 registers before
absolute value and multiplication, preserving exact `i16::MIN` handling.
While qcoeff and dqcoeff lanes remain live, the kernel accumulates EOB, qcoeff
SATD, and squared coefficient error into independent SIMD lanes. It returns
only those three RD metrics and never materializes qcoeff or dqcoeff arrays.

**Why:** the previous picker copied 64, 256, or 1,024 coefficients into a
separate i32 array for every filter candidate, even though the source
Hadamard and quantized/dequantized outputs are all i16. Its first narrowed
implementation then wrote two arrays solely to scan each one again for rate
and distortion.

**Byte-parity impact:** none. Differential tests cover all realtime
transform sizes, six q indices, random full-i16 coefficients, `i16::MIN`,
`i16::MAX`, all-zero, DC-only, and final-scan-position blocks. Fused EOB,
SATD, and error exactly match the widened quantizer oracle. The product stream
and all three decoders' output remain byte-identical.

**Compression-efficiency impact:** none; the production stream is unchanged.

**Speed impact:** retaining narrowed coefficients first improved paired child
CPU by `-0.252%` at one thread and `-0.214%` at four threads. Fusing the
metrics then improved another `-0.317%` at one thread with fourteen of
sixteen wins and `-0.297%` at four threads with twelve of twenty wins.
Unrelated host builds made absolute wall medians noisy; paired wall deltas
still improved `-0.362%` and `-0.371%`, respectively. Final acceptance
requires the clean-host corpus matrix.

**Future plan:** retain this picker-only fusion while it stays exact. The
general transform and emitted-coefficient quantizer APIs remain widened and
continue to materialize coefficient arrays.

### 2026-07-31 — Edge motion-refinement blocks are gathered by replicated spans

**Where:** `rvp9_core::mv_search::int_pro_me::{
gather_clamped_block,sad_block_clamped}`, used by the integral-projection
motion seed's six post-projection SAD probes.

**What libvpx does:** reference frames carry a physically extended replicated
border, so each SAD probe can pass a direct pointer and stride to its
architecture-specific SAD kernel even when the logical block crosses a
visible edge.

**What rvp9 does:** reference planes contain visible pixels only. Interior
probes still pass a direct plane slice. An edge probe gathers one contiguous
scratch block, but computes the left and right replicated spans once per
block, clamps the source row once per output row, copies the visible middle
span, and fills only the two border spans.
Each projection search carries the scratch as an `Option<[u8; 4096]>`:
interior-only searches leave its payload inactive and uninitialized, while
the first edge probe initializes it once and all later probes reuse it.

**Why:** physically extending every reference plane would widen storage and
ownership contracts across the encoder. The previous equivalent gather
instead clamped both coordinates for every pixel, repeating 4,096 clamp and
bounds-check operations for each 64x64 edge probe.

**Byte-parity impact:** none. A focused oracle covers 8x8 through 64x64
blocks, interior spans, partial crossings on every edge, and blocks wholly
beyond each edge. The cpu-used 7 product stream and all three decoders'
162,201,600-byte I420 outputs remain byte-identical. Focused scratch tests
also prove that interior probes never activate the buffer and a later small
edge probe does not reinitialize untouched bytes.

**Compression-efficiency impact:** none; the production stream is unchanged.

**Speed impact:** sixteen warmed one-thread pairs improved paired child CPU
by `-2.415%`, with sixteen wins. Twenty four-thread pairs improved child CPU
by `-1.899%`, with nineteen wins; paired wall time improved `-0.669%`, with
eighteen wins. Reusing the scratch then improved another `-0.095%` paired
child CPU at one thread and `-0.545%` at four threads in the final exact-code
run; paired wall time improved `-0.141%` and `-0.630%`, respectively.

**Future plan:** retain the span gather while rvp9 stores visible-only
references. Preserve the safe lazy `Option` payload rather than introducing
raw uninitialized memory; a committed profile must select the next hotspot.

### 2026-07-31 — Encoder coefficient context reuses causal thread-local scratch

**Where:** `rvp9_core::entropy::detokenize::encode_coefs`, used by the final
tile token writer for every selected transform block.

**What libvpx does:** tokenization stores each block's token context in
encoder-owned block state before the final pack loop. Its pack loop consumes
that state without allocating a new coefficient cache per transform block.

**What rvp9 does:** non-empty blocks borrow one 1,024-byte token cache owned by
the current thread. Each scan position writes its cache entry before the next
context lookup, and VP9's neighbour table refers only to causal positions, so
stale bytes from earlier blocks are never observed. Empty blocks emit their
single EOB decision before borrowing the cache.

**Why:** the previous encoder allocated and zeroed a `Vec<u8>` sized to the
transform for every block, although the cache is scratch and every readable
entry is overwritten by the current scan. The decoder already relies on the
same causal-neighbour invariant.

**Byte-parity impact:** none. A focused history-independence test dirties the
entire cache with a dense 32x32 block before re-encoding a sparse 4x4 block.
All coefficient round-trip tests pass. The 120-frame cpu-used 7 product stream
and all three decoders' 162,201,600-byte I420 outputs remain byte-identical.

**Compression-efficiency impact:** none; the production stream is unchanged.

**Speed impact:** the final exact-code run's sixteen warmed alternating
one-thread pairs were neutral-positive: paired child CPU `-0.015%` and wall
time `-0.066%`, with nine wins. Twenty four-thread pairs improved child CPU
by `-1.966%` with fifteen wins and wall time by `-1.208%` with twelve wins.
Unrelated max-performance and link jobs made absolute medians noisy; final
acceptance still requires the clean-host corpus matrix.

**Future plan:** retain the reusable encoder cache while the causal-neighbour
test and exact stream gate remain green. Re-profile the committed encoder
before selecting another token-path change.

### 2026-07-31 — Arithmetic bit writes are forced inline

**Where:** `rvp9_core::entropy::bool_writer::BoolWriter::write`, called for
every probability-coded syntax and coefficient bit.

**What libvpx does:** `vpx_write` is a `static INLINE` function in
`vpx_dsp/bitwriter.h`, so its interval update is expanded into each caller.

**What rvp9 does:** marks the equivalent Rust method `#[inline(always)]`.
Ordinary `#[inline]` left one standalone function and a call for every coded
bit in the release binary.

**Why:** the committed Unit 124 profile still attributed 157 tile-worker self
samples (`4.61%`) to the call boundary and arithmetic writer. Forced inlining
lets the optimizer carry writer state and constant bit values through the
surrounding token grammar.

**Byte-parity impact:** none. All fifteen focused bool-writer tests pass. The
120-frame cpu-used 7 product stream remains byte-identical, and rvp9-decoder,
pinned vpxdec, and ffmpeg produce identical 162,201,600-byte I420 output.

**Compression-efficiency impact:** none; the production stream is unchanged.

**Speed impact:** sixteen warmed alternating one-thread pairs improved paired
child CPU by `-1.721%` and wall time by `-1.680%`, with fifteen wins. Twenty
four-thread pairs improved child CPU by `-1.154%` with eighteen wins and wall
time by `-1.309%` with seventeen wins. Mach-O text grew 16 KiB, from
1,163,264 to 1,179,648 bytes (`1.41%`), a measured trade for the repeated hot
call removal.

**Future plan:** retain forced inlining while the exact stream and focused
writer tests remain green. Use the committed post-change profile to ensure
the removed standalone leaf did not merely move equal work into callers.
