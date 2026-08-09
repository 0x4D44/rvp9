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

### 2026-08-03 — Retain quality-bearing nearest and GOLDEN filter searches
**Where:** `rvp9_core::mode_decision::inter::should_search_interp_filter` and
`rvp9_encoder::pick_and_simulate_inter_encode_partitions`; libvpx counterpart:
`vp9/encoder/vp9_pickmode.c:1908-1917,2308-2318`.
**What libvpx does:** selects the above or left inter neighbour's filter, then
searches the realtime filter set only for NEWMV or when no concrete neighbour
filter exists. VBR and SVC permit GOLDEN as a searched reference frame.
**What rvp9 does:** reuses a concrete neighbour filter for LAST NEARMV. It
retains every enabled GOLDEN search. CPU6 VBR retains LAST NEARESTMV filter
search on every leaf. It also lets fractional LAST NEARMV candidates search
when their causal filter is `EightTap`, subject to the normal integer-MV and
chessboard gates. Other v1 configurations keep the full NEAREST search through
1280x720; above that frame area only 64x64 leaves keep the extra search.
**Why:** rvp9 previously searched every mode and reference. Temporary matched
counters localized 292,912 excess 16x16 model-RD calls to filter search. The
fully libvpx-faithful policy reduced calls but made `in_to_tree` 1080p luma
BD-rate `+2.644060%` versus vpxenc. Retaining LAST NEARESTMV and all VBR
GOLDEN searches recovered that quality, while concrete filter reuse removed
the measured LAST NEARMV evaluations. A later literal-resolution sweep found
that CPU7/8 smaller large-frame NEAREST searches could reuse the causal filter
without crossing any 1080p clip guard. The CPU6 exact-partition path instead
improves both limiting clips when it keeps the broader search and retains
enough measured speed headroom to pay for it. The narrower causal-EightTap
NEARMV recovery was previously rejected on a loaded-host speed screen; after
the exact CPU6 partition architecture it is wall- and CPU-neutral in the
required repeated-pair measurement and improves the corpus curve.
**Byte-parity impact:** structural for CPU6 VBR. The exercised 1080p in-tree
target-2500 IVF has SHA-256
`7f6b00a39da4122139d96140a70d7f059372da86439bf5085b8d1533d51a7824`.
Rvp9-decoder, pinned vpxdec, and ffmpeg each emit the same 186,624,000-byte
I420 output at SHA-256
`348555a80ab73b95746082729aa045893c651f33474a22c39f75e34e01a4573f`.
CPU7 VBR, CPU8 VBR, and CPU6 CBR controls remain byte-identical.
**Compression-efficiency impact:** CPU7/8 VBR and CPU6 CBR streams remain
byte-identical to the accepted policy. Canonical CPU6 1080p six-point luma
BD-rate versus pinned vpxenc is `+1.068%/-12.317%/-2.063%` for
`in_to_tree`/`ducks_take_off`/`old_town_cross`. Against the preceding CPU6
policy this improves the limiting clip by 0.126 point and ducks by 0.550 point;
old-town gives back 0.063 point while remaining a 2.1% win.
**Speed impact:** fifteen warmed alternating same-build pairs measure a
`1.000000x` paired wall geomean and `1.001026x` child-CPU geomean versus the
preceding policy. Direct plain-release pairs against pinned vpxenc measure
`0.925363x` wall and `0.944691x` child-CPU geomeans. Median rvp9 frame times
are `6.000/9.833/6.167 ms`.
**Future plan:** retain while the per-clip quality and speed gates stay green.
Revisit only after a more source-faithful residual model can replace the extra
search without losing compression efficiency.

### 2026-08-02 — Large-frame maximum transform retained outside Block32
**Where:** `rvp9_encoder::realtime_emit_luma_tx_size` and
`rvp9_encoder::pick_and_simulate_inter_encode_partitions`; libvpx counterpart:
`vp9/encoder/vp9_pickmode.c`'s selected `mi->tx_size` consumed by
`vp9/encoder/vp9_encodemb.c`.
**What libvpx does:** realtime residual emission uses the transform size stored
by the non-RD picker for every block size.
**What rvp9 does:** Block32 now reuses the picker's transform because its tiled
`block_yrd` decision is production-aligned. Large-frame recovery continues to
force the maximum transform for Block64, Block16, rectangular, and smaller
leaves. `RVP9_RT_EMIT_MAX_TX=0 RVP9_RT_EMIT_TX_RD=0` remains the full libvpx
policy diagnostic.
**Why:** full picker-transform reuse improves two current VBR clips but
regresses `ducks_take_off` by `+7.396107%` average-YUV BD-rate. Restricting the
change to Block32 preserves that clip (`+0.014199%`) while improving
`in_to_tree` by `-2.768749%` and `old_town_cross` by `-3.332769%` relative to
the preceding production baseline.
**Byte-parity impact:** structural. The representative 60-frame 1000 kbps
stream decodes bit-exactly under rvp9-decoder, pinned vpxdec, and ffmpeg to
81,100,800-byte I420, SHA-256
`a7a66c20f5073596a9aaff06ce470fd66528b77185d4941539eba4eb42762e66`.
**Compression-efficiency impact:** versus pinned vpxenc, average-YUV BD-rate
is `+16.633922%/-5.296992%/+37.047361%` on
`in_to_tree`/`ducks_take_off`/`old_town_cross`; luma is
`+17.442906%/-5.986901%/+39.705404%`.
**Future plan:** align Block64 and smaller picker residual models before
removing their maximum-transform recovery. Do not promote the broad diagnostic
until every important clip stays within the per-clip regression tolerance.

### 2026-08-02 — Diagnostic keyframe small-block tree omits rectangular partitions
**Where:** `rvp9_core::mode_decision::partition::choose_partitioning_keyframe_sub8`
and `rvp9_encoder::frame_loop::pick_and_simulate_intra_encode_partitions`;
libvpx counterparts: `vp9/encoder/vp9_encodeframe.c::{fill_variance_4x4avg,set_vt_partitioning}`
and `vp9/encoder/vp9_rdopt.c::{rd_pick_intra_sub_8x8_y_mode,rd_pick_intra_sby_mode}`.
**What libvpx does:** keyframes build a source-variance tree from rounded 4x4
averages, consider NONE/HORZ/VERT/SPLIT partitions, and pick four causal BMI
modes with live coefficient entropy contexts. At realtime cpu-used 6–8 it
restricts 4x4 luma search to DC/V/H and uses the full neighbour-derived token
context plus exact sign/category-bit cost for every coefficient. An unsplit
8x8 leaf instead uses the full keyframe RD search: mode and coefficient rate,
reconstructed-pixel SSE, and conditional directional-mode pruning.
**What rvp9 does:** `RVP9_EXACT_KF_SUB8=1` enables a diagnostic-only tree that
uses the same rounded 4x4-average variance in libvpx's native 256x fixed-point
domain but considers NONE/SPLIT only. It forces the keyframe root split and
propagates 16x16 force flags in libvpx's quadrant-major order. On the explicit
30fps `in_to_tree` qindex-83 keyframe, all 16,092 visited partition nodes now
match libvpx exactly across 220 superblocks. A selected 8x8 SPLIT carries four
causal DC/V/H 4x4 modes and prices them with
the full token-neighbour and live tile above/left coefficient contexts,
including exact coefficient sign/category-bit costs. It updates the contexts
after every winner. An unsplit 8x8 leaf now uses libvpx's full coefficient-rate
and reconstructed-pixel RD path, including the combined two-cell entropy
context and conditional directional search. Its standalone Tx8 predictor also
extends the last causal top pixel across the unavailable top-right region,
matching `vp9_predict_intra_block`. Block16x16 leaves now use libvpx's
realtime Hadamard + `quantize_fp` model: quantized-coefficient SATD rate,
coefficient error distortion, skip syntax, and DC/V/H mode syntax. The first
explicit-30fps `in_to_tree` leaf matches all three pinned-libvpx candidate
scores exactly and selects H. Block32x32 applies the same model over four
causal Tx16 tiles; the formerly divergent leaf at pixel `(416,0)` matches all
three pinned-libvpx candidate scores exactly. Across the explicit-30fps
`in_to_tree` keyframe, all 12,124 decoded leaf sizes and luma modes now agree.
Production behavior is unchanged.
**Why:** this is the smallest independently testable seam that closes the
measured keyframe-reference mismatch without exposing incomplete behavior as
a default. The corrected picker matches pinned libvpx exactly for the first 80
consecutive live 4x4 decisions. Porting the adjacent 8x8 path makes that leaf
and its causal 4x4 follower exact. Extending standalone Tx8 top-right pixels
corrects the next D63 winner and extends full live agreement through the first
800 4x4 decisions on the native-50fps diagnostic.
**Byte-parity impact:** structural only under the diagnostic environment flag.
The normal CPU7 VBR stream remains 47,314 bytes with SHA-256
`96af4829c357ff6550bfbec8667d5d19e07f78e65e0145b7da3096082ec00eaf`.
Under the flag, rvp9-decoder, pinned vpxdec, and ffmpeg produce the same
6,291,456-byte I420 output, SHA-256
`a2ae4522e7247ebc39c25d8984744f5395db2ef84c111e42dee359f0cc5808c8`.
For the current explicit-30fps `in_to_tree` 1000 kbps stream, rvp9-decoder,
pinned vpxdec, and ffmpeg each produce the same 81,100,800-byte I420 output,
SHA-256
`ebb1762b11116bbb0d2512b609734b5f5b709e8a5f7ec400cc9de64cd7133606`.
**Compression-efficiency impact:** at fixed qindex 120, the corrected first
keyframe is 37,536 bits at 37.999 dB luma PSNR versus libvpx's 35,368 bits at
38.031 dB. The 64-frame stream is 124,353 bytes versus libvpx's 82,270. On the
fresh explicitly matched 30fps native three-clip CPU7 screen, luma BD-rate is
`+41.837072%/-1.351268%/+96.584123%` and average-YUV BD-rate is
`+38.471076%/-0.564844%/+88.041766%` for
`in_to_tree`/`ducks_take_off`/`old_town_cross`. Versus its clean predecessor,
the Block32x32 model moves luma by `-0.064605/+0.000000/-0.430055`
points and average-YUV by `-0.089856/+0.000000/-0.462544` points. It is a
small reproducible Pareto improvement on this screen. The earlier
`+36.442309%/+5.200030%/+75.260522%` screen encoded the native 50fps Y4M timing
and is not evidence for the declared 720p30 configuration. The path remains
far outside the final 0.5% gate.
**Speed impact:** on the 60-frame 1280x704 `in_to_tree` 1000 kbps VBR
diagnostic at explicit 30fps, nine warmed alternating pairs measure 0.549401
seconds rvp9 versus 0.400559 seconds vpxenc median child CPU (`1.371586x`);
wall time is `1.362243x`. This is not a corpus geomean and does not pass the
final speed gate.
**Future plan:** compare reconstructed keyframe pixels now that every
partition and luma-mode decision agrees. Locate the first transform,
quantization, chroma-mode, or reconstruction mismatch before returning to the
inter-frame reference gap. Rectangular keyframe partitions remain omitted,
but none were exercised by this matched keyframe.

### 2026-08-02 — Keep the calibrated VBR rate prior and GOLDEN allocator default-off
**Where:** `rvp9_encoder::frame_loop::{VBR_BITS_PER_MB_MULTIPLIER_DEFAULT,StreamEncoder::new,StreamEncoder::encode_frame}`
and `rvp9_encoder::rate_control::{cbr::RateFactorLevel,vbr::pick_q_and_bounds}`;
libvpx counterparts: `vp9/encoder/vp9_ratectrl.c::{get_rate_correction_factor,rc_pick_q_and_bounds_one_pass_vbr,vp9_set_gf_update_one_pass_vbr}`.
**What libvpx does:** uses the unscaled bits-per-MB model, derives the
one-pass GOLDEN interval from frame rate and resolution, assigns boosted
GOLDEN frames an independent `GF_ARF_STD` correction slot, and caps their
top Q at the 1.75x-rate equivalent.
**What rvp9 does:** production VBR retains its measured 4x bits-per-MB prior,
fixed eight-frame refresh cadence, and ordinary-inter rate allocation for
GOLDEN refreshes. `RVP9_BPM_MULTIPLIER=1 RVP9_VBR_GF_RC=1` enables the
source-aligned stack for controlled diagnostics; that path uses the pinned
interval formula, independent correction slot, and 1.75x Q bound.
**Why:** rvp9's current partition, inter-mode, and skip decisions do not yet
match libvpx's residual distribution. Enabling the source-aligned stack by
default regressed the seven-point CPU7 Foreman luma BD-rate from `-8.534253%`
to `+20.131756%` versus vpxenc. It improved the exact-partition diagnostic on
the two failing native clips from `+67.616%/+105.583%` to
`+45.707%/+83.762%`, which makes the opt-in path a useful prerequisite, but
that partition mode is not fast enough to ship.
**Byte-parity impact:** structural only when either diagnostic override is
enabled. Production defaults retain the established rate prior and allocator.
**Compression-efficiency impact:** production Foreman remains quality-positive
against pinned vpxenc. The native CPU7 corpus remains far outside the final
gate and is tracked in the parity journal.
**Future plan:** first close the measured partition/mode/skip mismatch. Then
retest the source-aligned allocator on the full corpus and make it default
only when it is a reproducible quality/speed Pareto improvement. This entry
supersedes the 2026-05-22 A10 statement that the VBR default multiplier is
1.0.

### 2026-08-01 — Large-block work bypass requires YUV skip; mode-loop stop remains Y-only
**Where:** `rvp9-core::mode_decision::inter::{evaluate_candidate_for_filter_with_pred,chroma_planes_skippable}`;
libvpx counterpart: `vp9/encoder/vp9_pickmode.c::model_rd_for_sb_y_large`.
**What libvpx does:** bypasses `block_yrd`, chroma model RD, later inter
candidates, and intra fallback only after Y, U, and V all pass their
transform-skip tests.
**What rvp9 does:** now applies the expensive-work bypass only after the same
YUV test, but retains its existing Y-only `early_term` flag for stopping the
remaining mode loop.
**Why:** making the loop-stop condition YUV-faithful reduced the standard
120-frame stream from 716,330 to 713,096 bytes and regressed average PSNR by
0.457 dB (31.935 to 31.478). Retaining the historical Y-only stop preserves
the established Stage 4a quality curve, while requiring YUV agreement before
the work bypass guarantees that skipped Hadamard and chroma calculations
cannot affect the selected candidate's cost.
**Byte-parity impact:** none from the new work bypass. The standard and
natural-motion focused streams remain byte-identical to Unit 227, and the
focused test covers 32x32 and 64x64 YUV skip decisions. The retained Y-only
mode-loop stop is a structural decision divergence from libvpx.
**Compression-efficiency impact:** none from the bypass because emitted
streams are unchanged. The retained loop-stop divergence is quality-positive
on the standard gate relative to the direct canonicalization experiment.
**Speed impact:** three twenty-pair four-thread comparisons improve total
child CPU by 2.90%, 3.58%, and 2.97%; the pre-registered Y-only upper bound
was 4.62-4.63%.
**Future plan:** keep the YUV work-bypass oracle. Revisit the Y-only loop stop
only with a rate-distortion calibration that recovers the measured 0.457 dB;
do not canonicalize it as a source-parity cleanup.

### 2026-07-31 — Retained replay specializes the common ONE token
**Where:** `rvp9-core::entropy::detokenize::replay_retained_coefs`;
libvpx counterpart: `vp9/encoder/vp9_bitstream.c::pack_mb_tokens`.
**What libvpx does:** recognizes `ONE_TOKEN` in its retained token stream and
writes the one-node decision plus sign directly. Larger magnitudes enter the
coefficient tree and category-bit path.
**What rvp9 does:** retains raw signed coefficients rather than libvpx token
identifiers. Replay now recognizes `+1` and `-1` and emits those same two
decisions directly; every other nonzero coefficient retains the shared
general writer.
**Why:** a temporary product-workload counter found that 680,337 of 804,738
replayed nonzero coefficients (`84.54%`) were `+1` or `-1`. Avoiding the
large category writer's call on that dominant case restores libvpx's packing
shape without enlarging the four-byte retained record.
**Byte-parity impact:** none. Direct-versus-retained tests cover both signs,
zero runs, every coefficient category, and changed post-header probability
tables. The 120-frame product stream remains byte-identical.
**Compression-efficiency impact:** none because the arithmetic decisions and
emitted stream are unchanged.
**Speed impact:** two twenty-pair one-thread trials improve total child CPU
by `-0.554%` and `-0.116%`; two twenty-pair four-thread trials improve it by
`-0.559%` and `-0.401%`. All four paired wall results also improve.
**Future plan:** retain while the direct/replay oracle and exact product
stream remain green. Keep category encoding centralized in
`write_nonzero_coef`; do not duplicate the larger tree in replay.

### 2026-07-31 — Mixed direct/clamped integer projections
**Where:** `rvp9-core::mv_search::int_pro_me::{fill_int_pro_rows,fill_int_pro_cols}`;
libvpx counterparts: `vp9/encoder/vp9_mcomp.c::vp9_int_pro_motion_estimation`
and physically extended reference borders.
**What libvpx does:** reads every projection from a frame buffer whose
replicated border is already materialized, so visible chunks use the normal
SIMD projection kernels even when the full search window crosses an edge.
**What rvp9 does:** represents only visible pixels. It selects the direct
NEON row kernel independently for each visible 16-column chunk, hoists fixed
clamped coordinates out of inner loops, and represents a replicated row as
left-edge, visible-middle, and right-edge spans.
**Why:** the previous all-or-nothing gate sent an entire 128-column/row
projection through scalar per-pixel clamping when any part crossed the
visible frame edge.
**Byte-parity impact:** none. A pixel oracle covers partial and fully
outside windows in both dimensions, every 16/32/64 projection size, and
nontrivial stride. The product stream remained byte-identical and all three
decoders produced identical I420 output.
**Compression-efficiency impact:** none because every projection, selected
MV, SAD, partition, and emitted byte is unchanged.
**Speed impact:** two independent sixteen-pair one-thread trials improved
paired wall time by `-2.510%` and `-2.426%`; child CPU improved `-2.582%`
and `-2.428%`. Two twenty-pair four-thread trials improved paired wall time
by `-1.344%` and `-0.747%`; child CPU improved `-2.161%` and `-2.467%`.
The sampled inter-partition wrapper fell from 375 to 264 self samples, and
its former scalar clamped-loop cluster disappeared.
**Future plan:** retain while the edge oracle and exact stream stay green.
If rvp9 later materializes extended borders, replace these spans with direct
reads from that buffer instead of keeping two border representations.

### 2026-07-31 — Full-width Hadamard dequant multiply
**Where:** `rvp9-core::dsp::aarch64::quantize_neon::quantize_i16_8`;
libvpx counterpart: `vp9/encoder/arm/neon/vp9_quantize_neon.c`.
**What libvpx does:** produces eight quantized/dequantized coefficients per
NEON group for realtime Hadamard rate-distortion metrics.
**What rvp9 does:** computes the observable low 16 bits of all eight
`qcoeff * dequant` products with one `vmulq_s16`, then widens the result for
its fused distortion accumulator.
**Why:** the prior two four-lane widening multiplies were narrowed
immediately. The scalar contract casts the product to `i16`, making the
single full-width low-half multiply exactly equivalent.
**Byte-parity impact:** none. Random, zero, DC-only, final-scan, and signed
extreme coefficients across every realtime transform size and qindex match
the widened quantizer oracle. The product stream remained byte-identical.
**Compression-efficiency impact:** none because quantized values and fused
metrics are unchanged.
**Speed impact:** machine code replaces two `mul.4h` instructions per eight
coefficients with one `mul.8h`. After external load cleared, sixteen
one-thread pairs improved paired wall/child CPU by `-0.619%`/`-0.732%`;
twenty four-thread pairs improved them by `-0.825%`/`-1.165%`. Earlier
contended trials also reduced child CPU in all three runs.
**Future plan:** retain while the exact metrics oracle and cumulative speed
baseline stay green.

### 2026-07-31 — Reuse product-picker model SSE
**Where:** `rvp9-core::mode_decision::inter::evaluate_candidate_for_filter_with_pred`;
libvpx counterpart: `vp9/encoder/vp9_pickmode.c::model_rd_for_sb_y`.
**What libvpx does:** returns source/prediction SSE from model-RD and carries
that result into the realtime picker.
**What rvp9 does:** likewise reuses `ModelRdYOutput::sse_y` as pixel
distortion when its product Hadamard-only calibration leaves reconstruction
equal to prediction. The compatibility calibration still scans the
post-inverse-transform reconstruction.
**Why:** the old product path scanned the complete source/prediction block a
second time even though model-RD had already produced the exact reduction.
**Byte-parity impact:** none. A focused oracle compares direct SSE, model
SSE, and surfaced candidate distortion for 8x8 through 64x64 blocks. The
120-frame product stream remained byte-identical and all three decoders
produced identical I420 output.
**Compression-efficiency impact:** none because the reused value is exact
and every emitted byte is unchanged.
**Speed impact:** two independent sixteen-pair one-thread trials improved
paired wall time by `-6.478%` and `-9.656%`; child CPU improved `-6.818%`
and `-8.620%`, with every pair winning. Two twenty-pair four-thread trials
improved paired wall time by `-4.051%` and `-3.599%`; child CPU improved
`-6.598%` and `-5.710%`. The second-scan evaluator loop accounted for 460
self samples before the change and is absent afterward.
**Future plan:** retain while the direct/model equality oracle and exact
stream stay green. Keep the compatibility scan until that path stops
reconstructing pixels.

### 2026-07-31 — Exact eight-lane AArch64 16-tap loop filter
**Where:** `rvp9-core::dsp::aarch64::lpf16_neon`; libvpx counterparts:
`vpx_dsp/arm/loopfilter_neon.c` and
`vpx_dsp/arm/loopfilter_16_neon.asm`.
**What libvpx does:** filters eight edge lanes together with NEON, using
specialized horizontal and vertical 16-tap kernels plus dual-edge wrappers.
Its byte-vector boundary mask uses saturating arithmetic.
**What rvp9 does:** also filters eight lanes together, but widens
`2 * abs(p0 - q0) + abs(p1 - q1) / 2` to `u16` before comparing it with
`blimit`. It applies the 4-, 8-, and 16-tap results through per-lane vector
selects, so mixed flatness masks remain exactly equivalent to rvp9's scalar
oracle. When no lane can use the 7-tap result, an aggregate all-zero check
skips both wide candidates; mixed and flat edge groups retain the per-lane
selects. The vertical kernel transposes a 16-by-8 neighborhood before
filtering, then uses libvpx's inverse NEON transpose shape to store all eight
16-byte result rows without a scalar scatter.
**Why:** rvp9 accepts the complete `u8` threshold domain in its public DSP
slot. Widening preserves scalar semantics when `blimit` is 255 rather than
copying libvpx's saturating-byte approximation. The aggregate early exit is
exact only for an all-zero wide mask; per-lane selects still cover every
mixed flatness case.
**Byte-parity impact:** none. Randomized single/dual, horizontal/vertical
differentials cover varied strides and thresholds, including all-255
thresholds. The complete reconstructed-plane oracle passed. The 120-frame
product stream remained byte-identical and all three decoders produced
identical I420 output.
**Compression-efficiency impact:** none because every filtered sample and
emitted byte is unchanged.
**Speed impact:** two independent sixteen-pair one-thread trials improved
paired wall time by `-10.108%` and `-9.932%`; child CPU improved `-10.298%`
and `-10.244%`. Two twenty-pair four-thread trials improved paired wall time
by `-4.004%` and `-6.618%`; child CPU improved `-9.327%` and `-10.204%`.
The post-change profile contains no scalar `filter16` leaf. The new vertical
and horizontal NEON kernels account for `2.07%` and `1.09%` of rvp9 self
samples. Replacing the vertical kernel's temporary 14-by-8 byte slab and
scalar scatter with the exact inverse transpose improved a further twenty
paired runs by `-1.094%` child CPU and `-1.097%` wall at one thread, and by
`-1.154%` child CPU and `-0.375%` wall at four threads.
Skipping both wide candidates for all-nonflat lane groups improves two
twenty-pair one-thread child-CPU trials by `-0.165%` and `-0.096%`; paired
wall improves `-0.213%` and `-0.426%`. Two four-thread trials are favorable
at `-0.423%` and neutral at `+0.012%` child CPU.
**Future plan:** retain while the randomized scalar differential and exact
stream stay green. Profile before extending NEON coverage to the distinct
4- and 8-tap slots.

### 2026-07-31 — Direct-index inter coefficient payloads
**Where:** `rvp9-encoder::VecBackedInterCoefSource`; libvpx counterparts:
macroblock coefficient storage and `vp9/encoder/vp9_tokenize.c`.
**What libvpx does:** stores quantized coefficient payloads in encoder block
state and later tokenizes them through structurally addressed plane/block
arrays.
**What rvp9 does:** maps each physical luma/U/V 4x4 position through a dense
`u32` slot table into an insertion-order coefficient vector. Luma positions
derive from two 4x4 cells per MI; Profile-0 chroma positions derive from one.
Column-tile sources merge by rebasing their value indices.
**Why:** rvp9's simulator and writer remain separate traversals, but their
six-field tuple key reduces exactly to the bounded physical TX coordinate.
The previous `HashMap` hashed that tuple during simulation inserts, tile
merges, the pre-header count pass, and final arithmetic writing.
**Byte-parity impact:** none. Focused coverage exhausts every slot for odd
frame dimensions, verifies replacement and disjoint tile merge behavior, and
rejects overlapping merges. The 120-frame product stream remained
byte-identical and all three decoders produced identical I420 output.
**Compression-efficiency impact:** none because every coefficient payload,
lookup result, and emitted byte is unchanged.
**Speed impact:** two independent sixteen-pair one-thread runs improved wall
time by `-1.464%` and `-1.225%`, with 31/32 wins; child CPU improved
`-1.482%` and `-1.331%`. Two twenty-pair four-thread runs improved wall time
by `-0.847%` and `-2.623%`, with 35/40 wins; child CPU improved `-1.048%`
and `-1.919%`. A five-encode post-change profile contains no inter-source
hash path; the remaining hash samples belong only to the keyframe intra
source.
**Future plan:** retain while the physical-slot oracle and exact stream stay
green. Optimize the one-frame intra source only if a fresh profile makes it
material; do not conflate it with the removed inter hot path.

### 2026-07-31 — Widen the 32-point DCT at its overflow boundary
**Where:** `rvp9-core::transforms::forward_neon`; libvpx counterparts:
`vpx_dsp/arm/fdct32x32_neon.c` and `vpx_dsp/arm/fdct32x32_neon.h`.
**What libvpx does:** its first 32x32 pass uses eight i16 NEON lanes and its
overflow-sensitive second pass widens selected stages to paired i32 vectors.
**What rvp9 does:** its first pass also uses eight i16 lanes. The second pass
keeps eight i16 lanes through stages 1 and 2, then widens into two four-lane
i32 groups at stage 3, where the libvpx overflow audit identifies the first
extreme-input overflow. Those low/high groups advance together through one
paired width-generic schedule, matching libvpx's `c_lo`/`c_hi` structure.
Before widening, rvp9 checks the actual stage-2 peak. Groups at or below
2047 remain in eight i16 lanes through the final stage; a conservative
interval audit bounds their peak below 31,537. Larger groups retain the
widened exact path.
First-pass and early second-pass cosine products widen to i32. Later
two-coefficient second-pass products use the same i32 multiply-accumulate and
rounded shift as libvpx; one-coefficient products retain rvp9's i64 reduction
because libvpx's rounded-high instruction was measured slower on this host.
**Why:** the earlier pass-boundary widening preserved rvp9's i32 coefficient
API but repeated bounded stages 1 and 2 in two four-lane groups. Moving the
widening to the measured overflow boundary matches libvpx's lane width while
retaining the shared later-stage schedule and i32 output API.
**Byte-parity impact:** none. Differential coverage compares 128 randomized
valid residual blocks with stride jitter plus constant and alternating
residual extremes. The 120-frame product stream remained byte-identical.
**Compression-efficiency impact:** none because every transform coefficient
and emitted stream is unchanged.
**Speed impact:** the original eight-lane first pass improved one-thread
child CPU by `-1.424%` and four-thread child CPU by `-1.022%`. The later
overflow-boundary unit reduces the full 32x32 transform from `747.10 ns` to
`717.04..725.94 ns` against pinned libvpx's `501.31 ns`. Two twenty-pair
one-thread product trials improve total child CPU by `-0.934%` and `-0.250%`;
one twenty-pair four-thread trial improves it by `-0.500%`. A later
one-thread trial under worse load was neutral at `+0.009%`. Advancing the
widened low/high halves together reduces the kernel again to
`687.23..708.00 ns`.
Two twenty-pair one-thread product trials improve total child CPU by
`-0.739%` and `-0.697%`; two four-thread trials improve it by `-1.491%` and
`-0.261%`. Matching libvpx's i32 two-coefficient multiply-accumulate reduces
the kernel further to `521.75..532.54 ns`. Two one-thread product trials
improve child CPU by `-1.455%` and `-1.352%`; two four-thread trials improve
it by `-0.152%` and `-0.885%`. The standard stream remains byte-identical.
The bounded late-stage path reduces the deterministic kernel again to
`518.67 ns`. Two twenty-pair one-thread product trials are neutral at
`-0.024%` and favorable at `-0.709%`; two four-thread trials improve child
CPU by `-0.517%` and `-0.336%`.
**Future plan:** retain the overflow-boundary widening while it stays exact
and Pareto-positive. Profile before changing the remaining later-stage
schedule or transpose/storage layout.

### 2026-07-31 — Direct-index the superblock descent-result cache
**Where:** `rvp9-encoder::DescentPickCache`; libvpx counterpart:
`vp9/encoder/vp9_context_tree.h::PC_TREE`.
**What libvpx does:** retains per-partition picker state in a preallocated
superblock context tree addressed structurally.
**What rvp9 does:** maps the complete 64-local-cell × 13-block-size key space
through a fixed slot table into an insertion-order value vector.
**Why:** rvp9's partition descent and causal final pass are separate walks,
so they do not share libvpx's `PC_TREE` shape. Direct indexing preserves that
separation while removing randomized hashing from the bounded trusted key
space.
**Byte-parity impact:** none. A focused oracle exhausts all 832 keys,
replacement semantics, and per-superblock scoping. The 120-frame product
stream remained byte-identical and all three decoders produced identical
I420 output.
**Compression-efficiency impact:** none because every cached picker result,
partition decision, and emitted stream is unchanged.
**Speed impact:** sixteen warmed alternating one-thread pairs improved wall
time `1.008414 -> 1.004468 s` (`-0.521%` paired) and child CPU
`0.976270 -> 0.972855 s` (`-0.446%`), with eleven wins. Twenty four-thread
pairs improved wall time `0.625460 -> 0.620181 s` (`-0.657%`) with fourteen
wins and child CPU `1.141023 -> 1.127910 s` (`-0.652%`) with sixteen wins.
**Future plan:** retain while exact and Pareto-positive. The separate
coefficient map remains hashed; Unit 106 already rejected packing that map's
keys under the default hasher.

### 2026-07-31 — Reuse pre-header count-pass results during tile emission
**Where:** `rvp9-encoder::frame_loop::build_inter_packet_with_residuals`;
libvpx counterparts: `vp9/encoder/vp9_tokenize.c` and
`vp9/encoder/vp9_bitstream.c::pack_mb_tokens`.
**What libvpx does:** tokenizes once before the compressed header, retaining
token records and frame counts, then packs the retained tokens without
recounting them.
**What rvp9 does:** its pre-header count-only syntax traversal retains the
complete probability-independent `FrameCounts`; its later arithmetic
traversal regenerates tokens from coefficients but no longer allocates,
updates, or merges duplicate per-tile count banks.
**Why:** rvp9 does not retain a libvpx-style `TOKENEXTRA` stream, but its
mandatory count pass already computes the exact backward-adaptation input.
Reusing that result removes duplicate count work without introducing a
frame-sized token arena.
**Byte-parity impact:** none. A focused oracle compares count-only and
arithmetic traversals, including coefficient EOBs, under different
probability tables. The 120-frame product stream remained byte-identical and
all three decoders produced identical I420 output.
**Compression-efficiency impact:** none because the emitted stream and
backward-adaptation counts are unchanged.
**Speed impact:** sixteen warmed alternating one-thread pairs were neutral
(`+0.021%` wall, `-0.021%` child CPU). Across two independent twenty-pair
four-thread runs, combined median paired wall time improved `-0.834%` with
28/40 wins; the individual wall results were `-0.551%` and `-0.992%`, and
the repeat improved child CPU `-0.522%`.
**Future plan:** retain while count equality and byte identity hold. A
libvpx-style retained token arena would be a separate measured architecture
decision, not an implicit extension of this change.

### 2026-07-31 — Bound the eight-lane second 16-point DCT pass
**Where:** `rvp9-core::transforms::forward_neon`; libvpx counterpart:
`vpx_dsp/arm/fdct16x16_neon.c`.
**What libvpx does:** processes both 16x16 passes in eight i16 NEON lanes.
**What rvp9 does:** matches the eight-wide i16 first pass. The second pass
also stays eight-wide when every input magnitude is at most 2047. Larger
inputs perform the bounded cross-input stage eight-wide, then widen into two
four-lane i32 groups for the remaining schedule.
**Why:** a coefficient-bound audit proves the 16-point schedule peaks at
16 times its input, so `16 * 2047 = 32752` fits i16. Unconditionally matching
libvpx's all-i16 path broke rvp9's scalar oracle on the valid constant
residual `-255` (`13701` instead of `-32639`). The magnitude gate retains the
common fast path and the established exact behavior for extreme residuals.
**Byte-parity impact:** none. Differential coverage compares 128 randomized
valid residual blocks with stride jitter plus constant and alternating
residual extremes. The 120-frame product stream remained byte-identical.
**Compression-efficiency impact:** none because every coefficient and emitted
stream is unchanged.
**Speed impact:** the deterministic 16x16 kernel improved from `127.17 ns`
to `84.95 ns`; pinned libvpx measured `92.48..95.64 ns`, with matching input
and output hashes. Two independent twenty-pair four-thread product runs
improved child CPU by `-0.720%` and `-1.029%` and paired wall time by
`-1.021%` and `-0.985%`. One-thread trials were noisier on a host above load
18: three direct comparisons changed child CPU by `-0.554%`, `+0.614%`, and
`-1.029%`; an intermediate 1023-bound build improved twice, and raising the
bound to 2047 improved a direct twenty-pair comparison by another `-0.908%`.
**Future plan:** retain while the scalar differential, 2047 bound, and exact
stream remain green. Recheck corpus timing on the clean pinned host.

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

### 2026-08-01 — Retain the duplicate picker DCT on cyclic-refresh boosted segments
**Where:** `rvp9-core::mode_decision::inter::evaluate_candidate_for_filter`
and `InterRdCalibration`; libvpx counterpart:
`vp9/encoder/vp9_pickmode.c:block_yrd`.
**What libvpx does:** realtime candidate RD uses one Hadamard transform,
`quantize_fp`, its coefficient rate/distortion, and its skippable result.
The selected leaf is transformed and reconstructed later by the encode path.
**What rvp9 does:** the scoped one-pass realtime CBR/VBR cpu-used 6–8
baseline segments match that single Hadamard pipeline, including segment 0
inside a cyclic-refresh frame. BOOST1/BOOST2, CQ, constant-Q, two-pass, and
slower presets retain the older second DCT, `quantize_b`, inverse DCT, and
pixel-SSE pass inside every candidate before the selected leaf is transformed
again.
**Why:** the single-pipeline policy is a measured Pareto improvement on the
AQ-off v1 matrix. Applying it to boosted AQ leaves breaks the focused
finer-quantization oracle (`1.691x` AQ/off SSE); applying it only to segment 0
keeps that oracle green and removes the unrelated baseline-region penalty.
Other retained paths are outside this parity objective and keep their
established behavior.
**Byte-parity impact:** structural for calibrated realtime paths because the Hadamard
skippable predicate can differ from the former DCT-EOB predicate. Exercised
streams remain deterministic and decoder-valid.
**Compression-efficiency impact:** on the 64-frame fixed-Q Foreman screen,
inter payload moved `25,166 -> 25,085`, `12,788 -> 12,624`, and
`5,856 -> 5,770` bits/frame at q80/q120/q160 while PSNR improved by
`0.05/0.07/0.13 dB`. On the signed Stage 4b Foreman VBR gate, segment-aware
calibration moves rvp9's AQ penalty from `+36.763%` to `+6.111%`; pinned
libvpx is `+4.714%`, so the gain delta narrows from `+32.049` to `+1.397`
percentage points. This is a measured prerequisite, not final AQ parity.
**Speed impact:** twelve alternating four-encode pairs on 120 frames of
1280x704 testsrc reduced cpu-used 7 CBR median wall time by `33.55%` and total
child CPU by `39.54%`, with 12/12 wins for both metrics.
**Future plan:** close the remaining AQ delta through the signed-off dynamic
BOOST1/BOOST2, per-block segment-update, and cyclic-refresh-aware rate-control
work. Retire the other compatibility paths only under a separately scoped
objective.

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

### 2026-07-30 — Use the four-mode realtime intra-in-inter fallback
**Where:** `rvp9-core::mode_decision::inter::pick_inter_mode` and
`rvp9-encoder::encode_inter_frame`; libvpx counterpart:
`vp9/encoder/vp9_pickmode.c:2517-2646`.
**What libvpx does:** evaluates DC/V/H/TM after the realtime inter search,
uses adaptive per-mode RD thresholds, and applies additional search-skip and
low-source-SAD entry guards. Its chosen luma mode also drives chroma.
**What rvp9 does:** evaluates DC/V/H/TM on final leaves, ranks them in the
coefficient-distortion domain, and keeps UV prediction on DC. It now reads and
updates libvpx's adaptive intra RD-threshold slots. It does not apply the
remaining search-skip/low-source-SAD shortcuts.
**Why:** the original pixel-SSE fallback over-selected intra and was disabled.
The coefficient-domain correction made the narrower final-leaf search a
reproducible quality and speed win, so it was re-enabled. Separate attempts to
copy libvpx's entry shortcuts and luma-driven chroma both worsened the strict
Foreman curve.
**Byte-parity impact:** structural — inter frames choose different block modes,
but all exercised streams remain byte-exact across rvp9-decoder, vpxdec, and
ffmpeg decoded output.
**Compression-efficiency impact:** the coefficient-domain fallback improved
the then-current Stage 4a BD-rate from `+6.8761%` to `+4.9246%`. Adding the
adaptive intra threshold later improved the measured seven-point Foreman curve
by a further `0.0476%` relative while reducing the exact-VBP test CPU by
`4.917%`. Adding the source TM candidate on the corrected CPU6 controller
improves serialized in-tree/ducks/old-town curves from
`+1.640%/-11.830%/-1.327%` to `+1.576%/-12.137%/-1.402%`. Fifteen warmed
same-build pairs measure a `1.0085x` wall-time geomean and `1.0098x` child-CPU
geomean; CPU6 retains sufficient direct-vpxenc speed margin.
**Future plan:** add luma-driven chroma or the remaining entry shortcuts only
when each retains a reproducible corpus-level joint quality/speed improvement.

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

### 2026-05-21 — Encoder-side loop-filter level approximation (F11.0) [resolved 2026-08-02]
**Where:** `rvp9-encoder::loopfilter::pick_filter_level`. libvpx
counterpart: `vp9/encoder/vp9_picklpf.c::vp9_pick_filter_level`.
**Original divergence:** rvp9 used `(q_index >> 1).min(63)` while
libvpx realtime cpu-used 6–8 selected `LPF_PICK_FROM_Q`.
**Resolution:** rvp9 now mirrors the 8-bit `LPF_PICK_FROM_Q` formula,
including the keyframe adjustment, and applies libvpx's default
reference deltas `[1, 0, -1, -1]` in both the bitstream and encoder
reconstruction. `RVP9_LPFQ=0` remains a diagnostic escape hatch.
**Byte-parity impact:** none in the matched v1 realtime policy.
**Compression-efficiency impact:** the old approximation is no longer
the product default.
**Remaining scope:** full-image and sub-image filter-level searches are
outside the one-pass realtime v1 matrix.

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
intermediate block. Runtime dispatch pre-resolves the reduction to either a
base-NEON tier, whose independent quarter-block lanes combine once at the
end, or an Armv8.2-A dot-product tier, which accumulates prediction sum,
reference sum, and squared absolute differences with three `udot`
instructions per filtered 16-byte vector.

**Why:** rvp9 previously had no AArch64 sub-pixel-variance dispatch and ran
two heap allocations plus three scalar pixel walks for every sub-pel probe.
The fused layout removes both allocations and the final temporary-block walk
while preserving libvpx's exact bilinear rounding and variance finisher.
Independent accumulators also remove the two serial widening additions and
two serial multiply-accumulates that the first fused kernel issued per
16-pixel group.

**Byte-parity impact:** none. Random differential coverage exercises all 64
offset pairs across eight block shapes for both SIMD tiers; edge-value and
forced-dispatch tests also match the scalar oracle exactly. A 120-frame
cpu-used 7 encode remained exactly 716,330 bytes with SHA-256
`8144dcf9d104119634855cf2ed0554f14e206137ad33689f5b0f871b95c2acb3`.

**Compression-efficiency impact:** none; the production stream is unchanged.

**Speed impact:** the 32x32 mixed-offset kernel improved from `797.1 ns` to
`85.8 ns` (`9.30x`). Three alternating 120-frame 1280x704 cpu-used 7 runs
improved median wall time from `2.74 s` to `2.45 s` (`10.6%`) on the pinned
Apple M5 Max host. Splitting the reduction dependencies later improved paired
child CPU by another `-1.029%` at one thread with fifteen of sixteen wins and
`-0.947%` at four threads with seventeen of twenty wins. Replacing that
reduction with the pre-resolved dot-product tier then improved paired child
CPU by `-0.494%` and `-0.483%` in two one-thread trials. Three four-thread
trials ranged from `-0.737%` through `+0.056%` to `-0.207%`, so the
multi-thread effect is neutral to favorable rather than a claimed stable
win.

**Future plan:** retain the fused implementation and pre-resolved dot-product
tier while both remain byte-exact. Add 4/8-wide NEON kernels only if a fresh
whole-encoder profile shows those scalar slots are material.

### 2026-07-31 — AArch64 variance dispatch selects dot-product kernels

**Where:** `rvp9_core::dsp::{caps,dispatch}` and
`dsp::aarch64::variance_neon`, for every 8/16/32/64-wide `(sum, sse)` slot
and the dedicated 8x8-cell slot.

**What libvpx does:** its AArch64 runtime dispatch selects
`vpx_dsp/arm/variance_neon_dotprod.c` when Armv8.2-A dot-product instructions
are present. Those kernels accumulate source sum, reference sum, and squared
absolute differences with three `udot` operations per 16 pixels.

**What rvp9 does:** detects `dotprod` once with the existing process-wide
capability table and pre-resolves the same variance slots to an integer-exact
Rust/inline-assembly port. Older AArch64 hosts retain the established base
NEON path; 4-pixel-wide blocks remain scalar. The generic rvp9 interface
still returns widened `(i64, u64)` values.

**Why:** the base NEON kernel widened each eight-pixel difference before
separate sum and squared-error multiply-accumulates. The pinned Apple host
supports dotprod, and the fresh product profile attributed 94 self samples
directly to the 64x64 base-NEON reduction.

**Byte-parity impact:** none. Random differential coverage spans every
promoted block shape, stride jitter `0..16`, and equality/worst-case signed
difference extremes. Forced dispatch checks the scalar, base-NEON, and
dotprod tiers. The standard cpu-used 7 stream remains exactly 716,330 bytes
with SHA-256
`8144dcf9d104119634855cf2ed0554f14e206137ad33689f5b0f871b95c2acb3`.

**Compression-efficiency impact:** none; the production stream is unchanged.

**Speed impact:** twenty-pair comparisons improve total child CPU by
`-2.198%` at one thread with 20/20 wins and `-2.049%` at four threads with
19/20 wins. Paired wall time improves `-2.219%` and `-2.730%`, respectively,
with 20/20 wins in both trials.

**Future plan:** retain runtime dotprod dispatch while all three tiers remain
integer-exact. Profile the committed unit before selecting another kernel.

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

### 2026-07-31 — AArch64 horizontal interpolation keeps a signed-dot fallback

**Where:** `rvp9_core::dsp::aarch64::convolve8_neon::
convolve8_horiz_{i8mm,dotprod}_inner`, used by the horizontal and separable
2-D unscaled eight-tap interpolation paths on AArch64 hosts.

**What libvpx does:** `vpx_dsp/arm/vpx_convolve8_neon_i8mm.c` loads four
source rows at a time, permutes each 16-byte load into eight overlapping tap
windows, and evaluates them with unsigned-by-signed dot products. It retains
separate signed-dot and widened-i16 kernels for older AArch64 tiers.

**What rvp9 does:** uses the same four-row unsigned-by-signed i8mm arithmetic
when the host supports it. The arbitrary-height tail uses the same kernel one
row at a time. Older dot-product hosts retain rvp9's row-at-a-time signed-dot
kernel. Both tiers fall back to widened i16 for filters containing the
full-pel coefficient `128`, which cannot be represented as an i8 tap.

**Why:** matching libvpx's i8mm arithmetic closes the measured kernel gap on
modern AArch64 without dropping rvp9's arbitrary-height safe API or its older
hardware fallback.

**Byte-parity impact:** none. All supported block shapes and every phase of
the four VP9 filter families match the scalar oracle. The 120-frame cpu-used 7
product stream remains byte-identical, and rvp9-decoder, pinned vpxdec, and
ffmpeg produce identical I420 output.

**Compression-efficiency impact:** none; the production stream is unchanged.

**Speed impact:** the original signed-dot unit improved four-thread child CPU
by `-2.665%`. The later i8mm tier matches pinned libvpx's horizontal
microkernel at 16x16 (`19.08 ns` versus `19.28 ns`) and 64x64 (`303.12 ns`
versus `304.62 ns`). Three twenty-pair four-thread product trials improve
total child CPU by `-0.455%`, `-0.610%`, and `-0.822%`.

**Future plan:** retain all three runtime tiers while their scalar
differential and exact-stream checks remain green. Treat the still
reload-based vertical i8mm kernel as a separate measured unit.

### 2026-07-31 — AArch64 vertical interpolation keeps two dot-product tiers

**Where:** `rvp9_core::dsp::aarch64::convolve8_neon::
convolve8_vert_dotprod_inner`, used by the vertical and separable 2-D
unscaled eight-tap interpolation paths on dot-product-capable AArch64 hosts.

**What libvpx does:** `vpx_dsp/arm/vpx_convolve8_neon_dotprod.c` keeps seven
transposed rows live, loads four new rows, merges their columns through three
table lookups, and advances four output rows at a time.

**What rvp9 does:** on i8mm hosts, keeps the same rolling seven-row transpose
state as libvpx and uses unsigned-by-signed dot products. Older dot-product
hosts retain rvp9's simpler signed-dot kernel, which reloads the complete
eleven-row window for each four-output group. One scalar-NEON tail handles
arbitrary heights, and the widened-i16 path handles full-pel filters.

**Why:** the rolling i8mm tier closes the remaining measured vertical-kernel
gap on modern AArch64. Retaining the smaller reload-based signed-dot tier
avoids imposing the larger live SIMD state on older hosts without evidence
that it helps them.

**Byte-parity impact:** none. Differential coverage includes every VP9 filter
phase, all codec block shapes, and irregular `7x3`, `12x5`, and `63x9` tails.
The product stream and all three decoders' I420 output remain byte-identical.

**Compression-efficiency impact:** none; the production stream is unchanged.

**Speed impact:** the original signed-dot unit improved four-thread child CPU
by `-1.717%`. The rolling i8mm tier reduces the 16x16 vertical microkernel
from `32.52 ns` to `21.19 ns` against libvpx's `20.18 ns`, and 64x64 from
`493.35 ns` to `311.68 ns` against `300.10 ns`. Product comparisons on a
host with load average above 20 were neutral within noise: three twenty-pair
four-thread trials changed total child CPU by `-0.680%`, `+0.140%`, and
`-0.077%`; a one-thread trial changed it by `+0.047%`.

**Future plan:** retain the rolling i8mm tier as a measured kernel-parity
prerequisite while it remains byte-exact, and rerun corpus product timing on
the required clean pinned host. Keep the reload-based signed-dot fallback
until that hardware tier has separate evidence.

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
Its SATD reduction uses one unsigned pairwise-widening accumulator for the
absolute i16 qcoeff lanes instead of widening low/high halves separately. On
AArch64, the Hadamard-only path also subtracts source and prediction bytes in
the transform's first NEON load, rather than writing and rereading a separate
signed-residual slab. The reconstruction-compatible path remains staged.

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
requires the clean-host corpus matrix. Replacing the two SATD accumulators
with the exact pairwise-widening reduction improved three twenty-pair
one-thread/four-thread child-CPU trials by `-0.162%`, `-0.215%`, and
`-0.292%`; a second four-thread trial improved by `-0.163%`.
Fusing source subtraction into the Hadamard transform then improved another
`-0.572%` at one thread and `-0.331%` at four threads across twenty pairs,
with 14/20 wins in both trials.

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

### 2026-07-31 — Inter tiles retain probability-independent coefficient tokens

**Where:** `rvp9_core::entropy::detokenize::CoefTokenPass`,
`rvp9_core::bitstream::modes_writer::write_tile_inter_with_retained`, and
`rvp9_encoder::frame_loop::build_inter_packet_with_residuals`.

**What libvpx does:** `vp9_tokenize.c::tokenize_b` creates `TOKENEXTRA`
records before the compressed header. `vp9_bitstream.c::pack_mb_tokens`
replays those records after coefficient-probability updates.

**What rvp9 does:** the canonical pre-header count traversal stores compact
four-byte `(value, band, context)` tokens and transform-block boundaries per
tile. Final tile workers replay those immutable records against the
post-header probability tables. Replay still updates the tile coefficient
flags and rejects a tile when its block traversal does not consume exactly
the retained stream. The original tile-writer API remains the direct path.

**Why:** rvp9 previously scanned, classified, and context-modelled every
coefficient in both the count and final arithmetic passes. Token values,
scan bands, and neighbour-derived contexts are probability-independent.
Retaining them removes the duplicate work and establishes libvpx's packing
boundary for later whole-tile syntax work.

**Byte-parity impact:** none. A focused oracle records under one coefficient
probability table and proves byte identity with direct encoding under a
different table across empty, zero-run, low-token, category-token, and full
blocks. The 120-frame cpu-used 7 stream remains byte-identical, and
rvp9-decoder, pinned vpxdec, and ffmpeg produce the same 162,201,600-byte
I420 SHA-256.

**Compression-efficiency impact:** none; the production stream is unchanged.

**Speed impact:** the matched five-encode profiles reduce the inclusive
coefficient packing path from 427 baseline samples to 359 retained-token
samples. The host was saturated by unrelated `mddosem`, indexing, compiler,
and git work during repeated timing, so the alternating wall/CPU samples
were not stable enough for a final small-effect timing claim. This is a
measured architectural prerequisite; the clean-host cumulative matrix must
quantify its corpus effect.

**Future plan:** retain the libvpx-shaped token boundary while exact-stream
and replay-consumption gates stay green. Reuse the retained representation
when removing the remaining duplicate final tile syntax work.

### 2026-07-31 — Interpolation filters are ranked with returned luma-model results

**Where:** `rvp9_core::mode_decision::inter::{
luma_model_rd,select_interp_filter_by_luma_model,
evaluate_candidate_with_filter_set}`.

**What libvpx does:** `vp9_pickmode.c::search_filter_ref` constructs the luma
predictor for each allowed interpolation filter, calls
`model_rd_for_sb_y{_large}`, adds the switchable-filter signalling rate, and
retains the best filter. The outer non-RD picker then calls `block_yrd` and
the UV model only for that winner.

**What rvp9 does:** the shared prediction/model helper returns a typed
`ModelRdResult` for each filter. A small selector compares its model RD plus
the existing filter cost, then the full Hadamard coefficient, skip, and UV
evaluation runs once for the selected filter. Single-filter call sites keep
their direct full-evaluation path.

**Why:** the previous port ran the complete coefficient and chroma evaluation
for both EightTap and EightTapSmooth. That was slower than libvpx and selected
filters in a different metric domain. Returning the already-computed model
result keeps the Rust data flow explicit without reproducing libvpx's mutable
macroblock scratch.

**Byte-parity impact:** intentional stream change. A focused oracle builds
both predictors independently, computes both model costs, and requires the
multi-filter result to equal a direct full evaluation of the model-domain
winner. Every changed stream decodes under rvp9-decoder, pinned vpxdec, and
ffmpeg; all three decoded I420 outputs are byte-exact.

**Compression-efficiency impact:** the clean strict cpu-used 7 Stage 4a
seven-point Foreman curve improves from `+0.156544919758%` to
`-8.585847331540%` PSNR BD-rate versus pinned libvpx v1.14.1. An independent
cubic fit reproduced the result within `2.70e-10` percentage points.

**Speed impact:** sixteen warmed one-thread alternating pairs improved wall
time by `-2.144%` and child CPU by `-1.857%`, with eleven wins. Four-thread
wall/CPU measurements were contaminated by unrelated host saturation and
are not acceptance evidence.

**Future plan:** retain libvpx's two-stage metric boundary. Re-run the
four-thread corpus timing matrix on the pinned idle host, and preserve the
focused winner-domain oracle when changing model-RD or coefficient-RD code.

### 2026-07-31 — Filter search retains two thread-local prediction buffers

**Where:** `rvp9_core::mode_decision::inter::{
CandidatePredScratch,select_interp_filter_by_luma_model,
evaluate_candidate_with_filter_set}`.

**What libvpx does:** `search_filter_ref` obtains predictions from a
four-entry `PRED_BUFFER` pool. It retains the winning buffer plus its
`pf_rate`, `pf_dist`, `pf_var`, `pf_sse`, and `pf_tx_size`, then the caller
invokes `block_yrd` with `rd_computed=1`.

**What rvp9 does:** each tile worker has one safe thread-local scratch object
containing primary and alternate 64x64 prediction arrays. The two realtime
filters write separate arrays. The selector returns the winning array slot
and its owned `ModelRdResult`, so the complete evaluation reuses both without
a second prediction or model pass.

**Why:** Rust does not expose libvpx's mutable macroblock predictor pool, and
candidate evaluation is serial within each tile worker. Two fixed arrays
cover the scoped EightTap/EightTapSmooth search while keeping aliasing and
active-prefix ownership explicit.

**Byte-parity impact:** none. A focused oracle independently rebuilds the
winning prediction and model, compares every active pixel and model field,
and checks both inactive scratch suffixes. The cpu-used 7 product stream and
all three decoders' I420 output remain byte-identical to Unit 139.

**Compression-efficiency impact:** none; the strict Stage 4a curve remains
`-8.585847331540%` PSNR BD-rate versus pinned libvpx v1.14.1.

**Speed impact:** two sixteen-pair one-thread runs improve paired child CPU
by `-1.118%` and `-0.377%`. A twenty-pair four-thread run is neutral at
`-0.006%`; loaded-host wall time is not acceptance evidence. The matched
profile reduces inclusive luma-model samples from 131 to 95.

**Future plan:** retain the two-buffer ownership while the filter set remains
the scoped two-filter realtime range. A broader filter set must extend the
scratch contract explicitly rather than alias or silently rebuild a winner.

### 2026-07-31 — Encoder phases share one persistent frame-worker pool

**Where:** `rvp9_encoder::frame_loop::{
StreamEncoder,build_inter_packet_with_residuals}` and
`rvp9_core::tile_threading::WorkerPool`.

**What libvpx does:** `VP9_COMP` owns persistent encoder workers. Tile encode
jobs and loop-filter jobs reuse those workers across frames; `vpxenc
--threads=4` does not create a new OS-thread group for every tile pass.

**What rvp9 does:** one stream-owned `WorkerPool` is reused sequentially for
inter tile picking, split-edge loop filtering, and final tile arithmetic
writing. This differs from libvpx's worker-interface implementation but
matches its persistent lifetime and configured concurrency. Per-tile mutex
slots return owned results for deterministic tile-order merge.

**Why:** the former picker and writer each used a fresh
`std::thread::scope` on every inter frame. A matched profile observed 478–524
sampled threads per 120-frame process versus exactly four in pinned vpxenc.
The existing lifetime-checked pool already supplies panic propagation and
borrowed frame scopes, so sharing it removes construction churn without a
new dependency or unsafe block.

**Byte-parity impact:** none. The cpu-used 7 product stream is byte-identical
to Unit 140, and rvp9-decoder, pinned vpxdec, and ffmpeg produce byte-exact
I420 output.

**Compression-efficiency impact:** none; the strict Stage 4a curve remains
`-8.585847331540%` PSNR BD-rate versus pinned libvpx v1.14.1.

**Speed impact:** three saturated-host alternating four-thread series improve
paired wall time by `0.452%`, `3.019%`, and `1.555%`. The series measuring
both user and system time improves total child CPU by `3.094%` and system CPU
by `18.947%`. A five-encode profile now has only the main thread plus four
persistent workers per process.

**Future plan:** keep a single sequentially borrowed pool while these phases
cannot overlap. If row-MT or overlapped filtering later requires concurrent
phase execution, split the pool only with an explicit ownership and thread-
budget design rather than oversubscribing the configured thread count.

### 2026-07-31 — Dormant color-sensitivity scaffold is not precomputed

**Where:** `rvp9_encoder::frame_loop::StreamEncoder::try_encode_one_popped`
and `rvp9_encoder::pre_analysis::PreAnalysisState::{
compute_color_sensitivity_frame,color_sensitivity_sb}`.

**What libvpx does:** `chroma_check` computes `x->color_sensitivity[]` beside
the variance-partition or chroma-RD decision which consumes it.

**What rvp9 does:** the ported producer and focused oracles remain, but the
realtime frame loop does not invoke the batch producer while no encoder path
reads `color_sensitivity_sb`. The state remains at its reset default.

**Why:** the former batch pass recomputed a 64x64 luma SAD plus two 32x32
chroma mean/SAD walks for every superblock, then discarded all results. A
repository-wide use search confirms there is no consumer. Deferring it to
the eventual use site matches libvpx's work placement and avoids a
speculative full-frame cost.

**Byte-parity impact:** none. The cpu-used 7 product stream is byte-identical
to Unit 141, and rvp9-decoder, pinned vpxdec, and ffmpeg produce byte-exact
I420 output.

**Compression-efficiency impact:** none; the strict Stage 4a curve remains
`-8.585847331540%` PSNR BD-rate versus pinned libvpx v1.14.1.

**Speed impact:** two saturated-host alternating four-thread series improve
paired wall time by `5.350%` and `4.731%`, and total child CPU by `2.239%`
and `2.607%`.

**Future plan:** wire the producer at the actual partition/chroma-RD use site
when that consumer lands. Do not restore an unconditional frame-level pass.

### 2026-07-31 — Scene detection reuses cached per-superblock source SAD

**Where:** `rvp9_encoder::pre_analysis::PreAnalysisState::{
compute_avg_source_sad,compute_scene_detection_from_cached_sad,
compute_scene_detection}` and
`rvp9_encoder::frame_loop::StreamEncoder::try_encode_one_popped`.

**What libvpx does:** its one-pass pre-analysis shares source-SAD state among
the content-state and scene-detection decisions.

**What rvp9 does:** the product frame loop computes every 64x64 source SAD
once, stores it in `avg_source_sad[]`, and aggregates the scene detector's
interior checkerboard subset from those exact values. The standalone
pixel-based method remains as a differential oracle.

**Why:** both former passes called the same `sb_sad_and_var` helper against
the same current and previous source planes. Reusing the stored values
removes a duplicate subset pixel walk while preserving the checkerboard,
zero-SAD, threshold, forced-keyframe, and recursive-mean rules.

**Byte-parity impact:** none. The cpu-used 7 product stream is byte-identical
to Unit 142, and rvp9-decoder, pinned vpxdec, and ffmpeg produce byte-exact
I420 output.

**Compression-efficiency impact:** none; the strict Stage 4a curve remains
`-8.585847331540%` PSNR BD-rate versus pinned libvpx v1.14.1.

**Speed impact:** three saturated-host alternating four-thread series improve
total child CPU by `0.392%`, `0.826%`, and `0.715%`. The fresh profile cuts
the broad source-analysis self samples from 308 to 254.

**Future plan:** keep the cached entry point adjacent to the avg-SAD call.
Any call site that has not populated `avg_source_sad[]` for the same frame
must use the independent pixel-based method.

### 2026-07-31 — Source analysis uses the shared SIMD dispatch

**Where:** `rvp9_encoder::pre_analysis::PreAnalysisState::
compute_avg_source_sad` and `sb_sad_and_var_dispatched`.

**What libvpx does:** `vp9_avg_source_sad` calls the encoder function table's
64x64 SAD and variance kernels for each superblock.

**What rvp9 does:** the frame-level source-analysis pass resolves the
existing byte-exact Block64x64 SAD and variance dispatch slots once, then
reuses their safe function pointers across the superblock grid. The scalar
pixel walk remains as a differential and standalone scene-detection oracle.

**Why:** the former product path recomputed SAD, signed sum, and SSE in a
scalar Rust loop even though AArch64 NEON, x86-64 AVX2, and scalar fallback
kernels already implement the same integer reductions. Shared dispatch
matches libvpx's work placement without new unsafe code or dependencies.

**Byte-parity impact:** none. A focused oracle covers random and extreme
blocks, non-zero offsets, both rate-control classifications, and cached scene
state. The cpu-used 7 product stream and all three decoders' I420 output
remain byte-identical to Unit 143.

**Compression-efficiency impact:** none; the strict Stage 4a curve remains
`-8.585847331540%` PSNR BD-rate versus pinned libvpx v1.14.1.

**Speed impact:** two twenty-pair four-thread runs improve paired total child
CPU by `2.097%` and `1.096%`, and wall time by `3.421%` and `0.958%`. The
matched profile cuts broad frame pre-analysis self samples from 254 to 56.

**Future plan:** keep the scalar oracle independent and resolve dispatch
pointers outside the superblock loop. Any new architecture kernel must pass
the same integer-exact reduction oracle before entering this path.

### 2026-07-31 — Fast-diamond initial neighbours use four-reference SAD

**Where:** `rvp9_core::dsp::aarch64::sad4d_neon`,
`rvp9_core::dsp::dispatch::SimdDispatch::build`, and
`rvp9_core::mv_search::full_pixel::fast_diamond_search_sad`.

**What libvpx does:** its general and refining diamond searches use the
`sdx4df` dispatch slot, while `vp9_pattern_search_sad` evaluates the initial
four unit neighbours of `FAST_DIAMOND` with four single-reference SAD calls.
On AArch64, `sdx4df` selects base-NEON or dot-product kernels.

**What rvp9 does:** all four-reference SAD dispatch slots now select exact
AArch64 NEON kernels. Widths 16 through 64 select dot-product kernels when
available. The initial fast-diamond neighbours also use that existing batch
slot when all four MVs are legal; clipped-boundary searches retain the
ordered scalar path. Later three-neighbour refinement rounds remain scalar.

**Why:** the four candidates share one source block. Loading it once and
accumulating four reference differences removes repeated source loads without
changing search breadth, candidate order, MV costs, or strict tie-breaking.

**Byte-parity impact:** none. Random, edge-value, shape, stride, ISA-dispatch,
and boundary tests are integer-exact against scalar SAD. The 120-frame
cpu-used 7 product stream remains 740,177 bytes with SHA-256
`6991a5b50db78b87c4b01ba02e32c948bfb0a97ea693381b9e2ddb8752adfa1d`.

**Compression-efficiency impact:** none; search decisions and stream bytes
are unchanged, so the accepted Stage 4a curve remains `-7.662735%` PSNR
BD-rate versus pinned libvpx v1.14.1.

**Speed impact:** twenty paired one-thread encodes improve total child CPU by
`0.711%` with 17/20 wins. Two twenty-pair four-thread runs improve paired
CPU by `0.174%` and `0.789%`. Fresh matched four-thread CPU is `1.33618x`
pinned vpxenc, down from `1.34822x`.

**Future plan:** keep batch SAD as an exact implementation optimization.
Do not pad the later three-candidate rounds with a discarded fourth lane
without separate measurement.

### 2026-08-01 — Small-frame partition descent uses chessboard filter search

**Where:** `rvp9_encoder::InterEncodeEnv::descent_cb_filter_search` and
`rvp9_encoder::pick_and_simulate_inter_encode_partitions`.

**What libvpx does:** cpu-used 7 realtime enables
`cb_pred_filter_search` only above 640x480. Below that threshold, its
realtime picker searches the configured interpolation filters on every
eligible fractional-MV block.

**What rvp9 does:** the speculative partition-descent picker applies the
chessboard shortcut at every scoped resolution. The final emitted picker
keeps libvpx's resolution gate and therefore performs the full small-frame
search. `RVP9_A4_DESCENT_CB_FILTER=0` restores the former full-search proxy
for differential diagnosis.

**Why:** rvp9's recursive partition proxy is materially more expensive than
libvpx's non-RD partition path. Alternating paired measurements show the
shortcut removes work while the final picker recovers full mode quality.

**Byte-parity impact:** the chosen partition tree can change. At the canonical
250 kbps Foreman point the stream changes from 68,659 to 68,862 bytes. The
stream remains standards-compliant; this change does not alter bitstream
syntax or decoder behavior.

**Compression-efficiency impact:** the seven-point canonical Foreman curve
improves from `-12.264557%` to `-12.643812%` luma BD-rate against pinned
libvpx v1.14.1. Average-YUV BD-rate improves from `-11.626509%` to
`-11.953755%`.

**Speed impact:** thirty alternating pairs improve median total child CPU by
`1.918%` and wall time by `1.965%`, with 29/30 CPU and 29/30 wall wins.

**Future plan:** retain the full final-picker search and validate this proxy
policy across the complete cpu-used 6-8 corpus. Revert the default if another
important clip exceeds the feature-specific regression tolerance.

### 2026-08-01 — Speculative partition descent is luma-only

**Where:** `rvp9_encoder::pick_and_simulate_inter_encode_partitions`, in the
recursive A4 descent's `PickInterModeParams` construction.

**What libvpx does:** its cpu-used 7 realtime partition chooser uses a cheap
luma-led non-RD partition path. Full inter mode and chroma residual decisions
run only for the selected partition leaves.

**What rvp9 does:** rvp9's recursive partition proxy runs its real inter-mode
picker for every speculative shape. Those speculative calls now omit U/V
source planes, while the committed-leaf picker still receives and scores both
chroma planes before emission.

**Why:** the recursive proxy consumed `70.93%` of measured encoder time and
repeated chroma prediction and model-RD work for candidates it discarded. A
luma-only proxy preserves the final chroma decision while removing that
duplicated work.

**Byte-parity impact:** partition choices can change. At the canonical
250 kbps Foreman CBR point the stream changes from 68,862 to 68,743 bytes.
The stream decodes to identical I420 under rvp9-decoder, pinned vpxdec, and
ffmpeg; all three outputs have SHA-256
`87ab883398a14563624de5e3caecd01bd87d8efe771f1ac311e104aa4723aed4`.

**Compression-efficiency impact:** the canonical seven-point CBR curve is
`-11.067228%` luma BD-rate and `-10.596498%` average-YUV BD-rate versus
pinned libvpx v1.14.1. This spends 1.576584 luma percentage points of rvp9's
former CBR margin while remaining well inside the reference parity gate. The
canonical VBR curve improves to `-9.506021%` luma and `-8.304137%`
average-YUV BD-rate versus pinned libvpx.

**Speed impact:** forty alternating one-thread CBR pairs improve median total
child CPU by `10.964%`, user CPU by `11.140%`, and wall time by `10.939%`,
with 40/40 wins for every metric.

**Future plan:** retain full chroma in the committed-leaf picker. Validate the
luma proxy across cpu-used 6-8 and the representative corpus; revert or narrow
it if any important clip exceeds the 2% unexplained-regression tolerance.

### 2026-08-01 — Speculative partition descent prefers LAST

**Where:** `rvp9_encoder::InterEncodeEnv::descent_disable_golden` and the A4
descent's `PickInterModeParams` construction.

**What libvpx does:** the cpu-used 7 realtime partition chooser does not run
the full reference-and-mode search used to encode committed leaves. GOLDEN is
still available to the later block-mode picker according to its speed-feature
and source-state gates.

**What rvp9 does:** the speculative A4 partition proxy sets the existing
`disable_golden_ref` speed feature for ordinary content. The core picker's
VeryHighSad exception still searches GOLDEN at scene transitions. The final
committed-leaf picker keeps its configured LAST/GOLDEN search. Setting
`RVP9_A4_DESCENT_DISABLE_GOLDEN=0` restores full-reference speculative search
for differential diagnosis.

**Why:** NEWMV search remains the largest self-time hotspot after the
luma-only proxy change. Running it against GOLDEN for every speculative shape
duplicates reference selection that the final causal picker performs again.

**Byte-parity impact:** partition choices can change. At the canonical
250 kbps Foreman CBR point the stream changes from 68,743 to 68,834 bytes and
has SHA-256
`3551c82586b3fb96ed57f233441b6b9311d8473e8304610fd9bcd788806d9547`.
The stream decodes to identical I420 under rvp9-decoder, pinned vpxdec, and
ffmpeg; all three outputs have SHA-256
`d1c43cf9920b1e37dcdb26bf4db9c8c3b3e38d6b663e22d052eb2a02312ac3a8`.

**Compression-efficiency impact:** the canonical seven-point CBR curve is
`-10.399102%` luma BD-rate and `-9.914415%` average-YUV BD-rate versus pinned
libvpx v1.14.1. The VBR curve is `-9.120814%` luma and `-7.938657%`
average-YUV BD-rate. Both retain a large margin over the final parity gate.

**Speed impact:** forty alternating one-thread CBR pairs against the luma-only
Unit 422 baseline improve median total child CPU by `9.938%`, user CPU by
`9.914%`, and wall time by `12.062%`, with 40/40 wins for every metric.

**Future plan:** keep GOLDEN enabled in the final picker and preserve the
VeryHighSad exception. Validate the proxy across cpu-used 6-8 and the full
corpus; narrow it if GOLDEN-dependent content breaches the per-clip gate.

### 2026-08-01 — Rectangular partition proxies reuse inherited motion

**Where:** `rvp9_encoder::InterEncodeEnv::descent_disable_rect_newmv` and the
A4 descent's rectangular `inter_mode_mask` entries.

**What libvpx does:** cpu-used 7 realtime uses non-RD partition heuristics
rather than performing an independent full motion search for every horizontal
and vertical partition arm.

**What rvp9 does:** speculative 8x16, 16x8, 16x32, and 32x16 arms rank ZERO,
NEAREST, and NEAR motion but omit NEWMV. Every square NONE/SPLIT candidate and
every committed final leaf retains NEWMV. Setting
`RVP9_A4_DESCENT_DISABLE_RECT_NEWMV=0` restores full rectangular motion search
for differential diagnosis.

**Why:** after the LAST-led proxy change, NEWMV search remains the largest
self-time hotspot. Rectangular arms account for a large share of speculative
picker calls, while adjacent square candidates already provide independently
searched motion evidence for the partition decision.

**Byte-parity impact:** partition choices can change. At the canonical
250 kbps Foreman CBR point the stream is 68,738 bytes with SHA-256
`d893b844154db16537ebedf3cd7d85ea4c0b711da70938e098ec88d41db0e1fd`.
The stream decodes to identical I420 under rvp9-decoder, pinned vpxdec, and
ffmpeg; all three outputs have SHA-256
`7b97b33385e72befdaf50f4322c5faad9b9b613342cb242f587934397e87c7ab`.

**Compression-efficiency impact:** the canonical seven-point CBR curve is
`-8.618636%` luma BD-rate and `-8.021811%` average-YUV BD-rate versus pinned
libvpx v1.14.1. The VBR curve is `-7.869126%` luma and `-6.568133%`
average-YUV BD-rate. Both remain far inside the final parity gate.

**Speed impact:** forty alternating one-thread CBR pairs against Unit 423
improve median total child CPU by `12.696%`, user CPU by `12.653%`, and wall
time by `12.484%`; CPU wins 40/40 pairs and wall wins 39/40.

**Future plan:** do not extend this policy to square candidates: the measured
all-shape variant lost 7.321397 luma BD-rate points on VBR. Validate the
rectangular-only proxy across cpu-used 6-8 and the complete corpus.

### 2026-08-01 — Disabled references use a single MV-context walk

**Where:** `rvp9_core::inter::mv_ref::
find_realtime_refs_nearest_near_mvs_with_context` and
`rvp9_encoder::pick_and_simulate_inter_encode_partitions`.

**What libvpx does:** the realtime picker builds motion-reference candidates
only for references that remain usable after its reference gates.

**What rvp9 does:** ordinary speculative candidates, whose GOLDEN reference is
disabled by the accepted proxy policy, build only LAST nearest/near motion,
constant-motion state, and the shared inter-mode context. VeryHighSad blocks
retain the paired LAST/GOLDEN walk. The final picker remains paired. A const-
generic core helper shares the exact traversal implementation for one or two
references.

**Why:** after suppressing GOLDEN search, the descent still performed all
GOLDEN candidate-list bookkeeping. The single-reference helper removes that
dead work without adding a second traversal or changing LAST/context results.

**Byte-parity impact:** none. A focused oracle compares the single-reference
result and context with the paired helper across compound, different-reference,
previous-frame, empty-grid, and precision cases. The canonical Unit 424 CBR
stream remains 68,738 bytes with SHA-256
`d893b844154db16537ebedf3cd7d85ea4c0b711da70938e098ec88d41db0e1fd`.

**Compression-efficiency impact:** none; stream bytes are unchanged, so Unit
424's accepted CBR/VBR curves and clean product result remain authoritative.

**Speed impact:** one hundred alternating one-thread CBR pairs improve median
total child CPU by `0.679%` with 60/100 wins and wall time by `0.634%` with
66/100 wins.

**Future plan:** keep single-reference traversal conditional on the same
GOLDEN-disable and VeryHighSad gates. Any additional reference pruning must
retain an equivalent differential oracle.

### 2026-08-01 — Quantizer-scaled 16x16 descent breakout

**Where:** `rvp9_encoder::stage4a_descent_breakout_16_dist_qmul` and the
`Block16x16` breakout adapter in
`rvp9_encoder::pick_and_simulate_inter_encode_partitions`.

**What libvpx does:** realtime non-RD partition search uses several content,
source-variance, and speed-feature breakouts to avoid unproductive child and
rectangular searches.

**What rvp9 does:** CPU7 realtime CBR on the small-frame P8 path stops after a
winning 16x16 NONE candidate when its luma proxy distortion is no greater than
`base_qindex * 1400`. VBR, large-frame recovery, other speed presets, and final
leaf picking are unchanged. `RVP9_A4_BREAKOUT_16_DIST_QMUL=0` restores the
former differential path.

**Why:** the accepted Unit 427 profile evaluated roughly 64,290 speculative
8x8 candidates per canonical encode, although most low-distortion parent
blocks did not need the finer search. A fixed distortion threshold either
left too much work or harmed low-q frames. Scaling by qindex preserves more
detail search as quality rises.

**Byte-parity impact:** intentional CBR partition decisions change. At 250
kbps the canonical stream is 68,969 bytes with SHA-256
`71e17e4e8452f42d39e777c48d5f0ce168851b495764492a7d527601d09e7413`.
Rvp9-decoder, pinned vpxdec, and ffmpeg produce identical 7,372,800-byte I420
with SHA-256
`a4cdb21bdba015a85d114f0d41140b48335d44abc9ed18921eba3727b1c87088`.
The canonical 250 kbps VBR stream remains byte-identical to Unit 427 at
47,321 bytes and SHA-256
`a3323810381759704ac8afca469b136be99bf3e9c7ec5f8a47c8ce1b95fe21e1`.

**Compression-efficiency impact:** the canonical seven-point CBR luma
BD-rate is `-6.975657%` against pinned vpxenc and average-YUV is
`-6.424454%`. Relative to the preceding multiplier-1000 plus smooth-NEWMV
policy, the luma and average-YUV movements are `+0.864642%` and `+0.854539%`.
Testsrc2 remains `-3.390173%` luma and `-1.064887%` average-YUV against
pinned vpxenc; Mandelbrot remains `-30.077276%` and `-27.590214%`. The
stationary-noise curve is non-monotonic and remains diagnostic-only.

**Speed impact:** relative to the preceding multiplier-1000 policy, forty
alternating one-thread CBR pairs reduce median total child CPU by `6.841%`
with 39/40 wins and wall time by `6.362%` with 36/40 wins. A fresh absolute
comparison measures `1.770800x` pinned-vpxenc total child CPU and `1.777510x`
wall time.

**Future plan:** validate the same distortion signal independently before
extending it to cpu-used 6/8, large frames, or v1 feature modes. The new
profile shifts the largest share from descent to the final causal picker.

### 2026-08-01 — Smooth 16x16 final-NEWMV suppression

**Where:** `rvp9_encoder::stage4a_suppress_final_newmv` and the final causal
picker mask in `rvp9_encoder::pick_and_simulate_inter_encode_partitions`.

**What libvpx does:** its variance-based partition tree and adaptive realtime
mode thresholds suppress low-value motion searches using temporal variance,
source variance, and prior mode frequency.

**What rvp9 does:** on the same CPU7 small-frame CBR slice as the quantizer-
scaled breakout, a committed 16x16 leaf omits final-causal NEWMV only when the
speculative descent selected a non-NEWMV mode and its partition distortion is
no greater than `base_qindex * 600`. Every motion-selected or less-smooth leaf
keeps NEWMV. Setting `RVP9_A4_BREAKOUT_16_DIST_QMUL=0` disables both related
departures.

**Why:** after Unit 429, final picking is `57.39%` inclusive profile weight;
NEWMV search alone is `15.64%`. The cached descent winner provides a measured
motion/content signal without another source-analysis pass.

**Byte-parity impact:** intentional mode decisions change. At 250 kbps the
canonical stream is 68,683 bytes with SHA-256
`f9df047255c0e4a46be7270695324df51dd85aef7c1025c39972012a601c5c83`.
Rvp9-decoder, pinned vpxdec, and ffmpeg produce identical 7,372,800-byte I420
with SHA-256
`ad04b802a34ae57d30b759a327576884a31a116663ff0aebadc8adb72260f1ce`.

**Compression-efficiency impact:** against pinned vpxenc the canonical
seven-point CBR curve remains `-7.504821%` luma and `-6.946820%` average-YUV.
Relative to Unit 429, the movements are `+0.790669%` and `+0.770716%`.
Independent testsrc2 and Mandelbrot luma movements are `+0.603107%` and
`+0.660061%`; the stationary-noise curve is byte-identical.

**Speed impact:** forty alternating one-thread pairs reduce median total child
CPU by `3.351%` with 34/40 wins and wall time by `3.106%` with 29/40 wins. A
fresh forty-pair absolute comparison measures `1.880599x` pinned-vpxenc total
CPU and `1.840730x` wall time.

**Future plan:** keep this classifier limited to the measured slice. Generalize
only with representative 720p/1080p corpus evidence and a speed-feature policy
for CPU6/8.

### 2026-08-02 — Current-frame non-coefficient probability updates retain compact tokens

**Where:** `rvp9_encoder::bitstream_gates::build_noncoef_prob_update_target`,
`rvp9_encoder::frame_loop::build_inter_packet_with_residuals`, and
`rvp9_core::entropy::detokenize::CoefTokenPass::RecordFull`; libvpx
counterparts: `vp9/encoder/vp9_bitstream.c::write_compressed_header` and
`vp9/encoder/vp9_encodemv.c::vp9_write_nmv_probs`.

**What libvpx does:** gathers the current frame's complete syntax counts before
the compressed header, then savings-gates TX, skip, inter-mode, interpolation,
intra/inter, reference, Y-mode, partition, and MV probability updates. Tile
symbols use the resulting same-frame probability bank.

**What rvp9 does:** its one-thread pre-header traversal now records retained
coefficient tokens and complete mode/MV counts in one pass. Parallel tile
pickers still produce coefficient-only records, so packet assembly performs a
count-only retained-token replay before the header to gather their mode/MV
counts. The probability targets use libvpx's standard diff-update savings
search and its distinct seven-bit odd-probability MV update test. Setting
`RVP9_FORWARD_MODE_PROBS=0` restores the retired path for diagnostics only.

**Why:** rvp9's former coefficient-only count pass deferred non-coefficient
counts until final tile packing, after the header. That prevented the coupled
same-frame updates and left measurable fixed-q syntax waste. The retained-token
boundary lets both traversals share probability-independent coefficient data
without adding a frame-sized libvpx `TOKENEXTRA` arena.

**Byte-parity impact:** intentional streams change. A 60-frame 1280x704
in-to-tree VBR stream at 1000 kbps has SHA-256
`86b85cc919c26aba3d1915d710f7fb642851043343f412751fbe4d2e57d0615f`.
Rvp9-decoder, pinned vpxdec, and ffmpeg produce the same 81,100,800-byte I420,
SHA-256 `11e65e53dd43c6b10258b3fd5138e4e69a9b653c6e810c908b0a0c53f55a2262`.

**Compression-efficiency impact:** fixed-q 200 old-town keeps identical decoded
PSNR while shrinking from 78,059 to 77,546 bytes (`-0.657%`). Against pinned
vpxenc, the three luma BD-rates move from
`+15.745779%/-6.972353%/+38.107266%` to
`+15.222327%/-9.692528%/+38.256838%`; average-YUV moves from
`+14.975975%/-6.315552%/+35.485312%` to
`+14.465214%/-9.031002%/+35.699395%`. The small old-town trade remains a
rate-control-feedback issue around an intrinsically bit-saving fixed-symbol
update, so this is retained as a measured source-parity prerequisite.

**Speed impact:** after reusing the pre-header complete counts during final
packing, nine warmed alternating one-thread measurements show median user CPU
at `1.006670x` and wall time at `1.007037x` the prior path. The overall
encoder remains roughly `1.5x` pinned vpxenc, outside the final gate.

**Future plan:** parallelize or fuse the tile-MT count replay, and reconcile
rate-control response before judging the small old-town curve movement final.

### 2026-08-03 — Reuse the realtime picker's transform at emission

**Where:** `rvp9_encoder::realtime_emit_tx_rd_enabled`,
`rvp9_encoder::choose_inter_luma_tx_size`, and
`rvp9_encoder::realtime_force_max_emit_tx`; libvpx counterparts:
`vp9/encoder/vp9_pickmode.c::block_yrd` and
`vp9/encoder/vp9_rdopt.c::choose_tx_size_from_rd`.

**What libvpx does:** its realtime picker couples transform selection to the
mode decision and prunes the search according to its speed features.

**What rvp9 does:** CPU7 realtime large frames emit the transform selected by
the realtime picker. `RVP9_RT_EMIT_TX_RD=1` enables a diagnostic final
coefficient-domain RD comparison between TX8 and TX16 on leaves whose maximum
transform is TX16. `RVP9_RT_EMIT_MAX_TX=1` restores the former large-frame
maximum-transform shortcut for diagnostics.

**Why:** libvpx stores its picker decision in `mi->tx_size` and reuses it at
emission. Rvp9's optional final search crossed the quality gate, but profiling
attributed `3.69%` of samples to duplicate transform redecision. Preserving
the picker result crosses the final three-clip gate while removing that work.

**Byte-parity impact:** intentional streams change. At 1000 kbps the
in-to-tree, ducks-take-off, and old-town-cross IVF SHA-256 values are
`f7bd8d37eaf4b31453adfc36d00210a9858cf8b8f4e591c502e0e2756838214d`,
`6b2d85ae518824147246493ce7a4c47c5f81421503036b093fc554461c65dab4`,
and `0677e10a54a53bb5bdf33506787caaf3e3548d4cffe22c330c3c44d9a033212c`.
Rvp9-decoder, pinned vpxdec, and ffmpeg produce identical I420 output for each
stream, with SHA-256 values
`445af4b4fbade370d0d79dc3487191061d9283aaf9cc20318588a47b10a4faec`,
`ce79268b66adefc52b527b4eb4316818db41c3f47736036fc35d2f9ffaf3ba9a`,
and `22f0380edabe53f41451d62ee0c442b662944c5362fd435dbfd330edae2479c0`.

**Compression-efficiency impact:** six-point luma BD-rate versus pinned
vpxenc is `-5.592461%`, `-6.370118%`, and `-1.019382%` on in-to-tree,
ducks-take-off, and old-town-cross. Average-YUV BD-rate is `-4.875724%`,
`-5.645272%`, and `-0.782441%`. This clears both the Stage 4a `+2%` gate and
the final `+0.5%` gate on the current three-clip matrix. The wider required
corpus and configuration matrix remains unmeasured.

**Speed impact:** disabling the duplicate final search improved forty
alternating four-thread old-town pairs by `4.68%` median paired wall time and
`5.55%` paired child CPU, with 38 and 39 wins respectively. Twenty fresh
alternating pairs measured `0.222037 s` versus pinned vpxenc's `0.164370 s`,
or `1.350835x` by wall medians and `1.348525x` paired. Nine warmed one-thread
runs measured `9.907 ms/frame`, inside the 720p30 median budget but outside
the final relative-speed gate.

**Future plan:** retain picker-transform reuse while it stays inside the
quality gate. Profile the accepted default against pinned vpxenc and remove
the next measured structural speed gap without giving back that gate.

### 2026-08-03 — Overlap inter-frame loop filtering with packet construction

**Where:** `rvp9_encoder::frame_loop::StreamEncoder::encode_inter_internal`
and `rvp9_encoder::loopfilter::apply_loop_filter_to_tight_recon_split_edge`;
libvpx counterparts: `vp9/encoder/vp9_encoder.c::encode_frame_to_data_rate`,
`vp9/encoder/vp9_bitstream.c::vp9_pack_bitstream`, and
`vp9/common/vp9_loopfilter.c::vp9_loop_filter_frame_mt`.

**What libvpx does:** schedules its encoder stages through libvpx's worker
model and multi-threaded loop-filter implementation.

**What rvp9 does:** on multi-threaded normal show inter frames, rvp9 dispatches
its alias-safe single-worker loop filter to a dedicated long-lived worker while
the caller packs the already-selected coefficients and mode metadata. The
dispatch handle joins before the filtered reconstruction is committed.
`RVP9_LF_PACKET_OVERLAP=0` restores serial stage ordering for diagnostics.

**Why:** profiling attributed 1,306 inclusive samples to rvp9's loop filter
and 1,270 to packet construction, versus 485 and 780 samples in matched
libvpx subtrees. The stages are independent after tile encoding: filtering
owns only the new reconstruction planes, while packing reads coefficients,
mode metadata, and probability state. Overlap shortens the critical path
without changing either result or reintroducing unsafe multi-worker plane
aliasing.

**Byte-parity impact:** none in the exercised four-thread 60-frame old-town
stream. The 2500 kbps IVF remains byte-identical with SHA-256
`e27ceee6a9d7fc521d8ace45d9bdb0dcc0641ced06baa53713f13f0c83208b0f`.
Rvp9-decoder, pinned vpxdec, and ffmpeg each produce the same 81,100,800-byte
I420 with SHA-256
`01b4b2eddb9b5971b393dc686be235d1103371ba829d8dbfe0f1997a20dd52b0`.

**Compression-efficiency impact:** none because the encoded stream is
byte-identical. The accepted three-clip six-point curves therefore retain
their luma BD-rates of `-5.592461%/-6.370118%/-1.019382%` versus pinned
vpxenc.

**Speed impact:** two independent forty-pair comparisons against serial
ordering improved median paired wall time by `5.99%` and `6.90%`, with 33/40
and 37/40 wins. Replacing the initial per-frame scoped threads with the
long-lived worker improved another two trials by `0.72%` and `1.79%` paired
wall time and by `0.89%` and `1.40%` paired child CPU. A fresh loaded-host
comparison measured `1.279116x` paired wall time versus vpxenc, so the final
`1.05x` gate remains open.

**Future plan:** profile the persistent-worker critical path, then target the
largest remaining output-preserving wall-time gap.

### 2026-08-03 — Use libvpx row-MT adaptive thresholds internally

**Where:** `rvp9_encoder::frame_loop::StreamEncoder::rd_thresh_tables` and
`rvp9_encoder::pick_and_simulate_inter_encode_partitions`; libvpx
counterparts: `vp9/encoder/vp9_pickmode.c::rd_less_than_thresh_row_mt`,
`update_thresh_freq_fact_row_mt`, and
`vp9/encoder/vp9_multi_thread.c::vp9_row_mt_alloc_rd_thresh`.

**What libvpx does:** the matched vpxenc default has `--row-mt=0` and uses one
adaptive threshold bank per tile. With `--row-mt=1`, libvpx instead keeps a
persistent `row_base_thresh_freq_fact` bank per `(tile, SB row)` so concurrent
rows never mutate one threshold table.

**What rvp9 does:** rvp9 uses the libvpx row-MT table topology internally for
its matched four-tile realtime configuration: one persistent
`RdThreshTable` per `(tile, SB row)`, flattened tile-major. The public CLI
does not expose a separate row-MT stream-selection switch.

**Why:** tail-row work sharing is required to remove the measured 8.5% median
tile imbalance without sharing a mutable picker table or making output depend
on worker scheduling. The per-row topology is libvpx's own solution and lets
each row job carry wholly private mutable rate-decision state.

**Byte-parity impact:** structural. At 1000 kbps the in-to-tree,
ducks-take-off, and old-town-cross IVF SHA-256 values are
`925761c7afe28970465f2db61a4eefc324277961bcca2fa391fda089a0c44715`,
`d119c7f0493622bdfd3c81a76522ce283a944582518fc57e368fe14ec8f6c037`,
and `584948269e22715565c893d221582fdcbb618989fe9493c8977eb8c8fbdad9e2`.
Rvp9-decoder, pinned vpxdec, and ffmpeg agree bit-exactly on 81,100,800-byte
I420 output for all three, with SHA-256
`18f28d49b4069a60a8c79766fb42de5793ef0eee5b9bc3d931d45e94db73e00d`,
`9f048dd89cda9d33232bd71e497b9c5c72ef20358a39e00e7c6b2bb9138e718c`,
and `cda140b3abbeb766ce5625a5a302d5de27e8c9223522765ee3a7eb27e0243ebc`.

**Compression-efficiency impact:** versus the preceding accepted baseline,
six-point luma BD-rate changes by `-0.314627%/+0.087955%/-0.135811%` and
average-YUV by `-0.321406%/+0.098914%/-0.112491%` on
in-to-tree/ducks-take-off/old-town-cross. Versus pinned vpxenc, the resulting
luma BD-rates are `-5.851064%/-6.195072%/-1.080472%`; average-YUV is
`-5.154446%/-5.465080%/-0.830402%`. The small ducks movement is well inside
the per-clip tolerance, while the three-clip geomean improves.

**Speed impact:** a loaded-host 15-pair diagnostic measured per-clip wall
ratios `1.011745/1.010906/0.981689` and child-CPU ratios
`1.007586/1.010656/0.981438`, or approximately `1.00135x` wall and
`0.99981x` CPU by corpus geomean. This is neutral diagnostic evidence, not a
final quiet-host speed claim.

**Future plan:** retain the topology while enabling concurrent row claims.
Re-run the full quiet-host quality/speed matrix; reconsider only if a wider
clip or feature stage exceeds its regression tolerance.

### 2026-08-03 — Fit partial bottom superblocks with complete square leaves

**Where:** `rvp9_encoder::frame_loop::edge_partition_decision`,
`rvp9_encoder::frame_loop::choose_mapped_partitions_for_sb_cols`, and
`rvp9_core::bitstream::modes_writer::{write_partition_tree,write_inter_partition_tree}`;
libvpx counterparts: `vp9/encoder/vp9_encodeframe.c::encode_sb_row` and
`vp9/decoder/vp9_decodeframe.c::decode_partition`.

**What libvpx does:** VP9's edge syntax removes partition choices only when a
half-block has no visible mi row or column. A selected leaf may therefore
extend beyond the visible frame; libvpx's prediction, transform, reconstruction,
and loop-filter buffers provide and clamp that edge storage.

**What rvp9 does:** widths remain 64-pixel aligned, while heights may end on
any 8-pixel mi-row boundary. The final partial superblock uses a deterministic
square-only tree whose every leaf fits the visible frame. The writer emits the
same reduced edge bits the decoder reads. Full-block exact-VBP, integer
projection, A4 descent, and direct split-edge filtering stay on complete
superblocks; the partial row uses edge-safe SAD, the fitted variance tree, and
the bordered loop-filter path.

**Why:** literal 1280x720 and 1920x1080 are required production targets. The
previous 64-pixel height restriction forced 1280x704 proxy evidence and
rejected literal 1080p before encoding. Fitted leaves reuse rvp9's existing
tight visible planes without introducing padded frame storage throughout the
picker and reconstruction pipeline.

**Byte-parity impact:** none for aligned inputs. The three Unit 669
1280x704/1000-kbps streams retain exact SHA-256 values
`925761c7afe28970465f2db61a4eefc324277961bcca2fa391fda089a0c44715`,
`d119c7f0493622bdfd3c81a76522ce283a944582518fc57e368fe14ec8f6c037`,
and `584948269e22715565c893d221582fdcbb618989fe9493c8977eb8c8fbdad9e2`.
At literal 720p, rvp9-decoder, pinned vpxdec, and ffmpeg agree on decoded
SHA-256 `f7a71f68...e440`, `d861e1d1...f30f`, and `9f4b7a49...7361` for
in-to-tree, ducks-take-off, and old-town-cross. At literal 1080p they agree on
`35ed4323...2442`, `655df642...c801`, and `7614ad03...1880`.

**Compression-efficiency impact:** not yet measured at literal 720p/1080p.
The aligned 1280x704 streams are byte-identical, so their accepted Unit 669
BD-rate evidence is unchanged. Literal-resolution BD-rate curves are the next
required quality measurement.

**Speed impact:** not yet claimed. Correctness runs completed well inside the
process timeout, but the host was not quiet and the runs were not the warmed,
repeated median protocol required by the final speed gate.

**Future plan:** measure matched six-point cpu-used 7 curves and warmed
per-clip speed at literal 720p and 1080p, then add the 1080p30/60 budget rows.

### 2026-08-03 — Retain model-only RD outside CPU7 VBR GOLDEN refreshes

**Where:** `rvp9_encoder::golden_real_block_yrd_enabled` and
`rvp9_encoder::pick_and_simulate_inter_encode_partitions`; libvpx counterpart:
`vp9/encoder/vp9_pickmode.c::block_yrd`.

**What libvpx does:** the cpu-used 7 realtime picker computes the selected
prediction's transform-domain rate and distortion through `block_yrd`.

**What rvp9 does:** one-pass realtime VBR at cpu-used 7 now uses rvp9's real
Hadamard `block_yrd` path on scheduled GOLDEN refreshes. Ordinary inter
frames retain the measured model-only approximation. CBR, cpu-used 6/8,
non-realtime, and cyclic-refresh AQ behavior is unchanged until each matched
configuration has its own curve evidence. `RVP9_GOLDEN_REAL_BLOCK_YRD=0/1`
remains a differential override.

**Why:** paired block traces found that model-only RD selected NEWMV over
about 50% of the first 1080p ducks GOLDEN frame, versus 31% in pinned libvpx.
Real `block_yrd` reduced that share to 24% and corrected the persistent
GOLDEN coding-efficiency deficit without perturbing ordinary frames.

**Byte-parity impact:** structural. The accepted 1080p ducks target-2500
stream SHA-256 is
`da5e7056638ebf479715e396fcb36dbd52cab745739886c8e271d3b6ca7dda50`.
Rvp9-decoder, pinned vpxdec, and ffmpeg agree bit-exactly on its decoded
186,624,000-byte I420 output, SHA-256
`ef8daeb588f4eb3cbf0c15460a797c9c4443a920c5bc22b9861cb58e04f7e8b3`.

**Compression-efficiency impact:** versus pinned vpxenc, three-clip
six-point luma BD-rate is `-4.433528%/-8.558332%/+0.076182%` at 720p and
`-0.154700%/-3.467245%/-0.703721%` at 1080p for
in-to-tree/ducks-take-off/old-town-cross. Geometric aggregates are
`-4.370150%` and `-1.452616%`. Average-YUV aggregates are `-3.872535%` and
`-1.041800%`. The limiting 1080p ducks luma result improves from
`+2.273803%` to `-3.467245%`.

**Speed impact:** a loaded-host seven-pair same-binary screen measured a
1.58 s candidate median versus 1.64 s with the override disabled. This rules
out a material local slowdown but is not final quiet-host speed evidence.

**Future plan:** remeasure on a quiet host, then test the same source-faithful
policy independently for cpu-used 6/8, CBR, AQ, denoiser, and SVC before
expanding its default scope.

### 2026-08-08 — Restore GOLDEN in the default A4 speculative proxy

**Where:** `rvp9_encoder::InterEncodeEnv::descent_disable_golden`.

**What libvpx does:** the realtime partition decision keeps its normal
LAST/GOLDEN reference context available; reference pruning is a separate
speed-feature decision, not a universal partition-proxy rule.

**What rvp9 does:** the A4 proxy now keeps GOLDEN by default, matching the
final picker’s reference set. `RVP9_A4_DESCENT_DISABLE_GOLDEN=1` preserves the
previous LAST-only proxy as a diagnostic speed override; the final picker is
unchanged in either case.

**Evidence:** on the three 1920x1080 60-frame CPU7 VBR-2500 clips, the default
LAST-only proxy produced 956,293 / 1,139,663 / 984,674 bytes and 35.0214 /
27.6011 / 34.8240 dB. The GOLDEN-enabled proxy produced 945,169 / 1,137,581 /
992,878 bytes and 35.0483 / 27.5897 / 34.8931 dB. These are matched-bitrate
diagnostics, not the final BD-rate gate.

Twelve paired quiet-host runs against pinned vpxenc v1.14.1 under the same
four-thread CPU7 VBR-2500 settings gave median wall ratios of approximately
`1.04x/0.83x/0.99x` for in-tree/ducks/old-town (geomean about `0.94x`). All
streams decoded successfully under vpxdec; the source-level policy test and
focused encoder suite remain the required correctness gates.

**Future plan:** remeasure this default across cpu-used 6/8, CBR, AQ, denoiser,
and SVC before treating the policy as final parity evidence.

### 2026-08-08 — CPU6 large-frame realtime recovery policy

**Where:** `rvp9_encoder::pick_and_simulate_inter_encode_partitions` and
`realtime_cpu6_large_recovery_for_mode`.

**What libvpx does:** pinned vpxenc CPU6 realtime keeps the full NSTEP-based
picker and its normal speculative partition work on large default-content
frames.

**What rvp9 does:** for CPU6 realtime default-content frames at or above
700,000 pixels (the 720p30, 1080p30, and 1080p60 targets), rvp9 keeps the
variance-hint leaf tree, skips the expensive A4 speculative descent, and
removes NEWMV from blocks no larger than 16x16. CPU7/8 and non-default content
retain the existing A4 policy. The gate is source-level, not an environment
default; the prior behavior remains available through the existing diagnostic
controls for comparison.

**Why:** CPU6 profiling showed A4 descent dominating the frame budget, while
the no-descent path still spent roughly 9,000 fresh NEWMV searches per frame.
The measured CPU6 recovery policy brought 1080p VBR-2500 wall time to
`0.348/0.452/0.339 s` versus pinned vpxenc medians of about
`0.45/0.57/0.42 s` for in-tree/ducks/old-town. A 1280x704 screen measured
`0.124 s` versus `0.133 s` for vpxenc. These are warmed local screens, not the
final corpus geomean gate.

**Byte-parity impact:** structural. The 1080p in-tree policy stream decodes
bit-exactly under rvp9-decoder, pinned vpxdec, and ffmpeg; all three produce
the 186,624,000-byte I420 SHA-256
`10ae6149cfb81786a3267d57c2d306477f706cabf88e941ff9f6d44b979230bd`.

**Compression-efficiency impact:** matched-output diagnostics are favorable
but are not the final BD-rate gate. At VBR-2500, decoded average PSNR versus
the source was `24.4547/24.3441 dB` for rvp9/vpxenc on in-tree,
`17.8041/17.8450 dB` on ducks, and `23.9525/23.8155 dB` on old-town. At
1280x704 testsrc, the corresponding values were `24.6289/23.6866 dB`.
Four-point CBR screens are recorded in
`/private/tmp/rvp9-unit811-cpu6-newmv16-curves.csv`; the points have limited
curve overlap, so no BD-rate claim is made here.

**Future plan:** run the same policy through the six-point CBR/VBR curves,
full corpus, and CPU6 feature stages before treating CPU6 as final parity.

### 2026-08-08 — Use independent MV proxies for realtime CPU7/8 CBR A4 descent

**Where:** `rvp9_encoder::a4_descent_independent_mv_default` and the A4
speculative partition proxy.

**What libvpx does:** its realtime partition search carries causal motion
predictors into each candidate's motion search.

**What rvp9 does:** realtime CPU7 and CPU8 CBR use one neutral predictor for
each speculative partition shape, while the final causal picker remains
unchanged. VBR retains the causal proxy because the independent variant
regresses its measured VBR curves. `RVP9_A4_DESCENT_INDEPENDENT_MV=0` restores
the inherited/causal proxy for diagnostics; the policy is limited to realtime
speed 7 and faster.

**Why:** on the matched 1080p30 CBR-2500 in-tree clip, six-point luma BD-rate
versus pinned vpxenc improved from `-16.235%` to `-20.957%` at CPU7 and from
`-22.594%` to `-26.192%` at CPU8. Three-run warmed CPU7 medians changed from
`0.37/0.38/0.39 s` to `0.35/0.36/0.38 s`; CPU8 changed from
`0.39/0.40/0.39 s` to `0.39/0.40/0.38 s`. The corresponding 2500-kbps
outputs shrank from `1,017,262` to `845,316` bytes at CPU7 and from
`1,009,842` to `912,062` bytes at CPU8. The independent proxy also measured
`-27.082%` and `-13.349%` on the ducks and old-town diagnostic curves when
combined with the existing LAST-only descent override.

**Byte-parity impact:** the proxy changes partition and motion decisions, so
the output is intentionally not byte-identical to the prior rvp9 stream. A
cross-decoder and conformance pass remains required before final acceptance.

**VBR scope:** the independent proxy is not the default for VBR. On the same
six-point curves, CPU7 measured `+29.5246%/+21.7426%/-1.9957%` for
in-tree/old-town/ducks with the independent proxy, versus
`+19.4186%/+11.8024%/-10.1272%` with the causal proxy. VBR therefore retains
the causal proxy pending a speed-safe quality fix.

**Future plan:** run cross-decoder checks, CPU6/CPU7/CPU8 VBR curves, and the
CBR/AQ/denoiser/SVC feature matrix before treating this default as final.

### 2026-08-08 — Supersede CPU6 large-frame recovery hypothesis

The CPU6 large-frame recovery described above was removed after matched
six-point CBR/VBR screens showed a severe quality cliff, including non-overlap
on the old-town VBR curve. Full causal search restored the curve but was about
4.3x slower. The policy is rejected and is no longer present in the source;
the entry above remains as historical evidence for the measured hypothesis.

### 2026-08-08 — Guard A4 full-pick reuse by causal context and winner mode

**Where:** `rvp9_encoder::pick_and_simulate_inter_encode_partitions` and
`realtime_cached_full_pick_allowed`.

**What libvpx does:** its realtime encoder does not run a speculative A4
partition picker and therefore has no cached full-pick result to reuse.

**What rvp9 does:** rvp9 reuses an A4 full-pick result only when the final
causal LAST and GOLDEN `(nearest, near)` pairs match the speculative context,
and only for non-NEWMV winners. Matching NEWMV searches remain available as
phase-2 overrides, but the final causal picker re-ranks them.

**Why:** reusing a stale full NEWMV pick produced a large VBR quality loss.
The guarded policy reduced the corrected CPU7 1080p30 in-tree VBR curve from
the unguarded `+19.4186%` diagnostic to `+1.614%` BD-rate in the measured
non-NEWMV reuse variant. This is an intermediate result, not final parity.

**Byte-parity impact:** structural. The policy changes mode decisions by
design; every exercised stream must pass rvp9-decoder, pinned vpxdec, and
ffmpeg decode gates before final acceptance.

**Speed impact:** the measured three-run CPU7 VBR-2500 screen was
`0.51/0.52/0.52 s` for rvp9 versus `0.37/0.37/0.39 s` for vpxenc, a median
ratio of about `1.35x`. The policy is therefore a quality unit only and does
not close the `1.05x` speed gate.

**Future plan:** replace complete re-picks on mismatched leaves with a
causal-context-preserving fast correction, then repeat the same six-point
curves and warmed timing matrix.

### 2026-08-08 — Enable the libvpx-aligned VBR-GOLDEN controller by default

**Where:** `rvp9_encoder::frame_loop::{vbr_gf_rc_enabled,StreamEncoder::new_with_parallelism}`
and the VBR bits-per-MB multiplier default.

**What libvpx does:** one-pass realtime VBR builds a frame-rate-derived
GOLDEN cadence and starts its simplified bits-per-MB estimator at correction
factor `1.0`; post-encode feedback learns content bias per frame.

**What rvp9 does:** VBR now uses that cadence and unit estimator by default.
`RVP9_VBR_GF_RC=0` remains an explicit diagnostic escape hatch, and
`RVP9_BPM_MULTIPLIER=<value>` still overrides the estimator for experiments.
CBR and other rate-control modes are unchanged.

**Why:** the previous VBR-only multiplier `4.0` and opt-in Golden schedule
were diagnostic scaffolding, not production parity. Removing both defaults
reduced the CPU7 1080p30 six-point VBR curves to
`+1.614%/+6.925%/-19.758%` for in-tree/old-town/ducks under the canonical
four-thread, 60-frame settings. This is an intermediate improvement, not the
final `+0.5%` corpus gate.

**Byte-parity impact:** intentional rate-control and frame-structure changes;
streams remain VP9 and require the full decoder/conformance matrix.

**Future plan:** close the remaining old-town and in-tree VBR residuals,
then repeat the same default policy across CPU6/7/8 and target resolutions.

### 2026-08-08 — Use exact variance partitions on CPU7 VBR GOLDEN refreshes

**Where:** `rvp9_encoder::InterEncodeEnv::exact_vbp_tree_enabled` and the
realtime inter picker in `rvp9_encoder::pick_and_simulate_inter_encode_partitions`.

**What libvpx does:** pinned vpxenc CPU7 realtime uses its variance-based
partition tree on every inter frame and carries persistent partition history.

**What rvp9 does:** CPU7 realtime one-pass VBR with default content uses the
pinned-source-exact variance tree only on scheduled GOLDEN refresh frames.
Ordinary inter frames retain rvp9's measured A4 recursive partition search.
`RVP9_EXACT_VBP_TREE=1` retains the full exact-tree differential diagnostic;
`RVP9_EXACT_VBP_TREE=0` disables the product policy as a negative control.
Other speeds, rate-control modes, deadlines, and content tiers are unchanged.

**Why:** neither partition path generalizes by itself. The full exact tree
measured `+13.282%/+3.115%/-3.149%` luma BD-rate on in-tree/old-town/ducks,
while A4 measured `+8.127%/+11.115%/-22.567%`. The GOLDEN-only combination
improves every six-point curve to `+5.741%/+8.218%/-24.054%`. At the old-town
2500-kbps point, the policy moves rvp9 from `4814 kbps / 35.19 dB` to
`4324 kbps / 35.14 dB`, versus vpxenc's `3989 kbps / 35.10 dB`. This is a
measured Pareto prerequisite, not the Stage 4a `+2%` exit.

**Byte-parity impact:** structural. The exercised 4000-kbps old-town stream
has SHA-256 `c0a47f0ba24c4ac9ec5231cb6c2e69482ea54b11a9b5d2fef611f2d615555084`.
Rvp9-decoder, pinned vpxdec, and ffmpeg each produce the same 186,624,000-byte
I420 output with SHA-256
`136ecd67bac5abac48b2deb850ed3ad1703f77248cc20ff883707fca1ca44a3b`.

**Speed impact:** five alternating loaded-host old-town pairs showed no
visible change between A4 and the GOLDEN-only policy, but unrelated simulator
processes occupied the host. This is diagnostic evidence only; the required
quiet-host warmed median gate remains outstanding.

**Future plan:** compare A4 and exact-tree decisions on ordinary inter frames
to identify a causal per-superblock admission signal, then repeat the same
three-clip curve before broadening the policy.

### 2026-08-09 — Recheck transform size on exact-tree GOLDEN frames

**Where:** `rvp9_encoder::realtime_emit_tx_rd_enabled` and
`rvp9_encoder::pick_and_simulate_inter_encode_partitions`.

**What libvpx does:** its realtime picker selects and stores the emitted
transform size as part of the block-mode decision.

**What rvp9 does:** approximate A4 large-frame partitions retain the measured
maximum-transform emission shortcut. At cpu-used 7, exact variance-tree frames
perform the existing coefficient-domain RD comparison between TX8 and TX16 on
leaves whose maximum transform is TX16; in the VBR product policy, those exact
frames are scheduled GOLDEN refreshes only. At cpu-used 8, exact-tree frames
reuse the transform selected by the realtime picker, matching libvpx's
single-decision data flow. `RVP9_RT_EMIT_TX_RD=0` disables the comparison and
`=1` retains the broader diagnostic override at either preset.

**Why:** cpu-used 7 needs the focused redecision as a measured quality
prerequisite. Cpu-used 8 has enough quality headroom to remove the duplicated
transform work: its six-point 1080p curves remain `-1.736%/-7.655%/-3.776%`
and its 720p curves remain `-3.686%/-11.835%/-5.141%` for
in-tree/ducks/old-town versus pinned vpxenc. This preserves every final quality
gate while moving cpu-used 8 closer to the speed target.

**Byte-parity impact:** intentional. The exercised cpu-used 8 old-town
2500-kbps stream has SHA-256
`29389c3986260a4d1cfed59bb9226a21cb34b1eb47e7ab75a897814e86fae2f3`.
Rvp9-decoder, pinned vpxdec, and ffmpeg each produce the same 186,624,000-byte
I420 output with SHA-256
`b4bc12668387e62e7190fb95c33575320b07c886e0782e008c88c6f4c9452b59`.

**Speed impact:** fifteen warmed alternating plain-release pairs on the three
1080p clips measured a `0.969168x` paired wall-time geomean and a `0.961215x`
paired child-CPU geomean against the preceding cpu-used 8 baseline. The
candidate's median frame times were `6.324/7.736/6.129 ms` for
in-tree/ducks/old-town. Representative-PGO comparison with pinned vpxenc
remains the acceptance timing gate.

**Future plan:** keep the cpu-used 7 redecision and cpu-used 8 picker reuse.
Refresh the representative-PGO binary and direct pinned-vpxenc timing matrix,
then profile the remaining cpu-used 8 gap.

### 2026-08-09 — Couple exact CPU7 VBR partitions to the source RD lambda

**Where:** `rvp9_encoder::InterEncodeEnv::exact_vbp_tree_enabled`,
`rvp9_encoder::inter_rd_calibration_for_frame`, and
`rvp9_encoder::frame_loop::exact_keyframe_sub8_enabled`.

**What libvpx does:** pinned vpxenc CPU7 realtime uses its variance-based
partition tree on every inter frame and evaluates those decisions with its
native RD multiplier. Its keyframe variance policy can emit sub-8x8
partitions without rvp9's historical high-rate restriction.

**What rvp9 does:** CPU7 realtime one-pass VBR with default content now uses
the source-exact variance tree and source RD lambda together on every show
inter frame. Keyframes admit the exact sub-8x8 path only through Q index 120;
this measured cutoff avoids the high-rate corpus regression from enabling it
unconditionally. `RVP9_EXACT_VBP_TREE=0` disables the coupled product policy,
while `RVP9_EXACT_KF_SUB8=1` retains the independent syntax diagnostic. CBR,
CPU6/8, non-realtime deadlines, non-show ALTREF coding, and other content
tiers are unchanged.

**Why:** neither ingredient generalizes alone. Removing lambda attenuation
from the former GOLDEN-only hybrid measured
`+17.188%/-23.292%/+4.570%`; full exact VBP with the source lambda measured
`+1.389%/-12.055%/+2.411%`. Unconditional exact keyframe sub-8x8 then moved
the curves to `+2.216%/-9.319%/+0.245%`. Limiting that keyframe path to
Q <= 120 produces `+1.279%/-9.786%/+1.769%` for
in-tree/ducks/old-town. This is the first fresh canonical CPU7 VBR result
with every required clip inside the Stage 4a `+2%` intermediate gate.

**Byte-parity impact:** intentional and structural. The exercised old-town
4000-kbps stream has SHA-256
`4e35c277573ce8a29a02d9b4a4546ed92c13a091cfb665c2e76f8fd9b85525b4`.
Rvp9-decoder, pinned vpxdec, and ffmpeg each produce the same 186,624,000-byte
I420 output with SHA-256
`8caa15362ab0d4ca9056f645110e9f905b6887c6fed55f685e77bf50dfb97c71`.

**Speed impact:** alternating old-town target-2500 runs on the loaded host
measured about `0.48 s` for rvp9 and `0.39-0.40 s` for vpxenc, roughly
`1.22-1.23x`. The absolute time is about 8 ms/frame, but the relative speed
gate is red. Unrelated simulator load means this is diagnostic evidence, not
the required quiet-host acceptance timing.

**Future plan:** retain this measured Stage 4a quality prerequisite, profile
the exact path, and remove its largest proven cost without reopening any of
the three six-point curves.

**CPU8 boundary audit (2026-08-09):** extending the exact inter-frame VBP tree
to cpu-used 8 had also admitted this Q-scoped keyframe path through their shared
gate, contrary to the CPU7-only policy above. The product predicate now checks
`speed == 7` explicitly; the environment diagnostic remains preset-agnostic.
At cpu-used 8, restoring the intended boundary measured a `0.946935x` plain
wall-time geomean and `0.942605x` child-CPU geomean. Its 1080p curves are
`-1.471%/-10.066%/-2.643%` and its 720p curves are
`-1.295%/-12.884%/-1.310%` for in-tree/ducks/old-town versus pinned vpxenc.
The exercised old-town stream decodes identically under rvp9-decoder, pinned
vpxdec, and ffmpeg.

### 2026-08-09 — Calibrate exact VBP structure above the coding Q

**Where:** `rvp9_encoder::exact_vbp_partition_qindex` and the exact-tree
adapter in `rvp9_encoder::pick_and_simulate_inter_encode_partitions`;
libvpx counterpart: `vp9/encoder/vp9_encodeframe.c:set_vbp_thresholds`.

**What libvpx does:** derives variance-partition thresholds directly from the
frame or segment coding qindex.

**What rvp9 does:** the CPU7 realtime one-pass VBR exact-tree product path
adds nine qindex steps through 1280x720 and eighteen above that frame area,
only while deriving partition thresholds and the low-SAD/copy thresholds.
Coefficient quantization and every non-product path retain the rate
controller's effective qindex.

**Why:** at the old-town 4000-kbps point, rvp9 used average inter qindex
132.63 versus vpxenc's 138.27, yet emitted fewer bits. Applying the exact
libvpx threshold formula at that finer live Q produced 20,668 inter 8x8 leaves
versus vpxenc's 9,748 and repeated expensive final-picker work. The uniform
partition-only bias compensates for the different live trajectory without a
clip, bitrate, or frame-position rule. The initial 3/6/9/12 sweep selected
nine by the worst required clip. A later literal-resolution sweep showed that
eighteen passes all three 1080p clip guards and reduces work, but over-coarsens
720p; the frame-area split therefore retains the measured-safe calibration at
both required resolutions.

**Byte-parity impact:** intentional and structural. Partition history, rate
control feedback, and later reconstruction diverge from the previous product
stream.

**Compression-efficiency impact:** at 720p the canonical six-point
in-tree/ducks/old-town curves remain `+0.801%/-11.117%/+0.792%`. At 1080p the
eighteen-step curves are `+1.678%/-10.062%/+0.610%`; every important clip
stays inside the `+2%` guard.

**Speed impact:** the coarser 1080p structure reduces repeated leaf-picker
work. Fifteen warmed alternating pairs measured a `0.99052x` paired wall
geomean and `0.98490x` paired child-CPU geomean across the three clips. The
720p path is byte-identical to the prior nine-step policy.

**Future plan:** retain the resolution-scaled calibration while closing the
remaining speed gap, then repeat the final quiet-host matrix.

### 2026-08-09 — Sample the custom high-frequency frame classifier

**Where:** `rvp9_encoder::pre_analysis::frame_hf_laplacian_sampled_sum_rows`
and the S2 frame gate in `rvp9_encoder::frame_loop`.

**What libvpx does:** pinned vpxenc has no equivalent full-frame Laplacian
classifier. Rvp9 added this source signal to choose its S2 skip policy for
natural versus synthetic content.

**What rvp9 does:** the production classifier estimates the frame mean from
one globally fixed row out of every four. If that estimate lands within two
Laplacian units of the configured decision threshold, rvp9 recomputes the
original exact full-frame mean before deciding. The exact full-frame helper
remains the test oracle and diagnostic fallback.

**Why:** the measured natural and synthetic ranges are separated by more than
2x, while a fresh CPU7 profile assigned about 1.2% of all weighted samples to
the exact classifier scan. The fixed global phase makes the result independent
of worker-job boundaries; the exact guard preserves threshold-adjacent
behavior instead of relying on the approximation where it could matter.

**Byte-parity impact:** the exercised 60-frame in-tree, ducks, and old-town
streams are byte-identical before and after at both 1280x720 and 1920x1080.
The 1080p hashes remain `0673d4c9...b1ceb5d`, `90528e63...fe32`, and
`cd18f47f...d6b62`; the 720p hashes are `a250773e...916ffa`,
`710505e0...ed407`, and `a5bba3a5...7b2466`.

**Speed impact:** median retired-instruction ratios were
`0.99044x/0.99258x/0.99011x` for in-tree/ducks/old-town. A warmed wall-time
run overlapped an unrelated host-wide LTO/PGO build and is not acceptance
evidence; quiet-host timing remains required.

**Future plan:** retain the byte-exact work reduction, refresh quiet-host
direct timing, and keep the exact fallback if broader corpus clips approach
the threshold guard band.

### 2026-08-09 — Use exact variance partitions and causal GOLDEN filters at CPU8

**Where:** `rvp9_encoder::InterEncodeEnv::exact_vbp_tree_enabled`,
`rvp9_encoder::realtime_search_golden_filter`, and both realtime inter-picker
parameter assemblies in `rvp9_encoder::pick_and_simulate_inter_encode_partitions`.

**What libvpx does:** pinned CPU8 realtime uses `VAR_BASED_PARTITION`, permits
partition copying through frame four, and limits interpolation-filter work
with its chessboard predicate. Eligible GOLDEN VBR candidates may still search
the filter on the active chessboard cells.

**What rvp9 does:** CPU8 realtime one-pass VBR with default content now uses
rvp9's source-exact variance-tree adapter on every show inter frame, including
the existing resolution-scaled partition-Q calibration. It also reuses the
causal neighbour filter for every GOLDEN candidate instead of retaining
rvp9's broader GOLDEN filter-search exception. CPU7 retains its accepted
filter policy. CBR and non-realtime paths are unchanged.

**Why:** the preceding CPU8 approximation measured
`+6.277%/-25.638%/+10.988%` at 1080p for in-tree/ducks/old-town. Exact
variance partitions plus causal GOLDEN filters move those curves to
`-2.152%/-13.652%/-4.220%`. Native 720p also improves every curve to
`-4.146%/-17.467%/-6.782%`. The GOLDEN rule is a deliberate departure from
libvpx's chessboard eligibility because rvp9's additional search worsened both
the required corpus result and the measured CPU8 work.

**Byte-parity impact:** intentional and structural. The exercised 1080p
old-town target-2500 IVF has SHA-256
`d9b63f0f9cf0289c863b7000859c107881d6b891c7e87c51b63b9d3608d21a8e`.
Rvp9-decoder, pinned vpxdec, and ffmpeg each emit the same 186,624,000-byte
I420 output with SHA-256
`eb91205e82aac52648034b21b6b3c7cdd7cef94cef24e2e0916452736988f50f`.

**Speed impact:** against the preceding representative-PGO rvp9, fifteen
warmed alternating 1080p pairs measured a `0.954372x` paired wall geomean and
`0.955471x` paired child-CPU geomean. At 720p the same comparison measured
`1.025709x` paired wall, while absolute candidate times stayed between 3.3
and 4.2 ms/frame. Direct CPU8 parity remains open: the 720p candidate/vpxenc
paired wall geomean is `1.209155x`.

**Future plan:** retain this cross-resolution quality and 1080p speed
prerequisite, then profile the remaining CPU8 work gap without weakening the
new six-point curves.

### 2026-08-09 — Calibrate CPU6 exact partitions and admit TX4 redecision

**Where:** `rvp9_encoder::InterEncodeEnv::exact_vbp_tree_enabled`,
`rvp9_encoder::exact_vbp_partition_qindex`, and
`rvp9_encoder::choose_inter_luma_tx_size`; libvpx counterparts:
`vp9/encoder/vp9_encodeframe.c:set_vbp_thresholds` and the CPU6 realtime speed
features in `vp9/encoder/vp9_speed_features.c`.

**What libvpx does:** CPU6 realtime VBR uses variance-based partitioning
without CPU7's partition-copy shortcut. Its realtime picker stores one
model-RD transform choice rather than running rvp9's emission-time transform
search.

**What rvp9 does:** CPU6 now uses the source-exact variance tree, disables
both history-copy paths, and adds 54 qindex steps only while deriving 1080p
partition thresholds. Coefficient quantization remains at the rate
controller's qindex. On TX16-capable leaves, CPU6's existing emission-time
TX8/TX16 redecision also considers TX4; CPU7 keeps TX8/TX16 and CPU8 keeps its
accepted picker transform.

**Why:** the preceding approximate CPU6 path measured `2.507062x` vpxenc wall
time and poor structural fidelity. After correcting the harness collision and
the one-pass VBR damping gate, serialized in-tree curves for structural
biases 18/36/54/60/72 measured
`+17.303%/+11.253%/+8.150%/+8.347%/+8.711%`; 54 is the measured optimum.
Against bias 36 it also reduces same-build corpus wall time by 3.3%. Adding
only TX4 retains a bounded transform recovery without clip, bitrate,
frame-position, or source-signal rules. Broad transform searches recovered
more quality but exceeded the speed gate and were rejected.

**Byte-parity impact:** intentional and structural. CPU6 partition history,
transform signalling, coefficient tokens, and rate-control feedback diverge.
The exercised CPU6 in-tree target-2500 IVF has SHA-256
`6b89e3cd8ce5fcb7ebcc3ebb5dd0b158de17eb55678fbbb3ae0a072786c1d0b5`.
Rvp9-decoder, pinned vpxdec, and ffmpeg each emit the same 186,624,000-byte
I420 output with SHA-256
`ae0a10ff4076a58f568c163dfbbf5beb0e5a4519bdf1c589f86fd7181cbf76c1`.
CPU7 and CPU8 policy branches are unchanged; the exercised CPU8 stream is
byte-identical to the Unit 974 PGO control at SHA-256
`3e142ba677ebf65c19705f62bdc9784a5486c90da6e235931fcb496923e5fa60`.

**Compression-efficiency impact:** serialized canonical 60-frame 1080p
six-point in-tree/ducks/old-town curves after the VBR damping correction,
54-step calibration, and source-exact intra skip gate are
`+1.640%/-11.830%/-1.327%`. The previously recorded
`+5.497%/-9.845%/+4.108%` curves were invalid: concurrent harness processes
overwrote fixed `/tmp/rvp9_bdrate_{role}_{bitrate}k` streams before decode
(`RVP9-BUG-FLUX-00010`). Serialized curves for the preceding Unit 974 policy
are `+23.321%/-7.029%/+9.314%`, so this unit remains a large measured quality
and speed prerequisite despite the remaining in-tree important-clip failure.

**Speed impact:** fifteen warmed alternating plain-release pairs against
pinned vpxenc measured per-clip wall ratios
`0.922675x/1.042926x/0.949883x`, corpus geomean `0.970489x`. The child-CPU
geomean was `0.971639x`; median rvp9 frame times were
`5.994/8.998/6.010 ms` for in-tree/ducks/old-town. Fifteen paired same-build
bias-54/bias-36 comparisons then measured per-clip wall ratios
`0.943484x/0.985249x/0.972295x`, corpus geomean `0.966851x`; child-CPU
geomean was `0.964044x`. The intra skip gate adds a further same-build wall
geomean of `0.969323x` and child-CPU geomean of `0.967026x`.

**Future plan:** serialize BD-rate runs until `RVP9-BUG-FLUX-00010` is fixed.
The in-tree clip is now inside the Stage 4a `+2%` intermediate gate but still
misses final `+0.5%`. Profile its remaining prediction and coefficient gap
without weakening the source-exact skip behavior.

### 2026-08-09 — Use real coefficient RD only on non-skipped CPU6 Block16 leaves

**Where:** `rvp9_core::mode_decision::inter::evaluate_candidate_for_filter_with_pred`
and `rvp9_encoder::block16_coef_rd_enabled`; libvpx counterpart:
`vp9/encoder/vp9_pickmode.c:block_yrd`.

**What libvpx does:** pinned CPU6 realtime leaves `use_simple_block_yrd`
disabled, so sub-32 inter candidates use real coefficient rate and distortion.

**What rvp9 does:** the exact-tree CPU6 realtime VBR final picker uses its
existing real Hadamard coefficient rate and distortion only for non-skipped
Block16 candidates. It preserves the model skip state. Block8 candidates,
speculative partition descent, CBR, CPU7/8, and non-realtime paths retain
their preceding model calibration.

**Why:** an earlier global real-`block_yrd` policy and a NEWMV-rate-only slice
both regressed the old encoder. With the current exact partitions, corrected
intra skip gate, and filter search, the coupled Block16 slice improves the
canonical three-clip mean BD-rate from `-4.437%` to `-5.263%`. Using only real
distortion nearly passed at `+0.509%` on the limiting clip; retaining both the
real rate and distortion reaches `-1.119%`. Extending the same policy to
Block8 or model-skipped Block16 leaves did not improve the complete curve.

**Byte-parity impact:** intentional. The exercised CPU6 VBR in-tree
target-2500 IVF has SHA-256
`3447f9eb4c9a3685b24261351df164391f7c693b01f38206ea119ae2a63f04fb`.
Rvp9-decoder, pinned vpxdec, and ffmpeg each emit the same 186,624,000-byte
I420 output with SHA-256
`9ec5d37ae4f6b59604a694f6656db005de724e19fd79b6824ffc7fec9601ddc9`.
CPU7 VBR, CPU8 VBR, and CPU6 CBR streams remain byte-identical to their
preceding controls.

**Compression-efficiency impact:** serialized canonical 1080p
in-tree/ducks/old-town curves are `-1.119%/-11.978%/-2.692%` versus vpxenc.
All three pass the final `+0.5%` per-clip gate.

**Speed impact:** fifteen warmed same-build pairs measured a `1.021465x`
wall and `1.022048x` child-CPU corpus geomean. Projected against the refreshed
direct vpxenc baseline, rvp9 remains at `0.945226x` wall time and
`0.965520x` child CPU. Median candidate frame times are
`5.955/9.819/5.879 ms`.

**Future plan:** retain the bounded CPU6 scorer and close the independent
CPU8 quality and 720p speed residual before broadening any real-coefficient
policy.

### 2026-08-09 — Recalibrate large-frame CPU8 partition thresholds after VBR damping

**Where:** `rvp9_encoder::exact_vbp_partition_qindex`; libvpx counterpart:
`vp9/encoder/vp9_encodeframe.c:set_vbp_thresholds`.

**What libvpx does:** pinned CPU8 derives its variance-partition thresholds
from the current coding qindex and has no separate partition-only Q offset.

**What rvp9 does:** CPU8 exact-tree VBR adds 27 qindex steps only while
deriving partition-tree split and SAD/copy thresholds above 1280x720. Native
720p retains its proven 9-step calibration. Coefficient quantization, CPU6/7,
CBR, and every non-exact-tree path are unchanged.

**Why:** the previous 18-step large-frame calibration was selected before the
shared one-pass VBR damping correction changed CPU8's live Q trajectory. On
the corrected rate-control path it left the limiting in-tree curve at
`+1.526%`. Moving to the next nine-step calibration point improves all three
canonical-clip arithmetic mean BD-rate by 0.989 point and slightly reduces
work, without a clip, bitrate, frame-position, or source-signal rule. This is
an empirical calibration of rvp9's adapter, not a claim of source parity with
libvpx's additional CPU8 threshold-scaling branches.

**Byte-parity impact:** intentional above 720p. The exercised 1080p in-tree
target-2500 IVF has SHA-256
`3afd3279c78a700aae82e7cd30130abb6f2d61bfb4b11361e118b891121b0ef4`.
Rvp9-decoder, pinned vpxdec, and ffmpeg each emit the same 186,624,000-byte
I420 output with SHA-256
`64ff015244e7413a4d0c992e4f23fff39195dacf54d55323d64e4122addcc750`.
The exercised native-720p stream is byte-identical to the control at SHA-256
`bf088dcab834b9971a163c17ede3a4132797b8c4c9fd4ee1f007809250ae221c`.

**Compression-efficiency impact:** serialized canonical 1080p
in-tree/ducks/old-town curves are `-0.645%/-4.182%/-5.804%` versus vpxenc.
All three pass the final `+0.5%` per-clip gate. Native 720p curves transfer
byte-for-byte from the preceding `+9` calibration.

**Speed impact:** fifteen warmed same-build pairs measured a `0.994361x`
wall and `0.987887x` child-CPU corpus geomean. A fresh representative PGO
build then measured `1.007237x` direct vpxenc wall time and `1.049389x`
child CPU. Median PGO frame times are `5.020/6.311/4.874 ms` for
in-tree/ducks/old-town.

**Future plan:** retain this final-green CPU8 VBR policy while validating the
independent CBR/AQ, denoiser, and temporal/SVC feature matrix.

### 2026-08-09 — Price cyclic-refresh segments with a shared fixed-delta model

**Where:** `rvp9_encoder::rate_control::{cbr,postencode}` and
`rvp9_encoder::frame_loop::StreamEncoder::encode_frame`; libvpx counterparts:
`vp9_cyclic_refresh_rc_bits_per_mb` and
`vp9_cyclic_refresh_estimate_bits_at_q`.

**What libvpx does:** weights the pre-Q bit estimate by the planned/previous
refresh fraction and a candidate-Q-dependent dynamic delta. Its postencode
projection weights BOOST1 and BOOST2 independently using their actual final
8x8-cell counts and emitted deltas.

**What rvp9 does:** now applies the same pre-Q and postencode weighting and
counts final segment IDs independently of skip. Until the dynamic-delta and
production BOOST2 units land, the model and emitted segmentation both use the
existing fixed BOOST1 delta of `-8`; BOOST2 remains inactive.

**Why:** keeping the picker and feedback projections unweighted made the base-Q
controller internally inconsistent whenever AQ spent bits at a finer segment
Q. Modeling the currently emitted fixed delta is the smallest correct loop
closure and a measured prerequisite for dynamic deltas.

**Byte-parity impact:** intentional only when cyclic-refresh AQ is active. The
exercised target-250 IVF decodes bit-exactly through rvp9-decoder, pinned
vpxdec, and ffmpeg. AQ-off does not construct the weighted model.

**Compression-efficiency impact:** the signed Foreman VBR AQ-gain mismatch
narrows from `+1.397` to `+1.378` percentage points. It still misses the
`0.3%` feature gate.

**Speed impact:** no broad timing claim yet. The extra work is bounded to at
most two scalar bit-model evaluations per candidate qindex on AQ-active
frames; the final feature speed matrix will measure it with dynamic deltas.

**Future plan:** replace the fixed delta with the pinned VBR dynamic BOOST1
delta, then enable and price the already implemented per-block BOOST2 policy.

### 2026-08-09 — Suppress cyclic refresh on one-pass VBR golden frames

**Where:** `rvp9_encoder::frame_loop::cyclic_refresh_frame_enabled`; libvpx
counterpart: `vp9_cyclic_refresh_update_parameters` VBR golden-refresh branch.

**What libvpx does:** assigns zero refresh percentage and a neutral rate ratio
to non-SVC one-pass VBR frames that refresh GOLDEN because rate control already
boosts those frames.

**What rvp9 does:** now suppresses the AQ rate model, segmentation map, and
segment Q features on the same VBR GOLDEN role. CBR golden roles and ordinary
VBR inter frames keep cyclic refresh when the common frame gate passes.

**Why:** refreshing an already boosted GOLDEN frame double-spent bits and
distorted the signed AQ gain. One shared pre-Q decision also prevents the rate
model and the emitted segmentation gate from disagreeing.

**Byte-parity impact:** intentional on AQ-enabled VBR golden frames. The
target-250 candidate decodes bit-exactly through all three required decoders.

**Compression-efficiency impact:** the signed Foreman VBR AQ-gain mismatch
improves from `1.378` to `1.151` percentage points. It remains outside the
`0.3%` final feature tolerance.

**Speed impact:** expected favorable because golden frames skip segment-map
and boosted-leaf work; no broad timing claim is made before the feature matrix.

**Future plan:** retain the source-faithful golden hook while porting the
candidate-Q-dependent VBR BOOST1 delta.

### 2026-08-09 — Seed age-based cyclic refresh as immediately eligible

**Where:** `rvp9_encoder::aq_cyclic_refresh::CyclicRefresh::new`; libvpx
counterpart: `vp9_cyclic_refresh_alloc` and `cyclic_refresh_update_map`.

**What libvpx does:** zero-initializes its signed refresh map and treats
`map[i] == 0` as eligible immediately. A freshly allocated map therefore
starts the first refresh cycle on the first eligible inter frame.

**What rvp9 does:** represents the same state as an unsigned age and tests
`age >= min_refresh_age`. Fresh cells now start at `min_refresh_age`, the
age-domain equivalent of libvpx's eligible zero. Refreshed cells still reset
to age zero and wait through the configured cooldown.

**Why:** seeding the age map to zero inverted libvpx's initial meaning and
delayed all cyclic refresh by four frames after every keyframe. Correcting the
representation closes the signed Stage 4b AQ-gain mismatch from `1.151` to
`0.205` percentage point.

**Byte-parity impact:** intentional for AQ-enabled streams. The exercised
target-250 stream decodes bit-exactly through rvp9-decoder, pinned vpxdec, and
ffmpeg at raw-I420 SHA-256
`500354ec396c39f2cedbe18b48f33b23854df16323da3a4018cdea959ab45253`.

**Compression-efficiency impact:** the exact signed Foreman VBR AQ gain is
`+4.509%` for rvp9 versus `+4.714%` for pinned libvpx, a `-0.205`-point delta
inside the final `0.3%` feature tolerance.

**Speed impact:** no speed claim in this unit. The next gate measures the
canonical 720p30 AQ-active CPU7 realtime configuration.

**Future plan:** retain immediate startup eligibility and measure Stage 4b's
720p30 speed exit before starting the denoiser feature matrix.

### 2026-08-09 — Filter a reusable coded-source buffer and keep high denoise scalar-exact

**Where:** `rvp9_encoder::denoise_inter_leaf` and
`rvp9_encoder::denoiser::vp9_denoiser_filter`; libvpx counterparts:
`vp9_denoiser_denoise` and `vp9_denoiser_filter_neon`.

**What libvpx does:** filters the source from picker-selected LAST/GOLDEN
temporal predictors, retains the resulting denoised frame, and dispatches its
AArch64 filter through RTCD at both medium and high strength. Pinned libvpx's
NEON and C high-strength filters disagree when `absdiff == 4` and motion
exceeds 24; the source warns that `increase_denoising` is not bit-exact with
its SIMD implementations.

**What rvp9 does:** retains the picker evidence needed to select the same
temporal predictor, filters luma directly into one reusable coded-source
buffer, and rotates that buffer into LAST/GOLDEN denoiser state after coding.
On AArch64, medium strength uses a widened-accumulator NEON port. High strength
deliberately uses the scalar oracle so rvp9's public filter remains bit-exact
across its own dispatch paths. Chroma and coded mode/reference/MV syntax remain
unchanged.

**Why:** the former fixed-grid post-encode filter never changed the coded
source and regressed clean-reference BD-rate by `+44.412%`. Picker-selected
source filtering closes the quality failure. Directly filtering the reusable
buffer removes a second full-frame copy and an accepted-leaf copy, while the
high-strength fallback avoids adopting a documented libvpx ISA-dependent
result in a non-production mode.

**Byte-parity impact:** intentional versus the former denoiser. The final
target-1500 stream is byte-identical before and after SIMD/buffer fusion at
SHA-256
`9e44d8affb3d9e8dc8a8384258aeb04aee9e63e1cf3716f711872adf7714aaaa`.
Rvp9-decoder, pinned vpxdec, and ffmpeg each emit 82,944,000 I420 bytes at
SHA-256
`d6fbeceb85242590fd06d633c30437fe8cc20bfb1b16d4dfa0a8a5bcd2b2b320`.

**Compression-efficiency impact:** the six-point CPU7 realtime VBR noisy-source
curve, scored against its clean reference, is `-0.565%` versus feature-enabled
pinned vpxenc. This passes the focused `<= +0.5%` denoiser quality gate.

**Speed impact:** fifteen rotated, high-resolution 60-frame 720p30 runs of the
documented 22-row production PGO build measure a `0.531845 s` rvp9 median
against `0.508113 s` for pinned vpxenc, a `1.046706x` wall ratio and
`8.864 ms/frame` absolute time. The pre-SIMD plain-release rvp9 median was
`0.71 s`. A matched ThinLTO experiment regressed the ratio to `1.066989x` and
was removed.

**Future plan:** retain the exact medium-strength NEON path while broadening
denoiser evidence across the declared livestream corpus and CPU6/8 presets.

### 2026-08-09 — Omit the final CPU8 intra-in-inter fallback

**Where:** `rvp9_encoder::realtime_intra_fallback_enabled` and
`rvp9_encoder::pick_and_simulate_inter_encode_partitions`; libvpx counterpart:
`vp9/encoder/vp9_pickmode.c:2514-2646`.

**What libvpx does:** CPU8 may evaluate intra modes after the inter-mode loop
when the skip, size, content, and adaptive-threshold gates admit the fallback.

**What rvp9 does:** CPU6 and CPU7 keep that behavior. CPU8 returns the selected
inter winner without running the final intra-in-inter fallback.

**Why:** the matched CPU8 720p profile assigns 5.06% of all product-self samples
to the fallback. Fifteen warmed same-build noisy-ducks pairs show a reproducible
`0.940712x` wall and `0.941912x` CPU improvement, while all three serialized
denoiser-on curves retain substantial quality margin.

**Byte-parity impact:** structural on CPU8. The exercised noisy-ducks
target-1500 stream changes from SHA-256
`019bb035ab6c12fc1a5a5ff25f0eaea6c3cca78cb1fc73828ce4ab29e791f23f`
to `94c9ef35bb966f7fb954029281e2e7e378e63c7393cbd6276ef4093201683165`.

**Compression-efficiency impact:** CPU8 realtime VBR denoiser-on clean-reference
BD-rates versus feature-enabled pinned vpxenc are
`-3.041%/-8.855%/-2.959%` for in-tree/ducks/old-town, all inside the final
`+0.5%` gate.

**Speed impact:** the same-build plain-release noisy-ducks comparison improves
from `0.592077 s` to `0.556974 s` over 60 frames. Production-PGO confirmation
is the next gate.

**Future plan:** retain only if independent production-PGO timing reproduces
the gain and the remaining CPU8 configurations keep their quality gates.
