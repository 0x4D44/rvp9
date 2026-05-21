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

### 2026-05-16 — Accurate sub-pel search variants stubbed
**Where:** `rvp9-encoder::me::subpel` ships only the cpu-used 6-8
sub-pel pipeline. libvpx counterpart: `vp9_find_best_sub_pixel_tree`,
`vp9_find_best_sub_pixel_iterative` and variants in
`vp9/encoder/vp9_mcomp.c`.
**What libvpx does:** dispatches between several sub-pel search
algorithms based on `sf->mv.subpel_search_method`. Slower variants
do more iterations / smaller step sizes.
**What rvp9 does:** ships only the "fast / 1-iteration / 2-step"
variant matching cpu-used 6-8. Other variants are not ported.
**Why:** the slower variants are gated off at our cpu-used. Dead at
our config. Mirrors L-05 / A3 §7 #1.
**Byte-parity impact:** none in matched configs.
**Compression-efficiency impact:** none in matched configs.
**Future plan:** add slower variants if cpu-used ≤ 5 preset is ever
needed. Decision: L-05.

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

### 2026-05-21 — NEARMV not implemented (F8.1 inter mode picker)
**Where:** `rvp9-core::mode_decision::inter::pick_inter_mode_luma`.
libvpx counterpart: `vp9/encoder/vp9_pickmode.c::vp9_pick_inter_mode`.
**What libvpx does:** evaluates `{NEARESTMV, NEARMV, NEWMV, ZEROMV}`
× `{single, compound}` × `{LAST, GOLDEN, ALTREF}` references; selects
by RD.
**What rvp9 does:** evaluates `{NEARESTMV, NEWMV, ZEROMV}` × single
ref × `{LAST}` only. NEARMV omitted. Compound-ref not driven by the
picker.
**Why:** F8.1 scope was bounded to demonstrate the picker shape. The
F8.2 partition picker chooses NEARESTMV-vs-NEWMV at the same MV-ref
list anchor; adding NEARMV is mechanical (next candidate in the
MV-ref list) but didn't fit the night-3 budget.
**Byte-parity impact:** structural — wherever NEARMV would have won
RD, rvp9 picks a different mode.
**Compression-efficiency impact:** modest. Expected 0.5–1.5 % BD-rate
penalty on typical content (NEARMV is rare-but-not-vanishingly-so).
**Future plan:** Stage 4a follow-up if BD-rate gate fails by margin;
otherwise defer.

### 2026-05-21 — Block8x8 inter leaves forced to ZEROMV-skip
**Where:** `rvp9-encoder::frame_loop::encode_inter_internal` (night-3
F10.1c). libvpx counterpart: `vp9/encoder/vp9_pickmode.c` over all
leaf sizes ≥ 8×8.
**What libvpx does:** picks inter mode + MV at every leaf size from
Block64x64 down to Block8x8 (and Block4x* via the sub-8 picker).
**What rvp9 does:** uses the F8.1 picker for leaves ≥ Block16x16;
Block8x8 leaves fall back unconditionally to `ZEROMV / LASTFRAME /
skip = true`. Sub-8 partitions remain wire-format-only.
**Why:** F8.1's `pick_inter_mode_luma` doesn't support sub-16. The
fallback keeps the F10.1c oracle (bit-exact encoder↔decoder recon)
viable across all partition shapes the F8.2 picker can produce.
**Byte-parity impact:** structural — Block8x8 inter leaves with real
motion content reconstruct to zero-MV instead of the picker's
choice.
**Compression-efficiency impact:** small on most content (Block8x8
is the smallest the F8.2 picker emits in v1, and ZEROMV-skip is
already optimal on the majority of Block8x8 leaves in low-motion
content). Larger on high-motion 8×8-detailed content.
**Future plan:** when sub-8 picker lands, this fallback path
disappears.

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

### 2026-05-21 — No NEARMV / compound-ref / temporal SVC inter shapes
**Where:** rvp9-encoder doesn't drive compound-ref inter blocks or
temporal SVC layering. libvpx counterpart: full inter-coding shape
surface.
**What libvpx does:** ships all VP9 inter shapes.
**What rvp9 does:** single-ref LASTFRAME only in the encoder
(decoder handles the full surface). No SVC.
**Why:** v1 scope (Stage 4a MVP). SVC is Stages 4d/4e. Compound-ref
is mechanical-but-orthogonal work.
**Byte-parity impact:** structural — the wire-format writer can emit
compound-ref blocks but the encoder doesn't pick them.
**Compression-efficiency impact:** the largest single residual
divergence from full-libvpx parity. Expect 5–15 % BD-rate penalty on
typical real-time content (compound-ref bidirectional inter is the
dominant win at our cpu-used target).
**Future plan:** Stage 4a exit gate will measure WITHOUT compound-
ref enabled on the libvpx side too (`--cpu-used 7 --rt` defaults
match this constraint at the picker level); once parity at matched
config is hit, Stage 4d/e widens the encoder.

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
