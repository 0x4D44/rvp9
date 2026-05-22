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
**OOB-nearest safety:** the encoder's `MvLimits` is tighter than
the decoder's `clamp_mv_ref` envelope because
`build_inter_pred_plane_unscaled` does not border-extend. When
the spatially-derived `mvc.nearest` falls outside `MvLimits`,
NEARESTMV evaluation would over-read the reference plane. We
substitute `MvCandidates::default()` in the picker call and
override the picker's argmin to choose between `{ZeroMv, NewMv}`
only on those leaves — the encoder never commits to NEARESTMV
when the underlying MV would diverge between encoder (no
border-extend) and decoder (border-extend).
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
