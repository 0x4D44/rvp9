# LLD: denoiser

## 1. Purpose

The content-aware temporal denoiser cleans the source picture in motion-compensated
"running-average" buffers so that quantization at low bitrate produces fewer visible
artefacts on noisy content (camera sensor noise, low-light, livestream uplink). It
runs per-block from inside the tile worker, between mode-decision finalisation and
block encode, on inter frames where the noise estimator (a sibling subsystem,
also owned by this LLD) reports `kDenLow` or higher.

The denoiser owns three things:

1. **Noise estimator** (`vp9_noise_estimate.{h,c}` — 302 + 54 LOC). Runs every
   `frame_period = 8` frames on the main thread, builds a 20-bin variance
   histogram across low-motion 16×16 blocks of the source, and classifies the
   frame's noise into `{kLowLow, kLow, kMedium, kHigh}`. The level drives the
   denoiser's per-block thresholds.

2. **Per-block kernel** (`vp9_denoiser_filter` — the only RTCD entry point this
   LLD owns; `vp9/encoder/vp9_denoiser.c:68-180` scalar). Reads source +
   motion-compensated reference, writes a denoised average. Returns
   `COPY_BLOCK` (no denoise applied) or `FILTER_BLOCK` (apply the denoised
   buffer back into source).

3. **Per-block decision + ref-frame swap** (`vp9_denoiser_denoise` plus
   `perform_motion_compensation` plus `vp9_denoiser_update_ref_frame`;
   `vp9/encoder/vp9_denoiser.c:336-437`, `:187-334`, `:782-822`). The decision
   gates the kernel and is the mode-decision integration boundary —
   `FILTER_ZEROMV_BLOCK` triggers `recheck_zeromv_after_denoising` in the
   mode-decision LLD.

What this LLD does **not** own:

- The `recheck_zeromv_after_denoising` cost-recompute itself
  (`vp9/encoder/vp9_pickmode.c:1376-1450`, mode-decision LLD §recheck;
  00-overview §4.7 sets the boundary explicitly).
- Skin detection (`vp9_skin_detection.c`) — provides `vp9_compute_skin_block` /
  `vp9_compute_skin_sb`; called as a kernel by this LLD but its implementation
  belongs to mode-decision (it is also used by CR-AQ and the variance-tree
  partitioner). See §3 below for the call-out.
- `cpi->consec_zero_mv[]` lifecycle (per-8×8 counter, updated by mode-decision;
  the denoiser is a read-only consumer for skin gating).
- Cyclic-refresh segmentation map writes (CR-AQ writes per-block segment IDs
  into `cm->seg_map`; the denoiser does **not** touch the segmentation map).
  L-67 is resolved at the HLD-threading level: the segmentation map is owned by
  the AQ-CR LLD and read by the denoiser only via the segment-aware filter
  gating that AQ-CR exposes via `ctx.sb_skip_denoising`. See §7-D5 / L-67.

Stage 1 source: `notes/vp9-analysis/05-rate-control.md` §2.3-§2.5, §3.2, §4.13,
§4.14, §4.15, §5.9, §5.10, §6, §7 (Q-RC-8..Q-RC-11, Q-RC-21, Q-RC-23).
Stage 2 HLD anchors: `notes/hld/02-simd-dispatch.md` §3.25, §5.5;
`notes/hld/01-threading.md` §5.2 row "denoiser",
`notes/hld/00-architecture.md` §3.6 (`denoiser: Option<Denoiser>`),
`notes/hld/04-svc.md` (per-layer denoising — only top 1 or 2 spatial layers).

---

## 2. libvpx origin

| Subject | File | Lines | Stage 1 ref |
|---|---|---|---|
| Denoiser struct | `vp9/encoder/vp9_denoiser.h` | 22-118 | A5 §2.3 |
| Pickmode context | `vp9/encoder/vp9_denoiser.h` | 58-68 | A5 §2.4 |
| `VP9_DENOISER_DECISION` enum | `vp9/encoder/vp9_denoiser.h` | 45-49 | A5 §2.3 |
| `VP9_DENOISER_LEVEL` enum | `vp9/encoder/vp9_denoiser.h` | 51-56 | A5 §2.3 |
| Per-block decision | `vp9/encoder/vp9_denoiser.c` | 336-437 | A5 §3.2, §4.14 |
| Per-pixel filter (scalar) | `vp9/encoder/vp9_denoiser.c` | 68-180 | A5 §4.13 |
| Per-pixel filter (SSE2) | `vp9/encoder/x86/vp9_denoiser_sse2.c` | 263-340 | A5 §6 |
| Per-pixel filter (NEON) | `vp9/encoder/arm/neon/vp9_denoiser_neon.c` | 296-373 | A5 §6 |
| Motion-comp setup | `vp9/encoder/vp9_denoiser.c` | 187-334 | A5 §4.14 |
| Frame-info commit | `vp9/encoder/vp9_denoiser.c` | 464-524 | A5 §3.2 |
| Per-frame ref update | `vp9/encoder/vp9_denoiser.c` | 782-822 | A5 §3.2 |
| Per-mode stat collection | `vp9/encoder/vp9_denoiser.c` | 533-548 | A5 §3.2 |
| SVC ref allocation | `vp9/encoder/vp9_denoiser.c` | 553-720 | A5 §2.3 |
| Set noise level | `vp9/encoder/vp9_denoiser.c` | 732-743 | A5 §3.2 |
| Threshold formulae | `vp9/encoder/vp9_denoiser.c` | 28-63 | A5 §4.13, §5.9 |
| Noise-estimate struct | `vp9/encoder/vp9_noise_estimate.h` | 30-40 | A5 §2.5 |
| Noise-estimate update | `vp9/encoder/vp9_noise_estimate.c` | 109-302 | A5 §4.15 |
| Noise-estimate init/disable rules | `vp9/encoder/vp9_noise_estimate.c` | 38-74, 145-160 | A5 §2.5, §4.15 |
| Mode-decision call site (per-mode) | `vp9/encoder/vp9_pickmode.c` | 2438, 2438-2478 | A5 §3.2 |
| Mode-decision call site (per-block) | `vp9/encoder/vp9_pickmode.c` | 2676-2695 | A5 §3.2 |
| `vp9_denoiser_realloc_svc` | `vp9/encoder/vp9_denoiser.c` | 553-720 | A5 §2.3 |
| RTCD signature | `vp9/common/vp9_rtcd_defs.pl` | 122-126 | A5 §6 / HLD-simd §3.25 |
| Skin block detector (called from denoiser) | `vp9/encoder/vp9_skin_detection.c` | 47-118 | A5 §4.14 (called by `perform_motion_compensation`) |
| `consec_zero_mv` consumer site | `vp9/encoder/vp9_denoiser.c` | 211-221 (skin gate) | A5 §4.14 |
| Mode-decision feedback hook (out of scope here, cited for boundary) | `vp9/encoder/vp9_pickmode.c` | 1376-1450 | 00-overview §4.7 |

**Total libvpx LOC owned by this LLD:** `vp9_denoiser.{h,c}` (839 + 132) +
`vp9_noise_estimate.{h,c}` (302 + 54) + the denoiser-filter SSE2/NEON kernels
(roughly 800 SLOC of C + intrinsics) = **~2.1 KLOC of C** to port. Stage 1
estimate (00-overview §S0-4) was 1.3 KLOC of C, 3-5 KLOC of Rust. This LLD's
scope-aware estimate revises down: SVC-realloc is a small minority and HBD is
deleted. **Working budget for Stage 4c implementation: ~2-3 KLOC of Rust.**

---

## 3. Public API

The denoiser is internal to `rvp9-encoder`. It is **not** re-exported from the
crate root; consumers (mode-decision, frame-loop) call methods on
`Encoder::denoiser_mut()` (only when `cfg.denoiser_enabled() == true`) or
read-only accessors via `Encoder::denoiser()`.

Module path: `rvp9_encoder::denoiser`.

### 3.1 Encoder-side construction / lifecycle

All methods below are main-thread-only. Per-block worker calls go through
`TileDenoiserView` (§3.2).

```rust
impl Denoiser {
    /// Allocate for given frame geometry. `num_layers >= 1` (1 for
    /// non-SVC). `border_in_pixels` matches reconstructed-frame border
    /// (libvpx: `VP9_ENC_BORDER_IN_PIXELS = 96`).
    /// Mirrors `vp9_denoiser_alloc` (`vp9_denoiser.c:725-780`).
    /// Returns `Err(OutOfMemory)` on allocation failure; never panics.
    pub fn new(
        num_layers: usize, mi_cols: u32, mi_rows: u32,
        border_in_pixels: u32,
    ) -> Result<Self, DenoiserError>;

    /// `vp9_denoiser_free` (`vp9_denoiser.c:824-839`). Called on
    /// `Encoder::reconfigure` when frame size changes.
    pub fn free(&mut self);

    /// Set strength from noise-estimator output. Mirrors
    /// `vp9_denoiser_set_noise_level` (`vp9_denoiser.c:732-743`).
    /// On `kDenLowLow → higher` transition, sets `self.reset = true`
    /// (`vp9_denoiser.c:741-742`).
    pub fn set_noise_level(&mut self, level: DenoiserLevel);

    /// Pre-encode hook: handle KEY_FRAME / `reset` re-init from source.
    /// Mirrors the KF/reset branch at `vp9_denoiser.c:790-806`. Caller:
    /// main thread, after RC picks frame_type, before workers spawn.
    pub fn pre_encode_frame(
        &mut self, frame_type: FrameType,
        source: &FrameBuffer, spatial_layer: u8,
    );

    /// Post-encode hook: swap denoised current → designated ref slot.
    /// Mirrors `vp9_denoiser_update_ref_frame` (`vp9_denoiser.c:782-822`).
    /// Caller: main thread, after all workers complete, before
    /// `update_reference_frames`.
    pub fn post_encode_frame(
        &mut self, refresh_mask: RefreshMask,
        spatial_layer: u8, svc_first_layer_denoise: bool,
    );

    /// SVC layer-count change. Mirrors `vp9_denoiser_realloc_svc`
    /// (`vp9_denoiser.c:553-720`). SVC-only.
    pub fn realloc_svc(
        &mut self, new_num_layers: usize,
    ) -> Result<(), DenoiserError>;
}
```

### 3.2 Per-block worker API

The hot path. Called once per superblock from the tile worker after
`vp9_pick_inter_mode_sb` writes `best_pickmode` into `mi`. Operates on a
tile-disjoint slice of `running_avg_y[]` (HLD-threading §5.2).

```rust
impl<'frame> TileDenoiserView<'frame> {
    /// Per-mode SSE stat collection. Called from the per-mode inner loop
    /// in mode-decision (`vp9_pickmode.c:2438`, kernel
    /// `vp9_denoiser.c:533-548`). Populates `best_sse_*`, `zeromv_sse`,
    /// `zeromv_lastref_sse`, `best_zeromv_reference_frame` on `ctx`.
    /// NOT the encoder's chosen mode — auxiliary stats only.
    pub fn update_frame_stats(
        ctx: &mut PickModeCtx, mi: &ModeInfo,
        sse_y: u32, mode: PredictionMode,
    );

    /// Decide + apply per-block denoise. Mirrors `vp9_denoiser_denoise`
    /// (`vp9_denoiser.c:336-437`). See §5.3.
    ///
    /// `consec_zero_mv_min`: precomputed min over the SB's 8×8
    /// sub-blocks of `cpi->consec_zero_mv[]` (A5 §4.14 step 1).
    /// `is_skin_sb`: precomputed `vp9_compute_skin_block` result; `None`
    /// when skin gate is off.
    ///
    /// Caller (mode-decision LLD) uses `FilterZeromv` return to gate
    /// `recheck_zeromv_after_denoising`.
    ///
    /// **Safety:** internally aliases `xd->plane[0].pre[0]` /
    /// `dst.buf` via raw pointers during motion-comp setup
    /// (`vp9_denoiser.c:284-321`); restores before return. See
    /// HLD-threading §5.4 Template A.
    pub fn denoise(
        &mut self, mi: &ModeInfo, mb_xd: &mut MacroBlockD,
        mi_row: u32, mi_col: u32, bsize: BlockSize,
        ctx: &mut PickModeCtx,
        use_gf_temporal_ref: bool,
        consec_zero_mv_min: u32,
        is_skin_sb: Option<bool>,
        spatial_layer: u8,
    ) -> DenoiserDecision;

    /// Snapshot best mode/MV/ref/filter into `PickModeCtxDen` for the
    /// `recheck_zeromv_after_denoising` hook. **Owned by mode-decision
    /// LLD**; signature stub for the boundary. Mirrors
    /// `vp9_pickmode_ctx_den_update` (`vp9_pickmode.c:1352-1376`).
    pub fn pickmode_ctx_den_update(
        ctx_den: &mut PickModeCtxDen,
        best_pickmode: &BestPickMode,
        rdc: &RdCost, zero_last_cost_orig: i64,
    );
}
```

### 3.3 Noise-estimator sub-API

Main-thread-only. Workers do not touch the noise estimator
(HLD-threading §5.2, strategy F).

```rust
impl NoiseEstimate {
    /// `vp9_noise_estimate_init` (`vp9_noise_estimate.c:25-50`). Sets
    /// `thresh` from resolution (90/115/140/200; A5 §2.5).
    pub fn init(width: u32, height: u32, denoiser_enabled: bool) -> Self;

    /// Per-frame update. Mirrors `vp9_update_noise_estimate`
    /// (`vp9_noise_estimate.c:109-302`). Called early in the frame
    /// loop before mode-decision.
    /// Returns the (cached) level; if window completed, caller passes
    /// to `Denoiser::set_noise_level`.
    pub fn update(
        &mut self,
        source: &FrameBuffer, last_source: &FrameBuffer,
        consec_zero_mv: &[u8],
        frame_counter: u32, avg_frame_low_motion: u32,
        screen_content: bool,
    ) -> NoiseLevel;

    /// `vp9_noise_estimate_extract_level` (`:94-107`).
    pub fn extract_level(&self) -> NoiseLevel;
}
```

### 3.4 Error type

```rust
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DenoiserError {
    OutOfMemory,
    /// SVC realloc requested with non-SVC denoiser. Library bug if
    /// reached — callers should check `cfg.svc_enabled()` before
    /// calling `realloc_svc`.
    NotSvc,
}
```

No panics on hot path. Allocation failure on `new` / `realloc_svc` returns
`DenoiserError::OutOfMemory`. All hot-path calls (`denoise`,
`update_frame_stats`) are infallible at the API surface; internal
`debug_assert!`s guard SB geometry preconditions.

---

## 4. Types

This section names types concretely. Field-by-field comments follow libvpx
semantics with explicit citations.

### 4.1 `Denoiser`

Origin: `VP9_DENOISER` struct, `vp9_denoiser.h:45-56`. Lives on `Encoder` as
`denoiser: Option<Denoiser>` (HLD-arch §3.6); `None` when the `denoiser`
Cargo feature is off or `cfg.noise_sensitivity == 0`.

```rust
pub struct Denoiser {
    /// Running denoised average per ref-slot per spatial layer. Flat
    /// storage indexed by `layer * num_ref_frames + ref_idx`; accessor
    /// is 2D `[layer][ref]` (§7-D1). Slot 0 (`INTRA_FRAME`-indexed)
    /// is the *current* denoised output buffer (§7-D2).
    /// Origin: `vp9_denoiser.c:344-353`.
    running_avg_y: Box<[FrameBuffer]>,
    /// Motion-compensated denoised reference, one per spatial layer.
    /// Recomputed each block. Origin: `vp9_denoiser.h:48`.
    mc_running_avg_y: Box<[FrameBuffer]>,
    /// Used **only** by the noise estimator (previous source frame's
    /// luma plane). Origin: `vp9_denoiser.h:50`.
    last_source: FrameBuffer,
    /// True once `running_avg_y[]` has been initialised.
    frame_buffer_initialized: bool,
    /// `set_noise_level` sets this on kDenLowLow → higher transition
    /// (`vp9_denoiser.c:741-742`); next `pre_encode_frame` re-inits.
    reset: bool,
    /// `NONSVC_REF_FRAMES = 3` (non-SVC) or `SVC_REF_FRAMES = 9` (SVC).
    /// Origin: `vp9_denoiser.h:26-30, 53`.
    num_ref_frames: u8,
    /// Number of spatial layers. Origin: `vp9_denoiser.h:54`.
    num_layers: u8,
    /// Bumped in `post_encode_frame`. Origin: `vp9_denoiser.h:55`.
    current_denoiser_frame: u32,
    /// Current strength. Origin: `vp9_denoiser.h:56`.
    denoising_level: DenoiserLevel,
    /// Previous strength; compared by `set_noise_level` to detect the
    /// kDenLowLow → higher transition. Origin: `vp9_denoiser.h:57`.
    prev_denoising_level: DenoiserLevel,
}

unsafe impl Send for Denoiser {}
// Deliberately NOT Sync. Two access disciplines:
//   1. Whole-struct &mut from the main thread (pre/post-encode).
//   2. Per-tile mutable slice borrowed by exactly one worker.
// Both hand out a `TileDenoiserView<'frame>` rather than sharing
// `&Denoiser` directly. HLD-threading §5.2, row "denoiser".
```

### 4.2 `TileDenoiserView<'frame>`

One tile worker's mutable view. Constructed by `Denoiser::split_for_tiles`
(HLD-threading §5.2). Lifetime `'frame` bounds the worker's lease; view
drops at frame end and `Denoiser` becomes uniquely borrowable again.

**Aliasing contract** (HLD-threading §5.4 Template A): each tile worker
owns a non-overlapping column slice of `running_avg_y[]` and
`mc_running_avg_y[]`. `last_source` is read-only from this view.

```rust
pub struct TileDenoiserView<'frame> {
    /// Tile-column slices of each `running_avg_y[layer][ref]` slot.
    /// Writes go only to the worker's `spatial_layer` slot.
    running_avg_y: TilePlaneMatrix<'frame>,
    /// Tile-column slice of `mc_running_avg_y[spatial_layer]`.
    mc_running_avg_y: TilePlaneRow<'frame>,
    /// Frame-immutable denoiser state.
    state: &'frame DenoiserFrameState,
    /// Kernel dispatch (HLD-simd §3.25).
    kernel: DenoiserFilterFn,
    /// Worker's tile geometry, for safety asserts.
    tile: TileGeom,
}
```

`TilePlaneMatrix` / `TilePlaneRow` are the standard tile-disjoint
primitives (HLD-threading §5.2): raw-pointer + stride + safe
`as_slice_mut(row)` accessor.

### 4.3 `DenoiserFrameState`

Per-frame immutable state that workers read:

```rust
/// Captured snapshot of `Denoiser` fields needed by the kernel and
/// per-block decision. Set on the main thread in `pre_encode_frame`,
/// borrowed `&` by all tile workers during the frame.
pub struct DenoiserFrameState {
    pub denoising_level: DenoiserLevel,
    pub reset: bool,
    pub current_denoiser_frame: u32,
    /// `1` if the current frame may use LAST as a denoised reference
    /// (i.e., last-frame was a reference and is denoised). Computed
    /// once per frame by the main thread.
    pub last_is_reference: bool,
    /// `use_gf_temporal_ref` from `cpi->oxcf` / SVC state, captured
    /// here so workers don't reach into `Svc`. Origin:
    /// `vp9_denoiser.c:336`, `gf_temporal_ref` arg.
    pub use_gf_temporal_ref: bool,
}
```

### 4.4 Enums and constants

```rust
/// Denoiser strength. Origin: `vp9_denoiser_level`,
/// `vp9_denoiser.h:51-56`.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum DenoiserLevel {
    LowLow = 0,
    Low = 1,
    Medium = 2,
    High = 3,
}

/// Per-block denoiser decision. Origin:
/// `VP9_DENOISER_DECISION`, `vp9_denoiser.h:45-49`.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(i32)]
pub enum DenoiserDecision {
    Copy = 0,                // COPY_BLOCK
    Filter = 1,              // FILTER_BLOCK
    FilterZeromv = 2,        // FILTER_ZEROMV_BLOCK
}

/// Noise classification from the estimator. Origin: `NOISE_LEVEL`,
/// `vp9_noise_estimate.h:30`.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum NoiseLevel {
    LowLow = 0,
    Low = 1,
    Medium = 2,
    High = 3,
}

/// `MAX_REF_FRAMES = 4` (LAST, GOLDEN, ALTREF, INTRA in some contexts).
/// Origin: `vp9_blockd.h:MAX_REF_FRAMES`.
pub const MAX_REF_FRAMES: usize = 4;
/// Origin: `vp9_denoiser.h:26`.
pub const NONSVC_REF_FRAMES: usize = MAX_REF_FRAMES - 1;
/// Origin: `vp9_denoiser.h:30`.
pub const SVC_REF_FRAMES: usize = 9;
/// Origin: `vp9_denoiser.h:22` — `(8 * 3)`.
pub const MOTION_MAGNITUDE_THRESHOLD: i32 = 24;
/// Origin: `vp9_denoiser.c:NOISE_MOTION_THRESHOLD`.
pub const NOISE_MOTION_THRESHOLD: i32 = 625;
/// Origin: `vp9_denoiser.c` `adj_val[3]`.
pub const ADJ_VAL_BASE: [i32; 3] = [3, 4, 6];
```

Threshold helpers (Rust ports of `vp9_denoiser.c:28-63`, A5 §5.9). Implemented
as `const fn` so the compiler folds them when `increase_denoising` is known
constant at the call site:

```rust
/// Origin: `vp9_denoiser.c:28-34`.
#[inline]
const fn absdiff_thresh(_bs: BlockSize, increase: bool) -> i32 {
    3 + if increase { 1 } else { 0 }
}

/// Origin: `vp9_denoiser.c:36-38`.
#[inline]
const fn delta_thresh(_bs: BlockSize, _increase: bool) -> i32 { 4 }

/// Origin: `vp9_denoiser.c:40-42`.
#[inline]
const fn sse_thresh(bs: BlockSize, increase: bool) -> u32 {
    bs.num_pixels() * if increase { 80 } else { 40 }
}

/// Origin: `vp9_denoiser.c:44-50`.
#[inline]
const fn sse_diff_thresh(bs: BlockSize, increase: bool, motion_mag: i32) -> i32 {
    if motion_mag > NOISE_MOTION_THRESHOLD {
        if increase { (bs.num_pixels() as i32) << 2 } else { 0 }
    } else {
        (bs.num_pixels() as i32) << 4
    }
}

/// Origin: `vp9_denoiser.c:52-58` — same formula for "weak" and "strong".
#[inline]
const fn total_adj_thresh(bs: BlockSize, increase: bool) -> i32 {
    (bs.num_pixels() as i32) * if increase { 3 } else { 2 }
}
```

### 4.5 `NoiseEstimate`

Origin: `NOISE_ESTIMATE`, `vp9_noise_estimate.h:30-40`. Lives at
`Encoder.noise_estimate` per HLD-threading §5.2 (strategy F — main thread
only; workers do not touch it).

```rust
pub struct NoiseEstimate {
    /// Set by resolution gate (`vp9_noise_estimate.c:52-74`).
    enabled: bool,
    /// Current classification.
    level: NoiseLevel,
    /// Smoothed `(3*old + max_bin*40) >> 2`. Origin: `:280`.
    value: i32,
    /// Resolution-derived: 90 (<640×360), 115 (≥640×360),
    /// 140 (≥1280×720), 200 (≥1920×1080). Origin: `:38-47`.
    thresh: i32,
    /// `(3*thresh) >> 1`. Mid-window force-reclassify trigger
    /// (`:287-293`).
    adapt_thresh: i32,
    /// Window accumulator.
    count: u32,
    /// Cached geometry, for resize detection.
    last_w: u32,
    last_h: u32,
    /// 15 first window, 30 thereafter (`:213-217`).
    num_frames_estimate: u32,
}
```

### 4.6 `PickModeCtx` (denoiser-relevant fields only)

The full `PickModeCtx` is owned by the mode-decision LLD. Denoiser-relevant
fields per A5 §2.4 / `vp9_denoiser.h:58-68`:

```rust
// In rvp9_encoder::mode_decision::ctx:
pub struct PickModeCtx {
    // ... (mode-decision fields omitted)

    // --- denoiser stats, populated by update_frame_stats() ---
    pub newmv_sse: u32,              // best non-zero-MV SSE
    pub zeromv_sse: u32,             // best ZEROMV SSE
    pub zeromv_lastref_sse: u32,     // ZEROMV SSE for LAST specifically
    pub best_sse_inter_mode: PredictionMode,
    pub best_sse_mv: MotionVector,
    pub best_reference_frame: MvReference,
    pub best_zeromv_reference_frame: MvReference,
    /// Upstream-set per `vp9_pickmode.c:1721-1747`. If true, denoiser
    /// returns Copy unconditionally.
    pub sb_skip_denoising: bool,
}

/// `VP9_PICKMODE_CTX_DEN`, `vp9_denoiser.h:58-68`. Owned by
/// mode-decision LLD; reproduced here for boundary.
pub struct PickModeCtxDen {
    pub zero_last_cost_orig: i64,
    pub ref_frame_cost: [i32; MAX_REF_FRAMES],
    pub frame_mv: [[MotionVector; MAX_REF_FRAMES]; MB_MODE_COUNT],
    pub reuse_inter_pred: bool,
    pub best_tx_size: TxSize,
    pub best_mode: PredictionMode,
    pub best_ref_frame: MvReference,
    pub best_pred_filter: InterpFilter,
    pub best_mode_skip_txfm: u8,
}
```

### 4.7 RTCD function-pointer type

Mirrors `vp9_denoiser_filter` RTCD signature, restated from HLD-simd §3.25:

```rust
/// `vp9_common/vp9_rtcd_defs.pl:122-126`. The only RTCD entry owned by
/// this LLD.
///
/// Returns `0` for COPY_BLOCK, `1` for FILTER_BLOCK. **No
/// `FILTER_ZEROMV_BLOCK` return from the kernel**; the per-block
/// decision layer (`denoise()`) attaches that based on
/// `zeromv_filter`.
pub type DenoiserFilterFn = unsafe extern "C" fn(
    sig: *const u8, sig_stride: i32,
    mc_avg: *const u8, mc_avg_stride: i32,
    avg: *mut u8, avg_stride: i32,
    increase_denoising: i32,
    bs: i32,
    motion_magnitude: i32,
) -> i32;
```

Lives in the global `SimdDispatch::denoiser_filter` slot (HLD-simd §4.1 line
1351). Selected once at encoder init by `SimdDispatch::detect()`.

---

## 5. Internal algorithm

This section gives the pseudocode at a level a Rust implementer can write
against, with file:line citations for every non-trivial step.

### 5.1 Per-frame lifecycle (main thread)

```
fn frame_loop(enc: &mut Encoder, src: &FrameBuffer):
    // Step 1: noise estimator (A5 §3.2 "decoupled" / §4.15)
    if enc.noise_estimate.enabled && enc.cfg.noise_sensitivity > 0:
        // `last_source` is `enc.denoiser.last_source` from the prior
        // frame's "show_frame" point.
        let new_level = enc.noise_estimate.update(
            src,
            &enc.denoiser.last_source,
            &enc.consec_zero_mv,
            enc.current_video_frame,
            enc.rc.avg_frame_low_motion,
            enc.sf.screen_content,
        );
        if enc.noise_estimate.count == enc.noise_estimate.num_frames_estimate:
            // Hit a classification window boundary.
            enc.denoiser.set_noise_level(new_level.into());
            enc.noise_estimate.count = 0;
            enc.noise_estimate.num_frames_estimate = 30;  // ← only 15 first time

    // Step 2: pre-encode hook (KEY_FRAME / reset re-init)
    enc.denoiser.pre_encode_frame(
        enc.cm.frame_type,
        src,
        enc.svc.current_spatial_layer,
    );

    // Step 3: tile-worker dispatch (HLD-threading §4). Each worker
    // gets a TileDenoiserView (§3.2 above).

    // Step 4: post-encode hook (swap denoised current → ref slot)
    enc.denoiser.post_encode_frame(
        enc.cm.refresh_mask(),
        enc.svc.current_spatial_layer,
        enc.svc.first_layer_denoise,
    );

    // Step 5: cache the source for the next frame's noise estimator
    enc.denoiser.last_source.copy_from(src);
```

Citations: A1 §3.4 step 43 (denoiser per-block) and step 52 (post-encode update),
A5 §3.2 step 3, `vp9_encoder.c:vp9_get_compressed_data` call order around
`vp9_denoiser_update_ref_frame`.

### 5.2 `pre_encode_frame` (KEY_FRAME / reset re-init)

```
fn pre_encode_frame(self, frame_type, source, spatial_layer):
    if frame_type == KEY_FRAME or self.reset:
        for ref_idx in 0..self.num_ref_frames:
            // copy_y_yv12 of source into running_avg_y[layer*N + ref_idx]
            self.running_avg_y[spatial_layer][ref_idx].copy_y_from(source)
        self.reset = false
        self.frame_buffer_initialized = true
```

Origin: `vp9_denoiser.c:790-806`. Note: libvpx does the init inside
`vp9_denoiser_update_frame_info` post-encode rather than pre-encode; the
practical effect is the same because that function is called immediately
before the next frame's encode starts. We do it pre-encode in our port so the
ordering with `noise_estimate.update` is unambiguous (which depends on
`last_source` only, not `running_avg_y`).

**Divergence note:** §7-D3 records this ordering choice; byte-parity impact
is **none** (the read+write set of pre-vs-post is disjoint between frames).

### 5.3 `denoise()` — per-block top-level decision

The hot path. Mirrors `vp9_denoiser_denoise` (`vp9_denoiser.c:336-437`) plus
`perform_motion_compensation` (`:187-334`). Steps:

1. **Compute motion magnitude:** `motion_mag = mv.row² + mv.col²`
   (`vp9_denoiser.c:342`).
2. **Skin override** (`vp9_denoiser.c:357-378`): caller passes precomputed
   `is_skin_sb` (only meaningful when `use_skin_detection && bsize <=
   BLOCK_32X32 && level < kDenHigh`). If `!is_skin` and `level == kDenHigh`,
   set `increase_denoising = true`.
3. **Early COPY** (`vp9_denoiser.c:380-388`): if `!last_is_reference ||
   level < kDenLow || ctx.sb_skip_denoising`, commit COPY and return.
4. **Motion-comp setup** (§5.4 below; `vp9_denoiser.c:187-334`). Populates
   `mc_running_avg_y` from `running_avg_y[chosen_ref]` via the inter
   predictor. May return `COPY_BLOCK` directly on hard-cap rejections.
5. **Kernel dispatch** (`vp9_denoiser.c:407-422`): if step 4 returned
   `FILTER_BLOCK`, call `(view.kernel)(sig, mc_avg, avg, increase_denoising,
   bsize, motion_mag)`. Map kernel return code `{0, 1}` → decision
   `{Copy, Filter}`. The `unsafe` call site uses the HLD-threading §5.4
   Template A safety comment (pointers from `TileDenoiserView` prove
   tile-disjointness).
6. **Commit** (`vp9_denoiser.c:425-432`): on `Filter`, `convolve_copy(avg →
   src)` so the denoised pixels become the source for bitstream encoding
   of this block. On `Copy`, `convolve_copy(src → avg)` so `running_avg_y`
   carries the un-denoised src forward.
7. **`FilterZeromv` elevation:** if step 5 returned `Filter` and step 4 set
   `zeromv_filter = true` (motion-comp chose the ZEROMV reference per
   `vp9_denoiser.c:329-333`), the public decision is `FilterZeromv` (which
   triggers `recheck_zeromv_after_denoising` in the mode-decision LLD,
   §5.8 boundary).

### 5.4 `perform_motion_compensation` — motion-comp setup + ref selection

Mirrors `vp9_denoiser.c:187-334`. Steps with file:line:

1. **Skin rejection** (`:202-216`): `is_skin && (motion_mag > 0 ||
   consec_zero_mv_min < 4)` → `Copy`.
2. **Small-block rejection** (`:218-227`): `bsize ∈ {8X8, 8X16, 16X8}` →
   `Copy`. `bsize == 16X16 && width > 480 && level <= kDenLow` → `Copy`.
3. **Ref selection** (`:237-272`): bias toward LAST_FRAME except when
   newmv is materially better. Let `sse_diff = max(0, zeromv_sse -
   newmv_sse)`. If `best_reference_frame ∈ {LAST, GOLDEN_if_!gf_temporal}`
   AND `sse_diff > sse_diff_thresh(bsize, increase_denoising,
   motion_mag)`, use newmv (`chosen_ref = best_reference_frame`, `chosen_mv
   = best_sse_mv`, `zeromv_filter = false`). Else fall back to ZEROMV:
   `chosen_ref = best_zeromv_reference_frame` if it's LAST or
   GOLDEN_if_!gf_temporal else `Copy`; `chosen_mv = 0`; `zeromv_filter =
   true`.
4. **Hard-cap rejections** (`:273-282`): `newmv_sse > sse_thresh(bsize,
   increase_denoising)` → `Copy`. `motion_mag > NOISE_MOTION_THRESHOLD <<
   3 = 5000` → `Copy`.
5. **Build motion-compensated predictor** (`:284-321`): shadow
   `xd->plane[0].pre[0].buf` to point at
   `running_avg_y[chosen_ref].y_buffer`, run
   `vp9_build_inter_predictors_sby(xd, mi_row, mi_col, bsize)` to write
   `mc_running_avg_y`, restore `xd` pointers (`:317-321`). The shadow-and-
   restore is the dangerous bit — the LLD wraps it in
   `with_shadowed_predictor(&mut xd, |xd_shadow| ...)` so the restore is
   automatic on `Drop`.

`sse_diff_thresh`, `sse_thresh`, threshold constants verbatim in §4.4.

### 5.5 `vp9_denoiser_filter_c` — scalar kernel

The canonical implementation per L-22 / R-03 (see §7-D5). Mirrors
`vp9_denoiser.c:68-180`. The algorithm is reproduced verbatim in A5 §4.13;
this LLD does not re-paste it. Key implementation points the Rust port must
respect:

1. `adj_val = [3, 4, 6]` per-pixel adjustment buckets, with a `+1` (or `+2`
   when `increase_denoising`) shift when `motion_magnitude <=
   MOTION_MAGNITUDE_THRESHOLD = 24` (`vp9_denoiser.c:91-95`).
2. **First pass** (`vp9_denoiser.c:97-138`): for each pixel, `diff =
   mc_avg - sig`. If `|diff| <= absdiff_thresh` write `mc_avg`, else write
   `sig +/- adj_val[bucket]` clipped to `[0, 255]`. Accumulate `total_adj`
   as a single `i32`.
3. **Early accept** (`vp9_denoiser.c:140-141`): if
   `|total_adj| <= total_adj_thresh` return `FILTER_BLOCK`.
4. **Second pass with damping** (`vp9_denoiser.c:143-176`): `delta =
   (|total_adj| - thresh) / num_pixels + 1`. If `delta >= 4` return
   `COPY_BLOCK`; else clip each pixel's contribution by `delta` and
   re-accumulate.
5. **Final accept/reject** (`vp9_denoiser.c:177-179`): if final
   `|total_adj| <= total_adj_thresh` return `FILTER_BLOCK`, else
   `COPY_BLOCK`.

`total_adj` width: `i32`. `num_pixels` peaks at 4096 (64×64); `|diff|`
peaks at 255; `|total_adj|` peaks at ~1.04M, well within `i32`. Match
libvpx widths exactly (D7).

**`total_adj` divergence (R-03 / L-22):** libvpx's scalar walks row-major
and accumulates a single `i32`. SSE2 reduces 16-pixel vectors with
`_mm_sad_epu8`-style horizontal adds and accumulates partial sums per row.
With `increase_denoising = 1` the larger `adj_val` magnitudes can push the
two `total_adj`s across the `delta_thresh` boundary in opposite directions
on rare inputs. Our scalar matches libvpx-C verbatim; our SSE2 matches
libvpx-SSE2 verbatim. See §7-D5.

### 5.6 Post-encode ref-frame swap

`vp9_denoiser_update_ref_frame` (`vp9_denoiser.c:782-822`) does three things:

1. KEY_FRAME or `reset`: copy source into all `running_avg_y[]` slots (we
   pushed this to `pre_encode_frame` per §5.2 / §7-D3).
2. Decide which ref slot to refresh based on `refresh_*` flags from RC /
   frame_loop.
3. Swap or copy the current denoised buffer (`running_avg_y[INTRA_FRAME +
   shift]`) into that ref slot.

The swap-or-copy logic is in `vp9_denoiser_update_frame_info`
(`vp9_denoiser.c:464-524`). For one ref refresh: swap pointers (zero-copy).
For multiple ref refreshes: copy for all but one, then swap the last.

Rust port:

```
fn post_encode_frame(self, refresh_mask, spatial_layer, svc_first_layer_denoise):
    self.current_denoiser_frame += 1;

    if !is_top_or_top_minus_one_spatial_layer(spatial_layer,
                                              svc_first_layer_denoise):
        return;   // Only top 1 or 2 spatial layers denoised
                  // (vp9_denoiser.c:625-631; A5 §3.4)

    let to_refresh: SmallVec<[RefSlot; 3]> = refresh_mask
        .iter_ones()
        .map(RefSlot::from_index)
        .collect();

    let cur = &mut self.running_avg_y[spatial_layer][INTRA_FRAME];
    match to_refresh.len() {
        0 => {}
        1 => {
            std::mem::swap(cur,
                &mut self.running_avg_y[spatial_layer][to_refresh[0]]);
        }
        _ => {
            // Copy all but the last; swap the last.
            for r in &to_refresh[..to_refresh.len()-1] {
                self.running_avg_y[spatial_layer][*r].copy_y_from(cur);
            }
            let last = to_refresh[to_refresh.len()-1];
            std::mem::swap(cur, &mut self.running_avg_y[spatial_layer][last]);
        }
    }
```

Per Q-RC-8 / §7-D1, the `[layer][ref]` indexing replaces libvpx's
`INTRA_FRAME + shift` arithmetic.

### 5.7 Noise estimator (`NoiseEstimate::update`)

Mirrors `vp9_update_noise_estimate` (`vp9_noise_estimate.c:109-302`).
Algorithm reproduced verbatim in A5 §4.15; the LLD does not re-paste it.
Steps with file:line:

1. **Disable conditions** (`vp9_noise_estimate.c:52-74`; A5 §4.15 forced
   overrides). `screen_content || resize_state != Orig || width*height <
   640*360 || svc_is_enhancement_layer()` → set `enabled = false`, return
   `LowLow`.
2. **Forced-low-motion escape** (`:145-160`): `frame_counter > 60 &&
   avg_frame_low_motion < low_motion_thresh(width)` → force `level =
   LowLow`. This is an asymmetric escape hatch — disables the denoiser but
   only `num_frames_estimate` cycles of re-estimation can re-enable it
   (Q-RC-21).
3. **Frame-period gate** (`:228-231`): run only every `frame_period = 8`
   frames; else return cached level.
4. **Low-motion block count** (`:233-244`): count 8×8 blocks where
   `consec_zero_mv[i] > thresh_consec_zeromv`. Require `>= 3/8 *
   total_blocks` for `frame_low_motion`; else return cached level.
5. **Variance histogram** (`:246-268`): 20-bin histogram of variance
   between current and last source over non-skin low-motion 16×16 blocks
   sampled at stride 4 (1/16 of the frame). `bin_size = 100`,
   `MAX_VAR_HIST_BINS = 20`.
6. **Dark-scene roll-off** (`:247-256`): adjust histogram if mean luma
   is low (dark-content correction).
7. **3-bin smoothed argmax** (`:270-278`): find `max_bin`.
8. **EMA** (`:280`): `value = (3*old_value + max_bin * 40) >> 2`.
9. **Mid-window force-reclassify** (`:287-293`): if `value >
   adapt_thresh && level < kMedium`, set `count = num_frames_estimate`
   so step 10 fires immediately.
10. **Window completion** (`:295-301`): if `count >= num_frames_estimate`,
    classify via `extract_level()`, reset `count = 0`, set
    `num_frames_estimate = 30` (was 15 on first window).

Helper functions (verbatim A5 §5.10):

```
fn extract_level(&self) -> NoiseLevel:        // :94-107
    if value > 2*thresh:    High
    elif value > thresh:    Medium
    elif value > thresh/2:  Low
    else:                   LowLow

fn thresh_consec_zeromv(denoise_on, width) -> u8:
    if denoise_on && 640 < width <= 1920: 2 else 6

fn low_motion_thresh(width) -> u32:
    if width >= 1280: 40 else 60
```

### 5.8 Integration boundary with mode-decision

A1 §3.4 step 43 places denoiser per-block **inside** the worker, after
`vp9_pick_inter_mode_sb` finalises `best_pickmode`. The two crossing points:

1. **`update_frame_stats` is called from inside the mode-decision per-mode
   inner loop.** Mode-decision LLD owns this call site; this LLD provides
   the function. Boundary: the function takes `&mut PickModeCtx` —
   denoiser-specific fields only (see §4.6).

2. **`recheck_zeromv_after_denoising` is owned by mode-decision LLD.**
   When `denoise()` returns `FilterZeromv`, the mode-decision LLD's
   `recheck` re-runs the ZEROMV cost computation on the denoised buffer
   and may swap the chosen mode. The denoiser supplies the denoised
   pixels (now in `xd->plane[0].dst`) and the `PickModeCtxDen`
   snapshot. Q-RC-10 explicitly: **we mirror this exactly in v1; the
   temptation to "clean up" is what silently breaks BD-rate.**

The denoiser does NOT call back into mode-decision; mode-decision drives
the recheck. The boundary is therefore one-way (mode-decision → denoiser,
denoiser → mode-decision via return value, mode-decision optionally calls
recheck).

### 5.9 Skin detection coupling (Q-RC-11)

`is_skin` is computed inside `perform_motion_compensation`
(`vp9_denoiser.c:357-378`) only when:

- `cpi->use_skin_detection` (speed feature, set by A6)
- `bsize <= BLOCK_32X32`
- `denoising_level < kDenHigh`

The denoiser is then a **read-only consumer** of `vp9_compute_skin_block`. We
pass the precomputed result as `is_skin_sb: Option<bool>` (the option allows
the caller to elide the skin computation when the gating conditions are not
met). `cpi->consec_zero_mv[]` is read but **not written** by the denoiser; it
is owned by mode-decision (§1 above). See §7-D6 for the boundary commitment.

### 5.10 AQ-3 (cyclic refresh) interaction

CR-AQ uses `cm->seg_map` to flag refresh blocks. The denoiser does **not**
read or write `cm->seg_map`. The only AQ-3 coupling is via
`ctx.sb_skip_denoising`, which is set upstream (`vp9_pickmode.c:1721-1747`) on
SBs that the encoder has flagged as "do not denoise"; the denoiser respects
the flag and returns `COPY_BLOCK`. This means **L-67 (cyclic-refresh
segmentation map ownership) does not touch the denoiser** — the
segmentation map is owned by `rvp9_encoder::aq::cyclic_refresh`; the denoiser
reads only the derived `sb_skip_denoising` boolean. See §7-D4.

---

## 6. SIMD dispatch

### 6.1 RTCD entry

This LLD owns exactly one RTCD entry: `vp9_denoiser_filter`
(`vp9_rtcd_defs.pl:122-126`). Signature, ISAs, and dispatch-table slot are
specified in HLD-simd §3.25:

```rust
pub type DenoiserFilterFn = unsafe extern "C" fn(
    sig: *const u8, sig_stride: i32,
    mc_avg: *const u8, mc_avg_stride: i32,
    avg: *mut u8, avg_stride: i32,
    increase_denoising: i32,
    bs: i32,
    motion_magnitude: i32,
) -> i32;
```

ISAs: scalar (always), SSE2 (x86_64), NEON (aarch64). No SSE4.1 or AVX2 — libvpx
ships only `sse2` for this kernel (`vp9_rtcd_defs.pl:125 — qw/neon sse2/`).

Block-size dispatch within the kernel (mirrors libvpx; see HLD-simd §3.25
caveat) splits the SSE2/NEON paths between `vp9_denoiser_NxM_small` (4×4
through 8×8 plus 4×8, 8×4) and `vp9_denoiser_NxM_big` (16×16 and up).

### 6.2 Scalar fallback

Per HLD-simd §1 / §5, every kernel has a scalar reference. The scalar
implementation lives at `rvp9_encoder::denoiser::filter::scalar` and follows
§5.5 above. It is the **canonical implementation** for L-22 / R-03 purposes
(see §7-D5).

### 6.3 Test strategy

Per HLD-simd §3.6:

- **Bit-exact differential test** between rvp9-scalar and libvpx-C for every
  block size × every `(motion_magnitude, increase_denoising)` pair. Block
  sizes: 4×4, 4×8, 8×4, 8×8, 8×16, 16×8, 16×16, 16×32, 32×16, 32×32, 32×64,
  64×32, 64×64. `motion_magnitude ∈ {0, 24, 25, 100, 625, 626, 5000, 5001}`
  (covers each threshold edge). `increase_denoising ∈ {0, 1}`.

- **Bit-exact differential test** between rvp9-scalar and rvp9-SSE2 / rvp9-NEON
  for `increase_denoising = 0` only. SSE2/NEON match scalar exactly when
  `increase_denoising = 0`.

- **Epsilon-bounded test** between rvp9-scalar and rvp9-SSE2 / rvp9-NEON for
  `increase_denoising = 1`: every pixel ≤ 1 ULP (i.e., differ by at most ±1).
  Justification: §7-D5 — libvpx itself does not guarantee bit-parity here.

Pseudo-random inputs cover the threshold edges; deterministic seeds (PRNG
seeded by block-size index).

### 6.4 Performance regression CI

Per HLD-simd §7 (perf gate): the denoiser kernel has a bench gate at 1080p30
livestream content. Regression > 5% wall time on the kernel fails CI.

---

## 7. Deliberate divergences from libvpx

Each entry is in divergence-ledger format and will be promoted into
`notes/divergence-ledger.md` when this LLD is approved.

### 7-D1 — Denoiser `running_avg_y` indexing flattened to `[layer][ref]`

**Where:** `rvp9_encoder::denoiser::Denoiser::running_avg_y`. libvpx
counterpart: `vp9/encoder/vp9_denoiser.c:344-353`
(`INTRA_FRAME + shift` arithmetic).

**What libvpx does:** stores `running_avg_y[]` as a flat array of
`num_ref_frames * num_layers` `YV12_BUFFER_CONFIG`s and indexes via
`running_avg_y[INTRA_FRAME + shift]` where `shift = layer * num_ref_frames`.
This is a layered array disguised as a flat one.

**What rvp9 does:** uses a 2D `[layer][ref]` accessor over the same underlying
storage. The flat layout is preserved for SIMD-friendly stride math, but the
public method `Denoiser::ref_slot_mut(layer, ref) -> &mut FrameBuffer` makes
the layering explicit.

**Why:** the flat indexing is the source of Q-RC-8 ("layered array disguised
as a flat one"). It also creates a per-block aliasing question (whether two
slots overlap) that we resolve by making the layer dimension explicit at
compile time. No runtime overhead; the flattening is purely a public-API
choice.

**Byte-parity impact:** none. The pixel storage is identical; only the access
pattern differs.

**Compression-efficiency impact:** none.

**Future plan:** keep this in v2 unless the storage becomes an SIMD bottleneck.
Decision: L-22 (sub-issue).

### 7-D2 — Rename "slot 0" of `running_avg_y` to `current_denoised_frame`

**Where:** internal API only: `Denoiser::current_denoised_frame_mut(layer)` and
`Denoiser::ref_slot_mut(layer, ref)`. libvpx counterpart: the implicit dual
use of `running_avg_y[INTRA_FRAME + shift]` as both "current denoised output"
and (after `update_frame_info`) "previous-frame ref slot 0".

**What libvpx does:** writes the per-block denoised pixels to
`running_avg_y[INTRA_FRAME + shift]` during the frame, then swaps that slot
into a ref-frame slot post-encode. The same `YV12_BUFFER_CONFIG` therefore
plays two roles at different points in the frame loop. Q-RC-9 documents this
as "easy to get wrong in a port."

**What rvp9 does:** keeps the same storage but exposes two named accessors so
the dual role is visible at every call site. `current_denoised_frame_mut` is
the per-block write target during the frame; `ref_slot_mut(layer, ref)` is
the post-encode swap target. Internally they alias the same `FrameBuffer`
when `ref == 0`; the swap in `post_encode_frame` moves the buffer to a
non-zero ref slot.

**Why:** Q-RC-9 / A5 §7 lists this as a documented trap. Naming costs nothing
and prevents class of bugs.

**Byte-parity impact:** none.

**Compression-efficiency impact:** none.

**Future plan:** keep. Decision: L-22 (sub-issue).

### 7-D3 — Move `running_avg_y[]` re-init from post-encode to pre-encode

**Where:** `rvp9_encoder::denoiser::Denoiser::pre_encode_frame`. libvpx
counterpart: `vp9/encoder/vp9_denoiser.c:790-806`
(`vp9_denoiser_update_ref_frame` re-init branch).

**What libvpx does:** on KEY_FRAME or `reset == 1`, copies the source into all
`running_avg_y` slots inside `vp9_denoiser_update_ref_frame`, which is
called **after** the frame is encoded. The effect is that the next frame's
encode reads the just-initialised slots.

**What rvp9 does:** moves the re-init to `pre_encode_frame`, which runs
**before** the workers spawn. Read-write order across frames is unchanged;
the slot is read by next-frame workers either way.

**Why:** ordering between `noise_estimate.update` (reads `last_source`) and
`pre_encode_frame` (reads `source`) is unambiguous in our split. In libvpx
the timing is implicit; in Rust we want the per-frame hooks to be obvious
fn calls in a single function.

**Byte-parity impact:** none (read and write sets of the two hooks are
disjoint across the frame boundary).

**Compression-efficiency impact:** none.

**Future plan:** keep. Decision: this LLD.

### 7-D4 — Denoiser does not read or write the segmentation map (L-67 resolution)

**Where:** `rvp9_encoder::denoiser`. libvpx counterpart: `cm->seg_map` is
read/written by `vp9_aq_cyclicrefresh.c` and read by no part of
`vp9_denoiser.c`.

**What libvpx does:** AQ-CR (cyclic refresh) owns `cm->seg_map`. The denoiser
does not read it. There is a transitive coupling via the speed-feature
cascade and `sb_skip_denoising`, which AQ-CR populates upstream
(`vp9_pickmode.c:1721-1747`).

**What rvp9 does:** make the boundary explicit. `rvp9_encoder::aq::cyclic_refresh`
owns `seg_map` (HLD-threading §5.2 row "segmentation_map"). The denoiser
reads only the derived `PickModeCtx::sb_skip_denoising` boolean.

**Why:** L-67 (devil's advocate #14) raised the segmentation-map ownership
question. The right answer is "the denoiser is a read-only consumer of a
derived boolean, not of the map." Documenting it here closes L-67 for this
LLD; HLD-threading owns the map's tile-disjoint write discipline for AQ-CR.

**Byte-parity impact:** none.

**Compression-efficiency impact:** none.

**Future plan:** keep. Decision: L-67.

### 7-D5 — Denoiser SIMD bit-parity caveat (L-22 / R-03 — the load-bearing entry)

**Where:** `rvp9_encoder::denoiser::filter::{scalar,sse2,neon}`. libvpx
counterpart: `vp9/encoder/vp9_denoiser.c:65-67` TODO comment; the scalar
implementation at `:68-180`; the SSE2 implementation at
`vp9/encoder/x86/vp9_denoiser_sse2.c`; the NEON implementation at
`vp9/encoder/arm/neon/vp9_denoiser_neon.c`.

**What libvpx does:** ships three implementations of `vp9_denoiser_filter`.
The scalar version walks pixels row-major and accumulates `total_adj` as a
single running `i32`. The SSE2 version reduces 16-pixel vectors with
`_mm_sad_epu8`-style horizontal adds and accumulates a partial sum per row.
The NEON version is structurally similar to SSE2. **For
`increase_denoising = 0`** all three produce identical pixels and identical
return values: the per-pixel computation has no accumulation-order
sensitivity. **For `increase_denoising = 1`** the larger `adj_val`
magnitudes can push `total_adj` close to the threshold from different
directions in the scalar vs SIMD reductions; the `delta` derived from
`total_adj` in the second pass then differs by 0 or 1, which causes a
≤1-pixel divergence in some second-pass outputs and (rarely) a different
`FILTER_BLOCK` vs `COPY_BLOCK` decision. libvpx has a TODO at lines 65-67
acknowledging this is not bit-exact and **does not fix it**.

> /* If increase_denoising is enabled in the future, we might need to
>  * update the code for calculating 'total_adj' in case the C code is
>  * not bit-exact with corresponding sse2 code. */
> — `vp9/encoder/vp9_denoiser.c:65-67`

**What rvp9 does:**

1. **Designate the scalar Rust implementation as the canonical
   implementation.** It mirrors `vp9_denoiser_filter_c` line-for-line
   (§5.5). The C scalar reference is libvpx's "spec-of-record" — when in
   doubt, the scalar wins.

2. **The SSE2 Rust implementation mirrors `vp9_denoiser_filter_sse2`
   line-for-line.** Goal: matched-config byte-parity with libvpx-SSE2 at the
   pixel level. This means a user who runs both libvpx-with-SSE2 and
   rvp9-with-SSE2 on the same input gets identical bitstreams, even though
   that output may diverge from libvpx-C-without-SSE2 by ≤ 1 pixel per
   block. **Same constraint applies to NEON ↔ libvpx-NEON.**

3. **Differential testing:**
   - Scalar-vs-scalar (rvp9 vs libvpx): bit-exact, no epsilon, all
     `increase_denoising` values.
   - SSE2-vs-SSE2 (rvp9 vs libvpx): bit-exact, no epsilon, all
     `increase_denoising` values.
   - NEON-vs-NEON (rvp9 vs libvpx): bit-exact, no epsilon, all
     `increase_denoising` values.
   - Scalar-vs-SSE2 (within rvp9): bit-exact for `increase_denoising = 0`;
     ≤ 1 ULP per pixel for `increase_denoising = 1` (allowed epsilon).
     `FILTER_BLOCK` vs `COPY_BLOCK` mismatches are permitted with
     `increase_denoising = 1` and noted in test output for triage.
   - Scalar-vs-NEON (within rvp9): same as scalar-vs-SSE2.

4. **Production dispatch:** at runtime the dispatch table picks SSE2 if
   available, else NEON if available, else scalar. The choice is deterministic
   per machine. A "force-scalar" environment variable
   (`RVP9_DENOISER_FORCE_SCALAR=1`) is provided for byte-parity diagnostic
   runs against libvpx-without-SSE2.

5. **BD-rate impact:** in matched-config CI (rvp9-SSE2 vs libvpx-SSE2), zero.
   In mixed configs (rvp9-scalar vs libvpx-SSE2), the divergence ledger
   notes that we expect ≤ 0.01% BD-rate noise on
   `increase_denoising = 1` content (i.e., `denoising_level == kDenHigh`).

**Why:** L-22 + R-03 in `decisions-needed.md`; Q-RC-23 in A5;
HLD-simd §5.5 / divergence-ledger L-RTCD-04. The alternative — picking one
canonical impl and forcing all ISAs to match it bit-exactly — would require
either (a) a non-SIMD-friendly scalar-emulation in the SSE2 path (defeats
the purpose), or (b) a non-libvpx-compatible SIMD reduction in the scalar
path (introduces divergence with libvpx-C, our spec-of-record). Neither is
worth ≤ 0.01% BD-rate noise on a narrow subset of content.

**Byte-parity impact:** minor. Matched-config byte-parity is preserved at
every supported ISA. Mixed-config byte-parity (one side using SSE2, the
other not) admits ≤ 1-pixel per-block divergence on
`increase_denoising = 1` blocks only. PLAN.md D8 makes byte-parity a
diagnostic tool, not a release gate; this divergence does not gate release.

**Compression-efficiency impact:** ≤ 0.01% BD-rate noise on
`denoising_level == kDenHigh` content. Well within the 0.5% PLAN.md gate.

**Future plan:** revisit if libvpx fixes their TODO. As of v1.14.1 (the
pinned version), the TODO is open and we mirror as-is. Decision: L-22 / R-03;
shares L-RTCD-04 with HLD-simd §5.5.

### 7-D6 — Skin detection moved to mode-decision LLD

**Where:** `rvp9_encoder::mode_decision::skin` (mode-decision LLD).
libvpx counterpart: `vp9/encoder/vp9_skin_detection.c` is its own file in the
encoder directory.

**What libvpx does:** ships skin detection as a standalone file called by the
denoiser, the variance partitioner, and AQ-CR.

**What rvp9 does:** mode-decision LLD owns `vp9_compute_skin_block` and
`vp9_compute_skin_sb` (since variance partitioner and AQ-CR both call it
too). The denoiser calls `mode_decision::skin::compute_skin_block(...)` as a
pure function.

**Why:** keeping skin detection out of `rvp9_encoder::denoiser` reduces the
denoiser module's surface and avoids a circular dependency with
mode-decision (which already calls into denoiser via `update_frame_stats`).

**Byte-parity impact:** none.

**Compression-efficiency impact:** none.

**Future plan:** keep. Decision: this LLD.

---

## 8. Open questions

### 8-Q1 — `vp9_denoiser_realloc_svc` worst-case allocation

When SVC layer count changes mid-stream (typically because the host
reconfigures), `vp9_denoiser_realloc_svc` (`vp9_denoiser.c:553-720`) may
allocate up to `(SVC_REF_FRAMES - NONSVC_REF_FRAMES) * num_layers = 6 *
num_layers` new `FrameBuffer`s. For 3-layer SVC at 4K (purely hypothetical;
PLAN.md tops out at 1080p60), that's 18 × ~12 MB = ~220 MB of new
allocations in a single call. **Should `realloc_svc` block the main thread
or be hoisted to a background thread?**

Recommendation: keep it on the main thread for v1 (matches libvpx; SVC
layer-count changes are rare). Profile in Stage 4d (temporal SVC) and Stage
4e (spatial SVC); revisit if it becomes an actual issue.

**Owner:** SVC LLD writer to confirm at Stage 4d.

### 8-Q2 — `cpi->consec_zero_mv[]` ownership clarification

The denoiser reads `consec_zero_mv_min` (the min over an SB's 8×8 sub-blocks)
as an input to its skin-rejection logic (`vp9_denoiser.c:208-216`). The
counter itself is incremented per-encoded-block by mode-decision
(`vp9_pickmode.c:2666-2674` approximately). HLD-threading §5.2 lists
`consec_zero_mv` as a tile-disjoint write target by workers (per-SB).

**Question:** does the denoiser read **this frame's** `consec_zero_mv[]`
values, **last frame's** values, or some mix? `vp9_denoiser_denoise` runs
per-SB after `vp9_pick_inter_mode_sb` (`vp9_pickmode.c:2676`), but
`consec_zero_mv[]` is updated at *encode* time, not pick time. So the
denoiser is reading a **half-updated** counter (its own SB and earlier SBs
in raster order have last frame's value; later SBs have this frame's value
- but wait, no, no SB has this frame's value yet because the counter is
bumped during *encode*, after the denoiser).

**Tentative resolution:** the denoiser reads last frame's `consec_zero_mv[]`
exclusively, because the per-encoded-block bump happens after denoise. This
should be documented in mode-decision LLD; this LLD records the expectation.

**Owner:** mode-decision LLD writer to confirm at Stage 4a.

### 8-Q3 — Per-tile parallelism of denoiser SVC ref slots

HLD-threading §5.2 says `denoiser.running_avg_y[]` is split by tile column
into tile-disjoint slices. The flat indexing in libvpx
(`running_avg_y[INTRA_FRAME + shift]`) makes me want to confirm: are the
per-layer slots stored in *contiguous* memory (so a tile column slice cuts
across all layers' refs at once) or in *layer-major* memory (so each
layer's ref slots are contiguous)?

**Tentative resolution:** layer-major. Each `running_avg_y[layer][ref]` is
its own `FrameBuffer`, so tile-disjoint slicing happens *within* one
`FrameBuffer`'s plane. Concurrent writes by workers to different
`FrameBuffer`s (different layers, different refs) are trivially disjoint;
concurrent writes by workers to the same `FrameBuffer`'s plane (same layer,
same ref slot, different tile columns) are disjoint by tile geometry.

**Owner:** confirmed at LLD; not blocking.

### 8-Q4 — Noise-estimator's `screen_content` input is float-derived (R-04)

The noise estimator gates on `cpi->oxcf.content == VP9E_CONTENT_SCREEN`
(`vp9_noise_estimate.c:52-55`) which itself is part of the per-frame
speed-feature derivation. R-04 (float determinism) flags this as a libm-
dependent path. **Does the noise estimator need any float math itself?**

Scan of `vp9_noise_estimate.c`: no `log`, `pow`, or `exp` calls. All
arithmetic is integer (variance is `(sum_sq - sum*sum/n)`-style; the EMA
update is `(3*old + new) >> 2`). So the noise estimator is **not** a
float-determinism source. R-04 affects the rate-control loop, not the
denoiser/noise-estimate. **Resolved: no float work needed here.**

### 8-Q5 — SVC `first_layer_denoise` policy

A5 §3.4 ("only the top 1 or 2 spatial layers are denoised, controlled by
`svc->first_layer_denoise`"). How is `first_layer_denoise` set? Search of
libvpx finds `vp9_svc_layercontext.c:set_flags_and_fb_idx_for_temporal_mode3`
references it, but the setter is not obvious. **Owner:** SVC LLD writer
to document at Stage 4d. For this LLD: the field is exposed as
`SvcState::first_layer_denoise` and read by `Denoiser::post_encode_frame`.

---

## 9. Test strategy

### 9.1 Unit tests

Per-helper:

- `absdiff_thresh`, `delta_thresh`, `sse_thresh`, `sse_diff_thresh`,
  `total_adj_thresh`: each tested against the verbatim libvpx values
  (A5 §5.9) for every block size × `increase_denoising` pair.
- `DenoiserLevel::from_noise_level` (`vp9_denoiser.c:732-743`): table of
  4 inputs × 4 outputs.
- `NoiseEstimate::init` for each resolution band (90/115/140/200).
- `NoiseEstimate::extract_level` for the four `value > k*thresh` bands.

### 9.2 Property-based tests

- **`denoiser_filter_c` invariant:** for any
  `(sig, mc_avg, increase_denoising, bs, motion_magnitude)`, the output
  `avg[i] - sig[i]` has the same sign as `mc_avg[i] - sig[i]` (or is
  zero). Property: filter never moves a pixel **away** from the
  motion-compensated reference.
- **`total_adj` accounting:** `total_adj` equals
  `sum(avg[i] - sig[i])` over all pixels after pass 1. Property
  asserted at function exit.
- **`COPY_BLOCK` is a fixed point:** if the kernel returns
  `COPY_BLOCK`, then on a second call with the same inputs but `avg`
  pre-initialised to `sig`, `avg` is unchanged.
- **Decision monotonicity:** for fixed `(sig, mc_avg, bs,
  motion_magnitude)`, increasing `increase_denoising` from 0 to 1
  may change `FILTER_BLOCK` to `COPY_BLOCK` or vice versa, but the
  decision is deterministic.

### 9.3 Differential tests against libvpx

- **Scalar kernel:** every block size × every `(motion_magnitude,
  increase_denoising)` combo against `vp9_denoiser_filter_c` from
  the pinned libvpx 1.14.1 source. Bit-exact, no epsilon, all
  parameters.
- **SSE2 kernel:** same, against `vp9_denoiser_filter_sse2`. Bit-exact.
- **NEON kernel:** same, against `vp9_denoiser_filter_neon`. Bit-exact.

Per §7-D5, no epsilon is allowed for matched-ISA tests. Epsilon is allowed
only for cross-ISA tests **within rvp9** when `increase_denoising = 1`.

### 9.4 Per-block decision tests

- **Skin rejection:** `is_skin = true` plus `(motion_mag > 0 ||
  consec_zero_mv_min < 4)` → `COPY_BLOCK`.
- **Small-block rejection:** `bsize ∈ {BLOCK_8X8, BLOCK_8X16, BLOCK_16X8}`
  → `COPY_BLOCK`. `BLOCK_16X16` + `width > 480` + `level ≤ kDenLow` →
  `COPY_BLOCK`.
- **Ref selection:** for given `(best_ref, best_zeromv_ref,
  use_gf_temporal_ref, sse_diff)`, verify the dispatch matches the
  `vp9_denoiser.c:237-272` table.
- **Hard-cap rejections:** `newmv_sse > sse_thresh` → `COPY_BLOCK`;
  `motion_mag > 5000` → `COPY_BLOCK`.

### 9.5 Noise-estimator tests

- **Resolution gating:** verify `thresh` for the four resolution bands.
- **Histogram update:** seed with synthetic histograms, verify
  `value = (3*old + max_bin*40) >> 2`.
- **Disable conditions:** screen content, SVC enhancement layer, resize
  state, low resolution.
- **Forced-low-motion override:** `frame_counter > 60 +
  avg_frame_low_motion < 60` (or 40 for HD) → level forced to LowLow
  (Q-RC-21).
- **Re-classify mid-window:** `value > adapt_thresh + level < kMedium`
  bumps `count` to `num_frames_estimate` immediately.

### 9.6 Frame-loop integration tests

- **KEY_FRAME re-init:** on the first frame and every key frame, all
  `running_avg_y[]` slots equal the source.
- **`reset` re-init:** transition from `kDenLowLow` to higher level
  sets `reset = true`; next `pre_encode_frame` re-initialises slots.
- **`FILTER_ZEROMV_BLOCK` → `recheck_zeromv_after_denoising`:**
  cross-LLD test with mode-decision; an integration test fixture
  that wires a denoiser into a stub mode-decision and verifies the
  recheck is invoked exactly when the denoiser returns
  `FilterZeromv`.
- **Post-encode swap correctness:** for every `refresh_mask` pattern
  (single ref, multiple refs, no refs), verify that
  `running_avg_y[layer][ref]` after `post_encode_frame` equals the
  buffer pointed to by the chosen current-denoised slot before the
  swap.

### 9.7 Real-time budget regression

- Bench gate: 720p30 livestream + noise input. With denoiser on at
  `kDenMedium`, the kernel + per-block decision must add no more than
  1.5 ms/frame to the median latency (gross budget number; refined
  during Stage 4c implementation).

### 9.8 Soak / Miri

- Miri runs against all non-SIMD paths (scalar kernel, decision logic,
  noise estimator). The kernel's `&mut [u8]` slice writes are Miri-clean
  (no `unsafe`).
- `unsafe` blocks in `TileDenoiserView` (raw-pointer slice splits) are
  excluded from Miri but covered by the differential tests and the
  HLD-threading §5.4 audit (Stage 5).

---

**One-line summary:**
`denoiser.md` ≈ 1360 lines; key §7 entries: **7-D5 (L-22 / R-03 — denoiser
SIMD bit-parity caveat resolved by matching libvpx-C scalar and libvpx-SSE2
verbatim, with cross-ISA epsilon ≤ 1 ULP allowed only at
`increase_denoising=1`)**, 7-D4 (L-67 — denoiser does not touch the
segmentation map; AQ-CR owns it), plus 7-D1/D2/D3/D6 minor port-shape
divergences; §8 has no Stage 3 blockers (8-Q1 deferred to Stage 4d SVC,
8-Q2 deferred to mode-decision LLD).
