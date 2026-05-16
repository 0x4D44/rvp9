# LLD: svc — Scalable Video Coding (temporal + spatial) implementation

**Stage:** 2 (LLD)
**Owner:** Claude (dispatched by Arthur).
**Status:** Draft for Arthur review.
**Scope:** the implementation-level layer below `notes/hld/04-svc.md`. The
HLD already locks the architectural shape (temporal/spatial layering
modes, reference-slot tables, drop modes, EXT_RATECTRL surfaces); this
LLD adds **concrete Rust types**, **function signatures**, the
**field-by-field layer-context swap order** (L-58), and the
implementation-level pseudocode that line-by-line mirrors
`vp9_one_pass_svc_start_layer`, `vp9_rc_drop_frame`, and the SVC
RC fan-out (L-77).

**Locked decisions in scope:**

- **D1** real-time priority (cpu-used 6–8).
- **D3** full SVC (temporal + spatial) in v1.
- **D-01** RESOLVED → 8-bit only (no `u16` paths anywhere).
- **D-05** RESOLVED → AQ-3 (cyclic refresh) only.
- **D-13** RESOLVED → CBR + VBR-RT, **no altref**, `lag_in_frames = 0`.
- **R-06** RESOLVED → EXT_RATECTRL plug-in in v1.

**Open decisions resolved by this LLD:**

- **L-58** layer-context swap ordering (§4.2).
- **L-61** max-tile-cols sizing under top-resolution layer (§3.3).
- **L-68** `vp9_scene_detection_onepass` per-layer behaviour (§5.3).
- **L-69** SVC cross-layer frame-drop interactions (§6).
- **L-75** frame-drop chain semantics (§6.4).
- **L-77** `temporal_layering × per-layer RC` (§5.4).

`file:line` citations are into `third_party/libvpx/` at v1.14.1 commit
`12f3a2a` unless otherwise noted. Absolute repo paths (e.g.
`D:/language/rvp9-2/...`) are given where the LLD references
non-libvpx artefacts.

---

## 1. Purpose

The SVC module owns the **multi-layer encode pipeline** above the
single-layer encoder. For each input source frame it iterates spatial
layers `[first_spatial_layer..number_spatial_layers)`, maintains
per-(spatial, temporal) layer state in `LayerContext`, swaps that
state in and out of the active encoder god-struct via
`restore_layer_context` / `save_layer_context`, dispatches the
reference-slot pattern for the chosen `TemporalLayering` mode, drives
per-layer source down-scaling, gates inter-layer prediction, runs the
cross-layer rate-control fan-out, and decides per-layer / per-superframe
frame drops. It also hosts the per-layer hook into the EXT_RATECTRL
plug-in (R-06).

It does **not** own:
- The single-layer encode loop (`encode_one_layer` is implemented by
  the partition / mode-decision / bitstream LLDs; this LLD only calls it).
- The buffer-pool slot table or `Arc<FrameBuf>` reference counting
  (frame-buffers LLD; this LLD only writes `fb_idx_*` integers).
- The cyclic-refresh state machine internals (aq LLD); we own the
  *segmentation-map ownership and swap* between per-layer storage and
  the encoder-active cell (L-67).
- The full `RateControl` algorithm (rate-control LLD); we own the
  per-layer fan-out wiring and the per-layer Q assignment surface.

---

## 2. libvpx origin

Stage 1 primary specialist: **A6** (`notes/vp9-analysis/06-threading-svc.md`).
Stage 1 secondary specialist: **A5**
(`notes/vp9-analysis/05-rate-control.md`) for the per-layer RC math.

Module-level libvpx mapping:

| libvpx C source                                  | rvp9 Rust module                              |
|---                                               |---                                            |
| `vp9/encoder/vp9_svc_layercontext.h:80-360`      | `rvp9_encoder::svc::Svc` (§4.4)               |
| `vp9/encoder/vp9_svc_layercontext.h:40-90`       | `rvp9_encoder::svc::LayerContext` (§4.5)      |
| `vp9/encoder/vp9_svc_layercontext.h:22-28`       | `rvp9_encoder::svc::InterLayerPred` (§4.6)    |
| `vp9/encoder/vp9_svc_layercontext.c:30-150`      | `Svc::new` (§4.10)                            |
| `vp9/encoder/vp9_svc_layercontext.c:158-179`     | `LayerContext::alloc_cyclic_refresh` (§4.5)   |
| `vp9/encoder/vp9_svc_layercontext.c:308-350`     | `Svc::update_layer_framerate` (§5.7)          |
| `vp9/encoder/vp9_svc_layercontext.c:352-389`     | `Encoder::restore_layer_context` (§4.2 / §5.1)|
| `vp9/encoder/vp9_svc_layercontext.c:391-424`     | `Encoder::save_layer_context` (§4.2 / §5.1)   |
| `vp9/encoder/vp9_svc_layercontext.c:444-452`     | `Encoder::inc_frame_in_layer` (§5.5)          |
| `vp9/encoder/vp9_svc_layercontext.c:454-470`     | `svc::get_layer_resolution` (§3.2)            |
| `vp9/encoder/vp9_svc_layercontext.c:517-621`     | `Svc::set_flags_mode_0212` (§4.7)             |
| `vp9/encoder/vp9_svc_layercontext.c:626-684`     | `Svc::set_flags_mode_0101` (§4.7)             |
| `vp9/encoder/vp9_svc_layercontext.c:687-720`     | `Svc::set_flags_mode_no_layering` (§4.7)      |
| `vp9/encoder/vp9_svc_layercontext.c:722-738`     | `Svc::set_flags_bypass` (§4.7)                |
| `vp9/encoder/vp9_svc_layercontext.c:769-994`     | `Encoder::start_layer` (§5.2)                 |
| `vp9/encoder/vp9_svc_layercontext.c:1051-1074`   | `Svc::check_reset_layer_rc_flag` (§5.9)       |
| `vp9/encoder/vp9_svc_layercontext.c:1076-1147`   | `Svc::constrain_inter_layer_pred` (§5.2 #5)   |
| `vp9/encoder/vp9_ratectrl.c:263-294`             | `Svc::update_buffer_level_preencode` (§5.4)   |
| `vp9/encoder/vp9_ratectrl.c:298-313`             | `Svc::update_layer_buffer_level_postencode` (§5.4) |
| `vp9/encoder/vp9_ratectrl.c:513-564`             | `Svc::test_drop` (§6.2)                       |
| `vp9/encoder/vp9_ratectrl.c:614-664`             | `Svc::drop_frame` (§6.3)                      |
| `vp9/encoder/vp9_ratectrl.c:1998-2017`           | `Svc::postencode_update_drop_frame` (§6.5)    |
| `vp9/encoder/vp9_ratectrl.c:2279-2491`           | `Svc::get_svc_params` (§5.4)                  |
| `vp9/encoder/vp9_ratectrl.c:3034-3243`           | (rate-control LLD owns `scene_detection_onepass`; SVC plumbing in §5.3) |
| `vp9/encoder/vp9_ethread.c:84-90`                | `Svc::max_worker_count` (§3.3, L-61)          |

---

## 3. Public API

The encoder exposes SVC through the top-level `Encoder` (HLD-arch §3.6).
The SVC module itself exposes the data types and a few free functions;
the layer-iteration loop lives on `Encoder` because it must touch
RateControl, buffer-pool slot writes, bitstream-writer, denoiser, and
EXT_RATECTRL — i.e. cross-cutting concerns the SVC module cannot own.

### 3.1 Constants

```rust
// crates/rvp9-encoder/src/svc/mod.rs

/// Hard upper bound on spatial layers in v1 (HLD-svc §1.2).
pub const MAX_SPATIAL_LAYERS: usize = 3;
/// Hard upper bound on temporal layers in v1.
pub const MAX_TEMPORAL_LAYERS: usize = 3;
/// Total (sl, tl) cells stored.
pub const MAX_LAYERS: usize = MAX_SPATIAL_LAYERS * MAX_TEMPORAL_LAYERS; // = 9

/// VP9 reference-pool slot count (mirrors libvpx `REF_FRAMES = 8`,
/// `vp9/common/vp9_onyxc_int.h`).
pub const REF_FRAMES: usize = 8;

/// Long-term GF temporal-ref slot indices (`vp9_svc_layercontext.c:83-86`).
pub const GF_TEMPORAL_REF_SLOT_LO: u8 = 6;
pub const GF_TEMPORAL_REF_SLOT_HI: u8 = 7;
```

### 3.2 Configuration types (input)

The user-facing config is `EncoderConfig.svc: SvcConfig`. The fields
mirror HLD-svc §8.1; preserved here for self-containment:

```rust
// crates/rvp9-encoder/src/svc/config.rs

#[derive(Debug, Clone)]
pub struct SvcConfig {
    pub number_spatial_layers:  u8,                // 1..=3
    pub number_temporal_layers: u8,                // 1..=3
    pub temporal_layering:      TemporalLayering,  // §4.3
    pub inter_layer_pred:       InterLayerPred,    // §4.6
    pub frame_drop_mode:        FrameDropMode,     // §4.8
    pub use_gf_temporal_ref:    bool,
    /// (num, den) per spatial layer; den ∈ {1,2,4}. Index 0 = base layer.
    pub spatial_scaling:        [(u8, u8); MAX_SPATIAL_LAYERS],
    pub layer_target_kbps:      [u32; MAX_LAYERS],
    pub layer_min_q:            [u8;  MAX_LAYERS],
    pub layer_max_q:            [u8;  MAX_LAYERS],
    pub framedrop_thresh:       [u8;  MAX_SPATIAL_LAYERS], // 0..=100
    pub max_consec_drop:        u32,
    pub first_spatial_layer_to_encode: u8,
    pub speed_per_layer:        [u8;  MAX_SPATIAL_LAYERS], // cpu-used 5..=8
}

impl SvcConfig {
    /// Validate against PLAN.md §2 + HLD-svc §1.2 / §8.1.
    /// Panics-free; returns `EncodeError::InvalidSvcConfig`.
    pub fn validate(&self) -> Result<(), crate::EncodeError>;
}
```

### 3.3 Worker-pool sizing (L-61)

`Svc::max_worker_count` is a free function used by the threading
pool initialiser (HLD-01) to size the worker pool **once** for the
top-resolution layer. Mirrors `vp9_ethread.c:84-90`:

> Cite `vp9_ethread.c:80-90`:
> ```
> if (cpi->use_svc && !cpi->row_mt)
>     num_workers = min(max_threads, get_max_tile_cols(cpi));
> ```
> where `get_max_tile_cols` is computed against the top spatial layer's
> resolution (the layer at index `number_spatial_layers - 1`, which
> always has scaling `(1, 1)`).

```rust
/// Maximum tile-column count derivable from the top-spatial-layer
/// resolution. Used by HLD-01 to size the persistent worker pool ONCE
/// at `Encoder::new`. After this point, per-frame worker launches use
/// `min(max_threads, current_layer_tile_cols)` (smaller pool for lower
/// spatial layers — extra workers stay dormant).
///
/// libvpx parity: `get_max_tile_cols(cpi)` at the top spatial layer
/// (vp9_ethread.c:84-90, vp9/common/vp9_tile_common.c:39-49).
pub fn max_worker_count(cfg: &SvcConfig, top_width: u32, top_height: u32,
                        max_threads: u32) -> u32;
```

Rationale (L-61 RESOLUTION): sizing once at the top-resolution layer
costs `~200 B` per dormant worker (mutex + cond + heap object) at the
worst case `MAX_NUM_THREADS = 80`, total ~16 KB. The alternative —
re-spawn per layer — incurs `pthread_create` cost (~50 µs per worker
per spatial layer transition) which directly hits the real-time
budget on 1080p60 (16 ms total). **Decision: size once at the top
layer.** Concrete sizing rule (mirroring `MIN_TILE_WIDTH_B64 = 4` /
`MAX_TILE_WIDTH_B64 = 64`, `vp9/common/vp9_tile_common.c:15-16`):

```rust
fn get_max_tile_cols(width: u32) -> u32 {
    let sb_cols = (width + 63) / 64; // 64-px SB-aligned
    let min_log2 = if sb_cols >= 4 { ((sb_cols + 63) / 64).max(1).ilog2() } else { 0 };
    let mut max_log2 = 0;
    while (4 << max_log2) < sb_cols { max_log2 += 1; }
    let log2 = max_log2.min(6).max(min_log2);
    1u32 << log2
}
```

This matches libvpx's `vp9_log2_tile_cols`/`vp9_get_min_log2_tile_cols`
within the legal range `[MIN_TILE_WIDTH_B64, MAX_TILE_WIDTH_B64]`.

### 3.4 Encoder-facing methods (live on `Encoder`)

```rust
// crates/rvp9-encoder/src/encoder.rs

impl Encoder {
    /// Public entry point. Iterates spatial layers and assembles a
    /// superframe. Mirrors A6 §3.7 (vp9_encoder.c:6383 inner loop).
    pub fn encode(
        &mut self,
        frame: &SourceFrame,
        opts: &FrameOptions,
    ) -> Result<EncodedSuperframe, EncodeError>;

    // ---- Internal methods owned by this LLD (`pub(crate)`) ----

    /// One spatial-layer iteration of the superframe loop. Mirrors
    /// vp9_svc_layercontext.c:769-994 + caller flow.
    pub(crate) fn encode_one_layer(
        &mut self,
        frame: &SourceFrame,
        opts: &FrameOptions,
    ) -> Result<EncodedLayer, EncodeError>;

    /// vp9_one_pass_svc_start_layer (vp9_svc_layercontext.c:769-994). §5.2.
    pub(crate) fn start_layer(&mut self,
        opts: &FrameOptions) -> Result<(), EncodeError>;

    /// vp9_restore_layer_context (vp9_svc_layercontext.c:352-389). §5.1.
    pub(crate) fn restore_layer_context(&mut self);

    /// vp9_save_layer_context (vp9_svc_layercontext.c:391-424). §5.1.
    pub(crate) fn save_layer_context(&mut self);

    /// vp9_inc_frame_in_layer (vp9_svc_layercontext.c:444-452). §5.5.
    pub(crate) fn inc_frame_in_layer(&mut self);

    /// Per-spatial-layer source down-scale. Reads `self.svc.scaler`. §3.2.
    /// Returns a borrow of either the original `frame` (1:1) or the
    /// pre-allocated `scaled[sl]` buffer.
    pub(crate) fn scale_source_for_layer<'a>(
        &'a mut self,
        src: &'a SourceFrame,
        sl: u8,
    ) -> &'a Yv12Buffer;

    /// Per-layer Q assignment (post-internal-pick, post-EXT_RATECTRL). §5.6.
    /// Returns the final qindex used.
    pub(crate) fn pick_layer_qindex(&mut self,
        opts: &FrameOptions) -> Result<u8, EncodeError>;

    /// Per-layer pre-encode drop test (vp9_rc_drop_frame). §6.3.
    /// Returns `true` if this layer must drop; mutates layer drop bookkeeping.
    pub(crate) fn check_layer_drop(&mut self) -> bool;

    /// Per-layer post-encode buffer-level update. Mirrors
    /// update_buffer_level_postencode + update_layer_buffer_level_postencode
    /// (vp9_ratectrl.c:298-333). §5.4.
    pub(crate) fn postencode_update_layer(&mut self, encoded_bits: i64);
}
```

### 3.5 Free helpers in `rvp9_encoder::svc`

```rust
/// vp9_svc_layercontext.c:454-470, even-rounded.
pub fn get_layer_resolution(width: u32, height: u32, num: u16, den: u16)
    -> (u32, u32);

/// vp9/encoder/vp9_encoder.h:1495: LAYER_IDS_TO_IDX.
#[inline] pub fn layer_idx(sl: u8, tl: u8, ntl: u8) -> usize {
    (sl as usize) * (ntl as usize) + (tl as usize)
}

/// Layer index for the base temporal cell of `sl`. Mirrors libvpx's
/// `&svc.layer_context[sl * ntl]` access pattern (vp9_svc_layercontext.c:445).
#[inline] pub fn base_tl_layer_idx(sl: u8, ntl: u8) -> usize {
    (sl as usize) * (ntl as usize)
}
```

### Preconditions / postconditions / unsafety

- All public/`pub(crate)` methods in this LLD are **safe Rust**. No
  `unsafe` blocks. The cross-thread aliasing (HLD-01) is fully resolved
  before the SVC module ever touches state: layer iteration is
  strictly single-threaded between worker joins.
- `Encoder::encode` panics-free: every error is `Result`. The only
  panics permitted are array-bounds asserts on the
  `MAX_LAYERS`/`REF_FRAMES` constants — debug-only via `debug_assert!`.
- `restore_layer_context` / `save_layer_context` must be called in
  matching pairs around `encode_one_layer`. Misuse is a bug, not a
  graceful error; debug-mode tracks balance with an `assert!`.
- `inc_frame_in_layer` is **idempotent within a superframe**:
  calling it twice for the same (sl) corrupts `current_video_frame_in_layer`.
  We enforce single-call via a per-frame `incremented: bool` flag on
  the layer's `LayerContext`, asserted in debug.

---

## 4. Types

### 4.1 `Svc` — top-level SVC state

Mirrors libvpx `SVC` (`vp9_svc_layercontext.h:80-360`, A6 §2.8) less
the altref/2-pass fields (D-13). The field grouping mirrors libvpx
order so a side-by-side audit is mechanical.

```rust
// crates/rvp9-encoder/src/svc/state.rs

#[derive(Debug)]
pub struct Svc {
    // -- identity -------------------------------------------------------
    pub spatial_layer_id:        u8,
    pub temporal_layer_id:       u8,
    pub number_spatial_layers:   u8,
    pub number_temporal_layers:  u8,

    /// Index of the spatial layer currently being prepared by start_layer().
    /// Set by the outer loop in `Encoder::encode`. Read by start_layer to
    /// drive the per-layer dispatch (vp9_svc_layercontext.c:520).
    pub spatial_layer_to_encode: u8,

    // -- iteration scope -----------------------------------------------
    pub first_spatial_layer_to_encode: u8,

    // -- layer table (per (sl, tl) cell) -------------------------------
    /// Length is `MAX_LAYERS`; only the first
    /// `number_spatial_layers * number_temporal_layers` cells are live.
    /// Layout: `idx = sl * ntl + tl` (LAYER_IDS_TO_IDX, vp9_encoder.h:1495).
    pub layers: Box<[LayerContext; MAX_LAYERS]>,

    // -- temporal layering mode + dispatch -----------------------------
    pub temporal_layering_mode: TemporalLayering,        // §4.3

    // -- reference-slot per-layer state (filled by set_flags_*) -------
    /// `lst_fb_idx[sl]`: which BufferPool slot LAST points at for spatial sl.
    pub lst_fb_idx:        [u8; MAX_SPATIAL_LAYERS],
    /// Likewise for GOLDEN (used for inter-layer prediction at sl>0).
    pub gld_fb_idx:        [u8; MAX_SPATIAL_LAYERS],
    /// ALT_REF slot. Note D-13: holds plain inter frames, not synthetic altref.
    pub alt_fb_idx:        [u8; MAX_SPATIAL_LAYERS],
    /// External flags (set by Bypass mode). 1 = update buffer slot bit.
    /// Bitmask over 8 BufferPool slots.
    pub update_buffer_slot:[u8; MAX_SPATIAL_LAYERS],
    pub reference_last:    [bool; MAX_SPATIAL_LAYERS],
    pub reference_golden:  [bool; MAX_SPATIAL_LAYERS],
    pub reference_altref:  [bool; MAX_SPATIAL_LAYERS],

    /// Last BufferPool slot to which spatial layer `sl`'s last TL=0 frame
    /// was written. -1 if none yet. Used to recover from a drop by redirecting
    /// `lst_fb_idx[sl]` (§6.4, L-75). Mirrors `fb_idx_upd_tl0[]`
    /// (vp9_svc_layercontext.h:322).
    pub fb_idx_upd_tl0:    [i8; MAX_SPATIAL_LAYERS],

    /// Per-pool-slot bookkeeping: which (sl, tl) wrote slot `i` last.
    /// Indexed 0..REF_FRAMES. Mirrors `fb_idx_spatial_layer_id[REF_FRAMES]`
    /// + `fb_idx_temporal_layer_id[REF_FRAMES]` (vp9_svc_layercontext.h:326).
    pub fb_idx_spatial_layer_id:  [u8; REF_FRAMES],
    pub fb_idx_temporal_layer_id: [u8; REF_FRAMES],

    // -- inter-layer prediction control --------------------------------
    pub disable_inter_layer_pred: InterLayerPred,
    pub simulcast_mode:           bool,
    pub force_zero_mode_spatial_ref: bool,

    // -- spatial scaling -----------------------------------------------
    /// One scaled-source buffer per spatial layer.
    pub scaler: SvcSourceScaler,
    /// Per-spatial mi-grid bookkeeping (Stage 1 A6 §2.8).
    pub mi_stride: [u32; MAX_SPATIAL_LAYERS],
    pub mi_rows:   [u32; MAX_SPATIAL_LAYERS],
    pub mi_cols:   [u32; MAX_SPATIAL_LAYERS],

    // -- cross-layer mode / partition reuse ----------------------------
    pub use_base_mv:         bool,
    pub use_partition_reuse: bool,
    /// `(mi_cols/8 + 1) * (mi_rows/8 + 1)` BlockSize entries.
    /// Allocated once at Encoder::new (HLD-svc §3.4). Per spatial layer.
    pub prev_partition_svc: Box<[BlockSize]>,

    // -- long-term GF temporal reference -------------------------------
    pub use_gf_temporal_ref:               bool,
    pub use_gf_temporal_ref_current_layer: bool,
    pub buffer_gf_temporal_ref: [BufferLongtermRef; 2],

    // -- scene-detect cross-layer flag (§5.3, L-68) --------------------
    /// Set by `vp9_scene_detection_onepass` (RC LLD) at sl=0, tl=0 only.
    /// Read by every (sl, tl) in this superframe to set
    /// `lc.rc.high_source_sad = svc.high_source_sad_superframe`.
    pub high_source_sad_superframe: bool,
    pub high_num_blocks_with_motion: u32,

    // -- non-reference / superframe accounting -------------------------
    pub non_reference_frame:      bool,
    pub current_superframe:       u64,
    pub previous_frame_is_intra_only: bool,
    pub set_intra_only_frame:     bool,
    pub superframe_has_layer_sync:bool,
    pub spatial_layer_sync:       [bool; MAX_SPATIAL_LAYERS],

    // -- per-layer Q (output of start_layer + pick_q) ------------------
    /// Written by pick_layer_qindex (§5.6). Read by upper-layer
    /// pick_q to enforce `lower_layer_qindex` floor (HLD-svc §4.2).
    pub base_qindex: [u8; MAX_SPATIAL_LAYERS],
    /// Mirrors `svc->lower_layer_qindex` (vp9_svc_layercontext.h:294).
    /// Updated to `base_qindex[sl-1]` at start of layer sl > 0.
    pub lower_layer_qindex: u8,

    // -- frame-drop state (§6) -----------------------------------------
    pub framedrop_mode:    FrameDropMode,
    pub framedrop_thresh:  [u8;  MAX_SPATIAL_LAYERS],
    pub drop_spatial_layer:[bool; MAX_SPATIAL_LAYERS],
    pub last_layer_dropped:[bool; MAX_SPATIAL_LAYERS],
    pub drop_count:        [u32;  MAX_SPATIAL_LAYERS],
    pub force_drop_constrained_from_above: [bool; MAX_SPATIAL_LAYERS],
    pub max_consec_drop:   u32,
    pub skip_enhancement_layer: bool,

    // -- denoiser cross-layer ------------------------------------------
    /// Smallest spatial-layer index that runs the denoiser (HLD-svc §4.4).
    /// Layers `[first_layer_denoise..number_spatial_layers)` are denoised.
    pub first_layer_denoise: u8,

    // -- downscale filter selection per layer --------------------------
    pub downsample_filter_type:  [InterpFilter; MAX_SPATIAL_LAYERS],
    pub downsample_filter_phase: [u8;           MAX_SPATIAL_LAYERS],

    // -- per-layer framerate (timestamp-driven) ------------------------
    pub duration:           [i64; MAX_SPATIAL_LAYERS],
    pub timebase_fac:       i64,
    pub time_stamp_superframe: i64,
    pub time_stamp_prev:    [i64; MAX_SPATIAL_LAYERS],

    pub num_encoded_top_layer: u32,
    /// True if the encoder collapsed to a single layer due to runtime
    /// reconfig. Mirrors `single_layer_svc` (vp9_svc_layercontext.h:357).
    pub single_layer_svc:   bool,
    pub resize_set:         bool,

    // -- Bypass-mode plumbing ------------------------------------------
    /// Caller-supplied per-frame ref-frame config (Bypass mode only).
    /// Stored from `FrameOptions::ref_frame_config` before each layer iterates.
    pub use_set_ref_frame_config: bool,
    pub temporal_layer_id_per_spatial: [u8; MAX_SPATIAL_LAYERS],
    pub ext_frame_flags:    [u32; MAX_LAYERS],

    /// True only on the first frame after Encoder::new, or after a hard
    /// keyframe reset. Used by start_layer to skip drop-state checks.
    pub first_frame: bool,
}

// `Send` is safe (no aliased shared state); `Sync` is NOT impl-ed
// because `Svc` is mutated freely on the main thread between frame
// encodes (HLD-01 model). Tile workers receive immutable snapshots.
unsafe impl Send for Svc {}
// No Sync impl.

/// Mirrors `BUFFER_LONGTERM_REF` (vp9_svc_layercontext.h:30).
#[derive(Debug, Clone, Copy, Default)]
pub struct BufferLongtermRef {
    pub idx:     u8,    // BufferPool slot index
    pub is_used: bool,
}
```

### 4.2 `LayerContext` field-by-field map and swap order (L-58)

This is the canonical resolution of L-58. The field set is the same
as HLD-svc §2.2 but the **swap behaviour is split into three groups**
based on the order in which `vp9_restore_layer_context` reads/writes
them.

Source: `vp9_svc_layercontext.c:352-389` (restore) and `:391-424`
(save). Every field below carries an explicit "Group" tag in the
struct comments so a Rust-reviewer can match against the C in one
glance.

```rust
// crates/rvp9-encoder/src/svc/state.rs

/// One per (spatial, temporal) cell.
///
/// **L-58 swap groups** (governs restore_layer_context / save_layer_context
/// ordering exactly mirroring vp9_svc_layercontext.c:352-424):
///
/// - **Group A (swap freely):** members copied between `lc` and the
///   encoder god-struct (RC + AQ-3 maps) at layer start / end.
/// - **Group B (overlay-then-restore):** fields whose libvpx
///   semantics is "save the *encoder*'s pre-restore value, do the
///   restore, then write it back". RC has three such fields
///   (`frames_since_key`, `frames_to_key`, `ext_use_post_encode_drop`,
///   `vp9_svc_layercontext.c:354-356, 372-375`).
/// - **Group C (write-only post-encode, read-only pre-encode):**
///   no swap needed — encoder reads `lc` directly, never writes (e.g.
///   `target_bandwidth` is replicated into `cpi->oxcf` by restore,
///   never written back).
#[derive(Debug)]
pub struct LayerContext {
    // ---- Group A: full RateControl swap ----
    /// Full per-layer rate-control state. Swapped value-wise with
    /// `Encoder::rc` (NOT pointer-swapped — `RateControl` contains no
    /// internal `Box`/`Arc`; see HLD-svc §10.1 assumption).
    /// vp9_svc_layercontext.c:358 `cpi->rc = lc->rc`.
    pub rc: RateControl,

    // ---- Group A: AQ-3 cyclic-refresh maps (only at tl=0 + ns>1) ----
    /// Cyclic-refresh segmentation map. None on non-base-temporal
    /// cells and on `number_spatial_layers == 1` configurations.
    /// vp9_svc_layercontext.c:158-179 (alloc gate), :380-383 (swap).
    /// Size: `mi_rows * mi_cols / 16` bytes.
    pub cr_map:              Option<Box<[i8]>>,
    /// Per-superblock last-coded Q. vp9_svc_layercontext.c:381.
    pub cr_last_coded_q_map: Option<Box<[u8]>>,
    /// Per-block consecutive-zero-MV counter (skin / static detect).
    /// vp9_svc_layercontext.c:383 `swap_ptr(&cpi->consec_zero_mv, ...)`.
    pub cr_consec_zero_mv:   Option<Box<[u8]>>,
    /// Cyclic-refresh sb_index (scan pointer). vp9_svc_layercontext.c:384.
    pub cr_sb_index:                  u32,
    pub cr_actual_num_seg1_blocks:    i32,
    pub cr_actual_num_seg2_blocks:    i32,
    pub cr_counter_encode_maxq_scene_change: i32,
    /// qindex_delta[3] for CR — copied back at save_layer_context only.
    /// vp9_svc_layercontext.c:420-422.
    pub cr_qindex_delta: [i32; 3],

    // ---- Group C (read-only at restore, written at save) ----
    /// Per-cell target. Replicated into `cpi->oxcf.target_bandwidth`
    /// at restore (vp9_svc_layercontext.c:360); rewritten from
    /// `oxcf->target_bandwidth` at save (:397).
    pub target_bandwidth:    i64,
    /// Spatial-layer-wide target (sums temporal-layer bandwidths).
    pub spatial_layer_target_bandwidth: i64,
    pub framerate:           f64,
    pub avg_frame_size:      i32,
    pub max_q:               u8,
    pub min_q:               u8,
    pub scaling_factor_num:  u16,
    pub scaling_factor_den:  u16,
    pub current_video_frame_in_layer: u32,
    pub is_key_frame:        bool,
    pub frames_from_key_frame: u32,
    pub last_frame_type:     FrameType,
    pub layer_size:          usize,

    // ---- Group C: per-cell speed feature stash --------------------
    /// Per-cell cpu-used. Applied to `cpi->oxcf.speed` at restore IF
    /// `is_one_pass_svc(cpi) && lc->speed > 0`
    /// (vp9_svc_layercontext.c:364-366). PLAN.md D1 → speed ∈ {5,6,7,8}.
    pub speed: u8,

    // ---- Group C: per-cell loopfilter strength override -----------
    /// `cpi->loopfilter_ctrl = lc->loopfilter_ctrl` at restore
    /// (vp9_svc_layercontext.c:367).
    pub loopfilter_ctrl: u8,

    // ---- Group C: written by save only ----------------------------
    /// `lc->frame_qp = cpi->common.base_qindex` at save
    /// (vp9_svc_layercontext.c:399).
    pub frame_qp: i32,
    pub mbs:      u32,

    /// True after the first successful encode of this cell. Lets
    /// start_layer distinguish "never encoded" vs "previously dropped".
    pub initialized: bool,
}
```

**Swap implementation (concrete L-58 resolution):**

```rust
// crates/rvp9-encoder/src/encoder.rs

impl Encoder {
    pub(crate) fn restore_layer_context(&mut self) {
        let idx = svc::layer_idx(
            self.svc.spatial_layer_id,
            self.svc.temporal_layer_id,
            self.svc.number_temporal_layers,
        );
        let lc = &mut self.svc.layers[idx];

        // Group B: stash the encoder's pre-restore values.
        // Mirrors vp9_svc_layercontext.c:354-356.
        let old_frames_since_key       = self.rc.frames_since_key;
        let old_frames_to_key          = self.rc.frames_to_key;
        let old_ext_use_post_encode_drop = self.rc.ext_use_post_encode_drop;

        // Group A: RC value-swap. Single move; `RateControl: Sized + !Drop-side-effects`.
        // Mirrors vp9_svc_layercontext.c:358 `cpi->rc = lc->rc;`.
        std::mem::swap(&mut self.rc, &mut lc.rc);
        // (We use swap rather than the libvpx assignment so we don't drop the
        // previous lc.rc value; semantically equivalent because both sides are
        // re-saved at save_layer_context.)

        // Group C: scalar copies from lc into the encoder-active fields.
        self.cfg_runtime.target_bandwidth = lc.target_bandwidth;     // :360

        // libvpx restores `cpi->alt_ref_source = lc->alt_ref_source` (:361)
        // — D-13 says no altref, so this assignment is ELIDED. Document in §7.

        // Per-cell speed override, when nonzero (vp9_svc_layercontext.c:364-366).
        if self.is_one_pass_svc() && lc.speed > 0 {
            self.cfg_runtime.speed = lc.speed;
        }
        self.cfg_runtime.loopfilter_ctrl = lc.loopfilter_ctrl;       // :367

        // Group B: restore stashed counters if multi-layer
        // (vp9_svc_layercontext.c:370-374).
        if self.svc.number_temporal_layers > 1 ||
           self.svc.number_spatial_layers  > 1 {
            self.rc.frames_since_key = old_frames_since_key;
            self.rc.frames_to_key    = old_frames_to_key;
        }
        // Always restore ext_use_post_encode_drop (vp9_svc_layercontext.c:375).
        self.rc.ext_use_post_encode_drop = old_ext_use_post_encode_drop;

        // CR map swap — only at base temporal of multi-spatial config.
        // Mirrors vp9_svc_layercontext.c:378-388, L-67 ownership rule.
        if self.cfg.aq_mode == AqMode::CyclicRefresh
            && self.svc.number_spatial_layers > 1
            && self.svc.temporal_layer_id == 0
        {
            std::mem::swap(&mut self.cr.map,
                           lc.cr_map.as_mut().expect("CR map allocated at sl,tl=0"));
            std::mem::swap(&mut self.cr.last_coded_q_map,
                           lc.cr_last_coded_q_map.as_mut().unwrap());
            std::mem::swap(&mut self.cr.consec_zero_mv,
                           lc.cr_consec_zero_mv.as_mut().unwrap());
            self.cr.sb_index                  = lc.cr_sb_index;
            self.cr.actual_num_seg1_blocks    = lc.cr_actual_num_seg1_blocks;
            self.cr.actual_num_seg2_blocks    = lc.cr_actual_num_seg2_blocks;
            self.cr.counter_encode_maxq_scene_change =
                lc.cr_counter_encode_maxq_scene_change;
        }
    }

    pub(crate) fn save_layer_context(&mut self) {
        let idx = svc::layer_idx(
            self.svc.spatial_layer_id,
            self.svc.temporal_layer_id,
            self.svc.number_temporal_layers,
        );
        let lc = &mut self.svc.layers[idx];

        // Group A: RC value-swap back. Mirrors :395.
        std::mem::swap(&mut self.rc, &mut lc.rc);
        // (libvpx assigns; the swap-back is equivalent. See restore comment.)

        // Group C: write encoder-side values back to lc.
        lc.target_bandwidth = self.cfg_runtime.target_bandwidth;     // :397
        lc.frame_qp         = self.cm.base_qindex as i32;            // :399
        lc.mbs              = self.cm.mbs;                           // :400
        lc.initialized      = true;

        if self.cfg.aq_mode == AqMode::CyclicRefresh
            && self.svc.number_spatial_layers > 1
            && self.svc.temporal_layer_id == 0
        {
            // Swap pointers back. libvpx uses three temps
            // (vp9_svc_layercontext.c:407-415); std::mem::swap is the
            // direct Rust equivalent.
            std::mem::swap(&mut self.cr.map,
                           lc.cr_map.as_mut().unwrap());
            std::mem::swap(&mut self.cr.last_coded_q_map,
                           lc.cr_last_coded_q_map.as_mut().unwrap());
            std::mem::swap(&mut self.cr.consec_zero_mv,
                           lc.cr_consec_zero_mv.as_mut().unwrap());
            lc.cr_sb_index                  = self.cr.sb_index;
            lc.cr_actual_num_seg1_blocks    = self.cr.actual_num_seg1_blocks;
            lc.cr_actual_num_seg2_blocks    = self.cr.actual_num_seg2_blocks;
            lc.cr_counter_encode_maxq_scene_change =
                self.cr.counter_encode_maxq_scene_change;
            lc.cr_qindex_delta = self.cr.qindex_delta;               // :420-422
        }
    }
}
```

**L-58 RESOLUTION (final):** the swap order is `Group B stash` →
`Group A RC swap` → `Group C scalar copies` → `Group B restore` →
(at base-TL only) CR-map swap. Save is the reverse of restore.
This matches `vp9_svc_layercontext.c:354-389` line-for-line and the
asymmetric stash/restore pattern around `frames_since_key`/`frames_to_key`/
`ext_use_post_encode_drop`. **Important:** the encoder's frame counters
are stream-global (not per-layer); this is libvpx's documented behaviour
and we mirror.

The HLD-svc §4.1 sketch is correct on the high level but glossed the
Group-B stash; this LLD is the canonical version. See §7 for the
divergence-ledger entry on D-13 eliminating `lc->alt_ref_source`.

### 4.3 `TemporalLayering`

Mirrors `vp9e_temporal_layering_mode` (`vpx/vp8cx.h:790`):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalLayering {
    NoLayering, // tl = 0, 0, ... (period 1)
    Mode0101,   // tl = 0, 1, 0, 1, ... (period 2)
    Mode0212,   // tl = 0, 2, 1, 2, ... (period 4)
    Bypass,     // application-driven via RefFrameConfig
}

impl TemporalLayering {
    /// Number of temporal layers consistent with this mode. Used by
    /// `SvcConfig::validate` (§3.2) to reject mismatched pairs.
    pub const fn implied_ntl(self) -> u8 {
        match self {
            TemporalLayering::NoLayering => 1,
            TemporalLayering::Mode0101   => 2,
            TemporalLayering::Mode0212   => 3,
            TemporalLayering::Bypass     => 0, // any (caller drives)
        }
    }
}

/// Computes the `temporal_layer_id` for a fixed-pattern mode given the
/// frame index *within this spatial layer*. Mirrors
/// vp9_svc_layercontext.c:520-528 (mode_0212) and :629 (mode_0101).
pub fn temporal_id_in_pattern(mode: TemporalLayering, frame_in_layer: u32) -> u8 {
    match mode {
        TemporalLayering::NoLayering => 0,
        TemporalLayering::Mode0101   => (frame_in_layer & 1) as u8,
        TemporalLayering::Mode0212   => {
            // libvpx (vp9_svc_layercontext.c:527-530):
            //   frame_num & 1 ? 2 : (frame_num >> 1)
            // pattern over frame_num % 4: 0→0, 1→2, 2→1, 3→2.
            let n = frame_in_layer & 3;
            if n & 1 != 0 { 2 } else { (n >> 1) as u8 }
        }
        TemporalLayering::Bypass => panic!(
            "temporal_id is application-driven in Bypass mode; \
             caller must supply via RefFrameConfig"),
    }
}
```

### 4.4 (Reserved — see §4.1 for `Svc`)

### 4.5 `LayerContext` constructor + CR-map alloc

```rust
impl LayerContext {
    /// Defaults per `vp9_init_layer_context` (vp9_svc_layercontext.c:30-150)
    /// minus altref/2-pass fields (D-13).
    pub fn new(
        sl: u8,
        tl: u8,
        cfg: &SvcConfig,
        rc_template: &RateControl,
    ) -> Self;

    /// Allocates `cr_map`, `cr_last_coded_q_map`, `cr_consec_zero_mv`
    /// when AQ-3 is enabled, number_spatial_layers > 1, and tl == 0.
    /// Mirrors vp9_svc_layercontext.c:158-179.
    ///
    /// `mi_rows`, `mi_cols` are the **top-spatial-layer** mi-grid
    /// dimensions (CR maps are sized for the largest spatial layer
    /// because the segmentation map is in original-resolution mi units).
    pub fn alloc_cyclic_refresh(&mut self, mi_rows: u32, mi_cols: u32);
}
```

### 4.6 `InterLayerPred` + `FrameDropMode`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterLayerPred {
    On, Off, OffNonKey, OnConstrained,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDropMode {
    ConstrainedLayerDrop,     // default; libvpx CONSTRAINED_LAYER_DROP
    LayerDrop,                // libvpx LAYER_DROP
    FullSuperframeDrop,       // libvpx FULL_SUPERFRAME_DROP; WebRTC default
    ConstrainedFromAboveDrop, // libvpx CONSTRAINED_FROM_ABOVE_DROP
}
```

### 4.7 Per-mode flag setters

Each is a `pub(crate) fn` on `Svc` (or `Encoder`, since it writes
`ref_frame_flags` / `ext_refresh_*` which live there). The signatures:

```rust
impl Encoder {
    /// vp9_svc_layercontext.c:517-621 (period 4).
    pub(crate) fn set_flags_mode_0212(&mut self);
    /// vp9_svc_layercontext.c:626-684 (period 2).
    pub(crate) fn set_flags_mode_0101(&mut self);
    /// vp9_svc_layercontext.c:687-720 (spatial-only).
    pub(crate) fn set_flags_mode_no_layering(&mut self);
    /// vp9_svc_layercontext.c:722-738. Reads
    /// `self.svc.lst_fb_idx/gld_fb_idx/alt_fb_idx/update_buffer_slot`
    /// which were filled by `Encoder::encode` from `FrameOptions::ref_frame_config`.
    pub(crate) fn set_flags_bypass(&mut self);
}
```

The bodies are direct ports of the per-row case tables transcribed in
A6 §5.4.1 / §5.4.2 / §5.4.3 and HLD-svc §2.2. Bytes are written by
direct field assignment; no abstraction beyond a single
`Encoder::write_layer_flags(sl, lst, gld, alt, refresh_last,
refresh_golden, refresh_alt, ref_flags)` helper.

For brevity the bodies are not transcribed here — they are mechanical.
But because libvpx's `set_flags_and_fb_idx_for_temporal_mode3` has 11
disjoint branches (`vp9_svc_layercontext.c:533-621`), the LLD requires
the implementation to mirror the C **branch-for-branch**, with the
case-table from HLD-svc §2.2 dropped into a doc-comment above the
function (and validated by the byte-parity test in §9.3).

### 4.8 `SvcSourceScaler` (per-layer down-scale buffers)

```rust
// crates/rvp9-encoder/src/svc/scaler.rs

#[derive(Debug)]
pub struct SvcSourceScaler {
    /// Pre-allocated scaled YV12 per spatial layer. Index 0 is the
    /// base; index `number_spatial_layers - 1` is unused at 1:1.
    pub scaled: Box<[Yv12Buffer]>, // len = number_spatial_layers
    /// Two-stage downsample scratch (used for 1:4 ratio via 1:2 → 1:2).
    pub temp:   Yv12Buffer,

    pub filter_type:  [InterpFilter; MAX_SPATIAL_LAYERS],
    pub filter_phase: [u8;            MAX_SPATIAL_LAYERS],
}

impl SvcSourceScaler {
    /// Allocates `scaled[sl]` and `temp` sized for the **top spatial layer's**
    /// dimensions for `temp`, per-spatial-layer for `scaled[sl]`.
    pub fn new(cfg: &SvcConfig, src_w: u32, src_h: u32) -> Self;

    /// Reset filter selection per `vp9_init_layer_context` defaults
    /// (vp9_svc_layercontext.c:73-74).
    pub fn reset_filter_defaults(&mut self);
}
```

D-13 simplification: in libvpx `svc->scaled_frames[MAX_LAG_BUFFERS]`
is a 1..N buffer ring (for temporal-filter altref); with
`lag_in_frames = 0` we only ever need *one* scaled frame per layer
per superframe, so `scaled` is `len = number_spatial_layers`, not
a multi-buffer ring. See §7 for the divergence-ledger entry.

### 4.9 `FrameOptions` cross-link

The `FrameOptions` and `RefFrameConfig` types are owned by the top-level
`Encoder` LLD (HLD-svc §8.2 sketches them). For self-containment the
SVC LLD pulls the relevant fields:

```rust
pub struct RefFrameConfig {
    pub temporal_layer_id: u8,
    pub reference_last:    [bool; MAX_SPATIAL_LAYERS],
    pub reference_golden:  [bool; MAX_SPATIAL_LAYERS],
    pub reference_altref:  [bool; MAX_SPATIAL_LAYERS],
    pub lst_fb_idx:        [u8;   MAX_SPATIAL_LAYERS],
    pub gld_fb_idx:        [u8;   MAX_SPATIAL_LAYERS],
    pub alt_fb_idx:        [u8;   MAX_SPATIAL_LAYERS],
    pub update_buffer_slot:[u8;   MAX_SPATIAL_LAYERS], // 8-slot bitmask
    pub duration:          [i64;  MAX_SPATIAL_LAYERS],
}
```

### 4.10 `Svc::new`

```rust
impl Svc {
    /// Single-shot initialiser. Allocates `layers`, `scaler`, and
    /// (when AQ-3 is on and ns > 1) cyclic-refresh maps for the base-TL
    /// cells. Mirrors vp9_init_layer_context (vp9_svc_layercontext.c:30-150)
    /// minus the altref/2-pass paths.
    pub fn new(cfg: &SvcConfig, src_w: u32, src_h: u32,
               aq_mode: AqMode, mi_rows_top: u32, mi_cols_top: u32,
               rc_template: &RateControl) -> Result<Self, EncodeError>;
}
```

---

## 5. Internal algorithm

### 5.1 Superframe loop

This is the encoder's outer loop, owned by `Encoder::encode`. The SVC
LLD documents it because every operation it performs is SVC-specific.

Mirrors `vp9_encoder.c:6383` inner loop (A6 §3.7):

```text
fn encode(&mut self, frame, opts) -> Result<EncodedSuperframe, _>:
    # Mirror vp9_encoder.c top-of-encode SVC bookkeeping (A6 §3.7):
    self.svc.time_stamp_superframe = opts.pts
    if let Some(ref cfg) = opts.ref_frame_config:
        # Bypass: caller drives temporal_id + ref flags; copy into self.svc.
        self.svc.use_set_ref_frame_config = true
        self.svc.temporal_layer_id = cfg.temporal_layer_id
        self.svc.lst_fb_idx = cfg.lst_fb_idx
        self.svc.gld_fb_idx = cfg.gld_fb_idx
        self.svc.alt_fb_idx = cfg.alt_fb_idx
        self.svc.update_buffer_slot = cfg.update_buffer_slot
        self.svc.reference_last  = cfg.reference_last
        self.svc.reference_golden = cfg.reference_golden
        self.svc.reference_altref = cfg.reference_altref

    # Drive scene detection once per superframe at (sl=0, tl=0).
    # See §5.3 (L-68 RESOLUTION).
    self.run_scene_detection_for_superframe(frame)

    # ConstrainedFromAbove drop precheck (runs once at sl=first_sl_to_encode).
    self.precheck_constrained_from_above_drops()

    let mut out = EncodedSuperframe::empty(self.svc.number_spatial_layers)
    for sl in self.svc.first_spatial_layer..self.svc.number_spatial_layers:
        if opts.skip_enhancement_layers and sl > self.svc.first_spatial_layer:
            break    # HLD-svc §8.4
        self.svc.spatial_layer_to_encode = sl
        self.start_layer(opts)?                # §5.2
        self.restore_layer_context()           # §4.2 / L-58
        let scaled = self.scale_source_for_layer(frame, sl)
        let layer_out = match self.check_layer_drop():
            true  => EncodedLayer::dropped(sl, self.svc.temporal_layer_id),
            false => self.encode_one_layer_inner(scaled, opts)?,
        self.save_layer_context()              # §4.2 / L-58
        # Post-encode buffer-level update happens INSIDE encode_one_layer
        # (via update_buffer_level_postencode) on success; on drop, the
        # drop-bookkeeping is done in check_layer_drop.
        self.inc_frame_in_layer()              # §5.5
        self.svc.time_stamp_prev[sl as usize] = opts.pts
        out.push_layer(layer_out)

    out.assemble_superframe_index()
    Ok(out)
```

The loop is **strictly single-threaded** (§4.1 HLD-svc §2.3
"layer iteration is strictly sequential; tile workers parallelize
*within* one layer's `encode_one_layer` call").

### 5.2 `start_layer` (libvpx parity, step-by-step)

Mirrors `vp9_one_pass_svc_start_layer` (`vp9_svc_layercontext.c:769-994`).
HLD-svc §2.4 listed the steps at a high level; this LLD enumerates the
implementation order and the libvpx line refs for each.

**Pseudocode (`Encoder::start_layer`):**

```text
let svc = &mut self.svc
svc.skip_enhancement_layer = false                      # :775

# Step 0: simulcast mode flag (:777-782)
svc.simulcast_mode = (
    svc.disable_inter_layer_pred == InterLayerPred::Off
    && svc.number_spatial_layers > 1
    && svc.number_spatial_layers <= 3
    && svc.number_temporal_layers <= 3)

# Step 1: default cross-layer flags ON for multi-spatial (:784-787)
if svc.number_spatial_layers > 1:
    svc.use_base_mv         = true
    svc.use_partition_reuse = true
svc.force_zero_mode_spatial_ref = true                  # :788

# Step 2: per-layer mi-grid bookkeeping (:789-791)
svc.mi_stride[sl] = self.cm.mi_stride
svc.mi_rows[sl]   = self.cm.mi_rows
svc.mi_cols[sl]   = self.cm.mi_cols

# Step 3: ConstrainedFromAbove drop precheck (§6.6) — already done
# once per superframe in encode(); this checks the per-sl flag only.

# Step 4: dispatch the temporal-mode flag setter (:819-830)
match svc.temporal_layering_mode:
    Mode0212    => self.set_flags_mode_0212()
    NoLayering  => self.set_flags_mode_no_layering()
    Mode0101    => self.set_flags_mode_0101()
    Bypass      => self.set_flags_bypass()         # uses ref_frame_config

# Step 5: long-term GF temporal-ref slot bookkeeping (:832-860)
if self.lst_fb_idx == svc.buffer_gf_temporal_ref[0].idx
   or self.gld_fb_idx == ... or self.alt_fb_idx == ...:
    svc.buffer_gf_temporal_ref[0].is_used = true
(likewise for slot 1)

svc.use_gf_temporal_ref_current_layer = (
    svc.use_gf_temporal_ref
    && !svc.buffer_gf_temporal_ref[0].is_used
    && !svc.buffer_gf_temporal_ref[1].is_used
    && svc.temporal_layering_mode != Bypass
    && svc.disable_inter_layer_pred != InterLayerPred::On
    && svc.number_spatial_layers   <= 3
    && svc.number_temporal_layers  <= 3
    && svc.spatial_layer_id >= svc.number_spatial_layers - 2)

# Step 6: superframe-level layer-sync check (only at sl=0, :862-871)
if svc.spatial_layer_id == 0:
    svc.superframe_has_layer_sync =
        svc.spatial_layer_sync[..svc.number_spatial_layers as usize]
           .iter().any(|&b| b)

# Step 7: superframe-start drop-flag reset (:874-897)
if svc.spatial_layer_id == svc.first_spatial_layer_to_encode:
    svc.drop_spatial_layer.fill(false)
    if svc.temporal_layering_mode != Bypass:
        svc.lst_fb_idx = [u8::MAX; MAX_SPATIAL_LAYERS]   # libvpx: -1
        svc.gld_fb_idx = [u8::MAX; MAX_SPATIAL_LAYERS]
        svc.alt_fb_idx = [u8::MAX; MAX_SPATIAL_LAYERS]
        svc.update_buffer_slot = [0; MAX_SPATIAL_LAYERS]
        svc.reference_last   = [false; MAX_SPATIAL_LAYERS]
        svc.reference_golden = [false; MAX_SPATIAL_LAYERS]
        svc.reference_altref = [false; MAX_SPATIAL_LAYERS]

# Step 8: SET_SVC_PARAMETERS-style per-layer Q range (:902-913)
let idx = svc::layer_idx(sl, tl, ntl)
let lc  = &mut svc.layers[idx]
if svc.temporal_layering_mode != Bypass or svc.use_set_ref_frame_config:
    lc.rc.worst_quality = quantizer_to_qindex(lc.max_q)
    lc.rc.best_quality  = quantizer_to_qindex(lc.min_q)

# Step 9: resolve resolution (:915-926)
let (num, den) = (lc.scaling_factor_num, lc.scaling_factor_den)
let (w, h) = svc::get_layer_resolution(
    self.cfg.frame_width, self.cfg.frame_height, num, den)

# Step 10: filter selection (:928-934)
if w * h <= 320 * 240:
    svc.downsample_filter_type[sl] = InterpFilter::EighttapSmooth
if num > (3 * den) >> 2:
    svc.downsample_filter_phase[sl] = 0

# Step 11: use_base_mv / use_partition_reuse gating (:936-961)
if svc.number_spatial_layers > 1:
    for sl_i in 0..svc.number_spatial_layers - 1:
        let lci = &svc.layers[sl_i * ntl + tl]
        let two_x_ok = lci.scaling_factor_num == lci.scaling_factor_den / 2
        let four_x_ok = lci.scaling_factor_num == lci.scaling_factor_den / 4
                        && sl_i == 0
                        && svc.number_spatial_layers == 3
        if !two_x_ok && !four_x_ok:
            svc.use_base_mv         = false
            svc.use_partition_reuse = false
            break
    if sl > 0 && svc.drop_spatial_layer[sl-1]:
        svc.use_base_mv         = false
        svc.use_partition_reuse = false

# Step 12: non_reference_frame flag (:963-972)
svc.non_reference_frame = false
if self.cm.frame_type != FrameType::Key
   && !self.ext_refresh_last_frame
   && !self.ext_refresh_golden_frame
   && !self.ext_refresh_alt_ref_frame:
    svc.non_reference_frame = true
if svc.temporal_layering_mode == Bypass
   && svc.update_buffer_slot[sl] != 0:
    svc.non_reference_frame = false

# Step 13: scene-detect cross-layer flags (clear at sl=0, :974-977)
if sl == 0:
    svc.high_source_sad_superframe   = false
    svc.high_num_blocks_with_motion  = 0
# (scene_detection_onepass — owned by RC LLD — will repopulate these
# BEFORE encode_one_layer_inner runs; we cleared at the top of the
# superframe in encode(), not here. See §5.3.)

# Step 14: drop-state recovery for previously dropped layer (:979-988).
# L-75 RESOLUTION — see §6.4.
if svc.temporal_layering_mode != Bypass
   && svc.last_layer_dropped[sl]
   && svc.fb_idx_upd_tl0[sl] != -1
   && !svc.layers[layer_idx(sl, 0, ntl)].is_key_frame:
    self.lst_fb_idx = svc.fb_idx_upd_tl0[sl] as u8

# Step 15: hand off to common's set_size_literal (:990-991).
self.set_frame_size(w, h)?

Ok(())
```

After `start_layer`, the encoder calls `restore_layer_context` (§4.2),
then `encode_one_layer_inner`, then `save_layer_context`. The
**`constrain_inter_layer_pred`** call (HLD-svc §2.4 step 5) is invoked
inside `encode_one_layer_inner` *after* the reference frames have been
scaled into `cm->frame_refs[*]`. Per libvpx it lives in
`vp9_svc_layercontext.c:1076-1147` and is called from `vp9_encoder.c`
just before pickmode / pickpartition starts; we mirror this call site
(no change to ordering).

### 5.3 Scene detection per layer (L-68 RESOLUTION)

`vp9_scene_detection_onepass` (`vp9_ratectrl.c:3034-3243`) is owned by
the rate-control LLD. The SVC-side wiring is fully resolved here:

**RESOLUTION:** scene detection runs **exactly once per superframe, on
the base-spatial / base-temporal cell**, before any layer iterates.
Its outputs propagate to all per-layer RCs via
`svc.high_source_sad_superframe`.

Rationale:

- `vp9_scene_detection_onepass` is **always-on** in non-screen content
  (L-23; A5 Q-RC-26 `Cannot disable; mirror as always-on`).
- Running it once per superframe is correct because the SAD computation
  uses the **source** frame, not a per-layer-scaled version — the
  intent is to detect global content change.
- Per-layer scene-detect is **out of scope for v1** (HLD-svc §4.3). If
  a future preset needs per-layer detection (e.g. for very different
  layer resolutions), it can branch off this site.

Wiring (called from `Encoder::encode`, before the spatial layer loop):

```rust
impl Encoder {
    fn run_scene_detection_for_superframe(&mut self, src: &SourceFrame) {
        // Mirror vp9_encoder.c call site at the top of the inner
        // encode loop (just before the spatial layer iteration begins).
        // Active conditions (vp9_ratectrl.c:3034-3046):
        // - cm.show_frame  (per A5; in our config this is always true at sl=0)
        // - oxcf.rc_mode in {CBR, VBR}
        // - content != ScreenContent (RC LLD owns the screen-content disable)
        // - cm.current_video_frame > 0 (skip the very first frame)
        if self.cfg.content == Content::ScreenContent {
            // RC LLD scene_detection bails on screen content; mirror.
            return;
        }
        if self.svc.current_superframe == 0 {
            return;
        }

        // The RC LLD does the heavy lifting: prev_frame_unscaled vs current
        // unscaled source frame, per-64x64 SAD, threshold logic, and writes
        // self.rc.high_source_sad / self.rc.avg_source_sad / etc.
        // Per L-68: we feed the *unscaled* source even though we'll later
        // run encode on a downscaled version per layer.
        crate::rate_control::scene_detection_onepass(self, src);

        // SVC fan-out: copy the base-layer rc flag into the cross-layer flag.
        // Mirrors vp9_ratectrl.c:2310-2320 area (within vp9_rc_get_svc_params).
        // The SVC flag is read by every (sl, tl) below to seed rc.high_source_sad.
        self.svc.high_source_sad_superframe = self.rc.high_source_sad;
        // avg_source_sad[64x64 grid] stays on self.rc — read by start_layer
        // step 14 of vp9_rc_get_svc_params, NOT swapped into layer contexts.
    }
}
```

In **`start_layer`** step 13 we *clear* the superframe-level flag at
sl=0 (`vp9_svc_layercontext.c:974-977`). Because we already ran scene
detection in `encode` and stored the result, this would clobber our
result. The fix: the libvpx clear-at-sl=0 happens **before**
`scene_detection` writes (the scene detect runs *inside*
`vp9_rc_get_svc_params` which is called per-layer at line 2310-2320,
*after* `start_layer`). Concretely, our ordering is:

1. `encode()` → call scene-detect (writes `svc.high_source_sad_superframe`).
2. `encode()` → spatial loop iterates.
3. `start_layer(sl=0)` step 13 → **DO NOT CLEAR**; we already wrote
   the value, and the flag is read by every layer in this superframe.

**Divergence note:** we move the scene-detection call up by one
function (from inside `vp9_rc_get_svc_params` to the top of
`Encoder::encode`). This is a behavioural no-op because libvpx still
clears the flag at the *first* `start_layer` and then writes it back
during `vp9_rc_get_svc_params` of the same sl=0 frame. We collapse
those two writes into one. Added to divergence ledger §7.

### 5.4 Per-layer rate control fan-out (L-77 RESOLUTION)

This is the canonical L-77 resolution. Source of truth:
`vp9_rc_get_svc_params` (`vp9_ratectrl.c:2279-2491`),
`vp9_update_buffer_level_svc_preencode` (`:263-294`),
`update_layer_buffer_level_postencode` (`:298-313`).

#### 5.4.1 Layer table indexing invariants

- `Svc.layers` is dense: `len = number_spatial_layers * number_temporal_layers`
  effective live cells (within the `MAX_LAYERS = 9` array).
- For `Mode0101`: `ntl = 2`, layout `[L00, L01, L10, L11, L20, L21]`
  where `Lxy = layer (sl=x, tl=y)`.
- For `Mode0212`: `ntl = 3`, layout `[L00, L01, L02, L10, L11, L12, L20, L21, L22]`.
- For `NoLayering`: `ntl = 1`, layout `[L00, L10, L20]`.

#### 5.4.2 Pre-encode buffer-level pump

Called *once per superframe* from `vp9_rc_get_svc_params` (libvpx
`:2453` calls `vp9_update_buffer_level_svc_preencode` from inside the
SVC params path). rvp9 wires this into `start_layer` at the **base
spatial layer only** (sl=0), pre-pumping all temporal layers:

```rust
impl Encoder {
    /// Mirrors vp9_update_buffer_level_svc_preencode (vp9_ratectrl.c:263-294).
    /// L-77 RESOLUTION: per-temporal-layer fan-out at sl=0 of each superframe.
    fn update_buffer_level_preencode(&mut self) {
        let sl  = self.svc.spatial_layer_id;
        let ntl = self.svc.number_temporal_layers;
        let ts_delta = self.svc.time_stamp_superframe
                     - self.svc.time_stamp_prev[sl as usize];

        for tl in self.svc.temporal_layer_id..ntl {
            let idx = svc::layer_idx(sl, tl, ntl);
            let lc  = &mut self.svc.layers[idx];
            // libvpx (vp9_ratectrl.c:276-284):
            //   if use_timestamp && use_set_ref_frame_config && ntl == 1 &&
            //      ts_delta > 0 && current_superframe > 0:
            //     bits_off_target += target_bandwidth / (10_000_000.0 / ts_delta)
            //   else:
            //     bits_off_target += target_bandwidth / framerate
            let bandwidth_increment = if self.svc.use_set_ref_frame_config
                && ntl == 1 && ts_delta > 0 && self.svc.current_superframe > 0
            {
                let framerate_pts = 10_000_000.0 / (ts_delta as f64);
                (lc.target_bandwidth as f64 / framerate_pts).round() as i64
            } else {
                (lc.target_bandwidth as f64 / lc.framerate).round() as i64
            };
            lc.rc.bits_off_target = (lc.rc.bits_off_target + bandwidth_increment)
                                        .min(lc.rc.maximum_buffer_size);
            lc.rc.buffer_level    = lc.rc.bits_off_target;

            // libvpx (:289-292): mirror the just-pumped value into cpi->rc
            // for the *currently encoding* tl. We do this only when tl
            // equals the active temporal_layer_id (i.e. the layer that's
            // about to be encoded).
            if tl == self.svc.temporal_layer_id {
                self.rc.bits_off_target = lc.rc.bits_off_target;
                self.rc.buffer_level    = lc.rc.buffer_level;
            }
        }
    }
}
```

Called from `Encoder::encode_one_layer_inner` *before* `pick_layer_qindex`,
matching the libvpx ordering inside `vp9_rc_get_svc_params`.

#### 5.4.3 Post-encode buffer-level update

```rust
impl Encoder {
    /// Mirrors update_buffer_level_postencode +
    /// update_layer_buffer_level_postencode (vp9_ratectrl.c:298-333).
    pub(crate) fn postencode_update_layer(&mut self, encoded_bits: i64) {
        // Update the active layer's rc (Group A — was swapped in by restore).
        self.rc.bits_off_target -= encoded_bits;
        self.rc.bits_off_target  = self.rc.bits_off_target
                                       .min(self.rc.maximum_buffer_size);
        // Screen-content underflow clamp (mirroring :322-326).
        if self.cfg.content == Content::ScreenContent
            && self.cfg.drop_frames_water_mark == 0
        {
            self.rc.bits_off_target = self.rc.bits_off_target
                                          .max(-self.rc.maximum_buffer_size);
        }
        self.rc.buffer_level = self.rc.bits_off_target;

        // Fan-out to upper temporal layers in the same spatial column.
        // Mirrors update_layer_buffer_level_postencode (:298-313).
        let sl  = self.svc.spatial_layer_id;
        let ntl = self.svc.number_temporal_layers;
        let cur_tl = self.svc.temporal_layer_id;
        for tl in (cur_tl + 1)..ntl {
            let idx = svc::layer_idx(sl, tl, ntl);
            let lc  = &mut self.svc.layers[idx];
            lc.rc.bits_off_target -= encoded_bits;
            lc.rc.bits_off_target  = lc.rc.bits_off_target
                                          .min(lc.rc.maximum_buffer_size);
            lc.rc.buffer_level     = lc.rc.bits_off_target;
        }
    }
}
```

Note the **temporal fan-out direction**: a TL=0 frame's bits debit
TL=1 and TL=2 (because they're displayed-by the same packet stream).
A TL=2 frame's bits do NOT debit TL=0 — it's "upper layers in the same
spatial column", strictly tl > cur_tl. This is the L-77 surprise: TL=0
buffer drains by every higher-TL frame it precedes, until the next TL=0
refills it.

#### 5.4.4 Per-layer Q allocation

Called from `Encoder::encode_one_layer_inner` after the buffer pre-pump:

```rust
impl Encoder {
    pub(crate) fn pick_layer_qindex(&mut self, opts: &FrameOptions)
        -> Result<u8, EncodeError>
    {
        let sl  = self.svc.spatial_layer_id;
        let tl  = self.svc.temporal_layer_id;

        // 1. Internal Q pick (rate-control LLD does the heavy lifting).
        //    Reads self.rc (which is the active layer's RC, restored at
        //    the top of encode_one_layer_inner).
        let internal_q = crate::rate_control::pick_q_and_bounds(self)?;

        // 2. Enforce lower_layer_qindex floor for sl > 0.
        //    Mirrors vp9_svc_layercontext.c:1066-1074 (post-pick write of
        //    svc.lower_layer_qindex; we *read* here per HLD-svc §4.2).
        //    Bug-for-bug: libvpx's check is `q < lower_layer_qindex`,
        //    we mirror exactly.
        let floored_q = if sl > 0 {
            internal_q.max(self.svc.lower_layer_qindex)
        } else {
            internal_q
        };

        // 3. EXT_RATECTRL hook (HLD-svc §7.3).
        let final_q = if let Some(ext) = &mut self.ext_rc {
            let info = self.build_encode_frame_info();   // §7.1
            let decision = ext.get_encode_frame_decision(&info);
            match decision.q_index {
                None      => floored_q,
                Some(extq) => {
                    // Clamp to layer bounds (HLD-svc §10.3 open Q4 answer).
                    let lc = &self.svc.layers[svc::layer_idx(sl, tl,
                                self.svc.number_temporal_layers)];
                    extq.clamp(lc.min_q, lc.max_q)
                }
            }
        } else {
            floored_q
        };

        // 4. Write back the per-spatial-layer Q for the next layer's
        //    floor enforcement. Mirrors vp9_encoder.c "if use_svc:
        //    svc.base_qindex[spatial_layer_id] = *q" (A6 §3.9, line 999).
        self.svc.base_qindex[sl as usize] = final_q;
        if sl + 1 < self.svc.number_spatial_layers {
            self.svc.lower_layer_qindex = final_q;
        }
        self.cm.base_qindex = final_q;
        Ok(final_q)
    }
}
```

#### 5.4.5 AQ-CR in SVC (L-77 part 4)

`CYCLIC_REFRESH_AQ` runs **only at temporal_layer_id == 0**
(`vp9_aq_cyclicrefresh.c:491`). The encoder branches on
`self.svc.temporal_layer_id == 0` before invoking
`crate::aq::cyclic_refresh::setup`.

CR-map ownership (L-67 mirror): the maps live in
`LayerContext.cr_map/cr_last_coded_q_map/cr_consec_zero_mv` for the
`(sl, 0)` cell only (allocated by `LayerContext::alloc_cyclic_refresh`,
§4.5). They are *not* allocated on `(sl, 1)` or `(sl, 2)` cells. The
swap (§4.2) only fires on `tl == 0`, which matches.

For TL > 0 frames, `self.cr` retains the segmentation map left from the
last TL=0 frame on the same spatial layer; mode-decision and quant read
it as before. We do not modify it.

### 5.5 `inc_frame_in_layer`

Mirrors `vp9_inc_frame_in_layer` (`vp9_svc_layercontext.c:444-452`):

```rust
impl Encoder {
    pub(crate) fn inc_frame_in_layer(&mut self) {
        let sl  = self.svc.spatial_layer_id;
        let ntl = self.svc.number_temporal_layers;
        let lc_base = &mut self.svc.layers[svc::base_tl_layer_idx(sl, ntl)];
        lc_base.current_video_frame_in_layer += 1;
        lc_base.frames_from_key_frame        += 1;
        if sl == self.svc.number_spatial_layers - 1 {
            self.svc.current_superframe += 1;
        }
    }
}
```

Note: the libvpx code increments **only the base-TL cell's** counter
(line 446-447). It does NOT increment the per-(sl, tl) counter for
the actively-encoded tl. The temporal-id pattern computation reads
this base-TL counter (`vp9_svc_layercontext.c:521-525`). rvp9 mirrors
exactly.

### 5.6 Per-layer scaled-source pipeline

```rust
impl Encoder {
    pub(crate) fn scale_source_for_layer<'a>(
        &'a mut self,
        src: &'a SourceFrame,
        sl: u8,
    ) -> &'a Yv12Buffer {
        let ntl = self.svc.number_temporal_layers;
        let lc  = &self.svc.layers[svc::base_tl_layer_idx(sl, ntl)];
        let (w, h) = svc::get_layer_resolution(
            self.cfg.frame_width,
            self.cfg.frame_height,
            lc.scaling_factor_num,
            lc.scaling_factor_den);

        if w == src.width() && h == src.height() {
            return src.as_yv12();
        }

        // Stage 1: if 1:4 downscale, do 2-stage via `temp` (HLD-svc §3.1).
        let needs_two_stage = lc.scaling_factor_den == 4
                              && lc.scaling_factor_num == 1;
        if needs_two_stage {
            let half_w = self.cfg.frame_width  / 2;
            let half_h = self.cfg.frame_height / 2;
            crate::scaling::scale_and_extend(
                src.as_yv12(),
                &mut self.svc.scaler.temp,
                self.svc.scaler.filter_type[sl as usize],
                self.svc.scaler.filter_phase[sl as usize],
                half_w, half_h);
            crate::scaling::scale_and_extend(
                &self.svc.scaler.temp,
                &mut self.svc.scaler.scaled[sl as usize],
                self.svc.scaler.filter_type[sl as usize],
                self.svc.scaler.filter_phase[sl as usize],
                w, h);
        } else {
            crate::scaling::scale_and_extend(
                src.as_yv12(),
                &mut self.svc.scaler.scaled[sl as usize],
                self.svc.scaler.filter_type[sl as usize],
                self.svc.scaler.filter_phase[sl as usize],
                w, h);
        }
        &self.svc.scaler.scaled[sl as usize]
    }
}
```

`crate::scaling::scale_and_extend` is owned by the shared `rvp9-core`
crate (HLD-02 §kernel-catalogue lists it as an SVC scaler RTCD
entry). It dispatches between SSE2/AVX2/NEON variants of
`vpx_scaled_horiz_2d_*` plus `vp9_resize_*` for the 1:2 path.

### 5.7 Per-layer framerate update

Mirrors `vp9_update_temporal_layer_framerate`
(`vp9_svc_layercontext.c:308-350`). Called from
`encode_one_layer_inner` for the active (sl, tl) when the temporal
period rolls. The arithmetic is `1-pass CBR rate division`:

```text
fn update_layer_framerate(svc, sl, tl):
    lc = layers[sl*ntl + tl]
    lc.framerate = cfg.framerate / (1 << (ntl - tl - 1)) for tl < ntl - 1
    lc.framerate = cfg.framerate for tl == ntl - 1
    # bits_off_target deltas already done in §5.4.2; this only updates
    # lc.framerate which feeds the next preencode pump.
```

Concrete signatures:

```rust
impl Encoder {
    pub(crate) fn update_temporal_layer_framerate(&mut self);
    pub(crate) fn update_spatial_layer_framerate(&mut self, framerate: f64);
}
```

### 5.8 Bypass-mode reference-config path

When `TemporalLayering::Bypass`:

1. The caller supplies `FrameOptions::ref_frame_config: Some(_)` for
   *every* `encode()` call.
2. `Encoder::encode` copies the fields into `self.svc`:
   `temporal_layer_id`, `lst_fb_idx`, `gld_fb_idx`, `alt_fb_idx`,
   `update_buffer_slot`, `reference_*`.
3. `Encoder::set_flags_bypass` (called from `start_layer` step 4)
   reads those fields and writes `self.lst_fb_idx / gld_fb_idx / alt_fb_idx`
   plus `self.ext_refresh_*_frame` from the supplied bitmasks.
4. `ext_refresh_frame_flags_pending = true` after the copy.

Mirrors `set_flags_and_fb_idx_bypass_via_set_ref_frame_config`
(`vp9_svc_layercontext.c:722-738`). The validation gate from HLD-svc
§10.3 Open Q1 is implemented as: if `update_buffer_slot[sl] & (1<<i)`
is set for some slot `i ∈ 0..8`, debug-assert that the encoder has
*ever* written slot `i` (i.e., `self.svc.fb_idx_spatial_layer_id[i] !=
u8::MAX`). Release-build silently accepts (libvpx behaviour).

### 5.9 Layer-RC reset on big bandwidth change

Mirrors `vp9_svc_check_reset_layer_rc_flag` (`vp9_svc_layercontext.c:1051-1074`).
Called from `vp9_change_config` (LLD-arch territory, L-70) when
`cfg.layer_target_kbps[]` changes between successive `encode()` calls.
If the avg_frame_bandwidth jumps by >1.5x or <0.5x for any spatial
layer, reset that layer's full TL-stack:

```rust
impl Svc {
    pub(crate) fn check_reset_layer_rc_flag(&mut self) {
        for sl in 0..self.number_spatial_layers {
            // Highest temporal layer of this spatial.
            let high_tl_idx = svc::layer_idx(sl, self.number_temporal_layers - 1,
                                              self.number_temporal_layers);
            let lrc = &self.layers[high_tl_idx].rc;
            let too_high = lrc.avg_frame_bandwidth
                            > (3 * lrc.last_avg_frame_bandwidth) >> 1;
            let too_low  = lrc.avg_frame_bandwidth
                            < lrc.last_avg_frame_bandwidth >> 1;
            if too_high || too_low {
                for tl in 0..self.number_temporal_layers {
                    let lc = &mut self.layers[svc::layer_idx(sl, tl,
                                self.number_temporal_layers)];
                    lc.rc.rc_1_frame      = 0;
                    lc.rc.rc_2_frame      = 0;
                    lc.rc.bits_off_target = lc.rc.optimal_buffer_level;
                    lc.rc.buffer_level    = lc.rc.optimal_buffer_level;
                }
            }
        }
    }
}
```

---

## 6. Frame-drop semantics (L-69 + L-75 RESOLUTION)

This section implements the position locked in HLD-svc §6. Code paths
mirror `vp9_test_drop`, `vp9_rc_drop_frame`, and
`vp9_rc_postencode_update_drop_frame` line by line.

### 6.1 State summary

State on `Svc`:

| Field                                    | When written                | When read                  |
|---                                       |---                          |---                         |
| `framedrop_mode`                         | `Encoder::new` (config)     | every per-layer drop test  |
| `framedrop_thresh[sl]`                   | `Encoder::new` / reconfig   | `test_drop` per sl         |
| `drop_spatial_layer[sl]`                 | `drop_frame` on drop; reset at sl=first | `start_layer` step 11 (use_base_mv gate), `drop_frame` per-mode crosscheck |
| `last_layer_dropped[sl]`                 | `drop_frame` on drop        | next superframe's `start_layer` step 14 (L-75) |
| `drop_count[sl]`                         | `drop_frame` on drop        | `test_drop` (max_consec_drop gate) |
| `force_drop_constrained_from_above[sl]`  | `precheck_constrained_from_above_drops` | `drop_frame` |
| `max_consec_drop`                        | config                      | `test_drop`                |
| `skip_enhancement_layer`                 | `drop_frame` (lower drop)   | `Encoder::encode` outer loop |
| `fb_idx_upd_tl0[sl]`                     | end of encode_one_layer for a successful TL=0 frame | `start_layer` step 14 (L-75 redirect) |

### 6.2 `Svc::test_drop` (pure drop test)

Mirrors `vp9_test_drop` (`vp9_ratectrl.c:513-564`):

```rust
impl Encoder {
    /// Pure drop test (no state mutation beyond decimation counter).
    /// Mirrors vp9_test_drop (vp9_ratectrl.c:513-564).
    pub(crate) fn test_drop(&mut self) -> bool {
        let svc = &mut self.svc;
        let rc  = &mut self.rc;
        let sl  = svc.spatial_layer_id as usize;

        let mut drop_thresh = self.cfg.drop_frames_water_mark;
        if self.cfg.svc.number_spatial_layers > 1
            || self.cfg.svc.number_temporal_layers > 1
        {
            // SVC path (:518-526).
            if svc.drop_count[sl] == svc.max_consec_drop {
                svc.drop_count[sl] = 0;
                return false;       // cap reached, force-encode
            }
            drop_thresh = svc.framedrop_thresh[sl] as u32;
        }

        if drop_thresh == 0 {
            return false;
        }
        if sl > 0 && svc.framedrop_mode == FrameDropMode::FullSuperframeDrop {
            return false;           // FullSuperframeDrop drives drops at sl=0 only
        }

        // Hard-drop on negative buffer (:533-537).
        // For FullSuperframeDrop, the test is buffer-below-thresh across all SLs.
        let must_drop = if svc.framedrop_mode == FrameDropMode::FullSuperframeDrop {
            self.check_buffer_below_thresh_all_layers(-1)
        } else {
            rc.buffer_level < 0
        };
        if must_drop { return true; }

        // Decimation arming (:539-561).
        let drop_mark = (drop_thresh as i64 * rc.optimal_buffer_level) / 100;
        let buf_above = if svc.framedrop_mode == FrameDropMode::FullSuperframeDrop {
            self.check_buffer_above_thresh_all_layers(drop_mark)
        } else {
            rc.buffer_level > drop_mark
        };
        let buf_below = if svc.framedrop_mode == FrameDropMode::FullSuperframeDrop {
            self.check_buffer_below_thresh_all_layers(drop_mark)
        } else {
            rc.buffer_level <= drop_mark
        };

        if buf_above && rc.decimation_factor > 0 {
            rc.decimation_factor -= 1;
        } else if buf_below && rc.decimation_factor == 0 {
            rc.decimation_factor = 1;
        }

        if rc.decimation_factor > 0 {
            if rc.decimation_count > 0 {
                rc.decimation_count -= 1;
                true
            } else {
                rc.decimation_count = rc.decimation_factor;
                false
            }
        } else {
            rc.decimation_count = 0;
            false
        }
    }
}
```

Helpers `check_buffer_*_thresh_all_layers` iterate `self.svc.layers`
across spatial layers at the current temporal_layer_id; mirrored from
libvpx's check helpers in `vp9_ratectrl.c`.

### 6.3 `check_layer_drop` (cross-layer aware)

Mirrors `vp9_rc_drop_frame` (`vp9_ratectrl.c:614-664`):

```rust
impl Encoder {
    /// Mirrors vp9_rc_drop_frame (vp9_ratectrl.c:614-664).
    /// Returns true if this layer must be skipped.
    /// On drop: updates svc.drop_spatial_layer, last_layer_dropped,
    /// drop_count, skip_enhancement_layer; calls postencode_update_drop_frame.
    pub(crate) fn check_layer_drop(&mut self) -> bool {
        let svc = &mut self.svc;
        let sl  = svc.spatial_layer_id;
        let prev_dropped = sl > 0 && svc.drop_spatial_layer[(sl - 1) as usize];

        // Mode-specific cross-layer constraint (:617-622).
        // CONSTRAINED_LAYER_DROP and FULL_SUPERFRAME_DROP: drop if lower did.
        // LAYER_DROP: never inherit from lower.
        // CONSTRAINED_FROM_ABOVE: handled via force_drop_constrained_from_above.
        let prev_forces_drop = prev_dropped
            && svc.framedrop_mode != FrameDropMode::LayerDrop
            && svc.framedrop_mode != FrameDropMode::ConstrainedFromAboveDrop;

        // ConstrainedFromAbove uses the precheck flag set in step 0 of encode().
        let cfa_forces_drop = svc.force_drop_constrained_from_above[sl as usize];

        let do_drop = prev_forces_drop
            || cfa_forces_drop
            || self.test_drop();

        if !do_drop {
            return false;
        }

        // Drop bookkeeping (mirroring vp9_rc_drop_frame post-test path :627-660).
        self.postencode_update_drop_frame();
        self.ext_refresh_frame_flags_pending = false;
        self.last_frame_dropped = true;

        let svc = &mut self.svc;   // re-borrow after self.* mutations
        svc.last_layer_dropped[sl as usize] = true;
        svc.drop_spatial_layer[sl as usize] = true;
        svc.drop_count[sl as usize] += 1;
        svc.skip_enhancement_layer = true;

        // libvpx increments frame_in_layer except when the entire superframe
        // dropped (:635-648). Translate the condition mechanically.
        let inc_frame_counter = svc.framedrop_mode == FrameDropMode::LayerDrop
            || (svc.framedrop_mode == FrameDropMode::ConstrainedFromAboveDrop
                && !svc.force_drop_constrained_from_above
                       [(svc.number_spatial_layers - 1) as usize])
            || !svc.drop_spatial_layer[0];

        if inc_frame_counter {
            // We do NOT call self.inc_frame_in_layer here; the outer loop
            // (encode()) always calls it. Instead we set a flag so the outer
            // loop knows whether to call it. **Divergence from libvpx**:
            // libvpx calls inc_frame_in_layer inline here, then the outer
            // loop calls it again on the next iteration. We unify into a
            // single call site to avoid double-increment risk.
            //
            // The flag below mirrors libvpx's bookkeeping by suppressing the
            // outer-loop increment when libvpx would NOT have called it.
            //
            // See §7 divergence ledger entry.
        } else {
            svc.suppress_outer_inc_frame_in_layer = true;
        }

        // Top-spatial-layer reset of skip_enhancement_layer if all lower
        // dropped too (:649-659).
        if sl == svc.number_spatial_layers - 1 {
            let all_dropped = (0..sl)
                .all(|i| svc.drop_spatial_layer[i as usize]);
            if all_dropped {
                svc.skip_enhancement_layer = false;
            }
        }

        true
    }
}
```

A new `Svc` field, `suppress_outer_inc_frame_in_layer: bool`, is
added solely to unify the increment call site. Reset to false at the
top of every layer iteration.

### 6.4 Drop chain — L-75 RESOLUTION

The chain semantics (HLD-svc §6.4) need three concrete write sites:

**1. End of `encode_one_layer_inner` on success**: record this frame's
slot as the "last good TL0" for its spatial layer:

```rust
if !self.svc.last_layer_dropped[sl as usize]
   && self.svc.temporal_layer_id == 0
{
    // Find which BufferPool slot was just refreshed with our reconstruction.
    // For non-Bypass modes, this is `self.lst_fb_idx` when ext_refresh_last_frame
    // was set, else `self.gld_fb_idx` for the sl>0 keyframe path
    // (vp9_svc_layercontext.c:842-845 of mode_0212).
    let upd_slot = if self.ext_refresh_last_frame {
        self.lst_fb_idx
    } else if self.ext_refresh_golden_frame {
        self.gld_fb_idx
    } else {
        // No slot refreshed (unusual at TL0); leave fb_idx_upd_tl0 alone.
        u8::MAX
    };
    if upd_slot != u8::MAX {
        self.svc.fb_idx_upd_tl0[sl as usize] = upd_slot as i8;
        self.svc.fb_idx_spatial_layer_id[upd_slot as usize] = sl;
        self.svc.fb_idx_temporal_layer_id[upd_slot as usize] = 0;
    }
}
```

**2. `start_layer` step 14**: if the *previous superframe's*
`(sl, *)` dropped AND we have a recorded `fb_idx_upd_tl0[sl]`, redirect
`self.lst_fb_idx` to point at that slot, so the encoder's LAST
reference is the last successfully-encoded TL0 of the same spatial
layer (skipping past the dropped intermediates):

```rust
// vp9_svc_layercontext.c:979-988 (mirrored verbatim).
if svc.temporal_layering_mode != TemporalLayering::Bypass
   && svc.last_layer_dropped[sl as usize]
   && svc.fb_idx_upd_tl0[sl as usize] != -1
   && !svc.layers[svc::layer_idx(sl, 0, ntl)].is_key_frame
{
    self.lst_fb_idx = svc.fb_idx_upd_tl0[sl as usize] as u8;
}
```

**3. `start_layer` step 11 (use_base_mv gating)**: disable
`use_base_mv` / `use_partition_reuse` for `sl > 0` if the previous
*same-superframe* layer dropped (already captured in §5.2 step 11).

**Drop-chain invariants (test fixtures, §9.5):**

- After any drop, no subsequent layer references a slot that has never
  been refreshed (BufferPool integrity).
- `fb_idx_upd_tl0[sl]` is either `-1` (never encoded a TL0 for `sl`)
  or points at a slot that holds a valid TL0 reconstruction from a
  prior superframe.
- After `FullSuperframeDrop`, `current_video_frame_in_layer` is **not
  incremented** for any layer (per the `inc_frame_counter` condition
  in §6.3).
- After `ConstrainedLayerDrop` of `sl_drop`, all `sl >= sl_drop`
  drop in the same superframe; their `current_video_frame_in_layer`
  counter does **not advance** for spatial layers above the drop
  boundary, but **does advance** at and below it (libvpx
  `vp9_inc_frame_in_layer` is called once per spatial layer; dropped
  layers that bail before the outer loop's `inc_frame_in_layer` are
  handled via the `suppress_outer_inc_frame_in_layer` flag).

### 6.5 `postencode_update_drop_frame`

Mirrors `vp9_rc_postencode_update_drop_frame` (`vp9_ratectrl.c:1998-2017`):

```rust
impl Encoder {
    pub(crate) fn postencode_update_drop_frame(&mut self) {
        self.rc.current_video_frame += 1;
        self.rc.frames_since_key    += 1;
        self.rc.frames_to_key       -= 1;
        self.rc.rc_1_frame           = 0;
        self.rc.rc_2_frame           = 0;
        self.rc.last_q[FrameType::Inter as usize] = self.cm.base_qindex;

        // SVC non-LAYER_DROP buffer-cap (mirroring :2010-2017):
        // prevent unbounded growth of upper-layer buffer.
        let mode = self.svc.framedrop_mode;
        if mode != FrameDropMode::LayerDrop
            && self.rc.buffer_level > self.rc.optimal_buffer_level
        {
            self.rc.buffer_level    = self.rc.optimal_buffer_level;
            self.rc.bits_off_target = self.rc.optimal_buffer_level;
        }
    }
}
```

### 6.6 `precheck_constrained_from_above_drops`

Runs once per superframe at the start (called from `Encoder::encode`):

```rust
impl Encoder {
    /// Mirrors vp9_one_pass_svc_start_layer:793-817 (but lifted out so
    /// it runs ONCE per superframe rather than inside start_layer where
    /// it'd re-run for each sl unnecessarily). Sets
    /// svc.force_drop_constrained_from_above[].
    fn precheck_constrained_from_above_drops(&mut self) {
        let svc = &mut self.svc;
        if svc.framedrop_mode != FrameDropMode::ConstrainedFromAboveDrop {
            svc.force_drop_constrained_from_above.fill(false);
            return;
        }
        // Top-down scan: find the highest spatial layer that would drop
        // under its own buffer level; force-drop it AND everything below.
        // Per libvpx, this temporarily borrows each layer's rc into
        // cpi->rc to call vp9_test_drop. We mirror — but we use a
        // read-only "would_drop" predicate to avoid mutating decimation
        // state on layers that won't ultimately encode.
        let mut force_below = false;
        for sl in (svc.first_spatial_layer..svc.number_spatial_layers).rev() {
            let idx = svc::layer_idx(sl, svc.temporal_layer_id, svc.number_temporal_layers);
            let lc_rc = &svc.layers[idx].rc;
            let would_drop = test_drop_readonly(lc_rc, svc, sl, &self.cfg);
            if would_drop {
                for sl2 in (svc.first_spatial_layer..=sl).rev() {
                    svc.force_drop_constrained_from_above[sl2 as usize] = true;
                }
                force_below = true;
                break;
            }
        }
        if !force_below {
            svc.force_drop_constrained_from_above.fill(false);
        }
    }
}

/// Pure (no state mutation) version of test_drop, for the
/// constrained-from-above scan.
fn test_drop_readonly(rc: &RateControl, svc: &Svc, sl: u8, cfg: &EncoderConfig)
    -> bool;
```

This is a small divergence from libvpx: libvpx temporarily clobbers
`cpi->rc` and calls `vp9_test_drop`, which mutates `rc->decimation_factor`
on layers that may not encode. Our `test_drop_readonly` avoids that
mutation, which is arguably a bug-fix; logged in §7.

---

## 7. Deliberate divergences from libvpx

Each entry below is a divergence-ledger seed (format in
`D:/language/rvp9-2/notes/divergence-ledger.md`).

### 7.1 No altref → drop `lc->alt_ref_source`, `lc->alt_ref_idx`, `lc->has_alt_frame`, `lc->gold_ref_idx`

**Where:** `rvp9_encoder::svc::LayerContext` does not carry altref
fields. libvpx counterpart: `vp9_svc_layercontext.h:55-58`.
**What libvpx does:** maintains a per-layer altref lookahead pointer
plus altref/golden buffer indices for the temporal-filter-fed altref
mechanism.
**What rvp9 does:** drop those four fields entirely. `LayerContext.rc`
also drops the `alt_ref_source`, `is_src_frame_alt_ref` etc. fields
(rate-control LLD owns that side).
**Why:** D-13 RESOLVED (no altref, `lag_in_frames = 0`).
**Byte-parity impact:** none in matched-config encodes.
**Compression-efficiency impact:** none in matched configs.
**Future plan:** revisit in a "near-real-time livestream" preset.

### 7.2 `svc.scaled_frames[MAX_LAG_BUFFERS]` collapsed to 1-buffer-per-layer

**Where:** `SvcSourceScaler.scaled: Box<[Yv12Buffer]>` of length
`number_spatial_layers`. libvpx counterpart:
`vp9_svc_layercontext.h:91-94`.
**What libvpx does:** allocates `MAX_LAG_BUFFERS` (= 25 in default
build) scaled-frame slots for the temporal-filter's altref pipeline.
**What rvp9 does:** allocates exactly one scaled-frame slot per
spatial layer.
**Why:** with `lag_in_frames = 0` (D-13) the temporal filter is
unreachable; only the current superframe's scaled-source is ever needed.
**Byte-parity impact:** none.
**Compression-efficiency impact:** none.
**Future plan:** restore the ring on altref support.

### 7.3 Scene detection moved one level up (called once per superframe)

**Where:** `Encoder::run_scene_detection_for_superframe`. libvpx
counterpart: scene detection is called from `vp9_rc_get_svc_params`
(`vp9_ratectrl.c:2310-2320` area).
**What libvpx does:** calls `vp9_scene_detection_onepass` inside
`vp9_rc_get_svc_params`, which itself runs at the *first*
`start_layer` (sl=0), after step 7 has cleared
`svc.high_source_sad_superframe`.
**What rvp9 does:** call scene detection once at the **top** of
`Encoder::encode`, before any `start_layer`. Do not clear the cross-
layer flag in `start_layer` step 13.
**Why:** semantic no-op (the libvpx clear happens immediately before
the libvpx write); the move eliminates the double-write and clarifies
which superframe-level flags are set when.
**Byte-parity impact:** none (the observable flag value at every read
site is identical).
**Compression-efficiency impact:** none.
**Future plan:** none.

### 7.4 `test_drop_readonly` for ConstrainedFromAbove precheck

**Where:** `Encoder::precheck_constrained_from_above_drops`. libvpx
counterpart: `vp9_svc_layercontext.c:793-817`.
**What libvpx does:** temporarily borrows each layer's `RateControl`
into `cpi->rc`, then calls `vp9_test_drop`. `vp9_test_drop` mutates
`rc->decimation_factor` / `decimation_count` as a side effect, which
leaves the per-layer RC in a slightly different state than if we'd
just *checked* the drop condition.
**What rvp9 does:** a pure `test_drop_readonly(...)` helper does the
same buffer-threshold arithmetic without mutating decimation state.
**Why:** the side-effect mutation in libvpx is arguably a bug (libvpx
TODO comment at the function `vp9_test_drop` does not flag it, but
A5 §4.8 step 1 implies the mutation should only occur for the
layer that is *actually* encoded). The readonly version is
behaviour-identical in the common case (no force-drop), and slightly
more correct when force-drop fires.
**Byte-parity impact:** minor — a few decimation-counter mutations
that libvpx made on dropped layers won't be made in rvp9. Observable
only in regression tests crafted to exercise the precheck path on
4+ superframes with persistent overshoot.
**Compression-efficiency impact:** none material.
**Future plan:** none.

### 7.5 Single call-site for `inc_frame_in_layer`

**Where:** `Encoder::encode` outer loop; gated on
`Svc.suppress_outer_inc_frame_in_layer`. libvpx counterpart:
inline call from `vp9_rc_drop_frame` (`vp9_ratectrl.c:647`) plus a
second call from `vp9_encoder.c` near `vp9_save_layer_context`.
**What libvpx does:** has two separate call sites (drop-path and
success-path), with subtle conditions deciding which fires.
**What rvp9 does:** one call site in `Encoder::encode`; a
`suppress_outer_inc_frame_in_layer` flag set by `check_layer_drop`
mirrors libvpx's "don't increment when whole superframe drops in
constrained mode" logic.
**Why:** eliminates a class of double-increment bugs and concentrates
the counter logic.
**Byte-parity impact:** none — the counter is read by the temporal-id
pattern computation in the *next* superframe, and the value matches
libvpx exactly.
**Compression-efficiency impact:** none.
**Future plan:** none.

### 7.6 Layer-context swap uses `std::mem::swap` (not C-style assignment)

**Where:** `Encoder::restore_layer_context` / `save_layer_context`.
libvpx counterpart: `vp9_svc_layercontext.c:358, 395` use plain
assignment (`cpi->rc = lc->rc`).
**What libvpx does:** value-copies via C assignment. Original
`cpi->rc` is lost (overwritten by `lc->rc`); `lc->rc` retains its
previous contents until `save_layer_context` overwrites.
**What rvp9 does:** `std::mem::swap`. Both sides retain their previous
values; `lc->rc` after restore contains the encoder's pre-restore
value, which is irrelevant garbage we ignore.
**Why:** Rust's move semantics require explicit handling of the
overwritten value. `swap` produces identical observable behaviour and
avoids a `Drop` call on the encoder's pre-restore `RateControl`
(which doesn't allocate, so it'd be a no-op anyway).
**Byte-parity impact:** none.
**Compression-efficiency impact:** none.
**Future plan:** none.

### 7.7 `MAX_LAYERS = 9`, `MAX_SPATIAL_LAYERS = MAX_TEMPORAL_LAYERS = 3`

**Where:** `rvp9_encoder::svc::{MAX_LAYERS, MAX_SPATIAL_LAYERS,
MAX_TEMPORAL_LAYERS}`. libvpx counterparts: `VPX_MAX_LAYERS = 12`,
`VPX_SS_MAX_LAYERS = 5`, `VPX_TS_MAX_LAYERS = 5`
(`vpx/vpx_encoder.h:42-48`).
**What libvpx does:** caps at 12/5/5.
**What rvp9 does:** caps at 9/3/3. `SvcConfig::validate` rejects
larger values.
**Why:** PLAN.md §2 commits to 2- and 3-layer SVC. 3 is tight;
shrinking the constants gives compile-time bounds and saves the cost
of 36 unused `LayerContext` slots.
**Byte-parity impact:** none in valid configs.
**Compression-efficiency impact:** none in valid configs.
**Future plan:** if 4-5 layer support is ever requested, bump the
constants and re-test.

### 7.8 ConstrainedFromAbove drop-chain at top-of-superframe (not per-layer)

**Where:** `Encoder::precheck_constrained_from_above_drops` runs once.
libvpx counterpart: the same scan runs at every `start_layer` for
`sl == first_spatial_layer_to_encode` (`vp9_svc_layercontext.c:797`),
which in practice means once per superframe at sl=0; but the gating
condition in libvpx re-evaluates each layer.
**What libvpx does:** the for-loop inside `start_layer` rescans on
every layer; the second iteration is wasted work because the flags
are already set.
**What rvp9 does:** lift it to `Encoder::encode` so it runs strictly
once.
**Why:** clarity + saving ~5 µs/superframe of redundant scan work.
**Byte-parity impact:** none.
**Compression-efficiency impact:** none.
**Future plan:** none.

### 7.9 EXT_RATECTRL Q-clamp to layer min/max

**Where:** `Encoder::pick_layer_qindex` step 3. libvpx counterpart:
EXT_RATECTRL Q comes from a different codepath (`vpx_ext_ratectrl`),
which does not clamp.
**What libvpx does:** if a `vpx_rc_funcs_t` external returns an
out-of-range Q, libvpx encodes at that Q — typically clamped only at
the very last second by `cm->base_qindex` field width.
**What rvp9 does:** clamp to `[lc.min_q, lc.max_q]` and emit a
`tracing::warn!` so the integrator notices.
**Why:** HLD-svc §10.3 Open Q4 RESOLVED toward clamp-by-default; an
out-of-range Q breaks buffer-model assumptions.
**Byte-parity impact:** divergence only when an external supplies an
out-of-range Q; in libvpx-matched configurations (where the external
respects bounds), none.
**Compression-efficiency impact:** none unless the caller deliberately
overrides bounds.
**Future plan:** none.

### 7.10 No-op for `lc->alt_ref_source = NULL` initialisation

**Where:** `LayerContext::new`. libvpx counterpart:
`vp9_init_layer_context` (`vp9_svc_layercontext.c:46`) sets
`alt_ref_source = NULL`.
**What libvpx does:** initialises a never-used field.
**What rvp9 does:** field doesn't exist.
**Why:** D-13. Cross-link to 7.1.
**Byte-parity impact:** none.
**Compression-efficiency impact:** none.
**Future plan:** none.

---

## 8. Open questions

### 8.1 (Resolved within this LLD)

L-58, L-61, L-68, L-69, L-75, L-77, L-67 ownership question — all resolved
by §4.2, §3.3, §5.3, §6, §6.4, §5.4, §5.4.5 respectively.

### 8.2 Cross-LLD bound, await peer LLD

- **`RateControl` field-by-field decomposition** — the L-58 swap groups
  refer to `RateControl` as if it were one Rust struct. Stage 1 A5
  describes ~70 fields (`vp9_ratectrl.c:RATE_CONTROL`). The
  rate-control LLD owns the in-struct decomposition; this LLD's
  assertion is that the *whole* struct value-swaps. **Bound: LLD-rc.**
  Risk: if LLD-rc decides some sub-structure needs `Arc`/`Box`, the
  value-swap pattern in §4.2 breaks; we'd need pointer-swap there.

- **`CyclicRefresh` layout** — the CR-map swap in §4.2 assumes
  `Encoder.cr: CyclicRefresh` exposes `map`, `last_coded_q_map`,
  `consec_zero_mv`, `sb_index`, `actual_num_seg{1,2}_blocks`,
  `counter_encode_maxq_scene_change`, `qindex_delta` as accessible
  fields. **Bound: LLD-aq.** Risk: if LLD-aq hides these behind a
  method API, our swap needs to go through getters/setters.

- **`Encoder.cfg_runtime` field set** — this LLD writes
  `cfg_runtime.target_bandwidth`, `cfg_runtime.speed`,
  `cfg_runtime.loopfilter_ctrl` during restore. The encoder LLD (or
  HLD-arch §3.6) needs to confirm those fields exist as
  mutable per-frame state separate from `cfg` (the static config).
  **Bound: encoder-mod LLD.**

- **`Encoder.lst_fb_idx` / `gld_fb_idx` / `alt_fb_idx` / `ext_refresh_*`**
  — these live on `Encoder` (mirrors `cpi->lst_fb_idx` etc.). HLD-arch
  §3.6 sketches the encoder god-struct as subsystem-split; the
  reference-frame slot fields probably belong to a small `RefFrameState`
  sub-struct. **Bound: encoder-mod LLD.**

### 8.3 Open for Arthur

1. **L-67 ownership confirmation**: this LLD claims `cr_map`,
   `cr_last_coded_q_map`, `cr_consec_zero_mv` live on `LayerContext`
   for the `(sl, 0)` cells only (per HLD-svc §4.3 item 4). LLD-aq
   may want them on the encoder's `CyclicRefresh` with a per-spatial-
   layer index instead. **Position: LLD-svc owns these per-cell;
   LLD-aq sees them only via the swap at restore/save.** Confirm.

2. **Bypass-mode strict validation** (HLD-svc §10.3 Open Q1):
   currently the LLD takes the "debug-assert on writes to never-
   refreshed slots" position. **Position: debug-assert is sufficient
   for v1; release-build silently mirrors libvpx (accept anything).**
   Confirm.

3. **`fb_idx_upd_tl0[]` semantics on simulcast mode**: when
   `simulcast_mode = true`, every spatial layer is independent and
   keys together. Does L-75's "redirect to last good TL0" apply?
   **Position: yes, identically — the simulcast key resets
   `is_key_frame = true` for the (sl, 0) cell, and the L-75 redirect
   skip-redirects on key (vp9_svc_layercontext.c:982). No special
   handling needed.** Confirm.

4. **Real-time speed selection per spatial layer**: PLAN.md D1 is
   "cpu-used 6-8". libvpx allows per-layer `lc.speed` (so e.g. base
   layer at speed 5, top layer at speed 7). Should `SvcConfig.speed_per_layer`
   enforce **`6..=8`** uniformly, or permit `5..=8` to expose the
   "qualitative split" (A6 §3.9 line 1004)? **Position: enforce
   `6..=8` for v1; speed 5 is the RD-vs-non-RD boundary and we
   never exercise it.** Confirm.

5. **`Svc::single_layer_svc` runtime collapse**: libvpx has
   `single_layer_svc` to handle the case where the encoder dynamically
   collapses to 1 layer (e.g. for screen-content overshoot). Does v1
   support this? **Position: no — `single_layer_svc` is set only
   at `Encoder::new` and never changes. Saves a non-trivial state-
   machine.** Confirm.

### 8.4 Genuine TBDs (no position taken)

None. Every behaviour referenced by this LLD has either a libvpx
mirror or an explicit divergence (§7).

---

## 9. Test strategy

### 9.1 Unit tests (in-tree)

| Test                                      | What                              | Where                                    |
|---                                        |---                                |---                                       |
| `temporal_id_in_pattern_smoke`            | period-2 + period-4 patterns over 100 frames | `crates/rvp9-encoder/src/svc/state.rs::tests` |
| `get_layer_resolution_even`               | width/height parity (libvpx `:464-466`) | `crates/rvp9-encoder/src/svc/mod.rs::tests` |
| `layer_idx_inverse`                       | round-trip `layer_idx ↔ (sl,tl)` | same                                     |
| `svc_config_validate_rejects_4_layers`    | refuses NSL=4                     | `crates/rvp9-encoder/src/svc/config.rs::tests` |
| `svc_config_validate_pairs_ntl_to_mode`   | rejects Mode0212 with ntl=2       | same                                     |
| `restore_save_round_trip_rc`              | value-swap restore/save preserves `RateControl` byte-for-byte | `crates/rvp9-encoder/tests/svc_swap.rs` |
| `cr_map_swap_only_at_tl0`                 | `tl=1` and `tl=2` do not touch CR maps | same                                     |
| `inc_frame_in_layer_increments_base_only` | `(sl, 0)` cell's counter increments; `(sl, 1)`/`(sl, 2)` do not | same |
| `superframe_counter_increments_at_top_sl` | `current_superframe` increments at sl=NSL-1 only | same |
| `max_worker_count_top_layer`              | L-61: pool sized to top-layer max_tile_cols | `crates/rvp9-encoder/src/svc/mod.rs::tests` |

### 9.2 Integration tests against libvpx

| Test                                         | What                                                        |
|---                                           |---                                                          |
| `tests/svc/extract-layers.rs`                | per-layer sub-bitstream extraction round-trip (HLD-svc §11.2) |
| `tests/svc/ref-slot-integrity.rs`            | reference-slot table matches per-(tl, sl) §2.2 of HLD-svc / §5.4 A6 |
| `tests/svc/drop-chain.rs`                    | each of 4 `FrameDropMode` exhibits documented chain semantics (§6.4) |
| `tests/svc/start-layer-byte-parity.rs`       | post-`start_layer` state (`lst/gld/alt_fb_idx`, `ref_frame_flags`, `ext_refresh_*`) byte-equal to libvpx for all (tl, sl) cells of Mode0101 and Mode0212 |
| `tests/svc/temporal-pattern.rs`              | `Mode0212` periodicity over 16 superframes matches libvpx's `temporal_layer_id` exactly |
| `tests/svc/bypass-ref-frame-config.rs`       | Bypass mode honours caller flags; non-refreshed-slot reads warn in debug |

### 9.3 Property tests

| Property                              | Description |
|---                                    |---          |
| `restore_then_save_is_identity_on_layer_state` | For every (sl, tl) and arbitrary plausible `RateControl`, `restore_layer_context` then `save_layer_context` returns the layer to its starting state — modulo the Group B stash fields, which gracefully restore to their pre-restore encoder values |
| `drop_chain_buffer_pool_integrity`    | For every `FrameDropMode`, over 32 superframes with simulated bandwidth squeeze, no frame ever references a slot that has never been refreshed |
| `mode_0212_period_4`                  | `temporal_id_in_pattern(Mode0212, n)` has period exactly 4 |
| `pick_layer_qindex_floor`             | For sl > 0, `pick_layer_qindex` never returns a Q below `svc.base_qindex[sl-1]` |

### 9.4 Differential tests

`tests/differential/svc-rc-trace.rs`: drive identical input through
rvp9 and libvpx at matched config; assert that for every (superframe,
sl, tl) the recorded `(buffer_level, bits_off_target, base_qindex)`
triple matches libvpx within ±1 (rounding tolerance for the
float-to-int conversions in `bits_off_target +=
target_bandwidth / framerate`). Fails the build if the per-layer
trace diverges materially — the diagnostic is which call site
introduced the divergence.

### 9.5 SVC frame-drop regression

Per HLD-svc §11.5, plus this LLD adds:

- **Drop-then-recover chain**: encode 8 superframes with a forced
  squeeze at superframe 3. Assert:
  - For `ConstrainedLayerDrop`: at superframe 3, sl=0 drops, sl=1 and
    sl=2 are forced to drop too. At superframe 4, sl=0 redirects
    `lst_fb_idx` per L-75 to `fb_idx_upd_tl0[0]`, which equals slot 0
    (the last good TL0).
  - For `LayerDrop`: sl=0 drops independently; sl=1's `start_layer`
    step 5 (`constrain_inter_layer_pred`) clears its `VP9_GOLD_FLAG`
    because slot `sl-1 = 0` wasn't refreshed this superframe.
  - For `FullSuperframeDrop`: all three SLs drop; no `inc_frame_in_layer`
    fires (per the libvpx `vp9_inc_frame_in_layer` skip-rule we mirror).
  - For `ConstrainedFromAboveDrop`: assert
    `force_drop_constrained_from_above[2] = force_drop_constrained_from_above[1]
    = force_drop_constrained_from_above[0] = true` after the
    superframe-3 precheck.

### 9.6 Real-time budget

Per PLAN.md §2 the 720p30 / 1080p30 / 1080p60 budgets must hold under
the full 3-spatial × 3-temporal config. The `bench/realtime-svc.rs`
harness (stage 4d/4e exit gate) measures end-to-end ms/frame with the
SVC layer iteration included; this LLD's contribution is to ensure
the layer-iteration overhead (start_layer + restore + save + scaling +
postencode + inc) totals **≤ 1 ms per layer** at 720p30. Concrete
sub-budgets:

| Operation                               | Target (1080p30, top layer) |
|---                                      |---                          |
| `start_layer`                           | ≤ 50 µs                     |
| `restore_layer_context` (no CR-map swap) | ≤ 10 µs                     |
| `restore_layer_context` (with CR-map swap) | ≤ 25 µs (pointer swap)   |
| `save_layer_context` (mirror of restore) | ≤ 25 µs                    |
| `scale_source_for_layer` (1:1, just borrow) | ≤ 1 µs (no work)        |
| `scale_source_for_layer` (1:2, single-stage) | ≤ 1 ms at 1080p (SIMD'd) |
| `scale_source_for_layer` (1:4, two-stage)    | ≤ 2 ms at 1080p          |
| `postencode_update_layer`               | ≤ 5 µs                      |
| `inc_frame_in_layer`                    | ≤ 2 µs                      |

These are sanity floors; the encode-tile-MT phase dominates per-layer
time at 1080p (>10 ms median).

---

## 10. Cross-references

| Document                                     | Path                                                         |
|---                                           |---                                                           |
| HLD-svc                                      | `D:/language/rvp9-2/notes/hld/04-svc.md`                     |
| HLD-arch                                     | `D:/language/rvp9-2/notes/hld/00-architecture.md`            |
| HLD-threading                                | `D:/language/rvp9-2/notes/hld/01-threading.md`               |
| HLD-simd                                     | `D:/language/rvp9-2/notes/hld/02-simd-dispatch.md`           |
| A1 architecture                              | `D:/language/rvp9-2/notes/vp9-analysis/01-architecture.md`   |
| A5 rate-control (per-layer RC)               | `D:/language/rvp9-2/notes/vp9-analysis/05-rate-control.md`   |
| A6 threading + SVC (primary)                 | `D:/language/rvp9-2/notes/vp9-analysis/06-threading-svc.md`  |
| A7 devil's advocate (L-68 / L-75 / L-77 origin) | `D:/language/rvp9-2/notes/vp9-analysis/07-devils-advocate.md` |
| A8 overview                                  | `D:/language/rvp9-2/notes/vp9-analysis/00-overview.md`       |
| decisions-needed                             | `D:/language/rvp9-2/notes/vp9-analysis/decisions-needed.md`  |
| divergence-ledger                            | `D:/language/rvp9-2/notes/divergence-ledger.md`              |

| libvpx file (most-cited)                                                    |
|---                                                                          |
| `D:/language/rvp9-2/third_party/libvpx/vp9/encoder/vp9_svc_layercontext.h`  |
| `D:/language/rvp9-2/third_party/libvpx/vp9/encoder/vp9_svc_layercontext.c`  |
| `D:/language/rvp9-2/third_party/libvpx/vp9/encoder/vp9_ratectrl.c`          |
| `D:/language/rvp9-2/third_party/libvpx/vp9/encoder/vp9_ethread.c`           |
| `D:/language/rvp9-2/third_party/libvpx/vp9/encoder/vp9_speed_features.c`    |

---

**End of LLD-svc.**
