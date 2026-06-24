//! SVC (Scalable Video Coding) layer-context scaffold.
//!
//! Port of libvpx `vp9/encoder/vp9_svc_layercontext.{h,c}`. This is the
//! **W6.5 + W6.6 scaffold**: data-structure definitions, the per-layer
//! container, the layer-id → flat-index helper, save/restore plumbing,
//! and the four temporal-layering pattern setters. The encoder driver
//! is *not* wired up — `StreamEncoder` does not call into this module
//! yet; that lands in the integration wave that follows.
//!
//! ## Scope
//!
//! WHAT IS PORTED
//! - [`Svc`] container — mirrors libvpx `SVC` (`vp9_svc_layercontext.h:80-208`).
//! - [`LayerContext`] — mirrors libvpx `LAYER_CONTEXT`
//!   (`vp9_svc_layercontext.h:40-78`).
//! - Limit constants (`MAX_SPATIAL_LAYERS`, `MAX_TEMPORAL_LAYERS`,
//!   `MAX_VPX_LAYERS`, …) — mirror libvpx
//!   `VPX_SS_MAX_LAYERS` / `VPX_TS_MAX_LAYERS` / `VPX_MAX_LAYERS`
//!   (`vpx/vpx_encoder.h:42-48`).
//! - [`layer_ids_to_idx`] — mirrors libvpx macro
//!   `LAYER_IDS_TO_IDX(sl, tl, num_tl) ((sl) * (num_tl) + (tl))`
//!   (`vp9_encoder.h:1495`).
//! - [`Svc::is_one_pass_svc`] — mirrors libvpx
//!   `is_one_pass_svc(cpi)` (`vp9_encoder.h:1411-1413`).
//! - [`Svc::alloc_layers`] — mirrors the `LAYER_CONTEXT` alloc/init
//!   loop in libvpx `vp9_init_layer_context`
//!   (`vp9_svc_layercontext.c:105-181`) for the CBR / pass==0 path.
//! - [`Svc::save_layer_context`] / [`Svc::restore_layer_context`]
//!   — mirror libvpx `vp9_save_layer_context` and
//!   `vp9_restore_layer_context` (`vp9_svc_layercontext.c:352-424`).
//!   The cyclic-refresh swap-pointer dance is **stubbed** (TODO below).
//! - Four temporal-mode setters (W6.6):
//!     - [`Svc::set_flags_and_fb_idx_for_temporal_mode3`]
//!       — 0-2-1-2 / 3-layer pattern, libvpx
//!       `set_flags_and_fb_idx_for_temporal_mode3`
//!       (`vp9_svc_layercontext.c:517-621`).
//!     - [`Svc::set_flags_and_fb_idx_for_temporal_mode2`]
//!       — 0-1-0-1 / 2-layer pattern, libvpx
//!       `set_flags_and_fb_idx_for_temporal_mode2`
//!       (`vp9_svc_layercontext.c:626-684`).
//!     - [`Svc::set_flags_and_fb_idx_for_temporal_mode_no_layering`]
//!       — no temporal layering, libvpx
//!       `set_flags_and_fb_idx_for_temporal_mode_noLayering`
//!       (`vp9_svc_layercontext.c:688-720`).
//!     - [`Svc::set_flags_and_fb_idx_bypass_via_set_ref_frame_config`]
//!       — application-driven bypass, libvpx
//!       `set_flags_and_fb_idx_bypass_via_set_ref_frame_config`
//!       (`vp9_svc_layercontext.c:722-738`).
//!
//! WHAT IS DELIBERATELY DEFERRED (TODOs cross-referenced to libvpx)
//! - **Two-pass SVC** (`oxcf.pass != 0`): the `TWO_PASS` payload inside
//!   `LAYER_CONTEXT` is carried as a stub field
//!   (libvpx `vp9_svc_layercontext.h:53`); the
//!   `vp9_init_second_pass_spatial_svc` path
//!   (`vp9_svc_layercontext.c:427-441`) is not wired.
//! - **Lookahead empty frame** for multi-frame-context workaround
//!   (libvpx `vp9_svc_layercontext.h:89-91`,
//!   `vp9_svc_layercontext.c:88-103`).
//! - **Scaled-source / temporal-filter alt-ref buffers**
//!   (libvpx `vp9_svc_layercontext.h:95-99`).
//! - **Cyclic-refresh map swap** in save/restore (libvpx
//!   `vp9_svc_layercontext.c:376-388, 402-423`) — rvp9
//!   `aq_cyclic_refresh` does not own swappable buffers yet.
//! - **Denoiser** (`CONFIG_VP9_TEMPORAL_DENOISING` paths).
//! - **HBD** (skipped per project policy).
//! - **`vp9_one_pass_svc_start_layer` / `vp9_svc_start_frame`** — the
//!   per-frame dispatch wrappers that call the four setters. The
//!   integration wave will own these.
//!
//! ## Design choice: per-layer `RateControl`
//!
//! libvpx's `LAYER_CONTEXT` embeds a full `RATE_CONTROL` by value
//! (`vp9_svc_layercontext.h:41`). A naive Rust mirror — array
//! `[LayerContext; MAX_VPX_LAYERS]` with `RateControl` embedded — would
//! be ~12 × `sizeof(RateControl)` on the stack (RateControl carries
//! `[f64; 5]` factor tables + history). We instead box per-layer
//! `RateControl` (`Box<RateControl>`). The layer-context vector is
//! sized to `number_spatial_layers * number_temporal_layers` at alloc
//! time and indexed via [`layer_ids_to_idx`]. This keeps the heap
//! footprint proportional to the configured layer count (1 box per
//! active layer) and matches libvpx behaviour: a single layer encode
//! still uses exactly one `RATE_CONTROL`-shaped object.
//!
//! ## Tests
//!
//! Module-level PROOF tests live in [`tests`] (≥5, each citing
//! libvpx file:line). No encoder integration tests yet.

use crate::rate_control::{RateControl, RateControlConfig};

// ---------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------

/// Maximum spatial layers. Mirrors libvpx `VPX_SS_MAX_LAYERS = 5`
/// (`vpx/vpx_encoder.h:48`).
pub const MAX_SPATIAL_LAYERS: usize = 5;

/// Maximum temporal layers. Mirrors libvpx `VPX_TS_MAX_LAYERS = 5`
/// (`vpx/vpx_encoder.h:42`).
pub const MAX_TEMPORAL_LAYERS: usize = 5;

/// Maximum total (spatial × temporal) layers in libvpx's flat
/// `layer_context[]` array. Mirrors libvpx `VPX_MAX_LAYERS = 12`
/// (`vpx/vpx_encoder.h:45`). The libvpx comment notes "3 temporal +
/// 4 spatial layers are allowed" — i.e. up to 4 × 3 = 12 active slots.
pub const MAX_VPX_LAYERS: usize = 12;

/// Reference-frame slot count. Mirrors libvpx `REF_FRAMES = 8`
/// (`vp9_onyxc_int.h:38`, `REF_FRAMES = 1 << REF_FRAMES_LOG2 = 1 << 3`).
/// Used to size `fb_idx_spatial_layer_id` / `fb_idx_temporal_layer_id`
/// / `fb_idx_base`.
pub const REF_FRAMES: usize = 8;

/// `LAYER_IDS_TO_IDX(sl, tl, num_tl) == sl * num_tl + tl`.
/// Mirrors libvpx macro (`vp9_encoder.h:1495`). The flat index points
/// into `Svc::layer_context`.
#[inline]
#[must_use]
pub const fn layer_ids_to_idx(sl: usize, tl: usize, num_tl: usize) -> usize {
    sl * num_tl + tl
}

// ---------------------------------------------------------------------
// Enums (mirror libvpx)
// ---------------------------------------------------------------------

/// Inter-layer prediction policy. Mirrors libvpx `INTER_LAYER_PRED`
/// (`vp9_svc_layercontext.h:22-33`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum InterLayerPred {
    /// Inter-layer prediction on every frame. libvpx `INTER_LAYER_PRED_ON = 0`.
    On = 0,
    /// Inter-layer prediction off on every frame. libvpx `INTER_LAYER_PRED_OFF = 1`.
    Off = 1,
    /// Off on non-key non-sync frames. libvpx `INTER_LAYER_PRED_OFF_NONKEY = 2`.
    OffNonKey = 2,
    /// On but constrained: layer S only predicts from layer S-1 in
    /// the same superframe. libvpx `INTER_LAYER_PRED_ON_CONSTRAINED = 3`.
    OnConstrained = 3,
}

/// Temporal-layering mode selector. Mirrors libvpx
/// `VP9E_TEMPORAL_LAYERING_MODE` (`vpx/vp8cx.h:790-809`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TemporalLayeringMode {
    /// No temporal layering. libvpx `VP9E_TEMPORAL_LAYERING_MODE_NOLAYERING = 0`.
    NoLayering = 0,
    /// External bypass; application provides ref-frame config per
    /// spatial layer. libvpx `VP9E_TEMPORAL_LAYERING_MODE_BYPASS = 1`.
    Bypass = 1,
    /// 2-layer 0-1-0-1 pattern. libvpx `VP9E_TEMPORAL_LAYERING_MODE_0101 = 2`.
    Pattern0101 = 2,
    /// 3-layer 0-2-1-2 pattern. libvpx `VP9E_TEMPORAL_LAYERING_MODE_0212 = 3`.
    Pattern0212 = 3,
}

/// Frame-drop mode for SVC. Mirrors libvpx `SVC_LAYER_DROP_MODE` enum
/// values used in `vp9_svc_layercontext.h:149` and `vp9_ratectrl.c`.
/// Default in libvpx is `CONSTRAINED_LAYER_DROP`
/// (`vp9_svc_layercontext.c:51`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum SvcLayerDropMode {
    /// Layers may drop independently per superframe.
    LayerDrop = 0,
    /// Constrained: if base drops, enhancement drops too. **Default**
    /// (libvpx `vp9_svc_layercontext.c:51`).
    #[default]
    ConstrainedLayerDrop = 1,
    /// Drop from highest enhancement layer downward.
    FullSuperframeDrop = 2,
    /// Drop propagates from higher layers to lower until a layer fits.
    ConstrainedFromAboveDrop = 3,
}

// ---------------------------------------------------------------------
// VP9 frame-flag bit values (mirror libvpx).
// ---------------------------------------------------------------------
//
// libvpx uses `VP9_LAST_FLAG`, `VP9_GOLD_FLAG`, `VP9_ALT_FLAG` as
// bitfield masks in `cpi->ref_frame_flags`. Defined in
// `vpx/vp8.h` (the VP8/9 shared header):
//   #define VP8_LAST_FRAME   (1 << 0)
//   #define VP8_GOLD_FRAME   (1 << 1)
//   #define VP8_ALTR_FRAME   (1 << 2)
// VP9 inherits these directly via the VP9_*_FLAG aliases used in
// `vp9_svc_layercontext.c`.

/// `LAST` reference bit in `ref_frame_flags`. libvpx `VP8_LAST_FRAME =
/// 1 << 0`.
pub const VP9_LAST_FLAG: u32 = 1 << 0;
/// `GOLDEN` reference bit. libvpx `VP8_GOLD_FRAME = 1 << 1`.
pub const VP9_GOLD_FLAG: u32 = 1 << 1;
/// `ALTREF` reference bit. libvpx `VP8_ALTR_FRAME = 1 << 2`.
pub const VP9_ALT_FLAG: u32 = 1 << 2;

// ---------------------------------------------------------------------
// LayerContext — the per-layer state container
// ---------------------------------------------------------------------

/// Per-(spatial, temporal) layer encoder state. Mirrors libvpx
/// `LAYER_CONTEXT` (`vp9_svc_layercontext.h:40-78`).
///
/// Field naming follows libvpx wherever possible. The `RateControl` is
/// boxed (see module doc "Design choice"). Several libvpx fields are
/// stubbed as either `()` placeholders or marked TODO — they're
/// carried for shape-fidelity so future waves can light them up
/// without an ABI break.
#[derive(Debug)]
pub struct LayerContext {
    /// Per-layer rate-control state. Mirrors libvpx `LAYER_CONTEXT::rc`
    /// (`vp9_svc_layercontext.h:41`). Boxed to keep
    /// `[LayerContext; N]` cheap on the stack.
    pub rc: Box<RateControl>,

    /// Target bitrate for this exact (sl, tl) layer in bits / second.
    /// libvpx `LAYER_CONTEXT::target_bandwidth`
    /// (`vp9_svc_layercontext.h:42`).
    pub target_bandwidth: i32,

    /// Per-spatial-layer aggregate target. libvpx
    /// `spatial_layer_target_bandwidth` (`vp9_svc_layercontext.h:43`).
    pub spatial_layer_target_bandwidth: i32,

    /// Effective framerate for this layer (parent fps / ts_decimator).
    /// libvpx `LAYER_CONTEXT::framerate` (`vp9_svc_layercontext.h:44`).
    pub framerate: f64,

    /// Running average of encoded frame size at this layer (bits).
    /// libvpx `LAYER_CONTEXT::avg_frame_size`
    /// (`vp9_svc_layercontext.h:45`).
    pub avg_frame_size: i32,

    /// Per-layer Q clamps. libvpx `max_q` / `min_q`
    /// (`vp9_svc_layercontext.h:46-47`).
    pub max_q: i32,
    /// See `max_q`.
    pub min_q: i32,

    /// Spatial scaling fraction (`num / den`) — libvpx
    /// `LAYER_CONTEXT::scaling_factor_num` / `scaling_factor_den`
    /// (`vp9_svc_layercontext.h:48-49`).
    pub scaling_factor_num: i32,
    /// See `scaling_factor_num`.
    pub scaling_factor_den: i32,

    /// Single-layer SVC resize scaling factors. libvpx
    /// `scaling_factor_num_resize` / `scaling_factor_den_resize`
    /// (`vp9_svc_layercontext.h:51-52`).
    pub scaling_factor_num_resize: i32,
    /// See `scaling_factor_num_resize`.
    pub scaling_factor_den_resize: i32,

    // TODO(W6.x integration): TWO_PASS state — libvpx
    // `LAYER_CONTEXT::twopass` (`vp9_svc_layercontext.h:53`) and
    // `rc_twopass_stats_in` (`vp9_svc_layercontext.h:54`). Not modeled
    // yet (rvp9 only ports pass==0 / one-pass CBR today).
    /// Count of frames already encoded *at this layer*. libvpx
    /// `current_video_frame_in_layer` (`vp9_svc_layercontext.h:55`).
    /// This is the field the temporal-mode setters read modulo the
    /// pattern length to compute `temporal_layer_id`.
    pub current_video_frame_in_layer: u32,

    /// "Is this frame the key frame for this layer?" libvpx
    /// `is_key_frame` (`vp9_svc_layercontext.h:56`). Read by the
    /// temporal setters when deciding which buffer to refresh on tl0.
    pub is_key_frame: bool,

    /// Frames since this layer's most recent key. libvpx
    /// `frames_from_key_frame` (`vp9_svc_layercontext.h:57`).
    pub frames_from_key_frame: u32,

    /// Last frame type encoded at this layer (`KEY` / `INTER`).
    /// libvpx `last_frame_type` (`vp9_svc_layercontext.h:58`); the
    /// enum is `FRAME_TYPE` (`{KEY_FRAME, INTER_FRAME, FRAME_TYPES}`).
    /// We carry `None` for the libvpx sentinel `FRAME_TYPES`.
    pub last_frame_type: Option<LayerFrameType>,

    // TODO(integration): alt_ref_source — libvpx
    // `LAYER_CONTEXT::alt_ref_source` (`vp9_svc_layercontext.h:59`).
    // Pointer into the lookahead queue; not yet plumbed.
    /// Reference-buffer index for this layer's alt-ref. libvpx
    /// `alt_ref_idx` (`vp9_svc_layercontext.h:60`); init value
    /// `INVALID_IDX = -1` in libvpx.
    pub alt_ref_idx: i32,

    /// Reference-buffer index for golden. libvpx `gold_ref_idx`
    /// (`vp9_svc_layercontext.h:61`).
    pub gold_ref_idx: i32,

    /// Boolean: layer has an alt-ref frame allocated. libvpx
    /// `has_alt_frame` (`vp9_svc_layercontext.h:62`).
    pub has_alt_frame: bool,

    /// Most recent encoded size at this layer in bytes. libvpx
    /// `layer_size` (`vp9_svc_layercontext.h:63`).
    pub layer_size: usize,

    // ---- Cyclic refresh side-state (per-layer copies, libvpx
    // ---- `vp9_svc_layercontext.h:64-72`).
    // TODO(integration): the libvpx swap-pointer dance in
    // save/restore (libvpx `vp9_svc_layercontext.c:376-388, 402-423`)
    // copies CYCLIC_REFRESH map / last_coded_q_map / consec_zero_mv
    // pointers in/out of the layer context. We carry the metadata
    // counters but defer the buffer plumbing.
    /// Cyclic-refresh superblock cursor. libvpx `sb_index`
    /// (`vp9_svc_layercontext.h:66`).
    pub sb_index: i32,
    /// Per-superblock segment id map (one entry per MI block). libvpx
    /// `signed char *map` (`vp9_svc_layercontext.h:67`).
    /// `Vec<i8>` is empty until the integration wave allocates.
    pub map: Vec<i8>,
    /// Per-superblock last-coded-Q snapshot. libvpx `uint8_t
    /// *last_coded_q_map` (`vp9_svc_layercontext.h:68`).
    pub last_coded_q_map: Vec<u8>,
    /// Consecutive-zero-MV counter per MI block. libvpx `uint8_t
    /// *consec_zero_mv` (`vp9_svc_layercontext.h:69`).
    pub consec_zero_mv: Vec<u8>,
    /// Number of seg-1 blocks counted last refresh pass. libvpx
    /// `actual_num_seg1_blocks` (`vp9_svc_layercontext.h:70`).
    pub actual_num_seg1_blocks: i32,
    /// Number of seg-2 blocks counted last refresh pass. libvpx
    /// `actual_num_seg2_blocks` (`vp9_svc_layercontext.h:71`).
    pub actual_num_seg2_blocks: i32,
    /// Scene-change counter (resets on detected scene change). libvpx
    /// `counter_encode_maxq_scene_change`
    /// (`vp9_svc_layercontext.h:72`).
    pub counter_encode_maxq_scene_change: i32,

    /// Per-segment Q-index deltas (3 segments). libvpx
    /// `qindex_delta[3]` (`vp9_svc_layercontext.h:73`).
    pub qindex_delta: [i32; 3],

    /// Encoder speed setting for this layer. libvpx `uint8_t speed`
    /// (`vp9_svc_layercontext.h:74`). Used by libvpx
    /// `vp9_restore_layer_context` to override `cpi->oxcf.speed`
    /// when `is_one_pass_svc && speed > 0`.
    pub speed: u8,

    /// Loop-filter strength override. libvpx `loopfilter_ctrl`
    /// (`vp9_svc_layercontext.h:75`).
    pub loopfilter_ctrl: i32,

    /// `cm->base_qindex` at the moment the layer was last saved.
    /// libvpx `frame_qp` (`vp9_svc_layercontext.h:76`).
    pub frame_qp: i32,

    /// MB count at the layer's resolution. libvpx `MBs`
    /// (`vp9_svc_layercontext.h:77`).
    pub mbs: i32,
}

/// Per-layer frame type. Mirrors libvpx `FRAME_TYPE`
/// (`vp9_common.h`-level enum, `KEY_FRAME = 0`, `INTER_FRAME = 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LayerFrameType {
    /// `KEY_FRAME`.
    Key = 0,
    /// `INTER_FRAME`.
    Inter = 1,
}

impl LayerContext {
    /// Construct an initialized layer context. Mirrors the init body
    /// in libvpx `vp9_init_layer_context` for one (sl, tl) slot
    /// (`vp9_svc_layercontext.c:108-152`).
    ///
    /// `rc_config` seeds the boxed [`RateControl`]; `target_bandwidth`
    /// becomes `lc->target_bandwidth = oxcf->layer_target_bitrate[layer]`
    /// (libvpx line 131).
    #[must_use]
    pub fn new(rc_config: RateControlConfig, target_bandwidth: i32) -> Self {
        Self {
            rc: Box::new(RateControl::new(rc_config)),
            target_bandwidth,
            spatial_layer_target_bandwidth: 0,
            framerate: 0.0,
            avg_frame_size: 0,
            max_q: 0,
            min_q: 0,
            scaling_factor_num: 1,
            scaling_factor_den: 1,
            scaling_factor_num_resize: 1,
            scaling_factor_den_resize: 1,
            current_video_frame_in_layer: 0,
            is_key_frame: false,
            frames_from_key_frame: 0,
            // libvpx seeds to `FRAME_TYPES` (the sentinel past end of
            // valid range); represented here as `None`.
            last_frame_type: None,
            // libvpx `INVALID_IDX = -1` (`vp9_ratectrl.h`).
            alt_ref_idx: -1,
            gold_ref_idx: -1,
            has_alt_frame: false,
            layer_size: 0,
            sb_index: 0,
            map: Vec::new(),
            last_coded_q_map: Vec::new(),
            consec_zero_mv: Vec::new(),
            actual_num_seg1_blocks: 0,
            actual_num_seg2_blocks: 0,
            counter_encode_maxq_scene_change: 0,
            qindex_delta: [0; 3],
            speed: 0,
            loopfilter_ctrl: 0,
            frame_qp: 0,
            mbs: 0,
        }
    }
}

// ---------------------------------------------------------------------
// Svc — top-level container
// ---------------------------------------------------------------------

/// Top-level SVC state. Mirrors libvpx `SVC`
/// (`vp9_svc_layercontext.h:80-208`).
///
/// Field naming follows libvpx; arrays sized by [`MAX_VPX_LAYERS`] /
/// [`MAX_SPATIAL_LAYERS`] / [`REF_FRAMES`] match the libvpx fixed
/// array bounds. `layer_context` itself is a heap `Vec` whose length
/// is `number_spatial_layers * number_temporal_layers` after
/// [`Svc::alloc_layers`] runs.
#[derive(Debug)]
pub struct Svc {
    // ---- Core ids (libvpx `vp9_svc_layercontext.h:81-86`) ----
    /// Spatial layer being encoded *right now*. libvpx
    /// `SVC::spatial_layer_id`.
    pub spatial_layer_id: i32,
    /// Temporal layer of the current frame. libvpx
    /// `SVC::temporal_layer_id`.
    pub temporal_layer_id: i32,
    /// Number of spatial layers configured. libvpx
    /// `SVC::number_spatial_layers`.
    pub number_spatial_layers: i32,
    /// Number of temporal layers configured. libvpx
    /// `SVC::number_temporal_layers`.
    pub number_temporal_layers: i32,
    /// "Spatial layer the encoder driver intends to encode next" —
    /// the temporal setters read this and assign it into
    /// `spatial_layer_id`. libvpx `SVC::spatial_layer_to_encode`.
    pub spatial_layer_to_encode: i32,

    /// Master switch: is SVC enabled at all? Mirrors libvpx
    /// `VP9_COMP::use_svc` (`vp9_encoder.h:709`). The libvpx
    /// `is_one_pass_svc` predicate ANDs this with `pass == 0`.
    pub use_svc: bool,

    /// Encoder pass: 0 (single-pass / CBR), 1 (first-pass stats),
    /// 2 (second pass). Mirrors libvpx `VP9EncoderConfig::pass`. We
    /// hang this off `Svc` here because the only consumer in scope
    /// is [`Svc::is_one_pass_svc`].
    pub pass: u8,

    /// Layer-context array, length `number_spatial_layers *
    /// number_temporal_layers`. libvpx
    /// `SVC::layer_context[VPX_MAX_LAYERS]`
    /// (`vp9_svc_layercontext.h:103`).
    pub layer_context: Vec<LayerContext>,

    /// Active temporal-layering mode. libvpx
    /// `SVC::temporal_layering_mode` (`vp9_svc_layercontext.h:106`).
    pub temporal_layering_mode: TemporalLayeringMode,

    /// Per-spatial-layer external frame flags. libvpx
    /// `ext_frame_flags[VPX_MAX_LAYERS]` (`vp9_svc_layercontext.h:109`).
    pub ext_frame_flags: [i32; MAX_VPX_LAYERS],
    /// Per-spatial-layer reference index: `last` / `golden` / `altref`.
    /// libvpx `lst_fb_idx` / `gld_fb_idx` / `alt_fb_idx`
    /// (`vp9_svc_layercontext.h:110-112`).
    pub lst_fb_idx: [i32; MAX_VPX_LAYERS],
    /// See `lst_fb_idx`.
    pub gld_fb_idx: [i32; MAX_VPX_LAYERS],
    /// See `lst_fb_idx`.
    pub alt_fb_idx: [i32; MAX_VPX_LAYERS],

    /// libvpx `force_zero_mode_spatial_ref`
    /// (`vp9_svc_layercontext.h:113`).
    pub force_zero_mode_spatial_ref: i32,

    /// Sequence-level: enable second (long-term) temporal reference.
    /// libvpx `use_gf_temporal_ref` (`vp9_svc_layercontext.h:115`).
    pub use_gf_temporal_ref: bool,
    /// Frame-level mirror. libvpx `use_gf_temporal_ref_current_layer`
    /// (`vp9_svc_layercontext.h:117`).
    pub use_gf_temporal_ref_current_layer: bool,

    /// Current superframe counter (incremented when the top spatial
    /// layer is encoded). libvpx `current_superframe`
    /// (`vp9_svc_layercontext.h:120`).
    pub current_superframe: i32,

    /// libvpx `non_reference_frame` (`vp9_svc_layercontext.h:121`).
    pub non_reference_frame: bool,

    /// libvpx `single_layer_svc` (`vp9_svc_layercontext.h:206`).
    pub single_layer_svc: bool,
    /// libvpx `simulcast_mode` (`vp9_svc_layercontext.h:203`).
    pub simulcast_mode: bool,

    /// Application-driven ref-frame config selector (gate for
    /// `set_flags_and_fb_idx_bypass_via_set_ref_frame_config`).
    /// libvpx `use_set_ref_frame_config`
    /// (`vp9_svc_layercontext.h:187`).
    pub use_set_ref_frame_config: bool,

    /// Per-spatial-layer external temporal-layer id (bypass mode).
    /// libvpx `temporal_layer_id_per_spatial[VPX_SS_MAX_LAYERS]`
    /// (`vp9_svc_layercontext.h:189`).
    pub temporal_layer_id_per_spatial: [i32; MAX_SPATIAL_LAYERS],

    /// External enhancement-layer reference selectors (bypass mode).
    /// libvpx `reference_last` / `reference_golden` / `reference_altref`
    /// (`vp9_svc_layercontext.h:161-163`).
    pub reference_last: [u8; MAX_SPATIAL_LAYERS],
    /// See `reference_last`.
    pub reference_golden: [u8; MAX_SPATIAL_LAYERS],
    /// See `reference_last`.
    pub reference_altref: [u8; MAX_SPATIAL_LAYERS],

    /// Frame-drop policy. libvpx `framedrop_mode`
    /// (`vp9_svc_layercontext.h:149`).
    pub framedrop_mode: SvcLayerDropMode,

    /// Inter-layer prediction policy. libvpx `disable_inter_layer_pred`
    /// (`vp9_svc_layercontext.h:151`).
    pub disable_inter_layer_pred: InterLayerPred,

    // ---- ext_refresh_*_frame and lst/gld/alt_fb_idx of the
    // ---- *current* frame — libvpx hangs these off VP9_COMP, not
    // ---- SVC. The temporal setters write them, so we mirror them
    // ---- here as scratch state. The integration wave will route
    // ---- them out to the StreamEncoder frame_loop.
    /// `cpi->ext_refresh_frame_flags_pending` — gate flag set by the
    /// temporal setters. libvpx `VP9_COMP::ext_refresh_frame_flags_pending`
    /// (`vp9_encoder.h`).
    pub ext_refresh_frame_flags_pending: bool,
    /// `cpi->ext_refresh_last_frame`. libvpx
    /// `VP9_COMP::ext_refresh_last_frame`.
    pub ext_refresh_last_frame: bool,
    /// `cpi->ext_refresh_golden_frame`.
    pub ext_refresh_golden_frame: bool,
    /// `cpi->ext_refresh_alt_ref_frame`.
    pub ext_refresh_alt_ref_frame: bool,
    /// `cpi->ref_frame_flags` — bit-OR of `VP9_LAST_FLAG` / `_GOLD_` /
    /// `_ALT_`. libvpx `VP9_COMP::ref_frame_flags`.
    pub ref_frame_flags: u32,
    /// `cpi->lst_fb_idx` / `gld_fb_idx` / `alt_fb_idx` for the
    /// current frame. libvpx `VP9_COMP::lst_fb_idx` etc.
    pub current_lst_fb_idx: i32,
    /// See `current_lst_fb_idx`.
    pub current_gld_fb_idx: i32,
    /// See `current_lst_fb_idx`.
    pub current_alt_fb_idx: i32,
}

impl Default for Svc {
    fn default() -> Self {
        Self::new()
    }
}

impl Svc {
    /// Empty SVC container; matches libvpx defaults set in
    /// `vp9_init_layer_context` prologue
    /// (`vp9_svc_layercontext.c:38-65`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            // libvpx `vp9_svc_layercontext.c:38-39`.
            spatial_layer_id: 0,
            temporal_layer_id: 0,
            number_spatial_layers: 0,
            number_temporal_layers: 0,
            spatial_layer_to_encode: 0,
            use_svc: false,
            pass: 0,
            layer_context: Vec::new(),
            temporal_layering_mode: TemporalLayeringMode::NoLayering,
            ext_frame_flags: [0; MAX_VPX_LAYERS],
            // libvpx `vp9_svc_layercontext.c:70-72`:
            //   svc->lst_fb_idx[sl] = 0; gld = 1; alt = 2;
            // Per-spatial-layer init.
            lst_fb_idx: [0; MAX_VPX_LAYERS],
            gld_fb_idx: [1; MAX_VPX_LAYERS],
            alt_fb_idx: [2; MAX_VPX_LAYERS],
            // libvpx `vp9_svc_layercontext.c:40-58`.
            force_zero_mode_spatial_ref: 0,
            use_gf_temporal_ref: true,
            use_gf_temporal_ref_current_layer: false,
            current_superframe: 0,
            non_reference_frame: false,
            single_layer_svc: false,
            simulcast_mode: false,
            use_set_ref_frame_config: false,
            temporal_layer_id_per_spatial: [0; MAX_SPATIAL_LAYERS],
            reference_last: [0; MAX_SPATIAL_LAYERS],
            reference_golden: [0; MAX_SPATIAL_LAYERS],
            reference_altref: [0; MAX_SPATIAL_LAYERS],
            // libvpx `vp9_svc_layercontext.c:51`:
            // svc->framedrop_mode = CONSTRAINED_LAYER_DROP;
            framedrop_mode: SvcLayerDropMode::ConstrainedLayerDrop,
            // libvpx `vp9_svc_layercontext.c:50`:
            // svc->disable_inter_layer_pred = INTER_LAYER_PRED_ON;
            disable_inter_layer_pred: InterLayerPred::On,
            ext_refresh_frame_flags_pending: false,
            ext_refresh_last_frame: false,
            ext_refresh_golden_frame: false,
            ext_refresh_alt_ref_frame: false,
            ref_frame_flags: 0,
            current_lst_fb_idx: 0,
            current_gld_fb_idx: 0,
            current_alt_fb_idx: 0,
        }
    }

    /// Allocate and initialize the `layer_context` array.
    ///
    /// Mirrors the LAYER_CONTEXT alloc + init loop in libvpx
    /// `vp9_init_layer_context` (`vp9_svc_layercontext.c:105-181`)
    /// for the CBR / pass==0 path.
    ///
    /// # Panics
    /// Panics if `number_spatial_layers * number_temporal_layers >
    /// MAX_VPX_LAYERS` (the libvpx array bound). libvpx would silently
    /// trash adjacent memory; we'd rather fail loud.
    pub fn alloc_layers(
        &mut self,
        number_spatial_layers: i32,
        number_temporal_layers: i32,
        per_layer_rc_config: RateControlConfig,
        per_layer_target_bandwidth: &[i32],
    ) {
        assert!(number_spatial_layers >= 1, "need >= 1 spatial layer");
        assert!(number_temporal_layers >= 1, "need >= 1 temporal layer");
        let total = (number_spatial_layers as usize)
            .checked_mul(number_temporal_layers as usize)
            .expect("layer count overflow");
        assert!(
            total <= MAX_VPX_LAYERS,
            "sl*tl = {} exceeds VPX_MAX_LAYERS = {}",
            total,
            MAX_VPX_LAYERS
        );
        assert_eq!(
            per_layer_target_bandwidth.len(),
            total,
            "per_layer_target_bandwidth must have len = sl*tl"
        );

        self.number_spatial_layers = number_spatial_layers;
        self.number_temporal_layers = number_temporal_layers;
        self.layer_context.clear();
        self.layer_context.reserve_exact(total);
        for &bw in per_layer_target_bandwidth {
            self.layer_context
                .push(LayerContext::new(per_layer_rc_config, bw));
        }
    }

    /// Returns the flat layer index for the current (sl, tl).
    /// Mirrors the libvpx pattern
    /// `cpi->svc.spatial_layer_id * cpi->svc.number_temporal_layers +
    /// cpi->svc.temporal_layer_id` used throughout
    /// `vp9_svc_layercontext.c`
    /// (e.g. `vp9_svc_layercontext.c:298-301`).
    #[inline]
    #[must_use]
    pub fn current_layer_idx(&self) -> usize {
        layer_ids_to_idx(
            self.spatial_layer_id as usize,
            self.temporal_layer_id as usize,
            self.number_temporal_layers as usize,
        )
    }

    /// `is_one_pass_svc(cpi) := use_svc && pass == 0`. Mirrors libvpx
    /// `is_one_pass_svc` (`vp9_encoder.h:1411-1413`).
    #[inline]
    #[must_use]
    pub fn is_one_pass_svc(&self) -> bool {
        self.use_svc && self.pass == 0
    }

    // -----------------------------------------------------------------
    // Save / restore
    // -----------------------------------------------------------------
    //
    // libvpx `vp9_restore_layer_context` (`vp9_svc_layercontext.c:352-389`)
    // copies the layer's saved `rc` / `twopass` / `target_bandwidth` /
    // `alt_ref_source` INTO `cpi` (and conditionally overrides
    // `cpi->oxcf.speed` when `is_one_pass_svc && lc->speed > 0`).
    //
    // libvpx `vp9_save_layer_context` (`vp9_svc_layercontext.c:391-424`)
    // is the inverse: lc <- cpi.
    //
    // rvp9 has no `cpi` god-struct yet; encoder integration is later.
    // The scaffold version below works directly on the `RateControl`
    // and `target_bandwidth` of the current layer:
    //
    //   - `save_layer_context(into, &rc, target)` stashes the current
    //     stream-level RC into the current layer slot
    //     (libvpx lines 391-401).
    //   - `restore_layer_context(out_rc, out_target)` pulls the
    //     current layer slot's RC back out (libvpx lines 352-360).
    //
    // The cyclic-refresh swap-pointer dance (libvpx lines 376-388 /
    // 402-423) is intentionally omitted at this layer — see module
    // TODO list.

    /// Restore: pull the current layer's saved RC + target_bandwidth
    /// out. Returns `(rate_control, target_bandwidth)`. Mirrors the
    /// `cpi->rc = lc->rc; cpi->oxcf.target_bandwidth = lc->target_bandwidth;`
    /// block in libvpx `vp9_restore_layer_context`
    /// (`vp9_svc_layercontext.c:358-360`).
    ///
    /// The boxed `RateControl` is cloned out (libvpx assigns by value;
    /// we clone to avoid double-ownership). `frames_since_key` /
    /// `frames_to_key` preservation (libvpx lines 354-356, 370-374)
    /// is the caller's job — they have to be carried over from the
    /// pre-restore RC because they describe the stream, not the layer.
    #[must_use]
    pub fn restore_layer_context(&self) -> (RateControl, i32) {
        let lc = &self.layer_context[self.current_layer_idx()];
        ((*lc.rc).clone(), lc.target_bandwidth)
    }

    /// Save: stash the stream RC + target_bandwidth into the current
    /// layer slot. Mirrors the `lc->rc = cpi->rc;` /
    /// `lc->target_bandwidth = oxcf->target_bandwidth;` block in
    /// libvpx `vp9_save_layer_context`
    /// (`vp9_svc_layercontext.c:395-397`).
    ///
    /// `frame_qp` / `MBs` come from the encoder common state in
    /// libvpx (lines 399-400); pass them through as parameters.
    pub fn save_layer_context(
        &mut self,
        rc: &RateControl,
        target_bandwidth: i32,
        frame_qp: i32,
        mbs: i32,
    ) {
        let idx = self.current_layer_idx();
        let lc = &mut self.layer_context[idx];
        *lc.rc = rc.clone();
        lc.target_bandwidth = target_bandwidth;
        lc.frame_qp = frame_qp;
        lc.mbs = mbs;
    }

    // -----------------------------------------------------------------
    // Temporal-mode setters (W6.6)
    // -----------------------------------------------------------------

    /// Reset per-frame ref/refresh state to "clear" before a setter
    /// computes new values. Mirrors the
    /// `cpi->ext_refresh_last_frame = cpi->ext_refresh_golden_frame =
    /// cpi->ext_refresh_alt_ref_frame = 0;` line that opens each
    /// libvpx setter
    /// (e.g. `vp9_svc_layercontext.c:531`).
    fn clear_ext_refresh(&mut self) {
        self.ext_refresh_last_frame = false;
        self.ext_refresh_golden_frame = false;
        self.ext_refresh_alt_ref_frame = false;
    }

    /// 3-layer (0-2-1-2) temporal pattern.
    /// Mirrors libvpx `set_flags_and_fb_idx_for_temporal_mode3`
    /// (`vp9_svc_layercontext.c:517-621`).
    ///
    /// Drives:
    /// - `temporal_layer_id` from the base-tl0 layer's
    ///   `current_video_frame_in_layer & 3`:
    ///   `(n & 1) ? 2 : (n >> 1)`  →  [0, 2, 1, 2] for n = 0..4.
    /// - `ref_frame_flags`, `ext_refresh_*` flags, and
    ///   `current_{lst,gld,alt}_fb_idx` per libvpx lines 533-616.
    pub fn set_flags_and_fb_idx_for_temporal_mode3(&mut self) {
        // libvpx 520: spatial_id = svc.spatial_layer_id =
        //              svc.spatial_layer_to_encode;
        self.spatial_layer_id = self.spatial_layer_to_encode;
        let spatial_id = self.spatial_layer_id;

        // libvpx 521-526: read current_video_frame_in_layer from the
        // tl0 slot at this spatial layer
        //   = layer_context[spatial_id * number_temporal_layers]
        let tl0_idx = (spatial_id as usize) * (self.number_temporal_layers as usize);
        let frame_num_within_temporal_struct =
            (self.layer_context[tl0_idx].current_video_frame_in_layer % 4) as i32;

        // libvpx 527-530.
        let temporal_id = if (frame_num_within_temporal_struct & 1) != 0 {
            2
        } else {
            frame_num_within_temporal_struct >> 1
        };
        self.temporal_layer_id = temporal_id;

        self.clear_ext_refresh();
        let nsl = self.number_spatial_layers;
        let is_base_key = self.layer_context[temporal_id as usize].is_key_frame;

        if temporal_id == 0 {
            // libvpx 533-545.
            self.ext_refresh_frame_flags_pending = true;
            self.ext_refresh_last_frame = true;
            if spatial_id == 0 {
                self.ref_frame_flags = VP9_LAST_FLAG;
            } else if is_base_key {
                self.ref_frame_flags = VP9_LAST_FLAG;
                self.ext_refresh_last_frame = false;
                self.ext_refresh_golden_frame = true;
            } else {
                self.ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
            }
        } else if temporal_id == 1 {
            // libvpx 546-553.
            self.ext_refresh_frame_flags_pending = true;
            self.ext_refresh_alt_ref_frame = true;
            if spatial_id == 0 {
                self.ref_frame_flags = VP9_LAST_FLAG;
            } else {
                self.ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
            }
        } else {
            // temporal_id == 2 — the tl2 slots (libvpx 554-590).
            if frame_num_within_temporal_struct == 1 {
                // first tl2 picture, libvpx 555-571.
                if spatial_id == nsl - 1 {
                    self.ext_refresh_frame_flags_pending = true;
                    self.ref_frame_flags = if spatial_id == 0 {
                        VP9_LAST_FLAG
                    } else {
                        VP9_LAST_FLAG | VP9_GOLD_FLAG
                    };
                } else if spatial_id == 0 {
                    self.ext_refresh_frame_flags_pending = true;
                    self.ext_refresh_alt_ref_frame = true;
                    self.ref_frame_flags = VP9_LAST_FLAG;
                } else if spatial_id < nsl - 1 {
                    self.ext_refresh_frame_flags_pending = true;
                    self.ext_refresh_alt_ref_frame = true;
                    self.ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
                }
            } else {
                // second tl2 picture, libvpx 572-589.
                if spatial_id == nsl - 1 {
                    self.ext_refresh_frame_flags_pending = true;
                    self.ref_frame_flags = if spatial_id == 0 {
                        VP9_LAST_FLAG
                    } else {
                        VP9_LAST_FLAG | VP9_GOLD_FLAG
                    };
                } else if spatial_id == 0 {
                    self.ext_refresh_frame_flags_pending = true;
                    self.ref_frame_flags = VP9_LAST_FLAG;
                    self.ext_refresh_alt_ref_frame = true;
                } else {
                    self.ext_refresh_frame_flags_pending = true;
                    self.ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
                    self.ext_refresh_alt_ref_frame = true;
                }
            }
        }

        // Buffer-index assignment, libvpx 591-616.
        if temporal_id == 0 {
            self.current_lst_fb_idx = spatial_id;
            if spatial_id != 0 {
                if is_base_key {
                    self.current_lst_fb_idx = spatial_id - 1;
                    self.current_gld_fb_idx = spatial_id;
                } else {
                    self.current_gld_fb_idx = spatial_id - 1;
                }
            } else {
                self.current_gld_fb_idx = 0;
            }
            self.current_alt_fb_idx = 0;
        } else if temporal_id == 1 || frame_num_within_temporal_struct == 1 {
            self.current_lst_fb_idx = spatial_id;
            self.current_gld_fb_idx = nsl + spatial_id - 1;
            self.current_alt_fb_idx = nsl + spatial_id;
        } else {
            self.current_lst_fb_idx = nsl + spatial_id;
            self.current_gld_fb_idx = nsl + spatial_id - 1;
            self.current_alt_fb_idx = nsl + spatial_id;
        }

        // TODO(integration): libvpx lines 618-620 call
        // non_reference_frame_simulcast() and reset_fb_idx_unused().
        // Both are stream-level helpers that depend on lookahead and
        // frame-buffer pool wiring; not in scope for the scaffold.
    }

    /// 2-layer (0-1-0-1) temporal pattern.
    /// Mirrors libvpx `set_flags_and_fb_idx_for_temporal_mode2`
    /// (`vp9_svc_layercontext.c:626-684`).
    pub fn set_flags_and_fb_idx_for_temporal_mode2(&mut self) {
        // libvpx 628.
        self.spatial_layer_id = self.spatial_layer_to_encode;
        let spatial_id = self.spatial_layer_id;

        // libvpx 629-634: tl = current_video_frame_in_layer & 1
        //                 from the tl0 slot.
        let tl0_idx = (spatial_id as usize) * (self.number_temporal_layers as usize);
        let temporal_id = (self.layer_context[tl0_idx].current_video_frame_in_layer & 1) as i32;
        self.temporal_layer_id = temporal_id;

        self.clear_ext_refresh();
        let nsl = self.number_spatial_layers;
        let is_base_key = self.layer_context[temporal_id as usize].is_key_frame;

        if temporal_id == 0 {
            // libvpx 637-649.
            self.ext_refresh_frame_flags_pending = true;
            self.ext_refresh_last_frame = true;
            if spatial_id == 0 {
                self.ref_frame_flags = VP9_LAST_FLAG;
            } else if is_base_key {
                self.ref_frame_flags = VP9_LAST_FLAG;
                self.ext_refresh_last_frame = false;
                self.ext_refresh_golden_frame = true;
            } else {
                self.ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
            }
        } else if temporal_id == 1 {
            // libvpx 650-659.
            self.ext_refresh_frame_flags_pending = true;
            self.ext_refresh_alt_ref_frame = true;
            if spatial_id == 0 {
                self.ref_frame_flags = VP9_LAST_FLAG;
            } else {
                if spatial_id == nsl - 1 {
                    self.ext_refresh_alt_ref_frame = false;
                }
                self.ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
            }
        }

        // libvpx 662-679.
        if temporal_id == 0 {
            self.current_lst_fb_idx = spatial_id;
            if spatial_id != 0 {
                if is_base_key {
                    self.current_lst_fb_idx = spatial_id - 1;
                    self.current_gld_fb_idx = spatial_id;
                } else {
                    self.current_gld_fb_idx = spatial_id - 1;
                }
            } else {
                self.current_gld_fb_idx = 0;
            }
            self.current_alt_fb_idx = 0;
        } else if temporal_id == 1 {
            self.current_lst_fb_idx = spatial_id;
            self.current_gld_fb_idx = nsl + spatial_id - 1;
            self.current_alt_fb_idx = nsl + spatial_id;
        }

        // TODO(integration): libvpx 681-683 simulcast / reset_fb_idx
        // helpers, see mode3 comment.
    }

    /// No-temporal-layering pattern.
    /// Mirrors libvpx `set_flags_and_fb_idx_for_temporal_mode_noLayering`
    /// (`vp9_svc_layercontext.c:688-720`). `temporal_layer_id` is
    /// always 0 — every frame is a tl0 frame.
    pub fn set_flags_and_fb_idx_for_temporal_mode_no_layering(&mut self) {
        // libvpx 690-691.
        self.spatial_layer_id = self.spatial_layer_to_encode;
        let spatial_id = self.spatial_layer_id;
        // No layering ⇒ temporal_id is always 0 (libvpx implicitly
        // never sets it here; the function operates as if tl == 0).
        self.temporal_layer_id = 0;

        self.clear_ext_refresh();
        // libvpx 694-695.
        self.ext_refresh_frame_flags_pending = true;
        self.ext_refresh_last_frame = true;
        // libvpx 696-704.
        let is_base_key = self.layer_context[0].is_key_frame;
        if spatial_id == 0 {
            self.ref_frame_flags = VP9_LAST_FLAG;
        } else if is_base_key {
            self.ref_frame_flags = VP9_LAST_FLAG;
            self.ext_refresh_last_frame = false;
            self.ext_refresh_golden_frame = true;
        } else {
            self.ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
        }

        // libvpx 705-715.
        self.current_lst_fb_idx = spatial_id;
        if spatial_id != 0 {
            if is_base_key {
                self.current_lst_fb_idx = spatial_id - 1;
                self.current_gld_fb_idx = spatial_id;
            } else {
                self.current_gld_fb_idx = spatial_id - 1;
            }
        } else {
            self.current_gld_fb_idx = 0;
        }

        // TODO(integration): libvpx 717-719 simulcast / reset_fb_idx
        // helpers, see mode3 comment.
    }

    /// Bypass mode: the application supplied an explicit ref-frame
    /// config per spatial layer via `set_ref_frame_config`.
    /// Mirrors libvpx
    /// `set_flags_and_fb_idx_bypass_via_set_ref_frame_config`
    /// (`vp9_svc_layercontext.c:722-738`).
    pub fn set_flags_and_fb_idx_bypass_via_set_ref_frame_config(&mut self) {
        // libvpx 725-726.
        self.spatial_layer_id = self.spatial_layer_to_encode;
        let sl = self.spatial_layer_id as usize;
        self.temporal_layer_id = self.temporal_layer_id_per_spatial[sl];

        // libvpx 727.
        self.ext_refresh_frame_flags_pending = true;
        // libvpx 728-730.
        self.current_lst_fb_idx = self.lst_fb_idx[sl];
        self.current_gld_fb_idx = self.gld_fb_idx[sl];
        self.current_alt_fb_idx = self.alt_fb_idx[sl];
        // libvpx 731-733.
        self.ext_refresh_last_frame = false;
        self.ext_refresh_golden_frame = false;
        self.ext_refresh_alt_ref_frame = false;
        // libvpx 734-737.
        self.ref_frame_flags = 0;
        if self.reference_last[sl] != 0 {
            self.ref_frame_flags |= VP9_LAST_FLAG;
        }
        if self.reference_golden[sl] != 0 {
            self.ref_frame_flags |= VP9_GOLD_FLAG;
        }
        if self.reference_altref[sl] != 0 {
            self.ref_frame_flags |= VP9_ALT_FLAG;
        }
    }

    /// Dispatch by `temporal_layering_mode`. Mirrors the dispatch
    /// block in libvpx `vp9_one_pass_svc_start_layer`
    /// (`vp9_svc_layercontext.c:819-830`).
    pub fn apply_temporal_layering(&mut self) {
        match self.temporal_layering_mode {
            TemporalLayeringMode::Pattern0212 => {
                self.set_flags_and_fb_idx_for_temporal_mode3();
            }
            TemporalLayeringMode::NoLayering => {
                self.set_flags_and_fb_idx_for_temporal_mode_no_layering();
            }
            TemporalLayeringMode::Pattern0101 => {
                self.set_flags_and_fb_idx_for_temporal_mode2();
            }
            TemporalLayeringMode::Bypass => {
                if self.use_set_ref_frame_config {
                    self.set_flags_and_fb_idx_bypass_via_set_ref_frame_config();
                }
                // libvpx silently does nothing if Bypass &&
                // !use_set_ref_frame_config (lines 826-830 guard).
            }
        }
    }
}

// =====================================================================
// Tests (PROOF, libvpx-cited)
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_rc_config() -> RateControlConfig {
        // Tiny but valid: any positive setup works for unit tests.
        // We mirror the lib's default and shrink it.
        RateControlConfig {
            frame_width: 320,
            frame_height: 240,
            target_bitrate_bps: 200_000,
            ..Default::default()
        }
    }

    // ---------------- W6.5 ----------------

    /// PROOF: `Svc::alloc_layers` populates `layer_context` with
    /// `number_spatial_layers * number_temporal_layers` slots, each
    /// carrying the supplied per-layer target_bandwidth. Mirrors
    /// libvpx `vp9_init_layer_context` loop bounds
    /// (`vp9_svc_layercontext.c:105-181`).
    #[test]
    fn w6_5_svc_alloc_creates_layers() {
        let mut svc = Svc::new();
        let rc_cfg = small_rc_config();
        // 3 spatial × 3 temporal = 9 slots (≤ MAX_VPX_LAYERS=12).
        let targets: Vec<i32> = (0..9).map(|i| 100_000 + 1_000 * i).collect();
        svc.alloc_layers(3, 3, rc_cfg, &targets);
        assert_eq!(svc.number_spatial_layers, 3);
        assert_eq!(svc.number_temporal_layers, 3);
        assert_eq!(svc.layer_context.len(), 9);
        for (i, lc) in svc.layer_context.iter().enumerate() {
            assert_eq!(
                lc.target_bandwidth, targets[i],
                "layer {i} target_bandwidth mismatch"
            );
        }
    }

    /// PROOF: `layer_ids_to_idx` matches libvpx macro
    /// `LAYER_IDS_TO_IDX(sl, tl, num_tl) ((sl) * (num_tl) + (tl))`
    /// (`vp9_encoder.h:1495`). Spot-checks several (sl, tl, num_tl)
    /// triples against the closed-form `sl * num_tl + tl`.
    #[test]
    fn w6_5_layer_ids_to_idx_matches_libvpx() {
        // libvpx: ((sl) * (num_tl) + (tl)).
        assert_eq!(layer_ids_to_idx(0, 0, 3), 0);
        assert_eq!(layer_ids_to_idx(0, 2, 3), 2);
        assert_eq!(layer_ids_to_idx(1, 0, 3), 3);
        assert_eq!(layer_ids_to_idx(2, 2, 3), 8);
        // 2-temporal-layer case.
        assert_eq!(layer_ids_to_idx(0, 0, 2), 0);
        assert_eq!(layer_ids_to_idx(0, 1, 2), 1);
        assert_eq!(layer_ids_to_idx(3, 1, 2), 7);
        // single-layer (degenerate).
        assert_eq!(layer_ids_to_idx(4, 0, 1), 4);
    }

    /// PROOF: `is_one_pass_svc` truth-table. Mirrors libvpx
    /// `is_one_pass_svc` (`vp9_encoder.h:1411-1413`):
    /// `cpi->use_svc && cpi->oxcf.pass == 0`.
    #[test]
    fn w6_5_is_one_pass_svc_predicate() {
        let mut svc = Svc::new();
        // (use_svc=false, pass=0) → false.
        svc.use_svc = false;
        svc.pass = 0;
        assert!(!svc.is_one_pass_svc());
        // (use_svc=true, pass=0) → true.
        svc.use_svc = true;
        svc.pass = 0;
        assert!(svc.is_one_pass_svc());
        // (use_svc=true, pass=1) → false.
        svc.pass = 1;
        assert!(!svc.is_one_pass_svc());
        // (use_svc=true, pass=2) → false.
        svc.pass = 2;
        assert!(!svc.is_one_pass_svc());
        // (use_svc=false, pass=1) → false.
        svc.use_svc = false;
        svc.pass = 1;
        assert!(!svc.is_one_pass_svc());
    }

    /// PROOF: 1-spatial × 3-temporal 0-2-1-2 setter produces the
    /// libvpx-spec pattern over four consecutive base-tl0 frames.
    /// Mirrors libvpx
    /// `set_flags_and_fb_idx_for_temporal_mode3:527-530`:
    ///   tl = (n & 1) ? 2 : (n >> 1)
    /// where `n = current_video_frame_in_layer % 4`.
    #[test]
    fn w6_6_temporal_setter_0_2_1_2_three_layers() {
        let mut svc = Svc::new();
        let rc_cfg = small_rc_config();
        // Single spatial × 3 temporal — minimal sl=0 / tl in {0,1,2}.
        svc.alloc_layers(1, 3, rc_cfg, &[100_000, 50_000, 25_000]);
        svc.temporal_layering_mode = TemporalLayeringMode::Pattern0212;
        svc.spatial_layer_to_encode = 0;

        // Step the base-tl0 layer's frame counter and verify the
        // computed temporal_layer_id matches the libvpx pattern
        // [0, 2, 1, 2]. The setter reads
        // layer_context[sl * num_temporal_layers].current_video_frame_in_layer.
        let expected = [0i32, 2, 1, 2];
        for (n, &want_tl) in expected.iter().enumerate() {
            svc.layer_context[0].current_video_frame_in_layer = n as u32;
            svc.apply_temporal_layering();
            assert_eq!(
                svc.temporal_layer_id, want_tl,
                "0-2-1-2 pattern step {n}: got tl={}, want tl={}",
                svc.temporal_layer_id, want_tl
            );
            // ext_refresh_frame_flags_pending must be set for every
            // frame in the pattern (libvpx 534, 547, plus all tl2 arms
            // for the single-spatial / spatial_id==nsl-1 case).
            assert!(
                svc.ext_refresh_frame_flags_pending,
                "step {n}: pending refresh flag not raised"
            );
        }
    }

    /// PROOF: no-layering setter pins `temporal_layer_id` to 0 for
    /// every frame and sets `ref_frame_flags = VP9_LAST_FLAG` at
    /// spatial_id == 0 (libvpx
    /// `set_flags_and_fb_idx_for_temporal_mode_noLayering:696-697`).
    #[test]
    fn w6_6_temporal_setter_no_layering_yields_layer_zero() {
        let mut svc = Svc::new();
        let rc_cfg = small_rc_config();
        // 1 spatial, 1 temporal — the no-layering canonical config.
        svc.alloc_layers(1, 1, rc_cfg, &[150_000]);
        svc.temporal_layering_mode = TemporalLayeringMode::NoLayering;
        svc.spatial_layer_to_encode = 0;

        for n in 0..10u32 {
            // Even if we lie about the base counter — no-layering
            // ignores it and always returns tl=0.
            svc.layer_context[0].current_video_frame_in_layer = n;
            svc.apply_temporal_layering();
            assert_eq!(svc.temporal_layer_id, 0, "frame {n}: tl must stay at 0");
            assert!(svc.ext_refresh_frame_flags_pending);
            assert!(svc.ext_refresh_last_frame);
            assert_eq!(
                svc.ref_frame_flags, VP9_LAST_FLAG,
                "frame {n}: sl=0 path must set LAST_FLAG only"
            );
        }
    }

    // ---------------- Extra PROOFs (above the ≥5 minimum) ----------------

    /// PROOF: 0-1-0-1 setter alternates tl ∈ {0, 1} over 4 frames.
    /// Mirrors libvpx
    /// `set_flags_and_fb_idx_for_temporal_mode2:629-634`:
    ///   tl = current_video_frame_in_layer & 1.
    #[test]
    fn w6_6_temporal_setter_0_1_0_1_two_layers() {
        let mut svc = Svc::new();
        let rc_cfg = small_rc_config();
        // 1 spatial × 2 temporal.
        svc.alloc_layers(1, 2, rc_cfg, &[120_000, 60_000]);
        svc.temporal_layering_mode = TemporalLayeringMode::Pattern0101;
        svc.spatial_layer_to_encode = 0;

        let expected = [0i32, 1, 0, 1];
        for (n, &want_tl) in expected.iter().enumerate() {
            svc.layer_context[0].current_video_frame_in_layer = n as u32;
            svc.apply_temporal_layering();
            assert_eq!(svc.temporal_layer_id, want_tl, "0-1-0-1 step {n}");
        }
    }

    /// PROOF: save → restore round-trips RateControl and target
    /// bandwidth. Mirrors libvpx `vp9_save_layer_context` →
    /// `vp9_restore_layer_context` round-trip
    /// (`vp9_svc_layercontext.c:391-424` then 352-389).
    #[test]
    fn w6_5_save_restore_layer_context_roundtrip() {
        let mut svc = Svc::new();
        let rc_cfg = small_rc_config();
        svc.alloc_layers(2, 2, rc_cfg, &[100_000, 50_000, 80_000, 40_000]);
        svc.spatial_layer_id = 1;
        svc.temporal_layer_id = 1;
        // Sanity: current_layer_idx == sl * num_tl + tl == 3.
        assert_eq!(svc.current_layer_idx(), 3);

        // Construct an RC and shove a distinctive value into it so we
        // can verify the round-trip.
        let mut rc = RateControl::new(rc_cfg);
        rc.last_picked_qindex = 137;
        rc.buffer_level = 12_345;
        rc.total_actual_bits = 67_890;

        svc.save_layer_context(
            &rc, /*target=*/ 87_654, /*qp=*/ 100, /*mbs=*/ 50,
        );
        // Mutate the original RC — confirm restore returns the *saved*
        // snapshot, not the post-mutation state.
        rc.last_picked_qindex = 0;
        rc.buffer_level = 0;
        let (restored_rc, restored_target) = svc.restore_layer_context();
        assert_eq!(restored_target, 87_654);
        assert_eq!(restored_rc.last_picked_qindex, 137);
        assert_eq!(restored_rc.buffer_level, 12_345);
        assert_eq!(restored_rc.total_actual_bits, 67_890);
        // frame_qp / mbs land on the LayerContext, not the RC.
        assert_eq!(svc.layer_context[3].frame_qp, 100);
        assert_eq!(svc.layer_context[3].mbs, 50);
    }

    /// PROOF: bypass setter honours `temporal_layer_id_per_spatial`
    /// and the per-spatial `reference_*` masks. Mirrors libvpx
    /// `set_flags_and_fb_idx_bypass_via_set_ref_frame_config:722-738`.
    #[test]
    fn w6_6_temporal_setter_bypass_uses_app_overrides() {
        let mut svc = Svc::new();
        let rc_cfg = small_rc_config();
        svc.alloc_layers(2, 2, rc_cfg, &[100_000, 50_000, 80_000, 40_000]);
        svc.temporal_layering_mode = TemporalLayeringMode::Bypass;
        svc.use_set_ref_frame_config = true;
        svc.spatial_layer_to_encode = 1;

        // App says: at sl=1, tl=1, reference LAST + ALT only.
        svc.temporal_layer_id_per_spatial[1] = 1;
        svc.lst_fb_idx[1] = 4;
        svc.gld_fb_idx[1] = 5;
        svc.alt_fb_idx[1] = 6;
        svc.reference_last[1] = 1;
        svc.reference_golden[1] = 0;
        svc.reference_altref[1] = 1;

        svc.apply_temporal_layering();

        assert_eq!(svc.spatial_layer_id, 1);
        assert_eq!(svc.temporal_layer_id, 1);
        assert!(svc.ext_refresh_frame_flags_pending);
        // libvpx 731-733: bypass clears all ext_refresh_* flags.
        assert!(!svc.ext_refresh_last_frame);
        assert!(!svc.ext_refresh_golden_frame);
        assert!(!svc.ext_refresh_alt_ref_frame);
        assert_eq!(svc.current_lst_fb_idx, 4);
        assert_eq!(svc.current_gld_fb_idx, 5);
        assert_eq!(svc.current_alt_fb_idx, 6);
        // ref_frame_flags built from reference_last / reference_altref.
        assert_eq!(svc.ref_frame_flags, VP9_LAST_FLAG | VP9_ALT_FLAG);
    }

    /// PROOF: constants match libvpx values verbatim. Anchors the
    /// scaffold against `vpx/vpx_encoder.h:42-48` /
    /// `vp9_onyxc_int.h:38`.
    #[test]
    fn w6_5_constants_match_libvpx() {
        // libvpx `vpx/vpx_encoder.h:42`: VPX_TS_MAX_LAYERS = 5.
        assert_eq!(MAX_TEMPORAL_LAYERS, 5);
        // libvpx `vpx/vpx_encoder.h:45`: VPX_MAX_LAYERS = 12.
        assert_eq!(MAX_VPX_LAYERS, 12);
        // libvpx `vpx/vpx_encoder.h:48`: VPX_SS_MAX_LAYERS = 5.
        assert_eq!(MAX_SPATIAL_LAYERS, 5);
        // libvpx `vp9_onyxc_int.h:38`: REF_FRAMES = 1 << 3 = 8.
        assert_eq!(REF_FRAMES, 8);
    }
}
