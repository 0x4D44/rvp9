# LLD: mode-decision

**Stage 2 LLD.** Owner of the non-RD pick-mode path in the rvp9-2
encoder: the 30-iteration ref × mode loop, mode-set selection,
`block_yrd` fast-path bypass, NEWMV diff-bias × encode-breakout
ordering, filter-ref selection, sub-8×8 mode picking, intra-fallback
gate, and the Laplacian R-D model glue. Primary specialist source
`notes/vp9-analysis/02-encoder-core.md` §2-§5 (A2). Secondary sources
`03-motion-estimation.md` §3.1-§3.4 (A3) and `04-bitstream-entropy.md`
§5.11.0 (A4 — `vp9_prob_cost[256]`).

**Date:** 2026-05-16.
**Reference implementation:** `third_party/libvpx/` at v1.14.1 (commit
`12f3a2a`). All `third_party/libvpx/<file>:<line>` citations resolve
there.

**Scope guardrails (PLAN.md):** 8-bit only (D-01 RESOLVED), no altref
(D-13 RESOLVED), AQ-3 / cyclic-refresh only (D-05 RESOLVED), EXT_RATECTRL
in v1 (R-06 RESOLVED).

---

## 1. Purpose

`rvp9-encoder::mode_decision` owns the per-superblock-leaf decision
"what (ref_frame, prediction_mode, mv, interp_filter, tx_size, skip_txfm,
intra_mode) tuple do we encode this block as?" at cpu-used 6-8. It
consumes inputs from rate-control (`base_qindex`, segment Q-deltas via
AQ-3), motion-estimation (the actual MV search via
`cpi->find_fractional_mv_step` and `vp9_full_pixel_search`), and the
partition module (`bsize`, partition geometry, `variance_low[25]` flags).
It produces the per-block `ModeInfo` that the bitstream writer
serializes.

The module is the **inner hot loop of the encoder** — every superblock
leaf goes through `pick_inter_mode` (or its sub-8×8 / intra-only
siblings). Production-grade real-time performance depends on this
module's speed even more than the partition module's; correctness
depends on byte-parity with libvpx for the 30-iteration cascade of
ref-frame × mode tests.

The module does **not** own motion search itself (that's
`rvp9-encoder::motion_est`), the variance-based partition decision
(`rvp9-encoder::partition`), the cost LUT construction
(`rvp9-encoder::mode_decision::cost_tables` is here, but the LUTs are
derived from `rvp9-core::entropy` defaults each frame), or the actual
bitstream serialization of the chosen mode (`rvp9-encoder::bitstream_writer`).

---

## 2. libvpx origin

| rvp9 item | libvpx origin |
|---|---|
| `pick_inter_mode` (the 30-iter loop) | `vp9/encoder/vp9_pickmode.c:1685-2742` (`vp9_pick_inter_mode`) |
| `pick_inter_mode_sub8x8` | `vp9/encoder/vp9_pickmode.c:2744-2992` (`vp9_pick_inter_mode_sub8x8`) |
| `pick_intra_mode` | `vp9/encoder/vp9_pickmode.c:1037-1086` (`vp9_pick_intra_mode`) (sub-8×8 intra only) and the intra branch inside `vp9_pick_inter_mode` at `vp9_pickmode.c:2516-2645` |
| Initial best-pickmode setup | `vp9/encoder/vp9_pickmode.c:1674-1696` (`init_best_pickmode`) |
| `BlockYrd` non-RD cost | `vp9/encoder/vp9_pickmode.c:718-844` (`block_yrd`) — A2 §4.3 |
| Laplacian R-D model | `vp9/encoder/vp9_rd.c:505-541` (`model_rd_norm`, `vp9_model_rd_from_var_lapndz`) — A2 §5.4 verbatim |
| `model_rd_for_sb_y` (small) | `vp9/encoder/vp9_pickmode.c:635-716` |
| `model_rd_for_sb_y_large` | `vp9/encoder/vp9_pickmode.c:429-633` |
| `model_rd_for_sb_uv` | `vp9/encoder/vp9_pickmode.c:846-914` |
| `encode_breakout_test` | `vp9/encoder/vp9_pickmode.c:932-1035` — A2 §4.9 |
| `vp9_NEWMV_diff_bias` | `vp9/encoder/vp9_pickmode.c:1298-1355` — A2 §4.8 |
| `search_filter_ref` | `vp9/encoder/vp9_pickmode.c:1488-1573` — L-48 |
| `calculate_tx_size` | `vp9/encoder/vp9_pickmode.c:353-384` — A2 §4.10 |
| `compute_intra_yprediction` | `vp9/encoder/vp9_pickmode.c:386-427` |
| `rd_less_than_thresh_row_mt`, `update_thresh_freq_fact[_row_mt]` | `vp9/encoder/vp9_pickmode.c:1112-1153` — A2 §4.6 |
| `mode_idx`, `intra_mode_list`, `pos_shift_16x16` | `vp9/encoder/vp9_pickmode.c:55-1098` (tables) |
| `ref_mode_set[12]`, `ref_mode_set_svc[8]` | `vp9/encoder/vp9_pickmode.c:1232-1253` — A2 §2.3 |
| `THR_MODES` enum | `vp9/encoder/vp9_rd.h:53-93` — A2 §5.5 |
| `find_predictors` | `vp9/encoder/vp9_pickmode.c:1255-1296` (calls into A3) |
| `mv_refs_rt` | `vp9/encoder/vp9_pickmode.c:59-152` (called from A3 ME but driven by mode-decision invariants) |
| `get_force_skip_low_temp_var` | `vp9/encoder/vp9_pickmode.c:1441-1486` |

Stage 1 specialist: **A2** (`notes/vp9-analysis/02-encoder-core.md`).
The 20 LLD-bound questions in A2 §7 are all dispositioned in §7 / §8 of
this LLD. Cross-references: A3 §3.1-§3.4 (the ME hand-off), A4 §5.11.0
(`vp9_prob_cost[256]` and cost macros — D-14 RESOLVED), A4 §5.1-§5.3
(`FRAME_CONTEXT` defaults used for cost LUT derivation).

---

## 3. Public API

### 3.1 Crate-public functions

The encoder's frame-loop (in `rvp9-encoder::frame_loop`) and the
partition module (in `rvp9-encoder::partition`) call into
`mode_decision`. The decoder never calls into this module.

```rust
/// Pick the best (ref, mode, mv, filter, tx_size, skip_txfm, intra_mode)
/// tuple for one inter-frame coding block at bsize >= BLOCK_8X8.
///
/// Mirrors `vp9_pick_inter_mode` (`vp9_pickmode.c:1685-2742`).
///
/// # Preconditions
/// - `block.bsize >= BlockSize::B8X8` (sub-8×8 uses
///   [`pick_inter_mode_sub8x8`]).
/// - `frame.frame_type != FrameType::Key`. Intra-only is dispatched
///   via [`pick_intra_mode_inter_frame`] or
///   [`pick_intra_mode_key_frame`] by the caller (which is
///   `partition::nonrd_pick_sb_modes`, mirroring
///   `vp9_encodeframe.c:4711`).
/// - `td.mb` plane source pointers point at the partition's source
///   pixels; `xd.plane[*].pre` will be re-pointed inside this function.
/// - `state` is the per-frame view: see §4.1 (`PickModeState`).
/// - `pc_ctx.pred_pixel_ready == 0` on entry; on exit, set to 1.
///
/// # Postconditions
/// - `td.mb.mi[0]` is populated with the chosen mode info.
/// - `pc_ctx` is updated with `rate`, `dist`, `rdcost`, `skip`,
///   `skip_txfm[0]`, `pred_pixel_ready=1`, `best_mode_index`,
///   `pred_mv[ref]`, `sum_y_eobs`, `skip_ref_frame_mask`.
/// - The returned `RdCost` is also written to `*out_rdc`.
/// - `td.mb.skip_txfm[0]` is `SkipTxfm::None | AcDc | AcOnly`.
/// - `td.mb.skip` is 0 or 1.
/// - On `td.mb.skip == 1`, the caller treats the block as non-residual.
///
/// # Panics
/// Debug-asserts:
/// - `block.bsize >= BlockSize::B8X8`
/// - `state.usable_ref_frame as u8 >= MvReferenceFrame::Last as u8`
///
/// # Unsafety
/// Internally uses `MacroBlock` interior raw-pointer rewiring of plane
/// buffers; safe at the API boundary. See §6 for the unsafe contract.
pub fn pick_inter_mode(
    state: &PickModeState<'_>,
    td: &mut ThreadData<'_>,
    block: BlockPosition,
    pc_ctx: &mut PickModeContext,
    out_rdc: &mut RdCost,
);

/// Pick a mode for one sub-8×8 inter block (BLOCK_4X4 / 4X8 / 8X4).
///
/// Mirrors `vp9_pick_inter_mode_sub8x8` (`vp9_pickmode.c:2744-2992`).
///
/// # Preconditions
/// - `block.bsize < BlockSize::B8X8`.
/// - `frame.frame_type != FrameType::Key`. (Key-frame sub-8×8 still
///   uses `pick_intra_mode_key_frame`.)
///
/// # Postconditions
/// As in `pick_inter_mode`, but with these specifics from A2 §3.8:
/// - `mi.tx_size == TxSize::Tx4X4`.
/// - `mi.uv_mode == IntraMode::Dc`.
/// - `mi.ref_frame[0] == best_ref_frame`, `mi.ref_frame[1] == None`.
/// - `mi.mode == bmi[3].as_mode` (the **last** sub-block's mode in
///   raster order — L-47).
/// - `mi.interp_filter == EIGHTTAP` (or `cm.interp_filter` if not
///   `SWITCHABLE`); no filter search.
/// - `td.mb.skip == 0`, `td.mb.skip_txfm[0] == SkipTxfm::None`
///   (no breakout / skip tracking in this path).
///
/// No intra check, no ALTREF, no compound, no denoiser, no
/// encode_breakout. See L-47 for the chosen-mode-from-bmi[3] semantics.
pub fn pick_inter_mode_sub8x8(
    state: &PickModeState<'_>,
    td: &mut ThreadData<'_>,
    block: BlockPosition,
    pc_ctx: &mut PickModeContext,
    out_rdc: &mut RdCost,
);

/// Pick an intra mode for a sub-8×8 block on a key frame, or for any
/// block when the caller has determined the frame is intra-only.
///
/// Mirrors `vp9_pick_intra_mode` (`vp9_pickmode.c:1037-1086`) and the
/// `hybrid_intra_mode_search` route at `vp9_encodeframe.c:4672`.
///
/// Tries DC_PRED, V_PRED, H_PRED, TM_PRED in order, applies the
/// `intra_y_mode_bsize_mask` gate from `SpeedFeatures` (A2 §5.6), and
/// picks the lowest-rdcost result. No early exit on small blocks at
/// cpu-used 6-8 (since `nonrd_keyframe = 1` is set per
/// `vp9_speed_features.c:585+` for non-SVC and SVC+enhancement-layer
/// paths).
///
/// # Preconditions
/// - `state.frame_is_intra_only || state.svc_enhancement_baseiskey`
/// - `block.bsize` is any BLOCK_4X4..BLOCK_64X64.
///
/// # Postconditions
/// - `mi.ref_frame[0] == MvReferenceFrame::Intra`, `[1] == None`.
/// - `mi.mode` ∈ {DC, V, H, TM}.
/// - `mi.tx_size` from `calculate_tx_size(is_intra=true)`.
pub fn pick_intra_mode_inter_frame(
    state: &PickModeState<'_>,
    td: &mut ThreadData<'_>,
    block: BlockPosition,
    pc_ctx: &mut PickModeContext,
    out_rdc: &mut RdCost,
);

/// Build cost LUTs once per frame on the main thread, before worker
/// dispatch. Reads `Common::fc` (the current `FrameContext`) and
/// populates `mode_decision::cost_tables::FrameCosts`.
///
/// Mirrors `vp9_initialize_rd_consts` (`vp9_rd.c:559-700` area).
///
/// # Preconditions
/// - Main thread only; no workers running.
/// - `common.fc` is the post-adaptation context for the previous frame
///   (or the default at frame 0 / KF reset).
///
/// # Postconditions
/// - `costs.inter_mode_cost[ctx][INTER_OFFSET]` populated.
/// - `costs.intra_uv_mode_cost`, `costs.mbmode_cost`,
///   `costs.partition_cost`, `costs.tx_size_cost`,
///   `costs.switchable_interp_costs`, `costs.skip_cost[ctx]` populated.
/// - `costs.intra_cost_penalty` set per
///   `vp9_get_intra_cost_penalty` (`vp9_rd.c:794-815`).
pub fn initialize_frame_costs(
    common: &Common,
    rd: &RdState,
    costs: &mut FrameCosts,
);
```

### 3.2 Internal module structure

```text
rvp9-encoder/src/mode_decision/
├── mod.rs                  pub fn pick_inter_mode, pick_inter_mode_sub8x8,
│                           pick_intra_mode_inter_frame,
│                           initialize_frame_costs
├── pick_inter.rs           the 30-iter ref×mode loop
├── pick_intra.rs           intra fallback inside inter pick + sub-8×8 intra
├── pick_sub8x8.rs          sub-8×8 inter
├── block_yrd.rs            Hadamard + SATD non-RD cost
├── model_rd.rs             Laplacian R-D model + model_rd_for_sb_*
├── encode_breakout.rs      encode_breakout_test
├── newmv_bias.rs           vp9_NEWMV_diff_bias
├── filter_ref.rs           search_filter_ref
├── intra_cost.rs           intra-mode cost penalty + intra_y_mode_bsize_mask
├── tx_size.rs              calculate_tx_size
├── rd_thresh.rs            rd_less_than_thresh, update_thresh_freq_fact[_row_mt]
├── cost_tables.rs          FrameCosts builder (cost LUT derivation)
├── ref_mode_set.rs         const tables (ref_mode_set, ref_mode_set_svc,
│                           mode_idx, intra_mode_list, pos_shift_16x16,
│                           THR_MODES)
├── pred_buffer.rs          PRED_BUFFER pool (4-entry) — L-50
├── laplacian.rs            rate_tab_q10, dist_tab_q10, xsq_iq_q10,
│                           model_rd_norm (A2 §5.4 verbatim)
└── types.rs                PickModeState, PickModeContext, BestPickMode,
                            RdCost, MvReferenceFrame, etc.
```

The split keeps each file <500 LoC, mirrors libvpx's call graph
(`vp9_pickmode.c` is one big file but logically segmented; the Rust
split exposes the structure that `vp9_pickmode.c` only hints at via
static helpers).

---

## 4. Types

### 4.1 `PickModeState` — frame-level immutable view

The bag of references a worker thread reads (never writes) during one
frame's encode. Constructed once per frame on the main thread, after
all speed-feature, RC, AQ, and scaled-source setup is complete (steps
22-30 in `00-overview.md` §2 timeline). Passed to every
`pick_inter_mode` call by reference.

```rust
/// Immutable view of all encoder state mode-decision needs during one
/// frame's encode. Constructed in `prepare_frame_for_workers` on the
/// main thread. Workers hold `&PickModeState<'frame>` for the duration
/// of the frame; no field is interior-mutable.
///
/// Lifetime `'frame`: tied to the `FrameScope<'frame>` borrow from
/// HLD-threading §3.2.
pub struct PickModeState<'frame> {
    // === Frame geometry & quant ===
    pub frame_type: FrameType,
    pub width: u32,
    pub height: u32,
    pub base_qindex: u8,                  // cm->base_qindex
    pub y_dc_delta_q: i8,
    pub y_dequant: &'frame [[i16; 2]; MAX_SEGMENTS], // [seg][0=dc, 1=ac]
    pub uv_dequant: &'frame [[i16; 2]; MAX_SEGMENTS],
    pub tx_mode: TxMode,
    pub interp_filter: InterpFilterSetting, // SWITCHABLE / fixed

    // === References ===
    pub ref_frame_flags: u8,              // VP9_LAST_FLAG | _GOLD_FLAG | _ALT_FLAG
    pub usable_ref_frame: MvReferenceFrame, // computed per A2 §4.2 init
    pub ref_buf: &'frame [Option<&'frame YV12Buffer>; 3], // [LAST-1, GOLDEN-1, ALTREF-1]
    pub ref_scaled: &'frame [bool; 3],    // any reference scaled?
    pub ref_sign_bias: &'frame [i8; 3],

    // === Speed features (D-10: snapshot, immutable during frame) ===
    pub sf: &'frame SpeedFeatures,

    // === Rate control & segmentation (read-only for workers) ===
    pub rc: &'frame RateControl,
    pub aq_mode: AqMode,                  // OFF or CyclicRefresh (D-05)
    pub seg: &'frame Segmentation,        // enabled, abs_delta, feature_data
    pub cyclic_refresh: &'frame CyclicRefreshSnapshot,

    // === Cost LUTs (built per frame in initialize_frame_costs) ===
    pub costs: &'frame FrameCosts,

    // === Variance / SAD function pointers (RTCD-dispatched once at init) ===
    pub fn_ptr: &'frame [VarianceFns; BLOCK_SIZES],

    // === SVC state (Option<...> => Some only when use_svc) ===
    pub svc: Option<&'frame SvcView>,

    // === Per-frame thresholds (set in vp9_set_variance_partition_thresholds) ===
    pub vbp_thresholds: &'frame [i64; 4],

    // === Variance-low partition flags (set by partition module) ===
    /// For the SB this block belongs to. Per `00-overview.md` §3.5,
    /// `variance_low[25]` lives in `MacroBlock` (workers own).
    /// Mode-decision *reads* it via `get_force_skip_low_temp_var`.
    pub variance_low_lookup: VarianceLowLookup,

    // === Noise estimate snapshot ===
    pub noise_level: NoiseLevel,          // kInvalid, kLow, kMedium, kHigh

    // === Source-content flags (set by avg_source_sad, partition module) ===
    pub content_state_sb: ContentStateSb, // CONTENT_STATE_SB enum

    // === Misc per-frame flags ===
    pub allow_comp_inter_inter: bool,
    pub use_prev_frame_mvs: bool,
    pub scene_change: bool,
    pub high_source_sad: bool,
    pub previous_frame_is_intra_only: bool,
    pub frames_since_golden: u32,
    pub frames_since_key: u32,
}
```

Notes:

- **Pixel = u8.** No `Pixel: PixelTrait` generics — D-01 RESOLVED in
  HLD-arch §7. Every plane buffer is `&[u8]`.
- **No raw `&Encoder`.** Subsystem-split per HLD-arch §3.6 means
  `PickModeState` is what the encoder's worker dispatch hands the
  workers; the workers never see `&Encoder` or `Common` directly.
- **No `ext_ratectrl` here.** EXT_RATECTRL Q-decisions enter via
  `RateControl::base_qindex` and per-segment Q-deltas — see §8 Q1 and
  HLD-svc §6 (the EXT_RATECTRL hook surface). EXT_RATECTRL never calls
  into mode-decision directly.

### 4.2 `BestPickMode` — per-block running best

Mirrors `BEST_PICKMODE` (`vp9_pickmode.c:44`, A2 §2.1).

```rust
/// Running best mode-pick for one block. Stack-allocated inside
/// `pick_inter_mode`; not exposed across module boundary.
#[derive(Clone)]
pub(super) struct BestPickMode {
    pub best_pred: Option<PredBufferHandle>, // None until first inter pick
    pub best_mode: PredictionMode,           // ZEROMV at init
    pub best_tx_size: TxSize,                // TxSizes (sentinel) at init
    pub best_intra_tx_size: TxSize,          // TxSizes sentinel
    pub best_ref_frame: MvReferenceFrame,    // Last
    pub best_second_ref_frame: MvReferenceFrame, // None
    pub best_mode_skip_txfm: SkipTxfm,       // SkipTxfm::None
    pub best_pred_filter: InterpFilter,      // EIGHTTAP
}

impl BestPickMode {
    pub(super) fn init() -> Self {
        // Mirrors `init_best_pickmode` at `vp9_pickmode.c:1674`.
        Self {
            best_pred: None,
            best_mode: PredictionMode::ZeroMv,
            best_tx_size: TxSize::Sentinel,
            best_intra_tx_size: TxSize::Sentinel,
            best_ref_frame: MvReferenceFrame::Last,
            best_second_ref_frame: MvReferenceFrame::None,
            best_mode_skip_txfm: SkipTxfm::None,
            best_pred_filter: InterpFilter::EightTap,
        }
    }
}
```

### 4.3 `PickModeContext` — what `partition` hands us

Mirrors `PICK_MODE_CONTEXT` (`vp9_context_tree.h:26-97`, A2 §2.8).
Owned by `partition::PcTree`; one node per partition-tree leaf.
Mode-decision reads `pred_pixel_ready` on entry and writes the
mode-decision results on exit. The fat coefficient buffers in this
struct are pre-allocated by the partition module (heap allocation on
worker init, not per-block).

```rust
/// Per-partition-leaf node in the `PcTree`. Owned by the partition
/// module; mode-decision mutates fields as documented.
pub struct PickModeContext {
    pub mi: ModeInfo,                       // libvpx `mic`
    pub mbmi_ext: MbModeInfoExt,
    // Coefficient buffers (heap, pre-allocated). Mode-decision writes
    // into these via `block_yrd` and intra-prediction quantize paths.
    pub coeff:   [[ScratchTranLow; 3]; MAX_MB_PLANE],
    pub qcoeff:  [[ScratchTranLow; 3]; MAX_MB_PLANE],
    pub dqcoeff: [[ScratchTranLow; 3]; MAX_MB_PLANE],
    pub eobs:    [[ScratchEobs;    3]; MAX_MB_PLANE],
    // Dual buffers (in-use vs best-in-store) — `_pbuf` in libvpx.
    pub coeff_pbuf:   [[ScratchTranLow; 3]; MAX_MB_PLANE],
    pub qcoeff_pbuf:  [[ScratchTranLow; 3]; MAX_MB_PLANE],
    pub dqcoeff_pbuf: [[ScratchTranLow; 3]; MAX_MB_PLANE],
    pub eobs_pbuf:    [[ScratchEobs;    3]; MAX_MB_PLANE],

    pub is_coded: bool,
    pub num_4x4_blk: u32,
    pub skip: bool,
    pub pred_pixel_ready: bool,
    pub skippable: bool,
    pub skip_txfm: [SkipTxfm; MAX_MB_PLANE * 4],

    pub best_mode_index: i32,
    pub hybrid_pred_diff: i32,
    pub comp_pred_diff: i32,
    pub single_pred_diff: i32,
    pub best_filter_diff: [i64; SWITCHABLE_FILTER_CONTEXTS],

    pub rate: i32,
    pub dist: i64,
    pub rdcost: i64,

    // Denoiser cross-talk (CONFIG_VP9_TEMPORAL_DENOISING in libvpx;
    // gated by the `denoiser` cargo feature).
    #[cfg(feature = "denoiser")]
    pub newmv_sse: u32,
    #[cfg(feature = "denoiser")]
    pub zeromv_sse: u32,
    #[cfg(feature = "denoiser")]
    pub zeromv_lastref_sse: u32,
    #[cfg(feature = "denoiser")]
    pub best_sse_inter_mode: PredictionMode,
    #[cfg(feature = "denoiser")]
    pub best_sse_mv: IntMv,
    #[cfg(feature = "denoiser")]
    pub best_reference_frame: MvReferenceFrame,
    #[cfg(feature = "denoiser")]
    pub best_zeromv_reference_frame: MvReferenceFrame,
    #[cfg(feature = "denoiser")]
    pub sb_skip_denoising: bool,

    pub pred_mv: [MotionVector; MAX_REF_FRAMES],
    pub pred_interp_filter: InterpFilter,
    pub sum_y_eobs: i32,
    pub skip_ref_frame_mask: u8,
}
```

`Send` / `Sync`: `PickModeContext` is **not** `Sync` (scratch buffers
are exclusively mutated by the owning worker). It is `Send` (workers
can take it from the pool). Marker traits derived automatically.

### 4.4 `RefMode` and the const tables — L-55 resolved

The 12-entry `ref_mode_set[12]` and 8-entry `ref_mode_set_svc[8]` are
**const arrays** (not enums). The Rust port preserves the libvpx
ordering exactly (`vp9_pickmode.c:1232-1253`, A2 §2.3) because
`adaptive_rd_thresh` cost-decay depends on the iteration index.

```rust
/// One (reference frame, prediction mode) pair tested in the non-RD
/// inter-mode cascade. Mirrors libvpx `REF_MODE`.
#[derive(Clone, Copy)]
pub(super) struct RefMode {
    pub ref_frame: MvReferenceFrame,
    pub pred_mode: PredictionMode,
}

pub(super) const RT_INTER_MODES: usize = 12;
pub(super) const RT_INTER_MODES_SVC: usize = 8;

/// The 12-entry ref×mode cascade for non-SVC encoding.
/// `vp9_pickmode.c:1238-1245`. Iteration order is load-bearing —
/// adaptive_rd_thresh decay (§4.6) uses iteration index.
pub(super) const REF_MODE_SET: [RefMode; RT_INTER_MODES] = [
    RefMode { ref_frame: MvReferenceFrame::Last,   pred_mode: PredictionMode::ZeroMv },
    RefMode { ref_frame: MvReferenceFrame::Last,   pred_mode: PredictionMode::NearestMv },
    RefMode { ref_frame: MvReferenceFrame::Golden, pred_mode: PredictionMode::ZeroMv },
    RefMode { ref_frame: MvReferenceFrame::Last,   pred_mode: PredictionMode::NearMv },
    RefMode { ref_frame: MvReferenceFrame::Last,   pred_mode: PredictionMode::NewMv },
    RefMode { ref_frame: MvReferenceFrame::Golden, pred_mode: PredictionMode::NearestMv },
    RefMode { ref_frame: MvReferenceFrame::Golden, pred_mode: PredictionMode::NearMv },
    RefMode { ref_frame: MvReferenceFrame::Golden, pred_mode: PredictionMode::NewMv },
    RefMode { ref_frame: MvReferenceFrame::Altref, pred_mode: PredictionMode::ZeroMv },
    RefMode { ref_frame: MvReferenceFrame::Altref, pred_mode: PredictionMode::NearestMv },
    RefMode { ref_frame: MvReferenceFrame::Altref, pred_mode: PredictionMode::NearMv },
    RefMode { ref_frame: MvReferenceFrame::Altref, pred_mode: PredictionMode::NewMv },
];

/// The 8-entry cascade for SVC. `vp9_pickmode.c:1247-1252`.
pub(super) const REF_MODE_SET_SVC: [RefMode; RT_INTER_MODES_SVC] = [
    RefMode { ref_frame: MvReferenceFrame::Last,   pred_mode: PredictionMode::ZeroMv },
    RefMode { ref_frame: MvReferenceFrame::Last,   pred_mode: PredictionMode::NearestMv },
    RefMode { ref_frame: MvReferenceFrame::Last,   pred_mode: PredictionMode::NearMv },
    RefMode { ref_frame: MvReferenceFrame::Golden, pred_mode: PredictionMode::ZeroMv },
    RefMode { ref_frame: MvReferenceFrame::Golden, pred_mode: PredictionMode::NearestMv },
    RefMode { ref_frame: MvReferenceFrame::Golden, pred_mode: PredictionMode::NearMv },
    RefMode { ref_frame: MvReferenceFrame::Last,   pred_mode: PredictionMode::NewMv },
    RefMode { ref_frame: MvReferenceFrame::Golden, pred_mode: PredictionMode::NewMv },
];

/// Selects the active cascade slice. `vp9_pickmode.c:2060-2063`.
pub(super) fn active_ref_mode_set(use_svc: bool) -> &'static [RefMode] {
    if use_svc { &REF_MODE_SET_SVC } else { &REF_MODE_SET }
}
```

**No altref in v1 (D-13 RESOLVED).** The four `Altref` entries in
`REF_MODE_SET` (indices 8-11) are tested but the `usable_ref_frame`
computation (§4.1 init code in libvpx — `vp9_pickmode.c:1907-1929`,
A3 §3.4 step 1) sets `usable_ref_frame = GoldenFrame` for non-altref
configs, and the loop's first early-exit
(`if (ref_frame > usable_ref_frame) continue`,
`vp9_pickmode.c:2065-2066`) skips all altref iterations. The const
array still has the 12 entries because **the iteration index is used
for `mode_rd_thresh` indexing**; renumbering would change which entry
maps to which `THR_MODES` slot.

In our v1 build, `usable_ref_frame` is computed by the
`compute_usable_ref_frame` helper (§5.1 below); for non-SVC non-altref
it returns either `Last` (when `frames_since_golden == 0 &&
gf_temporal_ref` per A2 §4.2) or `Golden`. The 4 altref iterations
become "skip" continues at near-zero cost.

### 4.5 `mode_idx` table and `INTER_OFFSET`

```rust
/// Mirrors `mode_idx[MAX_REF_FRAMES][4]` at `vp9_pickmode.c:1088-1093`.
/// Maps (ref_frame, mode_column) -> THR_MODES slot. Used to index
/// `rd.threshes[seg][bsize][thr_mode]` and `freq_fact[bsize][thr_mode]`.
pub(super) const MODE_IDX: [[ThrMode; 4]; MAX_REF_FRAMES] = [
    // INTRA_FRAME row: [DC_PRED, V_PRED, H_PRED, TM_PRED]
    [ThrMode::Dc, ThrMode::VPred, ThrMode::HPred, ThrMode::Tm],
    // LAST_FRAME row: [NEARESTMV, NEARMV, ZEROMV, NEWMV]
    [ThrMode::NearestMv, ThrMode::NearMv, ThrMode::ZeroMv, ThrMode::NewMv],
    // GOLDEN_FRAME row
    [ThrMode::NearestG, ThrMode::NearG, ThrMode::ZeroG, ThrMode::NewG],
    // ALTREF_FRAME row
    [ThrMode::NearestA, ThrMode::NearA, ThrMode::ZeroA, ThrMode::NewA],
];

/// Mirrors `INTER_OFFSET(mode) = mode - NEARESTMV`
/// (`vp9_entropymode.h:27`). Used as a column index into MODE_IDX
/// for inter modes.
#[inline]
pub(super) const fn inter_offset(mode: PredictionMode) -> u8 {
    (mode as u8).wrapping_sub(PredictionMode::NearestMv as u8)
}

pub(super) const INTRA_MODE_LIST: [PredictionMode; 4] = [
    PredictionMode::Dc,
    PredictionMode::V,
    PredictionMode::H,
    PredictionMode::Tm,
];
```

`THR_MODES` is the verbatim 30-entry enum from A2 §5.5 / `vp9_rd.h:53-93`.

### 4.6 `RdCost` — running rate/distortion/cost triple

```rust
/// Running rate/distortion/cost triple. Mirrors libvpx `RD_COST` at
/// `vp9_blockd.h:43-47`.
#[derive(Clone, Copy, Default)]
pub struct RdCost {
    pub rate: i32,
    pub dist: i64,
    pub rdcost: i64,
}

impl RdCost {
    /// Mirrors RDCOST(rdmult, rddiv, rate, dist) macro at
    /// `vp9_rd.h:170-175`:
    /// `(((rdmult * rate) + (dist << 9)) + (rddiv >> 1)) / rddiv`
    /// We preserve the i32-promoted arithmetic exactly; see L-04 for
    /// the float-determinism caveat (does not apply to this integer
    /// computation but flagged for sibling code).
    #[inline]
    pub fn rdcost(rdmult: i32, rddiv: i32, rate: i32, dist: i64) -> i64 {
        // libvpx uses ROUND_POWER_OF_TWO((rate * rdmult), 9) for the
        // rate term + dist. rvp9-encoder::rd::rdcost owns the canonical
        // implementation; this is a re-export.
        crate::rd::rdcost(rdmult, rddiv, rate, dist)
    }
}
```

`rdcost` itself lives in `rvp9-encoder::rd::rdcost`; mode-decision
re-exports the macro for ergonomics.

### 4.7 `PickModeError` — never panics in real-time

In real-time encoding, every error path is **soft** (skip the
iteration, fall back to ZEROMV/LAST). We log via tracing, never
panic, never return. The `pick_inter_mode` API does not return
`Result` because there is no failure mode at this layer; allocation
failures are pre-empted by pool sizing at `Encoder::new`.

### 4.8 `BlockYrdOutcome` — fast-path bypass result (L-41 resolved)

```rust
/// What `block_yrd` produced. At speed >= 7 the fast-path bypass
/// returns `Bypass` instead of running the Hadamard pass. Caller
/// adapts: in Bypass, take rate/dist from the `model_rd_for_sb_y`
/// output (the caller already has them in `this_rdc`).
///
/// Mirrors the `if (use_simple_block_yrd && ...) { *sse = INT_MAX;
/// *skippable = 0; return; }` early return at
/// `vp9_pickmode.c:735-744`.
pub(super) enum BlockYrdOutcome {
    Computed { sse: i64, skippable: bool },
    Bypass,                          // sse=INT_MAX, skippable=false in libvpx
}
```

This is L-41 RESOLVED. The caller pattern:

```rust
let outcome = block_yrd::block_yrd(state, td, ...);
match outcome {
    BlockYrdOutcome::Bypass => {
        // sf.use_simple_block_yrd path. Caller does NOT pay the
        // cost_bit(skip,*) adjustment — model_rd already accounted.
        // Caller does NOT touch skip_txfm[0] beyond what model_rd set.
    }
    BlockYrdOutcome::Computed { sse, skippable } => {
        // Apply the SKIP_TXFM decision logic from A2 §4.2:
        // (vp9_pickmode.c:2347-2375)
        td.mb.skip_txfm[0] = if skippable {
            SkipTxfm::AcDc
        } else if RdCost::rdcost(_, _, this_rdc.rate, this_rdc.dist)
                  < RdCost::rdcost(_, _, 0, sse) {
            this_rdc.rate += cost_bit(skip_prob, 0);
            SkipTxfm::None
        } else {
            this_rdc.rate = cost_bit(skip_prob, 1);
            this_rdc.dist = sse;
            SkipTxfm::AcDc
        };
    }
}
```

### 4.9 `PredBufferHandle` and the 4-entry pool (L-50 resolved)

The non-RD path reuses prediction buffers across mode tests via a
stack-allocated 4-entry pool with manual in-use tracking
(`vp9_pickmode.c:1728-1733`, A2 §2.2).

```rust
/// Pool of 4 luma prediction buffers reused across mode tests within
/// one `pick_inter_mode` call. Allocated on the stack (boxed only on
/// debug builds for stack-pressure safety).
///
/// L-50 resolution: typed handle pattern, mirrors libvpx
/// in_use=1/0 with Drop. The 3 KB of pixel storage is statically
/// allocated as a `[u8; 3*64*64]` slab inside the pool.
pub(super) struct PredBufferPool {
    // tmp[0..2] each point into `pred_buf[0..64*64]`,
    // `pred_buf[64*64..2*64*64]`, `pred_buf[2*64*64..3*64*64]`.
    // tmp[3] aliases the actual destination plane (no own storage).
    pred_buf: [[u8; 64 * 64]; 3],          // 12 KB, aligned 16
    in_use: [bool; 4],
}

/// Lease handle for one of the 4 slots. Drop releases.
pub(super) struct PredBufferHandle<'pool> {
    pool: &'pool mut PredBufferPool,
    slot: usize,                            // 0..=3
}
```

For byte parity the API is restored to the libvpx flat
`get_pred_buffer(tmp, len)` / `free_pred_buffer(p)` semantics
internally; the `PredBufferHandle` wrapper is the public face inside
the module.

### 4.10 `FrameCosts` — cost LUTs

```rust
/// Per-frame derived cost LUTs. Built in `initialize_frame_costs` on
/// the main thread from `Common::fc` (the current `FrameContext`).
/// Workers read these via `&PickModeState::costs`.
///
/// Mirrors the cost arrays inside `VP9_COMP`:
/// `cpi->inter_mode_cost`, `cpi->mbmode_cost`, etc.
pub struct FrameCosts {
    pub inter_mode_cost: [[i32; INTER_MODES]; INTER_MODE_CONTEXTS],
    pub mbmode_cost: [i32; INTRA_MODES],
    pub intra_uv_mode_cost: [[i32; INTRA_MODES]; FRAME_TYPES],
    pub y_mode_costs: [[[i32; INTRA_MODES]; INTRA_MODES]; INTRA_MODES],
    pub switchable_interp_costs: [[i32; SWITCHABLE_FILTERS]; SWITCHABLE_FILTER_CONTEXTS],
    pub partition_cost: [[i32; PARTITION_TYPES]; PARTITION_CONTEXTS],
    pub tx_size_cost: [[[i32; TX_SIZES]; TX_SIZE_CONTEXTS]; TX_SIZES],
    pub skip_cost: [[i32; 2]; SKIP_CONTEXTS],
    pub ref_frame_cost: [i32; MAX_REF_FRAMES],
    pub intra_cost_penalty: i32,        // `vp9_get_intra_cost_penalty`
}
```

All entries are in 9-bit fixed-point (`VP9_PROB_COST_SHIFT = 9`,
`vp9_cost.h:24`). Computed by walking the tree-prob arrays through
`vp9_prob_cost[256]` (A4 §5.11.0 — D-14 RESOLVED). The `cost_tables`
submodule owns the builder.

### 4.11 Marker traits

| Type | Send | Sync | Notes |
|---|---|---|---|
| `PickModeState<'frame>` | Yes (all refs are `'frame`) | Yes (no interior mutability) | Workers read concurrently. |
| `PickModeContext` | Yes | No (scratch buffers owned by one worker) | Per-worker. |
| `BestPickMode` | Yes | No (stack-local) | Stack-local in `pick_inter_mode`. |
| `PredBufferPool` | Yes | No | Stack-local. |
| `FrameCosts` | Yes | Yes | Read-only after init. |
| `FrameCounts` (from `rvp9-core::entropy`) | Yes | No | Per-worker `Box<FrameCounts>` per HLD-threading §5.4. |

---

## 5. Internal algorithm

### 5.1 Top-level `pick_inter_mode` (A2 §3.7 / §4.2)

Pseudocode that maps directly to `vp9_pick_inter_mode`
(`vp9_pickmode.c:1685-2742`):

```text
fn pick_inter_mode(state, td, block, pc_ctx, out_rdc) {
    // === Init (vp9_pickmode.c:1697-1763) ===
    let mut best = BestPickMode::init();
    let mut best_rdc = RdCost { rate: INT_MAX, dist: INT_MAX, rdcost: INT64_MAX };
    let intra_cost_penalty = state.costs.intra_cost_penalty;
    let inter_mode_thresh = rdcost!(state.rdmult, state.rddiv,
                                    intra_cost_penalty, 0);
    // §4.2 — base_qindex source-variance override for screen-content
    if state.sf.short_circuit_flat_blocks || state.sf.limit_newmv_early_exit {
        td.mb.source_variance = sby_perpixel_variance(state, td, block);
        // Screen+CR-AQ stationary-skin demotion (vp9_pickmode.c:1714-1722)
        if state.is_screen && state.aq_mode == AqMode::CyclicRefresh
            && segment_id > 0 && td.mb.zero_temp_sad_source
            && td.mb.source_variance == 0
        {
            demote_segment_to_0(state, td);
            reinit_quantizers_for_segment(state, td, 0);
        }
    }

    // === usable_ref_frame computation (A2 §4.2 init / A3 §3.4) ===
    let mut usable = compute_usable_ref_frame(state, td, block);
    // NOTE: D-13 (no altref) — `usable` is at most GoldenFrame in v1.
    let mut comp_modes = 0;  // L-46: compound deferred to v2 (always 0)

    // === Force-skip low-temp-var mask (A2 §4.15) ===
    let force_skip_low_temp_var = if state.sf.short_circuit_low_temp_var > 0 {
        state.variance_low_lookup.lookup_for_block(block.mi_row, block.mi_col,
                                                   block.bsize)
    } else { false };

    // === Per-ref find_predictors loop (A2 §4.13 / A3 §3.4 step 6) ===
    let mut skip_ref_find_pred = [false; MAX_REF_FRAMES];
    for r in MvReferenceFrame::Last..=usable {
        if state.ref_frame_flags & ref_flag(r) == 0 {
            skip_ref_find_pred[r as usize] = true;
            // ref_frame_skip_mask |= ...
            continue;
        }
        find_predictors(state, td, block, r, force_skip_low_temp_var,
                        /* comp_pred_allowed */ comp_modes > 0);
        // Sets frame_mv[*][r], yv12_mb[r][*], pred_mv_sad[r],
        // mode_context[r]; calls A3's vp9_mv_pred for reference masking.
    }

    // === `large_block` and `use_model_yrd_large` (A2 §4.2 init) ===
    let large_block = compute_large_block(state, td, block);
    let use_model_yrd_large = state.rc.rc_mode == VPX_CBR
        && large_block
        && !state.cr_boosted_for_block(block)
        && state.base_qindex > 0;

    // === The 30-iteration main loop (A2 §4.2 main loop, §3.7) ===
    //
    // Iteration count breakdown for v1 (D-13: no altref):
    //   - num_inter_modes ∈ {12 (non-SVC), 8 (SVC)}.
    //   - comp_modes == 0 (L-46).
    //   So we iterate 0..num_inter_modes => 8 or 12 outer iterations.
    //   With `usable = GoldenFrame` (non-altref): 4 of 12 iterations
    //   skip immediately at `if (ref > usable) continue` (the altref
    //   entries 8..11).
    //   With `usable = LastFrame`: 8 of 12 skip immediately.
    //   With intra fallback (§4.4): up to 4 more iterations after.
    //
    // The "30 iterations" claim in A2 §1 is the upper bound counting
    // intra; L-40 RESOLVED: documented in §5.2 below.
    let mode_set = active_ref_mode_set(state.svc.is_some());
    let num_inter_modes = mode_set.len();
    let mut mode_checked: [[bool; MAX_REF_FRAMES]; MB_MODE_COUNT] = Default::default();
    let mut best_early_term = false;
    let mut best_sse_sofar: u32 = INT_MAX as u32;
    let mut sse_zeromv_normalized: u32 = INT_MAX as u32;

    let mut idx = 0;
    while idx < num_inter_modes + comp_modes {
        let rm = mode_set[idx];           // unpack (this_mode, ref_frame)
        let (this_mode, ref_frame) = (rm.pred_mode, rm.ref_frame);
        let (this_mode, ref_frame, second_ref, comp_pred) =
            if idx >= num_inter_modes {
                // Compound (always 0 in v1 per L-46).
                unreachable!("compound deferred to v2 (L-46)")
            } else {
                (this_mode, ref_frame, MvReferenceFrame::None, false)
            };

        // === Skip-rules cascade (A2 §4.2; about 30 continue points) ===
        if skip_iteration(state, td, &skip_ref_find_pred, usable, ref_frame,
                          this_mode, &best, force_skip_low_temp_var,
                          sse_zeromv_normalized, idx)
        {
            idx += 1;
            continue;
        }

        // === Per-iteration setup ===
        setup_ref_for_iteration(state, td, ref_frame, second_ref);

        // === Adaptive RD threshold (A2 §4.6, L-49) ===
        let thr_idx = MODE_IDX[ref_frame as usize][inter_offset(this_mode) as usize];
        let mode_rd_thresh = adaptive_rd_thresh(state, td, block, thr_idx, &best);
        if rd_less_than_thresh(best_rdc.rdcost, mode_rd_thresh,
                               freq_fact_for(state, td, block, thr_idx))
        {
            // If we'd skip but mv is non-zero, still continue.
            if td.mb.frame_mv[this_mode][ref_frame].as_int != 0 {
                idx += 1;
                continue;
            }
        }

        // === NEWMV search (A3 §3.1 step 3) ===
        let mut rate_mv = 0;
        let mut rd_computed = false;
        if this_mode == PredictionMode::NewMv && !force_mv_inter_layer(state, ref_frame) {
            let r = motion_est::search_new_mv(state, td, block, ref_frame, &mut rate_mv);
            if r == SearchNewMvResult::Skip {
                idx += 1;
                continue;
            }
        }

        // === Duplicate-MV skip (A2 §4.2 mid-loop) ===
        if let Some(other) = duplicate_mv_check(td, this_mode, ref_frame, &mode_checked) {
            idx += 1;
            continue;
        }

        // === Skip near-nearest (A2 §4.2: "if (mode != NEARESTMV ... mv==nearest_mv) continue") ===
        if this_mode != PredictionMode::NearestMv && !comp_pred {
            if td.mb.frame_mv[this_mode][ref_frame] ==
               td.mb.frame_mv[PredictionMode::NearestMv][ref_frame] {
                idx += 1;
                continue;
            }
        }

        // === Configure mi for this iteration ===
        td.mb.mi.mode = this_mode;
        td.mb.mi.mv[0] = td.mb.frame_mv[this_mode][ref_frame];
        td.mb.mi.mv[1] = MotionVector::ZERO;
        td.mb.mi.ref_frame[0] = ref_frame;
        td.mb.mi.ref_frame[1] = second_ref;

        // === Reuse-inter-pred buffer allocation (L-50) ===
        let mut this_mode_pred: Option<PredBufferHandle> = None;
        if state.sf.reuse_inter_pred_sby {
            let h = td.pred_buf_pool.acquire(/* asking_for=3? no, slot policy */);
            redirect_dst_plane_to_handle(td, &h);
            this_mode_pred = Some(h);
        }

        // === Filter-ref search (L-48) ===
        let mut this_early_term = false;
        let pred_filter_search = pick_pred_filter_search(state, td, ref_frame,
                                                         this_mode, force_mv_inter_layer);
        let mv_has_subpel = subpel_bits(td.mb.mi.mv[0]) != 0;
        if (this_mode == PredictionMode::NewMv
            || state.sf.cb_pred_filter_search > 0
            || filter_ref_is_switchable(state))
            && pred_filter_search
            && (ref_frame == MvReferenceFrame::Last
                || (ref_frame == MvReferenceFrame::Golden
                    && !force_mv_inter_layer(state, ref_frame)
                    && (state.svc.is_some() || state.rc.rc_mode == VPX_VBR)))
            && mv_has_subpel
        {
            rd_computed = true;
            filter_ref::search_filter_ref(state, td, block, &mut this_rdc,
                                          &mut this_mode_pred, &mut this_early_term,
                                          use_model_yrd_large);
            // Sets mi->interp_filter to best; possibly rebuilds pred.
        } else {
            td.mb.mi.interp_filter = pick_default_filter(state, ref_frame, this_mode);
            // SVC golden zero-mv override (filter_gf_svc):
            if let Some(svc) = state.svc {
                if ref_frame == MvReferenceFrame::Golden
                    && td.mb.mi.mv[0].as_int == 0
                    && svc.filter_gf_svc.is_some()
                {
                    td.mb.mi.interp_filter = svc.filter_gf_svc.unwrap();
                }
            }
            inter_pred::build_inter_predictors_sby(state, td, block);
            if use_model_yrd_large {
                model_rd::for_sb_y_large(state, td, block, &mut this_rdc,
                                         &mut sse_y, &mut var_y, &mut this_early_term,
                                         &mut flag_preduv_computed);
            } else {
                model_rd::for_sb_y(state, td, block, /* is_intra= */ false,
                                   &mut this_rdc, &mut sse_y, &mut var_y);
            }
            // Track sse_zeromv_normalized for the golden-skip CBR heuristic.
            if ref_frame == MvReferenceFrame::Last
                && td.mb.mi.mv[0].as_int == 0
            {
                sse_zeromv_normalized = sse_y >> (log2_bw + log2_bh);
            }
            if (sse_y as u32) < best_sse_sofar {
                best_sse_sofar = sse_y as u32;
            }
        }

        // === block_yrd / skip_txfm decision (A2 §4.3, L-41) ===
        if !this_early_term {
            let outcome = block_yrd::block_yrd(state, td, block, &mut this_rdc,
                                               min(td.mb.mi.tx_size, TxSize::Tx16X16),
                                               rd_computed,
                                               /* is_intra= */ false);
            apply_block_yrd_outcome(state, td, &mut this_rdc, outcome);
            // Switchable filter rate (only when SWITCHABLE+sub-pel mv).
            if state.tx_mode_is_switchable() && mv_has_subpel {
                this_rdc.rate += switchable_interp_rate(state, td);
            }
        } else {
            // early_term branch — skip block_yrd entirely.
            if state.tx_mode_is_switchable() && mv_has_subpel {
                this_rdc.rate += switchable_interp_rate(state, td);
            }
            this_rdc.rate += cost_bit(state.costs.skip_cost[td.mb.skip_ctx], 1);
        }

        // === UV color check (A2 §4.12) ===
        if !this_early_term &&
           (td.mb.color_sensitivity[0] || td.mb.color_sensitivity[1])
        {
            if !flag_preduv_computed[0] {
                inter_pred::build_inter_predictors_uv(state, td, block);
            }
            let uv_rdc = model_rd::for_sb_uv(state, td, block);
            this_rdc.rate += uv_rdc.rate;
            this_rdc.dist += uv_rdc.dist;
        }

        // === Mode + ref costs ===
        this_rdc.rate += rate_mv;
        this_rdc.rate += state.costs.inter_mode_cost[mode_ctx[ref_frame]][inter_offset(this_mode)];
        this_rdc.rate += state.costs.ref_frame_cost[ref_frame as usize];
        this_rdc.rdcost = rdcost!(state.rdmult, state.rddiv,
                                  this_rdc.rate, this_rdc.dist);

        // === NEWMV diff-bias (A2 §4.8, L-43) — runs BEFORE encode_breakout
        //     ORDERING IS LOAD-BEARING; see L-43 / A2 §7 Q16 ===
        if state.rc.rc_mode == VPX_CBR && state.sf.speed >= 5
            && !state.is_screen()
        {
            newmv_bias::vp9_newmv_diff_bias(
                state.noise_level,
                td,
                this_mode,
                &mut this_rdc,
                block.bsize,
                td.mb.mi.mv[0].row,
                td.mb.mi.mv[0].col,
                /* ref_is_last= */ ref_frame == MvReferenceFrame::Last,
                td.mb.lowvar_highsumdiff,
                td.mb.sb_is_skin,
            );
            // This mutates this_rdc.rdcost in place.
        }

        // === encode_breakout test (A2 §4.9) ===
        if state.allow_encode_breakout(block.bsize) && !state.cm.lossless
            && !state.scene_change
            && !state.svc_high_num_blocks_with_motion()
        {
            encode_breakout::encode_breakout_test(state, td, block, ref_frame,
                                                  this_mode, var_y, sse_y,
                                                  &mut td.mb.skip);
            if td.mb.skip {
                this_rdc.rate = state.costs.inter_mode_cost[mode_ctx[ref_frame]][inter_offset(this_mode)];
                this_rdc.dist = (sse_y as i64) << 4;
                this_rdc.rdcost = rdcost!(state.rdmult, state.rddiv,
                                          this_rdc.rate, this_rdc.dist);
            }
        }

        // === Screen-content stationary penalty (vp9_pickmode.c:2447-2454) ===
        if state.is_screen()
            && (state.scene_change || state.high_motion())
            && ref_frame == MvReferenceFrame::Last
            && td.mb.mi.mv[0].as_int == 0
            && state.svc_spatial_layer_id() == 0
            && td.mb.source_variance == 0
            && sse_y > 0
        {
            this_rdc.rdcost = this_rdc.rdcost.saturating_mul(4);
        }

        // === Denoiser stats hook (only with `denoiser` feature) ===
        #[cfg(feature = "denoiser")]
        denoiser_hook::update_denoiser_stats(state, td, ref_frame, this_mode,
                                             sse_y, &mut this_rdc, &mut zero_last_cost_orig);

        // === Update best ===
        mode_checked[this_mode][ref_frame] = true;
        if this_rdc.rdcost < best_rdc.rdcost || td.mb.skip {
            best_rdc = this_rdc;
            best_early_term = this_early_term;
            best = BestPickMode {
                best_pred: this_mode_pred.take(),  // transfer ownership
                best_mode: this_mode,
                best_tx_size: td.mb.mi.tx_size,
                best_intra_tx_size: best.best_intra_tx_size,
                best_ref_frame: ref_frame,
                best_second_ref_frame: second_ref,
                best_mode_skip_txfm: td.mb.skip_txfm[0],
                best_pred_filter: td.mb.mi.interp_filter,
            };
            // Swap pred buffer if better.
        } else {
            // Release this_mode_pred back to pool.
            drop(this_mode_pred);
        }

        // === Early-exit conditions ===
        if td.mb.skip
            && (!state.force_test_gf_zeromv()
                || mode_checked[PredictionMode::ZeroMv][MvReferenceFrame::Golden])
        {
            break;
        }
        if best_early_term && idx > 0 && !state.scene_change
            && (!state.force_test_gf_zeromv()
                || mode_checked[PredictionMode::ZeroMv][MvReferenceFrame::Golden])
        {
            td.mb.skip = true;
            break;
        }

        idx += 1;
    }  // end main mode loop

    // === Commit best inter pick ===
    td.mb.mi.mode = best.best_mode;
    td.mb.mi.tx_size = best.best_tx_size;
    td.mb.mi.ref_frame = [best.best_ref_frame, best.best_second_ref_frame];
    td.mb.mi.interp_filter = best.best_pred_filter;
    td.mb.skip_txfm[0] = best.best_mode_skip_txfm;

    // === Intra-search gate (§5.4) ===
    if intra_gate_open(state, td, block, &best, &best_rdc, inter_mode_thresh) {
        pick_intra::search_intra_inside_inter(state, td, block, &mut best, &mut best_rdc);
    }

    // === Finalise (vp9_pickmode.c:2647-2740) ===
    finalize_mode_info(td, block, &best);

    // === Buffer copy: best inter pred -> orig_dst ===
    if state.sf.reuse_inter_pred_sby && best.best_ref_frame != MvReferenceFrame::Intra {
        if let Some(p) = best.best_pred.as_ref() {
            if p.data_ptr() != orig_dst.buf {
                vpx_convolve_copy(p.data_ptr(), p.stride(),
                                  orig_dst.buf, orig_dst.stride,
                                  /* filter= */ None, 0, 0, 0, 0,
                                  bw(block.bsize), bh(block.bsize));
            }
        }
    }

    // === Denoiser per-block (only with feature) ===
    #[cfg(feature = "denoiser")]
    denoiser_hook::denoise_block_and_maybe_recheck_zeromv(state, td, block, &mut best, &mut best_rdc);

    // === Update ARF / lastgolden usage counters ===
    update_ref_usage_counters(state, td, block, &best);

    // === Update adaptive RD threshold freq-fact (A2 §4.6) ===
    if state.sf.adaptive_rd_thresh > 0 {
        update_thresh_freq_fact(state, td, block,
                                best.best_mode, best.best_ref_frame);
    }

    *out_rdc = best_rdc;
    pc_ctx.pred_pixel_ready = true;
}
```

### 5.2 30-iteration upper bound — L-40 validation

A2 §1 claims "30-iteration ref×mode loop". The actual upper bound,
**confirmed by reading `vp9_pickmode.c:2039-2645`**:

| Phase | Iterations | Source |
|---|---|---|
| Inter modes (non-SVC) | up to 12 | `ref_mode_set[RT_INTER_MODES=12]` |
| Compound modes (v1: 0; libvpx: 0..=2) | 0 | L-46 RESOLVED: compound deferred to v2 |
| Intra modes (inside `pick_inter_mode`) | up to 4 | `intra_mode_list[4]` |
| **Total upper bound in v1** | **up to 16** | (non-SVC); SVC: up to 12 |

The "30" was a casual estimate in A2 §1 (counting all distinct
`continue` points, not iterations). **L-40 RESOLVED:** documented in
the LLD here; the actual loop is bounded by 16 in v1. The `idx <
num_inter_modes + comp_modes` while-condition (12 + 0 = 12) plus the
intra branch (4 more) gives 16 max. No multipliers anywhere.

This matters because the per-iteration cost is roughly: motion search
(NEWMV only, ~50 μs) → inter-pred build (~5 μs) → model_rd (~2 μs) →
block_yrd (~10 μs at speed ≤ 6, 0 μs at speed ≥ 7) → mode cost (~0.1
μs) → encode_breakout (~3 μs). Worst case 16 iterations × 70 μs ≈
1.1 ms per superblock; at 720p30 with 264 SBs that's 290 ms/frame —
way over budget. **Real-time only works because most iterations
early-exit at the skip-rules cascade.** A2's measured frame time of
~5 ms at 720p comes from average ~3 iterations per SB.

### 5.3 Skip-rules cascade

The cascade is mechanical — 30+ `continue` points in
`vp9_pickmode.c:2065-2236`. The Rust port preserves each one in
order; refactoring is a byte-parity hazard (A2 §7 Q3, L-40
disposition: **preserve C structure**).

The cascade in order (file:line refs into `vp9_pickmode.c`):

1. `ref_frame > usable_ref_frame` — line 2065-2066. Crosses 4 altref entries in v1.
2. `skip_ref_find_pred[ref_frame]` — 2067-2068.
3. `previous_frame_is_intra_only && (ref != LAST || mv != 0)` — 2070-2072.
4. `seg.SEG_LVL_REF_FRAME && seg_ref != ref_frame` — 2074-2077.
5. `flag_svc_subpel` enforce NEWMV-equals-svc-mv (SVC enhancement-layer subpel) — 2079-2100.
6. Compound checks (all 0 in v1, L-46) — 2102-2118.
7. `ref == GOLDEN && rc == CBR && sse_zeromv_normalized < thresh_skip_golden` — 2122-2126.
8. `!(ref_frame_flags & ref_flag(ref))` — 2127-2128.
9. Screen-content stationary skip — 2125-2137.
10. `!(sf.inter_mode_mask[bsize] & (1 << this_mode))` — 2139.
11. VBR-lag alt_ref_gf_group skip rules — 2141-2163. (L-76, v1: VBR-RT no altref → these dead.)
12. `const_motion[ref] && this_mode == NEARMV` — 2169-2170.
13. `force_skip_low_temp_var && ref == GOLDEN && mv != 0` — 2172-2174.
14. Low-temp-var + state != kVeryHighSad + (level 2 or 3) + ref==LAST + NEWMV — 2178-2184.
15. `svc && !force_mv_inter_layer && svc_force_zero_mode[ref] && mv != 0` — 2188-2191.
16. `sf.reference_masking && !(mv==0 && ref==LAST)` — 2197-2214.
17. `ref_frame_skip_mask & (1 << ref)` — 2218-2219.

Each gets a named helper in `pick_inter.rs::skip_rules` and a unit
test per rule. The Rust port keeps the **flat sequence of
`if … { continue; }` blocks** rather than refactoring to a
filter-chain (libvpx reads top-to-bottom; matching that makes diff
debugging linear).

### 5.4 Intra search inside `pick_inter_mode` (A2 §4.5)

After the inter loop, an intra search may run if the gate at
`vp9_pickmode.c:2516-2523` opens. The gate (one big OR):

```text
fn intra_gate_open(state, td, block, best, best_rdc, inter_mode_thresh) -> bool {
    if best_rdc.rdcost == INT64_MAX { return true; }
    if state.is_screen() && td.mb.source_variance == 0 { return true; }
    if state.scene_change && state.perform_intra_pred(block) { return true; }

    // Default path
    let force_skip_relax = !force_skip_low_temp_var
        || block.bsize < BlockSize::B32X32
        || state.content_state_sb == ContentStateSb::VeryHighSad;
    force_skip_relax
        && state.perform_intra_pred(block)
        && !td.mb.skip
        && best_rdc.rdcost > inter_mode_thresh
        && block.bsize <= state.sf.max_intra_bsize
        && !td.mb.skip_low_source_sad
        && !td.mb.lowvar_highsumdiff
}
```

`perform_intra_pred` is set to 0 if `svc_spatial_layer_id > 0 &&
!is_base_layer` or if alt-ref reference is in use
(`vp9_pickmode.c:2494-2512`). In v1 (no altref) the alt-ref test
collapses to a constant `false`, but we preserve the code shape for
when D-13 unblocks later.

The per-intra-mode body is the inner intra cascade (A2 §4.5):

```text
for mode in [Dc, V, H, Tm] {
    let mode_rd_thresh = state.rd_threshes[seg][bsize][MODE_IDX[INTRA][i]];
    // Short-circuit flat blocks
    if state.sf.short_circuit_flat_blocks && td.mb.source_variance == 0
       && ((td.mb.zero_temp_sad_source && mode != Dc) || i > 2) { continue; }
    // intra_y_mode_bsize_mask (A2 §5.6)
    if (1 << mode) & state.sf.intra_y_mode_bsize_mask[bsize] == 0 { continue; }
    // rt_intra_dc_only_low_content (speed 8)
    if state.sf.rt_intra_dc_only_low_content && mode != Dc
       && state.content_state_sb != ContentStateSb::VeryHighSad { continue; }
    // adaptive RD thresh
    if rd_less_than_thresh(best_rdc.rdcost, mode_rd_thresh, freq_fact) {
        if !state.is_screen() || state.svc_spatial_layer_id() > 0
           || (!state.scene_change && !state.high_motion()) { continue; }
    }
    td.mb.mi.mode = mode;
    td.mb.mi.ref_frame = [MvReferenceFrame::Intra, MvReferenceFrame::None];
    intra_pred::compute_intra_yprediction(state, td, block, mode);
    model_rd::for_sb_y(state, td, block, /* is_intra= */ true,
                       &mut this_rdc, &mut sse_y, &mut var_y);
    let outcome = block_yrd::block_yrd(state, td, block, &mut this_rdc,
                                       /* rd_computed= */ true, /* is_intra= */ true);
    apply_block_yrd_outcome_intra(...);
    // Screen-content per-plane intra check
    if state.is_screen()
       && (td.mb.color_sensitivity[0] || td.mb.color_sensitivity[1])
    {
        intra_pred::foreach_transformed_block_in_plane(state, td, block,
                                                       /* plane= */ 1 + i,
                                                       estimate_block_intra);
    }
    this_rdc.rate += state.costs.mbmode_cost[mode as usize];
    this_rdc.rate += state.costs.ref_frame_cost[MvReferenceFrame::Intra as usize];
    this_rdc.rate += state.costs.intra_cost_penalty;
    this_rdc.rdcost = rdcost!(state.rdmult, state.rddiv, this_rdc.rate, this_rdc.dist);
    if this_rdc.rdcost < best_rdc.rdcost {
        best_rdc = this_rdc;
        best = BestPickMode {
            best_mode: mode,
            best_ref_frame: MvReferenceFrame::Intra,
            best_intra_tx_size: td.mb.mi.tx_size,
            best_mode_skip_txfm: td.mb.skip_txfm[0],
            best_pred: None,             // intra doesn't lease pred_buf
            ..best
        };
    }
}
```

### 5.5 NEWMV diff-bias vs encode_breakout ordering (L-43 RESOLVED)

The cascade order in `vp9_pickmode.c:2403-2424` is:

```text
1. compute this_rdc.rdcost from rate + dist via RDCOST macro
2. if CBR && speed >= 5 && !screen:
       vp9_NEWMV_diff_bias(...)              // MUTATES this_rdc.rdcost
3. if allow_encode_breakout && !lossless && !scene_change
   && !svc.high_num_blocks_with_motion:
       encode_breakout_test(...)              // may set x->skip = 1
       if x->skip:
           this_rdc.rate = inter_mode_cost
           this_rdc.dist = sse << 4
           this_rdc.rdcost = RDCOST(rate, dist)   // re-cost
```

**Ordering matters because:**

1. `vp9_NEWMV_diff_bias` reads `this_rdc.rdcost` and writes
   `this_rdc.rdcost`. If we reorder it after `encode_breakout_test`,
   then when `encode_breakout_test` re-costs `this_rdc.rdcost`,
   the bias is applied to a different starting value.
2. `encode_breakout_test` reads `var_y, sse_y` (which came from
   `model_rd_for_sb_y[_large]`), not `this_rdc.dist`. So the bias
   doesn't affect what `encode_breakout_test` decides; it affects only
   the comparison of `this_rdc.rdcost < best_rdc.rdcost` later when
   `x->skip` is **not** set.

**Test plan:** unit test forces a config where NEWMV diff-bias would
fire (CBR + speed 5 + neighbors disagree about MV by > 48 pels) and
verifies that `this_rdc.rdcost` going into `encode_breakout_test` is
**post-bias**, and that the re-cost after `x->skip` ignores the bias
(uses the inter_mode_cost rate + sse<<4 dist freshly).

### 5.6 `block_yrd` fast-path bypass (L-41 RESOLVED)

The fast-path bypass at `vp9_pickmode.c:735-744`:

```c
if (sf->use_simple_block_yrd && cm->frame_type != KEY_FRAME &&
    (bsize < BLOCK_32X32 ||
     (use_svc && (bsize < BLOCK_32X32 || svc.temporal_layer_id > 0)))) {
    if (!rd_computed) {
        model_rd_for_sb_y(cpi, bsize, x, xd, &this_rdc->rate, &this_rdc->dist,
                          &var_y, &sse_y, 0);
    }
    *sse = INT_MAX;
    *skippable = 0;
    return;
}
```

`sf->use_simple_block_yrd` is set to **1** at:

- cpu-used 7 with `(temporal_layer > 0 || spatial > 1)`
  (`vp9_speed_features.c:712`).
- cpu-used 8 unconditionally (`vp9_speed_features.c:783`).

In v1, this means:

- **cpu-used 6:** never bypassed. Always runs Hadamard + SATD path.
- **cpu-used 7 single-layer:** never bypassed (the SVC condition
  fails).
- **cpu-used 7 SVC enhancement:** bypassed for `bsize < 32×32` or
  `temporal_layer > 0`.
- **cpu-used 8:** bypassed for `bsize < 32×32` always
  (`(bsize < BLOCK_32X32 || (use_svc && ...))` — the non-SVC branch
  collapses to `bsize < 32x32`).

The Rust port encodes this gate as a method on `SpeedFeatures` and the
`BlockYrdOutcome::Bypass` enum variant carries it through (§4.8). The
caller's `apply_block_yrd_outcome` (above in §5.1) maps:

- `Bypass`: do **not** touch `skip_txfm[0]` (leave whatever
  `model_rd_for_sb_y` set inside the `Bypass` branch — which is
  written inside `model_rd_for_sb_y` itself via `x->skip_txfm[0]`).
  Do **not** add `cost_bit(skip_prob, *)` adjustments (model_rd
  already accounts for them implicitly via its rate output? — **NO**,
  it does not. Reading libvpx: in the Bypass path, the caller's
  subsequent `cost_bit(skip_prob, 1)` is **not** added because the
  follow-up `if (cm->interp_filter == SWITCHABLE && mv_subpel)` rate
  adjustment runs unconditionally; `cost_bit(skip,1)` runs only in
  the `early_term` branch (line 2381). So `Bypass` means "rate/dist
  from model_rd verbatim, plus interp_filter rate, no skip_bit
  adjustment beyond what model_rd's `SKIP_TXFM_AC_DC` already
  represents").
- `Computed`: apply the SKIP_TXFM decision logic (§4.8 code block).

This is the **only** algorithmic divergence between speed 6 and speed
7+ that's worth a unit test (alongside the speed-feature mask
differences in §4.1). A "ms/frame at speed 6 vs 7" benchmark gates
the savings — should be ~30% drop in mode-decision time at speed 8
small blocks.

### 5.7 Filter-ref selection (L-48 RESOLVED)

From `vp9_pickmode.c:1488-1573` (reproduced in the source dive
above):

```text
filter_start = if force_smooth_filter { EightTapSmooth } else { EightTap }
filter_end   = EightTapSmooth
for filter in filter_start..=filter_end {
    mi.interp_filter = filter
    vp9_build_inter_predictors_sby()
    if use_model_yrd_large { model_rd_for_sb_y_large(...) }
    else                   { model_rd_for_sb_y(...) }
    pf_rate[filter] += vp9_get_switchable_rate()
    cost = RDCOST(rate, dist)
    if cost < best_cost:
        best_filter = filter
        ...
}
mi.interp_filter = best_filter
```

Key facts (resolved from L-48 / A2 §7 Q11 / async A7 #23):

1. **Only `EightTap` and `EightTapSmooth` are tested in non-RD.**
   `EightTapSharp` is **never** tested in the cpu-used 6-8 path.
2. **`force_smooth_filter` is read from `sf.force_smooth_interpol`**
   (a speed-feature flag). When set, `filter_start = EightTapSmooth`
   so only `EightTapSmooth` is tested.
3. **The decoder must accept `cm.interp_filter = SWITCHABLE` or a
   fixed filter.** When SWITCHABLE, the per-block `mi.interp_filter`
   is serialized; when fixed (`cm.interp_filter == EightTap`,
   `EightTapSmooth`), the per-block field is set to the same fixed
   value and not serialized.
4. **The result commits `mi.interp_filter` immediately.** No
   fall-back to `Switchable`. If `sf.cb_pred_filter_search` is 0 (the
   "no filter search" case), `mi.interp_filter` is set to either
   `cm.interp_filter` (if fixed) or `EightTap` (if SWITCHABLE) without
   running this function.

**Gate condition** for entering `search_filter_ref` (from
`vp9_pickmode.c:2294-2317`):

```text
((this_mode == NEWMV || sf.cb_pred_filter_search > 0
  || cm.interp_filter == SWITCHABLE)
 && pred_filter_search
 && (ref_frame == Last
     || (ref_frame == Golden && !force_mv_inter_layer
         && (svc || rc.rc_mode == VBR)))
 && mv_has_subpel_bits)
```

`pred_filter_search` itself is a complex per-context flag from
`vp9_pickmode.c:2287-2291`:

```c
pred_filter_search = cm->interp_filter == SWITCHABLE
                     ? sf->cb_pred_filter_search > 0
                     : 0;
if (pred_filter_search == 2)
    pred_filter_search = (sf->cb_pred_filter_search == 2 && (mi_row & 7) == 0);
```

So `sf.cb_pred_filter_search = 2` enables the filter search only on
every 8-row-aligned superblock — a speed optimization at higher
resolutions (1080p+: `vp9_speed_features.c:792`).

The Rust port:

```rust
pub(super) fn search_filter_ref(
    state: &PickModeState,
    td: &mut ThreadData,
    block: BlockPosition,
    this_rdc: &mut RdCost,
    this_mode_pred: &mut Option<PredBufferHandle>,
    this_early_term: &mut bool,
    use_model_yrd_large: bool,
) {
    let bw = bw(block.bsize);
    let force_smooth = state.sf.force_smooth_interpol;
    let filter_start = if force_smooth {
        InterpFilter::EightTapSmooth
    } else {
        InterpFilter::EightTap
    };
    let filter_end = InterpFilter::EightTapSmooth;

    let mut best_cost = i64::MAX;
    let mut best_filter = InterpFilter::Switchable;
    let mut best_skip_txfm = SkipTxfm::None;
    let mut best_early_term = false;
    let mut best_flag_preduv = [false; 2];

    // Per-filter state — array indexed by InterpFilter value (0..=2,
    // only 0 and 1 actually used here).
    let mut pf_rate = [0i32; 3];
    let mut pf_dist = [0i64; 3];
    let mut pf_var  = [0u32; 3];
    let mut pf_sse  = [0u32; 3];
    let mut pf_tx_size = [TxSize::Tx4X4; 3];
    let mut curr_rate = [0i32; 3];

    for filter in filter_start as u8..=filter_end as u8 {
        let f = InterpFilter::from_u8(filter);
        td.mb.mi.interp_filter = f;
        inter_pred::build_inter_predictors_sby(state, td, block);
        if use_model_yrd_large {
            model_rd::for_sb_y_large(state, td, block,
                &mut pf_rate[filter as usize],
                &mut pf_dist[filter as usize],
                &mut pf_var[filter as usize],
                &mut pf_sse[filter as usize],
                this_early_term,
                /* flag_preduv_computed= */ &mut [...]);
        } else {
            model_rd::for_sb_y(state, td, block, false,
                &mut pf_rate[filter as usize],
                &mut pf_dist[filter as usize],
                &mut pf_var[filter as usize],
                &mut pf_sse[filter as usize]);
        }
        curr_rate[filter as usize] = pf_rate[filter as usize];
        pf_rate[filter as usize] +=
            switchable_interp_rate_for_filter(state, td, f);
        let cost = rdcost!(td.mb.rdmult, td.mb.rddiv,
                           pf_rate[filter as usize],
                           pf_dist[filter as usize]);
        pf_tx_size[filter as usize] = td.mb.mi.tx_size;
        if cost < best_cost {
            best_filter = f;
            best_cost = cost;
            best_skip_txfm = td.mb.skip_txfm[0];
            best_early_term = *this_early_term;
            // Update best_flag_preduv from output.
            if state.sf.reuse_inter_pred_sby
                && let Some(handle) = this_mode_pred.as_mut()
                && handle.slot != /* current_pred slot */ {
                // Swap pred buffer per L-50 semantics.
                ...
            }
        }
    }
    // ... finalise best, commit mi.interp_filter, restore predictor.
    td.mb.mi.interp_filter = best_filter;
    td.mb.mi.tx_size = pf_tx_size[best_filter as usize];
    this_rdc.rate = curr_rate[best_filter as usize];
    this_rdc.dist = pf_dist[best_filter as usize];
    td.mb.skip_txfm[0] = best_skip_txfm;
    *this_early_term = best_early_term;
    // If not reuse_inter_pred && best_filter < filter_end, rebuild
    // predictor at best filter (vp9_pickmode.c:1569-1572).
    if !state.sf.reuse_inter_pred_sby && best_filter as u8 < filter_end as u8 {
        td.mb.mi.interp_filter = best_filter;
        inter_pred::build_inter_predictors_sby(state, td, block);
    }
}
```

### 5.8 `vp9_NEWMV_diff_bias` (A2 §4.8 verbatim)

```rust
/// Mirrors `vp9_NEWMV_diff_bias` (`vp9_pickmode.c:1298-1355`).
///
/// L-43 partial resolution. The constants 48 (pel), 8 (pel), 16 (pel)
/// are byte-parity-locked (PLAN.md D8). Not tunable.
pub(super) fn vp9_newmv_diff_bias(
    noise_level: NoiseLevel,
    td: &ThreadData,
    this_mode: PredictionMode,
    rdc: &mut RdCost,
    bsize: BlockSize,
    mv_row: i16,
    mv_col: i16,
    ref_is_last: bool,
    lowvar_highsumdiff: bool,
    sb_is_skin: bool,
) {
    if this_mode == PredictionMode::NewMv {
        let (a_valid, a_row, a_col) = if let Some(above) = td.xd.above_mi {
            (true, above.mv[0].row as i32, above.mv[0].col as i32)
        } else { (false, 0, 0) };
        let (l_valid, l_row, l_col) = if let Some(left) = td.xd.left_mi {
            (true, left.mv[0].row as i32, left.mv[0].col as i32)
        } else { (false, 0, 0) };
        let (al_row, al_col) = match (a_valid, l_valid) {
            (true, true)   => ((a_row + l_row) / 2, (a_col + l_col) / 2),
            (true, false)  => (a_row, a_col),
            (false, true)  => (l_row, l_col),
            (false, false) => (0, 0),
        };
        let row_diff = al_row - mv_row as i32;
        let col_diff = al_col - mv_col as i32;
        if row_diff.abs() > 48 || col_diff.abs() > 48 {
            if bsize > BlockSize::B32X32 {
                rdc.rdcost = rdc.rdcost.saturating_mul(2);          // <<1
            } else {
                rdc.rdcost = (3 * rdc.rdcost) >> 1;                  // *1.5
            }
        }
    }
    if noise_level >= NoiseLevel::Medium
        && bsize >= BlockSize::B32X32
        && ref_is_last
        && mv_row.abs() < 8 && mv_col.abs() < 8
    {
        rdc.rdcost = 7 * (rdc.rdcost >> 3);                          // *0.875
    } else if lowvar_highsumdiff
        && !sb_is_skin
        && bsize >= BlockSize::B16X16
        && ref_is_last
        && mv_row.abs() < 16 && mv_col.abs() < 16
    {
        rdc.rdcost = 7 * (rdc.rdcost >> 3);
    }
}
```

### 5.9 `encode_breakout_test` (A2 §4.9 verbatim)

```rust
/// Mirrors `vp9_pickmode.c:932-1035` (`encode_breakout_test`).
/// Sets `td.mb.skip = true` when the prediction is good enough that
/// the residual encodes as skip=1.
pub(super) fn encode_breakout_test(
    state: &PickModeState,
    td: &mut ThreadData,
    block: BlockPosition,
    ref_frame: MvReferenceFrame,
    this_mode: PredictionMode,
    var_y: u32,
    sse_y: u32,
    skip_out: &mut bool,
) {
    // Per A2 §4.9 verbatim. SVC golden short-circuit at top:
    if state.svc.is_some() && ref_frame == MvReferenceFrame::Golden {
        return;
    }
    let mv = td.mb.mi.mv[0];
    let motion_low = mv.row.abs() <= 64 && mv.col.abs() <= 64;
    let breakout = td.mb.encode_breakout;
    let (thresh_ac, thresh_dc) = if breakout > 0 && motion_low {
        let max_thresh = 36_000_i64;
        let min_thresh = (i64::from(breakout) << 4).min(max_thresh);
        let dq1 = state.dq_for_block(td, block).1 as i64;       // ac dequant
        let mut thresh_ac = (dq1 * dq1) >> 3;
        thresh_ac = thresh_ac.clamp(min_thresh, max_thresh);
        thresh_ac >>= 8 - (log2_bw(block.bsize) + log2_bh(block.bsize));
        let dq0 = state.dq_for_block(td, block).0 as i64;       // dc dequant
        let thresh_dc = (dq0 * dq0) >> 6;
        (thresh_ac, thresh_dc)
    } else { (0, 0) };

    if (var_y as i64) <= thresh_ac
        && ((sse_y as i64) - (var_y as i64)) <= thresh_dc
    {
        let (thresh_ac_uv, thresh_dc_uv) = if td.mb.sb_is_skin {
            (0, 0)
        } else {
            (thresh_ac, thresh_dc)
        };

        // Build UV predictors if not already.
        ensure_uv_predictors_built(state, td, block);

        let (var_u, sse_u) = state.fn_ptr_uv(block).variance(
            td.uv_src(0, block), td.uv_pred(0, block));
        if ((var_u as i64) << 2) <= thresh_ac_uv
            && (sse_u as i64 - var_u as i64) <= thresh_dc_uv
        {
            let (var_v, sse_v) = state.fn_ptr_uv(block).variance(
                td.uv_src(1, block), td.uv_pred(1, block));
            if ((var_v as i64) << 2) <= thresh_ac_uv
                && (sse_v as i64 - var_v as i64) <= thresh_dc_uv
            {
                *skip_out = true;
                // Caller patches this_rdc using inter_mode_cost + sse << 4.
            }
        }
    }
}
```

### 5.10 Sub-8×8 path (L-47 RESOLVED)

Verbatim from A2 §3.8:

```rust
pub fn pick_inter_mode_sub8x8(
    state: &PickModeState,
    td: &mut ThreadData,
    block: BlockPosition,
    pc_ctx: &mut PickModeContext,
    out_rdc: &mut RdCost,
) {
    // Setup ref-frame masks.
    let mut ref_frame_skip_mask: u8 = 0;
    let mut frame_mv = [[MotionVector::ZERO; MAX_REF_FRAMES]; PredictionMode::COUNT];

    for ref_frame in [MvReferenceFrame::Last, MvReferenceFrame::Golden] {
        if state.ref_frame_flags & ref_flag(ref_frame) != 0 {
            motion_est::find_mv_refs(state, td, block, ref_frame);
            motion_est::find_best_ref_mvs(state, td, block, ref_frame, &mut frame_mv);
        } else {
            ref_frame_skip_mask |= 1 << (ref_frame as u8);
        }
    }

    // Per-mi setup.
    td.mb.mi.sb_type = block.bsize;
    td.mb.mi.tx_size = TxSize::Tx4X4;
    td.mb.mi.uv_mode = IntraMode::Dc;
    td.mb.mi.ref_frame[0] = MvReferenceFrame::Last;
    td.mb.mi.interp_filter = if state.cm_interp_filter_is_switchable() {
        InterpFilter::EightTap
    } else {
        state.cm.interp_filter
    };

    let mut best_rd = i64::MAX;
    let mut best_ref_frame = MvReferenceFrame::Last;
    let mut bsi: [[B_ModeInfo; 4]; 2] = Default::default();

    for ref_frame in [MvReferenceFrame::Last, MvReferenceFrame::Golden] {
        if ref_frame_skip_mask & (1 << (ref_frame as u8)) != 0 { continue; }
        if state.ref_scaled[ref_frame_idx(ref_frame)] { continue; }

        let mut this_rd: i64 = 0;
        for i in 0..4 {
            // 4×4 sub-position inside the 8×8 (raster).
            let mut b_mv = [MotionVector::ZERO; PredictionMode::COUNT];
            b_mv[PredictionMode::ZeroMv as usize] = MotionVector::ZERO;
            b_mv[PredictionMode::NewMv as usize] = MotionVector::INVALID;
            motion_est::append_sub8x8_mvs_for_idx(state, td, block, i, &mut b_mv);

            let mut b_best_rd = i64::MAX;
            let mut b_rate = 0i32;
            for this_mode in PredictionMode::NearestMv..=PredictionMode::NewMv {
                let mut sub_rdc = RdCost::default();
                if this_mode == PredictionMode::NewMv {
                    motion_est::set_mv_search_range(...);
                    motion_est::full_pixel_search_sadperbit4(...);
                    if rate_exceeds_nearestmv_budget(...) { continue; }
                    motion_est::find_fractional_mv_step(...);
                } else {
                    b_rate += state.costs.inter_mode_cost
                              [mode_ctx][inter_offset(this_mode)];
                }
                inter_pred::build_inter_predictor_4x4(state, td, block, i,
                    td.mb.mi.interp_filter);
                model_rd::for_sb_y(state, td, block, false,
                    &mut sub_rdc.rate, &mut sub_rdc.dist, &mut _v, &mut _s);
                sub_rdc.rate += b_rate;
                let sub_rd = rdcost!(td.mb.rdmult, td.mb.rddiv,
                                     sub_rdc.rate, sub_rdc.dist);
                if sub_rd < b_best_rd {
                    b_best_rd = sub_rd;
                    bsi[ref_frame_idx(ref_frame)][i].as_mode = this_mode;
                    bsi[ref_frame_idx(ref_frame)][i].as_mv = ...;
                }
            }
            this_rd += b_best_rd;
        }
        if this_rd < best_rd {
            best_rd = this_rd;
            best_ref_frame = ref_frame;
        }
    }

    // Commit.
    td.mb.mi.tx_size = TxSize::Tx4X4;
    td.mb.mi.ref_frame[0] = best_ref_frame;
    for i in 0..4 {
        td.mb.bmi[i] = bsi[ref_frame_idx(best_ref_frame)][i];
    }
    // L-47 LOAD-BEARING: mi.mode is taken from the LAST sub-block's mode.
    td.mb.mi.mode = td.mb.bmi[3].as_mode;
    out_rdc.rdcost = best_rd;
}
```

L-47 RESOLVED: `mi.mode = bmi[3].as_mode` mirrored exactly. The VP9
spec disambiguates per-sub-block via `bmi[]`; `mi.mode` is the
syntax-default and is the **last** sub-block's mode by raster order.
A unit test verifies this against a libvpx reference encode.

### 5.11 Cost LUT derivation

`initialize_frame_costs` walks the per-frame `FrameContext` (the
adapted probability tables from the previous frame's
`vp9_adapt_*_probs`) and produces the integer cost LUTs. Each LUT
entry is computed by walking the prob-tree with `vp9_prob_cost[256]`
(A4 §5.11.0). Mirrors `vp9_initialize_rd_consts`
(`vp9_rd.c:559-700`).

The per-frame `intra_cost_penalty` follows `vp9_get_intra_cost_penalty`
(`vp9_rd.c:794-815`):

```rust
fn intra_cost_penalty(qindex: u8, qdelta: i8, bsize: BlockSize,
                      noise_level: NoiseLevel) -> i32 {
    // 8-bit only — D-01. dc_quant for 8-bit:
    let dc_quant = quant::dc_quant(qindex, qdelta, BitDepth::Eight);
    let reduction = match bsize {
        BlockSize::B4X4 | BlockSize::B4X8 | BlockSize::B8X4 | BlockSize::B8X8 => 4,
        BlockSize::B8X16 | BlockSize::B16X8 | BlockSize::B16X16 => 2,
        _ => 0,
    };
    let reduction = if noise_level == NoiseLevel::High { 0 } else { reduction };
    (20 * (dc_quant as i32)) >> reduction
}
```

This LUT is read by every `pick_inter_mode` invocation; it lives in
`FrameCosts` (§4.10).

### 5.12 Adaptive RD threshold (A2 §4.6, L-49)

Storage decision (L-49 RESOLVED): **per-tile** for tile-MT,
**per-row** when `sf.adaptive_rd_thresh_row_mt = 1` (cpu-used 7+
with row-MT). The `TileSlot` (HLD-threading §5.5) carries both:

```rust
pub struct TileSlot<'frame> {
    pub thresh_freq_fact: UnsafeCell<[[i32; MAX_MODES]; BLOCK_SIZES]>,
    pub row_base_thresh_freq_fact: Option<UnsafeCell<Box<[i32]>>>,
    pub mode_map: UnsafeCell<[[i8; MAX_MODES]; BLOCK_SIZES]>,
    // ... other fields per HLD-threading §5.5
}
```

The mode-decision module accesses these through a typed wrapper
`FreqFactAccess<'a>` that picks the right storage based on
`sf.adaptive_rd_thresh_row_mt`:

```rust
pub(super) struct FreqFactAccess<'a> {
    per_tile: *mut [[i32; MAX_MODES]; BLOCK_SIZES],
    per_row: Option<(*mut [i32], usize /* sb_row */)>,
}

impl<'a> FreqFactAccess<'a> {
    pub(super) fn read(&self, bsize: BlockSize, thr_mode: ThrMode) -> i32 {
        match self.per_row {
            Some((row_ptr, sb_row)) => unsafe {
                let row_slice = &*row_ptr;
                let stride = BLOCK_SIZES * MAX_MODES;
                row_slice[sb_row * stride + bsize as usize * MAX_MODES + thr_mode as usize]
            },
            None => unsafe {
                (*self.per_tile)[bsize as usize][thr_mode as usize]
            },
        }
    }
    // SAFETY: this_row sync guaranteed by RowSync (HLD-threading §4.3-§4.5).
    pub(super) unsafe fn write(&mut self, bsize: BlockSize, thr_mode: ThrMode, v: i32) { ... }
}
```

The `rd_less_than_thresh_row_mt` function from A2 §4.6 is then:

```rust
pub(super) fn rd_less_than_thresh(best_rd: i64, thresh: i64,
                                  thresh_fact: i32) -> bool {
    best_rd < (thresh * thresh_fact as i64) >> 5 || thresh == i64::MAX
}
```

And the update is mirrored from A2 §4.6 verbatim, with the speed-6
SVC bump (`adaptive_rd_thresh = 4`) and the speed-8 CBR clamp-to-1
all in `SpeedFeatures` (set by `speed_features` module, not
mode-decision).

### 5.13 Filter-ref vs intra interaction

When intra wins the final pick, `mi.interp_filter` is the
`SWITCHABLE` default (or whatever was last committed in the inter
loop). The serializer ignores `interp_filter` for intra-coded blocks
(`vp9_bitstream.c:write_inter_mode`), so this is informational. The
LLD doesn't need to clear `mi.interp_filter` on intra path.

---

## 6. SIMD dispatch

This module has **no SIMD kernels of its own.** Every SIMD-touched
operation it performs is a call into another module:

| Operation | Owner module | Kernel name | A2 §6 ref |
|---|---|---|---|
| Variance (`fn_ptr[bsize].vf`) | `rvp9-core::simd::variance` | `vpx_variance{...}` | A2 §6.1 |
| SAD for reference masking (`vp9_mv_pred`) | `rvp9-core::simd::sad` | `vpx_sad{...}` | A2 §6.3 |
| 4-way SAD | `rvp9-core::simd::sad` | `vpx_sad{...}x4d` | A2 §6.3 |
| Avg-pred SAD (compound — v2) | n/a in v1 | `vpx_sad{...}_avg` | A2 §6.3 |
| Hadamard 4×4 / 8×8 / 16×16 | `rvp9-core::simd::hadamard` | `vpx_hadamard_*` | A2 §6.6 |
| SATD (sum-abs-coeff) | `rvp9-core::simd::hadamard` | `vpx_satd` | A2 §6.7 |
| Block-error fp | `rvp9-core::simd::block_error` | `vp9_block_error_fp` | A2 §6.8 |
| Quantize-fp | `rvp9-core::simd::quant` | `vp9_quantize_fp` | A2 §6.9 |
| Subtract block | `rvp9-core::simd::residual` | `vpx_subtract_block` | A2 §6.10 |
| Sub-pixel variance | `rvp9-core::simd::subpel_var` | `vpx_sub_pixel_variance_*` | A2 §6.12 |
| Inter-predictor convolve | `rvp9-core::simd::convolve` | `vpx_convolve8*` | A2 §6.13 |
| Intra predictor | `rvp9-core::simd::intra_pred` | `vpx_*_predictor_*` | (in transforms LLD) |

All accessed via `&PickModeState::fn_ptr[bsize]` (variance/SAD) or
`rvp9_core::simd::dispatch::CONVOLVE.intra_pred(...)` (the global
RTCD table — HLD-simd §3.1). RTCD initialization runs once before
the first worker spawn (R-09; HLD-threading §3.5).

The module has **no scalar kernels**. Its scalar fallback is the same
scalar fallback every SIMD kernel ships per HLD-simd §5.

**Test strategy:** the mode-decision module's tests use the scalar
kernels by default (forced via `cfg(test)` setting the dispatch
table). Differential tests cycle through every available SIMD ISA
per `rvp9-core::simd::test_dispatch`.

---

## 7. Deliberate divergences from libvpx

Each entry feeds `notes/divergence-ledger.md`.

### 7.1 Compound modes deferred to v2 (L-46)

**Where:** `rvp9-encoder::mode_decision::pick_inter`,
`active_ref_mode_set` always returns the non-compound slice.
libvpx counterpart: `vp9_pickmode.c:2039-2053` compound modes branch
(`comp_modes` 0..=2, only reached when `sf.use_compound_nonrd_pickmode
&& usable_ref_frame == AltrefFrame`).

**What libvpx does:** when CBR + `use_compound_nonrd_pickmode = 1`
and `usable_ref_frame == AltrefFrame`, tries 2 compound modes
forced to ZEROMV: `(Last, Altref)` and `(Golden, Altref)`. Set by
`vp9_speed_features.c:639-640` (VBR + lag) and
`vp9_speed_features.c:670-671` (VBR + lag at speed 6).

**What rvp9 does:** in v1 we have no altref (D-13 RESOLVED), so
`usable_ref_frame` is never `Altref`, so the gate is never opened.
The `comp_modes = 0` is hard-coded. The compound branch in
`pick_inter_mode` is `unreachable!("compound deferred to v2 (L-46)")`.

**Why:** altref deferred (D-13). When altref returns post-v1, this
LLD will gain a `compound` submodule. We do not stub the compound
path in v1; the only artifact is the `unreachable!()` guard.

**Byte-parity impact:** none. v1 never enables the gate that compound
would reach.

**Compression-efficiency impact:** none in v1 (compound is only
useful with altref).

**Future plan:** v2 when altref returns.

### 7.2 Altref iterations 8..11 in REF_MODE_SET kept but always skip

**Where:** `mode_decision::ref_mode_set::REF_MODE_SET` keeps the 12
entries verbatim; the loop's `if ref > usable continue` filter at
`vp9_pickmode.c:2065-2066` short-circuits all 4 altref entries in v1.

**What libvpx does:** iterates the full 12-entry table; if altref is
disabled at config (no `use_altref_onepass`), `usable = Golden` and
the 4 altref entries are skipped at the first `continue`.

**What rvp9 does:** identical.

**Why:** the iteration index drives `mode_rd_thresh` indexing into
`THR_MODES`; renumbering would change byte semantics.

**Byte-parity impact:** none — identical loop shape.

**Compression-efficiency impact:** ~4 wasted iterations per block
(~30 ns each). Total: ~120 ns per SB → ~30 μs/frame at 720p30. Not
a real performance concern; revisit when altref returns.

**Future plan:** unchanged when altref ships in v2; the table is
already "altref-shaped".

### 7.3 `intra_cost_penalty` `BITS_8` assumption

**Where:** `mode_decision::intra_cost::intra_cost_penalty` always
calls `quant::dc_quant(qindex, qdelta, BitDepth::Eight)`. libvpx
counterpart: `vp9_rd.c:794-815`.

**What libvpx does:** dispatches on `xd->bd` (the bit depth) and
calls `vp9_dc_quant(qindex, qdelta, xd->bd)`.

**What rvp9 does:** hard-coded `BitDepth::Eight`. Per D-01.

**Byte-parity impact:** none for Profile 0.

**Compression-efficiency impact:** none for Profile 0.

**Future plan:** v2 generic-over-pixel-type — the `quant::dc_quant`
signature is already `(qindex, qdelta, BitDepth)` so re-introducing
HBD requires only generalising the caller.

### 7.4 No `EightTapSharp` in filter-ref search

**Where:** `mode_decision::filter_ref::search_filter_ref`
hard-codes `filter_end = EightTapSmooth`. libvpx counterpart:
`vp9_pickmode.c:1513`.

**What libvpx does:** identical — `filter_end = EightTAP_SMOOTH` is
hardcoded in the non-RD path.

**What rvp9 does:** identical.

**Byte-parity impact:** none. This is a libvpx-mirrored choice.

**Future plan:** never. The cpu-used 6-8 path never tests
`EightTapSharp`; the GOOD/best-quality path does but that's not in
our scope.

### 7.5 NEWMV diff-bias constants hardcoded

**Where:** `mode_decision::newmv_bias::vp9_newmv_diff_bias` uses the
hardcoded constants 48 (neighbour-diff threshold), 8 (low-motion
threshold for noise path), 16 (low-motion threshold for
lowvar-highsumdiff path), plus the bias multipliers 2 (<<1), 1.5
(*3/2), and 0.875 (*7/8). libvpx counterpart: `vp9_pickmode.c:1335-1355`.

**What libvpx does:** same constants, hardcoded.

**What rvp9 does:** same constants, hardcoded. **No tunability.**

**Byte-parity impact:** none.

**Compression-efficiency impact:** none.

**Future plan:** L-43 disposition is "byte parity (D8)". Revisit
only when we have a different rate-control target where these
heuristics misbehave.

### 7.6 30-iteration claim corrected to 16

**Where:** This LLD §5.2 documents the actual upper bound of 16
iterations (12 inter + 4 intra) in v1. The A2 §1 "30-iteration loop"
phrase was loose.

**What libvpx does:** the loop runs 0..=11 (inter, 12 entries) + 0..=3
(intra, 4 entries), with 0..=2 optional compound. v1 max: 16.

**What rvp9 does:** same.

**Byte-parity impact:** none. Naming-only correction.

**Future plan:** when compound returns in v2, max climbs to 18 in v1
config + altref + compound enabled.

### 7.7 `BlockYrdOutcome` enum boundary (L-41)

**Where:** `mode_decision::block_yrd::block_yrd` returns
`BlockYrdOutcome` instead of three out-params (`*sse`, `*skippable`,
plus rate/dist in the `RdCost`). libvpx counterpart:
`vp9_pickmode.c:735-744` (the early-bailout) and the rest of
`block_yrd`.

**What libvpx does:** `*sse = INT_MAX; *skippable = 0; return;` —
caller checks `*sse == INT_MAX` implicitly via the path-divergent
control flow.

**What rvp9 does:** explicit enum `Bypass | Computed { sse, skippable }`.
The caller's `apply_block_yrd_outcome` switches on the variant.

**Byte-parity impact:** **none**, because the variant maps 1:1 to the
libvpx out-param values (Bypass = (INT_MAX, false)).

**Compression-efficiency impact:** none.

**Why divergence:** Rust idiom + clarity for L-41. A C-shaped "set
out-params + return void" would compile but the enum makes the
caller's switch explicit and lets the compiler check exhaustiveness.

**Future plan:** stable.

### 7.8 `variance_low[25]` indexing helper

**Where:** `mode_decision::types::VarianceLowLookup`. libvpx
counterpart: `get_force_skip_low_temp_var` (`vp9_pickmode.c:1441-1486`).

**What libvpx does:** flat `uint8_t[25]` array with a hand-coded
switch on `sb_type` and computed index `pos_shift_16x16[i][j]`.

**What rvp9 does:** wraps the flat `[u8; 25]` in `VarianceLowLookup`
with a typed `lookup_for_block(mi_row, mi_col, bsize)` accessor. The
underlying byte layout is preserved (per L-39 / async A7 #17:
Morton-ordered for 16×16 entries).

**Byte-parity impact:** none — same bytes, typed accessor.

**Future plan:** stable.

### 7.9 `PredBufferHandle` RAII wrapper

**Where:** `mode_decision::pred_buffer::PredBufferHandle`. libvpx
counterpart: `get_pred_buffer` / `free_pred_buffer` +
`PRED_BUFFER::in_use` field (`vp9_pickmode.c:916-928`).

**What libvpx does:** manual in_use=1/0 tracking with a 4-entry
stack array.

**What rvp9 does:** same 4-entry stack array, but the lease is a
`PredBufferHandle<'pool>` whose `Drop` clears `in_use`.

**Byte-parity impact:** none — same allocation + release sequence.

**Future plan:** stable.

### 7.10 Cost LUT location

**Where:** `rvp9-encoder::mode_decision::cost_tables::FrameCosts`
lives in `rvp9-encoder`, not `rvp9-core`. libvpx counterpart:
`cpi->inter_mode_cost`, `cpi->mbmode_cost`, etc.

**What libvpx does:** fields on `VP9_COMP`, derived per-frame in
`vp9_initialize_rd_consts`.

**What rvp9 does:** owned by `mode_decision`. Per HLD-arch §3.4: the
encoder builds derived cost LUTs (`inter_mode_cost`,
`intra_uv_mode_cost`, `partition_cost`, etc.) from the current
`Common::fc` per-frame on the main thread; tile workers read the
cost LUTs, never `Common::fc` directly.

**Byte-parity impact:** none — identical numeric values.

**Future plan:** stable.

### 7.11 cpu-used 5 dispatch excluded from this LLD's scope

**Where:** `pick_inter_mode` is the cpu-used 6-8 path. cpu-used 5
also uses `use_nonrd_pick_mode = 1` but adds `REFERENCE_PARTITION` +
some RDO retry behaviour. The Rust port supports cpu-used 5 in the
config validation layer but mode-decision treats it identically to
cpu-used 6 (the only diff at this layer is `partition_search_type`,
which is partition's concern).

**What libvpx does:** at cpu-used 5, `partition_search_type =
REFERENCE_PARTITION` (or `VAR_BASED_PARTITION` for VBR+ARF group).
The mode-decision body is identical to cpu-used 6.

**What rvp9 does:** identical.

**Byte-parity impact:** none.

### 7.12 Iteration-index-based mode_rd_thresh table

**Where:** `mode_decision::rd_thresh::adaptive_rd_thresh` reads from
`state.rd_threshes[seg][bsize]` indexed by the `THR_MODES` slot
derived from `MODE_IDX[ref][col]`. libvpx counterpart: same.

**What libvpx does:** identical lookup pattern.

**What rvp9 does:** identical.

**Byte-parity impact:** none.

---

## 8. Open questions

### 8.1 EXT_RATECTRL injection point in mode-decision

EXT_RATECTRL (R-06 RESOLVED in v1 per HLD-svc §6) feeds Q-decisions
into the frame loop **before** workers spawn. So by the time
`pick_inter_mode` runs, the per-frame `state.base_qindex` and
per-segment `state.y_dequant[seg]` are already the EXT-RC-provided
values. **Mode-decision does not call into EXT_RATECTRL**; it reads
the already-applied effect through `PickModeState`.

This LLD does not need an EXT_RATECTRL hook surface. The integration
point is in `rate_control`'s frame-start callback. Confirmed with
HLD-svc §6.

**Status:** no decision required at this LLD.

### 8.2 Speed-6 `scale_part_thresh_sumdiff` no-op (A2 §7 Q8, L-45)

A2 §7 Q8: "Reading the code, speed 6 falls through both `if speed >=
8` and `elif speed == 7` checks → returns `threshold_base` unchanged.
Confirm this is intended."

**Status:** L-45 owned by `partition.md` LLD (not this one), since
the threshold scaling is partition-side. Mode-decision reads
`state.vbp_thresholds[]` and doesn't care which speed produced them.
Cross-ref to partition LLD.

### 8.3 `flag_preduv_computed` lifecycle clarity

`flag_preduv_computed[2]` is an out-param of
`model_rd_for_sb_y_large` that the caller uses to decide whether to
rebuild UV predictors in `encode_breakout_test`. In Rust, it lives
on `MacroBlock` as `td.mb.flag_preduv_computed: [bool; 2]` and is
reset to `[false; 2]` at the top of every mode iteration. This is
slightly different from libvpx (which uses a stack local), but the
behaviour is identical.

**Status:** documented; not a divergence (interior storage location).

### 8.4 Should `search_filter_ref` always rebuild on second filter?

libvpx's `search_filter_ref` runs the inter-pred + model_rd twice
(once per filter). On the second filter, it rebuilds inter-pred
into a *different* `PRED_BUFFER` slot (via `current_pred =
&tmp[get_pred_buffer(tmp, 3)]`). This means the first filter's
predictor stays alive in case it's the winner.

In Rust, the `PredBufferHandle::lease_alt` pattern (§4.9) preserves
this. The implementation pulls one extra `tmp` slot for the
"current pred under test" while the previous best lives in a
different slot.

**Status:** implementation detail; documented in §5.7 code. No
external decision needed.

### 8.5 Cross-LLD ordering — partition runs before mode-decision

`pick_inter_mode` requires `variance_low[25]` (set by
`partition::set_low_temp_var_flag`) and `content_state_sb` (set by
`partition::avg_source_sad`). The partition LLD owns those writes;
this LLD reads them as `state.variance_low_lookup` and
`state.content_state_sb`. The contract is:

> By the time `partition::nonrd_pick_partition` calls
> `mode_decision::pick_inter_mode`, the following partition-side
> state is final for the SB: `variance_low[25]`, `content_state_sb`,
> `sb_is_skin`, `lowvar_highsumdiff`, `skip_low_source_sad`,
> `zero_temp_sad_source`.

**Status:** cross-LLD contract; mirror in partition LLD §3 (the
contract is symmetric — partition's §3 lists what it writes; this
LLD §3 / §4.1 lists what it reads).

### 8.6 PRED_BUFFER stack pressure

Stack-allocating 3 × 64 × 64 bytes = 12 KiB inside `pick_inter_mode`
is fine on every target we ship (Linux/Windows ≥ 1 MiB default
stack). `pick_inter_mode` is called from worker threads; HLD-threading
sizes worker stacks at 4 MiB.

In `cfg(target_os = "wasi")` or other constrained targets, we'd
need to heap-allocate the pool. For v1, `target_arch = "wasm32"`
support is non-goals (PLAN.md §8).

**Status:** documented in §4.9; stack-allocate.

### 8.7 Tail-call / inline pressure on `skip_iteration` helper

`skip_iteration` (§5.1, §5.3) is one big function with 17+
short-circuits. Performance matters: it runs ≤16 times per SB ≤264
SBs per frame ≤30 fps = 127k calls/sec at 720p30 single-tile. If
each call adds 50 ns (a single L1 miss), that's 6 ms/sec — 0.6% of
budget.

The Rust port relies on `#[inline]` on the helper, plus `match`-based
short-circuiting (the compiler should fold the boolean checks). A
post-Stage-4a benchmark gates whether we need to manually inline the
checks back into `pick_inter_mode`.

**Status:** flagged as performance-watch in Stage 4a; not blocking
this LLD.

### 8.8 Sub-pel cost-list invariants (L-07, owned by ME LLD)

`pick_inter_mode` passes a 4-entry cost-list down to the sub-pel
search. The list layout is L-07's concern (motion-estimation LLD).
Mode-decision treats it as a `[i32; 5]` slice and forwards. No
decision needed here.

**Status:** delegated to motion-estimation LLD.

### 8.9 Cross-tile freq-fact starting state

When two tile workers start a frame, do they share a starting
`thresh_freq_fact[][]` snapshot? In libvpx, each tile's `TileDataEnc`
holds its own `thresh_freq_fact[]`, **carried across frames** (not
reset per-frame). The values are mutated only by that tile's worker.

In rvp9, each `TileSlot` owns its own per-frame freq-fact array and
the values persist across frames. No cross-tile sharing.

**Status:** documented; matches libvpx.

---

## 9. Test strategy

### 9.1 Unit tests

Per submodule:

- `pick_inter.rs`:
  - Skip-rules cascade — 17+ tests, one per skip rule (§5.3). Each
    test constructs a synthetic `PickModeState + ThreadData` that
    triggers exactly that rule and asserts the iteration is skipped.
  - 30-iter upper-bound test (L-40 validation): construct a
    pathological SB that would attempt all 12 inter + 4 intra
    iterations. Count actual iterations. Assert ≤ 16 in v1.
  - ZeroMV/Last priority: a flat block (variance == 0) picks
    `(Last, ZeroMv)` before any other mode is tested.
- `block_yrd.rs`:
  - Fast-path bypass test (L-41): at cpu-used 8, BLOCK_16X16 inter,
    `block_yrd` returns `BlockYrdOutcome::Bypass`; at cpu-used 6,
    `BlockYrdOutcome::Computed`.
  - Hadamard rate proxy validates against libvpx's `block_yrd` on
    fixed seeds (50 input vectors).
- `model_rd.rs`:
  - `vp9_model_rd_from_var_lapndz` table-driven test: for `var ∈
    {0, 1, 100, 65536}` and `qstep ∈ {2..=64}` and `n_log2 ∈ {4..=12}`,
    output matches libvpx's value (verbatim table read; signed-int
    arithmetic — §5.4 byte-parity trap).
  - Edge case `var == 0` returns `(rate=0, dist=0)`.
  - Edge case `xsq_q10` clipped to `MAX_XSQ_Q10 = 245727`.
- `encode_breakout.rs`:
  - Motion threshold (|mv| > 64) skips the breakout.
  - SVC + GOLDEN early-returns.
  - skin sets uv thresholds to 0.
  - Skip is set only when both U and V pass their thresholds.
- `newmv_bias.rs`:
  - Above-only neighbour valid: bias uses `a_mv` directly.
  - Both neighbours valid: bias uses `((a_mv + l_mv) / 2)`.
  - Threshold edge: `row_diff == 48` does **not** trigger (`> 48`).
  - Threshold edge: `row_diff == 49` triggers (BLOCK_64X64 → ×2;
    BLOCK_32X32 → ×1.5).
  - Noise-medium + ref==LAST + bsize≥32×32 + |mv|<8 → ×0.875.
  - lowvar_highsumdiff + ref==LAST + bsize≥16×16 + |mv|<16 + not skin
    → ×0.875.
  - Bias is applied **before** encode_breakout (L-43 ordering test).
- `filter_ref.rs`:
  - `force_smooth_filter = true` skips EIGHTTAP, tests only
    EIGHTTAP_SMOOTH.
  - Best filter selection: forge model_rd outputs such that filter A
    wins, assert `mi.interp_filter = A`.
  - `reuse_inter_pred_sby = false`: rebuilds predictor at best filter
    when it differs from `filter_end`.
- `rd_thresh.rs`:
  - `rd_less_than_thresh(best_rd, thresh, fact)`: classical
    threshold-multiplier behaviour.
  - `update_thresh_freq_fact`: best-mode decay (×15/16), non-best
    increment by 1 capped to `adaptive_rd_thresh << 6`.
- `pred_buffer.rs`:
  - 4 leases at once exhausts the pool.
  - Drop releases the slot.
  - Lease-find-free returns the lowest-index free slot.

### 9.2 Property tests

- For 1000 random `(var, qstep, n_log2)` triples, `model_rd_norm`'s
  Rust output equals libvpx's C output (use `quickcheck` with
  shrinking).
- For 1000 random `MV (row, col)` pairs and neighbours, the bias
  output is monotonic in `|row_diff| + |col_diff|`.
- For 1000 random SB inputs at varying speeds, `pick_inter_mode`
  produces a `mi.ref_frame[0]` in `{Intra, Last, Golden}` (never
  `Altref` in v1).

### 9.3 Differential tests against libvpx

- Build a synthetic test corpus of 64 superblocks across different
  content states (low-var, high-motion, screen, noisy, flat). Run
  libvpx vpxenc cpu-used 7 single-tile single-frame and rvp9 with
  `mode_decision` invoked directly. Compare:
  - Selected `(mi.mode, mi.ref_frame[0], mi.tx_size, mi.interp_filter)`.
  - `mi.mv[0]` (within 1 pel sub-pel tolerance — A3 owns the
    bit-exact gate).
  - `mi.skip` and `mi.skip_txfm[0]`.

  Acceptance: 99.9% byte-match per (mode, ref) pair; the 0.1% slack
  is for floating-point rate-control upstream (L-04) bleeding into
  `base_qindex`.

- Cross-decoder differential: encode 100 frames through rvp9, decode
  through libvpx, compare YUV pixel-wise. Expect bit-exact in v1
  (no float in mode-decision; all integer arithmetic).

### 9.4 Conformance vectors

- Mode-decision itself does not have conformance vectors (those are
  decoder-side). The end-to-end "rvp9 encoder produces a stream that
  passes libvpx decoder" gate at Stage 4a exercises this LLD.

### 9.5 Performance regression

- `bench/mode_decision.rs`: encode a 720p single tile through 30
  frames at cpu-used 7, measure mean `pick_inter_mode` runtime per
  SB. Target: <50 μs per SB on a modern x86_64 core. Gate: >70 μs
  fails CI.
- `bench/mode_decision_speed_sweep.rs`: same corpus at cpu-used 6,
  7, 8. Speed-7 should be 1.3-1.6× faster than speed-6 (the
  `block_yrd` fast-path bypass is the main speed-7 lever); speed-8
  another 1.2-1.4× over speed-7.

### 9.6 Fuzz / soak

- Mode-decision itself doesn't take untrusted input — it takes the
  encoder's own state. `cargo fuzz` for this module isn't useful at
  the function level. The 72-hour soak (Stage 5) exercises this
  module via end-to-end encoding.

### 9.7 Miri runs

- `pick_inter_mode` on a single SB under Miri verifies the
  `PickModeContext` mutation pattern and the `PredBufferHandle`
  `UnsafeCell` access discipline. Run as part of CI per
  HLD-threading §9.

### 9.8 Loom tests for cross-thread interaction

- Mode-decision doesn't directly interact with other threads (the
  freq-fact read/write is per-tile, sync'd by partition/row-mt).
  No loom tests at this level; HLD-threading's row-MT loom tests
  cover the relevant sync edges.

---

## Summary

**Line count:** 1422 lines. **Key §7 entries:** L-46 (compound deferred to v2), 7.2 (altref iterations kept-but-skip), 7.5 (NEWMV diff-bias constants byte-locked), 7.6 (30-iter claim corrected to 16 via L-40), 7.7 (`BlockYrdOutcome` enum boundary for L-41), 7.9 (`PredBufferHandle` RAII for L-50). **Blocking questions in §8:** none — every open question is delegated to a partner LLD (partition, motion-estimation, rate-control, svc) or is a performance-watch item for Stage 4a.
