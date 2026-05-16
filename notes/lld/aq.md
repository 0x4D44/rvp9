# LLD: aq (adaptive quantization — CYCLIC_REFRESH only)

**Scope (locked):** AQ mode 3 (`CYCLIC_REFRESH_AQ`) only. Modes 1
(variance, `vp9_aq_variance.c`) and 2 (complexity, `vp9_aq_complexity.c`)
are **dropped from v1** per `PLAN.md` §2 and D-05 RESOLVED
(`notes/vp9-analysis/decisions-needed.md:101-122`, divergence-ledger
2026-05-16 entry "AQ modes 1 and 2 not ported"). 8-bit Profile 0 only
(D-01 RESOLVED). No altref (D-13 RESOLVED).

## 1. Purpose

`aq::cyclic_refresh` implements VP9's **segment-based cyclic background
refresh**: each P-frame, ~`percent_refresh` % (default 10%) of 8×8-mi
blocks are coded at a *boosted* (lower) Q so that, over a refresh
period of ~`100 / percent_refresh` frames, every block has been
refreshed at higher quality. This raises the quality floor of static /
low-motion regions without raising average bitrate, by amortising the
boost across many frames. The subsystem owns:

1. The persistent per-8×8-mi `cr.map` (refresh schedule) and
   `cr.last_coded_q_map` (last-coded qindex per block).
2. Per-frame segmentation setup — BOOST1 / BOOST2 qindex-delta
   computation, segmentation map population,
   `cm.seg.feature[..][SEG_LVL_ALT_Q]` wire-up.
3. Per-block segment-id assignment during mode decision
   (`candidate_refresh_aq`).
4. Post-encode bookkeeping (actual seg block counts, low-content
   accumulator, optional golden-frame demotion).
5. The screen-content frame-level Q-decrease cap
   (`vp9_cyclic_refresh_limit_q`).

All state mutations between frames run on the main encoder thread.
Tile workers read `cr.map`, `cr.last_coded_q_map`, and
`cr.qindex_delta[3]` immutably during a frame; the only worker-side
mutation is per-block writes to the per-block-disjoint slice of
`cpi.segmentation_map` (HLD-threading strategy C,
`notes/hld/01-threading.md:537-539`).

## 2. libvpx origin

Sole source file: `third_party/libvpx/vp9/encoder/vp9_aq_cyclicrefresh.c`
(703 ll) + header `vp9_aq_cyclicrefresh.h` (147 ll). Stage 1 primary:
A5 (`notes/vp9-analysis/05-rate-control.md` §2.2, §3.3, §4.9, §5.5,
§6, §7 Q-RC-12 / 13 / 22 / 24 / 29).

| libvpx symbol | File:line | This LLD covers in |
|---|---|---|
| `struct CYCLIC_REFRESH` | `vp9_aq_cyclicrefresh.h:32-74` | §4.1 |
| `CR_SEGMENT_ID_*` macros | `vp9_aq_cyclicrefresh.h:23-30` | §4.2 |
| `vp9_cyclic_refresh_alloc` | `vp9_aq_cyclicrefresh.c:32-53` | §3.1, §5.1 |
| `vp9_cyclic_refresh_free` | `vp9_aq_cyclicrefresh.c:55-61` | §3.1 (Drop) |
| `candidate_refresh_aq` (static) | `vp9_aq_cyclicrefresh.c:67-87` | §5.4 |
| `compute_deltaq` (static) | `vp9_aq_cyclicrefresh.c:90-99` | §5.3 |
| `vp9_cyclic_refresh_estimate_bits_at_q` | `vp9_aq_cyclicrefresh.c:105-130` | §3.2, §5.5 |
| `vp9_cyclic_refresh_rc_bits_per_mb` | `vp9_aq_cyclicrefresh.c:137-155` | §3.2, §5.5 |
| `vp9_cyclic_refresh_update_segment` | `vp9_aq_cyclicrefresh.c:160-222` | §3.2, §5.4 |
| `vp9_cyclic_refresh_update_sb_postencode` | `vp9_aq_cyclicrefresh.c:224-254` | §3.2, §5.6 |
| `vp9_cyclic_refresh_postencode` | `vp9_aq_cyclicrefresh.c:260-316` | §3.2, §5.7 |
| `vp9_cyclic_refresh_set_golden_update` | `vp9_aq_cyclicrefresh.c:319-333` | §3.2, §5.8 |
| `is_superblock_flat_static` (static) | `vp9_aq_cyclicrefresh.c:335-355` | §5.9 |
| `cyclic_refresh_update_map` (static) | `vp9_aq_cyclicrefresh.c:363-475` | §5.9 |
| `vp9_cyclic_refresh_update_parameters` | `vp9_aq_cyclicrefresh.c:478-592` | §3.2, §5.2, §5.5 |
| `vp9_cyclic_refresh_setup` | `vp9_aq_cyclicrefresh.c:595-677` | §3.2, §5.3 |
| `vp9_cyclic_refresh_get_rdmult` | `vp9_aq_cyclicrefresh.c:679-681` | §3.2 |
| `vp9_cyclic_refresh_reset_resize` | `vp9_aq_cyclicrefresh.c:683-693` | §3.2 |
| `vp9_cyclic_refresh_limit_q` | `vp9_aq_cyclicrefresh.c:695-702` | §3.2, §5.10 |
| `cyclic_refresh_segment_id_boosted` (inline) | `vp9_aq_cyclicrefresh.h:127-130` | §3.3 |
| `cyclic_refresh_segment_id` (inline) | `vp9_aq_cyclicrefresh.h:132-139` | §3.3 |
| `VP9_VAR_OFFS[64]` constant | `vp9_aq_cyclicrefresh.c:24-30` | §5.9 |

External libvpx hooks this LLD plugs into (covered by their own LLDs;
listed here for completeness):

| Where libvpx calls into CR | File:line | rvp9 LLD |
|---|---|---|
| Per-frame CR parameters / golden update | `vp9_ratectrl.c:2493+` (in `vp9_rc_get_one_pass_cbr_params`) | `rate-control.md` |
| Frame-level setup (segmentation + delta-Q) | `vp9_encoder.c:4209-4239` (AQ frame-setup dispatch); for CR this routes to `vp9_cyclic_refresh_setup` | `rate-control.md` (caller), this LLD (callee) |
| Q-bounds search using weighted bits-per-MB | `vp9_ratectrl.c:702-704` (`vp9_rc_regulate_q`) | `rate-control.md` |
| Q-cap for screen-content (`limit_q`) | `vp9_ratectrl.c:683-685` (inside `adjust_q_cbr`) | `rate-control.md` |
| RCF update — uses `estimate_bits_at_q` | `vp9_ratectrl.c:757-759` (inside `vp9_rc_update_rate_correction_factors`) | `rate-control.md` |
| Per-block segment-id write (mode decision) | `vp9_encodeframe.c:1810-1814` and `:2533` | `mode-decision.md` |
| Post-encode SB update (last_coded_q_map) | `vp9_encodeframe.c:6576-6578` | `mode-decision.md` (caller) |
| Post-encode frame-level pass | `vp9_encoder.c:` (after pack, inside `encode_frame_to_data_rate`); per A5 §3.1 line 555-556 | `rate-control.md` |
| RD-multiplier for boosted segments | `vp9_encodemb.c` (call to `vp9_cyclic_refresh_get_rdmult`) | `quant.md` / `mode-decision.md` |
| Resize-pending reset | `vp9_cyclic_refresh_setup:605` invokes `vp9_cyclic_refresh_reset_resize` | this LLD §3.2 |

## 3. Public API

Module: `rvp9_encoder::aq::cyclic_refresh`. The encoder god-struct
holds a single `CyclicRefresh` (no `Option`) per
`notes/hld/00-architecture.md:398`; with `cfg.aq_mode == AqMode::Off`
the struct is allocated but `apply_cyclic_refresh` stays `false` and
the entry points are no-ops, matching libvpx's `aq_mode == 0`.

### 3.1 Construction / teardown

```rust
/// Cyclic-refresh state, allocated once per encoder and reused
/// frame-to-frame. Mirrors libvpx's `CYCLIC_REFRESH`
/// (`vp9_aq_cyclicrefresh.h:32-74`). Buffers (`map`,
/// `last_coded_q_map`) size `mi_rows * mi_cols`; at 1080p that's
/// 270×480 = ~260 KiB combined. Allocation failure → `EncodeError`.
pub struct CyclicRefresh { /* §4 */ }

impl CyclicRefresh {
    /// libvpx `vp9_cyclic_refresh_alloc` (`vp9_aq_cyclicrefresh.c:32-53`).
    /// `mi_rows`, `mi_cols` must be > 0.
    pub fn new(mi_rows: u32, mi_cols: u32) -> Result<Self, EncodeError>;

    /// Re-allocate maps for RESIZE_DYNAMIC; resets state per
    /// `vp9_cyclic_refresh_reset_resize` (`:683-693`).
    pub fn reallocate_for_dims(&mut self, mi_rows: u32, mi_cols: u32)
        -> Result<(), EncodeError>;
}

// `Drop` releases the boxed slices automatically.
```

### 3.2 Per-frame entry points (main thread only)

All accept `&mut CyclicRefresh` and the surrounding context borrows. The
caller order is fixed by `rate-control.md` §3.1 ("Per-frame rate-control
sequence") and matches A5 §3.1 (`notes/vp9-analysis/05-rate-control.md:496-573`).

```rust
/// Golden-frame update interval picker. Algorithm: §5.8. libvpx:
/// `vp9_cyclic_refresh_set_golden_update`
/// (`vp9_aq_cyclicrefresh.c:319-333`). Called from
/// `RateControl::get_one_pass_cbr_params` when
/// `frames_till_gf_update_due == 0`. Returns the chosen
/// `baseline_gf_interval`.
pub fn set_golden_update(
    &self,
    rc: &RateControlReadOnly,    // avg_frame_low_motion, frames_since_key
    rc_mode: RcMode,             // CBR vs VBR — branches the formula
) -> u32;

/// Decide whether CR runs this frame; pick per-frame tuning and
/// `weight_segment`. Sets `self.apply_cyclic_refresh`. Algorithm: §5.2.
/// libvpx: `vp9_cyclic_refresh_update_parameters`
/// (`vp9_aq_cyclicrefresh.c:478-592`). Called from
/// `RateControl::get_one_pass_{cbr,vbr}_params` immediately after
/// `set_golden_update` (A5 §3.1, `05-rate-control.md:509-510`).
/// Preconditions: `rc.avg_frame_qindex`, `rc.frames_since_key`,
/// `rc.avg_frame_low_motion`, `rc.avg_frame_bandwidth`, `noise` reflect
/// the *previous* encoded frame. Postcondition: if
/// `apply_cyclic_refresh`, all per-frame tuning fields are freshly set;
/// otherwise downstream code must check the flag before reading.
pub fn update_parameters(
    &mut self,
    common: &EncoderCommon,    // frame_type, mi_rows, mi_cols, MBs, bit_depth
    rc: &RateControlReadOnly,
    cfg: &EncoderConfig,       // oxcf.content, oxcf.rc_mode
    noise: &NoiseEstimate,
    svc: &SvcState,
);

/// Bits-per-MB estimate consumed by `RateControl::regulate_q`.
/// Algorithm: §5.5. libvpx: `vp9_cyclic_refresh_rc_bits_per_mb`
/// (`vp9_aq_cyclicrefresh.c:137-155`). Precondition:
/// `apply_cyclic_refresh && weight_segment` reflect the current frame.
pub fn rc_bits_per_mb(
    &self,
    frame_type: FrameType,
    q: QIndex,
    correction_factor: f64,
    bit_depth: BitDepth,        // always Bd8 in v1, kept for API parity
    speed: u8,
) -> u32;

/// Post-encode bit estimate consumed by
/// `RateControl::update_rate_correction_factors`. libvpx:
/// `vp9_cyclic_refresh_estimate_bits_at_q`
/// (`vp9_aq_cyclicrefresh.c:105-130`). Uses **actual** seg1/seg2 block
/// counts (not `weight_segment`) — see §7.3 for the asymmetry.
pub fn estimate_bits_at_q(
    &self,
    frame_type: FrameType,
    base_qindex: QIndex,
    mbs: u32,
    correction_factor: f64,
    bit_depth: BitDepth,
) -> u32;

/// Build this frame's segmentation map and compute segment-1 / 2
/// qindex deltas. Algorithm: §5.3. libvpx: `vp9_cyclic_refresh_setup`
/// (`vp9_aq_cyclicrefresh.c:595-677`). Called after
/// `RateControl::pick_q_and_bounds` set `cm.base_qindex`, before
/// `encode_frame()` (A5 §3.1, `05-rate-control.md:529-533`). Writes
/// `cm.seg.*` and `seg_map[..]` on the main thread between frame
/// encodes; tile workers do not observe an inconsistent state.
pub fn setup(
    &mut self,
    common: &mut EncoderCommon,
    rc: &RateControlReadOnly,
    cfg: &EncoderConfig,
    noise: &NoiseEstimate,
    svc: &SvcState,
    source: &SourceFrame,        // for `is_superblock_flat_static`
    last_source: Option<&SourceFrame>,
    consec_zero_mv: &[u8],       // per-8x8-mi counter from RC
    seg_map: &mut [u8],          // cpi->segmentation_map
);

/// Per-block update during mode decision (worker context — §3.4).
/// Algorithm: §5.4. libvpx: `vp9_cyclic_refresh_update_segment`
/// (`vp9_aq_cyclicrefresh.c:160-222`). Mutates the per-block-disjoint
/// `cr.map` slice and writes the chosen segment id into both
/// `mi.segment_id` and `seg_map[..]`. The `&mut [i8]` map slice and
/// `&mut [u8]` seg_map slice must reference disjoint per-block index
/// ranges (HLD-threading §5.2 strategy C,
/// `notes/hld/01-threading.md:537-539`); callers obtain these via the
/// tile-worker aliasing contract owned by `tile-threading.md`.
pub fn update_segment(
    &self,                            // immutable borrow: rate/dist/motion thresholds + qindex_delta
    cr_map_slice: &mut [i8],          // tile-disjoint slice of self.map
    seg_map_slice: &mut [u8],         // tile-disjoint slice of cpi->segmentation_map
    mi: &mut ModeInfo,
    mi_row: u32, mi_col: u32, bsize: BlockSize,
    rate: i64, dist: i64, skip: bool,
    skin_block_probe: Option<SkinProbeFn>,    // None if use_skin_detection off
    use_nonrd_pick_mode: bool,
    cm_mi_cols: u32, cm_mi_rows: u32,
);

/// Per-SB post-encode update of `last_coded_q_map`. Algorithm: §5.6.
/// libvpx: `vp9_cyclic_refresh_update_sb_postencode`
/// (`vp9_aq_cyclicrefresh.c:224-254`). Worker-side, tile-disjoint
/// slice; same aliasing contract as `update_segment`.
pub fn update_sb_postencode(
    &self,                           // immutable: qindex_delta + base_qindex passed in
    last_q_slice: &mut [u8],         // tile-disjoint slice
    mi: &ModeInfo,
    mi_row: u32, mi_col: u32, bsize: BlockSize,
    base_qindex: QIndex,
    cm_mi_cols: u32, cm_mi_rows: u32,
);

/// Post-encode frame-level pass: counts actual seg blocks, accumulates
/// low-content motion, optionally demotes `refresh_golden_frame`.
/// Algorithm: §5.7. libvpx: `vp9_cyclic_refresh_postencode`
/// (`vp9_aq_cyclicrefresh.c:260-316`). Returns a `PostEncodeDecision`
/// the rate-control loop applies (golden refresh / interval). The
/// explicit return is divergence §7.5.
pub fn postencode(
    &mut self,
    common: &EncoderCommon,
    rc: &RateControlReadOnly,
    cfg: &EncoderConfig,
    svc: &SvcState,
    seg_map: &[u8],
    mi_grid: &[ModeInfo],            // cm.mi_grid_visible (read-only)
    resize_pending: i32,
) -> PostEncodeDecision;

/// Screen-content frame-level Q decrease cap. Algorithm: §5.10.
/// libvpx: `vp9_cyclic_refresh_limit_q` (`vp9_aq_cyclicrefresh.c:695-702`).
/// Call site (`RateControl::adjust_q_cbr`) MUST gate on
/// `cfg.content == ScreenContent && cfg.aq_mode == CyclicRefresh`
/// (L-24, §7.4).
pub fn limit_q(&self, prev_frame_q: QIndex, q: &mut QIndex);

/// RD-multiplier for BOOST1, computed in `setup`. libvpx:
/// `vp9_cyclic_refresh_get_rdmult` (`vp9_aq_cyclicrefresh.c:679-681`).
pub fn rdmult(&self) -> i32;

/// Resize reset (`vp9_cyclic_refresh_reset_resize`, `:683-693`).
/// Called from `setup` when `cm.resize_pending != 0`.
pub fn reset_resize(&mut self, mi_rows: u32, mi_cols: u32);
```

### 3.3 Helpers (free functions, `const fn` where possible)

```rust
/// `segment_id == BOOST1 || segment_id == BOOST2`.
/// libvpx inline `cyclic_refresh_segment_id_boosted`
/// (`vp9_aq_cyclicrefresh.h:127-130`).
#[inline] pub const fn segment_id_boosted(id: u8) -> bool;

/// Normalize an arbitrary segment id to BASE / BOOST1 / BOOST2.
/// libvpx inline `cyclic_refresh_segment_id`
/// (`vp9_aq_cyclicrefresh.h:132-139`).
#[inline] pub const fn normalize_segment_id(id: u8) -> u8;
```

### 3.4 Threading classification

`&mut self` entry points run on the main thread between frame encodes
(HLD-threading classifications A / F, `notes/hld/01-threading.md:537-539`).
`update_segment` and `update_sb_postencode` take `&self` plus
tile-disjoint `&mut [i8]` / `&mut [u8]` slices — strategy C. Read-only
helpers (`rc_bits_per_mb`, `estimate_bits_at_q`, `rdmult`, `limit_q`,
`segment_id_boosted`, `normalize_segment_id`) are `&self` / `const fn`.
The struct has no `UnsafeCell` / interior mutability; aliasing happens
only at the slice level via the tile aliasing contract in
`tile-threading.md`.

## 4. Types

### 4.1 `CyclicRefresh` struct

```rust
#[derive(Debug)]
pub struct CyclicRefresh {
    // ---- Per-frame tuning (refreshed by `update_parameters`) ----

    /// Percentage of 8×8-mi blocks targeted for refresh per frame
    /// (default 10; 5 if `reduce_refresh`; 0 on VBR golden-refresh).
    /// libvpx `cr->percent_refresh`; A5 §4.9, §5.5
    /// (`05-rate-control.md:1031,1483`).
    pub percent_refresh: i32,
    /// Max segment-delta as % of base-Q (default 60). `cr->max_qdelta_perc`.
    pub max_qdelta_perc: i32,
    /// Reject-from-refresh MV threshold, 1/8-pel units (default 32).
    /// `cr->motion_thresh`.
    pub motion_thresh: i16,
    /// BOOST2 amplifier (default 15; 10 disables BOOST2).
    /// `cr->rate_boost_fac`.
    pub rate_boost_fac: i32,
    /// `cr->rate_ratio_qdelta` (default 2.0; 3.0 for first ~4 cycles
    /// after KF; 1.7 if noisy; 1.5 for VBR; ≥2.5 for low-res non-low-rate).
    pub rate_ratio_qdelta: f64,
    /// Screen-content flat-static-skip flag.
    pub skip_flat_static_blocks: bool,
    /// Smoothed fraction of frame in non-base segment, consumed by
    /// `rc_bits_per_mb`. SET BY `update_parameters` — see §5.2 for the
    /// L-21 integer-shift trap. `cr->weight_segment`.
    pub weight_segment: f64,
    /// Global on/off for this frame. False on intra-only, non-base TL,
    /// lossless, undershoot, starved, or ROI background-skip past
    /// `FRAMES_NO_SKIPPING_AFTER_KEY=20`. libvpx
    /// `cr->apply_cyclic_refresh` (`vp9_aq_cyclicrefresh.c:490-505`).
    pub apply_cyclic_refresh: bool,

    // ---- Persistent across frames (mutated only between encodes) ----

    /// Cyclic-refresh map at 8×8-mi resolution; values per §4.2.
    /// `cr->map` (`vp9_aq_cyclicrefresh.h:51`). Sized `mi_rows * mi_cols`.
    pub map: Box<[i8]>,
    /// Last qindex (including segment delta) each 8×8-mi block was
    /// coded at; init MAXQ. `cr->last_coded_q_map` (`:53`).
    pub last_coded_q_map: Box<[u8]>,
    /// Wrap-around cursor for `update_map`. `cr->sb_index`.
    pub sb_index: u32,
    /// Halves next frame's `percent_refresh` when fewer than 3/4 of
    /// `map==0` blocks passed the qindex/consec-zero-mv gate. libvpx
    /// `cr->reduce_refresh`.
    pub reduce_refresh: bool,
    /// Incremented in `setup`, reset on scene change / KF. Drives
    /// screen-content "fast refresh after maxQ" (`:539-540`).
    pub counter_encode_maxq_scene_change: i32,
    /// 3:1 IIR low-content fraction used by `postencode` to demote
    /// `refresh_golden_frame`. `cr->low_content_avg`.
    pub low_content_avg: f64,
    /// `cr->content_mode` — always `true` in libvpx today (`:51`).
    /// Kept as a field for API parity / future tunability.
    pub content_mode: bool,

    // ---- Per-frame scratch (set by `setup`, read by workers) ----

    /// `(sb64_target_rate << 8) << 2` rate threshold for
    /// `candidate_refresh_aq`. `cr->thresh_rate_sb`.
    pub thresh_rate_sb: i64,
    /// `((q * q) as i64) << 2` distortion threshold, quadratic in real Q.
    /// `cr->thresh_dist_sb`.
    pub thresh_dist_sb: i64,
    /// Segment qindex deltas. `[0]==0` (BASE = baseline Q). `[1]`,
    /// `[2]` signed, NEGATIVE (boost). Written only by `setup`.
    /// `cr->qindex_delta[3]` (`vp9_aq_cyclicrefresh.c:657, 671`).
    pub qindex_delta: [i32; 3],
    /// RD-multiplier for BOOST1: `vp9_compute_rd_mult(cm.base_qindex +
    /// cm.y_dc_delta_q + qindex_delta[1])` clamped to MAXQ.
    /// `cr->rdmult`.
    pub rdmult: i32,

    // ---- Stats produced by per-block / post-encode passes ----

    /// `percent_refresh * mi_rows * mi_cols / 100` from `update_map`.
    /// `cr->target_num_seg_blocks`.
    pub target_num_seg_blocks: i32,
    /// Counted by `postencode`; consumed by next-frame
    /// `update_parameters` and by `estimate_bits_at_q` (RCF).
    /// `cr->actual_num_seg{1,2}_blocks`.
    pub actual_num_seg1_blocks: i32,
    pub actual_num_seg2_blocks: i32,

    // ---- Dimensions (cached) ----
    pub mi_rows: u32,
    pub mi_cols: u32,

    // DIVERGENCE: no `time_for_refresh` field. libvpx writes only `0`
    // (`:509`); see §7.2 (L-03 RESOLVED).
}
```

`Send` and `Sync` are both derived (auto-derived; no `UnsafeCell`, no
raw pointers, no `*const`/*`*mut`). The struct lives in the encoder
god-struct (`Encoder.cyclic_refresh: CyclicRefresh`,
`notes/hld/00-architecture.md:398`) and is borrowed `&` by tile
workers; tile workers gain `&mut [i8]` / `&mut [u8]` slices of
`self.map` and the external `seg_map` / `last_coded_q_map` only via
the per-tile aliasing contract owned by `tile-threading.md`.

### 4.2 `cr.map` value semantics

The map is `i8` per 8×8-mi position. Three states:

| Value | Meaning | Source |
|---|---|---|
| `1` | Not a candidate for refresh this cycle. | `vp9_aq_cyclicrefresh.c:211` |
| `0` | Candidate for refresh (eligible). | `:208` |
| Negative `-N` | Was refreshed at distance `N` from now; decremented to 0 each subsequent frame. | `:203, :442-444` |

In libvpx the magnitude is `cr->time_for_refresh`, but that field is
hard-coded to 0 (see §7 L-03), so in practice the only ever-stored
negative value is **0** (i.e., the value goes 0 → 0 on refresh — the
distinction between "just refreshed" and "candidate" collapses for the
current libvpx tunings). For byte-parity we still write `-0i8`
(= `0i8`) here; the difference vs libvpx is **none** because libvpx
also writes `-0`. We document this explicitly so a future tuning that
changes `time_for_refresh` cannot be silently introduced. See §7 (L-03).

### 4.3 Segment-id constants

```rust
/// Baseline (no boost). libvpx: `CR_SEGMENT_ID_BASE`
/// (`vp9_aq_cyclicrefresh.h:25`).
pub const CR_SEGMENT_ID_BASE: u8 = 0;
/// Normal boost (negative qindex delta).
/// libvpx: `CR_SEGMENT_ID_BOOST1` (`vp9_aq_cyclicrefresh.h:26`).
pub const CR_SEGMENT_ID_BOOST1: u8 = 1;
/// Aggressive boost (more-negative delta) for large zero-MV blocks.
/// libvpx: `CR_SEGMENT_ID_BOOST2` (`vp9_aq_cyclicrefresh.h:27`).
pub const CR_SEGMENT_ID_BOOST2: u8 = 2;
/// Cap on `rate_ratio_qdelta * 0.1 * rate_boost_fac` for BOOST2.
/// libvpx: `CR_MAX_RATE_TARGET_RATIO` (`vp9_aq_cyclicrefresh.h:30`).
pub const CR_MAX_RATE_TARGET_RATIO: f64 = 4.0;
```

These three segment-ids are also referenced by the segmentation
machinery in the bitstream LLD (`bitstream.md`) when emitting
`segment_id` codes; that LLD MUST import the same constants from
`rvp9_encoder::aq::cyclic_refresh` (no re-declaration) to prevent drift.

### 4.4 Static lookup `VP9_VAR_OFFS`

```rust
/// Flat-static reference block: 64 bytes all equal to 128, used as a
/// "fake mid-gray reference" when `is_superblock_flat_static` calls
/// the BLOCK_64X64 variance kernel. libvpx:
/// `vp9_aq_cyclicrefresh.c:24-30`.
pub(crate) static VP9_VAR_OFFS: [u8; 64] = [128; 64];
```

The full 64-byte buffer is required because libvpx passes
`stride = 0` so the kernel re-reads the same row 64 times for a
BLOCK_64X64 variance; this is bit-identical to a constant-128 64×64
source. See `is_superblock_flat_static` (`vp9_aq_cyclicrefresh.c:340-355`)
and §5.9 below.

### 4.5 `PostEncodeDecision` return type

```rust
/// Returned by `CyclicRefresh::postencode`. Aggregates the two state
/// changes the rate-control loop must apply.
#[derive(Default, Debug)]
pub struct PostEncodeDecision {
    /// True ⇒ force a golden refresh this frame even if RC says no.
    /// libvpx: `force_gf_refresh = 1`
    /// (`vp9_aq_cyclicrefresh.c:299`).
    pub force_golden_refresh: bool,
    /// True ⇒ demote a pending `refresh_golden_frame`. libvpx:
    /// `cpi->refresh_golden_frame = 0` (line 310).
    pub demote_golden_refresh: bool,
    /// New `baseline_gf_interval` to write into RC when
    /// `force_golden_refresh` is true. libvpx: `:295`.
    pub new_baseline_gf_interval: u32,
}
```

The return-as-struct is a deliberate divergence from libvpx's
direct mutation of `cpi` (see §7 entry "explicit PostEncodeDecision").

## 5. Internal algorithm

Pseudocode-level. Cite libvpx file:line for every non-obvious step.
Throughout, `mi_rows * mi_cols` is shorthand for the 8×8-mi grid count;
"sb" means 64×64 superblock; "block_index" is `mi_row * mi_cols + mi_col`.

### 5.1 Allocation (`CyclicRefresh::new`)

```text
new(mi_rows, mi_cols):
  map               = vec![0i8;  mi_rows * mi_cols].into_boxed_slice()
  last_coded_q_map  = vec![MAXQ; mi_rows * mi_cols].into_boxed_slice()
  sb_index          = 0
  counter_encode_maxq_scene_change = 0
  content_mode      = true            // libvpx: cr->content_mode = 1
  // All other f64/i32 fields are zero-default;
  // `apply_cyclic_refresh = false` until first `update_parameters`.
```

Equivalent to `vp9_cyclic_refresh_alloc`
(`vp9_aq_cyclicrefresh.c:32-53`). The `last_coded_q_map = MAXQ`
init is load-bearing for `update_map`: a freshly-allocated block must
"appear stale" so the first refresh cycle includes every position.

### 5.2 `update_parameters` — per-frame tuning

Mirrors `vp9_cyclic_refresh_update_parameters`
(`vp9_aq_cyclicrefresh.c:478-592`) verbatim, in the same order so the
arithmetic is bit-identical.

```text
update_parameters(common, rc, cfg, noise, svc):
  num8x8bl  = common.MBs * 4
  qp_thresh = min(if cfg.content == Screen { 35 } else { 20 },
                  rc.best_quality << 1)
  qp_max_thresh = (117 * MAXQ) >> 7        // = 233
  self.apply_cyclic_refresh = true

  // Gates — line 491-505.
  if  common.frame_is_intra_only()
   || svc.temporal_layer_id > 0
   || cfg.is_lossless_requested()
   || rc.avg_frame_qindex[INTER_FRAME] < qp_thresh
   || (svc.use_svc && svc.layer_ctx[svc.temporal_layer_id].is_key_frame)
   || (!svc.use_svc && self.content_mode
       && rc.avg_frame_low_motion < 20
       && rc.frames_since_key > 40)
   || (!svc.use_svc && rc.avg_frame_qindex[INTER_FRAME] > qp_max_thresh
       && rc.frames_since_key > 20)
   || (cfg.roi.enabled && cfg.roi.skip[BACKGROUND_SEG_SKIP_ID]
       && rc.frames_since_key > FRAMES_NO_SKIPPING_AFTER_KEY=20):
    self.apply_cyclic_refresh = false
    return

  // Defaults (line 506-511).
  self.percent_refresh    = if self.reduce_refresh { 5 } else { 10 }
  self.max_qdelta_perc    = 60
  // `cr->time_for_refresh = 0;` from libvpx :509 — NOT stored (L-03).
  self.motion_thresh      = 32
  self.rate_boost_fac     = 15

  // First ~4 refresh periods after KF use larger delta (line 515-518).
  if self.percent_refresh > 0
     && rc.frames_since_key
        < 4 * svc.number_temporal_layers
            * (100 / self.percent_refresh):
    self.rate_ratio_qdelta = 3.0
  else:
    self.rate_ratio_qdelta = 2.0
    if self.content_mode && noise.enabled && noise.level >= kMedium:
      self.rate_ratio_qdelta = 1.7
      self.rate_boost_fac    = 13

  // Screen-content (line 531-543).
  if cfg.content == Screen:
    if svc.spatial_layer_id == svc.number_spatial_layers - 1:
      self.skip_flat_static_blocks = true
    self.percent_refresh = if self.skip_flat_static_blocks { 5 } else { 10 }
    if self.content_mode && self.counter_encode_maxq_scene_change < 30:
      self.percent_refresh = if self.skip_flat_static_blocks { 10 } else { 15 }
    self.rate_ratio_qdelta = 2.0
    self.rate_boost_fac    = 10

  // Low-resolution (line 545-553).
  if common.width * common.height <= 352 * 288:
    if rc.avg_frame_bandwidth < 3000:
      self.motion_thresh   = 64
      self.rate_boost_fac  = 13
    else:
      self.max_qdelta_perc  = 70
      self.rate_ratio_qdelta = max(self.rate_ratio_qdelta, 2.5)

  // VBR (line 554-565).
  if cfg.rc_mode == VBR:
    self.percent_refresh   = 10
    self.rate_ratio_qdelta = 1.5
    self.rate_boost_fac    = 10
    if refresh_golden_frame_pending() && !svc.use_svc:
      self.percent_refresh   = 0
      self.rate_ratio_qdelta = 1.0

  // === weight_segment math — L-21 trap, see §7. ===
  target_refresh = self.percent_refresh * mi_rows * mi_cols / 100
  weight_segment_target = target_refresh as f64 / num8x8bl as f64
  // The (target + actual1 + actual2) >> 1 division is INTEGER, performed
  // BEFORE the cast to f64. libvpx writes:
  //   weight_segment = (double)((target + actual1 + actual2) >> 1) / num8x8bl
  // — the `>> 1` is an i32 shift on the sum, not a /2.0 on the result.
  // Re-ordering to `(target as f64 + actual1 as f64 + actual2 as f64) * 0.5`
  // changes the result by up to 0.5 / num8x8bl, which compounds across
  // frames into a 0.1-0.3% BD-rate shift. SEE §7 (L-21 RESOLVED).
  sum_i32  = (target_refresh + self.actual_num_seg1_blocks
              + self.actual_num_seg2_blocks) >> 1     // i32 ops
  weight_segment = sum_i32 as f64 / num8x8bl as f64

  if weight_segment_target < 7.0 * weight_segment / 8.0:
    weight_segment = weight_segment_target

  // Screen-content override (line 581-584): drop target from blend.
  if cfg.content == Screen:
    weight_segment =
      (self.actual_num_seg1_blocks + self.actual_num_seg2_blocks) as f64
      / num8x8bl as f64

  self.weight_segment = weight_segment

  if !self.content_mode:
    self.actual_num_seg1_blocks = self.percent_refresh * mi_rows * mi_cols / 100
    self.actual_num_seg2_blocks = 0
    self.weight_segment = self.actual_num_seg1_blocks as f64 / num8x8bl as f64
```

### 5.3 `setup` — per-frame segmentation build

Mirrors `vp9_cyclic_refresh_setup`
(`vp9_aq_cyclicrefresh.c:595-677`). Called after
`RateControl::pick_q_and_bounds` has set `cm.base_qindex`.

```text
setup(common, rc, cfg, noise, svc, source, last_source,
      consec_zero_mv, seg_map):
  scene_change = rc.high_source_sad
                 || (svc.use_svc && svc.high_source_sad_superframe)

  if common.current_video_frame == 0:
    self.low_content_avg = 0.0
  if common.resize_pending != 0:
    self.reset_resize(common.mi_rows, common.mi_cols)

  if !self.apply_cyclic_refresh
     || common.force_update_segmentation
     || scene_change:
    seg_map.fill(CR_SEGMENT_ID_BASE)
    common.seg.disable()
    if common.frame_type == KeyFrame || scene_change:
      self.last_coded_q_map.fill(MAXQ)
      self.sb_index = 0
      self.reduce_refresh = false
      self.counter_encode_maxq_scene_change = 0
    return

  // Active CR setup.
  self.counter_encode_maxq_scene_change += 1
  // libvpx calls vpx_clear_system_state() here — Rust has no FPU
  // control-word leak; omit (see §7 "vpx_clear_system_state dropped").

  q_real = qindex_to_q(common.base_qindex, common.bit_depth)
  self.thresh_rate_sb = (rc.sb64_target_rate as i64) << 10   // <<8 then <<2
  self.thresh_dist_sb = ((q_real * q_real) as i64) << 2

  common.seg.enable()
  common.seg.clear_all_features()
  common.seg.abs_delta = SEGMENT_DELTADATA
  common.seg.disable_feature(CR_SEGMENT_ID_BASE,   SEG_LVL_ALT_Q)
  common.seg.enable_feature (CR_SEGMENT_ID_BOOST1, SEG_LVL_ALT_Q)
  common.seg.enable_feature (CR_SEGMENT_ID_BOOST2, SEG_LVL_ALT_Q)

  // Segment 1 delta (line 656-657).
  qindex_delta_1 = compute_deltaq(common.base_qindex, self.rate_ratio_qdelta)
  self.qindex_delta[0] = 0
  self.qindex_delta[1] = qindex_delta_1

  // RD-mult for segment 1 (line 660-662).
  qindex2 = clamp(common.base_qindex + common.y_dc_delta_q + qindex_delta_1,
                  0, MAXQ as i32)
  self.rdmult = compute_rd_mult(qindex2)            // from quant.md
  common.seg.set_data(CR_SEGMENT_ID_BOOST1, SEG_LVL_ALT_Q, qindex_delta_1)

  // Segment 2 delta — line 666-672. Note the asymmetric clamp:
  // both `CR_MAX_RATE_TARGET_RATIO=4.0` AND `compute_deltaq`'s own
  // `max_qdelta_perc` clamp apply; the latter is applied INSIDE
  // compute_deltaq.
  ratio2 = min(CR_MAX_RATE_TARGET_RATIO,
               0.1 * self.rate_boost_fac as f64 * self.rate_ratio_qdelta)
  qindex_delta_2 = compute_deltaq(common.base_qindex, ratio2)
  self.qindex_delta[2] = qindex_delta_2
  common.seg.set_data(CR_SEGMENT_ID_BOOST2, SEG_LVL_ALT_Q, qindex_delta_2)

  // Build the segmentation map.
  self.update_map(common, cfg, noise, source, last_source, consec_zero_mv,
                  seg_map)


compute_deltaq(q, rate_factor):
  // vp9_aq_cyclicrefresh.c:90-99 (static)
  raw = vp9_compute_qdelta_by_rate(rc, common.frame_type, q, rate_factor,
                                   common.bit_depth)    // from rate-control.md
  let cap = -(self.max_qdelta_perc * q / 100)           // negative cap
  if -raw > self.max_qdelta_perc * q / 100:
    return cap
  return raw
```

The `compute_deltaq` math here matches libvpx :95-97 literally. Note
the integer division order `(self.max_qdelta_perc * q / 100)` —
all i32, no f64; preserving order matters for byte-parity at edge
qindex values where rounding can flip by 1.

### 5.4 `update_segment` — per-block during mode decision

Mirrors `vp9_cyclic_refresh_update_segment`
(`vp9_aq_cyclicrefresh.c:160-222`). The most subtle invariants are
(a) the optional skin-block override; (b) the segment-id reset on
skip; and (c) the special map-update encoding when transitioning
states. This runs from `mode-decision.md` inside the non-RD picker
once a best mode is chosen.

```text
update_segment(cr_map_slice, seg_map_slice, mi, mi_row, mi_col, bsize,
               rate, dist, skip, skin_probe, use_nonrd_pick_mode,
               cm_mi_cols, cm_mi_rows):
  bw = num_8x8_blocks_wide_lookup[bsize]
  bh = num_8x8_blocks_high_lookup[bsize]
  xmis = min(cm_mi_cols - mi_col, bw)
  ymis = min(cm_mi_rows - mi_row, bh)
  block_index = (mi_row * cm_mi_cols + mi_col) as usize

  refresh_this_block = candidate_refresh_aq(mi, rate, dist, bsize)
                       // §5.4.1

  // Map default: keep existing value.
  new_map_value = cr_map_slice[<map relative-offset of block_index>]

  // Skin-block override (line 178-184). Force BOOST1 if skin AND
  // candidate said BASE AND small block AND skin-detection enabled.
  if refresh_this_block == CR_SEGMENT_ID_BASE
     && bsize <= BLOCK_16X16
     && cfg.use_skin_detection:
    if (skin_probe)(p0.src, p1.src, p2.src, p0.stride, p1.stride, bsize, 0, 0):
      refresh_this_block = CR_SEGMENT_ID_BOOST1   // libvpx writes 1 here

  // VBR golden-frame: do not refresh blocks that already used GOLDEN ref.
  if cfg.rc_mode == VBR && mi.ref_frame[0] == GoldenFrame:
    refresh_this_block = CR_SEGMENT_ID_BASE

  // Segment-id reset on skip (line 191-196).
  if use_nonrd_pick_mode && segment_id_boosted(mi.segment_id):
    mi.segment_id = refresh_this_block as u8
    if skip:
      mi.segment_id = CR_SEGMENT_ID_BASE

  // Map state update (line 202-212).
  if segment_id_boosted(mi.segment_id):
    new_map_value = 0    // libvpx writes `-cr->time_for_refresh = 0`
                         // (L-03: field dropped; constant 0 baked in)
  elif refresh_this_block != CR_SEGMENT_ID_BASE:
    if cr_map_slice[block_index] == 1:
      new_map_value = 0
  else:
    new_map_value = 1

  // Write back for every 8x8 covered by this MI (line 216-221).
  for y in 0..ymis:
    for x in 0..xmis:
      let off = block_index + (y as usize) * cm_mi_cols as usize + x as usize
      cr_map_slice[off]   = new_map_value
      seg_map_slice[off]  = mi.segment_id
```

#### 5.4.1 `candidate_refresh_aq`

Mirrors the static `candidate_refresh_aq`
(`vp9_aq_cyclicrefresh.c:67-87`):

```text
candidate_refresh_aq(mi, rate, dist, bsize) -> u8:
  mv = mi.mv[0]
  if dist > self.thresh_dist_sb
     && ( mv.row.abs() > self.motion_thresh
       || mv.col.abs() > self.motion_thresh
       || !mi.is_inter_block() ):
    return CR_SEGMENT_ID_BASE
  if  bsize >= BLOCK_16X16
   && rate < self.thresh_rate_sb
   && mi.is_inter_block()
   && mi.mv[0] == MotionVector::ZERO
   && self.rate_boost_fac > 10:
    return CR_SEGMENT_ID_BOOST2
  return CR_SEGMENT_ID_BOOST1
```

Notes:

- `motion_thresh` is `int16_t` in libvpx and the comparison uses
  `> motion_thresh || < -motion_thresh` form; we use `.abs() >`
  which is identical because `MotionVector::row`/`col` is `i16`
  (libvpx `MV.row`/`MV.col` is also `int16_t`). The libvpx form
  guards against `mv.row == i16::MIN` (whose abs() in two's complement
  is undefined behaviour); since libvpx clamps MVs at search time to
  `[MV_LOW, MV_UPP]` (`MV_LOW = i16::MIN + 1`), the difference is
  unreachable. We use `.wrapping_abs()` to make the absence of UB
  explicit.

### 5.5 `rc_bits_per_mb` — pre-encode bits-per-MB

Mirrors `vp9_cyclic_refresh_rc_bits_per_mb`
(`vp9_aq_cyclicrefresh.c:137-155`).

```text
rc_bits_per_mb(frame_type, q, correction_factor, bit_depth, speed) -> u32:
  // Line 143-146.
  deltaq = if speed < 8 {
    compute_deltaq(q, self.rate_ratio_qdelta)         // §5.3
  } else {
    -(self.max_qdelta_perc * q) / 200                 // fast-path
  }

  // Line 148-153. Weighted average of base-Q and (Q+delta) bits-per-MB.
  base_bpm  = rc_bits_per_mb(frame_type, q,           correction_factor,
                              bit_depth)              // from rate-control.md §4.1
  boost_bpm = rc_bits_per_mb(frame_type, q + deltaq,  correction_factor,
                              bit_depth)
  // Order matters: weighted-average must be computed as
  //   (1.0 - w) * base + w * boost
  // libvpx writes it in exactly that order; do not reorder.
  ((1.0 - self.weight_segment) * base_bpm as f64
   + self.weight_segment * boost_bpm as f64) as u32
```

The weighted-average is the single biggest BD-rate trap in the entire
LLD; see §7 "L-21 RESOLVED" for the divergence-ledger entry that
documents the operation order constraint.

### 5.6 `update_sb_postencode` — `last_coded_q_map` update

Mirrors `vp9_cyclic_refresh_update_sb_postencode`
(`vp9_aq_cyclicrefresh.c:224-254`):

```text
update_sb_postencode(last_q_slice, mi, mi_row, mi_col, bsize,
                    base_qindex, cm_mi_cols, cm_mi_rows):
  xmis = min(cm_mi_cols - mi_col, num_8x8_blocks_wide_lookup[bsize])
  ymis = min(cm_mi_rows - mi_row, num_8x8_blocks_high_lookup[bsize])
  block_index = mi_row * cm_mi_cols + mi_col

  for y in 0..ymis:
    for x in 0..xmis:
      let off = (block_index + y * cm_mi_cols + x) as usize
      let coded = clamp(
        base_qindex as i32 + self.qindex_delta[mi.segment_id as usize],
        0, MAXQ as i32) as u8
      if (!mi.is_inter_block() || !mi.skip) && mi.segment_id <= CR_SEGMENT_ID_BOOST2:
        last_q_slice[off] = coded
      elif mi.is_inter_block() && mi.skip && mi.segment_id <= CR_SEGMENT_ID_BOOST2:
        last_q_slice[off] = min(coded, last_q_slice[off])
```

The "min" branch (line 247-251) is for inter-skip blocks: they
weren't *actually* coded at `coded`, so they only update the map if
the new coded-Q is *lower* than the prior (i.e., better resolution at
that block).

### 5.7 `postencode` — frame-level bookkeeping

Mirrors `vp9_cyclic_refresh_postencode` (`vp9_aq_cyclicrefresh.c:260-316`).
Note that the `mi` pointer in libvpx steps through `mi_grid_visible`
with a stride-trick (`mi += 8` at the end of each row), reflecting the
hidden padding between rows in `mi_grid_visible`. The Rust port should
expose this stride via `cm.mi_stride` rather than hard-coding the +8.

```text
postencode(common, rc, cfg, svc, seg_map, mi_grid, resize_pending)
    -> PostEncodeDecision:
  self.actual_num_seg1_blocks = 0
  self.actual_num_seg2_blocks = 0
  low_content_frame = 0
  decision = PostEncodeDecision::default()

  for mi_row in 0..common.mi_rows:
    for mi_col in 0..common.mi_cols:
      let idx = (mi_row * common.mi_cols + mi_col) as usize
      let normalized = normalize_segment_id(seg_map[idx])
      if normalized == CR_SEGMENT_ID_BOOST1:
        self.actual_num_seg1_blocks += 1
      elif normalized == CR_SEGMENT_ID_BOOST2:
        self.actual_num_seg2_blocks += 1

      let mi = &mi_grid[mi_row * common.mi_stride + mi_col]
      let mv = mi.mv[0]
      if mi.is_inter_block() && mv.row.abs() < 16 && mv.col.abs() < 16:
        low_content_frame += 1

  // Golden-frame logic — non-SVC, non-explicit-refresh, non-gf-boost only
  // (line 289-315). NOTE: `cpi->ext_refresh_frame_flags_pending` is the
  // libwebrtc API hook (R-06 in scope but the field merely gates this
  // codepath; the trait callback itself does not interact here).
  if !svc.use_svc
     && !common.ext_refresh_frame_flags_pending
     && cfg.gf_cbr_boost_pct == 0:
    if resize_pending != 0:
      decision.force_golden_refresh = true
      decision.new_baseline_gf_interval = rc.baseline_gf_interval  // RC owns
      // ... set_golden_update is called by RC after we return
    let fraction_low = low_content_frame as f64
                       / (common.mi_rows * common.mi_cols) as f64
    self.low_content_avg = (fraction_low + 3.0 * self.low_content_avg) / 4.0
    if !decision.force_golden_refresh
       && refresh_golden_frame_pending
       && rc.frames_since_key > rc.frames_since_golden + 1:
      if fraction_low < 0.65 || self.low_content_avg < 0.6:
        decision.demote_golden_refresh = true
      self.low_content_avg = fraction_low       // reset for next interval

  decision
```

The "stride-trick" in libvpx (`mi += 8` at the end of each row) is a
consequence of `mi_grid_visible` being a flattened pointer-grid that
includes the right/bottom padding rings. The Rust port either:

- mirrors the trick by exposing `cm.mi_stride > cm.mi_cols`, or
- uses `mi_grid[mi_row * common.mi_stride + mi_col]` which is the
  same arithmetic written differently.

`frame-buffers.md` and `tile-threading.md` together pin
`cm.mi_stride`; this LLD only consumes it.

### 5.8 `set_golden_update` — GF interval picker

Mirrors `vp9_cyclic_refresh_set_golden_update`
(`vp9_aq_cyclicrefresh.c:319-333`):

```text
set_golden_update(rc, rc_mode) -> u32:
  // Default: 4 refresh cycles, capped at 40 frames (line 325-326).
  let mut interval = if self.percent_refresh > 0 {
    min(4 * (100 / self.percent_refresh), 40)
  } else {
    40
  } as u32

  if rc_mode == VBR:
    interval = 20

  if rc.avg_frame_low_motion < 50
     && rc.frames_since_key > 40
     && self.content_mode:
    interval = 10

  interval
```

### 5.9 `update_map` and `is_superblock_flat_static`

Mirrors `cyclic_refresh_update_map`
(`vp9_aq_cyclicrefresh.c:363-475`). Walks superblocks in raster order
starting at `self.sb_index`, marking each as a refresh candidate or
not, accumulating until `target_num_seg_blocks >= block_count` or a
full wrap. This is the only loop in the subsystem where the cost is
non-trivial (up to all SBs touched once); see §6 for SIMD discussion.

```text
update_map(common, cfg, noise, source, last_source, consec_zero_mv,
           seg_map):
  seg_map.fill(CR_SEGMENT_ID_BASE)
  sb_cols = (common.mi_cols + MI_BLOCK_SIZE - 1) / MI_BLOCK_SIZE       // MI_BLOCK_SIZE = 8
  sb_rows = (common.mi_rows + MI_BLOCK_SIZE - 1) / MI_BLOCK_SIZE
  sbs_in_frame = sb_cols * sb_rows
  block_count = self.percent_refresh * common.mi_rows * common.mi_cols / 100
  assert!(self.sb_index < sbs_in_frame)
  let mut i = self.sb_index
  self.target_num_seg_blocks = 0

  // Content-aware thresholds (line 385-399).
  let mut consec_zero_mv_thresh = if cfg.content != Screen { 100 } else { 0 }
  let mut qindex_thresh = if cfg.content == Screen {
    common.seg.get_qindex(CR_SEGMENT_ID_BOOST2, common.base_qindex)
  } else {
    common.seg.get_qindex(CR_SEGMENT_ID_BOOST1, common.base_qindex)
  }
  if noise.enabled && noise.level >= kMedium && self.content_mode:
    consec_zero_mv_thresh = 60
    qindex_thresh = max(
      common.seg.get_qindex(CR_SEGMENT_ID_BOOST1, common.base_qindex),
      common.base_qindex)

  // count_sel / count_tot drive the `reduce_refresh` decision (line 473).
  let (mut count_sel, mut count_tot) = (0i32, 0i32)

  loop:
    let sb_row = i / sb_cols
    let sb_col = i - sb_row * sb_cols
    let mi_row = sb_row * MI_BLOCK_SIZE
    let mi_col = sb_col * MI_BLOCK_SIZE

    // compute_content gating (line 415-418): require last_source
    // present and dimensions match.
    let mut compute_content = true
    if !self.content_mode
       || last_source.is_none()
       || last_source.unwrap().y_width != source.y_width
       || last_source.unwrap().y_height != source.y_height:
      compute_content = false

    let bl_index = mi_row * common.mi_cols + mi_col
    let xmis = min(common.mi_cols - mi_col, MI_BLOCK_SIZE)
    let ymis = min(common.mi_rows - mi_row, MI_BLOCK_SIZE)
    let consec_zero_mv_thresh_block =
      if noise.enabled && noise.level >= kMedium && (xmis <= 2 || ymis <= 2) {
        4
      } else {
        consec_zero_mv_thresh
      }

    let mut sum_map = 0
    for y in 0..ymis:
      for x in 0..xmis:
        let bl2 = bl_index + y * common.mi_cols + x
        if self.map[bl2] == 0:
          count_tot += 1
          if !self.content_mode
             || self.last_coded_q_map[bl2] > qindex_thresh
             || consec_zero_mv[bl2] < consec_zero_mv_thresh_block:
            sum_map += 1
            count_sel += 1
        elif self.map[bl2] < 0:
          self.map[bl2] += 1

    // Half-of-SB rule (line 449).
    if sum_map >= xmis * ymis / 2:
      let mut flat_static = false
      if compute_content && self.skip_flat_static_blocks:
        flat_static = is_superblock_flat_static(source, last_source, sb_row, sb_col)
      if !flat_static:
        for y in 0..ymis:
          for x in 0..xmis:
            seg_map[bl_index + y * common.mi_cols + x] = CR_SEGMENT_ID_BOOST1
        self.target_num_seg_blocks += xmis * ymis

    i += 1
    if i == sbs_in_frame: i = 0
    if self.target_num_seg_blocks >= block_count || i == self.sb_index: break

  self.sb_index = i
  self.reduce_refresh = false
  if cfg.content != Screen && count_sel < (3 * count_tot) >> 2:
    self.reduce_refresh = true


is_superblock_flat_static(source, last_source, sb_row, sb_col) -> bool:
  // vp9_aq_cyclicrefresh.c:335-355.
  src_y     = &source.y_plane[(sb_row << 6) * source.y_stride + (sb_col << 6) ..]
  let sse: u32;
  let source_variance = variance_64x64(src_y, source.y_stride,
                                       &VP9_VAR_OFFS, 0, &mut sse)
  if source_variance != 0:
    return false
  if let Some(last) = last_source:
    last_src_y = &last.y_plane[(sb_row << 6) * source.y_stride + (sb_col << 6) ..]
    let block_sad = sad_64x64(src_y, source.y_stride,
                               last_src_y, last.y_stride)
    return block_sad == 0
  false
```

Note: libvpx's `is_superblock_flat_static` passes `cpi->Last_Source`'s
**source** y_stride into the offset arithmetic for both pointers
(`vp9_aq_cyclicrefresh.c:349` uses `ystride` for the `last_src_y`
offset). This is a libvpx quirk that we must replicate for
byte-parity; the variances differ only when last and current source
have different strides, which they do not in non-resize frames. See
§7 ("`Last_Source` stride quirk preserved").

### 5.10 `limit_q` — screen-content frame-Q cap

Mirrors `vp9_cyclic_refresh_limit_q`
(`vp9_aq_cyclicrefresh.c:695-702`). Called from
`RateControl::adjust_q_cbr` only when
`cfg.content == Screen && cfg.aq_mode == CyclicRefresh` (L-24,
`rate-control.md` will assert this gate at the call site).

```text
limit_q(prev_frame_q, q):
  if self.percent_refresh > 0 && prev_frame_q - *q > 8:
    *q = prev_frame_q - 8
```

The cap is a *hard* clamp on frame-level Q *decreases* (Q drop > 8
indices per frame is blocked). It does NOT cap Q increases. The
condition `percent_refresh > 0` excludes the VBR-golden-refresh case
where CR is effectively dormant for that frame.

### 5.11 `reset_resize`

Mirrors `vp9_cyclic_refresh_reset_resize`
(`vp9_aq_cyclicrefresh.c:683-693`):

```text
reset_resize(mi_rows, mi_cols):
  self.map.fill(0)
  self.last_coded_q_map.fill(MAXQ)
  self.sb_index = 0
  // libvpx also writes:
  //   cpi->refresh_golden_frame = 1
  //   cpi->refresh_alt_ref_frame = 1
  //   cr->counter_encode_maxq_scene_change = 0
  // The first two are RC fields (not CR fields). Our `reset_resize`
  // takes a `RcRefreshFlags` out-param OR is wrapped by RC's resize
  // handler so the RC writes happen on the RC side. The third (counter
  // reset) belongs to CR and IS written here.
  self.counter_encode_maxq_scene_change = 0
```

The split is documented in §7 ("`reset_resize` no longer writes RC
fields directly").

## 6. SIMD dispatch

The AQ-3 subsystem owns **zero** SIMD kernels. It calls two kernels
already inventoried by HLD-SIMD:

1. **`variance_64x64`** — `vpx_variance64x64` per
   `notes/hld/02-simd-dispatch.md:540-569` (§3.6 Variance kernel). Used
   exclusively by `is_superblock_flat_static`. Scalar reference:
   `vpx_dsp/variance.c:160-178` (`VAR` macro). ISAs: scalar, sse2, avx2,
   neon.
2. **`sad_64x64`** — `vpx_sad64x64` per
   `notes/hld/02-simd-dispatch.md` §3.1 (SAD kernel, line 381-401). Used
   exclusively by `is_superblock_flat_static`. Scalar reference:
   `vpx_dsp/sad.c` `SADMXN` macro at line 60. ISAs: scalar, sse2, avx2,
   neon.

Both kernels are called with `stride = 0` (for the `VP9_VAR_OFFS`
reference) and (for the `last_source`) the libvpx `last.y_stride`. The
variance with `stride = 0` is identical to a constant-128 reference of
the same dimensions; we therefore do not need a special-cased kernel
for the zero-stride case.

The map walk in `update_map` (§5.9) is loop-bound on `sbs_in_frame`
(up to ~512 for 1080p) and per-SB on `xmis * ymis` (up to 64). The
per-SB inner loop does i8 increments and counter updates — vectorizable
in principle, but the cost is < 0.5% of frame encode time, so we ship
scalar only for v1 and revisit if profiling surfaces it. See §7
divergence entry on "AQ-3 ships no SIMD kernels".

`accumulate_seg_blocks` (the inner loop of `postencode`, §5.7) is
similarly scalar-only.

## 7. Deliberate divergences from libvpx

Each entry feeds `notes/divergence-ledger.md` (format per
ledger §1). Several are already in the ledger; cross-references given.

### 7.1 AQ modes 1 and 2 not ported (already in ledger)

**Where:** `rvp9_encoder::aq::cyclic_refresh` is the entire `aq`
module surface. There is no `aq::variance` (mode 1) or `aq::complexity`
(mode 2) sibling. **Status:** already documented at
`notes/divergence-ledger.md:49-67`. **Origin:** D-05 RESOLVED
(`notes/vp9-analysis/decisions-needed.md:119-122`),
`notes/vp9-analysis/05-rate-control.md:1817-1827` (A5 Q-RC-15).
**libvpx counterpart:** `vp9_aq_variance.c` (255 ll),
`vp9_aq_complexity.c` (228 ll) — both guarded by
`!CONFIG_REALTIME_ONLY` per A5 Q-RC-15 and the encoder dispatch
`vp9_encoder.c:4209-4239` (specialised setup callsite). At cpu-used 5-8
(our target, PLAN.md D1) neither is reachable. **Byte-parity impact:**
none in matched configs. **CE impact:** none. **Future plan:** out of
scope; revisit only if a non-realtime preset is ever added.

### 7.2 `time_for_refresh` field dropped (already in ledger)

**Where:** `CyclicRefresh` struct has no `time_for_refresh` field.
**Status:** already documented at
`notes/divergence-ledger.md:195-206`. **Origin:** L-03 RESOLVED
(`notes/vp9-analysis/decisions-needed.md:153-154`),
A5 Q-RC-24 (`notes/vp9-analysis/05-rate-control.md:1916-1922`).
**libvpx counterpart:** `cr->time_for_refresh`
(`vp9_aq_cyclicrefresh.h:43`) — declared and written to 0 in
`update_parameters:509`, read only in `update_segment:203` (where the
write is `new_map_value = -cr->time_for_refresh` = `-0` = `0`).
**What rvp9 does:** the field is omitted from the struct; the
`new_map_value = 0` write in `update_segment` is hard-coded (§5.4).
**Byte-parity impact:** none — `-0i8 == 0i8`; libvpx writes `0` and so
do we. **Future plan:** none; if an external API ever needed
`time_for_refresh` for tuning, a separate `cycle_extra_wait_frames:
u8` field could be added without revisiting this divergence.

### 7.3 `weight_segment` integer-shift order preserved (L-21 RESOLVED) — NEW ledger entry

**Where:** §5.2, the `weight_segment` computation block. **Origin:**
L-21 (`notes/vp9-analysis/decisions-needed.md:211-212`), A5 Q-RC-22
(`notes/vp9-analysis/05-rate-control.md:1885-1898`). **libvpx
counterpart:** `vp9_aq_cyclicrefresh.c:572-575`,
`weight_segment = (double)((target + s1 + s2) >> 1) / num8x8bl` —
i.e., integer `>> 1` on the i32 sum BEFORE the cast to double. **What
rvp9 does:** preserve the integer shift exactly (i32 sum → i32 shift
→ `as f64 / num8x8bl as f64`). Do NOT rewrite as
`(target + s1 + s2) as f64 * 0.5 / num8x8bl as f64`. **Why:** the i32
`>> 1` discards the low bit before division; re-ordering changes
`weight_segment` by up to `0.5 / num8x8bl` and compounds across frames
into 0.1-0.3% BD-rate drift (A5 Q-RC-22). **Byte-parity:** none.
**CE impact:** none. **Future plan:** revisit only if `weight_segment`
is replaced with a deterministic alternative.

### 7.4 `vp9_cyclic_refresh_limit_q` only active for screen content (L-24 RESOLVED)

**Where:** the call site is `rate-control.md`'s `adjust_q_cbr`; this
LLD's `limit_q` is unconditional. **Origin:** L-24
(`notes/vp9-analysis/decisions-needed.md:221-222`), A5 Q-RC-29
(`notes/vp9-analysis/05-rate-control.md:1961-1965`). **What libvpx
does:** at `vp9_ratectrl.c:683-685`, `vp9_cyclic_refresh_limit_q` is
called only when `cpi->oxcf.content == VP9E_CONTENT_SCREEN &&
cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ`. **What rvp9 does:** the call
site in `rate-control.md` is gated by the exact same predicate;
`CyclicRefresh::limit_q` is callable in any context, but the only
call site enforces the gate. The function's own internal check
(`self.percent_refresh > 0`) further filters out the VBR-golden case.
**Byte-parity impact:** none. **CE impact:** none. **Future plan:**
none. **Ledger status:** **NEW** entry — needed because the gate is a
load-bearing assumption (without it, non-screen-content CR would
*also* clamp frame-level Q decreases, changing the BD-rate). The
ledger entry documents the gate to prevent silent regression.

### 7.5 `PostEncodeDecision` returned, not mutated through `cpi` pointer

**Where:** `CyclicRefresh::postencode` returns a `PostEncodeDecision`
struct; the caller (`rate-control.md`) applies the fields to
`RateControl`. **libvpx counterpart:**
`vp9_cyclic_refresh_postencode` mutates `cpi->refresh_golden_frame`
and `rc->baseline_gf_interval` directly (`:295`, `:298`, `:310`).
**Why:** Rust ownership; `CyclicRefresh` does not own `RateControl`.
This is purely a structural rewrite — the values written are
identical. **Byte-parity impact:** none. **CE impact:** none.
**Future plan:** none. **Ledger status:** NEW entry — minor structural
divergence worth recording so a reviewer doesn't think the missing
mutations are a bug.

### 7.6 `vpx_clear_system_state` dropped from `setup`

**Where:** §5.3, beginning of the active-CR branch.
**libvpx counterpart:** `vp9_aq_cyclicrefresh.c:625` calls
`vpx_clear_system_state()` (clears x87 FPU control word leak). **What
rvp9 does:** no equivalent call. **Why:** Rust uses SSE2-only float
arithmetic; there is no x87 FPU control word leak between the entropy
coder and the AQ float math. **Byte-parity impact:** none (libvpx's
clear is a no-op on the SSE2 build path; the only architecture where
it matters is 32-bit x86 with x87, which we don't target).
**CE impact:** none. **Future plan:** none. **Ledger status:** NEW
entry.

### 7.7 `Last_Source` stride quirk preserved in `is_superblock_flat_static`

**Where:** §5.9 `is_superblock_flat_static`. **libvpx counterpart:**
`vp9_aq_cyclicrefresh.c:349` uses `ystride` (the *current* source's
stride) for the offset arithmetic into `Last_Source->y_buffer`. This is
a latent bug in libvpx that only triggers when source and last-source
strides differ (impossible on non-resize frames). **What rvp9 does:**
replicate the quirk; use `source.y_stride` for both offsets. **Why:**
byte-parity. The downstream BD-rate impact is zero in any in-spec
input. **Future plan:** fix in a future version with a divergence
entry that flips the sign.

### 7.8 AQ-3 ships no SIMD kernels

**Where:** §6 above. **libvpx counterpart:** none — libvpx itself
ships no SIMD in `vp9_aq_cyclicrefresh.c`. The `vpx_variance64x64` and
`vpx_sad64x64` SIMD kernels we re-use are already inventoried by
HLD-SIMD. **What rvp9 does:** no SIMD added for `update_map` or
`postencode` loops in v1. **Why:** scalar cost is < 0.5% of frame
encode; not worth the additional kernel surface in v1. **Future
plan:** revisit if profiling surfaces these loops. **Ledger status:**
NEW (minor; purely additive).

## 8. Open questions

### 8.1 Skin-detection function trait shape (non-blocking)

`update_segment` takes `skin_probe: Option<SkinProbeFn>` where
`SkinProbeFn` is a function pointer / `fn(...)` type. libvpx's call is
`vp9_compute_skin_block(...)` which lives in `vp9_skin_detection.c`
(not in scope of any LLD yet; will land via `mode-decision.md` or a
dedicated `skin-detection.md`). **Question:** does
`skin-detection.md`/`mode-decision.md` expose the skin probe as a
free function with the libvpx signature, or as a closure captured by
`mode-decision`? **Recommendation:** free function with `extern "C"`-
compatible signature (`fn`, not `Fn`) so this LLD can take a function
pointer with no allocation. **Resolved-by:** the LLD that ships skin
detection; this LLD's API accepts whichever shape is chosen.

### 8.2 `mi_grid_visible` stride exposure (non-blocking)

§5.7 reads `mi_grid` with `cm.mi_stride`. **Question:** does
`frame-buffers.md` expose `mi_stride` as `EncoderCommon.mi_stride: u32`
or as a method? **Recommendation:** field for simplicity; the libvpx
type uses a bare field. **Resolved-by:** `frame-buffers.md`.

### 8.3 `RcRefreshFlags` out-param for `reset_resize`

§5.11 mentions that `reset_resize` historically writes
`cpi->refresh_golden_frame = 1` and `cpi->refresh_alt_ref_frame = 1`.
**No-altref policy:** the alt-ref-frame write is dropped (D-13). The
golden-refresh write still belongs to RC. **Question:** should
`reset_resize` return a `RcRefreshFlags { golden: bool }` value, or
should the caller (also on the RC side) write directly after the call?
**Recommendation:** return the struct, mirroring §7.5 ("explicit
return, no remote mutation"). Resolved at LLD-merge time with
`rate-control.md`.

### 8.4 Tile-disjoint slice access for `cr.map` writes

§3.2 / §5.4 require workers to obtain `&mut [i8]` slices into
`self.map` that are tile-disjoint. **Question:** does
`tile-threading.md` expose a `TileMapGuard<'a, T>` that gives
`&mut [T]` for the worker's tile's mi rows, or do we cast through
raw pointers? **Recommendation:** match the pattern HLD-threading §6.3
adopts for `cpi->segmentation_map` — strategy C with `UnsafeCell` plus
documented per-tile-row aliasing contract. **Resolved-by:**
`tile-threading.md`; this LLD's API is compatible with either choice
(it takes `&mut [i8]` and `&mut [u8]` slices).

### 8.5 `update_parameters` and SVC layer-context swap interaction

Per `notes/hld/04-svc.md:476-491`, the SVC layer-context restore /
save swaps `cr.map` / `cr.last_coded_q_map` / `cr.consec_zero_mv` with
the active layer's `LayerContext` between frame encodes. **Question:**
in what order, relative to `update_parameters`, does this swap
happen? **Likely answer:** `restore_layer_context` happens FIRST
(re-establishing per-layer state), then `update_parameters` (which
reads the post-swap state). **Resolved-by:** `svc.md` / `rate-control.md`
when those LLDs land; this LLD's `update_parameters` is order-agnostic
as long as the inputs reflect the active layer.

## 9. Test strategy

### 9.1 Unit tests (scalar / no SIMD dependence)

| Test | Asserts |
|---|---|
| `cr_alloc_zero_initialised_correctly` | post-`new` invariants (`map==0`, `last_coded_q_map==MAXQ`, `sb_index==0`, `content_mode==true`). |
| `cr_apply_gates` | parameterised: intra-only / non-base TL / low avg-Q / high avg-Q post-KF / lossless → `apply_cyclic_refresh==false`. |
| `cr_apply_enabled_on_normal_inter` | sanity check the negative case. |
| `cr_screen_top_layer_sets_skip_flat_static` | top spatial layer → `skip_flat_static_blocks==true`; lower → false. |
| `cr_rate_ratio_qdelta_schedule` | parameterised: first ~4 cycles → 3.0; later → 2.0; noisy → 1.7; VBR → 1.5; VBR-golden → 1.0. |
| `cr_low_res_low_rate_bumps_motion_thresh` | 352×288, `avg_frame_bandwidth<3000` → `motion_thresh==64`. |
| `cr_weight_segment_integer_shift_preserved` | hand-computed expected value matching `(target+s1+s2) >> 1 as f64 / n8`. **L-21 regression guard.** |
| `cr_weight_segment_target_lt_7_8_clamps` | `weight_segment_target = 0.5 * weight_segment` → clamp branch fires. |
| `cr_segment_id_boosted_helpers` | BOOST1/BOOST2 true; BASE / out-of-range false. |
| `cr_compute_deltaq_clamps_at_max_qdelta_perc` | raw delta below `-max_qdelta_perc * q / 100` → clamped. |
| `cr_limit_q_caps_decrease_only` | drop > 8 → clamp; rise → unchanged; `percent_refresh==0` → unchanged. |
| `cr_postencode_counts_actual_seg_blocks_exactly` | populate `seg_map` with known mix; assert counts. |
| `cr_postencode_demotes_golden_when_low_content` | covers both legs of the `fraction_low < 0.65 \|\| low_content_avg < 0.6` branch. |
| `cr_set_golden_update_branches` | parameterised: CBR default / VBR / low-motion post-KF (returns 4*100/pr capped 40 / 20 / 10). |
| `cr_reset_resize_clears_map` | post-call invariants. |
| `cr_update_segment_state_machine` | parameterised: skin override → BOOST1; skip with prior BOOST1 → BASE; non-RD-path reset semantics; map encoding {1,0,-N} transitions per §4.2. |

### 9.2 Property tests

`proptest` strategies:

1. **`update_map` is idempotent under wrap-around.** Run
   `cyclic_refresh_update_map` until `sb_index` returns to its
   starting value; assert all `map` positions have been visited
   exactly once per cycle (modulo skipped flat-static blocks).
2. **`weight_segment` lies in `[0, 1]`.** For random
   `actual_num_seg{1,2}_blocks` in `[0, mi_rows * mi_cols]` and
   random `percent_refresh ∈ [1, 100]`, after `update_parameters`,
   `0.0 <= weight_segment <= 1.0`.
3. **`qindex_delta[1]` and `[2]` are non-positive** for any
   `base_qindex ∈ [1, MAXQ]`, any valid `rate_ratio_qdelta`,
   `rate_boost_fac`, `max_qdelta_perc`. (Boost == lower Q == negative
   delta.)
4. **`qindex_delta[2] <= qindex_delta[1]`** (BOOST2 is more
   aggressive than BOOST1) for any valid tuning combination where both
   are computed.

### 9.3 Differential tests against libvpx (Stage 4b gate)

The gating differential tests for AQ-3 — covered by `coverage` /
`bench/bdrate.rs` plumbing in PLAN.md Stage 4b §exit-criteria — are:

1. **Per-frame state vector**: dump `CyclicRefresh` post-`setup` from
   both libvpx (via patched build) and rvp9 on a fixed 100-frame
   clip; compare:
   - `apply_cyclic_refresh` (bool)
   - `percent_refresh, max_qdelta_perc, motion_thresh,
     rate_boost_fac, rate_ratio_qdelta, skip_flat_static_blocks`
   - `qindex_delta[0..3]` (exact)
   - `thresh_rate_sb, thresh_dist_sb` (exact)
   - `weight_segment` (within 1e-12)
   - `sb_index`, `target_num_seg_blocks` (exact)

2. **Segmentation map differential**: after each frame's `setup +
   encode + postencode`, diff `cpi->segmentation_map` and rvp9's
   `seg_map`. Required identical on the matched test clip.

3. **`last_coded_q_map` differential**: after each frame, diff.
   Required identical.

4. **`actual_num_seg{1,2}_blocks` differential**: per-frame, exact
   equality.

5. **BD-rate gate**: AQ-3 on vs AQ-3 off, the *gain* should match
   libvpx's gain to within 0.3% per PLAN.md Stage 4b. This is the
   primary BD-rate trap for L-21 (`weight_segment`): a silent shift
   manifests here as a 0.1-0.3% widening.

### 9.4 Conformance / interop

AQ-3 is encoder-only; conformance vectors do not exercise it. Interop
testing: rvp9-encoder with AQ-3 on must produce streams that decode
under (a) rvp9-decoder, (b) `vpxdec`, (c) `ffmpeg`. This is part of
the `tests/conformance/runner.rs` cross-product (PLAN.md §2).

### 9.5 Multi-thread safety tests

`update_segment` and `update_sb_postencode` are the only worker-side
mutations. The Miri test in `tile-threading.md`'s test plan covers
the per-block-disjoint slice aliasing; this LLD does not add separate
Miri tests for them (the aliasing pattern is generic, not
CR-specific).

---

**Summary:** ~1430 lines; **§7 entries:** (7.1) AQ modes 1/2 not ported
[already in ledger], (7.2) `time_for_refresh` dropped [already in
ledger], (7.3) `weight_segment` integer-shift order frozen [NEW — L-21
regression guard], (7.4) `limit_q` screen-content-only gate preserved
[NEW — L-24], (7.5) explicit `PostEncodeDecision` return [NEW —
structural], (7.6) `vpx_clear_system_state` dropped [NEW], (7.7)
`Last_Source` stride quirk preserved [NEW — bug-for-bug parity], (7.8)
AQ-3 ships no SIMD kernels [NEW]; **§8 open questions:** five
non-blocking (skin-probe signature, `mi_stride` field, `RcRefreshFlags`
out-param, tile-disjoint slice pattern, SVC swap ordering) — none
block Stage 4b start.
