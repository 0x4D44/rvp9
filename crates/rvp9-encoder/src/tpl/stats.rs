//! W6.7 — TPL data structures + per-frame init.
//!
//! Faithful port of libvpx's per-block / per-frame "Temporal-dependency
//! Model" (TPL) state. The TPL machinery propagates a backward
//! mc_dep_cost from leaf frames up to the GOP's golden anchor, so the
//! RDO picker can scale `rdmult` per-superblock by how much downstream
//! distortion depends on the block being coded faithfully (a leaf-only
//! block can be coded cheaply; a block heavily referenced by future
//! frames should be coded with finer rdmult).
//!
//! ## libvpx references
//!
//! - `TplDepStats` struct — `vp9/encoder/vp9_encoder.h:292-301`
//!   (v1.14.1 commit `12f3a2ac6`).
//! - `TplDepFrame` struct — `vp9/encoder/vp9_encoder.h:312-326`.
//! - `init_gop_frames` — `vp9/encoder/vp9_tpl_model.c:25-155`.
//! - `init_tpl_stats` — `vp9/encoder/vp9_tpl_model.c:157-166`.
//! - `TPL_DEP_COST_SCALE_LOG2` — `vp9/encoder/vp9_encoder.h:328`.
//! - `MAX_ARF_GOP_SIZE = 2 * MAX_LAG_BUFFERS = 50`
//!   (`vp9_encoder.h:515`, `vp9_lookahead.h:22`).
//! - `MAX_ARF_LAYERS = 6` — `vp9/encoder/vp9_firstpass.h:27`.
//! - `MI_SIZE = 8` — `vp9/common/vp9_enums.h:24`.
//!
//! ## Scope (W6.7)
//!
//! - Pure data layout + zero-init. No motion search, no propagation —
//!   those are W6.8 ([`super::update`]) and W6.9+ (the cost estimator
//!   that drives `mode_estimation`).
//! - 8-bit only. HBD pieces of libvpx's TPL pipeline (e.g.
//!   `vp9_highbd_wht_fwd_txfm`) are skipped.
//! - Single-frame init paths only. The `init_gop_frames` port is a
//!   skeleton that records frame topology (which GOP slot references
//!   which others); it does NOT pull from a lookahead queue —
//!   integration with [`crate::lookahead`] is a later wave.

use rvp9_core::mode_info::Mv;

/// libvpx `MI_SIZE` — pixels per "mode-info" unit (8).
/// Mirrors `vp9/common/vp9_enums.h:24`.
pub const MI_SIZE: usize = 8;

/// libvpx `MI_SIZE_LOG2` — `log2(MI_SIZE) = 3`.
/// Mirrors `vp9/common/vp9_enums.h:21`.
pub const MI_SIZE_LOG2: u32 = 3;

/// libvpx `TPL_DEP_COST_SCALE_LOG2 = 4`. All `intra_cost` /
/// `inter_cost` values written by `mode_estimation` are pre-multiplied
/// by `1 << TPL_DEP_COST_SCALE_LOG2` so that the propagation step
/// (which divides by `mi_height * mi_width`) keeps integer headroom.
/// Mirrors `vp9/encoder/vp9_encoder.h:328`.
pub const TPL_DEP_COST_SCALE_LOG2: u32 = 4;

/// libvpx `MAX_LAG_BUFFERS = 25`.
/// Mirrors `vp9/encoder/vp9_lookahead.h:22`.
pub const MAX_LAG_BUFFERS: usize = 25;

/// libvpx `MAX_ARF_GOP_SIZE = 2 * MAX_LAG_BUFFERS = 50`.
/// Mirrors `vp9/encoder/vp9_encoder.h:515`. This is the maximum
/// number of `TplDepFrame`s the TPL state preallocates.
pub const MAX_ARF_GOP_SIZE: usize = 2 * MAX_LAG_BUFFERS;

/// libvpx `MAX_ARF_LAYERS = 6`.
/// Mirrors `vp9/encoder/vp9_firstpass.h:27`.
pub const MAX_ARF_LAYERS: usize = 6;

/// libvpx `REFS_PER_FRAME = 3` (LAST, GOLDEN, ALTREF).
/// Mirrors `vp9/common/vp9_onyxc_int.h:35`.
pub const REFS_PER_FRAME: usize = 3;

/// libvpx `MAX_INTER_REF_FRAMES`. The TPL `gf_picture[].ref_frame[]`
/// is indexed `[0..3]`, matching `REFS_PER_FRAME`.
pub const MAX_INTER_REF_FRAMES: usize = REFS_PER_FRAME;

/// Sentinel for "no ref-frame index assigned". libvpx initialises
/// `gf_picture[*].ref_frame[*]` and the various per-frame ARF/LAST/
/// GLD indices to `-1` (`vp9_tpl_model.c:30-32, 73-74`).
pub const NO_REF_FRAME_INDEX: i32 = -1;

/// Per-block (8×8 mi-unit-resolution) dependency stats.
///
/// Mirrors libvpx `TplDepStats` (`vp9/encoder/vp9_encoder.h:292-301`):
///
/// ```c
/// typedef struct TplDepStats {
///   int64_t intra_cost;
///   int64_t inter_cost;
///   int64_t mc_flow;
///   int64_t mc_dep_cost;
///   int64_t mc_ref_cost;
///
///   int ref_frame_index;
///   int_mv mv;
/// } TplDepStats;
/// ```
///
/// libvpx stores `int_mv` (a packed union of `MV` + `int32_t`); rvp9
/// uses [`rvp9_core::mode_info::Mv`] which already wraps the same
/// `row`/`col` pair as Q3 sub-pixel units.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TplDepStats {
    /// Best intra-prediction transform cost (SATD, pre-scaled by
    /// `1 << TPL_DEP_COST_SCALE_LOG2`). See
    /// `vp9_tpl_model.c:722-723`.
    pub intra_cost: i64,
    /// Best inter-prediction transform cost (SATD, pre-scaled). See
    /// `vp9_tpl_model.c:720-721`.
    pub inter_cost: i64,
    /// Forward-flow accumulator — set by [`super::update::tpl_model_update`]
    /// when a downstream frame's MV points back into this block.
    /// `vp9_tpl_model.c:464`.
    pub mc_flow: i64,
    /// `intra_cost + mc_flow`. Recomputed by `tpl_model_store`
    /// (`vp9_tpl_model.c:387`).
    pub mc_dep_cost: i64,
    /// Accumulator of `(intra_cost - inter_cost) * overlap_area /
    /// pix_num` contributions from downstream frames. Used by the RD
    /// model's "how much rate did we save by referencing this block?"
    /// statistic. `vp9_tpl_model.c:465-467`.
    pub mc_ref_cost: i64,
    /// Index (into the surrounding `TplDepFrame` array) of the
    /// best-matching reference frame slot, or
    /// [`NO_REF_FRAME_INDEX`]. Written by `mode_estimation`
    /// (`vp9_tpl_model.c:724`).
    pub ref_frame_index: i32,
    /// Best-matching MV (Q3 sub-pel units, matching libvpx's
    /// `int_mv`). Written by `mode_estimation`
    /// (`vp9_tpl_model.c:725`).
    pub mv: Mv,
}

impl TplDepStats {
    /// Fresh, all-zero stats. The libvpx equivalent is the bulk
    /// `memset(tpl_stats_ptr, 0, ...)` in `init_tpl_stats`
    /// (`vp9_tpl_model.c:161-163`).
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Self {
            intra_cost: 0,
            inter_cost: 0,
            mc_flow: 0,
            mc_dep_cost: 0,
            mc_ref_cost: 0,
            ref_frame_index: NO_REF_FRAME_INDEX,
            mv: Mv { row: 0, col: 0 },
        }
    }
}

/// Per-frame TPL state. A flattened 2-D array of [`TplDepStats`]
/// indexed `mi_row * stride + mi_col`, plus the frame dimensions
/// needed to walk it.
///
/// Mirrors libvpx `TplDepFrame` (`vp9/encoder/vp9_encoder.h:312-326`):
///
/// ```c
/// typedef struct TplDepFrame {
///   uint8_t is_valid;
///   TplDepStats *tpl_stats_ptr;
///   int stride;
///   int width;
///   int height;
///   int mi_rows;
///   int mi_cols;
///   int base_qindex;
/// } TplDepFrame;
/// ```
///
/// libvpx's `width` / `height` fields hold the 8×8-unit grid extent
/// (i.e. `mi_cols` / `mi_rows`); we keep both names for citation
/// fidelity but the per-block stats are sized `mi_rows * stride`.
#[derive(Clone, Debug)]
pub struct TplDepFrame {
    /// Flat backing storage for per-mi-unit stats, length
    /// `mi_rows * stride`. Indexed `[mi_row * stride + mi_col]`.
    pub tpl_stats: Vec<TplDepStats>,
    /// Row stride in 8×8 units. Mirrors libvpx `tpl_frame->stride`.
    pub stride: usize,
    /// libvpx `width` field — 8×8-unit grid width (same as `mi_cols`).
    pub width: usize,
    /// libvpx `height` field — 8×8-unit grid height (same as `mi_rows`).
    pub height: usize,
    /// 8×8-unit rows in the frame. Mirrors libvpx `tpl_frame->mi_rows`.
    pub mi_rows: usize,
    /// 8×8-unit cols in the frame. Mirrors libvpx `tpl_frame->mi_cols`.
    pub mi_cols: usize,
    /// Per-frame base qindex used by `get_rdmult_delta` /
    /// `vp9_get_adaptive_rdmult`. Mirrors libvpx
    /// `tpl_frame->base_qindex` (`vp9_tpl_model.c:36`).
    pub base_qindex: i32,
    /// `is_valid` flag — false until `mc_flow_dispenser` has populated
    /// this slot, gates the picker's rdmult adjustment
    /// (`vp9_encodeframe.c:3738`).
    pub is_valid: bool,
}

impl TplDepFrame {
    /// Allocate a fresh per-frame TPL state for an `mi_rows × mi_cols`
    /// grid. Stats start at [`TplDepStats::zero`] and `is_valid` is
    /// false.
    ///
    /// Mirrors libvpx's per-frame allocation pattern: `vp9_init_tpl_buffer`
    /// allocates `mi_rows * mi_cols` slots and `init_tpl_stats` zeroes
    /// them (`vp9_tpl_model.c:157-166`).
    #[must_use]
    pub fn new(mi_rows: usize, mi_cols: usize) -> Self {
        let stride = mi_cols;
        let tpl_stats = vec![TplDepStats::zero(); mi_rows * stride];
        Self {
            tpl_stats,
            stride,
            width: mi_cols,
            height: mi_rows,
            mi_rows,
            mi_cols,
            base_qindex: 0,
            is_valid: false,
        }
    }

    /// Zero every per-block stat slot and clear `is_valid`. Mirrors
    /// libvpx `init_tpl_stats` (`vp9_tpl_model.c:157-166`).
    pub fn reset(&mut self) {
        for s in &mut self.tpl_stats {
            *s = TplDepStats::zero();
        }
        self.is_valid = false;
    }

    /// Fetch the stat at `(mi_row, mi_col)`. Bounds-checked debug
    /// assert mirrors the libvpx `assert(...)` pattern inside the
    /// propagation loop.
    #[inline]
    #[must_use]
    pub fn stat(&self, mi_row: usize, mi_col: usize) -> &TplDepStats {
        debug_assert!(mi_row < self.mi_rows);
        debug_assert!(mi_col < self.mi_cols);
        &self.tpl_stats[mi_row * self.stride + mi_col]
    }

    /// Mutable variant of [`stat`](Self::stat).
    #[inline]
    pub fn stat_mut(&mut self, mi_row: usize, mi_col: usize) -> &mut TplDepStats {
        debug_assert!(mi_row < self.mi_rows);
        debug_assert!(mi_col < self.mi_cols);
        let stride = self.stride;
        &mut self.tpl_stats[mi_row * stride + mi_col]
    }
}

/// Per-GOP-slot frame metadata. Records which frames in the GOP
/// reference which others, mirroring libvpx `GF_PICTURE`
/// (`vp9/encoder/vp9_tpl_model.h:25-29`):
///
/// ```c
/// typedef struct GF_PICTURE {
///   YV12_BUFFER_CONFIG *frame;
///   int ref_frame[3];
///   FRAME_UPDATE_TYPE update_type;
/// } GF_PICTURE;
/// ```
///
/// The owned-pixel-buffer pointer is intentionally elided here — the
/// TPL scaffold does not own frame buffers, the caller threads them
/// in for `mode_estimation`. We keep the topology fields.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GfPicture {
    /// Slot indices in the surrounding GOP array for the (LAST,
    /// GOLDEN, ALTREF) references. `NO_REF_FRAME_INDEX` when not
    /// assigned. Mirrors libvpx `gf_picture[].ref_frame[3]`.
    pub ref_frame: [i32; MAX_INTER_REF_FRAMES],
    /// libvpx `FRAME_UPDATE_TYPE` for this slot — chained from
    /// `gf_group->update_type[frame_idx]`. We carry it as a plain
    /// `u8` (matching libvpx's enum width) until the rvp9 GF-group
    /// scaffold lands.
    pub update_type: u8,
}

impl GfPicture {
    /// Fresh slot with no references assigned, `update_type = 0` (the
    /// libvpx `KF_UPDATE` enumerant). Matches the implicit
    /// post-`memset` state of `gf_picture[]` in libvpx.
    #[inline]
    #[must_use]
    pub fn empty() -> Self {
        Self {
            ref_frame: [NO_REF_FRAME_INDEX; MAX_INTER_REF_FRAMES],
            update_type: 0,
        }
    }
}

/// FRAME_UPDATE_TYPE values used by the GOP scaffold. Mirrors libvpx
/// `vp9/encoder/vp9_firstpass.h:35-44` (`KF_UPDATE, LF_UPDATE,
/// GF_UPDATE, ARF_UPDATE, OVERLAY_UPDATE, USE_BUF_FRAME`).
#[allow(missing_docs)]
pub mod update_type {
    pub const KF_UPDATE: u8 = 0;
    pub const LF_UPDATE: u8 = 1;
    pub const GF_UPDATE: u8 = 2;
    pub const ARF_UPDATE: u8 = 3;
    pub const OVERLAY_UPDATE: u8 = 4;
    pub const USE_BUF_FRAME: u8 = 5;
}

/// Aggregate GOP-scope TPL state — the `MAX_ARF_GOP_SIZE` array of
/// per-frame stats libvpx hangs off `VP9_COMP::tpl_stats`.
///
/// Mirrors libvpx `cpi->tpl_stats[MAX_ARF_GOP_SIZE]`
/// (`vp9_encoder.h:752`). The owning struct also carries the GOP
/// `GF_PICTURE` array used by `init_gop_frames` /
/// `mc_flow_dispenser`.
#[derive(Debug)]
pub struct TplGopState {
    /// Per-frame stats; up to [`MAX_ARF_GOP_SIZE`] frames. Indices in
    /// `[0, tpl_group_frames)` are populated.
    pub frames: Vec<TplDepFrame>,
    /// Per-frame topology. `gf_picture[i].ref_frame[k]` is the GOP
    /// slot index of frame `i`'s `k`-th reference (LAST/GOLDEN/
    /// ALTREF), or [`NO_REF_FRAME_INDEX`].
    pub gf_picture: [GfPicture; MAX_ARF_GOP_SIZE],
    /// Number of populated frames (libvpx `tpl_group_frames`).
    pub tpl_group_frames: usize,
}

impl TplGopState {
    /// Allocate a fresh GOP-scope TPL state. `frames` is populated
    /// with `gop_frames` per-frame slots, each sized to the supplied
    /// `mi_rows × mi_cols` grid. `gf_picture` is all-empty.
    ///
    /// `gop_frames` must be ≤ [`MAX_ARF_GOP_SIZE`].
    ///
    /// Mirrors the libvpx allocation pattern: `vp9_init_tpl_buffer`
    /// allocates `MAX_ARF_GOP_SIZE` slots, then `init_gop_frames`
    /// stamps the topology and `init_tpl_stats` zeroes the payload.
    #[must_use]
    pub fn new(gop_frames: usize, mi_rows: usize, mi_cols: usize) -> Self {
        assert!(
            gop_frames <= MAX_ARF_GOP_SIZE,
            "TPL gop_frames {gop_frames} exceeds MAX_ARF_GOP_SIZE {MAX_ARF_GOP_SIZE}",
        );
        let frames = (0..gop_frames)
            .map(|_| TplDepFrame::new(mi_rows, mi_cols))
            .collect();
        Self {
            frames,
            gf_picture: [GfPicture::empty(); MAX_ARF_GOP_SIZE],
            tpl_group_frames: 0,
        }
    }

    /// Re-zero all per-block stats and clear all `is_valid` flags.
    /// Mirrors libvpx `init_tpl_stats` (`vp9_tpl_model.c:157-166`).
    pub fn reset_all_stats(&mut self) {
        for f in &mut self.frames {
            f.reset();
        }
    }
}

/// Skeleton port of libvpx `init_gop_frames` (`vp9_tpl_model.c:25-155`).
///
/// Sets up the per-slot `gf_picture[].ref_frame[]` topology for a
/// linear chain GOP: slot 0 is the golden anchor, slot 1 is the base
/// ARF (refers back to slot 0), slots 2..n are LF_UPDATE frames that
/// chain through `lst_index`.
///
/// **Deviations from libvpx (deliberate, W6.7 scope):**
///
/// - No lookahead queue, no `cm->buffer_pool` allocation. libvpx walks
///   `frame_bufs[]` to claim recon buffers; rvp9's scaffold runs on
///   externally-supplied frame buffers (the caller threads them into
///   [`super::mode_est::mode_estimation`] directly).
/// - No ARF stack — the W6.7 skeleton handles only LF_UPDATE chains.
///   Multi-ARF GOPs (the `ARF_UPDATE` / `OVERLAY_UPDATE` /
///   `USE_BUF_FRAME` branches at `vp9_tpl_model.c:102-119`) land when
///   the rvp9 GF-group scaffold catches up.
/// - No "extend two frames outside the GOP" tail
///   (`vp9_tpl_model.c:134-152`) — the caller supplies the exact
///   `gop_frames` count.
///
/// Returns the GF slot index of the last LF_UPDATE in the chain (the
/// libvpx `lst_index`, useful for sanity tests).
pub fn init_gop_frames(state: &mut TplGopState, gop_frames: usize) -> i32 {
    assert!(gop_frames <= state.frames.len());
    assert!(gop_frames <= MAX_ARF_GOP_SIZE);

    let mut gld_index: i32 = -1;
    let mut alt_index: i32 = -1;
    let mut lst_index: i32 = -1;

    // libvpx vp9_tpl_model.c:73-77: slot 0 is the golden anchor.
    if gop_frames > 0 {
        state.gf_picture[0] = GfPicture {
            ref_frame: [NO_REF_FRAME_INDEX; MAX_INTER_REF_FRAMES],
            update_type: update_type::GF_UPDATE,
        };
        gld_index = 0;
    }

    // libvpx vp9_tpl_model.c:80-86: slot 1 is the base-layer ARF.
    if gop_frames > 1 {
        state.gf_picture[1] = GfPicture {
            ref_frame: [gld_index, lst_index, alt_index],
            update_type: update_type::ARF_UPDATE,
        };
        alt_index = 1;
    }

    // libvpx vp9_tpl_model.c:88-128: linear P-frame chain. We model
    // the LF_UPDATE branch only (`vp9_tpl_model.c:108`).
    for frame_idx in 2..gop_frames {
        state.gf_picture[frame_idx] = GfPicture {
            ref_frame: [gld_index, lst_index, alt_index],
            update_type: update_type::LF_UPDATE,
        };
        lst_index = frame_idx as i32;
    }

    state.tpl_group_frames = gop_frames;
    lst_index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tpl_dep_stats_zero_is_zero() {
        let s = TplDepStats::zero();
        assert_eq!(s.intra_cost, 0);
        assert_eq!(s.inter_cost, 0);
        assert_eq!(s.mc_flow, 0);
        assert_eq!(s.mc_dep_cost, 0);
        assert_eq!(s.mc_ref_cost, 0);
        assert_eq!(s.ref_frame_index, NO_REF_FRAME_INDEX);
        assert_eq!(s.mv, Mv { row: 0, col: 0 });
    }

    #[test]
    fn tpl_dep_frame_new_sizes_correctly() {
        let f = TplDepFrame::new(9, 16);
        assert_eq!(f.mi_rows, 9);
        assert_eq!(f.mi_cols, 16);
        assert_eq!(f.stride, 16);
        assert_eq!(f.tpl_stats.len(), 9 * 16);
        assert!(!f.is_valid);
    }
}
