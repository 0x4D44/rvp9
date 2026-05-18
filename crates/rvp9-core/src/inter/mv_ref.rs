//! Decoder-side motion-vector reference resolution.
//!
//! Mirrors libvpx's `dec_find_mv_refs`,
//! `append_sub8x8_mvs_for_idx`, and `vp9_find_best_ref_mvs`
//! (`vp9_decodemv.c:482-668`, `vp9_mvref_common.{h,c}:213-300`).
//! The output is the set of candidate motion vectors that the
//! inter-mode decoder uses for `NEARESTMV` / `NEARMV` (filled
//! directly from the list) and for `NEWMV` (the `ref_mv` that
//! `read_mv` adds the decoded residual on top of).
//!
//! ## Decoder-only spec
//!
//! libvpx has two variants of `find_mv_refs` — a common-path one in
//! `vp9_mvref_common.c` and a decoder-specific
//! `dec_find_mv_refs` in `vp9_decodemv.c` which differs in three
//! ways:
//!
//! - `mode_context` is **not** filled in (the decoder builds it via
//!   a separate `get_mode_context` pass before reading the inter
//!   mode).
//! - `early_break` short-circuits the search for any mode other
//!   than `NEARMV` (the caller only needs `mv_list[0]`).
//! - The returned `refmv_count` is `MAX_MV_REF_CANDIDATES = 2` when
//!   mode is `NEARMV`, else `1` (so callers can index
//!   `mv_list[refmv_count - 1]` for "the best ref MV").
//!
//! This file mirrors the decoder variant exclusively. The encoder
//! variant is covered by `find_mv_refs_rt` in
//! `notes/lld/motion-estimation.md` §5.9, which lives elsewhere
//! (Phase D, encoder side).
//!
//! ## What this module does NOT cover
//!
//! - **Prev-frame MV salvage** (`cm->use_prev_frame_mvs`). libvpx
//!   pulls one extra candidate per ref from the co-located block in
//!   the previous frame when no spatial candidate was found. This
//!   path requires a per-frame `MV_REF` grid which v1 doesn't have
//!   yet (Phase D / Wave 3 K). The hook is exposed as the
//!   `prev_frame_mv` parameter; pass `None` until that lands.
//! - **`mode_context` array fill**. The decoder builds this via the
//!   separate `get_mode_context` pre-pass (libvpx
//!   `vp9_decodemv.c:670-689`), not via `find_mv_refs`. Callers
//!   compute it via a separate helper (not yet ported).
//!
//! ## libvpx call sites
//!
//! Parent (≥8×8) inter blocks:
//!
//! - `vp9_decodemv.c:776-781` (`mi->mode != ZEROMV` branch).
//! - For `NEARMV`, libvpx uses `tmp_mvs[refmv_count - 1]`
//!   (the second entry, with the "different" candidate after dedup).
//! - For `NEARESTMV`/`NEWMV`, `refmv_count == 1` and `tmp_mvs[0]`
//!   is the ref.
//! - For `ZEROMV`, the loop body is skipped entirely; `best_ref_mvs
//!   = {0, 0}`.
//!
//! Sub-8 inter blocks:
//!
//! - `vp9_decodemv.c:734-766`, one per 4×4 sub-block `j ∈ {0..3}`.
//! - `b_mode == NEARESTMV / NEARMV` → `append_sub8x8_mvs_for_idx`,
//!   which calls `dec_find_mv_refs` with `block = j`.
//! - `b_mode == NEWMV` → `dec_find_mv_refs` once (shared across all
//!   sub-blocks), `lower_mv_precision`, store in `best_ref_mvs`.

use crate::block::BlockSize;
use crate::inter::mv::use_mv_hp;
use crate::mode_info::{BlockMode, InterMode, ModeInfo, Mv, RefFrame};

/// libvpx `MAX_MV_REF_CANDIDATES` (`vp9_blockd.h:67`). The maximum
/// number of distinct candidate MVs `find_mv_refs` returns per
/// reference frame.
pub const MAX_MV_REF_CANDIDATES: usize = 2;

/// libvpx `MVREF_NEIGHBOURS` (`vp9_mvref_common.h:24`). Number of
/// neighbour positions scanned per block size.
pub const MVREF_NEIGHBOURS: usize = 8;

/// libvpx `MV_BORDER = 16 << 3` (`vp9_mvref_common.h:214`). Expressed
/// in 1/8-pel units. `clamp_mv_ref` extends the block's available
/// MV range by `±MV_BORDER` on all sides before clamping candidate
/// MVs.
pub const MV_BORDER: i32 = 16 << 3;

/// `MI_SIZE` (pixels per 8×8 MI unit), needed to convert mi-row /
/// mi-col into the 1/8-pel edge offsets `clamp_mv_ref` consumes.
const MI_SIZE: i32 = 8;

/// A neighbour position relative to the current block, in 8×8 MI
/// units. Mirrors libvpx `POSITION` (`vp9_mvref_common.h:26-29`).
///
/// `row` / `col` are *signed* offsets — negative values reach into
/// the already-decoded area above / to the left.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Position {
    /// Row offset in 8×8 MI units (negative = above current block).
    pub row: i8,
    /// Column offset in 8×8 MI units (negative = left of current).
    pub col: i8,
}

/// libvpx `mv_ref_blocks[BLOCK_SIZES][MVREF_NEIGHBOURS]`
/// (`vp9_mvref_common.h:89-207`). Per-block-size table of neighbour
/// positions to scan, in scan order. The first two entries are the
/// "nearest" pair used for sub-8 sub-block MVs; the remaining 6 are
/// fallback neighbours.
#[rustfmt::skip]
pub static MV_REF_BLOCKS: [[Position; MVREF_NEIGHBOURS]; 13] = [
    // 4X4
    [Position{row:-1,col:0}, Position{row:0,col:-1}, Position{row:-1,col:-1},
     Position{row:-2,col:0}, Position{row:0,col:-2}, Position{row:-2,col:-1},
     Position{row:-1,col:-2}, Position{row:-2,col:-2}],
    // 4X8
    [Position{row:-1,col:0}, Position{row:0,col:-1}, Position{row:-1,col:-1},
     Position{row:-2,col:0}, Position{row:0,col:-2}, Position{row:-2,col:-1},
     Position{row:-1,col:-2}, Position{row:-2,col:-2}],
    // 8X4
    [Position{row:-1,col:0}, Position{row:0,col:-1}, Position{row:-1,col:-1},
     Position{row:-2,col:0}, Position{row:0,col:-2}, Position{row:-2,col:-1},
     Position{row:-1,col:-2}, Position{row:-2,col:-2}],
    // 8X8
    [Position{row:-1,col:0}, Position{row:0,col:-1}, Position{row:-1,col:-1},
     Position{row:-2,col:0}, Position{row:0,col:-2}, Position{row:-2,col:-1},
     Position{row:-1,col:-2}, Position{row:-2,col:-2}],
    // 8X16
    [Position{row:0,col:-1}, Position{row:-1,col:0}, Position{row:1,col:-1},
     Position{row:-1,col:-1}, Position{row:0,col:-2}, Position{row:-2,col:0},
     Position{row:-2,col:-1}, Position{row:-1,col:-2}],
    // 16X8
    [Position{row:-1,col:0}, Position{row:0,col:-1}, Position{row:-1,col:1},
     Position{row:-1,col:-1}, Position{row:-2,col:0}, Position{row:0,col:-2},
     Position{row:-1,col:-2}, Position{row:-2,col:-1}],
    // 16X16
    [Position{row:-1,col:0}, Position{row:0,col:-1}, Position{row:-1,col:1},
     Position{row:1,col:-1}, Position{row:-1,col:-1}, Position{row:-3,col:0},
     Position{row:0,col:-3}, Position{row:-3,col:-3}],
    // 16X32
    [Position{row:0,col:-1}, Position{row:-1,col:0}, Position{row:2,col:-1},
     Position{row:-1,col:-1}, Position{row:-1,col:1}, Position{row:0,col:-3},
     Position{row:-3,col:0}, Position{row:-3,col:-3}],
    // 32X16
    [Position{row:-1,col:0}, Position{row:0,col:-1}, Position{row:-1,col:2},
     Position{row:-1,col:-1}, Position{row:1,col:-1}, Position{row:-3,col:0},
     Position{row:0,col:-3}, Position{row:-3,col:-3}],
    // 32X32
    [Position{row:-1,col:1}, Position{row:1,col:-1}, Position{row:-1,col:2},
     Position{row:2,col:-1}, Position{row:-1,col:-1}, Position{row:-3,col:0},
     Position{row:0,col:-3}, Position{row:-3,col:-3}],
    // 32X64
    [Position{row:0,col:-1}, Position{row:-1,col:0}, Position{row:4,col:-1},
     Position{row:-1,col:2}, Position{row:-1,col:-1}, Position{row:0,col:-3},
     Position{row:-3,col:0}, Position{row:2,col:-1}],
    // 64X32
    [Position{row:-1,col:0}, Position{row:0,col:-1}, Position{row:-1,col:4},
     Position{row:2,col:-1}, Position{row:-1,col:-1}, Position{row:-3,col:0},
     Position{row:0,col:-3}, Position{row:-1,col:2}],
    // 64X64
    [Position{row:-1,col:3}, Position{row:3,col:-1}, Position{row:-1,col:4},
     Position{row:4,col:-1}, Position{row:-1,col:-1}, Position{row:-1,col:0},
     Position{row:0,col:-1}, Position{row:-1,col:6}],
];

/// libvpx `idx_n_column_to_subblock[4][2]`
/// (`vp9_mvref_common.h:209-211`). For sub-8 partitions, maps
/// `(block_idx, is_search_col_zero)` to the index of the 4×4
/// sub-block within the parent's `bmi[4]` array. Used by
/// [`get_sub_block_mv`].
pub static IDX_N_COLUMN_TO_SUBBLOCK: [[usize; 2]; 4] =
    [[1, 2], [1, 3], [3, 2], [3, 3]];

/// libvpx `get_sub_block_mv` (`vp9_mvref_common.h:222-230`). For
/// sub-8 candidates, picks one of the parent's `bmi[]` MVs based on
/// the relative scan position. For ≥8×8 candidates, returns the
/// parent block's `mv[which_mv]`.
///
/// `which_mv` is `0` or `1` for first / second reference.
/// `search_col == 0` means "the neighbour shares our column"
/// (libvpx uses `col == 0` as a proxy for "neighbour is directly
/// above us" vs. "neighbour is to the upper-right"). The
/// `block_idx >= 0` predicate gates the sub-8 lookup — pass
/// `block_idx = -1` for the parent path.
#[inline]
fn get_sub_block_mv(candidate: &ModeInfo, which_mv: usize, search_col: i8, block_idx: i32) -> Mv {
    let candidate_is_sub8 = matches!(
        candidate.block_size,
        BlockSize::Block4x4 | BlockSize::Block4x8 | BlockSize::Block8x4
    );
    if block_idx >= 0 && candidate_is_sub8 {
        let key = if search_col == 0 { 1 } else { 0 };
        let sub_idx = IDX_N_COLUMN_TO_SUBBLOCK[block_idx as usize][key];
        candidate.bmi[sub_idx].as_mv[which_mv]
    } else {
        candidate_mv(candidate, which_mv)
    }
}

/// Read the parent block's MV for the given reference slot. Inter
/// blocks store MVs in `BlockMode::Inter { mvs, .. }`; intra blocks
/// have no MV (return `Mv::ZERO`, but the caller gates this on
/// `is_inter_block` so the value isn't actually consumed).
#[inline]
fn candidate_mv(mi: &ModeInfo, which_mv: usize) -> Mv {
    match mi.mode {
        BlockMode::Inter { mvs, .. } => mvs[which_mv],
        BlockMode::Intra { .. } => Mv::ZERO,
    }
}

/// Read the reference frame for the candidate's slot. Inter blocks
/// have `ref_frames[which]`; intra blocks have `RefFrame::IntraFrame`
/// (which is what libvpx treats as `INTRA_FRAME = 0` —
/// `is_inter_block` returns `false` on it, so neighbour walks skip).
#[inline]
fn candidate_ref(mi: &ModeInfo, which: usize) -> RefFrame {
    match mi.mode {
        BlockMode::Inter { ref_frames, .. } => ref_frames[which],
        BlockMode::Intra { .. } => RefFrame::IntraFrame,
    }
}

/// libvpx `is_inter_block` (`vp9_blockd.h:103-105`): `true` iff
/// `ref_frame[0] > INTRA_FRAME`. Our [`BlockMode::Inter`] variant
/// always implies that; the helper is just a readability alias.
#[inline]
fn is_inter_block(mi: &ModeInfo) -> bool {
    matches!(mi.mode, BlockMode::Inter { .. })
}

/// `has_second_ref` — libvpx `vp9_blockd.h:107-109`: `true` iff
/// `ref_frame[1] > INTRA_FRAME`. For our `BlockMode::Inter` that
/// maps to `ref_frames[1].is_inter()`.
#[inline]
fn has_second_ref(mi: &ModeInfo) -> bool {
    match mi.mode {
        BlockMode::Inter { ref_frames, .. } => ref_frames[1].is_inter(),
        BlockMode::Intra { .. } => false,
    }
}

/// 1/8-pel edge offsets used by `clamp_mv_ref` and other MV-clamping
/// helpers. Mirrors libvpx's `xd->mb_to_{left,right,top,bottom}_edge`
/// (`vp9_blockd.h:178-181`).
///
/// All four values are in 1/8-pel units. Negative `top` / `left`
/// values are the typical case (the block sits in the interior, so
/// the upper / left edges are *above* / *to-the-left-of* the MV
/// origin). Positive `bottom` / `right` values denote room remaining
/// to the right / below.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BlockBounds {
    /// `mb_to_left_edge   = -(mi_col * MI_SIZE * 8)` 1/8-pel units.
    pub left: i32,
    /// `mb_to_right_edge  = (mi_cols - bw - mi_col) * MI_SIZE * 8`.
    pub right: i32,
    /// `mb_to_top_edge    = -(mi_row * MI_SIZE * 8)`.
    pub top: i32,
    /// `mb_to_bottom_edge = (mi_rows - bh - mi_row) * MI_SIZE * 8`.
    pub bottom: i32,
}

impl BlockBounds {
    /// Mirror libvpx `set_mi_row_col` (`vp9_onyxc_int.h:422-428`).
    /// `bw`, `bh` are the block's width / height in 8×8 MI units
    /// (the libvpx `num_8x8_blocks_{wide,high}_lookup` values).
    #[must_use]
    pub fn new(mi_row: i32, mi_col: i32, mi_rows: i32, mi_cols: i32, bw: i32, bh: i32) -> Self {
        Self {
            top: -(mi_row * MI_SIZE) * 8,
            bottom: (mi_rows - bh - mi_row) * MI_SIZE * 8,
            left: -(mi_col * MI_SIZE) * 8,
            right: (mi_cols - bw - mi_col) * MI_SIZE * 8,
        }
    }
}

/// libvpx `clamp_mv_ref` (`vp9_mvref_common.h:216-220`). Clamps the
/// MV to the block's allowed range extended by `±MV_BORDER` on each
/// side.
#[inline]
fn clamp_mv_ref(mv: Mv, b: &BlockBounds) -> Mv {
    let min_col = b.left - MV_BORDER;
    let max_col = b.right + MV_BORDER;
    let min_row = b.top - MV_BORDER;
    let max_row = b.bottom + MV_BORDER;
    Mv {
        col: (mv.col as i32).clamp(min_col, max_col) as i16,
        row: (mv.row as i32).clamp(min_row, max_row) as i16,
    }
}

/// libvpx `lower_mv_precision` (`vp9_mvref_common.h:294-300`).
/// Rounds the MV to the nearest even value (1/4-pel) when the
/// per-frame `allow_hp` is `false`, OR when the magnitude is too
/// large for high-precision (per [`use_mv_hp`]).
///
/// The rounding is "toward zero" for the odd bit (libvpx adds `-1`
/// for positive odd, `+1` for negative odd — which is equivalent to
/// `mv & !1` for non-negatives and `(mv + 1) & !1` for negative).
pub fn lower_mv_precision(mv: Mv, allow_hp: bool) -> Mv {
    let use_hp = allow_hp && use_mv_hp(mv);
    if use_hp {
        return mv;
    }
    let adjust = |v: i16| -> i16 {
        if v & 1 == 0 {
            v
        } else if v > 0 {
            v - 1
        } else {
            v + 1
        }
    };
    Mv {
        row: adjust(mv.row),
        col: adjust(mv.col),
    }
}

/// libvpx `scale_mv` (`vp9_mvref_common.h:234-243`). Negates the MV
/// if the candidate's reference has opposite sign-bias from the
/// current block's reference.
///
/// `ref_sign_bias` is indexed by `RefFrame`-as-discriminant (0 =
/// INTRA, 1 = LAST, 2 = GOLDEN, 3 = ALTREF). The INTRA slot is never
/// queried — `is_inter_block` callers exclude intra candidates.
#[inline]
fn scale_mv(
    candidate_ref: RefFrame,
    this_ref: RefFrame,
    sign_bias: &[bool; 4],
    mv: Mv,
) -> Mv {
    let cand = ref_sign_bias_at(sign_bias, candidate_ref);
    let this_ = ref_sign_bias_at(sign_bias, this_ref);
    if cand != this_ {
        Mv {
            row: mv.row.wrapping_neg(),
            col: mv.col.wrapping_neg(),
        }
    } else {
        mv
    }
}

/// Index the 4-slot `[INTRA, LAST, GOLDEN, ALTREF]` sign-bias table
/// by [`RefFrame`]. Out-of-range refs ([`RefFrame::NoRefFrame`])
/// return `false` — they are never queried in practice but we keep
/// the function total.
#[inline]
fn ref_sign_bias_at(sign_bias: &[bool; 4], rf: RefFrame) -> bool {
    match rf {
        RefFrame::NoRefFrame => false,
        RefFrame::IntraFrame => sign_bias[0],
        RefFrame::LastFrame => sign_bias[1],
        RefFrame::GoldenFrame => sign_bias[2],
        RefFrame::AltrefFrame => sign_bias[3],
    }
}

/// Convert a frame-header `[bool; 3]` sign-bias (libvpx indexes by
/// `LAST=0, GOLDEN=1, ALTREF=2`) into the 4-slot `[INTRA, LAST,
/// GOLDEN, ALTREF]` form `find_mv_refs` consumes.
///
/// The INTRA slot is set to `false` — libvpx leaves it
/// uninitialised (default 0) and never reads it; we initialise to a
/// deterministic value to keep the function total.
#[inline]
#[must_use]
pub fn sign_bias_from_header(per_ref: &[bool; 3]) -> [bool; 4] {
    [false, per_ref[0], per_ref[1], per_ref[2]]
}

/// Closure-shaped neighbour lookup. The caller (per-block decoder)
/// translates a relative `Position` into the corresponding
/// `&ModeInfo` if one exists *and* lies within the current tile.
/// Out-of-tile and out-of-frame positions return `None`.
///
/// libvpx threads this via `xd->mi[]` indexed by `mv_ref->col +
/// mv_ref->row * xd->mi_stride`, gated by `is_inside(tile, ...)`.
/// The decoder LLD (`notes/lld/bitstream.md` §… ) wires the
/// neighbour grid through `TileDecodeCtx`; until that's plumbed,
/// the test suite provides synthetic lookups directly.
pub trait NeighbourLookup {
    /// Return the `&ModeInfo` at `pos` relative to the current
    /// block, or `None` if the position is outside the tile.
    fn get(&self, pos: Position) -> Option<&ModeInfo>;
}

impl<F> NeighbourLookup for F
where
    F: Fn(Position) -> Option<&'static ModeInfo>,
{
    fn get(&self, pos: Position) -> Option<&ModeInfo> {
        self(pos)
    }
}

/// Reference-MV candidate list, paralleling libvpx
/// `int_mv mv_ref_list[MAX_MV_REF_CANDIDATES]` plus the
/// `refmv_count` the decoder reports back.
///
/// - `count == 0` → no candidate found; `mvs[0]` is `Mv::ZERO`.
/// - `count == 1` → one candidate in `mvs[0]`, `mvs[1]` is the
///   clamped zero (libvpx `memset`'s the list to zero up front and
///   we mirror that).
/// - `count == 2` → both slots populated, distinct after dedup.
///
/// The decoder uses `mvs[count - 1]` as the "best ref MV" for
/// `NEARMV` (so it picks `mvs[1]` when found) and `mvs[0]` for
/// every other mode (per `dec_find_mv_refs` post-processing).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct RefMvList {
    /// The candidate MVs (`MAX_MV_REF_CANDIDATES = 2`). Always
    /// clamped to `clamp_mv_ref` bounds. Unused slots are
    /// `Mv::ZERO` (libvpx `memset`'s the array up front).
    pub mvs: [Mv; MAX_MV_REF_CANDIDATES],
    /// Number of valid candidates in [`Self::mvs`]. For modes
    /// other than `NEARMV` the decoder force-sets this to `1`; for
    /// `NEARMV` libvpx force-sets it to `MAX_MV_REF_CANDIDATES = 2`
    /// (so the caller can use `mvs[count - 1]` uniformly).
    pub count: usize,
}

impl RefMvList {
    /// An empty list (all zeros, `count == 0`).
    pub const EMPTY: Self = Self {
        mvs: [Mv::ZERO; MAX_MV_REF_CANDIDATES],
        count: 0,
    };
}

/// Inter-mode "early break" gate. Mirrors libvpx
/// `vp9_decodemv.c:499`:
///
/// > `const int early_break = (mode != NEARMV);`
///
/// `early_break == true` means the search stops after the first
/// candidate is found — the caller only consumes `mv_list[0]`.
#[inline]
fn early_break(mode: InterMode) -> bool {
    !matches!(mode, InterMode::NearMv)
}

/// Macro analog of libvpx `ADD_MV_REF_LIST_EB`
/// (`vp9_decodemv.c:452-464`). Returns `true` if the MV addition
/// triggers the function's `goto Done` (i.e. the list is full or
/// `early_break` says we can stop after the first add).
#[inline]
fn add_mv_ref_list_eb(
    mv: Mv,
    list: &mut [Mv; MAX_MV_REF_CANDIDATES],
    count: &mut usize,
    early_break: bool,
) -> bool {
    if *count == 0 {
        list[0] = mv;
        *count = 1;
        // libvpx: only the first add can short-circuit via early_break.
        early_break
    } else {
        // count == 1 (we never call this with count == 2 — the
        // caller cleared it via the early goto).
        if mv != list[0] {
            list[1] = mv;
            *count = 2;
            true
        } else {
            false
        }
    }
}

/// Core walk shared by the parent (`block == -1`) and sub-8
/// (`block >= 0`) paths. Mirrors libvpx `dec_find_mv_refs`
/// (`vp9_decodemv.c:484-607`). Returns the post-processed
/// `RefMvList`.
///
/// `prev_frame_mv` is the co-located block's `MV_REF` entry from
/// the previous frame, used by libvpx's "salvage" passes. v1
/// doesn't carry that grid yet (Phase D); pass `None` until it
/// lands.
#[allow(clippy::too_many_arguments)]
pub fn dec_find_mv_refs<N: NeighbourLookup>(
    mode: InterMode,
    block_size: BlockSize,
    ref_frame: RefFrame,
    neighbours: &N,
    sign_bias: &[bool; 4],
    bounds: &BlockBounds,
    prev_frame_mv: Option<PrevFrameMvRef>,
    block: i32,
) -> RefMvList {
    let mv_ref_search = &MV_REF_BLOCKS[block_size.idx()];
    let eb = early_break(mode);
    let mut list = [Mv::ZERO; MAX_MV_REF_CANDIDATES];
    let mut count: usize = 0;
    let mut different_ref_found = false;

    // The first 2 neighbours are scanned with sub-8 awareness when
    // `block >= 0`. libvpx hoists the `i = 0` initialisation outside
    // the `if (block >= 0)` block, but only loops when `block >= 0`.
    let mut start_i: usize = 0;
    if block >= 0 {
        for &mv_ref in mv_ref_search.iter().take(2) {
            if let Some(candidate_mi) = neighbours.get(mv_ref) {
                different_ref_found = true;
                if candidate_ref(candidate_mi, 0) == ref_frame {
                    let mv = get_sub_block_mv(candidate_mi, 0, mv_ref.col, block);
                    if add_mv_ref_list_eb(mv, &mut list, &mut count, eb) {
                        return finish(list, count, mode, bounds);
                    }
                } else if candidate_ref(candidate_mi, 1) == ref_frame {
                    let mv = get_sub_block_mv(candidate_mi, 1, mv_ref.col, block);
                    if add_mv_ref_list_eb(mv, &mut list, &mut count, eb) {
                        return finish(list, count, mode, bounds);
                    }
                }
            }
        }
        start_i = 2;
    }

    // Remaining neighbours: parent-block MV only (no sub-8
    // dispatch). For the `block == -1` parent path, this loop runs
    // from i = 0; for the sub-8 path, from i = 2.
    for &mv_ref in mv_ref_search.iter().take(MVREF_NEIGHBOURS).skip(start_i) {
        if let Some(candidate_mi) = neighbours.get(mv_ref) {
            different_ref_found = true;
            if candidate_ref(candidate_mi, 0) == ref_frame {
                let mv = candidate_mv(candidate_mi, 0);
                if add_mv_ref_list_eb(mv, &mut list, &mut count, eb) {
                    return finish(list, count, mode, bounds);
                }
            } else if candidate_ref(candidate_mi, 1) == ref_frame {
                let mv = candidate_mv(candidate_mi, 1);
                if add_mv_ref_list_eb(mv, &mut list, &mut count, eb) {
                    return finish(list, count, mode, bounds);
                }
            }
        }
    }

    // Prev-frame "matching ref" salvage.
    if let Some(prev) = prev_frame_mv {
        if prev.ref_frames[0] == ref_frame {
            if add_mv_ref_list_eb(prev.mvs[0], &mut list, &mut count, eb) {
                return finish(list, count, mode, bounds);
            }
        } else if prev.ref_frames[1] == ref_frame
            && add_mv_ref_list_eb(prev.mvs[1], &mut list, &mut count, eb)
        {
            return finish(list, count, mode, bounds);
        }
    }

    // Second walk: pull MVs from candidates whose ref_frame differs
    // from this block's ref_frame (after sign-bias scaling).
    if different_ref_found {
        for &mv_ref in mv_ref_search.iter().take(MVREF_NEIGHBOURS) {
            if let Some(candidate_mi) = neighbours.get(mv_ref) {
                if !is_inter_block(candidate_mi) {
                    continue;
                }
                let r0 = candidate_ref(candidate_mi, 0);
                if r0 != ref_frame {
                    let mv = scale_mv(r0, ref_frame, sign_bias, candidate_mv(candidate_mi, 0));
                    if add_mv_ref_list_eb(mv, &mut list, &mut count, eb) {
                        return finish(list, count, mode, bounds);
                    }
                }
                if has_second_ref(candidate_mi) {
                    let r1 = candidate_ref(candidate_mi, 1);
                    let mv0 = candidate_mv(candidate_mi, 0);
                    let mv1 = candidate_mv(candidate_mi, 1);
                    if r1 != ref_frame && mv1 != mv0 {
                        let mv = scale_mv(r1, ref_frame, sign_bias, mv1);
                        if add_mv_ref_list_eb(mv, &mut list, &mut count, eb) {
                            return finish(list, count, mode, bounds);
                        }
                    }
                }
            }
        }
    }

    // Final prev-frame "different ref" salvage (with sign-bias scale).
    if let Some(prev) = prev_frame_mv {
        if prev.ref_frames[0] != ref_frame && prev.ref_frames[0].is_inter() {
            let mv = scale_mv(prev.ref_frames[0], ref_frame, sign_bias, prev.mvs[0]);
            if add_mv_ref_list_eb(mv, &mut list, &mut count, eb) {
                return finish(list, count, mode, bounds);
            }
        }
        if prev.ref_frames[1].is_inter()
            && prev.ref_frames[1] != ref_frame
            && prev.mvs[1] != prev.mvs[0]
        {
            let mv = scale_mv(prev.ref_frames[1], ref_frame, sign_bias, prev.mvs[1]);
            if add_mv_ref_list_eb(mv, &mut list, &mut count, eb) {
                return finish(list, count, mode, bounds);
            }
        }
    }

    finish(list, count, mode, bounds)
}

/// libvpx finalisation: force `refmv_count` to 1 / 2 based on the
/// inter mode, then clamp the populated entries.
#[inline]
fn finish(
    mut list: [Mv; MAX_MV_REF_CANDIDATES],
    count: usize,
    mode: InterMode,
    bounds: &BlockBounds,
) -> RefMvList {
    // libvpx `vp9_decodemv.c:596-600`. Note this only runs if we
    // fell off the end without `goto Done`; if we early-broke, the
    // caller jumped past this block. In our port we always reach
    // this finalisation, so we need to gate the rewrite the same
    // way: only run it when count < what the mode wants.
    //
    // Equivalently: libvpx writes `refmv_count = (mode == NEARMV) ?
    // 2 : 1` unconditionally on the "fall-through" path, then
    // clamps refmv_count entries. We mirror the unconditional
    // form — the early-break path always returns `count >= target`
    // so the equivalence holds.
    let target = if matches!(mode, InterMode::NearMv) {
        MAX_MV_REF_CANDIDATES
    } else {
        1
    };
    let effective_count = core::cmp::max(count, target);
    for slot in list.iter_mut().take(effective_count) {
        *slot = clamp_mv_ref(*slot, bounds);
    }
    RefMvList {
        mvs: list,
        count: effective_count,
    }
}

/// Co-located block's `MV_REF` from the previous frame. Mirrors
/// libvpx `MV_REF` (`vp9_blockd.h:42-46`) — a `(mv[2], ref_frame[2])`
/// pair captured per-mi-block when the previous frame finished
/// decoding. Used by `cm->use_prev_frame_mvs` paths in
/// `dec_find_mv_refs`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PrevFrameMvRef {
    /// MVs for ref slots 0 / 1.
    pub mvs: [Mv; 2],
    /// Reference frames for ref slots 0 / 1.
    pub ref_frames: [RefFrame; 2],
}

impl Default for PrevFrameMvRef {
    /// Zero-init form used by callers that pre-allocate a frame-sized
    /// grid before the per-block writes. Matches libvpx's
    /// `vpx_calloc(...)` of `MV_REF` arrays (`vp9_decodeframe.c:1472`)
    /// — all MVs zero, both ref slots set to `NO_REF_FRAME` so the
    /// `dec_find_mv_refs` "salvage from prev frame" passes treat
    /// unwritten cells as non-candidates.
    fn default() -> Self {
        Self {
            mvs: [Mv::ZERO; 2],
            ref_frames: [RefFrame::NoRefFrame; 2],
        }
    }
}

/// Public parent-block API. Calls [`dec_find_mv_refs`] with
/// `block = -1`. Use this from the parent (≥8×8) inter-mode decoder.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn find_mv_refs<N: NeighbourLookup>(
    mode: InterMode,
    block_size: BlockSize,
    ref_frame: RefFrame,
    neighbours: &N,
    sign_bias: &[bool; 4],
    bounds: &BlockBounds,
    prev_frame_mv: Option<PrevFrameMvRef>,
) -> RefMvList {
    dec_find_mv_refs(
        mode,
        block_size,
        ref_frame,
        neighbours,
        sign_bias,
        bounds,
        prev_frame_mv,
        -1,
    )
}

/// Pick the best ref MV from the candidate list for use as the
/// `NEWMV` residual reference. Mirrors libvpx
/// `vp9_decodemv.c:779-780` (`tmp_mvs[refmv_count - 1]`).
///
/// Apply [`lower_mv_precision`] before returning — libvpx does the
/// precision lowering at the call site, but we fold it in here for
/// convenience.
#[inline]
#[must_use]
pub fn mv_pred(list: &RefMvList, allow_hp: bool) -> Mv {
    let idx = list.count.saturating_sub(1);
    lower_mv_precision(list.mvs[idx], allow_hp)
}

/// Pick the `(nearest, near)` MV pair for `NEARESTMV` / `NEARMV`.
/// The decoder doesn't actually use the same names — libvpx folds
/// this into the per-mode branch via `tmp_mvs[refmv_count - 1]`.
/// We expose it here for the encoder side (`vp9_find_best_ref_mvs`,
/// `vp9_mvref_common.c:141-151`) and for symmetry with the spec.
///
/// `lower_mv_precision` is applied to each entry; the entries are
/// already clamped by `clamp_mv_ref` upstream.
#[must_use]
pub fn find_best_ref_mvs(list: &RefMvList, allow_hp: bool) -> (Mv, Mv) {
    (
        lower_mv_precision(list.mvs[0], allow_hp),
        lower_mv_precision(list.mvs[1], allow_hp),
    )
}

/// Sub-8 driver. Mirrors libvpx `append_sub8x8_mvs_for_idx`
/// (`vp9_decodemv.c:609-668`). For sub-8 partitions, each 4×4 sub-
/// block (`j ∈ {0..3}`) gets its own "best sub-8 ref MV" derived
/// from the parent block's neighbours plus the already-decoded
/// peer sub-blocks (`bmi[0]`, `bmi[1]`, `bmi[2]`).
///
/// Parameters:
///
/// - `b_mode`: this sub-block's inter mode (NEARESTMV / NEARMV).
///   `NEWMV` is handled by the parent path; `ZEROMV` doesn't reach
///   this function.
/// - `parent`: the parent block's [`ModeInfo`] (carrying the
///   already-decoded `bmi[0..j-1]` MVs).
/// - `block_size`: the parent block's sb_type (one of
///   `Block4x4 / Block4x8 / Block8x4`).
/// - `ref_frame`: the parent's `ref_frames[ref]` (single ref or
///   compound slot 0 / 1).
/// - `ref_idx`: 0 or 1 (which of `parent.bmi[].as_mv[ref_idx]` to
///   read).
/// - `j`: the 4×4 sub-block index (0..=3).
///
/// Returns the resolved MV for `parent.bmi[j].as_mv[ref_idx]`
/// (already clamped, ready to assign without further processing).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn append_sub8x8_mvs_for_idx<N: NeighbourLookup>(
    b_mode: InterMode,
    parent: &ModeInfo,
    block_size: BlockSize,
    ref_frame: RefFrame,
    ref_idx: usize,
    j: usize,
    neighbours: &N,
    sign_bias: &[bool; 4],
    bounds: &BlockBounds,
    prev_frame_mv: Option<PrevFrameMvRef>,
) -> Mv {
    debug_assert!(j < 4, "sub-8 block index out of range");
    debug_assert!(
        matches!(b_mode, InterMode::NearestMv | InterMode::NearMv),
        "append_sub8x8_mvs_for_idx is only valid for NEARESTMV / NEARMV"
    );
    let bmi = &parent.bmi;
    match j {
        0 => {
            let list = dec_find_mv_refs(
                b_mode,
                block_size,
                ref_frame,
                neighbours,
                sign_bias,
                bounds,
                prev_frame_mv,
                0,
            );
            list.mvs[list.count - 1]
        }
        1 | 2 => {
            if matches!(b_mode, InterMode::NearestMv) {
                bmi[0].as_mv[ref_idx]
            } else {
                let list = dec_find_mv_refs(
                    b_mode,
                    block_size,
                    ref_frame,
                    neighbours,
                    sign_bias,
                    bounds,
                    prev_frame_mv,
                    j as i32,
                );
                // libvpx: scan `mv_list` for the first entry that
                // differs from `bmi[0].as_mv[ref]`.
                if bmi[0].as_mv[ref_idx] != list.mvs[0] {
                    list.mvs[0]
                } else if bmi[0].as_mv[ref_idx] != list.mvs[1] {
                    list.mvs[1]
                } else {
                    Mv::ZERO
                }
            }
        }
        3 => {
            if matches!(b_mode, InterMode::NearestMv) {
                bmi[2].as_mv[ref_idx]
            } else {
                // libvpx: scan `[bmi[1], bmi[0], mv_list[0], mv_list[1]]`
                // for the first entry differing from `bmi[2]`.
                if bmi[2].as_mv[ref_idx] != bmi[1].as_mv[ref_idx] {
                    bmi[1].as_mv[ref_idx]
                } else if bmi[2].as_mv[ref_idx] != bmi[0].as_mv[ref_idx] {
                    bmi[0].as_mv[ref_idx]
                } else {
                    let list = dec_find_mv_refs(
                        b_mode,
                        block_size,
                        ref_frame,
                        neighbours,
                        sign_bias,
                        bounds,
                        prev_frame_mv,
                        3,
                    );
                    if bmi[2].as_mv[ref_idx] != list.mvs[0] {
                        list.mvs[0]
                    } else if bmi[2].as_mv[ref_idx] != list.mvs[1] {
                        list.mvs[1]
                    } else {
                        Mv::ZERO
                    }
                }
            }
        }
        _ => Mv::ZERO,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::InterpFilter;
    use crate::intra_pred::IntraMode;
    use crate::transforms::TxSize;

    // ---------- Test helpers --------------------------------------------

    /// Build an inter [`ModeInfo`] with the given ref + MV in slot 0,
    /// no second ref. Other fields use sane defaults.
    fn inter_mi(block_size: BlockSize, ref_frame: RefFrame, mv: Mv) -> ModeInfo {
        ModeInfo {
            block_size,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::NewMv,
                ref_frames: [ref_frame, RefFrame::NoRefFrame],
                mvs: [mv, Mv::ZERO],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        }
    }

    /// Same as [`inter_mi`] but populates the 2nd reference + MV.
    fn inter_mi_compound(
        block_size: BlockSize,
        ref_frames: [RefFrame; 2],
        mvs: [Mv; 2],
    ) -> ModeInfo {
        ModeInfo {
            block_size,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::NewMv,
                ref_frames,
                mvs,
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        }
    }

    /// Build an intra [`ModeInfo`] (skipped by `find_mv_refs`).
    fn intra_mi(block_size: BlockSize) -> ModeInfo {
        ModeInfo {
            block_size,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        }
    }

    /// Per-test neighbour lookup backed by an explicit `(Position,
    /// &ModeInfo)` table. Positions not in the table return `None`.
    struct ListNeighbours<'a> {
        entries: Vec<(Position, &'a ModeInfo)>,
    }
    impl<'a> ListNeighbours<'a> {
        fn new() -> Self {
            Self { entries: Vec::new() }
        }
        fn with(mut self, pos: Position, mi: &'a ModeInfo) -> Self {
            self.entries.push((pos, mi));
            self
        }
    }
    impl NeighbourLookup for ListNeighbours<'_> {
        fn get(&self, pos: Position) -> Option<&ModeInfo> {
            self.entries
                .iter()
                .find_map(|(p, mi)| if *p == pos { Some(*mi) } else { None })
        }
    }

    /// A `BlockBounds` that doesn't clamp (room on all sides).
    fn loose_bounds() -> BlockBounds {
        BlockBounds {
            left: -10_000,
            right: 10_000,
            top: -10_000,
            bottom: 10_000,
        }
    }

    // ---------- Core walks ----------------------------------------------

    /// Empty neighbours: every mode falls through to a zero-filled
    /// list. NEARMV gets `count = 2`, others `count = 1`. The
    /// clamping doesn't affect the zeros.
    #[test]
    fn empty_neighbours_returns_zero() {
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let sign_bias = [false; 4];

        for mode in [InterMode::NearestMv, InterMode::NewMv, InterMode::NearMv] {
            let list = find_mv_refs(
                mode,
                BlockSize::Block16x16,
                RefFrame::LastFrame,
                &nb,
                &sign_bias,
                &bounds,
                None,
            );
            assert_eq!(list.mvs, [Mv::ZERO, Mv::ZERO], "mode={mode:?}");
            let expected_count = if matches!(mode, InterMode::NearMv) { 2 } else { 1 };
            assert_eq!(list.count, expected_count, "mode={mode:?}");
        }
    }

    /// Single neighbour above with a known MV. `mv_ref_blocks[16X16]`
    /// scan order starts with `(-1, 0)` (above) so the candidate is
    /// the first match. NEARESTMV → list.mvs[0] = that MV;
    /// NEWMV uses the same (count = 1).
    #[test]
    fn single_above_neighbour_with_mv() {
        let above_mv = Mv { row: 24, col: -16 };
        let above = inter_mi(BlockSize::Block16x16, RefFrame::LastFrame, above_mv);
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &above);
        let bounds = loose_bounds();
        let sign_bias = [false; 4];

        let list = find_mv_refs(
            InterMode::NearestMv,
            BlockSize::Block16x16,
            RefFrame::LastFrame,
            &nb,
            &sign_bias,
            &bounds,
            None,
        );
        // nearest MV is the above neighbour's MV.
        assert_eq!(list.mvs[0], above_mv);
        assert_eq!(list.count, 1);
    }

    /// Single left + single above with the SAME MV → libvpx dedups.
    /// nearest = that MV, near = (0, 0). NEARMV gets count = 2 with
    /// the second slot zero.
    #[test]
    fn left_and_above_same_mv_dedups() {
        let common_mv = Mv { row: 8, col: 8 };
        let above = inter_mi(BlockSize::Block16x16, RefFrame::LastFrame, common_mv);
        let left = inter_mi(BlockSize::Block16x16, RefFrame::LastFrame, common_mv);
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &above)
            .with(Position { row: 0, col: -1 }, &left);
        let bounds = loose_bounds();
        let sign_bias = [false; 4];

        let list = find_mv_refs(
            InterMode::NearMv,
            BlockSize::Block16x16,
            RefFrame::LastFrame,
            &nb,
            &sign_bias,
            &bounds,
            None,
        );
        assert_eq!(list.mvs[0], common_mv);
        assert_eq!(list.mvs[1], Mv::ZERO);
        assert_eq!(list.count, 2);
    }

    /// Different above + left MVs (same ref) → nearest = above
    /// (priority), near = left. Both populated in the list.
    #[test]
    fn different_above_and_left_distinct_mvs() {
        let above_mv = Mv { row: 12, col: -4 };
        let left_mv = Mv { row: -8, col: 16 };
        let above = inter_mi(BlockSize::Block16x16, RefFrame::LastFrame, above_mv);
        let left = inter_mi(BlockSize::Block16x16, RefFrame::LastFrame, left_mv);
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &above)
            .with(Position { row: 0, col: -1 }, &left);
        let bounds = loose_bounds();
        let sign_bias = [false; 4];

        let list = find_mv_refs(
            InterMode::NearMv,
            BlockSize::Block16x16,
            RefFrame::LastFrame,
            &nb,
            &sign_bias,
            &bounds,
            None,
        );
        assert_eq!(list.mvs[0], above_mv, "nearest = above");
        assert_eq!(list.mvs[1], left_mv, "near = left");
        assert_eq!(list.count, 2);
    }

    /// Sign-bias mismatch: candidate's ref has opposite sign bias
    /// from the block's ref → MV is negated.
    ///
    /// Setup: block ref = LAST (sign_bias[LAST] = false), neighbour
    /// uses GOLDEN (sign_bias[GOLDEN] = true) on slot 0. The walk
    /// falls through the first pass (no LAST match) and the second
    /// pass scales the GOLDEN MV.
    #[test]
    fn sign_bias_opposite_negates_mv() {
        let orig_mv = Mv { row: 16, col: -32 };
        let above = inter_mi(BlockSize::Block16x16, RefFrame::GoldenFrame, orig_mv);
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &above);
        let bounds = loose_bounds();
        let mut sign_bias = [false; 4];
        sign_bias[RefFrame::GoldenFrame as usize] = true;

        let list = find_mv_refs(
            InterMode::NearestMv,
            BlockSize::Block16x16,
            RefFrame::LastFrame,
            &nb,
            &sign_bias,
            &bounds,
            None,
        );
        assert_eq!(list.mvs[0], Mv { row: -16, col: 32 });
        assert_eq!(list.count, 1);
    }

    /// Sign-bias match (both biases equal): candidate's MV is NOT
    /// negated even though refs differ. Mirrors libvpx's `scale_mv`
    /// no-op branch.
    #[test]
    fn sign_bias_same_does_not_negate() {
        let orig_mv = Mv { row: 16, col: -32 };
        let above = inter_mi(BlockSize::Block16x16, RefFrame::GoldenFrame, orig_mv);
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &above);
        let bounds = loose_bounds();
        let sign_bias = [false; 4]; // all equal → no negation.

        let list = find_mv_refs(
            InterMode::NearestMv,
            BlockSize::Block16x16,
            RefFrame::LastFrame,
            &nb,
            &sign_bias,
            &bounds,
            None,
        );
        assert_eq!(list.mvs[0], orig_mv);
    }

    /// Intra candidate is skipped by `find_mv_refs` (libvpx's
    /// `is_inter_block` gate on the second-walk).
    #[test]
    fn intra_neighbour_skipped() {
        let intra = intra_mi(BlockSize::Block16x16);
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &intra);
        let bounds = loose_bounds();
        let sign_bias = [false; 4];

        let list = find_mv_refs(
            InterMode::NearestMv,
            BlockSize::Block16x16,
            RefFrame::LastFrame,
            &nb,
            &sign_bias,
            &bounds,
            None,
        );
        // Intra neighbour found but skipped: list collapses to zero.
        assert_eq!(list.mvs[0], Mv::ZERO);
    }

    /// Clamping at frame edges: a large candidate MV gets clamped
    /// to `bounds.left - MV_BORDER` / `bounds.right + MV_BORDER`.
    #[test]
    fn clamping_at_frame_edges() {
        let huge_mv = Mv { row: 10_000, col: -10_000 };
        let above = inter_mi(BlockSize::Block16x16, RefFrame::LastFrame, huge_mv);
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &above);
        // Block at top-left of frame: left=0, top=0 (negative-zero
        // origin), right/bottom = whatever's left in the frame.
        let bounds = BlockBounds {
            left: 0,
            right: 64,
            top: 0,
            bottom: 64,
        };
        let sign_bias = [false; 4];

        let list = find_mv_refs(
            InterMode::NearestMv,
            BlockSize::Block16x16,
            RefFrame::LastFrame,
            &nb,
            &sign_bias,
            &bounds,
            None,
        );
        // Expected clamp range:
        //   col ∈ [0 - 128, 64 + 128] = [-128, 192].
        //   row ∈ [0 - 128, 64 + 128] = [-128, 192].
        assert_eq!(list.mvs[0], Mv { row: 192, col: -128 });
    }

    /// `BlockBounds::new` mirrors libvpx `set_mi_row_col`.
    #[test]
    fn block_bounds_from_mi_coords() {
        // mi_row=10, mi_col=20, frame 1080x1920 → mi_rows = 135, mi_cols = 240.
        // 16x16 block → bw = bh = 2 (mi units).
        let b = BlockBounds::new(10, 20, 135, 240, 2, 2);
        assert_eq!(b.top, -(10 * 8 * 8)); // -640
        assert_eq!(b.bottom, (135 - 2 - 10) * 8 * 8); // 7872
        assert_eq!(b.left, -(20 * 8 * 8)); // -1280
        assert_eq!(b.right, (240 - 2 - 20) * 8 * 8); // 14080
    }

    /// `lower_mv_precision` clears the LSB of each component when
    /// HP is gated off. Verified against libvpx's `lower_mv_precision`
    /// (round-toward-zero on odd).
    #[test]
    fn lower_mv_precision_hp_disabled() {
        // allow_hp = false → always clear LSB.
        let cases = [
            (Mv { row: 5, col: 7 }, Mv { row: 4, col: 6 }),
            (Mv { row: -5, col: -7 }, Mv { row: -4, col: -6 }),
            (Mv { row: 4, col: 8 }, Mv { row: 4, col: 8 }),
            (Mv { row: 1, col: -1 }, Mv { row: 0, col: 0 }),
        ];
        for (mv, expected) in cases {
            assert_eq!(lower_mv_precision(mv, false), expected, "mv={mv:?}");
        }
    }

    /// `lower_mv_precision` preserves HP when allow_hp = true AND
    /// the magnitude is small enough (`use_mv_hp`).
    #[test]
    fn lower_mv_precision_hp_enabled_small_mv() {
        let mv = Mv { row: 5, col: 7 };
        assert_eq!(lower_mv_precision(mv, true), mv);
    }

    /// `lower_mv_precision` falls back to 1/4-pel when the
    /// magnitude exceeds the HP threshold even with allow_hp=true.
    #[test]
    fn lower_mv_precision_hp_falls_back_for_large_mv() {
        let mv = Mv { row: 100, col: 7 }; // |row| ≥ 64 → use_mv_hp = false
        assert_eq!(lower_mv_precision(mv, true), Mv { row: 100, col: 6 });
    }

    /// `mv_pred` selects the right slot per mode:
    /// - NEARMV path: count = 2, returns `mvs[1]`.
    /// - Other modes: count = 1, returns `mvs[0]`.
    ///
    /// Both go through `lower_mv_precision`.
    #[test]
    fn mv_pred_picks_correct_slot() {
        let list = RefMvList {
            mvs: [Mv { row: 5, col: 9 }, Mv { row: -5, col: -9 }],
            count: 2, // NEARMV
        };
        // With allow_hp=false, both LSBs cleared.
        assert_eq!(mv_pred(&list, false), Mv { row: -4, col: -8 });

        let list_short = RefMvList {
            mvs: [Mv { row: 5, col: 9 }, Mv::ZERO],
            count: 1,
        };
        assert_eq!(mv_pred(&list_short, false), Mv { row: 4, col: 8 });
    }

    /// `find_best_ref_mvs` returns `(nearest, near)` from slots 0
    /// and 1. Apply HP lowering to each.
    #[test]
    fn find_best_ref_mvs_pair() {
        let list = RefMvList {
            mvs: [Mv { row: 5, col: 9 }, Mv { row: 7, col: 11 }],
            count: 2,
        };
        let (nearest, near) = find_best_ref_mvs(&list, false);
        assert_eq!(nearest, Mv { row: 4, col: 8 });
        assert_eq!(near, Mv { row: 6, col: 10 });
    }

    /// Sub-8 (`Block4x4`) `append_sub8x8_mvs_for_idx`: case j=0
    /// pulls from `dec_find_mv_refs` with `block=0`. Set up an
    /// above-left neighbour that is *itself* sub-8 with distinct
    /// `bmi` MVs and verify `get_sub_block_mv` picks the right one.
    ///
    /// For `Block4x4`, `mv_ref_blocks[0]` first entry is
    /// `(-1, 0)` — directly above. `search_col == 0` so the key
    /// into `IDX_N_COLUMN_TO_SUBBLOCK` is `1` (the "col==0" path),
    /// and for `block_idx = 0` that gives `subblock_idx = 2` (i.e.
    /// the candidate's lower-left 4×4).
    #[test]
    fn sub8_append_block0_uses_get_sub_block_mv() {
        let mut above = inter_mi(BlockSize::Block4x4, RefFrame::LastFrame, Mv::ZERO);
        // Override `bmi` with distinct per-sub-block MVs.
        above.bmi[0].as_mv[0] = Mv { row: 1, col: 1 };
        above.bmi[1].as_mv[0] = Mv { row: 2, col: 2 };
        above.bmi[2].as_mv[0] = Mv { row: 3, col: 3 }; // expected pick
        above.bmi[3].as_mv[0] = Mv { row: 4, col: 4 };
        let parent = inter_mi(BlockSize::Block4x4, RefFrame::LastFrame, Mv::ZERO);
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &above);
        let bounds = loose_bounds();
        let sign_bias = [false; 4];

        let got = append_sub8x8_mvs_for_idx(
            InterMode::NearestMv,
            &parent,
            BlockSize::Block4x4,
            RefFrame::LastFrame,
            0,
            0,
            &nb,
            &sign_bias,
            &bounds,
            None,
        );
        // dec_find_mv_refs scan: i=0 → (-1,0). Above is sub-8, so
        // `get_sub_block_mv(above, 0, 0, 0)` reads
        // `IDX_N_COLUMN_TO_SUBBLOCK[0][1]` = 2 → bmi[2].as_mv[0].
        assert_eq!(got, Mv { row: 3, col: 3 });
    }

    /// Sub-8 vs. parent path produce different candidates: a sub-8
    /// neighbour gives a `bmi`-indexed MV under the sub-8 path,
    /// but the parent (`block = -1`) path always reads `candidate->mv[0]`.
    ///
    /// Setup: same neighbour, same scan position, but pass
    /// `block = -1` vs. `block = 0`. The parent path reads
    /// `candidate->mv[0]` (= Mv::ZERO here, because we constructed
    /// the inter MI with mv = ZERO). The sub-8 path reads
    /// `bmi[2].as_mv[0]` = Mv { row: 3, col: 3 }.
    #[test]
    fn sub8_path_differs_from_parent_path() {
        let mut above = inter_mi(BlockSize::Block4x4, RefFrame::LastFrame, Mv::ZERO);
        above.bmi[2].as_mv[0] = Mv { row: 3, col: 3 };
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &above);
        let bounds = loose_bounds();
        let sign_bias = [false; 4];

        let parent_list = dec_find_mv_refs(
            InterMode::NearestMv,
            BlockSize::Block4x4,
            RefFrame::LastFrame,
            &nb,
            &sign_bias,
            &bounds,
            None,
            /* block = */ -1,
        );
        assert_eq!(parent_list.mvs[0], Mv::ZERO, "parent path uses candidate.mv[0]");

        let sub8_list = dec_find_mv_refs(
            InterMode::NearestMv,
            BlockSize::Block4x4,
            RefFrame::LastFrame,
            &nb,
            &sign_bias,
            &bounds,
            None,
            /* block = */ 0,
        );
        assert_eq!(
            sub8_list.mvs[0],
            Mv { row: 3, col: 3 },
            "sub-8 path uses candidate.bmi[..].as_mv[0]"
        );
    }

    /// `NEARESTMV` early-break: after the very first candidate is
    /// found, the search exits. This is observable by giving two
    /// different neighbours and confirming `count == 1` (not 2).
    #[test]
    fn nearestmv_early_breaks_after_first() {
        let above = inter_mi(BlockSize::Block16x16, RefFrame::LastFrame, Mv { row: 1, col: 1 });
        let left = inter_mi(BlockSize::Block16x16, RefFrame::LastFrame, Mv { row: 2, col: 2 });
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &above)
            .with(Position { row: 0, col: -1 }, &left);
        let bounds = loose_bounds();
        let sign_bias = [false; 4];

        let list = find_mv_refs(
            InterMode::NearestMv,
            BlockSize::Block16x16,
            RefFrame::LastFrame,
            &nb,
            &sign_bias,
            &bounds,
            None,
        );
        assert_eq!(list.count, 1, "NEARESTMV stops after first");
        assert_eq!(list.mvs[0], Mv { row: 1, col: 1 });
        // Second slot stays zero (memset).
        assert_eq!(list.mvs[1], Mv::ZERO);
    }

    /// Prev-frame MV salvage: with no spatial neighbours, the
    /// previous-frame co-located block fills slot 0.
    #[test]
    fn prev_frame_mv_fills_when_no_spatial() {
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let sign_bias = [false; 4];
        let prev = PrevFrameMvRef {
            mvs: [Mv { row: 24, col: -12 }, Mv::ZERO],
            ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
        };

        let list = find_mv_refs(
            InterMode::NearestMv,
            BlockSize::Block16x16,
            RefFrame::LastFrame,
            &nb,
            &sign_bias,
            &bounds,
            Some(prev),
        );
        assert_eq!(list.mvs[0], Mv { row: 24, col: -12 });
        assert_eq!(list.count, 1);
    }

    /// Sign-bias helper round-trip: `sign_bias_from_header` maps
    /// the 3-slot header form (LAST/GOLDEN/ALTREF) into the
    /// 4-slot form (INTRA / LAST / GOLDEN / ALTREF) the
    /// ref-frame indexer uses.
    #[test]
    fn sign_bias_from_header_layout() {
        let per_ref = [true, false, true];
        let full = sign_bias_from_header(&per_ref);
        assert_eq!(full, [false, true, false, true]);
        assert!(!ref_sign_bias_at(&full, RefFrame::IntraFrame));
        assert!(ref_sign_bias_at(&full, RefFrame::LastFrame));
        assert!(!ref_sign_bias_at(&full, RefFrame::GoldenFrame));
        assert!(ref_sign_bias_at(&full, RefFrame::AltrefFrame));
    }

    /// Compound neighbour: when the ref frame matches `ref_frame[1]`
    /// (and not `ref_frame[0]`), the slot-1 MV is pulled in. Mirror
    /// libvpx's `else if (candidate->ref_frame[1] == ref_frame)`.
    #[test]
    fn compound_neighbour_slot1_match() {
        let above = inter_mi_compound(
            BlockSize::Block16x16,
            [RefFrame::GoldenFrame, RefFrame::LastFrame],
            [Mv { row: 9, col: 9 }, Mv { row: 7, col: -7 }],
        );
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &above);
        let bounds = loose_bounds();
        let sign_bias = [false; 4];

        let list = find_mv_refs(
            InterMode::NearestMv,
            BlockSize::Block16x16,
            RefFrame::LastFrame,
            &nb,
            &sign_bias,
            &bounds,
            None,
        );
        // ref_frame[1] = LAST matches, so pull slot 1's MV.
        assert_eq!(list.mvs[0], Mv { row: 7, col: -7 });
        assert_eq!(list.count, 1);
    }

    /// `IDX_N_COLUMN_TO_SUBBLOCK` matches libvpx.
    #[test]
    fn idx_n_column_to_subblock_table_shape() {
        assert_eq!(IDX_N_COLUMN_TO_SUBBLOCK, [[1, 2], [1, 3], [3, 2], [3, 3]]);
    }

    /// `MV_REF_BLOCKS` length: 13 block sizes × 8 neighbours.
    #[test]
    fn mv_ref_blocks_table_shape() {
        assert_eq!(MV_REF_BLOCKS.len(), 13);
        for row in &MV_REF_BLOCKS {
            assert_eq!(row.len(), MVREF_NEIGHBOURS);
        }
        // Spot-check a few entries from the libvpx table.
        // Block16x16, neighbour 0: (-1, 0).
        assert_eq!(
            MV_REF_BLOCKS[BlockSize::Block16x16.idx()][0],
            Position { row: -1, col: 0 }
        );
        // Block64x64, neighbour 0: (-1, 3).
        assert_eq!(
            MV_REF_BLOCKS[BlockSize::Block64x64.idx()][0],
            Position { row: -1, col: 3 }
        );
        // Block8x16, neighbour 0: (0, -1).
        assert_eq!(
            MV_REF_BLOCKS[BlockSize::Block8x16.idx()][0],
            Position { row: 0, col: -1 }
        );
    }
}
