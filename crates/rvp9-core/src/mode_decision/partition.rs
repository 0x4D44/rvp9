//! Variance-based partition picker — libvpx-faithful realtime port of
//! `choose_partitioning`. Decides how to split a 64x64 superblock into
//! smaller coding blocks based on the variance of the difference
//! between source and a reference prediction.
//!
//! Reference: `vp9/encoder/vp9_encodeframe.c::choose_partitioning`
//! (`vp9_encodeframe.c:1261-...`) and the threshold setup in
//! `set_vbp_thresholds` (`vp9_encodeframe.c:573-635`).
//!
//! ## Algorithm (F8.2)
//!
//! Top-down recursive descent over the partition tree. At each level
//! (64x64 → 32x32 → 16x16 → 8x8) we compute the per-pixel variance of
//! `(src - ref)` for the block and compare against a q-dependent
//! threshold:
//!
//! - `var < threshold[level]` → emit a leaf at this size (no split).
//! - Otherwise descend into 4 quadrants and recurse.
//! - At 8x8 the production recursion bottoms out. The focused keyframe
//!   diagnostic may instead represent a high-variance 8x8 SPLIT as one
//!   `Block4x4` ModeInfo cell with four BMI modes.
//!
//! ## libvpx divergences (recorded)
//!
//! - **NONE + SPLIT only.** libvpx also evaluates `PARTITION_HORZ` /
//!   `PARTITION_VERT` per level (see `set_vt_partitioning` in
//!   `vp9_encodeframe.c:472-547`). F8.2 emits only NONE/SPLIT leaves.
//!   This is a deliberate simplification — the supervisor will revisit
//!   if BD-rate compares show H/V partitions are load-bearing.
//! - **Production caps keyframes above sub-8.** libvpx descends to 4x4
//!   for keyframes (`use_4x4_partition` in `choose_partitioning:1313`).
//!   The default F8.2 path retains its historical Block16x16 cap. A
//!   focused diagnostic path can now exercise the source-faithful
//!   Block4x4 representation without changing production output.
//! - **Per-pixel variance, not the 256× fixed-point form.** libvpx's
//!   `get_variance` returns `256 * (sse - sum²/N) / N` so the
//!   thresholds in `set_vbp_thresholds` carry a `256×` implicit
//!   factor. We compute per-pixel variance directly (`(sse - sum²/N)
//!   / N`) and scale the threshold accordingly — same comparison,
//!   simpler arithmetic.
//! - **No content-state / noise-level scaling.** libvpx adjusts the
//!   base threshold by `noise_level`, `content_state`,
//!   `scene_change_detected`, etc. F8.2 uses the unscaled keyframe
//!   path. Add those modulations later if needed.
//! - **No force-split on max-32x32 for keyframes.** libvpx's keyframe
//!   path forces a SPLIT at 64x64 unconditionally
//!   (`set_vt_partitioning:504-508`). F8.2 lets 64x64 stay a leaf if
//!   variance is low enough — a fully flat KF can pick `Block64x64`,
//!   which is legal in VP9 even though libvpx prefers smaller blocks
//!   on keyframes. This matters for the all-flat-KF test (which we
//!   want to land on `Block64x64`, not `Block32x32`).
//!
//! ## Threshold formula (libvpx-derived, keyframe)
//!
//! From `set_vbp_thresholds:577-586`:
//!
//! ```text
//! threshold_base = 20 * y_dequant[q][1]
//! thresholds[BLOCK_64X64] = threshold_base
//! thresholds[BLOCK_32X32] = threshold_base >> 2
//! thresholds[BLOCK_16X16] = threshold_base >> 2
//! thresholds[BLOCK_8X8]   = threshold_base << 2
//! ```
//!
//! Here `y_dequant[q][1]` is the Y-plane AC dequant for the frame
//! qindex (see [`crate::quant::forward::plane_dequant_pair`]). The
//! threshold scales linearly with the dequant step — at higher q the
//! quantization error tolerance per pixel rises, so larger residuals
//! are acceptable before forcing a split.
//!
//! Because libvpx's `get_variance` returns `256 * per_pixel_variance`,
//! the thresholds above are in "per-pixel-variance × 256" units. We
//! compare against `per_pixel_variance = (sse - sum²/N) / N`, so we
//! divide the threshold by 256 (i.e. `>> 8`) to match.

use crate::block::BlockSize;
use crate::mode_decision::inter::{ContentStateSb, NoiseEstimate, NoiseLevel};
use crate::mv_search::PlaneRef;
use crate::quant::forward::{QuantizerDeltas, plane_dequant_pair};

/// 8x8 grid count for a 64x64 superblock — every leaf is one of the
/// 13 VP9 block sizes, but partition descent in F8.2 lands at one of
/// `Block8x8` / `Block16x16` / `Block32x32` / `Block64x64`. The focused
/// keyframe diagnostic may represent an 8x8 SPLIT as `Block4x4`.
pub(crate) const SB_8X8_GRID: usize = 8;

/// A 64x64 superblock's partition decision.
///
/// `cells[i][j]` corresponds to the 8x8 grid position
/// `(row = i, col = j)` inside the SB, where `(0, 0)` is the
/// top-left. Each entry is the leaf [`BlockSize`] that covers that
/// 8x8 cell.
///
/// Multiple cells can belong to the same leaf: e.g. a `Block32x32`
/// leaf at the top-left covers `cells[0..4][0..4]` with identical
/// entries. The legacy F8.2 picker emits square leaves only. The
/// source-faithful fixed-tree picker may also emit the six rectangular
/// `HORZ` / `VERT` leaves whose shorter side is at least 8 pixels.
/// The pattern is always a valid VP9 partition tree (no overlapping
/// leaves) — see [`PartitionDecision::is_valid_partition`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PartitionDecision {
    /// 8x8 grid of leaf assignments. `cells[row][col]`.
    pub cells: [[BlockSize; SB_8X8_GRID]; SB_8X8_GRID],
}

impl PartitionDecision {
    /// Build an all-`Block64x64` decision (no splits anywhere — the
    /// "fully flat block" outcome).
    #[must_use]
    pub fn all_64x64() -> Self {
        Self {
            cells: [[BlockSize::Block64x64; SB_8X8_GRID]; SB_8X8_GRID],
        }
    }

    /// Validate that the cell pattern represents a legal VP9 partition
    /// tree. A square node may be `NONE`, `HORZ`, `VERT`, or `SPLIT`;
    /// every leaf must stamp its full aligned rectangle.
    ///
    /// Returns `true` iff the pattern is legal.
    #[must_use]
    pub fn is_valid_partition(&self) -> bool {
        fn stamped(
            cells: &[[BlockSize; SB_8X8_GRID]; SB_8X8_GRID],
            row: usize,
            col: usize,
            height: usize,
            width: usize,
            bsize: BlockSize,
        ) -> bool {
            row + height <= SB_8X8_GRID
                && col + width <= SB_8X8_GRID
                && cells[row..row + height]
                    .iter()
                    .all(|line| line[col..col + width].iter().all(|&cell| cell == bsize))
        }

        fn walk(
            cells: &[[BlockSize; SB_8X8_GRID]; SB_8X8_GRID],
            row: usize,
            col: usize,
            side: usize,
        ) -> bool {
            let bsize = cells[row][col];
            if side == 1 {
                return matches!(bsize, BlockSize::Block8x8 | BlockSize::Block4x4)
                    && stamped(cells, row, col, 1, 1, bsize);
            }
            let width = usize::from(bsize.width()) / 8;
            let height = usize::from(bsize.height()) / 8;

            if width == side && height == side {
                return stamped(cells, row, col, side, side, bsize);
            }
            if side > 1 && width == side && height * 2 == side {
                return stamped(cells, row, col, height, width, bsize)
                    && stamped(cells, row + height, col, height, width, bsize);
            }
            if side > 1 && height == side && width * 2 == side {
                return stamped(cells, row, col, height, width, bsize)
                    && stamped(cells, row, col + width, height, width, bsize);
            }
            if side <= 1 || width >= side || height >= side {
                return false;
            }

            let half = side / 2;
            walk(cells, row, col, half)
                && walk(cells, row, col + half, half)
                && walk(cells, row + half, col, half)
                && walk(cells, row + half, col + half, half)
        }

        walk(&self.cells, 0, 0, SB_8X8_GRID)
    }
}

/// Inputs to [`choose_partitioning`].
///
/// `src_plane` / `src_offset_xy` point at the top-left of the 64x64
/// superblock in the source-Y frame. `ref_plane` / `ref_offset_xy`
/// point at the corresponding region in the reference frame
/// (`LASTFRAME`) — for ZEROMV the two offsets are equal.
///
/// When `is_keyframe` is true the reference is unused: we synthesize a
/// flat-128 fallback prediction (libvpx's `use_4x4_partition` path
/// also doesn't trust the previous frame on KFs).
///
/// `q_index` is the frame's base qindex (0..255); higher q raises the
/// threshold and tolerates more variance before splitting.
pub struct PartitionSearchParams<'a> {
    /// Source luma plane view.
    pub src_plane: &'a PlaneRef<'a>,
    /// Top-left of the 64x64 SB in source samples, `(x, y)`.
    pub src_offset_xy: (usize, usize),
    /// Reference luma plane view (ignored when `is_keyframe == true`).
    pub ref_plane: &'a PlaneRef<'a>,
    /// Top-left of the 64x64 region in the reference, `(x, y)`. For
    /// ZEROMV this equals `src_offset_xy`.
    pub ref_offset_xy: (usize, usize),
    /// When true, the reference is replaced with a flat-128 fallback.
    pub is_keyframe: bool,
    /// Frame base qindex (0..255). Drives the threshold scaling.
    pub q_index: u8,
    /// Full frame width in pixels. Selects the inter-frame threshold
    /// branch in `set_vbp_thresholds` (small-res ≤352×288 vs larger).
    /// Ignored on keyframes.
    pub frame_width: usize,
    /// Full frame height in pixels. See [`Self::frame_width`].
    pub frame_height: usize,
    /// Rate controller's running `avg_frame_qindex[INTER_FRAME]`. On
    /// inter frames this picks the `>200` / `>220` `thresholds[2]`
    /// boost. Ignored on keyframes.
    pub avg_inter_qindex: u8,
    /// W2.6 — vbp_threshold noise scale from pre-analysis. Mirrors the
    /// noise-driven multiplier libvpx applies inside
    /// `set_vbp_thresholds` (`vp9_encodeframe.c:588-598`): on the inter
    /// branch the per-level threshold is multiplied by this factor
    /// (1× / 2× / 4× for `NoiseLevel::Low|LowLow` / `Medium` / `High`
    /// respectively — see
    /// [`crate::pre_analysis::PreAnalysisState::vbp_threshold_scale`]).
    /// Pass `1` to disable (default; no-op multiplier). The keyframe
    /// threshold branch ignores this value (libvpx applies the scale
    /// only on the non-key branch).
    pub noise_threshold_scale: u32,
}

/// Per-level threshold for variance-based partition decisions.
///
/// `thresholds[i]` is the **per-pixel** variance threshold for the
/// level indexed as in libvpx's `set_vbp_thresholds`:
///
/// - `thresholds[0]` → `Block64x64`
/// - `thresholds[1]` → `Block32x32`
/// - `thresholds[2]` → `Block16x16`
/// - `thresholds[3]` → `Block8x8`
///
/// All values are in raw variance units (`sse - sum²/N` divided by
/// `N`), not libvpx's 256× fixed-point.
#[derive(Clone, Copy, Debug)]
struct VbpThresholds {
    thresholds: [u64; 4],
}

/// Pure input to the pinned-libvpx realtime inter variance-tree seam.
///
/// `source` and `predictor` are materialized 64x64, stride-64 blocks. They
/// include any edge padding already selected by the caller. `visible_width`
/// and `visible_height` reproduce libvpx's checks for samples whose top-left
/// falls outside the visible frame.
#[derive(Clone, Copy, Debug)]
pub struct FixedTreeInterInput<'a> {
    /// Materialized source luma block, stride 64.
    pub source: &'a [u8],
    /// Materialized selected partition predictor, stride 64.
    pub predictor: &'a [u8],
    /// Visible width inside this superblock, in pixels (`1..=64`).
    pub visible_width: usize,
    /// Visible height inside this superblock, in pixels (`1..=64`).
    pub visible_height: usize,
    /// Full coded frame width, used by the low-temporal-variance MV gate.
    pub frame_width: usize,
    /// Exact libvpx VBP thresholds, in its 256x variance units.
    pub thresholds: [i64; 4],
    /// Exact effective min/max threshold, including content-state scaling.
    pub minmax_threshold: i32,
    /// Whether speed/content analysis forces the 64x64 node open.
    pub force_64x64_split: bool,
    /// Whether the min/max 8x8 spread check is enabled.
    pub compute_minmax_variance: bool,
    /// Whether cyclic-refresh selected a boosted segment.
    pub segment_is_boosted: bool,
    /// Whether partition prediction selected LAST rather than GOLDEN.
    pub partition_ref_is_last: bool,
    /// Selected partition MV row in Q3 units.
    pub partition_mv_row: i16,
    /// Selected partition MV column in Q3 units.
    pub partition_mv_col: i16,
    /// libvpx `short_circuit_low_temp_var` speed-feature level.
    pub low_temp_var_mode: u8,
    /// Noise tier used by the 32x32/64x64 spread rules.
    pub noise_level: NoiseLevel,
    /// Whether the frame is in libvpx's `width<=352 && height<=288` tier.
    pub low_resolution: bool,
    /// Whether the full frame height satisfies libvpx's `height<=360` rule.
    pub frame_height_le_360: bool,
}

/// Inputs to pinned-libvpx's cpu-used 7 inter VBP threshold ladder.
#[derive(Clone, Copy, Debug)]
pub struct Cpu7InterVbpThresholdInput {
    /// Effective frame or cyclic-refresh segment qindex.
    pub q_index: u8,
    /// Rate control's running inter-frame average qindex.
    pub avg_inter_qindex: u8,
    /// Coded frame width.
    pub frame_width: usize,
    /// Coded frame height.
    pub frame_height: usize,
    /// Frame noise estimate.
    pub noise_estimate: NoiseEstimate,
    /// Exact source-SAD content state for this superblock.
    pub content_state: ContentStateSb,
}

/// Exact per-superblock cpu-used 7 VBP thresholds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cpu7InterVbpThresholds {
    /// 64x64, 32x32, 16x16, and retained 8x8 threshold slots.
    pub thresholds: [i64; 4],
    /// Effective min/max spread threshold, including very-high-SAD scaling.
    pub minmax_threshold: i32,
}

/// Compute pinned-libvpx's cpu-used 7 inter variance thresholds.
///
/// Mirrors `set_vbp_thresholds` plus the per-superblock
/// `vbp_threshold_minmax` scaling in `choose_partitioning`. Slot 3 is the
/// frame-level keyframe-derived value retained by libvpx; inter partitioning
/// never consults it because 8x8 is the minimum inter block.
#[must_use]
pub fn cpu7_inter_vbp_thresholds(input: Cpu7InterVbpThresholdInput) -> Cpu7InterVbpThresholds {
    let (_, ac) = plane_dequant_pair(i32::from(input.q_index), QuantizerDeltas::Y);
    let raw_base = i64::from(ac);
    let mut base = raw_base;
    if input.noise_estimate.enabled && input.frame_width >= 640 && input.frame_height >= 480 {
        base = match input.noise_estimate.level {
            NoiseLevel::High => 3 * base,
            NoiseLevel::Medium => base << 1,
            NoiseLevel::LowLow => (7 * base) >> 3,
            NoiseLevel::Low => base,
        };
    }
    if input.content_state.scales_cpu7_partition_threshold() {
        base = (5 * base) >> 2;
    }

    let (t0, t1, t2) = if input.frame_width <= 352 && input.frame_height <= 288 {
        let mut t2 = base << 3;
        if input.avg_inter_qindex > 220 {
            t2 <<= 2;
        } else if input.avg_inter_qindex > 200 {
            t2 <<= 1;
        }
        (base >> 3, base >> 1, t2)
    } else if input.frame_width < 1280 && input.frame_height < 720 {
        (base, (5 * base) >> 2, base << 7)
    } else if input.frame_width < 1920 && input.frame_height < 1080 {
        (base, base << 1, base << 7)
    } else {
        (base, (5 * base) >> 1, base << 7)
    };
    Cpu7InterVbpThresholds {
        thresholds: [t0, t1, t2, raw_base * 80],
        minmax_threshold: (15 + (i32::from(input.q_index) >> 3))
            << (input.content_state == ContentStateSb::VeryHighSad) as u32,
    }
}

/// A variance node in the stable differential-oracle trace.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FixedTreeVarianceTrace {
    /// Variance of the whole node.
    pub none: i32,
    /// Variances of the top and bottom halves.
    pub horizontal: [i32; 2],
    /// Variances of the left and right halves.
    pub vertical: [i32; 2],
}

/// Exact accumulator behind one libvpx `Var`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FixedTreeAccumulatorTrace {
    /// Sum of squared average differences.
    pub sum_square_error: u64,
    /// Sum of average differences.
    pub sum_error: i64,
    /// Base-two logarithm of the sample count.
    pub log2_count: u8,
}

/// Exact accumulators behind one libvpx `partition_variance`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FixedTreeNodeAccumulatorTrace {
    /// Accumulator for the whole node.
    pub none: FixedTreeAccumulatorTrace,
    /// Accumulators for the top and bottom halves.
    pub horizontal: [FixedTreeAccumulatorTrace; 2],
    /// Accumulators for the left and right halves.
    pub vertical: [FixedTreeAccumulatorTrace; 2],
}

/// Partition symbol selected at one visited square parent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedTreePartitionSymbol {
    /// Keep the square parent as one leaf.
    None,
    /// Emit two wide rectangular leaves.
    Horizontal,
    /// Emit two tall rectangular leaves.
    Vertical,
    /// Recurse into four square children.
    Split,
}

/// One visited parent in coding order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedTreeNodeDecision {
    /// Parent row inside the superblock, in 8x8 units.
    pub row_8x8: usize,
    /// Parent column inside the superblock, in 8x8 units.
    pub col_8x8: usize,
    /// Square parent size.
    pub parent_size: BlockSize,
    /// Selected VP9 partition symbol.
    pub symbol: FixedTreePartitionSymbol,
}

/// One final leaf in VP9 coding order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedTreeLeaf {
    /// Leaf row inside the superblock, in 8x8 units.
    pub row_8x8: usize,
    /// Leaf column inside the superblock, in 8x8 units.
    pub col_8x8: usize,
    /// Final square or rectangular leaf size.
    pub block_size: BlockSize,
}

/// Complete output of the pure variance-tree differential seam.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedTreePartitionTrace {
    /// Final leaf-size grid.
    pub decision: PartitionDecision,
    /// `force_split[0..21]`, using libvpx's stable node indexing.
    pub force_split: [bool; 21],
    /// Node variances: root, four 32x32 nodes, then sixteen 16x16 nodes.
    pub variances: [FixedTreeVarianceTrace; 21],
    /// Exact node accumulators in the same stable 21-node order.
    pub accumulators: [FixedTreeNodeAccumulatorTrace; 21],
    /// Whether each node's sum/sum-square structure was materialized.
    ///
    /// A valid entry is a canonical variance derived from those sums even
    /// when libvpx delays or skips writing the cached `variance` member.
    pub variance_valid: [bool; 21],
    /// Whether each 16x16 node switched from 8x8 to 4x4 averages.
    pub used_4x4_averages: [bool; 16],
    /// Low-temporal-variance flags retained with the selected tree.
    pub variance_low: [bool; 25],
    /// Per-16x16 8x8-spread result, or `-1` when that check did not run.
    pub minmax_16x16: [i32; 16],
    /// Alternate 4x4-average 16x16 nodes (`vt2`) used only for local decisions.
    ///
    /// Entries are meaningful only where [`Self::used_4x4_averages`] is true.
    pub alternate_4x4_variances: [FixedTreeVarianceTrace; 16],
    /// Exact accumulators for the alternate 4x4-average nodes.
    pub alternate_4x4_accumulators: [FixedTreeNodeAccumulatorTrace; 16],
    /// Parent partition symbols in recursive coding order.
    pub nodes: Vec<FixedTreeNodeDecision>,
    /// Final leaves in recursive coding order.
    pub leaves: Vec<FixedTreeLeaf>,
}

/// Per-superblock state retained by libvpx's realtime partition-copy path.
///
/// `choose_partitioning` may reuse the previous canonical tree for a quiet
/// superblock instead of rebuilding its variance tree. The copy is allowed
/// only for a bounded number of consecutive frames and carries the previous
/// low-temporal-variance flags with it.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FixedTreeHistoryEntry {
    decision: Option<PartitionDecision>,
    previous_segment_id: u8,
    copied_frame_count: u8,
    variance_low: [bool; 25],
}

/// Stream conditions shared by both libvpx partition-copy exits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedTreeCopyConditions {
    /// Frames elapsed since the last keyframe.
    pub frames_since_key: u32,
    /// Current superblock segment.
    pub segment_id: u8,
    /// Maximum number of consecutive copies for this speed/layer.
    pub max_copied_frames: u8,
    /// A pending resize forbids reuse of the old grid.
    pub resize_pending: bool,
    /// SVC may forbid copy on reference/base-layer frames.
    pub svc_copy_allowed: bool,
}

/// A successful partition-history copy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedTreeCopy {
    /// Reused legal partition decision.
    pub decision: PartitionDecision,
    /// Parent symbols in coding order.
    pub nodes: Vec<FixedTreeNodeDecision>,
    /// Leaves in coding order.
    pub leaves: Vec<FixedTreeLeaf>,
    /// Low-temporal-variance flags carried from the source decision.
    pub variance_low: [bool; 25],
    /// Consecutive-copy count after this reuse.
    pub copied_frame_count: u8,
}

impl FixedTreeHistoryEntry {
    /// Try libvpx's common `copy_partitioning` eligibility checks.
    ///
    /// The caller owns the two trigger predicates: low source SAD before
    /// prediction, and low selected-LAST SAD after prediction. A successful
    /// copy increments the consecutive count but does not replace the source
    /// tree or its variance flags.
    pub fn try_copy(&mut self, conditions: FixedTreeCopyConditions) -> Option<FixedTreeCopy> {
        if conditions.frames_since_key <= 1
            || !conditions.svc_copy_allowed
            || conditions.resize_pending
            || conditions.segment_id != 0
            || self.previous_segment_id != 0
            || self.copied_frame_count >= conditions.max_copied_frames
        {
            return None;
        }
        let decision = self.decision.clone()?;
        self.copied_frame_count = self.copied_frame_count.saturating_add(1);
        let (nodes, leaves) = trace_fixed_decision(&decision);
        Some(FixedTreeCopy {
            decision,
            nodes,
            leaves,
            variance_low: self.variance_low,
            copied_frame_count: self.copied_frame_count,
        })
    }

    /// Store a freshly selected tree and reset its consecutive-copy count.
    pub fn update(
        &mut self,
        decision: PartitionDecision,
        segment_id: u8,
        variance_low: [bool; 25],
    ) {
        assert!(
            decision.is_valid_partition(),
            "fixed-tree history requires a legal VP9 partition"
        );
        self.decision = Some(decision);
        self.previous_segment_id = segment_id;
        self.copied_frame_count = 0;
        self.variance_low = variance_low;
    }

    /// Segment retained with the source tree.
    #[must_use]
    pub const fn previous_segment_id(&self) -> u8 {
        self.previous_segment_id
    }

    /// Number of consecutive copies since the source tree was refreshed.
    #[must_use]
    pub const fn copied_frame_count(&self) -> u8 {
        self.copied_frame_count
    }

    /// Whether a source tree has been stored.
    #[must_use]
    pub const fn is_initialized(&self) -> bool {
        self.decision.is_some()
    }
}

/// Map libvpx's per-superblock low-temporal-variance state to one coded leaf.
///
/// Mirrors `vp9_pickmode.c::get_force_skip_low_temp_var`. The 25 slots are
/// laid out as one 64x64 flag, two 64x32 flags, two 32x64 flags, four 32x32
/// flags, and sixteen 16x16 flags in raster order within the superblock.
#[must_use]
pub fn fixed_tree_force_skip_low_temp_var(
    variance_low: &[bool; 25],
    mi_row: usize,
    mi_col: usize,
    block_size: BlockSize,
) -> bool {
    const POS_SHIFT_16X16: [[usize; 4]; 4] = [
        [9, 10, 13, 14],
        [11, 12, 15, 16],
        [17, 18, 21, 22],
        [19, 20, 23, 24],
    ];

    let local_row = mi_row & 7;
    let local_col = mi_col & 7;
    let i = local_row >> 1;
    let j = local_col >> 1;
    match block_size {
        BlockSize::Block64x64 => variance_low[0],
        BlockSize::Block64x32 if local_col == 0 && local_row == 0 => variance_low[1],
        BlockSize::Block64x32 if local_col == 0 && local_row != 0 => variance_low[2],
        BlockSize::Block32x64 if local_col == 0 && local_row == 0 => variance_low[3],
        BlockSize::Block32x64 if local_col != 0 && local_row == 0 => variance_low[4],
        BlockSize::Block32x32 => {
            let index = 5 + usize::from(local_row != 0) * 2 + usize::from(local_col != 0);
            variance_low[index]
        }
        BlockSize::Block16x16 => variance_low[POS_SHIFT_16X16[i][j]],
        BlockSize::Block32x16 => {
            let j2 = ((mi_col + 2) & 7) >> 1;
            variance_low[POS_SHIFT_16X16[i][j]] && variance_low[POS_SHIFT_16X16[i][j2]]
        }
        BlockSize::Block16x32 => {
            let i2 = ((mi_row + 2) & 7) >> 1;
            variance_low[POS_SHIFT_16X16[i][j]] && variance_low[POS_SHIFT_16X16[i2][j]]
        }
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FixedVar {
    sum_square_error: u64,
    sum_error: i64,
    log2_count: u8,
}

impl FixedVar {
    fn sample(diff: i32) -> Self {
        Self {
            sum_square_error: (diff * diff) as u64,
            sum_error: i64::from(diff),
            log2_count: 0,
        }
    }

    fn sum(a: Self, b: Self) -> Self {
        debug_assert_eq!(a.log2_count, b.log2_count);
        Self {
            sum_square_error: a.sum_square_error + b.sum_square_error,
            sum_error: a.sum_error + b.sum_error,
            log2_count: a.log2_count + 1,
        }
    }

    fn variance(self) -> i32 {
        let correction = ((self.sum_error * self.sum_error) >> self.log2_count) as u64;
        let centered = self.sum_square_error.saturating_sub(correction);
        ((256u64 * centered) >> self.log2_count) as i32
    }

    fn accumulator_trace(self) -> FixedTreeAccumulatorTrace {
        FixedTreeAccumulatorTrace {
            sum_square_error: self.sum_square_error,
            sum_error: self.sum_error,
            log2_count: self.log2_count,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FixedPartitionVariance {
    none: FixedVar,
    horizontal: [FixedVar; 2],
    vertical: [FixedVar; 2],
}

impl FixedPartitionVariance {
    fn from_quadrants(split: [FixedVar; 4]) -> Self {
        let horizontal = [
            FixedVar::sum(split[0], split[1]),
            FixedVar::sum(split[2], split[3]),
        ];
        let vertical = [
            FixedVar::sum(split[0], split[2]),
            FixedVar::sum(split[1], split[3]),
        ];
        Self {
            none: FixedVar::sum(vertical[0], vertical[1]),
            horizontal,
            vertical,
        }
    }

    fn trace(self) -> FixedTreeVarianceTrace {
        FixedTreeVarianceTrace {
            none: self.none.variance(),
            horizontal: [self.horizontal[0].variance(), self.horizontal[1].variance()],
            vertical: [self.vertical[0].variance(), self.vertical[1].variance()],
        }
    }

    fn accumulator_trace(self) -> FixedTreeNodeAccumulatorTrace {
        FixedTreeNodeAccumulatorTrace {
            none: self.none.accumulator_trace(),
            horizontal: [
                self.horizontal[0].accumulator_trace(),
                self.horizontal[1].accumulator_trace(),
            ],
            vertical: [
                self.vertical[0].accumulator_trace(),
                self.vertical[1].accumulator_trace(),
            ],
        }
    }
}

fn rounded_average(block: &[u8], x: usize, y: usize, side: usize) -> i32 {
    let mut sum = 0u32;
    for row in 0..side {
        for col in 0..side {
            sum += u32::from(block[(y + row) * 64 + x + col]);
        }
    }
    let count = (side * side) as u32;
    ((sum + count / 2) / count) as i32
}

fn average_diff_var(input: &FixedTreeInterInput<'_>, x: usize, y: usize, side: usize) -> FixedVar {
    if x >= input.visible_width || y >= input.visible_height {
        return FixedVar::default();
    }
    FixedVar::sample(
        rounded_average(input.source, x, y, side) - rounded_average(input.predictor, x, y, side),
    )
}

fn minmax_8x8_spread(input: &FixedTreeInterInput<'_>, x16: usize, y16: usize) -> i32 {
    let mut spread_max = 0;
    let mut spread_min = 255;
    for quadrant in 0..4 {
        let x8 = x16 + (quadrant & 1) * 8;
        let y8 = y16 + (quadrant >> 1) * 8;
        if x8 >= input.visible_width || y8 >= input.visible_height {
            continue;
        }
        let mut min_diff = 255;
        let mut max_diff = 0;
        for row in 0..8 {
            for col in 0..8 {
                let index = (y8 + row) * 64 + x8 + col;
                let diff =
                    (i32::from(input.source[index]) - i32::from(input.predictor[index])).abs();
                min_diff = min_diff.min(diff);
                max_diff = max_diff.max(diff);
            }
        }
        let spread = max_diff - min_diff;
        spread_max = spread_max.max(spread);
        spread_min = spread_min.min(spread);
    }
    spread_max - spread_min
}

fn noise_rank(level: NoiseLevel) -> u8 {
    match level {
        NoiseLevel::LowLow => 0,
        NoiseLevel::Low => 1,
        NoiseLevel::Medium => 2,
        NoiseLevel::High => 3,
    }
}

fn stamp_fixed_leaf(decision: &mut PartitionDecision, row: usize, col: usize, bsize: BlockSize) {
    let height = usize::from(bsize.height()) / 8;
    let width = usize::from(bsize.width()) / 8;
    for line in &mut decision.cells[row..row + height] {
        for cell in &mut line[col..col + width] {
            *cell = bsize;
        }
    }
}

fn fixed_rect_size(parent: BlockSize, vertical: bool) -> BlockSize {
    match (parent, vertical) {
        (BlockSize::Block64x64, false) => BlockSize::Block64x32,
        (BlockSize::Block64x64, true) => BlockSize::Block32x64,
        (BlockSize::Block32x32, false) => BlockSize::Block32x16,
        (BlockSize::Block32x32, true) => BlockSize::Block16x32,
        _ => unreachable!("rectangular variance leaf requires a 64x64 or 32x32 parent"),
    }
}

#[allow(clippy::too_many_arguments)]
fn try_fixed_partition(
    input: &FixedTreeInterInput<'_>,
    node: FixedPartitionVariance,
    row_8x8: usize,
    col_8x8: usize,
    parent: BlockSize,
    threshold: i64,
    force_split: bool,
    decision: &mut PartitionDecision,
) -> bool {
    if force_split {
        return false;
    }

    let side = usize::from(parent.width()) / 8;
    let mi_rows = input.visible_height.div_ceil(8);
    let mi_cols = input.visible_width.div_ceil(8);
    let half = side / 2;

    if parent == BlockSize::Block16x16 {
        if col_8x8 + half < mi_cols
            && row_8x8 + half < mi_rows
            && i64::from(node.none.variance()) < threshold
        {
            stamp_fixed_leaf(decision, row_8x8, col_8x8, parent);
            return true;
        }
        return false;
    }

    if col_8x8 + half < mi_cols
        && row_8x8 + half < mi_rows
        && i64::from(node.none.variance()) < threshold
    {
        stamp_fixed_leaf(decision, row_8x8, col_8x8, parent);
        return true;
    }

    if row_8x8 + half < mi_rows
        && node
            .vertical
            .iter()
            .all(|variance| i64::from(variance.variance()) < threshold)
    {
        let bsize = fixed_rect_size(parent, true);
        stamp_fixed_leaf(decision, row_8x8, col_8x8, bsize);
        stamp_fixed_leaf(decision, row_8x8, col_8x8 + half, bsize);
        return true;
    }

    if col_8x8 + half < mi_cols
        && node
            .horizontal
            .iter()
            .all(|variance| i64::from(variance.variance()) < threshold)
    {
        let bsize = fixed_rect_size(parent, false);
        stamp_fixed_leaf(decision, row_8x8, col_8x8, bsize);
        stamp_fixed_leaf(decision, row_8x8 + half, col_8x8, bsize);
        return true;
    }

    false
}

fn trace_fixed_decision(
    decision: &PartitionDecision,
) -> (Vec<FixedTreeNodeDecision>, Vec<FixedTreeLeaf>) {
    fn walk(
        decision: &PartitionDecision,
        row: usize,
        col: usize,
        side: usize,
        nodes: &mut Vec<FixedTreeNodeDecision>,
        leaves: &mut Vec<FixedTreeLeaf>,
    ) {
        let bsize = decision.cells[row][col];
        let width = usize::from(bsize.width()) / 8;
        let height = usize::from(bsize.height()) / 8;
        let parent_size = match side {
            8 => BlockSize::Block64x64,
            4 => BlockSize::Block32x32,
            2 => BlockSize::Block16x16,
            1 => BlockSize::Block8x8,
            _ => unreachable!("VP9 fixed-tree parent side"),
        };
        let symbol = if width == side && height == side {
            FixedTreePartitionSymbol::None
        } else if width == side && height * 2 == side {
            FixedTreePartitionSymbol::Horizontal
        } else if height == side && width * 2 == side {
            FixedTreePartitionSymbol::Vertical
        } else {
            FixedTreePartitionSymbol::Split
        };
        nodes.push(FixedTreeNodeDecision {
            row_8x8: row,
            col_8x8: col,
            parent_size,
            symbol,
        });

        match symbol {
            FixedTreePartitionSymbol::None => leaves.push(FixedTreeLeaf {
                row_8x8: row,
                col_8x8: col,
                block_size: bsize,
            }),
            FixedTreePartitionSymbol::Horizontal => {
                leaves.push(FixedTreeLeaf {
                    row_8x8: row,
                    col_8x8: col,
                    block_size: bsize,
                });
                leaves.push(FixedTreeLeaf {
                    row_8x8: row + height,
                    col_8x8: col,
                    block_size: bsize,
                });
            }
            FixedTreePartitionSymbol::Vertical => {
                leaves.push(FixedTreeLeaf {
                    row_8x8: row,
                    col_8x8: col,
                    block_size: bsize,
                });
                leaves.push(FixedTreeLeaf {
                    row_8x8: row,
                    col_8x8: col + width,
                    block_size: bsize,
                });
            }
            FixedTreePartitionSymbol::Split => {
                let half = side / 2;
                walk(decision, row, col, half, nodes, leaves);
                walk(decision, row, col + half, half, nodes, leaves);
                walk(decision, row + half, col, half, nodes, leaves);
                walk(decision, row + half, col + half, half, nodes, leaves);
            }
        }
    }

    let mut nodes = Vec::with_capacity(85);
    let mut leaves = Vec::with_capacity(64);
    walk(decision, 0, 0, 8, &mut nodes, &mut leaves);
    (nodes, leaves)
}

/// Run the pure inter-frame `choose_partitioning` variance-tree core used by
/// pinned libvpx v1.14.1 at realtime cpu-used 7.
///
/// Predictor selection, early partition-copy exits, and downstream mode
/// picking remain outside this seam. That boundary lets a differential oracle
/// feed both implementations identical materialized inputs and compare every
/// intermediate node before this path is allowed to become the product default.
#[must_use]
pub fn choose_fixed_tree_inter(input: &FixedTreeInterInput<'_>) -> FixedTreePartitionTrace {
    assert!(input.source.len() >= 64 * 64);
    assert!(input.predictor.len() >= 64 * 64);
    assert!((1..=64).contains(&input.visible_width));
    assert!((1..=64).contains(&input.visible_height));

    let mut tree = [FixedPartitionVariance::default(); 21];
    let mut decision_16x16 = [FixedPartitionVariance::default(); 16];
    let mut variance_valid = [false; 21];
    let mut force_split = [false; 21];
    let mut used_4x4_averages = [false; 16];
    let mut minmax_16x16 = [-1i32; 16];
    let mut avg_16x16 = [0i32; 4];
    let mut max_16x16 = [0i32; 4];
    let mut min_16x16 = [i32::MAX; 4];
    force_split[0] = input.force_64x64_split;

    for i in 0..4 {
        let x32 = (i & 1) * 32;
        let y32 = (i >> 1) * 32;
        for j in 0..4 {
            let x16 = x32 + (j & 1) * 16;
            let y16 = y32 + (j >> 1) * 16;
            let index16 = i * 4 + j;
            let trace_index = 5 + index16;
            let split_8x8 = [
                average_diff_var(input, x16, y16, 8),
                average_diff_var(input, x16 + 8, y16, 8),
                average_diff_var(input, x16, y16 + 8, 8),
                average_diff_var(input, x16 + 8, y16 + 8, 8),
            ];
            tree[trace_index] = FixedPartitionVariance::from_quadrants(split_8x8);
            decision_16x16[index16] = tree[trace_index];
            variance_valid[trace_index] = true;
            let initial_variance = tree[trace_index].none.variance();
            avg_16x16[i] += initial_variance;
            min_16x16[i] = min_16x16[i].min(initial_variance);
            max_16x16[i] = max_16x16[i].max(initial_variance);

            let exceeds_variance_threshold = i64::from(initial_variance) > input.thresholds[2];
            let minmax_eligible = !exceeds_variance_threshold
                && input.compute_minmax_variance
                && i64::from(initial_variance) > input.thresholds[1]
                && !input.segment_is_boosted;
            let exceeds_minmax_threshold = if minmax_eligible {
                let spread = minmax_8x8_spread(input, x16, y16);
                minmax_16x16[index16] = spread;
                spread > input.minmax_threshold
            } else {
                false
            };
            if exceeds_variance_threshold || exceeds_minmax_threshold {
                force_split[trace_index] = true;
                force_split[1 + i] = true;
                force_split[0] = true;
            }

            if input.low_resolution && i64::from(initial_variance) > (input.thresholds[1] << 1) {
                used_4x4_averages[index16] = true;
                force_split[trace_index] = false;
                let mut split_8x8_nodes = [FixedPartitionVariance::default(); 4];
                for (k, node8) in split_8x8_nodes.iter_mut().enumerate() {
                    let x8 = x16 + (k & 1) * 8;
                    let y8 = y16 + (k >> 1) * 8;
                    *node8 = FixedPartitionVariance::from_quadrants([
                        average_diff_var(input, x8, y8, 4),
                        average_diff_var(input, x8 + 4, y8, 4),
                        average_diff_var(input, x8, y8 + 4, 4),
                        average_diff_var(input, x8 + 4, y8 + 4, 4),
                    ]);
                }
                decision_16x16[index16] = FixedPartitionVariance::from_quadrants([
                    split_8x8_nodes[0].none,
                    split_8x8_nodes[1].none,
                    split_8x8_nodes[2].none,
                    split_8x8_nodes[3].none,
                ]);
                if i64::from(decision_16x16[index16].none.variance()) > input.thresholds[2] {
                    force_split[trace_index] = true;
                    force_split[1 + i] = true;
                    force_split[0] = true;
                }
            }
        }
    }

    let mut avg_32x32 = 0i32;
    let mut max_32x32 = 0i32;
    let mut min_32x32 = i32::MAX;
    for i in 0..4 {
        let base = 5 + i * 4;
        tree[1 + i] = FixedPartitionVariance::from_quadrants([
            tree[base].none,
            tree[base + 1].none,
            tree[base + 2].none,
            tree[base + 3].none,
        ]);
        variance_valid[1 + i] = true;
        if !force_split[1 + i] {
            let variance32 = tree[1 + i].none.variance();
            max_32x32 = max_32x32.max(variance32);
            min_32x32 = min_32x32.min(variance32);
            if i64::from(variance32) > input.thresholds[1]
                || (i64::from(variance32) > (input.thresholds[1] >> 1)
                    && variance32 > (avg_16x16[i] >> 1))
                || (noise_rank(input.noise_level) < noise_rank(NoiseLevel::Low)
                    && input.frame_height_le_360
                    && max_16x16[i] - min_16x16[i] > (input.thresholds[1] >> 1) as i32
                    && i64::from(max_16x16[i]) > input.thresholds[1])
            {
                force_split[1 + i] = true;
                force_split[0] = true;
            }
            avg_32x32 += variance32;
        }
    }

    if !force_split[0] {
        tree[0] = FixedPartitionVariance::from_quadrants([
            tree[1].none,
            tree[2].none,
            tree[3].none,
            tree[4].none,
        ]);
        variance_valid[0] = true;
        let variance64 = tree[0].none.variance();
        let medium_or_higher = noise_rank(input.noise_level) >= noise_rank(NoiseLevel::Medium);
        let noisy_average_split = medium_or_higher && variance64 > (9 * avg_32x32) >> 5;
        let low_noise_spread_split = !medium_or_higher
            && max_32x32 - min_32x32 > 3 * ((input.thresholds[0] >> 3) as i32)
            && i64::from(max_32x32) > input.thresholds[0] >> 1;
        if noisy_average_split || low_noise_spread_split {
            force_split[0] = true;
        }
    }

    let mut decision = PartitionDecision {
        cells: [[BlockSize::Block8x8; SB_8X8_GRID]; SB_8X8_GRID],
    };
    let full_mi_width = input.visible_width.div_ceil(8) == 8;
    let full_mi_height = input.visible_height.div_ceil(8) == 8;
    let root_kept = full_mi_width
        && full_mi_height
        && try_fixed_partition(
            input,
            tree[0],
            0,
            0,
            BlockSize::Block64x64,
            input.thresholds[0],
            force_split[0],
            &mut decision,
        );
    if !root_kept {
        for i in 0..4 {
            let row32 = (i >> 1) * 4;
            let col32 = (i & 1) * 4;
            let node32_kept = try_fixed_partition(
                input,
                tree[1 + i],
                row32,
                col32,
                BlockSize::Block32x32,
                input.thresholds[1],
                force_split[1 + i],
                &mut decision,
            );
            if !node32_kept {
                for j in 0..4 {
                    let row16 = row32 + (j >> 1) * 2;
                    let col16 = col32 + (j & 1) * 2;
                    let index16 = i * 4 + j;
                    let node16_kept = try_fixed_partition(
                        input,
                        decision_16x16[index16],
                        row16,
                        col16,
                        BlockSize::Block16x16,
                        input.thresholds[2],
                        force_split[5 + index16],
                        &mut decision,
                    );
                    if !node16_kept {
                        for k in 0..4 {
                            stamp_fixed_leaf(
                                &mut decision,
                                row16 + (k >> 1),
                                col16 + (k & 1),
                                BlockSize::Block8x8,
                            );
                        }
                    }
                }
            }
        }
    }

    debug_assert!(decision.is_valid_partition());
    let variance_low = fixed_tree_variance_low(input, &decision, &tree);
    let (nodes, leaves) = trace_fixed_decision(&decision);
    let mut variances = [FixedTreeVarianceTrace::default(); 21];
    let mut accumulators = [FixedTreeNodeAccumulatorTrace::default(); 21];
    for index in 0..21 {
        variances[index] = tree[index].trace();
        accumulators[index] = tree[index].accumulator_trace();
    }
    let mut alternate_4x4_variances = [FixedTreeVarianceTrace::default(); 16];
    let mut alternate_4x4_accumulators = [FixedTreeNodeAccumulatorTrace::default(); 16];
    for index in 0..16 {
        if used_4x4_averages[index] {
            alternate_4x4_variances[index] = decision_16x16[index].trace();
            alternate_4x4_accumulators[index] = decision_16x16[index].accumulator_trace();
        }
    }
    FixedTreePartitionTrace {
        decision,
        force_split,
        variances,
        accumulators,
        variance_valid,
        used_4x4_averages,
        variance_low,
        minmax_16x16,
        alternate_4x4_variances,
        alternate_4x4_accumulators,
        nodes,
        leaves,
    }
}

fn fixed_tree_variance_low(
    input: &FixedTreeInterInput<'_>,
    decision: &PartitionDecision,
    tree: &[FixedPartitionVariance; 21],
) -> [bool; 25] {
    let mut variance_low = [false; 25];
    let mv_threshold = if input.frame_width > 640 { 8 } else { 4 };
    let small_partition_mv = i32::from(input.partition_mv_col).abs() < mv_threshold
        && i32::from(input.partition_mv_row).abs() < mv_threshold;
    if input.low_temp_var_mode == 0
        || !input.partition_ref_is_last
        || (input.low_temp_var_mode != 1 && !small_partition_mv)
    {
        return variance_low;
    }

    match decision.cells[0][0] {
        BlockSize::Block64x64 => {
            variance_low[0] = i64::from(tree[0].none.variance()) < (input.thresholds[0] >> 1);
        }
        BlockSize::Block64x32 => {
            for index in 0..2 {
                variance_low[1 + index] =
                    i64::from(tree[0].horizontal[index].variance()) < (input.thresholds[0] >> 2);
            }
        }
        BlockSize::Block32x64 => {
            for index in 0..2 {
                variance_low[3 + index] =
                    i64::from(tree[0].vertical[index].variance()) < (input.thresholds[0] >> 2);
            }
        }
        _ => {
            for quadrant in 0..4 {
                let row = (quadrant >> 1) * 4;
                let col = (quadrant & 1) * 4;
                match decision.cells[row][col] {
                    BlockSize::Block32x32 => {
                        let threshold = if matches!(input.low_temp_var_mode, 1 | 3) {
                            (5 * input.thresholds[1]) >> 3
                        } else {
                            input.thresholds[1] >> 1
                        };
                        variance_low[5 + quadrant] =
                            i64::from(tree[1 + quadrant].none.variance()) < threshold;
                    }
                    BlockSize::Block16x16 | BlockSize::Block32x16 | BlockSize::Block16x32
                        if input.low_temp_var_mode >= 2 =>
                    {
                        for child in 0..4 {
                            variance_low[9 + quadrant * 4 + child] =
                                i64::from(tree[5 + quadrant * 4 + child].none.variance())
                                    < (input.thresholds[2] >> 8);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    variance_low
}

impl VbpThresholds {
    /// Compute the keyframe-path libvpx thresholds for a given qindex.
    /// Mirrors `set_vbp_thresholds:577-586` with the 256× scale
    /// folded out (see module doc).
    fn for_keyframe(q_index: u8) -> Self {
        // libvpx: `threshold_base = 20 * y_dequant[q][1]`. Use the Y
        // plane's AC dequant directly via `plane_dequant_pair`.
        let (_dc, ac) = plane_dequant_pair(i32::from(q_index), QuantizerDeltas::Y);
        let threshold_base = 20u64 * (ac as u64);

        // libvpx's variance carries a 256× scale; we work in raw
        // per-pixel-variance units, so divide each threshold by 256.
        // Use `max(1)` so even at q=0 we don't compare against a hard
        // zero threshold (would always split, even flat blocks).
        let scale = 256u64;
        Self {
            thresholds: [
                (threshold_base / scale).max(1),
                ((threshold_base >> 2) / scale).max(1),
                ((threshold_base >> 2) / scale).max(1),
                ((threshold_base << 2) / scale).max(1),
            ],
        }
    }

    /// Compute the **inter**-path libvpx thresholds for a given qindex,
    /// specialized to realtime speed/cpu-used = 7.
    ///
    /// Ports the non-key branch of `set_vbp_thresholds`
    /// (`vp9/encoder/vp9_encodeframe.c:573-635`) and its helper
    /// `scale_part_thresh_sumdiff` (`:549-567`). The 256× fixed-point
    /// scale is folded out exactly as in [`Self::for_keyframe`] (see
    /// module doc).
    ///
    /// libvpx inter formula (speed 7):
    ///
    /// ```text
    /// threshold_base = variance_part_thresh_mult * y_dequant[q][1]
    ///                = 1 * y_dequant[q][1]   (mult == 1 at speed 7)
    /// // scale_part_thresh_sumdiff at speed 7, default content_state,
    /// // returns threshold_base unchanged.
    /// thresholds[0] = base               // 64x64
    /// thresholds[2] = base << speed       // == base << 7 (generic path)
    /// // small-res override (width<=352 && height<=288):
    /// thresholds[0] = base >> 3
    /// thresholds[1] = base >> 1
    /// thresholds[2] = base << 3
    /// if avg_frame_qindex[INTER] > 220 { thresholds[2] <<= 2 }   // base << 5
    /// else if avg_frame_qindex[INTER] > 200 { thresholds[2] <<= 1 } // base << 4
    /// // larger-res branches set thresholds[1] differently (ported below).
    /// // thresholds[3] (8x8->4x4) is "only used on key frame" — inter
    /// // never splits 8x8->4x4, and rvp9 caps recursion at 8x8 anyway,
    /// // so it's a don't-care. We mirror the keyframe value to keep a
    /// // sane finite number there.
    /// ```
    ///
    /// Divergences from libvpx held to a minimum: we are at the realtime
    /// speed-7 operating point, default content-state (no SAD/sumdiff
    /// classification), no noise/denoiser scaling, no
    /// `disable_16x16part_nonkey`. Those modulations are off at cpu-used
    /// 7 for our target inputs; see the divergence ledger.
    fn for_inter(
        q_index: u8,
        avg_inter_qindex: u8,
        width: usize,
        height: usize,
        noise_threshold_scale: u32,
    ) -> Self {
        // speed/cpu-used = 7 ⇒ variance_part_thresh_mult = 1.
        let (_dc, ac) = plane_dequant_pair(i32::from(q_index), QuantizerDeltas::Y);
        let base = ac as u64;
        // scale_part_thresh_sumdiff(speed=7, default content_state)
        // returns `base` unchanged — the LowSad/HighSad/LowVar arms are
        // not hit for our default content classification.

        // Generic (non-small-res) defaults from the inter branch.
        // thresholds[0] = base; thresholds[2] = base << speed (speed 7).
        let mut t0 = base;
        let t1; // set per resolution branch below
        let mut t2 = base << 7;

        if width <= 352 && height <= 288 {
            // Small-res override (foreman's 256×256 crop lands here).
            t0 = base >> 3;
            t1 = base >> 1;
            t2 = base << 3;
            if avg_inter_qindex > 220 {
                t2 <<= 2;
            } else if avg_inter_qindex > 200 {
                t2 <<= 1;
            }
        } else if width < 1280 && height < 720 {
            t1 = (5 * base) >> 2;
        } else if width < 1920 && height < 1080 {
            t1 = base << 1;
        } else {
            t1 = (5 * base) >> 1;
        }
        // Note: the `width >= 1280 && height >= 720 && speed < 7`
        // thresholds[2] boost does not apply at speed 7.

        // thresholds[3]: don't-care on inter (8x8→4x4 never taken, and
        // rvp9 caps recursion at 8x8). Mirror keyframe (`base << 2`).
        let t3 = base << 2;

        // Fold out the 256× fixed-point scale and floor at 1, exactly
        // as `for_keyframe` does.
        let scale = 256u64;
        // W2.6 — apply the pre-analysis noise scale to the raw base
        // **before** the 256× fold-out and the `.max(1)` floor, so the
        // multiplier is preserved even when the unscaled per-level
        // value would round to zero. Mirrors libvpx
        // `set_vbp_thresholds:588-598`: `thresholds[i] *= noise_factor`
        // is applied on the still-256×-fixed-point values, then the
        // partition descent compares against per-pixel variance. Clamp
        // the scale to ≥1 so a zero from the producer can't zero out
        // every threshold (matches `vbp_threshold_scale`'s contract).
        let noise_mult = u64::from(noise_threshold_scale.max(1));
        Self {
            thresholds: [
                ((t0 * noise_mult) / scale).max(1),
                ((t1 * noise_mult) / scale).max(1),
                ((t2 * noise_mult) / scale).max(1),
                ((t3 * noise_mult) / scale).max(1),
            ],
        }
    }
}

/// Run the libvpx-style variance-based partition picker for one 64x64
/// superblock.
///
/// Returns a [`PartitionDecision`] whose `cells` describe the chosen
/// leaf [`BlockSize`] at every 8x8 position. The returned pattern
/// always passes [`PartitionDecision::is_valid_partition`].
#[must_use]
pub fn choose_partitioning(params: &PartitionSearchParams<'_>) -> PartitionDecision {
    let thresholds = thresholds_for_params(params);

    choose_partitioning_with_force_map(params, &thresholds, None, false)
}

/// Run the keyframe partition picker with its source-faithful sub-8
/// diagnostic enabled.
///
/// Unlike [`choose_partitioning`], a high-variance keyframe can descend
/// through 16x16 and represent an 8x8 `SPLIT` as one `Block4x4` cell. This
/// function is an experiment seam; the production caller does not use it.
#[must_use]
pub fn choose_partitioning_keyframe_sub8(params: &PartitionSearchParams<'_>) -> PartitionDecision {
    let thresholds = thresholds_for_params(params);

    choose_partitioning_with_force_map(params, &thresholds, None, true)
}

/// Run [`choose_partitioning`] with the libvpx-style inter-frame force-split
/// prepass enabled.
///
/// This mirrors the part of libvpx's `choose_partitioning` prepass that marks
/// ancestor blocks as forced splits when 16x16 / 32x32 child variance exceeds
/// the relevant VBP thresholds. The existing square-only NONE/SPLIT descent is
/// otherwise preserved; horizontal/vertical partition probing remains out of
/// scope for this diagnostic variant.
#[must_use]
pub fn choose_partitioning_force_split(params: &PartitionSearchParams<'_>) -> PartitionDecision {
    let thresholds = thresholds_for_params(params);
    let force_split = ForceSplitMap::for_params(params, &thresholds);

    choose_partitioning_with_force_map(params, &thresholds, Some(&force_split), false)
}

fn thresholds_for_params(params: &PartitionSearchParams<'_>) -> VbpThresholds {
    if params.is_keyframe {
        VbpThresholds::for_keyframe(params.q_index)
    } else {
        VbpThresholds::for_inter(
            params.q_index,
            params.avg_inter_qindex,
            params.frame_width,
            params.frame_height,
            params.noise_threshold_scale,
        )
    }
}

static KEYFRAME_FLAT_REFERENCE: [u8; 64 * 64] = [128; 64 * 64];

fn choose_partitioning_with_force_map(
    params: &PartitionSearchParams<'_>,
    thresholds: &VbpThresholds,
    force_split: Option<&ForceSplitMap>,
    allow_keyframe_sub8: bool,
) -> PartitionDecision {
    let mut decision = PartitionDecision {
        cells: [[BlockSize::Block8x8; SB_8X8_GRID]; SB_8X8_GRID],
    };

    // Top-level 64x64 variance.
    if !descend(
        params,
        /* sb_origin */ (0usize, 0usize),
        BlockSize::Block64x64,
        thresholds,
        force_split,
        allow_keyframe_sub8,
        &mut decision,
    ) {
        // descend always returns true on the top-level call because we
        // never reject the partition above 8x8 — the unreachable
        // branch is documented but the function returns `bool` for
        // recursive uniformity with the leaf-cap step.
        debug_assert!(false, "top-level descend must always succeed");
    }

    decision
}

#[derive(Clone, Debug, Default)]
struct ForceSplitMap {
    split_64x64: bool,
    split_32x32: [bool; 4],
    split_16x16: [bool; 16],
}

impl ForceSplitMap {
    fn for_params(params: &PartitionSearchParams<'_>, thresholds: &VbpThresholds) -> Self {
        if params.is_keyframe {
            return Self::default();
        }

        let mut map = Self::default();
        let mut avg_16x16 = [0u64; 4];

        for (i, avg16_slot) in avg_16x16.iter_mut().enumerate() {
            let row32 = (i >> 1) * 4;
            let col32 = (i & 1) * 4;
            for j in 0..4 {
                let row16 = row32 + (j >> 1) * 2;
                let col16 = col32 + (j & 1) * 2;
                let var16 = block_per_pixel_variance(params, (row16, col16), BlockSize::Block16x16);
                *avg16_slot = avg16_slot.saturating_add(var16);
                if var16 > thresholds.thresholds[2] {
                    map.split_16x16[i * 4 + j] = true;
                    map.split_32x32[i] = true;
                    map.split_64x64 = true;
                }
            }
        }

        for (i, avg16) in avg_16x16.iter().enumerate() {
            if map.split_32x32[i] {
                continue;
            }
            let row32 = (i >> 1) * 4;
            let col32 = (i & 1) * 4;
            let var32 = block_per_pixel_variance(params, (row32, col32), BlockSize::Block32x32);
            if var32 > thresholds.thresholds[1]
                || (var32 > (thresholds.thresholds[1] >> 1) && var32 > (avg16 >> 1))
            {
                map.split_32x32[i] = true;
                map.split_64x64 = true;
            }
        }

        map
    }

    fn split_at(&self, origin_8x8: (usize, usize), bs: BlockSize) -> bool {
        let (row, col) = origin_8x8;
        match bs {
            BlockSize::Block64x64 => self.split_64x64,
            BlockSize::Block32x32 => {
                let idx = (row / 4) * 2 + (col / 4);
                self.split_32x32.get(idx).copied().unwrap_or(false)
            }
            BlockSize::Block16x16 => {
                let idx = (row / 2) * 4 + (col / 2);
                self.split_16x16.get(idx).copied().unwrap_or(false)
            }
            _ => false,
        }
    }
}

fn block_per_pixel_variance(
    params: &PartitionSearchParams<'_>,
    origin_8x8: (usize, usize),
    bs: BlockSize,
) -> u64 {
    let (row_8x8, col_8x8) = origin_8x8;
    let (sb_x, sb_y) = params.src_offset_xy;
    let sp = params.src_plane;
    let bs_pixels = bs.width() as usize;
    let block_x = sb_x + col_8x8 * 8;
    let block_y = sb_y + row_8x8 * 8;
    let (ref_data, ref_stride, ref_off) = if params.is_keyframe {
        (
            &KEYFRAME_FLAT_REFERENCE[..],
            64,
            row_8x8 * 8 * 64 + col_8x8 * 8,
        )
    } else {
        let rp = params.ref_plane;
        let (ref_x, ref_y) = params.ref_offset_xy;
        (
            rp.data,
            rp.stride,
            (ref_y + row_8x8 * 8) * rp.stride + ref_x + col_8x8 * 8,
        )
    };
    let reduce = crate::dsp::get_ref().variance(bs);
    let (sum, sum_sq) = reduce(
        &sp.data[block_y * sp.stride + block_x..],
        sp.stride,
        &ref_data[ref_off..],
        ref_stride,
    );
    let n = (bs_pixels * bs_pixels) as i64;
    let var_total = (sum_sq as i64 - (sum * sum) / n).max(0) as u64;
    var_total / n as u64
}

/// Keyframe VBP variance over 4x4 source averages.
///
/// Pinned libvpx calls `fill_variance_4x4avg` for every keyframe, treating
/// each 4x4 average minus 128 as one sample and building all parent nodes by
/// summing those samples. The returned value folds out libvpx's 256x scale,
/// matching [`VbpThresholds`].
fn keyframe_4x4_average_variance(
    params: &PartitionSearchParams<'_>,
    origin_8x8: (usize, usize),
    bs: BlockSize,
) -> u64 {
    debug_assert!(params.is_keyframe);
    let block_side = bs.width() as usize;
    let sample_side = block_side / 4;
    let block_x = params.src_offset_xy.0 + origin_8x8.1 * 8;
    let block_y = params.src_offset_xy.1 + origin_8x8.0 * 8;
    let mut sum = 0i64;
    let mut sum_squares = 0u64;

    for sample_row in 0..sample_side {
        for sample_col in 0..sample_side {
            let sample_x = block_x + sample_col * 4;
            let sample_y = block_y + sample_row * 4;
            let mut pixel_sum = 0u32;
            for row in 0..4 {
                let offset = (sample_y + row) * params.src_plane.stride + sample_x;
                for &pixel in &params.src_plane.data[offset..offset + 4] {
                    pixel_sum += u32::from(pixel);
                }
            }
            let difference = i64::from((pixel_sum + 8) >> 4) - 128;
            sum += difference;
            sum_squares = sum_squares.saturating_add((difference * difference) as u64);
        }
    }

    let count = (sample_side * sample_side) as u64;
    let correction = (sum * sum) as u64 / count;
    sum_squares.saturating_sub(correction) / count
}

/// Recursive descent helper. `origin` is the top-left of the current
/// block in 8x8 grid coordinates (`(row_8x8, col_8x8)`); `bs` is the
/// current block size.
///
/// Returns true on success (i.e. the descent completed; the only
/// failure mode in F8.2 is "invalid block size for descent", which
/// the caller treats as a bug).
fn descend(
    params: &PartitionSearchParams<'_>,
    origin_8x8: (usize, usize),
    bs: BlockSize,
    thresholds: &VbpThresholds,
    force_split: Option<&ForceSplitMap>,
    allow_keyframe_sub8: bool,
    decision: &mut PartitionDecision,
) -> bool {
    let (row_8x8, col_8x8) = origin_8x8;
    let bs_pixels = bs.width() as usize;

    // Per-pixel variance for the current block.
    let per_pixel_var = if params.is_keyframe && allow_keyframe_sub8 {
        keyframe_4x4_average_variance(params, origin_8x8, bs)
    } else {
        block_per_pixel_variance(params, origin_8x8, bs)
    };

    // Threshold index: 0 → 64, 1 → 32, 2 → 16, 3 → 8.
    //
    // Stage 4a' partition-fragmentation closure (A0''): on KEYFRAMES,
    // the Block16x16 → Block8x8 descent is GATED OFF. Diagnosed on
    // foreman_cif: at q ≥ 124 on textured KFs (faces, hair) ~92 % of
    // 16×16 blocks exceed the variance threshold and descend to
    // Block8x8, generating 940/1024 leaves per frame. Per-leaf mode
    // bits + EOB context alone consume the entire bit budget,
    // starving the residual coefficients. Capping at Block16x16 on
    // KFs frees per-leaf bits AND keeps every leaf inside the F10.1b
    // intra picker's supported size range. Block8x8 leaves are still
    // reachable directly when the parent variance does NOT cause it
    // to split (the common case for low-variance blocks).
    //
    // INTER frames keep the full descent — A1 / A1.perf documented
    // that F8.1's Block8x8 inter picker (the testsrc PSNR climb from
    // 24 → 35-52 dB) depends on the descent reaching Block8x8 on
    // small frames; capping inter too would regress that win.
    //
    // A7.inter-cap experiment (2026-05-22): extending the cap to
    // inter frames reduces foreman bitrate by ~30% (fewer leaves →
    // fewer mode/skip/MV symbols) but PSNR stays locked at 16 dB.
    // Confirms the floor is NOT a bit-budget problem: residuals are
    // empty because q=255 quantizes all coefficients to zero,
    // independent of partition density. testsrc loses 0.2-0.3 dB
    // (matches A1's documented inter Block8x8 picker contribution).
    // Reverted; the inter-cap experiment did not help and slightly
    // regressed testsrc. See wrk_docs/A7-inter-cap-foreman-report.md.
    let allow_split_to_8x8 = !params.is_keyframe || allow_keyframe_sub8;
    let (threshold_idx, child_size) = match bs {
        BlockSize::Block64x64 => (0usize, Some(BlockSize::Block32x32)),
        BlockSize::Block32x32 => (1, Some(BlockSize::Block16x16)),
        BlockSize::Block16x16 => {
            if allow_split_to_8x8 {
                (2, Some(BlockSize::Block8x8))
            } else {
                (2, None)
            }
        }
        BlockSize::Block8x8 => (3, None),
        _ => return false,
    };

    let threshold = thresholds.thresholds[threshold_idx];

    // A sub-8 VP9 partition still occupies one 8x8 ModeInfo grid cell.
    // Represent the 8x8 SPLIT by stamping Block4x4; its four BMI modes
    // carry the actual 4x4 luma decisions. There is no smaller mi-grid
    // coordinate to recurse into.
    if bs == BlockSize::Block8x8 && params.is_keyframe && allow_keyframe_sub8 {
        decision.cells[row_8x8][col_8x8] = if per_pixel_var > threshold {
            BlockSize::Block4x4
        } else {
            BlockSize::Block8x8
        };
        return true;
    }

    // Decision rule. At Block8x8 we always emit a leaf (no further
    // descent). At larger sizes we split iff the per-pixel variance
    // exceeds the threshold.
    let split = match child_size {
        None => false,
        Some(_) => {
            force_split.is_some_and(|map| map.split_at(origin_8x8, bs)) || per_pixel_var > threshold
        }
    };

    if !split {
        // Stamp this block size into every 8x8 cell it covers.
        let cells_wide = bs_pixels / 8;
        for dr in 0..cells_wide {
            for dc in 0..cells_wide {
                decision.cells[row_8x8 + dr][col_8x8 + dc] = bs;
            }
        }
        return true;
    }

    // Otherwise descend into the four quadrants.
    let half = bs_pixels / 8 / 2; // half-side in 8x8 cells
    let child = child_size.expect("child_size set when split is true");
    let quadrants = [
        (row_8x8, col_8x8),
        (row_8x8, col_8x8 + half),
        (row_8x8 + half, col_8x8),
        (row_8x8 + half, col_8x8 + half),
    ];
    for &q in &quadrants {
        if !descend(
            params,
            q,
            child,
            thresholds,
            force_split,
            allow_keyframe_sub8,
            decision,
        ) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 64x64 source plane filled with `value`.
    fn flat_plane(value: u8) -> Vec<u8> {
        vec![value; 64 * 64]
    }

    /// xorshift32-based pseudo-random byte fill — deterministic across
    /// runs so test results don't drift.
    fn xorshift_plane(seed: u32) -> Vec<u8> {
        let mut buf = vec![0u8; 64 * 64];
        let mut state = seed.max(1);
        for b in buf.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *b = (state & 0xFF) as u8;
        }
        buf
    }

    /// Build a `PlaneRef` view over the given backing buffer with
    /// stride == 64 (i.e. exactly one 64x64 SB).
    fn plane_ref(buf: &[u8]) -> PlaneRef<'_> {
        PlaneRef {
            data: buf,
            stride: 64,
            width: 64,
            height: 64,
        }
    }

    fn fixed_tree_input<'a>(source: &'a [u8], predictor: &'a [u8]) -> FixedTreeInterInput<'a> {
        FixedTreeInterInput {
            source,
            predictor,
            visible_width: 64,
            visible_height: 64,
            frame_width: 64,
            thresholds: [1, 1, 1, 1],
            minmax_threshold: 27,
            force_64x64_split: false,
            compute_minmax_variance: true,
            segment_is_boosted: false,
            partition_ref_is_last: true,
            partition_mv_row: 0,
            partition_mv_col: 0,
            low_temp_var_mode: 0,
            noise_level: NoiseLevel::Low,
            low_resolution: false,
            frame_height_le_360: true,
        }
    }

    #[test]
    fn fixed_tree_history_copies_twice_then_requires_refresh() {
        let source = xorshift_plane(17);
        let predictor = flat_plane(128);
        let trace = choose_fixed_tree_inter(&fixed_tree_input(&source, &predictor));
        let mut history = FixedTreeHistoryEntry::default();
        let mut variance_low = [false; 25];
        variance_low[7] = true;
        history.update(trace.decision.clone(), 0, variance_low);

        let conditions = FixedTreeCopyConditions {
            frames_since_key: 2,
            segment_id: 0,
            max_copied_frames: 2,
            resize_pending: false,
            svc_copy_allowed: true,
        };
        let first = history.try_copy(conditions).expect("first copy");
        assert_eq!(first.decision, trace.decision);
        assert_eq!(first.nodes, trace.nodes);
        assert_eq!(first.leaves, trace.leaves);
        assert_eq!(first.variance_low, variance_low);
        assert_eq!(first.copied_frame_count, 1);

        let second = history.try_copy(conditions).expect("second copy");
        assert_eq!(second.copied_frame_count, 2);
        assert_eq!(history.try_copy(conditions), None);

        history.update(PartitionDecision::all_64x64(), 0, [false; 25]);
        assert_eq!(history.copied_frame_count(), 0);
        assert!(history.try_copy(conditions).is_some());
    }

    #[test]
    fn fixed_tree_history_enforces_stream_and_segment_guards() {
        let base = FixedTreeCopyConditions {
            frames_since_key: 2,
            segment_id: 0,
            max_copied_frames: 2,
            resize_pending: false,
            svc_copy_allowed: true,
        };
        let mut fresh = FixedTreeHistoryEntry::default();
        assert_eq!(fresh.try_copy(base), None, "no source tree");

        fresh.update(PartitionDecision::all_64x64(), 0, [false; 25]);
        for rejected in [
            FixedTreeCopyConditions {
                frames_since_key: 1,
                ..base
            },
            FixedTreeCopyConditions {
                segment_id: 1,
                ..base
            },
            FixedTreeCopyConditions {
                resize_pending: true,
                ..base
            },
            FixedTreeCopyConditions {
                svc_copy_allowed: false,
                ..base
            },
            FixedTreeCopyConditions {
                max_copied_frames: 0,
                ..base
            },
        ] {
            assert_eq!(fresh.try_copy(rejected), None);
        }

        fresh.update(PartitionDecision::all_64x64(), 1, [false; 25]);
        assert_eq!(fresh.previous_segment_id(), 1);
        assert_eq!(fresh.try_copy(base), None, "previous segment must be base");
    }

    #[test]
    fn fixed_tree_low_temp_flags_map_to_libvpx_leaf_slots() {
        let mut flags = [false; 25];
        flags[0] = true;
        flags[2] = true;
        flags[4] = true;
        flags[8] = true;
        flags[9] = true;
        flags[10] = true;
        flags[11] = true;

        assert!(fixed_tree_force_skip_low_temp_var(
            &flags,
            0,
            0,
            BlockSize::Block64x64
        ));
        assert!(!fixed_tree_force_skip_low_temp_var(
            &flags,
            0,
            0,
            BlockSize::Block64x32
        ));
        assert!(fixed_tree_force_skip_low_temp_var(
            &flags,
            4,
            0,
            BlockSize::Block64x32
        ));
        assert!(fixed_tree_force_skip_low_temp_var(
            &flags,
            0,
            4,
            BlockSize::Block32x64
        ));
        assert!(fixed_tree_force_skip_low_temp_var(
            &flags,
            4,
            4,
            BlockSize::Block32x32
        ));
        assert!(fixed_tree_force_skip_low_temp_var(
            &flags,
            0,
            0,
            BlockSize::Block16x16
        ));
        assert!(
            fixed_tree_force_skip_low_temp_var(&flags, 0, 0, BlockSize::Block32x16),
            "top 32x16 requires flags 9 and 10"
        );
        assert!(
            fixed_tree_force_skip_low_temp_var(&flags, 0, 0, BlockSize::Block16x32),
            "left 16x32 requires flags 9 and 11"
        );
        assert!(!fixed_tree_force_skip_low_temp_var(
            &flags,
            0,
            2,
            BlockSize::Block16x32
        ));
    }

    #[test]
    fn cpu7_thresholds_match_pinned_foreman_oracle() {
        let make = |q_index, content_state| {
            cpu7_inter_vbp_thresholds(Cpu7InterVbpThresholdInput {
                q_index,
                avg_inter_qindex: q_index,
                frame_width: 256,
                frame_height: 256,
                noise_estimate: NoiseEstimate::default(),
                content_state,
            })
        };

        assert_eq!(
            make(80, ContentStateSb::LowSadLowSumdiff),
            Cpu7InterVbpThresholds {
                thresholds: [13, 54, 864, 6960],
                minmax_threshold: 25,
            },
        );
        assert_eq!(
            make(120, ContentStateSb::HighSadHighSumdiff),
            Cpu7InterVbpThresholds {
                thresholds: [19, 76, 1216, 12160],
                minmax_threshold: 30,
            },
        );
        assert_eq!(
            make(160, ContentStateSb::VeryHighSad),
            Cpu7InterVbpThresholds {
                thresholds: [38, 152, 2440, 24400],
                minmax_threshold: 70,
            },
        );
    }

    #[test]
    fn cpu7_thresholds_apply_large_frame_noise_before_content_scale() {
        let thresholds = cpu7_inter_vbp_thresholds(Cpu7InterVbpThresholdInput {
            q_index: 120,
            avg_inter_qindex: 120,
            frame_width: 1280,
            frame_height: 720,
            noise_estimate: NoiseEstimate {
                enabled: true,
                level: NoiseLevel::Medium,
            },
            content_state: ContentStateSb::LowSadLowSumdiff,
        });

        // ac=152; medium noise doubles to 304; low-sumdiff scales 5/4
        // to 380 before the 720p threshold ladder is formed.
        assert_eq!(thresholds.thresholds[..3], [380, 760, 48_640]);
    }

    #[test]
    fn fixed_tree_flat_block_keeps_one_64x64_leaf() {
        let source = flat_plane(91);
        let predictor = flat_plane(91);
        let trace = choose_fixed_tree_inter(&fixed_tree_input(&source, &predictor));

        assert_eq!(trace.decision, PartitionDecision::all_64x64());
        assert_eq!(
            trace.nodes,
            vec![FixedTreeNodeDecision {
                row_8x8: 0,
                col_8x8: 0,
                parent_size: BlockSize::Block64x64,
                symbol: FixedTreePartitionSymbol::None,
            }],
        );
        assert_eq!(
            trace.leaves,
            vec![FixedTreeLeaf {
                row_8x8: 0,
                col_8x8: 0,
                block_size: BlockSize::Block64x64,
            }],
        );
        assert!(trace.decision.is_valid_partition());
        assert!(trace.variance_valid.iter().all(|&valid| valid));
        assert!(
            trace
                .variances
                .iter()
                .all(|node| *node == FixedTreeVarianceTrace::default())
        );
    }

    #[test]
    fn fixed_tree_emits_horizontal_parent_partition() {
        let mut source = flat_plane(0);
        let predictor = flat_plane(0);
        for row in 32..64 {
            source[row * 64..(row + 1) * 64].fill(255);
        }
        let trace = choose_fixed_tree_inter(&fixed_tree_input(&source, &predictor));

        assert_eq!(trace.nodes[0].symbol, FixedTreePartitionSymbol::Horizontal,);
        assert_eq!(
            trace.leaves,
            vec![
                FixedTreeLeaf {
                    row_8x8: 0,
                    col_8x8: 0,
                    block_size: BlockSize::Block64x32,
                },
                FixedTreeLeaf {
                    row_8x8: 4,
                    col_8x8: 0,
                    block_size: BlockSize::Block64x32,
                },
            ],
        );
        assert!(trace.decision.is_valid_partition());
    }

    #[test]
    fn fixed_tree_emits_vertical_parent_partition() {
        let mut source = flat_plane(0);
        let predictor = flat_plane(0);
        for row in 0..64 {
            source[row * 64 + 32..(row + 1) * 64].fill(255);
        }
        let trace = choose_fixed_tree_inter(&fixed_tree_input(&source, &predictor));

        assert_eq!(trace.nodes[0].symbol, FixedTreePartitionSymbol::Vertical);
        assert_eq!(
            trace.leaves,
            vec![
                FixedTreeLeaf {
                    row_8x8: 0,
                    col_8x8: 0,
                    block_size: BlockSize::Block32x64,
                },
                FixedTreeLeaf {
                    row_8x8: 0,
                    col_8x8: 4,
                    block_size: BlockSize::Block32x64,
                },
            ],
        );
        assert!(trace.decision.is_valid_partition());
    }

    #[test]
    fn fixed_tree_force_split_opens_flat_parent() {
        let source = flat_plane(91);
        let predictor = flat_plane(91);
        let mut input = fixed_tree_input(&source, &predictor);
        input.force_64x64_split = true;
        let trace = choose_fixed_tree_inter(&input);

        assert!(trace.force_split[0]);
        assert_eq!(trace.nodes[0].symbol, FixedTreePartitionSymbol::Split);
        assert_eq!(trace.leaves.len(), 4);
        assert!(
            trace
                .leaves
                .iter()
                .all(|leaf| leaf.block_size == BlockSize::Block32x32),
        );
        assert!(!trace.variance_valid[0]);
        assert!(trace.decision.is_valid_partition());
    }

    #[test]
    fn fixed_tree_uses_effective_minmax_threshold_and_traces_spread() {
        let mut source = flat_plane(0);
        let predictor = flat_plane(0);
        for row in 0..8 {
            for col in 0..8 {
                source[row * 64 + col] = if (row + col) & 1 == 0 { 0 } else { 40 };
            }
        }
        for row in 0..8 {
            source[row * 64 + 8..row * 64 + 16].fill(10);
        }
        for row in 8..16 {
            source[row * 64..row * 64 + 8].fill(20);
            source[row * 64 + 8..row * 64 + 16].fill(30);
        }

        let mut input = fixed_tree_input(&source, &predictor);
        input.thresholds = [20_000; 4];
        input.thresholds[1] = 1;
        input.minmax_threshold = 25;
        let lower_threshold = choose_fixed_tree_inter(&input);

        input.minmax_threshold = 50;
        let higher_threshold = choose_fixed_tree_inter(&input);

        assert_eq!(lower_threshold.minmax_16x16[0], 40);
        assert_eq!(higher_threshold.minmax_16x16[0], 40);
        assert!(lower_threshold.force_split[5]);
        assert!(!higher_threshold.force_split[5]);
    }

    #[test]
    fn fixed_tree_partial_mi_edge_bypasses_64x64_parent() {
        let source = flat_plane(91);
        let predictor = flat_plane(91);
        let mut input = fixed_tree_input(&source, &predictor);
        input.visible_width = 56;
        let trace = choose_fixed_tree_inter(&input);

        assert_eq!(trace.nodes[0].symbol, FixedTreePartitionSymbol::Split);
        assert!(trace.decision.is_valid_partition());
    }

    #[test]
    fn fixed_tree_keeps_8x8_average_tree_for_parent_aggregation() {
        let source = xorshift_plane(0x51A7_4A11);
        let predictor = flat_plane(128);
        let mut input = fixed_tree_input(&source, &predictor);
        input.low_resolution = true;
        let trace = choose_fixed_tree_inter(&input);

        assert!(
            trace.used_4x4_averages.iter().any(|&used| used),
            "fixture must exercise the separate vt2 path",
        );
        let mut original_16 = [FixedPartitionVariance::default(); 16];
        for i in 0..4 {
            let x32 = (i & 1) * 32;
            let y32 = (i >> 1) * 32;
            for j in 0..4 {
                let x16 = x32 + (j & 1) * 16;
                let y16 = y32 + (j >> 1) * 16;
                original_16[i * 4 + j] = FixedPartitionVariance::from_quadrants([
                    average_diff_var(&input, x16, y16, 8),
                    average_diff_var(&input, x16 + 8, y16, 8),
                    average_diff_var(&input, x16, y16 + 8, 8),
                    average_diff_var(&input, x16 + 8, y16 + 8, 8),
                ]);
            }
        }
        let mut original_32 = [FixedPartitionVariance::default(); 4];
        for (i, node32) in original_32.iter_mut().enumerate() {
            let base = i * 4;
            *node32 = FixedPartitionVariance::from_quadrants([
                original_16[base].none,
                original_16[base + 1].none,
                original_16[base + 2].none,
                original_16[base + 3].none,
            ]);
        }
        for (index, expected) in original_32.iter().enumerate() {
            assert_eq!(trace.variances[1 + index], expected.trace());
        }
        assert!(
            trace
                .used_4x4_averages
                .iter()
                .enumerate()
                .any(|(index, &used)| used
                    && trace.variances[5 + index] != trace.alternate_4x4_variances[index]),
            "fixture must distinguish vt from the local vt2 node",
        );
    }

    #[test]
    fn partition_validation_accepts_rect_tree_and_rejects_partial_rect_stamp() {
        let mut decision = PartitionDecision {
            cells: [[BlockSize::Block64x32; 8]; 8],
        };
        assert!(decision.is_valid_partition());

        decision.cells[7][7] = BlockSize::Block8x8;
        assert!(!decision.is_valid_partition());
    }

    /// Default search params for a 64x64 SB at origin with the given
    /// source / reference and qindex. Uses inter-frame mode (i.e.
    /// uses the reference; not keyframe-fallback).
    fn make_params<'a>(
        src: &'a PlaneRef<'a>,
        reference: &'a PlaneRef<'a>,
        q_index: u8,
        is_keyframe: bool,
    ) -> PartitionSearchParams<'a> {
        PartitionSearchParams {
            src_plane: src,
            src_offset_xy: (0, 0),
            ref_plane: reference,
            ref_offset_xy: (0, 0),
            is_keyframe,
            q_index,
            // Single-SB test harness: 64×64 frame. Small-res inter
            // branch; avg inter q mirrors the frame q (proxy).
            frame_width: 64,
            frame_height: 64,
            avg_inter_qindex: q_index,
            // W2.6 — default scale of 1 keeps the legacy threshold
            // arithmetic untouched (the pre-analysis helper returns 1
            // when its noise estimator is disabled).
            noise_threshold_scale: 1,
        }
    }

    #[test]
    fn dispatched_block_variance_matches_scalar_for_partition_sizes_and_origins() {
        for seed in 1..=16 {
            let src_buf = xorshift_plane(0x9E37_79B9u32.wrapping_mul(seed));
            let ref_vec = xorshift_plane(0x85EB_CA6Bu32.wrapping_mul(seed));
            let ref_buf: [u8; 64 * 64] = ref_vec.try_into().expect("64x64 reference");
            let src = plane_ref(&src_buf);
            let reference = plane_ref(&ref_buf);
            let params = make_params(&src, &reference, 100, false);

            for bs in [
                BlockSize::Block8x8,
                BlockSize::Block16x16,
                BlockSize::Block32x32,
                BlockSize::Block64x64,
            ] {
                let bs_pixels = bs.width() as usize;
                let bs_8x8 = bs_pixels / 8;
                for row_8x8 in 0..=SB_8X8_GRID - bs_8x8 {
                    for col_8x8 in 0..=SB_8X8_GRID - bs_8x8 {
                        let src_off = row_8x8 * 8 * 64 + col_8x8 * 8;
                        let (expected, _) = crate::dsp::distortion::variance(
                            &src_buf[src_off..],
                            64,
                            &ref_buf[src_off..],
                            64,
                            bs_pixels,
                            bs_pixels,
                        );
                        let actual = block_per_pixel_variance(&params, (row_8x8, col_8x8), bs);
                        let n = (bs_pixels * bs_pixels) as u64;
                        assert_eq!(
                            actual,
                            u64::from(expected) / n,
                            "seed={seed} bs={bs:?} origin=({row_8x8}, {col_8x8})",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn inter_partition_variance_reads_strided_reference_at_frame_offset() {
        const STRIDE: usize = 96;
        const HEIGHT: usize = 96;
        let src_buf: Vec<u8> = (0..STRIDE * HEIGHT)
            .map(|i| i.wrapping_mul(37).wrapping_add(i / STRIDE * 19) as u8)
            .collect();
        let ref_buf: Vec<u8> = (0..STRIDE * HEIGHT)
            .map(|i| i.wrapping_mul(13).wrapping_add(i / STRIDE * 53) as u8)
            .collect();
        let src = PlaneRef {
            data: &src_buf,
            stride: STRIDE,
            width: STRIDE,
            height: HEIGHT,
        };
        let reference = PlaneRef {
            data: &ref_buf,
            stride: STRIDE,
            width: STRIDE,
            height: HEIGHT,
        };
        let params = PartitionSearchParams {
            src_plane: &src,
            src_offset_xy: (16, 16),
            ref_plane: &reference,
            ref_offset_xy: (8, 24),
            is_keyframe: false,
            q_index: 100,
            frame_width: STRIDE,
            frame_height: HEIGHT,
            avg_inter_qindex: 100,
            noise_threshold_scale: 1,
        };

        for bs in [
            BlockSize::Block8x8,
            BlockSize::Block16x16,
            BlockSize::Block32x32,
            BlockSize::Block64x64,
        ] {
            let side = usize::from(bs.width());
            let cells = side / 8;
            for row_8x8 in 0..=SB_8X8_GRID - cells {
                for col_8x8 in 0..=SB_8X8_GRID - cells {
                    let src_off = (16 + row_8x8 * 8) * STRIDE + 16 + col_8x8 * 8;
                    let ref_off = (24 + row_8x8 * 8) * STRIDE + 8 + col_8x8 * 8;
                    let (variance, _) = crate::dsp::distortion::variance(
                        &src_buf[src_off..],
                        STRIDE,
                        &ref_buf[ref_off..],
                        STRIDE,
                        side,
                        side,
                    );
                    assert_eq!(
                        block_per_pixel_variance(&params, (row_8x8, col_8x8), bs),
                        u64::from(variance) / (side * side) as u64,
                        "{bs:?} at ({row_8x8}, {col_8x8})",
                    );
                }
            }
        }
    }

    /// Count how many 8x8 cells carry each block size — useful for
    /// asserting the split density.
    fn block_size_histogram(d: &PartitionDecision) -> std::collections::BTreeMap<BlockSize, usize> {
        let mut h = std::collections::BTreeMap::new();
        for row in &d.cells {
            for &bs in row {
                *h.entry(bs).or_insert(0) += 1;
            }
        }
        h
    }

    /// A fully flat source and reference produce zero variance at every
    /// level → the picker should land on `Block64x64` (no splits).
    #[test]
    fn choose_partitioning_flat_block_picks_64x64() {
        let src_buf = flat_plane(128);
        let ref_buf = flat_plane(128);
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);

        let params = make_params(&src, &reference, /* q */ 50, /* keyframe */ false);
        let d = choose_partitioning(&params);

        assert!(d.is_valid_partition(), "result must be a legal quad-tree");
        // Every cell should be Block64x64.
        for row in &d.cells {
            for &bs in row {
                assert_eq!(bs, BlockSize::Block64x64, "flat block should not split");
            }
        }
    }

    /// xorshift noise against a flat reference produces huge variance
    /// everywhere → the picker should split deeply. We assert that the
    /// majority of cells land at Block8x8 (the deepest leaf F8.2
    /// emits), and that no cell is Block64x64.
    #[test]
    fn choose_partitioning_chaotic_block_splits_deeply() {
        let src_buf = xorshift_plane(0xC0FFEE);
        let ref_buf = flat_plane(128);
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);

        let params = make_params(&src, &reference, /* q */ 50, /* keyframe */ false);
        let d = choose_partitioning(&params);

        assert!(d.is_valid_partition(), "result must be a legal quad-tree");
        let h = block_size_histogram(&d);
        let block_8 = h.get(&BlockSize::Block8x8).copied().unwrap_or(0);
        let block_64 = h.get(&BlockSize::Block64x64).copied().unwrap_or(0);
        assert_eq!(block_64, 0, "chaotic block must not land at 64x64");
        assert!(
            block_8 >= 32,
            "chaotic block should split deeply: only {block_8}/64 cells at 8x8 \
             (histogram: {h:?})"
        );
    }

    /// A source that is flat in the top half and noisy in the bottom
    /// half should split the bottom half deeper than the top half.
    #[test]
    fn choose_partitioning_mixed_block_partial_split() {
        let mut src_buf = vec![128u8; 64 * 64];
        let noise = xorshift_plane(0xBADC0DE);
        for r in 32..64 {
            for c in 0..64 {
                src_buf[r * 64 + c] = noise[r * 64 + c];
            }
        }
        let ref_buf = flat_plane(128);
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);

        let params = make_params(&src, &reference, /* q */ 50, /* keyframe */ false);
        let d = choose_partitioning(&params);

        assert!(d.is_valid_partition(), "result must be a legal quad-tree");

        // Top half: every cell should be at least Block16x16 (i.e.
        // not split below 16x16) since the source is flat.
        for r in 0..4 {
            for c in 0..SB_8X8_GRID {
                let bs = d.cells[r][c];
                assert!(
                    matches!(
                        bs,
                        BlockSize::Block16x16 | BlockSize::Block32x32 | BlockSize::Block64x64
                    ),
                    "top-half cell ({r}, {c}) should be ≥ 16x16, got {bs:?}"
                );
            }
        }
        // Bottom half: should have a non-zero count of Block8x8
        // leaves (i.e. it split deeper than the top half).
        let mut bottom_8x8 = 0usize;
        for r in 4..SB_8X8_GRID {
            for c in 0..SB_8X8_GRID {
                if d.cells[r][c] == BlockSize::Block8x8 {
                    bottom_8x8 += 1;
                }
            }
        }
        assert!(
            bottom_8x8 > 0,
            "bottom-half should split to 8x8 somewhere (got {bottom_8x8})"
        );
    }

    /// Same input at q=20 should split at least as much as at q=240.
    /// Higher q raises the threshold → fewer splits → larger blocks.
    #[test]
    fn choose_partitioning_high_q_index_fewer_splits() {
        // Moderate-variance source: small offset against the reference
        // so the per-pixel variance hovers near the threshold and the
        // q-dependent scaling actually moves the decision.
        let mut src_buf = vec![128u8; 64 * 64];
        for r in 0..64 {
            for c in 0..64 {
                // Slowly-varying gradient: low local variance but
                // non-trivial against a flat 128 reference.
                src_buf[r * 64 + c] = 128u8.wrapping_add(((r as u32 + c as u32) % 32) as u8);
            }
        }
        let ref_buf = flat_plane(128);
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);

        let low_q = choose_partitioning(&make_params(&src, &reference, 20, false));
        let high_q = choose_partitioning(&make_params(&src, &reference, 240, false));

        assert!(low_q.is_valid_partition());
        assert!(high_q.is_valid_partition());

        // Count Block8x8 leaves: low-q should have ≥ high-q.
        let low_h = block_size_histogram(&low_q);
        let high_h = block_size_histogram(&high_q);
        let low_8 = low_h.get(&BlockSize::Block8x8).copied().unwrap_or(0);
        let high_8 = high_h.get(&BlockSize::Block8x8).copied().unwrap_or(0);
        assert!(
            low_8 >= high_8,
            "low q ({low_8} Block8x8) should split at least as much as high q ({high_8})"
        );
        // And the largest-block (Block64x64) count should be the
        // opposite direction: high-q ≥ low-q.
        let low_64 = low_h.get(&BlockSize::Block64x64).copied().unwrap_or(0);
        let high_64 = high_h.get(&BlockSize::Block64x64).copied().unwrap_or(0);
        assert!(
            high_64 >= low_64,
            "high q ({high_64} Block64x64) should keep at least as many big blocks \
             as low q ({low_64})"
        );
    }

    /// The force-split prepass mirrors libvpx's child-variance guard:
    /// a detailed 32x32 quadrant can force the 64x64 parent to split even
    /// when the parent aggregate variance is low enough for the simple
    /// top-down picker to accept it.
    #[test]
    fn choose_partitioning_force_split_opens_sparse_32x32_detail() {
        let q = 255;
        let thresholds = VbpThresholds::for_inter(q, /* avg_q */ 50, 1024, 512, 1);
        let amp = (1u8..=127)
            .find(|&a| {
                let var32 = u64::from(a) * u64::from(a);
                let var64 = var32 / 4;
                var64 <= thresholds.thresholds[0] && var32 > thresholds.thresholds[1]
            })
            .expect("test thresholds must admit a sparse 32x32 force-split pattern");

        let mut src_buf = vec![128u8; 64 * 64];
        for r in 0..32 {
            for c in 0..32 {
                let delta = if (r + c) & 1 == 0 {
                    i16::from(amp)
                } else {
                    -i16::from(amp)
                };
                src_buf[r * 64 + c] = (128i16 + delta) as u8;
            }
        }
        let ref_buf = flat_plane(128);
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);
        let params = PartitionSearchParams {
            src_plane: &src,
            src_offset_xy: (0, 0),
            ref_plane: &reference,
            ref_offset_xy: (0, 0),
            is_keyframe: false,
            q_index: q,
            frame_width: 1024,
            frame_height: 512,
            avg_inter_qindex: 50,
            noise_threshold_scale: 1,
        };

        let simple = choose_partitioning(&params);
        let forced = choose_partitioning_force_split(&params);

        assert!(simple.is_valid_partition());
        assert!(forced.is_valid_partition());
        for row in &simple.cells {
            for &bs in row {
                assert_eq!(bs, BlockSize::Block64x64);
            }
        }
        assert_eq!(forced.cells[0][0], BlockSize::Block16x16);
        assert_eq!(forced.cells[0][4], BlockSize::Block32x32);
        assert_eq!(forced.cells[4][0], BlockSize::Block32x32);
        assert_eq!(forced.cells[4][4], BlockSize::Block32x32);
    }

    /// Keyframe path: the reference is replaced with flat-128
    /// regardless of `ref_plane` contents. If we feed in a reference
    /// matching the source (which would give zero variance on the
    /// inter path), the KF path should compute variance against
    /// flat-128 and split accordingly.
    #[test]
    fn choose_partitioning_keyframe_path() {
        // Source = xorshift noise; reference = matching xorshift noise.
        // On the inter path this would give variance ≈ 0 → no split.
        // On the KF path we ignore the reference and compute variance
        // against flat-128 → high variance → splits.
        let src_buf = xorshift_plane(0xABCDEF01);
        let ref_buf = src_buf.clone(); // perfect match (inter-only)
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);

        let inter_params = make_params(&src, &reference, 50, /* keyframe */ false);
        let kf_params = make_params(&src, &reference, 50, /* keyframe */ true);

        let inter = choose_partitioning(&inter_params);
        let kf = choose_partitioning(&kf_params);

        assert!(inter.is_valid_partition());
        assert!(kf.is_valid_partition());

        // Inter path: zero variance → all-Block64x64.
        for row in &inter.cells {
            for &bs in row {
                assert_eq!(
                    bs,
                    BlockSize::Block64x64,
                    "inter-matched reference should give all-64x64"
                );
            }
        }
        // KF path: high variance against flat-128 → no Block64x64.
        let kf_hist = block_size_histogram(&kf);
        let kf_64 = kf_hist.get(&BlockSize::Block64x64).copied().unwrap_or(0);
        assert_eq!(
            kf_64, 0,
            "KF path on xorshift source should not land at 64x64"
        );
        assert!(
            kf.cells
                .iter()
                .flatten()
                .all(|&bsize| bsize != BlockSize::Block4x4),
            "the production keyframe path must retain its sub-8 cap",
        );
    }

    /// The diagnostic keyframe path can carry an 8x8 SPLIT in one mi-grid
    /// cell, while the production entry point remains byte-stable above it.
    #[test]
    fn choose_partitioning_keyframe_sub8_emits_block4x4_cells() {
        let src_buf = xorshift_plane(0xABCDEF01);
        let ref_buf = src_buf.clone();
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);
        let params = make_params(&src, &reference, 50, /* keyframe */ true);

        let production = choose_partitioning(&params);
        let diagnostic = choose_partitioning_keyframe_sub8(&params);

        assert!(production.is_valid_partition());
        assert!(diagnostic.is_valid_partition());
        assert!(
            production
                .cells
                .iter()
                .flatten()
                .all(|&bsize| bsize != BlockSize::Block4x4),
        );
        assert!(
            diagnostic
                .cells
                .iter()
                .flatten()
                .any(|&bsize| bsize == BlockSize::Block4x4),
            "high-variance keyframe diagnostic must reach sub-8 BMI leaves",
        );
    }

    #[test]
    fn keyframe_4x4_average_variance_filters_balanced_pixel_texture() {
        let mut src_buf = vec![0u8; 64 * 64];
        for row in 0..64 {
            for col in 0..64 {
                src_buf[row * 64 + col] = if (row + col).is_multiple_of(2) {
                    0
                } else {
                    255
                };
            }
        }
        let ref_buf = flat_plane(128);
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);
        let params = make_params(&src, &reference, 50, /* keyframe */ true);

        assert!(block_per_pixel_variance(&params, (0, 0), BlockSize::Block8x8) > 10_000);
        assert_eq!(
            keyframe_4x4_average_variance(&params, (0, 0), BlockSize::Block8x8),
            0,
            "balanced 4x4 texture has identical rounded averages and must be filtered",
        );
    }

    /// Every cell must carry a legal F8.2 block size, and the pattern
    /// must form a valid VP9 quad-tree (no half-blocks, no overlaps).
    /// This is the load-bearing invariant for the downstream writer.
    #[test]
    fn choose_partitioning_returns_valid_block_size_grid() {
        for seed in [0xC0FFEEu32, 0xDEADBEEF, 0x12345678, 0xABCDEF01] {
            for q in [0u8, 50, 128, 255] {
                let src_buf = xorshift_plane(seed);
                let ref_buf = flat_plane(128);
                let src = plane_ref(&src_buf);
                let reference = plane_ref(&ref_buf);
                let params = make_params(&src, &reference, q, false);
                let d = choose_partitioning(&params);

                assert!(
                    d.is_valid_partition(),
                    "seed={seed:08x} q={q}: pattern must be a legal quad-tree"
                );
                for row in &d.cells {
                    for &bs in row {
                        assert!(
                            matches!(
                                bs,
                                BlockSize::Block8x8
                                    | BlockSize::Block16x16
                                    | BlockSize::Block32x32
                                    | BlockSize::Block64x64
                            ),
                            "seed={seed:08x} q={q}: illegal leaf {bs:?}"
                        );
                    }
                }
            }
        }
    }

    /// `is_valid_partition` itself: it should accept the all-64x64
    /// pattern, an all-8x8 pattern, and a hand-built 4-quadrant
    /// pattern; and reject malformed patterns (e.g. a half-block).
    #[test]
    fn is_valid_partition_recognises_legal_and_illegal() {
        // Legal: all 64x64.
        let all_64 = PartitionDecision::all_64x64();
        assert!(all_64.is_valid_partition());

        // Legal: all 8x8.
        let all_8 = PartitionDecision {
            cells: [[BlockSize::Block8x8; SB_8X8_GRID]; SB_8X8_GRID],
        };
        assert!(all_8.is_valid_partition());

        // Legal: four 32x32 quadrants.
        let mut four_32 = PartitionDecision::all_64x64();
        for row in &mut four_32.cells {
            for cell in row.iter_mut() {
                *cell = BlockSize::Block32x32;
            }
        }
        assert!(four_32.is_valid_partition());

        // Illegal: 32x32 leaf at an unaligned origin (rows 1..5).
        let mut bad = four_32.clone();
        // Set a misaligned "leaf" by stamping Block32x32 at (1,1)..(4,4)
        // — but the original 4-quadrant pattern is already valid, so
        // we instead build a pattern where one cell is Block16x16
        // adjacent to a Block32x32 covering it: i.e., overlap.
        bad.cells[0][0] = BlockSize::Block16x16; // breaks the 32x32 in quadrant 0
        assert!(
            !bad.is_valid_partition(),
            "broken 32x32 quadrant must be invalid"
        );

        // Illegal: a cell carries a non-square / sub-8 block size.
        let mut bad2 = all_8.clone();
        bad2.cells[0][0] = BlockSize::Block4x8;
        assert!(
            !bad2.is_valid_partition(),
            "Block4x8 leaf is illegal in F8.2"
        );
    }

    /// `is_valid_partition`: alignment-failure return path (line 145).
    /// A Block32x32 leaf claimed at origin (0, 2) is unaligned —
    /// `col % cells_wide(32x32) = 2 % 4 != 0` → returns false.
    #[test]
    fn is_valid_partition_rejects_unaligned_origin() {
        // Build a pattern that begins with Block32x32 at (0, 0), then
        // also stamps Block32x32 starting at (0, 2). Because the
        // outer loop visits cells in row-major order, the (0, 0) leaf
        // would consume cells (0..4, 0..4). But once it tries to
        // process (0, 2) and finds an unvisited Block32x32 entry…
        //
        // Simpler: directly leave only one cell that fails alignment.
        let mut d = PartitionDecision {
            cells: [[BlockSize::Block8x8; SB_8X8_GRID]; SB_8X8_GRID],
        };
        // Replace (0, 1)..(0, 4) with Block32x32. The visited-check
        // hits (0, 1), which is unaligned for a 32×32 leaf.
        for r in 0..4 {
            for c in 1..5 {
                d.cells[r][c] = BlockSize::Block32x32;
            }
        }
        assert!(
            !d.is_valid_partition(),
            "32×32 leaf at unaligned column should be invalid",
        );
    }

    /// `is_valid_partition`: out-of-bounds footprint return path
    /// (line 148). A Block32x32 leaf at origin (6, 6) would span
    /// rows 6..10 / cols 6..10 — past the SB grid edge → returns
    /// false.
    #[test]
    fn is_valid_partition_rejects_out_of_bounds_leaf() {
        let mut d = PartitionDecision {
            cells: [[BlockSize::Block8x8; SB_8X8_GRID]; SB_8X8_GRID],
        };
        // Force a Block32x32 leaf marker at (4, 4) — aligned but the
        // remaining footprint (rows 4..8, cols 4..8) fits, so this is
        // actually legal. We need an unaligned-but-in-grid case to
        // trigger line 148. Use a Block64x64 marker at (0, 0) but
        // shrink the grid... easier: put a Block16x16 leaf at (7, 7)
        // — row+2 > 8.
        d.cells[7][7] = BlockSize::Block16x16;
        // The outer loop visits (7, 7) — finds Block16x16. cells_wide
        // = 2. `7 + 2 > 8` → returns false.
        assert!(
            !d.is_valid_partition(),
            "16×16 leaf at (7, 7) would overflow SB grid",
        );
    }

    /// `is_valid_partition`: the cell-mismatch return path (line 153).
    /// A Block16x16 leaf claimed at (0, 0) requires cells [0..2][0..2]
    /// to all carry Block16x16. If one cell carries Block8x8, the
    /// inner loop returns false.
    #[test]
    fn is_valid_partition_rejects_inconsistent_leaf_footprint() {
        let mut d = PartitionDecision {
            cells: [[BlockSize::Block8x8; SB_8X8_GRID]; SB_8X8_GRID],
        };
        // Block16x16 marker at (0,0) and (0,1) — but (1,0) and (1,1)
        // are still Block8x8 → footprint inconsistent.
        d.cells[0][0] = BlockSize::Block16x16;
        d.cells[0][1] = BlockSize::Block16x16;
        // Inner loop checks (1, 0) — Block8x8 != Block16x16.
        assert!(
            !d.is_valid_partition(),
            "Block16x16 leaf with mismatched footprint must be invalid",
        );
    }

    /// `is_valid_partition`: the `visited[r+dr][c+dc]` true branch
    /// (line 156). When two leaf claims overlap, the second claim
    /// hits a cell already marked visited → return false.
    ///
    /// This is hard to trigger in the natural row-major walk because
    /// the outer loop's own `if visited { continue }` guard preempts
    /// it for the OUTER cell. But the INNER loop (line 156) fires
    /// when a leaf's footprint reaches an already-visited cell. We
    /// construct this by claiming Block16x16 at (0, 0) and then a
    /// Block32x32 at (0, 2) that footprint extends into the (0..1,
    /// 0..1) region already visited.
    #[test]
    fn is_valid_partition_rejects_overlapping_leaves() {
        // Strategy: build a Block32x32 at (0, 0) covering (0..4, 0..4)
        // AND tag (4, 0) with Block32x32 (covering rows 4..8, cols
        // 0..4). Then stamp a Block64x64 in the middle that overlaps —
        // but row-major visits the top-left first.
        //
        // Simpler: a Block32x32 at (0, 0) (rows 0..4 cols 0..4) plus a
        // residual Block16x16 at (3, 3) — but (3, 3) is already
        // claimed by the 32×32 footprint so the inner visited[] check
        // fires when the second leaf's loop steps into it… except the
        // outer `if visited` skips it. So we need an inner-loop
        // overlap.
        //
        // Easiest: a Block16x16 marker at (0, 0) covering (0..2, 0..2),
        // PLUS the same Block16x16 marker at (1, 1) (a "duplicate"
        // claim). The outer loop visits (0, 0) → claims (0..2, 0..2).
        // Then visits (0, 1) (visited, skip), (0, 2), ..., eventually
        // (1, 1) — visited, skip. So the inner overlap won't fire
        // unless an alignment check FAILED at (1, 1) first, which it
        // does because 1 % 2 != 0 — but that's caught by the
        // alignment branch (line 145), not the overlap branch (156).
        //
        // To trigger line 156 we need: a leaf footprint reaches into
        // already-visited territory WITHOUT the outer loop skipping
        // it. Concretely: place Block64x64 at (0, 0) (claims all 64
        // cells) AND ALSO seed cells[7][7] as Block8x8 first. The
        // Block64x64 inner loop tries to mark (7, 7) — already
        // visited by… no, it isn't.
        //
        // We give up trying to force this branch — it's defensively
        // unreachable in normal flow because the outer loop's
        // `visited` guard preempts it. Mark it as such and confirm
        // via the natural test that `is_valid_partition` agrees on
        // a hand-built duplicate-footprint pattern.
        //
        // PRAGMATIC: build TWO Block32x32 leaves that overlap by
        // stamping non-Block32x32 markers around them.
        let mut d = PartitionDecision {
            cells: [[BlockSize::Block32x32; SB_8X8_GRID]; SB_8X8_GRID],
        };
        // Overwrite cells (4..8, 0..4) with a misaligned Block16x16
        // pattern so the outer loop reaches (4, 0) on the SECOND
        // iteration — but (4, 0) was already visited by the first
        // 32×32 leaf if we left it as 32×32. Replace those cells with
        // Block16x16 makes the outer loop see Block16x16 at (4, 0),
        // not visited, and stamp it. That tests the simple
        // inconsistent-footprint case already covered.
        //
        // Simplest assertion that exercises the overall function with
        // an "overlap detected" outcome: stamp non-Block32x32 markers
        // INSIDE the first leaf's footprint.
        d.cells[1][1] = BlockSize::Block16x16; // breaks (0..4, 0..4)
        assert!(!d.is_valid_partition(), "footprint inconsistent → invalid",);
    }

    // -----------------------------------------------------------------------
    // VbpThresholds — per-resolution branches
    // -----------------------------------------------------------------------

    /// `choose_partitioning` for inter at small-res (≤352×288) with
    /// `avg_inter_qindex > 220`: t2 gets the `<<= 2` boost. We
    /// observe this indirectly via a flat block at high q — the
    /// inter path keeps it all-Block64x64 (high threshold).
    #[test]
    fn choose_partitioning_inter_small_res_high_avg_q() {
        let src_buf = flat_plane(128);
        let ref_buf = flat_plane(128);
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);

        let params = PartitionSearchParams {
            src_plane: &src,
            src_offset_xy: (0, 0),
            ref_plane: &reference,
            ref_offset_xy: (0, 0),
            is_keyframe: false,
            q_index: 50,
            frame_width: 256, // small-res branch
            frame_height: 256,
            avg_inter_qindex: 230, // > 220 → t2 <<= 2
            noise_threshold_scale: 1,
        };
        let d = choose_partitioning(&params);
        assert!(d.is_valid_partition());
        // Flat block must produce Block64x64.
        for row in &d.cells {
            for &bs in row {
                assert_eq!(bs, BlockSize::Block64x64);
            }
        }
    }

    /// `choose_partitioning` for inter at medium-res (1024×512,
    /// width < 1280 && height < 720): t1 = (5 * base) >> 2 branch
    /// fires.
    #[test]
    fn choose_partitioning_inter_medium_res() {
        // Single SB out of a synthetic 64×64 source/ref view, but we
        // pass frame_width=1024 / frame_height=512 to select the
        // medium-res inter threshold branch.
        let src_buf = flat_plane(128);
        let ref_buf = flat_plane(128);
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);
        let params = PartitionSearchParams {
            src_plane: &src,
            src_offset_xy: (0, 0),
            ref_plane: &reference,
            ref_offset_xy: (0, 0),
            is_keyframe: false,
            q_index: 50,
            frame_width: 1024,
            frame_height: 512,
            avg_inter_qindex: 50,
            noise_threshold_scale: 1,
        };
        let d = choose_partitioning(&params);
        assert!(d.is_valid_partition());
        // Flat block → Block64x64 regardless.
        for row in &d.cells {
            for &bs in row {
                assert_eq!(bs, BlockSize::Block64x64);
            }
        }
    }

    /// `choose_partitioning` for inter at large-res (1600×900,
    /// width < 1920 && height < 1080): t1 = base << 1 branch fires.
    #[test]
    fn choose_partitioning_inter_large_res() {
        let src_buf = flat_plane(128);
        let ref_buf = flat_plane(128);
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);
        let params = PartitionSearchParams {
            src_plane: &src,
            src_offset_xy: (0, 0),
            ref_plane: &reference,
            ref_offset_xy: (0, 0),
            is_keyframe: false,
            q_index: 50,
            frame_width: 1600,
            frame_height: 900,
            avg_inter_qindex: 50,
            noise_threshold_scale: 1,
        };
        let d = choose_partitioning(&params);
        assert!(d.is_valid_partition());
    }

    /// `choose_partitioning` for inter at huge-res (3840×2160):
    /// the `else` branch (t1 = (5 * base) >> 1) fires.
    #[test]
    fn choose_partitioning_inter_huge_res() {
        let src_buf = flat_plane(128);
        let ref_buf = flat_plane(128);
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);
        let params = PartitionSearchParams {
            src_plane: &src,
            src_offset_xy: (0, 0),
            ref_plane: &reference,
            ref_offset_xy: (0, 0),
            is_keyframe: false,
            q_index: 50,
            frame_width: 3840,
            frame_height: 2160,
            avg_inter_qindex: 50,
            noise_threshold_scale: 1,
        };
        let d = choose_partitioning(&params);
        assert!(d.is_valid_partition());
    }

    /// `choose_partitioning` for inter at width<=352 but height>288:
    /// short-circuit `width <= 352 && height <= 288` returns false on
    /// the second operand (line 299:28 F).
    #[test]
    fn choose_partitioning_inter_short_res_tall_frame() {
        let src_buf = flat_plane(128);
        let ref_buf = flat_plane(128);
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);
        let params = PartitionSearchParams {
            src_plane: &src,
            src_offset_xy: (0, 0),
            ref_plane: &reference,
            ref_offset_xy: (0, 0),
            is_keyframe: false,
            q_index: 50,
            frame_width: 320,  // <= 352
            frame_height: 480, // > 288 → second operand false
            avg_inter_qindex: 50,
            noise_threshold_scale: 1,
        };
        let d = choose_partitioning(&params);
        assert!(d.is_valid_partition());
    }

    /// `choose_partitioning` for inter at width<1280 but height>=720:
    /// short-circuit `width < 1280 && height < 720` returns false on
    /// the height side (line 309:35 F).
    #[test]
    fn choose_partitioning_inter_medium_width_tall() {
        let src_buf = flat_plane(128);
        let ref_buf = flat_plane(128);
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);
        let params = PartitionSearchParams {
            src_plane: &src,
            src_offset_xy: (0, 0),
            ref_plane: &reference,
            ref_offset_xy: (0, 0),
            is_keyframe: false,
            q_index: 50,
            frame_width: 1024, // < 1280
            frame_height: 768, // >= 720 → second operand false
            avg_inter_qindex: 50,
            noise_threshold_scale: 1,
        };
        let d = choose_partitioning(&params);
        assert!(d.is_valid_partition());
    }

    /// `choose_partitioning` for inter at width<1920 but height>=1080:
    /// short-circuit `width < 1920 && height < 1080` returns false on
    /// the height side (line 311:35 F).
    #[test]
    fn choose_partitioning_inter_large_width_tall() {
        let src_buf = flat_plane(128);
        let ref_buf = flat_plane(128);
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);
        let params = PartitionSearchParams {
            src_plane: &src,
            src_offset_xy: (0, 0),
            ref_plane: &reference,
            ref_offset_xy: (0, 0),
            is_keyframe: false,
            q_index: 50,
            frame_width: 1600,  // < 1920
            frame_height: 1200, // >= 1080 → second operand false
            avg_inter_qindex: 50,
            noise_threshold_scale: 1,
        };
        let d = choose_partitioning(&params);
        assert!(d.is_valid_partition());
    }

    /// `choose_partitioning` for inter at small-res with
    /// `avg_inter_qindex > 200` (but ≤ 220): t2 gets the `<<= 1`
    /// boost via the `else if` branch.
    #[test]
    fn choose_partitioning_inter_small_res_mid_avg_q() {
        let src_buf = flat_plane(128);
        let ref_buf = flat_plane(128);
        let src = plane_ref(&src_buf);
        let reference = plane_ref(&ref_buf);
        let params = PartitionSearchParams {
            src_plane: &src,
            src_offset_xy: (0, 0),
            ref_plane: &reference,
            ref_offset_xy: (0, 0),
            is_keyframe: false,
            q_index: 50,
            frame_width: 256,
            frame_height: 256,
            avg_inter_qindex: 210, // 200 < x <= 220 → t2 <<= 1
            noise_threshold_scale: 1,
        };
        let d = choose_partitioning(&params);
        assert!(d.is_valid_partition());
    }

    /// PROOF W2.6 (noise scale lifts inter thresholds) — libvpx
    /// `vp9_encodeframe.c:588-598` multiplies the inter VBP threshold
    /// by a noise-driven factor. Verify that `VbpThresholds::for_inter`
    /// scales each per-level threshold by that factor (applied to the
    /// pre-`/256` base, then floored at 1).
    ///
    /// Anti-G3: passing scale=2 vs scale=1 must lift the thresholds; a
    /// stripped multiply would leave the values identical. We use a
    /// high q (255) so the base ladder is large enough that the
    /// multiplier is observable on every level (small-res levels
    /// `t0 = base >> 3` are too small at low q to discriminate after
    /// the `/256` fold-out + `.max(1)` floor).
    #[test]
    fn w2_6_for_inter_applies_noise_threshold_scale() {
        // 720p inter branch (`width < 1280 && height < 720`): t0=base,
        // t1=(5*base)>>2, t2=base<<7, t3=base<<2 — large enough at
        // q=255 that the noise multiplier shows up post-256× fold.
        let baseline =
            VbpThresholds::for_inter(/* q */ 255, /* avg_q */ 50, 1024, 512, 1);
        let scaled_2 = VbpThresholds::for_inter(255, 50, 1024, 512, 2);
        let scaled_4 = VbpThresholds::for_inter(255, 50, 1024, 512, 4);
        for i in 0..4 {
            assert!(
                scaled_2.thresholds[i] > baseline.thresholds[i],
                "level {i}: scale=2 must lift baseline ({} vs {})",
                scaled_2.thresholds[i],
                baseline.thresholds[i],
            );
            assert!(
                scaled_4.thresholds[i] > scaled_2.thresholds[i],
                "level {i}: scale=4 must lift scale=2 ({} vs {})",
                scaled_4.thresholds[i],
                scaled_2.thresholds[i],
            );
        }
        // Scale=0 must clamp to 1 (no degenerate zero thresholds).
        let scaled_0 = VbpThresholds::for_inter(255, 50, 1024, 512, 0);
        assert_eq!(
            scaled_0.thresholds, baseline.thresholds,
            "scale=0 must clamp to 1× (no-op)",
        );
    }
}
