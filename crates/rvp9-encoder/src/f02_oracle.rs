//! F.0.2 — per-block libvpx-vs-rvp9 picker-decision oracle sink.
//!
//! Mirrors the env-gated thread-local pattern of [`crate::predstat`],
//! [`crate::skipstat`], [`crate::e12stat`], and
//! [`rvp9_core::mv_search::e12stat`]. When the env var `RVP9_F02_ORACLE`
//! is unset, every hook is one relaxed atomic load and zero allocations.
//!
//! Hook sites — in the encoder's picker driver, not the picker itself
//! (the picker lives in `rvp9-core` and would create a circular
//! dependency if it called into this module). See HLD §3.2 — the hook
//! "snapshots the winner into the sink" right after the picker returns,
//! which is functionally identical to instrumenting inside the picker
//! and avoids the dependency cycle.
//!
//! Output: end-of-frame, the harness binary calls [`drain_frame`] to
//! pull the accumulated `Vec<BlockDecision>` and either serialize each
//! to JSONL (via [`BlockDecision::write_jsonl`]) or copy out for an
//! in-process comparison.
//!
//! See `wrk_docs/Track_F/F0.2_HLD_per_block_oracle.md` (§3.2, §3.4).

use core::sync::atomic::{AtomicBool, Ordering};
use std::cell::RefCell;
use std::fmt::Write as _;
use std::io;

use rvp9_core::bitstream::frame::InterpFilter;
use rvp9_core::block::BlockSize;
use rvp9_core::intra_pred::IntraMode;
use rvp9_core::mode_decision::inter::{NewMvGateReason, NewMvSearchTrace};
use rvp9_core::mode_info::{InterMode, RefFrame};
use rvp9_core::transforms::TxSize;

/// Global enable flag, set once at process startup by
/// [`init_from_env`]. One relaxed load per picker call site when off.
static F02_ON: AtomicBool = AtomicBool::new(false);

/// Combined intra/inter mode discriminant for the oracle schema.
///
/// HLD §3.4 calls for one `mode` field of shape "DC/V/H/.../TM |
/// NEAREST/NEAR/ZERO/NEW". The schema column is flat (no
/// intra-vs-inter envelope), so this enum unifies the two libvpx
/// mode spaces into a single tag.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BlockDecisionMode {
    /// Intra mode — wraps [`IntraMode`] (DC/V/H/D45/D135/D117/D153/D207/D63/TM).
    Intra(IntraMode),
    /// Inter mode — wraps [`InterMode`] (NEARESTMV/NEARMV/ZEROMV/NEWMV).
    Inter(InterMode),
}

impl BlockDecisionMode {
    /// Stable lowercase tag for the JSONL `mode` field. Mirrors the
    /// libvpx printf side of the oracle so cross-leg diff is a plain
    /// string equality check.
    #[must_use]
    pub fn as_tag(self) -> &'static str {
        match self {
            Self::Intra(m) => match m {
                IntraMode::Dc => "DC",
                IntraMode::V => "V",
                IntraMode::H => "H",
                IntraMode::D45 => "D45",
                IntraMode::D135 => "D135",
                IntraMode::D117 => "D117",
                IntraMode::D153 => "D153",
                IntraMode::D207 => "D207",
                IntraMode::D63 => "D63",
                IntraMode::Tm => "TM",
            },
            Self::Inter(m) => match m {
                InterMode::NearestMv => "NEARESTMV",
                InterMode::NearMv => "NEARMV",
                InterMode::ZeroMv => "ZEROMV",
                InterMode::NewMv => "NEWMV",
            },
        }
    }
}

/// One LAST-reference candidate in the diagnostic mode-order
/// `[ZERO, NEAREST, NEAR, NEW]`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct CandidateProbe {
    /// Whether the encoder evaluated this candidate.
    pub available: bool,
    /// Candidate MV row in 1/8-pel units.
    pub mv_row: i16,
    /// Candidate MV column in 1/8-pel units.
    pub mv_col: i16,
    /// Total candidate rate in `vp9_prob_cost` units.
    pub rate: u32,
    /// Candidate ranking distortion.
    pub dist: i64,
    /// Final candidate RD cost.
    pub rd_cost: i64,
    /// Raw luma coefficient-model rate.
    pub luma_rate: u64,
    /// Raw luma coefficient-model distortion.
    pub luma_dist: i64,
    /// UV model rate.
    pub uv_rate: u32,
    /// UV model distortion.
    pub uv_dist: i64,
    /// `SKIP_TXFM_*` state (`0=None`, `1=AcOnly`, `2=AcDc`).
    pub skip_txfm: u8,
    /// Why LAST/NEWMV was evaluated or gated. Unknown for other slots and
    /// legacy/pinned-libvpx records.
    pub newmv_gate: NewMvGateReason,
    /// Fresh LAST/NEWMV search stages. `None` when search was gated, reused,
    /// or produced by a legacy sidecar.
    pub newmv_search: Option<NewMvSearchTrace>,
}

/// One picker-decision record. HLD §3.4 schema.
///
/// The original F.0.2 fields and the F.1 coefficient estimate are live.
/// Later parity investigations have extended the schema with effective
/// qindex, transform size, and emitted-luma-EOB state.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockDecision {
    /// 0-based IVF presentation-order frame index.
    pub frame_idx: u32,
    /// Spatial-layer identifier. Ordinary single-stream records use zero.
    pub spatial_id: u8,
    /// 8×8-cell row of the leaf's top-left mi cell (`0..frame_h/8`).
    pub mi_row: u16,
    /// 8×8-cell col of the leaf's top-left mi cell (`0..frame_w/8`).
    pub mi_col: u16,
    /// Coded block size (`Block4x4 .. Block64x64`).
    pub bsize: BlockSize,
    /// Final mode the picker chose for this leaf.
    pub mode: BlockDecisionMode,
    /// Reference frame (NONE/INTRA for intra leaves; LAST/GOLDEN/ALTREF
    /// for inter).
    pub ref_frame: RefFrame,
    /// MV row in 1/8-pel units. `0` for intra and ZEROMV.
    pub mv_row: i16,
    /// MV col in 1/8-pel units. `0` for intra and ZEROMV.
    pub mv_col: i16,
    /// Picker's mode-rate estimate (`vp9_prob_cost` units).
    pub mode_bits_est: u32,
    /// Picker's MV-rate estimate (`vp9_prob_cost` units). `0` for
    /// non-NEW modes.
    pub mv_bits_est: u32,
    /// Selected reference-frame syntax cost (`vp9_prob_cost` units).
    pub ref_bits_est: u32,
    /// Switchable-filter syntax cost (`vp9_prob_cost` units). Zero for
    /// integer-pel MVs, which do not signal a filter.
    pub filter_bits_est: u32,
    /// Skip-flag syntax cost (`vp9_prob_cost` units).
    pub skip_bits_est: u32,
    /// Picker's coefficient-bit estimate in plain bits.
    pub coef_bits_est: u32,
    /// Winning candidate's total picker rate in `vp9_prob_cost` units.
    pub picker_rate: u32,
    /// Winning candidate's surfaced picker distortion. On rvp9 this is
    /// decoded-pixel SSE; pinned libvpx surfaces `best_rdc.dist`, its
    /// coefficient/model-RD ranking distortion. Compare
    /// [`Self::picker_rd_cost`] when the ranking domain matters.
    pub picker_dist: i64,
    /// Winning candidate's final RD cost.
    pub picker_rd_cost: i64,
    /// Luma variance returned by `model_rd_for_sb_y`.
    pub model_var_y: u32,
    /// Picker model-RD transform size before any emit-side redecision.
    pub model_tx_size: TxSize,
    /// Picker model-RD `SKIP_TXFM_*` state (`0=None`, `1=AcOnly`,
    /// `2=AcDc`).
    pub model_skip_txfm: u8,
    /// Raw luma `block_yrd` coefficient-rate proxy in
    /// `vp9_prob_cost` units, before the skip tie-break.
    pub luma_coef_rate: u64,
    /// Raw luma `block_yrd` coefficient-domain distortion, before the
    /// skip tie-break.
    pub luma_coef_dist: i64,
    /// UV variance-model rate added to the winning candidate.
    pub uv_model_rate: u32,
    /// UV variance-model distortion added to the winning candidate.
    pub uv_model_dist: i64,
    /// Whether the U plane participated in the UV model.
    pub uv_u_sensitive: bool,
    /// Whether the V plane participated in the UV model.
    pub uv_v_sensitive: bool,
    /// SB partition predictor's U-plane color-sensitivity result.
    pub partition_u_sensitive: bool,
    /// SB partition predictor's V-plane color-sensitivity result.
    pub partition_v_sensitive: bool,
    /// LAST-reference candidates in `[ZERO, NEAREST, NEAR, NEW]` order.
    /// Unavailable entries have `available == false`.
    pub last_candidates: [CandidateProbe; 4],
    /// `SSE(source − prediction)` over the Y plane, pre-quant.
    pub sse_after_pred: u64,
    /// Winning inter-prediction filter. `None` for intra leaves.
    pub interp_filter: Option<InterpFilter>,
    /// Picker's final skip flag — the value that lands in the
    /// bitstream's leaf `skip` syntax.
    pub skip_flag: bool,
    /// Effective per-leaf qindex after segmentation.
    pub q_index: u8,
    /// Transform size selected for the luma residual.
    pub tx_size: TxSize,
    /// Whether the emitted luma residual has any non-zero coefficient.
    /// Intra-only paths that record before residual emission report `false`.
    pub luma_eob_nonzero: bool,
}

/// One exact low-resolution ML partition-controller node.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MlPartitionTrace {
    /// Presentation-order frame index.
    pub frame_idx: u32,
    /// Spatial-layer identifier.
    pub spatial_id: u8,
    /// Absolute 8x8-cell row of this model node.
    pub mi_row: u16,
    /// Absolute 8x8-cell column of this model node.
    pub mi_col: u16,
    /// Square node size evaluated by the model.
    pub bsize: BlockSize,
    /// Frame DC quantizer value.
    pub dc_q: i32,
    /// FNV-1a hash of this node's source pixels.
    pub source_hash: u64,
    /// FNV-1a hash of this node's estimated-prediction pixels.
    pub predictor_hash: u64,
    /// Whole-node source-minus-prediction variance.
    pub variance: u32,
    /// Quadrant variances in raster order.
    pub sub_variances: [u32; 4],
    /// Exact IEEE-754 bits of the six model inputs.
    pub feature_bits: [u32; 6],
    /// Exact IEEE-754 bits of the model output.
    pub score_bits: u32,
    /// Whether the CPU8 zero-threshold controller chooses SPLIT.
    pub split: bool,
}

impl MlPartitionTrace {
    /// Write a stable tab-separated record shared with the pinned-libvpx
    /// instrumentation. Integer and float-bit fields make `diff` exact.
    pub fn write_tsv<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        write!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.frame_idx,
            self.spatial_id,
            self.mi_row,
            self.mi_col,
            bsize_tag(self.bsize),
            self.dc_q,
            self.source_hash,
            self.predictor_hash,
            self.variance,
            self.sub_variances[0],
            self.sub_variances[1],
            self.sub_variances[2],
            self.sub_variances[3],
        )?;
        for bits in self.feature_bits {
            write!(writer, "\t{bits}")?;
        }
        write!(writer, "\t{}", self.score_bits)?;
        writeln!(writer, "\t{}", u8::from(self.split))
    }
}

impl BlockDecision {
    /// Serialise the record as one JSONL line (terminated by `\n`).
    ///
    /// Hand-rolled to avoid a workspace-wide `serde` dependency for a
    /// single diagnostic harness. The shape is a flat JSON object —
    /// no nested envelopes — so a 1-pass tokeniser handles it.
    /// `mode`, `ref_frame`, and `bsize` are encoded as their stable
    /// uppercase tags (mirroring the libvpx-side patch's printf).
    ///
    /// # Errors
    ///
    /// Returns `io::Error` if the underlying writer fails.
    pub fn write_jsonl<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        // One owned buffer to keep the write atomic (single syscall on
        // line-buffered stdio — matches the libvpx-side `_IOLBF`
        // guarantee in HLD §3.3 revision 6).
        let mut s = String::with_capacity(256);
        writeln!(
            s,
            "{{\"frame_idx\":{},\"spatial_id\":{},\"mi_row\":{},\"mi_col\":{},\
             \"bsize\":\"{}\",\"mode\":\"{}\",\"ref_frame\":\"{}\",\
             \"mv_row\":{},\"mv_col\":{},\
             \"mode_bits_est\":{},\"mv_bits_est\":{},\"ref_bits_est\":{},\
             \"filter_bits_est\":{},\"skip_bits_est\":{},\"coef_bits_est\":{},\
             \"picker_rate\":{},\"picker_dist\":{},\"picker_rd_cost\":{},\
             \"model_var_y\":{},\"model_tx_size\":\"{}\",\
             \"model_skip_txfm\":{},\
             \"luma_coef_rate\":{},\"luma_coef_dist\":{},\
             \"uv_model_rate\":{},\"uv_model_dist\":{},\
             \"uv_u_sensitive\":{},\"uv_v_sensitive\":{},\
             \"partition_u_sensitive\":{},\"partition_v_sensitive\":{},\
             \"last_candidates\":\"{}\",\
             \"sse_after_pred\":{},\"interp_filter\":\"{}\",\"skip_flag\":{},\
             \"q_index\":{},\"tx_size\":\"{}\",\"luma_eob_nonzero\":{}}}",
            self.frame_idx,
            self.spatial_id,
            self.mi_row,
            self.mi_col,
            bsize_tag(self.bsize),
            self.mode.as_tag(),
            ref_frame_tag(self.ref_frame),
            self.mv_row,
            self.mv_col,
            self.mode_bits_est,
            self.mv_bits_est,
            self.ref_bits_est,
            self.filter_bits_est,
            self.skip_bits_est,
            self.coef_bits_est,
            self.picker_rate,
            self.picker_dist,
            self.picker_rd_cost,
            self.model_var_y,
            tx_size_tag(self.model_tx_size),
            self.model_skip_txfm,
            self.luma_coef_rate,
            self.luma_coef_dist,
            self.uv_model_rate,
            self.uv_model_dist,
            self.uv_u_sensitive,
            self.uv_v_sensitive,
            self.partition_u_sensitive,
            self.partition_v_sensitive,
            candidate_probes_tag(&self.last_candidates),
            self.sse_after_pred,
            interp_filter_tag(self.interp_filter),
            self.skip_flag,
            self.q_index,
            tx_size_tag(self.tx_size),
            self.luma_eob_nonzero,
        )
        .expect("String fmt cannot fail");
        w.write_all(s.as_bytes())
    }
}

fn candidate_probes_tag(probes: &[CandidateProbe; 4]) -> String {
    probes
        .iter()
        .map(|p| {
            let trace = p.newmv_search.unwrap_or(NewMvSearchTrace {
                start_mv: rvp9_core::mode_info::Mv {
                    row: i16::MIN,
                    col: i16::MIN,
                },
                fullpel_mv: rvp9_core::mode_info::Mv {
                    row: i16::MIN,
                    col: i16::MIN,
                },
                final_mv: rvp9_core::mode_info::Mv {
                    row: i16::MIN,
                    col: i16::MIN,
                },
                ref_mv: rvp9_core::mode_info::Mv {
                    row: i16::MIN,
                    col: i16::MIN,
                },
                near_mv: rvp9_core::mode_info::Mv {
                    row: i16::MIN,
                    col: i16::MIN,
                },
                sb_pred_mv: None,
                mv_pred_mv: rvp9_core::mode_info::Mv {
                    row: i16::MIN,
                    col: i16::MIN,
                },
                error_per_bit: i32::MIN,
                source_hash: 0,
                reference_hash: 0,
                subpel: None,
            });
            let (
                subpel_col_min,
                subpel_col_max,
                subpel_row_min,
                subpel_row_max,
                subpel_forced_stop,
                subpel_iters_per_step,
                subpel_allow_hp,
                subpel_center_distortion,
                subpel_center_mv_cost,
                half_coordinate_hash,
                half_distortion_hash,
                half_mv_cost_hash,
                half_winner_row,
                half_winner_col,
                quarter_coordinate_hash,
                quarter_distortion_hash,
                quarter_mv_cost_hash,
                quarter_winner_row,
                quarter_winner_col,
            ) = trace.subpel.map_or(
                (
                    i32::MIN,
                    i32::MIN,
                    i32::MIN,
                    i32::MIN,
                    i32::MIN,
                    i32::MIN,
                    -1,
                    u32::MAX,
                    u32::MAX,
                    0,
                    0,
                    0,
                    i16::MIN,
                    i16::MIN,
                    0,
                    0,
                    0,
                    i16::MIN,
                    i16::MIN,
                ),
                |subpel| {
                    (
                        subpel.limits.col_min,
                        subpel.limits.col_max,
                        subpel.limits.row_min,
                        subpel.limits.row_max,
                        subpel.forced_stop,
                        subpel.iters_per_step,
                        i32::from(subpel.allow_hp),
                        subpel.center_distortion,
                        subpel.center_mv_cost,
                        subpel.half.coordinate_hash,
                        subpel.half.distortion_hash,
                        subpel.half.mv_cost_hash,
                        subpel.half.winner.row,
                        subpel.half.winner.col,
                        subpel.quarter.coordinate_hash,
                        subpel.quarter.distortion_hash,
                        subpel.quarter.mv_cost_hash,
                        subpel.quarter.winner.row,
                        subpel.quarter.winner.col,
                    )
                },
            );
            let sb_pred_mv = trace.sb_pred_mv.unwrap_or(rvp9_core::mode_info::Mv {
                row: i16::MIN,
                col: i16::MIN,
            });
            format!(
                "{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
                u8::from(p.available),
                p.mv_row,
                p.mv_col,
                p.rate,
                p.dist,
                p.rd_cost,
                p.luma_rate,
                p.luma_dist,
                p.uv_rate,
                p.uv_dist,
                p.skip_txfm,
                p.newmv_gate as u8,
                trace.start_mv.row,
                trace.start_mv.col,
                trace.fullpel_mv.row,
                trace.fullpel_mv.col,
                trace.final_mv.row,
                trace.final_mv.col,
                trace.ref_mv.row,
                trace.ref_mv.col,
                trace.error_per_bit,
                trace.source_hash,
                trace.reference_hash,
                subpel_col_min,
                subpel_col_max,
                subpel_row_min,
                subpel_row_max,
                subpel_forced_stop,
                subpel_iters_per_step,
                subpel_allow_hp,
                subpel_center_distortion,
                subpel_center_mv_cost,
                half_coordinate_hash,
                half_distortion_hash,
                half_mv_cost_hash,
                half_winner_row,
                half_winner_col,
                quarter_coordinate_hash,
                quarter_distortion_hash,
                quarter_mv_cost_hash,
                quarter_winner_row,
                quarter_winner_col,
                trace.near_mv.row,
                trace.near_mv.col,
                sb_pred_mv.row,
                sb_pred_mv.col,
                trace.mv_pred_mv.row,
                trace.mv_pred_mv.col,
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

/// Stable transform-size tag shared with the libvpx patch.
#[inline]
#[must_use]
pub const fn tx_size_tag(tx_size: TxSize) -> &'static str {
    match tx_size {
        TxSize::Tx4x4 => "4x4",
        TxSize::Tx8x8 => "8x8",
        TxSize::Tx16x16 => "16x16",
        TxSize::Tx32x32 => "32x32",
    }
}

/// Stable interpolation-filter tag shared with the libvpx patch.
#[inline]
#[must_use]
pub fn interp_filter_tag(filter: Option<InterpFilter>) -> &'static str {
    match filter {
        None => "NONE",
        Some(InterpFilter::EightTap) => "EIGHTTAP",
        Some(InterpFilter::EightTapSmooth) => "EIGHTTAP_SMOOTH",
        Some(InterpFilter::EightTapSharp) => "EIGHTTAP_SHARP",
        Some(InterpFilter::Bilinear) => "BILINEAR",
        Some(InterpFilter::Switchable) => "SWITCHABLE",
    }
}

/// Stable uppercase tag for a [`BlockSize`]. Mirrors libvpx's
/// `BLOCK_*` enum names so the libvpx-side patch can emit the same
/// strings.
#[inline]
#[must_use]
pub fn bsize_tag(b: BlockSize) -> &'static str {
    match b {
        BlockSize::Block4x4 => "4x4",
        BlockSize::Block4x8 => "4x8",
        BlockSize::Block8x4 => "8x4",
        BlockSize::Block8x8 => "8x8",
        BlockSize::Block8x16 => "8x16",
        BlockSize::Block16x8 => "16x8",
        BlockSize::Block16x16 => "16x16",
        BlockSize::Block16x32 => "16x32",
        BlockSize::Block32x16 => "32x16",
        BlockSize::Block32x32 => "32x32",
        BlockSize::Block32x64 => "32x64",
        BlockSize::Block64x32 => "64x32",
        BlockSize::Block64x64 => "64x64",
    }
}

/// Stable uppercase tag for a [`RefFrame`]. Mirrors libvpx
/// `MV_REFERENCE_FRAME` names.
#[inline]
#[must_use]
pub fn ref_frame_tag(r: RefFrame) -> &'static str {
    match r {
        RefFrame::NoRefFrame => "NONE",
        RefFrame::IntraFrame => "INTRA",
        RefFrame::LastFrame => "LAST",
        RefFrame::GoldenFrame => "GOLDEN",
        RefFrame::AltrefFrame => "ALTREF",
    }
}

thread_local! {
    /// Per-thread accumulator. Empty when the oracle is disabled or
    /// after [`drain_frame`].
    static SINK: RefCell<Vec<BlockDecision>> = const { RefCell::new(Vec::new()) };
}

thread_local! {
    /// Exact low-resolution partition-model records for the current thread.
    static PARTITION_SINK: RefCell<Vec<MlPartitionTrace>> = const { RefCell::new(Vec::new()) };
}

thread_local! {
    /// Per-thread "current frame index" the picker hooks attach to
    /// each recorded decision. Set by [`set_frame_idx`] at the top of
    /// each frame's encode loop.
    static FRAME_IDX: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
}

thread_local! {
    /// Spatial component currently being encoded. Ordinary streams and code
    /// outside a scoped spatial component use layer zero.
    static SPATIAL_ID: core::cell::Cell<u8> = const { core::cell::Cell::new(0) };
}

/// Run `f` with the diagnostic spatial-layer tag set to `spatial_id`.
/// Restores the prior tag even if `f` unwinds.
pub(crate) fn with_spatial_id<T>(spatial_id: u8, f: impl FnOnce() -> T) -> T {
    struct Restore(u8);
    impl Drop for Restore {
        fn drop(&mut self) {
            SPATIAL_ID.with(|current| current.set(self.0));
        }
    }

    let prior = SPATIAL_ID.with(|current| current.replace(spatial_id));
    let _restore = Restore(prior);
    f()
}

/// Read the current diagnostic spatial-layer identifier.
#[inline]
#[must_use]
pub(crate) fn current_spatial_id() -> u8 {
    SPATIAL_ID.with(core::cell::Cell::get)
}

/// Enable the oracle if `RVP9_F02_ORACLE` is set and non-zero.
///
/// One relaxed-store; subsequent picker-hook calls observe the new
/// value via a relaxed load. Safe to call once at process startup.
pub fn init_from_env() {
    let on = std::env::var_os("RVP9_F02_ORACLE").is_some_and(|v| v != "0");
    F02_ON.store(on, Ordering::Relaxed);
}

/// Whether the oracle is active. Inlined relaxed load.
#[inline]
#[must_use]
pub fn is_enabled() -> bool {
    F02_ON.load(Ordering::Relaxed)
}

/// Set the current frame index. Harness drivers call this at the top
/// of each frame's encode so the per-block records carry the right
/// `frame_idx`. No-op when the oracle is disabled.
#[inline]
pub fn set_frame_idx(idx: u32) {
    if !is_enabled() {
        return;
    }
    FRAME_IDX.with(|c| c.set(idx));
}

/// Record one picker decision. No-op when disabled (single relaxed
/// atomic load).
#[inline]
pub fn record_decision(d: BlockDecision) {
    if !is_enabled() {
        return;
    }
    SINK.with(|s| s.borrow_mut().push(d));
}

/// Record one low-resolution partition-model node. No-op when disabled.
#[inline]
pub fn record_partition_trace(trace: MlPartitionTrace) {
    if !is_enabled() {
        return;
    }
    PARTITION_SINK.with(|sink| sink.borrow_mut().push(trace));
}

/// Drain the per-thread per-frame sink. Returns an empty `Vec` when
/// the oracle is disabled.
#[must_use]
pub fn drain_frame() -> Vec<BlockDecision> {
    if !is_enabled() {
        return Vec::new();
    }
    SINK.with(|s| std::mem::take(&mut *s.borrow_mut()))
}

/// Drain exact low-resolution partition-model records from the current thread.
#[must_use]
pub fn drain_partition_traces() -> Vec<MlPartitionTrace> {
    if !is_enabled() {
        return Vec::new();
    }
    PARTITION_SINK.with(|sink| std::mem::take(&mut *sink.borrow_mut()))
}

/// Read the per-thread current frame index. Used by the picker hooks
/// that record decisions.
#[inline]
#[must_use]
pub fn current_frame_idx() -> u32 {
    FRAME_IDX.with(core::cell::Cell::get)
}

// ---------------------------------------------------------------------------
// Unit tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_inter() -> BlockDecision {
        BlockDecision {
            frame_idx: 7,
            spatial_id: 2,
            mi_row: 4,
            mi_col: 12,
            bsize: BlockSize::Block16x16,
            mode: BlockDecisionMode::Inter(InterMode::NewMv),
            ref_frame: RefFrame::LastFrame,
            mv_row: 24,
            mv_col: -8,
            mode_bits_est: 150,
            mv_bits_est: 320,
            ref_bits_est: 461,
            filter_bits_est: 200,
            skip_bits_est: 512,
            coef_bits_est: 0,
            picker_rate: 7_500,
            picker_dist: 12_345,
            picker_rd_cost: 98_765,
            model_var_y: 4_321,
            model_tx_size: TxSize::Tx8x8,
            model_skip_txfm: 0,
            luma_coef_rate: 6_400,
            luma_coef_dist: 9_876,
            uv_model_rate: 700,
            uv_model_dist: 1_234,
            uv_u_sensitive: true,
            uv_v_sensitive: false,
            partition_u_sensitive: false,
            partition_v_sensitive: true,
            last_candidates: [CandidateProbe::default(); 4],
            sse_after_pred: 1_234_567,
            interp_filter: Some(InterpFilter::EightTapSmooth),
            skip_flag: false,
            q_index: 143,
            tx_size: TxSize::Tx16x16,
            luma_eob_nonzero: true,
        }
    }

    fn sample_intra() -> BlockDecision {
        BlockDecision {
            frame_idx: 0,
            spatial_id: 0,
            mi_row: 0,
            mi_col: 0,
            bsize: BlockSize::Block8x8,
            mode: BlockDecisionMode::Intra(IntraMode::Dc),
            ref_frame: RefFrame::IntraFrame,
            mv_row: 0,
            mv_col: 0,
            mode_bits_est: 90,
            mv_bits_est: 0,
            ref_bits_est: 2_473,
            filter_bits_est: 0,
            skip_bits_est: 0,
            coef_bits_est: 0,
            picker_rate: 900,
            picker_dist: 1_234,
            picker_rd_cost: 5_678,
            model_var_y: 0,
            model_tx_size: TxSize::Tx8x8,
            model_skip_txfm: 2,
            luma_coef_rate: 0,
            luma_coef_dist: 0,
            uv_model_rate: 0,
            uv_model_dist: 0,
            uv_u_sensitive: false,
            uv_v_sensitive: false,
            partition_u_sensitive: false,
            partition_v_sensitive: false,
            last_candidates: [CandidateProbe::default(); 4],
            sse_after_pred: 0,
            interp_filter: None,
            skip_flag: true,
            q_index: 65,
            tx_size: TxSize::Tx8x8,
            luma_eob_nonzero: false,
        }
    }

    #[test]
    fn spatial_scope_tags_records_and_restores_the_default_layer() {
        let _guard = crate::env_test_mutex();
        F02_ON.store(true, Ordering::Relaxed);
        let _ = drain_frame();
        assert_eq!(current_spatial_id(), 0);
        with_spatial_id(2, || {
            let mut decision = sample_inter();
            decision.spatial_id = current_spatial_id();
            record_decision(decision);
        });
        let records = drain_frame();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].spatial_id, 2);
        assert_eq!(current_spatial_id(), 0);
        F02_ON.store(false, Ordering::Relaxed);
    }

    #[test]
    fn jsonl_round_trip_shape() {
        // The schema's serialization must be parseable by the same
        // shape the libvpx-side patch emits.
        let d = sample_inter();
        let mut buf = Vec::new();
        d.write_jsonl(&mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.ends_with('\n'), "JSONL line must end with newline");
        assert!(s.contains("\"frame_idx\":7"), "frame_idx field present");
        assert!(s.contains("\"spatial_id\":2"), "spatial_id field present");
        assert!(s.contains("\"mi_row\":4"), "mi_row field present");
        assert!(s.contains("\"mi_col\":12"), "mi_col field present");
        assert!(s.contains("\"bsize\":\"16x16\""), "bsize tag");
        assert!(s.contains("\"mode\":\"NEWMV\""), "inter mode tag");
        assert!(s.contains("\"ref_frame\":\"LAST\""), "ref_frame tag");
        assert!(s.contains("\"mv_row\":24"), "mv_row field");
        assert!(s.contains("\"mv_col\":-8"), "mv_col field (signed)");
        assert!(
            s.contains("\"interp_filter\":\"EIGHTTAP_SMOOTH\""),
            "interpolation-filter tag"
        );
        assert!(s.contains("\"skip_flag\":false"), "skip_flag bool");
        assert!(s.contains("\"q_index\":143"), "qindex field");
        assert!(s.contains("\"tx_size\":\"16x16\""), "transform-size tag");
        assert!(
            s.contains("\"luma_eob_nonzero\":true"),
            "luma EOB indicator"
        );
        assert!(
            s.contains("\"coef_bits_est\":0"),
            "coefficient-bit estimate"
        );
        // Exactly one '\n' (one line).
        assert_eq!(s.matches('\n').count(), 1);
    }

    #[test]
    fn jsonl_intra_mode_tag() {
        let d = sample_intra();
        let mut buf = Vec::new();
        d.write_jsonl(&mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("\"mode\":\"DC\""), "intra DC tag");
        assert!(s.contains("\"ref_frame\":\"INTRA\""), "ref_frame INTRA tag");
        assert!(s.contains("\"skip_flag\":true"), "skip_flag true");
    }

    #[test]
    fn ml_partition_tsv_preserves_float_bits_and_field_order() {
        let trace = MlPartitionTrace {
            frame_idx: 1,
            spatial_id: 0,
            mi_row: 2,
            mi_col: 3,
            bsize: BlockSize::Block32x32,
            dc_q: 80,
            source_hash: 11,
            predictor_hash: 12,
            variance: 13,
            sub_variances: [14, 15, 16, 17],
            feature_bits: [18, 19, 20, 21, 22, 23],
            score_bits: 24,
            split: true,
        };
        let mut output = Vec::new();
        trace.write_tsv(&mut output).unwrap();
        let line = String::from_utf8(output).unwrap();
        let fields = line.trim_end().split('\t').collect::<Vec<_>>();
        assert_eq!(fields.len(), 21);
        assert_eq!(fields[4], "32x32");
        assert_eq!(&fields[13..19], ["18", "19", "20", "21", "22", "23"]);
        assert_eq!(fields[19], "24");
        assert_eq!(fields[20], "1");
    }

    #[test]
    fn sink_off_by_default() {
        let _guard = crate::env_test_mutex();
        // Without any env init the static atomic is `false`. Whatever
        // any other test may have set, force off for this assertion.
        // The crate-wide guard serializes the shared enable atomic with
        // encoder tests that call `init_from_env`.
        F02_ON.store(false, Ordering::Relaxed);
        // Drain first in case a previous test populated this thread's
        // sink.
        let _ = drain_frame();
        record_decision(sample_inter());
        let v = drain_frame();
        assert!(v.is_empty(), "disabled sink must drop record_decision");
    }

    #[test]
    fn sink_on_records_and_drains() {
        let _guard = crate::env_test_mutex();
        F02_ON.store(true, Ordering::Relaxed);
        // Reset thread-local state for determinism.
        let _ = drain_frame();
        set_frame_idx(42);
        assert_eq!(current_frame_idx(), 42);
        record_decision(sample_inter());
        record_decision(sample_intra());
        let v = drain_frame();
        assert_eq!(v.len(), 2);
        // After drain, the sink is empty again.
        let v2 = drain_frame();
        assert!(v2.is_empty());
        // Reset for other tests.
        F02_ON.store(false, Ordering::Relaxed);
    }

    #[test]
    fn init_from_env_unset_yields_disabled() {
        let _guard = crate::env_test_mutex();
        // SAFETY: the crate-wide guard serializes this process-environment
        // mutation with every encoder test that touches F02 state.
        unsafe { std::env::remove_var("RVP9_F02_ORACLE") };
        init_from_env();
        assert!(!is_enabled());
    }

    #[test]
    fn bsize_and_ref_tags_cover_every_variant() {
        // All 13 BlockSize variants emit a tag.
        for bs in [
            BlockSize::Block4x4,
            BlockSize::Block4x8,
            BlockSize::Block8x4,
            BlockSize::Block8x8,
            BlockSize::Block8x16,
            BlockSize::Block16x8,
            BlockSize::Block16x16,
            BlockSize::Block16x32,
            BlockSize::Block32x16,
            BlockSize::Block32x32,
            BlockSize::Block32x64,
            BlockSize::Block64x32,
            BlockSize::Block64x64,
        ] {
            let t = bsize_tag(bs);
            assert!(!t.is_empty());
            assert!(t.contains('x'));
        }
        for r in [
            RefFrame::NoRefFrame,
            RefFrame::IntraFrame,
            RefFrame::LastFrame,
            RefFrame::GoldenFrame,
            RefFrame::AltrefFrame,
        ] {
            let t = ref_frame_tag(r);
            assert!(!t.is_empty());
        }
    }
}
