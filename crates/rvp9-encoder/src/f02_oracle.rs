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

/// One picker-decision record. HLD §3.4 schema.
///
/// The original F.0.2 fields and the F.1 coefficient estimate are live.
/// Later parity investigations have extended the schema with effective
/// qindex, transform size, and emitted-luma-EOB state.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockDecision {
    /// 0-based IVF presentation-order frame index.
    pub frame_idx: u32,
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
            "{{\"frame_idx\":{},\"mi_row\":{},\"mi_col\":{},\
             \"bsize\":\"{}\",\"mode\":\"{}\",\"ref_frame\":\"{}\",\
             \"mv_row\":{},\"mv_col\":{},\
             \"mode_bits_est\":{},\"mv_bits_est\":{},\"coef_bits_est\":{},\
             \"picker_rate\":{},\"picker_dist\":{},\"picker_rd_cost\":{},\
             \"model_var_y\":{},\"model_tx_size\":\"{}\",\
             \"model_skip_txfm\":{},\
             \"luma_coef_rate\":{},\"luma_coef_dist\":{},\
             \"uv_model_rate\":{},\"uv_model_dist\":{},\
             \"uv_u_sensitive\":{},\"uv_v_sensitive\":{},\
             \"sse_after_pred\":{},\"interp_filter\":\"{}\",\"skip_flag\":{},\
             \"q_index\":{},\"tx_size\":\"{}\",\"luma_eob_nonzero\":{}}}",
            self.frame_idx,
            self.mi_row,
            self.mi_col,
            bsize_tag(self.bsize),
            self.mode.as_tag(),
            ref_frame_tag(self.ref_frame),
            self.mv_row,
            self.mv_col,
            self.mode_bits_est,
            self.mv_bits_est,
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
    /// Per-thread "current frame index" the picker hooks attach to
    /// each recorded decision. Set by [`set_frame_idx`] at the top of
    /// each frame's encode loop.
    static FRAME_IDX: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
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

/// Drain the per-thread per-frame sink. Returns an empty `Vec` when
/// the oracle is disabled.
#[must_use]
pub fn drain_frame() -> Vec<BlockDecision> {
    if !is_enabled() {
        return Vec::new();
    }
    SINK.with(|s| std::mem::take(&mut *s.borrow_mut()))
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
            mi_row: 4,
            mi_col: 12,
            bsize: BlockSize::Block16x16,
            mode: BlockDecisionMode::Inter(InterMode::NewMv),
            ref_frame: RefFrame::LastFrame,
            mv_row: 24,
            mv_col: -8,
            mode_bits_est: 150,
            mv_bits_est: 320,
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
            mi_row: 0,
            mi_col: 0,
            bsize: BlockSize::Block8x8,
            mode: BlockDecisionMode::Intra(IntraMode::Dc),
            ref_frame: RefFrame::IntraFrame,
            mv_row: 0,
            mv_col: 0,
            mode_bits_est: 90,
            mv_bits_est: 0,
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
            sse_after_pred: 0,
            interp_filter: None,
            skip_flag: true,
            q_index: 65,
            tx_size: TxSize::Tx8x8,
            luma_eob_nonzero: false,
        }
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
    fn sink_off_by_default() {
        // Without any env init the static atomic is `false`. Whatever
        // any other test may have set, force off for this assertion.
        // (Thread-locals are per-test-thread so a second thread isn't
        // contaminated by env churn.)
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
        // SAFETY: this test is single-threaded by virtue of `cargo
        // test`'s per-test isolation. We mutate process env then
        // re-read. No other threads in this crate touch this var.
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
