//! F.0.2 — per-block oracle JSONL schema (harness side).
//!
//! Re-exports [`rvp9_encoder::f02_oracle::BlockDecision`] (the
//! canonical schema; see HLD §3.4) and adds the JSONL **parser** that
//! both encoder legs share. The encoder leg's binary writes records
//! via [`BlockDecision::write_jsonl`]; the libvpx leg's patched
//! `vpxenc` writes the same shape; this module's [`parse_jsonl_line`]
//! turns a line back into a `BlockDecision` so the diff binary can
//! line up records pairwise.
//!
//! Hand-rolled to avoid a workspace-wide `serde` dependency — the
//! schema is a flat JSON object with no nesting; a 1-pass
//! tokeniser handles it. The unit tests below guarantee round-trip
//! consistency with the canonical writer.

use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};

pub use rvp9_encoder::f02_oracle::{
    BlockDecision, BlockDecisionMode, CandidateProbe, bsize_tag, interp_filter_tag, ref_frame_tag,
    tx_size_tag,
};

use rvp9_core::bitstream::frame::InterpFilter;
use rvp9_core::block::BlockSize;
use rvp9_core::intra_pred::IntraMode;
use rvp9_core::mode_decision::inter::{NewMvGateReason, NewMvSearchTrace};
use rvp9_core::mode_info::{InterMode, Mv, RefFrame};
use rvp9_core::mv_search::mv_limits::MvLimits;
use rvp9_core::mv_search::sub_pixel::{SubPixelLevelTrace, SubPixelSearchTrace};
use rvp9_core::transforms::TxSize;

/// Error type for the JSONL parser.
#[derive(Debug)]
pub enum ParseError {
    /// Required field missing from the JSON object.
    MissingField(&'static str),
    /// Field value didn't parse to the expected type.
    BadValue { field: &'static str, raw: String },
    /// JSON syntax violation (e.g. missing brace, unbalanced quotes).
    Malformed(&'static str),
    /// Underlying I/O failure.
    Io(io::Error),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField(n) => write!(f, "F.0.2 schema: missing field {n:?}"),
            Self::BadValue { field, raw } => {
                write!(f, "F.0.2 schema: bad value for {field:?}: {raw:?}")
            }
            Self::Malformed(why) => write!(f, "F.0.2 schema: malformed JSONL: {why}"),
            Self::Io(e) => write!(f, "F.0.2 schema: I/O error: {e}"),
        }
    }
}

impl std::error::Error for ParseError {}

impl From<io::Error> for ParseError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Parse a single JSONL line into a [`BlockDecision`]. Trailing
/// newline is optional. Whitespace between tokens is tolerated.
///
/// # Errors
///
/// Returns [`ParseError::MissingField`] / [`ParseError::BadValue`] /
/// [`ParseError::Malformed`] on any deviation from the canonical
/// schema (see HLD §3.4 / R5).
pub fn parse_jsonl_line(line: &str) -> Result<BlockDecision, ParseError> {
    let line = line.trim();
    let inner = line
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or(ParseError::Malformed("expected outer braces"))?;

    let mut frame_idx: Option<u32> = None;
    let mut spatial_id: Option<u8> = None;
    let mut mi_row: Option<u16> = None;
    let mut mi_col: Option<u16> = None;
    let mut bsize: Option<BlockSize> = None;
    let mut mode_tag: Option<String> = None;
    let mut ref_tag: Option<String> = None;
    let mut mv_row: Option<i16> = None;
    let mut mv_col: Option<i16> = None;
    let mut mode_bits_est: Option<u32> = None;
    let mut mv_bits_est: Option<u32> = None;
    let mut ref_bits_est: Option<u32> = None;
    let mut filter_bits_est: Option<u32> = None;
    let mut skip_bits_est: Option<u32> = None;
    let mut coef_bits_est: Option<u32> = None;
    let mut picker_rate: Option<u32> = None;
    let mut picker_dist: Option<i64> = None;
    let mut picker_rd_cost: Option<i64> = None;
    let mut model_var_y: Option<u32> = None;
    let mut model_tx_size: Option<TxSize> = None;
    let mut model_skip_txfm: Option<u8> = None;
    let mut luma_coef_rate: Option<u64> = None;
    let mut luma_coef_dist: Option<i64> = None;
    let mut uv_model_rate: Option<u32> = None;
    let mut uv_model_dist: Option<i64> = None;
    let mut uv_u_sensitive: Option<bool> = None;
    let mut uv_v_sensitive: Option<bool> = None;
    let mut partition_u_sensitive: Option<bool> = None;
    let mut partition_v_sensitive: Option<bool> = None;
    let mut last_candidates: Option<[CandidateProbe; 4]> = None;
    let mut sse_after_pred: Option<u64> = None;
    let mut interp_filter: Option<Option<InterpFilter>> = None;
    let mut skip_flag: Option<bool> = None;
    let mut q_index: Option<u8> = None;
    let mut tx_size: Option<TxSize> = None;
    let mut luma_eob_nonzero: Option<bool> = None;

    for kv in split_top_level_commas(inner)? {
        let (k, v) = split_kv(kv)?;
        match k {
            "frame_idx" => frame_idx = Some(parse_num(v, "frame_idx")?),
            "spatial_id" => spatial_id = Some(parse_num(v, "spatial_id")?),
            "mi_row" => mi_row = Some(parse_num(v, "mi_row")?),
            "mi_col" => mi_col = Some(parse_num(v, "mi_col")?),
            "bsize" => {
                let raw = parse_string(v, "bsize")?;
                bsize = Some(bsize_from_tag(&raw).ok_or(ParseError::BadValue {
                    field: "bsize",
                    raw,
                })?);
            }
            "mode" => mode_tag = Some(parse_string(v, "mode")?),
            "ref_frame" => ref_tag = Some(parse_string(v, "ref_frame")?),
            "mv_row" => mv_row = Some(parse_num(v, "mv_row")?),
            "mv_col" => mv_col = Some(parse_num(v, "mv_col")?),
            "mode_bits_est" => mode_bits_est = Some(parse_num(v, "mode_bits_est")?),
            "mv_bits_est" => mv_bits_est = Some(parse_num(v, "mv_bits_est")?),
            "ref_bits_est" => ref_bits_est = Some(parse_num(v, "ref_bits_est")?),
            "filter_bits_est" => filter_bits_est = Some(parse_num(v, "filter_bits_est")?),
            "skip_bits_est" => skip_bits_est = Some(parse_num(v, "skip_bits_est")?),
            "coef_bits_est" => coef_bits_est = Some(parse_num(v, "coef_bits_est")?),
            "picker_rate" => picker_rate = Some(parse_num(v, "picker_rate")?),
            "picker_dist" => picker_dist = Some(parse_num(v, "picker_dist")?),
            "picker_rd_cost" => picker_rd_cost = Some(parse_num(v, "picker_rd_cost")?),
            "model_var_y" => model_var_y = Some(parse_num(v, "model_var_y")?),
            "model_tx_size" => {
                let raw = parse_string(v, "model_tx_size")?;
                model_tx_size = Some(tx_size_from_tag(&raw).ok_or(ParseError::BadValue {
                    field: "model_tx_size",
                    raw,
                })?);
            }
            "model_skip_txfm" => model_skip_txfm = Some(parse_num(v, "model_skip_txfm")?),
            "luma_coef_rate" => luma_coef_rate = Some(parse_num(v, "luma_coef_rate")?),
            "luma_coef_dist" => luma_coef_dist = Some(parse_num(v, "luma_coef_dist")?),
            "uv_model_rate" => uv_model_rate = Some(parse_num(v, "uv_model_rate")?),
            "uv_model_dist" => uv_model_dist = Some(parse_num(v, "uv_model_dist")?),
            "uv_u_sensitive" => uv_u_sensitive = Some(parse_bool(v, "uv_u_sensitive")?),
            "uv_v_sensitive" => uv_v_sensitive = Some(parse_bool(v, "uv_v_sensitive")?),
            "partition_u_sensitive" => {
                partition_u_sensitive = Some(parse_bool(v, "partition_u_sensitive")?);
            }
            "partition_v_sensitive" => {
                partition_v_sensitive = Some(parse_bool(v, "partition_v_sensitive")?);
            }
            "last_candidates" => last_candidates = Some(parse_candidate_probes(v)?),
            "sse_after_pred" => sse_after_pred = Some(parse_num(v, "sse_after_pred")?),
            "interp_filter" => {
                let raw = parse_string(v, "interp_filter")?;
                interp_filter = Some(interp_filter_from_tag(&raw).ok_or(ParseError::BadValue {
                    field: "interp_filter",
                    raw,
                })?);
            }
            "skip_flag" => skip_flag = Some(parse_bool(v, "skip_flag")?),
            "q_index" => q_index = Some(parse_num(v, "q_index")?),
            "tx_size" => {
                let raw = parse_string(v, "tx_size")?;
                tx_size = Some(tx_size_from_tag(&raw).ok_or(ParseError::BadValue {
                    field: "tx_size",
                    raw,
                })?);
            }
            "luma_eob_nonzero" => luma_eob_nonzero = Some(parse_bool(v, "luma_eob_nonzero")?),
            _ => {} // forward-compatible: ignore unknown extra fields
        }
    }

    let ref_frame_resolved = ref_frame_from_tag(
        ref_tag
            .as_deref()
            .ok_or(ParseError::MissingField("ref_frame"))?,
    )
    .ok_or_else(|| ParseError::BadValue {
        field: "ref_frame",
        raw: ref_tag.clone().unwrap_or_default(),
    })?;
    let mode_str = mode_tag.ok_or(ParseError::MissingField("mode"))?;
    let mode_resolved =
        mode_from_tag(&mode_str, ref_frame_resolved).ok_or(ParseError::BadValue {
            field: "mode",
            raw: mode_str,
        })?;

    Ok(BlockDecision {
        frame_idx: frame_idx.ok_or(ParseError::MissingField("frame_idx"))?,
        spatial_id: spatial_id.unwrap_or(0),
        mi_row: mi_row.ok_or(ParseError::MissingField("mi_row"))?,
        mi_col: mi_col.ok_or(ParseError::MissingField("mi_col"))?,
        bsize: bsize.ok_or(ParseError::MissingField("bsize"))?,
        mode: mode_resolved,
        ref_frame: ref_frame_resolved,
        mv_row: mv_row.ok_or(ParseError::MissingField("mv_row"))?,
        mv_col: mv_col.ok_or(ParseError::MissingField("mv_col"))?,
        mode_bits_est: mode_bits_est.ok_or(ParseError::MissingField("mode_bits_est"))?,
        mv_bits_est: mv_bits_est.ok_or(ParseError::MissingField("mv_bits_est"))?,
        ref_bits_est: ref_bits_est.ok_or(ParseError::MissingField("ref_bits_est"))?,
        filter_bits_est: filter_bits_est.ok_or(ParseError::MissingField("filter_bits_est"))?,
        skip_bits_est: skip_bits_est.ok_or(ParseError::MissingField("skip_bits_est"))?,
        coef_bits_est: coef_bits_est.ok_or(ParseError::MissingField("coef_bits_est"))?,
        picker_rate: picker_rate.ok_or(ParseError::MissingField("picker_rate"))?,
        picker_dist: picker_dist.ok_or(ParseError::MissingField("picker_dist"))?,
        picker_rd_cost: picker_rd_cost.ok_or(ParseError::MissingField("picker_rd_cost"))?,
        model_var_y: model_var_y.ok_or(ParseError::MissingField("model_var_y"))?,
        model_tx_size: model_tx_size.ok_or(ParseError::MissingField("model_tx_size"))?,
        model_skip_txfm: model_skip_txfm.ok_or(ParseError::MissingField("model_skip_txfm"))?,
        luma_coef_rate: luma_coef_rate.ok_or(ParseError::MissingField("luma_coef_rate"))?,
        luma_coef_dist: luma_coef_dist.ok_or(ParseError::MissingField("luma_coef_dist"))?,
        uv_model_rate: uv_model_rate.ok_or(ParseError::MissingField("uv_model_rate"))?,
        uv_model_dist: uv_model_dist.ok_or(ParseError::MissingField("uv_model_dist"))?,
        uv_u_sensitive: uv_u_sensitive.ok_or(ParseError::MissingField("uv_u_sensitive"))?,
        uv_v_sensitive: uv_v_sensitive.ok_or(ParseError::MissingField("uv_v_sensitive"))?,
        partition_u_sensitive: partition_u_sensitive
            .ok_or(ParseError::MissingField("partition_u_sensitive"))?,
        partition_v_sensitive: partition_v_sensitive
            .ok_or(ParseError::MissingField("partition_v_sensitive"))?,
        last_candidates: last_candidates.ok_or(ParseError::MissingField("last_candidates"))?,
        sse_after_pred: sse_after_pred.ok_or(ParseError::MissingField("sse_after_pred"))?,
        interp_filter: interp_filter.ok_or(ParseError::MissingField("interp_filter"))?,
        skip_flag: skip_flag.ok_or(ParseError::MissingField("skip_flag"))?,
        q_index: q_index.ok_or(ParseError::MissingField("q_index"))?,
        tx_size: tx_size.ok_or(ParseError::MissingField("tx_size"))?,
        luma_eob_nonzero: luma_eob_nonzero.ok_or(ParseError::MissingField("luma_eob_nonzero"))?,
    })
}

/// Inverse of [`interp_filter_tag`]. Returns `None` for unknown tags.
#[must_use]
pub fn interp_filter_from_tag(s: &str) -> Option<Option<InterpFilter>> {
    Some(match s {
        "NONE" => None,
        "EIGHTTAP" => Some(InterpFilter::EightTap),
        "EIGHTTAP_SMOOTH" => Some(InterpFilter::EightTapSmooth),
        "EIGHTTAP_SHARP" => Some(InterpFilter::EightTapSharp),
        "BILINEAR" => Some(InterpFilter::Bilinear),
        "SWITCHABLE" => Some(InterpFilter::Switchable),
        _ => return None,
    })
}

/// Inverse of [`tx_size_tag`].
#[must_use]
pub fn tx_size_from_tag(s: &str) -> Option<TxSize> {
    match s {
        "4x4" => Some(TxSize::Tx4x4),
        "8x8" => Some(TxSize::Tx8x8),
        "16x16" => Some(TxSize::Tx16x16),
        "32x32" => Some(TxSize::Tx32x32),
        _ => None,
    }
}

/// Parse an entire JSONL stream. Blank lines are skipped (matches
/// libvpx-side patch behaviour when `setvbuf(_IOLBF)` is set per HLD
/// §3.3 revision 6 — no partial-flush interleaves to recover from).
///
/// # Errors
///
/// Returns the first per-line parse error or any underlying I/O
/// error from the reader.
pub fn parse_jsonl_stream<R: BufRead>(r: R) -> Result<Vec<BlockDecision>, ParseError> {
    let mut out = Vec::new();
    for line in r.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(parse_jsonl_line(&line)?);
    }
    Ok(out)
}

/// Stable alignment key for one coded block.
///
/// Both encoders report the top-left 8×8 mi cell of each leaf. Block size is
/// deliberately not part of the key because partition choice is one of the
/// live fields the oracle compares.
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct BlockKey {
    /// 0-based presentation-order frame index.
    pub frame_idx: u32,
    /// Spatial-layer identifier. Ordinary streams use zero.
    pub spatial_id: u8,
    /// Top-left 8×8 mi-cell row.
    pub mi_row: u16,
    /// Top-left 8×8 mi-cell column.
    pub mi_col: u16,
}

impl From<&BlockDecision> for BlockKey {
    fn from(decision: &BlockDecision) -> Self {
        Self {
            frame_idx: decision.frame_idx,
            spatial_id: decision.spatial_id,
            mi_row: decision.mi_row,
            mi_col: decision.mi_col,
        }
    }
}

/// One aligned libvpx/rvp9 record pair and its comparison result.
#[derive(Clone, Debug)]
pub struct DecisionPair {
    /// Shared frame/block alignment key.
    pub key: BlockKey,
    /// First rvp9 record at this key, or `None` when the leg omitted it.
    pub rvp9: Option<BlockDecision>,
    /// First libvpx record at this key, or `None` when the leg omitted it.
    pub libvpx: Option<BlockDecision>,
    /// Number of rvp9 records at this key. Values other than one mismatch.
    pub rvp9_records: usize,
    /// Number of libvpx records at this key. Values other than one mismatch.
    pub libvpx_records: usize,
    /// Live decision fields whose values differ.
    pub mismatched_fields: Vec<&'static str>,
}

impl DecisionPair {
    /// Whether both legs emitted exactly one record and every live field agrees.
    #[must_use]
    pub fn is_match(&self) -> bool {
        self.rvp9_records == 1 && self.libvpx_records == 1 && self.mismatched_fields.is_empty()
    }
}

/// Complete pairwise comparison of two F.0.2 decision streams.
#[derive(Clone, Debug)]
pub struct OracleComparison {
    /// All keys present in either leg, in frame/row/column order.
    pub pairs: Vec<DecisionPair>,
}

impl OracleComparison {
    /// Number of aligned keys that are missing, duplicated, or field-divergent.
    #[must_use]
    pub fn mismatch_count(&self) -> usize {
        self.pairs.iter().filter(|pair| !pair.is_match()).count()
    }

    /// Enforce the default strict-oracle exit policy.
    ///
    /// Report-only mode still performs the comparison and emits diagnostics,
    /// but does not turn observed divergence into a failing process result.
    ///
    /// # Errors
    ///
    /// Returns [`OracleMismatch`] when mismatches exist and `report_only` is
    /// false.
    pub fn require_match(&self, report_only: bool) -> Result<(), OracleMismatch> {
        let mismatches = self.mismatch_count();
        if mismatches > 0 && !report_only {
            Err(OracleMismatch { mismatches })
        } else {
            Ok(())
        }
    }

    /// Write one paired JSONL diagnostic per aligned frame/block key.
    ///
    /// # Errors
    ///
    /// Returns the first I/O error from `writer`.
    pub fn write_paired_jsonl<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        for pair in &self.pairs {
            write!(
                writer,
                "{{\"frame_idx\":{},\"spatial_id\":{},\"mi_row\":{},\"mi_col\":{},\
                 \"status\":\"{}\",\"rvp9_records\":{},\"libvpx_records\":{},\
                 \"mismatched_fields\":[",
                pair.key.frame_idx,
                pair.key.spatial_id,
                pair.key.mi_row,
                pair.key.mi_col,
                if pair.is_match() { "match" } else { "mismatch" },
                pair.rvp9_records,
                pair.libvpx_records,
            )?;
            for (index, field) in pair.mismatched_fields.iter().enumerate() {
                if index > 0 {
                    writer.write_all(b",")?;
                }
                write!(writer, "\"{field}\"")?;
            }
            writer.write_all(b"],\"rvp9\":")?;
            write_optional_decision(writer, pair.rvp9.as_ref())?;
            writer.write_all(b",\"libvpx\":")?;
            write_optional_decision(writer, pair.libvpx.as_ref())?;
            writer.write_all(b"}\n")?;
        }
        Ok(())
    }
}

/// Strict-oracle failure with the number of divergent aligned keys.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct OracleMismatch {
    /// Number of divergent aligned keys.
    pub mismatches: usize,
}

impl std::fmt::Display for OracleMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "F.0.2 oracle found {} mismatched frame/block key(s)",
            self.mismatches
        )
    }
}

impl std::error::Error for OracleMismatch {}

/// Visible-8x8-cell-weighted explanation of rvp9's LAST-reference NEWMV
/// candidate wherever libvpx selected a non-zero LAST/NEWMV decision.
///
/// Exact-leaf comparison cannot answer this question when the encoders choose
/// different partition trees. These counters first splat each leaf over its
/// visible 8x8 cells, then compare the two records covering each shared cell.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct NewMvCellAnalysis {
    /// Visible cells where libvpx selected LAST/NEWMV with a non-zero vector.
    pub libvpx_last_nonzero_newmv_cells: usize,
    /// Visible cells covered by more than one rvp9 record.
    pub rvp9_overlapped_cells: usize,
    /// Visible cells covered by more than one libvpx record.
    pub libvpx_overlapped_cells: usize,
    /// Target cells for which rvp9 emitted no covering decision.
    pub rvp9_missing_cells: usize,
    /// Target cells for which rvp9's covering decision was ambiguous.
    pub rvp9_ambiguous_cells: usize,
    /// Target cells where rvp9 did not evaluate the LAST NEWMV candidate.
    pub rvp9_new_unavailable_cells: usize,
    /// Unavailable cells by [`NewMvGateReason`] numeric code.
    pub rvp9_new_unavailable_by_gate: [usize; 9],
    /// Target cells where rvp9's LAST NEWMV candidate was the zero vector.
    pub rvp9_new_zero_cells: usize,
    /// Target cells where rvp9 selected the same LAST NEWMV vector as libvpx.
    pub rvp9_new_exact_selected_cells: usize,
    /// Target cells where the same LAST NEWMV vector was available but lost.
    pub rvp9_new_exact_lost_cells: usize,
    /// Target cells where a nearby LAST NEWMV vector was selected by rvp9.
    pub rvp9_new_near_selected_cells: usize,
    /// Target cells where a nearby LAST NEWMV vector was available but lost.
    pub rvp9_new_near_lost_cells: usize,
    /// Target cells where a different non-zero LAST NEWMV vector was selected.
    pub rvp9_new_different_selected_cells: usize,
    /// Target cells where a different non-zero LAST NEWMV vector was available but lost.
    pub rvp9_new_different_lost_cells: usize,
    /// Rejected non-zero candidates whose RD cost exceeded the selected winner.
    pub rvp9_new_rejected_higher_rd_cells: usize,
    /// Rejected non-zero candidates whose RD cost did not exceed the selected winner.
    pub rvp9_new_rejected_not_higher_rd_cells: usize,
    /// Exact/near rejected candidates whose winner was LAST/NEARESTMV.
    pub rvp9_exact_or_near_lost_to_last_nearest_cells: usize,
    /// Exact/near rejected candidates whose winner was LAST/NEARMV.
    pub rvp9_exact_or_near_lost_to_last_near_cells: usize,
    /// Exact/near rejected candidates whose winner was LAST/ZEROMV.
    pub rvp9_exact_or_near_lost_to_last_zero_cells: usize,
    /// Exact/near rejected candidates whose winner was another LAST/NEWMV.
    pub rvp9_exact_or_near_lost_to_last_new_cells: usize,
    /// Exact/near rejected candidates whose winner used another reference.
    pub rvp9_exact_or_near_lost_to_other_ref_cells: usize,
    /// Exact/near rejected candidates whose winner was intra.
    pub rvp9_exact_or_near_lost_to_intra_cells: usize,
}

impl NewMvCellAnalysis {
    /// Number of libvpx target cells assigned to an rvp9 outcome category.
    #[must_use]
    pub const fn classified_cells(self) -> usize {
        self.rvp9_missing_cells
            + self.rvp9_ambiguous_cells
            + self.rvp9_new_unavailable_cells
            + self.rvp9_new_zero_cells
            + self.rvp9_new_exact_selected_cells
            + self.rvp9_new_exact_lost_cells
            + self.rvp9_new_near_selected_cells
            + self.rvp9_new_near_lost_cells
            + self.rvp9_new_different_selected_cells
            + self.rvp9_new_different_lost_cells
    }
}

#[derive(Copy, Clone)]
enum CellRecord<'a> {
    One(&'a BlockDecision),
    Overlap,
}

fn covered_cells<'a>(
    records: &'a [BlockDecision],
    mi_rows: u16,
    mi_cols: u16,
) -> BTreeMap<BlockKey, CellRecord<'a>> {
    let mut cells = BTreeMap::new();
    for decision in records {
        let row_end = decision
            .mi_row
            .saturating_add(u16::from(decision.bsize.n_8x8_high()))
            .min(mi_rows);
        let col_end = decision
            .mi_col
            .saturating_add(u16::from(decision.bsize.n_8x8_wide()))
            .min(mi_cols);
        for mi_row in decision.mi_row.min(mi_rows)..row_end {
            for mi_col in decision.mi_col.min(mi_cols)..col_end {
                let key = BlockKey {
                    frame_idx: decision.frame_idx,
                    spatial_id: decision.spatial_id,
                    mi_row,
                    mi_col,
                };
                cells
                    .entry(key)
                    .and_modify(|record| *record = CellRecord::Overlap)
                    .or_insert(CellRecord::One(decision));
            }
        }
    }
    cells
}

/// Compare rvp9's LAST NEWMV search and selection over the visible cells where
/// libvpx selected a non-zero LAST/NEWMV decision.
///
/// `mi_rows` and `mi_cols` are the visible frame dimensions in 8x8 units. A
/// candidate is "near" when both q3 vector components differ by at most one
/// pixel (8 q3 units).
#[must_use]
pub fn analyze_newmv_cells(
    rvp9: &[BlockDecision],
    libvpx: &[BlockDecision],
    mi_rows: u16,
    mi_cols: u16,
) -> NewMvCellAnalysis {
    const LAST_NEWMV_CANDIDATE: usize = 3;
    const NEAR_MV_Q3: i32 = 8;

    let rvp9_cells = covered_cells(rvp9, mi_rows, mi_cols);
    let libvpx_cells = covered_cells(libvpx, mi_rows, mi_cols);
    let mut analysis = NewMvCellAnalysis {
        rvp9_overlapped_cells: rvp9_cells
            .values()
            .filter(|record| matches!(record, CellRecord::Overlap))
            .count(),
        libvpx_overlapped_cells: libvpx_cells
            .values()
            .filter(|record| matches!(record, CellRecord::Overlap))
            .count(),
        ..NewMvCellAnalysis::default()
    };

    for (key, libvpx_record) in libvpx_cells {
        let CellRecord::One(libvpx) = libvpx_record else {
            continue;
        };
        if libvpx.mode != BlockDecisionMode::Inter(InterMode::NewMv)
            || libvpx.ref_frame != RefFrame::LastFrame
            || (libvpx.mv_row == 0 && libvpx.mv_col == 0)
        {
            continue;
        }
        analysis.libvpx_last_nonzero_newmv_cells += 1;

        let Some(rvp9_record) = rvp9_cells.get(&key) else {
            analysis.rvp9_missing_cells += 1;
            continue;
        };
        let CellRecord::One(rvp9) = rvp9_record else {
            analysis.rvp9_ambiguous_cells += 1;
            continue;
        };
        let candidate = rvp9.last_candidates[LAST_NEWMV_CANDIDATE];
        if !candidate.available {
            analysis.rvp9_new_unavailable_cells += 1;
            analysis.rvp9_new_unavailable_by_gate[candidate.newmv_gate as usize] += 1;
            continue;
        }
        if candidate.mv_row == 0 && candidate.mv_col == 0 {
            analysis.rvp9_new_zero_cells += 1;
            continue;
        }

        let exact = candidate.mv_row == libvpx.mv_row && candidate.mv_col == libvpx.mv_col;
        let near = i32::from(candidate.mv_row).abs_diff(i32::from(libvpx.mv_row))
            <= NEAR_MV_Q3 as u32
            && i32::from(candidate.mv_col).abs_diff(i32::from(libvpx.mv_col)) <= NEAR_MV_Q3 as u32;
        let selected = rvp9.mode == BlockDecisionMode::Inter(InterMode::NewMv)
            && rvp9.ref_frame == RefFrame::LastFrame
            && rvp9.mv_row == candidate.mv_row
            && rvp9.mv_col == candidate.mv_col;
        if !selected {
            if candidate.rd_cost > rvp9.picker_rd_cost {
                analysis.rvp9_new_rejected_higher_rd_cells += 1;
            } else {
                analysis.rvp9_new_rejected_not_higher_rd_cells += 1;
            }
            if near {
                match (rvp9.ref_frame, rvp9.mode) {
                    (RefFrame::LastFrame, BlockDecisionMode::Inter(InterMode::NearestMv)) => {
                        analysis.rvp9_exact_or_near_lost_to_last_nearest_cells += 1
                    }
                    (RefFrame::LastFrame, BlockDecisionMode::Inter(InterMode::NearMv)) => {
                        analysis.rvp9_exact_or_near_lost_to_last_near_cells += 1;
                    }
                    (RefFrame::LastFrame, BlockDecisionMode::Inter(InterMode::ZeroMv)) => {
                        analysis.rvp9_exact_or_near_lost_to_last_zero_cells += 1;
                    }
                    (RefFrame::LastFrame, BlockDecisionMode::Inter(InterMode::NewMv)) => {
                        analysis.rvp9_exact_or_near_lost_to_last_new_cells += 1;
                    }
                    (RefFrame::LastFrame, BlockDecisionMode::Intra(_)) => {
                        analysis.rvp9_exact_or_near_lost_to_intra_cells += 1;
                    }
                    (_, BlockDecisionMode::Inter(_)) => {
                        analysis.rvp9_exact_or_near_lost_to_other_ref_cells += 1;
                    }
                    (_, BlockDecisionMode::Intra(_)) => {
                        analysis.rvp9_exact_or_near_lost_to_intra_cells += 1;
                    }
                }
            }
        }
        match (exact, near, selected) {
            (true, _, true) => analysis.rvp9_new_exact_selected_cells += 1,
            (true, _, false) => analysis.rvp9_new_exact_lost_cells += 1,
            (false, true, true) => analysis.rvp9_new_near_selected_cells += 1,
            (false, true, false) => analysis.rvp9_new_near_lost_cells += 1,
            (false, false, true) => analysis.rvp9_new_different_selected_cells += 1,
            (false, false, false) => analysis.rvp9_new_different_lost_cells += 1,
        }
    }

    analysis
}

/// Aggregate candidate costs for exact-origin, exact-size leaves where both
/// encoders evaluated the same non-zero LAST/NEWMV vector selected by libvpx.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct MatchingNewMvCandidateCosts {
    /// Number of matching candidate records.
    pub records: usize,
    /// Records where rvp9 did not select the matching candidate.
    pub rejected_records: usize,
    /// Sum of rvp9 candidate rates.
    pub rvp9_rate_sum: u64,
    /// Sum of libvpx candidate rates.
    pub libvpx_rate_sum: u64,
    /// Sum of rvp9 luma model rates inside the matching candidates.
    pub rvp9_luma_rate_sum: u64,
    /// Sum of libvpx luma model rates inside the matching candidates.
    pub libvpx_luma_rate_sum: u64,
    /// Sum of rvp9 chroma model rates inside the matching candidates.
    pub rvp9_uv_rate_sum: u64,
    /// Sum of libvpx chroma model rates inside the matching candidates.
    pub libvpx_uv_rate_sum: u64,
    /// Matching-record counts indexed by `rvp9_skip * 3 + libvpx_skip`.
    pub skip_pair_counts: [usize; 9],
    /// Matching candidates whose source and reference-footprint hashes agree.
    pub exact_input_records: usize,
    /// Exact-input counts indexed by `rvp9_skip * 3 + libvpx_skip`.
    pub exact_input_skip_pair_counts: [usize; 9],
    /// Exact-input skip disagreements indexed by VP9 block-size enum value.
    pub exact_input_skip_mismatch_by_bsize: [usize; 13],
    /// Exact-input records where rvp9 also selected the matching candidate.
    pub exact_input_selected_records: usize,
    /// Exact-input selected records whose final interpolation filter agrees.
    pub exact_input_selected_filter_matches: usize,
    /// Sum of rvp9 model variance on exact-input records.
    pub exact_input_rvp9_var_sum: u64,
    /// Sum of libvpx model variance on exact-input records.
    pub exact_input_libvpx_var_sum: u64,
    /// Sum of rvp9 model SSE on exact-input records.
    pub exact_input_rvp9_sse_sum: u64,
    /// Sum of libvpx model SSE on exact-input records.
    pub exact_input_libvpx_sse_sum: u64,
    /// Exact-input records with the same model transform size.
    pub exact_input_tx_size_matches: usize,
    /// Sum of non-negative rvp9 candidate distortions.
    pub rvp9_dist_sum: u64,
    /// Sum of non-negative libvpx candidate distortions.
    pub libvpx_dist_sum: u64,
    /// Sum of non-negative rvp9 candidate RD costs.
    pub rvp9_rd_cost_sum: u64,
    /// Sum of non-negative libvpx candidate RD costs.
    pub libvpx_rd_cost_sum: u64,
}

/// Compare like-for-like LAST/NEWMV candidate costs on exact matching leaves.
#[must_use]
pub fn compare_matching_newmv_candidates(
    rvp9: &[BlockDecision],
    libvpx: &[BlockDecision],
) -> MatchingNewMvCandidateCosts {
    const LAST_NEWMV_CANDIDATE: usize = 3;

    let mut rvp9_by_key = BTreeMap::<BlockKey, Vec<&BlockDecision>>::new();
    for decision in rvp9 {
        rvp9_by_key
            .entry(BlockKey::from(decision))
            .or_default()
            .push(decision);
    }
    let mut costs = MatchingNewMvCandidateCosts::default();
    for libvpx in libvpx {
        if libvpx.mode != BlockDecisionMode::Inter(InterMode::NewMv)
            || libvpx.ref_frame != RefFrame::LastFrame
            || (libvpx.mv_row == 0 && libvpx.mv_col == 0)
        {
            continue;
        }
        let Some([rvp9]) = rvp9_by_key.get(&BlockKey::from(libvpx)).map(Vec::as_slice) else {
            continue;
        };
        if rvp9.bsize != libvpx.bsize {
            continue;
        }
        let rvp9_candidate = rvp9.last_candidates[LAST_NEWMV_CANDIDATE];
        let libvpx_candidate = libvpx.last_candidates[LAST_NEWMV_CANDIDATE];
        if !rvp9_candidate.available
            || !libvpx_candidate.available
            || rvp9_candidate.mv_row != libvpx.mv_row
            || rvp9_candidate.mv_col != libvpx.mv_col
            || libvpx_candidate.mv_row != libvpx.mv_row
            || libvpx_candidate.mv_col != libvpx.mv_col
        {
            continue;
        }

        costs.records += 1;
        let selected = rvp9.mode == BlockDecisionMode::Inter(InterMode::NewMv)
            && rvp9.ref_frame == RefFrame::LastFrame
            && rvp9.mv_row == rvp9_candidate.mv_row
            && rvp9.mv_col == rvp9_candidate.mv_col;
        costs.rejected_records += usize::from(!selected);
        costs.rvp9_rate_sum += u64::from(rvp9_candidate.rate);
        costs.libvpx_rate_sum += u64::from(libvpx_candidate.rate);
        costs.rvp9_luma_rate_sum += rvp9_candidate.luma_rate;
        costs.libvpx_luma_rate_sum += libvpx_candidate.luma_rate;
        costs.rvp9_uv_rate_sum += u64::from(rvp9_candidate.uv_rate);
        costs.libvpx_uv_rate_sum += u64::from(libvpx_candidate.uv_rate);
        let skip_pair = usize::from(rvp9_candidate.skip_txfm.min(2)) * 3
            + usize::from(libvpx_candidate.skip_txfm.min(2));
        costs.skip_pair_counts[skip_pair] += 1;
        if rvp9_candidate
            .newmv_search
            .zip(libvpx_candidate.newmv_search)
            .is_some_and(|(rvp9_search, libvpx_search)| {
                rvp9_search.source_hash == libvpx_search.source_hash
                    && rvp9_search.reference_hash == libvpx_search.reference_hash
            })
        {
            costs.exact_input_records += 1;
            costs.exact_input_skip_pair_counts[skip_pair] += 1;
            if selected {
                costs.exact_input_selected_records += 1;
                costs.exact_input_selected_filter_matches +=
                    usize::from(rvp9.interp_filter == libvpx.interp_filter);
            }
            costs.exact_input_rvp9_var_sum += u64::from(rvp9_candidate.model_var_y);
            costs.exact_input_libvpx_var_sum += u64::from(libvpx_candidate.model_var_y);
            costs.exact_input_rvp9_sse_sum += u64::from(rvp9_candidate.model_sse_y);
            costs.exact_input_libvpx_sse_sum += u64::from(libvpx_candidate.model_sse_y);
            costs.exact_input_tx_size_matches +=
                usize::from(rvp9_candidate.model_tx_size == libvpx_candidate.model_tx_size);
            if rvp9_candidate.skip_txfm != libvpx_candidate.skip_txfm {
                costs.exact_input_skip_mismatch_by_bsize[rvp9.bsize.idx()] += 1;
            }
        }
        costs.rvp9_dist_sum += rvp9_candidate.dist.max(0) as u64;
        costs.libvpx_dist_sum += libvpx_candidate.dist.max(0) as u64;
        costs.rvp9_rd_cost_sum += rvp9_candidate.rd_cost.max(0) as u64;
        costs.libvpx_rd_cost_sum += libvpx_candidate.rd_cost.max(0) as u64;
    }
    costs
}

/// Stage agreement for fresh LAST/NEWMV searches on exact matching leaves.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct MatchingNewMvSearchStages {
    /// Exact-origin, exact-size leaves with a fresh search trace on both legs.
    pub records: usize,
    /// Searches that chose the same full-pel start.
    pub start_matches: usize,
    /// Searches that produced the same integer-pel result.
    pub fullpel_matches: usize,
    /// Searches that produced the same final fractional result.
    pub final_matches: usize,
    /// Searches with the same second causal MV-reference input.
    pub near_mv_matches: usize,
    /// Searches with the same optional superblock predictor input.
    pub sb_pred_mv_matches: usize,
    /// Searches with the same predictor-SAD winner before full-pel search.
    pub mv_pred_mv_matches: usize,
    /// Same start, but different integer-pel result.
    pub same_start_fullpel_diverged: usize,
    /// Same integer-pel result, but different fractional result.
    pub same_fullpel_final_diverged: usize,
    /// Records with the same integer-pel result on both legs.
    pub same_fullpel_records: usize,
    /// Same-fullpel records with the same reference MV.
    pub same_fullpel_ref_mv_matches: usize,
    /// Same-fullpel records with the same fractional MV-cost multiplier.
    pub same_fullpel_error_per_bit_matches: usize,
    /// Same-fullpel records with the same visible source pixels.
    pub same_fullpel_source_hash_matches: usize,
    /// Same-fullpel records with the same fractional-search reference pixels.
    pub same_fullpel_reference_hash_matches: usize,
    /// Same-fullpel records where both pixel hashes, reference MV, and MV-cost
    /// multiplier all match and both hashes are live.
    pub exact_input_records: usize,
    /// Exact-input records that produce the same final fractional MV.
    pub exact_input_final_matches: usize,
    /// Exact-input records with a complete ordinary-tree trace on both legs.
    pub exact_input_subpel_records: usize,
    /// Complete traces with the same derived MV window.
    pub subpel_limit_matches: usize,
    /// Complete traces with the same forced-stop, extension, and HP policy.
    pub subpel_policy_matches: usize,
    /// Complete traces with the same integer-center pixel distortion.
    pub subpel_center_distortion_matches: usize,
    /// Complete traces with the same integer-center MV syntax cost.
    pub subpel_center_mv_cost_matches: usize,
    /// Complete traces with the same half-pel candidate coordinates.
    pub subpel_half_coordinate_matches: usize,
    /// Complete traces with the same half-pel pixel costs.
    pub subpel_half_distortion_matches: usize,
    /// Complete traces with the same half-pel MV syntax costs.
    pub subpel_half_mv_cost_matches: usize,
    /// Complete traces with the same half-pel winner.
    pub subpel_half_winner_matches: usize,
    /// Complete traces with the same quarter-pel candidate coordinates.
    pub subpel_quarter_coordinate_matches: usize,
    /// Complete traces with the same quarter-pel pixel costs.
    pub subpel_quarter_distortion_matches: usize,
    /// Complete traces with the same quarter-pel MV syntax costs.
    pub subpel_quarter_mv_cost_matches: usize,
    /// Complete traces with the same quarter-pel winner.
    pub subpel_quarter_winner_matches: usize,
}

/// Compare fresh LAST/NEWMV search stages on exact matching leaves.
#[must_use]
pub fn compare_matching_newmv_search_stages(
    rvp9: &[BlockDecision],
    libvpx: &[BlockDecision],
) -> MatchingNewMvSearchStages {
    const LAST_NEWMV_CANDIDATE: usize = 3;

    let mut rvp9_by_key = BTreeMap::<BlockKey, Vec<&BlockDecision>>::new();
    for decision in rvp9 {
        rvp9_by_key
            .entry(BlockKey::from(decision))
            .or_default()
            .push(decision);
    }
    let mut stages = MatchingNewMvSearchStages::default();
    for libvpx in libvpx {
        let Some([rvp9]) = rvp9_by_key.get(&BlockKey::from(libvpx)).map(Vec::as_slice) else {
            continue;
        };
        if rvp9.bsize != libvpx.bsize {
            continue;
        }
        let Some(rvp9_search) = rvp9.last_candidates[LAST_NEWMV_CANDIDATE].newmv_search else {
            continue;
        };
        let Some(libvpx_search) = libvpx.last_candidates[LAST_NEWMV_CANDIDATE].newmv_search else {
            continue;
        };

        stages.records += 1;
        let same_start = rvp9_search.start_mv == libvpx_search.start_mv;
        let same_fullpel = rvp9_search.fullpel_mv == libvpx_search.fullpel_mv;
        let same_final = rvp9_search.final_mv == libvpx_search.final_mv;
        stages.start_matches += usize::from(same_start);
        stages.fullpel_matches += usize::from(same_fullpel);
        stages.final_matches += usize::from(same_final);
        stages.near_mv_matches += usize::from(rvp9_search.near_mv == libvpx_search.near_mv);
        stages.sb_pred_mv_matches +=
            usize::from(rvp9_search.sb_pred_mv == libvpx_search.sb_pred_mv);
        stages.mv_pred_mv_matches +=
            usize::from(rvp9_search.mv_pred_mv == libvpx_search.mv_pred_mv);
        stages.same_start_fullpel_diverged += usize::from(same_start && !same_fullpel);
        stages.same_fullpel_final_diverged += usize::from(same_fullpel && !same_final);
        if same_fullpel {
            stages.same_fullpel_records += 1;
            let same_ref_mv = rvp9_search.ref_mv == libvpx_search.ref_mv;
            let same_error_per_bit = rvp9_search.error_per_bit == libvpx_search.error_per_bit;
            let same_source = rvp9_search.source_hash == libvpx_search.source_hash;
            let same_reference = rvp9_search.reference_hash == libvpx_search.reference_hash;
            stages.same_fullpel_ref_mv_matches += usize::from(same_ref_mv);
            stages.same_fullpel_error_per_bit_matches += usize::from(same_error_per_bit);
            stages.same_fullpel_source_hash_matches += usize::from(same_source);
            stages.same_fullpel_reference_hash_matches += usize::from(same_reference);
            let hashes_live = rvp9_search.source_hash != 0
                && libvpx_search.source_hash != 0
                && rvp9_search.reference_hash != 0
                && libvpx_search.reference_hash != 0;
            let exact_inputs =
                hashes_live && same_ref_mv && same_error_per_bit && same_source && same_reference;
            stages.exact_input_records += usize::from(exact_inputs);
            stages.exact_input_final_matches += usize::from(exact_inputs && same_final);
            if exact_inputs
                && let (Some(rvp9_subpel), Some(libvpx_subpel)) =
                    (rvp9_search.subpel, libvpx_search.subpel)
            {
                stages.exact_input_subpel_records += 1;
                stages.subpel_limit_matches +=
                    usize::from(rvp9_subpel.limits == libvpx_subpel.limits);
                stages.subpel_policy_matches += usize::from(
                    rvp9_subpel.forced_stop == libvpx_subpel.forced_stop
                        && rvp9_subpel.iters_per_step == libvpx_subpel.iters_per_step
                        && rvp9_subpel.allow_hp == libvpx_subpel.allow_hp,
                );
                stages.subpel_center_distortion_matches +=
                    usize::from(rvp9_subpel.center_distortion == libvpx_subpel.center_distortion);
                stages.subpel_center_mv_cost_matches +=
                    usize::from(rvp9_subpel.center_mv_cost == libvpx_subpel.center_mv_cost);
                stages.subpel_half_coordinate_matches += usize::from(
                    rvp9_subpel.half.coordinate_hash == libvpx_subpel.half.coordinate_hash,
                );
                stages.subpel_half_distortion_matches += usize::from(
                    rvp9_subpel.half.distortion_hash == libvpx_subpel.half.distortion_hash,
                );
                stages.subpel_half_mv_cost_matches +=
                    usize::from(rvp9_subpel.half.mv_cost_hash == libvpx_subpel.half.mv_cost_hash);
                stages.subpel_half_winner_matches +=
                    usize::from(rvp9_subpel.half.winner == libvpx_subpel.half.winner);
                stages.subpel_quarter_coordinate_matches += usize::from(
                    rvp9_subpel.quarter.coordinate_hash == libvpx_subpel.quarter.coordinate_hash,
                );
                stages.subpel_quarter_distortion_matches += usize::from(
                    rvp9_subpel.quarter.distortion_hash == libvpx_subpel.quarter.distortion_hash,
                );
                stages.subpel_quarter_mv_cost_matches += usize::from(
                    rvp9_subpel.quarter.mv_cost_hash == libvpx_subpel.quarter.mv_cost_hash,
                );
                stages.subpel_quarter_winner_matches +=
                    usize::from(rvp9_subpel.quarter.winner == libvpx_subpel.quarter.winner);
            }
        }
    }
    stages
}

/// Align and compare the fields that are live on both F.0.2 legs.
///
/// The three rate-estimate fields remain present in paired diagnostics but are
/// not strict comparison fields: the pinned libvpx patch intentionally emits
/// zero placeholders for them. Missing records and duplicate keys are always
/// mismatches.
#[must_use]
pub fn compare_decisions(rvp9: &[BlockDecision], libvpx: &[BlockDecision]) -> OracleComparison {
    let mut aligned = BTreeMap::<BlockKey, (Vec<BlockDecision>, Vec<BlockDecision>)>::new();
    for decision in rvp9 {
        aligned
            .entry(BlockKey::from(decision))
            .or_default()
            .0
            .push(*decision);
    }
    for decision in libvpx {
        aligned
            .entry(BlockKey::from(decision))
            .or_default()
            .1
            .push(*decision);
    }

    let pairs = aligned
        .into_iter()
        .map(|(key, (rvp9_records, libvpx_records))| {
            let rvp9 = rvp9_records.first().copied();
            let libvpx = libvpx_records.first().copied();
            let mismatched_fields = match (rvp9.as_ref(), libvpx.as_ref()) {
                (Some(rvp9), Some(libvpx)) => live_field_mismatches(rvp9, libvpx),
                _ => Vec::new(),
            };
            DecisionPair {
                key,
                rvp9,
                libvpx,
                rvp9_records: rvp9_records.len(),
                libvpx_records: libvpx_records.len(),
                mismatched_fields,
            }
        })
        .collect();
    OracleComparison { pairs }
}

fn live_field_mismatches(rvp9: &BlockDecision, libvpx: &BlockDecision) -> Vec<&'static str> {
    let mut fields = Vec::new();
    if rvp9.bsize != libvpx.bsize {
        fields.push("bsize");
    }
    if rvp9.mode != libvpx.mode {
        fields.push("mode");
    }
    if rvp9.ref_frame != libvpx.ref_frame {
        fields.push("ref_frame");
    }
    if rvp9.mv_row != libvpx.mv_row {
        fields.push("mv_row");
    }
    if rvp9.mv_col != libvpx.mv_col {
        fields.push("mv_col");
    }
    if rvp9.sse_after_pred != libvpx.sse_after_pred {
        fields.push("sse_after_pred");
    }
    if rvp9.skip_flag != libvpx.skip_flag {
        fields.push("skip_flag");
    }
    fields
}

fn write_optional_decision<W: Write>(
    writer: &mut W,
    decision: Option<&BlockDecision>,
) -> io::Result<()> {
    let Some(decision) = decision else {
        return writer.write_all(b"null");
    };
    write!(
        writer,
        "{{\"bsize\":\"{}\",\"mode\":\"{}\",\"ref_frame\":\"{}\",\
         \"mv_row\":{},\"mv_col\":{},\"mode_bits_est\":{},\
         \"mv_bits_est\":{},\"coef_bits_est\":{},\"sse_after_pred\":{},\
         \"skip_flag\":{}}}",
        bsize_tag(decision.bsize),
        decision.mode.as_tag(),
        ref_frame_tag(decision.ref_frame),
        decision.mv_row,
        decision.mv_col,
        decision.mode_bits_est,
        decision.mv_bits_est,
        decision.coef_bits_est,
        decision.sse_after_pred,
        decision.skip_flag,
    )
}

/// Splits top-level commas inside a JSON-like object body.
///
/// **Assumes string values contain no escape sequences** — true for the
/// F.0.2 schema (all values are integers, plain ASCII tags, or
/// booleans; no `\"` or `\\` inside any string). The backslash
/// short-circuit handles the simple case `\"` (one backslash before
/// one quote) but does NOT correctly handle consecutive backslashes
/// like `\\\"`. Do NOT reuse this function for general JSONL parsing.
fn split_top_level_commas(inner: &str) -> Result<Vec<&str>, ParseError> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut prev_backslash = false;
    let mut start = 0usize;
    for (i, ch) in inner.char_indices() {
        if in_str {
            if ch == '\\' && !prev_backslash {
                prev_backslash = true;
                continue;
            }
            if ch == '"' && !prev_backslash {
                in_str = false;
            }
            prev_backslash = false;
            continue;
        }
        match ch {
            '"' => in_str = true,
            '{' | '[' => depth += 1,
            '}' | ']' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&inner[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if in_str {
        return Err(ParseError::Malformed("unterminated string"));
    }
    if depth != 0 {
        return Err(ParseError::Malformed("unbalanced braces"));
    }
    let tail = &inner[start..];
    if !tail.trim().is_empty() {
        out.push(tail);
    }
    Ok(out)
}

fn split_kv(kv: &str) -> Result<(&str, &str), ParseError> {
    let kv = kv.trim();
    let colon = kv
        .find(':')
        .ok_or(ParseError::Malformed("missing ':' in kv pair"))?;
    let (k_raw, v_raw) = kv.split_at(colon);
    let v_raw = &v_raw[1..]; // skip the ':'
    let k = k_raw
        .trim()
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .ok_or(ParseError::Malformed("unquoted key"))?;
    Ok((k, v_raw.trim()))
}

fn parse_string(v: &str, field: &'static str) -> Result<String, ParseError> {
    v.strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .map(|s| s.to_string())
        .ok_or(ParseError::BadValue {
            field,
            raw: v.to_string(),
        })
}

fn parse_num<T: std::str::FromStr>(v: &str, field: &'static str) -> Result<T, ParseError> {
    v.parse::<T>().map_err(|_| ParseError::BadValue {
        field,
        raw: v.to_string(),
    })
}

fn parse_bool(v: &str, field: &'static str) -> Result<bool, ParseError> {
    match v {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(ParseError::BadValue {
            field,
            raw: v.to_string(),
        }),
    }
}

fn parse_candidate_probes(v: &str) -> Result<[CandidateProbe; 4], ParseError> {
    let raw = parse_string(v, "last_candidates")?;
    let mut probes = Vec::with_capacity(4);
    for encoded in raw.split('|') {
        let fields = encoded.split(':').collect::<Vec<_>>();
        if !matches!(fields.len(), 11 | 12 | 18 | 23 | 42 | 46 | 48 | 51) {
            return Err(ParseError::BadValue {
                field: "last_candidates",
                raw,
            });
        }
        let available = match fields[0] {
            "0" => false,
            "1" => true,
            _ => {
                return Err(ParseError::BadValue {
                    field: "last_candidates",
                    raw,
                });
            }
        };
        let newmv_gate = fields
            .get(11)
            .map_or(Ok(NewMvGateReason::Unknown), |field| {
                let code = parse_num(field, "last_candidates")?;
                NewMvGateReason::from_code(code).ok_or_else(|| ParseError::BadValue {
                    field: "last_candidates",
                    raw: raw.clone(),
                })
            })?;
        let newmv_search = if matches!(fields.len(), 18 | 23 | 42 | 46 | 48 | 51) {
            let values = fields[12..18]
                .iter()
                .map(|field| parse_num::<i16>(field, "last_candidates"))
                .collect::<Result<Vec<_>, _>>()?;
            if values.iter().all(|&value| value == i16::MIN) {
                None
            } else if values.contains(&i16::MIN) {
                return Err(ParseError::BadValue {
                    field: "last_candidates",
                    raw,
                });
            } else {
                let has_input_trace = matches!(fields.len(), 23 | 42 | 46 | 48 | 51);
                let ref_mv = if has_input_trace {
                    Mv {
                        row: parse_num(fields[18], "last_candidates")?,
                        col: parse_num(fields[19], "last_candidates")?,
                    }
                } else {
                    Mv::ZERO
                };
                let subpel = if matches!(fields.len(), 42 | 46 | 48 | 51) {
                    let limits = fields[23..27]
                        .iter()
                        .map(|field| parse_num::<i32>(field, "last_candidates"))
                        .collect::<Result<Vec<_>, _>>()?;
                    if limits.iter().all(|&value| value == i32::MIN) {
                        None
                    } else if limits.contains(&i32::MIN) {
                        return Err(ParseError::BadValue {
                            field: "last_candidates",
                            raw,
                        });
                    } else {
                        let allow_hp = match parse_num::<i32>(fields[29], "last_candidates")? {
                            0 => false,
                            1 => true,
                            _ => {
                                return Err(ParseError::BadValue {
                                    field: "last_candidates",
                                    raw,
                                });
                            }
                        };
                        Some(SubPixelSearchTrace {
                            limits: MvLimits::new(limits[0], limits[1], limits[2], limits[3]),
                            forced_stop: parse_num(fields[27], "last_candidates")?,
                            iters_per_step: parse_num(fields[28], "last_candidates")?,
                            allow_hp,
                            center_distortion: parse_num(fields[30], "last_candidates")?,
                            center_mv_cost: parse_num(fields[31], "last_candidates")?,
                            half: SubPixelLevelTrace {
                                coordinate_hash: parse_num(fields[32], "last_candidates")?,
                                distortion_hash: parse_num(fields[33], "last_candidates")?,
                                mv_cost_hash: parse_num(fields[34], "last_candidates")?,
                                winner: Mv {
                                    row: parse_num(fields[35], "last_candidates")?,
                                    col: parse_num(fields[36], "last_candidates")?,
                                },
                            },
                            quarter: SubPixelLevelTrace {
                                coordinate_hash: parse_num(fields[37], "last_candidates")?,
                                distortion_hash: parse_num(fields[38], "last_candidates")?,
                                mv_cost_hash: parse_num(fields[39], "last_candidates")?,
                                winner: Mv {
                                    row: parse_num(fields[40], "last_candidates")?,
                                    col: parse_num(fields[41], "last_candidates")?,
                                },
                            },
                        })
                    }
                } else {
                    None
                };
                Some(NewMvSearchTrace {
                    start_mv: Mv {
                        row: values[0],
                        col: values[1],
                    },
                    fullpel_mv: Mv {
                        row: values[2],
                        col: values[3],
                    },
                    final_mv: Mv {
                        row: values[4],
                        col: values[5],
                    },
                    ref_mv,
                    near_mv: if matches!(fields.len(), 46 | 48 | 51) {
                        Mv {
                            row: parse_num(fields[42], "last_candidates")?,
                            col: parse_num(fields[43], "last_candidates")?,
                        }
                    } else {
                        Mv::ZERO
                    },
                    sb_pred_mv: if matches!(fields.len(), 46 | 48 | 51) {
                        let row = parse_num(fields[44], "last_candidates")?;
                        let col = parse_num(fields[45], "last_candidates")?;
                        match (row == i16::MIN, col == i16::MIN) {
                            (true, true) => None,
                            (false, false) => Some(Mv { row, col }),
                            _ => {
                                return Err(ParseError::BadValue {
                                    field: "last_candidates",
                                    raw,
                                });
                            }
                        }
                    } else {
                        None
                    },
                    mv_pred_mv: if matches!(fields.len(), 48 | 51) {
                        Mv {
                            row: parse_num(fields[46], "last_candidates")?,
                            col: parse_num(fields[47], "last_candidates")?,
                        }
                    } else {
                        Mv::ZERO
                    },
                    error_per_bit: if has_input_trace {
                        parse_num(fields[20], "last_candidates")?
                    } else {
                        0
                    },
                    source_hash: if has_input_trace {
                        parse_num(fields[21], "last_candidates")?
                    } else {
                        0
                    },
                    reference_hash: if has_input_trace {
                        parse_num(fields[22], "last_candidates")?
                    } else {
                        0
                    },
                    subpel,
                })
            }
        } else {
            None
        };
        probes.push(CandidateProbe {
            available,
            mv_row: parse_num(fields[1], "last_candidates")?,
            mv_col: parse_num(fields[2], "last_candidates")?,
            rate: parse_num(fields[3], "last_candidates")?,
            dist: parse_num(fields[4], "last_candidates")?,
            rd_cost: parse_num(fields[5], "last_candidates")?,
            luma_rate: parse_num(fields[6], "last_candidates")?,
            luma_dist: parse_num(fields[7], "last_candidates")?,
            uv_rate: parse_num(fields[8], "last_candidates")?,
            uv_dist: parse_num(fields[9], "last_candidates")?,
            skip_txfm: parse_num(fields[10], "last_candidates")?,
            model_var_y: fields
                .get(48)
                .map_or(Ok(0), |field| parse_num(field, "last_candidates"))?,
            model_sse_y: fields
                .get(49)
                .map_or(Ok(0), |field| parse_num(field, "last_candidates"))?,
            model_tx_size: fields
                .get(50)
                .map_or(Ok(0), |field| parse_num(field, "last_candidates"))?,
            newmv_gate,
            newmv_search,
        });
    }
    probes.try_into().map_err(|_| ParseError::BadValue {
        field: "last_candidates",
        raw,
    })
}

/// Inverse of [`bsize_tag`]. Returns `None` for unknown tags.
#[must_use]
pub fn bsize_from_tag(s: &str) -> Option<BlockSize> {
    Some(match s {
        "4x4" => BlockSize::Block4x4,
        "4x8" => BlockSize::Block4x8,
        "8x4" => BlockSize::Block8x4,
        "8x8" => BlockSize::Block8x8,
        "8x16" => BlockSize::Block8x16,
        "16x8" => BlockSize::Block16x8,
        "16x16" => BlockSize::Block16x16,
        "16x32" => BlockSize::Block16x32,
        "32x16" => BlockSize::Block32x16,
        "32x32" => BlockSize::Block32x32,
        "32x64" => BlockSize::Block32x64,
        "64x32" => BlockSize::Block64x32,
        "64x64" => BlockSize::Block64x64,
        _ => return None,
    })
}

/// Inverse of [`ref_frame_tag`]. Returns `None` for unknown tags.
#[must_use]
pub fn ref_frame_from_tag(s: &str) -> Option<RefFrame> {
    Some(match s {
        "NONE" => RefFrame::NoRefFrame,
        "INTRA" => RefFrame::IntraFrame,
        "LAST" => RefFrame::LastFrame,
        "GOLDEN" => RefFrame::GoldenFrame,
        "ALTREF" => RefFrame::AltrefFrame,
        _ => return None,
    })
}

/// Resolve the flat `mode` tag against the leaf's `ref_frame`. Intra
/// tags require `ref_frame == IntraFrame`; inter tags require
/// `ref_frame.is_inter()`.
#[must_use]
pub fn mode_from_tag(s: &str, ref_frame: RefFrame) -> Option<BlockDecisionMode> {
    let intra = match s {
        "DC" => Some(IntraMode::Dc),
        "V" => Some(IntraMode::V),
        "H" => Some(IntraMode::H),
        "D45" => Some(IntraMode::D45),
        "D135" => Some(IntraMode::D135),
        "D117" => Some(IntraMode::D117),
        "D153" => Some(IntraMode::D153),
        "D207" => Some(IntraMode::D207),
        "D63" => Some(IntraMode::D63),
        "TM" => Some(IntraMode::Tm),
        _ => None,
    };
    if let Some(m) = intra {
        if matches!(ref_frame, RefFrame::IntraFrame) {
            return Some(BlockDecisionMode::Intra(m));
        }
        // Otherwise the libvpx-emitted tags are mismatched; treat as
        // a value error so the caller surfaces it.
        return None;
    }
    let inter = match s {
        "NEARESTMV" => Some(InterMode::NearestMv),
        "NEARMV" => Some(InterMode::NearMv),
        "ZEROMV" => Some(InterMode::ZeroMv),
        "NEWMV" => Some(InterMode::NewMv),
        _ => None,
    }?;
    if ref_frame.is_inter() {
        Some(BlockDecisionMode::Inter(inter))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Unit tests — schema round-trip is the contract.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

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
            last_candidates: [
                CandidateProbe {
                    available: true,
                    mv_row: -8,
                    mv_col: 16,
                    rate: 7_000,
                    dist: 12_000,
                    rd_cost: 99_000,
                    luma_rate: 6_000,
                    luma_dist: 11_000,
                    uv_rate: 500,
                    uv_dist: 1_000,
                    skip_txfm: 1,
                    model_var_y: 8_000,
                    model_sse_y: 9_000,
                    model_tx_size: 1,
                    newmv_gate: NewMvGateReason::Unknown,
                    newmv_search: Some(NewMvSearchTrace {
                        start_mv: Mv { row: 8, col: -8 },
                        fullpel_mv: Mv { row: 16, col: -8 },
                        final_mv: Mv { row: 24, col: -8 },
                        ref_mv: Mv { row: -8, col: 16 },
                        near_mv: Mv { row: 0, col: 8 },
                        sb_pred_mv: Some(Mv { row: 0, col: 48 }),
                        mv_pred_mv: Mv { row: 0, col: 8 },
                        error_per_bit: 321,
                        source_hash: 123,
                        reference_hash: 456,
                        subpel: Some(SubPixelSearchTrace {
                            limits: MvLimits::new(-128, 128, -64, 64),
                            forced_stop: 1,
                            iters_per_step: 0,
                            allow_hp: false,
                            center_distortion: 789,
                            center_mv_cost: 12,
                            half: SubPixelLevelTrace {
                                coordinate_hash: 1_001,
                                distortion_hash: 1_002,
                                mv_cost_hash: 1_003,
                                winner: Mv { row: 20, col: -8 },
                            },
                            quarter: SubPixelLevelTrace {
                                coordinate_hash: 2_001,
                                distortion_hash: 2_002,
                                mv_cost_hash: 2_003,
                                winner: Mv { row: 22, col: -8 },
                            },
                        }),
                    }),
                },
                CandidateProbe::default(),
                CandidateProbe::default(),
                CandidateProbe::default(),
            ],
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
            mode: BlockDecisionMode::Intra(IntraMode::Tm),
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
    fn round_trip_inter() {
        let d = sample_inter();
        let mut buf = Vec::new();
        d.write_jsonl(&mut buf).unwrap();
        let line = std::str::from_utf8(&buf).unwrap();
        let parsed = parse_jsonl_line(line).unwrap();
        assert_eq!(d, parsed, "inter BlockDecision round-trips through JSONL");
    }

    #[test]
    fn round_trip_intra() {
        let d = sample_intra();
        let mut buf = Vec::new();
        d.write_jsonl(&mut buf).unwrap();
        let line = std::str::from_utf8(&buf).unwrap();
        let parsed = parse_jsonl_line(line).unwrap();
        assert_eq!(d, parsed, "intra BlockDecision round-trips through JSONL");
    }

    #[test]
    fn round_trip_stream_multi_record() {
        let records = [sample_inter(), sample_intra(), sample_inter()];
        let mut buf = Vec::new();
        for r in &records {
            r.write_jsonl(&mut buf).unwrap();
        }
        let parsed = parse_jsonl_stream(Cursor::new(buf)).unwrap();
        assert_eq!(parsed.len(), records.len());
        for (a, b) in records.iter().zip(parsed.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn missing_field_reports_field_name() {
        // Drop "skip_flag" from a valid line and confirm the error
        // names it.
        let line = "{\"frame_idx\":0,\"mi_row\":0,\"mi_col\":0,\
                    \"bsize\":\"8x8\",\"mode\":\"DC\",\"ref_frame\":\"INTRA\",\
                    \"mv_row\":0,\"mv_col\":0,\
                    \"mode_bits_est\":0,\"mv_bits_est\":0,\"ref_bits_est\":0,\
                    \"filter_bits_est\":0,\"skip_bits_est\":0,\"coef_bits_est\":0,\
                    \"picker_rate\":0,\"picker_dist\":0,\"picker_rd_cost\":0,\
                    \"model_var_y\":0,\"model_tx_size\":\"8x8\",\
                    \"model_skip_txfm\":0,\
                    \"luma_coef_rate\":0,\"luma_coef_dist\":0,\
                    \"uv_model_rate\":0,\"uv_model_dist\":0,\
                    \"uv_u_sensitive\":false,\"uv_v_sensitive\":false,\
                    \"partition_u_sensitive\":false,\"partition_v_sensitive\":false,\
                    \"last_candidates\":\"0:0:0:0:0:0:0:0:0:0:0|0:0:0:0:0:0:0:0:0:0:0|0:0:0:0:0:0:0:0:0:0:0|0:0:0:0:0:0:0:0:0:0:0\",\
                    \"sse_after_pred\":0,\"interp_filter\":\"NONE\"}";
        let err = parse_jsonl_line(line).unwrap_err();
        match err {
            ParseError::MissingField(n) => assert_eq!(n, "skip_flag"),
            other => panic!("expected MissingField, got {other:?}"),
        }
    }

    #[test]
    fn bad_bsize_tag_rejected() {
        let line = "{\"frame_idx\":0,\"mi_row\":0,\"mi_col\":0,\
                    \"bsize\":\"NOT_A_BSIZE\",\"mode\":\"DC\",\"ref_frame\":\"INTRA\",\
                    \"mv_row\":0,\"mv_col\":0,\
                    \"mode_bits_est\":0,\"mv_bits_est\":0,\"coef_bits_est\":0,\
                    \"sse_after_pred\":0,\"interp_filter\":\"NONE\",\"skip_flag\":true}";
        let err = parse_jsonl_line(line).unwrap_err();
        assert!(
            matches!(err, ParseError::BadValue { field: "bsize", .. }),
            "expected ParseError::BadValue {{ field: \"bsize\", .. }}, got {err:?}"
        );
    }

    #[test]
    fn intra_mode_with_non_intra_ref_rejected() {
        // Inter ref + intra mode tag is a schema violation (drift R5).
        let line = "{\"frame_idx\":0,\"mi_row\":0,\"mi_col\":0,\
                    \"bsize\":\"8x8\",\"mode\":\"DC\",\"ref_frame\":\"LAST\",\
                    \"mv_row\":0,\"mv_col\":0,\
                    \"mode_bits_est\":0,\"mv_bits_est\":0,\"coef_bits_est\":0,\
                    \"sse_after_pred\":0,\"interp_filter\":\"NONE\",\"skip_flag\":true}";
        let err = parse_jsonl_line(line).unwrap_err();
        assert!(
            matches!(err, ParseError::BadValue { field: "mode", .. }),
            "expected ParseError::BadValue {{ field: \"mode\", .. }}, got {err:?}"
        );
    }

    #[test]
    fn unknown_field_is_forward_compatible() {
        // A future schema may add fields; existing parsers must skip
        // them without rejecting valid records.
        let line = "{\"frame_idx\":1,\"mi_row\":2,\"mi_col\":3,\
                    \"bsize\":\"8x8\",\"mode\":\"DC\",\"ref_frame\":\"INTRA\",\
                    \"mv_row\":0,\"mv_col\":0,\
                    \"mode_bits_est\":1,\"mv_bits_est\":0,\"ref_bits_est\":0,\
                    \"filter_bits_est\":0,\"skip_bits_est\":0,\"coef_bits_est\":0,\
                    \"picker_rate\":0,\"picker_dist\":0,\"picker_rd_cost\":0,\
                    \"model_var_y\":0,\"model_tx_size\":\"8x8\",\
                    \"model_skip_txfm\":0,\
                    \"luma_coef_rate\":0,\"luma_coef_dist\":0,\
                    \"uv_model_rate\":0,\"uv_model_dist\":0,\
                    \"uv_u_sensitive\":false,\"uv_v_sensitive\":false,\
                    \"partition_u_sensitive\":false,\"partition_v_sensitive\":false,\
                    \"last_candidates\":\"0:0:0:0:0:0:0:0:0:0:0|0:0:0:0:0:0:0:0:0:0:0|0:0:0:0:0:0:0:0:0:0:0|0:0:0:0:0:0:0:0:0:0:0\",\
                    \"sse_after_pred\":0,\"interp_filter\":\"NONE\",\"skip_flag\":true,\
                    \"q_index\":65,\"tx_size\":\"8x8\",\"luma_eob_nonzero\":false,\
                    \"future_field\":\"ignored\"}";
        let d = parse_jsonl_line(line).unwrap();
        assert_eq!(d.frame_idx, 1);
        assert_eq!(d.mi_row, 2);
        assert_eq!(d.mi_col, 3);
    }

    #[test]
    fn divergent_live_field_is_a_strict_oracle_failure() {
        let rvp9 = sample_inter();
        let mut libvpx = rvp9;
        libvpx.mv_col += 1;

        let comparison = compare_decisions(&[rvp9], &[libvpx]);

        assert_eq!(comparison.mismatch_count(), 1);
        assert_eq!(comparison.pairs[0].mismatched_fields, ["mv_col"]);
        assert!(comparison.require_match(false).is_err());
        assert!(comparison.require_match(true).is_ok());
    }

    #[test]
    fn comparison_aligns_by_key_and_ignores_rate_placeholders() {
        let first = sample_inter();
        let mut second = sample_intra();
        second.frame_idx = 8;
        let mut libvpx_first = first;
        libvpx_first.mode_bits_est = 0;
        libvpx_first.mv_bits_est = 0;
        libvpx_first.coef_bits_est = 0;

        let comparison = compare_decisions(&[second, first], &[libvpx_first, second]);

        assert_eq!(comparison.mismatch_count(), 0);
        assert_eq!(comparison.pairs[0].key.frame_idx, 7);
        assert_eq!(comparison.pairs[1].key.frame_idx, 8);
    }

    #[test]
    fn missing_and_duplicate_keys_are_mismatches() {
        let decision = sample_inter();
        let missing = compare_decisions(&[decision], &[]);
        let duplicate = compare_decisions(&[decision, decision], &[decision]);

        assert_eq!(missing.mismatch_count(), 1);
        assert_eq!(missing.pairs[0].libvpx_records, 0);
        assert_eq!(duplicate.mismatch_count(), 1);
        assert_eq!(duplicate.pairs[0].rvp9_records, 2);
    }

    #[test]
    fn newmv_analysis_aligns_different_partition_trees_by_covered_cell() {
        let mut libvpx = sample_inter();
        libvpx.bsize = BlockSize::Block32x32;

        let mut exact_selected = sample_inter();
        exact_selected.bsize = BlockSize::Block16x16;
        exact_selected.last_candidates[3] = CandidateProbe {
            available: true,
            mv_row: libvpx.mv_row,
            mv_col: libvpx.mv_col,
            ..CandidateProbe::default()
        };

        let mut exact_lost = exact_selected;
        exact_lost.mi_col += 2;
        exact_lost.mode = BlockDecisionMode::Inter(InterMode::ZeroMv);
        exact_lost.mv_row = 0;
        exact_lost.mv_col = 0;

        let mut near_lost = exact_lost;
        near_lost.mi_row += 2;
        near_lost.mi_col -= 2;
        near_lost.last_candidates[3].mv_row += 4;
        near_lost.last_candidates[3].mv_col += 4;

        let mut unavailable = exact_lost;
        unavailable.mi_row += 2;
        unavailable.last_candidates[3] = CandidateProbe::default();
        unavailable.last_candidates[3].newmv_gate = NewMvGateReason::AdaptiveRdThreshold;

        let analysis = analyze_newmv_cells(
            &[exact_selected, exact_lost, near_lost, unavailable],
            &[libvpx],
            8,
            16,
        );

        assert_eq!(analysis.libvpx_last_nonzero_newmv_cells, 16);
        assert_eq!(analysis.rvp9_new_exact_selected_cells, 4);
        assert_eq!(analysis.rvp9_new_exact_lost_cells, 4);
        assert_eq!(analysis.rvp9_new_near_lost_cells, 4);
        assert_eq!(analysis.rvp9_new_unavailable_cells, 4);
        assert_eq!(
            analysis.rvp9_new_unavailable_by_gate[NewMvGateReason::AdaptiveRdThreshold as usize],
            4
        );
        assert_eq!(analysis.classified_cells(), 16);
    }

    #[test]
    fn newmv_analysis_distinguishes_different_and_zero_candidates() {
        let mut libvpx_first = sample_inter();
        libvpx_first.bsize = BlockSize::Block8x8;
        let mut libvpx_second = libvpx_first;
        libvpx_second.mi_col += 1;

        let mut different = libvpx_first;
        different.mv_row = 80;
        different.mv_col = 80;
        different.last_candidates[3] = CandidateProbe {
            available: true,
            mv_row: different.mv_row,
            mv_col: different.mv_col,
            ..CandidateProbe::default()
        };

        let mut zero = libvpx_second;
        zero.mode = BlockDecisionMode::Inter(InterMode::ZeroMv);
        zero.mv_row = 0;
        zero.mv_col = 0;
        zero.last_candidates[3] = CandidateProbe {
            available: true,
            ..CandidateProbe::default()
        };

        let analysis =
            analyze_newmv_cells(&[different, zero], &[libvpx_first, libvpx_second], 8, 16);

        assert_eq!(analysis.libvpx_last_nonzero_newmv_cells, 2);
        assert_eq!(analysis.rvp9_new_different_selected_cells, 1);
        assert_eq!(analysis.rvp9_new_zero_cells, 1);
        assert_eq!(analysis.classified_cells(), 2);
    }

    #[test]
    fn newmv_analysis_reports_the_winner_and_same_leaf_candidate_costs() {
        let mut libvpx = sample_inter();
        libvpx.bsize = BlockSize::Block16x16;
        libvpx.last_candidates[3] = CandidateProbe {
            available: true,
            mv_row: libvpx.mv_row,
            mv_col: libvpx.mv_col,
            rate: 200,
            dist: 300,
            rd_cost: 400,
            luma_rate: 120,
            uv_rate: 30,
            skip_txfm: 1,
            model_var_y: 40,
            model_sse_y: 50,
            model_tx_size: 1,
            ..CandidateProbe::default()
        };
        let exact_trace = sample_inter().last_candidates[0].newmv_search;
        libvpx.last_candidates[3].newmv_search = exact_trace;

        let mut rvp9 = libvpx;
        rvp9.mode = BlockDecisionMode::Inter(InterMode::NearestMv);
        rvp9.mv_row = 0;
        rvp9.mv_col = 0;
        rvp9.last_candidates[3].rate = 180;
        rvp9.last_candidates[3].dist = 290;
        rvp9.last_candidates[3].rd_cost = 380;
        rvp9.last_candidates[3].luma_rate = 100;
        rvp9.last_candidates[3].uv_rate = 20;
        rvp9.last_candidates[3].skip_txfm = 2;
        rvp9.last_candidates[3].model_var_y = 35;
        rvp9.last_candidates[3].model_sse_y = 45;
        rvp9.last_candidates[3].newmv_search = exact_trace;

        let cells = analyze_newmv_cells(&[rvp9], &[libvpx], 8, 16);
        let costs = compare_matching_newmv_candidates(&[rvp9], &[libvpx]);

        assert_eq!(cells.rvp9_exact_or_near_lost_to_last_nearest_cells, 4);
        assert_eq!(costs.records, 1);
        assert_eq!(costs.rejected_records, 1);
        assert_eq!(costs.rvp9_rate_sum, 180);
        assert_eq!(costs.libvpx_rate_sum, 200);
        assert_eq!(costs.rvp9_luma_rate_sum, 100);
        assert_eq!(costs.libvpx_luma_rate_sum, 120);
        assert_eq!(costs.rvp9_uv_rate_sum, 20);
        assert_eq!(costs.libvpx_uv_rate_sum, 30);
        assert_eq!(costs.skip_pair_counts[7], 1);
        assert_eq!(costs.exact_input_records, 1);
        assert_eq!(costs.exact_input_selected_records, 0);
        assert_eq!(costs.exact_input_rvp9_var_sum, 35);
        assert_eq!(costs.exact_input_libvpx_var_sum, 40);
        assert_eq!(costs.exact_input_rvp9_sse_sum, 45);
        assert_eq!(costs.exact_input_libvpx_sse_sum, 50);
        assert_eq!(costs.exact_input_tx_size_matches, 1);
        assert_eq!(costs.rvp9_dist_sum, 290);
        assert_eq!(costs.libvpx_dist_sum, 300);
        assert_eq!(costs.rvp9_rd_cost_sum, 380);
        assert_eq!(costs.libvpx_rd_cost_sum, 400);
    }

    #[test]
    fn newmv_search_stage_comparison_localizes_first_divergence() {
        let mut libvpx = sample_inter();
        libvpx.last_candidates[3].newmv_search = Some(NewMvSearchTrace {
            start_mv: Mv { row: 8, col: 0 },
            fullpel_mv: Mv { row: 16, col: 0 },
            final_mv: Mv { row: 18, col: 0 },
            ref_mv: Mv::ZERO,
            near_mv: Mv { row: 0, col: 8 },
            sb_pred_mv: Some(Mv { row: 0, col: 48 }),
            mv_pred_mv: Mv { row: 0, col: 8 },
            error_per_bit: 100,
            source_hash: 1,
            reference_hash: 2,
            subpel: None,
        });
        let mut rvp9 = libvpx;
        rvp9.last_candidates[3].newmv_search = Some(NewMvSearchTrace {
            start_mv: Mv { row: 8, col: 0 },
            fullpel_mv: Mv { row: 24, col: 0 },
            final_mv: Mv { row: 26, col: 0 },
            ref_mv: Mv::ZERO,
            near_mv: Mv { row: 0, col: 8 },
            sb_pred_mv: Some(Mv { row: 0, col: 48 }),
            mv_pred_mv: Mv { row: 0, col: 8 },
            error_per_bit: 100,
            source_hash: 1,
            reference_hash: 2,
            subpel: None,
        });

        let stages = compare_matching_newmv_search_stages(&[rvp9], &[libvpx]);

        assert_eq!(stages.records, 1);
        assert_eq!(stages.start_matches, 1);
        assert_eq!(stages.fullpel_matches, 0);
        assert_eq!(stages.final_matches, 0);
        assert_eq!(stages.same_start_fullpel_diverged, 1);
        assert_eq!(stages.same_fullpel_final_diverged, 0);
        assert_eq!(stages.near_mv_matches, 1);
        assert_eq!(stages.sb_pred_mv_matches, 1);
        assert_eq!(stages.mv_pred_mv_matches, 1);
    }

    #[test]
    fn newmv_search_stage_comparison_requires_every_exact_input() {
        let mut libvpx = sample_inter();
        libvpx.last_candidates[3].newmv_search = Some(NewMvSearchTrace {
            start_mv: Mv::ZERO,
            fullpel_mv: Mv { row: 16, col: -8 },
            final_mv: Mv { row: 18, col: -8 },
            ref_mv: Mv { row: 8, col: 0 },
            near_mv: Mv { row: 0, col: 8 },
            sb_pred_mv: Some(Mv { row: 0, col: 48 }),
            mv_pred_mv: Mv { row: 0, col: 8 },
            error_per_bit: 321,
            source_hash: 123,
            reference_hash: 456,
            subpel: None,
        });
        let mut rvp9 = libvpx;
        rvp9.last_candidates[3].newmv_search = Some(NewMvSearchTrace {
            final_mv: Mv { row: 16, col: -6 },
            source_hash: 123,
            ..libvpx.last_candidates[3].newmv_search.unwrap()
        });

        let stages = compare_matching_newmv_search_stages(&[rvp9], &[libvpx]);

        assert_eq!(stages.same_fullpel_records, 1);
        assert_eq!(stages.same_fullpel_ref_mv_matches, 1);
        assert_eq!(stages.same_fullpel_error_per_bit_matches, 1);
        assert_eq!(stages.same_fullpel_source_hash_matches, 1);
        assert_eq!(stages.same_fullpel_reference_hash_matches, 1);
        assert_eq!(stages.exact_input_records, 1);
        assert_eq!(stages.exact_input_final_matches, 0);
    }

    #[test]
    fn paired_jsonl_contains_both_legs_and_mismatch_fields() {
        let rvp9 = sample_inter();
        let mut libvpx = rvp9;
        libvpx.skip_flag = !libvpx.skip_flag;
        let comparison = compare_decisions(&[rvp9], &[libvpx]);
        let mut output = Vec::new();

        comparison.write_paired_jsonl(&mut output).unwrap();

        let json: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(json["status"], "mismatch");
        assert_eq!(json["mismatched_fields"][0], "skip_flag");
        assert!(json["rvp9"].is_object());
        assert!(json["libvpx"].is_object());
    }
}
