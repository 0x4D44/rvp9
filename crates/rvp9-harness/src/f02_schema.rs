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
//! schema is a flat 13-field JSON object, no nesting; a 1-pass
//! tokeniser handles it. The unit tests below guarantee round-trip
//! consistency with the canonical writer.

use std::io::{self, BufRead};

pub use rvp9_encoder::f02_oracle::{BlockDecision, BlockDecisionMode, bsize_tag, ref_frame_tag};

use rvp9_core::block::BlockSize;
use rvp9_core::intra_pred::IntraMode;
use rvp9_core::mode_info::{InterMode, RefFrame};

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
    let mut mi_row: Option<u16> = None;
    let mut mi_col: Option<u16> = None;
    let mut bsize: Option<BlockSize> = None;
    let mut mode_tag: Option<String> = None;
    let mut ref_tag: Option<String> = None;
    let mut mv_row: Option<i16> = None;
    let mut mv_col: Option<i16> = None;
    let mut mode_bits_est: Option<u32> = None;
    let mut mv_bits_est: Option<u32> = None;
    let mut coef_bits_est: Option<u32> = None;
    let mut sse_after_pred: Option<u64> = None;
    let mut skip_flag: Option<bool> = None;

    for kv in split_top_level_commas(inner)? {
        let (k, v) = split_kv(kv)?;
        match k {
            "frame_idx" => frame_idx = Some(parse_num(v, "frame_idx")?),
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
            "coef_bits_est" => coef_bits_est = Some(parse_num(v, "coef_bits_est")?),
            "sse_after_pred" => sse_after_pred = Some(parse_num(v, "sse_after_pred")?),
            "skip_flag" => skip_flag = Some(parse_bool(v, "skip_flag")?),
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
        mi_row: mi_row.ok_or(ParseError::MissingField("mi_row"))?,
        mi_col: mi_col.ok_or(ParseError::MissingField("mi_col"))?,
        bsize: bsize.ok_or(ParseError::MissingField("bsize"))?,
        mode: mode_resolved,
        ref_frame: ref_frame_resolved,
        mv_row: mv_row.ok_or(ParseError::MissingField("mv_row"))?,
        mv_col: mv_col.ok_or(ParseError::MissingField("mv_col"))?,
        mode_bits_est: mode_bits_est.ok_or(ParseError::MissingField("mode_bits_est"))?,
        mv_bits_est: mv_bits_est.ok_or(ParseError::MissingField("mv_bits_est"))?,
        coef_bits_est: coef_bits_est.ok_or(ParseError::MissingField("coef_bits_est"))?,
        sse_after_pred: sse_after_pred.ok_or(ParseError::MissingField("sse_after_pred"))?,
        skip_flag: skip_flag.ok_or(ParseError::MissingField("skip_flag"))?,
    })
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
            sse_after_pred: 1_234_567,
            skip_flag: false,
        }
    }

    fn sample_intra() -> BlockDecision {
        BlockDecision {
            frame_idx: 0,
            mi_row: 0,
            mi_col: 0,
            bsize: BlockSize::Block8x8,
            mode: BlockDecisionMode::Intra(IntraMode::Tm),
            ref_frame: RefFrame::IntraFrame,
            mv_row: 0,
            mv_col: 0,
            mode_bits_est: 90,
            mv_bits_est: 0,
            coef_bits_est: 0,
            sse_after_pred: 0,
            skip_flag: true,
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
                    \"mode_bits_est\":0,\"mv_bits_est\":0,\"coef_bits_est\":0,\
                    \"sse_after_pred\":0}";
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
                    \"sse_after_pred\":0,\"skip_flag\":true}";
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
                    \"sse_after_pred\":0,\"skip_flag\":true}";
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
                    \"mode_bits_est\":1,\"mv_bits_est\":0,\"coef_bits_est\":0,\
                    \"sse_after_pred\":0,\"skip_flag\":true,\
                    \"future_field\":\"ignored\"}";
        let d = parse_jsonl_line(line).unwrap();
        assert_eq!(d.frame_idx, 1);
        assert_eq!(d.mi_row, 2);
        assert_eq!(d.mi_col, 3);
    }
}
