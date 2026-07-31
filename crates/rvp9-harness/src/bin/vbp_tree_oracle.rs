//! Replay pinned-libvpx VBP JSONL records through rvp9's pure fixed-tree seam.
//!
//! The input is produced by `third_party/libvpx-patches/vbp_oracle.patch`.
//! Every captured source/predictor pair is identical on both sides, so a
//! mismatch identifies tree arithmetic or control-flow drift rather than
//! divergent reconstructed references.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use rvp9_core::mode_decision::inter::{ContentStateSb, NoiseEstimate, NoiseLevel};
use rvp9_core::mode_decision::partition::{
    Cpu7InterVbpThresholdInput, FixedTreeAccumulatorTrace, FixedTreeCopyConditions,
    FixedTreeHistoryEntry, FixedTreeInterInput, FixedTreeNodeAccumulatorTrace,
    FixedTreePartitionSymbol, FixedTreePartitionTrace, FixedTreeVarianceTrace, PartitionDecision,
    choose_fixed_tree_inter, cpu7_inter_vbp_thresholds,
};
use serde::Deserialize;

type OracleVar = (u64, i64, u8, i32);
type OraclePartitionVariance = [OracleVar; 5];

#[derive(Debug, Deserialize)]
struct OracleRecord {
    frame: u64,
    mi_row: usize,
    mi_col: usize,
    visible_width: usize,
    visible_height: usize,
    frame_width: usize,
    frame_height: usize,
    frame_height_le_360: bool,
    low_resolution: bool,
    q_index: u8,
    avg_inter_qindex: u8,
    thresholds: [i64; 4],
    minmax_threshold: i32,
    force_64x64_split: bool,
    compute_minmax_variance: bool,
    segment_is_boosted: bool,
    noise_estimate_enabled: bool,
    noise_level: u8,
    content_state: u8,
    partition_ref: u8,
    partition_mv_row: i16,
    partition_mv_col: i16,
    #[serde(default)]
    low_temp_var_mode: u8,
    segment_id: u8,
    previous_segment_before: u8,
    previous_segment_after: u8,
    copied_before: u8,
    copied_after: u8,
    source: String,
    predictor: String,
    variance_low: Vec<u8>,
    history_variance_low: Vec<u8>,
    force_split: Vec<u8>,
    used_4x4_averages: Vec<u8>,
    minmax_16x16: Vec<i32>,
    variance_valid: Vec<bool>,
    variances: Vec<Option<OraclePartitionVariance>>,
    alternate_4x4_variances: Vec<Option<OraclePartitionVariance>>,
    nodes: Vec<[i32; 4]>,
    leaves: Vec<[i32; 3]>,
}

#[derive(Debug, Deserialize)]
struct OracleHistoryRecord {
    early_exit: String,
    frame: u64,
    mi_row: usize,
    mi_col: usize,
    frames_since_key: u32,
    segment_id: u8,
    previous_segment_before: u8,
    previous_segment_after: u8,
    copied_before: u8,
    copied_after: u8,
    max_copied_frame: u8,
    variance_low: Vec<u8>,
    history_variance_low: Vec<u8>,
    nodes: Vec<[i32; 4]>,
    leaves: Vec<[i32; 3]>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OracleLine {
    Tree(Box<OracleRecord>),
    History(OracleHistoryRecord),
}

fn decode_hex_4096(value: &str) -> Result<Vec<u8>, String> {
    if value.len() != 8192 {
        return Err(format!("expected 8192 hex characters, got {}", value.len()));
    }
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(4096);
    for index in (0..bytes.len()).step_by(2) {
        let high = hex_nibble(bytes[index])?;
        let low = hex_nibble(bytes[index + 1])?;
        out.push((high << 4) | low);
    }
    Ok(out)
}

fn hex_nibble(value: u8) -> Result<u8, String> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(format!("invalid hex byte {value:#04x}")),
    }
}

fn noise_level(value: u8) -> Result<NoiseLevel, String> {
    match value {
        0 => Ok(NoiseLevel::LowLow),
        1 => Ok(NoiseLevel::Low),
        2 => Ok(NoiseLevel::Medium),
        3 => Ok(NoiseLevel::High),
        _ => Err(format!("invalid libvpx noise level {value}")),
    }
}

fn content_state(value: u8) -> Result<ContentStateSb, String> {
    match value {
        1 => Ok(ContentStateSb::LowSadLowSumdiff),
        2 => Ok(ContentStateSb::LowSadHighSumdiff),
        3 => Ok(ContentStateSb::HighSadLowSumdiff),
        4 => Ok(ContentStateSb::HighSadHighSumdiff),
        5 => Ok(ContentStateSb::LowVarHighSumdiff),
        6 => Ok(ContentStateSb::VeryHighSad),
        _ => Err(format!("invalid libvpx content state {value}")),
    }
}

fn accumulator_values(node: FixedTreeNodeAccumulatorTrace) -> [FixedTreeAccumulatorTrace; 5] {
    [
        node.none,
        node.horizontal[0],
        node.horizontal[1],
        node.vertical[0],
        node.vertical[1],
    ]
}

fn variance_values(node: FixedTreeVarianceTrace) -> [i32; 5] {
    [
        node.none,
        node.horizontal[0],
        node.horizontal[1],
        node.vertical[0],
        node.vertical[1],
    ]
}

fn compare_partition_variance(
    label: &str,
    expected: &OraclePartitionVariance,
    accumulators: FixedTreeNodeAccumulatorTrace,
    variances: FixedTreeVarianceTrace,
) -> Result<(), String> {
    let actual_accumulators = accumulator_values(accumulators);
    let actual_variances = variance_values(variances);
    for slot in 0..5 {
        let expected_var = expected[slot];
        let actual_accumulator = actual_accumulators[slot];
        let actual = (
            actual_accumulator.sum_square_error,
            actual_accumulator.sum_error,
            actual_accumulator.log2_count,
            actual_variances[slot],
        );
        if actual != expected_var {
            return Err(format!(
                "{label} slot {slot}: expected {expected_var:?}, got {actual:?}"
            ));
        }
    }
    Ok(())
}

fn symbol_number(symbol: FixedTreePartitionSymbol) -> i32 {
    match symbol {
        FixedTreePartitionSymbol::None => 0,
        FixedTreePartitionSymbol::Horizontal => 1,
        FixedTreePartitionSymbol::Vertical => 2,
        FixedTreePartitionSymbol::Split => 3,
    }
}

fn compare_record(record: &OracleRecord) -> Result<FixedTreePartitionTrace, String> {
    if record.force_split.len() != 21
        || record.variance_valid.len() != 21
        || record.variances.len() != 21
        || record.used_4x4_averages.len() != 16
        || record.minmax_16x16.len() != 16
        || record.alternate_4x4_variances.len() != 16
    {
        return Err("oracle record has invalid fixed-array lengths".into());
    }
    let source = decode_hex_4096(&record.source)?;
    let predictor = decode_hex_4096(&record.predictor)?;
    let noise_level = noise_level(record.noise_level)?;
    let content_state = content_state(record.content_state)?;
    let expected_thresholds = cpu7_inter_vbp_thresholds(Cpu7InterVbpThresholdInput {
        q_index: record.q_index,
        avg_inter_qindex: record.avg_inter_qindex,
        frame_width: record.frame_width,
        frame_height: record.frame_height,
        noise_estimate: NoiseEstimate {
            enabled: record.noise_estimate_enabled,
            level: noise_level,
        },
        content_state,
    });
    // Slot 3 is a keyframe-only 8x8 threshold retained from the most recent
    // frame-level setup; the inter helper rewrites only slots 0..=2.
    if expected_thresholds.thresholds[..3] != record.thresholds[..3] {
        return Err(format!(
            "thresholds: expected {:?}, got {:?}",
            record.thresholds, expected_thresholds.thresholds
        ));
    }
    if expected_thresholds.minmax_threshold != record.minmax_threshold {
        return Err(format!(
            "minmax_threshold: expected {}, got {}",
            record.minmax_threshold, expected_thresholds.minmax_threshold
        ));
    }
    let input = FixedTreeInterInput {
        source: &source,
        predictor: &predictor,
        visible_width: record.visible_width,
        visible_height: record.visible_height,
        frame_width: record.frame_width,
        thresholds: record.thresholds,
        minmax_threshold: record.minmax_threshold,
        force_64x64_split: record.force_64x64_split,
        compute_minmax_variance: record.compute_minmax_variance,
        segment_is_boosted: record.segment_is_boosted,
        partition_ref_is_last: record.partition_ref == 1,
        partition_mv_row: record.partition_mv_row,
        partition_mv_col: record.partition_mv_col,
        low_temp_var_mode: record.low_temp_var_mode,
        noise_level,
        low_resolution: record.low_resolution,
        frame_height_le_360: record.frame_height_le_360,
    };
    let actual = choose_fixed_tree_inter(&input);

    for index in 0..21 {
        let expected_force = record.force_split[index] != 0;
        if actual.force_split[index] != expected_force {
            return Err(format!(
                "force_split[{index}]: expected {expected_force}, got {}",
                actual.force_split[index]
            ));
        }
        if actual.variance_valid[index] != record.variance_valid[index] {
            return Err(format!(
                "variance_valid[{index}]: expected {}, got {}",
                record.variance_valid[index], actual.variance_valid[index]
            ));
        }
        match &record.variances[index] {
            Some(expected) => compare_partition_variance(
                &format!("variances[{index}]"),
                expected,
                actual.accumulators[index],
                actual.variances[index],
            )?,
            None if actual.variance_valid[index] => {
                return Err(format!("variances[{index}] is unexpectedly valid"));
            }
            None => {}
        }
    }

    for index in 0..16 {
        let expected_used = record.used_4x4_averages[index] != 0;
        if actual.used_4x4_averages[index] != expected_used {
            return Err(format!(
                "used_4x4_averages[{index}]: expected {expected_used}, got {}",
                actual.used_4x4_averages[index]
            ));
        }
        if actual.minmax_16x16[index] != record.minmax_16x16[index] {
            return Err(format!(
                "minmax_16x16[{index}]: expected {}, got {}",
                record.minmax_16x16[index], actual.minmax_16x16[index]
            ));
        }
        match &record.alternate_4x4_variances[index] {
            Some(expected) => compare_partition_variance(
                &format!("alternate_4x4_variances[{index}]"),
                expected,
                actual.alternate_4x4_accumulators[index],
                actual.alternate_4x4_variances[index],
            )?,
            None if expected_used => {
                return Err(format!(
                    "alternate_4x4_variances[{index}] missing for a used node"
                ));
            }
            None => {}
        }
    }

    let actual_nodes = numeric_nodes(&actual.nodes);
    if actual_nodes != record.nodes {
        return Err(format!(
            "parent symbols differ: expected {:?}, got {:?}",
            record.nodes, actual_nodes
        ));
    }

    let actual_leaves = numeric_leaves(&actual.leaves);
    if actual_leaves != record.leaves {
        return Err(format!(
            "ordered leaves differ: expected {:?}, got {:?}",
            record.leaves, actual_leaves
        ));
    }
    if !actual.decision.is_valid_partition() {
        return Err("rvp9 returned an invalid partition tree".into());
    }
    let expected_variance_low = bool_array_25(&record.variance_low, "variance_low")?;
    if actual.variance_low != expected_variance_low {
        return Err(format!(
            "low-temporal-variance flags differ: expected {:?}, got {:?}",
            expected_variance_low, actual.variance_low
        ));
    }
    Ok(actual)
}

fn bool_array_25(values: &[u8], label: &str) -> Result<[bool; 25], String> {
    if values.len() != 25 || values.iter().any(|&value| value > 1) {
        return Err(format!("{label}: expected 25 zero/one entries"));
    }
    let mut result = [false; 25];
    for (index, &value) in values.iter().enumerate() {
        result[index] = value != 0;
    }
    Ok(result)
}

fn numeric_nodes(
    nodes: &[rvp9_core::mode_decision::partition::FixedTreeNodeDecision],
) -> Vec<[i32; 4]> {
    nodes
        .iter()
        .map(|node| {
            [
                node.row_8x8 as i32,
                node.col_8x8 as i32,
                node.parent_size as i32,
                symbol_number(node.symbol),
            ]
        })
        .collect()
}

fn numeric_leaves(leaves: &[rvp9_core::mode_decision::partition::FixedTreeLeaf]) -> Vec<[i32; 3]> {
    leaves
        .iter()
        .map(|leaf| {
            [
                leaf.row_8x8 as i32,
                leaf.col_8x8 as i32,
                leaf.block_size as i32,
            ]
        })
        .collect()
}

fn check_history_before(
    history: &FixedTreeHistoryEntry,
    previous_segment: u8,
    copied_count: u8,
) -> Result<(), String> {
    if history.previous_segment_id() != previous_segment {
        return Err(format!(
            "previous segment before: expected {previous_segment}, got {}",
            history.previous_segment_id()
        ));
    }
    if history.copied_frame_count() != copied_count {
        return Err(format!(
            "copied count before: expected {copied_count}, got {}",
            history.copied_frame_count()
        ));
    }
    Ok(())
}

fn check_history_after(
    history: &FixedTreeHistoryEntry,
    previous_segment: u8,
    copied_count: u8,
) -> Result<(), String> {
    if history.previous_segment_id() != previous_segment {
        return Err(format!(
            "previous segment after: expected {previous_segment}, got {}",
            history.previous_segment_id()
        ));
    }
    if history.copied_frame_count() != copied_count {
        return Err(format!(
            "copied count after: expected {copied_count}, got {}",
            history.copied_frame_count()
        ));
    }
    Ok(())
}

fn compare_history_record(
    record: &OracleHistoryRecord,
    history: &mut FixedTreeHistoryEntry,
) -> Result<(), String> {
    check_history_before(
        history,
        record.previous_segment_before,
        record.copied_before,
    )?;
    let history_variance_low = bool_array_25(&record.history_variance_low, "history_variance_low")?;
    match record.early_exit.as_str() {
        "low_source_sad_copy" | "low_last_sad_copy" => {
            let copied = history
                .try_copy(FixedTreeCopyConditions {
                    frames_since_key: record.frames_since_key,
                    segment_id: record.segment_id,
                    max_copied_frames: record.max_copied_frame,
                    resize_pending: false,
                    svc_copy_allowed: true,
                })
                .ok_or_else(|| format!("{} was not eligible to copy", record.early_exit))?;
            if numeric_nodes(&copied.nodes) != record.nodes {
                return Err(format!("{} parent symbols differ", record.early_exit));
            }
            if numeric_leaves(&copied.leaves) != record.leaves {
                return Err(format!("{} ordered leaves differ", record.early_exit));
            }
            let current_variance_low = bool_array_25(&record.variance_low, "variance_low")?;
            if copied.variance_low != current_variance_low
                || copied.variance_low != history_variance_low
            {
                return Err(format!("{} variance history differs", record.early_exit));
            }
        }
        "low_selected_sad_64x64" => {
            let expected_nodes = vec![[
                0,
                0,
                rvp9_core::block::BlockSize::Block64x64 as i32,
                symbol_number(FixedTreePartitionSymbol::None),
            ]];
            let expected_leaves = vec![[0, 0, rvp9_core::block::BlockSize::Block64x64 as i32]];
            if record.nodes != expected_nodes || record.leaves != expected_leaves {
                return Err("low selected-SAD exit did not select one 64x64 leaf".into());
            }
            history.update(
                PartitionDecision::all_64x64(),
                record.segment_id,
                history_variance_low,
            );
        }
        other => return Err(format!("unsupported early exit {other:?}")),
    }
    check_history_after(history, record.previous_segment_after, record.copied_after)
}

fn replay_file(path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    let reader = BufReader::new(File::open(path)?);
    let mut histories = HashMap::<(usize, usize), FixedTreeHistoryEntry>::new();
    let mut count = 0usize;
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: OracleLine = serde_json::from_str(&line)?;
        let result = match &record {
            OracleLine::Tree(record) => {
                let history = histories.entry((record.mi_row, record.mi_col)).or_default();
                check_history_before(
                    history,
                    record.previous_segment_before,
                    record.copied_before,
                )
                .and_then(|()| compare_record(record))
                .and_then(|trace| {
                    let history_variance_low =
                        bool_array_25(&record.history_variance_low, "history_variance_low")?;
                    history.update(trace.decision, record.segment_id, history_variance_low);
                    check_history_after(history, record.previous_segment_after, record.copied_after)
                })
            }
            OracleLine::History(record) => {
                let history = histories.entry((record.mi_row, record.mi_col)).or_default();
                compare_history_record(record, history)
            }
        };
        result.map_err(|message| {
            let (frame, mi_row, mi_col) = match &record {
                OracleLine::Tree(record) => (record.frame, record.mi_row, record.mi_col),
                OracleLine::History(record) => (record.frame, record.mi_row, record.mi_col),
            };
            format!(
                "{}:{} frame={} mi=({},{}): {message}",
                path.display(),
                line_index + 1,
                frame,
                mi_row,
                mi_col,
            )
        })?;
        count += 1;
    }
    Ok(count)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<_> = std::env::args_os().skip(1).collect();
    if paths.is_empty() {
        return Err("usage: vbp-tree-oracle <capture.jsonl> [capture.jsonl ...]".into());
    }

    let mut total = 0usize;
    for path in paths {
        total += replay_file(Path::new(&path))?;
    }
    println!("PASS: {total} pinned-libvpx VBP records matched exactly");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history_record(
        early_exit: &str,
        frame: u64,
        copied_before: u8,
        copied_after: u8,
    ) -> OracleHistoryRecord {
        let mut variance_low = vec![0; 25];
        variance_low[0] = 1;
        OracleHistoryRecord {
            early_exit: early_exit.into(),
            frame,
            mi_row: 0,
            mi_col: 0,
            frames_since_key: frame as u32,
            segment_id: 0,
            previous_segment_before: 0,
            previous_segment_after: 0,
            copied_before,
            copied_after,
            max_copied_frame: 2,
            variance_low: variance_low.clone(),
            history_variance_low: variance_low,
            nodes: vec![[
                0,
                0,
                rvp9_core::block::BlockSize::Block64x64 as i32,
                symbol_number(FixedTreePartitionSymbol::None),
            ]],
            leaves: vec![[0, 0, rvp9_core::block::BlockSize::Block64x64 as i32]],
        }
    }

    #[test]
    fn hex_decoder_accepts_both_cases() {
        let mut encoded = String::with_capacity(8192);
        encoded.push_str("aF");
        encoded.push_str(&"00".repeat(4095));
        let decoded = decode_hex_4096(&encoded).expect("valid fixture");
        assert_eq!(decoded.len(), 4096);
        assert_eq!(decoded[0], 0xaf);
        assert!(decoded[1..].iter().all(|&value| value == 0));
    }

    #[test]
    fn history_replay_updates_then_copies_the_pinned_tree() {
        let mut history = FixedTreeHistoryEntry::default();
        compare_history_record(
            &history_record("low_selected_sad_64x64", 1, 0, 0),
            &mut history,
        )
        .expect("selected-SAD exit stores its tree");
        compare_history_record(
            &history_record("low_source_sad_copy", 2, 0, 1),
            &mut history,
        )
        .expect("next quiet frame copies the stored tree");
    }
}
