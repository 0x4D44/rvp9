//! Replay pinned-libvpx VBP JSONL records through rvp9's pure fixed-tree seam.
//!
//! The input is produced by `third_party/libvpx-patches/vbp_oracle.patch`.
//! Every captured source/predictor pair is identical on both sides, so a
//! mismatch identifies tree arithmetic or control-flow drift rather than
//! divergent reconstructed references.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use rvp9_core::mode_decision::inter::NoiseLevel;
use rvp9_core::mode_decision::partition::{
    FixedTreeAccumulatorTrace, FixedTreeInterInput, FixedTreeNodeAccumulatorTrace,
    FixedTreePartitionSymbol, FixedTreeVarianceTrace, choose_fixed_tree_inter,
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
    frame_height_le_360: bool,
    low_resolution: bool,
    thresholds: [i64; 4],
    minmax_threshold: i32,
    force_64x64_split: bool,
    compute_minmax_variance: bool,
    segment_is_boosted: bool,
    noise_level: u8,
    source: String,
    predictor: String,
    force_split: Vec<u8>,
    used_4x4_averages: Vec<u8>,
    minmax_16x16: Vec<i32>,
    variance_valid: Vec<bool>,
    variances: Vec<Option<OraclePartitionVariance>>,
    alternate_4x4_variances: Vec<Option<OraclePartitionVariance>>,
    nodes: Vec<[i32; 4]>,
    leaves: Vec<[i32; 3]>,
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

fn compare_record(record: &OracleRecord) -> Result<(), String> {
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
    let input = FixedTreeInterInput {
        source: &source,
        predictor: &predictor,
        visible_width: record.visible_width,
        visible_height: record.visible_height,
        thresholds: record.thresholds,
        minmax_threshold: record.minmax_threshold,
        force_64x64_split: record.force_64x64_split,
        compute_minmax_variance: record.compute_minmax_variance,
        segment_is_boosted: record.segment_is_boosted,
        noise_level: noise_level(record.noise_level)?,
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

    let actual_nodes: Vec<[i32; 4]> = actual
        .nodes
        .iter()
        .map(|node| {
            [
                node.row_8x8 as i32,
                node.col_8x8 as i32,
                node.parent_size as i32,
                symbol_number(node.symbol),
            ]
        })
        .collect();
    if actual_nodes != record.nodes {
        return Err(format!(
            "parent symbols differ: expected {:?}, got {:?}",
            record.nodes, actual_nodes
        ));
    }

    let actual_leaves: Vec<[i32; 3]> = actual
        .leaves
        .iter()
        .map(|leaf| {
            [
                leaf.row_8x8 as i32,
                leaf.col_8x8 as i32,
                leaf.block_size as i32,
            ]
        })
        .collect();
    if actual_leaves != record.leaves {
        return Err(format!(
            "ordered leaves differ: expected {:?}, got {:?}",
            record.leaves, actual_leaves
        ));
    }
    if !actual.decision.is_valid_partition() {
        return Err("rvp9 returned an invalid partition tree".into());
    }
    Ok(())
}

fn replay_file(path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    let reader = BufReader::new(File::open(path)?);
    let mut count = 0usize;
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: OracleRecord = serde_json::from_str(&line)?;
        compare_record(&record).map_err(|message| {
            format!(
                "{}:{} frame={} mi=({},{}): {message}",
                path.display(),
                line_index + 1,
                record.frame,
                record.mi_row,
                record.mi_col,
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
}
