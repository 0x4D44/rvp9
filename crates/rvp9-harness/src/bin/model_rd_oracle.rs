//! Replay exact libvpx regular luma model-RD inputs through rvp9.
//!
//! Build the instrumented reference with `scripts/build-f02-libvpx.sh`,
//! run it with `VPX_F02_MODEL_ORACLE_OUT=<path>`, then compare:
//!
//! ```text
//! cargo run -p rvp9-harness --release --bin model-rd-oracle -- \
//!   --input target/model-rd/libvpx.jsonl
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use rvp9_core::block::BlockSize;
use rvp9_core::mode_decision::model_rd::{
    ModelRdResult, ModelRdYInput, SkipTxfm, model_rd_for_sb_y,
};
use rvp9_core::transforms::types::TxMode;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Sample {
    sample: u64,
    frame_idx: u64,
    bsize: u8,
    mode: u8,
    mv_row: i16,
    mv_col: i16,
    bw: usize,
    bh: usize,
    dc_quant: u32,
    ac_quant: u32,
    dc_thr: i64,
    ac_thr: i64,
    tx_mode: u8,
    source_variance: u32,
    rate: i32,
    dist: i64,
    var_y: u32,
    sse_y: u32,
    tx_size: u8,
    skip_txfm: u8,
    src_hex: String,
    pred_hex: String,
}

#[derive(Default)]
struct Totals {
    samples: usize,
    exact: usize,
    malformed: usize,
    var_sse: usize,
    tx_size: usize,
    skip_txfm: usize,
    rate_dist: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("model-rd-oracle: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let input = parse_input(std::env::args_os().skip(1))?;
    let file =
        File::open(&input).map_err(|error| format!("cannot open {}: {error}", input.display()))?;
    let mut totals = Totals::default();

    for (line_idx, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|error| format!("{}:{}: {error}", input.display(), line_idx + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        totals.samples += 1;
        match replay_line(&line) {
            Ok((sample, actual)) => {
                let mut differences = Vec::new();
                if (actual.var_y, actual.sse_y) != (sample.var_y, sample.sse_y) {
                    totals.var_sse += 1;
                    differences.push(format!(
                        "var/sse {}/{} != {}/{}",
                        actual.var_y, actual.sse_y, sample.var_y, sample.sse_y
                    ));
                }
                if actual.tx_size as u8 != sample.tx_size {
                    totals.tx_size += 1;
                    differences.push(format!(
                        "tx_size {} != {}",
                        actual.tx_size as u8, sample.tx_size
                    ));
                }
                if skip_number(actual.skip_txfm) != sample.skip_txfm {
                    totals.skip_txfm += 1;
                    differences.push(format!(
                        "skip {} != {}",
                        skip_number(actual.skip_txfm),
                        sample.skip_txfm
                    ));
                }
                if (actual.rate, actual.dist) != (sample.rate, sample.dist) {
                    totals.rate_dist += 1;
                    differences.push(format!(
                        "rate/dist {}/{} != {}/{}",
                        actual.rate, actual.dist, sample.rate, sample.dist
                    ));
                }
                if differences.is_empty() {
                    totals.exact += 1;
                } else if totals.samples - totals.exact - totals.malformed <= 12 {
                    eprintln!(
                        "sample {} frame {} bsize {} mode {} mv ({},{}), source_var {}: {}",
                        sample.sample,
                        sample.frame_idx,
                        sample.bsize,
                        sample.mode,
                        sample.mv_row,
                        sample.mv_col,
                        sample.source_variance,
                        differences.join("; ")
                    );
                }
            }
            Err(error) => {
                totals.malformed += 1;
                if totals.malformed <= 12 {
                    eprintln!("line {}: {error}", line_idx + 1);
                }
            }
        }
    }

    println!(
        "samples={} exact={} malformed={} var_sse_mismatch={} tx_size_mismatch={} \
         skip_mismatch={} rate_dist_mismatch={}",
        totals.samples,
        totals.exact,
        totals.malformed,
        totals.var_sse,
        totals.tx_size,
        totals.skip_txfm,
        totals.rate_dist
    );

    if totals.samples == 0 {
        return Err("input contained no samples".into());
    }
    if totals.exact != totals.samples {
        return Err(format!(
            "{} of {} samples differed",
            totals.samples - totals.exact,
            totals.samples
        ));
    }
    Ok(())
}

fn parse_input(args: impl Iterator<Item = std::ffi::OsString>) -> Result<PathBuf, String> {
    let args: Vec<_> = args.collect();
    match args.as_slice() {
        [flag, path] if flag == "--input" => Ok(PathBuf::from(path)),
        _ => Err("usage: model-rd-oracle --input <libvpx.jsonl>".into()),
    }
}

fn replay_line(line: &str) -> Result<(Sample, ModelRdResult), String> {
    let sample: Sample =
        serde_json::from_str(line).map_err(|error| format!("invalid JSON: {error}"))?;
    let bsize = BlockSize::from_dimensions(
        u8::try_from(sample.bw).map_err(|_| "block width exceeds u8")?,
        u8::try_from(sample.bh).map_err(|_| "block height exceeds u8")?,
    )
    .ok_or_else(|| format!("invalid block dimensions {}x{}", sample.bw, sample.bh))?;
    if bsize as u8 != sample.bsize {
        return Err(format!(
            "bsize {} disagrees with dimensions {}x{} ({})",
            sample.bsize, sample.bw, sample.bh, bsize as u8
        ));
    }
    let expected_pixels = sample
        .bw
        .checked_mul(sample.bh)
        .ok_or("block dimensions overflow")?;
    let src = decode_hex(&sample.src_hex)?;
    let pred = decode_hex(&sample.pred_hex)?;
    if src.len() != expected_pixels || pred.len() != expected_pixels {
        return Err(format!(
            "pixel length src={} pred={} expected={expected_pixels}",
            src.len(),
            pred.len()
        ));
    }
    let tx_mode = parse_tx_mode(sample.tx_mode)?;
    let actual = model_rd_for_sb_y(ModelRdYInput {
        src: &src,
        src_stride: sample.bw,
        pred: &pred,
        pred_stride: sample.bw,
        bw: sample.bw,
        bh: sample.bh,
        bsize,
        dc_quant: sample.dc_quant,
        ac_quant: sample.ac_quant,
        dc_thr: sample.dc_thr,
        ac_thr: sample.ac_thr,
        tx_mode,
    });
    Ok((sample, actual))
}

fn parse_tx_mode(raw: u8) -> Result<TxMode, String> {
    match raw {
        0 => Ok(TxMode::OnlyTx4x4),
        1 => Ok(TxMode::AllowTx8x8),
        2 => Ok(TxMode::AllowTx16x16),
        3 => Ok(TxMode::AllowTx32x32),
        4 => Ok(TxMode::TxModeSelect),
        _ => Err(format!("invalid tx_mode {raw}")),
    }
}

fn skip_number(skip: SkipTxfm) -> u8 {
    match skip {
        SkipTxfm::None => 0,
        SkipTxfm::AcOnly => 1,
        SkipTxfm::AcDc => 2,
    }
}

fn decode_hex(text: &str) -> Result<Vec<u8>, String> {
    if !text.len().is_multiple_of(2) {
        return Err("hex string has odd length".into());
    }
    text.as_bytes()
        .chunks_exact(2)
        .enumerate()
        .map(|(index, pair)| {
            let high = hex_nibble(pair[0]).ok_or_else(|| format!("invalid hex at byte {index}"))?;
            let low = hex_nibble(pair[1]).ok_or_else(|| format!("invalid hex at byte {index}"))?;
            Ok((high << 4) | low)
        })
        .collect()
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_decoder_accepts_both_cases() {
        assert_eq!(decode_hex("00aF10").unwrap(), [0, 0xaf, 0x10]);
    }

    #[test]
    fn hex_decoder_rejects_bad_input() {
        assert!(decode_hex("0").is_err());
        assert!(decode_hex("gg").is_err());
    }

    #[test]
    fn tx_mode_numbers_match_libvpx() {
        for raw in 0..=4 {
            assert_eq!(parse_tx_mode(raw).unwrap() as u8, raw);
        }
        assert!(parse_tx_mode(5).is_err());
    }

    #[test]
    fn parse_input_requires_explicit_path() {
        let path =
            parse_input([std::ffi::OsString::from("--input"), "capture.jsonl".into()].into_iter())
                .unwrap();
        assert_eq!(path, PathBuf::from("capture.jsonl"));
    }
}
