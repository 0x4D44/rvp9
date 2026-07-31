//! Replay exact libvpx `block_yrd` coefficient arrays through rvp9.
//!
//! The corresponding pinned-libvpx capture is enabled with
//! `VPX_F02_BLOCK_YRD_ORACLE_OUT=<path>`. Only non-skippable calls are
//! sampled, so the comparison covers the pure coefficient aggregate and
//! leaves libvpx's caller-side all-zero override out of scope.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use rvp9_core::mode_decision::block_yrd::{BlockYrdResult, block_yrd};
use rvp9_core::transforms::TxSize;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Sample {
    sample: u64,
    frame_idx: u64,
    bsize: u8,
    tx_size: u8,
    tile_count: usize,
    rate: u64,
    dist: i64,
    skippable: bool,
    coeff: Vec<i32>,
    qcoeff: Vec<i32>,
    dqcoeff: Vec<i32>,
    eobs: Vec<u16>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("block-yrd-oracle: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let input = parse_input(std::env::args_os().skip(1))?;
    let file =
        File::open(&input).map_err(|error| format!("cannot open {}: {error}", input.display()))?;
    let mut samples = 0usize;
    let mut exact = 0usize;
    let mut malformed = 0usize;
    let mut rate = 0usize;
    let mut dist = 0usize;
    let mut skippable = 0usize;

    for (line_idx, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|error| format!("{}:{}: {error}", input.display(), line_idx + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        samples += 1;
        match replay_line(&line) {
            Ok((sample, actual)) => {
                let mut differences = Vec::new();
                if actual.coef_rate != sample.rate {
                    rate += 1;
                    differences.push(format!("rate {} != {}", actual.coef_rate, sample.rate));
                }
                if actual.coef_dist != sample.dist {
                    dist += 1;
                    differences.push(format!("dist {} != {}", actual.coef_dist, sample.dist));
                }
                if actual.skippable != sample.skippable {
                    skippable += 1;
                    differences.push(format!(
                        "skippable {} != {}",
                        actual.skippable, sample.skippable
                    ));
                }
                if differences.is_empty() {
                    exact += 1;
                } else if samples - exact - malformed <= 12 {
                    eprintln!(
                        "sample {} frame {} bsize {} tx {} tiles {}: {}",
                        sample.sample,
                        sample.frame_idx,
                        sample.bsize,
                        sample.tx_size,
                        sample.tile_count,
                        differences.join("; ")
                    );
                }
            }
            Err(error) => {
                malformed += 1;
                if malformed <= 12 {
                    eprintln!("line {}: {error}", line_idx + 1);
                }
            }
        }
    }

    println!(
        "samples={samples} exact={exact} malformed={malformed} \
         rate_mismatch={rate} dist_mismatch={dist} skippable_mismatch={skippable}"
    );
    if samples == 0 {
        return Err("input contained no samples".into());
    }
    if exact != samples {
        return Err(format!("{} of {samples} samples differed", samples - exact));
    }
    Ok(())
}

fn parse_input(args: impl Iterator<Item = std::ffi::OsString>) -> Result<PathBuf, String> {
    let args: Vec<_> = args.collect();
    match args.as_slice() {
        [flag, path] if flag == "--input" => Ok(PathBuf::from(path)),
        _ => Err("usage: block-yrd-oracle --input <libvpx.jsonl>".into()),
    }
}

fn replay_line(line: &str) -> Result<(Sample, BlockYrdResult), String> {
    let sample: Sample =
        serde_json::from_str(line).map_err(|error| format!("invalid JSON: {error}"))?;
    let tx_size = parse_tx_size(sample.tx_size)?;
    let n = tx_size.n_coeffs();
    let expected = sample
        .tile_count
        .checked_mul(n)
        .ok_or("tile coefficient count overflow")?;
    if sample.coeff.len() != expected
        || sample.qcoeff.len() != expected
        || sample.dqcoeff.len() != expected
        || sample.eobs.len() != sample.tile_count
    {
        return Err(format!(
            "array lengths coeff={} q={} dq={} eobs={} expected={expected}/{}",
            sample.coeff.len(),
            sample.qcoeff.len(),
            sample.dqcoeff.len(),
            sample.eobs.len(),
            sample.tile_count
        ));
    }
    let qcoeff = narrow_coefficients(&sample.qcoeff, "qcoeff")?;
    let dqcoeff = narrow_coefficients(&sample.dqcoeff, "dqcoeff")?;
    let coeff_tiles: Vec<&[i32]> = sample.coeff.chunks_exact(n).collect();
    let qcoeff_tiles: Vec<&[i16]> = qcoeff.chunks_exact(n).collect();
    let dqcoeff_tiles: Vec<&[i16]> = dqcoeff.chunks_exact(n).collect();
    let actual = block_yrd(
        &coeff_tiles,
        &qcoeff_tiles,
        &dqcoeff_tiles,
        &sample.eobs,
        tx_size,
    );
    Ok((sample, actual))
}

fn parse_tx_size(raw: u8) -> Result<TxSize, String> {
    match raw {
        0 => Ok(TxSize::Tx4x4),
        1 => Ok(TxSize::Tx8x8),
        2 => Ok(TxSize::Tx16x16),
        3 => Ok(TxSize::Tx32x32),
        _ => Err(format!("invalid tx_size {raw}")),
    }
}

fn narrow_coefficients(values: &[i32], field: &str) -> Result<Vec<i16>, String> {
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            i16::try_from(value)
                .map_err(|_| format!("{field}[{index}]={value} does not fit in i16"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tx_size_numbers_match_libvpx() {
        for raw in 0..=3 {
            assert_eq!(parse_tx_size(raw).unwrap() as u8, raw);
        }
        assert!(parse_tx_size(4).is_err());
    }

    #[test]
    fn narrowing_rejects_out_of_range_coefficients() {
        assert_eq!(
            narrow_coefficients(&[-32_768, 0, 32_767], "q").unwrap(),
            [-32_768, 0, 32_767]
        );
        assert!(narrow_coefficients(&[32_768], "q").is_err());
    }

    #[test]
    fn parse_input_requires_explicit_path() {
        let path =
            parse_input([std::ffi::OsString::from("--input"), "capture.jsonl".into()].into_iter())
                .unwrap();
        assert_eq!(path, PathBuf::from("capture.jsonl"));
    }
}
