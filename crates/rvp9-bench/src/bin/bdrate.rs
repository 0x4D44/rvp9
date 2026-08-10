//! BD-rate (Bjontegaard delta rate) measurement.
//!
//! Runs two encoders (`anchor` and `test`) at a target-bitrate sweep, decodes
//! each output, computes PSNR vs the source, builds two RD curves, and
//! reports BD-rate (the percentage bitrate difference between curves at
//! matched PSNR).
//!
//! In Stage 0 this runs `libvpx vpxenc` as both anchor and test. Expected
//! BD-rate: ~0% (any non-zero is harness noise — informative as a baseline).
//! In Stage 4 we point `--test-encoder` at `rvp9enc` and require BD-rate
//! within the limits stated in `PLAN.md` §2.
//!
//! Stage 0 status: end-to-end functional. Encode sweep, vpxdec-driven Y4M
//! decode, mean luma PSNR vs the source Y4M, cubic Bjontegaard fit, BD-rate
//! reported. (PSNR machinery lives in `rvp9_bench::mean_y_psnr`.)
//!
//! Usage:
//!   cargo run --release --bin bdrate -- \
//!       --anchor-encoder D:/language/libvpx-stock/vpxenc.exe \
//!       --test-encoder   D:/language/libvpx-stock/vpxenc.exe \
//!       --decoder        D:/language/libvpx-stock/vpxdec.exe \
//!       --input          tests/corpus/parkrun_720p30.y4m \
//!       [--reference-input tests/corpus/parkrun_720p30-clean.y4m] \
//!       --cpu-used       7 \
//!       --bitrates       500,1000,1500,2000,3000,4500 \
//!       --fps            30/1 --threads 4 --tile-columns 2 --limit 60 \
//!       --end-usage     cbr --tune-content default \
//!       [--aq-mode 3] [--noise-sensitivity 1] \
//!       [--anchor-arg VALUE]... [--test-arg VALUE]... \
//!       [--csv           bdrate.csv]
//!
//! With `--csv <path>`, also emits a machine-readable CSV with the schema
//!   role,target_kbps,measured_kbps,psnr_db
//! plus a trailing `bdrate_pct,<v>` summary row. Designed for CI ingestion
//! so Stage 4a regressions can gate the build.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

struct Args {
    anchor_encoder: PathBuf,
    test_encoder: PathBuf,
    decoder: PathBuf,
    input: PathBuf,
    reference_input: PathBuf,
    cpu_used: u32,
    bitrates_kbps: Vec<u32>,
    end_usage: String,
    fps: String,
    threads: u32,
    tile_columns: u32,
    lag_in_frames: u32,
    tune_content: String,
    aq_mode: Option<u32>,
    noise_sensitivity: Option<u32>,
    anchor_args: Vec<String>,
    test_args: Vec<String>,
    limit: Option<u32>,
    /// Optional CSV output path for CI ingestion.
    csv: Option<PathBuf>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    parse_args_from(&argv)
}

fn parse_args_from(argv: &[String]) -> Args {
    let get = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1).cloned())
    };
    let get_all = |flag: &str| -> Vec<String> {
        let inline_prefix = format!("{flag}=");
        argv.iter()
            .enumerate()
            .filter_map(|(index, arg)| {
                if arg == flag {
                    argv.get(index + 1).cloned()
                } else {
                    arg.strip_prefix(&inline_prefix).map(str::to_owned)
                }
            })
            .collect()
    };

    let input = get("--input").map(PathBuf::from).expect("--input required");
    let reference_input = get("--reference-input")
        .map(PathBuf::from)
        .unwrap_or_else(|| input.clone());

    Args {
        anchor_encoder: get("--anchor-encoder")
            .map(PathBuf::from)
            .expect("--anchor-encoder required"),
        test_encoder: get("--test-encoder")
            .map(PathBuf::from)
            .expect("--test-encoder required"),
        decoder: get("--decoder")
            .map(PathBuf::from)
            .expect("--decoder required"),
        input,
        reference_input,
        cpu_used: get("--cpu-used")
            .map(|s| s.parse().expect("--cpu-used u32"))
            .unwrap_or(7),
        bitrates_kbps: get("--bitrates")
            .map(|s| {
                s.split(',')
                    .map(|t| t.parse::<u32>().expect("bitrate u32"))
                    .collect()
            })
            .unwrap_or_else(|| vec![500, 1000, 1500, 2000, 3000, 4500]),
        end_usage: get("--end-usage").unwrap_or_else(|| "cbr".to_owned()),
        fps: get("--fps").unwrap_or_else(|| "30/1".to_owned()),
        threads: get("--threads")
            .map(|s| s.parse().expect("--threads u32"))
            .unwrap_or(1),
        tile_columns: get("--tile-columns")
            .map(|s| s.parse().expect("--tile-columns u32"))
            .unwrap_or(0),
        lag_in_frames: get("--lag-in-frames")
            .map(|s| s.parse().expect("--lag-in-frames u32"))
            .unwrap_or(0),
        tune_content: get("--tune-content").unwrap_or_else(|| "default".to_owned()),
        aq_mode: get("--aq-mode").map(|s| s.parse().expect("--aq-mode u32")),
        noise_sensitivity: get("--noise-sensitivity")
            .map(|s| s.parse().expect("--noise-sensitivity u32")),
        anchor_args: get_all("--anchor-arg"),
        test_args: get_all("--test-arg"),
        limit: get("--limit").map(|s| s.parse().expect("--limit u32")),
        csv: get("--csv").map(PathBuf::from),
    }
}

#[derive(Debug, Clone, Copy)]
struct RdPoint {
    /// Realised bitrate in kbps (from the IVF, not the requested target).
    bitrate_kbps: f64,
    /// PSNR vs source in dB.
    psnr_db: f64,
}

struct EncodeConfig<'a> {
    end_usage: &'a str,
    fps: &'a str,
    threads: u32,
    tile_columns: u32,
    lag_in_frames: u32,
    tune_content: &'a str,
    aq_mode: Option<u32>,
    noise_sensitivity: Option<u32>,
    limit: Option<u32>,
}

fn encoder_command(
    encoder: &Path,
    input: &Path,
    output: &Path,
    cpu_used: u32,
    bitrate_kbps: u32,
    config: &EncodeConfig<'_>,
    extra_args: &[String],
) -> Command {
    let mut command = Command::new(encoder);
    command.args([
        "--codec=vp9".to_owned(),
        "--passes=1".to_owned(),
        "--rt".to_owned(),
        format!("--cpu-used={cpu_used}"),
        format!("--end-usage={}", config.end_usage),
        format!("--target-bitrate={bitrate_kbps}"),
        format!("--fps={}", config.fps),
        format!("--threads={}", config.threads),
        format!("--tile-columns={}", config.tile_columns),
        format!("--lag-in-frames={}", config.lag_in_frames),
        format!("--tune-content={}", config.tune_content),
        "--quiet".to_owned(),
        "--ivf".to_owned(),
    ]);
    if let Some(aq_mode) = config.aq_mode {
        command.arg(format!("--aq-mode={aq_mode}"));
    }
    if let Some(noise_sensitivity) = config.noise_sensitivity {
        command.arg(format!("--noise-sensitivity={noise_sensitivity}"));
    }
    if let Some(limit) = config.limit {
        command.arg(format!("--limit={limit}"));
    }
    command.args(extra_args).arg("-o").arg(output).arg(input);
    command
}

/// Encode `input` with `encoder` at the given target bitrate. Returns the
/// path to the resulting IVF.
fn encode(
    encoder: &Path,
    input: &Path,
    cpu_used: u32,
    bitrate_kbps: u32,
    tag: &str,
    config: &EncodeConfig<'_>,
    extra_args: &[String],
) -> PathBuf {
    let out = std::env::temp_dir().join(format!("rvp9_bdrate_{tag}_{bitrate_kbps}k.ivf"));
    let status = encoder_command(
        encoder,
        input,
        &out,
        cpu_used,
        bitrate_kbps,
        config,
        extra_args,
    )
    .status()
    .expect("encoder failed");
    assert!(
        status.success(),
        "encoder {} failed at {bitrate_kbps}k",
        encoder.display()
    );
    out
}

/// Decode `ivf` with `decoder` (writing a tmp Y4M), then compute the
/// mean luma PSNR vs `source_y4m`. Returns `NaN` on any failure so the
/// caller's NaN-guard can detect harness problems without aborting.
fn decode_and_psnr(decoder: &Path, ivf: &Path, source_y4m: &Path) -> f64 {
    let stem = ivf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("rvp9_bdrate_decode");
    let y4m_out = std::env::temp_dir().join(format!("{stem}.dec.y4m"));
    let status = Command::new(decoder)
        .arg("-o")
        .arg(&y4m_out)
        .arg(ivf)
        .status();
    let Ok(status) = status else {
        return f64::NAN;
    };
    if !status.success() {
        return f64::NAN;
    }
    rvp9_bench::mean_y_psnr(source_y4m, &y4m_out).unwrap_or(f64::NAN)
}

/// File size of `ivf` in kilobits, divided by clip duration in seconds.
fn measured_bitrate_kbps(ivf: &Path, frames: u32, fps: f64) -> f64 {
    let bytes = std::fs::metadata(ivf).expect("ivf metadata").len() as f64;
    let duration_s = f64::from(frames) / fps;
    (bytes * 8.0 / 1000.0) / duration_s
}

/// Bjontegaard delta rate, cubic polynomial fit on log-bitrate vs PSNR.
/// Returns percent (positive = test costs more bits for same quality).
///
/// Reference: Bjontegaard, "Calculation of Average PSNR Differences between
/// RD-curves", ITU-T VCEG Q6/16, Doc. VCEG-M33, 2001.
fn bd_rate(anchor: &[RdPoint], test: &[RdPoint]) -> Option<f64> {
    if anchor.len() < 4 || test.len() < 4 {
        return None;
    }
    // Convert bitrate to log-scale for the fit.
    let to_xy = |p: &RdPoint| (p.psnr_db, p.bitrate_kbps.ln());
    let anchor_xy: Vec<(f64, f64)> = anchor.iter().map(to_xy).collect();
    let test_xy: Vec<(f64, f64)> = test.iter().map(to_xy).collect();

    // Cubic least-squares fit of log_bitrate as a function of PSNR.
    let a = polyfit3(&anchor_xy);
    let b = polyfit3(&test_xy);

    // Integration range: the OVERLAP of the two PSNR ranges (Bjontegaard
    // integrates only over PSNR values where both curves are defined).
    let anchor_min = anchor_xy.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let anchor_max = anchor_xy
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let test_min = test_xy.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let test_max = test_xy
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let range_lo = anchor_min.max(test_min);
    let range_hi = anchor_max.min(test_max);
    if range_hi <= range_lo {
        return None;
    }

    // Definite integral of a cubic on [lo, hi].
    let integrate = |c: &[f64; 4]| -> f64 {
        let f = |x: f64| {
            c[0] * x + c[1] * x * x / 2.0 + c[2] * x * x * x / 3.0 + c[3] * x * x * x * x / 4.0
        };
        f(range_hi) - f(range_lo)
    };
    let int_a = integrate(&a);
    let int_b = integrate(&b);
    let avg_diff = (int_b - int_a) / (range_hi - range_lo);
    Some((avg_diff.exp() - 1.0) * 100.0)
}

fn measured_bd_rate(anchor: &[RdPoint], test: &[RdPoint]) -> Result<f64, &'static str> {
    if anchor.iter().any(|p| !p.psnr_db.is_finite()) || test.iter().any(|p| !p.psnr_db.is_finite())
    {
        return Err("PSNR measurement failed");
    }
    let value = bd_rate(anchor, test).ok_or("insufficient or non-overlapping RD points")?;
    if !value.is_finite() {
        return Err("BD-rate calculation produced a non-finite result");
    }
    Ok(value)
}

/// Least-squares cubic fit y = c0 + c1*x + c2*x^2 + c3*x^3.
/// Uses the normal equations (4x4 system) with Gauss-Jordan elimination.
///
/// The index-based loops below are deliberate — the matrix-elimination
/// invariants are clearer with explicit `m[i][j]` than with iterator chains
/// over rows. We allow `clippy::needless_range_loop` for that reason.
#[allow(clippy::needless_range_loop)]
fn polyfit3(pts: &[(f64, f64)]) -> [f64; 4] {
    // Build the normal-equations augmented matrix [A | b] where
    //   A[i][j] = sum(x^(i+j))
    //   b[i]    = sum(y * x^i)
    let mut m = [[0.0f64; 5]; 4];
    for &(x, y) in pts {
        let mut xp = [1.0; 7];
        for i in 1..7 {
            xp[i] = xp[i - 1] * x;
        }
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] += xp[i + j];
            }
            m[i][4] += y * xp[i];
        }
    }
    // Gauss-Jordan.
    for i in 0..4 {
        // Pivot.
        let mut max = i;
        for k in (i + 1)..4 {
            if m[k][i].abs() > m[max][i].abs() {
                max = k;
            }
        }
        m.swap(i, max);
        let pivot = m[i][i];
        if pivot.abs() < 1e-12 {
            return [0.0; 4]; // degenerate; caller should ignore.
        }
        for j in 0..5 {
            m[i][j] /= pivot;
        }
        for k in 0..4 {
            if k != i {
                let f = m[k][i];
                for j in 0..5 {
                    m[k][j] -= f * m[i][j];
                }
            }
        }
    }
    [m[0][4], m[1][4], m[2][4], m[3][4]]
}

fn main() -> ExitCode {
    let args = parse_args();

    println!("=== BD-rate sweep ===");
    println!("anchor: {}", args.anchor_encoder.display());
    println!("test:   {}", args.test_encoder.display());
    println!("input:  {}", args.input.display());
    println!("reference input: {}", args.reference_input.display());
    println!(
        "config: --rt --cpu-used={} --end-usage={} --fps={} --threads={} --tile-columns={} --lag-in-frames={} --tune-content={}{}{}{}",
        args.cpu_used,
        args.end_usage,
        args.fps,
        args.threads,
        args.tile_columns,
        args.lag_in_frames,
        args.tune_content,
        args.aq_mode
            .map_or(String::new(), |value| format!(" --aq-mode={value}")),
        args.noise_sensitivity.map_or(String::new(), |value| {
            format!(" --noise-sensitivity={value}")
        }),
        args.limit
            .map_or(String::new(), |limit| format!(" --limit={limit}")),
    );
    if !args.anchor_args.is_empty() {
        println!("anchor args: {}", args.anchor_args.join(" "));
    }
    if !args.test_args.is_empty() {
        println!("test args:   {}", args.test_args.join(" "));
    }
    println!();

    // Parse fps from the source Y4M header so realised-bitrate math is
    // accurate for non-30fps clips. Fall back to 30.0 if the header
    // can't be read for some reason.
    let fps_hz = std::fs::read(&args.input)
        .ok()
        .and_then(|d| rvp9_bench::parse_y4m_header(&d).map(|(h, _)| h.fps()))
        .unwrap_or(30.0);
    let encode_config = EncodeConfig {
        end_usage: &args.end_usage,
        fps: &args.fps,
        threads: args.threads,
        tile_columns: args.tile_columns,
        lag_in_frames: args.lag_in_frames,
        tune_content: &args.tune_content,
        aq_mode: args.aq_mode,
        noise_sensitivity: args.noise_sensitivity,
        limit: args.limit,
    };

    let mut anchor_rd: Vec<RdPoint> = Vec::new();
    let mut test_rd: Vec<RdPoint> = Vec::new();

    for &br in &args.bitrates_kbps {
        let a_ivf = encode(
            &args.anchor_encoder,
            &args.input,
            args.cpu_used,
            br,
            "anchor",
            &encode_config,
            &args.anchor_args,
        );
        let t_ivf = encode(
            &args.test_encoder,
            &args.input,
            args.cpu_used,
            br,
            "test",
            &encode_config,
            &args.test_args,
        );

        let a_frames = rvp9_bench::count_ivf_frames(&a_ivf);
        let t_frames = rvp9_bench::count_ivf_frames(&t_ivf);

        let a_br = measured_bitrate_kbps(&a_ivf, a_frames, fps_hz);
        let t_br = measured_bitrate_kbps(&t_ivf, t_frames, fps_hz);

        let a_psnr = decode_and_psnr(&args.decoder, &a_ivf, &args.reference_input);
        let t_psnr = decode_and_psnr(&args.decoder, &t_ivf, &args.reference_input);

        println!(
            "target {br:>5}k | anchor {a_br:>6.0}k psnr {a_psnr:>5.2} | test {t_br:>6.0}k psnr {t_psnr:>5.2}"
        );

        anchor_rd.push(RdPoint {
            bitrate_kbps: a_br,
            psnr_db: a_psnr,
        });
        test_rd.push(RdPoint {
            bitrate_kbps: t_br,
            psnr_db: t_psnr,
        });
    }

    let bd = match measured_bd_rate(&anchor_rd, &test_rd) {
        Ok(value) => {
            println!("\nBD-rate: {value:+.3}%  (anchor → test)");
            value
        }
        Err(error) => {
            eprintln!("\nBD-rate failed: {error}");
            return ExitCode::FAILURE;
        }
    };

    // Optional CSV output. Schema is stable so CI can ingest:
    //   role,target_kbps,measured_kbps,psnr_db
    // then a trailing one-row summary:
    //   bdrate_pct,<value>
    if let Some(csv_path) = &args.csv {
        if let Err(e) = write_csv(
            csv_path,
            &args.bitrates_kbps,
            &anchor_rd,
            &test_rd,
            Some(bd),
        ) {
            eprintln!("error: --csv write failed: {e}");
            return ExitCode::FAILURE;
        } else {
            println!("\ncsv: {}", csv_path.display());
        }
    }
    ExitCode::SUCCESS
}

fn write_csv(
    path: &Path,
    targets: &[u32],
    anchor: &[RdPoint],
    test: &[RdPoint],
    bdrate_pct: Option<f64>,
) -> std::io::Result<()> {
    use std::io::Write;
    let bdrate_pct = bdrate_pct
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "finite BD-rate measurement required",
            )
        })?;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "role,target_kbps,measured_kbps,psnr_db")?;
    for (i, &t) in targets.iter().enumerate() {
        if let Some(p) = anchor.get(i) {
            writeln!(f, "anchor,{t},{:.3},{:.4}", p.bitrate_kbps, p.psnr_db)?;
        }
        if let Some(p) = test.get(i) {
            writeln!(f, "test,{t},{:.3},{:.4}", p.bitrate_kbps, p.psnr_db)?;
        }
    }
    writeln!(f, "bdrate_pct,{bdrate_pct:.4}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_preserves_repeated_role_specific_arguments() {
        let argv = [
            "bdrate",
            "--anchor-encoder",
            "anchor",
            "--test-encoder",
            "test",
            "--decoder",
            "decoder",
            "--input",
            "input.y4m",
            "--anchor-arg",
            "--row-mt=1",
            "--test-arg",
            "--enable-tpl=0",
            "--anchor-arg",
            "--static-thresh=1",
        ]
        .map(str::to_owned);

        let args = parse_args_from(&argv);

        assert_eq!(args.anchor_args, ["--row-mt=1", "--static-thresh=1"]);
        assert_eq!(args.test_args, ["--enable-tpl=0"]);
        assert_eq!(args.reference_input, PathBuf::from("input.y4m"));
    }

    #[test]
    fn parser_accepts_inline_role_specific_arguments() {
        let argv = [
            "bdrate",
            "--anchor-encoder",
            "anchor",
            "--test-encoder",
            "test",
            "--decoder",
            "decoder",
            "--input",
            "input.y4m",
            "--anchor-arg=--noise-sensitivity=0",
            "--test-arg=--noise-sensitivity=1",
        ]
        .map(str::to_owned);

        let args = parse_args_from(&argv);

        assert_eq!(args.anchor_args, ["--noise-sensitivity=0"]);
        assert_eq!(args.test_args, ["--noise-sensitivity=1"]);
    }

    #[test]
    fn parser_accepts_a_separate_quality_reference_input() {
        let argv = [
            "bdrate",
            "--anchor-encoder",
            "anchor",
            "--test-encoder",
            "test",
            "--decoder",
            "decoder",
            "--input",
            "noisy.y4m",
            "--reference-input",
            "clean.y4m",
        ]
        .map(str::to_owned);

        let args = parse_args_from(&argv);

        assert_eq!(args.input, PathBuf::from("noisy.y4m"));
        assert_eq!(args.reference_input, PathBuf::from("clean.y4m"));
    }

    #[test]
    fn encoder_command_keeps_matched_and_role_specific_arguments_separate() {
        let config = EncodeConfig {
            end_usage: "cbr",
            fps: "30/1",
            threads: 4,
            tile_columns: 2,
            lag_in_frames: 0,
            tune_content: "default",
            aq_mode: Some(3),
            noise_sensitivity: Some(1),
            limit: Some(60),
        };
        let anchor_args = vec!["--row-mt=1".to_owned(), "--static-thresh=1".to_owned()];
        let command = encoder_command(
            Path::new("anchor"),
            Path::new("input.y4m"),
            Path::new("output.ivf"),
            7,
            1_500,
            &config,
            &anchor_args,
        );
        let args: Vec<String> = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();

        assert!(args.contains(&"--aq-mode=3".to_owned()));
        assert!(args.contains(&"--noise-sensitivity=1".to_owned()));
        assert!(args.contains(&"--limit=60".to_owned()));
        let output_index = args.iter().position(|arg| arg == "-o").unwrap();
        assert_eq!(
            &args[output_index - 2..output_index],
            ["--row-mt=1", "--static-thresh=1"]
        );
        assert_eq!(&args[output_index..], ["-o", "output.ivf", "input.y4m"]);
    }

    #[test]
    fn csv_writer_rejects_missing_bdrate_without_creating_output() {
        let path = std::env::temp_dir().join(format!(
            "rvp9-bdrate-missing-summary-{}.csv",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        let error = write_csv(&path, &[], &[], &[], None).unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(!path.exists(), "failed measurement created a CSV");
    }

    #[test]
    fn measured_bdrate_rejects_failed_psnr_and_insufficient_points() {
        let failed = [RdPoint {
            bitrate_kbps: 100.0,
            psnr_db: f64::NAN,
        }];
        assert_eq!(
            measured_bd_rate(&failed, &failed),
            Err("PSNR measurement failed")
        );

        let single = [RdPoint {
            bitrate_kbps: 100.0,
            psnr_db: 30.0,
        }];
        assert_eq!(
            measured_bd_rate(&single, &single),
            Err("insufficient or non-overlapping RD points")
        );
    }
}
