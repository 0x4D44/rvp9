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
//!       [--work-dir      run/artifacts] \
//!       [--csv           bdrate.csv]
//!
//! With `--work-dir <dir>`, every IVF this run encodes and every Y4M it
//! decodes is written beneath `<dir>` under a deterministic
//! `<role>_<bitrate>k` name (`anchor_1500k.ivf`, `anchor_1500k.dec.y4m`, …)
//! and is retained after the run, so a caller can cross-decode the exact
//! bitstreams that were measured. The directory is created if missing;
//! nothing already in it is read or removed. Without the flag the artifacts
//! land in the system temp directory under a process-unique prefix, so
//! concurrent runs cannot overwrite each other's encodes.
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
    /// Optional caller-owned directory that retains this run's encoded IVF
    /// and decoded Y4M artifacts.
    work_dir: Option<PathBuf>,
    /// Optional CSV output path for CI ingestion.
    csv: Option<PathBuf>,
}

const USAGE: &str = "Usage: bdrate --anchor-encoder PATH --test-encoder PATH --decoder PATH \\
  --input INPUT.y4m [--reference-input CLEAN.y4m] [options]\n\n\
Options:\n\
  --cpu-used N --bitrates LIST --end-usage cbr|vbr --fps NUM/DEN\n\
  --threads N --tile-columns N --lag-in-frames N --tune-content NAME\n\
  --aq-mode N --noise-sensitivity N --limit N\n\
  --anchor-arg ARG --test-arg ARG\n\
  --work-dir PATH   retain deterministic anchor/test IVF and decoded Y4M files\n\
  --csv PATH        write machine-readable measurements\n";

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
        work_dir: get("--work-dir").map(PathBuf::from),
        csv: get("--csv").map(PathBuf::from),
    }
}

/// Where this run writes its generated IVF and decoded Y4M files, and how it
/// names them.
///
/// With `--work-dir` the caller owns `dir`, the prefix is empty, and every
/// artifact keeps a deterministic `<role>_<bitrate>k` name so the files stay
/// identifiable after the run. Without it `dir` is the system temp directory
/// and the prefix carries this process's id, so two concurrent runs cannot
/// encode over each other.
#[derive(Debug)]
struct Artifacts {
    dir: PathBuf,
    prefix: String,
}

impl Artifacts {
    /// Path of the IVF for one role (`anchor` / `test`) at one target bitrate.
    fn ivf(&self, role: &str, bitrate_kbps: u32) -> PathBuf {
        let name = format!("{}{role}_{bitrate_kbps}k.ivf", self.prefix);
        self.dir.join(name)
    }
}

/// Resolve the artifact location, creating a requested `--work-dir` if it is
/// absent. Errors carry the directory so a failure names what to fix.
fn prepare_artifacts(work_dir: Option<&Path>) -> Result<Artifacts, String> {
    let Some(dir) = work_dir else {
        return Ok(Artifacts {
            dir: std::env::temp_dir(),
            prefix: format!("rvp9_bdrate_{}_", std::process::id()),
        });
    };
    std::fs::create_dir_all(dir)
        .map_err(|error| format!("--work-dir {}: {error}", dir.display()))?;
    check_writable(dir)?;
    Ok(Artifacts {
        dir: dir.to_path_buf(),
        prefix: String::new(),
    })
}

/// Reject an unwritable work directory up front, rather than failing deep
/// inside the first encoder invocation minutes later.
fn check_writable(dir: &Path) -> Result<(), String> {
    let probe = dir.join(format!(".rvp9-bdrate-probe-{}", std::process::id()));
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
    {
        // Only ever remove the probe this process just created. Treat a
        // cleanup failure as an unusable directory rather than leaving a
        // hidden artifact in caller-owned storage.
        Ok(file) => {
            drop(file);
            std::fs::remove_file(&probe).map_err(|error| {
                format!(
                    "--work-dir {} write probe cleanup failed: {error}",
                    dir.display()
                )
            })
        }
        Err(error) => Err(format!(
            "--work-dir {} is not writable: {error}",
            dir.display()
        )),
    }
}

/// Decoded Y4M for `ivf`, kept beside it so `--work-dir` holds the whole set.
fn decoded_y4m_path(ivf: &Path) -> PathBuf {
    let stem = ivf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("rvp9_bdrate_decode");
    let name = format!("{stem}.dec.y4m");
    match ivf.parent() {
        Some(dir) => dir.join(name),
        None => PathBuf::from(name),
    }
}

/// Refuse to overwrite artifacts retained by an earlier or concurrent run.
fn reject_existing_artifacts(paths: &[&Path]) -> Result<(), String> {
    if let Some(path) = paths.iter().find(|path| path.exists()) {
        return Err(format!(
            "artifact {} already exists; choose an empty --work-dir",
            path.display()
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct RdPoint {
    /// Realised bitrate in kbps (from the IVF, not the requested target).
    bitrate_kbps: f64,
    /// PSNR vs source in dB.
    psnr_db: f64,
}

struct EncodeConfig<'a> {
    timeout: std::time::Duration,
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

/// Encode `input` with `encoder` at the given target bitrate, writing the IVF
/// to `out`. The file is left in place for the caller to measure and reuse.
fn encode(
    encoder: &Path,
    input: &Path,
    cpu_used: u32,
    bitrate_kbps: u32,
    out: &Path,
    config: &EncodeConfig<'_>,
    extra_args: &[String],
) {
    let status = rvp9_bench::process::run_status(
        &mut encoder_command(
            encoder,
            input,
            out,
            cpu_used,
            bitrate_kbps,
            config,
            extra_args,
        ),
        "BD-rate encoder",
        config.timeout,
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert!(
        status.success(),
        "encoder {} failed at {bitrate_kbps}k",
        encoder.display()
    );
}

/// Decode `ivf` with `decoder` (writing the Y4M beside it), then compute the
/// mean luma PSNR vs `source_y4m`. Returns `NaN` on any failure so the
/// caller's NaN-guard can detect harness problems without aborting.
fn decode_and_psnr(decoder: &Path, ivf: &Path, source_y4m: &Path) -> f64 {
    decode_and_psnr_with_timeout(
        decoder,
        ivf,
        source_y4m,
        rvp9_bench::process::DEFAULT_TIMEOUT,
    )
}

fn decode_and_psnr_with_timeout(
    decoder: &Path,
    ivf: &Path,
    source_y4m: &Path,
    timeout: std::time::Duration,
) -> f64 {
    let y4m_out = decoded_y4m_path(ivf);
    let status = rvp9_bench::process::run_status(
        Command::new(decoder).arg("-o").arg(&y4m_out).arg(ivf),
        "BD-rate decoder",
        timeout,
    );
    let status = match status {
        Ok(status) => status,
        Err(error) => {
            eprintln!("{error}");
            return f64::NAN;
        }
    };
    if !status.success() {
        return f64::NAN;
    }
    rvp9_bench::mean_y_psnr(source_y4m, &y4m_out).unwrap_or(f64::NAN)
}

/// Match the last FPS argument actually passed to this encoder role.
fn effective_frame_rate(default: &str, extra_args: &[String]) -> Result<(u32, u32), String> {
    let mut fps = default;
    let mut args = extra_args.iter();
    while let Some(arg) = args.next() {
        if arg == "--fps" {
            fps = args
                .next()
                .ok_or("effective encoder framerate: --fps requires a value")?;
        } else if let Some(value) = arg.strip_prefix("--fps=") {
            fps = value;
        }
    }
    let parsed = fps
        .split_once('/')
        .and_then(|(num, den)| Some((num.parse::<u32>().ok()?, den.parse::<u32>().ok()?)));
    match parsed {
        Some((num, den)) if num != 0 && den != 0 => Ok((num, den)),
        _ => Err(format!("invalid effective encoder framerate: {fps}")),
    }
}

/// Measure a CFR encode using its effective encoder FPS contract. An IVF clock
/// describes timestamp ticks, not frames: libvpx commonly uses 1000 ticks/s.
/// Validate every packet timestamp against floor/ceil tick quantization of the
/// contract before trusting it. The final frame lasts one full CFR period,
/// including for single-frame clips and fractional rates.
fn measured_bitrate_kbps(ivf: &Path, frames: u32, fps: (u32, u32)) -> Result<f64, String> {
    let data = std::fs::read(ivf)
        .map_err(|error| format!("could not read IVF {}: {error}", ivf.display()))?;
    let packets = rvp9_harness::container::read_ivf_packets(&data)
        .map_err(|error| format!("invalid IVF {}: {error}", ivf.display()))?;
    let clock_rate = u32::from_le_bytes(data[16..20].try_into().expect("validated IVF header"));
    let clock_scale = u32::from_le_bytes(data[20..24].try_into().expect("validated IVF header"));
    if clock_rate == 0 || clock_scale == 0 {
        return Err(format!(
            "IVF {} has a missing or zero framerate in its timebase",
            ivf.display()
        ));
    }
    if frames == 0 || packets.len() != frames as usize {
        return Err(format!(
            "IVF {} has no frames or an inconsistent packet count",
            ivf.display()
        ));
    }
    if fps.0 == 0 || fps.1 == 0 {
        return Err("effective encoder framerate must be nonzero".to_owned());
    }
    let mut first_pts = None;
    let mut previous_pts = 0;
    for (index, packet) in packets.iter().enumerate() {
        let pts = u64::from_le_bytes(
            data[packet.start - 8..packet.start]
                .try_into()
                .expect("validated IVF packet"),
        );
        let first = *first_pts.get_or_insert(pts);
        let elapsed = pts
            .checked_sub(first)
            .ok_or_else(|| format!("IVF {} timestamp precedes its first packet", ivf.display()))?;
        // u128 keeps all products exact even at u32 FPS/clock limits. Repeated
        // timestamps are legitimate when the clock is coarser than one frame.
        let numerator = index as u128 * u128::from(clock_rate) * u128::from(fps.1);
        let denominator = u128::from(clock_scale) * u128::from(fps.0);
        let floor = numerator / denominator;
        let ceil = numerator.div_ceil(denominator);
        if pts < previous_pts || !(floor..=ceil).contains(&u128::from(elapsed)) {
            return Err(format!(
                "IVF {} timestamp at packet {index} is inconsistent with effective encoder framerate {}/{} (relative ticks {elapsed}, expected {floor}..={ceil})",
                ivf.display(),
                fps.0,
                fps.1
            ));
        }
        previous_pts = pts;
    }
    let duration_s = f64::from(frames) * f64::from(fps.1) / f64::from(fps.0);
    Ok(data.len() as f64 * 8.0 / 1000.0 / duration_s)
}

/// Bjontegaard delta rate, cubic polynomial fit on log-bitrate vs PSNR.
/// Returns percent (positive = test costs more bits for same quality).
///
/// Reference: Bjontegaard, "Calculation of Average PSNR Differences between
/// RD-curves", ITU-T VCEG Q6/16, Doc. VCEG-M33, 2001.
fn bd_rate(anchor: &[RdPoint], test: &[RdPoint]) -> Option<f64> {
    let anchor_curve: Vec<(f64, f64)> = anchor
        .iter()
        .map(|point| (point.bitrate_kbps, point.psnr_db))
        .collect();
    let test_curve: Vec<(f64, f64)> = test
        .iter()
        .map(|point| (point.bitrate_kbps, point.psnr_db))
        .collect();
    let result = rvp9_harness::bdrate::bdrate(&anchor_curve, &test_curve);
    result.is_finite().then_some(result)
}

fn measured_bd_rate(anchor: &[RdPoint], test: &[RdPoint]) -> Result<f64, &'static str> {
    if anchor.iter().any(|p| !p.psnr_db.is_finite()) || test.iter().any(|p| !p.psnr_db.is_finite())
    {
        return Err("PSNR measurement failed");
    }
    let value =
        bd_rate(anchor, test).ok_or("insufficient, singular, or non-overlapping RD points")?;
    if !value.is_finite() {
        return Err("BD-rate calculation produced a non-finite result");
    }
    Ok(value)
}

fn main() -> ExitCode {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let args = parse_args();

    let artifacts = match prepare_artifacts(args.work_dir.as_deref()) {
        Ok(artifacts) => artifacts,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };

    println!("=== BD-rate sweep ===");
    println!("anchor: {}", args.anchor_encoder.display());
    println!("test:   {}", args.test_encoder.display());
    println!("input:  {}", args.input.display());
    println!("reference input: {}", args.reference_input.display());
    println!("artifacts: {}", artifacts.dir.display());
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
        timeout: rvp9_bench::process::DEFAULT_TIMEOUT,
    };

    let role_fps = |role, extra: &[String]| {
        effective_frame_rate(&args.fps, extra).map_err(|error| format!("{role}: {error}"))
    };
    let (anchor_fps, test_fps) = match (
        role_fps("anchor", &args.anchor_args),
        role_fps("test", &args.test_args),
    ) {
        (Ok(anchor), Ok(test)) => (anchor, test),
        (Err(error), _) | (_, Err(error)) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };

    let mut anchor_rd: Vec<RdPoint> = Vec::new();
    let mut test_rd: Vec<RdPoint> = Vec::new();

    for &br in &args.bitrates_kbps {
        let a_ivf = artifacts.ivf("anchor", br);
        let t_ivf = artifacts.ivf("test", br);
        let a_y4m = decoded_y4m_path(&a_ivf);
        let t_y4m = decoded_y4m_path(&t_ivf);
        if let Err(error) = reject_existing_artifacts(&[&a_ivf, &t_ivf, &a_y4m, &t_y4m]) {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
        encode(
            &args.anchor_encoder,
            &args.input,
            args.cpu_used,
            br,
            &a_ivf,
            &encode_config,
            &args.anchor_args,
        );
        encode(
            &args.test_encoder,
            &args.input,
            args.cpu_used,
            br,
            &t_ivf,
            &encode_config,
            &args.test_args,
        );

        let a_frames = rvp9_bench::count_ivf_frames(&a_ivf);
        let t_frames = rvp9_bench::count_ivf_frames(&t_ivf);

        let a_br = match measured_bitrate_kbps(&a_ivf, a_frames, anchor_fps) {
            Ok(bitrate) => bitrate,
            Err(error) => {
                eprintln!("error: anchor bitrate at target {br}k: {error}");
                return ExitCode::FAILURE;
            }
        };
        let t_br = match measured_bitrate_kbps(&t_ivf, t_frames, test_fps) {
            Ok(bitrate) => bitrate,
            Err(error) => {
                eprintln!("error: test bitrate at target {br}k: {error}");
                return ExitCode::FAILURE;
            }
        };

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
    fn effective_fps_tracks_last_override_and_refuses_invalid_rates() {
        let extra = ["--fps=60/1", "--fps", "24/1", "--fps=30000/1001"].map(str::to_owned);
        assert_eq!(effective_frame_rate("30/1", &extra).unwrap(), (30000, 1001));
        assert_eq!(effective_frame_rate("30/1", &extra[..3]).unwrap(), (24, 1));
        for invalid in ["0/1", "30/0", "not-fps", "1/2/3", "4294967296/1"] {
            assert!(
                effective_frame_rate(invalid, &[]).is_err(),
                "accepted {invalid}"
            );
        }
        assert!(effective_frame_rate("30/1", &["--fps".to_owned()]).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn measured_rates_use_cfr_duration_for_quantized_clocks_and_real_header_sizes() {
        let fixture = crate::process_test_support::Fixture::new();
        let path = fixture.dir.join("clock.ivf");
        // The first two sizes/clock pairs reproduce retained beta Tree outputs:
        // pinned libvpx milliseconds vs candidate frame ticks, each60frames30fps.
        for (clock, scale, fps, frames, bytes, offset) in [
            (1000u32, 1u32, (30u32, 1u32), 60u32, 198914usize, 0u64),
            (30, 1, (30, 1), 60, 122523, 0),
            (1000, 1, (30000, 1001), 60, 198914, 73),
            (1000, 1, (30000, 1001), 1, 400, 73),
            (1, 1, (60, 1), 60, 4000, 0),
            (
                u32::MAX,
                u32::MAX,
                (u32::MAX, u32::MAX),
                60,
                4000,
                u64::MAX - 100,
            ),
        ] {
            let mut data = b"DKIF\0\0\x20\0VP90\0\x05\xd0\x02".to_vec();
            data.extend_from_slice(&clock.to_le_bytes());
            data.extend_from_slice(&scale.to_le_bytes());
            data.extend_from_slice(&frames.to_le_bytes());
            data.extend_from_slice(&0u32.to_le_bytes());
            let payload_bytes = bytes - 32 - frames as usize * 12;
            for index in 0..frames {
                let size = payload_bytes / frames as usize
                    + usize::from((index as usize) < payload_bytes % frames as usize);
                data.extend_from_slice(&(size as u32).to_le_bytes());
                let pts = u128::from(index) * u128::from(clock) * u128::from(fps.1)
                    / (u128::from(scale) * u128::from(fps.0));
                data.extend_from_slice(&(offset + u64::try_from(pts).unwrap()).to_le_bytes());
                data.resize(data.len() + size, 0);
            }
            std::fs::write(&path, data).unwrap();
            let rate = measured_bitrate_kbps(&path, frames, fps).unwrap();
            let expected = bytes as f64 * 8.0 * f64::from(fps.0)
                / (1000.0 * f64::from(frames) * f64::from(fps.1));
            assert!(
                (rate - expected).abs() < 1e-9,
                "clock {clock}/{scale}, fps {fps:?}: {rate} != {expected}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn external_encoder_timeout() {
        let fixture = crate::process_test_support::Fixture::new();
        let config = EncodeConfig {
            end_usage: "cbr",
            fps: "30/1",
            threads: 1,
            tile_columns: 0,
            lag_in_frames: 0,
            tune_content: "default",
            aq_mode: None,
            noise_sensitivity: None,
            limit: Some(1),
            timeout: crate::process_test_support::DEADLINE,
        };
        let failure = std::panic::catch_unwind(|| {
            encode(
                &fixture.tool,
                Path::new("input.y4m"),
                7,
                100,
                &fixture.dir.join("output.ivf"),
                &config,
                &[],
            )
        });
        let failure = failure.expect_err("encoder must reject child that outlives its deadline");
        let message = failure.downcast_ref::<String>().unwrap();
        assert!(
            message.contains("BD-rate encoder") && message.contains("timed out"),
            "{message}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn external_decoder_timeout() {
        let fixture = crate::process_test_support::Fixture::new();
        let reference = fixture.dir.join("reference.y4m");
        std::fs::write(&reference, crate::process_test_support::Y4M).unwrap();
        let value = decode_and_psnr_with_timeout(
            &fixture.tool,
            &fixture.dir.join("input.ivf"),
            &reference,
            crate::process_test_support::DEADLINE,
        );
        assert!(
            value.is_nan(),
            "decoder past its deadline must fail, got {value}"
        );
    }

    #[test]
    fn usage_advertises_retained_artifact_inputs() {
        assert!(USAGE.contains("--work-dir PATH"));
        assert!(USAGE.contains("--reference-input CLEAN.y4m"));
    }

    /// Per-test scratch root under the OS temp dir, unique per process so
    /// concurrent runs of this suite cannot collide.
    fn scratch_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("rvp9-bdrate-{tag}-{}", std::process::id()))
    }

    fn minimal_argv() -> Vec<String> {
        [
            "bdrate",
            "--anchor-encoder",
            "anchor",
            "--test-encoder",
            "test",
            "--decoder",
            "decoder",
            "--input",
            "input.y4m",
        ]
        .map(str::to_owned)
        .to_vec()
    }

    #[test]
    fn parser_defaults_the_work_directory_to_none_and_accepts_an_override() {
        assert_eq!(parse_args_from(&minimal_argv()).work_dir, None);

        let mut argv = minimal_argv();
        argv.extend(["--work-dir".to_owned(), "run/artifacts".to_owned()]);

        let args = parse_args_from(&argv);

        assert_eq!(args.work_dir, Some(PathBuf::from("run/artifacts")));
        assert_eq!(args.csv, None);
        assert_eq!(args.input, PathBuf::from("input.y4m"));
    }

    #[test]
    fn work_directory_artifacts_are_created_deterministic_and_collision_free() {
        let root = scratch_dir("work-dir-artifacts");
        let dir = root.join("nested");
        assert!(!root.exists(), "unexpected stale test directory");

        let artifacts = prepare_artifacts(Some(&dir)).unwrap();

        assert!(dir.is_dir(), "--work-dir was not created");
        let anchor_500 = artifacts.ivf("anchor", 500);
        assert_eq!(anchor_500, dir.join("anchor_500k.ivf"));
        assert_eq!(artifacts.ivf("test", 1500), dir.join("test_1500k.ivf"));
        // Role and bitrate both have to reach the name, or two encodes of the
        // sweep would overwrite one another.
        assert_ne!(anchor_500, artifacts.ivf("test", 500));
        assert_ne!(anchor_500, artifacts.ivf("anchor", 1500));
        assert_eq!(
            artifacts.ivf("anchor", 500),
            anchor_500,
            "not deterministic"
        );
        assert_eq!(
            decoded_y4m_path(&anchor_500),
            dir.join("anchor_500k.dec.y4m")
        );
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            0,
            "preparing the work dir left a file behind"
        );

        std::fs::remove_dir(&dir).unwrap();
        std::fs::remove_dir(&root).unwrap();
    }

    #[test]
    fn default_artifacts_stay_in_the_temp_directory_under_a_process_unique_prefix() {
        let artifacts = prepare_artifacts(None).unwrap();

        let anchor_500 = artifacts.ivf("anchor", 500);
        assert_eq!(anchor_500.parent().unwrap(), std::env::temp_dir());
        let name = anchor_500.file_name().unwrap().to_str().unwrap();
        assert_eq!(
            name,
            format!("rvp9_bdrate_{}_anchor_500k.ivf", std::process::id())
        );
        assert_ne!(anchor_500, artifacts.ivf("test", 500));
        assert_ne!(anchor_500, artifacts.ivf("anchor", 1500));
        assert_eq!(
            decoded_y4m_path(&anchor_500).parent().unwrap(),
            std::env::temp_dir()
        );
    }

    #[test]
    fn prepare_artifacts_reports_an_unusable_work_directory() {
        let path = scratch_dir("unusable-work-dir");
        assert!(!path.exists(), "unexpected stale test path");
        std::fs::write(&path, b"a file, not a directory").unwrap();

        let error = prepare_artifacts(Some(&path)).unwrap_err();

        assert!(error.starts_with("--work-dir "), "{error}");
        assert!(error.contains(&path.display().to_string()), "{error}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn retained_artifacts_are_never_overwritten() {
        let dir = scratch_dir("existing-artifact");
        assert!(!dir.exists(), "unexpected stale test directory");
        std::fs::create_dir(&dir).unwrap();
        let existing = dir.join("anchor_500k.ivf");
        std::fs::write(&existing, b"retained").unwrap();
        let unused = dir.join("test_500k.ivf");

        let error = reject_existing_artifacts(&[&existing, &unused]).unwrap_err();

        assert!(error.contains(&existing.display().to_string()), "{error}");
        assert!(error.contains("empty --work-dir"), "{error}");
        assert_eq!(std::fs::read(&existing).unwrap(), b"retained");
        std::fs::remove_file(&existing).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }

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
            timeout: rvp9_bench::process::DEFAULT_TIMEOUT,
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
            Err("insufficient, singular, or non-overlapping RD points")
        );
    }

    fn synthetic_curve(psnr_values: &[f64], bitrate_scale: f64) -> Vec<RdPoint> {
        psnr_values
            .iter()
            .map(|&psnr_db| RdPoint {
                bitrate_kbps: bitrate_scale * (100.0 * (0.05 * psnr_db).exp()),
                psnr_db,
            })
            .collect()
    }

    #[test]
    fn measured_bdrate_matches_high_precision_narrow_psnr_reference() {
        let anchor = [
            RdPoint {
                bitrate_kbps: 1152.068,
                psnr_db: 25.670945,
            },
            RdPoint {
                bitrate_kbps: 1174.084,
                psnr_db: 25.723668,
            },
            RdPoint {
                bitrate_kbps: 1207.220,
                psnr_db: 25.767877,
            },
            RdPoint {
                bitrate_kbps: 1253.696,
                psnr_db: 25.822174,
            },
            RdPoint {
                bitrate_kbps: 1326.328,
                psnr_db: 25.884390,
            },
            RdPoint {
                bitrate_kbps: 1569.352,
                psnr_db: 26.307338,
            },
        ];
        let test = [
            RdPoint {
                bitrate_kbps: 1151.356,
                psnr_db: 25.697093,
            },
            RdPoint {
                bitrate_kbps: 1175.428,
                psnr_db: 25.747508,
            },
            RdPoint {
                bitrate_kbps: 1207.796,
                psnr_db: 25.794290,
            },
            RdPoint {
                bitrate_kbps: 1251.228,
                psnr_db: 25.841255,
            },
            RdPoint {
                bitrate_kbps: 1326.504,
                psnr_db: 25.908804,
            },
            RdPoint {
                bitrate_kbps: 1568.004,
                psnr_db: 26.298732,
            },
        ];
        let actual = measured_bd_rate(&anchor, &test).expect("valid curves");
        let expected = -1.788_791_640_314_064;
        assert!(
            (actual - expected).abs() < 1e-8,
            "narrow-curve BD-rate should match Decimal80 reference {expected:.12}%, got {actual:.12}%",
        );
    }

    #[test]
    fn measured_bdrate_rejects_repeated_psnr_on_anchor_with_overlap() {
        let singular = synthetic_curve(&[0.0, 1.0, 1.0, 2.0], 1.0);
        let full_rank = synthetic_curve(&[0.0, 1.0, 2.0, 3.0], 1.25);

        assert!(
            measured_bd_rate(&singular, &full_rank).is_err(),
            "rank-deficient anchor curve produced a BD-rate"
        );
    }

    #[test]
    fn measured_bdrate_rejects_repeated_psnr_on_test_with_overlap() {
        let full_rank = synthetic_curve(&[0.0, 1.0, 2.0, 3.0], 1.0);
        let singular = synthetic_curve(&[0.0, 1.0, 1.0, 2.0], 1.25);

        assert!(
            measured_bd_rate(&full_rank, &singular).is_err(),
            "rank-deficient test curve produced a BD-rate"
        );
    }

    #[test]
    fn measured_bdrate_rejects_repeated_psnr_on_both_curves_with_overlap() {
        let singular = synthetic_curve(&[0.0, 1.0, 1.0, 2.0], 1.0);
        let proportional = synthetic_curve(&[0.0, 1.0, 1.0, 2.0], 1.25);

        assert!(
            measured_bd_rate(&singular, &proportional).is_err(),
            "two rank-deficient curves produced a BD-rate"
        );
    }

    #[test]
    fn measured_bdrate_rejects_repeated_realistic_psnr_with_overlap() {
        let singular = synthetic_curve(&[28.0, 34.0, 34.0, 40.0], 1.0);
        let full_rank = synthetic_curve(&[28.0, 34.0, 40.0, 46.0], 1.25);

        assert!(
            measured_bd_rate(&singular, &full_rank).is_err(),
            "rank-deficient curve at realistic PSNR values produced a BD-rate"
        );
    }

    #[test]
    fn measured_bdrate_keeps_full_rank_fits_with_repeated_observations() {
        let anchor = synthetic_curve(&[28.0, 34.0, 34.0, 40.0, 46.0, 46.0], 1.0);
        let proportional = synthetic_curve(&[28.0, 34.0, 34.0, 40.0, 46.0, 46.0], 1.25);
        let identical = anchor.clone();

        let proportional_delta = measured_bd_rate(&anchor, &proportional).unwrap();
        assert!(
            (proportional_delta - 25.0).abs() < 1e-6,
            "expected a 25% delta for proportional bitrates, got {proportional_delta}"
        );

        let identity_delta = measured_bd_rate(&anchor, &identical).unwrap();
        assert!(
            identity_delta.abs() < 1e-6,
            "expected zero delta for identical curves, got {identity_delta}"
        );
    }
}

#[cfg(all(test, unix))]
#[path = "../process_test_support.rs"]
mod process_test_support;
