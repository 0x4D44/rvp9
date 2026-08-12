//! Reproducible head-to-head VP9 encoder speed gate.
//!
//! Runs matched, warmed encoder pairs over a named corpus. It alternates arm
//! order, reverses clip order between rotations, reduces to median per-clip
//! times, and gates the corpus geometric mean. The manifest binds every result
//! to executable and input hashes. `--max-load1` makes acceptance runs fail
//! closed when the host is not idle enough for a trustworthy comparison.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

use rvp9_bench::count_ivf_frames;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EncoderKind {
    Vpxenc,
    TemporalSvc,
}

impl EncoderKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "vpxenc" => Ok(Self::Vpxenc),
            "temporal-svc" => Ok(Self::TemporalSvc),
            _ => Err(format!(
                "encoder kind must be vpxenc or temporal-svc, got `{value}`"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Vpxenc => "vpxenc",
            Self::TemporalSvc => "temporal-svc",
        }
    }
}

#[derive(Clone, Debug)]
struct Clip {
    name: String,
    path: PathBuf,
    header: Y4mHeader,
    sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Y4mHeader {
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
}

#[derive(Debug)]
struct Args {
    anchor_encoder: PathBuf,
    test_encoder: PathBuf,
    anchor_kind: EncoderKind,
    test_kind: EncoderKind,
    anchor_revision: String,
    test_revision: String,
    clip_specs: Vec<(String, PathBuf)>,
    work_dir: PathBuf,
    cpu_used: u32,
    target_bitrate: u32,
    end_usage: String,
    fps_num: u32,
    fps_den: u32,
    threads: u32,
    tile_columns: u32,
    temporal_layers: Option<u32>,
    aq_mode: u32,
    noise_sensitivity: u32,
    limit: u32,
    keyframe_interval: u32,
    warmups: u32,
    pairs: u32,
    rotations: u32,
    max_ratio: f64,
    max_load1: Option<f64>,
    anchor_args: Vec<String>,
    test_args: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Arm {
    Anchor,
    Test,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Anchor => "anchor",
            Self::Test => "test",
        }
    }
}

#[derive(Debug)]
struct RawRow {
    clip: String,
    rotation: u32,
    iteration: String,
    arm: Arm,
    order: u64,
    seconds: f64,
    frames: u32,
    sha256: String,
}

#[derive(Debug)]
struct ClipSummary {
    name: String,
    test_median: f64,
    anchor_median: f64,
    separate_ratio: f64,
    paired_ratio: f64,
    test_ms_per_frame: f64,
    anchor_ms_per_frame: f64,
    samples: usize,
    test_sha256: String,
    anchor_sha256: String,
}

fn usage() -> String {
    "Usage: encode-headtohead --anchor-encoder PATH --test-encoder PATH \\\n+  --anchor-kind {vpxenc,temporal-svc} --test-kind {vpxenc,temporal-svc} \\\n+  --anchor-revision REV --test-revision REV --clip NAME=PATH [--clip ...] \\\n+  --work-dir EMPTY_OR_MISSING_DIR [options]\n\n\
Options:\n  --cpu-used {6,7,8} --target-bitrate KBPS --end-usage {cbr,vbr}\n  \
--fps NUM/DEN --threads N --tile-columns 0..6 --temporal-layers {2,3}\n  \
--aq-mode {0,3} --noise-sensitivity 0..6 --limit N\n  \
--keyframe-interval N --warmups N --pairs ODD_N --rotations N\n  \
--max-ratio 1.05 [--max-load1 N]\n  \
--anchor-arg ARG [--anchor-arg ARG ...] --test-arg ARG\n"
        .to_owned()
}

fn required_value(argv: &[String], index: &mut usize) -> Result<String, String> {
    *index += 1;
    argv.get(*index)
        .cloned()
        .ok_or_else(|| format!("{} requires a value", argv[*index - 1]))
}

fn parse_u32(flag: &str, value: &str) -> Result<u32, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} requires an unsigned integer, got `{value}`"))
}

fn parse_positive_f64(flag: &str, value: &str) -> Result<f64, String> {
    let number = value
        .parse::<f64>()
        .map_err(|_| format!("{flag} requires a number, got `{value}`"))?;
    if !number.is_finite() || number <= 0.0 {
        return Err(format!("{flag} must be finite and greater than zero"));
    }
    Ok(number)
}

fn parse_fps(value: &str) -> Result<(u32, u32), String> {
    let (num, den) = value
        .split_once('/')
        .ok_or_else(|| "--fps requires NUM/DEN".to_owned())?;
    let num = parse_u32("--fps", num)?;
    let den = parse_u32("--fps", den)?;
    if num == 0 || den == 0 {
        return Err("--fps numerator and denominator must be positive".to_owned());
    }
    Ok((num, den))
}

fn valid_clip_name(name: &str) -> bool {
    !matches!(name, "" | "." | "..")
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
}

fn validate_extra_args(role: &str, extra: &[String]) -> Result<(), String> {
    const MATCHED_FLAGS: &[&str] = &[
        "--codec",
        "--passes",
        "--rt",
        "--cpu-used",
        "--end-usage",
        "--target-bitrate",
        "--fps",
        "--threads",
        "--tile-columns",
        "--lag-in-frames",
        "--tune-content",
        "--aq-mode",
        "--noise-sensitivity",
        "--temporal-layers",
        "--limit",
        "--kf-max-dist",
        "--keyframe-interval",
        "--input",
        "--output",
        "--quiet",
        "--ivf",
    ];
    for argument in extra {
        let conflicts = argument == "-o"
            || MATCHED_FLAGS
                .iter()
                .any(|flag| argument == flag || argument.starts_with(&format!("{flag}=")));
        if conflicts {
            return Err(format!(
                "--{role}-arg `{argument}` conflicts with the runner's matched configuration"
            ));
        }
    }
    Ok(())
}

fn parse_clip(value: &str) -> Result<(String, PathBuf), String> {
    let (name, path) = value
        .split_once('=')
        .ok_or_else(|| "--clip requires NAME=PATH".to_owned())?;
    if !valid_clip_name(name) {
        return Err(format!(
            "clip name `{name}` must contain only ASCII letters, digits, `_`, `.`, or `-`"
        ));
    }
    if path.is_empty() {
        return Err("--clip path must not be empty".to_owned());
    }
    Ok((name.to_owned(), PathBuf::from(path)))
}

fn parse_args_from(argv: &[String]) -> Result<Args, String> {
    let mut anchor_encoder = None;
    let mut test_encoder = None;
    let mut anchor_kind = None;
    let mut test_kind = None;
    let mut anchor_revision = None;
    let mut test_revision = None;
    let mut clip_specs = Vec::new();
    let mut work_dir = None;
    let mut cpu_used = 7;
    let mut target_bitrate = 1500;
    let mut end_usage = "vbr".to_owned();
    let (mut fps_num, mut fps_den) = (30, 1);
    let mut threads = 1;
    let mut tile_columns = 0;
    let mut temporal_layers = None;
    let mut aq_mode = 0;
    let mut noise_sensitivity = 0;
    let mut limit = 60;
    let mut keyframe_interval = 60;
    let mut warmups = 2;
    let mut pairs = 15;
    let mut rotations = 2;
    let mut max_ratio = 1.05;
    let mut max_load1 = None;
    let mut anchor_args = Vec::new();
    let mut test_args = Vec::new();

    let mut index = 1;
    while index < argv.len() {
        let arg = &argv[index];
        let (flag, inline) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(flag, value)| (flag, Some(value)));
        let value = |index: &mut usize| -> Result<String, String> {
            inline
                .map(str::to_owned)
                .map_or_else(|| required_value(argv, index), Ok)
        };
        match flag {
            "--anchor-encoder" => anchor_encoder = Some(PathBuf::from(value(&mut index)?)),
            "--test-encoder" => test_encoder = Some(PathBuf::from(value(&mut index)?)),
            "--anchor-kind" => anchor_kind = Some(EncoderKind::parse(&value(&mut index)?)?),
            "--test-kind" => test_kind = Some(EncoderKind::parse(&value(&mut index)?)?),
            "--anchor-revision" => anchor_revision = Some(value(&mut index)?),
            "--test-revision" => test_revision = Some(value(&mut index)?),
            "--clip" => clip_specs.push(parse_clip(&value(&mut index)?)?),
            "--work-dir" => work_dir = Some(PathBuf::from(value(&mut index)?)),
            "--cpu-used" => cpu_used = parse_u32(flag, &value(&mut index)?)?,
            "--target-bitrate" => target_bitrate = parse_u32(flag, &value(&mut index)?)?,
            "--end-usage" => end_usage = value(&mut index)?,
            "--fps" => (fps_num, fps_den) = parse_fps(&value(&mut index)?)?,
            "--threads" => threads = parse_u32(flag, &value(&mut index)?)?,
            "--tile-columns" => tile_columns = parse_u32(flag, &value(&mut index)?)?,
            "--temporal-layers" => temporal_layers = Some(parse_u32(flag, &value(&mut index)?)?),
            "--aq-mode" => aq_mode = parse_u32(flag, &value(&mut index)?)?,
            "--noise-sensitivity" => noise_sensitivity = parse_u32(flag, &value(&mut index)?)?,
            "--limit" => limit = parse_u32(flag, &value(&mut index)?)?,
            "--keyframe-interval" => keyframe_interval = parse_u32(flag, &value(&mut index)?)?,
            "--warmups" => warmups = parse_u32(flag, &value(&mut index)?)?,
            "--pairs" => pairs = parse_u32(flag, &value(&mut index)?)?,
            "--rotations" => rotations = parse_u32(flag, &value(&mut index)?)?,
            "--max-ratio" => max_ratio = parse_positive_f64(flag, &value(&mut index)?)?,
            "--max-load1" => max_load1 = Some(parse_positive_f64(flag, &value(&mut index)?)?),
            "--anchor-arg" => anchor_args.push(value(&mut index)?),
            "--test-arg" => test_args.push(value(&mut index)?),
            "--help" | "-h" => return Err(usage()),
            _ => return Err(format!("unknown argument: {arg}\n\n{}", usage())),
        }
        index += 1;
    }

    let args = Args {
        anchor_encoder: anchor_encoder.ok_or_else(|| "--anchor-encoder is required".to_owned())?,
        test_encoder: test_encoder.ok_or_else(|| "--test-encoder is required".to_owned())?,
        anchor_kind: anchor_kind.ok_or_else(|| "--anchor-kind is required".to_owned())?,
        test_kind: test_kind.ok_or_else(|| "--test-kind is required".to_owned())?,
        anchor_revision: anchor_revision
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "--anchor-revision is required".to_owned())?,
        test_revision: test_revision
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "--test-revision is required".to_owned())?,
        clip_specs,
        work_dir: work_dir.ok_or_else(|| "--work-dir is required".to_owned())?,
        cpu_used,
        target_bitrate,
        end_usage,
        fps_num,
        fps_den,
        threads,
        tile_columns,
        temporal_layers,
        aq_mode,
        noise_sensitivity,
        limit,
        keyframe_interval,
        warmups,
        pairs,
        rotations,
        max_ratio,
        max_load1,
        anchor_args,
        test_args,
    };
    validate_args(&args)?;
    Ok(args)
}

fn validate_args(args: &Args) -> Result<(), String> {
    if !args.anchor_encoder.is_file() || !args.test_encoder.is_file() {
        return Err("both encoder paths must name existing files".to_owned());
    }
    if !(6..=8).contains(&args.cpu_used) {
        return Err("--cpu-used must be 6, 7, or 8".to_owned());
    }
    if args.target_bitrate == 0 || args.threads == 0 || args.limit == 0 {
        return Err("bitrate, threads, and limit must be greater than zero".to_owned());
    }
    if !matches!(args.end_usage.as_str(), "cbr" | "vbr") {
        return Err("--end-usage must be cbr or vbr".to_owned());
    }
    if args.tile_columns > 6 {
        return Err("--tile-columns must be in 0..=6".to_owned());
    }
    if args
        .temporal_layers
        .is_some_and(|layers| !matches!(layers, 2 | 3))
    {
        return Err("--temporal-layers must be 2 or 3".to_owned());
    }
    if (matches!(args.anchor_kind, EncoderKind::TemporalSvc)
        || matches!(args.test_kind, EncoderKind::TemporalSvc))
        && args.temporal_layers.is_none()
    {
        return Err("temporal-svc encoders require --temporal-layers".to_owned());
    }
    if !matches!(args.aq_mode, 0 | 3) || args.noise_sensitivity > 6 {
        return Err("--aq-mode must be 0 or 3 and noise sensitivity must be in 0..=6".to_owned());
    }
    if args.keyframe_interval == 0 || args.warmups == 0 {
        return Err("keyframe interval and warmups must be greater than zero".to_owned());
    }
    if args.pairs == 0 || args.pairs.is_multiple_of(2) {
        return Err("--pairs must be a positive odd number".to_owned());
    }
    if args.rotations < 2 {
        return Err("--rotations must be at least 2".to_owned());
    }
    if args.clip_specs.is_empty() {
        return Err("at least one --clip is required".to_owned());
    }
    let unique = args
        .clip_specs
        .iter()
        .map(|(name, _)| name)
        .collect::<BTreeSet<_>>();
    if unique.len() != args.clip_specs.len() {
        return Err("clip names must be unique".to_owned());
    }
    validate_extra_args("anchor", &args.anchor_args)?;
    validate_extra_args("test", &args.test_args)?;
    if args.work_dir.exists() {
        let mut entries = fs::read_dir(&args.work_dir)
            .map_err(|error| format!("cannot inspect work directory: {error}"))?;
        if entries.next().is_some() {
            return Err("--work-dir must be empty or missing".to_owned());
        }
    }
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    parse_args_from(&std::env::args().collect::<Vec<_>>())
}

fn parse_y4m_header_line(line: &[u8]) -> Result<Y4mHeader, String> {
    let text = std::str::from_utf8(line).map_err(|_| "Y4M header is not UTF-8".to_owned())?;
    let text = text.trim_end_matches(['\r', '\n']);
    let mut fields = text.split_ascii_whitespace();
    if fields.next() != Some("YUV4MPEG2") {
        return Err("input does not start with YUV4MPEG2".to_owned());
    }
    let mut width = None;
    let mut height = None;
    let mut fps = None;
    for field in fields {
        let (tag, value) = field.split_at(1);
        match tag {
            "W" => width = value.parse().ok(),
            "H" => height = value.parse().ok(),
            "F" => {
                let (num, den) = value
                    .split_once(':')
                    .ok_or_else(|| "invalid Y4M F tag".to_owned())?;
                fps = Some((
                    num.parse::<u32>()
                        .map_err(|_| "invalid Y4M fps".to_owned())?,
                    den.parse::<u32>()
                        .map_err(|_| "invalid Y4M fps".to_owned())?,
                ));
            }
            _ => {}
        }
    }
    let (fps_num, fps_den) = fps.ok_or_else(|| "Y4M header is missing F tag".to_owned())?;
    let header = Y4mHeader {
        width: width.ok_or_else(|| "Y4M header is missing W tag".to_owned())?,
        height: height.ok_or_else(|| "Y4M header is missing H tag".to_owned())?,
        fps_num,
        fps_den,
    };
    if header.width == 0 || header.height == 0 || fps_num == 0 || fps_den == 0 {
        return Err("Y4M W, H, and F values must be positive".to_owned());
    }
    Ok(header)
}

fn read_y4m_header(path: &Path) -> Result<Y4mHeader, String> {
    let file =
        File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut line = Vec::new();
    BufReader::new(file)
        .take(4096)
        .read_until(b'\n', &mut line)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    if !line.ends_with(b"\n") {
        return Err(format!("{} has no bounded Y4M header", path.display()));
    }
    parse_y4m_header_line(&line).map_err(|error| format!("{}: {error}", path.display()))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let file =
        File::open(path).map_err(|error| format!("cannot hash {}: {error}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("cannot hash {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn load_clips(args: &Args) -> Result<Vec<Clip>, String> {
    let mut clips = Vec::with_capacity(args.clip_specs.len());
    for (name, path) in &args.clip_specs {
        let header = read_y4m_header(path)?;
        clips.push(Clip {
            name: name.clone(),
            path: path.clone(),
            header,
            sha256: sha256_file(path)?,
        });
    }
    let first = clips[0].header;
    if clips.iter().any(|clip| clip.header != first) {
        return Err("all clips must have identical width, height, and frame rate".to_owned());
    }
    let temporal_wrapper_used = matches!(args.anchor_kind, EncoderKind::TemporalSvc)
        || matches!(args.test_kind, EncoderKind::TemporalSvc);
    if temporal_wrapper_used
        && u64::from(first.fps_num) * u64::from(args.fps_den)
            != u64::from(args.fps_num) * u64::from(first.fps_den)
    {
        return Err(format!(
            "temporal-svc uses the Y4M frame rate {}/{}, which does not match --fps {}/{}",
            first.fps_num, first.fps_den, args.fps_num, args.fps_den
        ));
    }
    Ok(clips)
}

fn command_args(
    kind: EncoderKind,
    args: &Args,
    input: &Path,
    output: &Path,
    extra: &[String],
) -> Vec<OsString> {
    let mut result = Vec::<OsString>::new();
    let push = |result: &mut Vec<OsString>, value: String| result.push(value.into());
    match kind {
        EncoderKind::Vpxenc => {
            result.extend(
                ["--codec=vp9", "--passes=1", "--rt"]
                    .into_iter()
                    .map(OsString::from),
            );
            push(&mut result, format!("--cpu-used={}", args.cpu_used));
            push(&mut result, format!("--end-usage={}", args.end_usage));
            push(
                &mut result,
                format!("--target-bitrate={}", args.target_bitrate),
            );
            push(
                &mut result,
                format!("--fps={}/{}", args.fps_num, args.fps_den),
            );
            push(&mut result, format!("--threads={}", args.threads));
            push(&mut result, format!("--tile-columns={}", args.tile_columns));
            result.extend(
                ["--lag-in-frames=0", "--tune-content=default"]
                    .into_iter()
                    .map(OsString::from),
            );
            push(&mut result, format!("--aq-mode={}", args.aq_mode));
            push(
                &mut result,
                format!("--noise-sensitivity={}", args.noise_sensitivity),
            );
            if let Some(layers) = args.temporal_layers {
                push(&mut result, format!("--temporal-layers={layers}"));
            }
            push(&mut result, format!("--limit={}", args.limit));
            push(
                &mut result,
                format!("--kf-max-dist={}", args.keyframe_interval),
            );
            result.extend([OsString::from("--quiet"), OsString::from("--ivf")]);
            result.push(OsString::from("-o"));
            result.push(output.as_os_str().to_owned());
            result.push(input.as_os_str().to_owned());
        }
        EncoderKind::TemporalSvc => {
            result.push(OsString::from("--input"));
            result.push(input.as_os_str().to_owned());
            result.push(OsString::from("--output"));
            result.push(output.as_os_str().to_owned());
            for (flag, value) in [
                ("--target-bitrate", args.target_bitrate),
                ("--cpu-used", args.cpu_used),
            ] {
                result.push(flag.into());
                result.push(value.to_string().into());
            }
            result.push(OsString::from("--end-usage"));
            result.push(args.end_usage.clone().into());
            for (flag, value) in [
                ("--threads", args.threads),
                ("--tile-columns", args.tile_columns),
                (
                    "--temporal-layers",
                    args.temporal_layers.expect("validated temporal layers"),
                ),
                ("--aq-mode", args.aq_mode),
                ("--noise-sensitivity", args.noise_sensitivity),
                ("--limit", args.limit),
                ("--keyframe-interval", args.keyframe_interval),
            ] {
                result.push(flag.into());
                result.push(value.to_string().into());
            }
        }
    }
    result.extend(extra.iter().map(OsString::from));
    result
}

fn remove_rvp9_environment(command: &mut Command) {
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("RVP9_") {
            command.env_remove(name);
        }
    }
}

fn run_encoder(
    encoder: &Path,
    kind: EncoderKind,
    args: &Args,
    input: &Path,
    output: &Path,
    extra: &[String],
) -> Result<(f64, u32, String), String> {
    if output.exists() {
        fs::remove_file(output)
            .map_err(|error| format!("cannot replace {}: {error}", output.display()))?;
    }
    let mut command = Command::new(encoder);
    command.args(command_args(kind, args, input, output, extra));
    remove_rvp9_environment(&mut command);
    let start = Instant::now();
    let outcome = command
        .output()
        .map_err(|error| format!("failed to launch {} ({kind:?}): {error}", encoder.display()))?;
    let seconds = start.elapsed().as_secs_f64();
    if !outcome.status.success() {
        return Err(format!(
            "{} ({kind:?}) exited {}\nstdout:\n{}\nstderr:\n{}",
            encoder.display(),
            outcome.status,
            String::from_utf8_lossy(&outcome.stdout),
            String::from_utf8_lossy(&outcome.stderr)
        ));
    }
    let frames = count_ivf_frames(output);
    if frames != args.limit {
        return Err(format!(
            "{} emitted {frames} IVF frames, expected {}",
            encoder.display(),
            args.limit
        ));
    }
    Ok((seconds, frames, sha256_file(output)?))
}

fn parse_load1(text: &str) -> Option<f64> {
    text.split(|character: char| {
        !(character.is_ascii_digit() || matches!(character, '.' | '-' | '+' | 'e' | 'E'))
    })
    .filter(|part| !part.is_empty())
    .find_map(|part| part.parse::<f64>().ok())
}

fn current_load1() -> Option<f64> {
    if let Ok(text) = fs::read_to_string("/proc/loadavg") {
        return parse_load1(&text);
    }
    let output = Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_load1(&String::from_utf8_lossy(&output.stdout))
}

fn enforce_load(
    args: &Args,
    point: &str,
    observations: &mut Vec<(String, f64)>,
) -> Result<(), String> {
    let Some(max_load) = args.max_load1 else {
        if let Some(load) = current_load1() {
            observations.push((point.to_owned(), load));
        }
        return Ok(());
    };
    let load = current_load1().ok_or_else(|| {
        format!("cannot read host one-minute load at `{point}`; --max-load1 fails closed")
    })?;
    observations.push((point.to_owned(), load));
    if load > max_load {
        return Err(format!(
            "host one-minute load {load:.3} exceeds --max-load1 {max_load:.3} at `{point}`"
        ));
    }
    Ok(())
}

fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn geometric_mean(values: &[f64]) -> f64 {
    assert!(!values.is_empty());
    (values.iter().map(|value| value.ln()).sum::<f64>() / values.len() as f64).exp()
}

fn summarize(clips: &[Clip], rows: &[RawRow], limit: u32) -> Result<Vec<ClipSummary>, String> {
    let mut summaries = Vec::with_capacity(clips.len());
    for clip in clips {
        let measured = rows
            .iter()
            .filter(|row| row.clip == clip.name && !row.iteration.starts_with("warmup"))
            .collect::<Vec<_>>();
        let test = measured
            .iter()
            .filter(|row| row.arm == Arm::Test)
            .map(|row| row.seconds)
            .collect::<Vec<_>>();
        let anchor = measured
            .iter()
            .filter(|row| row.arm == Arm::Anchor)
            .map(|row| row.seconds)
            .collect::<Vec<_>>();
        if test.is_empty() || test.len() != anchor.len() {
            return Err(format!("{} has incomplete measured arms", clip.name));
        }
        let mut pairs = BTreeMap::<(u32, String), BTreeMap<Arm, f64>>::new();
        for row in &measured {
            pairs
                .entry((row.rotation, row.iteration.clone()))
                .or_default()
                .insert(row.arm, row.seconds);
        }
        let paired = pairs
            .values()
            .map(|pair| {
                let test = pair.get(&Arm::Test).ok_or("missing test half of pair")?;
                let anchor = pair
                    .get(&Arm::Anchor)
                    .ok_or("missing anchor half of pair")?;
                Ok(test / anchor)
            })
            .collect::<Result<Vec<_>, &str>>()
            .map_err(|error| format!("{}: {error}", clip.name))?;
        let stable_hash = |arm| -> Result<String, String> {
            let hashes = rows
                .iter()
                .filter(|row| row.clip == clip.name && row.arm == arm)
                .map(|row| row.sha256.as_str())
                .collect::<BTreeSet<_>>();
            if hashes.len() != 1 {
                return Err(format!(
                    "{} {} emitted {} distinct IVF hashes",
                    clip.name,
                    arm.name(),
                    hashes.len()
                ));
            }
            Ok((*hashes.first().expect("one stable hash")).to_owned())
        };
        let test_median = median(&test);
        let anchor_median = median(&anchor);
        summaries.push(ClipSummary {
            name: clip.name.clone(),
            test_median,
            anchor_median,
            separate_ratio: test_median / anchor_median,
            paired_ratio: median(&paired),
            test_ms_per_frame: 1000.0 * test_median / f64::from(limit),
            anchor_ms_per_frame: 1000.0 * anchor_median / f64::from(limit),
            samples: test.len(),
            test_sha256: stable_hash(Arm::Test)?,
            anchor_sha256: stable_hash(Arm::Anchor)?,
        });
    }
    Ok(summaries)
}

fn command_text(program: &Path, arguments: &[OsString]) -> String {
    std::iter::once(program.as_os_str())
        .chain(arguments.iter().map(OsString::as_os_str))
        .map(|argument| format!("{argument:?}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn probe_command(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn write_raw(path: &Path, rows: &[RawRow]) -> Result<(), String> {
    let mut writer = BufWriter::new(
        File::create(path).map_err(|error| format!("cannot create {}: {error}", path.display()))?,
    );
    writeln!(
        writer,
        "clip,rotation,iteration,arm,order,seconds,frames,sha256"
    )
    .map_err(|error| error.to_string())?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{:.9},{},{}",
            row.clip,
            row.rotation,
            row.iteration,
            row.arm.name(),
            row.order,
            row.seconds,
            row.frames,
            row.sha256
        )
        .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

fn write_summary(
    path: &Path,
    summaries: &[ClipSummary],
    separate_geomean: f64,
    paired_geomean: f64,
) -> Result<(), String> {
    let mut writer = BufWriter::new(
        File::create(path).map_err(|error| format!("cannot create {}: {error}", path.display()))?,
    );
    writeln!(writer, "clip,test_median,anchor_median,separate_ratio,paired_ratio,test_ms_per_frame,anchor_ms_per_frame,samples,test_sha256,anchor_sha256")
        .map_err(|error| error.to_string())?;
    for row in summaries {
        writeln!(
            writer,
            "{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{}",
            row.name,
            row.test_median,
            row.anchor_median,
            row.separate_ratio,
            row.paired_ratio,
            row.test_ms_per_frame,
            row.anchor_ms_per_frame,
            row.samples,
            row.test_sha256,
            row.anchor_sha256
        )
        .map_err(|error| error.to_string())?;
    }
    writeln!(
        writer,
        "corpus,,,{separate_geomean:.9},{paired_geomean:.9},,,,,"
    )
    .map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())
}

struct ManifestResults<'a> {
    anchor_sha256: &'a str,
    test_sha256: &'a str,
    loads: &'a [(String, f64)],
    separate_geomean: f64,
    paired_geomean: f64,
}

fn write_manifest(
    path: &Path,
    args: &Args,
    clips: &[Clip],
    results: &ManifestResults<'_>,
) -> Result<(), String> {
    let mut writer = BufWriter::new(
        File::create(path).map_err(|error| format!("cannot create {}: {error}", path.display()))?,
    );
    writeln!(writer, "rvp9 encode-headtohead manifest schema=1").map_err(|e| e.to_string())?;
    writeln!(writer, "argv={:?}", std::env::args_os().collect::<Vec<_>>())
        .map_err(|e| e.to_string())?;
    writeln!(
        writer,
        "git_commit={}",
        probe_command("git", &["rev-parse", "HEAD"])
    )
    .map_err(|e| e.to_string())?;
    writeln!(writer, "host={}", probe_command("uname", &["-a"])).map_err(|e| e.to_string())?;
    writeln!(writer, "rustc={}", probe_command("rustc", &["--version"]))
        .map_err(|e| e.to_string())?;
    writeln!(
        writer,
        "available_parallelism={}",
        std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get)
    )
    .map_err(|e| e.to_string())?;
    for (role, encoder, kind, revision, hash, extra) in [
        (
            "anchor",
            &args.anchor_encoder,
            args.anchor_kind,
            &args.anchor_revision,
            results.anchor_sha256,
            &args.anchor_args,
        ),
        (
            "test",
            &args.test_encoder,
            args.test_kind,
            &args.test_revision,
            results.test_sha256,
            &args.test_args,
        ),
    ] {
        writeln!(writer, "{role}_encoder={}", encoder.display()).map_err(|e| e.to_string())?;
        writeln!(writer, "{role}_kind={}", kind.name()).map_err(|e| e.to_string())?;
        writeln!(writer, "{role}_revision={revision}").map_err(|e| e.to_string())?;
        writeln!(writer, "{role}_sha256={hash}").map_err(|e| e.to_string())?;
        writeln!(writer, "{role}_extra_args={extra:?}").map_err(|e| e.to_string())?;
    }
    writeln!(writer, "cpu_used={}", args.cpu_used).map_err(|e| e.to_string())?;
    writeln!(writer, "target_bitrate_kbps={}", args.target_bitrate).map_err(|e| e.to_string())?;
    writeln!(writer, "end_usage={}", args.end_usage).map_err(|e| e.to_string())?;
    writeln!(writer, "fps={}/{}", args.fps_num, args.fps_den).map_err(|e| e.to_string())?;
    writeln!(writer, "threads={}", args.threads).map_err(|e| e.to_string())?;
    writeln!(writer, "tile_columns={}", args.tile_columns).map_err(|e| e.to_string())?;
    writeln!(writer, "temporal_layers={:?}", args.temporal_layers).map_err(|e| e.to_string())?;
    writeln!(writer, "aq_mode={}", args.aq_mode).map_err(|e| e.to_string())?;
    writeln!(writer, "noise_sensitivity={}", args.noise_sensitivity).map_err(|e| e.to_string())?;
    writeln!(writer, "limit={}", args.limit).map_err(|e| e.to_string())?;
    writeln!(writer, "keyframe_interval={}", args.keyframe_interval).map_err(|e| e.to_string())?;
    writeln!(
        writer,
        "warmups={} pairs={} rotations={}",
        args.warmups, args.pairs, args.rotations
    )
    .map_err(|e| e.to_string())?;
    writeln!(
        writer,
        "ordering=even rotations reverse clips; every pair alternates first arm"
    )
    .map_err(|e| e.to_string())?;
    writeln!(
        writer,
        "max_ratio={:.9} max_load1={:?}",
        args.max_ratio, args.max_load1
    )
    .map_err(|e| e.to_string())?;
    for clip in clips {
        writeln!(writer, "clip_{}_path={}", clip.name, clip.path.display())
            .map_err(|e| e.to_string())?;
        writeln!(writer, "clip_{}_sha256={}", clip.name, clip.sha256).map_err(|e| e.to_string())?;
        writeln!(
            writer,
            "clip_{}_video={}x{}@{}/{}",
            clip.name,
            clip.header.width,
            clip.header.height,
            clip.header.fps_num,
            clip.header.fps_den
        )
        .map_err(|e| e.to_string())?;
    }
    let sample = &clips[0];
    writeln!(
        writer,
        "anchor_command={}",
        command_text(
            &args.anchor_encoder,
            &command_args(
                args.anchor_kind,
                args,
                &sample.path,
                Path::new("<OUTPUT>"),
                &args.anchor_args
            )
        )
    )
    .map_err(|e| e.to_string())?;
    writeln!(
        writer,
        "test_command={}",
        command_text(
            &args.test_encoder,
            &command_args(
                args.test_kind,
                args,
                &sample.path,
                Path::new("<OUTPUT>"),
                &args.test_args
            )
        )
    )
    .map_err(|e| e.to_string())?;
    for (point, load) in results.loads {
        writeln!(writer, "load1_{point}={load:.6}").map_err(|e| e.to_string())?;
    }
    writeln!(
        writer,
        "corpus_separate_ratio={:.9}",
        results.separate_geomean
    )
    .map_err(|e| e.to_string())?;
    writeln!(writer, "corpus_paired_ratio={:.9}", results.paired_geomean)
        .map_err(|e| e.to_string())?;
    writer.flush().map_err(|error| error.to_string())
}

fn run() -> Result<bool, String> {
    let args = parse_args()?;
    let mut loads = Vec::new();
    enforce_load(&args, "preflight", &mut loads)?;
    let clips = load_clips(&args)?;
    let anchor_sha256 = sha256_file(&args.anchor_encoder)?;
    let test_sha256 = sha256_file(&args.test_encoder)?;
    fs::create_dir_all(args.work_dir.join("outputs"))
        .map_err(|error| format!("cannot create work directory: {error}"))?;

    let mut rows = Vec::new();
    let mut order = 0u64;
    for rotation in 1..=args.rotations {
        let indices = if rotation % 2 == 0 {
            (0..clips.len()).rev().collect::<Vec<_>>()
        } else {
            (0..clips.len()).collect::<Vec<_>>()
        };
        for index in indices {
            let clip = &clips[index];
            enforce_load(
                &args,
                &format!("rotation{rotation}_{}", clip.name),
                &mut loads,
            )?;
            for warmup in 1..=args.warmups {
                let first = if (warmup + rotation) % 2 == 0 {
                    Arm::Test
                } else {
                    Arm::Anchor
                };
                for arm in [
                    first,
                    if first == Arm::Test {
                        Arm::Anchor
                    } else {
                        Arm::Test
                    },
                ] {
                    order += 1;
                    let output = args.work_dir.join("outputs").join(format!(
                        "{}-{rotation}-{}.ivf",
                        clip.name,
                        arm.name()
                    ));
                    let (encoder, kind, extra) = match arm {
                        Arm::Anchor => (&args.anchor_encoder, args.anchor_kind, &args.anchor_args),
                        Arm::Test => (&args.test_encoder, args.test_kind, &args.test_args),
                    };
                    let (seconds, frames, sha256) =
                        run_encoder(encoder, kind, &args, &clip.path, &output, extra)?;
                    rows.push(RawRow {
                        clip: clip.name.clone(),
                        rotation,
                        iteration: format!("warmup-{warmup}"),
                        arm,
                        order,
                        seconds,
                        frames,
                        sha256,
                    });
                }
            }
            for pair in 1..=args.pairs {
                let first = if (pair + rotation) % 2 == 0 {
                    Arm::Test
                } else {
                    Arm::Anchor
                };
                for arm in [
                    first,
                    if first == Arm::Test {
                        Arm::Anchor
                    } else {
                        Arm::Test
                    },
                ] {
                    order += 1;
                    let output = args.work_dir.join("outputs").join(format!(
                        "{}-{rotation}-{}.ivf",
                        clip.name,
                        arm.name()
                    ));
                    let (encoder, kind, extra) = match arm {
                        Arm::Anchor => (&args.anchor_encoder, args.anchor_kind, &args.anchor_args),
                        Arm::Test => (&args.test_encoder, args.test_kind, &args.test_args),
                    };
                    let (seconds, frames, sha256) =
                        run_encoder(encoder, kind, &args, &clip.path, &output, extra)?;
                    rows.push(RawRow {
                        clip: clip.name.clone(),
                        rotation,
                        iteration: pair.to_string(),
                        arm,
                        order,
                        seconds,
                        frames,
                        sha256,
                    });
                }
            }
        }
    }
    enforce_load(&args, "complete", &mut loads)?;
    let summaries = summarize(&clips, &rows, args.limit)?;
    let separate_geomean = geometric_mean(
        &summaries
            .iter()
            .map(|row| row.separate_ratio)
            .collect::<Vec<_>>(),
    );
    let paired_geomean = geometric_mean(
        &summaries
            .iter()
            .map(|row| row.paired_ratio)
            .collect::<Vec<_>>(),
    );
    write_raw(&args.work_dir.join("raw.csv"), &rows)?;
    write_summary(
        &args.work_dir.join("summary.csv"),
        &summaries,
        separate_geomean,
        paired_geomean,
    )?;
    write_manifest(
        &args.work_dir.join("manifest.txt"),
        &args,
        &clips,
        &ManifestResults {
            anchor_sha256: &anchor_sha256,
            test_sha256: &test_sha256,
            loads: &loads,
            separate_geomean,
            paired_geomean,
        },
    )?;

    for row in &summaries {
        println!(
            "{}: separate {:.6}x, paired {:.6}x, test {:.3} ms/frame, anchor {:.3} ms/frame",
            row.name,
            row.separate_ratio,
            row.paired_ratio,
            row.test_ms_per_frame,
            row.anchor_ms_per_frame
        );
    }
    println!(
        "corpus: separate {separate_geomean:.6}x, paired {paired_geomean:.6}x (limit {:.6}x)",
        args.max_ratio
    );
    Ok(separate_geomean <= args.max_ratio && paired_geomean <= args.max_ratio)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("encode-headtohead: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_args(kind: EncoderKind) -> Args {
        Args {
            anchor_encoder: PathBuf::from("anchor"),
            test_encoder: PathBuf::from("test"),
            anchor_kind: kind,
            test_kind: kind,
            anchor_revision: "a".into(),
            test_revision: "t".into(),
            clip_specs: vec![("clip".into(), PathBuf::from("clip.y4m"))],
            work_dir: PathBuf::from("work"),
            cpu_used: 8,
            target_bitrate: 1500,
            end_usage: "vbr".into(),
            fps_num: 30,
            fps_den: 1,
            threads: 1,
            tile_columns: 0,
            temporal_layers: Some(3),
            aq_mode: 0,
            noise_sensitivity: 0,
            limit: 60,
            keyframe_interval: 60,
            warmups: 2,
            pairs: 15,
            rotations: 2,
            max_ratio: 1.05,
            max_load1: None,
            anchor_args: vec![],
            test_args: vec![],
        }
    }

    fn strings(values: Vec<OsString>) -> Vec<String> {
        values
            .into_iter()
            .map(|value| value.into_string().unwrap())
            .collect()
    }

    #[test]
    fn vpxenc_command_contains_matched_realtime_surface() {
        let args = fake_args(EncoderKind::Vpxenc);
        let command = strings(command_args(
            EncoderKind::Vpxenc,
            &args,
            Path::new("input.y4m"),
            Path::new("output.ivf"),
            &[],
        ));
        for expected in [
            "--codec=vp9",
            "--passes=1",
            "--rt",
            "--cpu-used=8",
            "--fps=30/1",
            "--lag-in-frames=0",
            "--tune-content=default",
            "--temporal-layers=3",
            "--kf-max-dist=60",
        ] {
            assert!(command.iter().any(|argument| argument == expected));
        }
        assert_eq!(command.last().unwrap(), "input.y4m");
    }

    #[test]
    fn temporal_svc_command_contains_wrapper_surface() {
        let args = fake_args(EncoderKind::TemporalSvc);
        let command = strings(command_args(
            EncoderKind::TemporalSvc,
            &args,
            Path::new("input.y4m"),
            Path::new("output.ivf"),
            &["--extra".into()],
        ));
        assert_eq!(
            &command[..4],
            ["--input", "input.y4m", "--output", "output.ivf"]
        );
        assert!(
            command
                .windows(2)
                .any(|pair| pair == ["--temporal-layers", "3"])
        );
        assert!(
            command
                .windows(2)
                .any(|pair| pair == ["--keyframe-interval", "60"])
        );
        assert_eq!(command.last().unwrap(), "--extra");
    }

    #[test]
    fn parsing_requires_safe_unique_clips_and_odd_pairs() {
        assert!(parse_clip("ducks=/tmp/clip.y4m").is_ok());
        assert!(parse_clip("bad,name=/tmp/clip.y4m").is_err());
        assert!(parse_clip("missing-separator").is_err());
        assert!(valid_clip_name("old-town_720p.30"));
        assert!(!valid_clip_name("old town"));
        assert!(!valid_clip_name("."));
        assert!(!valid_clip_name(".."));
    }

    #[test]
    fn argument_parser_applies_defaults_and_rejects_even_pairs() {
        let executable = std::env::current_exe().unwrap();
        let work_dir = std::env::temp_dir().join(format!(
            "rvp9-encode-headtohead-parse-{}-must-not-exist",
            std::process::id()
        ));
        assert!(!work_dir.exists());
        let mut argv = vec![
            "encode-headtohead".to_owned(),
            "--anchor-encoder".to_owned(),
            executable.display().to_string(),
            "--test-encoder".to_owned(),
            executable.display().to_string(),
            "--anchor-kind".to_owned(),
            "vpxenc".to_owned(),
            "--test-kind".to_owned(),
            "vpxenc".to_owned(),
            "--anchor-revision".to_owned(),
            "anchor".to_owned(),
            "--test-revision".to_owned(),
            "test".to_owned(),
            "--clip".to_owned(),
            "clip=/not/read/by-parser.y4m".to_owned(),
            "--work-dir".to_owned(),
            work_dir.display().to_string(),
        ];
        let parsed = parse_args_from(&argv).unwrap();
        assert_eq!(parsed.cpu_used, 7);
        assert_eq!(parsed.pairs, 15);
        argv.extend(["--pairs".to_owned(), "2".to_owned()]);
        assert!(parse_args_from(&argv).is_err());
    }

    #[test]
    fn extra_arguments_cannot_override_matched_configuration() {
        assert!(validate_extra_args("anchor", &["--trace".into()]).is_ok());
        assert!(validate_extra_args("anchor", &["--cpu-used=6".into()]).is_err());
        assert!(validate_extra_args("test", &["--output".into()]).is_err());
        assert!(validate_extra_args("test", &["-o".into()]).is_err());
    }

    #[test]
    fn y4m_header_requires_dimensions_and_explicit_rate() {
        let header =
            parse_y4m_header_line(b"YUV4MPEG2 W1280 H720 F30:1 Ip A1:1 C420jpeg\n").unwrap();
        assert_eq!(
            header,
            Y4mHeader {
                width: 1280,
                height: 720,
                fps_num: 30,
                fps_den: 1
            }
        );
        assert!(parse_y4m_header_line(b"YUV4MPEG2 W1280 H720\n").is_err());
        assert!(parse_y4m_header_line(b"not-y4m\n").is_err());
    }

    #[test]
    fn median_geomean_and_load_parsing_are_deterministic() {
        assert_eq!(median(&[4.0, 1.0, 3.0]), 3.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert!((geometric_mean(&[1.0, 4.0]) - 2.0).abs() < 1e-12);
        assert_eq!(parse_load1("{ 2.75 3.0 4.0 }"), Some(2.75));
        assert_eq!(parse_load1("1.25 0.5 0.2 1/100 42"), Some(1.25));
        assert_eq!(parse_load1("unavailable"), None);
    }

    #[test]
    fn summary_uses_paired_ratios_and_rejects_hash_instability() {
        let clip = Clip {
            name: "c".into(),
            path: PathBuf::from("c.y4m"),
            header: Y4mHeader {
                width: 8,
                height: 8,
                fps_num: 30,
                fps_den: 1,
            },
            sha256: "input".into(),
        };
        let make = |rotation, iteration: &str, arm, seconds, hash: &str| RawRow {
            clip: "c".into(),
            rotation,
            iteration: iteration.into(),
            arm,
            order: 0,
            seconds,
            frames: 60,
            sha256: hash.into(),
        };
        let mut rows = vec![
            make(1, "1", Arm::Test, 2.0, "t"),
            make(1, "1", Arm::Anchor, 1.0, "a"),
            make(2, "1", Arm::Test, 4.0, "t"),
            make(2, "1", Arm::Anchor, 2.0, "a"),
        ];
        let summary = summarize(std::slice::from_ref(&clip), &rows, 60).unwrap();
        assert_eq!(summary[0].separate_ratio, 2.0);
        assert_eq!(summary[0].paired_ratio, 2.0);
        rows[0].sha256 = "changed".into();
        assert!(summarize(&[clip], &rows, 60).is_err());
    }
}
