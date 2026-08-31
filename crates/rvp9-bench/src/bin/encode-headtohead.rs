//! Reproducible head-to-head VP9 encoder speed gate.
//!
//! Runs matched, warmed encoder pairs over a named corpus. It alternates arm
//! order, reverses clip order between rotations, reduces to median per-clip
//! times, and gates the corpus geometric mean. The manifest binds every result
//! to executable and input hashes. `--max-load1` makes acceptance runs fail
//! closed when the host is not idle enough for a trustworthy comparison.
//!
//! The `temporal-svc` and `spatial-svc` kinds drive the matched SVC wrappers.
//! Neither accepts a generic `--fps`, so both take the frame rate from the Y4M
//! header. Every arm must emit one stable output hash;
//! `--allow-anchor-hash-instability` is the single narrow exception, for the
//! pinned threaded spatial example.

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
    SpatialSvc,
}

impl EncoderKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "vpxenc" => Ok(Self::Vpxenc),
            "temporal-svc" => Ok(Self::TemporalSvc),
            "spatial-svc" => Ok(Self::SpatialSvc),
            _ => Err(format!(
                "encoder kind must be vpxenc, temporal-svc, or spatial-svc, got `{value}`"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Vpxenc => "vpxenc",
            Self::TemporalSvc => "temporal-svc",
            Self::SpatialSvc => "spatial-svc",
        }
    }

    /// Wrapper kinds take the frame rate from the Y4M header, not `--fps`.
    fn is_svc_wrapper(self) -> bool {
        matches!(self, Self::TemporalSvc | Self::SpatialSvc)
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
    spatial_layers: Option<u32>,
    inter_layer_pred: u32,
    aq_mode: u32,
    noise_sensitivity: u32,
    limit: u32,
    keyframe_interval: u32,
    warmups: u32,
    pairs: u32,
    rotations: u32,
    max_ratio: f64,
    max_load1: Option<f64>,
    allow_anchor_hash_instability: bool,
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
    test_distinct_hashes: usize,
    anchor_distinct_hashes: usize,
}

fn usage() -> String {
    "Usage: encode-headtohead --anchor-encoder PATH --test-encoder PATH \\\n  --anchor-kind {vpxenc,temporal-svc,spatial-svc} --test-kind {vpxenc,temporal-svc,spatial-svc} \\\n  --anchor-revision REV --test-revision REV --clip NAME=PATH [--clip ...] \\\n  --work-dir EMPTY_OR_MISSING_DIR [options]\n\n\
Options:\n  --cpu-used {6,7,8} --target-bitrate KBPS --end-usage {cbr,vbr}\n  \
--fps NUM/DEN --threads N --tile-columns 0..6\n  \
--temporal-layers {2,3} | --spatial-layers {2,3} [--inter-layer-pred {0,3}]\n  \
--aq-mode {0,3} --noise-sensitivity 0..6 --limit N\n  \
--keyframe-interval N --warmups N --pairs ODD_N --rotations N\n  \
--max-ratio 1.05 [--max-load1 N] [--allow-anchor-hash-instability]\n  \
--anchor-arg ARG [--anchor-arg ARG ...] --test-arg ARG\n"
        .to_owned()
}

/// A capability probe: `--help` / `-h` anywhere in argv answers with usage and
/// exits 0, so a caller can discover the flag surface without supplying any of
/// the required arguments.
fn help_requested(argv: &[String]) -> bool {
    argv.iter().any(|arg| arg == "--help" || arg == "-h")
}

/// Canonical libvpx spatial bitrate weights: each layer carries 4x the one
/// below it, matching the 1/2 and 1/4 scale-factor ladders.
fn spatial_layer_weights(spatial_layers: u32) -> Result<&'static [u32], String> {
    match spatial_layers {
        2 => Ok(&[1, 4]),
        3 => Ok(&[1, 4, 16]),
        _ => Err("--spatial-layers must be 2 or 3".to_owned()),
    }
}

/// Splits the matched target across spatial layers, refusing any target that
/// cannot be allocated exactly; an inexact split would silently change the
/// bitrate the anchor actually encodes at.
fn spatial_layer_bitrates(target_kbps: u32, spatial_layers: u32) -> Result<String, String> {
    let weights = spatial_layer_weights(spatial_layers)?;
    let total = weights.iter().sum::<u32>();
    if !target_kbps.is_multiple_of(total) {
        return Err(format!(
            "spatial target bitrate {target_kbps} kbps must be divisible by {total}"
        ));
    }
    Ok(weights
        .iter()
        .map(|weight| (target_kbps / total * weight).to_string())
        .collect::<Vec<_>>()
        .join(","))
}

fn spatial_scale_factors(spatial_layers: u32) -> Result<&'static str, String> {
    match spatial_layers {
        2 => Ok("1/2,1/1"),
        3 => Ok("1/4,1/2,1/1"),
        _ => Err("--spatial-layers must be 2 or 3".to_owned()),
    }
}

/// Spatial Q bounds on the libvpx example's 0..=63 external scale.
fn libvpx_spatial_q_bounds(end_usage: &str) -> (u32, u32) {
    if end_usage == "cbr" { (2, 56) } else { (1, 63) }
}

/// The same bounds on the vpxenc-style 0..=255 qindex scale.
fn rvp9_spatial_q_bounds(end_usage: &str) -> (u32, u32) {
    if end_usage == "cbr" {
        (8, 224)
    } else {
        (4, 255)
    }
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
        // Matched spatial controls, on both the vpxenc and libvpx-example
        // spellings.
        "--spatial-layers",
        "--inter-layer-pred",
        "--min-q",
        "--max-q",
        "--frames",
        "--speed",
        "--aqmode",
        "--rc-end-usage",
        "--kf-dist",
        "--scale-factors",
        "--bitrates",
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
    let mut spatial_layers = None;
    let mut inter_layer_pred = 0;
    let mut aq_mode = 0;
    let mut noise_sensitivity = 0;
    let mut limit = 60;
    let mut keyframe_interval = 60;
    let mut warmups = 2;
    let mut pairs = 15;
    let mut rotations = 2;
    let mut max_ratio = 1.05;
    let mut max_load1 = None;
    let mut allow_anchor_hash_instability = false;
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
            "--spatial-layers" => spatial_layers = Some(parse_u32(flag, &value(&mut index)?)?),
            "--inter-layer-pred" => inter_layer_pred = parse_u32(flag, &value(&mut index)?)?,
            "--aq-mode" => aq_mode = parse_u32(flag, &value(&mut index)?)?,
            "--noise-sensitivity" => noise_sensitivity = parse_u32(flag, &value(&mut index)?)?,
            "--limit" => limit = parse_u32(flag, &value(&mut index)?)?,
            "--keyframe-interval" => keyframe_interval = parse_u32(flag, &value(&mut index)?)?,
            "--warmups" => warmups = parse_u32(flag, &value(&mut index)?)?,
            "--pairs" => pairs = parse_u32(flag, &value(&mut index)?)?,
            "--rotations" => rotations = parse_u32(flag, &value(&mut index)?)?,
            "--max-ratio" => max_ratio = parse_positive_f64(flag, &value(&mut index)?)?,
            "--max-load1" => max_load1 = Some(parse_positive_f64(flag, &value(&mut index)?)?),
            "--allow-anchor-hash-instability" => {
                if inline.is_some() {
                    return Err(format!("{flag} does not take a value"));
                }
                allow_anchor_hash_instability = true;
            }
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
        spatial_layers,
        inter_layer_pred,
        aq_mode,
        noise_sensitivity,
        limit,
        keyframe_interval,
        warmups,
        pairs,
        rotations,
        max_ratio,
        max_load1,
        allow_anchor_hash_instability,
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
    if args
        .spatial_layers
        .is_some_and(|layers| !matches!(layers, 2 | 3))
    {
        return Err("--spatial-layers must be 2 or 3".to_owned());
    }
    if !matches!(args.inter_layer_pred, 0 | 3) {
        return Err("--inter-layer-pred must be 0 or 3".to_owned());
    }
    if args.temporal_layers.is_some() && args.spatial_layers.is_some() {
        return Err("--temporal-layers and --spatial-layers are mutually exclusive".to_owned());
    }
    if args.spatial_layers.is_none() && args.inter_layer_pred != 0 {
        return Err("--inter-layer-pred requires --spatial-layers".to_owned());
    }
    if (matches!(args.anchor_kind, EncoderKind::SpatialSvc)
        || matches!(args.test_kind, EncoderKind::SpatialSvc))
        && args.spatial_layers.is_none()
    {
        return Err("spatial-svc encoders require --spatial-layers".to_owned());
    }
    if let Some(layers) = args.spatial_layers {
        spatial_layer_bitrates(args.target_bitrate, layers)?;
    }
    if args.allow_anchor_hash_instability && !matches!(args.anchor_kind, EncoderKind::SpatialSvc) {
        return Err(
            "--allow-anchor-hash-instability is accepted only for a spatial-svc anchor".to_owned(),
        );
    }
    if !matches!(args.aq_mode, 0 | 3) || args.noise_sensitivity > 6 {
        return Err("--aq-mode must be 0 or 3 and noise sensitivity must be in 0..=6".to_owned());
    }
    if args.spatial_layers.is_some() && args.noise_sensitivity != 0 {
        return Err("spatial SVC requires --noise-sensitivity 0".to_owned());
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
    // Neither SVC wrapper accepts a generic `--fps`, so both silently encode at
    // the Y4M header rate; a mismatch would make the arms incomparable.
    let wrapper_used = args.anchor_kind.is_svc_wrapper() || args.test_kind.is_svc_wrapper();
    if wrapper_used
        && u64::from(first.fps_num) * u64::from(args.fps_den)
            != u64::from(args.fps_num) * u64::from(first.fps_den)
    {
        return Err(format!(
            "the SVC wrapper uses the Y4M frame rate {}/{}, which does not match --fps {}/{}",
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
    // The spatial example takes its input as the final positional argument, so
    // it has to follow any role-specific extra arguments.
    let mut trailing_input = None;
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
            if let Some(layers) = args.spatial_layers {
                let (min_q, max_q) = rvp9_spatial_q_bounds(&args.end_usage);
                push(&mut result, format!("--spatial-layers={layers}"));
                push(
                    &mut result,
                    format!("--inter-layer-pred={}", args.inter_layer_pred),
                );
                push(&mut result, format!("--min-q={min_q}"));
                push(&mut result, format!("--max-q={max_q}"));
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
        EncoderKind::SpatialSvc => {
            let layers = args.spatial_layers.expect("validated spatial layers");
            let rates = spatial_layer_bitrates(args.target_bitrate, layers)
                .expect("validated spatial target bitrate");
            let scales = spatial_scale_factors(layers).expect("validated spatial layer count");
            let (min_q, max_q) = libvpx_spatial_q_bounds(&args.end_usage);
            let per_layer = |value: u32| {
                std::iter::repeat_n(value.to_string(), layers as usize)
                    .collect::<Vec<_>>()
                    .join(",")
            };
            push(&mut result, format!("--frames={}", args.limit));
            let mut output_flag = OsString::from("--output=");
            output_flag.push(output.as_os_str());
            result.push(output_flag);
            push(
                &mut result,
                format!("--target-bitrate={}", args.target_bitrate),
            );
            push(&mut result, format!("--speed={}", args.cpu_used));
            push(&mut result, format!("--threads={}", args.threads));
            // The example spells cyclic refresh `--aqmode=1`; vpxenc spells the
            // same mode `--aq-mode=3`.
            push(
                &mut result,
                format!("--aqmode={}", u32::from(args.aq_mode == 3)),
            );
            // The example takes the raw rate-control enum: 0 = VBR, 1 = CBR.
            push(
                &mut result,
                format!("--rc-end-usage={}", u32::from(args.end_usage == "cbr")),
            );
            push(&mut result, format!("--spatial-layers={layers}"));
            push(&mut result, "--temporal-layers=1".to_owned());
            push(&mut result, format!("--kf-dist={}", args.keyframe_interval));
            push(&mut result, "--lag-in-frames=0".to_owned());
            push(&mut result, format!("--scale-factors={scales}"));
            push(&mut result, format!("--bitrates={rates}"));
            push(&mut result, format!("--min-q={}", per_layer(min_q)));
            push(&mut result, format!("--max-q={}", per_layer(max_q)));
            // The pinned example's enum parser accepts the raw
            // VP9E_CONTENT_DEFAULT value (0); this is the same default mode as
            // the human-readable `default` name vpxenc takes.
            push(&mut result, "--tune-content=0".to_owned());
            push(
                &mut result,
                format!("--inter-layer-pred={}", args.inter_layer_pred),
            );
            trailing_input = Some(input.as_os_str().to_owned());
        }
    }
    result.extend(extra.iter().map(OsString::from));
    result.extend(trailing_input);
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

#[cfg(not(windows))]
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

#[cfg(windows)]
fn current_load1() -> Option<f64> {
    // Windows has no Unix load average. Sample aggregate processor time for a
    // full minute and convert it to normalized busy logical cores, so the
    // existing 0.75 ceiling keeps the same "less than one busy core" meaning.
    // Thirteen five-second PDH samples span sixty seconds after discarding the
    // first sample, whose baseline may predate this invocation.
    const SCRIPT: &str = r#"
$ErrorActionPreference = "Stop"
$values = @((Get-Counter -Counter '\Processor(_Total)\% Processor Time' -SampleInterval 5 -MaxSamples 13).CounterSamples | Select-Object -Skip 1 -ExpandProperty CookedValue)
if ($values.Count -ne 12) { exit 2 }
$meanPercent = ($values | Measure-Object -Average).Average
$busyCores = $meanPercent * [Environment]::ProcessorCount / 100.0
[Console]::Out.WriteLine($busyCores.ToString("0.######", [Globalization.CultureInfo]::InvariantCulture))
"#;
    let powershell = PathBuf::from(std::env::var_os("SystemRoot")?)
        .join("System32/WindowsPowerShell/v1.0/powershell.exe");
    let output = Command::new(powershell)
        .args(["-NoProfile", "-NonInteractive", "-Command", SCRIPT])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_load1(&String::from_utf8_lossy(&output.stdout)).filter(|load| {
        load.is_finite()
            && *load >= 0.0
            && *load <= std::thread::available_parallelism().map_or(f64::MAX, |n| n.get() as f64)
    })
}

fn load_with_retry<F>(mut read: F) -> Option<f64>
where
    F: FnMut() -> Option<f64>,
{
    read().or_else(read)
}

#[cfg(windows)]
fn load_metric_name() -> &'static str {
    "windows_60s_mean_busy_logical_cores"
}

#[cfg(not(windows))]
fn load_metric_name() -> &'static str {
    "unix_one_minute_load_average"
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
    let load = load_with_retry(current_load1).ok_or_else(|| {
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

fn summarize(
    clips: &[Clip],
    rows: &[RawRow],
    limit: u32,
    allow_anchor_hash_instability: bool,
) -> Result<Vec<ClipSummary>, String> {
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
        // Determinism is fail-closed. The one opt-in exception covers the
        // pinned threaded spatial example on the anchor arm; the test arm must
        // always be reproducible.
        let representative_hash = |arm: Arm,
                                   allow_instability: bool|
         -> Result<(String, usize), String> {
            let hashes = rows
                .iter()
                .filter(|row| row.clip == clip.name && row.arm == arm)
                .map(|row| row.sha256.as_str())
                .collect::<BTreeSet<_>>();
            if hashes.len() != 1 && !allow_instability {
                return Err(format!(
                    "{} {} emitted {} distinct IVF hashes{}",
                    clip.name,
                    arm.name(),
                    hashes.len(),
                    if arm == Arm::Anchor {
                        "; only a spatial-svc anchor may opt in with --allow-anchor-hash-instability"
                    } else {
                        ""
                    }
                ));
            }
            let first = hashes
                .first()
                .ok_or_else(|| format!("{} {} produced no outputs", clip.name, arm.name()))?;
            Ok(((*first).to_owned(), hashes.len()))
        };
        let (test_sha256, test_distinct_hashes) = representative_hash(Arm::Test, false)?;
        let (anchor_sha256, anchor_distinct_hashes) =
            representative_hash(Arm::Anchor, allow_anchor_hash_instability)?;
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
            test_sha256,
            anchor_sha256,
            test_distinct_hashes,
            anchor_distinct_hashes,
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
    // `*_sha256` is the first of `*_distinct_hash_count` observed output
    // hashes; a count above one is only ever possible on an anchor arm that
    // opted in with --allow-anchor-hash-instability.
    writeln!(writer, "clip,test_median,anchor_median,separate_ratio,paired_ratio,test_ms_per_frame,anchor_ms_per_frame,samples,test_sha256,anchor_sha256,test_distinct_hash_count,anchor_distinct_hash_count")
        .map_err(|error| error.to_string())?;
    for row in summaries {
        writeln!(
            writer,
            "{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{},{},{}",
            row.name,
            row.test_median,
            row.anchor_median,
            row.separate_ratio,
            row.paired_ratio,
            row.test_ms_per_frame,
            row.anchor_ms_per_frame,
            row.samples,
            row.test_sha256,
            row.anchor_sha256,
            row.test_distinct_hashes,
            row.anchor_distinct_hashes
        )
        .map_err(|error| error.to_string())?;
    }
    writeln!(
        writer,
        "corpus,,,{separate_geomean:.9},{paired_geomean:.9},,,,,,,"
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
    writeln!(writer, "spatial_layers={:?}", args.spatial_layers).map_err(|e| e.to_string())?;
    writeln!(writer, "inter_layer_pred={}", args.inter_layer_pred).map_err(|e| e.to_string())?;
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
    writeln!(
        writer,
        "allow_anchor_hash_instability={}",
        args.allow_anchor_hash_instability
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
    writeln!(writer, "host_load_metric={}", load_metric_name()).map_err(|e| e.to_string())?;
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
    let summaries = summarize(
        &clips,
        &rows,
        args.limit,
        args.allow_anchor_hash_instability,
    )?;
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
    for row in &summaries {
        if row.anchor_distinct_hashes > 1 {
            println!(
                "{}: anchor emitted {} distinct IVF hashes under --allow-anchor-hash-instability",
                row.name, row.anchor_distinct_hashes
            );
        }
    }
    println!(
        "corpus: separate {separate_geomean:.6}x, paired {paired_geomean:.6}x (limit {:.6}x)",
        args.max_ratio
    );
    Ok(separate_geomean <= args.max_ratio && paired_geomean <= args.max_ratio)
}

fn main() -> ExitCode {
    let argv = std::env::args().collect::<Vec<_>>();
    if help_requested(&argv) {
        print!("{}", usage());
        return ExitCode::SUCCESS;
    }
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
            spatial_layers: None,
            inter_layer_pred: 0,
            aq_mode: 0,
            noise_sensitivity: 0,
            limit: 60,
            keyframe_interval: 60,
            warmups: 2,
            pairs: 15,
            rotations: 2,
            max_ratio: 1.05,
            max_load1: None,
            allow_anchor_hash_instability: false,
            anchor_args: vec![],
            test_args: vec![],
        }
    }

    /// A spatial pair: no temporal layers, a 3-layer ladder, and a target the
    /// 1:4:16 allocation divides exactly.
    fn fake_spatial_args(end_usage: &str) -> Args {
        Args {
            anchor_kind: EncoderKind::SpatialSvc,
            test_kind: EncoderKind::Vpxenc,
            end_usage: end_usage.to_owned(),
            temporal_layers: None,
            spatial_layers: Some(3),
            inter_layer_pred: 3,
            target_bitrate: 210,
            aq_mode: 3,
            threads: 4,
            keyframe_interval: 100,
            ..fake_args(EncoderKind::SpatialSvc)
        }
    }

    fn strings(values: Vec<OsString>) -> Vec<String> {
        values
            .into_iter()
            .map(|value| value.into_string().unwrap())
            .collect()
    }

    /// Parses a minimal valid invocation plus `extra`. The work directory is
    /// never created, so every test may share the same missing path.
    fn parse_with(extra: &[&str]) -> Result<Args, String> {
        let executable = std::env::current_exe().unwrap();
        let work_dir = std::env::temp_dir().join(format!(
            "rvp9-encode-headtohead-parse-{}-must-not-exist",
            std::process::id()
        ));
        let mut argv = vec![
            "encode-headtohead".to_owned(),
            format!("--anchor-encoder={}", executable.display()),
            format!("--test-encoder={}", executable.display()),
            "--anchor-revision=anchor".to_owned(),
            "--test-revision=test".to_owned(),
            "--clip=clip=/not/read/by-parser.y4m".to_owned(),
            format!("--work-dir={}", work_dir.display()),
        ];
        argv.extend(extra.iter().map(|item| (*item).to_owned()));
        parse_args_from(&argv)
    }

    #[test]
    fn explicit_help_is_a_clean_capability_probe() {
        let argv =
            |items: &[&str]| -> Vec<String> { items.iter().map(ToString::to_string).collect() };
        // Both spellings answer on an argv that carries none of the required
        // arguments, which is exactly what a capability probe supplies.
        assert!(help_requested(&argv(&["encode-headtohead", "--help"])));
        assert!(help_requested(&argv(&["encode-headtohead", "-h"])));
        assert!(!help_requested(&argv(&["encode-headtohead"])));
        // The parser itself refuses that same argv, so `main` answering the
        // probe first is what makes `--help` exit 0.
        assert_eq!(
            parse_args_from(&argv(&["encode-headtohead", "--help"])).unwrap_err(),
            usage()
        );

        let help = usage();
        assert!(help.starts_with("Usage: encode-headtohead"));
        assert!(
            !help.lines().any(|line| line.trim_start().starts_with('+')),
            "usage must not carry diff artifacts:\n{help}"
        );
        for flag in [
            "--anchor-encoder",
            "--test-encoder",
            "--anchor-kind",
            "--test-kind",
            "--anchor-revision",
            "--test-revision",
            "--clip",
            "--work-dir",
            "--cpu-used",
            "--target-bitrate",
            "--end-usage",
            "--fps",
            "--threads",
            "--tile-columns",
            "--temporal-layers",
            "--spatial-layers",
            "--inter-layer-pred",
            "--aq-mode",
            "--noise-sensitivity",
            "--limit",
            "--keyframe-interval",
            "--warmups",
            "--pairs",
            "--rotations",
            "--max-ratio",
            "--max-load1",
            "--allow-anchor-hash-instability",
            "--anchor-arg",
            "--test-arg",
        ] {
            assert!(help.contains(flag), "usage must advertise {flag}");
        }
    }

    fn sample_clip() -> Clip {
        Clip {
            name: "c".into(),
            path: PathBuf::from("c.y4m"),
            header: Y4mHeader {
                width: 8,
                height: 8,
                fps_num: 30,
                fps_den: 1,
            },
            sha256: "input".into(),
        }
    }

    fn raw_row(rotation: u32, iteration: &str, arm: Arm, seconds: f64, hash: &str) -> RawRow {
        RawRow {
            clip: "c".into(),
            rotation,
            iteration: iteration.into(),
            arm,
            order: 0,
            seconds,
            frames: 60,
            sha256: hash.into(),
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_idle_guard_produces_a_normalized_load_measurement() {
        let load = current_load1().expect("Windows host load should be measurable");
        assert!(load.is_finite());
        assert!(load >= 0.0);
        assert!(load <= std::thread::available_parallelism().unwrap().get() as f64);
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
    fn spatial_svc_cbr_command_pins_the_libvpx_example_surface() {
        let args = fake_spatial_args("cbr");
        let command = strings(command_args(
            EncoderKind::SpatialSvc,
            &args,
            Path::new("input.y4m"),
            Path::new("output.ivf"),
            &["--extra".into()],
        ));
        for expected in [
            "--frames=60",
            "--output=output.ivf",
            "--target-bitrate=210",
            "--speed=8",
            "--threads=4",
            "--aqmode=1",
            "--rc-end-usage=1",
            "--spatial-layers=3",
            "--temporal-layers=1",
            "--kf-dist=100",
            "--lag-in-frames=0",
            "--scale-factors=1/4,1/2,1/1",
            "--bitrates=10,40,160",
            "--min-q=2,2,2",
            "--max-q=56,56,56",
            "--tune-content=0",
            "--inter-layer-pred=3",
        ] {
            assert!(
                command.contains(&expected.to_owned()),
                "missing anchor {expected}"
            );
        }
        // The example takes its input last, after any role-specific extras.
        assert_eq!(&command[command.len() - 2..], ["--extra", "input.y4m"]);
    }

    #[test]
    fn spatial_svc_cbr_command_pins_the_matched_rvp9_surface() {
        let args = fake_spatial_args("cbr");
        let command = strings(command_args(
            EncoderKind::Vpxenc,
            &args,
            Path::new("input.y4m"),
            Path::new("output.ivf"),
            &[],
        ));
        for expected in [
            "--codec=vp9",
            "--rt",
            "--cpu-used=8",
            "--end-usage=cbr",
            "--target-bitrate=210",
            "--fps=30/1",
            "--threads=4",
            "--aq-mode=3",
            "--spatial-layers=3",
            "--inter-layer-pred=3",
            "--min-q=8",
            "--max-q=224",
            "--kf-max-dist=100",
        ] {
            assert!(
                command.contains(&expected.to_owned()),
                "missing test {expected}"
            );
        }
        assert!(
            !command
                .iter()
                .any(|argument| argument.starts_with("--temporal-layers"))
        );
        assert_eq!(command.last().unwrap(), "input.y4m");
    }

    #[test]
    fn spatial_svc_vbr_widens_the_q_bounds_on_both_arms() {
        let args = fake_spatial_args("vbr");
        let anchor = strings(command_args(
            EncoderKind::SpatialSvc,
            &args,
            Path::new("input.y4m"),
            Path::new("output.ivf"),
            &[],
        ));
        let test = strings(command_args(
            EncoderKind::Vpxenc,
            &args,
            Path::new("input.y4m"),
            Path::new("output.ivf"),
            &[],
        ));
        for expected in ["--rc-end-usage=0", "--min-q=1,1,1", "--max-q=63,63,63"] {
            assert!(
                anchor.contains(&expected.to_owned()),
                "missing anchor {expected}"
            );
        }
        for expected in ["--end-usage=vbr", "--min-q=4", "--max-q=255"] {
            assert!(
                test.contains(&expected.to_owned()),
                "missing test {expected}"
            );
        }
    }

    #[test]
    fn spatial_targets_must_split_exactly_across_layers() {
        assert_eq!(spatial_layer_bitrates(210, 3).unwrap(), "10,40,160");
        assert_eq!(spatial_layer_bitrates(1500, 2).unwrap(), "300,1200");
        assert!(spatial_layer_bitrates(1500, 3).is_err());
        assert!(spatial_layer_bitrates(211, 2).is_err());
        assert_eq!(spatial_scale_factors(2).unwrap(), "1/2,1/1");
        assert!(spatial_scale_factors(4).is_err());
        let spatial = ["--anchor-kind=spatial-svc", "--test-kind=vpxenc"];
        assert!(
            parse_with(
                &[
                    spatial.as_slice(),
                    &["--spatial-layers=3", "--target-bitrate=1500"]
                ]
                .concat()
            )
            .is_err()
        );
        assert!(
            parse_with(
                &[
                    spatial.as_slice(),
                    &["--spatial-layers=3", "--target-bitrate=1512"]
                ]
                .concat()
            )
            .is_ok()
        );
    }

    #[test]
    fn topologies_are_mutually_exclusive_and_matched_to_the_wrapper_kind() {
        let spatial = [
            "--anchor-kind=spatial-svc",
            "--test-kind=vpxenc",
            "--spatial-layers=2",
        ];
        let parsed = parse_with(&spatial).unwrap();
        assert_eq!(parsed.spatial_layers, Some(2));
        assert_eq!(parsed.temporal_layers, None);
        assert_eq!(parsed.inter_layer_pred, 0);
        for rejected in [
            // Temporal and spatial layers cannot be combined.
            [spatial.as_slice(), &["--temporal-layers=3"]].concat(),
            // Each wrapper kind requires its own topology.
            vec!["--anchor-kind=spatial-svc", "--test-kind=vpxenc"],
            vec![
                "--anchor-kind=temporal-svc",
                "--test-kind=vpxenc",
                "--spatial-layers=2",
            ],
            // Out-of-range spatial controls.
            vec![
                "--anchor-kind=vpxenc",
                "--test-kind=vpxenc",
                "--spatial-layers=4",
            ],
            [spatial.as_slice(), &["--inter-layer-pred=1"]].concat(),
            vec![
                "--anchor-kind=vpxenc",
                "--test-kind=vpxenc",
                "--inter-layer-pred=3",
            ],
            [spatial.as_slice(), &["--noise-sensitivity=1"]].concat(),
        ] {
            assert!(parse_with(&rejected).is_err(), "accepted {rejected:?}");
        }
    }

    #[test]
    fn anchor_hash_instability_is_a_narrow_opt_in() {
        let spatial_anchor = [
            "--anchor-kind=spatial-svc",
            "--test-kind=vpxenc",
            "--spatial-layers=2",
            "--allow-anchor-hash-instability",
        ];
        assert!(
            parse_with(&spatial_anchor)
                .unwrap()
                .allow_anchor_hash_instability
        );
        for rejected in [
            // A spatial-svc test arm does not license anchor instability.
            vec![
                "--anchor-kind=vpxenc",
                "--test-kind=spatial-svc",
                "--spatial-layers=2",
                "--allow-anchor-hash-instability",
            ],
            vec![
                "--anchor-kind=temporal-svc",
                "--test-kind=vpxenc",
                "--temporal-layers=3",
                "--allow-anchor-hash-instability",
            ],
            // The flag takes no value.
            [
                spatial_anchor.as_slice(),
                &["--allow-anchor-hash-instability=false"],
            ]
            .concat(),
        ] {
            assert!(parse_with(&rejected).is_err(), "accepted {rejected:?}");
        }

        let clip = sample_clip();
        let mut rows = vec![
            raw_row(1, "1", Arm::Test, 2.0, "t"),
            raw_row(1, "1", Arm::Anchor, 1.0, "a1"),
            raw_row(2, "1", Arm::Test, 4.0, "t"),
            raw_row(2, "1", Arm::Anchor, 2.0, "a2"),
        ];
        assert!(summarize(std::slice::from_ref(&clip), &rows, 60, false).is_err());
        let summary = summarize(std::slice::from_ref(&clip), &rows, 60, true).unwrap();
        assert_eq!(summary[0].anchor_distinct_hashes, 2);
        assert_eq!(summary[0].anchor_sha256, "a1");
        assert_eq!(summary[0].test_distinct_hashes, 1);
        assert_eq!(summary[0].paired_ratio, 2.0);
        // The opt-in never extends to the test arm.
        rows[0].sha256 = "t2".into();
        assert!(summarize(&[clip], &rows, 60, true).is_err());
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
        for spatial in [
            "--spatial-layers=2",
            "--inter-layer-pred=3",
            "--min-q=4",
            "--max-q=255",
            "--frames=10",
            "--speed=8",
            "--aqmode=1",
            "--rc-end-usage=1",
            "--kf-dist=60",
            "--scale-factors=1/2,1/1",
            "--bitrates=300,1200",
        ] {
            assert!(
                validate_extra_args("anchor", &[spatial.into()]).is_err(),
                "{spatial} must be refused"
            );
        }
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
    fn load_retry_recovers_after_one_transient_failure() {
        let mut calls = 0;
        let load = load_with_retry(|| {
            calls += 1;
            (calls == 2).then_some(0.5)
        });
        assert_eq!(load, Some(0.5));
        assert_eq!(calls, 2);
    }

    #[test]
    fn load_retry_fails_closed_after_two_failures() {
        let mut calls = 0;
        let load = load_with_retry(|| {
            calls += 1;
            None
        });
        assert_eq!(load, None);
        assert_eq!(calls, 2);
    }

    #[test]
    fn load_retry_does_not_repeat_a_valid_reading() {
        let mut calls = 0;
        let load = load_with_retry(|| {
            calls += 1;
            Some(0.5)
        });
        assert_eq!(load, Some(0.5));
        assert_eq!(calls, 1);
    }

    #[test]
    fn summary_uses_paired_ratios_and_rejects_hash_instability() {
        let clip = sample_clip();
        let mut rows = vec![
            raw_row(1, "1", Arm::Test, 2.0, "t"),
            raw_row(1, "1", Arm::Anchor, 1.0, "a"),
            raw_row(2, "1", Arm::Test, 4.0, "t"),
            raw_row(2, "1", Arm::Anchor, 2.0, "a"),
        ];
        let summary = summarize(std::slice::from_ref(&clip), &rows, 60, false).unwrap();
        assert_eq!(summary[0].separate_ratio, 2.0);
        assert_eq!(summary[0].paired_ratio, 2.0);
        assert_eq!(summary[0].test_distinct_hashes, 1);
        assert_eq!(summary[0].anchor_distinct_hashes, 1);
        rows[0].sha256 = "changed".into();
        assert!(summarize(&[clip], &rows, 60, false).is_err());
    }
}
