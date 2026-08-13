//! Per-operating-point BD-rate measurement for temporal or spatial SVC.
//!
//! Temporal mode uses `tools/libvpx_temporal_svc_reference`, which exposes the
//! pinned libvpx encoder core's built-in 0101 / 0212 modes. Spatial mode uses
//! `vp9_spatial_svc_encoder` and rebuilds cumulative low-to-high layer
//! packets. Every operating point is decoded through the same decoder and
//! compared with a topology-matched Y4M reference.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rvp9_core::bitstream::frame::{ChromaFormat, InterpFilter};
use rvp9_core::bitstream::{Superframe, SuperframeBuilder, parse_superframe_index};
use rvp9_encoder::scale_source::scale_i420_2x;
use rvp9_harness::bdrate::{bdrate, psnr_delta_summary};
use rvp9_harness::container::{IvfHeader, read_ivf_packets, write_ivf};
use rvp9_harness::y4m::{Y4mReader, parse_header, write_frame, write_header};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SvcKind {
    Temporal,
    Spatial,
}

impl SvcKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "temporal" => Ok(Self::Temporal),
            "spatial" => Ok(Self::Spatial),
            _ => Err(format!(
                "--svc-kind must be temporal or spatial, got {value}"
            )),
        }
    }

    fn layer_label(self, layer: usize) -> String {
        match self {
            Self::Temporal => format!("TL{layer}"),
            Self::Spatial => format!("SL{layer}"),
        }
    }
}

#[derive(Debug)]
struct Args {
    anchor_encoder: PathBuf,
    test_encoder: PathBuf,
    decoder: PathBuf,
    input: PathBuf,
    reference_input: PathBuf,
    cpu_used: u32,
    bitrates_kbps: Vec<u32>,
    end_usage: String,
    svc_kind: SvcKind,
    threads: u32,
    tile_columns: u32,
    temporal_layers: u32,
    spatial_layers: u32,
    inter_layer_pred: u32,
    aq_mode: u32,
    noise_sensitivity: u32,
    limit: usize,
    keyframe_interval: u32,
    max_bdrate_pct: f64,
    anchor_args: Vec<String>,
    test_args: Vec<String>,
    work_dir: Option<PathBuf>,
    csv: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
struct RdPoint {
    bitrate_kbps: f64,
    psnr_db: f64,
}

#[derive(Debug)]
struct EncodeOutcome {
    path: PathBuf,
    wall_ms_per_frame: f64,
}

#[derive(Debug)]
struct OperatingPoint {
    path: PathBuf,
    payload_bytes: u64,
    frame_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct VideoInfo {
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
}

#[derive(Clone, Copy)]
enum EncoderRole {
    Anchor,
    Test,
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
        .map_err(|_| format!("{flag} requires an unsigned integer, got {value}"))
}

fn spatial_weight_sum(spatial_layers: u32) -> Result<u32, String> {
    match spatial_layers {
        2 => Ok(1 + 4),
        3 => Ok(1 + 4 + 16),
        other => Err(format!("--spatial-layers must be 2 or 3, got {other}")),
    }
}

fn spatial_layer_bitrates(target_kbps: u32, spatial_layers: u32) -> Result<String, String> {
    let weights: &[u32] = match spatial_layers {
        2 => &[1, 4],
        3 => &[1, 4, 16],
        other => {
            return Err(format!("--spatial-layers must be 2 or 3, got {other}"));
        }
    };
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
        other => Err(format!("--spatial-layers must be 2 or 3, got {other}")),
    }
}

fn parse_args_from(argv: &[String]) -> Result<Args, String> {
    let mut anchor_encoder = None;
    let mut test_encoder = None;
    let mut decoder = None;
    let mut input = None;
    let mut reference_input = None;
    let mut cpu_used = 7;
    let mut bitrates_kbps = vec![500, 750, 1000, 1500, 2500, 4000];
    let mut end_usage = "vbr".to_owned();
    let mut svc_kind = SvcKind::Temporal;
    let mut threads = 1;
    let mut tile_columns = 0;
    let mut temporal_layers = 3;
    let mut temporal_layers_explicit = false;
    let mut spatial_layers = 2;
    let mut inter_layer_pred = 0;
    let mut aq_mode = 0;
    let mut noise_sensitivity = 0;
    let mut limit = 60usize;
    let mut keyframe_interval = 60;
    let mut max_bdrate_pct = 0.5;
    let mut anchor_args = Vec::new();
    let mut test_args = Vec::new();
    let mut work_dir = None;
    let mut csv = None;

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
            "--decoder" => decoder = Some(PathBuf::from(value(&mut index)?)),
            "--input" => input = Some(PathBuf::from(value(&mut index)?)),
            "--reference-input" => reference_input = Some(PathBuf::from(value(&mut index)?)),
            "--cpu-used" => cpu_used = parse_u32(flag, &value(&mut index)?)?,
            "--bitrates" => {
                bitrates_kbps = value(&mut index)?
                    .split(',')
                    .map(|item| parse_u32(flag, item))
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--end-usage" => end_usage = value(&mut index)?,
            "--svc-kind" => svc_kind = SvcKind::parse(&value(&mut index)?)?,
            "--threads" => threads = parse_u32(flag, &value(&mut index)?)?,
            "--tile-columns" => tile_columns = parse_u32(flag, &value(&mut index)?)?,
            "--temporal-layers" => {
                temporal_layers = parse_u32(flag, &value(&mut index)?)?;
                temporal_layers_explicit = true;
            }
            "--spatial-layers" => {
                spatial_layers = parse_u32(flag, &value(&mut index)?)?;
            }
            "--inter-layer-pred" => {
                inter_layer_pred = parse_u32(flag, &value(&mut index)?)?;
            }
            "--aq-mode" => aq_mode = parse_u32(flag, &value(&mut index)?)?,
            "--noise-sensitivity" => {
                noise_sensitivity = parse_u32(flag, &value(&mut index)?)?;
            }
            "--limit" => {
                limit = value(&mut index)?
                    .parse()
                    .map_err(|_| "--limit requires an unsigned integer".to_owned())?;
            }
            "--keyframe-interval" => {
                keyframe_interval = parse_u32(flag, &value(&mut index)?)?;
            }
            "--max-bdrate-pct" => {
                max_bdrate_pct = value(&mut index)?
                    .parse()
                    .map_err(|_| "--max-bdrate-pct requires a number".to_owned())?;
            }
            "--anchor-arg" => anchor_args.push(value(&mut index)?),
            "--test-arg" => test_args.push(value(&mut index)?),
            "--work-dir" => work_dir = Some(PathBuf::from(value(&mut index)?)),
            "--csv" => csv = Some(PathBuf::from(value(&mut index)?)),
            "--help" | "-h" => return Err(usage()),
            _ => return Err(format!("unknown argument: {arg}\n\n{}", usage())),
        }
        index += 1;
    }

    let input = input.ok_or_else(|| "--input is required".to_owned())?;
    if svc_kind == SvcKind::Spatial && !temporal_layers_explicit {
        temporal_layers = 1;
    }
    let args = Args {
        anchor_encoder: anchor_encoder.ok_or_else(|| "--anchor-encoder is required".to_owned())?,
        test_encoder: test_encoder.ok_or_else(|| "--test-encoder is required".to_owned())?,
        decoder: decoder.ok_or_else(|| "--decoder is required".to_owned())?,
        reference_input: reference_input.unwrap_or_else(|| input.clone()),
        input,
        cpu_used,
        bitrates_kbps,
        end_usage,
        svc_kind,
        threads,
        tile_columns,
        temporal_layers,
        spatial_layers,
        inter_layer_pred,
        aq_mode,
        noise_sensitivity,
        limit,
        keyframe_interval,
        max_bdrate_pct,
        anchor_args,
        test_args,
        work_dir,
        csv,
    };
    validate_args(&args)?;
    Ok(args)
}

fn usage() -> String {
    "Usage: bdrate-svc --anchor-encoder PATH --test-encoder PATH \\\n+\n  --decoder PATH --input CLIP.y4m [options]\n\n\
Options:\n  --reference-input CLEAN.y4m\n  --cpu-used {6,7,8}\n  \
--bitrates 500,750,1000,1500,2500,4000\n  --end-usage {cbr,vbr}\n  \
--svc-kind {temporal,spatial}\n  --spatial-layers {2,3} --inter-layer-pred {0,3}\n  \
--threads N --tile-columns N\n  --temporal-layers {1,2,3}\n  --aq-mode {0,3}\n  \
--noise-sensitivity 0..6\n  --limit N --keyframe-interval N\n  \
--max-bdrate-pct 0.5\n  --work-dir EMPTY_DIR --csv PATH\n"
        .to_owned()
}

fn validate_args(args: &Args) -> Result<(), String> {
    if !(6..=8).contains(&args.cpu_used) {
        return Err("--cpu-used must be 6, 7, or 8".to_owned());
    }
    if !matches!(args.end_usage.as_str(), "cbr" | "vbr") {
        return Err("--end-usage must be cbr or vbr".to_owned());
    }
    if args.threads == 0 {
        return Err("--threads must be greater than zero".to_owned());
    }
    if args.tile_columns > 6 {
        return Err("--tile-columns must be in 0..=6".to_owned());
    }
    match args.svc_kind {
        SvcKind::Temporal if !matches!(args.temporal_layers, 2 | 3) => {
            return Err("--temporal-layers must be 2 or 3 for temporal SVC".to_owned());
        }
        SvcKind::Spatial if args.temporal_layers != 1 => {
            return Err("spatial SVC requires --temporal-layers 1".to_owned());
        }
        _ => {}
    }
    if !matches!(args.spatial_layers, 2 | 3) {
        return Err("--spatial-layers must be 2 or 3".to_owned());
    }
    if !matches!(args.inter_layer_pred, 0 | 3) {
        return Err("--inter-layer-pred must be 0 or 3".to_owned());
    }
    if !matches!(args.aq_mode, 0 | 3) {
        return Err("--aq-mode must be 0 or 3".to_owned());
    }
    if args.noise_sensitivity > 6 {
        return Err("--noise-sensitivity must be in 0..=6".to_owned());
    }
    if args.limit == 0 || args.keyframe_interval == 0 {
        return Err("--limit and --keyframe-interval must be greater than zero".to_owned());
    }
    if args.bitrates_kbps.len() < 4 || args.bitrates_kbps.contains(&0) {
        return Err("--bitrates requires at least four positive points".to_owned());
    }
    if !args.max_bdrate_pct.is_finite() {
        return Err("--max-bdrate-pct must be finite".to_owned());
    }
    if args.svc_kind == SvcKind::Spatial {
        if args.noise_sensitivity != 0 {
            return Err("spatial SVC requires --noise-sensitivity 0".to_owned());
        }
        let weight_sum = spatial_weight_sum(args.spatial_layers)?;
        for &target in &args.bitrates_kbps {
            if target % weight_sum != 0 {
                return Err(format!(
                    "spatial target bitrate {target} kbps must be divisible by {weight_sum}"
                ));
            }
        }
    }
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    parse_args_from(&std::env::args().collect::<Vec<_>>())
}

fn temporal_id(frame_index: usize, temporal_layers: u32) -> u32 {
    if temporal_layers == 2 {
        (frame_index & 1) as u32
    } else {
        match frame_index & 3 {
            0 => 0,
            2 => 1,
            _ => 2,
        }
    }
}

fn temporal_decimator(temporal_layers: u32, operating_point: u32) -> u32 {
    match (temporal_layers, operating_point) {
        (2, 0) => 2,
        (2, 1) => 1,
        (3, 0) => 4,
        (3, 1) => 2,
        (3, 2) => 1,
        _ => unreachable!("validated temporal layer and operating point"),
    }
}

fn source_info(path: &Path) -> Result<VideoInfo, String> {
    let mut file =
        File::open(path).map_err(|error| format!("opening {}: {error}", path.display()))?;
    let header =
        parse_header(&mut file).map_err(|error| format!("parsing {}: {error}", path.display()))?;
    Ok(VideoInfo {
        width: header.w,
        height: header.h,
        fps_num: header.fps_num,
        fps_den: header.fps_den,
    })
}

fn count_source_frames(path: &Path, info: VideoInfo, limit: usize) -> Result<usize, String> {
    let mut reader = Y4mReader::open(path, info.width, info.height, limit)
        .map_err(|error| format!("opening {}: {error}", path.display()))?;
    let mut count = 0;
    while reader
        .next_frame()
        .map_err(|error| format!("reading {}: {error}", path.display()))?
        .is_some()
    {
        count += 1;
    }
    if count == 0 {
        return Err("input contains no frames".to_owned());
    }
    Ok(count)
}

fn prepare_work_dir(requested: Option<&Path>) -> Result<PathBuf, String> {
    let path = if let Some(path) = requested {
        path.to_path_buf()
    } else {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock: {error}"))?
            .as_nanos();
        std::env::temp_dir().join(format!("rvp9-bdrate-svc-{}-{stamp}", std::process::id()))
    };
    if path.exists() {
        let mut entries = std::fs::read_dir(&path)
            .map_err(|error| format!("reading {}: {error}", path.display()))?;
        if entries.next().is_some() {
            return Err(format!("--work-dir must be empty: {}", path.display()));
        }
    } else {
        std::fs::create_dir_all(&path)
            .map_err(|error| format!("creating {}: {error}", path.display()))?;
    }
    Ok(path)
}

#[allow(clippy::too_many_arguments)]
fn encoder_command(
    role: EncoderRole,
    encoder: &Path,
    input: &Path,
    output: &Path,
    bitrate_kbps: u32,
    args: &Args,
    info: VideoInfo,
    expected_frames: usize,
) -> Command {
    let mut command = Command::new(encoder);
    match role {
        EncoderRole::Anchor => {
            match args.svc_kind {
                SvcKind::Temporal => {
                    command.args([
                        "--input".to_owned(),
                        input.display().to_string(),
                        "--output".to_owned(),
                        output.display().to_string(),
                        "--target-bitrate".to_owned(),
                        bitrate_kbps.to_string(),
                        "--cpu-used".to_owned(),
                        args.cpu_used.to_string(),
                        "--end-usage".to_owned(),
                        args.end_usage.clone(),
                        "--threads".to_owned(),
                        args.threads.to_string(),
                        "--tile-columns".to_owned(),
                        args.tile_columns.to_string(),
                        "--temporal-layers".to_owned(),
                        args.temporal_layers.to_string(),
                        "--aq-mode".to_owned(),
                        args.aq_mode.to_string(),
                        "--noise-sensitivity".to_owned(),
                        args.noise_sensitivity.to_string(),
                        "--limit".to_owned(),
                        args.limit.to_string(),
                        "--keyframe-interval".to_owned(),
                        args.keyframe_interval.to_string(),
                    ]);
                }
                SvcKind::Spatial => {
                    let rates = spatial_layer_bitrates(bitrate_kbps, args.spatial_layers)
                        .expect("validated spatial target bitrate");
                    let scales = spatial_scale_factors(args.spatial_layers)
                        .expect("validated spatial layer count");
                    let (min_q, max_q) = if args.end_usage == "cbr" {
                        (2, 56)
                    } else {
                        (1, 63)
                    };
                    let min_qs =
                        std::iter::repeat_n(min_q.to_string(), args.spatial_layers as usize)
                            .collect::<Vec<_>>()
                            .join(",");
                    let max_qs =
                        std::iter::repeat_n(max_q.to_string(), args.spatial_layers as usize)
                            .collect::<Vec<_>>()
                            .join(",");
                    command.args([
                        format!("--frames={expected_frames}"),
                        format!("--output={}", output.display()),
                        format!("--target-bitrate={bitrate_kbps}"),
                        format!("--speed={}", args.cpu_used),
                        format!("--threads={}", args.threads),
                        format!("--aqmode={}", if args.aq_mode == 3 { "1" } else { "0" }),
                        format!(
                            "--rc-end-usage={}",
                            if args.end_usage == "cbr" { "1" } else { "0" }
                        ),
                        format!("--spatial-layers={}", args.spatial_layers),
                        "--temporal-layers=1".to_owned(),
                        format!("--kf-dist={}", args.keyframe_interval),
                        "--lag-in-frames=0".to_owned(),
                        format!("--scale-factors={scales}"),
                        format!("--bitrates={rates}"),
                        format!("--min-q={min_qs}"),
                        format!("--max-q={max_qs}"),
                        // The pinned example's enum parser accepts the raw
                        // VP9E_CONTENT_DEFAULT value (0); this is the same
                        // default mode as the human-readable `default` name.
                        "--tune-content=0".to_owned(),
                        format!("--inter-layer-pred={}", args.inter_layer_pred),
                    ]);
                }
            }
            command.args(&args.anchor_args);
            if args.svc_kind == SvcKind::Spatial {
                command.arg(input);
            }
        }
        EncoderRole::Test => {
            command.args([
                "--codec=vp9".to_owned(),
                "--passes=1".to_owned(),
                "--rt".to_owned(),
                format!("--cpu-used={}", args.cpu_used),
                format!("--end-usage={}", args.end_usage),
                format!("--target-bitrate={bitrate_kbps}"),
                format!("--fps={}/{}", info.fps_num, info.fps_den),
                format!("--threads={}", args.threads),
                format!("--tile-columns={}", args.tile_columns),
                "--lag-in-frames=0".to_owned(),
                "--tune-content=default".to_owned(),
                format!("--aq-mode={}", args.aq_mode),
                format!("--noise-sensitivity={}", args.noise_sensitivity),
                format!("--temporal-layers={}", args.temporal_layers),
                format!("--limit={}", args.limit),
                format!("--kf-max-dist={}", args.keyframe_interval),
                "--quiet".to_owned(),
                "--ivf".to_owned(),
            ]);
            if args.svc_kind == SvcKind::Spatial {
                command.args([
                    format!("--spatial-layers={}", args.spatial_layers),
                    format!("--inter-layer-pred={}", args.inter_layer_pred),
                    if args.end_usage == "cbr" {
                        "--min-q=8".to_owned()
                    } else {
                        "--min-q=4".to_owned()
                    },
                    if args.end_usage == "cbr" {
                        "--max-q=224".to_owned()
                    } else {
                        "--max-q=255".to_owned()
                    },
                ]);
            }
            command
                .args(&args.test_args)
                .arg("-o")
                .arg(output)
                .arg(input);
        }
    }
    command
}

fn encode(
    role: EncoderRole,
    encoder: &Path,
    output: PathBuf,
    bitrate_kbps: u32,
    args: &Args,
    info: VideoInfo,
    expected_frames: usize,
) -> Result<EncodeOutcome, String> {
    let start = Instant::now();
    let status = encoder_command(
        role,
        encoder,
        &args.input,
        &output,
        bitrate_kbps,
        args,
        info,
        expected_frames,
    )
    .status()
    .map_err(|error| format!("launching {}: {error}", encoder.display()))?;
    let elapsed = start.elapsed();
    if !status.success() {
        return Err(format!(
            "encoder {} failed at {bitrate_kbps} kbps with {status}",
            encoder.display()
        ));
    }
    Ok(EncodeOutcome {
        path: output,
        wall_ms_per_frame: elapsed.as_secs_f64() * 1000.0 / expected_frames as f64,
    })
}

fn extract_operating_points(
    full_ivf: &Path,
    prefix: &str,
    work_dir: &Path,
    info: VideoInfo,
    temporal_layers: u32,
    expected_frames: usize,
) -> Result<Vec<OperatingPoint>, String> {
    let data = std::fs::read(full_ivf)
        .map_err(|error| format!("reading {}: {error}", full_ivf.display()))?;
    let packet_ranges = read_ivf_packets(&data)
        .map_err(|error| format!("parsing {}: {error}", full_ivf.display()))?;
    if packet_ranges.len() != expected_frames {
        return Err(format!(
            "{} has {} packets; expected {expected_frames}",
            full_ivf.display(),
            packet_ranges.len()
        ));
    }
    let mut operating_points = Vec::new();
    for operating_point in 0..temporal_layers {
        let retained: Vec<&[u8]> = packet_ranges
            .iter()
            .enumerate()
            .filter(|(index, _)| temporal_id(*index, temporal_layers) <= operating_point)
            .map(|(_, range)| &data[range.start..range.end])
            .collect();
        let payload_bytes = retained.iter().try_fold(0u64, |sum, packet| {
            u64::try_from(packet.len())
                .ok()
                .and_then(|len| sum.checked_add(len))
                .ok_or_else(|| "payload byte count overflow".to_owned())
        })?;
        let decimator = temporal_decimator(temporal_layers, operating_point);
        let fps_den = info
            .fps_den
            .checked_mul(decimator)
            .ok_or_else(|| "operating-point framerate overflow".to_owned())?;
        let path = work_dir.join(format!("{prefix}_tl{operating_point}.ivf"));
        let mut output = BufWriter::new(
            File::create(&path).map_err(|error| format!("creating {}: {error}", path.display()))?,
        );
        let header = IvfHeader::vp9(
            u16::try_from(info.width).map_err(|_| "width exceeds IVF".to_owned())?,
            u16::try_from(info.height).map_err(|_| "height exceeds IVF".to_owned())?,
            info.fps_num,
            fps_den,
        );
        write_ivf(&mut output, &header, &retained)
            .map_err(|error| format!("writing {}: {error}", path.display()))?;
        output
            .flush()
            .map_err(|error| format!("flushing {}: {error}", path.display()))?;
        operating_points.push(OperatingPoint {
            path,
            payload_bytes,
            frame_count: retained.len(),
        });
    }
    Ok(operating_points)
}

fn extract_spatial_operating_points(
    full_ivf: &Path,
    prefix: &str,
    work_dir: &Path,
    info: VideoInfo,
    spatial_layers: u32,
    expected_frames: usize,
) -> Result<Vec<OperatingPoint>, String> {
    let data = std::fs::read(full_ivf)
        .map_err(|error| format!("reading {}: {error}", full_ivf.display()))?;
    let packet_ranges = read_ivf_packets(&data)
        .map_err(|error| format!("parsing {}: {error}", full_ivf.display()))?;
    if packet_ranges.len() != expected_frames {
        return Err(format!(
            "{} has {} packets; expected {expected_frames}",
            full_ivf.display(),
            packet_ranges.len()
        ));
    }
    let layer_info = spatial_layer_info(info, spatial_layers)?;
    let mut components: Vec<Vec<Vec<u8>>> = (0..spatial_layers as usize)
        .map(|_| Vec::with_capacity(expected_frames))
        .collect();
    for (frame_index, range) in packet_ranges.iter().enumerate() {
        let packet = &data[range.start..range.end];
        let superframe = parse_superframe_index(packet).map_err(|error| {
            format!(
                "{} packet {frame_index} has an invalid spatial superframe index: {error}",
                full_ivf.display()
            )
        })?;
        let Superframe::Multi {
            sizes,
            count,
            index_size,
        } = superframe
        else {
            return Err(format!(
                "{} packet {frame_index} is not a {spatial_layers}-component spatial superframe",
                full_ivf.display()
            ));
        };
        if count != spatial_layers as usize {
            return Err(format!(
                "{} packet {frame_index} has {count} spatial components; expected {spatial_layers}",
                full_ivf.display()
            ));
        }
        let body_len = packet.len().checked_sub(index_size).ok_or_else(|| {
            format!(
                "{} packet {frame_index} has an invalid spatial index size",
                full_ivf.display()
            )
        })?;
        let mut offset = 0usize;
        for layer in 0..count {
            let size = usize::try_from(sizes[layer]).map_err(|_| {
                format!(
                    "{} packet {frame_index} SL{layer} size does not fit usize",
                    full_ivf.display()
                )
            })?;
            let end = offset.checked_add(size).ok_or_else(|| {
                format!(
                    "{} packet {frame_index} spatial component sizes overflow",
                    full_ivf.display()
                )
            })?;
            if end > body_len {
                return Err(format!(
                    "{} packet {frame_index} spatial component sizes exceed packet body",
                    full_ivf.display()
                ));
            }
            components[layer].push(packet[offset..end].to_vec());
            offset = end;
        }
        if offset != body_len {
            return Err(format!(
                "{} packet {frame_index} spatial component sizes cover {offset} of {body_len} body bytes",
                full_ivf.display()
            ));
        }
    }

    let mut operating_points = Vec::with_capacity(layer_info.len());
    for (operating_point, layer) in layer_info.iter().enumerate() {
        let mut packets = Vec::with_capacity(expected_frames);
        let mut payload_bytes = 0u64;
        for (frame_index, _) in components[0].iter().enumerate() {
            let mut builder = SuperframeBuilder::new();
            for component in components.iter().take(operating_point + 1) {
                builder
                    .push_frame(component[frame_index].clone())
                    .map_err(|error| {
                        format!(
                            "building {} SL{operating_point} packet {frame_index}: {error}",
                            full_ivf.display()
                        )
                    })?;
            }
            let packet = builder.finish();
            payload_bytes = payload_bytes
                .checked_add(u64::try_from(packet.len()).map_err(|_| {
                    format!(
                        "{} SL{operating_point} payload byte count overflow",
                        full_ivf.display()
                    )
                })?)
                .ok_or_else(|| {
                    format!(
                        "{} SL{operating_point} payload byte count overflow",
                        full_ivf.display()
                    )
                })?;
            packets.push(packet);
        }
        let path = work_dir.join(format!("{prefix}_sl{operating_point}.ivf"));
        let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        let mut output = BufWriter::new(
            File::create(&path).map_err(|error| format!("creating {}: {error}", path.display()))?,
        );
        let header = IvfHeader::vp9(
            u16::try_from(layer.width).map_err(|_| "spatial layer width exceeds IVF".to_owned())?,
            u16::try_from(layer.height)
                .map_err(|_| "spatial layer height exceeds IVF".to_owned())?,
            info.fps_num,
            info.fps_den,
        );
        write_ivf(&mut output, &header, &refs)
            .map_err(|error| format!("writing {}: {error}", path.display()))?;
        output
            .flush()
            .map_err(|error| format!("flushing {}: {error}", path.display()))?;
        operating_points.push(OperatingPoint {
            path,
            payload_bytes,
            frame_count: expected_frames,
        });
    }
    Ok(operating_points)
}

fn write_reference_operating_points(
    reference_input: &Path,
    work_dir: &Path,
    info: VideoInfo,
    temporal_layers: u32,
    expected_frames: usize,
) -> Result<Vec<PathBuf>, String> {
    let reference_info = source_info(reference_input)?;
    if (reference_info.width, reference_info.height) != (info.width, info.height) {
        return Err("reference input dimensions differ from encoder input".to_owned());
    }
    let mut reader = Y4mReader::open(reference_input, info.width, info.height, expected_frames)
        .map_err(|error| format!("opening {}: {error}", reference_input.display()))?;
    let paths: Vec<PathBuf> = (0..temporal_layers)
        .map(|layer| work_dir.join(format!("reference_tl{layer}.y4m")))
        .collect();
    let mut outputs: Vec<BufWriter<File>> = paths
        .iter()
        .map(|path| {
            File::create(path)
                .map(BufWriter::new)
                .map_err(|error| format!("creating {}: {error}", path.display()))
        })
        .collect::<Result<_, _>>()?;
    for output in &mut outputs {
        write_header(
            output,
            info.width,
            info.height,
            info.fps_num,
            info.fps_den,
            ChromaFormat::Yuv420,
        )
        .map_err(|error| format!("writing reference header: {error}"))?;
    }
    for frame_index in 0..expected_frames {
        let (y, u, v) = reader
            .next_frame()
            .map_err(|error| format!("reading {}: {error}", reference_input.display()))?
            .ok_or_else(|| {
                format!(
                    "{} ended before frame {frame_index}",
                    reference_input.display()
                )
            })?;
        let id = temporal_id(frame_index, temporal_layers);
        for (operating_point, output) in outputs.iter_mut().enumerate() {
            if id <= operating_point as u32 {
                write_frame(output, &y, &u, &v)
                    .map_err(|error| format!("writing sampled reference: {error}"))?;
            }
        }
    }
    for output in &mut outputs {
        output
            .flush()
            .map_err(|error| format!("flushing sampled reference: {error}"))?;
    }
    Ok(paths)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SpatialLayerInfo {
    width: u32,
    height: u32,
    source_filter: Option<InterpFilter>,
}

fn spatial_layer_info(
    info: VideoInfo,
    spatial_layers: u32,
) -> Result<Vec<SpatialLayerInfo>, String> {
    let scales: &[(u32, u32)] = match spatial_layers {
        2 => &[(1, 2), (1, 1)],
        3 => &[(1, 4), (1, 2), (1, 1)],
        other => {
            return Err(format!("--spatial-layers must be 2 or 3, got {other}"));
        }
    };
    scales
        .iter()
        .map(|&(num, den)| {
            let (width, height) = rvp9_encoder::svc::layer_resolution(
                info.width,
                info.height,
                num,
                den,
            )
            .ok_or_else(|| {
                format!(
                    "{}x{} cannot form the requested {spatial_layers}-layer spatial pyramid",
                    info.width, info.height
                )
            })?;
            let source_filter = if num == den {
                None
            } else if u64::from(width) * u64::from(height) <= 320 * 240 {
                Some(InterpFilter::EightTapSmooth)
            } else {
                Some(InterpFilter::Bilinear)
            };
            Ok(SpatialLayerInfo {
                width,
                height,
                source_filter,
            })
        })
        .collect()
}

fn write_spatial_references(
    reference_input: &Path,
    work_dir: &Path,
    info: VideoInfo,
    spatial_layers: u32,
    expected_frames: usize,
) -> Result<Vec<PathBuf>, String> {
    let reference_info = source_info(reference_input)?;
    if (reference_info.width, reference_info.height) != (info.width, info.height) {
        return Err("reference input dimensions differ from encoder input".to_owned());
    }
    let layer_info = spatial_layer_info(info, spatial_layers)?;
    let paths: Vec<PathBuf> = (0..spatial_layers as usize)
        .map(|layer| work_dir.join(format!("reference_sl{layer}.y4m")))
        .collect();
    let mut outputs: Vec<BufWriter<File>> = paths
        .iter()
        .map(|path| {
            File::create(path)
                .map(BufWriter::new)
                .map_err(|error| format!("creating {}: {error}", path.display()))
        })
        .collect::<Result<_, _>>()?;
    for (output, layer) in outputs.iter_mut().zip(&layer_info) {
        write_header(
            output,
            layer.width,
            layer.height,
            info.fps_num,
            info.fps_den,
            ChromaFormat::Yuv420,
        )
        .map_err(|error| format!("writing spatial reference header: {error}"))?;
    }

    let mut reader = Y4mReader::open(reference_input, info.width, info.height, expected_frames)
        .map_err(|error| format!("opening {}: {error}", reference_input.display()))?;
    for frame_index in 0..expected_frames {
        let (y, u, v) = reader
            .next_frame()
            .map_err(|error| format!("reading {}: {error}", reference_input.display()))?
            .ok_or_else(|| {
                format!(
                    "{} ended before frame {frame_index}",
                    reference_input.display()
                )
            })?;
        let mut scaled_layers = vec![None; layer_info.len()];
        let top = layer_info.len() - 1;
        scaled_layers[top] = Some((y, u, v));
        for layer in (0..top).rev() {
            let source = scaled_layers[layer + 1]
                .as_ref()
                .expect("higher spatial layer was built first");
            let filter = layer_info[layer]
                .source_filter
                .expect("non-top spatial layer has a source filter");
            let scaled = scale_i420_2x(
                &source.0,
                &source.1,
                &source.2,
                layer_info[layer + 1].width as usize,
                layer_info[layer + 1].height as usize,
                filter,
            )
            .map_err(|error| format!("scaling reference SL{layer}: {error}"))?;
            if (scaled.width as u32, scaled.height as u32)
                != (layer_info[layer].width, layer_info[layer].height)
            {
                return Err(format!(
                    "spatial reference SL{layer} scaled to {}x{}, expected {}x{}",
                    scaled.width, scaled.height, layer_info[layer].width, layer_info[layer].height
                ));
            }
            scaled_layers[layer] = Some((scaled.y, scaled.u, scaled.v));
        }
        for (output, frame) in outputs.iter_mut().zip(scaled_layers) {
            let (y, u, v) = frame.expect("every spatial reference layer was built");
            write_frame(output, &y, &u, &v)
                .map_err(|error| format!("writing spatial reference frame: {error}"))?;
        }
    }
    for output in &mut outputs {
        output
            .flush()
            .map_err(|error| format!("flushing spatial reference: {error}"))?;
    }
    Ok(paths)
}

fn decode_and_psnr(
    decoder: &Path,
    input: &Path,
    reference: &Path,
    output: &Path,
) -> Result<f64, String> {
    let status = Command::new(decoder)
        .arg("-o")
        .arg(output)
        .arg(input)
        .status()
        .map_err(|error| format!("launching {}: {error}", decoder.display()))?;
    if !status.success() {
        return Err(format!(
            "decoder failed for {} with {status}",
            input.display()
        ));
    }
    rvp9_bench::mean_y_psnr(reference, output).ok_or_else(|| {
        format!(
            "PSNR comparison failed for {} against {}",
            output.display(),
            reference.display()
        )
    })
}

fn measured_bitrate_kbps(payload_bytes: u64, full_frames: usize, info: VideoInfo) -> f64 {
    let duration_seconds = full_frames as f64 * f64::from(info.fps_den) / f64::from(info.fps_num);
    payload_bytes as f64 * 8.0 / duration_seconds / 1000.0
}

fn write_csv(
    path: &Path,
    targets: &[u32],
    anchor: &[Vec<RdPoint>],
    test: &[Vec<RdPoint>],
    anchor_wall: &[f64],
    test_wall: &[f64],
    bdrates: &[f64],
) -> Result<(), String> {
    let mut output = BufWriter::new(
        File::create(path).map_err(|error| format!("creating {}: {error}", path.display()))?,
    );
    writeln!(
        output,
        "role,layer,target_kbps,measured_kbps,psnr_db,encode_wall_ms_per_frame"
    )
    .map_err(|error| error.to_string())?;
    for (target_index, target) in targets.iter().enumerate() {
        for layer in 0..anchor.len() {
            let point = anchor[layer][target_index];
            writeln!(
                output,
                "anchor,{layer},{target},{:.3},{:.6},{:.6}",
                point.bitrate_kbps, point.psnr_db, anchor_wall[target_index]
            )
            .map_err(|error| error.to_string())?;
            let point = test[layer][target_index];
            writeln!(
                output,
                "test,{layer},{target},{:.3},{:.6},{:.6}",
                point.bitrate_kbps, point.psnr_db, test_wall[target_index]
            )
            .map_err(|error| error.to_string())?;
        }
    }
    for (layer, value) in bdrates.iter().enumerate() {
        writeln!(output, "bdrate_pct,{layer},{value:.6}").map_err(|error| error.to_string())?;
    }
    output.flush().map_err(|error| error.to_string())
}

fn run(args: &Args) -> Result<bool, String> {
    let info = source_info(&args.input)?;
    let expected_frames = count_source_frames(&args.input, info, args.limit)?;
    let work_dir = prepare_work_dir(args.work_dir.as_deref())?;
    let references = match args.svc_kind {
        SvcKind::Temporal => write_reference_operating_points(
            &args.reference_input,
            &work_dir,
            info,
            args.temporal_layers,
            expected_frames,
        )?,
        SvcKind::Spatial => write_spatial_references(
            &args.reference_input,
            &work_dir,
            info,
            args.spatial_layers,
            expected_frames,
        )?,
    };
    let layer_count = match args.svc_kind {
        SvcKind::Temporal => args.temporal_layers as usize,
        SvcKind::Spatial => args.spatial_layers as usize,
    };
    println!("work-dir: {}", work_dir.display());
    println!(
        "config: cpu={} kind={} layers={} mode={} aq={} threads={} tiles={} frames={}",
        args.cpu_used,
        match args.svc_kind {
            SvcKind::Temporal => "temporal",
            SvcKind::Spatial => "spatial",
        },
        layer_count,
        args.end_usage,
        args.aq_mode,
        args.threads,
        args.tile_columns,
        expected_frames
    );

    let mut anchor_rd = vec![Vec::new(); layer_count];
    let mut test_rd = vec![Vec::new(); layer_count];
    let mut anchor_wall = Vec::new();
    let mut test_wall = Vec::new();

    for (target_index, &target) in args.bitrates_kbps.iter().enumerate() {
        let anchor_path = work_dir.join(format!("anchor_{target}k_full.ivf"));
        let test_path = work_dir.join(format!("test_{target}k_full.ivf"));
        let (anchor, test) = if target_index.is_multiple_of(2) {
            let anchor = encode(
                EncoderRole::Anchor,
                &args.anchor_encoder,
                anchor_path,
                target,
                args,
                info,
                expected_frames,
            )?;
            let test = encode(
                EncoderRole::Test,
                &args.test_encoder,
                test_path,
                target,
                args,
                info,
                expected_frames,
            )?;
            (anchor, test)
        } else {
            let test = encode(
                EncoderRole::Test,
                &args.test_encoder,
                test_path,
                target,
                args,
                info,
                expected_frames,
            )?;
            let anchor = encode(
                EncoderRole::Anchor,
                &args.anchor_encoder,
                anchor_path,
                target,
                args,
                info,
                expected_frames,
            )?;
            (anchor, test)
        };
        anchor_wall.push(anchor.wall_ms_per_frame);
        test_wall.push(test.wall_ms_per_frame);
        let anchor_points = match args.svc_kind {
            SvcKind::Temporal => extract_operating_points(
                &anchor.path,
                &format!("anchor_{target}k"),
                &work_dir,
                info,
                args.temporal_layers,
                expected_frames,
            )?,
            SvcKind::Spatial => extract_spatial_operating_points(
                &anchor.path,
                &format!("anchor_{target}k"),
                &work_dir,
                info,
                args.spatial_layers,
                expected_frames,
            )?,
        };
        let test_points = match args.svc_kind {
            SvcKind::Temporal => extract_operating_points(
                &test.path,
                &format!("test_{target}k"),
                &work_dir,
                info,
                args.temporal_layers,
                expected_frames,
            )?,
            SvcKind::Spatial => extract_spatial_operating_points(
                &test.path,
                &format!("test_{target}k"),
                &work_dir,
                info,
                args.spatial_layers,
                expected_frames,
            )?,
        };
        println!(
            "target {target:>5}k | encode anchor {:.3} ms/f test {:.3} ms/f ratio {:.4}x",
            anchor.wall_ms_per_frame,
            test.wall_ms_per_frame,
            test.wall_ms_per_frame / anchor.wall_ms_per_frame
        );
        for layer in 0..layer_count {
            if anchor_points[layer].frame_count != test_points[layer].frame_count {
                return Err(format!("TL{layer} frame counts differ between encoders"));
            }
            let anchor_decoded = work_dir.join(format!("anchor_{target}k_tl{layer}.y4m"));
            let test_decoded = work_dir.join(format!("test_{target}k_tl{layer}.y4m"));
            let anchor_psnr = decode_and_psnr(
                &args.decoder,
                &anchor_points[layer].path,
                &references[layer],
                &anchor_decoded,
            )?;
            let test_psnr = decode_and_psnr(
                &args.decoder,
                &test_points[layer].path,
                &references[layer],
                &test_decoded,
            )?;
            let anchor_bitrate =
                measured_bitrate_kbps(anchor_points[layer].payload_bytes, expected_frames, info);
            let test_bitrate =
                measured_bitrate_kbps(test_points[layer].payload_bytes, expected_frames, info);
            anchor_rd[layer].push(RdPoint {
                bitrate_kbps: anchor_bitrate,
                psnr_db: anchor_psnr,
            });
            test_rd[layer].push(RdPoint {
                bitrate_kbps: test_bitrate,
                psnr_db: test_psnr,
            });
            println!(
                "  {}: anchor {anchor_bitrate:>7.1}k {anchor_psnr:>7.3} dB | test {test_bitrate:>7.1}k {test_psnr:>7.3} dB",
                args.svc_kind.layer_label(layer)
            );
        }
    }

    let mut gate_green = true;
    let mut bdrates = Vec::new();
    println!();
    for layer in 0..layer_count {
        let anchor_curve: Vec<(f64, f64)> = anchor_rd[layer]
            .iter()
            .map(|point| (point.bitrate_kbps, point.psnr_db))
            .collect();
        let test_curve: Vec<(f64, f64)> = test_rd[layer]
            .iter()
            .map(|point| (point.bitrate_kbps, point.psnr_db))
            .collect();
        let value = bdrate(&anchor_curve, &test_curve);
        bdrates.push(value);
        if value.is_finite() {
            let passes = value <= args.max_bdrate_pct;
            gate_green &= passes;
            println!(
                "{} BD-rate: {value:+.4}%  gate <= {:+.4}%: {}",
                args.svc_kind.layer_label(layer),
                args.max_bdrate_pct,
                if passes { "PASS" } else { "FAIL" }
            );
        } else {
            gate_green = false;
            println!(
                "{} BD-rate: undefined (non-overlapping or invalid curves): FAIL",
                args.svc_kind.layer_label(layer)
            );
        }
        if let Some(delta) = psnr_delta_summary(&anchor_curve, &test_curve) {
            println!(
                "  matched-bitrate anchor-test PSNR delta: mean {:+.4} dB min {:+.4} max {:+.4} ({} in, {} out)",
                delta.mean_delta_db,
                delta.min_delta_db,
                delta.max_delta_db,
                delta.count_in_range,
                delta.count_out_of_range
            );
        }
    }
    if let Some(csv) = &args.csv {
        write_csv(
            csv,
            &args.bitrates_kbps,
            &anchor_rd,
            &test_rd,
            &anchor_wall,
            &test_wall,
            &bdrates,
        )?;
        println!("csv: {}", csv.display());
    }
    Ok(gate_green)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("bdrate-svc: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn fixed_temporal_patterns_match_libvpx() {
        let two: Vec<u32> = (0..8).map(|index| temporal_id(index, 2)).collect();
        let three: Vec<u32> = (0..8).map(|index| temporal_id(index, 3)).collect();
        assert_eq!(two, [0, 1, 0, 1, 0, 1, 0, 1]);
        assert_eq!(three, [0, 2, 1, 2, 0, 2, 1, 2]);
    }

    #[test]
    fn parser_keeps_matched_and_role_specific_controls() {
        let parsed = parse_args_from(&argv(&[
            "bdrate-svc",
            "--anchor-encoder",
            "anchor",
            "--test-encoder=test",
            "--decoder",
            "decoder",
            "--input",
            "clip.y4m",
            "--temporal-layers=2",
            "--anchor-arg",
            "--reference-only=1",
            "--test-arg=--candidate-only=1",
        ]))
        .unwrap();
        assert_eq!(parsed.temporal_layers, 2);
        assert_eq!(parsed.anchor_args, ["--reference-only=1"]);
        assert_eq!(parsed.test_args, ["--candidate-only=1"]);
        assert_eq!(parsed.reference_input, PathBuf::from("clip.y4m"));
    }

    #[test]
    fn parser_validates_spatial_shape_and_exact_rate_allocation() {
        let parsed = parse_args_from(&argv(&[
            "bdrate-svc",
            "--anchor-encoder",
            "anchor",
            "--test-encoder",
            "test",
            "--decoder",
            "decoder",
            "--input",
            "clip.y4m",
            "--svc-kind=spatial",
            "--spatial-layers=3",
            "--inter-layer-pred=3",
            "--temporal-layers=1",
            "--bitrates=210,420,630,840",
        ]))
        .unwrap();
        assert_eq!(parsed.svc_kind, SvcKind::Spatial);
        assert_eq!(parsed.spatial_layers, 3);
        assert_eq!(parsed.inter_layer_pred, 3);
        assert_eq!(spatial_layer_bitrates(210, 3).unwrap(), "10,40,160");

        let error = parse_args_from(&argv(&[
            "bdrate-svc",
            "--anchor-encoder",
            "anchor",
            "--test-encoder",
            "test",
            "--decoder",
            "decoder",
            "--input",
            "clip.y4m",
            "--svc-kind=spatial",
            "--spatial-layers=3",
            "--temporal-layers=1",
            "--bitrates=211,420,630,840",
        ]))
        .unwrap_err();
        assert!(error.contains("divisible by 21"));
    }

    #[test]
    fn spatial_extractor_rebuilds_cumulative_packets_and_layer_headers() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let work_dir = std::env::temp_dir().join(format!(
            "rvp9-bdrate-svc-spatial-test-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir_all(&work_dir).unwrap();
        let full = work_dir.join("full.ivf");
        let mut packets = Vec::new();
        for frame in 0..2u8 {
            let mut builder = SuperframeBuilder::new();
            builder.push_frame(vec![frame; 3]).unwrap();
            builder.push_frame(vec![frame + 10; 4]).unwrap();
            packets.push(builder.finish());
        }
        let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        let mut output = File::create(&full).unwrap();
        write_ivf(&mut output, &IvfHeader::vp9(128, 64, 30, 1), &refs).unwrap();
        let points = extract_spatial_operating_points(
            &full,
            "synthetic",
            &work_dir,
            VideoInfo {
                width: 128,
                height: 64,
                fps_num: 30,
                fps_den: 1,
            },
            2,
            2,
        )
        .unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].frame_count, 2);
        assert_eq!(points[1].frame_count, 2);
        assert_eq!(points[0].payload_bytes, 6);
        assert_eq!(points[1].payload_bytes, 22);
        let sl0 = std::fs::read(&points[0].path).unwrap();
        let sl1 = std::fs::read(&points[1].path).unwrap();
        assert_eq!(&sl0[12..16], &[64, 0, 32, 0]);
        assert_eq!(&sl1[12..16], &[128, 0, 64, 0]);
        assert_eq!(read_ivf_packets(&sl0).unwrap().len(), 2);
        assert_eq!(read_ivf_packets(&sl1).unwrap().len(), 2);
        let sl0_packets = read_ivf_packets(&sl0).unwrap();
        let sl0_data = &sl0;
        for range in sl0_packets {
            assert_eq!(
                parse_superframe_index(range.slice(sl0_data)).unwrap(),
                Superframe::Single
            );
        }
        let sl1_packets = read_ivf_packets(&sl1).unwrap();
        let sl1_data = &sl1;
        for range in sl1_packets {
            let Superframe::Multi { count, .. } =
                parse_superframe_index(range.slice(sl1_data)).unwrap()
            else {
                panic!("cumulative spatial packet must remain a superframe");
            };
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn spatial_reference_filters_follow_target_layer_area() {
        let info = VideoInfo {
            width: 1280,
            height: 720,
            fps_num: 30,
            fps_den: 1,
        };
        assert_eq!(
            spatial_layer_info(info, 3).unwrap(),
            vec![
                SpatialLayerInfo {
                    width: 320,
                    height: 180,
                    source_filter: Some(InterpFilter::EightTapSmooth),
                },
                SpatialLayerInfo {
                    width: 640,
                    height: 360,
                    source_filter: Some(InterpFilter::Bilinear),
                },
                SpatialLayerInfo {
                    width: 1280,
                    height: 720,
                    source_filter: None,
                },
            ]
        );
    }

    #[test]
    fn command_builders_use_the_same_scoped_configuration() {
        let parsed = parse_args_from(&argv(&[
            "bdrate-svc",
            "--anchor-encoder",
            "anchor",
            "--test-encoder",
            "test",
            "--decoder",
            "decoder",
            "--input",
            "clip.y4m",
            "--cpu-used=7",
            "--end-usage=cbr",
            "--threads=4",
            "--tile-columns=2",
            "--temporal-layers=3",
            "--aq-mode=3",
            "--noise-sensitivity=1",
            "--limit=60",
        ]))
        .unwrap();
        let info = VideoInfo {
            width: 1280,
            height: 720,
            fps_num: 30,
            fps_den: 1,
        };
        let anchor = encoder_command(
            EncoderRole::Anchor,
            Path::new("anchor"),
            Path::new("clip.y4m"),
            Path::new("anchor.ivf"),
            1500,
            &parsed,
            info,
            60,
        );
        let test = encoder_command(
            EncoderRole::Test,
            Path::new("test"),
            Path::new("clip.y4m"),
            Path::new("test.ivf"),
            1500,
            &parsed,
            info,
            60,
        );
        let anchor_args: Vec<String> = anchor
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        let test_args: Vec<String> = test
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        for value in ["7", "cbr", "4", "2", "3", "60"] {
            assert!(anchor_args.contains(&value.to_owned()));
        }
        for value in [
            "--cpu-used=7",
            "--end-usage=cbr",
            "--threads=4",
            "--tile-columns=2",
            "--temporal-layers=3",
            "--limit=60",
        ] {
            assert!(test_args.contains(&value.to_owned()));
        }
    }

    #[test]
    fn spatial_command_builders_pin_libvpx_and_rvp9_controls() {
        let parsed = parse_args_from(&argv(&[
            "bdrate-svc",
            "--anchor-encoder",
            "anchor",
            "--test-encoder",
            "test",
            "--decoder",
            "decoder",
            "--input",
            "clip.y4m",
            "--svc-kind=spatial",
            "--spatial-layers=3",
            "--inter-layer-pred=3",
            "--end-usage=cbr",
            "--aq-mode=3",
            "--cpu-used=8",
            "--threads=4",
            "--keyframe-interval=100",
            "--bitrates=210,420,630,840",
        ]))
        .unwrap();
        let info = VideoInfo {
            width: 1280,
            height: 720,
            fps_num: 30,
            fps_den: 1,
        };
        let anchor = encoder_command(
            EncoderRole::Anchor,
            Path::new("anchor"),
            Path::new("clip.y4m"),
            Path::new("anchor.ivf"),
            210,
            &parsed,
            info,
            60,
        );
        let test = encoder_command(
            EncoderRole::Test,
            Path::new("test"),
            Path::new("clip.y4m"),
            Path::new("test.ivf"),
            210,
            &parsed,
            info,
            60,
        );
        let anchor_args: Vec<String> = anchor
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        let test_args: Vec<String> = test
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        for value in [
            "--frames=60",
            "--speed=8",
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
            "clip.y4m",
        ] {
            assert!(
                anchor_args.contains(&value.to_owned()),
                "missing anchor {value}"
            );
        }
        for value in [
            "--spatial-layers=3",
            "--inter-layer-pred=3",
            "--min-q=8",
            "--max-q=224",
            "--temporal-layers=1",
        ] {
            assert!(
                test_args.contains(&value.to_owned()),
                "missing test {value}"
            );
        }
    }

    #[test]
    fn extractor_keeps_cumulative_operating_points() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let work_dir = std::env::temp_dir().join(format!(
            "rvp9-bdrate-svc-test-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir_all(&work_dir).unwrap();
        let full = work_dir.join("full.ivf");
        let packets: Vec<Vec<u8>> = (1..=8).map(|size| vec![size as u8; size]).collect();
        let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        let mut output = File::create(&full).unwrap();
        write_ivf(&mut output, &IvfHeader::vp9(16, 16, 30, 1), &refs).unwrap();
        let points = extract_operating_points(
            &full,
            "synthetic",
            &work_dir,
            VideoInfo {
                width: 16,
                height: 16,
                fps_num: 30,
                fps_den: 1,
            },
            3,
            8,
        )
        .unwrap();
        assert_eq!(points.len(), 3);
        assert_eq!(points[0].frame_count, 2);
        assert_eq!(points[1].frame_count, 4);
        assert_eq!(points[2].frame_count, 8);
        assert_eq!(points[0].payload_bytes, 6);
        assert_eq!(points[1].payload_bytes, 16);
        assert_eq!(points[2].payload_bytes, 36);
        for point in points {
            let bytes = std::fs::read(point.path).unwrap();
            assert_eq!(read_ivf_packets(&bytes).unwrap().len(), point.frame_count);
        }
    }

    #[test]
    fn operating_point_bitrate_uses_full_source_duration() {
        let info = VideoInfo {
            width: 1280,
            height: 720,
            fps_num: 30,
            fps_den: 1,
        };
        // 100,000 bytes over a two-second, 60-source-frame interval.
        assert_eq!(measured_bitrate_kbps(100_000, 60, info), 400.0);
    }
}
