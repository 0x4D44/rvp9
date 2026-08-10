//! Per-temporal-operating-point BD-rate measurement.
//!
//! The anchor is `tools/libvpx_temporal_svc_reference`, which exposes the
//! pinned libvpx encoder core's built-in 0101 / 0212 modes. Both full streams
//! are split by the fixed temporal pattern, decoded through the same decoder,
//! and compared with source frames sampled at the same operating point.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rvp9_core::bitstream::frame::ChromaFormat;
use rvp9_harness::bdrate::{bdrate, psnr_delta_summary};
use rvp9_harness::container::{IvfHeader, read_ivf_packets, write_ivf};
use rvp9_harness::y4m::{Y4mReader, parse_header, write_frame, write_header};

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
    threads: u32,
    tile_columns: u32,
    temporal_layers: u32,
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

fn parse_args_from(argv: &[String]) -> Result<Args, String> {
    let mut anchor_encoder = None;
    let mut test_encoder = None;
    let mut decoder = None;
    let mut input = None;
    let mut reference_input = None;
    let mut cpu_used = 7;
    let mut bitrates_kbps = vec![500, 750, 1000, 1500, 2500, 4000];
    let mut end_usage = "vbr".to_owned();
    let mut threads = 1;
    let mut tile_columns = 0;
    let mut temporal_layers = 3;
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
            "--threads" => threads = parse_u32(flag, &value(&mut index)?)?,
            "--tile-columns" => tile_columns = parse_u32(flag, &value(&mut index)?)?,
            "--temporal-layers" => {
                temporal_layers = parse_u32(flag, &value(&mut index)?)?;
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
    let args = Args {
        anchor_encoder: anchor_encoder.ok_or_else(|| "--anchor-encoder is required".to_owned())?,
        test_encoder: test_encoder.ok_or_else(|| "--test-encoder is required".to_owned())?,
        decoder: decoder.ok_or_else(|| "--decoder is required".to_owned())?,
        reference_input: reference_input.unwrap_or_else(|| input.clone()),
        input,
        cpu_used,
        bitrates_kbps,
        end_usage,
        threads,
        tile_columns,
        temporal_layers,
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
--threads N --tile-columns N\n  --temporal-layers {2,3}\n  --aq-mode {0,3}\n  \
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
    if !matches!(args.temporal_layers, 2 | 3) {
        return Err("--temporal-layers must be 2 or 3".to_owned());
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

fn encoder_command(
    role: EncoderRole,
    encoder: &Path,
    input: &Path,
    output: &Path,
    bitrate_kbps: u32,
    args: &Args,
    info: VideoInfo,
) -> Command {
    let mut command = Command::new(encoder);
    match role {
        EncoderRole::Anchor => {
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
            command.args(&args.anchor_args);
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
    let references = write_reference_operating_points(
        &args.reference_input,
        &work_dir,
        info,
        args.temporal_layers,
        expected_frames,
    )?;
    println!("work-dir: {}", work_dir.display());
    println!(
        "config: cpu={} layers={} mode={} aq={} threads={} tiles={} frames={}",
        args.cpu_used,
        args.temporal_layers,
        args.end_usage,
        args.aq_mode,
        args.threads,
        args.tile_columns,
        expected_frames
    );

    let mut anchor_rd = vec![Vec::new(); args.temporal_layers as usize];
    let mut test_rd = vec![Vec::new(); args.temporal_layers as usize];
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
        let anchor_points = extract_operating_points(
            &anchor.path,
            &format!("anchor_{target}k"),
            &work_dir,
            info,
            args.temporal_layers,
            expected_frames,
        )?;
        let test_points = extract_operating_points(
            &test.path,
            &format!("test_{target}k"),
            &work_dir,
            info,
            args.temporal_layers,
            expected_frames,
        )?;
        println!(
            "target {target:>5}k | encode anchor {:.3} ms/f test {:.3} ms/f ratio {:.4}x",
            anchor.wall_ms_per_frame,
            test.wall_ms_per_frame,
            test.wall_ms_per_frame / anchor.wall_ms_per_frame
        );
        for layer in 0..args.temporal_layers as usize {
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
                "  TL{layer}: anchor {anchor_bitrate:>7.1}k {anchor_psnr:>7.3} dB | test {test_bitrate:>7.1}k {test_psnr:>7.3} dB"
            );
        }
    }

    let mut gate_green = true;
    let mut bdrates = Vec::new();
    println!();
    for layer in 0..args.temporal_layers as usize {
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
                "TL{layer} BD-rate: {value:+.4}%  gate <= {:+.4}%: {}",
                args.max_bdrate_pct,
                if passes { "PASS" } else { "FAIL" }
            );
        } else {
            gate_green = false;
            println!("TL{layer} BD-rate: undefined (non-overlapping or invalid curves): FAIL");
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
        );
        let test = encoder_command(
            EncoderRole::Test,
            Path::new("test"),
            Path::new("clip.y4m"),
            Path::new("test.ivf"),
            1500,
            &parsed,
            info,
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
