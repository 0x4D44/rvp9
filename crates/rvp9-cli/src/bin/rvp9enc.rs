//! rvp9enc: VP9 encoder CLI.
//!
//! Mirrors libvpx `vpxenc`'s command-line interface where practical, so
//! drop-in benchmarking against libvpx needs no harness changes.
//!
//! The encoder core lives in `rvp9-encoder` (Stage 4a — 1-pass CBR
//! real-time). This binary is the I/O shell: Y4M / raw YUV in →
//! `StreamEncoder` → IVF out.
//!
//! Only the flags that map to implemented behaviour are wired. The rest
//! of the `vpxenc` surface is listed by `--help` and errors with
//! "not supported yet" if used (see `rvp9_cli::args`) — so support grows
//! automatically as the encoder does, with no separate catalogue to
//! keep in sync.

use rvp9_cli::args::{self, Arg, Flag};
use rvp9_encoder::{AqMode, EncoderConfig, StreamEncoder, YuvFrameRef};
use rvp9_io::container::{IvfHeader, write_ivf};
use rvp9_io::y4m;
use std::path::Path;
use std::process::ExitCode;

/// Parsed encoder options (the config builder the flag table fills in).
struct EncOpts {
    width: Option<u32>,
    height: Option<u32>,
    fps_num: Option<u32>,
    fps_den: Option<u32>,
    target_bitrate_kbps: u32,
    gop_length: Option<u32>,
    aq_mode: AqMode,
    output: Option<String>,
    limit: Option<usize>,
    verbose: bool,
    help: bool,
    version: bool,
}

impl Default for EncOpts {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            fps_num: None,
            fps_den: None,
            // Mirror EncoderConfig's 1 Mbps default (1000 kbps).
            target_bitrate_kbps: 1000,
            gop_length: None,
            aq_mode: AqMode::Off,
            output: None,
            limit: None,
            verbose: false,
            help: false,
            version: false,
        }
    }
}

/// The flag table — single source of truth. `apply: Some` ⇒ wired;
/// `apply: None` ⇒ recognised-but-not-yet (errors if used).
fn flag_table() -> Vec<Flag<EncOpts>> {
    vec![
        Flag { long: "help", short: None, arg: Arg::None, help: "show this help and exit",
            apply: Some(|o, _| { o.help = true; Ok(()) }) },
        Flag { long: "version", short: None, arg: Arg::None, help: "print version and exit",
            apply: Some(|o, _| { o.version = true; Ok(()) }) },
        Flag { long: "codec", short: None, arg: Arg::Required, help: "codec to use (vp9)",
            apply: Some(|_, v| validate_eq(v, "vp9", "--codec")) },
        Flag { long: "output", short: Some('o'), arg: Arg::Required, help: "output IVF file",
            apply: Some(|o, v| { o.output = Some(req(v, "--output")?); Ok(()) }) },
        Flag { long: "width", short: Some('w'), arg: Arg::Required, help: "coded width (mult of 64)",
            apply: Some(|o, v| { o.width = Some(args::parse_value(v, "rvp9enc", "--width")?); Ok(()) }) },
        Flag { long: "height", short: Some('h'), arg: Arg::Required, help: "coded height (mult of 64)",
            apply: Some(|o, v| { o.height = Some(args::parse_value(v, "rvp9enc", "--height")?); Ok(()) }) },
        Flag { long: "fps", short: None, arg: Arg::Required, help: "frame rate num/den (e.g. 30/1)",
            apply: Some(|o, v| { let (n, d) = parse_fps(v)?; o.fps_num = Some(n); o.fps_den = Some(d); Ok(()) }) },
        Flag { long: "target-bitrate", short: None, arg: Arg::Required, help: "target bitrate in kbps",
            apply: Some(|o, v| { o.target_bitrate_kbps = args::parse_value(v, "rvp9enc", "--target-bitrate")?; Ok(()) }) },
        Flag { long: "end-usage", short: None, arg: Arg::Required, help: "rate control mode (cbr)",
            apply: Some(|_, v| validate_eq(v, "cbr", "--end-usage")) },
        Flag { long: "passes", short: Some('p'), arg: Arg::Required, help: "number of passes (1)",
            apply: Some(|_, v| validate_eq(v, "1", "--passes")) },
        Flag { long: "kf-max-dist", short: None, arg: Arg::Required, help: "max keyframe interval",
            apply: Some(|o, v| { o.gop_length = Some(args::parse_value(v, "rvp9enc", "--kf-max-dist")?); Ok(()) }) },
        Flag { long: "aq-mode", short: None, arg: Arg::Required, help: "adaptive quant (0=off, 3=cyclic)",
            apply: Some(|o, v| { o.aq_mode = parse_aq_mode(v)?; Ok(()) }) },
        Flag { long: "limit", short: None, arg: Arg::Required, help: "stop after N frames",
            apply: Some(|o, v| { o.limit = Some(args::parse_value(v, "rvp9enc", "--limit")?); Ok(()) }) },
        Flag { long: "ivf", short: None, arg: Arg::None, help: "output IVF (the only container)",
            apply: Some(|_, _| Ok(())) },
        Flag { long: "verbose", short: Some('v'), arg: Arg::None, help: "verbose progress",
            apply: Some(|o, _| { o.verbose = true; Ok(()) }) },
        // Recognised but not implemented yet — error if used.
        Flag { long: "profile", short: None, arg: Arg::Required, help: "bitstream profile", apply: None },
        Flag { long: "rt", short: None, arg: Arg::None, help: "real-time deadline", apply: None },
        Flag { long: "good", short: None, arg: Arg::None, help: "good-quality deadline", apply: None },
        Flag { long: "best", short: None, arg: Arg::None, help: "best-quality deadline", apply: None },
        Flag { long: "cpu-used", short: None, arg: Arg::Required, help: "speed/quality tradeoff", apply: None },
        Flag { long: "threads", short: Some('t'), arg: Arg::Required, help: "encode threads", apply: None },
        Flag { long: "tile-columns", short: None, arg: Arg::Required, help: "log2 tile columns", apply: None },
        Flag { long: "tile-rows", short: None, arg: Arg::Required, help: "log2 tile rows", apply: None },
        Flag { long: "kf-min-dist", short: None, arg: Arg::Required, help: "min keyframe interval", apply: None },
        Flag { long: "disable-kf", short: None, arg: Arg::None, help: "disable keyframes", apply: None },
        Flag { long: "min-q", short: None, arg: Arg::Required, help: "minimum quantizer", apply: None },
        Flag { long: "max-q", short: None, arg: Arg::Required, help: "maximum quantizer", apply: None },
        Flag { long: "cq-level", short: None, arg: Arg::Required, help: "constant/constrained quality level", apply: None },
        Flag { long: "buf-sz", short: None, arg: Arg::Required, help: "client buffer size (ms)", apply: None },
        Flag { long: "buf-initial-sz", short: None, arg: Arg::Required, help: "initial buffer size (ms)", apply: None },
        Flag { long: "buf-optimal-sz", short: None, arg: Arg::Required, help: "optimal buffer size (ms)", apply: None },
        Flag { long: "undershoot-pct", short: None, arg: Arg::Required, help: "datarate undershoot %", apply: None },
        Flag { long: "overshoot-pct", short: None, arg: Arg::Required, help: "datarate overshoot %", apply: None },
        Flag { long: "lossless", short: None, arg: Arg::Required, help: "lossless coding", apply: None },
        Flag { long: "frame-parallel", short: None, arg: Arg::Required, help: "frame-parallel decoding", apply: None },
        Flag { long: "error-resilient", short: None, arg: Arg::Required, help: "error resilient mode", apply: None },
        Flag { long: "tune", short: None, arg: Arg::Required, help: "distortion metric to tune", apply: None },
        Flag { long: "tune-content", short: None, arg: Arg::Required, help: "content type to tune", apply: None },
        Flag { long: "noise-sensitivity", short: None, arg: Arg::Required, help: "denoiser strength", apply: None },
        Flag { long: "svc", short: None, arg: Arg::None, help: "scalable video coding", apply: None },
        Flag { long: "bit-depth", short: None, arg: Arg::Required, help: "bit depth (8/10/12)", apply: None },
        Flag { long: "psnr", short: None, arg: Arg::None, help: "report PSNR", apply: None },
        Flag { long: "webm", short: None, arg: Arg::None, help: "WebM output", apply: None },
    ]
}

fn validate_eq(v: Option<String>, want: &str, flag: &str) -> Result<(), String> {
    match v.as_deref() {
        Some(s) if s == want => Ok(()),
        Some(other) => Err(format!("rvp9enc: {flag}={other} not supported yet (only {want})")),
        None => Err(format!("rvp9enc: {flag} requires a value")),
    }
}

fn parse_aq_mode(v: Option<String>) -> Result<AqMode, String> {
    match v.as_deref() {
        Some("0") => Ok(AqMode::Off),
        Some("3") => Ok(AqMode::CyclicRefresh),
        Some(other) => Err(format!(
            "rvp9enc: --aq-mode={other} not supported yet (only 0=off, 3=cyclic-refresh)"
        )),
        None => Err("rvp9enc: --aq-mode requires a value".to_string()),
    }
}

fn parse_fps(v: Option<String>) -> Result<(u32, u32), String> {
    let s = v.ok_or_else(|| "rvp9enc: --fps requires a value".to_string())?;
    match s.split_once('/') {
        Some((n, d)) => {
            let num = n.parse::<u32>().map_err(|_| format!("rvp9enc: --fps: bad numerator '{n}'"))?;
            let den = d.parse::<u32>().map_err(|_| format!("rvp9enc: --fps: bad denominator '{d}'"))?;
            if den == 0 {
                return Err("rvp9enc: --fps denominator must be non-zero".to_string());
            }
            Ok((num, den))
        }
        None => {
            let num = s.parse::<u32>().map_err(|_| format!("rvp9enc: --fps: bad value '{s}'"))?;
            Ok((num, 1))
        }
    }
}

fn req(v: Option<String>, flag: &str) -> Result<String, String> {
    v.ok_or_else(|| format!("rvp9enc: {flag} requires a value"))
}

const USAGE: &str = "rvp9enc [options] -o <out.ivf> <input.y4m|input.yuv>";

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let table = flag_table();

    if argv.iter().any(|a| a == "--help") {
        args::print_help(&table, "rvp9enc", USAGE);
        return ExitCode::SUCCESS;
    }
    if argv.iter().any(|a| a == "--version") {
        println!("rvp9enc {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    if argv.is_empty() {
        eprintln!("rvp9enc: no input. Try `rvp9enc --help`.\nUsage: {USAGE}");
        return ExitCode::FAILURE;
    }

    let mut opts = EncOpts::default();
    let positionals = match args::parse(&table, &mut opts, &argv, "rvp9enc") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    if opts.help {
        args::print_help(&table, "rvp9enc", USAGE);
        return ExitCode::SUCCESS;
    }

    match run(&opts, &positionals) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rvp9enc: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(opts: &EncOpts, positionals: &[String]) -> Result<(), String> {
    let input = match positionals {
        [one] => Path::new(one),
        [] => return Err("no input file given".to_string()),
        _ => return Err("expected exactly one input file".to_string()),
    };
    let output = opts
        .output
        .as_deref()
        .ok_or_else(|| "no output file (-o <out.ivf>)".to_string())?;

    // Read source frames + source framerate.
    let is_y4m = input
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("y4m"));

    let (frames, (fps_num, fps_den), coded_w, coded_h) = if is_y4m {
        read_y4m_input(input, opts)?
    } else {
        read_raw_input(input, opts)?
    };

    if frames.is_empty() {
        return Err("input contained no frames".to_string());
    }

    // Build and validate the encoder config (width/height must be a
    // multiple of 64 — a Stage 4a limitation, surfaced clearly).
    let config = EncoderConfig {
        width: coded_w,
        height: coded_h,
        framerate: f64::from(fps_num) / f64::from(fps_den),
        target_bitrate_bps: opts.target_bitrate_kbps.saturating_mul(1000),
        gop_length: opts.gop_length.unwrap_or(60),
        aq_mode: opts.aq_mode,
    };
    let mut encoder = StreamEncoder::new(config).map_err(|e| {
        format!(
            "{e} (note: the Stage 4a encoder requires coded dimensions \
             that are multiples of 64 — use -w/-h to crop)"
        )
    })?;

    // Encode every frame; collect the per-frame VP9 packets.
    let mut packets: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    let mut total_bits: u64 = 0;
    for (y, u, v) in &frames {
        let view = YuvFrameRef {
            y,
            u,
            v,
            y_stride: coded_w as usize,
            u_stride: (coded_w / 2) as usize,
            v_stride: (coded_w / 2) as usize,
        };
        let enc = encoder
            .encode_frame(&view)
            .map_err(|e| format!("encode_frame: {e}"))?;
        total_bits += u64::from(enc.actual_bits);
        packets.push(enc.data);
    }

    // Mux into IVF.
    let packet_refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
    let header = IvfHeader::vp9(coded_w as u16, coded_h as u16, fps_num, fps_den);
    let mut out = std::fs::File::create(output).map_err(|e| format!("creating {output}: {e}"))?;
    write_ivf(&mut out, &header, &packet_refs).map_err(|e| format!("writing IVF: {e}"))?;

    if opts.verbose {
        let secs = frames.len() as f64 * f64::from(fps_den) / f64::from(fps_num);
        let kbps = if secs > 0.0 {
            (total_bits as f64 / secs) / 1000.0
        } else {
            0.0
        };
        eprintln!(
            "rvp9enc: encoded {} frame(s) at {}x{}, ~{:.1} kbps -> {output}",
            frames.len(),
            coded_w,
            coded_h,
            kbps
        );
    }
    Ok(())
}

/// Read a Y4M input, centre-cropping to the coded dimensions
/// (`-w`/`-h` if given, else the source size). Returns
/// `(frames, (fps_num, fps_den), coded_w, coded_h)`.
type FramesAndDims = (Vec<y4m::YuvFrameBuf>, (u32, u32), u32, u32);

fn read_y4m_input(input: &Path, opts: &EncOpts) -> Result<FramesAndDims, String> {
    let mut f = std::fs::File::open(input).map_err(|e| format!("opening {}: {e}", input.display()))?;
    let hdr = y4m::parse_header(&mut f).map_err(|e| format!("parsing y4m header: {e}"))?;
    drop(f);

    let coded_w = opts.width.unwrap_or(hdr.w);
    let coded_h = opts.height.unwrap_or(hdr.h);
    if coded_w > hdr.w || coded_h > hdr.h {
        return Err(format!(
            "coded size {coded_w}x{coded_h} exceeds source {}x{}",
            hdr.w, hdr.h
        ));
    }

    // Frame count: limit if set, else estimate from the file size so
    // read_cropped reads to EOF without over-allocating wildly.
    let n = match opts.limit {
        Some(l) => l,
        None => {
            let fsize = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0) as usize;
            let body = (hdr.w as usize) * (hdr.h as usize) * 3 / 2;
            fsize / (body + 6) + 1
        }
    };

    let (frames, _fr) = y4m::read_cropped(input, coded_w, coded_h, n)
        .map_err(|e| format!("reading y4m frames: {e}"))?;

    let fps = match (opts.fps_num, opts.fps_den) {
        (Some(n), Some(d)) => (n, d),
        _ => (hdr.fps_num, hdr.fps_den),
    };
    Ok((frames, fps, coded_w, coded_h))
}

/// Read a raw I420 (planar 4:2:0) input. Requires `-w`/`-h`; framerate
/// defaults to 30/1 unless `--fps` is given.
fn read_raw_input(input: &Path, opts: &EncOpts) -> Result<FramesAndDims, String> {
    let coded_w = opts
        .width
        .ok_or_else(|| "raw input requires -w/--width".to_string())?;
    let coded_h = opts
        .height
        .ok_or_else(|| "raw input requires -h/--height".to_string())?;
    let fps = (opts.fps_num.unwrap_or(30), opts.fps_den.unwrap_or(1));

    let data = std::fs::read(input).map_err(|e| format!("reading {}: {e}", input.display()))?;
    let w = coded_w as usize;
    let h = coded_h as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    let frame_size = y_size + 2 * uv_size;
    if frame_size == 0 || data.len() % frame_size != 0 {
        return Err(format!(
            "raw input length {} is not a whole number of {coded_w}x{coded_h} I420 frames ({frame_size} bytes each)",
            data.len()
        ));
    }

    let mut frames = Vec::new();
    let max = opts.limit.unwrap_or(usize::MAX);
    for chunk in data.chunks_exact(frame_size).take(max) {
        let y = chunk[..y_size].to_vec();
        let u = chunk[y_size..y_size + uv_size].to_vec();
        let v = chunk[y_size + uv_size..].to_vec();
        frames.push((y, u, v));
    }
    Ok((frames, fps, coded_w, coded_h))
}
