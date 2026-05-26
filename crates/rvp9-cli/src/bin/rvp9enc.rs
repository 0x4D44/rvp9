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
use rvp9_encoder::{AqMode, Deadline, EncoderConfig, StreamEncoder, YuvFrameRef};
use rvp9_io::container::{IvfHeader, write_ivf};
use rvp9_io::y4m;
use std::path::Path;
use std::process::ExitCode;

/// vpxenc `--tune` distortion metric (libvpx `vpxenc.c` tune arg).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TuneMetric {
    Psnr,
    Ssim,
}

/// vpxenc `--tune-content` (libvpx `vpxenc.c:500-508`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TuneContent {
    Default,
    Screen,
    Film,
}

/// Parsed encoder options (the config builder the flag table fills in).
///
/// Fields below the `// ---- W0.1: stored, not yet plumbed` divider are
/// parsed and stored verbatim; later waves in the master inventory plumb
/// them through to [`EncoderConfig`] / `SpeedFeatures` / SVC. Storing
/// (instead of rejecting) lets us land the long tail of `vpxenc` flag
/// parity in one cheap step, with a single PROOF test guarding against
/// silent regression to the prior "recognised but not supported yet"
/// error path.
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

    // ---- W0.1: stored, not yet plumbed (master inventory A12 P0).
    /// `--profile` (libvpx `vpxenc.c:200`). VP9: 0..=3.
    profile: Option<u8>,
    /// `--best` / `--good` / `--rt` (libvpx `vpxenc.c:125-130`).
    deadline: Option<Deadline>,
    /// `--cpu-used` (libvpx `vpxenc.c:434-435`). VP9 range: -9..=9.
    cpu_used: Option<i32>,
    /// `--threads` / `-t` (libvpx `vpxenc.c:198-199`).
    threads: Option<u32>,
    /// `--tile-columns` (libvpx `vpxenc.c:439-440`). log2, 0..=6.
    tile_columns: Option<u8>,
    /// `--tile-rows` (libvpx `vpxenc.c:441-443`). log2, 0..=2.
    tile_rows: Option<u8>,
    /// `--kf-min-dist` (libvpx `vpxenc.c:358-359`).
    kf_min_dist: Option<u32>,
    /// `--disable-kf` (libvpx `vpxenc.c` kf_disabled near line 362).
    disable_kf: bool,
    /// `--min-q` (libvpx `vpxenc.c` min_quantizer).
    min_q: Option<u32>,
    /// `--max-q` (libvpx `vpxenc.c` max_quantizer).
    max_q: Option<u32>,
    /// `--cq-level` (libvpx `vpxenc.c:385`).
    cq_level: Option<u32>,
    /// `--buf-sz` (libvpx `vpxenc.c:275-276`, ms).
    buf_sz_ms: Option<u32>,
    /// `--buf-initial-sz` (libvpx `vpxenc.c:277-278`, ms).
    buf_initial_sz_ms: Option<u32>,
    /// `--buf-optimal-sz` (libvpx `vpxenc.c:279`, ms).
    buf_optimal_sz_ms: Option<u32>,
    /// `--undershoot-pct` (libvpx `vpxenc.c:271-272`).
    undershoot_pct: Option<u32>,
    /// `--overshoot-pct` (libvpx `vpxenc.c:273-274`).
    overshoot_pct: Option<u32>,
    /// `--lossless` (libvpx `vpxenc.c:448-449`). 0/1.
    lossless: Option<bool>,
    /// `--frame-parallel` (libvpx `vpxenc.c:450-451`). 0/1.
    frame_parallel: Option<bool>,
    /// `--error-resilient` (libvpx `vpxenc.c:218-219`).
    error_resilient: Option<u32>,
    /// `--tune` (libvpx `vpxenc.c` tune arg).
    tune: Option<TuneMetric>,
    /// `--tune-content` (libvpx `vpxenc.c:500-508`).
    tune_content: Option<TuneContent>,
    /// `--noise-sensitivity` (libvpx `vpxenc.c:367-368`).
    noise_sensitivity: Option<u32>,
    /// `--svc` (libvpx `vpxenc.c` SVC enable).
    svc: bool,
    /// `--bit-depth` (libvpx `vpxenc.c:492-495`). 8/10/12.
    bit_depth: Option<u8>,
    /// `--psnr` (libvpx `vpxenc.c:135-136`).
    psnr: bool,
    /// `--webm` — WebM container output toggle.
    webm: bool,
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

            // ---- W0.1: stored, not yet plumbed.
            profile: None,
            deadline: None,
            cpu_used: None,
            threads: None,
            tile_columns: None,
            tile_rows: None,
            kf_min_dist: None,
            disable_kf: false,
            min_q: None,
            max_q: None,
            cq_level: None,
            buf_sz_ms: None,
            buf_initial_sz_ms: None,
            buf_optimal_sz_ms: None,
            undershoot_pct: None,
            overshoot_pct: None,
            lossless: None,
            frame_parallel: None,
            error_resilient: None,
            tune: None,
            tune_content: None,
            noise_sensitivity: None,
            svc: false,
            bit_depth: None,
            psnr: false,
            webm: false,
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
        // W0.1: parsed and stored on EncOpts; later waves consume.
        Flag { long: "profile", short: None, arg: Arg::Required, help: "bitstream profile (0..=3)",
            apply: Some(|o, v| { o.profile = Some(parse_profile(v)?); Ok(()) }) },
        Flag { long: "rt", short: None, arg: Arg::None, help: "real-time deadline",
            apply: Some(|o, _| { o.deadline = Some(Deadline::Realtime); Ok(()) }) },
        Flag { long: "good", short: None, arg: Arg::None, help: "good-quality deadline",
            apply: Some(|o, _| { o.deadline = Some(Deadline::Good); Ok(()) }) },
        Flag { long: "best", short: None, arg: Arg::None, help: "best-quality deadline",
            apply: Some(|o, _| { o.deadline = Some(Deadline::Best); Ok(()) }) },
        Flag { long: "cpu-used", short: None, arg: Arg::Required, help: "speed/quality tradeoff (-9..=9)",
            apply: Some(|o, v| { o.cpu_used = Some(parse_cpu_used(v)?); Ok(()) }) },
        Flag { long: "threads", short: Some('t'), arg: Arg::Required, help: "encode threads",
            apply: Some(|o, v| { o.threads = Some(args::parse_value(v, "rvp9enc", "--threads")?); Ok(()) }) },
        Flag { long: "tile-columns", short: None, arg: Arg::Required, help: "log2 tile columns (0..=6)",
            apply: Some(|o, v| { o.tile_columns = Some(parse_log2(v, "--tile-columns", 6)?); Ok(()) }) },
        Flag { long: "tile-rows", short: None, arg: Arg::Required, help: "log2 tile rows (0..=2)",
            apply: Some(|o, v| { o.tile_rows = Some(parse_log2(v, "--tile-rows", 2)?); Ok(()) }) },
        Flag { long: "kf-min-dist", short: None, arg: Arg::Required, help: "min keyframe interval",
            apply: Some(|o, v| { o.kf_min_dist = Some(args::parse_value(v, "rvp9enc", "--kf-min-dist")?); Ok(()) }) },
        Flag { long: "disable-kf", short: None, arg: Arg::None, help: "disable keyframes",
            apply: Some(|o, _| { o.disable_kf = true; Ok(()) }) },
        Flag { long: "min-q", short: None, arg: Arg::Required, help: "minimum quantizer",
            apply: Some(|o, v| { o.min_q = Some(args::parse_value(v, "rvp9enc", "--min-q")?); Ok(()) }) },
        Flag { long: "max-q", short: None, arg: Arg::Required, help: "maximum quantizer",
            apply: Some(|o, v| { o.max_q = Some(args::parse_value(v, "rvp9enc", "--max-q")?); Ok(()) }) },
        Flag { long: "cq-level", short: None, arg: Arg::Required, help: "constant/constrained quality level",
            apply: Some(|o, v| { o.cq_level = Some(args::parse_value(v, "rvp9enc", "--cq-level")?); Ok(()) }) },
        Flag { long: "buf-sz", short: None, arg: Arg::Required, help: "client buffer size (ms)",
            apply: Some(|o, v| { o.buf_sz_ms = Some(args::parse_value(v, "rvp9enc", "--buf-sz")?); Ok(()) }) },
        Flag { long: "buf-initial-sz", short: None, arg: Arg::Required, help: "initial buffer size (ms)",
            apply: Some(|o, v| { o.buf_initial_sz_ms = Some(args::parse_value(v, "rvp9enc", "--buf-initial-sz")?); Ok(()) }) },
        Flag { long: "buf-optimal-sz", short: None, arg: Arg::Required, help: "optimal buffer size (ms)",
            apply: Some(|o, v| { o.buf_optimal_sz_ms = Some(args::parse_value(v, "rvp9enc", "--buf-optimal-sz")?); Ok(()) }) },
        Flag { long: "undershoot-pct", short: None, arg: Arg::Required, help: "datarate undershoot %",
            apply: Some(|o, v| { o.undershoot_pct = Some(args::parse_value(v, "rvp9enc", "--undershoot-pct")?); Ok(()) }) },
        Flag { long: "overshoot-pct", short: None, arg: Arg::Required, help: "datarate overshoot %",
            apply: Some(|o, v| { o.overshoot_pct = Some(args::parse_value(v, "rvp9enc", "--overshoot-pct")?); Ok(()) }) },
        Flag { long: "lossless", short: None, arg: Arg::Required, help: "lossless coding (0/1)",
            apply: Some(|o, v| { o.lossless = Some(parse_bool01(v, "--lossless")?); Ok(()) }) },
        Flag { long: "frame-parallel", short: None, arg: Arg::Required, help: "frame-parallel decoding (0/1)",
            apply: Some(|o, v| { o.frame_parallel = Some(parse_bool01(v, "--frame-parallel")?); Ok(()) }) },
        Flag { long: "error-resilient", short: None, arg: Arg::Required, help: "error resilient mode",
            apply: Some(|o, v| { o.error_resilient = Some(args::parse_value(v, "rvp9enc", "--error-resilient")?); Ok(()) }) },
        Flag { long: "tune", short: None, arg: Arg::Required, help: "distortion metric to tune (psnr/ssim)",
            apply: Some(|o, v| { o.tune = Some(parse_tune(v)?); Ok(()) }) },
        Flag { long: "tune-content", short: None, arg: Arg::Required, help: "content type to tune (default/screen/film)",
            apply: Some(|o, v| { o.tune_content = Some(parse_tune_content(v)?); Ok(()) }) },
        Flag { long: "noise-sensitivity", short: None, arg: Arg::Required, help: "denoiser strength",
            apply: Some(|o, v| { o.noise_sensitivity = Some(args::parse_value(v, "rvp9enc", "--noise-sensitivity")?); Ok(()) }) },
        Flag { long: "svc", short: None, arg: Arg::None, help: "scalable video coding",
            apply: Some(|o, _| { o.svc = true; Ok(()) }) },
        Flag { long: "bit-depth", short: None, arg: Arg::Required, help: "bit depth (8/10/12)",
            apply: Some(|o, v| { o.bit_depth = Some(parse_bit_depth(v)?); Ok(()) }) },
        Flag { long: "psnr", short: None, arg: Arg::None, help: "report PSNR",
            apply: Some(|o, _| { o.psnr = true; Ok(()) }) },
        Flag { long: "webm", short: None, arg: Arg::None, help: "WebM output",
            apply: Some(|o, _| { o.webm = true; Ok(()) }) },
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

// ---- W0.1 helpers: small parsers for the formerly-rejected flags.
// Each cites the libvpx range it enforces (see EncOpts field comments).

fn parse_profile(v: Option<String>) -> Result<u8, String> {
    let n: u8 = args::parse_value(v, "rvp9enc", "--profile")?;
    if n > 3 {
        return Err(format!("rvp9enc: --profile={n} out of range (0..=3)"));
    }
    Ok(n)
}

fn parse_cpu_used(v: Option<String>) -> Result<i32, String> {
    let n: i32 = args::parse_value(v, "rvp9enc", "--cpu-used")?;
    if !(-9..=9).contains(&n) {
        return Err(format!("rvp9enc: --cpu-used={n} out of range (-9..=9)"));
    }
    Ok(n)
}

fn parse_log2(v: Option<String>, flag: &str, max: u8) -> Result<u8, String> {
    let n: u8 = args::parse_value(v, "rvp9enc", flag)?;
    if n > max {
        return Err(format!("rvp9enc: {flag}={n} out of range (0..={max})"));
    }
    Ok(n)
}

fn parse_bool01(v: Option<String>, flag: &str) -> Result<bool, String> {
    match v.as_deref() {
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("rvp9enc: {flag}={other} not understood (use 0 or 1)")),
        None => Err(format!("rvp9enc: {flag} requires a value")),
    }
}

fn parse_tune(v: Option<String>) -> Result<TuneMetric, String> {
    match v.as_deref() {
        Some("psnr") => Ok(TuneMetric::Psnr),
        Some("ssim") => Ok(TuneMetric::Ssim),
        Some(other) => Err(format!("rvp9enc: --tune={other} not understood (use psnr or ssim)")),
        None => Err("rvp9enc: --tune requires a value".to_string()),
    }
}

fn parse_tune_content(v: Option<String>) -> Result<TuneContent, String> {
    match v.as_deref() {
        Some("default") => Ok(TuneContent::Default),
        Some("screen") => Ok(TuneContent::Screen),
        Some("film") => Ok(TuneContent::Film),
        Some(other) => Err(format!(
            "rvp9enc: --tune-content={other} not understood (use default/screen/film)"
        )),
        None => Err("rvp9enc: --tune-content requires a value".to_string()),
    }
}

fn parse_bit_depth(v: Option<String>) -> Result<u8, String> {
    match v.as_deref() {
        Some("8") => Ok(8),
        Some("10") => Ok(10),
        Some("12") => Ok(12),
        Some(other) => Err(format!("rvp9enc: --bit-depth={other} not understood (use 8/10/12)")),
        None => Err("rvp9enc: --bit-depth requires a value".to_string()),
    }
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
    //
    // W0.2(vpxenc parity, A12 P1) — `--cpu-used` and the
    // `--rt`/`--good`/`--best` deadline tri-state plumb through here.
    //
    // W0.4(vpxenc parity, A7 P1+P2) — defaults pin to rvp9's actual
    // op-point: `cpu-used=7 --rt`. vpxenc's library defaults are
    // (`cpu-used=0 --good`, `vpxenc.c:434-435, 127-128`), but rvp9
    // doesn't yet have a functional good-quality / best-quality path
    // (those land with Wave 4's RDO driver); until then the CLI's
    // implicit defaults must mirror what rvp9 can actually encode
    // today. Flip back to (0, Good) when W4 lands.
    let config = EncoderConfig {
        width: coded_w,
        height: coded_h,
        framerate: f64::from(fps_num) / f64::from(fps_den),
        target_bitrate_bps: opts.target_bitrate_kbps.saturating_mul(1000),
        gop_length: opts.gop_length.unwrap_or(60),
        aq_mode: opts.aq_mode,
        use_prev_frame_mvs: false,
        speed: opts.cpu_used.unwrap_or(7),
        deadline: opts.deadline.unwrap_or(Deadline::Realtime),
        noise_sensitivity: 0,
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

#[cfg(test)]
mod tests {
    //! PROOF tests for Wave 0 of the vpxenc parity port.
    //!
    //! These tests fail loudly if a formerly-wired flag is reverted to
    //! `apply: None` (the prior "recognised but not supported yet"
    //! rejection path), which is the master inventory's W0.1 anti-G3
    //! guard. Each assertion below pins one flag → one EncOpts slot.

    use super::*;

    fn s(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_string()).collect()
    }

    /// W0.1 PROOF — every formerly-rejected vpxenc flag is now parsed
    /// and stored on `EncOpts`. If any `apply` slot regresses to `None`,
    /// `args::parse` rejects the flag with "not supported yet" and this
    /// test fails. Citations: libvpx `vpxenc.c:125-130, 198-200, 218,
    /// 271-279, 358-368, 385, 434-450, 492, 500-508`.
    #[test]
    fn w0_1_all_formerly_rejected_flags_are_now_stored() {
        let table = flag_table();
        let argv = s(&[
            "--profile=2",
            "--rt",
            "--cpu-used=5",
            "--threads=4",
            "--tile-columns=2",
            "--tile-rows=1",
            "--kf-min-dist=8",
            "--disable-kf",
            "--min-q=2",
            "--max-q=63",
            "--cq-level=32",
            "--buf-sz=6000",
            "--buf-initial-sz=4000",
            "--buf-optimal-sz=5000",
            "--undershoot-pct=25",
            "--overshoot-pct=50",
            "--lossless=0",
            "--frame-parallel=1",
            "--error-resilient=1",
            "--tune=psnr",
            "--tune-content=screen",
            "--noise-sensitivity=2",
            "--svc",
            "--bit-depth=10",
            "--psnr",
            "--webm",
        ]);
        let mut opts = EncOpts::default();
        let pos = args::parse(&table, &mut opts, &argv, "rvp9enc").unwrap_or_else(|e| {
            panic!(
                "W0.1 PROOF: parse() rejected a formerly-wired flag — \
                 has any `apply: Some(_)` regressed to `None`? err={e}"
            )
        });
        assert!(pos.is_empty(), "no positionals expected, got {pos:?}");
        // Per-flag pinning — fails if the apply callback no longer
        // writes to the corresponding EncOpts slot.
        assert_eq!(opts.profile, Some(2));
        assert_eq!(opts.deadline, Some(Deadline::Realtime));
        assert_eq!(opts.cpu_used, Some(5));
        assert_eq!(opts.threads, Some(4));
        assert_eq!(opts.tile_columns, Some(2));
        assert_eq!(opts.tile_rows, Some(1));
        assert_eq!(opts.kf_min_dist, Some(8));
        assert!(opts.disable_kf);
        assert_eq!(opts.min_q, Some(2));
        assert_eq!(opts.max_q, Some(63));
        assert_eq!(opts.cq_level, Some(32));
        assert_eq!(opts.buf_sz_ms, Some(6000));
        assert_eq!(opts.buf_initial_sz_ms, Some(4000));
        assert_eq!(opts.buf_optimal_sz_ms, Some(5000));
        assert_eq!(opts.undershoot_pct, Some(25));
        assert_eq!(opts.overshoot_pct, Some(50));
        assert_eq!(opts.lossless, Some(false));
        assert_eq!(opts.frame_parallel, Some(true));
        assert_eq!(opts.error_resilient, Some(1));
        assert_eq!(opts.tune, Some(TuneMetric::Psnr));
        assert_eq!(opts.tune_content, Some(TuneContent::Screen));
        assert_eq!(opts.noise_sensitivity, Some(2));
        assert!(opts.svc);
        assert_eq!(opts.bit_depth, Some(10));
        assert!(opts.psnr);
        assert!(opts.webm);
    }

    /// `--good` and `--best` are mutually exclusive with `--rt` (the
    /// last one wins, mirroring vpxenc). Pin both directions so a
    /// regression that drops `Deadline::Good`/`Best` is caught.
    #[test]
    fn w0_1_deadline_good_and_best_distinct_from_rt() {
        let table = flag_table();
        let mut a = EncOpts::default();
        args::parse(&table, &mut a, &s(&["--good"]), "rvp9enc").unwrap();
        assert_eq!(a.deadline, Some(Deadline::Good));
        let mut b = EncOpts::default();
        args::parse(&table, &mut b, &s(&["--best"]), "rvp9enc").unwrap();
        assert_eq!(b.deadline, Some(Deadline::Best));
    }

    /// Range checks survive: libvpx VP9 enforces `--cpu-used` in
    /// -9..=9 (`vpxenc.c:434-435`), so out-of-range must still reject.
    #[test]
    fn w0_1_cpu_used_range_enforced() {
        let table = flag_table();
        let mut o = EncOpts::default();
        let err = args::parse(&table, &mut o, &s(&["--cpu-used=10"]), "rvp9enc").unwrap_err();
        assert!(err.contains("out of range"), "{err}");
    }

    /// W0.2 PROOF — `--cpu-used` and the deadline tri-state are
    /// plumbed onto [`EncoderConfig`] (libvpx `vpxenc.c:434-435`,
    /// `123-130`). Constructs an EncoderConfig via the same helper
    /// `run()` uses internally; if the wiring is stripped, the assert
    /// on `encoder.config().speed` / `.deadline` fails.
    #[test]
    fn w0_2_cpu_used_and_deadline_plumb_into_encoder_config() {
        let table = flag_table();

        // Real-time + cpu-used=7 — typical RT preset.
        let mut o = EncOpts::default();
        args::parse(&table, &mut o, &s(&["--cpu-used=7", "--rt"]), "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.speed, 7);
        assert_eq!(cfg.deadline, Deadline::Realtime);

        // Good + cpu-used=2 — middle of the good-quality band.
        let mut o = EncOpts::default();
        args::parse(&table, &mut o, &s(&["--cpu-used=2", "--good"]), "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.speed, 2);
        assert_eq!(cfg.deadline, Deadline::Good);

        // Best + cpu-used=-1 — slow path, libvpx allows negative values.
        let mut o = EncOpts::default();
        args::parse(&table, &mut o, &s(&["--cpu-used=-1", "--best"]), "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.speed, -1);
        assert_eq!(cfg.deadline, Deadline::Best);

        // Defaults — neither flag given. W0.4 pins rvp9's CLI default
        // to (speed=7, Realtime), the actual op point rvp9 can encode
        // today. (The vpxenc-canonical default is (0, Good); W4 brings
        // the RDO path online and flips this back.)
        let o = EncOpts::default();
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.speed, 7);
        assert_eq!(cfg.deadline, Deadline::Realtime);

        // The encoder accepts the config end-to-end (sanity check —
        // we wired a real public field, not a phantom one).
        let mut o = EncOpts::default();
        args::parse(&table, &mut o, &s(&["--cpu-used=4", "--rt"]), "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        let enc = StreamEncoder::new(cfg).expect("StreamEncoder must accept the W0.2 config");
        assert_eq!(enc.config().speed, 4);
        assert_eq!(enc.config().deadline, Deadline::Realtime);
    }
}

/// Test-only — builds a minimal valid `EncoderConfig` from `EncOpts`
/// for the W0.2 PROOF, mirroring the production builder in `run()`.
/// Kept out of `#[cfg(test)]` so dead-code lints stay quiet; the
/// function has no callers outside the test module.
#[cfg(test)]
fn build_encoder_config_for_test(opts: &EncOpts) -> EncoderConfig {
    EncoderConfig {
        width: 64,
        height: 64,
        framerate: 30.0,
        target_bitrate_bps: opts.target_bitrate_kbps.saturating_mul(1000),
        gop_length: opts.gop_length.unwrap_or(60),
        aq_mode: opts.aq_mode,
        use_prev_frame_mvs: false,
        speed: opts.cpu_used.unwrap_or(7),
        deadline: opts.deadline.unwrap_or(Deadline::Realtime),
        noise_sensitivity: 0,
    }
}
