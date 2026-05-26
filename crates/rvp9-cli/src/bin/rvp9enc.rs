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
use rvp9_encoder::lookahead::MAX_LAG_BUFFERS;
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

/// vpxenc `--test-decode` (libvpx `vpxenc.c:138-145`). Tri-state matching
/// libvpx's `test_decode_enum`: off (silent), warn (log mismatch), fatal
/// (abort on mismatch).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TestDecode {
    Off,
    Warn,
    Fatal,
}

/// vpxenc `--color-space` (libvpx `vpxenc.c:471-485`). Mirrors the
/// `color_space_enum` set; numeric values track `VPX_CS_*` for later
/// wiring. Kept here as an enum so unknown values reject at the CLI
/// layer rather than reaching libvpx.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ColorSpace {
    Unknown,
    Bt601,
    Bt709,
    Smpte170,
    Smpte240,
    Bt2020,
    Reserved,
    SRgb,
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

    // ---- W7.8: long-tail vpxenc flags (master inventory A12 tail).
    //
    // Each field below is stored verbatim by the flag table; plumbing
    // into [`EncoderConfig`] / two-pass / SVC / loop-filter paths lands
    // in later waves. Unknown values reject at the CLI layer so we
    // never silently misconfigure libvpx-style downstreams. Skipped
    // here per Arthur's audit: `--arnr-type` (vpxenc.c:378-379),
    // `--output-partitions` / `-P` (vpxenc.c:151-153) — both
    // OUT-OF-SCOPE for the vpxenc parity port.
    //
    /// `--auto-alt-ref` (libvpx `vpxenc.c:436-438`). VP9 range 0..=6:
    /// 0 disables, 1 single ARF, 2+ enables multi-layer ARF.
    auto_alt_ref: Option<u32>,
    /// `--arnr-maxframes` (libvpx `vpxenc.c:374-375`). 0..=15.
    arnr_maxframes: Option<u32>,
    /// `--arnr-strength` (libvpx `vpxenc.c:376-377`). 0..=6.
    arnr_strength: Option<u32>,
    /// `--pass` (libvpx `vpxenc.c:115-116`). 1 or 2.
    pass: Option<u32>,
    /// `--fpf` (libvpx `vpxenc.c:117-118`). First-pass stats file name.
    fpf: Option<String>,
    /// `--rate-hist` (libvpx `vpxenc.c:156-157`). Bucket count for the
    /// per-frame rate histogram.
    rate_hist: Option<u32>,
    /// `--q-hist` (libvpx `vpxenc.c:154-155`). Bucket count for the
    /// quantizer histogram.
    q_hist: Option<u32>,
    /// `--test-decode` (libvpx `vpxenc.c:138-145`). off / warn / fatal.
    test_decode: Option<TestDecode>,
    /// `--enable-tpl` (libvpx `vpxenc.c:445-446`). 0/1 — temporal
    /// dependency model.
    enable_tpl: Option<bool>,
    /// `--frame-boost` (libvpx `vpxenc.c:459-461`). 0/1 — periodic
    /// boost.
    frame_boost: Option<bool>,
    /// `--max-inter-rate` (libvpx `vpxenc.c:462-463`). Max P-frame
    /// bitrate (pct).
    max_inter_rate_pct: Option<u32>,
    /// `--min-gf-interval` (libvpx `vpxenc.c:464-466`).
    min_gf_interval: Option<u32>,
    /// `--max-gf-interval` (libvpx `vpxenc.c:467-469`).
    max_gf_interval: Option<u32>,
    /// `--color-space` (libvpx `vpxenc.c:483-485`, enum at 471-481).
    color_space: Option<ColorSpace>,
    /// `--alt-ref-aq` (libvpx `vpxenc.c:456-458`). 0/1.
    alt_ref_aq: Option<bool>,
    /// `--input-bit-depth` (libvpx `vpxenc.c:496-497`).
    input_bit_depth: Option<u8>,
    /// `--lag-in-frames` (libvpx `vpxenc.c:220-221`).
    lag_in_frames: Option<u32>,
    /// `--target-level` (libvpx `vpxenc.c:510-519`). 0/255 + level
    /// values (10..=62); the libvpx help text catalogues the semantics.
    target_level: Option<u32>,
    /// `--row-mt` (libvpx `vpxenc.c:521-523`). 0/1.
    row_mt: Option<bool>,
    /// `--resize-allowed` (libvpx `vpxenc.c:248-249`). 0/1.
    resize_allowed: Option<bool>,
    /// `--resize-width` (libvpx `vpxenc.c:250-251`).
    resize_width: Option<u32>,
    /// `--resize-height` (libvpx `vpxenc.c:252-253`).
    resize_height: Option<u32>,
    /// `--quiet` / `-q` (libvpx `vpxenc.c:131-132`). Suppress progress.
    quiet: bool,
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

            // ---- W7.8: long-tail vpxenc flags.
            auto_alt_ref: None,
            arnr_maxframes: None,
            arnr_strength: None,
            pass: None,
            fpf: None,
            rate_hist: None,
            q_hist: None,
            test_decode: None,
            enable_tpl: None,
            frame_boost: None,
            max_inter_rate_pct: None,
            min_gf_interval: None,
            max_gf_interval: None,
            color_space: None,
            alt_ref_aq: None,
            input_bit_depth: None,
            lag_in_frames: None,
            target_level: None,
            row_mt: None,
            resize_allowed: None,
            resize_width: None,
            resize_height: None,
            quiet: false,
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
        // ---- W7.8: long-tail vpxenc flags (parsed + stored; later
        // waves plumb into EncoderConfig / two-pass / SVC / LF paths).
        // SKIPPED per Arthur's audit: `--arnr-type` (vpxenc.c:378-379)
        // and `--output-partitions` / `-P` (vpxenc.c:151-153) are
        // OUT-OF-SCOPE — they survive in the "recognised-but-not-yet"
        // bucket via absence from the flag table.
        Flag { long: "auto-alt-ref", short: None, arg: Arg::Required, help: "auto alt-ref frames (0..=6)",
            apply: Some(|o, v| { o.auto_alt_ref = Some(parse_range(v, "--auto-alt-ref", 0, 6)?); Ok(()) }) },
        Flag { long: "arnr-maxframes", short: None, arg: Arg::Required, help: "AltRef max frames (0..=15)",
            apply: Some(|o, v| { o.arnr_maxframes = Some(parse_range(v, "--arnr-maxframes", 0, 15)?); Ok(()) }) },
        Flag { long: "arnr-strength", short: None, arg: Arg::Required, help: "AltRef filter strength (0..=6)",
            apply: Some(|o, v| { o.arnr_strength = Some(parse_range(v, "--arnr-strength", 0, 6)?); Ok(()) }) },
        Flag { long: "pass", short: None, arg: Arg::Required, help: "pass to execute (1 or 2)",
            apply: Some(|o, v| { o.pass = Some(parse_pass(v)?); Ok(()) }) },
        Flag { long: "fpf", short: None, arg: Arg::Required, help: "first-pass statistics file",
            apply: Some(|o, v| { o.fpf = Some(req(v, "--fpf")?); Ok(()) }) },
        Flag { long: "rate-hist", short: None, arg: Arg::Required, help: "rate histogram bucket count",
            apply: Some(|o, v| { o.rate_hist = Some(args::parse_value(v, "rvp9enc", "--rate-hist")?); Ok(()) }) },
        Flag { long: "q-hist", short: None, arg: Arg::Required, help: "quantizer histogram bucket count",
            apply: Some(|o, v| { o.q_hist = Some(args::parse_value(v, "rvp9enc", "--q-hist")?); Ok(()) }) },
        Flag { long: "test-decode", short: None, arg: Arg::Required, help: "encode/decode mismatch test (off/warn/fatal)",
            apply: Some(|o, v| { o.test_decode = Some(parse_test_decode(v)?); Ok(()) }) },
        Flag { long: "enable-tpl", short: None, arg: Arg::Required, help: "enable temporal dependency model (0/1)",
            apply: Some(|o, v| { o.enable_tpl = Some(parse_bool01(v, "--enable-tpl")?); Ok(()) }) },
        Flag { long: "frame-boost", short: None, arg: Arg::Required, help: "frame periodic boost (0/1)",
            apply: Some(|o, v| { o.frame_boost = Some(parse_bool01(v, "--frame-boost")?); Ok(()) }) },
        Flag { long: "max-inter-rate", short: None, arg: Arg::Required, help: "max P-frame bitrate (pct)",
            apply: Some(|o, v| { o.max_inter_rate_pct = Some(args::parse_value(v, "rvp9enc", "--max-inter-rate")?); Ok(()) }) },
        Flag { long: "min-gf-interval", short: None, arg: Arg::Required, help: "min gf/arf frame interval",
            apply: Some(|o, v| { o.min_gf_interval = Some(args::parse_value(v, "rvp9enc", "--min-gf-interval")?); Ok(()) }) },
        Flag { long: "max-gf-interval", short: None, arg: Arg::Required, help: "max gf/arf frame interval",
            apply: Some(|o, v| { o.max_gf_interval = Some(args::parse_value(v, "rvp9enc", "--max-gf-interval")?); Ok(()) }) },
        Flag { long: "color-space", short: None, arg: Arg::Required, help: "color space of input content",
            apply: Some(|o, v| { o.color_space = Some(parse_color_space(v)?); Ok(()) }) },
        Flag { long: "alt-ref-aq", short: None, arg: Arg::Required, help: "adaptive quant for alt-ref (0/1)",
            apply: Some(|o, v| { o.alt_ref_aq = Some(parse_bool01(v, "--alt-ref-aq")?); Ok(()) }) },
        Flag { long: "input-bit-depth", short: None, arg: Arg::Required, help: "input bit depth (8/10/12)",
            apply: Some(|o, v| { o.input_bit_depth = Some(parse_bit_depth(v)?); Ok(()) }) },
        Flag { long: "lag-in-frames", short: None, arg: Arg::Required, help: "max frames to lag",
            apply: Some(|o, v| { o.lag_in_frames = Some(args::parse_value(v, "rvp9enc", "--lag-in-frames")?); Ok(()) }) },
        Flag { long: "target-level", short: None, arg: Arg::Required, help: "target level (0/255 + 10..=62)",
            apply: Some(|o, v| { o.target_level = Some(args::parse_value(v, "rvp9enc", "--target-level")?); Ok(()) }) },
        Flag { long: "row-mt", short: None, arg: Arg::Required, help: "row-based MT in VP9 (0/1)",
            apply: Some(|o, v| { o.row_mt = Some(parse_bool01(v, "--row-mt")?); Ok(()) }) },
        Flag { long: "resize-allowed", short: None, arg: Arg::Required, help: "spatial resampling (0/1)",
            apply: Some(|o, v| { o.resize_allowed = Some(parse_bool01(v, "--resize-allowed")?); Ok(()) }) },
        Flag { long: "resize-width", short: None, arg: Arg::Required, help: "resized encoded width",
            apply: Some(|o, v| { o.resize_width = Some(args::parse_value(v, "rvp9enc", "--resize-width")?); Ok(()) }) },
        Flag { long: "resize-height", short: None, arg: Arg::Required, help: "resized encoded height",
            apply: Some(|o, v| { o.resize_height = Some(args::parse_value(v, "rvp9enc", "--resize-height")?); Ok(()) }) },
        Flag { long: "quiet", short: Some('q'), arg: Arg::None, help: "do not print encode progress",
            apply: Some(|o, _| { o.quiet = true; Ok(()) }) },
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

// ---- W7.8 helpers: parsers for the long-tail flags. Each enforces
// the same domain libvpx documents at the cited line.

fn parse_range(v: Option<String>, flag: &str, lo: u32, hi: u32) -> Result<u32, String> {
    let n: u32 = args::parse_value(v, "rvp9enc", flag)?;
    if n < lo || n > hi {
        return Err(format!("rvp9enc: {flag}={n} out of range ({lo}..={hi})"));
    }
    Ok(n)
}

/// libvpx `vpxenc.c:115-116` — `--pass` is 1 or 2.
fn parse_pass(v: Option<String>) -> Result<u32, String> {
    match v.as_deref() {
        Some("1") => Ok(1),
        Some("2") => Ok(2),
        Some(other) => Err(format!("rvp9enc: --pass={other} not understood (use 1 or 2)")),
        None => Err("rvp9enc: --pass requires a value".to_string()),
    }
}

/// libvpx `vpxenc.c:138-145` — `--test-decode` tri-state.
fn parse_test_decode(v: Option<String>) -> Result<TestDecode, String> {
    match v.as_deref() {
        Some("off") => Ok(TestDecode::Off),
        Some("warn") => Ok(TestDecode::Warn),
        Some("fatal") => Ok(TestDecode::Fatal),
        Some(other) => Err(format!(
            "rvp9enc: --test-decode={other} not understood (use off/warn/fatal)"
        )),
        None => Err("rvp9enc: --test-decode requires a value".to_string()),
    }
}

/// libvpx `vpxenc.c:471-485` — `--color-space` enum.
fn parse_color_space(v: Option<String>) -> Result<ColorSpace, String> {
    match v.as_deref() {
        Some("unknown") => Ok(ColorSpace::Unknown),
        Some("bt601") => Ok(ColorSpace::Bt601),
        Some("bt709") => Ok(ColorSpace::Bt709),
        Some("smpte170") => Ok(ColorSpace::Smpte170),
        Some("smpte240") => Ok(ColorSpace::Smpte240),
        Some("bt2020") => Ok(ColorSpace::Bt2020),
        Some("reserved") => Ok(ColorSpace::Reserved),
        Some("sRGB") => Ok(ColorSpace::SRgb),
        Some(other) => Err(format!(
            "rvp9enc: --color-space={other} not understood \
             (unknown/bt601/bt709/smpte170/smpte240/bt2020/reserved/sRGB)"
        )),
        None => Err("rvp9enc: --color-space requires a value".to_string()),
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
        // W5.1 plumb (re-audit pass 2 follow-up): CLI's
        // `--noise-sensitivity <0..=6>` now feeds the encoder's
        // `noise_sensitivity` gate so the W2.4 noise estimator + the
        // downstream W2.6 vbp_threshold_scale activate when the user
        // requests denoising. Default stays `0` (estimator off) when
        // the flag is omitted, matching vpxenc's CLI default. Clamp
        // to the 0..=6 range libvpx supports (range-checked at parse
        // time by `set_encoder_config`, but be defensive: cap to u8).
        noise_sensitivity: opts
            .noise_sensitivity
            .map(|v| v.min(6) as u8)
            .unwrap_or(0),
        // PHASE-3 integration: CLI `--min-q` / `--max-q` (W7.8) →
        // EncoderConfig. Defaults mirror RateControlConfig::default().
        min_qindex: opts.min_q.map(|v| v.min(255) as u8).unwrap_or(4),
        max_qindex: opts.max_q.map(|v| v.min(255) as u8).unwrap_or(255),
        // W6.2(vpxenc parity, HBD scaffold): pixel bit-depth. rvp9
        // only encodes 8-bit today; --bit-depth=10/12 is parsed
        // (W0.1) but the encoder refuses non-Bd8 until the HBD kernel
        // pairs land (W6.4 follow-up).
        bit_depth: rvp9_core::BitDepth::Bd8,
        // vpxenc `--lag-in-frames` (libvpx `vpxenc.c:220-221`,
        // capped at `MAX_LAG_BUFFERS = 25` per
        // `vp9_lookahead.h::MAX_LAG_BUFFERS`). The lookahead context
        // also clamps internally; the explicit cap here keeps the
        // user-visible domain matching libvpx.
        lag_in_frames: opts
            .lag_in_frames
            .map(|v| v.min(MAX_LAG_BUFFERS as u32))
            .unwrap_or(0),
    };
    let mut encoder = StreamEncoder::new(config).map_err(|e| {
        format!(
            "{e} (note: the Stage 4a encoder requires coded dimensions \
             that are multiples of 64 — use -w/-h to crop)"
        )
    })?;

    // Encode every frame; collect the per-frame VP9 packets.
    //
    // libvpx `vp9_encoder.c::vp9_get_compressed_data` splits the
    // pipeline so the encoder may return no packet on the first
    // `lag_in_frames` pushes (the lookahead ring fills before the
    // first pop fires). rvp9 mirrors that with
    // [`StreamEncoder::encode_frame_delayed`] (returns `None` until
    // steady-state) + [`StreamEncoder::flush`] (drains the tail at
    // end-of-stream — libvpx's `vpxenc.c::main` flush loop passing
    // a NULL frame).
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
        if let Some(enc) = encoder
            .encode_frame_delayed(&view)
            .map_err(|e| format!("encode_frame: {e}"))?
        {
            total_bits += u64::from(enc.actual_bits);
            packets.push(enc.data);
        }
    }
    // End-of-stream: drain any frames still buffered in the
    // lookahead. With `lag_in_frames == 0` this is a no-op (the
    // ring is empty after the synchronous push/pop in
    // `encode_frame_delayed`).
    for r in encoder.flush() {
        let enc = r.map_err(|e| format!("encode_frame (flush): {e}"))?;
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

    /// W7.8 PROOF — every long-tail vpxenc flag added by Wave 7.8 is
    /// parsed and stored on [`EncOpts`]. Like the W0.1 PROOF, this
    /// fails loudly if any `apply` slot regresses to `None` (which
    /// would route the flag through the "recognised but not supported
    /// yet" rejection path and break vpxenc CLI parity for benchmark
    /// harnesses).
    ///
    /// Citations: libvpx `vpxenc.c:115-118` (`--pass`/`--fpf`),
    /// `131-132` (`--quiet`), `138-145` (`--test-decode`),
    /// `154-157` (`--q-hist`/`--rate-hist`), `220-221` (`--lag-in-frames`),
    /// `248-253` (`--resize-*`), `374-377` (`--arnr-maxframes`/`-strength`),
    /// `436-438` (`--auto-alt-ref`), `445-446` (`--enable-tpl`),
    /// `456-461` (`--alt-ref-aq`/`--frame-boost`),
    /// `462-469` (`--max-inter-rate`/`--min/max-gf-interval`),
    /// `471-485` (`--color-space`), `496-497` (`--input-bit-depth`),
    /// `510-519` (`--target-level`), `521-523` (`--row-mt`).
    /// Skipped: `--arnr-type` (`vpxenc.c:378-379`) and
    /// `--output-partitions` / `-P` (`vpxenc.c:151-153`) — OUT-OF-SCOPE
    /// per Arthur's audit, intentionally absent from the flag table.
    #[test]
    fn w7_8_long_tail_flags_all_stored() {
        let table = flag_table();
        let argv = s(&[
            "--auto-alt-ref=2",
            "--arnr-maxframes=7",
            "--arnr-strength=5",
            "--pass=2",
            "--fpf=stats.fpf",
            "--rate-hist=64",
            "--q-hist=32",
            "--test-decode=warn",
            "--enable-tpl=1",
            "--frame-boost=0",
            "--max-inter-rate=120",
            "--min-gf-interval=4",
            "--max-gf-interval=16",
            "--color-space=bt709",
            "--alt-ref-aq=1",
            "--input-bit-depth=8",
            "--lag-in-frames=25",
            "--target-level=41",
            "--row-mt=1",
            "--resize-allowed=1",
            "--resize-width=640",
            "--resize-height=360",
            "--quiet",
        ]);
        let mut opts = EncOpts::default();
        let pos = args::parse(&table, &mut opts, &argv, "rvp9enc").unwrap_or_else(|e| {
            panic!(
                "W7.8 PROOF: parse() rejected a Wave-7.8 long-tail flag — \
                 has any `apply: Some(_)` regressed to `None`? err={e}"
            )
        });
        assert!(pos.is_empty(), "no positionals expected, got {pos:?}");
        assert_eq!(opts.auto_alt_ref, Some(2));
        assert_eq!(opts.arnr_maxframes, Some(7));
        assert_eq!(opts.arnr_strength, Some(5));
        assert_eq!(opts.pass, Some(2));
        assert_eq!(opts.fpf.as_deref(), Some("stats.fpf"));
        assert_eq!(opts.rate_hist, Some(64));
        assert_eq!(opts.q_hist, Some(32));
        assert_eq!(opts.test_decode, Some(TestDecode::Warn));
        assert_eq!(opts.enable_tpl, Some(true));
        assert_eq!(opts.frame_boost, Some(false));
        assert_eq!(opts.max_inter_rate_pct, Some(120));
        assert_eq!(opts.min_gf_interval, Some(4));
        assert_eq!(opts.max_gf_interval, Some(16));
        assert_eq!(opts.color_space, Some(ColorSpace::Bt709));
        assert_eq!(opts.alt_ref_aq, Some(true));
        assert_eq!(opts.input_bit_depth, Some(8));
        assert_eq!(opts.lag_in_frames, Some(25));
        assert_eq!(opts.target_level, Some(41));
        assert_eq!(opts.row_mt, Some(true));
        assert_eq!(opts.resize_allowed, Some(true));
        assert_eq!(opts.resize_width, Some(640));
        assert_eq!(opts.resize_height, Some(360));
        assert!(opts.quiet);
    }

    /// W7.8 — `--auto-alt-ref` range guard (libvpx `vpxenc.c:436-438`
    /// documents 0..=6). Out-of-range must still reject so we don't
    /// silently forward bogus values into a later config-build wave.
    #[test]
    fn w7_8_auto_alt_ref_range_enforced() {
        let table = flag_table();
        let mut o = EncOpts::default();
        let err = args::parse(&table, &mut o, &s(&["--auto-alt-ref=7"]), "rvp9enc").unwrap_err();
        assert!(err.contains("out of range"), "{err}");
    }

    /// W7.8 — `--arnr-type` is OUT-OF-SCOPE per Arthur's audit; the
    /// flag is intentionally absent from the table so libvpx-style use
    /// fails with `unknown option`, not silently accepted. (Mirrors the
    /// design principle that absence = "not recognised", presence with
    /// `apply: None` = "recognised but not yet". Arnr-type is the
    /// former; it should never become the latter.)
    #[test]
    fn w7_8_arnr_type_remains_out_of_scope() {
        let table = flag_table();
        let mut o = EncOpts::default();
        let err = args::parse(&table, &mut o, &s(&["--arnr-type=1"]), "rvp9enc").unwrap_err();
        assert!(err.contains("unknown option"), "{err}");
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

    /// PROOF (W5.1 CLI plumb) — re-audit pass 2 follow-up. The
    /// `--noise-sensitivity <0..=6>` flag is parsed but was not
    /// previously threaded into [`EncoderConfig::noise_sensitivity`]
    /// (literal `0` hardcoded in both the production `run()` and the
    /// test-only `build_encoder_config_for_test`). The
    /// post-89beb5e noise-estimator gate keys off this field, so the
    /// CLI value MUST land on the config — otherwise the user's
    /// `--noise-sensitivity > 0` request is silently ignored.
    ///
    /// Anti-G3: passing `--noise-sensitivity=3` must surface as
    /// `cfg.noise_sensitivity == 3`. A stripped wiring would leave
    /// it at the default `0`.
    #[test]
    fn w5_1_noise_sensitivity_plumbs_into_encoder_config() {
        let table = flag_table();

        // Default (flag omitted) → 0 (estimator off).
        let o = EncOpts::default();
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.noise_sensitivity, 0, "default must be 0");

        // Explicit non-zero value lands on the config field.
        let mut o = EncOpts::default();
        args::parse(&table, &mut o, &s(&["--noise-sensitivity=3"]), "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.noise_sensitivity, 3, "--noise-sensitivity=3 must plumb");

        // Out-of-range values clamp to 6 (libvpx caps the denoiser
        // at level 6; the parser's range-check should catch this too,
        // but the saturation guard is the defensive belt-and-braces).
        let mut o = EncOpts::default();
        o.noise_sensitivity = Some(99);
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.noise_sensitivity, 6, "99 must clamp to 6");

        // Verify the encoder accepts the wired config end-to-end.
        let mut o = EncOpts::default();
        args::parse(&table, &mut o, &s(&["--noise-sensitivity=1"]), "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        let enc =
            StreamEncoder::new(cfg).expect("StreamEncoder must accept --noise-sensitivity=1");
        assert_eq!(enc.config().noise_sensitivity, 1);
    }

    /// PROOF (W7.8 q-range CLI plumb) — `--min-q <0..=255>` and
    /// `--max-q <0..=255>` flow through to
    /// `EncoderConfig::{min,max}_qindex` (and from there to the
    /// rate-controller's CBR Q clamp). Re-audit pass 2 #7 flagged
    /// these as "parse-and-store without consumer". This PROOF pins
    /// the wiring; the consumer side is `RateControl::pick_q` which
    /// already honours `cfg.{min,max}_qindex`.
    ///
    /// Anti-G3: passing `--min-q=10 --max-q=200` must surface as
    /// `cfg.min_qindex == 10` / `cfg.max_qindex == 200`. A stripped
    /// wiring would pin both to defaults (4, 255).
    #[test]
    fn w7_8_q_range_plumbs_into_encoder_config() {
        let table = flag_table();

        // Default (flags omitted) → libvpx CBR preset (4, 255).
        let o = EncOpts::default();
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.min_qindex, 4, "default min_qindex must be 4");
        assert_eq!(cfg.max_qindex, 255, "default max_qindex must be 255");

        // Explicit values land on the config.
        let mut o = EncOpts::default();
        args::parse(&table, &mut o, &s(&["--min-q=10", "--max-q=200"]), "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.min_qindex, 10);
        assert_eq!(cfg.max_qindex, 200);

        // The encoder accepts the wired config (a sanity end-to-end —
        // `StreamEncoder::new` validates `min_qindex <= max_qindex`
        // via `RateControlConfig::validate`).
        let enc = StreamEncoder::new(cfg).expect("min<max must accept");
        assert_eq!(enc.config().min_qindex, 10);
        assert_eq!(enc.config().max_qindex, 200);

        // Inverted range must be rejected at construction time.
        let mut o = EncOpts::default();
        args::parse(&table, &mut o, &s(&["--min-q=200", "--max-q=10"]), "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        assert!(
            StreamEncoder::new(cfg).is_err(),
            "min_qindex > max_qindex must be rejected by validation",
        );
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
        noise_sensitivity: opts
            .noise_sensitivity
            .map(|v| v.min(6) as u8)
            .unwrap_or(0),
        // Re-audit pass 2 #7 follow-up: plumb the CLI's `--min-q` /
        // `--max-q` selection through to the rate controller. Default
        // (4, 255) mirrors EncoderConfig's default + the libvpx CBR
        // preset; unspecified callers keep byte-identical behaviour.
        min_qindex: opts.min_q.map(|v| v.min(255) as u8).unwrap_or(4),
        max_qindex: opts.max_q.map(|v| v.min(255) as u8).unwrap_or(255),
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
    }
}
