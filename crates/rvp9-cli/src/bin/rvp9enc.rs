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

#[path = "../path_guard.rs"]
mod path_guard;

use rvp9_cli::args::{self, Arg, Flag};
use rvp9_encoder::lookahead::MAX_LAG_BUFFERS;
use rvp9_encoder::rate_control::config::{
    MAX_CODED_DIMENSION, MAX_CODED_LUMA_PIXELS, coded_dimensions_supported,
};
use rvp9_encoder::{
    AqMode, Deadline, EncoderConfig, EncoderParallelism, RcMode, SpatialEncoder, StreamEncoder,
    YuvFrameRef, encode_two_pass,
    svc::{InterLayerPred, SpatialSvcConfig},
};
use rvp9_io::container::{IvfHeader, IvfStreamWriter, write_ivf};
use rvp9_io::y4m;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

/// Parsed encoder options (the config builder the flag table fills in).
struct EncOpts {
    width: Option<u32>,
    height: Option<u32>,
    fps_num: Option<u32>,
    fps_den: Option<u32>,
    target_bitrate_kbps: u32,
    gop_length: Option<u32>,
    aq_mode: AqMode,
    /// `--end-usage` (libvpx `vpxenc.c` `--end-usage`,
    /// `vpx_encoder.h enum vpx_rc_mode`). Rate-control regulator:
    /// `cbr` (default) / `vbr` / `cq` / `q`. Plumbed onto
    /// [`EncoderConfig::rc_mode`] at construction.
    rc_mode: RcMode,
    output: Option<String>,
    limit: Option<usize>,
    verbose: bool,
    help: bool,
    version: bool,

    /// `--best` / `--good` / `--rt` (libvpx `vpxenc.c:125-130`).
    deadline: Option<Deadline>,
    /// `--cpu-used` (libvpx `vpxenc.c:434-435`). VP9 range: -9..=9.
    cpu_used: Option<i32>,
    /// `--threads` / `-t` (libvpx `vpxenc.c:198-199`).
    threads: Option<u32>,
    /// `--tile-columns` (libvpx `vpxenc.c:439-440`). log2, 0..=6.
    tile_columns: Option<u8>,
    /// `--min-q` (libvpx `vpxenc.c` min_quantizer).
    min_q: Option<u32>,
    /// `--max-q` (libvpx `vpxenc.c` max_quantizer).
    max_q: Option<u32>,
    /// `--cq-level` (libvpx `vpxenc.c:385`).
    cq_level: Option<u32>,
    /// `--noise-sensitivity` (libvpx `vpxenc.c:367-368`).
    noise_sensitivity: Option<u32>,
    /// A18 — `--temporal-layers <N>` (1..=3). Frame-rate-scalable
    /// temporal SVC. `1` (default) disables SVC; `2`/`3` activate the
    /// libvpx 0-1-0-1 / 0-2-1-2 patterns (`vp9_svc_layercontext.c`).
    /// libvpx's equivalent is `--ts-number-layers` on the `vpxenc` SVC
    /// path; rvp9 exposes the temporal count directly.
    temporal_layers: Option<u32>,
    /// Number of spatial SVC layers. `1` disables spatial SVC; `2` and `3`
    /// select the canonical half/full or quarter/half/full pyramids.
    spatial_layers: u8,
    /// Same-access-unit enhancement-layer prediction policy.
    inter_layer_pred: InterLayerPred,
    /// `--auto-alt-ref` (libvpx `vpxenc.c:436-438`). VP9 range 0..=6:
    /// 0 disables, 1 single ARF, 2+ enables multi-layer ARF.
    auto_alt_ref: Option<u32>,
    /// `--passes` / `-p` (libvpx `vpxenc.c:113-114`). Number of passes:
    /// `1` (default, the one-pass regulators) or `2` (two-pass VBR).
    /// When `2`, the encoder runs the per-sequence two-pass driver
    /// (`rvp9_encoder::encode_two_pass`): a stats-only first pass over the
    /// whole input, then a stats-driven second pass emitting the
    /// bitstream. That driver uses synchronous scheduling, one thread, and
    /// one tile, so positive `--lag-in-frames` / `--auto-alt-ref` requests,
    /// explicit `--threads` values above 1, and explicit `--tile-columns`
    /// values above 0 are rejected.
    passes: u32,
    /// `--enable-tpl` (libvpx `vpxenc.c:445-446`). 0/1 — temporal
    /// dependency model.
    enable_tpl: Option<bool>,
    /// `--lag-in-frames` (libvpx `vpxenc.c:220-221`).
    lag_in_frames: Option<u32>,
    /// Benchmark diagnostic: optional per-source-frame encode timing CSV.
    /// This is intentionally limited to the synchronous one-pass path so
    /// each timed push maps to exactly one emitted packet.
    frame_time_csv: Option<String>,
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
            // libvpx default `--end-usage=cbr` (the only mode rvp9's CLI
            // exposed pre-re-audit-pass-3); the cross-decoder gate runs
            // at this default so it stays byte-identical.
            rc_mode: RcMode::Cbr,
            output: None,
            limit: None,
            verbose: false,
            help: false,
            version: false,

            deadline: None,
            cpu_used: None,
            threads: None,
            tile_columns: None,
            min_q: None,
            max_q: None,
            cq_level: None,
            noise_sensitivity: None,
            temporal_layers: None,
            spatial_layers: 1,
            inter_layer_pred: InterLayerPred::On,
            auto_alt_ref: None,
            // Default 1 pass (the one-pass regulators / cross-decoder gate
            // config). `--passes 2` opts into two-pass VBR.
            passes: 1,
            enable_tpl: None,
            lag_in_frames: None,
            frame_time_csv: None,
        }
    }
}

/// The flag table — single source of truth. `apply: Some` ⇒ wired;
/// `apply: None` ⇒ recognised-but-not-yet (errors if used).
fn flag_table() -> Vec<Flag<EncOpts>> {
    vec![
        Flag {
            long: "help",
            short: None,
            arg: Arg::None,
            help: "show this help and exit",
            apply: Some(|o, _| {
                o.help = true;
                Ok(())
            }),
        },
        Flag {
            long: "version",
            short: None,
            arg: Arg::None,
            help: "print version and exit",
            apply: Some(|o, _| {
                o.version = true;
                Ok(())
            }),
        },
        Flag {
            long: "codec",
            short: None,
            arg: Arg::Required,
            help: "codec to use (vp9)",
            apply: Some(|_, v| validate_eq(v, "vp9", "--codec")),
        },
        Flag {
            long: "output",
            short: Some('o'),
            arg: Arg::Required,
            help: "output IVF file",
            apply: Some(|o, v| {
                o.output = Some(req(v, "--output")?);
                Ok(())
            }),
        },
        Flag {
            long: "width",
            short: Some('w'),
            arg: Arg::Required,
            help: "coded width (mult of 64)",
            apply: Some(|o, v| {
                o.width = Some(args::parse_value(v, "rvp9enc", "--width")?);
                Ok(())
            }),
        },
        Flag {
            long: "height",
            short: Some('h'),
            arg: Arg::Required,
            help: "coded height (positive and even)",
            apply: Some(|o, v| {
                o.height = Some(args::parse_value(v, "rvp9enc", "--height")?);
                Ok(())
            }),
        },
        Flag {
            long: "fps",
            short: None,
            arg: Arg::Required,
            help: "frame rate num/den (e.g. 30/1)",
            apply: Some(|o, v| {
                let (n, d) = parse_fps(v)?;
                o.fps_num = Some(n);
                o.fps_den = Some(d);
                Ok(())
            }),
        },
        Flag {
            long: "target-bitrate",
            short: None,
            arg: Arg::Required,
            help: "target bitrate in kbps",
            apply: Some(|o, v| {
                o.target_bitrate_kbps = args::parse_value(v, "rvp9enc", "--target-bitrate")?;
                Ok(())
            }),
        },
        Flag {
            long: "end-usage",
            short: None,
            arg: Arg::Required,
            help: "rate control mode (cbr/vbr/cq/q)",
            apply: Some(|o, v| {
                o.rc_mode = parse_end_usage(v)?;
                Ok(())
            }),
        },
        Flag {
            long: "passes",
            short: Some('p'),
            arg: Arg::Required,
            help: "number of passes (1 or 2; two-pass uses one thread/tile, no delayed scheduling; --lag-in-frames/--auto-alt-ref must be omitted or 0)",
            apply: Some(|o, v| {
                o.passes = parse_passes(v)?;
                Ok(())
            }),
        },
        Flag {
            long: "kf-max-dist",
            short: None,
            arg: Arg::Required,
            help: "max keyframe interval",
            apply: Some(|o, v| {
                o.gop_length = Some(args::parse_value(v, "rvp9enc", "--kf-max-dist")?);
                Ok(())
            }),
        },
        Flag {
            long: "aq-mode",
            short: None,
            arg: Arg::Required,
            help: "adaptive quant (0=off, 3=cyclic)",
            apply: Some(|o, v| {
                o.aq_mode = parse_aq_mode(v)?;
                Ok(())
            }),
        },
        Flag {
            long: "limit",
            short: None,
            arg: Arg::Required,
            help: "stop after N frames",
            apply: Some(|o, v| {
                o.limit = Some(args::parse_value(v, "rvp9enc", "--limit")?);
                Ok(())
            }),
        },
        Flag {
            long: "ivf",
            short: None,
            arg: Arg::None,
            help: "output IVF (the only container)",
            apply: Some(|_, _| Ok(())),
        },
        Flag {
            long: "verbose",
            short: Some('v'),
            arg: Arg::None,
            help: "verbose progress",
            apply: Some(|o, _| {
                o.verbose = true;
                Ok(())
            }),
        },
        // Recognized vpxenc-compatible flags without encoder plumbing remain
        // explicit `None` entries so help and parse errors describe them honestly.
        Flag {
            long: "profile",
            short: None,
            arg: Arg::Required,
            help: "bitstream profile (0..=3)",
            apply: None,
        },
        Flag {
            long: "rt",
            short: None,
            arg: Arg::None,
            help: "real-time deadline",
            apply: Some(|o, _| {
                o.deadline = Some(Deadline::Realtime);
                Ok(())
            }),
        },
        Flag {
            long: "good",
            short: None,
            arg: Arg::None,
            help: "good-quality deadline",
            apply: Some(|o, _| {
                o.deadline = Some(Deadline::Good);
                Ok(())
            }),
        },
        Flag {
            long: "best",
            short: None,
            arg: Arg::None,
            help: "best-quality deadline",
            apply: Some(|o, _| {
                o.deadline = Some(Deadline::Best);
                Ok(())
            }),
        },
        Flag {
            long: "cpu-used",
            short: None,
            arg: Arg::Required,
            help: "speed/quality tradeoff (-9..=9)",
            apply: Some(|o, v| {
                o.cpu_used = Some(parse_cpu_used(v)?);
                Ok(())
            }),
        },
        Flag {
            long: "threads",
            short: Some('t'),
            arg: Arg::Required,
            help: "encode threads (two-pass accepts 0 or 1)",
            apply: Some(|o, v| {
                o.threads = Some(args::parse_value(v, "rvp9enc", "--threads")?);
                Ok(())
            }),
        },
        Flag {
            long: "tile-columns",
            short: None,
            arg: Arg::Required,
            help: "log2 tile columns (0..=6; two-pass accepts 0)",
            apply: Some(|o, v| {
                o.tile_columns = Some(parse_range(v, "--tile-columns", 0, 6)? as u8);
                Ok(())
            }),
        },
        Flag {
            long: "tile-rows",
            short: None,
            arg: Arg::Required,
            help: "log2 tile rows (0..=2)",
            apply: None,
        },
        Flag {
            long: "kf-min-dist",
            short: None,
            arg: Arg::Required,
            help: "min keyframe interval",
            apply: None,
        },
        Flag {
            long: "disable-kf",
            short: None,
            arg: Arg::None,
            help: "disable keyframes",
            apply: None,
        },
        Flag {
            long: "min-q",
            short: None,
            arg: Arg::Required,
            help: "minimum quantizer",
            apply: Some(|o, v| {
                o.min_q = Some(args::parse_value(v, "rvp9enc", "--min-q")?);
                Ok(())
            }),
        },
        Flag {
            long: "max-q",
            short: None,
            arg: Arg::Required,
            help: "maximum quantizer",
            apply: Some(|o, v| {
                o.max_q = Some(args::parse_value(v, "rvp9enc", "--max-q")?);
                Ok(())
            }),
        },
        Flag {
            long: "cq-level",
            short: None,
            arg: Arg::Required,
            help: "constant/constrained quality level",
            apply: Some(|o, v| {
                o.cq_level = Some(args::parse_value(v, "rvp9enc", "--cq-level")?);
                Ok(())
            }),
        },
        Flag {
            long: "buf-sz",
            short: None,
            arg: Arg::Required,
            help: "client buffer size (ms)",
            apply: None,
        },
        Flag {
            long: "buf-initial-sz",
            short: None,
            arg: Arg::Required,
            help: "initial buffer size (ms)",
            apply: None,
        },
        Flag {
            long: "buf-optimal-sz",
            short: None,
            arg: Arg::Required,
            help: "optimal buffer size (ms)",
            apply: None,
        },
        Flag {
            long: "undershoot-pct",
            short: None,
            arg: Arg::Required,
            help: "datarate undershoot %",
            apply: None,
        },
        Flag {
            long: "overshoot-pct",
            short: None,
            arg: Arg::Required,
            help: "datarate overshoot %",
            apply: None,
        },
        Flag {
            long: "lossless",
            short: None,
            arg: Arg::Required,
            help: "lossless coding (0/1)",
            apply: None,
        },
        Flag {
            long: "frame-parallel",
            short: None,
            arg: Arg::Required,
            help: "frame-parallel decoding (0/1)",
            apply: None,
        },
        Flag {
            long: "error-resilient",
            short: None,
            arg: Arg::Required,
            help: "error resilient mode",
            apply: None,
        },
        Flag {
            long: "tune",
            short: None,
            arg: Arg::Required,
            help: "distortion metric to tune (psnr/ssim)",
            apply: None,
        },
        Flag {
            long: "tune-content",
            short: None,
            arg: Arg::Required,
            help: "content type to tune (default)",
            apply: Some(|_, v| validate_eq(v, "default", "--tune-content")),
        },
        Flag {
            long: "noise-sensitivity",
            short: None,
            arg: Arg::Required,
            help: "denoiser strength (0..=6)",
            apply: Some(|o, v| {
                o.noise_sensitivity = Some(parse_range(v, "--noise-sensitivity", 0, 6)?);
                Ok(())
            }),
        },
        Flag {
            long: "svc",
            short: None,
            arg: Arg::None,
            help: "scalable video coding",
            apply: None,
        },
        Flag {
            long: "temporal-layers",
            short: None,
            arg: Arg::Required,
            help: "temporal SVC layers (1..=3)",
            apply: Some(|o, v| {
                o.temporal_layers = Some(args::parse_value(v, "rvp9enc", "--temporal-layers")?);
                Ok(())
            }),
        },
        Flag {
            long: "spatial-layers",
            short: None,
            arg: Arg::Required,
            help: "spatial SVC layers (1..=3)",
            apply: Some(|o, v| {
                o.spatial_layers = parse_range(v, "--spatial-layers", 1, 3)? as u8;
                Ok(())
            }),
        },
        Flag {
            long: "inter-layer-pred",
            short: None,
            arg: Arg::Required,
            help: "spatial prediction (0=on, 3=constrained)",
            apply: Some(|o, v| {
                o.inter_layer_pred = parse_inter_layer_pred(v)?;
                Ok(())
            }),
        },
        Flag {
            long: "bit-depth",
            short: None,
            arg: Arg::Required,
            help: "bit depth (8/10/12)",
            apply: None,
        },
        Flag {
            long: "psnr",
            short: None,
            arg: Arg::None,
            help: "report PSNR",
            apply: None,
        },
        Flag {
            long: "webm",
            short: None,
            arg: Arg::None,
            help: "WebM output (not supported yet)",
            apply: None,
        },
        // ---- Long-tail vpxenc flags.
        // Unimplemented options remain explicit `None` entries until their
        // values affect encoder behavior.
        // SKIPPED per Arthur's audit: `--arnr-type` (vpxenc.c:378-379)
        // and `--output-partitions` / `-P` (vpxenc.c:151-153) are
        // OUT-OF-SCOPE — they survive in the "recognised-but-not-yet"
        // bucket via absence from the flag table.
        Flag {
            long: "auto-alt-ref",
            short: None,
            arg: Arg::Required,
            help: "auto alt-ref frames (0..=6; two-pass requires omitted or 0)",
            apply: Some(|o, v| {
                o.auto_alt_ref = Some(parse_range(v, "--auto-alt-ref", 0, 6)?);
                Ok(())
            }),
        },
        Flag {
            long: "arnr-maxframes",
            short: None,
            arg: Arg::Required,
            help: "AltRef max frames (0..=15)",
            apply: None,
        },
        Flag {
            long: "arnr-strength",
            short: None,
            arg: Arg::Required,
            help: "AltRef filter strength (0..=6)",
            apply: None,
        },
        Flag {
            long: "pass",
            short: None,
            arg: Arg::Required,
            help: "pass to execute (1 or 2)",
            apply: None,
        },
        Flag {
            long: "fpf",
            short: None,
            arg: Arg::Required,
            help: "first-pass statistics file",
            apply: None,
        },
        Flag {
            long: "rate-hist",
            short: None,
            arg: Arg::Required,
            help: "rate histogram bucket count",
            apply: None,
        },
        Flag {
            long: "q-hist",
            short: None,
            arg: Arg::Required,
            help: "quantizer histogram bucket count",
            apply: None,
        },
        Flag {
            long: "test-decode",
            short: None,
            arg: Arg::Required,
            help: "encode/decode mismatch test (off/warn/fatal)",
            apply: None,
        },
        Flag {
            long: "enable-tpl",
            short: None,
            arg: Arg::Required,
            help: "enable temporal dependency model (0/1)",
            apply: Some(|o, v| {
                o.enable_tpl = Some(parse_bool01(v, "--enable-tpl")?);
                Ok(())
            }),
        },
        Flag {
            long: "frame-boost",
            short: None,
            arg: Arg::Required,
            help: "frame periodic boost (0/1)",
            apply: None,
        },
        Flag {
            long: "max-inter-rate",
            short: None,
            arg: Arg::Required,
            help: "max P-frame bitrate (pct)",
            apply: None,
        },
        Flag {
            long: "min-gf-interval",
            short: None,
            arg: Arg::Required,
            help: "min gf/arf frame interval",
            apply: None,
        },
        Flag {
            long: "max-gf-interval",
            short: None,
            arg: Arg::Required,
            help: "max gf/arf frame interval",
            apply: None,
        },
        Flag {
            long: "color-space",
            short: None,
            arg: Arg::Required,
            help: "color space of input content",
            apply: None,
        },
        Flag {
            long: "alt-ref-aq",
            short: None,
            arg: Arg::Required,
            help: "adaptive quant for alt-ref (0/1)",
            apply: None,
        },
        Flag {
            long: "input-bit-depth",
            short: None,
            arg: Arg::Required,
            help: "input bit depth (8/10/12)",
            apply: None,
        },
        Flag {
            long: "lag-in-frames",
            short: None,
            arg: Arg::Required,
            help: "max frames to lag (two-pass requires omitted or 0)",
            apply: Some(|o, v| {
                o.lag_in_frames = Some(args::parse_value(v, "rvp9enc", "--lag-in-frames")?);
                Ok(())
            }),
        },
        Flag {
            long: "frame-time-csv",
            short: None,
            arg: Arg::Required,
            help: "benchmark diagnostic: per-frame encode times CSV",
            apply: Some(|o, v| {
                o.frame_time_csv = Some(req(v, "--frame-time-csv")?);
                Ok(())
            }),
        },
        Flag {
            long: "target-level",
            short: None,
            arg: Arg::Required,
            help: "target level (0/255 + 10..=62)",
            apply: None,
        },
        Flag {
            long: "row-mt",
            short: None,
            arg: Arg::Required,
            help: "row-based MT in VP9 (0/1)",
            apply: None,
        },
        Flag {
            long: "resize-allowed",
            short: None,
            arg: Arg::Required,
            help: "spatial resampling (0/1)",
            apply: None,
        },
        Flag {
            long: "resize-width",
            short: None,
            arg: Arg::Required,
            help: "resized encoded width",
            apply: None,
        },
        Flag {
            long: "resize-height",
            short: None,
            arg: Arg::Required,
            help: "resized encoded height",
            apply: None,
        },
        Flag {
            long: "quiet",
            short: Some('q'),
            arg: Arg::None,
            help: "do not print encode progress",
            // Encoding is already silent unless `--verbose` is present, so
            // libvpx's explicit quiet flag is a behavior-preserving no-op.
            apply: Some(|_, _| Ok(())),
        },
    ]
}

fn validate_eq(v: Option<String>, want: &str, flag: &str) -> Result<(), String> {
    match v.as_deref() {
        Some(s) if s == want => Ok(()),
        Some(other) => Err(format!(
            "rvp9enc: {flag}={other} not supported yet (only {want})"
        )),
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

/// Parse `--end-usage` (libvpx `vpxenc.c` `--end-usage`,
/// `vpx_encoder.h enum vpx_rc_mode`). rvp9 ports all four one-pass
/// regulators (CBR + the VBR/CQ/Q arms of
/// `RateControl::pick_q_and_bounds`); the 2-pass driver is out of
/// scope, so only the single-pass mode names are accepted here.
fn parse_end_usage(v: Option<String>) -> Result<RcMode, String> {
    match v.as_deref() {
        Some("cbr") => Ok(RcMode::Cbr),
        Some("vbr") => Ok(RcMode::Vbr),
        Some("cq") => Ok(RcMode::Cq),
        Some("q") => Ok(RcMode::Q),
        Some(other) => Err(format!(
            "rvp9enc: --end-usage={other} not understood (use cbr/vbr/cq/q)"
        )),
        None => Err("rvp9enc: --end-usage requires a value".to_string()),
    }
}

fn parse_fps(v: Option<String>) -> Result<(u32, u32), String> {
    let s = v.ok_or_else(|| "rvp9enc: --fps requires a value".to_string())?;
    match s.split_once('/') {
        Some((n, d)) => {
            let num = n
                .parse::<u32>()
                .map_err(|_| format!("rvp9enc: --fps: bad numerator '{n}'"))?;
            let den = d
                .parse::<u32>()
                .map_err(|_| format!("rvp9enc: --fps: bad denominator '{d}'"))?;
            if den == 0 {
                return Err("rvp9enc: --fps denominator must be non-zero".to_string());
            }
            Ok((num, den))
        }
        None => {
            let num = s
                .parse::<u32>()
                .map_err(|_| format!("rvp9enc: --fps: bad value '{s}'"))?;
            Ok((num, 1))
        }
    }
}

fn req(v: Option<String>, flag: &str) -> Result<String, String> {
    v.ok_or_else(|| format!("rvp9enc: {flag} requires a value"))
}

fn parse_cpu_used(v: Option<String>) -> Result<i32, String> {
    let n: i32 = args::parse_value(v, "rvp9enc", "--cpu-used")?;
    if !(-9..=9).contains(&n) {
        return Err(format!("rvp9enc: --cpu-used={n} out of range (-9..=9)"));
    }
    Ok(n)
}

fn parse_bool01(v: Option<String>, flag: &str) -> Result<bool, String> {
    match v.as_deref() {
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "rvp9enc: {flag}={other} not understood (use 0 or 1)"
        )),
        None => Err(format!("rvp9enc: {flag} requires a value")),
    }
}

fn parse_inter_layer_pred(v: Option<String>) -> Result<InterLayerPred, String> {
    match v.as_deref() {
        Some("0") => Ok(InterLayerPred::On),
        Some("3") => Ok(InterLayerPred::OnConstrained),
        Some(other) => Err(format!(
            "rvp9enc: --inter-layer-pred={other} not understood (use 0 or 3)"
        )),
        None => Err("rvp9enc: --inter-layer-pred requires a value".to_string()),
    }
}

fn parse_range(v: Option<String>, flag: &str, lo: u32, hi: u32) -> Result<u32, String> {
    let n: u32 = args::parse_value(v, "rvp9enc", flag)?;
    if n < lo || n > hi {
        return Err(format!("rvp9enc: {flag}={n} out of range ({lo}..={hi})"));
    }
    Ok(n)
}

/// libvpx `vpxenc.c:113-114` — `--passes` / `-p` is 1 or 2. `2` selects
/// the two-pass VBR driver (`rvp9_encoder::encode_two_pass`).
fn parse_passes(v: Option<String>) -> Result<u32, String> {
    match v.as_deref() {
        Some("1") => Ok(1),
        Some("2") => Ok(2),
        Some(other) => Err(format!(
            "rvp9enc: --passes={other} not understood (use 1 or 2)"
        )),
        None => Err("rvp9enc: --passes requires a value".to_string()),
    }
}

fn validate_frame_time_csv(opts: &EncOpts) -> Result<(), String> {
    if opts.frame_time_csv.is_none() {
        return Ok(());
    }
    if opts.passes != 1 {
        return Err(
            "--frame-time-csv requires one-pass encoding (--passes 1); two-pass encoding is not supported"
                .to_string(),
        );
    }
    if opts.lag_in_frames.unwrap_or(0) != 0 {
        return Err(
            "--frame-time-csv requires --lag-in-frames=0; nonzero lookahead cannot map push time to emitted frames"
                .to_string(),
        );
    }
    Ok(())
}

fn validate_two_pass_options(opts: &EncOpts) -> Result<(), String> {
    if opts.passes != 2 {
        return Ok(());
    }
    if opts.threads.is_some_and(|threads| threads > 1) {
        return Err(
            "--passes 2 supports one thread and one tile only; --threads must be omitted, 0, or 1"
                .to_string(),
        );
    }
    if opts
        .tile_columns
        .is_some_and(|tile_columns| tile_columns > 0)
    {
        return Err(
            "--passes 2 supports one thread and one tile only; --tile-columns must be omitted or 0"
                .to_string(),
        );
    }
    if opts
        .lag_in_frames
        .is_some_and(|lag_in_frames| lag_in_frames > 0)
    {
        return Err(
            "--passes 2 does not support delayed scheduling; --lag-in-frames must be omitted or 0"
                .to_string(),
        );
    }
    if opts
        .auto_alt_ref
        .is_some_and(|auto_alt_ref| auto_alt_ref > 0)
    {
        return Err(
            "--passes 2 does not support delayed scheduling; --auto-alt-ref must be omitted or 0"
                .to_string(),
        );
    }
    Ok(())
}

const USAGE: &str = "rvp9enc [options] -o <out.ivf> <input.y4m|input.yuv>";

fn main() -> ExitCode {
    let argv = match args::collect_args(std::env::args_os().skip(1), "rvp9enc") {
        Ok(argv) => argv,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
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
    validate_frame_time_csv(opts)?;
    validate_two_pass_options(opts)?;

    let input = match positionals {
        [one] => Path::new(one),
        [] => return Err("no input file given".to_string()),
        _ => return Err("expected exactly one input file".to_string()),
    };
    let output = opts
        .output
        .as_deref()
        .ok_or_else(|| "no output file (-o <out.ivf>)".to_string())?;

    let mut paths = vec![("input", input), ("output", Path::new(output))];
    if let Some(csv) = opts.frame_time_csv.as_deref() {
        paths.push(("frame-time CSV", Path::new(csv)));
    }
    let mut path_guard = path_guard::PathGuard::new(&paths)?;
    let (mut frames, info) = open_input(input, opts)?;
    let Some(first_frame) = frames.next_frame()? else {
        return Err("input contained no frames".to_string());
    };

    // Build and validate the encoder config via the shared
    // `build_encoder_config` mapping (the single CLI→encoder source of
    // truth; see its doc-comment for the per-flag libvpx citations).
    // Width stays SB-column aligned. Height may end in a partial SB row,
    // which is required by literal 720p and 1080p inputs.
    let config = build_encoder_config(
        opts,
        info.coded_w,
        info.coded_h,
        f64::from(info.fps_num) / f64::from(info.fps_den),
        opts.target_bitrate_kbps.saturating_mul(1000),
    );
    // Two-pass VBR (`--passes 2`): run the per-sequence two-pass driver.
    // A stats-only first pass over the whole input gathers per-frame
    // coded-error stats, then a stats-driven second pass emits the
    // bitstream. libvpx `vpxenc.c:113-114 --passes`; driver:
    // `rvp9_encoder::encode_two_pass`. Kept entirely separate from the
    // one-pass synchronous path below so that path stays byte-identical.
    if opts.spatial_layers > 1 && opts.passes != 1 {
        return Err("spatial SVC requires one-pass encoding (--passes 1)".to_string());
    }
    if opts.passes == 2 {
        let mut all_frames = vec![first_frame];
        while let Some(frame) = frames.next_frame()? {
            all_frames
                .try_reserve(1)
                .map_err(|_| "cannot allocate input frame list for two-pass encode".to_string())?;
            all_frames.push(frame);
        }
        return run_two_pass(
            opts,
            &config,
            &all_frames,
            info.coded_w,
            info.coded_h,
            info.fps_num,
            info.fps_den,
            output,
            &mut path_guard,
        );
    }

    // libvpx's VP9 codec defaults are g_threads=8 and a requested
    // log2_tile_columns=6. The latter is clamped to the legal range for
    // the coded width by the encoder. Explicit `--threads=0` is treated
    // as one worker, matching the no-extra-worker interpretation.
    let parallelism = build_encoder_parallelism(opts);
    if opts.spatial_layers > 1 {
        return run_spatial_one_pass(
            opts,
            config,
            parallelism,
            &first_frame,
            &mut frames,
            info,
            output,
            &mut path_guard,
        );
    }
    let mut encoder = StreamEncoder::new_with_parallelism(config, parallelism).map_err(|e| {
        format!(
            "{e} (note: coded width must be a multiple of 64 and coded \
             height must be positive and even — use -w/-h to crop)"
        )
    })?;

    // Encode one source frame at a time and write each available packet
    // immediately. The source frame and encoded packet are dropped before
    // the next iteration, keeping one-pass memory independent of duration.
    //
    // libvpx `vp9_encoder.c::vp9_get_compressed_data` splits the
    // pipeline so the encoder may return no packet on the first
    // `lag_in_frames` pushes (the lookahead ring fills before the
    // first pop fires). rvp9 mirrors that with
    // [`StreamEncoder::encode_frame_delayed`] (returns `None` until
    // steady-state) + [`StreamEncoder::flush`] (drains the tail at
    // end-of-stream — libvpx's `vpxenc.c::main` flush loop passing
    // a NULL frame).
    let header = IvfHeader::vp9(
        info.coded_w as u16,
        info.coded_h as u16,
        info.fps_num,
        info.fps_den,
    );
    let out = path_guard.create("output")?;
    let mut writer = IvfStreamWriter::new(BufWriter::new(out), &header)
        .map_err(|e| format!("writing IVF header: {e}"))?;
    let mut total_bits: u64 = 0;
    let mut frame_count: usize = 0;
    let mut frame_timings = Vec::new();
    {
        let mut encode_frame = |(y, u, v): &y4m::YuvFrameBuf| -> Result<(), String> {
            let view = YuvFrameRef {
                y,
                u,
                v,
                y_stride: info.coded_w as usize,
                u_stride: (info.coded_w / 2) as usize,
                v_stride: (info.coded_w / 2) as usize,
            };
            let started = opts.frame_time_csv.as_ref().map(|_| Instant::now());
            let encoded = encoder
                .encode_frame_delayed(&view)
                .map_err(|e| format!("encode_frame: {e}"))?;
            let encode_ns = started.map(|start| start.elapsed().as_nanos());
            let Some(enc) = encoded else {
                if opts.frame_time_csv.is_some() {
                    return Err(format!(
                        "--frame-time-csv expected one emitted packet for source frame {frame_count}, but lag-zero encode produced none"
                    ));
                }
                frame_count += 1;
                return Ok(());
            };
            if let Some(encode_ns) = encode_ns {
                frame_timings.push(FrameTiming {
                    frame_index: frame_count,
                    temporal_layer_id: enc.temporal_layer_id,
                    encode_ns,
                });
            }
            total_bits += u64::from(enc.actual_bits);
            writer
                .write_packet(&enc.data)
                .map_err(|e| format!("writing IVF packet: {e}"))?;
            frame_count += 1;
            Ok(())
        };
        encode_frame(&first_frame)?;
        let mut frame = y4m::YuvFrameBuf::default();
        while frames.next_frame_into(&mut frame)? {
            encode_frame(&frame)?;
        }
    }

    // End-of-stream: drain any frames still buffered in the
    // lookahead. With `lag_in_frames == 0` this is a no-op (the
    // ring is empty after the synchronous push/pop in
    // `encode_frame_delayed`).
    for r in encoder.flush() {
        let enc = r.map_err(|e| format!("encode_frame (flush): {e}"))?;
        if opts.frame_time_csv.is_some() {
            return Err(
                "--frame-time-csv expected no delayed packets after the lag-zero encode loop"
                    .to_string(),
            );
        }
        total_bits += u64::from(enc.actual_bits);
        writer
            .write_packet(&enc.data)
            .map_err(|e| format!("writing IVF packet: {e}"))?;
    }
    writer
        .finish()
        .map_err(|e| format!("finalizing IVF: {e}"))?;
    if opts.frame_time_csv.is_some() {
        write_frame_time_csv(&mut path_guard, &frame_timings)?;
    }

    // C2-β — measurement-only fire-rate of the size-gated ML-partition
    // breakout. Reads the global counter exposed by the encoder; off
    // unless `RVP9_C2B_STATS=1`. Proves the size gate: high on large
    // frames, ZERO on small.
    if std::env::var_os("RVP9_C2B_STATS").is_some() {
        let fired = rvp9_encoder::rdo::inter_dispatch::ML_PARTITION_BREAKOUT_FIRED
            .load(std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "rvp9enc: C2B ml-breakout fired {fired} times ({}x{})",
            info.coded_w, info.coded_h
        );
    }

    if opts.verbose {
        let secs = frame_count as f64 * f64::from(info.fps_den) / f64::from(info.fps_num);
        let kbps = if secs > 0.0 {
            (total_bits as f64 / secs) / 1000.0
        } else {
            0.0
        };
        eprintln!(
            "rvp9enc: encoded {} frame(s) at {}x{}, ~{:.1} kbps -> {output}",
            frame_count, info.coded_w, info.coded_h, kbps
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_spatial_one_pass(
    opts: &EncOpts,
    config: EncoderConfig,
    parallelism: EncoderParallelism,
    first_frame: &y4m::YuvFrameBuf,
    frames: &mut FrameSource,
    info: InputInfo,
    output: &str,
    path_guard: &mut path_guard::PathGuard,
) -> Result<(), String> {
    let spatial = SpatialSvcConfig {
        number_spatial_layers: opts.spatial_layers,
        inter_layer_pred: opts.inter_layer_pred,
    };
    let mut encoder = SpatialEncoder::new_with_parallelism(config, spatial, parallelism)
        .map_err(|error| format!("spatial SVC: {error}"))?;
    let header = IvfHeader::vp9(
        info.coded_w as u16,
        info.coded_h as u16,
        info.fps_num,
        info.fps_den,
    );
    let out = path_guard.create("output")?;
    let mut writer = IvfStreamWriter::new(BufWriter::new(out), &header)
        .map_err(|e| format!("writing IVF header: {e}"))?;
    let mut total_bits = 0_u64;
    let mut access_unit_count = 0_usize;
    let mut frame_timings = Vec::new();
    {
        let mut encode_access_unit = |(y, u, v): &y4m::YuvFrameBuf| -> Result<(), String> {
            let view = YuvFrameRef {
                y,
                u,
                v,
                y_stride: info.coded_w as usize,
                u_stride: (info.coded_w / 2) as usize,
                v_stride: (info.coded_w / 2) as usize,
            };
            let started = opts.frame_time_csv.as_ref().map(|_| Instant::now());
            let encoded = encoder
                .encode_access_unit(&view)
                .map_err(|e| format!("encode spatial access unit: {e}"))?;
            if let Some(started) = started {
                frame_timings.push(FrameTiming {
                    frame_index: access_unit_count,
                    temporal_layer_id: 0,
                    encode_ns: started.elapsed().as_nanos(),
                });
            }
            total_bits += encoded
                .layers
                .iter()
                .map(|layer| u64::from(layer.actual_bits))
                .sum::<u64>();
            writer
                .write_packet(&encoded.data)
                .map_err(|e| format!("writing spatial IVF packet: {e}"))?;
            access_unit_count += 1;
            Ok(())
        };
        encode_access_unit(first_frame)?;
        let mut frame = y4m::YuvFrameBuf::default();
        while frames.next_frame_into(&mut frame)? {
            encode_access_unit(&frame)?;
        }
    }
    writer
        .finish()
        .map_err(|e| format!("finalizing IVF: {e}"))?;
    if opts.frame_time_csv.is_some() {
        write_frame_time_csv(path_guard, &frame_timings)?;
    }

    if opts.verbose {
        let secs = access_unit_count as f64 * f64::from(info.fps_den) / f64::from(info.fps_num);
        let kbps = if secs > 0.0 {
            (total_bits as f64 / secs) / 1000.0
        } else {
            0.0
        };
        eprintln!(
            "rvp9enc: spatial SVC encoded {} access unit(s), {} layer(s) at {}x{}, ~{:.1} kbps -> {output}",
            access_unit_count, opts.spatial_layers, info.coded_w, info.coded_h, kbps
        );
    }
    Ok(())
}

struct FrameTiming {
    frame_index: usize,
    temporal_layer_id: u8,
    encode_ns: u128,
}

fn write_frame_time_csv(
    path_guard: &mut path_guard::PathGuard,
    timings: &[FrameTiming],
) -> Result<(), String> {
    let mut csv = String::from("frame_index,temporal_layer_id,encode_ns\n");
    for timing in timings {
        csv.push_str(&format!(
            "{},{},{}\n",
            timing.frame_index, timing.temporal_layer_id, timing.encode_ns
        ));
    }
    path_guard
        .create("frame-time CSV")?
        .write_all(csv.as_bytes())
        .map_err(|e| format!("writing frame-time CSV: {e}"))
}

/// Two-pass VBR encode path (`--passes 2`).
///
/// Collects every input frame into a borrowed [`YuvFrameRef`] and hands
/// the whole sequence to `rvp9_encoder::encode_two_pass`, which runs the
/// stats-only first pass then the stats-driven second pass and returns
/// the per-frame packets. Muxes the result into IVF exactly like the
/// one-pass path. Kept separate so the one-pass path stays untouched.
#[allow(clippy::too_many_arguments)]
fn run_two_pass(
    opts: &EncOpts,
    config: &EncoderConfig,
    frames: &[y4m::YuvFrameBuf],
    coded_w: u32,
    coded_h: u32,
    fps_num: u32,
    fps_den: u32,
    output: &str,
    path_guard: &mut path_guard::PathGuard,
) -> Result<(), String> {
    let views: Vec<YuvFrameRef<'_>> = frames
        .iter()
        .map(|(y, u, v)| YuvFrameRef {
            y,
            u,
            v,
            y_stride: coded_w as usize,
            u_stride: (coded_w / 2) as usize,
            v_stride: (coded_w / 2) as usize,
        })
        .collect();

    let encoded = encode_two_pass(*config, &views).map_err(|e| {
        format!(
            "two-pass encode: {e} (note: coded width must be a multiple \
             of 64 and coded height must be positive and even — use -w/-h to crop)"
        )
    })?;

    let total_bits: u64 = encoded.iter().map(|e| u64::from(e.actual_bits)).sum();
    let packets: Vec<&[u8]> = encoded.iter().map(|e| e.data.as_slice()).collect();
    let header = IvfHeader::vp9(coded_w as u16, coded_h as u16, fps_num, fps_den);
    let mut out = path_guard.create("output")?;
    write_ivf(&mut out, &header, &packets).map_err(|e| format!("writing IVF: {e}"))?;

    if opts.verbose {
        let secs = frames.len() as f64 * f64::from(fps_den) / f64::from(fps_num);
        let kbps = if secs > 0.0 {
            (total_bits as f64 / secs) / 1000.0
        } else {
            0.0
        };
        eprintln!(
            "rvp9enc: two-pass encoded {} frame(s) at {}x{}, ~{:.1} kbps -> {output}",
            frames.len(),
            coded_w,
            coded_h,
            kbps
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct InputInfo {
    fps_num: u32,
    fps_den: u32,
    coded_w: u32,
    coded_h: u32,
}

enum FrameSource {
    Y4m(y4m::Y4mReader),
    Raw(RawFrameReader),
}

impl FrameSource {
    fn next_frame(&mut self) -> Result<Option<y4m::YuvFrameBuf>, String> {
        match self {
            Self::Y4m(reader) => reader
                .next_frame()
                .map_err(|e| format!("reading y4m frame: {e}")),
            Self::Raw(reader) => reader.next_frame(),
        }
    }

    fn next_frame_into(&mut self, frame: &mut y4m::YuvFrameBuf) -> Result<bool, String> {
        match self {
            Self::Y4m(reader) => reader
                .next_frame_into(frame)
                .map_err(|e| format!("reading y4m frame: {e}")),
            Self::Raw(reader) => match reader.next_frame()? {
                Some(next) => {
                    *frame = next;
                    Ok(true)
                }
                None => Ok(false),
            },
        }
    }
}

struct RawFrameReader {
    file: BufReader<std::fs::File>,
    y_size: usize,
    uv_size: usize,
    remaining: usize,
}

impl RawFrameReader {
    fn open(
        input: &Path,
        coded_w: u32,
        coded_h: u32,
        limit: Option<usize>,
    ) -> Result<Self, String> {
        if !coded_dimensions_supported(coded_w, coded_h) {
            return Err(format!(
                "raw input geometry {coded_w}x{coded_h} is outside supported encoder limits \
                 (dimensions must be 1..={MAX_CODED_DIMENSION} and luma pixels must be 1..={MAX_CODED_LUMA_PIXELS})"
            ));
        }
        let width = coded_w as usize;
        let height = coded_h as usize;
        let y_size = width
            .checked_mul(height)
            .ok_or_else(|| format!("raw input geometry {coded_w}x{coded_h} is too large"))?;
        let uv_size = (width / 2)
            .checked_mul(height / 2)
            .ok_or_else(|| format!("raw input geometry {coded_w}x{coded_h} is too large"))?;
        let frame_size =
            y_size
                .checked_add(uv_size.checked_mul(2).ok_or_else(|| {
                    format!("raw input geometry {coded_w}x{coded_h} is too large")
                })?)
                .ok_or_else(|| format!("raw input geometry {coded_w}x{coded_h} is too large"))?;
        if frame_size == 0 {
            return Err(format!(
                "raw input geometry {coded_w}x{coded_h} has a zero-sized frame"
            ));
        }
        let file =
            std::fs::File::open(input).map_err(|e| format!("opening {}: {e}", input.display()))?;
        let file_len = file
            .metadata()
            .map_err(|e| format!("reading metadata for {}: {e}", input.display()))?
            .len();
        let frame_size_u64 =
            u64::try_from(frame_size).map_err(|_| "raw frame size is too large".to_string())?;
        let available_frames = file_len / frame_size_u64;
        let remaining = match limit {
            Some(limit) => {
                let limit_bytes = frame_size
                    .checked_mul(limit)
                    .ok_or_else(|| "--limit byte count is too large".to_string())?;
                if file_len < u64::try_from(limit_bytes).unwrap_or(u64::MAX)
                    && !file_len.is_multiple_of(frame_size_u64)
                {
                    return Err(format!(
                        "raw input length {file_len} is not a whole number of \
                         {coded_w}x{coded_h} I420 frames ({frame_size} bytes each)"
                    ));
                }
                usize::try_from(available_frames)
                    .unwrap_or(usize::MAX)
                    .min(limit)
            }
            None => {
                if !file_len.is_multiple_of(frame_size_u64) {
                    return Err(format!(
                        "raw input length {file_len} is not a whole number of \
                         {coded_w}x{coded_h} I420 frames ({frame_size} bytes each)"
                    ));
                }
                usize::try_from(available_frames).map_err(|_| {
                    format!("raw input contains too many frames: {available_frames}")
                })?
            }
        };
        Ok(Self {
            file: BufReader::new(file),
            y_size,
            uv_size,
            remaining,
        })
    }

    fn next_frame(&mut self) -> Result<Option<y4m::YuvFrameBuf>, String> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let y = read_raw_plane(&mut self.file, self.y_size, "luma")?;
        let u = read_raw_plane(&mut self.file, self.uv_size, "U chroma")?;
        let v = read_raw_plane(&mut self.file, self.uv_size, "V chroma")?;
        self.remaining -= 1;
        Ok(Some((y, u, v)))
    }
}

fn read_raw_plane(
    input: &mut BufReader<std::fs::File>,
    len: usize,
    label: &str,
) -> Result<Vec<u8>, String> {
    let mut plane = Vec::new();
    plane
        .try_reserve_exact(len)
        .map_err(|_| format!("cannot allocate {len} bytes for raw {label} plane"))?;
    plane.resize(len, 0);
    input
        .read_exact(&mut plane)
        .map_err(|e| format!("reading raw {label} plane: {e}"))?;
    Ok(plane)
}

fn open_input(input: &Path, opts: &EncOpts) -> Result<(FrameSource, InputInfo), String> {
    let is_y4m = input
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("y4m"));
    if is_y4m {
        let mut file =
            std::fs::File::open(input).map_err(|e| format!("opening {}: {e}", input.display()))?;
        let header =
            y4m::parse_header(&mut file).map_err(|e| format!("parsing y4m header: {e}"))?;
        let coded_w = opts.width.unwrap_or(header.w);
        let coded_h = opts.height.unwrap_or(header.h);
        if coded_w > header.w || coded_h > header.h {
            return Err(format!(
                "coded size {coded_w}x{coded_h} exceeds source {}x{}",
                header.w, header.h
            ));
        }
        let reader =
            y4m::Y4mReader::open(input, coded_w, coded_h, opts.limit.unwrap_or(usize::MAX))
                .map_err(|e| format!("opening y4m frames: {e}"))?;
        let (fps_num, fps_den) = match (opts.fps_num, opts.fps_den) {
            (Some(num), Some(den)) => (num, den),
            _ => (header.fps_num, header.fps_den),
        };
        Ok((
            FrameSource::Y4m(reader),
            InputInfo {
                fps_num,
                fps_den,
                coded_w,
                coded_h,
            },
        ))
    } else {
        let coded_w = opts
            .width
            .ok_or_else(|| "raw input requires -w/--width".to_string())?;
        let coded_h = opts
            .height
            .ok_or_else(|| "raw input requires -h/--height".to_string())?;
        let reader = RawFrameReader::open(input, coded_w, coded_h, opts.limit)?;
        Ok((
            FrameSource::Raw(reader),
            InputInfo {
                fps_num: opts.fps_num.unwrap_or(30),
                fps_den: opts.fps_den.unwrap_or(1),
                coded_w,
                coded_h,
            },
        ))
    }
}

/// Build the [`EncoderConfig`] from the parsed [`EncOpts`] plus the
/// runtime-derived geometry/bitrate. This is the SINGLE source of truth
/// for the CLI→encoder mapping: `run()` calls it for the real encode,
/// and the PROOF tests call it (via [`build_encoder_config_for_test`])
/// so they exercise exactly the production plumbing — if a field's
/// wiring is stripped, both the encode and the test break.
///
/// libvpx flag→config citations: `--end-usage` /
/// `vpx_encoder.h enum vpx_rc_mode` → `rc_mode`; `--cq-level`
/// (`vpxenc.c:385`) → `cq_level`; `--auto-alt-ref` (`vpxenc.c:436-438`)
/// → `enable_alt_ref`.
fn build_encoder_config(
    opts: &EncOpts,
    width: u32,
    height: u32,
    framerate: f64,
    target_bitrate_bps: u32,
) -> EncoderConfig {
    EncoderConfig {
        width,
        height,
        framerate,
        target_bitrate_bps,
        gop_length: opts.gop_length.unwrap_or(60),
        aq_mode: opts.aq_mode,
        use_prev_frame_mvs: false,
        // W0.2/W0.4 (vpxenc parity): `--cpu-used` + `--rt`/`--good`/
        // `--best` deadline tri-state. Defaults pin to rvp9's actual
        // op-point (`cpu-used=7 --rt`); flip to (0, Good) when W4 lands.
        speed: opts.cpu_used.unwrap_or(7),
        deadline: opts.deadline.unwrap_or(Deadline::Realtime),
        // W5.1 plumb (re-audit pass 2 follow-up): CLI's
        // `--noise-sensitivity <0..=6>` feeds the encoder's
        // `noise_sensitivity` gate. Default `0` (estimator off) matches
        // vpxenc's CLI default; cap to the 0..=6 range libvpx supports.
        noise_sensitivity: opts.noise_sensitivity.map(|v| v.min(6) as u8).unwrap_or(0),
        // PHASE-3 integration: CLI `--min-q` / `--max-q` (W7.8) →
        // EncoderConfig. Defaults mirror RateControlConfig::default().
        min_qindex: opts.min_q.map(|v| v.min(255) as u8).unwrap_or(4),
        max_qindex: opts.max_q.map(|v| v.min(255) as u8).unwrap_or(255),
        // W6.2(vpxenc parity, HBD scaffold): pixel bit-depth. rvp9 only
        // encodes 8-bit today; --bit-depth remains explicitly unsupported
        // until the HBD kernel pairs land.
        bit_depth: rvp9_core::BitDepth::Bd8,
        // vpxenc `--lag-in-frames` (libvpx `vpxenc.c:220-221`, capped at
        // `MAX_LAG_BUFFERS = 25`). The lookahead context also clamps
        // internally; the explicit cap keeps the domain matching libvpx.
        lag_in_frames: opts
            .lag_in_frames
            .map(|v| v.min(MAX_LAG_BUFFERS as u32))
            .unwrap_or(0),
        // Re-audit pass 3: `--auto-alt-ref` (libvpx `vpxenc.c:436-438`)
        // now opts ALTREF emission in. Any non-zero value enables it
        // (libvpx's 1 = single ARF, 2+ = multi-layer; rvp9 emits the
        // single-ARF path either way). The encoder self-gates on
        // `lag_in_frames` being large enough to feed the temporal
        // filter's peek, so absent/0 keeps the legacy
        // single-show-frame path (gate stays byte-identical).
        enable_alt_ref: opts.auto_alt_ref.map(|v| v > 0).unwrap_or(false),
        // Re-audit pass 3: `--end-usage` (libvpx
        // `vpx_encoder.h enum vpx_rc_mode`) selects the regulator
        // `RateControl::pick_q_and_bounds` dispatches on. Default Cbr.
        rc_mode: opts.rc_mode,
        // Re-audit pass 3: `--cq-level` (libvpx `vpxenc.c:385`) — the
        // quality target consumed by the Cq (floor) / Q (pinned) arms.
        // Inert under Cbr/Vbr. Cap to the 0..=255 q-index domain.
        cq_level: opts.cq_level.map(|v| v.min(255) as u8).unwrap_or(0),
        // A18 — `--temporal-layers <N>` (1..=3) → temporal SVC. Default 1
        // (SVC off). `StreamEncoder::new` validates the range and the
        // SVC/ALTREF mutual exclusion; values outside 1..=3 are rejected
        // there with a clear `InvalidConfig`.
        temporal_layers: opts.temporal_layers.unwrap_or(1),
        // A23 — `--enable-tpl` (libvpx `vpxenc.c:445-446`) → TPL rdmult
        // pass. Default OFF (cross-decoder gate runs `enable_tpl = false`).
        enable_tpl: opts.enable_tpl.unwrap_or(false),
        stage4a_product_split_override: Default::default(),
    }
}

/// Map vpxenc's worker/tile CLI controls onto the encoder-side runtime
/// configuration.
fn build_encoder_parallelism(opts: &EncOpts) -> EncoderParallelism {
    EncoderParallelism {
        max_threads: opts.threads.unwrap_or(8).max(1),
        requested_log2_tile_cols: opts.tile_columns.unwrap_or(6),
    }
}

/// Test-only thin wrapper over [`build_encoder_config`] pinning a
/// minimal 64×64 / 30 fps geometry so the PROOF tests can drive the
/// real production mapping with just an [`EncOpts`].
#[cfg(test)]
fn build_encoder_config_for_test(opts: &EncOpts) -> EncoderConfig {
    build_encoder_config(
        opts,
        64,
        64,
        30.0,
        opts.target_bitrate_kbps.saturating_mul(1000),
    )
}

#[cfg(test)]
mod tests {
    //! Focused parser and configuration tests for implemented encoder flags.

    use super::*;
    use std::path::PathBuf;

    fn s(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_string()).collect()
    }

    fn raw_geometry_test_path(tag: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rvp9enc_raw_geometry_{}_{}_{}.yuv",
            std::process::id(),
            id,
            tag
        ))
    }

    #[test]
    fn raw_reader_checks_encoder_geometry_before_file_io() {
        let invalid = [
            (MAX_CODED_DIMENSION + 64, 64),
            (64, MAX_CODED_DIMENSION + 64),
            (8_192, 8_194),
            (u32::MAX, u32::MAX),
            (0, 64),
            (64, 0),
        ];
        for (width, height) in invalid {
            let missing = raw_geometry_test_path("missing");
            let error = match RawFrameReader::open(&missing, width, height, None) {
                Err(error) => error,
                Ok(_) => panic!("invalid geometry {width}x{height} unexpectedly opened"),
            };
            assert!(
                error.contains("is outside supported encoder limits"),
                "geometry {width}x{height} must fail on the shared geometry limit before file I/O, got: {error}"
            );
            assert!(
                error.contains(&format!("{width}x{height}")),
                "geometry-limit error must include {width}x{height}, got: {error}"
            );
            assert!(
                error.contains(&MAX_CODED_DIMENSION.to_string())
                    && error.contains(&MAX_CODED_LUMA_PIXELS.to_string()),
                "geometry-limit error must include supported encoder limits, got: {error}"
            );
            assert!(
                !error.contains("opening")
                    && !error.contains("too large")
                    && !error.contains("zero-sized"),
                "geometry {width}x{height} must be rejected before file or size errors, got: {error}"
            );
        }

        let empty = raw_geometry_test_path("empty");
        std::fs::write(&empty, []).expect("create empty raw control file");
        for (width, height) in [(64, 64), (MAX_CODED_DIMENSION, 2), (8_192, 8_192)] {
            let mut reader = RawFrameReader::open(&empty, width, height, None)
                .unwrap_or_else(|error| panic!("valid geometry {width}x{height}: {error}"));
            assert!(
                reader
                    .next_frame()
                    .expect("empty file has no raw frame")
                    .is_none(),
                "empty control file should have no frame for {width}x{height}"
            );
        }
        std::fs::remove_file(&empty).expect("remove empty raw control file");
    }

    #[test]
    fn threads_and_tile_columns_reach_encoder_parallelism() {
        let table = flag_table();
        let mut opts = EncOpts::default();
        args::parse(
            &table,
            &mut opts,
            &s(&["--threads=4", "--tile-columns=2"]),
            "rvp9enc",
        )
        .unwrap();
        let parallelism = build_encoder_parallelism(&opts);
        assert_eq!(
            parallelism,
            EncoderParallelism {
                max_threads: 4,
                requested_log2_tile_cols: 2,
            }
        );

        let defaults = build_encoder_parallelism(&EncOpts::default());
        assert_eq!(defaults.max_threads, 8);
        assert_eq!(defaults.requested_log2_tile_cols, 6);

        let mut zero_threads = EncOpts::default();
        args::parse(&table, &mut zero_threads, &s(&["--threads=0"]), "rvp9enc").unwrap();
        assert_eq!(build_encoder_parallelism(&zero_threads).max_threads, 1);
    }

    #[test]
    fn spatial_svc_flags_select_supported_topology_and_prediction() {
        let table = flag_table();
        let mut opts = EncOpts::default();
        args::parse(
            &table,
            &mut opts,
            &s(&["--spatial-layers=3", "--inter-layer-pred=3"]),
            "rvp9enc",
        )
        .unwrap();
        assert_eq!(opts.spatial_layers, 3);
        assert_eq!(opts.inter_layer_pred, InterLayerPred::OnConstrained);

        for unsupported in ["1", "2"] {
            let mut invalid = EncOpts::default();
            let error = args::parse(
                &table,
                &mut invalid,
                &s(&[&format!("--inter-layer-pred={unsupported}")]),
                "rvp9enc",
            )
            .unwrap_err();
            assert!(error.contains("use 0 or 3"), "{error}");
        }
    }

    #[test]
    fn matched_default_content_and_quiet_are_supported() {
        let table = flag_table();
        let mut opts = EncOpts::default();
        args::parse(
            &table,
            &mut opts,
            &s(&["--tune-content=default", "--quiet"]),
            "rvp9enc",
        )
        .unwrap();
    }

    #[test]
    fn frame_time_csv_parses_and_requires_synchronous_one_pass() {
        let table = flag_table();
        let mut opts = EncOpts::default();
        args::parse(
            &table,
            &mut opts,
            &s(&["--frame-time-csv=/tmp/frame-times.csv"]),
            "rvp9enc",
        )
        .unwrap();
        assert_eq!(opts.frame_time_csv.as_deref(), Some("/tmp/frame-times.csv"));
        assert!(validate_frame_time_csv(&opts).is_ok());

        opts.passes = 2;
        let err = validate_frame_time_csv(&opts).unwrap_err();
        assert!(
            err.contains("one-pass") && err.contains("two-pass"),
            "{err}"
        );

        opts.passes = 1;
        opts.lag_in_frames = Some(1);
        let err = validate_frame_time_csv(&opts).unwrap_err();
        assert!(
            err.contains("lag-in-frames=0") && err.contains("nonzero"),
            "{err}"
        );
    }

    #[test]
    fn two_pass_rejects_positive_lookahead() {
        for lag_in_frames in [1, 25] {
            let opts = EncOpts {
                passes: 2,
                lag_in_frames: Some(lag_in_frames),
                ..EncOpts::default()
            };
            let result = validate_two_pass_options(&opts);
            assert!(
                result.is_err(),
                "--passes 2 must reject --lag-in-frames={lag_in_frames}"
            );
            let error = result.expect_err("positive two-pass lag must be rejected");
            assert!(
                error.contains("--passes 2")
                    && error.contains("--lag-in-frames")
                    && error.contains("delayed scheduling")
                    && error.contains("omitted or 0"),
                "two-pass lag rejection must explain the supported contract, got: {error}"
            );
        }
    }

    #[test]
    fn two_pass_rejects_positive_auto_alt_ref() {
        for auto_alt_ref in [1, 6] {
            let opts = EncOpts {
                passes: 2,
                auto_alt_ref: Some(auto_alt_ref),
                ..EncOpts::default()
            };
            let result = validate_two_pass_options(&opts);
            assert!(
                result.is_err(),
                "--passes 2 must reject --auto-alt-ref={auto_alt_ref}"
            );
            let error = result.expect_err("positive two-pass ALTREF must be rejected");
            assert!(
                error.contains("--passes 2")
                    && error.contains("--auto-alt-ref")
                    && error.contains("delayed scheduling")
                    && error.contains("omitted or 0"),
                "two-pass ALTREF rejection must explain the supported contract, got: {error}"
            );
        }
    }

    #[test]
    fn two_pass_lookahead_controls_accept_defaults_zero_and_one_pass_values() {
        let defaults = EncOpts {
            passes: 2,
            ..EncOpts::default()
        };
        assert!(
            validate_two_pass_options(&defaults).is_ok(),
            "omitted two-pass lookahead controls must remain valid"
        );

        let explicit_zero = EncOpts {
            passes: 2,
            lag_in_frames: Some(0),
            auto_alt_ref: Some(0),
            ..EncOpts::default()
        };
        assert!(
            validate_two_pass_options(&explicit_zero).is_ok(),
            "explicit zero two-pass lookahead controls must remain valid"
        );

        let one_pass_positive = EncOpts {
            passes: 1,
            lag_in_frames: Some(25),
            auto_alt_ref: Some(6),
            ..EncOpts::default()
        };
        assert!(
            validate_two_pass_options(&one_pass_positive).is_ok(),
            "one-pass positive lookahead controls must remain valid"
        );
    }

    #[test]
    fn non_default_content_tuning_remains_unsupported() {
        let table = flag_table();
        let mut opts = EncOpts::default();
        let err =
            args::parse(&table, &mut opts, &s(&["--tune-content=screen"]), "rvp9enc").unwrap_err();
        assert!(err.contains("not supported yet (only default)"), "{err}");
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

        // The documented range, including both endpoints, lands on the
        // encoder config without remapping valid values.
        for value in [0, 1, 3, 6] {
            let mut o = EncOpts::default();
            let argument = format!("--noise-sensitivity={value}");
            let parsed = args::parse(&table, &mut o, &s(&[argument.as_str()]), "rvp9enc");
            assert!(
                parsed.is_ok(),
                "valid {argument} must be accepted: {parsed:?}"
            );
            let cfg = build_encoder_config_for_test(&o);
            assert_eq!(
                cfg.noise_sensitivity, value as u8,
                "{argument} must plumb exactly"
            );
        }

        // Keep the defensive builder clamp to the documented CLI/API
        // range even though parsing rejects out-of-range user input.
        let o = EncOpts {
            noise_sensitivity: Some(99),
            ..Default::default()
        };
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.noise_sensitivity, 6, "99 must clamp to 6");

        // Verify the encoder accepts the wired config end-to-end.
        let mut o = EncOpts::default();
        args::parse(&table, &mut o, &s(&["--noise-sensitivity=1"]), "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        let enc = StreamEncoder::new(cfg).expect("StreamEncoder must accept --noise-sensitivity=1");
        assert_eq!(enc.config().noise_sensitivity, 1);
    }

    #[test]
    fn noise_sensitivity_rejects_out_of_range_values_in_both_cli_forms() {
        let table = flag_table();
        let cases = [
            ("inline", "7", s(&["--noise-sensitivity=7"])),
            ("separate", "7", s(&["--noise-sensitivity", "7"])),
            ("inline", "99", s(&["--noise-sensitivity=99"])),
            ("separate", "99", s(&["--noise-sensitivity", "99"])),
            (
                "inline",
                "4294967295",
                s(&["--noise-sensitivity=4294967295"]),
            ),
            (
                "separate",
                "4294967295",
                s(&["--noise-sensitivity", "4294967295"]),
            ),
        ];
        let mut failures = Vec::new();

        for (form, value, argv) in cases {
            let mut opts = EncOpts::default();
            let error = args::parse(&table, &mut opts, &argv, "rvp9enc");
            let expected = format!("--noise-sensitivity={value} out of range (0..=6)");
            match error {
                Ok(_) => failures.push(format!("{form} value {value} was accepted")),
                Err(error) if !error.contains(&expected) => failures.push(format!(
                    "{form} value {value} returned {error:?}, expected {expected:?}"
                )),
                Err(_) => {}
            }
        }

        assert!(
            failures.is_empty(),
            "--noise-sensitivity must reject every value above 6 in both forms: {}",
            failures.join("; ")
        );
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

        // Default (flags omitted) → measured production bounds (4, 255).
        let o = EncOpts::default();
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.min_qindex, 4, "default min_qindex must be 4");
        assert_eq!(cfg.max_qindex, 255, "default max_qindex must be 255");

        // Explicit values land on the config.
        let mut o = EncOpts::default();
        args::parse(
            &table,
            &mut o,
            &s(&["--min-q=10", "--max-q=200"]),
            "rvp9enc",
        )
        .unwrap();
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
        args::parse(
            &table,
            &mut o,
            &s(&["--min-q=200", "--max-q=10"]),
            "rvp9enc",
        )
        .unwrap();
        let cfg = build_encoder_config_for_test(&o);
        assert!(
            StreamEncoder::new(cfg).is_err(),
            "min_qindex > max_qindex must be rejected by validation",
        );
    }

    /// PROOF (re-audit pass 3 — `--end-usage` CLI reachability).
    ///
    /// Re-audit pass 3 (`wrk_docs/2026.05.26 - REAUDIT pass 3 ...`,
    /// commit 511ee0e) found the VBR/CQ/Q one-pass regulators are
    /// fully ported and byte-exact-verified, but a CLI user could not
    /// reach them: `--end-usage` only accepted the literal "cbr" and
    /// `EncoderConfig` had no `rc_mode` field, so
    /// `to_rate_control_config` always fell back to the
    /// `RateControlConfig` default (Cbr).
    ///
    /// This PROOF drives the END-TO-END path: parse → `EncOpts` →
    /// `build_encoder_config` (the production mapping) → `EncoderConfig`
    /// → `StreamEncoder::new` → `rate_control().config()`. The last hop
    /// proves `to_rate_control_config` actually carried `rc_mode` to
    /// the controller (the `EncoderConfig.rc_mode` assertion alone
    /// would only prove parsing). A stripped `..defaults` mapping or a
    /// reverted `--end-usage` handler reverts the controller to Cbr and
    /// fails this test.
    ///
    /// libvpx flag: `vpxenc.c --end-usage` /
    /// `vpx_encoder.h enum vpx_rc_mode`.
    #[test]
    fn reaudit3_end_usage_plumbs_into_rate_control() {
        let table = flag_table();
        let req = ["--target-bitrate=500", "-w", "64", "-h", "64"];

        // Default (flag omitted) → Cbr (the gate's mode; must stay so).
        let o = EncOpts::default();
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.rc_mode, RcMode::Cbr, "default --end-usage must be cbr");
        let enc = StreamEncoder::new(cfg).expect("cbr config must build");
        assert_eq!(
            enc.rate_control().config().rc_mode,
            RcMode::Cbr,
            "default mode must reach the controller as Cbr",
        );

        // --end-usage=vbr → Vbr, end to end.
        let mut o = EncOpts::default();
        let mut a = vec!["--end-usage=vbr".to_string()];
        a.extend(req.iter().map(|s| s.to_string()));
        args::parse(&table, &mut o, &a, "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.rc_mode, RcMode::Vbr, "--end-usage=vbr must set rc_mode");
        let enc = StreamEncoder::new(cfg).expect("vbr config must build");
        assert_eq!(
            enc.rate_control().config().rc_mode,
            RcMode::Vbr,
            "vbr must reach the controller (proves to_rate_control_config carried it)",
        );

        // --end-usage=cq + --cq-level=N → Cq, cq_level == N, end to end.
        let mut o = EncOpts::default();
        let mut a = vec!["--end-usage=cq".to_string(), "--cq-level=48".to_string()];
        a.extend(req.iter().map(|s| s.to_string()));
        args::parse(&table, &mut o, &a, "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.rc_mode, RcMode::Cq, "--end-usage=cq must set rc_mode");
        assert_eq!(cfg.cq_level, 48, "--cq-level=48 must set cq_level");
        let enc = StreamEncoder::new(cfg).expect("cq config must build");
        assert_eq!(enc.rate_control().config().rc_mode, RcMode::Cq);
        assert_eq!(
            enc.rate_control().config().cq_level,
            48,
            "cq_level must reach the controller",
        );

        // --end-usage=q → Q.
        let mut o = EncOpts::default();
        let mut a = vec!["--end-usage=q".to_string(), "--cq-level=60".to_string()];
        a.extend(req.iter().map(|s| s.to_string()));
        args::parse(&table, &mut o, &a, "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(cfg.rc_mode, RcMode::Q, "--end-usage=q must set rc_mode");
        assert_eq!(cfg.cq_level, 60);
        let enc = StreamEncoder::new(cfg).expect("q config must build");
        assert_eq!(enc.rate_control().config().rc_mode, RcMode::Q);

        // --cq-level out of range clamps to 255 (q-index domain).
        let o = EncOpts {
            cq_level: Some(9999),
            ..Default::default()
        };
        let cfg = build_encoder_config_for_test(&o);
        assert_eq!(
            cfg.cq_level, 255,
            "cq_level must clamp to the 0..=255 q-index domain"
        );

        // An unknown mode is rejected at parse time (not silently Cbr).
        let mut o = EncOpts::default();
        assert!(
            args::parse(&table, &mut o, &s(&["--end-usage=bogus"]), "rvp9enc").is_err(),
            "--end-usage=bogus must reject, not default to cbr",
        );
    }

    /// PROOF (re-audit pass 3 — `--auto-alt-ref` CLI reachability).
    ///
    /// `--auto-alt-ref` was parsed into `opts.auto_alt_ref` (range
    /// 0..=6) but never consumed: `enable_alt_ref: false` was hardcoded
    /// at the `EncoderConfig` construction site, so the ALTREF emission
    /// path (fully ported + byte-exact in `w8_altref_emit`) was
    /// CLI-unreachable. This PROOF pins the mapping: any non-zero value
    /// enables ALTREF; 0 / absent leaves it off (the gate's default).
    ///
    /// libvpx flag: `vpxenc.c:436-438 --auto-alt-ref` (VP9 0..=6).
    #[test]
    fn reaudit3_auto_alt_ref_plumbs_into_encoder_config() {
        let table = flag_table();

        // Absent → false (the cross-decoder gate's default).
        let o = EncOpts::default();
        let cfg = build_encoder_config_for_test(&o);
        assert!(!cfg.enable_alt_ref, "absent --auto-alt-ref must be false");

        // --auto-alt-ref=0 → false (explicit disable).
        let mut o = EncOpts::default();
        args::parse(&table, &mut o, &s(&["--auto-alt-ref=0"]), "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        assert!(!cfg.enable_alt_ref, "--auto-alt-ref=0 must be false");

        // --auto-alt-ref=1 → true (single ARF).
        let mut o = EncOpts::default();
        args::parse(&table, &mut o, &s(&["--auto-alt-ref=1"]), "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        assert!(cfg.enable_alt_ref, "--auto-alt-ref=1 must enable ALTREF");

        // --auto-alt-ref=6 (max) → true (multi-layer maps to enable).
        let mut o = EncOpts::default();
        args::parse(&table, &mut o, &s(&["--auto-alt-ref=6"]), "rvp9enc").unwrap();
        let cfg = build_encoder_config_for_test(&o);
        assert!(cfg.enable_alt_ref, "--auto-alt-ref=6 must enable ALTREF");
    }
}
