//! rvp9dec: VP9 decoder CLI.
//!
//! Mirrors libvpx `vpxdec`'s command-line interface where practical.
//! The decoder core lives in `rvp9-decoder` (Stage 3 — conformant
//! against the official VP9 test vectors); this binary is the I/O shell
//! around it: container in → decode → Y4M / raw / MD5 out.
//!
//! Flags that map to implemented behaviour are wired; everything else in
//! the `vpxdec` surface is listed by `--help` and errors with
//! "not supported yet" if used (see `rvp9_cli::args`).

use rvp9_cli::args::{self, Arg, Flag};
use rvp9_core::frame::PlaneRef;
use rvp9_decoder::Decoder;
use rvp9_io::container::read_packets;
use rvp9_io::md5::{hex16, md5_of_yv12_visible};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::ExitCode;

/// Output pixel format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutFmt {
    /// Not chosen yet — resolved from `-o` extension / flags at run time.
    Unset,
    /// Y4M (YUV4MPEG2) container.
    Y4m,
    /// Raw I420 planes (Y, U, V), no container.
    RawI420,
    /// Raw YV12 planes (Y, V, U), no container.
    RawYv12,
}

/// Per-frame MD5 reporting mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Md5Mode {
    /// No digest output.
    Off,
    /// One `<hex>` line per displayed frame (`vpxdec --md5`).
    PerFrame,
    /// `<index>: <hex>` per displayed frame (`vpxdec --framemd5`).
    Indexed,
}

/// Parsed decoder options.
struct DecOpts {
    output: Option<String>,
    format: OutFmt,
    md5: Md5Mode,
    limit: Option<usize>,
    skip: usize,
    summary: bool,
    noblit: bool,
    help: bool,
    version: bool,
}

impl Default for DecOpts {
    fn default() -> Self {
        Self {
            output: None,
            format: OutFmt::Unset,
            md5: Md5Mode::Off,
            limit: None,
            skip: 0,
            summary: false,
            noblit: false,
            help: false,
            version: false,
        }
    }
}

/// The flag table — the single source of truth for the CLI surface.
/// `apply: Some` ⇒ wired; `apply: None` ⇒ recognised-but-not-yet.
fn flag_table() -> Vec<Flag<DecOpts>> {
    vec![
        Flag { long: "help", short: None, arg: Arg::None, help: "show this help and exit",
            apply: Some(|o, _| { o.help = true; Ok(()) }) },
        Flag { long: "version", short: None, arg: Arg::None, help: "print version and exit",
            apply: Some(|o, _| { o.version = true; Ok(()) }) },
        Flag { long: "codec", short: None, arg: Arg::Required, help: "codec to use (vp9)",
            apply: Some(|_, v| validate_codec(v)) },
        Flag { long: "output", short: Some('o'), arg: Arg::Required, help: "output file (else stdout)",
            apply: Some(|o, v| { o.output = Some(req(v, "--output")?); Ok(()) }) },
        Flag { long: "y4m", short: None, arg: Arg::None, help: "output YUV4MPEG2",
            apply: Some(|o, _| { o.format = OutFmt::Y4m; Ok(()) }) },
        Flag { long: "i420", short: None, arg: Arg::None, help: "output raw I420 (Y,U,V)",
            apply: Some(|o, _| { o.format = OutFmt::RawI420; Ok(()) }) },
        Flag { long: "yv12", short: None, arg: Arg::None, help: "output raw YV12 (Y,V,U)",
            apply: Some(|o, _| { o.format = OutFmt::RawYv12; Ok(()) }) },
        Flag { long: "rawvideo", short: None, arg: Arg::None, help: "output raw I420, no container",
            apply: Some(|o, _| { o.format = OutFmt::RawI420; Ok(()) }) },
        Flag { long: "md5", short: None, arg: Arg::None, help: "print per-frame MD5",
            apply: Some(|o, _| { o.md5 = Md5Mode::PerFrame; Ok(()) }) },
        Flag { long: "framemd5", short: None, arg: Arg::None, help: "print indexed per-frame MD5",
            apply: Some(|o, _| { o.md5 = Md5Mode::Indexed; Ok(()) }) },
        Flag { long: "limit", short: None, arg: Arg::Required, help: "stop after N displayed frames",
            apply: Some(|o, v| { o.limit = Some(args::parse_value(v, "rvp9dec", "--limit")?); Ok(()) }) },
        Flag { long: "skip", short: None, arg: Arg::Required, help: "skip N displayed frames",
            apply: Some(|o, v| { o.skip = args::parse_value(v, "rvp9dec", "--skip")?; Ok(()) }) },
        Flag { long: "summary", short: None, arg: Arg::None, help: "print a decode summary",
            apply: Some(|o, _| { o.summary = true; Ok(()) }) },
        Flag { long: "noblit", short: None, arg: Arg::None, help: "decode only, no pixel output",
            apply: Some(|o, _| { o.noblit = true; Ok(()) }) },
        // Recognised but not implemented yet — error if used.
        Flag { long: "threads", short: Some('t'), arg: Arg::Required, help: "decode threads", apply: None },
        Flag { long: "frame-buffers", short: None, arg: Arg::Required, help: "external frame buffers", apply: None },
        Flag { long: "scale", short: Some('S'), arg: Arg::None, help: "scale to stream size", apply: None },
        Flag { long: "progress", short: None, arg: Arg::None, help: "show progress", apply: None },
        Flag { long: "postproc", short: None, arg: Arg::None, help: "enable VP8 post-processing", apply: None },
        Flag { long: "deblock", short: None, arg: Arg::None, help: "postproc: deblock", apply: None },
        Flag { long: "demacroblock", short: None, arg: Arg::None, help: "postproc: demacroblock", apply: None },
        Flag { long: "mfqe", short: None, arg: Arg::None, help: "postproc: MFQE", apply: None },
        Flag { long: "error-concealment", short: None, arg: Arg::None, help: "error concealment", apply: None },
        Flag { long: "output-bit-depth", short: None, arg: Arg::Required, help: "output bit depth", apply: None },
    ]
}

fn validate_codec(v: Option<String>) -> Result<(), String> {
    match v.as_deref() {
        Some("vp9") => Ok(()),
        Some(other) => Err(format!("rvp9dec: --codec={other} unsupported (only vp9)")),
        None => Err("rvp9dec: --codec requires a value".to_string()),
    }
}

fn req(v: Option<String>, flag: &str) -> Result<String, String> {
    v.ok_or_else(|| format!("rvp9dec: {flag} requires a value"))
}

const USAGE: &str = "rvp9dec [options] <input.ivf|input.webm>";

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let table = flag_table();

    // Honour --help / --version regardless of other tokens.
    if argv.iter().any(|a| a == "--help") {
        args::print_help(&table, "rvp9dec", USAGE);
        return ExitCode::SUCCESS;
    }
    if argv.iter().any(|a| a == "--version") {
        println!("rvp9dec {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    if argv.is_empty() {
        eprintln!("rvp9dec: no input. Try `rvp9dec --help`.\nUsage: {USAGE}");
        return ExitCode::FAILURE;
    }

    let mut opts = DecOpts::default();
    let positionals = match args::parse(&table, &mut opts, &argv, "rvp9dec") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    if opts.help {
        args::print_help(&table, "rvp9dec", USAGE);
        return ExitCode::SUCCESS;
    }

    match run(&opts, &positionals) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rvp9dec: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(opts: &DecOpts, positionals: &[String]) -> Result<(), String> {
    let input = match positionals {
        [one] => Path::new(one),
        [] => return Err("no input file given".to_string()),
        _ => return Err("expected exactly one input file".to_string()),
    };

    let (data, packets) =
        read_packets(input).map_err(|e| format!("reading {}: {e}", input.display()))?;

    // Resolve output format: explicit flag wins, else infer from the
    // output extension, else raw I420.
    let want_pixels = !opts.noblit && opts.md5 == Md5Mode::Off;
    let fmt = resolve_format(opts);

    // Framerate for the Y4M header (IVF carries it; default 30/1 else).
    let (fps_num, fps_den) = ivf_framerate(&data).unwrap_or((30, 1));

    // Open the output sink (file or stdout) lazily — only if we emit.
    let emit = want_pixels || opts.md5 != Md5Mode::Off;
    let mut sink: Option<BufWriter<Box<dyn Write>>> = if emit {
        Some(BufWriter::new(open_output(opts.output.as_deref())?))
    } else {
        None
    };

    let mut decoder = Decoder::new();
    let mut shown: usize = 0; // displayed frames seen
    let mut emitted: usize = 0; // displayed frames written/limited
    let mut y4m_header_written = false;

    for range in &packets {
        let pkt = range.slice(&data);
        let parsed = decoder
            .parse_packet(pkt)
            .map_err(|e| format!("parse_packet: {e}"))?;
        for frame in parsed {
            let is_shown = frame.header.show_frame
                || matches!(
                    frame.header.kind,
                    rvp9_core::bitstream::FrameKind::ShowExistingFrame { .. }
                );
            let buf = decoder
                .decode_frame(&frame)
                .map_err(|e| format!("decode_frame: {e:?}"))?;
            if !is_shown {
                continue;
            }
            // --skip / --limit operate on displayed frames.
            if shown < opts.skip {
                shown += 1;
                continue;
            }
            if let Some(lim) = opts.limit {
                if emitted >= lim {
                    shown += 1;
                    continue;
                }
            }

            if let Some(w) = sink.as_mut() {
                match opts.md5 {
                    Md5Mode::PerFrame => {
                        writeln!(w, "{}", hex16(md5_of_yv12_visible(&buf)))
                            .map_err(io_err)?;
                    }
                    Md5Mode::Indexed => {
                        writeln!(w, "{}: {}", emitted, hex16(md5_of_yv12_visible(&buf)))
                            .map_err(io_err)?;
                    }
                    Md5Mode::Off => {
                        if want_pixels {
                            write_frame(w, &buf, fmt, fps_num, fps_den, &mut y4m_header_written)?;
                        }
                    }
                }
            }

            shown += 1;
            emitted += 1;
        }
    }

    if let Some(mut w) = sink {
        w.flush().map_err(io_err)?;
    }
    if opts.summary {
        eprintln!("rvp9dec: decoded {shown} displayed frame(s), emitted {emitted}");
    }
    Ok(())
}

/// Resolve the effective output format from flags / output extension.
fn resolve_format(opts: &DecOpts) -> OutFmt {
    if opts.format != OutFmt::Unset {
        return opts.format;
    }
    match opts.output.as_deref() {
        Some(p) if p.to_ascii_lowercase().ends_with(".y4m") => OutFmt::Y4m,
        _ => OutFmt::RawI420,
    }
}

/// Write one displayed frame in the chosen pixel format.
fn write_frame<W: Write>(
    w: &mut W,
    buf: &rvp9_core::frame::Yv12Buffer,
    fmt: OutFmt,
    fps_num: u32,
    fps_den: u32,
    y4m_header_written: &mut bool,
) -> Result<(), String> {
    let (y, u, v) = visible_planes(buf);
    match fmt {
        OutFmt::Y4m => {
            if !*y4m_header_written {
                rvp9_io::y4m::write_header(w, buf.width(), buf.height(), fps_num, fps_den)
                    .map_err(io_err)?;
                *y4m_header_written = true;
            }
            rvp9_io::y4m::write_frame(w, &y, &u, &v).map_err(io_err)?;
        }
        OutFmt::RawI420 => {
            w.write_all(&y).and_then(|()| w.write_all(&u)).and_then(|()| w.write_all(&v))
                .map_err(io_err)?;
        }
        OutFmt::RawYv12 => {
            w.write_all(&y).and_then(|()| w.write_all(&v)).and_then(|()| w.write_all(&u))
                .map_err(io_err)?;
        }
        OutFmt::Unset => unreachable!("format resolved before write"),
    }
    Ok(())
}

/// Extract the visible (cropped) Y, U, V planes as tight-stride buffers.
fn visible_planes(buf: &rvp9_core::frame::Yv12Buffer) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = buf.width() as usize;
    let ch = buf.height() as usize;
    let uvw = buf.chroma_width() as usize;
    let uvh = buf.chroma_height() as usize;
    let pack = |plane: PlaneRef<'_>, w: usize, h: usize| {
        let stride = plane.stride();
        let data = plane.data();
        let mut out = vec![0u8; w * h];
        for r in 0..h {
            out[r * w..(r + 1) * w].copy_from_slice(&data[r * stride..r * stride + w]);
        }
        out
    };
    (
        pack(buf.y_plane(), cw, ch),
        pack(buf.u_plane(), uvw, uvh),
        pack(buf.v_plane(), uvw, uvh),
    )
}

/// Read the framerate `(num, den)` from an IVF file header, if `data`
/// is an IVF stream. WebM streams return `None` (we don't parse their
/// timing) and the caller defaults to 30/1.
fn ivf_framerate(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 32 || &data[0..4] != b"DKIF" {
        return None;
    }
    let num = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
    let den = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
    if num == 0 || den == 0 {
        None
    } else {
        Some((num, den))
    }
}

/// Open the output sink: a file if `path` is `Some`, else stdout.
fn open_output(path: Option<&str>) -> Result<Box<dyn Write>, String> {
    match path {
        Some(p) => {
            let f = std::fs::File::create(p).map_err(|e| format!("creating {p}: {e}"))?;
            Ok(Box::new(f))
        }
        None => Ok(Box::new(std::io::stdout())),
    }
}

fn io_err(e: std::io::Error) -> String {
    format!("io: {e}")
}
