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

#[path = "../path_guard.rs"]
mod path_guard;

use rvp9_cli::args::{self, Arg, Flag};
use rvp9_core::bitstream::ChromaFormat;
use rvp9_core::frame::PlaneRef;
use rvp9_decoder::Decoder;
use rvp9_io::container::PacketReader;
use rvp9_io::md5::{hex16, md5_of_yv12_visible};
use std::io::{BufWriter, Write};
use std::num::NonZeroUsize;
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
    /// Max decode threads. `None` ⇒ flag absent ⇒ auto-detect via
    /// `Decoder::new()`. `Some(n)` ⇒ explicit `Decoder::with_threads(n)`.
    /// libvpx parity: `--threads 0` is normalised to `Some(NonZeroUsize::new(1))`
    /// at parse time (1 thread, no auto-detect).
    threads: Option<NonZeroUsize>,
    /// Print decode progress to stderr after each displayed frame.
    progress: bool,
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
            threads: None,
            progress: false,
        }
    }
}

/// The flag table — the single source of truth for the CLI surface.
/// `apply: Some` ⇒ wired; `apply: None` ⇒ recognised-but-not-yet.
fn flag_table() -> Vec<Flag<DecOpts>> {
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
            apply: Some(|_, v| validate_codec(v)),
        },
        Flag {
            long: "output",
            short: Some('o'),
            arg: Arg::Required,
            help: "output file (else stdout)",
            apply: Some(|o, v| {
                o.output = Some(req(v, "--output")?);
                Ok(())
            }),
        },
        Flag {
            long: "y4m",
            short: None,
            arg: Arg::None,
            help: "output YUV4MPEG2",
            apply: Some(|o, _| {
                o.format = OutFmt::Y4m;
                Ok(())
            }),
        },
        Flag {
            long: "i420",
            short: None,
            arg: Arg::None,
            help: "output raw I420 (Y,U,V)",
            apply: Some(|o, _| {
                o.format = OutFmt::RawI420;
                Ok(())
            }),
        },
        Flag {
            long: "yv12",
            short: None,
            arg: Arg::None,
            help: "output raw YV12 (Y,V,U)",
            apply: Some(|o, _| {
                o.format = OutFmt::RawYv12;
                Ok(())
            }),
        },
        Flag {
            long: "rawvideo",
            short: None,
            arg: Arg::None,
            help: "output raw I420, no container",
            apply: Some(|o, _| {
                o.format = OutFmt::RawI420;
                Ok(())
            }),
        },
        Flag {
            long: "md5",
            short: None,
            arg: Arg::None,
            help: "print per-frame MD5",
            apply: Some(|o, _| {
                o.md5 = Md5Mode::PerFrame;
                Ok(())
            }),
        },
        Flag {
            long: "framemd5",
            short: None,
            arg: Arg::None,
            help: "print indexed per-frame MD5",
            apply: Some(|o, _| {
                o.md5 = Md5Mode::Indexed;
                Ok(())
            }),
        },
        Flag {
            long: "limit",
            short: None,
            arg: Arg::Required,
            help: "stop after N displayed frames",
            apply: Some(|o, v| {
                o.limit = Some(args::parse_value(v, "rvp9dec", "--limit")?);
                Ok(())
            }),
        },
        Flag {
            long: "skip",
            short: None,
            arg: Arg::Required,
            help: "skip N displayed frames",
            apply: Some(|o, v| {
                o.skip = args::parse_value(v, "rvp9dec", "--skip")?;
                Ok(())
            }),
        },
        Flag {
            long: "summary",
            short: None,
            arg: Arg::None,
            help: "print a decode summary",
            apply: Some(|o, _| {
                o.summary = true;
                Ok(())
            }),
        },
        Flag {
            long: "noblit",
            short: None,
            arg: Arg::None,
            help: "decode only, no pixel output",
            apply: Some(|o, _| {
                o.noblit = true;
                Ok(())
            }),
        },
        Flag {
            long: "threads",
            short: Some('t'),
            arg: Arg::Required,
            help: "max decode threads (0 = 1 thread, omit for auto-detect)",
            apply: Some(|o, v| {
                let n: usize = args::parse_value(v, "rvp9dec", "--threads")?;
                // libvpx parity: --threads 0 means "1 thread" (no extra workers),
                // NOT auto-detect. Users wanting auto-detect omit the flag.
                // `NonZeroUsize::new(0)` is the only way `new` returns None
                // for a `usize`, so fall back to MIN (=1) handles exactly the
                // `--threads 0` case.
                let n = NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN);
                o.threads = Some(n);
                Ok(())
            }),
        },
        // -- Progress ---------------------------------------------------
        Flag {
            long: "progress",
            short: None,
            arg: Arg::None,
            help: "show progress",
            apply: Some(|o, _| {
                o.progress = true;
                Ok(())
            }),
        },
        // -- Accepted with warning (no-op) ------------------------------
        Flag {
            long: "scale",
            short: Some('S'),
            arg: Arg::None,
            help: "scale to stream size",
            apply: Some(|_, _| {
                eprintln!(
                    "rvp9dec: --scale accepted but display-size scaling not yet implemented (output at coded size)"
                );
                Ok(())
            }),
        },
        Flag {
            long: "output-bit-depth",
            short: None,
            arg: Arg::Required,
            help: "output bit depth",
            apply: Some(|_, v| {
                let val =
                    v.ok_or_else(|| "rvp9dec: --output-bit-depth requires a value".to_string())?;
                if val != "8" {
                    eprintln!(
                        "rvp9dec: --output-bit-depth={val}: only 8-bit output supported (Profile 0 decoder)"
                    );
                }
                Ok(())
            }),
        },
        Flag {
            long: "error-concealment",
            short: None,
            arg: Arg::None,
            help: "error concealment",
            apply: Some(|_, _| {
                eprintln!(
                    "rvp9dec: --error-concealment accepted (not yet implemented, errors will propagate)"
                );
                Ok(())
            }),
        },
        Flag {
            long: "frame-buffers",
            short: None,
            arg: Arg::Required,
            help: "external frame buffers",
            apply: Some(|_, v| {
                let val =
                    v.ok_or_else(|| "rvp9dec: --frame-buffers requires a value".to_string())?;
                eprintln!("rvp9dec: --frame-buffers={val} accepted (internal buffer pool used)");
                Ok(())
            }),
        },
        // -- VP8-only post-processing (silent no-op) --------------------
        Flag {
            long: "postproc",
            short: None,
            arg: Arg::None,
            help: "enable VP8 post-processing",
            apply: Some(|_, _| Ok(())),
        },
        Flag {
            long: "deblock",
            short: None,
            arg: Arg::None,
            help: "postproc: deblock",
            apply: Some(|_, _| Ok(())),
        },
        Flag {
            long: "demacroblock",
            short: None,
            arg: Arg::None,
            help: "postproc: demacroblock",
            apply: Some(|_, _| Ok(())),
        },
        Flag {
            long: "mfqe",
            short: None,
            arg: Arg::None,
            help: "postproc: MFQE",
            apply: Some(|_, _| Ok(())),
        },
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

    let emit = !opts.noblit || opts.md5 != Md5Mode::Off;
    let mut paths = vec![("input", input)];
    if emit && let Some(output) = opts.output.as_deref() {
        paths.push(("output", Path::new(output)));
    }
    let mut path_guard = path_guard::PathGuard::new(&paths)?;
    let mut packets =
        PacketReader::open(input).map_err(|e| format!("reading {}: {e}", input.display()))?;

    // Resolve output format: explicit flag wins, else infer from the
    // output extension, else raw I420.
    let want_pixels = !opts.noblit && opts.md5 == Md5Mode::Off;
    let fmt = resolve_format(opts);

    // Framerate for the Y4M header (IVF carries it; default 30/1 else).
    let (fps_num, fps_den) = packets.framerate().unwrap_or((30, 1));

    // Do not open or truncate the output until a decoded frame is ready
    // to emit. Container, packet, decode, and worker-startup failures must
    // leave an existing destination untouched.
    let mut sink: Option<BufWriter<Box<dyn Write>>> = None;

    let mut decoder = match opts.threads {
        Some(n) => Decoder::try_with_threads(n),
        None => Decoder::try_new(),
    }
    .map_err(|error| format!("creating decoder worker threads: {error}"))?;
    let mut shown: usize = 0; // displayed frames seen
    let mut emitted: usize = 0; // displayed frames written/limited
    let mut frames_in: usize = 0; // decoder calls (= vpxdec's frame_in)
    // Accumulated microseconds spent strictly inside `decode_frame`.
    // Matches vpxdec's `dx_time` measurement scope (the decode call only;
    // not packet parsing, not pixel output). u128 has effectively unlimited
    // headroom — a u64 us counter holds ~584,000 years.
    let mut dx_time_us: u128 = 0;
    let mut y4m_format = None;

    'packets: while opts.limit != Some(0) {
        let Some(pkt) = packets
            .next_packet()
            .map_err(|e| format!("reading {}: {e}", input.display()))?
        else {
            break;
        };
        let parsed = decoder
            .parse_packet(&pkt)
            .map_err(|e| format!("parse_packet: {e}"))?;
        frames_in += 1;
        // Libvpx decodes every component in a superframe packet, then its
        // public decoder interface exposes only the final shown frame from
        // that decode call (`vp9_dx_iface.c:377-415`). Spatial SVC packets
        // rely on this: their shown components are ordered low to high, and
        // the default output is the final/top layer.
        let output_frame_index = parsed.output_frame_index();
        let mut displayed = None;
        for (frame_index, frame) in parsed.into_iter().enumerate() {
            let t0 = std::time::Instant::now();
            let decode_result = decoder.decode_frame(&frame);
            dx_time_us += t0.elapsed().as_micros();
            let buf = decode_result.map_err(|e| format!("decode_frame: {e:?}"))?;
            if Some(frame_index) == output_frame_index {
                displayed = Some(buf);
            }
        }
        let Some(buf) = displayed else {
            continue;
        };
        // --skip / --limit operate on displayed packet outputs.
        if shown < opts.skip {
            shown += 1;
            continue;
        }
        if let Some(lim) = opts.limit
            && emitted >= lim
        {
            shown += 1;
            continue;
        }

        if want_pixels {
            let chroma = buf.chroma();
            match fmt {
                OutFmt::RawI420 if chroma != ChromaFormat::Yuv420 => {
                    return Err(format!(
                        "I420/--i420/--rawvideo output requires decoded 4:2:0 chroma; \
                         got {chroma:?}"
                    ));
                }
                OutFmt::RawYv12 if chroma != ChromaFormat::Yuv420 => {
                    return Err(format!(
                        "YV12/--yv12 output requires decoded 4:2:0 chroma; got {chroma:?}"
                    ));
                }
                OutFmt::Y4m if chroma == ChromaFormat::Yuv440 && y4m_format.is_none() => {
                    return Err(
                        "--y4m output for 4:4:0 (Yuv440) is not representable in YUV4MPEG2; \
                         use --noblit, --md5, or --framemd5"
                            .to_string(),
                    );
                }
                _ => {}
            }
        }

        if emit && sink.is_none() {
            sink = Some(BufWriter::new(open_output(
                opts.output.as_deref(),
                &mut path_guard,
            )?));
        }
        if let Some(w) = sink.as_mut() {
            match opts.md5 {
                Md5Mode::PerFrame => {
                    writeln!(w, "{}", hex16(md5_of_yv12_visible(&buf))).map_err(io_err)?;
                }
                Md5Mode::Indexed => {
                    writeln!(w, "{}: {}", emitted, hex16(md5_of_yv12_visible(&buf)))
                        .map_err(io_err)?;
                }
                Md5Mode::Off => {
                    if want_pixels {
                        write_frame(w, &buf, fmt, fps_num, fps_den, &mut y4m_format, emitted + 1)?;
                    }
                }
            }
        }

        shown += 1;
        emitted += 1;
        if opts.progress {
            eprint!("\r[rvp9dec] {emitted} frames decoded");
        }
        if opts.limit == Some(emitted) {
            break 'packets;
        }
    }

    if opts.progress {
        // End the progress line cleanly.
        eprintln!();
    }
    if let Some(mut w) = sink {
        w.flush().map_err(io_err)?;
    }
    if opts.summary {
        // Match vpxdec's `show_progress` format (printed to stderr):
        //   "<frame_in> decoded frames/<frame_out> showed frames in <us> us (<fps> fps)"
        // — frame_in counts compressed-data decode calls (one per transport
        //   packet, including a packet containing several superframe
        //   components), while frame_out counts displayed frames. fps is
        //   derived from displayed frames over decode-call wall time.
        // Guard against div-by-zero on empty inputs.
        let fps = if dx_time_us == 0 {
            0.0
        } else {
            (shown as f64) * 1_000_000.0 / (dx_time_us as f64)
        };
        eprintln!(
            "{frames_in} decoded frames/{shown} showed frames in {dx_time_us} us ({fps:.2} fps)"
        );
        // Note: `emitted` (displayed frames actually written, after --skip /
        // --limit) is intentionally omitted to stay byte-compatible with
        // vpxdec's summary line. Callers wanting that figure can count
        // output frames externally.
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
    y4m_format: &mut Option<(u32, u32, ChromaFormat)>,
    output_frame: usize,
) -> Result<(), String> {
    let current_format = (buf.width(), buf.height(), buf.chroma());
    if fmt == OutFmt::Y4m
        && let Some(first_format) = *y4m_format
        && first_format != current_format
    {
        return Err(format!(
            "Y4M output format changed on output frame {output_frame}: first frame format {}x{} {:?}, current frame format {}x{} {:?}",
            first_format.0,
            first_format.1,
            first_format.2,
            current_format.0,
            current_format.1,
            current_format.2,
        ));
    }

    let (y, u, v) = visible_planes(buf);
    match fmt {
        OutFmt::Y4m => {
            if y4m_format.is_none() {
                rvp9_io::y4m::write_header(
                    w,
                    current_format.0,
                    current_format.1,
                    fps_num,
                    fps_den,
                    current_format.2,
                )
                .map_err(io_err)?;
                *y4m_format = Some(current_format);
            }
            rvp9_io::y4m::write_frame(w, &y, &u, &v).map_err(io_err)?;
        }
        OutFmt::RawI420 => {
            w.write_all(&y)
                .and_then(|()| w.write_all(&u))
                .and_then(|()| w.write_all(&v))
                .map_err(io_err)?;
        }
        OutFmt::RawYv12 => {
            w.write_all(&y)
                .and_then(|()| w.write_all(&v))
                .and_then(|()| w.write_all(&u))
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

/// Open the output sink: a file if `path` is `Some`, else stdout.
fn open_output(
    path: Option<&str>,
    path_guard: &mut path_guard::PathGuard,
) -> Result<Box<dyn Write>, String> {
    match path {
        Some(_) => {
            let f = path_guard.create("output")?;
            Ok(Box::new(f))
        }
        None => Ok(Box::new(std::io::stdout())),
    }
}

fn io_err(e: std::io::Error) -> String {
    format!("io: {e}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::frame::{BorderUsage, ByteAlignment, Yv12Buffer};
    use std::io;

    fn buffer(width: u32, height: u32, chroma: ChromaFormat) -> Yv12Buffer {
        Yv12Buffer::alloc(
            width,
            height,
            chroma,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("allocate test frame")
    }

    fn assert_format_change_rejected(
        first_format: (u32, u32, ChromaFormat),
        changed_format: (u32, u32, ChromaFormat),
        output_frame: usize,
    ) {
        let first = buffer(first_format.0, first_format.1, first_format.2);
        let changed = buffer(changed_format.0, changed_format.1, changed_format.2);
        let mut output = Vec::new();
        let mut y4m_format = None;
        write_frame(&mut output, &first, OutFmt::Y4m, 30, 1, &mut y4m_format, 1)
            .expect("write first Y4M frame");
        let prefix = output.clone();

        let result = write_frame(
            &mut output,
            &changed,
            OutFmt::Y4m,
            30,
            1,
            &mut y4m_format,
            output_frame,
        );
        assert!(
            result.is_err(),
            "Y4M format change on output frame {output_frame} must be rejected"
        );
        assert_eq!(output, prefix, "changed frame must write no bytes");
        let error = result.expect_err("format change error");
        let expected = [
            format!("output frame {output_frame}"),
            format!("{}x{} {:?}", first_format.0, first_format.1, first_format.2),
            format!(
                "{}x{} {:?}",
                changed_format.0, changed_format.1, changed_format.2
            ),
        ];
        for expected in expected {
            assert!(error.contains(&expected), "{error:?} lacks {expected:?}");
        }
    }

    #[test]
    fn y4m_rejects_width_change_before_writing_frame() {
        assert_format_change_rejected(
            (64, 64, ChromaFormat::Yuv420),
            (128, 64, ChromaFormat::Yuv420),
            2,
        );
    }

    #[test]
    fn y4m_rejects_height_change_before_writing_frame() {
        assert_format_change_rejected(
            (64, 64, ChromaFormat::Yuv420),
            (64, 128, ChromaFormat::Yuv420),
            2,
        );
    }

    #[test]
    fn y4m_rejects_chroma_changes_before_writing_frame() {
        for chroma in [
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
            ChromaFormat::Yuv440,
        ] {
            assert_format_change_rejected((64, 64, ChromaFormat::Yuv420), (64, 64, chroma), 2);
        }
    }

    #[test]
    fn y4m_writes_multiple_frames_with_the_same_format() {
        let frame = buffer(64, 64, ChromaFormat::Yuv420);
        let mut output = Vec::new();
        let mut y4m_format = None;
        for output_frame in 1..=2 {
            write_frame(
                &mut output,
                &frame,
                OutFmt::Y4m,
                30,
                1,
                &mut y4m_format,
                output_frame,
            )
            .expect("write same-format Y4M frame");
        }

        assert_eq!(
            output
                .windows(b"YUV4MPEG2".len())
                .filter(|window| *window == b"YUV4MPEG2")
                .count(),
            1,
            "one stream header"
        );
        assert_eq!(
            output
                .windows(b"FRAME\n".len())
                .filter(|window| *window == b"FRAME\n")
                .count(),
            2,
            "two frame tags"
        );
    }

    #[test]
    fn y4m_format_state_is_unset_when_header_write_fails() {
        struct FailingWriter;

        impl Write for FailingWriter {
            fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("intentional header write failure"))
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let frame = buffer(64, 64, ChromaFormat::Yuv420);
        let mut y4m_format = None;
        let error = write_frame(
            &mut FailingWriter,
            &frame,
            OutFmt::Y4m,
            30,
            1,
            &mut y4m_format,
            1,
        )
        .expect_err("failed Y4M header write");

        assert!(error.contains("intentional header write failure"));
        assert_eq!(y4m_format, None, "failed header must not latch a format");
    }
}
