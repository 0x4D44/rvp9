# rvp9enc / rvp9dec CLI design

Status: proposed (2026-05-22). Wires the two stub binaries in `rvp9-cli`
into real CLI front-ends over the existing encoder/decoder libraries,
mirroring `vpxenc` / `vpxdec` flags where practical.

## Decisions (signed off by Arthur)

- **I/O sharing**: extract `container.rs` (IVF/WebM) + `y4m.rs` out of
  `rvp9-harness` into a new shared crate **`rvp9-io`**. `rvp9-harness`
  and `rvp9-cli` both depend on it. (Alternative considered: harness as
  a CLI dep — rejected, makes a test/tooling crate a runtime dep of
  shipped binaries.)
- **Arg parsing**: hand-rolled `vpxenc`-style `--flag=value` parser, no
  `clap`. Keeps drop-in parity with existing benchmark scripts.
- **Flag scope**: full `vpxenc`/`vpxdec` surface catalogued in `--help`.
- **Unsupported flags ERROR, support is emergent (Arthur's refinement)**:
  there is NO separate "supported list". Each flag is a table entry
  pointing at a handler. Unimplemented features point at a stub handler
  that returns `Err("--<flag>: not supported yet")`. Wiring a feature =
  swapping the stub for a real handler; the flag then auto-enables and
  `--help`'s supported/not-yet tag is *derived from the same table*.
  One source of truth, no drift, no silent no-ops.

## New crate: rvp9-io

Move (not copy) from `rvp9-harness/src`:
- `container.rs` — `read_packets`, `read_ivf_packets`, `write_ivf`,
  `read_webm_packets`, `IvfHeader`, `PacketRange`.
- `y4m.rs` — `Y4mHeader`, `parse_header`, `read_cropped`, `YuvFrameBuf`.

Add (new):
- `y4m::write_header` + `y4m::write_frame` — Y4M *writer* (only a reader
  exists today). Needed by `rvp9dec --y4m`.

`rvp9-harness/src/lib.rs`: replace `mod container; mod y4m;` with
`pub use rvp9_io::{container, y4m};` (or re-export shims) so existing
harness binaries (`conformance`, `diff_bitstream`, `e1_block_stats`,
`real_world` tests) compile unchanged.

RISK: many `agent-*` worktrees are active. Moving files harness depends
on can collide with in-flight work. Mitigation: surgical move + a single
re-export edit in harness lib.rs; check `git status` immediately before.

## rvp9enc

Pipeline: parse args -> open input (Y4M, or raw YUV with explicit
`-w/-h`) -> `StreamEncoder::new(EncoderConfig)` -> per frame
`encode_frame(YuvFrameRef)` -> mux each `EncodedFrame.data` as one IVF
packet via `rvp9_io::container::write_ivf`.

### Wired now (map to EncoderConfig)
| vpxenc flag                  | maps to                  |
|------------------------------|--------------------------|
| `-w/--width`, `-h/--height`  | `width`, `height`        |
| `--fps=<n>/<d>`              | `framerate`              |
| `--target-bitrate=<kbps>`    | `target_bitrate_bps` (*1000) |
| `--kf-max-dist=<n>`          | `gop_length`             |
| `--aq-mode=<0|3>`            | `aq_mode` (0=Off, 3=CyclicRefresh) |
| `--end-usage=cbr`            | required value (vbr/cq/q -> err) |
| `--passes=1`                 | required value (2 -> err) |
| `--codec=vp9`                | validated                |
| `-o <file>` / positional in  | output / input paths     |
| `--limit=<n>`, `--skip=<n>`  | CLI-level frame gating   |
| `--ivf`                      | output container (only one supported) |

### Recognized -> error ("not supported yet")
`--passes=2` + two-pass stats, `--svc`/spatial+temporal layers,
`--denoise`/`--noise-sensitivity`, `--lossless`, `--cq-level`,
`--bit-depth=10|12`, `--profile!=0`, `--webm`, `--tile-columns>0`,
threading knobs not yet exposed, etc. (Auto-enable as the encoder grows.)

## rvp9dec

Pipeline: parse args -> read packets (IVF/WebM via
`rvp9_io::container`) -> `Decoder::parse_packet` -> `.frames()` ->
`decode_frame` -> `Yv12Buffer` -> write planes.

### Wired now
| vpxdec flag                  | behaviour                |
|------------------------------|--------------------------|
| positional input / `-o`      | input / output paths     |
| `--y4m`                      | Y4M output (new writer)  |
| `--i420`/`--yv12`/`--rawvideo` | raw plane dump         |
| `--md5`, `--framemd5`        | MD5 (reuse harness md5)  |
| `--limit`, `--skip`          | frame gating             |
| `--codec=vp9`, `--summary`, `--noblit` | validate / stats / no-output |

### Recognized -> error
`--scale`, postproc (`--pp-*`), `--svc-decode`, output bit-depth
conversion, `--threads` (if not yet wired), etc.

## Test plan (oracle = round-trip + libvpx)
1. `rvp9enc` Y4M -> IVF, then `rvp9dec` IVF -> Y4M; assert decoder
   output matches encoder's internal recon (round-trip identity for the
   frames the encoder can faithfully reproduce).
2. `rvp9dec` of a libvpx-produced IVF == `vpxdec` `--md5` (cross-tool
   parity; harness conformance already proves the decoder core).
3. Unit tests on the arg parser: every "not supported yet" flag exits
   non-zero with the right message; every wired flag sets the expected
   config field.
4. `--help` golden test: lists the full surface, tags derived from the
   handler table (no hand-maintained list).
