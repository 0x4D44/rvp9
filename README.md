# RVP9

An open-source Rust implementation of the VP9 video codec, following the
[libvpx](https://chromium.googlesource.com/webm/libvpx) reference encoder and
decoder. RVP9 targets production-grade real-time encoding for WebRTC and
livestreaming.

> **Status:** `v0.2.0-alpha.3` is an 8-bit real-time preview. It is ready for
> evaluation, not a production-parity release. See [`PLAN.md`](./PLAN.md) for
> the remaining work and [`notes/divergence-ledger.md`](./notes/divergence-ledger.md)
> for deliberate departures from libvpx.

> **Platform coverage:** Testing and release qualification to date have been
> performed only on macOS on Apple Silicon (AArch64). RVP9 has not yet been
> tested or qualified on x86, including x86_64, or on other platforms.

## Alpha scope

The alpha encoder supports one-pass real-time VP9 at CPU-used 6, 7, and 8,
with VBR or CBR rate control, cyclic-refresh AQ mode 3, denoising, two- or
three-layer temporal SVC, and two- or three-layer spatial SVC. The qualified
production-quality surface is single-stream CBR, denoiser CBR, and spatial
CBR. VBR and temporal SVC are experimental in v0.2.0. The qualified targets
are 720p30, 1080p30, and 1080p60. It emits 8-bit profile-0 IVF streams.

The alpha decoder passes all 4,816 displayed frames across all 307 pinned
official vectors with native and forced-scalar dispatch and with one- and
two-thread decoding.

Every stream in the 32-row encoder qualification matrix decodes byte-exactly
under `rvp9dec`, pinned libvpx `vpxdec`, and ffmpeg. Run the same gates with:

```bash
./scripts/qualify-conformance.sh
./scripts/qualify-alpha-encoder.sh
./scripts/qualify-reference-decoder.sh
```

The reverse 24-row gate uses pinned `vpxenc` streams across CPU-used 6--8,
CBR/VBR, AQ3, and denoising, then requires `rvp9dec` and pinned `vpxdec` to
produce byte-identical decoded output.

Every configuration retains the full correctness, cross-decoder,
relative-speed, and latency gates. Single-stream and denoiser CBR keep the
`+0.5%` aggregate quality gate and `+2.0%` important-clip guard. The explicit
experimental ceilings are `+15.0%` for single-stream/AQ/denoiser VBR, `+5.0%`
for temporal CBR, `+35.0%` for temporal VBR, and `+25.0%` for spatial VBR.
Spatial CBR remains on the strict production-quality path. This alpha does not
claim final compression parity across the complete v1 matrix.

## Project goals

- **Correctness:** 100% pass on the official VP9 conformance test vectors;
  encoder output decodes cleanly under `vpxdec`, `ffmpeg`, and our decoder.
- **Compression:** BD-rate within 0.5% of libvpx vpxenc at matched
  `--cpu-used` setting (6, 7, 8) on the production-quality CBR surface.
  Experimental modes use the explicit ceilings above and are excluded from
  the compression-parity claim.
- **Real-time:** 720p30, 1080p30, and 1080p60 all hit budget on commodity
  x86_64 hardware.
- **Features:** 1-pass CBR/VBR, content-aware denoiser, cyclic-refresh AQ,
  and temporal/spatial SVC; VBR and temporal SVC are experimental.

See [`PLAN.md`](./PLAN.md) §2 for the formal success criteria.

## Non-goals (v1)

- AV1 / VP8 codec support.
- Hardware encoding (NVENC, QuickSync, etc.).
- 2-pass encoding (incompatible with real-time).
- Slow-path RDO modes (cpu-used 0–5).
- C ABI / FFI bindings, WASM target.

## Workspace layout

```
rvp9/
├── PLAN.md                  # multi-stage plan
├── LICENSE                  # BSD 3-Clause licence for RVP9
├── PATENTS                  # WebM patent grants and their limits
├── ATTRIBUTION.md           # libvpx licence and acknowledgements
├── crates/
│   ├── rvp9-core/           # shared primitives (bitstream, entropy, transforms, ...)
│   ├── rvp9-decoder/        # spec-conformant decoder (Stage 3)
│   ├── rvp9-encoder/        # libvpx-mirrored real-time encoder (Stage 4)
│   ├── rvp9-cli/            # rvp9enc, rvp9dec
│   ├── rvp9-bench/          # ms/frame and BD-rate benchmarks
│   └── rvp9-harness/        # conformance + differential test runners
├── third_party/
│   └── libvpx/              # git submodule, pinned to v1.14.1
├── tests/
│   ├── conformance/         # official VP9 test vectors (gitignored)
│   └── corpus/              # encoder test clips (gitignored)
├── scripts/                 # build-libvpx, fetch-vectors, fetch-corpus
└── notes/                   # design docs and the divergence ledger
```

## Prerequisites

- Rust 1.95 or later (workspace pins `stable` via `rust-toolchain.toml`).
- For building the libvpx reference on Windows: MSYS2 with `gcc`, `make`,
  `yasm`, `perl`.

## Quick start

```bash
# 1. Clone with submodules (libvpx is a submodule).
git clone --recurse-submodules https://github.com/0x4D44/rvp9
cd rvp9

# 2. Build the pinned libvpx reference.
./scripts/build-libvpx.sh

# 3. Fetch the official vectors and encoder corpus (large downloads).
./scripts/fetch-vectors.sh
./scripts/fetch-corpus.sh

# 4. Build the Rust workspace.
cargo build --workspace --release --locked

# 5. Run the current correctness gates.
./scripts/qualify-conformance.sh
./scripts/qualify-alpha-encoder.sh
./scripts/qualify-reference-decoder.sh
```

On Windows, run the `.sh` scripts in Git Bash, MSYS2, or WSL. A PowerShell
corpus downloader is also available as `scripts/fetch-corpus.ps1`.

### Production encoder build

The parity-qualified encoder uses host-trained LLVM profile-guided
optimization in addition to the workspace's native-CPU release settings.
Build it on the host that will run it; profiles are not portable across source
revisions, Rust/LLVM versions, or machines.

```bash
rustup component add llvm-tools-preview
./scripts/build-encoder-pgo.sh
target/encoder-pgo/optimized/release/rvp9enc --version
```

The fixed 28-row training matrix covers cpu-used 6–8, 720p and 1080p, VBR,
CBR with cyclic-refresh AQ, denoising, temporal and spatial SVC, and bitrate
extremes.
It requires the canonical corpus from `scripts/fetch-corpus.sh` plus `ffmpeg`.
The build records its commit, toolchain, host, and exact matrix under
`target/encoder-pgo/runs/<run-id>/manifest.txt`. Plain `cargo build --release`
remains useful for development, but it is not the production speed artifact.

### Encoder speed acceptance

Use `encode-headtohead` for relative speed evidence. It runs warmed pairs in
both arm orders, reverses clip order, reports per-clip medians and corpus
geomeans, and requires both independent-median and paired-ratio gates to pass.
It also removes inherited `RVP9_*` overrides, checks every IVF frame count and
output hash, and records the executable, input, revision, host, command, and
load evidence in the run directory.

The temporal-SVC anchor uses the pinned adapter documented in
`tools/libvpx_temporal_svc_reference/README.md`. Prepare equal-rate clip
prefixes from the fetched corpus, then use a new empty run directory:

```bash
speed_corpus="$(mktemp -d)"
ffmpeg -hide_banner -loglevel error \
  -i tests/corpus/in_to_tree_420_720p50.y4m \
  -vf fps=30 -frames:v 60 "$speed_corpus/in-tree.y4m"
ffmpeg -hide_banner -loglevel error \
  -i tests/corpus/ducks_take_off_420_720p50.y4m \
  -vf fps=30 -frames:v 60 "$speed_corpus/ducks.y4m"
ffmpeg -hide_banner -loglevel error \
  -i tests/corpus/old_town_cross_420_720p50.y4m \
  -vf fps=30 -frames:v 60 "$speed_corpus/oldtown.y4m"

run_dir="$(mktemp -d)"
cargo run --release -p rvp9-bench --bin encode-headtohead -- \
  --anchor-encoder build/libvpx/vpx_temporal_svc_reference \
  --test-encoder target/encoder-pgo/optimized/release/rvp9enc \
  --anchor-kind temporal-svc --test-kind vpxenc \
  --anchor-revision "$(git -C third_party/libvpx rev-parse HEAD)" \
  --test-revision "$(git rev-parse HEAD)" \
  --clip "in-tree=$speed_corpus/in-tree.y4m" \
  --clip "ducks=$speed_corpus/ducks.y4m" \
  --clip "oldtown=$speed_corpus/oldtown.y4m" \
  --work-dir "$run_dir" \
  --cpu-used 8 --target-bitrate 1500 --end-usage vbr \
  --fps 30/1 --threads 1 --tile-columns 0 --temporal-layers 3 \
  --aq-mode 0 --noise-sensitivity 0 --limit 60 \
  --keyframe-interval 60 --warmups 2 --pairs 15 --rotations 2 \
  --max-ratio 1.05 --max-load1 0.75
```

Treat `--max-load1` as a host qualification, not a tuning knob. Choose and
record the idle-host threshold before the run; never raise it to admit a noisy
result. `summary.csv`, `raw.csv`, and `manifest.txt` are the acceptance record.

## Licence

RVP9 is available under the [BSD 3-Clause licence](./LICENSE). It follows and
partly transcribes libvpx, whose copyright notice is preserved in
[ATTRIBUTION.md](./ATTRIBUTION.md).

Google offers VP9/WebM under separate no-charge patent terms. See
[PATENTS](./PATENTS) for the grants and their limits. Third-party patent rights
may exist. RVP9 provides no patent-clearance warranty or indemnity.
