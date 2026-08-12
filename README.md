# RVP9

An open-source Rust implementation of the VP9 video codec, following the
[libvpx](https://chromium.googlesource.com/webm/libvpx) reference encoder and
decoder. RVP9 targets production-grade real-time encoding for WebRTC and
livestreaming.

> **Status:** early development. Stage 0 (foundation) in progress. See
> [`PLAN.md`](./PLAN.md) for the multi-stage roadmap and
> [`notes/divergence-ledger.md`](./notes/divergence-ledger.md) for any
> deliberate departures from libvpx.

## Goals

- **Correctness:** 100% pass on the official VP9 conformance test vectors;
  encoder output decodes cleanly under `vpxdec`, `ffmpeg`, and our decoder.
- **Compression:** BD-rate within 0.5% of libvpx vpxenc at matched
  `--cpu-used` setting (6, 7, 8).
- **Real-time:** 720p30, 1080p30, and 1080p60 all hit budget on commodity
  x86_64 hardware.
- **Features:** 1-pass CBR/VBR, content-aware denoiser, adaptive
  quantization (3 modes), full SVC (temporal + spatial).

See [`PLAN.md`](./PLAN.md) §2 for the formal success criteria.

## Non-goals (v1)

- AV1 / VP8 codec support.
- Hardware encoding (NVENC, QuickSync, etc.).
- 2-pass encoding (incompatible with real-time).
- Slow-path RDO modes (cpu-used 0–5).
- C ABI / FFI bindings, WASM target.

## Workspace layout

```
rvp9-2/
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

```powershell
# 1. Clone with submodules (libvpx is a submodule).
git clone --recurse-submodules <repo-url>
cd rvp9-2

# 2. Build the libvpx reference (Windows / MSYS2).
./scripts/build-libvpx.ps1

# 3. Fetch test vectors + corpus (~hundreds of MB).
./scripts/fetch-vectors.ps1
./scripts/fetch-corpus.ps1

# 4. Build the Rust workspace.
cargo build --workspace --release

# 5. Sanity-check the harness against libvpx.
cargo run --release --bin conformance -- `
    --decoder build/libvpx/vpxdec.exe `
    --vectors-dir tests/conformance/vectors `
    --limit 5

cargo run --release --bin realtime -- `
    --encoder build/libvpx/vpxenc.exe `
    --input tests/corpus/parkrun_720p.y4m `
    --cpu-used 7 --budget-ms 33.333
```

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

The fixed 22-row training matrix covers cpu-used 6–8, 720p and 1080p, VBR,
CBR with cyclic-refresh AQ, denoising, temporal SVC, and bitrate extremes.
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
