# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rvp9-2` is a Rust port of the [libvpx](https://chromium.googlesource.com/webm/libvpx)
**VP9 encoder and decoder**, targeting production-grade **real-time** encoding
(WebRTC / livestreaming) at `--cpu-used` 6–8. libvpx is both the reference
implementation (for differential testing / byte-parity work) and the source of
probability tables, scan orders, and constants transcribed verbatim.

Release gates (PLAN.md §2): 100% pass on official VP9 conformance vectors;
BD-rate within 0.5% of libvpx `vpxenc` at matched `--cpu-used`; 720p30 / 1080p30 /
1080p60 real-time on commodity x86_64.

**Beware stale module docs.** Many `lib.rs` / module headers still say "Stage 3
skeleton" or "what does NOT work yet". The project is far past that — see recent
git commits, `PLAN.md`, and `wrk_journals/` for the true state (encoder is
byte-exact with libvpx on most features; the open work is closing an *emergent*
BD-rate gap). Trust the code, commits, and journals over the older doc prose.

## Build / test / run

Toolchain is pinned to **Rust 1.95.0** (`rust-toolchain.toml`), edition 2024.
This is a **virtual workspace** (no root package).

```bash
cargo build --workspace --release      # release: lto=fat, codegen-units=1
cargo build --workspace                # dev: opt-level=1 (video pipeline needs it)
cargo test  --workspace                # all tests
cargo test -p rvp9-decoder            # one crate
cargo test -p rvp9-encoder --test cross_decoder_gate   # one integration test file
cargo test -p rvp9-core quant::        # tests matching a path filter
cargo clippy --workspace --all-targets # lints (see lint policy below)
cargo fmt --all
```

> **`target-cpu=native` is set in `.cargo/config.toml`** — release binaries use the
> full host instruction set (BMI2/FMA/AVX-512 where present) and are **NOT portable**
> across x86_64 hosts. Override with `RUSTFLAGS="" cargo build --release` for a
> portable build.

### Reference binaries, vectors, corpus

- libvpx is a git submodule pinned to **v1.14.1** at `third_party/libvpx/`
  (`git clone --recurse-submodules`). Build the reference with
  `scripts/build-libvpx.sh` (Windows: MSYS2 with `gcc make yasm perl`).
- Reference binaries land at `build/libvpx/vpxenc.exe` and `build/libvpx/vpxdec.exe`.
- Conformance vectors and the test corpus are **gitignored** (hundreds of MB);
  fetch via `scripts/fetch-vectors.sh` and `scripts/fetch-corpus.{sh,ps1}`.

### Harness & benchmarks (the real validation surface)

```bash
# Decoder conformance: decode every vector, MD5-compare against sibling .md5
cargo run --release -p rvp9-harness --bin conformance -- \
    --vectors-dir tests/conformance/vectors [--limit N] [--strict] [--verbose]

# Other harness bins: diff-bitstream, decode-coefs-oracle-diff, f02_block_oracle,
#   extract-divergences, dump_frame_diff, ...  (crates/rvp9-harness/src/bin/)

# Benches (crates/rvp9-bench/src/bin/): realtime (latency budget), bdrate (BD-rate
#   vs libvpx), decode-throughput / decode-headtohead (decode speed ratchet),
#   dsp-microbench, soak.
cargo run --release -p rvp9-bench --bin bdrate -- ...
```

`bench/results/decode-throughput.csv` and `decode-headtohead.{md,csv}` are the only
checked-in bench artifacts (a tracked perf ratchet); everything else under
`bench/results/` is gitignored.

### Fuzzing

`crates/rvp9-fuzz` is **deliberately excluded** from the workspace (cargo-fuzz uses
`#![no_main]`, which breaks `cargo build --workspace` with LNK1561 on Windows).
Run it separately: `cd crates/rvp9-fuzz && cargo +nightly fuzz run fuzz_parse_packet`.

## Architecture

Seven workspace crates (data flows top → bottom):

| Crate | Role |
|---|---|
| `rvp9-core` | **Shared, bit-exact primitives** used by both encoder and decoder. The bulk of the code (~145 files). |
| `rvp9-io` | Container (IVF/WebM) + Y4M I/O + MD5 helpers. |
| `rvp9-decoder` | Spec-conformant VP9 decoder. Thin: drives `rvp9-core` per tile/superblock. |
| `rvp9-encoder` | Real-time encoder (libvpx-mirrored cpu-used 6–8). Owns mode/partition/RC/AQ. |
| `rvp9-cli` | `rvp9enc` / `rvp9dec` binaries (mirror `vpxenc` / `vpxdec` CLI surface). |
| `rvp9-bench` | Latency + BD-rate + decode-throughput benchmarks. |
| `rvp9-harness` | Conformance + differential test runners / oracle diff tools. |

Dependency direction: encoder and decoder both depend **only** on `rvp9-core`
(+ `rvp9-io`); they do not depend on each other. Dependency footprint is
intentionally tiny — std plus `crossbeam-channel` (tile-threaded decode needs MPMC;
`std::sync::mpsc` is SPSC on the receive side).

### `rvp9-core` module map

`bitstream/` (reader/writer + bool arithmetic coder, header parsers/writers),
`entropy/` (probability tables, `FrameContext`, `BoolWriter`), `transforms/`
(forward+inverse DCT/ADST 4×4…32×32), `quant/`, `intra_pred.rs`, `inter/`
(MV refs, compound modes), `loopfilter/`, `dsp/` (per-arch SIMD dispatch:
scalar / SSE2 / SSSE3 / SSE4.1 / AVX2 / NEON, with HBD stubs under
`--features highbitdepth`), `mode_decision/`, `mv_search/`, `coef_ctx.rs`,
`frame.rs` (frame buffers + ref pool), `tile_threading/`.

### `rvp9-encoder` module map

`frame_loop.rs` (`StreamEncoder`, `EncoderConfig`, the public entry point), `rdo/`,
`rate_control/`, `gop.rs`, `tpl/`, `pre_analysis.rs`, `lookahead.rs`,
`aq_cyclic_refresh.rs` + `segmentation.rs` (adaptive quant, CYCLIC_REFRESH mode),
`temporal_filter.rs`, `denoiser.rs`, `compound.rs`, `svc.rs`, `ml_partition/`,
plus `*stat.rs` / `*oracle*.rs` diagnostic-instrumentation modules.

### v1 scope guardrails (from `notes/hld/00-architecture.md`)

8-bit pixels only (`Pixel = u8`; HBD is a feature-gated scaffold, kernels unimplemented).
AQ = CYCLIC_REFRESH (mode 3) only. CBR + VBR-RT, **no altref** in v1. Single-pass
real-time path; **not** in scope: 2-pass, cpu-used 0–5 slow RDO, AV1/VP8, hardware
encode, C ABI/FFI, WASM.

## Work tracking

The persistent defect ledger is `bugs/`: one markdown file per bug, filename ==
the bug ID, state as a field inside the file. There is no root `BUGS.md`.
Accepted enhancements, design follow-ups, and durable behavioural specs live in
the sibling `reqs/` ledger, one markdown file per requirement. There is no root
`REQS.md`.

Read `bugs/README.md` before raising, fixing, or verifying a tracked bug. Read
`reqs/README.md` before capturing or building a requirement. Use the fleet
`bug-tracking`, `reqs-capture`, and `reqs-loop` skills when their trigger
phrases apply. Mint records with `deltic bugs new` and `deltic reqs new`; never
allocate IDs by hand.

Ordinary development test failures do not automatically become bug-ledger
entries; promote only separate durable defects. A tracked bug fix moves the bug
to `Fixed` with regression evidence; a different verifier closes it. RVP9
requirements are manual/heavy for now: manual/heavy implementations move
`Accepted -> Implemented` when the work lands and never self-promote to
`Satisfied`. Keep `Flow: light` disabled until this workspace has a checked-in
integration policy and a proven workspace-specific oracle.

## Repo-specific conventions

- **libvpx parity is the discipline.** Byte-parity with libvpx is a *diagnostic
  tool*, not a release gate — **BD-rate** is the gate (PLAN.md §3 D8). Every
  deliberate departure from libvpx behaviour MUST be recorded in
  `notes/divergence-ledger.md` (it has a fixed entry format: Where / What libvpx
  does / What rvp9 does / Why / Byte-parity impact / Compression-efficiency impact /
  Future plan). When porting, cite the libvpx file:line you mirrored.

- **Integer widths mirror libvpx exactly** (PLAN.md §3 D7). Wrap-reliant sites use
  explicit `wrapping_*`; everywhere else plain operators — a debug overflow panic is
  a feature, not a bug. `Coeff = i16` (`tran_low_t`), `TranLow = i32` (`tran_high_t`).

- **`RVP9_*` environment variables** gate a large set of trace/diagnostic/instrumentation
  paths (grep `RVP9_` for the full list). Useful ones: `RVP9_FORCE_SCALAR` (bypass
  SIMD for differential testing), `RVP9_DISABLE_ROW_MT` / `RVP9_DISABLE_LF_MT` /
  `RVP9_DISABLE_COLUMN_WAVEFRONT` (serialize threading to isolate bugs), `RVP9_FIXED_Q`,
  `RVP9_TRELLIS`, and many `RVP9_DEBUG_*` / `RVP9_DIAG_*` dumps. These are debugging
  scaffolds — don't wire them into production logic.

- **Lint policy.** Library crates (`rvp9-core`, `rvp9-decoder`, `rvp9-encoder`)
  inherit `[workspace.lints]`: `clippy::all` = warn, plus `deny` on
  `undocumented_unsafe_blocks`, `print_stderr`, `print_stdout`, and `#![deny(missing_docs)]`.
  Every `unsafe {}` in a library crate needs a `// SAFETY:` comment. The CLI / bench /
  harness crates intentionally **do not** inherit these (println/eprintln is their
  correct output mechanism). Some decoder modules carry a temporary crate-level
  `#![allow(clippy::print_stderr)]` for env-var-gated debug breadcrumbs pending a triage sweep.

- **Code changes are versioned during integration.** The workspace version lives
  in `[workspace.package]` in the root `Cargo.toml`. Do not pre-bump task branches;
  `deltic integrate` owns the single bump and keeps `Cargo.toml` and `Cargo.lock`
  in lockstep.

## Documentation layers

- `PLAN.md` — the authoritative multi-stage roadmap and success criteria.
- `wrk_docs/` — HLD/PLN/CR design & review docs (`YYYY.MM.DD - TAG - desc.md`).
- `wrk_journals/` — append-only engineer's log, one file per task; the richest record
  of *why* decisions were made and what was tried.
- `notes/` — design docs, the **divergence ledger**, `hld/` architecture notes,
  and ad-hoc analysis (`notes/_*` is gitignored scratch).
