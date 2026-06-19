# Lessons learnt

Distilled, non-obvious gotchas future sessions need — dated one-liners.
Durable project facts/conventions belong in `CLAUDE.md` / `AGENTS.md`, not here.
Hard cap: 20 entries.

- 2026-06-17: Module / `lib.rs` doc headers are STALE ("Stage 3 skeleton", "what does NOT work yet"). The encoder and decoder are far more complete than the prose claims — trust `git log`, `PLAN.md`, and `wrk_journals/` over module doc comments.
- 2026-06-17: `.cargo/config.toml` sets `target-cpu=native`, so `--release` binaries are NOT portable across x86_64 hosts (SIGILL/illegal-instruction on an older CPU than the build host). For a portable build use `RUSTFLAGS="" cargo build --release`.
- 2026-06-17: `crates/rvp9-fuzz` is deliberately OUTSIDE the workspace (cargo-fuzz `#![no_main]` → LNK1561 on `cargo build --workspace`). Run it via `cargo +nightly fuzz run <target>` from inside the crate, never through the workspace.
- 2026-06-17: Byte-parity with libvpx is a *diagnostic tool*, not the release gate — **BD-rate** is (PLAN.md §3 D8). Don't chase byte-exactness as the end goal; record every deliberate departure in `notes/divergence-ledger.md` (fixed entry format).
- 2026-06-17: Library crates (`rvp9-core/decoder/encoder`) deny `print_stderr`/`print_stdout`/`undocumented_unsafe_blocks` + `missing_docs`; CLI/bench/harness do NOT inherit these. Debug output in libraries goes through `RVP9_*` env-gated breadcrumbs or `DecodeError`, not `eprintln!`.
- 2026-06-17: README/AGENTS quick-start cite `scripts/build-libvpx.ps1` and `scripts/fetch-vectors.ps1`, but only the `.sh` versions exist (only `fetch-corpus` ships both). Use the `.sh` scripts (MSYS2 on Windows).
- 2026-06-17: The track2 block-size-aware diamond start and coefficient sign-bit costing wins do **not** transfer to current Stage 4a main; A/B restored the known baseline only when disabled.
- 2026-06-17: S2/3D is already a default Stage 4a quality contributor; `RVP9_3D=0` worsens mean matched-bitrate gap from `+2.36 dB` to `+4.25 dB`, so do not count it as an unused recovery lever.
- 2026-06-17: Skipping interpolation-filter search for all integer-pel MVs is a large-frame speed lever but regresses Foreman BD-rate (`+22.88%` -> `+27.24%`); keep that shortcut large-frame gated.
- 2026-06-17: Current fixed-Q Foreman E9 still shows libvpx ahead by `+1.29..+1.69 dB`, so the Stage 4a residual is coding/prediction efficiency rather than rate-control alone.
- 2026-06-19: The decoder LF-overlap wavefront (`spin_then_yield_wavefront`, `loopfilter/apply.rs`) has NO abort/poison escape and `handle.wait()` joins it BEFORE `merge_err` is inspected — so ANY decode error that returns before publishing `decode_progress` (e.g. a `?` short-circuit mid-tile) **deadlocks** the default multithreaded decode instead of returning `Err`. `RVP9_DISABLE_LF_OVERLAP_POOL=1` isolates it. (KILN-00006)
- 2026-06-19: `vpxenc --rt --cpu-used=7 --lag-in-frames=0 --tune-content=default` uses default one-pass VBR, not CBR; keep Stage 4a quality/E1 comparisons in `RcMode::Vbr` unless `--end-usage=cbr` is explicit.
