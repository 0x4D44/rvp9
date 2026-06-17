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
