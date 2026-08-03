# Lessons learnt
<!-- lessons-format: index-v1 -->

Distilled, non-obvious gotchas future sessions need — dated one-liners.
Durable project facts/conventions belong in `CLAUDE.md` / `AGENTS.md`, not here.
Hard cap: 20 entries.

- 2026-08-03: Partial LF can pad to the next SB row (`loopfilter.rs:apply_loop_filter_for_packet_overlap`).
- 2026-08-02: Keyframe VBP comparisons must retain libvpx's 256x variance precision; folding both sides to integers changes `504 > 450` into `1 > 1` (`partition.rs:keyframe_4x4_average_variance`).
- 2026-08-02: Realtime keyframe leaves >=16 use Hadamard+`quantize_fp` SATD pricing, not full-RD DCT pricing (`intra.rs:pick_realtime_intra_mode_luma_large`).
- 2026-06-17: Module / `lib.rs` doc headers are STALE ("Stage 3 skeleton", "what does NOT work yet"). The encoder and decoder are far more complete than the prose claims — trust `git log`, `PLAN.md`, and `wrk_journals/` over module doc comments.
- 2026-06-17: `.cargo/config.toml` sets `target-cpu=native`, so `--release` binaries are NOT portable across x86_64 hosts (SIGILL/illegal-instruction on an older CPU than the build host). For a portable build use `RUSTFLAGS="" cargo build --release`.
- 2026-06-17: `crates/rvp9-fuzz` is deliberately OUTSIDE the workspace (cargo-fuzz `#![no_main]` → LNK1561 on `cargo build --workspace`). Run it via `cargo +nightly fuzz run <target>` from inside the crate, never through the workspace.
- 2026-06-17: Byte-parity with libvpx is a *diagnostic tool*, not the release gate — **BD-rate** is (PLAN.md §3 D8). Don't chase byte-exactness as the end goal; record every deliberate departure in `notes/divergence-ledger.md` (fixed entry format).
- 2026-06-17: Library crates (`rvp9-core/decoder/encoder`) deny `print_stderr`/`print_stdout`/`undocumented_unsafe_blocks` + `missing_docs`; CLI/bench/harness do NOT inherit these. Debug output in libraries goes through `RVP9_*` env-gated breadcrumbs or `DecodeError`, not `eprintln!`.
- 2026-06-17: The track2 block-size-aware diamond start and coefficient sign-bit costing wins do **not** transfer to current Stage 4a main; A/B restored the known baseline only when disabled.
- 2026-06-17: S2/3D is already a default Stage 4a quality contributor; `RVP9_3D=0` worsens mean matched-bitrate gap from `+2.36 dB` to `+4.25 dB`, so do not count it as an unused recovery lever.
- 2026-06-17: Skipping interpolation-filter search for all integer-pel MVs is a large-frame speed lever but regresses Foreman BD-rate (`+22.88%` -> `+27.24%`); keep that shortcut large-frame gated.
- 2026-06-17: Current fixed-Q Foreman E9 still shows libvpx ahead by `+1.29..+1.69 dB`, so the Stage 4a residual is coding/prediction efficiency rather than rate-control alone.
- 2026-06-19: The decoder LF-overlap wavefront (`spin_then_yield_wavefront`, `loopfilter/apply.rs`) has NO abort/poison escape and `handle.wait()` joins it BEFORE `merge_err` is inspected — so ANY decode error that returns before publishing `decode_progress` (e.g. a `?` short-circuit mid-tile) **deadlocks** the default multithreaded decode instead of returning `Err`. `RVP9_DISABLE_LF_OVERLAP_POOL=1` isolates it. (KILN-00006)
- 2026-06-19: `vpxenc --rt --cpu-used=7 --lag-in-frames=0 --tune-content=default` uses one-pass VBR; native Xiph Y4Ms declare 50fps, so pass `--fps=30/1` to both legs for the 720p30 gate.
- 2026-06-19: libvpx seeds `avg_frame_qindex` to worst only for one-pass CBR; one-pass VBR/CQ/Q/two-pass use `(worst+best)/2`, otherwise first VBR keyframes starve at q≈195 instead of q≈57.
- 2026-06-19: Zero-EOB inter TX blocks are hot in the 720p realtime emit path; store empty `eob == 0` markers and let the coef source return a static zero slice instead of allocating zero qcoeff/dqcoeff vectors.
- 2026-06-19: E9 fixed-Q Foreman is not a pure equal-reference oracle: libvpx `--end-usage=q --cq-level=N` still gives the keyframe a much lower base qindex (Q80 maps to KF q=15), so interpret inter residuals as reference-contaminated.
- 2026-06-20: Current E12 is a stale single-ref LAST oracle; use E13 dual-ref argmin counters for current RT NEWMV rejection/search claims.
- 2026-06-20: `bitacct` must suppress dry-run tile encodes and reset `BoolWriter` boundaries to `Other`; otherwise E9/E4 category bits double-count A6 count passes and smear compressed-header bits into tile buckets.
- 2026-06-20: The clippy gate is `cargo clippy --workspace --all-targets` WITHOUT `-D warnings`. `[workspace.lints]` keeps `clippy::all = warn` (and even `missing_docs = warn`); only `undocumented_unsafe_blocks`/`print_stderr`/`print_stdout` are `deny`. Adding `-D warnings` surfaces ~111 deliberately-non-blocking style/complexity warnings — only the denied lints are real gate failures; don't mass-"fix" the rest.
- 2026-06-20: A fresh `git worktree` does NOT inherit the main clone's gitignored fetched data, so `cargo test --workspace` there hard-fails 3 data-dependent tests (`bitstream::*captured*round_trip`, `real_world_corpus_decodes_bit_exact`) with "test vector must be present" though the code is fine. Run them from the data-populated clone or fetch first; don't chase them as code bugs.
