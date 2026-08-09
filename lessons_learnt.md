# Lessons learnt
<!-- lessons-format: index-v1 -->

Distilled, non-obvious gotchas future sessions need — dated one-liners.
Durable project facts/conventions belong in `CLAUDE.md` / `AGENTS.md`, not here.
Hard cap: 20 entries.

- 2026-08-09: Translating libvpx's eligible `map==0` to an age map means seeding at the age threshold, not zero (`aq_cyclic_refresh.rs:CyclicRefresh::new`).
- 2026-08-09: Libvpx `x->skip` is loop termination, not `skip_txfm`; rvp9's broader gate is a measured safety divergence (`inter.rs:should_evaluate_intra_fallback`).
- 2026-08-09: Intra-in-inter must not replace a prediction-only skip winner (`inter.rs:should_evaluate_intra_fallback`).
- 2026-08-09: Parallel BD-rate runs collide on fixed temp names; serialize until `bdrate.rs:encode` fixes RVP9-BUG-FLUX-00010.
- 2026-08-09: CPU6 needs exact VBP without history copy plus structural-Q 54 (`lib.rs:exact_vbp_partition_qindex`).
- 2026-08-09: CPU8 approximate VBP lost quality and speed; start from the exact tree (`lib.rs:exact_vbp_tree_enabled`).
- 2026-08-09: Encoder PGO needs preset/feature breadth to win wall time (`scripts/build-encoder-pgo.sh:training_rows`).
- 2026-08-09: A tuning miss at one resolution can need a frame-area policy, not global rejection (`lib.rs:exact_vbp_partition_qindex`).
- 2026-08-09: A journaled retained experiment may still lack code; verify both tree and evidence (`y4m.rs:Y4mReader::next_frame_into`).
- 2026-08-09: A captured false flag may survive worker closure lowering; put feature constants at each hot branch (`lib.rs:RUNTIME_PROFILING_COMPILED`).
- 2026-08-09: Exact VBP arithmetic can still over-split at finer live Q; calibrate structural Q separately (`lib.rs:exact_vbp_partition_qindex`).
- 2026-08-09: Exact VBP and source lambda are coupled; either alone regresses curves (`lib.rs:inter_rd_calibration_for_frame`).
- 2026-08-07: Serial compact recon costs ~1–2%; keep it opt-in until row parallelism repays the merge (`lib.rs:compact_recon_rows`).
- 2026-08-07: Compact row recon keeps reference/input coordinates global and maps only destinations (`lib.rs:InterReconPlane`).
- 2026-08-07: Replicated borders: copy row spans + fill halos; per-pixel clamps cost ~1% (`filter.rs:copy_edge_replicated_window`).
- 2026-08-07: Borrow hot immutable SIMD dispatch; Arc clones cost 2.7% wall (`inter.rs:evaluate_intra_fallback_with_scratch`).
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
