# HLD — T-5: Per-tile dispatch overhead investigation

**Author:** T-5 design agent
**Date:** 2026-05-24
**Status:** DRAFT — instrumentation-first plan, no fix scoped yet

## 0. Problem statement

`bench/results/decode-headtohead.csv` shows on `vp90-2-08-tile_1x4.webm` (1080p, 4 tile columns):

| Decoder | t=1 | t=4 | Scaling |
|---|---|---|---|
| rvp9 | ~18 ms/frame | ~10 ms/frame | **1.79×** |
| vpxdec | ~8 ms/frame | ~2.5 ms/frame | **~3.3×** |

A 4-tile-column fixture under 4 worker threads is the *ideal* shape for embarrassing parallelism — one tile column per worker, no inter-tile residual dependencies. rvp9 leaves ~45% of theoretical scaling on the table. We need data to localize that lost time inside the rvp9 per-frame execution.

**Constraint repeat.** Sandbox blocks admin profilers (samply, perf, VTune, ETW with PMU access). The plan must work with **Rust-native, unprivileged instrumentation built into the crate behind a feature flag**.

## 1. Tile-dispatch path inventory (read-only audit)

### 1.1 Architecture: persistent pool, scoped fan-out per frame

The tile worker pool is **persistent for the lifetime of the `Decoder`** — not per-frame spawn. Constructed in `Decoder::with_threads` at `D:\language\rvp9-2\crates\rvp9-decoder\src\lib.rs:642-695` via `rvp9_core::tile_threading::WorkerPool::new(n)`. That call spawns `n` OS threads up-front (`D:\language\rvp9-2\crates\rvp9-core\src\tile_threading\pool.rs:66-86`), each spinning in `worker_loop` (`pool.rs:226-241`) on an MPMC `crossbeam_channel::unbounded::<WorkerJob>` receiver.

**This already rules out per-frame OS-level thread spawn as the cost.** What the bench is paying per frame is the *job-submission, locking, and join-barrier* cost on top of the long-lived pool — plus whatever serial work straddles the parallel region.

### 1.2 Per-frame fan-out shape (the load-bearing site)

Two near-identical sites:

- **Intra/keyframe path** — `D:\language\rvp9-2\crates\rvp9-decoder\src\lib.rs:1479-1712`.
- **Inter path** — `D:\language\rvp9-2\crates\rvp9-decoder\src\lib.rs:2342-2609`.

Each frame's parallel region has this shape (inter quoted; intra is structurally identical with fewer per-tile slot fields):

1. **Pre-scope serial setup** (`lib.rs:2342-2400`): compute `per_tile_bounds` for every tile; pull `num_tiles` scratch `Yv12Buffer`s from `self.tile_scratch_pool`; allocate per-tile `InterTileSlot` with five `Mutex<Option<…>>` fields (scratch, seg_ids, mv_grid, mv_counts, coef_counts, mode_counts, result). For the inter path the per-tile `seg_ids` and `mv_grid` are full `mi_rows * mi_cols`-sized buffers — `1080p / 8 = 270×240 = 64800` cells × 4 tiles ≈ 260k entries each, with `mv_grid` carrying `PrevFrameMvRef` (likely ≥ 24 bytes/entry → ~6 MiB allocated per frame just for the four per-tile mv_grids).
2. **`worker_pool.scope(|scope| { for each tile: scope.submit(move || { … }) })`** (`lib.rs:2401-2470`). Each `submit` does: bump submission idx + pending counter (two `SeqCst` atomics), clone the per-scope `done_tx`, transmute the closure to `'static`, send it onto the unbounded crossbeam channel (`scope.rs:80-188`).
3. **Inside the worker closure** (`lib.rs:2406-2468`): lock six per-slot mutexes serially to `.take()` owned scratch state, run `decode_tile_inter(…)` (the actual work — the sole interesting CPU consumer), then lock the same six mutexes again to store the results back, plus a seventh `result` mutex.
4. **Scope return** is a join barrier (`pool.rs:107-194`): main thread spins in `done_rx.recv()` until `pending == 0`, decrementing per worker completion. Every tile must finish before main can advance.
5. **Post-scope serial merge** (`lib.rs:2471-2574`): iterate slots in submission order; for each tile (a) take three counts banks and add them additively into `self.{mv,coef,mode}_counts`; (b) `yv12.merge_tile_rect_from(scratch_ref, luma_row_start..luma_row_end, luma_col_start..luma_col_end)` — a serial memcpy from the per-tile scratch frame into the final `yv12`; (c) row-by-row copy of seg_ids + mv_grid slabs into the frame-wide slabs (`lib.rs:2558-2564`). For 1080p / 4 tile-cols, the luma merge memcpy alone is ~2 MiB of pixel data per frame, single-threaded.
6. **Single-threaded postlude**: stitch per-tile `mi_grid`s via `build_frame_mi_grid` (`lib.rs:2707-2711`), then `apply_loop_filter_to_frame` (`lib.rs:2713-2724` → `lib.rs:2981-3083`) which iterates **every superblock sequentially** building `LoopFilterMask`s and then runs `loop_filter_frame_with_non420` — **the loopfilter is single-threaded entirely**. Then `extend_frame_borders(&mut yv12)` (`lib.rs:2732`), another serial memcpy.

### 1.3 Synchronization primitives in the hot loop

- **6 per-slot `std::sync::Mutex<Option<T>>`** per tile, locked twice per worker (take + restore). 4 tiles × 6 mutexes × 2 lock cycles = **48 mutex lock acquire/release pairs per frame**, all on slots only ever touched by one worker (i.e., uncontended but still ~25-40 ns each).
- **2 atomic `SeqCst` ops per `submit`** on `next_submission_idx` and `pending` (`scope.rs:91-92`).
- **`crossbeam_channel::unbounded` MPMC channel** for job dispatch; one send per submit, one recv per worker; on Windows this hits an event/futex per send when workers are parked.
- **Per-scope `done_tx` MPSC** — one send per worker completion, one recv per completion on main.
- **`RVP9_G5_1_PARALLEL_JOBS.fetch_add(1, Relaxed)`** inside every worker closure (`lib.rs:1549-1550`, `lib.rs:2407-2408`) — cheap (Relaxed) but a contended cache line if every tile bumps it.

## 2. Hypothesis tree — what could explain 1.79× on 4 tile cols / 4 threads?

Ranked by likelihood with the evidence each carries:

| # | Hypothesis | Evidence for | Evidence against | Likelihood |
|---|---|---|---|---|
| **H1** | **Serial post-scope merge + loopfilter dominate.** The loopfilter is unconditionally single-threaded (`lib.rs:2981-3083` walks every SB on main). Merge memcpy is ~2 MiB/frame on main. If serial work is, say, 4-5 ms/frame, Amdahl caps 4-thread scaling at ~2× regardless of how clean the parallel region is. | rvp9's serial loopfilter is structurally there; vpxdec runs loopfilter as a separate worker (`vp9_lpf_worker.c`) overlapped with row decode of the next frame; the `wrk_docs/Track_T` README references this. The 1.79× ≈ Amdahl(parallel_fraction ≈ 0.65) on 4 cores — consistent with ~35% serial. | None obvious. | **HIGH** |
| **H2** | **Per-tile mutex locking + the 6 Mutex<Option> handshake adds significant per-tile overhead.** 48 mutex ops/frame are small individually but the cache-line bouncing on `Sender<WorkerJob>::clone`, `pending` SeqCst, and the worker-local locks within a tightly co-located `Vec<InterTileSlot>` can add up. | Each mutex lock-unlock is ~25-40 ns uncontended; 48 × 40 ns ≈ 2 μs — too small to explain 50% scaling loss. But the *false sharing* between adjacent `InterTileSlot`s (likely co-allocated < 64 B apart in the `Vec`) could cost much more. | Mutex math is tiny compared to multi-millisecond gaps; would need false sharing to dominate. | LOW (cost dominates: only if false sharing → MEDIUM) |
| **H3** | **Per-frame per-tile allocations** (3+ `Box<…CountsBank>` per slot, `Vec<u8>` for seg_ids, `Vec<PrevFrameMvRef>` for mv_grid — all sized `mi_rows * mi_cols`). At 1080p / 4 tiles this is ~6 MiB of allocator churn per frame. Allocator may serialize through the global heap. | Pre-scope allocation is single-threaded but runs **before** the parallel region — it's serial setup. Post-scope `.into_inner()` drops are also serial. | The `tile_scratch_pool` already exists for `Yv12Buffer` recycling — but the *counts banks*, *mv_grid*, and *seg_ids* allocations are **not** pooled and reallocate every frame. | **MEDIUM-HIGH** |
| **H4** | **MC border copy serialization** — `build_inter_predictor_with_mc_border` (`reconstruct_inter.rs:1039+`) runs inside per-tile workers but reads from shared `extend_borders`-padded ref frames. No serialization at the per-tile level; the borders are populated once per ref frame, *before* decode starts. | The `extend_borders` calls live in `reconstruct_inter.rs:1334, 1599, 1608` — but on **the destination side** during the previous frame's `extend_frame_borders` (lib.rs:2732). For tile_1x4 inter frames, ref borders are pre-populated; per-tile MC reads are independent. | No shared-mutable on the read side; this is unlikely to be the bottleneck. | LOW |
| **H5** | **Inter-tile work imbalance.** If one tile column has substantially more residual / motion vectors than the others, that tile is the long pole; other 3 workers sit idle waiting. | tile_1x4 is a *synthetic* fixture — content is uniform across columns. Imbalance unlikely to exceed ~10-15%. | None. | LOW |
| **H6** | **`crossbeam_channel::unbounded` MPMC send is heavier than expected on Windows.** Each `submit` does `Box::new(closure)` + atomic enqueue + a futex/event signal to wake a sleeping worker. At 4 jobs/frame × ~5-10 μs/dispatch on Windows = ~40 μs/frame. | Plausible cost is ~40 μs/frame; against an 8 ms parallel region that's <1%. | The send cost itself is small; unless worker wake-up latency is large on Windows (typically a few μs), this is sub-dominant. | LOW |
| **H7** | **Tile-grid stitching (`build_frame_mi_grid`) does an extra alloc + copy of the entire frame's mi grid on every inter frame** (`lib.rs:2707-2711`). | This runs serially between the parallel scope and the loopfilter. For 1080p it's ~65k `Option<ModeInfo>` cells. | Should be <500 μs; not enough to explain the gap alone, but contributes to the serial fraction. | MEDIUM (contributes to H1) |
| **H8** | **`extend_frame_borders` is unconditionally serial** (`lib.rs:2732`). For 1080p YUV420 it touches ~3 MiB. | Definitely serial; contributes to Amdahl serial fraction. | A few hundred microseconds at memcpy speeds. | MEDIUM (contributes to H1) |

**Lead hypothesis (H1 + H3 + H7 + H8 stack):** the dominant cost is **serial work straddling the parallel region** — loopfilter + merge memcpy + grid-stitching + border extend + un-pooled per-frame allocator churn. These together likely consume 30-40% of frame time. With ~35% serial, Amdahl's law predicts (1 / (0.35 + 0.65/4)) = **1.97×** speedup on 4 cores — uncannily close to the observed 1.79×. The data we collect should either confirm this geometry or invalidate it cleanly.

## 3. Instrumentation design

### 3.1 Choice: `rdtsc` + thread-local accumulators behind `profile-tiles` cargo feature

**Why rdtsc, not `Instant`:**

- `std::time::Instant::now()` on Windows calls `QueryPerformanceCounter` (~20 ns/call). Acceptable for ms-scale phases but borderline for sub-millisecond per-superblock-block timings.
- `core::arch::x86_64::_rdtsc()` is ~20-30 cycles (~10 ns at 3 GHz), unprivileged, and reads the invariant TSC on all post-Nehalem CPUs (which the build host is). Perturbation < 0.1% on tile-decode phases.
- No `unsafe` block needed at the call site beyond what the intrinsic requires — `_rdtsc()` itself is `unsafe fn`, but a small wrapper hides it.

**Why a cargo feature flag, not env var or runtime knob:**

- Code is `#[cfg(feature = "profile-tiles")]`-gated. **With the feature off (default), the instrumentation compiles to nothing** — zero runtime overhead, conformance bit-exact, bench numbers unchanged (validated in §8).
- Env-gated runtime checks would still cost a load + branch in the hot loop even when off; we want true zero cost.
- A feature flag is also CI-visible: `cargo test --features profile-tiles -p rvp9-decoder` runs the conformance suite *with* instrumentation, validating bit-exactness under the timing path.

**Why thread-local accumulators, not a global Mutex:**

- Per-tile workers update their own thread-local timing buckets without contention.
- At scope-join time, the main thread walks the worker handles (or, since the workers persist, queries a `thread_local!` from each known worker ID via a side-channel) and aggregates.
- For the per-frame report, we use a simpler approach: each worker writes its per-phase cycle deltas into the per-tile `InterTileSlot` (already exists, locked once at end-of-job to restore state). That avoids a parallel thread-local discovery problem.

### 3.2 Phases to time

Defined as cycle deltas between rdtsc reads, summed per tile + per frame:

| Phase ID | Where (file:line range) | Where exactly | Owner thread |
|---|---|---|---|
| `T_SETUP` | `lib.rs:2342-2400` (inter) / `lib.rs:1479-1543` (intra) | Pre-scope: `per_tile_bounds` build + `tile_scratch_pool.take` + slot Vec allocation | main |
| `T_SUBMIT` | `lib.rs:2401-2470` (inter) / `lib.rs:1543-1587` (intra) | From `worker_pool.scope` open to first job dispatched (measures submit-loop overhead on main; per-tile cycles spent in `scope.submit`) | main |
| `T_WAIT` | First worker-side rdtsc inside closure | `(rdtsc_at_worker_start - submission_idx_wallclock)` — proxy for "time until worker picked up the job" | worker (cross-thread delta requires shared timestamp; see §3.4) |
| `T_TILE_LOCK_IN` | `lib.rs:2409-2444` (inter) | Six per-slot mutex `.lock().take()` ops at job start | worker |
| `T_TILE_DECODE` | `lib.rs:2445-2459` (inter) / `lib.rs:1569-1578` (intra) | The actual `decode_tile_inter`/`decode_tile_intra` call — this is the *useful* work | worker |
| `T_TILE_LOCK_OUT` | `lib.rs:2461-2467` (inter) | Six per-slot mutex restores at job end | worker |
| `T_JOIN` | Inside `pool.rs:scope` drain loop, or wrapped at the `lib.rs:2470` scope-return boundary | From "last worker submitted" to "scope returns" — measures join-barrier wait on main | main |
| `T_MERGE` | `lib.rs:2471-2571` (inter) | Post-scope serial merge of pixels + seg_ids + mv_grid + counts | main |
| `T_GRID_STITCH` | `lib.rs:2707-2711` | `build_frame_mi_grid` | main |
| `T_LOOPFILTER` | `lib.rs:2712-2724` → `apply_loop_filter_to_frame` | The whole LF | main |
| `T_BORDER` | `lib.rs:2732` | `extend_frame_borders` | main |

Sum-of-phases = **total frame time** (sanity check the instrumentation: `Σ(T_*) ± 2%` of wall-time via `Instant`).

### 3.3 The wrapper

```rust
// crates/rvp9-decoder/src/profile.rs (NEW, only compiled under feature)
#[cfg(feature = "profile-tiles")]
pub mod profile_tiles {
    // Inline wrapper around _rdtsc()
    #[inline(always)]
    pub fn now_cycles() -> u64 {
        // SAFETY: _rdtsc is unconditionally safe on x86_64 builds;
        // we gate compilation on cfg(target_arch = "x86_64").
        unsafe { core::arch::x86_64::_rdtsc() }
    }

    // Per-frame, per-tile timing bucket. One per InterTileSlot
    // (and TileSlot for intra) under the feature.
    #[derive(Default, Copy, Clone, Debug)]
    pub struct PerTileTiming {
        pub lock_in: u64,
        pub decode: u64,
        pub lock_out: u64,
    }

    // Per-frame main-thread timing bucket. Aggregated per decode_frame.
    #[derive(Default, Copy, Clone, Debug)]
    pub struct PerFrameTiming {
        pub setup: u64,
        pub submit: u64,
        pub join: u64,
        pub merge: u64,
        pub grid_stitch: u64,
        pub loopfilter: u64,
        pub border: u64,
        pub per_tile: Vec<PerTileTiming>,
        pub frame_total_wall_us: f64, // QPC sanity check
    }

    // Process-wide sink. After decode_frame returns, the wrapper appends
    // a CSV row to a path from RVP9_PROFILE_TILES_OUT env var; if unset,
    // emits a single line to stderr.
    pub fn emit_row(t: &PerFrameTiming, frame_idx: u64);
}
```

Use sites are `#[cfg(feature = "profile-tiles")]`-gated single-line additions, e.g.:

```rust
#[cfg(feature = "profile-tiles")] let t0 = profile_tiles::now_cycles();
// ... phase body ...
#[cfg(feature = "profile-tiles")] { timing.merge = profile_tiles::now_cycles() - t0; }
```

**The wrapper compiles to literally nothing when the feature is off** — `cfg(feature)` is evaluated at parse time before borrow checking.

### 3.4 Cross-thread T_WAIT (queue latency)

T_WAIT is tricky because we need cycle counts on two threads. Solutions, in order of preference:

1. **Just measure cycles on the worker thread from a baseline rdtsc taken at submit time, captured in the closure.** `_rdtsc()` returns invariant TSC monotonically across all cores on Skylake+; the delta `(worker_start_tsc - submit_tsc)` is a valid measure of "queue + wake-up latency". This is the recommended approach. Carry `submit_tsc` as a `u64` capture into the closure.
2. **Fallback:** if TSC sync across cores is suspect (older CPUs), use `Instant::now()` for `T_WAIT` only — sub-microsecond is fine, the cost is in the milliseconds.

### 3.5 Aggregation across N runs

Each frame emits one CSV row with one column per phase (units: cycles, plus a derived μs column via `cycles * 1e6 / cpu_hz_estimate`). CPU frequency estimated once at decoder init by spinning rdtsc against `Instant` for 10 ms (~100 K samples) and dividing.

CSV schema:
```
schema=1,frame_idx,num_tiles,num_threads,wall_us,setup,submit,join,merge,grid_stitch,loopfilter,border,tile0_decode,tile0_lockin,tile0_lockout,tile1_decode,…
```

Schema bumps with column changes. Reader (analysis tool, §5) verifies column 0.

## 4. Data collection plan

### 4.1 Capture point

Single CSV file, path from `RVP9_PROFILE_TILES_OUT` env var (default: skip emission). On unset, no I/O — the timing buckets accumulate per frame but the row is dropped. This decouples build (always include the instrumentation when feature is on) from capture (opt-in per process invocation).

### 4.2 Run procedure

A new bench binary `crates/rvp9-bench/src/bin/decode-profile-tiles.rs` (NEW, under feature-gated build) drives:

```text
cargo run --release --features rvp9-decoder/profile-tiles \
    --bin decode-profile-tiles -- \
    --fixture tests/conformance/vectors/vp90-2-08-tile_1x4.webm \
    --threads 4 \
    --runs 10 \
    --warmup 30 \
    --min-timed 1000 \
    --out bench/results/t5-tile-profile.tile_1x4.t4.csv
```

The binary:

1. Constructs `Decoder::with_threads(4)`.
2. Estimates CPU MHz once.
3. Sets `RVP9_PROFILE_TILES_OUT=<--out path>`.
4. Replays the fixture in a loop until `--min-timed` frames have been timed (matches the existing `decode-headtohead` discipline at `crates/rvp9-bench/src/throughput.rs:54-66`).
5. Repeats for `--runs` independent decoder instances (re-construct each run; we want allocator + cache cold state once per run, not once per process).

### 4.3 Avoiding the Heisenbug

- rdtsc cost (~10 ns) is < 0.001% of a ~10 ms frame — categorically safe at frame scale and ≪ 0.1% at per-tile-phase scale (1-3 ms).
- CSV write is **outside** the timed window (done in a `Drop`-style finalizer after `decode_frame` returns, with the row pre-formatted in a `String` and a single `write_all`).
- We do **not** time per-superblock or per-MI-block in the first cut — that would change the cache footprint. Stays at phase granularity.
- The `RVP9_G5_1_PARALLEL_JOBS.fetch_add(1, Relaxed)` already in the worker closure (`lib.rs:1549-1550, 2407-2408`) is a known Relaxed bump and is already in the baseline.

## 5. Analysis plan — what patterns confirm each hypothesis

The CSV produces 10 runs × ~1000 timed frames each = ~10,000 rows. The analysis script (`scripts/t5-analyze.py` or a `rvp9-bench` binary `t5-summarize`) computes per-phase **median, p95, and sum-share** across all rows.

| Observation | Confirms | Action |
|---|---|---|
| `median(T_LOOPFILTER + T_MERGE + T_BORDER + T_GRID_STITCH + T_SETUP) ≥ 30%` of frame total | **H1: serial work dominates.** | Pipeline loopfilter against next-frame decode (libvpx's approach) OR parallelize LF by SB rows; pool the per-tile counts/seg/mv allocations. |
| Worker `T_TILE_DECODE` p50 is balanced across tiles (within 15%) but `T_JOIN` on main is > 1 ms | **H1 join-barrier flavor**: workers finish near-simultaneously but main waits in `done_rx.recv`. Likely Windows wake-up latency. | Reduce join cost by replacing per-job done-channel with a single AtomicI32 + condvar; or pipeline (see H1). |
| Median `T_TILE_LOCK_IN + T_TILE_LOCK_OUT` per tile > 5% of `T_TILE_DECODE` | **H2: mutex overhead non-trivial.** | Replace per-slot `Mutex<Option<T>>` handshake with owned-pass via channel or `UnsafeCell<MaybeUninit>` + scope-bounded `unsafe`. |
| Variance of `T_TILE_DECODE` across tiles > 20% on a tile_1x4 fixture | **H5: imbalance** (unexpected on a synthetic). | Probably content-dependent; widen test corpus. |
| `T_SETUP` > 500 μs on inter frames | **H3: alloc churn.** | Extend `tile_scratch_pool` to also pool `Box<NmvCounts>`, `Box<CoefCountsBank>`, `Box<FrameCounts>`, `Vec<u8>` (seg_ids), `Vec<PrevFrameMvRef>` (mv_grid). |
| `T_WAIT` (queue + wake) > 50 μs per tile on average | **H6: dispatch/wake latency**. | Replace crossbeam channel with a tighter spsc-per-worker + lock-free queue, or switch to a `Barrier`-based "broadcast wake, parallel-fan-in" pattern. |
| `T_BORDER + T_GRID_STITCH` together > 1 ms | **H7/H8** contributes to H1 serial fraction. | Lazy-stitch grid (build during merge); fold border extend into the LF pass. |

The output is a **ranked table of % of frame time attributable to each phase**, with the dominant phase pointing directly at the fix.

## 6. First-slice deliverable (achievable overnight)

This is what the implementation agent should land in one PR. **No fixes** — purely instrumentation + data.

### 6.1 Files to change

1. **NEW** `crates/rvp9-decoder/src/profile.rs` (~80 lines) — `PerTileTiming`, `PerFrameTiming`, `now_cycles`, `emit_row`, CPU-MHz estimator. Fully `#[cfg(feature = "profile-tiles")]`.
2. **EDIT** `crates/rvp9-decoder/Cargo.toml` — add `[features] profile-tiles = []` (default off).
3. **EDIT** `crates/rvp9-decoder/src/lib.rs` — add `#[cfg(feature)] mod profile;` and ~30 small cfg-gated insertions around the phase boundaries listed in §3.2 (intra + inter paths). Each insertion is 1-3 lines and a single `let t = profile::now_cycles()`. **Zero changes to the parallel/serial control flow.** This is critical: instrumentation must not perturb scheduling.
4. **NEW** `crates/rvp9-bench/src/bin/decode-profile-tiles.rs` (~120 lines, modeled on `decode-headtohead.rs`). Drives the run procedure from §4.2.
5. **EDIT** `crates/rvp9-bench/Cargo.toml` — add the new binary entry; depend on `rvp9-decoder/profile-tiles` only when its own (matching) `profile-tiles` feature is on.
6. **NEW** `scripts/t5-analyze.py` (~60 lines) — reads the CSV, prints the per-phase summary table from §5.
7. **NEW** `wrk_docs/Track_T/T5_findings.md` — the data + ranked-hypothesis interpretation (written by the implementation agent after the run).

### 6.2 Exit gates for the first slice

- Conformance: `cargo test -p rvp9-harness --features rvp9-decoder/profile-tiles` ⇒ 201/0/0/0/201 still green.
- Bit-exact: `cargo test -p rvp9-harness` (feature off) ⇒ identical to today.
- Bench delta: `decode-headtohead t=4 tile_1x4` with feature OFF ⇒ p50 within ±2% of today's ledger.
- Bench delta: same with feature ON ⇒ document the perturbation; expect < 5% slowdown.
- Data: `bench/results/t5-tile-profile.tile_1x4.t4.csv` with at least 10,000 rows; `t5-analyze.py` runs and prints the summary table.
- Doc: `T5_findings.md` checked in with the dominant-phase interpretation and a recommended next-step fix.

## 7. Action plan from findings (decision tree, follow-up scope)

Based on which hypothesis the data confirms:

### 7.1 If H1 (serial work dominates — most likely)

**Two libvpx-known fixes, choose by sub-phase ownership:**

- **If loopfilter is the dominant serial phase** (>40% of serial budget): pipeline LF against the next frame's tile decode (libvpx's `vp9_loop_filter_worker.c` pattern — a dedicated LF worker that the main thread fires-and-continues, joined at the start of the *next* frame's decode). Detailed design: a second `WorkerPool` slot (or a dedicated `JoinHandle`) owns the LF; the per-frame loop becomes `decode_tiles → fire_lf_async → return frame buffer`. Effectively turns the LF cost into ~0 wall-time amortized.
- **If merge+grid-stitch+border together dominate**: fold them into the LF pass (LF already reads/writes pixels) and pool the per-tile counts/seg/mv allocations. Cheaper PR — single-file change in `lib.rs`.

### 7.2 If H2 (mutex overhead — surprising but possible)

Replace the `Mutex<Option<T>>` handshake with a scope-bounded `UnsafeCell` pass: each worker owns its slot's `&mut T` by construction (one-worker-per-slot is structural — the loop in `lib.rs:2402` indexes by tile_idx and never aliases). Costs an `unsafe { … }` with a SAFETY comment, gives ~50 mutex acquires/frame for free. Reference: the same pattern is used in `rayon::scope`.

### 7.3 If H3 (alloc churn)

Extend `tile_scratch_pool` (currently `Yv12Buffer`-only, `crates/rvp9-core/src/frame.rs` near `Yv12ScratchPool`) to a generic `Box<T: Default> / Vec<T: Default>` pool. ~150 LOC; pure refactor; bit-exact by construction.

### 7.4 If H6 (dispatch overhead)

Investigate Windows futex wake latency vs the cost of `crossbeam_channel`. Worth comparing against `std::sync::Barrier` + per-worker `Condvar` (each worker has a private "go" flag flipped by main). Or move to a persistent per-worker SPSC channel (4 channels of capacity 1).

All four are plausibly multi-week-scoped depending on which lands.

## 8. Validation strategy — bit-exact safety

The instrumentation **must not** change decode output. Mechanisms:

1. **Default-off feature flag.** `cargo build` ⇒ zero instrumentation in the binary. The entire `profile.rs` module + all use-site lines are `#[cfg(feature = "profile-tiles")]`-gated.
2. **No mutation of decode state.** Every instrumentation line is either `let t = profile::now_cycles()` (a register read) or `timing_struct.field = t1 - t0` (a write to a fresh per-frame allocation that is dropped after CSV emission). Zero touches on `Decoder` fields, `frame_contexts`, `yv12`, or any other decode output.
3. **CI sanity:** the `profile-tiles` feature is wired into one CI job that runs `cargo test -p rvp9-harness --features rvp9-decoder/profile-tiles` against the full 201-vector corpus, the foreman regression, and the profile1 cross-chroma test. Conformance ledger must stay 201/0/0/0/201.
4. **Bench-off regression check:** the first-slice PR runs `decode-headtohead` (feature off) and compares to the most recent ledger entry. p50/mean within ±2%.
5. **Worker safety:** the instrumentation never holds the per-slot mutex longer than today. The worker still writes its `PerTileTiming` *into the existing slot* (one new field on `InterTileSlot` / `TileSlot`, gated on the feature) under the existing `result` mutex pattern — no new locking primitives.

## 9. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | Instrumentation perturbs the bottleneck (Heisenbug). | LOW | HIGH | rdtsc is ~10 ns/call vs millisecond-scale phases — perturbation < 0.1%. CSV write is outside the timed region. We avoid per-MI-block timing entirely — only phase boundaries. Validated by comparing instrumentation-on vs instrumentation-off bench p50 in the first-slice exit gate (§6.2). |
| **R2** | Per-thread accumulators require thread-local discovery on persistent pool workers — non-trivial. | MEDIUM | MEDIUM | Sidestepped: workers write timing into the per-tile `InterTileSlot.timing` field (one new field, feature-gated). Main thread reads it post-scope. No `thread_local!` needed. |
| **R3** | User-mode rdtsc unreliable on Windows or non-Skylake CPUs (TSC not invariant, or `RDTSCP` needed for serialization). | LOW | MEDIUM | (a) The build host is post-Skylake (invariant TSC universal). (b) For per-phase timing, sub-cycle drift across cores is irrelevant — phases are 100K-1M cycles. (c) Fallback: feature-gate the rdtsc behind `cfg(target_arch = "x86_64", target_feature = "sse2")` AND env var `RVP9_PROFILE_TILES_USE_QPC=1` to switch to `Instant::now()` (QPC on Windows). Validated by a one-time MHz-calibration round at start of each run; large drift is logged. |
| **R4** | The first slice surfaces data that demands a multi-week refactor (e.g., pipelined LF), and the data analysis is incorrectly framed. | MEDIUM | LOW | The HLD is explicit (§6, §7): first slice is **instrumentation + data + findings doc** only. Action plan from findings is scoped *after* data lands; no commitment to a particular fix today. |
| **R5** | The feature flag is left on accidentally in a release build. | LOW | MEDIUM | Default off in `[features]`. CI release-build job (already exists) builds without the flag; `cargo audit` style check confirms `profile-tiles` is not in any release CI invocation. The bench binary `decode-profile-tiles` requires the feature, so accidental "release with profiling off" produces a clear compile error. |

## 10. Out of scope for this HLD

- The fix itself. §7 is a *decision tree* for the follow-up; the fix is whatever the data points to, scoped in its own HLD.
- Frame-parallel decode (Track Q.1.C territory).
- libvpx CPU-time comparison (different problem; we already have the wall-time gap — the question is rvp9-internal localization).
- Per-MI-block profiling (out of first-slice; needs different instrumentation discipline).

## 11. Open questions

1. **Should the loopfilter timing distinguish "mask build" from "kernel dispatch"?** `apply_loop_filter_to_frame` has both (`lib.rs:2992-3050` is mask build; `lib.rs:3058+` is the per-SB kernel). Recommendation: add it (one extra rdtsc pair, free), so the fix path is clearer.
2. **Should we also time the single-threaded `decode_compressed_partition_inter` path** (before the parallel scope)? It's part of every frame and contributes to the serial fraction. Recommendation: yes, add a `T_COMPRESSED_HEADER` phase. One rdtsc pair, zero cost.
3. **Is `tile_1x4` representative?** Per HLD it's a synthetic. We should also profile a real-world clip (e.g., the foreman-fp1 fixture, even though it has 1 tile) to see whether the serial-fraction story holds on production content.

---

### Critical Files for Implementation

- `D:\language\rvp9-2\crates\rvp9-decoder\src\lib.rs` (insert instrumentation at the phase boundaries listed in §3.2: intra path `lib.rs:1479-1712`, inter path `lib.rs:2342-2609`, post-merge `lib.rs:2685-2732`)
- `D:\language\rvp9-2\crates\rvp9-decoder\src\profile.rs` (NEW — feature-gated timing types, rdtsc wrapper, CSV emit, MHz estimator)
- `D:\language\rvp9-2\crates\rvp9-decoder\Cargo.toml` (add `[features] profile-tiles = []`)
- `D:\language\rvp9-2\crates\rvp9-core\src\tile_threading\pool.rs` (read-only reference: the drain-loop join barrier at `pool.rs:107-194` is the structural reason `T_JOIN` exists)
- `D:\language\rvp9-2\crates\rvp9-bench\src\bin\decode-profile-tiles.rs` (NEW — driver binary, modeled on `decode-headtohead.rs`)
