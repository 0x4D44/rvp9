# HLD — Q.1.A: End-to-end decode performance comparison rvp9dec vs vpxdec

**Author:** Q.1.A design agent
**Date:** 2026-05-23
**Status:** SIGNED OFF by lead 2026-05-23 (with revisions; see §11)

## 1. Scope and gate semantics

This phase delivers **two artifacts**, neither of which is a hard CI gate yet:

1. **Informational head-to-head ledger** (`bench/results/decode-headtohead.md` + `.csv`) — one row per `(fixture, threads)`, columns for rvp9dec and vpxdec medians + a ratio. Committed.
2. **Hooks for the PLAN.md §2 hard real-time gates** (`PLAN.md:31-34`): emit a `PASS/FAIL` column comparing median per-frame decode time against the budget (33 ms or 16 ms), but flag the row as *informational on unpinned hosts* — same policy as the existing soft ratchet at `crates/rvp9-bench/tests/decode_throughput_ratchet.rs:45` (`PINNED_HOSTS: &[&str] = &[]`).

What is **out of scope for this phase**:
- A panicking gating test on PLAN §2 budgets. The existing ratchet panics only on pinned hosts (`decode_throughput_ratchet.rs:190-201`); we adopt the same posture and *do not* introduce a new pinned-host requirement in Q.1.A.
- BD-rate / quality numbers (decoder is bit-exact already — quality is the encoder's job).
- A direct comparison on the per-frame timing axis. `vpxdec --framestats` emits only `bytes,qp` (verified by inspection of the help output and a sample run on `vp90-2-08-tile_1x4.webm`); there is no per-frame timing surface to read. We therefore time vpxdec as a whole-run wall-time and convert to a per-frame mean — see §3.

## 2. Fixture set

PLAN §2 (`PLAN.md:32-34`) demands 720p30, 1080p30, 1080p60. The four conformance vectors in `crates/rvp9-bench/src/throughput.rs:91-96` are 10-frame micro-clips at unknown small resolutions — they are useful for the *steady-state* soft ratchet but cannot exercise the PLAN budgets. We therefore split the fixture set:

| Slot | Fixture | Provenance | Why |
|---|---|---|---|
| A | `tests/conformance/vectors/vp90-2-00-quantizer-00.webm` | already on disk (`148 KB`) | Single-tile control; reused from `throughput.rs:92`. Detects regressions in shared decode helpers separate from tile dispatch. |
| B | `tests/conformance/vectors/vp90-2-08-tile_1x4.webm` | already on disk | Multi-tile control; exercises the `WorkerPool` dispatch path. |
| C | `tests/corpus/fixtures/testsrc_720p30.ivf` | **encode required** from `tests/corpus/testsrc_720p30.y4m` (198 MB, on disk) via `scripts/encode-corpus.ps1:28-32` | Maps to PLAN §2 720p30 budget (33 ms / 50 ms p99). |
| D | `tests/corpus/fixtures/testsrc_1080p30.ivf` | **encode required** from `tests/corpus/testsrc_1080p30.y4m` (445 MB, on disk) | Maps to PLAN §2 1080p30 budget. |
| E | `tests/corpus/fixtures/testsrc_1080p60.ivf` | **encode required** from `tests/corpus/testsrc_1080p60.y4m` (890 MB, on disk) | Maps to PLAN §2 1080p60 budget. |
| F | `tests/corpus/fixtures/foreman-fp1.ivf` | already on disk (30 frames, verified decode via `vpxdec --summary --noblit`) | Profile-1 4:2:0 8-bit non-default. Exercises Profile-1 parser path. |

Encoding policy for C/D/E uses `encode-corpus.ps1` with default `--passes=1 --lag-in-frames=0 --auto-alt-ref=0` (already the script default — `encode-corpus.ps1:38`) and the script adds `--codec=vp9 --ivf` at `encode-corpus.ps1:58`. To exercise multi-tile decode at 1080p we add `--tile-columns=2` to `-ExtraArgs` on D/E only. C stays single-tile (matches a single-core real-time encode profile).

Encoded fixtures are gitignored binaries (consistent with `tests/conformance/vectors/` policy — fetched not committed); only the encode-recipe and the resulting ledger are committed.

## 3. Measurement protocol

**rvp9dec side — reuse existing library.** All per-frame stats come from `rvp9_bench::throughput::run_one` (`crates/rvp9-bench/src/throughput.rs:196`). Properties we inherit:
- Decoder constructed **once per fixture**, reused across replay loops (`throughput.rs:217-221`).
- File I/O and container walk excluded; only `Decoder::decode_frame` is timed (`throughput.rs:239-241`).
- 30 warm-up frames dropped (`throughput.rs:56`), looped until ≥1000 timed frames (`throughput.rs:62`).
- Reductions: median / p95 / p99 in microseconds (`throughput.rs:285-294`).

We extend the library with `run_one_threaded(path, threads, warmup, min_timed)` that calls `Decoder::with_threads(NonZeroUsize::new(threads).unwrap())` (`crates/rvp9-decoder/src/lib.rs:639`) instead of `Decoder::new()`. No change to the timing window or stats math.

**vpxdec side — subprocess wall-time, no calibration.** (Lead revision per §11 below.) vpxdec emits no per-frame timing surface (verified: `--framestats` is bytes/qp only). The `--summary` line prints `<N decoded>/<N showed> frames in <us> us (<fps>)`. Protocol:

1. Pre-load the .ivf into the OS page cache by reading it once.
2. Invocation: `vpxdec --noblit --summary -t <N> <fixture>` for each `(fixture, threads)`. `--noblit` removes blit cost and matches our rvp9 timing window.
3. Repeat **R = 9 times** (odd; robust to one scheduler outlier). Compute `vpxdec_mean_us = median(wall_times) / decoded_frames`.
4. **No calibration constant.** On long fixtures (C/D/E) construction overhead is <1% of total decode and falls below host jitter; on short fixtures (A, B, F) it's noted in the ledger header that vpxdec mean for fixtures <100 frames includes setup smear and the rvp9 column is the authoritative number, with vpxdec serving as a sanity check.

**Reductions:**
- rvp9 columns: **p50 + mean** (mean for cross-comparison with vpxdec; p50 for tail-stability inspection).
- vpxdec column: **mean** (whole-run / frame-count).
- Comparison column: `rvp9_mean / vpxdec_mean` ratio (mean-to-mean, apples to apples).

## 4. Comparison output (mock-up)

`bench/results/decode-headtohead.md` (committed, regenerated by a script):

```markdown
# decode head-to-head — 2026-05-23T14:22:01Z

host: windows-x86_64  unpinned  build: release
rvp9: <git sha>  vpxdec: libvpx 1.14.1 (build/libvpx/vpxdec.exe)
note: vpxdec value is per-frame mean from `vpxdec --noblit --summary -t N`,
      including construction/teardown overhead (negligible on ≥1000-frame
      fixtures; up to a few percent on short fixtures — see methodology).
      rvp9 mean shown alongside p50 for tail-stability inspection;
      ratio compares means.

| fixture            | res     | tiles | threads | rvp9 p50 us | rvp9 mean us | vpxdec mean us | ratio (rvp9/vpx) | budget us | gate    |
|--------------------|---------|-------|---------|-------------|--------------|----------------|------------------|-----------|---------|
| quantizer-00       |  ?x?    |  1    |    1    |       142.1 |        148.0 |          120.4 |             1.23 |         - | n/a     |
| tile_1x4           |  ?x?    |  4    |    1    |       205.7 |        211.2 |          190.5 |             1.11 |         - | n/a     |
| tile_1x4           |  ?x?    |  4    |    4    |       110.2 |        115.8 |           62.1 |             1.86 |         - | n/a     |
| testsrc_720p30     | 1280x720|  1    |    1    |      14200.0|      14800.0 |        11200.0 |             1.32 |     33000 | PASS-info |
| testsrc_1080p30    |1920x1080|  2    |    4    |      28900.0|      29900.0 |        19400.0 |             1.54 |     33000 | PASS-info |
| testsrc_1080p60    |1920x1080|  2    |    8    |      15800.0|      16100.0 |        10900.0 |             1.48 |     16000 | PASS-info |
| foreman-fp1        |  ?x?    |  1    |    1    |       310.1 |        318.5 |          265.3 |             1.20 |         - | n/a     |
```

`gate` is `PASS-info` / `FAIL-info` (rvp9 mean vs PLAN budget) until a pinned host promotes it to `PASS` / `FAIL`. `-` for fixtures with no PLAN budget.

A parallel CSV (`bench/results/decode-headtohead.csv`) carries the same rows in machine-readable form using the existing schema-v1 conventions from `throughput.rs:392-417` extended with `vpxdec_mean_us, ratio, budget_us, gate`.

## 5. Thread axis

Per fixture, we sweep a **fixed minimal set** rather than the full `{1,2,4,8}` cross-product:
- A, F (single-tile): `threads=1` only.
- B (tile_1x4): `threads ∈ {1, 4}` (saturates at 4 tiles).
- C (720p30): `threads=1` (PLAN §2 says single-core).
- D (1080p30): `threads ∈ {2, 4}` (PLAN §2 says 2–4 cores).
- E (1080p60): `threads ∈ {4, 8}` (PLAN §2 says 4–8 cores).

The chosen set is 9 rows total: one each for A, C, and F, plus two thread settings each for B, D, and E.

**Independence from Q.1.B:** the bench uses the library `Decoder::with_threads(N)` directly, not the CLI subprocess. So Q.1.A is **not blocked on Q.1.B**.

## 6. Repeatability

Same posture as the existing soft ratchet (`decode_throughput_ratchet.rs:38-45`):
- Dev-box runs append rows with `unpinned=true` and the gate column is `PASS-info` / `FAIL-info`. No panic.
- A future pinned Linux runner flips those to `PASS` / `FAIL` with panic-on-FAIL semantics. **This HLD does not introduce that runner.**

## 7. Artifacts

Committed:
- `bench/results/decode-headtohead.md` — the ledger. Latest run only.
- `bench/results/decode-headtohead.csv` — append-only schema-v1 trend.
- `scripts/bench-headtohead.ps1` — orchestration script.

Gitignored:
- `tests/corpus/fixtures/testsrc_720p30.ivf`, `testsrc_1080p30.ivf`, `testsrc_1080p60.ivf` — regenerated locally.

## 8. Plan-of-record vs deferrals

**In scope (Q.1.A delivery):**
- New bench binary `crates/rvp9-bench/src/bin/decode-headtohead.rs` that drives §5's matrix on the rvp9 side via the library.
- New PowerShell orchestrator `scripts/bench-headtohead.ps1` that (a) ensures C/D/E exist, (b) runs the rvp9 bench binary, (c) runs the vpxdec subprocess matrix, (d) joins the two result sets, (e) writes ledger + CSV.
- `crates/rvp9-bench/src/throughput.rs` adds `run_one_threaded`.
- Journal entry under `wrk_journals/`.

**Deferred (explicit non-goals):**
- Hard CI gate on PLAN §2 budgets. Awaits pinned runner.
- Frame-parallel decode performance (Q.1.C territory).
- Linux equivalent of the orchestrator.

## 9. Implementation surface

Reused, no changes:
- `crates/rvp9-bench/src/throughput.rs` — `run_one`, `Stats`, `host_fingerprint`, `build_profile_hash`, `append_csv`.
- `crates/rvp9-decoder/src/lib.rs:639` (`Decoder::with_threads`).
- `scripts/encode-corpus.ps1`.
- `build/libvpx/vpxdec.exe`.

New, additive:
- `crates/rvp9-bench/src/throughput.rs` — add `run_one_threaded` (~30 lines).
- `crates/rvp9-bench/src/bin/decode-headtohead.rs` — new binary, ~150 lines.
- `scripts/bench-headtohead.ps1` — orchestrator, ~120 lines.

## 10. Open questions for Arthur

1. **Tile config for C/D/E.** Single-tile for C, `--tile-columns=2` for D/E. Acceptable?
2. **Encoded-fixture commit policy.** Confirm leaving C/D/E .ivf gitignored.
3. **Ratchet integration.** Recommend NO for this phase — head-to-head is artifact, not regression target.

## 11. Lead revisions on sign-off

The HLD was signed off with these mandatory revisions vs the agent's draft:
1. **Drop the per-host calibration constant** for vpxdec. On long fixtures the construction overhead is <1% of total; on short fixtures it's a known caveat we disclose in the ledger header. Simpler and less brittle.
2. **Report both rvp9 p50 AND rvp9 mean.** Ratio uses mean-to-mean only (apples-to-apples with vpxdec which has no p50 access).
3. **Q.1.A is not blocked on Q.1.B** — explicit. They run in parallel.

### Critical Files for Implementation

- `D:\language\rvp9-2\crates\rvp9-bench\src\throughput.rs`
- `D:\language\rvp9-2\crates\rvp9-bench\src\bin\decode-throughput.rs`
- `D:\language\rvp9-2\crates\rvp9-decoder\src\lib.rs`
- `D:\language\rvp9-2\scripts\encode-corpus.ps1`
- `D:\language\rvp9-2\crates\rvp9-bench\tests\decode_throughput_ratchet.rs`
