# HLD — R.1: Eliminate per-block `env::var` syscalls from the decoder hot path

**Author:** Lead (Claude), bypassing investigation agent (hit session
limit before producing a doc). Diagnosis substantiated by direct grep
inspection of the decoder.
**Date:** 2026-05-23
**Status:** SIGNED OFF by lead (self) for direct implementation dispatch.

## 1. One-line summary

Cache the 43 `std::env::var("RVP9_*")` calls scattered through the
decoder behind a single process-wide `OnceLock<TraceFlags>` so that
runtime trace gating becomes a single atomic-load + struct-field access
instead of a `GetEnvironmentVariable` syscall on every block / transform
/ per-plane iteration.

## 2. Diagnosis

Grep inventory (excluding tests):

| File | Calls | Hot-path? |
|---|---|---|
| `crates/rvp9-decoder/src/tile.rs` | 12 | **Yes — multiple are inside `decode_block` (per mi-block), `decode_inter_block` (per mi-block), and the per-transform loop in `reconstruct_intra_planes` (line 1089).** |
| `crates/rvp9-decoder/src/lib.rs` | 9 | Mostly per-frame; not per-block. Lower priority. |
| `crates/rvp9-decoder/src/detokenize.rs` | 6 | **Yes — line 173 is inside `decode_block_tokens` (per tx-block).** |
| `crates/rvp9-decoder/src/reconstruct.rs` | 4 | **Yes — per-block.** |
| `crates/rvp9-decoder/src/reconstruct_inter.rs` | 3 | **Yes — `:402-403` are per-inter-block MC.** |
| `crates/rvp9-core/src/intra_pred.rs` | 1 | **Possibly per-tx inside intra pred kernel.** |
| `crates/rvp9-core/src/entropy/detokenize.rs` | 2 | **Possibly per-token.** |
| Others (encoder paths) | 6 | Not decoder hot path — out of scope for R.1. |

**Worst single site** (`tile.rs:1089`): inside the per-transform inner
loop of `reconstruct_intra_planes`. For a 1080p keyframe this fires:

- 3 planes × ~(W/4 × H/4) tx-block iterations ≈ **30,000–100,000 calls
  per frame** for `RVP9_TRACE_INTRA_AT` alone.

Windows `GetEnvironmentVariableW` (which `std::env::var` calls through)
involves a process-environment lock + UTF-16 conversion + string scan.
Even with a missing key (the common case), it's typically 0.5–5 µs
per call. **At 100k calls/frame × 1 µs = 100 ms of overhead per
keyframe** purely from "is the trace flag set?" checks.

Measured baseline (Q.1.A ledger, 2026-05-23):
- testsrc_720p30 single-thread p50: **87 ms/frame** vs vpxdec 0.63 ms
  (133× gap).
- tile_1x4 1080p single-thread p50: **271 ms/frame** vs vpxdec 3.96 ms
  (62× gap).

If this fix moves the 720p number from 87 ms → ~10–20 ms, we close
roughly an order of magnitude of the gap in one wave. Even if the
effect is smaller, the diff is mechanical and worth it.

## 3. Design

### 3.1 New module: `crates/rvp9-decoder/src/trace_flags.rs`

```rust
//! Process-wide cache of RVP9_TRACE_* / RVP9_DEBUG_* env vars.
//!
//! `env::var` on Windows is a syscall (GetEnvironmentVariableW). The
//! per-block tracing infrastructure that survives in this decoder is
//! debug-only; toggling it once at process start is plenty. We snap
//! the env once, on first access, and serve every subsequent check
//! from a struct field (single atomic load).

pub struct TraceFlags {
    pub kfsb: bool,
    pub kfpartition: bool,
    pub kfleaf: bool,
    pub kf_dq: bool,
    pub debug_boolcursor: bool,
    pub debug_dqcoefs: Option<String>,         // "1", "tx8", "2", "all"
    pub debug_dqcoefs_at: Option<String>,      // filter spec
    pub debug_buffer_dump: Option<String>,     // path
    pub intra_at: Option<String>,              // filter spec
    pub intermi: bool,
    pub frame_filter: Option<u32>,             // RVP9_TRACE_FRAME
    pub mc: bool,
    pub mc_filter: Option<String>,
    pub mc_src: bool,
    pub seg_id: bool,
    pub partition_inter: bool,
    pub rdmv: bool,
    // ... full inventory from the grep
}

impl TraceFlags {
    fn read_env() -> Self { /* one env::var per field, once */ }
}

static TRACE_FLAGS: std::sync::OnceLock<TraceFlags> = std::sync::OnceLock::new();

#[inline]
pub fn trace_flags() -> &'static TraceFlags {
    TRACE_FLAGS.get_or_init(TraceFlags::read_env)
}
```

### 3.2 Call-site transformation

Every existing hot-path call:

```rust
// BEFORE
if std::env::var("RVP9_TRACE_KFLEAF").is_ok_and(|v| v == "1") { ... }
```

becomes:

```rust
// AFTER
if crate::trace_flags::trace_flags().kfleaf { ... }
```

For filter strings (e.g. `RVP9_TRACE_INTRA_AT`):

```rust
// BEFORE
if let Ok(filter) = std::env::var("RVP9_TRACE_INTRA_AT") { ... }

// AFTER
if let Some(filter) = trace_flags().intra_at.as_deref() { ... }
```

For sites that combine multiple env reads (e.g. `tile.rs:1777-1784`),
all reads go to the cached struct in one statement.

### 3.3 Bit-exactness invariant

The cache reads env ONCE at first access. Every conformance/triangulated
test that exercises these flags either:

- Doesn't set them (the common case — conformance, profile1_decode,
  frame_parallel_regression). The cache is all-defaults → all-false →
  identical to the pre-fix behavior of "env::var returns Err".
- Sets them via the test harness's `std::env::set_var(...)` in a
  `#[test]` BEFORE constructing the Decoder. The cache lazy-inits on
  the first `trace_flags()` call, which happens during decode (after
  the test's set_var). Identical to pre-fix.

The only mode that breaks is "set env::var MID-decode" — which no
test does (and would be racy anyway). Documented as a deliberate
divergence in the new module's doc comment.

### 3.4 Out of scope

- The 9 `lib.rs` calls are mostly per-frame setup (e.g. `decode_frame`
  prologue). They could also be cached but the impact is 30 calls/sec,
  not 100k. Cache them as a follow-up if R.1's bench post-mortem
  shows they're worth chasing.
- The 6 encoder-side `mode_decision/*` / `mv_search/*` calls are out of
  scope (Track F territory).

## 4. Test plan

1. **Conformance** (`cargo test -p rvp9-harness --test conformance --release`):
   201/201 must stay green. Bit-exactness gate.
2. **Profile-1 triangulated** (`cargo test -p rvp9-harness --test profile1_decode --release`):
   All 6 vectors green.
3. **Frame-parallel regression**
   (`cargo test -p rvp9-harness --test frame_parallel_regression --release`):
   30/30 green.
4. **Tile-threading** (`cargo test -p rvp9-harness --test tile_threading --release`):
   All 7 tests green.
5. **Re-run Q.1.A bench** (`scripts/bench-headtohead.ps1`) on the
   `testsrc_720p30` and `tile_1x4` rows. Append to
   `bench/results/decode-headtohead.csv` and rewrite the .md.

## 5. Rollout

Single commit. No phasing. Mechanical change, ~50-60 line edits across
6 files + ~80 LOC new module.

## 6. Risks

- **Stale cache.** If a test sets env after the decoder constructs, no
  trace fires. Acceptable — no test does this in the workspace today
  (grepped). Document it.
- **Unsafe?** No. `OnceLock` is safe + lock-free after first init.
- **Sync impact?** `OnceLock` provides `Sync`; the `&'static TraceFlags`
  return crosses thread boundaries trivially.
- **Could this fix not move the needle?** Possible — if Windows
  `GetEnvironmentVariableW` is cheaper than I think (~50 ns) then the
  43 calls/block × 8000 blocks × 30 fps ≈ ~10% perf only. Even at the
  low end this is worth shipping; at the high end it could be the
  dominant fix.
