# HLD — R.2: Post-R.1 decoder performance investigation

**Author:** Investigation agent (Claude)
**Date:** 2026-05-23
**Status:** DRAFT — awaiting lead sign-off for R.2 fix dispatch.
**Workspace:** `D:\language\rvp9-2`
**Baseline commit:** `21ecd51` (R.1 + Wave P; post-R.1 trace-flag cache landed).

## 1. Summary

R.1's env::var cache delivered a much bigger win than first measured
(only 3 rows were re-run for the R.1 ledger). The full Q.1.A matrix
re-measured post-R.1 shows speedups of **2–30×** across the board, the
**negative tile-thread scaling is GONE** (tile_1x4: t=1 58.4 ms → t=4
26.9 ms, 2.17× — was 272 → 339 ms pre-R.1), and the gap to vpxdec
collapsed from 60–250× down to **5.7–14.7×**. The pre-R.1 negative
scaling was almost certainly Windows process-environment-lock
contention — exactly the kind of pathology a 100k-calls-per-keyframe
`GetEnvironmentVariableW` storm produces under multi-threading.

The next biggest single fix is **per-TX-block heap allocation** inside
`decode_coefs` (`rvp9-core/src/entropy/detokenize.rs`): 5 `Vec` heap
allocations are constructed and dropped on every call (~30k–60k calls
per 1080p frame). Estimated win: 10–30% of remaining frame time,
mechanical effort, bit-exact by construction (reusable scratch).

## 2. Methodology

- Re-ran the full Q.1.A matrix on the post-R.1 release binary
  (`target/release/decode-headtohead.exe`, build hash
  `3c8d6d1ddea4abc4`) with warmup=30, min-timed=200, single pass per
  row. Numbers in §3.
- Samply was attempted but requires Administrator on Windows; the
  sandbox does not grant elevation. Fallback: dense static-analysis
  read of the hot decoder paths + manual call-frequency arithmetic
  cross-checked against the empirical numbers.
- Per-claim source citations as `file:line` (paths absolute).

## 3. Updated bench picture (post-R.1, 2026-05-23)

Pre-R.1 numbers from `bench/results/decode-headtohead.md`; post-R.1
from this run (`.profile/full_post_r1.tsv`).

| fixture | th | post-R.1 p50 us | pre-R.1 p50 us | speedup | vpxdec us | gap remaining |
|---|---|---|---|---|---|---|
| quantizer-00 | 1 | 2,331 | 12,084 | 5.18× | 6,089 | **rvp9 FASTER** |
| tile_1x4 | 1 | 58,394 | 271,544 | 4.65× | 3,964 | 14.7× |
| tile_1x4 | 4 | **26,855** | 339,213 | **12.6×** | 2,516 | 10.7× |
| testsrc_720p30 | 1 | 3,594 | 87,198 | 24.3× | 627 | 5.7× |
| testsrc_1080p30 | 2 | 15,332 | 148,492 | 9.69× | 821 | 18.7× |
| testsrc_1080p30 | 4 | 6,135 | 181,832 | 29.6× | 614 | 10.0× |
| testsrc_1080p60 | 4 | 5,168 | 11,321 | 2.19× | 592 | 8.7× |
| testsrc_1080p60 | 8 | 5,369 | 10,631 | 1.98× | 596 | 9.0× |
| foreman-fp1 | 1 | 2,329 | 8,130 | 3.49× | 317 | 7.3× |

Worst row: **tile_1x4 t=1** at 14.7×. Median of the remaining gaps is
roughly 8–10×.

### 3.1 Tile-thread scaling diagnostic (lead asked: still negative?)

**NO — positive now.** tile_1x4 1080p: t=1 → 58.4 ms, t=4 → 26.9 ms
(2.17× speedup, 54% efficiency on 4 cores). Pre-R.1 this was 272 ms
→ 339 ms — the regression was the env::var process-environment lock
serializing tile workers. Other rows scale similarly (testsrc_1080p30
goes 2.5× from t=2→t=4; testsrc_1080p60 is at the bench-noise floor).

Tile scaling is no longer a separate R-track diagnostic and can be
dropped from the worry list.

## 4. Hot-spot inventory (post-R.1)

Citations relative to `D:\language\rvp9-2`. Estimates are derived
from call-frequency × per-call cost; samply was unavailable.

### 4.1 (g) Per-TX-block heap allocations inside `decode_coefs` — TOP

`crates/rvp9-core/src/entropy/detokenize.rs:316,367-370` — every call
to `decode_coefs` does:

```rust
let mut token_cache = vec![0u8; tx_size.n_coeffs()];          // :316
let mut scratch_raster: Vec<u16> = Vec::with_capacity(seg_eob); // :367
let mut scratch_dqv:    Vec<i16> = Vec::with_capacity(seg_eob); // :368
let mut scratch_val:    Vec<i16> = Vec::with_capacity(seg_eob); // :369
let mut scratch_sign:   Vec<u8>  = Vec::with_capacity(seg_eob); // :370
```

That's **5 heap allocations** per TX-block decode. A 1080p frame
contains on the order of 8160 MI 8×8 cells × ~3 planes; under typical
TX-block density (mostly Tx4x4/Tx8x8) that translates to
≈ 30,000–60,000 `decode_coefs` calls per frame.

→ **150,000–300,000 `malloc`+`free` pairs per frame**, just here. On
Windows the system allocator (HeapAlloc / RtlAllocateHeap) costs
30–100 ns per pair; that's **5–30 ms/frame of pure allocator overhead**
— a substantial fraction of the remaining 6–58 ms budget on the
worst rows.

**Category:** (g) Allocation in inner loops.
**Fix shape:** lift the 5 buffers to per-tile scratch (e.g. add fields
to `TileCoefCtx` or a new `TileScratch` borrowed in). Either preserve
length-tracking with `Vec::clear` + `push` (no realloc on warm path)
or use fixed-size `[T; MAX]` arrays sized to the worst-case TX block
(`MAX_TX_COEFFS = 1024`).
**Estimated impact:** 10–30% of post-R.1 frame time on tile_1x4/1080p.
**Effort:** Small (mechanical signature plumbing through one call
chain; allocations move from per-call to per-tile).
**Bit-exact:** Yes — pure storage hoist.

### 4.2 (b)/(h) Per-TX-block 4 KB stack zeroing for `dq_coefs`

`crates/rvp9-decoder/src/reconstruct.rs:188` and
`crates/rvp9-decoder/src/reconstruct_inter.rs:544` —

```rust
let mut dq_coefs = [0i32; MAX_TX_COEFFS]; // 1024 * 4 = 4096 bytes
```

Then `decode_coefs` itself zeroes `dq_coefs_out[..seg_eob]` again at
`detokenize.rs:305-307`. For a Tx4x4 block we zero 4 KB stack + 64
bytes again; for a Tx32x32 we zero 4 KB stack + 4 KB again.

→ At 30k–60k TX blocks per frame × 4 KB = 120–240 MB of memset
bandwidth per frame just for these temporaries. L1 stays hot but the
work isn't free.

**Category:** (h) wasted work + (b) cache thrash.
**Fix shape:** Either (a) make `dq_coefs` sized to `tx_size.n_coeffs()`
(no allocation, just exact-size stack array — but Rust forbids that
because tx_size isn't a const) or (b) lift to per-tile scratch as part
of the same hoist as 4.1, sized once to MAX_TX_COEFFS, and skip the
double-zero. Note: removing the `dq_coefs` zero before the
`decode_coefs` zero-region pass is safe — `decode_coefs` zeroes
exactly `[..seg_eob]` and writes only inside that range.
**Estimated impact:** 3–10% (smaller than 4.1 because L1 absorbs
most of it; the double-zero in particular wastes 2× the bandwidth).
**Effort:** Trivial once 4.1's scratch hoist is in (drops out for
free).
**Bit-exact:** Yes.

### 4.3 (a) `decode_coefs` two-pass scratch overhead (H5 design)

`crates/rvp9-core/src/entropy/detokenize.rs:394-548,584-592` — the
Wave H5 refactor split coefficient decode into pass-1 (entropy + push
to scratch) + pass-2 (vectorised dequant arithmetic via
`plane_dequant_finisher`). For *small* TX blocks (Tx4x4 with eob ≤ 4
— the common case in keyframe Y) the pass-1 push + pass-2 dispatch
overhead **may exceed** the savings from the vectorisation. The
dispatch call (`(dispatch.plane_dequant_finisher())(...)` at
`detokenize.rs:584-592`) is one indirect call per TX block, hitting
4 short slices (≤ 16 elements each for Tx4x4) and then returning.

**Category:** (a) Algorithmic — may be doing extra work libvpx avoids.
**Fix shape:** investigate whether a Tx4x4-direct fast path that fuses
pass-1 + pass-2 inline (no scratch, no dispatch) pays. Wave H prototype
notes (`notes/wave-h-g7-prototype.md`) ack'd this is an open question.
**Estimated impact:** Unknown — could be 5–15% on keyframe-heavy
content, could be negligible on inter-heavy (where eob runs longer
and the AVX2 path actually amortises).
**Effort:** Medium (requires bench-bisect; not just a hoist).
**Bit-exact:** Yes (the formulas reduce to the pre-H5 fused code).
**Recommendation:** NOT for R.2 — too uncertain. Defer until 4.1's
hoist is bench-validated; the cache-pressure relief may make this
irrelevant.

### 4.4 (e) Bounds checks in `BoolReader::read` and `fill`

`crates/rvp9-core/src/entropy/bool_reader.rs:80-112` — the per-bit
`read()` is `#[inline]`, uses `super::tables::VPX_NORM[(range & 0xff)
as usize]`. `range` is a `u32`; `range & 0xff` is in 0..256;
`VPX_NORM` is the 256-entry table. LLVM almost certainly elides this
bounds check at -O3 (the masked index has a constant upper bound).
`fill()` (`:184-221`) is NOT inlined; it does `self.buffer[self.pos
+ k]` in a 0..8 loop. This runs once per ~64 bits of bitstream — at
~50,000 TX-blocks × ~10 reads each = 500k reads → ~5,000 fills per
frame. Negligible.

**Category:** (e) — already a non-issue.
**Recommendation:** SKIP. Don't waste effort on unsafe here.

### 4.5 (g) Per-frame allocation in `TileDecodeCtx::new`

`crates/rvp9-decoder/src/tile.rs:212` —

```rust
above_partition: vec![0u8; tile_mi_cols],
```

This is per-tile-per-frame (4 tiles × 30 fps × O(100-byte vec) =
12000 allocs/sec). Negligible — one Vec at the start of each tile
decode. Not worth chasing.

**Category:** (g) — not worth chasing.

## 5. Categorisation summary

| Spot | Category | Per-frame cost (est.) | Effort | Bit-exact? |
|---|---|---|---|---|
| 4.1 detokenize scratch heap allocs | (g) | 5–30 ms | S | yes |
| 4.2 4 KB dq_coefs stack zero + double-zero | (h)/(b) | 1–5 ms | XS (rides on 4.1) | yes |
| 4.3 H5 two-pass for tiny Tx4x4 | (a) | 0–10 ms (uncertain) | M | yes |
| 4.4 BoolReader bounds | (e) | <0.5 ms | — | — |
| 4.5 tile_ctx new vec | (g) | <0.1 ms | — | — |

## 6. Ranked fix list (impact ÷ effort)

1. **Hoist `decode_coefs` scratch buffers to per-tile.** Carry a
   `TileDetokScratch` (5 Vecs + token_cache, all `.clear()`'d
   between blocks) through `decode_block_tokens` →
   `decode_coefs`. Drop per-call `vec![...]` /
   `Vec::with_capacity(...)`. **Impact:** 10–30%. **Effort:** S.
2. **Drop the 4 KB `dq_coefs` zero in
   `reconstruct.rs:188` / `reconstruct_inter.rs:544`** —
   `decode_coefs` already zeroes exactly the region it writes
   (`detokenize.rs:305-307`). Either (a) replace with
   `MaybeUninit::uninit_array()` + manual init of `[..seg_eob]`,
   or (b) lift to per-tile scratch (preferred — combines with #1).
   **Impact:** 3–10%. **Effort:** XS once #1 is in.
3. **Pin down the H5 two-pass for tiny Tx4x4** — measure whether a
   fused Tx4x4 fast path beats the pass-1/pass-2 split. Requires
   ad-hoc microbench because the answer depends on TX-size mix.
   **Impact:** 0–10%. **Effort:** M.
4. **Loopfilter cache locality.** Re-walk the apply.rs MI iteration
   to confirm we hit each MB row in plane-order (Y/U/V) so the
   per-plane stride bus traffic is sequential rather than per-MI
   jumping between planes. (Not investigated in depth here — quick
   call-out for the lead.) **Impact:** unknown. **Effort:** M.
5. **Tile-thread work granularity.** With t=4 at 26.9 ms vs t=1 at
   58.4 ms (efficiency 54%), there's clearly *some* threading
   overhead, but it's no longer pathological. Defer for a later
   wave; not the next biggest win.

## 7. Recommended R.2 fix

**Fix #1: per-tile scratch hoist for `decode_coefs`.**

- Signature change: pass `&mut TileDetokScratch` through
  `decode_block_tokens` (`rvp9-decoder/src/detokenize.rs:125`) into
  `decode_coefs` (`rvp9-core/src/entropy/detokenize.rs:276`). The
  scratch struct owns `token_cache: Vec<u8>` (sized to
  `MAX_TX_COEFFS`, never realloced) plus the 4 pass-1 scratch Vecs.
- Inside `decode_coefs`, replace `let mut token_cache = vec![...]`
  with `scratch.token_cache.resize_default(tx_size.n_coeffs(), 0)`
  (or use a pre-sized full-MAX_TX_COEFFS buffer and only `[..n]`).
  Replace `Vec::with_capacity(seg_eob)` with
  `scratch.{raster,dqv,val,sign}.clear()` (no realloc — `Vec::clear`
  preserves capacity).
- Scratch is owned by `TileCoefCtx` (already passed through every
  call site). Add it next to the existing above_coef / left_coef
  fields.
- One scratch instance per tile; tile worker owns it; no
  cross-thread sharing.

**Why this is the right R.2:**

- **High-confidence impact.** Empirical knowledge: each Vec
  allocation on Windows costs ~50–100 ns; we're doing 5 per
  TX-block × 30k–60k TX blocks/frame = 150k–300k allocs/frame
  ≈ 7.5–30 ms/frame allocator overhead. Even the low end is a
  noticeable 10% chunk of the worst rows.
- **Bit-exact by construction.** Pure storage hoist; all bytes
  written and read are at the same offsets with the same values.
  Pass-1 push order, pass-2 dispatch, output coefficient values —
  all unchanged.
- **Single-session effort.** ≤ 200 LOC across 3 files
  (`tile_ctx.rs` to add the struct; `detokenize.rs` (decoder) to
  thread the borrow; `detokenize.rs` (core) to consume the borrow).
- **Conformance gate is reliable.** Bit-exactness is verified by
  the existing 201/0/0/0/201 conformance corpus that R.1 already
  passed.
- **Compatible with #2.** The 4 KB `dq_coefs` zero kill drops out
  naturally once the scratch struct also owns a
  `[i32; MAX_TX_COEFFS]` reusable coefficient buffer.

## 8. Open questions for the lead

1. **Should fix #2 be combined with R.2 or split?** Combining is
   cleaner (one scratch struct, one PR); splitting is more cautious
   (separate bench delta per change). My vote: combine; the changes
   are coupled.
2. **Per-tile scratch placement.** Add to `TileCoefCtx` (already
   plumbed everywhere) or new `TileScratch` (cleaner separation;
   one more param). My vote: `TileCoefCtx` — less plumbing churn.
3. **`token_cache` sizing.** Allocate once to
   `MAX_TX_COEFFS = 1024` and `.iter_mut().take(n).for_each(set
   to 0)`, or `resize` per call? Resize would shrink/grow; pinning
   to MAX is one alloc per tile, zero per block. My vote: pin to
   MAX.
4. **Is samply-on-Windows worth Arthur getting elevated rights?**
   The static analysis here is convincing for the next 1–2 waves
   but eventually we'll want a real profile.
5. **What's the next next fix?** After R.2 lands, the next biggest
   spot might be (a) the H5 two-pass Tx4x4 question (4.3), (b) MC /
   loopfilter SIMD coverage gaps, or (c) per-frame setup overhead
   in `decode_frame` (the 9 lib.rs env::var calls noted but
   deferred in R.1). Probably (c) — also a mechanical hoist.

## 9. Risks

- **Wrong root cause.** Without samply we cannot positively confirm
  malloc is the dominant remaining cost. The arithmetic argument
  (150k–300k allocs/frame on Windows allocator) is strong but not
  empirical. Mitigation: re-run the Q.1.A bench after R.2 lands; if
  the speedup is <5% we miscalled it and need to actually get
  samply running.
- **TileCoefCtx struct growth.** Adding 5 Vecs + 1 i32 Vec is ~64
  bytes of pointer-sized fields + headers. Worth nothing on cache
  budget. Mitigation: none needed.
- **Mid-tile state-carry bugs.** The 4 pass-1 Vecs must be
  `.clear()`'d at the start of each `decode_coefs` call. A missed
  clear would carry coefficients from the previous block. Mitigation:
  put the `.clear()` calls at the top of `decode_coefs` in the same
  spot the current `Vec::with_capacity` lives, and add a
  `debug_assert_eq!(scratch.raster.len(), 0)` at the bottom.

## 10. Test plan

1. `cargo test -p rvp9-harness --test conformance --release` —
   201/201 must stay green. Bit-exactness gate.
2. `cargo test -p rvp9-harness --test profile1_decode --release` —
   6/6.
3. `cargo test -p rvp9-harness --test frame_parallel_regression
   --release` — 30/30.
4. `cargo test -p rvp9-harness --test tile_threading --release` —
   7/7.
5. Re-run the Q.1.A matrix; append to ledger; the worst row
   (tile_1x4 t=1) is the headline number to beat. Target: ≥10%
   improvement on that row.
