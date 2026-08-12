# RVP9-BUG-KILN-00020 - Public AVX2 predictor wrapper exposes unsafe CPU and slice preconditions

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** core/dsp x86_64 SIMD API
- **Raised:** 2026-07-07
- **Owner:** -
- **Owner role:** -
- **Owner run:** -
- **Owner host:** -
- **Owner branch:** -
- **Owner base:** -
- **Owner fingerprint:** -
- **Owner since:** -
- **Owner until:** -
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-core/` at `af4c7a2d7533`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T171112Z-p5310-n568069000-c1 branch=task/bug-RVP9-BUG-KILN-00020-run-fix-20260730T171112Z-p5310-n568069000-c1 code=ce657a77ef476ba946f167a581c7f111e5bdb97d gate=manual) -> Open (2026-07-31, reopened by independent two-eyes verify pass on `4abe0d8`: the fix regresses the encoder's 64x64 intra fallback) -> Fixed (2026-08-12T14:43:28Z, deltic:auto role=fix run=fix-20260812T143904Z-p46812-n519221000-c1 branch=task/bug-RVP9-BUG-KILN-00020-run-fix-20260812T143904Z-p46812-n519221000-c1 code=5d694b76 gate=manual)

## Observation

`crates/rvp9-core/src/dsp/mod.rs:42` exposes `pub mod x86_64`, and `crates/rvp9-core/src/dsp/x86_64/mod.rs:169` re-exports `v_predictor_avx2`. The exported `v_predictor_avx2` wrapper is a safe public function at `crates/rvp9-core/src/dsp/x86_64/v_predictor_avx2.rs:126`, but it calls the AVX2 `#[target_feature]` implementation at `crates/rvp9-core/src/dsp/x86_64/v_predictor_avx2.rs:133`.

The inner function documents CPU, stride, length, and block-size requirements at `crates/rvp9-core/src/dsp/x86_64/v_predictor_avx2.rs:50`, but the checked conditions are only `debug_assert!`s at `crates/rvp9-core/src/dsp/x86_64/v_predictor_avx2.rs:67`. Release callers can therefore use this safe API on a non-AVX2 CPU, with undersized slices, or with invalid geometry before the raw loads and stores at `crates/rvp9-core/src/dsp/x86_64/v_predictor_avx2.rs:76`.

## Fix

Open. Make the public boundary safe: either keep AVX2 implementations private behind runtime-dispatched safe callers, mark direct target-feature entry points `unsafe` with documented caller obligations, or add release validation before raw pointer access. Add a regression that exercises the public path with invalid geometry and proves it returns an error or panics before unsafe memory access.

**Reopened 2026-07-31 — attempt 1 (`ce657a77`) regressed a live encoder path.**
The guard is correct in intent but too strict on `bs`. `validate_intra_predictor_layout`
(`crates/rvp9-core/src/dsp/mod.rs:53`) rejects any `bs` outside `{4, 8, 16, 32}`, and
`v_predictor_avx2` (`crates/rvp9-core/src/dsp/x86_64/v_predictor_avx2.rs:133`) turns that
into a hard panic. But `v_predictor_avx2_inner` deliberately carries a scalar fallback arm
for other sizes, and a real caller uses it: `evaluate_intra_fallback`
(`crates/rvp9-core/src/mode_decision/inter.rs:2370`) calls `intra_predict`
(`crates/rvp9-core/src/intra_pred.rs:762`) with `bw = 64` on `BLOCK_64X64`. On an AVX2 host
the V_PRED dispatch slot now panics mid-encode.

Any refix must keep the geometry/CPU validation but accept every `bs` the inner kernel
supports (route non-`{4,8,16,32}` sizes to the existing scalar arm rather than panicking),
and add an encoder-level regression covering `BLOCK_64X64` V_PRED intra-in-inter.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 core review.

### 2026-07-31 verify-pass evidence (independent, not the fixer)

- Repo gate `cargo test --workspace --locked` on `4abe0d8` (this worktree, AVX2 host):
  **FAILED**, 2 failures, both in `-p rvp9-encoder --test w_tpl`:
  `w_tpl_on_stream_differs_from_off` and `w_tpl_rdmult_adjusted_only_when_enabled`,
  each panicking `v_predictor_avx2: unsupported block size` at
  `crates/rvp9-core/src/dsp/x86_64/v_predictor_avx2.rs:134`.
- Backtrace path: `StreamEncoder::flush` -> `encode_inter_internal` ->
  `nonrd_pick_partition_with_pruners` -> `pick_inter_mode` -> `evaluate_intra_fallback`
  -> `intra_predict` -> `v_predictor_avx2`.
- Causation probe: with the `validate_intra_predictor_layout` panic in the wrapper
  neutralised (scratch edit, reverted, never committed), both tests pass. The panic
  introduced by `ce657a77` is the sole cause; `crates/rvp9-encoder/tests/w_tpl.rs` has
  not been touched since `1e22d91`, well before the fix.
- Because this failure is on trunk, the workspace gate is red and no other `Fixed` bug
  can be honestly closed until it is refixed.
