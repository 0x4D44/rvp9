# RVP9-BUG-KILN-00020 - Public AVX2 predictor wrapper exposes unsafe CPU and slice preconditions

- **State:** Open
- **Priority:** Should
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-core/` at `af4c7a2d7533`)

## Observation

`crates/rvp9-core/src/dsp/mod.rs:42` exposes `pub mod x86_64`, and `crates/rvp9-core/src/dsp/x86_64/mod.rs:169` re-exports `v_predictor_avx2`. The exported `v_predictor_avx2` wrapper is a safe public function at `crates/rvp9-core/src/dsp/x86_64/v_predictor_avx2.rs:126`, but it calls the AVX2 `#[target_feature]` implementation at `crates/rvp9-core/src/dsp/x86_64/v_predictor_avx2.rs:133`.

The inner function documents CPU, stride, length, and block-size requirements at `crates/rvp9-core/src/dsp/x86_64/v_predictor_avx2.rs:50`, but the checked conditions are only `debug_assert!`s at `crates/rvp9-core/src/dsp/x86_64/v_predictor_avx2.rs:67`. Release callers can therefore use this safe API on a non-AVX2 CPU, with undersized slices, or with invalid geometry before the raw loads and stores at `crates/rvp9-core/src/dsp/x86_64/v_predictor_avx2.rs:76`.

## Fix

Open. Make the public boundary safe: either keep AVX2 implementations private behind runtime-dispatched safe callers, mark direct target-feature entry points `unsafe` with documented caller obligations, or add release validation before raw pointer access. Add a regression that exercises the public path with invalid geometry and proves it returns an error or panics before unsafe memory access.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 core review.
