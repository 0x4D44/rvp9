# RVP9-BUG-FLU-00037 — AArch64 dependent-library Miri builds cannot resolve scalar transform helpers

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** core/transforms-miri
- **Raised:** 2026-10-15T22:32:06Z
- **Discovery source:** Agent
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
- **State history:** Open (2026-10-15T22:32:06Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium)

## Observation

At candidate 1a84f60a24e8d82e8e6454743fbb80521280df0e on aarch64-apple-darwin with installed nightly rustc bd08c9e71 (2026-06-25), `CARGO_TARGET_DIR="$PWD/.scratch/miri-target" RVP9_FORCE_SCALAR=1 MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --locked -p rvp9-core -p rvp9-io -p rvp9-decoder -p rvp9-encoder --lib -- --test-threads=1` exits 101 before dependent-library Miri tests. Observation: eight E0425 errors for fdct4x4_scalar, fht4x4_scalar, fht8x8_scalar, fht16x16_scalar, fdct32x32_scalar and idct16x16_add_scalar in transforms/forward.rs and inverse.rs. A second narrower `cargo +nightly miri test --locked -p rvp9-io --lib -- --test-threads=1` reproduces the same eight errors. Expected: the non-test rvp9-core library compiles for cfg(miri), allowing dependent encoder/decoder/io library safety checks. Confirmed in source: several scalar definitions have cfg(any(not(target_arch="aarch64"), test)), while their dispatchers call them under cfg(any(not(target_arch="aarch64"), miri)). A core-only test target can compile because cfg(test) masks this ordinary-library gap; it does not settle the dependent-library failure. Raw logs retained in final-release-readiness evidence. No source changes; fixing remains with the separate bug owner.

## Fix

<unfixed — raised only>

## Notes
