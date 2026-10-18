# RVP9-BUG-FLU-00037 — AArch64 dependent-library Miri builds cannot resolve scalar transform helpers

- **State:** Fixed
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
- **State history:** Open (2026-10-15T22:32:06Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium) -> Fixed (2026-10-18T19:54:31Z, deltic:auto role=fix run=fix-20261018T195040Z-e3bd6992 branch=task/bug-RVP9-BUG-FLU-00037-run-fix-20261018T195040Z-e3bd6992 code=019f28983fa563954233d1b99892578d4c123657 gate=manual)

## Observation

At candidate 1a84f60a24e8d82e8e6454743fbb80521280df0e on aarch64-apple-darwin with installed nightly rustc bd08c9e71 (2026-06-25), `CARGO_TARGET_DIR="$PWD/.scratch/miri-target" RVP9_FORCE_SCALAR=1 MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --locked -p rvp9-core -p rvp9-io -p rvp9-decoder -p rvp9-encoder --lib -- --test-threads=1` exits 101 before dependent-library Miri tests. Observation: eight E0425 errors for fdct4x4_scalar, fht4x4_scalar, fht8x8_scalar, fht16x16_scalar, fdct32x32_scalar and idct16x16_add_scalar in transforms/forward.rs and inverse.rs. A second narrower `cargo +nightly miri test --locked -p rvp9-io --lib -- --test-threads=1` reproduces the same eight errors. Expected: the non-test rvp9-core library compiles for cfg(miri), allowing dependent encoder/decoder/io library safety checks. Confirmed in source: several scalar definitions have cfg(any(not(target_arch="aarch64"), test)), while their dispatchers call them under cfg(any(not(target_arch="aarch64"), miri)). A core-only test target can compile because cfg(test) masks this ordinary-library gap; it does not settle the dependent-library failure. Raw logs retained in final-release-readiness evidence. No source changes; fixing remains with the separate bug owner.

On 2026-10-18, the exact product source `fb5c0c917038db6c640b3dbc7c2b84b567233333` reproduced the build failure with `nice -n 15 deltic timeout 7200 env CARGO_TARGET_DIR=<registered-deltic-scratch>/target CARGO_BUILD_JOBS=1 MIRIFLAGS=-Zmiri-disable-isolation cargo +nightly miri test --locked --offline --package rvp9-core --test frame_alloc -- --test-threads=1 </dev/null` on `aarch64-apple-darwin`, rustc `bd08c9e71` (2026-06-25). The command exited 101 before the integration-test binary started, reporting the same eight E0425 missing-helper errors; none of its three tests ran. Expected: the core dependency compiles and the target starts its three tests. [Exact-source command, output, source identity and hashes](../wrk_docs/evidence/2026-10-15-release-readiness/recovered-validation-20261017/postfix-main-qualification/mac-current-source-miri-20261018-frame-alloc-integration/README.md).

## Fix

<unfixed — raised only>

## Notes

On 2026-10-17, exact-candidate Miri attempts on Apple Silicon reproduced the
same compile-time blocker in dependent crates. The locked, offline
`rvp9-io` parser suite exited 101 while compiling `rvp9-core`; all eight E0425
missing-scalar-helper diagnostics appeared before the test harness started, so
no parser or container tests ran. The `rvp9-decoder` `panic_safety` target also
exited 101 before its four tests ran with the same missing-helper errors. The
tested product source was `fb5c0c917038db6c640b3dbc7c2b84b567233333`; the
`rvp9-io` runner checkout's relevant source paths were verified identical to
that candidate. These attempts reproduce this configuration defect and add no
parser or decoder runtime-safety result. [IO parser run identity and outcome](../wrk_docs/evidence/2026-10-15-release-readiness/recovered-validation-20261017/postfix-main-qualification/mac-current-source-miri-20261017/io-parser-miri-attempt-20261017/qualification.json), [decoder panic-safety run identity and outcome](../wrk_docs/evidence/2026-10-15-release-readiness/recovered-validation-20261017/postfix-main-qualification/mac-current-source-miri-decoder-panic-safety-20261017/qualification.json).
