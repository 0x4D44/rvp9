# RVP9-BUG-FLU-00038 — Decoder and encoder archives omit development dependencies required by included tests

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** packaging/archive-tests
- **Raised:** 2026-10-15T22:37:30Z
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
- **State history:** Open (2026-10-15T22:37:30Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium) -> Fixed (2026-10-18T19:32:35Z, deltic:auto role=fix run=fix-20261018T191216Z-cbe17809 branch=task/bug-RVP9-BUG-FLU-00038-run-fix-20261018T191216Z-cbe17809 code=1322be6091e42b7789c34443cfbdbb2364f99386 gate=manual) -> Closed (2026-10-18T21:49:16Z, deltic:manual role=verify run=verify-20261018T214313Z-eab04ebd verified_fix_run=fix-20261018T191216Z-cbe17809 verdict=close)

## Observation

At candidate code 1a84f60a24e8d82e8e6454743fbb80521280df0e (documentation/ledger-only packaging revision 3dd780ccfb04358dcbb1dab0bfe766041e071cd1), all five `cargo package --locked -p <crate>` assembly/production-verification checks succeed. However unchanged extracted decoder and encoder archives fail included integration tests outside the workspace. Reproduced twice each under registered external scratch with no manifest edits: decoder `cargo test --locked --manifest-path <archive>/Cargo.toml --test panic_safety --jobs 4` exits 101 with E0432 unresolved import rvp9_encoder at tests/panic_safety.rs:29:5; encoder `cargo test --locked --manifest-path <archive>/Cargo.toml --test quality_gate --jobs 4` exits 101 with E0432/E0433 unresolved rvp9_decoder and rvp9_harness imports in quality_gate.rs and common/stage4a_product.rs. Normalized packaged manifests omit path-only development dependencies, while their dependent tests remain included. Expected: included tarball tests can resolve their declared dependencies and run, or the packaging contract clearly supplies the required test environment without source edits. These are archive-test failures, not a demonstrated production-library build failure. Do not substitute Cargo package verification builds for the requested archive test evidence. Hashes and both failing logs are retained in final-release-readiness evidence. No fix performed; separate bug agent owns fixes.

## Fix

The decoder and encoder Cargo package manifests exclude their workspace-only integration tests; the encoder archive also excludes the harness-based `sad_simd_oracle` example. The encoder's existing `rvp9-decoder` dev-dependency now includes a compatible registry version as well as its workspace path, so extracted archive unit tests can resolve it. The crate READMEs direct maintainers to run integration tests from a workspace checkout.

Validated both `cargo package --locked --offline` commands and `cargo test --locked` from each extracted archive. Decoder archive tests passed (224 unit tests and one compile-fail doctest); encoder archive tests passed (824 unit tests, one ignored). The excluded `panic_safety` and `quality_gate` integration targets still compile in the workspace. Fix landed in `1322be6091e42b7789c34443cfbdbb2364f99386`.

### Verification summary (2026-10-18T21:49:16Z, deltic:auto run=verify-20261018T214313Z-eab04ebd verified_fix_run=fix-20261018T191216Z-cbe17809 verdict=close)

Verifier note: Manual verify: Independent verification at 5b4b3f62: both cargo package --locked --offline assembly/verification checks pass. Unchanged extracted archives tested externally offline: decoder 224 unit tests plus 1 compile-fail doctest passed (3 ignored); encoder 824 passed (1 ignored). Inspected actual archive exclusions and versioned decoder dev dependency. Four manifest predicates fail against pre-fix bytes and pass fixed; this is content proof, not a claim of pre-fix test assertion execution. README now explicitly limits integration tests to workspace; source panic_safety and quality_gate tar...

## Notes
