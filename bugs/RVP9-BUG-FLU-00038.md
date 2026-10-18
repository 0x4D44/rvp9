# RVP9-BUG-FLU-00038 — Decoder and encoder archives omit development dependencies required by included tests

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** packaging/archive-tests
- **Raised:** 2026-10-15T22:37:30Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261018T191216Z-cbe17809
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00038-run-fix-20261018T191216Z-cbe17809
- **Owner base:** 61006435807ff4551f0f065aeb5cf716b5d7e39c
- **Owner fingerprint:** -
- **Owner since:** 2026-10-18T19:12:16Z
- **Owner until:** 2026-10-18T23:12:16Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-10-15T22:37:30Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium)

## Observation

At candidate code 1a84f60a24e8d82e8e6454743fbb80521280df0e (documentation/ledger-only packaging revision 3dd780ccfb04358dcbb1dab0bfe766041e071cd1), all five `cargo package --locked -p <crate>` assembly/production-verification checks succeed. However unchanged extracted decoder and encoder archives fail included integration tests outside the workspace. Reproduced twice each under registered external scratch with no manifest edits: decoder `cargo test --locked --manifest-path <archive>/Cargo.toml --test panic_safety --jobs 4` exits 101 with E0432 unresolved import rvp9_encoder at tests/panic_safety.rs:29:5; encoder `cargo test --locked --manifest-path <archive>/Cargo.toml --test quality_gate --jobs 4` exits 101 with E0432/E0433 unresolved rvp9_decoder and rvp9_harness imports in quality_gate.rs and common/stage4a_product.rs. Normalized packaged manifests omit path-only development dependencies, while their dependent tests remain included. Expected: included tarball tests can resolve their declared dependencies and run, or the packaging contract clearly supplies the required test environment without source edits. These are archive-test failures, not a demonstrated production-library build failure. Do not substitute Cargo package verification builds for the requested archive test evidence. Hashes and both failing logs are retained in final-release-readiness evidence. No fix performed; separate bug agent owns fixes.

## Fix

<unfixed — raised only>

## Notes
