# RVP9-BUG-FLUX-00001 — AArch64 all-targets gate writes an x86-only avx2 capability field

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** core/dsp dispatch tests
- **Raised:** 2026-07-30
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`)

## Observation

Observed on aarch64-apple-darwin with Rust 1.95.0: cargo clippy --workspace --all-targets fails with E0609 at crates/rvp9-core/src/dsp/dispatch.rs:3714 because the test assigns caps_s.avx2 on AArch64Caps, whose only field is neon. Expected: the documented all-targets clippy gate compiles on a supported AArch64 development host. Actual: the gate cannot compile rvp9-core tests, which also blocks Deltic integration.

## Fix

<unfixed — raised only>

## Notes
