# RVP9-BUG-FLUX-00001 — AArch64 all-targets gate writes an x86-only avx2 capability field

- **State:** Closed
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
- **Attempts:** fix=1, doubt=0, indeterminate=0
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) → Fixed (2026-07-30, `ecb6d88`, fixed by OpenAI Codex) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

Observed on aarch64-apple-darwin with Rust 1.95.0: cargo clippy --workspace --all-targets fails with E0609 at crates/rvp9-core/src/dsp/dispatch.rs:3714 because the test assigns caps_s.avx2 on AArch64Caps, whose only field is neon. Expected: the documented all-targets clippy gate compiles on a supported AArch64 development host. Actual: the gate cannot compile rvp9-core tests, which also blocks Deltic integration.

## Fix

Commit `ecb6d88` restricts `forced_scalar_caps_installs_scalar_directional` to
`x86_64`, the only target whose capability struct and directional SIMD slots
the test exercises. `cargo clippy --workspace --all-targets --locked` passes on
`aarch64-apple-darwin`; the workspace reports only its two pre-existing,
non-denied `rvp9-bench` dead-code warnings.

## Notes

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Original observation reproduced and cleared. `cargo check -p rvp9-core --all-targets
--target aarch64-unknown-linux-gnu`: with the fix's `#[cfg(target_arch = "x86_64")]`
temporarily removed (scratch edit, reverted, never committed) it fails with exactly the
recorded error -- `error[E0609]: no field 'avx2' on type 'caps::AArch64Caps'` at
`crates/rvp9-core/src/dsp/dispatch.rs:3714`; with the fix in place it compiles clean.
That is a true fails-before / passes-after on the reported target.
`forced_scalar_caps_installs_scalar_directional` still runs and passes on x86_64, so the
gating did not silence the test on its own arch, and a sweep of the module found no other
ungated cap-mutating test.
Residual noted, not blocking: AArch64 now has no coverage of forced-scalar directional
dispatch; no NEON equivalent was added.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
