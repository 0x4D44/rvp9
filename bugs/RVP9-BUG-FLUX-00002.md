# RVP9-BUG-FLUX-00002 — Captured-vector tests fail clean worktrees when optional conformance data is absent

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** core/bitstream tests
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) → Fixed (2026-07-30, `ecb6d88`, fixed by OpenAI Codex) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5); residual split to RVP9-BUG-KILN-00063)

## Observation

Observed in a fresh detached worktree at 27f79a0 on aarch64-apple-darwin with Rust 1.95.0: cargo test --workspace --locked runs 1,508 rvp9-core tests successfully, then bitstream::header::tests::captured_header_round_trip and bitstream::compressed_header::tests::write_compressed_header_captured_frame_round_trip panic because tests/conformance/vectors/vp90-2-05-resize.ivf is gitignored and absent. Expected: the documented workspace gate passes in a clean worktree while these optional real-world round-trips still run when fetched vectors are present. Actual: both tests require machine-local data and make every clean Deltic task worktree red.

## Fix

Commit `ecb6d88` makes both captured-frame tests return only when the optional
vector is absent. Any other read error still fails, and all parsing and
round-trip assertions remain unchanged when data is present. Both focused tests
pass without the vector and pass again with the official
`vp90-2-05-resize.ivf`; `cargo test --workspace --locked` passes in the clean
checkout state.

## Notes

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

The reported symptom is gone: `captured_header_round_trip` and
`write_compressed_header_captured_frame_round_trip` both pass in a clean worktree, and only
`ErrorKind::NotFound` returns early -- every other read error still panics and all assertions
remain live when the vector is present (verified with vectors fetched: both still pass).
Residual split to RVP9-BUG-KILN-00063: the chosen remedy is a SILENT skip, so both tests
report green having asserted nothing when vectors are absent. That is the pattern
RVP9-BUG-KILN-00019 was raised to eliminate, and the harness crate solved it visibly with a
`build.rs`-driven ignore. The original defect is fixed; the inconsistency is now tracked.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
