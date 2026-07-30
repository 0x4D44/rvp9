# RVP9-BUG-FLUX-00002 — Captured-vector tests fail clean worktrees when optional conformance data is absent

- **State:** Fixed
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
- **State history:** Open (2026-07-30, raised via `deltic bugs new`) → Fixed (2026-07-30, `ecb6d88`, fixed by OpenAI Codex)

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
