# RVP9-BUG-KILN-00062 — tile_threading scratch-allocation test races its sibling tests and reds the workspace gate

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** harness/tile threading tests
- **Raised:** 2026-07-31T10:21:41Z
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260812T144948Z-p77389-n527102000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00062-run-fix-20260812T144948Z-p77389-n527102000-c1
- **Owner base:** e1ea8c76d8af3572050f8ee674526026ebf9f4d9
- **Owner fingerprint:** -
- **Owner since:** 2026-08-12T14:49:48Z
- **Owner until:** 2026-08-12T16:49:48Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31T10:21:41Z, raised via `deltic bugs new` model=claude-opus-5@high)

## Observation

Split from RVP9-BUG-KILN-00024 during its 2026-07-31 two-eyes verification.

crates/rvp9-harness/tests/tile_threading.rs:380-427 (parallel_tile_decode_reuses_isolated_scratch_buffers) samples the PROCESS-GLOBAL counter rvp9_core::frame::RVP9_G5_2_SCRATCH_ALLOCS before and after each decode_frame call, then asserts the first frame's delta is exactly 8 (one scratch frame per tile). libtest runs the seven tests in that binary concurrently by default, and the sibling multi-tile decode tests bump the same global counter inside that window.

Observed on trunk 4abe0d8 + docs commit 3a7aba2, Windows x86_64, with conformance vectors fetched:
- cargo test -p rvp9-harness --test tile_threading           -> FAILED, 'left: 12, right: 8'  (reproduced twice, deterministic)
- cargo test -p rvp9-harness --test tile_threading -- --test-threads=1 -> ok, 7 passed
- cargo test -p rvp9-harness --test tile_threading parallel_tile_decode_reuses_isolated_scratch_buffers -> ok (alone)

Impact is not confined to this test: cargo test --workspace --locked is the repo's documented full gate and the .deltic-integrate.toml fallback gate, so any machine that has run scripts/fetch-vectors.sh gets a red workspace gate. The test is #[cfg_attr(not(rvp9_conformance_vectors), ignore)]-equivalent (it needs tests/conformance/vectors/vp90-2-08-tile_1x8.webm), which is why it went unnoticed: a vectorless worktree skips it.

This is the same defect class as RVP9-BUG-FLUX-00004 (bit-accounting tests racing a process-wide switch), which was fixed with a serializing mutex.

Expected: the scratch-allocation assertion is isolated from sibling tests (a shared serializing lock as in entropy::bitacct, or a per-decoder counter instead of a global).
Actual: a global counter sampled across a concurrent window, so the assertion measures other tests' allocations.

## Fix

<unfixed — raised only>

## Notes
