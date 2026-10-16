# RVP9-BUG-KIL-00120 — Two-pass encoding ignores thread and tile-column options

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9enc option contract
- **Raised:** 2026-08-22T20:44:02Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T005409Z-e51506ee
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00120-run-fix-20261016T005409Z-e51506ee
- **Owner base:** 26451e5abc990c32500b7cf1807bb75a05909101
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T00:54:09Z
- **Owner until:** 2026-10-16T04:54:09Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T20:44:02Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

rvp9enc accepts --threads and --tile-columns as supported, but the --passes 2 branch returns before build_encoder_parallelism and encode_two_pass constructs default single-threaded, single-tile encoders for both passes. Expected: carry EncoderParallelism through both passes or reject these combinations. Confirmed statically at crates/rvp9-cli/src/bin/rvp9enc.rs:958-982,1257,1567-1573 and crates/rvp9-encoder/src/frame_loop.rs:850,892; no app or test was run.

## Fix

<unfixed — raised only>

## Notes
