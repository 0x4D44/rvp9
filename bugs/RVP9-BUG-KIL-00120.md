# RVP9-BUG-KIL-00120 — Two-pass encoding ignores thread and tile-column options

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9enc option contract
- **Raised:** 2026-08-22T20:44:02Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T163733Z-a17060cf
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00120-run-verify-20261018T163733Z-a17060cf
- **Owner base:** 94756a00a1fc645e068fc0717ddac0ee871fbbe2
- **Owner fingerprint:** sha256:8433e6cf11896f2cfa0e273674f5cc40928ed1ee8c6737dba90b307bc2e91480
- **Owner since:** 2026-10-18T16:37:33Z
- **Owner until:** 2026-10-18T20:37:33Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T20:44:02Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-10-16T01:00:57Z, deltic:auto role=fix run=fix-20261016T005409Z-e51506ee branch=task/bug-RVP9-BUG-KIL-00120-run-fix-20261016T005409Z-e51506ee code=889687990c5a59ab34550e05832340c2d57fa640 gate=manual)

## Observation

rvp9enc accepts --threads and --tile-columns as supported, but the --passes 2 branch returns before build_encoder_parallelism and encode_two_pass constructs default single-threaded, single-tile encoders for both passes. Expected: carry EncoderParallelism through both passes or reject these combinations. Confirmed statically at crates/rvp9-cli/src/bin/rvp9enc.rs:958-982,1257,1567-1573 and crates/rvp9-encoder/src/frame_loop.rs:850,892; no app or test was run.

## Fix

<unfixed — raised only>

## Notes
