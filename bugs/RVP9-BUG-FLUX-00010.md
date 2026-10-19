# RVP9-BUG-FLUX-00010 — BD-rate runs collide through fixed temporary filenames

- **State:** Closed
- **Priority:** Must
- **Severity:** High
- **Area:** bench/bdrate
- **Raised:** 2026-08-09T13:33:20Z
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
- **State history:** Open (2026-08-09T13:33:20Z, raised via `deltic bugs new` model=gpt-5.6@xhigh) -> Fixed (2026-08-22T13:10:53Z, deltic:auto role=fix run=fix-20260822T130927Z-501b37bd branch=task/bug-RVP9-BUG-FLUX-00010-run-fix-20260822T130927Z-501b37bd code=0fdff494e gate=manual) -> Closed (2026-10-19T06:33:39Z, deltic:manual role=verify run=verify-20261019T061236Z-3055779e verified_fix_run=fix-20260822T130927Z-501b37bd verdict=close)

## Observation

Running three bdrate processes concurrently for in-tree, ducks, and old-town silently produced cross-contaminated curves. crates/rvp9-bench/src/bin/bdrate.rs:encode writes every process to the same rvp9_bdrate_{role}_{bitrate}k.ivf names, and decode_and_psnr derives the same decoded-output names. Expected: independent concurrent runs use isolated artifacts and reproduce serialized results. Actual: processes overwrite each other's IVF/Y4M files; the corrupted CPU6 curves were +5.497%/-9.845%/+4.108%, while serialized reruns on the same pinned binaries and sources measured +13.297%/-13.717%/-0.566%.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-19T06:33:39Z, deltic:auto run=verify-20261019T061236Z-3055779e verified_fix_run=fix-20260822T130927Z-501b37bd verdict=close)

Verifier note: Manual verify: Independent verification on current origin/main: the bdrate default-artifacts regression passed (1 selected test). Fails-before reversal changed only prepare_artifacts(None) from the PID prefix to the prior fixed rvp9_bdrate_ prefix; the same selected test failed at the filename assertion, reporting rvp9_bdrate_anchor_500k.ivf instead of the PID-prefixed name. Restored the PID prefix; the test passed again, and the source diff is empty. The fix namespaces IVF paths by process ID, and decoded Y4M paths derive from those IVF stems. The original three-run numeric BD-rate observatio...

## Notes
