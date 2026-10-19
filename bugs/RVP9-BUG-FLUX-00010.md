# RVP9-BUG-FLUX-00010 — BD-rate runs collide through fixed temporary filenames

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** bench/bdrate
- **Raised:** 2026-08-09T13:33:20Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T061236Z-3055779e
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00010-run-verify-20261019T061236Z-3055779e
- **Owner base:** 623a30fcd612d3af9b4ba1aa80d576dd6c4f06fe
- **Owner fingerprint:** sha256:1ea5bbf95a20090e8701d8d9399773484d70e7c5da51cd534f7fb8ca34dffd0c
- **Owner since:** 2026-10-19T06:12:36Z
- **Owner until:** 2026-10-19T10:12:36Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-09T13:33:20Z, raised via `deltic bugs new` model=gpt-5.6@xhigh) -> Fixed (2026-08-22T13:10:53Z, deltic:auto role=fix run=fix-20260822T130927Z-501b37bd branch=task/bug-RVP9-BUG-FLUX-00010-run-fix-20260822T130927Z-501b37bd code=0fdff494e gate=manual)

## Observation

Running three bdrate processes concurrently for in-tree, ducks, and old-town silently produced cross-contaminated curves. crates/rvp9-bench/src/bin/bdrate.rs:encode writes every process to the same rvp9_bdrate_{role}_{bitrate}k.ivf names, and decode_and_psnr derives the same decoded-output names. Expected: independent concurrent runs use isolated artifacts and reproduce serialized results. Actual: processes overwrite each other's IVF/Y4M files; the corrupted CPU6 curves were +5.497%/-9.845%/+4.108%, while serialized reruns on the same pinned binaries and sources measured +13.297%/-13.717%/-0.566%.

## Fix

<unfixed — raised only>

## Notes
