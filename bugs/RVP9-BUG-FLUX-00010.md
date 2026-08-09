# RVP9-BUG-FLUX-00010 — BD-rate runs collide through fixed temporary filenames

- **State:** Open
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
- **State history:** Open (2026-08-09T13:33:20Z, raised via `deltic bugs new` model=gpt-5.6@xhigh)

## Observation

Running three bdrate processes concurrently for in-tree, ducks, and old-town silently produced cross-contaminated curves. crates/rvp9-bench/src/bin/bdrate.rs:encode writes every process to the same rvp9_bdrate_{role}_{bitrate}k.ivf names, and decode_and_psnr derives the same decoded-output names. Expected: independent concurrent runs use isolated artifacts and reproduce serialized results. Actual: processes overwrite each other's IVF/Y4M files; the corrupted CPU6 curves were +5.497%/-9.845%/+4.108%, while serialized reruns on the same pinned binaries and sources measured +13.297%/-13.717%/-0.566%.

## Fix

<unfixed — raised only>

## Notes
