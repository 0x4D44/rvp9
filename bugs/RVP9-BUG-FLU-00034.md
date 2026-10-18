# RVP9-BUG-FLU-00034 — AVX2 vertical-16 mask filters edges scalar path rejects at blimit 255

- **State:** Fixed
- **Priority:** Could
- **Severity:** Low
- **Area:** core/loopfilter
- **Raised:** 2026-09-27T19:07:34Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T215459Z-34070918
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00034-run-verify-20261018T215459Z-34070918
- **Owner base:** e6379d34563b0ed1bb7afc79e73abab46ec9508c
- **Owner fingerprint:** sha256:5329dd138fa65b70498e05270630ddf479eac7b3d21bb62224019a97b3cf922a
- **Owner since:** 2026-10-18T21:54:59Z
- **Owner until:** 2026-10-19T01:54:59Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-09-27T19:07:34Z, raised via `deltic bugs new --land`) -> Fixed (2026-10-16T07:02:56Z, deltic:auto role=fix run=fix-20261016T065134Z-2fd058d6 branch=task/bug-RVP9-BUG-FLU-00034-run-fix-20261016T065134Z-2fd058d6 code=4c47469dcd60cd10eaf7d1e23bdae410295ca312 gate=manual)

## Observation

On x86-64, the vertical-16 AVX2 loop-filter mask can accept an edge that the scalar path rejects when blimit=255. For p0=246 and q0=3, the scalar macroblock-edge difference exceeds 255, but the SIMD u8 sum saturates to 255 and leaves the mask enabled; the new full-filter differential in task/20260927-TSK-HUM-share-vertical16-avx2-wide-filter-sums failed at case=4 pitch=16 before a widened comparison. Expected: scalar and AVX2 filtered buffers match for every u8 threshold. This does not appear reachable from the normal encoder threshold table, whose maximum blimit is 193.

Evidence fingerprint: `manual:v1:core-avx2-vertical16-blimit255-saturation`


## Fix

<unfixed — raised only>

## Notes
