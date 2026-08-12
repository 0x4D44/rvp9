# RVP9-BUG-FLUX-00022 — Decoder cannot decode the pinned spatial-SVC conformance vector

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** decoder/spatial SVC
- **Raised:** 2026-08-12T14:08:04Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260812T172201Z-p80631-n983536000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00022-run-verify-20260812T172201Z-p80631-n983536000-c1
- **Owner base:** ff8fa82252217b76ada2f3356e7eaa8db750d854
- **Owner fingerprint:** sha256:835b160d171259a26c4b1c1b07d0a06cc80bd3c47fd02cbc0e5c16f7df4a04c2
- **Owner since:** 2026-08-12T17:22:01Z
- **Owner until:** 2026-08-12T19:22:01Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-12T14:08:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-12T17:18:45Z, deltic:auto role=fix run=fix-20260812T163429Z-p42548-n323105000-c1 branch=task/bug-RVP9-BUG-FLUX-00022-run-fix-20260812T163429Z-p42548-n323105000-c1 code=d2be2f6dcc935270ea173d043275d178f8872687 gate=manual)

## Observation

The rvp9 decoder mismatches the first three displayed frames of vp90-2-22-svc_1280x720_3.ivf, then rejects packet 2 because a 1280x720 reference is outside the size envelope for a 320x180 frame. This is the pinned libvpx spatial-SVC vector. Reproduced in the 307-vector strict run and an isolated 7-vector rerun on 2026-08-12; spatial SVC encoding remains outside the alpha scope, but decoder conformance is incomplete.

## Fix

<unfixed — raised only>

## Notes
