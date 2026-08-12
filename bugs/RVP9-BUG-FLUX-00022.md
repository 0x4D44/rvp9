# RVP9-BUG-FLUX-00022 — Decoder cannot decode the pinned spatial-SVC conformance vector

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** decoder/spatial SVC
- **Raised:** 2026-08-12T14:08:04Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260812T163429Z-p42548-n323105000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00022-run-fix-20260812T163429Z-p42548-n323105000-c1
- **Owner base:** 29612424473922311ed1e53c0358642024cafba8
- **Owner fingerprint:** -
- **Owner since:** 2026-08-12T16:34:29Z
- **Owner until:** 2026-08-12T18:34:29Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-12T14:08:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

The rvp9 decoder mismatches the first three displayed frames of vp90-2-22-svc_1280x720_3.ivf, then rejects packet 2 because a 1280x720 reference is outside the size envelope for a 320x180 frame. This is the pinned libvpx spatial-SVC vector. Reproduced in the 307-vector strict run and an isolated 7-vector rerun on 2026-08-12; spatial SVC encoding remains outside the alpha scope, but decoder conformance is incomplete.

## Fix

<unfixed — raised only>

## Notes
