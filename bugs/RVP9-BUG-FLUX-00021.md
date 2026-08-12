# RVP9-BUG-FLUX-00021 — Decoder output mismatches every frame of the official Hantro subpixel vector

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** decoder/inter prediction
- **Raised:** 2026-08-12T14:08:04Z
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
- **State history:** Open (2026-08-12T14:08:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-12T14:31:46Z, deltic:auto role=fix run=fix-20260812T141802Z-p14933-n235872000-c1 branch=task/bug-RVP9-BUG-FLUX-00021-run-fix-20260812T141802Z-p14933-n235872000-c1 code=640625f3 gate=manual)

## Observation

The rvp9 decoder produces the wrong visible MD5 for all 20 frames of vp90-2-09-subpixel-00.ivf versus the pinned libvpx ledger. The stream decodes without an error, making this silent wrong output. Reproduced in the 307-vector strict run and an isolated 7-vector rerun on 2026-08-12.

## Fix

<unfixed — raised only>

## Notes
