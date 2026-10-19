# RVP9-BUG-FLUX-00021 — Decoder output mismatches every frame of the official Hantro subpixel vector

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** decoder/inter prediction
- **Raised:** 2026-08-12T14:08:04Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T065214Z-42e66faf
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00021-run-verify-20261019T065214Z-42e66faf
- **Owner base:** a989c5c80d2197b1787d55f24ad5a9cfbe5357b8
- **Owner fingerprint:** sha256:cfe3dff760db9da2f0564e2a97c2a64d15af68495462226cccada3b32158bd80
- **Owner since:** 2026-10-19T06:52:14Z
- **Owner until:** 2026-10-19T10:52:14Z
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
