# RVP9-BUG-FLUX-00023 — Decoder output mismatches every frame at the official 226x210 edge size

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** decoder/odd frame dimensions
- **Raised:** 2026-08-12T14:15:51Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260812T145724Z-p26247-n379818000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00023-run-fix-20260812T145724Z-p26247-n379818000-c1
- **Owner base:** f143af189cb39f00694997887d3b5ad6d1fc8b78
- **Owner fingerprint:** -
- **Owner since:** 2026-08-12T14:57:24Z
- **Owner until:** 2026-08-12T16:57:24Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-12T14:15:51Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

The rvp9 decoder produces the wrong visible MD5 for all 10 frames of vp90-2-03-size-226x210.webm, while neighboring official size vectors pass. The stream decodes without an error, making this silent wrong output. Reproduced in two independent full 307-vector strict runs on 2026-08-12; pinned vpxdec and ffmpeg decode the SHA-1-verified stream successfully.

## Fix

<unfixed — raised only>

## Notes
