# RVP9-BUG-FLUX-00020 — Decoder rejects three official VP9 tile/skip vectors during entropy decoding

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** decoder/tile entropy
- **Raised:** 2026-08-12T14:08:04Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20260812T154108Z-p87925-n351109000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00020-run-verify-20260812T154108Z-p87925-n351109000-c1
- **Owner base:** 94a19c89818529f350f0c8787f9c6ffb07b1e0a0
- **Owner fingerprint:** sha256:ba86e62547ee94061221d7bc95000cb1694fa4afb4bcad46b9439157855ed410
- **Owner since:** 2026-08-12T15:41:08Z
- **Owner until:** 2026-08-12T17:41:08Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-12T14:08:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-12T15:39:07Z, deltic:auto role=fix run=fix-20260812T150933Z-p46593-n297689000-c1 branch=task/bug-RVP9-BUG-FLUX-00020-run-fix-20260812T150933Z-p46593-n297689000-c1 code=83bd7566 gate=manual)

## Observation

The rvp9 decoder fails vp90-2-08-tile-4x1.webm and vp90-2-08-tile-4x4.webm on packet 1, plus vp90-2-19-skip-02.webm on packet 6, with tile payload ended during entropy decoding. These are SHA-1-verified pinned libvpx vectors with MD5 ledgers. Reproduced in the 307-vector strict run and an isolated 7-vector rerun on 2026-08-12.

## Fix

<unfixed — raised only>

## Notes
