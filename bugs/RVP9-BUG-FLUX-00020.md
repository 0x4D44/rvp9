# RVP9-BUG-FLUX-00020 — Decoder rejects three official VP9 tile/skip vectors during entropy decoding

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** decoder/tile entropy
- **Raised:** 2026-08-12T14:08:04Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260812T150933Z-p46593-n297689000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00020-run-fix-20260812T150933Z-p46593-n297689000-c1
- **Owner base:** 4c84285a2c5d770ebf3a4d43f536dc82f6b3ca46
- **Owner fingerprint:** -
- **Owner since:** 2026-08-12T15:09:33Z
- **Owner until:** 2026-08-12T17:09:33Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-12T14:08:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

The rvp9 decoder fails vp90-2-08-tile-4x1.webm and vp90-2-08-tile-4x4.webm on packet 1, plus vp90-2-19-skip-02.webm on packet 6, with tile payload ended during entropy decoding. These are SHA-1-verified pinned libvpx vectors with MD5 ledgers. Reproduced in the 307-vector strict run and an isolated 7-vector rerun on 2026-08-12.

## Fix

<unfixed — raised only>

## Notes
