# RVP9-BUG-FLUX-00020 — Decoder rejects three official VP9 tile/skip vectors during entropy decoding

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** decoder/tile entropy
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
- **State history:** Open (2026-08-12T14:08:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

The rvp9 decoder fails vp90-2-08-tile-4x1.webm and vp90-2-08-tile-4x4.webm on packet 1, plus vp90-2-19-skip-02.webm on packet 6, with tile payload ended during entropy decoding. These are SHA-1-verified pinned libvpx vectors with MD5 ledgers. Reproduced in the 307-vector strict run and an isolated 7-vector rerun on 2026-08-12.

## Fix

<unfixed — raised only>

## Notes
