# RVP9-BUG-FLUX-00019 — Official skip-vector ledgers with ordinal filenames fail the conformance harness

- **State:** Open
- **Priority:** Must
- **Severity:** Low
- **Area:** harness/conformance
- **Raised:** 2026-08-12T14:08:03Z
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
- **State history:** Open (2026-08-12T14:08:03Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

The pinned libvpx ledgers for vp90-2-19-skip.webm and vp90-2-19-skip-01.webm use 01.yuv-style filenames. crates/rvp9-harness/src/md5file.rs rejects them because it requires WxH-NNNN in every filename, so the strict official-corpus gate fails before decoding. Reproduced in the 307-vector run and an isolated 7-vector rerun on 2026-08-12.

## Fix

<unfixed — raised only>

## Notes
