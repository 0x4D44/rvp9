# RVP9-BUG-FLU-00035 — AVX2 horizontal-16 mask filters edges scalar path rejects at blimit 255

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** core/loopfilter
- **Raised:** 2026-09-27T19:23:02Z
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
- **State history:** Open (2026-09-27T19:23:02Z, raised via `deltic bugs new --land`)

## Observation

On x86-64, the horizontal-16 AVX2 single filter accepts some edges that the scalar filter rejects when blimit=255. The new dual-kernel differential first passed dual versus two AVX2 singles, then failed dual versus scalar at thresholds [255,255,255] because the shared existing u8 macroblock-edge mask calculation saturates above 255. Expected: scalar and AVX2 filtered buffers match for every u8 threshold. Normal encoder-generated blimit values reach at most 193, so this appears to affect the public kernel contract rather than the standard encode path. This is distinct from the vertical-16 sibling RVP9-BUG-FLU-00034.

Evidence fingerprint: `manual:v1:core-avx2-horizontal16-blimit255-saturation`


## Fix

<unfixed — raised only>

## Notes
