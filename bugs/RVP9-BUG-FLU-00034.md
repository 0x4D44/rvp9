# RVP9-BUG-FLU-00034 — AVX2 vertical-16 mask filters edges scalar path rejects at blimit 255

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** core/loopfilter
- **Raised:** 2026-09-27T19:07:34Z
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
- **State history:** Open (2026-09-27T19:07:34Z, raised via `deltic bugs new --land`)

## Observation

On x86-64, the vertical-16 AVX2 loop-filter mask can accept an edge that the scalar path rejects when blimit=255. For p0=246 and q0=3, the scalar macroblock-edge difference exceeds 255, but the SIMD u8 sum saturates to 255 and leaves the mask enabled; the new full-filter differential in task/20260927-TSK-HUM-share-vertical16-avx2-wide-filter-sums failed at case=4 pitch=16 before a widened comparison. Expected: scalar and AVX2 filtered buffers match for every u8 threshold. This does not appear reachable from the normal encoder threshold table, whose maximum blimit is 193.

Evidence fingerprint: `manual:v1:core-avx2-vertical16-blimit255-saturation`


## Fix

<unfixed — raised only>

## Notes
