# RVP9-BUG-FLUX-00015 — Rectangular-partition proof does not enable full-motion diagnostic

- **State:** Open
- **Priority:** Should
- **Severity:** Low
- **Area:** encoder tests
- **Raised:** 2026-08-09T22:49:48Z
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
- **State history:** Open (2026-08-09T22:49:48Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

The w_rd_partition_rect proof expects a rectangular winner on content whose two bands require different motion vectors, but it inherits the production default that disables independent NEWMV search for rectangular proxy candidates. The test therefore reports zero rectangular leaves even though its intended diagnostic override produces 132 and the stream decodes cleanly. Make the focused proof explicitly enable full rectangular motion search and restore the environment afterward.

## Fix

<unfixed — raised only>

## Notes
