# RVP9-BUG-FLUX-00015 — Rectangular-partition proof does not enable full-motion diagnostic

- **State:** Fixed
- **Priority:** Should
- **Severity:** Low
- **Area:** encoder tests
- **Raised:** 2026-08-09T22:49:48Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T064452Z-871b2222
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00015-run-verify-20261019T064452Z-871b2222
- **Owner base:** 22fca7b71aaeca973c013a985da7541983759738
- **Owner fingerprint:** sha256:cbf2be636538d8bfeed0c4e674103b60dd8157b19e8ddd0877bd69c8edc3244f
- **Owner since:** 2026-10-19T06:44:52Z
- **Owner until:** 2026-10-19T10:44:52Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-09T22:49:48Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-09T23:01:25Z, deltic:auto role=fix run=fix-20260809T230104Z-p97610-n246368000-c1 branch=task/bug-RVP9-BUG-FLUX-00015-run-fix-20260809T230104Z-p97610-n246368000-c1 code=dc46d43708262dfebdca2188a5009b2f0653fb96 gate=manual)

## Observation

The w_rd_partition_rect proof expects a rectangular winner on content whose two bands require different motion vectors, but it inherits the production default that disables independent NEWMV search for rectangular proxy candidates. The test therefore reports zero rectangular leaves even though its intended diagnostic override produces 132 and the stream decodes cleanly. Make the focused proof explicitly enable full rectangular motion search and restore the environment afterward.

## Fix

<unfixed — raised only>

## Notes
