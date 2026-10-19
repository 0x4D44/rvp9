# RVP9-BUG-FLUX-00015 — Rectangular-partition proof does not enable full-motion diagnostic

- **State:** Closed
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
- **State history:** Open (2026-08-09T22:49:48Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-09T23:01:25Z, deltic:auto role=fix run=fix-20260809T230104Z-p97610-n246368000-c1 branch=task/bug-RVP9-BUG-FLUX-00015-run-fix-20260809T230104Z-p97610-n246368000-c1 code=dc46d43708262dfebdca2188a5009b2f0653fb96 gate=manual) -> Closed (2026-10-19T06:47:20Z, deltic:manual role=verify run=verify-20261019T064452Z-871b2222 verified_fix_run=fix-20260809T230104Z-p97610-n246368000-c1 verdict=close)

## Observation

The w_rd_partition_rect proof expects a rectangular winner on content whose two bands require different motion vectors, but it inherits the production default that disables independent NEWMV search for rectangular proxy candidates. The test therefore reports zero rectangular leaves even though its intended diagnostic override produces 132 and the stream decodes cleanly. Make the focused proof explicitly enable full rectangular motion search and restore the environment afterward.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-19T06:47:20Z, deltic:auto run=verify-20261019T064452Z-871b2222 verified_fix_run=fix-20260809T230104Z-p97610-n246368000-c1 verdict=close)

Verifier note: Manual verify: Independent verification: the ledger SHA dc46d43708262dfebdca2188a5009b2f0653fb96 is absent; matching landed fix f94a942ec4cfd4b8b13d98c3af51edb7f6941fc8 is an ancestor of origin/main and explicitly enables independent rectangular NEWMV for the focused proof, then restores the production default. cargo test -p rvp9-encoder --test w_rd_partition_rect a4_rect_partition_is_evaluated_and_decodes --locked -- --exact --nocapture passed with 38 rectangular leaves; the built-in decoder check passed and optional vpxdec was skipped because no external vpxdec is installed. Fails-before set...

## Notes
