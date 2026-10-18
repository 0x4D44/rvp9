# RVP9-BUG-KILN-00072 — Saturating stream counters stop GOP and keyframe cadence

- **State:** Closed
- **Priority:** Could
- **Severity:** Medium
- **Area:** encoder/GOP cadence
- **Raised:** 2026-08-16T08:19:23Z
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
- **State history:** Open (2026-08-16T08:19:23Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T09:00:38Z, deltic:auto role=fix run=fix-20261016T084100Z-a89929bf branch=task/bug-RVP9-BUG-KILN-00072-run-fix-20261016T084100Z-a89929bf code=b2a0e12a17d551719b70cd4443b376fd95bddff2 gate=manual) -> Closed (2026-10-18T17:22:06Z, deltic:manual role=verify run=verify-20261018T171235Z-d2b55406 verified_fix_run=fix-20261016T084100Z-a89929bf verdict=close)

## Observation

StreamEncoder increments its u32 frame_idx with saturating_add at D:\worktrees\rvp9-2\20260816-REV-RVP9-CDX@KILN-code-review-085355\crates\rvp9-encoder\src\frame_loop.rs:3239, while GfGroup derives the schedule with frame_idx modulo gop_length at gop.rs:199. After u32::MAX frames the offset freezes, so ordinary GOP and keyframe boundaries stop; for an interval dividing u32::MAX, every later frame can be a boundary. SpatialEncoder independently saturates its u32 access_unit_index at spatial_encoder.rs:319 and uses that value for keyframe cadence. At 60 fps the boundary occurs after about 2.27 years of continuous encoding.

## Fix

Use a non-saturating cadence representation, such as a u64 frame count or explicitly wrapped modulo counters, for both streaming paths. Add tests seeded near u32::MAX that cross the boundary and preserve normal keyframe, golden-frame, and ALTREF scheduling.

### Verification summary (2026-10-18T17:22:06Z, deltic:auto run=verify-20261018T171235Z-d2b55406 verified_fix_run=fix-20261016T084100Z-a89929bf verdict=close)

Verifier note: Manual verify: GOP 10/15 cadence, ALTREF scheduling and spatial SVC cadence pass after u32 counter saturation. Reversing the four GOP-phase schedule reads and spatial keyframe-phase check to the original saturating frame/AU counters makes all four retained cadence tests fail at the boundary; exact sources restored and all four pass again.

## Notes
