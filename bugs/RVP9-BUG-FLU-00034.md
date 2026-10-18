# RVP9-BUG-FLU-00034 — AVX2 vertical-16 mask filters edges scalar path rejects at blimit 255

- **State:** Closed
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
- **State history:** Open (2026-09-27T19:07:34Z, raised via `deltic bugs new --land`) -> Fixed (2026-10-16T07:02:56Z, deltic:auto role=fix run=fix-20261016T065134Z-2fd058d6 branch=task/bug-RVP9-BUG-FLU-00034-run-fix-20261016T065134Z-2fd058d6 code=4c47469dcd60cd10eaf7d1e23bdae410295ca312 gate=manual) -> Closed (2026-10-18T22:04:31Z, deltic:manual role=verify run=verify-20261018T215459Z-34070918 verified_fix_run=fix-20261016T065134Z-2fd058d6 verdict=close)

## Observation

On x86-64, the vertical-16 AVX2 loop-filter mask can accept an edge that the scalar path rejects when blimit=255. For p0=246 and q0=3, the scalar macroblock-edge difference exceeds 255, but the SIMD u8 sum saturates to 255 and leaves the mask enabled; the new full-filter differential in task/20260927-TSK-HUM-share-vertical16-avx2-wide-filter-sums failed at case=4 pitch=16 before a widened comparison. Expected: scalar and AVX2 filtered buffers match for every u8 threshold. This does not appear reachable from the normal encoder threshold table, whose maximum blimit is 193.

Evidence fingerprint: `manual:v1:core-avx2-vertical16-blimit255-saturation`


## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T22:04:31Z, deltic:auto run=verify-20261018T215459Z-34070918 verified_fix_run=fix-20261016T065134Z-2fd058d6 verdict=close)

Verifier note: Manual verify: Independent native Windows x86-64 Crucible verification at a3113bd40d0e7efd1d962f532bc6d893ba3ee243 with both fix ancestors confirmed. Explicit CPU probe establishes AVX2 available. cargo test -p rvp9-core --lib dsp::x86_64::lpf_vertical_16_avx2::tests::blimit_255_saturation_matches_scalar -- --exact --nocapture selects one test and passes. Includes original p0=246/q0=3 metric607 plus threshold controls. Reversing only the special blimit255 mask comparison selects one test and fails scalar/full-buffer equality at pitch=16 blimit=255; original bytes restored, one test passes agai...

## Notes
