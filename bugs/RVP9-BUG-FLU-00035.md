# RVP9-BUG-FLU-00035 — AVX2 horizontal-16 mask filters edges scalar path rejects at blimit 255

- **State:** Closed
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
- **State history:** Open (2026-09-27T19:23:02Z, raised via `deltic bugs new --land`) -> Fixed (2026-10-16T05:40:36Z, deltic:auto role=fix run=fix-20261016T053336Z-d0e38b83 branch=task/bug-RVP9-BUG-FLU-00035-run-fix-20261016T053336Z-d0e38b83 code=769561ad6be45d5c105d6e6e754f53268ad44b14 gate=manual) -> Closed (2026-10-18T22:04:41Z, deltic:manual role=verify run=verify-20261018T215507Z-4c570064 verified_fix_run=fix-20261016T053336Z-d0e38b83 verdict=close)

## Observation

On x86-64, the horizontal-16 AVX2 single filter accepts some edges that the scalar filter rejects when blimit=255. The new dual-kernel differential first passed dual versus two AVX2 singles, then failed dual versus scalar at thresholds [255,255,255] because the shared existing u8 macroblock-edge mask calculation saturates above 255. Expected: scalar and AVX2 filtered buffers match for every u8 threshold. Normal encoder-generated blimit values reach at most 193, so this appears to affect the public kernel contract rather than the standard encode path. This is distinct from the vertical-16 sibling RVP9-BUG-FLU-00034.

Evidence fingerprint: `manual:v1:core-avx2-horizontal16-blimit255-saturation`


## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T22:04:41Z, deltic:auto run=verify-20261018T215507Z-4c570064 verified_fix_run=fix-20261016T053336Z-d0e38b83 verdict=close)

Verifier note: Manual verify: Independent native Windows x86-64 Crucible verification at a3113bd40d0e7efd1d962f532bc6d893ba3ee243 with both fix ancestors confirmed. Explicit CPU probe establishes AVX2 available. cargo test -p rvp9-core --lib dsp::x86_64::lpf_horizontal_16_avx2::tests::blimit_255_saturation_matches_scalar -- --exact --nocapture selects one test and passes. Includes original p0=246/q0=3 metric607 plus threshold controls. Reversing only the special blimit255 mask comparison selects one test and fails scalar/full-buffer equality at pitch=8 blimit=255; original bytes restored, one test passes aga...

## Notes
