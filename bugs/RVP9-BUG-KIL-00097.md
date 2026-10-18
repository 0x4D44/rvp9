# RVP9-BUG-KIL-00097 — Y4M writer emits headers rejected by its reader

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** io/y4m writer
- **Raised:** 2026-08-22T14:34:05Z
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
- **State history:** Open (2026-08-22T14:34:05Z, raised via `deltic bugs new`) -> Fixed (2026-10-15T23:54:39Z, deltic:auto role=fix run=fix-20261015T234959Z-a7a363cd branch=task/bug-RVP9-BUG-KIL-00097-run-fix-20261015T234959Z-a7a363cd code=c7356aab42551b93f72dfd887caa0e235179a640 gate=manual) -> Closed (2026-10-18T15:43:09Z, deltic:manual role=verify run=verify-20261018T153922Z-9cc6182a verified_fix_run=fix-20261015T234959Z-a7a363cd verdict=close)

## Observation

write_header at crates/rvp9-io/src/y4m.rs:482-523 validates only the chroma enum. For Yuv420 it emits zero dimensions, odd dimensions, oversized dimensions, or a zero framerate numerator/denominator. parse_header rejects those fields at :257-266 through checked_yuv420_layout at :47-97, violating the documented 4:2:0 writer/reader round-trip. The write_frame caller-responsibility note covers plane lengths, not header validity. Expected: invalid Yuv420 metadata returns InvalidInput before any bytes are written. Fix by sharing the geometry envelope and nonzero framerate checks before writeln. Add table-driven zero, odd, oversized, and zero-framerate no-partial-output regressions. Estimated effort: small. Confirmed by maintainability review, an independent skeptic, and the lead; no tests or application were run.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T15:43:09Z, deltic:auto run=verify-20261018T153922Z-9cc6182a verified_fix_run=fix-20261015T234959Z-a7a363cd verdict=close)

Verifier note: Manual verify: Independent session01a14fa5, not fixer; base09697b55 includes c7356aab. Original invalid 420 header observation: 10 cases return InvalidInput without adding output; write_header filter selects3, passes (lead rerun too). Remove only framerate guard: exact regression fails both zero-FPS cases accepted/output changed. Remove only geometry guard: regression fails all8 dimensions accepted/output changed. Original source restored exactly; scoped diff empty; restored3 pass. Guards precede writeln and preserve reader InvalidData mapping. Workspace fmt and documented clippy pass. Full ev...

## Notes
