# RVP9-BUG-KIL-00097 — Y4M writer emits headers rejected by its reader

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** io/y4m writer
- **Raised:** 2026-08-22T14:34:05Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261015T234959Z-a7a363cd
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00097-run-fix-20261015T234959Z-a7a363cd
- **Owner base:** cfbda57360fcfdd40124ab73d7b7d6c3772433f7
- **Owner fingerprint:** -
- **Owner since:** 2026-10-15T23:49:59Z
- **Owner until:** 2026-10-16T03:49:59Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T14:34:05Z, raised via `deltic bugs new`)

## Observation

write_header at crates/rvp9-io/src/y4m.rs:482-523 validates only the chroma enum. For Yuv420 it emits zero dimensions, odd dimensions, oversized dimensions, or a zero framerate numerator/denominator. parse_header rejects those fields at :257-266 through checked_yuv420_layout at :47-97, violating the documented 4:2:0 writer/reader round-trip. The write_frame caller-responsibility note covers plane lengths, not header validity. Expected: invalid Yuv420 metadata returns InvalidInput before any bytes are written. Fix by sharing the geometry envelope and nonzero framerate checks before writeln. Add table-driven zero, odd, oversized, and zero-framerate no-partial-output regressions. Estimated effort: small. Confirmed by maintainability review, an independent skeptic, and the lead; no tests or application were run.

## Fix

<unfixed — raised only>

## Notes
