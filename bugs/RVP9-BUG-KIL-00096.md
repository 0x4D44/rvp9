# RVP9-BUG-KIL-00096 — Zero-frame Y4M reads allocate a full source frame

- **State:** Open
- **Priority:** Could
- **Severity:** Medium
- **Area:** io/y4m reader
- **Raised:** 2026-08-22T14:33:55Z
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
- **State history:** Open (2026-08-22T14:33:55Z, raised via `deltic bugs new`)

## Observation

Y4mReader::open always allocates and zeroes source.frame_bytes at crates/rvp9-io/src/y4m.rs:312-326 before recording frame_limit. next_frame_into returns immediately for remaining == 0 at :357-360, so a zero-frame request never uses that allocation. A valid maximum-size 16384x16384 4:2:0 header therefore causes 384 MiB of allocation and zeroing for an operation guaranteed to return no frames. This is separate from the deliberately benchmarked allocation layout for active full-frame reads. Fix by leaving body empty when frame_limit is zero, or lazily allocating only before a cropped frame is read. Add a zero-limit constructor oracle that inspects capacity through a focused internal test and preserves header validation. Estimated effort: small. Confirmed by performance review, a devil advocate, and the lead; no benchmark, test, or application was run.

## Fix

<unfixed — raised only>

## Notes
