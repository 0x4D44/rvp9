# RVP9-BUG-KIL-00096 — Zero-frame Y4M reads allocate a full source frame

- **State:** Closed
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
- **State history:** Open (2026-08-22T14:33:55Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T06:51:09Z, deltic:auto role=fix run=fix-20261016T064334Z-c44f9b5a branch=task/bug-RVP9-BUG-KIL-00096-run-fix-20261016T064334Z-c44f9b5a code=ba4383249a2af57dfeabecc4b09e82c53cccfb1f gate=manual) -> Closed (2026-10-18T15:51:52Z, deltic:manual role=verify run=verify-20261018T154807Z-312588d7 verified_fix_run=fix-20261016T064334Z-c44f9b5a verdict=close)

## Observation

Y4mReader::open always allocates and zeroes source.frame_bytes at crates/rvp9-io/src/y4m.rs:312-326 before recording frame_limit. next_frame_into returns immediately for remaining == 0 at :357-360, so a zero-frame request never uses that allocation. A valid maximum-size 16384x16384 4:2:0 header therefore causes 384 MiB of allocation and zeroing for an operation guaranteed to return no frames. This is separate from the deliberately benchmarked allocation layout for active full-frame reads. Fix by leaving body empty when frame_limit is zero, or lazily allocating only before a cropped frame is read. Add a zero-limit constructor oracle that inspects capacity through a focused internal test and preserves header validation. Estimated effort: small. Confirmed by performance review, a devil advocate, and the lead; no benchmark, test, or application was run.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T15:51:52Z, deltic:auto run=verify-20261018T154807Z-312588d7 verified_fix_run=fix-20261016T064334Z-c44f9b5a verdict=close)

Verifier note: Manual verify: Independent session01a14fa5, not fixer. Fixba438324 present. Exact y4m_zero_frame_limit_leaves_source_body_unallocated selects1, passes before/after. Original maximum16384x16384 header retains body len/capacity0; header/crop errors and caller buffers preserved. Restore unconditional zeroed_buffer: selected1 assertion fails capacity96!=0 on small full-frame source before maximum allocation. Exact bytes restored/source diff empty; lead inspected raw proof. Workspace fmt/clippy pass; workspace failure/retry and IO doctest artifact failure disclosed in2026.10.18 verification journal...

## Notes
