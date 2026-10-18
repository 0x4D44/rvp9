# RVP9-BUG-FLU-00032 — SSE2 square subtraction stride arithmetic can wrap before unsafe loads

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** core/dsp/subtract_sse2
- **Raised:** 2026-09-07T19:53:47Z
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
- **State history:** Open (2026-09-07T19:53:47Z, raised via `deltic bugs new`) -> Fixed (2026-10-15T23:39:40Z, deltic:auto role=fix run=fix-20261015T233530Z-803bc9fe branch=task/bug-RVP9-BUG-FLU-00032-run-fix-20261015T233530Z-803bc9fe code=65fd7a30561c7b2bf065e48141e6a0b1e0d90ae8 gate=manual) -> Closed (2026-10-18T17:32:55Z, deltic:manual role=verify run=verify-20261018T172040Z-895c46bc verified_fix_run=fix-20261015T233530Z-803bc9fe verdict=close)

## Observation

On x86_64 release builds, subtract_square_sse2 and subtract_square_sse2_with_nonzero validate the input span with unchecked (side - 1) * stride + side. For side=8 and stride=usize::MAX/7, the sum wraps to 6 on 64-bit targets; a 64-byte slice passes the guard even though later rows are outside the allocation. The safe wrappers must reject arithmetic overflow before entering their raw-pointer SSE2 loop. Found by source review on d74b1100; normal encoder geometry does not generate this stride. The unfinished rectangular candidate is being tested with checked span arithmetic; do not assume it will be retained.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T17:32:55Z, deltic:auto run=verify-20261018T172040Z-895c46bc verified_fix_run=fix-20261015T233530Z-803bc9fe verdict=close)

Verifier note: Manual verify: Checked square-input span overflow is verified with the retained rejects_overflowing_input_spans test. The exact source passes on x86_64 under Rosetta; reversing only checked addition makes the retained explicit-rejection assertion fail. A cfg(test) sentinel intercepted the unsafe path before any invalid SSE2 access, and its caught noncanonical panic was rejected by that assertion. Exact source restored; all three module tests pass. Full Cargo test attempts timed out during compilation, so verification used a standalone harness that included the claimed source and only stubbed t...

## Notes

2026-09-08 — The rectangular performance experiment was rejected and reverted
by `a4483295`; this bug remains Open. Candidate commit `1b7849e2` contains a
checked-span implementation and focused overflow regression that can be
extracted for a standalone fix without the rectangular API. Its predecessor
`52df6546` was tested on native Windows release and under Rosetta. The regression
checks the specific overflow panic with otherwise adequate buffers; a temporary
unchecked-arithmetic mutation and pre-unsafe sentinel made it fail on its own
assertion. Do not mark Fixed until a standalone correction lands and is tested.
