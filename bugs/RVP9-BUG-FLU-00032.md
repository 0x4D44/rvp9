# RVP9-BUG-FLU-00032 — SSE2 square subtraction stride arithmetic can wrap before unsafe loads

- **State:** Open
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
- **State history:** Open (2026-09-07T19:53:47Z, raised via `deltic bugs new`)

## Observation

On x86_64 release builds, subtract_square_sse2 and subtract_square_sse2_with_nonzero validate the input span with unchecked (side - 1) * stride + side. For side=8 and stride=usize::MAX/7, the sum wraps to 6 on 64-bit targets; a 64-byte slice passes the guard even though later rows are outside the allocation. The safe wrappers must reject arithmetic overflow before entering their raw-pointer SSE2 loop. Found by source review on d74b1100; normal encoder geometry does not generate this stride. The unfinished rectangular candidate is being tested with checked span arithmetic; do not assume it will be retained.

## Fix

<unfixed — raised only>

## Notes

2026-09-08 — The rectangular performance experiment was rejected and reverted
by `a4483295`; this bug remains Open. Candidate commit `1b7849e2` contains a
checked-span implementation and focused overflow regression that can be
extracted for a standalone fix without the rectangular API. Its predecessor
`52df6546` was tested on native Windows release and under Rosetta. The regression
checks the specific overflow panic with otherwise adequate buffers; a temporary
unchecked-arithmetic mutation and pre-unsafe sentinel made it fail on its own
assertion. Do not mark Fixed until a standalone correction lands and is tested.
