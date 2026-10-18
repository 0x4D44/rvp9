# RVP9-BUG-FLU-00032 — SSE2 square subtraction stride arithmetic can wrap before unsafe loads

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** core/dsp/subtract_sse2
- **Raised:** 2026-09-07T19:53:47Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T172040Z-895c46bc
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00032-run-verify-20261018T172040Z-895c46bc
- **Owner base:** 595bc8401af588246db7988962825e2c09143c01
- **Owner fingerprint:** sha256:e05532b2daef6820a47b5d9d35db74d0286480096102fde23fd1282e67fedaa9
- **Owner since:** 2026-10-18T17:20:40Z
- **Owner until:** 2026-10-18T21:20:40Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-09-07T19:53:47Z, raised via `deltic bugs new`) -> Fixed (2026-10-15T23:39:40Z, deltic:auto role=fix run=fix-20261015T233530Z-803bc9fe branch=task/bug-RVP9-BUG-FLU-00032-run-fix-20261015T233530Z-803bc9fe code=65fd7a30561c7b2bf065e48141e6a0b1e0d90ae8 gate=manual)

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
