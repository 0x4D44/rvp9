# RVP9-BUG-FLU-00032 — SSE2 square subtraction stride arithmetic can wrap before unsafe loads

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** core/dsp/subtract_sse2
- **Raised:** 2026-09-07T19:53:47Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260907T195440Z-af212a54
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00032-run-fix-20260907T195440Z-af212a54
- **Owner base:** 08419950e8ee7ac3dcb106e8b18b09d472685ce1
- **Owner fingerprint:** -
- **Owner since:** 2026-09-07T19:54:40Z
- **Owner until:** 2026-09-07T21:54:40Z
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
