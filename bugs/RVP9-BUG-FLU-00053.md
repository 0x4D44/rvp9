# RVP9-BUG-FLU-00053 — Scaled NEON convolution overflows Q4 coordinates before pointer validation

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** core/dsp convolve8-scaled-neon
- **Raised:** 2026-10-17T20:46:58Z
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
- **State history:** Open (2026-10-17T20:46:58Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium) -> Fixed (2026-10-18T19:10:54Z, deltic:auto role=fix run=fix-20261018T183319Z-ba7a9bfd branch=task/bug-RVP9-BUG-FLU-00053-run-fix-20261018T183319Z-ba7a9bfd code=2bf1c72d4ecce5606a34da5065481ff3d2f1d879 gate=manual)

## Observation

The safe AArch64 convolve8_scaled_neon dispatch path accepts nonnegative i32 Q4 coordinates, then computes intermediate_height at crates/rvp9-core/src/dsp/aarch64/convolve8_scaled_neon.rs:321-343 and per-row y_q4 at lines 281-282 using unchecked i32 multiply/add before checked usize span calculations. In a 64-bit release build, w=4, h=3, x0=0, x_step=0, y0=2, y_step=i32::MAX, src_stride=8, src.len()=64, dst_stride=4, and dst.len()=12 pass the apparent footprint checks: the height expression wraps to eight rows. At output row one, y0 + y_step wraps negative; shifting and casting it to usize yields a huge row index, and line 305 forms an intermediate pointer outside the allocation. Expected: reject coordinate overflow before pointer formation. Actual: the safe API validates spans only after signed coordinate arithmetic has wrapped. This differs from RVP9-BUG-FLU-00046's horizontal stride-span overflow. Source-derived; no runtime reproduction was attempted.

## Fix

The safe AArch64 wrapper now computes the final horizontal and vertical Q4 coordinates with checked multiplication and addition before deriving the source footprint or entering the raw-pointer passes. Nonnegative steps and validated dimensions mean each checked endpoint bounds all row or column coordinates used by the kernel. Intermediate height is derived from the checked vertical endpoint.

The release regression failed before the fix because the selected vertical overflow case did not panic; unchecked arithmetic wrapped and the undersized fixture passed footprint validation. On the final source, both horizontal and vertical overflow tests pass under the release profile. The complete scaled-NEON test module passed all four tests on arm64, including both scalar-vs-NEON differential tests. No invalid coordinate reached a pointer operation.

## Notes
