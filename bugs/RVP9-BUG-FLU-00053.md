# RVP9-BUG-FLU-00053 — Scaled NEON convolution overflows Q4 coordinates before pointer validation

- **State:** Open
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
- **State history:** Open (2026-10-17T20:46:58Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium)

## Observation

The safe AArch64 convolve8_scaled_neon dispatch path accepts nonnegative i32 Q4 coordinates, then computes intermediate_height at crates/rvp9-core/src/dsp/aarch64/convolve8_scaled_neon.rs:321-343 and per-row y_q4 at lines 281-282 using unchecked i32 multiply/add before checked usize span calculations. In a 64-bit release build, w=4, h=3, x0=0, x_step=0, y0=2, y_step=i32::MAX, src_stride=8, src.len()=64, dst_stride=4, and dst.len()=12 pass the apparent footprint checks: the height expression wraps to eight rows. At output row one, y0 + y_step wraps negative; shifting and casting it to usize yields a huge row index, and line 305 forms an intermediate pointer outside the allocation. Expected: reject coordinate overflow before pointer formation. Actual: the safe API validates spans only after signed coordinate arithmetic has wrapped. This differs from RVP9-BUG-FLU-00046's horizontal stride-span overflow. Source-derived; no runtime reproduction was attempted.

## Fix

<unfixed — raised only>

## Notes
