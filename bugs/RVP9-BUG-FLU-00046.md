# RVP9-BUG-FLU-00046 — AArch64 convolution stride validation wraps before raw-pointer reads

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** core/dsp/convolve8-neon
- **Raised:** 2026-10-17T09:23:28Z
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
- **State history:** Open (2026-10-17T09:23:28Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium)

## Observation

On 64-bit AArch64 release builds, the public safe `convolve8_horiz_neon` wrapper validates its source extent with unchecked `(h - 1) * src_stride + w + 7` at `crates/rvp9-core/src/dsp/aarch64/convolve8_neon.rs:625`. With `w=8`, `h=3`, `src_stride=1<<63`, a non-narrow filter such as `[128; 8]`, and a 16-byte source slice, the required extent wraps modulo 2^64 to 15, so the assertion passes. The fallback 8-wide NEON loop computes the second row with `src.add(1<<63)` at `:586`, outside the allocation and beyond `isize::MAX`, violating `ptr.add`'s safety requirements. The destination can be valid (`dst_stride=8`, 24-byte destination). Release profile does not enable overflow checks; checked/debug builds panic at validation instead. Expected: safe slice validation rejects an overflowing extent before any raw-pointer loop. Actual: on AArch64 release, the safe API can enter undefined behavior for valid Rust slices paired with an extreme stride. This is a source-derived arithmetic counterexample only; no runtime reproduction or normal-encoder reachability is claimed. Source inspected is unchanged from candidate product commit `474dfd99a673df85f750136f7552ac156b7a24e9`.

## Fix

<unfixed — raised only>

## Notes


### Related unchecked span guards found in source audit (2026-10-17)

The exact-candidate AArch64 SIMD review found the same unchecked span-arithmetic pattern in sibling wrappers: unscaled convolution, copy, IDCT8, integer projection row kernels, loopfilter, minmax, SAD/SAD4D, subpixel variance, subtraction, variance, VBP, forward/inverse transforms, and Hadamard. A representative release counterexample is `minmax_neon.rs`: `7 * stride + 8` wraps to 6 for `stride = usize::MAX / 7` on 64-bit targets, so a 64-byte source passes the apparent span assertion before row-one pointer construction leaves the allocation. This is static evidence only; normal encoder geometry and runtime reproduction are not claimed. The existing observation above remains the original concrete `convolve8_horiz_neon` case; no separate record was minted for the repeated span-arithmetic pattern.
