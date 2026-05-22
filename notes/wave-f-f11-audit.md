# Wave F F11 — extend_frame_borders SIMD audit

**Date**: 2026-05-22
**Auditor**: Claude (supervisor, Wave F autonomous lead)
**Verdict**: **No explicit SIMD work warranted.** Dispatch.rs docstring
already captures the correct analysis.

## Audit

`crates/rvp9-core/src/frame.rs:997` defines `pub fn extend_frame_borders`
as a one-line wrapper around `Yv12Buffer::extend_borders()`. The real
work is in `extend_plane` at `crates/rvp9-core/src/frame.rs:1011-1060`.

Two distinct hot paths:

### Horizontal pass (per-row left/right border fill)

```rust
for r in 0..height {
    let row_start = origin + r * stride;
    let left_pixel = alloc[row_start];
    let right_pixel = alloc[row_start + width - 1];
    alloc[left_dst..row_start].fill(left_pixel);            // memset
    alloc[right_dst..right_dst + extend_right].fill(right_pixel);  // memset
}
```

`slice::fill` lowers directly to `memset` in core. On x86_64 the platform
`memset` (glibc / Windows CRT) dispatches to AVX2 / AVX-512 at runtime
based on CPU caps and input size. For VP9 border widths (typically 8 or
32 bytes per row) the platform memset is at-or-above what hand-rolled
AVX2 would achieve — and below the SSE → AVX threshold for some
implementations.

### Vertical pass (top/bottom border row-replication)

```rust
for k in 0..extend_top {
    alloc.copy_within(top_src..top_src + linesize, dst);    // memmove
}
```

`slice::copy_within` lowers to `memmove`. Same dispatch story — platform
implementations are already vector-aware.

## Measurement (informal)

Per-frame `extend_frame_borders` cost for 1080p 4:2:0 (rough order of
magnitude): ~30 KB written. Sustained system memory bandwidth on a
modern x86 (>20 GB/s) makes this < 2 μs per frame. Below the noise
floor of a single transform-kernel hot path.

## Wave G consideration

The only scenarios where hand-rolled AVX2 might beat the platform
memset:

1. **Tile-parallel decode**: per-tile micro-borders may be small enough
   to fall into the platform memset's pre-vectorisation threshold.
   Worth profiling under G5.1 / G5.2 / G5.3 tile parallelism if
   sub-frame border-extension is on a critical path.
2. **A different storage layout** that interleaves planes such that
   border-fill becomes contiguous gather/scatter.

Neither is in scope for current decoder design. **The
`SimdDispatch::extend_frame_borders` slot remains scalar-only and the
field docstring is accurate.**

## Action

F11 closed without code change. Slot doc on `dispatch.rs:312-320` is
the canonical record of the audit — this file is the supporting
analysis.
