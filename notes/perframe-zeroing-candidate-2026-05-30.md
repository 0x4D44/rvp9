# Candidate: eliminate per-frame frame-buffer zeroing (O(area), content-independent)

## The gap fingerprint that points here
- dense (quantizer-00, small): rvp9 **0.29× (3.4× FASTER)**
- real video (foreman, CIF 352×288): 1.74×
- synthetic (testsrc, **720p** 1280×720): 6.2×

The gap tracks **resolution × content-cheapness**, NOT coefficient count. That is the
signature of a **fixed O(frame-area), content-independent** per-frame cost — not kernel
throughput (kernels are all at parity/winning per prior audit).

## The smoking gun
`crates/rvp9-core/src/frame.rs:265` (in `Yv12Buffer::alloc`):
```rust
// Allocate zeroed.
let buffer_alloc: Box<[u8]> = vec![0u8; total_usize].into_boxed_slice();
```
Every frame allocates a **freshly-zeroed** full YUV buffer (Y+U+V incl. 32px borders).
- 720p 4:2:0 with border=32: Y 784×1344 + 2×(UV 392×672) ≈ **1.58 MB zeroed per frame**.
- At 30fps ≈ 47 MB/s of pure zeroing + allocation churn that vpxdec avoids.
- `decode_frame` allocates a fresh `yv12` each call, wraps in `Arc` (`lib.rs:2251`),
  parks it in the ref pool. The pool holds Arcs for refcounting but the **allocation is
  fresh every frame** — it is NOT recycling the backing `Box<[u8]>`.

vpxdec: ref-counted frame-buffer **pool**, `vpx_calloc` once, recycles buffers whose
refcount hit 0; never re-zeros (reconstruction overwrites).

## Byte-exactness proof that the zero is REDUNDANT
Claim: every byte ever READ from a decode-target buffer is WRITTEN before the read.

1. **Crop/visible region** [0,crop_w)×[0,crop_h): written by reconstruction. Coding
   blocks tile [0, 8·mi_cols)×[0, 8·mi_rows) = the **aligned** region ⊇ crop. Each leaf
   block writes its full pixel extent (intra: predict+residual; inter: MC+residual; skip:
   MC/predict copy). So the full aligned region (≥ crop) is written.
2. **Crop↔aligned gap + borders**: written by `extend_frame_borders` →
   `extend_borders` (`frame.rs:601`) → `extend_plane` (`frame.rs:1009`):
   - `extend_right  = border + (y_w - crop_w)` → fills cols [crop_w, 2·border+y_w)
   - `extend_bottom = border + (y_h - crop_h)` → fills rows [crop_h+…, 2·border+y_h)
   - vertical pass replicates first/last crop row across full linesize 2·border+y_w.
   So borders + the gap are fully written.
3. **Stride round-up padding** [2·border+y_w, y_stride): NEVER read. MC halo reaches at
   most col `2·border+y_w-1` (border past aligned edge). Output reads crop only.
   Loop filter reads only reconstructed visible pixels.

⇒ Union (1)∪(2) covers every readable byte. The `vec![0]` zero is dead for output. ∎

## Safe (no-UB) way to capture it
Do NOT go to `MaybeUninit`/uninitialized reads (UB even if "always overwritten";
sanitizers/compiler may bite). Instead mirror vpxdec: a frame-buffer **pool that recycles
the `Box<[u8]>`** from a buffer whose external refcount dropped to 1, and **skips the
re-zero**. A recycled buffer holds valid (stale) `u8` data — not uninit — so reading it is
not UB; reconstruction overwrites it. Geometry must match (resolution change ⇒ realloc).

## MEASURED upper bound (notes/zerobench.rs, pinned, min-of-30, 720p geometry)
```
total buffer = 1580544 bytes (1.58 MB), y_stride=1344
A fresh vec![0;n] (alloc+zero, no write)     7436 ns   <- alloc_zeroed = lazy OS zero pages
B fresh alloc+zero + FULL overwrite        430607 ns   <- CURRENT per-frame cost (page faults!)
C pool + rezero + overwrite                 44612 ns
D pool + NO rezero + overwrite              20621 ns   <- PROPOSED
```
**Key finding: the dominant cost is FIRST-TOUCH PAGE FAULTS on freshly-allocated pages,
not the memset.** `vec![0u8;n]` (A) is cheap because Windows hands back lazily-zeroed pages;
the cost lands when reconstruction touches every byte (B) and faults in ~386 fresh pages.
A buffer pool (resident pages) removes the faults: **B − D ≈ 0.41 ms/frame @720p.**

Real-decoder representativeness: the decoder streams alloc(1.58MB)/free(1.58MB) (Arc parked
in ref pool, freed a few frames later when the ref slot is replaced). Large allocs go via
VirtualAlloc/VirtualFree ⇒ fresh decommitted (faulting) pages every frame ⇒ matches (B).

Scaling estimate (page-fault cost ∝ buffer area):
- testsrc 720p: ~410µs off 3761µs ≈ **~11%** end-to-end
- foreman CIF (~0.24MB): ~60µs off 556µs ≈ **~11%**, ratio 1.74× → ~1.55×

This is the single biggest lever found. Note: pooling alone (even keeping the zero, C=0.045ms
vs B=0.43ms) captures ~90% of the win — so the LOW-RISK variant is "pool the allocation,
keep the zero" (trivially byte-exact), and skipping the zero (D) is a smaller extra ~0.024ms.

## LANDED 2026-05-30 (Arthur greenlit "full structural batch")
Implemented the low-risk variant (recycle allocation, KEEP the zero):
- `frame.rs`: `Yv12Buffer::zero_pixels()` + `DecoderFramePool::take_recyclable(w,h,chroma)`
  — finds a slot with `Arc::strong_count==1 && ref_counts==0` (same freeness predicate as
  `acquire_free`, so reuse can NEVER alias a live ref — guards the G12 aliasing bug),
  `try_unwrap`s the `Yv12Buffer` out for reuse. Geometry-matched; mismatched left for
  `acquire_free` to drop.
- `lib.rs`: `Decoder::acquire_decode_target()` (recycle+zero, else fresh alloc + bump
  `RVP9_DECODE_TARGET_ALLOCS`); replaced both decode-target alloc sites (intra+inter).
VALIDATED: conformance 6/6 byte-exact; real_world foreman-fp1 30/30 MD5 (recycle path,
frames dropped); 225 decoder unit tests; NEW efficacy oracle
`buffer_pool_recycle.rs` → **5 fresh allocs over 107 frames** (pool size 9) ⇒ recycling
confirmed, page faults eliminated after warm-up. Mechanism = win (fewer faults = strictly
less work; can't regress). Microbench upper-bounds it at ~0.41 ms/frame @720p.

## Status / risk
- ARCHITECTURAL (buffer-pool semantics) ⇒ discuss with Arthur before landing.
- Effort: MEDIUM. Risk: pool acquire/recycle plumbing + geometry-match guard.
- Validation: conformance 6/6 MD5 (byte-exact) + interleaved-A/B rdtsc per-frame.
- Expected: removes a content-independent O(area) cost; should help most where the gap
  is worst (high-res / cheap content). Magnitude TBD by measurement (zeroing may be
  partly cheap if the allocator hands back fresh OS-zeroed pages; pooling also removes the
  alloc/first-touch page-fault cost, which the fresh path always pays).
</content>
</invoke>
