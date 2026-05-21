# `unsafe` review checklist (Wave A G11)

Mechanical companion to the workspace lint
`clippy::undocumented_unsafe_blocks = "deny"`. Every PR that
introduces, moves, or modifies an `unsafe { ... }` block (or an
`unsafe fn` body) in a library crate (`rvp9-core`, `rvp9-decoder`,
`rvp9-encoder`) must clear every item below. The
`undocumented_unsafe_blocks = deny` workspace lint catches missing
`// SAFETY:` comments at `cargo clippy` time (locally today, and at
any future CI invocation of clippy — no CI workflow is wired up in
this sandbox repository yet); this checklist is the reviewer's mental
model when reading the comments themselves.

Anchor in the HLD: `wrk_docs/2026.05.19 - HLD - decoder hardening.md`,
§G11 deliverable 2 (p.1749-1759). Process simplified per V5 SD4-M2 —
no CI grep job, reviewer (Arthur) runs this checklist during PR
review.

## State today (worktree start)

`grep -rn "unsafe { " crates/*/src` returns only documentation hits.
**Zero live `unsafe` blocks exist in any library crate.** Wave C G4
(target_feature intrinsic calls) and G5 (shared plane writes from the
tile worker pool) introduce the first ones. This file is the
establishment commit; the discipline runs forever after.

## Per-block review steps

For every new `unsafe { ... }` block (and for every `unsafe fn`
body):

1. **SAFETY comment present and specific.** The lint enforces
   *presence*; the reviewer enforces *specificity*. The comment must
   name the invariant being asserted (e.g. "scan index `c` is bounded
   above by `seg_eob <= 1024` which is also the buffer length"), not
   restate that it's unsafe.

2. **Invariant is correct.** Trace upward from the unsafe site:
   does every caller actually uphold the precondition? For an
   `unsafe fn`, walk the call graph. For a `target_feature` intrinsic
   call, the precondition is "CPU has `cpuid` reporting feature X" —
   verify the runtime dispatch table actually gates on that.

3. **Block is minimal.** Only the operations that require `unsafe`
   live inside the block. Don't bury a 50-line function inside one
   `unsafe { ... }` — that makes step 2 unreviewable.

4. **Can it be safe instead?** Common alternatives:
   - `slice::get_unchecked(i)` → `slice[i]` and trust the optimizer.
   - `transmute<T, U>` → `From`/`TryFrom` or a tagged union.
   - Raw pointer arithmetic → split-mut + index, or `chunks_mut`.
   - `target_feature` intrinsics → keep `unsafe` (no alternative);
     gate dispatch by `is_x86_feature_detected!()` at the boundary.

5. **Miri passes for the single-threaded test exercising this
   block.** Per HLD TQ3-M8 follow-up: Miri scope is
   - single-threaded SAFETY tests (catches pointer-provenance,
     unaligned-load class issues);
   - scalar reference kernels (no intrinsics — Miri does not model
     AVX2);
   - skipped for the actual AVX2 intrinsic call sites and for the
     multi-threaded tile-pool join (Miri's threading model would
     report false positives on the legitimate cross-tile borrowed
     slices).

   For an `unsafe` block that *only* exists inside an AVX2 intrinsic
   wrapper: write a Miri-runnable scalar reference test that exercises
   the same indexing pattern with a scalar fallback. If you cannot
   articulate what a scalar test would look like, that's a flag that
   step 4 wasn't done thoroughly.

6. **Multi-threaded reasoning, if applicable.** For Wave C G5
   (shared plane writes), the SAFETY chain has named links (a)..(e)
   in the HLD. The new block's SAFETY comment must reference which
   link it depends on and where that link is established
   (`yv12.rs::split_into_tile_views` etc.).

## What does NOT need re-review

- Pre-existing `unsafe` blocks the PR didn't touch. The lint
  enforces ongoing-write discipline; archaeology of legacy unsafe
  is a separate concern.
- Comment-only `unsafe` references (`//! ... unsafe pointer dance
  ...`) — these are documentation and don't bind any contract.

## Why no CI grep job

V3 of the HLD proposed `unsafe:` in the PR title as the review
trigger; V4 swapped to a CI grep job. Round 4 (HLD §G11 SD4-M2) found
the CI grep had three concrete defects (undefined
`${PR_BODY_FILE}`, regex matches comments + doctests + tests, treats
refactors as new additions). V5 dropped it entirely: the
`undocumented_unsafe_blocks` lint already does the load-bearing work
(no SAFETY comment → CI fails). Reviewer discipline runs the
checklist above. The two-person team size (Arthur + Claude) supports
the reviewer-discipline model; revisit if the team grows.
