# RVP9-REQ-KIL-00106 — Remove the per-frame full-frame loop-filter adapter

- **State:** Draft
- **Priority:** Should
- **Area:** encoder/loop-filter performance
- **Raised:** 2026-08-22T18:32:54Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-08-22T18:32:54Z, raised via `deltic reqs new`)

## Statement

The normal streaming loop-filter path must not allocate a bordered full-frame adapter and copy every Y/U/V plane into and back out of it for each filtered frame. Today `D:\worktrees\rvp9-2\20260822-REV-RVP9-CDX@KILN-code-review-190928\crates\rvp9-encoder\src\loopfilter.rs:224` allocates `Yv12Buffer`, copies tight reconstruction into it at `loopfilter.rs:238`, filters it, and copies the complete result back at `loopfilter.rs:314`. The fallback covers the default encoder without a worker pool and partial final superblock rows such as 720p and 1080p.

Retain bounded scratch across frames or extend the tight-plane filter so serial and partial-row inputs avoid the adapter. Preserve byte-for-byte filtered reconstruction and bitstream output. The acceptance oracle must compare serial and threaded output before/after at a 64-row-aligned geometry plus representative 720p and 1080p partial-row geometries, and use an inspectable allocation/copy counter to prove steady-state filtered frames perform no full-frame adapter allocation or round-trip copy.

## Notes

Static review confirmed the work and its default-path trigger, but did not measure runtime impact. The existing one-worker split-edge restriction is explicit and depends on alias-safe row-bounded plane access, so this is Heavy structural performance work rather than a correctness bug. No matching open requirement was found.
