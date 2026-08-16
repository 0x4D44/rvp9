# RVP9-REQ-KILN-00074 — Reuse temporal-filter working buffers across macroblocks

- **State:** Draft
- **Priority:** Should
- **Area:** encoder/temporal filter
- **Raised:** 2026-08-16T08:19:40Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-08-16T08:19:40Z, raised via `deltic reqs new`)

## Statement

Replace per-macroblock and per-candidate heap allocations in the ALTREF temporal filter with bounded reusable scratch owned per row or worker. D:\worktrees\rvp9-2\20260816-REV-RVP9-CDX@KILN-code-review-085355\crates\rvp9-encoder\src\temporal_filter.rs:1287 allocates accumulator and count vectors for every macroblock; temporal_filter.rs:897 allocates three difference vectors for every candidate application; temporal_filter.rs:694 and :758 allocate predictor staging. Preserve byte-for-byte filter results and add an allocation-count or reusable-capacity oracle at representative 1080p geometry.

## Notes
