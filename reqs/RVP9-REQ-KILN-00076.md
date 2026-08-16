# RVP9-REQ-KILN-00076 — Make spatial decoder-visible state a single exchange unit

- **State:** Draft
- **Priority:** Should
- **Area:** encoder/spatial state
- **Raised:** 2026-08-16T08:19:57Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-08-16T08:19:57Z, raised via `deltic reqs new`)

## Statement

Represent decoder-visible spatial state as one owned value inside StreamEncoder and exchange that value atomically with the coordinator. D:\worktrees\rvp9-2\20260816-REV-RVP9-CDX@KILN-code-review-085355\crates\rvp9-encoder\src\frame_loop.rs:1916 defines nine shared fields, while frame_loop.rs:2406 manually swaps each matching field. The parallel lists make a future decoder-visible field easy to omit without a compiler error. Keep rate-control and analysis state layer-local and add a state-roundtrip test that covers every decoder-visible field.

## Notes
