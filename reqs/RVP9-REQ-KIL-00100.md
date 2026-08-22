# RVP9-REQ-KIL-00100 — Share WebM structural parsing across reader modes

- **State:** Draft
- **Priority:** Should
- **Area:** io/webm parser
- **Raised:** 2026-08-22T14:34:36Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-08-22T14:34:36Z, raised via `deltic reqs new`)

## Statement

The streaming PacketReader and in-memory read_webm_packets paths must share one schema-aware EBML structural decoder for element sizes, parent/sibling boundaries, traversal order, depth/resource limits, track selection, and block validation, while retaining separate streaming packet ownership and zero-copy PacketRange outputs. Current independent implementations at crates/rvp9-io/src/container.rs:591-877 and :882-1206 already diverge: only streaming rejects unknown-size TrackEntry, only in-memory accumulates an unbounded pending-container vector, and unknown-size Cluster boundary fixes must otherwise be duplicated. Implement incrementally through shared parsing primitives or events rather than a wholesale rewrite. The acceptance oracle must feed identical valid and malformed fixtures to both public APIs and compare selected VP9 packet bytes plus equivalent error classes, including unknown-size Cluster siblings, forbidden unknown-size elements, multiple recognized siblings, lacing, and depth limits.

## Notes
