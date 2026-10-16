# RVP9-BUG-KIL-00098 — WebM track discovery has unbounded pending-container growth

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** io/webm track discovery
- **Raised:** 2026-08-22T14:34:14Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T014916Z-059eed45
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00098-run-fix-20261016T014916Z-059eed45
- **Owner base:** f5a8104621cef90f64f98a29387e80a7c3d2aefc
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T01:49:16Z
- **Owner until:** 2026-10-16T05:49:16Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T14:34:14Z, raised via `deltic bugs new`)

## Observation

find_vp9_video_track at crates/rvp9-io/src/container.rs:962-999 scans a container fully while pushing every recognized EBML, Segment, or Tracks child onto an unbounded Vec. A malformed input made from many empty recognized siblings can retain millions of (start,end,flag) tuples and exhaust memory before returning VP9 video track not found. The later packet-walk depth cap cannot protect this earlier allocation, while the streaming finder processes bounded depth and returns on the first match. Expected: malformed input returns a bounded BadEbml error rather than amplifying memory. Fix by processing child containers immediately in document order with an explicit depth/pending budget, or cap the pending vector. Add a generated wide-container regression and parity check against PacketReader. Estimated effort: medium. Confirmed independently by performance, security, reliability, the lead, and a skeptic; no tests or application were run.

## Fix

<unfixed — raised only>

## Notes
