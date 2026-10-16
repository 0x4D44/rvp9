# RVP9-BUG-KIL-00098 — WebM track discovery has unbounded pending-container growth

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** io/webm track discovery
- **Raised:** 2026-08-22T14:34:14Z
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
- **State history:** Open (2026-08-22T14:34:14Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T01:57:18Z, deltic:auto role=fix run=fix-20261016T014916Z-059eed45 branch=task/bug-RVP9-BUG-KIL-00098-run-fix-20261016T014916Z-059eed45 code=d7966e45956ad70dae2088a9901dc504adf45042 gate=manual)

## Observation

find_vp9_video_track at crates/rvp9-io/src/container.rs:962-999 scans a container fully while pushing every recognized EBML, Segment, or Tracks child onto an unbounded Vec. A malformed input made from many empty recognized siblings can retain millions of (start,end,flag) tuples and exhaust memory before returning VP9 video track not found. The later packet-walk depth cap cannot protect this earlier allocation, while the streaming finder processes bounded depth and returns on the first match. Expected: malformed input returns a bounded BadEbml error rather than amplifying memory. Fix by processing child containers immediately in document order with an explicit depth/pending budget, or cap the pending vector. Add a generated wide-container regression and parity check against PacketReader. Estimated effort: medium. Confirmed independently by performance, security, reliability, the lead, and a skeptic; no tests or application were run.

## Fix

<unfixed — raised only>

## Notes
