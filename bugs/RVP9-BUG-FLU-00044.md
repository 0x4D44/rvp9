# RVP9-BUG-FLU-00044 — Malformed WebM parent-crossing header panics in overflow-checked packet reader

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** io/webm
- **Raised:** 2026-10-16T14:24:03Z
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
- **State history:** Open (2026-10-16T14:24:03Z, raised via `deltic bugs new`) -> Fixed (2026-10-18T19:50:22Z, deltic:auto role=fix run=fix-20261018T194654Z-740c1eec branch=task/bug-RVP9-BUG-FLU-00044-run-fix-20261018T194654Z-740c1eec code=52a15ed22037ed5fb7b89aa16b452a1d8d877898 gate=manual)

## Observation

At exact candidate code ebca64b0, public read_webm_packets called with bytes [18,53,80,67,81,18,53,80,67,ff] (hex) panics at container.rs:1077:13 attempt to subtract with overflow. Outer Segment declares a one-byte payload; nested unknown-size Segment header extends beyond that parent. Tiny diagnostic includes the actual unchanged crates/rvp9-io/src/container.rs and calls the public function under catch_unwind; two independent executions of the overflow-checked Rust1.95 build produce the same panic. find_vp9_video_track reads child ID/vint against full slice, advances pos beyond parent end, then computes end-pos before checking the parent boundary. Expected: malformed input returns ContainerError in every supported build. Actual: overflow-checked/debug API panics. Optimized rustc -O panic=abort with default unchecked arithmetic returns BadEbml(payload size overflow), so no optimized release crash, out-of-bounds read or memory corruption is claimed. Streaming PacketReader::open returns VP9 video track not found for the same bytes. Root cause existed before recent width-growth fix; distinct from KIL-00093/94/98. Raw source/probes/build outputs are preserved under release-readiness candidate-20261016 evidence. No product source changes or test suppressions; separate bug owner owns any repair.

## Fix

<unfixed — raised only>

## Notes
