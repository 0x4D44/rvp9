# RVP9-BUG-KIL-00119 — Decoder emits invalid Y4M across dynamic stream changes

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** cli/rvp9dec output correctness
- **Raised:** 2026-08-22T20:44:02Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T163710Z-80f4c01e
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00119-run-verify-20261018T163710Z-80f4c01e
- **Owner base:** bc5a3e1094d41cba0efacfaeed389508ba211ee8
- **Owner fingerprint:** sha256:d7892924040f92bda1d4657d25d3459cf7111ab66385963dabe983b05982228e
- **Owner since:** 2026-10-18T16:37:10Z
- **Owner until:** 2026-10-18T20:37:10Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T20:44:02Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-10-16T03:31:12Z, deltic:auto role=fix run=fix-20261016T032211Z-01b803ca branch=task/bug-RVP9-BUG-KIL-00119-run-fix-20261016T032211Z-01b803ca code=bebdf8eb5ceaab40999fa602124e4945bb413931 gate=manual)

## Observation

rvp9dec writes one Y4M header from the first displayed frame, then writes later frame payloads without checking width, height, or chroma. The decoder supports mid-stream geometry and chroma changes, so later payload sizes can disagree with the latched header and corrupt downstream parsing while the CLI exits successfully. Expected: reject a changed tuple before writing it, naming the frame and both formats, or implement normalization. Confirmed statically at crates/rvp9-cli/src/bin/rvp9dec.rs:430,582-589,609-627 with supporting decoder paths; no app or test was run.

## Fix

<unfixed — raised only>

## Notes
