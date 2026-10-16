# RVP9-BUG-KIL-00119 — Decoder emits invalid Y4M across dynamic stream changes

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** cli/rvp9dec output correctness
- **Raised:** 2026-08-22T20:44:02Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T032211Z-01b803ca
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00119-run-fix-20261016T032211Z-01b803ca
- **Owner base:** bdc62da33b06e39440fe9b4dffad2e17e28643d3
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T03:22:11Z
- **Owner until:** 2026-10-16T07:22:11Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T20:44:02Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

rvp9dec writes one Y4M header from the first displayed frame, then writes later frame payloads without checking width, height, or chroma. The decoder supports mid-stream geometry and chroma changes, so later payload sizes can disagree with the latched header and corrupt downstream parsing while the CLI exits successfully. Expected: reject a changed tuple before writing it, naming the frame and both formats, or implement normalization. Confirmed statically at crates/rvp9-cli/src/bin/rvp9dec.rs:430,582-589,609-627 with supporting decoder paths; no app or test was run.

## Fix

<unfixed — raised only>

## Notes
