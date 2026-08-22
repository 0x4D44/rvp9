# RVP9-BUG-KIL-00094 — WebM accepts unknown sizes on schema-bounded elements

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** io/webm parser
- **Raised:** 2026-08-22T14:33:35Z
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
- **State history:** Open (2026-08-22T14:33:35Z, raised via `deltic bugs new`)

## Observation

EBML permits unknown sizes only for Master elements whose schema enables them. crates/rvp9-io/src/container.rs:645-663, :779-877, and :1089-1129 apply parent-end semantics to unknown-size SimpleBlock and Block binary leaves, so a malformed block absorbs following elements into one returned packet; the streaming reader can allocate the entire remainder. The in-memory track finder at :973-994 also accepts unknown-size TrackEntry even though the streaming finder rejects it at :707-710. Expected: reject unknown sizes for Block, SimpleBlock, TrackEntry, and every other element not explicitly allowed by the Matroska schema. Fix by preserving the unknown marker through dispatch and enforcing an element-ID allowlist before parsing payloads. Add equivalent streaming/in-memory regressions for an unknown-size block followed by a sibling and for unknown-size TrackEntry. Estimated effort: small to medium. Confirmed independently by correctness, reliability, the lead, and a devil advocate; no tests or application were run.

## Fix

<unfixed — raised only>

## Notes
