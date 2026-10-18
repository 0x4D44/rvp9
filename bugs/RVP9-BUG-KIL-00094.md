# RVP9-BUG-KIL-00094 — WebM accepts unknown sizes on schema-bounded elements

- **State:** Closed
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
- **State history:** Open (2026-08-22T14:33:35Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T01:34:26Z, deltic:auto role=fix run=fix-20261016T012403Z-bce91f0c branch=task/bug-RVP9-BUG-KIL-00094-run-fix-20261016T012403Z-bce91f0c code=8aca657f32a24406b047d13b2010431bb3497930 gate=manual) -> Closed (2026-10-18T16:05:23Z, deltic:manual role=verify run=verify-20261018T160147Z-fb24c539 verified_fix_run=fix-20261016T012403Z-bce91f0c verdict=close)

## Observation

EBML permits unknown sizes only for Master elements whose schema enables them. crates/rvp9-io/src/container.rs:645-663, :779-877, and :1089-1129 apply parent-end semantics to unknown-size SimpleBlock and Block binary leaves, so a malformed block absorbs following elements into one returned packet; the streaming reader can allocate the entire remainder. The in-memory track finder at :973-994 also accepts unknown-size TrackEntry even though the streaming finder rejects it at :707-710. Expected: reject unknown sizes for Block, SimpleBlock, TrackEntry, and every other element not explicitly allowed by the Matroska schema. Fix by preserving the unknown marker through dispatch and enforcing an element-ID allowlist before parsing payloads. Add equivalent streaming/in-memory regressions for an unknown-size block followed by a sibling and for unknown-size TrackEntry. Estimated effort: small to medium. Confirmed independently by correctness, reliability, the lead, and a devil advocate; no tests or application were run.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T16:05:23Z, deltic:auto run=verify-20261018T160147Z-fb24c539 verified_fix_run=fix-20261016T012403Z-bce91f0c verdict=close)

Verifier note: Manual verify: Independent session01a14fa5, not fixer; base283ea929 includes8aca657f. Original forbidden unknown-size element observation: IO error_paths75 pass. Disable only shared schema allowlist guard:4 execute, forbidden TrackEntry/block/other-ID tests assertion-fail in both readers; legal Segment control passes. Reversal permits block payloads to absorb siblings. Exact bytes restored/diff empty; restored75 pass. Lead source review and executable rerun4 pass. Shared repository gate limits/raw proof in2026.10.18 verification journal.

## Notes
