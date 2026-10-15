# RVP9-BUG-KIL-00093 — Unknown-size WebM Clusters are mistaken for nested containers

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** io/webm packet reader
- **Raised:** 2026-08-22T14:33:24Z
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
- **State history:** Open (2026-08-22T14:33:24Z, raised via `deltic bugs new`) -> Fixed (2026-10-15T23:02:21Z, deltic:auto role=fix run=fix-20261015T225626Z-402753f6 branch=task/bug-RVP9-BUG-KIL-00093-run-fix-20261015T225626Z-402753f6 code=0359a602e11b758090db28644b08b19f77713dcf gate=manual)

## Observation

RFC 8794 section 6.2 requires an unknown-sized element to end when the next valid element shares its direct parent, and RFC 9559 section 5.1.3 permits unknown-size Cluster elements. crates/rvp9-io/src/container.rs:645-663 and :1089-1113 instead assign every unknown element the parent end. Consecutive valid Cluster siblings are therefore parsed as recursively nested Clusters until MAX_EBML_NESTING_DEPTH rejects a normal long stream. Expected: each unknown-size Cluster ends at the next Segment-level sibling. Actual: both streaming and in-memory readers consume nesting budget and eventually return BadEbml. Fix both walkers with schema/path-aware sibling-boundary handling while retaining the depth cap for genuine nesting. Add parity regressions with more than 32 consecutive unknown-size Clusters. Estimated effort: medium. Confirmed independently by the lead, reliability reviewer, security skeptic, and devil advocate; no tests or application were run.

## Fix

<unfixed — raised only>

## Notes
