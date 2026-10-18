# RVP9-BUG-KIL-00093 — Unknown-size WebM Clusters are mistaken for nested containers

- **State:** Fixed
- **Priority:** Should
- **Severity:** High
- **Area:** io/webm packet reader
- **Raised:** 2026-08-22T14:33:24Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T155636Z-287856fc
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00093-run-verify-20261018T155636Z-287856fc
- **Owner base:** 283ea9293f87eb88d82f0aa1f1219b2420bb5e57
- **Owner fingerprint:** sha256:3b03b9413bc7d3e9cded6b155eb5effb68f1eba4de70b5ca4c1ec9f202f6b819
- **Owner since:** 2026-10-18T15:56:36Z
- **Owner until:** 2026-10-18T19:56:36Z
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
