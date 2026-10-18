# RVP9-BUG-KIL-00090 — F02 sidecar parser silently overwrites duplicate fields

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** harness/f02 schema
- **Raised:** 2026-08-22T13:20:53Z
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
- **State history:** Open (2026-08-22T13:20:53Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T08:30:00Z, deltic:auto role=fix run=fix-20261016T082026Z-135aa26a branch=task/bug-RVP9-BUG-KIL-00090-run-fix-20261016T082026Z-135aa26a code=0ef46d9c724f69c97ba73587ff1695ec371c00ef gate=manual) -> Closed (2026-10-18T17:09:54Z, deltic:manual role=verify run=verify-20261018T170758Z-4264e2c4 verified_fix_run=fix-20261016T082026Z-135aa26a verdict=close)

## Observation

The F.0.2 sidecar parser assigns each known field whenever it appears at
`crates/rvp9-harness/src/f02_schema.rs:119-187` without recording that the key
was already seen. A duplicate `frame_idx`, `mode`, or other known key is
accepted with last-value-wins semantics. `parse_decision_stream` at lines
282-290 adds no duplicate check before the record enters the differential
comparison.

Expected: malformed oracle records with duplicate schema keys fail parsing.
Actual: the later occurrence silently changes the value compared by the
strict oracle, so corrupt or incompatible sidecar output can mask its first
value.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T17:09:54Z, deltic:auto run=verify-20261018T170758Z-4264e2c4 verified_fix_run=fix-20261016T082026Z-135aa26a verdict=close)

Verifier note: Manual verify: Independent Codex verification on d7cf0fb0 containing fix0ef46d9c (merge-base ancestor proof). cargo test -p rvp9-harness --lib f02_schema:: selected24 PASS before/after. Matrix rejects identical and differing duplicate known fields including frame_idx/mode across inter/intra; stream propagates Malformed duplicate schema field; valid roundtrips/unknown extras preserved. Removing only set_once duplicate guard selected2 duplicate regressions, both FAIL expected assertions: matrix lists accepted keys and stream returnedOk BlockDecision. Exact source bytes restoredfinally; scoped di...

## Notes

Reject a second occurrence of every known field, or use a schema-aware parser
with an explicit duplicate-key error. Add one identity-key and one decision-key
regression. Confirmed statically; no sidecar was parsed in this pass.
