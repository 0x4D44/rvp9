# RVP9-BUG-KIL-00090 — F02 sidecar parser silently overwrites duplicate fields

- **State:** Open
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
- **State history:** Open (2026-08-22T13:20:53Z, raised via `deltic bugs new`)

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

## Notes

Reject a second occurrence of every known field, or use a schema-aware parser
with an explicit duplicate-key error. Add one identity-key and one decision-key
regression. Confirmed statically; no sidecar was parsed in this pass.
