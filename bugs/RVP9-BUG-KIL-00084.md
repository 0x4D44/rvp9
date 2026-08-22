# RVP9-BUG-KIL-00084 — F02 strict comparison omits three advertised fields

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/f02 oracle
- **Raised:** 2026-08-22T13:20:52Z
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
- **State history:** Open (2026-08-22T13:20:52Z, raised via `deltic bugs new`)

## Observation

`f02-block-oracle` advertises `q_index`, `tx_size`, and
`luma_eob_nonzero` as strict fields at
`crates/rvp9-harness/src/bin/f02_block_oracle.rs:884-887`. However,
`live_field_mismatches` at `crates/rvp9-harness/src/f02_schema.rs:1256-1279`
never compares those three fields. `OracleComparison::require_match` derives
the process result solely from that mismatch list, so two aligned records that
differ only in any of these advertised strict fields count as a match and a
strict run can exit successfully.

Expected: every advertised strict field contributes to the mismatch count and
appears in paired diagnostics. Actual: quantizer, transform-size, and luma-EOB
divergence is silently accepted.

## Fix

<unfixed — raised only>

## Notes

Confirmed by static source inspection and an independent adversarial review.
No test or oracle run was performed in this code-review pass.
