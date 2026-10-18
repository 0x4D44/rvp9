# RVP9-BUG-KIL-00084 — F02 strict comparison omits three advertised fields

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/f02 oracle
- **Raised:** 2026-08-22T13:20:52Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T162838Z-907066ed
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00084-run-verify-20261018T162838Z-907066ed
- **Owner base:** 19a0e5d901745b2344add62fed51ab341be9f8aa
- **Owner fingerprint:** sha256:1741df1edd10bbefba0c424f182654eabce35208237fb9401566effc5582118e
- **Owner since:** 2026-10-18T16:28:38Z
- **Owner until:** 2026-10-18T20:28:38Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T13:20:52Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T04:47:41Z, deltic:auto role=fix run=fix-20261016T044334Z-7aa5f4a5 branch=task/bug-RVP9-BUG-KIL-00084-run-fix-20261016T044334Z-7aa5f4a5 code=cf7305e6c3454b17e73803cbf56826841d6edb5f gate=manual)

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
