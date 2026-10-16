# RVP9-BUG-KIL-00115 — Concurrent throughput writers can corrupt the shared CSV ledger

- **State:** Open
- **Priority:** Could
- **Severity:** Medium
- **Area:** bench/throughput CSV
- **Raised:** 2026-08-22T19:31:49Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T062937Z-fc9f9cba
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00115-run-fix-20261016T062937Z-fc9f9cba
- **Owner base:** 92c6a96b794f6f93d8d7c6a6d168f240832a2e33
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T06:29:37Z
- **Owner until:** 2026-10-16T10:29:37Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T19:31:49Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

crates/rvp9-bench/src/throughput.rs:439-471 checks whether the CSV path exists before opening it in append mode, then writes a header and each row with separate writes. Concurrent benchmark processes can both observe a missing file and emit duplicate headers; their row writes are not protected as one transaction. read_csv later silently skips malformed rows, so lost or interleaved evidence may go unnoticed. Expected: concurrent appenders preserve one header and complete rows. Actual: initialization and append are check-then-use races. Serialize writers with an exclusive file lock or append through an atomic ledger helper, and add a concurrent first-writer regression. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
