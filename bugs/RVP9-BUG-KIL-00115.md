# RVP9-BUG-KIL-00115 — Concurrent throughput writers can corrupt the shared CSV ledger

- **State:** Fixed
- **Priority:** Could
- **Severity:** Medium
- **Area:** bench/throughput CSV
- **Raised:** 2026-08-22T19:31:49Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T181112Z-64c7264d
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00115-run-verify-20261018T181112Z-64c7264d
- **Owner base:** bb00e6ded9e6d75d595b69b9dc68ecaf11d27e63
- **Owner fingerprint:** sha256:68a21226da912cb023abfbc105d47bef0f5b765bba241839191c3eb31d656a45
- **Owner since:** 2026-10-18T18:11:12Z
- **Owner until:** 2026-10-18T22:11:12Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T19:31:49Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-10-16T06:43:12Z, deltic:auto role=fix run=fix-20261016T062937Z-fc9f9cba branch=task/bug-RVP9-BUG-KIL-00115-run-fix-20261016T062937Z-fc9f9cba code=0feeb38012da67e486be4c2eea11883783ffb282 gate=manual)

## Observation

crates/rvp9-bench/src/throughput.rs:439-471 checks whether the CSV path exists before opening it in append mode, then writes a header and each row with separate writes. Concurrent benchmark processes can both observe a missing file and emit duplicate headers; their row writes are not protected as one transaction. read_csv later silently skips malformed rows, so lost or interleaved evidence may go unnoticed. Expected: concurrent appenders preserve one header and complete rows. Actual: initialization and append are check-then-use races. Serialize writers with an exclusive file lock or append through an atomic ledger helper, and add a concurrent first-writer regression. Confirmed by independent source review only; no tests or binaries were run.

## Fix

<unfixed — raised only>

## Notes
