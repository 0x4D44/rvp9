# RVP9-BUG-FLU-00051 — decode_coefs accepts a scan shorter than its transform size

- **State:** Closed
- **Priority:** Must
- **Severity:** High
- **Area:** core/entropy detokenize
- **Raised:** 2026-10-17T20:46:58Z
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
- **State history:** Open (2026-10-17T20:46:58Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium) -> Fixed (2026-10-18T16:53:35Z, deltic:auto role=fix run=fix-20261018T164502Z-6acc6ec1 branch=task/bug-RVP9-BUG-FLU-00051-run-fix-20261018T164502Z-6acc6ec1 code=cbc5a550a8956ccc13455ce5ec304fb82428d755 gate=manual) -> Closed (2026-10-18T17:11:06Z, deltic:manual role=verify run=verify-20261018T170410Z-db0285fe verified_fix_run=fix-20261018T164502Z-6acc6ec1 verdict=close)

## Observation

The public safe crates/rvp9-core/src/entropy/detokenize.rs::decode_coefs accepts ScanOrder and TxSize independently. A safe caller can pass default_scan_4x4(), TxSize::Tx32x32, and seg_eob = 17; the entry assertions accept seg_eob because it is within Tx32x32. The only scan-length guard at line 562 is debug_assert!, while the loop uses get_unchecked at lines 610 and 640. With a token stream that continues through scan position 16, the safe call indexes past the 16-entry scan table; continuing can also exceed the matching neighbor table. Expected: reject a scan/transform size mismatch before unchecked access. Actual: the safe API's release path trusts the size relation and can read outside the static scan table. This is a residual size-mismatch variant of the closed RVP9-BUG-KILN-00021: privatizing ScanOrder fields prevents forged tables but does not prevent passing a valid scan for the wrong TxSize. Production decoder callers select matching tables; this report is limited to the public API. Source-derived; no runtime reproduction was attempted.

## Fix

`decode_coefs` now unconditionally checks that the scan and inverse-scan lengths equal `tx_size.n_coeffs()` and that the neighbor table has `(n_coeffs + 1) * MAX_NEIGHBORS` entries, before the EOB fast path or any unchecked access. The public panic contract and the unchecked-access proof comment now state these requirements. The regression uses the reported 4×4 scan with `Tx32x32` and an EOB-at-zero stream, so the pre-fix call safely returned without indexing the scan while the test still proved the missing rejection.

### Verification summary (2026-10-18T17:11:06Z, deltic:auto run=verify-20261018T170410Z-db0285fe verified_fix_run=fix-20261018T164502Z-6acc6ec1 verdict=close)

Verifier note: Manual verify: Independent verification in a fresh origin/main worktree: fix cbc5a550 is an ancestor of tested HEAD. Baseline/restored detokenize suite passed 55 tests. Removing the three scan-size guards made the retained public mismatch regression fail with "test did not panic as expected"; immediate-EOB input proves missing rejection without unchecked access. Exact source restored and scoped diff clean. Optimized dev tests only; no separate release or Miri run.

## Notes
