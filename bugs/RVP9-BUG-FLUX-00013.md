# RVP9-BUG-FLUX-00013 — buffer-pool byte-identity test pins stale encoder golden

- **State:** Closed
- **Priority:** Must
- **Severity:** Low
- **Area:** encoder/buffer-pool test
- **Raised:** 2026-08-09T22:43:45Z
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
- **State history:** Open (2026-08-09T22:43:45Z, raised via `deltic bugs new`) -> Fixed (2026-08-09T22:53:08Z, deltic:auto role=fix run=fix-20260809T225244Z-p90219-n341290000-c1 branch=task/bug-RVP9-BUG-FLUX-00013-run-fix-20260809T225244Z-p90219-n341290000-c1 code=39a4b510e22f99b298a14e1b5dd4f79f81511380 gate=manual) -> Closed (2026-10-18T16:16:24Z, deltic:manual role=verify run=verify-20261018T160905Z-f4ff83c6 verified_fix_run=fix-20260809T225244Z-p90219-n341290000-c1 verdict=close)

## Observation

Observation: cargo test -p rvp9-encoder --test w_buffer_pool a10_byte_identity_pre_vs_post_refactor fails deterministically because the current accepted encoder emits 7136 bytes while the test pins 7053 from the historical A10 refactor. Expected: the regression proves deterministic current bytes and preserves the decoder check; actual: unrelated deliberate parity changes make the stale length fail before hash and decode evidence.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T16:16:24Z, deltic:auto run=verify-20261018T160905Z-f4ff83c6 verified_fix_run=fix-20260809T225244Z-p90219-n341290000-c1 verdict=close)

Verifier note: Manual verify: Independent session01a14fa5, not fixer. Ledger39a4b510 absent; reachable matchingf5d2de1c superseded byb3817a33/173d76b7. Original exact command1 pass; full pool suite2 pass twice, stable6446/hash20793807d02f6e58 with sibling decoder>=12frames. Restore pre-fix golden7053/0d37a3ba82589d9a: selected1 length assertion fails6446vs7053. Exact source restored/diff empty; restored2 pass, lead2 pass. No historical7136 reproduction/bisect or pool-equivalence/quality claim. Gates/limits/provenance in2026.10.18 journal.

## Notes
