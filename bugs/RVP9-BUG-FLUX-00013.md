# RVP9-BUG-FLUX-00013 — buffer-pool byte-identity test pins stale encoder golden

- **State:** Fixed
- **Priority:** Must
- **Severity:** Low
- **Area:** encoder/buffer-pool test
- **Raised:** 2026-08-09T22:43:45Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T160905Z-f4ff83c6
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00013-run-verify-20261018T160905Z-f4ff83c6
- **Owner base:** c985d59af0c1a9079f7386726d4505a5cc865818
- **Owner fingerprint:** sha256:c7474c9a5ef9259047878e306df7dc4fbab61764b60c1e7887e4215702941a66
- **Owner since:** 2026-10-18T16:09:05Z
- **Owner until:** 2026-10-18T20:09:05Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-09T22:43:45Z, raised via `deltic bugs new`) -> Fixed (2026-08-09T22:53:08Z, deltic:auto role=fix run=fix-20260809T225244Z-p90219-n341290000-c1 branch=task/bug-RVP9-BUG-FLUX-00013-run-fix-20260809T225244Z-p90219-n341290000-c1 code=39a4b510e22f99b298a14e1b5dd4f79f81511380 gate=manual)

## Observation

Observation: cargo test -p rvp9-encoder --test w_buffer_pool a10_byte_identity_pre_vs_post_refactor fails deterministically because the current accepted encoder emits 7136 bytes while the test pins 7053 from the historical A10 refactor. Expected: the regression proves deterministic current bytes and preserves the decoder check; actual: unrelated deliberate parity changes make the stale length fail before hash and decode evidence.

## Fix

<unfixed — raised only>

## Notes
