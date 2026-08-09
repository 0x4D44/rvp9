# RVP9-BUG-FLUX-00013 — buffer-pool byte-identity test pins stale encoder golden

- **State:** Open
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
- **State history:** Open (2026-08-09T22:43:45Z, raised via `deltic bugs new`)

## Observation

Observation: cargo test -p rvp9-encoder --test w_buffer_pool a10_byte_identity_pre_vs_post_refactor fails deterministically because the current accepted encoder emits 7136 bytes while the test pins 7053 from the historical A10 refactor. Expected: the regression proves deterministic current bytes and preserves the decoder check; actual: unrelated deliberate parity changes make the stale length fail before hash and decode evidence.

## Fix

<unfixed — raised only>

## Notes
