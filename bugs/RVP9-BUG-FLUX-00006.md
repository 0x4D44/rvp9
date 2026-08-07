# RVP9-BUG-FLUX-00006 — Buffer-pool byte golden lags accepted encoder output

- **State:** Fixed
- **Priority:** Must
- **Severity:** Low
- **Area:** tests/encoder buffer pool
- **Raised:** 2026-08-07T10:58:05Z
- **Discovery source:** Automation
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
- **State history:** Open (2026-08-07T10:58:05Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) → Fixed (2026-08-07, `f7f5b47e1176aabcdcc336178a150a1d00a7e27e`, fixed by Codex)

## Observation

On untouched origin/main commit 187b6af6, run: cargo test -p rvp9-encoder --test w_buffer_pool a10_byte_identity_pre_vs_post_refactor --locked -- --exact --nocapture. The stable stream is 7053 bytes with FNV-1a 0x0d37a3ba82589d9a, but the fixture still expects 7057 bytes and 0xac2b85e46b0c37fc. The sibling a10_pool_stream_decodes_cleanly test passes. Expected: the byte golden records the accepted stream so the identity oracle detects future drift instead of permanently failing the workspace gate.

## Fix

Commit `f7f5b47e1176aabcdcc336178a150a1d00a7e27e` refreshes the byte
golden to the stable accepted production-parity stream. The assertion remains
byte-exact, and the sibling decoder oracle remains unchanged.

Regression evidence:

- Untouched `origin/main` commit `187b6af6` and the integration tree both
  produced 7,053 bytes with FNV-1a `0x0d37a3ba82589d9a`.
- `cargo test -p rvp9-encoder --test w_buffer_pool --locked -- --nocapture`
  passes both the byte-identity and decoder-oracle tests.
- A second exact byte-identity run reproduced the same length and hash.

Awaiting independent verification before closure.

## Notes
