# RVP9-BUG-FLUX-00006 — Buffer-pool byte golden lags accepted encoder output

- **State:** Closed
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
- **State history:** Open (2026-08-07T10:58:05Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) → Fixed (2026-08-07, `f7f5b47e1176aabcdcc336178a150a1d00a7e27e`, fixed by Codex) -> Fixed (2026-10-19, provenance reconciliation by Codex only; original fixing attribution retained above; code=92a70d745e69cd680040d62a88659f2163ab2777) → Closed (2026-10-19T11:43:45Z, deltic:manual role=verify run=verify-20261019T114003Z-82cc144e verified_fix_run=manual:92a70d745e69cd680040d62a88659f2163ab2777 verdict=close)

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

### Provenance reconciliation (2026-10-19)

Authoritative fixing commit: `92a70d745e69cd680040d62a88659f2163ab2777` (test(enc): refresh buffer-pool golden (RVP9-BUG-FLUX-00006)).
The original reference `f7f5b47e1176aabcdcc336178a150a1d00a7e27e` does not resolve locally and is retained above for audit.
Historical Fixed ledger commit `5b91a3ab9112b332bc87c34bcbd62b03bd7dcceb` has this fixing commit as its immediate parent.
All original old/new length and FNV constants match exact candidate patch in w_buffer_pool.rs.
The recovered commit is an ancestor of `origin/main`. This is a metadata correction only;
no regression was rerun and no new fix or verification verdict is recorded.

### Verification summary (2026-10-19T11:43:45Z, deltic:auto run=verify-20261019T114003Z-82cc144e verified_fix_run=manual:92a70d745e69cd680040d62a88659f2163ab2777 verdict=close)

Verifier note: Manual verify: Independent verification: current tree w_buffer_pool suite passes 2/2 and reports its later accepted golden 6446 bytes / FNV-1a 0x20793807d02f6e58; the sibling decoder accepts all 12 frames. To verify FLUX-00006 at accepted fix 92a70d745e69cd680040d62a88659f2163ab2777 before later stream changes, a clean Deltic-managed tree reported 7053 bytes / 0x0d37a3ba82589d9a and both tests passed. Fails-before: changed only the golden constants back to pre-fix 7057 / 0xac2b85e46b0c37fc; the selected byte test emitted actual 7053 / 0x0d37a3ba82589d9a and failed the length assertion (7053 vs...

## Notes
