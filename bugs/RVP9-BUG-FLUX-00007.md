# RVP9-BUG-FLUX-00007 — GOP byte golden lags accepted encoder output

- **State:** Fixed
- **Priority:** Must
- **Severity:** Low
- **Area:** tests/encoder GOP topology
- **Raised:** 2026-08-07T11:00:30Z
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
- **State history:** Open (2026-08-07T11:00:30Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) → Fixed (2026-08-07, `bb188d336838e30fd600795e318397c99c340780`, fixed by Codex) -> Fixed (2026-10-19, provenance reconciliation by Codex only; original fixing attribution retained above; code=f613c0fbe60bae8160bdd86b5796354810882023)

## Observation

On untouched origin/main commit 187b6af6, run: cargo test -p rvp9-encoder --test w_gop_topology gop_topology_byte_identity --locked -- --exact --nocapture. The stable stream is 23186 bytes with FNV-1a 0xc5a9eaff599669d2, but the fixture expects 23293 bytes and 0xa34dc1d6f4307467. The sibling gop_topology_schedule_matches_emitted_frames test passes. Expected: the byte golden records the accepted stream while the schedule oracle continues to guard GOP behavior.

## Fix

Commit `bb188d336838e30fd600795e318397c99c340780` refreshes the byte
golden to the stable accepted production-parity stream. The byte assertion and
the independent emitted-frame schedule oracle remain intact.

Regression evidence:

- Untouched `origin/main` commit `187b6af6` and the integration tree both
  produced 23,186 bytes with FNV-1a `0xc5a9eaff599669d2`.
- `cargo test -p rvp9-encoder --test w_gop_topology --locked -- --nocapture`
  passes both the byte-identity and schedule-oracle tests.
- A second exact byte-identity run reproduced the same length and hash.

Awaiting independent verification before closure.

### Provenance reconciliation (2026-10-19)

Authoritative fixing commit: `f613c0fbe60bae8160bdd86b5796354810882023` (test(enc): refresh GOP stream golden (RVP9-BUG-FLUX-00007)).
The original reference `bb188d336838e30fd600795e318397c99c340780` does not resolve locally and is retained above for audit.
Historical Fixed ledger commit `b7a35b7edef16fb6f95cba88e9eccc4788077d81` has this fixing commit as its immediate parent.
All original old/new length and FNV constants match exact candidate patch in w_gop_topology.rs.
The recovered commit is an ancestor of `origin/main`. This is a metadata correction only;
no regression was rerun and no new fix or verification verdict is recorded.

## Notes
