# RVP9-BUG-KILN-00033 - Conformance binary can exit successfully after skipped errors

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** harness/conformance-cli
- **Raised:** 2026-07-07
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T171626Z-p21311-n210115000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00033-run-fix-20260730T171626Z-p21311-n210115000-c1
- **Owner base:** 8df548d98211b89a078fefc9f6a0580cf31485f2
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T17:16:26Z
- **Owner until:** 2026-07-30T19:16:26Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-harness/` at `cfabd55`)

## Observation

The standalone conformance binary can exit 0 without proving all MD5-ledger frames decoded correctly.

`crates/rvp9-harness/src/bin/conformance.rs:111` handles `parse_packet` errors by incrementing `skipped`, advancing `shown_idx` at line `127`, and continuing. Non-`Unimplemented` decode errors are also counted as skipped at lines `198` and `209`. Skips only fail when `--strict` is set at line `319`.

The binary prints the expected MD5 row count at `crates/rvp9-harness/src/bin/conformance.rs:300`, but `run_one_vector` returns success at line `217` without checking that the decoded shown-frame count equals `md5_entries.len()`.

The integration test is narrower but still has the same parse-error gap: `crates/rvp9-harness/tests/conformance.rs:155` treats `parse_packet` errors as skipped frames. It does hard-fail non-`Unimplemented` decode errors at lines `225` and `435`.

## Fix

Open. Make parse errors and non-`Unimplemented` decode errors fatal in the conformance binary unless an explicit non-gating mode is requested, and assert shown-frame accounting against the MD5 ledger length. Mirror the parse-error accounting in the integration test so parse failures cannot disappear into skipped counts.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review. Existing `RVP9-BUG-KILN-00032` covers `real_world.rs` and decoder conformance coverage losing oracle data; it does not cover the standalone harness conformance binary's exit contract.
