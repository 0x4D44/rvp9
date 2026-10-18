# RVP9-BUG-FLUX-00011 — s11 intra restriction test rejects a valid control mode

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** core/mode-decision tests
- **Raised:** 2026-08-09T20:08:59Z
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
- **State history:** Open (2026-08-09T20:08:59Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T09:10:40Z, deltic:auto role=fix run=fix-20261016T090208Z-9ab99630 branch=task/bug-RVP9-BUG-FLUX-00011-run-fix-20261016T090208Z-9ab99630 code=bdb9f9a4bfe58795ea4d5bff604b375946e3883f gate=manual) -> Closed (2026-10-18T15:40:58Z, deltic:manual role=verify run=verify-20261018T153710Z-b344a7c3 verified_fix_run=fix-20261016T090208Z-9ab99630 verdict=close)

## Observation

Observation: `cargo test -p rvp9-core --lib mode_decision::inter::tests::s11_rt_intra_dc_only_low_content_restricts_to_dc -- --exact` fails deterministically at the control assertion because `picked_intra_mode` can be a mode outside DC/V/H. The same failure reproduces from a clean archive of commit f3a330c2, so it is not caused by the denoiser worktree. Expected: the focused test and full core suite pass while still proving that the active gate restricts normal content to DC; actual: the control-side anti-stub assertion rejects a valid broader-mode result.

## Fix

Already fixed on trunk by `bdb9f9a4bfe58795ea4d5bff604b375946e3883f` (`fix(core): accept unrestricted intra control (RVP9-BUG-FLUX-00011)`, 2026-08-09). The unrestricted control now accepts any valid intra mode; the active-gate DC assertion remains. This reconciliation records the original fixing commit rather than claiming a new behavior change.

### Verification summary (2026-10-18T15:40:58Z, deltic:auto run=verify-20261018T153710Z-b344a7c3 verified_fix_run=fix-20261016T090208Z-9ab99630 verdict=close)

Verifier note: Manual verify: Independent session 01a14fa5, not fixer; base09697b55 includes bdb9f9a4. Original exact s11_rt_intra_dc_only_low_content_restricts_to_dc command selects 1 and passes; lead rerun also passes. Reversing only control predicate to DC[PATH] selects 1 and fails that assertion at inter.rs:13649 (exit101). Restored identical source SHA256 714fdb1b41a1de1a12de280c9aca2e4de3a27168db3bc4ae873ed704256bdb93; exact test passes again, git diff empty. Production unrestricted candidate set includes TM; active DC assertion remains intact. Evidence in 2026.10.18 independent verification journal. S...

## Notes

2026-10-16 provenance reconciliation:

- Confirmed the earlier fixing commit is an ancestor of `origin/main` and its corrected control predicate remains present.
- Current exact test selected one test and passed. Reversing only the control assertion to its former DC/V/H restriction selected one test and failed at that assertion: `got Some(Tm)` (exit 101). The preceding active-gate DC assertion remained intact.
- Restored the source byte for byte from the pre-proof snapshot; SHA-256 `714fdb1b41a1de1a12de280c9aca2e4de3a27168db3bc4ae873ed704256bdb93`, with no source diff. Final restored validation is recorded in the task journal.
- Released only the expired August claim before taking a fresh claim; the old worktree was left alone. No permanent source change was needed.

See [provenance and validation journal](../wrk_journals/2026.10.16%20-%20BUG%20-%20Reconcile%20unrestricted%20intra%20control%20provenance.md).
