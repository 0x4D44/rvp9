# RVP9-BUG-KILN-00007 - kfonly Profile-1 tests left #[ignore]d after the harvest fixed their bug — SIG1 multi-keyframe independence has no committed-vector regression guard

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/test-coverage (decoder SIG1/SIG2)
- **Raised:** 2026-06-19 by Claude overnight code review
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
- **State history:** 2026-06-19 Opened by Claude from the track2 decoder-fix harvest review pass. 2026-06-21 Fixed by Claude (task/fix-kfonly-tests): un-ignored the three kfonly follow-up tests in `crates/rvp9-harness/tests/profile1_decode.rs`, flipped the inverted `assert!(mismatched > 0)` self-check to `assert_eq!(mismatched, 0)`, renamed them to `profile1_{444,422,440}_kfonly_decodes_bit_exact`, and rewrote the stale doc comment. They now decode bit-exact and run by default (verified mismatched==0 first). Test-only; no version bump. Awaiting two-eyes verification → Closed. -> 2026-06-23 Closed by Codex (task/20260623-TSK-HUM-verify-fixed-bugs): independently verified the original observation is resolved. The three kfonly Profile-1 regression tests are active, not ignored, and `$null | mdtimeout 300 cargo test -p rvp9-harness --test profile1_decode --release kfonly -- --nocapture` passed all 3 tests with `mismatched == 0`.

## Observation

`crates/rvp9-harness/tests/profile1_decode.rs:313-345` keeps
`profile1_444_kfonly_follow_up`, `profile1_422_kfonly_follow_up`, and
`profile1_440_kfonly_follow_up` `#[ignore]d`. Each carries the comment
"When the bug is fixed, change this to `assert_eq!(mismatched, 0)`" and a runtime
self-check `assert!(mismatched > 0, "Bug appears fixed — remove #[ignore] and move
to PROFILE1_VECTORS")`. Their failure hypothesis (lines 305-310: "state that
accumulates across frames … ref-frame management or above/left context") is exactly
the surface SIG1 (reset all `FrameContext` banks on every keyframe) and SIG2
(above/left coef-context edge clamp) fix.

Running them on the post-harvest tree
(`cargo test -p rvp9-harness --test profile1_decode --release -- --ignored kfonly`)
**fails all three on their `mismatched > 0` self-check** — i.e. `mismatched == 0`,
so the harvest already fixed the tracked T-2 bug. Because the tests stay
`#[ignore]d`, `cargo test --workspace` skips them, so there is currently **no
committed-vector end-to-end regression guard for SIG1 multi-keyframe independence**
(the default `PROFILE1_VECTORS` are single-keyframe I-P-P-P streams; only these
`kfonly` streams place multiple independent keyframes back-to-back). The fixtures
`foreman_c{444,422,440}_kfonly.ivf` are already committed (CIF 352×288, partial on
both right and bottom edges), so activation is free.

(Note: SIG2's intra partial-edge `max_blocks_wide/high` geometry is already
exercised end-to-end on every default run by the keyframe-0 of `foreman_c444.ivf`
in `PROFILE1_VECTORS` — also 352×288. The genuinely dormant surface is SIG1
cross-keyframe independence.)

## Notes

Confirmed; 0/3 skeptics refuted. Panel downgraded high→medium: the shipped code is
correct and SIG1/SIG2 each have dedicated *unit* coverage
(`keyframe_resets_all_frame_context_banks` in `lib.rs`; the two
`set_coef_flags_clamps_at_frame_edge*` tests in `tile_ctx.rs`), so this is a missing
*end-to-end* guard rather than a behavioural defect. The harvest itself edited this
file (it added the SIG4 fix-lock `profile1_422_440_inter_decodes_bit_exact_sig4`),
so updating fix-locks was in scope — leaving the now-passing kfonly tests ignored is
an inconsistency, not a deliberate exclusion.

**Fix (trivial, test-only):** remove `#[ignore]` from the three kfonly follow-ups,
replace each `assert!(mismatched > 0, …)` with `assert_eq!(mismatched, 0)`, and
ideally move the fixtures into `PROFILE1_VECTORS` so they run by default — exactly
what the test's own comment instructs. Deferred to the fix-loop (not fixed in this
read-only review pass), though it is borderline trivial enough to land immediately.

**Supersedes** the `kfonly` portion of the scratchpad entry dated 2026-06-18
("Known decoder bug, pre-existing … the kfonly 444/422/440 cases"): that bug is now
fixed by SIG1+SIG2. Only the `profile1_444_aq2_follow_up` / `profile1_720p_aq2_follow_up`
("aq2 444 surfaces rvp9 segmentation bug — 9/12 mismatch") portion of that entry
remains open and unverified by this pass.
