# RVP9-BUG-KILN-00011 - prev_mv_grid contains IntraFrame cells after an inter encode (a18 invariant violated) — pre-existing

- **State:** Blocked
- **Priority:** Should
- **Severity:** Low
- **Area:** encoder/prev_mv_grid population (Stage 4a' picker)
- **Raised:** 2026-06-20 by Claude (green-the-build pass)
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
- **State history:** 2026-06-20 Opened by Claude from the green-the-build pass; confirmed pre-existing (fails at baseline 01f8f2b and at HEAD). -> 2026-06-23 Blocked by Codex during ledger-drain triage; current source and focused test evidence show the recorded no-`IntraFrame` invariant is stale, not an unattended-fixable encoder defect. Needs Arthur/maintainer disposition as non-defect or conversion into a docs-cleanup requirement.

## Observation

`a18_prev_frame_mvs_grid_populated_after_first_inter`
(`crates/rvp9-encoder/tests/a18_prev_frame_mvs_fires.rs`) is RED on `origin/main`
(`0b62ecc`), panicking at `a18_prev_frame_mvs_fires.rs:188`:

> the encoder's Stage 4a' picker does not emit intra-in-inter cells, so
> prev_mv_grid should not contain IntraFrame entries after an inter encode (HLD §8.5)

i.e. after an inter encode the `prev_mv_grid` DOES contain at least one cell with
`ref_frames[0] == RefFrame::IntraFrame`, which the test's HLD §8.5 invariant
forbids.

## Notes

**Pre-existing — NOT a track2-harvest regression.** Bisected: the test FAILS
identically at the pre-harvest baseline `01f8f2b` and at HEAD `0b62ecc`. The
harvest (`01f8f2b..0b62ecc`) touched no prev-MV-grid population logic, so it
neither caused nor changed this failure. (The scratchpad's earlier "only 4 SIMD
tests fail on clean main" was an incomplete sweep that did not run the full
rvp9-encoder integration-test suite.)

Two possibilities to triage (out of scope for the green-the-build pass; must NOT
be conflated with the SIG1 regression KILN-00009):
1. **Picker defect** — the Stage 4a' picker writes `IntraFrame` `ref_frames` into
   `prev_mv_grid` after an inter encode, violating the documented HLD §8.5
   invariant; fix the picker so intra-in-inter cells are not emitted.
2. **Stale test expectation** — the picker legitimately emits intra-in-inter cells
   now (HLD §8.5 evolved); correct the test's assertion.

Sibling test `a18_prev_frame_mvs_grid_chains_across_inter_frames` passes, so the
grid plumbing itself is alive; the issue is specifically the IntraFrame-cell
content.

2026-06-23 drain triage:

- The recorded regression,
  `cargo test -p rvp9-encoder --test a18_prev_frame_mvs_fires a18_prev_frame_mvs_grid_populated_after_first_inter -- --nocapture`,
  now passes in the task worktree.
- The current decoder writes `(IntraFrame, NoRefFrame)` into `cur_mv_grid` for
  intra-in-inter blocks (`crates/rvp9-decoder/src/tile.rs:2688`), and the encoder
  mirrors that write-back for intra-in-inter leaves
  (`crates/rvp9-encoder/src/lib.rs:6073`).
- Therefore an `IntraFrame` cell in `prev_mv_grid` after an inter encode is no
  longer evidence of an encoder defect; the old HLD/test invariant has been
  superseded by the current decoder parity rule.

### Arthur disposition confirmed (2026-10-17)

Arthur approved independently reconfirming the current behavior and resolving
the stale report. The exact recorded test selected one test and passed; the
full `a18_prev_frame_mvs_fires` module passed both tests, and the lead
independently reran that module with two passes. Existing assertions check
frame kinds, the 16x16 / 256-cell grid, LAST entries, and frame-2 eligibility;
they no longer forbid intra cells. No assertion or implementation was changed.

The lead re-derived encoder `InterPartitionLeafDesc::mv_ref` and its grid
splat (`lib.rs:4073-4118`), decoder intra writeback (`tile.rs:2875-2895`),
and inter-only previous-MV salvage (`mv_ref.rs:538-595`). Both sides
intentionally store `[IntraFrame, NoRefFrame]` with zero MVs for intra leaves.
The remaining stale HLD section 8.5 in
`wrk_docs/Track_F/F0.3_HLD_prev_frame_mvs.md` now carries a dated supersession.
Resolve as `by-design`: the cells are real and valid. This is not a new
behavior fix or a claim of a fresh upstream differential comparison.
