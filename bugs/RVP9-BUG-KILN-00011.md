# RVP9-BUG-KILN-00011 - prev_mv_grid contains IntraFrame cells after an inter encode (a18 invariant violated) — pre-existing

- **State:** Open
- **Severity:** Low
- **Area:** encoder/prev_mv_grid population (Stage 4a' picker)
- **Raised:** 2026-06-20 by Claude (green-the-build pass)
- **State history:** 2026-06-20 Opened by Claude from the green-the-build pass; confirmed pre-existing (fails at baseline 01f8f2b and at HEAD).

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
