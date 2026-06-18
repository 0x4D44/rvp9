# Scratchpad

Out-of-scope observations parked for a separate, human-invoked review. Dated
`- [ ]` entries with `file:line` pointers. Don't fix inline.

- [ ] 2026-06-18: Re-establish the **longclip differential gate** dropped during the
  track2 harvest. Needs: bring `crates/rvp9-harness/tests/longclip_{differential,determinism,parse_replay}.rs`
  from `track2-longclip-differential-gate`, the ~27 MB `tests/corpus/longclips/` corpus
  (41 files; or fetch via `scripts/build-longclip-corpus.ps1`), and **re-baseline**
  `tests/corpus/longclips/expectations.tsv` against current main + the 6 harvested
  fixes (track2's baseline won't match). This is the end-to-end net that originally
  found SIG1/SIG2; only unit oracles guard them now. Repo-weight decision (Arthur).
- [ ] 2026-06-18: Run the **conformance suite** against the harvest branch before
  integration — vectors aren't fetched in any local clone, so SIG1/SIG2 (decode-state
  changes) weren't run through it. `cargo run --release -p rvp9-harness --bin conformance -- --vectors-dir tests/conformance/vectors`.
- [ ] 2026-06-18: Known decoder bug, pre-existing (NOT from this harvest):
  `crates/rvp9-harness/tests/profile1_decode.rs` has 5 `#[ignore]`d tests —
  `profile1_444_aq2_follow_up` / `profile1_720p_aq2_follow_up` ("aq2 444 surfaces rvp9
  segmentation bug — 9/12 mismatch") and the kfonly 444/422/440 cases. The aq2-444
  one may be segmentation-adjacent to G-C; worth checking whether G-C narrowed it.
