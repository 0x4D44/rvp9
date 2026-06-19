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
- [ ] 2026-06-19: **Pre-existing trunk test failures** (present on `origin/main`
  01f8f2b, NOT from the track2 harvest — verified by running on the clean main clone):
  4 SIMD equivalence tests fail —
  `dsp::x86_64::idct16x16_sse2_256::tests::idct16x16_sse2_eq_avx2_{uniform_12,eob_dense}`
  (directed, deterministic) and `dsp::x86_64::lpf_{horizontal,vertical}_4_avx2::tests::scalar_eq_avx2_proptest`.
  Conformance still passes 201/201 (runtime decode picks one tier and is correct), so
  these are SSE2-vs-AVX2 / scalar-vs-AVX2 mismatches on edge inputs, likely
  `target-cpu=native` / host-CPU specific. Needs separate triage; `RVP9_FORCE_SCALAR`
  + per-kernel diff to localize.
- [ ] 2026-06-18: Known decoder bug, pre-existing (NOT from this harvest):
  `crates/rvp9-harness/tests/profile1_decode.rs` has 5 `#[ignore]`d tests —
  `profile1_444_aq2_follow_up` / `profile1_720p_aq2_follow_up` ("aq2 444 surfaces rvp9
  segmentation bug — 9/12 mismatch") and the kfonly 444/422/440 cases. The aq2-444
  one may be segmentation-adjacent to G-C; worth checking whether G-C narrowed it.
