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
  **UPDATE 2026-06-19:** the **kfonly 444/422/440** part is now FIXED by the harvest
  (SIG1+SIG2) — those 3 tests pass `mismatched==0` but stay `#[ignore]d` with an
  inverted self-check; tracked as `RVP9-BUG-KILN-00007` (un-ignore + flip assertion).
  Only the **aq2-444 / 720p-aq2** part of this entry remains open/unverified.
- [ ] 2026-06-19: Doc cross-ref drift (cosmetic, from the track2 harvest review):
  `crates/rvp9-decoder/src/tile.rs:1219` cites `reconstruct_inter.rs:521-548` for the
  intra `max_blocks` geometry, but the referenced block actually runs `519-562`
  (~14-line tail imprecision). Refresh the line range when next touching that file.
- [x] 2026-06-20: **UPDATE to the 2026-06-19 SIMD note above** (green-the-build pass).
  The "4 SIMD equivalence tests fail … likely target-cpu=native / host-CPU specific"
  hypothesis was wrong — these are deterministic *integer* kernels. Resolved:
  (1) the 2 loop-filter ones (`lpf_{horizontal,vertical}_4_avx2`) were an
  out-of-domain proptest bug (`blimit=255` unreachable in valid VP9, max 193) —
  **FIXED** in this pass (the loop-filter blimit-cap commit); (2) the 2 idct ones (`idct16x16_sse2_eq_avx2_*`)
  are a possible **real SSE2 saturation bug** (out-of-domain hypothesis was
  adversarially refuted) — promoted to ledger **KILN-00010** (do NOT mask by
  narrowing the test before triage).
- [ ] 2026-06-20: **Vector/corpus-dependent unit tests hard-fail in a fresh worktree**
  (green-the-build pass). `bitstream::header::tests::captured_header_round_trip` and
  `bitstream::compressed_header::tests::write_compressed_header_captured_frame_round_trip`
  panic "test vector must be present: NotFound" (`header.rs:1461`,
  `compressed_header.rs:2455`), and `real_world_corpus_decodes_bit_exact`
  (`crates/rvp9-harness/tests/real_world.rs:199`) needs a hidden/altref fixture that
  isn't committed. They pass on the data-populated main clone but hard-fail anywhere
  the gitignored vectors/corpus aren't fetched (every new worktree). Consider making
  these skip-with-a-log when the fixture is absent (test robustness), or document the
  fetch as a prerequisite, so `cargo test --workspace` is green on a clean checkout.
