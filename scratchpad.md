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
- [ ] 2026-06-21: **Skip RD tie-break over-triggers skip when the skip bit is cheap**
  (`crates/rvp9-core/src/mode_decision/inter.rs` `evaluate_candidate_for_filter`, the
  `rd_encode >= rd_skip` tie-break). Measured: feeding libvpx's frame-adapted skip
  prob (`cm->fc->skip_probs[skip_ctx]`) into the tie-break — the documented "follow-up"
  — regresses both oracles (fixed-Q E9 mode-field degradation, e.g. Q120 prediction
  −2.6 dB when picker-skip is honoured; VBR product gate curve-collapses, BD-rate
  +6.43%→+12.69%). Root cause: skip-side dist `sse_after_pred << s3_skip_shift()` is not
  calibrated against `dist_encode = coef_dist` / `lambda` like libvpx, so a cheaper skip
  prob makes too many blocks pick skip. To match libvpx's higher skip rate (34.7% vs
  rvp9 ~27% at q80) the skip tie-break **dist-domain must be recalibrated first**; only
  then re-test the adapted prob + honour-picker-skip. Full A/B in
  `wrk_journals/2026.06.21 - JRN - adapted skip-prob picker tie-break.md`.
- [ ] 2026-06-21: **VBR product-gate divergence roadmap** — first clean block-level
  rvp9-vs-libvpx diagnosis on the actual Stage 4a VBR gate (not contaminated fixed-Q):
  `wrk_docs/2026.06.21 - CR - VBR product-gate divergence analysis.md` (+ raw DATA
  companion). Top levers for the remaining +6.43% BD-rate / +1.16 dB gap:
  (1) **reference-allocation cadence** — rvp9 starves the keyframe (q164/16608b vs
  libvpx q57/74536b @100kbps) and lacks libvpx's ~15kbit q~130 golden boosts every ~10
  frames; bounded keyframe+GF investment, swept vs the 4-point gate (prior unbounded
  attempts overspent: keyframe-BPM-split +69%, VBR-GOLDEN port +46%);
  (2) **ZEROMV labelling** — rvp9 ~26% vs libvpx ~4% area; NOT missing temporal MVs
  (`use_prev_frame_mvs` is a vestigial no-op, frame_loop.rs:2570-2589) — likely an
  inter-mode cost/tie-break issue beyond `nearest_tie_fix` (static blocks coded ZEROMV
  by rvp9 vs NEARESTMV by libvpx);
  (3) **RT partition descent under-selects 32x32** (rvp9 94% 16x16 vs libvpx 32x32 mix)
  — downstream of the same picker RD-domain calibration as the skip tie-break above.
- [ ] 2026-06-21: **Dead config field `EncoderConfig::use_prev_frame_mvs`** — the
  encoder ignores it (computes the temporal-MV gate purely structurally per the SIG-1
  drift fix, frame_loop.rs:2570-2589). The field still exists and several tests set it
  `false` as a "regression guard" but it has NO effect on output (verified byte-identical
  true vs false on the Stage 4a gate). Candidate for removal (or doc as inert) when next
  touching the config surface, so future sessions don't chase it as a lever.
