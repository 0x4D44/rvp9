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
  **UPDATE 2026-06-21:** the `idct16x16_sse2_eq_avx2_{uniform_12,eob_dense}` pair is now
  tracked as KILN-00010. Two MORE pre-existing trunk failures observed (fail identically
  on clean `origin/main` e678386, unrelated to any LF/threading work):
  `bitstream::header::tests::captured_header_round_trip` and
  `bitstream::compressed_header::tests::write_compressed_header_captured_frame_round_trip`
  — captured-fixture round-trips. Needs triage (are the captured fixtures stale vs current
  header writer, or a real round-trip regression?). Not yet ticketed.
- [ ] 2026-06-18: Known decoder bug, pre-existing (NOT from this harvest):
  `crates/rvp9-harness/tests/profile1_decode.rs` has 5 `#[ignore]`d tests —
  `profile1_444_aq2_follow_up` / `profile1_720p_aq2_follow_up` ("aq2 444 surfaces rvp9
  segmentation bug — 9/12 mismatch") and the kfonly 444/422/440 cases. The aq2-444
  one may be segmentation-adjacent to G-C; worth checking whether G-C narrowed it.
  **UPDATE 2026-06-19:** the **kfonly 444/422/440** part is now FIXED by the harvest
  (SIG1+SIG2) — those 3 tests pass `mismatched==0` but stay `#[ignore]d` with an
  inverted self-check; tracked as `RVP9-BUG-KILN-00007` (un-ignore + flip assertion).
  **UPDATE 2026-06-21:** KILN-00007 FIXED (task/fix-kfonly-tests) — the 3 kfonly tests
  are now un-ignored, assert `mismatched==0`, renamed `*_kfonly_decodes_bit_exact`,
  and run by default. Only the **aq2-444 / 720p-aq2** part of this entry remains
  open/unverified (`profile1_{444,720p}_aq2_follow_up` still `#[ignore]d`).
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
  **UPDATE 2026-06-24:** the `real_world_corpus_decodes_bit_exact` hidden/altref fixture
  part is fixed by `5ad1f80` (`tests/corpus/fixtures/foreman_c444_2pass.ivf` + md5).
  The captured-header vector prerequisite remains open because `tests/conformance/vectors/`
  is still gitignored/fetched data.
- [x] 2026-06-24: **`scripts/fetch-vectors.sh` cannot reliably populate the deterministic
  test vectors on this Windows worktree.** The script has CRLF line endings, which makes
  Bash report `$'\r': command not found` before it can run (`scripts/fetch-vectors.sh:10`),
  and its Stage 0 vector list still includes `vp90-2-02-size-202x202.webm`
  (`scripts/fetch-vectors.sh:30`), whose upstream URL returned 404 during the green-build
  pass. Normalize the script and refresh/remove the dead vector entry so new worktrees can
  fetch the deterministic test data without a manual curl fallback.
  **UPDATE 2026-06-25:** fixed by adding a `.gitattributes` LF checkout rule for shell
  scripts and correcting the vector to upstream's `vp90-2-03-size-202x202.webm`.
- [ ] 2026-06-25: Add honest 1080p/1080p60 encoder realtime performance rows after the
  encoder supports render-width/render-height or another explicit coded/render-size
  strategy. **UPDATE 2026-08-03:** Unit 671 removed the coded-height blocker and proved
  literal 1920x1080 streams under rvp9-decoder, pinned vpxdec, and ffmpeg. The honest
  warmed 1080p30/60 performance rows remain open and are now the next benchmark step.
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
- [ ] 2026-06-22: **Skip branch DROPS the per-mode rate (libvpx keeps it) — but
  restoring it is RD-NEGATIVE in isolation.** `inter.rs:3197-3206` collapses
  `final_rate` to `skip_true_cost` on `take_skip`, dropping mode/mv/ref/coef;
  libvpx (`vp9_pickmode.c:2352-2399`) drops ONLY coef and re-adds
  `rate_mv + inter_mode_cost + ref_frame_cost` unconditionally. This is THE cause
  of the ZEROMV over-use (26% vs 4%): on skip, ZEROMV/NEARESTMV rank by pred-SSE
  alone so the (0,0) predictor wins, erasing ZEROMV's ~7-bit mode handicap.
  Env-gated fix (`RVP9_SKIP_MODECOST`) measured + REVERTED: regresses E9 fixed-Q
  (+1.29→+1.79 / +1.62→+1.70 / +1.69→+2.63 dB at Q80/120/160, skip-frac 26.7→15.7%)
  AND the VBR gate (+1.16→+2.08 dB, curve-shift). RC-independent (fixed-Q) ⇒ NOT
  an RC artifact: rvp9's NEAREST is non-zero on the affected blocks (full-scan
  `dec_find_mv_refs`, same as libvpx's actual NEAREST), so pricing ZEROMV correctly
  forces a *worse-predicting* NEARESTMV/non-skip mode that rvp9's calibration can't
  pay back. This RETIRES the sharpest isolated lever and confirms the coupled
  RD-domain thesis. Full A/B + mechanism:
  `wrk_journals/2026.06.22 - JRN - skip-branch mode-cost (ZEROMV over-use).md`.
  Re-test ONLY as part of the joint co-calibration (lambda + skip tie-break + mode
  cost + NEWMV affordability), never alone.
- [ ] 2026-06-22: **MV-search cost `sad_per_bit`/`error_per_bit` (hardcoded 16, L-43)
  is INERT** on the Stage 4a gate. libvpx derives them per-q; rvp9 pins 16. Swept
  `RVP9_SADPB`/`RVP9_EPB` (env override, reverted) at 4/8/16: matched-Δ flat at
  +1.16-1.17 dB, PSNR curve unchanged — cheaper/better MV search does NOT change
  selection because the per-block RD rejects the better MVs downstream. Do NOT
  re-probe the MV-search cost in isolation; the bottleneck is the RD comparison's
  calibration, not search quality. (Two independent levers — mode-cost RANKING and
  motion-SEARCH quality — both dead-end at the same coupled RD knot this session.)
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
- [ ] 2026-06-21: **BUG (libvpx-faithfulness): `KF_ACTIVE_QUALITY_BOOST = 32` is wrong**
  (`crates/rvp9-encoder/src/rate_control/cbr.rs:387`; also `KF_HIGH = 5000` at :377).
  Verified against the libvpx submodule: `DEFAULT_KF_BOOST = 2000` (vp9_ratectrl.h:30),
  `kf_low = 400`, `kf_high = 4000` (vp9_ratectrl.c:93-94), and one-pass VBR calls
  `get_kf_active_quality(rc, avg_frame_qindex[KEY], …)` with `rc->kf_boost = 2000`. Since
  `400 < 2000 < 4000` libvpx BLENDS toward the low-motion-minq table (low keyframe
  active-best ≈ q57); rvp9's 32 (< KF_LOW) forces pure high-motion-minq → high active-best.
  The code comment ("DEFAULT_KF_BOOST is well below KF_LOW") is factually wrong. **Inert
  on the single-GOP Stage 4a gate** (the lone first-ever keyframe has active_worst=255 and
  is `regulate_q`-bound under BPM=5.0, so the active-best floor doesn't bind — A/B with
  boost=2000/KF_HIGH=4000 was byte-identical), but it WILL matter for multi-GOP realtime
  keyframes (subsequent KFs use `last_q_key*2` active_worst + `get_kf_active_quality`).
  Fix with a multi-GOP measurement, inside the reference-allocation program. NOTE the
  shared cbr.rs const also feeds CBR — make the fix mode-aware or verify CBR.
- [ ] 2026-06-21: **Stage-4a-gate keyframe is `regulate_q`/BPM-bound, not floor-bound.**
  The q164 keyframe (vs libvpx q57) is NOT caused by the keyframe-share caps (RVP9_KFCAP=0,
  RVP9_KFLB=0 both byte-identical — the KF sits far under the 40% cap) nor the active-best
  floor (KF boost A/B no-op). It is `regulate_q` hitting the 25×-avg KF target under the
  VBR `bits_per_mb_multiplier=5.0`. The keyframe-specific BPM range is now TESTED via an
  env-gated `RVP9_KF_BPM` sweep (vbr.rs:298, reverted): KF_BPM 5.0→4.0→3.0→2.5→2.0→1.5
  gives mean Δ +1.16→+1.21→+1.28→+1.38→+1.42→+1.59 — i.e. **more keyframe investment
  monotonically WORSENS** matched-bitrate quality. DEFINITIVE: rvp9's starved-keyframe
  allocation is already near-optimal for rvp9's prediction efficiency; a better KF doesn't
  pay back because the inters don't capitalize. **Reference allocation (A) is a dead end
  until inter-prediction efficiency (B) improves — do B first.** Cross-decoder stayed
  byte-exact throughout.

<!-- Added 2026-07-31 by the independent two-eyes verify/closure pass over the
     Fixed-awaiting-closure queue (52 bugs). Each item is an out-of-scope
     observation surfaced while verifying someone else's fix; none of them
     blocked a closure, and none is a tracked bug. -->

- [ ] 2026-07-31: **Three missing `notes/divergence-ledger.md` entries.** CLAUDE.md
  requires every deliberate departure from libvpx behaviour to be recorded there.
  Three landed fixes departed without an entry: (a) `crates/rvp9-encoder/src/denoiser.rs:786`
  and `:836` widened `sse_diff` to `i64` and the bias-to-LAST comparison to `u64` where
  libvpx uses `int`/`unsigned int` (RVP9-BUG-KILN-00018); (b) `crates/rvp9-io/src/y4m.rs`
  now refuses odd-dimension 4:2:0 y4m that libvpx's `y4minput.c` accepts with ceil chroma,
  and this reaches the shipped `rvp9enc` (RVP9-BUG-KILN-00042); (c) `crates/rvp9-io/src/container.rs`
  now hard-errors on a WebM with no `Tracks` element instead of yielding zero packets
  (RVP9-BUG-KILN-00044). All three are paperwork, not correctness — the values in (a)
  cannot reach the divergent range for any real block size.
- [ ] 2026-07-31: `crates/rvp9-bench/src/bin/bdrate.rs` carries the same defects
  RVP9-BUG-KILN-00061 fixed in `soak.rs`, untracked: `encode_one` (~`:106-111`) uses
  `.status().expect()` plus `assert!` (panic, exit 101, not a clean error), and
  `decode_and_psnr` (~`:123-128`) writes a shared temp Y4M `{stem}.dec.y4m` under
  `temp_dir()` and waits on the decoder with no timeout.
- [ ] 2026-07-31: `crates/rvp9-encoder/tests/phase2_reuse_positive_control.rs`
  (`phase2_newmv_reuse_fires`) and `crates/rvp9-encoder/tests/w_ml_partition.rs`
  (`a21_ml_partition_breakout_fires_and_decodes`) were correctly changed from silent
  false-green to `#[ignore]` by RVP9-BUG-KILN-00019, which means neither proof now runs
  in the normal gate. Giving each a synthetic fixture would restore coverage without a
  corpus dependency.
- [ ] 2026-07-31: `crates/rvp9-harness/src/bin/diff_bitstream.rs` — two follow-ups from
  the RVP9-BUG-KILN-00036 closure. `--exit-on-diff` stops at LEVEL granularity, so a level
  emitting 500 records still emits all 500; and `tests::exit_on_diff_stops_after_first_level_zero_record`
  may be non-discriminating, because its fixture's 4-byte payloads are not valid VP9 so
  L1-L4 likely produce no records either way. Cheapest check: revert the four `should_stop()`
  call sites and re-run — if it still passes, it is not a gate.
- [ ] 2026-07-31: `crates/rvp9-encoder/src/frame_loop.rs` — `EncoderConfig::use_prev_frame_mvs`
  is dead public API. An earlier DRIFT-FIX (~`:2716`) made it vestigial; it no longer gates
  anything, but callers can still set it and reasonably expect it to matter. Deprecate or
  remove.
