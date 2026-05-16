# decisions-needed.md — Stage 1 deliverable

> **Status:** A1–A7 contributions integrated. A8 synthesizer is still in
> flight (agentId `ab5db17fcdd6cdf3c`); when 00-overview.md lands, A8's
> cross-cutting concerns may add new D / L / R entries which will be
> appended here.
>
> Cross-references use specialist §7 question IDs and A7 challenge
> numbers (#NN in A7 = `07-devils-advocate.md`).

Numbering convention (per `_stage2-dispatch-prep.md`):

- **D-NN** — Decision required before any Stage 3 implementation begins
- **L-NN** — Decision required during Stage 2 LLD writing (per-LLD detail)
- **R-NN** — Risk register (no decision yet; flagged for later stage gates)

---

## D — Stage 3 blockers (decide before Stage 2 exits)

### D-01 — High-bit-depth (10/12-bit) support in v1 scope?

**Origin:** PLAN.md Q1; A4 §7 #8 (HBD coefficient ranges); A5 §5 (HBD
quant tables deferred); A3 implicit via `comp_avg_pred_highbd` SIMD.

**Default:** **no HBD in v1.** PLAN.md doesn't list HBD as a success
criterion. Including it roughly doubles every kernel and table (separate
8-bit / 10-bit / 12-bit code paths in libvpx).

**Cost of inclusion:** ~3-5 weeks added across Stages 3-4 (every
transform, quant, intra-pred, loop-filter, denoiser, ME variance kernel
needs both 8-bit and HBD variants; the bitstream coefficient-range
handling also branches).

**Cost of exclusion:** HDR livestream + premium WebRTC use cases lose
us. PLAN.md target use cases don't strictly require HBD in v1.

**Decision needed by:** start of Stage 2 (HLD-arch will commit to one).

### D-02 — RTCD dispatch table: hand-written vs build.rs generator?

**Origin:** PLAN.md Q2.

**Options:**
1. Hand-write Rust dispatch tables per kernel family.
2. `build.rs` script that parses `vpx_dsp_rtcd_defs.pl` and emits Rust.

**Recommendation:** option 1 to start. Each kernel family is small
(transforms 4, intra-pred 10×4, etc.) and libvpx's `vpx_dsp_rtcd_defs.pl`
syntax is non-trivial to parse. Revisit if kernel count grows
unmanageably (e.g. >100 families).

**Decision needed by:** start of Stage 3 (HLD-simd).

### D-03 — Tile-threading primitives in Rust?

**Origin:** PLAN.md Q3; D6 ("mirror libvpx tile-based pthread model").

**Options:**
1. `std::thread` + `Mutex` + `Condvar` + `Barrier` (closest to pthreads).
2. `crossbeam` channels + scoped threads (idiomatic but diverges from libvpx).
3. `rayon` (parallel iterators; very different mental model).

**Recommendation:** option 1. D6 explicitly commits us to mirroring
libvpx; option 1 preserves the semantic 1:1 mapping.

**Decision needed by:** start of Stage 3 (HLD-threading).

### D-04 — Frame-parallel encoding in v1 vs v2?

**Origin:** PLAN.md Q4; A6 expected to weigh in.

**Default:** **v2.** Per-frame parallelism is real-time-critical at
1080p60 (forces 16ms budget) but tile-only parallelism may suffice at
1080p30 / 720p30. Test in Stage 4a; if real-time fails at 1080p60
without frame-parallel, escalate to v1.

**Decision needed by:** Stage 4a entry.

### D-05 — AQ modes 1 (variance) and 2 (complexity) in v1?

**Origin:** A5 Q-RC-15. PLAN.md §2 commits to "3 AQ modes matching
libvpx" but A5 found that **AQ modes 1, 2, 4 are guarded by
`!CONFIG_REALTIME_ONLY`** at the encoder dispatch level. Under our
real-time target (D1: cpu-used 6-8), libvpx itself doesn't exercise
modes 1/2. Only mode 3 (CYCLIC_REFRESH_AQ) is real-time-reachable.

**Options:**
1. Ship mode 3 only (matches what libvpx actually runs at our config).
2. Ship all three (1/2/3); modes 1/2 won't be reachable from our
   user-facing config but exist for future non-realtime extension.
3. Defer modes 1/2 to v2.

**Recommendation:** option 1 or 3 (skip in v1).

**Decision needed by:** Stage 4b entry.

### D-06 — Region-of-interest (ROI) / segmentation API in v1?

**Origin:** A5 Q-RC-28. libvpx supports ROI + segmentation through
user API; not in PLAN.md §2 success criteria.

**Default:** out of v1.

**Decision needed by:** Stage 4a entry.

### D-07 — Post-encode frame drop (`post_encode_drop_cbr`) in v1?

**Origin:** A5 Q-RC-25. libvpx's `post_encode_drop_cbr` is opt-in
default-off (`use_post_encode_drop=0`). Standard CBR never invokes it.

**Default:** out of v1. Matches libvpx default behaviour.

**Decision needed by:** Stage 4a entry.

---

## L — Per-LLD decisions (resolve during Stage 2 LLD writing)

### L-01 — `buffer_level` vs `bits_off_target` consolidation
**Origin:** A5 Q-RC-3. libvpx's own TODO. In our port, collapse to one.

### L-02 — Use optimised variance in AQ-variance code path
**Origin:** A5 Q-RC-27. libvpx's TODO — don't preserve the
inefficiency. Add to divergence ledger.

### L-03 — Cyclic refresh `time_for_refresh` dead field
**Origin:** A5 Q-RC-24. Drop for API simplicity (low risk).

### L-04 — Float determinism in RC `log10` calls
**Origin:** A5 Q-RC-5. Pick a deterministic log impl (or use libm-as-is
and document).

### L-05 — `accurate_sub_pel_search` stubbing in real-time
**Origin:** A3 §7 #1. At cpu-used 6-8 this is off; document gate in LLD.

### L-06 — `pred_mv_sad << 1` reference-masking threshold
**Origin:** A3 §7 (referenced in agent summary). Document the exact
threshold; concrete byte-parity trap.

### L-07 — Sub-pel cost-list invariants
**Origin:** A3 §7. The 4-entry cost list passed between subpel variants
has specific layout; mirror exactly.

### L-08 — `vp9_refining_search_8p_c` dead code in real-time
**Origin:** A3 §7. Compound search not reachable from rt path; can stub.

### L-09 — Encoder/decoder probability table sharing strategy
**Origin:** A4 §7 #1. Locked: share via `rvp9-core`.

### L-10 — Backward probability update protocol
**Origin:** A4 §7 #2. Document exact conditions for prob updates.

### L-11 — Range-coder renormalization edge cases (frame-end flush)
**Origin:** A4 §7 #4. `vpx_stop_encode` semantics.

### L-12 — Sign-bit and magnitude coding for high-frequency coefficients
**Origin:** A4 §7 #5. Golomb-Rice tail state machine.

### L-13 — Tile-data byte alignment + size-prefix layout
**Origin:** A4 §7 #6.

### L-14 — Superframe index encoding layout
**Origin:** A4 §7 #7.

### L-15 — Show-existing-frame wire format
**Origin:** A4 §7 #10. Validated by conformance vector
`vp90-2-10-show-existing-frame.webm`.

### L-16 — `frame_scale_factor` vs `vp9_resize_one_pass_cbr` relationship
**Origin:** A5 Q-RC-17. Two scaling paths; document.

### L-17 — Active worst-Q "small image format" adjustment
**Origin:** A5 Q-RC-18. ≤352×288 → q_adj_factor -= 0.25 in three sites.

### L-18 — Buffer critical_level = optimal/8 derived constant
**Origin:** A5 Q-RC-19. Document as derived.

### L-19 — `vp9_encodedframe_overshoot` huge state-mutation block
**Origin:** A5 Q-RC-20. Single most opaque path; must port exactly.

### L-20 — Noise estimator's forced-low-motion override
**Origin:** A5 Q-RC-21. Asymmetric escape hatch; document.

### L-21 — Cyclic refresh `weight_segment` smoothing math
**Origin:** A5 Q-RC-22. Sub-0.3% BD-rate trap if operations reorder.

### L-22 — Denoiser scalar/SSE2 non-bit-exactness handling
**Origin:** A5 Q-RC-23. Differential test strategy must allow ULP
differences for `increase_denoising=1`, OR canonicalize one impl.

### L-23 — `vp9_scene_detection_onepass` always-on behavior
**Origin:** A5 Q-RC-26. Cannot disable; mirror as always-on.

### L-24 — `vp9_cyclic_refresh_limit_q` screen-content-only gate
**Origin:** A5 Q-RC-29. Document gate.

### L-25 — `rate_correction_factors[]` resolution-change preservation
**Origin:** A5 Q-RC-30. `rcf_mult[]` scaling on read/write; underlying
value preserved. Off-by-one risk.

### L-26 — `vp9_compute_qdelta_by_rate` O(QINDEX_RANGE) implementation
**Origin:** A5 Q-RC-16. Match libvpx exactly; test exhaustively.

### L-27 — Probability-update damping during tentative pack
**Origin:** A4 §7 #9. Verify if tentative pack leaks into final pack.

### L-28 — Trellis-quantization (`vp9_optimize_b`) skip on real-time
**Origin:** A4 §7 #3. At cpu-used 6-8 `sf->optimize_coefficients=0`;
confirm.

---

## R — Risk register

### R-01 — A4 was truncated; §6 and §7 were synthesized
**Origin:** A4 transcript ended mid-§5. Stage 1 driver appended §6
(RTCD) and §7 (10 open questions) with explicit notes. Devil's
advocate (A7) should pressure-test whether the synthesized §6/§7
contain gaps that re-dispatch would catch.

### R-02 — AQ modes 1/2 unreachable on real-time builds — PLAN.md mismatch
**Origin:** A5 Q-RC-15. If we keep PLAN.md's "3 AQ modes" commitment,
we must un-gate CONFIG_REALTIME_ONLY for modes 1/2. Otherwise PLAN.md
needs updating to "1 AQ mode (CYCLIC_REFRESH)".

### R-03 — Denoiser SIMD bit-parity caveat
**Origin:** A5 Q-RC-23. Even libvpx itself doesn't guarantee SSE2/NEON
bit-parity with C for `increase_denoising=1`. Implication for our
byte-parity diagnostic tooling.

### R-04 — Float determinism in RC
**Origin:** A5 Q-RC-5. `log10` calls in RC make byte-parity contingent
on libm choice. Stage 4 differential tests need a known-good libm or
custom impl.

### R-05 — vp9_encodedframe_overshoot opacity
**Origin:** A5 Q-RC-20. Single most opaque state mutation in the
encoder; high risk of silent BD-rate drift if implemented wrong.

---

---

## Additional entries — A1 + A2 + A6 contributions

The following are summary entries; see each specialist's §7 for full
rationale + file:line citations. A8 will synthesize and finalise.

### A1 (architecture) contributions

| Tag | Topic | Source |
|---|---|---|
| D-08 | Frame-buffer pool ownership: expose to public API or hide behind `Encoder` type? | A1 §7.1 |
| L-29 | `VP9_COMP` decomposition strategy: subsystem split vs flat mega-struct | A1 §7.2 |
| L-30 | Lookahead bypass when `lag_in_frames == 0`: keep thin wrapper vs skip | A1 §7.3 + A5 |
| D-09 | `setjmp`/`longjmp` panic path: `Result<T,_>` chain vs `panic = abort` | A1 §7.4 |
| L-31 | Worker-pool sizing: eager (at `Encoder::new`) vs lazy (libvpx behaviour) | A1 §7.5 + A6 §7 |
| L-32 | Pin encoder to `REALTIME` mode at construction (drop `GOOD` mode dispatch) | A1 §7.6 |
| R-06 | External rate-control plug-in (`EXT_RATECTRL`) — WebRTC dependency, in v1 or v1.1? | A1 §7.7 + A5 |
| L-33 | Drop libvpx build-time `#ifdef` knobs (CONFIG_RATE_CTRL etc.); document in divergence ledger | A1 §7.8 |
| L-34 | Confirm `encode_with_recode_loop` is dropped (real-time uses single-pass only) | A1 §7.9 |
| L-35 | Bitstream packing threading: scalar-only in v1 (no parallel pack) | A1 §7.10 + A4 |
| L-36 | SVC reference-slot management: 8-slot map shared across spatial + temporal layers | A1 §7.11 + A6 |
| L-37 | Arc<FrameBuf> vs Rc<FrameBuf> for the ref-counted frame pool | A1 §7.1 |

### A2 (encoder core / non-RD path) contributions

A2 surfaced 20 LLD-bound questions; cross-references below.

| Tag | Topic | Source |
|---|---|---|
| L-38 | Variance-tree representation in Rust (nested structs vs flat array) | A2 §7 Q1 |
| L-39 | `x->variance_low[25]` indexing and packing | A2 §7 Q2 + Q14 |
| L-40 | `vp9_pick_inter_mode` 30-iteration loop control flow: refactor vs byte-parity | A2 §7 Q3 |
| L-41 | `block_yrd` fast-path bypass at speed ≥ 7 | A2 §7 Q4 |
| L-42 | Laplacian R-D table `rate_tab_q10` / `dist_tab_q10` ownership (A2 or A4 patch) | A2 §7 Q5; R-07 below |
| L-43 | `vp9_NEWMV_diff_bias` semantics + ordering relative to `encode_breakout_test` | A2 §7 Q6 + Q16 |
| L-44 | `set_low_temp_var_flag` 32×32 threshold expression | A2 §7 Q7 |
| L-45 | `scale_part_thresh_sumdiff` behaviour at speed 6 | A2 §7 Q8 |
| L-46 | Compound mode handling — `comp_modes` in `vp9_pick_inter_mode` | A2 §7 Q9; defer to v2? |
| L-47 | `sub8x8` chosen-mode-from-bmi[3] semantics (libvpx picks LAST sub-block's mode) | A2 §7 Q10 |
| L-48 | Filter-ref selection in `search_filter_ref` for non-RD path | A2 §7 Q11 |
| L-49 | Threshold-freq-fact storage layout (per-frame vs per-tile) | A2 §7 Q12 |
| L-50 | `reuse_inter_pred_sby` 4-entry buffer pool | A2 §7 Q13 |
| L-51 | `encode_breakout_test` chroma path conditional | A2 §7 Q15 |
| L-52 | `set_vt_partitioning` block_height == block_width invariant | A2 §7 Q17 |
| L-53 | Partition decision at frame edges (cropping behaviour) | A2 §7 Q18 |
| L-54 | `cpi->vbp_thresholds[4]` location (per-frame vs init) | A2 §7 Q19 |
| L-55 | `ref_mode_set[12]` Rust representation (const array vs enum) | A2 §7 Q20 |
| R-07 | Laplacian R-D tables NOT transcribed by any specialist — gap | A2 §5.4 declares this; A7 hypothesis test H2 |

### A6 (threading + SVC + speed features) contributions

A6 surfaced 14 LLD-bound questions, several covering load-bearing
aliasing patterns the Rust port must mirror.

| Tag | Topic | Source |
|---|---|---|
| D-10 | `SPEED_FEATURES` representation: snapshot-per-frame (immutable) vs `UnsafeCell` | A6 §7.1 |
| D-11 | `ThreadData::mb` (`MACROBLOCK`) aliasing: `tile_split + std::thread::scope` for tile-MT; `UnsafeCell + raw pointers` for row-MT (A6 recommendation) | A6 §7.2 |
| L-56 | `VPxWorker` → `std::thread::scope` mapping (A6 recommends over rayon) | A6 §7.3 |
| L-57 | Per-worker `FRAME_COUNTS` allocation strategy + accumulation back to main | A6 §7.4 + §7.6 |
| L-58 | SVC layer-context swap ordering (between recode loops and frame loops) | A6 §7.5 |
| L-59 | `cpi->row_mt` vs `cpi->oxcf.max_threads` distinct knobs modelling | A6 §7.7 |
| L-60 | Decoder parse/recon split (`tile_worker_hook` vs `row_decode_worker_hook`) | A6 §7.8 |
| L-61 | SVC max-tile-cols sizing under thread pool: pool sized to top-resolution layer? | A6 §7.11 |
| L-62 | Assert `log2_tile_rows == 0` in v1 (real-time never uses tile rows) | A6 §7.12 |
| L-63 | Worker thread priority elevation | A6 §7.13 |
| L-64 | Golden-file test for `SPEED_FEATURES` byte-parity per (speed, framesize, frame_type) tuple | A6 §7.14 |
| R-08 | `adaptive_rd_thresh_row_mt` non-determinism at cpu-used 6 default; A6 recommends bit-exact-by-default (divergence-ledger entry) | A6 §7.7 + §7 |
| R-09 | RTCD must complete before any worker spawns (init-order invariant) | A6 §7.10 + A1 §7.5 |
| R-10 | Speed-feature setter depends on per-frame condition tuple — byte-parity testing must be condition-tuple based, not just `(speed, w, h)` | A6 key finding |

### Items still missing (await A8)

- A8's resolution of any cross-cutting concerns that surface decisions
- Final cross-check: does each L-NN have a clear LLD owner?

---

## A7 (devil's advocate) — 32 challenges integrated by severity

The full text of each challenge — including libvpx file:line evidence
and recommended resolution path — lives in
`notes/vp9-analysis/07-devils-advocate.md`. This section maps challenge
numbers (#NN) to D / L / R categories.

> **Note:** A7's #1 and #30 (A4 truncation; missing inter-mode probability
> tables) are **RESOLVED** by the async A4 agent's complete rewrite
> (commit follows this one). R-01 below is similarly **RESOLVED**.

### Blocker (1)
- #2  Laplacian R-D tables `rate_tab_q10`/`dist_tab_q10` not transcribed.
      Maps to → **L-42** (already present). Action: A2 patch in Stage 2.

### High (8 active, 2 resolved)
- #1   A4 truncation — **RESOLVED** by async A4 (4556-line rewrite).
- #3   AQ modes 1/2 unreachable on real-time → **D-05** (already present).
- #5   `vp9_encodedframe_overshoot` opacity → **L-19** (already present).
- #7   `cpi->td` aliasing — fields beyond `mb`/`counts`/`rd_counts`.
       **NEW: D-12** — re-dispatch A1 or A6 to catalogue, OR Stage 2
       HLD-threading produces the exhaustive map.
- #8   `adaptive_rd_thresh_row_mt` non-determinism at cpu-used 6 default →
       **R-08** (already present).
- #19  REALTIME-VBR + altref support — does altref exist in VBR-RT?
       **NEW: D-13** — affects Stage 4a scope (altref doubles frame-loop).
- #25  EXT_RATECTRL WebRTC dependency → **R-06** (already present).
- #29  Speed-feature condition-tuple testing → **R-10** (already present).
- #30  A4 missing inter-mode probability defaults — **RESOLVED** by
       async A4 (vp9_pareto8_full and inter-mode probs now in §5).

### Medium (15)
- #4   Denoiser SIMD bit-parity caveat → **R-03** (already present).
- #6   Encoder-side frame-parallel encoding not in libvpx → **D-04** (present).
- #9   Probability table sharing → **L-09** (already present).
- #10  Conformance vector coverage gaps. **NEW: L-65** — Stage 3
       expands `scripts/fetch-vectors.sh`.
- #11  CONFIG_VP9_HIGHBITDEPTH leakage → **D-01** (already present).
- #13  `setjmp` mechanism × worker threads. **NEW: L-66** — owner:
       HLD-threading (Stage 2).
- #14  Cyclic-refresh segmentation map ownership.
       **NEW: L-67** — owner: HLD-threading.
- #15  `variance_low[25]` exact layout → **L-39** (already present).
- #17  `vp9_scene_detection_onepass` per-layer behaviour under SVC.
       **NEW: L-68** — owner: LLD-svc.
- #18  SVC cross-layer frame-drop interactions.
       **NEW: L-69** — owner: LLD-svc.
- #23  Sub-pixel filter selection in `search_filter_ref` → **L-48** (present).
- #26  `vp9_change_config` reallocation behaviour.
       **NEW: L-70** — owner: LLD-arch.
- #28  4×4 transform paths in real-time — exercised by KF/intra-only?
       **NEW: L-71** — owner: LLD-transforms.
- #32  VP9_COMP per-field commentary gaps → expand in Stage 2 LLD-arch.
       **NEW: L-72** — owner: LLD-arch.

### Low (8)
- #12  Worker thread RTCD init ordering → **R-09** (already present).
- #16  A2's "30 iterations" claim not validated → **L-40** (already present).
- #20  cpu-used 5 vs 6 boundary scope question.
       **NEW: L-73** — Arthur confirms test scope in PLAN.md.
- #21  Tile-rows assertion under SVC top-resolution → **L-62** (present).
- #22  BufferPool 12 vs 8-slot API exposure → **D-08** (already present).
- #24  Compound mode handling at speed 6+ → **L-46** (already present).
- #27  Recode loop dead-code verification → **L-34** (already present).
- #31  Trellis-quant skip on cpu-used 6-8 → **L-28** (already present).

---

## Updated running total

After A1–A7 integration (A8 pending):

- **D-prefix** (Stage 3 blockers): **13** (D-01 through D-13)
- **L-prefix** (LLD decisions): **41** (L-01 through L-73, with gaps where
  IDs were skipped; effective count is 41 active entries)
- **R-prefix** (risks): **10** (R-01 through R-10; R-01 RESOLVED below)

### Resolved entries (kept for audit trail; remove when archiving)

- **R-01** A4 truncation — RESOLVED 2026-05-16 by async A4 rewrite.
- **R-07** Laplacian R-D tables NOT transcribed — A4 likewise expanded;
  verify if A4's §5.7 or §5.8 covers `rate_tab_q10`/`dist_tab_q10` (they
  live in `vp9/encoder/vp9_rd.c`, possibly out of A4 scope). If still
  missing, A2 patch remains needed (per **L-42**).
