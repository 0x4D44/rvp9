# decisions-needed.md — DRAFT (in-flight)

> **Status:** in-progress seed from A3 + A4 + A5 specialist outputs.
> Will be completed and renamed to `decisions-needed.md` (no underscore)
> after A1, A2, A6, A7, A8 land. Cross-references use specialist §7
> question IDs (Q-A3-NN, Q-RC-NN — A4's were renumbered into Q-BE-NN
> here for consistency).

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

## Items still missing (await A1, A2, A6)

- L-NN for architecture decisions (A1) — error model, panic policy
- L-NN for encoder mode-decision (A2) — non-RD cost approximations,
  partition decision tree, skip detection
- L-NN for threading + SVC + speed features (A6) — UnsafeCell catalogue,
  Send/Sync boundary, speed-feature flag enumeration
- All A7 challenges that result in DEFERRED entries here
