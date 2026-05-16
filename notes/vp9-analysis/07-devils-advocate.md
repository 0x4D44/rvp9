# Devil's advocate: hostile review of Stage 1 specialist outputs

**Reviewer:** Stage 1 driver (manual fallback while async A7 agent ran).
**Source docs reviewed:** A1 (3081 ll), A2 (2311 ll), A3 (1896 ll),
A4 (1599 ll + §6/§7 driver-synthesized), A5 (1979 ll), A6 (2002 ll).
**Date:** 2026-05-16.

> **Note on provenance:** This document was produced manually while the
> async A7 agent ran in background (agentId `abdbfa0864dec8082`). When
> the async A7 lands, the synthesizer (A8) should compare and merge —
> any challenges raised by the async agent NOT present below should be
> appended; any challenges below that the async agent has also raised
> serve as cross-validation.

The challenges below are categorised:
- **H_NN**: hypothesis tests stated in the A7 dispatch prompt
- **A_NN**: cross-references to specialist A_N's §7 or §M where applicable
- **NEW**: novel finding from cross-reading the six specialists

Each has: statement, affected docs, libvpx file:line evidence, impact,
severity (Blocker/High/Medium/Low), resolution.

---

### 1. A4 truncated mid-§5; §6 and §7 are driver-synthesized (H1, R-01)

**Affects:** 04-bitstream-entropy.md §6, §7.

**libvpx evidence:** Original A4 transcript ended at line 1599 of
`04-bitstream-entropy.md` (commit `cce8189`), mid-table inside §5.
The Stage 1 driver appended §6 + §7 (commit `cce8189` notes).

**Why it bites us:** The driver-synthesized §7 has 10 entries but is
not based on a code walk; it's inferred from cross-references in §2–§5
of the agent's output. Specifically: did the agent transcribe the
segmentation probability tables (`vp9_default_pred_probs`,
`vp9_default_segment_tree_probs`)? Skip probability defaults? Tx-size
probability defaults? Inter-pred filter probability defaults? If §5
stopped mid-coefficient tables, all those are missing.

**Severity:** **High.** Probability tables are load-bearing for byte
parity. Any missing table forces Stage 3 to re-read libvpx anyway.

**Resolution:** Re-dispatch A4 with explicit scope "verify §5 covers
ALL probability tables enumerated below: segmentation, skip, tx_size,
intra (kf + non-kf), inter, partition (kf + non-kf), interp_filter,
MV, coefficient. List which are present and which are not."

---

### 2. Laplacian R-D tables `rate_tab_q10` / `dist_tab_q10` not transcribed (H2, R-07)

**Affects:** 02-encoder-core.md §5.4 (declared gap); 04-bitstream-entropy.md §5 (not covered).

**libvpx evidence:** `vp9/encoder/vp9_rd.c` lines ~410-510 contain
`vp9_model_rd_norm` and the associated `rate_tab_q10[256]` /
`dist_tab_q10[256]` lookup tables.

**Why it bites us:** A2 §4 documents `model_rd_for_sb_y[_large]`
calling `vp9_model_rd_from_var_lapndz`, which reads these tables.
Without verbatim transcription, Stage 3 can't byte-parity the non-RD
cost approximation. A2 declared the gap; A4 didn't pick it up because
the tables live in encoder/, not common/.

**Severity:** **Blocker** for Stage 4a (encoder MVP needs these tables
to compute non-RD costs).

**Resolution:** A8 arbitrates ownership — either A2 patch (encoder-
internal tables stay with A2) or A4 re-dispatch with explicit scope
covering `vp9_rd.c` Laplacian tables.

---

### 3. AQ modes 1/2 unreachable on real-time builds — PLAN.md mismatch (H4, R-02)

**Affects:** 05-rate-control.md §3 (AQ integration), PLAN.md §2.

**libvpx evidence:** `vp9_encoder.c:4206-4239` wraps VARIANCE_AQ,
EQUATOR360_AQ, COMPLEXITY_AQ, LOOKAHEAD_AQ in
`#if !CONFIG_REALTIME_ONLY`. CYCLIC_REFRESH_AQ is unconditional.

**Why it bites us:** PLAN.md §2 commits to "Adaptive quantization (3
AQ modes, matching libvpx)". A5 confirms only mode 3 is real-time-
reachable. Either:
- We un-gate `CONFIG_REALTIME_ONLY` for modes 1/2 in our port
  (extra LLD work; matches PLAN.md text)
- We drop modes 1/2 from v1 (simpler; PLAN.md needs editing)

**Severity:** **High.** Affects Stage 4b scope.

**Resolution:** D-05 in decisions-needed.md — Arthur decides at Stage 2.

---

### 4. Denoiser SIMD bit-parity caveat for `increase_denoising=1` (H5, R-03)

**Affects:** 05-rate-control.md Q-RC-23.

**libvpx evidence:** `vp9_denoiser.c:65-67` TODO acknowledges scalar C
is NOT bit-exact with SSE2 in this configuration.

**Why it bites us:** Our PLAN.md D8 makes byte-parity a diagnostic
tool but BD-rate parity a release gate. A divergence-allowed kernel
breaks the diagnostic flow: when we diff rvp9 output against libvpx
output, we can't tell "rvp9 bug" from "libvpx self-inconsistency."

**Severity:** **Medium.** Affects Stage 5 hardening (fuzz oracles).

**Resolution:** R-03 in decisions-needed.md. Differential testing
strategy: pick scalar libvpx as the canonical reference; treat SIMD
variants as "matches scalar within libvpx's own ULP tolerance."

---

### 5. `vp9_encodedframe_overshoot` is opaque (H6, R-05)

**Affects:** 05-rate-control.md Q-RC-20.

**libvpx evidence:** `vp9_ratectrl.c:3248-3351` — single function
mutates `*q`, `cyclic_refresh->counter_encode_maxq_scene_change`,
`rate_correction_factors`, `avg_frame_qindex[INTER_FRAME]`,
`buffer_level`, `bits_off_target`, `rc_1_frame`, `rc_2_frame`, plus
all SVC layers if applicable.

**Why it bites us:** It's called after a *tentative* pack and can
re-write state used by subsequent frame's encode. A5 calls it the
"single most opaque step in the pipeline." Silent BD-rate trap.

**Severity:** **High.** Direct hit to BD-rate parity if mis-ported.

**Resolution:** L-19 in decisions-needed.md. Stage 4a should
implement this with extensive byte-parity testing against libvpx
under scene-change inputs.

---

### 6. Encoder-side frame-parallel encoding not in libvpx (H3)

**Affects:** 06-threading-svc.md §1, PLAN.md Q4.

**libvpx evidence:** A6 explicit claim; verified by absence of any
`pp_lookahead_ctx` / multi-frame-in-flight machinery in
`vp9_encoder.c`. `frame_parallel_decoding_mode` is a *decoder* hint
written into the bitstream.

**Why it bites us:** PLAN.md Q4 asks about frame-parallel encoding in
v1 vs v2. The answer is: there's no libvpx precedent to mirror.
Either we design it ourselves (high cost, deviates from D6 "mirror
libvpx") or defer.

**Severity:** **Medium.** Affects 1080p60 real-time feasibility.

**Resolution:** D-04 in decisions-needed.md. Default: defer to v2;
re-test at Stage 4a if 1080p60 misses budget without frame-parallel.

---

### 7. `cpi->td` aliasing invariant not catalogued comprehensively (H7)

**Affects:** 01-architecture.md §2.4 (calls it the "master invariant");
06-threading-svc.md §4 (covers tile-MT in detail).

**libvpx evidence:** `vp9_ethread.c:209-211, 244`: workers copy
`thread_data->td->mb = cpi->td.mb` OUT at frame start and accumulate
`td->counts` BACK. But what about: cyclic refresh segmentation map
(`cpi->cyclic_refresh->map`), lookahead buffer
(`cpi->lookahead`), denoiser running state
(`cpi->denoiser.running_avg_y`), source-SAD averages
(`cpi->source_sad`), `vp9_scale_factors` for ref frames?

**Why it bites us:** Any field that's WRITTEN by tile workers and not
explicitly in the "copy out / accumulate back" protocol becomes a
hidden aliasing violation in our Rust port.

**Severity:** **High.** Threading safety in our Rust port hinges on
exhaustive knowledge of which fields workers mutate.

**Resolution:** Re-dispatch A1 or A6 with focused scope: produce a
complete table of "field × worker-write?" for every field of
VP9_COMP, ThreadData, MACROBLOCK, and the relevant per-frame
contexts. Or fold into Stage 2 HLD-threading work.

---

### 8. Adaptive-RD threshold non-determinism at default (H8, R-08)

**Affects:** 06-threading-svc.md §7.7.

**libvpx evidence:** `adaptive_rd_thresh_row_mt = 0` is the default
under speed < 7. A6 §7 explicitly: setting to 1 (speed ≥ 7 conditions)
restores determinism.

**Why it bites us:** PLAN.md D1 targets cpu-used 6 as well as 7-8. At
cpu-used 6, libvpx's own behaviour is non-deterministic across thread
counts. Byte-parity testing falls apart.

**Severity:** **High** for cpu-used 6 byte-parity testing.

**Resolution:** R-08 in decisions-needed.md. A6 recommends "bit-exact
by default" (force adaptive_rd_thresh_row_mt=1 at cpu-used 6 in our
port) — divergence-ledger entry. Accept BD-rate cost (small).

---

### 9. Probability table sharing across encoder/decoder needs explicit ownership doc (H9)

**Affects:** 04-bitstream-entropy.md §7 #1 (driver-synthesized).

**libvpx evidence:** `vp9/common/vp9_entropy.{h,c}`,
`vp9_entropymode.{h,c}`, `vp9_entropymv.{h,c}` — all in common/
because both encoder and decoder reference them.

**Why it bites us:** Our PLAN.md §5 commits to "shared via
rvp9-core." But which encoder paths read these defaults vs. read
*adapted* tables (forward + backward adaptation)? Without a clear
ownership map, Stage 3 will accidentally have decoder updating tables
the encoder also wrote.

**Severity:** **Medium.** Caught by conformance tests but easier to
design right.

**Resolution:** L-09 in decisions-needed.md, plus Stage 2 LLD owner
(probably `entropy.md` LLD).

---

### 10. Conformance vector coverage gaps (H10)

**Affects:** Stage 0 `tests/conformance/vectors/`.

**libvpx evidence:** `third_party/libvpx/test/test-data.sha1` has 873
lines (~430 VP9 vectors). Stage 0 seeded only 14.

**Why it bites us:** Our 14 covers basic quantizer + sharpness + size
+ delta-Q + resize + a few tile configs + AQ + show-existing-frame.
Missing: temporal SVC vectors (`vp90-2-10-svc-*`), spatial SVC,
high-bit-depth (if in scope), error-resilient mode, frame parallel
mode.

**Severity:** **Medium.** Will surface in Stage 3 decoder testing.

**Resolution:** Expand `scripts/fetch-vectors.sh` at Stage 3. Not a
Stage 1 blocker.

---

### 11. CONFIG_VP9_HIGHBITDEPTH leakage in real-time paths (H11)

**Affects:** All specialists (preserve-on-leak risk).

**libvpx evidence:** `tran_low_t` typedef in
`vpx_dsp/vpx_dsp_common.h` switches between int16_t and int32_t based
on `CONFIG_VP9_HIGHBITDEPTH`. Many encoder structs have HBD-conditional
fields (`x->plane[i].src_buf_high` etc).

**Why it bites us:** Even excluding HBD from v1 (D-01), some struct
layouts and types change. Our Rust port commits to D7 (mirror libvpx
integer widths). If we pick 8-bit-only, our types differ from
libvpx's defaults, breaking some byte-parity diagnostics.

**Severity:** **Medium.** Pure design question.

**Resolution:** D-01 in decisions-needed.md. Resolve in HLD-arch
(Stage 2).

---

### 12. Worker thread RTCD init ordering (H12, R-09)

**Affects:** 01-architecture.md §7.5, 06-threading-svc.md §7.10.

**libvpx evidence:** `once(vp9_initialize_enc)` at
`vp9_encoder.c:2400-ish` triggers `vpx_dsp_rtcd()` /
`vp9_rtcd()`. Workers may be allocated lazily by
`vp9_encode_tiles_*`.

**Why it bites us:** If we use lazy worker creation and a worker
spawn-then-execute can happen before `once` completes, the workers
see a partially-initialised RTCD table. Undefined behaviour in
libvpx; in Rust, this needs explicit synchronization.

**Severity:** **Low** if we go eager workers; **Medium** if lazy.

**Resolution:** L-31 in decisions-needed.md, plus design constraint in
HLD-threading.

---

### 13. `setjmp` panic mechanism interaction with worker threads (H13)

**Affects:** 01-architecture.md §7.4.

**libvpx evidence:** `cm->error.jmp` is the longjmp target. Set by
`setjmp` in the main thread's `vp9_get_compressed_data` call paths.
What happens if a worker thread hits `vpx_internal_error`?

**Why it bites us:** longjmp across threads is undefined behaviour in
C. libvpx must have some convention here — possibly workers don't
allocate (no OOM possible in worker), or workers communicate failure
back via shared state. Unclear from A1.

**Severity:** **Medium.** Affects our error-handling design.

**Resolution:** Re-dispatch focused on `vpx_internal_error` call
sites in tile-worker paths, OR Stage 2 HLD-arch documents.

---

### 14. Cyclic refresh segmentation map ownership (H14)

**Affects:** 05-rate-control.md (covers CR algorithm); 01-architecture.md (should list
the field).

**libvpx evidence:** `cpi->cyclic_refresh->map` and `cpi->segmentation_map`.

**Why it bites us:** These maps are READ by workers (to know which segment
each block belongs to) and WRITTEN by main thread (between frames). The
read/write boundary determines whether the map needs `Mutex`, `Atomic`,
or just a read-only-after-init shared reference per frame.

**Severity:** **Medium.** Pure aliasing-design question.

**Resolution:** L in decisions-needed.md (add). Owner: HLD-threading.

---

### 15. `variance_low[25]` exact field layout / packing (H15)

**Affects:** 02-encoder-core.md (calls out connective-tissue role).

**libvpx evidence:** `vp9_block.h:158` `int variance_low[25];` —
1 64×64 + 2 64×32 + 2 32×64 + 4 32×32 + 16 16×16. Accessed via the
`pos_shift_16x16[4][4]` switch in `get_force_skip_low_temp_var`.

**Why it bites us:** This is the connective tissue between partition
decision and inter-mode skip. The exact 25-entry packing is a hand-
designed encoding that must be preserved verbatim or rebuilt cleanly.
A2 §7 Q14 flags this.

**Severity:** **Medium.** Byte parity hinges on exact ordering.

**Resolution:** L-39 in decisions-needed.md. Stage 2 LLD for
mode-decision.

---

### 16. NEW: A2's mode-loop "30 iterations" claim not validated by RTCD/SIMD count

**Affects:** 02-encoder-core.md §4 (claims `vp9_pick_inter_mode` has
"~30 iterations" through ref_mode_set + sub8x8).

**libvpx evidence:** 12-entry `ref_mode_set` × possibly 3
filter-candidates × possibly compound = up to ~36+ mode candidates.

**Why it bites us:** The actual iteration count varies with
`inter_mode_mask`, ref-frame availability, and speed features. A2's
"30" is a rough number, not a verified loop bound. If our port uses
an array of size 30, it will overflow in edge cases.

**Severity:** **Low.** Easy to fix once observed.

**Resolution:** L-40 in decisions-needed.md.

---

### 17. NEW: A5's `vp9_scene_detection_onepass` "always called" claim — under SVC?

**Affects:** 05-rate-control.md Q-RC-26.

**libvpx evidence:** Called from `vp9_encoder.c` before
`get_one_pass_*_params`. But is it called per spatial layer or once
per superframe? A5 doesn't disambiguate.

**Why it bites us:** SVC introduces additional encode passes; scene
detection per layer would be 3× compute for 3-layer spatial SVC.
Mirror exactly.

**Severity:** **Medium.** Affects SVC CPU budget.

**Resolution:** Stage 2 LLD-svc; verify by reading
`vp9_one_pass_svc_*` paths.

---

### 18. NEW: Frame dropping interactions across SVC layers

**Affects:** 05-rate-control.md (frame-drop algorithms); 06-threading-svc.md
(SVC structure).

**libvpx evidence:** `vp9_rc_drop_frame` at `vp9_ratectrl.c:1660ish`
and SVC-aware drop logic. If layer N drops, layer N+1 (which may
reference it) is forced to drop too, breaking the reference structure.

**Why it bites us:** Cross-layer drop semantics aren't explicitly
documented in either A5 or A6. Bug-prone.

**Severity:** **Medium.** WebRTC-critical.

**Resolution:** Stage 2 LLD-svc owns; A8 should ensure cross-reference
between RC and SVC docs catches this.

---

### 19. NEW: `is_altref_enabled` returns 0 for REALTIME+CBR — but VBR-RT?

**Affects:** 01-architecture.md (key finding).

**libvpx evidence:** A1 reports `is_altref_enabled == 0` for REALTIME
+ CBR. PLAN.md commits to both CBR AND VBR rate control. Does VBR-RT
enable altref?

**Why it bites us:** Altref support roughly doubles the frame-loop
complexity (hidden frames, reference-frame rotation, alt-ref interval
selection). If REALTIME-VBR uses altref, we need it in v1.

**Severity:** **High.** Affects Stage 4a scope.

**Resolution:** Verify via `vp9_extend_active_map` / VBR-RT setup
paths. Add to decisions-needed.md.

---

### 20. NEW: `cpu-used` 5 vs 6+ split is the use_nonrd_pick_mode boundary

**Affects:** 06-threading-svc.md key finding.

**libvpx evidence:** `set_rt_speed_feature_framesize_independent` at
speed 5 sets `sf->use_nonrd_pick_mode = 1`. This is the qualitative
split between RD-loop and non-RD-loop encoding.

**Why it bites us:** PLAN.md D1 says cpu-used 6-8 priority. cpu-used 5
also uses non-RD path. But PLAN.md Stage 4a targets cpu-used 7 by
default. Should we also test cpu-used 5 and 6 for completeness, or
target only cpu-used 6, 7, 8?

**Severity:** **Low.** Pure scope question.

**Resolution:** Confirm in PLAN.md / decisions-needed.md. Default:
test 6/7/8 in CI, allow 5 as a "best quality real-time" knob.

---

### 21. NEW: A6's "tile-rows always 0" assertion — verified for SVC?

**Affects:** 06-threading-svc.md §7.12.

**libvpx evidence:** A6 §7.12 recommends asserting `log2_tile_rows == 0`
in v1 (real-time never uses tile rows). But does SVC enabling at any
layer trigger tile-rows? Real-time SVC at 4K spatial layer could
plausibly need them.

**Why it bites us:** Hard-coded assumption that we can never enable
tile-rows would silently fail at SVC top-layer 4K.

**Severity:** **Low** (4K SVC isn't in our target res per PLAN.md §2).

**Resolution:** L-62 in decisions-needed.md. Keep assertion in v1;
revisit if SVC at 4K is added later.

---

### 22. NEW: `BufferPool` 12-slot capacity — vs 8-ref-frame VP9 spec

**Affects:** 01-architecture.md key finding.

**libvpx evidence:** A1: 8 ref + 1 new + 3 scaled = 12 slots.

**Why it bites us:** VP9 spec defines 8 reference frame slots. The
"3 scaled" are libvpx-internal (for SVC inter-layer prediction and
resize). Our Rust port should NOT expose them in the public API;
they're an implementation detail.

**Severity:** **Low.** API design question.

**Resolution:** D-08 in decisions-needed.md.

---

### 23. NEW: Sub-pixel filter selection in `search_filter_ref`

**Affects:** 02-encoder-core.md §7 (Q11); 03-motion-estimation.md (calls into).

**libvpx evidence:** `vp9_pickmode.c:search_filter_ref` selects from
EIGHTTAP / EIGHTTAP_SMOOTH / EIGHTTAP_SHARP / BILINEAR based on a
quick SAD test. The exact selection logic isn't transcribed in A2
or A3.

**Why it bites us:** Filter choice affects every sub-pixel-refined
inter block. Off-by-one in selection threshold = different bitstream.

**Severity:** **Medium.** Direct byte-parity risk.

**Resolution:** Re-read `vp9_pickmode.c` in Stage 2 LLD-mode-decision.

---

### 24. NEW: Compound mode handling at speed 6+ — claimed dead

**Affects:** 02-encoder-core.md §7 Q9.

**libvpx evidence:** `comp_modes = 0` at cpu-used ≥ 5 per
`set_rt_speed_feature_framesize_independent`. So `vp9_pick_inter_mode`'s
compound-mode branch is dead code in our scope.

**Why it bites us:** A2 didn't fully prove this — we may have missed
a code path that enables compound modes (e.g. for ALTREF interactions).

**Severity:** **Low.** A defensible exclusion.

**Resolution:** L-46 in decisions-needed.md. Drop compound mode from
Stage 4 LLD; add divergence-ledger entry.

---

### 25. NEW: External rate-control plug-in (EXT_RATECTRL) — WebRTC dependency

**Affects:** 01-architecture.md §7.7; 05-rate-control.md (not deeply covered).

**libvpx evidence:** `cpi->ext_ratectrl` plumbing in
`vp9_encoder.c` (~5+ call sites). WebRTC (libwebrtc) uses this
heavily for its own loss-aware rate control.

**Why it bites us:** PLAN.md doesn't mention EXT_RATECTRL. If our
v1 target includes "WebRTC viability" (D3 implies it), and WebRTC
uses EXT_RATECTRL extensively, we need it. R-06 in decisions-needed.

**Severity:** **High** if WebRTC is a real v1 target user.

**Resolution:** Arthur clarifies the WebRTC integration model.
Default: include EXT_RATECTRL as opt-in API surface.

---

### 26. NEW: `vp9_change_config` reallocation behaviour on dimension change

**Affects:** 01-architecture.md (mode dispatch).

**libvpx evidence:** `vp9_change_config` in `vp9_encoder.c` —
called every encode if user changes config. Reallocates buffers if
dimensions changed.

**Why it bites us:** Real-time use cases often change config
mid-stream (bitrate ramp, layer drop). Our port needs to handle
this gracefully without dropping frames.

**Severity:** **Medium.** Real-time usability concern.

**Resolution:** Stage 4a LLD covers; document API contract.

---

### 27. NEW: Recode loop NOT in cpu-used 6-8 — but is the *function* dead, or just unused?

**Affects:** 01-architecture.md §7.9.

**libvpx evidence:** `encode_with_recode_loop` exists; called from
`vp9_encode_frame` if `sf.recode_loop != DISALLOW_RECODE`. At
cpu-used 6-8, `sf.recode_loop = DISALLOW_RECODE`.

**Why it bites us:** We can drop the function from our Rust port
(per L-34) — but then `encode_without_recode_loop` becomes the
only encode path. Verify NO speed-feature-dependent code in
real-time touches the recode-loop variables (e.g. `cpi->frames_since_key`,
`cpi->oxcf.recode_loop`).

**Severity:** **Low.** Easy to verify.

**Resolution:** Stage 2 LLD-arch confirms.

---

### 28. NEW: 4×4 transform paths in real-time — are they exercised?

**Affects:** 02-encoder-core.md (mentions TX_16X16 cap); 04-bitstream-entropy.md (scan orders for 4×4).

**libvpx evidence:** A2 says non-RD uses TX_16X16 max; `block_yrd`
won't recurse to 4×4. But 4×4 intra-block-prediction (for intra-only
KF blocks) does.

**Why it bites us:** A4 transcribed 4×4 scan orders verbatim;
implementing 4×4 transforms in Stage 3 might be wasted effort if
real-time NEVER hits 4×4 transform. Need to confirm with KF +
intra-only frame analysis.

**Severity:** **Low.** Affects Stage 3 scope.

**Resolution:** Stage 2 LLD-transforms confirms.

---

### 29. NEW: Speed-feature setter is per-frame condition-tuple dependent (R-10)

**Affects:** 06-threading-svc.md key finding.

**libvpx evidence:** `set_rt_speed_feature_framesize_independent`
runs every frame in the recode loop and depends on `cm->frame_type`,
`cm->base_qindex`, SVC state, source-SAD scene-cut, noise estimate,
previous-frame deadline mode.

**Why it bites us:** Byte-parity testing for speed features can't
just compare (speed, w, h) tuples; needs full condition tuple. Test
harness must replay exact frame sequences.

**Severity:** **High** for byte-parity CI.

**Resolution:** R-10 in decisions-needed.md. Stage 2 HLD-threading
documents the test harness shape.

---

### 30. NEW: A4 §5 likely missing inter_mode probability updates

**Affects:** 04-bitstream-entropy.md §5 (truncated).

**libvpx evidence:** `vp9_entropymode.{h,c}` ships
`vp9_inter_mode_tree`, `default_inter_mode_probs`, plus the per-frame
`vp9_inter_mode_prob_diff` machinery. The truncated A4 may have stopped
before transcribing these.

**Why it bites us:** Inter-mode probabilities are core to byte-parity;
any missing default trips Stage 3 conformance.

**Severity:** **High** (paired with #1 above).

**Resolution:** Re-dispatch A4 (same scope as resolution to #1).

---

### 31. NEW: Trellis-quant skip on cpu-used 6-8 — verified in A4?

**Affects:** 02-encoder-core.md (mentions `vp9_optimize_b`); 04-bitstream-entropy.md.

**libvpx evidence:** `sf->optimize_coefficients = 0` at cpu-used ≥ 5
disables `vp9_optimize_b`. A4 §7 #3 (driver-synthesized) flagged this.

**Why it bites us:** A4's original §5 covers coefficient tokens; if
trellis-quant is off, the post-quant token stream is direct. Our
Rust port can skip `vp9_optimize_b` entirely in v1.

**Severity:** **Low.** Defensible exclusion.

**Resolution:** L-28 in decisions-needed.md.

---

### 32. NEW: A1 §2 god-struct list — VP9_COMP has ~120 fields but A1 lists ~30 explicitly

**Affects:** 01-architecture.md §2.

**libvpx evidence:** `vp9_encoder.h:727-1069` (343 lines, ~120 fields).

**Why it bites us:** A1 §2 reproduces VP9_COMP verbatim (per the
agent's report) but the field-by-field semantics commentary may not
cover all 120. Some fields are obvious (config, common); some are
subtle (`frame_info`, `tpl_*`, `mbgraph_*`, `kmeans_data_*`). The
LLD writer must understand every field.

**Severity:** **Medium.** Affects LLD completeness.

**Resolution:** Stage 2 HLD-arch + LLD-encoder-shell expand.

---

## Disposition summary

For A8 synthesizer to digest:

- **Blocker (1)**: #2 (Laplacian R-D tables)
- **High (10)**: #1, #3, #5, #7, #8, #19, #25, #30 — plus #4, #29 partially
- **Medium (15)**: #4, #6, #9, #10, #11, #13, #14, #15, #17, #18, #23, #26, #32
- **Low (6)**: #12, #16, #20, #21, #22, #24, #27, #28, #31

**Re-dispatches recommended:** A4 (#1, #30) to verify probability-table
coverage; A1 or A6 (#7) to catalogue worker-mutated fields exhaustively;
optionally A1 (#13) for setjmp/threading interaction.

**Stage 2 LLD owners flagged:** HLD-arch (#11, #19, #22, #25, #26, #32),
HLD-threading (#7, #8, #14, #29), LLD-mode-decision (#15, #23, #24),
LLD-rate-control (#18), LLD-svc (#17, #21).
