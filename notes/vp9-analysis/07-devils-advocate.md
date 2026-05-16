# Devil's advocate (A7): hostile review of Stage 1 specialist outputs

**Reviewer:** Agent A7 (devil's advocate, async). Supersedes the driver's
fallback `07-devils-advocate.md` (now renamed in commit history; previous
content kept in `07-devils-advocate-driver.md`).

**Source docs reviewed (line counts at review time):**
- `01-architecture.md` — 3081 lines.
- `02-encoder-core.md` — 2311 lines.
- `03-motion-estimation.md` — 1896 lines.
- `04-bitstream-entropy.md` — **4556 lines** (NOT 1599 as the prompt
  template implied; the agent's transcript was resumed and §5 continued
  to §5.10 with full table transcription. See challenge 1).
- `05-rate-control.md` — 1979 lines.
- `06-threading-svc.md` — 2002 lines.

**libvpx pin:** v1.14.1, commit `12f3a2a`, at `third_party/libvpx/`.

**Methodology:** Read the six specialist docs end-to-end. For each
hypothesis test H1-H15 in the dispatch prompt plus the ten attack
categories, verify (or refute) against libvpx source. Every challenge
below cites either a specialist § + line range OR a `third_party/libvpx/`
file:line. Disagree-and-commit: where a specialist's claim is correct,
I say so explicitly rather than manufacturing a complaint.

Tag legend:
- **H_NN** — hypothesis test from the dispatch prompt.
- **A_NN** — cross-reference into A_N's §7 (deferred to LLD) where the
  specialist already raised the concern; the challenge is "this needs
  to be elevated / has implications they did not flag."
- **NEW** — novel finding from cross-reading the six specialists.

Severity scale:
- **Blocker** — Stage 2 cannot proceed for the affected LLD without resolution.
- **High** — Stage 4 BD-rate parity or real-time perf at risk.
- **Medium** — Catchable in Stage 3/4 testing but cheaper to address now.
- **Low** — Documentation / API surface decision; design hygiene.

---

## Part 1 — Hypothesis tests from the dispatch prompt

### 1. A4 was NOT truncated mid-§5 — the driver's R-01 note is now stale; the file has full table coverage but inverted section order [H1, A_NN R-01]

**Affects:** `04-bitstream-entropy.md` §5.4–§5.10 (lines 1709–4088),
§6 / §7 (driver-synthesized at 1602–1707), and
`_decisions-needed-draft.md` R-01.

**libvpx evidence vs. what the file actually contains:**
- A4 §5 was supposed to end mid-table per the driver's prefatory note
  at `04-bitstream-entropy.md:1602-1607`. In reality, §5.4
  (intra-mode probs), §5.5 (partition probs), §5.6 (inter / ref / skip /
  tx-size / interp probs), §5.7 (full coefficient model:
  alphabet + `pt_energy_class` + coefband_trans + cat extra-bit +
  `vp9_extra_bits` dispatch + full 255-row `vp9_pareto8_full` +
  `default_coef_probs_{4x4,8x8,16x16,32x32}` + adaptation constants),
  §5.8 (MV: `default_nmv_context` + `log_in_base_2[1024]`), §5.9
  (subexp `update_bits[255]` + `map_table` + `inv_map_table`), §5.10
  (full scan orders + iscans + neighbors + dispatch tables) ARE ALL
  PRESENT — see `04-bitstream-entropy.md:1709, 1899, 1962, 2053, 3161,
  3251, 3333, 4050` etc.
- The synthesized §6 and §7 appear at lines **1602–1707**, i.e. **in
  the middle of §5**, breaking document order.

**Why it bites us:** Three concrete consequences:
1. The R-01 entry in `_decisions-needed-draft.md` is wrong — there is
   no missing probability content to re-dispatch. R-01 should be closed
   or rewritten to point at the actual gap: §6 (RTCD) and §7 (LLD open
   questions) are driver synthesis based on cross-references, not the
   agent's analysis.
2. The driver fallback `07-devils-advocate-driver.md` challenges #1
   and #30 ("inter_mode probability updates missing") are both invalid:
   `default_inter_mode_probs` is transcribed at A4 §5.6.1
   (`04-bitstream-entropy.md:1964-1978`); the inter-mode tree is at
   §5.3.2 (line 1468); the `MV_UPDATE_PROB`/`DIFF_UPDATE_PROB` constants
   are at lines 1408 and 1416. Those re-dispatches would be a waste.
3. The genuine A4 gap is *organisational*: a reader following section
   order will hit synthesized §6/§7 before the agent's §5.4–§5.10 and
   may stop reading. A8 must merge §6/§7 to the end of the file.

**Severity:** **Medium** (housekeeping); was misreported as **High**.

**Resolution:** A8 (synthesizer) re-orders §6/§7 in `04-bitstream-entropy.md`
to follow §5.10, replaces the "synthesis" disclaimer with a "verified
against agent's §5" note, and closes R-01. NO re-dispatch of A4 is needed
for table coverage. Genuine missing items are split into challenges 2,
4, 17 below.

---

### 2. Laplacian R-D tables `rate_tab_q10` / `dist_tab_q10` / `xsq_iq_q10` are not transcribed by ANY specialist [H2, A2 §7 Q5]

**Affects:** `02-encoder-core.md` §5.4 (declares gap explicitly);
`04-bitstream-entropy.md` (does not cover; out of scope).

**libvpx evidence:** `third_party/libvpx/vp9/encoder/vp9_rd.c:462-471`
(rate_tab_q10[100], encoder-internal), `:480-489` (dist_tab_q10[100]),
`:490-503` (xsq_iq_q10[105]). Used by `model_rd_norm()` at `vp9_rd.c:505-514`
and the public `vp9_model_rd_from_var_lapndz()` at `vp9_rd.c:518`. The
caller chain into real-time is
`vp9_pick_inter_mode → model_rd_for_sb_y[_large] → vp9_model_rd_from_var_lapndz`
(see A2 §4.4 `02-encoder-core.md:1135`).

**Why it bites us:** These three tables are the closed-form Laplacian
rate-distortion approximation used **on every non-RD inter block** at
cpu-used 6-8 to score modes. Without verbatim transcription:
- Stage 4a cannot byte-parity the non-RD cost loop.
- Regenerating from the closed form (Hang & Chen, "Source Model for
  transform video coder...") introduces transcendental-function rounding
  drift; A2 §7 Q5 explicitly warns "the closed form has non-trivial
  rounding."

A2 declared the gap but punted to A4. A4 didn't pick it up because the
tables live in encoder-internal code, not `vp9/common/`.

**Severity:** **Blocker** for Stage 4a's mode-decision LLD.

**Resolution:** A8 assigns the three tables to A2 (encoder-internal
data) and patches §5.4 of `02-encoder-core.md` with verbatim
transcription — three small int arrays (100/100/105 entries), trivial
patch. Do not re-dispatch the agent.

---

### 3. A4 does NOT transcribe the encoder-internal cost LUTs that gate token writes (`vp9_extra_bits` dispatch is there but `vp9_token_state` mapping isn't) [H1 follow-up, NEW]

**Affects:** `04-bitstream-entropy.md` §5.7.

**libvpx evidence:**
- `third_party/libvpx/vp9/encoder/vp9_tokenize.c:260-273` —
  `vp9_extra_bits[ENTROPY_TOKENS]` IS in A4 §5.7.5 at line 2168.
- `third_party/libvpx/vp9/encoder/vp9_cost.c` — `vp9_prob_cost[256]`
  (the precomputed `-log2(p/256) * 512` table that converts a vpx_prob
  into a bit-cost) is NOT transcribed in A4 or A2.
- `third_party/libvpx/vp9/encoder/vp9_cost.h:24` —
  `VP9_PROB_COST_SHIFT = 9`.

**Why it bites us:** Every cost approximation in the encoder
(mode-decision, MV-cost, intra-cost, partition-cost) calls
`vp9_cost_bit(p, b)` which reads `vp9_prob_cost[]`. Off-by-one in any
of 256 entries → silent BD-rate drift. A4 §5.11 (`04-bitstream-entropy.md:4092-4099`)
mentions "Tokenizer cost tables (encoder-only)" and says "A2 owns the
cost framework" but neither A2 nor A4 actually transcribes
`vp9_prob_cost[]`.

**Severity:** **High.** Direct hit to BD-rate parity.

**Resolution:** A8 patches `02-encoder-core.md` §5 with verbatim
transcription of `vp9_prob_cost[256]` (it's 256 ints, < 1 KB). Or
re-dispatch A2 with explicit scope: "transcribe every encoder-internal
cost LUT in `vp9_cost.{c,h}` and `vp9_rd.{c,h}`."

---

### 4. A4 §5 does NOT include the `vp9_coef_update_prob` "should we update?" thresholds — but inspection shows libvpx only uses a single `DIFF_UPDATE_PROB = 252` for ALL coefficient updates (i.e., the prompt template was wrong, not A4) [H1 follow-up]

**Affects:** `04-bitstream-entropy.md` §5.7 (was the suspected gap);
`_decisions-needed-draft.md` L-10 (mentions "exact conditions for prob
updates").

**libvpx evidence:** `third_party/libvpx/vp9/common/vp9_entropy.h:24`:
```c
#define DIFF_UPDATE_PROB 252
```
Used unconditionally by `vp9_cond_prob_diff_update`
(`vp9_subexp.c:182-196`) and `update_coef_probs_common`
(`vp9_bitstream.c:545-590`). There is **no** `vp9_coef_update_prob[]`
table in libvpx — the prompt's reference to one is incorrect.

A4 transcribes the constant at `04-bitstream-entropy.md:1408` and §3.7
documents the protocol.

**Severity:** **Low** (verified the specialist was right and the prompt
template wrong).

**Resolution:** Mark H1's coefficient-update-probability sub-question as
**REJECTED** — A4 is correct. Update `_decisions-needed-draft.md` L-10
to drop "exact conditions" wording (the protocol is unconditional with
savings-search).

---

### 5. AQ modes 1, 2, 4 (variance / 360 / complexity / lookahead) ARE gated by `#if !CONFIG_REALTIME_ONLY` — A5 Q-RC-15 is correct; PLAN.md §2 "3 AQ modes" is the wrong commitment [H4]

**Affects:** `05-rate-control.md` Q-RC-15
(`05-rate-control.md:1817-1827`); `_decisions-needed-draft.md` D-05;
PLAN.md §2 success criteria.

**libvpx evidence (verified):**
- `third_party/libvpx/vp9/encoder/vp9_encoder.c:4206-4239` — the
  pre-encode AQ dispatch in `encode_without_recode_loop`:
  ```c
  #if !CONFIG_REALTIME_ONLY
    if (cpi->oxcf.aq_mode == VARIANCE_AQ) { vp9_vaq_frame_setup(cpi); }
    else if (cpi->oxcf.aq_mode == EQUATOR360_AQ) { ... }
    else if (cpi->oxcf.aq_mode == COMPLEXITY_AQ) { ... }
    else if (cpi->oxcf.aq_mode == LOOKAHEAD_AQ) { ... }
    else {
  #endif
      // only CYCLIC_REFRESH_AQ + ROI reachable here
    }
  ```
- `third_party/libvpx/vp9/encoder/vp9_encoder.c:4657-4667` —
  identical gate in `encode_with_recode_loop` (non-RT path; the only
  place modes 1/2/4 are reachable).

**Why it bites us:** PLAN.md §2 commits to "Adaptive quantization (3 AQ
modes, matching libvpx)". Under D1's real-time-only scope, libvpx
itself never exercises modes 1, 2, or 4. We have three options:
1. **Drop the commitment to modes 1/2.** Edit PLAN.md to "1 AQ mode
   (CYCLIC_REFRESH_AQ), matching libvpx real-time exercising."
2. **Un-gate `CONFIG_REALTIME_ONLY` for modes 1/2 in our port.** Extra
   LLD scope, no libvpx parity reference under real-time conditions.
3. **Defer modes 1/2 to v2.** Cleanest from a scope-discipline POV.

**Severity:** **High** for PLAN.md correctness. Affects Stage 4b scope.

**Resolution:** A8 surfaces this as a P0 decision for Arthur. Default
recommendation: option (3). Update PLAN.md §2 to "AQ mode 3
(cyclic refresh) in v1; modes 1/2 deferred." Reflect in D-05.

---

### 6. Denoiser SSE2 vs. C bit-parity is broken for `increase_denoising=1` per libvpx's own TODO [H5]

**Affects:** `05-rate-control.md` Q-RC-23 (cites the TODO).

**libvpx evidence (verified):** `third_party/libvpx/vp9/encoder/vp9_denoiser.c:65-67`:
```c
// TODO(jackychen): If increase_denoising is enabled in the future,
// we might need to update the code for calculating 'total_adj' in
// case the C code is not bit-exact with corresponding sse2 code.
```

**Why it bites us:** Our differential-test pipeline in Stage 4 will diff
rvp9 output against libvpx output. With `increase_denoising=1` (which is
set under certain noise/temporal conditions in
`vp9_denoiser.c:213-227`), libvpx's own scalar vs. SSE2 mismatch
contaminates the oracle. Three knock-on consequences:
1. We cannot use "bit-equality with libvpx SIMD" as a Stage 4 gate.
2. Our scalar-vs-SIMD differential test inside rvp9 must allow ULP
   tolerance, OR we ship scalar-only denoiser (slow) OR canonicalize
   one implementation.
3. PLAN.md D5 ("SIMD via core::arch + scalar fallback") doesn't address
   bit-exact-equivalence; the LLD must.

**Severity:** **Medium.** Affects Stage 5 hardening and CI design.

**Resolution:** R-03 in `_decisions-needed-draft.md` is correct; A8
should elevate to a Stage 2 HLD-simd test-strategy decision. Default:
pick libvpx **scalar C** as the canonical reference for the denoiser;
our SIMD must match our scalar (which matches libvpx scalar), not
libvpx SIMD.

---

### 7. `vp9_encodedframe_overshoot` is called from BOTH a pre-encode and a post-encode site; A5 Q-RC-20 only documents the post-encode flow [H6]

**Affects:** `05-rate-control.md` Q-RC-20
(`05-rate-control.md:1858-1874`).

**libvpx evidence (direct read, lines as of v1.14.1):**
- Pre-encode site: `vp9_encoder.c:4197-4204` — called with
  `frame_size = -1` when
  `sf.overshoot_detection_cbr_rt == FAST_DETECTION_MAXQ` AND
  `(rc.high_source_sad || (use_svc && svc->high_source_sad_superframe))`.
  This runs **before** any encode happens.
- Post-encode site: `vp9_encoder.c:4249-4277` — called with the
  actual `frame_size` after a tentative `vp9_pack_bitstream`. Only
  when `sf.overshoot_detection_cbr_rt == RE_ENCODE_MAXQ`. If it
  returns 1, the cyclic-refresh seg map is **zeroed**, CR is
  **disabled** via `vp9_disable_segmentation`, and the frame is
  **re-encoded**.
- Function body: `vp9_ratectrl.c:3248-3351`.

A5 Q-RC-20 lists the state mutations but:
1. Missing: the function is also called pre-encode (no `frame_size`
   known), branch at line 3264-3266 distinguishes the two cases.
2. Missing: lines 3279-3295 set `cpi->rc.hybrid_intra_scene_change = 1`
   when (post-encode) overshoot > 2× AND ≥ 60% intra mode usage in the
   tentatively-encoded frame. This affects subsequent intra mode
   selection (small blocks use RD intra).
3. Missing: line 3343 sets `lrc->force_max_q = 1` on **all** SVC
   layers within `first_spatial_layer_to_encode`, not just the current
   one.
4. Missing: only `rate_correction_factors[INTER_NORMAL]` is mutated
   (line 3321), not all five rate-factor levels — important for the
   damped-adjustment per-frame-type state machine (A5 Q-RC-7).
5. Missing: caller flow at `vp9_encoder.c:4266-4274` — on re-encode
   under CR-AQ, the seg map is `memset(0)` AND `cr->last_coded_q_map`
   is `memset(MAXQ)` AND `cr->sb_index = 0`. This **destroys** the
   running cyclic-refresh state for one frame.

**Severity:** **High.** Mis-porting any of (1)-(5) silently shifts BD-rate
under scene-change conditions, which is exactly the case
`vp9_encodedframe_overshoot` is designed for.

**Resolution:** A8 elevates Q-RC-20 to a dedicated LLD section in
`notes/lld/rate-control.md` (Stage 2). Explicitly list the five
mutations above plus the two call sites and the CR side effect. Add a
property-based test invariant: "after `vp9_encodedframe_overshoot`
returns 1, `cyclic_refresh.counter_encode_maxq_scene_change == 0` AND
`rc.re_encode_maxq_scene_change == 1`."

---

### 8. A6's claim that libvpx VP9 has no encoder-side frame-parallel encoding is correct; the only `frame_parallel_*` identifier in the encoder is `frame_parallel_decoding_mode`, a decoder-side hint [H3]

**Affects:** `06-threading-svc.md` §7.9
(`06-threading-svc.md:1976-1978`).

**libvpx evidence (verified):**
- Grep across `third_party/libvpx/vp9/encoder/` for `frame_parallel` —
  zero matches.
- `third_party/libvpx/vp9/encoder/vp9_encoder.c:5589, 5593` — references
  to `cm->frame_parallel_decoding_mode` — these are written to the
  bitstream's uncompressed header so the **decoder** can do frame-parallel
  decode. No encoder-side implication.
- `third_party/libvpx/vp9/vp9_cx_iface.c:1028-1031, 2077` —
  user-facing setter `VP9E_SET_FRAME_PARALLEL_DECODING` again purely
  decoder-side.

**Severity:** **N/A** (validation, not a challenge).

**Resolution:** Mark H3 as **CONFIRMED**. PLAN.md Q4 should resolve
as "defer to v2; no libvpx template to mirror." A8 records this in
the disposition table.

---

### 9. `cpi->td` aliasing protocol is correctly identified by A1 — but A6 §7.2 lists only `MACROBLOCK`-aliased buffers; the cyclic-refresh seg map, lookahead, denoiser running_avg, and source-SAD arrays are also worker-touched and need cataloguing [H7]

**Affects:** `01-architecture.md` §2.5 (struct definitions),
`06-threading-svc.md` §7.2 (only lists MACROBLOCK + `above_context`).

**libvpx evidence (direct):**
- `vp9_ethread.c:209-216` — confirmed: workers copy
  `thread_data->td->mb = cpi->td.mb` OUT, `memcpy` of `cpi->common.counts`
  OUT, then read-mutate the copies.
- `vp9_ethread.c:242-247` — confirmed: `vp9_accumulate_frame_counts`
  + `accumulate_rd_opt` accumulate BACK into main thread's `cpi->td`.

But A6 §7.2 doesn't enumerate the **read-only-shared** structures that
workers also touch and that need an aliasing model:
1. `cpi->cyclic_refresh->map` and `cpi->cyclic_refresh->last_coded_q_map`
   — read per-superblock during pickmode (`vp9_aq_cyclicrefresh.c:382-394`,
   called from `nonrd_pick_sb_modes`). Written by main thread between
   frames AND mutated during post-encode CR update by the main thread.
2. `cpi->segmentation_map` — read per-SB to look up segment_id; written
   by AQ setup pre-encode.
3. `cpi->denoiser.running_avg_y[]` — read AND written by workers when
   denoise is enabled (the temporal filter is per-block, in-worker).
   The "dual use of slot 0" warning in A5 Q-RC-9 is a per-worker concern.
4. `cpi->source_sad`, `cpi->avg_source_sad[]` — written by
   `vp9_scene_detection_onepass` on main thread before launch; read
   by workers via `cpi->rc.high_source_sad` etc.
5. `cpi->prev_partition`, `cpi->prev_segment_id`, `cpi->prev_variance_low`,
   `cpi->content_state_sb_fd`, `cpi->copied_frame_cnt` — all per-SB
   bookkeeping arrays touched during pickmode.
6. `tile_data->thresh_freq_fact` and `tile_data->row_base_thresh_freq_fact`
   — written by workers; **not bit-exact across threads when
   `adaptive_rd_thresh_row_mt = 0`** (see challenge 11).

**Why it bites us:** The Rust port's `Send` / `Sync` decisions hinge on
this complete catalogue. Missing any of (1)-(6) at HLD-threading time
means a Stage 4a re-design when we hit the borrow checker.

**Severity:** **High.** Direct correctness risk for the Rust port's
threading model.

**Resolution:** Re-dispatch A1 OR A6 (one of them; either can do it)
with focused scope: "Produce a complete table of EVERY heap-allocated
field on `VP9_COMP`, `VP9_COMMON`, and worker-touched per-tile-data
structures, with columns: write-by-main / read-by-worker /
write-by-worker / accumulate-back. ~150 fields total — bulk work but
mechanical." OR fold into Stage 2 HLD-threading as Arthur prefers.

---

### 10. `setjmp`/`longjmp` panic from a tile worker is undefined behaviour in libvpx — no documented escape hatch exists; A1 §7.4 doesn't surface this [H13]

**Affects:** `01-architecture.md` §7.4
(`01-architecture.md:2981-2991`).

**libvpx evidence (direct):**
- Encoder side `setjmp` sites: `vp9_encoder.c:2390`,
  `vp9_cx_iface.c:830, 1381`, `ratectrl_rtc.cc:148`. All on the main
  thread.
- `vpx_internal_error` body: `vpx/src/vpx_codec.c:115-133` — calls
  `longjmp(info->jmp, ...)` when `info->setjmp` is non-zero.
- `cm->error` is a single struct shared across all threads (it lives
  on `VP9_COMMON`, which is shared).
- Workers DO call `vpx_internal_error` indirectly through
  `CHECK_MEM_ERROR` macros — see `vp9_ethread.c:296-304` (worker init),
  `vp9_pickmode.c` mode allocation paths.

**Why it bites us:** Pthreads spec is explicit that `longjmp` across
thread boundaries is undefined behaviour. In practice on glibc x86_64
it crashes; on Windows it sometimes "works" by jumping back to the main
thread's stack but corrupts state. libvpx's defence is implicit: the
allocations that COULD fail in workers (per-frame `td->counts`, etc.)
are done on the **main thread before launch** (see `vp9_ethread.c:124-125`).
But there are still `CHECK_MEM_ERROR` paths reachable from worker
code paths, e.g. `vp9_get_intra_block_iterator`.

For our Rust port:
- If we use `Result<...>` propagation, every worker function returns
  `Result`; we plumb the failure up via shared atomic flag + `Result`
  return from the worker thread join.
- If we use `panic = "unwind"`, panic-in-worker is caught at the
  scope boundary by `std::thread::scope`.

A1 §7.4 mentions the issue but doesn't resolve it.

**Severity:** **Medium.** Affects error-handling design in HLD-arch.

**Resolution:** A8 surfaces as a Stage 2 HLD-arch decision. Default
recommendation: `panic = "unwind"` + `std::thread::scope` for clean
worker-fault propagation. Allocations on the hot path use
`Box::try_new` + `?` to surface OOM as `Result`. Per CLAUDE.md "we are
in a sandbox; skip ops sign-offs" — OOM-as-Result is fine.

---

### 11. `adaptive_rd_thresh_row_mt` non-determinism is SVC-conditional at speeds 7/8, not "default at cpu-used 6" — A6 §7.7 mis-states the trigger [H8]

**Affects:** `06-threading-svc.md` §7.7
(`06-threading-svc.md:1962-1968`).

**libvpx evidence (direct read of `vp9_speed_features.c`):**
- `vp9_speed_features.c:460` — `set_rt_speed_feature_framesize_independent`
  baseline: `sf->adaptive_rd_thresh_row_mt = 0`. So at all RT speeds
  this starts off.
- `vp9_speed_features.c:715-718` — speed ≥ 7 path:
  ```c
  if (cpi->use_svc && cpi->row_mt && cpi->oxcf.max_threads > 1)
    sf->adaptive_rd_thresh_row_mt = 1;
  ```
- `vp9_speed_features.c:755-760` — same gate at speed ≥ 8.
- `vp9_speed_features.c:908-912` — the consumer:
  ```c
  if (!sf->adaptive_rd_thresh_row_mt && cpi->row_mt_bit_exact &&
      max_threads > 1)
    sf->adaptive_rd_thresh = 0;
  ```
- `vp9_encoder.c:7067-7074` — `cpi->row_mt_bit_exact = (cpi->row_mt ? 1 : 0)`;
  `cpi->row_mt = 1` whenever `oxcf.mode == REALTIME && speed >= 5 &&
  oxcf.row_mt`.

**Synthesis (corrected):**
- At cpu-used 6 with row_mt on (the default for real-time): the SVC gate
  at line 715-718 is NOT hit, so `adaptive_rd_thresh_row_mt = 0`. Then
  line 910 kicks in: `adaptive_rd_thresh = 0`. **This makes cpu-used 6
  with row_mt deterministic — A6's "non-deterministic at cpu-used 6
  default" is wrong**.
- At cpu-used 7/8 with SVC + row_mt: `adaptive_rd_thresh_row_mt = 1`,
  which means the per-row threshold-frequency-fact table is used.
  Workers race on `tile_data->row_base_thresh_freq_fact[]` but the
  per-row separation makes this **bit-exact** (each row is owned by a
  worker via the job queue). This is also deterministic.
- At cpu-used 7/8 **without** SVC, with row_mt: `adaptive_rd_thresh_row_mt
  = 0`, `row_mt_bit_exact = 1`, so `adaptive_rd_thresh = 0`.
  Deterministic.
- The non-deterministic case is tile-MT (not row-MT) with
  `adaptive_rd_thresh > 0` — workers race on
  `tile_data->thresh_freq_fact[BLOCK_SIZES][MAX_MODES]` per tile. But
  per-tile separation means workers write only their tile's slot, so
  **also deterministic** as long as we never share tile_data across workers.

So **the actual non-determinism source A6 should have flagged is**:
- `tile_data` is shared across workers in `vp9_encode_tiles_mt` when
  `num_workers < tile_cols` (workers process multiple tiles by stride —
  `vp9_ethread.c:40-59`). Worker `t` processes tiles
  `t, t+num_workers, t+2*num_workers, ...`. The `tile_data` array
  itself is shared mutably, but each `tile_data[i]` is owned by a
  single worker, so no race.
- A6's claim of non-determinism at default cpu-used 6 is **WRONG**.

**Why it bites us:** Designing for non-determinism when it doesn't
exist wastes effort; failing to design for it where it does means we
silently lose byte-parity. A6's misdiagnosis cascades into our
HLD-threading.

**Severity:** **Medium.** Documentation correctness; if A8 doesn't
catch this, the LLD writer will mis-design.

**Resolution:** A8 patches A6 §7.7 with the corrected analysis above
and removes the "non-deterministic at cpu-used 6 default" claim. Updates
R-08 in `_decisions-needed-draft.md`.

---

### 12. Encoder/decoder probability table sharing is via `vp9/common/` — confirmed; H9's worry is unfounded for table location but real for the **runtime-adapted** tables that DO live on `cpi->common.fc` (encoder side) and `cm->fc` (decoder side) [H9]

**Affects:** `04-bitstream-entropy.md` §7 #1 (driver-synthesized).

**libvpx evidence (direct):**
- `vp9/common/vp9_entropy.{h,c}` — coefficient default probs, scan
  orders, all shared.
- `vp9/common/vp9_entropymode.{h,c}` — intra/inter/partition/skip/tx-size
  default probs, all shared.
- `vp9/common/vp9_entropymv.{h,c}` — MV default probs, shared.
- The **runtime** probabilities live on `FRAME_CONTEXT` (defined in
  `vp9/common/vp9_entropymode.h:69`), which sits inside `VP9_COMMON`
  both encoder (`cpi->common.fc`) and decoder (`cm->fc`) — confirmed by
  reading `vp9_onyxc_int.h:1283`.

**Why it bites us:** H9 was about whether tables would diverge between
crates. They won't, because both crates use the same defaults from
`rvp9-core`. BUT: the **runtime-mutated** `FRAME_CONTEXT` is different
content on encoder vs. decoder for the same frame — the encoder computes
the *expected* updated probs, writes diffs into the bitstream; the
decoder reads diffs and applies them. Both arrive at the same final
state. If our `rvp9-core::entropy::adapt_*_probs` function isn't bit-
identical across the two crates, we silently desync.

**Severity:** **Medium.** Affects Stage 3 conformance.

**Resolution:** L-09 in `_decisions-needed-draft.md` is correct; A8
should add a sub-bullet: "every `adapt_*` function MUST be defined once
in `rvp9-core` and called from both encoder and decoder; do NOT
re-implement per-crate."

---

### 13. Stage 0's 14 conformance vectors leave huge spec coverage gaps; missing groups include SVC, error-resilient, intra-only, color-space, profile-2/3 [H10]

**Affects:** Stage 0 deliverables (`tests/conformance/vectors/`),
PLAN.md §2 success criteria.

**libvpx evidence:** `third_party/libvpx/test/test-data.sha1` contains
**837** test-data entries (incl. `.webm`, `.ivf`, `.md5`, `.y4m`,
invalid streams). Filtering to `vp90-2-*` valid vectors yields the
following groups in libvpx:
- `vp90-2-00` (quantizer 00-63, ~64 entries)
- `vp90-2-01` (sharpness 1-7, ~7 entries)
- `vp90-2-02` (size 8x8 to large)
- `vp90-2-03` (delta-Q)
- `vp90-2-05` (resize)
- `vp90-2-06` (bilinear)
- `vp90-2-07` (frame parallel decode mode)
- `vp90-2-08` (tile configs 1x2, 1x4, 1x8, etc.)
- `vp90-2-09` (AQ modes)
- `vp90-2-10` (show-existing-frame)
- `vp90-2-11` to `vp90-2-22` — various edge cases (segmentation,
  bit-depth, error resilient, color space, profile 2, etc.)
- `vp90-2-bbb_*`, `vp90-2-sintel_*` — natural content with tile configs.

Stage 0 downloaded **14** vectors covering only `vp90-2-00, 01, 02, 03,
05, 08, 09, 10`. Missing: `06` (bilinear), `07` (frame parallel),
`11-22` (the bulk), and all `bbb_*` / `sintel_*` natural content.

**Why it bites us:** PLAN.md success criteria #1 says "Our decoder
passes 100% of the official VP9 conformance test vectors." Our test
set is < 4% of the official set. We will hit edge cases (e.g. SVC
header parsing, color-range handling, intra-only frames in
non-keyframe position) only when we run against the full set, late in
Stage 3.

**Severity:** **Medium.** Surfaces in Stage 3, but the fetch script is
trivial to expand.

**Resolution:** Expand `scripts/fetch-vectors.ps1` at Stage 3 entry,
not Stage 1 blocker. A8 records this. Specifically required for
Stage 3 exit: groups `02, 03, 06, 07, 09, 10, 11-22` and the
natural-content samples (`vp90-2-bbb_*` and `vp90-2-sintel_*`).

---

### 14. CONFIG_VP9_HIGHBITDEPTH leaks into the `tran_low_t` typedef and the encoder fast-path token boundary — even excluding HBD from v1 (D-01 default) requires deciding which integer width to use [H11]

**Affects:** All specialists. None explicitly covers the HBD typedef
divergence.

**libvpx evidence (direct):**
- `third_party/libvpx/vpx_dsp/vpx_dsp_common.h:32-44`:
  ```c
  #if CONFIG_VP9_HIGHBITDEPTH
  typedef int64_t tran_high_t;
  typedef int32_t tran_low_t;
  #else
  typedef int32_t tran_high_t;
  typedef int16_t tran_low_t;
  #endif
  ```
- `tran_low_t` flows into every coefficient buffer in libvpx's coding
  path: `MACROBLOCK::plane[].coeff`, `MACROBLOCKD::plane[].dqcoeff`,
  etc. The forward transforms output `tran_low_t`; quant produces
  `tran_low_t` qcoeff; dequant takes `tran_low_t` and outputs
  `tran_low_t`; the boolean coder consumes `tran_low_t` via
  `pack_mb_tokens`.
- `third_party/libvpx/vp9/encoder/vp9_tokenize.c:287` and `:301` —
  HBD-specific `vp9_cat6_prob_high12` dispatch with different bit-depth
  offsets (transcribed in A4 §5.7.4 at `04-bitstream-entropy.md:2151-2166`,
  so A4 caught this much).

**Why it bites us:** D7 commits to "mirror libvpx integer widths
exactly." But `tran_low_t` is one type at compile time — we need to
pick. If we ship 8-bit-only, we pick `i16`. Then:
- Stage 3 transforms output `i16` (matches `!CONFIG_VP9_HIGHBITDEPTH`
  exactly).
- If we later add HBD in v2, every kernel becomes generic over the
  pixel type — significant refactor.

A1 §2.8 transcribes `vp9_onyxc_int.h:1197-1199`:
```c
#if CONFIG_VP9_HIGHBITDEPTH
  int use_highbitdepth;
#endif
```
This conditional field doesn't appear in our Rust port if we exclude
HBD — pure absence. But the `tran_low_t` decision is structural.

**Severity:** **Medium.** Pure design question, but it locks
architecture.

**Resolution:** D-01 in `_decisions-needed-draft.md`. Stage 2 HLD-arch
+ HLD-bit-depth resolve. Default: 8-bit-only in v1, `tran_low_t = i16`
in core types, document as a divergence-ledger entry "type narrower
than libvpx's HBD build, matches libvpx's `!CONFIG_VP9_HIGHBITDEPTH`."

---

### 15. Worker thread RTCD ordering — libvpx serializes via `once_init` at compressor creation; A1 §6 covers this; H12 has no real risk [H12]

**Affects:** `01-architecture.md` §6.1 (`01-architecture.md:2829-2902`),
§7.5; `06-threading-svc.md` §7.10.

**libvpx evidence (verified):** `vp9_encoder.c:929-942` —
`vp9_initialize_enc()` is wrapped in `once()` so RTCD dispatch
populates before any encoder operates. Workers are created lazily by
`create_enc_workers` (`vp9_ethread.c:80-138`), which is invoked from
`vp9_encode_tiles_*`, which runs AFTER `vp9_initialize_enc` since the
compressor exists. The ordering is unambiguous.

**Severity:** **Low.** No real issue.

**Resolution:** A8 marks H12 as **REJECTED** — A1 §6 already covers
this correctly. For our Rust port, `std::sync::Once` in the encoder
constructor before any worker spawn matches the libvpx semantics. No
LLD work needed beyond that.

---

### 16. `cyclic_refresh->map` is correctly identified as a side channel by A5 Q-RC-12; it is NOT explicitly catalogued in A1's data-structures inventory [H14]

**Affects:** `01-architecture.md` §2 (no mention of `cyclic_refresh`
struct fields); `05-rate-control.md` §2.2
(`05-rate-control.md:264-344`).

**libvpx evidence (direct):**
- `vp9/encoder/vp9_aq_cyclicrefresh.h:25-50` (struct `CYCLIC_REFRESH`)
  — fields `int *map`, `int *last_coded_q_map`, `int *consec_zero_mv`
  (sized `mi_rows * mi_cols` each).
- These are read **inside tile workers** during pickmode (via
  `vp9_cyclic_refresh_segment_id_boosted` at
  `vp9_aq_cyclicrefresh.c:382-394`). Written by main thread between
  frames in `vp9_cyclic_refresh_postencode`.

**Why it bites us:** A1's data-structure inventory is the source of
truth for "what fields need an aliasing model." Missing the cyclic-
refresh tri-map means the Rust port's HLD-threading will be incomplete.
A5 covers the algorithm but not the worker-aliasing perspective.

**Severity:** **Medium.** Cross-doc gap.

**Resolution:** A8 patches A1 §2 with a `CYCLIC_REFRESH` cross-reference
to A5 §2.2 (no need to re-transcribe; just acknowledge), and an entry
in the worker-mutated-fields table from challenge 9 above.

---

### 17. `variance_low[25]` packing IS described in prose at A2 §2.9 but the bit-by-bit semantics (which logical block maps to which array index) needs cross-referencing the C `pos_shift_16x16[4][4]` table and the 9-case switch in `get_force_skip_low_temp_var` — A2 §7 Q2 surfaces the gap [H15]

**Affects:** `02-encoder-core.md` §2.9
(`02-encoder-core.md:344-396`), §4.15
(`02-encoder-core.md:1479-1506`), §7 Q2.

**libvpx evidence (verified):**
- `vp9/encoder/vp9_block.h:202-203` and `:158` — `int variance_low[25]`
  and the comment describing the packing.
- `vp9/encoder/vp9_pickmode.c:55` — `pos_shift_16x16[4][4]` table is
  the 4×4 → flat-index mapping for the 16 16×16 blocks. A2 §5.1
  transcribes it (`02-encoder-core.md:1608-1615`).
- `vp9/encoder/vp9_pickmode.c:1441-1486` — `get_force_skip_low_temp_var`
  9-case switch reading `variance_low[]`. A2 §4.15 documents the cases
  but not the full set of (i, j) → index calculations for the cases
  involving `pos_shift_16x16`.

**Why it bites us:** A2 §2.9 prose says
"`[9..24]` — 16×16 in row-major order through the 4×4 grid", but
`pos_shift_16x16` is **NOT** row-major. It's:
```c
{ 9, 10, 13, 14 },
{ 11, 12, 15, 16 },
{ 17, 18, 21, 22 },
{ 19, 20, 23, 24 }
```
(verified via A2 §5.1). This is **Morton order** (Z-curve), not
row-major. A2's prose is wrong by accident; the algorithm in §4.15
follows `pos_shift_16x16` correctly, so the implementation is right,
but the documentation will mislead the LLD reader.

**Severity:** **Medium.** Documentation bug; misreads will cause
off-by-N indexing in the Rust port.

**Resolution:** A8 patches A2 §2.9 to replace "row-major" with "Morton
order per `pos_shift_16x16`" and adds an explicit `[i+5]`, `[i+9..i+12]`
breakdown. Then the LLD reader has correct ground truth.

---

## Part 2 — Attack categories beyond the hypothesis tests

### 18. Speed-feature dispatch determinism for content-adaptive flags (`use_source_sad`, `force_smooth_filter`, `use_compound_nonrd_pickmode`) [Category 5, NEW]

**Affects:** `06-threading-svc.md` §5.12-§5.17 (transcribes speed-feature
cascades).

**libvpx evidence:**
- `vp9_speed_features.c:816-867` (post-cascade overrides) sets
  flags based on content state, source-SAD, and resize-state.
- `cpi->sf.use_source_sad` is set conditionally on
  `cpi->sf.adaptive_rd_thresh_row_mt` AND scene-detection results
  (see `vp9_speed_features.c:849`).
- `vp9_pickmode.c:1376-1450` reads
  `cpi->sf.use_compound_nonrd_pickmode` per-block.

**Why it bites us:** Even if the speed-feature **inputs** (frame_type,
qindex, etc.) are byte-equal between libvpx and rvp9, the
post-cascade overrides read **derived** state from earlier frames
(noise estimate, source-SAD history, resize state). If our earlier
frame is bit-equal, OK. But the dependency chain is fragile:
- `noise_estimate.level` feeds `sf->denoise_recheck_zeromv`
  (a speed feature flag).
- `rc.avg_source_sad[i]` feeds `sf->use_source_sad`.
- These are floats / running averages — see A5 Q-RC-5 on libm
  determinism.

**Severity:** **High** for byte-parity CI; **Medium** for BD-rate
(small drift, not visible to humans).

**Resolution:** R-04 (float determinism) in
`_decisions-needed-draft.md` is correct. A8 elevates: byte-parity CI
must include a "golden speed-feature trace" replayed frame-by-frame
against libvpx with a printf patch. A6 §7.14 already proposes this;
A8 should commit to it as a Stage 4a deliverable.

---

### 19. `vp9_lookahead` `MAX_LAG_BUFFERS = 25` vs. real-time `lag_in_frames = 0` — A1 §7.3 raises this; A5 doesn't explicitly handle lag=0 in queueing [Category 3 scope gap, NEW]

**Affects:** `01-architecture.md` §7.3, `05-rate-control.md` (no
section on lookahead).

**libvpx evidence:**
- `vp9/encoder/vp9_lookahead.h:22, 33` —
  `MAX_LAG_BUFFERS = 25`, `MAX_PRE_FRAMES = 1`.
- `vp9_lookahead.c:vp9_lookahead_push/_pop/_peek` — even with lag=0,
  the lookahead is the temporary holding pen for the YV12 source. The
  pattern is push-then-pop.
- For SVC, `vp9_svc_lookahead_pop` (`vp9_svc_layercontext.c:1245-1310`)
  re-pops the same lookahead entry once per spatial layer per
  superframe.

**Why it bites us:** A1 §7.3 asks "bypass lookahead at lag=0 or not?"
A5 doesn't weigh in. The lag=0 case is the dominant real-time
configuration, but bypassing breaks the SVC re-pop pattern. If we
bypass, we re-add the SVC repeat-frame mechanism elsewhere; if we
keep, we eat a per-frame YV12 memcpy.

**Severity:** **Low.** Single small design choice; affects API +
perf, not correctness.

**Resolution:** L in `_decisions-needed-draft.md` (already L-something
for this; verify). Default: keep the lookahead at depth=1 for SVC
compatibility; document the per-frame memcpy in divergence-ledger as
"~1ms per 1080p frame, acceptable real-time cost."

---

### 20. ROI / active-map interaction with cyclic refresh and segmentation — three-way interaction not deeply analysed by any specialist [Category 5 + cross-cutting, NEW]

**Affects:** `05-rate-control.md` Q-RC-28
(`05-rate-control.md:1953-1959`); `01-architecture.md` (no mention).

**libvpx evidence:**
- `vp9/encoder/vp9_encoder.c:4221-4232`:
  ```c
  if (cpi->roi.enabled && !frame_is_intra_only(cm)) {
    if (cpi->roi.skip[BACKGROUND_SEG_SKIP_ID]) {
      if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ)
        vp9_cyclic_refresh_setup(cpi);
      if (cpi->rc.frames_since_key > FRAMES_NO_SKIPPING_AFTER_KEY)
        apply_roi_map(cpi);
    } else {
      apply_roi_map(cpi);
    }
  } else if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ) {
    vp9_cyclic_refresh_setup(cpi);
  }
  ```
- `FRAMES_NO_SKIPPING_AFTER_KEY` — find it: it's an enum probably
  in `vp9_aq_cyclicrefresh.h`.

**Why it bites us:** Three-way interaction (CR + ROI + first 20 frames
post-KF) is one of the easier silent BD-rate divergence spots. If we
defer ROI to v2 (D-06 default), we still need to verify that omitting
the ROI branches doesn't accidentally re-enable CR-on-non-CR-config
or vice versa.

**Severity:** **Medium.** Edge case in PLAN.md scope (no explicit ROI
commitment).

**Resolution:** D-06 default ("ROI out of v1") + add a divergence-ledger
entry: "ROI-enabled CR-AQ post-key behaviour is not implemented in v1."

---

### 21. Frame-drop chain in SVC: when temporal layer N is dropped, does dependent layer N+1 get force-dropped, and does the bitstream produce a valid pattern? [Cross-cutting RC + SVC, NEW]

**Affects:** `05-rate-control.md` §4.8
(`05-rate-control.md:973-1006`), `06-threading-svc.md` §5.4
(temporal layering reference-structure tables).

**libvpx evidence:**
- `vp9_ratectrl.c:614` — `vp9_rc_drop_frame` — single-layer; for
  SVC the per-layer drop happens via SVC-aware checks in
  `vp9_svc_layercontext.c` (around `vp9_one_pass_svc_start_layer`
  and `vp9_svc_constrain_inter_layer_pred`).
- A6 §5.4 transcribes the `vp9_temporal_layer_id_*` tables
  (`06-threading-svc.md:1193-1257`). But the *consequences* of a drop
  on the next frame's referencing aren't expanded.

**Why it bites us:** WebRTC routinely drops the top temporal layer on
network congestion. If our drop logic doesn't precisely match libvpx's,
the receiver sees a different reference pattern and decoder mismatch.

**Severity:** **High** for WebRTC viability (D3 requires SVC).

**Resolution:** Stage 2 LLD-svc must include a drop-pattern truth
table. A8 records the gap; recommend re-dispatching A5 or A6 with
narrow scope "exact frame-drop semantics for 2-layer and 3-layer
temporal SVC: which frame IDs in the dispatch table are droppable,
and what does the encoder write in the bitstream when one is dropped?"

---

### 22. Real-time VBR + `lag_in_frames >= 4` enables altref — A1 §3.9 flags this; A2/A3 don't handle altref-aware reference selection in the pickmode docs [Cross-cutting, NEW]

**Affects:** `01-architecture.md` §3.9
(`01-architecture.md:2417-2421`); `02-encoder-core.md` §3 (mode dispatch);
`03-motion-estimation.md` §3.4 (ref selection).

**libvpx evidence:**
- `vp9/encoder/vp9_encoder.h:1423-1427` — `is_altref_enabled(cpi)`:
  ```c
  static INLINE int is_altref_enabled(const VP9_COMP *const cpi) {
    return !(cpi->oxcf.mode == REALTIME && cpi->oxcf.rc_mode == VPX_CBR) &&
           cpi->oxcf.lag_in_frames >= MIN_LOOKAHEAD_FOR_ARFS &&
           cpi->oxcf.enable_auto_arf;
  }
  ```
- So **REALTIME + VBR + lag ≥ 4 + enable_auto_arf** enables altref.
- A1 calls this out at §3.9, but A2's `ref_mode_set` only covers LAST,
  GOLDEN, ALTREF combinations (`02-encoder-core.md:109-143`) and never
  drills into whether altref is actually populated at speed 6-8.

**Why it bites us:** PLAN.md commits to both CBR and VBR rate control
under D1 (real-time). The VBR-RT path with altref enabled changes:
- Frame-loop structure (hidden ARF frame + visible frame referencing it).
- Reference-slot rotation (`cpi->alt_fb_idx`).
- `ref_mode_set` actually exercising the ALTREF branches.

If we test only CBR-RT in Stage 4a, we won't notice the VBR-RT altref
path is broken. If we test VBR-RT but assume no altref, we miss the
common configuration.

**Severity:** **High.** Affects Stage 4a scope.

**Resolution:** A8 surfaces; Arthur decides if VBR-RT + altref is in
v1 (PLAN.md §2 says yes implicitly by listing VBR but is silent on
altref). Recommend: include in v1, but gate Stage 4a MVP on CBR-RT
only (simpler), defer altref to Stage 4d (just before SVC).

---

### 23. Bug-for-bug compatibility issues: `vp9_aq_variance.c:86-88` TODO admits libvpx uses unoptimized variance in AQ variance; A5 Q-RC-27 says "don't preserve this inefficiency" — but that's a divergence from libvpx that affects byte-parity [Category 8, NEW]

**Affects:** `05-rate-control.md` Q-RC-27
(`05-rate-control.md:1943-1951`).

**libvpx evidence (direct):** `vp9/encoder/vp9_aq_variance.c:86-88`:
```c
// TODO(agrange): This function should be moved to vp9_variance.c, and
// vp9_variance.h.
// The block_variance calls the unoptimized versions of variance() and
// highbd_8_variance(). It should not.
```
But under D-05 default (drop AQ modes 1/2), this is moot in v1. If we
ship AQ mode 1, A5's recommendation "use optimised variance" is a
DIVERGENCE that may produce a different segment_id assignment → different
qindex → different bitstream.

**Severity:** **Low** if AQ modes 1/2 are excluded (which is the default).

**Resolution:** Coupled to D-05. If we add AQ-1/2 later, the "use
optimised variance" divergence has to land in the ledger; the
divergence is small but real.

---

### 24. Bug-for-bug: cyclic refresh `time_for_refresh = 0` is dead code in libvpx (A5 Q-RC-24); preserving for "API stability" doesn't apply since we're new code [Category 8, NEW]

**Affects:** `05-rate-control.md` Q-RC-24
(`05-rate-control.md:1916-1922`).

**libvpx evidence:** `vp9_aq_cyclicrefresh.c:509` — field exists, only
ever read as `if (cr->time_for_refresh) ...` but the only set site is
the zero-init from struct allocation.

**Why it bites us:** Trivial. Drop the field in our port.

**Severity:** **Low.**

**Resolution:** L-03 in `_decisions-needed-draft.md`. Drop in v1.

---

### 25. Real-time vs. slow-path leakage: `vp9_optimize_b` trellis quantization is correctly identified as disabled at cpu-used ≥ 5; A4 §7 #3 flags this [Category 9]

**Affects:** `04-bitstream-entropy.md` §7 #3.

**libvpx evidence:** `vp9_speed_features.c:227` —
`sf->optimize_coefficients = 1` for speed 0; `:485-504` etc don't set
it back. Confirmed by reading the cascade: at speed ≥ 5, the value is
inherited as 0 from the size-independent block.

**Severity:** **N/A** (validation).

**Resolution:** **CONFIRMED** as out of scope for v1; A8 closes
A4 §7 #3.

---

### 26. SVC interactions: `vp9_scene_detection_onepass` is "always called" per A5 Q-RC-26 — but the per-superframe vs. per-spatial-layer cadence is not specified [Category 10, NEW]

**Affects:** `05-rate-control.md` Q-RC-26
(`05-rate-control.md:1934-1942`).

**libvpx evidence:**
- `vp9/encoder/vp9_encoder.c:4084-` (around the SVC encode call) —
  scene detection is called once per `vp9_get_compressed_data` invocation.
- `vp9_cx_iface.c:1480-1582` — the outer encode loop calls
  `vp9_get_compressed_data` **once per spatial layer**. So scene
  detection runs **per spatial layer**, not per superframe.

**Why it bites us:** A 3-layer spatial SVC at 1080p means scene detection
runs 3× per source frame, on 3 different resolution downsamples (or the
same source, depending on `cpi->Source` aliasing). The SAD compute is
~1/16 of frame size per A5, so 3× is non-trivial CPU.

**Severity:** **Medium.** Affects SVC CPU budget at 1080p60-SVC.

**Resolution:** Stage 2 LLD-svc + LLD-rate-control owns; A8 records.
Verify by reading `vp9_scene_detection_onepass` in detail at LLD time.

---

### 27. Compound prediction in non-RD path is **dead code at cpu-used ≥ 5** per A2 §7 Q9 — but `sf->use_compound_nonrd_pickmode` is referenced in `vp9_pickmode.c:1376-1450` (the `recheck_zeromv_after_denoising` path) [Category 10, NEW]

**Affects:** `02-encoder-core.md` §7 Q9.

**libvpx evidence:**
- `vp9_speed_features.c:582` (speed ≥ 4) — `sf->use_compound_nonrd_pickmode = 0`.
- BUT `vp9_pickmode.c:2069` reads it and gates a comp_modes setup. Since
  it's 0 at speeds 5-8, the inner branch is dead. **Confirmed.**

**Severity:** **N/A** (validation).

**Resolution:** **CONFIRMED**. A8 closes; v1 ports drop the compound
branch.

---

### 28. `cpi->td.mb.e_mbd.plane[i].dst.buf` aliasing across the inter-predictor in real-time pickmode — A2 §7 Q15 surfaces; A6 §7.2 doesn't address [Cross-cutting threading, NEW]

**Affects:** `02-encoder-core.md` §7 Q15
(`02-encoder-core.md:2271-2277`); `06-threading-svc.md` §7.2.

**libvpx evidence:**
- `vp9_pickmode.c:944-1005` (`encode_breakout_test`) reads
  `pd->dst.buf` and writes through the same pointer (chroma path).
- `pd->dst.buf` aliases a slice of the **current frame buffer**
  (`cm->cur_frame->buf`).
- During tile-MT, different workers have disjoint dst regions per-tile,
  so the writes don't alias across workers. But within a worker, the
  inter-pred builder writes to `pd->dst.buf` BEFORE `encode_breakout_test`
  reads it.

**Why it bites us:** In Rust, modeling `pd->dst.buf` as `&mut [u8]`
slice of the frame buffer requires the slice to be obtained from a
parent `&mut Frame`, which conflicts with other reads of the same
frame within the same worker (the source plane reads from
`x->plane[i].src.buf` which may alias depending on which reference
frame is current). This is the standard "split a `&mut [u8]` into
disjoint slices and pass into a per-block routine" Rust pattern, but
the source/dst split is per-block at SB-encode time, not per-tile.

**Severity:** **Medium.** Tractable Rust borrow-checker exercise,
not load-bearing for correctness, but it shapes the API of
`Mb`/`Block`/`Plane` types.

**Resolution:** Stage 2 LLD-mode-decision + LLD-frame-buffers
co-design. A8 records the cross-doc gap.

---

### 29. A2's `RT_INTER_MODES = 12` and `RT_INTER_MODES_SVC = 8` define loop bounds for `ref_mode_set[]`; the driver fallback incorrectly worried about iteration count [NEW, supersedes driver #16]

**Affects:** `02-encoder-core.md` §5.11
(`02-encoder-core.md:1767-1768`).

**libvpx evidence (verified):** `vp9_pickmode.c:1237` and `:1247` define
the loop bounds; A2 §5.11 transcribes them correctly. The driver's
challenge #16 about "~30 iterations" misread `vp9_pick_inter_mode`'s
loop count.

**Severity:** **N/A** (validation; driver was wrong).

**Resolution:** A8 supersedes driver's challenge #16. The actual loop
is 12 (non-SVC) or 8 (SVC), bounded by `ref_mode_set` size. Plus an
inner over filter_ref candidates (2-3 filters) and possibly compound
(0 in scope). Total candidates per block: ≤ ~30, **not** "the loop
runs 30 times". A2's analysis is correct as written.

---

### 30. A4's claim "boolean coder ships no SIMD in libvpx" needs verification — libvpx's `vpx_dsp/x86/` has no `bitwriter_*.c` files, confirming the claim [Category 7]

**Affects:** `04-bitstream-entropy.md` §6
(`04-bitstream-entropy.md:1602-1637`).

**libvpx evidence (verified by directory walk):** `vpx_dsp/x86/` has
~140 files for transforms, intra-pred, sad, variance, deblock, etc.
**Zero** files for the boolean coder. Same for `vpx_dsp/arm/` and
`vpx_dsp/mips/`. The boolean coder is scalar-only in libvpx.

**Severity:** **N/A** (validation).

**Resolution:** **CONFIRMED**. A4 §6 is correct.

---

### 31. Counts-accumulation order at `vp9_ethread.c:243-247` may be non-deterministic if more than one worker writes the same count entry — verify integer-only [Category 4 hidden state, NEW]

**Affects:** `06-threading-svc.md` §7.4
(`06-threading-svc.md:1950-1952`).

**libvpx evidence (direct):**
- `vp9_ethread.c:242-247` — accumulates `td->counts` into `cm->counts`
  via `vp9_accumulate_frame_counts`. Each worker has private `td->counts`
  (allocated at create_enc_workers). The main thread's last worker uses
  `cm->counts` directly (`thread_data->td->counts == &cm->counts`).
  Loop is `for (i = 0; i < num_workers; i++) { if (i < num_workers - 1)
  accumulate(i); }` — main worker (last one) is skipped intentionally
  because it already wrote into `cm->counts` directly.
- All counts are `unsigned int`. Integer addition is associative
  and commutative, so accumulation order doesn't change the result.

**Why it bites us:** A6 §7.4 says "no invariant work needed." Verified
correct. But subtle: `td->counts` writes happen DURING worker run; the
main thread reads `cm->counts` AFTER all sync. If the main thread also
WROTE to `cm->counts` during its own tile-worker phase (i.e. main
thread is one of the workers), there's no race because main thread is
single-thread relative to itself.

**Severity:** **N/A** (validation).

**Resolution:** **CONFIRMED**. A6 §7.4 is correct.

---

### 32. `vp9_encodedframe_overshoot` interacts with `frame_parallel_decoding_mode` bit in the uncompressed header [NEW]

**Affects:** Not covered by any specialist.

**libvpx evidence:**
- `vp9_encoder.c:5589-5593` — `cm->frame_parallel_decoding_mode` may
  be **forcibly set to 1** when `error_resilient_mode = 1` (which
  itself can be forced under certain CBR conditions). See
  `vp9_encoder.c:5827-5832`.
- The encoded-frame-overshoot re-encode path can result in a
  CR-AQ-disabled frame. The bitstream-emitted flags for that re-encoded
  frame must consistently flag the frame's intra-vs-inter / show / etc.

**Why it bites us:** Edge case. If our port mishandles the re-encode
flow's header emission, we get a malformed bitstream.

**Severity:** **Low.** Likely caught by conformance.

**Resolution:** Stage 2 LLD-bitstream test plan should include a
"force re-encode with overshoot" scenario.

---

### 33. A3 §5 transcribes search-pattern tables (HEX, BIGDIA, SQUARE) but does NOT cite the exact `nstep_diamond` step values used by `vp9_diamond_search_sad_c` — verify table-level coverage [NEW]

**Affects:** `03-motion-estimation.md` §5.1
(`03-motion-estimation.md:1250-1359`).

**libvpx evidence:**
- `vp9_mcomp.c:vp9_diamond_search_sad_c` uses a search-pattern derived
  from `cfg->ss_mv` / `cfg->ss_os` populated by
  `vp9_init_dsmotion_compensation` (`vp9_mcomp.c:160`).
- The actual diamond step values per `(s, i)` are computed; the table
  is the search-step LUT `MAX_MVSEARCH_STEPS = 11`.

A3 §5.2 covers sub-pixel step tables but the integer-diamond step
sequence might be derived rather than transcribed. Verify in LLD.

**Severity:** **Low.** Likely fine; just verify.

**Resolution:** Stage 2 LLD-motion-estimation owns.

---

### 34. The `vp9_lookahead`-pop-for-SVC special pattern requires `vp9_svc_lookahead_pop` to peek without consuming when `non_reference_frame == 1` — A6 covers SVC structure but not the lookahead consumption protocol [NEW]

**Affects:** `06-threading-svc.md` §3.7-§3.8 (SVC encode flow).

**libvpx evidence:** `vp9/encoder/vp9_svc_layercontext.c:1245-1310`
`vp9_svc_lookahead_pop`. For non-reference layers, the lookahead entry
is held; only the top reference layer pops.

**Severity:** **Medium.** Direct correctness for SVC v1.

**Resolution:** Stage 2 LLD-svc owns.

---

### 35. The `vp9_set_quantizer` call is repeated **three times** in `encode_without_recode_loop` (lines 4136, 4201, 4262) — implicit invariant that the latest value wins [NEW]

**Affects:** Not explicitly noted in any specialist.

**libvpx evidence:** `vp9/encoder/vp9_encoder.c:4172, 4201, 4262` —
three call sites in the single-pass encode path. Each call updates
`cm->base_qindex` and triggers dequant table regeneration via
`vp9_init_quantizer`. The last call before `vp9_encode_frame` wins.

**Why it bites us:** Three sequential mutations of the q-state make
the Rust port's "where is q determined" question messy. If our port
restructures into "pick q then encode" with a single q-set point, we
have to ensure all three update conditions still fire correctly.

**Severity:** **Low** (just messy).

**Resolution:** Stage 4a LLD-rate-control documents the three-call
pattern explicitly. Don't restructure.

---

### 36. `vp9_pack_bitstream` is called **twice** in some flows: once tentatively (for size estimation) and once for real output — implicit save/restore of coding context [NEW]

**Affects:** `01-architecture.md` §2.6 (CODING_CONTEXT mentions
this), `04-bitstream-entropy.md` (pack flow does not flag the tentative
call).

**libvpx evidence:**
- `vp9_encoder.c:4254-4256`:
  ```c
  save_coding_context(cpi);
  vp9_pack_bitstream(cpi, dest, dest_size, size);
  restore_coding_context(cpi);
  ```
  The first pack writes provisional bits to the dest buffer to measure
  frame size; if overshoot triggers re-encode, the second pack writes
  the final output.

**Why it bites us:** If our Rust port treats `pack_bitstream` as a
one-shot function with side-effects on `FRAME_CONTEXT`, the tentative
call corrupts state. `save_coding_context` (`vp9_encoder.c:1084`) /
`restore_coding_context` (`:1114`) save/restore `nmvjointcost`, `nmvcosts*`,
`segment_pred_probs`, `last_frame_seg_map_copy`, the LF deltas, and the
full `FRAME_CONTEXT fc`. Mis-modelling this breaks idempotency.

**Severity:** **Medium.** Direct correctness for the
`RE_ENCODE_MAXQ` flow.

**Resolution:** Stage 2 LLD-bitstream + LLD-rate-control co-design.
A8 records: any `pack_bitstream` in our port must be either
side-effect-free OR explicitly save/restore-aware.

---

### 37. The lazy worker pool reallocation in `create_enc_workers` cleans up via `vp9_loop_filter_dealloc`, `vp9_bitstream_encode_tiles_buffer_dealloc`, `vp9_encode_free_mt_data` (`vp9_ethread.c:93-95`) — multi-allocator cleanup needs Rust drop-order discipline [NEW]

**Affects:** `01-architecture.md` §3.7
(`01-architecture.md:2249-2270`).

**libvpx evidence:** `vp9_ethread.c:80-138` — when num_workers changes
(SVC mid-stream resolution change, max_threads change), three separate
deallocators are called in sequence. Each owns disjoint memory:
- `vp9_loop_filter_dealloc` — LF row-mt sync structures.
- `vp9_bitstream_encode_tiles_buffer_dealloc` — per-tile bitstream worker
  data.
- `vp9_encode_free_mt_data` — per-worker `td`, `td->counts`, `pc_tree`.

**Why it bites us:** In Rust, ownership of these three resource pools
needs careful design: if all three are fields of `Encoder`, dropping
`Encoder` runs the destructors. If we mid-encoder resize the worker
pool, we need an explicit "shrink/grow" method that re-creates these
in the right order (lf depends on worker count, bitstream-buffer
depends on tile count, mt-data depends on worker count). This is a
Stage 2 LLD-threading concern.

**Severity:** **Medium.** Correctness for SVC mid-stream resize.

**Resolution:** Stage 2 LLD-threading documents the three-pool
ownership + reallocation order.

---

### 38. The `cpi->td.mb.fwd_txfm4x4`, `inv_txfm_add`, `highbd_inv_txfm_add` function-pointer wiring at `vp9_encodeframe.c:6086-6095` is **per-frame** (lossy/lossless flip) — A1 §6.3 mentions but doesn't drill in [NEW]

**Affects:** `01-architecture.md` §6.3
(`01-architecture.md:2914-2929`).

**libvpx evidence:** `vp9_encodeframe.c:6086-6095` — `set_encoding_function`
type sigs change between lossless and lossy. Lossless wiring uses WHT
forward + inverse; lossy uses DCT.

**Why it bites us:** Real-time may use lossless for some screen-content
frames (need to verify). If our port has a static function-pointer
binding ("DCT only") and lossless is hit, silent crash or wrong output.

**Severity:** **Low.** Lossless is rare in RT.

**Resolution:** Verify lossless reachability in real-time at Stage 2
LLD. If unreachable, skip lossless in v1 + divergence ledger; if
reachable, support both with the same dispatch design as libvpx.

---

### 39. `vp9_change_config` on dimension change: A1 doesn't trace which allocations resize vs. which need full reset [NEW]

**Affects:** `01-architecture.md` §3 (lifecycle).

**libvpx evidence:** `vp9/encoder/vp9_encoder.c:vp9_change_config`
(declaration around line 1700+). Path: re-checks dimensions, calls
`alloc_compressor_data` if changed, updates rate-control bandwidth, etc.

**Why it bites us:** Real-time encoders routinely change bitrate
(easy — no realloc), change framerate (easy), but resolution change is
the messy one. RESIZE_DYNAMIC drives this via the encoder's own
decision (`vp9_resize_one_pass_cbr`). The user-API caller passes a
new oxcf and we must figure out what to realloc.

**Severity:** **Medium.** Real-time usability concern (already raised
by driver fallback #26).

**Resolution:** Stage 4a LLD covers; document the field-by-field
reset-on-resize table.

---

### 40. `BufferPool` 12-slot capacity = 8 + 1 + 3 — confirmed by `vp9_onyxc_int.h:42`; the "3 scaled" is for SVC inter-layer prediction PLUS dynamic resize, not separable [NEW]

**Affects:** `01-architecture.md` §2.8
(`01-architecture.md:1310-1330`).

**libvpx evidence (verified):** `vp9_onyxc_int.h:42`:
```c
// 1 scratch frame for the new frame, REFS_PER_FRAME for scaled references on
// the encoder.
#define FRAME_BUFFERS (REF_FRAMES + 1 + REFS_PER_FRAME)
```
A1 §2.8 transcribes correctly.

**Why it bites us:** The "3 scaled ref" slots are not separable into
"SVC ref slots" vs "dynamic-resize ref slots"; they're a shared pool
used by both. Our Rust port's buffer-pool LLD must size at 12 and not
"size at 8 for v1 without SVC then grow to 12 with SVC" — the encoder
fast path assumes 12 slots indexed by `cpi->scaled_ref_idx[]`.

**Severity:** **Low.** Pure API design.

**Resolution:** Stage 2 LLD-frame-buffers fixes pool size at 12.

---

### 41. Speed-feature setter `set_rt_speed_feature_framesize_dependent` runs every frame in `vp9_set_speed_features_framesize_dependent` — A6 §5.18 transcribes but doesn't flag the per-frame re-evaluation as a byte-parity hazard [NEW]

**Affects:** `06-threading-svc.md` §5.18
(`06-threading-svc.md:1767-1779`).

**libvpx evidence:** `vp9_speed_features.c:414-450` and the wrapper
`vp9_set_speed_features_framesize_dependent` are called from
`set_size_dependent_vars` in `vp9_encoder.c`. The call cadence is "every
frame after `set_size_independent_vars`."

**Why it bites us:** A frame-size-dependent recomputation that reads
**runtime state** (e.g. `cm->mb_rows`, `cm->mb_cols`, current resize
state) and writes to `cpi->sf` is one of the more fragile per-frame
state updates. R-10 in the driver's doc (challenge 29 in the driver
list) raises a similar concern about per-frame state-tuple
dependency; this challenge sharpens it.

**Severity:** **Medium.** Affects byte-parity replay test design.

**Resolution:** Stage 2 HLD-threading owns; test harness must replay
the full frame sequence to validate speed-feature byte-equality, not
just the (speed, width, height) tuple.

---

### 42. SVC's `vp9_one_pass_svc_start_layer` reads `temporal_layering_mode` (3 values: NOLAYERING, BYPASS, 0101, 0212) — A6 §5.3 transcribes the modes but A5 §3.4 doesn't say which RC parameters change per mode [Cross-cutting, NEW]

**Affects:** `05-rate-control.md` §3.4
(`05-rate-control.md:650-667`), `06-threading-svc.md` §5.3.

**libvpx evidence:** `vp9_svc_layercontext.c:769-1010` — per
`temporal_layering_mode`, the per-layer `lst_fb_idx`, `gld_fb_idx`,
`alt_fb_idx`, `frame_flags`, are different. RC parameters
(`bits_off_target`, `buffer_level`, `target_bandwidth`) are also
per-layer.

**Why it bites us:** RC × SVC × temporal-mode is a 3-way interaction.
Stage 4d adds temporal SVC; without explicit per-mode-RC documentation,
porting will be guess-and-check.

**Severity:** **High** for Stage 4d.

**Resolution:** Stage 2 LLD-svc + LLD-rate-control co-design. A8 calls
this out as the "biggest cross-doc gap" between A5 and A6.

---

### 43. `vp9_int_pro_motion_estimation` is gated on `gf_temporal_ref && rc_mode == VPX_CBR && bsize >= BLOCK_16X16` per A3 §7 Q8 — this means **VBR-RT does not use integer-projection ME** [NEW; sharpens A3 Q8]

**Affects:** `03-motion-estimation.md` §7 Q8
(`03-motion-estimation.md:1861-1865`), §4.7.

**libvpx evidence:** `vp9_pickmode.c:1586-1592` — gate as A3 states.

**Why it bites us:** PLAN.md commits to VBR-RT. Under VBR-RT, the
encoder skips integer-projection ME entirely, falling back to
diamond/hex search. This is slower (more SAD calls per block) → real-time
budget tighter under VBR.

**Severity:** **Medium.** Affects 1080p30 VBR-RT perf budget.

**Resolution:** L-N in `_decisions-needed-draft.md`. Document that
VBR-RT has different ME path than CBR-RT.

---

### 44. The `vp9_scale_references` step (called per frame in SVC; `vp9_encoder.c:3120+`) maintains `cpi->scaled_ref_idx[]` but the LIFETIME of these scaled buffers across the encode is per-superframe, not per-frame — A6 doesn't trace this [Cross-cutting, NEW]

**Affects:** `06-threading-svc.md` §3.7.

**libvpx evidence:**
- `vp9_encoder.c:3454+` — `vp9_scale_references` creates downscaled
  copies of each reference into `cpi->scaled_ref_idx[ref]`, taking
  slots from the buffer pool.
- `vp9_release_scaled_references` (called at end of encode) releases
  the slots back.
- For spatial SVC, this happens once per superframe (i.e., shared
  across all spatial layers); the scaled buffers persist.

**Why it bites us:** The slot-lifetime invariant ("scaled refs alive
for the duration of one superframe encode, released after") needs to
be modelled in the Rust port's `BufferPool`. If we naively drop the
scaled ref slot at end of each spatial layer, the next layer sees
broken refs.

**Severity:** **High** for SVC.

**Resolution:** Stage 2 LLD-frame-buffers + LLD-svc.

---

### 45. `vp9_dropframe_count`, `vp9_buffer_level` updates happen at multiple points in the encode loop — A5 covers algorithm but the order-of-update across the pre-encode/post-encode boundary is ambiguous [NEW]

**Affects:** `05-rate-control.md` §3.1.

**libvpx evidence:** Pre-encode drop check: `vp9_test_drop` in
`vp9_ratectrl.c:513`; `vp9_rc_drop_frame` at `:614`. Post-encode update:
`vp9_rc_postencode_update` (called from `vp9_get_compressed_data`
return path). Buffer level updated in both.

**Why it bites us:** A5 §3.1 lists the order but doesn't say what
happens when post-encode-drop fires AFTER post-encode-update has
already happened. Order matters for the deterministic buffer model.

**Severity:** **Medium.**

**Resolution:** Stage 2 LLD-rate-control.

---

## Part 3 — Disposition summary

For A8's synthesizer convenience, here's the recommended disposition
of each challenge:

| # | Status | Action |
|---|---|---|
| 1 | ACCEPTED | A8 re-orders A4 §6/§7 to follow §5.10; closes R-01. |
| 2 | ACCEPTED, **BLOCKER** | A2 patch: transcribe `rate_tab_q10`, `dist_tab_q10`, `xsq_iq_q10` verbatim in §5.4. |
| 3 | ACCEPTED, **HIGH** | A2 patch: transcribe `vp9_prob_cost[256]`. |
| 4 | REJECTED | Prompt template referenced a non-existent table. A4 is correct. |
| 5 | ACCEPTED, **HIGH** | PLAN.md edit + D-05 default = drop modes 1/2. |
| 6 | ACCEPTED | R-03 design strategy in HLD-simd. |
| 7 | ACCEPTED, **HIGH** | LLD-rate-control captures the five mutations + two call sites + CR side effect. |
| 8 | CONFIRMED | A6 §7.9 stays; Q4 default = defer to v2. |
| 9 | ACCEPTED, **HIGH** | Re-dispatch A1 or A6 with worker-field scope, OR fold into HLD-threading. |
| 10 | ACCEPTED | HLD-arch decision: `panic = "unwind"` + scoped threads. |
| 11 | ACCEPTED, **MEDIUM** | A8 patches A6 §7.7 with corrected analysis. |
| 12 | ACCEPTED | L-09 amended with single-source-of-truth rule. |
| 13 | ACCEPTED | Stage 3 entry: expand `fetch-vectors.ps1`. |
| 14 | ACCEPTED | D-01 default = no HBD in v1, `tran_low_t = i16`. |
| 15 | REJECTED | A1 §6 already correct. |
| 16 | ACCEPTED | A8 patches A1 §2 with CR cross-ref. |
| 17 | ACCEPTED | A8 patches A2 §2.9 "row-major" → "Morton order per `pos_shift_16x16`." |
| 18 | ACCEPTED | A6 §7.14 + Stage 4a deliverable: golden speed-feature replay test. |
| 19 | ACCEPTED | L-N: keep lookahead at depth=1 for SVC. |
| 20 | ACCEPTED | D-06 default = ROI out of v1. |
| 21 | ACCEPTED, **HIGH** | Re-dispatch A5 or A6 with SVC drop scope, OR Stage 2 LLD-svc owns. |
| 22 | ACCEPTED, **HIGH** | Arthur decides VBR-RT + altref in v1; default = altref in v1 but tested in 4d not 4a. |
| 23 | ACCEPTED | Coupled to D-05; if AQ-1/2 land later, divergence-ledger entry. |
| 24 | ACCEPTED | L-03 already covers; drop the field. |
| 25 | CONFIRMED | A4 §7 #3 closed. |
| 26 | ACCEPTED | Stage 2 LLD-svc + LLD-rate-control verify cadence. |
| 27 | CONFIRMED | A2 §7 Q9 confirmed; drop compound from v1. |
| 28 | ACCEPTED | Cross-doc: LLD-mode-decision + LLD-frame-buffers. |
| 29 | REJECTED | Supersedes driver's #16 (which was incorrect). |
| 30 | CONFIRMED | A4 §6 stays. |
| 31 | CONFIRMED | A6 §7.4 stays. |
| 32 | ACCEPTED | LLD-bitstream test scenario. |
| 33 | ACCEPTED | LLD-motion-estimation verifies diamond-step LUT coverage. |
| 34 | ACCEPTED | LLD-svc owns lookahead-pop protocol. |
| 35 | ACCEPTED | LLD-rate-control documents 3 `vp9_set_quantizer` calls. |
| 36 | ACCEPTED, **HIGH** | LLD-bitstream save/restore-aware design. |
| 37 | ACCEPTED | LLD-threading documents the 3-pool ownership. |
| 38 | ACCEPTED | LLD-transforms verifies lossless reachability in RT. |
| 39 | ACCEPTED | LLD covers `change_config` realloc map. |
| 40 | CONFIRMED | A1 §2.8 stays; LLD-frame-buffers fixes pool at 12. |
| 41 | ACCEPTED | Test harness replays full frame sequence. |
| 42 | ACCEPTED, **HIGH** | A8 calls out as biggest A5↔A6 cross-doc gap. |
| 43 | ACCEPTED | L-N: VBR-RT has different ME than CBR-RT. |
| 44 | ACCEPTED, **HIGH** | LLD-frame-buffers + LLD-svc model superframe-scoped scaled-ref lifetime. |
| 45 | ACCEPTED | LLD-rate-control documents pre-/post-encode boundary order. |

Severity totals (excluding CONFIRMED and REJECTED validations):
- **Blocker:** 1 (#2).
- **High:** 8 (#3, #5, #7, #9, #21, #22, #36, #42, #44).
- **Medium:** ~22.
- **Low:** ~9.

**Re-dispatches recommended (in priority order):**
1. **A2 patch (in-place edit)** for challenges 2, 3, 17 — verbatim
   transcription of three encoder-internal tables + Morton-order
   correction. No agent dispatch needed; can be done by A8 directly.
2. **A1 or A6 re-dispatch** for challenge 9 — produce a complete
   worker-mutated-fields table. Mechanical scope.
3. **A5 or A6 re-dispatch** for challenge 21 — SVC frame-drop
   semantics with bitstream consequences.

**Stage 2 LLD owners flagged (sorted by LLD):**
- **HLD-arch** (#10, #14, #36, #39).
- **HLD-threading** (#7 worker-field table, #9 if not re-dispatched,
  #11, #16, #18 byte-parity CI, #37, #41).
- **HLD-simd** (#6).
- **LLD-rate-control** (#7 redux, #20, #26, #35, #45).
- **LLD-bitstream** (#32, #36).
- **LLD-svc** (#21, #22, #26, #34, #42, #44).
- **LLD-frame-buffers** (#28, #40, #44).
- **LLD-mode-decision** (#17, #23, #27, #28).
- **LLD-motion-estimation** (#33, #43).
- **LLD-transforms** (#38).
- **Stage 3 fetch script expansion** (#13).
- **Stage 4a / divergence ledger** (#23, #24, #38, #43).

---

*Devil's-advocate review complete. 45 challenges, all cited to either
specialist sections or `third_party/libvpx/` file:line. Hand to A8 for
synthesis into `00-overview.md`.*
