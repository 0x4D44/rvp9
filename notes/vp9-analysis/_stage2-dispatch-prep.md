# Stage 2 dispatch-prep (scratchpad)

In-flight working doc preparing for Stage 2 (HLD + per-module LLDs) so that
the moment Stage 1's `00-overview.md` + `decisions-needed.md` land, Stage 2
can dispatch promptly.

The leading underscore on the filename is the convention for "scratchpad,
do not promote to a Stage 1 deliverable."

This file is written DURING Stage 1, used DURING Stage 2 dispatch. Cleanup
notes at the bottom.

---

## Stage 2 scope (recap from PLAN.md §6)

5 HLDs in `notes/hld/`:
- `00-architecture.md` — top-level Rust architecture, crate boundaries,
  module graph, type-system overview, error model, `unsafe` policy
- `01-threading.md` — tile-based threading in Rust, `Send`/`Sync` strategy,
  `UnsafeCell` policy, synchronization primitives, Q4 (frame-parallel)
- `02-simd-dispatch.md` — RTCD-equivalent design (Q2), per-arch intrinsics
  organization, runtime feature detection, scalar-fallback policy
- `03-bit-depth.md` — Q1 decision: 10/12-bit support in v1? If yes, design
  pattern (generics over pixel type vs separate code paths)
- `04-svc.md` — SVC layer pipeline, reference dependency tracking, cross-
  layer rate control

~15 LLDs in `notes/lld/`:
Shared infrastructure (Stage 3 implementation targets):
- `bitstream.md`, `entropy.md`, `transforms.md`, `quant.md`,
  `intra-pred.md`, `loopfilter.md`, `frame-buffers.md`

Encoder core (Stage 4 implementation targets):
- `motion-estimation.md`, `mode-decision.md`, `partition.md`,
  `rate-control.md`, `denoiser.md`, `aq.md`, `svc.md`, `tile-threading.md`

Each LLD: types, public API, internal algorithm, libvpx file:line origin,
divergence notes.

---

## Dispatch strategy for Stage 2

Unlike Stage 1, Stage 2 is **writing-heavy, not reading-heavy**. The
specialists in Stage 1 consume libvpx source and produce analysis docs;
Stage 2 consumes those analysis docs and produces Rust design docs. The
agent prompts are different in shape:

- Stage 1 prompt: "Read libvpx/X.c and document it as analysis."
- Stage 2 prompt: "Read notes/vp9-analysis/X.md and produce a Rust LLD."

Stage 2 agents will have access to (and need to read) `notes/vp9-analysis/`
output from Stage 1. They do NOT need to read libvpx directly except to
verify a citation.

**Phased dispatch (3 phases):**

**Phase 2a:** 4 HLD writers in parallel
- HLD-arch  → `notes/hld/00-architecture.md`
- HLD-threading → `notes/hld/01-threading.md`
- HLD-simd → `notes/hld/02-simd-dispatch.md`
- HLD-svc → `notes/hld/04-svc.md`
- (`03-bit-depth.md` resolved inline as part of HLD-arch since it's a
  scoping decision, not a 1500-line doc)

Each HLD writer reads ALL six Stage 1 specialist docs plus 00-overview
and decisions-needed. They produce a Rust design that respects PLAN.md
D1-D9.

**Phase 2b:** Arthur-led review of HLDs, refinement, sign-off. HUMAN GATE.

**Phase 2c:** ~15 LLD writers in parallel, one per LLD. Each reads the
relevant Stage 1 specialist doc(s) + the four HLDs + decisions-needed.
Produces an LLD against the structure below.

**Phase 2d:** Arthur-led LLD review, refinement, sign-off. HUMAN GATE.

**Phase 2e:** Compile divergence ledger seed, refine PLAN.md Stage 3 task
list based on what the LLDs surfaced.

---

## LLD template (every LLD follows this)

```
# LLD: <module name>

## 1. Purpose
One paragraph: what this module owns and why it exists.

## 2. libvpx origin
File:line ranges in third_party/libvpx/ that this module ports. Reference
the Stage 1 specialist doc covering this area.

## 3. Public API
Rust signatures. No bodies. Use the types from §4. Document each function's
preconditions, postconditions, panics, and unsafety.

## 4. Types
Rust struct / enum / type definitions. Field-by-field comments. Marker
trait bounds (Send, Sync, Copy, ...). UnsafeCell / interior mutability
explicit.

## 5. Internal algorithm
Pseudocode-level. Cite libvpx file:line for each non-obvious step.

## 6. SIMD dispatch
Which kernels in this module have SIMD? Per-arch (sse2/sse4_1/avx2/neon)?
RTCD signature. Test strategy (scalar baseline + per-arch differential).

## 7. Deliberate divergences from libvpx
Each entry feeds notes/divergence-ledger.md. Format per ledger spec.

## 8. Open questions
Anything that surfaced during LLD writing that needs Arthur or another
LLD writer to resolve.

## 9. Test strategy
Unit tests, property tests, differential tests against libvpx, conformance
vectors (where applicable).
```

---

## HLD-arch prompt (draft — fire when Stage 1 lands)

```
You are the architecture HLD writer for Stage 2 of the rvp9-2 project.

Working directory: D:\language\rvp9-2. Read in order:
- PLAN.md (D1-D9 locked decisions; Stage 2 scope; success criteria)
- notes/vp9-analysis/00-overview.md (Stage 1 synthesis)
- notes/vp9-analysis/01-architecture.md (libvpx top-level)
- notes/vp9-analysis/decisions-needed.md (open questions)

YOUR TASK: write notes/hld/00-architecture.md — the top-level Rust
architecture for the rvp9-2 encoder + decoder. This is the load-bearing
HLD; later HLDs and LLDs anchor to it.

OUTPUT must cover:

1. **Crate boundaries** — review the existing six-crate split
   (rvp9-core / decoder / encoder / cli / bench / harness) against what
   Stage 1 revealed. Refine if needed.

2. **Module graph within rvp9-core** — the shared primitives subdivided.
   Per PLAN.md §6 Stage 3, expect: bitstream, entropy, transforms, quant,
   intra_pred, loopfilter, frame, simd. Refine based on Stage 1 findings.

3. **Type system overview** — the load-bearing types that cross module
   boundaries (frame buffers, motion vectors, reference frames,
   probability contexts). For each: where it's defined, who owns it,
   sync semantics.

4. **Error model** — `Result` policy, error type design, panic policy
   (which conditions panic, which return errors).

5. **`unsafe` policy** — D6 commits us to libvpx-mirrored threading
   meaning substantial `unsafe`. State the policy: where `unsafe` is
   permitted (SIMD, threading, FFI), where it's forbidden, what comments
   are required.

6. **Integer-width policy** — D7 commits us to mirroring libvpx exactly.
   Document the wrapping_* convention surfaced in PLAN.md scaffolding.

7. **High-bit-depth (Q1) decision** — based on Stage 1 findings, is HBD
   in v1 scope? If yes, design pattern (generics, separate types, runtime
   dispatch). Argue.

8. **Build-time configuration** — Cargo features for HBD, denoiser,
   SVC, etc. — what's a feature flag, what's always on.

9. **Documentation policy** — rustdoc requirements, examples policy.

QUALITY BAR:
- Cite Stage 1 specialist docs (with section refs) for every architectural
  claim.
- Be specific. "We'll have a frame buffer type" is bad; "Frame buffers
  are `Arc<FrameBuffer>` with a recycling pool managed by FramePool,
  defined in rvp9_core::frame::pool, ownership transfers to tile workers
  via raw-pointer cast under a documented aliasing contract" is good.
- No Rust code yet — this is design, not implementation.
- Do not write outside notes/hld/.

Output target: 600-1200 lines of Markdown. Take time on the unsafe
policy and HBD decision — those ripple through every later LLD.
```

---

## HLD-threading prompt (draft)

```
You are the threading HLD writer for Stage 2 of rvp9-2.

Working directory: D:\language\rvp9-2. Read in order:
- PLAN.md (D6 locked: "mirror libvpx tile-based pthread model")
- notes/vp9-analysis/00-overview.md
- notes/vp9-analysis/06-threading-svc.md (libvpx threading deep-dive)
- notes/vp9-analysis/01-architecture.md (where threading sits in lifecycle)
- notes/vp9-analysis/decisions-needed.md (Q3, Q4)
- notes/hld/00-architecture.md (must be Arthur-signed-off before you start)

YOUR TASK: write notes/hld/01-threading.md — the Rust threading design
that mirrors libvpx's tile-based pthread model while remaining sound
under Rust's aliasing rules.

OUTPUT must cover:

1. **Q3 resolution** — which threading primitives? std::thread + Mutex /
   Condvar / Barrier? crossbeam? rayon? Argue.

2. **Q4 resolution** — frame-parallel encoding in v1, or v2? Argue.

3. **Worker thread design** — pool sizing, thread lifecycle (creation /
   reuse / teardown), error propagation across threads.

4. **Tile assignment** — how tiles are partitioned across workers,
   matching libvpx's behavior. Tile-row barriers. Top/left boundary
   predictor sync.

5. **Shared state contract** — every field of the rvp9 equivalent of
   VP9_COMP and where it lives. For each:
   - Owner thread (main vs worker)
   - Sync mechanism (Mutex / atomic / read-only after init / UnsafeCell
     with documented aliasing contract)
   - Required safety comment template

6. **UnsafeCell strategy** — D6 implies substantial UnsafeCell. Define
   the pattern: which types use UnsafeCell, how they're accessed safely,
   what invariants must hold.

7. **Send + Sync impls** — manual impls list with safety justification.

8. **Test strategy** — Miri runs, ThreadSanitizer (where available), loom
   for the trickiest sync primitives, soak tests in CI.

QUALITY BAR: every claim cited to either 06-threading-svc.md or
third_party/libvpx/ file:line. Concrete patterns, not generalities.
No Rust code bodies (signatures + aliasing comments are fine).

Output target: 800-1500 lines. This is the hardest HLD — get it right.
```

---

## HLD-simd prompt (draft)

```
You are the SIMD HLD writer for Stage 2 of rvp9-2.

Working directory: D:\language\rvp9-2. Read in order:
- PLAN.md (D5 locked: core::arch intrinsics + scalar fallback)
- notes/vp9-analysis/00-overview.md
- All Stage 1 specialist docs §6 (RTCD / SIMD surfaces) — across A1-A6
- notes/vp9-analysis/decisions-needed.md (Q2)
- notes/hld/00-architecture.md

YOUR TASK: write notes/hld/02-simd-dispatch.md — the RTCD-equivalent
design for runtime SIMD dispatch.

OUTPUT must cover:

1. **Q2 resolution** — hand-written Rust dispatch tables vs build.rs
   generator from libvpx's vpx_dsp_rtcd_defs.pl. Argue with concrete
   pro/con on maintenance burden.

2. **Kernel catalogue** — every SIMD kernel libvpx ships, aggregated
   from A1-A6 §6 sections. For each:
   - Function signature (Rust)
   - libvpx file:line for scalar reference
   - ISAs we ship in v1 (scalar, sse2, sse4_1, avx2, neon)
   - Differential test plan

3. **Runtime feature detection** — `is_x86_feature_detected!` pattern,
   one-time-init dispatch table at encoder boot.

4. **Function-pointer dispatch table layout** — concrete struct
   definition (pseudocode is fine).

5. **Scalar fallback policy** — every kernel has a scalar reference
   used for: (a) cargo build without target features, (b) differential
   testing, (c) cross-platform correctness.

6. **Testing strategy** — for each kernel, a property test or
   differential test ensures every SIMD variant matches the scalar
   reference bit-exactly on random inputs and known edge cases.

7. **Performance regression CI** — bench gate per kernel.

QUALITY BAR: kernel catalogue must be COMPLETE (every kernel libvpx
ships in v9 of A1-A6 §6 must appear). No silent omissions.

Output target: 1500-2500 lines (the kernel catalogue is long).
```

---

## HLD-svc prompt (draft)

```
You are the SVC HLD writer for Stage 2 of rvp9-2.

Working directory: D:\language\rvp9-2. Read in order:
- PLAN.md (D3 locked: full SVC in v1)
- notes/vp9-analysis/00-overview.md
- notes/vp9-analysis/06-threading-svc.md
- notes/vp9-analysis/05-rate-control.md (per-layer RC)
- notes/vp9-analysis/04-bitstream-entropy.md (layer metadata encoding)
- notes/vp9-analysis/decisions-needed.md
- notes/hld/00-architecture.md
- notes/hld/01-threading.md

YOUR TASK: write notes/hld/04-svc.md — the SVC layer pipeline design.

OUTPUT must cover:

1. **Temporal SVC** — 2- and 3-layer hierarchies, prediction structure,
   ref-frame management per layer.

2. **Spatial SVC** — multi-resolution pipeline, inter-layer texture
   prediction, inter-layer motion prediction.

3. **Cross-layer rate control** — per-layer Q assignment, cross-layer
   bit allocation, denoiser interaction.

4. **Bitstream layout** — superframe index, layer headers, dropped-frame
   semantics.

5. **API surface** — how the encoder configures SVC, what the caller
   sees, frame-by-frame layer-selection hooks.

6. **Test strategy** — per-layer extraction, layer correctness against
   libvpx, BD-rate per layer.

QUALITY BAR: cite libvpx file:line and Stage 1 docs throughout.

Output target: 800-1500 lines.
```

---

## LLD dispatch (Phase 2c)

Once HLDs are Arthur-signed-off, dispatch ~15 LLD writers in parallel.
Each writer's prompt template:

```
You are LLD writer for `<module-name>` in Stage 2 of rvp9-2.

Working directory: D:\language\rvp9-2. Read in order:
- PLAN.md  (note the locked scope: D-01 RESOLVED -> 8-bit only;
  D-05 RESOLVED -> AQ mode 3 only; D-13 RESOLVED -> CBR + VBR-RT,
  NO altref; R-06 RESOLVED -> EXT_RATECTRL in v1)
- notes/vp9-analysis/<relevant specialist>.md (specific to module)
- notes/vp9-analysis/00-overview.md
- notes/vp9-analysis/decisions-needed.md
- ALL four HLDs in notes/hld/  (00-architecture, 01-threading,
  02-simd-dispatch, 04-svc)

YOUR TASK: write notes/lld/<module-name>.md following the LLD template
documented in notes/vp9-analysis/_stage2-dispatch-prep.md "LLD template".

Specific module focus: <focus paragraph from the section below>

SCOPE GUARDRAILS (apply throughout):
- 8-bit only (no `u16` pixel-type paths, no HBD-specific kernels).
  If a libvpx file you cite has #if CONFIG_VP9_HIGHBITDEPTH branches,
  document only the non-HBD path; cite line ranges for the HBD path
  but mark "v2".
- No altref (no lag-in-frames > 0; no alt-ref-frame buffer; no
  GF_GROUP altref slots).
- AQ: only the cyclic-refresh path is in scope unless your LLD is
  explicitly `aq.md`.

QUALITY BAR: file:line citations for every algorithmic claim. Concrete
Rust types and signatures (no "TBD" markers without an explicit Arthur
ask). List entries for the divergence ledger in §7.
```

LLD module → Stage 1 specialist owner map:

| LLD | Primary specialist | Secondary |
|---|---|---|
| `bitstream.md`        | A4 | A1 (for header→encoder integration)
| `entropy.md`          | A4 | —
| `transforms.md`       | A4 (for scan orders) | A2 (for coeff handling)
| `quant.md`            | A5 (for quant tables) | A4
| `intra-pred.md`       | A2 | —
| `loopfilter.md`       | A1 (overall flow) | —  *(libvpx-specific; Stage 1 may need a re-dispatch for deep loopfilter analysis if A1 didn't cover it)*
| `frame-buffers.md`    | A1 | A6 (sync)
| `motion-estimation.md`| A3 | —
| `mode-decision.md`    | A2 | A3, A4
| `partition.md`        | A2 | A3
| `rate-control.md`     | A5 | —
| `denoiser.md`         | A5 | A2 (mode integration)
| `aq.md`               | A5 | —
| `svc.md`              | A6 | A5
| `tile-threading.md`   | A6 | A1

If any LLD's primary specialist didn't cover sufficient depth, that's a
Stage 1 re-dispatch signal — the synthesizer (A8) should have caught it.

---

## Per-LLD focus paragraphs (drop into the "Specific module focus" slot)

### bitstream.md
The bitstream-layout LLD: uncompressed header, compressed header,
tile-data layout, superframe index. Primary source A4 §3.1-3.4 and
§5.1 (sync codes, profile codes, markers). API: `BitstreamWriter`
streaming-out interface; per-tile buffer ownership (L-13, L-35);
superframe index assembly (L-14); `show_existing_frame` wire format
(L-15). Bit-exactness is mandatory (D8); cite §4.9 (bit-exact contract).
EXT_RATECTRL doesn't touch this LLD directly but its Q-decisions
upstream affect what gets serialized; document the boundary.

### entropy.md
Boolean coder + frame-context probability machinery. Primary A4 §2.1
(vpx_writer), §2.2 (vpx_reader), §2.6 (FRAME_CONTEXT), §2.7
(FRAME_COUNTS), §4.1-4.2 (encode/decode pseudocode), §4.8 (merge_probs),
§4.9 (bit-exact contract). Includes the cost framework: §5.11.0
**vp9_prob_cost[256]** (just patched in by Stage 2 prep work — D-14
RESOLVED) and the three cost macros. §2.5 prob/tree convention.
§2.10 VP9BitstreamWorkerData per-tile state. Probability sharing
strategy across encoder/decoder per L-09 (locked: share via rvp9-core).
Bit-exactness mandatory.

### transforms.md
The DCT/ADST/iDCT/iADST forward and inverse transform pipeline at
4×4/8×8/16×16/32×32. Primary A4 §5.10 (scan/iscan/neighbor tables —
all four block sizes verbatim) plus A2 §3-§4 for coeff handling
context. Scan-order tables are load-bearing; transcribe references.
SIMD dispatch reaches into HLD-simd kernel catalogue. **8-bit only —
no `vpx_highbd_*` variants.** Trellis-quant (`vp9_optimize_b`) is
gated off at cpu-used 6-8 per L-28; document the gate but skip the
implementation (defer to a quality-preset LLD). 4×4 reachability
under real-time per L-71 must be confirmed (intra-only / KF paths).

### quant.md
Quantizer machinery: q_index tables (`vp9_quant.c`), dequant tables,
the qindex→qstep mapping, AC/DC quant pairs, segmentation-feature
delta application. Primary A5 (quant tables shipped with rate
control); secondary A4 for the bitstream-side qindex serialization.
Per-segment quant deltas tie into AQ-3 (cyclic refresh). **8-bit only:
no `vp9_*_quant_b_high*` paths.** The qindex resolution-change
preservation per L-25 (`rate_correction_factors[]` × `rcf_mult[]`) is
upstream RC concern; this LLD only needs the qindex→quant arithmetic.

### intra-pred.md
The ten intra-prediction modes (DC, V, H, D45, D135, D117, D153,
D207, D63, TM) at 4×4/8×8/16×16/32×32. Primary A2 (mode decision
calls intra-pred during pick-mode); the actual predictor kernels are
shared with the decoder and live in vp9/common/vp9_reconintra.c
(line range citations sufficient — the math is local). SIMD per
HLD-simd. **8-bit only.** Edge-pixel availability and replication
rules at frame boundaries (L-53 partition-at-edges is related).

### loopfilter.md
Deblocking loop filter: per-mode/per-ref filter-level selection, the
edge-strength × filter-strength × filter-mask machinery, vertical
then horizontal passes. **Stage 1 coverage on this is thin — A1 §3
mentions it in passing only.** If primary-specialist read is
insufficient, the LLD writer should escalate as a Stage 1.5
re-dispatch signal (per the owner-map note). Source files:
vp9/common/vp9_loopfilter.c (parameters), vp9/encoder/vp9_pickmode.c
+ vp9_encoder.c (per-frame filter-level RDO bypass at non-RD).
**8-bit only.** SIMD dispatch reaches into HLD-simd.

### frame-buffers.md
The BufferPool (A1 §7.1): 12-slot pool (8 ref + 1 new + 3 scaled).
Primary A1 (lifecycle), secondary A6 (cross-thread sync). The "3
scaled" slots are libvpx-internal for SVC inter-layer prediction;
**do not expose in public API** per A1 §7.11 / L-36. Reference-frame
slot map is shared across spatial + temporal SVC layers. Arc<FrameBuf>
vs Rc<FrameBuf> per L-37 (Arc — workers cross threads). Frame
padding/alignment matching libvpx (load-bearing for SIMD predictors).
8-bit only (so the buffer type is `Box<[u8]>` not generic over Pixel).

### motion-estimation.md
Full ME pipeline at cpu-used 6-8 (non-RD speed). Primary A3
(verbatim search patterns, sub-pel variants, cost-list invariants).
Diamond → full-pel → sub-pel halve/quarter pipeline. Reference
masking via `pred_mv_sad << 1` threshold per L-06. Sub-pel cost-list
4-entry layout per L-07. `accurate_sub_pel_search` stubbed off at
cpu-used 6-8 per L-05; document the gate. `vp9_refining_search_8p_c`
dead code (compound search) per L-08 — stub. **8-bit only — no
`comp_avg_pred_highbd` SIMD.** NEWMV neighbour-difference bias
(`vp9_NEWMV_diff_bias`) per L-43.

### mode-decision.md
The non-RD pick-mode path: 30-iteration ref×mode loop, mode-set
selection, `block_yrd` fast-path bypass at speed ≥ 7. Primary A2 §2-§4
(verbatim). The 30-iteration claim must be validated (L-40).
`ref_mode_set[12]` per L-55 (const array vs enum). `vp9_NEWMV_diff_bias`
× `encode_breakout_test` ordering per L-43. Filter-ref selection in
non-RD path per L-48. Compound modes deferred to v2 per L-46. Uses
the Laplacian R-D model (02-encoder-core.md §5.4, just patched —
R-07 RESOLVED). EXT_RATECTRL Q-decisions enter here as inputs.

### partition.md
The variance-tree partition machinery: `choose_partitioning`,
`set_vt_partitioning`, `vbp_thresholds[4]`, `variance_low[25]`
(Morton-order — async A7 #17), `scale_part_thresh_sumdiff` at speed 6,
`set_low_temp_var_flag` 32×32 threshold expressions per L-44. Primary
A2 §2-§4. Partition at frame edges (cropping) per L-53.
`set_vt_partitioning` block_height == block_width invariant per L-52.
`coord_lookup` (32×32 → 16×16 sub-positions in 64×64 SB) per A2 §5.14.

### rate-control.md
Single-pass CBR + VBR-RT rate control: buffer-model arithmetic, frame
drop, qindex-from-bits inverse, rate_correction_factors, the
"small image format" Q-adjustment (L-17), the buffer critical_level
= optimal/8 constant (L-18), `vp9_encodedframe_overshoot` huge state
mutation (L-19, L-74). **No altref** per D-13 (RESOLVED) — skip
GF_GROUP altref slot logic entirely. Primary A5. `rate_correction_factors[]`
resolution-change preservation per L-25 (off-by-one trap).
**EXT_RATECTRL plug-in interface** per R-06 (RESOLVED) — Rust-trait
callback surface for libwebrtc consumers; the per-HLD-svc design
states the trait shape, this LLD wires it into the RC loop.
Float-determinism caveat per L-04 / R-04 (libm log10 calls).

### denoiser.md
Content-aware denoiser: spatial + temporal denoising, mode-decision
integration. Primary A5 §denoiser, secondary A2 for the mode-integration
boundary. **SIMD bit-parity caveat** per R-03 / L-22:
`increase_denoising=1` is not bit-exact across scalar / SSE2 / NEON
even within libvpx itself. The LLD must pick a canonical impl OR
explicitly allow ULP differences in differential tests, AND document
the choice as a divergence-ledger entry. **8-bit only — no HBD
denoiser variants.** Cyclic-refresh segmentation map ownership per L-67
(also a threading HLD concern).

### aq.md
**CYCLIC_REFRESH (AQ mode 3) only** per D-05 (RESOLVED). Modes 1
(variance) and 2 (complexity) are dropped from v1 — do **not** port.
Primary A5. The cyclic-refresh state machine: refresh-period
scheduling, segment-Q assignment, segmentation map maintenance,
screen-content gate per L-24 (`vp9_cyclic_refresh_limit_q`),
`weight_segment` smoothing math per L-21 (sub-0.3% BD-rate trap),
`time_for_refresh` dead field per L-03 (drop). Integrates with
rate-control.md for Q-delta application.

### svc.md
SVC layer pipeline implementation: per-layer state contexts, layer
swap ordering (L-58), max-tile-cols sizing under SVC top-resolution
layer per L-61, `vp9_scene_detection_onepass` per-layer behaviour per
L-68, cross-layer frame-drop chain per L-69 / L-75, temporal_layering
× per-layer RC per L-77. Primary A6, secondary A5 for the cross-layer
RC math. **No altref** per D-13 (RESOLVED) — skip altref × SVC
interactions. EXT_RATECTRL gets per-layer hooks from HLD-svc; this
LLD wires them into the per-layer Q assignment.

### tile-threading.md
The tile-MT + row-MT worker pool, primary A6 §1-§4 plus §7.2-§7.4.
ThreadData::mb aliasing per D-11 (RESOLVED in HLD-threading; this LLD
implements the chosen pattern). FRAME_COUNTS per-worker allocation
+ accumulation per L-57. RTCD-init-before-worker-spawn per R-09.
Worker thread priority per L-63. Assert log2_tile_rows == 0 in v1
per L-62 (real-time never uses tile rows). `cpi->row_mt` vs
`cpi->oxcf.max_threads` modelling per L-59. Bitstream packing scalar
only per L-35.

---

## Cleanup at end of Stage 2

When Stage 2 completes (Arthur signs off on HLDs + LLDs):
1. Move this scratchpad to `notes/.archive/stage2-dispatch-prep.md` (or
   just delete it — git history preserves)
2. Delete `_dispatch-prep.md` (the Stage 1 scratchpad)
3. Update PLAN.md Stage 2 completion log
