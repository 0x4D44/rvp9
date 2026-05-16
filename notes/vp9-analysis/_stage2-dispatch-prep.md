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
- PLAN.md
- notes/vp9-analysis/<relevant specialist>.md (specific to module)
- notes/vp9-analysis/00-overview.md
- notes/vp9-analysis/decisions-needed.md
- ALL four HLDs in notes/hld/

YOUR TASK: write notes/lld/<module-name>.md following the LLD template
documented in notes/vp9-analysis/_stage2-dispatch-prep.md "LLD template".

Specific module focus: <one paragraph from the table below>

QUALITY BAR: file:line citations for every algorithmic claim. Concrete
types and signatures (no "TBD" markers without an explicit Arthur ask).
List entries for the divergence ledger.
```

LLD module → Stage 1 specialist owner map (for the per-LLD prompt's
"relevant specialist" pointer):

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

## Cleanup at end of Stage 2

When Stage 2 completes (Arthur signs off on HLDs + LLDs):
1. Move this scratchpad to `notes/.archive/stage2-dispatch-prep.md` (or
   just delete it — git history preserves)
2. Delete `_dispatch-prep.md` (the Stage 1 scratchpad)
3. Update PLAN.md Stage 2 completion log
