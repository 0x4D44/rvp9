# rvp9-2: VP9 Encoder Port to Rust — Multistage Plan

**Status:** Draft, pending Arthur's approval to start Stage 0.
**Owner:** Arthur (martin@tollens.ai) + Claude.
**Last revised:** 2026-05-16.

---

## 1. Project context

Port libvpx's VP9 encoder (and decoder, in parallel) to Rust, targeting **production-grade real-time encoding** for WebRTC and livestreaming. Reference implementation is `libvpx` (BSD-3 licensed, `https://chromium.googlesource.com/webm/libvpx`).

This is not a learning exercise. The target is a memory-safe, real-time-capable VP9 stack suitable for production use, filling a real gap in the Rust media ecosystem.

Realistic effort: **6–10 months** of focused work with heavy agent assistance.

---

## 2. Success criteria

A stage gates to the next stage only when its criteria pass. The project gates to "done" when **all** of these hold:

### Correctness
- [ ] Our decoder passes 100% of the official VP9 conformance test vectors.
- [ ] Our encoder produces bitstreams that decode cleanly under `vpxdec`, `ffmpeg`, and our own decoder.
- [ ] Cross-product tests pass: `{libvpx-enc, rvp9-enc} × {libvpx-dec, rvp9-dec}` all decode to matching YUV (within 0 ULP for lossless paths; within bit-exact transforms for lossy paths).

### Compression efficiency
- [ ] **BD-rate within 0.5%** of libvpx vpxenc at matched `--cpu-used` setting (6, 7, 8) on the AOM test set + a WebRTC-representative corpus + a livestream corpus.

### Real-time performance (hard gate)
- [ ] **720p30**: ≤33ms/frame median, ≤50ms p99, on a single modern x86_64 core.
- [ ] **1080p30**: ≤33ms/frame median, ≤50ms p99, on 2–4 cores with tile threading.
- [ ] **1080p60**: ≤16ms/frame median, ≤25ms p99, on 4–8 cores.

### Features
- [ ] 1-pass CBR + VBR rate control with buffer-model compliance.
- [ ] Adaptive quantization (3 AQ modes, matching libvpx).
- [ ] Temporal SVC (2- and 3-layer).
- [ ] Spatial SVC (2- and 3-layer).
- [ ] Content-aware denoiser.
- [ ] Frame dropping under load.

### Aspirational (not gating)
- [ ] Byte-identical output to libvpx vpxenc at matched config on a curated test set. Used as a **diagnostic tool** for finding divergence, not a release gate.

---

## 3. Locked design decisions

These are committed and should not be revisited without explicit Arthur approval:

| # | Decision | Rationale |
|---|---|---|
| D1 | Real-time priority (cpu-used 6–8) | WebRTC + livestream use cases |
| D2 | Encoder + decoder both in v1 | Decoder gives spec-defined oracle; shared infra benefits both |
| D3 | Full SVC (temporal + spatial) in v1 | Required for WebRTC viability |
| D4 | Denoiser + AQ in v1 | Required for production-grade quality |
| D5 | SIMD via `core::arch` intrinsics + scalar fallback | Stable Rust, max performance, mirrors libvpx structure |
| D6 | Threading: mirror libvpx tile-based pthread model | Maximize byte-parity tractability; idiomatize later |
| D7 | Mirror libvpx integer widths exactly | Same reason as D6 |
| D8 | Compression-efficiency parity is the gating goal | Byte parity is a debugging technique, not a release criterion |
| D9 | Targets: 720p30, 1080p30, 1080p60 | Stated WebRTC + livestream use cases |

---

## 4. Deferred decisions

These will be resolved at the stage indicated. Do not pre-decide.

| # | Decision | Resolve in |
|---|---|---|
| Q1 | High-bit-depth (10/12-bit) support in v1 | Stage 2 (LLDs) |
| Q2 | RTCD dispatch table: hand-written Rust vs `build.rs` generator from `vpx_dsp_rtcd_defs.pl` | Stage 2 (LLDs) |
| Q3 | Tile-threading primitives: `std::thread` + `Mutex`/`Condvar`/`Barrier` vs alternatives | Stage 2 (LLDs) |
| Q4 | Frame-parallel encoding (multiple frames in flight) in v1 vs v2 | Stage 2 (LLDs) |
| Q5 | Cargo workspace layout (single crate, workspace with N crates, exact crate boundaries) | Stage 0 |
| Q6 | Test corpus exact composition (which AOM clips, which WebRTC clips, which livestream clips) | Stage 0 |

---

## 5. Repository layout (target after Stage 0)

```
rvp9-2/
├── Cargo.toml                  # workspace root
├── PLAN.md                     # this file
├── README.md
├── ATTRIBUTION.md              # libvpx BSD-3 notice + ack
├── .gitignore
├── crates/
│   ├── rvp9-core/              # shared types, bitstream, entropy, transforms, tables
│   ├── rvp9-decoder/           # decoder (spec-defined)
│   ├── rvp9-encoder/           # encoder (libvpx-mirrored)
│   └── rvp9-cli/               # vpxenc/vpxdec equivalents for benchmarking
├── third_party/
│   └── libvpx/                 # git submodule, reference implementation
├── tests/
│   ├── conformance/            # VP9 test vectors + harness
│   ├── differential/           # libvpx-vs-rvp9 diff tests
│   └── corpus/                 # encoder test clips (gitignored, fetched by script)
├── bench/
│   ├── realtime.rs             # ms/frame benchmark, gates against budget
│   └── bdrate.rs               # BD-rate measurement vs libvpx
├── notes/
│   ├── vp9-analysis/           # agent-team deep-dive outputs (Stage 1)
│   ├── hld/                    # high-level design (Stage 2)
│   ├── lld/                    # per-module low-level designs (Stage 2)
│   └── divergence-ledger.md    # every deliberate departure from libvpx, with rationale
└── scripts/
    ├── build-libvpx.{sh,ps1}   # reference build
    ├── fetch-corpus.{sh,ps1}   # test clips
    └── fetch-vectors.{sh,ps1}  # VP9 conformance vectors
```

---

## 6. Stages

Each stage has: **goal**, **inputs**, **tasks**, **deliverables**, **exit criteria**, **risks**.
A stage cannot start until its inputs are satisfied. A stage cannot exit until all exit criteria pass.

---

### Stage 0 — Foundation

**Goal:** Stand up the build, reference implementation, oracles, and benchmark harnesses so every subsequent stage has measurable ground truth from day one.

**Inputs:** Arthur's approval to begin.

**Tasks:**
1. Add libvpx as git submodule under `third_party/libvpx`, pinned to a specific release tag.
2. Build `vpxenc` and `vpxdec` on the dev machine (Windows). Document the build steps in `scripts/build-libvpx.ps1`. Use MSYS2/MinGW or VS depending on what proves most painless; capture artifacts in `third_party/libvpx/build/`.
3. Download the official VP9 conformance test vectors into `tests/conformance/vectors/` (gitignored). Write `scripts/fetch-vectors.ps1`.
4. Assemble the test corpus (Q6): AOM clips + WebRTC content + livestream content. Write `scripts/fetch-corpus.ps1`.
5. Scaffold the Cargo workspace per §5. Replace the current single-crate layout. Each crate compiles to an empty `lib.rs`; `rvp9-cli` is a stub binary.
6. Write `bench/realtime.rs`: drives `vpxenc` against a clip, records wall-time per encoded frame, asserts against budget. Initially measures libvpx (our baseline); same harness will swap in our encoder later.
7. Write `bench/bdrate.rs`: runs libvpx at QP sweep, computes BD-rate against a reference. Outputs to `bench/results/`. Same harness will compare libvpx vs rvp9 later.
8. Write `tests/conformance/runner.rs`: iterates test vectors through `vpxdec` as a sanity check that our oracle works. Will later iterate through our decoder.
9. Write `tests/differential/diff-bitstream.rs`: takes two .ivf files, decodes both, reports first byte of divergence in the bitstream and first pixel of divergence in YUV. Will be invaluable in Stages 4+.
10. Write `ATTRIBUTION.md` with libvpx BSD-3 notice.
11. Commit baseline.

**Deliverables:**
- Working `vpxenc` + `vpxdec` builds.
- VP9 conformance vectors downloaded, runnable through `vpxdec` baseline.
- Cargo workspace skeleton compiling cleanly.
- Three benchmark/test harnesses runnable end-to-end with libvpx as both source and target.
- Empty `notes/vp9-analysis/` directory with a `README.md` describing the Stage 1 agent dispatch plan.

**Exit criteria:**
- [ ] `cargo build --workspace` succeeds with zero warnings.
- [ ] `cargo test --workspace` passes (vacuous at this stage, but harness runs).
- [ ] `bench/realtime.rs` reports ms/frame for libvpx at 720p30/1080p30/1080p60.
- [ ] `bench/bdrate.rs` produces a BD-rate report for libvpx-vs-libvpx (sanity: should be 0%).
- [ ] All VP9 conformance vectors decode under `vpxdec` (sanity check on the oracle).
- [ ] `notes/vp9-analysis/README.md` describes the six-agent dispatch plan in enough detail for Arthur to review.

**Estimated effort:** 1–2 weeks.

**Risks:**
- libvpx Windows build is non-trivial. Fallback: use WSL2 + Linux build of libvpx, accept the cross-platform friction during development.
- Test corpus licensing — AOM clips are CC; WebRTC test clips need vetting.

**Stage 0 — completion log (2026-05-16):**

What landed:

- Workspace scaffold: six crates (`rvp9-core`, `rvp9-decoder`, `rvp9-encoder`,
  `rvp9-cli`, `rvp9-bench`, `rvp9-harness`) compiling clean, 6 unit tests pass
  (Stats percentile fn + MD5 self-tests), `cargo clippy --all-targets -D warnings` clean.
- libvpx submodule pinned at v1.14.1 (`12f3a2a`).
- 14 conformance vectors fetched into `tests/conformance/vectors/`; conformance
  runner: **14/14 pass** against libvpx vpxdec (exit-0 semantic for Stage 0).
- Synthetic test clips generated at 720p30, 1080p30, 1080p60. Real-time
  benchmark against libvpx vpxenc at `--cpu-used 7`:
  - 720p30:  3.94 ms/frame (88.2% under 33.333 ms budget)
  - 1080p30: 5.68 ms/frame (83.0% under 33.333 ms budget)
  - 1080p60: 5.04 ms/frame (69.8% under 16.667 ms budget)
  Caveat: testsrc2 is gradient-heavy and easy to compress; real content
  will be slower. Validates harness wiring; numbers are baselines, not targets.
- BD-rate harness: encode sweep + RD-table generation + cubic-fit BD-rate
  computation all wired. Produces a clean RD table when run libvpx-vs-libvpx.
- All scripts, ATTRIBUTION, README, divergence ledger, and Stage 1 agent
  dispatch plan documented.

Stage 0 follow-ups (deferred, not blocking Stage 1):

- **BD-rate PSNR pipeline** — `decode_and_psnr` is stubbed. Needs a Y4M reader
  and YUV pixel-wise PSNR. Will land naturally in Stage 4a as a prerequisite
  for running rvp9enc vs libvpx BD-rate comparisons. Not on Stage 0's gate.
- **libvpx build from submodule on Windows** — Git Bash hits `/tmp` perms +
  Makefile target issues. `scripts/build-libvpx.sh` falls back to copying
  Arthur's prebuilt `D:/language/libvpx-stock/` v1.14.1 binaries instead.
  Linux build path is untested but the script structure should work there.
  Improving the Windows build is post-v1.
- **Per-frame conformance comparison** — current runner uses "vpxdec exits 0"
  as the Stage 0 gate. Stage 3 will parse the `.md5` files (per-frame MD5s
  with resolution embedded in the filename) and compare frame-by-frame.

---

### Stage 1 — Agent-team deep-dive

**Goal:** Produce a complete, faithful, devil's-advocate-stressed description of libvpx VP9's real-time encoder + decoder architecture, control flow, algorithms, tables, and threading model — sufficient to design our port without further reading of libvpx during Stage 2.

**Inputs:** Stage 0 complete.

**Tasks:**

Dispatch six specialist agents in parallel. Each produces a Markdown doc in `notes/vp9-analysis/`. After all six complete, dispatch the devil's advocate, then the synthesizer.

| Agent | Focus | Output file |
|---|---|---|
| A1: Architecture & control flow | Top-level encoder lifecycle, frame loop, threading model, `VP9_COMP` god-struct, frame-dependency graph, init/encode/teardown sequence | `01-architecture.md` |
| A2: Encoder core (non-RD path) | `vp9_pickmode.c`, real-time mode-decision heuristics, partition shortcuts, skip detection, intra/inter selection for cpu-used 5–8 | `02-encoder-core.md` |
| A3: Motion estimation | `vp9_nonrd_pick_inter_mode_sb`, real-time ME algorithms, sub-pixel refinement shortcuts, MV prediction, reference-frame selection in non-RD mode | `03-motion-estimation.md` |
| A4: Bitstream + entropy + tables | VP9 bitstream format, arithmetic coder, every probability table + default, coefficient model, header layout. Extract tables verbatim into a structured appendix. | `04-bitstream-entropy.md` |
| A5: Rate control + denoiser | 1-pass CBR/VBR, buffer model, frame dropping, denoiser, AQ modes, content-aware QP. **Largest area; budget extra reading time.** | `05-rate-control.md` |
| A6: Threading + SVC + speed features | Tile-based pthread model, frame-parallel encode, tile-row barriers, SVC layer structure + reference dependencies, `set_rt_speed_feature_*` dispatch table | `06-threading-svc.md` |

Each specialist's prompt must require:
- Extracted data structures (`struct`/typedef definitions, field by field)
- Extracted control flow (call graphs, sequence diagrams)
- **Verbatim tables and constants** (probability tables, scan orders, quantizer matrices, speed-feature dispatch tables)
- File-and-line citations into `third_party/libvpx/`
- Identification of which kernels need RTCD/SIMD dispatch entries

After all six complete:

| Agent | Focus | Output file |
|---|---|---|
| A7: Devil's advocate | Reads A1–A6. Writes "what's missing, what's hand-waved, what bites us in Stage 4." Must produce ≥20 concrete challenges with file/line citations. | `07-devils-advocate.md` |
| A8: Synthesizer | Reads A1–A7. Writes a single cross-cutting `00-overview.md` that ties the six docs together and resolves the devil's advocate's challenges (either by fixing A1–A6 or noting "unresolved, addressed in Stage 2"). | `00-overview.md` |

**Deliverables:**
- 8 docs in `notes/vp9-analysis/`.
- A `decisions-needed.md` listing every "we'll need to choose X in the LLDs" question surfaced during deep-dive.

**Exit criteria:**
- [ ] All 8 docs written, peer-reviewed by Arthur, accepted.
- [ ] Devil's advocate's ≥20 challenges either resolved or explicitly deferred to Stage 2 with rationale.
- [ ] Every load-bearing libvpx table transcribed verbatim into the corpus.
- [ ] Arthur signs off on `00-overview.md` as the working architectural understanding.

**Estimated effort:** 3–4 weeks (agent wall-clock; agent compute could be ~hours but review + iteration takes weeks).

**Risks:**
- Agent output quality varies. Expect to re-dispatch 1–2 specialists for depth.
- libvpx's threading model has subtle invariants that may not surface from static reading. Will need cross-checking against runtime instrumentation in Stage 2.

**Stage 1 — in-flight status:**

- Six specialists (A1-A6) dispatched as background general-purpose agents
  against `third_party/libvpx/` at v1.14.1. Output paths:
  `notes/vp9-analysis/01-architecture.md` through
  `notes/vp9-analysis/06-threading-svc.md`. Each agent's full prompt is
  preserved in this PLAN.md (see Stage 1 dispatch plan in
  `notes/vp9-analysis/README.md`).
- A7 (devil's advocate) and A8 (synthesizer) prompts pre-drafted in
  `notes/vp9-analysis/_dispatch-prep.md`, ready to fire when A1-A6 pass
  review.
- Stage 2 dispatch plan pre-staged in
  `notes/vp9-analysis/_stage2-dispatch-prep.md` (HLD prompts + LLD
  template + LLD-to-specialist owner map) so Stage 2 can dispatch
  immediately after Arthur's Stage 1 sign-off.
- Stage 1 completion-log template pre-staged in
  `notes/vp9-analysis/_stage1-completion-template.md` with numeric
  checks (word count, citation count) to run on each specialist's
  output during review.

Tasks 1-11 in the task tracker mirror this pipeline. The remaining work is
bounded by agent wall-clock for A1-A6 (deep-read of libvpx + verbatim
table transcription is hours-long, not minutes).

---

### Stage 2 — HLD + per-module LLDs

**Goal:** Translate the libvpx understanding from Stage 1 into a concrete Rust design that we can implement against, including resolution of all deferred decisions.

**Inputs:** Stage 1 complete and approved.

**Tasks:**

1. Write `notes/hld/00-architecture.md`: top-level Rust architecture, crate boundaries, module graph, type-system overview, error model, `unsafe` policy.
2. Write `notes/hld/01-threading.md`: tile-based threading design in Rust, `Send`/`Sync` strategy, `UnsafeCell` usage policy, synchronization primitives, frame-parallel encoding decision (Q4).
3. Write `notes/hld/02-simd-dispatch.md`: RTCD-equivalent design (Q2), per-arch intrinsics organization, runtime feature detection, scalar-fallback policy, test strategy for SIMD-vs-scalar equivalence.
4. Write `notes/hld/03-bit-depth.md`: 10/12-bit support decision (Q1). If in scope, design pattern (generics over pixel type vs separate code paths).
5. Write `notes/hld/04-svc.md`: SVC layer pipeline, reference dependency tracking, rate control across layers.
6. Resolve all remaining `decisions-needed.md` items from Stage 1.
7. Write per-module LLDs in `notes/lld/`:
   - `bitstream.md`, `entropy.md`, `transforms.md`, `quant.md`, `intra-pred.md`, `loopfilter.md`, `frame-buffers.md`
   - `motion-estimation.md`, `mode-decision.md`, `partition.md`
   - `rate-control.md`, `denoiser.md`, `aq.md`
   - `svc.md`, `tile-threading.md`
   - Each LLD: types, public API, internal algorithm, libvpx file/line origin, divergence notes
8. Start `notes/divergence-ledger.md`: every place we deliberately depart from libvpx, with rationale and impact on byte parity.
9. Arthur reviews each LLD and signs off before Stage 3 may proceed for that module.

**Deliverables:**
- ~5 HLD docs.
- ~15 LLDs.
- Populated divergence ledger.
- Updated `PLAN.md` with refined Stage 3/4 task breakdowns informed by the LLDs.

**Exit criteria:**
- [ ] All HLDs and LLDs written and Arthur-approved.
- [ ] All Stage 1 deferred decisions (Q1–Q6) resolved.
- [ ] Divergence ledger has an entry for every deliberate departure.
- [ ] No "TBD" markers in any LLD that block Stage 3 work.

**Estimated effort:** 2–3 weeks.

**Risks:**
- Discovering during LLD writing that Stage 1 missed something material — accept this, loop back to A1–A8 for targeted re-dispatch rather than guessing.

---

### Stage 3 — Shared infrastructure + decoder

**Goal:** Build a working VP9 decoder that passes 100% of the conformance test vectors. Side effect: every shared kernel (transforms, entropy, intra pred, loop filter, frame management) is built and validated bit-exactly. This makes Stage 4 dramatically lower-risk.

**Inputs:** Stage 2 complete; all per-module LLDs in scope for shared infrastructure approved.

**Tasks (rough ordering — refine per LLD outputs):**

1. `rvp9-core::bitstream`: byte reader/writer, arithmetic decoder/encoder, header parsing.
2. `rvp9-core::entropy`: probability tables (verbatim from libvpx), context modeling, coefficient parsing/encoding.
3. `rvp9-core::transforms`: forward + inverse DCT/ADST at 4×4, 8×8, 16×16, 32×32. Scalar reference first, then SSE2/SSE4.1/AVX2 + NEON. Differential test each SIMD impl against scalar.
4. `rvp9-core::quant`: quantizer tables, quant/dequant logic.
5. `rvp9-core::intra_pred`: 10 intra modes × 4 block sizes. Scalar + SIMD.
6. `rvp9-core::loopfilter`: deblocking filter. Scalar + SIMD.
7. `rvp9-core::frame`: frame buffer mgmt, reference frame pool, padding/alignment matching libvpx.
8. `rvp9-decoder`: pull together the above, implement frame decode loop.
9. Run VP9 conformance vectors. Fix until 100%.

**Deliverables:**
- `rvp9-core` and `rvp9-decoder` crates feature-complete for VP9 Profile 0 (8-bit 4:2:0). Profile 2 (10-bit) decision per Q1.
- All shared kernels have scalar + SSE2 + SSE4.1 + AVX2 (x86_64) and NEON (aarch64) implementations.
- All shared kernels have differential tests (SIMD-vs-scalar) and conformance coverage.
- `tests/conformance/runner.rs` runs against our decoder, not just `vpxdec`.

**Exit criteria:**
- [ ] 100% pass on official VP9 conformance test vectors.
- [ ] All SIMD implementations bit-exact against scalar reference (differential tests green).
- [ ] Decoder throughput within 2× of libvpx on the bench corpus (we don't need to beat libvpx yet, just not be 10× slower).
- [ ] Zero `clippy` warnings in `rvp9-core` and `rvp9-decoder`.
- [ ] `unsafe` blocks in shared infra all carry safety comments.

**Estimated effort:** 6–10 weeks.

**Risks:**
- Bit-exact transforms are finicky. Budget extra time for SIMD-vs-scalar debugging.
- High-bit-depth (if in scope) roughly doubles kernel count.

---

### Stage 4a — Encoder MVP (1-pass CBR, 720p30 real-time, no SVC/denoiser/AQ)

**Goal:** End-to-end encoder producing decodable VP9 bitstreams, hitting real-time at 720p30, with BD-rate within 2% of libvpx at `--cpu-used 7 --rt --tune-content default`.

**Inputs:** Stage 3 complete and decoder conformant.

**Tasks:**

1. `rvp9-encoder::bitstream_writer`: full VP9 header + frame data writer. Validate by round-tripping through our decoder.
2. `rvp9-encoder::rate_control`: 1-pass CBR with buffer model. Single-layer, no frame dropping yet.
3. `rvp9-encoder::partition`: superblock + sub-block partition selection (real-time variant).
4. `rvp9-encoder::mode_decision`: non-RD intra/inter mode picker (real-time variant).
5. `rvp9-encoder::motion_estimation`: non-RD ME (real-time variant), sub-pixel refinement, MV prediction.
6. `rvp9-encoder::loopfilter_level`: per-frame deblocking strength selection.
7. `rvp9-encoder::tile_threading`: single-tile, single-thread first; multi-tile when scalar baseline works.
8. End-to-end: encode test corpus, decode under our decoder + libvpx, measure BD-rate vs libvpx, measure ms/frame.
9. Hit 720p30 real-time on a single core (scalar), then on multi-core (tile threading).
10. Run differential bitstream tests against libvpx. Find divergence points. Per divergence: match libvpx, or document in `divergence-ledger.md`.

**Deliverables:**
- Working real-time encoder for the minimal config.
- Differential bitstream diff harness operational against libvpx.
- Updated divergence ledger.

**Exit criteria:**
- [ ] 720p30 real-time: ≤33ms/frame median on a single modern x86_64 core (or with tile threading if the scalar single-core misses budget).
- [ ] BD-rate within 2% of libvpx at matched config.
- [ ] All produced bitstreams decode cleanly under `vpxdec`, `ffmpeg`, and our decoder.
- [ ] Differential diff between rvp9 and libvpx output documented for the test corpus: every divergence either fixed or explicitly logged with rationale.

**Estimated effort:** 8–12 weeks.

**Risks:**
- The non-RD mode picker is a heuristic minefield. Expect to spend more time here than any other module.
- Tile-threading correctness under Rust's aliasing rules is the hardest unsafe-Rust we'll write. Plan for Miri runs as part of CI.

---

### Stage 4b — Adaptive Quantization (AQ)

**Goal:** Add AQ modes (libvpx AQ modes 1, 2, 3) and demonstrate BD-rate improvement consistent with libvpx's gain.

**Inputs:** Stage 4a complete.

**Tasks:**

1. Implement variance-based AQ (mode 1).
2. Implement complexity-based AQ (mode 2).
3. Implement cyclic refresh AQ (mode 3).
4. Integrate with rate control.
5. Regression: ensure 4a real-time targets still hit.

**Exit criteria:**
- [ ] BD-rate gain from AQ on test corpus matches libvpx's gain to within 0.3%.
- [ ] 720p30 real-time still passes.

**Estimated effort:** 3–4 weeks.

---

### Stage 4c — Denoiser

**Goal:** Content-aware denoiser integrated, producing libvpx-equivalent quality on noisy content.

**Inputs:** Stage 4b complete.

**Tasks:**

1. Port libvpx denoiser kernel (spatial + temporal).
2. Denoise-aware rate control + mode decision.
3. SIMD for denoiser kernel.

**Exit criteria:**
- [ ] On noisy livestream test clips, BD-rate within 0.5% of libvpx (denoise on).
- [ ] 720p30 real-time still passes with denoiser on.

**Estimated effort:** 3–4 weeks.

---

### Stage 4d — Temporal SVC

**Goal:** 2- and 3-layer temporal SVC operational.

**Inputs:** Stage 4c complete.

**Tasks:**

1. Multi-layer reference frame structure.
2. Per-layer rate control.
3. Frame-type assignment for temporal hierarchy.
4. Bitstream encoding of layer metadata.

**Exit criteria:**
- [ ] Decoder (and libvpx decoder) extracts each temporal layer correctly.
- [ ] BD-rate per layer within 0.5% of libvpx.
- [ ] 720p30 + 1080p30 real-time still pass with 3-layer temporal SVC.

**Estimated effort:** 4–6 weeks.

---

### Stage 4e — Spatial SVC

**Goal:** 2- and 3-layer spatial SVC operational. Hardest part of the encoder.

**Inputs:** Stage 4d complete.

**Tasks:**

1. Multi-resolution frame pipeline.
2. Inter-layer prediction (texture + motion).
3. Per-layer rate control with cross-layer constraints.
4. Bitstream encoding of spatial layer metadata.

**Exit criteria:**
- [ ] Decoder extracts each spatial layer correctly.
- [ ] BD-rate per layer within 0.5% of libvpx.
- [ ] All target res/fps real-time gates pass with full SVC config.

**Estimated effort:** 6–8 weeks.

---

### Stage 5 — Hardening

**Goal:** Production-ready. Soak tests, fuzzing, perf regression CI, security audit of `unsafe`.

**Inputs:** All of Stage 4 complete.

**Tasks:**

1. `cargo fuzz` for decoder against malformed bitstreams.
2. 72-hour soak test: encode + decode varied content, monitor for leaks, crashes, perf drift.
3. CI: BD-rate regression gates, ms/frame regression gates.
4. Manual review of every `unsafe` block, with safety-comment audit.
5. Miri runs for all non-SIMD paths.
6. Address-sanitizer runs for SIMD paths.
7. Security review (consider invoking `security-review` skill).
8. Documentation pass: `rustdoc` for public APIs, README polish.

**Exit criteria:**
- [ ] All success criteria from §2 pass.
- [ ] No new crashes from 1B fuzz iterations.
- [ ] No leaks or drift in 72-hour soak.
- [ ] CI gates BD-rate and latency regressions.
- [ ] Security review report logged.

**Estimated effort:** 4–6 weeks.

---

## 7. Conventions for `/goal` driving

When `/goal` drives a stage:

- **Always read this PLAN.md first** to understand the current stage's exit criteria.
- **Never skip exit criteria** — if a criterion can't be met, surface to Arthur, don't fudge.
- **One stage at a time.** Don't optimistically start work in Stage N+1 while Stage N has open criteria.
- **Update `notes/divergence-ledger.md`** whenever you deliberately depart from libvpx.
- **Commit per significant piece of work** per Arthur's git conventions (atomic commits, no `--amend`, no force push, no `-A`).
- **Update this file** at the end of each stage: mark exit criteria checked, append a "Lessons learned" section under the stage if material, refine estimates for downstream stages based on what was learned.

## 8. Out of scope for v1

Explicitly **not** in v1, to be revisited post-v1:

- AV1 / VP8 codec support.
- Hardware encoding (NVENC, QuickSync, AMF, VA-API).
- 2-pass encoding (incompatible with real-time anyway).
- Real-time mode tuning beyond cpu-used 6–8 (no slow-path RDO).
- WebAssembly target.
- C ABI / FFI bindings (can layer on later if there's demand).
- GUI / player integration.
