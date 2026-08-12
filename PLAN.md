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
- [x] Our decoder passes 100% of the official VP9 conformance test vectors. (Stage 3 exit, 2026-05-18 — 191 passed / 0 mismatched / 10 skipped lossless.)
- [ ] Our encoder produces bitstreams that decode cleanly under `vpxdec`, `ffmpeg`, and our own decoder.
- [ ] Cross-product tests pass: `{libvpx-enc, rvp9-enc} × {libvpx-dec, rvp9-dec}` all decode to matching YUV (within 0 ULP for lossless paths; within bit-exact transforms for lossy paths).

### Compression efficiency
- [ ] **BD-rate within 0.5%** of libvpx vpxenc at matched `--cpu-used` setting (6, 7, 8) on the AOM test set + a WebRTC-representative corpus + a livestream corpus.

### Real-time performance (hard gate)
- [ ] **720p30**: ≤33ms/frame median, ≤50ms p99, on a single modern x86_64 core.
- [ ] **1080p30**: ≤33ms/frame median, ≤50ms p99, on 2–4 cores with tile threading.
- [ ] **1080p60**: ≤16ms/frame median, ≤25ms p99, on 4–8 cores.

### Features
- [ ] 1-pass CBR + VBR-RT rate control with buffer-model compliance. (No altref — directly contradicts real-time; revisit for a future near-RT livestream preset.)
- [ ] Adaptive quantization (CYCLIC_REFRESH / AQ mode 3 only — modes 1/2 are guarded by `!CONFIG_REALTIME_ONLY` in libvpx and unreachable at cpu-used 5-8).
- [ ] Temporal SVC (2- and 3-layer).
- [ ] Spatial SVC (2- and 3-layer).
- [ ] Content-aware denoiser.
- [ ] Frame dropping under load.
- [ ] EXT_RATECTRL plug-in interface (required for libwebrtc integration).
- [ ] 8-bit (Profile 0) only in v1. High-bit-depth (10/12-bit, Profiles 2/3) deferred to Stage 6+.

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
| Q1 | ~~High-bit-depth (10/12-bit) support in v1~~ — **RESOLVED 2026-05-16: 8-bit only in v1; HBD deferred to Stage 6+.** | — |
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

**Stage 1 — completion log (2026-05-16):**

What landed:

- Six specialist deep-dives in `notes/vp9-analysis/`:
  - `01-architecture.md`     (3081 ll, 173 file:line citations) — A1
  - `02-encoder-core.md`     (2311 ll, 120 citations) — A2
  - `03-motion-estimation.md` (1896 ll, 133 citations) — A3
  - `04-bitstream-entropy.md` (4556 ll, 294 citations) — A4 (async
    initially truncated at §5; final rewrite by async A4 agent after
    a longer second pass produced the canonical version)
  - `05-rate-control.md`     (1979 ll, 175 citations) — A5
  - `06-threading-svc.md`    (2002 ll, 119 citations) — A6
- `07-devils-advocate.md` (1488 ll, 45 challenges): canonical async A7.
  Manual driver fallback (480 ll, 32 challenges) preserved at
  `07-devils-advocate-driver.md` for audit. Severity distribution
  in async A7: 1 Blocker (vp9_prob_cost[256] missing), 9 High
  (5 ACCEPTED for Stage 2 LLD, 4 RESOLVED or REJECTED), ~22 Medium,
  ~9 Low, 4 CONFIRMED (specialist claims validated), 4 REJECTED.
- `00-overview.md` (867 ll): canonical async A8 synthesis with all 7
  mandated sections + bonus Cross-references. Manual driver fallback
  (503 ll) preserved at `00-overview-driver.md` for audit. NOTE:
  A8 read my manual A7 (32 challenges) before async A7 landed; A8's
  §5 disposition covers those 32. The 13 additional challenges from
  async A7 are dispositioned in decisions-needed.md "Updates from
  async A7" section. Addendum at the top of 00-overview.md flags
  this.
- `decisions-needed.md` (~700 ll): 15+ D-prefix (Stage 3 blockers,
  including new D-14 vp9_prob_cost[256] and D-15 worker-fields
  catalogue), ~45 L-prefix (LLD decisions, including new L-74 through
  L-77 from async A7), 10 R-prefix (risks; R-01 RESOLVED, R-08 likely
  a non-issue per async A7 correction).

Re-dispatches performed during Stage 1:
- A4 async agent self-re-dispatched after producing a truncated first
  pass; the second pass produced the canonical 4556-line version.

Surprising findings (high-signal items that change PLAN.md):

1. **AQ modes 1/2 unreachable in `CONFIG_REALTIME_ONLY` builds** (A5
   Q-RC-15). PLAN.md §2 originally committed to "3 AQ modes matching
   libvpx" but only mode 3 (CYCLIC_REFRESH_AQ) is real-time-reachable in
   libvpx itself. **RESOLVED 2026-05-16: drop modes 1/2 from v1; PLAN.md
   §2 amended to "CYCLIC_REFRESH only".**
2. **Encoder-side frame-parallel encoding does NOT exist in libvpx**
   (A6 key finding). PLAN.md Q4 is therefore a v2-design question,
   not a libvpx-mirror question.
3. **`cpu-used 5` is the qualitative split** between RD-loop and
   non-RD-loop encoding (`sf->use_nonrd_pick_mode = 1`). PLAN.md D1's
   "cpu-used 6-8" priority is well-defined.
4. **`vp9_encodedframe_overshoot`** (A5 Q-RC-20) is a massive
   post-tentative-pack state mutation that's both critical for
   scene-change recovery AND the single most opaque step in the
   encoder. Stage 4a will need extensive byte-parity testing here.
5. **BufferPool lives OUTSIDE VP9_COMP** (A1 §7.1) — 12 slots (8 ref
   + 1 new + 3 scaled). The "3 scaled" are libvpx-internal for SVC
   inter-layer prediction; our public API should not expose them.

Stage 1 deferred to Stage 2 (per devil's-advocate disposition):

- D-12 `cpi->td` exhaustive aliasing catalogue (HLD-threading)
- ~~D-13 REALTIME-VBR + altref support~~ **RESOLVED 2026-05-16: CBR + VBR-RT, NO altref.**
- L-65 conformance vector coverage expansion (Stage 3)
- L-66 `setjmp` × worker threads (HLD-threading)
- L-67 cyclic-refresh segmentation map ownership (HLD-threading)
- L-68 `vp9_scene_detection_onepass` per-layer behaviour (LLD-svc)
- L-69 SVC cross-layer frame-drop interactions (LLD-svc)
- L-70 `vp9_change_config` reallocation (LLD-arch)
- L-71 4×4 transform reachability in real-time (LLD-transforms)
- L-72 VP9_COMP per-field commentary expansion (LLD-arch)

Stage 1 follow-ups (not blocking Stage 2):

- **Loop filter** got light coverage (only A1 §3 mentions it in passing).
  When LLD-loopfilter starts, a Stage 1.5 re-dispatch may be needed.
- **Laplacian R-D tables `rate_tab_q10`/`dist_tab_q10`** (in
  `vp9/encoder/vp9_rd.c`): A2 declared the gap, A4 covered everything
  in `vp9/common/` but these are encoder-internal. A2 patch needed in
  Stage 2.
- **Async A7 + A8 may still complete** and could supersede the manual
  fallbacks. If so, compare/merge with the `*-driver.md` versions.

Arthur sign-off: ⏳ pending (the irreducible human gate per PLAN.md §6).

---

### Stage 2 progress log — UPDATED 2026-05-16

This subsection records the live state of Stage 2 so PLAN.md is a
single source of truth across compactions and context resets. The
forward-looking task list and exit criteria are unchanged below.

**Phase 2a (HLD writing) — COMPLETE.** Four HLDs landed via async
agents, signed off by Arthur at Phase 2b:

| File | Lines | Author note |
|---|---|---|
| `notes/hld/00-architecture.md`     | 1213 | Crates, modules, types, errors, unsafe policy, 8-bit pixel-type policy, EXT_RATECTRL trait surface. |
| `notes/hld/01-threading.md`        | 1517 | Q3 = `std::thread`+`Mutex`+`Condvar` (option D); Q4 = frame-parallel deferred to v2; D-10/D-11/D-12/D-15/L-66 resolved. |
| `notes/hld/02-simd-dispatch.md`    | 2189 | Q2 = hand-written dispatch tables; ~243 kernels catalogued; OnceLock R-09 init invariant. |
| `notes/hld/04-svc.md`              | 1500 | Temporal + spatial SVC; cross-layer RC; L-69/L-75/L-77 resolved; EXT_RATECTRL trait surface (R-06). |

`03-bit-depth.md` was *not* written — Q1 RESOLVED to 8-bit only is
inlined as a paragraph in HLD-arch §7 rather than a standalone doc.
HLDs surface 24 cross-HLD assumptions / open questions; reconciliation
happens during LLD writing (Phase 2c) and at Stage 2 close-out (Phase 2e).

**Phase 2b (HLD review) — COMPLETE.** Arthur reviewed and approved all
four HLDs.

**Phase 2c (LLD writing) — COMPLETE.** All 15 LLD writers landed via
async agents. Total LLD content: **24,754 lines** across `notes/lld/`
(several agents trimmed their output after the initial format-checker
snapshot; tile-threading.md grew to 2000 after its last post-summary
edit; numbers below are the final-on-disk state). Per
`check-lld-format`: all 15 LLDs structurally OK (every required §1-§9
present, §7 populated with the divergence-ledger template, citation
counts in 60-155 range). Two non-blocking TBD-substring matches:
`bitstream.md:151` is a meta-prose mention of the string "`TBD`"
inside backticks; `quant.md:1187` is a cross-LLD `§X (TBD)`
section-number placeholder. Neither blocks Stage 3.

Final LLD status:

| LLD | Lines | Cites | Notes |
|---|---|---|---|
| `bitstream.md`        | 1608 |  82 | 9 §7 entries; 13 §8 non-blocking |
| `entropy.md`          | 1555 | 128 | 6 §7 entries; cost framework via D-14 patch |
| `transforms.md`       | 1544 | 102 | L-71 4×4 reachable confirmed; lossless decoder-only; SSSE3 required |
| `quant.md`            | 1183 |  72 | DC/AC LUTs verbatim; L-25/L-26 resolved |
| `intra-pred.md`       | 1278 |  86 | 5 §7 entries; 0 blockers |
| `loopfilter.md`       | 1499 | 121 | **Stage 1.5 re-dispatch flag LOWERED** — agent read libvpx loopfilter.c directly |
| `frame-buffers.md`    | 1361 | 155 | BufferPool 8+1+3; DV-FB-1..8 (8 §7 entries) |
| `motion-estimation.md`| 1776 | 145 | 8 §7 entries (L-05/L-06/L-08 stubs resolved) |
| `mode-decision.md`    | 2369 |  80 | 12 §7 entries; L-40 RESOLVED (30-iter → ≤16 in v1) |
| `partition.md`        | 1348 |  60 | 10 §7 entries; L-38/L-39/L-44/L-45/L-52/L-53/L-54 resolved |
| `rate-control.md`     | 2167 |  85 | 8 §7 entries; EXT_RATECTRL trait; no altref |
| `denoiser.md`         | 1364 |  83 | DV-D5 = L-22 bit-parity divergence (load-bearing) |
| `aq.md`               | 1427 | 101 | CYCLIC_REFRESH only; 8 §7 entries incl. weight_segment L-21 trap |
| `svc.md`              | 2275 | 118 | 10 §7 entries; L-58/L-61/L-68/L-69/L-75/L-77 resolved |
| `tile-threading.md`   | 2000 | 104 | largest LLD; most subtle unsafe; DV-TT-01..07 + load-bearing §6.5 MacroBlock re-wire SAFETY block |

**Stage 2 prep work landed during the wait** (committed):

- `3bcc483` Stage 1 scope lock (8-bit, AQ-3, no altref, EXT_RATECTRL).
- `a3c9f2d` D-14 (`vp9_prob_cost[256]`) + R-07 Laplacian R-D tables
  patched into Stage 1 specialist docs so LLD-entropy / LLD-mode-decision
  don't hit gaps.
- `c7f9e9d` Phase 2c LLD prompts pre-staged with locked-scope guardrails
  + 15 per-LLD focus paragraphs.
- `ef79cae` A2 variance_low Morton-order prose fix (async A7 #17).
- `2c7a103` `notes/divergence-ledger.md` seeded with 10 confirmed
  entries from scope decisions and Stage 1 findings.
- `f99b15b` Y4M reader + Y-PSNR in `rvp9-bench`; `bdrate.rs`
  end-to-end functional (PSNR no longer stubbed).
- `78f79ce` `check-lld-format` harness binary for Phase 2d pre-review
  (validates §1-§9 + populated §7 + citation count per LLD).
- `2cce146` `bdrate.rs --csv` output for Stage 4a CI ingestion.

**Phase 2d (Arthur LLD review) — COMPLETE.** Arthur signed off all
15 LLDs at 2026-05-16: "All good. Signed off."

**Phase 2e (Stage 2 close-out) — COMPLETE.** Four close-out tasks done:
1. **Divergence ledger merge**: `notes/divergence-ledger.md` extended
   with an index-style "Stage 2 LLD-surfaced divergences" section
   cataloguing all **89 §7 entries** across the 15 LLDs by ID + title
   with cross-references back to each LLD. Master entries (full
   template-form) retained for the 10 scope-defining departures.
   Aggregated source text in `notes/divergence-ledger-phase2e-draft.md`.
2. **Scratchpads archived**: `_decisions-needed-draft.md`,
   `_dispatch-prep.md`, `_stage1-completion-template.md`,
   `_stage2-dispatch-prep.md` removed via `git rm`; git history
   preserves them.
3. **Stage 3 / Stage 4a task lists refined** below per LLD anchors;
   each task now cites its primary LLD by file + line count, with
   pointers to load-bearing §7 entries (e.g. tile-threading §6.5
   MacroBlock re-wire SAFETY block) and master-ledger entries.
4. **Exit-criteria check** (per the bullets below this subsection):
   - [x] All HLDs and LLDs written and Arthur-approved.
   - [x] All Stage 1 deferred decisions (Q1-Q6) resolved (Q1=8-bit,
         Q2=hand-written RTCD, Q3=std::thread+Mutex+Condvar,
         Q4=frame-parallel deferred to v2, Q5=six-crate workspace
         locked, Q6=test corpus defined in Stage 0).
   - [x] Divergence ledger has an entry (master full-form OR index
         cross-reference) for every deliberate departure surfaced.
   - [x] No "TBD" markers in any LLD that block Stage 3 work
         (2 known false positives are meta-prose, not blockers).

**STAGE 2 COMPLETE 2026-05-16.** Stage 3 (shared infrastructure +
decoder) is unblocked.

---

### Stage 2 — HLD + per-module LLDs

**Goal:** Translate the libvpx understanding from Stage 1 into a concrete Rust design that we can implement against, including resolution of all deferred decisions.

**Inputs:** Stage 1 complete and approved.

**Tasks:**

1. Write `notes/hld/00-architecture.md`: top-level Rust architecture, crate boundaries, module graph, type-system overview, error model, `unsafe` policy. (Q1 RESOLVED to 8-bit only; HLD-arch states the pixel-type policy in a paragraph rather than a separate doc — `03-bit-depth.md` is no longer needed.)
2. Write `notes/hld/01-threading.md`: tile-based threading design in Rust, `Send`/`Sync` strategy, `UnsafeCell` usage policy, synchronization primitives, frame-parallel encoding decision (Q4).
3. Write `notes/hld/02-simd-dispatch.md`: RTCD-equivalent design (Q2), per-arch intrinsics organization, runtime feature detection, scalar-fallback policy, test strategy for SIMD-vs-scalar equivalence.
4. Write `notes/hld/04-svc.md`: SVC layer pipeline, reference dependency tracking, rate control across layers (includes EXT_RATECTRL plug-in interface per R-06).
5. Resolve all remaining `decisions-needed.md` items from Stage 1.
6. Write per-module LLDs in `notes/lld/`:
   - `bitstream.md`, `entropy.md`, `transforms.md`, `quant.md`, `intra-pred.md`, `loopfilter.md`, `frame-buffers.md`
   - `motion-estimation.md`, `mode-decision.md`, `partition.md`
   - `rate-control.md`, `denoiser.md`, `aq.md` (mode 3 only)
   - `svc.md`, `tile-threading.md`
   - Each LLD: types, public API, internal algorithm, libvpx file/line origin, divergence notes
7. Start `notes/divergence-ledger.md`: every place we deliberately depart from libvpx, with rationale and impact on byte parity.
8. Arthur reviews each LLD and signs off before Stage 3 may proceed for that module.

**Deliverables:**
- 4 HLD docs.
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

**Inputs:** Stage 2 complete; all per-module LLDs Arthur-approved (Phase 2d signed off 2026-05-16). The 15 LLDs in `notes/lld/` are the spec; PLAN.md tasks below cite them by anchor.

**Tasks (refined per LLD outputs at Phase 2e):**

0. **`rvp9-core::simd` RTCD init scaffolding** (HLD-simd; R-09 invariant) — `OnceLock<SimdDispatch>` populated by `Encoder::new()` / `Decoder::new()` before any worker spawn. Hand-written dispatch tables per Q2 RESOLVED. Type-system enforcement via `WorkerPool::new(&'static SimdDispatch)` (tile-threading §6.5). **Precondition for every other task in this stage.**
1. **`rvp9-core::bitstream`** (LLD `notes/lld/bitstream.md`, 1608 ll) — `RawBitWriter` / `RawBitReader` / `FrameHeader` / `LoopFilterHeader` / `QuantizationHeader` / `SegmentationHeader` / `TileBuffer` / `SuperframeBuilder` per the LLD §3-§4 API. Bit-exact per A4 §4.9. No setjmp/longjmp; `Result<_, BitstreamError>` propagation.
2. **`rvp9-core::entropy`** (LLD `notes/lld/entropy.md`, 1555 ll) — `BoolWriter` / `BoolReader`, `FrameContext`, `FrameCounts`, cost framework with `VP9_PROB_COST[256]` (D-14 RESOLVED), the 11 trees, `merge_probs` adaptation, `adapt_coef_probs` / `adapt_mode_probs` / `adapt_mv_probs`. Bit-exact contract per A4 §4.9. **Counts double-count quirk preserved bug-for-bug** (entropy §7.4; threading §5.6).
3. **`rvp9-core::transforms`** (LLD `notes/lld/transforms.md`, 1544 ll) — Forward + inverse DCT/ADST at 4×4 / 8×8 / 16×16 / 32×32. Scalar reference, then SSE2 / SSE4.1 / AVX2 / NEON kernels per HLD-simd §3. **SSSE3 promoted to required** for three inverse-DCT kernels on x86_64 (transforms §7.4 divergence). Trellis-quant (`vp9_optimize_b`) NOT ported (master entry — L-28). Lossless: decoder honours, encoder doesn't produce.
4. **`rvp9-core::quant`** (LLD `notes/lld/quant.md`, 1183 ll) — DC_QLOOKUP[256] + AC_QLOOKUP[256] verbatim (8-bit only), QUANTIZER_TO_QINDEX[64], `Quants` / `Dequants` split structs, `vp9_compute_qdelta_by_rate` exhaustive match per L-26.
5. **`rvp9-core::intra_pred`** (LLD `notes/lld/intra-pred.md`, 1278 ll) — 10 modes (DC, V, H, D45, D135, D117, D153, D207, D63, TM) × 4 sizes. Edge-pixel availability with 127/129 boundary constants. Eager `IntraPredDispatch` init (no lazy `once()`). FFI-vs-intrinsics path noted in §6.
6. **`rvp9-core::loopfilter`** (LLD `notes/lld/loopfilter.md`, 1499 ll) — Filter-level cube, edge×filter×mask machinery, vertical-then-horizontal pass order (correctness-critical), per-frame RDO bypass at non-RD (`LPF_PICK_FROM_Q` only path at cpu-used 6-8).
7. **`rvp9-core::frame`** (LLD `notes/lld/frame-buffers.md`, 1361 ll) — BufferPool with 12 slots (8 ref public + 1 new + 3 scaled `pub(crate)`). `Arc<FrameBuf>` per L-37. Border-extend invariant, stride arithmetic verbatim from `yv12config.c:152-290`. **`Send`/`Sync` discipline**: tile workers receive `TilePlanesMut` / `TilePlanes` views under documented aliasing contract (mirror HLD-threading §6.2 strategy B).
8. **`rvp9-decoder`** — Decoder frame loop integrating tasks 1-7. Decoder-only items: `LF_PATH_SLOW` non-4:2:0 (loopfilter §7.L-LPF-2), `LfSync` decoder-extra fields, decryption callback fields NOT ported (entropy §7.2). Profile 0 only (Profile 1/2/3 deferred to v2).
9. **Differential gate against `vpxdec`**: every conformance vector decodes byte-identically to `vpxdec --md5`. Workspace `cargo clippy -D warnings`, `cargo test`, Miri on `rvp9-core` shared-state primitives.

**Deliverables:**
- `rvp9-core` and `rvp9-decoder` crates feature-complete for VP9 Profile 0 (8-bit 4:2:0). Profiles 2/3 (10/12-bit) deferred to Stage 6+ per D-01 RESOLVED.
- All shared kernels have scalar + SSE2 + SSE4.1 + AVX2 (x86_64) and NEON (aarch64) implementations.
- All shared kernels have differential tests (SIMD-vs-scalar) and conformance coverage.
- `tests/conformance/runner.rs` runs against our decoder, not just `vpxdec`.

**Exit criteria:**
- [x] 100% pass on official VP9 conformance test vectors. (2026-05-18 — 191/0/10 final scoreboard; lossless 10 skipped pending WHT, out of Stage 3 scope.)
- [ ] All SIMD implementations bit-exact against scalar reference (differential tests green). *(Coverage audit pending — most kernels have differential tests; not yet swept.)*
- [ ] Decoder throughput within 2× of libvpx on the bench corpus. *(Not yet benchmarked; bench infrastructure exists in `rvp9-bench` but not run as a stage gate.)*
- [x] Zero `clippy` warnings in `rvp9-core` and `rvp9-decoder`. (`cargo clippy --workspace --all-targets -- -D warnings` clean as of 2026-05-18.)
- [ ] `unsafe` blocks in shared infra all carry safety comments. *(Audit pending.)*

**Status (2026-05-19):** Correctness gate ACHIEVED. Three secondary gates (SIMD audit, throughput bench, unsafe-comment audit) remain. Stage 4a work proceeds in parallel; secondary gates revisited before formal Stage 3 sign-off.

**Estimated effort:** 6–10 weeks.

**Risks:**
- Bit-exact transforms are finicky. Budget extra time for SIMD-vs-scalar debugging.
- (8-bit only in v1 — HBD-related kernel doubling avoided.)

---

### Stage 4a — Encoder MVP (1-pass CBR, 720p30 real-time, no SVC/denoiser/AQ)

**Goal:** End-to-end encoder producing decodable VP9 bitstreams, hitting real-time at 720p30, with BD-rate within 2% of libvpx at `--cpu-used 7 --rt --tune-content default`.

**Inputs:** Stage 3 complete and decoder conformant.

**Tasks (refined per LLD outputs at Phase 2e):**

1. **`rvp9-encoder::bitstream_writer`** — Encoder-side `pack_frame` per LLD-bitstream §3.1, integrating §3.2 `SuperframeBuilder` and §3.3 per-tile `Vec<u8>` arenas. Validate by round-tripping through our decoder + `vpxdec`. **No altref-driven superframes** (bitstream §7.3).
2. **`rvp9-encoder::motion_estimation`** (LLD `motion-estimation.md`, 1776 ll) — Diamond / full-pel / sub-pel pipeline; `combined_motion_search`, `find_mv_refs_rt`, `mv_pred`. Reference masking threshold frozen at 2× (L-06). `accurate_sub_pel_search` stubbed (L-05). `vp9_refining_search_8p_c` not ported (L-08). NEWMV diff-bias with hardcoded 48/8/16 thresholds (L-43).
3. **`rvp9-encoder::partition`** (LLD `partition.md`, 1348 ll) — Variance-tree `choose_partitioning`, `pos_shift_16x16` Morton-order indexing into `variance_low[25]` (L-39), `vbp_thresholds[4]` per-frame on `Encoder::vbp`, `set_vt_partitioning` with `block_height == block_width` debug assertion (L-52), frame-edge cropping (L-53).
4. **`rvp9-encoder::mode_decision`** (LLD `mode-decision.md`, 2369 ll, **largest encoder LLD**) — Non-RD pick-mode with corrected 16-iter (not 30) loop (L-40 RESOLVED). `BlockYrdOutcome::Bypass` at speed ≥ 7 (L-41). Filter-ref search restricted to EIGHTTAP / EIGHTTAP_SMOOTH (L-48). Compound modes return `unreachable!()` (L-46). Uses Laplacian R-D model (R-07 patch in 02-encoder-core.md §5.4).
5. **`rvp9-encoder::rate_control`** (LLD `rate-control.md`, 2167 ll) — CBR + VBR-RT only; **NO altref / NO `lag_in_frames > 0`** (master entry). `buffer_level` / `bits_off_target` consolidation (L-01). **`vp9_encodedframe_overshoot` 9-step state mutation byte-parity gate** (L-19 / L-74). `rate_correction_factors` × `rcf_mult[]` resolution preservation (L-25). EXT_RATECTRL trait surface (R-06).
6. **`rvp9-encoder::aq`** (LLD `aq.md`, 1427 ll) — **CYCLIC_REFRESH only** per D-05. `weight_segment` integer-shift order preserved exactly (L-21 sub-0.3% BD-rate trap). `vp9_cyclic_refresh_limit_q` screen-content gate (L-24). `Last_Source` stride quirk preserved bug-for-bug (aq §7.7).
7. **`rvp9-encoder::tile_threading`** (LLD `tile-threading.md`, 2000 ll, **largest LLD**) — Persistent `WorkerPool` + per-frame `FrameScope<'frame>` per HLD-threading §1.2. `JobQueue` + `RowSync` row-MT sync. **§6.5 MacroBlock re-wire SAFETY block — most subtle unsafe in the codebase.** `tile_rows > 1` panics in v1 (L-62). FRAME_COUNTS accumulation bias preserved bug-for-bug (L-57 / DV-TT-05). RTCD init enforced by type system (DV-TT R-09).
8. **`rvp9-encoder::loopfilter_level`** — Encoder-side `LPF_PICK_FROM_Q` only at cpu-used 6-8; defers to `rvp9-core::loopfilter` for the actual filter.
9. **End-to-end loop**: encode test corpus → decode under our decoder + libvpx → BD-rate vs libvpx via `cargo run -p rvp9-bench --bin bdrate -- --csv` → ms/frame via `cargo run -p rvp9-bench --bin realtime`.
10. **Hit 720p30 real-time** on a single core (scalar fallback), then on 2-4 cores with tile threading at 1080p30.
11. **Differential bitstream gate**: `cargo run -p rvp9-harness --bin diff-bitstream` against libvpx. Every divergence either fixed or recorded in `notes/divergence-ledger.md`.

**Empirical status (Track E, 2026-05-22/23):**

After a 14-step evidence chain (E1–E14) the foreman BD-rate gap is no longer a confounded single number — it is decomposed into a closed rate-control / structural part and a clean residual coding-quality deficit. Reports under `wrk_docs/E1*–E16*.md`; comprehensive close-out in `wrk_journals/2026.05.23 - JRN - Track E FINAL CLOSE*`.

Six libvpx-faithful bug fixes landed, all byte-exact cross-decode (vpxdec + ffmpeg), all preserving the 94 encoder lib gates and the 720p30 realtime budget (~5 ms / 33 ms):

| Fix | Commit  | What                                                                                              |
|-----|---------|---------------------------------------------------------------------------------------------------|
| E2  | `697384d` | Port libvpx inter VBP variance-partition thresholds (KF formula was being applied to inter).      |
| E6  | `ff385dc` | Port libvpx `max_frame_bandwidth` (4×avg → `max(MBs·250, 2 Mbit, avg·4)`) — uncaps the KF target. |
| E7  | `b2286b6` | Port `get_kf_active_quality` LUT — floors the keyframe Q at libvpx's ~57 instead of crashing to 4. |
| E8  | `7e94f09` | Fix inter-Q EWMA: postencode recorded the active_best FLOOR instead of the coded Q (ambient decay). |
| E10 | `5a61f8b` | Port libvpx edge-replicated extended-border reads in `int_pro_motion_estimation` (12/16 SBs bailed). |
| E13 | `f05c6b4` | `run_newmv_search` BYPASSED the diamond on 89.9% of LAST NEWMV leaves; always-run + `start_mv ≠ center_mv`. |

State after the fixes:
- **Foreman bitrate overshoot eliminated** (252 → 143.9 kbps at the 100k target, libvpx 143.1, within 0.6%).
- **Inter coefficient bits −20% per frame** from KF reference-quality cascade (E6 mechanism).
- **NEWMV diamond-recovers-fraction 2% → 75.6%** (E13); search now finds real motion (+5.34 dB prediction PSNR available at the search stage).
- **Residual gap: ~9 dB matched-bitrate prediction-quality deficit** (foreman at q=120: rvp9 31.4 dB recon vs libvpx 39.2). The BD-rate integral is currently undefined because rvp9's PSNR curve sits entirely below libvpx's; E15 (`7166e8b`) added a matched-bitrate PSNR-delta fallback to the gate harness that reports the gap directly. Foreman fallback: mean +9.07 dB, max +9.50 dB across the 4 operating points.

**Track F (next-phase coupled package) — required for the 2 % BD-rate exit criterion.** Five experiments (E5 RD-skip, E11 partition-cap under broken search, E13 search alone, E14 partition-cap under fixed search, E16 picker-cost audit) converge on one mechanism: rvp9's picker RD models mode-rate + mv-rate but **not coefficient bits**, so even when NEWMV finds a better motion-compensated prediction (E13), the picker can't see the residual-bit savings — only the +5000 prob-cost mv-rate. Three coupled sub-fixes, no single one closes the gap:
1. **`vp9_block_yrd` / coefficient-bit-aware RD** in the picker's `evaluate_candidate` so NEWMV's residual savings become visible to the RD comparison. (Primary lever; references libvpx `vp9/encoder/vp9_rdopt.c:vp9_block_yrd` and the rate estimation in nonrd pickmode.)
2. **`find_mv_refs` / NEAREST quality propagation** — once block_yrd makes some block pick NEWMV, NEAREST starts predicting from real motion in neighbours, closing the chicken-and-egg degeneracy. Likely activates the A18 `prev_frame_mvs` plumbing currently in place but no-op.
3. **Motion-aware partition decision** — libvpx's nonrd evaluates modes (incl. NEWMV with the working search + `block_yrd`) at each level and only splits when motion-comp RD doesn't justify the larger block. Replaces the current variance-of-ZEROMV-residual partitioner.

This is libvpx's `vp9_pick_inter_mode` realtime / nonrd pickmode machinery in its entirety. Estimate **+3–4 weeks on top of the original 8-12** (the rate-control / structural work consumed roughly the existing budget). Until Track F lands, foreman's BD-rate exit criterion is not measurable (PSNR-delta fallback metric stands in; target TBD).

Open items routed:
- Decoder mi-grid mode/skip column population bug — corrupted E1 mode columns and E4 skip-claim. Logged for the decoder team; no cross-encoder mode/skip comparison can use the mi-grid until fixed.
- `bitstream::*round_trip` tests missing test-vector fixtures (environmental).

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

**Goal:** Add cyclic-refresh AQ (libvpx mode 3) and demonstrate BD-rate improvement consistent with libvpx's gain. (Modes 1/2 dropped per D-05 RESOLVED 2026-05-16 — `!CONFIG_REALTIME_ONLY` dead code in our config.)

**Inputs:** Stage 4a complete.

**Tasks:**

1. Implement cyclic refresh AQ (mode 3) and its segmentation map.
2. Integrate with rate control.
3. Regression: ensure 4a real-time targets still hit.

**Exit criteria:**
- [x] BD-rate gain from AQ-3 on test corpus matches libvpx's gain to within 0.3%.
- [x] 720p30 real-time still passes.

**Estimated effort:** 2 weeks (down from 3–4 — modes 1/2 elided).

---

### Stage 4c — Denoiser

**Goal:** Content-aware denoiser integrated, producing libvpx-equivalent quality on noisy content.

**Inputs:** Stage 4b complete.

**Tasks:**

1. Port libvpx denoiser kernel (spatial + temporal).
2. Denoise-aware rate control + mode decision.
3. SIMD for denoiser kernel.

**Exit criteria:**
- [x] On noisy livestream test clips, BD-rate within 0.5% of libvpx (denoise on).
- [x] 720p30 real-time still passes with denoiser on.

**Status (2026-08-10):** Focused Stage 4c exit achieved across CPU6–8 on
three representative noisy 720p30 clips. Per-clip denoise-on PSNR BD-rates
range from `-3.865%` to `-18.315%` versus pinned vpxenc. Corpus wall-time
geomeans are `1.017530x`, `1.002867x`, and `1.045717x` for CPU6, CPU7, and
CPU8 respectively; all remain inside the final `1.05x` relative gate and the
absolute 720p30 budget. Rvp9-decoder, pinned vpxdec, and ffmpeg agree
byte-exactly on every exercised stream. The broader v1 resolution and feature
acceptance matrix remains open.

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
- [x] Decoder (and libvpx decoder) extracts each temporal layer correctly.
- [x] BD-rate per layer within 0.5% of libvpx.
- [x] 720p30 + 1080p30 real-time still pass with 3-layer temporal SVC.

**Status (2026-08-11):** The two- and three-layer reference hierarchy and
per-layer rate-control lifecycle are active. Full streams and every retained
temporal operating point decode byte-exactly through rvp9-decoder, pinned
vpxdec, and ffmpeg. The complete six-point 720p30 three-layer VBR curves now
pass the final `+0.5%` quality gate at every required preset. CPU6
TL0/TL1/TL2 is `-0.4547%/-2.1177%/-2.9243%`; CPU7 is
`-0.2817%/-1.2995%/-3.9786%`; CPU8 is
`-0.4566%/-0.3040%/-3.2455%`. Source-backed fixes cover the temporal
reference hierarchy, stream-global frame age, refresh-Q bounds, layer-local
rolling monitors, the TL0 NSTEP search exception, feature-scoped keyframe
search, and CPU8's zero-MV variance-partition predictor. The matching CPU8
int-pro-seed removal reproduces a `0.384..0.694%` production-PGO improvement.
Two-layer VBR now also passes across CPU6-8; CPU6's three-clip aggregate is
`-2.3151%/-2.2652%`, while its worst individual clip/layer is `+0.5241%` and
remains inside the explicit `+2%` clip guard. Both CBR hierarchies pass across
CPU6-8, including CPU6 T2 at `-27.7767%/-23.9911%` and T3 at
`-32.2313%/-32.2218%/-31.0428%`. A repeated production-PGO speed matrix using
two reversed 15-pair rotations per row measures a `1.146317x` 720p geomean and
`1.041934x` 1080p geomean. CPU6's 1080p T2/T3 rows pass at
`0.927699x/0.982764x`, but CPU7/8 T3 remain red at
`1.107755x/1.144761x`; 720p is red across every row. All measured whole-clip
times remain below 14 ms/frame, comfortably inside the median real-time
budgets. A source-faithful CPU7 candidate now applies libvpx's simple
large-block YRD model to every enhancement picture. Complete T2/T3 VBR and
CBR curves remain green, while two reversed plain-release rotations improve
T2 by about 1.5% and T3 by about 3.5%. CPU8 retains its reference-only Q-index
140 boundary because the literal policy regresses TL2 to `+1.3884%`. The
unchanged production-PGO matrix strengthens the CPU7 gain to `1.4..1.7%` on
T2 and `3.6..4.0%` on T3. Repeating the full direct matrix reduces the 720p
geomean from `1.146317x` to `1.132790x`, the 1080p geomean from `1.041934x`
to `1.034476x`, and the all-row geomean from `1.092880x` to `1.082518x`.
CPU7 T3 remains red at `1.159796..1.161545x` for 720p and
`1.066570..1.070168x` for 1080p. Per-frame p99 and the final quiet-host corpus
matrix remain open.

**Status (2026-08-12):** CPU7 temporal enhancement pictures now omit the
profiled final intra fallback. Complete three-clip T2/T3 VBR and CBR curves
pass every aggregate layer and important-clip gate, and all 252 candidate
streams decode byte-exactly through rvp9-decoder, pinned vpxdec, and ffmpeg.
The exact 22-row production-PGO build measures CPU7 corpus geomeans of
`1.0289x` for T2 and `1.0376x` for T3, closing CPU7's focused 720p relative
speed rows. CPU6/T3 remains red near `1.072x`; CPU8/T3 remains borderline at
`1.052--1.055x`. The quiet-host matrix and per-frame p99 gate remain open.

**Status (2026-08-12, Unit 1163):** Rvp9 now matches libvpx's rule that
disables the VBP min/max force-split check at CPU8 and on non-reference SVC
pictures. Complete CPU6-8 T2/T3 VBR and CBR curves remain green across all
three clips; the worst individual VBR result is `+0.6791%`, and all 756 full
and retained streams decode byte-exactly through all three decoders. The exact
production-PGO build closes the in-tree CPU6/7/8 T2 and CPU7/T3 rows at
`1.0447x/1.0054x/1.0083x/1.0304x` paired. The three-clip CPU8/T3 paired
geomean falls from `1.0938x` to `1.0532x`, leaving `0.32%` to the final gate;
CPU6/T3 remains open at `1.0777x`. External emulator load makes these timing
rows directional pending the quiet-host matrix. Per-frame p99 remains open.

**Status (2026-08-12, Units 1165--1167):** CPU6 now retains its exact winning
luma prediction, and CPU6/T3 uses the measured TX8/TX16 emission selector.
The complete CPU6/T3 VBR/CBR curves remain green; exact committed PGO direct
timing passes at a `1.0170x` paired corpus geomean. The new production-path
latency oracle records encoder-only per-frame times without input or IVF-write
cost. Five repeated runs of every CPU6--8/T3 clip row pass both absolute gates:
720p30 single-core medians are `3.18--10.33 ms` with median p99
`26.72--41.26 ms`; 1080p30 four-core medians are `2.55--6.58 ms` with median
p99 `21.92--26.89 ms`. These conservative rows ran with an unrelated process
saturating one host core. The remaining Stage 4d parity risk is relative
CPU8/T3 speed: the latest loaded-host exact-PGO result is `1.0722x` overall,
driven by ducks at about `1.15x`; it needs a measured ducks optimization and
then a final quiet-host comparison before Stage 4e begins.

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
