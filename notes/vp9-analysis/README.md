# Stage 1 — Agent-team deep-dive (dispatch plan)

This directory is the working corpus for the Stage 1 deep-dive on libvpx's
VP9 real-time encoder + decoder. See `PLAN.md` §6 / Stage 1 for the gating
criteria.

**This README is the dispatch plan. Do not start Stage 1 until Stage 0's
exit criteria are met and Arthur has signed off on this plan.**

## Agents to dispatch

Six specialists in parallel, then the devil's advocate, then the synthesizer.
Each specialist's prompt must include:

1. A pointer to `third_party/libvpx/` at the pinned commit.
2. A pointer to `PLAN.md` so they know the gating constraints (real-time,
   cpu-used 6–8, full SVC, denoiser+AQ, libvpx-mirrored threading, mirrored
   integer widths, BD-rate parity).
3. Required output: `notes/vp9-analysis/<filename>.md` with the structure
   below.

### Required output structure (per specialist)

```
# <Topic> in libvpx VP9

## 1. Scope
What this doc covers and what it does not.

## 2. Data structures
For each load-bearing struct/typedef:
- Verbatim definition (with file/line citation into third_party/libvpx/)
- Field-by-field semantics
- Ownership / lifetime / aliasing notes

## 3. Control flow
Call graphs, sequence diagrams, state machines as appropriate.
File/line citations on every claim.

## 4. Algorithms
Step-by-step description of each algorithm in this domain. Use pseudocode
where useful; cite libvpx file/line.

## 5. Tables and constants
**Verbatim** transcription of every load-bearing table (probability,
quantizer, scan order, speed-feature dispatch, ...). Citations required.

## 6. RTCD / SIMD surfaces
Identify every kernel in this domain that has SIMD implementations in
libvpx. For each: function signature, ISAs covered, expected complexity
class for our Rust port.

## 7. Open questions for LLD
Things this specialist needs the LLD writer to resolve.
```

### Specialists

| # | File | Topic |
|---|---|---|
| A1 | `01-architecture.md` | Top-level encoder lifecycle, frame loop, threading model overview, `VP9_COMP` god-struct, init/encode/teardown sequence, frame-dependency graph |
| A2 | `02-encoder-core.md` | Real-time mode-decision (non-RD path in `vp9_pickmode.c`), partition shortcuts, skip detection, intra/inter selection for cpu-used 5–8 |
| A3 | `03-motion-estimation.md` | Real-time ME (`vp9_nonrd_pick_inter_mode_sb`), sub-pixel refinement, MV prediction, reference-frame selection in non-RD mode |
| A4 | `04-bitstream-entropy.md` | VP9 bitstream format, arithmetic coder, **all** probability tables verbatim, coefficient model, header layout |
| A5 | `05-rate-control.md` | 1-pass CBR/VBR, buffer model, frame dropping, denoiser, AQ modes, content-aware QP. **Biggest area — budget extra reading time.** |
| A6 | `06-threading-svc.md` | Tile-based pthread model, frame-parallel encode, tile-row barriers, SVC layer structure + reference dependencies, `set_rt_speed_feature_*` dispatch table |

### Cross-cutting

| # | File | Topic |
|---|---|---|
| A7 | `07-devils-advocate.md` | Reads A1–A6. Writes ≥20 concrete challenges with file/line citations. |
| A8 | `00-overview.md` | Reads A1–A7. Cross-cutting synthesis that resolves the devil's-advocate challenges (fix in A1–A6 or defer to Stage 2 with rationale). |

### Surfaced for the LLD writer

| # | File | Topic |
|---|---|---|
| — | `decisions-needed.md` | Every "we'll need to choose X in the LLDs" question that surfaced during Stage 1. Resolved in Stage 2. |

## Dispatch ordering

1. Run A1–A6 in parallel (six concurrent agents).
2. Wait for all to complete and pass an Arthur-led review pass.
3. Run A7 (devil's advocate).
4. Run A8 (synthesizer), which may loop back to A1–A6 for targeted re-dispatch.
5. Compile `decisions-needed.md` from A1–A8.
6. Hand off to Stage 2.

## Quality bar

A specialist's output is accepted when:

- [ ] Every table transcribed verbatim, not paraphrased.
- [ ] Every algorithmic claim cited to `third_party/libvpx/<file>:<line>`.
- [ ] Devil's advocate cannot identify a load-bearing structure or table the
      specialist missed.
- [ ] Arthur signs off.

Re-dispatch with targeted scope if any bar is missed; do not move to the
next stage.
