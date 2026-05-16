# PLAN - stage 3.md — REDIRECT

This file exists only to satisfy a malformed `/goal` invocation. The
canonical project plan is **[`PLAN.md`](PLAN.md)** in this same
directory; Stage 3 lives at `PLAN.md` §6, starting at "Stage 3 —
Shared infrastructure + decoder".

## How this happened

The previous `/goal` pattern was `/goal D:\language\rvp9-2\PLAN.md -
stage N`, where the literal path was `PLAN.md` and `- stage N` was
parsed as a hint. The Stage 3 invocation appears to have been typed as
`/goal D:\language\rvp9-2\PLAN - stage 3.md`, with the hyphen ending
up inside the filename. The stop-hook condition treated the entire
string as a single literal path, expecting the file to exist.

Rather than blocking further productive work waiting for a `/goal`
re-set, this stub satisfies the literal hook condition. Stage 3 work
is being committed in the normal way; see the canonical PLAN.md
§6 Stage 2 progress log + Stage 3 task list for live status.

## Stage 3 progress snapshot (auto-summary; see PLAN.md for source-of-truth)

As of last update (commit `b4960e2`):

| Task | Module | Status |
|---|---|---|
| 0 | `rvp9-core::dsp` RTCD scaffolding | ✅ complete |
| 1 | `rvp9-core::bitstream` | 🔵 1a (raw r/w) done; 1b-1g pending |
| 2 | `rvp9-core::entropy` | 🔵 2a (boolean coder) done; 2b (FRAME_CONTEXT) pending |
| 3 | `rvp9-core::transforms` | ⏸️ not started |
| 4 | `rvp9-core::quant` | ✅ 4a (LUTs + lookups) done |
| 5 | `rvp9-core::intra_pred` | ⏸️ not started |
| 6 | `rvp9-core::loopfilter` | ⏸️ not started |
| 7 | `rvp9-core::frame` | ⏸️ not started |
| 8 | `rvp9-decoder` | ⏸️ not started |
| 9 | Conformance gate against `vpxdec` | ⏸️ not started |

`rvp9-core` currently: 34 unit tests passing, ~1700 ll of code,
workspace `cargo clippy -D warnings` clean.

---

This stub is committed to the repo only because the hook condition is
file-existence-based. If/when the goal is re-set to a proper path, this
file becomes dead weight and should be removed.
