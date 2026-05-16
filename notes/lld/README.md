# LLD index — Stage 2 deliverables

15 LLDs covering every Rust implementation module of rvp9-2's encoder
and decoder. Each follows the §1-§9 template documented in
`notes/vp9-analysis/_stage2-dispatch-prep.md` "LLD template". Generated
in Phase 2c (commits between `c7f9e9d` and `b90b1a7`).

Recommended review order: top down — shared infrastructure first, then
encoder pipeline, then cross-cutting concerns. Phase 2d gates on
Arthur's sign-off across all 15.

## Shared infrastructure (Stage 3 implementation targets)

| LLD | Lines | Specialist | Focus |
|---|---|---|---|
| [`bitstream.md`](bitstream.md) | 1608 | A4 | uncompressed/compressed headers, tile-data, superframe index, no-altref simplification |
| [`entropy.md`](entropy.md) | 1555 | A4 | boolean coder, FRAME_CONTEXT, cost framework (incl. `vp9_prob_cost[256]`), bit-exact contract |
| [`transforms.md`](transforms.md) | 1544 | A4+A2 | DCT/ADST 4-32, scan orders, trellis-quant gate-off, lossless decoder-only |
| [`quant.md`](quant.md) | 1378 | A5+A4 | DC/AC quant LUTs, `vp9_compute_qdelta_by_rate`, per-segment quant deltas |
| [`intra-pred.md`](intra-pred.md) | 1278 | A2 | 10 intra modes × 4 block sizes, edge replication, FFI→intrinsics path |
| [`loopfilter.md`](loopfilter.md) | 1499 | A1 + libvpx-source | edge-strength × filter-strength × mask machinery, per-frame RDO bypass at non-RD |
| [`frame-buffers.md`](frame-buffers.md) | 1534 | A1+A6 | BufferPool 8+1+3 slots, Arc-pool, scaled slots hidden from public API |

## Encoder pipeline (Stage 4 implementation targets)

| LLD | Lines | Specialist | Focus |
|---|---|---|---|
| [`motion-estimation.md`](motion-estimation.md) | 1836 | A3 | diamond/full-pel/sub-pel, ref masking, NEWMV bias, compound search stub |
| [`mode-decision.md`](mode-decision.md) | 2369 | A2+A3+A4 | non-RD pick-mode ≤16-iter loop, Laplacian R-D, EXT_RATECTRL input |
| [`partition.md`](partition.md) | 2503 | A2+A3 | variance-tree, Morton-order `variance_low[25]`, vbp_thresholds, coord_lookup |
| [`rate-control.md`](rate-control.md) | 2164 | A5 | CBR + VBR-RT (NO altref), `vp9_encodedframe_overshoot`, EXT_RATECTRL trait |
| [`denoiser.md`](denoiser.md) | 1540 | A5+A2 | spatial + temporal denoise, **L-22 bit-parity divergence load-bearing** |
| [`aq.md`](aq.md) | 1460 | A5 | **CYCLIC_REFRESH only**; weight_segment math, screen-content gate |

## Cross-cutting

| LLD | Lines | Specialist | Focus |
|---|---|---|---|
| [`svc.md`](svc.md) | 2275 | A6+A5 | temporal + spatial SVC, per-layer state, cross-layer drop chain, EXT_RATECTRL per-layer hooks |
| [`tile-threading.md`](tile-threading.md) | 2895 | A6+A1 | worker pool, JobQueue, RowSync, `FrameScope<'frame>`; **most subtle unsafe in the codebase** |

## Total

- **15 LLDs**, **27,397 lines** of design content
- All structurally OK per `check-lld-format` (zero §1-§9 failures)
- 2 non-blocking TBD markers (false positives — meta-prose + cross-LLD §X placeholder)
- Aggregated §7 divergences (~50+ entries) extracted to
  `notes/divergence-ledger-phase2e-draft.md` for Phase 2e merge

## Tooling for review

Two harness binaries available to support Phase 2d review:

```sh
# Structural sweep — confirms §1-§9 present, §7 populated, TBD count
cargo run --release -p rvp9-harness --bin check-lld-format

# §7 aggregation — regenerates the Phase 2e divergence-ledger draft
cargo run --release -p rvp9-harness --bin extract-divergences \
    -- notes/lld --write notes/divergence-ledger-phase2e-draft.md
```

## Locked scope (D-01, D-05, D-13, R-06 RESOLVED)

Every LLD honours the Stage 2 entry scope decisions:
- **8-bit only in v1** (Profile 0); no HBD code paths
- **AQ mode 3 (CYCLIC_REFRESH) only**; modes 1/2 not ported
- **CBR + VBR-RT, no altref**; `lag_in_frames > 0` rejected at construction
- **EXT_RATECTRL plug-in IN v1** (libwebrtc integration)

See `notes/vp9-analysis/decisions-needed.md` "Stage 2 entry — Arthur
scope (RESOLVED)" for the full rationale.

## Cross-references

- HLDs: [`notes/hld/00-architecture.md`](../hld/00-architecture.md),
  [`01-threading.md`](../hld/01-threading.md),
  [`02-simd-dispatch.md`](../hld/02-simd-dispatch.md),
  [`04-svc.md`](../hld/04-svc.md)
- Stage 1 specialists: [`notes/vp9-analysis/`](../vp9-analysis/)
- Divergence ledger (seed): [`notes/divergence-ledger.md`](../divergence-ledger.md)
- Plan: [`PLAN.md`](../../PLAN.md) §6 Stage 2 progress log
