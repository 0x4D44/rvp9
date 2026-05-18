# Overnight 2026-05-17 → 2026-05-18 onward — Stage 3 ledger

**Owner:** Arthur. Lead: Claude (supervisor / coordinator).
**Started:** 2026-05-17.
**Goal (2026-05-17, overnight):** First keyframe bit-exact under vpxdec MD5 (floor) → all keyframes bit-exact (stretch).
**🎯 FLOOR HIT 2026-05-17:** `vp90-2-02-size-08x08.webm` frame 1 matches `2f90d606edc511c8c960530dd915cb98` ✓.
**Goal (2026-05-18 onward, Arthur-approved):** **Full Stage 3 exit** — 100% pass on official VP9 conformance test vectors (all 201 frames).
**Wedge policy:** skip + flag, never cascade.

The 2026-05-17 overnight run hit the floor + landed 19 commits. This
ledger now continues across additional sessions toward Stage 3 exit;
see Phase B/C/D/E/F sections appended at the bottom as each phase begins.

## Architecture

Supervisor pattern. Lead holds this ledger + dispatches per-sub-task
agents. Agents run in their own context, return ≤500-word summary.
Lead reviews, runs `cargo test`, commits, updates ledger, moves on.

## Wedge policy (Skip + flag)

If a sub-task burns >3 agent attempts without measurable progress OR
returns failing tests that can't be diagnosed from the agent's summary:

1. Mark sub-task as `WEDGED` in this ledger with the symptom + last
   attempt commit hash (if any).
2. Insert a `// TODO(overnight-2026-05-17): <reason>` in the code at
   the wedge point.
3. Move to the next independent sub-task.
4. Continue. Do NOT cascade a wedge — downstream tasks that need the
   wedged piece get `WEDGED-BLOCKED` and the loop moves on to
   independent work.

Arthur triages all `WEDGED` and `WEDGED-BLOCKED` entries on wake.

## Sub-task wave structure

### Wave 1 — parallel (all independent)

| ID | Sub-task | LLD anchor | libvpx ref | Status |
|---|---|---|---|---|
| A | 8.2.5 read_tx_size + tx_size context | LLD bitstream §8 / mode-decision | vp9_decodemv.c:read_tx_size | ✅ `02c5329` |
| B | 6.3.4 setup_mask partition-tree walker | LLD loopfilter §6 | vp9_loopfilter.c:vp9_setup_mask | ✅ `d2e021c` |
| C | vpxdec MD5 conformance harness through rvp9-decoder | n/a (Stage 0 stub at tests/conformance/runner.rs) | n/a | ✅ `1f2905a` |

### Wave 2 — gated on Wave 1.A (D dispatched parallel with B2)

| ID | Sub-task | Status |
|---|---|---|
| D | 8.2.6 neighbour-context plumbing (above/left ModeInfo grid) | ✅ `3b907d2` |
| B2 | 6.3.5 adjust_mask post-pass | ✅ `68777f6` |
| E | 8.2.7 partial MV-component decoder (inter leaves stub `unimplemented!()`) | ⏬ deferred during overnight; resurrected as Phase B.1 for Stage 3 exit drive (see below) |

### Wave 3 — gated on A+D (split F preemptively to reduce wedge risk)

| ID | Sub-task | Status |
|---|---|---|
| F.A | Coef-context arrays (above/left per-plane u8 flags + combine_entropy_contexts helper) | ✅ `898a087` |
| F.B | `decode_coefs` core (scan walk + EOB tree + coef value reading) | ✅ `52736be` |
| F.C | `decode_block_tokens` wrapper (per-TX-block: pt-compute + decode_coefs + writeback) | ✅ `e41ca55` |
| G | Dequant glue (per-block Quants → Dequants pipeline) | ⏬ MERGED into K — F.B already does inline dequant; per-block (dc,ac) computation is just LUT lookup, folds into tile decode loop |
| H | Intra-block reconstruction (`reconstruct_intra_block` — intra_pred + IT-add) | ✅ `475e876` |
| I | IT block-level dispatch | ⏬ MERGED into H — IT-add is consumed only by recon, so H wraps both |

### Wave 4 — gated on B+B2+H (K split per H's findings)

| ID | Sub-task | Status |
|---|---|---|
| B2 | 6.3.5 `adjust_mask` post-pass | ✅ `68777f6` |
| J | Frame-level loopfilter apply driver | ✅ `1686783` |
| K1 | Intra-pred edge-builder (`build_intra_predictors` — the 127/129 + above-right-replicate machinery) | ✅ `8662546` |
| K2 | Decoder integration: per-coding-block tile decode loop (iterates TX positions, builds PredCtx via K1, calls H) | 🟡 in-flight (staged delivery: K2.a partition walker → K2.b per-block decode → K2.c LF wireup → K2.d MD5 smoke test) |

### Wave 5 — gated on C+K

| ID | Sub-task | Status |
|---|---|---|
| L | Run conformance vectors, assert MD5 match on first keyframe — FLOOR | 🎯 **HIT** — `3364c60` + `3ba4741`. `vp90-2-02-size-08x08.webm` frame 1 MD5-matches libvpx exactly. 1 passed / 10 mismatched / 190 skipped / 201 total. |
| M.1 | Differential-trace 32×32 keyframe mismatch (smallest delta from passing 8×8) | ✅ `27f2f3c` + ratchet 10→9 (1 bug found+fixed; cascade bug surfaced for M.2) |
| M.2 | Hand-trace DctAdst 4×4 H-mode path — bisect decode_coefs vs iadst4 vs dq_shift | ✅ partial `8707f6a` — 2 of 3 suspects RULED OUT (decode_coefs col/row_scan dispatch fine; iadst4 DC-only numerics fine; dq_shift not in play for TX_4×4). Remaining: iht4x4 multi-coef OR dq_coefs production mismatch. Closed pending follow-up. |
| M.3 | iht4x4 multi-coef oracle (Phase 1) + dq_coefs instrumentation (Phase 2) | ✅ `36c1a04` + `287b4b1` — iht4x4 CLEARED (N=100 random-input differential vs verbatim libvpx port, no divergence). Bug LOCALIZED to dq_coefs production (decode_coefs or upstream). M.3 agent's line-by-line review of decode_coefs vs libvpx found no source-level divergence. |
| M.4 | decode_coefs verbatim port + N=100 differential; if clean, audit call-site (initial_pt, writeback timing, scan, probs) | ✅ partial — Phase 1+2 CLEARED decode_coefs (N=100 random across 10 (TxSize × scan) configs vs verbatim libvpx port — no divergence). Phase 4 call-site audit on `vp90-2-02-size-32x32.webm` frame 1 sub-1: `initial_pt=1, tx_type=DctAdst, scan=col_scan_4x4 ([0,4,8,1,...]), dequant=(177,231), probs[b0][ctx1]=[84,49,136]` — all match libvpx defaults EXACTLY. **decode_coefs is bit-exact AND its inputs are correct.** Remaining hypothesis: bit-stream cursor diverges BEFORE sub-1 due to a prior mis-counted read OR an interaction with the boolean coder we haven't characterised. Lead: dispatch focused follow-up to diff vpxdec's per-block bool-reader state against ours. |
| N.1 | Parser-rejection investigation (15 frames: 11× first_partition_size=0, 4× log2_tile_cols=7) | ✅ `78df111` — single root cause (bit-stream misalignment in `setup_frame_size_with_refs`: loop ate all 3 size_in_found bits; libvpx breaks on first 1-bit). Same bug, two symptoms. 14 frames moved from parser-reject to unimplemented-inter (165→179). Conformance headline unchanged — no parser-rejected frames were keyframes. |
| M.4 | Extend to all keyframes across all 14 vectors — full stretch | ⏸ blocked on M.3 findings |

## Run log

- **00:00** — ledger + journal + goal stub written.
- **00:01** — Wave 1 dispatched: A (8.2.5 read_tx_size), B (6.3.4 setup_mask), C (MD5 harness) — all background.
- **A returned** — commit `02c5329`. 10 tests added; 369→377 rvp9-core passing. `mode_info.rs` +498 lines. Clippy clean for touched file. Agent self-flagged a "subtle divergence" at frame edges between `Option`-shape `get_tx_size_context` and libvpx's zeroed-dummy-border default. **Lead investigation:** libvpx `vp9_pred_common.h:156-171` uses `up_available`/`left_available` flags, not dummy-border reads — the `(None, None)` arm returning `max_tx_size` for both sides matches libvpx exactly. No real divergence; agent's flag was a false alarm. NO ledger entry needed. Wave 2 D should pass `None` at frame edges (NOT `Some((Tx4x4, false))` as the agent's TODO suggests).
- **B returned** — commit `d2e021c`. 8 tests added; 369→377 rvp9-core passing (sum with A). `setup_mask.rs` 733 lines new. Clippy clean for `rvp9-core`. Deliberate scope cut: `vp9_adjust_mask` (frame-edge TX_32X32→TX_16X16 merge, 4×4-border promotion, left-edge clear) deferred to Task 6.3.5 per LLD §5.3/§5.4 split. **Lead note:** acceptable — adjust_mask is a separable post-pass; adding 6.3.5 to the ledger as a new sub-task between J and current 6.3.4. Frame-level LF apply (J) needs adjust_mask done first.
- **C returned** — commit `1f2905a`. Conformance harness wired through `rvp9-decoder::Decoder::decode_frame`. Initial baseline: 0 passed / 0 mismatched / 201 skipped / 201 total. 13 harness lib tests + 1 integration test (walks all 14 vectors). New `Decoder::decode_frame(&ParsedFrame) -> Result<FrameBuf, DecodeError>` with `DecodeError::Unimplemented { reason }`. Routes ShowExistingFrame → existing helper, everything else → Unimplemented for now. **Discovered:** 11+4 frames are being rejected by the uncompressed-header parser (`first_partition_size: 0`, `log2_tile_cols: 7`) — these are pre-existing bugs / spec edge cases NOT introduced by this work. Flagged for follow-up. **Pre-existing doctest failure** on `Decoder::parse_compressed_partition` (commit `20bc60d`, NOT mine) — CI gate uses `--lib --bins --tests` to avoid this. Workspace tests now: 418 passing, clippy clean.
- **Wave 1 COMPLETE.** All three agents returned green. 418/418 tests, clippy clean.
- **00:24** — Wave 2 dispatched: D (8.2.6 neighbour-context plumbing, in rvp9-decoder) + B2 (6.3.5 adjust_mask, in rvp9-core/loopfilter) — different crates, no overlap, both background. E (MV-component partial) removed from critical path (was Wave 2; now DEFERRED — F doesn't actually depend on it for keyframe-only).
- **D returned** — commit `3b907d2`. New `TileDecodeCtx` in `crates/rvp9-decoder/src/tile_ctx.rs` (458 lines, 14 tests). Row+col shape (not full 2-D grid). Accessors `above(mi_col) -> Option<&ModeInfo>`, `left(mi_row_in_sb) -> Option<&ModeInfo>`, `reset_left()`, `write_block(...)`. **Key findings for downstream:** (1) NW neighbour NOT preserved — fine for keyframe-only intra, but inter blocks (later) will need full 2-D grid. (2) `mi_col` is tile-relative; caller (Wave 3 K) must subtract `tile.mi_col_start` before indexing. (3) Token coef decode (F) needs its OWN above/left flag arrays per plane — that's exactly the F.A split below. Clean. 418→433+ tests.
- **B2 returned** — commit `68777f6`. `adjust_mask` post-pass in `crates/rvp9-core/src/loopfilter/adjust_mask.rs` (567 lines, 9 tests). Bit-twiddling simplifications noted (`& ~(x & 0xcccc)` → `& ~0xcccc` etc.); semantic match to libvpx. No LLD §7 blockers. rvp9-core 377→386.
- **Wave 2 COMPLETE.** Critical path advanced; F.A queued.
- **00:32** — Wave 3 F.A dispatched (coef-context arrays infra) — single agent, background. Touches `crates/rvp9-decoder/src/tile_ctx.rs` (extending D's work) — no concurrent overlap since B2 is done. F preemptively split into F.A (infra) / F.B (decode_coefs core) / F.C (per-plane driver) to bound wedge risk.
- **F.A returned** — commit `898a087`. Sibling struct `TileCoefCtx` in `tile_ctx.rs` (decoupled from `TileDecodeCtx`'s ModeInfo plumbing because the two have different reset schedules and storage shapes — mirrors libvpx's separate `above_context` / `mi_grid_visible` allocations). +9 tests; 442→451. Clippy clean. **`combine_entropy_contexts(a, l) = (a!=0) + (l!=0)` → 0..=2** pinned. For tx>4×4, F.B should OR-fold across the spanned 4×4 cells. ctx mutates ONLY mid-loop via nb+token_cache; above/left arrays only touched at block entry + exit. Reset: above persists across SB rows, left resets between SB rows, both reset between tiles.
- **F.B dispatched** — decode_coefs core. Single agent, background. Most wedge-prone sub-task of the night. Strict 3-attempt cap. Wedge falls back to "commit WIP if compiles+clippy-clean, else git restore + report symptom + lead splits further".
- **F.B returned** — commit `52736be`. **0/3 wedge attempts consumed.** 13 hand-traced tests green incl. CAT6 (14 extra bits), mid-loop ctx mutation (load-bearing test), run-to-seg_eob via zero-run. 977-line `crates/rvp9-core/src/entropy/detokenize.rs`. Clean. Workspace 451→464.
- **Lead decisions on the basis of F.B's interface:**
  - F.B's inline dequant means **G shrinks to a per-block (dc,ac) LUT lookup** — that's trivially folded into the tile decode loop (K). G removed as a separate sub-task.
  - F.B is a pure entropy primitive — it does NOT writeback to TileCoefCtx. That makes F.C (the wrapper) and H (recon) cleanly separable.
  - **I (IT-block dispatch) merged into H** — IT-add is consumed only by recon, so H wraps `intra_pred + inverse_transform_add` in one function.
  - Critical path is now: F.C → H → K → L (+ J = LF apply in parallel). Just 4 sub-tasks for the floor.
- **F.C dispatched** — per-TX-block wrapper. Small (~50-100 lines). Background.
- **F.C returned** — commit `e41ca55`. 9 hand-traced tests. New module `crates/rvp9-decoder/src/detokenize.rs`. Workspace 464→473; clippy clean. **Key for H:** dq_coefs are in **raster order** (already reordered by F.B), eob is scan-order fast-zero hint. Intra-pred kernels write directly into destination plane buffer; H orchestrates predict → decode_block_tokens → (if eob>0) inverse_transform_add. No FrameCounts type yet — pass None.
- **H dispatched** — `reconstruct_intra_block` (intra-pred + IT-add for one TX block, mirrors libvpx `predict_and_reconstruct_intra_block` at vp9_decodeframe.c:307-336). Background. ~150-200 lines expected.
- **H returned** — commit `475e876`. 7 hand-traced tests (zero-residual + DC-only differential across DC/V/H/TM modes × Tx4x4/Tx16x16/Tx32x32 sizes + UV plane). 0/3 wedges. 731-line `crates/rvp9-decoder/src/reconstruct.rs`. Workspace 473→480; clippy clean. **Critical finding:** intra-pred edge-builder (the 127/129 unavailable-pixel + above-right-replicate + above-left machinery — libvpx `build_intra_predictors` in `vp9_reconintra.c`) is NOT yet implemented. H assumed the caller materialised `PredCtx`. Need K1 (edge-builder) before K2 (tile decode loop). Also flagged: lossless (WHT) path debug-asserts; sparse fast paths (idct8x8_12_add etc.) deferred — bit-exact via full kernels just slower.
- **K split into K1 + K2:**
  - K1 = `build_intra_predictors` mirror in `crates/rvp9-core/src/intra_pred/`. Self-contained — pure function from (dst_plane, pos, tx_size, mode, availability) → PredCtx.
  - K2 = the per-coding-block + per-plane + per-TX-position iteration that calls K1 then H. Big integration.
- **J unblocked** — B + B2 + H all done. J is the frame-level LF apply driver in `crates/rvp9-core/src/loopfilter/`. Doesn't depend on K2 — runs after K2 produces a reconstructed frame.
- **Wave 4 dispatched in parallel:** K1 (intra_pred edge-builder) + J (frame LF apply). Different submodules, no overlap, both background.
- **K1 returned** — commit `8662546`. `build_intra_predictors` in `crates/rvp9-core/src/intra_pred.rs` + new `EdgeBuffers` (repr-C 16-byte-aligned scratch, mirrors libvpx's DECLARE_ALIGNED). PredCtx shape preserved. 11 hand-traced tests; 480→491; clippy clean. **For K2:** `have_top = loff != 0 || above_mi != NULL`; `have_left = aoff != 0 || left_mi != NULL`; `have_right = (aoff + txw) < bw` (coding-block width, NOT frame); frame-edge clamping is internal to `build_intra_predictors` via `plane_width`/`plane_height`.
- **K2 dispatched** — the biggest integration task of the night. Staged delivery: K2.a partition walker scaffold → K2.b per-block decode pipeline → K2.c LF wireup (if J landed) → K2.d MD5 conformance smoke. Each stage commits independently so wedges don't kill prior progress. Background.
- **J returned** — commit `1686783`. `loop_filter_frame(yv12, lfm_grid, ..., info, start_mi_row, end_mi_row)`. Pass order V-then-H per SB (libvpx `vp9_loopfilter.c:1239-1322`). Plane order Y → U → V. 8 hand-traced tests; clippy clean. **K2.c precondition met:** J is now on main; if K2 reaches K2.c it can wire in `loop_filter_frame` after the per-coding-block decode pass. Skip-when-zero is **caller's** responsibility (`if frame_filter_level == 0 { return; }`); driver is pure given-the-masks-apply-them. Chroma 4×4 mask cells span **8 chroma pixels** each (not 4) — covers full chroma SB area at 4:2:0.
- **K2 RETURNED — FLOOR HIT 🎯** — Three commits landed clean:
  - `6685d7c` K2.a partition+tile walker scaffold (~600 lines)
  - `3e576fb` K2.b per-block decode pipeline (predict + decode + recon via K1+H)
  - `3364c60` K2.d conformance smoke test
  - K2.c (LF wireup) **deferred** — needs a frame-wide 2-D `mi8x8: &[ModeInfo]` grid that today's `TileDecodeCtx` only stores as a row+column reduction. That's a larger refactor than one sub-task. Code is annotated `TODO(overnight-2026-05-17)` at the wireup site.
  - **First MD5 match: `vp90-2-02-size-08x08.webm` frame 1** (`2f90d606edc511c8c960530dd915cb98` ✓).
  - 1 passed / 10 mismatched / 190 skipped / 201 total. 0/3 wedges per stage.
- **Lead follow-up commit `3ba4741`** — added a lossless-frame guard in `Decoder::decode_keyframe` (H's debug_assert was panicking in debug mode on lossless frames; release silently mismatched). Ratcheted the conformance integration test to `MIN_PASSES=1, MAX_MISMATCHES=10` (Stage 3 exit gate is still `0 mismatched`; ratchet is for regression signal during stretch iteration). Workspace 506+ tests passing, clippy clean.
- **Stretch dispatched** — M.1 = differential trace on `vp90-2-02-size-32x32.webm` frame 1 mismatch. Smallest delta from the passing 8×8 case → highest yield per agent-hour. Exercises TX_32X32 IT + intra-pred + TX_16X16 UV that 8×8 doesn't.
- **M.2 partial-landing** — added oracle unit tests bisecting the three hypotheses from M.1's localization:
  1. `decode_coefs` with `col_scan_4x4` / `row_scan_4x4` → both **pass** (rules out the non-default-scan decode hypothesis for the simple ONE_TOKEN+EOB case).
  2. `iht4x4_16_add(DctAdst, ...)` and `(AdstDct, ...)` with `dq_coefs = [64, 0, ..., 0]` → both pass against fully hand-traced expected output (rules out a simple DC-only iadst4/idct4 dispatch bug).
  3. Decoder-level instrumentation (the actual H-mode 4×4 dq_coefs trace M.1 asked for) **blocked** — `crates/rvp9-decoder` does not build right now due to concurrent in-flight refactor of `Decoder::ref_frame_map → frame_pool` (header.rs / lib.rs uncommitted changes by another agent). Cannot run the conformance binary to capture the failing-block trace until that lands.
  - Drive-by fix: corrected the `col_scan_4x4` / `row_scan_4x4` doc-comments — they swapped which TxType they're used for. Dispatch behaviour was already correct; only the doc-strings were wrong.
  - Remaining bug ranking after this bisect (highest-yield first): (a) iht4x4 with **multi-coef** dq_coefs inputs (DC-only DctAdst trace is correct but a non-DC + non-zero pattern hasn't been oracled), (b) the actual `dq_coefs` array passed to iht4x4 differs from libvpx's (requires decoder-level instrumentation to confirm — blocked until rvp9-decoder builds again), (c) `dq_shift` is `Tx32x32`-only so not in play for 4×4.
- **M.1 returned** — commit `27f2f3c`. **Root cause: sub-8 luma per-block iteration count.** In `tile.rs::decode_block`, the per-TX iteration extent used `plane_bsize.n_4x4_*()` for all sizes; for `bsize < BLOCK_8X8` this returned 1 instead of libvpx's `n4_w = (1 << 1) >> ss_x = 2` for luma. Effect: only 1 of 4 sub-blocks of each Block4x4 was decoded; the rest stayed at zero, also corrupting bool-reader alignment. Fix: special-case `bsize < BLOCK_8X8` to use `(2 >> ss_x, 2 >> ss_y)`. Conformance: 1 passed / 10 → **9** mismatched / 190 → 191 skipped (ratchet tightened to MAX_MISMATCHES=9). Floor 8×8 still bit-exact.
  - **Cascade bug surfaced:** sub-8 luma sub-blocks with non-DCT_DCT TX type still produce wrong residuals. Concretely on the 32×32 vector's first Block4x4 with `bmi=[Dc, H, Dc, Dc]`: sub-1 (H → DctAdst → col_scan) decodes wrong residual `[+0f, +09, -10, +03]` vs libvpx `[+02, +02, -0e, -06]`. Sub-0 / 2 (DCT_DCT default_scan) are bit-exact. Sub-3 propagates sub-1's error.
  - **Suspects ranked:** (1) iadst4 sign/rounding; (2) decode_coefs mid-loop ctx with non-monotonic scan; (3) dq_shift timing.
  - **Bonus drive-by:** doc-comments on `col_scan_4x4()` and `row_scan_4x4()` in `scans.rs:137-156` are swapped (dispatch behaviour correct). Lead can fix this drive-by later.
- **M.2 dispatched** — bisect via hand-traced unit tests: validate decode_coefs with col_scan; then validate iht4x4 on synthetic input; then localize on the actual failing case. Background.

---
