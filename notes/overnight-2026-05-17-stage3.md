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
| M.4 | decode_coefs verbatim port + N=100 differential; if clean, audit call-site (initial_pt, writeback timing, scan, probs) | ✅ `793103d` — Phase 1+2 CLEARED decode_coefs (N=100 random across 10 (TxSize × scan) configs vs verbatim libvpx port — no divergence). Phase 4 call-site audit on `vp90-2-02-size-32x32.webm` frame 1 sub-1: `initial_pt=1, tx_type=DctAdst, scan=col_scan_4x4, dequant=(177,231), probs[b0][ctx1]=[84,49,136]` — all match libvpx defaults EXACTLY. **decode_coefs is bit-exact AND its inputs are correct.** Bug must be bool-coder cursor drift earlier in decode_block. |
| M.5 | Bool-coder cursor bisect: instrument range/value/count at every read in decode_block; hand-trace libvpx for the failing block; suspect per-bmi `get_y_mode_probs` neighbour lookup | ✅ `c83d16e` — Full instrumentation + cursor trace. **Hypothesis FALSIFIED:** probs and decoded modes match libvpx exactly for sub-0..3 of the failing block (bmi[0]/[1]/[2] use [Dc][Dc] probs, bmi[3] uses [H][Dc]). Cursor advances cleanly across sub-block boundaries. Latent flag for downstream: `above_block_mode` / `left_block_mode` `!= 0` sentinel treats legit Dc-bmi as missing — doesn't affect this block (no neighbours) but will affect downstream sub-8 neighbours. |
| M.6 | Runtime differential: run M.4 ref port side-by-side with production at every decode_coefs call, panic on divergence | ✅ `df82a7e` — Phase 2b CONFIRMED. Production and M.4 ref port agree bit-for-bit on every decode_coefs call; both produce the same wrong output. Established "shared transcription bug" finding that M.8 ultimately localized to iht_dispatch (NOT decode_coefs). |
| M.7 | Exhaustive default-probs audit (all coef/mode/mv/kf/partition/skip/tx/inter/comp/single_ref probs) | ✅ `8b13c3e` — every byte matches libvpx. Hypothesis #2 falsified. |
| M.8 | libvpx C oracle harness (build minimal C harness, run decode_coefs + iht_add on captured snapshot) | 🎯 `d982791` — **BUG FOUND.** Root cause: `iht_{4,8,16}_dispatch` had row/col SWAPPED. libvpx's `transform_2d` struct is `{ cols, rows; }` with cols FIRST. For DCT_ADST: cols=idct, rows=iadst. We had it reversed. M.3 ref port shared the same mistake (classic shared-bug). M.2's hand-traced tests had expected outputs swapped — were validating the wrong path. Fix applied to iht_4_dispatch, iht_8_dispatch, iht_16_dispatch + M.3 ref port + M.2 tests. 32×32 MD5 shifted (different cascade now) but conformance gate unchanged. Reusable oracle infra committed to `tools/libvpx_decode_coefs_oracle/`. |
| M.9 | Iterate the oracle: capture next failing block (e.g. 16×16 keyframe), run through libvpx C, find next cascade bug | 🟢 **NEGATIVE RESULT** — drilled into the 32×32 frame-1 first-divergent Y block (col=4,row=0) post-M.8: +3-pixel diff at one corner. Probed all 13 libvpx intra-pred kernels via extended oracle — **none** produce the residual-implied target prediction. Root cause: libvpx's vpxdec output is **post-loop-filter** (filter_level=24 for this vector), ours is pre-LF. **The +3 diff IS the LF inner-filter** smoothing the 4×4 block edge. All 7 failing keyframes have `filter_level > 0` (13..54); the single passing keyframe has level=0. **The keyframe-MD5 ratchet is now wedged on Wave-4-K2.c LF wireup, NOT further M-series transcription hunts.** Deliverables: `crates/rvp9-harness/src/bin/dump_frame_diff.rs` (per-pixel diff harness), `crates/rvp9-core/src/intra_pred.rs` (RVP9_TRACE_INTRA env-gated trace), `tools/libvpx_decode_coefs_oracle/oracle.c` (13-mode predictor probe). Conformance unchanged 1/9/191/201. |
| N.1 | Parser-rejection investigation (15 frames: 11× first_partition_size=0, 4× log2_tile_cols=7) | ✅ `78df111` — single root cause (bit-stream misalignment in `setup_frame_size_with_refs`: loop ate all 3 size_in_found bits; libvpx breaks on first 1-bit). Same bug, two symptoms. 14 frames moved from parser-reject to unimplemented-inter (165→179). Conformance headline unchanged — no parser-rejected frames were keyframes. |
| M.6 | Extend to all keyframes across all 14 vectors — full stretch | ⏸ blocked on M.5 fix |
| M.17 | libvpx oracle on captured (pre-state, probs, initial_pt, dq) for resize block (px 280, 0) | ✅ NEGATIVE — M.17 oracle MATCH (eob=3, dq_coefs=[-698, 550, 0, ...]). decode_coefs + its inputs bit-exact with libvpx on the captured state. Bug surface narrowed to (a) captured pre-state ≠ libvpx pre-state (cumulative cursor drift), or (b) pre-LF col 279 differs (cascade-from-upstream), or (c) per-block input (skip / segment_id / tx_size / mode) diverges with cancelling bit consumption. Conformance unchanged 6/155/40. Instrumentation `RVP9_DEBUG_DQCOEFS=tx8` + tile-buffer dump + `RVP9_DEBUG_DUMP_WINDOW` + M.17 oracle scaffold committed for next-agent hand-off. |

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

## D.14 returned — q63 keyframe cursor drift is CUMULATIVE per-SB (NEGATIVE-with-instrumentation)

- **Target:** `vp90-2-00-quantizer-63.webm` frame 1 — 352x288 keyframe,
  876-byte tile buffer, no segmentation, no tile split. One of the four
  failing keyframes per M.17/M.18.
- **Approach:** added `RVP9_TRACE_KFSB=1` cursor-state checkpoint in
  `decode_tile_intra` that logs `(range, value, count, pos)` before
  every SB's `decode_partition`. Ran on q63 f1 and observed bool-reader
  byte position per SB.
- **Smoking-gun finding:** the bool-reader is **fully exhausted by
  mi=(16, 40)** (= SB(row=2, col=5), the rightmost SB of the 3rd SB
  row). At that point `pos=876` (= tile_data.len()) and `value=0`,
  `count=LOTS_OF_BITS` — the in-range end-of-stream state. Every SB
  thereafter (rows 2..4 cols 0..5) starts at pos=876 → all subsequent
  decode_coefs reads return 0 → DC-only-no-residual reconstruction →
  the constant-block-of-4 pre-LF pattern at row 128+ (mi_col≥40) seen
  in `dump_frame_diff RVP9_DISABLE_LF=1` cols 320..351.
- **Cursor drift is CUMULATIVE, not localized.** Per-SB byte consumption
  pattern (cumulative pos at SB-entry):
  ```
  SB row 0: 8 → 78 → 134 → 204 → 274 → 337 → 372  (avg 60 B/SB)
  SB row 1: 372 → 428 → 477 → 526 → 582 → 617 → 645  (avg 49 B/SB)
  SB row 2: 645 → 666 → 687 → 708 → 757 → 876 (EOS)  (last SB ate 119 B)
  ```
  Total budget: 876 B / 30 SBs = 29 B/SB. We're consuming 60 B/SB in
  row 0 — **double the available budget**. This is a per-SB drift, not
  a single-block bug.
- **Hypothesis confirmed by passing-keyframe characterisation.** The
  two passing keyframes (`vp90-2-02-size-08x08`, `vp90-2-02-size-32x32`)
  are **trivially small** — single SB each. **Cursor drift cannot
  manifest within a single SB (no cumulative effect).** Every failing
  keyframe (q63, sharpness-1, sharpness-7, deltaq, resize, three tile
  vectors) is 352x288 (or 1920x1080 for tiles) — multi-SB. **This
  pins the bug class to a per-SB cumulative consumer** (something
  inside `decode_block`/`reconstruct_intra_planes` that reads more bits
  than libvpx for typical Q63 high-quantizer skip-heavy blocks).
- **Top suspects (post-M.16/M.17/M.18 narrowing — kernels + decode_coefs
  cleared, so the over-consumption is in the syntax-element reads or
  the per-block token decode invocation pattern):**
  1. **Skip path over-consumption.** Q63 has very high quantizer → most
     blocks have `skip=1` → no token reads. But our `decode_block`
     ALWAYS calls `reset_skip_context` then iterates planes; if
     `mi.skip` is true we skip the `reconstruct_intra_block`
     (token-reading) call but still run `intra_predict`. The skip
     short-circuit looks correct (lines 1008-1016 in
     `reconstruct.rs`). Worth verifying: is `mi.skip` being read
     correctly? At high Q the bitstream is dominated by skip-bit
     reads, so a wrong context (`skip_ctx`) or wrong `skip_probs`
     entry would systematically over-consume.
  2. **`get_tx_size_context` neighbour disagreement.** Our gate
     `if max_tx >= TxSize::Tx8x8` is equivalent to libvpx's
     `if bsize >= BLOCK_8X8`, but the per-neighbour `(tx_size, skip)`
     pair fed in might differ in a frame-edge case. For Block8x8
     blocks (very common at high Q) `max_tx = Tx8x8` → tx_ctx is read.
     If `above_tx_pair`/`left_tx_pair` disagree with libvpx, the
     `tx_probs[max_tx][ctx]` row diverges, the wrong tree-walk
     consumes the wrong bits.
  3. **Coef-context iteration when `bsize < BLOCK_8X8`** (the per-bmi
     2x2 sub-block walk at lines 886-893 of `reconstruct.rs`). Our
     `plane_bw_4x4 = (2 >> ss_x)` matches libvpx's `pd->n4_w` for
     sub-8 luma, but the chroma `plane_bw_4x4` for sub-8 might not.
     If Y reads 4 sub-blocks but UV reads 1 or 2 (instead of 1), the
     decode_block_tokens call count diverges.
  4. **`uv_mode` read ordering.** Our code reads `y_mode` then
     `uv_mode` (matches libvpx). But the prob row used —
     `VP9_KF_UV_MODE_PROB[y_mode as usize]` — depends on `y_mode`
     numerically. If our `IntraMode::from_u8` ordering differs from
     libvpx's `PREDICTION_MODE` enum, the prob row is wrong.
- **No fix committed.** D.14 produced one diagnostic deliverable:
  - `crates/rvp9-decoder/src/tile.rs` — `RVP9_TRACE_KFSB=1` env-gated
    per-SB bool-cursor checkpoint trace in `decode_tile_intra`.
    Reusable for any future keyframe cursor-drift investigation (and
    needed by the next agent to compare against a libvpx-side per-SB
    dump). Output format:
    ```
    RVP9_DBG kfsb tile_data.len=876 bounds=mi_row[0..36] mi_col[0..44]
    RVP9_DBG kfsb mi=(0,0) (range=128, value=0x79..., count=56, pos=8)
    RVP9_DBG kfsb mi=(0,8) ...
    ```
- **Conformance unchanged:** 22 / 169 / 10 / 201. Tests + clippy clean.
- **Recommendation for D.15:**
  1. Hack a libvpx debug build to dump `(range, value, count, buffer_pos)`
     at the entry to `decode_partition` in `vp9_decodeframe.c` for
     every SB on q63 f1. Diff against our `RVP9_TRACE_KFSB=1` output.
     **First divergent SB = bug origin.**
  2. Inside that SB, bisect by adding bool-reader register dumps at
     finer-grained intervals — at the entry to `decode_block`
     (mode-info read), then after each leaf read (segment_id, skip,
     tx_size, y_mode, uv_mode). The smallest-magnitude divergence will
     point at the specific syntax element being read incorrectly.
  3. Prioritize suspects (1) and (2) above — q63 is dominated by skip=1
     blocks at the maximum quantizer, so the skip + tx_size reads are
     the highest-bandwidth syntax elements in this bitstream.
  4. Note `dump_frame_diff` first-divergence pixel was at
     (col=320, row=127) — but that's the POST-LF effect of pre-LF
     SB(2, 5) being decoded with EOS-cursor (constant blocks). The
     ACTUAL drift starts at SB(0, 0) (60 B vs ~29 B budget). Don't
     waste time drilling on (320, 127) — the cursor was already
     wrong many SBs earlier.

---

## D.15 returned — sub-8 luma skip-context shrinks n4 to 1 instead of 2 (FIXED, +9 keyframes; commit `2598b25`)

- **Target:** `vp90-2-00-quantizer-63.webm` frame 1 (the D.14 wedge),
  plus the cascade across all 352×288/352×240 keyframes that share the
  same partition-tree shape.
- **Approach:** built a libvpx-side per-SB cursor dump by patching
  `libvpx-stock/vp9/decoder/vp9_decodeframe.c` (env-gated
  `RVP9_TRACE_KFSB=1` `fprintf` at the entry to `decode_partition`
  for `bsize == BLOCK_64X64`) + rebuilt `vpxdec.exe` against the
  modified `libvpx.a`. Diff'd against rvp9's matching trace.
- **Bisect 1 — per-SB cursor:** SBs (0,0) through (16,32) match
  bit-for-bit; SB(16,40) diverges (rvp9 already at EOS pos=876, libvpx
  at pos=785). **Bug origin = inside SB(16,32).**
- **Bisect 2 — per-decode_block cursor:** added matching
  `RVP9_TRACE_KFLEAF=1` traces on both sides. Within SB(16,32), the
  divergence appears at the (20,32) Block16×16 plane-0 Tx16×16
  decode_coefs: libvpx consumes ~50 bits, rvp9 consumes ~10. eob differs
  → cursor diverges.
- **Bisect 3 — per-decode_coefs inputs:** dumped `ctx` (initial pt) +
  the raw above/left coef-context cells on both sides at decode_coefs
  entry. **rvp9: `a=[1,1,1,1] l=[0,1,0,0] ctx=2`. libvpx:
  `a=[1,1,1,1] l=[0,0,0,0] ctx=1`.** The stray `left[9]=1` flag in rvp9
  flips the initial_pt up by one, picks a totally different
  `coef_probs[band][ctx]` slice, decode_coefs takes a different path.
- **Root cause:** for sub-8 luma blocks (Block4×4 / Block4×8 / Block8×4)
  libvpx's `decode_partition` `hbs == 0` branch calls
  `decode_block(... bwl=bhl=1)` — i.e., it pins `bw = bh = 1` mi and
  thus `pd->n4_w = pd->n4_h = 2` 4×4 cells for plane 0 regardless of
  the sub-block's actual 4×{4,8} or {4,8}×4 shape. rvp9's
  `reset_skip_context` was using `plane_bsize.n_4x4_*` which returns
  1×1 / 1×2 / 2×1 / 2×2 for sub-8 — short by one cell in at least one
  axis on every sub-8 skip block. The un-cleared cell stayed `1` from
  the previous (non-skip) decode at the same y-offset, poisoning
  `initial_pt` for the next decode_coefs that lands on the same SB-row
  y-offset.
- **Fix:** `crates/rvp9-decoder/src/tile.rs::reset_skip_context` —
  for `bsize < Block8×8` && `plane == 0`, force `n4_w = n4_h = 2`.
  Chroma stays at `uv_plane_block_size(Block4×4).n_4x4_*` = 1 (already
  matches libvpx's `(1<<1) >> ss = 1` for ss=1). 6-line change inside
  the existing function; companion comment explaining the libvpx call
  pattern.
- **Conformance: 22 / 169 / 10 → 31 / 160 / 10** (+9 frames). Every
  previously-failing 352×288/352×240 **keyframe** now passes:
  `vp90-2-00-quantizer-26.webm`, `vp90-2-00-quantizer-63.webm`,
  `vp90-2-01-sharpness-1.webm`, `vp90-2-01-sharpness-7.webm`,
  `vp90-2-03-deltaq.webm`, `vp90-2-05-resize.ivf`. Plus
  `vp90-2-02-size-32x32.webm` lights up frames 2-9 (9/10, only frame
  10 still mismatches). Tests + clippy clean. Ratchet tightened
  (`MIN_PASSES=31`, `MAX_MISMATCHES=160`).
- **Deliverables (committed):**
  - `crates/rvp9-decoder/src/tile.rs` — the fix + a `RVP9_TRACE_KFLEAF=1`
    per-decode_block cursor + (per-syntax-element) checkpoint trace.
    Companion to D.14's per-SB `RVP9_TRACE_KFSB=1`.
  - `crates/rvp9-decoder/src/detokenize.rs` — a `RVP9_TRACE_KFLEAF=1`
    per-decode_coefs-call `(initial_pt, above-cells, left-cells)` dump
    for diff vs the libvpx-side `kfleaf_al` line.
  - `libvpx-stock/vp9/decoder/{vp9_decodeframe.c,vp9_detokenize.c}`
    (separate tree, NOT in this repo): per-SB + per-decode_block +
    per-decode_coefs `RVP9_TRACE_KFSB=1` / `RVP9_TRACE_KFLEAF=1`
    fprintf traces. Rebuilds `vpxdec.exe` with the instrumentation
    when env unset → behaves like stock vpxdec. Reusable for any
    future keyframe cursor-drift hunt; mirrors rvp9's trace format
    line-for-line.
- **Cascade NOT lit up (suspect — independent bugs):** inter frames
  (`q26 f2+`, `sharpness f2+`, `deltaq f2+`, `resize f2+`, `aq2 f2+`,
  `tile_*` f2+) all still mismatch. The keyframe-adapt path may be
  miscounting now-correct coef-flag transitions, OR there's a separate
  inter-frame bug. D.13's investigation suggested keyframe-adapt
  produces wrong probs for frame 2 — that finding now needs re-running
  against the post-D.15 frame 1.
- **Next agent recipe for D.16:**
  1. Re-run D.13's frame-context dump on `vp90-2-09-aq2.webm` frame 1
     end-of-keyframe-adapt and `vp90-2-00-quantizer-26.webm` frame 1
     end-of-keyframe-adapt. Compare against libvpx's adapt result.
  2. If keyframe-adapt produces correct probs but frame 2 still
     mismatches, fall back to the same per-SB / per-decode_block /
     per-decode_coefs differential infra D.15 added but for the
     **inter** path — likely the bug is now in `read_inter_block_mode_info`
     neighbour cells or in the per-block coef-context for inter blocks.
  3. The size-32x32 frame 10 is the lone interior mismatch with the
     rest of that vector passing — likely a small downstream bug
     (probably also a coef-context edge case at a frame boundary).
     Tractable; useful single-frame test target for any inter cascade
     investigation.

---

## D.16 returned — aq2 keyframe-adapt + post-CH FC + per-block decode all MATCH libvpx; bug isolated to per-pixel MC/recon at mi=(0,2) (NEGATIVE-with-localization)

- **Target:** `vp90-2-09-aq2.webm` frame 2 (= first shown inter post-altref).
  The D.13 "MV decode at mi=(0,0) is wrong" hypothesis turned out to be a
  false-trail: D.13's "expected libvpx output" was the wrong frame (frame
  0 keyframe, not frame 2). After fixing the comparison file
  (`vpxdec --i420 --limit=2` → take the SECOND frame only), the first
  Y divergence shifts from (4, 0) to **(29, 0)**.
- **Step 1 — FrameContext post-compressed-header dump.** Built libvpx
  `RVP9_TRACE_FC=1` instrumentation in `vp9_decodeframe.c::read_compressed_header`
  + matching rvp9-side trace in `Decoder::decode_inter_frame` (right
  after `parse_compressed_partition_inter`). Both dump every FC field
  the per-block decode actually consumes (skip_probs, intra_inter_prob,
  comp_inter_prob, single_ref_prob, comp_ref_prob, inter_mode_probs,
  nmvc.joints + per-component sign/classes/class0/bits). **EVERY byte
  matches libvpx for frame 1 (altref) AND frame 2.** That fully clears
  D.13's suspect #1 (keyframe-adapt) AND its suspect #3 (backward-adapt
  of intra_inter_prob/mv probs). The FC bank value rvp9 hands to the
  per-block decoder is bit-for-bit identical to libvpx's.
- **Step 2 — per-block INTERMI trace.** Added libvpx
  `RVP9_TRACE_INTERMI=1` instrumentation in
  `vp9_decodemv.c::read_inter_frame_mode_info` (cursor checkpoint
  before/after each syntax element + final mode-info summary) +
  matching rvp9-side trace in `tile.rs::decode_inter_block`. Diff'd
  the per-block summaries across frame 2:
  - **mi=(0,0) matches exactly** — bsize=6, seg=0, skip=1, tx=2,
    mode=NEWMV(13), ref=[Last, NoRef], mv0=(-2,-8). rvp9 = libvpx.
  - **mi=(0,2) matches exactly** — bsize=6, seg=0, skip=1, tx=2,
    mode=NEAREST(10), ref=[Last,NoRef], mv0=(-2,-8). rvp9 = libvpx.
  - The first 12 frame-2 blocks (mi=(0,0..6) parent + 4 sub-blocks
    inside the partition tree) all match.
  - First semantic divergence is **mi=(0,8) seg=1 (rvp9) vs seg=0
    (libvpx)** — but mi=(0,8) is at pixel col 64+, well AFTER the
    first picture divergence at pixel (29, 0) which is inside mi=(0,2).
  - The "mv1=(-32768,-32768)" lines are libvpx's uninitialised-mv
    sentinel for single-ref blocks (mi->mv[1] never written when
    ref_frame[1] = NO_REF_FRAME). Cosmetic, not a real divergence.
- **Conclusion: the bug is downstream of mode-info decode**, at the
  per-pixel **MC / residual / loopfilter** stage for the first BLOCK_16X16
  inter block at mi=(0,2). Per-pixel diff vs libvpx I420 dump:
  - First Y diff at (29, 0): got=68, expected=67. The block at pixel
    cols 28..31, rows 0..3 (= right edge of mi=(0,2) BLOCK_16X16):
    ```
    got       expected      diff
    67 68 68 69    67 67 67 67    +0 +1 +1 +2
    67 67 68 68    67 67 67 67    +0  0 +1 +1
    67 67 68 68    67 67 67 67    +0  0 +1 +1
    67 67 67 68    67 66 66 66    +0 +1 +1 +2
    ```
  - mi=(0,2) is `skip=1` → no residual → output IS pure MC from LAST
    (keyframe). The keyframe is bit-exact vs libvpx (verified via
    dump_frame_diff frame 1 → all planes MATCH). With MV=(-2,-8) the
    16x16 MC reads from keyframe cols 15..30, rows -0.25..14.75.
  - Cols 0..27 of mi=(0,2) **match** libvpx (rows 0..3). Cols 28..31
    diverge. Same MC kernel call, same source rows, same single-ref
    convolve — yet output differs at the rightmost 4×4. Possible
    causes:
    1. **Border-replication asymmetry** — `extend_frame_borders` (or
       `BorderedPlaneRef`) misbehaves at one edge but not the other.
       The MV col=-8 means MC reads from ref cols 15..30; col 30 is
       inside the visible 352-wide keyframe so no border read on the
       right. Top border (row -1) is read for ALL columns at row=0
       though, including cols 28..31. If our top-row replication writes
       (78, 78, ..., 78) but libvpx's writes (...keyframe row 0 ...),
       cols 28..31 specifically would diverge if our keyframe row 0
       at cols 27..30 differs from the top-border content we wrote.
       **Worth checking: are the top-border cells = visible row 0
       cells byte-for-byte?**
    2. **Subpel filter rounding asymmetry** — at row fractional
       position 12/16 (subpel_y_q4 = (mv_row * 2) & 15 = -4 & 15 = 12),
       the 8-tap filter is `[-1, 4, -11, 37, 112, -16, 4, -1]`. Single
       per-axis rounding `(sum + 64) >> 7`. Same logic for every
       output pixel. Hard to see how this asymmetrises across cols.
    3. **Loop-filter divergence** at frame 2 — frame 2 has
       `filter_level=24` (non-zero). The +0/+1/+2 pattern at a block
       boundary IS consistent with an off-by-one LF call (e.g.
       applying inner-edge filter where libvpx applies outer-edge,
       or filter-strength miscompute for adjacent blocks).
  - Suspect (3) — loop-filter — is the **highest prior**. The diff
    pattern (small +0/+1/+2, localized to a 4×4 block at the
    boundary, frame_level=24) matches the M.9 / J wireup signature
    that was fixed at the LF stage previously. Re-test with
    `RVP9_DISABLE_LF=1` to confirm.
- **Cascade not yet lit.** Conformance unchanged: 31 / 160 / 10 / 201.
  Tests + clippy clean. Diagnostic infrastructure committed for D.17.
- **Deliverables (in rvp9 repo):**
  - `crates/rvp9-decoder/src/lib.rs` — env-gated `RVP9_TRACE_FC=1`
    dump in `decode_inter_frame` post-compressed-header.
  - `crates/rvp9-decoder/src/tile.rs` — env-gated `RVP9_TRACE_INTERMI=1`
    one-line per-block summary in `decode_inter_block`.
- **Deliverables NOT in this repo (separate tree at
  `D:/language/libvpx-stock/`):**
  - `vp9/decoder/vp9_decodeframe.c::read_compressed_header` —
    `RVP9_TRACE_FC=1` post-CH FC dump.
  - `vp9/decoder/vp9_decodemv.c::read_inter_frame_mode_info` —
    `RVP9_TRACE_INTERMI=1` per-syntax-element + per-block summary
    dump. Filters on `cm->show_frame && cm->current_video_frame >= 1`
    so frame 0 (keyframe) and frame 1 (altref non-shown) are silent.
  - `libs-x86_64-win64-gcc.mk` — added `export TMPDIR/TMP/TEMP=/tmp`
    so gcc-mingw rebuilds succeed under our shell env. `gcc_wrap.sh`
    scratch (unused, can remove).
- **Next agent recipe for D.17:**
  1. Run `RVP9_DISABLE_LF=1` + `dump_frame_diff` on aq2 frame 2 to
     confirm/refute the LF-divergence hypothesis. If the pre-LF picture
     matches libvpx, the bug is purely in the LF apply for inter frames.
     (Reminder: D.13 already verified `RVP9_DISABLE_LF=1` showed the
     same cascade pattern — but that was against the WRONG expected
     file. Re-run with the correct frame-2-only expected.)
  2. If pre-LF diverges too, walk in to the MC at mi=(0,2): dump the
     keyframe Y-plane top-border row (rows -1..-7, cols 14..32) on
     both rvp9 and libvpx. Compare. Look for asymmetric replication.
  3. Separately track the seg_id divergence at mi=(0,8) — that's an
     INDEPENDENT bug (compound-ref or temporal_update inheritance in
     `prev_segment_ids` from frame 1's altref). Will need its own
     investigation but is **not** the cause of the first-pixel
     divergence.
  4. The diagnostic infra committed by D.16 (FC + INTERMI traces) is
     ready to drop into any other failing vector. Apply to
     `vp90-2-00-quantizer-26.webm` frame 2 if aq2's bug doesn't
     generalise.

## D.19 — 2026-05-19 (Arthur asleep, lead dispatch)

- **Pre-LF A/B test outcome: pre-LF DIFFERS.** D.16's three-way
  suspect list (LF, MC border replication, downstream seg_id)
  collapses to **MC / reconstruction** — the LF is NOT the primary
  bug. `RVP9_DISABLE_LF=1` + `dump_frame_diff` against
  `notes/d17/aq2_libvpx_nolf_f2.yuv` shows our pre-LF picture
  diverges from libvpx's pre-LF picture too. **One bit for D.20.**
- **`dump_frame_diff` was skipping non-shown altref frames** —
  introduced when the binary was first written. The dumper called
  `continue` on `!shown`, so the ALTREF slot in the ref-pool was
  uninitialised when frame 2 reached `decode_frame`. This made
  every "where's the first diff" analysis lie by ~32 columns:
  with altref skipped, first Y diff at (32, 0); with altref
  decoded, first Y diff at (64, 0). **Conformance harness was
  NOT affected** — it always decodes every frame. Bug was
  diagnosis-only but had cost D.17 ~3 hours chasing the wrong
  block. Fix: always call `decode_frame(&frame)`, just gate the
  MD5 save / `shown_idx` advance on `shown`.
- **Bug surface now precisely localized:**
  - First pre-LF divergence at `(col 64, row 0)`. Cols 0..63 of
    row 0 MATCH libvpx pre-LF byte-for-byte.
  - mi=(0,8) = pixel (col 64, row 0) is **the FIRST compound
    prediction block in frame 2** (per `notes/d17/libvpx_imi.txt`:
    `bsize=6 ref=[1,3] mv0=(-2,-6) mv1=(24,66) mode=13`). All
    earlier blocks (mi=(0,0), (0,2), (0,4), (0,6)) are single-ref.
  - Mode-info parse for mi=(0,8) matches libvpx (verified via
    `RVP9_TRACE_MC` on rvp9 side vs `LIBVPX_TRACE_MC` on libvpx
    side: identical mv, mv_q4, int_x/int_y, subpel_x/subpel_y for
    both refs).
  - MV-to-Q4 conversion is correct (`mv_q4_col = mv.col *
    (1 << (1 - ss_x))` matches libvpx `scaled_mv`).
  - No `clamp_mv_to_umv_border_sb` in rvp9 — but for the failing
    mi=(0,8) the clamp is a no-op (block sits well inside the
    frame); so this isn't the immediate bug.
  - Suspect (1): **our altref (frame 1) reconstruction differs
    from libvpx's altref** at the reference-read region. Cannot
    confirm directly — no on-disk libvpx altref dump and the
    libvpx-stock build environment had stale `.asm.d` files from
    D.17 that block `make vpxdec.exe` rebuilds (~20 broken deps
    with `: 1` literal; I deleted them but the link step still
    won't relink the existing `.o` files). **Path forward for
    D.20:** add an rvp9-side dump of the altref Y plane after
    decode (env-gated), and/or fix the libvpx tree's `.d` files
    so an instrumented vpxdec can rebuild.
  - Suspect (2): **our compound averaging path is wrong**.
    Hand-math via the trace: ref 1 (ALTREF, mv=(24,66),
    subpel_x=4) at output (64, 0) yields `pred = 49`. Final pixel
    = 62, so ref 0 (LAST, mv=(-2,-6), 2D convolve) must yield
    `pred = 74` (since `(74 + 49 + 1) >> 1 = 62`). Libvpx final
    = 74 ⇒ libvpx ref1 ~73 (if ref0 also ~74). 24-LSB gap between
    our ref1_pred and libvpx's ref1_pred is what's killing us.
    Could be: altref content wrong, OR subpel filter on this
    specific tap row wrong (subpel_x=4 = filter row
    `[-1, 4, -16, 112, 37, -11, 4, -1]`), OR wrong reference
    frame index being read.
- **Diagnostic infra added (committed under D.19):**
  - `crates/rvp9-decoder/src/reconstruct_inter.rs` —
    `RVP9_TRACE_MC=1` per-MC trace (mi, plane, ref, bw, bh, dst,
    mv, mv_q4, total_q4, int_x/int_y, subpel_x/subpel_y, rf).
  - Same file — `RVP9_TRACE_MC_SRC=1` source-data dump on the
    fast path (first 18 cols of the convolve footprint after the
    `-3` pre-shift).
  - `crates/rvp9-harness/src/bin/dump_frame_diff.rs` — fix for
    altref-skip bug. **Diagnosis output now reliable for any
    future inter-frame analysis.**
- **Cascade not yet lit.** Conformance unchanged: 31 / 160 / 10 /
  201. Tests + clippy clean.
- **Recipe for D.20:**
  1. Confirm compound averaging is correct by writing a
     differential test: synthesize two refs, run compound MC,
     compare against a libvpx-faithful manual reference. (~30 min)
  2. If the diff test passes, the bug IS in our altref content.
     Add an env-gated altref-plane dump in `decode_inter_frame`
     post-extend_borders (gate on `RVP9_DUMP_ALTREF=1`), then
     build a libvpx altref dump via the modified `vpxdec.exe`
     (after fixing the broken `.d` files in
     `D:/language/libvpx-stock/`: `find . -name "*.asm.d" -exec
     grep -l ": 1$" {} \; -exec rm {} \;` — already done by D.19;
     just need to relink). Diff our altref vs libvpx altref to
     localize.
  3. If altref matches, the bug is in subpel filter (specifically
     filter row 4 of EIGHTTAP_KERNEL?) or the 2D convolve.
     Differential against a precomputed convolve8 reference.
  4. Probe second compound block (mi=(4,0) at altref) by adding
     a row-32 dump to verify same bug pattern.

## D.20 — 2026-05-19 (Arthur asleep, lead dispatch)

- **D.19's "altref content wrong" hypothesis: CONFIRMED.** Our
  altref Y plane at row 0..3 cols 64..71 = `73 73 73 73 73 73 73
  73` for all four rows — a 4×8 flat patch of 73's. The trace
  shows altref's mi=(0,8) is `bsize=Block32x16 mv=(-24,-80)
  int=(54,-3) subpel=(0,0)` — integer-pel MC reading keyframe at
  (row -3..13, col 54..86). With border extend, KF rows -3..-1
  replicate row 0; KF(row 0, cols 54..61) = `73 73 73 73 73 73 73
  73` (flat region in the keyframe). So MC yields 73's at altref
  (row 0..3, cols 64..71). Block has `skip=0` (residual decoded)
  but our output is exactly MC ⇒ residual is zero in our decode.
- **Localized root cause: altref's mi=(0,8) seg_id is WRONG.**
  Added env-gated trace (`RVP9_TRACE_SEG_ID=1`) at
  `read_inter_segment_id`. Our altref decodes seg=1 at
  mi=(0,8) (Block32x16). LIBVPX's altref must produce seg=0 at
  the same location: libvpx_imi.txt for frame 2 shows mi=(0,8)
  `post_segid seg=0`, and frame 2 has `update_map=false ⇒
  inherits altref's seg map directly. Our frame 2 trace at
  mi=(0,8) shows seg=1 (which it inherits from our altref's
  wrong seg map). Block-count match for frame 2 (484 rvp9 / 484
  libvpx) ⇒ partition decisions agree post-altref; the
  bool-coder cursor resyncs at start of frame 2.
- **Mechanism: bool-coder divergence inside altref's mi=(0,8)
  syntax.** Our trace shows pred_flag=0 (less-likely under
  pred_probs[ctx=1]=81 ≈ 32% for bit 0) + tree-decode → 1. For
  libvpx altref to yield 0 either:
   (a) pred_flag=1 → inherit predicted=0 (68% under prob=81)
   (b) pred_flag=0 + tree-decode → 0 (≈ 84% under tree_probs
       [255, 217, …])
  Either way our bit reads diverge from libvpx's. The divergence
  is local: our pred_flag bits at mi=(0,2)..(0,6) all matched
  expectations (seg_id_predicted=true ≈ 68%); the cursor only
  goes off at mi=(0,8). Suspect:
  1. Partition tree at SB(0,8) reads different than libvpx's
     altref (we picked PARTITION_SPLIT@Block64x64 +
     PARTITION_HORZ@Block32x32 → two Block32x16; libvpx may
     have picked different).
  2. Bool-coder cursor was off by 1 bit between end of SB(0,0)
     and start of SB(0,8) (some block's syntax read different
     length).
  3. Bug in seg_id_predicted "right neighbor" context: when
     ctx is computed for mi=(0,8)'s LEFT neighbor, we should
     read from mi=(0,7)'s `seg_id_predicted`. The Block16x16 at
     mi=(0,6) covers cells (0,6..7) ⇒ left neighbor at mi=(0,7)
     is the same Block16x16 ⇒ its seg_id_predicted = 1. Matches
     trace.
- **Cannot pin bool-coder divergence to a specific bit without a
  libvpx altref bit-cursor trace.** vpxdec.exe needs to be
  rebuilt with INTERMI instrumentation that drops the
  `cm->show_frame` filter (currently silences altref). Build env
  for libvpx-stock is wedged: no `make`/`mingw32-make` in our
  shell path. **Required for D.21:** install make (e.g.,
  `pacman -S make` under msys2) and rebuild libvpx-stock vpxdec
  with `cm->show_frame` filter relaxed.
- **Diagnostic infra added under D.20:**
  - `crates/rvp9-decoder/src/lib.rs` —
    `RVP9_DUMP_ALTREF=<path>` env hook dumps the visible Y
    plane of any non-shown frame post-extend_borders.
  - `crates/rvp9-decoder/src/tile.rs` —
    `RVP9_TRACE_SEG_ID=1` env-gated per-block seg-id trace
    (predicted, pred_ctx, segment_id, seg_id_predicted bit).
  - `crates/rvp9-decoder/src/lib.rs` — `RVP9_TRACE_SEG=1`
    extended to dump the first SB-row of the per-frame seg map
    (rows 0..3, cols 0..15).
- **Cascade not yet lit.** Conformance unchanged: 31 / 160 / 10
  / 201. Tests + clippy clean.
- **Recipe for D.21:**
  1. **Fix libvpx build env first.** Confirm `make` is on PATH;
     `pacman -S make` under msys2 if not. Then
     `cd D:/language/libvpx-stock && touch vpxdec.c && make`
     should relink `vpxdec.exe`. (.asm.d files are still
     ":1"-corrupted from D.17 — delete + rebuild.)
  2. Drop the `cm->show_frame` filter from
     `vp9/decoder/vp9_decodemv.c` so INTERMI trace fires on
     non-shown frames (altref). Re-decode aq2 with
     `RVP9_TRACE_INTERMI=1` to capture libvpx's altref seg-id
     trace.
  3. Diff rvp9 vs libvpx altref INTERMI line-by-line. The first
     divergent line is the bool-coder cursor target.
  4. The divergence likely points at one of:
     - Partition tree at SB(0,8) (Block64x64 split decision)
     - Specific block's syntax-element bit count mismatch in
       SB(0,0)'s rightmost cells (mi cols 4..7)
     - The pred_flag computation in `seg_id_pred_ctx`
       (`above_sip + left_sip`) for Block32x16
  5. Once the offending bit is identified, fix and re-run
     conformance. Expected cascade range: 5-100+ (every inter
     vector that exercises segmentation with temporal_update
     hits the same path).
- **Alternative D.21 path if libvpx oracle can't be rebuilt:**
  Write a synthesis test: build a minimal aq2-like bitstream
  with known mode-info content for SB(0,8) of an altref frame,
  decode with our decoder, compare against hand-computed
  expected bit positions / segment_id values. Tedious but
  doesn't depend on libvpx rebuild.

## D.21 — 2026-05-19 (Arthur asleep, lead dispatch)

- **Root cause: wrong `size_group_lookup` mapping in
  `read_intra_in_inter_mode_info`.** Our match expression had
  Block8x8 mapped to size_group=0 (shared with Block4x4/4x8/8x4)
  and the Block16x16/Block32x32 cohorts each shifted one slot
  too low. Libvpx (`vp9_common_data.c:32-33`) is
  `{0,0,0, 1,1,1, 2,2,2, 3,3,3,3}` indexed by BLOCK_SIZE. We had
  `{0,0,0,0, 1,1,1, 2,2,2, 3,3,3}`. The default-arm intra Y mode
  read for non-keyframe intra blocks (Block8x8 and larger) uses
  `fc.y_mode_prob[size_group]`; the wrong row produced a
  different y_mode and desynced the bool-coder cursor on every
  inter stream containing an intra block ≥ Block8x8.
- **Validation path:** rebuilt libvpx-stock vpxdec (TMPDIR=C:/tmp
  in `libs-x86_64-win64-gcc.mk`, added same to
  `examples-x86_64-win64-gcc.mk` and `gxx_wrap.sh`; deleted
  19 corrupted ".asm.d" files), patched
  `vp9/decoder/vp9_decodemv.c` to drop the `cm->show_frame`
  filter on RVP9_TRACE_INTERMI, added per-frame counter
  (`rvp9_trace_current_frame`, distinct from
  `cm->current_video_frame` which conflates altref+visible into
  one f= value), added RVP9_TRACE_PART for partition decisions.
  Diff against rvp9's existing RVP9_TRACE_PARTITION / SEG_ID /
  INTERMI traces on aq2 frame 2 pinpoint mi=(5,0): both
  decoders read partition_post range=224 count=39, skip=0 from
  range=230, is_inter=0 from range=144 count=34, tx=0 from
  range=130 count=33 — all matching. Then libvpx
  post_modeinfo reads range=216 count=29 with mode=0 (DC);
  rvp9 reads range=232 count=28 with mode=2 (H). Same cursor
  entry but different y_mode ⇒ wrong probability row.
- **Cascade scope:** aq2 frame 2 lights up plus 24 other inter
  vectors that exercise intra-in-inter blocks. Conformance
  **31/160/10 → 56/135/10** (25 new vectors). All tests + clippy
  clean. Doctest pre-existing failure on
  `Decoder::parse_compressed_partition` ignored as per CI shape.
- **Diagnostic artefacts:**
  - `D:/language/libvpx-stock/vpxdec.exe` rebuilt with
    RVP9_TRACE_INTERMI + RVP9_TRACE_PART firing on altref.
  - `C:/tmp/libvpx_imi_d21.txt` — full libvpx altref + visible
    INTERMI/PART trace (~29 MB).
  - `C:/tmp/rvp9_trace_d21.txt` — matching rvp9 trace via
    `dump_frame_diff --frame 2 RVP9_TRACE_PARTITION=1
    RVP9_TRACE_SEG_ID=1 RVP9_TRACE_INTERMI=1`.
  - Build-env fix preserved in
    `D:/language/libvpx-stock/{libs,examples}-x86_64-win64-gcc.mk`
    and `gxx_wrap.sh`; `vp9/decoder/vp9_decodemv.c` and
    `vp9/decoder/vp9_decodeframe.c` carry persistent
    rvp9_trace_current_frame bump + show_frame-filter drop.
- **Followups (no fix landed):**
  - 135 vectors still mismatch. Cascade-likely paths next: most
    failing keyframes are the 1×N tile vectors with no aq2-style
    cause; D.22 should pick the densest mismatch cluster.
  - 32×32 frame 10 still mismatches (D.20 prediction confirmed —
    independent of seg_id-specific bug).

## D.22 — 2026-05-19 (Arthur asleep, lead dispatch)

- **Two root causes (both surfaced via a single libvpx INTERMI
  altref diff on aq2 `f=16`):**

  1. **MV-component HP count gating.** `read_mv_component`'s bump
     of `counts.hp[hp]` / `counts.class0_hp[hp]` was gated on
     `use_hp` (the per-block 1/8-pel flag). Libvpx's
     `inc_mv_component` (`vp9_entropymv.c:111-136`) is also gated
     on a `usehp` parameter — but the only caller, `vp9_inc_mv`
     (`:139-151`), **hard-codes `usehp=1`**. So libvpx ALWAYS
     bumps `hp[e]` regardless of the block-level `use_hp`; when
     the read had `use_hp=false`, the hp bit is hard-coded to 1
     at decode time and `e = o & 1 = 1` falls out of the
     composition formula, so `hp[1]` still gets the bump. Our
     gating under-counted `hp[1]`, drifting `comp.hp` /
     `comp.class0_hp` away from libvpx every frame that adapts MV
     probs with `allow_high_precision_mv` enabled.

  2. **Altref `prev_frame->mvs` swap.** `decode_inter_frame`
     pulled `prev_mv_grid` from `mv_grids[ref_frame_indices[0]]`
     — the LAST_FRAME ref-map slot's grid. Libvpx's
     `cm->prev_frame->mvs` is set by `cm->prev_frame =
     cm->cur_frame` (`vp9_decoder.c:478`) at the end of EVERY
     non-show-existing decode, independent of `refresh_frame_flags`.
     For altref frames whose LAST_FRAME index points at a slot
     holding an OLDER frame's grid, the two diverge. Added
     `Decoder::prev_frame_mv_grid` + dims to track the truly-
     most-recently-decoded grid directly, mirroring libvpx's
     swap exactly. (This fix is necessary but not sufficient on
     its own — moves aq2 mv0 from `(-30,-90)` to `(-32,-98)` but
     bits still differ; the second fix above resolves the
     remaining `(-32,-98)` → `(-32,-106)` gap.)

- **Validation path:** added libvpx-side fine-grained `read_mv` +
  `read_mv_component` traces (RVP9_TRACE_RDMV /
  RVP9_TRACE_RDMV_COMP) emitting pre-state, joints/comps probs,
  per-bit `bits[i]` reads, fr/hp probs+bits. Same instrumentation
  on rvp9. On aq2 `f=16` (2nd altref) mi=(0,0) first NEWMV: both
  impls enter `read_mv` with bit-identical state (`range=150
  value=0x95 count=38`), bit-identical probs (`joints=[3,75,103]`,
  `comp0.classes=[232,117,...]`, `comp1.bits=[117,128,...]`),
  identical `read_mv_component` for the row component. The first
  divergence is at the ROW component's HP read: libvpx prob=99,
  rvp9 prob=131. That HP-prob mismatch caused the row decode to
  end with `range=141` (libvpx) vs `range=224` (rvp9), which
  cascaded into the col decode reading different `bits[i]` values
  → final col=-106 (libvpx) vs col=-98 (rvp9). Diagnostic traces
  removed post-fix; libvpx left clean.

- **Cascade scope:** aq2 14→100 (+86), sharpness-1 3→10 (+7),
  sharpness-7 3→10 (+7), tile_1x2 2→10 (+8),
  show-existing-frame 1→5 (+4). **Conformance 56/135/10 →
  168/23/10 (+112 frames).** All tests + clippy clean.

- **Diagnostic artefacts:**
  - `crates/rvp9-decoder/src/lib.rs`: pub
    `RVP9_TRACE_FRAME_COUNTER` (AtomicU32, mirrors libvpx
    `rvp9_trace_current_frame`) bumped at top of `decode_frame`.
  - `crates/rvp9-decoder/src/tile.rs`: per-element `RVP9_TRACE_INTERMI`
    fine-grained trace (`enter` / `post_segid` / `post_skip` /
    `post_isinter` / `post_txsize` / `post_modeinfo`) tagged with
    `f=N` matching libvpx's altref-inclusive format. New env hook
    `RVP9_TRACE_FRAME=N` filters to a single frame.
  - Existing libvpx-stock instrumentation (D.21) preserved.

- **Followups (D.23):**
  - 23 mismatches remain: resize 7, tile_1x4 8, tile_1x8 8.
  - tile_1x2 fully lit up (10/10) but tile_1x4 / tile_1x8 still
    8/2 mismatched each. Likely a multi-tile entropy-isolation
    bug specific to >2 tiles per row. Densest cluster for D.23.
  - resize 7 mismatches: separate cause — frame-resize state
    machine.

## 2026-05-19 — D.24 (resize / scaled-MC)

- **Picked resize lane** (7 mismatches on `vp90-2-05-resize.ivf`)
  in parallel with D.23 (multi-tile). Stream resizes 352×288 →
  282×173 → 352×288; mid-stream `setup_frame_size_with_refs`
  fires and refs are wrong size from current. Frame 1 (kf)
  already passed; inter frames 2-10 failed.

- **Two bugs found, both in scaled-MC integer/sub-pel math:**

  1. **ScaleFactors built with aligned dims, not crop dims.**
     We used `ref_buf.aligned_y_width()` and `yv12.aligned_y_width()`
     (the 8-aligned size). Libvpx feeds `y_crop_width` /
     `cm->width` (the raw header values) to
     `vp9_setup_scale_factors_for_frame` at
     `vp9_decodeframe.c:2928-2935`. For 282 the aligned dim is
     288, for 173 it's 176 — so x_scale_fp drifted by 6/288
     and y_scale_fp by 3/144 in the Q14 ratio, perturbing every
     scaled-MC block's integer base + subpel. Fix: pass
     `ref_buf.width() / height()` + `header.width / height`.
     This alone fixed all 7 frames' Y plane bit-exactly.

  2. **Chroma sub-pel mask used plane coord, not luma coord.**
     Libvpx's `build_inter_predictors` (`vp9_reconinter.c:155-186`)
     in the scaled path uses an asymmetric mapping:
     `scaled_buffer_offset(x_start, y_start, ...)` uses the
     *plane* coord (chroma = luma>>1) for the integer base, but
     `vp9_scale_mv(mv_q4, mi_x, mi_y, sf)` uses *luma* pixel
     coords (`mi_x = mi_col * 8`) for the sub-pel mask. Our
     `scaled_x_q4(plane_x, mv_q4)` collapsed both into one
     plane-coord-based sum, so for 4:2:0 chroma the sub-pel-Q4
     result rotated by a different amount mod 16, picking a
     different filter row and producing off-by-1 chroma on
     every chroma block in a scaled frame. Fix: in the
     `sf.is_scaled()` branch, split the sum the way libvpx
     does (helper `mul_q14` for the i64-widened Q14 multiply).
     Unscaled path keeps the cheap unified `scaled_x_q4` (where
     the asymmetry collapses to a no-op).

- **Validation path:**
  - `dump_frame_diff` per-frame on resize: after bug 1, Y
    matched on all 10 frames; chroma still off-by-1.
  - Worked through libvpx's `build_inter_predictors` math
    line-by-line. Concrete numerics for chroma column 38 on
    frame 7 (cur=352×288, ref=282×173, x_scale_fp=13125):
    `scaled(mi_x << 4) & 15 = 10` vs `scaled(plane_x << 4) & 15
    = 13` — same block, different subpel filter row.
  - After both fixes: vp90-2-05-resize.ivf 3→10 (+7), all
    frames bit-exact.

- **Conformance ratchet:** 168/23/10 → **175/16/10 (+7)**.
  `MIN_PASSES` raised to 175, `MAX_MISMATCHES` lowered to 16.
  All `-p rvp9-core -p rvp9-decoder -p rvp9-harness --lib
  --bins --tests` green; clippy `-D warnings` clean.

- **Files touched:**
  - `crates/rvp9-decoder/src/lib.rs` — `cur_y_width` /
    `cur_y_height` now use `frame.header.width / height` (crop).
  - `crates/rvp9-decoder/src/reconstruct_inter.rs` —
    `ScaleFactors::new` fed `ref_buf.width() / height()`; new
    libvpx-style branch in `reconstruct_inter_block` for the
    `sf.is_scaled()` chroma-aware sub-pel split + `mul_q14`
    helper.
  - `crates/rvp9-decoder/src/tile.rs` — comment update on
    `InterFrameDecodeParams::cur_y_width / height`.
  - `crates/rvp9-harness/tests/conformance.rs` — ratchet
    bump + multi-paragraph fix-rationale block.

- **Followups (D.25 / D.26):**
  - 16 mismatches remain: tile_1x4 8 + tile_1x8 8. D.23's
    lane (concurrent). Multi-tile entropy-isolation bug
    specific to >2 tiles per row.

- **Lessons:**
  - When porting libvpx code that takes `(plane_x, plane_y,
    mi_x, mi_y)` arguments, double-check whether the function
    treats them as the same coord system or not. Here the
    integer base and the sub-pel mask take coords in *different*
    systems — easy to miss when reading the code top-down.
  - For scale-factor envelope checks: aligned vs cropped dims
    must match libvpx exactly. Even a 6-pixel discrepancy in
    width changes the Q14 ratio by ~280 and breaks every
    scaled-MC block.

## D.23 (2026-05-19) — multi-tile (>2 tile cols) — NEGATIVE / localization only

- **Targets:** `vp90-2-08-tile_1x4.webm` (8 mismatches) +
  `vp90-2-08-tile_1x8.webm` (8 mismatches). Cumulative 16
  mismatches after D.24's resize landing.

- **What we confirmed (diagnostic infra):**
  - Frames 1 & 2 of tile_1x4 MD5-match libvpx. Frame 3 is the
    first failing frame.
  - First-divergent Y pixel: `(960, 1018)` on frame 3 of
    tile_1x4. Col 960 = px boundary between **tile-1 and
    tile-2** for tile_1x4 (mi_col=120). For tile_1x8 frame 3,
    first Y diff at `(1716, 2106)`.
  - tile_1x4 tile boundaries: `mi_col ∈ {0, 56, 120, 176, 240}`
    (px {0, 448, 960, 1408, 1920}). The first-diff column sits
    at the boundary that **also** exists in tile_1x2 (which
    passes), so the bug isn't intrinsic to that boundary in
    isolation — it depends on having a tile **between** two
    boundaries (i.e. interior tiles).
  - Spatial distribution analysis (`diff_y_dump.py`) of tile_1x4
    frame 3: **all** Y diffs concentrate in SB row 16
    (mi_row 128..135, the partial-bottom SB row). Breakdown:
    - `SB(16, 13)` (px 832..896): 2681 diffs
    - `SB(16, 14)` (px 896..960): 3618 diffs
    - `SB(16, 15)` (px 960..1024): 338 diffs
    - All other SBs: 0 diffs.
  - The cluster straddles the **tile-1/tile-2** boundary at
    mi_col=120 (px 960), heavily skewed to the LEFT side
    (in tile 1).

- **What we ruled out:**
  - Entropy-decode parity. `RVP9_TRACE_INTERMI` on tile_1x4
    frame 3 produces **identical** bool-reader cursor state
    (`range`/`count`) at every single per-block syntax element
    between rvp9 and libvpx. Sort-by-mi diff shows ONLY cosmetic
    `mv1=(0,0)` vs `(-32768,-32768)` (single-ref blocks where
    libvpx leaves mv[1] uninitialised) and the analogous mv0
    diff for intra blocks. **The decoded MVs / modes / coef
    streams are bit-identical.**
  - Block-walk order. libvpx interleaves SB-rows across tile
    cols (`tile_row outer → mi_row inner → tile_col → mi_col`),
    rvp9 walks tile-by-tile (each tile fully decoded before
    moving to the next). Despite the different walk order, the
    bool-coder consumption of each tile is per-tile-private and
    identical to libvpx's. Walk order alone is provably not the
    bug.
  - Tile-bound `is_inside` semantics for `find_mv_refs`. Our
    `TileNeighbourLookup` correctly clamps neighbour reads to
    tile-relative `(0..tile_mi_rows × 0..tile_mi_cols)`. The
    `BlockBounds::new` for `clamp_mv2` uses **frame-wide**
    `mi_rows`/`mi_cols`, which matches libvpx's
    `set_mi_row_col` 1:1 — that's deliberately not tile-clamped
    (libvpx's `xd->mb_to_*_edge` is set against frame dims).
  - `cur_mv_grid` / `cur_segment_ids` ordering. Both writes are
    based on the decoded block's (mi_row, mi_col, x_mis, y_mis)
    and produce identical content at end of frame regardless of
    tile-walk order, because each block only writes cells the
    block covers and overlap with adjacent tiles is bounded by
    `x_mis`/`y_mis` (clamped to frame edge only) — libvpx does
    the same.
  - Frame-2 output. Pre/post-LF Y/U/V planes match libvpx
    bit-exactly via `dump_frame_diff --frame 2`. So frame 3
    sees the **correct** reference pixels.

- **Open question / where we ran out of time:**
  - **Why does pre-LF recon of frame 3 diverge in SB row 16
    (partial-bottom SB)** despite identical entropy state and
    correct reference inputs?
  - Diff cluster suggests either:
    1. A subtle bug in pre-LF MC/IT-add at the
       **`mi_row + bh > mi_rows`** edge (this is the only SB
       row that triggers it for 1080-px frame).
    2. A bug in `apply_loop_filter_to_frame` specific to the
       LAST visible SB row, where `adjust_mask`'s
       `rows == 7` branch (mi_rows - mi_row = 7 for tile_1x4)
       takes a numerically-special path
       (`((1u32 << 16) - 1) as u16 = 0xffff`).
  - The fact that diffs CLUSTER on the LEFT side of the
    tile boundary (3618 vs 338) suggests it's tile-1's pre-LF
    recon that's wrong, and LF then partially "spreads" the
    diff into tile 2.

- **Files left as diagnostic state (uncommitted):**
  - `diff_y_dump.py` — spatial diff distribution helper for
    `dump_frame_diff`'s 4×4 grid output. Reusable for future
    multi-tile bisects.

- **Files NOT changed:** none. The 16 mismatches remain.

- **Conformance ratchet:** unchanged at 175/16/10.

- **Followups (D.25+):**
  - **First lever to pull:** add an env-gated pre-LF Y-plane
    dump (`RVP9_DUMP_PRELF=<path>` on shown frames) and a
    libvpx-side mirror probe that dumps `cm->cur_frame->y` just
    before `vp9_loop_filter_frame_mt` runs. Diff the two on
    tile_1x4 frame 3 — if pre-LF differs, the bug is in recon
    of SB row 16; if pre-LF matches, the bug is in LF.
  - **Second lever:** decode tile_1x4 frame 3 with our walk
    order **swapped to libvpx's interleaved order** (a
    surgical experiment, not a fix). If the diff goes away,
    the bug is in something correlated with our walk order
    (e.g. a per-frame mutable resource that gets touched in
    different orders); if not, we've conclusively ruled out
    walk order.
  - **Third lever:** the `partition_context` in `PartitionCtx`
    is reset at each SB row. For multi-tile, libvpx does **not**
    actually share `above_seg_context` across tile cols (each
    tile's `above_seg_context` slice is its own column range).
    Verify our `PartitionCtx::above_partition` is similarly
    isolated per tile — re-read against libvpx's `vp9_tile_init`
    + `init_macroblockd`.

- **Lessons:**
  - When the entropy decode of EVERY per-block syntax element
    matches the oracle bit-exactly but pixel output diverges,
    the bug is in pixel-domain code (recon / LF / border-extend),
    not in the bitstream parser. This shaves the search space
    dramatically. Use the spatial diff distribution to localize
    further before reading code.
  - "Both work the same in principle" reasoning is not enough
    when there's a behavioral diff in production — push for a
    concrete differential dump (pre-LF Y) to PROVE which pipeline
    stage diverges.

## D.25 — Stage 3 EXIT (2026-05-19) 🎯

**Result:** 175 passed / 16 mismatched / 10 skipped → **191 passed /
0 mismatched / 10 skipped / 201 total.** All 16 remaining mismatches
(tile_1x4 8 + tile_1x8 8) lit up; conformance gate switched from
ratchet (`MAX_MISMATCHES = 16`) to hard `assert_eq!(total_mismatched
== 0)`. Stage 3 closed.

**Root cause:** Residual decode loop in `reconstruct_inter_block`
(`crates/rvp9-decoder/src/reconstruct_inter.rs`) iterated the full
`plane_bh_4x4` × `plane_bw_4x4` extent of every block, including the
4×4 cells whose top-left lies outside the visible frame for partial-
bottom / partial-right blocks. Libvpx's `predict_recon_inter`
(`vp9_decodeframe.c:892-921`) clamps that iteration via:

```c
max_blocks_wide = n4_w + (mb_to_right_edge  < 0 ? mb_to_right_edge  >> (5 + ss_x) : 0)
max_blocks_high = n4_h + (mb_to_bottom_edge < 0 ? mb_to_bottom_edge >> (5 + ss_y) : 0)
```

The unclamped iteration in rvp9 reads coef tokens for those
invisible cells — drifting the bool-coder cursor relative to libvpx
on every partial-edge inter block. With 1920×1080 (mi_rows=135 = 16×8
+ 7) and tile_1x4 / tile_1x8 vectors, the partial-bottom SB row
of any non-first tile accumulates enough drift that the **altref
(non-show) frame** decodes wrong partition / mode for the partial-
bottom-row SBs near the tile's right edge. That altref then gets
splatted into the ref-pool, and the **next visible inter frame**
reads it as a compound reference — surfacing as the ~3000-pixel diff
band on SB row 16 of tile_1x4 frame 3 and the analogous bands on
tile_1x8.

**Fix:** add the same `max_blocks_{wide,high}` clamp to rvp9's
residual decode loop. Threads new fields `mi_rows` / `mi_cols` through
`InterBlockInputs` (the obvious shape — libvpx pulls them from
`cm->mi_{rows,cols}` for the same formula). Six lines of math
extending the existing `loff < plane_bh_4x4` / `aoff < plane_bw_4x4`
loop bounds to clamped versions. Test sites in
`reconstruct_inter.rs` get the obvious sentinel
(`mi_{rows,cols} = 1 << 16`) — the synthetic blocks don't sit near
frame edges, so the clamp is a no-op there.

**Why this didn't surface earlier:**

  - tile_1x2 has only ONE tile boundary (at col 960). The partial-
    bottom SBs near col 960 are inside tile 0, which starts at
    mi_col=0 — no tile-relative drift accumulates because tile 0's
    BoolReader sees the same bits libvpx's tile 0 sees. tile_1x4 /
    tile_1x8 have additional internal tile boundaries, so the
    failing SBs land inside tile 1 / 2 / 3, whose BoolReaders have
    their own buffer + cursor and accumulate drift independently.
  - The keyframe MD5s match bit-exactly today (filter_level > 0
    keyframes light up via the K2.c LF wireup) because the intra
    path in `reconstruct_intra_planes` (`tile.rs:1500-1700`)
    already clamps its TX iteration to the visible region via
    `n_4x4` adjustments — only the inter path's parallel loop in
    `reconstruct_inter_block` was missing the clamp.
  - D.23's entropy-parity finding ("every per-block syntax element
    cursor matches libvpx") looked at INTERMI logging which fires
    AFTER `read_inter_block_mode_info` returns — i.e. before the
    residual decode loop. The cursor was identical at the entry of
    every per-block trace point; the drift accumulates INSIDE
    decode_block_tokens for the invisible-edge TX cells, which the
    INTERMI trace doesn't cover.

**Localization path (for future "looks-like-recon-but-it's-parser" bugs):**

  1. `RVP9_DISABLE_LF=1` on both sides (rvp9 + libvpx-stock; D.17
     wired the env hook on libvpx). Diff pre-LF Y via
     `dump_frame_diff`. We confirmed pre-LF differs ⇒ ruled out LF
     as the bug surface.
  2. Add `RVP9_TRACE_MC_FILTER=row,colmin,colmax` to scope the MC
     trace to a specific SB band (`reconstruct_inter.rs`). Diff
     rvp9 vs libvpx MC events at the failing SBs. Different bsize
     /mv/ref at same `(mi_row, mi_col)` ⇒ ruled out MC kernel and
     pointed at *upstream* mode-info or partition divergence.
  3. Re-trace `RVP9_TRACE_INTERMI` for every decoded frame (not
     just the visible ones; libvpx's trace tags each frame with
     `f=N` so altrefs are visible). Diff per-frame. Frame 4 (the
     altref preceding frame 3) had a 5-bit cursor drift entering
     mi=(128, 104) — first divergence-point.
  4. Walk backwards from there. Cursor matched at mi=(134, 100)
     `post_modeinfo` (count=11 both sides), diverged at
     mi=(128, 104) `enter`. The only thing between those two
     events is **the residual decode for the previous block**.
     Inspect `reconstruct_inter_block`'s loop bounds vs libvpx's
     `predict_recon_inter` ⇒ missing clamp.

**Code touched:** `crates/rvp9-decoder/src/reconstruct_inter.rs`
(add fields + clamp), `crates/rvp9-decoder/src/tile.rs` (thread
`mi_rows` / `mi_cols`), `crates/rvp9-harness/tests/conformance.rs`
(swap ratchet for `assert_eq!(total_mismatched, 0)` + raise
`MIN_PASSES` to 191).

**Status: Stage 3 EXIT achieved. No mismatches across all 201
conformance frames.**
