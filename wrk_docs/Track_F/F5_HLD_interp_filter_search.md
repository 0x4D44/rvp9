# Track F.5 — HLD: Interpolation Filter Selection in the Inter Mode Picker

**Status:** Draft (HLD)
**Owner:** Arthur (with Claude)
**Date:** 2026-05-23
**Parent track:** F — convergence + structural closing of the foreman PSNR gap
**Depends on:** F.1 (block_yrd port), F.4 (calibration / parametric sweeps shipped)
**Blocks:** none (independent lever)
**Estimated effort:** medium (1–2 days of impl + verification, ~500–800 LoC)

---

## 1. Problem statement

The current rvp9 realtime encoder always uses `InterpFilter::EightTap` for every inter block, regardless of the residual / sub-pel pattern. This is documented at the top of the inter mode picker:

> `crates\rvp9-core\src\mode_decision\inter.rs:39-43`
>
> *"No interpolation-filter search: we always pick `EightTap` (libvpx's `INTERP_FILTER` default `EIGHTTAP = 0`). The decoder will read `interp_filter = EightTap` from the per-block header bits the encoder writes; switchable filter selection is an F8.x-deferred feature."*

This is a structural divergence from libvpx, which actively searches over the 8-tap family (and optionally bilinear) per inter block and picks the filter that minimises the picker's local RD cost. With the F.1 ship-state of `+9.00 dB` PSNR-delta on foreman (vs the libvpx baseline) and four exhaustive parametric sweeps (K1–K4, F.4-3, F.4-5, Hadamard) having demonstrated that parametric tuning cannot close the remaining gap, the next concrete remaining lever is structural: turn on per-block filter selection.

Hypothesis: on natural-motion content (foreman), the prediction residual for a non-trivial fraction of inter blocks is better-modelled by the smooth (`EIGHTTAP_SMOOTH`) or sharp (`EIGHTTAP_SHARP`) kernel than by the default regular 8-tap kernel. Restricting the encoder to a single filter therefore leaves recoverable PSNR on the table.

**Expected gain:** ~0.5–2 dB on foreman based on libvpx tuning history. This does *not* close the 8 dB gap on its own — but it is the most concrete remaining single lever, and the kernels and decoder support already exist (so the surface area of the change is small relative to the potential gain).

---

## 2. libvpx reference

Primary function: **`search_filter_ref`** in `D:\language\libvpx\vp9\encoder\vp9_pickmode.c:1488-1573`.

Signature:

```c
static void search_filter_ref(VP9_COMP *cpi, MACROBLOCK *x, RD_COST *this_rdc,
                              int mi_row, int mi_col, PRED_BUFFER *tmp,
                              BLOCK_SIZE bsize, int reuse_inter_pred,
                              PRED_BUFFER **this_mode_pred, unsigned int *var_y,
                              unsigned int *sse_y, int force_smooth_filter,
                              int *this_early_term, int *flag_preduv_computed,
                              int use_model_yrd_large);
```

Key behaviour (file:line annotations from libvpx v1.14.1):

- **Filter range** (`vp9_pickmode.c:1512-1514`):
  ```c
  INTERP_FILTER filter_start = force_smooth_filter ? EIGHTTAP_SMOOTH : EIGHTTAP;
  INTERP_FILTER filter_end = EIGHTTAP_SMOOTH;
  for (filter = filter_start; filter <= filter_end; ++filter) { … }
  ```
  Note: the realtime-path search loop iterates `EIGHTTAP (0) … EIGHTTAP_SMOOTH (1)` *by enum value*, but because of the enum ordering in `vp9_filter.h` (`EIGHTTAP=0, EIGHTTAP_SMOOTH=1, EIGHTTAP_SHARP=2, BILINEAR=3`), the loop actually covers `{EIGHTTAP, EIGHTTAP_SMOOTH}` in the realtime path. The non-realtime / RDOPT path (`vp9_rdopt.c`) extends to `EIGHTTAP_SHARP` as well. **Our v1 ports the realtime variant: we search 2 filters minimum, optionally 3 with SHARP as an extension knob.**
- **Per-filter inner loop** (`vp9_pickmode.c:1516-1530`):
  1. `mi->interp_filter = filter;`
  2. `vp9_build_inter_predictors_sby(...)` — re-runs sub-pel interpolation with the new kernel.
  3. `model_rd_for_sb_y_large` or `model_rd_for_sb_y` — picker's transform-domain rate/distortion model (rvp9 substitutes the F.1 SSE + block_yrd surrogate here).
  4. `pf_rate[filter] += vp9_get_switchable_rate(cpi, xd);` — add the per-block bits needed to *signal* the filter choice (only when `cm->interp_filter == SWITCHABLE`).
  5. `cost = RDCOST(x->rdmult, x->rddiv, pf_rate[filter], pf_dist[filter]);`
- **Best-cost tracking** (`vp9_pickmode.c:1531-1550`): keep the filter with lowest `cost`, remember `tx_size`, `skip_txfm`, and (if `reuse_inter_pred`) preserve the predicted-pixel buffer so it can be reused downstream without re-running interpolation.
- **Commit best filter** (`vp9_pickmode.c:1556-1572`): write `best_filter` back into `mi->interp_filter` and (if needed) re-run `vp9_build_inter_predictors_sby` so the downstream code uses the winning predicted pixels.

Caller in libvpx: `vp9_pick_inter_mode` invokes `search_filter_ref` (gated by `pred_filter_search` at `vp9_pickmode.c:1721`, set when `cm->interp_filter == SWITCHABLE` and the partition is large enough) before the final RD update.

**Filter kernel tables** (verified in `D:\language\libvpx\vp9\common\vp9_filter.c`):
- `sub_pel_filters_8` — `vp9_filter.c:28-38` — Lagrangian, mapped to `EIGHTTAP = 0`.
- `sub_pel_filters_8lp` — `vp9_filter.c:54-64` — low-pass / "smooth", mapped to `EIGHTTAP_SMOOTH = 1`.
- `sub_pel_filters_8s` — `vp9_filter.c:41-51` — DCT-based / "sharp", mapped to `EIGHTTAP_SHARP = 2`.
- `bilinear_filters` — `vp9_filter.c:15-25` — 2-tap, mapped to `BILINEAR = 3`.
- Dispatch: `vp9_filter_kernels[5]` array at `vp9_filter.c:79-82` — note the index ordering: `{ sub_pel_filters_8, sub_pel_filters_8lp, sub_pel_filters_8s, bilinear_filters, sub_pel_filters_4 }`, which matches the `INTERP_FILTER` enum.

Yes, the SMOOTH and SHARP coefficients live in `vp9/common/vp9_filter.c` (confirmed).

---

## 3. rvp9 current state

### 3.1 Bitstream / decoder surface

The bitstream already carries per-block `interp_filter` bits:

- `crates/rvp9-core/src/bitstream/frame.rs` defines `InterpFilter::{EightTap, EightTapSmooth, EightTapSharp, Bilinear, Switchable}`.
- The frame-header writer already supports `cm->interp_filter == Switchable`, in which case the per-block writer emits 1–2 bits to signal the choice.
- The decoder consumes these bits in the inverse direction — bitstream level filter selection is *already plumbed end-to-end*; the encoder side just never exercises it.

### 3.2 Filter kernels

Already present in `crates/rvp9-core/src/inter/filter.rs`:

| rvp9 symbol               | libvpx source                          | File:line |
|---------------------------|----------------------------------------|-----------|
| `EIGHTTAP_KERNEL`         | `sub_pel_filters_8`                    | `inter/filter.rs:57-83` |
| `EIGHTTAP_SMOOTH_KERNEL`  | `sub_pel_filters_8lp`                  | `inter/filter.rs:85-106` |
| `EIGHTTAP_SHARP_KERNEL`   | `sub_pel_filters_8s`                   | `inter/filter.rs:108-131` |
| `BILINEAR_KERNEL`         | `bilinear_filters`                     | `inter/filter.rs:133-155` |
| `kernel_for(InterpFilter)` | n/a — rvp9 helper that dispatches by enum | `inter/filter.rs:160-170` |

All four kernels have byte-exact unit-test coverage against the libvpx coefficients (e.g. `filter.rs:786-797`). **No new kernel code is required.**

### 3.3 The actual divergence

`crates/rvp9-core/src/mode_decision/inter.rs::pick_inter_mode_luma` and `evaluate_candidate` hard-wire `EIGHTTAP_KERNEL` (via the imports at line 95). The result of the picker — `InterModeSearchResult` (or whichever surface it returns to the partition walker) — has no `picked_interp_filter` field; the bitstream writer therefore defaults to `EightTap`.

So the change is local: add a filter loop, plumb the chosen filter into the picker result, and have the bitstream writer read from that result.

---

## 4. Proposed design (high-level)

### 4.1 Picker surface change

Add `picked_interp_filter: InterpFilter` to `InterModeSearchResult` (or whatever the picker returns to the caller). Default value remains `EightTap` so callers not touching filter selection see no behavioural change.

### 4.2 Filter loop placement

Inside `evaluate_candidate` (or the immediate caller in `pick_inter_mode_luma`), introduce an inner loop after the MV is fixed:

```text
for filter in [EightTap, EightTapSmooth, EightTapSharp]:
    1. Re-run sub-pel interpolation with `kernel_for(filter)`.
    2. Re-run residual / forward DCT / quant / inverse DCT / SSE
       (this is the existing F.1 block_yrd path, unchanged).
    3. Compute D + lambda*R as today, plus a small extra rate term for
       the switchable-filter signalling cost (see 4.4).
    4. Track the filter with lowest RD cost.

Commit best filter into the candidate's result.
```

The MV search (full-pel diamond + sub-pel refinement) stays **outside** the filter loop — it runs once with `EightTap`, same as libvpx's realtime path. Re-running MV search per filter would cube the cost for negligible quality benefit.

### 4.3 Filter set

Start with the libvpx realtime set: `{EightTap, EightTapSmooth, EightTapSharp}` — 3 filters. Skip `Bilinear` (libvpx's 4th option) — it's only useful for very-low-bitrate / very-fast modes and is explicitly deferred.

We extend one filter beyond libvpx's realtime-path range (`EIGHTTAP..=EIGHTTAP_SMOOTH`) to include `EIGHTTAP_SHARP` because rvp9 does not (yet) ship the speed-feature `force_smooth_filter` heuristic that libvpx uses to *avoid* trying SHARP on flat content. Including SHARP costs us one extra interp pass per candidate but lets the picker discover its own filter preference; the picker's RD math already penalises any filter that doesn't actually improve distortion.

### 4.4 Rate model for switchable signalling

When `cm->interp_filter == Switchable`, each inter block pays 1–2 bits to signal its filter (entropy-coded against a context derived from neighbour filter choices). libvpx computes this via `vp9_get_switchable_rate` (`vp9_pickmode.c:1528`).

For F.5 v1, use a **flat 1.5-bit approximation** (i.e. `1.5 * 512 = 768` in `vp9_prob_cost` units) per filter selection. This is a conservative over-estimate that ensures the picker only switches filters when the distortion improvement clearly justifies the bits. A faithful per-context switchable rate is an F.5 follow-up (it depends on neighbour state that the F.1 picker doesn't yet thread).

If `cm->interp_filter` is **not** `Switchable` (i.e. a frame-level fixed filter), skip the search entirely and use the fixed filter.

### 4.5 Bitstream writer

The writer reads `picked_interp_filter` from the picker result and emits the per-block filter bits in the switchable-filter path it already supports. Bit-exact symmetry with the decoder is already covered by the existing decode-roundtrip tests (which we should run with the new picker output to confirm).

### 4.6 What we are NOT changing

- MV search runs once at `EightTap` (see 4.2).
- TX size search — unchanged (still `max_txsize` capped at `Tx32x32`).
- Skip-txfm logic — unchanged.
- Chroma path — unchanged: chroma still inherits the picked luma filter.
- Frame-level `cm->interp_filter` decision — the encoder still picks `Switchable` per the existing frame-header logic; F.5 only changes the *block-level* choice within Switchable frames.

---

## 5. Realtime budget impact

Cost model:

- F.1 ship state: one full inter-candidate evaluation (interp + fdct + quant + idct + SSE) costs ~300 ns at 16x16 on the foreman test rig.
- Adding 2 extra filter trials per candidate (`EightTapSmooth`, `EightTapSharp`) re-runs steps 1–6 of the algorithm in §4.2, i.e. 3× the per-candidate cost.
- Per-candidate cost: ~300 ns × 3 = ~1 µs.
- Foreman has roughly 600 inter candidates per frame (4 modes × ~150 partitions). Today's per-frame inter-picker cost: ~0.2 ms. After F.5: ~0.5–0.7 ms.

Net per-frame impact: **+0.3–0.5 ms** on top of today's ~5 ms median frame time. That's a **6–10% realtime budget hit** — comfortably inside the 33 ms realtime gate.

The largest risk to the budget is that re-running interpolation also re-touches L1/L2 cache lines; we may see super-linear cost on some block sizes. The F.0.1 realtime gate will catch this.

---

## 6. Test plan

### 6.1 Existing tests (should pass unchanged)

- All kernel unit tests in `inter/filter.rs` — these already validate byte-exact match against libvpx (no change here).
- Decode-roundtrip tests on synthetic streams — must still pass; the decoder reads whatever filter the encoder writes.

### 6.2 New tests

1. **Picker filter-choice smoke test.** Construct a synthetic frame whose residual is genuinely better-matched by `EightTapSmooth` (e.g. a gradient-blur block) and assert the picker selects `EightTapSmooth` for at least one leaf. Do the same for `EightTapSharp` with a sharp-edge synthetic. This is the "filter loop is wired and exercised" assertion.
2. **Cross-decoder gate.** Encode a short clip with F.5 enabled, decode with `vpxdec` (libvpx reference decoder), confirm bit-exact pixel match. This is the load-bearing test — if our SMOOTH/SHARP kernels have any coefficient drift, vpxdec will produce different pixels and the test fails.
3. **No-switchable regression.** With `cm->interp_filter` forced to `EightTap`, confirm the picker skips the loop and the output is bit-identical to pre-F.5.

### 6.3 Quality / performance gates

- **F.0.1 PSNR-delta gate on foreman.** Run pre/post: expect a delta improvement of 0.5–2 dB (i.e. ship-state moves from `+9.00 dB` toward `+7–+8.5 dB`). If the delta moves the *wrong way*, that's a sign of a sign error in the rate or kernel indexing and we revert.
- **Realtime gate.** Frame time must stay ≤ 33 ms (p95). Expect ~5.5 ms median (was ~5 ms).
- **Decode-bit-exact gate.** All existing bit-exact tests pass.

### 6.4 Synthetic-test fallout

Bit-exact synthetic tests that lock in encoder output (e.g. `encode_synthetic_two_frame_kf_inter_non_skip` and friends) will likely produce *different* output after F.5 because the picker may now choose a non-EightTap filter on some candidates. Plan to update the expected fixtures *after* manual inspection confirms the new output decodes correctly through vpxdec. **Not** before — flipping fixtures before cross-decoder validation hides the very bug we'd be looking for.

---

## 7. Risks

### 7.1 Kernel coefficient drift (high impact / low probability)

The SMOOTH and SHARP kernel coefficients must match libvpx **byte-exactly** or the decoder produces wrong pixels (silently — no error path, just bad PSNR). rvp9 already has byte-exact kernel tests against libvpx, so this risk is largely mitigated; the residual risk is that the kernel *tables exist but aren't wired through `kernel_for`*. The cross-decoder gate (6.2 item 2) is the catch-all.

Reference for coefficients:
- `D:\language\libvpx\vpx_dsp\vpx_filter.h` — `InterpKernel` typedef, `SUBPEL_TAPS`, `SUBPEL_BITS`.
- `D:\language\libvpx\vp9\common\vp9_filter.c:15-82` — coefficient tables and dispatch array.

### 7.2 Cross-decoder gate breakage (high impact / medium probability)

If our switchable-filter signalling bits don't match the decoder's expectation (wrong context, wrong order, wrong probability model), vpxdec will decode garbage. The bitstream writer's switchable-filter path is already exercised by decoder-roundtrip tests, but those tests have only ever fed `EightTap` — so we will be the first to drive the path with non-default values. Expect to find at least one wiring bug here.

Mitigation: encode a *single* block with `EightTapSmooth` first, decode with vpxdec, validate pixels. Only then expand to full-frame tests.

### 7.3 Synthetic-test churn (low impact / high probability)

Existing bit-exact synthetic tests will need updating because filter choice changes encoder output. This is mechanical but tedious — and risky if we update fixtures before cross-decoder validation passes.

### 7.4 Realtime budget creep (low impact / low probability)

Per §5, we expect a 6–10% budget hit, well within margin. The risk is cache-pressure non-linearity on small block sizes. The F.0.1 realtime gate catches this.

### 7.5 Quality regression (low impact / low probability)

If the switchable-rate approximation in §4.4 is too aggressive (under-charges bits), the picker may over-switch filters, paying more rate than the distortion improvement is worth. We'd see this as a *negative* PSNR-delta improvement (the gap widens). Detection is automatic via F.0.1.

---

## 8. Out of scope

- **`Bilinear` (libvpx's 4th filter).** Defer to F.5-followup.
- **Frame-header-level `interp_filter` decision.** Already supported in the writer; this HLD only touches block-level choice within Switchable frames.
- **Per-context switchable-rate model.** Use a flat 1.5-bit approximation; a faithful neighbour-context model is an F.5 follow-up.
- **MV search per filter.** Single MV search at `EightTap`, then filter loop. (libvpx-realtime does the same.)
- **Chroma-specific filter choice.** Chroma inherits luma's filter (libvpx-standard behaviour).
- **`force_smooth_filter` speed-feature heuristic.** libvpx skips the SHARP filter on visually-flat content for speed; rvp9 will always try all three. Lift this heuristic if/when the realtime budget gets tight.
- **`reuse_inter_pred` predicted-buffer optimisation.** libvpx avoids re-running interpolation by holding onto the best filter's predicted pixels. rvp9 v1 re-runs interpolation; the buffer-reuse optimisation is a perf follow-up.

---

## 9. Open questions

1. **Is `EightTapSharp` worth the 33% extra cost over the libvpx-realtime 2-filter set?** Answer empirically on foreman: if SHARP wins on <2% of blocks, drop it to match libvpx-realtime's filter range.
2. **Switchable-rate model fidelity.** Is the flat 1.5-bit approximation good enough, or do we need a neighbour-context model to converge on libvpx's PSNR? Decide after F.0.1 measurement.
3. **Where exactly does the `picked_interp_filter` live?** `InterModeSearchResult` is the obvious surface, but the picker may already have multiple return-type surfaces (per-leaf vs per-partition). Confirm during the L-LD pass.
4. **Test-fixture churn budget.** How many existing synthetic-encode-roundtrip tests will need updating? Survey before starting impl — if it's >10, batch the fixture updates into a separate commit.

---

## 10. Done definition

F.5 is done when:

- The picker selects per-block from `{EightTap, EightTapSmooth, EightTapSharp}` on every inter block in Switchable frames.
- The bitstream writer emits the chosen filter in the per-block switchable-filter bits.
- vpxdec decodes our output bit-exactly.
- F.0.1 PSNR-delta on foreman shows ≥ 0.5 dB improvement.
- Realtime gate passes (≤ 33 ms p95).
- All existing tests pass (with fixture updates where necessary, justified by manual cross-decoder validation).
