# F.1 — HLD: Port libvpx `block_yrd` (coefficient-bit-aware transform-domain RD) into rvp9's picker

| Field | Value |
|---|---|
| Track | F (nonrd_pickmode port) |
| Sub-track | F.1 (block_yrd — primary lever) |
| Author | F.1 HLD-writer agent |
| Status | LANDED — infrastructure-only (see §14 ship-state addendum, 2026-05-23) |
| Last revised | 2026-05-23 |
| Repo HEAD at writing | `f1a8327` (post-F.0 foundations) |
| libvpx HEAD at writing | `12f3a2ac6` (`v1.14.1`) at `D:\language\libvpx` |
| Dependencies | F.0.1 (PSNR-delta gate), F.0.2 (per-block oracle) — both signed-off and landing |
| Blocks | F.2 (`find_mv_refs` + A18 default), F.3 (motion-aware partition), F.4 (tuning + byte-exact re-establishment) |

**Sign-off:** 2026-05-23 — Track F team-lead approved this HLD with revisions 1–9. Q1–Q7 resolved (see §10).

> The picker's `evaluate_candidate` currently sees only pixel-domain SSE
> after the forward/inverse round-trip plus `mode_rate + mv_rate`. NEWMV's
> residual-bit savings are invisible. F.1 introduces a coefficient-domain
> rate model that makes the savings visible, by porting libvpx's
> `block_yrd` from `vp9/encoder/vp9_pickmode.c:718` (v1.14.1).

---

## 1. Problem statement & motivation

After Track E (E1–E16) landed and rate-control / structural confounds were
closed, the foreman PSNR-delta at matched bitrate sits at **mean +9.07 dB
(max +9.50, min +8.65)** — `wrk_docs/Track_F_baseline_numbers.md`. The
E9 matched-Q oracle attributed the residual gap to **prediction quality,
not transform / quant / entropy**: at the same Q, rvp9's reconstruction
PSNR is ~31 dB vs libvpx's ~39 dB.

F.0.3 (`a18_prev_frame_mvs_fires`) confirmed the mechanism end-to-end: the
picker emits `(NEARESTMV, LASTFRAME, (0,0))` on consecutive inter frames
even on textured-motion content. The reason is in
`crates/rvp9-core/src/mode_decision/inter.rs:648-678` and the preamble
lines 80-88: `evaluate_candidate` builds `rd_cost = (R * lambda) >> 9 +
(D << 7)` where `R = mode_rate + mv_rate + ref_frame_rate_cost`. The
**residual rate component is absent**, so a NEWMV candidate with a
strictly better prediction but ~5000 prob-cost units of `mv_rate` always
loses to a rate-free NEARESTMV anchored at `(0,0)`. The result is large
residuals everywhere, coefficient bits dominate the bitstream, and PSNR
collapses.

This is also why F.0.3 (A18 prev-frame-MVs) had to land default-OFF: with
prev-frame-MVs activated, NEARMV gets a non-zero anchor and starts
*winning* against refined NEWMV, regressing testsrc 240 pp BD-rate
(`notes/divergence-ledger.md:676-697`). The picker has no RD signal to
distinguish "rate-free anchor with a worse prediction" from
"rate-paying NEWMV with a better prediction".

**F.1 is the primary Track F lever.** F.2 (`find_mv_refs` proper,
prev-frame-MVs default-ON) and F.3 (motion-aware partition) are coupled
but cannot deliver signal until block_yrd makes residual savings
RD-visible. See HLD F.0.3 §2.5 for the exact regression mechanism.

The goal: drive foreman mean PSNR-delta from +9.07 toward the
curve-overlap threshold (~+1 dB, where BD-rate becomes measurable again).
F.1 alone is not expected to close the full gap — it is the prerequisite
that unblocks F.2 / F.3 (see §9).

---

## 2. libvpx reference walkthrough — the meat

All paths absolute. Line numbers verified against `D:\language\libvpx`
HEAD `12f3a2ac6` (v1.14.1). The rvp9 picker preamble cites
`vp9_pickmode.c:1056-1110` for `block_yrd`; the **live v1.14.1 location
is `vp9_pickmode.c:718`**. (The `:1056-1110` block at HEAD is actually
the intra-mode branch's call site, which calls `block_yrd` indirectly via
`estimate_block_intra` — see §3 below.)

### 2.1 `block_yrd` function signature

`D:\language\libvpx\vp9\encoder\vp9_pickmode.c:718-844`:

```c
static void block_yrd(VP9_COMP *cpi, MACROBLOCK *x, RD_COST *this_rdc,
                      int *skippable, int64_t *sse, BLOCK_SIZE bsize,
                      TX_SIZE tx_size, int rd_computed, int is_intra);
```

| Parameter | Meaning |
|---|---|
| `cpi` | encoder context (read-only here: speed-features + frame_type only) |
| `x` | macroblock — the residual / coef / qcoeff / dqcoeff scratch buffers, the quantizer state on `x->plane[0]`, and `x->skip_block` (asserted off) |
| `this_rdc` | output `RD_COST { rate, dist, sse, skip, rdcost }` |
| `skippable` | output bool — `1` iff every block's `eob == 0` |
| `sse` | in/out — caller passes the model-domain SSE from `model_rd_for_sb_y`; `block_yrd` returns either `INT_MAX` (simple path) or the same value rescaled for downstream skip-vs-non-skip RD compare |
| `bsize` | block size (e.g. `BLOCK_16X16`) |
| `tx_size` | transform size — capped at `TX_16X16` by the caller (line 2350: `VPXMIN(mi->tx_size, TX_16X16)`) |
| `rd_computed` | `1` iff the caller already populated `this_rdc->rate`/`dist` via `model_rd_for_sb_y` (matters only on the simple-yrd shortcut path) |
| `is_intra` | `1` for intra calls; affects the simple-yrd shortcut's `model_rd_for_sb_y` invocation |

### 2.2 What it computes (the body, line by line)

**Step A — simple-yrd shortcut** (lines 737-749). Gated on
`cpi->sf.use_simple_block_yrd && frame_type != KEY_FRAME && (bsize <
BLOCK_32X32 || (svc-specific gates))`. In that case `block_yrd` is just
a thin wrapper around `model_rd_for_sb_y`: if `!rd_computed`, call
`model_rd_for_sb_y(...)` to fill `this_rdc->rate / dist`, set
`*sse = INT_MAX`, `*skippable = 0`, return.

`use_simple_block_yrd` is set in `vp9_speed_features.c:791` (`speed >=
8`) and `:713` (SVC non-base layers). **At cpu-used 7 on a single-layer
encode (the foreman baseline) `use_simple_block_yrd = 0`** — the full
Hadamard path runs. F.1's port must match the cpu-used-7 behaviour,
which is the full path. The simple-yrd shortcut is out of scope until
F.1.1 cpu-used-8 tuning.

**Step B — assertions** (lines 754, 784). `tx_size != TX_32X32` —
block_yrd never runs at 32×32 (caller caps to 16×16). `!x->skip_block`
— per-block skip is decided earlier.

**Step C — residual subtraction** (lines 765-766). `vpx_subtract_block(bh,
bw, p->src_diff, bw, src.buf, src.stride, dst.buf, dst.stride)`. Writes
`p->src_diff = src - pred` as `int16_t`.

**Step D — forward transform + quantize, per 4×4-stepped sub-block**
(lines 771-809). Iterates `(r, c)` over the block in `block_step`
units (`block_step = 1 << tx_size`, so for TX_16X16 it's `4`, meaning
*one* iteration covering the whole 16×16). For each iteration:

- `coeff = BLOCK_OFFSET(p->coeff, block)`, `qcoeff =
  BLOCK_OFFSET(p->qcoeff, block)`, `dqcoeff =
  BLOCK_OFFSET(pd->dqcoeff, block)`.
- **Forward transform**: `vpx_hadamard_16x16` / `vpx_hadamard_8x8` /
  `x->fwd_txfm4x4` (`vp9_fwht4x4` for lossless else `vpx_fdct4x4`).
  Note: `block_yrd` uses the **Hadamard transform**, NOT the standard DCT,
  at 8×8 / 16×16. The Hadamard is cheaper and is a model — exact DCT/quant
  reserved for the post-pick encode path.
- **Forward quantize**: `vp9_quantize_fp(coeff, n_coeffs, p, qcoeff,
  dqcoeff, pd->dequant, eob, scan_order)`. `vp9_quantize_fp` (not
  `vp9_quantize_b`) — the lower-overhead variant without the zbin
  pre-test.
- `*skippable &= (*eob == 0)`. `eob_cost += 1` (one per 4×4 sub-block
  written).

**Step E — early-exit for skippable blocks** (lines 812-818). If all
`eob == 0` then `this_rdc->dist = *sse` (after rescaling
`*sse = (*sse << 6) >> 2 = *sse << 4`); `this_rdc->rate` is left at 0
and the function returns. Note: at the early-exit, `*sse` is whatever
the caller passed in (e.g. `(int64_t)sse_y` from `model_rd_for_sb_y`).
The bitwise `<< 6 >> 2` is equivalent to `<< 4`.

**Step F — entropy rate estimate + distortion** (lines 820-839). Second
pass over the same sub-blocks:

- **Rate**: `*eob == 1 ? rate += abs(qcoeff[0]) : eob > 1 ? rate +=
  vpx_satd(qcoeff, step << 4)`. **The rate estimate is the SAD of the
  quantized coefficients** (sum of absolute values, `vpx_satd_c` at
  `vpx_dsp/avg.c:334-341`). Not a probability-tree walk — a much
  cheaper proxy. The intuition: quantized-coefficient magnitude is a
  monotone proxy for the eventual bool-coder bits because higher
  magnitudes burn more category tokens + sign + extra bits.
- **Distortion**: `dist += vp9_block_error_fp(coeff, dqcoeff, step <<
  4) >> 2`. `vp9_block_error_fp` (`vp9_rdopt.c:334-345`) is
  `sum((coeff[i] - dqcoeff[i])^2)` over the 4×4-stepped sub-block:
  **the squared quantization error in coefficient domain** (Parseval-
  equivalent to pixel-domain SSE up to the transform's energy scale).
  The `>> 2` accounts for libvpx's 4× transform-coefficient scaling.

**Step G — scale the rate into `vp9_prob_cost` units** (lines 842-843):

```c
this_rdc->rate <<= (2 + VP9_PROB_COST_SHIFT);
this_rdc->rate += (eob_cost << VP9_PROB_COST_SHIFT);
```

`VP9_PROB_COST_SHIFT = 9` (`vp9_cost.h:24`). The `<< 2` converts the
sub-block SATD sum into a per-coefficient-bit estimate; the `<< 9`
puts it on the same 1/512-bit scale as `mode_rate` and `mv_rate`. The
`eob_cost << 9` adds one bit (in 1/512 units) per emitted sub-block —
a coarse estimate of the EOB-token cost.

### 2.3 Outputs

`RD_COST { rate, dist, sse, skip, rdcost }`:

| Field | Source | Units / scale |
|---|---|---|
| `rate` | sum of sub-block SATD(qcoeff) × 4, scaled to `vp9_prob_cost` (1/512-bit) | 1/512-bit |
| `dist` | sum of sub-block block_error_fp(coeff, dqcoeff) >> 2 | coefficient-domain squared error |
| `*sse` (out-param) | caller's input shifted `<< 4` (only consumed by caller for the skip-vs-non-skip RDCOST compare at line 2355) | pixel-domain SSE × 16 |
| `*skippable` | all-EOB-zero flag | bool |

**Important — `sse` here is post-prediction, NOT post-reconstruction**.
The caller computes pixel-domain `sse_y` via `model_rd_for_sb_y` (using
`source - prediction`, the variance function in
`vp9_pickmode.c:651-652`), then passes it to `block_yrd`. `block_yrd`
itself does not re-derive `sse` — it just rescales the caller's value.
This is the **second** distortion path: the caller uses `sse` as the
"would-be skip distortion" in the `skip_txfm` RD comparison at line
2355 (`RDCOST(rdmult, rddiv, this_rdc.rate, this_rdc.dist) <
RDCOST(rdmult, rddiv, 0, this_sse)` — i.e. "is it cheaper to encode the
residual or to skip it and accept SSE?").

### 2.4 The `block_yrd_txfm` inner — there isn't one

Inspection of the v1.14.1 source confirms `block_yrd` is a single
function with no `block_yrd_txfm` sub-helper. The transform-and-quant
pipeline is inlined per `tx_size` switch (lines 786-803).

### 2.5 Entropy-cost estimation: model, not exact rate

**`block_yrd` does NOT call `vp9_cost_coeffs`.** The rate is `4 * SATD(qcoeff)`
in `vp9_prob_cost` units. This is intentionally a coarse model — the
full coef-prob-tree cost (which rvp9 already has at
`crates/rvp9-core/src/cost/coef_cost.rs`, `CoefCostTable`) is reserved
for the RDO path (`vp9_rdopt.c:cost_coeffs`). The nonrd-realtime path
trades accuracy for ~10–20× lower per-coefficient cost.

The implication for F.1: **we do NOT need to plumb `CoefCostTable` into
the picker.** We only need a SATD-of-quantized-coefficients accumulator,
which rvp9 doesn't currently expose but which is a 4-line function (see
§6).

### 2.6 `model_rd_for_sb_y` vs `block_yrd` — which one does the realtime path call?

This is the key question for choosing the right port target. The realtime
nonrd picker (`vp9_pick_inter_mode`, `vp9_pickmode.c:~1750` onward)
calls **both, in sequence**:

1. **`model_rd_for_sb_y` / `model_rd_for_sb_y_large`** first — pixel-
   variance-domain Laplacian-distribution model
   (`vp9_pickmode.c:2330,2335`). Outputs an *initial* rate / dist /
   skip_txfm prediction. Cheap (variance + LUT). Also decides
   `x->skip_txfm[0]` short-circuit (`SKIP_TXFM_AC_DC` etc).
2. **`block_yrd`** second, **only if `!this_early_term`** (line 2347-
   2350). Refines the rate with the actual coefficient-domain measure.
3. **The skip-vs-non-skip RDCOST tie-break** (lines 2355-2362):
   ```c
   if (RDCOST(rdmult, rddiv, this_rdc.rate, this_rdc.dist) <
       RDCOST(rdmult, rddiv, 0, this_sse)) {
     this_rdc.rate += vp9_cost_bit(skip_prob, 0);
   } else {
     this_rdc.rate = vp9_cost_bit(skip_prob, 1);
     this_rdc.dist = this_sse;
     x->skip_txfm[0] = SKIP_TXFM_AC_DC;
   }
   ```

**For F.1, we port `block_yrd`, NOT `model_rd_for_sb_y`.** Justification:

- The rvp9 picker already does the full forward/inverse round-trip per
  candidate (`reconstruct_via_round_trip` at `inter.rs:821-976`). It
  has the qcoeff/dqcoeff buffers in scope already. Reusing them for
  SATD + block_error is essentially free.
- `model_rd_for_sb_y` is a *variance-domain* model — it estimates rate
  from `sse - var` (DC component) and `var` (AC component) via
  `vp9_model_rd_from_var_lapndz`. Porting it would require an entire
  separate Laplacian-distribution rate-distortion LUT (`rate_tab_q10`
  / `dist_tab_q10`, see PLAN.md Stage 1 follow-ups). Rvp9 does not
  have those tables.
- The chief virtue of `model_rd_for_sb_y` is *cheapness as a triage
  step before block_yrd*. F.1 ships block_yrd alone; the cheap model
  is an F.1.1+ optimization (see §8).
- `block_yrd` is what makes coefficient bits visible. That is the
  primary lever and the whole point of F.1.

§8 covers this trade-off explicitly as the headline open question.

---

## 3. Callers — how the RD model fits into libvpx's picker

The mode-eval loop is `vp9_pick_inter_mode` in `vp9_pickmode.c`. The
relevant outer structure (lines 1750-2500ish):

```text
for (ref_frame in usable_ref_frames) {            // LAST / GOLDEN / ALTREF
  for (this_mode in {NEAREST, NEAR, ZERO, NEW}) {
    [setup MV, build inter prediction]
    if (use_model_yrd_large)
      model_rd_for_sb_y_large(...)                // pixel-variance model
    else
      model_rd_for_sb_y(...)                      // pixel-variance model
    if (!this_early_term) {
      this_sse = sse_y;
      block_yrd(this_rdc, ..., this_sse, ...)     // coef-domain refinement
      [skip-vs-non-skip RDCOST tie-break]
    } else {
      this_rdc.rate += cost_bit(skip_prob, 1)
    }
    [add chroma rate / dist via model_rd_for_sb_uv if color-sensitive]
    this_rdc.rate += rate_mv
    this_rdc.rate += inter_mode_cost[ctx][mode]
    this_rdc.rate += ref_frame_cost[ref_frame]
    this_rdc.rdcost = RDCOST(rdmult, rddiv, this_rdc.rate, this_rdc.dist)
    [vp9_NEWMV_diff_bias, encode_breakout_test]
    [thresh-frequency-factor update]
    if (this_rdc.rdcost < best_rdc.rdcost) best_rdc = this_rdc
  }
}
```

**Key points for F.1 integration design:**

- `block_yrd`'s output is composed with `mode_rate`, `mv_rate`, and
  `ref_frame_rate_cost` *additively in the rate domain* — they all
  end up in the same `this_rdc.rate` before `RDCOST`. That matches
  the rvp9 picker's existing arithmetic (`inter.rs:676-680`).
- `rd_thresh` short-circuits (gating which modes are even tried) are
  in `rd_less_than_thresh_row_mt` (`vp9_pickmode.c:1112-1117`). These
  are speed-feature thresholds, not block_yrd-specific. Out of F.1
  scope (deferred to F.4 tuning).
- `skip_txfm[]` short-circuits (lines 2351-2362) are entangled with
  the skip-vs-non-skip RDCOST tie-break and the `model_rd_for_sb_y`
  skip-detection path. F.1 needs the **tie-break** (otherwise we
  can't decide which candidates ship skip=true vs skip=false), but
  can defer the model-based skip *short-circuit* to F.4.

---

## 4. rvp9 integration point

All paths absolute. Line numbers from `main @ f1a8327`.

### 4.1 The leaf-decision call site (the caller)

`D:\language\rvp9-2\crates\rvp9-encoder\src\lib.rs:2660-3035` — within
`pick_and_simulate_inter_encode_partitions`. At line 2987 a closure
`pick_for_ref` builds `InterModeSearchParams` and calls
`pick_inter_mode_luma(&params)` once per ref frame. The picker's output
is consumed at lines 3032-3034 (`pick_last`, `pick_golden`). The
argmin-across-refs lives below (not shown here) and writes the winning
mode/MV into `sb_partitions[...]`.

**F.1 does not change this site.** The integration happens **one level
down**, inside `evaluate_candidate`.

### 4.2 The current per-candidate RD computation (where F.1 plugs in)

`D:\language\rvp9-2\crates\rvp9-core\src\mode_decision\inter.rs:545-689`
— `evaluate_candidate`. The pertinent steps today:

1. **Lines 555-566**: build prediction `pred[0..bw*bh]` via
   `build_inter_prediction`.
2. **Lines 568-574**: copy `pred` into `recon`, then
   `reconstruct_via_round_trip(params, &pred, &mut recon, bw, bh,
   tx_size)` runs the forward DCT + forward quant + inverse DCT and
   writes the reconstructed block. **This is the load-bearing
   transform-quant pipeline F.1 will tap into.** The internal scratch
   buffers `coefs`, `qcoeff`, `dqcoeff` are local to
   `reconstruct_via_round_trip` and are dropped at function end (lines
   821-976).
3. **Line 604**: `sse = sse_block(src, recon, ...)` — pixel-domain
   `sum((src - recon)^2)`. This is `D` in the current RD.
4. **Lines 658-678**: `rate = mode_rate + mv_rate +
   ref_frame_rate_cost`. **No coefficient-rate component.**
5. **Line 680**: `this_rd = rd_cost(lambda, rate, dist)`.

The two surfaces F.1 has to change:

- **`reconstruct_via_round_trip` needs to surface qcoeff + dqcoeff
  + eob** so `evaluate_candidate` can compute SATD(qcoeff) and
  block_error(coeff, dqcoeff). Currently those are inner scratch
  arrays.
- **`evaluate_candidate` needs to add a `coef_rate` term** to the
  rate sum and (per libvpx tie-break) optionally swap `dist` between
  pixel-SSE-of-recon and coefficient-block-error.

### 4.3 The reconstruction tile loop already has everything we need

`inter.rs:861-974` tiles a block into `(bw/tx_side) × (bh/tx_side)`
transforms. For Block16x16 / Block32x32 that is one transform; for
Block64x64 it is four 32×32 transforms. Each iteration:
- builds `residual = src - pred`,
- runs `fdct{8,16,32}` into `coefs[..n]`,
- runs `vp9_quantize_b{,_32x32}` into `qcoeff[..n]` / `dqcoeff[..n]`
  with `eob` returned,
- runs `idct{8,16,32}_add` to write the reconstruction tile.

**F.1's coef-domain SATD + block-error accumulator slots in between
the quant and the inverse-DCT.** Each per-tile chunk contributes one
SATD(qcoeff[..n]) and one block_error(coeffs[..n], dqcoeff[..n]) to
the running totals; the inverse DCT continues to write `recon` so
pixel-SSE is still computable for the F.4 tie-break.

---

## 5. Proposed rvp9 design

### 5.1 New module: `crates/rvp9-core/src/mode_decision/block_yrd.rs`

Justification for a separate module (not a free function in `inter.rs`):

- `inter.rs` is already 1700+ lines. The `block_yrd` body (~80 lines of
  Rust, well-commented) deserves its own home with its own libvpx-faithful
  module doc-comment.
- It will be reused unchanged by F.x's future intra path: libvpx's
  `block_yrd` is called from both inter (`vp9_pickmode.c:2349`) and
  intra (`vp9_pickmode.c:1073`, via `estimate_block_intra`). Putting it
  in a sibling module to `inter.rs` / `intra.rs` matches that layout.
- It pairs naturally with `cost/coef_cost.rs` (the existing exact-rate
  estimator) — making it easy to A/B them at F.4.

### 5.2 Function signature

```rust
/// Coefficient-bit-aware transform-domain RD estimate for one Y plane
/// block. Mirrors libvpx `block_yrd` (`vp9_pickmode.c:718-844`,
/// v1.14.1, 12f3a2ac6).
///
/// `qcoeff_tiles` and `dqcoeff_tiles` are the per-tile output of the
/// picker's existing forward DCT + quantize pipeline. The caller must
/// also supply the un-quantized `coeff_tiles` (for the block_error_fp
/// dist computation) and the `eob_tiles` (for the skip flag and the
/// rate's EOB-cost term).
///
/// Returns an estimate of `(coef_rate, coef_dist, skippable)` where
/// `coef_rate` is in `vp9_prob_cost` units (1/512-bit) matching the
/// scale of `mode_rate` and `mv_rate`, and `coef_dist` is the
/// coefficient-domain squared error (suitable for `rd_cost`'s D term).
///
/// The caller composes the final RD as
/// `rd_cost(lambda, mode_rate + mv_rate + ref_frame_rate_cost +
/// coef_rate, coef_dist)` — see `inter.rs` integration.
pub fn block_yrd(
    coeff_tiles: &[&[i32]],       // n_tiles entries, each tx_side*tx_side
    qcoeff_tiles: &[&[i16]],
    dqcoeff_tiles: &[&[i16]],
    eob_tiles: &[u16],
    tx_size: TxSize,
) -> BlockYrdResult;

pub struct BlockYrdResult {
    /// Coefficient-bit rate estimate in vp9_prob_cost units.
    pub coef_rate: u32,
    /// Coefficient-domain squared error: sum_i (coeff[i] - dqcoeff[i])^2.
    /// Accumulated as i64 to avoid overflow on saturated 32×32 quant-error
    /// blocks (Q7 resolution). Caller saturate-casts to u32 at the rd_cost
    /// call site.
    pub coef_dist: i64,
    /// All tiles had eob == 0 (residual was fully quantized to zero).
    pub skippable: bool,
}
```

### 5.3 Algorithm (libvpx-faithful, bullet-for-bullet)

For each `(qcoeff_tile, dqcoeff_tile, coeff_tile, eob)` quadruple:
1. Increment `eob_cost += 1` (per-tile EOB-token charge, libvpx line
   805).
2. If `eob == 1`: `rate_satd += abs(qcoeff[0])` (libvpx line 831).
3. Else if `eob > 1`: `rate_satd += satd(qcoeff[0..n_coeffs])` —
   `sum |qcoeff[i]|` (libvpx line 833, `vpx_satd_c` at
   `vpx_dsp/avg.c:334-341`).
4. `dist += block_error_fp(coeff[0..n], dqcoeff[0..n]) >> 2` where
   `block_error_fp` is `sum (coeff[i] - dqcoeff[i])^2` (libvpx line
   835, `vp9_rdopt.c:334-345`). The `>> 2` is libvpx's
   transform-scale normalization.
5. Update `skippable &= (eob == 0)`.

After the loop:
1. `rate = (rate_satd << 2) << VP9_PROB_COST_SHIFT` = `rate_satd << 11`
   — libvpx line 842, `<<(2 + 9)`. The `<< 2` is the rate-per-coef
   weight, the `<< 9` puts it on prob_cost scale.
2. `rate += eob_cost << VP9_PROB_COST_SHIFT` = `eob_cost << 9` —
   libvpx line 843. One bit (in 1/512 units) of EOB-cost per tile.
3. Return `BlockYrdResult { coef_rate: rate, coef_dist: dist,
   skippable }`.

**One libvpx semantic NOT ported in F.1**: the simple-yrd shortcut
(libvpx lines 737-749). That gate is only active at `speed >= 8` or for
SVC non-base layers; at cpu-used 7 single-layer (the foreman baseline),
it's off. Documented in §8 as an F.1.1+ deferral.

### 5.4 Wiring into `reconstruct_via_round_trip`

`reconstruct_via_round_trip` (`inter.rs:821-976`) currently consumes
the per-tile qcoeff/dqcoeff internally and discards them. F.1 changes
the function to **either return per-tile coef arrays** or to **invoke
block_yrd inline before discarding them**. The latter is preferred
(keeps the API narrow and avoids returning slice-of-slice from a
helper).

**Extend `reconstruct_via_round_trip`'s return type from `()` to `BlockYrdResult` — keep the name.** Renaming ripples through callers and obscures the change in `git log`. The function already runs forward DCT + quant + inverse DCT; surfacing the coefficient-domain accumulator alongside the reconstruction is a return-type extension, not a new concern.

The function signature becomes:

```rust
fn reconstruct_via_round_trip(
    params: &InterModeSearchParams<'_>,
    pred: &[u8],
    recon: &mut [u8],
    bw: usize,
    bh: usize,
    tx_size: TxSize,
) -> BlockYrdResult;
```

The existing caller (`evaluate_candidate`) destructures the result and uses `recon` (for pixel-SSE), `coef_rate` (added to the rate sum), and `skippable` (for the libvpx skip-flag tie-break in Q5). `coef_dist` is computed and returned but **not consumed** by the F.1 `rd_cost` per Q3 — kept available for F.4 A/B.

Per-tile, inside the existing loop:
- After the quant call (which already produces `qcoeff[..n]`,
  `dqcoeff[..n]`, `eob`):
  - Accumulate `eob_cost += 1`.
  - If `eob == 1`: `rate_satd += abs(qcoeff[0] as i32) as u32`.
  - Else if `eob > 1`: `rate_satd += satd_i16(&qcoeff[..n])`.
  - `dist += block_error_fp_i32(&coefs[..n], &dqcoeff[..n]) >> 2`.
  - `skippable &= (eob == 0)`.
- After the loop: scale and return as in §5.3.

`satd_i16` and `block_error_fp_i32` are new free helpers — see §6.

### 5.5 Wiring into `evaluate_candidate`

`inter.rs:545-689`. The `rate` line at 676-680 becomes:

```rust
// rate = mode + mv + ref_frame + coef
let rate = mode_rate
    .saturating_add(mv_rate)
    .saturating_add(params.ref_frame_rate_cost)
    .saturating_add(block_yrd_result.coef_rate);
```

**The `dist` term remains pixel-SSE-of-recon** (computed as today via `sse_block(src, recon, ...)` at line 604). This is a **deliberate F.1 divergence from libvpx's `block_yrd` flow** (which uses `block_error_fp >> 2` in coefficient domain). Justification:
- Changing the meaning of `dist` would change `rd_cost(lambda, rate, dist)`'s rate-distortion ranking on every candidate, which requires lambda recalibration as a side effect.
- The two values are Parseval-equivalent up to transform energy scale, so the F.1 first-cut keeps the pixel-domain `dist` that the F8.0 contract already documents.
- The implementer surfaces `coef_dist` in `BlockYrdResult` so F.4 tuning can A/B both. This is recorded as **divergence DV-F1-3** in `notes/divergence-ledger.md` when F.1 lands.

Then the **libvpx skip-flag tie-break** runs per Q5. After computing `rate` above and the existing `dist = sse_block(...)` at line 604:

```rust
// libvpx skip-flag tie-break (vp9_pickmode.c:2355-2362).
// Decide whether to keep the encoded residual or skip it entirely.
let rd_encode = rd_cost(lambda, rate, dist as u32);
let rd_skip = rd_cost(lambda, skip_cost, sse_after_pred as u32);
let (final_rate, final_dist, picked_skip) = if rd_encode < rd_skip {
    // Cheaper to encode. Add skip=false cost bit.
    (rate.saturating_add(skip_false_cost), dist, false)
} else {
    // Cheaper to skip. Pay the skip=true cost bit; dist is the
    // post-prediction SSE the decoder will see.
    (skip_true_cost, sse_after_pred as u32, true)
};
```

Where `sse_after_pred = sum((src - pred)^2)` (NOT post-recon). The picker's existing `pred` buffer is the source of truth. `skip_prob` is looked up from the entropy state per libvpx's `vp9_cost_bit(skip_prob, 0/1)`.

### 5.6 Interfaces touched

| Surface | Change |
|---|---|
| `FullPixelSearchParams` | None. block_yrd runs post-search. |
| `MvCandidates` | None. |
| `InterModeSearchParams` | None — block_yrd needs no extra inputs beyond what `evaluate_candidate` already receives. |
| `evaluate_candidate` (private fn) | One added rate term (~3 lines). |
| `reconstruct_via_round_trip` (private fn) | Either rename to `reconstruct_and_block_yrd` with new return type, or add a sibling. |
| `MvCosts` / `InterModeCosts` | None. |
| `InterModeSearchResult` | **Add `coef_rate: u32` and `coef_dist: i64` fields. Mandatory per Q4 resolution — F.0.2's `coef_bits_est` schema field becomes live at F.1 by reading `coef_rate >> VP9_PROB_COST_SHIFT` (back-convert from prob_cost to bits). Plumbed in the same PR.** |

### 5.7 Use of existing rvp9 primitives

| Operation | rvp9 function path |
|---|---|
| Forward DCT 8×8 | `crates/rvp9-core/src/transforms/forward.rs:fdct8x8` (already called by `reconstruct_via_round_trip`) |
| Forward DCT 16×16 | `crates/rvp9-core/src/transforms/forward.rs:fdct16x16` |
| Forward DCT 32×32 | `crates/rvp9-core/src/transforms/forward.rs:fdct32x32` |
| Forward quant | `crates/rvp9-core/src/quant/forward.rs:vp9_quantize_b{,_32x32}` |
| Scan order | `crates/rvp9-core/src/transforms/scans.rs:scan_for`, `quant::forward::scan_for_forward` |
| Inverse DCT (post-recon SSE path stays) | `crates/rvp9-core/src/transforms/inverse.rs:idct{8x8_64,16x16_256,32x32_1024}_add` |
| `VP9_PROB_COST_SHIFT` | `crates/rvp9-core/src/cost/mod.rs:46` (re-exported) |
| `rd_cost` | `crates/rvp9-core/src/mode_decision/mod.rs:55` |
| `compute_rd_mult` | `crates/rvp9-core/src/cost/rd_mult.rs:78` |

### 5.8 Hadamard vs DCT — divergence note

libvpx's `block_yrd` uses `vpx_hadamard_{8x8,16x16}` as the forward
transform; rvp9 already uses the **full forward DCT**
(`fdct{8,16,32}`) in `reconstruct_via_round_trip` because pixel-SSE
needs the same coefficients the inverse will undo.

**This is a deliberate divergence.** F.1's `block_yrd` accumulator
operates on the DCT qcoeffs that rvp9 already has, not on a separate
Hadamard pass. The trade-off:

- **Pro**: Halves the transform cost (no second Hadamard pass).
- **Pro**: Aligns the rate model with the distortion model (the
  qcoeff array determines both `dist` and the eventual bitstream).
- **Con**: The SATD-of-qcoeff bit-rate proxy was empirically tuned on
  Hadamard coefficients in libvpx. DCT magnitudes are not identical to
  Hadamard magnitudes; the SATD-to-bits proportionality may need a
  scalar correction.

Mitigation: F.1's primary signal is the per-block oracle (F.0.2)
showing rvp9's picker now prefers NEWMV on blocks where libvpx does. If
the SATD-to-bits scale is off by a constant, that affects PSNR-delta
magnitude but not pick directionality. F.4 tuning recalibrates the
scalar if needed.

Recorded as divergence DV-F1-1 in `notes/divergence-ledger.md` when
F.1 lands. The `vpx_hadamard_{8,16}` port can land as F.1.2 if a F.4
A/B shows the Hadamard variant is materially closer to libvpx behavior.

---

## 6. Coefficient-rate estimator: what we need

### 6.1 What exists in rvp9 today

`crates/rvp9-core/src/cost/coef_cost.rs:1-220` — `CoefCostTable`,
`build_coef_cost_table`, `vp9_cost_tokens`, `vp9_cost_tokens_skip`.
This is libvpx's *exact* per-token probability-tree walk
(`vp9_rd.c:135-152`, the `fill_token_costs` family). **It is not used
by `block_yrd`** — `block_yrd`'s SATD is a much cheaper coarse model.

### 6.2 What F.1 needs to add

Two scalar helpers — neither exists in rvp9 today:

| Helper | Purpose | libvpx ref |
|---|---|---|
| `fn satd_i16(qcoeff: &[i16]) -> u32` | sum of absolute values of quantized coefficients | `vpx_satd_c`, `vpx_dsp/avg.c:334-341` |
| `fn block_error_fp_i32(coeff: &[i32], dqcoeff: &[i16]) -> i64` | sum of squared coef-vs-dqcoeff differences | `vp9_block_error_fp_c`, `vp9_rdopt.c:334-345` |

Both live in `block_yrd.rs` (private to the module). They are pure
functions, trivially unit-testable, and identical to libvpx's `_c`
references. SIMD ports are deferred (the existing per-candidate
overhead is on the order of microseconds; see §8). Document in the
divergence ledger and call out as future SIMD candidates if cpu-used 7
budget pressure increases.

### 6.3 No new probability tables

`block_yrd` does not consume `entropy::default_probs::*` or any prob
table. The SATD-based rate model is parameter-free (modulo the `<<
(2 + 9)` scale shift, which is a hard-coded libvpx constant).

---

## 7. Test plan

### 7.1 Unit tests in `block_yrd.rs`

| Test | Coverage |
|---|---|
| `block_yrd_all_zero_residual_is_skippable` | Pass all-zero qcoeff/dqcoeff/coef arrays; assert `skippable == true`, `coef_rate == 0` (no SATD bits, no EOB charge: `eob_cost == 0`). |
| `block_yrd_all_zero_except_first_uses_dc_path` | Single non-zero qcoeff at position 0, eob == 1; assert `coef_rate` = `(abs(qcoeff[0]) << 2 + eob_cost) << VP9_PROB_COST_SHIFT`. |
| `block_yrd_satd_path_for_eob_gt_1` | Multiple non-zero qcoeffs with eob > 1; assert `coef_rate` matches `(satd(qcoeff[..n]) << 2 + eob_cost) << VP9_PROB_COST_SHIFT`. |
| `block_yrd_dist_is_block_error_fp_over_4` | coef = [10, 20, 30], dqcoeff = [8, 18, 28]; assert `coef_dist == ((4+4+4) >> 2) == 3`. |
| `block_yrd_monotone_in_residual_energy` | Two synthetic blocks where block B has 2× block A's qcoeff magnitudes; assert `coef_rate_B > coef_rate_A`. (The single most important sanity check — exposes any off-by-one in the SATD path.) |
| `block_yrd_multi_tile_64x64_sums_correctly` | Synthetic Block64x64 → 4× Tx32x32 tiles; assert `block_yrd` over the four tiles equals the sum of per-tile calls. |

### 7.2 Differential test — F.0.2 oracle pre/post-F.1

**Primary signal for F.1 sign-off.** Run the F.0.2 per-block oracle
(`crates/rvp9-harness/src/bin/f02_block_oracle.rs`) over foreman_cif at
the four Track F target bitrates (25 / 50 / 100 / 250 kbps) with the
F.0.2 schema's `coef_bits_est` column live (per the F.0.2 liveness
matrix, `coef_bits_est` becomes live "after F.1"). Compare:

| Metric | Pre-F.1 (main @ `f1a8327`) | Post-F.1 (target) |
|---|---|---|
| Mean fraction of inter leaves picking NEWMV (foreman) | E13 reports 0.1% to ~2% | Material non-zero — qualitative: noticeable fraction. Quantitative threshold left to team-lead per F.0.1 milestone convention. |
| `partition_match%` vs libvpx | E1 reports ~5% | Unchanged (block_yrd doesn't move partition). |
| `mode_match%` on matched-partition leaves | low (NEAREST-dominated) | Materially higher — NEWMV-on-NEWMV is the headline. |
| `mean Δ coef_bits` (rvp9 − libvpx) | rvp9 = 0 (forward-looking) | finite-and-bounded — should converge toward zero from the +∞ side (rvp9 < libvpx is plausible because the SATD proxy under-estimates real entropy-coded bits). |

### 7.3 PSNR-delta gate (F.0.1 primary gate)

Run `cargo test -p rvp9-encoder --release --test quality_gate
quality_gate_stage4a_realcontent -- --ignored --nocapture` pre and
post F.1.

**Expectation**: foreman mean PSNR-delta drops from +9.07 dB.
**F.1 sign-off threshold**: per HLD F.0.1 §3.2 *the threshold is a
measurement, not a prediction*. The implementer measures the new mean
Δ and reports it; the team-lead sets the F.1 milestone constant in
`crates/rvp9-encoder/tests/quality_gate.rs:TRACK_F_PSNR_DELTA_THRESHOLD_DB`
post-measurement. The kickoff journal estimates a drop to ~+6-7 dB
based on E16; HLD F.0.1 §7 explicitly cautions this is a best-guess.

### 7.4 Smoke test — synthetic 2-frame KF+inter with known motion

`crates/rvp9-encoder/tests/f1_block_yrd_newmv_winner.rs` (new file).
Use the existing `encode_synthetic_two_frame_kf_inter_non_skip`
helper (from S4.16m) with a content variant where the inter frame is
the KF shifted by a known full-pel offset (e.g. +8 pixels horizontal).
Assert: at least one leaf on frame 2 picks `NEWMV` (with MV
approximating (0, 8) in Q3). Pre-F.1: this almost certainly picks
NEAREST/ZERO (the chicken-and-egg degeneracy F.0.3 §1 documents).
Post-F.1: NEWMV wins because the coef_rate term punishes
ZERO's large residual.

### 7.5 Byte-exact cross-decode — INTENTIONALLY BROKEN

F.1 changes which (mode, MV) the picker emits → bitstream changes →
existing `cross_decode_byte_exact` tests will fail. **Per the kickoff
journal and HLD F.0.1, byte-exact cross-decode is temporarily off the
gate set during F.1-F.3 and is re-established as the F.4 exit
criterion.** Document this in the F.1 PR description; mark the cross-
decode tests `#[ignore]` only on the foreman gate (other byte-exact
tests that use synthetic KF-only or skip-only paths can stay green).

The F.0.2 oracle (which records picker decisions independently of
bitstream output) is the byte-exact substitute during F.1-F.3.

### 7.6 Realtime budget — re-run `realtime_gate_720p30`

`cargo test -p rvp9-encoder --release --test realtime_gate -- --ignored
--nocapture`. **Must stay ≤33 ms median.** Current 5.34 ms median /
33 ms budget (6× headroom). F.1's added cost is two extra accumulator
walks (SATD + block_error_fp) per coefficient per candidate per leaf —
see §8 for the estimate. Critical gate; if it crosses 25 ms, F.1
requires the simple-yrd shortcut (§8 / Q1).

### 7.7 Clippy + unit-test workspace gate

`cargo clippy --workspace --all-targets -- -D warnings` must stay
clean. The 94 encoder unit tests must stay green (or, if a unit test
in `inter::tests` was asserting on the no-coef-bit RD value, update
the assertion with a clear comment).

---

## 8. Realtime-budget impact estimate

### 8.1 Per-candidate cost of `block_yrd`

For one Block16x16 inter candidate:
- Tile count: 1 (one 16×16 transform).
- Coefficients per tile: 256.
- `satd_i16`: 256 abs+add = ~256 ops (~100 ns scalar; SIMD trivial
  but deferred).
- `block_error_fp_i32`: 256 sub+square+add = ~256 mul-add ops (~200 ns
  scalar).
- Total: ~300 ns per candidate.

For one Block32x32: 1024 coefficients → ~1.2 µs per candidate.

For Block64x64: 4 × Tx32x32 = 4096 coefficients → ~4.8 µs per
candidate.

### 8.2 Per-leaf cost

Current picker evaluates 4 inter modes × 2 ref frames = 8 candidates
per leaf today (per the dual-ref `pick_for_ref` closure at
`lib.rs:2973`). F.1 adds 8 × `block_yrd` calls per leaf.

For a Block64x64 leaf with 8 candidates: 8 × 4.8 µs = ~38 µs.
For a Block16x16 leaf: 8 × 0.3 µs = ~2.4 µs.

### 8.3 Per-frame cost — foreman_cif 256×256 baseline

Foreman_cif 256×256 with the current variance-tree partitioner is
heavily 8×8 (E1 reports ~85% rvp9 8×8). Block8x8 is below F.1's
supported set today (the picker bails to ZERO+skip per `lib.rs:2660-
2711`); F.1 inherits that bail. Treat the live leaves as Block16x16
on average → ~2.4 µs / leaf × ~256 leaves / frame ≈ **0.6 ms / frame**.

### 8.4 Per-frame cost — realtime gate 720p

`realtime_gate_720p30` runs at 1280×720 with the existing F8.x picker
pipeline (`crates/rvp9-encoder/src/lib.rs:2330,2340,2359` — the
picker is gated off for `w*h > 307_200`, i.e. >720p picker pipeline is
already disabled per A4.rt-recovery from night-7). **Within
`pick_and_simulate_inter_encode_partitions`, F.1's cost is zero for
720p+ because the picker doesn't run there at all.** The realtime
gate is therefore *very* unlikely to regress.

For < 720p (e.g. CIF baseline gates), the worst-case per-frame
overhead is bounded by ~5 ms (Block64x64 × 8 candidates × ~64
partitions, ~1.2 ms; or Block16x16 × 8 × 1024 partitions, ~1.5 ms).
Foreman_cif at 256×256 sees < 1 ms.

### 8.5 Verdict

**F.1 should not blow the 33 ms median.** Headroom remains ~5× even
in the worst-case CIF projection. **Risk flagged regardless** — the
foreman CIF measurement and a 720p picker-on stress run are part of
the implementer's sign-off. If F.1's measured cost crosses 10 ms
median, fall back to the simple-yrd shortcut (the
`model_rd_for_sb_y`-Laplacian path) as F.1.1.

### 8.6 What F.1 does NOT bring in

- `model_rd_for_sb_y` / `model_rd_for_sb_y_large` Laplacian R-D LUTs
  (`rate_tab_q10` / `dist_tab_q10`). These would be the "free triage
  step" libvpx uses before block_yrd. Porting them is a separate task
  (Stage 1 PLAN.md follow-up: "Laplacian R-D tables `rate_tab_q10`/
  `dist_tab_q10`"). F.1 ships without them.
- `vpx_hadamard_{8x8,16x16}` as a cheaper-than-DCT pre-pass. The DCT
  is already done by `reconstruct_via_round_trip`; F.1 reuses those
  coefficients. (DV-F1-1.)
- `skip_txfm[]` early-exit short-circuit. F.1 always runs the round-
  trip and always computes `block_yrd`. Skip-detection is F.4 tuning.

---

## 9. Coupling with F.2 / F.3

F.1 alone may not move PSNR-delta materially. The chain:

1. **F.1 (this HLD)**: makes coefficient bits visible to RD; NEWMV
   starts winning on blocks where its prediction is genuinely better.
2. **F.2 (`find_mv_refs` proper + A18 default-ON)**: now that NEWMV
   wins on real-motion blocks, NEAREST's anchor (built from
   neighbours' picks) starts seeing real MVs. The chicken-and-egg
   degeneracy collapses.
3. **F.3 (motion-aware partition)**: with NEWMV picks happening,
   the partitioner can sense that motion-comp RD justifies larger
   blocks. The current variance-of-ZEROMV-residual partitioner
   over-splits to 8×8 because ZEROMV's residual is always large.

**Explicit commitment**: F.1 ships even if foreman PSNR-delta does
not move from +9.07 dB by a milestone-defining amount. The F.1
sign-off signal is the **F.0.2 oracle** showing that rvp9's picker
now considers NEWMV winning on blocks where libvpx picks NEWMV. The
end-to-end PSNR-delta gate returns to F.2 and F.3 once the cluster
closes. (Kickoff journal §"Plan" — explicit team-lead expectation:
F.1 alone won't close the +9 dB.)

**Foreman-specific bound on F.1's measurable impact.** Per E1's block_count histogram (`wrk_docs/E1_encoder_gap_analysis.md` §3), foreman's current variance-partitioner descends to Block8x8 on ~85% of leaves. F.1's `block_yrd` only runs on Block16x16+ (the picker bails to ZERO+skip at 8×8 per `lib.rs:2660-2711`). So F.1's RD signal only reaches ~15% of foreman's leaves until F.3 (motion-aware partition) lifts the typical leaf size. This is the *expected* reason F.1 alone may not move foreman PSNR-delta materially. F.1 sign-off is the F.0.2 oracle's rvp9-side `mode_match%` and `coef_bits_est` movement on the Block16x16+ subset; foreman PSNR-delta is informational.

---

## 10. Risks / open questions

### Risks (R-numbered, mitigation included)

| # | Risk | Mitigation |
|---|---|---|
| R1 | **Wrong port target.** F.1 ports `block_yrd`. If libvpx's realtime path actually depends on the `model_rd_for_sb_y` triage being run *first* (so that `x->skip_txfm[0]` is set before `block_yrd` is even called), then porting block_yrd in isolation may not give the expected gain. | The model_rd_for_sb_y skip-triage is decoupled in libvpx's flow: it only short-circuits the `block_yrd` call entirely (`this_early_term`). If we always run block_yrd (no early term), we get a strict superset of the rate signal. F.4 can layer in the triage to reclaim cycles. |
| R2 | **Quantizer-scale mismatch.** libvpx's `block_yrd` uses `x->plane[i]` quantizer state in `vp9_quantize_fp`; rvp9 has `PlaneQuantizers` (initialized via `init_plane_quantizers`) and uses `vp9_quantize_b`. Different quant variants → different `qcoeff` magnitudes → different SATD. | rvp9 already uses `vp9_quantize_b` in `reconstruct_via_round_trip` so the *picker's* qcoeff is already on rvp9's quant_b scale. F.1 consumes those qcoeffs, so the picker is internally consistent. The libvpx-rvp9 mismatch shows up in the F.0.2 oracle as a non-zero `Δ coef_bits` mean — record it, do not chase it for F.1. F.4 tuning may swap rvp9's picker to `vp9_quantize_fp` to match libvpx. (DV-F1-2.) |
| R3 | **`vp9_cost_coeffs` is RDO-only; the realtime path's rate model is `vpx_satd`.** Verified in §2.5: libvpx's nonrd `block_yrd` does NOT call `vp9_cost_coeffs`. F.1 must NOT pull in `CoefCostTable` — that would be the wrong abstraction. | Documented. The exact probability-tree cost stays in `cost/coef_cost.rs` for the eventual RDO path; the realtime path uses `block_yrd`'s SATD proxy. |
| R4 | **`skip_txfm[]` short-circuit deferred.** F.1 always runs the full round-trip, including for blocks where libvpx would short-circuit on `model_rd_for_sb_y`'s skip detection. | Documented in §5.3. The added cost is the model_rd_for_sb_y-skippable fraction × per-leaf round-trip cost. From foreman E1 numbers (~24% skip area), this is ~0.25 ms of avoidable work per frame. Below the budget impact threshold. F.4 will layer in the short-circuit. |
| R5 | **F.0.2 oracle libvpx patch isn't built yet.** F.0.2's manual PowerShell build step is documented (`scripts/build_f02_libvpx.ps1`) but not auto-run by CI. The differential test plan in §7.2 depends on it. | Sequence the F.1 implementer task after F.0.2 implementer task. The team-lead's TaskList already orders them this way (F.0.2 is #87, F.1 is post-#87). |
| R6 | **Hadamard-vs-DCT scale (§5.8).** SATD-to-bits proportionality may be off by a constant. | Acceptable for F.1 — does not affect pick directionality, only PSNR-delta magnitude. F.4 tuning recalibrates. (DV-F1-1.) |
| R7 | **Lambda mismatch.** rvp9's `compute_rd_mult` matches libvpx's. The coef_rate's `<< 11` scale (`<< 2` rate + `<< 9` prob_cost) puts it on the same units as mode_rate / mv_rate, so RDCOST composition is consistent. Verified by inspection of `rd_cost` (`mode_decision/mod.rs:55-66`). | No issue. Unit test in §7.1 (monotonicity) catches any regression. |

### Open questions for the team-lead — RESOLVED

**Q1 — Defer simple-yrd shortcut to F.1.1, confirmed?** §5.3 / §8.6.
At cpu-used 7 single-layer, `use_simple_block_yrd = 0`; at cpu-used 8
it's `1`. Foreman's gate is cpu-used 7. PLAN.md target is cpu-used 6-8.
The full-path port handles cpu-used 7 correctly; cpu-used 8 will need
the shortcut later. **Proposal: defer to F.1.1, document the
deferral.**
**RESOLVED (team-lead, 2026-05-23): APPROVED.** Defer simple-yrd shortcut to F.1.1.

**Q2 — Which RD-thresh / `rd_thresh_freq_fact` machinery, if any, lands
with F.1?** libvpx's nonrd picker has a per-`(bsize, mode)` adaptive
threshold (`rd_less_than_thresh_row_mt`, `update_thresh_freq_fact`).
This gates *which modes are even tried*, independent of `block_yrd`'s
rate model. rvp9 does not have it today (the picker tries all 4 inter
modes unconditionally). **Proposal: defer to F.4.** Adding the
threshold machinery now risks over-pruning before block_yrd has even
made NEWMV competitive.
**RESOLVED (team-lead, 2026-05-23): APPROVED.** Defer `rd_thresh_freq_fact` machinery to F.4.

**Q3 — `dist` term: pixel-SSE-of-recon vs `coef_dist`?** §5.5. libvpx
uses `block_error_fp >> 2` (coefficient-domain) for `this_rdc.dist` in
RDCOST. rvp9 today uses pixel-SSE-of-recon. The two should be
Parseval-equivalent up to a transform scale, but the constants differ.
**Proposal: F.1 first-cut keeps pixel-SSE-of-recon (preserves the F8.0
contract);** implementer A/Bs both and reports.
**RESOLVED (team-lead, 2026-05-23): APPROVED — pixel-SSE-of-recon for F.1 first-cut.** Document as divergence DV-F1-3 in the implementer's divergence-ledger update. Justification: changing `dist` semantics also changes RDCOST's rate-distortion ranking; F.1 should not require lambda recalibration as a side effect. F.4 tuning revisits.

**Q4 — Should the F.0.2 `coef_bits_est` schema field be filled from
`coef_rate` directly?** F.0.2's liveness matrix says `coef_bits_est`
goes live after F.1. The natural mapping is `coef_bits_est =
coef_rate >> 9` (back-convert from prob_cost to bits). Implementer
should plumb this in the same PR.
**RESOLVED (team-lead, 2026-05-23): APPROVED.** Plumb `coef_bits_est` into the F.0.2 oracle in the same PR.

**Q5 — Skip-flag tie-break (`block_yrd`'s libvpx caller at lines
2355-2362) — port now or defer?** The libvpx caller has a small
"if RDCOST(rate, dist) < RDCOST(0, sse) then keep encoded else
skip" tie-break. Without it, the picker can't decide skip=true vs
skip=false on the basis of `block_yrd`. **Proposal: port it with F.1**
— it's 5 lines of code, sits in `evaluate_candidate`, and is needed
for correct skip output.
**RESOLVED (team-lead, 2026-05-23): APPROVED.** Port the libvpx tie-break (vp9_pickmode.c:2355-2362) into `evaluate_candidate` as part of F.1. Without it the picker can't decide skip=true vs skip=false based on coef_rate vs sse.

**Q6 — Hadamard vs DCT — accept divergence?** §5.8 / R6.
**Proposal: accept the divergence for F.1.** A faithful Hadamard port
is F.1.2 if the F.4 A/B shows it matters.
**RESOLVED (team-lead, 2026-05-23): APPROVED.** Accept DCT-vs-Hadamard divergence; recorded as DV-F1-1.

**Q7 — Distortion clamp.** `inter.rs:605` clamps `sse` to `u32::MAX`.
With `block_yrd`'s `coef_dist` arriving as `i64` and shifted `>> 2`,
it can still overflow `u32` on a fully-saturated 32×32 quant-error
block. **Proposal: use `i64` for the internal accumulator;
saturate-cast to `u32` only at the `rd_cost` boundary** (which already
takes `u32`).
**RESOLVED (team-lead, 2026-05-23): APPROVED.** Internal accumulator is `i64`; saturate-cast to `u32` only at the `rd_cost` boundary in `evaluate_candidate`.

---

## 11. Out of scope (explicit)

- **Intra block_yrd path.** libvpx calls block_yrd from
  `estimate_block_intra` at `vp9_pickmode.c:1073`. rvp9's intra picker
  has its own (simpler) RD model. Porting block_yrd into the intra
  path is a follow-up — only worth it once F.2/F.3 show whether the
  intra mix is closing the gap.
- **`model_rd_for_sb_y` Laplacian model.** Stage 1 PLAN.md follow-up.
- **`vpx_hadamard_*` Hadamard kernels.** Future SIMD candidate; F.1
  reuses the existing DCT path.
- **SIMD for `satd_i16` / `block_error_fp_i32`.** Scalar reference
  only in F.1.
- **Chroma `block_yrd_uv` / `model_rd_for_sb_uv`.** Single-plane
  (Y-only) RD model; rvp9 doesn't have chroma sensitivity gating in
  the picker. Defer to a future "color-sensitive picker" task.
- **`thresh_freq_fact` adaptive mode-skip.** F.4.
- **`vp9_NEWMV_diff_bias` neighbour-MV consensus bias.** Already
  partially in place per E13 / divergence ledger; F.1 doesn't touch
  it.
- **Trellis-quant `vp9_optimize_b`.** Decoder-only-relevant (per
  LLD-transforms §7.4); not in any encoder path.

---

## 12. Acceptance — HLD signed off

- [x] Team-lead reviewed §1-§11.
- [x] Q1–Q7 resolved (see §10 — each open question marked RESOLVED).
- [x] HLD signed off — F.1 ready for implementer dispatch.
- [ ] Implementer task created against this HLD with explicit references to §5 (design), §7 (test plan), §8 (realtime budget), §10 (risks), and §13 (implementer-facing summary).

## 13. Implementer-facing summary

1. Add `crates/rvp9-core/src/mode_decision/block_yrd.rs` with
   `BlockYrdResult` + `block_yrd(...)` + `satd_i16` + `block_error_fp_i32`.
2. Update `crates/rvp9-core/src/mode_decision/inter.rs`:
   - Rename `reconstruct_via_round_trip` → `reconstruct_and_block_yrd`
     (or add a sibling returning `BlockYrdResult`).
   - Inside the existing tile loop, accumulate `satd_i16`,
     `block_error_fp_i32`, `eob_cost` per tile.
   - Return `BlockYrdResult` alongside the existing reconstruction.
   - In `evaluate_candidate`, add `coef_rate` to the rate sum.
   - Plumb the libvpx skip-flag tie-break (per Q5).
3. Optional (recommended) — add `coef_rate` and `coef_dist` to
   `InterModeSearchResult` for F.0.2 oracle.
4. Add unit tests per §7.1.
5. Add smoke test per §7.4.
6. Run F.0.2 oracle pre/post (§7.2), `quality_gate_stage4a_realcontent`
   pre/post (§7.3), `realtime_gate_720p30` pre/post (§7.6).
7. Mark byte-exact cross-decode foreman test `#[ignore]` for F.1-F.3;
   add a clear comment explaining the F.4 re-establishment plan.
8. Run `/code-reviewer` and `/code-review-apply-feedback` per the
   Track F workflow.
9. Update `notes/divergence-ledger.md` with:
   - **DV-F1-1**: rvp9 uses DCT qcoeffs for block_yrd SATD where libvpx uses Hadamard qcoeffs (§5.8).
   - **DV-F1-2**: rvp9's picker uses `vp9_quantize_b` where libvpx uses `vp9_quantize_fp` (R2).
   - **DV-F1-3**: F.1 keeps pixel-SSE-of-recon as the RD `dist` term where libvpx uses `block_error_fp >> 2` (coefficient-domain). F.4 tuning may revisit.
10. Report:
    - Foreman PSNR-delta pre/post (number, not threshold).
    - Foreman per-block oracle pre/post: fraction-NEWMV-picks,
      mode_match%, `Δ coef_bits` mean.
    - 720p realtime median pre/post.
    - All test results verbatim in PR description.

---

## 14. Ship-state addendum (team-lead, 2026-05-23)

F.1 implementer landed all infrastructure plus both RD-active
mechanisms (`coef_rate` add + libvpx skip-flag tie-break). Empirical
measurement on foreman_cif showed PSNR-delta regressed from +9.07
to +12.22 dB (mean), with the 250 kbps point catastrophically
dropping from 26.96 to 20.91 dB when `coef_rate` was active alone.

**Root cause** (see `notes/divergence-ledger.md` F.1 ship-state
entry): both mechanisms have lambda-calibration coupling. libvpx's
`<< (2 + VP9_PROB_COST_SHIFT)` SATD-to-prob-cost scale is tuned
against libvpx's `compute_rd_mult`; rvp9's lambda was calibrated
when the picker had no `coef_rate` term. Activating either
mechanism without recalibrating lambda overprices candidates at
higher bitrate operating points, where the picker has real coding
choice → falls back to worse modes → PSNR collapses.

**Team-lead decision (Arthur, 2026-05-23):** F.1 ships
**INFRASTRUCTURE-ONLY**. Both RD-active mechanisms gated behind
env-vars, default **OFF**:

- `RVP9_F1_COEF_RATE_ENABLE=1` — activates `coef_rate` in the
  picker's rate sum (per §5.5 design).
- `RVP9_F1_SKIP_TIE_BREAK_ENABLE=1` — activates the libvpx
  skip-flag tie-break (per §5.5 / Q5 design).

At the default (both unset), the picker is byte-identical to
pre-F.1. Foreman PSNR-delta = +9.07 dB exactly (PASS at 9.5 dB
threshold). **No behavioural regression.**

**What does land:**
- `crates/rvp9-core/src/mode_decision/block_yrd.rs` (full module,
  6 unit tests).
- `BlockYrdResult { coef_rate, coef_dist, skippable }` flowing
  through `reconstruct_via_round_trip`.
- `InterModeSearchResult { coef_rate, coef_dist }` fields
  populated.
- F.0.2 oracle's `coef_bits_est` schema field reads
  `coef_rate >> VP9_PROB_COST_SHIFT` (live on rvp9 leg per the
  HLD F.0.2 liveness matrix).
- The libvpx skip-flag tie-break code itself (gated, dormant).
- DV-F1-1, DV-F1-2, DV-F1-3 documented in
  `notes/divergence-ledger.md`.

**What does NOT land:**
- Any *behavioural* change in the picker's RD comparison.

**F.4 inheritance**: F.4 owns lambda recalibration. Once
`compute_rd_mult` or the SATD-to-bits scale is recalibrated against
rvp9's RD ranking, F.4 flips the two activation env vars to "always
on" (most likely by removing the env-var gates entirely, after the
A/B sweep confirms no regression).

**Why this is the right call**: F.2 and F.3 land on this
infrastructure without any of F.1's regression. The `coef_rate`
flow is observable in the F.0.2 oracle's `coef_bits_est` column
for diagnostic A/B even with the activation gates off — the
infrastructure is complete; only the RD-comparison plug is wired
later.

**Cross-references:**
- Decision recorded in `notes/divergence-ledger.md` F.1 ship-state
  entry (2026-05-23).
- F.1 implementer task #88 marked completed.
- F.4 task (#84) inherits the lambda recalibration work-items.
