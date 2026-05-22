# E9 — decomposing the ~9 dB matched-bitrate coding-quality gap

Diagnostic-only. foreman_cif 256×256 crop, 64 frames, AQ off. Matched-Q
oracle: rvp9 pinned to a fixed `base_qindex` via env-gated `RVP9_FIXED_Q`
(no logic change — RC still runs, the pick is overridden); libvpx via
`vpxenc --rt --cpu-used=7 --end-usage=q --cq-level=N` where CLI quantizer
`N = qindex/4` (exact over this range; `quantizer_to_qindex[N] = 4N`).
rvp9-side prediction SSE tapped at the `residual = src − pred` site
(`predstat`, Y plane, before transform); recon PSNR from the real decoder;
bits from `bitacct`. Reproduce:
`cargo test --release -p rvp9-encoder --test e9_quality_gap -- --ignored --nocapture`.

## 1. Matched-Q PSNR + bits table (per-frame averages, all frames)

| qindex | side   | recon PSNR | pred PSNR (rvp9 inter) | bits/frame |
|-------:|--------|-----------:|-----------------------:|-----------:|
| 80  | rvp9   | 31.68 dB | 26.32 dB | 66 350 |
| 80  | libvpx | **41.78 dB** | — | **30 595** |
| 120 | rvp9   | 31.45 dB | 26.19 dB | 37 330 |
| 120 | libvpx | **39.23 dB** | — | **15 602** |
| 160 | rvp9   | 28.02 dB | 25.88 dB | 18 321 |
| 160 | libvpx | **36.19 dB** | — | **7 329** |

At matched Q, **libvpx is +8 to +10 dB higher PSNR AND uses ~half the
bits.** Per the brief's decision logic ("higher PSNR AND fewer bits →
prediction quality dominates"), this is unambiguously **bucket 1**.

## 2. Prediction error vs reconstruction error (the smoking gun)

rvp9's **prediction PSNR sits at ~26 dB and is essentially Q-independent**
(26.3 → 25.9 across q=80→160). The reconstruction PSNR (31.7 → 28.0)
*improves* on the prediction — rvp9 is spending 16k–64k bits/frame of coef
tokens dragging a weak 26 dB prediction up. libvpx reaches 36–42 dB recon
at half the bits, which is only possible from a far better prediction
(low-to-mid 30s) leaving a tiny residual to code.

- Q-independence of pred PSNR ⇒ the deficit is in mode/MV/partition, NOT
  transform/quant (those scale with Q; this doesn't).
- coef bits/nonzero-coef ≈ **1.7 bits/coef** at every Q ⇒ entropy coding
  efficiency is healthy (**bucket 3 is small**). The bit excess is *volume*
  of coefficients (weak prediction → big residual), not cost-per-coef.

## 3. Transform type / size usage

- **tx_type**: inter = DCT-only (correct — VP9 inter is always DCT);
  intra/KF uses ADST hybrid via `plane_mode.tx_type()`. No tx_type bug.
- **tx_size**: rvp9 always uses `max_txsize(block_size)` — **no per-block
  tx-size RD search** (documented divergence, `inter.rs:44`). Inter Y hist
  (per 63 frames) is dominated by 8×8 (`[4x4=0, 8x8≈50k, 16x16≈3k, 32x32≈40]`)
  because the variance partitioner descends densely to Block8x8 and each
  8×8 leaf forces one 8×8 transform. This is a **secondary (bucket 2)**
  contributor but minor vs prediction.

## 4. Root cause inside bucket 1 (from `RVP9_E3_MVDIAG`)

Per-leaf trace at q=80: nearly every leaf is **Block8x8 with
`nearest=(0,0)`**, and **NEARESTMV wins the 4-way RD argmin almost always**.
- MV-ref seeding is degenerate: `find_mv_refs` returns (0,0) nearest
  everywhere, so NEARESTMV codes a *zero-motion* prediction even where the
  scene moves → large residual → 26 dB.
- NEWMV's diamond+sub-pel *does* find lower-distortion MVs (e.g. NEW d=355
  vs NEAREST d=525) but its MV-rate (~5–19k units vs 1698) loses the RD
  comparison. The `rd_cost` / lambda / `MV_COST_WEIGHT=108` math is
  libvpx-faithful (verified `mode_decision/mod.rs:55`), so this is *not* an
  RD-formula bug — it is that the picker enters with bad MV candidates and
  evaluates over too-small partitions, exactly where libvpx's nonrd path
  (good MV propagation + larger partitions + `block_yrd`/skip_txfm) wins.

## Attribution (ranked)

1. **Prediction quality (bucket 1) — DOMINANT, ~7–9 dB of the gap.**
   ~26 dB MC/intra prediction vs libvpx's inferred low-30s. Mostly
   MV-reference seeding + over-fine Block8x8 partitioning + ZEROMV bias.
2. **Transform/quant (bucket 2) — minor.** No per-block tx-size search;
   8×8-locked transforms on a densely-split frame. Couples with (1).
3. **Coefficient entropy (bucket 3) — negligible.** ~1.7 bits/coef is
   healthy; bit excess is residual *volume*, not cost-per-coef.

## Recommended first fix

Target **MV-reference quality and the partition/mode picker**, mapped to
libvpx `vp9_pick_inter_mode` (`vp9/encoder/vp9_pickmode.c`):
- Fix `find_mv_refs` seeding so NEAREST/NEAR carry real neighbour/temporal
  MVs (the picker is starved — every nearest is (0,0)).
- Then the nonrd mode-decision path itself (`vp9_pickmode.c:1300-1700`):
  larger usable partitions and the `block_yrd` rate model so NEWMV/larger
  blocks become affordable. This is where the prediction PSNR climbs.

**Honest scope note:** the gap is *not* diffuse across all three buckets —
it is sharply localized to prediction. But within bucket 1 it is itself a
small chain (MV seeding → partition granularity → mode RD affordability),
so expect 2–3 picker-side fixes rather than one. Transform/quant and
entropy need no work until prediction is fixed.
