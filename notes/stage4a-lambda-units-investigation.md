# Stage 4a' — Lambda units-mismatch investigation (2026-05-22)

## Brief hypothesis (NOT confirmed)

> Lambda is over-weighted by ~256× because rvp9's rate is in
> `vp9_prob_cost` units (~512 = 1 bit) while libvpx's lambda formula
> assumes rate in some other unit. So `RD = D + λ·R` over-charges rate
> by 256×, making NEWMV always lose on foreman_cif.

**Verdict: the hypothesis is wrong.**

## Side-by-side comparison

| Quantity | libvpx | rvp9 | Match? |
|---|---|---|---|
| `RDCOST(RM, DM, R, D)` macro | `ROUND_POWER_OF_TWO(R*RM, 9) + (D << DM)` | same | YES |
| `RDDIV_BITS` (DM in macro) | 7 | 7 | YES |
| `VP9_PROB_COST_SHIFT` (shift in macro) | 9 | 9 | YES |
| Rate unit | `vp9_prob_cost` (1/512-bit) | same | YES |
| Distortion unit | sum-of-squared-error | same | YES |
| `compute_rd_mult_based_on_qindex` | `q*q * (a + b*q)` with `(a,b) = (4.15, 0.001)` for inter | identical | YES |
| Lambda at q=255 inter | ~7.8M | ~7.8M | YES |

References:
- libvpx macro: `D:\language\libvpx-stock\vp9\encoder\vp9_rd.h:29-30`
- libvpx compute_rd_mult: `D:\language\libvpx-stock\vp9\encoder\vp9_rd.c:241-276`
- rvp9 macro: `D:\language\rvp9-2\crates\rvp9-core\src\mode_decision\mod.rs:55-66`
- rvp9 compute_rd_mult: `D:\language\rvp9-2\crates\rvp9-core\src\cost\rd_mult.rs:60-72`

The agent's "estimated lambda = 7,800,000" figure is CORRECT for q=255
inter. That value is also what libvpx uses. It is large because
`q*q*4.15 ≈ 1336*1336*4.6 ≈ 8.2M` at q=255. Libvpx is not 3 orders of
magnitude lower; the brief's claim that "libvpx's typical lambda
values range from a few hundred at q=20 to a few thousand at q=255"
is mistaken.

## The one real (small) divergence — RESOLVED 2026-05-22 (A13 / commit 2c07ec4)

libvpx's `vp9_mv_bit_cost` (mcomp.c:80-84) returns
`ROUND_POWER_OF_TWO(mv_cost(diff) * MV_COST_WEIGHT, 7)` with
`MV_COST_WEIGHT = 108`. That is, MV rate is scaled by `108/128 ≈ 0.84`
before being charged in RDCOST.

rvp9's picker (`inter.rs:533-540`) charged raw `mv_cost(diff_row,
diff_col, mv_costs)` without the `MV_COST_WEIGHT` scaling.

Net effect: rvp9 over-charged MV bits by `128/108 ≈ 1.185×`. This was
~19% over, not 256×. As predicted below, fixing this is a real
bug-for-bug fidelity gap but doesn't unstick foreman from its
~24-29 dB band — see "actual blocker" below.

**Resolution (A13):** ported the `ROUND_POWER_OF_TWO(raw * 108, 7)`
scaling at `inter.rs:601-609`. F12 testsrc gains +1.15 to +4.6 dB on
the 50-100 kbps targets (the band where the picker has room to swap
to NEWMV). F12 foreman flat (rate-model bottleneck, not RD-formula
bottleneck). Cross-decoder gate PASS byte-exact. Divergence ledger
updated.

## The actual blocker on foreman_cif

With `RVP9_DIAG_PICKER=1` on the foreman gate, every leaf prints to
stderr. Across all four bitrate targets (25/50/100/250 kbps) and all
64 frames:

- 86,115 non-skip leaves total
- **0 NEWMV picks** (grep `NewMv` on the diag log: zero matches)
- Effective per-leaf q ranges from 143 (frame 0 KF) to 255 (motion
  frames)
- Most motion leaves report `q=255` (RC has saturated qindex max) and
  `leaf_total_eob=0 (skip=true)` (residual quantizes to zero)

Per-leaf cost decomposition at q=255 (lambda ≈ 7.8M, dist in raw SSE):
```
  ZEROMV    rate=3584  → rate_term = 3584*7.8M/512 = 54.6M  ← always large
  NEARESTMV rate=295   → rate_term = 295*7.8M/512  =  4.5M  ← much cheaper
  NEWMV     rate≈5000+ → rate_term = 5000*7.8M/512 = 76.2M  ← worse than ZEROMV
  dist_term = dist << 7 ≈ 100K..8M (saturates at high-motion 8x8 leaves)
```

At q=255 the rate term swamps the dist term for all rates >~ 300.
NEARESTMV always wins because its mode-cost is tiny (295). NEWMV is
killed by mode-cost + mv-cost = ~5000-12000 even with a perfect MV.

**At q=255 the residual ALSO quantizes to zero (`leaf_total_eob=0`),
so reconstruction = LASTFRAME + motion compensation only. Since
NEAREST/ZEROMV use MV=(0,0), the recon is literally `LASTFRAME` for
every leaf. PSNR is locked at LASTFRAME-staleness.**

The bitrate-actual is ~2× target (foreman 250 kbps target →
514 kbps actual) because the encoder emits one mode-rate symbol +
skip flag per leaf even when residual is empty. RC has no further
lever; q is already at 255.

## Where to look next (in priority order)

1. **Why the bitstream over-shoots target by 2-5× on foreman**. RC
   saturates q at 255, but bitrate stays well above target. This
   suggests per-leaf overhead (mode/skip/MV symbols) is the floor and
   the partition picker may be using too-small leaves. Investigate
   `pick_and_simulate_inter_encode_partitions` partitioning depth on
   foreman vs the per-leaf symbol budget at high q.

2. **NEWMV mode-cost might still be wrong**. The
   `params.inter_mode_costs[ctx][NewMv]` lookup at q=255 still gives
   a base mode rate of 3000+ prob_cost units (≈ 5-6 bits per leaf,
   even before MV bits). Check whether the inter-mode probability
   table is feeding from the default `FrameContext`, which has higher
   entropy for NEWMV than a trained context would. Cross-check
   against libvpx `cpi->inter_mode_cost` values at q=255.

3. **Block8x8 fallback**. A0 identified this as the dominant bug;
   the brief asks us to verify it's still in place. Audit
   `crates/rvp9-encoder/src/lib.rs` for the picker_supported gate
   that was at lines 2081-2099. If that fallback is gone, then the
   picker IS now running on Block8x8 — confirm via diag log.

4. **The 1.185× MV_COST_WEIGHT scaling**. Add `* 108 >> 7` in
   `inter.rs:536` to match libvpx exactly. Low priority — won't
   change foreman PSNR but tightens cross-encoder fidelity.

## What was NOT changed

No code shipped. The hypothesis the brief asked me to test is wrong.
Shipping a "fix" for a non-existent units mismatch would introduce
regressions without addressing the real blocker.

## Gates re-measured (post merge of main into worktree)

- Cross-decoder gate: PASS (byte-exact rvp9 vs vpxdec, vs ffmpeg)
- F12 testsrc: 35.93 / 38.68 / 43.08 / 59.69 dB at 68 / 113 / 175 / 262 kbps (unchanged)
- F12 foreman_cif: 16.36 / 16.21 / 16.28 / 16.55 dB at 127 / 160 / 249 / 515 kbps (unchanged from baseline)
- RT gate: 6.57 ms median (PASS, ≤ 33 ms)
- Workspace lib clippy: 2 pre-existing errors in decoder Wave F kernel
  files (`idct16x16_avx2.rs:522`, `idct32x32_avx2.rs:816`) — not my
  work, flagged in brief.
