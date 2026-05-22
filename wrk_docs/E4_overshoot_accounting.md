# E4 — foreman bitrate-overshoot accounting

Diagnostic-only. foreman_cif 256×256, 64 frames, target 100 kbps, AQ off
(same drive as `bdrate_gate_stage4a_realcontent`). All numbers from
rvp9's OWN bitstream writer (entropy-cost model `-log2(p_used)` per
binary symbol), NOT the decoder mi-grid. Reproduce:
`RVP9_E4_BITACCT=1 RVP9_E4_SKIPSTAT=1 cargo test --release -p rvp9-encoder
--test e4_overshoot_accounting -- --ignored --nocapture`.

Model validation: per-frame category sum 8197 b vs actual 8272 b (0.9%).

## 1. Per-category bit breakdown (avg over 63 inter frames)

| category    | bits/frame | % of coded |
|-------------|-----------:|-----------:|
| **coef (residual tokens)** | **5531** | **67.5%** |
| mode+ref    | 1267 | 15.5% |
| skip flag   |  731 |  8.9% |
| partition   |  621 |  7.6% |
| is_inter    |   47 |  0.6% |
| tx_size / mv_delta / seg_id | 0 | 0% |

**Coefficient tokens are the dominant cost (5531 b ≈ 67%).** This
overturns E1's "partitioning is the headline gap" framing: partition
signalling is only 7.6%. The 2.3× excess bits vs libvpx (~3565 b) live
in the residual stream, not the structural overhead.

## 2. Rate-model prediction error + Q ceiling

- Mean actual/predicted ratio = **1.032** (per frame: 0.74–1.25). The
  rate model is essentially ACCURATE at the picked Q — the converged
  `rate_correction_factors[]` feedback loop has closed the A10 "~10×
  under-prediction." **The overshoot is NOT a rate-model error.**
- Avg picked inter Q = 212, **max = 245 (< 255)**. On EVERY inter frame
  `picked_q == active_max_qindex` — Q is **pinned at the ceiling**.
- The ceiling is `active_worst_quality_inter` =
  `min(255, (avg_frame_qindex_inter·5)>>2)` (`cbr.rs:374-410`). With
  ambient ≈196 this clamps to 245. `regulate_q` *wants* a higher Q to
  hit the ~2950-bit target but falls through to `active_worst`.

This is the real overshoot mechanism: the RC predicts correctly, asks
for a higher Q to hit target, and the `ambient·5/4` clamp refuses it.
Frames 1–9 show it starkly — target steady at ~2950 b, yet Q is *pulled
down* 238→182 by the ambient EWMA while actual bits climb 2816→12384.

## 3. Skip utilisation

- Inter leaves: 57,402. **skip=1: 59.3%**; skip=0: 40.7%.
- Mean total-eob per skip=0 leaf = **6.9 nonzero coeffs**.
- **60.9% of skip=0 leaves carry ≤4 nonzero coefficients.**

rvp9 has **no RD skip decision** — `lib.rs:2969` hard-sets `skip=false`,
then only flips to skip=1 if *every* coeff quantises to zero
(`leaf_total_eob==0`, `lib.rs:3156`). libvpx makes an explicit
rate-distortion skip choice and drops blocks whose tiny residual costs
more than the distortion it removes. Those ~14k near-zero-residual
leaves are exactly the "blocks libvpx would skip outright."

## Verdict

Single dominant source: **coefficient tokens, 5531 bits/inter frame
(67%)** — driven by ~14k near-zero-residual (eob≤4) blocks rvp9 codes
non-skip that libvpx's RD skip decision would drop, with the
`ambient·5/4` Q-clamp preventing the RC from quantising them away.

**Tractable, not structural.** Two medium, well-isolated coding/RC fixes
(both libvpx-faithful ports, like E2's threshold port):
1. **RD skip decision** in the inter picker — port libvpx's
   `rate < dist` skip test so tiny-residual blocks code skip=1. Directly
   removes most of the 67% coef cost. (Largest lever.)
2. **Relax/replace the `ambient·5/4` `active_worst` clamp** so Q can
   reach the target — port the full `calc_active_worst_quality_one_pass_cbr`
   ceiling. Lets the RC quantise residuals away when fix 1 leaves any.

The rate model itself is fine (ratio ≈1.0); no multi-week
rate-model/block_yrd rebuild is required.
