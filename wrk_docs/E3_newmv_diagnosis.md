# E3 — Why rvp9 picks 0% NEWMV on foreman

**Verdict: Hypothesis 3 (RD imbalance) at the surface, but the root is a
hybrid of (2) + (3): the motion search runs and returns MVs, but those
MVs buy almost no distortion, so the MV-bit rate term — already correct
and libvpx-faithful — always wins. This is NOT a tunable rate-weight
fix.** Method: env-gated `RVP9_E3_MVDIAG` capture of LAST-ref
`pick_inter_mode_luma_all_costs` per leaf; 256×256 foreman crop, 64
frames; throwaway test `e3_mvdiag` (`RVP9_E3_TARGET_BPS` overrides the
100 kbps default to sweep q).

## Hypotheses 1 & 2 (partial): FALSIFIED at the wiring level

The picker IS invoked per leaf (`lib.rs:2925`) and NEWMV IS searched:
`newmv_nonzero_mv ≈ 99–100%` of leaves, `avg_mag` 80–194 (1/8-pel),
`max_mag` up to 2068. So ME is wired and returns non-zero MVs.

## The data (100 kbps, q≈170–243 — RC-saturated)

NEW wins the 4-way RD argmin on **0 of every ~900 leaves, every frame**.
NEAREST wins **100%**. Crucially, `nearest=(0,0)` on **2520/2520**
sampled leaves — so ZERO/NEAREST/NEAR all predict from MV=(0,0) and tie
on distortion; NEAREST wins purely on cheapest mode rate (1698 vs 4987).

Representative leaf `px=(200,8)` q=238 (the single most telling number):
- NEW distortion saving over NEAREST: 1939 SSE → **248,192 RD units**.
- NEW extra rate (5058 prob-cost units, ~all MV bits) × λ=2.5M / 512 →
  **24,709,802 RD units of penalty**.
- Net NEW loses by 24.46M — **99% explained by the MV-rate term.**

The MV-rate is NOT inflated: NEAREST=(0,0), so NEWMV encodes a large
delta from a zero predictor (~10 bits) — correct. λ is the libvpx
formula, unchanged.

## The decisive cross-check (8 Mbps, q drops to 7–19)

Forcing q low (so λ ≈ 0, rate term negligible) STILL yields **0% NEWMV**.
At q=15 the diamond returns huge MVs (mag 770–1280) but distortion barely
moves (NEAREST d=53 → NEW d=51). Over the 843 low-q leaves where NEW
distortion < NEAREST distortion: **avg gain = 12 SSE, median 7** — noise.
Even with MV bits set to **free**, all 843 flip — but the gain is trivial,
so it would not help BD-rate. The (0,0) LASTFRAME predictor is already
near-perfect frame-to-frame at 30fps; ME finds nothing materially better,
and ~20% of leaves wander to spurious far MVs (false SAD minima).

## Contributing defect (int_pro seed)

`int_pro_motion_estimation` returns `sad=u32::MAX` (sentinel/bail) on
**12 of 16 SBs** of a 256×256 frame — every edge SB. Only the 4 interior
SBs get a real seed, and those produce sane small MVs (e.g. (8,-24)).
Edge SBs fall back to a (0,0) diamond center → the wandering large MVs.

## Recommended fix scope

This is **NOT** a one-line wiring/tuning fix like E2, and lowering the
MV-cost weight will not move BD-rate (the distortion gains are noise).
The chain is: bitrate overshoot → q saturates → λ huge → MV-rate
dominates; AND independently the diamond barely beats zero-MV and the
int_pro seed is edge-disabled. Two tractable workstreams, in priority
order:
1. **Fix the bitrate overshoot / rate model first** (E1/E2's binding
   constraint). 0% NEWMV is largely a *symptom* of q=255 saturation; at a
   sane q the small genuine gains start mattering.
2. **Then a real ME build** (multi-week): proper int_pro seeding that
   works at frame edges, a search that tracks foreman's actual panning
   (the seed MVs show motion exists), and ideally libvpx's
   coefficient-domain `block_yrd` distortion model — post-recon SSE at
   high q collapses residuals and erases the differentiation NEWMV needs.
