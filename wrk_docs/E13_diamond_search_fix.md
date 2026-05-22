# E13 — full-pel diamond search fix

**Verdict: the diamond WAS broken and is now FIXED at the search level
(2.0% → 75.6% of reachable full-pel gain; +5.34 dB into the NEWMV
candidate). But it is necessary-NOT-sufficient: the RD picker rejects
the now-good NEWMV on 100% of leaves, so shipped recon/pred PSNR is
unchanged. The next lever is the picker/partition, not the search.**

## Phase 1 — root cause

`run_newmv_search` (`crates/rvp9-core/src/mode_decision/inter.rs`)
**short-circuited the diamond** whenever the int_pro `newmv_seed` was
`Some`, using the seed directly + sub-pel. The E13 oracle
(`RVP9_E13_DIAMOND`, foreman q120, 122 068 NEWMV leaves) showed:

- **89.9% of NEWMV leaves took the seed short-circuit** — the diamond
  never ran for them.
- Resulting full-pel MV was only +0.1 dB over (0,0): SAD@zero=810,
  SAD@newmv=800, SAD@oracle=310. **diamond-recovers-fraction = 2.0%.**

libvpx only short-circuits the diamond for the GOLDEN/ALTREF int_pro
fast path (`x->sb_use_mv_part`, `vp9_pickmode.c:207-210`). The
LAST-frame NEWMV path ALWAYS runs `vp9_full_pixel_search`
(`vp9_pickmode.c:154-261`), with the seed as the search *start*
(`mvp_full`) — not a replacement. rvp9 also conflated start-MV and
cost-center into one `center_mv`.

Secondary checks: `max_iters` = `total_steps` (not capped); step
pattern (`init_dsmotion_compensation`, len 1024→1) is faithful;
`sad_per_bit=16`. None of these starved the search — the bypass did.

## Phase 2 — fix (faithful, realtime-bounded)

Always run the diamond. Added `start_mv: Option<Mv>` to
`FullPixelSearchParams` (libvpx `mvp_full`), separating it from
`center_mv` (the rate-reference / `mvsad_err_cost` attractor =
`ref_mv`). `run_newmv_search` now seeds `start_mv` from the int_pro MV
(or NEAREST) and keeps `center_mv` = NEAREST, then runs the diamond +
sub-pel for every NEWMV leaf. `start_mv = None` preserves the legacy
start==center behaviour for existing call sites/tests.

## Phase 3 — validation (foreman 256×256, 64 frames)

| metric | before (E10) | after (E13) |
|---|---|---|
| diamond-recovers-fraction | 2.0% | **75.6%** |
| NEWMVraw SAD-PSNR @8×8 | 26.20 dB | **31.54 dB** (oracle 34.42) |
| reached-oracle / improved-on-zero | 9.2% / 62.2% | 19.9% / 79.6% |
| **NEWMV wins RD argmin** | 0% | **0.00%** (0/107 826) |
| matched-Q pred PSNR (q120) | 26.19 dB | 26.19 dB (unchanged) |
| matched-Q recon PSNR / bits (q120) | 31.44 / 37 357 | 31.44 / 37 357 |
| testsrc `bdrate_gate_stage4a` | 33.32/33.98/36.43/40.66 | identical |
| foreman BD-rate | not measurable | not measurable |
| cross-decode byte-exact | 94/0 | **94/0** |
| realtime 720p30 median | — | **5.03 ms** (budget 33) |

## Why recon didn't move

NEWMVraw−shipped jumped from +0.1 → **+5.34 dB** (the E12 lever fired
at the search level). But the E13 argmin counter shows NEWMV wins
**0.00%** of the RD comparison: at foreman's 8×8-dominated partition
granularity the MV-rate term (~thousands of cost units against
ref=(0,0)) outweighs the distortion gain, so the picker still ships
NEARESTMV-at-(0,0). This is exactly E9 §4 / E10's downstream chain.
**The remaining +8 dB lever is the picker/partition (larger usable
partitions + NEWMV RD-affordability), not the search.**
