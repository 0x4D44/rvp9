# E16 — `pick_inter_mode_ctx_affects_rate` test failure

**Date:** 2026-05-23
**Author:** E16 (agent)
**Status:** Verdict = **stale assertion**; test relaxed.

## TL;DR

The failing assertion (`r6.best_mode` must be a zero-MV mode at
`inter_mode_ctx = 6`) is **stale post-A13** (commit `2c07ec4`,
`MV_COST_WEIGHT = 108` scaling). At ctx 6 on identical src/ref planes,
NEWMV legitimately wins the pure rate-cost tiebreak with the
libvpx-default `inter_mode_probs[6] = [25, 29, 30]`. Test relaxed to
keep only the assertion that **matches the test's stated purpose** —
ctx 0 and ctx 6 produce a different (mode, rate) pair.

## Diagnostic — per-candidate breakdown (identical src/ref, q40, 16x16)

```
inter_mode_costs[0] = [NEAREST=295, NEAR=2329, ZERO=3584, NEW=943]
inter_mode_costs[6] = [NEAREST=1685, NEAR=1749, ZERO=1718, NEW=257]

ctx=0:
  mode=ZeroMv    mv=(0,0) rate=3584 dist=0 rd=49301
  mode=NearestMv mv=(0,0) rate= 295 dist=0 rd= 4058  <-- winner
  mode=NearMv    mv=(0,0) rate=2329 dist=0 rd=32037
  mode=NewMv     mv=(0,0) rate=2239 dist=0 rd=30799  (= 943 + 1296 mv_rate)

ctx=6:
  mode=ZeroMv    mv=(0,0) rate=1718 dist=0 rd=23633
  mode=NearestMv mv=(0,0) rate=1685 dist=0 rd=23179
  mode=NearMv    mv=(0,0) rate=1749 dist=0 rd=24059
  mode=NewMv     mv=(0,0) rate=1553 dist=0 rd=21363  <-- winner (= 257 + 1296)
```

`mv_rate(0,0) = ROUND_POWER_OF_TWO(joint_cost[MV_JOINT_ZERO] * 108, 7)
≈ 1296` (joint prob = 64/256 → ~1536 raw → 1296 after A13 scaling).

## Why this is libvpx-correct

`vp9_inter_mode_tree` decodes as `(ZERO?, NEAREST?, NEAR/NEW?)`. At
ctx 6 (two intra neighbours), `default_inter_mode_probs[6] = [25, 29,
30]` (`vp9_entropymode.c:233-242`) — all three probs are very small,
i.e. all three "is-it-the-left-branch" decisions are improbable. The
leaf reached by taking the right branch at all three nodes is NEWMV.
Hence:

- NEWMV leaf cost: cost_one(25)+cost_one(29)+cost_one(30) ≈ 257
- ZEROMV leaf cost: cost_zero(25) ≈ 1718
- NEAREST leaf cost: cost_one(25)+cost_zero(29) ≈ 1685
- NEARMV leaf cost: cost_one(25)+cost_one(29)+cost_zero(30) ≈ 1749

After A13's MV-bit scaling (`* 108/128 ≈ 0.84`), NEWMV's full encoded
cost (mode_rate 257 + scaled mv_rate 1296 = 1553) sits **below** any
zero-MV mode. This is the picker correctly modelling libvpx's prior
on motion in intra-bordered blocks. There is no carve-out for the
degenerate identical-src/ref scenario (which never occurs in real
content); the picker outputs the lowest-RD candidate and that is
NEWMV here.

## Why the test was wrong

The test's stated invariant (line 1450-1457 comment) is "ctx 0 vs
ctx 6 produces a different (mode, rate) pair". The "must be a zero-MV
mode" assertion was an over-strong corollary added when the test was
written — true under the pre-A13 picker where unscaled MV bits made
NEWMV's effective rate (~2239 at ctx 6 pre-scaling) unable to undercut
ZEROMV.

## Action taken

Relaxed `pick_inter_mode_ctx_affects_rate` in
`crates\rvp9-core\src\mode_decision\inter.rs:1411` to drop the
"winner must be zero-MV mode" asserts and keep only:

- `r0.best_distortion == 0` and `r6.best_distortion == 0` (sanity:
  flat-plane all-MV setup has zero distortion for every candidate).
- `r0` vs `r6` differ in either mode or rate (the stated invariant).

No picker LOGIC changes. Rationale documented inline in the test.

## Validation

- `cargo test -p rvp9-core --release --lib mode_decision::inter::tests::pick_inter_mode_ctx_affects_rate` → **PASS**.
- `cargo test -p rvp9-core --release --lib` → only pre-existing
  fixture-missing `bitstream::*round_trip` failures remain.
- `cargo test -p rvp9-encoder --release` → 94/94 unchanged
  (byte-exact preserved; no picker logic touched).
