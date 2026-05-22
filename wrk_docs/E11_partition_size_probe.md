# E11 — partition-size probe: does forcing larger inter partitions let NEWMV engage?

**Verdict: REFUTED for foreman (the real target). Capping inter partitions
at Block16×16 shifted the histogram to 16×16-dominated as designed, but
NEWMV RD-win share stayed 0.0%, prediction PSNR stayed pinned at ~26 dB,
and recon PSNR/bits barely moved. The hypothesised mv-rate amortization
did NOT occur. NEWMV's problem is deeper than partition granularity.
Probe-only — NOT landed.**

## Probe mechanism

Env-gated (`RVP9_E11_INTER_CAP16=1`) flip of the `allow_split_to_8x8`
gate in `crates/rvp9-core/src/mode_decision/partition.rs:461` so inter
frames cap the variance descent at Block16×16 (disable 16×16→8×8). Off by
default; reverted after measurement. Re-runs A7's old inter-cap idea, this
time under E9's *proper* matched-Q oracle (A7 ran at degenerate q=255 with
all-zero residuals and reached a misleading conclusion).

## Matched-Q table — foreman_cif 256×256, 64 frames (BEFORE → AFTER cap)

| q   | recon PSNR (all) | pred PSNR (inter) | bits/frame (all) | inter tx-hist [4,8,16,32] | libvpx PSNR/bits |
|----:|------------------|-------------------|------------------|---------------------------|------------------|
| 80  | 31.68 → **31.71** | 26.32 → **26.30** | 66350 → 67603 | [0,55032,2210,40] → **[0,0,15968,40]** | 41.78 / 30595 |
| 120 | 31.45 → **31.51** | 26.19 → **26.18** | 37330 → 37006 | [0,50524,3353,36] → **[0,0,15984,36]** | 39.23 / 15602 |
| 160 | 28.02 → **28.10** | 25.88 → **25.96** | 18321 → 16692 | [0,44084,4971,34] → **[0,0,15988,35]** | 36.19 / 7329  |

Partition histogram confirmed 16×16-dominated (8×8 leaves 55k→0). But
prediction PSNR is flat (±0.08 dB noise) and recon PSNR moves <0.1 dB at
the same ~half-of-libvpx-bits — no closing of the ~8–10 dB gap.

## NEWMV win-share (RVP9_E3_MVDIAG, 2 Mbps low-lambda, per-frame)

| metric                | BEFORE (8×8) | AFTER (16×16 cap) |
|-----------------------|--------------|-------------------|
| leaves / frame        | ~1018        | **256** (4× fewer) |
| NEWMV RD-win share    | **0.0%**     | **0.0%**          |
| NEWMV `mv_rate` (lost)| ~4900        | **~5100 (unchanged)** |
| NEWMV `rd_gap` (lost) | ~250k–310k   | **~220k–310k (unchanged)** |

**The smoking gun:** the hypothesis was that mv-rate amortized over 4×
pixels would make NEWMV competitive. It did not — `mv_rate` stayed ~5000
and `rd_gap` stayed constant. At 16×16 NEWMV's distortion advantage scales
down ~proportionally with the larger block, so the RD margin against
NEARESTMV-at-(0,0) is unchanged. The lever does not move the needle.

## testsrc effect (bdrate_gate_stage4a, BEFORE → AFTER)

| target | BEFORE kbps/PSNR | AFTER kbps/PSNR |
|-------:|------------------|-----------------|
| 25  | 62.2 / 33.32 | **45.5 / 33.67** |
| 50  | 75.5 / 33.98 | **58.0 / 35.38** |
| 100 | 111.9 / 36.43 | **96.6 / 38.50** |
| 250 | 226.5 / 40.66 | **213.2 / 42.00** |

The cap *helps* testsrc (lower bits AND higher PSNR — better RD on the
synthetic clip), the opposite of A7's q=255 finding. But it does nothing
for foreman, the real target. A lever that helps only the synthetic clip
is not a faithful fix; landing it would game the testsrc gate.

## Validation oracle

1. Pred/recon PSNR + bits + NEWMV win-share: table above (probe vs main).
2. Partition histogram: 16×16-dominated after (mechanism confirmed).
3. testsrc `bdrate_gate_stage4a`: cap improves it; default (cap off) green.
4. Cross-decode byte-exact: `cargo test -p rvp9-encoder --release` = 94
   pass / 0 fail (default, cap off — the landing state is unchanged main).
5. `cargo test -p rvp9-core --release --lib`: 1204 pass; 3 fail — all
   pre-existing & unrelated: known `pick_inter_mode_ctx_affects_rate`, plus
   `bitstream::*::captured_header_round_trip` /
   `write_compressed_header_captured_frame_round_trip` which fail with
   `NotFound` (missing captured test-vector fixtures in this worktree), not
   logic. None touched by an env-gated partition flag.
6. Clippy clean on `partition.rs`.

## Conclusion / next fix

Partition-size is **NOT** the lever for foreman. The next fix must target
**NEWMV's distortion/search/RD at the block level itself** — the rd_gap is
size-invariant, so the deficit is intrinsic to NEWMV's RD vs
NEARESTMV-at-(0,0): the MV-rate term (~5000 units) dominates a too-small
distortion gain regardless of block area. Candidates redirected by this
probe: (a) `find_mv_refs`/NEAREST quality so NEARESTMV stops coding a free
zero-motion prediction that NEWMV must beat from scratch, and (b) the
NEWMV MV-rate model / cost (is ~5000 units faithful, or inflated vs
libvpx's `mvjcost`/`mvcost`?). The motion-aware partitioner is a
red herring for this clip.
