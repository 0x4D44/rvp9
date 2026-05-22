# E10 — int_pro edge-SB seed fix (border-clamp, no bail)

**Verdict: the edge-SB bail was a real defect and is now FIXED (16/16
SBs seeded), but it is necessary-NOT-sufficient. The seed reaches the
picker frame-wide, yet NEWMV RD-win share stays 0.0% and recon/pred
PSNR + bits at matched Q are byte-identical to E9's pre-fix table. The
prediction cascade does not fire; the coupled partition/NEWMV-RD fixes
(E9's chain) are still required.**

## Phase 1 — root cause

`int_pro_motion_estimation` (`crates/rvp9-core/src/mv_search/int_pro_me.rs`)
bailed with the `u32::MAX`/`Mv::ZERO` sentinel whenever the SB sat too
close to a frame edge:

```rust
if sb_px_x < bw/2 || sb_px_y < bh/2 { return ZERO; }
if sb_px_x + bw + bw/2 > width || sb_px_y + bh + bh/2 > height { return ZERO; }
```

The 1-D projection window reads `ref[sb_y-bh/2 .. ][sb_x-bw/2 .. ]` plus
a 2D-SAD halo; on rvp9's bare `PlaneRef` (visible plane, no padding)
those reads are genuinely OOB at a border, so it bailed. On a 256×256
frame (4×4 grid of 64×64 SBs) that disabled the seed on **12 of 16 SBs
— every edge SB** (only the 4 interior SBs passed). Confirmed
analytically and by `RVP9_DIAG_INTPRO_SB`.

**Why libvpx doesn't bail:** `vp9_int_pro_motion_estimation`
(`vp9_mcomp.c:2264-2400`) runs against a reference frame whose borders
are physically extended by edge replication (`vpx_extend_frame_borders`).
Border reads return replicated edge pixels, so every SB gets a seed.

## Phase 2 — fix

Replaced the bail with **edge-clamped reference reads** — the faithful
equivalent of an extended-border frame. Added `clamp_px` (clamp `(y,x)`
to the visible plane), `int_pro_row_clamped`, `int_pro_col_clamped`
(projection builders over absolute, possibly-negative coords), and
`sad_block_clamped` (gather a clamped reference block, then SAD). The
projection window and 2D-SAD halo now route through these. Source reads
stay in-plane (SB top-left is always inside the frame). New unit test
`all_sbs_seeded_no_edge_bail` asserts **16/16** seeded.

## Phase 3 — measurement (foreman_cif 256×256, 64 frames)

**Seed level (RVP9_DIAG_INTPRO_SB) — FIXED:** edge SBs now produce real
motion seeds tracking foreman's leftward pan, e.g. `sb=(2,0)→(8,0)`,
`sb=(0,2)→(0,-8)`, `sb=(3,2)→(0,-24)`. The pre-fix wandering far-MVs
(E3's `max_mag` up to 2068) collapse to sane small magnitudes.

**Encoder output — UNCHANGED.** Matched-Q oracle (`RVP9_FIXED_Q`,
`e9_quality_gap`) is byte-identical before/after:

| qindex | recon PSNR | pred PSNR | bits/frame | inter tx hist [4,8,16,32] | libvpx PSNR / bits |
|-------:|-----------:|----------:|-----------:|---------------------------|--------------------|
| 80  | 31.68 dB | 26.32 dB | 66 350 | [0, 55032, 2210, 40] | 41.78 dB / 30 595 |
| 120 | 31.45 dB | 26.19 dB | 37 330 | [0, 50524, 3353, 36] | 39.23 dB / 15 602 |
| 160 | 28.02 dB | 25.88 dB | 18 321 | [0, 44084, 4971, 34] | 36.19 dB / 7 329  |

- **Prediction PSNR: still stuck ~26 dB** (not above).
- **NEWMV share: still 0.0%** (`RVP9_E3_MVDIAG`, even at a 2 Mbps
  target). NEWMV loses the RD argmin on MV-rate (~5000 units) vs
  NEARESTMV-at-(0,0) — exactly E3/E9's finding. `newmv_nonzero_mv`
  ~100%, but RD-win NEAREST=100%.
- **Partition: still 8×8-dominated** (~55k 8×8 vs ~2k 16×16); no easing.

## Validation oracle

1. Matched-Q PSNR/bits: unchanged (table above).
2. NEWMV share 0.0%; partition hist unchanged.
3. testsrc `bdrate_gate_stage4a`: no regression (33.32/33.98/36.43/40.66 dB).
   foreman `..._realcontent`: BD-rate not measurable (curves don't
   overlap), as the brief anticipated.
4. Cross-decode byte-exact: `cargo test -p rvp9-encoder --release` = 94
   pass / 0 fail (all encoder↔decoder bit-exact recon gates green).
5. Clippy clean on `mv_search/`.

## Why the cascade did not fire

The seed only steers the NEWMV diamond *center*; the picker then runs an
RD comparison that rejects NEWMV regardless of seed quality. With the
variance partitioner descending densely to Block8x8, the per-leaf
MV-rate term dominates the small distortion gain (E9 §4). The int_pro
edge fix removes one degeneracy (no frame-wide foothold) but the
downstream affordability problem remains. **Next:** E9's coupled chain —
`find_mv_refs`/NEAREST quality, larger usable partitions, and
`block_yrd`/skip_txfm so NEWMV/larger blocks become RD-affordable.
