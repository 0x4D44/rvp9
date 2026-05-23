# E14 — partition-size lever re-tested under E13's fixed search

**Verdict: REFUTED again. The coupled lever (E13's working full-pel
diamond + larger inter partitions) does NOT close the gap. At cap=16
NEWMV RD-win share stays effectively zero (0.006%) and recon PSNR
moves +0.06 dB. At cap=32 NEWMV does start winning occasionally
(1.14%) but recon PSNR REGRESSES by -0.47 dB. The hypothesised
mv-rate amortisation occurs numerically at cap=32 but is dwarfed by
the loss of fine detail. Probe-only — NOT landed. The next binding
constraint is NEAREST candidate quality / the NEWMV MV-rate model,
not partition granularity.**

## Mechanism

Env-gated `RVP9_E14_INTER_MIN={16,32}` in
`crates/rvp9-core/src/mode_decision/partition.rs`:

- `=16` — forbid Block16x16 → Block8x8 split on inter (cap descent at
  16×16). Mirrors E11's reverted probe.
- `=32` — forbid Block32x32 → Block16x16 split (and 16→8) on inter
  (cap descent at 32×32). New extension for E14.
- Off (default) — current main, exactly preserved (byte-exact).

Keyframes are unaffected — KF descent is already capped via the
`allow_split_to_8x8` gate (see partition.rs docstring).

## Matched-Q oracle — foreman_cif 256×256, 64 frames

`RVP9_FIXED_Q=120`, AQ off. **q=120 primary** (other q points show the
same pattern):

| metric                     | baseline (no cap)        | cap=16             | cap=32           | libvpx     |
|----------------------------|--------------------------|--------------------|------------------|------------|
| pred PSNR (inter)          | 26.19 dB                 | 26.18 dB           | 25.72 dB         | n/a        |
| recon PSNR (inter)         | 31.44 dB                 | **31.50 dB**       | **30.97 dB**     | —          |
| recon PSNR (all)           | 31.45 dB                 | **31.51 dB**       | **30.98 dB**     | 39.23 dB   |
| bits/frame (all)           | 37 330                   | 37 031             | 37 806           | 15 602     |
| inter coef bits/frame      | 34 865                   | 36 138             | 37 475           | n/a        |
| inter skip frac            | 12.8%                    | 6.5%               | 0.9%             | n/a        |
| inter Y tx-hist [4,8,16,32]| [0, 50524, 3353, 36]     | [0, 0, 15984, 36]  | [0, 0, 0, 4032]  | —          |
| KF Y tx-hist [4,8,16,32]   | [0, 0, 244, 3]           | [0, 0, 244, 3]     | [0, 0, 244, 3]   | —          |

Partition histogram confirms mechanism: cap=16 collapses 8×8 leaves
(50524 → 0) and shifts them to 16×16 (3353 → 15984); cap=32 collapses
everything to 32×32 (4032 leaves).

## NEWMV RD-win share — foreman, q=120 (`RVP9_E3_MVDIAG`, 63 inter frames)

| setting          | leaves total | NEW wins | NEW win-share | `mv_rate` (lost-leaf avg) | `rd_gap` (lost-leaf avg) |
|------------------|-------------:|---------:|--------------:|---------------------------|--------------------------|
| baseline (no cap)| 53 913       | 0        | **0.000%**    | ~6100                     | ~720k–870k               |
| cap=16           | 16 020       | 1        | **0.006%**    | ~6100                     | ~720k–830k               |
| cap=32           | 4 032        | 46       | **1.141%**    | ~6800                     | ~740k–890k               |

**Key observation:** `mv_rate` and `rd_gap` are essentially size-invariant.
At cap=16 NEWMV's distortion gain scales down proportionally with the
4×-larger block, so the RD margin against NEARESTMV-at-(0,0) is
unchanged. The lever fires only at cap=32 (where MV-rate amortises over
16× pixels) but at the cost of recon quality.

## BD-rate gate runs

### testsrc `bdrate_gate_stage4a` (synthetic gradient, 4 target bitrates)

| setting          | rvp9 curve (kbps, PSNR dB) |
|------------------|---------------------------|
| baseline (no cap)| 62.2/33.32, 75.5/33.98, 111.9/36.43, 226.7/40.66 |
| cap=16           | **45.5/33.67, 58.0/35.38, 96.6/38.50, 214.3/42.88** |
| cap=32           | **34.4/38.20, 54.3/40.31, 93.7/41.43, 204.1/45.80** |
| libvpx (anchor)  | 24.5/44.00, 29.6/53.59, 42.6/59.79, 68.4/78.80 |

Both caps *improve* testsrc — a flat colour-bar clip benefits from huge
blocks. Mirrors E11's earlier finding. **Landing the blunt cap would
game the testsrc gate without helping real content.**

### foreman `bdrate_gate_stage4a_realcontent`

| setting          | rvp9 curve (kbps, PSNR dB) |
|------------------|---------------------------|
| baseline (no cap)| 118.4/24.01, 133.7/24.10, 143.9/24.53, 261.7/26.96 |
| cap=16           | 83.5/24.08, 98.8/24.18, 124.0/25.19, 261.9/27.57 |
| cap=32           | 69.0/24.22, 88.9/24.51, 124.4/25.70, 263.0/27.63 |
| libvpx (anchor)  | 45.7/28.32, 74.0/30.51, 143.1/33.52, 307.9/37.25 |

Still **"not measurable"** BD-rate (rvp9 curve has no overlap with
libvpx; PSNR gap remains ~5–9 dB). PSNR shifts at fixed target are
mostly the same +5–10% bits/PSNR trade observed in matched-Q.

## Cross-decode byte-exact + clippy

- `cargo test -p rvp9-encoder --release` = **95 passed / 0 failed / 26 ignored**
  (default cap=None preserves byte-exact behaviour).
- All 7 `mode_decision::partition::tests` pass.
- Clippy on `partition.rs` — 0 warnings / 0 errors (other rvp9-core
  modules have pre-existing warnings the task explicitly rules out
  of scope).

## Why the lever didn't fire (the actual binding constraint)

The hypothesis was: at cap=16, MV-rate is amortised over 4× pixels →
NEWMV's distortion advantage stays the same → NEWMV wins on RD. The
data refutes step 2: `mv_rate` did not change (still ~5000–6500 cost
units) AND `rd_gap` did not change (still ~720k–870k). At cap=32 the
mechanism finally does fire (NEWMV wins 1.1% of leaves) but the RD gain
is too small to offset the loss of fine-grained partition.

The diagnosis: **NEAREST is already coding a zero-motion prediction for
free** (no MV bits, NEARESTMV mode bit only). NEWMV must beat NEAREST
by enough distortion to also pay a 5000-unit MV-rate. On foreman's
moderate-motion content, NEWMV's full-pel-only MV (E13 brought the
SAD-PSNR up to 31.5 dB inside the candidate) often *does* find better
motion — but not by ~5000-unit-rate worth. And block size doesn't
change that equation: distortion savings scale with area, MV-rate is
flat.

## Conclusion / recommendation

Probe-only. The env-gated `RVP9_E14_INTER_MIN` knob remains in the
tree (off by default) as a test scaffold. The chain's "next step" was
falsified. The actual remaining levers, in priority order:

1. **NEAREST candidate quality** (`find_mv_refs`) — if NEAREST often
   carried a *non-zero* libvpx-faithful pred_mv from neighbours / prev
   frame mvs, NEARESTMV would no longer be a free zero-motion baseline
   and NEWMV could compete with it on equal footing.
2. **`vp9_cost_mv_ref` / `mvjcost`/`mvcost` audit** — is the
   ~5000-unit NEWMV mv-rate faithful to libvpx, or inflated? The
   `rd_gap` ~800k vs mv-rate ~5000 ratio (~160×) suggests rate is not
   the only deficit, but it's the size-invariant one.
3. A motion-aware partitioner (the original E11/E14 "proper fix"
   theory) is **NOT** recommended — the data shows the RD math doesn't
   flip cleanly at any uniform cap, so a motion-aware version would
   inherit the same NEAREST/NEWMV asymmetry.

Nothing landed. Recommend E15 starts on the NEAREST / mv-cost levers.
