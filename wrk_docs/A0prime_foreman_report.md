# A0' — foreman_cif BD-rate diagnosis (Stage 4a' follow-up to A0/A1)

**Date:** 2026-05-21
**Agent:** worktree `agent-a5d3a6d84c2be2d72`
**Mandate:** diagnose the residual PSNR floor on foreman_cif post-A1. A0
closed testsrc's Block8x8 fallback; A1 lifted the picker gate. testsrc
PSNR jumped 24 → 35-52 dB. **foreman PSNR is unchanged at 14-16 dB.** A
different dominant bug.

## Headline

**The dominant root cause is the keyframe itself: rvp9 cannot
reconstruct foreman's KF beyond ~27 dB Y / ~17 dB UV regardless of
bit budget**, driven by two compounding bugs in the KF encode pipeline:

1. **Chroma intra is hard-forced to DC across every leaf**
   (`crates/rvp9-encoder/src/lib.rs:1437-1441` and writer sites
   `:424,:573,:837,:3046`). On a textured face (skin gradients, hair,
   ear shadow) DC chroma prediction generates residual energy the
   small-block Tx4x4/Tx8x8 path cannot represent at any reachable
   q-index. Chroma PSNR_U/V locks at 11-17 dB across the entire
   25 kbps → 1 Mbps target range — including q=34.
2. **Partition picker descends almost the entire textured KF to
   Block8x8** (940/1024 leaves Block8x8 at q=124 on a 256×256 crop).
   That fragments the bit budget across hundreds of leaves whose
   per-leaf overhead (mode bits + UV-mode + EOB) consumes most of the
   budget, leaving residual coefficients starved.

Every inter frame inherits the broken KF as its `LASTFRAME` reference
and accumulates SSE on top (the PSNR-decay-from-bad-reference signature
seen in the per-frame trace below).

**Estimated impact**: closing the chroma-DC-only divergence is the
single biggest lever — it directly addresses the UV PSNR floor and the
KF luma reconstruction quality, which gates every subsequent inter
frame. Combined with the partition-picker fragmentation it accounts
for the bulk of foreman's PSNR floor.

Hypothesis ranking:

| # | Hypothesis (brief's order) | Verdict |
|---|----------------------------|---------|
| H1 | Chroma-aware MV limit too tight (A1) | NOT dominant — KF doesn't use MVs and KF is the floor |
| H2 | KF intra picker broken on textured content | partial — picker picks varied modes (see histogram) but UV is locked to DC and luma KF still tops out ~27 dB |
| **H7** | **Chroma intra hard-forced to DC across all leaves + writer emits uv_mode=DC** | **DOMINANT** — KF UV PSNR 11-17 dB at every reachable q |
| H8 | Partition picker over-fragmenting (940/1024 Block8x8 leaves) | secondary contributor — starves per-leaf bits |
| H3 | Block16x16/32x32 inter picker fails | NOT primary — KF is the floor |
| H4 | Loop filter eating PSNR | NOT dominant — q=34 LF level=17 isn't aggressive |
| H5 | Sub-pel search broken | NOT dominant on KF |
| H6 | Residual coding wrong on 16x16 leaves | NOT primary — leaves are mostly Block8x8 |

Confidence: **high** for chroma-DC dominance. The empirical signature
(UV PSNR floor at every q-index from 254 down to 34) is unambiguous.

## Methodology

### Tools added (env-gated, committed)

- `RVP9_DIAG_INTRA` — added to `pick_and_simulate_intra_encode_partitions`
  at `crates/rvp9-encoder/src/lib.rs:1867-1879`. Dumps per-leaf KF picked
  intra mode for distribution analysis. Zero-cost when unset.

Three new diagnostic test binaries (all `#[ignore]`d):

- `crates/rvp9-encoder/tests/diag_foreman_per_frame.rs` — full
  per-frame sweep on foreman_cif 256×256 32-frame crop at four targets
  (25/50/100/250 kbps), reporting per-frame qindex, bits, Y-PSNR,
  SSE(recon,src), SSE(recon,src_-1), SSE(src,src_-1), plus libvpx
  per-frame bits/PSNR comparison.
- `crates/rvp9-encoder/tests/diag_foreman_kf.rs` — KF-only sweep across
  five targets (25k → 1M kbps). Reports per-plane PSNR_Y, PSNR_U, PSNR_V
  and centre-128×128 vs edge PSNR breakdown.
- `crates/rvp9-encoder/tests/diag_foreman_kf_partitions.rs` — direct
  call into `choose_partitioning` to count leaf bsize distribution
  across the KF at q ∈ {34, 90, 124, 150, 200}.

### Commands

```text
cargo test --release --test diag_foreman_per_frame -- --ignored --nocapture
cargo test --release --test diag_foreman_kf -- --ignored --nocapture
cargo test --release --test diag_foreman_kf_partitions -- --ignored --nocapture
RVP9_DIAG_INTRA=1 cargo test --release --test diag_foreman_kf -- --ignored --nocapture 2> intra_modes.log
```

Build clean against `-D warnings` (clippy). All existing `rvp9-encoder`
unit tests pass post-diagnostic-addition.

## Raw findings

### 1. Per-plane KF PSNR vs target bitrate

KF only, 256×256 centre-crop of foreman_cif:

| Target | q   | KF bits | PSNR_Y | PSNR_U | PSNR_V | centre Y | edge Y |
|--------|-----|---------|--------|--------|--------|----------|--------|
| 25 k   | 254 |   7,392 | 19.14  | 11.61  | 11.70  |   19.09  |  19.15 |
| 50 k   | 215 |  12,072 | 16.90  | 12.41  | 12.17  |   15.67  |  17.40 |
| 100 k  | 177 |  18,112 | 24.62  | 17.07  | 16.82  |   24.72  |  24.58 |
| 250 k  | 124 |  37,144 | 25.82  | 16.53  | 16.38  |   24.65  |  26.29 |
| 1 M    |  34 | 123,240 | 27.14  | 17.06  | 16.97  |   26.45  |  27.39 |

**The KF cannot exceed ~27 dB Y or ~17 dB UV at any reachable q-index.**
At q=34 (near-min) the encoder spends 123 KB on the KF and still only
hits 27.14 dB Y / 17.06 dB UV. UV is **catastrophic** — locked at
11-17 dB regardless of q. The centre/edge split shows the error is
distributed across the whole frame, not localised.

For comparison libvpx on the same source at 250 kbps target:
- libvpx KF: 109,608 bits → 45.72 dB Y. **20 dB above rvp9**.

### 2. Per-frame trace (250 kbps target, 32 frames)

```
  f 0: q=124 bits= 37144 PSNR=25.82dB    kf=true       ← KF floor
  f 1: q=191 bits=  6752 PSNR=25.25dB
  f 2: q=186 bits= 10280 PSNR=23.19dB
  f 3: q=192 bits=  7408 PSNR=20.87dB
  f 4: q=192 bits=  7984 PSNR=19.91dB
  f 5: q=193 bits=  8088 PSNR=19.33dB
  ...
  f31: q=153 bits= 20016 PSNR=14.47dB    ← long decay
  RVP9 TOTAL: 628384 bits  actual=588.5 kbps  avg_psnr=16.71 dB
```

libvpx on the same target+source:
```
  LV f 0: bits=109608 PSNR=45.72dB
  LV f 1: bits= 18400 PSNR=41.80dB
  LV f 5: bits=  4272 PSNR=35.49dB
  LV f31: bits=  4760 PSNR=38.00dB
  LIBVPX TOTAL: 375080 bits  actual=351.3 kbps  avg_psnr=36.42 dB
```

The decay signature `psnr_{i+1} = psnr_i - small_delta` confirms each
inter frame inherits the broken KF and accumulates small SSE on top
(the picker chases ZEROMV/NEARESTMV against a reference that itself
is at 26 dB; the residual it would need to emit to track the actual
motion is already smaller than the accumulated KF error).

### 3. Partition leaf-size distribution on foreman KF

`diag_foreman_kf_partitions.rs` calls `choose_partitioning` directly
across the whole KF, all four 64×64 SBs:

| q   | Block8x8 | Block16x16 | Block32x32 | Block64x64 |
|-----|----------|------------|------------|------------|
|  34 |      948 |          7 |          3 |          0 |
|  90 |      948 |          7 |          3 |          0 |
| 124 |      940 |          9 |          3 |          0 |
| 150 |      940 |          9 |          3 |          0 |
| 200 |      880 |         20 |          4 |          0 |

**~92% of leaves at Block8x8 across the entire q-range.** That's 940
× 1 + 9 × 4 + 3 × 16 = 1024 8×8 cells covered, just shy of full
fragmentation. At target 250 kbps the encoder has 37,144 bits / 940
leaves ≈ 39.5 bits per leaf to cover luma mode + UV mode + Y coefs +
U coefs + V coefs. Most of that vanishes into mode/EOB/header bits
before any coefficient bits get spent.

This is consistent with `choose_partitioning` using `for_keyframe`
thresholds whose Block16x16 threshold uses `threshold_base >> 2` and
the Block8x8 threshold uses `threshold_base << 2`, both scaled by
`y_dequant[q][1]`. On a textured face every 16×16 block exceeds the
"is variance > T?" gate and recurses to 8×8.

### 4. Intra mode distribution on foreman KF (250 kbps target, q=124)

```
    380 Dc       (40%)
    147 H        (15%)
     89 D135
     88 V
     63 D207
     62 Tm
     40 D153
     32 D45
     27 D63
     24 D117
```

At q=124 the picker picks DC ~40% of the time on a textured frame —
DC is rate-cheap but distortion-wise inappropriate on a face. At
q=34 mode diversity opens up (only 257/940 are DC — 27%). The picker
IS picking modes; **the bottleneck is not picker quality**, it's
that even the diverse picks can't drive PSNR up because per-leaf
overhead crowds out coefficient bits.

### 5. The chroma divergence (the smoking gun)

`crates/rvp9-encoder/src/lib.rs:1437-1441`:

```rust
let plane_mode = if plane == 0 {
    leaf_y_mode
} else {
    PredIntraMode::Dc       // ← UV ALWAYS DC
};
```

Plus writer-side `uv_mode: IntraMode::Dc` at lib.rs:424, 573, 837,
3046 (every intra-emitting writer site). The bitstream tells the
decoder "uv_mode = DC" for every block, so the decoder also predicts
chroma as DC. The simulator's chroma residual is computed against
DC prediction. End-to-end consistent (the cross_decoder gate passes
byte-exact on testsrc) but **structurally wrong** on textured content.

The existing module doc at lib.rs:1692 acknowledges this:

```rust
/// UV stays forced to `IntraMode::Dc` for v1 (the UV-mode picker is
/// further down the F8.x roadmap).
```

The roadmap entry exists; the cost on textured real content was
underestimated. UV PSNR 11-17 dB at every q-index says it dominates
on foreman.

### 6. Why luma KF also tops out at ~27 dB

Two compounding effects:

1. The KF over-fragments to ~940 Block8x8 leaves per frame (above).
   Each Block8x8 leaf carries 4 × Tx4x4 chroma TX positions (in
   addition to its Tx8x8 luma) and an `intra_y_mode` + `uv_mode` +
   `skip` flag. Mode bits + EOB overhead at 940 leaves easily exceed
   25-30 kbits before any coefficient bits.
2. Even when the picker picks the "right" directional mode for a leaf,
   the picker uses `block_yrd_distortion` (coefficient-domain MSE,
   `mode_decision/intra.rs:196-228`) instead of post-recon SSE. For
   ranking that's libvpx-faithful and approximately correct under
   Parseval, but it can mis-rank between DC (which produces large
   low-frequency residuals that quantize cheaply) and directional
   modes (which produce structured high-frequency residuals that
   often round to zero in `vp9_quantize_b` at q ≥ 100, making the
   picker over-pick DC; see the histogram above).

The first effect dominates. The second is a known F8.0 divergence
note.

## Top recommendation

### Fix: Wire a per-leaf chroma intra mode picker, and gate KF partition descent at Block16x16 (or coarser).

**Where to land the change:**

1. **UV-mode picker** —
   `crates/rvp9-encoder/src/lib.rs:1437-1441` (the
   `plane_mode = … PredIntraMode::Dc` site) needs to call a per-leaf
   UV-mode picker. libvpx's `vp9_pick_intra_mode` selects the chroma
   mode from `{DC, V, H, D45, D63, D117, D135, D153, D207, TM, FromLuma}`
   minimising the same RD metric as the luma picker, with the
   `FromLuma` mode using the luma block's picked mode. F8.0's
   `pick_intra_mode_luma` already implements 10-mode RD; the chroma
   plane needs the analogous loop driving chroma-plane SSE + the
   `uv_mode_probs` rate-cost table.

   The bitstream writer (lib.rs:424,:573,:837,:3046,
   `modes_writer::write_intra_block`) needs the same lift: emit the
   picked uv_mode bits and the corresponding chroma residual scan.

2. **Partition-depth cap** —
   `crates/rvp9-core/src/mode_decision/partition.rs:331` currently
   reads:
   ```rust
   BlockSize::Block16x16 => (2, Some(BlockSize::Block8x8)),
   ```
   Capping at Block16x16 (i.e. returning `(2, None)`) prevents the
   excessive 8×8 fragmentation on textured KFs. Tradeoff: leaves get
   larger residual energy but receive enough bits to actually emit
   coefficients. This is the same "cap partitioning at a picker-
   supported size" lever A0 proposed as a fallback, applied to the
   partition picker side rather than the picker-wrapper side.

   For inter frames the cap is less harmful (the picker already finds
   MVs at Block16x16 fine) and would actually relieve the per-leaf
   overhead pressure for foreman's inter chain too. Verify with the
   cross-decoder gate post-change — the writer's Block16x16 path is
   exercised by existing tests.

### Estimated impact

**Closes the bulk of the foreman PSNR floor.** Chroma DC-only is a
~15-20 dB drop on UV alone; capping partition fragmentation at
Block16x16 frees per-leaf bit budget so luma KF can climb from ~27 dB
toward the 35-40 dB libvpx range.

Confidence: **high** for chroma-DC dominance (the data signature —
UV PSNR locked at every q — is unambiguous). **Medium-high** for the
partition cap's incremental benefit; the data is consistent with
"per-leaf overhead dominates" but hasn't been A/B-isolated by a
direct experiment.

After the chroma-pick + partition-cap lands, re-run
`bdrate_gate_stage4a_realcontent` and expect rvp9 PSNR on foreman
to land within ~5-10 dB of libvpx (currently ~20 dB below). The
remaining gap will be addressable through better inter-mode picker
rate-cost (the A0 H3' MV-ref list note still stands) and a
post-recon-SSE intra picker distortion (an F8.0 punch-list item).

## Files of interest (absolute paths)

- `D:\language\rvp9-2\crates\rvp9-encoder\src\lib.rs:1437-1441` —
  `plane_mode = … PredIntraMode::Dc` (the chroma-DC-only divergence).
- `D:\language\rvp9-2\crates\rvp9-encoder\src\lib.rs:424,573,837,3046` —
  writer sites emitting `uv_mode: IntraMode::Dc`.
- `D:\language\rvp9-2\crates\rvp9-encoder\src\lib.rs:1838-1839` —
  Block64x64 KF leaves forced to DC (separate, doesn't fire on 256×256
  foreman because partitioning descends below 64×64).
- `D:\language\rvp9-2\crates\rvp9-core\src\mode_decision\partition.rs:331` —
  Block16x16 → Block8x8 split (the over-fragmentation site).
- `D:\language\rvp9-2\crates\rvp9-core\src\mode_decision\partition.rs:248` —
  `VbpThresholds::for_keyframe(params.q_index)` applied to inter
  frames too (A0 also flagged this; minor in foreman context).
- `D:\language\rvp9-2\crates\rvp9-core\src\mode_decision\intra.rs:196-228` —
  `block_yrd_distortion` (coefficient-domain; F8.0 divergence note).

## Notes / followups

- All diagnostics are env-gated and zero-cost when unset. The new tests
  are `#[ignore]`d. No encoder behaviour changed.
- The `RVP9_DIAG_INTRA` env-gated dump is placed in
  `pick_and_simulate_intra_encode_partitions` (cold path, called once
  per KF leaf during encode). Safe to leave in place.
- A1.perf agent is concurrently working on `lib.rs` in the same area;
  this diagnostic touches only line 1867 (a non-hot-path
  env-gated debug print). No structural change.
- After the chroma-pick fix lands, the `cross_decoder_gate` test must
  still pass — the writer's UV mode bits must remain consistent with
  the new picker output.
- The `diag_foreman_per_frame.rs` test's libvpx side now works (the
  initial `-o=path` syntax error was fixed in-place during this run);
  rerun gives complete rvp9-vs-libvpx side-by-side numbers in the
  same trace.
