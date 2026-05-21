# Stage 4a' foreman_cif PSNR floor — block_yrd noise-sensitivity analysis

**Date:** 2026-05-22
**Agent:** worktree-agent-a08f23cdbfe9b5a4b
**Mandate:** investigate and (if appropriate) ship a single-shot fix for the F8.0 `block_yrd` noise-sensitivity divergence (`notes/divergence-ledger.md:455-477`).

## Headline

**Did not ship a fix.** Analysis showed:

1. The worktree base (`bee2345`, Stage 4a/4b closeout) is *behind* the
   `main` branch state described in the brief. The chroma intra picker
   (`A0''-chroma`) and KF partition cap (`A0''-partition`) live on
   `main` at commits `2b2cff6` / `c6b0613` / `a0e79f2` — neither is
   present in this worktree.
2. Per-plane PSNR diagnostics run against the **main-branch** checkout
   show the chroma fix already landed and is working: KF UV PSNR
   ranges 17 dB at q≥215 to 32 dB at q=34. The intra-mode distribution
   on foreman_cif KF at q=34 is 34 % DC, 15 % H, 12 % V, 8 % D135 (the
   picker IS diversifying — DC dominance is not catastrophic).
3. The foreman_cif `bdrate_gate_stage4a_realcontent` average PSNR is
   stuck at 16–17 dB across all bitrates because **the inter chain
   decays monotonically from KF onwards**, not because the KF picker
   is over-picking DC. At target 250 kbps: f0 KF=30.9 dB, f5=19.3 dB,
   f10=18.0 dB, f31=14.7 dB. `sse(r,s)` climbs monotonically while
   `sse(s,s_-1)` (source-to-source motion energy) stays modest — the
   inter picker is failing to track motion, producing reconstructions
   that drift further from the source every frame.
4. The block_yrd noise-floor fix can only affect intra picking. Even a
   perfect block_yrd would only improve the KF (already at 25–34 dB
   per-bitrate) and would not address the 64-frame average sitting at
   16 dB.
5. Implementing Approach A (noise-floor offset) requires plumbing the
   dequant table into `block_yrd_distortion` and committing to a
   noise-floor formula. Two candidate formulas (`coef^2 → 0 when
   dq==0`, vs `coef^2 → min(coef^2, (dq/2)^2) when dq==0`) have
   different effects on mode ranking; neither has a clean
   mathematical derivation that matches libvpx (which uses Hadamard
   transform for `block_yrd`, not DCT/FHT — see
   `vp9_pickmode.c:786-803`). Approach B (source-domain SSE) was
   considered: libvpx's `vp9_pick_intra_mode` does NOT use
   source-domain SSE — it calls `block_yrd` (transform-domain) via
   `estimate_block_intra`. The brief's claim that B is what libvpx's
   fast path does is incorrect.

## What the actual bottleneck looks like

Per-frame PSNR trace at target 250 kbps (rvp9 vs libvpx, same source):

| f  | rvp9 q | rvp9 bits | rvp9 PSNR | libvpx PSNR |
|----|--------|-----------|-----------|-------------|
|  0 | 124    | 32 896    | 30.90     | 45.72       |
|  1 | 191    |  6 744    | 29.32     | 41.80       |
|  2 | 185    | 10 384    | 25.47     | 37.43       |
|  5 | ~200   |  ~6 000   | 19.95     | ~35         |
| 31 | 255    |  3 008    | 14.74     | ~38         |

**Total: rvp9 626k bits → 17 dB avg, libvpx 375k bits → 36 dB avg.**

The signature is "PSNR decay from KF onwards". `sse(r,s)` climbs by
~3M per frame from f1 through f15; `sse(s,s_-1)` is ~7M-18M (i.e.
modest motion). The inter picker is leaving 3-5M SSE per frame
on the table.

## Why the inter chain is collapsing (hypothesis)

The F8.1 inter picker (`mode_decision/inter.rs`) uses post-recon SSE
(not `block_yrd`), evaluates `{ZEROMV, NEARESTMV, NEWMV} × LASTFRAME`
only. NEARMV is omitted (documented divergence). At q≥200 (which the
rate controller drives the encoder to within 5-10 frames of the KF),
ZEROMV becomes the dominant pick: any small motion produces a residual
that quantizes to zero, so the rate-cost favours the cheapest mode
(ZEROMV). The reconstructed frame N then equals reconstructed frame
N-1 plus tiny residuals — but the SOURCE has moved, so SSE grows.

## What to do instead

The compression-efficiency closure on foreman_cif requires:

1. Land A0''-chroma and A0''-partition (already on main). Without
   them this worktree's KF is much worse than the post-A1.foreman
   baseline.
2. Investigate the inter picker on textured motion content:
   - Add NEARMV (mechanical extension of the candidate list)
   - Lower the rate-control's ceiling on inter qindex (currently
     pinning at 255 by f15 in the 25 kbps target; the q range is too
     aggressive for the content)
   - Verify the LASTFRAME ref is being updated with the previous
     frame's recon, not a stale frame
3. Only AFTER (1) and (2) revisit block_yrd noise sensitivity. At
   that point the patch can be measured against a baseline that
   isn't dominated by other effects.

## Files of interest (absolute paths into the worktree)

- `D:\language\rvp9-2\.claude\worktrees\agent-a08f23cdbfe9b5a4b\crates\rvp9-core\src\mode_decision\intra.rs:196-228` —
  `block_yrd_distortion`
- `D:\language\rvp9-2\.claude\worktrees\agent-a08f23cdbfe9b5a4b\crates\rvp9-core\src\mode_decision\inter.rs:1-100` —
  inter picker docstring (post-recon SSE, no NEARMV, single-ref)
- `D:\language\libvpx-stock\vp9\encoder\vp9_pickmode.c:718-844` —
  libvpx `block_yrd` (Hadamard-based)
- `D:\language\libvpx-stock\vp9\encoder\vp9_pickmode.c:1155-1213` —
  libvpx `vp9_pick_intra_mode` (uses `block_yrd` for luma,
  `model_rd_for_sb_uv` for chroma)

## Per-plane PSNR baseline (main branch, post-A1.foreman)

KF only, foreman_cif 256×256 centre-crop:

| target  | q   | KF bits | PSNR_Y | PSNR_U | PSNR_V |
|---------|-----|---------|--------|--------|--------|
| 25 k    | 254 |   3 896 | 25.32  | 18.13  | 18.99  |
| 50 k    | 215 |   7 272 | 26.01  | 17.65  | 17.40  |
| 100 k   | 177 |  13 680 | 27.10  | 29.56  | 29.31  |
| 250 k   | 124 |  32 896 | 30.90  | 28.23  | 28.14  |
| 1 M     |  34 | 117 728 | 34.78  | 32.01  | 31.57  |

Compare libvpx at 250 kbps: KF Y=45.72 dB. Gap is ~15 dB on the KF
itself, but more importantly the inter chain falls off by another 15 dB
over 30 frames.

## Conclusion

The F8.0 block_yrd noise-sensitivity is real and documented, but it is
**not the dominant cause** of the foreman_cif PSNR floor at HEAD.
Closing it without first addressing the inter-chain degradation would
produce a non-measurable change. A single-shot block_yrd patch is
therefore deferred until the inter chain is stabilised.
