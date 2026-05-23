# Track F baseline — the numbers to beat

Generated 2026-05-23, on `main` after the full Track E stack landed
(E2 + E6 + E7 + E8 + E10 + E13 + E14-doc + E15 + E16, last commit at the
time of writing was the realtime perf re-validation).

Track F's success is measured against THESE numbers. Any picker-quality
work (block_yrd port, find_mv_refs propagation, motion-aware partition)
should drive the **PSNR-delta fallback mean** down on foreman, ideally to
the point where rvp9's PSNR curve overlaps libvpx's and BD-rate becomes
measurable again.

## Foreman (real-content gate — `bdrate_gate_stage4a_realcontent`)

Source: `foreman_cif.y4m`, centre-cropped to 256×256, 64 frames.

| target kbps | rvp9 actual | rvp9 PSNR | libvpx actual | libvpx PSNR |
|------------:|------------:|----------:|--------------:|------------:|
| 25          | 118.4       | 24.01 dB  | 45.7          | 28.32 dB    |
| 50          | 133.7       | 24.10 dB  | 74.0          | 30.51 dB    |
| 100         | 143.9       | 24.53 dB  | 143.1         | 33.52 dB    |
| 250         | 261.7       | 26.96 dB  | 307.9         | 37.25 dB    |

- **BD-rate: not measurable** (rvp9 PSNR curve sits entirely below libvpx → no PSNR overlap).
- **PSNR-delta fallback (libvpx_interp − rvp9, in-range only): mean +9.07 dB, max +9.50, min +8.65, across all 4 in-range points.**

This +9.07 dB is the "real" gap after Track E closed the rate-control /
structural confounds. **Target: drive it down.** A 2 dB improvement
would put rvp9 within striking distance of curve overlap → BD-rate
becomes computable.

## testsrc (synthetic gate — `bdrate_gate_stage4a`)

Source: `testsrc_720p30.y4m`, centre-cropped to 256×256, 64 frames.

| target kbps | rvp9 actual | rvp9 PSNR | libvpx actual | libvpx PSNR |
|------------:|------------:|----------:|--------------:|------------:|
| 25          | 62.2        | 33.32 dB  | 24.5          | 44.00 dB    |
| 50          | 75.5        | 33.98 dB  | 29.6          | 53.59 dB    |
| 100         | 111.9       | 36.43 dB  | 42.6          | 59.79 dB    |
| 250         | 226.7       | 40.66 dB  | 68.4          | 78.80 dB    |

- **BD-rate: not measurable** (rvp9 bitrates run HIGHER than libvpx's — the inverse of foreman; libvpx's testsrc curve saturates at very high PSNR for low bits because the content is mostly flat colour bars).
- **PSNR-delta fallback: +41.64 dB at the 1 in-range point (62.2 kbps); 3 out-of-range, skipped.**

testsrc's huge delta is largely a content artefact (libvpx hits 78.8 dB
on the flat colour bars at 68 kbps — a regime no realtime encoder can
follow without specific optimisations for flat content). Foreman is the
fairer gate for Track F.

## Track B (AQ) — `aq5_gate_stage4b`

- libvpx AQ gain (`--aq-mode=3` vs `--aq-mode=0`): **−2.981 %** on
  foreman, **+9.312 %** on testsrc (negative = AQ saves bits at matched
  PSNR; positive = AQ costs bits; testsrc's positive value is also a
  content artefact — flat content doesn't benefit from cyclic refresh).
- rvp9 AQ gain: not fully measured in this run (the rvp9-aq curve was
  computed but the explicit AQ gain summary was overshadowed by the
  large rate-target overshoot on rvp9-aq). Track-B gate condition is
  ±0.3% delta vs libvpx-AQ — still open per PLAN.md.

## Realtime perf — `realtime_gate_720p30` (RC.3)

Run on the same `main` HEAD:
- median: **5.34 ms / frame**
- p25 / p75: 5.13 / 5.75 ms
- min / max: 4.71 / 6.47 ms
- Budget: ≤33 ms → **PASS with 6× headroom**.

The full Track E stack does not blow the realtime budget.

## How to reproduce

```bash
# Foreman + testsrc BD-rate gates with PSNR-delta fallback
cargo test -p rvp9-encoder --release --test bdrate_gate -- --ignored --nocapture

# 720p30 realtime frame-time gate
cargo test -p rvp9-encoder --release --test realtime_gate -- --ignored --nocapture
```

## How Track F's wins will appear here

When Track F's coupled fixes (`vp9_block_yrd` + `find_mv_refs` propagation
+ motion-aware partition) land, expect this doc to be regenerated and:
1. Foreman PSNR-delta fallback **mean** drops materially (target: cross
   the curve-overlap threshold so BD-rate becomes measurable).
2. Once measurable, foreman BD-rate moves toward the PLAN.md exit
   criterion of ±2 %.
3. testsrc remains a content-artefact corner case — track but don't
   gate Track F on it.
4. Realtime budget MUST be preserved (block_yrd is non-trivial cost;
   measure carefully during Track F).
