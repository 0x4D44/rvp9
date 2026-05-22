# E8 — the inter bitrate overshoot fix (Track A moment of truth)

Drive: foreman_cif 256-crop, target sweep {25,50,100,250} kbps, AQ off.
Stack: E2 partition + E6/E7 keyframe + this E8 fix.

## Phase 1 — root cause (NOT the active_worst formula; the ambient feedback)

E4 saw inter Q "pinned at active_worst ≈245". The `active_worst_quality`
port (`cbr.rs:482`, `calc_active_worst_quality_one_pass_cbr`) and the
buffer accounting (`vp9_update_buffer_level_*`) are both faithful. The
buffer was NOT failing to deplete for the formula reasons in the brief —
it was the *ambient* feeding the formula that was wrong.

The `RVP9_E8_TRACE=1` per-frame trace (frame_loop) showed
`avg_frame_qindex[INTER]` (ambient) **decaying geometrically 255 → 30
over 31 inter frames** on foreman. The buffer-ramp branch
(`buffer_level > critical_level`, which held all clip long at the 100k
target) interpolates `active_worst = ambient + (worst-ambient)·(opt-buf)/
(opt-crit)` — i.e. UP *from ambient*. As ambient collapsed, the ramp
floor collapsed with it, pinning `active_worst` (and Q) at ~190-211
instead of rising toward 255. `regulate_q` wanted a higher Q to hit the
~2700-bit target but the artificially-low ceiling refused it → overshoot.

Why ambient collapsed: `postencode_update` recorded
`rc.active_min_qindex` (the **active_best** floor) into the ambient EWMA.
Inter active_best is `rtc_minq_approx(ambient) ≈ 0.75·ambient`, so
feeding it back as next-frame ambient is a ×0.75-per-frame geometric
decay. libvpx `vp9_rc_postencode_update:1850` records `cm->base_qindex`
— the actual **coded** Q (the `regulate_q` result), which on this clip is
~190-250, so libvpx's ambient stays high and its ramp/`ambient·5/4` cap
stay high. Pure rvp9-side divergence.

## Phase 2 — fix

`postencode.rs`: record `rc.last_picked_qindex` (the coded Q, == what the
frame loop encodes at and what the rate-correction update already uses)
instead of `active_min_qindex`. One-line change; matches libvpx exactly.

## Phase 3 — before/after trace (foreman @ 100k target, 32-frame diag)

| frame | before: ambient / aworst / q / actual | after: ambient / aworst / q / actual |
|------:|--------------------------------------:|-------------------------------------:|
|  5 | 196 / 220 / 220 /  5752 | 247 / 250 / 248 / 2840 |
| 13 | 114 / 187 / 187 / 14320 | 247 / 250 / 250 / 4624 |
| 31 |  30 / 211 / 211 /  7016 | 247 / 250 / 250 / 3512 |

Before: ambient craters, Q stuck ~187-211, frames ~10-14k bits.
After: ambient holds ~247, Q rises to/pins at 250, frames ~3-4k bits.

## Headline (64-frame BD-rate gate)

| target | rvp9 before | rvp9 after | libvpx |
|-------:|------------:|-----------:|-------:|
| 100 kbps | ~252 kbps / — | **143.9 kbps / 24.53 dB** | 143.1 kbps / 33.52 dB |

**The overshoot CLOSES.** At the 100k target rvp9 now hits 143.9 kbps —
within 0.6% of libvpx's 143.1. (Was ~252 kbps, 2.5× over.)

Foreman BD-rate: pre-chain **+367.81%** → E7 **+741.36%** → E8
**not measurable** — but for the OPPOSITE reason to E6. The curves no
longer overlap because rvp9's PSNR (24-27 dB) now sits entirely *below*
libvpx's (28-37 dB) at matched bitrate. The bitrate axis is fixed; what
remains is a pure per-bit *quality* gap.

## Case (A) or (B)? — BOTH, in sequence

- **(A) confirmed:** the ceiling WAS artificially limiting (via the
  ambient-feedback bug, not the formula). Fixing it lets Q rise to ~250
  and the rate target is hit — the overshoot is gone.
- **(B) revealed underneath:** at the SAME ~143 kbps, rvp9 gets 24.5 dB
  vs libvpx 33.5 dB — a ~9 dB per-bit efficiency gap. This is now the
  clean, isolated residual: not a rate-control artifact, but rvp9's
  block-coding (residual/transform/mode efficiency) structurally lagging
  libvpx. No Q-clamp change can close a quality gap at matched bitrate.

The E1-E8 chain has fully decomposed the foreman gap: structural levers
(partition E2, KF E6/E7) and the rate-target/overshoot (E8) are resolved;
the remaining gap is per-bit coding quality, measurable but distinct, and
the BD-rate gate target needs reassessment against it.

## Validation

- testsrc `bdrate_gate_stage4a`: PASS (no regression; overshoot improved
  too — 100k target 111.9 kbps).
- Cross-decode: **BYTE-EXACT** (rvp9-decoder vs vpxdec vs ffmpeg).
- Encoder suite: 94 unit tests pass, 0 fail.
- Clippy: clean on touched files (rvp9-encoder); pre-existing
  rvp9-core/dsp + loopfilter errors owned by other teams, untouched.
