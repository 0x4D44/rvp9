# E6 — keyframe-boost: diagnosis & fix

Drive: foreman_cif 256-crop, 64 frames, 100 kbps, AQ off.

## Phase 1 — why q=177?

NOT the KF Q-bounds (first-frame `active_best = best_quality`, faithful)
and NOT the kf_boost ramp. Root cause: the KF **target** was hard-clamped.
`calc_iframe_target_size` first-frame branch requests
`starting_buffer_level/2` (200 000 b), then clamps to
`rc.max_frame_bandwidth`. rvp9 set `max_frame_bandwidth = 4*avg`
(≈13 348 b) — a divergence from libvpx `vp9_rc_update_framerate`, which
uses `max(MBs*250, 2_000_000, avg*400/100)` ≈ 2 Mbit. So the KF was
truncated to ~4× the inter target and coded at q=177/13 680 b.

## Phase 2 — fix

`state.rs`: ported the libvpx `max_frame_bandwidth` formula (the 2-Mbit
floor is operative at low res). KF-only in effect — inter targets sit
below `avg`, never near the ceiling. Inter `active_worst` clamp untouched.

## Phase 3 — mechanism check: it FIRES

| metric | before | after |
|---|---:|---:|
| KF Q / bits | 177 / 13 680 | 20 / 163 048 |
| inter coef-bits/frame | 5531 | **3239 (−41%)** |
| nonzero-coeff / skip0-leaf | 6.9 | 3.9 |
| skip=1 share | 59% | 67% |

Better KF → better LASTFRAME → smaller residuals → fewer coef bits. The
reference-quality hypothesis is confirmed. Inter Q stayed pinned at the
ceiling (212→228) but the bits *at* that ceiling fell sharply.

## Caveat — high-bitrate KF overspend

rvp9's KF `bits_per_mb` under-predicts, so regulate_q drives KF Q to 4–20
(libvpx q=57/124), overspending the now-uncapped target. The 250 kbps
PSNR regressed 28.98→27.48 dB, dropping below libvpx's range → 4-point
foreman BD-rate went from measurable **+367.81%** to **not measurable
(no PSNR overlap)**. Follow-up lever: KF rate model / KF-side multiplier.
