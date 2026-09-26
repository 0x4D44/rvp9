# Denoiser-off minimum tracking: exact, not repeatably faster

The `denoiser-off-minima.patch` prototype gave the output-backed inter picker an explicit no-denoiser entrypoint. The encoder called it only in its branch guarded by `job_denoiser.is_none()`. Existing public pickers kept collecting denoiser candidate minima. Exact-image Crucible IP attribution found 33/2,072 Ducks self samples in the minimum-update bookkeeping, an optimistic 1.59% Ducks ceiling, not a speed prediction. The real TLS accesses on the surrounding source line had zero samples.

The new focused differential compared the complete normal and opt-out pick results after normalizing only `.denoiser`, and compared their retained prediction bytes. Before the opt-out was implemented, the test failed because all three minima remained populated; after implementation, it passed with an empty opt-out context. All 96 focused core inter-picker tests passed on both AArch64 and x86, including the existing denoiser-on context test. The focused encoder check, format, and diff checks passed. Source SHA-256: `inter.rs` `1477478a67cb005ec4bffb642f78d338f7bd3f8db513d85a889ac24c83201d30`; encoder `lib.rs` `2120f83e27c0b1da1676ab8dbc0d11e9e9040f6dcb44c27d06953c2ab58111d8`.

Three Crucible screens used the pinned 60-frame 1080p Ducks, Tree, and Town clips at CPU6/CBR1260, one thread, no tiles, AQ0, noise0, lag0, and KF60. Each arm had one warmup per clip and 15 alternating-order pairs. Screen 03 pinned both arms to logical CPU 0. Control was clean integrated code commit `48cab42d` (source-equivalent to the later docs-only commits), EXE SHA-256 `c4b24b2c5b16add984a2dd6d32e5a7080037290e58b254aff838a0dcf26d63c1`; candidate EXE SHA-256 was `2aca121e3106078632449a0afa29b5b94ea3d59b71231d0728f89ad174c96fd3`. All **288** encodes exited zero and matched their expected IVF SHA-256. The checked-in `screen*/identity.csv`, `raw.csv`, and `summary.csv` retain every observation. Ratios below were independently recomputed from raw CPU seconds.

| Candidate/control process-CPU ratio | Ducks | Tree | Town | Equal-clip overall |
| --- | ---: | ---: | ---: | ---: |
| Screen 01 | 0.982756 | 1.015559 | 0.986295 | **0.994762** |
| Screen 02 | 1.002148 | 1.018521 | 1.003875 | **1.008154** |
| Screen 03, logical CPU 0 | 1.006080 | 0.971215 | 0.997654 | **0.991538** |

The combined 135 paired clip observations give **0.998126×**, an apparent 0.19% saving. Signs reverse across screens and clips; Tree regressed in two screens, and screen 02 regressed everywhere. One screen 03 Ducks pair had a 1.509× CPU ratio and remains in the calculation. Crucible wall stalls reached 15 seconds; no pairs were dropped or selected by host load. This does **not** show a repeatable production gain, so the code was not integrated. No fresh pinned-libvpx comparison was run; the latest integrated parity result remains 1.134647× equal-clip process CPU.
