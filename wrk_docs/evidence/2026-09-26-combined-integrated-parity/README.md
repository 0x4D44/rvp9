# Combined integrated parity after three retained gains

Crucible built `rvp9enc.exe` from clean `origin/main` `def9492429ec7bb6b343ff5c6338cb7e79cb10b3`, which includes the zero-EOB transform-selector specialization (`c802c099`), one-filter mode-result forwarding (`a4df9e5a`), and AVX2 EOB zero-chunk skip (`def94924`). The release EXE SHA-256 is `f210620190da713917e0659e87b1f5145fb629b0eaa2ec709d21e76e140e9d85`. Pinned libvpx v1.14.1 SHA-256 is `88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d`.

The runner here uses pinned 60-frame 1080p Ducks, Tree, and Town Y4Ms at CPU6/CBR1260, one thread, no tiles, AQ0, noise0, lag0, and KF60. Each clip had one warmup per arm plus 15 alternating-order pairs. All 96 encodes exited zero and matched their arm's pinned IVF SHA-256. `identity.csv`, `raw.csv`, and `summary.csv` retain exact binary/input identities, per-encode process CPU and wall time, and paired ratios. Lower RVP9/libvpx process-CPU ratio is better. The equal-clip overall is the geometric mean of the three clip ratios.

| Ducks | Tree | Town | Equal-clip overall |
| ---: | ---: | ---: | ---: |
| 1.309978× | 1.187078× | 1.079014× | **1.188293×** |

This is the fresh combined-product parity number, still above the 1.05× target. The previous integrated-control/libvpx run was 1.206728×, but it ran separately before these three landings, so its directional difference is **not** a paired before/after estimate. The three within-RVP9 A/B screens for each change are archived in their respective evidence directories and establish individual retention; their gains are not assumed additive here. Crucible lacks `deltic quiet`; no strict host-load threshold was applied. Paired process CPU, not wall time, is the comparison signal.

On the exact combined trunk, Crucible's actual x86 `quantize_fp_avx2` focused tests passed 5/5, including the sparse EOB regression; the release CLI built cleanly. The pinned IVF checks cover end-to-end output identity on this workload. No full workspace suite or BD-rate gate was run for this measurement.
