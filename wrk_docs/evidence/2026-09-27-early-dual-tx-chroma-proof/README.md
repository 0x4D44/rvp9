# Early dual-TX chroma proof: exact-output PGO screen

This evidence belongs to the bounded prototype on the task branch. The integrated control is the production-PGO encoder SHA-256 `3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee`, which established the 1.099305× pinned-libvpx equal-clip CPU baseline. The candidate source is local commit `f78e7d89`, mirrored into native Crucible worktree commit `0cc6405da`; source-file hashes matched before native tests and PGO build. The build uses the unchanged 28-row, 60-frame production PGO recipe on Crucible.

The paired runner checks pinned source hashes and canonical IVF output hashes, warms both arms, alternates arm and clip order, and retains every pair. It records process CPU separately from wall time. Its raw outputs and summary files are preserved below. A long wall-time interruption in the first screen's Ducks candidate warm-up does not enter any paired CPU ratio. The third screen's last pairs saw elevated process CPU in both arms; they remain in the results.

All 264 encodes across the three paired screens and direct comparison exited zero and matched their arm's expected IVF SHA-256. The candidate optimized EXE was SHA-256 `e3e4307c4f5d0570dbfd60abca6ce3f4d876e968b33f7d9bba9a4fb75b235c42` in every run; its source file was SHA-256 `eef9e6fed0f9790ea08249f1ba93b3d9562683008e437a8bc545e35d77c7e27f`. [The build wrapper](build-pgo-early-crucible.sh), [manifest](build-manifest.txt), and [paired runner](screen-early.ps1) retain the exact commands and compiler identity. The x86 worktree passed the two new focused proof/on-off tests and `cargo check --locked -p rvp9-encoder`; AArch64 passed those plus the existing compact-row, border, scaled-predictor, residual-scratch, and TX-scratch checks. The on/off three-frame Foreman IVF SHA-256 was identical, `babe148644f645bf56bf3f96c472f85c062330ca1336a3209c04e344b24b094e`; the packet test asserted a positive bypass hit. Both new tests were deliberately made red and restored.

| Candidate/control process CPU | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| Screen 1, 5 pairs | 0.928791× | 0.936667× | 1.007855× | 0.957122× |
| Screen 2, 10 pairs | 0.927605× | 0.978263× | 0.986659× | 0.963819× |
| Screen 3, 10 pairs | 0.969581× | 0.997254× | 0.982090× | 0.982910× |
| Pooled, all 25 pairs per clip | 0.944414× | 0.977286× | 0.989024× | 0.970056× |

The candidate saved about 3.0% equal-clip process CPU pooled, with a positive overall gain in each independent screen. Ducks improved most. The third screen's CPU surge in both arms increased noise and reduced its apparent gain; no pair was excluded. The result is a measured code gain, not a guarantee that the same percentage applies to a different comparison run.

A fresh 15-pair direct comparison against pinned libvpx `vpxenc.exe` SHA-256 `88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d` measured candidate/libvpx CPU of **1.202095× Ducks, 1.021863× Tree, 0.974605× Town, 1.061826× equal-clip**. This improves on the prior direct 1.099305× baseline, but remains above the ≤1.05× parity target. The direct figure, not the cross-run product of two ratios, is the new parity reading. Training includes the benchmark source clips, so this is not held-out-content evidence.

The committed raw `screen-0{1,2,3}-{identity,raw,summary}.csv` and `direct-libvpx-{identity,raw,summary}.csv` files retain every input, output hash, CPU/wall time, and pair ratio. No code should be integrated without checking the exact tree after any rebase.
