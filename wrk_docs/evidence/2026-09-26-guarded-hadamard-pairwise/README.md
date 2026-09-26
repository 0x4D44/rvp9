# Guarded SSE2 Hadamard pairwise error: exactness and CPU screen

Code commit `f9afa25d` on the task branch changes only `quantize_metrics_sse2.rs`. When all eight signed `(coeff − dqcoeff)` differences fit `i16`, it uses SSE2 `pmaddwd` pairwise squares and zero-extends each `u32` pair sum before 64-bit accumulation. The existing wide reducer handles every overflow case. The signed-16-bit guard is exact because wrapping and saturating subtraction agree exactly in range. The prior [opportunity census](../2026-09-26-hadamard-metric-opportunity-census/README.md) found every measured TX8/TX16 group in range; that selected the experiment, not its predicted gain. Source SHA-256 was `9e2fb48cb581f8953ac9be7496e1d8bd2b7e0825eae9d0591ba2be6c37cfbc79` both locally and on Crucible.

Crucible built the candidate from the combined product at `def94924` plus that one-file change. Candidate release EXE SHA-256 is `3a239ed45e81266a742291205da30204519340aacf59ad8d68225a73dc695aef`; combined-product control SHA-256 is `f210620190da713917e0659e87b1f5145fb629b0eaa2ec709d21e76e140e9d85`; pinned libvpx v1.14.1 SHA-256 is `88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d`. Each run's `identity.csv` records binary and input hashes.

Three RVP9 candidate/control screens and one candidate/libvpx screen used the pinned 60-frame 1080p Ducks, Tree, and Town inputs at CPU6/CBR1260, one thread, no tiles, AQ0, noise0, lag0, and KF60. Each used one warmup per arm and clip plus 15 alternating-order pairs per clip (96 encodes). All **384** encodes exited zero and reproduced their arm's pinned IVF SHA-256. The CSV files retain every per-encode process CPU and wall time and every paired ratio. Lower ratios are better; overall is the geometric mean of the three clip ratios.

| Process-CPU ratio | Ducks | Tree | Town | Equal-clip overall |
| --- | ---: | ---: | ---: | ---: |
| Candidate / combined control A | 1.005510 | 0.988748 | 0.981177 | 0.991760 |
| Candidate / combined control B | 1.000770 | 0.978198 | 0.968329 | 0.982339 |
| Candidate / combined control, logical CPU 0 | 0.999964 | 0.984889 | 0.983708 | 0.989493 |
| Candidate / pinned libvpx | 1.300058 | 1.155886 | 1.062024 | **1.168613** |

All three within-RVP9 screens favour the candidate overall. Tree and Town improve every time; Ducks is neutral to 0.6% slower. The one-core run pins both RVP9 arms to logical processor 0 to limit migration. Crucible lacks `deltic quiet`; no strict host-load threshold was applied. Process CPU, not wall time, is the retention signal; one run had variable background load. The candidate's fresh libvpx gap is 16.9%, still above the 5% target. The prior combined-control/libvpx 1.188293× result ran separately, so its difference is directional, **not** a paired before/after estimate.

Validation: the new direct accumulator test covers narrow differences, the two-`i16::MIN` pair sum (`0x80000000`, requiring unsigned widening), and overflowing differences that must use the wide fallback. Deliberately sign-extending the pair sum made error differ from scalar (3,221,799,849 versus 7,516,767,145); after restoration, both SSE2 tests passed on actual x86 Crucible and under x86 Rosetta. Native AArch64 core/CLI checks, formatting, diff checks, and the Crucible release CLI build passed. The pinned-IVF checks cover full output identity on the measured workload.
