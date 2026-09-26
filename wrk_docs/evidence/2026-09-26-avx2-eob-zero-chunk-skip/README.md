# AVX2 EOB zero-chunk skip: exact prototype and CPU screen

The unintegrated code commit `db0c8c05` changes only `quantize_fp_avx2.rs`: when a 16-coefficient quantized chunk is all zero, the EOB reducer skips its inverse-scan load, mask, and maximum update. The selected x86 IP profile assigned 91/2,224 Ducks, 48/1,611 Tree, and 51/1,457 Town target-process samples to the existing EOB fallback. An output-neutral census found 93.90–96.69% zero chunks in the measured TX8/TX16 paths. Those observations selected the experiment; they do not predict its gain.

Crucible built the candidate in an isolated task worktree from `quantize_fp_avx2.rs` SHA-256 `e013a57de14a06b27fd9ff291a9ab37ee57ef18371db49a9b286f9e9abf5819c`. Its release EXE SHA-256 is `2fd3e98361b850ec33bea588011d61d9fe42a8c137fd984dfec151f123681c6b`. The integrated-code control EXE SHA-256 is `c0db2195fa6d2ab722fbbd9828c4c6541942f42080e65231ea861714f8540b56`; `git diff 07edb682..origin/main -- crates` was empty before this screen. The x86 release disassembly shows a `vptest`/conditional jump before the masked inverse-scan load and maximum update. The pinned libvpx v1.14.1 EXE SHA-256 is `88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d`.

Three paired RVP9 screens and one fresh candidate/libvpx screen used pinned 60-frame 1080p Ducks, Tree, and Town inputs at CPU6/CBR1260, one thread, no tiles, AQ0, noise0, lag0, and KF60. Each run used one warmup per arm and clip plus 15 alternating-order pairs per clip (96 encodes). All 384 encodes exited zero and matched their arm's pinned IVF SHA-256. Each run directory contains binary/input identities, per-encode process CPU and wall time, and paired ratios. Lower candidate/control ratio is better; overall is the geometric mean of the three clip ratios. The RVP9 runner is archived at `wrk_docs/evidence/2026-09-26-one-filter-result-forwarding/result-forward-screen.ps1`; the libvpx runner there is `result-forward-libvpx.ps1` with this candidate passed via `-CandidatePath`.

| Run, process CPU ratio | Ducks | Tree | Town | Equal-clip overall |
| --- | ---: | ---: | ---: | ---: |
| Candidate / integrated control A | 0.993247 | 0.989469 | 0.988033 | 0.990247 |
| Candidate / integrated control B | 0.987378 | 0.991786 | 0.970683 | 0.983240 |
| Candidate / integrated control, logical CPU 0 | 0.974295 | 0.989730 | 0.989456 | 0.984467 |
| Candidate / pinned libvpx | 1.331822 | 1.200331 | 1.068429 | 1.195357 |

All three within-RVP9 screens favour the candidate on every clip. The fixed-core run pins both arms to logical processor 0 to limit migration. Crucible lacks `deltic quiet`; no strict host-load threshold was applied. Paired process CPU, not wall time, is the retention signal. The fresh candidate/libvpx gap remains about 19.5%, well above the 5% target. It cannot be compared directly with the earlier integrated-control/libvpx run as a before/after estimate because those are separate two-arm runs.

Validation: on actual x86 AVX2 hardware, a deliberate wrong skip condition made the sparse TX8/TX16 differential fail (`Tx8x8` EOB 0 versus scalar 13); after restoration, all five `quantize_fp_avx2` tests passed. AArch64 core/encoder compile checks, formatting, diff check, and the portable sparse fixture test passed. The x86 release encoder built, and all pinned-IVF comparisons passed. The code branch remains unintegrated pending explicit permission for a code push.
