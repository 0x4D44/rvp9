# One-filter mode-result forwarding: exact prototype and CPU screen

The unintegrated code commit `b5641550` changes only `mode_decision/inter.rs`: `evaluate_candidate_with_slots` returns `InterModeSearchResult` directly and writes the winning prediction-slot enum through an out-parameter. Candidate order, scoring, tie handling, and pixel slots are unchanged. The exact integrated x86 IP profile assigned 46/2,224 Ducks target-process samples (2.07%) to the old one-filter result-copy load at RVA `0x84d3f`; Tree and Town assigned 11/1,611 and 11/1,457 samples. That sample share was a reason to test, not a promised gain.

Crucible built the candidate from the `inter.rs` source with SHA-256 `36d847d96f562c57d65cf100b26bcfde8963c10b9d153296f824dc1a6f3527a0` in an isolated task worktree. Its release EXE SHA-256 is `0041a3a4cedc6aac59a29d4039a09c0981ec793a26d5bbafe0b437cab1efc83b`. Control EXE SHA-256 is `c0db2195fa6d2ab722fbbd9828c4c6541942f42080e65231ea861714f8540b56`, built at `07edb682`; `git diff 07edb682..origin/main -- crates` was empty, so it represents the current integrated product code. Both build identities and the pinned input hashes are in each run's `identity.csv`.

The x86 release disassembly confirms the intended machine-code effect. The control calls `evaluate_candidate_for_filter_with_pred` at RVA `0x84d2f` into `[rsp+0x198]`, loads two ZMM chunks at `0x84d34` and `0x84d3f`, then stores them to the caller's result in the common epilogue at `0x854c5`/`0x854cf`. The candidate's `evaluate_candidate_with_slots` at RVA `0x84500` instead calls that evaluator at `0x84fb7` with the caller's result pointer in `rcx`, immediately restores the frame and returns. The one-filter temporary copy is gone. This is not evidence that the whole path saves 2.07% CPU.

Three paired RVP9 screens and one fresh candidate/libvpx screen used the pinned 60-frame 1080p Ducks, Tree, and Town inputs at CPU6/CBR1260, one thread, no tiles, AQ0, noise0, lag0, and KF60. Each used one warmup per arm and clip plus 15 alternating-order pairs per clip (96 encodes). All 384 encodes exited zero and reproduced each arm's pinned IVF SHA-256. The CSV files retain identities, per-encode process CPU and wall time, and paired ratios. Lower candidate/control ratio is better; the overall column is the geometric mean of the three clip ratios. The pinned libvpx v1.14.1 EXE SHA-256 is `88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d`.

| Candidate/control run | Ducks | Tree | Town | Equal-clip overall |
| --- | ---: | ---: | ---: | ---: |
| A | 0.992111 | 0.988102 | 0.983731 | 0.987975 |
| B | 0.973481 | 0.999881 | 0.988898 | 0.987361 |
| One-core affinity | 0.970403 | 0.979733 | 0.964403 | 0.971493 |
| Fresh candidate/libvpx | 1.331043 | 1.187042 | 1.092979 | 1.199748 |

All three within-RVP9 equal-clip runs favor the candidate. Tree is neutral in run B, but Ducks and Town favor it in every run. The one-core run pins both RVP9 arms to logical processor 0 to limit migration. Crucible lacks `deltic quiet`; no strict host-load threshold was applied. Paired process CPU, not wall time, is the retention signal. Fresh candidate/libvpx parity is about 1.200×, still well short of the 1.05× target. The prior 1.182520× result was measured in a different run; do not infer a before/after parity delta from those separate libvpx comparisons.

Validation: a deliberate wrong-slot mutation made `output_backed_one_filter_preserves_result_and_poisoned_stride` fail on row-0 pixels; after restoration, all three output-backed tests passed on AArch64 and x86. Core and encoder compile checks, formatting, and diff checks passed on AArch64; the x86 release encoder built. The paired pinned-IVF checks cover full-output identity. The code remains unintegrated pending explicit permission for a code push.
