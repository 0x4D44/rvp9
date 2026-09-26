# Rejected 32-wide variance two-chain prototype

The exact-image Crucible profile made the second 16-byte chunk of the generic 32/64-wide AVX2 variance reducer a bounded candidate. The prototype split the 32-wide row's two chunks into independent `(sum, sse)` accumulator pairs, folding once at the end. It left other widths and the retained 16×16 four-chain path unchanged. The unintegrated code is in `w32-variance.patch.gz`; `w32-screen.ps1` records the pinned paired method.

The new 32-wide scalar differential failed deliberately on its first 32×16 case when the second source offset was changed from 16 to 15, then passed after restoration. Crucible's focused AVX2 suite passed 10/10; the AArch64 variance suite passed 35/35; formatting, diff check, and x86 cross-compilation passed. All 36 screen encodes (six warmups and 30 paired encodes) exited successfully and matched the established three IVF SHA-256 values.

The five alternating-order pairs per clip used CPU6/CBR1260, one thread, 60 frames, and the same pinned inputs and expected output hashes as the post-zero-EOB profile. The control was exact integrated native-release executable SHA-256 `c4b24b2c5b16add984a2dd6d32e5a7080037290e58b254aff838a0dcf26d63c1`; `identity.csv` records the candidate and input hashes. `raw.csv` keeps every process-CPU and wall-clock observation; `summary.csv` gives each clip's geometric mean of paired candidate/control process CPU. No pairs were dropped for host load.

| Clip | Candidate/control process CPU |
| --- | ---: |
| Ducks | 1.083032× |
| Tree | 1.014132× |
| Town | 1.015603× |
| Equal-clip geometric mean | **1.037098×** |

The candidate is slower, especially on Ducks, despite exact output. No code was integrated and no pinned-libvpx parity remeasurement is warranted. The latest integrated parity measurement remains 1.134647× equal-clip process CPU.
