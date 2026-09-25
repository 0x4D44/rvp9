# Unit 1435 — interior 64×64 integer-projection AVX2 prototype rejected

The integrated three-clip IP map gave the fused x86 SSE2 projection only 0.92% of Ducks, 1.49% of Tree, and 1.58% of Town direct encoder samples. I tried a narrow AVX2 replacement with runtime AVX2 detection and the existing SSE2 fallback. The exact prototype is retained only in the local task branch `task/20260925-TSK-HUM-x86-int-pro-64x64-avx2-prototype` at `ff7d2215` (not pushed or integrated).

The new AVX2 differential matched the scalar four-vector oracle across 18 patterns and three stride pairs on Crucible. Deliberately adding one to every source vertical-projection lane made it fail on the all-zero pattern; restoring the source made it pass again. `cargo fmt --all --check`, a native AArch64 `cargo check -p rvp9-core`, and a clean x86 cross-target test compile also passed. The candidate release EXE SHA-256 was `c481b8d351bbe9ca95f31cb6699e0f7df1d8d7e754b7d3a5d2ad081299ac90f4`; its PDB was `8213a58eb364200b454f9b2549d11ff88df3a5ba91ccf0a7c9745f0bad64e7e8`. The pinned integrated baseline EXE/PDB hashes were `1b64fc5f3de6dc3cfe9d486f6b2515be7a44354eeb0b025ca28e6e9903d1470a` / `27c0b186f4a2a5b2edd19a665ccf8cc1bff1786c90df3a906ed9a1a6e58f72b4`, built from shipped commit `88644d86` (later trunk movement was docs-only).

Fifteen alternating-order paired runs per clip, with one warmup per arm, used the same pinned CPU6/CBR1260 one-thread 60-frame inputs and exact IVF hashes as Unit 1434. All 90 measured rows and six warmups had exit 0, no timeout, pinned assets, and exact output hashes. No rows were excluded; host load and wall time are diagnostics. The paired **process-CPU** geometric-mean candidate/baseline ratios were:

| Ducks | Tree | Town | Equal-weight overall |
| ---: | ---: | ---: | ---: |
| 1.015913× | 1.012355× | 0.992507× | **1.006872×** |

Independent computation from `raw.csv` reproduced those ratios. The first seven and last eight pairs were 1.009240× and 1.004805× overall; odd/even-order pairs were 1.002188× and 1.012253×. The prototype is exact but does **not** show a reproducible whole-encode gain, so its code stays off trunk. The last separate pinned-libvpx process-CPU ratio remains 1.182520×; no new libvpx screen was warranted for a rejected candidate.

`int-pro-run-ab.ps1` is the exact runner. `identity.csv`, `raw.csv`, `load.csv`, `summary.csv`, `output-stability.csv`, and `manifest.txt` preserve the run. Encoded IVFs remain in Crucible's task worktree `.scratch/int-pro-run-ab/run-20260925T220657.320Z/`; only their expected hashes and per-run hashes are checked in.
