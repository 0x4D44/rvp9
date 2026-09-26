# AVX2 quantizer DC source reuse: exact, but slower

The current 28-row production PGO Ducks executable spends 1,184 of 52,418 direct samples at RVA `0xaa825`, a scalar load of `coeff[0]` in the inlined AVX2 quantizer. The immediately preceding code has already loaded the first coefficient vector. This motivated reusing that vector's DC lane rather than loading `coeff[0]` from memory again. The sample count does not establish the load's self-cost; instruction-pointer skid and dependencies remain possible.

The unintegrated prototype is local commit `40f420bd` on `task/20260926-TSK-HUM-prototype-tx4-vector-dc-source-reuse`. It extracts DC from the first AVX2 input vector and reuses that vector in the first quantizer loop iteration. The current PGO image has `vmovdqu 0x2000(%r13), %ymm3` followed by `movl 0x2000(%r13), %eax` at `0xaa825`. The candidate image has a first vector load at `0xaa528` and `vmovd %xmm4, %eax` at `0xaa5b4`; the duplicate scalar memory load is absent on that path. This proves the intended machine-code change, not a speedup.

On Crucible, all seven x86 AVX2 quantizer tests passed. The existing zero-EOB scalar differential failed deliberately when the wrong DC lane was extracted (TX4 DctDct DC-only EOB 0 rather than 1), then passed after restoring the correct lane. On AArch64, 24 forward-quant tests and `cargo check --locked -p rvp9-encoder` passed. Formatting and diff checks passed. These checks and the paired screen's IVF hashes support exact output for the tested paths.

Crucible built the candidate with the unchanged 28-row, 60-frame production PGO recipe. [Build manifest](build-manifest.txt) records the build identity. Candidate EXE SHA-256 was `f71d4b122f3a606917074d28d9ba7d6d1b9a0f5ab386d5eb4af57b85617e5a74`; control EXE SHA-256 was `0fa1028b1310fee93b22731794dcb3fdc14e9b5e05dbe428f3e112c4b30cb395`. Candidate source SHA-256 was `5b55c450835fb278accc0863237812925bb2eb5674c3db13b4ae6787667f2bfa`. The [build wrapper](build-pgo-dc-source-crucible.sh) and [paired runner](screen-dc-source.ps1) are the exact scripts used.

Both independent Crucible screens used the pinned 60-frame 1080p Ducks, In-to-Tree and Old-Town inputs at CPU6/CBR1260, one thread/no tiles, AQ0/noise0/lag0/KF60. Each warmed both arms, alternated arm and clip order, and retained every pair. All **132 encodes** exited zero and reproduced established IVF hashes. Each `screen*/identity.csv`, `raw.csv`, and `summary.csv` preserves executable and input hashes, process CPU/wall time, output hashes, and per-pair ratios. Lower candidate/current CPU is better.

| Candidate/current process CPU | Ducks | Tree | Town | Equal-clip overall |
| --- | ---: | ---: | ---: | ---: |
| Screen 01, 5 pairs | 1.006137× | 0.987026× | 1.021000× | 1.004625× |
| Screen 02, 15 pairs | 1.018599× | 0.999180× | 1.022708× | 1.013444× |

Ducks and Town were slower in both screens; the longer screen regressed overall by 1.35%. **Do not integrate this code.** Removing the duplicate load did not remove the actual Ducks bottleneck; the vector-to-GPR transfer or surrounding dependencies may offset the saved load, but this is an inference, not a proven cause. The current direct production PGO/libvpx result remains **1.096110×** equal-clip CPU, above the ≤1.05× target. No candidate/libvpx screen was run because the paired RVP9 control gate failed.
