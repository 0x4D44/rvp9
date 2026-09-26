# Hadamard full-metric arithmetic: exact x86 opportunity census

This is an output-neutral opportunity census, **not a speed measurement**. Temporary diagnostic commit `777b69a6` on `task/20260926-TSK-HUM-measure-hadamard-metric-fast-path-opport` counts eight-coefficient groups in the x86 SSE2 full-metric quantizer. It does not change `(nonzero, satd, error)` or mode decisions, and is not integrated. Its source SHA-256 values are `e7e79b085288e4dbdb86cbad65c02361057d51e45fb67f3d54afec71c914c7e6` for `quantize_metrics_sse2.rs`, `53a890d3ba697969b4aada8f6b5d4b54c426b06ada32dc590b91278bfc5cacad` for `dsp/mod.rs`, and `27bb7966ec997ecc1a450df9ca0a3a8eac6d544043fdbe462bb8355f40022931` for `rvp9enc.rs`. The copied Crucible sources matched all three hashes. The release diagnostic EXE SHA-256 is `0a1c214d5f63245f07f87e14e20f2597a297d501f600fb3e77ec693c4cbc4170`.

The exact integrated EXE (`c0db2195fa6d2ab722fbbd9828c4c6541942f42080e65231ea861714f8540b56`) and archived [Tree/Town sampled IPs](../2026-09-26-exact-subpel-pass-profile/README.md) give a more specific target than the broad `inter.rs:5590` source line. In that hash-matched EXE, RVAs `0x87e1d–0x87e8c` and `0x87ef7–0x87f6c` contain the inlined signed-extension, subtraction, absolute-value, `vpmuludq`, and 64-bit accumulation loops. They hold 113/1,611 Tree and 84/1,457 Town target-process samples (7.01% and 5.77%); Ducks has 2/2,224 (0.09%). These are sampled self-work ceilings, not predicted savings.

The runner here uses the pinned 60-frame 1080p CPU6/CBR1260 one-thread Ducks, Tree, and Town inputs, one warmup and one repeat for each control/candidate arm. All 12 encodes exited zero and matched the pinned IVF hash for their clip. The two diagnostic counts per clip were identical. `identity.csv`, `raw.csv`, and the six candidate stderr files retain the binary/input hashes and every reported count. The control ignores the diagnostic environment variable. Timings in `raw.csv` are invalid for speed comparison because the candidate scans every group and updates atomics.

| Clip | TX8 groups | TX8 all-zero quantized | TX8 zero share | TX16 groups | TX16 all-zero quantized | TX16 zero share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ducks | 199,936 | 190,471 | 95.27% | 1,664 | 1,600 | 96.15% |
| Tree | 43,812,480 | 38,968,146 | 88.94% | 85,760 | 60,987 | 71.11% |
| Town | 32,098,560 | 27,079,494 | 84.36% | 77,952 | 55,295 | 70.93% |

**Every measured TX8 and TX16 group had all eight signed `(coeff − dqcoeff)` differences within `i16`.** That supports testing a guarded SSE2 pairwise-square path, with the existing wide path retained for arbitrary inputs. The zero-quantized share separately suggests a zero-dequant path, but a branch must repay its cost; neither share proves a CPU gain. The guarded production prototype is a separate task and must pass full `(nonzero, satd, error)` differentials, exact pinned output, and repeatable paired process-CPU screens before retention.

The focused x86 counter test failed deliberately at expected total 4 versus observed 3, then passed after restoration. Native AArch64 and portable x86 core/CLI checks, formatting, and diff checks passed; Crucible's actual x86 test and release build passed. No instrumented CPU ratio is retained.
