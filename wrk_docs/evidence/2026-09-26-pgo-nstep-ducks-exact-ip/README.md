# Integrated-code PGO Ducks exact-IP profile

Three fresh-process Crucible WPR captures profiled the PGO encoder built from remote commit `ddedb6c5`, whose shipped source is equivalent to integrated trunk `a8f63fbe` (the NSTEP duplicate-variance change). The exact EXE SHA-256 is `3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee`; the PDB SHA-256 is `7672ad045a634a9278d7b9255e1a0b665e9e729f6a480438e9b22c50d330bc13`. Extraction checked their PE/PDB GUID `d13af019-ee7e-d008-4c4c-44205044422e`, age 1, and image size `0x211000` on each run. The pinned 60-frame 1080p Ducks CPU6/CBR1260 encode reproduced IVF SHA-256 `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089` three times.

The runs retained 16,810, 21,815, and 21,590 direct encoder samples: **60,215 total**, with zero lost WPR buffers and events. Encoder process CPU was 2.1719, 2.7969, and 2.7969 seconds; these are profiling-run observations, **not** a paired speed comparison. The per-run records, exact PE/PDB extraction summaries, aggregate CSVs, and the hash-pinned capture/extraction/aggregation scripts are here. Raw ETLs and full-IP rows remain under `D:\worktrees\rvp9-2\20260926-TSK-HUM-prototype-nstep-duplicate-variance-score\.scratch\pgo-nstep-ducks-ip\` on Crucible; the ETL hashes are in the run verification records. The ETLs are too large for Git.

| Exact RVA | Samples; run 01/02/03 | Disassembled sampled instruction | Source attribution |
| --- | ---: | --- | --- |
| `0xaa5d1` | 1,582; 409/545/628 | partition closure; inspect before attributing an operation | `lib.rs:11471` |
| `0xe3d33` | 1,462; 411/521/530 | picker closure; inspect before attributing an operation | `inter.rs:2428` |
| `0x10be76` | 725; 184/284/257 | `movq %r12, %r13`, followed by row-offset multiplication and bounds branches | `sub_pixel.rs:1341`, inlined AVX2 bilinear load |
| `0x153251` | 720; 190/276/254 | inter-prediction setup; inspect before attributing an operation | `scale.rs:499` |

The `0x10be76` neighborhood performs AVX2 bilinear arithmetic and then repeated row-offset/bounds work. `load_bytes_bounded` in `crates/rvp9-core/src/dsp/x86_64/subpel_variance_avx2.rs` checks the source slice on each left/right chunk. This is a *new* bounded candidate: prove the complete source footprint once, then remove repeated checks while preserving exact-width loads, arithmetic, and a checked fallback. The previous one-axis halfpel prototype changed interpolation arithmetic and does not test this mechanism. The 725 samples are **1.20% of all direct samples**, not a predicted saving: instruction sampling can skid, and nearby dependencies may own the cost. Conversely, the larger partition and picker symbol counts are broad inlined regions, not an identified removable operation; earlier DC-load reuse and denoiser-minima experiments did not give reliable gains.

The xperf dumper emitted a malformed CSV quote around a kernel `Unknown` module under the target process in run 02. The exact-image extractor now accepts only such non-encoder kernel samples after relaxed parsing; malformed target user-mode rows still fail. A focused regression was deliberately red before the fix and green after it. Re-extracting run 01 with the fix reproduced the prior `samples.csv` SHA-256 exactly: `71535f0103a877a0c50b9017d75c68092012d66b9b24ab9e17a24a0747871b14`.

This profile does **not** update the production parity ratio. The latest direct equal-clip PGO/libvpx ratio remains **1.099305×**, measured separately in `../2026-09-26-nstep-duplicate-variance-pgo/README.md`, above the ≤1.05× goal.
