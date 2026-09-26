# AVX2 inverse-scan EOB fallback: zero-chunk census

This is an **output-neutral opportunity count, not a speed comparison**. Diagnostic source commit `8837e98e` (unintegrated) counts 16-coefficient raster chunks seen by `eob_from_iscan` after the dense-tail probe misses. Its env gate is `RVP9_EOB_CHUNK_CENSUS=1`; the extra scans and atomics make the diagnostic binary unsuitable for timing. The production quantizer is unchanged on `origin/main`.

The exact integrated x86 profile assigned 91/2,224 Ducks, 48/1,611 Tree, and 51/1,457 Town process samples to the quantizer's fallback source line. In the hash-matched control EXE, RVA `0x8c8c3` is `vptestmw k1,ymm1,ymm1`, followed by a masked inverse-scan load and `vpmaxsw`. The zero test is already present; a new branch could only avoid some work *after* it, and its own cost remains unmeasured.

Crucible built diagnostic commit `8837e98e` in an isolated task worktree. Release EXE SHA-256: `0652877f0d8adbf2702ff654b1c6438983b51a4e15cb8631edeb195b652c3889`. Control SHA-256: `c0db2195fa6d2ab722fbbd9828c4c6541942f42080e65231ea861714f8540b56`. The runner checked pinned input hashes and ran one warmup plus one paired 60-frame CPU6/CBR1260 encode per clip. All 12 encodes exited zero and matched the pinned integrated IVF SHA-256. `raw.csv`, `identity.csv`, and the three candidate stderr files preserve the checks and counters.

| Clip | TX8 fallback calls | TX8 zero chunks / all chunks | TX8 all-zero blocks | TX16 fallback calls | TX16 zero chunks / all chunks | TX16 all-zero blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ducks | 3,030,342 | 11,636,651 / 12,121,368 (96.00%) | 2,556,502 (84.36%) | 505,382 | 7,592,934 / 8,086,112 (93.90%) | 240,537 (47.60%) |
| Tree | 1,383,352 | 5,331,218 / 5,533,408 (96.35%) | 1,214,613 (87.80%) | 65,387 | 872,826 / 1,046,192 (83.43%) | 19,896 (30.43%) |
| Town | 1,504,914 | 5,820,301 / 6,019,656 (96.69%) | 1,337,053 (88.85%) | 59,680 | 779,218 / 954,880 (81.60%) | 23,078 (38.67%) |

The focused counter fixture deliberately failed with expected `3` versus actual `2` zero chunks, then passed after restoration. AArch64 and x86 CLI checks, all 14 native CLI tests, formatting, and diff checks passed; Crucible's x86 focused counter test and release build passed. The rates justify a bounded zero-chunk branch prototype with exact quantizer and IVF differentials, then paired process-CPU screening. They do not predict its gain.
