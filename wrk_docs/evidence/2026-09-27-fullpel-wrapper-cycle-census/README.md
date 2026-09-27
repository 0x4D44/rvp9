# Ducks LAST full-pel wrapper timing discriminator

The fresh-profile diagnostic does **not** justify a production search-adapter rewrite. Across ten alternating-order Ducks pairs in two screens, RVP9's complete LAST/NSTEP full-pel wrapper used 1.006–1.020× libvpx's TSC cycles (screen geometric means **1.012273×** and **1.013193×**). The wrapper difference is small, and the wrappers do different work; it is not an estimate of removable encoder CPU. This was an instrumented diagnostic, not a new parity screen. The latest uninstrumented, integrated, production-PGO direct result remains **1.057638×** pinned libvpx equal-clip process CPU against the ≤1.05× target.

Both codecs encoded the same 60-frame 1080p Ducks input (SHA-256 `6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7`) at CPU6, CBR1260, one thread, no tiles, AQ0, noise0, lag0, and KF60. All twenty paired encodes exited zero. RVP9's IVF was always the canonical `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`; libvpx's was always `b24234d8db46c22e03585f1973b27c2e62315f0464da392a2ecd2769d2124fa8`. Every run reconciled timed entries with the unchanged [operation census](../2026-09-27-cross-codec-last-search-work/README.md): RVP9 393,364 LAST/NSTEP entries; libvpx 378,013. The second screen ran under a three-minute Crucible quiet lease with zero pre-existing heavy builds. The x86 focused RVP9 census test passed. On AArch64, the diagnostic test was deliberately made red by omitting its timed-entry increment, then restored green; formatting, CLI build, libvpx build, and gate-off/on exact-output checks passed.

| Pair | RVP9 LAST cycles | libvpx LAST cycles | RVP9/libvpx |
|---:|---:|---:|---:|
| 1 | 926,316,277 | 917,457,976 | 1.009655× |
| 2 | 925,654,249 | 918,600,658 | 1.007679× |
| 3 | 926,426,142 | 914,355,612 | 1.013201× |
| 4 | 935,924,799 | 917,649,068 | 1.019916× |
| 5 | 926,375,101 | 916,335,203 | 1.010957× |

The quiet-lease repeat was consistent:

| Pair | RVP9 LAST cycles | libvpx LAST cycles | RVP9/libvpx |
|---:|---:|---:|---:|
| 1 | 940,786,852 | 926,174,506 | 1.015777× |
| 2 | 931,419,130 | 913,796,053 | 1.019286× |
| 3 | 931,872,350 | 926,417,542 | 1.005888× |
| 4 | 942,458,343 | 929,867,604 | 1.013540× |
| 5 | 938,278,834 | 927,590,539 | 1.011523× |

`LFENCE;RDTSC` brackets RVP9's `nstep_search_sad` call at `run_newmv_search` and libvpx's whole `vp9_full_pixel_search` NSTEP body after reference attribution. The [RVP9 patch](rvp9-source.patch.b64) and [libvpx patch](libvpx-source.patch.b64) are diagnostic-only additions to the prior operation-census code; decoded patch SHA-256 values are `937c676846479f573d27449b39c7f825e0553eaefba2d938404792846e68beb7` and `59431cdf2b5b4933a72efe4852ab09b0c1e653f403251aeab7f9562fc00d8c0c`. The [first](paired-stderr.tar.gz) and [quiet-repeat](quiet-paired-stderr.tar.gz) stderr archives retain all shape rows and exact cycle counts. Neither patch belongs in production.

RVP9 source was integrated `22ddce38e6d80f655d7fe433c0ce221b8c4d1702` plus the patch; libvpx was pinned `12f3a2ac603e8f10742105519e0cd03c3b8f71dd` plus its patch. The fresh RVP9 28-row, 60-frame PGO build used `scripts/build-encoder-pgo.sh` with the canonical training corpus and emitted no control-flow hash mismatch warnings. Its profile SHA-256 was `3ba7bf8c6ebce76cacb307edffb5f79c5f200ad26fcdc139b27871eb3c61e170` and EXE SHA-256 `238e1f4d8424d629a0bed23158848077385250485f683f3d1d2e6f0a6b890438`. The libvpx diagnostic EXE SHA-256 was `eebea22f2a063135626e983f43857a45dde56c9098d62ee85276745d0df45545`.

An earlier RVP9 diagnostic build reused the integrated profile and emitted **11 control-flow hash mismatch warnings**, including these hot search functions. Its three apparent 6.5–7.7% wrapper wins are not comparable and are superseded by the fresh-profile pairs. The full-pel call boundaries still differ, and TSC cycles in an instrumented run are not production process CPU; no parity or speed claim follows. Continue looking for a specific exact-output kernel change with a larger measured ceiling, then gate it on paired production-PGO CPU.
