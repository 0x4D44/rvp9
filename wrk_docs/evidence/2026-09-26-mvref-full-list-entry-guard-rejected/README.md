# Realtime MV-reference full-list entry guard: exact, not repeatably faster

Reject the source candidate. The realtime LAST/GOLDEN MV-reference search already caps each list at two entries. Candidate commit `c145f0ca` skips the cached-neighbour scaled-candidate pass when **all** requested lists are full after the direct and previous-frame passes. The later scaled previous-frame pass and all context/`finish` logic stay unchanged. The exact current-PGO Ducks IP at RVA `0x14cbdd` had 148/60,215 samples in this cached scaled-neighbour loop (0.246%); that was a selection anchor, not a removable-CPU estimate. [The source patch](candidate.patch) is zero-context against integrated `865e1fd4` (`git apply --unidiff-zero`) and is **not integrated**.

The extended paired-versus-independent MV-reference differential covers all-full, one-full, and neither-full list states. Deliberately changing the entry predicate from `all` to `any` made the one-full case fail: paired GOLDEN `(0,0)` instead of independent `(-8,12)`. Restoring `all` passed. AArch64 and Crucible x86-64 each passed all 56 focused MV-reference tests; AArch64 locked core/encoder checks, formatting, and diff hygiene passed. All screen encodes reproduced the canonical RVP9 IVF hashes.

Crucible built the candidate using the unchanged [28-row production PGO recipe](pgo-manifest.txt), 60 frames per row and 31 raw profiles. [The build wrapper](build-pgo-mvref-crucible.sh) checked the exact changed source SHA-256 `ebf0d2ef70ea0fc4e7fa031dbb4c6d546419a00c15414633a9116ca89e4ba71d`. Candidate EXE SHA-256 was `81a9e2c3c6e188583ee16d0f17ff9942f4b007b95d355fbc26415f10591da119`; the code-equivalent current PGO control was `0fa1028b1310fee93b22731794dcb3fdc14e9b5e05dbe428f3e112c4b30cb395`. Every screen's `identity.csv` pins these binaries and the three 1080p inputs.

Four paired process-CPU screens used CPU6/CBR1260, one thread, no tiles, AQ0/noise0/lag0/KF60. Each warmed both arms and clips, alternated arm and clip order, and retained every pair. The fourth screen pinned both arms to logical CPU 0. The [runner](screen-mvref.ps1) is the final version with optional affinity; its initial version produced screens 01–03 with the same encode arguments and order. All **324 encodes** exited zero and matched their expected IVF SHA-256. `screen-NN/raw.csv` retains per-encode CPU, wall time, hash and exit status; `summary.csv` retains per-pair ratios. Lower candidate/control CPU is better.

| Screen | Pairs per clip | Ducks | Tree | Town | Equal-clip CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01, unpinned | 5 | 1.020178× | 0.984188× | 0.989897× | 0.997964× |
| 02, unpinned | 15 | 0.995914× | 0.996300× | 0.990380× | 0.994194× |
| 03, unpinned | 15 | 0.997170× | 0.966019× | 0.986031× | 0.982989× |
| 04, one-core affinity | 15 | 1.008868× | 1.001150× | 0.996613× | **1.002198×** |

Independent CSV parsing verified every row, pair, hash and geometric mean. The two longer unpinned screens favoured the candidate, but Ducks barely moved and the pinned-core screen reversed the equal-clip result while making Ducks 0.9% slower. The five-pair screen also made Ducks 2.0% slower. This does not meet the repeatable-gain retention bar. Do not integrate the guard on these data. No candidate/libvpx screen was run because a code candidate that loses its own control gate cannot improve the established direct-parity claim. Integrated production-PGO parity remains [1.099305× equal-clip process CPU](../2026-09-26-nstep-duplicate-variance-pgo/README.md), above the ≤1.05× goal.
