# Matching NEWMV seed SAD reuse: exact, not repeatably faster

Decision: reject the production source change. In `run_newmv_search`, the integer-projection seed sometimes rounds to the same full-pel position as the already-scored nearest/near predictor. The prototype reuses that predictor's raw SAD only in this matching case. The strict `<` seed choice, nonmatching SAD probe, and no-predictor fallback stay unchanged. This is narrower than earlier attempts to carry seed SAD into the diamond search.

The source is local unpushed commit `cd012cfc` on `task/20260927-TSK-HUM-prototype-exact-newmv-seed-sad-reuse`; `prototype.patch.b64` preserves its `git format-patch` output (decoded SHA-256 `820fc1e349f70eb3f8d4f35ac04b4e3fc853c2f1969e4a048d81faaa6175dbab`). A new matching/nonmatching/sentinel test went red when the equality guard was deliberately inverted and green after restoration. AArch64 passed the three focused NEWMV tests, `cargo check --locked -p rvp9-encoder`, formatting, and diff hygiene. Crucible x86-64 passed the same three focused tests. The production-PGO candidate used the unchanged 28-row, 60-frame training recipe. `pgo-manifest.txt` records that build; candidate EXE SHA-256 was `1e3667178bcc3845cd6d5db41dcbf5b47fdd5795690a75fe33bbaa7bc175c3e6`. The integrated-code control EXE SHA-256 was `3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee`.

Six alternating-order Crucible screens used pinned 60-frame 1080p Ducks, In-to-Tree, and Old-Town inputs at CPU6/CBR1260, one thread/no tiles, AQ0/noise0/lag0/KF60. Screen 01 had five pairs per clip; screens 02–06 each had 15. Screens 05–06 pinned both arms to logical CPU 0. All **516 encodes** exited zero and reproduced their arm's established IVF hash; every raw row was checked against its expected hash. The CSVs retain all observations; **no pair was excluded**. Process CPU, not wall time, is the retention metric. The wall-time stalls and occasional large process-CPU pair swings make a small gain hard to resolve.

| Candidate/control process CPU | Ducks | Tree | Town | Equal-clip overall |
| --- | ---: | ---: | ---: | ---: |
| Screen 01, 5 pairs | 0.978065× | 0.981615× | 0.964933× | **0.974844×** |
| Screen 02, 15 pairs | 1.003403× | 0.989001× | 0.996266× | **0.996206×** |
| Screen 03, 15 pairs | 0.971831× | 0.988229× | 1.000674× | **0.986841×** |
| Screen 04, 15 pairs | 1.005844× | 1.023154× | 1.029715× | **1.019521×** |
| Screen 05, 15 pairs, CPU 0 | 0.957077× | 0.972374× | 0.990598× | **0.973253×** |
| Screen 06, 15 pairs, CPU 0 | 1.012553× | 1.008497× | 1.013415× | **1.011486×** |

The two pinned-core screens disagree by nearly four percentage points overall, and an unpinned long screen reverses the apparent win. Screen 03 Ducks has a single 0.718× pair where control CPU spiked; screen 02 has a 1.146× Ducks pair in the opposite direction. Both remain in the summaries. There is no reproducible production gain to integrate, and no new candidate/libvpx parity claim. The last uninstrumented direct integrated-code comparison remains [1.099305× equal-clip process CPU](../2026-09-26-nstep-duplicate-variance-pgo/README.md).

`build-pgo-crucible.sh`, `screen-seed-sad-unpinned.ps1`, and `screen-seed-sad-affinity.ps1` preserve the commands. Each screen's `*-identity.csv`, `*-raw.csv`, and `*-summary.csv` preserves binary/input identities, individual process CPU and wall times, output hashes, and paired ratios. The CSVs are normalized from CRLF to LF for Git; values are unchanged. The local and Crucible prototype worktrees remain unintegrated; do not mistake the source commit for shipped code.
