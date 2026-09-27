# In-place large-model RD early exit: exact but slower

The earlier wrapper-based TX-cell early exit was exact but 2.5–2.8% slower in paired production-PGO CPU screens. This follow-up changed only the existing `model_rd_for_sb_y_large` loop: after both monotone AC/DC skip tests fail, it breaks the outer loop. The public function and its call structure stayed intact. Local code commit `b46fcd8e` and native Crucible code commit `be8e88836` hold the same source file (SHA-256 `e38ea765df3f126321858d21440eac0e752c3ec817faeed23254a662bb43138c`); neither code branch is integrated.

A test-only full-loop oracle compared all six result fields over 32×32/64×64 blocks, Tx8/Tx16/Tx32, and flat/both-fail/one-fail patterns. It also checked that the both-fail case saved visits. Removing the break deliberately failed the visit assertion; restoring it passed. Focused AArch64 and native x86 tests passed (24 and 23 filter-matched tests respectively), as did the local format and crate checks.

Crucible built the unchanged 28-row, 60-frame production-PGO recipe from integrated `5fbd7e72` plus that one file. Candidate EXE SHA-256: `d4c97e0b73227958e623519d4b676fc18c7c3362ed1a149a61795d6bb308e2b4`. Control was the exact integrated EXE SHA-256 `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`. The [paired runner](../2026-09-27-early-dual-tx-chroma-proof/screen-early.ps1) was adapted by [this wrapper](screen-in-place-scratch.ps1), copied to the task worktree's `.scratch` directory. It warmed each arm, alternated arm and clip order, retained every pair, and required canonical IVF hashes. A Crucible quiet lease was granted; two pre-existing heavy-build processes were reported, so the evidence is process CPU, not a claim of an idle wall clock.

| Candidate/control process CPU | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| 5 pairs/clip | 1.052688 | 1.018692 | 1.023306 | **1.031453** |
| 15 pairs/clip | 1.039400 | 1.018433 | 1.011123 | **1.022915** |

All 132 screen encodes, including warmups, matched the canonical IVF hashes. The candidate is slower in both screens and all three clips; the narrow in-place edit does not rescue the wrapper prototype. **Do not integrate either code branch.** The latest direct integrated RVP9/libvpx ratio remains 1.054068× equal-clip process CPU; this rejected candidate did not need a direct libvpx screen.

The paired [5-pair raw](screen-01-raw.csv), [summary](screen-01-summary.csv), and [identity](screen-01-identity.csv) CSVs and the matching [15-pair raw](screen-02-raw.csv), [summary](screen-02-summary.csv), and [identity](screen-02-identity.csv) CSVs retain the measurements. Native build and raw output artifacts remain in the Crucible task worktree `D:\worktrees\rvp9-2\20260927-TSK-HUM-measure-in-place-large-model-rd-early-ex` under `.scratch`.
