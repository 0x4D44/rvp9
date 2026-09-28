# TX4 entry-bound oracle: rate floor and whole-leaf limit

This is a **count-only diagnostic**, not a faster encoder. The production-PGO equal-clip process-CPU result remains [1.057638× pinned libvpx](../2026-09-27-integrated-pgo-direct-repeat/README.md) against the ≤1.05× target. The code used here is committed only on the unintegrated diagnostic branch `task/20260928-TSK-HUM-count-pre-tile-tx-selector-lower-bound` at `0d3abed2`; no production shortcut landed.

For every CPU6/CBR1260 Ducks TX4 selector visit, the diagnostic completes all 16 TX4 tiles after simulating the existing prune, then asks whether a hypothetical bound at candidate entry would have beaten the incumbent. Tier A includes the actual distortion of EOB0 tiles and TX4 signaling, treating every other contribution as zero. The floor tier also charges each actual EOB0 tile the *minimum* EOB0 coefficient cost over the three legal entropy contexts in the frame's existing cost table. Tier B uses the actual EOB0 coefficient context and cost. Tier C additionally uses the actual rates of nonzero tiles but still treats their distortion as zero. Thus A ≤ floor ≤ B ≤ C in this diagnostic; all tiers still have oracle knowledge of EOB0 outcomes and EOB0 transform-domain distortion. None is a proven cheap pre-transform bound.

| 59 inter frames of canonical 60-frame Ducks | Visits | Baseline TX4 tiles hypothetically avoided |
| --- | ---: | ---: |
| All TX4 candidates | 141,013 | 1,597,307 tiles actually processed after baseline prune |
| Tier A hits | 59,297 (42.05%) | 594,338 (37.21%) |
| Minimum legal EOB0-rate floor hits | 80,652 (57.19%) | 843,450 (52.80%) |
| Floor hits where **all 16 tiles** are EOB0 | 23,833 (16.90%) | 263,222 (16.48%) |
| Tier B hits | 84,899 (60.21%) | 889,687 (55.70%) |
| Tier C hits | 127,379 (90.33%) | 1,403,353 (87.86%) |

There were 23,867 all-zero candidate visits in total, 1,262 true TX4 winners, 2,256,208 fully walked tiles, and 1,952,720 EOB0 tiles. No tier hit a true winner. The floor tier's 57.19% hit rate falls short of the consultant's suggested 60% screening threshold even with oracle distortion. A certificate covering only *whole-leaf* EOB0 candidates is narrower still: its perfect-knowledge ceiling is 16.48% of baseline TX4 tile evaluations, before paying for any certificate or entry-bound calculation. The 52.80% floor-tier ceiling depends on handling mixed EOB0/nonzero leaves. These counts do **not** predict an overall CPU reduction.

Native x86 source SHA-256: `f2a3e77f590f6143fb51717533d0db35c80f67c2720606f06010554deb63253a`. Release EXE SHA-256: `f494eeabde87ea8fc4179c5ddbb1b9348cf33a161aed176c0adc7fdc34d00b36`. Canonical input SHA-256: `6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7`. Gate-on and gate-off encodes both produced the expected IVF SHA-256 `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`; gate-on emitted 59 reports and gate-off emitted none. Raw reports are on Crucible under `D:\worktrees\rvp9-2\20260928-TSK-HUM-count-pre-tile-tx-bound-native-x86\.scratch\tx4-oracle-allzero-ducks-20260928\ducks.stderr.txt`. The earlier floor-only run is under the sibling `tx4-oracle-floor-ducks-20260928` directory.

The focused counter test was made deliberately red and restored to green on AArch64; native x86 passed the same focused test and a release build. Formatting and diff checks passed. The diagnostic's gate-on output was exact, but its extra full walk makes its timing invalid for performance comparison.

**Decision:** do not add a whole-leaf zero-EOB TX4 skip on these counts alone. A production prototype needs an integer-safe lower bound on transform distortion from pre-transform residuals, an affordable EOB0 certificate, and a way to cover enough mixed leaves; then it must pass exact differentials and paired production-PGO CPU screens. Without that proof and cost model, choose another profiled hot candidate rather than claiming this oracle ceiling as a gain.
