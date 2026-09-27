# Nonzero inter-evaluator repetition census

The exact integrated production-PGO profile gives `evaluate_candidate_for_filter_with_pred` 1,594/18,127 Ducks, 1,346/8,195 Tree, and 1,124/8,629 Town direct samples. That symbol share is not a saving estimate. A diagnostic-only branch `4911cf47`, based on integrated `5d19f4bb`, counted *actual* `evaluate_candidate_with_slots` calls in the final inter picker. Within each picker call, it compared each evaluated mode with earlier evaluated modes for the same reference and nonzero MV. A strict match also required the same ordered interpolation-filter set. Skipped and vetoed modes were never entered as prior evaluations. No scoring, deduplication, or bitstream behavior changed.

The opt-in `RVP9_DIAG_DUP_INTER_EVAL=1` counter ran on native x86-64 Crucible from exact source SHA-256 `f174572e51690712ce45eb8addf4007b1e263aa75c57c94d06dd8577d9bd3fb8` (`inter.rs`) and `7b9a1e09cbc6409c3808b7650d1ba6ea12a0a26a1ba493cfd6ff57d2807858e6` (`rvp9enc.rs`). The release diagnostic EXE SHA-256 was `eb0e08a87c3af771d8e3062896dffcd25cf5fe7d532c9b48bfeff63a6fb0f80c`. Each encode used the pinned 60-frame 1080p input and the normal CPU6/CBR1260, single-thread, no-AQ command recorded in the [previous operation census](../2026-09-27-large-model-rd-skip-loop-census/README.md). The three encodes exited zero and matched canonical integrated IVF SHA-256 values: Ducks `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`, Tree `5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41`, Town `4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f`. The diagnostic build is not a valid CPU timing arm.

| 60-frame clip | Evaluator calls | Same ref + nonzero MV | Same ref + nonzero MV + filter set | Strict share |
| --- | ---: | ---: | ---: | ---: |
| Ducks | 2,391,685 | 13,137 | 9,901 | 0.4140% |
| Tree | 536,805 | 3,745 | 2,034 | 0.3789% |
| Town | 454,630 | 6,799 | 3,657 | 0.8044% |

Independent parsing of the [Ducks](ducks.stderr.txt), [Tree](tree.stderr.txt), and [Town](town.stderr.txt) raw maps gives the table. Every strict match was an earlier `NEARMV` followed by `NEWMV`, overwhelmingly on 16×16 blocks. A mode-specific syntax/MV rate and filter-policy side effects remain even for a strict match, so this is an upper bound on potentially shared prediction/transform work, **not** a proven reusable whole result or a CPU percentage. The counter-reconciliation unit test was deliberately red on a wrong strict expectation, then passed on AArch64 and native x86; the native release encoder build passed.

Decision: 0.4–0.8% of evaluator calls is too little to justify an ownership-heavy reuse path toward the remaining 0.4 percentage-point overall parity gap without operation-time evidence showing these particular calls are unusually costly. Do not integrate the diagnostic code or infer a PGO gain from this count. Integrated direct parity remains 1.054068× pinned libvpx equal-clip process CPU.
