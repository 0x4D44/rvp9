# NSTEP SAD4D batch-reuse census (26 September 2026)

An exact, output-neutral Crucible x86-64 diagnostic counted four-reference SAD batches in CPU6 NSTEP search and repeated `(step, center row, center col)` keys **within the same search**. A per-search cache could reuse only such repeats without changing SAD inputs. The source is committed on local diagnostic branch `task/20260926-TSK-HUM-census-repeated-nstep-sad4d-batches` (`64946735`), not integrated; `prototype.patch.gz` archives it.

| Pinned 60-frame clip | Eligible in-plane SAD4D batches | Repeated exact keys | Reuse rate |
| --- | ---: | ---: | ---: |
| Ducks | 5,814,834 | 320,592 | 5.513% |
| In-to-Tree | 868,181 | 29,956 | 3.450% |
| Old-Town | 639,648 | 10,798 | 1.688% |

The latest integrated Ducks IP profile assigns 120/2,451 direct samples to SAD4D and 74/2,451 to its adapter. Even a zero-overhead cache that removed the whole SAD4D+adapter cost of every repeated batch would save only about `5.513% × (120+74)/2451 = 0.44%` of Ducks process CPU. Real lookup/storage overhead lowers that ceiling. This misses the >20% repeat-rate precondition for a plausible ≥1% whole-encode gain, so **do not build the per-NSTEP cache**. These are opportunity counts, not performance measurements; they do not change the latest [1.168613× pinned-libvpx parity result](../2026-09-26-guarded-hadamard-pairwise/README.md).

The diagnostic used `RVP9_DIAG_NSTEP_BATCH_REUSE=1` on the pinned Ducks, Tree, and Town inputs at CPU6/CBR1260, one thread, 60 frames. Its x86 release executable SHA-256 was `722873317d3b83dee3b144afa20b6096ce5ddfc9177ad54678bd4cb68d41a836`. All three encodes exited zero and matched the accepted IVF SHA-256 values; `raw/manifest.csv` records binary, input, output, CPU, and wall identities, and `raw/*.stderr.txt` records the counters. The diagnostic itself adds hash-table work and is not suitable for speed comparison.

Validation: a wrong-key mutation that omitted `step` made the new counter test fail (`(5, 2)` versus `(5, 1)`), then restoration passed. AArch64 full-pixel tests 22/22, CLI check, format and diff checks passed; Crucible x86 focused tracker tests 2/2, release build, and three exact-output encodes passed. The next production experiment should target another measured kernel rather than this cache.
