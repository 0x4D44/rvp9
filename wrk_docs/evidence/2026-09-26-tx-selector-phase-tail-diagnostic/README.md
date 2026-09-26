# Inter TX selector phase/tail diagnostic (26 September 2026)

The three-clip, 60-frame Crucible diagnostic was **bit-exact** against integrated `origin/main`, but its per-tile clocks disturbed the selector enough to make its phase times unsuitable as production performance estimates. The temporary instrumentation is committed only on local branch `task/20260926-TSK-HUM-diagnose-tx-selector-phase-tail-cost` (`6bf83716`); no shipped code from it is integrated. Its source patch is archived as `prototype.patch.gz`.

The diagnostic sampled one of every 64 inter TX-selector calls. It recorded EOB buckets, exact prune positions, and cases where a distortion bound would have allowed skipping coefficient-rate work. The probe ran on Crucible x86-64 with the pinned Ducks, In-to-Tree, and Old-Town clips, CPU6/CBR1260, one thread, 60 frames. All three IVF SHA-256 values matched the accepted baseline; `raw/manifest.csv` records inputs, binary, outputs, process CPU, and wall time. `raw/*.stderr.txt` contains all frame-level counters and existing emit/runtime profiles. This was a diagnostic run, **not** an RVP9/libvpx speed comparison.

| Clip | Sampled selector calls | TX4 tiles | TX4 zero-EOB | TX4 would-prune-before-rate | TX4 wins | Sample-extrapolated selector / existing emit selector |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ducks | 6,869 | 67,081 | 62,082 (92.5%) | 5,052 (7.5%) | 18 | 1,583 / 365 ms |
| In-to-Tree | 685 | 8,516 | 7,362 (86.4%) | 528 (6.2%) | 3 | 195 / 44 ms |
| Old-Town | 650 | 8,508 | 7,582 (89.1%) | 532 (6.3%) | 17 | 186 / 41 ms |

The ~4× selector discrepancy is far larger than the empty-clock calibration (22–24 ns per probe). The clock placements and changed code shape perturb this small, inline-heavy kernel; do not use the sampled phase percentages to rank transform, quant, distortion, or rate kernels. The rate-prune counter remains a useful feasibility check: only 6–8% of TX4 tiles could skip rate work at that point, and the timed skipped-rate total was small even before correcting for the probe's own clock cost. An early-rate-prune implementation is unlikely to recover a meaningful share of process CPU; it does not warrant a production prototype yet.

Validation: deliberate red/green of the new bucket test, AArch64 feature and non-feature compilation, feature Clippy and rustfmt; Crucible x86-64 focused tests and release feature build; three exact encode hashes. The latest accepted pinned-libvpx parity result remains [1.168613× overall process CPU](../2026-09-26-guarded-hadamard-pairwise/README.md) against the ≤1.05× target. The last accepted integrated-code branch is unchanged by this diagnostic. A next candidate needs a low-overhead profile or exact counter-based hypothesis elsewhere; do not infer a gain from these phase timings.
