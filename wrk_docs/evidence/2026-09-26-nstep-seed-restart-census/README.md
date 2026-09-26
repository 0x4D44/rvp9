# NSTEP repeated-seed census on pinned production clips

The integrated PGO Ducks profile attributed 5,192/60,215 direct encoder samples to `nstep_search_sad` and 1,750 to `diamond_search_sad_from_step`. Astra suggested preparing the identical initial seed once across NSTEP diamond restarts. Before changing search code, a diagnostic-only release build counted actual calls and starts on the three pinned 60-frame CPU6/CBR1260 clips, one thread/no tiles. The source was integrated trunk `df37f223` plus [this zero-context instrumentation patch](instrumentation.patch), diagnostic commit `58148733`; reproduce it with `git apply --unidiff-zero`. The x86-64 EXE SHA-256 was `a3f2a7255522fc7ad3421c2c387fd2b144bc8ef0853230680a4fea9889e35d85`.

| Clip | NSTEP calls | Actual diamond starts | Extra starts | Extra starts/call |
| --- | ---: | ---: | ---: | ---: |
| Ducks | 629,530 | 1,903,948 | **1,274,418** | 2.024396 |
| Tree | 131,265 | 250,329 | **119,064** | 0.907051 |
| Town | 125,089 | 169,907 | **44,818** | 0.358289 |

The [runner](run-nstep-census.ps1) checked the exact EXE and input hashes, captured the counters from the CLI after each encode, and checked every IVF against its established hash. Two independent [runs](run01/census.csv) and [repeats](run02/census.csv) returned exactly the same integer counts and IVF hashes. The 21 focused AArch64 full-pixel tests, AArch64 CLI compile check, and `git diff --check` also passed for the diagnostic branch.

Each NSTEP call begins one diamond search. Every extra start re-enters `diamond_search_sad_from_step` with the same parameters and a different `start_step`; source-block validation, SAD dispatch, seed clamp, and initial SAD *attempt* run again. `extra starts` is thus a direct count of repeated setup and initial SAD attempts, **not** a count of successful SAD-kernel calls or a CPU-time estimate: an out-of-plane seed can return `None` before invoking the kernel. The code permits up to four extra starts per normal 11-step CPU6 call, but the actual counts above are what justify a bounded prototype. The diagnostic counters do not affect search decisions or encoded bytes. They do add atomic work, so their process CPU values must not be used as a parity measurement.

Next prototype: prepare the SAD functions, source view, clamped initial MV, rate-reference MV and initial augmented SAD once per NSTEP call, then pass the immutable seed into each diamond restart. Keep restart order, `last_site`, `num00`, strict ties, iteration counts, and final refinement independent. An old-algorithm differential must reach multiple starts and deliberately fail on a bad reused seed. Only exact outputs plus paired production PGO CPU screens can justify integration.
