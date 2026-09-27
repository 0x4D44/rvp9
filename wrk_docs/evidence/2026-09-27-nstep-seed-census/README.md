# NSTEP repeated-seed census

The [fresh integrated-commit PGO profile](../2026-09-27-fresh-integrated-pgo-ip/README.md) assigns substantial Ducks samples to NSTEP motion search. To test one bounded hypothesis, a diagnostic-only native x86 build from commit `75c23fe2` counted entries to `nstep_search_sad` and `diamond_search_sad_from_step`. The [compressed one-file instrumentation diff](source.diff.gz), [counter definitions](event-definitions.txt), [runner](run-nstep-diagnostic.ps1), [binary/source identities](identity.csv), and [raw stderr](ducks.stderr.txt) are retained. The decompressed archived diff differs from the remote original only by CRLF-to-LF normalization (decompressed SHA-256 `5107d62ce977b5a26409e73a34713c0067f5e27e01acbdd0ec2ad9be0dd54eee`; the identity records the original remote hash). The diagnostic code and binary are **not integrated** and provide no valid CPU timing.

| 60-frame CPU6 clip | NSTEP calls | Diamond entries | Repeated starting-SAD work | `num00`-skipped restarts | Repeats/call |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ducks | 629,530 | 1,903,948 | **1,274,418** | 122,575 | 2.024396 |
| Tree | 131,265 | 250,329 | **119,064** | 9,132 | 0.907051 |
| Town | 125,089 | 169,907 | **44,818** | 2,500 | 0.358289 |

Each diamond entry computes the same clamped starting MV and its augmented SAD from unchanged parameters; only `start_step` differs between restarts. For all three clips, `NSTEP calls + repeated restarts = diamond entries`. `num00` skips happen before a diamond entry and do not perform that work. The [census CSV](census.csv) records all counters and the input/output SHA-256 values. Every diagnostic encode exited zero and matched the integrated encoder's canonical IVF hash.

The 1.27 million repeated Ducks evaluations warrant an exact-output prepared-seed prototype, not a predicted speed gain. The proposed scope is one prepared source/dispatch/seed context per NSTEP call, with each restart's search state and strict comparison order unchanged. Keep the existing public diamond wrapper's behavior. Validate the prototype with an old-path differential, deliberate red test, native x86/AArch64 checks, canonical IVFs, and paired production-PGO CPU before integration.
