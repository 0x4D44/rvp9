# NSTEP prior-loser variance-score census

The exact-image [profile](../2026-09-27-fresh-integrated-pgo-ip/README.md) gives Ducks substantial full-pixel search samples. The current NSTEP [same-best guard](../../../crates/rvp9-core/src/mv_search/full_pixel.rs) skips a candidate whose MV equals the current best, but a previously variance-scored losing MV can recur after another diamond start. A diagnostic-only two-file x86 build counted every actual variance-score call and tracked its coordinates within each NSTEP invocation. It did **not** skip, cache, or reorder any score; its instrumentation makes CPU time invalid. The core library exposes a counter snapshot, and the CLI prints it after encoding. The source is uncommitted in local worktree `task/20260927-TSK-HUM-diagnose-repeated-nstep-variance-coordin`; the [identity](identity.csv) records source and binary SHA-256 values.

| 60-frame CPU6 clip | NSTEP calls | Variance cost calls | Existing same-best skips | Prior-loser repeat calls | Repeat share of cost calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ducks | 629,530 | 1,858,678 | 526,405 | 91,956 | 4.947% |
| Tree | 131,265 | 245,818 | 72,601 | 7,317 | 2.977% |
| Town | 125,089 | 165,000 | 48,450 | 2,719 | 1.648% |

The independent [first](run-1.csv) and [second](run-2.csv) three-clip runs produced byte-identical CSVs. All six diagnostic encodes exited zero and matched their integrated encoder's canonical IVF hashes. The input and output hashes, exact counts, and initial/candidate/refinement breakdowns are in those CSVs. The changed core and CLI files passed focused AArch64 checks, formatting, and all 21 native x86 full-pixel tests; the native x86 release build completed.

**Decision:** stop before a cache prototype. Even a free perfect cache can remove at most these repeated variance calls: under 5% of variance scores on the slowest clip, far less on the others. A cache must also probe every scored coordinate and keep per-search state across 629,530 Ducks NSTEP calls. This is an operation-level upper bound, not a measured whole-encoder CPU bound, but the small hit rate makes a repeatable gain toward the remaining parity gap implausible. Do not infer a whole-encode saving from NSTEP's broad profile share. No diagnostic code is integrated.
