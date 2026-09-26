# CPU6 full-tree subpixel rate-bound census

This output-neutral Crucible diagnostic measured whether the nonnegative MV
rate alone could reject a fractional probe before interpolation. On the pinned
60-frame Ducks CPU6 CBR1260 encode, **299,781 of 3,329,582 legal right,
down, and diagonal probes (9.0%)** met a safe rate bound. Almost all eligible
probes were diagonal: **297,344 (99.2%)**. That is 5.4% of the approximately
5.55 million fractional probes in this clip, so the opportunity is real but
not a broad search reduction.

The full tree evaluates the right and down costs to select a later diagonal.
Their rate-only score can replace an exact score only when it preserves both
the strict winner and the `<=` direction tie. The diagonal score is not used
for direction, so its safe bound is just `mv_cost >= besterr`.

| Shape | Round | Right eligible/legal | Down eligible/legal | Diagonal eligible/legal |
| --- | --- | ---: | ---: | ---: |
| 16×16 | half | 200/530,899 | 394/530,902 | 82,082/530,902 |
| 16×16 | quarter | 201/530,901 | 35/530,902 | 182,124/530,902 |
| Other | half | 309/24,029 | 799/24,029 | 14,759/24,029 |
| Other | quarter | 311/24,029 | 188/24,029 | 18,379/24,029 |

The diagnostic branch was local commit `fe02966b`; `diagnostic.patch.gz`
archives its exact diff. It starts from integrated `684185fe` and was built
on x86-64 Crucible with Rust 1.95.0. Instrumented EXE SHA-256 was
`3df3196912a4cbd9421e87312011ae9e4988375048bad8ef326ca5ff3ba85e99`.
The input SHA-256 was
`6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7`.
The instrumented output reproduced the canonical Ducks IVF SHA-256
`8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`.
The diagnostic asserts that every counted rate value lower-bounds the exact
score and that a qualifying bound preserves both the winner and cardinal
direction. The 19 focused subpixel tests passed on x86 and AArch64; the CLI
check passed on AArch64.

`run-bound-census.ps1` pins the command, paths and hashes; `counts.csv` is the
per-bucket result extracted from the diagnostic stderr. This is a work count, not a CPU
gain: any retained implementation must prove exact outputs and beat the
integrated PGO binary in repeated paired process-CPU screens.
