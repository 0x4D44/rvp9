# Matched-PGO filter-evaluator stage census

The exact-output Ducks path is overwhelmingly **model-only**, so a round-trip optimization cannot be the primary Ducks parity lever. Of 2,391,685 logical filter-evaluator candidates, 2,384,520 (99.700%) took the model-only branch and only 1,443 (0.060%) called the round-trip. Tree and Town differ: 38.655% and 35.139% of their candidates called the round-trip. This diagnostic identifies where work occurs, not a removable-CPU estimate. The latest uninstrumented integrated production-PGO direct result remains **1.057638×** pinned libvpx equal-clip process CPU, above the ≤1.05× target.

The diagnostic branch `090607ac` is **not** on `origin/main`. It adds opt-in `RVP9_DIAG_FILTER_STAGE_CYCLES=1` counters and `LFENCE;RDTSC` call-site timers to `crates/rvp9-core/src/mode_decision/inter.rs`, plus a CLI report hook. It begins at the logical `evaluate_candidate_with_slots` boundary: the two-filter selector builds and models *both* luma predictions before the winning filter reaches the evaluator, so timing only the evaluator would omit work. The [decoded patch](diagnostic.patch.b64) has SHA-256 `29c109f1e6aefbf787f5924a1feb1f0913b2baac62048d0fba5866bebd8207ab` and applies to integrated `0d1949cfe53ffe5493aa1bc2e30874c8a7b40043` (shipped encoder source unchanged from `75c23fe2`). No scoring or encode decision changed.

Crucible built the patched tree with the normal 28-row, 60-frame production-PGO recipe, using the canonical corpus. The fresh profile SHA-256 was `af767d63e71fa1efbf48c5fef26e23f5121cedf5c4134b3cd8e5c6a3a414d198`; the diagnostic EXE SHA-256 was `1dd5bd3bf34eb76264af63adfb55e9d464fabf8ede7b15bbf045e1973b43fb7b`. The build emitted no control-flow hash mismatch warnings. All nine CPU6/CBR1260, 60-frame, single-thread encodes ran under a short Crucible quiet lease with zero pre-existing heavy builds. Every IVF matched its canonical integrated hash: Ducks `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`, Tree `5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41`, Town `4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f`.

| Clip, 60 frames | Logical candidates | Two-filter selector trials | Luma prediction/model calls | Model-only | Round-trip | Early term |
|---|---:|---:|---:|---:|---:|---:|
| Ducks | 2,391,685 | 178,170 | 2,480,770 | 2,384,520 | 1,443 | 5,722 |
| Tree | 536,805 | 101,720 | 587,665 | 303,760 | 207,502 | 25,543 |
| Town | 454,630 | 94,862 | 502,061 | 268,645 | 159,753 | 26,232 |

Every run produced identical counts per clip. Branch counts sum to logical candidates, evaluator calls equal logical candidates, and luma calls equal logical candidates plus half the selector-trial count. A selector trial is one tested filter, hence two per two-filter candidate.

| Clip | Luma prediction cycles | Luma model cycles | Round-trip cycles |
|---|---:|---:|---:|
| Ducks | 306.5M | 390.8M | 2.6M |
| Tree | 182.5M | 247.4M | 449.8M |
| Town | 147.1M | 204.2M | 314.6M |

These are arithmetic means of three same-image runs; the [raw reports](three-runs-stderr.tar.gz) retain every block shape, call count and exact cycle total. The three Ducks luma-model totals were 390,270,838 / 389,445,195 / 392,771,589 cycles. Luma prediction and model timers are separate call-site spans; chroma skip-probe time is inclusive and must not be added to possible child work. `LFENCE;RDTSC`, timer call overhead, the diagnostic atomics outside spans, PGO trained with the gate off, and different branch mixes all limit comparisons. These cycles are **not** whole-encoder process CPU and are not a speed gain. There is no code candidate to integrate from this census alone.

The focused counter-reconciliation test passed on AArch64 and native x86. Omitting the selector-trial increment made the test fail (`0` versus expected `2`); restoring it passed. Formatting, diff checks, CLI build, and AArch64 gate-off/on tiny-IVF comparison passed. The next discriminator is to split the hot Ducks 16×16 luma-model call into variance reduction versus DC/AC model arithmetic before selecting a narrow exact-output kernel change. Prior dispatch-handle hoisting was CPU-neutral and is not reopened by this broad stage total.

## Follow-up: Ducks 16×16 inner model

The diagnostic-only branch now also has commit `b05099eb`, adding opt-in `RVP9_DIAG_MODEL16_INNER=1` counters and deterministic 1/64 sampled cycle spans in `model_rd_for_sb_y`. Its [incremental patch](model16-inner.patch.b64) decodes to SHA-256 `237b4b3c5256628a92d22adce473edef49b21c751ad0eb4601fdcd24cf69fa3c` and applies after `090607ac`; it is **not** integrated. The focused counter test passed on AArch64 and native x86, including a deliberately failing omitted-counter check. Formatting, diff checks, CLI build, and tiny gate-off/on IVF equality passed. A fresh matched 28-row production-PGO build on Crucible emitted no control-flow mismatch warning; the diagnostic EXE SHA-256 was `2f87f98409646a4d4f6793c78a1a9db69fe0bb2d432e120a5940775979f1160b`.

Three Ducks CPU6/CBR1260, 60-frame encodes with **both** diagnostic gates enabled each matched the canonical IVF SHA-256 `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`. Every run counted 2,858,463 calls to the 16×16 model: 1,106,799 `AcDc` skips, 111,739 `AcOnly` skips, and 1,639,925 `None`; 1,124,179 skipped DC work. The model attempted 627,485 DC and 1,751,664 AC Laplacian calculations, with zero zero-variance cases in each. The 16×16 model call count exceeds the 2,295,392 calls seen inside the filter-evaluator's luma-model timer because the model also has other call sites; the two counts must not be equated.

| Sampled span | Samples | Cycles, runs 1 / 2 / 3 | Mean cycles/sample |
|---|---:|---:|---:|
| Dispatch lookup | 44,663 | 2,164,663 / 2,219,531 / 2,165,996 | 48.9 |
| Variance reduction | 44,663 | 4,268,438 / 4,319,694 / 4,266,202 | 95.9 |
| Variance finish | 44,663 | 3,363,331 / 3,396,441 / 3,363,675 | 75.6 |
| DC Laplacian | 9,835 | 559,129 / 567,858 / 557,452 | 57.1 |
| AC Laplacian | 27,559 | 1,530,284 / 1,538,970 / 1,515,062 | 55.4 |

Variance reduction and finish are the largest measured inner spans, but the existing 16×16 reducer is already AVX2 and a previous dispatch-hoist experiment was CPU-neutral. `LFENCE;RDTSC` overhead matters especially for ~50-cycle spans; periodic sampling may bias mode mix. These numbers do **not** estimate removable CPU or revise the uninstrumented 1.057638× parity result. The next useful step is a narrow, exact-output variance-path hypothesis with paired production-PGO CPU screening; absent one, this evidence argues against another broad selector or dispatch rewrite.
