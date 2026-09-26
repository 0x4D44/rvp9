# Sub-pixel horizontal-row reuse census — 2026-09-26

This is an opportunity census, **not** a speed measurement. A temporary,
output-neutral diagnostic counts successful two-dimensional sub-pixel variance
calls whose horizontal work could reuse a preceding call in the same search.
The first diagnostic source is local task commit `bca02daa` on
`task/20260926-TSK-HUM-measure-subpel-horizontal-reuse-ceiling`, based on
integrated trunk `882e50f4`. Follow-up local commit `87fa4242` adds the
immediately preceding eligible-2D-probe count. Neither is integrated.

| 60-frame clip | Searches | 2D calls | Same-key calls with a different prior vertical phase | Share | Border 2D calls | Border repeats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ducks | 554,931 | 1,687,619 | 308,333 | 18.27% | 10,812 | 2,147 |
| Tree | 124,474 | 533,430 | 182,231 | 34.16% | 2,204 | 803 |
| Town | 120,450 | 501,106 | 163,582 | 32.64% | 2,784 | 814 |

A second exact-output run measured what a **one-entry** horizontal buffer could
capture. Its preceding probe means the preceding *eligible 2D* call in that
search; intervening non-2D calls do not overwrite the prospective buffer.

| Clip | Immediate different-phase repeats | Share of all 2D calls | Share of any-prior repeats | Immediate border repeats |
| --- | ---: | ---: | ---: | ---: |
| Ducks | 145,447 | 8.62% | 47.17% | 955 |
| Tree | 70,675 | 13.25% | 38.78% | 373 |
| Town | 64,771 | 12.93% | 39.60% | 265 |

The key includes reference-plane base, integer reference origin, horizontal
phase, block width, and block height. It is scoped to one search; the temporary
edge buffer's pointer cannot serve as identity. A repeat is counted only after
a successful variance call with nonzero horizontal and vertical phase, when
the same key has already appeared with a *different* vertical phase. Border
replication is counted separately. The counter does not estimate horizontal
filter cost, cache-management cost, or whole-encode savings.

Crucible ran the pinned 60-frame `--rt --cpu-used=6` three-clip command, with
`--codec=vp9 --passes=1 --end-usage=cbr --target-bitrate=1260 --fps=30/1
--threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default
--aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf` and
`RVP9_DIAG_SUBPEL_REUSE=1`. The matching release binary SHA-256 for all runs
was `a28fa4bff0e5fe8a0c63aeb27573ef3fb43d170778604d640ac895795bcfd289`.
The matching follow-up binary SHA-256 was
`944101d548b748601730d42e778d80ae6bb5c4806803e328649d2914a7352ab7`.
Each input hash was verified before encoding. Each output reproduced the pinned
integrated RVP9 hash:

| Clip | Input SHA-256 | Output IVF SHA-256 |
| --- | --- | --- |
| Ducks | `6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7` | `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089` |
| Tree | `ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791` | `5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41` |
| Town | `b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909` | `4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f` |

Raw IVF and stderr are in Crucible's task-owned
`D:\worktrees\rvp9-2\20260926-TSK-HUM-measure-subpel-horizontal-reuse-ceiling\.scratch\census\{ducks,tree,town}`.
The follow-up is under the adjacent `.scratch\census-immediate` directory.
The launch scripts are in that worktree's `.scratch` directory and in the
matching local task worktree. The diagnostic's pure counter test deliberately
failed once (`calls_2d`: expected 13, actual 12), then passed; all 20
`sub_pixel` tests, formatting, CLI check, and a two-frame output-on/off A/B
were green on macOS. Root independently reran the focused counter test. The
follow-up test also deliberately failed once (immediate repeats: expected 4,
actual 3), then passed. Root reran its focused test. All six Crucible 60-frame
output hashes reproduced the pinned values above, checking output neutrality.

The frequency is high enough to discuss a bounded search-local horizontal-row
cache, especially for Tree and Town. A one-entry buffer captures only about
40% of all-prior repeated calls on those clips; more entries cost lookup and
storage, so the right design is not settled by counts alone. No cache should
land without exact per-candidate costs, winners, edge replication, AArch64
fallback, and paired process-CPU gain. The latest paired RVP9/libvpx
process-CPU ratio remains **1.182520×**, not a new measurement.
