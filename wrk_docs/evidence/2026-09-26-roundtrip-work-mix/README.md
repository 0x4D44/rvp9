# Round-trip candidate work mix — 2026-09-26

This output-neutral, temporary census refines the [exact Tree/Town CPU profile](../2026-09-26-exact-subpel-pass-profile/README.md): it counts actual `reconstruct_via_round_trip` calls and transform-tile loop iterations by block shape, TX size, and metric options. Diagnostic source is local task commit `eb20de25` on `task/20260926-TSK-HUM-measure-roundtrip-hotpath-mix`, based on integrated `0b1da95f`; it is **not integrated**.

| Pinned 60-frame clip | Calls | Actual tiles | Main 32×32 / TX8 tiles | 64×64 / TX8 tiles |
| --- | ---: | ---: | ---: | ---: |
| Ducks | 1,575 | 25,044 | 24,992 (99.79%) | 0 |
| Tree | 261,124 | 5,479,240 | 3,730,256 (68.08%) | 1,744,576 (31.84%) |
| Town | 211,363 | 4,014,756 | 3,157,984 (78.66%) | 852,352 (21.23%) |

All reported rows had `reconstruct=false` and `collect_coef_dist=true`; no
reconstructing or error-only rows appeared on these CPU6/CBR inputs. The small
remaining tile shares are mainly 32×32 / TX16 and very rare 32×64 or 64×32
shapes. Exact per-row output is in the three `*-stderr.txt` files here. Tile
counts are actual loop iterations, not `calls × guessed block area`.

The diagnostic's focused counter test deliberately failed (0 observed calls
versus 2 expected), then passed. Core round-trip tests, formatting, CLI check,
and a local three-frame on/off byte-identical IVF smoke passed. Crucible built
the exact diagnostic source as release binary SHA-256
`01734f3ae5f838f7d628a9be23094b795430bf22268a649509781352446075d7`.
Each run verified the pinned input SHA-256, exited zero, and reproduced the
integrated RVP9 IVF hash: Ducks
`8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`,
Tree `5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41`,
Town `4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f`.
The task-owned raw IVFs, reports, and runner remain on Crucible under
`D:\worktrees\rvp9-2\20260926-TSK-HUM-measure-roundtrip-hotpath-mix\.scratch\roundtrip-mix`.

This is **work mix, not speed**. Enabled atomics perturb timing. The earlier
error-only metrics SIMD variant regressed paired CPU ([Unit1436](../../../wrk_journals/2026.09.26%20-%20JRN%20-%20Unit%201436%20error-only%20metrics%20AVX2%20screen.md)); this census does not reopen it. A stronger next question is how many of these full-metric calls belong to candidates that ultimately lose. The callee cannot know that: later candidates and intra fallback can displace an interim best. Exact winner attribution needs a candidate identity carried to final selection, and any pruning proposal needs an exact nonnegative lower bound plus paired CPU proof. The retained paired RVP9/libvpx process-CPU ratio remains **1.182520×**.
