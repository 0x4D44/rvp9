# TX4 2×2 staging prototype — Crucible screen A

The bounded TX4 prototype was byte-exact but lost performance, so it was not
integrated. It staged the transforms, quantization, and shifted block errors
of each 2×2 TX4 region while pricing coefficients, updating entropy contexts,
and pruning in full-leaf raster order. The production code still ran each
tile's scalar transform/quantizer; the batch saved no core-kernel operations.

The prototype is committed only on task branch
`task/20260925-TSK-HUM-prototype-exact-tx4-batching-in-inter-lu` at
`44a406ce`. It was built on Crucible from the exact transferred `lib.rs`
SHA-256 `beae5d05a910cff6c1a2573efa3612759b3eaf831b8897712455dfb583d73a6e`
over base `4718a4aa`. Candidate EXE SHA-256 was
`954b1f0d9e6ee15e218ea7d0081f2dcd372570bf002e044f036d78207f7a6e3c`;
the pinned integrated control EXE SHA-256 was
`c371d472690baba94dc1374b9ce4b65dddd3f3ac2381feca362efa25e55f0d98`.
`identity.csv` records the control PDB and three Y4M hashes. The exact
runner is `tx4-run-ab.ps1`, SHA-256
`6a2d032cfc842d6d5833d3612512ee6ac0c880a42fad3e2d7c623c70e3b616b6`.

The CPU6 single-thread protocol used CBR 1260 kb/s, FPS 30/1, tile 0,
AQ/noise/lag 0, 60 frames, and KF interval 60. There was one warmup per
arm and clip, then 15 serial current/TX4 pairs for each of Ducks, In-to-tree,
and Old-town-cross. Arm order alternated; even pairs reversed clip order.
Each encode had a 90-second limit and the full screen a 1200-second limit.
Process CPU was primary, wall time and sampled host load secondary. All 45
pairs were retained. I independently recomputed the geometric means from
`raw.csv` after removing its UTF-8 BOM for parsing.

| Clip | TX4/current process CPU | TX4/current wall |
|---|---:|---:|
| Ducks | 1.067092× | 1.011566× |
| In-to-tree | 1.028787× | 1.028900× |
| Old-town-cross | 0.992453× | 0.996856× |
| Equal-clip corpus | **1.028993×** | **1.012356×** |

All 96 encodes exited zero without timeout. Each arm and clip produced one
stable IVF hash across its 16 outputs, and both arms matched the prior
expected hash. Six separate smoke encodes also matched; their raw rows,
identity, load samples, and manifest are in `smoke/`. The smoke `raw.csv`
SHA-256 was `b713032f935703b03e383fef41d40952229c33e97e52d7f42ea50223aedb67ae`
before line-ending normalization and is
`db187961841dbdc6fc2b524cfca6087a25f21db749847551e6963de4429b146e`
checked in. Short Tree/Town
encodes often lacked a during-run load sample; Ducks had wall-time stalls.
No rows were removed or adjusted. The +2.9% corpus process-CPU result and
the clear Ducks/Tree regressions fail the retention gate. Screen B was not
run; its scratch was created but contains no result. This one screen is
enough to reject this particular scalar staging prototype, not to rule out
a future true four-tile SIMD kernel.

`raw.csv`, `load.csv`, `summary.csv`, `output-stability.csv`, `identity.csv`,
`manifest.txt`, and `tx4-run-ab.ps1` are retained here. The transferred
Windows files matched these remote SHA-256 values before LF normalization:
raw `31cfa4e0a51e0e74a5d4d441c9266a4f830a7bc171498399b708cc3f657d22d5`,
load `582e04759e9fa6f49f5187cf334d8606f572bd99642c0c7e9120dd1f12ba9aaf`,
summary `736a03729cab074f6b2c00bf4ecb908f3b2b18f6af9030248c3ef5a9a5e115ed`,
stability `5090e4ea82f3fac044874c0e7b27cd642b4a304b84c82131eeb1064cefa05a63`,
identity `466701ec1d2d5d6ca27fe45b6c4c87fa9f0ba83be5e1e2044888ad476dc299b3`,
manifest `8ad786aed0be3a379ceee9905035473b5f5b358472f061571f9f293635271662`.
The checked-in LF-normalized SHA-256 values are raw
`424aeecd746925cf47c17fddcabb7ae5b6b4fc5105b94b8d0a8bca1b796e692e`,
load `8131df388d44eedb19956f8d3fbde3297d9333d405a900fd59b1e2e189095280`,
summary `9de601f65d50e49d97e4398981d938d1f6b95c97e88b690b42c5095066f3abda`,
stability `939f16f12f63c644b8563b12c1438a8ca190cd1b4fdf9f7e1c40c0f86a47fa28`,
identity `97e65a6884ab9d5ec860804d71c10a727503687620409e6c5d01b640ebe470c5`,
and manifest `cdecaedb81499c9a5e69ca22e19fbff278a6291a70ae2d6e04269530ce0a911f`.

Focused AArch64 validation: the deliberate staged-DC mutation made the new
tile differential fail (exit 101), then the restored test and four other
`inter_tx` tests passed. Targeted encoder Clippy and workspace rustfmt
passed without new warnings. A separate x86 focused unit-test run was not
needed for retention after the negative CPU screen; x86 release built and
completed 102 byte-exact encodes (six smoke plus 96 screen A).
