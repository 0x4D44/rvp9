# Unit 1427 — AVX2 sub-pixel horizontal `maddubs` prototype, rejected

Unit 1426 profiled integrated source
`e588d13fbb12c9cbedd6a87e57a9236c1aedaea8` on Crucible. The candidate
worktree branched from its docs-only descendant
`bdb1abd289c8f7bd82c91921005a721aa1f10488`. Sub-pixel
variance held 15.46% of In-to-tree and 12.70% of Old-town encoder self
samples, making its horizontal bilinear pass a bounded candidate. This unit
replaces two widened multiplies with one unsigned-byte/signed-byte AVX2
multiply-add on adjacent source pairs. It keeps the scratch layout, vertical
pass, block variance reduction, and output semantics. The prototype source
is archived as `unit1427-subpel-maddubs.diff.gz`; it is **not retained**.
The archive SHA-256 is
`06e59d5b59757b6e9162439c5988b537b7cf3b146c5f6b47e1219e629a1b3e6b`.

## Correctness and build identity

The candidate was tested against the existing scalar bilinear/variance
oracle across every supported block shape, all eight horizontal offsets,
both the general and zero-vertical paths, tight and padded strides, and
exact source slice ends. On Crucible, the new x86 differential passed, the
AVX2 sub-pixel module passed 6/6 tests, `cargo check -p rvp9-core` and
`cargo fmt --all --check` passed. A deliberate `direct[0] ^= 1` mutation
failed the new test at width/height 4×4, xoffset 0, stride 5 (expected
223, mutated 222); removing it restored a passing test and the exact
prototype file hash. The AArch64 focused sub-pixel tests passed 3/3.

Both release executables used Rust 1.95.0, `CARGO_PROFILE_RELEASE_DEBUG=2`,
native target CPU, no PGO, and the same source except the candidate file.
The control executable/PDB SHA-256 values were
`c371d472690baba94dc1374b9ce4b65dddd3f3ac2381feca362efa25e55f0d98` /
`a3995992a6eb0ce343fed14f01830737ce577b708dde8379d6913902c7ae67fc`;
candidate values were
`3c595758037fc5d557ff3f8ce48288e9c92d7092d68171da715fe5869b10b4f2` /
`896a3f5bed8da4da62c4ca337ad6d40bd4f93f9c2fc2be5767b559742a7ae534`.
The prototype source SHA-256 was
`3fff96b480dcc2f2bb069f96441e1372515f98c4e6b55c11ec628729e21e6146`.
One candidate whole-encode smoke per clip and every benchmark output
matched the control IVF byte length and SHA-256 values in Unit 1426.

## Paired C: screen A

Crucible's D: drive had previously shown multi-second reads. A Deltic-
registered scratch on C: held both executables, the three hash-pinned Y4Ms,
and outputs. A control-only check produced five exact runs per clip; Tree
and Town wall times clustered near 2.02 seconds, while one Ducks run was
4.16 seconds against a 3.04-second median. The actual paired screen used
one warmup per arm per clip, then seven serial AB/BA-interleaved pairs per
clip over forward/reverse clip orders. It omitted the load ceiling, as
agreed with Arthur. All 48 outputs (six warmups plus 42 measured) matched
their expected sizes and hashes. No background counter sampler ran after
its transient provider failure; pre/post counters are preserved, but do
not establish load during each encode.

| Clip | Paired wall candidate/control | Paired process-CPU candidate/control | Control/candidate wall medians, s |
|---|---:|---:|---:|
| Ducks | 1.00203× | 1.06818× | 3.0541 / 3.0535 |
| In-to-tree | 0.99931× | 1.00182× | 2.0472 / 2.0486 |
| Old-town-cross | 1.00114× | 1.05004× | 2.0450 / 2.0462 |
| Equal-clip geometric mean | **1.00082×** | **1.03963×** | — |

These paired ratios were independently recomputed from all 42 rows in
`screen-a-measured.csv`; each pair contains one exact control and one exact
candidate output. Wall time is neutral, and process CPU regresses about
4%. Ducks' CPU ratio is 1.06818× despite a near-neutral wall median.
Screen A therefore rejects the candidate without a second screen or
pinned-libvpx run. There is no retained speedup and no new parity figure.

The runner completed all encodes, then exited 1 in its display-only
aggregation because a log string had entered the object stream. The raw
CSV has 42 valid rows and all 48 output files were verified afterward.
The separate postprocessor wrote ratios, summaries, and a manifest but
also exited 1 at its final display expression; those files were read and
hashed after the error. Neither post-run error affected an encode or caused
a rerun.

The seven CSV/JSON files were normalized from Windows CRLF to repository LF
after transfer; only line endings changed. The three test logs were converted
from Windows UTF-16/CRLF to UTF-8/LF and trailing whitespace was trimmed.
The manifest records the original
remote CSV/JSON hashes. The hashes below identify the checked-in files.

| Evidence artifact | Checked-in SHA-256 |
|---|---|
| `screen-a-measured.csv` | `02e28ac1d6ddddfcedbf3f8f0c2e730a583c827e28b6dbc05327234346f09145` |
| `screen-a-warmups.csv` | `8d2dfc3e6c91879ab23fa5c9b66ae3cf559bf142b6ecb20759da7a54f6685f41` |
| `screen-a-paired-ratios.csv` | `c4923a875f01d088d4419df984a176e4868b259961e4e6e73bd3e85e652bf3e9` |
| `screen-a-summary.csv` | `2ef05586f9047acb98edf15089eea6a39be01e3c99804b78cc8b09c78dd312df` |
| `screen-a-pre-post-counters.csv` | `e19f19ffe078bf7fa7a294b000aa58ed4f8dae0acd25f2d593b0c9eb2596ad55` |
| `screen-a-counter-summary.csv` | `9cb01010f3f688bc6408cfd9d4724518746d6fedc4fc7ef7f2c23682244c52f1` |
| `screen-a-manifest.json` | `7553f13aa5943dbde831c85dde27ec2e61c2840d578f65cf6862cb9723c1b8d3` |
| `unit1427-differential-red.txt` | `111b353f31666c5adc59d5598f1227ff0c8c40aadf1c55441202b58805084890` |
| `unit1427-differential-restored-green.txt` | `30b9ecc053a861c85e48f5edb2a18ba75d5007e9d73df24bffbfe1e203c3d054` |
| `unit1427-subpel-module-tests.txt` | `7d86c3ed0230c8049875bd4a715a608a8ed67dc135e89390d316cfdb568909b6` |

The raw counter windows showed C: read latency near 1.6 ms both before
and after the screen, but queue means were 12 and 10.3; D: recorded no
activity in those short windows. Total CPU means were 13.7% before and
33.5% after. These are context, not per-encode corrections. The retained
Unit 1426 profile still points at sub-pixel variance and the partition
picker, but a hot symbol is not a guarantee that this arithmetic rewrite
helps. Future candidates need their own exact and paired evidence.
