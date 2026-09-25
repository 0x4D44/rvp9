# Unit 1426 — current integrated x86 encoder CPU profile

This profile uses integrated source `e588d13fbb12c9cbedd6a87e57a9236c1aedaea8`
on Crucible. It changes no codec or benchmark code. The native Rust 1.95.0
release build has `CARGO_PROFILE_RELEASE_DEBUG=2` and no declared PGO. Its
`rvp9enc.exe` SHA-256 is
`c371d472690baba94dc1374b9ce4b65dddd3f3ac2381feca362efa25e55f0d98`;
the matching PDB SHA-256 is
`a3995992a6eb0ce343fed14f01830737ce577b708dde8379d6913902c7ae67fc`.
Each run encoded 60 frames at 1920×1080, CPU-used 6, CBR 1260 kb/s, one
thread, tile 0, AQ 0, noise 0, lag 0, and KF 60. The command was
`rvp9enc --codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr
--target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0
--lag-in-frames=0 --tune-content=default --aq-mode=0
--noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf
-o <output.ivf> <input.y4m>`.

| Clip | Input SHA-256 | IVF bytes | Stable IVF SHA-256 | Valid trace ETL SHA-256 | xperf report SHA-256 |
|---|---|---:|---|---|---|
| Ducks | `6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7` | 529,682 | `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089` | `91724a23344b9fa3c317994cf94910073c22e1d2ed5f78abf1a36de7005ca2d8` | `a18783394c2d7682b8cb0176a195ba34362f8345837f856bdab92530eb686445` |
| In-to-tree | `ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791` | 337,194 | `5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41` | `374ddc46e0e72fd5ca8250daff2ce5fdd051ef40d0b995bcc7563de231a49183` | `bc5d68545cb2a978453b080b44c61799a9bd8bccf4a582e21925760a041f4b89` |
| Old-town-cross | `b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909` | 358,110 | `4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f` | `f874773bf10f201e11daea12de59049d85c2771779232c12502a16bb8d4fce92` | `3692c1987c2da46e64d3e552289c719e47744d2235803d7a22f08a427ca72619` |

WPR memory-mode traces contain three Ducks, two In-to-tree, and two
Old-town-cross serial encodes. Every encode exited successfully and produced
the stable IVF hash above. Each retained WPR stop and xperf profile exited 0;
neither reported dropped events. This means *no loss reported*, not an
independent numeric zero-loss count. xperf resolved the matching encoder PDB.
System PDB misses are excluded from the `rvp9enc.exe` self-weight rankings.
Raw ETLs are 1.0–1.05 GB each and remain in Crucible task scratch at
`D:\worktrees\rvp9-2\20260925-TSK-CDX@CRUCIBLE-unit1421-cpu-profile\.scratch\unit1426-x86-profile`,
not the repository. The remote manifest is SHA-256
`2c334693dedb519882b8f9fbd5b39267b45079e9743f314df00cad7044bd1f5d`.

| Encoder symbol, self weight | Ducks | In-to-tree | Old-town-cross |
|---|---:|---:|---:|
| `pick_and_simulate_inter_encode_partitions::closure$10` | 13.67% | 7.87% | 9.21% |
| AVX2 sub-pixel variance (16×16 wrapper in Ducks) | 4.18% | 15.46% | 12.70% |
| `reconstruct_via_round_trip` | outside top 15 | 11.35% | 9.68% |
| `pick_inter_mode_impl` | 6.15% | 2.95% | 4.10% |
| `vp9_quantize_fp_avx2_inner` | 5.04% | 3.92% | 5.24% |
| `find_realtime_refs_nearest_near_mvs_with_context` | 4.63% | 2.70% | 2.50% |
| `fdct8x8` | 2.96% | 2.52% | 3.32% |

Denominators are 10,135,650 / 3,508,169 / 3,029,594 resolved encoder
self-weight samples respectively. These percentages are directional CPU
self-weight ceilings, **not** elapsed-time shares or predicted speedups.
`top15.csv` preserves each clip's full top-15 symbol ranking.
Ducks names the specific 16×16 sub-pixel wrapper; In-to-tree and Old-town
attribute the shared implementation. Optimizer inlining may explain this
symbol split, so compare their direction rather than treating the values as
an exact like-for-like function measure.

**Timing is unusable for retention.** Crucible's `D:` read latency averaged
0.94–2.03 seconds in the pre-profile health windows. During seven sampled
In-to-tree encode seconds, `D:` read latency averaged 272 ms (1.31 seconds
peak), queue depth averaged 31 (109 peak), and total CPU averaged 92.5%.
Later, with CPU averaging only 18.2%, `D:` read latency averaged 2.35 seconds
(6.78 seconds peak). Ducks elapsed/process-CPU seconds were
5.056/3.984, 4.145/3.750, 4.038/3.719; In-to-tree 3.032/1.984,
3.526/2.109; Old-town 3.250/1.906, 2.027/1.719. Neither low average
load nor exact outputs make these wall times a valid paired speed comparison.

Two earlier traces were rejected: a 10-run Ducks WPR file-mode trace reported
26,062 dropped events, and a three-run In-to-tree memory trace reported one
lost event in xperf even though WPR stop itself said it saved successfully.
The rankings above use only the subsequent memory traces with no reported loss.

**Next bounded study:** Attribute the sub-pixel variance and round-trip
reconstruction self samples to source lines and invocation mix before coding.
Sub-pixel variance is large on two clips, while the partition picker remains
material across all three. Choose a small exact-output kernel change only
after that attribution; then run differential tests with a deliberate red,
both-architecture validation, and paired pinned-libvpx timing once the host
can yield stable measurements. This unit claims no retained speedup and does
not revise Unit 1422's provisional libvpx ratio.

The bounded source-line follow-up stopped without an attribution result. The
saved xperf reports expose function aggregates, not sampled instruction
addresses. Reading the Unit 1426 ETLs through `Get-WinEvent` yielded opaque
providerless events, so the Unit 1417 DbgHelp mapper had no per-IP input.
No line counts can be inferred from the function percentages. A future
capture or extractor must first preserve validated sampled-IP rows with
the exact image base and PDB; the retained ETLs are still available in
Crucible task scratch for a read-only extraction attempt.
