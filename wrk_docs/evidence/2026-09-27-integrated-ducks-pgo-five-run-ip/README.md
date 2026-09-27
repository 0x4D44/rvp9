# Five-run exact integrated production-PGO Ducks IP repeat

Five captures of the same pinned production-PGO `rvp9enc.exe` produced **89,310 direct EXE samples** and the same canonical Ducks IVF SHA-256 in every run. This repeat strengthens hotspot attribution but does **not** identify a new removable operation or change the integrated RVP9/libvpx equal-clip CPU ratio of **1.057638×**. No code candidate was retained.

## Identity and capture

The encoder EXE SHA-256 was `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`; the paired PDB SHA-256 was `c41525ee2ccd56d6bda7e4e9e0880452437d5ac7981bc2af4f1b7c34a71943f8`, GUID `f615f4ff-a877-e7c8-4c4c-44205044422e`, age 1. The pinned 60-frame Ducks input SHA-256 was `6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7`; all five output IVFs matched `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`. The shipped source tree remained identical to the [original integrated image](../2026-09-27-fresh-integrated-pgo-ip/README.md). The existing capture is run 01; four more used the reviewed WPR/ETW wrapper with exact HEAD, EXE, PDB, input, and output checks. Every extraction reported zero lost events and buffers. Crucible had about 8% CPU and four heavy processes when the repeat began; this was a load observation, not an idle-box guarantee.

| Run | Direct EXE samples | ETL SHA-256 |
| --- | ---: | --- |
| 01 | 18,127 | See original [capture identity](../2026-09-27-fresh-integrated-pgo-ip/profile-identity.csv) |
| 02 | 15,208 | `103dc655296d1c3bd7a200ef01c5d58cbe573ab39075a9b4d2de6b8ca7243328` |
| 03 | 19,064 | `c1d37150438f7d967c46dd88b56c76186fc8ace3551058cc29a9fa3a3b7a98be` |
| 04 | 18,243 | `9771b0a465f9772d0bf57035ad87cb81d41177a4fda1d0e415540fbf6237385b` |
| 05 | 18,668 | `b2d1e55b15662336132353ffc6ace6be25b7b12b03eb70778f40184f23fd3791` |

The full ETLs, extracted instruction maps, capture logs, and aggregate CSV/JSON remain in Crucible scratch at `D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s\.scratch\fresh-integrated-pgo-ip`. They are not committed because they are bulky profiling artifacts. The local task worktree's `.scratch` retains the small capture/extraction/aggregation adapters and aggregate CSV/JSON until this task is integrated.

## Stable and unstable instruction locations

These are **direct sampled IPs**, not estimates of removable CPU. RVAs normalize ASLR and were resolved against the exact PDB. Each bracket lists runs 01–05. The first three instructions were already disassembled in the [original exact-IP evidence](../2026-09-27-fresh-integrated-pgo-ip/hotspot-disassembly.txt).

| RVA | Direct samples | Per run | Interpretation |
| --- | ---: | --- | --- |
| `0x1a3c58` | 2,193 | 400, 373, 467, 472, 481 | Picker stack load, `inter.rs:2428`; broad inlined bookkeeping, not a removable 2.46% cost. |
| `0x10c9bd` | 1,137 | 213, 167, 239, 271, 247 | Luma TX selection region, `lib.rs:11875`; selector forwarding was already tested and rejected. |
| `0x181c91` | 882 | 202, 142, 198, 176, 164 | Tight-plane prediction vector load, `scale.rs:499`; identity-scale rewrite was already tested and rejected. |
| `0xf8af0` | 801 | 142, 133, 169, 176, 181 | IP immediately after proof-counter `lock incq`; compile-out prototype was exact but flat/slower in paired PGO CPU. |
| `0x1db7b0` | 497 | 98, 73, 104, 121, 101 | NEWMV search; prepared-seed experiment was rejected. |
| `0x1d7872` | 389 | 81, 72, 78, 83, 75 | NEWMV seed region; same rejected experiment. |
| `0xcc005` | 367 | 73, 65, 83, 71, 75 | Emit MV context region, `lib.rs:9924`; a *diagnostic* lead only, requiring an operation-level redundancy census before a code experiment. |

Two high-ranking IPs are not stable: `0x19b478` in sub-pel `check` had 10, 131, 233, 238, 219 samples; `0x1daac3` in NEWMV search had 1, 84, 114, 114, 121. The same EXE, PDB, input, and IVF hashes rule out a code or encoded-content change. Run 02 also has 17.9% fewer samples than the mean of the other four. ETW sampling-interval changes are plausible but **not proven**: these artifacts do not record `xperf -ProfInt` for each run. Do not use either unstable IP to justify an optimization without a controlled recapture or an operation-level counter.

The next bounded step is to inspect `0xcc005` at instruction and source level and count any genuinely repeated MV-context construction. This is a hypothesis to test, not a CPU-saving claim. The main parity gap remains `0.007638×` pinned libvpx equal-clip CPU above the 1.05× bar.
