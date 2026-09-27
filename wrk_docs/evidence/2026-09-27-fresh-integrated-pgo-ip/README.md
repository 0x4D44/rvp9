# Fresh integrated production-PGO instruction profile

This profile samples the exact integrated-commit production-PGO EXE from the [direct parity screens](../2026-09-27-fresh-integrated-pgo-direct/README.md), SHA-256 `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`. The [capture wrapper](capture-fresh-pgo.ps1) reused the reviewed WPR/IP extraction path. Its [identity](profile-identity.csv) pins the PE/PDB GUID and age, PDB SHA-256 `c41525ee2ccd56d6bda7e4e9e0880452437d5ac7981bc2af4f1b7c34a71943f8`, ETL hashes, canonical output hashes, and zero lost buffers/events. Each clip's [summary](ducks-summary.json), capture log, `xperf tracestats`, top IPs, symbols, and source-line CSVs are retained here. The three raw ETLs and per-IP maps remain under `D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s\.scratch\fresh-integrated-pgo-ip` on Crucible; they are too large for the docs commit.

| Clip | Direct EXE samples | Unique IPs | Resolved source-line samples | Lost events/buffers |
| --- | ---: | ---: | ---: | ---: |
| Ducks | 18,127 | 7,440 | 17,955 | 0 / 0 |
| Tree | 8,195 | 3,843 | 8,140 | 0 / 0 |
| Town | 8,629 | 4,261 | 8,577 | 0 / 0 |

The [exact-IP disassembly](hotspot-disassembly.txt) and top-IP CSVs qualify the largest source labels:

- Ducks' top IP, RVA `0x1a3c58` (`movl 0xe00(%rsp), %ebx`), has 400 samples in a broad inlined picker. It is not a proven 400-sample removable stack-load cost.
- The cross-clip RVA `0x181c91` (`vmovdqu 0x30(%rcx), %xmm1`) has 202/84/100 Ducks/Tree/Town samples inside tight-plane inter prediction. A sampled load does not alone identify a safe or faster rewrite.
- Tree/Town's `0x1883b4–0x188435` cluster is the filter-candidate residual/metric loop; RVA `0x1883c9` alone has 139/78 samples. That arithmetic is intrinsic to candidate scoring unless a separate operation-level oracle finds redundant work.
- Ducks' NSTEP and full-pixel search symbols account for 1,798 and 1,053 samples, respectively, with another 604 in the diamond helper. These are disjoint symbol self-weights, not an estimated saving. The repeated NSTEP seed evaluation is a bounded hypothesis: count actual restart entries and identical initial-SAD calls before prototyping reuse.

The AVX2 sub-pixel variance wrappers together account for few direct Ducks samples in this image, so a broad bounds-check rewrite is not the first Ducks experiment. Broad source-line totals (`inter.rs:3986`, `full_pixel.rs:930`, `inter.rs:2428`, `lib.rs:11875`) include inlining and cannot be read as individual-operation costs. The traces identify where to measure repetition, not a promised speedup.
