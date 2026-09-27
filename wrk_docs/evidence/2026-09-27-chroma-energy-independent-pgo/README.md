# Independent production-PGO retention pair

The early chroma-energy shortcut is exact but its speed varies with the compiler profile. To decide whether to retain it, Crucible independently regenerated full 28-row, 60-frame production PGO profiles and optimized executables for unchanged integrated control source c717a24a and candidate source cf9282faf. The local candidate was later rebased without changing the encoder source hash. Both builds used the same recipe SHA-256 2d335a8b02c41a3dc370238add4b894d2f9d27900ee9e657dcc3b9e7ba2d54e5, rustc/cargo 1.95.0, LLVM 22.1.2, and ffmpeg 8.1.1. The [control identity](control-build-identity.txt), [candidate identity](candidate-build-identity.txt), and their [control](control-manifest.txt) and [candidate](candidate-manifest.txt) manifests preserve source trees, toolchain, all six training-input hashes, profile hashes, and output hashes.

| Fresh build | Profdata SHA-256 | Optimized EXE SHA-256 |
| --- | --- | --- |
| Integrated control | fc48ee8b2f0c58740092d0bc0a7cf62e9d4f416087f3092a7b5f53f33583f1af | aec16662e6c4cffcee88e3842b79a8795975eea545b24bfd272d31360dc77009 |
| Energy candidate | 3f4c740d0b378789a48025491d79c54c46bf42febb20969457c46ac9cb80a50f | c0dc53bc3a3b8ede9d9c45e8606d0f4b09ab0f60c10e7e300d790e79d8b7bfbb |

The [runner](screen-retention-natural-pgo.ps1) pinned both executables and the generated canonical 1080p inputs, warmed both arms, alternated arm and clip order, and measured process CPU. [Run context](run-context.txt) records 8.884% host CPU at start, default process affinity, no quiet lease, and no active compiler/linker process. A separate rustc process appeared only after the screen CSVs finished. The retained [identity](identity.csv), [raw pairs](raw.csv), and [summary](summary.csv) contain all observations.

| Candidate/control process CPU | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| Independent fresh PGO, 15 pairs/clip | 0.944390 | 1.001001 | 1.003566 | **0.982601** |

All 90 timed encodes and six warmups exited zero and matched their canonical IVF SHA-256; 48/48 control–candidate pairs had equal hashes. An independent parser recomputed 15 ratios per clip and found no failed or mismatched row. The candidate won all 15 Ducks pairs. Tree and Town are effectively flat on this build pair.

This is a second independently trained production-PGO pair with an equal-clip gain, following three positive screens of the first pair. The [shared-profile crossover](../2026-09-27-chroma-energy-shared-profile/README.md) still warns that effect size depends on the generated profile, but it does not negate the independently repeated natural-build gain. **Retention decision:** the exact shortcut now meets the repeatable-gain gate for integration; integrate the already-tested source, then build and measure the integrated trunk directly against pinned libvpx before claiming parity. The latest pre-integration direct candidate/libvpx ratio is 1.065957×, above the ≤1.05× target.
