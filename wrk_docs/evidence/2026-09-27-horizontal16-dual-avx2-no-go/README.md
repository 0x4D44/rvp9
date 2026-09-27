# Horizontal-16 dual AVX2 fusion: exact, no repeatable CPU gain

Do not integrate the dual-kernel prototype. Its 16-column AVX2 pass matches the two existing 8-column passes and scalar filter in focused x86 tests, and all 192 paired-screen encodes matched the integrated IVF hashes. The first 15-pair screen saved 0.70% equal-clip process CPU; the independent repeat was effectively flat and made Ducks and Town slower. A 951-line kernel does not clear the repeatable-gain bar on this evidence.

The prototype is committed but unpushed as `2bfbd1f2` on `task/20260927-TSK-HUM-fuse-horizontal16-avx2-dual-filter-lanes`. Its preceding `05677909` commit fixes the existing single-kernel blimit-255 saturation defect, recorded separately as `RVP9-BUG-FLU-00035`. Neither commit is on production trunk. The older vertical-16 shared-sums prototype likewise remains out after its own no-go screen.

## Correctness and identity

The x86 horizontal-16 filter suite passed 18/18 tests on Crucible after the fix. The new dual-vs-two-singles-vs-scalar differential failed deliberately when one right-half output byte was corrupted, then passed after restoration. The new blimit-255 single-kernel regression failed deliberately when its widened threshold path was disabled, then passed after restoration. Local AArch64 loopfilter tests passed 129/129, as did formatting and diff checks. The candidate PGO EXE SHA-256 was `656f86ea6309976658a781e5106cce1c9a731cf89ae6d296d5e824dbe7861cd7`, built with the integrated `encoder.profdata` SHA-256 `3f02af95fbeca067d6d94b233a353681bce4ed25c76930a94f0564895ac10335`. The integrated control EXE was `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`.

Each run used 15 paired 60-frame encodes per clip, plus one warmup per arm, on Crucible under a quiet lease. The wrapper checked binary and input hashes and every output hash. The per-encode raw CSVs, identities, IVFs, and wrapper remain in `D:/worktrees/rvp9-2/20260927-TSK-HUM-horizontal16-dual-avx2-candidate/.scratch/horizontal16-frozen-15-{a,b}` and `horizontal16-screen.ps1`. No fresh candidate profile was trained because the frozen-profile gain did not repeat.

| Candidate / control process CPU | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| 15-pair screen A | 0.995368 | 0.996902 | 0.986851 | **0.993030** |
| 15-pair screen B | 1.004049 | 0.987967 | 1.008427 | **1.000109** |

The integrated encoder/libvpx ratio remains **1.057638×**; neither this candidate nor the diagnostic-counter compile-out changed production performance.
