# Exact-source production-PGO image pair

Two natural Crucible PGO builds with identical shipped source were paired directly. Their different profiles did **not** produce a useful equal-clip CPU separation: the earlier image used **0.998297×** the integrated image's process CPU across 15 pairs per clip (about 0.17% faster). This is too small to explain the roughly 0.7% improvement needed to move the current measured **1.057638×** RVP9/libvpx ratio below 1.05×, and it is not a code gain or a fresh libvpx parity claim.

The control was the exact integrated production-PGO EXE SHA-256 `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`, profile `3f02af95fbeca067d6d94b233a353681bce4ed25c76930a94f0564895ac10335`, built from commit `75c23fe224eedb26c9a014040c42b67e3a1b5fa1`. The candidate was the earlier natural PGO EXE `c0dc53bc3a3b8ede9d9c45e8606d0f4b09ab0f60c10e7e300d790e79d8b7bfbb`, profile `3f4c740d0b378789a48025491d79c54c46ac9cb80a50f`, built from commit `cf9282faf96277d8089781ab0951aaa9d5b21acb`. A remote `git diff --name-only cf9282faf 75c23fe... -- crates scripts Cargo.toml Cargo.lock` returned no paths. Both build manifests record the same 28-row, 60-frame training recipe and toolchain. The different profile hashes and EXE hashes remain real; this comparison tests their resulting CPU, not why they differ.

Under a Crucible `deltic quiet` lease with zero heavy builds at grant, the [runner](runner.ps1) checked both EXE hashes and three canonical input hashes, warmed both arms, alternated clip and arm order, then ran 15 pairs per clip. All 96 encodes exited zero and produced the expected canonical IVF hashes. The [raw rows](raw.csv), [summary](summary.csv), and [identities](identity.csv) are retained here. An independent calculation from all 45 measured pairs found no failures or missing arms:

| Earlier PGO image / integrated PGO image, process CPU | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| 15 pairs per clip | 1.000572 | 1.010607 | 0.983893 | **0.998297** |

This direct same-run comparison is stronger evidence about the two PGO images than comparing their separate historical libvpx screens. It does not establish that profile training is generally stable, or that either image now passes the direct ≤1.05× libvpx bar. No encoder code or PGO recipe changed. Bulky IVFs were held only in registered Crucible scratch and need not be retained; the quiet lease was ended after the screen.
