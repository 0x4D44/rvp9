# Chroma-energy direct parity after code integration

The chroma-energy certificate code landed on `origin/main` as edc9e7e0. The second naturally trained Crucible PGO executable predates that landing, but its `crates/rvp9-encoder/src/lib.rs` SHA-256 is **f26ab81338156b0710fc8db8d8f392617c029133417482e05e9504ff27ecc98a**, identical to the integrated encoder source. This is a source-identical prelanding production-PGO binary, **not a binary rebuilt from the integrated commit**. Its SHA-256 is c0dc53bc3a3b8ede9d9c45e8606d0f4b09ab0f60c10e7e300d790e79d8b7bfbb; the pinned libvpx executable is 88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d. The [independent build identities](../2026-09-27-chroma-energy-independent-pgo/README.md) preserve the PGO recipe, toolchain, input hashes, profile hash, and manifest.

The [direct runner](direct-integrated-pgo.ps1) checked both executable hashes and all three 60-frame 1080p input hashes, warmed both arms, alternated arm and clip order, and measured process CPU. Three consecutive 15-pair screens used the same binaries and default affinity. Host CPU was about 1% before the first screen and about 12% after the second; no quiet lease was available. Every one of the 288 encodes exited zero and matched its expected IVF hash. The [first](screen-1-raw.csv), [second](screen-2-raw.csv), and [third](screen-3-raw.csv) raw rows, summaries, and identities are retained here; all 45 pairs per clip are included.

| RVP9/libvpx process CPU | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| Screen 1, 15 pairs/clip | 1.175272 | 1.024398 | 0.954754 | **1.047529** |
| Screen 2, 15 pairs/clip | 1.139234 | 1.047015 | 0.971681 | **1.050420** |
| Screen 3, 15 pairs/clip | 1.156936 | 1.030312 | 0.951356 | **1.042814** |
| Pooled geometric mean, 45 pairs/clip | 1.157054 | 1.033864 | 0.959223 | **1.046917** |

The pooled result is below the ≤1.05 target, but one screen is just above it. This supports **near-parity**, not a comfortable or fully repeatable pass. Ducks remains the main gap. An earlier production-PGO binary from the same encoder source measured 1.065957× libvpx; that spread is not explained by the code change and cautions against treating this one PGO build as a durable margin. A later [fresh build from the exact integrated commit](../2026-09-27-fresh-integrated-pgo-direct/README.md) measured 1.053225× and 1.054912× in two direct screens, so the ≤1.05 target remains unproven.
