# Fresh integrated-trunk production-PGO direct parity

Crucible built `rvp9enc.exe` from exact integrated commit `75c23fe224eedb26c9a014040c42b67e3a1b5fa1` (tree `48ea2b89b840ffd95dadb71dc53392ef02e53c53`). The encoder source SHA-256 is `f26ab81338156b0710fc8db8d8f392617c029133417482e05e9504ff27ecc98a`. It used the unchanged [28-row/60-frame PGO recipe](../../../scripts/build-encoder-pgo.sh), SHA-256 `2d335a8b02c41a3dc370238add4b894d2f9d27900ee9e657dcc3b9e7ba2d54e5`, with 31 raw profiles, Rust/Cargo 1.95.0, LLVM 22.1.2, and native Ryzen 9 9950X3D code generation. The [build identity](build-identity.csv) and [manifest](build-manifest.txt) retain the tree, toolchain, training and generated-input hashes, profile hash `3f02af95fbeca067d6d94b233a353681bce4ed25c76930a94f0564895ac10335`, and EXE hash `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`.

The [direct runner](direct-integrated-pgo.ps1) checked that EXE and pinned libvpx `vpxenc.exe` SHA-256 `88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d`, plus all three input hashes. It warmed both arms, alternated arm and clip order, and measured paired process CPU for 60-frame 1080p CPU6 CBR encodes. It required each IVF to match its arm's canonical hash. [Host context](host-context.txt) records no quiet lease, 0% load before the build and 2% at the second screen's start; it does not establish load throughout either screen. No pair was discarded.

| RVP9/libvpx process CPU | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| Screen 1, 15 pairs/clip | 1.180387 | 1.031704 | 0.959366 | **1.053225** |
| Screen 2, 15 pairs/clip | 1.160893 | 1.010733 | 1.000507 | **1.054912** |
| Pooled, 30 pairs/clip | 1.170600 | 1.021164 | 0.979721 | **1.054068** |

The [first](screen-01-raw.csv) and [second](screen-02-raw.csv) raw/summary/identity CSVs contain all 192 encodes, including 12 warmups. An independent recomputation from raw rows found 0 exit or output-hash failures, all 90 measured pairs, and the ratios above; [verification](independent-verification.csv) retains a separate check. The focused native x86 chroma-certificate tests and encoder all-target check passed; the latter reported two pre-existing dead-code warnings.

**Decision:** the exact integrated build misses the ≤1.05× equal-clip CPU target in both screens. A previous source-identical prelanding PGO artifact [pooled at 1.046917×](../2026-09-27-chroma-energy-direct-parity/README.md), with one of three screens above the bar. The two artifacts have different profiles and were measured in different runs, so the spread cannot be assigned to one cause. The code remains a reproducible gain over its control, but parity is not yet proven. Fresh instruction profiling of this exact image should select the next bounded experiment; Ducks remains the largest gap.
