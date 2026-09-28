# Serial loop-filter copy-elision production-PGO screens

**Decision:** the serial packet-overlap walker is a repeatable, exact-output equal-clip CPU gain over the integrated production-PGO control and may be integrated, with a disclosed Ducks trade-off. The one-worker split-edge variant is slower and must not be selected. Neither pair of short direct parity screens alone proves a stable host-independent ≤1.05× ratio; the [same-image baseline drift](../2026-09-28-same-image-direct-parity-drift/README.md) still applies.

The [inclusive ETL stack replay](../2026-09-28-loop-filter-inclusive-stack/README.md) attributed 5.4%/11.3%/8.2% of Ducks/Tree/Town target-process samples to allocation and copies in the serial bordered loop-filter adapter. For 1080p retained inter reconstruction, the new route uses already SB-padded tight planes instead of allocating a bordered Yv12 frame and copying every visible plane in and out. A full-backing differential initially found that direct filtering changed padded chroma-tail bytes. Both variants now save and restore all Y/U/V tail bytes. The production route is limited to partial-SB heights divisible by eight, widths divisible by 64, and exact SB-rounded Y/U/V backing lengths; the existing multi-worker dispatch and all other geometries stay unchanged.

| Candidate/integrated RVP9 process CPU | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| One-worker split-edge, screen 1, 15 pairs/clip | 1.029841 | 0.997852 | 0.996548 | **1.007964** |
| One-worker split-edge, screen 2, 15 pairs/clip | 1.042371 | 0.989455 | 0.990725 | **1.007219** |
| One-worker pooled, 30 pairs/clip | 1.036087 | 0.993645 | 0.993632 | **1.007591** |
| Packet-overlap walkers, screen 1, 15 pairs/clip | 1.014080 | 0.980897 | 0.984835 | **0.993161** |
| Packet-overlap walkers, screen 2, 15 pairs/clip | 1.011460 | 0.975392 | 0.970714 | **0.985688** |
| Packet-overlap pooled, 30 pairs/clip | 1.012769 | 0.978140 | 0.977749 | **0.989418** |

Ratios below one mean less process CPU. The packet route saves **1.06% equal-clip CPU** pooled and wins both independent screens. Ducks consistently costs **1.28% more**; Tree and Town save about 2.2%. This trade-off improves the explicit equal-clip parity objective, but Ducks remains the weakest clip and should be watched in the next integrated-trunk screen. The one-worker route loses 0.76% pooled and remains only as a local prototype history step.

| Packet candidate/pinned libvpx process CPU | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| Direct screen 1, 15 pairs/clip | 1.169602 | 0.970532 | 0.922152 | **1.015353** |
| Direct screen 2, 15 pairs/clip | 1.156716 | 0.972572 | 0.915874 | **1.010016** |
| Direct pooled, 30 pairs/clip | 1.163141 | 0.971552 | 0.919008 | **1.012681** |

The direct candidate is below 1.05× in both screens. The earlier exact integrated control measured 1.057638× over 60 pairs/clip with these same libvpx and input images, while shorter later control screens measured 1.028672×. Host/process-CPU regime drift affects the direct ratio; the paired candidate/control screens establish the code gain more reliably. A final parity claim requires rebuilding and repeating the direct screen from the exact integrated tree after the version bump.

The integrated control EXE SHA-256 was `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`; one-worker candidate was `36aadc145853d62b5600ad870bc7db9e0f5f2c26f6421befb2861ba3f6fff129`; packet candidate was `438c94602239f4c143c6efec36dd985feaf4d4e7d9905a629f17f280ca3f5b22`; pinned libvpx was `88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d`. Both candidates used the unchanged [28-row production PGO recipe](../../../scripts/build-encoder-pgo.sh), 60 frames per row, 31 raw profiles, Rust 1.95.0 and native Ryzen 9 9950X3D compilation. [One-worker](one-worker-pgo-manifest.txt) and [packet](packet-pgo-manifest.txt) manifests retain the distinct source commits and profiles.

Every raw encode in the six full screens exited zero and matched its arm's canonical IVF hash: 576/576 including 36 warmups, with 90 paired observations per candidate type and 90 paired direct observations. Each screen alternated arm and clip order. The [one-worker](screen-one-worker.ps1), [packet](screen-packet.ps1), and [direct](direct-packet.ps1) runners pin binary/input hashes, CPU6/CBR1260, one thread, tile-columns zero, and 60-frame clip outputs. Per-screen `raw.csv`, `summary.csv`, and `identity.csv` are retained in [one-worker-01](one-worker-01/), [one-worker-02](one-worker-02/), [packet-01](packet-01/), [packet-02](packet-02/), [direct-01](direct-01/), and [direct-02](direct-02/). Bulky IVFs and PGO build products remain under `D:\worktrees\rvp9-2\20260928-TSK-HUM-prototype-serial-tight-loop-filter-x86\.scratch` on Crucible.

The new mixed-inter test uses active Y/U/V step edges, 1920×1080 and 320×72 partial-SB frames, and distinct nonzero padded tails. It was deliberately red when the direct helper modified tail bytes, then green after tail preservation on AArch64, scalar-forced AArch64, native x86, and scalar-forced native x86. Existing packet-overlap and complete-height split-edge differentials passed. `cargo fmt --all --check` and `cargo clippy -p rvp9-encoder --lib` passed locally; clippy reported one pre-existing type-complexity warning in untouched `lib.rs:4846`. No full workspace suite or BD-rate run was performed for this exact-output runtime-path change.
