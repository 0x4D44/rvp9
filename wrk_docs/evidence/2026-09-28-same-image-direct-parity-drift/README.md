# Same-image production-PGO direct parity drift

Two further Crucible screens reused the exact integrated production-PGO RVP9 binary and pinned libvpx binary from the [60-pair baseline](../2026-09-27-integrated-pgo-direct-repeat/README.md). Their SHA-256 identities remained `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad` (RVP9) and `88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d` (libvpx). The same [direct runner](../2026-09-27-fresh-integrated-pgo-direct/direct-integrated-pgo.ps1) used `-Libvpx -Pairs 15` for each independent screen, with one warmup per arm and clip, alternating order, canonical 60-frame 1080p inputs, and exact expected IVF hashes. Every one of 192 encodes exited zero and matched its arm's expected output; 180 non-warmup encodes form 90 paired observations.

| RVP9/libvpx process CPU | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| Screen A, 15 pairs/clip | 1.149316 | 1.000628 | 0.942063 | **1.027063** |
| Screen B, 15 pairs/clip | 1.140943 | 1.010253 | 0.948804 | **1.030284** |
| Both screens, 30 pairs/clip | 1.145122 | 1.005429 | 0.945427 | **1.028672** |
| Earlier 60 pairs/clip, same images | 1.169951 | 1.034466 | 0.977524 | **1.057638** |

The two newer screens fall below the ≤1.05× target, but the stronger earlier baseline exceeds it with the *same binaries and inputs*. This is benchmark-environment drift, not a code gain. In the newer pooled screens the absolute CPU means rose for both arms relative to the earlier 30-pair group: Ducks RVP9/libvpx 1.918229/1.675000 s versus 1.827083/1.562500 s, Tree 1.030729/1.025521 s versus 0.988021/0.942708 s, and Town 0.953125/1.007813 s versus 0.902604/0.925521 s. Libvpx slowed proportionally more, which moved the ratio. Do not declare parity from these shorter screens; a final claim needs fresh repeated direct pairing under a controlled host regime.

The checked-in [screen A raw rows](screen-a-raw.csv), [summary](screen-a-summary.csv), and [identity](screen-a-identity.csv), plus corresponding [screen B raw rows](screen-b-raw.csv), [summary](screen-b-summary.csv), and [identity](screen-b-identity.csv), preserve all measured and warmup encodes. The original bulky IVFs remain under `D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s\.scratch\baseline-repeat-20260928` and its `-b` sibling on Crucible. No quiet lease was held; process CPU is the metric, and live host-load history was not recorded. These screens cannot explain the drift's cause.
