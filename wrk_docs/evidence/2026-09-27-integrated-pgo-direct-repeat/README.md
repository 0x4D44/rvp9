# Same-image integrated production-PGO direct parity repeat

This is a third direct paired screen of the **same exact** integrated RVP9 production-PGO EXE as the [prior two screens](../2026-09-27-fresh-integrated-pgo-direct/README.md), SHA-256 `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`, against pinned libvpx `vpxenc.exe` SHA-256 `88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d`. No encoder code or PGO profile changed. The same [direct runner](../2026-09-27-fresh-integrated-pgo-direct/direct-integrated-pgo.ps1) used `-Libvpx -Pairs 30 -RunLabel direct-screen-03-20260927` on Crucible. It checked both binaries and all three canonical 60-frame 1080p inputs, warmed both arms, alternated arm and clip order, and required each arm's canonical IVF hash. The run retained every pair.

Crucible reported zero heavy-build processes and approximately 1% CPU load before the screen. There was no quiet lease: the host was in the previous lease's cooldown. Load throughout the run was not captured, so these are **process CPU** results, not a claim about an idle wall clock.

| RVP9/libvpx process CPU | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| New screen, 30 pairs/clip | 1.169302 | 1.047941 | 0.975331 | **1.061219** |
| Earlier pooled screens, 30 pairs/clip | 1.170600 | 1.021164 | 0.979721 | **1.054068** |
| All 60 pairs/clip, pooled geometrically | 1.169951 | 1.034466 | 0.977524 | **1.057638** |

The exact same image exceeds the ≤1.05× target in both independent 30-pair groups. Ducks stays near 1.17×; Tree shifted upward in this screen. The difference cannot be assigned to code or profile changes and does not justify selecting or rejecting an optimization. Do not claim parity from the earlier source-identical prelanding image that happened to screen below 1.05×. [Raw rows](screen-03-raw.csv), [per-clip summary](screen-03-summary.csv), and [binary/input identities](screen-03-identity.csv) retain this run; the bulky IVFs remain under `D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s\.scratch\direct-screen-03-20260927` on Crucible.
