# Source-identical production-PGO repeat on Crucible

The production-PGO build from `cef7895fce53d40315946069324e8ff58324d902` used the unchanged 28-row/60-frame recipe, SHA-256 `2d335a8b02c41a3dc370238add4b894d2f9d27900ee9e657dcc3b9e7ba2d54e5`, and collected 31 raw profiles. Shipped files were unchanged from the previously [measured integrated source](../2026-09-27-fresh-integrated-pgo-direct/README.md). The preflight checked the worktree, recipe, source, pinned training inputs, and pinned control executable. The repeat's merged profile SHA-256 was `19f90b04c93ac4ce6f6301d6b72b4be89447e3f37ddfd41adc377ee19616259f`, versus `3f02af95fbeca067d6d94b233a353681bce4ed25c76930a94f0564895ac10335` previously. Its optimized EXE SHA-256 was `341d8d08f609913356216e367f7bac44ec39dfd1b1637a076efb41ae9e6c44f8`, versus `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad` previously. The [manifest](pgo-manifest.txt), [identity](identity.csv), and [protocol](protocol.txt) record the run.

The [paired screen](raw.csv) warmed both arms, alternated their order and clip order, and measured process CPU for 15 60-frame 1080p CPU6 CBR pairs per clip. Every encode exited normally and matched the canonical IVF hash; no rows were excluded. The new artifact was **slower** than the prior integrated artifact:

| Clip | Repeat / prior integrated CPU |
| --- | ---: |
| Ducks | 1.034305 |
| Tree | 1.015476 |
| Town | 1.024145 |
| Equal-clip geometric mean | **1.024613** |

The original integrated artifact directly measured at **1.054068× libvpx** over 30 pairs/clip. Multiplying the separate ratios suggests around 1.080× for this repeat, but that is **not** a direct repeat/libvpx measurement. Its CPU results establish a real build-to-build difference, not the cause of the profile difference; an EXE hash alone does not prove which code sections changed.

**Decision:** retain the earlier integrated artifact as the current direct parity baseline; do not claim reproducible ≤1.05× parity. Investigate production-PGO training reproducibility or code-section differences before attributing the observed gap to another encoder source hotspot. The run's full scratch outputs remain under `D:\worktrees\rvp9-2\20260927-TSK-HUM-repeat-integrated-pgo-parity\.scratch\repeat-pgo` and `.scratch\screen-01` on Crucible.
