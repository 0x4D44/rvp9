# Dual-TX chroma residual reuse: exact, slower on production PGO

Prototype commit `88cfdd57` fills each U/V 8×8 residual tile once, then tests TX8 and its four TX4 subtiles from that buffer. It keeps the original zero-quadrant skip and does not change entropy state. The test compares its decisions with the prior transform traversal over zero, sparse, strided, nonuniform, multi-tile, compact-window, and edge cases. Deliberately using the wrong fourth-quadrant offset made the test fail; restoring offset `36` made both focused tests pass. AArch64 encoder check and formatting passed. The identical source-file SHA-256 on Crucible was `e5aade199a02448a7070e4e8249d4e5e25772089ef1888c8aff92f6264db1e49`; native x86 ran both focused tests successfully.

The control is integrated source `c717a24a` (same shipped code as later docs-only `b5ee2247`), built with the 28-row production PGO recipe. Its EXE SHA-256 is `ce0b5bc008ab9700abacad99ea3445691e1a3ca8b5f114b9ce682e15d27730f2`. The candidate EXE SHA-256 is `0778cabc5d2939858e5faeeaddc07c6bbb92c0a85c512786fdf51eec542b7839`. [Control](control-pgo-manifest.txt) and [candidate](candidate-pgo-manifest.txt) manifests show the same host, compiler, training rows, and raw-profile count. The [paired runner](screen-residual.ps1) checks both binary identities, pinned clip inputs, and canonical outputs; it warms both arms, alternates arm and clip order, and retains all pairs. Every one of the 102 encodes in both screens exited zero and produced its expected IVF hash.

| Candidate/control process CPU | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| Screen 1, 5 pairs | 1.004831× | 1.023310× | 1.035661× | 1.021189× |
| Screen 2, 10 pairs | 1.003055× | 1.008009× | 1.023376× | 1.011443× |
| Pooled, 15 pairs per clip | 1.003647× | 1.013084× | 1.027455× | 1.014681× |

The complete `screen-0{1,2}-{identity,raw,summary}.csv` files preserve each CPU/wall observation and hash. The candidate was slower in both screens, including on Ducks, so its source commit stays **out of `origin/main`**. This rejects this implementation's performance, not the exact-output invariant. The last direct production-PGO result remains **1.061826× pinned libvpx equal-clip CPU**, above the ≤1.05× target. No candidate/libvpx run is warranted for a slower candidate.
