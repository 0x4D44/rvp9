# Frozen-profile AVX-512-off codegen screen — rejected

The integrated production-PGO encoder remains the best measured image. On Crucible, a codegen-only rebuild with AVX-512 disabled kept all three 60-frame output IVF hashes exact but used more process CPU on every clip. This was a five-pair diagnostic, not a production-PGO replacement.

## Controlled comparison

- Source: `335f8910`; no shipped-file difference from the measured integrated source.
- Host/toolchain: Crucible, AMD Ryzen 9 9950X3D, Rust 1.95.0.
- Frozen PGO profile: `D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s\.scratch\fresh-integrated-pgo\runs\20260927T111441Z-3232706\encoder.profdata`, SHA-256 `3f02af95fbeca067d6d94b233a353681bce4ed25c76930a94f0564895ac10335`.
- Control: integrated `rvp9enc.exe`, SHA-256 `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`, built with `-Ctarget-cpu=native -Cprofile-use=<frozen profile>`.
- Candidate: `rvp9enc.exe`, SHA-256 `b6b5d9987dab8b66815334938ff026a733963470d81c5f0fe4e5b993538e4437`, built with the same source and profile plus `-Ctarget-feature=-avx512f` using `cargo build --locked --release -p rvp9-cli --bin rvp9enc`. Its build completed cleanly in 42.46 seconds.
- Runner: `screen-avx512-off.ps1` was copied into the remote task worktree's `.scratch` and run from there. It adapts the existing exact-output paired runner, verifies binary/input hashes, warms both arms, alternates arm and clip order, and records process CPU. It ran five measured pairs per clip. Source script and raw CSVs are retained here.

| 60-frame clip | Candidate/control process CPU, geometric mean |
| --- | ---: |
| Ducks | 1.089808× |
| Tree | 1.044034× |
| Town | 1.072358× |

All candidate and control outputs matched each clip's canonical IVF SHA-256. The penalty is large and consistent enough to stop without a longer screen. This rejects a global AVX-512-off build flag, not narrower codegen changes. No shipped encoder code changed.

The integrated encoder's direct 60-pair comparison with pinned libvpx remains [1.057638× equal-clip CPU](../2026-09-27-integrated-pgo-direct-repeat/README.md), versus the 1.05× target.
