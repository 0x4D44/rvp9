# Single-invocation candidate package graph

Stable Cargo1.95 supports repeated package selection (installed docs record stabilization in1.90). The one offline/locked default five-package command exited0 and verified all five production packages, including CLI. Exact command/source/toolchain/config/env and terminal raw output are preserved.

Cargo created its own target/package/tmp-registry. Raw output explicitly shows sibling unpacking there. For all five archives, finalized archive bytes equal staging-registry bytes and index checksum. Every archived src/ file and build.rs matches candidate workspace source. The CLI packaged lock checksums for core/decoder/encoder/io equal these fresh archives, even though its declared registry source retains the crates.io URL. These checks distinguish freshly assembled candidate siblings from cached published alpha.4. graph-verification.json records each file/hash and checksum equality.

This qualifies default production package graph assembly/build verification under native Cargo staging. It does not publish a new version, establish a future versioned registry graph, run archive tests or repair FLU-00038 stripped dev dependencies. The earlier individual-invocation cached-sibling CLI failure remains accurate historical evidence for that separate command, but is not a blocker to this successful native multi-package staging path.

All generated archives/targets remain registered scratch, not committed evidence. No patches, operator config overrides, unstable flags, no-verify, allow-dirty, dependencies or version changes. No staging/commit. Files are final; integrity.json hashes each other compact artifact.
