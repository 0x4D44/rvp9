# Current candidate product-source identity

At 2026-10-18T01:52:33Z, `origin/main` resolved to `836361611ee2a9d8d369032d641409c98bff638e`. Git object IDs were compared against product candidate `fb5c0c917038db6c640b3dbc7c2b84b567233333` for the seven workspace crate `src` trees, the external fuzz-target tree, all eight crate manifests, root `Cargo.toml` and `Cargo.lock`, `.cargo/config.toml`, and the complete `scripts` tree. Every checked object ID matched.

`git diff --name-status fb5c0c9..8363616 -- crates Cargo.toml Cargo.lock .cargo scripts` found only `crates/rvp9-bench/tests/soak_cli.rs` (test fixture correction) and `crates/rvp9-fuzz/README.md` (documentation). No shipped Rust source, package manifest, dependency lock, build target configuration, or script differs from the candidate. See `qualification.json` for full tree/blob IDs and the exact limits.

This identity check supports carrying forward evidence tied to product source `fb5c0c91` for the integrated revision `8363616`. It does not prove binary equality, rerun platform tests, or satisfy remaining release gates. Pinned libvpx revision identity and binary/artifact hashes remain in their respective qualification records. No tests were run for this source-identity check.
