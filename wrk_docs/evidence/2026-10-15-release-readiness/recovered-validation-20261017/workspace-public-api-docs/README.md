# Full-workspace public documentation check

Source at invocation: `8f0b58acde72193278d0b3abf560008008216c7b`; Rust/manifests/scripts diff from ebca is empty except explicitly recorded lead-authorized fuzz README reproduction-doc correction. Runbook command `cargo doc --workspace --no-deps --offline --locked` exited0 with jobs2, closed stdin and1800-second bound, sequentially after package graph verification. Exact source/toolchain/config/env/command and original output are retained.

Generated HTML stays registered scratch doc-target. 143 individual diagnostics were parsed; per-crate/category counts and each raw line/source/message are in warnings.json and exactly match Cargo summaries. Cargo may replay cached diagnostics for unchanged documentation units; no claim that every rustdoc invocation recompiled from scratch. This is full workspace scope, not substituted five-shipped scope.

Success establishes documentation generation, not warning-free docs, public API compatibility or semantic documentation accuracy. No suppression, fix, source edit, dependency installation, staging or commit. Files are final; integrity.json hashes all other compact evidence.
