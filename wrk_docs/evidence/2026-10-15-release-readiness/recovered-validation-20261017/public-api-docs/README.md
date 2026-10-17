# Fresh public API documentation check

Source: `df2b1589161acbaf73c1878007aee09a92bea8c3`, with empty shipped source diff from `ebca64b0`. Exact command, toolchain/config/environment and terminal exit are in run.json; original output is cargo-doc.raw.txt.

The five-shipped-crate offline/locked/no-deps cargo doc command exited0. Generated HTML remains only in registered scratch doc-target. This establishes documentation generation, not warning-free documentation, API compatibility, semantic API review or packaging readiness.

141 individual warnings: core79, encoder45, decoder14, io2, cli1. Categories: broken intra-doc links81, private-item links55, redundant explicit links4, invalid HTML1. Each warning has original raw line, message, crate, category and source location in warnings.json. Individual per-crate counts exactly match Cargo generated-warning summaries; summaries were excluded from diagnostic counts.

No warnings were suppressed; no source/doc fixes or stronger warning-deny flags were used. No dependencies installed, staging or commit performed. Files in this directory are final; integrity.json hashes every other compact file.
