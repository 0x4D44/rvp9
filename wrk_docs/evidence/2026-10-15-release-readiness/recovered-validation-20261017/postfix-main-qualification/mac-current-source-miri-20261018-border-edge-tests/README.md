# Focused frame-border Miri tests

On the exact candidate product source, four small `Yv12Buffer` border tests completed under Apple Silicon Miri: constant fill, corner pixels, edge rows and columns, and independent planes. Each command selected exactly one test and passed with zero failures or ignores. The test source, `rvp9-core` tree and `Cargo.lock` match product source `fb5c0c917038db6c640b3dbc7c2b84b567233333`.

These are four new completed cases beyond the prior 747 completed selected-filter passes, bringing that subtotal to 751 passed, 0 failed and 4 existing ignores. The broad frame sweep remains partial: it timed out earlier on `extend_borders_awkward_stride`. Miri exercises scalar fallbacks and does not cover native NEON. This evidence does not complete frame/core Miri, dependent-library Miri, sanitizer, fuzz or release safety qualification.

`commands.txt`, gzip-compressed exact raw outputs, exit records, source identity, toolchain/host identity, `qualification.json`, and `artifact-hashes.sha256` preserve the run. The JSON records the uncompressed byte lengths and SHA-256 values. The registered scratch target is outside the repository and will be removed by Deltic integration.
