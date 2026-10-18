# Awkward-stride frame-border Miri test

The exact-source isolated Miri filter `frame::tests::extend_borders_awkward_stride` passed: 1 passed, 0 failed, 0 ignored, with 1,652 filtered out. The test allocates a 1923×1077 frame, checks alignment padding and extended borders, and completed in 537.79 seconds under a 1,800-second timeout.

The test source, `rvp9-core` tree, and `Cargo.lock` match product source `fb5c0c917038db6c640b3dbc7c2b84b567233333`. This test had timed out when reached late in the broad `frame::` Miri filter; running it alone completed the case. Along with the four focused border filters, it raises the fully completed selected-filter subtotal from 751 to 752 passed, 0 failed, 4 existing ignores. The broader frame sweep remains partial.

Miri exercises scalar fallbacks and does not execute native NEON. This one large-frame test does not complete frame/core, dependent-library, sanitizer, fuzz, or release safety qualification. The exact raw output is gzip-preserved in `awkward-stride.raw.txt.gz`; `qualification.json` records the original byte count and SHA-256.
