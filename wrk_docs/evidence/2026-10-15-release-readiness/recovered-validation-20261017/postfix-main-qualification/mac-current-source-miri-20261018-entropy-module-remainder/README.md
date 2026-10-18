# Entropy-module Miri remainder

The exact-source Apple Silicon Miri filter selected and passed 142 entropy tests with zero failures or ignores in 509.92 seconds. It skipped only `entropy::prob_diff_writer::tests::map_table_inverse_of_inv_map_table`, which already passed in the separate focused run ([qualification record](../mac-current-source-miri-20261017/entropy-prob-diff-focused/qualification.json)). Together, the 142-test run and that separate pass cover all 143 `#[test]` functions in `crates/rvp9-core/src/entropy/`.

The run completed three entropy tests not present in the earlier broad/focused raw passes. The earlier broad run's 138 passing cases and two focused unique cases are included once in the 143-case module total. This raises the completed selected-filter subtotal from 764 to 907 passed, zero failed, with four existing ignores. The full core sweep remains partial.

Miri uses scalar fallbacks and does not exercise native NEON, dependent-library, parser/full-frame production, sanitizer, fuzz, or soak paths. The gzip-preserved raw run, command, source/toolchain identity, test names, overlap accounting, and artifact hashes are recorded here.
