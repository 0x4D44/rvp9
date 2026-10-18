# Full `rvp9-core --lib` Miri suite

The exact-source Apple Silicon Miri library test suite completed in 3,366.34 seconds: 1,645 passed, 0 failed, 5 ignored, and 3 filtered by the command. The 3 filtered cases each have a separate exact-source Miri pass: aligned-width frame allocation, awkward-stride border extension, and the entropy map-table inverse. The combined runs cover all 1,648 nonignored library tests; the 5 existing ignored cases are listed in `ignored-tests.txt`.

The library suite contained 1,653 cases in total (1,645 passed + 5 ignored + 3 filtered). `rvp9-core/tests/frame_alloc.rs` is a separate integration-test target with three tests; it is not covered by this `--lib` command and remains pending a guarded Miri run.

The run used `MIRIFLAGS=-Zmiri-disable-isolation` for bitstream fixture access and serial tests. Miri uses scalar paths and does not exercise native NEON. Dependent-library Miri remains blocked by FLU37; this is not sanitizer, parser/full-frame production, fuzz, or soak qualification.

The exact command, source/toolchain identity, raw output, filtered-case dispositions, ignored-test names, and hashes are preserved in this bundle.
