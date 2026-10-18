# Frame-module Miri remainder

The exact-source Apple Silicon Miri filter selected 81 tests and passed all 81, with zero failures or ignores, in 486.94 seconds. The substring filter selected 53 `frame::tests` cases and also selected 28 `bitstream` tests (`frame` and `superframe`); those 28 were already included in the earlier 278-test bitstream filter.

The 53 root frame cases overlap 41 previously completed focused frame cases, leaving 12 new unique frame-test passes. This raises the fully completed selected-filter subtotal from 752 to 764 passed, zero failed, with four existing ignores. Those 12 new cases are recorded in `qualification.json`.

The run skipped `alloc_aligned_widths_round_to_8` and `extend_borders_awkward_stride`; each has a separate exact-source Miri pass ([aligned-width allocation](../mac-current-source-miri-20261018-aligned-frame-allocation/qualification.json), [awkward-stride borders](../mac-current-source-miri-20261018-awkward-frame-stride/qualification.json)). `frame.rs` contains 55 test functions: the 53 cases in this run plus those two separately passing cases cover all 55. The broader core sweep and entropy remain partial. Miri uses scalar fallbacks and does not exercise native NEON, dependent-library, parser/full-frame production, sanitizer, fuzz, or release safety paths.

The raw run is gzip-preserved. `qualification.json`, `source-identity.txt`, and the hash manifest retain the command, source/toolchain identity, exact counts, and artifact digests.
