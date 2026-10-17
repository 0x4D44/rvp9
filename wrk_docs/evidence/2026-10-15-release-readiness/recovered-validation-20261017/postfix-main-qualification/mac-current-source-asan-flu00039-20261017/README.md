# Exact-candidate Apple Silicon ASan reproduction of FLU-00039

The focused native Apple Silicon test launched on the current candidate product source and AddressSanitizer aborted it. One test started and none completed. ASan reported a 4-byte heap-buffer-overflow read exactly at the end of a 56-byte allocation. The stack runs through load_four_repeated, the width-four I8MM horizontal kernel, its safe NEON wrapper, and the existing tight-slice test.

The runner checkout is b1c87dbb8cc8768a486a81fb2d50c56935106f11, based on current origin/main 2b1333cc. Its convolve8_neon.rs blob matches candidate product source fb5c0c917038db6c640b3dbc7c2b84b567233333. This reproduces the existing width-four issue tracked by RVP9-BUG-FLU-00039; the bug remains Open and no source or test was changed. The separately audited width-eight overread remains untested at runtime.

The exact command, host and toolchain identities, source blob, ASan summary, and raw-log SHA-256 are in qualification.json. The complete sanitizer report is preserved in run.raw.txt.
