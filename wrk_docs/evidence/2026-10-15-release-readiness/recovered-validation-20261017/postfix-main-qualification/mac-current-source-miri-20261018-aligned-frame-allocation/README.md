# Focused aligned-frame allocation Miri check

On exact product source `fb5c0c917038db6c640b3dbc7c2b84b567233333`, Apple Silicon Miri passed `frame::tests::alloc_aligned_widths_round_to_8` (1 passed, 0 failed, 0 ignored; 1,652 filtered out; 567.35 seconds). This allocates a 1923×1077 YUV 4:2:0 decoder-border frame and checks that the luma width rounds up to 1928.

This targeted test resolves the previous broad-filter timeout for this one allocation test. The frame and core Miri sweeps remain partial; the run does not exercise full-frame decoding or native NEON because Miri uses scalar `cfg(miri)` paths. See [`qualification.json`](qualification.json) and the exact-byte compressed log [`run.raw.txt.gz`](run.raw.txt.gz).
