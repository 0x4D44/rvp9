# Focused frame-plane Miri check

On exact product source `fb5c0c917038db6c640b3dbc7c2b84b567233333`, the Apple Silicon Miri run passed `frame::tests::mutable_plane_views_stop_before_the_next_plane` (1 passed, 0 failed, 0 ignored; 1,652 filtered out). The test constructs 64×48 decoder-border buffers with V32 alignment for YUV 4:2:0, 4:2:2, 4:4:0 and 4:4:4, then verifies that the mutable Y and U data ranges do not overlap the following plane.

This is one focused core frame-layout test. It does not complete the frame or core Miri sweep, does not exercise full-frame decoding, and does not execute native NEON; Miri uses the scalar `cfg(miri)` paths. See [`qualification.json`](qualification.json) and the exact-byte compressed log [`run.raw.txt.gz`](run.raw.txt.gz).
