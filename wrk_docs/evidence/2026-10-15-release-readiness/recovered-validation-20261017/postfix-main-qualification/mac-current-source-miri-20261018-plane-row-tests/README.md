# Focused frame plane and row Miri tests

On exact product source `fb5c0c917038db6c640b3dbc7c2b84b567233333`, the Apple Silicon Miri filter `frame::tests::plane_` selected `plane_mut_fill_does_not_touch_border` and `plane_row_round_trip`: 2 passed, 0 failed, 0 ignored.

This adds two focused `rvp9-core` frame tests. It does not complete the frame or core Miri sweep, establish full-frame decoder coverage, or exercise native NEON; Miri uses scalar `cfg(miri)` paths. The raw output is stored byte-for-byte in [`run.raw.txt.gz`](run.raw.txt.gz), with both compressed and decompressed hashes in [`qualification.json`](qualification.json).
