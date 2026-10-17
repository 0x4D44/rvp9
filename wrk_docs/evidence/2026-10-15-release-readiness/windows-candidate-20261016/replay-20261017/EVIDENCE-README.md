# Native Windows candidate replay evidence

Source474dfd99, Rust1.95.0, native Crucible x86_64. Fresh completed correctness and PGO only; no timing result is implied. Lead verification files record307/4816 official conformance checks per each of4modes, encoder32/reverse24,48independent reverse output sizes/hashes,3,167library tests (2ignored),24CLI tests, and canonical28-row/60-frame PGO with31raw profiles.

**Original bytes are in `raw-replay-evidence.tar.gz`.** `text-evidence-integrity.json` maps each original file hash to its normalized review-copy hash. Visible text copies normalize native encodings/line endings, trailing spaces and blank EOF lines so documentation whitespace checks can pass. All original runtime hashes, remote/download hashes and earlier integrity maps refer to archived originals. Extract originals into registered scratch for byte-level comparisons. Do not compare those hashes with normalized display copies. The archive contains compact logs/manifests/scripts, not generated executables, profiles or large media.

External binaries/profile remain at the registered paths in the PGO manifest and lead verification. The task-local target is a checked-new junction to registered external storage. Source and reference identities, command/exit files, documented shasum/stat adapters, numeric probes, ignored tests and independent native checks are retained.

The previous lost candidate artifacts and historical intake campaigns are separate. This fresh replay does not reconstruct them or qualify macOS, safety campaigns, quality or performance.
