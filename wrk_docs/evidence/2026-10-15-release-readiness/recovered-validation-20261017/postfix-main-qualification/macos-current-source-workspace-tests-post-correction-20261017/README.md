# Post-correction macOS workspace tests

The full exact-candidate Rust workspace release suite passed on macOS Apple Silicon at revision `fcc837f50219ddfc00eb7fc349c4626ea5b00fc1`, which retains production source `fb5c0c917038db6c640b3dbc7c2b84b567233333`. The command passed 3,579 tests, failed none, and ignored 64 across 124 test-result summaries.

This rerun includes the corrected `soak_cli` fixture, which now reaches its missing-`vpxdec` preflight, and the previously transient harness stderr timeout test; both passed. The locked offline run used Rust/Cargo 1.95.0 with stdin closed. The complete raw output is losslessly preserved as `workspace-no-fail-fast.raw.txt.gz`; hashes are in `qualification.json`.

On the integrated documentation revision `890dc491dce76e7e4db1f640261ffd015dde8615`, the touched `rvp9-bench` crate also passed `cargo clippy --all-targets --locked --offline` with zero warnings. The code and test files are unchanged from the full-suite run.

The 64 ignored tests remain outside this result. This is a Mac workspace-test pass, not a Windows suite, production quality/performance/latency, or native/scalar conformance result.
