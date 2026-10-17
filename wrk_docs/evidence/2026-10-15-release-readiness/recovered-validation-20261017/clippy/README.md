# Fresh documented Clippy check

Source at start: `df2b1589161acbaf73c1878007aee09a92bea8c3`; exact command/config/toolchain/environment and exit in run.json. Sequential run after public docs, separate registered doc-target, jobs2, offline/locked, stdin closed,1800-second outer bound.

Plain `cargo clippy --workspace --all-targets --offline --locked` exited0, with two individual warnings: encoder lib-test match_like_matches_macro at spatial_encoder.rs:1082 and harness dump_frame_diff bin-test items_after_test_module. Each has raw line/source location/category in warnings.json; counts match Cargo summaries. No -Dwarnings, suppression, fix, retry, install, staging or commit. Successful configured lint check is not a warning-free check.

Postrun Rust/manifests/scripts diff from ebca is empty after explicitly excluding the concurrent lead-authorized fuzz README reproduction-doc correction, preserved separately. This documentation exception does not change lint inputs. Files are final; integrity.json hashes every other compact file.

Original compact files are retained byte for byte in raw-clippy-evidence.tar.gz. Visible copies normalize trailing whitespace and blank EOF (including space-only context lines in the captured git diff). Original/runtime/integrity hashes refer to archived bytes; text-evidence-integrity.json records original/review hashes. No diagnostic or measurement changed.
