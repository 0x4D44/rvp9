# Fresh offline package checks

Source: `33378791eace3bda2f927e629c30b805b54b56e1`; shipped delta from `ebca64b0` is empty. Rust1.95.0 / LLVM22.1.2, native aarch64-apple-darwin. Exact commands, working directories, times, environment, exits and archive hashes are in `results.json`. All commands used offline/locked resolution, two jobs and closed stdin.

| Check | Exit |
| --- | ---: |
| package-core | 0 |
| package-io | 0 |
| package-decoder | 0 |
| package-encoder | 0 |
| package-cli | 101 |
| archive-build-core | 0 |
| archive-build-io | 0 |
| archive-build-decoder | 0 |
| archive-build-encoder | 0 |
| archive-test-decoder | 101 |
| archive-test-encoder | 101 |

Core/io/decoder/encoder default package verification and unchanged extracted production library builds passed. CLI assembly created an archive but default verification failed: candidate CLI imports coded_dimensions_supported, which is private in cached registry encoder alpha.4. No retry, override or dependency/version change was attempted. This is an existing-version graph blocker/limitation, not a newly minted defect.

Decoder panic_safety and encoder quality_gate archive test commands both failed with unresolved stripped sibling dev dependencies, reproducing FLU-00038. Failures are preserved; no fix or broad test suite was attempted.

Package siblings resolve cached registry alpha.4, not the other newly assembled candidate archives. These checks do not qualify a newly versioned release dependency graph or full candidate library combination. Package verification does not run tests.

Every original regular file in each of the four externally extracted library archives was SHA-256 compared against its archive after checks; no mismatches. The failed CLI archive remains under Cargo target/package/tmp-crate; its hash/file list/normalized manifest are preserved, with no finalized CLI archive or separate CLI build pass claimed. Normalized manifests and archive file lists are preserved. All .crate archives remain in registered scratch; four earlier evidence-directory copies are locally ignored generated artifacts and excluded from staging.

The package source tree used target-cpu=native from repo Cargo config. Extracted commands ran outside the workspace with a separate target and no inherited Rust flag variables. Exact flags distinction is recorded in results.json.

Driver40682 completed. No source edits, staging, commits, publication, installs or external-service writes. Raw logs are original bytes; integrity.json covers all compact files except itself and generated archives/.
