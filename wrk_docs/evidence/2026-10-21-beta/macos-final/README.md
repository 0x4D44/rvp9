# macOS beta artifact qualification in progress

Compilation source15bf7162c9fb92a5165fab652460bd42da03a6e6, native Apple M5 Max, portable aarch64 release with both Rust flag variables explicitly empty. Packaged archive extraction and both executable versions/hashes passed. Four-mode official conformance passed307vectors/4816frames each; adapted actual-artifact encoder32 and reverse24 checks passed. Runner diffs preserve every acceptance assertion and only replace source builds/binary paths with the extracted executables.

The initial quality run is INVALID: RVP9-BUG-FLU-00058 inflated libvpx bitrate by mistaking its1000Hz timestamp clock for frame cadence. All apparent PASS rows are rejected, including any completed prefix. The stopped attempt's raw output is retained; raw streams remain in owned registered scratch for diagnosis. A fresh full matrix must use the integrated repaired benchmark. Timing remains unqualified; no Mac timing acceptance is claimed.

Exact beta workspace tests and source documentation checks are separately tracked in final Mac registered scratch. Native ASan/Miri/fuzz and initial full workspace baseline live in neighboring evidence directories with their source boundaries disclosed. No tag or publication has occurred.
