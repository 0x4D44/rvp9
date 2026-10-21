# Windows native prerequisite qualification

Source: `feeb4cf87c65e4be5402b5661f6c8e3103bd304e`, fetched from public `0x4D44/rvp9` main into the existing clone's separate `beta-public/main` ref. This is baseline evidence, not final beta artifact qualification.

Host CRUCIBLE ran native x86_64 MSVC Rust 1.95.0. Build/check/test commands used `RUSTFLAGS=-C target-cpu=x86-64`; final packaging must instead clear both Rust flag variables and pass an explicit target. Existing origin `0x4D44/rvp9-2` and its staged historical documentation were untouched.

Observed successful gates:

- `cargo build --workspace --locked --release`: exit 0.
- `cargo fmt --all --check`: exit 0.
- `cargo clippy --workspace --all-targets --locked`: exit 0; existing warnings retained.
- `cargo test --workspace --locked`: exit 0; 3,944 passed, 72 ignored, 124 reported test targets.
- Official strict conformance, native/scalar dispatch and one/two threads: all four runs passed exactly 307 vectors and 4,816 frames, zero mismatch/skip.
- Encoder qualification: 32/32 rows, 60 frames per row; rvp9/libvpx/ffmpeg decoded outputs matched.
- Reverse reference decoder qualification: 24/24 rows, 12 frames per row; decoded outputs matched pinned libvpx.
- Native AddressSanitizer: core library 2,133 tests and decoder panic safety four tests passed under installed nightly 1.97.0-nightly (2026-05-01).

ASan commands used `RUSTFLAGS="-Zsanitizer=address -C target-cpu=x86-64"`, `--target x86_64-pc-windows-msvc`, a separate target directory, and process-local PATH prefixed with existing `C:\Program Files\LLVM\lib\clang\21\lib\windows`. The VS18 ASan DLL satisfied a small smoke but failed the full test executable with `STATUS_ENTRYPOINT_NOT_FOUND`; LLVM21 runtime succeeded. Both attempts are retained. No tools or dependencies were installed.

Raw PowerShell logs remain in registered scratch `C:\Users\marti\AppData\Local\Temp\deltic-scratch\beta-win-qualification-d0ad2683`. Copied logs are UTF-8 normalized; manifests retain stream/source/reference SHA256 provenance. `metadata.json` records executable/runtime hashes; `SHA256SUMS` hashes local evidence.

The baseline checkout exposes historical documentation path length/line-ending differences under Windows. Before final qualification, create a fresh lane with process-local Git longpaths enabled and autocrlf disabled. No baseline files were restored or discarded.

Instrumentation verification: setting PowerShell `$env:CARGO_ENCODED_RUSTFLAGS=$null` removes the variable (`Test-Path Env:CARGO_ENCODED_RUSTFLAGS=False`). A repeated decoder run passed four tests, and `llvm-readobj --coff-imports` confirms ASan runtime imports plus load/store report hooks, initialization, global registration and instrumented memory functions. See `asan-imports.txt` and `asan-confirm.log`. Installed cargo-fuzz 0.13.1 was discovered; no campaign launched.
