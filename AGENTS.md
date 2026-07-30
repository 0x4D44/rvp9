# Repository Guidelines

## Project Structure & Module Organization

This is a Rust 2024 Cargo workspace for a VP9 encoder/decoder port. Core crates live under `crates/`: `rvp9-core`, `rvp9-io`, `rvp9-decoder`, `rvp9-encoder`, `rvp9-cli`, `rvp9-bench`, and `rvp9-harness`. `crates/rvp9-fuzz` is intentionally outside the workspace and should be run with cargo-fuzz. Use `tests/corpus/` and `tests/conformance/` for clips and vectors, `bench/results/` for benchmark outputs, `scripts/` for automation, `third_party/libvpx/` for the pinned reference implementation, and `notes/` or `wrk_docs/` for design context.

## Build, Test, and Development Commands

- `cargo build --workspace`: build all workspace crates.
- `cargo test --workspace`: run Rust unit and integration tests.
- `cargo test -p rvp9-encoder <test_name>`: run a focused crate test.
- `cargo fmt --all --check`: verify rustfmt formatting.
- `cargo clippy --workspace --all-targets`: run workspace lint checks.
- `./scripts/build-libvpx.sh`: build the libvpx reference (use MSYS2 on Windows).
- `./scripts/fetch-vectors.sh` and `./scripts/fetch-corpus.{sh,ps1}`: fetch test data.
- `cargo run --release --bin conformance -- --decoder build/libvpx/vpxdec.exe --vectors-dir tests/conformance/vectors --limit 5`: sanity-check decoder conformance.

Wrap long-running validation with `mdtimeout` where practical.

## Coding Style & Naming Conventions

Use rustfmt defaults, four-space indentation, and Rust naming conventions: `snake_case` for functions/modules, `PascalCase` for types, and `SCREAMING_SNAKE_CASE` for constants. Library crates inherit workspace lints. Keep public APIs documented enough to satisfy `missing_docs = "warn"`. Do not print from library code; return errors or use logging. Every unsafe block must include a `// SAFETY:` comment.

## Testing Guidelines

Place integration tests in each crate's `tests/` directory. Add focused regression tests for changed codec behavior, and use harness or differential tests when behavior must match libvpx. Follow existing naming patterns such as `cli_roundtrip`, `panic_safety`, `w_*`, `e*`, and `diag_*` when extending related suites. Run the smallest meaningful test first, then escalate to workspace checks when the change crosses crates.

## Work Tracking

Defects live in `bugs/`, one markdown file per bug; accepted enhancements and durable behavioural specs live in `reqs/`, one markdown file per requirement. Read `bugs/README.md` before raising, fixing, or verifying a tracked bug, and read `reqs/README.md` before capturing or building a requirement. Mint records with `deltic bugs new` and `deltic reqs new`; never allocate IDs by hand. Use the fleet `bug-tracking`, `reqs-capture`, and `reqs-loop` skills when their trigger phrases apply.

Ordinary development test failures do not automatically become `bugs/` entries; only separate durable defects should be promoted. A tracked bug fix moves the bug to `Fixed` with regression evidence, and a different verifier closes it. RVP9 requirements are manual/heavy by repo policy for now: manual/heavy implementations move `Accepted -> Implemented` when the work lands and never self-promote to `Satisfied`. Keep `Flow: light` disabled until this workspace has a checked-in integration policy and a proven workspace-specific oracle.

## Commit & Pull Request Guidelines

Recent history uses scoped prefixes such as `fix(enc):`, `perf(dec):`, `bench:`, `docs(enc):`, `scripts:`, and `journal:`. Keep subjects concise and outcome-focused. PRs should describe the behavior changed, list validation commands and results, link the relevant plan or issue, and include benchmark or BD-rate data for performance or quality changes.

## Security & Configuration Tips

Do not commit secrets or local credentials. Keep generated corpora and bulky profiling artifacts out of normal commits unless the change is explicitly about fixtures or benchmark baselines. Release builds use `.cargo/config.toml` with `target-cpu=native`, so binaries are tuned for the build host and may not be portable.
