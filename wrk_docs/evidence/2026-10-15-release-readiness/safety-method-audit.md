# Candidate safety audit (2026-10-15)

Read-only source/tool inspection; no builds, fuzzing, Miri, soak, dependency installation, or source changes performed. Initial source HEAD was `1a84f60a24e8d82e8e6454743fbb80521280df0e`; latest observed HEAD is `23162998476b2b410f10e4502793e9f06c124a56`. Seal the final candidate commit and binary hashes before executing this plan. Commands below are proposed, not passing evidence.

## Exact-current source execution update (2026-10-17)

The read-only scope above describes the initial inventory. A follow-up exact-source Apple Silicon Miri run on candidate fb5c0c917038db6c640b3dbc7c2b84b567233333 passed 23 rvp9-core tile_threading tests, with zero failures/ignores and 1,630 filtered, on installed nightly 1.98.0. It is one core subset only; no dependent-library, parser/full-frame, native NEON, ASan or fuzz coverage follows. The dependent-io Miri compile remains blocked by FLU-00037. [Command, raw output, exit status and toolchain](recovered-validation-20261017/postfix-main-qualification/mac-current-source-miri-20261017/tile-threading/qualification.json).

## Observed prerequisites and gaps

- Host: `aarch64-apple-darwin`. Pinned normal toolchain 1.95.0 exists. Installed `nightly` reports rustc 1.98.0-nightly `bd08c9e71` (2026-06-25), LLVM 22.1.7; Miri 0.1.0 at that revision, rust-src, llvm-tools, rust-std, cargo are installed. No automatic toolchain update is required for this plan.
- `cargo fuzz --version` fails: cargo-fuzz is not available in the current PATH. No cargo-fuzz binary was found in `/Users/md/.cargo/bin`. Bounded fuzz is blocked unless an existing approved tool location is supplied; do not install it under this task's constraint. Historical Aug20 isolated cargo-fuzz 0.13.2 tools are not evidence of current availability.
- `build/libvpx/vpxdec` and `target/release/soak` exist. Their presence does not prove their source, build flags, or hashes. The lead's current release build can establish soak provenance without a competing build.
- Installed nightly target specifications explicitly list address sanitizer support for both `aarch64-apple-darwin` and `x86_64-pc-windows-msvc`. This is compiler capability, not a passing sanitizer run. Windows still requires its own installed toolchain/runtime and native x86 hardware.
- Static search found 1,043 lines matching unsafe blocks/functions/impls/traits/extern across 100 production-library files. This is a search inventory, not a reviewed-block count; comments and declarations are included. Major seams include core SIMD/frame/tile-threading, decoder tile/context, and encoder frame loop/reference scaling/denoiser/pre-analysis. Do not claim a complete current unsafe audit from Clippy safety-comment checks alone.

## Exact-source invariants (inspected, not runtime assurances)

- `crates/rvp9-fuzz` is a separate workspace with two targets. `fuzz_parse_packet` uses structured Arbitrary inputs and does not reconstruct frames. `fuzz_decode_frame` accepts raw VP9 packets, ignores empty or >16 MiB inputs, creates a fresh one-thread decoder for each input, parses and decodes contained frames until error. Neither target exercises a persistent multi-packet decoder stream; `fuzz_full_stream` remains absent/deferred.
- Decoder fuzz seeds must be raw IVF packet payloads, not entire IVF/WebM files. Existing `scripts/setup-fuzz-corpus.ps1` copies containers for the parser corpus and is not a reconstruction-seed extractor. Extract valid keyframe packet payloads from pinned official IVF vectors into task scratch; preserve source vector hashes and extraction counts. Inter-only packets lack previous references in this target.
- `scripts/run_miri.sh` covers only transforms, intra_pred, quant and dsp. It omits tile_threading, frame, mv_search, mode_decision, parser/entropy/inter/loopfilter and encoder/decoder/io library tests. Current code routes several direct SIMD boundaries to scalar under `cfg(miri)`; native SIMD test modules are excluded under Miri. There are explicit Miri-ignored tests in DSP dispatch, compressed-header fixture handling and loopfilter. Record exclusions individually.
- Soak explicitly sets CPU speed 0 / Deadline::Good, CBR, AQ off, denoiser off, one temporal layer; sources are 256x256 crops plus a synthetic fallback. It therefore supplies generic encode/decode smoke, not production CPU6–8/denoiser/spatial coverage. The host's non-Windows RSS function always returns zero (`soak.rs:337`), so it cannot establish leak freedom on macOS. `errors=0` alone is insufficient: require positive completed iterations, decoded frames and encoded bytes.

## Minimal applicable execution plan, after the current cargo jobs finish

Use existing task scratch for logs/targets, capture command, exit code, toolchain, target, commit, binary hashes and passed/failed/ignored counts. Run serially; use harness-tracked background execution for long checks. Do not change tests or ignore new failures. Leave `RVP9_CROSS_DECODER_GATE` unset for Miri (one encoder unit test otherwise spawns native decoders).

### Scalar Miri

Full library sweep, including all source modules, rather than treating the four-module helper as complete:

```sh
env CARGO_TARGET_DIR="$PWD/.scratch/miri-target" RVP9_FORCE_SCALAR=1 MIRIFLAGS="-Zmiri-disable-isolation" deltic timeout 86400 cargo +nightly miri test --locked -p rvp9-core -p rvp9-io -p rvp9-decoder -p rvp9-encoder --lib -- --test-threads=1 </dev/null
```

Filesystem/environment access is needed by existing fixture tests. This command may expose unsupported operations or a toolchain/source mismatch; an unsupported operation is a coverage gap, not a passing test. If bounded runs need splitting, core modules are bitstream, block, coef_ctx, cost, dsp, entropy, frame, inter, intra_pred, loopfilter, mode_decision, mode_info, mv_search, quant, tile_threading, transforms, trace_flags and bit_depth_scaffold_tests. Run every listed module plus other libraries; reconcile counts against an unfiltered `--list`. Existing explicitly ignored tests remain disclosed. Unit-test completion proves the tested non-SIMD surface, not every possible path.

### Native ASan (includes SIMD differential tests)

```sh
env CARGO_TARGET_DIR="$PWD/.scratch/asan-target" RUSTFLAGS="-Zsanitizer=address -Ctarget-cpu=native" RUSTDOCFLAGS="-Zsanitizer=address" deltic timeout 14400 cargo +nightly test -Zbuild-std --target aarch64-apple-darwin --locked -p rvp9-core -p rvp9-io -p rvp9-decoder -p rvp9-encoder --lib -- --test-threads=1 </dev/null
```

Keep forced-scalar and SIMD-disabling diagnostic variables unset. Rebuilding std uses already installed rust-src; dependency/cache resolution may still be a prerequisite. Capture native feature detection and verify nonzero `dsp::aarch64` tests plus transform/motion/partition native tests in the log. ASan does not establish complete instruction-level coverage of inline assembly. Windows x86 SIMD requires a separate native `x86_64-pc-windows-msvc` run; Rosetta/cross-compilation cannot qualify AVX2. Do not transfer arm64 results to x86.

### Bounded ASan fuzz (blocked on existing cargo-fuzz executable)

With approved cargo-fuzz on PATH, separate fresh scratch corpus/artifact directories, and valid raw reconstruction seeds:

```sh
deltic timeout 600 cargo +nightly fuzz run --fuzz-dir crates/rvp9-fuzz fuzz_parse_packet .scratch/fuzz-parser-corpus -- -max_total_time=300 -dict=crates/rvp9-fuzz/fuzz.dict -artifact_prefix=.scratch/fuzz-parser-artifacts/ </dev/null
deltic timeout 600 cargo +nightly fuzz run --fuzz-dir crates/rvp9-fuzz fuzz_decode_frame .scratch/fuzz-decode-corpus -- -max_total_time=300 -dict=crates/rvp9-fuzz/fuzz.dict -artifact_prefix=.scratch/fuzz-decode-artifacts/ </dev/null
```

Use cargo-fuzz's default ASan; record the actual sanitizer/build command and ensure native dispatch is enabled. Compilation can consume the outer timeout; a timeout before a positive completed campaign is not evidence. Require nonzero executions, bounded completion, no crash artifact or sanitizer report, and coverage statistics. Neither campaign replaces native SIMD tests or stateful multi-packet tests.

### Nonzero bounded soak

After the lead's current build establishes the exact-source `target/release/soak`:

```sh
deltic timeout 180 target/release/soak --duration-secs 60 --frames-per-iter 30 --report-every-secs 10 --cross-decode --vpxdec "$PWD/build/libvpx/vpxdec" </dev/null
```

Require exit 0, final completion marker, iterations >0, decoded frames >0, encoded bytes >0 and no SOAK FAILURE. Record source list and counts. The 180-second outer cap can stop a slow iteration; report that as incomplete rather than loosening the acceptance claim. This run gives CPU0 generic smoke only. Current production and spatial quality/cross-decoder matrices provide the authorized CPU6–8 and denoiser/spatial bounded coverage. A requirement for a production-mode soak harness is a separate implementation/scope decision, not satisfied by relabelling this binary.

## Historical evidence only

`wrk_journals/2026.08.20 - JRN - Stage 5 safety qualification.md` records tile-threading Miri 23/23 (Unit1255), motion-search 71/71 (1256), mode-decision replayed sections (1257), parser ASan 1,192,692 executions/301 s and 8,040-frame soak (1258), and full-frame ASan 48,528 executions/301 s (1262). Units1262–1264 describe a review and accepted hardening; they do not supply a current complete unsafe-block inventory/review. The claims are historical and must not be carried to the current source without fresh results or explicit delta review.

PLAN.md:836–843 explicitly defers billion-iteration fuzz and the 72-hour monitored Intel soak. Keep those deferred; do not claim leak/drift assurance from zero-valued macOS RSS or bounded smoke. No new bug fixes or overlap with the bug owner are proposed here.
