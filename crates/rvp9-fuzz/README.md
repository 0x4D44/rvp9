# rvp9-fuzz

cargo-fuzz / libFuzzer harness for the rvp9-2 decoder. Implements
HLD §G9 — the parser fuzz target. See
`wrk_docs/2026.05.19 - HLD - decoder hardening.md` for the design
rationale (in particular the structured-Arbitrary decision; naive
byte mutation wastes ~99% of the fuzzer budget on inputs rejected
in the first 50 bytes of `parse_packet`).

## Targets

| Target              | Status            | Catches                          |
|---------------------|-------------------|----------------------------------|
| `fuzz_parse_packet` | **G9 (now)**      | parser panics / UB               |
| `fuzz_decode_frame` | **G9.1 (now)**    | reconstruction-path crashes      |
| `fuzz_full_stream`  | G9.2 (deferred)   | stateful (ref-map, FC) bugs      |

`fuzz_decode_frame` consumes raw packets and relies on valid encoder and
conformance seeds to reach the compressed-header, tile, mode, coefficient,
motion-compensation, transform, and loop-filter paths. `fuzz_full_stream`
remains deferred because it needs a stateful multi-packet corpus shape.

## One-time setup

```powershell
# Install cargo-fuzz (libFuzzer requires nightly Rust). Pin to
# 0.13.x — the cargo-fuzz CLI has had a couple of breaking flag
# renames between minor versions; locking the range keeps the
# README commands working across machines.
cargo install cargo-fuzz --version "^0.13"

# Seed corpus copy (one-time; corpus/ is gitignored). Safe to skip:
# libFuzzer will bootstrap its own corpus from the structured
# Arbitrary input shape if no seeds are present.
./scripts/setup-fuzz-corpus.ps1
```

### Windows-specific runtime path

cargo-fuzz links against `clang_rt.asan_dynamic-x86_64.dll` (the
AddressSanitizer runtime ships with the MSVC toolchain). The DLL is
not on the default `PATH`; a missing DLL can prevent the executable from
starting. Prepend the installed MSVC Hostx64 bin directory:

```powershell
# Adjust the version path for your installed VS edition.
$env:PATH = "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
```

The DLL must also export the symbols required by the chosen nightly. A
successful compile does not prove that the executable starts or that ASan
runs. Verify a native preflight and retain its exit/output before launching
a campaign. The current qualification found an incompatible installed
Windows runtime for the selected nightly; see the [readiness report](<../../wrk_docs/2026.10.15 - STATUS - final release readiness.md>).

On Linux / WSL / macOS the ASan runtime is statically linked by the
toolchain and no `PATH` tweak is needed.

## Running

The fuzz crate is NOT a member of the parent workspace (see the
"Layout notes" section below), so cargo-fuzz needs an explicit
`--fuzz-dir` pointer. Run from the repo root:

```powershell
# 30-second smoke (proves the target runs; useful in a hot loop).
cargo +nightly fuzz run --fuzz-dir crates/rvp9-fuzz fuzz_parse_packet -- -max_total_time=30 -dict=crates/rvp9-fuzz/fuzz.dict

# Reconstruction-path smoke; seed CORPUS with raw VP9 packet payloads.
cargo +nightly fuzz run --fuzz-dir crates/rvp9-fuzz fuzz_decode_frame CORPUS -- -max_total_time=30 -dict=crates/rvp9-fuzz/fuzz.dict

# 5-minute local regression smoke (historical HLD §G9 cadence).
cargo +nightly fuzz run --fuzz-dir crates/rvp9-fuzz fuzz_parse_packet -- -max_total_time=300 -dict=crates/rvp9-fuzz/fuzz.dict

# Optional 24-hour local bake (historical HLD §G9 recipe).
cargo +nightly fuzz run --fuzz-dir crates/rvp9-fuzz fuzz_parse_packet -- -max_total_time=86400 -dict=crates/rvp9-fuzz/fuzz.dict
```

Use a sanitizer supported by the installed Rust compiler. The installed
macOS nightly rejects `-Zsanitizer=undefined`; the former UBSan recipe does
not run on that toolchain. Default cargo-fuzz ASan checks instrumented memory
accesses and does not establish general undefined-behaviour or integer-overflow
coverage. Record the nightly identity, target, flags and actual runtime output.

### Smoke-run baseline (2026-05-21, Windows)

A 30-second smoke at this commit produced 55,109 executions, 1,314
coverage edges, 262 corpus entries discovered, and zero crash
artifacts. This is the floor — subsequent runs should not regress
below ~1,500 exec/s on a comparable host (the per-iter `Decoder::new`
construction dominates; G9-PERF tracks hoisting that out).

Crashes are written to `crates/rvp9-fuzz/artifacts/fuzz_parse_packet/`.

## Local regression checks

Release qualification uses local parser and full-frame campaigns. The older
HLD's proposed hosted PR/nightly lanes are historical planning; do not add
hosted CI workflows for the current release. Bound each command with
`deltic timeout`, close stdin, retain raw logs and require nonzero executions
with valid raw-packet reconstruction seeds. A startup failure or an empty
campaign is not a pass. Keep the approved billion-iteration fuzz and 72-hour
soak deferrals explicit; these smoke commands do not satisfy them.

## Layout notes (why not `[lints.workspace]`?)

The workspace lint table denies `print_stderr` / `print_stdout` and
warns on `missing_docs` for library crates. cargo-fuzz targets are
`#![no_main]` binaries and have synthetic entrypoints with no
attachable docs — inheriting the workspace lints generates spurious
warnings that obscure real fuzzer-found issues. Per HLD §G9, this
crate is fuzz infrastructure, not library code; it intentionally
opts out of the workspace lint inheritance.

## Crate layout (cargo-fuzz convention)

```
crates/rvp9-fuzz/
├── Cargo.toml
├── README.md
├── fuzz.dict                  # libFuzzer dictionary (committed)
├── fuzz_targets/
│   ├── fuzz_parse_packet.rs   # uncompressed-header parser target
│   └── fuzz_decode_frame.rs   # full frame reconstruction target
├── corpus/                    # GITIGNORED — populated by setup script
│   └── fuzz_parse_packet/
└── artifacts/                 # GITIGNORED — libFuzzer crash output
    └── fuzz_parse_packet/
```
