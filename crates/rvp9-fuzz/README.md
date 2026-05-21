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
| `fuzz_decode_frame` | G9.1 (deferred)   | reconstruction-path crashes      |
| `fuzz_full_stream`  | G9.2 (deferred)   | stateful (ref-map, FC) bugs      |

G9.1 / G9.2 require a structured-encoder good enough to round-trip
plausible-bit-exact frames through `parse_packet` → `decode_frame`.
That capability lands with Stage 4a.

## One-time setup

```powershell
# Install cargo-fuzz (libFuzzer requires nightly Rust). Pin to
# 0.13.x — the cargo-fuzz CLI has had a couple of breaking flag
# renames between minor versions; locking the range keeps the
# README commands working across machines.
cargo install cargo-fuzz --version "^0.13"

# Seed corpus copy (one-time; corpus/ is gitignored).
./scripts/setup-fuzz-corpus.ps1
```

## Running

cargo-fuzz expects to be run from the fuzz-target crate root:

```powershell
cd crates/rvp9-fuzz

# 5-minute smoke (matches CI cadence from HLD §G9).
cargo +nightly fuzz run fuzz_parse_packet -- -max_total_time=300 -dict=fuzz.dict

# 24-hour bake (matches nightly cadence from HLD §G9).
cargo +nightly fuzz run fuzz_parse_packet -- -max_total_time=86400 -dict=fuzz.dict

# UBSan pass (catches integer-overflow regressions media decoders
# accumulate). ASan is implicit with cargo-fuzz. Currently an
# optional local pass; HLD §G9 calls for promotion to a required
# CI lane — tracked separately as G9-CI follow-up.
$env:RUSTFLAGS = '-Zsanitizer=undefined'
cargo +nightly fuzz run fuzz_parse_packet
```

bash equivalent (Linux / WSL / macOS CI):

```bash
RUSTFLAGS="-Zsanitizer=undefined" cargo +nightly fuzz run fuzz_parse_packet
```

Crashes are written to `crates/rvp9-fuzz/artifacts/fuzz_parse_packet/`.

## CI integration

There is no GitHub Actions workflow wired up yet. HLD §G9 calls for
a 5-minute PR-blocking lane (`-max_total_time=300`) plus a nightly
24-hour bake; both are tracked as G9-CI follow-ups separate from
the parser-target landing in this commit.

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
│   └── fuzz_parse_packet.rs   # target body
├── corpus/                    # GITIGNORED — populated by setup script
│   └── fuzz_parse_packet/
└── artifacts/                 # GITIGNORED — libFuzzer crash output
    └── fuzz_parse_packet/
```
