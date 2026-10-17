diff --git a/crates/rvp9-fuzz/README.md b/crates/rvp9-fuzz/README.md
index b4df80ab..cb19e3b1 100644
--- a/crates/rvp9-fuzz/README.md
+++ b/crates/rvp9-fuzz/README.md
@@ -39,15 +39,20 @@ cargo install cargo-fuzz --version "^0.13"

 cargo-fuzz links against `clang_rt.asan_dynamic-x86_64.dll` (the
 AddressSanitizer runtime ships with the MSVC toolchain). The DLL is
-not on the default `PATH`, so the fuzzer exits with
-`STATUS_DLL_NOT_FOUND (0xc0000135)` unless you prepend the MSVC
-Hostx64 bin directory:
+not on the default `PATH`; a missing DLL can prevent the executable from
+starting. Prepend the installed MSVC Hostx64 bin directory:

 ```powershell
 # Adjust the version path for your installed VS edition.
 $env:PATH = "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
 ```

+The DLL must also export the symbols required by the chosen nightly. A
+successful compile does not prove that the executable starts or that ASan
+runs. Verify a native preflight and retain its exit/output before launching
+a campaign. The current qualification found an incompatible installed
+Windows runtime for the selected nightly; see the [readiness report](<../../wrk_docs/2026.10.15 - STATUS - final release readiness.md>).
+
 On Linux / WSL / macOS the ASan runtime is statically linked by the
 toolchain and no `PATH` tweak is needed.

@@ -64,26 +69,18 @@ cargo +nightly fuzz run --fuzz-dir crates/rvp9-fuzz fuzz_parse_packet -- -max_to
 # Reconstruction-path smoke; seed CORPUS with raw VP9 packet payloads.
 cargo +nightly fuzz run --fuzz-dir crates/rvp9-fuzz fuzz_decode_frame CORPUS -- -max_total_time=30 -dict=crates/rvp9-fuzz/fuzz.dict

-# 5-minute smoke (matches CI cadence from HLD §G9).
+# 5-minute local regression smoke (historical HLD §G9 cadence).
 cargo +nightly fuzz run --fuzz-dir crates/rvp9-fuzz fuzz_parse_packet -- -max_total_time=300 -dict=crates/rvp9-fuzz/fuzz.dict

-# 24-hour bake (matches nightly cadence from HLD §G9).
+# Optional 24-hour local bake (historical HLD §G9 recipe).
 cargo +nightly fuzz run --fuzz-dir crates/rvp9-fuzz fuzz_parse_packet -- -max_total_time=86400 -dict=crates/rvp9-fuzz/fuzz.dict
-
-# UBSan pass (catches integer-overflow regressions media decoders
-# accumulate). ASan is implicit with cargo-fuzz. Currently an
-# optional local pass; HLD §G9 calls for promotion to a required
-# CI lane — tracked separately as G9-CI follow-up.
-$env:RUSTFLAGS = '-Zsanitizer=undefined'
-cargo +nightly fuzz run --fuzz-dir crates/rvp9-fuzz fuzz_parse_packet
 ```

-bash equivalent (Linux / WSL / macOS CI):
-
-```bash
-RUSTFLAGS="-Zsanitizer=undefined" cargo +nightly fuzz run \
-    --fuzz-dir crates/rvp9-fuzz fuzz_parse_packet
-```
+Use a sanitizer supported by the installed Rust compiler. The installed
+macOS nightly rejects `-Zsanitizer=undefined`; the former UBSan recipe does
+not run on that toolchain. Default cargo-fuzz ASan checks instrumented memory
+accesses and does not establish general undefined-behaviour or integer-overflow
+coverage. Record the nightly identity, target, flags and actual runtime output.

 ### Smoke-run baseline (2026-05-21, Windows)

@@ -95,12 +92,15 @@ construction dominates; G9-PERF tracks hoisting that out).

 Crashes are written to `crates/rvp9-fuzz/artifacts/fuzz_parse_packet/`.

-## CI integration
+## Local regression checks

-There is no GitHub Actions workflow wired up yet. HLD §G9 calls for
-a 5-minute PR-blocking lane (`-max_total_time=300`) plus a nightly
-24-hour bake; both are tracked as G9-CI follow-ups separate from
-the parser-target landing in this commit.
+Release qualification uses local parser and full-frame campaigns. The older
+HLD's proposed hosted PR/nightly lanes are historical planning; do not add
+hosted CI workflows for the current release. Bound each command with
+`deltic timeout`, close stdin, retain raw logs and require nonzero executions
+with valid raw-packet reconstruction seeds. A startup failure or an empty
+campaign is not a pass. Keep the approved billion-iteration fuzz and 72-hour
+soak deferrals explicit; these smoke commands do not satisfy them.

 ## Layout notes (why not `[lints.workspace]`?)
