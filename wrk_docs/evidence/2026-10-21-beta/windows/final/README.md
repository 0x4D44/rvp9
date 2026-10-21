# Windows beta candidate evidence

Candidate `15bf7162c9fb92a5165fab652460bd42da03a6e6`, v0.2.0-beta.1, native CRUCIBLE x86_64 MSVC Rust 1.95.0. Public-origin clone and fresh clean task use long paths and disable automatic line-ending conversion.

Observed terminal results:

- Portable package build passed using checked-in `package-beta.sh`, explicit MSVC target and Bash empty Rust flag variables. Archive SHA256 `f4abf8f0c889ef6275c1e2313ee18cbe9e3689eddaf30aae4382a2029a8db8de`. Actual archive is preserved in local task `.scratch/windows-artifacts/`, outside tracked evidence.
- Release workspace tests passed: 3,945 passed, 71 ignored across 124 result blocks. Formatting and clippy passed. These source-tree tests and helper tools used repository native CPU flags; they do not establish portability.
- Strict official conformance passed all four native/scalar, one/two-thread combinations: 307 vectors / 4,816 frames per run, zero mismatch/skip.
- Actual staged portable binaries passed adapted 32-row encoder and 24-row reverse decoder checks. Original scripts, adapted runners and exact diffs show only removal of build commands and explicit packaged binary substitution. They are adapted artifact checks, not unchanged checked-in runner executions.
- Source-tree supporting AQ oracle passed with +4.977% candidate / +4.714% libvpx gains, delta +0.263 percentage points; both the initial native helper and separate portable-flag replay passed. Portable replay fingerprint records `[]` Rust flags.
- Native ASan decoder panic safety passed four tests at beta source with encoded flags absent. All crate Rust source files match baseline exactly; baseline ASan core coverage remains supporting evidence. Beta instrumented executable hashes are retained.

The artifact-bound quality census was stopped at the lead’s instruction after independent Windows/macOS reproduction with the unchanged checked-in beta profile. Its apparent numeric PASS results are currently invalid: `bdrate` mistakes IVF timebase for frame cadence. Pinned Windows libvpx emits timebase 1/1000 with timestamps 0,33,66,100 for 30fps; candidate emits 1/30 with 0,1,2,3. Anchor bitrate is inflated by 1000/30, yielding spurious approximately -97% BD-rate. Two independent scenes reproduce this; raw headers, bytes and first-row results are retained. This requires a harness repair and fresh valid qualification before release.

Timing phases have not started. The lead subsequently authorized independent timing attempts on the unchanged artifact/runner. A 10-minute quiet lease was granted, but the exact 60-second guard refused 17.3652 busy logical cores against 0.75. The lease was immediately released; no timing group started. A second read-only 60-second resample also refused 6.10777 busy cores. A following process snapshot observed transient unrelated rustc compilation; it was left untouched.

Remote registered output scratch: `C:\Users\marti\AppData\Local\Temp\deltic-scratch\beta-final-win-914ede5e`. `copy-map.json` binds original remote bytes to UTF-8 normalized local `.raw.txt` logs. No source edits, commits or publication were performed by this worker.
