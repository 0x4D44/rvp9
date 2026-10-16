# RVP9-BUG-KILN-00054 - rvp9dec raw I420/YV12 output writes native non-420 chroma planes

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9dec raw output
- **Raised:** 2026-07-07
- **Owner:** -
- **Owner role:** -
- **Owner run:** -
- **Owner host:** -
- **Owner branch:** -
- **Owner base:** -
- **Owner fingerprint:** -
- **Owner since:** -
- **Owner until:** -
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-cli/` at `8d66f33`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T183206Z-p60512-n528921000-c1 branch=task/bug-RVP9-BUG-KILN-00054-run-fix-20260730T183206Z-p60512-n528921000-c1 code=bc63dd52273b41280cd57d333ccd03bda69ec5e1 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`rvp9dec --i420` and `--yv12` write the decoded buffer's native chroma planes, even for legal Profile-1 non-4:2:0 streams where the output is not I420 or YV12-sized.

The raw writers emit `y,u,v` or `y,v,u` directly at `crates/rvp9-cli/src/bin/rvp9dec.rs:556`-`566`. `visible_planes` derives chroma dimensions from the decoded buffer's native `chroma_width()` and `chroma_height()` at `crates/rvp9-cli/src/bin/rvp9dec.rs:573`-`590`. The decoder supports legal non-420 layouts at `crates/rvp9-decoder/src/lib.rs:1320`-`1331`, and `Yv12Buffer::chroma_width` documents that Yuv444/Yuv440 chroma width equals luma width at `crates/rvp9-core/src/frame.rs:362`-`370`.

Expected: `--i420`/`--yv12` should either reject non-420 frames or perform explicit chroma conversion/downsampling. Actual: it writes native 4:2:2, 4:4:0, or 4:4:4 plane sizes under I420/YV12 option names.

## Fix

Reject `--i420` and `--yv12` unless `buf.chroma() == ChromaFormat::Yuv420`, or add explicit chroma conversion. If native-planar output is desired, expose it under a separate name that does not claim I420/YV12 layout.

## Notes

This is distinct from `RVP9-BUG-KILN-00031`, which tracks a decoder Profile-1 AQ2 correctness gap, and from `RVP9-BUG-KILN-00043`, which tracks Y4M header labeling in the shared IO writer.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Takes this record's "reject" branch: a chroma gate rejecting `OutFmt::RawI420` and
`OutFmt::RawYv12` unless `ChromaFormat::Yuv420` (`crates/rvp9-cli/src/bin/rvp9dec.rs:468-490`),
with the Yuv440 Y4M guard relocated here from `write_frame`. The gate sits ahead of the sink open
at `:493`, so a rejected non-420 stream also cannot clobber an existing output file -- a strict
improvement, since the Yuv440 guard used to fire only after the file was created.
`dec_i420_and_yv12_reject_profile1_444` observed passing. It spawns the real
`CARGO_BIN_EXE_rvp9dec` against the git-TRACKED Profile-1 fixture
`tests/corpus/profile1/foreman_c444_kfonly.ivf` (confirmed present, 288 KB), so it is not
silently skipped despite the corpus being generally gitignored, and it asserts stderr names the
flag and "4:2:0", which pre-fix could not appear.
Residual noted, not blocking: the gate is under `want_pixels`, which is false when
`--md5`/`--framemd5` is set, so an MD5 of a non-420 stream still hashes native planes without
comment. Outside this observation, which is about raw plane WRITING.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
