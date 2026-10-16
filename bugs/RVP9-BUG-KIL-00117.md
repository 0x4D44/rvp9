# RVP9-BUG-KIL-00117 — CLI destination aliases can destroy inputs and outputs

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** cli/path safety
- **Raised:** 2026-08-22T20:44:01Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T000225Z-17994e6d
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00117-run-fix-20261016T000225Z-17994e6d
- **Owner base:** fb2d60b2b3c823cf8c730e6e30c5028784936141
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T00:02:25Z
- **Owner until:** 2026-10-16T04:02:25Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T20:44:01Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

rvp9enc and rvp9dec open their inputs incrementally, then create output paths without checking file identity. If input, IVF output, decoder output, or frame-time CSV paths name the same file through an equal path, symlink, or hardlink, File::create or the final CSV write can truncate the source or replace the completed IVF. Expected: reject all aliased path roles or stage output and atomically rename after success. Confirmed statically at crates/rvp9-cli/src/bin/rvp9enc.rs:932,1019,1090,1144,1191,1267 and crates/rvp9-cli/src/bin/rvp9dec.rs:400,499,634; no app or test was run.

## Fix

<unfixed — raised only>

## Notes
