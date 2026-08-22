# RVP9-BUG-KIL-00117 — CLI destination aliases can destroy inputs and outputs

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** cli/path safety
- **Raised:** 2026-08-22T20:44:01Z
- **Discovery source:** Agent
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
- **State history:** Open (2026-08-22T20:44:01Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

rvp9enc and rvp9dec open their inputs incrementally, then create output paths without checking file identity. If input, IVF output, decoder output, or frame-time CSV paths name the same file through an equal path, symlink, or hardlink, File::create or the final CSV write can truncate the source or replace the completed IVF. Expected: reject all aliased path roles or stage output and atomically rename after success. Confirmed statically at crates/rvp9-cli/src/bin/rvp9enc.rs:932,1019,1090,1144,1191,1267 and crates/rvp9-cli/src/bin/rvp9dec.rs:400,499,634; no app or test was run.

## Fix

<unfixed — raised only>

## Notes
