# RVP9-BUG-KILN-00021 - Public ScanOrder can drive safe coefficient decoding into unchecked out-of-bounds access

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** core/entropy detokenize API
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-core/` at `af4c7a2d7533`)

## Observation

`crates/rvp9-core/src/transforms/scans.rs:42` exposes `ScanOrder` with public `scan`, `iscan`, and `neighbors` slice fields. Safe public `decode_coefs` accepts `&ScanOrder` at `crates/rvp9-core/src/entropy/detokenize.rs:375`, but validates only `seg_eob`, output length, and `initial_pt` before trusting the scan table.

The code later indexes the caller-supplied scan data through unchecked paths: `get_coef_context` uses unchecked `neighbors` and `token_cache` access at `crates/rvp9-core/src/entropy/detokenize.rs:278`, and token placement uses unchecked `scan.scan` and `token_cache` access at `crates/rvp9-core/src/entropy/detokenize.rs:596` and `crates/rvp9-core/src/entropy/detokenize.rs:629`. A safe caller can construct a malformed `ScanOrder` and reach out-of-bounds reads or writes through a safe API.

## Fix

Open. Make `ScanOrder` construction private or validated, and ensure public coefficient decode entry points cannot receive arbitrary scan tables. If custom scan orders remain public, validate slice lengths and raster indices before any unchecked indexing. Add regression coverage with malformed `ScanOrder` values that used to reach the unchecked paths.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 core review.
