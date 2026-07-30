# RVP9-BUG-KILN-00047 - BadFrameSize reports offsets as declared size diagnostics

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** io/container diagnostics
- **Raised:** 2026-07-07
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T183718Z-p70835-n325499000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00047-run-fix-20260730T183718Z-p70835-n325499000-c1
- **Owner base:** 965a9bed97c78e0d6127e432c832095a1d14fa86
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T18:37:18Z
- **Owner until:** 2026-07-30T20:37:18Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`)

## Observation

`ContainerError::BadFrameSize` reports IVF offset values through fields documented as byte counts.

`crates/rvp9-io/src/container.rs:53`-`56` document `declared` as the declared container size and `available` as remaining bytes in the file. The `Display` text at lines `72`-`75` uses those fields as byte counts. The IVF overflow arm at lines `190`-`195` instead assigns `declared: payload_end` and `available: data.len()`, which are absolute end offset and total file length. `crates/rvp9-io/tests/error_paths.rs:104`-`112` pins that offset-style output.

The error classification is still correct, but diagnostics can mislead callers and tests about the actual declared frame size and available payload bytes.

## Fix

Open. Either report `declared: size` and `available: data.len() - payload_start`, or rename/document the fields as absolute end offset and file length.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. This is tracked as low severity because it is diagnostic accuracy, not packet parsing correctness.
