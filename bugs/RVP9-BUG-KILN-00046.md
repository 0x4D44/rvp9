# RVP9-BUG-KILN-00046 - IVF reader ignores trailing partial frame records

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** io/ivf packet reader
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`)

## Observation

The IVF reader treats a trailing partial frame-record header as clean EOF.

`crates/rvp9-io/src/container.rs:178` only enters the frame loop when a full 12-byte IVF record header remains. If one to eleven bytes remain after the IVF file header or a complete frame payload, the loop ends and line `203` returns `Ok(packets)` without reporting truncation. `crates/rvp9-io/tests/error_paths.rs:143`-`151` explicitly pins the current behavior as success.

That means a truncated final frame record can be accepted as a shorter stream. A true zero-frame IVF file should stay valid, but trailing non-record bytes should not be silently ignored.

## Fix

Open. After the frame loop, return a truncation error when `pos != data.len()`, while preserving the valid 32-byte zero-frame IVF case.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. The existing test documents intent, but the parser's public error behavior still hides malformed tails.
