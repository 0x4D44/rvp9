# RVP9-BUG-KILN-00034 - Parallel harness oracles can pass on dropped or sentinel failures

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/threaded-test-oracles
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-harness/` at `cfabd55`)

## Observation

Several parallel decoder harness tests can report success after losing the evidence they are supposed to compare.

`crates/rvp9-harness/tests/tile_threading.rs:45` drops `parse_packet` failures with `continue`. The named-vector checks also skip missing vectors without a final checked-count assertion at lines `221`, `274`, `319`, and `416`.

The LF differential tests encode failures as digest values. `crates/rvp9-harness/tests/lf_overlap_pool_diff.rs:45` and line `61` push `[0u8; 16]` on parse or decode failure, then compare the vector at line `121`. `crates/rvp9-harness/tests/column_wavefront_lf_diff.rs:38` and line `54` do the same, then compare at line `116`. If both sides fail the same way, the tests compare identical sentinel bytes as if real decoded MD5s matched.

## Fix

Open. Fail missing required vectors in focused tests, return parse/decode failures as errors or explicit outcomes instead of MD5 bytes, and assert expected frame counts before comparing digests.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review. This is adjacent to `RVP9-BUG-KILN-00032`, but that bug covers real-world/decoder corpus oracle absence rather than threaded harness oracles and sentinel digest comparisons.
