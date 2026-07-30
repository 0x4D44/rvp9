# RVP9-BUG-KILN-00056 - Benchmark throughput drivers silently skip packet parse failures

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/throughput
- **Raised:** 2026-07-07
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T184006Z-p74554-n597592000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00056-run-fix-20260730T184006Z-p74554-n597592000-c1
- **Owner base:** 91b4732bfa529b5d3fefd281dad6ecffc07a7d30
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T18:40:06Z
- **Owner until:** 2026-07-30T20:40:06Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-bench/` at `f7e477d`)

## Observation

The shared throughput loop drops `Decoder::parse_packet` errors and continues measuring the remaining packets. `crates/rvp9-bench/src/throughput.rs:265`-`271` catches `Err(_)` from `decoder.parse_packet(packet)` and treats it as "not a decode sample" instead of a failed fixture.

The tile-profile driver repeats the same pattern: `crates/rvp9-bench/src/bin/decode-profile-tiles.rs:130`-`133` skips parse failures inside `drive_one_run`.

Expected: malformed or unparseable VP9 packets should fail the fixture measurement. Actual: a fixture with some parse failures and some successful frames can still return timing results, so the benchmark can report a successful row after silently omitting broken packets.

## Fix

Return a benchmark error on `parse_packet` failure, carrying the fixture and packet index. Keep `Unimplemented` decode skips explicit, but do not classify VP9 packet parse failure as a non-sample.

## Notes

`RVP9-BUG-KILN-00033` and `RVP9-BUG-KILN-00034` track similar false-green parse handling in harness conformance and threaded tests. This bug tracks the benchmark crate's measurement and profile drivers.
