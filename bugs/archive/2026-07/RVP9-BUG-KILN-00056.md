# RVP9-BUG-KILN-00056 - Benchmark throughput drivers silently skip packet parse failures

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** bench/throughput
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-bench/` at `f7e477d`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T184006Z-p74554-n597592000-c1 branch=task/bug-RVP9-BUG-KILN-00056-run-fix-20260730T184006Z-p74554-n597592000-c1 code=80a3ea074c7b761c4e5d93c518804f9347ed7a2c gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

The shared throughput loop drops `Decoder::parse_packet` errors and continues measuring the remaining packets. `crates/rvp9-bench/src/throughput.rs:265`-`271` catches `Err(_)` from `decoder.parse_packet(packet)` and treats it as "not a decode sample" instead of a failed fixture.

The tile-profile driver repeats the same pattern: `crates/rvp9-bench/src/bin/decode-profile-tiles.rs:130`-`133` skips parse failures inside `drive_one_run`.

Expected: malformed or unparseable VP9 packets should fail the fixture measurement. Actual: a fixture with some parse failures and some successful frames can still return timing results, so the benchmark can report a successful row after silently omitting broken packets.

## Fix

Return a benchmark error on `parse_packet` failure, carrying the fixture and packet index. Keep `Unimplemented` decode skips explicit, but do not classify VP9 packet parse failure as a non-sample.

## Notes

`RVP9-BUG-KILN-00033` and `RVP9-BUG-KILN-00034` track similar false-green parse handling in harness conformance and threaded tests. This bug tracks the benchmark crate's measurement and profile drivers.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Both `Err(_) => continue` arms are gone; a parse failure now becomes a typed/stringified error
carrying fixture and 1-based packet index (`crates/rvp9-bench/src/throughput.rs:285`, error
variant at `:121`; `crates/rvp9-bench/src/bin/decode-profile-tiles.rs:130`). Confirmed no other
swallowed `parse_packet` remains in `crates/rvp9-bench`.
`throughput::tests::malformed_ivf_packet_is_reported_with_fixture_and_index` and
`decode-profile-tiles` `tests::malformed_packet_fails_profile_run_with_fixture_and_index` both
observed passing. Each builds a real IVF whose single packet is one 0x00 byte, hitting the exact
swallowed path; pre-fix both returned `Ok` or looped to `NoFramesDecoded`.
Exit path: `decode-profile-tiles` propagates to `main` and calls `std::process::exit(1)`.
Residual noted, not blocking and tracked as RVP9-BUG-KILN-00066: in the default multi-fixture
`decode-throughput` sweep a failed fixture still only prints "ERROR:" and the process exits 0
provided another fixture recorded a row. This record's own Expected is only that the FIXTURE
measurement fails, which it now does.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
