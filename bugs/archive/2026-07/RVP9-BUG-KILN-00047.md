# RVP9-BUG-KILN-00047 - BadFrameSize reports offsets as declared size diagnostics

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** io/container diagnostics
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T183718Z-p70835-n325499000-c1 branch=task/bug-RVP9-BUG-KILN-00047-run-fix-20260730T183718Z-p70835-n325499000-c1 code=9d0bac4709ad13c4dfcace4a7f6831de12744d39 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`ContainerError::BadFrameSize` reports IVF offset values through fields documented as byte counts.

`crates/rvp9-io/src/container.rs:53`-`56` document `declared` as the declared container size and `available` as remaining bytes in the file. The `Display` text at lines `72`-`75` uses those fields as byte counts. The IVF overflow arm at lines `190`-`195` instead assigns `declared: payload_end` and `available: data.len()`, which are absolute end offset and total file length. `crates/rvp9-io/tests/error_paths.rs:104`-`112` pins that offset-style output.

The error classification is still correct, but diagnostics can mislead callers and tests about the actual declared frame size and available payload bytes.

## Fix

Open. Either report `declared: size` and `available: data.len() - payload_start`, or rename/document the fields as absolute end offset and file length.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. This is tracked as low severity because it is diagnostic accuracy, not packet parsing correctness.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Both `BadFrameSize` arms now emit `declared: size` and `available: data.len() - payload_start`
(`crates/rvp9-io/src/container.rs:355-371`), so the fields match their doc comments at `:51-58`
and the `Display` text at `:69-76`. Underflow is impossible because the loop guard
`pos + 12 <= data.len()` makes `payload_start <= data.len()`.
`ivf_frame_payload_running_past_eof_is_bad_frame_size` observed passing; it is a genuine
discriminator, previously asserting 144/total_len and now asserting 100/5.
The new streaming `IvfPacketReader` at `:296-304` uses the same corrected semantics, so the two
readers agree.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
