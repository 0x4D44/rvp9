# RVP9-BUG-KILN-00046 - IVF reader ignores trailing partial frame records

- **State:** Closed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T181227Z-p28158-n891950000-c1 branch=task/bug-RVP9-BUG-KILN-00046-run-fix-20260730T181227Z-p28158-n891950000-c1 code=152103f9e282e1a3f5c377bf4d7c49c53c203cbf gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

The IVF reader treats a trailing partial frame-record header as clean EOF.

`crates/rvp9-io/src/container.rs:178` only enters the frame loop when a full 12-byte IVF record header remains. If one to eleven bytes remain after the IVF file header or a complete frame payload, the loop ends and line `203` returns `Ok(packets)` without reporting truncation. `crates/rvp9-io/tests/error_paths.rs:143`-`151` explicitly pins the current behavior as success.

That means a truncated final frame record can be accepted as a shorter stream. A true zero-frame IVF file should stay valid, but trailing non-record bytes should not be silently ignored.

## Fix

Open. After the frame loop, return a truncation error when `pos != data.len()`, while preserving the valid 32-byte zero-frame IVF case.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. The existing test documents intent, but the parser's public error behavior still hides malformed tails.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Exactly the remedy this record proposed: `pos != data.len()` after the loop returns `Truncated`
(`crates/rvp9-io/src/container.rs:378-383`), mirrored in the streaming reader at `:286-291`. The
old test that PINNED the silent-success behaviour was deliberately rewritten, name and body both
inverted -- the right way to retire a false-green pin. The zero-frame 32-byte case still passes
since `pos == 32 == data.len()`.
`ivf_truncated_frame_record_header_is_rejected` (loops tail_len 1..12 after a complete frame)
observed passing.
Over-strict-guard check: all 20 checked-in `tests/corpus/**/*.ivf` files were parsed with an
independent script implementing the same walk -- zero have trailing bytes or bad sizes, so
nothing the repo already reads breaks. Not debug-only.
Residual noted, not blocking and pre-existing: `read_ivf_packets` hardcodes a 32-byte file header
and ignores the declared `header_size` field at offset 6.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
