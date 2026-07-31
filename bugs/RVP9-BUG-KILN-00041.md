# RVP9-BUG-KILN-00041 - Y4M reader accepts truncated structure as a short stream

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** io/y4m parser
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T181932Z-p41415-n886388000-c1 branch=task/bug-RVP9-BUG-KILN-00041-run-fix-20260730T181932Z-p41415-n886388000-c1 code=b1dff59e1dd6274ad790c977ed735c8921bcbef6 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

The Y4M reader can accept truncated or unterminated structure as a valid short stream.

`crates/rvp9-io/src/y4m.rs:47` says `parse_header` leaves the file positioned after the header newline, but line `63` calls `BufRead::read_line` without a length cap or any later check that a newline was actually present. A no-newline input that otherwise parses as `YUV4MPEG2 ...` can be accepted at lines `113`-`119`.

The frame reader has the same false-EOF shape. `read_cropped` documents errors for a missing `FRAME` tag or truncation at lines `136`-`138`, but lines `178`-`180` break out of the loop on any error while reading the 6-byte frame prefix. A tail such as `FRA` after valid frames is accepted as ordinary EOF rather than corruption.

## Fix

Open. Bound the header line length, require a terminating newline, and distinguish zero bytes at a frame boundary from partial `FRAME` header EOF.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. This is separate from harness false-green bugs because the defect is at the shared Y4M input boundary.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Both halves of the observation are addressed. The header is read through `.take(4097)` then
rejected if it exceeds 4 KiB or lacks a newline (`crates/rvp9-io/src/y4m.rs:177-192`), and the
FRAME prefix loop now distinguishes clean EOF (`Ok(None)`) from a partial prefix
(`UnexpectedEof "EOF in FRAME header"`) (`:334-346`). The tail-`FRA` case from the report is
literally the third test.
`y4m_unterminated_header_is_rejected`, `y4m_oversized_header_is_rejected` and
`read_cropped_partial_frame_header_is_rejected` all observed passing; all three return `Ok`
pre-fix, so they genuinely fail without it.
Over-strict-guard check: real Y4M headers are 40-60 bytes (the two checked-in files are 55), so
the 4 KiB cap cannot reach real input. Not debug-only. The logic survived a later refactor into
`Y4mReader::next_frame` intact.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
