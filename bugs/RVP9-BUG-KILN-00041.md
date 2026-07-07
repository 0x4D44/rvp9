# RVP9-BUG-KILN-00041 - Y4M reader accepts truncated structure as a short stream

- **State:** Open
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`)

## Observation

The Y4M reader can accept truncated or unterminated structure as a valid short stream.

`crates/rvp9-io/src/y4m.rs:47` says `parse_header` leaves the file positioned after the header newline, but line `63` calls `BufRead::read_line` without a length cap or any later check that a newline was actually present. A no-newline input that otherwise parses as `YUV4MPEG2 ...` can be accepted at lines `113`-`119`.

The frame reader has the same false-EOF shape. `read_cropped` documents errors for a missing `FRAME` tag or truncation at lines `136`-`138`, but lines `178`-`180` break out of the loop on any error while reading the 6-byte frame prefix. A tail such as `FRA` after valid frames is accepted as ordinary EOF rather than corruption.

## Fix

Open. Bound the header line length, require a terminating newline, and distinguish zero bytes at a frame boundary from partial `FRAME` header EOF.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. This is separate from harness false-green bugs because the defect is at the shared Y4M input boundary.
