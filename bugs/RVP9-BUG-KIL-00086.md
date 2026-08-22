# RVP9-BUG-KIL-00086 — F02 accepts malformed and zero-work numeric requests

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/f02 oracle
- **Raised:** 2026-08-22T13:20:52Z
- **Discovery source:** Agent
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
- **State history:** Open (2026-08-22T13:20:52Z, raised via `deltic bugs new`)

## Observation

`f02-block-oracle` silently substitutes defaults for malformed supplied values
of `--frames`, `--width`, `--height`, `--fps`, `--threads`, and
`--tile-columns` at
`crates/rvp9-harness/src/bin/f02_block_oracle.rs:453-502`. Its `--kbps` parser
at lines 503-507 silently drops malformed tokens and accepts an empty result.
For example, `--kbps nope` makes the work loop at line 614 execute zero times,
then the command reaches the successful `Done` path at line 1592.

Expected: omitted options use documented defaults, while supplied malformed or
zero-work values fail clearly. Actual: a mistyped experiment can run with
different parameters or report success after no encode or comparison work.

## Fix

<unfixed — raised only>

## Notes

Parse supplied numeric values fallibly, require positive frame/rate values, and
require at least one bitrate. Add parser-level regressions for malformed,
mixed-validity, zero, and empty lists. Confirmed statically; no oracle run was
performed.
