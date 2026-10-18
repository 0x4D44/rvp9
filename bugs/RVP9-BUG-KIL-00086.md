# RVP9-BUG-KIL-00086 — F02 accepts malformed and zero-work numeric requests

- **State:** Closed
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
- **State history:** Open (2026-08-22T13:20:52Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T00:02:04Z, deltic:auto role=fix run=fix-20261015T235458Z-d14ed932 branch=task/bug-RVP9-BUG-KIL-00086-run-fix-20261015T235458Z-d14ed932 code=666e899f3899a969f75e4b0650de487a8d5b4782 gate=manual) -> Closed (2026-10-18T16:45:02Z, deltic:manual role=verify run=verify-20261018T164154Z-d66b7b93 verified_fix_run=fix-20261015T235458Z-d14ed932 verdict=close)

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

### Verification summary (2026-10-18T16:45:02Z, deltic:auto run=verify-20261018T164154Z-d66b7b93 verified_fix_run=fix-20261015T235458Z-d14ed932 verdict=close)

Verifier note: Manual verify: Independent verification passed three focused numeric tests and 11 actual-binary rejection probes; each of three minimal parser/range/bitrate reversals made its retained assertion fail. Restored focused suite and all CLI probes pass with unchanged source bytes; adjacent temp/process/schema semantics were untouched.

## Notes

Parse supplied numeric values fallibly, require positive frame/rate values, and
require at least one bitrate. Add parser-level regressions for malformed,
mixed-validity, zero, and empty lists. Confirmed statically; no oracle run was
performed.
