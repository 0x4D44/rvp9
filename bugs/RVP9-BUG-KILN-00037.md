# RVP9-BUG-KILN-00037 - e1_block_stats shares temp files and waits unbounded for vpxenc

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/e1-block-stats
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

`e1_block_stats` can corrupt concurrent runs and leave files behind when the external libvpx leg fails or hangs.

`crates/rvp9-harness/src/bin/e1_block_stats.rs:609` through line `611` writes `e1_<kbps>.yuv` and `e1_<kbps>.ivf` directly under the shared system temp directory. Two runs at the same bitrate can truncate or overwrite each other's inputs and outputs.

The libvpx subprocess is run with plain `Command::new(...).output()` at lines `624` and `643`, with no in-process timeout. Early failure paths return before cleanup at line `645`, while cleanup happens only after successful encode, read, and decode at line `665`.

`crates/rvp9-harness/src/bin/f02_block_oracle.rs:302` has the same unbounded `Command::output()` risk for its external `vpxenc` leg, but the temp-file race is specific to `e1_block_stats`.

## Fix

Open. Use a unique per-run temp directory or `create_new` files guarded by cleanup on all exits, run external encoders with a bounded wait and kill-on-timeout path, close stdin, and cap or spool subprocess output.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review. This is a harness reliability defect rather than a codec correctness bug.
