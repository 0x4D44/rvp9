# RVP9-BUG-KILN-00037 - e1_block_stats shares temp files and waits unbounded for vpxenc

- **State:** Closed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-harness/` at `cfabd55`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T192602Z-p19018-n380158000-c1 branch=task/bug-RVP9-BUG-KILN-00037-run-fix-20260730T192602Z-p19018-n380158000-c1 code=068c152a83709c98d86f542490a8cb52b7271c60 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5); residual split to RVP9-BUG-KILN-00064)

## Observation

`e1_block_stats` can corrupt concurrent runs and leave files behind when the external libvpx leg fails or hangs.

`crates/rvp9-harness/src/bin/e1_block_stats.rs:609` through line `611` writes `e1_<kbps>.yuv` and `e1_<kbps>.ivf` directly under the shared system temp directory. Two runs at the same bitrate can truncate or overwrite each other's inputs and outputs.

The libvpx subprocess is run with plain `Command::new(...).output()` at lines `624` and `643`, with no in-process timeout. Early failure paths return before cleanup at line `645`, while cleanup happens only after successful encode, read, and decode at line `665`.

`crates/rvp9-harness/src/bin/f02_block_oracle.rs:302` has the same unbounded `Command::output()` risk for its external `vpxenc` leg, but the temp-file race is specific to `e1_block_stats`.

## Fix

Open. Use a unique per-run temp directory or `create_new` files guarded by cleanup on all exits, run external encoders with a bounded wait and kill-on-timeout path, close stdin, and cap or spool subprocess output.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review. This is a harness reliability defect rather than a codec correctness bug.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Every part of the e1_block_stats limb is addressed at the root: the shared `e1_<kbps>.yuv`/`.ivf`
files are replaced by a pid+atomic-sequence directory created with `create_dir`
(fails-and-retries on collision, so no silent reuse) whose `Drop` removes it recursively --
covering the early-failure return paths this record called out; and `run_with_timeout` closes
stdin, polls to a 30s deadline, then kills AND reaps
(`crates/rvp9-harness/src/bin/e1_block_stats.rs:57`, `:86`, adopted at `:685`).
`tests::scoped_temp_dirs_are_unique_and_removed_on_drop` and
`tests::run_with_timeout_kills_a_hung_child` both observed passing.
Residual split to RVP9-BUG-KILN-00064: this record's Observation ALSO named
`crates/rvp9-harness/src/bin/f02_block_oracle.rs:302` as carrying the same defect. That site is
untouched -- now at `:363`, still a plain `cmd.output()?` with no timeout and no
`.stdin(Stdio::null())`.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
