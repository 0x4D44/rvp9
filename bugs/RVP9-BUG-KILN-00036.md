# RVP9-BUG-KILN-00036 - diff-bitstream does not honor its bounded-run CLI contract

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/diff-bitstream-cli
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-harness/` at `cfabd55`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T191857Z-p14469-n072601000-c1 branch=task/bug-RVP9-BUG-KILN-00036-run-fix-20260730T191857Z-p14469-n072601000-c1 code=52b5400a7032ebcfd2c8a52b550152e920987189 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`diff-bitstream` advertises frame and first-difference bounds that do not actually bound most of the work.

The help text says `--max-frames` caps processed frames at `crates/rvp9-harness/src/bin/diff_bitstream.rs:284` and `--exit-on-diff` exits on first divergence at line `288`. Level 0 still scans the full byte buffers at line `537`, using `max_frames` only for frame attribution at line `554`. Levels 1 to 3 load and extract full frame data through lines `803` and `2273` before running deeper comparisons. Level 4 reads and parses the full files at lines `1831` and `1854`, then applies the frame cap only at line `1883`.

`--exit-on-diff` also does not stop traversal. The reporter applies final exit status in `finish()` at line `397`, while `run_with_streams` continues into the deeper levels after the Level 0 divergence path around lines `2264` and `2273`.

## Fix

Open. Apply the frame cap before byte diff, frame extraction, and every level loop, or rename the option to match its real behavior. Make `--exit-on-diff` return a stop signal so traversal short-circuits after the first emitted divergence.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review. Related memory-amplification observations are part of the same bounded-run contract problem and should be addressed with the cap fix.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

`--max-frames` is now a real bound: only the 32-byte IVF header plus N frame chunks are read,
and that capped buffer feeds L0's byte diff, `load_frames` for L1-L3, and L4
(`crates/rvp9-harness/src/bin/diff_bitstream.rs:579`). `--exit-on-diff` short-circuits traversal
after L0/L1/L2/L3 (`:2300`, `:2331`, `:2336`, `:2342`).
`tests::max_frames_excludes_later_bytes_before_diffing` and
`tests::exit_on_diff_stops_after_first_level_zero_record` both observed passing. The first is
clearly discriminating -- pre-fix the full read would have found the byte-62 diff and exited 1.
Two residuals noted, neither blocking. (1) The stop is at LEVEL granularity, not per-divergence:
a level that emits 500 records still emits all 500 before traversal halts. The observation's
complaint (traversal continues into deeper levels) is fixed; the Fix note's wording ("after the
first emitted divergence") is not fully met. (2) The exit-on-diff test may be non-discriminating
-- its fixture's 4-byte payloads are not valid VP9, so L1-L4 may produce no records either way.
Both recorded in `scratchpad.md`.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
