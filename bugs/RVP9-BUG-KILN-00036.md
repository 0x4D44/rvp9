# RVP9-BUG-KILN-00036 - diff-bitstream does not honor its bounded-run CLI contract

- **State:** Fixed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-harness/` at `cfabd55`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T191857Z-p14469-n072601000-c1 branch=task/bug-RVP9-BUG-KILN-00036-run-fix-20260730T191857Z-p14469-n072601000-c1 code=52b5400a7032ebcfd2c8a52b550152e920987189 gate=manual)

## Observation

`diff-bitstream` advertises frame and first-difference bounds that do not actually bound most of the work.

The help text says `--max-frames` caps processed frames at `crates/rvp9-harness/src/bin/diff_bitstream.rs:284` and `--exit-on-diff` exits on first divergence at line `288`. Level 0 still scans the full byte buffers at line `537`, using `max_frames` only for frame attribution at line `554`. Levels 1 to 3 load and extract full frame data through lines `803` and `2273` before running deeper comparisons. Level 4 reads and parses the full files at lines `1831` and `1854`, then applies the frame cap only at line `1883`.

`--exit-on-diff` also does not stop traversal. The reporter applies final exit status in `finish()` at line `397`, while `run_with_streams` continues into the deeper levels after the Level 0 divergence path around lines `2264` and `2273`.

## Fix

Open. Apply the frame cap before byte diff, frame extraction, and every level loop, or rename the option to match its real behavior. Make `--exit-on-diff` return a stop signal so traversal short-circuits after the first emitted divergence.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review. Related memory-amplification observations are part of the same bounded-run contract problem and should be addressed with the cap fix.
