# RVP9-BUG-KILN-00003 - Flush leaves queued lookahead frames after an encode error

- **State:** Open
- **Severity:** Medium
- **Area:** encoder/frame-loop flush
- **Raised:** 2026-06-17 by Codex overnight code review
- **State history:** 2026-06-17 Opened by Codex from the Stage 4a rate-control/frame-loop review pass.

## Observation

The `flush()` documentation promises that after flush the lookahead is empty and the encoder will accept new pushes (`crates/rvp9-encoder/src/frame_loop.rs:1609`). The implementation breaks immediately after the first encode error (`crates/rvp9-encoder/src/frame_loop.rs:1619`), while `try_encode_one_popped` has already popped the current frame before encoding (`crates/rvp9-encoder/src/frame_loop.rs:1643`).

If a delayed encode fails with multiple frames queued, the failed frame is consumed, later lookahead frames remain queued, and the documented reusable/empty state is not reached.

## Notes

Confirmed by the reliability lens and rechecked in source. The fix should choose an explicit contract: either poison the encoder after an encode error and document that no drain/reuse guarantee remains, or drain/discard the remaining lookahead and clear pending per-frame state before returning the error.
