# RVP9-BUG-KILN-00001 - Strided lookahead input can panic instead of returning an input-size error

- **State:** Open
- **Severity:** High
- **Area:** encoder/lookahead
- **Raised:** 2026-06-17 by Codex overnight code review
- **State history:** 2026-06-17 Opened by Codex from the Stage 4a rate-control/frame-loop review pass.

## Observation

`StreamEncoder::encode_frame_delayed` checks `yuv.y_stride * h` / UV analogues with unchecked `usize` multiplication at `crates/rvp9-encoder/src/frame_loop.rs:1571`, and `LookaheadCtx::push` validates direct callers against tight plane sizes (`w * h`) at `crates/rvp9-encoder/src/lookahead.rs:215` before copying rows with `row * stride` at `crates/rvp9-encoder/src/lookahead.rs:230`.

A padded-stride frame whose buffer length is at least `w * h` but shorter than the final row end offset can pass the direct lookahead validation and panic during `copy_from_slice`; sufficiently large stream-level strides can also overflow the `stride * rows` validation arithmetic before the intended `EncodeError::InvalidConfig` path.

## Notes

Confirmed by the maintainability, security/safety, reliability, and test-coverage review lenses, then rechecked in source. The fix should centralize plane-extent validation with checked arithmetic, preferably using `(rows - 1).checked_mul(stride).and_then(|base| base.checked_add(row_width))`, and return `InvalidPlaneSize` / `InvalidConfig` on overflow or short buffers. Add a regression that direct `LookaheadCtx::push` rejects padded-stride short buffers without panicking.
