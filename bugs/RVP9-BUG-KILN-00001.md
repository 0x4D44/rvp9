# RVP9-BUG-KILN-00001 - Strided lookahead input can panic instead of returning an input-size error

- **State:** Closed
- **Severity:** High
- **Area:** encoder/lookahead
- **Raised:** 2026-06-17 by Codex overnight code review
- **State history:** 2026-06-17 Opened by Codex from the Stage 4a rate-control/frame-loop review pass.
- **State history:** 2026-06-23 Fixed by Codex (task/20260623-TSK-HUM-fix-kiln-00001-lookahead-stride; fixing commit: this commit): centralized strided plane validation on the actual final row end, using checked arithmetic before either `StreamEncoder` or `LookaheadCtx` copies source rows. Regression coverage: `lookahead::tests::push_rejects_strided_short_plane_without_panicking`, `frame_loop::tests::encode_frame_rejects_overflowing_y_stride`, and `frame_loop::tests::encode_frame_rejects_overflowing_y_plane_extent_without_panicking`. Awaiting two-eyes verification.
- **State history:** 2026-06-24 Closed by Codex after independent two-eyes verification on `ffc4e7e`: fixing commit `305095e` is present; `cargo test -p rvp9-encoder --lib` passed 484 tests with 1 ignored, including all named regressions; a disposable old-behavior check restored the pre-fix tight-size and unchecked-stride validation and confirmed `push_rejects_strided_short_plane_without_panicking` plus `encode_frame_rejects_overflowing_y_stride` fail with the original panic/overflow behavior; independent verifier agent `019efbdd-633a-73d3-b5d2-73fe0d2a4525` also returned CONFIRMED.

## Observation

`StreamEncoder::encode_frame_delayed` checks `yuv.y_stride * h` / UV analogues with unchecked `usize` multiplication at `crates/rvp9-encoder/src/frame_loop.rs:1571`, and `LookaheadCtx::push` validates direct callers against tight plane sizes (`w * h`) at `crates/rvp9-encoder/src/lookahead.rs:215` before copying rows with `row * stride` at `crates/rvp9-encoder/src/lookahead.rs:230`.

A padded-stride frame whose buffer length is at least `w * h` but shorter than the final row end offset can pass the direct lookahead validation and panic during `copy_from_slice`; sufficiently large stream-level strides can also overflow the `stride * rows` validation arithmetic before the intended `EncodeError::InvalidConfig` path.

## Notes

Confirmed by the maintainability, security/safety, reliability, and test-coverage review lenses, then rechecked in source. The fix should centralize plane-extent validation with checked arithmetic, preferably using `(rows - 1).checked_mul(stride).and_then(|base| base.checked_add(row_width))`, and return `InvalidPlaneSize` / `InvalidConfig` on overflow or short buffers. Add a regression that direct `LookaheadCtx::push` rejects padded-stride short buffers without panicking.

## Resolution (2026-06-23)

Root cause was a mismatch between validation and the actual row-copy access pattern.
`LookaheadCtx::push` accepted tight `w * h` / `uvw * uvh` buffers even when a padded
stride meant later row windows extended past the slice, and the stream-level
`encode_frame_delayed` guard used unchecked `stride * rows` arithmetic.

The fix validates every input plane against the final row window:
`(rows - 1) * stride + row_width`, with checked arithmetic and an explicit
`stride >= row_width` requirement. Direct lookahead callers now receive
`LookaheadError::InvalidPlaneSize`, and public stream callers receive
`EncodeError::InvalidConfig`, before any row copy or overflowing multiplication.

Pre-fix proof:
- `$null | mdtimeout 120 cargo test -p rvp9-encoder lookahead::tests::push_rejects_strided_short_plane_without_panicking -- --exact`
  panicked at `lookahead.rs` row slicing with "range end index 272 out of range for
  slice of length 256".
- `$null | mdtimeout 120 cargo test -p rvp9-encoder frame_loop::tests::encode_frame_rejects_overflowing_y_stride -- --exact`
  panicked at the unchecked `yuv.y_stride * h` multiplication.

Post-fix focused proof:
- `$null | mdtimeout 240 cargo test -p rvp9-encoder --lib strided -- --nocapture`
  passed the direct lookahead strided-plane regression and existing strided-input
  coverage.
- `$null | mdtimeout 240 cargo test -p rvp9-encoder --lib overflowing_y -- --nocapture`
  passed the stream-level overflowing-stride regressions.
