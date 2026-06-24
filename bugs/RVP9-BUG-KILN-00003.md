# RVP9-BUG-KILN-00003 - Flush leaves queued lookahead frames after an encode error

- **State:** Closed
- **Severity:** Medium
- **Area:** encoder/frame-loop flush
- **Raised:** 2026-06-17 by Codex overnight code review
- **State history:** 2026-06-17 Opened by Codex from the Stage 4a rate-control/frame-loop review pass.
- **State history:** 2026-06-23 Fixed by Codex (task/20260623-TSK-HUM-fix-kiln-00001-lookahead-stride; fixing commit: this commit): `flush()` now reports the first post-pop encode error once, discards any remaining live lookahead frames, and clears pending per-frame state so the delayed encoder can accept fresh pushes. Regression coverage: `flush_discards_remaining_lookahead_after_encode_error`; neighboring delayed-output proofs remain green. Awaiting two-eyes verification.
- **State history:** 2026-06-24 Closed by Codex after independent two-eyes verification on `ffc4e7e`: fixing commit `4dd622b` is present; `cargo test -p rvp9-encoder --lib` passed 484 tests with 1 ignored, including `flush_discards_remaining_lookahead_after_encode_error` and neighboring delayed-output coverage; a disposable old-behavior check removed the post-error tail drain and confirmed the regression fails with `lookahead.depth()` left at 2; independent verifier agent `019efbdd-94fe-71d0-b05f-0a03a8b3e570` also returned CONFIRMED.

## Observation

The `flush()` documentation promises that after flush the lookahead is empty and the encoder will accept new pushes (`crates/rvp9-encoder/src/frame_loop.rs:1609`). The implementation breaks immediately after the first encode error (`crates/rvp9-encoder/src/frame_loop.rs:1619`), while `try_encode_one_popped` has already popped the current frame before encoding (`crates/rvp9-encoder/src/frame_loop.rs:1643`).

If a delayed encode fails with multiple frames queued, the failed frame is consumed, later lookahead frames remain queued, and the documented reusable/empty state is not reached.

## Notes

Confirmed by the reliability lens and rechecked in source. The fix should choose an explicit contract: either poison the encoder after an encode error and document that no drain/reuse guarantee remains, or drain/discard the remaining lookahead and clear pending per-frame state before returning the error.

## Resolution (2026-06-23)

Root cause was that `flush()` treated the first encode error as terminal for the loop without draining the live lookahead tail. `try_encode_one_popped` had already popped the frame that failed, but later queued frames remained in the ring, contradicting the documented post-flush empty/reusable state.

The fix keeps the existing return shape by reporting the first error once, then discards the remaining live lookahead entries and clears volatile per-frame state (`pending_arf_packet`, `svc_frame`, and the frame-local S2 gate). Clean-stream flush behavior is unchanged.

Pre-fix proof:
- `$null | mdtimeout 300 cargo test -p rvp9-encoder --lib flush_discards_remaining_lookahead_after_encode_error -- --nocapture` failed with `lookahead.depth() == 2` after `flush()` returned the injected post-pop encode error.

Post-fix focused proof:
- `$null | mdtimeout 300 cargo test -p rvp9-encoder --lib flush_discards_remaining_lookahead_after_encode_error -- --nocapture` passed.
- `$null | mdtimeout 300 cargo test -p rvp9-encoder --lib w_lag_in_frames_delays_output_then_flushes -- --nocapture` passed.
- `$null | mdtimeout 300 cargo test -p rvp9-encoder --lib w_lag_in_frames_zero_stays_synchronous -- --nocapture` passed.
- `$null | mdtimeout 900 cargo test -p rvp9-encoder --lib` passed 477 tests with 1 ignored.

Broader gate notes:
- `mdtimeout 120 cargo fmt --all -- --check` passed.
- `mdtimeout 120 git diff --check` passed.
- `cargo +stable clippy --workspace --all-targets -- -D warnings` remains blocked by existing warning debt beginning in `rvp9-core` (`duplicated_attributes`, `precedence`, `type_complexity`).
- `cargo +stable clippy --workspace --all-targets` remains blocked by existing unrelated test lint errors, including missing `SAFETY` comments in encoder integration tests and banned `eprintln!` in `phase2_reuse_positive_control`.
- `$null | mdtimeout 1200 cargo test --workspace` remains blocked in this fresh worktree by missing captured vectors for `bitstream::compressed_header::tests::write_compressed_header_captured_frame_round_trip` and `bitstream::header::tests::captured_header_round_trip`; the reported run reached 1889 passed and 2 failed.
