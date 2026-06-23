# RVP9-BUG-KILN-00002 - Accepted encoder and rate-control configs overflow derived arithmetic

- **State:** Fixed
- **Severity:** High
- **Area:** encoder/rate-control config validation
- **Raised:** 2026-06-17 by Codex overnight code review
- **State history:** 2026-06-17 Opened by Codex from the Stage 4a rate-control/frame-loop review pass.
- **State history:** 2026-06-23 Fixed by Codex (task/20260623-TSK-HUM-fix-kiln-00001-lookahead-stride; fixing commit: this commit): added explicit supported config bounds before derived arithmetic or allocation, saturated the public macroblock-count helper for unvalidated callers, and made SVC bitrate splitting checked and total-preserving. Regression coverage: `validate_rejects_absurd_finite_framerate`, `validate_rejects_overflowing_macroblock_count`, `mbs_does_not_panic_on_unvalidated_huge_dimensions`, `validate_rejects_buffer_size_ms_to_bits_overflow`, `validate_rejects_unbounded_gop_length`, `validate_rejects_unbounded_kf_boost`, `stream_encoder_rejects_unsupported_large_dimensions_before_allocating`, `stream_encoder_rejects_svc_layer_bitrate_that_cannot_fit_i32`, `stream_encoder_preserves_nondivisible_svc_bitrate_remainder`, and `stream_encoder_rejects_unbounded_tpl_gop_before_allocating`. Awaiting two-eyes verification.

## Observation

Public config validation accepts very large but non-zero dimensions, bitrates, buffer sizes, and finite framerates. Those accepted values later feed unchecked or narrowing derived arithmetic:

- `RateControlConfig::mbs()` multiplies `u32` macroblock counts at `crates/rvp9-encoder/src/rate_control/config.rs:162`.
- `ms_to_bits()` multiplies widened `u32` values in `i64` at `crates/rvp9-encoder/src/rate_control/state.rs:491`.
- `StreamEncoder::new` accepts positive multiples of 64 at `crates/rvp9-encoder/src/frame_loop.rs:1214` and allocates frame-area buffers at `crates/rvp9-encoder/src/frame_loop.rs:1345`.
- The SVC setup narrows `target_bitrate_bps` to `i32` at `crates/rvp9-encoder/src/frame_loop.rs:1307`, so values above `i32::MAX` can become negative per-layer bandwidth.

These paths can panic in debug, wrap or go negative in release, or attempt unbounded allocations instead of returning `EncodeError::InvalidConfig` / a validation error.

## Notes

Confirmed by the security/safety, reliability, and maintainability review lenses, then rechecked in source. The fix should set explicit supported maximum coded dimensions / pixel count / bitrate-buffer ranges, perform derived size math with checked `usize`/`u64`/`u128`, reject overflow during validation or construction, and avoid lossy SVC casts.

## Resolution (2026-06-23)

Root cause was that public validation only rejected zero or non-finite values, while later code assumed much narrower arithmetic domains. Accepted configs could overflow `RateControlConfig::mbs`, `ms_to_bits`, the CBR keyframe boost formula, TPL lookahead sizing, or the SVC per-layer `i32` bitrate conversion before returning a structured config error.

The fix adds explicit bounds to `RateControlConfig::validate`: coded dimension and luma-pixel caps, finite framerate ceiling, GOP length ceiling, keyframe-boost ceiling, macroblock-count fit, and buffer-ms-by-bitrate product fit. `RateControlConfig::mbs` now computes in `u64` and saturates for unvalidated callers. `StreamEncoder::new` inherits the validation before allocating frame/TPL buffers, and its SVC setup now rejects unrepresentable layer bitrates while preserving non-divisible bitrate remainders.

Pre-fix proof:
- `$null | mdtimeout 120 cargo test -p rvp9-encoder --lib validate_rejects_absurd_finite_framerate -- --nocapture` failed because validation returned `Ok(())`.
- `$null | mdtimeout 120 cargo test -p rvp9-encoder --lib validate_rejects_buffer_size_ms_to_bits_overflow -- --nocapture` failed because validation returned `Ok(())`.
- `$null | mdtimeout 120 cargo test -p rvp9-encoder --lib mbs_does_not_panic_on_unvalidated_huge_dimensions -- --nocapture` caught a debug overflow panic.
- `$null | mdtimeout 120 cargo test -p rvp9-encoder --lib stream_encoder_rejects_unsupported_large_dimensions_before_allocating -- --nocapture` failed because `StreamEncoder::new` accepted the config.
- `$null | mdtimeout 120 cargo test -p rvp9-encoder --lib stream_encoder_rejects_svc_layer_bitrate_that_cannot_fit_i32 -- --nocapture` failed because SVC accepted the lossy narrowing path.
- `$null | mdtimeout 120 cargo test -p rvp9-encoder --lib stream_encoder_preserves_nondivisible_svc_bitrate_remainder -- --nocapture` observed per-layer targets `[50, 50]` for a 101 bps target.

Post-fix focused proof:
- `$null | mdtimeout 600 cargo test -p rvp9-encoder --lib "validate_rejects" -- --nocapture` passed 18 validation tests.
- `$null | mdtimeout 900 cargo test -p rvp9-encoder --lib` passed 475 tests with 1 ignored, including all KILN-00002 regressions.

Full-gate caveats:
- `$null | mdtimeout 900 cargo +stable clippy --workspace --all-targets -- -D warnings` is blocked by existing lint debt outside this fix, beginning with duplicated `#[allow(clippy::too_many_arguments)]` attributes and broad `type_complexity` warnings in `rvp9-core`.
- `$null | mdtimeout 1200 cargo test --workspace` is blocked by missing captured vectors in this fresh worktree: `bitstream::compressed_header::tests::write_compressed_header_captured_frame_round_trip` and `bitstream::header::tests::captured_header_round_trip`.
