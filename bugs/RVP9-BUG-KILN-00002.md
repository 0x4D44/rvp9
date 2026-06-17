# RVP9-BUG-KILN-00002 - Accepted encoder and rate-control configs overflow derived arithmetic

- **State:** Open
- **Severity:** High
- **Area:** encoder/rate-control config validation
- **Raised:** 2026-06-17 by Codex overnight code review
- **State history:** 2026-06-17 Opened by Codex from the Stage 4a rate-control/frame-loop review pass.

## Observation

Public config validation accepts very large but non-zero dimensions, bitrates, buffer sizes, and finite framerates. Those accepted values later feed unchecked or narrowing derived arithmetic:

- `RateControlConfig::mbs()` multiplies `u32` macroblock counts at `crates/rvp9-encoder/src/rate_control/config.rs:162`.
- `ms_to_bits()` multiplies widened `u32` values in `i64` at `crates/rvp9-encoder/src/rate_control/state.rs:491`.
- `StreamEncoder::new` accepts positive multiples of 64 at `crates/rvp9-encoder/src/frame_loop.rs:1214` and allocates frame-area buffers at `crates/rvp9-encoder/src/frame_loop.rs:1345`.
- The SVC setup narrows `target_bitrate_bps` to `i32` at `crates/rvp9-encoder/src/frame_loop.rs:1307`, so values above `i32::MAX` can become negative per-layer bandwidth.

These paths can panic in debug, wrap or go negative in release, or attempt unbounded allocations instead of returning `EncodeError::InvalidConfig` / a validation error.

## Notes

Confirmed by the security/safety, reliability, and maintainability review lenses, then rechecked in source. The fix should set explicit supported maximum coded dimensions / pixel count / bitrate-buffer ranges, perform derived size math with checked `usize`/`u64`/`u128`, reject overflow during validation or construction, and avoid lossy SVC casts.
