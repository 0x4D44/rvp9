# RVP9-BUG-KILN-00005 - SVC refresh override can reset GOLDEN age without refreshing GOLDEN

- **State:** Open
- **Severity:** Medium
- **Area:** encoder/SVC GOP state
- **Raised:** 2026-06-17 by Codex overnight code review
- **State history:** 2026-06-17 Opened by Codex from the Stage 4a rate-control/frame-loop review pass.

## Observation

Temporal SVC overrides the GOP schedule's refresh mask so base-layer inter frames emit `0x01` and enhancement frames emit `0x00` (`crates/rvp9-encoder/src/frame_loop.rs:2427`, `crates/rvp9-encoder/src/frame_loop.rs:2900`). However `frames_since_golden_refresh` is reset from the scheduled update type (`refresh_golden_this_frame`) at `crates/rvp9-encoder/src/frame_loop.rs:3129`, not from the actual emitted `refresh_flags`.

On scheduled GOLDEN offsets under SVC, the encoder can emit no GOLDEN refresh but still reset the GOLDEN age counter. That counter feeds picker bias inputs at `crates/rvp9-encoder/src/frame_loop.rs:2763`, so later decisions can treat a stale GOLDEN as freshly refreshed.

## Notes

Confirmed by the correctness lens and rechecked in source. The fix should reset the counter only after all overrides, for example when `(refresh_flags & 0x02) != 0`; if SVC intentionally disables GOLDEN, the counter should age accordingly or the picker should ignore GOLDEN age while SVC owns the reference structure.
