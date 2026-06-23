# RVP9-BUG-KILN-00005 - SVC refresh override can reset GOLDEN age without refreshing GOLDEN

- **State:** Fixed
- **Severity:** Medium
- **Area:** encoder/SVC GOP state
- **Raised:** 2026-06-17 by Codex overnight code review
- **State history:** 2026-06-17 Opened by Codex from the Stage 4a rate-control/frame-loop review pass.
- **State history:** 2026-06-23 Fixed by Codex (task/20260623-TSK-HUM-fix-kiln-00001-lookahead-stride; fixing commit: this commit): `frames_since_golden_refresh` now resets from the final emitted GOLDEN refresh bit, not from the pre-SVC GOP schedule. Regression coverage: `a18_temporal_svc_golden_age_follows_emitted_refresh_flags`. Awaiting two-eyes verification.

## Observation

Temporal SVC overrides the GOP schedule's refresh mask so base-layer inter frames emit `0x01` and enhancement frames emit `0x00` (`crates/rvp9-encoder/src/frame_loop.rs:2427`, `crates/rvp9-encoder/src/frame_loop.rs:2900`). However `frames_since_golden_refresh` is reset from the scheduled update type (`refresh_golden_this_frame`) at `crates/rvp9-encoder/src/frame_loop.rs:3129`, not from the actual emitted `refresh_flags`.

On scheduled GOLDEN offsets under SVC, the encoder can emit no GOLDEN refresh but still reset the GOLDEN age counter. That counter feeds picker bias inputs at `crates/rvp9-encoder/src/frame_loop.rs:2763`, so later decisions can treat a stale GOLDEN as freshly refreshed.

## Notes

Confirmed by the correctness lens and rechecked in source. The fix should reset the counter only after all overrides, for example when `(refresh_flags & 0x02) != 0`; if SVC intentionally disables GOLDEN, the counter should age accordingly or the picker should ignore GOLDEN age while SVC owns the reference structure.

## Resolution (2026-06-23)

Root cause was that `encode_inter_internal()` computed a scheduled `GfUpdate` from the GOP topology, then let temporal SVC override the final `refresh_frame_flags`, but still reset `frames_since_golden_refresh` from the scheduled update type. Under two-layer SVC, GOP offset 8 is scheduled as a GOLDEN refresh while the SVC pattern makes that frame an enhancement layer and emits no GOLDEN refresh. The age counter reset even though the GOLDEN slot stayed stale.

The fix resets `frames_since_golden_refresh` only when the final emitted refresh mask has bit 1 set (`refresh_flags & 0x02 != 0`). This keeps the counter aligned with the same mask used to rotate the reference pool and decoder state.

Pre-fix proof:
- `$null | mdtimeout 300 cargo test -p rvp9-encoder --lib a18_temporal_svc_golden_age_follows_emitted_refresh_flags -- --nocapture` failed with `left: [0, 1, 2, 3, 4, 5, 6, 7, 0]` and `right: [0, 1, 2, 3, 4, 5, 6, 7, 8]`.

Post-fix focused proof:
- `$null | mdtimeout 300 cargo test -p rvp9-encoder --lib a18_temporal_svc_golden_age_follows_emitted_refresh_flags -- --nocapture` passed.
- `$null | mdtimeout 300 cargo test -p rvp9-encoder --lib a18_temporal_svc_2layer_pattern_live -- --nocapture` passed.
- `$null | mdtimeout 300 cargo test -p rvp9-encoder --lib a18_single_layer_tags_all_tl0 -- --nocapture` passed.
- `$null | mdtimeout 900 cargo test -p rvp9-encoder --lib` passed 478 tests with 1 ignored.
- `cargo +stable clippy -p rvp9-encoder --lib` exited 0 with existing warnings only.

Broader gate notes:
- `mdtimeout 120 cargo fmt --all -- --check` passed.
- `mdtimeout 120 git diff --check` passed.
- `cargo clippy --workspace --all-targets -- -D warnings` remains blocked because the default `1.95.0-x86_64-pc-windows-msvc` toolchain has no applicable `cargo-clippy.exe`.
- `cargo +stable clippy --workspace --all-targets -- -D warnings` remains blocked by existing warning debt beginning in `rvp9-core` (`duplicated_attributes`, `precedence`, `type_complexity`).
- `cargo +stable clippy --workspace --all-targets` remains blocked by existing unrelated test lint errors, including missing `SAFETY` comments in encoder integration tests.
- `$null | mdtimeout 1200 cargo test --workspace` remains blocked in this fresh worktree by missing captured vectors for `bitstream::compressed_header::tests::write_compressed_header_captured_frame_round_trip` and `bitstream::header::tests::captured_header_round_trip`; the reported run reached 1889 passed and 2 failed.
