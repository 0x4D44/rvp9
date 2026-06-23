# RVP9-BUG-KILN-00004 - ARF FC bank adapts from the post-header bank instead of the pre-header snapshot

- **State:** Fixed
- **Severity:** High
- **Area:** encoder/altref entropy context
- **Raised:** 2026-06-17 by Codex overnight code review
- **State history:** 2026-06-17 Opened by Codex from the Stage 4a rate-control/frame-loop review pass.
- **State history:** 2026-06-23 Fixed by Codex (task/20260623-TSK-HUM-fix-kiln-00001-lookahead-stride; fixing commit: this commit): ARF bank-1 backward adaptation now starts from the pre-header bank snapshot, matching the decoder's saved frame context. Regression coverage: `arf_decoder_fc_bank_one_stays_synced_after_non_show_frame`; adjacent bank-0 proof `a6_twoloop_decoder_fc_stays_synced_across_frames` remains green. Awaiting two-eyes verification.

## Observation

The normal show-frame inter path snapshots the decoder FC bank before `build_inter_packet_with_residuals` mutates it (`crates/rvp9-encoder/src/frame_loop.rs:2923`) and explicitly adapts from that pre-header snapshot (`crates/rvp9-encoder/src/frame_loop.rs:3022`). The ARF path clones bank 1 into `decoder_fc` (`crates/rvp9-encoder/src/frame_loop.rs:3371`), passes it mutably into `build_inter_packet_with_residuals` (`crates/rvp9-encoder/src/frame_loop.rs:3406`), then adapts bank 1 from the mutated `decoder_fc` at `crates/rvp9-encoder/src/frame_loop.rs:3433`.

This contradicts the show-frame path's documented decoder parity rule and can desynchronize the encoder's ARF FC bank mirror from the decoder's bank 1 after ARF frames.

## Notes

Confirmed by the correctness lens and rechecked against the source comments around the show-frame and ARF paths. The fix is small: clone a `pre_header_decoder_fc_arf` before the ARF packet builder mutates bank 1, then run mode, coefficient, and MV adaptation from that snapshot just as the show-frame path does.

## Resolution (2026-06-23)

Root cause was that the ARF path reused `decoder_fc` for two distinct roles: compressed-header diff state and backward-adaptation base. `build_inter_packet_with_residuals` mutates that value while coding header probability updates toward the picker FC, but the decoder adapts frame context bank 1 from its saved pre-header bank. The show-frame path already kept a pre-header snapshot; the ARF path did not.

The fix mirrors the show-frame rule in `encode_altref_internal`: clone `pre_header_decoder_fc_arf` before packet assembly mutates bank 1, then feed that snapshot to `adapt_mode_probs`, `adapt_coef_probs`, and `adapt_mv_probs`. The packet builder still receives the mutable bank-1 mirror for compressed-header coding.

Pre-fix proof:
- `$null | mdtimeout 300 cargo test -p rvp9-encoder --lib arf_decoder_fc_bank_one_stays_synced_after_non_show_frame -- --nocapture` failed because the encoder's ARF bank-1 mirror differed from `Decoder::frame_context(1)` immediately after decoding a non-show ARF.

Post-fix focused proof:
- `$null | mdtimeout 300 cargo test -p rvp9-encoder --lib arf_decoder_fc_bank_one_stays_synced_after_non_show_frame -- --nocapture` passed.
- `$null | mdtimeout 300 cargo test -p rvp9-encoder --lib a6_twoloop_decoder_fc_stays_synced_across_frames -- --nocapture` passed, preserving the neighboring show-frame bank-0 invariant.
- `$null | mdtimeout 900 cargo test -p rvp9-encoder --lib` passed 476 tests with 1 ignored.

Full-gate caveats:
- `mdtimeout 900 cargo clippy --workspace --all-targets -- -D warnings` did not run on the default `1.95.0-x86_64-pc-windows-msvc` toolchain because its clippy component is not applicable; `cargo +stable clippy --workspace --all-targets -- -D warnings` is blocked by existing warning debt outside this fix.
- `cargo +stable clippy --workspace --all-targets` is blocked by existing denied unsafe-comment lint failures in encoder integration tests, including `crates/rvp9-encoder/tests/w_rd_partition_rect.rs:247`, `crates/rvp9-encoder/tests/stage_sse_diag.rs:137`, and `crates/rvp9-encoder/tests/w_rd_partition.rs:248`.
- `cargo +stable clippy -p rvp9-encoder --lib` passed for the touched library target, with existing warnings only.
- `$null | mdtimeout 1200 cargo test --workspace` is blocked by missing captured vectors in this fresh worktree: `bitstream::compressed_header::tests::write_compressed_header_captured_frame_round_trip` and `bitstream::header::tests::captured_header_round_trip`.
