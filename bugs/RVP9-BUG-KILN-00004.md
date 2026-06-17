# RVP9-BUG-KILN-00004 - ARF FC bank adapts from the post-header bank instead of the pre-header snapshot

- **State:** Open
- **Severity:** High
- **Area:** encoder/altref entropy context
- **Raised:** 2026-06-17 by Codex overnight code review
- **State history:** 2026-06-17 Opened by Codex from the Stage 4a rate-control/frame-loop review pass.

## Observation

The normal show-frame inter path snapshots the decoder FC bank before `build_inter_packet_with_residuals` mutates it (`crates/rvp9-encoder/src/frame_loop.rs:2923`) and explicitly adapts from that pre-header snapshot (`crates/rvp9-encoder/src/frame_loop.rs:3022`). The ARF path clones bank 1 into `decoder_fc` (`crates/rvp9-encoder/src/frame_loop.rs:3371`), passes it mutably into `build_inter_packet_with_residuals` (`crates/rvp9-encoder/src/frame_loop.rs:3406`), then adapts bank 1 from the mutated `decoder_fc` at `crates/rvp9-encoder/src/frame_loop.rs:3433`.

This contradicts the show-frame path's documented decoder parity rule and can desynchronize the encoder's ARF FC bank mirror from the decoder's bank 1 after ARF frames.

## Notes

Confirmed by the correctness lens and rechecked against the source comments around the show-frame and ARF paths. The fix is small: clone a `pre_header_decoder_fc_arf` before the ARF packet builder mutates bank 1, then run mode, coefficient, and MV adaptation from that snapshot just as the show-frame path does.
