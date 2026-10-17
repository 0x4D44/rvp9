# One-pass rate-control buffer-model audit

The current release candidate is product source `fb5c0c917038db6c640b3dbc7c2b84b567233333` (`fb5c0c91`). PLAN §2 names “1-pass CBR + VBR-RT rate control with buffer-model compliance.” The proposed release boundary makes single-stream, denoiser and spatial CBR production modes; VBR remains experimental. This is an acceptance gap, not evidence that the controller is missing.

## Implementation and focused coverage

The candidate has a CBR leaky-bucket controller. `crates/rvp9-encoder/src/rate_control/mod.rs:61-98` describes buffer-driven per-frame target and Q selection plus post-encode actual-bit feedback. `crates/rvp9-encoder/src/rate_control/postencode.rs:255-268` updates the buffer and clamps its upper bound to `maximum_buffer_size` and lower bound to `-maximum_buffer_size`.

The lower clamp is a documented divergence from libvpx: ordinary content can underflow there because this encoder has no frame-dropping path. Therefore `buffer_level >= 0` is not an established invariant. The supported underflow contract needs an explicit release decision.

Existing unit tests cover Q response to positive/negative `bits_off_target` (`cbr.rs:1169,1188`), target adjustment for underfull and overfull buffers (`cbr.rs:1374,1393`), and buffer floor/ceiling plus synthetic convergence (`postencode.rs:401,416,696,709,722`). `tests/w_vbr_rc_proof.rs:68-89` advances the controller with synthetic exact-target frame sizes; it explicitly isolates controller decisions from encoded-frame feedback. These tests establish controller behavior for their inputs, not stream-level compliance.

On exact candidate source, the Windows encoder-crate suite ran as `deltic timeout 1800 cargo test -p rvp9-encoder --locked --offline`: 885 passed, 56 existing ignored, zero failed. The ignores include diagnostic, external-oracle, and tests needing corpus inputs beyond those available or pinned `vpxdec`; this is not a complete stream-level buffer qualification. No accepted buffer trace or comparison against a pinned libvpx oracle is recorded for this candidate on either native platform.

## Acceptance still needed

Before calling this gate a pass, Arthur must define the in-scope rate-control modes, allowed buffer underflow and upper bound, measurement window, oracle, and tolerance. Then run the exact candidate and the agreed reference on the same declared inputs and settings, retain source/binary/input identities and the per-frame target, actual bits, and computed buffer trace, and evaluate every agreed limit on each required native platform. The existing `notes/lld/rate-control.md:2019,2031` discussion supplies design context, not candidate release evidence or a complete executable acceptance procedure.

Until that contract exists, controller tests and encoder-crate tests remain useful evidence but do not satisfy PLAN §2 buffer-model compliance.
