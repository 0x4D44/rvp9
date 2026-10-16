# RVP9-BUG-KILN-00014 - Temporal SVC base frames can depend on a dropped enhancement frame's MV grid

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** encoder/temporal-SVC prev-MV state
- **Raised:** 2026-07-05
- **Owner:** -
- **Owner role:** -
- **Owner run:** -
- **Owner host:** -
- **Owner branch:** -
- **Owner base:** -
- **Owner fingerprint:** -
- **Owner since:** -
- **Owner until:** -
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-05, raised by Codex code-review pass over `crates/rvp9-encoder/` at `a86e58c260ce`) -> Fixed (2026-07-31, deltic:auto role=fix run=fix-20260730T224959Z-p22973-n531556000-c1 branch=task/bug-RVP9-BUG-KILN-00014-run-fix-20260730T224959Z-p22973-n531556000-c1 code=2ed53a00949ba2158b90e14e5ab323023ddf8a70 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

Temporal SVC marks enhancement frames as non-reference by setting `refresh_frame_flags = 0x00` at `crates/rvp9-encoder/src/frame_loop.rs:2513`. The inter encode path nevertheless commits every inter frame's current MV grid to `self.prev_mv_grid` at `crates/rvp9-encoder/src/frame_loop.rs:3213` and sets `last_show_frame = true` at `crates/rvp9-encoder/src/frame_loop.rs:3218`.

The next base-layer frame can therefore code MV residuals using previous-frame MV context from an enhancement frame that a base-layer-only decoder legitimately dropped. That decoder would keep the older previous-frame MV grid, so encoder and decoder can derive different reference MVs.

## Fix

Open. Temporal SVC should ensure base-layer frames do not depend on enhancement-layer previous-MV state. Possible fixes are to keep a base-layer-only previous-MV grid, or force the previous-MV salvage gate closed for base frames after droppable layers. Add a test that decodes only `temporal_layer_id == 0` packets.

## Notes

Confirmed by three skeptic passes in the 2026-07-05 encoder review.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Root cause closed structurally rather than papered over: error-resilient mode on SVC base
frames closes the decoder's `use_prev_frame_mvs` gate, and the decoder-side
`vp9_setup_past_independence` mirror resets FC banks, segment IDs and LF deltas in libvpx's
order (`crates/rvp9-encoder/src/frame_loop.rs:2565`, `:2726`, `:4901`;
`crates/rvp9-decoder/src/lib.rs:2644-2662`).
`temporal_svc_base_layer_decodes_identically_when_enhancement_is_dropped` is a true
differential oracle -- two independent decoders, one fed only `tid == 0` packets,
reconstructions compared -- and asserts every inter frame carries `error_resilient`, so it
cannot pass vacuously. Observed passing.
Residuals noted, not blocking: the encoder still commits every inter frame's MV grid to
`self.prev_mv_grid` including droppable enhancement frames (harmless only because the gate is
now closed), and the BD-rate cost of the per-frame FC reset on SVC streams is unmeasured.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
