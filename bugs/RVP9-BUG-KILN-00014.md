# RVP9-BUG-KILN-00014 - Temporal SVC base frames can depend on a dropped enhancement frame's MV grid

- **State:** Open
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
- **State history:** Open (2026-07-05, raised by Codex code-review pass over `crates/rvp9-encoder/` at `a86e58c260ce`)

## Observation

Temporal SVC marks enhancement frames as non-reference by setting `refresh_frame_flags = 0x00` at `crates/rvp9-encoder/src/frame_loop.rs:2513`. The inter encode path nevertheless commits every inter frame's current MV grid to `self.prev_mv_grid` at `crates/rvp9-encoder/src/frame_loop.rs:3213` and sets `last_show_frame = true` at `crates/rvp9-encoder/src/frame_loop.rs:3218`.

The next base-layer frame can therefore code MV residuals using previous-frame MV context from an enhancement frame that a base-layer-only decoder legitimately dropped. That decoder would keep the older previous-frame MV grid, so encoder and decoder can derive different reference MVs.

## Fix

Open. Temporal SVC should ensure base-layer frames do not depend on enhancement-layer previous-MV state. Possible fixes are to keep a base-layer-only previous-MV grid, or force the previous-MV salvage gate closed for base frames after droppable layers. Add a test that decodes only `temporal_layer_id == 0` packets.

## Notes

Confirmed by three skeptic passes in the 2026-07-05 encoder review.
