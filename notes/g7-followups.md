# G7.1 follow-up catalog

Filed 2026-05-21 as part of the G7.0 review (commit on top of `28d3ff72`).
The supervisor reclassified the original G7 commit (`28d3ff72`) to G7.0 /
G7-prep per reviewer C1 — the present commit lands the plumbing + detection
gate; the items below are explicitly deferred to G7.1.

## G7.1a — geometry refactor (unit-testable now)

These sites all encode the 4:2:0 (`ss_x = ss_y = 1`) assumption directly
in code. The G7.1a refactor must replace each with a per-frame lookup
driven by the active `ChromaFormat`. The DA argued (and the reviewer
concurred with a Goodhart-target caveat) that this work is unit-testable
in isolation, without a libvpx differential oracle — provided the tests
cover all four chroma formats and don't lock in geometry that turns out
to mismatch libvpx when G7.1b lands. Approach: cover the geometry LUT
correctness with property-style tests against
`(y_dim + ss) >> ss = uv_dim`.

### Inside `rvp9-decoder`

- `crates/rvp9-decoder/src/tile.rs:305-306` —
  `SUBSAMPLING_X / SUBSAMPLING_Y: [usize; 3] = [0, 1, 1]`. Consumed by
  the per-block decode chain (lines ~826, ~938 and the `uv_plane_*`
  helpers).
- `crates/rvp9-decoder/src/reconstruct_inter.rs:67-68` — symmetric pair
  used by the inter-block MC + residual loop and `RefPlaneCrop::for_plane`.

### Inside `rvp9-core`

- `crates/rvp9-core/src/coef_ctx.rs:35-36` —
  `PROFILE0_SUBSAMPLING_X / PROFILE0_SUBSAMPLING_Y: [usize; MAX_MB_PLANE] = [0, 1, 1]`.
  Consumed at lines 153-154 for the `above_coef` allocation. Note the
  explicit `PROFILE0_` prefix — refactor must rename and route through
  the per-frame chroma format.
- `crates/rvp9-core/src/loopfilter/apply.rs::filter_block_plane_ss11`
  (line ~245) — the per-SB chroma origin `(mi_row * MI_BLOCK_SIZE / 2)
  * stride + (mi_col * MI_BLOCK_SIZE / 2)` hardcodes `ss_x = ss_y = 1`.
  Function name `_ss11` is load-bearing; refactor must add an `_ss00`
  / `_ss10` / `_ss01` family (matching libvpx
  `vp9_filter_block_plane_ss00`) and dispatch on chroma format from
  the caller `loop_filter_frame`.

### Inside `rvp9-encoder` (symmetric work)

- `crates/rvp9-encoder/src/lib.rs:1004-1005` —
  `F5_SUBSAMPLING_X / F5_SUBSAMPLING_Y: [usize; 3] = [0, 1, 1]`. Same
  pattern as decoder; refactor when encoder gains Profile 1.

## G7.1b — parser propagation + e2e fixtures (blocked on vpxenc)

### Parser propagation

- `crates/rvp9-core/src/bitstream/header.rs:~368` — inter-frame chroma
  defaulting hardcodes `ChromaFormat::Yuv420`. The G7.0 gate's
  invariant chain depends on this: a Profile 1 inter frame today
  reports chroma=Yuv420 from the parser, so the gate doesn't fire on
  it; the keyframe rejection cascade (no Profile 1 keyframe decoded →
  no ref slot populated → inter frame fails at `EmptyInterRef`)
  protects correctness. When G7.1 lifts the gate, the parser MUST
  start propagating the keyframe's chroma forward across inter frames
  in the same commit, or chroma allocation will silently mismatch.

### Differential vectors

- `vpxenc` is not on `PATH` in this sandbox. Generating fresh Profile 1
  vectors (`vp90-2-22-svc_1280x720_3.ivf`-style 4:2:2 / 4:4:0 / 4:4:4
  IVF + matching `.md5` digests) must happen on a host with libvpx
  tools installed. No Profile 1 IVF fixtures are present in-tree as of
  this commit.
- Per HLD V6 §G7 the differential corpus is required for G7.1
  acceptance — locking in geometry by unit test alone would be a
  Goodhart target (the reviewer's caveat). Bit-exact decoders need
  bit-exact oracles.

## Cross-cutting

- Once G7.1a + G7.1b land, the G7.0 gate at
  `crates/rvp9-decoder/src/lib.rs::Decoder::decode_frame` is a one-line
  removal; remove the `UnsupportedProfile` arm at the same time.
- The defensive `debug_assert_eq!(chroma == Yuv420)` checks at
  `decode_tile_intra`, `decode_tile_inter`, `reconstruct_inter_block`
  and `loop_filter_frame` will need their conditions broadened (or
  removed) when the per-block chain gains per-chroma dispatch.
- The K-5 harness `total_gated` counter at
  `crates/rvp9-harness/tests/conformance.rs` should reach zero once
  Profile 1 vectors are added and pass; if a future feature gates on
  a different axis (HBD?), file a sibling variant rather than reusing
  `UnsupportedProfile`.
