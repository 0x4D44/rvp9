# Wave G G7 prototype — ONE_TOKEN-only two-pass dequant split

**Date filed**: 2026-05-22
**Author**: Claude (Wave G autonomous agent)
**Status**: PROTOTYPE landed — extension to remaining token arms deferred
to Wave H.

## What landed

A proof-of-concept split of the `decode_coefs` bool-coder loop into
two passes for the ONE_TOKEN branch only. Lives in
`crates/rvp9-core/src/entropy/detokenize.rs:325-360` (scratch buffers),
`:438-516` (pass-1 deferral inside the loop), and `:495-515` (pass-2
arithmetic loop after the bool-coder completes).

**Bit-exactness**: holds. Conformance corpus is 201 / 0 / 0 / 0 / 201
unchanged. All 40 detokenize unit tests pass, including the
`differential_fuzz_decode_coefs_vs_libvpx_ref` fuzz test (N=100
randomised inputs comparing the production decoder against a verbatim
libvpx reference port) and the `RVP9_DEBUG_REFCOEFS=1` runtime
differential gate that runs during conformance.

## How it works

The decode_coefs loop is a Markov state machine: each `BoolReader::read`
mutates the arithmetic-coder state, and the next read consumes it. The
dequant arithmetic `v = (val * dqv) >> dq_shift` and writeback to
`dq_coefs_out[raster_idx]` are NOT bool-coder reads — they are pure
side effects of the per-position state. They can therefore be deferred
to a second pass without affecting the entropy decode.

The sign bit (`reader.read_bit()` at the original line 448) IS entropy
coded and must stay inside the bool-coder loop. The prototype captures
the sign bit in pass 1 and applies it during the pass-2 negation.

### Pass 1 (inside the bool-coder loop)

For ONE_TOKEN positions:
- Skip the `v = dqv >> dq_shift` computation.
- Still read the sign bit (entropy coded — cannot defer).
- Push `(raster_idx, dqv, sign_bit)` onto three parallel scratch vectors.
- Leave `dq_coefs_out[raster_idx]` at the function-prologue zero.

For all other arms (CAT1..6, TWO/THREE/FOUR), the existing fused path
is preserved verbatim — `v` is computed, signed, and written immediately.

### Pass 2 (after the bool-coder loop terminates)

A vectorisable loop:

```rust
for i in 0..one_token_dqv.len() {
    let dqv_i = i32::from(one_token_dqv[i]);
    let raster_idx = one_token_raster[i] as usize;
    let sign = one_token_sign[i];
    let mag = dqv_i >> dq_shift;
    dq_coefs_out[raster_idx] = if sign != 0 { -mag } else { mag };
}
```

Each iteration is independent. LLVM should auto-vectorise this trivially;
an AVX2 hand-lift via the `SimdDispatch::plane_dequant_pair` family slot
would be straightforward (gather raster idx, broadcast dq_shift, scatter
result).

### Scratch buffer cost

3 × `Vec` with `seg_eob`-sized capacity, total `~4 * seg_eob` bytes per
decode_coefs call. For a 32x32 block: 4 KiB allocation per call. That's
not free — production extension should hoist the scratch into a per-tile
arena owned by the `TileCtx`, allocated once and reused. The prototype
allocates inline to keep the diff minimal.

## What the prototype does NOT prove

- **CAT1..6 / TWO / THREE / FOUR arms still use the fused path.** Their
  `v` computation depends on `val` (which is itself the result of more
  bool-coder reads for the CAT extension bits), so deferring them
  requires either:
  - A wider scratch entry per position (val + dqv + raster + sign) —
    ~10 bytes per position vs the ONE_TOKEN's 5 bytes.
  - Or a tagged-union scratch (variable-width entries with a 1-byte
    discriminant + per-arm payload), which complicates the pass-2
    vectorisation.
- **The `plane_dequant_pair` SIMD slot is still scalar-only.** The
  slot's docstring at `dsp/dispatch.rs:301-310` remains accurate. To
  wire AVX2, the slot's signature needs to change from
  `fn(q_index, deltas) -> (i16, i16)` (a per-frame LUT lookup) to a
  per-block kernel `fn(&scratch, dq_shift, &mut [i32])`. That's a
  family-rename, not just a kernel swap.

## Extension to all token arms (Wave H scope)

Estimated dev-hours for the full refactor:

| Phase | Work | Hours |
|-------|------|-------|
| 1 | Tagged-union scratch design + size analysis | 4 |
| 2 | Pass-1 split for CAT1..6 arms (4 arms × `val` capture + `dqv` capture) | 8 |
| 3 | Pass-1 split for TWO/THREE/FOUR arms (3 arms, shared scratch slot) | 4 |
| 4 | Pass-2 unified loop handling all arms (per-discriminant dispatch) | 6 |
| 5 | Hoist scratch into TileCtx arena, lifetime threading | 6 |
| 6 | `SimdDispatch::plane_dequant_pair` family rename + scalar reference impl | 4 |
| 7 | AVX2 kernel for pass 2 (gather/scatter for raster_idx, vectorised arithmetic) | 12 |
| 8 | Bit-exactness validation: fuzz test all-arms path, conformance regression | 4 |
| 9 | Caller-side API impact (`tile.rs`, `reconstruct_inter.rs` may need scratch lifetime) | 4 |
| 10 | Bench harness: measure pass-1 vs pass-2 cost split, confirm AVX2 win | 4 |
|   | **Total** | **~56 hours** |

This is **~7 dev-days of single-engineer work**, not the 3-5 calendar
days the original F12 estimate gave. The single-session-agent path is
infeasible without scope cuts.

## Recommended Wave H sequencing

1. **G7 prototype review + commit** (this work).
2. **G7.1 — extend to TWO_TOKEN + FOUR_TOKEN** (the two arms with no
   `val` capture — same shape as ONE_TOKEN). +2-4 hours, exercises the
   discriminant-based pass-2 dispatch.
3. **G7.2 — extend to THREE_TOKEN** (single extra bool-coder read for
   `three_or_four`). +2 hours.
4. **G7.3 — extend to CAT1..6** (the big one — variable `val` width,
   most complex pass-1 capture). +8-16 hours.
5. **G7.4 — TileCtx arena, kill the per-call `Vec` allocations**. +6
   hours.
6. **G7.5 — `plane_dequant_pair` family rename + AVX2 kernel**. +16
   hours.
7. **G7.6 — bench, conformance, ship**. +4 hours.

Total Wave H budget for the full G7: ~40-50 hours (less than the
greenfield estimate because the prototype proves the architecture; the
remaining work is mechanical extension + AVX2).

## Files touched by the prototype

- `crates/rvp9-core/src/entropy/detokenize.rs` — only file changed.
  Diff is +60 lines (scratch declarations + pass-2 loop + ONE_TOKEN
  branch rewrite).

No encoder files, no mode_decision files, no decoder-crate files
touched.

## How to verify

```bash
cd <worktree>
cargo test -p rvp9-core --lib entropy --release
cargo test -p rvp9-core --lib detokenize --release
cargo build --workspace --release
cargo run --release --bin conformance -p rvp9-harness \
    -- --vectors-dir tests/conformance/vectors 2>&1 | tail -3
```

Expected: 112 entropy tests pass, 40 detokenize tests pass, conformance
holds at 201 / 0 / 0 / 0 / 201.

## How to extend (next-engineer cheat sheet)

1. Pick the next-simplest arm — TWO_TOKEN at `detokenize.rs:434-436`.
2. Add a discriminant byte to scratch (currently sign is single byte;
   pack discriminant into upper bits OR add a fourth parallel vec).
3. Capture `(raster_idx, dqv, sign, discriminant=TWO)`. Pass-2
   computes `mag = (2 * dqv) >> dq_shift`.
4. Verify with `cargo test -p rvp9-core --lib detokenize --release` —
   the fuzz test will catch any divergence within a few iterations.
5. Repeat for THREE/FOUR/CAT* arms; complexity rises but the pattern
   is the same.
