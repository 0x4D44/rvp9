# Wave G F12 (deferred) — plane_dequant_pair entropy-fused refactor

**Date filed**: 2026-05-22
**Filed by**: Claude (supervisor, Wave F autonomous lead)
**Status**: BLOCKER — requires entropy-decoder architectural rewrite,
out of Wave F scope.

## What F12 originally asked for

The Wave F catalog (from Wave E close-out journal) included:

> F12 — plane_dequant_pair entropy-fused refactor

Intent: separate dequant arithmetic from the entropy-decode loop so it
can be vectorised in a batchable AVX2 kernel via the
`SimdDispatch::plane_dequant_pair` slot.

## Why it is blocked

The slot's own docstring at `crates/rvp9-core/src/dsp/dispatch.rs:301-310`
already explains the architectural barrier verbatim:

> **Currently scalar-only**: architecturally there is no batchable
> dequant kernel in the VP9 decoder because per-coefficient dequant
> (`v = (value * dqv) >> dq_shift`) is fused into the bool-coder loop in
> `decode_coefs` (see `crates/rvp9-core/src/entropy/detokenize.rs:415`).
> The slot here reserves the family name for a future entropy-decoder
> rewrite that would separate dequant into a vectorisable pass.

Inspection of `entropy/detokenize.rs:380-450` confirms the architecture:

```rust
// detokenize.rs:415
v = (val * dqv) >> dq_shift;
// detokenize.rs:425
v = (val * dqv) >> dq_shift;
// detokenize.rs:432
v = (three_or_four * dqv) >> dq_shift;
// detokenize.rs:435
v = (2 * dqv) >> dq_shift;
// detokenize.rs:444
v = dqv >> dq_shift;
```

Each of the five token-category arms applies its own dequant
arithmetic INSIDE the bool-coder `reader.read(...)` chain, which is
inherently sequential (Markov state machine — each read modifies the
arithmetic-coder state, the next read consumes it).

## Why "just batch the dequant after entropy decode" doesn't work

You could imagine deferring dequant: have `decode_coefs` write raw
`val` per coefficient into a scratch buffer, then run a separate AVX2
pass that does `for c in 0..eob { dq_coef[c] = (val[c] * dqv) >> dq_shift; }`.

Two problems:

1. **DC vs AC dequant**: `dqv` differs at scan position 0 (DC) vs
   positions 1..eob (AC). The vectorised pass would need to branch on
   the scan position, or write the dequant value alongside each coef
   (doubling scratch memory). Either approach loses the simplicity
   gain.

2. **Sign bit interleave**: `decode_coefs:448` reads the sign bit AFTER
   dequant via `reader.read_bit()`. The sign read is inside the
   sequential bool-coder loop too. To separate dequant from the loop,
   you also need to separate sign reads — but those are entropy-coded,
   not just inline data. Means you'd need to either (a) split the loop
   in two (one for raw vals + signs, one for dequant) or (b) keep dequant
   in the loop. (b) is the current state.

A meaningful refactor would touch:
- `decode_coefs` (split into two passes)
- The token cache management
- Scratch buffer ownership (callers would receive raw vals + dequant
  pair separately)
- Every consumer that takes the dequantised output today

Estimated scope: 3-5 day rewrite touching ~10 files across rvp9-core
and rvp9-decoder. Bit-exactness gate requires careful audit of the
two-pass equivalence (especially sign-bit ordering).

## Recommendation for Wave G

Two paths:

### Option A (proper fix): Split-loop refactor

Implement the two-pass entropy decode + dequant separation. Requires:
1. Profile-gated decision: is the existing fused approach actually
   on a critical path? (Currently the bool-coder dominates; dequant
   is ~2 mul-shift per coef, negligible vs the read chain.)
2. Two-pass rewrite preserving bit-exactness.
3. AVX2 dequant kernel via the slot.

Wave G estimate: 2-3 day implementer + 1 day reviewer.

### Option B (close the slot): Document and retire

If profiling shows dequant arithmetic is not a hot path, retire the
slot:
1. Remove `SimdDispatch::plane_dequant_pair`.
2. Update the slot table doc to drop the row.
3. Note in dispatch.rs that the quant family has no SIMD slot
   because dequant is fused into entropy decode.

Less work than Option A; appropriate if perf doesn't warrant the
architectural complexity.

## Action

F12 closed for Wave F as Wave G blocker. The slot remains scalar-only
with the existing accurate docstring. Future maintainers reading the
slot doc + this file will have full context.
