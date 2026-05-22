# Wave H — loopfilter AVX2 gate (H1 follow-up)

**Date filed**: 2026-05-22
**Filed by**: Claude (Wave H autonomous supervisor)
**Status**: ACTIVE — AVX2 loopfilter kernels gated OFF pending diagnosis & fix

## What happened

Wave H H1 wired `crates/rvp9-core/src/loopfilter/apply.rs` through the
`SimdDispatch` slots, replacing direct calls to `crate::loopfilter::lpf_*`
with `dispatch.lpf_X()(...)`. This activated the F10 + G3-G6 AVX2
loopfilter wrappers in production for the first time (Wave F/G had
populated the dispatch table but the loopfilter apply driver was
calling scalars directly, so the AVX2 wrappers were dead code).

Conformance regression: **58 mismatched / 201 total** (was 0/201).

## Root cause

The F10 + G3-G6 AVX2 loopfilter kernels passed their per-kernel unit
tests (`crates/rvp9-core/src/dsp/x86_64/lpf_*_avx2.rs` test modules)
but diverge from scalar on at least 58 of the conformance corpus's
201 vectors. The unit tests apparently check specific input patterns
that miss the divergent edge cases.

The Wave F/G journals claim "first-try bit-exactness on all 6
implementer rounds (G1-G6)" — that claim was based on unit-test
parity, not conformance-corpus parity. H1 surfaces the gap.

## Affected kernels (single-variant)

All 6 single-variant loopfilter AVX2 kernels are suspect:

| Kernel | Wave | File |
|--------|------|------|
| `lpf_vertical_4_avx2` | F10 | `crates/rvp9-core/src/dsp/x86_64/lpf_vertical_4_avx2.rs` |
| `lpf_horizontal_4_avx2` | G3 | `crates/rvp9-core/src/dsp/x86_64/lpf_horizontal_4_avx2.rs` |
| `lpf_vertical_8_avx2` | G4 | `crates/rvp9-core/src/dsp/x86_64/lpf_vertical_8_avx2.rs` |
| `lpf_horizontal_8_avx2` | G5 | `crates/rvp9-core/src/dsp/x86_64/lpf_horizontal_8_avx2.rs` |
| `lpf_vertical_16_avx2` | G6 | `crates/rvp9-core/src/dsp/x86_64/lpf_vertical_16_avx2.rs` |
| `lpf_horizontal_16_avx2` | G6 | `crates/rvp9-core/src/dsp/x86_64/lpf_horizontal_16_avx2.rs` |

It is not yet known which subset of the 6 is responsible — bisection
needed (see action plan below).

## Mitigation (H1 ships)

Added a runtime gate in `SimdDispatch::build`: the AVX2 paths for
the 6 single-variant lpf_* slots are taken **only** when the env var
`RVP9_DSP_LOOPFILTER_AVX2=1` is set. Default is unset → scalar.

This means:

- H1 wiring ships (apply.rs goes through dispatch slots).
- Production behavior unchanged (scalar everywhere, conformance 201/0/0/0/201).
- Development can enable AVX2 paths by exporting the env var to
  reproduce / bisect the failures.

The gate is **temporary**. Remove it once the kernels are
conformance-clean.

## Action plan (Wave H follow-up — H7 or similar)

1. **Bisect**: with `RVP9_DSP_LOOPFILTER_AVX2=1`, individually swap
   each AVX2 kernel back to scalar (by editing `dispatch.rs::build`)
   and re-run conformance. Find the minimal failing subset.
2. **Differential-test**: capture inputs from a failing conformance
   vector — feed identical inputs to both scalar and AVX2, hash
   outputs, find the first diverging frame / block / row.
3. **Fix the kernel**: re-derive the suspicious AVX2 path against
   libvpx `loopfilter_avx2.c` byte-by-byte.
4. **Add a corpus-level test**: a Rust test that runs scalar vs AVX2
   on a representative set of border-edge inputs (e.g. random
   pixels + random thresholds × N iterations). This would catch the
   divergence before another conformance regression.
5. **Remove the gate**: once all 6 kernels pass conformance, delete
   `loopfilter_avx2_enabled` and the env-var check from
   `SimdDispatch::build`, restore the per-kernel
   `forced_caps_installs_correct_lpf_*` tests.

## Why the unit tests missed it

The per-kernel unit tests typically cover:

- Identity (all-zero, all-constant inputs)
- A single step edge with bounded threshold
- A few hand-picked random patterns

Real VP9 streams hit edge cases the unit tests miss:

- Pixel values near saturation (0, 255)
- Thresholds at boundary values (0, 1, 63)
- Adjacent blocks with very different `filter_level` settings
- The 16-tap "wide" cascade firing on flat-mask boundaries
- Specific bit patterns in `flat_mask4` / `flat_mask5` computation

A property-based fuzzer (proptest / quickcheck) over the input
domain would be the natural fit for catching this class of bug.

## Cost estimate

- Bisection: ~30 min (six builds + six conformance runs)
- Fix per kernel: 1-4 hours (depends on root cause — sign error,
  off-by-one in pack saturation, etc.)
- Corpus test harness: 1-2 hours
- Total: ~8-24 dev-hours depending on how many kernels are broken
