# RVP9-BUG-KILN-00010 - idct16x16 SSE2 diverges from AVX2/scalar on in-domain i12 input — possible real SSE2 saturation bug (NOT confirmed out-of-domain)

- **State:** Open
- **Severity:** Medium
- **Area:** dsp/x86_64 idct16x16 (SSE2 256-point kernel)
- **Raised:** 2026-06-20 by Claude (green-the-build pass)
- **State history:** 2026-06-20 Opened by Claude from the green-the-build pass; out-of-domain hypothesis adversarially REFUTED (2/2 skeptics) — needs deeper triage.

## Observation

`dsp::x86_64::idct16x16_sse2_256::tests::idct16x16_sse2_eq_avx2_uniform_12` and
`…::idct16x16_sse2_eq_avx2_eob_dense` are RED on `origin/main` (`0b62ecc`). They
assert the SSE2 256-point inverse-DCT (`idct16x16_add_sse2_256`) is bit-identical
to the AVX2 oracle (`idct16x16_add_avx2`). The two *full-density* strategies fail;
the *sparse* strategies (`sparse_12`, `eob_38`) pass. The replayed regression seed
(`crates/rvp9-core/proptest-regressions/dsp/x86_64/idct16x16_sse2_256.txt`)
produces a single-byte output difference (e.g. SSE2=255 vs AVX2=0 at one pixel).

The two kernels use different overflow policies at the wraplow sites: SSE2
saturates (`_mm_packs_epi32`, clamps to ±i16::MAX); AVX2 uses modular wraplow
(`x as i16 as i32`, matching the scalar reference and libvpx C).

## Notes

**Status: root cause NOT confirmed — do not "fix" by narrowing the test.**

The initial hypothesis — "full-density ±2047 input drives intermediates past
i16::MAX, where saturate-vs-wrap legitimately diverges, so it is an out-of-spec
test-domain issue" — was **adversarially REFUTED** (2/2 skeptics, high confidence):
1. The shrunk regression seed is **sparse, not full-density**, so full-density is
   not the necessary trigger ("fail ONLY because full-density" is false).
2. The Track T-4 HLD (§6) explicitly lists `uniform_12` and `eob_bucket_dense` as
   strategies to **include**, expecting bit-exactness; the kernel commit (e7b90f7)
   claimed "all 4 strategies × 256 cases bit-exact"; the regression-seed commit
   (108060b) treated these as **unexpected** failures, not documented limitations.
3. The AVX2/scalar path is the spec-faithful one (modular wraplow). If SSE2
   diverges from it on an input whose intermediates do **not** actually overflow
   i16, that is a **real SSE2 correctness bug** — the SSE2 decode path would
   produce wrong pixels vs the spec on an SSE2-only host.

Because conformance passes 201/201, runtime AVX2 hosts are unaffected; the risk is
on SSE2-only hosts (or wherever the SSE2 256-point kernel is dispatched).

Next step (triage, not done here): instrument `idct16x16_add_sse2_256` to record
whether ANY intermediate actually saturates for the failing seed.
- If an intermediate genuinely exceeds i16::MAX → the input is out-of-spec and the
  proptest strategy domain should be tightened (with a derived bound) and recorded
  in `notes/divergence-ledger.md`.
- If NO intermediate overflows → the SSE2 kernel has a real saturation/rounding
  bug to fix so it matches the modular-wraplow spec output.
