# RVP9-BUG-KILN-00010 - idct16x16 SSE2 diverges from AVX2/scalar on in-domain i12 input — possible real SSE2 saturation bug (NOT confirmed out-of-domain)

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** dsp/x86_64 idct16x16 (SSE2 256-point kernel)
- **Raised:** 2026-06-20 by Claude (green-the-build pass)
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
- **State history:** 2026-06-20 Opened by Claude from the green-the-build pass; out-of-domain hypothesis adversarially REFUTED (2/2 skeptics) — needs deeper triage. 2026-06-21 Fixed by Claude (task/fix-idct16-sse2) — out-of-domain CONFIRMED by instrumentation; test-domain fix (kernel is libvpx-faithful and correct). Awaiting two-eyes verification → Closed. -> 2026-06-23 Closed by Codex (task/20260623-TSK-HUM-verify-fixed-bugs): independently verified the original SSE2-vs-AVX2 failure is resolved as an out-of-domain test issue, not a kernel defect. `notes/divergence-ledger.md` documents the accepted out-of-domain saturation divergence, `$null | mdtimeout 300 cargo test -p rvp9-core --lib idct16x16_sse2_indomain_exact_outofdomain_saturates -- --nocapture` passed, and `$null | mdtimeout 600 cargo test -p rvp9-core --lib idct16x16_sse2 -- --nocapture` passed all 12 SSE2 tests.

## Resolution (2026-06-21)

**The prior "real bug" framing was wrong; it IS out-of-domain — now confirmed
with instrumentation (the step the prior pass deferred).** The rvp9 SSE2 kernel
is faithful to libvpx's SSE2 idct16x16 (saturating `_mm_adds_epi16` final
rounding + `_mm_packs_epi32` packs, modular `_mm_add/sub_epi16` butterfly); the
AVX2/scalar kernels are libvpx-C-faithful (modular wraplow). libvpx's own SSE2
and C diverge identically on i16-overflowing inputs.

Decisive evidence (temp diagnostic, run on this host):
- The saved failing seed is **190/256 dense**, max|coef|=2046 (the prior pass's
  "sparse seed" premise was factually wrong).
- Full seed → 1 pixel mismatch; **seed/2 → 0 mismatches** (and /3,/4,/8,/16 → 0):
  the divergence is purely magnitude/overflow-driven. An algorithmic kernel bug
  would not vanish under scaling.
- **In-domain sweep**: 2000 blocks generated as `fdct16x16` of random
  `[-255,255]` residuals (the real 8-bit coefficient domain), coefficients up to
  ±5310 — **0/2000** SSE2-vs-AVX2 mismatches.
- `sparse_12` (1-8 nz) and `eob_38` (38 nz) already PASS; only the full-density
  (256-nz) strategies failed → the low-energy-passes / high-energy-fails
  saturation signature.
- Conformance 201/201 (valid bitstreams never reach this).

Adversarially verified 2-of-3 (the dissent agreed the test-only fix is
"mechanically correct" but asked for completeness — ledger entry, a pinned
regression, instrumented evidence — all now provided).

**Fix (test-only; kernel UNCHANGED):**
- `crates/rvp9-core/src/dsp/x86_64/idct16x16_sse2_256.rs`: the out-of-domain
  full-density strategies (`uniform_12`, `eob_bucket_dense`) are replaced by one
  in-domain `indomain_dense_strategy` (`fdct16x16` of a `[-255,255]` residual);
  the redundant `eob_dense` test is removed; a directed regression
  `idct16x16_sse2_indomain_exact_outofdomain_saturates` pins in-domain
  bit-exactness AND the accepted out-of-domain saturation divergence.
- The obsolete proptest regression seed
  (`proptest-regressions/dsp/x86_64/idct16x16_sse2_256.txt`) is deleted.
- `notes/divergence-ledger.md`: documents the SSE2-saturate vs C/AVX2-modular
  divergence (out-of-domain only), superseding the T-4 HLD §6.2 bit-exactness
  expectation.

**Residual note:** `sparse_12`/`eob_38` still use arbitrary low-density i12;
their energy is far below the overflow threshold (the seed needed 190 dense
coefficients to overflow marginally), so they are not flaky in practice. If one
ever flakes, apply the same in-domain (forward-transform) treatment.

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
