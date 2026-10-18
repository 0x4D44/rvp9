# RVP9-BUG-CRU-00001 — RVP9_FORCE_SCALAR no longer yields a scalar encoder on x86_64: new SSE2 kernels bypass SimdDispatch

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** core/dsp dispatch
- **Raised:** 2026-08-23T17:09:40Z
- **Discovery source:** Agent
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
- **State history:** Open (2026-08-23T17:09:40Z, raised via `deltic bugs new` model=claude-opus-5@high) -> Fixed (2026-10-16T02:27:37Z, deltic:auto role=fix run=fix-20261016T020804Z-09df404b branch=task/bug-RVP9-BUG-CRU-00001-run-fix-20261016T020804Z-09df404b code=d9e95399d5db09bcd7651581cbffb61806cf0823 gate=manual) -> Closed (2026-10-18T16:34:52Z, deltic:manual role=verify run=verify-20261018T162636Z-6b73a151 verified_fix_run=fix-20261016T020804Z-09df404b verdict=close)

## Observation

`RVP9_FORCE_SCALAR=1` is documented as the cross-process byte-identity oracle: encode
once forced-scalar, once native, byte-diff the bitstreams to prove no SIMD kernel moved
an encoder decision (`crates/rvp9-core/src/dsp/caps.rs:56-66`; CLAUDE.md "bypass SIMD for
differential testing"). It works by masking every ISA tier to `false` in
`X86Caps::detect()` (`crates/rvp9-core/src/dsp/caps.rs:67-79`), which only affects kernels
resolved through the runtime `SimdDispatch` table.

The five `perf(core)` commits landed 2026-08-21..23 (`faea4179f`, `755c8358f`, `edd62aa9e`,
`b6d453775`, `a39734335`) route six kernel families to SSE2 by **compile-time**
`#[cfg(all(target_arch = "x86_64", not(miri)))]` instead, with no `SimdDispatch` slot and no
runtime capability check:

- `crates/rvp9-core/src/transforms/forward.rs:506-509` → `sse2::fdct8x8`
- `crates/rvp9-core/src/transforms/forward.rs:1099-1102` → `sse2::fdct16x16`
- `crates/rvp9-core/src/mv_search/int_pro_me.rs:340-343` → `vpx_vector_var_sse2`
- `crates/rvp9-core/src/mv_search/int_pro_me.rs:716-726` → `int_pro_64x64_fused_sse2`
- `crates/rvp9-core/src/mode_decision/hadamard.rs:219-222, 235-238, 296-299` → SSE2 Hadamard
- `crates/rvp9-core/src/mode_decision/inter.rs:5580-5616` → `quantize_metrics_sse2`

Verified: `crates/rvp9-core/src/dsp/dispatch.rs` has no entry for any of
`fdct8x8`/`fdct16x16`/`hadamard_*`/`vpx_vector_var`/`int_pro`/`quantize_metrics`, and none of
the six files reference `force_scalar_env`, `X86Caps`, or `is_x86_feature_detected!`.

**Expected:** `RVP9_FORCE_SCALAR=1` selects the scalar arm of every SIMD kernel, so the
byte-identity oracle compares a genuinely scalar encode against a SIMD encode.
**Actual:** on x86_64 both legs of the oracle execute the identical SSE2 code for these six
families. A real divergence introduced into any of them is invisible to the oracle, because
there is no longer a runtime path that reaches the scalar reference.

This is a lost verification gate, not a wrong-output bug — no incorrect encode is known. It
matters because these six kernels are the newest and least-exercised code in the crate, and
the oracle was the mechanism intended to cover exactly them.

Scope note: before this batch, x86_64 ran the scalar arm for all six families, so the env var
was effectively honoured there. The same structural gap already existed on AArch64; this batch
makes it live on x86_64, the platform the release gates target (PLAN.md §2).

Note also that `scripts/qualify-conformance.sh:28,31` and `scripts/qualify-alpha-conformance.sh:25`
run a forced-scalar leg, but those gate the **decoder**; all six kernels here are encoder-only,
so those scripts never covered them and do not compensate.

Independently confirmed by four reviewers in the 2026-08-23 review pass (three lens agents plus
the lead's own read of `dispatch.rs` and `caps.rs`).

**Suggested fix:** give the six families `SimdDispatch` slots and route callers through
`crate::dsp::get_ref()`, mirroring what sibling commit `daae16aaa` did correctly for the IDCT
dispatch in the same batch. Minimum viable alternative: consult `force_scalar_env()` at the top
of each public wrapper and fall through to the scalar arm when set. Either way, correct the
now-inaccurate claims in `crates/rvp9-core/src/dsp/caps.rs:56-66` and CLAUDE.md.
See also RVP9-REQ-CRU-00006 for the structural version of this.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T16:34:52Z, deltic:auto run=verify-20261018T162636Z-6b73a151 verified_fix_run=fix-20261016T020804Z-09df404b verdict=close)

Verifier note: Manual verify: Independent x86_64 Mach-O under Rosetta/SSE2: routing oracle 1/1 with 21 backend entries passes; exact fdct8 bypass reversal triggers forced=true fdct8x8 SIMD assertion (1 expected 0). Restored oracle 1/1 and 13 numerical tests pass; exact source restored. No AVX/full-encode claim. Evidence: .scratch/cru01-proof.txt.

## Notes
