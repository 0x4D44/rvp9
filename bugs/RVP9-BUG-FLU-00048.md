# RVP9-BUG-FLU-00048 — Safe x86 SIMD dispatch getters expose kernels with debug-only bounds

- **State:** Closed
- **Priority:** Must
- **Severity:** High
- **Area:** core/dsp x86 SIMD
- **Raised:** 2026-10-17T20:46:57Z
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
- **State history:** Open (2026-10-17T20:46:57Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium) -> Fixed (2026-10-18T15:50:59Z, deltic:auto role=fix run=fix-20261018T153528Z-cff45905 branch=task/bug-RVP9-BUG-FLU-00048-run-fix-20261018T153528Z-cff45905 code=6d2c2ba9e1f53f51085163e43c48af7cd7693010 gate=manual) -> Closed (2026-10-18T17:45:24Z, deltic:manual role=verify run=verify-20261018T173454Z-558f6634 verified_fix_run=fix-20261018T153528Z-cff45905 verdict=close)

## Observation

On x86_64 hosts where normal runtime dispatch selects AVX2, safe external code can obtain the public table with rvp9_core::dsp::get() and call (dispatch.dc_predictor_4x4())(&mut [], 4, &[], &[]). The getter at crates/rvp9-core/src/dsp/dispatch.rs:2899 returns a safe function pointer. Its AVX2 wrapper at crates/rvp9-core/src/dsp/x86_64/dc_pred_4x4_avx2.rs:118-125 relies on dispatch only for ISA availability and does not validate slice lengths. The inner debug_assert checks at lines 71-73 disappear in release; read_unaligned at lines 76-77 then reads four bytes from each empty input slice, followed by raw output writes at line 105. Expected: a safe API call cannot reach out-of-bounds memory access. Actual: the accepted safe call enters raw reads and writes outside the supplied slices in optimized builds. This is a source-derived finding; no runtime reproduction was attempted. Fixed RVP9-BUG-KILN-00020 covered the former direct V_PRED surface, not this remaining dispatch getter surface.

## Fix

Safe x86 dispatch wrappers now check slice footprints and overflow before raw SIMD access, including the reported DC4 function-pointer getter. Coverage extends across transforms, predictors, LPF, convolution, distortion metrics and dequantization. Public signatures and dispatch selection remain unchanged.

Regression evidence and exact commands: `wrk_journals/2026.10.18 - JRN - RVP9-BUG-FLU-00048 safe SIMD dispatch bounds.md`. A safe zero-stride SSE2 canary failed on the expected assertion with its guard removed and passed after restoration. Shared footprint guard reversals also failed for short/overflowing storage. Forced-AVX2 dispatch getter tests reject invalid DC4 inputs before ISA entry.

Host limit: Apple Silicon Rosetta reports AVX2 unavailable. Release-semantics rejection checks execute; valid AVX2 differential execution and performance remain unmeasured. Separate IDCT32 final-pointer defect RVP9-BUG-FLU-00050 is not addressed here.

### Verification summary (2026-10-18T17:45:24Z, deltic:auto run=verify-20261018T173454Z-558f6634 verified_fix_run=fix-20261018T153528Z-cff45905 verdict=close)

Verifier note: Manual verify: Retained public-getter bounds test passed, failed at its bounds assertion after guard reversal, and passed after exact restore; forced AVX2 getter rejection tests passed 4/4. Mutation sentinel returned before unsafe SSE2. Executed x86_64 under Rosetta; no AVX2 execution.

## Notes
