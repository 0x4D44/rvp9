# RVP9-BUG-FLU-00048 — Safe x86 SIMD dispatch getters expose kernels with debug-only bounds

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** core/dsp x86 SIMD
- **Raised:** 2026-10-17T20:46:57Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261018T153528Z-cff45905
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00048-run-fix-20261018T153528Z-cff45905
- **Owner base:** 09697b558205594dcce310c8047827756597fb1a
- **Owner fingerprint:** -
- **Owner since:** 2026-10-18T15:35:28Z
- **Owner until:** 2026-10-18T19:35:28Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-10-17T20:46:57Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium)

## Observation

On x86_64 hosts where normal runtime dispatch selects AVX2, safe external code can obtain the public table with rvp9_core::dsp::get() and call (dispatch.dc_predictor_4x4())(&mut [], 4, &[], &[]). The getter at crates/rvp9-core/src/dsp/dispatch.rs:2899 returns a safe function pointer. Its AVX2 wrapper at crates/rvp9-core/src/dsp/x86_64/dc_pred_4x4_avx2.rs:118-125 relies on dispatch only for ISA availability and does not validate slice lengths. The inner debug_assert checks at lines 71-73 disappear in release; read_unaligned at lines 76-77 then reads four bytes from each empty input slice, followed by raw output writes at line 105. Expected: a safe API call cannot reach out-of-bounds memory access. Actual: the accepted safe call enters raw reads and writes outside the supplied slices in optimized builds. This is a source-derived finding; no runtime reproduction was attempted. Fixed RVP9-BUG-KILN-00020 covered the former direct V_PRED surface, not this remaining dispatch getter surface.

## Fix

<unfixed — raised only>

## Notes
