# RVP9-BUG-FLU-00043 — Conformance forced-scalar rows retain NEON dispatch on AArch64

- **State:** Fixed
- **Priority:** Must
- **Severity:** Medium
- **Area:** qualification/conformance
- **Raised:** 2026-10-16T14:14:38Z
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
- **State history:** Open (2026-10-16T14:14:38Z, raised via `deltic bugs new`) -> Fixed (2026-10-18T16:24:54Z, deltic:auto role=fix run=fix-20261018T155935Z-dd09e04f branch=task/bug-RVP9-BUG-FLU-00043-run-fix-20261018T155935Z-dd09e04f code=76481914b3c37b7028df3cb1bc9ab5e293365e23 gate=manual)

## Observation

At native Apple M5 Max candidate ebca64b0, scripts/qualify-conformance.sh labels two rows forced-scalar and executes them by setting RVP9_FORCE_SCALAR=1. A diagnostic compiling the actual current dsp/caps.rs and calling AArch64Caps::detect reports AArch64Caps { neon: true, dotprod: true } both without the variable and with its value1, exit0 both. Source AArch64Caps::detect does not consult the environment; decoder initialization builds dispatch from detected capabilities. Thus script success on these rows cannot establish its declared scalar decoder coverage on supported macOS Apple Silicon. Expected: a scalar-labelled acceptance row actually selects scalar, or the gate explicitly reports unsupported coverage and cannot count a scalar pass. Actual: it repeats native dispatch while advertising forced-scalar. Current prior report claimed four-mode macOS qualification; that scalar claim is now withdrawn. This is a qualification evidence defect, not a demonstrated pixel mismatch. CRU-00001 repaired x86 direct encoder routing and explicitly leaves Arm outside the hook; this report concerns separate AArch64 decoder conformance false coverage. REQ-CRU-00006 dispatch unification does not supply current scalar acceptance. Diagnostic source/commands/output are retained in readiness candidate-20261016 evidence. No fix or acceptance workaround performed.

## Fix

AArch64 capability detection now honors exact `RVP9_FORCE_SCALAR=1`, and all three public IDCT16 wrappers use cached dispatch capabilities instead of entering NEON unconditionally. Their existing scalar helper is compiled in ordinary Arm library builds. Explicit ISA entry points and Arm encoder direct kernels retain their prior scope.

Regression evidence: isolated capability-routing and actual NEON-entry trace parent tests both failed before the fixes, then passed after. The trace covers all three EOB buckets through direct and dispatch-getter calls; each child must execute exactly one selected test. Non-trigger values unset/0/true/01 preserve native capabilities. Existing NEON/scalar IDCT16 parity, normal decoder library check, targeted Clippy, formatting and diff checks passed.

Focused native/forced-scalar decoder smoke at one/two threads selected two official vectors and passed all 20 frames per run, with zero mismatch or skip. This does not retroactively validate prior scalar-labelled runs or claim a fresh full-vector gate.

Evidence: `wrk_journals/2026.10.18 - JRN - RVP9-BUG-FLU-00043 Arm decoder scalar routing.md`. Separate forward-transform Miri issue RVP9-BUG-FLU-00037 remains Open. Incidental native IDCT16 bounds overflow was recorded separately as RVP9-BUG-FLU-00056 without folding its repair into this fix.

## Notes

At exact revision d5947f2eccba2c1dd75e2f235df095d75d7af9fb on flux (macOS 26.6.2, arm64), the official 307-vector conformance runner was rebuilt and invoked with RVP9_FORCE_SCALAR=1 at one and two threads. Both runs passed 4,816 frames with zero mismatches, but the source confirms force_scalar_env is cfg-gated to x86_64 while AArch64Caps::detect uses native feature detection. These are native-dispatch results and do not provide scalar coverage. Runner SHA-256 and raw logs are recorded in [the exact-revision reproduction](../wrk_docs/evidence/2026-10-15-release-readiness/recovered-validation-20261017/postfix-main-qualification/macos-current-source-forced-scalar-conformance-20261018/qualification.json). The bug remains Open; no fix or status transition occurred.
