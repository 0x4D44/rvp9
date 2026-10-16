# RVP9-BUG-FLU-00043 — Conformance forced-scalar rows retain NEON dispatch on AArch64

- **State:** Open
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
- **State history:** Open (2026-10-16T14:14:38Z, raised via `deltic bugs new`)

## Observation

At native Apple M5 Max candidate ebca64b0, scripts/qualify-conformance.sh labels two rows forced-scalar and executes them by setting RVP9_FORCE_SCALAR=1. A diagnostic compiling the actual current dsp/caps.rs and calling AArch64Caps::detect reports AArch64Caps { neon: true, dotprod: true } both without the variable and with its value1, exit0 both. Source AArch64Caps::detect does not consult the environment; decoder initialization builds dispatch from detected capabilities. Thus script success on these rows cannot establish its declared scalar decoder coverage on supported macOS Apple Silicon. Expected: a scalar-labelled acceptance row actually selects scalar, or the gate explicitly reports unsupported coverage and cannot count a scalar pass. Actual: it repeats native dispatch while advertising forced-scalar. Current prior report claimed four-mode macOS qualification; that scalar claim is now withdrawn. This is a qualification evidence defect, not a demonstrated pixel mismatch. CRU-00001 repaired x86 direct encoder routing and explicitly leaves Arm outside the hook; this report concerns separate AArch64 decoder conformance false coverage. REQ-CRU-00006 dispatch unification does not supply current scalar acceptance. Diagnostic source/commands/output are retained in readiness candidate-20261016 evidence. No fix or acceptance workaround performed.

## Fix

<unfixed — raised only>

## Notes
