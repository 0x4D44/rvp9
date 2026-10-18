# RVP9-BUG-FLU-00054 — Unsafe FFI declaration blocks lack safety comments

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** unsafe documentation
- **Raised:** 2026-10-17T21:16:56Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261018T201337Z-24e3b39d
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00054-run-fix-20261018T201337Z-24e3b39d
- **Owner base:** 69c4ad39f8df8ffb5a14b2390a5735e433f1c6ca
- **Owner fingerprint:** -
- **Owner since:** 2026-10-18T20:13:37Z
- **Owner until:** 2026-10-19T00:13:37Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-10-17T21:16:56Z, raised via `deltic bugs new`)

## Observation

On exact candidate fb5c0c917038db6c640b3dbc7c2b84b567233333, a static source audit found three Windows unsafe extern declaration blocks without adjacent // SAFETY: comments or # Safety documentation: crates/rvp9-cli/src/path_guard.rs:186 and crates/rvp9-bench/src/bin/soak.rs:284,293. Their call sites have separate safety comments, but these do not document the ABI and declaration assumptions. Repository guidance requires every unsafe block to carry // SAFETY: reasoning. Expected: each declaration block records its safety assumptions; actual: no adjacent safety contract is present. No ABI mismatch or runtime defect is established; this record tracks the safety-documentation gap only.

## Fix

<unfixed — raised only>

## Notes
