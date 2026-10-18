# RVP9-BUG-FLU-00054 — Unsafe FFI declaration blocks lack safety comments

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** unsafe documentation
- **Raised:** 2026-10-17T21:16:56Z
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
- **State history:** Open (2026-10-17T21:16:56Z, raised via `deltic bugs new`) -> Fixed (2026-10-18T20:17:05Z, deltic:auto role=fix run=fix-20261018T201337Z-24e3b39d branch=task/bug-RVP9-BUG-FLU-00054-run-fix-20261018T201337Z-24e3b39d code=88cb8d8ea11121a7c9257800cc8b9d333792de6b gate=manual) -> Closed (2026-10-18T21:49:43Z, deltic:manual role=verify run=verify-20261018T214423Z-d91e438b verified_fix_run=fix-20261018T201337Z-24e3b39d verdict=close)

## Observation

On exact candidate fb5c0c917038db6c640b3dbc7c2b84b567233333, a static source audit found three Windows unsafe extern declaration blocks without adjacent // SAFETY: comments or # Safety documentation: crates/rvp9-cli/src/path_guard.rs:186 and crates/rvp9-bench/src/bin/soak.rs:284,293. Their call sites have separate safety comments, but these do not document the ABI and declaration assumptions. Repository guidance requires every unsafe block to carry // SAFETY: reasoning. Expected: each declaration block records its safety assumptions; actual: no adjacent safety contract is present. No ABI mismatch or runtime defect is established; this record tracks the safety-documentation gap only.

## Fix

Added adjacent `// SAFETY:` comments to the three Windows `unsafe extern "system"` declaration blocks. The comments record the system-ABI signatures, handle and scalar type widths, and the C layouts expected for `FILE_ID_INFO` and `PROCESS_MEMORY_COUNTERS`; call behavior is unchanged.

Validation: reviewed the declarations against Microsoft API documentation for the named functions and structures; `cargo fmt --all --check`, `cargo check -p rvp9-cli --locked`, and `cargo check -p rvp9-bench --bin soak --locked` passed on the host. The Windows-only blocks were cfg-disabled by this host's target, so no Windows-target build was run.

### Verification summary (2026-10-18T21:49:43Z, deltic:auto run=verify-20261018T214423Z-d91e438b verified_fix_run=fix-20261018T201337Z-24e3b39d verdict=close)

Verifier note: Manual verify: Independent content verification: all three Windows extern blocks now carry adjacent SAFETY comments; pre-fix content fails those predicates. Audited HANDLE/pointer, DWORD/u32, BOOL/i32, enum/i32, FILE_ID_INFO and eight SIZE_T PROCESS_MEMORY_COUNTERS fields against declarations. cargo check -p rvp9-cli --locked and cargo check -p rvp9-bench --bin soak --locked pass; fmt passes. Documentation-only change. Windows ABI paths cfg-disabled on macOS; no Windows runtime claim.

## Notes
