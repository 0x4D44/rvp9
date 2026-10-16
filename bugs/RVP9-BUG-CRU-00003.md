# RVP9-BUG-CRU-00003 — vpx_vector_var_scalar cfg omits test, so x86_64 cargo test validates SSE2 against a hand-rewritten copy

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** core/mv_search int_pro
- **Raised:** 2026-08-23T17:09:45Z
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
- **State history:** Open (2026-08-23T17:09:45Z, raised via `deltic bugs new` model=claude-opus-5@high) -> Fixed (2026-10-16T03:57:32Z, deltic:auto role=fix run=fix-20261016T035141Z-4b28e506 branch=task/bug-RVP9-BUG-CRU-00003-run-fix-20261016T035141Z-4b28e506 code=68516c6480e166cb07b80579f59145d24fa4185b gate=manual)

## Observation

Commit `edd62aa9e` ("perf(core): vectorize x86 integral projection") tightened the cfg on the
scalar oracle `vpx_vector_var_scalar` (`crates/rvp9-core/src/mv_search/int_pro_me.rs:316`)
from `#[cfg(any(not(target_arch = "aarch64"), miri))]` to:

```rust
#[cfg(any(not(any(target_arch = "aarch64", target_arch = "x86_64")), miri))]
```

It did not add `test`. Every sibling oracle touched in the same batch did:

- `crates/rvp9-core/src/transforms/forward.rs:1011` — `fdct16x16_scalar` →
  `#[cfg(any(not(any(target_arch = "aarch64", target_arch = "x86_64")), test, miri))]`
- `crates/rvp9-core/src/mode_decision/hadamard.rs:40, 82, 123, 179` — all four scalar
  helpers → `#[cfg(any(test, miri, all(not(target_arch = "aarch64"), not(target_arch = "x86_64"))))]`

**Expected:** on x86_64, `cargo test -p rvp9-core` compiles the crate's designated scalar
reference and compares the SSE2 kernel against it.
**Actual:** on x86_64 `vpx_vector_var_scalar` is compiled only under Miri. The SSE2 kernel's
only in-repo check is `scalar_vector_var`
(`crates/rvp9-core/src/dsp/x86_64/int_pro_sse2.rs:273-283`), a second implementation of the
formula hand-written inside the SSE2 file's own test module.

This is not a build break — on x86_64 non-Miri the scalar arm of `vpx_vector_var`
(`int_pro_me.rs:344-347`) is also cfg'd out, so nothing calls the missing function. The defect
is to test integrity: the production oracle (still live on non-x86/non-aarch64 targets and
under Miri) and the thing the SSE2 kernel is actually tested against are now two separate
pieces of code that nothing keeps in agreement. If they drift, no configuration that
`cargo test --workspace` runs on x86_64 will notice.

This matters more than usual here because, per the same review pass, none of the five new SSE2
files are reachable from any checked-in differential/conformance harness — the in-file unit
tests are the only net.

**Suggested fix:** add `test` to the cfg at `crates/rvp9-core/src/mv_search/int_pro_me.rs:316`,
matching the sibling pattern; raise `vpx_vector_var_scalar` to `pub(crate)`; and replace
`scalar_vector_var` in `crates/rvp9-core/src/dsp/x86_64/int_pro_sse2.rs:273-283` with a call to
it, so the SSE2 kernel is compared against the real oracle rather than a copy.

## Fix

<unfixed — raised only>

## Notes
