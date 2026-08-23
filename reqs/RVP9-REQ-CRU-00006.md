# RVP9-REQ-CRU-00006 — Unify SIMD kernel selection behind one force-scalar-aware dispatch point

- **State:** Draft
- **Priority:** Should
- **Area:** core/dsp dispatch
- **Raised:** 2026-08-23T17:11:33Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-08-23T17:11:33Z, raised via `deltic reqs new` model=claude-opus-5@high)

## Statement

`rvp9-core` must select every architecture-specific kernel through a single dispatch point
that honours the `RVP9_FORCE_SCALAR` mask, so that a forced-scalar process runs the scalar
arm of every kernel without exception.

Today there are two parallel mechanisms and only one of them respects the mask:

1. The runtime `SimdDispatch` table, built once from `X86Caps::detect()`
   (`crates/rvp9-core/src/dsp/dispatch.rs`, `crates/rvp9-core/src/dsp/caps.rs:67-79`).
   `RVP9_FORCE_SCALAR=1` masks every ISA tier to `false` here, so all table-resolved kernels
   fall to their scalar arm. This is what the byte-identity oracle documented at
   `crates/rvp9-core/src/dsp/caps.rs:56-66` depends on.
2. Bare compile-time `#[cfg(target_arch = ...)]` selection at the call site, which no runtime
   switch can reach. Six kernel families now use this on x86_64 — see RVP9-BUG-CRU-00001 for
   the file:line list — and the AArch64 NEON paths have used it for longer.

The second mechanism is why the oracle silently stopped covering the newest kernels. Adding
`force_scalar_env()` checks at each of those call sites would fix the symptom, but it
reproduces the mask logic at every site and the next kernel added by bare `#[cfg]` reopens
the hole. The structural fix is to make table-routed dispatch the only way a kernel gets
selected, so "kernel exists but is unreachable from the mask" stops being expressible.

**Oracle.** Two parts:
- A structural test that fails if any kernel is selectable outside the table — e.g. assert
  that a forced-scalar `SimdDispatch` and a native one differ in every slot that has a
  non-scalar arm, plus a lint or test over the crate that flags `#[cfg(target_arch)]`
  kernel selection outside `dsp/dispatch.rs`.
- The existing cross-process byte-identity check: encode a fixed clip twice, once with
  `RVP9_FORCE_SCALAR=1`, and require identical output MD5s. This currently passes
  vacuously for the six affected families because both legs run the same SSE2 code; after
  the change it becomes load-bearing again.

**Scope note for triage.** Doing this properly means adding table slots for forward
transforms, Hadamard/SATD, integral projection and quantizer metrics on both x86_64 and
AArch64, and reworking their call sites in `transforms/forward.rs`, `mv_search/int_pro_me.rs`,
`mode_decision/hadamard.rs` and `mode_decision/inter.rs`. Sibling commit `daae16aaa` shows
the target pattern on a smaller scale: it moved the large candidate IDCTs from hardcoded
calls to `crate::dsp::get_ref()`. Whether the full sweep is worth it, or whether the
narrower per-site `force_scalar_env()` fix in RVP9-BUG-CRU-00001 is enough, is the Gate-1
decision.

Raised by the 2026-08-23 code-review pass
(`wrk_docs/2026.08.23 - CR - 20260823-REV-RVP9-CLA@CRUCIBLE-code-review-175647.md`).

## Notes
