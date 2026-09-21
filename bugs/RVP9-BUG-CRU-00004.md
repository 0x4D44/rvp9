# RVP9-BUG-CRU-00004 — x86_64 realtime fdct8x8 pays a range check and full scalar redo; only AArch64 has the unchecked lowbd kernel

- **State:** Open
- **Priority:** Could
- **Severity:** Medium
- **Area:** core/transforms forward
- **Raised:** 2026-08-23T17:09:50Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260921T081636Z-99dbde25
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-CRU-00004-run-fix-20260921T081636Z-99dbde25
- **Owner base:** b69ae1269b2a5f062127b4f42af0770fd1e37591
- **Owner fingerprint:** -
- **Owner since:** 2026-09-21T08:16:36Z
- **Owner until:** 2026-09-21T10:16:36Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-23T17:09:50Z, raised via `deltic bugs new` model=claude-opus-5@high)

## Observation

`fdct8x8_libvpx_lowbd` (`crates/rvp9-core/src/transforms/forward.rs:521-530`) is the 8x8
forward DCT on the real-time encode path. Callers:
`crates/rvp9-core/src/mode_decision/inter.rs:5847` (RD round-trip),
`crates/rvp9-encoder/src/lib.rs:13172` and `:13947` (transform dispatch).

On AArch64 it calls a dedicated unchecked kernel — `neon::fdct8x8_libvpx_lowbd`
(`crates/rvp9-core/src/transforms/forward_neon.rs:1493-1500`), documented at
`forward.rs:518-519` as "libvpx's approximate first-pass multiply and unchecked i16 second
pass". On every other architecture, including x86_64, it falls through to the general
`fdct8x8` (`forward.rs:526-529`), which since commit `a39734335` is the **checked** SSE2
kernel: it runs the full first pass, range-checks the result with `pass1_is_safe`
(`crates/rvp9-core/src/transforms/forward_sse2.rs:102-110, 127-129`), and on failure discards
the vector work and recomputes both passes in scalar i64 via `fdct8x8_scalar`
(`forward_sse2.rs:154-156`).

Two consequences.

**1. Wasted work on energetic blocks (performance).** The guard fires whenever a first-pass
output leaves `[-4096, 4095]`. For a constant residual of magnitude `v` the DC output is
`round(22.62 * v)`, so the threshold is `|v| > 181` — about 71% of the legal `±255` range.
That is reachable on keyframes, scene cuts, fades, and any poorly-predicted block, i.e. the
content where real-time throughput is tightest. When it fires the cost is the SSE2 first pass
plus the check plus a full scalar redo — strictly worse than having called scalar directly.
How often this happens on real footage is not measured here; the threshold derivation is exact,
the frequency is not.

**2. Architecture-dependent encoder output (parity).** AArch64 realtime uses libvpx's
approximate kernel; x86_64 realtime uses an exact one. The same input therefore yields
different 8x8 coefficients, and so potentially different mode decisions and bitstreams, on the
two architectures. This predates the batch (x86_64 previously used the exact
`fdct8x8_scalar`), so it is not a regression — but it is undocumented.

Neither behaviour appears in `notes/divergence-ledger.md`: searches for `lowbd`, `4095`,
`pass1`, and "approximate first-pass" return no matches, and the ledger's `fdct16x16` NEON
entry (`notes/divergence-ledger.md:894-911`) shows the format such an entry should take.
CLAUDE.md requires every deliberate departure from libvpx behaviour to be recorded.

The doc comment at `forward.rs:518-520` ("Other architectures use that implementation") is also
now stale: on x86_64 "that implementation" is the SSE2 kernel, not the scalar one it described
when written.

**Suggested fix (in priority order):** (a) add the two missing divergence-ledger entries — the
`pass1_is_safe` guard and the arch-dependent lowbd split — since that is cheap and closes the
documented-process gap; (b) port a true unchecked SSE2 lowbd variant mirroring
`forward_neon.rs:1005-1020` so the x86_64 realtime path stops paying the check, and gate the
change on a BD-rate delta; (c) refresh the stale doc comment. Item (b) needs a measurement
before/after, so treat the effort as medium rather than mechanical.

## Fix

<unfixed — raised only>

## Notes
