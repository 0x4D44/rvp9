# RVP9-BUG-CRU-00004 — x86_64 realtime fdct8x8 pays a range check and full scalar redo; only AArch64 has the unchecked lowbd kernel

- **State:** Blocked
- **Priority:** Could
- **Severity:** Medium
- **Area:** core/transforms forward
- **Raised:** 2026-08-23T17:09:50Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T075939Z-5a200c8e
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-CRU-00004-run-fix-20261016T075939Z-5a200c8e
- **Owner base:** aa2188a8590c4a152835e82d9e8216266017f245
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T07:59:39Z
- **Owner until:** 2026-10-16T11:59:39Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-23T17:09:50Z, raised via `deltic bugs new` model=claude-opus-5@high) → Blocked (2026-10-16, Codex; Arthur decision required on the x86 lowbd output and performance contract)

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

### Output and performance contract decision required (2026-10-16)

Missing input: **Arthur** must choose whether to preserve exact x86 coefficients and pursue an output-preserving optimization, authorize an approximate unchecked x86 kernel subject to fresh native quality and speed gates, or explicitly accept/document the current architecture split and fallback cost. The question has been presented; no answer has been assumed. Documentation alone would not remove the checked path described in this bug.

Current `forward.rs:525` still routes AArch64 lowbd calls to the unchecked NEON variant and other architectures to general `fdct8x8`. On x86, `forward_sse2.rs:103`, `:127`, and `:156` still check first-pass lanes against `[-4096, 4095]` and recompute the complete transform in scalar code on rejection. Existing tests at `forward_sse2.rs:528`, `:570`, and `:603` cover the near-boundary and fallback behavior. CRU-00001 restored diagnostic scalar routing, and CRU-00002 strengthened this guard coverage; neither removed the lowbd split.

There is already a rejected implementation of the proposed unchecked x86 port: `wrk_journals/2026.09.21 - JRN - Unit 1384 x86 unchecked lowbd fdct8x8 rejected.md`, landed in `d1cd9ad0491cec4a7586fe84f47c6083e1249e1c`. Its two native paired corpus ratios were `1.000218` and `1.000033` (candidate/control), with matching IVF hashes on the screened clips. The candidate was not retained. These historical screens do not establish a material product speed gain, nor do they measure current fallback frequency or establish quality on a broader corpus. The original code-level wasted-work observation is insufficient to override that evidence or select a new output contract.

The specific 8x8 divergence entries remain absent. The current public comment's “Other architectures use that implementation” refers to general `fdct8x8`; it is vague, rather than independently demonstrated to be false. Pinned libvpx has separate Arm and x86 schedules, so mirroring the Arm lowbd kernel should not be called x86 libvpx parity without a separate oracle. No source changes, codec execution, new regression proof, quality measurements, or performance measurements were performed in this decision-only pass. This bug is neither Fixed nor Closed. Reopen after Arthur supplies the policy decision.
