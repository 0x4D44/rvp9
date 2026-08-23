# RVP9-REQ-CRU-00007 — Decide and record the SIMD width policy for the five new x86_64 kernels

- **State:** Draft
- **Priority:** Should
- **Area:** core/dsp x86_64 kernels
- **Raised:** 2026-08-23T17:11:34Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-08-23T17:11:34Z, raised via `deltic reqs new` model=claude-opus-5@high)

## Statement

The five x86_64 kernels added between 2026-08-21 and 2026-08-23 must either carry an AVX2
variant or carry a recorded reason for staying at SSE2, so the crate's SIMD-width choice is a
decision rather than an accident.

**Why this is worth tracking.** `crates/rvp9-core/src/dsp/x86_64/` contains roughly 55 AVX2
kernels (`sad_avx2.rs`, `variance_avx2.rs`, the `convolve*_avx2.rs` and `lpf_*_avx2.rs`
families, `idct16x16_avx2.rs`, `idct32x32_avx2*.rs`, …) against a handful of SSE2 ones, and the
pre-existing SSE2 exceptions are deliberate and documented — `idct16x16_sse2_256.rs` is kept at
SSE2 specifically to match libvpx's SSE2 saturation behaviour, recorded in
`notes/divergence-ledger.md` and in RVP9-BUG-KILN-00010. AVX2 is the house convention; SSE2 is
the exception that gets a reason.

These five landed as plain SSE2 with no stated rationale:

- `crates/rvp9-core/src/transforms/forward_sse2.rs` (fdct8x8, fdct16x16 first pass)
- `crates/rvp9-core/src/dsp/x86_64/int_pro_sse2.rs`
- `crates/rvp9-core/src/dsp/x86_64/quantize_metrics_sse2.rs`
- `crates/rvp9-core/src/mode_decision/hadamard_sse2.rs`
- the `int_pro_64x64_fused_sse2` path reached from `crates/rvp9-core/src/mv_search/int_pro_me.rs`

`.cargo/config.toml` sets `target-cpu=native`, and CLAUDE.md states release binaries are not
portable across hosts for that reason — so every host these actually run on has AVX2. The
arithmetic in all five is 16-wide-i16 or 32-wide-u8 friendly (butterflies, Hadamard combines,
row/column byte sums, quantizer magnitude), which is the shape that benefits from doubling the
lane width.

**What is fact and what is not.** The convention mismatch and the `target-cpu=native` setting
are verified. The size of the missed speedup is *not* measured — no benchmark was run in the
review that raised this. It could plausibly be small: several of these kernels are latency- or
load-bound rather than ALU-bound, and `target-cpu=native` already lets LLVM widen some of the
surrounding scalar code. Triage should treat "how much is on the table" as the open question.

**Oracle.** `cargo run --release -p rvp9-bench --bin dsp-microbench` before and after, per
kernel, plus the realtime latency-budget bench to confirm the win survives at the frame level.
Bit-exactness against the existing scalar oracles must hold unchanged — an AVX2 variant that
changes any coefficient is a different (and much larger) decision.

**Acceptable outcomes.** Either (a) AVX2 variants for the kernels where the microbench shows a
real win, or (b) a short comment in each file recording why SSE2 is sufficient, following the
`idct16x16_sse2_256.rs` precedent. Doing (b) for all five and (a) for none is a legitimate
result if the measurements say so.

Related but separate: RVP9-BUG-CRU-00004 covers the x86_64 realtime `fdct8x8` paying a range
check that AArch64 avoids, which is a bigger single win than lane width and should be settled
first. The review also noted two other x86_64-vs-AArch64 vectorization gaps worth folding into
this decision — `metrics_32_inner`
(`crates/rvp9-core/src/dsp/x86_64/quantize_metrics_sse2.rs:239-286`) drops to a scalar per-lane
loop for the Tx32x32 accumulation, and the residual subtract in the RDO hot loop
(`crates/rvp9-core/src/mode_decision/inter.rs:5729-5745`) is scalar on x86_64 while AArch64 has
a fused vector kernel.

Raised by the 2026-08-23 code-review pass
(`wrk_docs/2026.08.23 - CR - 20260823-REV-RVP9-CLA@CRUCIBLE-code-review-175647.md`).

## Notes
