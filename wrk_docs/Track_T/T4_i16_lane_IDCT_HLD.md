# Track T-4: AVX2 i16-lane IDCT kernels for the dense IDCT path — HLD

**Author:** T-4 planning agent
**Date:** 2026-05-24
**Status:** Design proposal; first-slice ports ONE kernel (`vpx_idct16x16_256_add_sse2`).

---

## 1. Kernel inventory

Existing i32-lane AVX2 IDCT kernels in `crates/rvp9-core/src/dsp/x86_64/` (LOC includes the per-file proptest):

| Kernel file | LOC | TxSize | Dispatch slot | EOB bucket | Call profile |
|---|---|---|---|---|---|
| `idct4x4_avx2.rs` | 658 | 4×4 | `idct4x4_add` | dense | High freq, but kernel is already trivial |
| `idct8x8_avx2.rs` | 803 | 8×8 | `idct8x8_add` | dense (eob > 12) | Hottest IDCT in real content (foreman, tile_1x4) |
| `idct8x8_12_sse2.rs` | 1091 | 8×8 | `idct8x8_12_add` | sparse eob ≤ 12 | Track-S Task A — i16 SSE2, exemplar |
| `idct16x16_avx2.rs` | 1058 | 16×16 | `idct16x16_add` | dense (eob > 38) | High freq on testsrc, also real content |
| `idct16x16_10_avx2.rs` | 837 | 16×16 | `idct16x16_10_add` | sparse eob ≤ 10 | Wave F F3, finisher-only port |
| `idct16x16_38_avx2.rs` | 775 | 16×16 | `idct16x16_38_add` | sparse eob ≤ 38 | Wave F F4, finisher-only port |
| `idct32x32_avx2.rs` | 1346 | 32×32 | `idct32x32_add` | dense (eob > 135) | Largest absolute cost per call; rare in foreman, more frequent in testsrc-style synthetic |
| `idct32x32_34_avx2.rs` | 957 | 32×32 | `idct32x32_34_add` | sparse eob ≤ 34 | Wave F F9, finisher-only |
| `idct32x32_135_avx2.rs` | 966 | 32×32 | `idct32x32_135_add` | sparse eob ≤ 135 | Wave F F9, finisher-only |
| `iht4x4_avx2.rs` | 794 | 4×4 hybrid | `iht4x4_add` | dense | ADST/DCT combinations |
| `iht8x8_avx2.rs` | 905 | 8×8 hybrid | `iht8x8_add` | dense | ADST/DCT combinations |
| `iht16x16_avx2.rs` | 1321 | 16×16 hybrid | `iht16x16_add` | dense | Wave F F5; row pass via scalar |

Dispatch table: `crates/rvp9-core/src/dsp/dispatch.rs:663-2100`; the row-pass / column-pass / finisher pattern is canonical. The i32-lane approach uses `_mm256_mullo_epi32` + `_mm256_packs_epi32` only at the finisher; the SSE2 Task A kernel uses `_mm_madd_epi16` end-to-end with `_mm_packs_epi32` at every wraplow site.

Honest call-frequency data is not in the repo at coefficient granularity. We have only fixture wall-clock (`wrk_docs/Track_R/next_session_brief.md:14-21`) and the qualitative ordering: `idct8x8_64_add` ≫ `idct16x16_256_add` ≫ `idct32x32_1024_add` in real content, with `idct16x16` rising sharply on testsrc synthetic content.

## 2. Saturating-pack envelope

**The semantic boundary.** Two i16 truncation policies exist in the IDCT pipeline:

- **Scalar reference** uses `wraplow(x) = x as i16 as i32` — modular truncation (libvpx `WRAPLOW`). Out-of-range values wrap silently. Implemented in `crates/rvp9-core/src/transforms/inverse.rs` and matched bit-exactly by the i32-lane AVX2 kernels via `wrap(x) = srai_epi32(slli_epi32(x, 16), 16)`.
- **libvpx i16 SSE2** uses `_mm_packs_epi32` — saturating to ±i16::MAX. The two policies agree iff every intermediate fits in i16. They diverge on adversarial synthetic inputs.

**VP9 spec bound.** §8.6 limits dequantised coefficients to the i16 range. After dequant, in-bitstream coefficients stay in `[i16::MIN, i16::MAX]`. The 4/8-stage butterflies of `idct16` can produce intermediates that *briefly* exceed i16 even from spec-bounded inputs — libvpx's `WRAPLOW` truncates them, and so does saturating-pack so long as the magnitudes are below ±i16::MAX. The empirical envelope from Task A: i12-bounded inputs (`[-2048, 2047]`) keep every intermediate inside i16 for `idct8`, hence bit-exact equality. The `idct16` and `idct32` butterflies have more stages and broader cospi mixing, so the safe per-input range shrinks slightly.

**Per Task A precedent** (`idct8x8_12_sse2.rs:30-47`, `task_a_port_plan.md:144` and inline tests): proptest tightened from i14 → i12. For `idct16`, the recommended per-helper proptest range is `[-2048, 2047]` (i12) for the `idct16_8col` differential; the outer-kernel differential against the i32-lane AVX2 kernel uses `[-2048, 2047]` with documented exclusions of the high-magnitude / alternating-sign strategies (deliberately omitted; they trip saturate-vs-wraplow). The 201/201 conformance corpus passes bit-exactly because all real bitstream coefficients stay within the envelope.

**For the dense path specifically:** we keep `bit-exact for in-spec bitstream coefficients` as the load-bearing guarantee. The dispatch is keyed on `caps.avx2 ∧ caps.sse2`, both of which are universal on x86_64.

## 3. Kernel selection — `idct16x16_256_add` (i16 SSE2 first slice)

Recommend porting **`vpx_idct16x16_256_add_sse2`** (`third_party/libvpx/vpx_dsp/x86/inv_txfm_sse2.c:403-431`) as the first slice. Trade-offs considered:

| Candidate | Pros | Cons | Verdict |
|---|---|---|---|
| `idct16x16_256_add` (SSE2 i16) | Already have nearly every helper from Task A (`pair_set_epi16`, `idct_madd_round_shift_sse2`, `idct_calc_wraplow_sse2`, `butterfly`, `idct8_sse2`). The `idct16_8col` inner is a 16-butterfly extension of `idct8`. Highest call frequency on testsrc, second-highest on real content. Achievable overnight. | i16-lane SSE2 means 8 lanes per vector, vs current i32-lane AVX2 at 8 lanes per ymm — *same lane count, half the vector width*. Per-instruction throughput parity, but per-vector latency lower → net modest win (~10-20% kernel-local). | **CHOSEN** |
| `idct16x16_256_add` (AVX2 i16, port libvpx `inv_txfm_avx2.c:277`) | 16 i16 lanes per ymm → 2× the work per instruction. Biggest win. | Requires porting `transpose_16bit_16x16_avx2` + `idct16_16col` (AVX2 ymm) + `butterfly16` macro. ~3× the new-helper LOC vs SSE2. Risk of overnight slip. | DEFERRED to slice 2 |
| `idct32x32_1024_add` (any flavour) | Biggest absolute per-call wall-clock. | 32-input butterfly across 4 quarters; ~2,000+ LOC; deeper stack of new helpers (`idct32_16x32_quarter_1..4`, `add_sub_butterfly_avx2`); higher overflow risk in butterflies. Single-slice porting risk is too high. | DEFERRED to slice 3 |
| `iadst` family | Largely independent helpers; ADST butterfly structure differs from DCT. | Adds work without leveraging Task A helpers. | DEFERRED indefinitely |

**Rationale for SSE2 over AVX2 i16 for the first slice:** Task A shipped the SSE2 helpers. Re-using them is the lowest-risk path to a bit-exact-verified merge tonight. AVX2 i16 lands in slice 2 once SSE2 i16 has proven the envelope on the conformance corpus end-to-end.

## 4. Port plan for `idct16x16_256_add_sse2`

**Reuse from Task A (`idct8x8_12_sse2.rs`):**
- `pair_set_epi16` (lines 97-102)
- `idct_madd_round_shift_sse2` (114-117)
- `idct_calc_wraplow_sse2` (130-134)
- `butterfly` (145-153) — already matches libvpx `inv_txfm_sse2.h:81-90`
- `idct8_sse2` (213-244) — not directly used by `idct16_8col`, but the *shape* (line-by-line port of libvpx) is the template
- `load_input_data4` (198-201) — variant `load_input_data8` is new

**New helpers required for slice 1:**

| New helper | Source | Est. LOC | Notes |
|---|---|---|---|
| `load_input_data8(*const i32) -> __m128i` | libvpx `inv_txfm_sse2.h:111-119` | 8 | Saturating-pack 8 × i32 → 8 × i16; our coefs are i32 throughout, so we always pack |
| `transpose_16bit_8x8([__m128i; 8])` | libvpx `transpose_sse2.h:157-220` | 35 | 3-stage unpack pyramid; pure shuffle, no arithmetic |
| `idct16_8col([__m128i; 16]) -> [__m128i; 16]` | libvpx `inv_txfm_sse2.h:343-427` | 90 | The core 1-D idct16, built from `butterfly` calls + add/sub epi16 |
| `recon_and_store_16(*mut u8, __m128i, __m128i)` | libvpx `inv_txfm_sse2.c:486-497` + `write_buffer_8x1` `inv_txfm_sse2.h:229-235` | 25 | The 16-byte residual add: round (+32 >> 6), zero-extend dst u8 → 2 × i16, add, pack |
| `idct16_load8x8(*const i32, &mut [__m128i; 8])` | libvpx `inv_txfm_sse2.c:386-401` | 12 | 8 calls to `load_input_data8` |
| `idct16x16_add_sse2_256_inner(input, dst, stride)` | libvpx `inv_txfm_sse2.c:403-431` | 50 | Two-pass `idct16_8col` outer driver matching libvpx exactly |
| `pub fn idct16x16_add_sse2_256(...)` safe wrapper | per Task A pattern | 10 | One `unsafe { ... }` call with SAFETY comment naming the OnceLock + caps invariant |
| Per-helper proptests + outer differential | Task A `mod tests` pattern | 250 | i12-bounded inputs, scalar oracle for each new helper, outer differential vs `idct16x16_add_avx2` (i32-lane, in-envelope = bit-exact oracle) |

**Estimated total new file LOC:** ~480 (kernel) + ~250 (tests) = **~730 LOC**. Roughly two-thirds the size of `idct8x8_12_sse2.rs` because we re-use the Task A helpers.

**Outer driver structure** (mirrors libvpx `vpx_idct16x16_256_add_sse2`):

```
let mut l: [__m128i; 16] = [...];   // left 8×16 intermediate (i16 lanes)
let mut r: [__m128i; 16] = [...];   // right 8×16 intermediate
// Row pass: process 16 rows in two 8-row groups.
for i in 0..2 {
    idct16_load8x8(input, &mut buf[0..8]);
    transpose_16bit_8x8(...);
    idct16_load8x8(input + 8, &mut buf[8..16]);
    transpose_16bit_8x8(...);
    idct16_8col(buf, buf);
    input += 128;
}
// Column pass: 2 strips of 8 cols.
for i in 0..2 {
    transpose_16bit_8x8(l + 8*i, &mut out[0..8]);
    transpose_16bit_8x8(r + 8*i, &mut out[8..16]);
    idct16_8col(out, out);
    for j in 0..16 { recon_and_store_16(dst + j*stride, out[j]); }
    dst += 8;   // wait — write 16-byte-wide
}
```

(The final loop in libvpx writes 8 cols per row × 2 strips, not 16. The `recon_and_store_16` name is misleading; libvpx uses `write_buffer_8x1` per row in the strip loop. Verify against `inv_txfm_sse2.c:419-430` during the port.)

## 5. Dispatch wiring

The `idct16x16_add` slot already exists at `dispatch.rs:703`. Decision: **add a second SSE2-specialised path** keyed on `caps.sse2 && !caps.avx2`, or **replace** the existing i32-lane AVX2 entry. The conservative choice that matches Task A's precedent is to **replace** when `caps.avx2 == true` (because the SSE2 i16 kernel is expected to outperform the i32 AVX2 kernel on per-instruction throughput) — but this is a perf bet that should be confirmed at bench gate. The **safer initial wiring** is:

```
let idct16x16_add: fn(&[i32; 256], &mut [u8], usize) =
    if caps.avx2 {
        crate::dsp::x86_64::idct16x16_add_avx2  // existing i32-lane
    } else if caps.sse2 {
        crate::dsp::x86_64::idct16x16_add_sse2_256  // NEW
    } else {
        scalar_idct16x16_add
    };
```

This installs the new kernel on the (vanishingly small) non-AVX2-but-SSE2 cohort first. Once bit-exact on conformance + bench-positive, **slice 1.5** flips the AVX2 arm to the new SSE2 i16 kernel and retires `idct16x16_avx2.rs` after a bench A/B. (Per Track-R R.16 b4dfaea, `WorkerPool::new(Arc<SimdDispatch>)` already threads dispatch through inter MC, so call sites in both `reconstruct.rs:447` and `reconstruct_inter.rs:859` pick up the new pointer with zero change.)

**No call-site changes needed.** Both intra `reconstruct.rs:429-454` and inter `reconstruct_inter.rs:842-861` already invoke `dispatch.idct16x16_add()`. The kernel pointer is selected once at `SimdDispatch::build` time; rebuild is automatic.

**Optional alternative — a separate slot.** If we want bench-comparable A/B between i32 AVX2 and i16 SSE2 without forcing one config, add a new field `idct16x16_add_sse2_i16` and gate at the call site behind an env-var-controlled mask (mirroring `LpfAvx2Mask` at `dispatch.rs:65-138`). Pro: enables bisection. Con: scaffolding cost. Recommend **deferring** to slice 2 alongside the AVX2 i16 port.

## 6. Validation strategy

Three concentric gates, all required for merge:

1. **Per-helper proptest** (i12 envelope, ~256 cases each, in `mod tests` colocated with the kernel):
   - `load_input_data8`: scalar oracle = saturating-clamp + cast; in-envelope inputs round-trip identically.
   - `transpose_16bit_8x8`: scalar oracle = matrix transpose of 8×8 i16. Bit-exact for arbitrary i16.
   - `idct16_8col`: scalar oracle = `crate::transforms::inverse::idct16` over an 8-column slice (one column passed through scalar idct16 per loop iteration, results compared lane-by-lane against the vector output). i12-bounded inputs only.
   - `recon_and_store_16`: scalar oracle = `((x + 32) >> 6) + dst` saturated to u8; verify with arbitrary i16 inputs and u8 dst.

2. **Outer differential proptest** vs the existing i32-lane AVX2 kernel:
   - `idct16x16_add_sse2_256(in, dst1, stride) == idct16x16_add_avx2(in, dst2, stride)` for all `in ∈ [-2048, 2047]^256`, with `dst1.clone() = dst2.clone()` start.
   - Strategies: uniform_12, sparse_12 (eob ≤ 38 region), eob_bucket_38, eob_bucket_dense (most positions in i12). Deliberately omit `pathological_strategy` (i16 extremes) per Task A precedent.
   - 256 cases per strategy. Bit-exact equality, no tolerances.

3. **Conformance gate — load-bearing.** `cargo test -p rvp9-conformance --release`: 201/201 vectors must pass. This is the empirical proof that the saturating-pack envelope holds on every real bitstream coefficient. Track-S Task A's 201/201 (commit bd1df7d) is the published precedent: if it holds for the sparse path it should hold for the dense path, but conformance is the load-bearing verifier, not a presumption.

4. **Profile-1, frame_parallel_regression, tile_threading.** Sanity gates from the Task A regression list (4 suites). All must remain green.

## 7. Bench expectation

**Honest baseline:** the dense `idct16x16_add` is hot on testsrc fixtures (synthetic high-frequency content fills the eob > 38 bucket). On `testsrc_1080p30 t=4` (the worst-case fixture per `next_session_brief.md:21`), the IDCT pipeline is a single-digit-percent contributor — bench noise floor is 13-21% per `next_session_brief.md:33`.

**Per-instruction theoretical win:** i16-lane SSE2 with `_mm_madd_epi16` does 4 mul + 1 add per instruction vs i32-lane AVX2 with separate `_mm256_mullo_epi32` + `_mm256_add_epi32`. Halving the multiply+add fold count for the inner butterfly is a ~15-25% kernel-local win. At the wall-clock level, with the IDCT accounting for ~10% of testsrc decode time, the expected fixture-level delta is **2-3%** — *below the noise floor*.

**Minimum detectable effect at n=5 + MWU:** Track S Task C (commit 405efe0) established a Mann-Whitney U gate at p < 0.05 with N=5 runs. With 13-21% noise, the MDE is ~10% per fixture. **Therefore: a single-kernel port is statistically indistinguishable from baseline on the testsrc fixtures.** The honest claim is "no regression, kernel-local win confirmed via microbench, conformance bit-exact." The accumulated portfolio across all dense IDCT kernels (slice 2 + slice 3) is where bench-detectable wall-clock should appear.

**Microbench (recommended):** add a Criterion benchmark `bench/idct16x16_dense.rs` that calls each variant (scalar / AVX2 i32 / SSE2 i16) 100k times on a representative coefficient set. This gives the kernel-local number outside the noise of full-frame decode. Microbench is a *secondary* gate; conformance is primary.

## 8. Phasing

**First slice (this overnight, ~6-8 hours focused work):**
- Port `idct16x16_256_add_sse2` per §4.
- Wire into `SimdDispatch::idct16x16_add` per §5 (conservative non-AVX2 arm only initially).
- All four validation gates per §6.
- Microbench evidence in `bench/`.
- Journal entry mirroring Track-S Task A close (`186a656` / `242edbd`).

**Slice 1.5 (next session, ~2 hours):**
- Bench A/B: flip the AVX2 arm to the new SSE2 i16 kernel.
- Retire `idct16x16_avx2.rs` if bench is positive or neutral; revert otherwise.

**Slice 2 (next overnight, ~6-8 hours):**
- Port `vpx_idct16x16_256_add_avx2` (libvpx `inv_txfm_avx2.c:277-294`). New helpers: `transpose_16bit_16x16_avx2`, `idct16_16col`, `butterfly16`, `write_buffer_16x1`, `PAIR256_SET_EPI16`. Est. ~900 LOC. Same validation envelope.

**Slice 3 (third overnight, ~8-10 hours):**
- Port `vpx_idct32x32_1024_add_avx2` (libvpx `inv_txfm_avx2.c:574-601`). Quarters 1-4 inner butterflies; biggest absolute win, highest porting risk. Est. ~1400 LOC.

**Slice 4+ (if pursued):** iadst variants, 16x16/32x32 EOB-bucket finishers in i16, hybrid `iht16x16` row pass in i16. Diminishing returns; gate each on a bench-positive prior slice.

## 9. Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| **R1: Pack-saturate trips on out-of-envelope inputs** — proptest finds an in-bitstream-spec coefficient pattern where intermediates exceed ±i16::MAX and saturate-pack diverges from scalar wraplow. | Medium | High (conformance regression) | Proptest range bounded to i12 per §6; if a 201/201 conformance vector trips this, the fix is to narrow the envelope or fall back to i32 for that bucket. Pre-flight: hand-trace `idct16_8col` worst-case magnitude (libvpx ships this kernel in production, so the answer is "safe for in-spec bitstream coefs" — but verify by running the existing 201 vectors against the new kernel BEFORE wiring it as the production path). |
| **R2: Transpose helpers wrong shape** — Task A's `transpose_16bit_4x4` is a 4-row transpose; idct16 needs `transpose_16bit_8x8` (libvpx `transpose_sse2.h:157`). The 8×8 transpose has 3 unpack stages instead of 2, and the operand ordering for `unpacklo`/`unpackhi` is easy to mis-transcribe. | High | Medium (bit-exact differential will catch it before merge) | Line-by-line port from libvpx with intent-comments preserved; per-helper proptest at §6.1 catches any mis-shuffle before integration. |
| **R3: Intermediate overflow during butterfly stages** — even in-envelope i12 inputs may produce stage-2 intermediates near ±i16::MAX. `_mm_add_epi16` does *not* saturate (it wraps modularly, same as `wraplow`), so this is mostly fine, but the `_mm_packs_epi32` at every `idct_calc_wraplow_sse2` site clamps. If a stage-3 input pair sums to > i16::MAX, the next-stage butterfly sees a clamped value instead of a wrapped value. | Medium | Medium (kernel-local; conformance is the catch-all) | This is the exact issue Task A tightened from i14 to i12 for. The 201 conformance vectors *empirically* don't trip it; if they do, narrow further to i11 (proptest envelope), accept the in-bitstream invariant, or revert. **Pre-flight: run conformance with the new kernel as a non-default arm before flipping it on for AVX2 caps** — gives concrete evidence before committing to retire the i32-lane AVX2 path. |

---

## Critical Files for Implementation

- `D:\language\rvp9-2\crates\rvp9-core\src\dsp\x86_64\idct8x8_12_sse2.rs` — the helper-rich exemplar; new kernel reuses 5 of its `#[inline] unsafe fn` helpers verbatim
- `D:\language\rvp9-2\third_party\libvpx\vpx_dsp\x86\inv_txfm_sse2.c` — lines 403-431 are the outer driver to line-by-line port
- `D:\language\rvp9-2\third_party\libvpx\vpx_dsp\x86\inv_txfm_sse2.h` — lines 343-427 (`idct16_8col`), 229-235 (`write_buffer_8x1`), 111-119 (`load_input_data8`); macros to port
- `D:\language\rvp9-2\third_party\libvpx\vpx_dsp\x86\transpose_sse2.h` — lines 157-220 (`transpose_16bit_8x8`)
- `D:\language\rvp9-2\crates\rvp9-core\src\dsp\dispatch.rs` — lines 1262-1269 (current `idct16x16_add` slot wiring; the surgical edit point)
- `D:\language\rvp9-2\crates\rvp9-core\src\dsp\x86_64\idct16x16_avx2.rs` — the i32-lane oracle for the outer differential proptest (in-envelope = bit-exact)
