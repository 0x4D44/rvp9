# Task A — Port `vpx_idct8x8_12_add_sse2` to Rust SSE2 intrinsics

**Status:** design-prep / port plan (no Rust code modified in this pass). Author: Track-S design-prep agent, 2026-05-24.
**Target file (NEW):** `D:\language\rvp9-2\crates\rvp9-core\src\dsp\x86_64\idct8x8_12_sse2.rs`
**Target dispatch slot:** `SimdDispatch::idct8x8_12_add` (slot needs to be ADDED — the R.6 attempt's slot was reverted, see §3).
**Target call sites:**
- `D:\language\rvp9-2\crates\rvp9-decoder\src\reconstruct.rs:411` (intra)
- `D:\language\rvp9-2\crates\rvp9-decoder\src\reconstruct_inter.rs:824` (inter)

> NOTE: the next-session brief used to read "`reconstruct.rs:411` and
> `reconstruct_inter.rs:818`". Path is `crates/rvp9-decoder/src/`, not
> `crates/rvp9-core/src/`. The inter call site line drifted to ~824 in
> the current tree (post-R.6 comment block); see §4.

---

## 1. libvpx kernel structure

### 1.1 Outer driver: `vpx_idct8x8_12_add_sse2`

`D:\language\rvp9-2\third_party\libvpx\vpx_dsp\x86\inv_txfm_sse2.c:174-185`

```c
void vpx_idct8x8_12_add_sse2(const tran_low_t *input, uint8_t *dest,
                             int stride) {
  __m128i io[8];

  io[0] = load_input_data4(input + 0 * 8);
  io[1] = load_input_data4(input + 1 * 8);
  io[2] = load_input_data4(input + 2 * 8);
  io[3] = load_input_data4(input + 3 * 8);

  idct8x8_12_add_kernel_sse2(io);
  write_buffer_8x8(io, dest, stride);
}
```

Four `__m128i` of 4 × i16 lanes are loaded (low half), then the inner
kernel + writer run. Rows 4..7 of `input` are *not* read — the caller's
contract is `eob <= 12` ⇒ only the top-left 4×4 sub-block contains
non-zero coefficients.

### 1.2 Inner kernel: `idct8x8_12_add_kernel_sse2`

`D:\language\rvp9-2\third_party\libvpx\vpx_dsp\x86\inv_txfm_sse2.h:285-341`

Sketch (full source in the header):
1. `transpose_16bit_4x4(io, io)` — 4×4 transpose of the loaded coefs.
2. Stage 1 (partial-rows row pass): compute `step1[4]`, `step1[5]` using
   `idct_calc_wraplow_sse2` with cospi-28/-4 and cospi-(-20)/12 pairs.
3. Stage 2: `step2[0..2]`, then `step2[4..6]`. Uses
   `idct_madd_round_shift_sse2(cp_16_16, lo_0)` for the DC pair and
   `idct_calc_wraplow_sse2(cp_8_24, cp_24_n8, lo_2)` for the
   secondary pair.
4. Stage 3: combines `step2[0]`/`step2[2]` and `step2[5]`/`step2[6]` via
   add/sub + `idct_calc_wraplow_sse2`.
5. Stage 4: final 4-row partial output `tmp[0..3]`.
6. `idct8x8_12_transpose_16bit_4x8(tmp, io)` — expands 4 rows into the
   8-row `io[]` layout (rows 4..7 zeroed).
7. `idct8(io, io)` — full 8-row SSE2 idct8 over i16 lanes (column pass).

Final `write_buffer_8x8` adds `+16` round, shift right 5, and saturated
adds to dst.

### 1.3 Helper inventory

Each helper is one-to-one with the libvpx C; signatures collapsed for brevity.

- **`pair_set_epi16(a, b)`** — `third_party/libvpx/vpx_dsp/x86/txfm_common_sse2.h:17-19`. Macro producing an `__m128i` of 8 alternating i16 lanes `a,b,a,b,a,b,a,b` (cospi pre-pack for `_mm_madd_epi16`). Rust: `_mm_set1_epi32(((b as u16 as u32) << 16) | (a as u16 as u32))` — single-broadcast of packed `(b,a)`.
- **`idct_madd_round_shift_sse2(in, cospi)`** — `inv_txfm_sse2.h:65-69`. `(_mm_madd_epi16(in, cospi) + 8192) >> 14`, returns 4 × i32 lanes. Rust: `_mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(in_, cospi), _mm_set1_epi32(1 << 13)), 14)`.
- **`idct_calc_wraplow_sse2(in0, in1, x)`** — `inv_txfm_sse2.h:72-78`. Two `idct_madd_round_shift_sse2` + `_mm_packs_epi32` (saturating i32→i16) → 8 i16 lanes.
- **`butterfly(in0, in1, c0, c1, *out0, *out1)`** — `inv_txfm_sse2.h:81-90`. Two-output butterfly using `idct_calc_wraplow_sse2` twice across `_mm_unpacklo_epi16`/`_mm_unpackhi_epi16` interleave with `pair_set_epi16(c0,-c1)` and `pair_set_epi16(c1,c0)`.
- **`transpose_16bit_4x4(in[4], out[2])`** — `transpose_sse2.h:95-113`. 4×4 transpose of i16 lanes in low halves of 4 `__m128i` → 2 `__m128i` (each holding 2 transposed rows). Body: `_mm_unpacklo_epi16` × 2 + `_mm_unpacklo_epi32` + `_mm_unpackhi_epi32`.
- **`idct8x8_12_transpose_16bit_4x8(in[4], out[4])`** — `inv_txfm_sse2.h:22-58`. Re-layout 4 Stage-4 outputs into 4-row 8-wide form for the column pass. ~14 unpack intrinsic calls.
- **`idct8(in[8], out[8])` (SSE2, i16 lanes)** — `inv_txfm_sse2.h:250-283`. Standard 4-stage 1-D idct8, using `butterfly` + `_mm_add_epi16`/`_mm_sub_epi16`. ~40 intrinsic calls.
- **`load_input_data4(data)`** — `inv_txfm_sse2.h:101-109`. In libvpx non-highbd builds it's `_mm_loadl_epi64` (already i16). Our coefs are i32; use `_mm_packs_epi32(_mm_loadu_si128(ptr), _mm_setzero_si128())` (sat-pack 4 × i32 → low 4 × i16). Spec-bounded inputs never saturate (see §6).
- **`write_buffer_8x8` / `recon_and_store` / `round_shift_8x8`** — `inv_txfm_sse2.h:135-181`. `(x+16)>>5` round + 8-byte dst load + zero-extend + add + sat-pack to u8 + 8-byte store. Mirrors the `__m256i` finisher in `idct8x8_avx2.rs:347-367` but on `__m128i` (each row already 8 i16 lanes — no extra pack step).

### 1.4 Constants used

| Name             | Value | Defined in (rvp9 side)                                                          |
|------------------|-------|---------------------------------------------------------------------------------|
| `DCT_CONST_BITS` | 14    | `D:\language\rvp9-2\crates\rvp9-core\src\transforms\constants.rs:14`            |
| `cospi_4_64`     | 16069 | `D:\language\rvp9-2\crates\rvp9-core\src\transforms\constants.rs:30`            |
| `cospi_8_64`     | 15137 | `:18`                                                                            |
| `cospi_12_64`    | 13623 | `:34`                                                                            |
| `cospi_16_64`    | 11585 | `:22`                                                                            |
| `cospi_20_64`    | 9102  | `:38`                                                                            |
| `cospi_24_64`    | 6270  | `:26`                                                                            |
| `cospi_28_64`    | 3196  | `:42`                                                                            |

All seven fit in i16. No new constants needed — every cospi used by
`idct8x8_12_add_kernel_sse2` is already exported.

---

## 2. Existing Rust AVX2 idct8x8 — the pattern

**File:** `D:\language\rvp9-2\crates\rvp9-core\src\dsp\x86_64\idct8x8_avx2.rs` (~803 LOC incl. tests).

### Structure

1. **Tiny intrinsic helpers** (file lines 90-123):
   - `drs(x)` — `dct_const_round_shift` (lane-parallel `(x+8192) >> 14`).
   - `wrap(x)` — modular `i16` truncation via `slli/srai` (i32 lanes).
   - `mul_const(x, c)` — `_mm256_mullo_epi32` against a broadcast `i32`.

2. **Transpose** (`transpose8x8_i32`, lines 141-186): 3-stage AVX2
   unpack pyramid for 8 × 8 i32 transpose.

3. **1-D idct8** (`idct8_avx2`, lines 197-254): mirrors the scalar
   `idct8` 1:1 across 8 ymm registers (one per row).

4. **Outer kernel** (`idct8x8_add_avx2_inner`, lines 303-368): load 8
   rows → transpose → row pass → transpose → column pass → finisher.

5. **Safe wrapper** (`idct8x8_add_avx2`, lines 380-389): calls the
   `unsafe fn` inner inside a single `unsafe { ... }` block with a
   `// SAFETY:` comment that names the OnceLock + caps invariant.

6. **`pub use` in `mod.rs`** (`x86_64/mod.rs:57`, `:106`):
   `pub mod idct8x8_avx2;` + `pub use idct8x8_avx2::idct8x8_add_avx2;`.

7. **Dispatch wiring** (`dsp/dispatch.rs:676`, `:1213-1225`, `:1916`,
   `:1976-1978`):
   - Field declaration on `SimdDispatch` (`pub(crate)` fn pointer slot).
   - `build()` selects AVX2 wrapper if `caps.avx2`, scalar otherwise.
   - Field included in struct literal.
   - Public accessor returning the fn pointer.

8. **Tests** (lines 394-803): 5-strategy proptest, single-coefficient
   impulse sweep, dst-pointer-skew regression, dispatch round-trip, and
   row/column-dominant directed tests. All gated on
   `is_x86_feature_detected!("avx2")`.

### Key architectural choice that *differs* for SSE2 _12

The AVX2 idct8x8 uses **i32 lanes** end-to-end (modular `wrap` ≠ libvpx's
saturating `_mm_packs_epi32`). This is **deliberate** — see the comment
at `idct8x8_avx2.rs:42-58`:

> "libvpx's SSE2 kernel uses `_mm_packs_epi32` (saturating) which
> **diverges from the C reference on out-of-range intermediates**. Our
> proptest fixtures exercise such inputs (the `pathological_strategy`
> includes `[i16::MAX; 64]`), so we MUST match the C truncation
> semantics, not libvpx's SSE2 saturation."

For the new SSE2 _12 kernel we cannot easily replicate this trick (the
whole point of porting the SSE2 kernel is to use the dense i16/`madd`
pipeline). The next-session brief acknowledges this and accepts the
saturating semantics as **the libvpx C reference for in-bounds bitstream
inputs**:

> "The new _12 SSE2 kernel should match the C reference for in-bounds
> bitstream inputs and is acceptable for production paths (matching
> vpxdec behavior)."

This means:
- The differential oracle is **libvpx's `vpx_idct8x8_12_add_sse2`**, not
  the scalar C reference, for the proptest range.
- Our scalar oracle (`idct8x8_64_add` with rows 4..7 zeroed) only
  matches when intermediates stay in i16 range — **proptest must use a
  reduced range** (see §6).

---

## 3. Dispatch slot infrastructure

### Where the table lives

`D:\language\rvp9-2\crates\rvp9-core\src\dsp\dispatch.rs`. The
`SimdDispatch` struct (`:664-1110-ish`) holds one fn-pointer field per
kernel.

### Existing idct8x8_add field (the template)

`dispatch.rs:670-676`:

```rust
/// G4.1 — 8×8 inverse DCT + residual add, full-density (eob > 1).
/// Signature mirrors the scalar reference
/// [`crate::transforms::inverse::idct8x8_64_add`]. On x86_64 AVX2
/// CPUs this points at the safe wrapper
/// [`crate::dsp::x86_64::idct8x8_add_avx2`]; otherwise it points
/// at the scalar reference.
pub(crate) idct8x8_add: fn(input: &[i32; 64], dst: &mut [u8], stride: usize),
```

`build()` clause at `:1212-1225`:

```rust
#[cfg(target_arch = "x86_64")]
let idct8x8_add: fn(&[i32; 64], &mut [u8], usize) = if caps.avx2 {
    crate::dsp::x86_64::idct8x8_add_avx2
} else {
    scalar_idct8x8_add
};
#[cfg(not(target_arch = "x86_64"))]
let idct8x8_add: fn(&[i32; 64], &mut [u8], usize) = {
    let _ = caps;
    scalar_idct8x8_add
};
```

Struct literal at `:1916`: `idct8x8_add,`.

Accessor at `:1971-1978`:

```rust
#[inline]
#[must_use]
pub fn idct8x8_add(&self) -> fn(&[i32; 64], &mut [u8], usize) {
    self.idct8x8_add
}
```

### Status of the new slot

The R.6 attempt (commit **`89a5d51`** — the revert) **did not commit
the failed attempt's slot definition**. The revert's diff (verified via
`git show 89a5d51 --stat`) only modified comments in
`reconstruct.rs` / `reconstruct_inter.rs`:

```
crates/rvp9-decoder/src/reconstruct.rs       | 6 ++++++
crates/rvp9-decoder/src/reconstruct_inter.rs | 3 +++
```

So **there is no `idct8x8_12_add` field / accessor / build clause in
the current tree** — the implementation agent must add the full
slot infrastructure (one field declaration, one build clause, one
struct-literal entry, one accessor). The pattern is exactly the
`idct8x8_add` template above; signature is identical
(`fn(&[i32; 64], &mut [u8], usize)`).

For the scalar fallback the implementation agent must also add a
scalar `idct8x8_12_add` to `crates/rvp9-core/src/transforms/inverse.rs`
(grep confirms there is no `idct8x8_12` symbol there today). The
scalar implementation is a literal port of `vpx_idct8x8_12_add_c`
from `third_party/libvpx/vpx_dsp/inv_txfm.c:350-373` and is ~20 LOC.

---

## 4. Call sites

### Intra: `crates/rvp9-decoder/src/reconstruct.rs:405-432`

```rust
TxSize::Tx8x8 => {
    let coefs: &[i32; 64] = input.try_into().expect("Tx8x8 input is 64 coefs");
    if tx_type == TxType::DctDct {
        if eob == 1 {
            idct8x8_1_add(coefs[0], dest, stride);
        } else {
            // libvpx splits eob<=12 → idct8x8_12_add; we ship
            // only the full _64_add variant (transforms LLD
            // §3.4 lists 12-bucket kernels but only _1 and
            // _64 are landed in inverse.rs today). Bit-exact
            // result; just slower for sparse blocks.
            //
            // R.6 (2026-05-23) tried adding a scalar
            // idct8x8_12_add but the scalar reduction (4 row
            // passes vs 8) doesn't beat the AVX2 _64 SIMD
            // kernel. Reverted; future AVX2 _12 ship can be
            // wired in if and when it's available.
            //
            // G4.1 — dispatch through SimdDispatch so the AVX2
            // implementation runs on supported CPUs.
            (dispatch.idct8x8_add())(coefs, dest, stride);
        }
    } else {
        // G4.3a — hybrid 8×8 via SimdDispatch (scalar-only;
        // Wave E lands AVX2).
        (dispatch.iht8x8_add())(coefs, dest, stride, tx_type);
    }
}
```

**The `eob <= 12` gate is NOT currently in place.** Today the dispatch
goes:
- `eob == 1` → `idct8x8_1_add` (DC-only).
- `eob > 1` (any value) → `idct8x8_add()` (the AVX2 _64 wrapper).

The implementation agent must add an `else if eob <= 12 { … }` arm
that calls `(dispatch.idct8x8_12_add())(coefs, dest, stride)`.

### Inter: `crates/rvp9-decoder/src/reconstruct_inter.rs:824-836`

```rust
TxSize::Tx8x8 => {
    let coefs: &[i32; 64] = input.try_into().expect("Tx8x8 input is 64 coefs");
    if eob == 1 {
        idct8x8_1_add(coefs[0], dest, stride);
    } else {
        // R.6 (2026-05-23) tried wiring a scalar idct8x8_12_add
        // for eob<=12 but it regressed bench — scalar < AVX2 _64
        // here. Slot remains in dispatch for a future AVX2 promotion.
        // G4.1 — dispatch through SimdDispatch so the AVX2
        // implementation runs on supported CPUs.
        (dispatch.idct8x8_add())(coefs, dest, stride);
    }
}
```

Same shape as intra. Same gate to add. (The comment claims the slot
"remains in dispatch" but per §3 it actually doesn't — the slot was
reverted along with everything else; the comment is stale.)

> NOTE: the inter call site line drifted from the brief's stated 818 to
> 824 (it's now the first executable statement after the eob==1 arm).
> The original 818 line in this file (today) is mid-Tx4x4 block — the
> brief's reference was from before some doc-shuffle. The
> implementation agent should grep for `TxSize::Tx8x8` in
> `reconstruct_inter.rs` to land at the right spot.

---

## 5. Scalar reference for validation

Scalar reference for the full-density 8×8 path:
`D:\language\rvp9-2\crates\rvp9-core\src\transforms\inverse.rs:207-240`

```rust
pub fn idct8x8_64_add(input: &[i32; 64], dest: &mut [u8], stride: usize) {
    // Row pass over all 8 rows of input.
    // Column pass + add for all 8 columns.
}
```

The scalar `idct8` (1-D core) is at lines 144-198 and uses `i64`
intermediate math + `wraplow` (modular i16 truncation). It does NOT
saturate; out-of-range intermediates wrap modularly.

**For validation of the new SSE2 `_12` kernel:**

The libvpx C reference `vpx_idct8x8_12_add_c`
(`third_party/libvpx/vpx_dsp/inv_txfm.c:350-373`) shows that the only
behavioural difference from `_64` is **skipping the row pass on rows
4..7** (rows 4..7 are zero-initialised in the local `out` buffer, then
the column pass runs as usual). Bit-exact equivalence:

> `idct8x8_12_add_c(input)` ≡ `idct8x8_64_add(input_with_rows_4_7_zeroed)`.

So for validation we can:
1. Run `idct8x8_64_add` on a coefficient array with `input[32..64] = 0`
   (rows 4..7 zeroed).
2. Run `idct8x8_12_add_sse2` on the *original* input (which is allowed
   to have non-zeros in rows 0..3 only by caller contract — the kernel
   ignores rows 4..7 anyway).
3. Compare dst byte-for-byte.

**Confirmed: this validation path is possible** — both sides are pure
functions of `(input, dst_init, stride)` and the scalar reference is
already in the codebase.

> Caveat: as called out in §2, the SSE2 kernel uses
> `_mm_packs_epi32` (saturating). For bitstream-bounded inputs (`-32768
> ≤ coef ≤ 32767`, with most ≤ a few thousand in magnitude), no
> intermediate saturates the i16 envelope and the two implementations
> are bit-exact. For synthetic test inputs near `i16::MAX` across many
> coefficients, intermediates can exceed i16 range and the SSE2 kernel
> will diverge from the scalar reference. The proptest range must be
> bounded accordingly (see §6).

---

## 6. Coefficient width concern

### Where dq_coefs are assembled

Search:

```
crates/rvp9-decoder/src/reconstruct.rs:406  let coefs: &[i32; 64] = ...
crates/rvp9-decoder/src/reconstruct_inter.rs:825  let coefs: &[i32; 64] = ...
```

The dispatch slot signature is
`fn(input: &[i32; 64], dst: &mut [u8], stride: usize)` (see
`dispatch.rs:676`). Coefficients reach the kernel as `i32` — they
were stored as i32 throughout the dequant pipeline.

The new SSE2 kernel's input slot must use the same signature so the
dispatch table is type-uniform; the kernel then packs to i16 internally
at load time.

### Saturating-pack at load

`_mm_packs_epi32` (saturating pack 8 i32 → 8 i16) is the libvpx
primitive for the load. Use:

```rust
let v = _mm_loadu_si128(ptr as *const __m128i);  // 4 × i32
let packed = _mm_packs_epi32(v, _mm_setzero_si128());  // low 4 lanes = i16
```

This produces the same `__m128i` layout as libvpx's
`load_input_data4` in the highbd build (the non-highbd build skips the
pack since `tran_low_t = int16_t` there — but our codebase always uses
i32 so we always pack).

### Bitstream bounds (proptest range)

VP9 spec §8.7.1 and libvpx `inv_txfm.c` use `tran_low_t = int16_t` by
default. The post-dequant coefficient range is bounded to `[-32768,
32767]` (i16 range) before reaching the inverse transform.

Empirically the existing AVX2 idct8x8 proptest at
`idct8x8_avx2.rs:494-501` documents this exact bound:

> "Uniform i32 coefficients clamped to `[-32768, 32767]`. VP9 spec /
> libvpx bound the post-dequant coefficient range to i16. Even though
> our scalar reference takes `i32`, the bit-exact contract only applies
> within that range; out-of-range inputs are undefined behaviour for
> libvpx too."

**For the new _12 SSE2 kernel proptest:**

- `uniform_strategy_12`: range `[-2048, 2047]` (i12) — realistic post-dequant; guarantees no Stage-2/3 intermediate exceeds i16. Production-realistic.
- `sparse_strategy_12`: up to 4 non-zero coefs in positions `0..32` (top-left 4×4) — the eob ≤ 12 contract.
- `eob_bucket_strategy_12`: K ∈ {1, 2, 8, 12} non-zero coefs at low raster positions — dispatcher seams.
- `high_magnitude_strategy_12`: magnitudes `[15000, 16384]` (≈ i14, below the sat-pack divergence point) — stress high end without diverging from scalar.
- `pathological_strategy_12`: OMIT — `[i16::MAX; 64]` saturates the libvpx kernel; the brief explicitly accepts this.

For positions: the eob ≤ 12 contract per libvpx says rows 4..7 are
zero (i.e. coefs `[32..64]` are zero). The proptest strategies above
should force `input[32..64] = 0` in every sample.

> **If the implementation agent wants tighter coverage:** add a small
> "differential against libvpx" hand-rolled C oracle behind a build
> feature, comparing against `vpx_idct8x8_12_add_sse2` directly. This
> is **out of scope** for Task A's initial port; the scalar oracle is
> sufficient for the in-bounds range.

---

## 7. Proposed work breakdown for the implementation agent

Each sub-task is sized so the agent can write the code AND a unit test
before moving to the next. The order below is bottom-up (helpers
first) so each new piece can be tested in isolation against a scalar
reference.

1. **Add `pair_set_epi16` helper** in new file
   `crates/rvp9-core/src/dsp/x86_64/idct8x8_12_sse2.rs`.
   - Inline `unsafe fn pair_set_epi16(a: i16, b: i16) -> __m128i`.
   - Unit test: extract 8 i16 lanes via `transmute` or
     `_mm_store_si128` into a `[i16; 8]` buffer and assert
     `[a,b,a,b,a,b,a,b]`.

2. **Add `idct_madd_round_shift_sse2`** + unit test.
   - Scalar oracle: lane-wise `(a0*c0 + a1*c1 + 8192) >> 14` over 8
     i16 inputs and 8 i16 cospi lanes (`_mm_madd_epi16` consumes pairs
     of adjacent i16 lanes).
   - Proptest: random 8 × i16 input / pair-set cospi → assert lane-wise
     equivalence to scalar oracle.

3. **Add `idct_calc_wraplow_sse2`** + unit test.
   - Composed of two `idct_madd_round_shift_sse2` + one
     `_mm_packs_epi32`. Scalar oracle: same as above, then
     `saturating_clamp_to_i16` on each lane.
   - Proptest: bounded inputs (i14 magnitude) → assert bit-exact match
     to scalar oracle.

4. **Add `butterfly` helper** + unit test.
   - Two-output butterfly using `idct_calc_wraplow_sse2` twice over
     `_mm_unpacklo_epi16` / `_mm_unpackhi_epi16` interleave.
   - Proptest: bounded i14 inputs, single fixed cospi pair → assert
     lane-wise match against scalar `dct_const_round_shift` butterfly.

5. **Add `transpose_16bit_4x4`** + unit test.
   - 4 `__m128i` in (low 4 lanes = data), 2 `__m128i` out.
   - Unit test: directed `[[0,1,2,3], [10,11,12,13], [20..23],
     [30..33]]` → assert output lanes are `[00 10 20 30 01 11 21 31]`
     and `[02 12 22 32 03 13 23 33]` (per libvpx comments at
     `transpose_sse2.h:97-112`).

6. **Add `load_input_data4`** equivalent + unit test.
   - `_mm_loadu_si128` of 4 i32 + `_mm_packs_epi32(v, zero)`.
   - Unit test: bounded inputs in `[-32768, 32767]` → assert low 4 i16
     lanes match input as i16, high 4 lanes are 0.

7. **Add the inner `idct8` (SSE2 i16 lanes)** + unit test.
   - Direct line-by-line port of `inv_txfm_sse2.h:250-283`. Reuses
     `butterfly` from step 4.
   - Differential test: take random `[i16; 64]` input (bounded
     magnitude), run scalar `idct8` per row, then compare to the SSE2
     output for each of 8 rows packed into 8 × `__m128i`.

8. **Add `idct8x8_12_transpose_16bit_4x8`** + unit test.
   - Direct port of `inv_txfm_sse2.h:22-58`.
   - Directed test: lane patterns from the comments verified by lane
     extraction.

9. **Compose `idct8x8_12_add_sse2(coefs, dst, stride)`** — outer
   kernel — + finisher (write_buffer_8x8 logic).
   - Differential proptest against scalar `idct8x8_64_add` with rows
     4..7 zeroed (per §5).
   - Strategies: as listed in §6 (uniform-i12, sparse, eob-bucket,
     high-magnitude-i14).
   - Single-impulse sweep over positions 0..32 (eob ≤ 12 ⇒ top-left
     4×4 sub-block only).
   - dst-pointer-skew regression (matching the existing AVX2 test).

10. **Wire dispatch slot** in `crates/rvp9-core/src/dsp/dispatch.rs`:
    - New field `idct8x8_12_add: fn(&[i32; 64], &mut [u8], usize)`.
    - `build()` clause: SSE2 wrapper if `caps.sse2` else scalar.
      (Caps already track SSE2 separately — check `dsp/caps.rs`.)
    - Add scalar `idct8x8_12_add` to `transforms/inverse.rs` (port of
      `vpx_idct8x8_12_add_c`).
    - Add to struct literal + public accessor.
    - Add `pub use idct8x8_12_sse2::idct8x8_12_add_sse2;` in
      `dsp/x86_64/mod.rs`.
    - Forced-caps test (mirror
      `forced_scalar_caps_installs_scalar_idct8x8` at
      `dispatch.rs:2570`).

11. **Activate at call sites**:
    - `reconstruct.rs` Tx8x8 arm: add `else if eob <= 12` branch
      before the catch-all `(dispatch.idct8x8_add())` call.
    - `reconstruct_inter.rs` Tx8x8 arm: same.
    - Conformance gate: `cargo test -p rvp9-harness --test
      conformance --release` (201/201) — required per the bit-exact
      gate in the next-session brief.

12. **Bench gate**:
    - Run statistical bench framework (Track-S Task C — already shipped
      per the git log: commit `405efe0`).
    - Acceptance: `testsrc_1080p30 t=4` improves ≥ 5% with `p < 0.05`.

---

## 8. Validation acceptance for each sub-task

1. `pair_set_epi16` — extract 8 i16 lanes via `_mm_storeu_si128` into `[i16;8]`, assert `[a,b,a,b,a,b,a,b]`.
2. `idct_madd_round_shift_sse2` — bounded-i14 proptest vs scalar `(in_lo[0]*c_lo[0] + in_lo[1]*c_lo[1] + 8192) >> 14` (i64 math) per i32 lane.
3. `idct_calc_wraplow_sse2` — bounded-i14 proptest vs scalar `(saturating-cast each i32 to i16)`.
4. `butterfly` — bounded-i14 proptest vs scalar `inv_txfm.c` butterfly (i64 mul + DCT_CONST_ROUNDING + sat-clamp).
5. `transpose_16bit_4x4` — directed: lane extraction confirms the C-doc mapping (`[00 10 20 30 01 11 21 31]` etc.).
6. `load_input_data4` equivalent — bounded i16-range i32 proptest vs `saturating_cast::<i32,i16>`.
7. inner `idct8` (SSE2 i16 lanes) — bounded-i14 proptest: pack 8 inputs per `__m128i`, run SSE2 idct8, compare to scalar `idct8` (`transforms/inverse.rs:144`) per row.
8. `idct8x8_12_transpose_16bit_4x8` — directed: lane extraction per `inv_txfm_sse2.h:22-58` comments.
9. outer `idct8x8_12_add_sse2` — differential proptest vs scalar `idct8x8_64_add` with `input[32..64] = 0`. Plus single-impulse sweep over positions 0..32, plus dst-skew regression.
10. dispatch wiring — `forced_scalar_caps_installs_scalar_idct8x8_12` test mirroring `dispatch.rs:2570`. Forced `caps.sse2 = false` returns scalar fn pointer.
11. call-site activation — conformance (201/201) + Profile-1 + frame_parallel_regression + tile_threading.
12. perf gate — Track-S MWU stat bench (commit `405efe0`); `p < 0.05`, ≥ 5% improvement on `testsrc_1080p30 t=4`.

---

## 9. Risk register

- **R1 — Saturation semantics divergence (Medium).** `_mm_packs_epi32` saturates at i16; scalar uses modular `wraplow`. Synthetic near-i16::MAX proptest inputs will diverge. Mitigation: bound proptest to i14 (§6); document the divergence in the new file's docstring mirroring `idct8x8_avx2.rs:42-58`; directed regression at `±16384` to confirm bit-exact at the boundary.
- **R2 — Transpose lane confusion (High).** With 8 i16 lanes per register, the two transposes and the row↔column orientation between passes are easy to get wrong with silent almost-right output. Mitigation: single-impulse sweep over all 32 positions (mirror `idct8x8_avx2.rs:651-661`) — historically the most effective transpose-bug catcher. Verify the C-doc lane mappings before writing.
- **R3 — Rounding direction (Low).** `(x + 8192) >> 14` with arithmetic shift; a typo `1<<14` would be the canonical IDCT bug. Mitigation: named const `DCT_CONST_ROUNDING: i32 = 1 << 13`; step 2 proptest catches this immediately.
- **R4 — SSE2 caps detection (Medium).** Unknown if `Caps` carries a separate `sse2` flag. Mitigation: read `dsp/caps.rs` before sub-task 10. If absent, gate on `caps.avx2` for now (every AVX2 host has SSE2; conservative + monotonic) and add a TODO.
- **R5 — Perf regression vs AVX2 _64 (High).** R.6 already saw a *scalar* _12 lose to AVX2 _64. Real win on SSE2 _12 is the saved column-pass on rows 4..7 (zero-skip) — theoretically ~30-50% Tx8x8 path saving. Mitigation: stat bench at sub-task 12; if testsrc_1080p30 t=4 doesn't improve ≥ 5% (MWU p<0.05), abandon (matches R.6 pattern). Pre-flight: hand-count ops vs existing AVX2 _64 to predict delta before coding.
- **R6 — Mixed-ISA dispatch (Low).** First SSE2-only kernel in the table; previously all are AVX2-or-scalar. Architecturally clean — each kernel picks its own ISA. Mitigation: land SSE2 first; AVX2 _12 follow-up only if SSE2 ships and benches well.

---

## 10. Estimated implementation budget

| Item | LOC estimate (new code) | Notes |
|---|---|---|
| `dsp/x86_64/idct8x8_12_sse2.rs` (file body) | ~220 | Helpers (~80) + inner kernel (~80) + outer + finisher (~60). |
| Tests in same file | ~250 | Mirrors the existing `idct8x8_avx2.rs` test scaffold (proptest + single-impulse + dst-skew + dispatch round-trip). |
| `transforms/inverse.rs` — scalar `idct8x8_12_add` | ~25 | Direct port of `vpx_idct8x8_12_add_c`. |
| `dsp/dispatch.rs` — new slot | ~30 | Field + build clause + struct literal + accessor + forced-caps test. |
| `dsp/x86_64/mod.rs` — module + use lines | ~2 | Trivial. |
| `reconstruct.rs` + `reconstruct_inter.rs` — eob<=12 branch | ~10 | Two ~5-line `else if` arms; remove the stale R.6 comment block. |
| **Total new code** | **~535 LOC** | Of which ~250 LOC is tests. |

**Agent-turn breakdown (rough):**

| Phase | Effort |
|---|---|
| Sub-tasks 1-6 (helpers + load) | 4-6 small turns. Each helper is ~10-20 LOC + ~30 LOC test; very low risk individually. |
| Sub-task 7 (inner idct8 SSE2) | 1-2 medium turns. The largest single piece; ~80 LOC + careful proptest. |
| Sub-task 8 (idct8x8_12_transpose_16bit_4x8) | 1 small turn. ~25 LOC + directed lane test. |
| Sub-task 9 (outer kernel + finisher + proptest) | 1-2 medium turns. Where bugs land — proptest will catch transpose / scaling errors here, then iterate. |
| Sub-tasks 10-11 (wiring + scalar reference) | 1 medium turn. ~70 LOC. |
| Sub-task 12 (bench gate) | 1 turn worth of waiting (bench runs ~5 min × 5 fixtures × 5 iterations). |
| **Total** | **~9-13 turns**, matches the brief's "4 hours focused work plus testing/debugging" budget. |

### Unexpected complications discovered during prep

1. **No live `idct8x8_12_add` slot.** `git show 89a5d51` confirms the R.6 slot was wholly reverted (the brief's "R.6 slot re-add" wording is misleading). Implementation agent adds the slot from scratch.
2. **No live scalar `idct8x8_12_add` either** — absent from `transforms/inverse.rs`. ~25 LOC to add (for non-x86_64 fallback).
3. **eob<=12 dispatch gate is NOT in place.** Both intra and inter Tx8x8 arms currently route all `eob > 1` to the AVX2 _64 kernel. Implementation agent adds the `else if eob <= 12` arms.
4. **Inter call site drifted to line ~824 (not 818).** Grep for `TxSize::Tx8x8` to find.
5. **`caps.sse2` plumbing may not exist** — read `dsp/caps.rs` first. If absent, gate on `caps.avx2` for now and add a TODO.
6. **Saturating-pack semantics are explicitly accepted as divergent** from the scalar reference at the upper proptest range. This is a deliberate choice per the brief — document it prominently in the new file's docstring (mirror the inverse-direction note at `idct8x8_avx2.rs:42-58`).

### Wall-clock distribution

Helpers (sub-tasks 1-6) are trivial; **sub-task 9 (outer composition + proptest debug) is where the wall-clock sits**. Past idct ports (G4-γ, Wave F F2) typically needed 2-3 proptest iterations before bit-exact match (each ~1-2 min compile + ~30 sec test). Libvpx kernel is fully exposed (no inline asm), every helper has a scalar oracle, and `idct8x8_avx2.rs` is a strong structural template — no new SIMD patterns to invent.

---

## End of plan
