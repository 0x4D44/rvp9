//! 8×8 inverse-DCT + add — x86_64 AVX2 implementation (G4.1 pilot).
//!
//! This is the **first** SIMD kernel landed in rvp9-core and the
//! reference for the safe-wrapper convention every subsequent
//! G4.x rollout mirrors:
//!
//! 1. The body uses `#[target_feature(enable = "avx2")]` on an
//!    `unsafe fn`. (`target_feature_1.1` is unstable on stable Rust
//!    today, so the function MUST be `unsafe`; see HLD V4 §G4
//!    RS3-M1.)
//! 2. A safe wrapper [`idct8x8_add_avx2`] performs a single
//!    `unsafe { ... }` call into the inner. The SAFETY comment names
//!    the OnceLock + caps invariant that makes the call sound — see
//!    [`crate::dsp::dispatch`] for the invariant.
//! 3. The dispatch slot
//!    [`crate::dsp::SimdDispatch::idct8x8_add`] points at the SAFE
//!    wrapper. Call sites stay safe Rust.
//!
//! ## Bit-exactness vs scalar reference
//!
//! VP9 conformance requires bit-exact match to libvpx's C reference.
//! The C reference [`crate::transforms::inverse::idct8x8_64_add`]
//! does the row + column 1-D idct8 in i32 with explicit `wraplow`
//! (`int16_t`-truncation) at every intermediate store, then a final
//! `(out + 16) >> 5` ROUND_POWER_OF_TWO with `clip_pixel_add`.
//!
//! For the G4.1 pilot we keep the i32 transform math identical to
//! the scalar reference (1:1 reuse of [`crate::transforms::inverse::idct8`])
//! and AVX2-accelerate the *load + final add-clamp-pack* stage —
//! the part where AVX2 actually moves the needle (eight u8 lanes
//! per row mean the inner loop is store-bandwidth limited in scalar,
//! but a vectorised pack saturates in two `vpackus` + one `vmovq`).
//! This pattern is what libvpx's `vpx_idct8x8_64_add_sse2` does in
//! its `write_buffer_8x8` finisher and is the cleanest way to
//! demonstrate the wrapper convention without re-deriving the
//! butterfly arithmetic in 16-bit `_mm_madd_epi16` form (which is
//! G4.2 work).
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Loads from `dst` are unaligned via `_mm_loadl_epi64`; stack scratch
//! `final_out` is read via `_mm256_loadu_si256`. The grep sentinel
//! below is checked by `scripts/check-simd-sentinels.sh` / `.ps1`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_loadl_epi64, _mm_packus_epi16, _mm_setzero_si128,
    _mm_storel_epi64, _mm_unpacklo_epi8, _mm256_add_epi32, _mm256_castsi256_si128,
    _mm256_loadu_si256, _mm256_packs_epi32, _mm256_permute4x64_epi64, _mm256_set1_epi32,
    _mm256_srai_epi32,
};

use crate::transforms::inverse::idct8;

/// AVX2 implementation of the 8×8 inverse-DCT + residual add.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`idct8x8_add_avx2`]) when `caps.avx2 == true`, and
/// caps are detected once at OnceLock init time.
///
/// `dst` must have at least `7 * stride + 8` bytes of valid storage.
///
/// # Implementation
///
/// Row + column idct8 math uses the scalar reference (i32, bit-exact
/// `wraplow`). AVX2 accelerates the final `(out + 16) >> 5` + add +
/// clip stage:
///
/// ```text
///   for r in 0..8:
///     loadu 8x i32 from row r           ; vmovdqu ymm
///     add round constant (+16)          ; vpaddd
///     srai 5                            ; vpsrad
///     pack to i16 (saturated)           ; vpackssdw + permute
///     loadl 8x u8 from dst+r*stride     ; vmovq
///     unpack to i16                     ; vpunpcklbw
///     add transform output              ; vpaddw
///     pack to u8 (saturated)            ; vpackuswb
///     storel 8x u8 to dst+r*stride      ; vmovq
/// ```
///
/// All loads/stores from / to `dst` are unaligned. See HLD V6 §G4
/// M6-6.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct8x8_add_avx2_inner(input: &[i32; 64], dst: &mut [u8], stride: usize) {
    debug_assert!(dst.len() >= stride * 7 + 8, "dst too small for 8x8 add");

    // ----- Stage 1: row pass (scalar idct8, bit-exact). -----
    let mut intermediate = [0i32; 64];
    for r in 0..8 {
        let row_in: &[i32; 8] = input[r * 8..r * 8 + 8].try_into().unwrap();
        let row_out_mut: &mut [i32; 8] = (&mut intermediate[r * 8..r * 8 + 8]).try_into().unwrap();
        idct8(row_in, row_out_mut);
    }

    // ----- Stage 2: column pass (scalar idct8). -----
    // Produces 8 columns of 8 i32 outputs. Stored in row-major
    // `final_out[r * 8 + c]` so the AVX2 row finisher can `loadu` a
    // whole row in one ymm op.
    let mut final_out = [0i32; 64];
    for c in 0..8 {
        let col_in: [i32; 8] = [
            intermediate[c],
            intermediate[8 + c],
            intermediate[16 + c],
            intermediate[24 + c],
            intermediate[32 + c],
            intermediate[40 + c],
            intermediate[48 + c],
            intermediate[56 + c],
        ];
        let mut col_out = [0i32; 8];
        idct8(&col_in, &mut col_out);
        for r in 0..8 {
            final_out[r * 8 + c] = col_out[r];
        }
    }

    // ----- Stage 3: AVX2-accelerated add + clamp + pack + store. -----
    //
    // For each row:
    //   1. Load 8 i32 outputs into a ymm register (unaligned, from a
    //      stack array `final_out`).
    //   2. Add the round constant +16 and shift right by 5
    //      (ROUND_POWER_OF_TWO equivalent, signed).
    //   3. Permute + pack i32 → i16 (saturated). The transform output
    //      bound fits in i16 by VP9 spec.
    //   4. Load 8 u8 destination samples (unaligned).
    //   5. Zero-extend u8 → i16, add, saturate-pack i16 → u8.
    //   6. Store 8 u8 back.
    let round = _mm256_set1_epi32(16);
    let zero = _mm_setzero_si128();
    for r in 0..8 {
        let row_ptr = final_out.as_ptr().add(r * 8) as *const __m256i;
        // (1) unaligned load 8 i32 from `final_out` (stack scratch).
        let mut out_i32 = _mm256_loadu_si256(row_ptr);
        // (2) add round constant +16, shift right arithmetic by 5.
        out_i32 = _mm256_add_epi32(out_i32, round);
        out_i32 = _mm256_srai_epi32(out_i32, 5);
        // (3) saturate-pack i32 → i16. `_mm256_packs_epi32(x, x)` is
        // per-128-bit-lane: the ymm result is
        //   lane 0 (low 128 bits)  = [a0, a1, a2, a3, a0, a1, a2, a3] as i16
        //   lane 1 (high 128 bits) = [a4, a5, a6, a7, a4, a5, a6, a7] as i16
        // To get a single 128-bit `__m128i` with the 8 i16s in
        // natural order [a0..a7] we permute by 64-bit qwords with
        // control 0xD8 = 0b_11_01_10_00 → qword order [0, 2, 1, 3]:
        //   - imm[1:0]=0 → low qword = input qword 0 = [a0..a3]
        //   - imm[3:2]=2 → next qword = input qword 2 = [a4..a7]
        //   - imm[5:4]=1, imm[7:6]=3 → high lane ignored; we only
        //     read the low 128 below.
        let packed_lanes = _mm256_packs_epi32(out_i32, out_i32);
        let packed_natural = _mm256_permute4x64_epi64(packed_lanes, 0xD8);
        // Cast: take low 128 bits = 8 × i16 in natural order.
        let out_i16_lo = _mm256_castsi256_si128(packed_natural);

        // (4) load 8 u8 from dst row (unaligned).
        // G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
        let dst_row_ptr = dst.as_mut_ptr().add(r * stride) as *const __m128i;
        let dst_u8 = _mm_loadl_epi64(dst_row_ptr);
        // (5) zero-extend u8 → i16 (only the low 8 bytes are valid).
        let dst_i16 = _mm_unpacklo_epi8(dst_u8, zero);
        // Add transform output and saturate-pack to u8.
        let sum_i16 = _mm_add_epi16(out_i16_lo, dst_i16);
        let sum_u8 = _mm_packus_epi16(sum_i16, sum_i16);
        // (6) store 8 u8 back (unaligned).
        let dst_row_ptr_mut = dst.as_mut_ptr().add(r * stride) as *mut __m128i;
        _mm_storel_epi64(dst_row_ptr_mut, sum_u8);
    }
}

/// Safe wrapper for the AVX2 idct8x8 inner kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct8x8_64_add`], so the
/// `SimdDispatch::idct8x8_add` slot can hold either.
///
/// The wrapper is `pub` (rather than `pub(crate)`) so unit tests in
/// the same crate can call it directly to do scalar-vs-AVX2
/// differential testing without going through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct8x8_add_avx2(input: &[i32; 64], dst: &mut [u8], stride: usize) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime (a CPU doesn't lose features mid-run).
    // The unit-test path in `idct8x8_avx2_tests` gates each call on a
    // runtime `is_x86_feature_detected!("avx2")` check, so calls from
    // tests are equally sound.
    unsafe { idct8x8_add_avx2_inner(input, dst, stride) }
}

// Miri cannot interpret AVX2 intrinsics — see the parallel comment
// in `dc_pred_4x4_avx2.rs`. Scalar reference paths (the
// `transforms::inverse` family) still run under miri unchanged.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `idct8x8_add`.
    //!
    //! Coverage strategy (per HLD V5 §G4 TQ4-M3 corrected EOB seam
    //! set for idct8x8 = {1, 2, 12, 13, 64}):
    //!
    //! - **`idct8x8_scalar_eq_avx2_proptest`** — proptest with five
    //!   stratified strategies (`prop_oneof!`):
    //!   1. `uniform_strategy` — random over the i32 coefficient
    //!      range (clamped to `[-32768, 32767]` per VP9 spec — the
    //!      dequantised coefficient bound from inv_txfm.c).
    //!   2. `sparse_strategy` — 1..=4 non-zero coefficients at random
    //!      positions.
    //!   3. `eob_bucket_strategy` — K ∈ {1, 2, 12, 13, 64} non-zero
    //!      coefficients to exercise dispatcher seams.
    //!   4. `high_magnitude_strategy` — coefficients near ±i16::MAX
    //!      (saturation paths).
    //!   5. `pathological_strategy` — all-equal / all-min / all-max /
    //!      alternating sign.
    //!
    //! - **`idct8x8_all_zero_input`** — DC-only fast path with
    //!   `input[0] = 0`.
    //!
    //! - **`idct8x8_single_coefficient_impulses`** — sweep a single
    //!   non-zero coefficient (`value = 100`) across every position
    //!   0..64 and confirm scalar / AVX2 agreement at each.
    //!
    //! - **`idct8x8_misaligned_dst_pointers`** — `dst` *backing
    //!   pointer* offset at byte offsets {0, 1, 3, 7, 15}. The kernel
    //!   uses only 8-byte loads (`_mm_loadl_epi64`), so this is a
    //!   per-offset arithmetic regression guard, NOT an alignment-fault
    //!   test (alignment exercise is deferred to G4.2 when 16/32-byte
    //!   loads land). Pointer arithmetic uses a `Box<[u8; 80]>`
    //!   over-allocation; sub-slicing then guarantees the requested
    //!   offset on the slice's `as_ptr`.
    //!
    //! Every test is gated on `is_x86_feature_detected!("avx2")` and
    //! returns early on non-AVX2 CPUs (CI runner check).

    use super::idct8x8_add_avx2;
    use crate::transforms::inverse::idct8x8_64_add;
    use proptest::prelude::*;

    /// Skip-marker macro: bail out of a test on a non-AVX2 host.
    /// On the project's tier-1 CI/dev hosts AVX2 is always present;
    /// the early return is a belt-and-braces guard for older hardware.
    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Compute the AVX2 output for one set of coefficients + starting
    /// destination state. Returns the final dst bytes.
    ///
    /// Callers are expected to gate on `is_x86_feature_detected!("avx2")`
    /// via the `require_avx2!()` macro at the start of each test; the
    /// internal `assert!` here is defensive — if it ever fires it means
    /// a test was added that forgot the macro gate.
    fn run_avx2(input: &[i32; 64], dst_init: &[u8; 64]) -> [u8; 64] {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host (missing require_avx2!() gate in caller)"
        );
        let mut dst = *dst_init;
        idct8x8_add_avx2(input, &mut dst, 8);
        dst
    }

    /// Same shape via the scalar reference.
    fn run_scalar(input: &[i32; 64], dst_init: &[u8; 64]) -> [u8; 64] {
        let mut dst = *dst_init;
        idct8x8_64_add(input, &mut dst, 8);
        dst
    }

    /// Differential check.
    fn assert_match(input: &[i32; 64], dst_init: &[u8; 64]) {
        let scalar = run_scalar(input, dst_init);
        let avx2 = run_avx2(input, dst_init);
        assert_eq!(
            scalar, avx2,
            "AVX2 mismatch:\ninput={input:?}\ndst_init={dst_init:?}\n\
             scalar={scalar:?}\navx2={avx2:?}"
        );
    }

    // -----------------------------------------------------------------
    // Strategies
    // -----------------------------------------------------------------

    /// Uniform i32 coefficients clamped to `[-32768, 32767]`. VP9
    /// spec / libvpx bound the post-dequant coefficient range to
    /// i16. Even though our scalar reference takes `i32`, the bit-
    /// exact contract only applies within that range; out-of-range
    /// inputs are undefined behaviour for libvpx too.
    fn uniform_coeff() -> impl Strategy<Value = i32> {
        -32768i32..=32767
    }

    fn uniform_strategy() -> impl Strategy<Value = [i32; 64]> {
        prop::collection::vec(uniform_coeff(), 64).prop_map(|v| {
            let mut out = [0i32; 64];
            out.copy_from_slice(&v);
            out
        })
    }

    /// 1..=4 non-zero coefficients at random positions.
    fn sparse_strategy() -> impl Strategy<Value = [i32; 64]> {
        prop::collection::vec((0usize..64, uniform_coeff()), 1..=4).prop_map(|nz| {
            let mut out = [0i32; 64];
            for (pos, val) in nz {
                out[pos] = val;
            }
            out
        })
    }

    /// Exactly K non-zero coefficients in raster positions
    /// `0..K`, with K drawn from the idct8x8 dispatcher-seam set
    /// {1, 2, 12, 13, 64}. K=1 is the DC-only fast-path seam; K=12 /
    /// K=13 straddle the libvpx `idct8x8_12_add` / `_64_add`
    /// dispatcher boundary; K=64 is full density.
    ///
    /// We place the K non-zero coefficients at the lowest K raster
    /// positions rather than random ones because (a) it tests the
    /// dispatcher seam without needing an enormous rejection-sample
    /// loop to draw K unique positions from a 64-element domain
    /// (proptest's `hash_set(0..64, k)` rejects too aggressively when
    /// `k` approaches the domain size — empirically aborts at K=64),
    /// and (b) the random-position case is already covered by
    /// `sparse_strategy` and `uniform_strategy`.
    fn eob_bucket_strategy() -> impl Strategy<Value = [i32; 64]> {
        prop_oneof![Just(1usize), Just(2), Just(12), Just(13), Just(64)].prop_flat_map(|k| {
            prop::collection::vec(uniform_coeff(), k).prop_map(move |values| {
                let mut out = [0i32; 64];
                for (i, val) in values.into_iter().enumerate() {
                    out[i] = val;
                }
                out
            })
        })
    }

    /// Coefficients near ±i16::MAX to stress the saturation path.
    fn high_magnitude_strategy() -> impl Strategy<Value = [i32; 64]> {
        // 30000..=32767 magnitude, random sign per coefficient.
        prop::collection::vec(
            (30000i32..=32767, any::<bool>()).prop_map(|(m, neg)| if neg { -m } else { m }),
            64,
        )
        .prop_map(|v| {
            let mut out = [0i32; 64];
            out.copy_from_slice(&v);
            out
        })
    }

    /// Pathological constant-pattern inputs.
    fn pathological_strategy() -> impl Strategy<Value = [i32; 64]> {
        prop_oneof![
            // all-equal at a random magnitude
            (-32768i32..=32767).prop_map(|v| [v; 64]),
            // all-min (i16::MIN)
            Just([i16::MIN as i32; 64]),
            // all-max (i16::MAX)
            Just([i16::MAX as i32; 64]),
            // alternating sign at i16::MAX magnitude
            Just({
                let mut out = [0i32; 64];
                for (i, slot) in out.iter_mut().enumerate() {
                    *slot = if i % 2 == 0 {
                        i16::MAX as i32
                    } else {
                        i16::MIN as i32
                    };
                }
                out
            }),
        ]
    }

    fn input_strategy() -> impl Strategy<Value = [i32; 64]> {
        prop_oneof![
            uniform_strategy(),
            sparse_strategy(),
            eob_bucket_strategy(),
            high_magnitude_strategy(),
            pathological_strategy(),
        ]
    }

    /// Random u8 destination starting state — exercises the
    /// `dst + transform` saturation paths.
    fn dst_strategy() -> impl Strategy<Value = [u8; 64]> {
        prop::collection::vec(any::<u8>(), 64).prop_map(|v| {
            let mut out = [0u8; 64];
            out.copy_from_slice(&v);
            out
        })
    }

    // -----------------------------------------------------------------
    // Proptest cases
    // -----------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 1024,
            .. ProptestConfig::default()
        })]

        /// Stratified scalar-vs-AVX2 differential.
        #[test]
        fn idct8x8_scalar_eq_avx2_proptest(
            input in input_strategy(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            let scalar = run_scalar(&input, &dst_init);
            let avx2 = run_avx2(&input, &dst_init);
            prop_assert_eq!(scalar, avx2);
        }
    }

    // -----------------------------------------------------------------
    // Directed fixtures
    // -----------------------------------------------------------------

    /// All-zero coefficients: dst unchanged after round-trip (the
    /// transform output is the all-zero matrix, +16 >> 5 = 0).
    #[test]
    fn idct8x8_all_zero_input() {
        require_avx2!();
        let input = [0i32; 64];
        let dst_init = [42u8; 64];
        assert_match(&input, &dst_init);
    }

    /// Single-coefficient impulse sweep: for every position 0..64,
    /// set just that coefficient to a magnitude that exercises both
    /// positive and negative saturation, and confirm scalar / AVX2
    /// agree.
    #[test]
    fn idct8x8_single_coefficient_impulses() {
        require_avx2!();
        for pos in 0..64 {
            for &val in &[100i32, -100, 30000, -30000, 1, -1] {
                let mut input = [0i32; 64];
                input[pos] = val;
                let dst_init = [128u8; 64];
                assert_match(&input, &dst_init);
            }
        }
    }

    /// Dst-pointer-offset correctness regression guard.
    ///
    /// The kernel uses `_mm_loadl_epi64` (8-byte, alignment-free) so
    /// this test doesn't exercise alignment-fault paths — that's
    /// deferred to G4.2 when 16/32-byte loads land. This test catches
    /// per-offset arithmetic bugs (e.g., a stride miscompute that
    /// only fires at certain byte offsets).
    ///
    /// Mechanics: allocate an 80-byte buffer on the heap (via `Box`)
    /// and sub-slice at the requested byte offset, then differentially
    /// compare scalar vs AVX2 outputs at each skew.
    #[test]
    fn idct8x8_misaligned_dst_pointers() {
        require_avx2!();
        // Use a strong synthetic input — eob ≈ 16 with a mix of
        // magnitudes; same input for every skew, only `dst` ptr varies.
        let mut input = [0i32; 64];
        for (i, slot) in input.iter_mut().take(16).enumerate() {
            *slot = (i as i32) * 137 - 800;
        }

        for skew in [0usize, 1, 3, 7, 15] {
            // Heap allocate so the platform allocator picks the base
            // alignment; we then offset by `skew` bytes.
            let mut buf: Box<[u8; 80]> = Box::new([0u8; 80]);
            // Initialise the working region to a known pattern so any
            // misaligned out-of-range write is caught by the assert.
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = (i as u8).wrapping_mul(7);
            }
            // Snapshot the full buffer for the scalar reference run.
            let mut scalar_buf = *buf;

            // Run AVX2 on the offset sub-slice.
            idct8x8_add_avx2(&input, &mut buf[skew..skew + 64], 8);
            // Run scalar on the SAME offset of an independent buffer.
            idct8x8_64_add(&input, &mut scalar_buf[skew..skew + 64], 8);

            assert_eq!(
                &buf[..],
                &scalar_buf[..],
                "AVX2/scalar mismatch at skew={skew}: \
                 per-offset arithmetic regression (kernel uses \
                 `_mm_loadl_epi64`, so this is not an alignment-fault \
                 test — see G4.2 follow-up for 16/32-byte loads)"
            );
        }
    }

    /// Property: dispatch through SimdDispatch picks AVX2 when
    /// caps.avx2 = true and scalar otherwise, and both end up with
    /// the same final dst.
    #[test]
    fn idct8x8_dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        // Forced-AVX2 dispatch (matches host caps for our CI matrix).
        let mut caps = detect();
        caps.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps);
        // Forced-scalar dispatch.
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        let mut input = [0i32; 64];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = ((i as i32) * 19) % 4001 - 2000;
        }
        let dst_init = [97u8; 64];
        let mut dst_avx2 = dst_init;
        let mut dst_scalar = dst_init;
        (dispatch_avx2.idct8x8_add())(&input, &mut dst_avx2, 8);
        (dispatch_scalar.idct8x8_add())(&input, &mut dst_scalar, 8);
        assert_eq!(dst_avx2, dst_scalar);
    }
}
