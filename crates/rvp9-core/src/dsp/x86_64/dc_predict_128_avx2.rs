//! DC_128 fallback intra-predictor — x86_64 AVX2 implementation
//! (Wave I I4, 128 variant).
//!
//! When neither neighbour is available (picture boundary at the
//! origin), DC_PRED collapses to a constant fill with the
//! maximum-entropy mid-grey value `128`. This is the simplest of the
//! four DC variants: zero arithmetic, pure store. The scalar
//! reference is [`crate::intra_pred::dc_128_predictor`].
//!
//! ## Size dispatch
//!
//! Single safe wrapper accepts `bs ∈ {4, 8, 16, 32}` as a runtime arg
//! and dispatches to one of four hand-unrolled wide-store loops:
//!
//! | `bs` | broadcast            | store width                       | rows |
//! |------|----------------------|-----------------------------------|------|
//! | 4    | u32 splat (scalar)   | `write_unaligned::<u32>`          | 4    |
//! | 8    | `_mm_set1_epi8`      | `_mm_storel_epi64` (64-bit)       | 8    |
//! | 16   | `_mm_set1_epi8`      | `_mm_storeu_si128` (128-bit)      | 16   |
//! | 32   | `_mm256_set1_epi8`   | `_mm256_storeu_si256` (256-bit)   | 32   |
//!
//! Mirrors the H_PRED kernel's per-row store pattern; the only
//! structural difference is that H_PRED broadcasts a per-row byte
//! from `left[r]` while this kernel broadcasts the constant `128`
//! across every row.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! No SIMD loads (no input edges). All stores into `dst` use
//! unaligned intrinsics.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    _mm_set1_epi8, _mm_storel_epi64, _mm_storeu_si128, _mm256_set1_epi8, _mm256_storeu_si256,
};

/// AVX2 implementation of DC_128 — fill a `bs × bs` block with 128.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`dc_predict_128_avx2`]) when `caps.avx2 == true`.
///
/// `dst` must hold at least `(bs - 1) * stride + bs` bytes of valid
/// storage. `bs` must be one of `{4, 8, 16, 32}`.
// G4: stores to `dst` use _mm256_storeu_si256 / _mm_storeu_si128 / _mm_storel_epi64 / write_unaligned (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dc_predict_128_avx2_inner(dst: &mut [u8], stride: usize, bs: usize) {
    debug_assert!(dst.len() >= (bs - 1) * stride + bs);
    debug_assert!(matches!(bs, 4 | 8 | 16 | 32));

    match bs {
        4 => {
            let row: u32 = 0x8080_8080; // 128 broadcast across 4 bytes
            for r in 0..4 {
                core::ptr::write_unaligned(dst.as_mut_ptr().add(r * stride) as *mut u32, row);
            }
        }
        8 => {
            let row_v = _mm_set1_epi8(128u8 as i8);
            for r in 0..8 {
                _mm_storel_epi64(dst.as_mut_ptr().add(r * stride) as *mut _, row_v);
            }
        }
        16 => {
            let row_v = _mm_set1_epi8(128u8 as i8);
            for r in 0..16 {
                _mm_storeu_si128(dst.as_mut_ptr().add(r * stride) as *mut _, row_v);
            }
        }
        32 => {
            let row_v = _mm256_set1_epi8(128u8 as i8);
            for r in 0..32 {
                _mm256_storeu_si256(dst.as_mut_ptr().add(r * stride) as *mut _, row_v);
            }
        }
        _ => unreachable!("dc_predict_128_avx2: bs must be in {{4, 8, 16, 32}}"),
    }
}

/// Safe wrapper for the AVX2 DC_128 fallback predictor.
///
/// The dispatch slot [`crate::dsp::SimdDispatch::dc_predict_128`]
/// points at this wrapper on AVX2 hosts and at the scalar reference
/// (`crate::intra_pred::dc_128_predictor`) elsewhere.
///
/// `bs ∈ {4, 8, 16, 32}` for the AVX2 specialisations; `bs = 64`
/// (Block64x64) falls back to scalar — see the matching note in
/// [`crate::dsp::x86_64::dc_predict_left_avx2`].
#[cfg(target_arch = "x86_64")]
pub fn dc_predict_128_avx2(dst: &mut [u8], stride: usize, bs: usize) {
    if bs == 64 {
        crate::intra_pred::dc_128_predictor(dst, stride, bs);
        return;
    }
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime. Test paths in this module's tests
    // gate each call on a runtime `is_x86_feature_detected!("avx2")`
    // check.
    unsafe { dc_predict_128_avx2_inner(dst, stride, bs) }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    use super::dc_predict_128_avx2;
    use crate::intra_pred::dc_128_predictor;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    fn run_scalar(bs: usize) -> Vec<u8> {
        let mut dst = vec![0u8; bs * bs];
        dc_128_predictor(&mut dst, bs, bs);
        dst
    }

    fn run_avx2(bs: usize) -> Vec<u8> {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host"
        );
        let mut dst = vec![0u8; bs * bs];
        dc_predict_128_avx2(&mut dst, bs, bs);
        dst
    }

    /// Differential vs scalar across all sizes. DC_128 has no inputs,
    /// so a single iter per size is sufficient — but we sweep 1024
    /// iters anyway to keep the test shape uniform with the other DC
    /// kernels. The work per iter is a single block fill.
    #[test]
    fn differential_1024_iters() {
        require_avx2!();
        for bs in [4usize, 8, 16, 32] {
            for _iter in 0..1024 {
                let scalar = run_scalar(bs);
                let avx2 = run_avx2(bs);
                assert_eq!(scalar, avx2, "AVX2/scalar mismatch at bs={bs}");
            }
        }
    }

    #[test]
    fn tautology_guard_avx2_path() {
        require_avx2!();
        assert!(
            is_x86_feature_detected!("avx2"),
            "tautology guard: AVX2 detection must report true on this host"
        );
        // Sanity: the kernel actually fills with 128 (not 0).
        let avx2 = run_avx2(32);
        for b in avx2 {
            assert_eq!(b, 128);
        }
    }

    #[test]
    fn dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps_avx2 = detect();
        caps_avx2.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps_avx2);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        for bs in [4usize, 8, 16, 32] {
            let mut dst_avx2 = vec![0u8; bs * bs];
            let mut dst_scalar = vec![0u8; bs * bs];
            (dispatch_avx2.dc_predict_128())(&mut dst_avx2, bs, bs);
            (dispatch_scalar.dc_predict_128())(&mut dst_scalar, bs, bs);
            assert_eq!(
                dst_avx2, dst_scalar,
                "dispatch round-trip mismatch at bs={bs}"
            );
        }
    }
}
