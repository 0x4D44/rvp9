//! V_PRED intra-predictor — x86_64 AVX2 implementation (Wave I I1).
//!
//! Mirrors the safe-wrapper convention established by
//! [`crate::dsp::x86_64::dc_pred_4x4_avx2`]. The scalar reference is
//! [`crate::intra_pred::v_predictor`].
//!
//! ## Algorithm
//!
//! V_PRED broadcasts the `above` neighbour row down across all `bs`
//! rows of the destination block — equivalent to writing
//! `above[..bs]` into every row of the `bs × bs` block. There is no
//! arithmetic; the kernel is a pure load+store and the speedup comes
//! purely from issuing wide stores instead of `copy_from_slice`
//! per-row.
//!
//! ## Size dispatch
//!
//! VP9 intra blocks come in four sizes `bs ∈ {4, 8, 16, 32}`. This
//! kernel is **size-parametric**: a single safe wrapper accepts `bs`
//! as a runtime argument and dispatches internally to the right wide
//! store width:
//!
//! | `bs` | load width | store width        | rows |
//! |------|------------|--------------------|------|
//! | 4    | 4-byte     | `write_unaligned::<u32>`            | 4  |
//! | 8    | 8-byte     | `_mm_storel_epi64` (64-bit)         | 8  |
//! | 16   | 16-byte    | `_mm_storeu_si128`  (128-bit)       | 16 |
//! | 32   | 32-byte    | `_mm256_storeu_si256` (256-bit)     | 32 |
//!
//! Each size loads its `above` row exactly once, then issues `bs`
//! wide-stride stores down through `dst`. This pattern is what the
//! I2/I3/I4 size-parametric slots (H_PRED / TM_PRED / DC family
//! extension) will mirror.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! All loads from `above` and stores to `dst` use unaligned
//! intrinsics — no alignment assumption on the caller's slices. The
//! grep sentinels below are checked by `scripts/check-simd-sentinels.sh`
//! / `.ps1`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    _mm256_loadu_si256, _mm256_storeu_si256, _mm_loadl_epi64, _mm_loadu_si128, _mm_storel_epi64,
    _mm_storeu_si128,
};

/// AVX2 implementation of the size-parametric V_PRED intra-predictor.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`v_predictor_avx2`]) when `caps.avx2 == true`.
///
/// `above` must have at least `bs` valid u8s. `dst` must have at
/// least `(bs - 1) * stride + bs` bytes of valid storage.
///
/// # Implementation
///
/// Dispatches on `bs` to one of four hand-unrolled width-specialised
/// blocks; any unexpected `bs` value (defensive — shouldn't happen)
/// falls back to a scalar `copy_from_slice` loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn v_predictor_avx2_inner(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    debug_assert!(above.len() >= bs);
    debug_assert!(
        dst.len() >= (bs - 1) * stride + bs,
        "dst too small for bs×bs block at stride={stride}"
    );

    match bs {
        4 => {
            // G4: loads from `above` use read_unaligned::<u32> (unaligned)
            let row: u32 = core::ptr::read_unaligned(above.as_ptr() as *const u32);
            for r in 0..4 {
                // G4: stores to `dst` use write_unaligned::<u32> (unaligned)
                core::ptr::write_unaligned(dst.as_mut_ptr().add(r * stride) as *mut u32, row);
            }
        }
        8 => {
            // G4: loads from `above` use _mm_loadl_epi64 (unaligned)
            let row = _mm_loadl_epi64(above.as_ptr() as *const _);
            for r in 0..8 {
                // G4: stores to `dst` use _mm_storel_epi64 (unaligned)
                _mm_storel_epi64(dst.as_mut_ptr().add(r * stride) as *mut _, row);
            }
        }
        16 => {
            // G4: loads from `above` use _mm_loadu_si128 (unaligned)
            let row = _mm_loadu_si128(above.as_ptr() as *const _);
            for r in 0..16 {
                // G4: stores to `dst` use _mm_storeu_si128 (unaligned)
                _mm_storeu_si128(dst.as_mut_ptr().add(r * stride) as *mut _, row);
            }
        }
        32 => {
            // G4: loads from `above` use _mm256_loadu_si256 (unaligned)
            let row = _mm256_loadu_si256(above.as_ptr() as *const _);
            for r in 0..32 {
                // G4: stores to `dst` use _mm256_storeu_si256 (unaligned)
                _mm256_storeu_si256(dst.as_mut_ptr().add(r * stride) as *mut _, row);
            }
        }
        _ => {
            // Defensive scalar fallback. VP9 only uses bs ∈ {4, 8, 16, 32};
            // we should never reach this arm, but `unreachable_unchecked`
            // would be UB if a future caller violates the contract.
            for r in 0..bs {
                let row_start = r * stride;
                dst[row_start..row_start + bs].copy_from_slice(&above[..bs]);
            }
        }
    }
}

/// Safe wrapper for the AVX2 size-parametric V_PRED predictor.
///
/// Signature mirrors a flattened slice of
/// [`crate::intra_pred::v_predictor`] (with `PredCtx` reduced to just
/// `above` — V_PRED reads nothing else). The dispatch slot
/// [`crate::dsp::SimdDispatch::v_predictor`] points at this wrapper on
/// AVX2 hosts and at the scalar adapter on every other platform.
#[cfg(target_arch = "x86_64")]
pub fn v_predictor_avx2(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime. Test paths in this module gate every
    // call on a runtime `is_x86_feature_detected!("avx2")` check via
    // the `require_avx2!()` macro.
    unsafe { v_predictor_avx2_inner(dst, stride, bs, above) }
}

// Miri cannot interpret AVX2 intrinsics (`_mm256_*` etc.), so this
// `tests` module is excluded from `cargo +nightly miri test`. The
// AVX2 kernel itself is x86_64-only, and the scalar reference is
// already covered by `crate::intra_pred::tests` which DOES run
// under miri.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `v_predictor_avx2`.
    //!
    //! V_PRED is a pure broadcast (no arithmetic), so the bug surface
    //! is alignment / stride correctness rather than rounding. Tests
    //! sweep every `bs ∈ {4, 8, 16, 32}` with deterministic LCG-seeded
    //! random `above` rows, plus a dispatch round-trip guard.
    use super::v_predictor_avx2;
    use crate::intra_pred::{v_predictor, PredCtx};

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Deterministic LCG (numerical recipes constants) — used for
    /// reproducible random-input generation in the differential test.
    fn lcg_next(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *state
    }

    fn run_scalar(above: &[u8], bs: usize) -> Vec<u8> {
        let mut dst = vec![0u8; bs * bs];
        let ctx = PredCtx {
            above,
            left: &[], // unused by V_PRED
            above_left: 0,
        };
        v_predictor(&mut dst, bs, bs, ctx);
        dst
    }

    fn run_avx2(above: &[u8], bs: usize) -> Vec<u8> {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host"
        );
        let mut dst = vec![0u8; bs * bs];
        v_predictor_avx2(&mut dst, bs, bs, above);
        dst
    }

    /// 1024-iter differential fuzz vs scalar across all 4 sizes.
    /// LCG-seeded so failures reproduce.
    #[test]
    fn differential_1024_iters() {
        require_avx2!();
        let mut state: u32 = 0xC0FF_EE42;
        for bs in [4usize, 8, 16, 32] {
            for _iter in 0..1024 {
                let mut above = vec![0u8; bs];
                for slot in above.iter_mut() {
                    *slot = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let scalar = run_scalar(&above, bs);
                let avx2 = run_avx2(&above, bs);
                assert_eq!(scalar, avx2, "AVX2/scalar mismatch at bs={bs}");
            }
        }
    }

    /// Tautology guard — assert the AVX2 path is actually queried by
    /// `is_x86_feature_detected!("avx2")` on the test host (mirrors
    /// the lpf module guards).
    #[test]
    fn tautology_guard_avx2_path() {
        require_avx2!();
        // If we reached here, `is_x86_feature_detected!("avx2")`
        // returned true on the running host — the AVX2 path under
        // `require_avx2!()` is genuinely exercised, not a no-op.
        let above = [42u8; 32];
        let avx2 = run_avx2(&above, 32);
        assert_eq!(avx2.len(), 32 * 32);
        // Every byte of every row must equal 42.
        for b in avx2 {
            assert_eq!(b, 42);
        }
    }

    /// Round-trip through SimdDispatch: forced-AVX2 and forced-scalar
    /// must produce identical output on a deterministic input.
    #[test]
    fn dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::caps::detect;
        use crate::dsp::SimdDispatch;

        let mut caps_avx2 = detect();
        caps_avx2.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps_avx2);

        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        for bs in [4usize, 8, 16, 32] {
            // Deterministic input — `above[i] = i*7 + bs as u8`.
            let above: Vec<u8> = (0..bs)
                .map(|i| (i as u32 * 7 + bs as u32) as u8)
                .collect();
            let mut dst_avx2 = vec![0u8; bs * bs];
            let mut dst_scalar = vec![0u8; bs * bs];
            (dispatch_avx2.v_predictor())(&mut dst_avx2, bs, bs, &above);
            (dispatch_scalar.v_predictor())(&mut dst_scalar, bs, bs, &above);
            assert_eq!(dst_avx2, dst_scalar, "dispatch round-trip mismatch at bs={bs}");
        }
    }

    /// Misaligned `dst` regression guard — V_PRED uses unaligned
    /// stores at every width, so this checks per-skew correctness
    /// rather than alignment-fault paths.
    #[test]
    fn v_predictor_misaligned_dst() {
        require_avx2!();
        for bs in [4usize, 8, 16, 32] {
            let above: Vec<u8> = (0..bs).map(|i| (i as u32 * 13) as u8).collect();
            for skew in [0usize, 1, 3, 7] {
                let buf_len = skew + (bs - 1) * bs + bs;
                let mut avx2_buf = vec![0u8; buf_len];
                let mut scalar_buf = vec![0u8; buf_len];
                v_predictor_avx2(&mut avx2_buf[skew..], bs, bs, &above);
                let ctx = PredCtx {
                    above: &above,
                    left: &[],
                    above_left: 0,
                };
                v_predictor(&mut scalar_buf[skew..], bs, bs, ctx);
                assert_eq!(avx2_buf, scalar_buf, "skew={skew} bs={bs}");
            }
        }
    }
}
