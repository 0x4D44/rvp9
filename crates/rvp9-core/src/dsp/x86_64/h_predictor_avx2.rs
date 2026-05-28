//! H_PRED intra-predictor — x86_64 AVX2 implementation (Wave I I2).
//!
//! Mirrors the safe-wrapper convention established by
//! [`crate::dsp::x86_64::v_predictor_avx2`] (Wave I I1). The scalar
//! reference is [`crate::intra_pred::h_predictor`].
//!
//! ## Algorithm
//!
//! H_PRED broadcasts `left[r]` across row `r` of the `bs × bs`
//! destination block — for each row `r ∈ [0, bs)`, every column gets
//! the single byte `left[r]`. There is no arithmetic; the kernel is a
//! pure broadcast+store and the speedup comes purely from issuing wide
//! `set1`+store stores instead of `slice::fill` per-row.
//!
//! ## Size dispatch
//!
//! VP9 intra blocks come in four sizes `bs ∈ {4, 8, 16, 32}`. This
//! kernel is **size-parametric**: a single safe wrapper accepts `bs`
//! as a runtime argument and dispatches internally to the right wide
//! store width:
//!
//! | `bs` | broadcast            | store width                | rows |
//! |------|----------------------|----------------------------|------|
//! | 4    | u32 splat (scalar)   | `write_unaligned::<u32>`   | 4    |
//! | 8    | `_mm_set1_epi8`      | `_mm_storel_epi64` (64-bit)| 8    |
//! | 16   | `_mm_set1_epi8`      | `_mm_storeu_si128` (128b)  | 16   |
//! | 32   | `_mm256_set1_epi8`   | `_mm256_storeu_si256` (256)| 32   |
//!
//! Each row broadcasts a single `left[r]` byte into a wide register,
//! then issues one wide store at `dst[r * stride..]`. This mirrors
//! the I1 V_PRED store pattern; the only structural difference is
//! that V_PRED loads its row from `above` once and stores it `bs`
//! times, while H_PRED constructs each row's broadcast freshly from
//! one byte of `left`.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! H_PRED performs no SIMD loads from input — `left[r]` is read as a
//! single u8 — so all the sentinel comments below are STORE side
//! only. Stores to `dst` use unaligned intrinsics with no alignment
//! assumption on the caller's slice. The grep sentinels below are
//! checked by `scripts/check-simd-sentinels.sh` / `.ps1`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    _mm_set1_epi8, _mm_storel_epi64, _mm_storeu_si128, _mm256_set1_epi8, _mm256_storeu_si256,
};

/// AVX2 implementation of the size-parametric H_PRED intra-predictor.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`h_predictor_avx2`]) when `caps.avx2 == true`.
///
/// `left` must have at least `bs` valid u8s. `dst` must have at least
/// `(bs - 1) * stride + bs` bytes of valid storage.
///
/// # Implementation
///
/// Dispatches on `bs` to one of four hand-unrolled width-specialised
/// blocks; any unexpected `bs` value (defensive — shouldn't happen)
/// falls back to a scalar `slice::fill` loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn h_predictor_avx2_inner(dst: &mut [u8], stride: usize, bs: usize, left: &[u8]) {
    debug_assert!(left.len() >= bs);
    debug_assert!(
        dst.len() >= (bs - 1) * stride + bs,
        "dst too small for bs×bs block at stride={stride}"
    );

    match bs {
        4 => {
            for r in 0..4 {
                // Broadcast `left[r]` to all 4 bytes of a u32 splat —
                // no SIMD intrinsic needed at this width; multiplying
                // the byte by 0x01010101 packs it across all 4 lanes.
                let v = left[r] as u32;
                let row: u32 = v | (v << 8) | (v << 16) | (v << 24);
                // G4: stores to `dst` use write_unaligned::<u32> (unaligned)
                core::ptr::write_unaligned(dst.as_mut_ptr().add(r * stride) as *mut u32, row);
            }
        }
        8 => {
            for r in 0..8 {
                let row = _mm_set1_epi8(left[r] as i8);
                // G4: stores to `dst` use _mm_storel_epi64 (unaligned)
                _mm_storel_epi64(dst.as_mut_ptr().add(r * stride) as *mut _, row);
            }
        }
        16 => {
            for r in 0..16 {
                let row = _mm_set1_epi8(left[r] as i8);
                // G4: stores to `dst` use _mm_storeu_si128 (unaligned)
                _mm_storeu_si128(dst.as_mut_ptr().add(r * stride) as *mut _, row);
            }
        }
        32 => {
            for r in 0..32 {
                let row = _mm256_set1_epi8(left[r] as i8);
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
                dst[row_start..row_start + bs].fill(left[r]);
            }
        }
    }
}

/// Safe wrapper for the AVX2 size-parametric H_PRED predictor.
///
/// Signature mirrors a flattened slice of
/// [`crate::intra_pred::h_predictor`] (with `PredCtx` reduced to just
/// `left` — H_PRED reads nothing else). The dispatch slot
/// [`crate::dsp::SimdDispatch::h_predictor`] points at this wrapper on
/// AVX2 hosts and at the scalar adapter on every other platform.
#[cfg(target_arch = "x86_64")]
pub fn h_predictor_avx2(dst: &mut [u8], stride: usize, bs: usize, left: &[u8]) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime. Test paths in this module gate every
    // call on a runtime `is_x86_feature_detected!("avx2")` check via
    // the `require_avx2!()` macro.
    unsafe { h_predictor_avx2_inner(dst, stride, bs, left) }
}

// Miri cannot interpret AVX2 intrinsics (`_mm256_*` etc.), so this
// `tests` module is excluded from `cargo +nightly miri test`. The
// AVX2 kernel itself is x86_64-only, and the scalar reference is
// already covered by `crate::intra_pred::tests` which DOES run
// under miri.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `h_predictor_avx2`.
    //!
    //! H_PRED is a pure broadcast (no arithmetic), so the bug surface
    //! is alignment / stride correctness rather than rounding. Tests
    //! sweep every `bs ∈ {4, 8, 16, 32}` with deterministic LCG-seeded
    //! random `left` rows, plus a dispatch round-trip guard.
    use super::h_predictor_avx2;
    use crate::intra_pred::{PredCtx, h_predictor};

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

    fn run_scalar(left: &[u8], bs: usize) -> Vec<u8> {
        let mut dst = vec![0u8; bs * bs];
        let ctx = PredCtx {
            above: &[], // unused by H_PRED
            left,
            above_left: 0,
        };
        h_predictor(&mut dst, bs, bs, ctx);
        dst
    }

    fn run_avx2(left: &[u8], bs: usize) -> Vec<u8> {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host"
        );
        let mut dst = vec![0u8; bs * bs];
        h_predictor_avx2(&mut dst, bs, bs, left);
        dst
    }

    /// 1024-iter differential fuzz vs scalar across all 4 sizes.
    /// LCG-seeded so failures reproduce.
    #[test]
    fn differential_1024_iters() {
        require_avx2!();
        let mut state: u32 = 0xC0FF_EE43;
        for bs in [4usize, 8, 16, 32] {
            for _iter in 0..1024 {
                let mut left = vec![0u8; bs];
                for slot in left.iter_mut() {
                    *slot = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let scalar = run_scalar(&left, bs);
                let avx2 = run_avx2(&left, bs);
                assert_eq!(scalar, avx2, "AVX2/scalar mismatch at bs={bs}");
            }
        }
    }

    /// Tautology guard — assert the AVX2 path is actually queried by
    /// `is_x86_feature_detected!("avx2")` on the test host (mirrors
    /// the v_predictor module guard).
    #[test]
    fn tautology_guard_avx2_path() {
        require_avx2!();
        // If we reached here, `is_x86_feature_detected!("avx2")`
        // returned true on the running host — the AVX2 path under
        // `require_avx2!()` is genuinely exercised, not a no-op.
        // Use a non-constant left[] so each row carries a distinct
        // broadcast byte; checks both broadcast correctness and the
        // per-row store offset.
        let left: Vec<u8> = (0..32)
            .map(|i| (i as u8).wrapping_mul(7).wrapping_add(11))
            .collect();
        let avx2 = run_avx2(&left, 32);
        assert_eq!(avx2.len(), 32 * 32);
        // Every byte of row r must equal left[r].
        for r in 0..32 {
            for c in 0..32 {
                assert_eq!(avx2[r * 32 + c], left[r], "mismatch at row {r} col {c}");
            }
        }
    }

    /// Round-trip through SimdDispatch: forced-AVX2 and forced-scalar
    /// must produce identical output on a deterministic input.
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
            // Deterministic input — `left[i] = i*11 + bs as u8`.
            let left: Vec<u8> = (0..bs).map(|i| (i as u32 * 11 + bs as u32) as u8).collect();
            let mut dst_avx2 = vec![0u8; bs * bs];
            let mut dst_scalar = vec![0u8; bs * bs];
            (dispatch_avx2.h_predictor())(&mut dst_avx2, bs, bs, &left);
            (dispatch_scalar.h_predictor())(&mut dst_scalar, bs, bs, &left);
            assert_eq!(
                dst_avx2, dst_scalar,
                "dispatch round-trip mismatch at bs={bs}"
            );
        }
    }

    /// Misaligned `dst` regression guard — H_PRED uses unaligned
    /// stores at every width, so this checks per-skew correctness
    /// rather than alignment-fault paths.
    #[test]
    fn h_predictor_misaligned_dst() {
        require_avx2!();
        for bs in [4usize, 8, 16, 32] {
            let left: Vec<u8> = (0..bs).map(|i| (i as u32 * 13) as u8).collect();
            for skew in [0usize, 1, 3, 7] {
                let buf_len = skew + (bs - 1) * bs + bs;
                let mut avx2_buf = vec![0u8; buf_len];
                let mut scalar_buf = vec![0u8; buf_len];
                h_predictor_avx2(&mut avx2_buf[skew..], bs, bs, &left);
                let ctx = PredCtx {
                    above: &[],
                    left: &left,
                    above_left: 0,
                };
                h_predictor(&mut scalar_buf[skew..], bs, bs, ctx);
                assert_eq!(avx2_buf, scalar_buf, "skew={skew} bs={bs}");
            }
        }
    }
}
