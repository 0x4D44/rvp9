//! x86_64-specific SIMD kernel implementations.
//!
//! Each sub-module owns one (kernel-family, ISA) pair. The bodies use
//! `#[target_feature(enable = "<isa>")]` on `unsafe fn` inner kernels
//! and expose a safe wrapper that the [`crate::dsp::SimdDispatch`]
//! slot points at. The safe wrapper's single `unsafe { ... }` block
//! carries a `// SAFETY:` comment naming the OnceLock + caps invariant
//! per the G11 `clippy::undocumented_unsafe_blocks = deny` lint.
//!
//! G4.1 pilot scope: `idct8x8_avx2` only. G4.2+ adds the rest of the
//! inverse-transform family, then intra-pred / loopfilter / etc. The
//! convention here is what every subsequent G4.x rollout mirrors.
//!
//! ## Source-side aligned-load discipline
//!
//! Every `unsafe fn` in this directory MUST use unaligned-load
//! intrinsics for reads from `dst` (`_mm256_loadu_si256` /
//! `_mm_loadu_si128` / `_mm_loadl_epi64`), and carry a sentinel
//! comment of the form:
//!
//! ```ignore
//! // G4: loads from `dst` use _mm256_loadu_si256 (unaligned)
//! ```
//!
//! `scripts/check-simd-sentinels.sh` (and `.ps1`) greps for the
//! sentinel; CI gates on it once the workflow lands.

#![allow(unsafe_op_in_unsafe_fn)]
// Each `unsafe fn` body opts back in to `unsafe { }`-wrapping for
// intrinsic calls explicitly. The workspace lint
// `unsafe_op_in_unsafe_fn = "warn"` would otherwise force every
// SAFETY-commented intrinsic call to be its own block; the allow
// permits the cleaner outer-fn-is-unsafe convention used by libvpx
// ports while still requiring the safe-wrapper to document
// soundness.

pub mod convolve8_avg_avx2;
pub mod convolve8_avx2;
pub mod convolve8_horiz_avx2;
pub mod convolve8_scaled_avx2;
pub mod convolve8_vert_avx2;
pub mod convolve_avg_avx2;
pub mod convolve_copy_avx2;
pub mod d117_predictor_avx2;
pub mod d135_predictor_avx2;
pub mod d153_predictor_avx2;
pub mod d207_predictor_avx2;
pub mod d45_predictor_avx2;
pub mod d63_predictor_avx2;
pub mod dc_pred_4x4_avx2;
pub mod dc_predict_128_avx2;
pub mod dc_predict_full_avx2;
pub mod dc_predict_left_avx2;
pub mod dc_predict_top_avx2;
pub mod h_predictor_avx2;
#[cfg(any(test, doc))]
pub mod idct16x16_10_avx2;
#[cfg(any(test, doc))]
pub mod idct16x16_38_avx2;
#[cfg(any(test, doc))]
pub mod idct16x16_avx2;
pub mod idct16x16_sse2_256;
#[cfg(any(test, doc))]
pub mod idct32x32_135_avx2;
#[cfg(any(test, doc))]
pub mod idct32x32_34_avx2;
#[cfg(any(test, doc))]
pub mod idct32x32_avx2;
pub mod idct32x32_avx2_1024;
pub mod idct32x32_avx2_135;
pub mod idct32x32_sse2_1024;
pub mod idct4x4_avx2;
pub mod idct8x8_12_sse2;
#[cfg(any(test, doc))]
pub mod idct8x8_avx2;
pub mod idct8x8_sse2_64;
pub mod iht16x16_avx2;
pub mod iht4x4_avx2;
pub mod iht8x8_avx2;
pub mod int_pro_sse2;
pub mod intra_directional_common_avx2;
pub mod lpf_horizontal_16_avx2;
pub mod lpf_horizontal_16_dual_avx2;
pub mod lpf_horizontal_4_avx2;
pub mod lpf_horizontal_4_dual_avx2;
pub mod lpf_horizontal_8_avx2;
pub mod lpf_horizontal_8_dual_avx2;
pub mod lpf_vertical_16_avx2;
pub mod lpf_vertical_16_dual_avx2;
pub mod lpf_vertical_4_avx2;
pub mod lpf_vertical_4_dual_avx2;
pub mod lpf_vertical_8_avx2;
pub mod lpf_vertical_8_dual_avx2;
pub mod minmax_sse2;
pub mod plane_dequant_finisher_avx2;
pub mod quantize_fp_avx2;
pub mod quantize_metrics_sse2;
pub mod sad4d_avx2;
pub mod sad_avx2;
pub mod sse_block_sse2;
pub mod subpel_variance_avx2;
pub mod subtract_sse2;
pub mod tm_predictor_avx2;
pub mod v_predictor_avx2;
pub mod variance_avx2;
pub mod vbp_sse2;

pub use convolve_avg_avx2::convolve_avg_avx2;
pub use convolve_copy_avx2::convolve_copy_avx2;
pub use convolve8_avg_avx2::{
    convolve8_avg_avx2, convolve8_avg_horiz_avx2, convolve8_avg_vert_avx2,
};
pub use convolve8_avx2::convolve8_avx2;
pub use convolve8_horiz_avx2::convolve8_horiz_avx2;
pub use convolve8_scaled_avx2::{convolve8_scaled_avg_avx2, convolve8_scaled_avx2};
pub use convolve8_vert_avx2::convolve8_vert_avx2;
pub use d45_predictor_avx2::d45_predictor_avx2;
pub use d63_predictor_avx2::d63_predictor_avx2;
pub use d117_predictor_avx2::d117_predictor_avx2;
pub use d135_predictor_avx2::d135_predictor_avx2;
pub use d153_predictor_avx2::d153_predictor_avx2;
pub use d207_predictor_avx2::d207_predictor_avx2;
pub use dc_pred_4x4_avx2::dc_predictor_4x4_avx2;
pub use dc_predict_128_avx2::dc_predict_128_avx2;
pub use dc_predict_full_avx2::dc_predict_full_avx2;
pub use dc_predict_left_avx2::dc_predict_left_avx2;
pub use dc_predict_top_avx2::dc_predict_top_avx2;
pub use h_predictor_avx2::h_predictor_avx2;
pub use idct4x4_avx2::idct4x4_add_avx2;
pub use idct8x8_12_sse2::idct8x8_add_sse2_12;
pub use idct8x8_sse2_64::idct8x8_add_sse2_64;
#[cfg(any(test, doc))]
pub use idct16x16_avx2::idct16x16_add_avx2;
pub use idct16x16_sse2_256::{
    idct16x16_add_sse2_10, idct16x16_add_sse2_38, idct16x16_add_sse2_256,
};
pub use idct32x32_avx2_135::idct32x32_add_avx2_135;
pub use idct32x32_avx2_1024::idct32x32_add_avx2_1024;
pub use idct32x32_sse2_1024::{
    idct32x32_add_sse2_34, idct32x32_add_sse2_135, idct32x32_add_sse2_1024,
};
pub use iht4x4_avx2::iht4x4_add_avx2;
pub use iht8x8_avx2::iht8x8_add_avx2;
pub use iht16x16_avx2::iht16x16_add_avx2;
pub use lpf_horizontal_4_avx2::lpf_horizontal_4_avx2;
pub use lpf_horizontal_4_dual_avx2::lpf_horizontal_4_dual_avx2;
pub use lpf_horizontal_8_avx2::lpf_horizontal_8_avx2;
pub use lpf_horizontal_8_dual_avx2::lpf_horizontal_8_dual_avx2;
pub use lpf_horizontal_16_avx2::lpf_horizontal_16_avx2;
pub use lpf_horizontal_16_dual_avx2::lpf_horizontal_16_dual_avx2;
pub use lpf_vertical_4_avx2::lpf_vertical_4_avx2;
pub use lpf_vertical_4_dual_avx2::lpf_vertical_4_dual_avx2;
pub use lpf_vertical_8_avx2::lpf_vertical_8_avx2;
pub use lpf_vertical_8_dual_avx2::lpf_vertical_8_dual_avx2;
pub use lpf_vertical_16_avx2::lpf_vertical_16_avx2;
pub use lpf_vertical_16_dual_avx2::lpf_vertical_16_dual_avx2;
pub use minmax_sse2::minmax_8x8_sse2;
pub use plane_dequant_finisher_avx2::plane_dequant_finisher_avx2;
pub use quantize_fp_avx2::vp9_quantize_fp_avx2;
pub use sad_avx2::{
    vpx_sad4x4_avx2, vpx_sad4x8_avx2, vpx_sad8x4_avx2, vpx_sad8x8_avx2, vpx_sad8x16_avx2,
    vpx_sad16x8_avx2, vpx_sad16x16_avx2, vpx_sad16x32_avx2, vpx_sad32x16_avx2, vpx_sad32x32_avx2,
    vpx_sad32x64_avx2, vpx_sad64x32_avx2, vpx_sad64x64_avx2,
};
pub use sad4d_avx2::{
    vpx_sad4x4x4d_avx2, vpx_sad4x8x4d_avx2, vpx_sad8x4x4d_avx2, vpx_sad8x8x4d_avx2,
    vpx_sad8x16x4d_avx2, vpx_sad16x8x4d_avx2, vpx_sad16x16x4d_avx2, vpx_sad16x32x4d_avx2,
    vpx_sad32x16x4d_avx2, vpx_sad32x32x4d_avx2, vpx_sad32x64x4d_avx2, vpx_sad64x32x4d_avx2,
    vpx_sad64x64x4d_avx2,
};
pub use subpel_variance_avx2::{
    vpx_sub_pixel_variance4x4_avx2, vpx_sub_pixel_variance8x8_avx2,
    vpx_sub_pixel_variance8x16_avx2, vpx_sub_pixel_variance16x8_avx2,
    vpx_sub_pixel_variance16x16_avx2, vpx_sub_pixel_variance16x32_avx2,
    vpx_sub_pixel_variance32x16_avx2, vpx_sub_pixel_variance32x32_avx2,
    vpx_sub_pixel_variance32x64_avx2, vpx_sub_pixel_variance64x32_avx2,
    vpx_sub_pixel_variance64x64_avx2,
};
pub use subtract_sse2::{subtract_square_sse2, subtract_square_sse2_with_nonzero};
pub use tm_predictor_avx2::tm_predictor_avx2;
pub use v_predictor_avx2::v_predictor_avx2;
pub use variance_avx2::{
    get_var_sse_8x8_avx2, var_sum_sse_4x4_avx2, var_sum_sse_4x8_avx2, var_sum_sse_8x4_avx2,
    var_sum_sse_8x8_avx2, var_sum_sse_8x16_avx2, var_sum_sse_16x8_avx2, var_sum_sse_16x16_avx2,
    var_sum_sse_16x32_avx2, var_sum_sse_32x16_avx2, var_sum_sse_32x32_avx2, var_sum_sse_32x64_avx2,
    var_sum_sse_64x32_avx2, var_sum_sse_64x64_avx2,
};
pub use vbp_sse2::avg_diff_8x8_grid_64x64_sse2;
