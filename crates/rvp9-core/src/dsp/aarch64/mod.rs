//! AArch64-specific SIMD kernel implementations.
//!
//! Safe wrappers are installed only when [`crate::dsp::caps::AArch64Caps`]
//! reports the required NEON tier. Each wrapper validates its slice footprint
//! before entering a `#[target_feature]` inner loop.

#![allow(unsafe_op_in_unsafe_fn)]

pub mod convolve8_neon;
pub mod convolve8_scaled_neon;
pub mod convolve_copy_neon;
pub mod idct8x8_neon;
pub mod int_pro_neon;
pub mod intra_pred_neon;
pub mod lpf16_neon;
pub mod minmax_neon;
pub mod quantize_neon;
pub mod sad4d_neon;
pub mod sad_neon;
pub mod subpel_variance_neon;
pub mod subtract_neon;
pub mod variance_neon;
pub mod vbp_neon;

pub use convolve_copy_neon::convolve_copy_neon;
pub use convolve8_neon::{convolve8_horiz_neon, convolve8_neon, convolve8_vert_neon};
pub use convolve8_scaled_neon::convolve8_scaled_neon;
pub use idct8x8_neon::{idct8x8_12_add_neon, idct8x8_add_neon, iht8x8_add_neon};
pub use int_pro_neon::{
    int_pro_64x64_fused_neon, vpx_int_pro_col_neon, vpx_int_pro_row_clamped_neon,
    vpx_int_pro_row_neon, vpx_vector_var_neon,
};
pub use intra_pred_neon::{dc_predict_full_neon, h_predictor_neon, v_predictor_neon};
pub use lpf16_neon::{
    lpf_horizontal_4_dual_neon, lpf_horizontal_4_neon, lpf_horizontal_8_dual_neon,
    lpf_horizontal_8_neon, lpf_horizontal_16_dual_neon, lpf_horizontal_16_neon,
    lpf_vertical_4_dual_neon, lpf_vertical_4_neon, lpf_vertical_8_dual_neon, lpf_vertical_8_neon,
    lpf_vertical_16_dual_neon, lpf_vertical_16_neon,
};
pub use minmax_neon::minmax_8x8_neon;
pub use quantize_neon::{
    vp9_quantize_b_32x32_neon, vp9_quantize_b_neon, vp9_quantize_fp_32x32_i16_metrics_neon,
    vp9_quantize_fp_32x32_i16_rate_metrics_neon, vp9_quantize_fp_32x32_neon,
    vp9_quantize_fp_all_zero_neon, vp9_quantize_fp_i16_metrics_neon,
    vp9_quantize_fp_i16_rate_metrics_neon, vp9_quantize_fp_neon,
};
pub use sad_neon::{
    vpx_sad4x4_neon, vpx_sad4x8_neon, vpx_sad8x4_neon, vpx_sad8x8_neon, vpx_sad8x16_neon,
    vpx_sad16x8_neon, vpx_sad16x16_neon, vpx_sad16x32_neon, vpx_sad32x16_neon, vpx_sad32x32_neon,
    vpx_sad32x64_neon, vpx_sad64x32_neon, vpx_sad64x64_neon,
};
pub use sad4d_neon::{
    vpx_sad4x4x4d_neon, vpx_sad4x8x4d_neon, vpx_sad8x4x4d_neon, vpx_sad8x8x4d_neon,
    vpx_sad8x16x4d_neon, vpx_sad16x8x4d_dotprod, vpx_sad16x8x4d_neon, vpx_sad16x16x4d_dotprod,
    vpx_sad16x16x4d_neon, vpx_sad16x32x4d_dotprod, vpx_sad16x32x4d_neon, vpx_sad32x16x4d_dotprod,
    vpx_sad32x16x4d_neon, vpx_sad32x32x4d_dotprod, vpx_sad32x32x4d_neon, vpx_sad32x64x4d_dotprod,
    vpx_sad32x64x4d_neon, vpx_sad64x32x4d_dotprod, vpx_sad64x32x4d_neon, vpx_sad64x64x4d_dotprod,
    vpx_sad64x64x4d_neon,
};
pub use subpel_variance_neon::{
    vpx_sub_pixel_variance8x8_dotprod, vpx_sub_pixel_variance8x8_neon,
    vpx_sub_pixel_variance8x16_dotprod, vpx_sub_pixel_variance8x16_neon,
    vpx_sub_pixel_variance16x8_dotprod, vpx_sub_pixel_variance16x8_neon,
    vpx_sub_pixel_variance16x16_dotprod, vpx_sub_pixel_variance16x16_neon,
    vpx_sub_pixel_variance16x32_dotprod, vpx_sub_pixel_variance16x32_neon,
    vpx_sub_pixel_variance32x16_dotprod, vpx_sub_pixel_variance32x16_neon,
    vpx_sub_pixel_variance32x32_dotprod, vpx_sub_pixel_variance32x32_neon,
    vpx_sub_pixel_variance32x64_dotprod, vpx_sub_pixel_variance32x64_neon,
    vpx_sub_pixel_variance64x32_dotprod, vpx_sub_pixel_variance64x32_neon,
    vpx_sub_pixel_variance64x64_dotprod, vpx_sub_pixel_variance64x64_neon,
};
pub use subtract_neon::{subtract_square_neon, subtract_square_neon_with_nonzero};
pub use variance_neon::{
    get_var_sse_8x8_dotprod, get_var_sse_8x8_neon, var_sum_sse_4x4_dotprod,
    var_sum_sse_4x8_dotprod, var_sum_sse_8x4_dotprod, var_sum_sse_8x4_neon,
    var_sum_sse_8x8_dotprod, var_sum_sse_8x8_grid_dotprod, var_sum_sse_8x8_grid_neon,
    var_sum_sse_8x8_neon, var_sum_sse_8x16_dotprod, var_sum_sse_8x16_neon,
    var_sum_sse_16x8_dotprod, var_sum_sse_16x8_neon, var_sum_sse_16x16_dotprod,
    var_sum_sse_16x16_neon, var_sum_sse_16x32_dotprod, var_sum_sse_16x32_neon,
    var_sum_sse_32x16_dotprod, var_sum_sse_32x16_neon, var_sum_sse_32x32_dotprod,
    var_sum_sse_32x32_neon, var_sum_sse_32x64_dotprod, var_sum_sse_32x64_neon,
    var_sum_sse_64x32_dotprod, var_sum_sse_64x32_neon, var_sum_sse_64x64_dotprod,
    var_sum_sse_64x64_neon,
};
pub use vbp_neon::avg_diff_8x8_grid_64x64_neon;
