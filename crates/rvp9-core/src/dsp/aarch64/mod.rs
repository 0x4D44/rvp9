//! AArch64-specific SIMD kernel implementations.
//!
//! Safe wrappers are installed only when [`crate::dsp::caps::AArch64Caps`]
//! reports NEON. Each wrapper validates its slice footprint before entering
//! a `#[target_feature]` inner loop.

#![allow(unsafe_op_in_unsafe_fn)]

pub mod convolve8_neon;
pub mod quantize_neon;
pub mod subpel_variance_neon;
pub mod variance_neon;

pub use convolve8_neon::{convolve8_horiz_neon, convolve8_neon, convolve8_vert_neon};
pub use quantize_neon::{vp9_quantize_b_neon, vp9_quantize_fp_neon};
pub use subpel_variance_neon::{
    vpx_sub_pixel_variance16x8_neon, vpx_sub_pixel_variance16x16_neon,
    vpx_sub_pixel_variance16x32_neon, vpx_sub_pixel_variance32x16_neon,
    vpx_sub_pixel_variance32x32_neon, vpx_sub_pixel_variance32x64_neon,
    vpx_sub_pixel_variance64x32_neon, vpx_sub_pixel_variance64x64_neon,
};
pub use variance_neon::{
    get_var_sse_8x8_neon, var_sum_sse_8x4_neon, var_sum_sse_8x8_neon, var_sum_sse_8x16_neon,
    var_sum_sse_16x8_neon, var_sum_sse_16x16_neon, var_sum_sse_16x32_neon, var_sum_sse_32x16_neon,
    var_sum_sse_32x32_neon, var_sum_sse_32x64_neon, var_sum_sse_64x32_neon, var_sum_sse_64x64_neon,
};
