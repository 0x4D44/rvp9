//! AArch64-specific SIMD kernel implementations.
//!
//! Safe wrappers are installed only when [`crate::dsp::caps::AArch64Caps`]
//! reports NEON. Each wrapper validates its slice footprint before entering
//! a `#[target_feature]` inner loop.

#![allow(unsafe_op_in_unsafe_fn)]

pub mod subpel_variance_neon;

pub use subpel_variance_neon::{
    vpx_sub_pixel_variance16x8_neon, vpx_sub_pixel_variance16x16_neon,
    vpx_sub_pixel_variance16x32_neon, vpx_sub_pixel_variance32x16_neon,
    vpx_sub_pixel_variance32x32_neon, vpx_sub_pixel_variance32x64_neon,
    vpx_sub_pixel_variance64x32_neon, vpx_sub_pixel_variance64x64_neon,
};
