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

pub mod dc_pred_4x4_avx2;
pub mod idct4x4_avx2;
pub mod idct8x8_avx2;
pub mod idct16x16_10_avx2;
pub mod idct16x16_38_avx2;
pub mod idct16x16_avx2;
pub mod idct32x32_135_avx2;
pub mod idct32x32_34_avx2;
pub mod idct32x32_avx2;
pub mod iht4x4_avx2;
pub mod iht8x8_avx2;
pub mod iht16x16_avx2;
pub mod lpf_horizontal_4_avx2;
pub mod lpf_horizontal_4_dual_avx2;
pub mod lpf_horizontal_8_avx2;
pub mod lpf_horizontal_8_dual_avx2;
pub mod lpf_horizontal_16_avx2;
pub mod lpf_horizontal_16_dual_avx2;
pub mod lpf_vertical_4_avx2;
pub mod lpf_vertical_4_dual_avx2;
pub mod lpf_vertical_8_avx2;
pub mod lpf_vertical_8_dual_avx2;
pub mod lpf_vertical_16_avx2;
pub mod lpf_vertical_16_dual_avx2;
pub mod plane_dequant_finisher_avx2;
pub mod v_predictor_avx2;

pub use dc_pred_4x4_avx2::dc_predictor_4x4_avx2;
pub use idct4x4_avx2::idct4x4_add_avx2;
pub use idct8x8_avx2::idct8x8_add_avx2;
pub use idct16x16_10_avx2::idct16x16_10_add_avx2;
pub use idct16x16_38_avx2::idct16x16_38_add_avx2;
pub use idct16x16_avx2::idct16x16_add_avx2;
pub use idct32x32_135_avx2::idct32x32_135_add_avx2;
pub use idct32x32_34_avx2::idct32x32_34_add_avx2;
pub use idct32x32_avx2::idct32x32_add_avx2;
pub use iht4x4_avx2::iht4x4_add_avx2;
pub use iht8x8_avx2::iht8x8_add_avx2;
pub use iht16x16_avx2::iht16x16_add_avx2;
pub use lpf_horizontal_4_avx2::lpf_horizontal_4_avx2;
pub use lpf_horizontal_4_dual_avx2::lpf_horizontal_4_dual_avx2;
pub use lpf_horizontal_8_avx2::lpf_horizontal_8_avx2;
pub use lpf_horizontal_8_dual_avx2::lpf_horizontal_8_dual_avx2;
pub use lpf_horizontal_16_avx2::lpf_horizontal_16_avx2;
pub use lpf_vertical_4_avx2::lpf_vertical_4_avx2;
pub use lpf_vertical_4_dual_avx2::lpf_vertical_4_dual_avx2;
pub use lpf_vertical_8_avx2::lpf_vertical_8_avx2;
pub use lpf_horizontal_16_dual_avx2::lpf_horizontal_16_dual_avx2;
pub use lpf_vertical_8_dual_avx2::lpf_vertical_8_dual_avx2;
pub use lpf_vertical_16_avx2::lpf_vertical_16_avx2;
pub use lpf_vertical_16_dual_avx2::lpf_vertical_16_dual_avx2;
pub use plane_dequant_finisher_avx2::plane_dequant_finisher_avx2;
pub use v_predictor_avx2::v_predictor_avx2;
