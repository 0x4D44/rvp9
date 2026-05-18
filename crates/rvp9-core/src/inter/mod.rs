//! VP9 inter-prediction support: motion-vector decode, reference-frame
//! resolution, sub-pel interpolation, and compound prediction.
//!
//! This module is the home of the per-block inter pipeline that
//! [`crate::mode_info`]'s `BlockMode::Inter` branch feeds. Each
//! sub-module corresponds to a logical phase of the inter decode:
//!
//! - [`mv`] — boolean-coded motion-vector decode (`read_mv` +
//!   `read_mv_component`, libvpx `vp9_decodemv.c:235-285`).
//! - [`mv_ref`] — decoder-side MV reference resolution
//!   (`dec_find_mv_refs`, `append_sub8x8_mvs_for_idx`,
//!   `vp9_find_best_ref_mvs`).
//! - [`filter`] — 8-tap sub-pel interpolation filter tables and the
//!   `convolve8` family of scalar kernels (libvpx
//!   `vpx_dsp/vpx_convolve.c` + `vp9/common/vp9_filter.c`). Includes
//!   both unscaled (`convolve8_*`) and scaled (`convolve8_scaled_*`)
//!   variants — the latter consumed by [`scale::inter_predictor`]
//!   when [`ScaleFactors::is_scaled`] is true.
//! - [`scale`] — reference-frame Q14 fixed-point scale factors and
//!   the `inter_predictor` per-block dispatcher (libvpx
//!   `vp9/common/vp9_scale.{h,c}` +
//!   `vp9/common/vp9_reconinter.h`).
//!
//! More sub-modules (compound prediction wireup) will land as
//! Stage 3 Tasks C.3+ slot in.

pub mod filter;
pub mod mv;
pub mod mv_ref;
pub mod scale;

pub use filter::{
    convolve8, convolve8_avg, convolve8_avg_horiz, convolve8_avg_vert, convolve8_horiz,
    convolve8_scaled, convolve8_scaled_avg, convolve8_scaled_avg_horiz,
    convolve8_scaled_avg_vert, convolve8_scaled_horiz, convolve8_scaled_vert, convolve8_vert,
    convolve_avg, convolve_copy, kernel_for, BILINEAR_KERNEL, EIGHTTAP_KERNEL,
    EIGHTTAP_SHARP_KERNEL, EIGHTTAP_SMOOTH_KERNEL, FILTER_BITS, FILTER_TAPS, SUBPEL_BITS,
    SUBPEL_MASK, SUBPEL_SHIFTS, SUBPEL_TAPS,
};
pub use mv::{read_mv, read_mv_component};
pub use mv_ref::{
    append_sub8x8_mvs_for_idx, dec_find_mv_refs, find_best_ref_mvs, find_mv_refs,
    lower_mv_precision, mv_pred, sign_bias_from_header, BlockBounds, NeighbourLookup, Position,
    PrevFrameMvRef, RefMvList, MAX_MV_REF_CANDIDATES, MV_BORDER, MVREF_NEIGHBOURS,
};
pub use scale::{
    inter_predictor, valid_ref_frame_size, ScaleFactors, REF_INVALID_SCALE, REF_NO_SCALE,
    REF_SCALE_SHIFT,
};
