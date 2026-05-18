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
//!
//! More sub-modules (interp filtering, compound prediction) will
//! land as Stage 3 Tasks B.3+ slot in.

pub mod mv;
pub mod mv_ref;

pub use mv::{read_mv, read_mv_component};
pub use mv_ref::{
    append_sub8x8_mvs_for_idx, dec_find_mv_refs, find_best_ref_mvs, find_mv_refs,
    lower_mv_precision, mv_pred, sign_bias_from_header, BlockBounds, NeighbourLookup, Position,
    PrevFrameMvRef, RefMvList, MAX_MV_REF_CANDIDATES, MV_BORDER, MVREF_NEIGHBOURS,
};
