//! VP9 inter-prediction support: motion-vector decode, reference-frame
//! resolution, sub-pel interpolation, and compound prediction.
//!
//! This module is the home of the per-block inter pipeline that
//! [`crate::mode_info`]'s `BlockMode::Inter` branch feeds. Each
//! sub-module corresponds to a logical phase of the inter decode:
//!
//! - [`mv`] — boolean-coded motion-vector decode (`read_mv` +
//!   `read_mv_component`, libvpx `vp9_decodemv.c:235-285`).
//!
//! More sub-modules (`find_mv_refs`, `mv_pred`, interp filtering)
//! will land as Stage 3 Tasks B.2+ slot in.

pub mod mv;

pub use mv::{read_mv, read_mv_component};
