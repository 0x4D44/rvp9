//! Encoder motion-vector search (full-pel diamond + sub-pel refinement).
//!
//! Ports the realtime-default MV search primitives from
//! `vp9/encoder/vp9_mcomp.c`. The realtime path uses:
//!
//! 1. [`diamond_search_sad`] — full-pixel diamond step search; mirrors
//!    `vp9_diamond_search_sad_c` (`vp9_mcomp.c:2055-2190`). Configured
//!    with the 4-site diamond pattern built by
//!    [`init_dsmotion_compensation`] (mirror of
//!    `vp9_init_dsmotion_compensation`, `vp9_mcomp.c:98-114`).
//! 2. [`find_best_sub_pixel_tree`] — the ordinary realtime sub-pel tree.
//! 3. [`find_best_sub_pixel_tree_pruned_more`] and
//!    [`find_best_sub_pixel_tree_pruned_evenmore`] — pruned SVC variants;
//!    the latter is selected for non-reference enhancement pictures at
//!    cpu-used 7 and 8.
//!
//! Supporting helpers:
//!
//! - [`PlaneRef`] — a stride-aware read-only view into one plane of a
//!   frame buffer. Used as the source/reference adapter for the search
//!   loop; defined here because the decoder's frame-pool view isn't
//!   suitable for direct encoder consumption.
//! - [`MvLimits`] — full-pel-unit search-range bounds. Built by
//!   [`set_mv_search_range`] (mirror of `vp9_set_mv_search_range`).
//!
//! ## libvpx-vs-Rust divergences (recorded)
//!
//! - **No `vp9_init3smotion_compensation`** — the 8-site NSTEP config
//!   is out of scope for F7.0 (only the 4-site diamond used at cpu-used
//!   ≥6 is ported).
//! - **No second-pred / compound search** — single-ref only.
//! - **MV cost scaling** — `mvsad_err_cost` uses
//!   `ROUND_POWER_OF_TWO(mv_cost * sad_per_bit, VP9_PROB_COST_SHIFT)`,
//!   where `VP9_PROB_COST_SHIFT = 9`. `mv_err_cost` uses
//!   `ROUND64_POWER_OF_TWO(mv_cost * error_per_bit, RDDIV_BITS +
//!   VP9_PROB_COST_SHIFT - RD_EPB_SHIFT + PIXEL_TRANSFORM_ERROR_SCALE)
//!   = 9 + 7 - 6 + 4 = 14` per `vp9_mcomp.c:91-94`. Both are mirrored
//!   exactly in [`mvsad_err_cost`] and [`mv_err_cost`].

pub mod e12stat;
pub mod full_pixel;
pub mod int_pro_me;
pub mod mv_limits;
pub mod mv_pred;
pub mod search_site;
pub mod sub_pixel;

pub use full_pixel::{
    FullPixelCostList, FullPixelSearchParams, FullPixelSearchResult, PlaneRef, diamond_search_sad,
    fast_diamond_search_sad_with_cost_list, mv_err_cost, mvsad_err_cost,
};
pub use int_pro_me::{IntProMeResult, int_pro_motion_estimation, int_pro_motion_estimation_seed};
pub use mv_limits::{
    MAX_FIRST_STEP, MAX_FULL_PEL_VAL, MAX_MVSEARCH_STEPS, MV_LOW, MV_UPP, MvLimits, is_mv_in,
    set_mv_search_range, set_subpel_mv_search_range,
};
pub use mv_pred::mv_pred_sad_seed;
pub use search_site::{SS_COUNT, SearchSiteConfig, init_dsmotion_compensation};
pub use sub_pixel::{
    SubPixelForcedStop, SubPixelSearchParams, SubPixelSearchResult, adaptive_subpel_force_stop,
    find_best_sub_pixel_tree, find_best_sub_pixel_tree_pruned_evenmore,
    find_best_sub_pixel_tree_pruned_more,
};
