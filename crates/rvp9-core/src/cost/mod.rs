//! Encoder bit-cost machinery — per-symbol bit-cost lookups and the
//! RD multiplier (lambda) used by mode-decision.
//!
//! This module is the rate side of the encoder's
//! `D + lambda * R` rate-distortion comparison. Distortion primitives
//! (variance / SAD) live in [`crate::dsp`]; the rate side lives here.
//!
//! ## Sub-modules
//!
//! - [`prob_cost`] — the 256-entry `vp9_prob_cost` lookup table and the
//!   `cost_zero` / `cost_one` / `cost_bit` helpers. Mirrors libvpx
//!   `vp9/encoder/vp9_cost.{c,h}`.
//! - [`tree_cost`] — `vp9_cost_tokens` / `vp9_cost_tokens_skip`: walk a
//!   flat `vpx_tree_index`-style tree and produce per-leaf bit costs.
//! - [`symbol_costs`] — per-context bit-cost tables for the mode /
//!   partition / inter-mode / skip / switchable-interp / tx-size /
//!   intra-Y / intra-UV trees. Mirrors libvpx `vp9_initialize_rd_consts`
//!   (`vp9_rd.c:396-`) and `fill_mode_costs` (`vp9_rd.c:92-133`).
//! - [`mod@mv_cost`] — MV joint-cost vector + per-axis component-cost
//!   tables. Mirrors libvpx `build_nmv_component_cost_table` and
//!   `vp9_build_nmv_cost_table` (`vp9_encodemv.c:69-237`).
//! - [`coef_cost`] — the big `[tx][plane][ref][band][ctx][token]`
//!   coefficient cost table built by libvpx's `fill_token_costs`
//!   (`vp9_rd.c:135-152`).
//! - [`rd_mult`] — the Lagrange-multiplier `lambda` derived from
//!   `qindex`. Mirrors libvpx `vp9_compute_rd_mult_based_on_qindex`
//!   (`vp9_rd.c:241-276`) — Path-A MVP only uses the inter-frame /
//!   key-frame branches; ARF/golden frames inherit the inter formula.
//!
//! All costs are in 1/512-bit fixed-point units (`VP9_PROB_COST_SHIFT =
//! 9`) — `512` represents one bit.

pub mod coef_cost;
pub mod mv_cost;
pub mod prob_cost;
pub mod rd_mult;
pub mod symbol_costs;
pub mod tree_cost;

pub use coef_cost::{CoefCostIndex, CoefCostTable, build_coef_cost_table, coef_token_cost};
pub use mv_cost::{
    MV_COST_MAX, MvComponentCosts, MvCosts, build_nmv_component_cost_table, build_nmv_cost_table,
    mv_component_cost, mv_cost,
};
pub use prob_cost::{
    VP9_PROB_COST, VP9_PROB_COST_SHIFT, vp9_cost_bit, vp9_cost_one, vp9_cost_zero,
};
pub use rd_mult::{
    FrameTypeForRd, compute_rd_mult, compute_rd_mult_based_on_qindex, get_intra_cost_penalty,
};
pub use symbol_costs::{
    InterModeCosts, KfPartitionCosts, PartitionCosts, SkipCosts, SwitchableInterpCosts,
    TxSizeCostTable, intra_uv_mode_costs_inter, intra_uv_mode_costs_kf, intra_y_mode_costs_inter,
    intra_y_mode_costs_kf, intra_y_mode_costs_kf_table, partition_costs, partition_costs_kf,
    skip_costs, switchable_interp_costs, tx_size_costs,
};
pub use tree_cost::{vp9_cost_tokens, vp9_cost_tokens_skip};
