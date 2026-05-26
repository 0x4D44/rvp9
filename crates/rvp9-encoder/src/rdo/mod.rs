//! Wave 4 — RDO mode-decision primitives.
//!
//! This module ports the core building blocks of libvpx's
//! rate-distortion-optimised (RDO) mode-decision path. The W4 batch
//! lands the **primitives** as standalone, unit-tested modules; later
//! waves (W4.5+) wire them into the inter-mode picker and partition
//! driver.
//!
//! ## libvpx mapping
//!
//! | rvp9 module     | libvpx source                                |
//! |-----------------|----------------------------------------------|
//! | [`block_yrd`]   | `vp9_rdopt.c:561-1042` (super_block_yrd, choose_tx_size_from_rd, dist_block) |
//! | [`block_error`] | `vp9_rdopt.c:319-332`, `vp9_rdopt.c:521-559` (block_error, pixel_sse) |
//! | [`cost_coeffs`] | `vp9_rdopt.c:358-459` (cost_coeffs + band_counts walker) |
//! | [`trellis`]     | `vp9_encodemb.c:66-487` (xform_quant_fp / dc / vp9_optimize_b) |
//!
//! ## Scope
//!
//! - **8-bit only.** HBD variants are out of scope (D-01 RESOLVED).
//! - **Scalar reference.** SIMD parity lands in Wave 6+.
//! - **Standalone.** No encoder integration in this batch; the W4.5+
//!   inter-mode picker will compose these.

pub mod block_error;
pub mod block_yrd;
pub mod cost_coeffs;
pub mod discount_newmv;
pub mod handle_inter;
pub mod pick_inter_sb;
pub mod ref_costs;
pub mod single_mv_search;
pub mod trellis;
