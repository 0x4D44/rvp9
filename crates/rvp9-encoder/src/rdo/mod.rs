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
//! | [`pick_partition`] | `vp9_encodeframe.c:3985-4489` (rd_pick_partition, NONE+SPLIT) + `vp9_encodeframe.c:1975-2107` (rd_pick_sb_modes) |
//! | [`sub8`]        | `vp9_rdopt.c:4285-5060` (vp9_rd_pick_inter_mode_sub8x8 skeleton) + `vp9_rdopt.c:2077-2428` (rd_pick_best_sub8x8_mode, compound arm) |
//! | [`sub8_kf_intra`] | `vp9_rdopt.c:1061-1297` (rd_pick_intra4x4block) + `vp9_rdopt.c:1299-1360` (rd_pick_intra_sub_8x8_y_mode) |
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
pub mod inter_dispatch;
pub mod pick_inter_sb;
pub mod pick_partition;
pub mod ref_costs;
pub mod single_mv_search;
pub mod sub8;
pub mod sub8_kf_intra;
pub mod trellis;
