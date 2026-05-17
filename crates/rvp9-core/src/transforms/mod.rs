//! VP9 transform layer — types, scan orders, and (Stage 3.3+) the forward
//! and inverse DCT/ADST kernels.
//!
//! Per the transforms LLD `notes/lld/transforms.md`. This module owns the
//! scalar reference implementations of every transform kernel libvpx
//! ships in `vp9/common/vp9_idct.c` plus the encoder forward kernels in
//! `vp9/encoder/vp9_dct.c`. SIMD specialisations dispatch through
//! [`crate::dsp::SimdDispatch`] (Stage 3.0 scaffolding).
//!
//! ## Status
//!
//! Stage 3 task 3a lands the type definitions, `ScanOrder` struct, and
//! the default 4×4 scan tables verbatim from libvpx. Larger sizes
//! (8×8 / 16×16 / 32×32) and per-`TxType` variants (col/row scans for
//! ADST_DCT / DCT_ADST / ADST_ADST) arrive in follow-up sub-tasks.
//!
//! L-28 RESOLVED (trellis-quant `vp9_optimize_b` not ported) is honoured
//! in the kernel layer once it lands; no per-mode RDO loop is exposed
//! from this module.

pub mod constants;
pub mod inverse;
pub mod scans;
pub mod types;

pub use inverse::{
    iadst4, iadst8, iadst16, idct4, idct4x4_1_add, idct4x4_16_add, idct8, idct8x8_1_add,
    idct8x8_64_add, idct16, idct16x16_1_add, idct16x16_256_add, idct32, idct32x32_1_add,
    idct32x32_1024_add, iht4x4_16_add, iht8x8_64_add, iht16x16_256_add,
};
pub use scans::{
    COL_ISCAN_4X4, COL_ISCAN_8X8, COL_ISCAN_16X16, COL_SCAN_4X4, COL_SCAN_4X4_NEIGHBORS,
    COL_SCAN_8X8, COL_SCAN_8X8_NEIGHBORS, COL_SCAN_16X16, COL_SCAN_16X16_NEIGHBORS,
    DEFAULT_ISCAN_4X4, DEFAULT_ISCAN_8X8, DEFAULT_ISCAN_16X16, DEFAULT_ISCAN_32X32,
    DEFAULT_SCAN_4X4, DEFAULT_SCAN_4X4_NEIGHBORS, DEFAULT_SCAN_8X8, DEFAULT_SCAN_8X8_NEIGHBORS,
    DEFAULT_SCAN_16X16, DEFAULT_SCAN_16X16_NEIGHBORS, DEFAULT_SCAN_32X32,
    DEFAULT_SCAN_32X32_NEIGHBORS, ROW_ISCAN_4X4, ROW_ISCAN_8X8, ROW_ISCAN_16X16, ROW_SCAN_4X4,
    ROW_SCAN_4X4_NEIGHBORS, ROW_SCAN_8X8, ROW_SCAN_8X8_NEIGHBORS, ROW_SCAN_16X16,
    ROW_SCAN_16X16_NEIGHBORS, ScanOrder, col_scan_4x4, col_scan_8x8, col_scan_16x16,
    default_scan, default_scan_4x4, default_scan_8x8, default_scan_16x16, default_scan_32x32,
    row_scan_4x4, row_scan_8x8, row_scan_16x16, scan_for, scan_order_4x4, scan_order_8x8,
    scan_order_16x16, scan_order_32x32,
};
pub use types::{MAX_NEIGHBORS, TxMode, TxSize, TxType};
