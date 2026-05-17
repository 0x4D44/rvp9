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
    idct4, idct4x4_1_add, idct4x4_16_add, idct8, idct8x8_1_add, idct8x8_64_add, idct16,
    idct16x16_1_add, idct16x16_256_add, idct32, idct32x32_1_add, idct32x32_1024_add,
};
pub use scans::{DEFAULT_ISCAN_4X4, DEFAULT_SCAN_4X4, DEFAULT_SCAN_4X4_NEIGHBORS, ScanOrder, default_scan_4x4};
pub use types::{MAX_NEIGHBORS, TxMode, TxSize, TxType};
