//! rvp9-core: Shared VP9 codec primitives.
//!
//! This crate hosts code used by both the encoder and decoder. It is the
//! foundation of the project — everything in here is bit-exact against
//! libvpx and validated by both the decoder conformance suite (Stage 3) and
//! per-kernel SIMD-vs-scalar differential tests.
//!
//! Planned modules (see `PLAN.md`, Stage 3):
//!
//! - [`dsp`] — per-arch SIMD dispatch (scalar / SSE2 / SSSE3 / SSE4.1 / AVX2 / NEON)
//! - `bitstream` — byte reader/writer, arithmetic coder *(Stage 3 task 1)*
//! - `entropy` — probability tables, context modelling *(Stage 3 task 2)*
//! - `transforms` — forward + inverse DCT/ADST at 4×4 / 8×8 / 16×16 / 32×32 *(Stage 3 task 3)*
//! - `quant` — quantizer tables, quant/dequant *(Stage 3 task 4)*
//! - `intra_pred` — 10 intra modes × 4 block sizes *(Stage 3 task 5)*
//! - `loopfilter` — deblocking filter *(Stage 3 task 6)*
//! - `frame` — frame buffer management + reference frame pool *(Stage 3 task 7)*
//!
//! Status: Stage 3 task 0 in progress (`dsp` dispatch scaffolding).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod bitstream;
pub mod dsp;
pub mod entropy;
pub mod frame;
pub mod intra_pred;
pub mod quant;
pub mod transforms;

/// The pixel type. Per **D-01 RESOLVED** (PLAN.md §3): 8-bit only in v1.
/// HBD (10-bit / 12-bit) is deferred to Stage 6+; v1 ships a concrete `u8`
/// type, not a generic over pixel depth.
///
/// See `notes/hld/00-architecture.md` §7 ("8-bit-only pixel-type policy").
pub type Pixel = u8;

/// The transform-domain coefficient type (low magnitude). 16-bit signed,
/// matching libvpx's `tran_low_t` at `CONFIG_VP9_HIGHBITDEPTH == 0`.
pub type Coeff = i16;

/// The transform-domain intermediate type (high magnitude). 32-bit signed,
/// matching libvpx's `tran_high_t`. Required even in 8-bit-only v1: forward
/// DCT intermediates can exceed 16 bits before the scale-back step. See
/// `notes/hld/00-architecture.md` §6 for the integer-width policy.
pub type TranLow = i32;
