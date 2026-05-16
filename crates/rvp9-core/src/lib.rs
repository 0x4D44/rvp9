//! rvp9-core: Shared VP9 codec primitives.
//!
//! This crate hosts code used by both the encoder and decoder. It is the
//! foundation of the project — everything in here is bit-exact against
//! libvpx and validated by both the decoder conformance suite (Stage 3) and
//! per-kernel SIMD-vs-scalar differential tests.
//!
//! Planned modules (see `PLAN.md`, Stage 3):
//!
//! - `bitstream` — byte reader/writer, arithmetic coder
//! - `entropy` — probability tables (verbatim from libvpx), context modelling
//! - `transforms` — forward + inverse DCT/ADST at 4x4 / 8x8 / 16x16 / 32x32
//! - `quant` — quantizer tables, quant/dequant
//! - `intra_pred` — 10 intra modes x 4 block sizes
//! - `loopfilter` — deblocking filter
//! - `frame` — frame buffer management + reference frame pool
//! - `simd` — per-arch dispatch (scalar / SSE2 / SSE4.1 / AVX2 / NEON)
//!
//! Status: empty scaffold (Stage 0).
