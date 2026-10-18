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
pub mod block;
pub mod coef_ctx;
pub mod cost;
pub mod dsp;
pub mod entropy;
pub mod frame;
pub mod inter;
pub mod intra_pred;
pub mod loopfilter;
pub mod mode_decision;
pub mod mode_info;
pub mod mv_search;
pub mod quant;
pub mod tile_threading;
pub mod transforms;
// R.1 — process-wide cache of RVP9_TRACE_* / RVP9_DEBUG_* env vars
// read by hot-path core kernels (intra-pred, entropy detokenize).
mod trace_flags;

/// The 8-bit pixel type. Per **D-01 RESOLVED** (PLAN.md §3): 8-bit is
/// the only pixel storage currently driven by kernel code.
///
/// libvpx parity: `uint8_t` plane storage when
/// `CONFIG_VP9_HIGHBITDEPTH == 0`. The HBD path uses
/// `HbdPixel` (`uint16_t` in libvpx) and lives behind the
/// `highbitdepth` feature flag (W6.1 scaffold); the kernel pairs
/// themselves are pending (see `dsp/highbd.rs`).
///
/// See `notes/hld/00-architecture.md` §7 ("8-bit-only pixel-type policy").
pub type Pixel = u8;

/// The 16-bit pixel container used by the HBD kernel pairs. Only
/// available under `--features highbitdepth`. Matches libvpx's
/// `uint16_t` plane storage in `CONFIG_VP9_HIGHBITDEPTH` (see
/// `vpx_scale/yv12config.h:67`, `YV12_FLAG_HIGHBITDEPTH = 8`).
///
/// W6.1 scaffold — the kernels themselves are unimplemented stubs in
/// [`dsp::highbd`].
#[cfg(feature = "highbitdepth")]
pub type HbdPixel = u16;

/// Bit-depth of the pixel pipeline. Default builds expose only
/// [`Bd8`](BitDepth::Bd8); `--features highbitdepth` adds
/// `Bd10` and `Bd12`. Mirrors libvpx
/// `enum vpx_bit_depth_t` (`vpx/vpx_image.h:42-46`).
///
/// W6.1 scaffold (vpxenc parity port). This is **separate** from
/// [`bitstream::BitDepth`] (`Eight`/`Ten`/`Twelve`) which is the
/// wire-format enum read from the uncompressed header; the
/// crate-root enum here tracks the pixel-storage choice used by
/// kernels and frame buffers.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum BitDepth {
    /// 8-bit (default).
    Bd8 = 8,
    /// 10-bit (only available under `--features highbitdepth`).
    #[cfg(feature = "highbitdepth")]
    Bd10 = 10,
    /// 12-bit (only available under `--features highbitdepth`).
    #[cfg(feature = "highbitdepth")]
    Bd12 = 12,
}

impl BitDepth {
    /// Numeric bit-depth (8/10/12).
    #[must_use]
    #[inline]
    pub const fn bits(self) -> u8 {
        match self {
            Self::Bd8 => 8,
            #[cfg(feature = "highbitdepth")]
            Self::Bd10 => 10,
            #[cfg(feature = "highbitdepth")]
            Self::Bd12 => 12,
        }
    }
}

impl Default for BitDepth {
    #[inline]
    fn default() -> Self {
        Self::Bd8
    }
}

/// The transform-domain coefficient type (low magnitude). 16-bit signed,
/// matching libvpx's `tran_low_t` at `CONFIG_VP9_HIGHBITDEPTH == 0`.
pub type Coeff = i16;

/// The transform-domain intermediate type (high magnitude). 32-bit signed,
/// matching libvpx's `tran_high_t`. Required even in 8-bit-only v1: forward
/// DCT intermediates can exceed 16 bits before the scale-back step. See
/// `notes/hld/00-architecture.md` §6 for the integer-width policy.
pub type TranLow = i32;

#[cfg(test)]
mod bit_depth_scaffold_tests {
    //! W6.1 scaffold — pin the cardinality and `.bits()` semantics of the
    //! crate-root [`BitDepth`] enum. Default builds see one variant
    //! (`Bd8`); the `highbitdepth` feature adds `Bd10` and `Bd12`. The
    //! `HbdPixel` alias is exercised under the feature too.

    use super::BitDepth;

    #[test]
    fn bd8_bits_is_eight() {
        assert_eq!(BitDepth::Bd8.bits(), 8);
        assert_eq!(BitDepth::default(), BitDepth::Bd8);
    }

    #[cfg(feature = "highbitdepth")]
    #[test]
    fn hbd_variants_have_correct_bits() {
        assert_eq!(BitDepth::Bd10.bits(), 10);
        assert_eq!(BitDepth::Bd12.bits(), 12);
        // HbdPixel exists and is u16-shaped — pin via a const-eval.
        const _ASSERT_SIZE: () = {
            assert!(std::mem::size_of::<super::HbdPixel>() == 2);
        };
    }
}
