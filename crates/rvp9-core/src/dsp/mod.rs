//! Per-arch SIMD dispatch and DSP kernel families.
//!
//! This module is the RTCD-equivalent for rvp9. It owns:
//!
//! - [`caps`] — runtime CPU capability detection (`X86Caps`, `AArch64Caps`).
//! - [`dispatch`] — the `SimdDispatch` table struct + `OnceLock`-based
//!   `init()` / `get()` API enforcing the **R-09 init invariant** (the
//!   table is populated exactly once at `Encoder::new` / `Decoder::new`
//!   before any worker thread is spawned).
//!
//! Per `notes/hld/02-simd-dispatch.md` §2.2: workers always read through
//! a fully-populated `&'static SimdDispatch`. This is statically enforced
//! by `WorkerPool::new(&'static SimdDispatch)` (see `notes/lld/tile-threading.md`
//! §6.5).
//!
//! Per HLD-simd §1.4 (Q2 RESOLVED): the dispatch tables are hand-written
//! Rust, not generated from `vpx_dsp_rtcd_defs.pl`. Kernel families live
//! under `rvp9_core::dsp::*` and each declares both a scalar reference
//! and zero-or-more per-ISA implementations.
//!
//! ## Adding a new kernel family
//!
//! 1. Add a sub-module under `src/dsp/` with the scalar implementation.
//! 2. Add an `Option<fn>` slot to `SimdDispatch` per kernel size.
//! 3. Populate the slot in `SimdDispatch::build`.
//! 4. Add a differential test in the same file: random inputs through
//!    scalar + every-ISA variant, byte-exact equality.
//!
//! ## Status
//!
//! Stage 3 Task 0: scaffolding only — `SimdDispatch` is declared with the
//! per-family slots empty (`Option<fn>` currently `None` everywhere). Real
//! kernel bodies land per Stage 3 tasks 1-7 (one task per shared-infra
//! module: bitstream, entropy, transforms, quant, intra-pred, loopfilter,
//! frame-buffers).

pub mod caps;
pub mod dispatch;
pub mod distortion;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

pub use dispatch::{SimdDispatch, get, get_ref, init};

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", test))]
pub(crate) fn validate_predictor_storage(
    dst_len: usize,
    stride: usize,
    bs: usize,
    edge_len: usize,
) -> Result<usize, &'static str> {
    let rows = bs.checked_sub(1).ok_or("unsupported block size")?;
    if stride < bs {
        return Err("stride shorter than block width");
    }
    let required = rows
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(bs))
        .ok_or("predictor layout overflow")?;
    if edge_len < bs {
        return Err("edge slice too short");
    }
    if dst_len < required {
        return Err("destination slice too short");
    }
    Ok(required)
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", test))]
pub(crate) fn validate_intra_predictor_layout(
    dst_len: usize,
    stride: usize,
    bs: usize,
    edge_len: usize,
) -> Result<usize, &'static str> {
    if !matches!(bs, 4 | 8 | 16 | 32) {
        return Err("unsupported block size");
    }
    validate_predictor_storage(dst_len, stride, bs, edge_len)
}

#[cfg(test)]
mod tests {
    use super::{validate_intra_predictor_layout, validate_predictor_storage};

    #[test]
    fn intra_predictor_layout_rejects_non_vp9_block_size() {
        assert_eq!(
            validate_intra_predictor_layout(9, 3, 3, 3),
            Err("unsupported block size")
        );
    }

    #[test]
    fn intra_predictor_layout_rejects_short_and_overflowing_storage() {
        assert_eq!(
            validate_intra_predictor_layout(16, 4, 4, 3),
            Err("edge slice too short")
        );
        assert_eq!(
            validate_intra_predictor_layout(15, 4, 4, 4),
            Err("destination slice too short")
        );
        assert_eq!(
            validate_intra_predictor_layout(usize::MAX, usize::MAX, 32, 32),
            Err("predictor layout overflow")
        );
    }

    #[test]
    fn predictor_storage_accepts_checked_scalar_fallback_size() {
        assert_eq!(validate_predictor_storage(64 * 64, 64, 64, 64), Ok(4096));
        assert_eq!(
            validate_predictor_storage(0, 0, 0, 0),
            Err("unsupported block size")
        );
    }
}
