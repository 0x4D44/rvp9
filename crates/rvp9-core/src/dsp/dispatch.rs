//! `SimdDispatch` — the per-arch kernel pointer table.
//!
//! Per HLD-simd §2.2 ("R-09 init invariant"): there is exactly one
//! default `SimdDispatch` per running process, lazily constructed at the
//! first call to [`init`] / [`get`] and held in a `OnceLock<Arc<SimdDispatch>>`.
//! Workers never observe a partially-initialised table: `OnceLock`
//! linearises concurrent constructors, and once a worker reads any
//! kernel pointer through a borrow of the returned `Arc`, an Acquire
//! memory barrier guarantees it has seen every other slot too.
//!
//! ## R-09 enforcement
//!
//! `WorkerPool::new(Arc<SimdDispatch>)` (see `notes/lld/tile-threading.md`
//! §6.5) takes the dispatch as a constructor argument; you cannot spawn
//! a worker without first having obtained the `Arc<SimdDispatch>`. This
//! statically enforces the "RTCD initialised before worker spawn"
//! invariant.
//!
//! ## Why `Arc<SimdDispatch>` (not `&'static SimdDispatch`)?
//!
//! V4 of the SIMD HLD (DA3-M6): in-crate tests need to construct a
//! `SimdDispatch` with FORCED caps — e.g. "force the scalar path on an
//! AVX2 CPU" — and pass it explicitly to a tile worker. A
//! `&'static SimdDispatch` from `OnceLock` precludes that test mode:
//! `OnceLock` keeps whichever value won the first init race, and the
//! caps are baked in. By holding `Arc<SimdDispatch>` everywhere in
//! production AND letting in-crate tests construct their own
//! `Arc<SimdDispatch>` via [`SimdDispatch::build`] (`pub(crate)` —
//! external callers route through [`init`] / [`get`] only), the test
//! ergonomics work without compromising the singleton model. The
//! `Arc` clone is one atomic per construction — cheap and bounded.
//!
//! ## Adding a kernel slot (G4.x rollout)
//!
//! 1. Add a `pub(crate)` field to [`SimdDispatch`] for the new
//!    kernel's function pointer, with the kernel's safe-wrapper
//!    signature.
//! 2. Add a `pub fn <kernel>(&self) -> fn(...)` accessor returning the
//!    raw `fn`. External crates dispatch via this accessor (the field
//!    itself is `pub(crate)` to keep slot additions crate-internal).
//! 3. Populate the slot inside [`SimdDispatch::build`], dispatching on
//!    `caps` to pick the best available implementation. Default to
//!    the scalar reference.
//! 4. Co-locate a scalar-vs-SIMD differential test under
//!    `crates/rvp9-core/src/dsp/<arch>/<kernel>_<isa>.rs` (or its
//!    `#[cfg(test)] mod tests`).
//!
//! G4.1 pilot lands the first slot: `idct8x8_add`. See
//! [`crate::dsp::x86_64::idct8x8_avx2`] for the AVX2 implementation
//! and the safe-wrapper convention this slot uses.

use std::sync::{Arc, OnceLock};

use super::caps::{Caps, detect};
use crate::frame::Yv12Buffer;

/// Bit mask of which AVX2 loopfilter kernels to activate. Wave H H1
/// gate; H7 bisection support. See the comment block in
/// [`SimdDispatch::build`] for the env-var grammar.
///
/// `repr(transparent)` over a `u8` so the parser can use simple
/// bit-or accumulation.
#[cfg(target_arch = "x86_64")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LpfAvx2Mask(u8);

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // Bisection flags read at runtime via `contains`.
impl LpfAvx2Mask {
    const NONE: Self = Self(0);
    const ALL: Self = Self(0x3F);
    const V4: Self = Self(1 << 0);
    const H4: Self = Self(1 << 1);
    const V8: Self = Self(1 << 2);
    const H8: Self = Self(1 << 3);
    const V16: Self = Self(1 << 4);
    const H16: Self = Self(1 << 5);

    /// Wave H H7 bisection (2026-05-22) confirmed only
    /// `lpf_vertical_16_avx2` (V16) caused conformance regressions
    /// (58/201 vectors); the bug was a swapped `flat_mask5` operand set
    /// (inner p1..p4 / q1..q4 instead of outer p4..p7 / q4..q7). Wave H
    /// H7 fix (2026-05-22) restored bit-exact parity, so V16 is now
    /// included in DEFAULT alongside the other 5 kernels.
    const DEFAULT: Self = Self::ALL;

    #[inline]
    fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Parse the `RVP9_DSP_LOOPFILTER_AVX2` env var into an [`LpfAvx2Mask`].
///
/// - `None` or `""` → [`LpfAvx2Mask::DEFAULT`] (all 6 AVX2 lpf kernels
///   active post-H7 fix)
/// - `"0"` → [`LpfAvx2Mask::NONE`] (all kernels scalar, for differential
///   testing)
/// - `"1"` → [`LpfAvx2Mask::ALL`] (all 6 AVX2 paths active — same as
///   DEFAULT post-H7)
/// - comma list (e.g. `"v4,h4"`) → only the listed kernels active;
///   useful for bisection or regression-bisecting future SIMD work
///
/// Unknown tokens are silently ignored; the function is deliberately
/// permissive so bisection scripts can pass partial lists without
/// failing the build.
#[cfg(target_arch = "x86_64")]
fn parse_lpf_avx2_mask(s: Option<&str>) -> LpfAvx2Mask {
    let s = match s {
        Some(s) if !s.is_empty() => s,
        _ => return LpfAvx2Mask::DEFAULT,
    };
    match s {
        "0" => return LpfAvx2Mask::NONE,
        "1" => return LpfAvx2Mask::ALL,
        _ => {}
    }
    let mut mask = LpfAvx2Mask::NONE;
    for tok in s.split(',') {
        let t = tok.trim();
        let bit = match t {
            "v4" => LpfAvx2Mask::V4,
            "h4" => LpfAvx2Mask::H4,
            "v8" => LpfAvx2Mask::V8,
            "h8" => LpfAvx2Mask::H8,
            "v16" => LpfAvx2Mask::V16,
            "h16" => LpfAvx2Mask::H16,
            _ => LpfAvx2Mask::NONE,
        };
        mask = mask.union(bit);
    }
    mask
}
use crate::transforms::inverse::{
    highbd_idct4x4_16_add as scalar_highbd_idct4x4_add, idct4x4_16_add as scalar_idct4x4_add,
    idct8x8_64_add as scalar_idct8x8_add, idct16x16_10_add as scalar_idct16x16_10_add,
    idct16x16_38_add as scalar_idct16x16_38_add, idct16x16_256_add as scalar_idct16x16_add,
    idct32x32_34_add as scalar_idct32x32_34_add, idct32x32_135_add as scalar_idct32x32_135_add,
    idct32x32_1024_add as scalar_idct32x32_add, iht4x4_16_add as scalar_iht4x4_add,
    iht8x8_64_add as scalar_iht8x8_add, iht16x16_256_add as scalar_iht16x16_add,
};
use crate::transforms::types::TxType;

/// Scalar adapter for the `dc_predictor_4x4` dispatch slot. The
/// underlying [`crate::intra_pred::dc_predictor`] takes a `PredCtx`
/// (which includes an `above_left` byte unused by DC); this adapter
/// flattens the signature to `(dst, stride, above, left)` so the slot
/// holds the same fn signature as
/// [`crate::dsp::x86_64::dc_predictor_4x4_avx2`].
fn scalar_dc_predictor_4x4(dst: &mut [u8], stride: usize, above: &[u8], left: &[u8]) {
    let ctx = crate::intra_pred::PredCtx {
        above,
        left,
        above_left: 0, // DC_PRED ignores `above_left`.
    };
    crate::intra_pred::dc_predictor(dst, stride, 4, ctx);
}

/// Scalar adapter for the `dc_predict_full` dispatch slot (Wave I I4).
/// The underlying [`crate::intra_pred::dc_predictor`] takes a `PredCtx`
/// (which includes an `above_left` byte unused by DC); this adapter
/// rebuilds a minimal context so the slot holds the flat
/// `(dst, stride, bs, above, left)` signature shared with the AVX2
/// kernel [`crate::dsp::x86_64::dc_predict_full_avx2`].
fn scalar_dc_predict_full(dst: &mut [u8], stride: usize, bs: usize, above: &[u8], left: &[u8]) {
    let ctx = crate::intra_pred::PredCtx {
        above,
        left,
        above_left: 0, // DC_PRED ignores `above_left`.
    };
    crate::intra_pred::dc_predictor(dst, stride, bs, ctx);
}

/// Scalar adapter for the `dc_predict_top` dispatch slot (Wave I I4).
/// Direct alias of [`crate::intra_pred::dc_top_predictor`] — the
/// scalar fn already has the same `(dst, stride, bs, above)`
/// signature as the AVX2 wrapper, so this adapter is structural-only
/// (kept for symmetry with `scalar_dc_predict_full` and to keep the
/// `super::scalar_*` indirection used by the routing tests below).
fn scalar_dc_predict_top(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    crate::intra_pred::dc_top_predictor(dst, stride, bs, above)
}

/// Scalar adapter for the `dc_predict_left` dispatch slot (Wave I I4).
/// Mirrors `scalar_dc_predict_top`: direct alias of
/// [`crate::intra_pred::dc_left_predictor`], structural-only.
fn scalar_dc_predict_left(dst: &mut [u8], stride: usize, bs: usize, left: &[u8]) {
    crate::intra_pred::dc_left_predictor(dst, stride, bs, left)
}

/// Scalar adapter for the `dc_predict_128` dispatch slot (Wave I I4).
/// Direct alias of [`crate::intra_pred::dc_128_predictor`].
fn scalar_dc_predict_128(dst: &mut [u8], stride: usize, bs: usize) {
    crate::intra_pred::dc_128_predictor(dst, stride, bs)
}

/// Scalar adapter for the `v_predictor` dispatch slot (Wave I I1).
/// The underlying [`crate::intra_pred::v_predictor`] takes a
/// `PredCtx`; V_PRED reads only `ctx.above`, so this adapter rebuilds
/// a minimal context (`left: &[]`, `above_left: 0`) to match the
/// flat slot signature `(dst, stride, bs, above)`.
fn scalar_v_predictor(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    let ctx = crate::intra_pred::PredCtx {
        above,
        left: &[],     // V_PRED ignores `left`.
        above_left: 0, // V_PRED ignores `above_left`.
    };
    crate::intra_pred::v_predictor(dst, stride, bs, ctx);
}

/// Scalar adapter for the `h_predictor` dispatch slot (Wave I I2).
/// The underlying [`crate::intra_pred::h_predictor`] takes a
/// `PredCtx`; H_PRED reads only `ctx.left`, so this adapter rebuilds
/// a minimal context (`above: &[]`, `above_left: 0`) to match the
/// flat slot signature `(dst, stride, bs, left)`.
fn scalar_h_predictor(dst: &mut [u8], stride: usize, bs: usize, left: &[u8]) {
    let ctx = crate::intra_pred::PredCtx {
        above: &[], // H_PRED ignores `above`.
        left,
        above_left: 0, // H_PRED ignores `above_left`.
    };
    crate::intra_pred::h_predictor(dst, stride, bs, ctx);
}

/// Scalar adapter for the `tm_predictor` dispatch slot (Wave I I3).
/// The underlying [`crate::intra_pred::tm_predictor`] takes a `PredCtx`;
/// this adapter flattens the signature to
/// `(dst, stride, bs, above, left, above_left)` so the slot holds the
/// same fn signature as [`crate::dsp::x86_64::tm_predictor_avx2`].
fn scalar_tm_predictor(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
    above_left: u8,
) {
    let ctx = crate::intra_pred::PredCtx {
        above,
        left,
        above_left,
    };
    crate::intra_pred::tm_predictor(dst, stride, bs, ctx);
}

/// Scalar adapter for the `d45_predictor` dispatch slot (Wave K K1).
/// Replicates the scalar's `bs == 4` specialization split internally so
/// the slot holds one flat-arg fn matching
/// [`crate::dsp::x86_64::d45_predictor_avx2`].
fn scalar_d45_predictor(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    if bs == 4 {
        crate::intra_pred::d45_predictor_4x4(dst, stride, above);
    } else {
        crate::intra_pred::d45_predictor(dst, stride, bs, above);
    }
}

/// Scalar adapter for the `d63_predictor` dispatch slot (Wave K K3).
fn scalar_d63_predictor(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    if bs == 4 {
        crate::intra_pred::d63_predictor_4x4(dst, stride, above);
    } else {
        crate::intra_pred::d63_predictor(dst, stride, bs, above);
    }
}

/// Scalar adapter for the `d207_predictor` dispatch slot (Wave K K2).
fn scalar_d207_predictor(dst: &mut [u8], stride: usize, bs: usize, left: &[u8]) {
    if bs == 4 {
        crate::intra_pred::d207_predictor_4x4(dst, stride, left);
    } else {
        crate::intra_pred::d207_predictor(dst, stride, bs, left);
    }
}

/// Scalar adapter for the `d117_predictor` dispatch slot (Wave K K5).
/// D117 has no 4×4 special — the generic kernel covers all sizes.
fn scalar_d117_predictor(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
    above_left: u8,
) {
    let ctx = crate::intra_pred::PredCtx {
        above,
        left,
        above_left,
    };
    crate::intra_pred::d117_predictor(dst, stride, bs, ctx);
}

/// Scalar adapter for the `d135_predictor` dispatch slot (Wave K K4).
fn scalar_d135_predictor(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
    above_left: u8,
) {
    let ctx = crate::intra_pred::PredCtx {
        above,
        left,
        above_left,
    };
    crate::intra_pred::d135_predictor(dst, stride, bs, ctx);
}

/// Scalar adapter for the `d153_predictor` dispatch slot (Wave K K6).
fn scalar_d153_predictor(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
    above_left: u8,
) {
    let ctx = crate::intra_pred::PredCtx {
        above,
        left,
        above_left,
    };
    crate::intra_pred::d153_predictor(dst, stride, bs, ctx);
}

/// Scalar reference for the `lpf_vertical_4` dispatch slot. Direct
/// alias of [`crate::loopfilter::lpf_vertical_4`]; the slot exists to
/// reserve the family naming for a future AVX2 implementation, see
/// [`SimdDispatch`] field doc.
fn scalar_lpf_vertical_4(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    crate::loopfilter::lpf_vertical_4(pixels, s_index, pitch, blimit, limit, thresh)
}

/// Scalar reference for the `lpf_horizontal_4` dispatch slot. Direct
/// alias of [`crate::loopfilter::lpf_horizontal_4`]; mirrors the
/// `scalar_lpf_vertical_4` shape — both slots take the same six-arg
/// signature and route to either an AVX2 wrapper or the scalar
/// reference per [`SimdDispatch::build`].
fn scalar_lpf_horizontal_4(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    crate::loopfilter::lpf_horizontal_4(pixels, s_index, pitch, blimit, limit, thresh)
}

/// Scalar reference for the `lpf_vertical_8` dispatch slot (Wave G G4).
fn scalar_lpf_vertical_8(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    crate::loopfilter::lpf_vertical_8(pixels, s_index, pitch, blimit, limit, thresh)
}

/// Scalar reference for the `lpf_horizontal_8` dispatch slot (Wave G G5).
fn scalar_lpf_horizontal_8(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    crate::loopfilter::lpf_horizontal_8(pixels, s_index, pitch, blimit, limit, thresh)
}

/// Scalar reference for the `lpf_vertical_16` dispatch slot (Wave G G6).
fn scalar_lpf_vertical_16(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    crate::loopfilter::lpf_vertical_16(pixels, s_index, pitch, blimit, limit, thresh)
}

/// Scalar reference for the `lpf_horizontal_16` dispatch slot (Wave G G6).
fn scalar_lpf_horizontal_16(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    crate::loopfilter::lpf_horizontal_16(pixels, s_index, pitch, blimit, limit, thresh)
}

// ---- Wave H H1 — dual-variant loopfilter slots (scalar-only initially).
// The 6 `_dual` variants below all take the 9-arg `(pixels, s_index,
// pitch, blimit0, limit0, thresh0, blimit1, limit1, thresh1)` form
// and run the matching single-variant filter on two adjacent 8-pixel
// groups with independent thresholds. AVX2 impls land in H2/H3/H4.

/// Scalar reference for the `lpf_vertical_4_dual` dispatch slot
/// (Wave H H1; AVX2 promotion in Wave H H2).
#[allow(clippy::too_many_arguments)]
fn scalar_lpf_vertical_4_dual(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit0: u8,
    limit0: u8,
    thresh0: u8,
    blimit1: u8,
    limit1: u8,
    thresh1: u8,
) {
    crate::loopfilter::lpf_vertical_4_dual(
        pixels, s_index, pitch, blimit0, limit0, thresh0, blimit1, limit1, thresh1,
    )
}

/// Scalar reference for the `lpf_horizontal_4_dual` dispatch slot
/// (Wave H H1; AVX2 promotion in Wave H H2).
#[allow(clippy::too_many_arguments)]
fn scalar_lpf_horizontal_4_dual(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit0: u8,
    limit0: u8,
    thresh0: u8,
    blimit1: u8,
    limit1: u8,
    thresh1: u8,
) {
    crate::loopfilter::lpf_horizontal_4_dual(
        pixels, s_index, pitch, blimit0, limit0, thresh0, blimit1, limit1, thresh1,
    )
}

/// Scalar reference for the `lpf_vertical_8_dual` dispatch slot
/// (Wave H H1; AVX2 promotion in Wave H H3).
#[allow(clippy::too_many_arguments)]
fn scalar_lpf_vertical_8_dual(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit0: u8,
    limit0: u8,
    thresh0: u8,
    blimit1: u8,
    limit1: u8,
    thresh1: u8,
) {
    crate::loopfilter::lpf_vertical_8_dual(
        pixels, s_index, pitch, blimit0, limit0, thresh0, blimit1, limit1, thresh1,
    )
}

/// Scalar reference for the `lpf_horizontal_8_dual` dispatch slot
/// (Wave H H1; AVX2 promotion in Wave H H3).
#[allow(clippy::too_many_arguments)]
fn scalar_lpf_horizontal_8_dual(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit0: u8,
    limit0: u8,
    thresh0: u8,
    blimit1: u8,
    limit1: u8,
    thresh1: u8,
) {
    crate::loopfilter::lpf_horizontal_8_dual(
        pixels, s_index, pitch, blimit0, limit0, thresh0, blimit1, limit1, thresh1,
    )
}

/// Scalar reference for the `lpf_vertical_16_dual` dispatch slot
/// (Wave H H1; AVX2 promotion in Wave H H4). Six-arg — the 16-tap
/// dual variants share one threshold set across both halves (libvpx
/// `vpx_lpf_vertical_16_dual_c` convention), unlike the 4/8-tap
/// duals.
fn scalar_lpf_vertical_16_dual(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    crate::loopfilter::lpf_vertical_16_dual(pixels, s_index, pitch, blimit, limit, thresh)
}

/// Scalar reference for the `lpf_horizontal_16_dual` dispatch slot
/// (Wave H H1; AVX2 promotion in Wave H H4). Six-arg — see
/// [`scalar_lpf_vertical_16_dual`] for the shared-threshold rationale.
fn scalar_lpf_horizontal_16_dual(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    crate::loopfilter::lpf_horizontal_16_dual(pixels, s_index, pitch, blimit, limit, thresh)
}

/// Scalar reference for the `plane_dequant_finisher` dispatch slot.
/// Direct alias of [`crate::quant::plane_dequant_finisher`].
///
/// Wave H H6 reshaped this slot. Prior to H6 it was a placeholder
/// `plane_dequant_pair: fn(i32, QuantizerDeltas) -> (i16, i16)` that
/// reflected the pre-H5 architecture (dequant fused into the
/// bool-coder loop, no batchable kernel). Wave H H5 split the
/// detokenize loop into a bool-coder pass that deposits four parallel
/// scratch arrays and a uniform pass-2 dequant arithmetic loop. H6
/// retargets the slot at that pass-2 work: 8 i32 lanes per iteration
/// via AVX2 (`_mm256_mullo_epi32` + `_mm256_sra_epi32` +
/// `_mm256_sign_epi32`), with a scalar tail for `< 8` remaining
/// entries.
#[allow(clippy::too_many_arguments)] // mirrors the six-slot scratch layout
fn scalar_plane_dequant_finisher(
    scratch_raster: &[u16],
    scratch_dqv: &[i16],
    scratch_val: &[i16],
    scratch_sign: &[u8],
    dq_shift: u32,
    dq_coefs_out: &mut [i32],
) {
    crate::quant::plane_dequant_finisher(
        scratch_raster,
        scratch_dqv,
        scratch_val,
        scratch_sign,
        dq_shift,
        dq_coefs_out,
    )
}

/// Scalar reference for the `extend_frame_borders` dispatch slot.
/// Direct alias of [`crate::frame::extend_frame_borders`]; the slot
/// exists to reserve the family naming. The inner per-row work is
/// already `slice::fill` and `slice::copy_within`, both of which
/// platform `memset` / `memmove` typically vectorise — adding a
/// manual SIMD layer here would not move the needle, see
/// [`SimdDispatch::extend_frame_borders`] field doc.
fn scalar_extend_frame_borders(yv12: &mut Yv12Buffer) {
    crate::frame::extend_frame_borders(yv12)
}

/// Flat-arg fn type for the ctx-taking directional intra-predictors
/// (D117 / D135 / D153, Wave K): `(dst, stride, bs, above, left,
/// above_left)`. Factored into a `type` alias per clippy's
/// `type_complexity` suggestion.
pub type DirectionalCtxPredFn =
    fn(dst: &mut [u8], stride: usize, bs: usize, above: &[u8], left: &[u8], above_left: u8);

/// Fn type for the TM_PRED slot.
pub type TmPredictorFn =
    fn(dst: &mut [u8], stride: usize, bs: usize, above: &[u8], left: &[u8], above_left: u8);

/// Fn type for dual-threshold 4/8-tap loopfilter slots.
pub type LpfDualFn = fn(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit0: u8,
    limit0: u8,
    thresh0: u8,
    blimit1: u8,
    limit1: u8,
    thresh1: u8,
);

/// Fn type for the split detokenize dequant finisher.
pub type PlaneDequantFinisherFn = fn(
    scratch_raster: &[u16],
    scratch_dqv: &[i16],
    scratch_val: &[i16],
    scratch_sign: &[u8],
    dq_shift: u32,
    dq_coefs_out: &mut [i32],
);

/// Fn type for one-axis 8-tap subpel convolution slots.
pub type Convolve8OneAxisFn = fn(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter: &[i16; 8],
    w: usize,
    h: usize,
);

/// Fn type for two-axis 8-tap subpel convolution slots.
pub type Convolve8TwoAxisFn = fn(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_x: &[i16; 8],
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
);

/// Fn type for the scaled 2-D sub-pel convolutions (Wave L L1):
/// `(src, src_stride, dst, dst_stride, filters, x0_q4, x_step_q4,
/// y0_q4, y_step_q4, w, h)`. Shared by the base and `_avg` slots and
/// factored into a `type` alias per clippy's `type_complexity`.
pub type ScaledConvolveFn = fn(
    &[u8],
    usize,
    &mut [u8],
    usize,
    &[[i16; 8]; crate::inter::filter::SUBPEL_SHIFTS],
    i32,
    i32,
    i32,
    i32,
    usize,
    usize,
);

/// Per-size SAD fn type (P1-B): `(src, src_stride, ref, ref_stride) -> SAD`.
/// Mirrors the scalar `crate::dsp::distortion::vpx_sad{W}x{H}` shims and
/// the AVX2 `crate::dsp::x86_64::vpx_sad{W}x{H}_avx2` wrappers — both share
/// this signature, so a slot may hold either with no caller-visible change.
pub type SadFn = fn(&[u8], usize, &[u8], usize) -> u32;

/// 13-entry SAD dispatch table, one fn-pointer per VP9 block size, indexed
/// by `BlockSize::idx()` (`0..13`). Mirrors libvpx's `cpi->fn_ptr[bsize].sdf`
/// indirection. Each entry is selected by one `if caps.avx2` clause in
/// [`SimdDispatch::build`] — the AVX2 wrapper on AVX2 hosts, the scalar
/// reference otherwise. Byte-exact across the two arms (SAD is an
/// integer-exact `Σ|a − b|`).
#[derive(Debug, Clone)]
pub struct SadTable {
    /// Indexed by `BlockSize::idx()` (`0..=12`).
    fns: [SadFn; crate::block::BLOCK_SIZES],
}

impl SadTable {
    /// Resolve the SAD kernel for `bsize`. Total over all 13 VP9 sizes.
    #[inline]
    #[must_use]
    pub fn get(&self, bsize: crate::block::BlockSize) -> SadFn {
        self.fns[bsize.idx()]
    }
}

/// 4-reference SAD fn type (P1-B / PR3):
/// `(src, src_stride, &refs[4], ref_stride, &mut out[4])`. Writes
/// `out[k] = Σ|src − refs[k]|` for `k ∈ 0..4`. The scalar and architecture
/// wrappers share this signature, so a slot may hold any ISA implementation
/// with no caller-visible change. Each output lane is byte-exact with the
/// corresponding single [`SadFn`] result.
pub type Sad4dFn = fn(&[u8], usize, &[&[u8]; 4], usize, &mut [u32; 4]);

/// 13-entry 4-reference SAD dispatch table, one fn-pointer per VP9 block
/// size, indexed by `BlockSize::idx()` (`0..13`). Mirrors libvpx's
/// `cpi->fn_ptr[bsize].sdx4df` indirection. Used by the diamond search's
/// `all_in` fast path (`crate::mv_search::full_pixel`) to evaluate the four
/// axis-aligned candidate sites of a step in one call. Each entry is selected
/// by ISA capability in [`SimdDispatch::build`] — AVX2 on x86-64, dot-product
/// or base NEON on AArch64, and the scalar four-`sad` reference otherwise.
/// Every arm is byte-exact with the corresponding single-SAD slot.
#[derive(Debug, Clone)]
pub struct Sad4dTable {
    /// Indexed by `BlockSize::idx()` (`0..=12`).
    fns: [Sad4dFn; crate::block::BLOCK_SIZES],
}

impl Sad4dTable {
    /// Resolve the 4-reference SAD kernel for `bsize`. Total over all 13 VP9
    /// sizes.
    #[inline]
    #[must_use]
    pub fn get(&self, bsize: crate::block::BlockSize) -> Sad4dFn {
        self.fns[bsize.idx()]
    }
}

/// Per-size block `(sum, sse)` reduction fn type (PR4):
/// `(src, src_stride, pred, pred_stride) -> (sum, sse)`. Returns the
/// integer-exact `sum = Σ(src − pred)` (signed `i64`) and `sse = Σ(src − pred)²`
/// (`u64`) — the reduction the RD-model variance finishers consume. Mirrors the
/// scalar `crate::dsp::distortion::var_sum_sse_{W}x{H}` shims and the SIMD
/// wrappers in `crate::dsp::{x86_64,aarch64}` — all share this
/// signature, so a slot may hold either with no caller-visible change. The
/// three scalar finishers in `crate::mode_decision::model_rd` stay untouched;
/// byte-identity reduces to "this reduction is integer-exact", which holds by
/// construction.
pub type VarFn = fn(&[u8], usize, &[u8], usize) -> (i64, u64);

/// Per-8×8-cell `(sum, sse)` reduction fn type (PR4):
/// `(src, src_stride, ref, ref_stride) -> (sum, sse)` over a fixed 8×8 cell.
/// Parallel to [`VarFn`] but with the narrower `(i32, u32)` accumulator the
/// large-block RD model's per-cell finisher consumes. The AVX2 wrapper and the
/// scalar reference share this signature, so a slot may hold either with no
/// caller-visible change.
pub type CellVarFn = fn(&[u8], usize, &[u8], usize) -> (i32, u32);

/// 13-entry block `(sum, sse)` reduction table, one fn-pointer per VP9 block
/// size, indexed by `BlockSize::idx()` (`0..13`). Feeds the inter RD model's
/// `compute_variance` (`crate::mode_decision::model_rd`), whose scalar
/// `saturating_sub` finisher consumes the reduction unchanged. Each entry is
/// selected by ISA capability in [`SimdDispatch::build`] — AVX2 on x86_64,
/// dot-product or base NEON for widths 8–64 on AArch64, and scalar otherwise.
/// All arms are byte-exact (integer-exact reduction, no overflow for any VP9
/// size).
#[derive(Debug, Clone)]
pub struct VarTable {
    /// Indexed by `BlockSize::idx()` (`0..=12`).
    fns: [VarFn; crate::block::BLOCK_SIZES],
}

impl VarTable {
    /// Resolve the block `(sum, sse)` reduction kernel for `bsize`. Total over
    /// all 13 VP9 sizes.
    #[inline]
    #[must_use]
    pub fn get(&self, bsize: crate::block::BlockSize) -> VarFn {
        self.fns[bsize.idx()]
    }
}

/// Per-size sub-pixel-variance fn type (PR4):
/// `(src, src_stride, xoffset, yoffset, ref, ref_stride) -> (var, sse)`.
/// Mirrors the scalar `crate::dsp::distortion::vpx_sub_pixel_variance{W}x{H}`
/// shims, AVX2 `crate::dsp::x86_64::vpx_sub_pixel_variance{W}x{H}_avx2`
/// wrappers, and AArch64 NEON wrappers. All share this signature (identical
/// to `crate::mv_search::sub_pixel::SvfFn`) and return byte-exact results.
pub type SubpelVarFn = fn(&[u8], usize, usize, usize, &[u8], usize) -> (u32, u32);

/// Per-size sub-pixel-variance dispatch table, indexed by `BlockSize::idx()`
/// (`0..13`). Only the 11 sizes the sub-pel search emits (5 square + 6
/// rectangular) carry a kernel; the two sub-8 rect sizes (4×8, 8×4) hold
/// `None`, preserving the fail-soft `svf_for` semantics
/// (`crate::mv_search::sub_pixel`). Each populated entry is selected by one
/// ISA clause in [`SimdDispatch::build`] — AVX2 on x86_64, NEON for widths
/// 16–64 on AArch64, and the scalar reference otherwise. Byte-exact across
/// all arms.
#[derive(Debug, Clone)]
pub struct SubpelVarTable {
    /// Indexed by `BlockSize::idx()` (`0..=12`); `None` for unsupported sizes.
    fns: [Option<SubpelVarFn>; crate::block::BLOCK_SIZES],
}

impl SubpelVarTable {
    /// Resolve the sub-pixel-variance kernel for `bsize`, or `None` for the
    /// two unsupported sub-8 rect sizes (4×8, 8×4).
    #[inline]
    #[must_use]
    pub fn get(&self, bsize: crate::block::BlockSize) -> Option<SubpelVarFn> {
        self.fns[bsize.idx()]
    }
}

/// The runtime SIMD kernel dispatch table.
///
/// Populated once per process via [`init`] (production path). In-crate
/// tests may also use the `pub(crate)` [`SimdDispatch::build`]
/// constructor with caps tweaked to force a particular ISA tier;
/// external callers reach `SimdDispatch` exclusively through
/// [`init`] / [`get`].
///
/// Slot table (grown one family at a time per Stage 3 / Wave C G4):
///
/// | Wave  | Family            | Slot                       | AVX2 impl |
/// |-------|-------------------|----------------------------|-----------|
/// | G4.1  | inverse 8×8       | `idct8x8_add`              | yes       |
/// | G4.2  | inverse 4×4       | `idct4x4_add`              | yes       |
/// | G4.2  | inverse 16×16     | `idct16x16_add`            | yes       |
/// | G4-α  | inverse 32×32     | `idct32x32_add`            | yes       |
/// | F3    | inverse 16×16 EOB≤10 | `idct16x16_10_add`      | yes       |
/// | F4    | inverse 16×16 EOB≤38 | `idct16x16_38_add`      | yes       |
/// | F9    | inverse 32×32 eob≤34  | `idct32x32_34_add`     | yes (finisher-only) |
/// | F9    | inverse 32×32 eob≤135 | `idct32x32_135_add`    | yes (finisher-only) |
/// | G4.3a | intra-pred DC 4×4 | `dc_predictor_4x4`         | yes       |
/// | I1    | intra-pred V (size-parametric) | `v_predictor`     | yes (Wave I I1) |
/// | G4-β  | hybrid ADST 4×4   | `iht4x4_add`               | yes       |
/// | G4-β  | hybrid ADST 8×8   | `iht8x8_add`               | yes       |
/// | G4.3b / F10 | loopfilter v4 | `lpf_vertical_4`           | yes (Wave F G F10) — H1 gated OFF by default |
/// | G3    | loopfilter h4     | `lpf_horizontal_4`         | yes (Wave G G3) — H1 gated OFF by default |
/// | G4    | loopfilter v8     | `lpf_vertical_8`           | yes (Wave G G4) — H1 gated OFF by default |
/// | G5    | loopfilter h8     | `lpf_horizontal_8`         | yes (Wave G G5) — H1 gated OFF by default |
/// | G6    | loopfilter v16    | `lpf_vertical_16`          | yes (Wave G G6) — H1 gated OFF by default |
/// | G6    | loopfilter h16    | `lpf_horizontal_16`        | yes (Wave G G6) — H1 gated OFF by default |
/// | H2    | loopfilter v4_dual | `lpf_vertical_4_dual`     | yes (Wave H H2) — two-singles delegation |
/// | H2    | loopfilter h4_dual | `lpf_horizontal_4_dual`   | yes (Wave H H2) — two-singles delegation |
/// | H3    | loopfilter v8_dual | `lpf_vertical_8_dual`     | yes (Wave H H3) — two-singles delegation |
/// | H3    | loopfilter h8_dual | `lpf_horizontal_8_dual`   | yes (Wave H H3) — two-singles delegation |
/// | H4    | loopfilter v16_dual| `lpf_vertical_16_dual`    | yes (Wave H H4, two-singles) |
/// | H4    | loopfilter h16_dual| `lpf_horizontal_16_dual`  | yes (Wave H H4, two-singles) |
/// | H6    | quant dequant     | `plane_dequant_finisher`   | yes (Wave H H6) — pass 2 of split detokenize |
/// | G4.4b | frame-buffers     | `extend_frame_borders`     | scalar only (memcpy/memset already vectorised) |
///
/// **Family coverage** (per HLD §G4 table line 543-552):
/// 1. frame-buffers — `extend_frame_borders` slot present (G4.4b).
/// 2. entropy decoder — n/a (scalar by design, BoolReader is sequential).
/// 3. transforms inverse — `idct4x4_add` (G4.2) + `idct8x8_add` (G4.1) + `idct16x16_add` (Wave D G4.2 full-kernel rollout) + `idct32x32_add` (Wave E G4-α) + `iht4x4_add` + `iht8x8_add` (Wave E G4-β) all have AVX2 impls.
/// 4. quant dequant — `plane_dequant_finisher` slot present (H6, AVX2-capable).
/// 5. intra-pred — `dc_predictor_4x4` has AVX2 impl (G4.3a).
/// 6. loopfilter — `lpf_vertical_4` (Wave F G F10), `lpf_horizontal_4`
///    (Wave G G3), `lpf_vertical_8` (Wave G G4), `lpf_horizontal_8`
///    (Wave G G5), `lpf_vertical_16` (Wave G G6), and
///    `lpf_horizontal_16` (Wave G G6) — 6 loopfilter AVX2 slots.
///
/// 6/6 families have a slot; 6/6 carry a non-scalar Tier-1 impl
/// (idct4x4, idct8x8, idct16x16, idct32x32, iht4x4, iht8x8,
/// dc_predictor_4x4, lpf_vertical_4, lpf_horizontal_4,
/// lpf_vertical_8, lpf_horizontal_8, lpf_vertical_16,
/// lpf_horizontal_16, plane_dequant_finisher). The deferred slot
/// (extend_frame_borders) reserves naming + dispatch shape so future
/// AVX2 work can fill it without changing call sites. See per-field
/// docstrings for the per-family deferral rationale.
///
/// Fields are `pub(crate)` to keep slot additions a crate-internal
/// concern; external dispatch sites call the per-kernel accessor
/// (e.g. [`SimdDispatch::idct8x8_add`]) and don't see the field set.
#[derive(Debug, Clone)]
pub struct SimdDispatch {
    /// The capability set the table was built against. Persisted for
    /// debug / instrumentation; the dispatch slots below are already
    /// specialised to this caps state at [`SimdDispatch::build`] time.
    pub caps: Caps,

    /// G4.1 — 8×8 inverse DCT + residual add, full-density (eob > 1).
    /// Signature mirrors the scalar reference
    /// [`crate::transforms::inverse::idct8x8_64_add`]. On x86_64 AVX2
    /// CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::idct8x8_add_avx2`]; otherwise it points
    /// at the scalar reference.
    pub(crate) idct8x8_add: fn(input: &[i32; 64], dst: &mut [u8], stride: usize),

    /// Task A — 8×8 inverse DCT + residual add for sparse blocks with
    /// `eob <= 12` (top-left 4×4 sub-block non-zero). Signature mirrors
    /// the scalar reference
    /// [`crate::transforms::inverse::idct8x8_12_add`]. This selects the
    /// safe wrappers [`crate::dsp::x86_64::idct8x8_add_sse2_12`] on
    /// x86_64 SSE2 and [`crate::dsp::aarch64::idct8x8_12_add_neon`] on
    /// AArch64 NEON; otherwise it points at the scalar reference. The kernel
    /// exploits the eob-12 contract
    /// (rows 4..7 of `input` are zero) and runs the row pass on 4 rows
    /// instead of 8.
    pub(crate) idct8x8_12_add: fn(input: &[i32; 64], dst: &mut [u8], stride: usize),

    /// G4.2 — 4×4 inverse DCT + residual add. Signature mirrors the
    /// scalar reference [`crate::transforms::inverse::idct4x4_16_add`].
    /// On x86_64 AVX2 CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::idct4x4_add_avx2`]; otherwise it points
    /// at the scalar reference.
    pub(crate) idct4x4_add: fn(input: &[i32; 16], dst: &mut [u8], stride: usize),

    /// Track T phase 0 — HBD 4×4 inverse DCT + residual add (scalar
    /// reference only; no SIMD HBD ports in this track).
    ///
    /// Signature mirrors the scalar reference
    /// [`crate::transforms::inverse::highbd_idct4x4_16_add`]. The
    /// extra `u32` parameter is the bit depth (`bd ∈ {8, 10, 12}`);
    /// callers pass `bd = h.bit_depth as u32` once the bit-depth gate
    /// in the decoder lifts (T-1 phase 1+). Until then this slot is
    /// reachable from unit tests only — no decoder call sites in
    /// phase 0.
    pub(crate) highbd_idct4x4_add: fn(input: &[i32; 16], dst: &mut [u16], stride: usize, bd: u32),

    /// G4.2 — 16×16 inverse DCT + residual add (full-density 256
    /// variant). Signature mirrors the scalar reference
    /// [`crate::transforms::inverse::idct16x16_256_add`]. On x86_64
    /// AVX2 CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::idct16x16_add_avx2`]; otherwise it points
    /// at the scalar reference. The 10/38-bucket fast paths are not
    /// yet ported — only the full `_256_add` variant is dispatched.
    pub(crate) idct16x16_add: fn(input: &[i32; 256], dst: &mut [u8], stride: usize),

    /// G4-α — 32×32 inverse DCT + residual add (full-density 1024
    /// variant). Signature mirrors the scalar reference
    /// [`crate::transforms::inverse::idct32x32_1024_add`]. On x86_64
    /// AVX2 CPUs this points at the i16-lane safe wrapper
    /// [`crate::dsp::x86_64::idct32x32_add_avx2_1024`]; on SSE2-only
    /// CPUs at [`crate::dsp::x86_64::idct32x32_add_sse2_1024`];
    /// otherwise at the scalar reference. The 34/135-bucket fast paths
    /// land in [`SimdDispatch::idct32x32_34_add`] /
    /// [`SimdDispatch::idct32x32_135_add`].
    pub(crate) idct32x32_add: fn(input: &[i32; 1024], dst: &mut [u8], stride: usize),

    /// F9 — 32×32 inverse DCT + add for sparse blocks with
    /// `eob <= 34` (top-left 8×8 sub-block non-zero). Signature
    /// mirrors the scalar reference
    /// [`crate::transforms::inverse::idct32x32_34_add`]. On x86_64
    /// AVX2 CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::idct32x32_34_add_avx2`]; otherwise it
    /// points at the scalar reference. Same finisher as the
    /// `idct32x32_add` AVX2 kernel — only the row count differs
    /// (Stage 1 runs for the first 8 rows only). Caller must respect
    /// the eob contract.
    pub(crate) idct32x32_34_add: fn(input: &[i32; 1024], dst: &mut [u8], stride: usize),

    /// F9 — 32×32 inverse DCT + add for sparse blocks with
    /// `eob <= 135` (top-left 16×16 sub-block non-zero). Signature
    /// mirrors the scalar reference
    /// [`crate::transforms::inverse::idct32x32_135_add`]. On x86_64
    /// AVX2 CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::idct32x32_135_add_avx2`]; otherwise it
    /// points at the scalar reference. Same finisher as the
    /// `idct32x32_add` AVX2 kernel — only the row count differs
    /// (Stage 1 runs for the first 16 rows only). Caller must respect
    /// the eob contract.
    pub(crate) idct32x32_135_add: fn(input: &[i32; 1024], dst: &mut [u8], stride: usize),

    /// Wave F F3 — 16×16 inverse DCT + add for sparse blocks with
    /// `eob <= 10` (top-left 4×4 sub-block non-zero). Signature
    /// mirrors the scalar reference
    /// [`crate::transforms::inverse::idct16x16_10_add`]. On x86_64 AVX2
    /// CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::idct16x16_10_add_avx2`] (finisher-only
    /// port: row pass reduced to 4 rows, Stage 3 reused from
    /// `idct16x16_add_avx2`); otherwise it points at the scalar
    /// reference.
    pub(crate) idct16x16_10_add: fn(input: &[i32; 256], dst: &mut [u8], stride: usize),

    /// Wave F F4 — 16×16 inverse DCT + add for sparse blocks with
    /// `eob <= 38` (top-left 8×8 sub-block non-zero). Signature
    /// mirrors the scalar reference
    /// [`crate::transforms::inverse::idct16x16_38_add`]. On x86_64 AVX2
    /// CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::idct16x16_38_add_avx2`] (finisher-only
    /// port: row pass reduced to 8 rows, Stage 3 reused from
    /// `idct16x16_add_avx2`); otherwise it points at the scalar
    /// reference.
    pub(crate) idct16x16_38_add: fn(input: &[i32; 256], dst: &mut [u8], stride: usize),

    /// G4-β — 4×4 hybrid inverse transform (ADST/DCT combinations).
    /// Signature mirrors [`crate::transforms::inverse::iht4x4_16_add`]
    /// — the `tx_type` arg picks the row/col 1D pair. On x86_64 AVX2
    /// CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::iht4x4_add_avx2`]; otherwise it points at
    /// the scalar reference.
    pub(crate) iht4x4_add: fn(input: &[i32; 16], dst: &mut [u8], stride: usize, tx_type: TxType),

    /// G4-β — 8×8 hybrid inverse transform. Signature mirrors
    /// [`crate::transforms::inverse::iht8x8_64_add`]. On x86_64 AVX2
    /// CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::iht8x8_add_avx2`]; otherwise it points at
    /// the scalar reference.
    pub(crate) iht8x8_add: fn(input: &[i32; 64], dst: &mut [u8], stride: usize, tx_type: TxType),

    /// Wave F F5 — 16×16 hybrid inverse transform (ADST/DCT
    /// combinations). Signature mirrors
    /// [`crate::transforms::inverse::iht16x16_256_add`] — the
    /// `tx_type` arg picks the row/col 1D pair. On x86_64 AVX2 CPUs
    /// this points at the safe wrapper
    /// [`crate::dsp::x86_64::iht16x16_add_avx2`]; otherwise it points
    /// at the scalar reference.
    pub(crate) iht16x16_add: fn(input: &[i32; 256], dst: &mut [u8], stride: usize, tx_type: TxType),

    /// G4.3a — DC intra-predictor at `bs = 4`. Signature is the
    /// flat-arg form `(dst, stride, above, left)` (compare:
    /// [`crate::intra_pred::dc_predictor`] takes a `PredCtx` which is
    /// indirect through `above_left` — unused by DC).
    /// On x86_64 AVX2 CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::dc_predictor_4x4_avx2`]; otherwise it
    /// points at a thin scalar adapter that re-builds a `PredCtx`.
    pub(crate) dc_predictor_4x4: fn(dst: &mut [u8], stride: usize, above: &[u8], left: &[u8]),

    /// Wave I I1 — size-parametric V_PRED intra-predictor. Signature
    /// is the flat-arg form `(dst, stride, bs, above)` (compare:
    /// [`crate::intra_pred::v_predictor`] takes a `PredCtx`; V_PRED
    /// reads only `ctx.above`). `bs ∈ {4, 8, 16, 32}`; the wrapper
    /// dispatches internally on `bs` to the matching wide-store width
    /// (4 / 8 / 16 / 32 bytes). On x86_64 AVX2 CPUs this points at
    /// the safe wrapper [`crate::dsp::x86_64::v_predictor_avx2`];
    /// otherwise it points at a thin scalar adapter that re-builds a
    /// `PredCtx`.
    pub(crate) v_predictor: fn(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]),

    /// Wave I I2 — size-parametric H_PRED intra-predictor. Signature
    /// is the flat-arg form `(dst, stride, bs, left)` (compare:
    /// [`crate::intra_pred::h_predictor`] takes a `PredCtx`; H_PRED
    /// reads only `ctx.left`). `bs ∈ {4, 8, 16, 32}`; the wrapper
    /// dispatches internally on `bs` to the matching wide-store width
    /// (4 / 8 / 16 / 32 bytes). On x86_64 AVX2 CPUs this points at
    /// the safe wrapper [`crate::dsp::x86_64::h_predictor_avx2`];
    /// otherwise it points at a thin scalar adapter that re-builds a
    /// `PredCtx`.
    pub(crate) h_predictor: fn(dst: &mut [u8], stride: usize, bs: usize, left: &[u8]),

    /// Wave I I3 — TM_PRED ("true motion") intra-predictor. Computes
    /// per-pixel `clip(left[r] + above[c] - above_left, 0, 255)` over
    /// the `bs × bs` block. Signature is the flat-arg form
    /// `(dst, stride, bs, above, left, above_left)`; compare
    /// [`crate::intra_pred::tm_predictor`] which takes a `PredCtx`.
    /// On x86_64 AVX2 CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::tm_predictor_avx2`]; otherwise it points
    /// at a thin scalar adapter that re-builds a `PredCtx`. The kernel
    /// is size-parametric on `bs ∈ {4, 8, 16, 32}`.
    pub(crate) tm_predictor: TmPredictorFn,

    /// Wave K K1 — D45_PRED directional intra-predictor (`above`-only).
    /// Flat-arg form `(dst, stride, bs, above)`; the slot fn internally
    /// routes the `bs == 4` libvpx specialization. On x86_64 AVX2 CPUs
    /// points at [`crate::dsp::x86_64::d45_predictor_avx2`]; otherwise a
    /// scalar adapter ([`scalar_d45_predictor`]).
    pub(crate) d45_predictor: fn(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]),

    /// Wave K K3 — D63_PRED directional intra-predictor (`above`-only).
    /// See [`SimdDispatch::d45_predictor`]; routes `bs == 4` internally.
    pub(crate) d63_predictor: fn(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]),

    /// Wave K K2 — D207_PRED directional intra-predictor (`left`-only).
    /// On AVX2 CPUs points at [`crate::dsp::x86_64::d207_predictor_avx2`].
    pub(crate) d207_predictor: fn(dst: &mut [u8], stride: usize, bs: usize, left: &[u8]),

    /// Wave K K5 — D117_PRED directional intra-predictor. Flat-arg form
    /// `(dst, stride, bs, above, left, above_left)`. Generic for all
    /// sizes (no 4×4 special). On AVX2 CPUs points at
    /// [`crate::dsp::x86_64::d117_predictor_avx2`].
    pub(crate) d117_predictor: DirectionalCtxPredFn,

    /// Wave K K4 — D135_PRED directional intra-predictor. See
    /// [`SimdDispatch::d117_predictor`].
    pub(crate) d135_predictor: DirectionalCtxPredFn,

    /// Wave K K6 — D153_PRED directional intra-predictor. See
    /// [`SimdDispatch::d117_predictor`].
    pub(crate) d153_predictor: DirectionalCtxPredFn,

    /// Wave I I4 — size-parametric DC_PRED predictor (full neighbours
    /// available). Covers the `(above_available, left_available) =
    /// (true, true)` arm of `dc_predict_with_availability`. Signature
    /// is the flat-arg form `(dst, stride, bs, above, left)` (cf.
    /// [`crate::intra_pred::dc_predictor`] which takes a `PredCtx`).
    /// `bs ∈ {4, 8, 16, 32}`; the AVX2 wrapper dispatches internally
    /// on `bs` to the matching SAD-sum + wide-store width. On x86_64
    /// AVX2 CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::dc_predict_full_avx2`]; otherwise it
    /// points at a thin scalar adapter that re-builds a `PredCtx`.
    ///
    /// **NOTE:** the legacy `dc_predictor_4x4` slot still handles the
    /// `(bs = 4, both available)` fast path; this `dc_predict_full`
    /// slot covers the wider sizes. The router in
    /// `crate::intra_pred::dc_predict_with_availability` picks between
    /// the two based on `bs`.
    pub(crate) dc_predict_full:
        fn(dst: &mut [u8], stride: usize, bs: usize, above: &[u8], left: &[u8]),

    /// Wave I I4 — size-parametric DC_PRED predictor (above-only).
    /// Covers the `(true, false)` availability arm. Signature is the
    /// flat-arg form `(dst, stride, bs, above)` — matches
    /// [`crate::intra_pred::dc_top_predictor`] verbatim. On x86_64
    /// AVX2 CPUs this points at
    /// [`crate::dsp::x86_64::dc_predict_top_avx2`]; otherwise at the
    /// scalar reference (no PredCtx wrapping needed).
    pub(crate) dc_predict_top: fn(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]),

    /// Wave I I4 — size-parametric DC_PRED predictor (left-only).
    /// Covers the `(false, true)` availability arm. Signature matches
    /// [`crate::intra_pred::dc_left_predictor`]. On x86_64 AVX2 CPUs
    /// this points at [`crate::dsp::x86_64::dc_predict_left_avx2`];
    /// otherwise at the scalar reference.
    pub(crate) dc_predict_left: fn(dst: &mut [u8], stride: usize, bs: usize, left: &[u8]),

    /// Wave I I4 — size-parametric DC_128 fallback predictor. Covers
    /// the `(false, false)` availability arm — neither neighbour
    /// available, fill block with 128. Signature matches
    /// [`crate::intra_pred::dc_128_predictor`]. On x86_64 AVX2 CPUs
    /// this points at [`crate::dsp::x86_64::dc_predict_128_avx2`];
    /// otherwise at the scalar reference.
    pub(crate) dc_predict_128: fn(dst: &mut [u8], stride: usize, bs: usize),

    /// G4.3b / Wave F G F10 — vertical 4-tap loopfilter edge kernel.
    /// Signature mirrors [`crate::loopfilter::lpf_vertical_4`]. On
    /// x86_64 AVX2 CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::lpf_vertical_4_avx2`] (transpose 8×8
    /// rows of 8 bytes into per-column vectors, run `filter4`
    /// lane-parallel across the 8 rows, transpose the inner 4 result
    /// columns back into row-major form and store); otherwise it
    /// points at the scalar reference.
    pub(crate) lpf_vertical_4:
        fn(pixels: &mut [u8], s_index: usize, pitch: usize, blimit: u8, limit: u8, thresh: u8),

    /// Wave G G3 — horizontal 4-tap loopfilter edge kernel.
    /// Signature mirrors [`crate::loopfilter::lpf_horizontal_4`]. On
    /// x86_64 AVX2 CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::lpf_horizontal_4_avx2`] (each tap is a
    /// contiguous 8-byte row, so 8 columns naturally line up as the
    /// 8 low bytes of each `__m128i` — `filter4` runs lane-parallel
    /// across the 8 columns without any transpose); otherwise it
    /// points at the scalar reference.
    pub(crate) lpf_horizontal_4:
        fn(pixels: &mut [u8], s_index: usize, pitch: usize, blimit: u8, limit: u8, thresh: u8),

    /// Wave G G4 — 8-tap vertical loopfilter edge kernel. Signature
    /// mirrors [`crate::loopfilter::lpf_vertical_8`]. On x86_64 AVX2
    /// CPUs this points at
    /// [`crate::dsp::x86_64::lpf_vertical_8_avx2`] (F10 transpose +
    /// new `filter8` + `flat_mask4` lane-parallel inner); otherwise
    /// at the scalar reference.
    pub(crate) lpf_vertical_8:
        fn(pixels: &mut [u8], s_index: usize, pitch: usize, blimit: u8, limit: u8, thresh: u8),

    /// Wave G G5 — 8-tap horizontal loopfilter edge kernel. Signature
    /// mirrors [`crate::loopfilter::lpf_horizontal_8`]. On x86_64 AVX2
    /// CPUs this points at
    /// [`crate::dsp::x86_64::lpf_horizontal_8_avx2`] (no-transpose
    /// row-load + same `filter8` + `flat_mask4` inner as G4);
    /// otherwise at the scalar reference.
    pub(crate) lpf_horizontal_8:
        fn(pixels: &mut [u8], s_index: usize, pitch: usize, blimit: u8, limit: u8, thresh: u8),

    /// Wave G G6 — 16-tap vertical "edge" loopfilter (macroblock
    /// boundary). Signature mirrors [`crate::loopfilter::lpf_vertical_16`].
    /// On x86_64 AVX2 CPUs this points at
    /// [`crate::dsp::x86_64::lpf_vertical_16_avx2`] (two side-by-side
    /// 8×8 transposes + filter16/filter8/filter4 cascade gated by
    /// `flat_mask4` and `flat_mask5`); otherwise at the scalar
    /// reference.
    pub(crate) lpf_vertical_16:
        fn(pixels: &mut [u8], s_index: usize, pitch: usize, blimit: u8, limit: u8, thresh: u8),

    /// Wave G G6 — 16-tap horizontal "edge" loopfilter (macroblock
    /// boundary). Signature mirrors
    /// [`crate::loopfilter::lpf_horizontal_16`]. On x86_64 AVX2 CPUs
    /// this points at [`crate::dsp::x86_64::lpf_horizontal_16_avx2`]
    /// (no-transpose row-load + same filter16 cascade as the G6
    /// vertical sibling); otherwise at the scalar reference.
    pub(crate) lpf_horizontal_16:
        fn(pixels: &mut [u8], s_index: usize, pitch: usize, blimit: u8, limit: u8, thresh: u8),

    /// Wave H H2 — dual 4-tap vertical loopfilter. Runs `filter4`
    /// over two adjacent 8-row × 8-col groups (16 rows total) with
    /// independent threshold sets `(blimit0,limit0,thresh0)` vs
    /// `(blimit1,limit1,thresh1)`. Mirrors
    /// [`crate::loopfilter::lpf_vertical_4_dual`]. On x86_64 AVX2
    /// CPUs this points at
    /// [`crate::dsp::x86_64::lpf_vertical_4_dual_avx2`] (two-singles
    /// delegation through the F10 single AVX2 wrapper).
    pub(crate) lpf_vertical_4_dual: LpfDualFn,

    /// Wave H H2 — dual 4-tap horizontal loopfilter. Lane-doubled
    /// counterpart of [`SimdDispatch::lpf_horizontal_4`]. On x86_64
    /// AVX2 CPUs this points at
    /// [`crate::dsp::x86_64::lpf_horizontal_4_dual_avx2`] (two-singles
    /// delegation through the G3 single AVX2 wrapper).
    pub(crate) lpf_horizontal_4_dual: LpfDualFn,

    /// Wave H H3 — dual 8-tap vertical loopfilter. Lane-doubled
    /// counterpart of [`SimdDispatch::lpf_vertical_8`]. On x86_64
    /// AVX2 CPUs this points at
    /// [`crate::dsp::x86_64::lpf_vertical_8_dual_avx2`] (two-singles
    /// delegation through the G4 single AVX2 wrapper).
    pub(crate) lpf_vertical_8_dual: LpfDualFn,

    /// Wave H H3 — dual 8-tap horizontal loopfilter. Lane-doubled
    /// counterpart of [`SimdDispatch::lpf_horizontal_8`]. On x86_64
    /// AVX2 CPUs this points at
    /// [`crate::dsp::x86_64::lpf_horizontal_8_dual_avx2`] (two-singles
    /// delegation through the G5 single AVX2 wrapper).
    pub(crate) lpf_horizontal_8_dual: LpfDualFn,

    /// Wave H H1 — dual 16-tap vertical loopfilter. Six-arg (single
    /// threshold set shared across both halves per libvpx
    /// `vpx_lpf_vertical_16_dual_c` convention; cf. 4/8-tap duals
    /// which take independent threshold sets). Scalar-only at H1
    /// close; H4 lands the AVX2 promotion (lane-doubling of G6).
    pub(crate) lpf_vertical_16_dual:
        fn(pixels: &mut [u8], s_index: usize, pitch: usize, blimit: u8, limit: u8, thresh: u8),

    /// Wave H H1 — dual 16-tap horizontal loopfilter. Six-arg — see
    /// [`SimdDispatch::lpf_vertical_16_dual`] field doc for the
    /// shared-threshold rationale. Scalar-only at H1 close; H4 lands
    /// the AVX2 promotion.
    pub(crate) lpf_horizontal_16_dual:
        fn(pixels: &mut [u8], s_index: usize, pitch: usize, blimit: u8, limit: u8, thresh: u8),

    /// Wave H H6 — pass 2 of the split detokenize loop. Mirrors
    /// [`crate::quant::plane_dequant_finisher`]. The bool-coder pass
    /// (pass 1) deposits four parallel scratch arrays
    /// `(raster, dqv, val, sign)` per non-zero coefficient; this slot
    /// consumes them, computing
    /// `mag = (val * dqv) >> dq_shift; out[raster] = sign ? -mag : mag`
    /// in a vectorisable form (no loop-carried state, no bool-coder
    /// reads). On x86_64 AVX2 CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::plane_dequant_finisher_avx2`] (8 i32
    /// lanes per iteration, scalar tail for `< 8` entries); otherwise
    /// it points at the scalar reference.
    pub(crate) plane_dequant_finisher: PlaneDequantFinisherFn,

    /// Wave I I5 — integer-pel copy: copy a `w × h` block from `src`
    /// to `dst`. Mirrors [`crate::inter::convolve_copy`] (libvpx
    /// `vpx_convolve_copy_c`). Wave J J1 promotes this slot to AVX2 on
    /// x86_64 hosts that report `caps.avx2`; AArch64 hosts use pinned
    /// libvpx's equivalent fixed-width NEON copy when `caps.neon` is set.
    pub(crate) convolve_copy:
        fn(src: &[u8], src_stride: usize, dst: &mut [u8], dst_stride: usize, w: usize, h: usize),

    /// Wave I I5 — integer-pel average: `dst = (dst + src + 1) >> 1`.
    /// Mirrors [`crate::inter::convolve_avg`] (libvpx
    /// `vpx_convolve_avg_c`). Wave J J2 promotes this slot to AVX2 on
    /// x86_64 hosts that report `caps.avx2` (a single `_mm256_avg_epu8`
    /// per chunk); non-AVX2 hosts keep the scalar reference.
    pub(crate) convolve_avg:
        fn(src: &[u8], src_stride: usize, dst: &mut [u8], dst_stride: usize, w: usize, h: usize),

    /// Wave I I5 — 8-tap horizontal subpel convolution. Mirrors
    /// [`crate::inter::convolve8_horiz`]. Wave J J3 promotes this slot
    /// to AVX2 on x86_64 hosts that report `caps.avx2`; non-AVX2 hosts
    /// keep the scalar reference.
    pub(crate) convolve8_horiz: Convolve8OneAxisFn,

    /// Wave I I5 — 8-tap vertical subpel convolution. Mirrors
    /// [`crate::inter::convolve8_vert`]. Wave J J4 promotes this slot to
    /// AVX2 on x86_64 hosts that report `caps.avx2`; non-AVX2 hosts keep
    /// the scalar reference.
    pub(crate) convolve8_vert: Convolve8OneAxisFn,

    /// Wave I I5 — 8-tap 2-D subpel convolution. Mirrors
    /// [`crate::inter::convolve8`]. Wave J J5 promotes this slot to AVX2
    /// on x86_64 hosts that report `caps.avx2`; non-AVX2 hosts keep the
    /// scalar reference.
    pub(crate) convolve8: Convolve8TwoAxisFn,

    /// Wave I I5 — 8-tap horizontal subpel convolution with compound
    /// average. Mirrors [`crate::inter::convolve8_avg_horiz`]. Wave J J6
    /// promotes this slot to AVX2 on x86_64 when `caps.avx2`.
    pub(crate) convolve8_avg_horiz: Convolve8OneAxisFn,

    /// Wave I I5 — 8-tap vertical subpel convolution with compound
    /// average. Mirrors [`crate::inter::convolve8_avg_vert`]. Wave J J7
    /// promotes this slot to AVX2 on x86_64 when `caps.avx2`.
    pub(crate) convolve8_avg_vert: Convolve8OneAxisFn,

    /// Wave I I5 — 8-tap 2-D subpel convolution with compound average.
    /// Mirrors [`crate::inter::convolve8_avg`]. Wave J J8 promotes this
    /// slot to AVX2 on x86_64 when `caps.avx2`.
    pub(crate) convolve8_avg: Convolve8TwoAxisFn,

    /// Wave L L1 — scaled 2-D sub-pel convolution (reference frame a
    /// different size: resize / spatial SVC). Mirrors
    /// [`crate::inter::filter::convolve8_scaled`]. AVX2 on x86_64 hosts
    /// reporting `caps.avx2`; scalar elsewhere.
    pub(crate) convolve8_scaled: ScaledConvolveFn,

    /// Wave L L1 — scaled 2-D sub-pel convolution with compound average.
    /// Mirrors [`crate::inter::filter::convolve8_scaled_avg`]. AVX2 on
    /// x86_64 when `caps.avx2`; scalar elsewhere.
    pub(crate) convolve8_scaled_avg: ScaledConvolveFn,

    /// P1-B — per-size SAD (`Σ|a − b|`) for the full-pel motion search.
    /// Mirrors the scalar `crate::dsp::distortion::vpx_sad{W}x{H}` shims;
    /// on x86_64 AVX2 hosts each entry points at the safe wrapper
    /// `crate::dsp::x86_64::vpx_sad{W}x{H}_avx2`, otherwise at the scalar
    /// reference. Byte-exact across the two arms (integer-exact reduction,
    /// no overflow in the u32 accumulator for any VP9 size). Resolved via
    /// [`SimdDispatch::sad`]; consumed by the diamond search
    /// (`crate::mv_search::full_pixel`) and `crate::mv_search::int_pro_me`.
    pub(crate) sad: SadTable,

    /// PR3 — per-size 4-reference SAD (`sdx4df`) for the diamond search's
    /// `all_in` fast path. Mirrors the scalar
    /// `crate::dsp::distortion::vpx_sad{W}x{H}x4d` shims; on x86_64 AVX2 hosts
    /// each entry points at the safe wrapper
    /// `crate::dsp::x86_64::vpx_sad{W}x{H}x4d_avx2`, otherwise at the scalar
    /// four-`sad` reference. Byte-exact across the two arms, and each output
    /// lane byte-exact with the single-SAD [`SimdDispatch::sad`] slot.
    /// Resolved via [`SimdDispatch::sad4d`]; consumed by
    /// `crate::mv_search::full_pixel`.
    pub(crate) sad4d: Sad4dTable,

    /// PR4 — per-size block `(sum, sse)` reduction for the inter RD model.
    /// Mirrors the scalar `crate::dsp::distortion::var_sum_sse_{W}x{H}` shims;
    /// on x86_64 AVX2 hosts each entry points at the safe wrapper
    /// `crate::dsp::x86_64::var_sum_sse_{W}x{H}_avx2`, otherwise at the scalar
    /// reference. Byte-exact across the two arms (integer-exact reduction).
    /// Resolved via [`SimdDispatch::variance`]; consumed by
    /// `crate::mode_decision::model_rd::compute_variance`, whose scalar
    /// `saturating_sub` finisher stays untouched.
    pub(crate) variance: VarTable,

    /// PR4 — per-8×8-cell `(sum, sse)` reduction for the large-block RD model.
    /// Mirrors the scalar `crate::dsp::distortion::get_var_sse_8x8`; on x86_64
    /// AVX2 hosts points at `crate::dsp::x86_64::get_var_sse_8x8_avx2`,
    /// otherwise at the scalar reference. Byte-exact across the two arms.
    /// Resolved via [`SimdDispatch::get_var_sse_8x8`]; consumed per cell by
    /// `crate::mode_decision::model_rd::compute_variance_8x8`, whose scalar
    /// abs-diff `>>6` finisher (and the downstream `aggregate_to_tx_size`
    /// abs-diff `>>tx_pixels_log2` finisher) stay untouched.
    pub(crate) get_var_sse_8x8: CellVarFn,

    /// PR4 — per-size sub-pixel variance for the sub-pel MV search. Mirrors the
    /// scalar `crate::dsp::distortion::vpx_sub_pixel_variance{W}x{H}` shims; on
    /// x86_64 AVX2 hosts each populated entry points at the AVX2 wrapper; on
    /// AArch64 NEON hosts the 16/32/64-wide entries point at fused NEON
    /// filter-and-reduction wrappers. Other entries use the scalar reference.
    /// The two sub-8 rect sizes (4×8, 8×4) hold `None` (fail-soft, preserving
    /// `svf_for`). All implementations are byte-exact. Resolved via
    /// [`SimdDispatch::subpel_variance`]; consumed by
    /// `crate::mv_search::sub_pixel`.
    pub(crate) subpel_variance: SubpelVarTable,

    /// G4.4b — frame-border edge replication. Signature mirrors
    /// [`crate::frame::extend_frame_borders`]. **Currently scalar-only**:
    /// the inner per-row work is `slice::fill` (left/right border
    /// columns) and `slice::copy_within` (top/bottom border rows),
    /// both of which the platform allocator's `memset` / `memmove`
    /// implementations already vectorise. Adding a manual AVX2 layer
    /// over those would not meaningfully change wall-clock perf; the
    /// slot exists to reserve the family name and dispatch shape.
    pub(crate) extend_frame_borders: fn(&mut Yv12Buffer),
}

impl SimdDispatch {
    /// Build a dispatch table for the given capability set. Used by
    /// [`init`] (production) and by tests that need a forced-caps
    /// scenario (e.g. `Caps { avx2: false, ..detect() }` to exercise
    /// the scalar fallback on an AVX2 CPU).
    ///
    /// One per-kernel selection clause runs per slot — currently just
    /// `idct8x8_add`. Adding a new slot adds one clause here.
    ///
    /// **Visibility:** `pub(crate)` — external callers MUST reach
    /// `SimdDispatch` via [`init`] / [`get`], which run caps detection
    /// internally. Allowing `pub` construction would let downstream
    /// callers fabricate `SimdDispatch::build(Caps { avx2: true, .. })`
    /// on a non-AVX2 host and execute AVX2 instructions there — UB.
    /// In-crate tests still need forced-caps construction (per the
    /// V4 design rationale in the module docstring) so the function
    /// stays crate-visible rather than fully private.
    #[must_use]
    pub(crate) fn build(caps: Caps) -> Self {
        // ---- G4.1 / G4.2 / G4.3a — kernels with a real AVX2 impl. ----
        // One per-kernel selection clause per slot; each clause picks
        // the AVX2 safe wrapper on supported CPUs and the scalar
        // reference otherwise.
        // i16-lane SSE2 dense 8×8 idct (libvpx `vpx_idct8x8_64_add_sse2`
        // port) — 2x throughput vs the i32-lane AVX2 path. SSE2 is
        // universal on x86_64, so the gate is always true; we keep the
        // `caps.sse2` check for parity with the other SIMD slots. The
        // AVX2 wrapper `idct8x8_add_avx2` is retained for EOB fast
        // paths and iht use.
        #[cfg(target_arch = "x86_64")]
        let idct8x8_add: fn(&[i32; 64], &mut [u8], usize) = if caps.sse2 {
            crate::dsp::x86_64::idct8x8_add_sse2_64
        } else {
            scalar_idct8x8_add
        };
        #[cfg(target_arch = "aarch64")]
        let idct8x8_add: fn(&[i32; 64], &mut [u8], usize) = if caps.neon {
            crate::dsp::aarch64::idct8x8_add_neon
        } else {
            scalar_idct8x8_add
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let idct8x8_add: fn(&[i32; 64], &mut [u8], usize) = scalar_idct8x8_add;

        // Task A — eob<=12 sparse 8×8 path. SSE2 (universal on x86_64)
        // skips rows 4..7 of the 1-D row pass per the bitstream's
        // eob-12 zero-tail contract; scalar elsewhere.
        #[cfg(target_arch = "x86_64")]
        let idct8x8_12_add: fn(&[i32; 64], &mut [u8], usize) = if caps.sse2 {
            crate::dsp::x86_64::idct8x8_add_sse2_12
        } else {
            crate::transforms::inverse::idct8x8_12_add
        };
        #[cfg(target_arch = "aarch64")]
        let idct8x8_12_add: fn(&[i32; 64], &mut [u8], usize) = if caps.neon {
            crate::dsp::aarch64::idct8x8_12_add_neon
        } else {
            crate::transforms::inverse::idct8x8_12_add
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let idct8x8_12_add: fn(&[i32; 64], &mut [u8], usize) = {
            let _ = caps;
            crate::transforms::inverse::idct8x8_12_add
        };

        #[cfg(target_arch = "x86_64")]
        let idct4x4_add: fn(&[i32; 16], &mut [u8], usize) = if caps.avx2 {
            crate::dsp::x86_64::idct4x4_add_avx2
        } else {
            scalar_idct4x4_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let idct4x4_add: fn(&[i32; 16], &mut [u8], usize) = scalar_idct4x4_add;

        // Track T phase 0 — HBD 4×4 inverse DCT scalar reference. No
        // SIMD HBD kernels in this track; the slot is populated with
        // the scalar reference unconditionally.
        let highbd_idct4x4_add: fn(&[i32; 16], &mut [u16], usize, u32) = scalar_highbd_idct4x4_add;

        // Track-T T-4 Slice 1.5 — `idct16x16_256_add` dispatch FLIPPED.
        //
        // The i16-lane SSE2 kernel (libvpx port `vpx_idct16x16_256_add_sse2`)
        // is now the primary path on all x86_64 hosts with at least SSE2.
        // The prior i32-lane AVX2 kernel was 2-4x slower per the libvpx
        // comparison — i32 lanes pack 8 elements per ymm vs i16's 16 per
        // ymm (or 8 per xmm), and the dense `_mm_madd_epi16` pipeline
        // matches libvpx's hot path 1:1. See
        // `wrk_docs/2026.05.24 - rvp9 vs libvpx decoder comparison.md`
        // Finding 1 + `crates/rvp9-core/src/dsp/x86_64/idct16x16_sse2_256.rs`
        // saturating-vs-modular note (i16 kernel uses `_mm_packs_epi32`
        // saturation; agrees with modular wraplow on every real VP9
        // bitstream and the 201-vector corpus — diverges only on
        // synthetic adversarial inputs > i16::MAX, same trade libvpx
        // makes).
        //
        // `idct16x16_add_avx2` is retained for the iht/EOB fast paths
        // that depend on its row pass (see `iht16x16_avx2.rs`).
        #[cfg(target_arch = "x86_64")]
        let idct16x16_add: fn(&[i32; 256], &mut [u8], usize) = if caps.sse2 {
            crate::dsp::x86_64::idct16x16_add_sse2_256
        } else {
            scalar_idct16x16_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let idct16x16_add: fn(&[i32; 256], &mut [u8], usize) = scalar_idct16x16_add;

        // Route dense 32×32 IDCT, preferring the AVX2 i16-lane kernel
        // (`idct32x32_add_avx2_1024`) when available, then the SSE2
        // i16-lane kernel (`idct32x32_add_sse2_1024`), then scalar.
        //
        // Both i16-lane kernels share the SAME flowgraph, cospi
        // constants, `(x+8192)>>14` round-shift and saturating
        // `packs_epi32` wraplow (the saturating-vs-modular divergence vs
        // the scalar reference is documented for the SSE2 kernel and
        // agrees on every real VP9 bitstream + the 201-vector corpus).
        // The AVX2 variant just widens 8 i16 lanes/xmm to 16 i16
        // lanes/ymm — 2 row-strips + 2 col-strips instead of 4+4,
        // halving the transpose / instruction count. This is a straight
        // port of libvpx's `vpx_idct32x32_1024_add_avx2`, mirroring the
        // i16-lane win pattern already proven for the 16×16 AVX2 flip.
        // See `crates/rvp9-core/src/dsp/x86_64/idct32x32_avx2_1024.rs`
        // (and the SSE2 sibling) module docstrings.
        //
        // The i32-lane `idct32x32_add_avx2` is now unused by any dispatch
        // slot (the EOB-34/135 paths use the SSE2 i16-lane kernels); it is
        // retained via its re-export and self-tests pending a possible
        // future removal.
        #[cfg(target_arch = "x86_64")]
        let idct32x32_add: fn(&[i32; 1024], &mut [u8], usize) = if caps.avx2 {
            crate::dsp::x86_64::idct32x32_add_avx2_1024
        } else if caps.sse2 {
            crate::dsp::x86_64::idct32x32_add_sse2_1024
        } else {
            scalar_idct32x32_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let idct32x32_add: fn(&[i32; 1024], &mut [u8], usize) = scalar_idct32x32_add;

        // F9 — 32×32 EOB-bucket fast paths. SSE2 i16-lane ports
        // (preferred over the prior AVX2 i32-lane kernels; same i16-lane
        // win pattern as the dense `idct32x32_add` flip in d1a948c).
        // AVX2 wrappers retained but unreferenced by the dispatch.
        // Scalar fallback when `caps.sse2 == false`.
        #[cfg(target_arch = "x86_64")]
        let idct32x32_34_add: fn(&[i32; 1024], &mut [u8], usize) = if caps.sse2 {
            crate::dsp::x86_64::idct32x32_add_sse2_34
        } else {
            scalar_idct32x32_34_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let idct32x32_34_add: fn(&[i32; 1024], &mut [u8], usize) = scalar_idct32x32_34_add;

        // EOB-135 fast path. Prefer the AVX2 i16-lane partial kernel
        // (`idct32x32_add_avx2_135`, a single row pass + 2 column passes
        // of the 256-bit `idct32_1024_16x32` engine — 16 i16/ymm) when
        // AVX2 is present, then the SSE2 i16-lane partial kernel (6 engine
        // runs × 8 i16/xmm), then scalar. Same i16-lane widening win as
        // the dense `idct32x32_add` flip; both SIMD kernels reuse the
        // bit-exact-gated 1024 helpers and saturate identically. The
        // i32-lane `idct32x32_135_add_avx2` remains unreferenced by any
        // slot (retained via re-export + self-tests).
        // See `crates/rvp9-core/src/dsp/x86_64/idct32x32_avx2_135.rs`.
        #[cfg(target_arch = "x86_64")]
        let idct32x32_135_add: fn(&[i32; 1024], &mut [u8], usize) = if caps.avx2 {
            crate::dsp::x86_64::idct32x32_add_avx2_135
        } else if caps.sse2 {
            crate::dsp::x86_64::idct32x32_add_sse2_135
        } else {
            scalar_idct32x32_135_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let idct32x32_135_add: fn(&[i32; 1024], &mut [u8], usize) = scalar_idct32x32_135_add;

        #[cfg(target_arch = "x86_64")]
        let dc_predictor_4x4: fn(&mut [u8], usize, &[u8], &[u8]) = if caps.avx2 {
            crate::dsp::x86_64::dc_predictor_4x4_avx2
        } else {
            scalar_dc_predictor_4x4
        };
        #[cfg(not(target_arch = "x86_64"))]
        let dc_predictor_4x4: fn(&mut [u8], usize, &[u8], &[u8]) = scalar_dc_predictor_4x4;

        // ---- Wave I I1 — size-parametric V_PRED intra-predictor. ----
        // bs ∈ {4, 8, 16, 32}; one slot covers all sizes (the wrapper
        // dispatches internally on `bs` to the matching wide-store
        // width). First non-DC intra-pred slot — pattern that I2/I3/I4
        // mirror for H_PRED / TM_PRED / DC family extension.
        #[cfg(target_arch = "x86_64")]
        let v_predictor: fn(&mut [u8], usize, usize, &[u8]) = if caps.avx2 {
            crate::dsp::x86_64::v_predictor_avx2
        } else {
            scalar_v_predictor
        };
        #[cfg(target_arch = "aarch64")]
        let v_predictor: fn(&mut [u8], usize, usize, &[u8]) = if caps.neon {
            crate::dsp::aarch64::v_predictor_neon
        } else {
            scalar_v_predictor
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let v_predictor: fn(&mut [u8], usize, usize, &[u8]) = scalar_v_predictor;

        // ---- Wave I I2 — size-parametric H_PRED intra-predictor. ----
        // Mirrors V_PRED structurally: one slot covers all four
        // `bs ∈ {4, 8, 16, 32}` sizes; the wrapper dispatches
        // internally on `bs` to the matching wide-store width. H_PRED
        // broadcasts `left[r]` across row r (V_PRED broadcasts
        // `above` down the column axis).
        #[cfg(target_arch = "x86_64")]
        let h_predictor: fn(&mut [u8], usize, usize, &[u8]) = if caps.avx2 {
            crate::dsp::x86_64::h_predictor_avx2
        } else {
            scalar_h_predictor
        };
        #[cfg(target_arch = "aarch64")]
        let h_predictor: fn(&mut [u8], usize, usize, &[u8]) = if caps.neon {
            crate::dsp::aarch64::h_predictor_neon
        } else {
            scalar_h_predictor
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let h_predictor: fn(&mut [u8], usize, usize, &[u8]) = scalar_h_predictor;

        // ---- Wave I I3 — TM_PRED intra-predictor (size-parametric).
        // AVX2-capable on x86_64 when `caps.avx2 == true`; scalar
        // fallback otherwise. The AVX2 kernel handles bs ∈ {4, 8, 16,
        // 32} internally via per-size code paths (SSE2 for 4/8, AVX2
        // for 16/32).
        #[cfg(target_arch = "x86_64")]
        let tm_predictor: TmPredictorFn = if caps.avx2 {
            crate::dsp::x86_64::tm_predictor_avx2
        } else {
            scalar_tm_predictor
        };
        #[cfg(not(target_arch = "x86_64"))]
        let tm_predictor: TmPredictorFn = scalar_tm_predictor;

        // ---- Wave K — directional intra-predictors (D45/D63/D207/
        // D117/D135/D153). Each slot covers all bs ∈ {4, 8, 16, 32};
        // the AVX2 wrappers (and scalar adapters) route the bs == 4
        // libvpx specialization internally for D45/D63/D207. D117/D135/
        // D153 are generic for all sizes. AVX2 on x86_64 when caps.avx2.
        #[cfg(target_arch = "x86_64")]
        let d45_predictor: fn(&mut [u8], usize, usize, &[u8]) = if caps.avx2 {
            crate::dsp::x86_64::d45_predictor_avx2
        } else {
            scalar_d45_predictor
        };
        #[cfg(not(target_arch = "x86_64"))]
        let d45_predictor: fn(&mut [u8], usize, usize, &[u8]) = scalar_d45_predictor;

        #[cfg(target_arch = "x86_64")]
        let d63_predictor: fn(&mut [u8], usize, usize, &[u8]) = if caps.avx2 {
            crate::dsp::x86_64::d63_predictor_avx2
        } else {
            scalar_d63_predictor
        };
        #[cfg(not(target_arch = "x86_64"))]
        let d63_predictor: fn(&mut [u8], usize, usize, &[u8]) = scalar_d63_predictor;

        #[cfg(target_arch = "x86_64")]
        let d207_predictor: fn(&mut [u8], usize, usize, &[u8]) = if caps.avx2 {
            crate::dsp::x86_64::d207_predictor_avx2
        } else {
            scalar_d207_predictor
        };
        #[cfg(not(target_arch = "x86_64"))]
        let d207_predictor: fn(&mut [u8], usize, usize, &[u8]) = scalar_d207_predictor;

        #[cfg(target_arch = "x86_64")]
        let d117_predictor: DirectionalCtxPredFn = if caps.avx2 {
            crate::dsp::x86_64::d117_predictor_avx2
        } else {
            scalar_d117_predictor
        };
        #[cfg(not(target_arch = "x86_64"))]
        let d117_predictor: DirectionalCtxPredFn = scalar_d117_predictor;

        #[cfg(target_arch = "x86_64")]
        let d135_predictor: DirectionalCtxPredFn = if caps.avx2 {
            crate::dsp::x86_64::d135_predictor_avx2
        } else {
            scalar_d135_predictor
        };
        #[cfg(not(target_arch = "x86_64"))]
        let d135_predictor: DirectionalCtxPredFn = scalar_d135_predictor;

        #[cfg(target_arch = "x86_64")]
        let d153_predictor: DirectionalCtxPredFn = if caps.avx2 {
            crate::dsp::x86_64::d153_predictor_avx2
        } else {
            scalar_d153_predictor
        };
        #[cfg(not(target_arch = "x86_64"))]
        let d153_predictor: DirectionalCtxPredFn = scalar_d153_predictor;

        // ---- Wave I I4 — DC_PRED family extension (size-parametric).
        // Four new slots covering all four availability variants:
        //   - dc_predict_full(above, left): (above_avail, left_avail) = (true, true)
        //   - dc_predict_top(above):        (true, false)
        //   - dc_predict_left(left):        (false, true)
        //   - dc_predict_128():             (false, false)
        // Each AVX2 wrapper dispatches internally on bs ∈ {4, 8, 16, 32}.
        // The legacy `dc_predictor_4x4` slot above still owns the
        // (bs=4, both-avail) fast path; the I4 router in
        // `intra_pred::dc_predict_with_availability` picks between the
        // legacy slot and the I4 slot based on bs.
        #[cfg(target_arch = "x86_64")]
        let dc_predict_full: fn(&mut [u8], usize, usize, &[u8], &[u8]) = if caps.avx2 {
            crate::dsp::x86_64::dc_predict_full_avx2
        } else {
            scalar_dc_predict_full
        };
        #[cfg(target_arch = "aarch64")]
        let dc_predict_full: fn(&mut [u8], usize, usize, &[u8], &[u8]) = if caps.neon {
            crate::dsp::aarch64::dc_predict_full_neon
        } else {
            scalar_dc_predict_full
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let dc_predict_full: fn(&mut [u8], usize, usize, &[u8], &[u8]) = scalar_dc_predict_full;

        #[cfg(target_arch = "x86_64")]
        let dc_predict_top: fn(&mut [u8], usize, usize, &[u8]) = if caps.avx2 {
            crate::dsp::x86_64::dc_predict_top_avx2
        } else {
            scalar_dc_predict_top
        };
        #[cfg(not(target_arch = "x86_64"))]
        let dc_predict_top: fn(&mut [u8], usize, usize, &[u8]) = scalar_dc_predict_top;

        #[cfg(target_arch = "x86_64")]
        let dc_predict_left: fn(&mut [u8], usize, usize, &[u8]) = if caps.avx2 {
            crate::dsp::x86_64::dc_predict_left_avx2
        } else {
            scalar_dc_predict_left
        };
        #[cfg(not(target_arch = "x86_64"))]
        let dc_predict_left: fn(&mut [u8], usize, usize, &[u8]) = scalar_dc_predict_left;

        #[cfg(target_arch = "x86_64")]
        let dc_predict_128: fn(&mut [u8], usize, usize) = if caps.avx2 {
            crate::dsp::x86_64::dc_predict_128_avx2
        } else {
            scalar_dc_predict_128
        };
        #[cfg(not(target_arch = "x86_64"))]
        let dc_predict_128: fn(&mut [u8], usize, usize) = scalar_dc_predict_128;

        // ---- Wave E G4-β — hybrid ADST slots. AVX2-capable on x86_64
        // when `caps.avx2 == true`; scalar fallback otherwise. The AVX2
        // inner kernel keeps the i32 1D math bit-exact to the scalar
        // reference and vectorises only the final add/clamp/pack stage
        // (same approach as the idct AVX2 family). The slot's signature
        // carries `tx_type` so the AVX2 / scalar dispatch routes to the
        // right row / col 1D pair internally.
        #[cfg(target_arch = "x86_64")]
        let iht4x4_add: fn(&[i32; 16], &mut [u8], usize, TxType) = if caps.avx2 {
            crate::dsp::x86_64::iht4x4_add_avx2
        } else {
            scalar_iht4x4_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let iht4x4_add: fn(&[i32; 16], &mut [u8], usize, TxType) = scalar_iht4x4_add;

        #[cfg(target_arch = "x86_64")]
        let iht8x8_add: fn(&[i32; 64], &mut [u8], usize, TxType) = if caps.avx2 {
            crate::dsp::x86_64::iht8x8_add_avx2
        } else {
            scalar_iht8x8_add
        };
        #[cfg(target_arch = "aarch64")]
        let iht8x8_add: fn(&[i32; 64], &mut [u8], usize, TxType) = if caps.neon {
            crate::dsp::aarch64::iht8x8_add_neon
        } else {
            scalar_iht8x8_add
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let iht8x8_add: fn(&[i32; 64], &mut [u8], usize, TxType) = scalar_iht8x8_add;

        // ---- Wave F F5 — 16×16 hybrid ADST slot. AVX2-capable on
        // x86_64 when `caps.avx2 == true`; scalar fallback otherwise.
        // Same finisher-only approach as the other hybrid AVX2 slots:
        // the i32 1D math is bit-exact to the scalar reference and the
        // final add/clamp/pack stage is vectorised.
        #[cfg(target_arch = "x86_64")]
        let iht16x16_add: fn(&[i32; 256], &mut [u8], usize, TxType) = if caps.avx2 {
            crate::dsp::x86_64::iht16x16_add_avx2
        } else {
            scalar_iht16x16_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let iht16x16_add: fn(&[i32; 256], &mut [u8], usize, TxType) = scalar_iht16x16_add;

        // ---- Wave F F3 / F4 — 16×16 idct EOB-bucket fast paths.
        // SSE2 i16-lane ports (preferred over the prior AVX2 i32-lane
        // kernels — 16 i16 lanes per xmm vs 8 i32 per ymm matches the
        // dense `_256` flip's win pattern). AVX2 wrappers retained for
        // the iht16x16 path. Bit-exact vs the scalar reference on the
        // 201-vector corpus; saturating-vs-modular divergence on
        // synthetic adversarial inputs (same trade as the dense
        // `idct16x16_add_sse2_256` kernel — see its module docstring).
        #[cfg(target_arch = "x86_64")]
        let idct16x16_10_add: fn(&[i32; 256], &mut [u8], usize) = if caps.sse2 {
            crate::dsp::x86_64::idct16x16_add_sse2_10
        } else {
            scalar_idct16x16_10_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let idct16x16_10_add: fn(&[i32; 256], &mut [u8], usize) = scalar_idct16x16_10_add;

        #[cfg(target_arch = "x86_64")]
        let idct16x16_38_add: fn(&[i32; 256], &mut [u8], usize) = if caps.sse2 {
            crate::dsp::x86_64::idct16x16_add_sse2_38
        } else {
            scalar_idct16x16_38_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let idct16x16_38_add: fn(&[i32; 256], &mut [u8], usize) = scalar_idct16x16_38_add;

        // ---- Wave H H1 — loopfilter AVX2 gate (granular for H7 bisect) ----
        //
        // The F10 + G3-G6 AVX2 loopfilter kernels passed unit tests but
        // exposed conformance regressions (58/201) when made live by H1.
        // Until the kernels are differential-tested against scalar on the
        // conformance corpus (Wave H follow-up H7), the AVX2 paths are
        // gated OFF by default.
        //
        // `RVP9_DSP_LOOPFILTER_AVX2` env var values:
        //   - unset / empty: ALL 6 AVX2 lpf_* paths route to scalar
        //   - `1`           : ALL 6 AVX2 paths active (full activation)
        //   - comma list    : individual kernels active (H7 bisection),
        //                     where tags are any subset of
        //                     `v4,h4,v8,h8,v16,h16`. Example:
        //                     `RVP9_DSP_LOOPFILTER_AVX2=v4,h4` enables
        //                     just the 4-tap kernels.
        //
        // The dual-variant slots are gated independently below
        // (H2 promoted 4-tap duals, H3 promoted 8-tap duals — both
        // by `caps.avx2` alone, no env-var override; the 16-tap
        // duals remain scalar pending H4).
        #[cfg(target_arch = "x86_64")]
        let lpf_avx2_mask = if caps.avx2 {
            parse_lpf_avx2_mask(std::env::var("RVP9_DSP_LOOPFILTER_AVX2").ok().as_deref())
        } else {
            LpfAvx2Mask::NONE
        };

        // ---- Wave F G F10 — vertical 4-tap loopfilter (AVX2 gated). ----
        #[cfg(target_arch = "x86_64")]
        let lpf_vertical_4: fn(&mut [u8], usize, usize, u8, u8, u8) =
            if lpf_avx2_mask.contains(LpfAvx2Mask::V4) {
                crate::dsp::x86_64::lpf_vertical_4_avx2
            } else {
                scalar_lpf_vertical_4
            };
        #[cfg(target_arch = "aarch64")]
        let lpf_vertical_4: fn(&mut [u8], usize, usize, u8, u8, u8) = if caps.neon {
            crate::dsp::aarch64::lpf_vertical_4_neon
        } else {
            scalar_lpf_vertical_4
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let lpf_vertical_4: fn(&mut [u8], usize, usize, u8, u8, u8) = scalar_lpf_vertical_4;

        // ---- Wave G G3 — horizontal 4-tap loopfilter (AVX2 gated). ----
        #[cfg(target_arch = "x86_64")]
        let lpf_horizontal_4: fn(&mut [u8], usize, usize, u8, u8, u8) =
            if lpf_avx2_mask.contains(LpfAvx2Mask::H4) {
                crate::dsp::x86_64::lpf_horizontal_4_avx2
            } else {
                scalar_lpf_horizontal_4
            };
        #[cfg(target_arch = "aarch64")]
        let lpf_horizontal_4: fn(&mut [u8], usize, usize, u8, u8, u8) = if caps.neon {
            crate::dsp::aarch64::lpf_horizontal_4_neon
        } else {
            scalar_lpf_horizontal_4
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let lpf_horizontal_4: fn(&mut [u8], usize, usize, u8, u8, u8) = scalar_lpf_horizontal_4;

        // ---- Wave G G4 — 8-tap vertical loopfilter (AVX2 gated). ----
        #[cfg(target_arch = "x86_64")]
        let lpf_vertical_8: fn(&mut [u8], usize, usize, u8, u8, u8) =
            if lpf_avx2_mask.contains(LpfAvx2Mask::V8) {
                crate::dsp::x86_64::lpf_vertical_8_avx2
            } else {
                scalar_lpf_vertical_8
            };
        #[cfg(target_arch = "aarch64")]
        let lpf_vertical_8: fn(&mut [u8], usize, usize, u8, u8, u8) = if caps.neon {
            crate::dsp::aarch64::lpf_vertical_8_neon
        } else {
            scalar_lpf_vertical_8
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let lpf_vertical_8: fn(&mut [u8], usize, usize, u8, u8, u8) = scalar_lpf_vertical_8;

        // ---- Wave G G5 — 8-tap horizontal loopfilter (AVX2 gated). ----
        #[cfg(target_arch = "x86_64")]
        let lpf_horizontal_8: fn(&mut [u8], usize, usize, u8, u8, u8) =
            if lpf_avx2_mask.contains(LpfAvx2Mask::H8) {
                crate::dsp::x86_64::lpf_horizontal_8_avx2
            } else {
                scalar_lpf_horizontal_8
            };
        #[cfg(target_arch = "aarch64")]
        let lpf_horizontal_8: fn(&mut [u8], usize, usize, u8, u8, u8) = if caps.neon {
            crate::dsp::aarch64::lpf_horizontal_8_neon
        } else {
            scalar_lpf_horizontal_8
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let lpf_horizontal_8: fn(&mut [u8], usize, usize, u8, u8, u8) = scalar_lpf_horizontal_8;

        // ---- Wave G G6 — 16-tap vertical edge loopfilter (AVX2 gated). ----
        #[cfg(target_arch = "x86_64")]
        let lpf_vertical_16: fn(&mut [u8], usize, usize, u8, u8, u8) =
            if lpf_avx2_mask.contains(LpfAvx2Mask::V16) {
                crate::dsp::x86_64::lpf_vertical_16_avx2
            } else {
                scalar_lpf_vertical_16
            };
        #[cfg(target_arch = "aarch64")]
        let lpf_vertical_16: fn(&mut [u8], usize, usize, u8, u8, u8) = if caps.neon {
            crate::dsp::aarch64::lpf_vertical_16_neon
        } else {
            scalar_lpf_vertical_16
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let lpf_vertical_16: fn(&mut [u8], usize, usize, u8, u8, u8) = scalar_lpf_vertical_16;

        // ---- Wave G G6 — 16-tap horizontal edge loopfilter (AVX2 gated). ----
        #[cfg(target_arch = "x86_64")]
        let lpf_horizontal_16: fn(&mut [u8], usize, usize, u8, u8, u8) =
            if lpf_avx2_mask.contains(LpfAvx2Mask::H16) {
                crate::dsp::x86_64::lpf_horizontal_16_avx2
            } else {
                scalar_lpf_horizontal_16
            };
        #[cfg(target_arch = "aarch64")]
        let lpf_horizontal_16: fn(&mut [u8], usize, usize, u8, u8, u8) = if caps.neon {
            crate::dsp::aarch64::lpf_horizontal_16_neon
        } else {
            scalar_lpf_horizontal_16
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let lpf_horizontal_16: fn(&mut [u8], usize, usize, u8, u8, u8) = scalar_lpf_horizontal_16;

        // ---- Wave H H2 + H3 — dual-variant 4/8-tap loopfilter slots
        // (AVX2 gated by `caps.avx2`). Unlike the single-variant slots
        // these are NOT controlled by `LpfAvx2Mask`/`RVP9_DSP_LOOPFILTER_AVX2`
        // — H2/H3 ship them on by default for any AVX2 CPU because
        // the underlying single-variant kernels they delegate to are
        // already H7-bisection clean (V4/H4/V8/H8). The 16-tap duals
        // remain scalar pending H4 (after H7 fixes V16). ----
        #[cfg(target_arch = "x86_64")]
        let lpf_vertical_4_dual: LpfDualFn = if caps.avx2 {
            crate::dsp::x86_64::lpf_vertical_4_dual_avx2
        } else {
            scalar_lpf_vertical_4_dual
        };
        #[cfg(target_arch = "aarch64")]
        let lpf_vertical_4_dual: LpfDualFn = if caps.neon {
            crate::dsp::aarch64::lpf_vertical_4_dual_neon
        } else {
            scalar_lpf_vertical_4_dual
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let lpf_vertical_4_dual: LpfDualFn = scalar_lpf_vertical_4_dual;

        #[cfg(target_arch = "x86_64")]
        let lpf_horizontal_4_dual: LpfDualFn = if caps.avx2 {
            crate::dsp::x86_64::lpf_horizontal_4_dual_avx2
        } else {
            scalar_lpf_horizontal_4_dual
        };
        #[cfg(target_arch = "aarch64")]
        let lpf_horizontal_4_dual: LpfDualFn = if caps.neon {
            crate::dsp::aarch64::lpf_horizontal_4_dual_neon
        } else {
            scalar_lpf_horizontal_4_dual
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let lpf_horizontal_4_dual: LpfDualFn = scalar_lpf_horizontal_4_dual;

        #[cfg(target_arch = "x86_64")]
        let lpf_vertical_8_dual: LpfDualFn = if caps.avx2 {
            crate::dsp::x86_64::lpf_vertical_8_dual_avx2
        } else {
            scalar_lpf_vertical_8_dual
        };
        #[cfg(target_arch = "aarch64")]
        let lpf_vertical_8_dual: LpfDualFn = if caps.neon {
            crate::dsp::aarch64::lpf_vertical_8_dual_neon
        } else {
            scalar_lpf_vertical_8_dual
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let lpf_vertical_8_dual: LpfDualFn = scalar_lpf_vertical_8_dual;

        #[cfg(target_arch = "x86_64")]
        let lpf_horizontal_8_dual: LpfDualFn = if caps.avx2 {
            crate::dsp::x86_64::lpf_horizontal_8_dual_avx2
        } else {
            scalar_lpf_horizontal_8_dual
        };
        #[cfg(target_arch = "aarch64")]
        let lpf_horizontal_8_dual: LpfDualFn = if caps.neon {
            crate::dsp::aarch64::lpf_horizontal_8_dual_neon
        } else {
            scalar_lpf_horizontal_8_dual
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let lpf_horizontal_8_dual: LpfDualFn = scalar_lpf_horizontal_8_dual;

        // 16-tap duals share a single threshold set (libvpx convention).
        // Wave H H4 promoted both to AVX2 via two-singles delegation
        // (H7 fixed v16; H4.1 fixed the parallel bug in h16's
        // `flat_mask5_avx2` call site).
        #[cfg(target_arch = "x86_64")]
        let lpf_vertical_16_dual: fn(&mut [u8], usize, usize, u8, u8, u8) = if caps.avx2 {
            crate::dsp::x86_64::lpf_vertical_16_dual_avx2
        } else {
            scalar_lpf_vertical_16_dual
        };
        #[cfg(not(target_arch = "x86_64"))]
        #[cfg(not(target_arch = "aarch64"))]
        let lpf_vertical_16_dual: fn(&mut [u8], usize, usize, u8, u8, u8) =
            scalar_lpf_vertical_16_dual;
        #[cfg(target_arch = "aarch64")]
        let lpf_vertical_16_dual: fn(&mut [u8], usize, usize, u8, u8, u8) = if caps.neon {
            crate::dsp::aarch64::lpf_vertical_16_dual_neon
        } else {
            scalar_lpf_vertical_16_dual
        };

        #[cfg(target_arch = "x86_64")]
        let lpf_horizontal_16_dual: fn(&mut [u8], usize, usize, u8, u8, u8) = if caps.avx2 {
            crate::dsp::x86_64::lpf_horizontal_16_dual_avx2
        } else {
            scalar_lpf_horizontal_16_dual
        };
        #[cfg(not(target_arch = "x86_64"))]
        #[cfg(not(target_arch = "aarch64"))]
        let lpf_horizontal_16_dual: fn(&mut [u8], usize, usize, u8, u8, u8) =
            scalar_lpf_horizontal_16_dual;
        #[cfg(target_arch = "aarch64")]
        let lpf_horizontal_16_dual: fn(&mut [u8], usize, usize, u8, u8, u8) = if caps.neon {
            crate::dsp::aarch64::lpf_horizontal_16_dual_neon
        } else {
            scalar_lpf_horizontal_16_dual
        };

        // ---- Wave H H6 — pass-2 dequant finisher (AVX2 on x86_64). ----
        // Pass 2 of the Wave H H5 split detokenize loop. Bit-exact
        // vs the scalar reference by construction: `_mm256_mullo_epi32`
        // returns the low 32 bits of the i32×i32 product (which always
        // fits in i32 here — see kernel docstring for the bound),
        // `_mm256_sra_epi32` matches scalar `>>` on i32, and
        // `_mm256_sign_epi32(mag, ±1)` matches `if sign { -mag } else
        // { mag }` exactly. Scalar tail covers `< 8` remaining entries.
        #[cfg(target_arch = "x86_64")]
        let plane_dequant_finisher: PlaneDequantFinisherFn = if caps.avx2 {
            crate::dsp::x86_64::plane_dequant_finisher_avx2
        } else {
            scalar_plane_dequant_finisher
        };
        #[cfg(not(target_arch = "x86_64"))]
        let plane_dequant_finisher: PlaneDequantFinisherFn = scalar_plane_dequant_finisher;

        // ---- Wave I I5 — motion-compensation dispatch slots. Direct
        // aliases of the existing scalar `crate::inter::filter::*` fns
        // — those already match the slot signatures verbatim (no
        // PredCtx-style wrapping needed). The slots reserve the family
        // name and dispatch shape so individual fns can be promoted to
        // AVX2 without touching the call sites in `inter_predictor`.
        //
        // ---- Wave J J1 — AVX2 convolve_copy promotion. ----
        #[cfg(target_arch = "x86_64")]
        let convolve_copy: fn(&[u8], usize, &mut [u8], usize, usize, usize) = if caps.avx2 {
            crate::dsp::x86_64::convolve_copy_avx2
        } else {
            crate::inter::filter::convolve_copy
        };
        #[cfg(target_arch = "aarch64")]
        let convolve_copy: fn(&[u8], usize, &mut [u8], usize, usize, usize) = if caps.neon {
            crate::dsp::aarch64::convolve_copy_neon
        } else {
            crate::inter::filter::convolve_copy
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let convolve_copy: fn(&[u8], usize, &mut [u8], usize, usize, usize) =
            crate::inter::filter::convolve_copy;
        // ---- Wave J J2 — AVX2 convolve_avg promotion. ----
        #[cfg(target_arch = "x86_64")]
        let convolve_avg: fn(&[u8], usize, &mut [u8], usize, usize, usize) = if caps.avx2 {
            crate::dsp::x86_64::convolve_avg_avx2
        } else {
            crate::inter::filter::convolve_avg
        };
        #[cfg(not(target_arch = "x86_64"))]
        let convolve_avg: fn(&[u8], usize, &mut [u8], usize, usize, usize) =
            crate::inter::filter::convolve_avg;
        // ---- Wave J J3 — AVX2 convolve8_horiz promotion. ----
        #[cfg(target_arch = "x86_64")]
        let convolve8_horiz: Convolve8OneAxisFn = if caps.avx2 {
            crate::dsp::x86_64::convolve8_horiz_avx2
        } else {
            crate::inter::filter::convolve8_horiz
        };
        #[cfg(target_arch = "aarch64")]
        let convolve8_horiz: Convolve8OneAxisFn = if caps.neon {
            crate::dsp::aarch64::convolve8_horiz_neon
        } else {
            crate::inter::filter::convolve8_horiz
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let convolve8_horiz: Convolve8OneAxisFn = crate::inter::filter::convolve8_horiz;
        // ---- Wave J J4 — AVX2 convolve8_vert promotion. ----
        #[cfg(target_arch = "x86_64")]
        let convolve8_vert: Convolve8OneAxisFn = if caps.avx2 {
            crate::dsp::x86_64::convolve8_vert_avx2
        } else {
            crate::inter::filter::convolve8_vert
        };
        #[cfg(target_arch = "aarch64")]
        let convolve8_vert: Convolve8OneAxisFn = if caps.neon {
            crate::dsp::aarch64::convolve8_vert_neon
        } else {
            crate::inter::filter::convolve8_vert
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let convolve8_vert: Convolve8OneAxisFn = crate::inter::filter::convolve8_vert;
        // ---- Wave J J5 — AVX2 convolve8 (full 2-D) promotion. ----
        #[cfg(target_arch = "x86_64")]
        let convolve8: Convolve8TwoAxisFn = if caps.avx2 {
            crate::dsp::x86_64::convolve8_avx2
        } else {
            crate::inter::filter::convolve8
        };
        #[cfg(target_arch = "aarch64")]
        let convolve8: Convolve8TwoAxisFn = if caps.neon {
            crate::dsp::aarch64::convolve8_neon
        } else {
            crate::inter::filter::convolve8
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let convolve8: Convolve8TwoAxisFn = crate::inter::filter::convolve8;
        // ---- Wave J J6 — AVX2 convolve8_avg_horiz promotion. ----
        #[cfg(target_arch = "x86_64")]
        let convolve8_avg_horiz: Convolve8OneAxisFn = if caps.avx2 {
            crate::dsp::x86_64::convolve8_avg_horiz_avx2
        } else {
            crate::inter::filter::convolve8_avg_horiz
        };
        #[cfg(not(target_arch = "x86_64"))]
        let convolve8_avg_horiz: Convolve8OneAxisFn = crate::inter::filter::convolve8_avg_horiz;
        // ---- Wave J J7 — AVX2 convolve8_avg_vert promotion. ----
        #[cfg(target_arch = "x86_64")]
        let convolve8_avg_vert: Convolve8OneAxisFn = if caps.avx2 {
            crate::dsp::x86_64::convolve8_avg_vert_avx2
        } else {
            crate::inter::filter::convolve8_avg_vert
        };
        #[cfg(not(target_arch = "x86_64"))]
        let convolve8_avg_vert: Convolve8OneAxisFn = crate::inter::filter::convolve8_avg_vert;
        // ---- Wave J J8 — AVX2 convolve8_avg (full 2-D) promotion. ----
        #[cfg(target_arch = "x86_64")]
        let convolve8_avg: Convolve8TwoAxisFn = if caps.avx2 {
            crate::dsp::x86_64::convolve8_avg_avx2
        } else {
            crate::inter::filter::convolve8_avg
        };
        #[cfg(not(target_arch = "x86_64"))]
        let convolve8_avg: Convolve8TwoAxisFn = crate::inter::filter::convolve8_avg;

        // ---- Wave L L1 — AVX2 scaled 2-D convolution promotion. ----
        #[cfg(target_arch = "x86_64")]
        let convolve8_scaled: ScaledConvolveFn = if caps.avx2 {
            crate::dsp::x86_64::convolve8_scaled_avx2
        } else {
            crate::inter::filter::convolve8_scaled
        };
        #[cfg(not(target_arch = "x86_64"))]
        let convolve8_scaled: ScaledConvolveFn = crate::inter::filter::convolve8_scaled;
        #[cfg(target_arch = "x86_64")]
        let convolve8_scaled_avg: ScaledConvolveFn = if caps.avx2 {
            crate::dsp::x86_64::convolve8_scaled_avg_avx2
        } else {
            crate::inter::filter::convolve8_scaled_avg
        };
        #[cfg(not(target_arch = "x86_64"))]
        let convolve8_scaled_avg: ScaledConvolveFn = crate::inter::filter::convolve8_scaled_avg;

        // ---- P1-B — per-size SAD table. ----
        // One clause per VP9 block size: the AVX2 safe wrapper on AVX2
        // hosts, the scalar `vpx_sad{W}x{H}` reference otherwise. The two
        // arms are byte-exact (integer `Σ|a − b|`, no u32 overflow), so the
        // motion search sees identical SADs either way.
        #[cfg(target_arch = "x86_64")]
        let sad: SadTable = {
            use crate::dsp::distortion as sc;
            use crate::dsp::x86_64 as av;
            macro_rules! sad_slot {
                ($scalar:path, $avx2:path) => {
                    if caps.avx2 {
                        $avx2 as SadFn
                    } else {
                        $scalar as SadFn
                    }
                };
            }
            // Order MUST match `BlockSize::idx()` (Block4x4 = 0 .. Block64x64 = 12).
            SadTable {
                fns: [
                    sad_slot!(sc::vpx_sad4x4, av::vpx_sad4x4_avx2),
                    sad_slot!(sc::vpx_sad4x8, av::vpx_sad4x8_avx2),
                    sad_slot!(sc::vpx_sad8x4, av::vpx_sad8x4_avx2),
                    sad_slot!(sc::vpx_sad8x8, av::vpx_sad8x8_avx2),
                    sad_slot!(sc::vpx_sad8x16, av::vpx_sad8x16_avx2),
                    sad_slot!(sc::vpx_sad16x8, av::vpx_sad16x8_avx2),
                    sad_slot!(sc::vpx_sad16x16, av::vpx_sad16x16_avx2),
                    sad_slot!(sc::vpx_sad16x32, av::vpx_sad16x32_avx2),
                    sad_slot!(sc::vpx_sad32x16, av::vpx_sad32x16_avx2),
                    sad_slot!(sc::vpx_sad32x32, av::vpx_sad32x32_avx2),
                    sad_slot!(sc::vpx_sad32x64, av::vpx_sad32x64_avx2),
                    sad_slot!(sc::vpx_sad64x32, av::vpx_sad64x32_avx2),
                    sad_slot!(sc::vpx_sad64x64, av::vpx_sad64x64_avx2),
                ],
            }
        };
        #[cfg(target_arch = "aarch64")]
        let sad: SadTable = {
            use crate::dsp::aarch64 as neon;
            use crate::dsp::distortion as sc;
            macro_rules! sad_slot {
                ($scalar:path, $neon:path) => {
                    if caps.neon {
                        $neon as SadFn
                    } else {
                        $scalar as SadFn
                    }
                };
            }
            SadTable {
                fns: [
                    sad_slot!(sc::vpx_sad4x4, neon::vpx_sad4x4_neon),
                    sad_slot!(sc::vpx_sad4x8, neon::vpx_sad4x8_neon),
                    sad_slot!(sc::vpx_sad8x4, neon::vpx_sad8x4_neon),
                    sad_slot!(sc::vpx_sad8x8, neon::vpx_sad8x8_neon),
                    sad_slot!(sc::vpx_sad8x16, neon::vpx_sad8x16_neon),
                    sad_slot!(sc::vpx_sad16x8, neon::vpx_sad16x8_neon),
                    sad_slot!(sc::vpx_sad16x16, neon::vpx_sad16x16_neon),
                    sad_slot!(sc::vpx_sad16x32, neon::vpx_sad16x32_neon),
                    sad_slot!(sc::vpx_sad32x16, neon::vpx_sad32x16_neon),
                    sad_slot!(sc::vpx_sad32x32, neon::vpx_sad32x32_neon),
                    sad_slot!(sc::vpx_sad32x64, neon::vpx_sad32x64_neon),
                    sad_slot!(sc::vpx_sad64x32, neon::vpx_sad64x32_neon),
                    sad_slot!(sc::vpx_sad64x64, neon::vpx_sad64x64_neon),
                ],
            }
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let sad: SadTable = {
            let _ = caps;
            use crate::dsp::distortion as sc;
            SadTable {
                fns: [
                    sc::vpx_sad4x4 as SadFn,
                    sc::vpx_sad4x8,
                    sc::vpx_sad8x4,
                    sc::vpx_sad8x8,
                    sc::vpx_sad8x16,
                    sc::vpx_sad16x8,
                    sc::vpx_sad16x16,
                    sc::vpx_sad16x32,
                    sc::vpx_sad32x16,
                    sc::vpx_sad32x32,
                    sc::vpx_sad32x64,
                    sc::vpx_sad64x32,
                    sc::vpx_sad64x64,
                ],
            }
        };

        // ---- PR3 — per-size 4-reference SAD (sdx4df) table. ----
        // One clause per VP9 block size, mirroring the single-SAD table: the
        // SIMD safe wrapper on supported hosts, the scalar four-`sad`
        // reference otherwise. Every arm is byte-exact (each lane is an
        // integer `Σ|a − b|`), so the search sees identical SADs either way.
        #[cfg(target_arch = "x86_64")]
        let sad4d: Sad4dTable = {
            use crate::dsp::distortion as sc;
            use crate::dsp::x86_64 as av;
            macro_rules! sad4d_slot {
                ($scalar:path, $avx2:path) => {
                    if caps.avx2 {
                        $avx2 as Sad4dFn
                    } else {
                        $scalar as Sad4dFn
                    }
                };
            }
            // Order MUST match `BlockSize::idx()` (Block4x4 = 0 .. Block64x64 = 12).
            Sad4dTable {
                fns: [
                    sad4d_slot!(sc::vpx_sad4x4x4d, av::vpx_sad4x4x4d_avx2),
                    sad4d_slot!(sc::vpx_sad4x8x4d, av::vpx_sad4x8x4d_avx2),
                    sad4d_slot!(sc::vpx_sad8x4x4d, av::vpx_sad8x4x4d_avx2),
                    sad4d_slot!(sc::vpx_sad8x8x4d, av::vpx_sad8x8x4d_avx2),
                    sad4d_slot!(sc::vpx_sad8x16x4d, av::vpx_sad8x16x4d_avx2),
                    sad4d_slot!(sc::vpx_sad16x8x4d, av::vpx_sad16x8x4d_avx2),
                    sad4d_slot!(sc::vpx_sad16x16x4d, av::vpx_sad16x16x4d_avx2),
                    sad4d_slot!(sc::vpx_sad16x32x4d, av::vpx_sad16x32x4d_avx2),
                    sad4d_slot!(sc::vpx_sad32x16x4d, av::vpx_sad32x16x4d_avx2),
                    sad4d_slot!(sc::vpx_sad32x32x4d, av::vpx_sad32x32x4d_avx2),
                    sad4d_slot!(sc::vpx_sad32x64x4d, av::vpx_sad32x64x4d_avx2),
                    sad4d_slot!(sc::vpx_sad64x32x4d, av::vpx_sad64x32x4d_avx2),
                    sad4d_slot!(sc::vpx_sad64x64x4d, av::vpx_sad64x64x4d_avx2),
                ],
            }
        };
        #[cfg(target_arch = "aarch64")]
        let sad4d: Sad4dTable = {
            use crate::dsp::aarch64 as neon;
            use crate::dsp::distortion as sc;
            macro_rules! sad4d_slot {
                ($scalar:path, $neon:path) => {
                    if caps.neon {
                        $neon as Sad4dFn
                    } else {
                        $scalar as Sad4dFn
                    }
                };
                ($scalar:path, $neon:path, $dotprod:path) => {
                    if caps.dotprod {
                        $dotprod as Sad4dFn
                    } else if caps.neon {
                        $neon as Sad4dFn
                    } else {
                        $scalar as Sad4dFn
                    }
                };
            }
            Sad4dTable {
                fns: [
                    sad4d_slot!(sc::vpx_sad4x4x4d, neon::vpx_sad4x4x4d_neon),
                    sad4d_slot!(sc::vpx_sad4x8x4d, neon::vpx_sad4x8x4d_neon),
                    sad4d_slot!(sc::vpx_sad8x4x4d, neon::vpx_sad8x4x4d_neon),
                    sad4d_slot!(sc::vpx_sad8x8x4d, neon::vpx_sad8x8x4d_neon),
                    sad4d_slot!(sc::vpx_sad8x16x4d, neon::vpx_sad8x16x4d_neon),
                    sad4d_slot!(
                        sc::vpx_sad16x8x4d,
                        neon::vpx_sad16x8x4d_neon,
                        neon::vpx_sad16x8x4d_dotprod
                    ),
                    sad4d_slot!(
                        sc::vpx_sad16x16x4d,
                        neon::vpx_sad16x16x4d_neon,
                        neon::vpx_sad16x16x4d_dotprod
                    ),
                    sad4d_slot!(
                        sc::vpx_sad16x32x4d,
                        neon::vpx_sad16x32x4d_neon,
                        neon::vpx_sad16x32x4d_dotprod
                    ),
                    sad4d_slot!(
                        sc::vpx_sad32x16x4d,
                        neon::vpx_sad32x16x4d_neon,
                        neon::vpx_sad32x16x4d_dotprod
                    ),
                    sad4d_slot!(
                        sc::vpx_sad32x32x4d,
                        neon::vpx_sad32x32x4d_neon,
                        neon::vpx_sad32x32x4d_dotprod
                    ),
                    sad4d_slot!(
                        sc::vpx_sad32x64x4d,
                        neon::vpx_sad32x64x4d_neon,
                        neon::vpx_sad32x64x4d_dotprod
                    ),
                    sad4d_slot!(
                        sc::vpx_sad64x32x4d,
                        neon::vpx_sad64x32x4d_neon,
                        neon::vpx_sad64x32x4d_dotprod
                    ),
                    sad4d_slot!(
                        sc::vpx_sad64x64x4d,
                        neon::vpx_sad64x64x4d_neon,
                        neon::vpx_sad64x64x4d_dotprod
                    ),
                ],
            }
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let sad4d: Sad4dTable = {
            let _ = caps;
            use crate::dsp::distortion as sc;
            Sad4dTable {
                fns: [
                    sc::vpx_sad4x4x4d as Sad4dFn,
                    sc::vpx_sad4x8x4d,
                    sc::vpx_sad8x4x4d,
                    sc::vpx_sad8x8x4d,
                    sc::vpx_sad8x16x4d,
                    sc::vpx_sad16x8x4d,
                    sc::vpx_sad16x16x4d,
                    sc::vpx_sad16x32x4d,
                    sc::vpx_sad32x16x4d,
                    sc::vpx_sad32x32x4d,
                    sc::vpx_sad32x64x4d,
                    sc::vpx_sad64x32x4d,
                    sc::vpx_sad64x64x4d,
                ],
            }
        };

        // ---- PR4 — per-size block (sum, sse) reduction table. ----
        // One clause per VP9 block size: the AVX2 safe wrapper on AVX2 hosts,
        // the scalar `var_sum_sse_{W}x{H}` reference otherwise. The two arms are
        // byte-exact (integer reduction, no overflow), so the RD-model variance
        // finishers see identical `(sum, sse)` either way.
        #[cfg(target_arch = "x86_64")]
        let variance: VarTable = {
            use crate::dsp::distortion as sc;
            use crate::dsp::x86_64 as av;
            macro_rules! var_slot {
                ($scalar:path, $avx2:path) => {
                    if caps.avx2 {
                        $avx2 as VarFn
                    } else {
                        $scalar as VarFn
                    }
                };
            }
            // Order MUST match `BlockSize::idx()` (Block4x4 = 0 .. Block64x64 = 12).
            VarTable {
                fns: [
                    var_slot!(sc::var_sum_sse_4x4, av::var_sum_sse_4x4_avx2),
                    var_slot!(sc::var_sum_sse_4x8, av::var_sum_sse_4x8_avx2),
                    var_slot!(sc::var_sum_sse_8x4, av::var_sum_sse_8x4_avx2),
                    var_slot!(sc::var_sum_sse_8x8, av::var_sum_sse_8x8_avx2),
                    var_slot!(sc::var_sum_sse_8x16, av::var_sum_sse_8x16_avx2),
                    var_slot!(sc::var_sum_sse_16x8, av::var_sum_sse_16x8_avx2),
                    var_slot!(sc::var_sum_sse_16x16, av::var_sum_sse_16x16_avx2),
                    var_slot!(sc::var_sum_sse_16x32, av::var_sum_sse_16x32_avx2),
                    var_slot!(sc::var_sum_sse_32x16, av::var_sum_sse_32x16_avx2),
                    var_slot!(sc::var_sum_sse_32x32, av::var_sum_sse_32x32_avx2),
                    var_slot!(sc::var_sum_sse_32x64, av::var_sum_sse_32x64_avx2),
                    var_slot!(sc::var_sum_sse_64x32, av::var_sum_sse_64x32_avx2),
                    var_slot!(sc::var_sum_sse_64x64, av::var_sum_sse_64x64_avx2),
                ],
            }
        };
        #[cfg(target_arch = "aarch64")]
        let variance: VarTable = {
            use crate::dsp::aarch64 as neon;
            use crate::dsp::distortion as sc;
            macro_rules! var_slot {
                ($scalar:path, $neon:path, $dotprod:path) => {
                    if caps.dotprod {
                        $dotprod as VarFn
                    } else if caps.neon {
                        $neon as VarFn
                    } else {
                        $scalar as VarFn
                    }
                };
            }
            VarTable {
                fns: [
                    var_slot!(
                        sc::var_sum_sse_4x4,
                        sc::var_sum_sse_4x4,
                        neon::var_sum_sse_4x4_dotprod
                    ),
                    var_slot!(
                        sc::var_sum_sse_4x8,
                        sc::var_sum_sse_4x8,
                        neon::var_sum_sse_4x8_dotprod
                    ),
                    var_slot!(
                        sc::var_sum_sse_8x4,
                        neon::var_sum_sse_8x4_neon,
                        neon::var_sum_sse_8x4_dotprod
                    ),
                    var_slot!(
                        sc::var_sum_sse_8x8,
                        neon::var_sum_sse_8x8_neon,
                        neon::var_sum_sse_8x8_dotprod
                    ),
                    var_slot!(
                        sc::var_sum_sse_8x16,
                        neon::var_sum_sse_8x16_neon,
                        neon::var_sum_sse_8x16_dotprod
                    ),
                    var_slot!(
                        sc::var_sum_sse_16x8,
                        neon::var_sum_sse_16x8_neon,
                        neon::var_sum_sse_16x8_dotprod
                    ),
                    var_slot!(
                        sc::var_sum_sse_16x16,
                        neon::var_sum_sse_16x16_neon,
                        neon::var_sum_sse_16x16_dotprod
                    ),
                    var_slot!(
                        sc::var_sum_sse_16x32,
                        neon::var_sum_sse_16x32_neon,
                        neon::var_sum_sse_16x32_dotprod
                    ),
                    var_slot!(
                        sc::var_sum_sse_32x16,
                        neon::var_sum_sse_32x16_neon,
                        neon::var_sum_sse_32x16_dotprod
                    ),
                    var_slot!(
                        sc::var_sum_sse_32x32,
                        neon::var_sum_sse_32x32_neon,
                        neon::var_sum_sse_32x32_dotprod
                    ),
                    var_slot!(
                        sc::var_sum_sse_32x64,
                        neon::var_sum_sse_32x64_neon,
                        neon::var_sum_sse_32x64_dotprod
                    ),
                    var_slot!(
                        sc::var_sum_sse_64x32,
                        neon::var_sum_sse_64x32_neon,
                        neon::var_sum_sse_64x32_dotprod
                    ),
                    var_slot!(
                        sc::var_sum_sse_64x64,
                        neon::var_sum_sse_64x64_neon,
                        neon::var_sum_sse_64x64_dotprod
                    ),
                ],
            }
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let variance: VarTable = {
            let _ = caps;
            use crate::dsp::distortion as sc;
            VarTable {
                fns: [
                    sc::var_sum_sse_4x4 as VarFn,
                    sc::var_sum_sse_4x8,
                    sc::var_sum_sse_8x4,
                    sc::var_sum_sse_8x8,
                    sc::var_sum_sse_8x16,
                    sc::var_sum_sse_16x8,
                    sc::var_sum_sse_16x16,
                    sc::var_sum_sse_16x32,
                    sc::var_sum_sse_32x16,
                    sc::var_sum_sse_32x32,
                    sc::var_sum_sse_32x64,
                    sc::var_sum_sse_64x32,
                    sc::var_sum_sse_64x64,
                ],
            }
        };

        // ---- PR4 — per-8×8-cell (sum, sse) reduction slot. ----
        #[cfg(target_arch = "x86_64")]
        let get_var_sse_8x8: CellVarFn = if caps.avx2 {
            crate::dsp::x86_64::get_var_sse_8x8_avx2
        } else {
            crate::dsp::distortion::get_var_sse_8x8
        };
        #[cfg(target_arch = "aarch64")]
        let get_var_sse_8x8: CellVarFn = if caps.dotprod {
            crate::dsp::aarch64::get_var_sse_8x8_dotprod
        } else if caps.neon {
            crate::dsp::aarch64::get_var_sse_8x8_neon
        } else {
            crate::dsp::distortion::get_var_sse_8x8
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let get_var_sse_8x8: CellVarFn = {
            let _ = caps;
            crate::dsp::distortion::get_var_sse_8x8
        };

        // ---- PR4 — per-size sub-pixel variance table. ----
        // One clause per supported VP9 block size: the AVX2 safe wrapper on AVX2
        // hosts, the scalar `vpx_sub_pixel_variance{W}x{H}` reference otherwise.
        // The two sub-8 rect sizes (4×8, 8×4) hold `None`, preserving the
        // fail-soft `svf_for` semantics. Byte-exact across the two arms.
        #[cfg(target_arch = "x86_64")]
        let subpel_variance: SubpelVarTable = {
            use crate::dsp::distortion as sc;
            use crate::dsp::x86_64 as av;
            macro_rules! subpel_slot {
                ($scalar:path, $avx2:path) => {
                    Some(if caps.avx2 {
                        $avx2 as SubpelVarFn
                    } else {
                        $scalar as SubpelVarFn
                    })
                };
            }
            // Order MUST match `BlockSize::idx()` (Block4x4 = 0 .. Block64x64 = 12).
            // Indices 1 (4×8) and 2 (8×4) are unsupported → None.
            SubpelVarTable {
                fns: [
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance4x4,
                        av::vpx_sub_pixel_variance4x4_avx2
                    ),
                    None,
                    None,
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance8x8,
                        av::vpx_sub_pixel_variance8x8_avx2
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance8x16,
                        av::vpx_sub_pixel_variance8x16_avx2
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance16x8,
                        av::vpx_sub_pixel_variance16x8_avx2
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance16x16,
                        av::vpx_sub_pixel_variance16x16_avx2
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance16x32,
                        av::vpx_sub_pixel_variance16x32_avx2
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance32x16,
                        av::vpx_sub_pixel_variance32x16_avx2
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance32x32,
                        av::vpx_sub_pixel_variance32x32_avx2
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance32x64,
                        av::vpx_sub_pixel_variance32x64_avx2
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance64x32,
                        av::vpx_sub_pixel_variance64x32_avx2
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance64x64,
                        av::vpx_sub_pixel_variance64x64_avx2
                    ),
                ],
            }
        };
        #[cfg(target_arch = "aarch64")]
        let subpel_variance: SubpelVarTable = {
            use crate::dsp::aarch64 as neon;
            use crate::dsp::distortion as sc;
            macro_rules! subpel_slot {
                ($scalar:path, $neon:path, $dotprod:path) => {
                    Some(if caps.dotprod {
                        $dotprod as SubpelVarFn
                    } else if caps.neon {
                        $neon as SubpelVarFn
                    } else {
                        $scalar as SubpelVarFn
                    })
                };
            }
            SubpelVarTable {
                fns: [
                    Some(sc::vpx_sub_pixel_variance4x4 as SubpelVarFn),
                    None,
                    None,
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance8x8,
                        neon::vpx_sub_pixel_variance8x8_neon,
                        neon::vpx_sub_pixel_variance8x8_dotprod
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance8x16,
                        neon::vpx_sub_pixel_variance8x16_neon,
                        neon::vpx_sub_pixel_variance8x16_dotprod
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance16x8,
                        neon::vpx_sub_pixel_variance16x8_neon,
                        neon::vpx_sub_pixel_variance16x8_dotprod
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance16x16,
                        neon::vpx_sub_pixel_variance16x16_neon,
                        neon::vpx_sub_pixel_variance16x16_dotprod
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance16x32,
                        neon::vpx_sub_pixel_variance16x32_neon,
                        neon::vpx_sub_pixel_variance16x32_dotprod
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance32x16,
                        neon::vpx_sub_pixel_variance32x16_neon,
                        neon::vpx_sub_pixel_variance32x16_dotprod
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance32x32,
                        neon::vpx_sub_pixel_variance32x32_neon,
                        neon::vpx_sub_pixel_variance32x32_dotprod
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance32x64,
                        neon::vpx_sub_pixel_variance32x64_neon,
                        neon::vpx_sub_pixel_variance32x64_dotprod
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance64x32,
                        neon::vpx_sub_pixel_variance64x32_neon,
                        neon::vpx_sub_pixel_variance64x32_dotprod
                    ),
                    subpel_slot!(
                        sc::vpx_sub_pixel_variance64x64,
                        neon::vpx_sub_pixel_variance64x64_neon,
                        neon::vpx_sub_pixel_variance64x64_dotprod
                    ),
                ],
            }
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let subpel_variance: SubpelVarTable = {
            let _ = caps;
            use crate::dsp::distortion as sc;
            SubpelVarTable {
                fns: [
                    Some(sc::vpx_sub_pixel_variance4x4 as SubpelVarFn),
                    None,
                    None,
                    Some(sc::vpx_sub_pixel_variance8x8),
                    Some(sc::vpx_sub_pixel_variance8x16),
                    Some(sc::vpx_sub_pixel_variance16x8),
                    Some(sc::vpx_sub_pixel_variance16x16),
                    Some(sc::vpx_sub_pixel_variance16x32),
                    Some(sc::vpx_sub_pixel_variance32x16),
                    Some(sc::vpx_sub_pixel_variance32x32),
                    Some(sc::vpx_sub_pixel_variance32x64),
                    Some(sc::vpx_sub_pixel_variance64x32),
                    Some(sc::vpx_sub_pixel_variance64x64),
                ],
            }
        };

        // ---- G4.4b — scalar-only slot (deferral rationale on field doc). ----
        let extend_frame_borders: fn(&mut Yv12Buffer) = scalar_extend_frame_borders;

        Self {
            caps,
            idct8x8_add,
            idct8x8_12_add,
            idct4x4_add,
            highbd_idct4x4_add,
            idct16x16_add,
            idct32x32_add,
            idct32x32_34_add,
            idct32x32_135_add,
            idct16x16_10_add,
            idct16x16_38_add,
            dc_predictor_4x4,
            v_predictor,
            h_predictor,
            tm_predictor,
            d45_predictor,
            d63_predictor,
            d207_predictor,
            d117_predictor,
            d135_predictor,
            d153_predictor,
            dc_predict_full,
            dc_predict_top,
            dc_predict_left,
            dc_predict_128,
            iht4x4_add,
            iht8x8_add,
            iht16x16_add,
            lpf_vertical_4,
            lpf_horizontal_4,
            lpf_vertical_8,
            lpf_horizontal_8,
            lpf_vertical_16,
            lpf_horizontal_16,
            lpf_vertical_4_dual,
            lpf_horizontal_4_dual,
            lpf_vertical_8_dual,
            lpf_horizontal_8_dual,
            lpf_vertical_16_dual,
            lpf_horizontal_16_dual,
            plane_dequant_finisher,
            convolve_copy,
            convolve_avg,
            convolve8_horiz,
            convolve8_vert,
            convolve8,
            convolve8_avg_horiz,
            convolve8_avg_vert,
            convolve8_avg,
            convolve8_scaled,
            convolve8_scaled_avg,
            sad,
            sad4d,
            variance,
            get_var_sse_8x8,
            subpel_variance,
            extend_frame_borders,
        }
    }

    /// Dispatch the 8×8 inverse-DCT + add kernel. Returns the
    /// installed function pointer; the call site invokes it directly.
    ///
    /// Equivalent to `(dispatch.idct8x8_add)(input, dst, stride)` but
    /// the field is `pub(crate)` — external callers must go through
    /// this accessor.
    #[inline]
    #[must_use]
    pub fn idct8x8_add(&self) -> fn(&[i32; 64], &mut [u8], usize) {
        self.idct8x8_add
    }

    /// Dispatch the sparse 8×8 inverse-DCT + add kernel for the
    /// `eob <= 12` bucket (Task A). Returns the installed function
    /// pointer; the call site invokes it directly. The kernel exploits
    /// the bitstream's eob-12 zero-tail contract — rows 4..7 of `input`
    /// must be zero on entry.
    #[inline]
    #[must_use]
    pub fn idct8x8_12_add(&self) -> fn(&[i32; 64], &mut [u8], usize) {
        self.idct8x8_12_add
    }

    /// Dispatch the 4×4 inverse-DCT + add kernel (G4.2).
    #[inline]
    #[must_use]
    pub fn idct4x4_add(&self) -> fn(&[i32; 16], &mut [u8], usize) {
        self.idct4x4_add
    }

    /// Dispatch the HBD 4×4 inverse-DCT + add kernel (Track T phase 0
    /// scalar reference). The trailing `u32` is the bit depth
    /// (`bd ∈ {8, 10, 12}`). No SIMD HBD kernels in Track T phase 0 —
    /// the slot always points at
    /// [`crate::transforms::inverse::highbd_idct4x4_16_add`].
    ///
    /// **Phase 0 has no decoder call sites** for this kernel — the
    /// bit-depth gate in `rvp9-decoder` still rejects HBD frames before
    /// they reach the per-block transform path. Wired in phase 1.
    #[inline]
    #[must_use]
    pub fn highbd_idct4x4_add(&self) -> fn(&[i32; 16], &mut [u16], usize, u32) {
        self.highbd_idct4x4_add
    }

    /// Dispatch the 16×16 inverse-DCT + add kernel (G4.2 full-kernel
    /// rollout). Full-density `_256_add` variant only; the 10/38-bucket
    /// fast paths are not yet ported.
    #[inline]
    #[must_use]
    pub fn idct16x16_add(&self) -> fn(&[i32; 256], &mut [u8], usize) {
        self.idct16x16_add
    }

    /// Dispatch the 32×32 inverse-DCT + add kernel (G4-α full-kernel
    /// rollout). Full-density `_1024_add` variant only — the
    /// 34/135-bucket fast paths land in
    /// [`SimdDispatch::idct32x32_34_add`] /
    /// [`SimdDispatch::idct32x32_135_add`].
    #[inline]
    #[must_use]
    pub fn idct32x32_add(&self) -> fn(&[i32; 1024], &mut [u8], usize) {
        self.idct32x32_add
    }

    /// Dispatch the 32×32 inverse-DCT + add `eob <= 34` fast-path
    /// kernel (Wave F F9). AVX2 finisher-only, same Stage 3 as
    /// `idct32x32_add` but the row pass (Stage 1) runs over the first
    /// 8 rows only. Caller must respect the eob contract — passing a
    /// block whose true eob exceeds 34 silently drops coefficients
    /// beyond row 8 and diverges from the full kernel.
    #[inline]
    #[must_use]
    pub fn idct32x32_34_add(&self) -> fn(&[i32; 1024], &mut [u8], usize) {
        self.idct32x32_34_add
    }

    /// Dispatch the 32×32 inverse-DCT + add `eob <= 135` fast-path
    /// kernel (Wave F F9). AVX2 finisher-only, same Stage 3 as
    /// `idct32x32_add` but the row pass (Stage 1) runs over the first
    /// 16 rows only. Caller must respect the eob contract — passing a
    /// block whose true eob exceeds 135 silently drops coefficients
    /// beyond row 16 and diverges from the full kernel.
    #[inline]
    #[must_use]
    pub fn idct32x32_135_add(&self) -> fn(&[i32; 1024], &mut [u8], usize) {
        self.idct32x32_135_add
    }

    /// Dispatch the 16×16 inverse-DCT + add `eob <= 10` fast-path
    /// kernel (Wave F F3 — finisher-only AVX2 port). Caller must
    /// respect the eob contract — passing a block whose true eob
    /// exceeds 10 silently drops coefficients beyond row 4 and
    /// diverges from the full kernel.
    #[inline]
    #[must_use]
    pub fn idct16x16_10_add(&self) -> fn(&[i32; 256], &mut [u8], usize) {
        self.idct16x16_10_add
    }

    /// Dispatch the 16×16 inverse-DCT + add `eob <= 38` fast-path
    /// kernel (Wave F F4 — finisher-only AVX2 port). Caller must
    /// respect the eob contract — passing a block whose true eob
    /// exceeds 38 silently drops coefficients beyond row 8 and
    /// diverges from the full kernel.
    #[inline]
    #[must_use]
    pub fn idct16x16_38_add(&self) -> fn(&[i32; 256], &mut [u8], usize) {
        self.idct16x16_38_add
    }

    /// Dispatch the 4×4 hybrid inverse-transform + add kernel
    /// (Wave E G4-β — ADST/DCT combinations). The `tx_type` arg routes
    /// to the right row / col 1D pair.
    #[inline]
    #[must_use]
    pub fn iht4x4_add(&self) -> fn(&[i32; 16], &mut [u8], usize, TxType) {
        self.iht4x4_add
    }

    /// Dispatch the 8×8 hybrid inverse-transform + add kernel
    /// (Wave E G4-β — ADST/DCT combinations).
    #[inline]
    #[must_use]
    pub fn iht8x8_add(&self) -> fn(&[i32; 64], &mut [u8], usize, TxType) {
        self.iht8x8_add
    }

    /// Dispatch the 16×16 hybrid inverse-transform + add kernel
    /// (Wave F F5 — ADST/DCT combinations). The `tx_type` arg routes
    /// to the right row / col 1D pair.
    #[inline]
    #[must_use]
    pub fn iht16x16_add(&self) -> fn(&[i32; 256], &mut [u8], usize, TxType) {
        self.iht16x16_add
    }

    /// Dispatch the 4×4 DC intra-predictor (G4.3a). The slot uses the
    /// flat `(dst, stride, above, left)` signature; the scalar adapter
    /// wraps [`crate::intra_pred::dc_predictor`] (which requires a
    /// `PredCtx` — the adapter sets `above_left = 0` since DC_PRED
    /// ignores that field).
    #[inline]
    #[must_use]
    pub fn dc_predictor_4x4(&self) -> fn(&mut [u8], usize, &[u8], &[u8]) {
        self.dc_predictor_4x4
    }

    /// Dispatch the size-parametric V_PRED intra-predictor (Wave I I1).
    /// The slot uses the flat `(dst, stride, bs, above)` signature;
    /// the AVX2 wrapper dispatches internally on `bs ∈ {4, 8, 16, 32}`
    /// to the matching wide-store width. The scalar adapter wraps
    /// [`crate::intra_pred::v_predictor`] (which requires a `PredCtx`
    /// — the adapter sets `left = &[]` and `above_left = 0` since
    /// V_PRED ignores those fields).
    #[inline]
    #[must_use]
    pub fn v_predictor(&self) -> fn(&mut [u8], usize, usize, &[u8]) {
        self.v_predictor
    }

    /// Dispatch the size-parametric H_PRED intra-predictor (Wave I I2).
    /// The slot uses the flat `(dst, stride, bs, left)` signature;
    /// the AVX2 wrapper dispatches internally on `bs ∈ {4, 8, 16, 32}`
    /// to the matching wide-store width. The scalar adapter wraps
    /// [`crate::intra_pred::h_predictor`] (which requires a `PredCtx`
    /// — the adapter sets `above = &[]` and `above_left = 0` since
    /// H_PRED ignores those fields).
    #[inline]
    #[must_use]
    pub fn h_predictor(&self) -> fn(&mut [u8], usize, usize, &[u8]) {
        self.h_predictor
    }

    /// Dispatch the TM_PRED ("true motion") intra-predictor (Wave I I3).
    /// Size-parametric on `bs ∈ {4, 8, 16, 32}`. The slot uses the
    /// flat `(dst, stride, bs, above, left, above_left)` signature;
    /// the scalar adapter wraps [`crate::intra_pred::tm_predictor`]
    /// (which takes a `PredCtx`).
    #[inline]
    #[must_use]
    pub fn tm_predictor(&self) -> TmPredictorFn {
        self.tm_predictor
    }

    /// Dispatch the D45_PRED directional intra-predictor (Wave K K1).
    /// Flat `(dst, stride, bs, above)` signature; routes the `bs == 4`
    /// libvpx specialization internally.
    #[inline]
    #[must_use]
    pub fn d45_predictor(&self) -> fn(&mut [u8], usize, usize, &[u8]) {
        self.d45_predictor
    }

    /// Dispatch the D63_PRED directional intra-predictor (Wave K K3).
    #[inline]
    #[must_use]
    pub fn d63_predictor(&self) -> fn(&mut [u8], usize, usize, &[u8]) {
        self.d63_predictor
    }

    /// Dispatch the D207_PRED directional intra-predictor (Wave K K2).
    /// Flat `(dst, stride, bs, left)` signature.
    #[inline]
    #[must_use]
    pub fn d207_predictor(&self) -> fn(&mut [u8], usize, usize, &[u8]) {
        self.d207_predictor
    }

    /// Dispatch the D117_PRED directional intra-predictor (Wave K K5).
    /// Flat `(dst, stride, bs, above, left, above_left)` signature.
    #[inline]
    #[must_use]
    pub fn d117_predictor(&self) -> DirectionalCtxPredFn {
        self.d117_predictor
    }

    /// Dispatch the D135_PRED directional intra-predictor (Wave K K4).
    #[inline]
    #[must_use]
    pub fn d135_predictor(&self) -> DirectionalCtxPredFn {
        self.d135_predictor
    }

    /// Dispatch the D153_PRED directional intra-predictor (Wave K K6).
    #[inline]
    #[must_use]
    pub fn d153_predictor(&self) -> DirectionalCtxPredFn {
        self.d153_predictor
    }

    /// Dispatch the size-parametric DC_PRED (full neighbours) predictor
    /// (Wave I I4). The slot uses the flat
    /// `(dst, stride, bs, above, left)` signature; the AVX2 wrapper
    /// dispatches internally on `bs ∈ {4, 8, 16, 32}` to the matching
    /// SAD-sum + wide-store width. The scalar adapter wraps
    /// [`crate::intra_pred::dc_predictor`] (which requires a `PredCtx`
    /// — the adapter sets `above_left = 0` since DC_PRED ignores that
    /// field).
    #[inline]
    #[must_use]
    pub fn dc_predict_full(&self) -> fn(&mut [u8], usize, usize, &[u8], &[u8]) {
        self.dc_predict_full
    }

    /// Dispatch the size-parametric DC_PRED (above-only) predictor
    /// (Wave I I4). The slot uses the flat
    /// `(dst, stride, bs, above)` signature, identical to
    /// [`crate::intra_pred::dc_top_predictor`]. AVX2 wrapper on
    /// caps.avx2, scalar reference otherwise.
    #[inline]
    #[must_use]
    pub fn dc_predict_top(&self) -> fn(&mut [u8], usize, usize, &[u8]) {
        self.dc_predict_top
    }

    /// Dispatch the size-parametric DC_PRED (left-only) predictor
    /// (Wave I I4). The slot uses the flat
    /// `(dst, stride, bs, left)` signature, identical to
    /// [`crate::intra_pred::dc_left_predictor`]. AVX2 wrapper on
    /// caps.avx2, scalar reference otherwise.
    #[inline]
    #[must_use]
    pub fn dc_predict_left(&self) -> fn(&mut [u8], usize, usize, &[u8]) {
        self.dc_predict_left
    }

    /// Dispatch the size-parametric DC_128 fallback predictor (Wave I
    /// I4). Fills the `bs × bs` block with constant 128 when neither
    /// neighbour is available. AVX2 wrapper on caps.avx2, scalar
    /// reference otherwise.
    #[inline]
    #[must_use]
    pub fn dc_predict_128(&self) -> fn(&mut [u8], usize, usize) {
        self.dc_predict_128
    }

    /// Dispatch the vertical 4-tap loopfilter kernel (G4.3b / Wave F
    /// G F10). On AVX2 x86_64 hosts this routes to the lane-parallel
    /// transpose-based wrapper; otherwise to the scalar reference.
    /// See field doc on [`SimdDispatch::lpf_vertical_4`].
    #[inline]
    #[must_use]
    pub fn lpf_vertical_4(&self) -> fn(&mut [u8], usize, usize, u8, u8, u8) {
        self.lpf_vertical_4
    }

    /// Dispatch the horizontal 4-tap loopfilter kernel (Wave G G3).
    /// On AVX2 x86_64 hosts this routes to the no-transpose
    /// lane-parallel wrapper; otherwise to the scalar reference. See
    /// field doc on [`SimdDispatch::lpf_horizontal_4`].
    #[inline]
    #[must_use]
    pub fn lpf_horizontal_4(&self) -> fn(&mut [u8], usize, usize, u8, u8, u8) {
        self.lpf_horizontal_4
    }

    /// Dispatch the 8-tap vertical loopfilter (Wave G G4). On AVX2
    /// x86_64 hosts this routes to the transpose+filter8 wrapper;
    /// otherwise to scalar. See field doc.
    #[inline]
    #[must_use]
    pub fn lpf_vertical_8(&self) -> fn(&mut [u8], usize, usize, u8, u8, u8) {
        self.lpf_vertical_8
    }

    /// Dispatch the 8-tap horizontal loopfilter (Wave G G5). On AVX2
    /// x86_64 hosts this routes to the no-transpose row-load wrapper
    /// with the same filter8 inner as G4; otherwise to scalar.
    #[inline]
    #[must_use]
    pub fn lpf_horizontal_8(&self) -> fn(&mut [u8], usize, usize, u8, u8, u8) {
        self.lpf_horizontal_8
    }

    /// Dispatch the 16-tap vertical edge loopfilter (Wave G G6). On
    /// AVX2 x86_64 hosts this routes to the transpose + filter16
    /// cascade wrapper; otherwise to scalar. See field doc.
    #[inline]
    #[must_use]
    pub fn lpf_vertical_16(&self) -> fn(&mut [u8], usize, usize, u8, u8, u8) {
        self.lpf_vertical_16
    }

    /// Dispatch the 16-tap horizontal edge loopfilter (Wave G G6). On
    /// AVX2 x86_64 hosts this routes to the no-transpose row-load
    /// wrapper with the same filter16 cascade as the G6 vertical
    /// sibling; otherwise to scalar. See field doc.
    #[inline]
    #[must_use]
    pub fn lpf_horizontal_16(&self) -> fn(&mut [u8], usize, usize, u8, u8, u8) {
        self.lpf_horizontal_16
    }

    /// Dispatch the dual 4-tap vertical loopfilter (Wave H H1). Scalar-only
    /// at H1 close; H2 fills with AVX2.
    #[inline]
    #[must_use]
    pub fn lpf_vertical_4_dual(&self) -> LpfDualFn {
        self.lpf_vertical_4_dual
    }

    /// Dispatch the dual 4-tap horizontal loopfilter (Wave H H1). Scalar-only
    /// at H1 close; H2 fills with AVX2.
    #[inline]
    #[must_use]
    pub fn lpf_horizontal_4_dual(&self) -> LpfDualFn {
        self.lpf_horizontal_4_dual
    }

    /// Dispatch the dual 8-tap vertical loopfilter (Wave H H1). Scalar-only
    /// at H1 close; H3 fills with AVX2.
    #[inline]
    #[must_use]
    pub fn lpf_vertical_8_dual(&self) -> LpfDualFn {
        self.lpf_vertical_8_dual
    }

    /// Dispatch the dual 8-tap horizontal loopfilter (Wave H H1). Scalar-only
    /// at H1 close; H3 fills with AVX2.
    #[inline]
    #[must_use]
    pub fn lpf_horizontal_8_dual(&self) -> LpfDualFn {
        self.lpf_horizontal_8_dual
    }

    /// Dispatch the dual 16-tap vertical loopfilter (Wave H H1). Six-arg
    /// (single threshold set shared across both halves; libvpx convention).
    /// Scalar-only at H1 close; H4 fills with AVX2.
    #[inline]
    #[must_use]
    pub fn lpf_vertical_16_dual(&self) -> fn(&mut [u8], usize, usize, u8, u8, u8) {
        self.lpf_vertical_16_dual
    }

    /// Dispatch the dual 16-tap horizontal loopfilter (Wave H H1). Six-arg
    /// (single threshold set shared across both halves). Scalar-only at
    /// H1 close; H4 fills with AVX2.
    #[inline]
    #[must_use]
    pub fn lpf_horizontal_16_dual(&self) -> fn(&mut [u8], usize, usize, u8, u8, u8) {
        self.lpf_horizontal_16_dual
    }

    /// Dispatch the pass-2 dequant finisher (Wave H H6). See field
    /// doc on [`SimdDispatch::plane_dequant_finisher`]. AVX2 wrapper
    /// when `caps.avx2`, scalar reference otherwise.
    #[inline]
    #[must_use]
    pub fn plane_dequant_finisher(&self) -> PlaneDequantFinisherFn {
        self.plane_dequant_finisher
    }

    /// Dispatch the integer-pel block copy kernel. Uses AVX2 on capable
    /// x86-64 hosts, NEON on capable AArch64 hosts, and scalar otherwise.
    #[inline]
    #[must_use]
    pub fn convolve_copy(&self) -> fn(&[u8], usize, &mut [u8], usize, usize, usize) {
        self.convolve_copy
    }

    /// Dispatch the integer-pel compound-average kernel (Wave I I5).
    /// Scalar only at I5; AVX2 promotion deferred to Wave J. See field
    /// doc on [`SimdDispatch::convolve_avg`].
    #[inline]
    #[must_use]
    pub fn convolve_avg(&self) -> fn(&[u8], usize, &mut [u8], usize, usize, usize) {
        self.convolve_avg
    }

    /// Dispatch the 8-tap horizontal subpel convolution (Wave I I5).
    /// Scalar only at I5; AVX2 promotion deferred to Wave J.
    #[inline]
    #[must_use]
    pub fn convolve8_horiz(&self) -> Convolve8OneAxisFn {
        self.convolve8_horiz
    }

    /// Dispatch the 8-tap vertical subpel convolution (Wave I I5).
    /// Scalar only at I5; AVX2 promotion deferred to Wave J.
    #[inline]
    #[must_use]
    pub fn convolve8_vert(&self) -> Convolve8OneAxisFn {
        self.convolve8_vert
    }

    /// Dispatch the 8-tap 2-D subpel convolution (Wave I I5). Scalar
    /// only at I5; AVX2 promotion deferred to Wave J.
    #[inline]
    #[must_use]
    pub fn convolve8(&self) -> Convolve8TwoAxisFn {
        self.convolve8
    }

    /// Dispatch the 8-tap horizontal subpel convolution with compound
    /// average (Wave I I5). Scalar only at I5; AVX2 promotion deferred
    /// to Wave J.
    #[inline]
    #[must_use]
    pub fn convolve8_avg_horiz(&self) -> Convolve8OneAxisFn {
        self.convolve8_avg_horiz
    }

    /// Dispatch the 8-tap vertical subpel convolution with compound
    /// average (Wave I I5). Scalar only at I5; AVX2 promotion deferred
    /// to Wave J.
    #[inline]
    #[must_use]
    pub fn convolve8_avg_vert(&self) -> Convolve8OneAxisFn {
        self.convolve8_avg_vert
    }

    /// Dispatch the 8-tap 2-D subpel convolution with compound average
    /// (Wave I I5). Scalar only at I5; AVX2 promotion deferred to
    /// Wave J.
    #[inline]
    #[must_use]
    pub fn convolve8_avg(&self) -> Convolve8TwoAxisFn {
        self.convolve8_avg
    }

    /// Dispatch the scaled 2-D sub-pel convolution (Wave L L1). AVX2 on
    /// x86_64 when `caps.avx2`; scalar elsewhere.
    #[inline]
    #[must_use]
    pub fn convolve8_scaled(&self) -> ScaledConvolveFn {
        self.convolve8_scaled
    }

    /// Dispatch the scaled 2-D sub-pel convolution with compound average
    /// (Wave L L1). AVX2 on x86_64 when `caps.avx2`; scalar elsewhere.
    #[inline]
    #[must_use]
    pub fn convolve8_scaled_avg(&self) -> ScaledConvolveFn {
        self.convolve8_scaled_avg
    }

    /// Dispatch the frame-border edge-replication kernel (G4.4b). Slot
    /// is currently scalar-only — see field doc on
    /// [`SimdDispatch::extend_frame_borders`].
    #[inline]
    #[must_use]
    pub fn extend_frame_borders(&self) -> fn(&mut Yv12Buffer) {
        self.extend_frame_borders
    }

    /// Resolve the per-size SAD kernel for `bsize` (P1-B). Returns the AVX2
    /// wrapper on AVX2 hosts and the scalar `vpx_sad{W}x{H}` reference
    /// otherwise; both are byte-exact. Resolve once per search call and
    /// reuse across the inner SAD loop (the table is a single fixed array,
    /// so this is a plain index — but hoisting `get_ref()` out of the hot
    /// loop still amortizes the OnceLock load).
    #[inline]
    #[must_use]
    pub fn sad(&self, bsize: crate::block::BlockSize) -> SadFn {
        self.sad.get(bsize)
    }

    /// Resolve the per-size 4-reference SAD kernel for `bsize` (PR3). Returns
    /// the AVX2 wrapper on AVX2 hosts and the scalar four-`sad` reference
    /// otherwise; both are byte-exact, and each output lane is byte-exact with
    /// the single-SAD [`SimdDispatch::sad`] slot. Resolve once per search call
    /// and reuse across the `all_in` step loop so the `OnceLock` load in
    /// `get_ref()` is amortized rather than paid per step.
    #[inline]
    #[must_use]
    pub fn sad4d(&self, bsize: crate::block::BlockSize) -> Sad4dFn {
        self.sad4d.get(bsize)
    }

    /// Resolve the per-size block `(sum, sse)` reduction kernel for `bsize`
    /// (PR4). Returns AVX2 on x86_64, dot-product or base NEON for widths 8–64
    /// on AArch64, and the scalar `var_sum_sse_{W}x{H}` reference otherwise;
    /// all are byte-exact. The
    /// caller (the inter RD model) applies its own scalar variance finisher.
    /// Resolve once and reuse across the RD-model call so the `OnceLock` load
    /// in `get_ref()` is amortized.
    #[inline]
    #[must_use]
    pub fn variance(&self, bsize: crate::block::BlockSize) -> VarFn {
        self.variance.get(bsize)
    }

    /// Resolve the per-8×8-cell `(sum, sse)` reduction kernel (PR4). Returns the
    /// AVX2 wrapper on x86_64, dot-product or base NEON on AArch64, and the
    /// scalar `get_var_sse_8x8` reference otherwise; all are byte-exact. The caller
    /// (the large-block RD model)
    /// applies its own scalar abs-diff `>>6` finisher per cell. Resolve once and
    /// reuse across the cell grid.
    #[inline]
    #[must_use]
    pub fn get_var_sse_8x8(&self) -> CellVarFn {
        self.get_var_sse_8x8
    }

    /// Resolve the per-size sub-pixel-variance kernel for `bsize` (PR4), or
    /// `None` for the two unsupported sub-8 rect sizes (4×8, 8×4). Returns the
    /// best available ISA wrapper (AVX2 or NEON) and the scalar
    /// `vpx_sub_pixel_variance{W}x{H}` reference otherwise; all are
    /// byte-exact. Resolve once per sub-pel search and reuse so the `OnceLock`
    /// load in `get_ref()` is amortized.
    #[inline]
    #[must_use]
    pub fn subpel_variance(&self, bsize: crate::block::BlockSize) -> Option<SubpelVarFn> {
        self.subpel_variance.get(bsize)
    }
}

/// The single process-wide default dispatch table, lazily initialised.
///
/// Production callers get the same `Arc<SimdDispatch>` clone from
/// every [`init`] / [`get`] call. Tests that need an alternative caps
/// configuration construct their own `Arc::new(SimdDispatch::build(...))`
/// independent of this OnceLock.
static DISPATCH: OnceLock<Arc<SimdDispatch>> = OnceLock::new();

/// Initialise (or retrieve) the process-wide default dispatch table.
///
/// Idempotent: concurrent first-calls linearise via `OnceLock`. CPU
/// caps don't change at runtime so the resulting table is
/// deterministic for a given binary on a given CPU.
///
/// Returns an `Arc<SimdDispatch>` clone. Cheap (one atomic increment).
///
/// **`Encoder::new` and `Decoder::new` MUST call this as their first
/// action** (HLD-simd §2.2 / R-09). The returned `Arc<SimdDispatch>`
/// is what gets passed to `WorkerPool::new`, statically enforcing the
/// init-before-spawn invariant.
#[inline]
#[must_use]
pub fn init() -> Arc<SimdDispatch> {
    DISPATCH
        .get_or_init(|| Arc::new(SimdDispatch::build(detect())))
        .clone()
}

/// Get a `&'static` reference to the process-wide dispatch table —
/// no Arc clone, no atomic ops. Use for read-only call sites in
/// the per-block / per-MC decode hot path where the dispatch table
/// is only inspected (not shared across threads or moved).
///
/// Internally goes through the same `OnceLock` as [`init`] so the
/// init-before-use invariant still holds; `init_dispatch_inner` is
/// idempotent.
///
/// R.13 (2026-05-23): added to replace `Arc::clone` calls in the
/// inter MC path (called ~24k times per 1080p frame). Each Arc
/// clone is an atomic increment + matching destructor decrement;
/// `get_ref` is a single OnceLock atomic load returning a borrowed
/// pointer. Bit-exact.
#[inline]
#[must_use]
pub fn get_ref() -> &'static SimdDispatch {
    let arc: &'static Arc<SimdDispatch> =
        DISPATCH.get_or_init(|| Arc::new(SimdDispatch::build(detect())));
    arc.as_ref()
}

/// Get the process-wide default dispatch table. Alias for [`init`];
/// named separately as a documentation hook for "read-after-init"
/// call sites (e.g. inside a decoder method that knows `init` has
/// already run).
#[inline]
#[must_use]
pub fn get() -> Arc<SimdDispatch> {
    init()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `init()` returns clones of the same underlying `Arc` on every
    /// call.
    #[test]
    fn init_is_idempotent() {
        let a = init();
        let b = init();
        // Both clones share one allocation: pointer equality on
        // `Arc::as_ptr` proves it.
        assert!(Arc::ptr_eq(&a, &b));
    }

    /// `get()` is an alias for `init()`.
    #[test]
    fn get_matches_init() {
        let a = init();
        let b = get();
        assert!(Arc::ptr_eq(&a, &b));
    }

    /// `Arc<SimdDispatch>` is `Send + Sync` — what tile workers rely
    /// on for the R-09 invariant.
    #[test]
    fn dispatch_arc_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<SimdDispatch>>();
    }

    /// Building with explicit caps lets tests force a SIMD tier without
    /// changing the running CPU.
    #[test]
    fn explicit_build() {
        let caps = detect();
        let d = SimdDispatch::build(caps);
        assert_eq!(d.caps, caps);
    }

    /// Forcing `sse2 = false` falls back to the scalar reference for
    /// the idct8x8 slot. Verified by function-pointer identity.
    ///
    /// **Skipped under miri**: miri does not preserve `fn` pointer
    /// identity across crate boundaries — the same source-level
    /// `fn`, when coerced to `fn(...)` via different call sites, can
    /// resolve to different miri-internal "alloc IDs" and the `as usize`
    /// cast yields distinct values. This is a known miri limitation
    /// (see rust-lang/miri#2123), not a bug in this dispatch slot.
    /// The native test on stable / nightly continues to assert the
    /// invariant; this attribute only suppresses spurious miri failures.
    /// Applied identically to every fn-pointer-identity test below.
    ///
    /// **Note:** the gate switched from `caps.avx2` to `caps.sse2` when
    /// the i16-lane SSE2 dense kernel `idct8x8_add_sse2_64` was wired in
    /// (libvpx `vpx_idct8x8_64_add_sse2` port). The scalar fallback is
    /// only selected when SSE2 is also unavailable (which on x86_64 is
    /// only reachable via forced caps in tests; SSE2 is architecturally
    /// universal).
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_idct8x8() {
        let mut caps = detect();
        caps.sse2 = false;
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 64], &mut [u8], usize) = crate::transforms::inverse::idct8x8_64_add;
        assert_eq!(
            d.idct8x8_add() as usize,
            scalar as usize,
            "sse2=false must install the scalar idct8x8_64_add"
        );
    }

    /// AArch64 selects the dense NEON 8x8 inverse-DCT kernel only when
    /// runtime capability detection reports NEON.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_caps_select_dense_idct8x8_kernel() {
        let mut caps = detect();
        caps.neon = false;
        let scalar_dispatch = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 64], &mut [u8], usize) = crate::transforms::inverse::idct8x8_64_add;
        assert_eq!(scalar_dispatch.idct8x8_add() as usize, scalar as usize);

        caps.neon = true;
        let neon_dispatch = SimdDispatch::build(caps);
        let neon: fn(&[i32; 64], &mut [u8], usize) = crate::dsp::aarch64::idct8x8_add_neon;
        assert_eq!(neon_dispatch.idct8x8_add() as usize, neon as usize);
        assert_ne!(neon_dispatch.idct8x8_add() as usize, scalar as usize);
    }

    /// AArch64 selects the sparse NEON 8x8 inverse-DCT kernel only when
    /// runtime capability detection reports NEON.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_caps_select_sparse_idct8x8_12_kernel() {
        let mut caps = detect();
        caps.neon = false;
        let scalar_dispatch = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 64], &mut [u8], usize) = crate::transforms::inverse::idct8x8_12_add;
        assert_eq!(scalar_dispatch.idct8x8_12_add() as usize, scalar as usize);

        caps.neon = true;
        let neon_dispatch = SimdDispatch::build(caps);
        let neon: fn(&[i32; 64], &mut [u8], usize) = crate::dsp::aarch64::idct8x8_12_add_neon;
        assert_eq!(neon_dispatch.idct8x8_12_add() as usize, neon as usize);
        assert_ne!(neon_dispatch.idct8x8_12_add() as usize, scalar as usize);
    }

    /// AArch64 selects the dense NEON 8×8 hybrid inverse-transform kernel
    /// only when runtime capability detection reports NEON.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_caps_select_hybrid_iht8x8_kernel() {
        let mut caps = detect();
        caps.neon = false;
        let scalar_dispatch = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 64], &mut [u8], usize, TxType) =
            crate::transforms::inverse::iht8x8_64_add;
        assert_eq!(scalar_dispatch.iht8x8_add() as usize, scalar as usize);

        caps.neon = true;
        let neon_dispatch = SimdDispatch::build(caps);
        let neon: fn(&[i32; 64], &mut [u8], usize, TxType) = crate::dsp::aarch64::iht8x8_add_neon;
        assert_eq!(neon_dispatch.iht8x8_add() as usize, neon as usize);
        assert_ne!(neon_dispatch.iht8x8_add() as usize, scalar as usize);
    }

    /// Forcing `sse2 = false` falls back to the scalar reference for
    /// the Task A idct8x8_12 sparse slot. See
    /// `forced_scalar_caps_installs_scalar_idct8x8` for the miri-skip
    /// rationale.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_idct8x8_12() {
        let mut caps = detect();
        caps.sse2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 64], &mut [u8], usize) = crate::transforms::inverse::idct8x8_12_add;
        assert_eq!(
            d.idct8x8_12_add() as usize,
            scalar as usize,
            "sse2=false must install the scalar idct8x8_12_add"
        );
    }

    /// Forcing `sse2 = true` installs the SSE2 wrapper for the Task A
    /// idct8x8_12 sparse slot. SSE2 is universally present on x86_64,
    /// so this is the production code path. See
    /// `forced_scalar_caps_installs_scalar_idct8x8` for the miri-skip
    /// rationale.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn sse2_caps_installs_sse2_idct8x8_12() {
        let mut caps = detect();
        caps.sse2 = true;
        let d = SimdDispatch::build(caps);
        let sse2: fn(&[i32; 64], &mut [u8], usize) = crate::dsp::x86_64::idct8x8_add_sse2_12;
        let scalar: fn(&[i32; 64], &mut [u8], usize) = crate::transforms::inverse::idct8x8_12_add;
        assert_eq!(
            d.idct8x8_12_add() as usize,
            sse2 as usize,
            "sse2=true must install idct8x8_add_sse2_12"
        );
        assert_ne!(
            d.idct8x8_12_add() as usize,
            scalar as usize,
            "sse2=true must not install the scalar idct8x8_12_add"
        );
    }

    /// Forcing `avx2 = false` falls back to the scalar reference for
    /// the G4.2 idct4x4 slot. See `forced_scalar_caps_installs_scalar_idct8x8`
    /// for the miri-skip rationale.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_idct4x4() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 16], &mut [u8], usize) = crate::transforms::inverse::idct4x4_16_add;
        assert_eq!(
            d.idct4x4_add() as usize,
            scalar as usize,
            "avx2=false must install the scalar idct4x4_16_add"
        );
    }

    /// Forcing `avx2 = false, sse2 = true` installs the SSE2 i16
    /// kernel for the idct16x16 slot (Track-T T-4 first slice). See
    /// `forced_scalar_caps_installs_scalar_idct8x8` for the miri-skip
    /// rationale.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn sse2_caps_installs_sse2_idct16x16() {
        let mut caps = detect();
        caps.avx2 = false;
        caps.sse2 = true;
        let d = SimdDispatch::build(caps);
        let sse2: fn(&[i32; 256], &mut [u8], usize) = crate::dsp::x86_64::idct16x16_add_sse2_256;
        assert_eq!(
            d.idct16x16_add() as usize,
            sse2 as usize,
            "avx2=false, sse2=true must install idct16x16_add_sse2_256"
        );
    }

    /// Track T phase 0 — the HBD `idct4x4_add` slot always installs the
    /// scalar reference regardless of caps (no SIMD HBD kernels in this
    /// track). Sanity-checks that the slot is wired and dispatches to
    /// the libvpx-mirroring scalar implementation.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn highbd_idct4x4_add_slot_is_scalar_reference() {
        let caps = detect();
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 16], &mut [u16], usize, u32) =
            crate::transforms::inverse::highbd_idct4x4_16_add;
        assert_eq!(
            d.highbd_idct4x4_add() as usize,
            scalar as usize,
            "highbd_idct4x4_add slot must point at the scalar reference"
        );
    }

    /// Forcing `avx2 = false, sse2 = false` falls back to the scalar
    /// reference for the idct16x16 slot.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_idct16x16() {
        let mut caps = detect();
        caps.avx2 = false;
        caps.sse2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 256], &mut [u8], usize) =
            crate::transforms::inverse::idct16x16_256_add;
        assert_eq!(
            d.idct16x16_add() as usize,
            scalar as usize,
            "avx2=false, sse2=false must install the scalar idct16x16_256_add"
        );
    }

    /// Forcing `avx2 = false, sse2 = false` falls back to the scalar
    /// reference for the dense idct32x32 slot. (The AVX2 i16-lane kernel
    /// is the preferred path on x86_64, then SSE2 i16-lane, then scalar
    /// — see dispatch wiring + `idct32x32_dense_slot_prefers_avx2`.)
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn forced_scalar_caps_installs_scalar_idct32x32() {
        let mut caps = detect();
        caps.avx2 = false;
        caps.sse2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 1024], &mut [u8], usize) =
            crate::transforms::inverse::idct32x32_1024_add;
        assert_eq!(
            d.idct32x32_add() as usize,
            scalar as usize,
            "avx2=false, sse2=false must install the scalar idct32x32_1024_add"
        );
    }

    /// The dense 32×32 slot prefers the AVX2 i16-lane kernel when AVX2
    /// is present, falls back to the SSE2 i16-lane kernel on SSE2-only
    /// CPUs (AVX2 forced off), and the scalar otherwise. Mirrors the
    /// `if caps.avx2 { avx2 } else if caps.sse2 { sse2 } else { scalar }`
    /// dispatch chain.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn idct32x32_dense_slot_prefers_avx2() {
        let avx2: fn(&[i32; 1024], &mut [u8], usize) = crate::dsp::x86_64::idct32x32_add_avx2_1024;
        let sse2: fn(&[i32; 1024], &mut [u8], usize) = crate::dsp::x86_64::idct32x32_add_sse2_1024;

        // avx2=true → AVX2 i16-lane kernel.
        let mut caps_avx2 = detect();
        caps_avx2.avx2 = true;
        caps_avx2.sse2 = true;
        let d_avx2 = SimdDispatch::build(caps_avx2);
        assert_eq!(
            d_avx2.idct32x32_add() as usize,
            avx2 as usize,
            "avx2=true must install the AVX2 i16-lane idct32x32_add_avx2_1024"
        );

        // avx2=false, sse2=true → SSE2 i16-lane kernel.
        let mut caps_sse2 = detect();
        caps_sse2.avx2 = false;
        caps_sse2.sse2 = true;
        let d_sse2 = SimdDispatch::build(caps_sse2);
        assert_eq!(
            d_sse2.idct32x32_add() as usize,
            sse2 as usize,
            "avx2=false, sse2=true must install the SSE2 i16-lane idct32x32_add_sse2_1024"
        );
    }

    /// Forcing `sse2 = false` falls back to the scalar reference for
    /// the Wave F F9 `idct32x32_34_add` slot (post i16-lane SSE2 flip).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn forced_scalar_caps_installs_scalar_idct32x32_34() {
        let mut caps = detect();
        caps.sse2 = false;
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 1024], &mut [u8], usize) =
            crate::transforms::inverse::idct32x32_34_add;
        assert_eq!(
            d.idct32x32_34_add() as usize,
            scalar as usize,
            "sse2=false must install the scalar idct32x32_34_add"
        );
    }

    /// Forcing `sse2 = false` falls back to the scalar reference for
    /// the Wave F F9 `idct32x32_135_add` slot (post i16-lane SSE2 flip).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn forced_scalar_caps_installs_scalar_idct32x32_135() {
        let mut caps = detect();
        caps.sse2 = false;
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 1024], &mut [u8], usize) =
            crate::transforms::inverse::idct32x32_135_add;
        assert_eq!(
            d.idct32x32_135_add() as usize,
            scalar as usize,
            "sse2=false must install the scalar idct32x32_135_add"
        );
    }

    /// The 32×32 EOB-bucket slots route through the SSE2 i16-lane
    /// safe wrappers on SSE2-only CPUs (AVX2 forced off), scalar when
    /// neither is present. (The EOB-135 slot prefers the AVX2 i16-lane
    /// kernel when AVX2 is present — see
    /// `idct32x32_135_slot_prefers_avx2`. The EOB-34 slot has no AVX2
    /// branch, so it stays SSE2 whenever `caps.sse2`.)
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn idct32x32_eob_bucket_slots_dispatch_correctly() {
        // sse2=false, avx2=false → scalar fn pointers.
        let mut caps = detect();
        caps.sse2 = false;
        caps.avx2 = false;
        let d_scalar = SimdDispatch::build(caps);
        let scalar_34: fn(&[i32; 1024], &mut [u8], usize) =
            crate::transforms::inverse::idct32x32_34_add;
        let scalar_135: fn(&[i32; 1024], &mut [u8], usize) =
            crate::transforms::inverse::idct32x32_135_add;
        assert_eq!(d_scalar.idct32x32_34_add() as usize, scalar_34 as usize);
        assert_eq!(d_scalar.idct32x32_135_add() as usize, scalar_135 as usize);

        // sse2=true, avx2=false → SSE2 wrappers (NOT the scalar pointers).
        let sse2_34: fn(&[i32; 1024], &mut [u8], usize) = crate::dsp::x86_64::idct32x32_add_sse2_34;
        let sse2_135: fn(&[i32; 1024], &mut [u8], usize) =
            crate::dsp::x86_64::idct32x32_add_sse2_135;
        let mut caps_sse2 = detect();
        caps_sse2.sse2 = true;
        caps_sse2.avx2 = false;
        let d_sse2 = SimdDispatch::build(caps_sse2);
        assert_eq!(
            d_sse2.idct32x32_34_add() as usize,
            sse2_34 as usize,
            "sse2=true must install the SSE2 wrapper for idct32x32_34_add"
        );
        assert_eq!(
            d_sse2.idct32x32_135_add() as usize,
            sse2_135 as usize,
            "sse2=true, avx2=false must install the SSE2 wrapper for idct32x32_135_add"
        );
    }

    /// The EOB-135 32×32 slot prefers the AVX2 i16-lane partial kernel
    /// when AVX2 is present, falls back to the SSE2 i16-lane partial
    /// kernel on SSE2-only CPUs (AVX2 forced off), and the scalar
    /// otherwise. Mirrors the dense slot's
    /// `idct32x32_dense_slot_prefers_avx2`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn idct32x32_135_slot_prefers_avx2() {
        let avx2: fn(&[i32; 1024], &mut [u8], usize) = crate::dsp::x86_64::idct32x32_add_avx2_135;
        let sse2: fn(&[i32; 1024], &mut [u8], usize) = crate::dsp::x86_64::idct32x32_add_sse2_135;
        let scalar: fn(&[i32; 1024], &mut [u8], usize) =
            crate::transforms::inverse::idct32x32_135_add;

        // avx2=true → AVX2 i16-lane partial kernel.
        let mut caps_avx2 = detect();
        caps_avx2.avx2 = true;
        caps_avx2.sse2 = true;
        let d_avx2 = SimdDispatch::build(caps_avx2);
        assert_eq!(
            d_avx2.idct32x32_135_add() as usize,
            avx2 as usize,
            "avx2=true must install the AVX2 i16-lane idct32x32_add_avx2_135"
        );

        // avx2=false, sse2=true → SSE2 i16-lane partial kernel.
        let mut caps_sse2 = detect();
        caps_sse2.avx2 = false;
        caps_sse2.sse2 = true;
        let d_sse2 = SimdDispatch::build(caps_sse2);
        assert_eq!(
            d_sse2.idct32x32_135_add() as usize,
            sse2 as usize,
            "avx2=false, sse2=true must install the SSE2 i16-lane idct32x32_add_sse2_135"
        );

        // avx2=false, sse2=false → scalar.
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        caps_scalar.sse2 = false;
        let d_scalar = SimdDispatch::build(caps_scalar);
        assert_eq!(
            d_scalar.idct32x32_135_add() as usize,
            scalar as usize,
            "avx2=false, sse2=false must install the scalar idct32x32_135_add"
        );
    }

    /// Forcing `avx2 = false` falls back to the scalar reference for
    /// the Wave E G4-β iht4x4 slot. See
    /// `forced_scalar_caps_installs_scalar_idct8x8` for the miri-skip
    /// rationale.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_iht4x4() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 16], &mut [u8], usize, TxType) =
            crate::transforms::inverse::iht4x4_16_add;
        assert_eq!(
            d.iht4x4_add() as usize,
            scalar as usize,
            "avx2=false must install the scalar iht4x4_16_add"
        );
    }

    /// Forcing `avx2 = false` falls back to the scalar reference for
    /// the Wave E G4-β iht8x8 slot.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn forced_scalar_caps_installs_scalar_iht8x8() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 64], &mut [u8], usize, TxType) =
            crate::transforms::inverse::iht8x8_64_add;
        assert_eq!(
            d.iht8x8_add() as usize,
            scalar as usize,
            "avx2=false must install the scalar iht8x8_64_add"
        );
    }

    /// Forcing `avx2 = false` falls back to the scalar reference for
    /// the Wave F F5 iht16x16 slot; forcing `avx2 = true` installs the
    /// AVX2 wrapper and the slot must NOT collapse to the scalar fn.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn forced_scalar_caps_installs_scalar_iht16x16() {
        let scalar: fn(&[i32; 256], &mut [u8], usize, TxType) =
            crate::transforms::inverse::iht16x16_256_add;
        let avx2_fp: fn(&[i32; 256], &mut [u8], usize, TxType) =
            crate::dsp::x86_64::iht16x16_add_avx2;

        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let d_scalar = SimdDispatch::build(caps_scalar);
        assert_eq!(
            d_scalar.iht16x16_add() as usize,
            scalar as usize,
            "avx2=false must install the scalar iht16x16_256_add"
        );

        let mut caps_avx2 = detect();
        caps_avx2.avx2 = true;
        let d_avx2 = SimdDispatch::build(caps_avx2);
        assert_ne!(
            d_avx2.iht16x16_add() as usize,
            scalar as usize,
            "avx2=true must NOT install the scalar iht16x16_256_add"
        );
        assert_eq!(
            d_avx2.iht16x16_add() as usize,
            avx2_fp as usize,
            "avx2=true must install the iht16x16_add_avx2 wrapper"
        );
    }

    /// Wave F F3 / F4 — 16×16 EOB-bucket fast paths route to the SSE2
    /// i16-lane safe wrappers when `caps.sse2 == true`, scalar
    /// otherwise. Same dispatch pattern as the dense `idct16x16_add`
    /// flip (d4b6ddd). See `forced_scalar_caps_installs_scalar_idct8x8`
    /// for the miri-skip rationale.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn idct16x16_eob_bucket_slots_dispatch_correctly() {
        let scalar_10: fn(&[i32; 256], &mut [u8], usize) =
            crate::transforms::inverse::idct16x16_10_add;
        let scalar_38: fn(&[i32; 256], &mut [u8], usize) =
            crate::transforms::inverse::idct16x16_38_add;
        let sse2_10: fn(&[i32; 256], &mut [u8], usize) = crate::dsp::x86_64::idct16x16_add_sse2_10;
        let sse2_38: fn(&[i32; 256], &mut [u8], usize) = crate::dsp::x86_64::idct16x16_add_sse2_38;

        // SSE2 caps → SSE2 wrapper installed (NOT scalar).
        let mut caps = detect();
        caps.sse2 = true;
        let d = SimdDispatch::build(caps);
        assert_ne!(
            d.idct16x16_10_add() as usize,
            scalar_10 as usize,
            "sse2=true must install the SSE2 wrapper for idct16x16_10_add, not scalar"
        );
        assert_eq!(
            d.idct16x16_10_add() as usize,
            sse2_10 as usize,
            "sse2=true must install idct16x16_add_sse2_10"
        );
        assert_ne!(
            d.idct16x16_38_add() as usize,
            scalar_38 as usize,
            "sse2=true must install the SSE2 wrapper for idct16x16_38_add, not scalar"
        );
        assert_eq!(
            d.idct16x16_38_add() as usize,
            sse2_38 as usize,
            "sse2=true must install idct16x16_add_sse2_38"
        );

        // Forced-scalar caps (sse2=false) → scalar reference installed.
        let mut caps_scalar = detect();
        caps_scalar.sse2 = false;
        caps_scalar.avx2 = false;
        let d_scalar = SimdDispatch::build(caps_scalar);
        assert_eq!(
            d_scalar.idct16x16_10_add() as usize,
            scalar_10 as usize,
            "sse2=false must install the scalar idct16x16_10_add"
        );
        assert_eq!(
            d_scalar.idct16x16_38_add() as usize,
            scalar_38 as usize,
            "sse2=false must install the scalar idct16x16_38_add"
        );
    }

    /// Forcing `avx2 = false` falls back to the scalar adapter for
    /// the G4.3a dc_predictor_4x4 slot. See
    /// `forced_scalar_caps_installs_scalar_idct8x8` for the miri-skip
    /// rationale.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_dc_predictor_4x4() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&mut [u8], usize, &[u8], &[u8]) = super::scalar_dc_predictor_4x4;
        assert_eq!(
            d.dc_predictor_4x4() as usize,
            scalar as usize,
            "avx2=false must install the scalar dc_predictor_4x4 adapter"
        );
    }

    /// Wave I I1 — on AVX2 hosts the `v_predictor` slot routes through
    /// the AVX2 wrapper and produces output bit-identical to the scalar
    /// adapter for a deterministic input across all four `bs` sizes.
    /// On non-AVX2 hosts the slot installs the scalar adapter and the
    /// test still runs (the bit-equality assertion is trivially true).
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn v_predictor_avx2_routing_avx2_cpu() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut caps = detect();
        caps.avx2 = true;
        let d = SimdDispatch::build(caps);
        // Fn-pointer-identity check: AVX2 caps must install the AVX2
        // safe wrapper.
        let avx2_fp: fn(&mut [u8], usize, usize, &[u8]) = crate::dsp::x86_64::v_predictor_avx2;
        assert_eq!(
            d.v_predictor() as usize,
            avx2_fp as usize,
            "avx2=true must install the v_predictor_avx2 wrapper"
        );
        // Output-parity check: AVX2 slot must agree with the scalar
        // adapter on a deterministic input across all four sizes.
        let scalar: fn(&mut [u8], usize, usize, &[u8]) = super::scalar_v_predictor;
        for bs in [4usize, 8, 16, 32] {
            let above: Vec<u8> = (0..bs).map(|i| (i as u32 * 17 + 3) as u8).collect();
            let mut dst_slot = vec![0u8; bs * bs];
            let mut dst_scalar = vec![0u8; bs * bs];
            (d.v_predictor())(&mut dst_slot, bs, bs, &above);
            scalar(&mut dst_scalar, bs, bs, &above);
            assert_eq!(
                dst_slot, dst_scalar,
                "v_predictor parity mismatch at bs={bs}"
            );
        }
    }

    /// Wave I I1 — when caps.avx2 is forced false, the `v_predictor`
    /// slot routes to the scalar adapter.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_v_predictor() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&mut [u8], usize, usize, &[u8]) = super::scalar_v_predictor;
        assert_eq!(
            d.v_predictor() as usize,
            scalar as usize,
            "avx2=false must install the scalar v_predictor adapter"
        );
    }

    /// Wave I I2 — on AVX2 hosts the `h_predictor` slot routes through
    /// the AVX2 wrapper and produces output bit-identical to the scalar
    /// adapter for a deterministic input across all four `bs` sizes.
    /// Mirrors `v_predictor_avx2_routing_avx2_cpu` (Wave I I1) — only
    /// the input axis differs (`left` here, `above` in I1).
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn h_predictor_avx2_routing_avx2_cpu() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut caps = detect();
        caps.avx2 = true;
        let d = SimdDispatch::build(caps);
        // Fn-pointer-identity check: AVX2 caps must install the AVX2
        // safe wrapper.
        let avx2_fp: fn(&mut [u8], usize, usize, &[u8]) = crate::dsp::x86_64::h_predictor_avx2;
        assert_eq!(
            d.h_predictor() as usize,
            avx2_fp as usize,
            "avx2=true must install the h_predictor_avx2 wrapper"
        );
        // Output-parity check: AVX2 slot must agree with the scalar
        // adapter on a deterministic input across all four sizes.
        let scalar: fn(&mut [u8], usize, usize, &[u8]) = super::scalar_h_predictor;
        for bs in [4usize, 8, 16, 32] {
            let left: Vec<u8> = (0..bs).map(|i| (i as u32 * 19 + 5) as u8).collect();
            let mut dst_slot = vec![0u8; bs * bs];
            let mut dst_scalar = vec![0u8; bs * bs];
            (d.h_predictor())(&mut dst_slot, bs, bs, &left);
            scalar(&mut dst_scalar, bs, bs, &left);
            assert_eq!(
                dst_slot, dst_scalar,
                "h_predictor parity mismatch at bs={bs}"
            );
        }
    }

    /// Wave I I2 — when caps.avx2 is forced false, the `h_predictor`
    /// slot routes to the scalar adapter.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_h_predictor() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&mut [u8], usize, usize, &[u8]) = super::scalar_h_predictor;
        assert_eq!(
            d.h_predictor() as usize,
            scalar as usize,
            "avx2=false must install the scalar h_predictor adapter"
        );
    }

    /// Wave I I3 — `tm_predictor` slot routes to AVX2 wrapper when
    /// `caps.avx2`, scalar adapter otherwise. Mirrors the
    /// `dc_predictor_4x4` slot's tautology-guard pattern.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn tm_predictor_avx2_routing_avx2_cpu() {
        type Slot = fn(&mut [u8], usize, usize, &[u8], &[u8], u8);

        // Forced AVX2 → AVX2 wrapper installed (NOT scalar adapter).
        let mut caps_a = detect();
        caps_a.avx2 = true;
        let d_a = SimdDispatch::build(caps_a);
        let avx2_fp: Slot = crate::dsp::x86_64::tm_predictor_avx2;
        let scalar_fp: Slot = super::scalar_tm_predictor;
        assert_eq!(
            d_a.tm_predictor() as usize,
            avx2_fp as usize,
            "avx2=true must install tm_predictor_avx2"
        );
        assert_ne!(
            d_a.tm_predictor() as usize,
            scalar_fp as usize,
            "avx2=true must NOT install the scalar tm_predictor adapter"
        );

        // Forced scalar → scalar adapter installed.
        let mut caps_s = detect();
        caps_s.avx2 = false;
        let d_s = SimdDispatch::build(caps_s);
        assert_eq!(
            d_s.tm_predictor() as usize,
            scalar_fp as usize,
            "avx2=false must install the scalar tm_predictor adapter"
        );
    }

    /// Wave K — directional slot routing. Forced-AVX2 installs the AVX2
    /// wrapper (fn-pointer identity, no call → host AVX2 not required);
    /// forced-scalar installs the scalar adapter. Covers all six modes.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn directional_avx2_routing_avx2_cpu() {
        type Slot4 = fn(&mut [u8], usize, usize, &[u8]);
        type Slot6 = fn(&mut [u8], usize, usize, &[u8], &[u8], u8);

        let mut caps_a = detect();
        caps_a.avx2 = true;
        let d_a = SimdDispatch::build(caps_a);

        // 4-arg slots: D45, D63, D207.
        let cases4: &[(Slot4, Slot4, Slot4, &str)] = &[
            (
                d_a.d45_predictor(),
                crate::dsp::x86_64::d45_predictor_avx2,
                super::scalar_d45_predictor,
                "d45",
            ),
            (
                d_a.d63_predictor(),
                crate::dsp::x86_64::d63_predictor_avx2,
                super::scalar_d63_predictor,
                "d63",
            ),
            (
                d_a.d207_predictor(),
                crate::dsp::x86_64::d207_predictor_avx2,
                super::scalar_d207_predictor,
                "d207",
            ),
        ];
        for &(slot, avx2_fp, scalar_fp, name) in cases4 {
            assert_eq!(
                slot as usize, avx2_fp as usize,
                "avx2=true must install {name}_avx2"
            );
            assert_ne!(
                slot as usize, scalar_fp as usize,
                "avx2=true must NOT install scalar {name}"
            );
        }

        // 6-arg slots: D117, D135, D153.
        let cases6: &[(Slot6, Slot6, Slot6, &str)] = &[
            (
                d_a.d117_predictor(),
                crate::dsp::x86_64::d117_predictor_avx2,
                super::scalar_d117_predictor,
                "d117",
            ),
            (
                d_a.d135_predictor(),
                crate::dsp::x86_64::d135_predictor_avx2,
                super::scalar_d135_predictor,
                "d135",
            ),
            (
                d_a.d153_predictor(),
                crate::dsp::x86_64::d153_predictor_avx2,
                super::scalar_d153_predictor,
                "d153",
            ),
        ];
        for &(slot, avx2_fp, scalar_fp, name) in cases6 {
            assert_eq!(
                slot as usize, avx2_fp as usize,
                "avx2=true must install {name}_avx2"
            );
            assert_ne!(
                slot as usize, scalar_fp as usize,
                "avx2=true must NOT install scalar {name}"
            );
        }
    }

    /// Wave K — forced-scalar caps install the scalar directional
    /// adapters for all six modes.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn forced_scalar_caps_installs_scalar_directional() {
        type Slot4 = fn(&mut [u8], usize, usize, &[u8]);
        type Slot6 = fn(&mut [u8], usize, usize, &[u8], &[u8], u8);

        let mut caps_s = detect();
        caps_s.avx2 = false;
        let d_s = SimdDispatch::build(caps_s);

        assert_eq!(
            d_s.d45_predictor() as usize,
            super::scalar_d45_predictor as Slot4 as usize,
            "avx2=false must install scalar d45"
        );
        assert_eq!(
            d_s.d63_predictor() as usize,
            super::scalar_d63_predictor as Slot4 as usize,
            "avx2=false must install scalar d63"
        );
        assert_eq!(
            d_s.d207_predictor() as usize,
            super::scalar_d207_predictor as Slot4 as usize,
            "avx2=false must install scalar d207"
        );
        assert_eq!(
            d_s.d117_predictor() as usize,
            super::scalar_d117_predictor as Slot6 as usize,
            "avx2=false must install scalar d117"
        );
        assert_eq!(
            d_s.d135_predictor() as usize,
            super::scalar_d135_predictor as Slot6 as usize,
            "avx2=false must install scalar d135"
        );
        assert_eq!(
            d_s.d153_predictor() as usize,
            super::scalar_d153_predictor as Slot6 as usize,
            "avx2=false must install scalar d153"
        );
    }

    /// Wave I I4 — `dc_predict_full` slot routes to AVX2 wrapper when
    /// `caps.avx2`, scalar adapter otherwise. Mirrors the
    /// `tm_predictor_avx2_routing_avx2_cpu` shape.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn dc_predict_full_avx2_routing() {
        type Slot = fn(&mut [u8], usize, usize, &[u8], &[u8]);

        let mut caps_a = detect();
        caps_a.avx2 = true;
        let d_a = SimdDispatch::build(caps_a);
        let avx2_fp: Slot = crate::dsp::x86_64::dc_predict_full_avx2;
        let scalar_fp: Slot = super::scalar_dc_predict_full;
        assert_eq!(
            d_a.dc_predict_full() as usize,
            avx2_fp as usize,
            "avx2=true must install dc_predict_full_avx2"
        );
        assert_ne!(
            d_a.dc_predict_full() as usize,
            scalar_fp as usize,
            "avx2=true must NOT install the scalar dc_predict_full adapter"
        );

        let mut caps_s = detect();
        caps_s.avx2 = false;
        let d_s = SimdDispatch::build(caps_s);
        assert_eq!(
            d_s.dc_predict_full() as usize,
            scalar_fp as usize,
            "avx2=false must install the scalar dc_predict_full adapter"
        );
    }

    /// Wave I I4 — `dc_predict_top` slot routes to AVX2 wrapper when
    /// `caps.avx2`, scalar reference otherwise.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn dc_predict_top_avx2_routing() {
        type Slot = fn(&mut [u8], usize, usize, &[u8]);

        let mut caps_a = detect();
        caps_a.avx2 = true;
        let d_a = SimdDispatch::build(caps_a);
        let avx2_fp: Slot = crate::dsp::x86_64::dc_predict_top_avx2;
        let scalar_fp: Slot = super::scalar_dc_predict_top;
        assert_eq!(
            d_a.dc_predict_top() as usize,
            avx2_fp as usize,
            "avx2=true must install dc_predict_top_avx2"
        );
        assert_ne!(
            d_a.dc_predict_top() as usize,
            scalar_fp as usize,
            "avx2=true must NOT install the scalar dc_predict_top"
        );

        let mut caps_s = detect();
        caps_s.avx2 = false;
        let d_s = SimdDispatch::build(caps_s);
        assert_eq!(
            d_s.dc_predict_top() as usize,
            scalar_fp as usize,
            "avx2=false must install the scalar dc_predict_top"
        );
    }

    /// Wave I I4 — `dc_predict_left` slot routes to AVX2 wrapper when
    /// `caps.avx2`, scalar reference otherwise.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn dc_predict_left_avx2_routing() {
        type Slot = fn(&mut [u8], usize, usize, &[u8]);

        let mut caps_a = detect();
        caps_a.avx2 = true;
        let d_a = SimdDispatch::build(caps_a);
        let avx2_fp: Slot = crate::dsp::x86_64::dc_predict_left_avx2;
        let scalar_fp: Slot = super::scalar_dc_predict_left;
        assert_eq!(
            d_a.dc_predict_left() as usize,
            avx2_fp as usize,
            "avx2=true must install dc_predict_left_avx2"
        );
        assert_ne!(
            d_a.dc_predict_left() as usize,
            scalar_fp as usize,
            "avx2=true must NOT install the scalar dc_predict_left"
        );

        let mut caps_s = detect();
        caps_s.avx2 = false;
        let d_s = SimdDispatch::build(caps_s);
        assert_eq!(
            d_s.dc_predict_left() as usize,
            scalar_fp as usize,
            "avx2=false must install the scalar dc_predict_left"
        );
    }

    /// Wave I I4 — `dc_predict_128` slot routes to AVX2 wrapper when
    /// `caps.avx2`, scalar reference otherwise.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn dc_predict_128_avx2_routing() {
        type Slot = fn(&mut [u8], usize, usize);

        let mut caps_a = detect();
        caps_a.avx2 = true;
        let d_a = SimdDispatch::build(caps_a);
        let avx2_fp: Slot = crate::dsp::x86_64::dc_predict_128_avx2;
        let scalar_fp: Slot = super::scalar_dc_predict_128;
        assert_eq!(
            d_a.dc_predict_128() as usize,
            avx2_fp as usize,
            "avx2=true must install dc_predict_128_avx2"
        );
        assert_ne!(
            d_a.dc_predict_128() as usize,
            scalar_fp as usize,
            "avx2=true must NOT install the scalar dc_predict_128"
        );

        let mut caps_s = detect();
        caps_s.avx2 = false;
        let d_s = SimdDispatch::build(caps_s);
        assert_eq!(
            d_s.dc_predict_128() as usize,
            scalar_fp as usize,
            "avx2=false must install the scalar dc_predict_128"
        );
    }

    /// G4.4b deferred slot is always scalar regardless of caps. (H6
    /// promoted `plane_dequant_finisher` to AVX2 — see
    /// `plane_dequant_finisher_dispatches_correctly` for that slot's
    /// parity assertion.)
    /// See `forced_scalar_caps_installs_scalar_idct8x8` for the
    /// miri-skip rationale.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn deferred_slots_are_always_scalar() {
        for &avx2 in &[true, false] {
            let mut caps = detect();
            caps.avx2 = avx2;
            let d = SimdDispatch::build(caps);
            let ef: fn(&mut Yv12Buffer) = super::scalar_extend_frame_borders;
            assert_eq!(d.extend_frame_borders() as usize, ef as usize);
        }
    }

    /// Wave H H6 — `plane_dequant_finisher` slot routes to AVX2 wrapper
    /// when `caps.avx2`, scalar reference otherwise. Mirrors the F10/G3
    /// tautology-guard pattern.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn plane_dequant_finisher_dispatches_correctly() {
        type Slot = fn(&[u16], &[i16], &[i16], &[u8], u32, &mut [i32]);

        // Forced scalar → scalar reference installed.
        let mut caps_s = detect();
        caps_s.avx2 = false;
        let d_s = SimdDispatch::build(caps_s);
        let scalar_fp: Slot = super::scalar_plane_dequant_finisher;
        assert_eq!(
            d_s.plane_dequant_finisher() as usize,
            scalar_fp as usize,
            "avx2=false must install scalar plane_dequant_finisher"
        );

        // Forced AVX2 → AVX2 wrapper installed (NOT scalar).
        let mut caps_a = detect();
        caps_a.avx2 = true;
        let d_a = SimdDispatch::build(caps_a);
        let avx2_fp: Slot = crate::dsp::x86_64::plane_dequant_finisher_avx2;
        assert_ne!(
            d_a.plane_dequant_finisher() as usize,
            scalar_fp as usize,
            "avx2=true must NOT install scalar plane_dequant_finisher"
        );
        assert_eq!(
            d_a.plane_dequant_finisher() as usize,
            avx2_fp as usize,
            "avx2=true must install plane_dequant_finisher_avx2"
        );
    }

    /// Wave H H1 + H7 — default loopfilter AVX2 routing after H7
    /// bisection identified `lpf_vertical_16_avx2` as the lone
    /// buggy kernel. The other 5 single-variant AVX2 lpf_* slots
    /// ship active by default on AVX2 CPUs; `lpf_vertical_16` stays
    /// scalar pending the per-kernel fix (still tracked under H7).
    ///
    /// Conformance verified at 201/0/0/0/201 with this routing.
    ///
    /// See `notes/wave-h-loopfilter-avx2-gate.md` for bisection
    /// details and the env-var grammar for forcing different routings
    /// at dev time.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn loopfilter_avx2_default_routing_avx2_cpu() {
        // Per Rust's docs `std::env::set_var` is not thread-safe across
        // parallel tests; we only READ the var here (the test relies on
        // it being unset in cargo's default test env). If a downstream
        // test sets it, this test must be marked `#[serial]` (we don't
        // depend on serial_test today).
        let mut caps = detect();
        caps.avx2 = true;
        let d = SimdDispatch::build(caps);
        type Lpf = fn(&mut [u8], usize, usize, u8, u8, u8);
        // All 6 AVX2 paths route through their AVX2 wrappers post-H7
        // (V16 was the bug fixed in 2026-05-22; flat_mask5 outer-pixel
        // operands were swapped for inner ones).
        let v4_avx2: Lpf = crate::dsp::x86_64::lpf_vertical_4_avx2;
        let h4_avx2: Lpf = crate::dsp::x86_64::lpf_horizontal_4_avx2;
        let v8_avx2: Lpf = crate::dsp::x86_64::lpf_vertical_8_avx2;
        let h8_avx2: Lpf = crate::dsp::x86_64::lpf_horizontal_8_avx2;
        let v16_avx2: Lpf = crate::dsp::x86_64::lpf_vertical_16_avx2;
        let h16_avx2: Lpf = crate::dsp::x86_64::lpf_horizontal_16_avx2;
        assert_eq!(d.lpf_vertical_4() as usize, v4_avx2 as usize);
        assert_eq!(d.lpf_horizontal_4() as usize, h4_avx2 as usize);
        assert_eq!(d.lpf_vertical_8() as usize, v8_avx2 as usize);
        assert_eq!(d.lpf_horizontal_8() as usize, h8_avx2 as usize);
        assert_eq!(d.lpf_vertical_16() as usize, v16_avx2 as usize);
        assert_eq!(d.lpf_horizontal_16() as usize, h16_avx2 as usize);
    }

    /// Wave H H1 + H7 — when caps.avx2 is forced false, all 6
    /// single-variant lpf_* slots route to scalar regardless of
    /// the env var or H7 bisection state.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn loopfilter_scalar_routing_no_avx2() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        type Lpf = fn(&mut [u8], usize, usize, u8, u8, u8);
        let v4: Lpf = super::scalar_lpf_vertical_4;
        let h4: Lpf = super::scalar_lpf_horizontal_4;
        let v8: Lpf = super::scalar_lpf_vertical_8;
        let h8: Lpf = super::scalar_lpf_horizontal_8;
        let v16: Lpf = super::scalar_lpf_vertical_16;
        let h16: Lpf = super::scalar_lpf_horizontal_16;
        assert_eq!(d.lpf_vertical_4() as usize, v4 as usize);
        assert_eq!(d.lpf_horizontal_4() as usize, h4 as usize);
        assert_eq!(d.lpf_vertical_8() as usize, v8 as usize);
        assert_eq!(d.lpf_horizontal_8() as usize, h8 as usize);
        assert_eq!(d.lpf_vertical_16() as usize, v16 as usize);
        assert_eq!(d.lpf_horizontal_16() as usize, h16 as usize);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn loopfilter_4_8_and_16_dispatch_to_neon() {
        type Lpf = fn(&mut [u8], usize, usize, u8, u8, u8);

        let mut scalar_caps = detect();
        scalar_caps.neon = false;
        let scalar = SimdDispatch::build(scalar_caps);
        let scalar_v4: Lpf = super::scalar_lpf_vertical_4;
        let scalar_h4: Lpf = super::scalar_lpf_horizontal_4;
        let scalar_v4_dual: LpfDualFn = super::scalar_lpf_vertical_4_dual;
        let scalar_h4_dual: LpfDualFn = super::scalar_lpf_horizontal_4_dual;
        let scalar_v8: Lpf = super::scalar_lpf_vertical_8;
        let scalar_h8: Lpf = super::scalar_lpf_horizontal_8;
        let scalar_v8_dual: LpfDualFn = super::scalar_lpf_vertical_8_dual;
        let scalar_h8_dual: LpfDualFn = super::scalar_lpf_horizontal_8_dual;
        let scalar_v16: Lpf = super::scalar_lpf_vertical_16;
        let scalar_h16: Lpf = super::scalar_lpf_horizontal_16;
        let scalar_v16_dual: Lpf = super::scalar_lpf_vertical_16_dual;
        let scalar_h16_dual: Lpf = super::scalar_lpf_horizontal_16_dual;
        assert_eq!(scalar.lpf_vertical_4() as usize, scalar_v4 as usize);
        assert_eq!(scalar.lpf_horizontal_4() as usize, scalar_h4 as usize);
        assert_eq!(
            scalar.lpf_vertical_4_dual() as usize,
            scalar_v4_dual as usize
        );
        assert_eq!(
            scalar.lpf_horizontal_4_dual() as usize,
            scalar_h4_dual as usize
        );
        assert_eq!(scalar.lpf_vertical_8() as usize, scalar_v8 as usize);
        assert_eq!(scalar.lpf_horizontal_8() as usize, scalar_h8 as usize);
        assert_eq!(
            scalar.lpf_vertical_8_dual() as usize,
            scalar_v8_dual as usize
        );
        assert_eq!(
            scalar.lpf_horizontal_8_dual() as usize,
            scalar_h8_dual as usize
        );
        assert_eq!(scalar.lpf_vertical_16() as usize, scalar_v16 as usize);
        assert_eq!(scalar.lpf_horizontal_16() as usize, scalar_h16 as usize);
        assert_eq!(
            scalar.lpf_vertical_16_dual() as usize,
            scalar_v16_dual as usize
        );
        assert_eq!(
            scalar.lpf_horizontal_16_dual() as usize,
            scalar_h16_dual as usize
        );

        let mut neon_caps = detect();
        neon_caps.neon = true;
        let neon = SimdDispatch::build(neon_caps);
        let neon_v4: Lpf = crate::dsp::aarch64::lpf_vertical_4_neon;
        let neon_h4: Lpf = crate::dsp::aarch64::lpf_horizontal_4_neon;
        let neon_v4_dual: LpfDualFn = crate::dsp::aarch64::lpf_vertical_4_dual_neon;
        let neon_h4_dual: LpfDualFn = crate::dsp::aarch64::lpf_horizontal_4_dual_neon;
        let neon_v8: Lpf = crate::dsp::aarch64::lpf_vertical_8_neon;
        let neon_h8: Lpf = crate::dsp::aarch64::lpf_horizontal_8_neon;
        let neon_v8_dual: LpfDualFn = crate::dsp::aarch64::lpf_vertical_8_dual_neon;
        let neon_h8_dual: LpfDualFn = crate::dsp::aarch64::lpf_horizontal_8_dual_neon;
        let neon_v16: Lpf = crate::dsp::aarch64::lpf_vertical_16_neon;
        let neon_h16: Lpf = crate::dsp::aarch64::lpf_horizontal_16_neon;
        let neon_v16_dual: Lpf = crate::dsp::aarch64::lpf_vertical_16_dual_neon;
        let neon_h16_dual: Lpf = crate::dsp::aarch64::lpf_horizontal_16_dual_neon;
        assert_eq!(neon.lpf_vertical_4() as usize, neon_v4 as usize);
        assert_eq!(neon.lpf_horizontal_4() as usize, neon_h4 as usize);
        assert_eq!(neon.lpf_vertical_4_dual() as usize, neon_v4_dual as usize);
        assert_eq!(neon.lpf_horizontal_4_dual() as usize, neon_h4_dual as usize);
        assert_eq!(neon.lpf_vertical_8() as usize, neon_v8 as usize);
        assert_eq!(neon.lpf_horizontal_8() as usize, neon_h8 as usize);
        assert_eq!(neon.lpf_vertical_8_dual() as usize, neon_v8_dual as usize);
        assert_eq!(neon.lpf_horizontal_8_dual() as usize, neon_h8_dual as usize);
        assert_eq!(neon.lpf_vertical_16() as usize, neon_v16 as usize);
        assert_eq!(neon.lpf_horizontal_16() as usize, neon_h16 as usize);
        assert_eq!(neon.lpf_vertical_16_dual() as usize, neon_v16_dual as usize);
        assert_eq!(
            neon.lpf_horizontal_16_dual() as usize,
            neon_h16_dual as usize
        );
    }

    /// Wave H H1 + H7 — `parse_lpf_avx2_mask` grammar regression test.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn parse_lpf_avx2_mask_grammar() {
        // Unset / empty -> DEFAULT (all 6 kernels post-H7 fix)
        assert_eq!(
            super::parse_lpf_avx2_mask(None),
            super::LpfAvx2Mask::DEFAULT
        );
        assert_eq!(
            super::parse_lpf_avx2_mask(Some("")),
            super::LpfAvx2Mask::DEFAULT
        );
        // DEFAULT == ALL post-H7 — keep the bisection grammar intact
        // for future SIMD work even though the gate is now a no-op.
        assert_eq!(super::LpfAvx2Mask::DEFAULT, super::LpfAvx2Mask::ALL);
        // "0" -> NONE
        assert_eq!(
            super::parse_lpf_avx2_mask(Some("0")),
            super::LpfAvx2Mask::NONE
        );
        // "1" -> ALL
        assert_eq!(
            super::parse_lpf_avx2_mask(Some("1")),
            super::LpfAvx2Mask::ALL
        );
        // Comma list: only listed kernels.
        let m = super::parse_lpf_avx2_mask(Some("v4,h4"));
        assert!(m.contains(super::LpfAvx2Mask::V4));
        assert!(m.contains(super::LpfAvx2Mask::H4));
        assert!(!m.contains(super::LpfAvx2Mask::V8));
        assert!(!m.contains(super::LpfAvx2Mask::V16));
        // Unknown tokens silently ignored; legitimate ones still pass.
        let m = super::parse_lpf_avx2_mask(Some("v4,xx,h4"));
        assert!(m.contains(super::LpfAvx2Mask::V4));
        assert!(m.contains(super::LpfAvx2Mask::H4));
    }

    /// Wave H H2 + H3 — on AVX2 CPUs the 4/8-tap dual slots route to
    /// the AVX2 dual wrappers; the 16-tap duals also route to AVX2
    /// after Wave H H4 + H7 (V16 fix made the two-singles delegation
    /// bit-exact). Note 4/8-tap duals are 9-arg (independent
    /// threshold sets) while 16-tap duals are 6-arg (shared
    /// thresholds, libvpx convention). See
    /// `forced_scalar_caps_installs_scalar_idct8x8` for the miri-skip
    /// rationale.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn dual_loopfilter_slots_avx2_routing_avx2_cpu() {
        type Dual9 = fn(&mut [u8], usize, usize, u8, u8, u8, u8, u8, u8);
        type Dual6 = fn(&mut [u8], usize, usize, u8, u8, u8);
        let mut caps = detect();
        caps.avx2 = true;
        let d = SimdDispatch::build(caps);
        // 4/8-tap duals route to their AVX2 wrappers.
        let v4_avx2: Dual9 = crate::dsp::x86_64::lpf_vertical_4_dual_avx2;
        let h4_avx2: Dual9 = crate::dsp::x86_64::lpf_horizontal_4_dual_avx2;
        let v8_avx2: Dual9 = crate::dsp::x86_64::lpf_vertical_8_dual_avx2;
        let h8_avx2: Dual9 = crate::dsp::x86_64::lpf_horizontal_8_dual_avx2;
        assert_eq!(d.lpf_vertical_4_dual() as usize, v4_avx2 as usize);
        assert_eq!(d.lpf_horizontal_4_dual() as usize, h4_avx2 as usize);
        assert_eq!(d.lpf_vertical_8_dual() as usize, v8_avx2 as usize);
        assert_eq!(d.lpf_horizontal_8_dual() as usize, h8_avx2 as usize);
        // Wave H H4 + H4.1 — 16-tap duals both route to AVX2.
        let v16_avx2: Dual6 = crate::dsp::x86_64::lpf_vertical_16_dual_avx2;
        let h16_avx2: Dual6 = crate::dsp::x86_64::lpf_horizontal_16_dual_avx2;
        assert_eq!(d.lpf_vertical_16_dual() as usize, v16_avx2 as usize);
        assert_eq!(d.lpf_horizontal_16_dual() as usize, h16_avx2 as usize);
    }

    /// Wave H H2 + H3 — when caps.avx2 is forced false, all 6 dual
    /// slots route to their scalar references.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn dual_loopfilter_slots_scalar_routing_no_avx2() {
        type Dual9 = fn(&mut [u8], usize, usize, u8, u8, u8, u8, u8, u8);
        type Dual6 = fn(&mut [u8], usize, usize, u8, u8, u8);
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let v4: Dual9 = super::scalar_lpf_vertical_4_dual;
        let h4: Dual9 = super::scalar_lpf_horizontal_4_dual;
        let v8: Dual9 = super::scalar_lpf_vertical_8_dual;
        let h8: Dual9 = super::scalar_lpf_horizontal_8_dual;
        let v16: Dual6 = super::scalar_lpf_vertical_16_dual;
        let h16: Dual6 = super::scalar_lpf_horizontal_16_dual;
        assert_eq!(d.lpf_vertical_4_dual() as usize, v4 as usize);
        assert_eq!(d.lpf_horizontal_4_dual() as usize, h4 as usize);
        assert_eq!(d.lpf_vertical_8_dual() as usize, v8 as usize);
        assert_eq!(d.lpf_horizontal_8_dual() as usize, h8 as usize);
        assert_eq!(d.lpf_vertical_16_dual() as usize, v16 as usize);
        assert_eq!(d.lpf_horizontal_16_dual() as usize, h16 as usize);
    }

    // =================================================================
    // Wave I I5 — motion-compensation slot routing.
    // -----------------------------------------------------------------
    // At I5 these slots were scalar regardless of caps. Wave J promotes
    // individual fns to AVX2 one by one — promoted slots split into the
    // `avx2-on / scalar forced-off` pattern used by I1-I4 / H6; the
    // remaining scalar slots keep the original single-test shape.
    // =================================================================

    /// Wave J J1 — on AVX2 hosts the `convolve_copy` slot routes
    /// through the AVX2 wrapper. Mirrors `v_predictor_avx2_routing_avx2_cpu`
    /// — fn-pointer-identity check only (output-parity is covered by
    /// the differential test in `convolve_copy_avx2::tests`).
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn convolve_copy_avx2_routing_avx2_cpu() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut caps = detect();
        caps.avx2 = true;
        let d = SimdDispatch::build(caps);
        let avx2_fp: fn(&[u8], usize, &mut [u8], usize, usize, usize) =
            crate::dsp::x86_64::convolve_copy_avx2;
        assert_eq!(
            d.convolve_copy() as usize,
            avx2_fp as usize,
            "avx2=true must install the convolve_copy_avx2 wrapper"
        );
    }

    /// Wave J J1 — when caps.avx2 is forced false, the `convolve_copy`
    /// slot routes to the scalar reference.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_convolve_copy() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[u8], usize, &mut [u8], usize, usize, usize) =
            crate::inter::filter::convolve_copy;
        assert_eq!(
            d.convolve_copy() as usize,
            scalar as usize,
            "avx2=false must install the scalar convolve_copy reference"
        );
    }

    /// Wave J J2 — on AVX2 hosts the `convolve_avg` slot routes through
    /// the AVX2 wrapper. Fn-pointer-identity check only (output-parity
    /// is covered by the differential test in `convolve_avg_avx2::tests`).
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn convolve_avg_avx2_routing_avx2_cpu() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut caps = detect();
        caps.avx2 = true;
        let d = SimdDispatch::build(caps);
        let avx2_fp: fn(&[u8], usize, &mut [u8], usize, usize, usize) =
            crate::dsp::x86_64::convolve_avg_avx2;
        assert_eq!(
            d.convolve_avg() as usize,
            avx2_fp as usize,
            "avx2=true must install the convolve_avg_avx2 wrapper"
        );
    }

    /// Wave J J2 — when caps.avx2 is forced false, the `convolve_avg`
    /// slot routes to the scalar reference.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_convolve_avg() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[u8], usize, &mut [u8], usize, usize, usize) =
            crate::inter::filter::convolve_avg;
        assert_eq!(
            d.convolve_avg() as usize,
            scalar as usize,
            "avx2=false must install the scalar convolve_avg reference"
        );
    }

    /// Wave J J3 — on AVX2 hosts the `convolve8_horiz` slot routes
    /// through the AVX2 wrapper. Fn-pointer-identity check (output
    /// parity is covered by `convolve8_horiz_avx2::tests`).
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn convolve8_horiz_avx2_routing_avx2_cpu() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut caps = detect();
        caps.avx2 = true;
        let d = SimdDispatch::build(caps);
        let avx2_fp: Convolve8OneAxisFn = crate::dsp::x86_64::convolve8_horiz_avx2;
        assert_eq!(
            d.convolve8_horiz() as usize,
            avx2_fp as usize,
            "avx2=true must install the convolve8_horiz_avx2 wrapper"
        );
    }

    /// Wave J J3 — forced-scalar caps install the scalar reference.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_convolve8_horiz() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: Convolve8OneAxisFn = crate::inter::filter::convolve8_horiz;
        assert_eq!(
            d.convolve8_horiz() as usize,
            scalar as usize,
            "avx2=false must install the scalar convolve8_horiz reference"
        );
    }

    /// Wave J J4 — on AVX2 hosts the `convolve8_vert` slot routes
    /// through the AVX2 wrapper. Fn-pointer-identity check (output
    /// parity is covered by `convolve8_vert_avx2::tests`).
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn convolve8_vert_avx2_routing_avx2_cpu() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut caps = detect();
        caps.avx2 = true;
        let d = SimdDispatch::build(caps);
        let avx2_fp: Convolve8OneAxisFn = crate::dsp::x86_64::convolve8_vert_avx2;
        assert_eq!(
            d.convolve8_vert() as usize,
            avx2_fp as usize,
            "avx2=true must install the convolve8_vert_avx2 wrapper"
        );
    }

    /// Wave J J4 — forced-scalar caps install the scalar reference.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_convolve8_vert() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: Convolve8OneAxisFn = crate::inter::filter::convolve8_vert;
        assert_eq!(
            d.convolve8_vert() as usize,
            scalar as usize,
            "avx2=false must install the scalar convolve8_vert reference"
        );
    }

    /// Wave J J5 — on AVX2 hosts the `convolve8` slot routes through the
    /// AVX2 wrapper. Fn-pointer-identity check (output parity covered by
    /// `convolve8_avx2::tests`).
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn convolve8_avx2_routing_avx2_cpu() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut caps = detect();
        caps.avx2 = true;
        let d = SimdDispatch::build(caps);
        let avx2_fp: Convolve8TwoAxisFn = crate::dsp::x86_64::convolve8_avx2;
        assert_eq!(
            d.convolve8() as usize,
            avx2_fp as usize,
            "avx2=true must install the convolve8_avx2 wrapper"
        );
    }

    /// Wave J J5 — forced-scalar caps install the scalar reference.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_convolve8() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: Convolve8TwoAxisFn = crate::inter::filter::convolve8;
        assert_eq!(
            d.convolve8() as usize,
            scalar as usize,
            "avx2=false must install the scalar convolve8 reference"
        );
    }

    /// Wave J J6 — AVX2-on routing for the `convolve8_avg_horiz` slot.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn convolve8_avg_horiz_avx2_routing_avx2_cpu() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut caps = detect();
        caps.avx2 = true;
        let d = SimdDispatch::build(caps);
        let avx2_fp: Convolve8OneAxisFn = crate::dsp::x86_64::convolve8_avg_horiz_avx2;
        assert_eq!(
            d.convolve8_avg_horiz() as usize,
            avx2_fp as usize,
            "avx2=true must install the convolve8_avg_horiz_avx2 wrapper"
        );
    }

    /// Wave J J6 — forced-scalar routing for `convolve8_avg_horiz`.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_convolve8_avg_horiz() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: Convolve8OneAxisFn = crate::inter::filter::convolve8_avg_horiz;
        assert_eq!(
            d.convolve8_avg_horiz() as usize,
            scalar as usize,
            "avx2=false must install the scalar convolve8_avg_horiz reference"
        );
    }

    /// Wave J J7 — AVX2-on routing for the `convolve8_avg_vert` slot.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn convolve8_avg_vert_avx2_routing_avx2_cpu() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut caps = detect();
        caps.avx2 = true;
        let d = SimdDispatch::build(caps);
        let avx2_fp: Convolve8OneAxisFn = crate::dsp::x86_64::convolve8_avg_vert_avx2;
        assert_eq!(
            d.convolve8_avg_vert() as usize,
            avx2_fp as usize,
            "avx2=true must install the convolve8_avg_vert_avx2 wrapper"
        );
    }

    /// Wave J J7 — forced-scalar routing for `convolve8_avg_vert`.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_convolve8_avg_vert() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: Convolve8OneAxisFn = crate::inter::filter::convolve8_avg_vert;
        assert_eq!(
            d.convolve8_avg_vert() as usize,
            scalar as usize,
            "avx2=false must install the scalar convolve8_avg_vert reference"
        );
    }

    /// Wave J J8 — AVX2-on routing for the `convolve8_avg` (2-D) slot.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn convolve8_avg_avx2_routing_avx2_cpu() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut caps = detect();
        caps.avx2 = true;
        let d = SimdDispatch::build(caps);
        let avx2_fp: Convolve8TwoAxisFn = crate::dsp::x86_64::convolve8_avg_avx2;
        assert_eq!(
            d.convolve8_avg() as usize,
            avx2_fp as usize,
            "avx2=true must install the convolve8_avg_avx2 wrapper"
        );
    }

    /// Wave J J8 — forced-scalar routing for `convolve8_avg`.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_convolve8_avg() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: Convolve8TwoAxisFn = crate::inter::filter::convolve8_avg;
        assert_eq!(
            d.convolve8_avg() as usize,
            scalar as usize,
            "avx2=false must install the scalar convolve8_avg reference"
        );
    }

    /// Wave L L1 — AVX2-on routing for both scaled-convolve slots.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn convolve8_scaled_avx2_routing_avx2_cpu() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut caps = detect();
        caps.avx2 = true;
        let d = SimdDispatch::build(caps);
        let base: ScaledConvolveFn = crate::dsp::x86_64::convolve8_scaled_avx2;
        let avg: ScaledConvolveFn = crate::dsp::x86_64::convolve8_scaled_avg_avx2;
        assert_eq!(
            d.convolve8_scaled() as usize,
            base as usize,
            "avx2=true must install convolve8_scaled_avx2"
        );
        assert_eq!(
            d.convolve8_scaled_avg() as usize,
            avg as usize,
            "avx2=true must install convolve8_scaled_avg_avx2"
        );
    }

    /// Wave L L1 — forced-scalar caps install the scalar references.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_convolve8_scaled() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let base: ScaledConvolveFn = crate::inter::filter::convolve8_scaled;
        let avg: ScaledConvolveFn = crate::inter::filter::convolve8_scaled_avg;
        assert_eq!(
            d.convolve8_scaled() as usize,
            base as usize,
            "avx2=false must install scalar convolve8_scaled"
        );
        assert_eq!(
            d.convolve8_scaled_avg() as usize,
            avg as usize,
            "avx2=false must install scalar convolve8_scaled_avg"
        );
    }

    /// I5 end-to-end smoke — `inter_predictor` routes the unscaled
    /// `(0, 0, false)` arm through `dispatch.convolve_copy()` and the
    /// output equals a direct scalar `convolve_copy` call. Proves the
    /// dispatch wiring did not perturb behaviour (Wave I5 is a pure
    /// refactor — the dispatched fn IS the original scalar fn). One
    /// representative arm is sufficient because all 8 arms route
    /// through the same slot-lookup pattern.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn inter_predictor_unscaled_routes_through_dispatch_convolve_copy() {
        use crate::bitstream::frame::InterpFilter;
        use crate::inter::filter::convolve_copy;
        use crate::inter::scale::{ScaleFactors, inter_predictor};

        // 16x16 src with a deterministic ramp; identity scale, zero
        // subpel offsets, no compound avg ⇒ the (false, false, false)
        // arm fires ⇒ dispatch.convolve_copy() should run.
        let w = 16usize;
        let h = 16usize;
        let src_stride = w;
        let dst_stride = w;
        let src: Vec<u8> = (0..w * h).map(|i| (i & 0xff) as u8).collect();
        let mut dst_via_predictor = vec![0u8; w * h];
        let mut dst_via_direct = vec![0u8; w * h];

        let sf = ScaleFactors::new(w as u32, h as u32, w as u32, h as u32);
        inter_predictor(
            &src,
            src_stride,
            &mut dst_via_predictor,
            dst_stride,
            0,
            0,
            &sf,
            InterpFilter::EightTap,
            w,
            h,
            false,
        );
        convolve_copy(&src, src_stride, &mut dst_via_direct, dst_stride, w, h);

        assert_eq!(
            dst_via_predictor, dst_via_direct,
            "I5: inter_predictor unscaled (0,0,false) arm must match direct convolve_copy"
        );
    }
}
