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
use crate::quant::QuantizerDeltas;

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
    /// `lpf_vertical_16_avx2` (V16) causes conformance regressions
    /// (58/201 vectors). The other 5 kernels are bit-exact and ship
    /// active by default. See `notes/wave-h-loopfilter-avx2-gate.md`.
    const DEFAULT: Self = Self(Self::ALL.0 & !Self::V16.0);

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
/// - `None` or `""` → [`LpfAvx2Mask::DEFAULT`] (5 known-good AVX2 kernels
///   active; `lpf_vertical_16` stays scalar pending H7 fix)
/// - `"0"` → [`LpfAvx2Mask::NONE`] (all kernels scalar, for differential
///   testing)
/// - `"1"` → [`LpfAvx2Mask::ALL`] (all 6 AVX2 paths active, including
///   the still-buggy V16; used to reproduce the regression for fix work)
/// - comma list (e.g. `"v4,h4"`) → only the listed kernels active;
///   used for H7 bisection
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
    idct4x4_16_add as scalar_idct4x4_add, idct8x8_64_add as scalar_idct8x8_add,
    idct16x16_10_add as scalar_idct16x16_10_add,
    idct16x16_38_add as scalar_idct16x16_38_add,
    idct16x16_256_add as scalar_idct16x16_add,
    idct32x32_135_add as scalar_idct32x32_135_add,
    idct32x32_34_add as scalar_idct32x32_34_add,
    idct32x32_1024_add as scalar_idct32x32_add,
    iht4x4_16_add as scalar_iht4x4_add, iht8x8_64_add as scalar_iht8x8_add,
    iht16x16_256_add as scalar_iht16x16_add,
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

/// Scalar reference for the `plane_dequant_pair` dispatch slot. Direct
/// alias of [`crate::quant::plane_dequant_pair`]; the slot exists to
/// reserve the family naming for a future per-coefficient dequant
/// AVX2 kernel (which would land alongside the entropy decoder
/// rewrite — currently dequant is fused into the per-token decode
/// loop and has no batchable entry point).
fn scalar_plane_dequant_pair(q_index: i32, deltas: QuantizerDeltas) -> (i16, i16) {
    crate::quant::plane_dequant_pair(q_index, deltas)
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
/// | G4-β  | hybrid ADST 4×4   | `iht4x4_add`               | yes       |
/// | G4-β  | hybrid ADST 8×8   | `iht8x8_add`               | yes       |
/// | G4.3b / F10 | loopfilter v4 | `lpf_vertical_4`           | yes (Wave F G F10) — H1 gated OFF by default |
/// | G3    | loopfilter h4     | `lpf_horizontal_4`         | yes (Wave G G3) — H1 gated OFF by default |
/// | G4    | loopfilter v8     | `lpf_vertical_8`           | yes (Wave G G4) — H1 gated OFF by default |
/// | G5    | loopfilter h8     | `lpf_horizontal_8`         | yes (Wave G G5) — H1 gated OFF by default |
/// | G6    | loopfilter v16    | `lpf_vertical_16`          | yes (Wave G G6) — H1 gated OFF by default |
/// | G6    | loopfilter h16    | `lpf_horizontal_16`        | yes (Wave G G6) — H1 gated OFF by default |
/// | H1    | loopfilter v4_dual | `lpf_vertical_4_dual`     | scalar (H2 lands AVX2) |
/// | H1    | loopfilter h4_dual | `lpf_horizontal_4_dual`   | scalar (H2 lands AVX2) |
/// | H1    | loopfilter v8_dual | `lpf_vertical_8_dual`     | scalar (H3 lands AVX2) |
/// | H1    | loopfilter h8_dual | `lpf_horizontal_8_dual`   | scalar (H3 lands AVX2) |
/// | H1    | loopfilter v16_dual| `lpf_vertical_16_dual`    | scalar (H4 lands AVX2) |
/// | H1    | loopfilter h16_dual| `lpf_horizontal_16_dual`  | scalar (H4 lands AVX2) |
/// | G4.4a | quant dequant     | `plane_dequant_pair`       | scalar only (architectural — no batchable kernel) |
/// | G4.4b | frame-buffers     | `extend_frame_borders`     | scalar only (memcpy/memset already vectorised) |
///
/// **Family coverage** (per HLD §G4 table line 543-552):
/// 1. frame-buffers — `extend_frame_borders` slot present (G4.4b).
/// 2. entropy decoder — n/a (scalar by design, BoolReader is sequential).
/// 3. transforms inverse — `idct4x4_add` (G4.2) + `idct8x8_add` (G4.1) + `idct16x16_add` (Wave D G4.2 full-kernel rollout) + `idct32x32_add` (Wave E G4-α) + `iht4x4_add` + `iht8x8_add` (Wave E G4-β) all have AVX2 impls.
/// 4. quant dequant — `plane_dequant_pair` slot present (G4.4a).
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
/// lpf_horizontal_16). The deferred slots
/// (plane_dequant_pair, extend_frame_borders) reserve naming +
/// dispatch shape so future AVX2 work can fill them without
/// changing call sites. See per-field docstrings for the per-family
/// deferral rationale.
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

    /// G4.2 — 4×4 inverse DCT + residual add. Signature mirrors the
    /// scalar reference [`crate::transforms::inverse::idct4x4_16_add`].
    /// On x86_64 AVX2 CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::idct4x4_add_avx2`]; otherwise it points
    /// at the scalar reference.
    pub(crate) idct4x4_add: fn(input: &[i32; 16], dst: &mut [u8], stride: usize),

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
    /// AVX2 CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::idct32x32_add_avx2`]; otherwise it points
    /// at the scalar reference. The 34/135-bucket fast paths land in
    /// [`SimdDispatch::idct32x32_34_add`] /
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
    pub(crate) iht4x4_add:
        fn(input: &[i32; 16], dst: &mut [u8], stride: usize, tx_type: TxType),

    /// G4-β — 8×8 hybrid inverse transform. Signature mirrors
    /// [`crate::transforms::inverse::iht8x8_64_add`]. On x86_64 AVX2
    /// CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::iht8x8_add_avx2`]; otherwise it points at
    /// the scalar reference.
    pub(crate) iht8x8_add:
        fn(input: &[i32; 64], dst: &mut [u8], stride: usize, tx_type: TxType),

    /// Wave F F5 — 16×16 hybrid inverse transform (ADST/DCT
    /// combinations). Signature mirrors
    /// [`crate::transforms::inverse::iht16x16_256_add`] — the
    /// `tx_type` arg picks the row/col 1D pair. On x86_64 AVX2 CPUs
    /// this points at the safe wrapper
    /// [`crate::dsp::x86_64::iht16x16_add_avx2`]; otherwise it points
    /// at the scalar reference.
    pub(crate) iht16x16_add:
        fn(input: &[i32; 256], dst: &mut [u8], stride: usize, tx_type: TxType),

    /// G4.3a — DC intra-predictor at `bs = 4`. Signature is the
    /// flat-arg form `(dst, stride, above, left)` (compare:
    /// [`crate::intra_pred::dc_predictor`] takes a `PredCtx` which is
    /// indirect through `above_left` — unused by DC).
    /// On x86_64 AVX2 CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::dc_predictor_4x4_avx2`]; otherwise it
    /// points at a thin scalar adapter that re-builds a `PredCtx`.
    pub(crate) dc_predictor_4x4: fn(dst: &mut [u8], stride: usize, above: &[u8], left: &[u8]),

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

    /// Wave H H1 — dual 4-tap vertical loopfilter. Runs `filter4`
    /// over two adjacent 8-row × 8-col groups (16 rows total) with
    /// independent threshold sets `(blimit0,limit0,thresh0)` vs
    /// `(blimit1,limit1,thresh1)`. Mirrors
    /// [`crate::loopfilter::lpf_vertical_4_dual`]. Scalar-only at
    /// H1 close; H2 lands the AVX2 promotion (lane-doubling of F10).
    pub(crate) lpf_vertical_4_dual: fn(
        pixels: &mut [u8],
        s_index: usize,
        pitch: usize,
        blimit0: u8,
        limit0: u8,
        thresh0: u8,
        blimit1: u8,
        limit1: u8,
        thresh1: u8,
    ),

    /// Wave H H1 — dual 4-tap horizontal loopfilter. Lane-doubled
    /// counterpart of [`SimdDispatch::lpf_horizontal_4`]. Scalar-only
    /// at H1 close; H2 lands the AVX2 promotion (lane-doubling of G3).
    pub(crate) lpf_horizontal_4_dual: fn(
        pixels: &mut [u8],
        s_index: usize,
        pitch: usize,
        blimit0: u8,
        limit0: u8,
        thresh0: u8,
        blimit1: u8,
        limit1: u8,
        thresh1: u8,
    ),

    /// Wave H H1 — dual 8-tap vertical loopfilter. Lane-doubled
    /// counterpart of [`SimdDispatch::lpf_vertical_8`]. Scalar-only
    /// at H1 close; H3 lands the AVX2 promotion (lane-doubling of G4).
    pub(crate) lpf_vertical_8_dual: fn(
        pixels: &mut [u8],
        s_index: usize,
        pitch: usize,
        blimit0: u8,
        limit0: u8,
        thresh0: u8,
        blimit1: u8,
        limit1: u8,
        thresh1: u8,
    ),

    /// Wave H H1 — dual 8-tap horizontal loopfilter. Lane-doubled
    /// counterpart of [`SimdDispatch::lpf_horizontal_8`]. Scalar-only
    /// at H1 close; H3 lands the AVX2 promotion (lane-doubling of G5).
    pub(crate) lpf_horizontal_8_dual: fn(
        pixels: &mut [u8],
        s_index: usize,
        pitch: usize,
        blimit0: u8,
        limit0: u8,
        thresh0: u8,
        blimit1: u8,
        limit1: u8,
        thresh1: u8,
    ),

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

    /// G4.4a — paired (DC, AC) dequant LUT lookup. Signature mirrors
    /// [`crate::quant::plane_dequant_pair`]. **Currently scalar-only**:
    /// architecturally there is no batchable dequant kernel in the
    /// VP9 decoder because per-coefficient dequant
    /// (`v = (value * dqv) >> dq_shift`) is fused into the bool-coder
    /// loop in `decode_coefs` (see
    /// `crates/rvp9-core/src/entropy/detokenize.rs:415`). The slot
    /// here reserves the family name for a future entropy-decoder
    /// rewrite that would separate dequant into a vectorisable pass.
    pub(crate) plane_dequant_pair: fn(q_index: i32, deltas: QuantizerDeltas) -> (i16, i16),

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
        #[cfg(target_arch = "x86_64")]
        let idct8x8_add: fn(&[i32; 64], &mut [u8], usize) = if caps.avx2 {
            crate::dsp::x86_64::idct8x8_add_avx2
        } else {
            scalar_idct8x8_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let idct8x8_add: fn(&[i32; 64], &mut [u8], usize) = {
            // Caps type doesn't carry `avx2` on non-x86_64; silence
            // the unused-variable warning while keeping the parameter
            // name uniform.
            let _ = caps;
            scalar_idct8x8_add
        };

        #[cfg(target_arch = "x86_64")]
        let idct4x4_add: fn(&[i32; 16], &mut [u8], usize) = if caps.avx2 {
            crate::dsp::x86_64::idct4x4_add_avx2
        } else {
            scalar_idct4x4_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let idct4x4_add: fn(&[i32; 16], &mut [u8], usize) = scalar_idct4x4_add;

        #[cfg(target_arch = "x86_64")]
        let idct16x16_add: fn(&[i32; 256], &mut [u8], usize) = if caps.avx2 {
            crate::dsp::x86_64::idct16x16_add_avx2
        } else {
            scalar_idct16x16_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let idct16x16_add: fn(&[i32; 256], &mut [u8], usize) = scalar_idct16x16_add;

        #[cfg(target_arch = "x86_64")]
        let idct32x32_add: fn(&[i32; 1024], &mut [u8], usize) = if caps.avx2 {
            crate::dsp::x86_64::idct32x32_add_avx2
        } else {
            scalar_idct32x32_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let idct32x32_add: fn(&[i32; 1024], &mut [u8], usize) = scalar_idct32x32_add;

        // F9 — 32×32 EOB-bucket fast paths. Finisher (Stage 3) is
        // shared with `idct32x32_add_avx2`; only Stage 1 row count
        // differs (8 / 16 rows respectively). Scalar fallback uses
        // the corresponding scalar reference.
        #[cfg(target_arch = "x86_64")]
        let idct32x32_34_add: fn(&[i32; 1024], &mut [u8], usize) = if caps.avx2 {
            crate::dsp::x86_64::idct32x32_34_add_avx2
        } else {
            scalar_idct32x32_34_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let idct32x32_34_add: fn(&[i32; 1024], &mut [u8], usize) = scalar_idct32x32_34_add;

        #[cfg(target_arch = "x86_64")]
        let idct32x32_135_add: fn(&[i32; 1024], &mut [u8], usize) = if caps.avx2 {
            crate::dsp::x86_64::idct32x32_135_add_avx2
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
        #[cfg(not(target_arch = "x86_64"))]
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
        // Finisher-only AVX2 ports (Stage 3 reused from idct16x16_avx2,
        // row pass reduced to 4 or 8 rows respectively). Bit-exact vs
        // the scalar references; pointer-compatible signatures.
        #[cfg(target_arch = "x86_64")]
        let idct16x16_10_add: fn(&[i32; 256], &mut [u8], usize) = if caps.avx2 {
            crate::dsp::x86_64::idct16x16_10_avx2::idct16x16_10_add_avx2
        } else {
            scalar_idct16x16_10_add
        };
        #[cfg(not(target_arch = "x86_64"))]
        let idct16x16_10_add: fn(&[i32; 256], &mut [u8], usize) = scalar_idct16x16_10_add;

        #[cfg(target_arch = "x86_64")]
        let idct16x16_38_add: fn(&[i32; 256], &mut [u8], usize) = if caps.avx2 {
            crate::dsp::x86_64::idct16x16_38_avx2::idct16x16_38_add_avx2
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
        // The dual-variant slots (H1) remain scalar regardless until
        // H2/H3/H4 lands AVX2 impls.
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
        #[cfg(not(target_arch = "x86_64"))]
        let lpf_vertical_4: fn(&mut [u8], usize, usize, u8, u8, u8) = scalar_lpf_vertical_4;

        // ---- Wave G G3 — horizontal 4-tap loopfilter (AVX2 gated). ----
        #[cfg(target_arch = "x86_64")]
        let lpf_horizontal_4: fn(&mut [u8], usize, usize, u8, u8, u8) =
            if lpf_avx2_mask.contains(LpfAvx2Mask::H4) {
                crate::dsp::x86_64::lpf_horizontal_4_avx2
            } else {
                scalar_lpf_horizontal_4
            };
        #[cfg(not(target_arch = "x86_64"))]
        let lpf_horizontal_4: fn(&mut [u8], usize, usize, u8, u8, u8) = scalar_lpf_horizontal_4;

        // ---- Wave G G4 — 8-tap vertical loopfilter (AVX2 gated). ----
        #[cfg(target_arch = "x86_64")]
        let lpf_vertical_8: fn(&mut [u8], usize, usize, u8, u8, u8) =
            if lpf_avx2_mask.contains(LpfAvx2Mask::V8) {
                crate::dsp::x86_64::lpf_vertical_8_avx2
            } else {
                scalar_lpf_vertical_8
            };
        #[cfg(not(target_arch = "x86_64"))]
        let lpf_vertical_8: fn(&mut [u8], usize, usize, u8, u8, u8) = scalar_lpf_vertical_8;

        // ---- Wave G G5 — 8-tap horizontal loopfilter (AVX2 gated). ----
        #[cfg(target_arch = "x86_64")]
        let lpf_horizontal_8: fn(&mut [u8], usize, usize, u8, u8, u8) =
            if lpf_avx2_mask.contains(LpfAvx2Mask::H8) {
                crate::dsp::x86_64::lpf_horizontal_8_avx2
            } else {
                scalar_lpf_horizontal_8
            };
        #[cfg(not(target_arch = "x86_64"))]
        let lpf_horizontal_8: fn(&mut [u8], usize, usize, u8, u8, u8) = scalar_lpf_horizontal_8;

        // ---- Wave G G6 — 16-tap vertical edge loopfilter (AVX2 gated). ----
        #[cfg(target_arch = "x86_64")]
        let lpf_vertical_16: fn(&mut [u8], usize, usize, u8, u8, u8) =
            if lpf_avx2_mask.contains(LpfAvx2Mask::V16) {
                crate::dsp::x86_64::lpf_vertical_16_avx2
            } else {
                scalar_lpf_vertical_16
            };
        #[cfg(not(target_arch = "x86_64"))]
        let lpf_vertical_16: fn(&mut [u8], usize, usize, u8, u8, u8) = scalar_lpf_vertical_16;

        // ---- Wave G G6 — 16-tap horizontal edge loopfilter (AVX2 gated). ----
        #[cfg(target_arch = "x86_64")]
        let lpf_horizontal_16: fn(&mut [u8], usize, usize, u8, u8, u8) =
            if lpf_avx2_mask.contains(LpfAvx2Mask::H16) {
                crate::dsp::x86_64::lpf_horizontal_16_avx2
            } else {
                scalar_lpf_horizontal_16
            };
        #[cfg(not(target_arch = "x86_64"))]
        let lpf_horizontal_16: fn(&mut [u8], usize, usize, u8, u8, u8) = scalar_lpf_horizontal_16;

        // ---- Wave H H1 — dual-variant loopfilter slots (scalar-only;
        // AVX2 promotion in H2/H3/H4). The slot type is the 9-arg
        // dual signature; the `_ = caps` keeps the caps param wired
        // for future promotion without a non-x86_64 warning. ----
        let lpf_vertical_4_dual: fn(
            &mut [u8],
            usize,
            usize,
            u8,
            u8,
            u8,
            u8,
            u8,
            u8,
        ) = scalar_lpf_vertical_4_dual;
        let lpf_horizontal_4_dual: fn(
            &mut [u8],
            usize,
            usize,
            u8,
            u8,
            u8,
            u8,
            u8,
            u8,
        ) = scalar_lpf_horizontal_4_dual;
        let lpf_vertical_8_dual: fn(
            &mut [u8],
            usize,
            usize,
            u8,
            u8,
            u8,
            u8,
            u8,
            u8,
        ) = scalar_lpf_vertical_8_dual;
        let lpf_horizontal_8_dual: fn(
            &mut [u8],
            usize,
            usize,
            u8,
            u8,
            u8,
            u8,
            u8,
            u8,
        ) = scalar_lpf_horizontal_8_dual;
        // 16-tap duals share a single threshold set (libvpx convention).
        let lpf_vertical_16_dual: fn(&mut [u8], usize, usize, u8, u8, u8) =
            scalar_lpf_vertical_16_dual;
        let lpf_horizontal_16_dual: fn(&mut [u8], usize, usize, u8, u8, u8) =
            scalar_lpf_horizontal_16_dual;

        // ---- G4.4a / G4.4b — scalar-only slots (deferral rationale
        // per field docstrings on `SimdDispatch`). ----
        let plane_dequant_pair: fn(i32, QuantizerDeltas) -> (i16, i16) = scalar_plane_dequant_pair;
        let extend_frame_borders: fn(&mut Yv12Buffer) = scalar_extend_frame_borders;

        Self {
            caps,
            idct8x8_add,
            idct4x4_add,
            idct16x16_add,
            idct32x32_add,
            idct32x32_34_add,
            idct32x32_135_add,
            idct16x16_10_add,
            idct16x16_38_add,
            dc_predictor_4x4,
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
            plane_dequant_pair,
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

    /// Dispatch the 4×4 inverse-DCT + add kernel (G4.2).
    #[inline]
    #[must_use]
    pub fn idct4x4_add(&self) -> fn(&[i32; 16], &mut [u8], usize) {
        self.idct4x4_add
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
    pub fn lpf_vertical_4_dual(
        &self,
    ) -> fn(&mut [u8], usize, usize, u8, u8, u8, u8, u8, u8) {
        self.lpf_vertical_4_dual
    }

    /// Dispatch the dual 4-tap horizontal loopfilter (Wave H H1). Scalar-only
    /// at H1 close; H2 fills with AVX2.
    #[inline]
    #[must_use]
    pub fn lpf_horizontal_4_dual(
        &self,
    ) -> fn(&mut [u8], usize, usize, u8, u8, u8, u8, u8, u8) {
        self.lpf_horizontal_4_dual
    }

    /// Dispatch the dual 8-tap vertical loopfilter (Wave H H1). Scalar-only
    /// at H1 close; H3 fills with AVX2.
    #[inline]
    #[must_use]
    pub fn lpf_vertical_8_dual(
        &self,
    ) -> fn(&mut [u8], usize, usize, u8, u8, u8, u8, u8, u8) {
        self.lpf_vertical_8_dual
    }

    /// Dispatch the dual 8-tap horizontal loopfilter (Wave H H1). Scalar-only
    /// at H1 close; H3 fills with AVX2.
    #[inline]
    #[must_use]
    pub fn lpf_horizontal_8_dual(
        &self,
    ) -> fn(&mut [u8], usize, usize, u8, u8, u8, u8, u8, u8) {
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

    /// Dispatch the paired (DC, AC) dequant LUT lookup (G4.4a). Slot
    /// is currently scalar-only — see field doc on
    /// [`SimdDispatch::plane_dequant_pair`].
    #[inline]
    #[must_use]
    pub fn plane_dequant_pair(&self) -> fn(i32, QuantizerDeltas) -> (i16, i16) {
        self.plane_dequant_pair
    }

    /// Dispatch the frame-border edge-replication kernel (G4.4b). Slot
    /// is currently scalar-only — see field doc on
    /// [`SimdDispatch::extend_frame_borders`].
    #[inline]
    #[must_use]
    pub fn extend_frame_borders(&self) -> fn(&mut Yv12Buffer) {
        self.extend_frame_borders
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

    /// Forcing `avx2 = false` falls back to the scalar reference for
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
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_idct8x8() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 64], &mut [u8], usize) = crate::transforms::inverse::idct8x8_64_add;
        assert_eq!(
            d.idct8x8_add() as usize,
            scalar as usize,
            "avx2=false must install the scalar idct8x8_64_add"
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

    /// Forcing `avx2 = false` falls back to the scalar reference for
    /// the G4.2 idct16x16 slot. See `forced_scalar_caps_installs_scalar_idct8x8`
    /// for the miri-skip rationale.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_scalar_caps_installs_scalar_idct16x16() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 256], &mut [u8], usize) =
            crate::transforms::inverse::idct16x16_256_add;
        assert_eq!(
            d.idct16x16_add() as usize,
            scalar as usize,
            "avx2=false must install the scalar idct16x16_256_add"
        );
    }

    /// Forcing `avx2 = false` falls back to the scalar reference for
    /// the G4-α idct32x32 slot.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn forced_scalar_caps_installs_scalar_idct32x32() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 1024], &mut [u8], usize) =
            crate::transforms::inverse::idct32x32_1024_add;
        assert_eq!(
            d.idct32x32_add() as usize,
            scalar as usize,
            "avx2=false must install the scalar idct32x32_1024_add"
        );
    }

    /// Forcing `avx2 = false` falls back to the scalar reference for
    /// the Wave F F9 `idct32x32_34_add` slot.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn forced_scalar_caps_installs_scalar_idct32x32_34() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 1024], &mut [u8], usize) =
            crate::transforms::inverse::idct32x32_34_add;
        assert_eq!(
            d.idct32x32_34_add() as usize,
            scalar as usize,
            "avx2=false must install the scalar idct32x32_34_add"
        );
    }

    /// Forcing `avx2 = false` falls back to the scalar reference for
    /// the Wave F F9 `idct32x32_135_add` slot.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn forced_scalar_caps_installs_scalar_idct32x32_135() {
        let mut caps = detect();
        caps.avx2 = false;
        let d = SimdDispatch::build(caps);
        let scalar: fn(&[i32; 1024], &mut [u8], usize) =
            crate::transforms::inverse::idct32x32_135_add;
        assert_eq!(
            d.idct32x32_135_add() as usize,
            scalar as usize,
            "avx2=false must install the scalar idct32x32_135_add"
        );
    }

    /// The 32×32 EOB-bucket slots route through their per-bucket fast
    /// paths under both caps states. Under forced AVX2 they install
    /// the corresponding AVX2 wrappers; under forced scalar they
    /// install the scalar references. Mirrors the 16×16 pattern in
    /// `idct16x16_eob_bucket_slots_are_always_scalar` (kept as a
    /// dispatch-shape test so a future scalar-only regression would
    /// surface here).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn idct32x32_eob_bucket_slots_dispatch_correctly() {
        // Scalar caps → scalar fn pointers.
        let mut caps = detect();
        caps.avx2 = false;
        let d_scalar = SimdDispatch::build(caps);
        let scalar_34: fn(&[i32; 1024], &mut [u8], usize) =
            crate::transforms::inverse::idct32x32_34_add;
        let scalar_135: fn(&[i32; 1024], &mut [u8], usize) =
            crate::transforms::inverse::idct32x32_135_add;
        assert_eq!(d_scalar.idct32x32_34_add() as usize, scalar_34 as usize);
        assert_eq!(d_scalar.idct32x32_135_add() as usize, scalar_135 as usize);

        // AVX2 caps → AVX2 wrappers (NOT the scalar pointers).
        let mut caps_avx2 = detect();
        caps_avx2.avx2 = true;
        let d_avx2 = SimdDispatch::build(caps_avx2);
        assert_ne!(
            d_avx2.idct32x32_34_add() as usize,
            scalar_34 as usize,
            "avx2=true must install the AVX2 wrapper for idct32x32_34_add"
        );
        assert_ne!(
            d_avx2.idct32x32_135_add() as usize,
            scalar_135 as usize,
            "avx2=true must install the AVX2 wrapper for idct32x32_135_add"
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

    /// Wave F F3 / F4 — 16×16 EOB-bucket fast paths route to AVX2
    /// safe wrappers when `caps.avx2 == true`, scalar otherwise.
    /// Mirrors the pattern used by `forced_scalar_caps_installs_scalar_idct16x16`.
    /// See `forced_scalar_caps_installs_scalar_idct8x8` for the miri-skip
    /// rationale.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn idct16x16_eob_bucket_slots_dispatch_correctly() {
        let scalar_10: fn(&[i32; 256], &mut [u8], usize) =
            crate::transforms::inverse::idct16x16_10_add;
        let scalar_38: fn(&[i32; 256], &mut [u8], usize) =
            crate::transforms::inverse::idct16x16_38_add;
        let avx2_10: fn(&[i32; 256], &mut [u8], usize) =
            crate::dsp::x86_64::idct16x16_10_add_avx2;
        let avx2_38: fn(&[i32; 256], &mut [u8], usize) =
            crate::dsp::x86_64::idct16x16_38_add_avx2;

        // AVX2 caps → AVX2 wrapper installed (NOT scalar).
        let mut caps = detect();
        caps.avx2 = true;
        let d = SimdDispatch::build(caps);
        assert_ne!(
            d.idct16x16_10_add() as usize,
            scalar_10 as usize,
            "avx2=true must install the AVX2 wrapper for idct16x16_10_add, not scalar"
        );
        assert_eq!(
            d.idct16x16_10_add() as usize,
            avx2_10 as usize,
            "avx2=true must install idct16x16_10_add_avx2"
        );
        assert_ne!(
            d.idct16x16_38_add() as usize,
            scalar_38 as usize,
            "avx2=true must install the AVX2 wrapper for idct16x16_38_add, not scalar"
        );
        assert_eq!(
            d.idct16x16_38_add() as usize,
            avx2_38 as usize,
            "avx2=true must install idct16x16_38_add_avx2"
        );

        // Forced-scalar caps → scalar reference installed.
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let d_scalar = SimdDispatch::build(caps_scalar);
        assert_eq!(
            d_scalar.idct16x16_10_add() as usize,
            scalar_10 as usize,
            "avx2=false must install the scalar idct16x16_10_add"
        );
        assert_eq!(
            d_scalar.idct16x16_38_add() as usize,
            scalar_38 as usize,
            "avx2=false must install the scalar idct16x16_38_add"
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

    /// G4.4a / G4.4b deferred slots are always scalar regardless of
    /// caps. (G4.3b `lpf_vertical_4` was promoted to AVX2 in Wave F
    /// G F10 and `lpf_horizontal_4` in Wave G G3; see
    /// `forced_caps_installs_correct_lpf_vertical_4` and
    /// `forced_caps_installs_correct_lpf_horizontal_4` for those
    /// slots' parity assertions.)
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
            let dq: fn(i32, QuantizerDeltas) -> (i16, i16) = super::scalar_plane_dequant_pair;
            let ef: fn(&mut Yv12Buffer) = super::scalar_extend_frame_borders;
            assert_eq!(d.plane_dequant_pair() as usize, dq as usize);
            assert_eq!(d.extend_frame_borders() as usize, ef as usize);
        }
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
        // 5 known-good AVX2 paths route through their AVX2 wrappers.
        let v4_avx2: Lpf = crate::dsp::x86_64::lpf_vertical_4_avx2;
        let h4_avx2: Lpf = crate::dsp::x86_64::lpf_horizontal_4_avx2;
        let v8_avx2: Lpf = crate::dsp::x86_64::lpf_vertical_8_avx2;
        let h8_avx2: Lpf = crate::dsp::x86_64::lpf_horizontal_8_avx2;
        let h16_avx2: Lpf = crate::dsp::x86_64::lpf_horizontal_16_avx2;
        assert_eq!(d.lpf_vertical_4() as usize, v4_avx2 as usize);
        assert_eq!(d.lpf_horizontal_4() as usize, h4_avx2 as usize);
        assert_eq!(d.lpf_vertical_8() as usize, v8_avx2 as usize);
        assert_eq!(d.lpf_horizontal_8() as usize, h8_avx2 as usize);
        assert_eq!(d.lpf_horizontal_16() as usize, h16_avx2 as usize);
        // The buggy v16 kernel stays scalar pending H7 fix.
        let v16_scalar: Lpf = super::scalar_lpf_vertical_16;
        assert_eq!(d.lpf_vertical_16() as usize, v16_scalar as usize);
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

    /// Wave H H1 + H7 — `parse_lpf_avx2_mask` grammar regression test.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn parse_lpf_avx2_mask_grammar() {
        // Unset / empty -> DEFAULT (5 kernels, v16 off)
        assert_eq!(super::parse_lpf_avx2_mask(None), super::LpfAvx2Mask::DEFAULT);
        assert_eq!(super::parse_lpf_avx2_mask(Some("")), super::LpfAvx2Mask::DEFAULT);
        // "0" -> NONE
        assert_eq!(super::parse_lpf_avx2_mask(Some("0")), super::LpfAvx2Mask::NONE);
        // "1" -> ALL
        assert_eq!(super::parse_lpf_avx2_mask(Some("1")), super::LpfAvx2Mask::ALL);
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

    /// Wave H H1 — the six `_dual` loopfilter slots are scalar-only at
    /// H1 close (regardless of caps). H2/H3/H4 swap individual slots
    /// to AVX2 impls; this test will need its entries removed
    /// as those promotions land. Note 4/8-tap duals are 9-arg
    /// (independent threshold sets) while 16-tap duals are 6-arg
    /// (shared thresholds, libvpx convention). See
    /// `forced_scalar_caps_installs_scalar_idct8x8` for the miri-skip
    /// rationale.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn dual_loopfilter_slots_are_scalar_at_h1_close() {
        type Dual9 = fn(&mut [u8], usize, usize, u8, u8, u8, u8, u8, u8);
        type Dual6 = fn(&mut [u8], usize, usize, u8, u8, u8);
        for &avx2 in &[true, false] {
            let mut caps = detect();
            caps.avx2 = avx2;
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
    }
}
