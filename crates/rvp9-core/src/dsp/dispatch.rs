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
use crate::transforms::inverse::{
    idct4x4_16_add as scalar_idct4x4_add, idct8x8_64_add as scalar_idct8x8_add,
    idct16x16_256_add as scalar_idct16x16_add, idct32x32_1024_add as scalar_idct32x32_add,
    iht4x4_16_add as scalar_iht4x4_add, iht8x8_64_add as scalar_iht8x8_add,
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
/// | G4.3a | intra-pred DC 4×4 | `dc_predictor_4x4`         | yes       |
/// | G4.3a | hybrid ADST 4×4   | `iht4x4_add`               | scalar only (Wave E follow-up) |
/// | G4.3a | hybrid ADST 8×8   | `iht8x8_add`               | scalar only (Wave E follow-up) |
/// | G4.3b | loopfilter v4     | `lpf_vertical_4`           | scalar only (deferred) |
/// | G4.4a | quant dequant     | `plane_dequant_pair`       | scalar only (architectural — no batchable kernel) |
/// | G4.4b | frame-buffers     | `extend_frame_borders`     | scalar only (memcpy/memset already vectorised) |
///
/// **Family coverage** (per HLD §G4 table line 543-552):
/// 1. frame-buffers — `extend_frame_borders` slot present (G4.4b).
/// 2. entropy decoder — n/a (scalar by design, BoolReader is sequential).
/// 3. transforms inverse — `idct4x4_add` (G4.2) + `idct8x8_add` (G4.1) + `idct16x16_add` (G4.2 full-kernel rollout) + `idct32x32_add` (G4-α full-kernel rollout) all have AVX2 impls; `iht4x4_add` + `iht8x8_add` (G4.3a) hold the ADST hybrid families with scalar fallbacks (Wave E will swap in SIMD).
/// 4. quant dequant — `plane_dequant_pair` slot present (G4.4a).
/// 5. intra-pred — `dc_predictor_4x4` has AVX2 impl (G4.3a).
/// 6. loopfilter — `lpf_vertical_4` slot present (G4.3b).
///
/// 6/6 families have a slot; 5/6 carry a non-scalar Tier-1 impl
/// (idct4x4, idct8x8, idct16x16, idct32x32, dc_predictor_4x4). The
/// hybrid ADST slots (iht4x4, iht8x8) plus the deferred slots
/// (lpf_vertical_4, plane_dequant_pair, extend_frame_borders) reserve
/// naming + dispatch shape so future AVX2 work can fill them without
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
    /// at the scalar reference. The 34/135-bucket fast paths are not
    /// yet ported — only the full `_1024_add` variant is dispatched.
    pub(crate) idct32x32_add: fn(input: &[i32; 1024], dst: &mut [u8], stride: usize),

    /// G4.3a — 4×4 hybrid inverse transform (ADST/DCT combinations).
    /// Signature mirrors [`crate::transforms::inverse::iht4x4_16_add`]
    /// — the `tx_type` arg picks the row/col 1D pair. **Currently
    /// scalar-only**: the AVX2 port of `vp9_iht4x4_16_add_sse2` is
    /// Wave E work; the slot exists so dispatch sites can be written
    /// against `(dispatch.iht4x4_add())(...)` today without a
    /// signature churn when Wave E lands the SIMD impl.
    pub(crate) iht4x4_add:
        fn(input: &[i32; 16], dst: &mut [u8], stride: usize, tx_type: TxType),

    /// G4.3a — 8×8 hybrid inverse transform. Same scalar-only-for-now
    /// deferral as [`SimdDispatch::iht4x4_add`]; signature mirrors
    /// [`crate::transforms::inverse::iht8x8_64_add`].
    pub(crate) iht8x8_add:
        fn(input: &[i32; 64], dst: &mut [u8], stride: usize, tx_type: TxType),

    /// G4.3a — DC intra-predictor at `bs = 4`. Signature is the
    /// flat-arg form `(dst, stride, above, left)` (compare:
    /// [`crate::intra_pred::dc_predictor`] takes a `PredCtx` which is
    /// indirect through `above_left` — unused by DC).
    /// On x86_64 AVX2 CPUs this points at the safe wrapper
    /// [`crate::dsp::x86_64::dc_predictor_4x4_avx2`]; otherwise it
    /// points at a thin scalar adapter that re-builds a `PredCtx`.
    pub(crate) dc_predictor_4x4: fn(dst: &mut [u8], stride: usize, above: &[u8], left: &[u8]),

    /// G4.3b — vertical 4-tap loopfilter edge kernel. Signature
    /// mirrors [`crate::loopfilter::lpf_vertical_4`]. **Currently
    /// scalar-only** — the libvpx SSE2 loopfilter port gathers 8
    /// transverse columns across an 8-row stripe via `_mm_loadu_si64`
    /// (then transpose, then `filter4` lane-parallel); that's a
    /// non-trivial port and is deferred to a follow-up G4 phase.
    /// The slot exists so dispatch sites can be retrofitted to call
    /// `dispatch.lpf_vertical_4()(...)` without a signature churn.
    pub(crate) lpf_vertical_4:
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

        #[cfg(target_arch = "x86_64")]
        let dc_predictor_4x4: fn(&mut [u8], usize, &[u8], &[u8]) = if caps.avx2 {
            crate::dsp::x86_64::dc_predictor_4x4_avx2
        } else {
            scalar_dc_predictor_4x4
        };
        #[cfg(not(target_arch = "x86_64"))]
        let dc_predictor_4x4: fn(&mut [u8], usize, &[u8], &[u8]) = scalar_dc_predictor_4x4;

        // ---- G4.3a hybrid ADST slots — scalar-only in this gap.
        // Wave E ports `vp9_iht4x4_16_add_sse2` / `vp9_iht8x8_64_add_sse2`
        // to AVX2 and swaps these slots' selection clauses to mirror
        // the idct slots above. The slot's signature carries `tx_type`
        // so the scalar adapter routes to the right row / col 1D pair.
        let iht4x4_add: fn(&[i32; 16], &mut [u8], usize, TxType) = scalar_iht4x4_add;
        let iht8x8_add: fn(&[i32; 64], &mut [u8], usize, TxType) = scalar_iht8x8_add;

        // ---- G4.3b / G4.4a / G4.4b — scalar-only slots (deferral
        // rationale per field docstrings on `SimdDispatch`). ----
        let lpf_vertical_4: fn(&mut [u8], usize, usize, u8, u8, u8) = scalar_lpf_vertical_4;
        let plane_dequant_pair: fn(i32, QuantizerDeltas) -> (i16, i16) = scalar_plane_dequant_pair;
        let extend_frame_borders: fn(&mut Yv12Buffer) = scalar_extend_frame_borders;

        Self {
            caps,
            idct8x8_add,
            idct4x4_add,
            idct16x16_add,
            idct32x32_add,
            dc_predictor_4x4,
            iht4x4_add,
            iht8x8_add,
            lpf_vertical_4,
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
    /// rollout). Full-density `_1024_add` variant only; the
    /// 34/135-bucket fast paths are not yet ported.
    #[inline]
    #[must_use]
    pub fn idct32x32_add(&self) -> fn(&[i32; 1024], &mut [u8], usize) {
        self.idct32x32_add
    }

    /// Dispatch the 4×4 hybrid inverse-transform + add kernel
    /// (G4.3a — ADST/DCT combinations). The `tx_type` arg routes to
    /// the right row / col 1D pair. **Scalar-only** in the current
    /// rollout; Wave E lands the AVX2 port.
    #[inline]
    #[must_use]
    pub fn iht4x4_add(&self) -> fn(&[i32; 16], &mut [u8], usize, TxType) {
        self.iht4x4_add
    }

    /// Dispatch the 8×8 hybrid inverse-transform + add kernel
    /// (G4.3a — ADST/DCT combinations). **Scalar-only** in the current
    /// rollout; Wave E lands the AVX2 port.
    #[inline]
    #[must_use]
    pub fn iht8x8_add(&self) -> fn(&[i32; 64], &mut [u8], usize, TxType) {
        self.iht8x8_add
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

    /// Dispatch the vertical 4-tap loopfilter kernel (G4.3b). Slot is
    /// currently scalar-only — see field doc on
    /// [`SimdDispatch::lpf_vertical_4`].
    #[inline]
    #[must_use]
    pub fn lpf_vertical_4(&self) -> fn(&mut [u8], usize, usize, u8, u8, u8) {
        self.lpf_vertical_4
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

    /// G4.3a hybrid ADST slots are always scalar in this gap (Wave E
    /// will swap in AVX2). Verify both caps states route to the scalar
    /// reference. See `forced_scalar_caps_installs_scalar_idct8x8`
    /// for the miri-skip rationale.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn iht_hybrid_slots_are_always_scalar() {
        for &avx2 in &[true, false] {
            let mut caps = detect();
            caps.avx2 = avx2;
            let d = SimdDispatch::build(caps);
            let iht4: fn(&[i32; 16], &mut [u8], usize, TxType) =
                crate::transforms::inverse::iht4x4_16_add;
            let iht8: fn(&[i32; 64], &mut [u8], usize, TxType) =
                crate::transforms::inverse::iht8x8_64_add;
            assert_eq!(
                d.iht4x4_add() as usize,
                iht4 as usize,
                "iht4x4 must be scalar regardless of caps (avx2={avx2})"
            );
            assert_eq!(
                d.iht8x8_add() as usize,
                iht8 as usize,
                "iht8x8 must be scalar regardless of caps (avx2={avx2})"
            );
        }
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

    /// G4.3b / G4.4a / G4.4b slots are always scalar regardless of caps.
    /// See `forced_scalar_caps_installs_scalar_idct8x8` for the miri-skip
    /// rationale.
    #[cfg(target_arch = "x86_64")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn deferred_slots_are_always_scalar() {
        for &avx2 in &[true, false] {
            let mut caps = detect();
            caps.avx2 = avx2;
            let d = SimdDispatch::build(caps);
            let lpf: fn(&mut [u8], usize, usize, u8, u8, u8) = super::scalar_lpf_vertical_4;
            let dq: fn(i32, QuantizerDeltas) -> (i16, i16) = super::scalar_plane_dequant_pair;
            let ef: fn(&mut Yv12Buffer) = super::scalar_extend_frame_borders;
            assert_eq!(d.lpf_vertical_4() as usize, lpf as usize);
            assert_eq!(d.plane_dequant_pair() as usize, dq as usize);
            assert_eq!(d.extend_frame_borders() as usize, ef as usize);
        }
    }
}
