//! RD multiplier (Lagrange `lambda`) computation.
//!
//! Mirrors libvpx `vp9_compute_rd_mult_based_on_qindex`
//! (`vp9_rd.c:241-276`) for the 8-bit non-HBD non-ext-rc path. The
//! formula is:
//!
//! ```text
//!     q       = dc_quant(qindex, 0)
//!     rdmult0 = q * q
//!     rdmult  = (rdmult0 * (a + b * qindex)) as i32
//! ```
//!
//! where `(a, b)` depends on the frame type:
//! - Key-frame:   `(4.35, 0.001)` — `def_kf_rd_multiplier`.
//! - ARF / Golden: `(4.25, 0.001)` — `def_arf_rd_multiplier`.
//! - Inter:       `(4.15, 0.001)` — `def_inter_rd_multiplier`.
//!
//! The result is clamped to `>= 1` and returned as `u32`.
//!
//! ## Path-A simplifications
//!
//! libvpx's `vp9_compute_rd_mult` also calls `modulate_rdmult`, which
//! adjusts the multiplier for two-pass / golden-frame boost. Path-A
//! MVP is single-pass CBR, so `modulate_rdmult` is a no-op and we
//! omit it. The ext-rc override and HBD adjustment are also out of
//! scope for v1.

use crate::quant::dc_quant;

/// Which frame type to use when picking the `(a, b)` coefficients.
/// Mirrors the `KEY_FRAME` / `refresh_golden_frame|refresh_alt_ref_frame`
/// / inter branches in libvpx.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FrameTypeForRd {
    /// Keyframe — use `(4.35, 0.001)`.
    Key,
    /// ARF or golden frame refresh — use `(4.25, 0.001)`.
    ArfGolden,
    /// Plain inter frame — use `(4.15, 0.001)`.
    Inter,
}

#[inline]
fn def_rd_multiplier(frame: FrameTypeForRd, qindex: i32) -> f64 {
    let (a, b) = match frame {
        FrameTypeForRd::Key => (4.35, 0.001),
        FrameTypeForRd::ArfGolden => (4.25, 0.001),
        FrameTypeForRd::Inter => (4.15, 0.001),
    };
    a + b * (qindex as f64)
}

/// Compute the qindex-derived RD multiplier `lambda`. Mirrors libvpx
/// `vp9_compute_rd_mult_based_on_qindex` (`vp9_rd.c:241-276`).
///
/// Returns the bare quantizer-based mult; for Path-A MVP this is also
/// what `vp9_compute_rd_mult` would return, since we don't apply
/// `modulate_rdmult`.
#[must_use]
pub fn compute_rd_mult_based_on_qindex(qindex: i32, frame: FrameTypeForRd) -> u32 {
    let q = i32::from(dc_quant(qindex, 0));
    let rdmult = q * q;
    let mult = def_rd_multiplier(frame, qindex);
    let scaled = (rdmult as f64) * mult;
    // Cast back. libvpx uses `(int)((double)rdmult * def_rd_q_mult)`
    // — a C truncating cast. `f64 as i32` in Rust is saturating, which
    // diverges only for absurd inputs; for `qindex ∈ [0, 255]` the
    // value never approaches i32::MAX.
    let i = scaled as i32;
    let clamped = if i > 0 { i } else { 1 };
    clamped as u32
}

/// Convenience: same as [`compute_rd_mult_based_on_qindex`].
/// Mirrors libvpx `vp9_compute_rd_mult` (`vp9_rd.c:294-302`) sans
/// `modulate_rdmult` (which is a no-op in single-pass Path-A MVP).
#[must_use]
pub fn compute_rd_mult(qindex: i32, frame: FrameTypeForRd) -> u32 {
    compute_rd_mult_based_on_qindex(qindex, frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At `qindex = 0`, `dc_quant(0) = 4` (per `DC_QLOOKUP[0]` in
    /// libvpx); `q * q = 16`; multiplier = `4.15 + 0.001 * 0 = 4.15`;
    /// so inter-frame rdmult = `(16 * 4.15) as i32 = 66`.
    #[test]
    fn inter_q0_matches_hand_derivation() {
        let v = compute_rd_mult(0, FrameTypeForRd::Inter);
        assert_eq!(v, 66, "inter@q0: expected 66, got {v}");
    }

    /// At `qindex = 0`, key-frame multiplier = `4.35`; rdmult =
    /// `(16 * 4.35) as i32 = 69`.
    #[test]
    fn key_q0_matches_hand_derivation() {
        let v = compute_rd_mult(0, FrameTypeForRd::Key);
        assert_eq!(v, 69);
    }

    /// At `qindex = 0`, ARF multiplier = `4.25`; rdmult =
    /// `(16 * 4.25) as i32 = 68`.
    #[test]
    fn arf_q0_matches_hand_derivation() {
        let v = compute_rd_mult(0, FrameTypeForRd::ArfGolden);
        assert_eq!(v, 68);
    }

    /// Sanity: at `qindex = 255` the RD multiplier should be a large
    /// positive number — libvpx's `DC_QLOOKUP[255] = 1336`, q^2 =
    /// 1_784_896, key-multiplier = `4.35 + 0.255 = 4.605` → ~8.2M.
    #[test]
    fn high_q_yields_large_positive_value() {
        let v = compute_rd_mult(255, FrameTypeForRd::Key);
        // 1336^2 = 1_784_896. (4.35 + 0.255) = 4.605.
        // 1_784_896 * 4.605 ≈ 8_219_446 (truncating).
        assert!(v > 8_000_000, "expected >8M at q=255 KF, got {v}");
        assert!(v < 10_000_000, "value sanity bound, got {v}");
    }

    /// Sanity: RD multiplier is monotone non-decreasing in qindex (q
    /// rises monotonically, so q^2 does too, and `(4.15 + 0.001 * q)`
    /// is monotone).
    #[test]
    fn monotone_in_qindex_inter() {
        let mut prev = compute_rd_mult(0, FrameTypeForRd::Inter);
        for q in 1..=255i32 {
            let v = compute_rd_mult(q, FrameTypeForRd::Inter);
            assert!(v >= prev, "non-monotone at q={q}: prev={prev}, v={v}");
            prev = v;
        }
    }

    /// At qindex = 128, libvpx's `DC_QLOOKUP[128] = 142`.
    /// q^2 = 20_164. inter mult = `4.15 + 0.128 = 4.278`.
    /// `(20164 * 4.278) as i32 = 86_261` (truncating).
    #[test]
    fn q128_within_expected_range() {
        let v = compute_rd_mult(128, FrameTypeForRd::Inter);
        assert!(
            (80_000..90_000).contains(&v),
            "q=128 inter mult out of expected range, got {v}"
        );
    }

    /// Key-frame multiplier should always be >= ARF >= inter at the
    /// same qindex (since 4.35 > 4.25 > 4.15 and the linear slopes
    /// are identical).
    #[test]
    fn key_arf_inter_ordering() {
        for q in [0i32, 60, 128, 200, 255] {
            let key = compute_rd_mult(q, FrameTypeForRd::Key);
            let arf = compute_rd_mult(q, FrameTypeForRd::ArfGolden);
            let inter = compute_rd_mult(q, FrameTypeForRd::Inter);
            assert!(key >= arf, "key {key} < arf {arf} at q={q}");
            assert!(arf >= inter, "arf {arf} < inter {inter} at q={q}");
        }
    }
}
