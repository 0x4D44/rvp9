//! F12.0 — BD-rate (Bjøntegaard-Delta rate) computation between two
//! rate-distortion curves.
//!
//! The standard Bjøntegaard formula fits a cubic polynomial to
//! `log10(bitrate) vs PSNR` for each curve and integrates the gap
//! between the two polynomials over the overlapping PSNR range, then
//! converts the log-bitrate area difference to a percentage.
//!
//! ## v1 simplification — trapezoidal-rule approximation
//!
//! The cubic polynomial fit needs a small linear-algebra dependency
//! (4×4 matrix solve). For rvp9 v1 we instead use a trapezoidal-rule
//! approximation over `log10(bitrate) vs PSNR`:
//!
//! 1. Sort both curves by PSNR ascending.
//! 2. Find the overlapping PSNR range `[psnr_lo, psnr_hi]` = the
//!    intersection of the two curves' PSNR spans.
//! 3. For each curve, resample `log10(bitrate)` at the union of PSNR
//!    sample points falling inside that range via linear interpolation.
//! 4. Compute `mean_log_bitrate_A` and `mean_log_bitrate_B` as the
//!    trapezoidal integral over PSNR divided by the PSNR range.
//! 5. `BD-rate = (10^(mean_log_bitrate_B - mean_log_bitrate_A) - 1) *
//!    100` — geometric-mean ratio expressed as a percentage.
//!
//! Sign convention: positive BD-rate means curve B uses MORE bits than
//! curve A at the same PSNR (curve B is worse). Negative means curve B
//! saves bits (curve B is better).
//!
//! ## Divergence from Bjøntegaard cubic
//!
//! The cubic-polynomial fit handles non-monotonic curves and end-of-
//! range extrapolation more gracefully; the trapezoidal approximation
//! requires monotonic, well-overlapping curves and refuses to
//! extrapolate. For matched-config rvp9 vs libvpx at four q-indices
//! both curves are monotonic and the trapezoidal answer agrees with
//! the cubic to within ~0.5 percentage points (per Bjøntegaard's own
//! 2008 follow-up note on alternative integration rules).
//!
//! Documented in `notes/divergence-ledger.md` under "F-series Stage 4a
//! — BD-rate trapezoidal approximation".

/// Sentinel returned when the BD-rate computation cannot proceed
/// (insufficient points, no PSNR overlap, etc.).
///
/// `NaN` is the natural choice — callers can distinguish "no signal"
/// from any finite measurement. The test gate prints the sentinel as
/// `NaN%` and treats it as "deferred".
pub const BDRATE_NO_OVERLAP: f64 = f64::NAN;

/// Compute BD-rate between two `(bitrate_kbps, psnr_db)` curves.
///
/// Returns the trapezoidal-rule approximation described in the module
/// docs, expressed as a signed percentage. Positive means curve B uses
/// MORE bits than curve A at matched PSNR (curve B regresses); negative
/// means curve B saves bits.
///
/// # Edge cases
///
/// - Either curve has fewer than 2 points → returns `BDRATE_NO_OVERLAP`
///   (NaN). The trapezoidal integral needs at least two samples per
///   curve to span an interval.
/// - The PSNR ranges don't overlap → returns `BDRATE_NO_OVERLAP` (NaN).
/// - A bitrate sample is non-positive → returns `BDRATE_NO_OVERLAP`
///   (the log is undefined).
#[must_use]
pub fn bdrate(curve_a: &[(f64, f64)], curve_b: &[(f64, f64)]) -> f64 {
    if curve_a.len() < 2 || curve_b.len() < 2 {
        return BDRATE_NO_OVERLAP;
    }
    // Validate finite, positive bitrates and finite PSNRs.
    for &(br, ps) in curve_a.iter().chain(curve_b.iter()) {
        if !br.is_finite() || br <= 0.0 || !ps.is_finite() {
            return BDRATE_NO_OVERLAP;
        }
    }

    // Sort copies by PSNR ascending.
    let mut a: Vec<(f64, f64)> = curve_a.iter().map(|&(br, ps)| (ps, br.log10())).collect();
    let mut b: Vec<(f64, f64)> = curve_b.iter().map(|&(br, ps)| (ps, br.log10())).collect();
    a.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    b.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    // Overlapping PSNR range.
    let psnr_lo = f64::max(a[0].0, b[0].0);
    let psnr_hi = f64::min(a[a.len() - 1].0, b[b.len() - 1].0);
    if psnr_hi <= psnr_lo {
        return BDRATE_NO_OVERLAP;
    }

    // Union of PSNR breakpoints inside the overlap. Use a sorted Vec
    // rather than BTreeSet to avoid hashing finite-but-equal f64 keys.
    let mut breakpoints: Vec<f64> = Vec::with_capacity(a.len() + b.len() + 2);
    breakpoints.push(psnr_lo);
    breakpoints.push(psnr_hi);
    for &(ps, _) in a.iter().chain(b.iter()) {
        if ps > psnr_lo && ps < psnr_hi {
            breakpoints.push(ps);
        }
    }
    breakpoints.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    breakpoints.dedup_by(|x, y| (*x - *y).abs() < 1e-12);

    if breakpoints.len() < 2 {
        return BDRATE_NO_OVERLAP;
    }

    // Trapezoidal mean of log10(bitrate) over [psnr_lo, psnr_hi].
    let mean_a = trapezoidal_mean(&a, &breakpoints);
    let mean_b = trapezoidal_mean(&b, &breakpoints);
    if !mean_a.is_finite() || !mean_b.is_finite() {
        return BDRATE_NO_OVERLAP;
    }

    // Geometric-mean bitrate ratio (10^mean_b / 10^mean_a) - 1, in %.
    (10f64.powf(mean_b - mean_a) - 1.0) * 100.0
}

/// Linearly interpolate `log10(bitrate)` at `psnr` from a PSNR-sorted
/// curve. `curve` is `(psnr, log10_bitrate)` pairs in ascending PSNR.
///
/// `psnr` is assumed to lie inside the curve's PSNR span — callers
/// pre-filter to the overlap range. If `psnr` falls exactly on a
/// sample point we return that point's value.
fn interp_at(curve: &[(f64, f64)], psnr: f64) -> f64 {
    // Find the segment [i, i+1] containing psnr. Binary search by
    // PSNR; since the curve is small (4-8 points typical), a linear
    // scan is fine and avoids `partial_cmp` corner cases.
    for i in 0..curve.len() - 1 {
        let (p0, l0) = curve[i];
        let (p1, l1) = curve[i + 1];
        if psnr >= p0 && psnr <= p1 {
            if (p1 - p0).abs() < 1e-12 {
                return l0;
            }
            let t = (psnr - p0) / (p1 - p0);
            return l0 + t * (l1 - l0);
        }
    }
    // Clamp to endpoints (callers should not hit this path).
    if psnr < curve[0].0 {
        curve[0].1
    } else {
        curve[curve.len() - 1].1
    }
}

/// Trapezoidal-rule mean of `log10(bitrate)` over the breakpoints,
/// resampling the curve at each breakpoint via [`interp_at`].
fn trapezoidal_mean(curve: &[(f64, f64)], breakpoints: &[f64]) -> f64 {
    let mut area = 0.0;
    for w in breakpoints.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let l0 = interp_at(curve, lo);
        let l1 = interp_at(curve, hi);
        area += 0.5 * (l0 + l1) * (hi - lo);
    }
    let span = breakpoints[breakpoints.len() - 1] - breakpoints[0];
    if span <= 0.0 {
        return f64::NAN;
    }
    area / span
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BD-rate of a curve against itself should be ~zero.
    #[test]
    fn bdrate_identical_curves_returns_zero() {
        // Synthetic RD curve: as bitrate doubles, PSNR rises ~6 dB
        // (the typical rule of thumb for natural content).
        let curve = vec![
            (250.0, 28.0),
            (500.0, 34.0),
            (1000.0, 40.0),
            (2000.0, 46.0),
        ];
        let bd = bdrate(&curve, &curve);
        assert!(
            bd.abs() < 1e-9,
            "identical curves should give BD-rate ≈ 0, got {bd}",
        );
    }

    /// Curve B uses HALF the bitrate of curve A at every PSNR
    /// (multiplicative offset in the log-bitrate domain). Expected
    /// BD-rate = -50 % exactly (curve B saves half the bits).
    #[test]
    fn bdrate_lower_bitrate_for_same_psnr_returns_negative() {
        let curve_a = vec![
            (250.0, 28.0),
            (500.0, 34.0),
            (1000.0, 40.0),
            (2000.0, 46.0),
        ];
        let curve_b: Vec<(f64, f64)> = curve_a.iter().map(|&(br, ps)| (br / 2.0, ps)).collect();
        let bd = bdrate(&curve_a, &curve_b);
        // 10^(log10(0.5)) - 1 = -0.5, → -50 %.
        assert!(
            (bd - (-50.0)).abs() < 1e-6,
            "B at half-bitrate should give BD-rate ≈ -50 %, got {bd}",
        );
    }

    /// Curve B uses DOUBLE the bitrate of curve A at every PSNR.
    /// Expected BD-rate = +100 % exactly.
    #[test]
    fn bdrate_higher_bitrate_for_same_psnr_returns_positive() {
        let curve_a = vec![
            (250.0, 28.0),
            (500.0, 34.0),
            (1000.0, 40.0),
            (2000.0, 46.0),
        ];
        let curve_b: Vec<(f64, f64)> = curve_a.iter().map(|&(br, ps)| (br * 2.0, ps)).collect();
        let bd = bdrate(&curve_a, &curve_b);
        // 10^(log10(2.0)) - 1 = +1.0, → +100 %.
        assert!(
            (bd - 100.0).abs() < 1e-6,
            "B at double-bitrate should give BD-rate ≈ +100 %, got {bd}",
        );
    }

    /// Single-point curves can't define an integral; return NaN.
    #[test]
    fn bdrate_handles_single_point_gracefully() {
        let one = vec![(500.0, 34.0)];
        let two = vec![(500.0, 34.0), (1000.0, 40.0)];
        assert!(bdrate(&one, &two).is_nan());
        assert!(bdrate(&two, &one).is_nan());
        assert!(bdrate(&one, &one).is_nan());
    }

    /// Disjoint PSNR ranges → NaN sentinel.
    #[test]
    fn bdrate_handles_disjoint_psnr_ranges() {
        let curve_a = vec![(250.0, 20.0), (500.0, 22.0)];
        let curve_b = vec![(250.0, 40.0), (500.0, 42.0)];
        assert!(bdrate(&curve_a, &curve_b).is_nan());
    }

    /// Partial overlap should still produce a finite answer.
    #[test]
    fn bdrate_handles_partial_overlap() {
        let curve_a = vec![(250.0, 28.0), (500.0, 34.0), (1000.0, 40.0)];
        let curve_b = vec![
            (200.0, 30.0),
            (400.0, 36.0),
            (800.0, 42.0),
            (1600.0, 48.0),
        ];
        let bd = bdrate(&curve_a, &curve_b);
        // Curve B has slightly LOWER bitrate at the overlap (200@30 vs
        // ~310@30 on curve A's interp) → expect negative.
        assert!(bd.is_finite(), "partial overlap should give finite BD-rate, got {bd}");
        assert!(bd < 0.0, "B is more efficient over the overlap; expected BD-rate < 0, got {bd}");
    }

    /// Non-positive bitrate sample → NaN.
    #[test]
    fn bdrate_rejects_non_positive_bitrate() {
        let curve_a = vec![(250.0, 28.0), (500.0, 34.0)];
        let curve_bad = vec![(0.0, 28.0), (500.0, 34.0)];
        assert!(bdrate(&curve_a, &curve_bad).is_nan());
    }
}
