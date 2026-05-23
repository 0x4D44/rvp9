//! F12.0 — BD-rate (Bjøntegaard-Delta rate) computation between two
//! rate-distortion curves.
//!
//! Implements the standard Bjøntegaard 2001 procedure:
//!
//! 1. For each curve, fit a least-squares cubic polynomial
//!    `ln(bitrate) = a + b*P + c*P^2 + d*P^3` where `P` is PSNR.
//! 2. Identify the overlapping PSNR range
//!    `[P_lo, P_hi] = [max(min_a, min_b), min(max_a, max_b)]`.
//! 3. Integrate each cubic over that range in closed form, take the
//!    difference, divide by the range width to get mean log-bitrate gap.
//! 4. Convert to a percentage:
//!    `BD-rate = (exp(mean_diff) - 1) * 100`.
//!
//! Sign convention: positive BD-rate means curve B uses MORE bits than
//! curve A at the same PSNR (curve B is worse). Negative means curve B
//! saves bits (curve B is better). Stage 4b's AQ.5 gate calls this as
//! `bdrate(curve_off, curve_on)` and expects a negative number when
//! AQ-on is more efficient.
//!
//! Reference: G. Bjøntegaard, "Calculation of Average PSNR Differences
//! between RD-curves", ITU-T VCEG Q6/16 Doc. VCEG-M33, 2001. Companion
//! Excel macro is at
//! <https://www.itu.int/wftp3/av-arch/video-site/0904_Sap/VCEG-AI11.zip>.
//! libvpx's `tools_common.c` implements the same formula.
//!
//! ## Why cubic rather than trapezoidal
//!
//! On sparse RD curves (4-6 points) trapezoidal integration can drift
//! ±1-2 percentage points away from the Bjøntegaard cubic — large
//! enough to swing the Stage 4b AQ.5 ±0.3 % exit gate. The cubic fit
//! is the industry-standard reference. For n = 4 the cubic interpolates
//! the points exactly; for n > 4 it is a least-squares fit.

/// Sentinel returned when the BD-rate computation cannot proceed
/// (insufficient points, no PSNR overlap, numerical instability).
///
/// `NaN` is the natural choice — callers can distinguish "no signal"
/// from any finite measurement. The test gate prints the sentinel as
/// `NaN%` and treats it as "deferred".
pub const BDRATE_NO_OVERLAP: f64 = f64::NAN;

/// Compute Bjøntegaard BD-rate between two `(bitrate_kbps, psnr_db)`
/// curves.
///
/// Returns the cubic-polynomial fit answer as a signed percentage.
/// Positive means curve B uses MORE bits than curve A at matched PSNR
/// (curve B regresses); negative means curve B saves bits.
///
/// # Edge cases
///
/// - Either curve has fewer than 4 points → returns `BDRATE_NO_OVERLAP`
///   (NaN). A cubic least-squares fit needs at least 4 samples.
/// - The PSNR ranges don't overlap → returns `BDRATE_NO_OVERLAP`.
/// - Any bitrate sample is non-positive or non-finite → returns NaN
///   (the natural log is undefined).
/// - The normal-equations matrix is numerically singular (e.g. all
///   PSNRs collinear) → returns NaN.
#[must_use]
pub fn bdrate(curve_a: &[(f64, f64)], curve_b: &[(f64, f64)]) -> f64 {
    if curve_a.len() < 4 || curve_b.len() < 4 {
        return BDRATE_NO_OVERLAP;
    }
    // Validate finite, positive bitrates and finite PSNRs.
    for &(br, ps) in curve_a.iter().chain(curve_b.iter()) {
        if !br.is_finite() || br <= 0.0 || !ps.is_finite() {
            return BDRATE_NO_OVERLAP;
        }
    }

    // Build (x = PSNR, y = ln(bitrate)) point sets.
    let a_xy: Vec<(f64, f64)> = curve_a.iter().map(|&(br, ps)| (ps, br.ln())).collect();
    let b_xy: Vec<(f64, f64)> = curve_b.iter().map(|&(br, ps)| (ps, br.ln())).collect();

    // Cubic fit for each curve.
    let Some(coef_a) = polyfit3(&a_xy) else {
        return BDRATE_NO_OVERLAP;
    };
    let Some(coef_b) = polyfit3(&b_xy) else {
        return BDRATE_NO_OVERLAP;
    };

    // Overlap range in PSNR.
    let a_min = a_xy.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let a_max = a_xy.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let b_min = b_xy.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let b_max = b_xy.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let lo = a_min.max(b_min);
    let hi = a_max.min(b_max);
    // lo and hi are derived from finite PSNRs (validated above), so a
    // direct comparison is safe — no NaN concern. Use `<=` rather than
    // `!(hi > lo)` to satisfy clippy::neg_cmp_op_on_partial_ord.
    if hi <= lo {
        return BDRATE_NO_OVERLAP;
    }

    // Mean log-bitrate over the overlap for each cubic, then ratio.
    let mean_a = integrate_cubic(&coef_a, lo, hi) / (hi - lo);
    let mean_b = integrate_cubic(&coef_b, lo, hi) / (hi - lo);
    if !mean_a.is_finite() || !mean_b.is_finite() {
        return BDRATE_NO_OVERLAP;
    }
    ((mean_b - mean_a).exp() - 1.0) * 100.0
}

/// Linearly interpolate a curve's PSNR at a given bitrate using
/// `ln(bitrate)` as the independent variable.
///
/// The curve is treated as a set of `(bitrate_kbps, psnr_db)` samples.
/// `bitrate_kbps` is queried at `target_kbps` and the corresponding PSNR
/// is returned by piecewise-linear interpolation between the two
/// bracketing samples (in `ln(bitrate)` space — the same domain the
/// Bjøntegaard cubic fit uses).
///
/// Returns `None` if:
///
/// - `target_kbps` is not finite or non-positive.
/// - The curve has fewer than 2 points.
/// - `target_kbps` is outside the curve's bitrate range (no
///   extrapolation — caller is expected to skip out-of-range queries).
/// - Any sample has a non-positive or non-finite bitrate / PSNR.
///
/// This is the fallback metric for the BD-rate gate when the cubic
/// integral is undefined (no PSNR overlap between curves). Linear
/// interpolation in `ln(bitrate)` is appropriate here because (a) the
/// underlying RD relationship is approximately linear in
/// `(ln(bitrate), PSNR)` at short ranges, and (b) on the 4-point gate
/// curves a cubic re-fit for a single query point would overshoot near
/// the endpoints.
#[must_use]
pub fn psnr_at_bitrate(curve: &[(f64, f64)], target_kbps: f64) -> Option<f64> {
    if !target_kbps.is_finite() || target_kbps <= 0.0 || curve.len() < 2 {
        return None;
    }
    // Validate samples are well-formed.
    for &(br, ps) in curve {
        if !br.is_finite() || br <= 0.0 || !ps.is_finite() {
            return None;
        }
    }
    // Sort a copy by bitrate ascending (the gate's curve is built in
    // target-bitrate order, but RC drift can shuffle the actual
    // bitrates).
    let mut sorted: Vec<(f64, f64)> = curve.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let br_min = sorted[0].0;
    let br_max = sorted[sorted.len() - 1].0;
    if target_kbps < br_min || target_kbps > br_max {
        return None;
    }

    // Find the bracketing pair.
    let lt = target_kbps.ln();
    for w in sorted.windows(2) {
        let (br0, ps0) = w[0];
        let (br1, ps1) = w[1];
        if target_kbps >= br0 && target_kbps <= br1 {
            let l0 = br0.ln();
            let l1 = br1.ln();
            if (l1 - l0).abs() < 1e-12 {
                // Coincident bitrates — average PSNR.
                return Some((ps0 + ps1) * 0.5);
            }
            let t = (lt - l0) / (l1 - l0);
            return Some(ps0 + t * (ps1 - ps0));
        }
    }
    // Numerically possible to fall through here if the windows iteration
    // didn't find a bracket (e.g. all samples equal). Conservative
    // fallback: nearest endpoint.
    Some(sorted[sorted.len() - 1].1)
}

/// Definite integral of the cubic `c[0] + c[1]*x + c[2]*x^2 + c[3]*x^3`
/// over `[lo, hi]`, computed in closed form.
fn integrate_cubic(c: &[f64; 4], lo: f64, hi: f64) -> f64 {
    let antideriv = |x: f64| -> f64 {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x3 * x;
        c[0] * x + c[1] * x2 / 2.0 + c[2] * x3 / 3.0 + c[3] * x4 / 4.0
    };
    antideriv(hi) - antideriv(lo)
}

/// Least-squares cubic fit `y = c0 + c1*x + c2*x^2 + c3*x^3` via the
/// 4×4 normal equations, solved by Gauss-Jordan with partial pivoting.
///
/// Returns `None` if the normal-equations matrix is numerically
/// singular (e.g. all x values equal, or fewer effective independent
/// samples than columns).
///
/// The index-based loops are deliberate: the matrix-elimination
/// invariants are clearer with explicit `m[i][j]` than iterator chains.
#[allow(clippy::needless_range_loop)]
fn polyfit3(pts: &[(f64, f64)]) -> Option<[f64; 4]> {
    // Build the normal-equations augmented matrix [A | b] where
    //   A[i][j] = sum(x^(i+j)) for i,j in 0..4
    //   b[i]    = sum(y * x^i)
    let mut m = [[0.0f64; 5]; 4];
    for &(x, y) in pts {
        let mut xp = [1.0f64; 7];
        for i in 1..7 {
            xp[i] = xp[i - 1] * x;
        }
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] += xp[i + j];
            }
            m[i][4] += y * xp[i];
        }
    }
    // Gauss-Jordan elimination with partial pivoting.
    for i in 0..4 {
        // Find pivot row.
        let mut max_row = i;
        for k in (i + 1)..4 {
            if m[k][i].abs() > m[max_row][i].abs() {
                max_row = k;
            }
        }
        m.swap(i, max_row);
        let pivot = m[i][i];
        if pivot.abs() < 1e-12 {
            return None;
        }
        for j in 0..5 {
            m[i][j] /= pivot;
        }
        for k in 0..4 {
            if k != i {
                let f = m[k][i];
                for j in 0..5 {
                    m[k][j] -= f * m[i][j];
                }
            }
        }
    }
    let coefs = [m[0][4], m[1][4], m[2][4], m[3][4]];
    for c in coefs {
        if !c.is_finite() {
            return None;
        }
    }
    Some(coefs)
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
            bd.abs() < 1e-6,
            "identical curves should give BD-rate ≈ 0, got {bd}",
        );
    }

    /// Curve B uses HALF the bitrate of curve A at every PSNR
    /// (multiplicative offset in the log-bitrate domain). Expected
    /// BD-rate = -50 % exactly (curve B saves half the bits). The cubic
    /// fit reproduces this exactly because a vertical translation
    /// `y -> y + ln(0.5)` is captured by the constant term.
    #[test]
    fn bdrate_lower_bitrate_for_same_psnr_returns_negative_50pct() {
        let curve_a = vec![
            (250.0, 28.0),
            (500.0, 34.0),
            (1000.0, 40.0),
            (2000.0, 46.0),
        ];
        let curve_b: Vec<(f64, f64)> = curve_a.iter().map(|&(br, ps)| (br / 2.0, ps)).collect();
        let bd = bdrate(&curve_a, &curve_b);
        // exp(ln(0.5)) - 1 = -0.5  →  -50 %.
        assert!(
            (bd - (-50.0)).abs() < 1e-6,
            "B at half-bitrate should give BD-rate ≈ -50 %, got {bd}",
        );
    }

    /// Curve B uses DOUBLE the bitrate of curve A at every PSNR.
    /// Expected BD-rate = +100 % exactly.
    #[test]
    fn bdrate_higher_bitrate_for_same_psnr_returns_positive_100pct() {
        let curve_a = vec![
            (250.0, 28.0),
            (500.0, 34.0),
            (1000.0, 40.0),
            (2000.0, 46.0),
        ];
        let curve_b: Vec<(f64, f64)> = curve_a.iter().map(|&(br, ps)| (br * 2.0, ps)).collect();
        let bd = bdrate(&curve_a, &curve_b);
        assert!(
            (bd - 100.0).abs() < 1e-6,
            "B at double-bitrate should give BD-rate ≈ +100 %, got {bd}",
        );
    }

    /// 10% bitrate inflation (curve_b = 1.10 * curve_a) → BD-rate ≈ +10 %.
    /// Multiplicative shifts in bitrate are exact under the cubic fit
    /// because they collapse to a translation in `ln(bitrate)`.
    #[test]
    fn bdrate_ten_percent_inflation_returns_plus_ten_pct() {
        let curve_a = vec![
            (300.0, 30.0),
            (600.0, 35.0),
            (1200.0, 40.0),
            (2400.0, 45.0),
            (4800.0, 50.0),
        ];
        let curve_b: Vec<(f64, f64)> = curve_a.iter().map(|&(br, ps)| (br * 1.10, ps)).collect();
        let bd = bdrate(&curve_a, &curve_b);
        assert!(
            (bd - 10.0).abs() < 1e-6,
            "B at 1.10× bitrate should give BD-rate ≈ +10 %, got {bd}",
        );
    }

    /// Single-point and triple-point curves can't define a cubic fit
    /// (need ≥4 samples). Return NaN.
    #[test]
    fn bdrate_handles_insufficient_points() {
        let three = vec![(250.0, 28.0), (500.0, 34.0), (1000.0, 40.0)];
        let four = vec![
            (250.0, 28.0),
            (500.0, 34.0),
            (1000.0, 40.0),
            (2000.0, 46.0),
        ];
        assert!(bdrate(&three, &four).is_nan());
        assert!(bdrate(&four, &three).is_nan());
        assert!(bdrate(&three, &three).is_nan());
    }

    /// Disjoint PSNR ranges → NaN sentinel.
    #[test]
    fn bdrate_handles_disjoint_psnr_ranges() {
        let curve_a = vec![
            (250.0, 20.0),
            (300.0, 21.0),
            (400.0, 22.0),
            (500.0, 23.0),
        ];
        let curve_b = vec![
            (250.0, 40.0),
            (300.0, 41.0),
            (400.0, 42.0),
            (500.0, 43.0),
        ];
        assert!(bdrate(&curve_a, &curve_b).is_nan());
    }

    /// Partial overlap should still produce a finite answer.
    /// Curve B has slightly lower bitrate at the overlap → expect negative.
    #[test]
    fn bdrate_handles_partial_overlap() {
        let curve_a = vec![
            (250.0, 28.0),
            (500.0, 34.0),
            (1000.0, 40.0),
            (1500.0, 43.0),
        ];
        let curve_b = vec![
            (200.0, 30.0),
            (400.0, 36.0),
            (800.0, 42.0),
            (1600.0, 48.0),
        ];
        let bd = bdrate(&curve_a, &curve_b);
        assert!(
            bd.is_finite(),
            "partial overlap should give finite BD-rate, got {bd}"
        );
        assert!(
            bd < 0.0,
            "B is more efficient over the overlap; expected BD-rate < 0, got {bd}",
        );
    }

    /// Non-positive bitrate sample → NaN.
    #[test]
    fn bdrate_rejects_non_positive_bitrate() {
        let curve_a = vec![
            (250.0, 28.0),
            (500.0, 34.0),
            (1000.0, 40.0),
            (2000.0, 46.0),
        ];
        let curve_bad = vec![
            (0.0, 28.0),
            (500.0, 34.0),
            (1000.0, 40.0),
            (2000.0, 46.0),
        ];
        assert!(bdrate(&curve_a, &curve_bad).is_nan());
    }

    /// Sign-convention sanity check matching the AQ.5 contract:
    /// `bdrate(curve_off, curve_on)` is negative when AQ-on is more
    /// efficient (uses fewer bits at matched PSNR).
    #[test]
    fn bdrate_sign_matches_aq5_contract() {
        // curve_off (AQ disabled) is the baseline.
        let curve_off = vec![
            (500.0, 30.0),
            (1000.0, 35.0),
            (2000.0, 40.0),
            (4000.0, 45.0),
        ];
        // curve_on (AQ enabled) at 90 % of the bitrate for the same PSNR
        // — i.e. AQ saves 10 % of the bits.
        let curve_on: Vec<(f64, f64)> = curve_off
            .iter()
            .map(|&(br, ps)| (br * 0.9, ps))
            .collect();
        let bd = bdrate(&curve_off, &curve_on);
        // exp(ln(0.9)) - 1 ≈ -0.10 → -10 %.
        assert!(
            (bd - (-10.0)).abs() < 1e-6,
            "AQ-on at 0.9× bitrate should give BD-rate ≈ -10 %, got {bd}",
        );
        assert!(bd < 0.0, "AQ-on more efficient must yield negative BD-rate, got {bd}");
    }

    /// Polyfit3 reproduces the coefficients of a known cubic exactly
    /// (within float epsilon) when given samples drawn from that cubic.
    #[test]
    fn polyfit3_recovers_known_cubic() {
        // y = 1 + 2x + 3x^2 + 4x^3
        let f = |x: f64| 1.0 + 2.0 * x + 3.0 * x * x + 4.0 * x * x * x;
        let xs = [0.5, 1.5, 2.5, 3.5, 4.5];
        let pts: Vec<(f64, f64)> = xs.iter().map(|&x| (x, f(x))).collect();
        let c = polyfit3(&pts).expect("non-singular");
        assert!((c[0] - 1.0).abs() < 1e-9, "c0={}", c[0]);
        assert!((c[1] - 2.0).abs() < 1e-9, "c1={}", c[1]);
        assert!((c[2] - 3.0).abs() < 1e-9, "c2={}", c[2]);
        assert!((c[3] - 4.0).abs() < 1e-9, "c3={}", c[3]);
    }

    /// `psnr_at_bitrate` reproduces an endpoint exactly.
    #[test]
    fn psnr_at_bitrate_endpoints_exact() {
        let curve = vec![
            (250.0, 28.0),
            (500.0, 34.0),
            (1000.0, 40.0),
            (2000.0, 46.0),
        ];
        let p = psnr_at_bitrate(&curve, 250.0).expect("in range");
        assert!((p - 28.0).abs() < 1e-9, "p={p}");
        let p = psnr_at_bitrate(&curve, 2000.0).expect("in range");
        assert!((p - 46.0).abs() < 1e-9, "p={p}");
    }

    /// `psnr_at_bitrate` interpolates the midpoint in log-bitrate space.
    /// Between (250, 28) and (500, 34) the geometric mean bitrate is
    /// `exp((ln 250 + ln 500)/2)` ≈ 353.55 kbps; the linear-in-log
    /// interpolant at that point is the arithmetic mean PSNR = 31.0 dB.
    #[test]
    fn psnr_at_bitrate_log_midpoint() {
        let curve = vec![
            (250.0, 28.0),
            (500.0, 34.0),
            (1000.0, 40.0),
            (2000.0, 46.0),
        ];
        let mid_br = (250.0_f64.ln() + 500.0_f64.ln()).exp().sqrt();
        let p = psnr_at_bitrate(&curve, mid_br).expect("in range");
        assert!((p - 31.0).abs() < 1e-9, "p={p} at mid_br={mid_br}");
    }

    /// `psnr_at_bitrate` refuses to extrapolate beyond the curve range.
    #[test]
    fn psnr_at_bitrate_out_of_range_returns_none() {
        let curve = vec![
            (250.0, 28.0),
            (500.0, 34.0),
            (1000.0, 40.0),
            (2000.0, 46.0),
        ];
        assert!(psnr_at_bitrate(&curve, 100.0).is_none());
        assert!(psnr_at_bitrate(&curve, 5000.0).is_none());
        assert!(psnr_at_bitrate(&curve, 0.0).is_none());
        assert!(psnr_at_bitrate(&curve, f64::NAN).is_none());
    }

    /// Curve out of order in input — `psnr_at_bitrate` should still work
    /// (it sorts internally).
    #[test]
    fn psnr_at_bitrate_unsorted_input_handled() {
        let curve = vec![
            (2000.0, 46.0),
            (250.0, 28.0),
            (1000.0, 40.0),
            (500.0, 34.0),
        ];
        let p = psnr_at_bitrate(&curve, 500.0).expect("in range");
        assert!((p - 34.0).abs() < 1e-9, "p={p}");
    }

    /// Degenerate input (all x values identical) → polyfit returns None,
    /// bdrate returns NaN.
    #[test]
    fn bdrate_returns_nan_on_collinear_input() {
        let curve_a = vec![
            (500.0, 35.0),
            (600.0, 35.0),
            (700.0, 35.0),
            (800.0, 35.0),
        ];
        let curve_b = vec![
            (500.0, 35.0),
            (600.0, 35.0),
            (700.0, 35.0),
            (800.0, 35.0),
        ];
        assert!(bdrate(&curve_a, &curve_b).is_nan());
    }
}
