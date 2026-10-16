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

fn curve_samples_are_valid(curve: &[(f64, f64)]) -> bool {
    curve
        .iter()
        .all(|&(br, ps)| br.is_finite() && br > 0.0 && ps.is_finite())
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
/// - Any sample bitrate is non-positive or non-finite.
/// - Any sample PSNR is non-finite (zero and negative PSNR values are valid).
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
    if !curve_samples_are_valid(curve) {
        return None;
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

/// Summary of matched-bitrate PSNR-delta evaluation between an rvp9
/// curve and a libvpx reference curve. Produced by
/// [`psnr_delta_summary`].
///
/// `count_out_of_range` rvp9 points fall outside the libvpx curve's
/// bitrate span and are *excluded* from the mean/min/max — there is
/// no extrapolation. Callers should report the count separately.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsnrDeltaSummary {
    /// Arithmetic mean of in-range Δ values (dB). Primary Track F
    /// signal — positive means rvp9 trails libvpx.
    pub mean_delta_db: f64,
    /// Minimum in-range Δ (dB).
    pub min_delta_db: f64,
    /// Maximum in-range Δ (dB).
    pub max_delta_db: f64,
    /// Number of rvp9 points where libvpx's curve covers the bitrate.
    pub count_in_range: usize,
    /// Number of rvp9 points outside libvpx's bitrate range (skipped,
    /// not extrapolated).
    pub count_out_of_range: usize,
}

/// Compute the libvpx-vs-rvp9 PSNR-delta summary at the rvp9 curve's
/// operating points. `libvpx_curve` is the reference; `rvp9_curve` is
/// queried point-by-point against it via [`psnr_at_bitrate`].
///
/// For each rvp9 sample `(br, psnr_rvp9)` the function evaluates
/// libvpx's PSNR at `br` by piecewise-linear interpolation in
/// `ln(bitrate)` space and computes
/// `Δ = psnr_libvpx_interp − psnr_rvp9`. Positive Δ means rvp9 trails
/// libvpx at that operating point; negative Δ means rvp9 beats
/// libvpx. The summary stats are taken over the **in-range subset
/// only** — out-of-range rvp9 points are counted but never
/// extrapolated.
///
/// Returns `None` when no rvp9 point falls inside the libvpx bitrate
/// range (caller should report DEFERRED — there is no signal), when
/// either curve contains invalid samples, or when arithmetic produces a
/// non-finite interpolation or summary. When `Some`, `count_in_range >= 1`.
#[must_use]
pub fn psnr_delta_summary(
    libvpx_curve: &[(f64, f64)],
    rvp9_curve: &[(f64, f64)],
) -> Option<PsnrDeltaSummary> {
    if !curve_samples_are_valid(rvp9_curve) {
        return None;
    }

    let mut deltas: Vec<f64> = Vec::with_capacity(rvp9_curve.len());
    let mut count_out_of_range: usize = 0;
    for &(br_r, psnr_r) in rvp9_curve {
        match psnr_at_bitrate(libvpx_curve, br_r) {
            Some(psnr_l) => {
                if !psnr_l.is_finite() {
                    return None;
                }
                let delta = psnr_l - psnr_r;
                if !delta.is_finite() {
                    return None;
                }
                deltas.push(delta);
            }
            None => count_out_of_range += 1,
        }
    }
    if deltas.is_empty() {
        return None;
    }
    let count_in_range = deltas.len();
    let sum: f64 = deltas.iter().sum();
    if !sum.is_finite() {
        return None;
    }
    let mean_delta_db = sum / count_in_range as f64;
    if !mean_delta_db.is_finite() {
        return None;
    }
    let min_delta_db = deltas.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_delta_db = deltas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !min_delta_db.is_finite() || !max_delta_db.is_finite() {
        return None;
    }
    Some(PsnrDeltaSummary {
        mean_delta_db,
        min_delta_db,
        max_delta_db,
        count_in_range,
        count_out_of_range,
    })
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
        let curve = vec![(250.0, 28.0), (500.0, 34.0), (1000.0, 40.0), (2000.0, 46.0)];
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
        let curve_a = vec![(250.0, 28.0), (500.0, 34.0), (1000.0, 40.0), (2000.0, 46.0)];
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
        let curve_a = vec![(250.0, 28.0), (500.0, 34.0), (1000.0, 40.0), (2000.0, 46.0)];
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
        let four = vec![(250.0, 28.0), (500.0, 34.0), (1000.0, 40.0), (2000.0, 46.0)];
        assert!(bdrate(&three, &four).is_nan());
        assert!(bdrate(&four, &three).is_nan());
        assert!(bdrate(&three, &three).is_nan());
    }

    /// Disjoint PSNR ranges → NaN sentinel.
    #[test]
    fn bdrate_handles_disjoint_psnr_ranges() {
        let curve_a = vec![(250.0, 20.0), (300.0, 21.0), (400.0, 22.0), (500.0, 23.0)];
        let curve_b = vec![(250.0, 40.0), (300.0, 41.0), (400.0, 42.0), (500.0, 43.0)];
        assert!(bdrate(&curve_a, &curve_b).is_nan());
    }

    /// Partial overlap should still produce a finite answer.
    /// Curve B has slightly lower bitrate at the overlap → expect negative.
    #[test]
    fn bdrate_handles_partial_overlap() {
        let curve_a = vec![(250.0, 28.0), (500.0, 34.0), (1000.0, 40.0), (1500.0, 43.0)];
        let curve_b = vec![(200.0, 30.0), (400.0, 36.0), (800.0, 42.0), (1600.0, 48.0)];
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
        let curve_a = vec![(250.0, 28.0), (500.0, 34.0), (1000.0, 40.0), (2000.0, 46.0)];
        let curve_bad = vec![(0.0, 28.0), (500.0, 34.0), (1000.0, 40.0), (2000.0, 46.0)];
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
        let curve_on: Vec<(f64, f64)> = curve_off.iter().map(|&(br, ps)| (br * 0.9, ps)).collect();
        let bd = bdrate(&curve_off, &curve_on);
        // exp(ln(0.9)) - 1 ≈ -0.10 → -10 %.
        assert!(
            (bd - (-10.0)).abs() < 1e-6,
            "AQ-on at 0.9× bitrate should give BD-rate ≈ -10 %, got {bd}",
        );
        assert!(
            bd < 0.0,
            "AQ-on more efficient must yield negative BD-rate, got {bd}"
        );
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
        let curve = vec![(250.0, 28.0), (500.0, 34.0), (1000.0, 40.0), (2000.0, 46.0)];
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
        let curve = vec![(250.0, 28.0), (500.0, 34.0), (1000.0, 40.0), (2000.0, 46.0)];
        let mid_br = (250.0_f64.ln() + 500.0_f64.ln()).exp().sqrt();
        let p = psnr_at_bitrate(&curve, mid_br).expect("in range");
        assert!((p - 31.0).abs() < 1e-9, "p={p} at mid_br={mid_br}");
    }

    /// `psnr_at_bitrate` refuses to extrapolate beyond the curve range.
    #[test]
    fn psnr_at_bitrate_out_of_range_returns_none() {
        let curve = vec![(250.0, 28.0), (500.0, 34.0), (1000.0, 40.0), (2000.0, 46.0)];
        assert!(psnr_at_bitrate(&curve, 100.0).is_none());
        assert!(psnr_at_bitrate(&curve, 5000.0).is_none());
        assert!(psnr_at_bitrate(&curve, 0.0).is_none());
        assert!(psnr_at_bitrate(&curve, f64::NAN).is_none());
    }

    /// Curve out of order in input — `psnr_at_bitrate` should still work
    /// (it sorts internally).
    #[test]
    fn psnr_at_bitrate_unsorted_input_handled() {
        let curve = vec![(2000.0, 46.0), (250.0, 28.0), (1000.0, 40.0), (500.0, 34.0)];
        let p = psnr_at_bitrate(&curve, 500.0).expect("in range");
        assert!((p - 34.0).abs() < 1e-9, "p={p}");
    }

    /// Degenerate input (all x values identical) → polyfit returns None,
    /// bdrate returns NaN.
    #[test]
    fn bdrate_returns_nan_on_collinear_input() {
        let curve_a = vec![(500.0, 35.0), (600.0, 35.0), (700.0, 35.0), (800.0, 35.0)];
        let curve_b = vec![(500.0, 35.0), (600.0, 35.0), (700.0, 35.0), (800.0, 35.0)];
        assert!(bdrate(&curve_a, &curve_b).is_nan());
    }

    // ---------------------------------------------------------------
    // psnr_delta_summary — F.0.1 additions.
    // ---------------------------------------------------------------

    /// All rvp9 points fall inside the libvpx curve's bitrate range.
    /// libvpx is uniformly +4 dB above rvp9 (mean = min = max = +4.0;
    /// count_in_range == 4, count_out_of_range == 0).
    #[test]
    fn psnr_delta_summary_all_in_range() {
        let libvpx = vec![(100.0, 30.0), (200.0, 34.0), (400.0, 38.0), (800.0, 42.0)];
        // rvp9 sits 4 dB below libvpx at every overlapping bitrate
        // sample. The shared bitrates make `psnr_at_bitrate` reproduce
        // libvpx's PSNR exactly at each query point.
        let rvp9 = vec![(100.0, 26.0), (200.0, 30.0), (400.0, 34.0), (800.0, 38.0)];
        let s = psnr_delta_summary(&libvpx, &rvp9).expect("in-range");
        assert_eq!(s.count_in_range, 4);
        assert_eq!(s.count_out_of_range, 0);
        assert!(
            (s.mean_delta_db - 4.0).abs() < 1e-9,
            "mean={}",
            s.mean_delta_db
        );
        assert!(
            (s.min_delta_db - 4.0).abs() < 1e-9,
            "min={}",
            s.min_delta_db
        );
        assert!(
            (s.max_delta_db - 4.0).abs() < 1e-9,
            "max={}",
            s.max_delta_db
        );
    }

    /// rvp9 has 4 points, 2 inside libvpx's [200, 800] span and 2
    /// outside (one below, one above). Mean is computed over the
    /// 2 in-range only; count_in_range == 2; count_out_of_range == 2.
    #[test]
    fn psnr_delta_summary_partial_out_of_range() {
        let libvpx = vec![(200.0, 30.0), (400.0, 34.0), (600.0, 36.0), (800.0, 38.0)];
        // Two in-range (400, 600) and two out-of-range (100, 1000).
        // At br=400 libvpx = 34 dB; rvp9 = 30 → Δ = +4.
        // At br=600 libvpx = 36 dB; rvp9 = 31 → Δ = +5.
        let rvp9 = vec![
            (100.0, 20.0),  // below libvpx range — skipped
            (400.0, 30.0),  // in-range, Δ = +4
            (600.0, 31.0),  // in-range, Δ = +5
            (1000.0, 35.0), // above libvpx range — skipped
        ];
        let s = psnr_delta_summary(&libvpx, &rvp9).expect("in-range");
        assert_eq!(s.count_in_range, 2);
        assert_eq!(s.count_out_of_range, 2);
        assert!(
            (s.mean_delta_db - 4.5).abs() < 1e-9,
            "mean={}",
            s.mean_delta_db
        );
        assert!(
            (s.min_delta_db - 4.0).abs() < 1e-9,
            "min={}",
            s.min_delta_db
        );
        assert!(
            (s.max_delta_db - 5.0).abs() < 1e-9,
            "max={}",
            s.max_delta_db
        );
    }

    /// All rvp9 points lie outside libvpx's bitrate span → `None`
    /// (caller should report DEFERRED). count_in_range would be 0,
    /// which the API encodes as the `None` sentinel.
    #[test]
    fn psnr_delta_summary_none_in_range() {
        let libvpx = vec![(200.0, 30.0), (300.0, 32.0), (400.0, 34.0), (500.0, 36.0)];
        let rvp9 = vec![(50.0, 22.0), (100.0, 24.0), (1000.0, 38.0), (2000.0, 42.0)];
        assert!(psnr_delta_summary(&libvpx, &rvp9).is_none());
    }

    /// Sign convention: when rvp9 beats libvpx (rvp9 PSNR higher than
    /// libvpx-interp at the same bitrate) the delta is negative. This
    /// is the eventual Track F success regime once block_yrd +
    /// find_mv_refs + motion-aware partition land.
    #[test]
    fn psnr_delta_summary_libvpx_below_rvp9() {
        let libvpx = vec![(100.0, 26.0), (200.0, 30.0), (400.0, 34.0), (800.0, 38.0)];
        // rvp9 is 2 dB BETTER than libvpx at every shared bitrate.
        let rvp9 = vec![(100.0, 28.0), (200.0, 32.0), (400.0, 36.0), (800.0, 40.0)];
        let s = psnr_delta_summary(&libvpx, &rvp9).expect("in-range");
        assert_eq!(s.count_in_range, 4);
        assert_eq!(s.count_out_of_range, 0);
        assert!(
            s.mean_delta_db < 0.0,
            "expected negative mean delta, got {}",
            s.mean_delta_db
        );
        assert!(
            (s.mean_delta_db - (-2.0)).abs() < 1e-9,
            "mean={}",
            s.mean_delta_db
        );
        assert!(
            (s.min_delta_db - (-2.0)).abs() < 1e-9,
            "min={}",
            s.min_delta_db
        );
        assert!(
            (s.max_delta_db - (-2.0)).abs() < 1e-9,
            "max={}",
            s.max_delta_db
        );
    }

    /// Exactly one rvp9 point in-range: min == max == mean, count == 1.
    /// (Note: the gate caller still requires `count_in_range >= 2` for
    /// PASS — that's a gate-level policy, not a summary-level one.)
    #[test]
    fn psnr_delta_summary_single_in_range() {
        let libvpx = vec![(200.0, 30.0), (300.0, 32.0), (400.0, 34.0), (500.0, 36.0)];
        // Only one rvp9 point (at br=300) falls inside the libvpx span.
        // At br=300 libvpx = 32 dB; rvp9 = 28 → Δ = +4.
        let rvp9 = vec![
            (50.0, 20.0),   // below — skipped
            (300.0, 28.0),  // in-range, Δ = +4
            (1000.0, 36.0), // above — skipped
            (2000.0, 38.0), // above — skipped
        ];
        let s = psnr_delta_summary(&libvpx, &rvp9).expect("in-range");
        assert_eq!(s.count_in_range, 1);
        assert_eq!(s.count_out_of_range, 3);
        assert!(
            (s.mean_delta_db - 4.0).abs() < 1e-9,
            "mean={}",
            s.mean_delta_db
        );
        assert!(
            (s.min_delta_db - s.mean_delta_db).abs() < 1e-12,
            "single point: min must equal mean, got min={}, mean={}",
            s.min_delta_db,
            s.mean_delta_db
        );
        assert!(
            (s.max_delta_db - s.mean_delta_db).abs() < 1e-12,
            "single point: max must equal mean, got max={}, mean={}",
            s.max_delta_db,
            s.mean_delta_db
        );
    }

    /// Invalid samples anywhere in the rvp9 curve invalidate the summary,
    /// even when another point is inside the reference bitrate range.
    #[test]
    fn psnr_delta_summary_rejects_invalid_rvp9_samples() {
        let libvpx = vec![(100.0, 30.0), (200.0, 32.0), (400.0, 36.0)];
        let invalid_psnr_curves = [
            vec![(200.0, f64::NAN), (300.0, 34.0)],
            vec![(300.0, 34.0), (200.0, f64::NAN)],
            vec![(200.0, f64::INFINITY), (300.0, 34.0)],
            vec![(300.0, 34.0), (200.0, f64::INFINITY)],
            vec![(200.0, f64::NEG_INFINITY), (300.0, 34.0)],
            vec![(300.0, 34.0), (200.0, f64::NEG_INFINITY)],
            vec![(50.0, f64::NAN), (200.0, 32.0)],
            vec![(200.0, 32.0), (50.0, f64::NAN)],
            vec![(50.0, f64::INFINITY), (200.0, 32.0)],
            vec![(200.0, 32.0), (50.0, f64::INFINITY)],
            vec![(50.0, f64::NEG_INFINITY), (200.0, 32.0)],
            vec![(200.0, 32.0), (50.0, f64::NEG_INFINITY)],
        ];
        for rvp9 in invalid_psnr_curves {
            let summary = psnr_delta_summary(&libvpx, &rvp9);
            assert!(
                summary.is_none(),
                "expected None for invalid PSNR curve, got {summary:?} from {rvp9:?}"
            );
        }

        // Zero and negative PSNR values are valid finite samples.
        let non_positive_psnr_libvpx = vec![(100.0, 0.0), (200.0, -1.0), (400.0, -3.0)];
        let non_positive_psnr_rvp9 = vec![(100.0, -1.0), (200.0, -2.0)];
        let summary = psnr_delta_summary(&non_positive_psnr_libvpx, &non_positive_psnr_rvp9)
            .expect("finite zero and negative PSNR values are valid");
        assert_eq!(summary.count_in_range, 2);
        assert_eq!(summary.mean_delta_db, 1.0);
    }

    /// Invalid rvp9 bitrates invalidate the complete curve regardless of
    /// where the sample appears relative to valid samples.
    #[test]
    fn psnr_delta_summary_rejects_invalid_rvp9_bitrates() {
        let libvpx = vec![(100.0, 30.0), (200.0, 32.0), (400.0, 36.0)];
        let invalid_bitrates = [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        for bitrate in invalid_bitrates {
            for rvp9 in [
                vec![(100.0, 30.0), (bitrate, 32.0), (300.0, 34.0)],
                vec![(bitrate, 32.0), (100.0, 30.0), (300.0, 34.0)],
            ] {
                let summary = psnr_delta_summary(&libvpx, &rvp9);
                assert!(
                    summary.is_none(),
                    "expected None for invalid bitrate curve, got {summary:?} from {rvp9:?}"
                );
            }
        }
    }

    /// Finite inputs can overflow while calculating deltas or their sum;
    /// those values must not escape as a present summary.
    #[test]
    fn psnr_delta_summary_rejects_non_finite_arithmetic() {
        let max_psnr = f64::MAX;
        let delta_overflow_cases = [
            (
                vec![(100.0, max_psnr), (200.0, max_psnr)],
                vec![(100.0, -max_psnr)],
            ),
            (
                vec![(100.0, -max_psnr), (200.0, -max_psnr)],
                vec![(100.0, max_psnr)],
            ),
        ];
        for (libvpx, rvp9) in delta_overflow_cases {
            let summary = psnr_delta_summary(&libvpx, &rvp9);
            assert!(
                summary.is_none(),
                "expected None for overflowing delta, got {summary:?}"
            );
        }

        // Keep the reference interpolation at zero so only accumulation of
        // finite same-sign deltas can overflow.
        let libvpx = vec![(100.0, 0.0), (200.0, 0.0)];
        let large_delta = max_psnr * 0.75;
        for rvp9_psnr in [-large_delta, large_delta] {
            let rvp9 = vec![(100.0, rvp9_psnr), (200.0, rvp9_psnr)];
            let summary = psnr_delta_summary(&libvpx, &rvp9);
            assert!(
                summary.is_none(),
                "expected None for overflowing delta sum, got {summary:?}"
            );
        }
    }
}
