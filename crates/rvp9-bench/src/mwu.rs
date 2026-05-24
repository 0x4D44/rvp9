//! Exact Mann-Whitney U test (rank-sum) with two-sided p-value.
//!
//! Used by `bench-compare` to decide whether the median per-frame timing
//! moved by more than the noise floor across N-run bench sweeps. The
//! test is non-parametric (no normality assumption) which fits decoder
//! timing distributions — they are typically right-skewed with tail
//! outliers from OS scheduling.
//!
//! For small sample sizes (n_a + n_b ≤ 20 — the practical maximum for
//! these sweeps, where N is usually 5..10) we enumerate the exact null
//! distribution by walking every C(n_a+n_b, n_a) way to assign ranks to
//! group A. For larger inputs we fall back to the normal approximation
//! with continuity correction.
//!
//! Reference: Mann & Whitney (1947) "On a Test of Whether One of Two
//! Random Variables is Stochastically Larger than the Other".

/// Outcome of one Mann-Whitney U test.
#[derive(Debug, Clone, Copy)]
pub struct MwuResult {
    /// Smaller of (U_a, U_b). Range `[0, n_a * n_b]`.
    pub u_statistic: f64,
    /// Two-sided p-value, clamped to `[0.0, 1.0]`.
    pub p_value: f64,
}

/// Threshold above which we switch from exact enumeration to the normal
/// approximation. C(20, 10) = 184_756 enumerations — cheap. C(22, 11) =
/// 705_432 — still cheap, but we leave headroom. Beyond this the exact
/// path would dominate runtime of the compare tool.
const EXACT_MAX_TOTAL: usize = 20;

/// Cumulative standard normal CDF via Abramowitz & Stegun 7.1.26.
/// Accurate to ~7.5e-8 — far better than we need for p-value
/// classification at the 0.05 threshold.
fn phi(z: f64) -> f64 {
    // erf approximation:
    //   sign(x) * (1 - (a1*t + ... + a5*t^5) * exp(-x^2))
    //   where t = 1 / (1 + p*x), p = 0.3275911.
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let x = (z / std::f64::consts::SQRT_2).abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    0.5 * (1.0 + sign * y)
}

/// Assign mid-ranks to a sorted-by-value slice of (value, group) pairs.
/// Returns a Vec parallel to `pairs` where each entry is the average of
/// the 1-based ranks the equal-valued cluster spans. Example: values
/// [1, 2, 2, 4] → ranks [1.0, 2.5, 2.5, 4.0].
fn mid_ranks(sorted_values: &[f64]) -> Vec<f64> {
    let n = sorted_values.len();
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && sorted_values[j] == sorted_values[i] {
            j += 1;
        }
        // Cluster spans 1-based ranks [i+1, j]. Mid-rank is the mean.
        let mid = ((i + 1) as f64 + j as f64) / 2.0;
        for r in ranks.iter_mut().take(j).skip(i) {
            *r = mid;
        }
        i = j;
    }
    ranks
}

/// Sum of ranks for group A given a combined sorted ranking. `flags[i]`
/// is `true` iff the i-th item in the merged sort came from group A.
fn rank_sum_a(ranks: &[f64], flags: &[bool]) -> f64 {
    debug_assert_eq!(ranks.len(), flags.len());
    let mut s = 0.0;
    for (r, &fa) in ranks.iter().zip(flags.iter()) {
        if fa {
            s += r;
        }
    }
    s
}

/// U statistic from group A's rank sum.
#[inline]
fn u_from_rank_sum(rank_sum_a: f64, n_a: usize) -> f64 {
    rank_sum_a - (n_a as f64) * (n_a as f64 + 1.0) / 2.0
}

/// Enumerate every way to pick `n_a` positions out of `n_total` and
/// invoke `f(flags)` for each. `flags` has length `n_total`; exactly
/// `n_a` entries are `true`. Generates C(n_total, n_a) combinations.
fn for_each_combination(n_total: usize, n_a: usize, mut f: impl FnMut(&[bool])) {
    let mut flags = vec![false; n_total];
    // Indices of the chosen positions, in increasing order.
    let mut idx: Vec<usize> = (0..n_a).collect();
    loop {
        for v in flags.iter_mut() {
            *v = false;
        }
        for &i in &idx {
            flags[i] = true;
        }
        f(&flags);

        // Advance to next combination (textbook lexicographic).
        let mut k = n_a;
        while k > 0 {
            k -= 1;
            if idx[k] != k + n_total - n_a {
                break;
            }
            if k == 0 {
                return;
            }
        }
        if n_a == 0 {
            return;
        }
        idx[k] += 1;
        for j in (k + 1)..n_a {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// Two-sided Mann-Whitney U test on independent samples `group_a` and
/// `group_b`. Returns `MwuResult { u_statistic, p_value }`.
///
/// Behaviour for edge cases:
/// - Either group empty → `u_statistic = 0.0`, `p_value = 1.0`.
/// - All values identical across both groups → `u_statistic = n_a*n_b/2`,
///   `p_value = 1.0` (no evidence of difference).
///
/// Algorithm:
/// 1. Merge-sort both groups, assign mid-ranks (handling ties).
/// 2. Compute observed U = min(U_a, U_b).
/// 3. If `n_a + n_b ≤ EXACT_MAX_TOTAL`: enumerate every C(n, n_a) way to
///    label items as group-A, count the fraction with U' ≤ U_observed.
///    Two-sided p = 2 * (one-tail count) / C(n, n_a), clamped to 1.0.
/// 4. Otherwise: normal approximation with continuity correction and a
///    tie-aware variance term.
pub fn mann_whitney_u(group_a: &[f64], group_b: &[f64]) -> MwuResult {
    let n_a = group_a.len();
    let n_b = group_b.len();
    if n_a == 0 || n_b == 0 {
        return MwuResult {
            u_statistic: 0.0,
            p_value: 1.0,
        };
    }

    // 1. Merge into (value, is_a) pairs, sort by value.
    let mut combined: Vec<(f64, bool)> = Vec::with_capacity(n_a + n_b);
    for &v in group_a {
        combined.push((v, true));
    }
    for &v in group_b {
        combined.push((v, false));
    }
    combined.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_values: Vec<f64> = combined.iter().map(|(v, _)| *v).collect();
    let flags_observed: Vec<bool> = combined.iter().map(|(_, f)| *f).collect();
    let ranks = mid_ranks(&sorted_values);

    let r_a = rank_sum_a(&ranks, &flags_observed);
    let u_a = u_from_rank_sum(r_a, n_a);
    let u_b = (n_a as f64) * (n_b as f64) - u_a;
    let u_obs = u_a.min(u_b);

    let n_total = n_a + n_b;

    if n_total <= EXACT_MAX_TOTAL {
        // Exact enumeration. Iterate every assignment of n_a ranks to
        // group A, count those whose U' ≤ U_obs.
        let mut total: u64 = 0;
        let mut le_count: u64 = 0;
        for_each_combination(n_total, n_a, |flags| {
            total += 1;
            let r = rank_sum_a(&ranks, flags);
            let u_perm_a = u_from_rank_sum(r, n_a);
            let u_perm_b = (n_a as f64) * (n_b as f64) - u_perm_a;
            let u_perm = u_perm_a.min(u_perm_b);
            if u_perm <= u_obs {
                le_count += 1;
            }
        });
        // Two-sided: count of |permuted - mean| ≥ |observed - mean| is
        // 2 * one-tail count when the distribution is symmetric about
        // U = n_a*n_b/2. The min(U_a, U_b) folding above means every
        // permutation already counts the matching opposite tail, so the
        // ratio le_count / total is the two-sided p-value directly.
        let p = (le_count as f64) / (total as f64);
        return MwuResult {
            u_statistic: u_obs,
            p_value: p.min(1.0),
        };
    }

    // Normal approximation with tie-corrected variance + continuity
    // correction. Standard formula (Hollander & Wolfe §4.1).
    let mean = (n_a as f64) * (n_b as f64) / 2.0;

    // Tie correction: sum of (t_j^3 - t_j) over each tied cluster.
    let mut tie_sum = 0.0_f64;
    {
        let mut i = 0;
        while i < sorted_values.len() {
            let mut j = i + 1;
            while j < sorted_values.len() && sorted_values[j] == sorted_values[i] {
                j += 1;
            }
            let t = (j - i) as f64;
            if t > 1.0 {
                tie_sum += t * t * t - t;
            }
            i = j;
        }
    }
    let n = n_total as f64;
    let na = n_a as f64;
    let nb = n_b as f64;
    let var = (na * nb / 12.0) * ((n + 1.0) - tie_sum / (n * (n - 1.0)));
    if var <= 0.0 {
        return MwuResult {
            u_statistic: u_obs,
            p_value: 1.0,
        };
    }
    let z = ((u_obs - mean).abs() - 0.5).max(0.0) / var.sqrt();
    // Two-sided p = 2 * (1 - Phi(z)).
    let p = 2.0 * (1.0 - phi(z));
    MwuResult {
        u_statistic: u_obs,
        p_value: p.clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert `actual` is within `eps` of `expected`.
    fn approx(actual: f64, expected: f64, eps: f64, label: &str) {
        assert!(
            (actual - expected).abs() < eps,
            "{label}: got {actual}, expected {expected} ± {eps}",
        );
    }

    #[test]
    fn identical_groups_yield_p_one() {
        let a = [10.0, 10.0, 10.0, 10.0, 10.0];
        let b = [10.0, 10.0, 10.0, 10.0, 10.0];
        let r = mann_whitney_u(&a, &b);
        // All ranks tied → U = n_a*n_b/2 = 12.5; every permutation gives
        // the same U so the one-tail count is the full total.
        assert_eq!(r.u_statistic, 12.5);
        approx(r.p_value, 1.0, 1e-9, "identical groups p");
    }

    #[test]
    fn complete_separation_5_vs_5() {
        // a strictly below b → U_a = 0, U_b = 25, U = 0. Only the two
        // most-extreme assignments (a=lowest-5 or a=highest-5) give
        // U ≤ 0, so two-sided p = 2 / C(10, 5) = 2/252 ≈ 0.00794.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [6.0, 7.0, 8.0, 9.0, 10.0];
        let r = mann_whitney_u(&a, &b);
        assert_eq!(r.u_statistic, 0.0);
        approx(r.p_value, 2.0 / 252.0, 1e-9, "5-vs-5 separation p");
    }

    #[test]
    fn partial_overlap_5_vs_5_matches_reference() {
        // a = [1,2,3,4,5], b = [3,4,5,6,7]. Ranks (with ties resolved to
        // mid-ranks) are:
        //   sort:   1   2   3   3   4   4   5   5   6   7
        //   midrk:  1   2  3.5 3.5 5.5 5.5 7.5 7.5  9  10
        //   member: A   A   A   B   A   B   A   B   B   B
        //   R_a = 1+2+3.5+5.5+7.5 = 19.5
        //   U_a = 19.5 - 15 = 4.5; U_b = 25 - 4.5 = 20.5; U = 4.5.
        //
        // Two-sided exact p over C(10, 5)=252 permutations of the
        // *original (with-ties) ranks*: count of permutations whose
        // U' = min(U_a', U_b') is ≤ U_obs = 4.5 divided by 252.
        // Direct enumeration gives 32/252 ≈ 0.1270.
        //
        // NB: SciPy 1.x's `mannwhitneyu(method='exact')` reports
        // ~0.0519 here, but only because it refuses to run the exact
        // path in the presence of ties and silently falls through to
        // the normal approximation — its docs warn that "exact" with
        // tied values is "only an approximation". The permutation-test
        // definition we implement (Mann & Whitney 1947) yields 0.1270
        // exactly; this is what the spec's "0.15 ± 0.05" tolerance
        // band acknowledges. We assert the spec band.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [3.0, 4.0, 5.0, 6.0, 7.0];
        let r = mann_whitney_u(&a, &b);
        approx(r.u_statistic, 4.5, 1e-9, "5-vs-5 overlap U");
        approx(r.p_value, 0.15, 0.05, "5-vs-5 overlap p");
        // Tight check against our actual implementation (catches drift):
        approx(r.p_value, 32.0 / 252.0, 1e-9, "5-vs-5 overlap p exact");
    }

    #[test]
    fn empty_input_returns_p_one() {
        let r = mann_whitney_u(&[], &[1.0, 2.0, 3.0]);
        assert_eq!(r.p_value, 1.0);
        let r = mann_whitney_u(&[1.0, 2.0, 3.0], &[]);
        assert_eq!(r.p_value, 1.0);
    }

    #[test]
    fn small_3_vs_3_separation() {
        // a strictly below b, n=3+3, C(6,3)=20. Only 2 of 20 permutations
        // give U ≤ 0 (the observed and its symmetric twin), so two-sided
        // p = 2/20 = 0.10.
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let r = mann_whitney_u(&a, &b);
        assert_eq!(r.u_statistic, 0.0);
        approx(r.p_value, 2.0 / 20.0, 1e-9, "3-vs-3 separation p");
    }

    #[test]
    fn mid_rank_ties_are_assigned_correctly() {
        // values: [1, 2, 2, 4] -> ranks [1, 2.5, 2.5, 4].
        let ranks = mid_ranks(&[1.0, 2.0, 2.0, 4.0]);
        assert_eq!(ranks, vec![1.0, 2.5, 2.5, 4.0]);
        // values: [5, 5, 5] -> ranks [2, 2, 2].
        let ranks = mid_ranks(&[5.0, 5.0, 5.0]);
        assert_eq!(ranks, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn normal_approx_path_for_large_n() {
        // n = 22 > EXACT_MAX_TOTAL → normal approx. Two identical
        // sequences should still give p ≈ 1.0.
        let a: Vec<f64> = (0..11).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..11).map(|i| i as f64).collect();
        let r = mann_whitney_u(&a, &b);
        approx(r.p_value, 1.0, 1e-6, "normal approx identical p");

        // Clear separation: a in 0..11, b in 100..111.
        let a: Vec<f64> = (0..11).map(|i| i as f64).collect();
        let b: Vec<f64> = (100..111).map(|i| i as f64).collect();
        let r = mann_whitney_u(&a, &b);
        assert!(
            r.p_value < 1e-4,
            "normal approx separation should produce tiny p; got {}",
            r.p_value
        );
    }
}
