//! Diamond search-site offset tables.
//!
//! Mirrors `vp9_init_dsmotion_compensation` (`vp9_mcomp.c:98-114`) —
//! the 4-site (axis-aligned) diamond used by the realtime full-pixel
//! diamond search in [`super::diamond_search_sad`].
//!
//! The table has `4 * MAX_MVSEARCH_STEPS = 44` entries. Each step's
//! 4 sites are `{(-len, 0), (len, 0), (0, -len), (0, len)}` with
//! `len` starting at `MAX_FIRST_STEP = 1024` and halving each step
//! until `len == 1`.
//!
//! The byte-offset table `ss_os[i] = ss_mv[i].row * stride + ss_mv[i].col`
//! is built at the same time and depends on the reference-buffer
//! stride. It must be recomputed if the stride changes (frame
//! resize).

use super::mv_limits::{MAX_FIRST_STEP, MAX_MVSEARCH_STEPS};

/// Total number of search sites — `4 * MAX_MVSEARCH_STEPS`.
/// libvpx allocates `8 * MAX_MVSEARCH_STEPS` to share the buffer with
/// the 8-site `vp9_init3smotion_compensation` config; we size the
/// table to exactly what the 4-site init produces.
pub const SS_COUNT: usize = 4 * MAX_MVSEARCH_STEPS as usize;

/// Search-site configuration for diamond search.
///
/// Mirrors libvpx `search_site_config` (`vp9_mcomp.h:36-42`) for the
/// 4-site diamond pattern only.
#[derive(Clone, Debug)]
pub struct SearchSiteConfig {
    /// MV offsets, grouped by step: `ss_mv[step * searches_per_step
    /// .. (step + 1) * searches_per_step]` are the 4 sites for step
    /// `step`. Stored as `(row, col)` pairs in full-pel units.
    pub ss_mv: Vec<(i16, i16)>,
    /// Pre-baked byte offsets: `ss_os[i] = ss_mv[i].row * stride +
    /// ss_mv[i].col`. Must be recomputed on stride change.
    pub ss_os: Vec<isize>,
    /// Number of sites tested per step (= 4 for the diamond config).
    pub searches_per_step: u32,
    /// Number of step levels (= `MAX_MVSEARCH_STEPS = 11`).
    pub total_steps: u32,
}

/// `vp9_init_dsmotion_compensation` (`vp9_mcomp.c:98-114`).
///
/// Build the 4-site diamond table for a given reference-buffer stride.
///
/// Step 0 has `len = MAX_FIRST_STEP = 1024`; each subsequent step
/// halves `len` until `len == 1`. Per step the 4 sites are emitted in
/// the order `(-len, 0), (len, 0), (0, -len), (0, len)`.
///
/// `stride` is the reference plane's byte-stride; it determines the
/// `ss_os` table values and must match what the search loop uses for
/// `ref_plane.stride`.
#[must_use]
pub fn init_dsmotion_compensation(stride: i32) -> SearchSiteConfig {
    let stride = stride as isize;
    let mut ss_mv: Vec<(i16, i16)> = Vec::with_capacity(SS_COUNT);
    let mut ss_os: Vec<isize> = Vec::with_capacity(SS_COUNT);

    let mut len = MAX_FIRST_STEP;
    while len > 0 {
        // Order matters: the `diamond_search_sad` all_in check at
        // `vp9_mcomp.c:2104-2107` peeks at indices [0..4] within each
        // step expecting (row_neg, row_pos, col_neg, col_pos).
        let sites = [
            (-len as i16, 0i16),
            (len as i16, 0),
            (0, -len as i16),
            (0, len as i16),
        ];
        for (r, c) in sites {
            ss_mv.push((r, c));
            ss_os.push(isize::from(r) * stride + isize::from(c));
        }
        len /= 2;
    }

    debug_assert_eq!(ss_mv.len(), SS_COUNT);
    let searches_per_step = 4u32;
    let total_steps = (ss_mv.len() as u32) / searches_per_step;

    SearchSiteConfig {
        ss_mv,
        ss_os,
        searches_per_step,
        total_steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ss_count_matches_4_per_step_times_11() {
        assert_eq!(SS_COUNT, 4 * 11);
    }

    #[test]
    fn init_dsmotion_compensation_emits_4_sites_per_step() {
        let cfg = init_dsmotion_compensation(64);
        assert_eq!(cfg.searches_per_step, 4);
        assert_eq!(cfg.total_steps, 11);
        assert_eq!(cfg.ss_mv.len(), 44);
        assert_eq!(cfg.ss_os.len(), 44);
    }

    #[test]
    fn step_0_has_max_first_step_diamond() {
        let cfg = init_dsmotion_compensation(64);
        // Step 0 sites: (-1024, 0), (1024, 0), (0, -1024), (0, 1024).
        assert_eq!(cfg.ss_mv[0], (-1024, 0));
        assert_eq!(cfg.ss_mv[1], (1024, 0));
        assert_eq!(cfg.ss_mv[2], (0, -1024));
        assert_eq!(cfg.ss_mv[3], (0, 1024));
    }

    #[test]
    fn last_step_has_unit_diamond() {
        let cfg = init_dsmotion_compensation(64);
        // The last 4 sites correspond to len=1 (after 10 halvings from 1024).
        let last = cfg.ss_mv.len() - 4;
        assert_eq!(cfg.ss_mv[last], (-1, 0));
        assert_eq!(cfg.ss_mv[last + 1], (1, 0));
        assert_eq!(cfg.ss_mv[last + 2], (0, -1));
        assert_eq!(cfg.ss_mv[last + 3], (0, 1));
    }

    #[test]
    fn ss_os_matches_row_times_stride_plus_col() {
        let stride = 80i32;
        let cfg = init_dsmotion_compensation(stride);
        for (i, &(r, c)) in cfg.ss_mv.iter().enumerate() {
            let expected = (r as isize) * (stride as isize) + (c as isize);
            assert_eq!(
                cfg.ss_os[i], expected,
                "ss_os mismatch at i={i} (r={r}, c={c})"
            );
        }
    }
}
