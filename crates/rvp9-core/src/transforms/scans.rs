//! Coefficient scan-order tables.
//!
//! `ScanOrder` couples three parallel arrays for one (TxSize, TxType)
//! pair:
//!
//! - `scan[k]` is the raster position of the k-th coefficient in scan
//!   order. Used by the encoder when packing coefficients and by the
//!   decoder when laying out received coefficients.
//! - `iscan[p]` is `scan_index(p) + 1` (EOB-position semantics; see
//!   libvpx `vp9_scan.c:514`). The `+1` lets the encoder write an EOB
//!   marker as a sentinel.
//! - `neighbors[2k..2k+1]` is the {top, left} pair of raster positions
//!   used to derive the coefficient-coding context (`get_coef_context`
//!   in libvpx `vp9_scan.h:32-40`).
//!
//! `MAX_NEIGHBORS` (`= 2`, see [`super::MAX_NEIGHBORS`]) is the second
//! index of `neighbors[(n_coeffs + 1) * MAX_NEIGHBORS]`. The extra `+1`
//! slot at the end is the EOB sentinel.
//!
//! Sources (libvpx `vp9/common/vp9_scan.c`):
//!
//! - `default_scan_4x4` at line 15-17.
//! - `vp9_default_iscan_4x4` at line 19-21 (above
//!   `vp9_col_iscan_4x4`).
//! - `default_scan_4x4_neighbors` at lines 195-198.
//!
//! ## Status
//!
//! Stage 3 task 3a landed the 4×4 default scan; task 3h adds the 4×4
//! `col` and `row` scan triples and the per-`TxType` dispatch helper
//! [`scan_order_4x4`]. The larger sizes (8×8 / 16×16 / 32×32 + their
//! type variants) and a top-level `scan_for(TxSize, TxType)` arrive in
//! follow-up sub-tasks.

/// Triple of `(scan, iscan, neighbors)` slices that fully describes the
/// coefficient ordering and coding context for one (TxSize, TxType)
/// pair. All slices are `&'static` references to static arrays in this
/// module. `ScanOrder` is `Copy`.
///
/// Mirrors libvpx `struct scan_order` (`vp9_scan.h:26-30`).
#[derive(Copy, Clone, Debug)]
pub struct ScanOrder {
    /// `scan[k]` = raster index of the k-th coefficient. Length is
    /// `TxSize::n_coeffs()`.
    pub scan: &'static [i16],
    /// `iscan[p]` = scan index of position `p` plus 1. Length is
    /// `TxSize::n_coeffs()`.
    pub iscan: &'static [i16],
    /// {top, left} neighbour raster positions per coefficient, then a
    /// trailing EOB sentinel pair. Length is
    /// `(TxSize::n_coeffs() + 1) * MAX_NEIGHBORS`.
    pub neighbors: &'static [i16],
}

/// 4×4 default scan (zig-zag-ish from top-left). 16 entries verbatim
/// from libvpx `vp9/common/vp9_scan.c:15-17` line 16.
#[rustfmt::skip]
pub static DEFAULT_SCAN_4X4: [i16; 16] = [
    0, 4, 1, 5, 8, 2, 12, 9, 3, 6, 13, 10, 7, 14, 11, 15,
];

/// 4×4 default inverse scan. `iscan[p] = (scan_position_of_raster_p) + 1`.
/// 16 entries verbatim from libvpx `vp9_scan.c:19-21` (the
/// `vp9_default_iscan_4x4` block).
#[rustfmt::skip]
pub static DEFAULT_ISCAN_4X4: [i16; 16] = [
    1, 3, 6, 9, 2, 4, 10, 13, 5, 8, 12, 15, 7, 11, 14, 16,
];

/// 4×4 default neighbours table. `(16 + 1) * MAX_NEIGHBORS = 34`
/// entries. The trailing `0, 0` pair is the EOB sentinel.
/// Verbatim from libvpx `vp9_scan.c:195-198`.
#[rustfmt::skip]
pub static DEFAULT_SCAN_4X4_NEIGHBORS: [i16; 34] = [
    0, 0, 0, 0, 0,  0, 1, 4, 4, 4,  1,  1, 8,  8,  5,  8, 2,
    2, 2, 5, 9, 12, 6, 9, 3, 6, 10, 13, 7, 10, 11, 14, 0, 0,
];

/// The 4×4 default `ScanOrder` triple. Wired to the three static arrays
/// above. Mirrors the `vp9_default_scan_orders[Tx4x4]` slot in libvpx
/// `vp9_scan.c:691`.
#[inline]
#[must_use]
pub const fn default_scan_4x4() -> ScanOrder {
    ScanOrder {
        scan: &DEFAULT_SCAN_4X4,
        iscan: &DEFAULT_ISCAN_4X4,
        neighbors: &DEFAULT_SCAN_4X4_NEIGHBORS,
    }
}

/// 4×4 column scan (ADST_DCT — vertical first). 16 entries from
/// libvpx `vp9_scan.c:19-21`.
#[rustfmt::skip]
pub static COL_SCAN_4X4: [i16; 16] = [
    0, 4, 8, 1, 12, 5, 9, 2, 13, 6, 10, 3, 7, 14, 11, 15,
];

/// 4×4 column inverse scan. 16 entries from libvpx
/// `vp9_scan.c:519-521` (`vp9_col_iscan_4x4`).
#[rustfmt::skip]
pub static COL_ISCAN_4X4: [i16; 16] = [
    1, 4, 8, 12, 2, 6, 10, 13, 3, 7, 11, 15, 5, 9, 14, 16,
];

/// 4×4 column scan neighbours. 34 entries (`(16 + 1) * 2`) from libvpx
/// `vp9_scan.c:201-205` (`col_scan_4x4_neighbors`).
#[rustfmt::skip]
pub static COL_SCAN_4X4_NEIGHBORS: [i16; 34] = [
    0, 0, 0, 0, 4, 4, 0, 0, 8, 8,  1,  1, 5, 5,  1,  1, 9,
    9, 2, 2, 6, 6, 2, 2, 3, 3, 10, 10, 7, 7, 11, 11, 0, 0,
];

/// 4×4 row scan (DCT_ADST — horizontal first). 16 entries from libvpx
/// `vp9_scan.c:23-25`.
#[rustfmt::skip]
pub static ROW_SCAN_4X4: [i16; 16] = [
    0, 1, 4, 2, 5, 3, 6, 8, 9, 7, 12, 10, 13, 11, 14, 15,
];

/// 4×4 row inverse scan. 16 entries from libvpx `vp9_scan.c:523-525`
/// (`vp9_row_iscan_4x4`).
#[rustfmt::skip]
pub static ROW_ISCAN_4X4: [i16; 16] = [
    1, 2, 4, 6, 3, 5, 7, 10, 8, 9, 12, 14, 11, 13, 15, 16,
];

/// 4×4 row scan neighbours. 34 entries from libvpx `vp9_scan.c:207-211`
/// (`row_scan_4x4_neighbors`).
#[rustfmt::skip]
pub static ROW_SCAN_4X4_NEIGHBORS: [i16; 34] = [
    0, 0, 0, 0, 0, 0, 1, 1,  4,  4,  2,  2,  5,  5,  4,  4, 8,
    8, 6, 6, 8, 8, 9, 9, 12, 12, 10, 10, 13, 13, 14, 14, 0, 0,
];

/// 4×4 column `ScanOrder` (used for `TxType::AdstDct`).
#[inline]
#[must_use]
pub const fn col_scan_4x4() -> ScanOrder {
    ScanOrder {
        scan: &COL_SCAN_4X4,
        iscan: &COL_ISCAN_4X4,
        neighbors: &COL_SCAN_4X4_NEIGHBORS,
    }
}

/// 4×4 row `ScanOrder` (used for `TxType::DctAdst`).
#[inline]
#[must_use]
pub const fn row_scan_4x4() -> ScanOrder {
    ScanOrder {
        scan: &ROW_SCAN_4X4,
        iscan: &ROW_ISCAN_4X4,
        neighbors: &ROW_SCAN_4X4_NEIGHBORS,
    }
}

/// Return the 4×4 `ScanOrder` for the given `TxType`.
///
/// Mirrors `vp9_scan_orders[Tx4x4][tx_type]` in libvpx
/// `vp9_scan.c:700-703`:
///
/// - `DCT_DCT`   → default scan
/// - `ADST_DCT`  → row scan (rows are iadst → horizontal frequencies
///   skew the per-position priority toward later columns; libvpx names
///   this `row_scan` from the encoder's perspective)
/// - `DCT_ADST`  → column scan
/// - `ADST_ADST` → default scan (libvpx falls back here; the dedicated
///   ADST_ADST scan is the default scan)
#[inline]
#[must_use]
pub const fn scan_order_4x4(tx_type: super::types::TxType) -> ScanOrder {
    use super::types::TxType;
    match tx_type {
        TxType::DctDct => default_scan_4x4(),
        TxType::AdstDct => row_scan_4x4(),
        TxType::DctAdst => col_scan_4x4(),
        TxType::AdstAdst => default_scan_4x4(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::types::{MAX_NEIGHBORS, TxSize};

    #[test]
    fn scan_4x4_length_matches_tx_size() {
        assert_eq!(DEFAULT_SCAN_4X4.len(), TxSize::Tx4x4.n_coeffs());
        assert_eq!(DEFAULT_ISCAN_4X4.len(), TxSize::Tx4x4.n_coeffs());
    }

    #[test]
    fn neighbors_4x4_length_matches_formula() {
        assert_eq!(
            DEFAULT_SCAN_4X4_NEIGHBORS.len(),
            (TxSize::Tx4x4.n_coeffs() + 1) * MAX_NEIGHBORS
        );
    }

    /// `scan[k]` is a permutation of `0..n_coeffs`. Every raster
    /// position appears exactly once.
    #[test]
    fn scan_is_a_permutation() {
        let n = TxSize::Tx4x4.n_coeffs();
        let mut seen = vec![false; n];
        for &raster in DEFAULT_SCAN_4X4.iter() {
            let i = raster as usize;
            assert!(i < n, "scan value {raster} out of range");
            assert!(!seen[i], "scan value {raster} duplicated");
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s), "not all raster positions hit");
    }

    /// `iscan[scan[k]] = k + 1` — the libvpx EOB-position convention.
    #[test]
    fn iscan_is_inverse_of_scan_plus_one() {
        for (k, &raster) in DEFAULT_SCAN_4X4.iter().enumerate() {
            assert_eq!(
                DEFAULT_ISCAN_4X4[raster as usize],
                (k as i16) + 1,
                "iscan[{raster}] should be {} (scan-index {k} + 1)",
                k + 1
            );
        }
    }

    /// The trailing two entries of `neighbors` are the `(0, 0)` EOB
    /// sentinel.
    #[test]
    fn neighbors_trailing_sentinel() {
        let n = DEFAULT_SCAN_4X4_NEIGHBORS.len();
        assert_eq!(DEFAULT_SCAN_4X4_NEIGHBORS[n - 2], 0);
        assert_eq!(DEFAULT_SCAN_4X4_NEIGHBORS[n - 1], 0);
    }

    /// `default_scan_4x4()` wires up the three statics correctly. We
    /// compare slice pointers via `as_ptr` since `ScanOrder` is `Copy`.
    #[test]
    fn default_scan_4x4_wires_to_statics() {
        let s = default_scan_4x4();
        assert_eq!(s.scan.as_ptr(), DEFAULT_SCAN_4X4.as_ptr());
        assert_eq!(s.iscan.as_ptr(), DEFAULT_ISCAN_4X4.as_ptr());
        assert_eq!(s.neighbors.as_ptr(), DEFAULT_SCAN_4X4_NEIGHBORS.as_ptr());
    }

    // ---- 4x4 col + row scans ----

    fn assert_scan_is_permutation(scan: &[i16], n: usize) {
        let mut seen = vec![false; n];
        for &raster in scan.iter() {
            let i = raster as usize;
            assert!(i < n, "scan value {raster} out of range");
            assert!(!seen[i], "scan value {raster} duplicated");
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s), "not all raster positions hit");
    }

    fn assert_iscan_is_inverse_of_scan_plus_one(scan: &[i16], iscan: &[i16]) {
        for (k, &raster) in scan.iter().enumerate() {
            assert_eq!(
                iscan[raster as usize],
                (k as i16) + 1,
                "iscan[{raster}] should be {} (scan-index {k} + 1)",
                k + 1
            );
        }
    }

    #[test]
    fn col_scan_4x4_is_a_permutation() {
        assert_scan_is_permutation(&COL_SCAN_4X4, TxSize::Tx4x4.n_coeffs());
    }

    #[test]
    fn col_iscan_4x4_is_inverse_of_col_scan() {
        assert_iscan_is_inverse_of_scan_plus_one(&COL_SCAN_4X4, &COL_ISCAN_4X4);
    }

    #[test]
    fn col_scan_4x4_neighbors_length() {
        assert_eq!(
            COL_SCAN_4X4_NEIGHBORS.len(),
            (TxSize::Tx4x4.n_coeffs() + 1) * MAX_NEIGHBORS
        );
    }

    #[test]
    fn row_scan_4x4_is_a_permutation() {
        assert_scan_is_permutation(&ROW_SCAN_4X4, TxSize::Tx4x4.n_coeffs());
    }

    #[test]
    fn row_iscan_4x4_is_inverse_of_row_scan() {
        assert_iscan_is_inverse_of_scan_plus_one(&ROW_SCAN_4X4, &ROW_ISCAN_4X4);
    }

    #[test]
    fn row_scan_4x4_neighbors_length() {
        assert_eq!(
            ROW_SCAN_4X4_NEIGHBORS.len(),
            (TxSize::Tx4x4.n_coeffs() + 1) * MAX_NEIGHBORS
        );
    }

    /// `scan_order_4x4` dispatch matches libvpx's
    /// `vp9_scan_orders[Tx4x4][tx_type]` table at lines 700-703.
    #[test]
    fn scan_order_4x4_dispatch() {
        use crate::transforms::types::TxType;

        let dct = scan_order_4x4(TxType::DctDct);
        assert_eq!(dct.scan.as_ptr(), DEFAULT_SCAN_4X4.as_ptr());

        let adst_dct = scan_order_4x4(TxType::AdstDct);
        assert_eq!(adst_dct.scan.as_ptr(), ROW_SCAN_4X4.as_ptr());

        let dct_adst = scan_order_4x4(TxType::DctAdst);
        assert_eq!(dct_adst.scan.as_ptr(), COL_SCAN_4X4.as_ptr());

        // ADST_ADST falls back to default (libvpx :703).
        let adst_adst = scan_order_4x4(TxType::AdstAdst);
        assert_eq!(adst_adst.scan.as_ptr(), DEFAULT_SCAN_4X4.as_ptr());
    }
}
