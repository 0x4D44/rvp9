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

// ============================================================================
// 8×8 scan family
// ============================================================================

/// 8×8 default scan. 64 entries verbatim from libvpx
/// `vp9/common/vp9_scan.c:27-32`.
#[rustfmt::skip]
pub static DEFAULT_SCAN_8X8: [i16; 64] = [
    0,  8,  1,  16, 9,  2,  17, 24, 10, 3,  18, 25, 32, 11, 4,  26,
    33, 19, 40, 12, 34, 27, 5,  41, 20, 48, 13, 35, 42, 28, 21, 6,
    49, 56, 36, 43, 29, 7,  14, 50, 57, 44, 22, 37, 15, 51, 58, 30,
    45, 23, 52, 59, 38, 31, 60, 53, 46, 39, 61, 54, 47, 62, 55, 63,
];

/// 8×8 column scan. 64 entries verbatim from libvpx `vp9_scan.c:34-39`.
#[rustfmt::skip]
pub static COL_SCAN_8X8: [i16; 64] = [
    0,  8,  16, 1,  24, 9,  32, 17, 2,  40, 25, 10, 33, 18, 48, 3,
    26, 41, 11, 56, 19, 34, 4,  49, 27, 42, 12, 35, 20, 57, 50, 28,
    5,  43, 13, 36, 58, 51, 21, 44, 6,  29, 59, 37, 14, 52, 22, 7,
    45, 60, 30, 15, 38, 53, 23, 46, 31, 61, 39, 54, 47, 62, 55, 63,
];

/// 8×8 row scan. 64 entries verbatim from libvpx `vp9_scan.c:41-46`.
#[rustfmt::skip]
pub static ROW_SCAN_8X8: [i16; 64] = [
    0,  1,  2,  8,  9,  3,  16, 10, 4,  17, 11, 24, 5,  18, 25, 12,
    19, 26, 32, 6,  13, 20, 33, 27, 7,  34, 40, 21, 28, 41, 14, 35,
    48, 42, 29, 36, 49, 22, 43, 15, 56, 37, 50, 44, 30, 57, 23, 51,
    58, 45, 38, 52, 31, 59, 53, 46, 60, 39, 61, 47, 54, 55, 62, 63,
];

/// 8×8 default inverse scan. 64 entries verbatim from libvpx
/// `vp9_scan.c:541-546` (`vp9_default_iscan_8x8`).
#[rustfmt::skip]
pub static DEFAULT_ISCAN_8X8: [i16; 64] = [
    1,  3,  6,  10, 15, 23, 32, 38, 2,  5,  9,  14, 20, 27, 39, 45,
    4,  7,  11, 18, 25, 31, 43, 50, 8,  12, 16, 22, 30, 37, 48, 54,
    13, 17, 21, 28, 35, 44, 53, 58, 19, 24, 29, 36, 42, 49, 57, 61,
    26, 33, 40, 46, 51, 56, 60, 63, 34, 41, 47, 52, 55, 59, 62, 64,
];

/// 8×8 column inverse scan. 64 entries verbatim from libvpx
/// `vp9_scan.c:527-532` (`vp9_col_iscan_8x8`).
#[rustfmt::skip]
pub static COL_ISCAN_8X8: [i16; 64] = [
    1,  4,  9,  16, 23, 33, 41, 48, 2,  6,  12, 19, 27, 35, 45, 52,
    3,  8,  14, 21, 29, 39, 47, 55, 5,  11, 17, 25, 32, 42, 51, 57,
    7,  13, 22, 28, 36, 44, 53, 59, 10, 18, 26, 34, 40, 49, 56, 61,
    15, 24, 31, 38, 46, 54, 60, 63, 20, 30, 37, 43, 50, 58, 62, 64,
];

/// 8×8 row inverse scan. 64 entries verbatim from libvpx
/// `vp9_scan.c:534-539` (`vp9_row_iscan_8x8`).
#[rustfmt::skip]
pub static ROW_ISCAN_8X8: [i16; 64] = [
    1,  2,  3,  6,  9,  13, 20, 25, 4,  5,  8,  11, 16, 21, 31, 40,
    7,  10, 14, 17, 22, 28, 38, 47, 12, 15, 18, 24, 29, 35, 45, 53,
    19, 23, 26, 32, 36, 42, 51, 58, 27, 30, 34, 39, 44, 50, 56, 60,
    33, 37, 43, 48, 52, 55, 61, 62, 41, 46, 49, 54, 57, 59, 63, 64,
];

/// 8×8 default scan neighbours. 130 entries (`(64 + 1) * 2`) from
/// libvpx `vp9_scan.c:235-244` (`default_scan_8x8_neighbors`).
#[rustfmt::skip]
pub static DEFAULT_SCAN_8X8_NEIGHBORS: [i16; 130] = [
    0,  0,  0,  0,  0,  0,  8,  8,  1,  8,  1,  1,  9,  16, 16, 16, 2,  9,  2,
    2,  10, 17, 17, 24, 24, 24, 3,  10, 3,  3,  18, 25, 25, 32, 11, 18, 32, 32,
    4,  11, 26, 33, 19, 26, 4,  4,  33, 40, 12, 19, 40, 40, 5,  12, 27, 34, 34,
    41, 20, 27, 13, 20, 5,  5,  41, 48, 48, 48, 28, 35, 35, 42, 21, 28, 6,  6,
    6,  13, 42, 49, 49, 56, 36, 43, 14, 21, 29, 36, 7,  14, 43, 50, 50, 57, 22,
    29, 37, 44, 15, 22, 44, 51, 51, 58, 30, 37, 23, 30, 52, 59, 45, 52, 38, 45,
    31, 38, 53, 60, 46, 53, 39, 46, 54, 61, 47, 54, 55, 62, 0,  0,
];

/// 8×8 column scan neighbours. 130 entries from libvpx
/// `vp9_scan.c:213-222` (`col_scan_8x8_neighbors`).
#[rustfmt::skip]
pub static COL_SCAN_8X8_NEIGHBORS: [i16; 130] = [
    0,  0,  0,  0,  8,  8,  0,  0,  16, 16, 1,  1,  24, 24, 9,  9,  1,  1,  32,
    32, 17, 17, 2,  2,  25, 25, 10, 10, 40, 40, 2,  2,  18, 18, 33, 33, 3,  3,
    48, 48, 11, 11, 26, 26, 3,  3,  41, 41, 19, 19, 34, 34, 4,  4,  27, 27, 12,
    12, 49, 49, 42, 42, 20, 20, 4,  4,  35, 35, 5,  5,  28, 28, 50, 50, 43, 43,
    13, 13, 36, 36, 5,  5,  21, 21, 51, 51, 29, 29, 6,  6,  44, 44, 14, 14, 6,
    6,  37, 37, 52, 52, 22, 22, 7,  7,  30, 30, 45, 45, 15, 15, 38, 38, 23, 23,
    53, 53, 31, 31, 46, 46, 39, 39, 54, 54, 47, 47, 55, 55, 0,  0,
];

/// 8×8 row scan neighbours. 130 entries from libvpx
/// `vp9_scan.c:224-233` (`row_scan_8x8_neighbors`).
#[rustfmt::skip]
pub static ROW_SCAN_8X8_NEIGHBORS: [i16; 130] = [
    0,  0,  0,  0,  1,  1,  0,  0,  8,  8,  2,  2,  8,  8,  9,  9,  3,  3,  16,
    16, 10, 10, 16, 16, 4,  4,  17, 17, 24, 24, 11, 11, 18, 18, 25, 25, 24, 24,
    5,  5,  12, 12, 19, 19, 32, 32, 26, 26, 6,  6,  33, 33, 32, 32, 20, 20, 27,
    27, 40, 40, 13, 13, 34, 34, 40, 40, 41, 41, 28, 28, 35, 35, 48, 48, 21, 21,
    42, 42, 14, 14, 48, 48, 36, 36, 49, 49, 43, 43, 29, 29, 56, 56, 22, 22, 50,
    50, 57, 57, 44, 44, 37, 37, 51, 51, 30, 30, 58, 58, 52, 52, 45, 45, 59, 59,
    38, 38, 60, 60, 46, 46, 53, 53, 54, 54, 61, 61, 62, 62, 0,  0,
];

/// 8×8 default `ScanOrder`. Mirrors `vp9_default_scan_orders[Tx8x8]`
/// at libvpx `vp9_scan.c:693`.
#[inline]
#[must_use]
pub const fn default_scan_8x8() -> ScanOrder {
    ScanOrder {
        scan: &DEFAULT_SCAN_8X8,
        iscan: &DEFAULT_ISCAN_8X8,
        neighbors: &DEFAULT_SCAN_8X8_NEIGHBORS,
    }
}

/// 8×8 column `ScanOrder` (for `TxType::DctAdst`).
#[inline]
#[must_use]
pub const fn col_scan_8x8() -> ScanOrder {
    ScanOrder {
        scan: &COL_SCAN_8X8,
        iscan: &COL_ISCAN_8X8,
        neighbors: &COL_SCAN_8X8_NEIGHBORS,
    }
}

/// 8×8 row `ScanOrder` (for `TxType::AdstDct`).
#[inline]
#[must_use]
pub const fn row_scan_8x8() -> ScanOrder {
    ScanOrder {
        scan: &ROW_SCAN_8X8,
        iscan: &ROW_ISCAN_8X8,
        neighbors: &ROW_SCAN_8X8_NEIGHBORS,
    }
}

/// Return the 8×8 `ScanOrder` for the given `TxType`. Mirrors
/// `vp9_scan_orders[Tx8x8][tx_type]` at libvpx `vp9_scan.c:705-708`
/// (same dispatch pattern as 4×4).
#[inline]
#[must_use]
pub const fn scan_order_8x8(tx_type: super::types::TxType) -> ScanOrder {
    use super::types::TxType;
    match tx_type {
        TxType::DctDct => default_scan_8x8(),
        TxType::AdstDct => row_scan_8x8(),
        TxType::DctAdst => col_scan_8x8(),
        TxType::AdstAdst => default_scan_8x8(),
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

    // ---- 8x8 family ----

    #[test]
    fn default_scan_8x8_is_a_permutation() {
        assert_scan_is_permutation(&DEFAULT_SCAN_8X8, TxSize::Tx8x8.n_coeffs());
    }

    #[test]
    fn col_scan_8x8_is_a_permutation() {
        assert_scan_is_permutation(&COL_SCAN_8X8, TxSize::Tx8x8.n_coeffs());
    }

    #[test]
    fn row_scan_8x8_is_a_permutation() {
        assert_scan_is_permutation(&ROW_SCAN_8X8, TxSize::Tx8x8.n_coeffs());
    }

    #[test]
    fn default_iscan_8x8_is_inverse_of_default_scan() {
        assert_iscan_is_inverse_of_scan_plus_one(&DEFAULT_SCAN_8X8, &DEFAULT_ISCAN_8X8);
    }

    #[test]
    fn col_iscan_8x8_is_inverse_of_col_scan() {
        assert_iscan_is_inverse_of_scan_plus_one(&COL_SCAN_8X8, &COL_ISCAN_8X8);
    }

    #[test]
    fn row_iscan_8x8_is_inverse_of_row_scan() {
        assert_iscan_is_inverse_of_scan_plus_one(&ROW_SCAN_8X8, &ROW_ISCAN_8X8);
    }

    #[test]
    fn default_scan_8x8_neighbors_length() {
        assert_eq!(
            DEFAULT_SCAN_8X8_NEIGHBORS.len(),
            (TxSize::Tx8x8.n_coeffs() + 1) * MAX_NEIGHBORS
        );
    }

    #[test]
    fn col_scan_8x8_neighbors_length() {
        assert_eq!(
            COL_SCAN_8X8_NEIGHBORS.len(),
            (TxSize::Tx8x8.n_coeffs() + 1) * MAX_NEIGHBORS
        );
    }

    #[test]
    fn row_scan_8x8_neighbors_length() {
        assert_eq!(
            ROW_SCAN_8X8_NEIGHBORS.len(),
            (TxSize::Tx8x8.n_coeffs() + 1) * MAX_NEIGHBORS
        );
    }

    /// 8×8 neighbours arrays must all end with the (0, 0) EOB sentinel.
    #[test]
    fn neighbors_8x8_trailing_sentinel() {
        for arr in [
            &DEFAULT_SCAN_8X8_NEIGHBORS[..],
            &COL_SCAN_8X8_NEIGHBORS[..],
            &ROW_SCAN_8X8_NEIGHBORS[..],
        ] {
            let n = arr.len();
            assert_eq!(arr[n - 2], 0);
            assert_eq!(arr[n - 1], 0);
        }
    }

    /// `scan_order_8x8` dispatch matches libvpx's
    /// `vp9_scan_orders[Tx8x8][tx_type]` table at lines 705-708.
    #[test]
    fn scan_order_8x8_dispatch() {
        use crate::transforms::types::TxType;

        assert_eq!(
            scan_order_8x8(TxType::DctDct).scan.as_ptr(),
            DEFAULT_SCAN_8X8.as_ptr()
        );
        assert_eq!(
            scan_order_8x8(TxType::AdstDct).scan.as_ptr(),
            ROW_SCAN_8X8.as_ptr()
        );
        assert_eq!(
            scan_order_8x8(TxType::DctAdst).scan.as_ptr(),
            COL_SCAN_8X8.as_ptr()
        );
        assert_eq!(
            scan_order_8x8(TxType::AdstAdst).scan.as_ptr(),
            DEFAULT_SCAN_8X8.as_ptr()
        );
    }
}
