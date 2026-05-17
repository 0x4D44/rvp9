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

// ============================================================================
// 16×16 scan family
// ============================================================================

/// 16×16 default scan. 256 entries verbatim from libvpx
/// `vp9/common/vp9_scan.c:48-67`.
#[rustfmt::skip]
pub static DEFAULT_SCAN_16X16: [i16; 256] = [
    0,   16,  1,   32,  17,  2,   48,  33,  18,  3,   64,  34,  49,  19,  65,
    80,  50,  4,   35,  66,  20,  81,  96,  51,  5,   36,  82,  97,  67,  112,
    21,  52,  98,  37,  83,  113, 6,   68,  128, 53,  22,  99,  114, 84,  7,
    129, 38,  69,  100, 115, 144, 130, 85,  54,  23,  8,   145, 39,  70,  116,
    101, 131, 160, 146, 55,  86,  24,  71,  132, 117, 161, 40,  9,   102, 147,
    176, 162, 87,  56,  25,  133, 118, 177, 148, 72,  103, 41,  163, 10,  192,
    178, 88,  57,  134, 149, 119, 26,  164, 73,  104, 193, 42,  179, 208, 11,
    135, 89,  165, 120, 150, 58,  194, 180, 27,  74,  209, 105, 151, 136, 43,
    90,  224, 166, 195, 181, 121, 210, 59,  12,  152, 106, 167, 196, 75,  137,
    225, 211, 240, 182, 122, 91,  28,  197, 13,  226, 168, 183, 153, 44,  212,
    138, 107, 241, 60,  29,  123, 198, 184, 227, 169, 242, 76,  213, 154, 45,
    92,  14,  199, 139, 61,  228, 214, 170, 185, 243, 108, 77,  155, 30,  15,
    200, 229, 124, 215, 244, 93,  46,  186, 171, 201, 109, 140, 230, 62,  216,
    245, 31,  125, 78,  156, 231, 47,  187, 202, 217, 94,  246, 141, 63,  232,
    172, 110, 247, 157, 79,  218, 203, 126, 233, 188, 248, 95,  173, 142, 219,
    111, 249, 234, 158, 127, 189, 204, 250, 235, 143, 174, 220, 205, 159, 251,
    190, 221, 175, 236, 237, 191, 206, 252, 222, 253, 207, 238, 223, 254, 239,
    255,
];

/// 16×16 column scan. 256 entries verbatim from libvpx `vp9_scan.c:69-88`.
#[rustfmt::skip]
pub static COL_SCAN_16X16: [i16; 256] = [
    0,   16,  32,  48,  1,   64,  17,  80,  33,  96,  49,  2,   65,  112, 18,
    81,  34,  128, 50,  97,  3,   66,  144, 19,  113, 35,  82,  160, 98,  51,
    129, 4,   67,  176, 20,  114, 145, 83,  36,  99,  130, 52,  192, 5,   161,
    68,  115, 21,  146, 84,  208, 177, 37,  131, 100, 53,  162, 224, 69,  6,
    116, 193, 147, 85,  22,  240, 132, 38,  178, 101, 163, 54,  209, 117, 70,
    7,   148, 194, 86,  179, 225, 23,  133, 39,  164, 8,   102, 210, 241, 55,
    195, 118, 149, 71,  180, 24,  87,  226, 134, 165, 211, 40,  103, 56,  72,
    150, 196, 242, 119, 9,   181, 227, 88,  166, 25,  135, 41,  104, 212, 57,
    151, 197, 120, 73,  243, 182, 136, 167, 213, 89,  10,  228, 105, 152, 198,
    26,  42,  121, 183, 244, 168, 58,  137, 229, 74,  214, 90,  153, 199, 184,
    11,  106, 245, 27,  122, 230, 169, 43,  215, 59,  200, 138, 185, 246, 75,
    12,  91,  154, 216, 231, 107, 28,  44,  201, 123, 170, 60,  247, 232, 76,
    139, 13,  92,  217, 186, 248, 155, 108, 29,  124, 45,  202, 233, 171, 61,
    14,  77,  140, 15,  249, 93,  30,  187, 156, 218, 46,  109, 125, 62,  172,
    78,  203, 31,  141, 234, 94,  47,  188, 63,  157, 110, 250, 219, 79,  126,
    204, 173, 142, 95,  189, 111, 235, 158, 220, 251, 127, 174, 143, 205, 236,
    159, 190, 221, 252, 175, 206, 237, 191, 253, 222, 238, 207, 254, 223, 239,
    255,
];

/// 16×16 row scan. 256 entries verbatim from libvpx `vp9_scan.c:90-109`.
#[rustfmt::skip]
pub static ROW_SCAN_16X16: [i16; 256] = [
    0,   1,   2,   16,  3,   17,  4,   18,  32,  5,   33,  19,  6,   34,  48,
    20,  49,  7,   35,  21,  50,  64,  8,   36,  65,  22,  51,  37,  80,  9,
    66,  52,  23,  38,  81,  67,  10,  53,  24,  82,  68,  96,  39,  11,  54,
    83,  97,  69,  25,  98,  84,  40,  112, 55,  12,  70,  99,  113, 85,  26,
    41,  56,  114, 100, 13,  71,  128, 86,  27,  115, 101, 129, 42,  57,  72,
    116, 14,  87,  130, 102, 144, 73,  131, 117, 28,  58,  15,  88,  43,  145,
    103, 132, 146, 118, 74,  160, 89,  133, 104, 29,  59,  147, 119, 44,  161,
    148, 90,  105, 134, 162, 120, 176, 75,  135, 149, 30,  60,  163, 177, 45,
    121, 91,  106, 164, 178, 150, 192, 136, 165, 179, 31,  151, 193, 76,  122,
    61,  137, 194, 107, 152, 180, 208, 46,  166, 167, 195, 92,  181, 138, 209,
    123, 153, 224, 196, 77,  168, 210, 182, 240, 108, 197, 62,  154, 225, 183,
    169, 211, 47,  139, 93,  184, 226, 212, 241, 198, 170, 124, 155, 199, 78,
    213, 185, 109, 227, 200, 63,  228, 242, 140, 214, 171, 186, 156, 229, 243,
    125, 94,  201, 244, 215, 216, 230, 141, 187, 202, 79,  172, 110, 157, 245,
    217, 231, 95,  246, 232, 126, 203, 247, 233, 173, 218, 142, 111, 158, 188,
    248, 127, 234, 219, 249, 189, 204, 143, 174, 159, 250, 235, 205, 220, 175,
    190, 251, 221, 191, 206, 236, 207, 237, 252, 222, 253, 223, 238, 239, 254,
    255,
];

/// 16×16 default inverse scan. 256 entries verbatim from libvpx
/// `vp9_scan.c:588-607` (`vp9_default_iscan_16x16`).
#[rustfmt::skip]
pub static DEFAULT_ISCAN_16X16: [i16; 256] = [
    1,   3,   6,   10,  18,  25,  37,  45,  56,  73,  89,  105, 129, 144, 167,
    180, 2,   5,   9,   14,  21,  31,  41,  55,  67,  80,  97,  114, 142, 155,
    179, 197, 4,   8,   12,  19,  26,  34,  47,  58,  72,  87,  102, 120, 149,
    165, 187, 202, 7,   13,  17,  24,  32,  40,  54,  65,  79,  93,  111, 128,
    154, 170, 194, 209, 11,  15,  20,  29,  38,  48,  59,  68,  85,  99,  115,
    134, 162, 177, 199, 215, 16,  22,  27,  35,  44,  53,  66,  78,  92,  107,
    121, 141, 166, 186, 206, 222, 23,  28,  33,  42,  49,  61,  74,  86,  100,
    117, 131, 152, 176, 191, 212, 226, 30,  36,  43,  50,  60,  70,  82,  96,
    109, 126, 140, 156, 183, 198, 218, 230, 39,  46,  52,  62,  69,  81,  94,
    106, 119, 135, 151, 169, 192, 208, 224, 235, 51,  57,  64,  75,  84,  95,
    110, 118, 130, 148, 164, 178, 200, 214, 229, 239, 63,  71,  77,  88,  98,
    108, 123, 132, 146, 160, 173, 189, 211, 223, 236, 243, 76,  83,  91,  103,
    113, 125, 139, 147, 158, 174, 188, 203, 220, 231, 241, 246, 90,  101, 112,
    124, 133, 143, 157, 168, 181, 190, 204, 217, 232, 238, 247, 251, 104, 116,
    127, 137, 150, 163, 172, 184, 195, 205, 216, 225, 237, 242, 249, 253, 122,
    136, 145, 159, 171, 182, 193, 201, 210, 219, 228, 234, 244, 245, 252, 255,
    138, 153, 161, 175, 185, 196, 207, 213, 221, 227, 233, 240, 248, 250, 254,
    256,
];

/// 16×16 column inverse scan. 256 entries verbatim from libvpx
/// `vp9_scan.c:548-565` (`vp9_col_iscan_16x16`).
#[rustfmt::skip]
pub static COL_ISCAN_16X16: [i16; 256] = [
    1,  5,  12,  21,  32,  44,  60,  76,  86,  110, 131, 151, 166, 182, 196, 199,
    2,  7,  15,  24,  35,  48,  65,  82,  96,  115, 136, 154, 172, 189, 202, 213,
    3,  9,  17,  26,  39,  53,  68,  84,  102, 117, 137, 158, 173, 191, 206, 217,
    4,  11, 19,  30,  42,  56,  72,  90,  104, 120, 142, 160, 177, 195, 209, 219,
    6,  13, 22,  33,  46,  59,  75,  94,  105, 124, 145, 165, 180, 197, 211, 224,
    8,  16, 27,  38,  50,  64,  79,  97,  113, 130, 147, 167, 183, 201, 216, 229,
    10, 20, 29,  40,  55,  70,  87,  103, 118, 133, 152, 171, 188, 207, 221, 231,
    14, 25, 36,  47,  61,  74,  92,  109, 123, 138, 155, 175, 190, 208, 225, 236,
    18, 31, 41,  54,  67,  83,  99,  116, 127, 143, 162, 181, 198, 214, 228, 238,
    23, 37, 49,  63,  77,  93,  106, 121, 134, 148, 168, 187, 204, 220, 233, 241,
    28, 45, 57,  71,  85,  100, 114, 128, 141, 157, 176, 194, 210, 227, 237, 245,
    34, 52, 69,  80,  95,  111, 126, 139, 150, 163, 185, 203, 218, 230, 242, 248,
    43, 62, 78,  91,  107, 122, 135, 149, 161, 174, 192, 212, 226, 239, 246, 252,
    51, 73, 88,  101, 119, 129, 146, 159, 169, 184, 205, 223, 234, 243, 250, 254,
    58, 81, 98,  112, 132, 144, 156, 170, 179, 193, 215, 232, 240, 247, 251, 255,
    66, 89, 108, 125, 140, 153, 164, 178, 186, 200, 222, 235, 244, 249, 253, 256,
];

/// 16×16 row inverse scan. 256 entries verbatim from libvpx
/// `vp9_scan.c:567-586` (`vp9_row_iscan_16x16`).
#[rustfmt::skip]
pub static ROW_ISCAN_16X16: [i16; 256] = [
    1,   2,   3,   5,   7,   10,  13,  18,  23,  30,  37,  44,  55,  65,  77,
    87,  4,   6,   8,   12,  16,  20,  26,  33,  39,  49,  60,  69,  85,  100,
    116, 131, 9,   11,  14,  19,  24,  28,  34,  43,  52,  61,  73,  89,  104,
    120, 143, 168, 15,  17,  21,  27,  32,  38,  45,  54,  62,  74,  86,  101,
    117, 136, 162, 186, 22,  25,  31,  36,  41,  48,  56,  66,  75,  82,  95,
    113, 134, 155, 180, 206, 29,  35,  40,  46,  51,  59,  68,  78,  88,  97,
    107, 122, 147, 170, 197, 213, 42,  47,  50,  57,  64,  71,  80,  91,  99,
    108, 123, 139, 160, 183, 208, 223, 53,  58,  63,  70,  76,  84,  94,  103,
    111, 121, 135, 151, 177, 196, 216, 227, 67,  72,  79,  83,  92,  98,  109,
    114, 128, 137, 149, 169, 189, 203, 222, 233, 81,  90,  93,  102, 106, 115,
    126, 132, 140, 152, 163, 178, 193, 209, 224, 235, 96,  105, 110, 118, 124,
    129, 144, 145, 156, 166, 176, 191, 207, 220, 234, 240, 112, 119, 125, 130,
    141, 148, 158, 165, 171, 182, 192, 204, 225, 231, 241, 244, 127, 133, 138,
    146, 154, 161, 175, 179, 185, 198, 205, 217, 232, 238, 245, 247, 142, 150,
    157, 167, 173, 181, 190, 200, 201, 211, 221, 229, 239, 243, 250, 252, 153,
    164, 172, 184, 187, 194, 202, 212, 215, 219, 228, 237, 246, 248, 253, 254,
    159, 174, 188, 195, 199, 210, 214, 218, 226, 230, 236, 242, 249, 251, 255,
    256,
];

/// 16×16 default scan neighbours. 514 entries (`(256 + 1) * 2`) from
/// libvpx `vp9_scan.c:324-361` (`default_scan_16x16_neighbors`).
#[rustfmt::skip]
pub static DEFAULT_SCAN_16X16_NEIGHBORS: [i16; 514] = [
    0,   0,   0,   0,   0,   0,   16,  16,  1,   16,  1,   1,   32,  32,  17,
    32,  2,   17,  2,   2,   48,  48,  18,  33,  33,  48,  3,   18,  49,  64,
    64,  64,  34,  49,  3,   3,   19,  34,  50,  65,  4,   19,  65,  80,  80,
    80,  35,  50,  4,   4,   20,  35,  66,  81,  81,  96,  51,  66,  96,  96,
    5,   20,  36,  51,  82,  97,  21,  36,  67,  82,  97,  112, 5,   5,   52,
    67,  112, 112, 37,  52,  6,   21,  83,  98,  98,  113, 68,  83,  6,   6,
    113, 128, 22,  37,  53,  68,  84,  99,  99,  114, 128, 128, 114, 129, 69,
    84,  38,  53,  7,   22,  7,   7,   129, 144, 23,  38,  54,  69,  100, 115,
    85,  100, 115, 130, 144, 144, 130, 145, 39,  54,  70,  85,  8,   23,  55,
    70,  116, 131, 101, 116, 145, 160, 24,  39,  8,   8,   86,  101, 131, 146,
    160, 160, 146, 161, 71,  86,  40,  55,  9,   24,  117, 132, 102, 117, 161,
    176, 132, 147, 56,  71,  87,  102, 25,  40,  147, 162, 9,   9,   176, 176,
    162, 177, 72,  87,  41,  56,  118, 133, 133, 148, 103, 118, 10,  25,  148,
    163, 57,  72,  88,  103, 177, 192, 26,  41,  163, 178, 192, 192, 10,  10,
    119, 134, 73,  88,  149, 164, 104, 119, 134, 149, 42,  57,  178, 193, 164,
    179, 11,  26,  58,  73,  193, 208, 89,  104, 135, 150, 120, 135, 27,  42,
    74,  89,  208, 208, 150, 165, 179, 194, 165, 180, 105, 120, 194, 209, 43,
    58,  11,  11,  136, 151, 90,  105, 151, 166, 180, 195, 59,  74,  121, 136,
    209, 224, 195, 210, 224, 224, 166, 181, 106, 121, 75,  90,  12,  27,  181,
    196, 12,  12,  210, 225, 152, 167, 167, 182, 137, 152, 28,  43,  196, 211,
    122, 137, 91,  106, 225, 240, 44,  59,  13,  28,  107, 122, 182, 197, 168,
    183, 211, 226, 153, 168, 226, 241, 60,  75,  197, 212, 138, 153, 29,  44,
    76,  91,  13,  13,  183, 198, 123, 138, 45,  60,  212, 227, 198, 213, 154,
    169, 169, 184, 227, 242, 92,  107, 61,  76,  139, 154, 14,  29,  14,  14,
    184, 199, 213, 228, 108, 123, 199, 214, 228, 243, 77,  92,  30,  45,  170,
    185, 155, 170, 185, 200, 93,  108, 124, 139, 214, 229, 46,  61,  200, 215,
    229, 244, 15,  30,  109, 124, 62,  77,  140, 155, 215, 230, 31,  46,  171,
    186, 186, 201, 201, 216, 78,  93,  230, 245, 125, 140, 47,  62,  216, 231,
    156, 171, 94,  109, 231, 246, 141, 156, 63,  78,  202, 217, 187, 202, 110,
    125, 217, 232, 172, 187, 232, 247, 79,  94,  157, 172, 126, 141, 203, 218,
    95,  110, 233, 248, 218, 233, 142, 157, 111, 126, 173, 188, 188, 203, 234,
    249, 219, 234, 127, 142, 158, 173, 204, 219, 189, 204, 143, 158, 235, 250,
    174, 189, 205, 220, 159, 174, 220, 235, 221, 236, 175, 190, 190, 205, 236,
    251, 206, 221, 237, 252, 191, 206, 222, 237, 207, 222, 238, 253, 223, 238,
    239, 254, 0,   0,
];

/// 16×16 column scan neighbours. 514 entries from libvpx
/// `vp9_scan.c:246-283` (`col_scan_16x16_neighbors`).
#[rustfmt::skip]
pub static COL_SCAN_16X16_NEIGHBORS: [i16; 514] = [
    0,   0,   0,   0,   16,  16,  32,  32,  0,   0,   48,  48,  1,   1,   64,
    64,  17,  17,  80,  80,  33,  33,  1,   1,   49,  49,  96,  96,  2,   2,
    65,  65,  18,  18,  112, 112, 34,  34,  81,  81,  2,   2,   50,  50,  128,
    128, 3,   3,   97,  97,  19,  19,  66,  66,  144, 144, 82,  82,  35,  35,
    113, 113, 3,   3,   51,  51,  160, 160, 4,   4,   98,  98,  129, 129, 67,
    67,  20,  20,  83,  83,  114, 114, 36,  36,  176, 176, 4,   4,   145, 145,
    52,  52,  99,  99,  5,   5,   130, 130, 68,  68,  192, 192, 161, 161, 21,
    21,  115, 115, 84,  84,  37,  37,  146, 146, 208, 208, 53,  53,  5,   5,
    100, 100, 177, 177, 131, 131, 69,  69,  6,   6,   224, 224, 116, 116, 22,
    22,  162, 162, 85,  85,  147, 147, 38,  38,  193, 193, 101, 101, 54,  54,
    6,   6,   132, 132, 178, 178, 70,  70,  163, 163, 209, 209, 7,   7,   117,
    117, 23,  23,  148, 148, 7,   7,   86,  86,  194, 194, 225, 225, 39,  39,
    179, 179, 102, 102, 133, 133, 55,  55,  164, 164, 8,   8,   71,  71,  210,
    210, 118, 118, 149, 149, 195, 195, 24,  24,  87,  87,  40,  40,  56,  56,
    134, 134, 180, 180, 226, 226, 103, 103, 8,   8,   165, 165, 211, 211, 72,
    72,  150, 150, 9,   9,   119, 119, 25,  25,  88,  88,  196, 196, 41,  41,
    135, 135, 181, 181, 104, 104, 57,  57,  227, 227, 166, 166, 120, 120, 151,
    151, 197, 197, 73,  73,  9,   9,   212, 212, 89,  89,  136, 136, 182, 182,
    10,  10,  26,  26,  105, 105, 167, 167, 228, 228, 152, 152, 42,  42,  121,
    121, 213, 213, 58,  58,  198, 198, 74,  74,  137, 137, 183, 183, 168, 168,
    10,  10,  90,  90,  229, 229, 11,  11,  106, 106, 214, 214, 153, 153, 27,
    27,  199, 199, 43,  43,  184, 184, 122, 122, 169, 169, 230, 230, 59,  59,
    11,  11,  75,  75,  138, 138, 200, 200, 215, 215, 91,  91,  12,  12,  28,
    28,  185, 185, 107, 107, 154, 154, 44,  44,  231, 231, 216, 216, 60,  60,
    123, 123, 12,  12,  76,  76,  201, 201, 170, 170, 232, 232, 139, 139, 92,
    92,  13,  13,  108, 108, 29,  29,  186, 186, 217, 217, 155, 155, 45,  45,
    13,  13,  61,  61,  124, 124, 14,  14,  233, 233, 77,  77,  14,  14,  171,
    171, 140, 140, 202, 202, 30,  30,  93,  93,  109, 109, 46,  46,  156, 156,
    62,  62,  187, 187, 15,  15,  125, 125, 218, 218, 78,  78,  31,  31,  172,
    172, 47,  47,  141, 141, 94,  94,  234, 234, 203, 203, 63,  63,  110, 110,
    188, 188, 157, 157, 126, 126, 79,  79,  173, 173, 95,  95,  219, 219, 142,
    142, 204, 204, 235, 235, 111, 111, 158, 158, 127, 127, 189, 189, 220, 220,
    143, 143, 174, 174, 205, 205, 236, 236, 159, 159, 190, 190, 221, 221, 175,
    175, 237, 237, 206, 206, 222, 222, 191, 191, 238, 238, 207, 207, 223, 223,
    239, 239, 0,   0,
];

/// 16×16 row scan neighbours. 514 entries from libvpx
/// `vp9_scan.c:285-322` (`row_scan_16x16_neighbors`).
#[rustfmt::skip]
pub static ROW_SCAN_16X16_NEIGHBORS: [i16; 514] = [
    0,   0,   0,   0,   1,   1,   0,   0,   2,   2,   16,  16,  3,   3,   17,
    17,  16,  16,  4,   4,   32,  32,  18,  18,  5,   5,   33,  33,  32,  32,
    19,  19,  48,  48,  6,   6,   34,  34,  20,  20,  49,  49,  48,  48,  7,
    7,   35,  35,  64,  64,  21,  21,  50,  50,  36,  36,  64,  64,  8,   8,
    65,  65,  51,  51,  22,  22,  37,  37,  80,  80,  66,  66,  9,   9,   52,
    52,  23,  23,  81,  81,  67,  67,  80,  80,  38,  38,  10,  10,  53,  53,
    82,  82,  96,  96,  68,  68,  24,  24,  97,  97,  83,  83,  39,  39,  96,
    96,  54,  54,  11,  11,  69,  69,  98,  98,  112, 112, 84,  84,  25,  25,
    40,  40,  55,  55,  113, 113, 99,  99,  12,  12,  70,  70,  112, 112, 85,
    85,  26,  26,  114, 114, 100, 100, 128, 128, 41,  41,  56,  56,  71,  71,
    115, 115, 13,  13,  86,  86,  129, 129, 101, 101, 128, 128, 72,  72,  130,
    130, 116, 116, 27,  27,  57,  57,  14,  14,  87,  87,  42,  42,  144, 144,
    102, 102, 131, 131, 145, 145, 117, 117, 73,  73,  144, 144, 88,  88,  132,
    132, 103, 103, 28,  28,  58,  58,  146, 146, 118, 118, 43,  43,  160, 160,
    147, 147, 89,  89,  104, 104, 133, 133, 161, 161, 119, 119, 160, 160, 74,
    74,  134, 134, 148, 148, 29,  29,  59,  59,  162, 162, 176, 176, 44,  44,
    120, 120, 90,  90,  105, 105, 163, 163, 177, 177, 149, 149, 176, 176, 135,
    135, 164, 164, 178, 178, 30,  30,  150, 150, 192, 192, 75,  75,  121, 121,
    60,  60,  136, 136, 193, 193, 106, 106, 151, 151, 179, 179, 192, 192, 45,
    45,  165, 165, 166, 166, 194, 194, 91,  91,  180, 180, 137, 137, 208, 208,
    122, 122, 152, 152, 208, 208, 195, 195, 76,  76,  167, 167, 209, 209, 181,
    181, 224, 224, 107, 107, 196, 196, 61,  61,  153, 153, 224, 224, 182, 182,
    168, 168, 210, 210, 46,  46,  138, 138, 92,  92,  183, 183, 225, 225, 211,
    211, 240, 240, 197, 197, 169, 169, 123, 123, 154, 154, 198, 198, 77,  77,
    212, 212, 184, 184, 108, 108, 226, 226, 199, 199, 62,  62,  227, 227, 241,
    241, 139, 139, 213, 213, 170, 170, 185, 185, 155, 155, 228, 228, 242, 242,
    124, 124, 93,  93,  200, 200, 243, 243, 214, 214, 215, 215, 229, 229, 140,
    140, 186, 186, 201, 201, 78,  78,  171, 171, 109, 109, 156, 156, 244, 244,
    216, 216, 230, 230, 94,  94,  245, 245, 231, 231, 125, 125, 202, 202, 246,
    246, 232, 232, 172, 172, 217, 217, 141, 141, 110, 110, 157, 157, 187, 187,
    247, 247, 126, 126, 233, 233, 218, 218, 248, 248, 188, 188, 203, 203, 142,
    142, 173, 173, 158, 158, 249, 249, 234, 234, 204, 204, 219, 219, 174, 174,
    189, 189, 250, 250, 220, 220, 190, 190, 205, 205, 235, 235, 206, 206, 236,
    236, 251, 251, 221, 221, 252, 252, 222, 222, 237, 237, 238, 238, 253, 253,
    254, 254, 0,   0,
];

/// 16×16 default `ScanOrder`. Mirrors `vp9_default_scan_orders[Tx16x16]`
/// at libvpx `vp9_scan.c:694`.
#[inline]
#[must_use]
pub const fn default_scan_16x16() -> ScanOrder {
    ScanOrder {
        scan: &DEFAULT_SCAN_16X16,
        iscan: &DEFAULT_ISCAN_16X16,
        neighbors: &DEFAULT_SCAN_16X16_NEIGHBORS,
    }
}

/// 16×16 column `ScanOrder` (for `TxType::DctAdst`).
#[inline]
#[must_use]
pub const fn col_scan_16x16() -> ScanOrder {
    ScanOrder {
        scan: &COL_SCAN_16X16,
        iscan: &COL_ISCAN_16X16,
        neighbors: &COL_SCAN_16X16_NEIGHBORS,
    }
}

/// 16×16 row `ScanOrder` (for `TxType::AdstDct`).
#[inline]
#[must_use]
pub const fn row_scan_16x16() -> ScanOrder {
    ScanOrder {
        scan: &ROW_SCAN_16X16,
        iscan: &ROW_ISCAN_16X16,
        neighbors: &ROW_SCAN_16X16_NEIGHBORS,
    }
}

/// Return the 16×16 `ScanOrder` for the given `TxType`. Mirrors
/// `vp9_scan_orders[Tx16x16][tx_type]` at libvpx `vp9_scan.c:710-714`.
#[inline]
#[must_use]
pub const fn scan_order_16x16(tx_type: super::types::TxType) -> ScanOrder {
    use super::types::TxType;
    match tx_type {
        TxType::DctDct => default_scan_16x16(),
        TxType::AdstDct => row_scan_16x16(),
        TxType::DctAdst => col_scan_16x16(),
        TxType::AdstAdst => default_scan_16x16(),
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

    // ---- 16x16 family ----

    #[test]
    fn default_scan_16x16_is_a_permutation() {
        assert_scan_is_permutation(&DEFAULT_SCAN_16X16, TxSize::Tx16x16.n_coeffs());
    }

    #[test]
    fn col_scan_16x16_is_a_permutation() {
        assert_scan_is_permutation(&COL_SCAN_16X16, TxSize::Tx16x16.n_coeffs());
    }

    #[test]
    fn row_scan_16x16_is_a_permutation() {
        assert_scan_is_permutation(&ROW_SCAN_16X16, TxSize::Tx16x16.n_coeffs());
    }

    #[test]
    fn default_iscan_16x16_is_inverse_of_default_scan() {
        assert_iscan_is_inverse_of_scan_plus_one(&DEFAULT_SCAN_16X16, &DEFAULT_ISCAN_16X16);
    }

    #[test]
    fn col_iscan_16x16_is_inverse_of_col_scan() {
        assert_iscan_is_inverse_of_scan_plus_one(&COL_SCAN_16X16, &COL_ISCAN_16X16);
    }

    #[test]
    fn row_iscan_16x16_is_inverse_of_row_scan() {
        assert_iscan_is_inverse_of_scan_plus_one(&ROW_SCAN_16X16, &ROW_ISCAN_16X16);
    }

    #[test]
    fn neighbors_16x16_lengths() {
        let expected = (TxSize::Tx16x16.n_coeffs() + 1) * MAX_NEIGHBORS;
        assert_eq!(DEFAULT_SCAN_16X16_NEIGHBORS.len(), expected);
        assert_eq!(COL_SCAN_16X16_NEIGHBORS.len(), expected);
        assert_eq!(ROW_SCAN_16X16_NEIGHBORS.len(), expected);
    }

    /// All 16x16 neighbours arrays end with the (0, 0) EOB sentinel.
    #[test]
    fn neighbors_16x16_trailing_sentinel() {
        for arr in [
            &DEFAULT_SCAN_16X16_NEIGHBORS[..],
            &COL_SCAN_16X16_NEIGHBORS[..],
            &ROW_SCAN_16X16_NEIGHBORS[..],
        ] {
            let n = arr.len();
            assert_eq!(arr[n - 2], 0);
            assert_eq!(arr[n - 1], 0);
        }
    }

    /// `scan_order_16x16` dispatch matches libvpx's
    /// `vp9_scan_orders[Tx16x16][tx_type]` table at lines 710-714.
    #[test]
    fn scan_order_16x16_dispatch() {
        use crate::transforms::types::TxType;

        assert_eq!(
            scan_order_16x16(TxType::DctDct).scan.as_ptr(),
            DEFAULT_SCAN_16X16.as_ptr()
        );
        assert_eq!(
            scan_order_16x16(TxType::AdstDct).scan.as_ptr(),
            ROW_SCAN_16X16.as_ptr()
        );
        assert_eq!(
            scan_order_16x16(TxType::DctAdst).scan.as_ptr(),
            COL_SCAN_16X16.as_ptr()
        );
        assert_eq!(
            scan_order_16x16(TxType::AdstAdst).scan.as_ptr(),
            DEFAULT_SCAN_16X16.as_ptr()
        );
    }
}
