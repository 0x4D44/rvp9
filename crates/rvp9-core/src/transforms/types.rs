//! Transform-layer types: `TxSize`, `TxType`, `TxMode`, plus the
//! `MAX_NEIGHBORS` constant.
//!
//! Mirrors libvpx `vp9/common/vp9_enums.h:77-99` for the enums and
//! `vp9/common/vp9_scan.h:24` for `MAX_NEIGHBORS`.

/// VP9 transform size. Each variant is a power-of-two block side length
/// from 4 to 32 pixels. Mirrors libvpx `TX_SIZE` (`vp9_enums.h:77-82`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum TxSize {
    /// 4×4
    Tx4x4 = 0,
    /// 8×8
    Tx8x8 = 1,
    /// 16×16
    Tx16x16 = 2,
    /// 32×32
    Tx32x32 = 3,
}

impl TxSize {
    /// `log2(side_length)` — returns `2`, `3`, `4`, or `5`. Useful for
    /// shift-based index math.
    #[inline]
    #[must_use]
    pub const fn block_size_log2(self) -> u8 {
        (self as u8) + 2
    }

    /// Number of coefficients in a block of this size:
    /// 16 / 64 / 256 / 1024.
    #[inline]
    #[must_use]
    pub const fn n_coeffs(self) -> usize {
        1usize << (2 * self.block_size_log2())
    }

    /// Side length in pixels: 4 / 8 / 16 / 32.
    #[inline]
    #[must_use]
    pub const fn side(self) -> usize {
        1usize << self.block_size_log2()
    }
}

/// VP9 transform type — selects DCT vs ADST per axis. Inter blocks and
/// chroma always use `DctDct`; intra-Y uses `intra_mode_to_tx_type` (a
/// table living elsewhere in this module). Mirrors libvpx `TX_TYPE`
/// (`vp9_enums.h:95-99`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum TxType {
    /// DCT on both axes (default for inter / chroma).
    DctDct = 0,
    /// ADST on rows, DCT on columns.
    AdstDct = 1,
    /// DCT on rows, ADST on columns.
    DctAdst = 2,
    /// ADST on both axes.
    AdstAdst = 3,
}

impl TxType {
    /// Numeric index for table indexing (`0..4`).
    #[inline]
    #[must_use]
    pub const fn idx(self) -> usize {
        self as usize
    }
}

/// Frame-level transform mode selector. Mirrors libvpx `TX_MODE`
/// (`vp9_enums.h:86-92`). Encoder writes this in the compressed header;
/// decoder reads it. `OnlyTx4x4` is forced when the frame is lossless.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TxMode {
    /// Force all blocks to 4×4 (also the lossless mode).
    OnlyTx4x4 = 0,
    /// Allow up to 8×8.
    AllowTx8x8 = 1,
    /// Allow up to 16×16.
    AllowTx16x16 = 2,
    /// Allow up to 32×32.
    AllowTx32x32 = 3,
    /// Per-block selection encoded in the bitstream.
    TxModeSelect = 4,
}

/// The number of {top, left} neighbour positions stored per coefficient
/// in a `ScanOrder::neighbors` table. Always `2` (libvpx
/// `vp9_scan.h:24`).
pub const MAX_NEIGHBORS: usize = 2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tx_size_invariants() {
        assert_eq!(TxSize::Tx4x4.side(), 4);
        assert_eq!(TxSize::Tx8x8.side(), 8);
        assert_eq!(TxSize::Tx16x16.side(), 16);
        assert_eq!(TxSize::Tx32x32.side(), 32);
        assert_eq!(TxSize::Tx4x4.n_coeffs(), 16);
        assert_eq!(TxSize::Tx8x8.n_coeffs(), 64);
        assert_eq!(TxSize::Tx16x16.n_coeffs(), 256);
        assert_eq!(TxSize::Tx32x32.n_coeffs(), 1024);
        assert_eq!(TxSize::Tx4x4.block_size_log2(), 2);
        assert_eq!(TxSize::Tx32x32.block_size_log2(), 5);
    }

    #[test]
    fn tx_type_repr() {
        assert_eq!(TxType::DctDct as u8, 0);
        assert_eq!(TxType::AdstDct as u8, 1);
        assert_eq!(TxType::DctAdst as u8, 2);
        assert_eq!(TxType::AdstAdst as u8, 3);
        assert_eq!(TxType::DctDct.idx(), 0);
        assert_eq!(TxType::AdstAdst.idx(), 3);
    }

    #[test]
    fn tx_mode_repr() {
        assert_eq!(TxMode::OnlyTx4x4 as u8, 0);
        assert_eq!(TxMode::TxModeSelect as u8, 4);
    }
}
