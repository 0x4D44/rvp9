//! Encoder-side per-TX-block coefficient feed for the inter-frame
//! writer.
//!
//! The writer's residual path (S4.16g phase 3) walks each non-skip
//! inter leaf's per-plane × per-TX grid in the same order the decoder
//! does (`rvp9_decoder::reconstruct_inter::reconstruct_inter_block`'s
//! `loff` / `aoff` loop). At each TX position the writer needs the
//! quantised, pre-dequant coefficient buffer plus the `eob` to pass
//! through [`crate::entropy::write_block_tokens`].
//!
//! Today the encoder doesn't yet have a residual path of its own, so
//! the per-TX coefs come from an [`InterCoefSource`] supplied by the
//! caller. This decouples the writer's bookkeeping (TX-walk math,
//! plane addressing, coef-context setup) from "where do the
//! coefficients actually come from" — the eventual RDO/quantise path
//! lands as another implementation of this trait, alongside today's
//! test-only [`AllZeroCoefSource`].
//!
//! ## Coordinate convention
//!
//! [`InterCoefSource::get_block_coefs`] receives the TX block's
//! position in the *leaf-local* 4×4-sub-block frame: `(tx_x4_local,
//! tx_y4_local)` is relative to the leaf's top-left in plane-relative
//! 4×4 units. The writer absorbs the conversion to tile / SB
//! coordinates internally (those are an artefact of the
//! [`crate::coef_ctx::TileCoefCtx`] addressing scheme).

use crate::transforms::TxSize;

/// Per-TX-block coefficient feed consumed by the inter-frame writer.
///
/// Implementations supply the quantised, pre-dequant coefficient
/// buffer (in raster order — same layout the decoder writes into) plus
/// the `eob` for one transform block. The writer iterates planes 0/1/2
/// and TX positions inside each plane; the implementation is queried
/// once per TX block.
pub trait InterCoefSource {
    /// Coefficients and `eob` for one transform block.
    ///
    /// - `mi_row`, `mi_col`: tile-absolute 8×8 mi-units of the leaf's
    ///   top-left. Useful for implementations that key coefs by block
    ///   position rather than by traversal order.
    /// - `plane`: `0` / `1` / `2` for Y / U / V.
    /// - `tx_x4_local`, `tx_y4_local`: TX block's top-left in
    ///   leaf-relative plane-4×4 sub-block units (so `(0, 0)` is the
    ///   leaf's top-left in this plane, `(2, 0)` would be the second
    ///   TX block of an 8×8 leaf if the TX is 4×4, etc.).
    /// - `tx_size`: TX size at this position. The returned slice's
    ///   length must be `>= tx_size.n_coeffs()`; only the first
    ///   `tx_size.n_coeffs()` entries are read.
    ///
    /// Returns `(raw_coefs, eob)`. `raw_coefs` is raster-ordered: the
    /// k-th entry corresponds to the k-th raster position inside the
    /// TX block, **not** the k-th scan position. `eob` is the number
    /// of scan positions to encode (`0..=tx_size.n_coeffs()`).
    fn get_block_coefs(
        &self,
        mi_row: usize,
        mi_col: usize,
        plane: usize,
        tx_x4_local: usize,
        tx_y4_local: usize,
        tx_size: TxSize,
    ) -> (&[i32], usize);
}

/// All-zero coefficient source. Every TX block reports
/// `(&[0; n_coeffs], eob = 0)`, which causes the writer to emit only
/// the terminal-EOB bit for each TX block (the cheapest non-skip
/// residual path).
///
/// Used as the default plumbing in tests / call sites that want to
/// exercise the writer's coef walk without supplying real residuals,
/// and as a no-op for the existing skip=true call sites (the writer
/// bypasses the coef source entirely when `mi.skip == true`).
#[derive(Copy, Clone, Debug, Default)]
pub struct AllZeroCoefSource;

impl AllZeroCoefSource {
    /// Construct a fresh `AllZeroCoefSource`.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Backing storage large enough for the biggest VP9 TX block:
/// `Tx32x32` has `1024` coefficients.
static ALL_ZERO_COEFS: [i32; 1024] = [0; 1024];

impl InterCoefSource for AllZeroCoefSource {
    fn get_block_coefs(
        &self,
        _mi_row: usize,
        _mi_col: usize,
        _plane: usize,
        _tx_x4_local: usize,
        _tx_y4_local: usize,
        tx_size: TxSize,
    ) -> (&[i32], usize) {
        (&ALL_ZERO_COEFS[..tx_size.n_coeffs()], 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `AllZeroCoefSource` returns a zero-filled slice of the exact
    /// `tx_size.n_coeffs()` length, with `eob == 0`, for every TX size.
    #[test]
    fn all_zero_coef_source_returns_zero_slice_per_tx_size() {
        let src = AllZeroCoefSource::new();
        for tx in [TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx32x32] {
            let (coefs, eob) = src.get_block_coefs(0, 0, 0, 0, 0, tx);
            assert_eq!(coefs.len(), tx.n_coeffs(), "len matches n_coeffs for {tx:?}");
            assert!(coefs.iter().all(|&v| v == 0), "all zero for {tx:?}");
            assert_eq!(eob, 0, "eob = 0 for {tx:?}");
        }
    }
}
