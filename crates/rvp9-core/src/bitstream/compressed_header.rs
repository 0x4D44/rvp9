//! VP9 compressed-header parser sub-helpers.
//!
//! The compressed header — also called the "first partition" — is the
//! variable-length boolean-coder-encoded section that follows the
//! uncompressed header. It contains per-frame updates to the
//! probability tables in the active `FrameContext`. Mirrors libvpx
//! `read_compressed_header` (`vp9/decoder/vp9_decodeframe.c:2863-2911`).
//!
//! ## Sub-parsers so far
//!
//! - [`read_tx_mode`] — reads the 2-3 bit `tx_mode` field. libvpx
//!   `vp9_decodeframe.c:67-71`.
//! - [`read_tx_mode_probs`] — diff-updates the
//!   [`DefaultTxProbs`] struct. libvpx `vp9_decodeframe.c:73-87`.
//! - [`read_skip_probs`] — diff-updates the per-context skip
//!   probabilities. libvpx `vp9_decodeframe.c:2880-2881`.
//! - [`read_coef_probs`] — diff-updates the per-tx-size coefficient
//!   probability tables. libvpx `vp9_decodeframe.c:1312-1330`.
//!   This is the largest sub-parser by call count: per TX size, up
//!   to `PLANE_TYPES * REF_TYPES * (3 + 5 * COEFF_CONTEXTS) *
//!   UNCONSTRAINED_NODES = 396` diff_update_prob slots, each
//!   gated by a 1-bit per-TX-size update flag.
//!
//! ## Deferred
//!
//! - Inter-frame sub-parsers: `read_inter_mode_probs`,
//!   `read_switchable_interp_probs`, `intra_inter_prob`,
//!   `comp_inter_prob`, `single_ref_prob`, `comp_ref_prob`,
//!   `y_mode_prob`, `partition_prob` updates, MV-prob updates.
//!   Task 1e.4.

use super::error::BitstreamError;
use crate::entropy::{
    BoolReader, COEF_BANDS, COEFF_CONTEXTS, CoeffProbsModel, DefaultTxProbs, PLANE_TYPES,
    REF_TYPES, SKIP_CONTEXTS, TX_SIZES, UNCONSTRAINED_NODES, diff_update_prob,
};
use crate::transforms::TxMode;

/// libvpx `TX_SIZE_CONTEXTS = 2` (`vp9_entropymode.h:25`).
pub const TX_SIZE_CONTEXTS: usize = 2;

/// libvpx `tx_mode_to_biggest_tx_size[TX_MODES]`
/// (`vp9_common_data.c:102-108`). Maps the frame-level `TxMode`
/// to the largest TX size whose `coef_probs` may be updated by
/// the compressed-header `read_coef_probs` call.
#[inline]
#[must_use]
pub const fn tx_mode_to_biggest_tx_size(tx_mode: TxMode) -> usize {
    match tx_mode {
        TxMode::OnlyTx4x4 => 0,
        TxMode::AllowTx8x8 => 1,
        TxMode::AllowTx16x16 => 2,
        TxMode::AllowTx32x32 => 3,
        TxMode::TxModeSelect => 3,
    }
}

/// Number of coefficient-context slots in a coefficient band.
/// Band 0 uses only 3 of `COEFF_CONTEXTS`; bands 1..5 use all 6.
/// Mirrors libvpx `BAND_COEFF_CONTEXTS(band)` (`vp9_entropy.h:108`).
#[inline]
#[must_use]
const fn band_coeff_contexts(band: usize) -> usize {
    if band == 0 { 3 } else { COEFF_CONTEXTS }
}

/// Read the `tx_mode` field from the boolean decoder. Mirrors libvpx
/// `read_tx_mode` (`vp9/decoder/vp9_decodeframe.c:67-71`).
///
/// Wire format:
/// - 2 bits literal in `[0, 3]` covering `OnlyTx4x4 / AllowTx8x8 /
///   AllowTx16x16 / AllowTx32x32`.
/// - If the value is `3` (`ALLOW_32X32`), 1 extra bit promotes to
///   `4` (`TxModeSelect`).
///
/// When `lossless == true`, the field is NOT in the bitstream and
/// the caller forces the result to `TxMode::OnlyTx4x4` per
/// `vp9_decodeframe.c:2876`. This function unconditionally reads
/// from the bitstream; the lossless gate is the caller's
/// responsibility.
pub fn read_tx_mode(r: &mut BoolReader<'_>) -> TxMode {
    let mut v = r.read_literal(2) as u8;
    if v == 3 {
        // Promote to TxModeSelect (=4) iff one more bit is set.
        v += r.read_bit();
    }
    match v {
        0 => TxMode::OnlyTx4x4,
        1 => TxMode::AllowTx8x8,
        2 => TxMode::AllowTx16x16,
        3 => TxMode::AllowTx32x32,
        // v can be 4 only when the original was 3 AND the extra bit was 1.
        _ => TxMode::TxModeSelect,
    }
}

/// Diff-update the per-context tx-size probabilities. Mirrors libvpx
/// `read_tx_mode_probs` (`vp9/decoder/vp9_decodeframe.c:73-87`).
///
/// Loop layout (in this order):
/// - For each of the 2 TX_SIZE_CONTEXTS:
///   - 1 diff_update_prob slot for `p8x8[ctx][0]`.
/// - For each of the 2 TX_SIZE_CONTEXTS:
///   - 2 diff_update_prob slots for `p16x16[ctx][j]`, j in 0..2.
/// - For each of the 2 TX_SIZE_CONTEXTS:
///   - 3 diff_update_prob slots for `p32x32[ctx][j]`, j in 0..3.
///
/// Total: 2 + 4 + 6 = 12 diff_update_prob calls.
///
/// Only invoked when `tx_mode == TxMode::TxModeSelect`
/// (`vp9_decodeframe.c:2877`).
///
/// # Errors
///
/// None — the boolean reader can produce arbitrary bytes; if the
/// underlying buffer is truncated, [`BoolReader`] silently returns 0
/// per the libvpx convention.
pub fn read_tx_mode_probs(
    r: &mut BoolReader<'_>,
    probs: &mut DefaultTxProbs,
) -> Result<(), BitstreamError> {
    for ctx in 0..TX_SIZE_CONTEXTS {
        for j in 0..1 {
            diff_update_prob(r, &mut probs.p8x8[ctx][j]);
        }
    }
    for ctx in 0..TX_SIZE_CONTEXTS {
        for j in 0..2 {
            diff_update_prob(r, &mut probs.p16x16[ctx][j]);
        }
    }
    for ctx in 0..TX_SIZE_CONTEXTS {
        for j in 0..3 {
            diff_update_prob(r, &mut probs.p32x32[ctx][j]);
        }
    }
    Ok(())
}

/// Diff-update the 3 per-context skip-flag probabilities. Mirrors
/// libvpx `vp9_decodeframe.c:2880-2881`.
pub fn read_skip_probs(
    r: &mut BoolReader<'_>,
    probs: &mut [u8; SKIP_CONTEXTS],
) -> Result<(), BitstreamError> {
    for slot in probs.iter_mut() {
        diff_update_prob(r, slot);
    }
    Ok(())
}

/// Diff-update the coefficient probability tables for every TX size
/// `[TX_4X4 .. max_tx_size(tx_mode)]`. Mirrors libvpx
/// `read_coef_probs` (`vp9_decodeframe.c:1325-1330`) and the
/// per-TX-size helper `read_coef_probs_common`
/// (`vp9_decodeframe.c:1312-1323`).
///
/// For each visited TX size the wire format is:
///
/// 1. One bit at probability `128` ("update_probs"). If 0, skip this
///    TX size entirely; the table stays at its prior value.
/// 2. If 1, walk every probability slot in
///    `[PLANE_TYPES][REF_TYPES][COEF_BANDS][BAND_COEFF_CONTEXTS(band)]
///    [UNCONSTRAINED_NODES]` order and call [`diff_update_prob`].
///    Band 0 uses only 3 of `COEFF_CONTEXTS`; bands 1..5 use all 6.
///
/// Total slots per TX size: `2 * 2 * (3 + 5 * 6) * 3 = 396`.
///
/// Lossless frames force `tx_mode = OnlyTx4x4`, so only the TX_4×4
/// table is visited (`vp9_decodeframe.c:2876-2877`); that gate is the
/// caller's responsibility.
///
/// # Errors
///
/// None — same convention as the other sub-parsers; the underlying
/// [`BoolReader`] silently returns 0 if the bitstream is exhausted.
pub fn read_coef_probs(
    r: &mut BoolReader<'_>,
    coef_probs: &mut [CoeffProbsModel; TX_SIZES],
    tx_mode: TxMode,
) -> Result<(), BitstreamError> {
    let max_tx_size = tx_mode_to_biggest_tx_size(tx_mode);
    for table in coef_probs.iter_mut().take(max_tx_size + 1) {
        read_coef_probs_common(r, table);
    }
    Ok(())
}

/// Per-TX-size body of [`read_coef_probs`]. Mirrors libvpx
/// `read_coef_probs_common` (`vp9_decodeframe.c:1312-1323`).
fn read_coef_probs_common(r: &mut BoolReader<'_>, coef_probs: &mut CoeffProbsModel) {
    if r.read_bit() == 0 {
        return;
    }
    for plane in coef_probs.iter_mut().take(PLANE_TYPES) {
        for ref_type in plane.iter_mut().take(REF_TYPES) {
            for (band_idx, band) in ref_type.iter_mut().take(COEF_BANDS).enumerate() {
                let n_contexts = band_coeff_contexts(band_idx);
                for ctx in band.iter_mut().take(n_contexts) {
                    for slot in ctx.iter_mut().take(UNCONSTRAINED_NODES) {
                        diff_update_prob(r, slot);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::{BoolWriter, DIFF_UPDATE_PROB};

    fn build_bitstream<F>(f: F) -> Vec<u8>
    where
        F: FnOnce(&mut BoolWriter),
    {
        let mut buf = vec![0u8; 256];
        let mut w = BoolWriter::start_encode(&mut buf);
        f(&mut w);
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);
        buf
    }

    /// `read_tx_mode` decodes the four 2-bit values directly, and
    /// extends to TxModeSelect via the extra bit. Hand-verified
    /// against libvpx.
    #[test]
    fn read_tx_mode_round_trip_all_5_values() {
        // For each TxMode value, write the wire format and verify
        // the read returns the same value.
        for (raw_lower, extra_bit, expected) in [
            (0u32, None, TxMode::OnlyTx4x4),
            (1, None, TxMode::AllowTx8x8),
            (2, None, TxMode::AllowTx16x16),
            (3, Some(0), TxMode::AllowTx32x32),
            (3, Some(1), TxMode::TxModeSelect),
        ] {
            let buf = build_bitstream(|w| {
                w.write_literal(raw_lower, 2);
                if let Some(b) = extra_bit {
                    w.write_bit(b);
                }
                // Pad.
                for _ in 0..16 {
                    w.write_bit(0);
                }
            });
            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_tx_mode(&mut r);
            assert_eq!(got, expected, "raw={raw_lower}, extra={extra_bit:?}");
        }
    }

    /// `read_tx_mode_probs` with all bits = 0 (no-update for every
    /// slot) leaves the input table unchanged.
    #[test]
    fn read_tx_mode_probs_no_update_when_all_zero_bits() {
        let buf = build_bitstream(|w| {
            // 12 slots, each gated by a 1-bit DIFF_UPDATE_PROB check.
            // Writing 0 at threshold 252 means "no update".
            for _ in 0..12 {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut probs = DefaultTxProbs {
            p32x32: [[3, 136, 37], [5, 52, 13]],
            p16x16: [[20, 152], [15, 101]],
            p8x8: [[100], [66]],
        };
        let original = probs;
        read_tx_mode_probs(&mut r, &mut probs).expect("ok");
        assert_eq!(probs, original);
    }

    /// `read_skip_probs` no-update path leaves the input unchanged.
    #[test]
    fn read_skip_probs_no_update() {
        let buf = build_bitstream(|w| {
            for _ in 0..SKIP_CONTEXTS {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut probs = [192u8, 128, 64]; // libvpx defaults
        read_skip_probs(&mut r, &mut probs).expect("ok");
        assert_eq!(probs, [192, 128, 64]);
    }

    /// `read_skip_probs` update path: one slot gets a new value.
    /// We write update=1 for slot 1 only, with delp=0 (4-bit literal
    /// path). For p=128, INV_MAP_TABLE[0]=7, recenter: (7&1)=1 →
    /// 127 - 4 = 123, +1 = 124. So slot 1 becomes 124.
    #[test]
    fn read_skip_probs_partial_update() {
        let buf = build_bitstream(|w| {
            w.write(0, DIFF_UPDATE_PROB); // slot 0 no-update
            w.write(1, DIFF_UPDATE_PROB); // slot 1 update
            w.write_bit(0); // term_subexp prefix → 4-bit literal
            w.write_literal(0, 4); // delp = 0
            w.write(0, DIFF_UPDATE_PROB); // slot 2 no-update
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut probs = [192u8, 128, 64];
        read_skip_probs(&mut r, &mut probs).expect("ok");
        assert_eq!(probs[0], 192);
        assert_eq!(probs[1], 124, "slot 1 should be remapped to 124");
        assert_eq!(probs[2], 64);
    }

    /// `TX_SIZE_CONTEXTS` matches libvpx (`= 2`).
    #[test]
    fn tx_size_contexts_constant() {
        assert_eq!(TX_SIZE_CONTEXTS, 2);
    }

    /// `tx_mode_to_biggest_tx_size` pins every libvpx mapping.
    /// Source: `vp9_common_data.c:102-108`.
    #[test]
    fn tx_mode_to_biggest_tx_size_mapping() {
        assert_eq!(tx_mode_to_biggest_tx_size(TxMode::OnlyTx4x4), 0);
        assert_eq!(tx_mode_to_biggest_tx_size(TxMode::AllowTx8x8), 1);
        assert_eq!(tx_mode_to_biggest_tx_size(TxMode::AllowTx16x16), 2);
        assert_eq!(tx_mode_to_biggest_tx_size(TxMode::AllowTx32x32), 3);
        assert_eq!(tx_mode_to_biggest_tx_size(TxMode::TxModeSelect), 3);
    }

    /// `band_coeff_contexts(0) = 3`; bands 1..5 = `COEFF_CONTEXTS = 6`.
    #[test]
    fn band_coeff_contexts_pins_libvpx() {
        assert_eq!(band_coeff_contexts(0), 3);
        for k in 1..COEF_BANDS {
            assert_eq!(band_coeff_contexts(k), COEFF_CONTEXTS);
        }
    }

    /// `read_coef_probs` with `OnlyTx4x4` only consumes the TX_4×4
    /// update-gate bit (1 bit). When that gate is 0, the TX_4×4 table
    /// is untouched and the larger TX-size tables are not visited at
    /// all (verified by leaving them at distinct sentinel values and
    /// checking they still match).
    #[test]
    fn read_coef_probs_only_4x4_no_update() {
        use crate::entropy::{
            DEFAULT_COEF_PROBS_4X4, DEFAULT_COEF_PROBS_8X8, DEFAULT_COEF_PROBS_16X16,
            DEFAULT_COEF_PROBS_32X32,
        };

        let buf = build_bitstream(|w| {
            w.write_bit(0); // TX_4×4 update flag = 0 → skip
            // Pad: trailing bytes are irrelevant since OnlyTx4x4
            // only visits TX_4×4 and the gate-0 path consumes one bit.
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut tables = [
            DEFAULT_COEF_PROBS_4X4,
            DEFAULT_COEF_PROBS_8X8,
            DEFAULT_COEF_PROBS_16X16,
            DEFAULT_COEF_PROBS_32X32,
        ];
        let original = tables;
        read_coef_probs(&mut r, &mut tables, TxMode::OnlyTx4x4).expect("ok");
        assert_eq!(tables, original);
    }

    /// `read_coef_probs` with `TxModeSelect` and every update-gate
    /// bit = 0 visits all four TX sizes but leaves every table
    /// unchanged. Pins that the loop iterates `TX_SIZES = 4` times
    /// and that the per-TX-size short-circuit works correctly.
    #[test]
    fn read_coef_probs_tx_mode_select_all_gates_zero() {
        use crate::entropy::{
            DEFAULT_COEF_PROBS_4X4, DEFAULT_COEF_PROBS_8X8, DEFAULT_COEF_PROBS_16X16,
            DEFAULT_COEF_PROBS_32X32,
        };

        let buf = build_bitstream(|w| {
            // 4 gate bits, all zero.
            for _ in 0..TX_SIZES {
                w.write_bit(0);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut tables = [
            DEFAULT_COEF_PROBS_4X4,
            DEFAULT_COEF_PROBS_8X8,
            DEFAULT_COEF_PROBS_16X16,
            DEFAULT_COEF_PROBS_32X32,
        ];
        let original = tables;
        read_coef_probs(&mut r, &mut tables, TxMode::TxModeSelect).expect("ok");
        assert_eq!(tables, original);
    }

    /// `read_coef_probs` with the TX_4×4 update flag set but every
    /// slot's update gate = 0: 1 gate bit + 396 slot-gate bits, none
    /// of which actually update. Verifies that the inner loop walks
    /// the right shape (`PLANE_TYPES × REF_TYPES × (3 + 5·6) ×
    /// UNCONSTRAINED_NODES = 396` slot gates).
    #[test]
    fn read_coef_probs_4x4_gate_set_but_all_slots_skip() {
        use crate::entropy::{
            DEFAULT_COEF_PROBS_4X4, DEFAULT_COEF_PROBS_8X8, DEFAULT_COEF_PROBS_16X16,
            DEFAULT_COEF_PROBS_32X32, DIFF_UPDATE_PROB,
        };

        // Total slot count: 2 * 2 * (3 + 5*6) * 3 = 396 per TX size.
        const N_SLOTS_PER_TX: usize = PLANE_TYPES
            * REF_TYPES
            * (3 + 5 * COEFF_CONTEXTS)
            * UNCONSTRAINED_NODES;

        let buf = build_bitstream(|w| {
            w.write_bit(1); // TX_4×4 update flag = 1
            for _ in 0..N_SLOTS_PER_TX {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut tables = [
            DEFAULT_COEF_PROBS_4X4,
            DEFAULT_COEF_PROBS_8X8,
            DEFAULT_COEF_PROBS_16X16,
            DEFAULT_COEF_PROBS_32X32,
        ];
        let original = tables;
        read_coef_probs(&mut r, &mut tables, TxMode::OnlyTx4x4).expect("ok");
        assert_eq!(tables, original, "no slots should change");

        // Sanity: 396 is the count we expect.
        assert_eq!(N_SLOTS_PER_TX, 396);
    }

    /// `read_coef_probs` with the TX_4×4 update flag set and the
    /// very first slot ([Y][Intra][band 0][ctx 0][node 0] = 195)
    /// receiving a `delp = 0` update. Hand-trace:
    /// - `INV_MAP_TABLE[0] = 7`
    /// - `m = 195 - 1 = 194`; `(m << 1) = 388 > 255`, take else
    /// - `inv_recenter_nonneg(7, 255 - 1 - 194 = 60)`: odd path
    ///   `60 - ((7+1)>>1) = 60 - 4 = 56`
    /// - new = `255 - 56 = 199`
    ///
    /// So `tables[0][0][0][0][0][0]` becomes `199`; every other
    /// slot stays at its default. The remaining 395 slot gates are 0.
    #[test]
    fn read_coef_probs_4x4_first_slot_updated() {
        use crate::entropy::{
            DEFAULT_COEF_PROBS_4X4, DEFAULT_COEF_PROBS_8X8, DEFAULT_COEF_PROBS_16X16,
            DEFAULT_COEF_PROBS_32X32, DIFF_UPDATE_PROB,
        };

        // Sanity: pin the initial value the trace depends on.
        assert_eq!(DEFAULT_COEF_PROBS_4X4[0][0][0][0][0], 195);

        const N_SLOTS_PER_TX: usize = PLANE_TYPES
            * REF_TYPES
            * (3 + 5 * COEFF_CONTEXTS)
            * UNCONSTRAINED_NODES;

        let buf = build_bitstream(|w| {
            w.write_bit(1); // TX_4×4 update flag = 1
            // Slot 0: update, delp = 0.
            w.write(1, DIFF_UPDATE_PROB);
            w.write_bit(0);
            w.write_literal(0, 4);
            // Remaining 395 slots: no update.
            for _ in 1..N_SLOTS_PER_TX {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut tables = [
            DEFAULT_COEF_PROBS_4X4,
            DEFAULT_COEF_PROBS_8X8,
            DEFAULT_COEF_PROBS_16X16,
            DEFAULT_COEF_PROBS_32X32,
        ];
        read_coef_probs(&mut r, &mut tables, TxMode::OnlyTx4x4).expect("ok");

        // Slot 0 of the TX_4×4 table changed.
        assert_eq!(tables[0][0][0][0][0][0], 199);
        // Everything else in the TX_4×4 table is unchanged.
        let mut expected_4x4 = DEFAULT_COEF_PROBS_4X4;
        expected_4x4[0][0][0][0][0] = 199;
        assert_eq!(tables[0], expected_4x4);
        // TX_8×8 / 16×16 / 32×32 untouched (not visited).
        assert_eq!(tables[1], DEFAULT_COEF_PROBS_8X8);
        assert_eq!(tables[2], DEFAULT_COEF_PROBS_16X16);
        assert_eq!(tables[3], DEFAULT_COEF_PROBS_32X32);
    }
}
