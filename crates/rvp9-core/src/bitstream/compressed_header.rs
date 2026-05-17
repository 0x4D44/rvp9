//! VP9 compressed-header parser sub-helpers.
//!
//! The compressed header — also called the "first partition" — is the
//! variable-length boolean-coder-encoded section that follows the
//! uncompressed header. It contains per-frame updates to the
//! probability tables in the active `FrameContext`. Mirrors libvpx
//! `read_compressed_header` (`vp9/decoder/vp9_decodeframe.c:2863-2911`).
//!
//! ## Scope of this commit (Task 1e.2)
//!
//! Lands the simple scalar-prob sub-parsers:
//!
//! - [`read_tx_mode`] — reads the 2-3 bit `tx_mode` field. libvpx
//!   `vp9_decodeframe.c:67-71`.
//! - [`read_tx_mode_probs`] — diff-updates the
//!   [`DefaultTxProbs`] struct. libvpx `vp9_decodeframe.c:73-87`.
//! - [`read_skip_probs`] — diff-updates the per-context skip
//!   probabilities. libvpx `vp9_decodeframe.c:2880-2881`.
//!
//! ## Deferred
//!
//! - `read_coef_probs` — the largest sub-parser by far
//!   (4 TX_SIZES × 2 plane_types × 2 ref_types × 6 bands × 6 contexts
//!   × 3 unconstrained nodes, gated by a per-TX-size update bit).
//!   Task 1e.3.
//! - Inter-frame sub-parsers: `read_inter_mode_probs`,
//!   `read_switchable_interp_probs`, `intra_inter_prob`,
//!   `comp_inter_prob`, `single_ref_prob`, `comp_ref_prob`,
//!   `y_mode_prob`, `partition_prob` updates, MV-prob updates.
//!   Task 1e.4.

use super::error::BitstreamError;
use crate::entropy::{BoolReader, DefaultTxProbs, SKIP_CONTEXTS, diff_update_prob};
use crate::transforms::TxMode;

/// libvpx `TX_SIZE_CONTEXTS = 2` (`vp9_entropymode.h:25`).
pub const TX_SIZE_CONTEXTS: usize = 2;

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
}
