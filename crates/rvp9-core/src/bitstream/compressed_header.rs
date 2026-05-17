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
//! ## Inter-frame sub-parsers (Task 1e.4)
//!
//! Mirroring `vp9_decodeframe.c:2883-2910`:
//!
//! - [`read_inter_mode_probs`] / [`read_switchable_interp_probs`] /
//!   [`read_intra_inter_probs`] — diff-update small per-context
//!   prob tables.
//! - [`read_frame_reference_mode`] — 0/1/2-bit field that selects
//!   `ReferenceMode`, gated by whether compound reference is
//!   legal for this frame's reference set.
//! - [`read_frame_reference_mode_probs`] — diff-updates
//!   `comp_inter_prob` / `single_ref_prob` / `comp_ref_prob`
//!   depending on the selected reference mode.
//! - [`read_y_mode_probs`] / [`read_partition_probs`].
//! - [`read_mv_probs`] / [`update_mv_probs`] — note: MV probs use
//!   a different wire format ("update bit at `MV_UPDATE_PROB`,
//!   then a 7-bit literal forced odd") and **not** `diff_update_prob`.
//!
//! ## Top-level driver (Task 1e.5)
//!
//! - [`parse_compressed_header`] composes every sub-parser above
//!   in the order libvpx `read_compressed_header`
//!   (`vp9_decodeframe.c:2863-2911`) does. Takes the
//!   compressed-header byte slice + the active [`FrameContext`] +
//!   a [`CompressedHeaderOptions`] descriptor; returns
//!   `(tx_mode, reference_mode)` as a [`CompressedHeaderResult`].

use super::error::BitstreamError;
use super::frame::{InterpFilter, ReferenceMode};
use crate::entropy::{
    BLOCK_SIZE_GROUPS, BoolReader, CLASS0_SIZE, COEF_BANDS, COEFF_CONTEXTS, COMP_INTER_CONTEXTS,
    CoeffProbsModel, DefaultTxProbs, FrameContext, INTER_MODE_CONTEXTS, INTER_MODES,
    INTRA_INTER_CONTEXTS, INTRA_MODES, MV_CLASSES, MV_FP_SIZE, MV_JOINTS, MV_OFFSET_BITS,
    NmvContext, PARTITION_CONTEXTS, PARTITION_TYPES, PLANE_TYPES, REF_CONTEXTS, REF_TYPES,
    SKIP_CONTEXTS, SWITCHABLE_FILTER_CONTEXTS, SWITCHABLE_FILTERS, TX_SIZES, UNCONSTRAINED_NODES,
    diff_update_prob,
};
use crate::transforms::TxMode;

/// Per-frame state the compressed-header parser needs from the
/// uncompressed-header parser. All five fields are derived from
/// values the uncompressed header decodes.
#[derive(Copy, Clone, Debug)]
pub struct CompressedHeaderOptions {
    /// Lossless mode (`base_qindex == 0 && y_dc_delta_q == 0 &&
    /// uv_dc_delta_q == 0 && uv_ac_delta_q == 0`). When set,
    /// `tx_mode` is forced to `OnlyTx4x4` (the field is not read
    /// from the stream).
    pub lossless: bool,
    /// True for keyframes and intra-only frames. When set, the
    /// inter-frame branch of `read_compressed_header` is skipped
    /// entirely. Mirrors libvpx `frame_is_intra_only`.
    pub intra_only: bool,
    /// Interp filter from the uncompressed header. The
    /// switchable-interp-probs sub-parser is only invoked when
    /// this is [`InterpFilter::Switchable`].
    pub interp_filter: InterpFilter,
    /// True iff the active reference frames allow compound
    /// reference (at least two go in opposite time directions).
    /// Mirrors libvpx `vp9_compound_reference_allowed`.
    pub compound_allowed: bool,
    /// `allow_high_precision_mv` from the uncompressed header.
    pub allow_high_precision_mv: bool,
}

/// Result of [`parse_compressed_header`] — the two frame-scope
/// values that the per-block decoder needs.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CompressedHeaderResult {
    /// `cm->tx_mode` (see [`TxMode`]).
    pub tx_mode: TxMode,
    /// `cm->reference_mode` (see [`ReferenceMode`]). Always
    /// [`ReferenceMode::Single`] for intra-only frames (the
    /// inter branch never runs).
    pub reference_mode: ReferenceMode,
}

/// libvpx `MV_UPDATE_PROB` (`vp9_entropymv.h:35`). The threshold the
/// boolean coder uses for each MV-prob's update gate bit. Numerically
/// equal to `DIFF_UPDATE_PROB`, but the wire format that follows the
/// gate is different (a 7-bit literal forced odd, not term-subexp).
pub const MV_UPDATE_PROB: u8 = 252;

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

/// Diff-update the inter-mode tree probabilities. Mirrors libvpx
/// `read_inter_mode_probs` (`vp9_decodeframe.c:96-101`).
///
/// `INTER_MODE_CONTEXTS × (INTER_MODES - 1) = 7 × 3 = 21` slots.
pub fn read_inter_mode_probs(
    r: &mut BoolReader<'_>,
    probs: &mut [[u8; INTER_MODES - 1]; INTER_MODE_CONTEXTS],
) -> Result<(), BitstreamError> {
    for ctx in probs.iter_mut() {
        for slot in ctx.iter_mut() {
            diff_update_prob(r, slot);
        }
    }
    Ok(())
}

/// Diff-update the switchable interp-filter probabilities. Mirrors
/// libvpx `read_switchable_interp_probs` (`vp9_decodeframe.c:89-94`).
///
/// `SWITCHABLE_FILTER_CONTEXTS × (SWITCHABLE_FILTERS - 1) =
/// 4 × 2 = 8` slots. Only called when `interp_filter == Switchable`
/// (`vp9_decodeframe.c:2889`); the caller's `InterpFilter` gate
/// remains the caller's responsibility.
pub fn read_switchable_interp_probs(
    r: &mut BoolReader<'_>,
    probs: &mut [[u8; SWITCHABLE_FILTERS - 1]; SWITCHABLE_FILTER_CONTEXTS],
) -> Result<(), BitstreamError> {
    for ctx in probs.iter_mut() {
        for slot in ctx.iter_mut() {
            diff_update_prob(r, slot);
        }
    }
    Ok(())
}

/// Diff-update the intra/inter classification probabilities. Mirrors
/// libvpx `vp9_decodeframe.c:2891-2892`.
pub fn read_intra_inter_probs(
    r: &mut BoolReader<'_>,
    probs: &mut [u8; INTRA_INTER_CONTEXTS],
) -> Result<(), BitstreamError> {
    for slot in probs.iter_mut() {
        diff_update_prob(r, slot);
    }
    Ok(())
}

/// Read the frame-level `reference_mode`. Mirrors libvpx
/// `read_frame_reference_mode` (`vp9_decodeframe.c:103-112`).
///
/// Wire format (when `compound_allowed == true`):
/// - 1 bit: 0 ⇒ [`ReferenceMode::Single`].
/// - 1 bit (read iff first was 1): 0 ⇒ [`ReferenceMode::Compound`],
///   1 ⇒ [`ReferenceMode::Select`].
///
/// When `compound_allowed == false` the field is not in the stream
/// and the result is forced to [`ReferenceMode::Single`].
///
/// The caller computes `compound_allowed` from the active reference
/// frames (at least two go in opposite time directions from the
/// current frame); see libvpx `vp9_compound_reference_allowed`.
pub fn read_frame_reference_mode(
    r: &mut BoolReader<'_>,
    compound_allowed: bool,
) -> ReferenceMode {
    if !compound_allowed {
        return ReferenceMode::Single;
    }
    if r.read_bit() == 0 {
        ReferenceMode::Single
    } else if r.read_bit() == 0 {
        ReferenceMode::Compound
    } else {
        ReferenceMode::Select
    }
}

/// Diff-update the reference-mode-related probability tables. Mirrors
/// libvpx `read_frame_reference_mode_probs` (`vp9_decodeframe.c:114-131`).
///
/// Updates are gated on the selected [`ReferenceMode`]:
/// - [`ReferenceMode::Select`] visits `comp_inter_prob`.
/// - Non-[`ReferenceMode::Compound`] visits `single_ref_prob` (both
///   slots per ref context).
/// - Non-[`ReferenceMode::Single`] visits `comp_ref_prob`.
pub fn read_frame_reference_mode_probs(
    r: &mut BoolReader<'_>,
    reference_mode: ReferenceMode,
    comp_inter_prob: &mut [u8; COMP_INTER_CONTEXTS],
    single_ref_prob: &mut [[u8; 2]; REF_CONTEXTS],
    comp_ref_prob: &mut [u8; REF_CONTEXTS],
) -> Result<(), BitstreamError> {
    if reference_mode == ReferenceMode::Select {
        for slot in comp_inter_prob.iter_mut() {
            diff_update_prob(r, slot);
        }
    }
    if reference_mode != ReferenceMode::Compound {
        for pair in single_ref_prob.iter_mut() {
            diff_update_prob(r, &mut pair[0]);
            diff_update_prob(r, &mut pair[1]);
        }
    }
    if reference_mode != ReferenceMode::Single {
        for slot in comp_ref_prob.iter_mut() {
            diff_update_prob(r, slot);
        }
    }
    Ok(())
}

/// Diff-update the intra-Y-mode-given-block-size probabilities.
/// Mirrors libvpx `vp9_decodeframe.c:2899-2901`.
///
/// `BLOCK_SIZE_GROUPS × (INTRA_MODES - 1) = 4 × 9 = 36` slots.
pub fn read_y_mode_probs(
    r: &mut BoolReader<'_>,
    probs: &mut [[u8; INTRA_MODES - 1]; BLOCK_SIZE_GROUPS],
) -> Result<(), BitstreamError> {
    for group in probs.iter_mut() {
        for slot in group.iter_mut() {
            diff_update_prob(r, slot);
        }
    }
    Ok(())
}

/// Diff-update the partition-tree probabilities. Mirrors libvpx
/// `vp9_decodeframe.c:2903-2905`.
///
/// `PARTITION_CONTEXTS × (PARTITION_TYPES - 1) = 16 × 3 = 48` slots.
pub fn read_partition_probs(
    r: &mut BoolReader<'_>,
    probs: &mut [[u8; PARTITION_TYPES - 1]; PARTITION_CONTEXTS],
) -> Result<(), BitstreamError> {
    for ctx in probs.iter_mut() {
        for slot in ctx.iter_mut() {
            diff_update_prob(r, slot);
        }
    }
    Ok(())
}

/// MV-prob update helper. Mirrors libvpx `update_mv_probs`
/// (`vp9_decodeframe.c:133-137`).
///
/// For each of `n` probability slots:
/// - Read 1 bit at [`MV_UPDATE_PROB`].
/// - If 0, slot stays put.
/// - If 1, read a 7-bit literal `v` and set `slot = (v << 1) | 1`
///   (forces the LSB on — MV probs are always odd in the range
///   `1, 3, 5, ..., 255`).
fn update_mv_probs(r: &mut BoolReader<'_>, probs: &mut [u8]) {
    for slot in probs.iter_mut() {
        if r.read(MV_UPDATE_PROB) != 0 {
            // v is u32 in [0, 127]; (v << 1) | 1 fits in u8 (in [1, 255]).
            let v = r.read_literal(7) as u8;
            *slot = (v << 1) | 1;
        }
    }
}

/// Update the motion-vector probability context from the boolean
/// reader. Mirrors libvpx `read_mv_probs` (`vp9_decodeframe.c:139-166`).
///
/// Walk order (must match libvpx byte-for-byte):
/// 1. `ctx.joints` (`MV_JOINTS - 1 = 3` slots).
/// 2. For each of 2 components: `sign` (1), `classes`
///    (`MV_CLASSES - 1 = 10`), `class0` (`CLASS0_SIZE - 1 = 1`),
///    `bits` (`MV_OFFSET_BITS = 10`).
/// 3. For each of 2 components: per `class0` value `j` in
///    `0..CLASS0_SIZE`, `class0_fp[j]` (`MV_FP_SIZE - 1 = 3`);
///    then `fp` (3).
/// 4. **Only if `allow_high_precision_mv`**: for each of 2
///    components, `class0_hp` (1) + `hp` (1).
pub fn read_mv_probs(
    r: &mut BoolReader<'_>,
    ctx: &mut NmvContext,
    allow_high_precision_mv: bool,
) -> Result<(), BitstreamError> {
    debug_assert_eq!(ctx.joints.len(), MV_JOINTS - 1);
    update_mv_probs(r, &mut ctx.joints);

    for comp in ctx.comps.iter_mut() {
        update_mv_probs(r, core::slice::from_mut(&mut comp.sign));
        debug_assert_eq!(comp.classes.len(), MV_CLASSES - 1);
        update_mv_probs(r, &mut comp.classes);
        debug_assert_eq!(comp.class0.len(), CLASS0_SIZE - 1);
        update_mv_probs(r, &mut comp.class0);
        debug_assert_eq!(comp.bits.len(), MV_OFFSET_BITS);
        update_mv_probs(r, &mut comp.bits);
    }

    for comp in ctx.comps.iter_mut() {
        for j in 0..CLASS0_SIZE {
            debug_assert_eq!(comp.class0_fp[j].len(), MV_FP_SIZE - 1);
            update_mv_probs(r, &mut comp.class0_fp[j]);
        }
        // libvpx hard-codes "3" here, which equals MV_FP_SIZE - 1.
        debug_assert_eq!(comp.fp.len(), MV_FP_SIZE - 1);
        update_mv_probs(r, &mut comp.fp);
    }

    if allow_high_precision_mv {
        for comp in ctx.comps.iter_mut() {
            update_mv_probs(r, core::slice::from_mut(&mut comp.class0_hp));
            update_mv_probs(r, core::slice::from_mut(&mut comp.hp));
        }
    }
    Ok(())
}

/// Parse a complete compressed-header partition over `data` and
/// apply every diff_update to `fc`. Mirrors libvpx
/// `read_compressed_header` (`vp9_decodeframe.c:2863-2911`).
///
/// Walk order (must remain byte-for-byte identical to libvpx):
///
/// 1. `tx_mode = lossless ? OnlyTx4x4 : read_tx_mode(r)`.
/// 2. If `tx_mode == TxModeSelect`, `read_tx_mode_probs(&fc.tx_probs)`.
/// 3. `read_coef_probs(&fc.coef_probs, tx_mode)`.
/// 4. `read_skip_probs(&fc.skip_probs)`.
/// 5. If `!intra_only`, in this order:
///    `read_inter_mode_probs` → (if `Switchable`)
///    `read_switchable_interp_probs` → `read_intra_inter_probs` →
///    `read_frame_reference_mode` → `read_frame_reference_mode_probs`
///    → `read_y_mode_probs` → `read_partition_probs` →
///    `read_mv_probs`.
///
/// Returns the `(tx_mode, reference_mode)` pair the per-block
/// decoder consumes.
///
/// # Errors
///
/// - [`BitstreamError::CompressedHeaderInvalidMarker`] if the
///   boolean decoder's marker-bit check fails on init.
/// - [`BitstreamError::CompressedHeaderReaderError`] if the
///   decoder ran past the end of `data` during parse.
pub fn parse_compressed_header(
    data: &[u8],
    fc: &mut FrameContext,
    opts: CompressedHeaderOptions,
) -> Result<CompressedHeaderResult, BitstreamError> {
    let mut r = BoolReader::new(data).map_err(|_| BitstreamError::CompressedHeaderInvalidMarker)?;

    // 1. tx_mode (forced to OnlyTx4x4 in lossless frames).
    let tx_mode = if opts.lossless {
        TxMode::OnlyTx4x4
    } else {
        read_tx_mode(&mut r)
    };

    // 2. tx_mode probs (only for TxModeSelect).
    if tx_mode == TxMode::TxModeSelect {
        read_tx_mode_probs(&mut r, &mut fc.tx_probs)?;
    }

    // 3. Coef probs (gated per TX size by tx_mode).
    read_coef_probs(&mut r, &mut fc.coef_probs, tx_mode)?;

    // 4. Skip probs.
    read_skip_probs(&mut r, &mut fc.skip_probs)?;

    // 5. Inter-only branch.
    let reference_mode = if opts.intra_only {
        ReferenceMode::Single
    } else {
        read_inter_mode_probs(&mut r, &mut fc.inter_mode_probs)?;

        if opts.interp_filter == InterpFilter::Switchable {
            read_switchable_interp_probs(&mut r, &mut fc.switchable_interp_prob)?;
        }

        read_intra_inter_probs(&mut r, &mut fc.intra_inter_prob)?;

        let reference_mode = read_frame_reference_mode(&mut r, opts.compound_allowed);
        read_frame_reference_mode_probs(
            &mut r,
            reference_mode,
            &mut fc.comp_inter_prob,
            &mut fc.single_ref_prob,
            &mut fc.comp_ref_prob,
        )?;

        read_y_mode_probs(&mut r, &mut fc.y_mode_prob)?;
        read_partition_probs(&mut r, &mut fc.partition_prob)?;
        read_mv_probs(&mut r, &mut fc.nmvc, opts.allow_high_precision_mv)?;

        reference_mode
    };

    if r.has_error() {
        return Err(BitstreamError::CompressedHeaderReaderError);
    }

    Ok(CompressedHeaderResult {
        tx_mode,
        reference_mode,
    })
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

    // ---- Inter-frame sub-parsers (Task 1e.4) ----

    /// `read_inter_mode_probs` walks the
    /// `INTER_MODE_CONTEXTS × (INTER_MODES - 1) = 7 × 3 = 21` slots
    /// and updates none of them when every gate bit is 0.
    #[test]
    fn read_inter_mode_probs_no_update() {
        use crate::entropy::{DEFAULT_INTER_MODE_PROBS, DIFF_UPDATE_PROB};

        const N: usize = INTER_MODE_CONTEXTS * (INTER_MODES - 1);
        assert_eq!(N, 21);
        let buf = build_bitstream(|w| {
            for _ in 0..N {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut probs = DEFAULT_INTER_MODE_PROBS;
        read_inter_mode_probs(&mut r, &mut probs).expect("ok");
        assert_eq!(probs, DEFAULT_INTER_MODE_PROBS);
    }

    /// `read_switchable_interp_probs`:
    /// `SWITCHABLE_FILTER_CONTEXTS × (SWITCHABLE_FILTERS - 1) =
    /// 4 × 2 = 8` slots; no-update path leaves the table intact.
    #[test]
    fn read_switchable_interp_probs_no_update() {
        use crate::entropy::{DEFAULT_SWITCHABLE_INTERP_PROB, DIFF_UPDATE_PROB};

        const N: usize = SWITCHABLE_FILTER_CONTEXTS * (SWITCHABLE_FILTERS - 1);
        assert_eq!(N, 8);
        let buf = build_bitstream(|w| {
            for _ in 0..N {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut probs = DEFAULT_SWITCHABLE_INTERP_PROB;
        read_switchable_interp_probs(&mut r, &mut probs).expect("ok");
        assert_eq!(probs, DEFAULT_SWITCHABLE_INTERP_PROB);
    }

    /// `read_intra_inter_probs`: `INTRA_INTER_CONTEXTS = 4` slots.
    #[test]
    fn read_intra_inter_probs_no_update() {
        use crate::entropy::{DEFAULT_INTRA_INTER_PROB, DIFF_UPDATE_PROB};

        let buf = build_bitstream(|w| {
            for _ in 0..INTRA_INTER_CONTEXTS {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut probs = DEFAULT_INTRA_INTER_PROB;
        read_intra_inter_probs(&mut r, &mut probs).expect("ok");
        assert_eq!(probs, DEFAULT_INTRA_INTER_PROB);
    }

    /// `read_frame_reference_mode` with `compound_allowed == false`
    /// reads no bits and returns `Single`.
    #[test]
    fn read_frame_reference_mode_compound_disallowed() {
        // Buffer is irrelevant (no read), but we still need a valid
        // initialised reader.
        let buf = build_bitstream(|w| {
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mode = read_frame_reference_mode(&mut r, false);
        assert_eq!(mode, ReferenceMode::Single);
    }

    /// `read_frame_reference_mode` with `compound_allowed == true`
    /// reads the wire format:
    ///   first bit 0     → Single
    ///   first bit 1, second bit 0 → Compound
    ///   first bit 1, second bit 1 → Select
    #[test]
    fn read_frame_reference_mode_wire_format() {
        for (bit0, bit1, expected) in [
            (0u8, None, ReferenceMode::Single),
            (1, Some(0u8), ReferenceMode::Compound),
            (1, Some(1), ReferenceMode::Select),
        ] {
            let buf = build_bitstream(|w| {
                w.write_bit(bit0);
                if let Some(b) = bit1 {
                    w.write_bit(b);
                }
                for _ in 0..16 {
                    w.write_bit(0);
                }
            });
            let mut r = BoolReader::new(&buf).expect("init");
            let mode = read_frame_reference_mode(&mut r, true);
            assert_eq!(mode, expected, "bit0={bit0}, bit1={bit1:?}");
        }
    }

    /// `read_frame_reference_mode_probs` with `Select` visits all
    /// three sub-tables (`comp_inter_prob`, `single_ref_prob`,
    /// `comp_ref_prob`): `COMP_INTER_CONTEXTS + REF_CONTEXTS*2 +
    /// REF_CONTEXTS` = `5 + 10 + 5 = 20` slots.
    #[test]
    fn read_frame_reference_mode_probs_select_visits_all() {
        use crate::entropy::DIFF_UPDATE_PROB;

        const N: usize = COMP_INTER_CONTEXTS + REF_CONTEXTS * 2 + REF_CONTEXTS;
        assert_eq!(N, 20);
        let buf = build_bitstream(|w| {
            for _ in 0..N {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut comp_inter = [192u8; COMP_INTER_CONTEXTS];
        let mut single_ref = [[100u8; 2]; REF_CONTEXTS];
        let mut comp_ref = [128u8; REF_CONTEXTS];
        read_frame_reference_mode_probs(
            &mut r,
            ReferenceMode::Select,
            &mut comp_inter,
            &mut single_ref,
            &mut comp_ref,
        )
        .expect("ok");
        // All three tables untouched.
        assert_eq!(comp_inter, [192u8; COMP_INTER_CONTEXTS]);
        assert_eq!(single_ref, [[100u8; 2]; REF_CONTEXTS]);
        assert_eq!(comp_ref, [128u8; REF_CONTEXTS]);
    }

    /// `read_frame_reference_mode_probs` with `Single` visits only
    /// `single_ref_prob` (`REF_CONTEXTS * 2 = 10` slots) — skipping
    /// `comp_inter_prob` (gated on Select) and `comp_ref_prob`
    /// (gated on != Single).
    #[test]
    fn read_frame_reference_mode_probs_single_visits_only_single_ref() {
        use crate::entropy::DIFF_UPDATE_PROB;

        const N: usize = REF_CONTEXTS * 2;
        let buf = build_bitstream(|w| {
            for _ in 0..N {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut comp_inter = [192u8; COMP_INTER_CONTEXTS];
        let mut single_ref = [[100u8; 2]; REF_CONTEXTS];
        let mut comp_ref = [128u8; REF_CONTEXTS];
        read_frame_reference_mode_probs(
            &mut r,
            ReferenceMode::Single,
            &mut comp_inter,
            &mut single_ref,
            &mut comp_ref,
        )
        .expect("ok");
        assert_eq!(comp_inter, [192u8; COMP_INTER_CONTEXTS]);
        assert_eq!(single_ref, [[100u8; 2]; REF_CONTEXTS]);
        assert_eq!(comp_ref, [128u8; REF_CONTEXTS]);
    }

    /// `read_y_mode_probs`:
    /// `BLOCK_SIZE_GROUPS × (INTRA_MODES - 1) = 4 × 9 = 36` slots.
    #[test]
    fn read_y_mode_probs_no_update() {
        use crate::entropy::{DEFAULT_IF_Y_PROBS, DIFF_UPDATE_PROB};

        const N: usize = BLOCK_SIZE_GROUPS * (INTRA_MODES - 1);
        assert_eq!(N, 36);
        let buf = build_bitstream(|w| {
            for _ in 0..N {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut probs = DEFAULT_IF_Y_PROBS;
        read_y_mode_probs(&mut r, &mut probs).expect("ok");
        assert_eq!(probs, DEFAULT_IF_Y_PROBS);
    }

    /// `read_partition_probs`:
    /// `PARTITION_CONTEXTS × (PARTITION_TYPES - 1) = 16 × 3 = 48`.
    #[test]
    fn read_partition_probs_no_update() {
        use crate::entropy::{DEFAULT_PARTITION_PROBS, DIFF_UPDATE_PROB};

        const N: usize = PARTITION_CONTEXTS * (PARTITION_TYPES - 1);
        assert_eq!(N, 48);
        let buf = build_bitstream(|w| {
            for _ in 0..N {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut probs = DEFAULT_PARTITION_PROBS;
        read_partition_probs(&mut r, &mut probs).expect("ok");
        assert_eq!(probs, DEFAULT_PARTITION_PROBS);
    }

    /// `read_mv_probs` no-update path with `allow_high_precision_mv =
    /// false`: every gate bit is 0 → context unchanged.
    /// Slot count (no HP): joints(3) + 2*(sign(1) + classes(10) +
    /// class0(1) + bits(10)) + 2*(CLASS0_SIZE*3 + 3) =
    /// 3 + 2*22 + 2*(2*3 + 3) = 3 + 44 + 18 = 65.
    #[test]
    fn read_mv_probs_no_update_no_hp() {
        use crate::entropy::DEFAULT_NMV_CONTEXT;

        const N: usize = (MV_JOINTS - 1)
            + 2 * (1 + (MV_CLASSES - 1) + (CLASS0_SIZE - 1) + MV_OFFSET_BITS)
            + 2 * (CLASS0_SIZE * (MV_FP_SIZE - 1) + (MV_FP_SIZE - 1));
        assert_eq!(N, 65);
        let buf = build_bitstream(|w| {
            for _ in 0..N {
                w.write(0, MV_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut ctx = DEFAULT_NMV_CONTEXT;
        read_mv_probs(&mut r, &mut ctx, false).expect("ok");
        assert_eq!(ctx, DEFAULT_NMV_CONTEXT);
    }

    /// `read_mv_probs` no-update path with `allow_high_precision_mv =
    /// true`: 65 normal slots + 2*(class0_hp + hp) = 4 more = 69 gates.
    #[test]
    fn read_mv_probs_no_update_with_hp() {
        use crate::entropy::DEFAULT_NMV_CONTEXT;

        const N: usize = (MV_JOINTS - 1)
            + 2 * (1 + (MV_CLASSES - 1) + (CLASS0_SIZE - 1) + MV_OFFSET_BITS)
            + 2 * (CLASS0_SIZE * (MV_FP_SIZE - 1) + (MV_FP_SIZE - 1))
            + 2 * (1 + 1);
        assert_eq!(N, 69);
        let buf = build_bitstream(|w| {
            for _ in 0..N {
                w.write(0, MV_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut ctx = DEFAULT_NMV_CONTEXT;
        read_mv_probs(&mut r, &mut ctx, true).expect("ok");
        assert_eq!(ctx, DEFAULT_NMV_CONTEXT);
    }

    // ---- Top-level parse_compressed_header (Task 1e.5) ----

    /// Helper: build a compressed-header byte slice the slow way —
    /// by re-writing every sub-parser's wire format via `BoolWriter`.
    /// Used by the integration tests below.
    fn build_compressed_header<F>(f: F) -> Vec<u8>
    where
        F: FnOnce(&mut BoolWriter),
    {
        let mut buf = vec![0u8; 4096];
        let mut w = BoolWriter::start_encode(&mut buf);
        f(&mut w);
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);
        buf
    }

    /// `parse_compressed_header` over a lossless intra-only frame
    /// with every sub-parser's gate bit = 0:
    /// - `tx_mode` forced to `OnlyTx4x4` (no field read).
    /// - `read_coef_probs` visits TX_4×4 only with a 0 gate bit.
    /// - `read_skip_probs` visits 3 slots with 0 gate bits.
    /// - Inter branch skipped (`intra_only == true`).
    ///
    /// Result: `(OnlyTx4x4, Single)`; `fc` unchanged.
    #[test]
    fn parse_compressed_header_lossless_intra_only_no_update() {
        use crate::entropy::{DIFF_UPDATE_PROB, FrameContext};

        let buf = build_compressed_header(|w| {
            // Coef-probs gate for TX_4×4 = 0 (skip).
            w.write_bit(0);
            // Skip probs: 3 zero gates.
            for _ in 0..SKIP_CONTEXTS {
                w.write(0, DIFF_UPDATE_PROB);
            }
            // Pad with zero bits — the BoolReader's marker bit check
            // happens at construction.
            for _ in 0..16 {
                w.write_bit(0);
            }
        });

        let mut fc = FrameContext::new_default();
        let original = fc.clone();
        let opts = CompressedHeaderOptions {
            lossless: true,
            intra_only: true,
            interp_filter: InterpFilter::EightTap, // ignored on intra
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let result = parse_compressed_header(&buf, &mut fc, opts).expect("ok");
        assert_eq!(result.tx_mode, TxMode::OnlyTx4x4);
        assert_eq!(result.reference_mode, ReferenceMode::Single);
        assert_eq!(fc, original);
    }

    /// `parse_compressed_header` over a non-lossless intra-only frame
    /// with `tx_mode = OnlyTx4x4` (the smallest non-lossless case).
    /// Pins:
    /// - 2-bit tx_mode literal = `0` (decoded to `OnlyTx4x4`).
    /// - No tx_mode_probs (only fires on `TxModeSelect`).
    /// - 1 coef-probs gate for TX_4×4.
    /// - 3 skip-probs gates.
    /// - No inter branch.
    #[test]
    fn parse_compressed_header_non_lossless_intra_only_tx4x4() {
        use crate::entropy::{DIFF_UPDATE_PROB, FrameContext};

        let buf = build_compressed_header(|w| {
            // tx_mode = 0 (OnlyTx4x4), 2-bit literal.
            w.write_literal(0, 2);
            // Coef probs gate = 0.
            w.write_bit(0);
            // Skip probs: 3 zero gates.
            for _ in 0..SKIP_CONTEXTS {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });

        let mut fc = FrameContext::new_default();
        let original = fc.clone();
        let opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: true,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let result = parse_compressed_header(&buf, &mut fc, opts).expect("ok");
        assert_eq!(result.tx_mode, TxMode::OnlyTx4x4);
        assert_eq!(result.reference_mode, ReferenceMode::Single);
        assert_eq!(fc, original);
    }

    /// `parse_compressed_header` with `tx_mode = TxModeSelect` (2-bit
    /// literal = 3 + extra bit = 1) triggers `read_tx_mode_probs`
    /// (12 extra slots). All gates zero → fc unchanged.
    #[test]
    fn parse_compressed_header_tx_mode_select_visits_tx_mode_probs() {
        use crate::entropy::{DIFF_UPDATE_PROB, FrameContext};

        let buf = build_compressed_header(|w| {
            // tx_mode = TxModeSelect (literal 3 then extra-bit 1).
            w.write_literal(3, 2);
            w.write_bit(1);
            // read_tx_mode_probs: 12 zero gates.
            for _ in 0..12 {
                w.write(0, DIFF_UPDATE_PROB);
            }
            // Coef probs gate per TX size: all zero (4 gates).
            for _ in 0..TX_SIZES {
                w.write_bit(0);
            }
            // Skip probs: 3 zero gates.
            for _ in 0..SKIP_CONTEXTS {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });

        let mut fc = FrameContext::new_default();
        let original = fc.clone();
        let opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: true,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let result = parse_compressed_header(&buf, &mut fc, opts).expect("ok");
        assert_eq!(result.tx_mode, TxMode::TxModeSelect);
        assert_eq!(result.reference_mode, ReferenceMode::Single);
        assert_eq!(fc, original);
    }

    /// `parse_compressed_header` over an inter frame with every
    /// sub-parser's gate bit = 0 and `interp_filter = Switchable`
    /// (forces the switchable-interp-probs sub-parser to run).
    /// Compound-disallowed → reference_mode = Single.
    /// allow_hp = false → no MV high-precision sub-walk.
    /// All gates 0 → fc unchanged.
    #[test]
    fn parse_compressed_header_inter_no_update_compound_disallowed() {
        use crate::entropy::{DIFF_UPDATE_PROB, FrameContext};

        // Slot counts:
        //   inter_mode_probs: 21
        //   switchable_interp_probs: 8 (interp_filter == Switchable)
        //   intra_inter_prob: 4
        //   reference_mode: 0 bits (compound_disallowed → Single)
        //   reference_mode_probs (Single): 10 (REF_CONTEXTS*2)
        //   y_mode_probs: 36
        //   partition_probs: 48
        //   mv_probs (no HP): 65
        let buf = build_compressed_header(|w| {
            // tx_mode = 0 (OnlyTx4x4).
            w.write_literal(0, 2);
            // Coef probs gate = 0.
            w.write_bit(0);
            // Skip probs.
            for _ in 0..SKIP_CONTEXTS {
                w.write(0, DIFF_UPDATE_PROB);
            }
            // Inter mode probs: 21 zero gates.
            for _ in 0..21 {
                w.write(0, DIFF_UPDATE_PROB);
            }
            // Switchable interp probs: 8 zero gates.
            for _ in 0..8 {
                w.write(0, DIFF_UPDATE_PROB);
            }
            // Intra/inter probs: 4 zero gates.
            for _ in 0..INTRA_INTER_CONTEXTS {
                w.write(0, DIFF_UPDATE_PROB);
            }
            // No reference_mode bits (compound_allowed = false).
            // reference_mode_probs (Single): single_ref only, 10 gates.
            for _ in 0..(REF_CONTEXTS * 2) {
                w.write(0, DIFF_UPDATE_PROB);
            }
            // y_mode probs: 36 zero gates.
            for _ in 0..36 {
                w.write(0, DIFF_UPDATE_PROB);
            }
            // partition probs: 48 zero gates.
            for _ in 0..48 {
                w.write(0, DIFF_UPDATE_PROB);
            }
            // MV probs: 65 zero gates (no HP).
            for _ in 0..65 {
                w.write(0, MV_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });

        let mut fc = FrameContext::new_default();
        let original = fc.clone();
        let opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: false,
            interp_filter: InterpFilter::Switchable,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let result = parse_compressed_header(&buf, &mut fc, opts).expect("ok");
        assert_eq!(result.tx_mode, TxMode::OnlyTx4x4);
        assert_eq!(result.reference_mode, ReferenceMode::Single);
        assert_eq!(fc, original);
    }

    /// `parse_compressed_header` over an inter frame with
    /// `compound_allowed = true` and the 2-bit reference-mode field
    /// = `(1, 1)` → `ReferenceMode::Select`. This forces the full
    /// reference-mode-probs walk (all 3 sub-tables: 5 + 10 + 5 = 20
    /// gates). `interp_filter = Bilinear` (not switchable) skips
    /// the switchable-interp-probs sub-parser. `allow_hp = true`
    /// triggers the 4 extra MV-HP gates (69 total).
    #[test]
    fn parse_compressed_header_inter_select_with_hp_no_update() {
        use crate::entropy::{DIFF_UPDATE_PROB, FrameContext};

        let buf = build_compressed_header(|w| {
            // tx_mode = 0.
            w.write_literal(0, 2);
            w.write_bit(0); // coef-probs gate
            for _ in 0..SKIP_CONTEXTS {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..21 {
                w.write(0, DIFF_UPDATE_PROB); // inter_mode_probs
            }
            // No switchable_interp_probs (interp_filter != Switchable)
            for _ in 0..INTRA_INTER_CONTEXTS {
                w.write(0, DIFF_UPDATE_PROB); // intra_inter
            }
            // reference_mode = Select: first bit 1, second bit 1.
            w.write_bit(1);
            w.write_bit(1);
            // reference_mode_probs (Select): visit all 3 tables.
            for _ in 0..COMP_INTER_CONTEXTS {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..(REF_CONTEXTS * 2) {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..REF_CONTEXTS {
                w.write(0, DIFF_UPDATE_PROB);
            }
            for _ in 0..36 {
                w.write(0, DIFF_UPDATE_PROB); // y_mode
            }
            for _ in 0..48 {
                w.write(0, DIFF_UPDATE_PROB); // partition
            }
            for _ in 0..69 {
                w.write(0, MV_UPDATE_PROB); // mv with HP
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });

        let mut fc = FrameContext::new_default();
        let original = fc.clone();
        let opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: false,
            interp_filter: InterpFilter::Bilinear,
            compound_allowed: true,
            allow_high_precision_mv: true,
        };
        let result = parse_compressed_header(&buf, &mut fc, opts).expect("ok");
        assert_eq!(result.tx_mode, TxMode::OnlyTx4x4);
        assert_eq!(result.reference_mode, ReferenceMode::Select);
        assert_eq!(fc, original);
    }

    /// A buffer whose first byte has its high bit set causes the
    /// bool-reader marker-bit check to fail (the first read at prob
    /// 128 returns 1, libvpx convention says marker-bit-must-be-zero).
    /// `parse_compressed_header` maps this to
    /// `CompressedHeaderInvalidMarker`.
    #[test]
    fn parse_compressed_header_bad_marker_returns_invalid_marker() {
        use crate::entropy::FrameContext;
        let mut fc = FrameContext::new_default();
        let opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: true,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        // First byte = 0xff → marker bit (first MSB) = 1.
        let err = parse_compressed_header(&[0xff, 0, 0, 0], &mut fc, opts)
            .expect_err("must fail");
        assert!(matches!(err, BitstreamError::CompressedHeaderInvalidMarker));
    }

    /// MV-prob update wire format: gate=1, literal=`v` → slot becomes
    /// `(v << 1) | 1`. Hand-check: literal=0 → 1; literal=127 → 255.
    /// We update the very first MV slot (`joints[0]`) and leave the
    /// rest as no-update.
    #[test]
    fn read_mv_probs_first_joint_update_v0_v127() {
        use crate::entropy::DEFAULT_NMV_CONTEXT;

        // Case 1: literal = 0 → slot becomes 1.
        const N: usize = (MV_JOINTS - 1)
            + 2 * (1 + (MV_CLASSES - 1) + (CLASS0_SIZE - 1) + MV_OFFSET_BITS)
            + 2 * (CLASS0_SIZE * (MV_FP_SIZE - 1) + (MV_FP_SIZE - 1));
        let buf = build_bitstream(|w| {
            w.write(1, MV_UPDATE_PROB); // joints[0] update
            w.write_literal(0, 7);
            for _ in 1..N {
                w.write(0, MV_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut ctx = DEFAULT_NMV_CONTEXT;
        read_mv_probs(&mut r, &mut ctx, false).expect("ok");
        assert_eq!(ctx.joints[0], 1);
        assert_eq!(&ctx.joints[1..], &DEFAULT_NMV_CONTEXT.joints[1..]);

        // Case 2: literal = 127 → slot becomes 255.
        let buf2 = build_bitstream(|w| {
            w.write(1, MV_UPDATE_PROB); // joints[0] update
            w.write_literal(127, 7);
            for _ in 1..N {
                w.write(0, MV_UPDATE_PROB);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r2 = BoolReader::new(&buf2).expect("init");
        let mut ctx2 = DEFAULT_NMV_CONTEXT;
        read_mv_probs(&mut r2, &mut ctx2, false).expect("ok");
        assert_eq!(ctx2.joints[0], 255);
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
