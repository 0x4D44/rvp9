//! VP9 uncompressed-header sub-parsers: loop filter, quantization,
//! segmentation, and tile info.
//!
//! These mirror the libvpx `setup_*` helpers called sequentially after
//! the frame-size / ref-frame block in `read_uncompressed_header`:
//!
//! - [`parse_loopfilter`] — `vp9/decoder/vp9_decodeframe.c:1382-1406`.
//! - [`parse_quantization`] — `vp9_decodeframe.c:1412-1425`.
//! - [`parse_segmentation`] — `vp9_decodeframe.c:1332-1380`.
//! - [`parse_tile_info`] — `vp9_decodeframe.c:1629-1645`
//!   (uses `vp9_get_tile_n_bits` from `vp9_tile_common.c:51-57`).
//!
//! These are intentionally **stateless** — they take a `RawBitReader`
//! and produce a populated parameter struct, no shared codec state.
//! The decoder will weave them into the full
//! `parse_uncompressed_header` once the inter-frame ref-loop is in
//! place (Task 1d.2 / 1d.3).

use super::error::BitstreamError;
use super::raw::{RawBitReader, RawBitWriter};

// ---------------------------------------------------------------- constants
// All constants here mirror libvpx values one-to-one. We re-declare
// them rather than import from elsewhere because (a) bitstream-layer
// constants are part of the parser's API surface, and (b) we want
// every numeric literal in this file to be one tap from a libvpx
// citation.

/// Bits used for `base_qindex` (`vp9_quant_common.h:24`).
pub const QINDEX_BITS: u8 = 8;

/// Loop-filter ref-frame delta slot count
/// (`vp9_loopfilter.h:29`).
pub const MAX_REF_LF_DELTAS: usize = 4;

/// Loop-filter mode delta slot count (`vp9_loopfilter.h:30`).
pub const MAX_MODE_LF_DELTAS: usize = 2;

/// Maximum number of segments per frame (`vp9_seg_common.h:23`).
pub const MAX_SEGMENTS: usize = 8;

/// Segment-tree probability count (`vp9_seg_common.h:24`).
pub const SEG_TREE_PROBS: usize = MAX_SEGMENTS - 1;

/// Segment-prediction probability count (`vp9_seg_common.h:26`).
pub const PREDICTION_PROBS: usize = 3;

/// Number of per-segment feature slots (`vp9_seg_common.h:39`).
/// Features 0/1 are signed; 2/3 are unsigned. See
/// [`SegFeature::is_signed`].
pub const SEG_LVL_MAX: usize = 4;

/// Maximum probability value (`prob = 255` is the "no update" sentinel).
pub const MAX_PROB: u8 = 255;

/// libvpx `MIN_TILE_WIDTH_B64` (`vp9_tile_common.c:15`).
const MIN_TILE_WIDTH_B64: i32 = 4;

/// libvpx `MAX_TILE_WIDTH_B64` (`vp9_tile_common.c:16`).
const MAX_TILE_WIDTH_B64: i32 = 64;

/// libvpx `MI_BLOCK_SIZE_LOG2` = `log2(SB_size / MI_size)` = `log2(64/8)`
/// = 3. (`vp9_enums.h:22`.)
pub const MI_BLOCK_SIZE_LOG2: i32 = 3;

// ---------------------------------------------------------------- loop filter

/// Parsed loop-filter parameters. Mirrors the `loopfilter` struct's
/// header-decoded fields (`vp9_loopfilter.h`).
#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct LoopFilterParams {
    /// Frame-level filter level (0..=63).
    pub filter_level: u8,
    /// Frame-level sharpness (0..=7).
    pub sharpness_level: u8,
    /// `true` if per-ref / per-mode deltas are in effect.
    pub mode_ref_delta_enabled: bool,
    /// `true` if this frame's header re-specified the delta values.
    /// (When false, the previous frame's deltas persist.)
    pub mode_ref_delta_update: bool,
    /// Per-ref-frame deltas (`Some(delta)` if the corresponding
    /// 1-bit "present" flag was set during this frame's header
    /// parse). `None` slots keep their previous value at the
    /// decoder level.
    pub ref_deltas: [Option<i8>; MAX_REF_LF_DELTAS],
    /// Per-mode deltas (same Some/None semantics as `ref_deltas`).
    pub mode_deltas: [Option<i8>; MAX_MODE_LF_DELTAS],
}

/// Parse the loop-filter sub-block of the uncompressed header.
/// Mirrors `setup_loopfilter` (`vp9_decodeframe.c:1382-1406`).
pub fn parse_loopfilter(rb: &mut RawBitReader<'_>) -> Result<LoopFilterParams, BitstreamError> {
    let filter_level = rb.read_literal(6)? as u8;
    let sharpness_level = rb.read_literal(3)? as u8;

    let mode_ref_delta_enabled = rb.read_bit()? != 0;
    let mut mode_ref_delta_update = false;
    let mut ref_deltas: [Option<i8>; MAX_REF_LF_DELTAS] = [None; MAX_REF_LF_DELTAS];
    let mut mode_deltas: [Option<i8>; MAX_MODE_LF_DELTAS] = [None; MAX_MODE_LF_DELTAS];

    if mode_ref_delta_enabled {
        mode_ref_delta_update = rb.read_bit()? != 0;
        if mode_ref_delta_update {
            for slot in ref_deltas.iter_mut() {
                if rb.read_bit()? != 0 {
                    *slot = Some(rb.read_inv_signed_literal(6)? as i8);
                }
            }
            for slot in mode_deltas.iter_mut() {
                if rb.read_bit()? != 0 {
                    *slot = Some(rb.read_inv_signed_literal(6)? as i8);
                }
            }
        }
    }

    Ok(LoopFilterParams {
        filter_level,
        sharpness_level,
        mode_ref_delta_enabled,
        mode_ref_delta_update,
        ref_deltas,
        mode_deltas,
    })
}

/// Encoder mirror of [`parse_loopfilter`]. Mirrors libvpx
/// `encode_loopfilter` (`vp9_bitstream.c:701-739`).
///
/// The libvpx encoder derives the `changed` bit by comparing each
/// delta against the previous frame's saved value; here we use the
/// `Option<i8>` slot semantics from [`LoopFilterParams`] directly —
/// `Some(_)` means the writer emits a present/value pair, `None`
/// means it emits a zero bit.
pub fn write_loopfilter(
    wb: &mut RawBitWriter<'_>,
    lf: &LoopFilterParams,
) -> Result<(), BitstreamError> {
    wb.write_literal(u32::from(lf.filter_level), 6);
    wb.write_literal(u32::from(lf.sharpness_level), 3);

    wb.write_bit(u8::from(lf.mode_ref_delta_enabled));
    if lf.mode_ref_delta_enabled {
        wb.write_bit(u8::from(lf.mode_ref_delta_update));
        if lf.mode_ref_delta_update {
            for slot in lf.ref_deltas.iter() {
                match slot {
                    Some(d) => {
                        wb.write_bit(1);
                        wb.write_inv_signed_literal(i32::from(*d), 6);
                    }
                    None => wb.write_bit(0),
                }
            }
            for slot in lf.mode_deltas.iter() {
                match slot {
                    Some(d) => {
                        wb.write_bit(1);
                        wb.write_inv_signed_literal(i32::from(*d), 6);
                    }
                    None => wb.write_bit(0),
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- quantization

/// Parsed quantization parameters. Mirrors `cm->base_qindex` /
/// `cm->{y_dc,uv_dc,uv_ac}_delta_q` (`vp9_decodeframe.c:1414-1417`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct QuantizationParams {
    /// Frame-level base quantization index (0..=255).
    pub base_qindex: u8,
    /// Luma DC delta (`-15..=15`).
    pub y_dc_delta_q: i8,
    /// Chroma DC delta (`-15..=15`).
    pub uv_dc_delta_q: i8,
    /// Chroma AC delta (`-15..=15`).
    pub uv_ac_delta_q: i8,
}

impl QuantizationParams {
    /// `true` if `base_qindex == 0` and all delta_q values are zero —
    /// the lossless mode flag (`vp9_decodeframe.c:1419-1420`).
    #[must_use]
    pub const fn is_lossless(&self) -> bool {
        self.base_qindex == 0
            && self.y_dc_delta_q == 0
            && self.uv_dc_delta_q == 0
            && self.uv_ac_delta_q == 0
    }
}

/// Read one `delta_q` field: 1 bit "present" flag, then if set a
/// signed 4-bit value. Mirrors libvpx `read_delta_q`
/// (`vp9_decodeframe.c:1408-1410`).
fn read_delta_q(rb: &mut RawBitReader<'_>) -> Result<i8, BitstreamError> {
    if rb.read_bit()? != 0 {
        Ok(rb.read_inv_signed_literal(4)? as i8)
    } else {
        Ok(0)
    }
}

/// Parse the quantization sub-block. Mirrors `setup_quantization`
/// (`vp9_decodeframe.c:1412-1425`).
pub fn parse_quantization(rb: &mut RawBitReader<'_>) -> Result<QuantizationParams, BitstreamError> {
    let base_qindex = rb.read_literal(QINDEX_BITS)? as u8;
    let y_dc_delta_q = read_delta_q(rb)?;
    let uv_dc_delta_q = read_delta_q(rb)?;
    let uv_ac_delta_q = read_delta_q(rb)?;
    Ok(QuantizationParams {
        base_qindex,
        y_dc_delta_q,
        uv_dc_delta_q,
        uv_ac_delta_q,
    })
}

/// Write one `delta_q` field: a present bit, then if the delta is
/// non-zero a signed 4-bit value. Mirrors libvpx `write_delta_q`
/// (`vp9_bitstream.c:741-749`).
fn write_delta_q(wb: &mut RawBitWriter<'_>, delta_q: i8) {
    if delta_q != 0 {
        wb.write_bit(1);
        wb.write_inv_signed_literal(i32::from(delta_q), 4);
    } else {
        wb.write_bit(0);
    }
}

/// Encoder mirror of [`parse_quantization`]. Mirrors libvpx
/// `encode_quantization` (`vp9_bitstream.c:751-757`).
pub fn write_quantization(
    wb: &mut RawBitWriter<'_>,
    q: &QuantizationParams,
) -> Result<(), BitstreamError> {
    wb.write_literal(u32::from(q.base_qindex), QINDEX_BITS);
    write_delta_q(wb, q.y_dc_delta_q);
    write_delta_q(wb, q.uv_dc_delta_q);
    write_delta_q(wb, q.uv_ac_delta_q);
    Ok(())
}

// ---------------------------------------------------------------- segmentation

/// Per-segment feature ids (`vp9_seg_common.h:35-40`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum SegFeature {
    /// Alternate quantizer (signed; max = `MAXQ` = 255).
    AltQ = 0,
    /// Alternate loop-filter level (signed; max =
    /// `MAX_LOOP_FILTER` = 63).
    AltLf = 1,
    /// Override reference frame (unsigned; max = 3).
    ReferenceFrame = 2,
    /// Skip flag (unsigned; max = 0 — i.e. a presence/absence flag).
    Skip = 3,
}

impl SegFeature {
    /// Per-feature data signedness (`vp9_seg_common.c:18`).
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(self, Self::AltQ | Self::AltLf)
    }

    /// Per-feature max value (`vp9_seg_common.c:20-21`).
    #[must_use]
    pub const fn data_max(self) -> u16 {
        match self {
            Self::AltQ => 255,         // MAXQ
            Self::AltLf => 63,         // MAX_LOOP_FILTER
            Self::ReferenceFrame => 3, // 4 reference frames
            Self::Skip => 0,
        }
    }
}

/// Parsed segmentation parameters. Mirrors the header-decoded subset
/// of `struct segmentation` (`vp9_seg_common.h:42-65`).
#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct SegmentationParams {
    /// Segmentation enabled bit.
    pub enabled: bool,
    /// Per-block segment-id update bit.
    pub update_map: bool,
    /// Segment tree probabilities (`MAX_PROB` = no update).
    pub tree_probs: [u8; SEG_TREE_PROBS],
    /// Temporal-update bit (only meaningful when `update_map`).
    pub temporal_update: bool,
    /// Prediction probabilities (`MAX_PROB` = no update).
    pub pred_probs: [u8; PREDICTION_PROBS],
    /// Per-feature data update bit.
    pub update_data: bool,
    /// `abs_delta` flag (`true` ⇒ feature_data is absolute; `false`
    /// ⇒ it's a delta from a baseline value).
    pub abs_delta: bool,
    /// Per-segment feature data. `feature_data[seg][feat]`. Slot is
    /// `None` if the feature is disabled for that segment.
    pub feature_data: [[Option<i16>; SEG_LVL_MAX]; MAX_SEGMENTS],
}

/// Bit-width of an unsigned field bounded by `max`. Mirrors
/// libvpx `get_unsigned_bits` (`vp9_common.h:45-47`):
/// `num_values > 0 ? get_msb(num_values) + 1 : 0`. `get_msb` is the
/// position of the highest set bit.
fn get_unsigned_bits(max: u32) -> u8 {
    if max == 0 {
        0
    } else {
        // MSB position + 1.
        (32 - max.leading_zeros()) as u8
    }
}

/// Read an unsigned field of `get_unsigned_bits(max)` bits, clamped
/// to `max`. Mirrors libvpx `decode_unsigned_max`
/// (`vp9_decodeframe.c:62-65`).
fn decode_unsigned_max(rb: &mut RawBitReader<'_>, max: u32) -> Result<u32, BitstreamError> {
    let bits = get_unsigned_bits(max);
    if bits == 0 {
        return Ok(0);
    }
    let data = rb.read_literal(bits)?;
    Ok(if data > max { max } else { data })
}

/// Write an unsigned field of `get_unsigned_bits(max)` bits. Mirrors
/// libvpx `encode_unsigned_max` (`vp9_bitstream.c:55-58`).
fn encode_unsigned_max(wb: &mut RawBitWriter<'_>, data: u32, max: u32) {
    let bits = get_unsigned_bits(max);
    if bits == 0 {
        return;
    }
    wb.write_literal(data, bits);
}

/// Parse the segmentation sub-block. Mirrors `setup_segmentation`
/// (`vp9_decodeframe.c:1332-1380`).
pub fn parse_segmentation(rb: &mut RawBitReader<'_>) -> Result<SegmentationParams, BitstreamError> {
    let mut seg = SegmentationParams {
        tree_probs: [MAX_PROB; SEG_TREE_PROBS],
        pred_probs: [MAX_PROB; PREDICTION_PROBS],
        ..Default::default()
    };

    seg.enabled = rb.read_bit()? != 0;
    if !seg.enabled {
        return Ok(seg);
    }

    // Map update.
    seg.update_map = rb.read_bit()? != 0;
    if seg.update_map {
        for slot in seg.tree_probs.iter_mut() {
            *slot = if rb.read_bit()? != 0 {
                rb.read_literal(8)? as u8
            } else {
                MAX_PROB
            };
        }
        seg.temporal_update = rb.read_bit()? != 0;
        if seg.temporal_update {
            for slot in seg.pred_probs.iter_mut() {
                *slot = if rb.read_bit()? != 0 {
                    rb.read_literal(8)? as u8
                } else {
                    MAX_PROB
                };
            }
        }
        // `else` branch in libvpx fills pred_probs with MAX_PROB; we
        // already initialised that way above.
    }

    // Data update.
    seg.update_data = rb.read_bit()? != 0;
    if seg.update_data {
        seg.abs_delta = rb.read_bit()? != 0;
        for segment in seg.feature_data.iter_mut() {
            for (fid, slot) in segment.iter_mut().enumerate() {
                let enabled = rb.read_bit()? != 0;
                if enabled {
                    let feature = match fid {
                        0 => SegFeature::AltQ,
                        1 => SegFeature::AltLf,
                        2 => SegFeature::ReferenceFrame,
                        3 => SegFeature::Skip,
                        _ => unreachable!("SEG_LVL_MAX = 4"),
                    };
                    let mag = decode_unsigned_max(rb, feature.data_max() as u32)? as i16;
                    let signed = if feature.is_signed() {
                        if rb.read_bit()? != 0 { -mag } else { mag }
                    } else {
                        mag
                    };
                    *slot = Some(signed);
                }
            }
        }
    }

    Ok(seg)
}

/// Encoder mirror of [`parse_segmentation`]. Mirrors libvpx
/// `encode_segmentation` (`vp9_bitstream.c:759-816`).
///
/// `tree_probs` / `pred_probs` slots equal to [`MAX_PROB`] are encoded
/// as "no update" (a single zero bit) — matching `prob != MAX_PROB`
/// in libvpx `:776 / :786`. `feature_data[seg][feat]` slot `None`
/// emits the disabled bit; `Some(value)` emits the active bit followed
/// by the magnitude (and a sign bit for signed features).
pub fn write_segmentation(
    wb: &mut RawBitWriter<'_>,
    seg: &SegmentationParams,
) -> Result<(), BitstreamError> {
    wb.write_bit(u8::from(seg.enabled));
    if !seg.enabled {
        return Ok(());
    }

    wb.write_bit(u8::from(seg.update_map));
    if seg.update_map {
        for &prob in seg.tree_probs.iter() {
            if prob != MAX_PROB {
                wb.write_bit(1);
                wb.write_literal(u32::from(prob), 8);
            } else {
                wb.write_bit(0);
            }
        }
        wb.write_bit(u8::from(seg.temporal_update));
        if seg.temporal_update {
            for &prob in seg.pred_probs.iter() {
                if prob != MAX_PROB {
                    wb.write_bit(1);
                    wb.write_literal(u32::from(prob), 8);
                } else {
                    wb.write_bit(0);
                }
            }
        }
    }

    wb.write_bit(u8::from(seg.update_data));
    if seg.update_data {
        wb.write_bit(u8::from(seg.abs_delta));
        for segment in seg.feature_data.iter() {
            for (fid, slot) in segment.iter().enumerate() {
                match slot {
                    Some(data) => {
                        wb.write_bit(1);
                        let feature = match fid {
                            0 => SegFeature::AltQ,
                            1 => SegFeature::AltLf,
                            2 => SegFeature::ReferenceFrame,
                            3 => SegFeature::Skip,
                            _ => unreachable!("SEG_LVL_MAX = 4"),
                        };
                        let data_max = u32::from(feature.data_max());
                        let mag = data.unsigned_abs() as u32;
                        if feature.is_signed() {
                            encode_unsigned_max(wb, mag, data_max);
                            wb.write_bit(u8::from(*data < 0));
                        } else {
                            encode_unsigned_max(wb, mag, data_max);
                        }
                    }
                    None => wb.write_bit(0),
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- tile info

/// Parsed tile-info parameters. Mirrors the `log2_tile_cols` /
/// `log2_tile_rows` fields populated by libvpx `setup_tile_info`
/// (`vp9_decodeframe.c:1629-1645`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct TileInfo {
    /// `log2(tile_cols)` — number of tile columns is `1 << log2_tile_cols`.
    pub log2_tile_cols: u8,
    /// `log2(tile_rows)`.
    pub log2_tile_rows: u8,
}

/// Round `mi_cols` (8×8 mode-info column count) up to the next
/// super-block boundary (64×64 → 8 mi columns). Mirrors libvpx
/// `mi_cols_aligned_to_sb` (`vp9_onyxc_int.h` inline).
#[must_use]
const fn mi_cols_aligned_to_sb(mi_cols: i32) -> i32 {
    // (mi_cols + MI_BLOCK_SIZE - 1) & ~(MI_BLOCK_SIZE - 1)
    // With MI_BLOCK_SIZE = 1 << MI_BLOCK_SIZE_LOG2 = 8.
    let mi_block = 1 << MI_BLOCK_SIZE_LOG2;
    (mi_cols + mi_block - 1) & !(mi_block - 1)
}

/// Compute `(min_log2_tile_cols, max_log2_tile_cols)` for a frame
/// with the given `mi_cols`. Mirrors libvpx `vp9_get_tile_n_bits`
/// (`vp9_tile_common.c:51-57`) + the static helpers above it.
#[must_use]
pub fn get_tile_n_bits(mi_cols: i32) -> (i32, i32) {
    let sb64_cols = mi_cols_aligned_to_sb(mi_cols) >> MI_BLOCK_SIZE_LOG2;
    // min_log2_tile_cols: smallest k such that (MAX_TILE_WIDTH_B64 << k) >= sb64_cols.
    let mut min_log2 = 0;
    while (MAX_TILE_WIDTH_B64 << min_log2) < sb64_cols {
        min_log2 += 1;
    }
    // max_log2_tile_cols: largest k such that (sb64_cols >> k) >= MIN_TILE_WIDTH_B64.
    let mut max_log2 = 1;
    while (sb64_cols >> max_log2) >= MIN_TILE_WIDTH_B64 {
        max_log2 += 1;
    }
    (min_log2, max_log2 - 1)
}

/// Parse the tile-info sub-block. Mirrors `setup_tile_info`
/// (`vp9_decodeframe.c:1629-1645`).
///
/// `mi_cols` is the frame's mode-info column count (frame width in
/// 8-pixel units).
pub fn parse_tile_info(
    rb: &mut RawBitReader<'_>,
    mi_cols: i32,
) -> Result<TileInfo, BitstreamError> {
    let (min_log2_tile_cols, max_log2_tile_cols) = get_tile_n_bits(mi_cols);

    // Columns: emit increment bits until we hit max or read a 0.
    let mut log2_tile_cols = min_log2_tile_cols;
    let mut max_ones = max_log2_tile_cols - min_log2_tile_cols;
    while max_ones > 0 && rb.read_bit()? != 0 {
        log2_tile_cols += 1;
        max_ones -= 1;
    }

    if log2_tile_cols > 6 {
        // libvpx rejects > 6 (`vp9_decodeframe.c:1638-1640`). We
        // surface this via the structural error variant.
        return Err(BitstreamError::InvalidTileCols {
            log2_tile_cols: log2_tile_cols as u8,
        });
    }

    // Rows: 1-bit prefix, optionally another bit to bump to 2.
    let mut log2_tile_rows = rb.read_bit()? as i32;
    if log2_tile_rows != 0 {
        log2_tile_rows += rb.read_bit()? as i32;
    }

    Ok(TileInfo {
        log2_tile_cols: log2_tile_cols as u8,
        log2_tile_rows: log2_tile_rows as u8,
    })
}

/// Encoder mirror of [`parse_tile_info`]. Mirrors libvpx
/// `write_tile_info` (`vp9_bitstream.c:886-900`).
///
/// `mi_cols` is the frame's mode-info column count (= `(width + 7) >> 3`).
pub fn write_tile_info(
    wb: &mut RawBitWriter<'_>,
    tile: &TileInfo,
    mi_cols: i32,
) -> Result<(), BitstreamError> {
    let invalid = || BitstreamError::InvalidTileInfo {
        mi_cols,
        log2_tile_cols: tile.log2_tile_cols,
        log2_tile_rows: tile.log2_tile_rows,
    };
    // `mi_cols_aligned_to_sb` adds seven before rounding. Reject invalid public
    // geometry before that arithmetic or any bit write.
    if !(1..=i32::MAX - 7).contains(&mi_cols) {
        return Err(invalid());
    }
    let (min_log2_tile_cols, max_log2_tile_cols) = get_tile_n_bits(mi_cols);
    let log2_tile_cols = tile.log2_tile_cols as i32;
    if !(min_log2_tile_cols..=max_log2_tile_cols).contains(&log2_tile_cols)
        || tile.log2_tile_rows > 2
    {
        return Err(invalid());
    }

    // columns: write (log2_tile_cols - min_log2_tile_cols) 1-bits...
    let mut ones = log2_tile_cols - min_log2_tile_cols;
    while ones > 0 {
        wb.write_bit(1);
        ones -= 1;
    }
    // ...then a terminating 0 unless we've already reached max.
    if log2_tile_cols < max_log2_tile_cols {
        wb.write_bit(0);
    }

    // rows: 0, 1, or 2 — encoded as `(rows != 0)`, then if non-zero `(rows != 1)`.
    let log2_tile_rows = tile.log2_tile_rows;
    wb.write_bit(u8::from(log2_tile_rows != 0));
    if log2_tile_rows != 0 {
        wb.write_bit(u8::from(log2_tile_rows != 1));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::raw::RawBitWriter;

    fn build<F>(f: F) -> Vec<u8>
    where
        F: FnOnce(&mut RawBitWriter),
    {
        let mut buf = vec![0u8; 256];
        let mut w = RawBitWriter::new(&mut buf);
        f(&mut w);
        let used = w.bytes_written();
        buf.truncate(used + 8);
        buf
    }

    // ---- loop filter ----

    #[test]
    fn loopfilter_disabled_deltas() {
        let data = build(|w| {
            w.write_literal(31, 6); // filter_level = 31
            w.write_literal(5, 3); // sharpness = 5
            w.write_bit(0); // mode_ref_delta_enabled = 0
        });
        let mut rb = RawBitReader::new(&data);
        let lf = parse_loopfilter(&mut rb).unwrap();
        assert_eq!(lf.filter_level, 31);
        assert_eq!(lf.sharpness_level, 5);
        assert!(!lf.mode_ref_delta_enabled);
        assert!(!lf.mode_ref_delta_update);
        assert_eq!(lf.ref_deltas, [None; 4]);
        assert_eq!(lf.mode_deltas, [None; 2]);
    }

    #[test]
    fn loopfilter_partial_delta_update() {
        // Enable + update; set ref_deltas[0] = -3 and mode_deltas[1] = +7.
        let data = build(|w| {
            w.write_literal(0, 6); // filter_level
            w.write_literal(0, 3); // sharpness
            w.write_bit(1); // mode_ref_delta_enabled
            w.write_bit(1); // mode_ref_delta_update
            // ref_deltas
            w.write_bit(1); // [0] present
            w.write_inv_signed_literal(-3, 6);
            w.write_bit(0); // [1] absent
            w.write_bit(0); // [2] absent
            w.write_bit(0); // [3] absent
            // mode_deltas
            w.write_bit(0); // [0] absent
            w.write_bit(1); // [1] present
            w.write_inv_signed_literal(7, 6);
        });
        let mut rb = RawBitReader::new(&data);
        let lf = parse_loopfilter(&mut rb).unwrap();
        assert!(lf.mode_ref_delta_enabled);
        assert!(lf.mode_ref_delta_update);
        assert_eq!(lf.ref_deltas, [Some(-3), None, None, None]);
        assert_eq!(lf.mode_deltas, [None, Some(7)]);
    }

    // ---- quantization ----

    #[test]
    fn quantization_lossless() {
        let data = build(|w| {
            w.write_literal(0, 8); // base_qindex
            w.write_bit(0); // no y_dc delta
            w.write_bit(0); // no uv_dc delta
            w.write_bit(0); // no uv_ac delta
        });
        let mut rb = RawBitReader::new(&data);
        let q = parse_quantization(&mut rb).unwrap();
        assert_eq!(q.base_qindex, 0);
        assert_eq!(q.y_dc_delta_q, 0);
        assert!(q.is_lossless());
    }

    #[test]
    fn quantization_with_deltas() {
        let data = build(|w| {
            w.write_literal(120, 8); // base_qindex
            w.write_bit(1);
            w.write_inv_signed_literal(-5, 4); // y_dc_delta = -5
            w.write_bit(1);
            w.write_inv_signed_literal(3, 4); // uv_dc = 3
            w.write_bit(0); // no uv_ac delta
        });
        let mut rb = RawBitReader::new(&data);
        let q = parse_quantization(&mut rb).unwrap();
        assert_eq!(q.base_qindex, 120);
        assert_eq!(q.y_dc_delta_q, -5);
        assert_eq!(q.uv_dc_delta_q, 3);
        assert_eq!(q.uv_ac_delta_q, 0);
        assert!(!q.is_lossless());
    }

    // ---- segmentation ----

    #[test]
    fn segmentation_disabled() {
        let data = build(|w| {
            w.write_bit(0); // not enabled
        });
        let mut rb = RawBitReader::new(&data);
        let seg = parse_segmentation(&mut rb).unwrap();
        assert!(!seg.enabled);
        // Probabilities default to MAX_PROB ⇒ no update.
        assert!(seg.tree_probs.iter().all(|&p| p == MAX_PROB));
    }

    #[test]
    fn segmentation_map_update_no_data() {
        let data = build(|w| {
            w.write_bit(1); // enabled
            w.write_bit(1); // update_map
            // Tree probs: set [0] = 100, rest absent.
            w.write_bit(1);
            w.write_literal(100, 8);
            for _ in 1..SEG_TREE_PROBS {
                w.write_bit(0); // absent ⇒ MAX_PROB
            }
            w.write_bit(0); // temporal_update = 0
            w.write_bit(0); // update_data = 0
        });
        let mut rb = RawBitReader::new(&data);
        let seg = parse_segmentation(&mut rb).unwrap();
        assert!(seg.enabled);
        assert!(seg.update_map);
        assert_eq!(seg.tree_probs[0], 100);
        for i in 1..SEG_TREE_PROBS {
            assert_eq!(seg.tree_probs[i], MAX_PROB);
        }
        assert!(!seg.temporal_update);
        assert!(!seg.update_data);
    }

    #[test]
    fn segmentation_feature_data_signed_and_unsigned() {
        // No map update; only data update with one signed (AltQ) and
        // one unsigned (ReferenceFrame) feature on segment 0.
        let data = build(|w| {
            w.write_bit(1); // enabled
            w.write_bit(0); // update_map = 0
            w.write_bit(1); // update_data = 1
            w.write_bit(1); // abs_delta = 1
            // Segment 0:
            w.write_bit(1); // AltQ enabled
            // AltQ.data_max = 255; bits = 8. value = -50 (signed mag).
            w.write_literal(50, 8);
            w.write_bit(1); // negative
            w.write_bit(0); // AltLf disabled
            w.write_bit(1); // ReferenceFrame enabled
            // ReferenceFrame.data_max = 3; bits = 2. value = 2.
            w.write_literal(2, 2);
            // unsigned → no sign bit
            w.write_bit(0); // Skip disabled
            // Segments 1..7: all four features disabled (1 bit each).
            for _ in 1..MAX_SEGMENTS {
                for _ in 0..SEG_LVL_MAX {
                    w.write_bit(0);
                }
            }
        });
        let mut rb = RawBitReader::new(&data);
        let seg = parse_segmentation(&mut rb).unwrap();
        assert!(seg.enabled);
        assert!(seg.update_data);
        assert!(seg.abs_delta);
        assert_eq!(seg.feature_data[0][0], Some(-50));
        assert_eq!(seg.feature_data[0][1], None);
        assert_eq!(seg.feature_data[0][2], Some(2));
        assert_eq!(seg.feature_data[0][3], None);
        for i in 1..MAX_SEGMENTS {
            assert_eq!(seg.feature_data[i], [None; SEG_LVL_MAX]);
        }
    }

    // ---- tile info ----

    #[test]
    fn tile_info_helpers_match_libvpx() {
        // 1920 width → 240 mi_cols. mi_cols_aligned_to_sb(240) = 240.
        // sb64_cols = 240 / 8 = 30.
        // min: smallest k where (64 << k) >= 30 ⇒ k = 0.
        // max: largest k where (30 >> k) >= 4 ⇒ k = 2 (30 >> 2 = 7,
        //      30 >> 3 = 3 < 4 ⇒ stop). max_log2 - 1 = 2.
        let (min, max) = get_tile_n_bits(240);
        assert_eq!(min, 0);
        assert_eq!(max, 2);
    }

    #[test]
    fn tile_info_min_only() {
        // For mi_cols = 240, min_log2_tile_cols = 0, max_log2_tile_cols = 2.
        // Read a 0 bit immediately ⇒ log2_tile_cols stays at 0.
        let data = build(|w| {
            w.write_bit(0); // first increment bit = 0 → stop at min
            w.write_bit(0); // log2_tile_rows = 0
        });
        let mut rb = RawBitReader::new(&data);
        let t = parse_tile_info(&mut rb, 240).unwrap();
        assert_eq!(t.log2_tile_cols, 0);
        assert_eq!(t.log2_tile_rows, 0);
    }

    #[test]
    fn tile_info_increment_to_max() {
        // For mi_cols = 240 (min = 0, max = 2): two 1-bits to reach 2.
        let data = build(|w| {
            w.write_bit(1); // bump to 1
            w.write_bit(1); // bump to 2 (= max; loop exits because max_ones is 0)
            // log2_tile_rows = 2: first bit 1, second bit 1.
            w.write_bit(1);
            w.write_bit(1);
        });
        let mut rb = RawBitReader::new(&data);
        let t = parse_tile_info(&mut rb, 240).unwrap();
        assert_eq!(t.log2_tile_cols, 2);
        assert_eq!(t.log2_tile_rows, 2);
    }

    #[test]
    fn get_unsigned_bits_matches_libvpx() {
        // libvpx: 0 → 0, 1 → 1, 3 → 2, 255 → 8.
        assert_eq!(super::get_unsigned_bits(0), 0);
        assert_eq!(super::get_unsigned_bits(1), 1);
        assert_eq!(super::get_unsigned_bits(3), 2);
        assert_eq!(super::get_unsigned_bits(255), 8);
        assert_eq!(super::get_unsigned_bits(256), 9);
    }

    // ---- sub-writer round-trips ----

    /// Helper: write `params` via `write` and parse back via `parse`,
    /// asserting equality.
    fn round_trip_lf(params: &LoopFilterParams) {
        let mut buf = vec![0u8; 64];
        let n = {
            let mut w = RawBitWriter::new(&mut buf);
            write_loopfilter(&mut w, params).unwrap();
            w.finish().unwrap()
        };
        let mut rb = RawBitReader::new(&buf[..n + 1]); // +1 byte padding for byte_cursor
        let got = parse_loopfilter(&mut rb).unwrap();
        assert_eq!(&got, params);
    }

    #[test]
    fn write_loopfilter_all_delta_combos() {
        // Disabled.
        round_trip_lf(&LoopFilterParams {
            filter_level: 31,
            sharpness_level: 5,
            ..Default::default()
        });
        // Enabled but no update.
        round_trip_lf(&LoopFilterParams {
            filter_level: 7,
            sharpness_level: 2,
            mode_ref_delta_enabled: true,
            mode_ref_delta_update: false,
            ..Default::default()
        });
        // Enabled + update with partial slot occupancy.
        round_trip_lf(&LoopFilterParams {
            filter_level: 63,
            sharpness_level: 7,
            mode_ref_delta_enabled: true,
            mode_ref_delta_update: true,
            ref_deltas: [Some(-3), None, Some(31), Some(-31)],
            mode_deltas: [None, Some(7)],
        });
        // Enabled + update with all four ref slots populated.
        round_trip_lf(&LoopFilterParams {
            filter_level: 0,
            sharpness_level: 0,
            mode_ref_delta_enabled: true,
            mode_ref_delta_update: true,
            ref_deltas: [Some(0), Some(1), Some(-1), Some(15)],
            mode_deltas: [Some(-2), Some(2)],
        });
    }

    fn round_trip_q(params: &QuantizationParams) {
        let mut buf = vec![0u8; 8];
        let n = {
            let mut w = RawBitWriter::new(&mut buf);
            write_quantization(&mut w, params).unwrap();
            w.finish().unwrap()
        };
        let mut rb = RawBitReader::new(&buf[..n + 1]);
        let got = parse_quantization(&mut rb).unwrap();
        assert_eq!(&got, params);
    }

    #[test]
    fn write_quantization_delta_extremes() {
        // Lossless.
        round_trip_q(&QuantizationParams::default());
        // Each delta at +15 / -15 extremes individually.
        round_trip_q(&QuantizationParams {
            base_qindex: 120,
            y_dc_delta_q: 15,
            uv_dc_delta_q: 0,
            uv_ac_delta_q: 0,
        });
        round_trip_q(&QuantizationParams {
            base_qindex: 250,
            y_dc_delta_q: -15,
            uv_dc_delta_q: 15,
            uv_ac_delta_q: -15,
        });
        // Each base_qindex extreme.
        round_trip_q(&QuantizationParams {
            base_qindex: 255,
            y_dc_delta_q: -1,
            uv_dc_delta_q: 1,
            uv_ac_delta_q: 0,
        });
    }

    fn round_trip_seg(params: &SegmentationParams) {
        let mut buf = vec![0u8; 1024];
        let n = {
            let mut w = RawBitWriter::new(&mut buf);
            write_segmentation(&mut w, params).unwrap();
            w.finish().unwrap()
        };
        let mut rb = RawBitReader::new(&buf[..n + 1]);
        let got = parse_segmentation(&mut rb).unwrap();
        assert_eq!(&got, params);
    }

    /// Build a `SegmentationParams` with the parser's "no update"
    /// defaults: `tree_probs` / `pred_probs` filled with [`MAX_PROB`].
    /// This matches what `parse_segmentation` produces when its early
    /// `enabled = false` exit fires — `Default::default()` would zero
    /// the prob arrays, which is not the parser's resting state.
    fn seg_with_parser_defaults() -> SegmentationParams {
        SegmentationParams {
            tree_probs: [MAX_PROB; SEG_TREE_PROBS],
            pred_probs: [MAX_PROB; PREDICTION_PROBS],
            ..Default::default()
        }
    }

    #[test]
    fn write_segmentation_various_patterns() {
        // Disabled (probabilities sit at MAX_PROB even when disabled,
        // because that's how the parser initialises them before the
        // early return).
        round_trip_seg(&seg_with_parser_defaults());

        // Enabled, map+temporal update but no data update.
        let mut seg = SegmentationParams {
            enabled: true,
            update_map: true,
            temporal_update: true,
            tree_probs: [MAX_PROB; SEG_TREE_PROBS],
            pred_probs: [MAX_PROB; PREDICTION_PROBS],
            ..Default::default()
        };
        seg.tree_probs[0] = 100;
        seg.tree_probs[3] = 200;
        seg.pred_probs[1] = 50;
        round_trip_seg(&seg);

        // Data update only, signed and unsigned features mixed.
        let mut seg = SegmentationParams {
            enabled: true,
            update_data: true,
            abs_delta: true,
            tree_probs: [MAX_PROB; SEG_TREE_PROBS],
            pred_probs: [MAX_PROB; PREDICTION_PROBS],
            ..Default::default()
        };
        seg.feature_data[0][0] = Some(-50); // AltQ signed
        seg.feature_data[0][2] = Some(2); // ReferenceFrame unsigned
        seg.feature_data[3][1] = Some(63); // AltLf max signed
        seg.feature_data[3][3] = Some(0); // Skip (data_max=0; mag bits = 0)
        seg.feature_data[7][0] = Some(255); // AltQ max
        round_trip_seg(&seg);

        // All features active on segment 0 with extreme values.
        let mut seg = SegmentationParams {
            enabled: true,
            update_data: true,
            abs_delta: false,
            tree_probs: [MAX_PROB; SEG_TREE_PROBS],
            pred_probs: [MAX_PROB; PREDICTION_PROBS],
            ..Default::default()
        };
        seg.feature_data[0] = [Some(-255), Some(63), Some(3), Some(0)];
        round_trip_seg(&seg);
    }

    fn round_trip_tile(tile: TileInfo, mi_cols: i32) {
        let mut buf = vec![0u8; 4];
        let n = {
            let mut w = RawBitWriter::new(&mut buf);
            write_tile_info(&mut w, &tile, mi_cols).unwrap();
            w.finish().unwrap()
        };
        let mut rb = RawBitReader::new(&buf[..n + 1]);
        let got = parse_tile_info(&mut rb, mi_cols).unwrap();
        assert_eq!(got, tile);
    }

    #[test]
    fn write_tile_info_all_combinations() {
        // mi_cols = 240 (1920w) → sb64_cols = 30 → min = 0, max = 2.
        assert_eq!(get_tile_n_bits(240), (0, 2));
        for cols in 0..=2u8 {
            for rows in 0..=2u8 {
                round_trip_tile(
                    TileInfo {
                        log2_tile_cols: cols,
                        log2_tile_rows: rows,
                    },
                    240,
                );
            }
        }
        // mi_cols = 80 (640w) → sb64_cols = 10 → min = 0, max = 1.
        assert_eq!(get_tile_n_bits(80), (0, 1));
        for cols in 0..=1u8 {
            for rows in 0..=2u8 {
                round_trip_tile(
                    TileInfo {
                        log2_tile_cols: cols,
                        log2_tile_rows: rows,
                    },
                    80,
                );
            }
        }
        // mi_cols = 40 (320w) → sb64_cols = 5 → min = 0, max = 0
        // (no column-increment bits emitted at all).
        assert_eq!(get_tile_n_bits(40), (0, 0));
        for rows in 0..=2u8 {
            round_trip_tile(
                TileInfo {
                    log2_tile_cols: 0,
                    log2_tile_rows: rows,
                },
                40,
            );
        }
    }

    fn assert_invalid_tile_info(tile: TileInfo, mi_cols: i32) {
        let mut buf = [0xa5; 4];
        let original = buf;
        let err = {
            let mut writer = RawBitWriter::new(&mut buf);
            write_tile_info(&mut writer, &tile, mi_cols).expect_err("invalid tile info must fail")
        };
        assert_eq!(
            err,
            BitstreamError::InvalidTileInfo {
                mi_cols,
                log2_tile_cols: tile.log2_tile_cols,
                log2_tile_rows: tile.log2_tile_rows,
            }
        );
        assert_eq!(buf, original, "validation must happen before writing bits");
    }

    #[test]
    fn write_tile_info_rejects_invalid_geometry_before_writing() {
        assert_eq!(get_tile_n_bits(1024), (1, 5));
        assert_invalid_tile_info(
            TileInfo {
                log2_tile_cols: 0,
                log2_tile_rows: 0,
            },
            1024,
        );
        assert_invalid_tile_info(
            TileInfo {
                log2_tile_cols: 1,
                log2_tile_rows: 0,
            },
            40,
        );
        assert_invalid_tile_info(
            TileInfo {
                log2_tile_cols: 0,
                log2_tile_rows: 3,
            },
            40,
        );
        assert_invalid_tile_info(TileInfo::default(), 0);
    }
}
