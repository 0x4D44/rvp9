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
use super::raw::RawBitReader;

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
pub fn parse_quantization(
    rb: &mut RawBitReader<'_>,
) -> Result<QuantizationParams, BitstreamError> {
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

/// Parse the segmentation sub-block. Mirrors `setup_segmentation`
/// (`vp9_decodeframe.c:1332-1380`).
pub fn parse_segmentation(
    rb: &mut RawBitReader<'_>,
) -> Result<SegmentationParams, BitstreamError> {
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
}
