//! VP9 uncompressed-header struct and parser.
//!
//! Mirrors libvpx `read_uncompressed_header`
//! (`vp9/decoder/vp9_decodeframe.c:2637-2861`) bit-for-bit. This module
//! consumes a byte slice (the start of a frame's payload) and produces a
//! populated [`UncompressedHeader`] plus the byte offset at which the
//! compressed header begins.
//!
//! ## Scope (Tasks 1c + 1d.2)
//!
//! Full support for:
//!
//! - frame-marker check (`vp9_decodeframe.c:2648`),
//! - profile decoding via the variable-width libvpx encoding
//!   (`vp9_read_profile`, `:2945-2950`),
//! - `show_existing_frame` early-exit path (`:2664-2679`),
//! - keyframe decoding through `setup_frame_size`
//!   (`:2685-2703`): sync code, bit-depth / color-space / sampling, frame
//!   size, render size,
//! - intra-only frame decoding through `setup_frame_size`
//!   (`:2710-2735`),
//! - inter-frame decoding: ref-index loop, `setup_frame_size_with_refs`
//!   (`:1553-1604`), `allow_high_precision_mv`, `read_interp_filter`
//!   (`:1454-1459`).
//!
//! **Not in this function (handled separately):**
//! - loop filter / quantization / segmentation / tile-info parsers
//!   ([`super::sub_parsers`]),
//! - `refresh_frame_context` / `frame_parallel_decoding_mode` /
//!   `frame_context_idx` (the post-tile-info block in libvpx
//!   `:2781-2792` — Task 1d.3),
//! - the trailing `first_partition_size` u16 (Task 1d.3),
//! - the compressed-header boolean-coder driver (Task 1e).

use super::error::BitstreamError;
use super::frame::{
    BitDepth, ChromaFormat, ColorRange, ColorSpace, FrameType, InterpFilter, MAX_INTER_REFS,
    Profile, REF_FRAMES, REF_FRAMES_LOG2, sync,
};
use super::raw::RawBitReader;
use super::sub_parsers::{
    LoopFilterParams, QuantizationParams, SegmentationParams, TileInfo, parse_loopfilter,
    parse_quantization, parse_segmentation, parse_tile_info,
};

/// Number of `FRAME_CONTEXTS_LOG2` bits used by `frame_context_idx`
/// (libvpx `vp9_entropymode.h` — `FRAME_CONTEXTS = 4`,
/// `FRAME_CONTEXTS_LOG2 = 2`).
pub const FRAME_CONTEXTS_LOG2: u8 = 2;

/// Decoded uncompressed-header. One value per VP9 frame. Field set
/// depends on `kind`:
///
/// - `ShowExistingFrame`: only `kind` and `slot` (the rest are
///   inherited from the existing frame and left at default).
/// - `KeyFrame` / `IntraOnly`: full color/sampling/size populated.
/// - `InterFrameTodo`: parser bailed early; do not use field values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UncompressedHeader {
    /// The frame's "kind" — determines which other fields are populated.
    pub kind: FrameKind,
    /// VP9 bitstream profile. Always populated (it's read before the
    /// `show_existing_frame` branch).
    pub profile: Profile,
    /// `true` if this frame is displayable. From libvpx field
    /// `cm->show_frame`. Defaults to `true` for `ShowExistingFrame`.
    pub show_frame: bool,
    /// `true` if `error_resilient_mode` was set in the header. Only
    /// meaningful when `kind` is `KeyFrame` / `IntraOnly` / future
    /// `InterFrame`.
    pub error_resilient: bool,
    /// Luma bit depth (8 / 10 / 12). HBD (10 / 12) gets rejected by
    /// the caller in v1 (D-01 RESOLVED).
    pub bit_depth: BitDepth,
    /// Color space (decoded from header for KEY frames and
    /// non-Profile-0 intra-only frames; defaults to `Bt601` for
    /// Profile-0 intra-only per libvpx `:2721`).
    pub color_space: ColorSpace,
    /// Color range (Studio / Full). Defaults to `Studio` for
    /// Profile-0 intra-only.
    pub color_range: ColorRange,
    /// Chroma format derived from the `(subsampling_x, subsampling_y)`
    /// bits.
    pub chroma: ChromaFormat,
    /// Coded luma frame width in pixels (after the `+1` decoding
    /// step). `0` if not yet decoded.
    pub width: u32,
    /// Coded luma frame height in pixels.
    pub height: u32,
    /// Render size as `(render_width, render_height)` if the header
    /// signalled a render-size-different-from-frame-size; otherwise
    /// `None` (callers treat as identical to `(width, height)`).
    pub render_size: Option<(u32, u32)>,
    /// Post-disposition fields (refresh_frame_context, the four
    /// sub-blocks, and the trailing first_partition_size). `None` for
    /// `ShowExistingFrame` since libvpx returns 0 immediately after
    /// resolving the show slot; otherwise populated.
    pub tail: Option<HeaderTail>,
}

/// Post-disposition fields of the uncompressed header — present for
/// every frame except `ShowExistingFrame`. Mirrors libvpx
/// `vp9_decodeframe.c:2781-2860`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeaderTail {
    /// `refresh_frame_context` (1 bit; 0 if `error_resilient_mode`).
    /// libvpx :2782 / :2786.
    pub refresh_frame_context: bool,
    /// `frame_parallel_decoding_mode` (1 bit; forced to `true` if
    /// `error_resilient_mode`). libvpx :2783 / :2787.
    pub frame_parallel_decoding_mode: bool,
    /// `frame_context_idx` (2 bits; index into the four frame context
    /// banks). libvpx :2792.
    pub frame_context_idx: u8,
    /// Decoded loop-filter parameters.
    pub loop_filter: LoopFilterParams,
    /// Decoded quantization parameters.
    pub quantization: QuantizationParams,
    /// Decoded segmentation parameters.
    pub segmentation: SegmentationParams,
    /// Decoded tile-info parameters.
    pub tile_info: TileInfo,
    /// `first_partition_size`: byte length of the compressed (boolean
    /// coder) header partition. libvpx :2854-2858.
    pub first_partition_size: u16,
}

/// Frame-kind variants. Mirrors the disposition libvpx makes after
/// reading `frame_type` + `show_existing_frame` + `intra_only`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FrameKind {
    /// Show a previously-decoded frame from the buffer pool. `slot`
    /// is the 3-bit index into `ref_frame_map`.
    ShowExistingFrame {
        /// Buffer-pool slot index (`0..REF_FRAMES`).
        slot: u8,
    },
    /// `frame_type == KEY_FRAME`.
    KeyFrame,
    /// `frame_type == INTER_FRAME && !show_frame && intra_only == 1`.
    IntraOnly {
        /// The 2-bit `reset_frame_context` field (0 if
        /// `error_resilient_mode`).
        reset_frame_context: u8,
        /// 8-bit `refresh_frame_flags` mask.
        refresh_frame_flags: u8,
    },
    /// `frame_type == INTER_FRAME && (show_frame || intra_only == 0)`
    /// — full inter-frame parsing.
    InterFrame {
        /// The 2-bit `reset_frame_context` field (0 if
        /// `error_resilient_mode`).
        reset_frame_context: u8,
        /// 8-bit `refresh_frame_flags` mask.
        refresh_frame_flags: u8,
        /// 3-bit ref-frame indices (one per
        /// [`MAX_INTER_REFS`] slot: LAST / GOLDEN / ALTREF).
        ref_frame_indices: [u8; MAX_INTER_REFS],
        /// Per-ref sign-bias bits (sign of the relative POC offset
        /// between reference and current frame).
        ref_frame_sign_biases: [bool; MAX_INTER_REFS],
        /// `true` if the header signalled 1/8-pel MV precision.
        allow_high_precision_mv: bool,
        /// Sub-pel interpolation filter for this frame.
        interp_filter: InterpFilter,
    },
}

impl FrameKind {
    /// Convenience: is this a KeyFrame or IntraOnly variant?
    #[must_use]
    pub const fn is_intra(self) -> bool {
        matches!(self, Self::KeyFrame | Self::IntraOnly { .. })
    }
}

/// Sizes of the frames currently held in the decoder's 8-slot
/// reference pool, indexed by `ref_frame_map` slot (the 3-bit value
/// the inter-frame header reads per logical ref). `None` means the
/// corresponding pool slot is empty.
///
/// Mirrors libvpx's `cm->ref_frame_map[ref] -> pool slot ->
/// frame_bufs[idx].buf.y_crop_{width,height}` lookup chain used by
/// `setup_frame_size_with_refs` (`vp9_decodeframe.c:1559-1574`).
pub type RefFrameSizes = [Option<(u32, u32)>; REF_FRAMES];

/// Decode a VP9 uncompressed header from `data`. Returns the populated
/// header and the byte offset (one past the last header byte) at which
/// the compressed-header partition begins.
///
/// `ref_sizes` provides the (luma width, luma height) of every frame
/// currently held in the 8-slot reference pool, indexed by
/// `ref_frame_map` slot. The parser uses the just-read
/// `ref_frame_indices[i]` to look up the size for each logical
/// inter-ref (LAST=0 / GOLDEN=1 / ALTREF=2). Empty slots (`None`)
/// surface as [`BitstreamError::InvalidRefFrameSize`] when the
/// header tries to inherit from them — mirroring libvpx's
/// `INVALID_IDX → "Failed to decode frame size"` rejection at
/// `vp9_decodeframe.c:1567-1570`.
///
/// Mirrors `read_uncompressed_header` (`vp9_decodeframe.c:2637-2861`)
/// up to and including the ref-frame loop and
/// `setup_frame_size_with_refs` (`:1553-1604`). The trailing sub-block
/// parsers (loopfilter / quant / segmentation / tile + the 16-bit
/// `first_partition_size`) are exposed separately via
/// [`super::sub_parsers`].
pub fn parse_uncompressed_header(
    data: &[u8],
    ref_sizes: &RefFrameSizes,
) -> Result<(UncompressedHeader, usize), BitstreamError> {
    let mut rb = RawBitReader::new(data);

    // 2-bit frame_marker, must be 0b10 (= 2). libvpx :2648.
    let marker = rb.read_literal(2u8)? as u8;
    if marker != sync::FRAME_MARKER {
        return Err(BitstreamError::InvalidFrameMarker { found: marker });
    }

    // Profile: 2 bits (low | high<<1), plus an extra bit if profile == 3.
    // libvpx vp9_read_profile :2945-2950.
    let profile_raw = read_profile_raw(&mut rb)?;
    let profile = match Profile::from_u8(profile_raw) {
        Some(p) => p,
        None => {
            return Err(BitstreamError::UnsupportedProfile {
                profile: profile_raw,
            });
        }
    };

    // show_existing_frame branch — early return without populating
    // bit-depth / color / size / tail. libvpx :2663-2679.
    if rb.read_bit()? != 0 {
        let slot = rb.read_literal(3)? as u8;
        return Ok((
            UncompressedHeader {
                kind: FrameKind::ShowExistingFrame { slot },
                profile,
                show_frame: true,
                error_resilient: false,
                bit_depth: BitDepth::Eight,
                color_space: ColorSpace::Unknown,
                color_range: ColorRange::Studio,
                chroma: ChromaFormat::Yuv420,
                width: 0,
                height: 0,
                render_size: None,
                tail: None,
            },
            rb.bytes_consumed(),
        ));
    }

    // frame_type (1) + show_frame (1) + error_resilient (1).
    // libvpx :2681-2683.
    let frame_type = FrameType::from_bit(rb.read_bit()?);
    let show_frame = rb.read_bit()? != 0;
    let error_resilient = rb.read_bit()? != 0;

    // Each branch below populates (kind, color, width, height, render_size)
    // and falls through to the shared post-disposition tail block.
    let kind: FrameKind;
    let color: ColorConfig;
    let width: u32;
    let height: u32;
    let render_size: Option<(u32, u32)>;

    if frame_type == FrameType::Key {
        // Keyframe path. libvpx :2685-2703.
        read_sync_code(&mut rb)?;
        color = read_bitdepth_colorspace_sampling(&mut rb, profile)?;
        (width, height) = read_frame_size(&mut rb)?;
        render_size = read_render_size(&mut rb, width, height)?;
        kind = FrameKind::KeyFrame;
    } else {

        // Inter-frame branch. libvpx :2704-2766.
        let intra_only = if show_frame {
            false
        } else {
            rb.read_bit()? != 0
        };
        let reset_frame_context = if error_resilient {
            0
        } else {
            rb.read_literal(2)? as u8
        };

        if intra_only {
            // Intra-only inter frame: sync code + maybe color config +
            // refresh_frame_flags + frame_size. libvpx :2710-2735.
            read_sync_code(&mut rb)?;
            color = if profile != Profile::Profile0 {
                read_bitdepth_colorspace_sampling(&mut rb, profile)?
            } else {
                // Profile 0 intra-only inherits defaults per spec.
                ColorConfig {
                    bit_depth: BitDepth::Eight,
                    color_space: ColorSpace::Bt601,
                    color_range: ColorRange::Studio,
                    chroma: ChromaFormat::Yuv420,
                }
            };
            let refresh_frame_flags = rb.read_literal(REF_FRAMES as u8)? as u8;
            (width, height) = read_frame_size(&mut rb)?;
            render_size = read_render_size(&mut rb, width, height)?;
            kind = FrameKind::IntraOnly {
                reset_frame_context,
                refresh_frame_flags,
            };
        } else {
            // Full inter-frame parse. Mirrors libvpx :2737-2765.
            let refresh_frame_flags = rb.read_literal(REF_FRAMES as u8)? as u8;

            let mut ref_frame_indices = [0u8; MAX_INTER_REFS];
            let mut ref_frame_sign_biases = [false; MAX_INTER_REFS];
            for i in 0..MAX_INTER_REFS {
                ref_frame_indices[i] = rb.read_literal(REF_FRAMES_LOG2 as u8)? as u8;
                ref_frame_sign_biases[i] = rb.read_bit()? != 0;
            }

            // setup_frame_size_with_refs. libvpx :1553-1604.
            //
            // Read up to REFS_PER_FRAME (= 3) "size_in_found" bits, one
            // per logical inter-ref (LAST/GOLDEN/ALTREF). On the FIRST
            // bit that's `1`, libvpx looks up the referenced pool slot's
            // size and BREAKS the loop — subsequent bits are NOT read.
            // If none of the bits are set, fresh 32-bit frame_size
            // follows.
            let mut inherited: Option<(u32, u32)> = None;
            for (i, ref_idx) in ref_frame_indices.iter().enumerate() {
                let use_size = rb.read_bit()? != 0;
                if use_size {
                    // Resolve `ref_frame_indices[i]` against the pool.
                    let pool_idx = *ref_idx as usize;
                    match ref_sizes.get(pool_idx).copied().flatten() {
                        Some(sz) => {
                            inherited = Some(sz);
                            break;
                        }
                        None => {
                            return Err(BitstreamError::InvalidRefFrameSize {
                                logical_ref: i as u8,
                                pool_slot: *ref_idx,
                            });
                        }
                    }
                }
            }
            (width, height) = if let Some(sz) = inherited {
                sz
            } else {
                read_frame_size(&mut rb)?
            };
            render_size = read_render_size(&mut rb, width, height)?;

            let allow_high_precision_mv = rb.read_bit()? != 0;
            let interp_filter = read_interp_filter(&mut rb)?;

            // Bit-depth and color inherit from the most recent
            // key/intra-only frame; libvpx pulls them from `cm->`.
            // The parser doesn't re-encode them here.
            color = ColorConfig {
                bit_depth: BitDepth::Eight,
                color_space: ColorSpace::Unknown,
                color_range: ColorRange::Studio,
                chroma: ChromaFormat::Yuv420,
            };
            kind = FrameKind::InterFrame {
                reset_frame_context,
                refresh_frame_flags,
                ref_frame_indices,
                ref_frame_sign_biases,
                allow_high_precision_mv,
                interp_filter,
            };
        }
    }

    // ----- Shared post-disposition tail. libvpx :2781-2858. -----

    // refresh_frame_context / frame_parallel_decoding_mode (libvpx :2781-2788).
    let (refresh_frame_context, frame_parallel_decoding_mode) = if error_resilient {
        (false, true)
    } else {
        let rfc = rb.read_bit()? != 0;
        let fpdm = rb.read_bit()? != 0;
        (rfc, fpdm)
    };

    // frame_context_idx (libvpx :2792).
    let frame_context_idx = rb.read_literal(FRAME_CONTEXTS_LOG2)? as u8;

    // Sub-block parsers (libvpx :2819-2824).
    let loop_filter = parse_loopfilter(&mut rb)?;
    let quantization = parse_quantization(&mut rb)?;
    let segmentation = parse_segmentation(&mut rb)?;
    // mi_cols for tile_info — `(width + 7) >> 3` per libvpx
    // mi-unit convention (`MI_SIZE_LOG2 = 3`).
    let mi_cols = ((width + 7) >> 3) as i32;
    let tile_info = parse_tile_info(&mut rb, mi_cols)?;

    // first_partition_size: 16-bit, must be non-zero (libvpx :2854-2858).
    let first_partition_size = rb.read_literal(16)? as u16;
    if first_partition_size == 0 {
        return Err(BitstreamError::InvalidFirstPartitionSize);
    }

    Ok((
        UncompressedHeader {
            kind,
            profile,
            show_frame,
            error_resilient,
            bit_depth: color.bit_depth,
            color_space: color.color_space,
            color_range: color.color_range,
            chroma: color.chroma,
            width,
            height,
            render_size,
            tail: Some(HeaderTail {
                refresh_frame_context,
                frame_parallel_decoding_mode,
                frame_context_idx,
                loop_filter,
                quantization,
                segmentation,
                tile_info,
                first_partition_size,
            }),
        },
        rb.bytes_consumed(),
    ))
}

/// Read the sub-pel interpolation filter. Mirrors libvpx
/// `read_interp_filter` (`vp9_decodeframe.c:1454-1459`).
fn read_interp_filter(rb: &mut RawBitReader<'_>) -> Result<InterpFilter, BitstreamError> {
    if rb.read_bit()? != 0 {
        Ok(InterpFilter::Switchable)
    } else {
        let literal = rb.read_literal(2)? as u8;
        Ok(InterpFilter::from_literal(literal))
    }
}

/// Read the profile field using libvpx's variable-width encoding
/// (`vp9_read_profile`, `vp9_decodeframe.c:2945-2950`). Returns 0..=4
/// where 4 is the out-of-range sentinel that the caller turns into
/// `UnsupportedProfile`.
fn read_profile_raw(rb: &mut RawBitReader<'_>) -> Result<u8, BitstreamError> {
    let mut profile = rb.read_bit()?;
    profile |= rb.read_bit()? << 1;
    if profile > 2 {
        profile += rb.read_bit()?;
    }
    Ok(profile)
}

/// Read and validate the 24-bit sync code. Mirrors
/// `vp9_read_sync_code` (`vp9_decodeframe.c:2933-2937`).
fn read_sync_code(rb: &mut RawBitReader<'_>) -> Result<(), BitstreamError> {
    let b0 = rb.read_literal(8)? as u8;
    let b1 = rb.read_literal(8)? as u8;
    let b2 = rb.read_literal(8)? as u8;
    if (b0, b1, b2) != (sync::SYNC_CODE_0, sync::SYNC_CODE_1, sync::SYNC_CODE_2) {
        return Err(BitstreamError::InvalidSyncCode {
            found: [b0, b1, b2],
        });
    }
    Ok(())
}

/// Decoded color configuration — bit depth, color space, color range,
/// chroma format.
#[derive(Copy, Clone, Debug)]
struct ColorConfig {
    bit_depth: BitDepth,
    color_space: ColorSpace,
    color_range: ColorRange,
    chroma: ChromaFormat,
}

/// Mirrors `read_bitdepth_colorspace_sampling`
/// (`vp9_decodeframe.c:2606-2635`).
fn read_bitdepth_colorspace_sampling(
    rb: &mut RawBitReader<'_>,
    profile: Profile,
) -> Result<ColorConfig, BitstreamError> {
    // Bit depth: Profile 0/1 → fixed 8-bit. Profile 2/3 → 1 bit
    // "ten_or_twelve" (0 → 10, 1 → 12). Mirrors libvpx :2610-2613.
    let bit_depth = if matches!(profile, Profile::Profile2 | Profile::Profile3) {
        if rb.read_bit()? != 0 {
            BitDepth::Twelve
        } else {
            BitDepth::Ten
        }
    } else {
        BitDepth::Eight
    };

    // Color space: 3-bit field. libvpx :2615.
    let cs_raw = rb.read_literal(3)? as u8;
    let color_space = ColorSpace::from_u8(cs_raw).expect("3-bit value always 0..=7");

    let (color_range, chroma) = if color_space != ColorSpace::Srgb {
        // 1 bit color_range, then for Profile 1/3 a pair of subsampling
        // bits; for Profile 0/2 the chroma format is forced to 4:2:0.
        let color_range = ColorRange::from_bit(rb.read_bit()?);
        let chroma = if matches!(profile, Profile::Profile1 | Profile::Profile3) {
            let sx = rb.read_bit()? != 0;
            let sy = rb.read_bit()? != 0;
            // libvpx :2624-2626: reserved bit must be 0.
            if rb.read_bit()? != 0 {
                return Err(BitstreamError::InconsistentSrgb);
            }
            ChromaFormat::from_subsampling(sx, sy)
        } else {
            ChromaFormat::Yuv420
        };
        (color_range, chroma)
    } else {
        // sRGB requires Profile 1/3 + 4:4:4. libvpx :2629-2634.
        if !matches!(profile, Profile::Profile1 | Profile::Profile3) {
            return Err(BitstreamError::InconsistentSrgb);
        }
        // libvpx reads 1 bit and demands it be 0.
        if rb.read_bit()? != 0 {
            return Err(BitstreamError::InconsistentSrgb);
        }
        (ColorRange::Full, ChromaFormat::Yuv444)
    };

    Ok(ColorConfig {
        bit_depth,
        color_space,
        color_range,
        chroma,
    })
}

/// Mirrors `read_frame_size` (`vp9_decodeframe.c:2939-2943`).
fn read_frame_size(rb: &mut RawBitReader<'_>) -> Result<(u32, u32), BitstreamError> {
    let width = rb.read_literal(16)? + 1;
    let height = rb.read_literal(16)? + 1;
    Ok((width, height))
}

/// Mirrors the render-size block at `vp9_decodeframe.c:2562-2577` —
/// 1 bit `render_and_frame_size_different`, then if set, two 16-bit
/// `render_width-1` / `render_height-1` fields.
fn read_render_size(
    rb: &mut RawBitReader<'_>,
    _frame_width: u32,
    _frame_height: u32,
) -> Result<Option<(u32, u32)>, BitstreamError> {
    if rb.read_bit()? != 0 {
        let rw = rb.read_literal(16)? + 1;
        let rh = rb.read_literal(16)? + 1;
        Ok(Some((rw, rh)))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::raw::RawBitWriter;
    use crate::bitstream::sub_parsers::get_tile_n_bits;

    /// Convenience: "no reference frames currently resolved" — the
    /// state at decoder boot and after a `need_resync`. Equivalent to
    /// `cm->frame_refs[i].idx == INVALID_IDX` for every slot.
    const NO_REFS: RefFrameSizes = [None; REF_FRAMES];

    /// Helper: build a header byte sequence by writing fields with a
    /// `RawBitWriter`, then return the populated buffer.
    fn build_header<F>(f: F) -> Vec<u8>
    where
        F: FnOnce(&mut RawBitWriter),
    {
        let mut buf = vec![0u8; 256];
        let mut w = RawBitWriter::new(&mut buf);
        f(&mut w);
        let used = w.bytes_written();
        buf.truncate(used + 8); // tail padding for the byte-aligned reader
        buf
    }

    /// Append a minimal "all zeros" post-disposition tail to the
    /// writer: 2 refresh/fpdm bits + 2 frame_context_idx bits + 10
    /// loopfilter bits + 11 quantization bits + 1 segmentation
    /// "disabled" bit + the right number of tile bits for `width` +
    /// 16 bits first_partition_size = 1. Used by every test that
    /// builds a non-show-existing valid header.
    fn write_minimal_tail(w: &mut RawBitWriter, width: u32) {
        // refresh_frame_context (1) + frame_parallel_decoding_mode (1)
        w.write_bit(0);
        w.write_bit(0);
        // frame_context_idx (2 bits)
        w.write_literal(0, 2);
        // loopfilter: filter_level (6) + sharpness (3) + delta_enabled (1) = 10 bits
        w.write_literal(0, 6);
        w.write_literal(0, 3);
        w.write_bit(0);
        // quantization: base_qindex (8) + three "no delta" 1-bit flags
        w.write_literal(0, 8);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        // segmentation enabled = 0
        w.write_bit(0);
        // tile_info increment bits: the parser's loop terminates on
        // the first 0 bit, so we only need to emit ONE zero (or zero
        // bits if min == max). Then 1 zero for log2_tile_rows.
        let mi_cols = ((width + 7) >> 3) as i32;
        let (min_log2, max_log2) = get_tile_n_bits(mi_cols);
        if max_log2 - min_log2 > 0 {
            w.write_bit(0); // stops the column-increment loop on iteration 1
        }
        w.write_bit(0); // log2_tile_rows = 0
        // first_partition_size = 1 (non-zero).
        w.write_literal(1, 16);
    }

    #[test]
    fn invalid_frame_marker_rejected() {
        let buf = build_header(|w| {
            // Wrong marker — should be 0b10 = 2.
            w.write_literal(0b11, 2);
            // Pad enough bytes for the reader to attempt the next reads.
            w.write_literal(0, 8);
        });
        let err = parse_uncompressed_header(&buf, &NO_REFS).unwrap_err();
        assert!(matches!(
            err,
            BitstreamError::InvalidFrameMarker { found: 3 }
        ));
    }

    #[test]
    fn unsupported_profile_4_rejected() {
        // profile == 3 then extra bit == 1 → raw profile = 4 (out of
        // range; libvpx clamps to MAX_PROFILES).
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(1); // profile low
            w.write_bit(1); // profile high → 3
            w.write_bit(1); // extra bit → 4
            // Pad.
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let err = parse_uncompressed_header(&buf, &NO_REFS).unwrap_err();
        assert!(matches!(
            err,
            BitstreamError::UnsupportedProfile { profile: 4 }
        ));
    }

    #[test]
    fn show_existing_frame_returns_slot() {
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(0); // profile low
            w.write_bit(0); // profile high → Profile 0
            w.write_bit(1); // show_existing_frame = 1
            w.write_literal(5, 3); // slot
            // Pad bytes (reader requires valid byte access for byte_cursor).
            w.write_literal(0, 8);
        });
        let (hdr, _) = parse_uncompressed_header(&buf, &NO_REFS).expect("valid header");
        assert_eq!(hdr.kind, FrameKind::ShowExistingFrame { slot: 5 });
        assert_eq!(hdr.profile, Profile::Profile0);
        assert!(hdr.show_frame);
    }

    #[test]
    fn keyframe_minimal_decode() {
        // Build a minimal keyframe header: profile 0, show_frame=1,
        // error_resilient=0, sync code, BT.601, studio range, frame
        // size 320x240, no render size.
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(0); // profile low
            w.write_bit(0); // profile high → Profile 0
            w.write_bit(0); // show_existing_frame = 0
            w.write_bit(0); // frame_type = KEY (0)
            w.write_bit(1); // show_frame = 1
            w.write_bit(0); // error_resilient = 0
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            // color config for Profile 0:
            w.write_literal(ColorSpace::Bt601 as u32, 3); // color_space
            w.write_bit(0); // color_range = Studio
            // frame size: width-1 = 319, height-1 = 239
            w.write_literal(319, 16);
            w.write_literal(239, 16);
            w.write_bit(0); // render_and_frame_size_different = 0
            write_minimal_tail(w, 320);
            // Pad.
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _cursor) = parse_uncompressed_header(&buf, &NO_REFS).expect("valid keyframe");
        assert_eq!(hdr.kind, FrameKind::KeyFrame);
        assert_eq!(hdr.profile, Profile::Profile0);
        assert!(hdr.show_frame);
        assert!(!hdr.error_resilient);
        assert_eq!(hdr.bit_depth, BitDepth::Eight);
        assert_eq!(hdr.color_space, ColorSpace::Bt601);
        assert_eq!(hdr.color_range, ColorRange::Studio);
        assert_eq!(hdr.chroma, ChromaFormat::Yuv420);
        assert_eq!(hdr.width, 320);
        assert_eq!(hdr.height, 240);
        assert_eq!(hdr.render_size, None);
        let tail = hdr.tail.expect("non-show-existing frames must carry a tail");
        assert!(!tail.refresh_frame_context);
        assert!(!tail.frame_parallel_decoding_mode);
        assert_eq!(tail.frame_context_idx, 0);
        assert_eq!(tail.first_partition_size, 1);
        assert!(!tail.loop_filter.mode_ref_delta_enabled);
        assert!(tail.quantization.is_lossless());
        assert!(!tail.segmentation.enabled);
        assert_eq!(tail.tile_info.log2_tile_cols, 0);
        assert_eq!(tail.tile_info.log2_tile_rows, 0);
    }

    #[test]
    fn keyframe_with_render_size() {
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(1);
            w.write_bit(0);
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            w.write_literal(ColorSpace::Bt709 as u32, 3);
            w.write_bit(1); // color_range = Full
            w.write_literal(1919, 16); // 1920x1080 coded
            w.write_literal(1079, 16);
            w.write_bit(1); // render different
            w.write_literal(639, 16); // render 640x360
            w.write_literal(359, 16);
            write_minimal_tail(w, 1920);
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _) = parse_uncompressed_header(&buf, &NO_REFS).expect("valid keyframe");
        assert_eq!(hdr.kind, FrameKind::KeyFrame);
        assert_eq!(hdr.width, 1920);
        assert_eq!(hdr.height, 1080);
        assert_eq!(hdr.color_space, ColorSpace::Bt709);
        assert_eq!(hdr.color_range, ColorRange::Full);
        assert_eq!(hdr.render_size, Some((640, 360)));
        assert!(hdr.tail.is_some());
    }

    #[test]
    fn keyframe_invalid_sync_code_rejected() {
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(1);
            w.write_bit(0);
            // Wrong sync bytes.
            w.write_literal(0xff, 8);
            w.write_literal(0x83, 8);
            w.write_literal(0x42, 8);
            for _ in 0..16 {
                w.write_literal(0, 8);
            }
        });
        let err = parse_uncompressed_header(&buf, &NO_REFS).unwrap_err();
        assert!(matches!(
            err,
            BitstreamError::InvalidSyncCode { found: [0xff, 0x83, 0x42] }
        ));
    }

    #[test]
    fn intra_only_inherits_profile0_defaults() {
        // frame_type = INTER (1), show_frame = 0, intra_only = 1.
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(0); // profile low → 0
            w.write_bit(0); // profile high → 0
            w.write_bit(0); // show_existing = 0
            w.write_bit(1); // frame_type = INTER
            w.write_bit(0); // show_frame = 0
            w.write_bit(0); // error_resilient = 0
            w.write_bit(1); // intra_only = 1
            w.write_literal(0, 2); // reset_frame_context = 0
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            // Profile 0: no color/sampling fields.
            w.write_literal(0xa3, 8); // refresh_frame_flags = 0xa3
            w.write_literal(319, 16); // width-1
            w.write_literal(239, 16); // height-1
            w.write_bit(0); // no render size
            write_minimal_tail(w, 320);
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _) = parse_uncompressed_header(&buf, &NO_REFS).expect("valid intra-only");
        assert_eq!(
            hdr.kind,
            FrameKind::IntraOnly {
                reset_frame_context: 0,
                refresh_frame_flags: 0xa3,
            }
        );
        assert_eq!(hdr.profile, Profile::Profile0);
        assert_eq!(hdr.color_space, ColorSpace::Bt601);
        assert_eq!(hdr.color_range, ColorRange::Studio);
        assert_eq!(hdr.bit_depth, BitDepth::Eight);
        assert_eq!(hdr.width, 320);
        assert_eq!(hdr.height, 240);
    }

    #[test]
    fn inter_frame_no_refs_reads_fresh_size() {
        // frame_type = INTER, show_frame = 1 (so intra_only = 0 by
        // construction). No refs available → setup_frame_size_with_refs
        // falls through to reading 32 bits of fresh frame size.
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(0); // profile low → 0
            w.write_bit(0); // profile high → 0
            w.write_bit(0); // show_existing = 0
            w.write_bit(1); // frame_type = INTER
            w.write_bit(1); // show_frame = 1
            w.write_bit(0); // error_resilient = 0
            w.write_literal(0b10, 2); // reset_frame_context = 2
            w.write_literal(0b00010101, 8); // refresh_frame_flags
            // Three ref slots: each is 3 bits index + 1 bit sign bias.
            // refs = [0, 4, 7]; sign_biases = [false, true, false].
            w.write_literal(0, 3);
            w.write_bit(0);
            w.write_literal(4, 3);
            w.write_bit(1);
            w.write_literal(7, 3);
            w.write_bit(0);
            // setup_frame_size_with_refs: 3 "size_in_found" bits — all
            // 0 because NO_REFS. Then 16+16 fresh size.
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_literal(639, 16); // width 640
            w.write_literal(479, 16); // height 480
            w.write_bit(0); // no render size
            w.write_bit(1); // allow_high_precision_mv
            w.write_bit(1); // interp_filter = Switchable
            write_minimal_tail(w, 640);
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _) = parse_uncompressed_header(&buf, &NO_REFS).expect("valid inter frame");
        match hdr.kind {
            FrameKind::InterFrame {
                reset_frame_context,
                refresh_frame_flags,
                ref_frame_indices,
                ref_frame_sign_biases,
                allow_high_precision_mv,
                interp_filter,
            } => {
                assert_eq!(reset_frame_context, 2);
                assert_eq!(refresh_frame_flags, 0b00010101);
                assert_eq!(ref_frame_indices, [0, 4, 7]);
                assert_eq!(ref_frame_sign_biases, [false, true, false]);
                assert!(allow_high_precision_mv);
                assert_eq!(interp_filter, InterpFilter::Switchable);
            }
            other => panic!("expected InterFrame, got {other:?}"),
        }
        assert_eq!(hdr.width, 640);
        assert_eq!(hdr.height, 480);
        assert_eq!(hdr.render_size, None);
    }

    #[test]
    fn inter_frame_inherits_size_from_ref() {
        // Pool slot 5 has 1920x1080. All three logical inter-refs
        // (LAST/GOLDEN/ALTREF) map to pool slot 5 via `ref_frame_indices`.
        // size_in_found bits `[0, 1, ...]` → parser reads bit 0 (= 0),
        // bit 1 (= 1), looks up `pool_sizes[ref_frame_indices[1]] =
        // pool_sizes[5] = Some((1920, 1080))`, and BREAKS (matches
        // libvpx `vp9_decodeframe.c:1559-1572`). The 3rd bit is NEVER
        // written nor read.
        let mut refs: RefFrameSizes = [None; REF_FRAMES];
        refs[5] = Some((1920, 1080));
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(1); // INTER
            w.write_bit(1); // show_frame = 1
            w.write_bit(0); // error_resilient = 0
            w.write_literal(0, 2); // reset_frame_context = 0
            w.write_literal(0xFF, 8); // refresh_frame_flags
            // Refs: indices [5, 5, 5], sign_biases [false, false, false].
            for _ in 0..3 {
                w.write_literal(5, 3);
                w.write_bit(0);
            }
            // size_in_found bits: skip first ref, inherit second.
            // Parser breaks on the second bit — no third bit written.
            w.write_bit(0);
            w.write_bit(1);
            // No fresh size — inherited from ref 1.
            w.write_bit(0); // no render size
            w.write_bit(0); // allow_high_precision_mv = 0
            // interp_filter = literal index 1 → EightTap.
            w.write_bit(0); // not switchable
            w.write_literal(1, 2); // literal index → EightTap
            write_minimal_tail(w, 1920);
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _) = parse_uncompressed_header(&buf, &refs).expect("valid inter frame");
        assert_eq!(hdr.width, 1920);
        assert_eq!(hdr.height, 1080);
        match hdr.kind {
            FrameKind::InterFrame {
                allow_high_precision_mv,
                interp_filter,
                ..
            } => {
                assert!(!allow_high_precision_mv);
                assert_eq!(interp_filter, InterpFilter::EightTap);
            }
            _ => panic!("expected InterFrame"),
        }
    }

    /// `size_in_found = 1` for a logical ref whose pool slot is empty
    /// surfaces as `InvalidRefFrameSize`. Matches libvpx
    /// `vp9_decodeframe.c:1567-1570` ("Failed to decode frame size").
    #[test]
    fn inter_frame_size_in_found_with_missing_ref_errors() {
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(1);
            w.write_bit(1);
            w.write_bit(0);
            w.write_literal(0, 2);
            w.write_literal(0, 8);
            for _ in 0..3 {
                w.write_literal(0, 3);
                w.write_bit(0);
            }
            // First size_in_found bit set, but pool slot 0 is None →
            // error.
            w.write_bit(1);
            // Padding so the reader doesn't truncate.
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let err = parse_uncompressed_header(&buf, &NO_REFS).unwrap_err();
        assert_eq!(
            err,
            BitstreamError::InvalidRefFrameSize {
                logical_ref: 0,
                pool_slot: 0,
            }
        );
    }

    /// `first_partition_size = 0` triggers `InvalidFirstPartitionSize`
    /// (libvpx :2856-2858).
    #[test]
    fn invalid_first_partition_size_rejected() {
        let buf = build_header(|w| {
            // Minimal keyframe up through render-size = 0.
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(1);
            w.write_bit(0);
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            w.write_literal(ColorSpace::Bt601 as u32, 3);
            w.write_bit(0); // studio
            w.write_literal(319, 16);
            w.write_literal(239, 16);
            w.write_bit(0); // no render size
            // Tail with first_partition_size = 0.
            w.write_bit(0); // refresh_frame_context
            w.write_bit(0); // frame_parallel_decoding_mode
            w.write_literal(0, 2); // frame_context_idx
            w.write_literal(0, 6); // loopfilter filter_level
            w.write_literal(0, 3); // sharpness
            w.write_bit(0); // delta_enabled
            w.write_literal(0, 8); // base_qindex
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0); // segmentation disabled
            w.write_bit(0); // log2_tile_rows = 0 (no col bits for width 320)
            w.write_literal(0, 16); // first_partition_size = 0 → reject
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let err = parse_uncompressed_header(&buf, &NO_REFS).unwrap_err();
        assert_eq!(err, BitstreamError::InvalidFirstPartitionSize);
    }
}
