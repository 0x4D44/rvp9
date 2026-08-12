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
use super::raw::{RawBitReader, RawBitWriter};
use super::sub_parsers::{
    LoopFilterParams, QuantizationParams, SegmentationParams, TileInfo, parse_loopfilter,
    parse_quantization, parse_segmentation, parse_tile_info, write_loopfilter, write_quantization,
    write_segmentation, write_tile_info,
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

// ============================================================================
//                              Writer side
// ============================================================================
//
// `write_uncompressed_header` and friends are the encoder mirrors of the
// `parse_*` helpers above. Each writer is a near-line-by-line inverse
// of its reader sibling — same field order, same conditional structure.
// libvpx reference: `write_uncompressed_header` at
// `vp9_bitstream.c:1243-1314`.

/// Write the VP9 uncompressed header for the supplied
/// [`UncompressedHeader`] into `wb`. Parser inverse of
/// [`parse_uncompressed_header`].
///
/// `first_partition_size` is taken as an explicit argument rather than
/// from `h.tail.first_partition_size`: the caller must compute the
/// final compressed-header byte length (after writing it) and pass it
/// in here. This mirrors libvpx's two-pass arrangement where the
/// uncompressed header is patched after the compressed header is
/// finalised (`vp9_bitstream.c:1442-1445`).
///
/// # Errors
///
/// Returns [`BitstreamError::BufferFull`] if the underlying writer
/// overflows, or [`BitstreamError::UnsupportedProfile`] if `h.profile`
/// is not [`Profile::Profile0`] (the only profile rvp9 v1 emits).
pub fn write_uncompressed_header(
    wb: &mut RawBitWriter<'_>,
    h: &UncompressedHeader,
    first_partition_size: u16,
) -> Result<(), BitstreamError> {
    if h.profile != Profile::Profile0 {
        return Err(BitstreamError::UnsupportedProfile {
            profile: h.profile as u8,
        });
    }

    // 2-bit frame_marker. libvpx :1248.
    wb.write_literal(u32::from(sync::FRAME_MARKER), 2);

    // Profile encoding. libvpx :1250 / `write_profile` :1206-1217.
    write_profile_raw(wb, h.profile);

    // show_existing_frame branch. libvpx :1253-1257.
    if let FrameKind::ShowExistingFrame { slot } = h.kind {
        wb.write_bit(1);
        wb.write_literal(u32::from(slot), 3);
        return finalise(wb);
    }
    wb.write_bit(0);

    // frame_type / show_frame / error_resilient. libvpx :1259-1261.
    let frame_type = match h.kind {
        FrameKind::KeyFrame => FrameType::Key,
        FrameKind::IntraOnly { .. } | FrameKind::InterFrame { .. } => FrameType::Inter,
        FrameKind::ShowExistingFrame { .. } => unreachable!("handled above"),
    };
    wb.write_bit(frame_type as u8);
    wb.write_bit(u8::from(h.show_frame));
    wb.write_bit(u8::from(h.error_resilient));

    let color = ColorConfig {
        bit_depth: h.bit_depth,
        color_space: h.color_space,
        color_range: h.color_range,
        chroma: h.chroma,
    };

    match h.kind {
        FrameKind::KeyFrame => {
            // Keyframe path. libvpx :1263-1266.
            write_sync_code(wb);
            write_bitdepth_colorspace_sampling(wb, h.profile, &color);
            write_frame_size(wb, h.width, h.height);
            write_render_size(wb, h.render_size);
        }
        FrameKind::IntraOnly {
            reset_frame_context,
            refresh_frame_flags,
        } => {
            // libvpx :1267-1282 (inter branch + intra_only=1).
            // intra_only bit only emitted when show_frame == 0.
            if !h.show_frame {
                wb.write_bit(1);
            }
            if !h.error_resilient {
                wb.write_literal(u32::from(reset_frame_context), 2);
            }
            write_sync_code(wb);
            // Profile 0 inherits color defaults — no color bits emitted.
            // libvpx :1277-1279.
            if h.profile != Profile::Profile0 {
                write_bitdepth_colorspace_sampling(wb, h.profile, &color);
            }
            wb.write_literal(u32::from(refresh_frame_flags), REF_FRAMES as u8);
            write_frame_size(wb, h.width, h.height);
            write_render_size(wb, h.render_size);
        }
        FrameKind::InterFrame {
            reset_frame_context,
            refresh_frame_flags,
            ref_frame_indices,
            ref_frame_sign_biases,
            allow_high_precision_mv,
            interp_filter,
        } => {
            // libvpx :1267-1299 (inter branch + intra_only=0).
            if !h.show_frame {
                // intra_only = 0 — only emitted when show_frame == 0.
                wb.write_bit(0);
            }
            if !h.error_resilient {
                wb.write_literal(u32::from(reset_frame_context), 2);
            }
            wb.write_literal(u32::from(refresh_frame_flags), REF_FRAMES as u8);
            for i in 0..MAX_INTER_REFS {
                wb.write_literal(u32::from(ref_frame_indices[i]), REF_FRAMES_LOG2 as u8);
                wb.write_bit(u8::from(ref_frame_sign_biases[i]));
            }
            // `setup_frame_size_with_refs` inverse: emit three
            // "size_in_found = 0" bits and a fresh 32-bit frame size.
            // This is always valid — the encoder may also inherit size
            // from a ref by setting one of these bits to 1 and skipping
            // the fresh size; for v1 we always emit fresh size so the
            // writer doesn't need access to the reference-pool sizes.
            // libvpx :1166-1198 (`write_frame_size_with_refs`).
            for _ in 0..MAX_INTER_REFS {
                wb.write_bit(0);
            }
            wb.write_literal(h.width - 1, 16);
            wb.write_literal(h.height - 1, 16);
            write_render_size(wb, h.render_size);
            wb.write_bit(u8::from(allow_high_precision_mv));
            write_interp_filter(wb, interp_filter);
        }
        FrameKind::ShowExistingFrame { .. } => unreachable!("handled above"),
    }

    // ----- Shared post-disposition tail. libvpx :1302-1313. -----
    let tail = h.tail.as_ref().ok_or(BitstreamError::BufferFull)?;
    // ^ `None` tail only occurs for ShowExistingFrame (returned above);
    // the encoder cannot fabricate the sub-block contents from nothing,
    // so we reject with BufferFull as a stable "caller error" signal.

    if !h.error_resilient {
        wb.write_bit(u8::from(tail.refresh_frame_context));
        wb.write_bit(u8::from(tail.frame_parallel_decoding_mode));
    }
    wb.write_literal(u32::from(tail.frame_context_idx), FRAME_CONTEXTS_LOG2);

    write_loopfilter(wb, &tail.loop_filter)?;
    write_quantization(wb, &tail.quantization)?;
    write_segmentation(wb, &tail.segmentation)?;
    let mi_cols = ((h.width + 7) >> 3) as i32;
    write_tile_info(wb, &tail.tile_info, mi_cols)?;

    // Trailing first_partition_size — 16 bits. libvpx writes this at
    // pack time after computing the compressed-header length.
    wb.write_literal(u32::from(first_partition_size), 16);

    finalise(wb)
}

/// Helper: turn writer-poison into a `Result`.
fn finalise(wb: &RawBitWriter<'_>) -> Result<(), BitstreamError> {
    if wb.has_error() {
        Err(BitstreamError::BufferFull)
    } else {
        Ok(())
    }
}

/// Encoder mirror of [`read_profile_raw`]. Mirrors libvpx
/// `write_profile` (`vp9_bitstream.c:1206-1217`). The wire encoding
/// puts the **low** bit of the profile value first; for Profile 0 that
/// reduces to writing two zero bits.
fn write_profile_raw(wb: &mut RawBitWriter<'_>, profile: Profile) {
    match profile {
        Profile::Profile0 => wb.write_literal(0, 2),
        Profile::Profile1 => wb.write_literal(2, 2),
        Profile::Profile2 => wb.write_literal(1, 2),
        Profile::Profile3 => wb.write_literal(6, 3),
    }
}

/// Encoder mirror of [`read_sync_code`]. Mirrors libvpx
/// `write_sync_code` (`vp9_bitstream.c:1200-1204`).
fn write_sync_code(wb: &mut RawBitWriter<'_>) {
    wb.write_literal(u32::from(sync::SYNC_CODE_0), 8);
    wb.write_literal(u32::from(sync::SYNC_CODE_1), 8);
    wb.write_literal(u32::from(sync::SYNC_CODE_2), 8);
}

/// Encoder mirror of [`read_bitdepth_colorspace_sampling`]. Mirrors
/// libvpx `write_bitdepth_colorspace_sampling`
/// (`vp9_bitstream.c:1219-1241`).
fn write_bitdepth_colorspace_sampling(
    wb: &mut RawBitWriter<'_>,
    profile: Profile,
    color: &ColorConfig,
) {
    if matches!(profile, Profile::Profile2 | Profile::Profile3) {
        // 1 bit: 0 → 10, 1 → 12. libvpx :1223.
        wb.write_bit(u8::from(color.bit_depth == BitDepth::Twelve));
    }
    wb.write_literal(color.color_space as u32, 3);
    if color.color_space != ColorSpace::Srgb {
        wb.write_bit(color.color_range as u8);
        if matches!(profile, Profile::Profile1 | Profile::Profile3) {
            wb.write_bit(u8::from(color.chroma.subsampling_x()));
            wb.write_bit(u8::from(color.chroma.subsampling_y()));
            wb.write_bit(0); // reserved
        }
        // Profile 0/2 force 4:2:0 — no subsampling bits emitted.
    } else {
        // sRGB. libvpx :1239 writes a single reserved 0 bit.
        wb.write_bit(0);
    }
}

/// Encoder mirror of [`read_frame_size`]. Mirrors libvpx
/// `write_frame_size` (`vp9_bitstream.c:1158-1164`) **minus the
/// trailing `write_render_size`** — render size is emitted by the
/// caller so that the inter-frame branch can place it after the
/// inheritance bits.
fn write_frame_size(wb: &mut RawBitWriter<'_>, width: u32, height: u32) {
    wb.write_literal(width - 1, 16);
    wb.write_literal(height - 1, 16);
}

/// Encoder mirror of [`read_render_size`]. Mirrors libvpx
/// `write_render_size` (`vp9_bitstream.c:1147-1156`).
fn write_render_size(wb: &mut RawBitWriter<'_>, render: Option<(u32, u32)>) {
    match render {
        Some((rw, rh)) => {
            wb.write_bit(1);
            wb.write_literal(rw - 1, 16);
            wb.write_literal(rh - 1, 16);
        }
        None => wb.write_bit(0),
    }
}

/// Encoder mirror of [`read_interp_filter`]. Mirrors libvpx
/// `write_interp_filter` (`vp9_bitstream.c:854-861`). The
/// `filter_to_literal` map `[1, 0, 2, 3]` is the inverse of the
/// reader's `literal_to_filter` map; see [`InterpFilter::from_literal`].
fn write_interp_filter(wb: &mut RawBitWriter<'_>, filter: InterpFilter) {
    if filter == InterpFilter::Switchable {
        wb.write_bit(1);
    } else {
        wb.write_bit(0);
        let literal = match filter {
            InterpFilter::EightTapSmooth => 0,
            InterpFilter::EightTap => 1,
            InterpFilter::EightTapSharp => 2,
            InterpFilter::Bilinear => 3,
            InterpFilter::Switchable => unreachable!("handled above"),
        };
        wb.write_literal(literal, 2);
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
        let tail = hdr
            .tail
            .expect("non-show-existing frames must carry a tail");
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
            BitstreamError::InvalidSyncCode {
                found: [0xff, 0x83, 0x42]
            }
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

    // ====================================================================
    //                       write_uncompressed_header
    // ====================================================================

    use crate::bitstream::sub_parsers::{
        LoopFilterParams, MAX_PROB, PREDICTION_PROBS, QuantizationParams, SEG_TREE_PROBS,
        SegmentationParams, TileInfo,
    };

    /// Build a minimal valid `HeaderTail` for round-trip tests. All
    /// sub-blocks at defaults; `first_partition_size` stored as 1 (the
    /// writer ignores this — caller passes its own).
    ///
    /// The segmentation `tree_probs` / `pred_probs` arrays are seeded
    /// with [`MAX_PROB`] to match what `parse_segmentation` produces
    /// when `enabled = false` (the parser initialises the arrays to
    /// MAX_PROB *before* the early return; `Default::default()` would
    /// give all zeros and the round-trip would fail).
    fn default_tail() -> HeaderTail {
        HeaderTail {
            refresh_frame_context: false,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams::default(),
            quantization: QuantizationParams::default(),
            segmentation: SegmentationParams {
                tree_probs: [MAX_PROB; SEG_TREE_PROBS],
                pred_probs: [MAX_PROB; PREDICTION_PROBS],
                ..Default::default()
            },
            tile_info: TileInfo::default(),
            first_partition_size: 1,
        }
    }

    /// Write `hdr` to a new buffer, parse it back via
    /// [`parse_uncompressed_header`], and return the parsed header.
    /// Asserts the parsed result equals the input.
    fn round_trip(hdr: &UncompressedHeader, refs: &RefFrameSizes, fp_size: u16) {
        let mut buf = vec![0u8; 256];
        let used = {
            let mut wb = RawBitWriter::new(&mut buf);
            write_uncompressed_header(&mut wb, hdr, fp_size).expect("write succeeds");
            wb.finish().expect("no overflow")
        };
        buf.truncate(used + 8); // padding so the byte-cursor reader has room
        let (got, _cursor) = parse_uncompressed_header(&buf, refs).expect("parse succeeds");

        // The parser produces fresh `first_partition_size` from the wire;
        // patch the input copy to match before comparing the rest.
        let mut expected = hdr.clone();
        if let Some(t) = expected.tail.as_mut() {
            t.first_partition_size = fp_size;
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn write_show_existing_frame_all_slots() {
        for slot in 0u8..8 {
            let hdr = UncompressedHeader {
                kind: FrameKind::ShowExistingFrame { slot },
                profile: Profile::Profile0,
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
            };
            round_trip(&hdr, &NO_REFS, 0); // fp_size unused for show-existing
        }
    }

    fn make_keyframe(width: u32, height: u32) -> UncompressedHeader {
        UncompressedHeader {
            kind: FrameKind::KeyFrame,
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width,
            height,
            render_size: None,
            tail: Some(default_tail()),
        }
    }

    #[test]
    fn write_keyframe_common_resolutions() {
        for (w, h) in [(320u32, 240u32), (1280, 720), (1920, 1080)] {
            round_trip(&make_keyframe(w, h), &NO_REFS, 1);
        }
    }

    #[test]
    fn write_keyframe_error_resilient() {
        let mut hdr = make_keyframe(640, 480);
        hdr.error_resilient = true;
        // When error_resilient is set, the parser forces fpdm = true.
        if let Some(t) = hdr.tail.as_mut() {
            t.refresh_frame_context = false;
            t.frame_parallel_decoding_mode = true;
        }
        round_trip(&hdr, &NO_REFS, 17);
    }

    #[test]
    fn write_keyframe_with_render_size() {
        let mut hdr = make_keyframe(1920, 1080);
        hdr.color_space = ColorSpace::Bt709;
        hdr.color_range = ColorRange::Full;
        hdr.render_size = Some((640, 360));
        round_trip(&hdr, &NO_REFS, 42);
    }

    #[test]
    fn write_intra_only_refresh_flags() {
        for rff in [0x00u8, 0x01, 0xA3, 0xFF] {
            let hdr = UncompressedHeader {
                kind: FrameKind::IntraOnly {
                    reset_frame_context: 0,
                    refresh_frame_flags: rff,
                },
                profile: Profile::Profile0,
                show_frame: false, // intra-only requires !show_frame to write the intra_only bit
                error_resilient: false,
                bit_depth: BitDepth::Eight,
                // Profile-0 intra-only inherits these defaults (libvpx :2721).
                color_space: ColorSpace::Bt601,
                color_range: ColorRange::Studio,
                chroma: ChromaFormat::Yuv420,
                width: 320,
                height: 240,
                render_size: None,
                tail: Some(default_tail()),
            };
            round_trip(&hdr, &NO_REFS, 7);
        }
    }

    fn make_inter(filter: InterpFilter, high_prec: bool) -> UncompressedHeader {
        UncompressedHeader {
            kind: FrameKind::InterFrame {
                reset_frame_context: 0,
                refresh_frame_flags: 0b0101_0101,
                ref_frame_indices: [0, 1, 2],
                ref_frame_sign_biases: [false, true, false],
                allow_high_precision_mv: high_prec,
                interp_filter: filter,
            },
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            // Inter frames inherit color from prior KEY/intra-only; the
            // parser populates them with defaults (lines 363-368). The
            // writer doesn't emit them, so we must use those same
            // defaults in the input here for the round-trip to match.
            color_space: ColorSpace::Unknown,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 640,
            height: 480,
            render_size: None,
            tail: Some(default_tail()),
        }
    }

    #[test]
    fn write_inter_frame_all_interp_filters() {
        for filter in [
            InterpFilter::EightTap,
            InterpFilter::EightTapSmooth,
            InterpFilter::EightTapSharp,
            InterpFilter::Bilinear,
            InterpFilter::Switchable,
        ] {
            round_trip(&make_inter(filter, false), &NO_REFS, 1);
        }
    }

    #[test]
    fn write_inter_frame_high_precision_mv() {
        round_trip(&make_inter(InterpFilter::Switchable, true), &NO_REFS, 1);
    }

    /// Parse the first keyframe out of `vp90-2-05-resize.ivf`, write
    /// it back via `write_uncompressed_header`, then parse the result
    /// again and assert the second parse equals the first. A clean
    /// checkout reports this optional oracle as ignored until the
    /// gitignored conformance vectors have been fetched.
    #[cfg_attr(
        not(rvp9_conformance_vectors),
        ignore = "requires scripts/fetch-vectors.sh"
    )]
    #[test]
    fn captured_header_round_trip() {
        // IVF: 32-byte file header, then per-frame (4-byte size LE +
        // 8-byte pts + payload). We just grab the first frame.
        let vector = "../../tests/conformance/vectors/vp90-2-05-resize.ivf";
        let bytes = std::fs::read(vector)
            .unwrap_or_else(|error| panic!("failed to read test vector {vector}: {error}"));
        assert_eq!(&bytes[..4], b"DKIF", "not an IVF file");
        let frame_size =
            u32::from_le_bytes(bytes[32..36].try_into().expect("4 bytes available")) as usize;
        let payload = &bytes[44..44 + frame_size];

        // First frame of a libvpx-encoded IVF is always a key frame.
        let (parsed, _) =
            parse_uncompressed_header(payload, &NO_REFS).expect("captured keyframe parses");
        assert_eq!(parsed.kind, FrameKind::KeyFrame);

        // Round-trip through the writer.
        let fp_size = parsed
            .tail
            .as_ref()
            .map(|t| t.first_partition_size)
            .expect("keyframe must carry a tail");
        round_trip(&parsed, &NO_REFS, fp_size);
    }

    // ------------------------------------------------------------------
    // Stage 9 — branch coverage gaps in bitstream/header.rs
    // ------------------------------------------------------------------

    /// Profile 1 keyframe: hits the `Profile::Profile1` branch in
    /// `read_bitdepth_colorspace_sampling`'s subsampling / color-config
    /// section. Profile 1 reads two subsampling bits plus a reserved 0
    /// bit when color_space != Srgb.
    #[test]
    fn keyframe_profile1_4_2_2_subsampling() {
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            // Profile 1: low=1, high=0 → raw = 1.
            w.write_bit(1);
            w.write_bit(0);
            w.write_bit(0); // show_existing = 0
            w.write_bit(0); // frame_type = KEY
            w.write_bit(1); // show_frame = 1
            w.write_bit(0); // error_resilient = 0
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            // Profile 1: no bit_depth bit (Profile 0/1 → 8-bit fixed).
            w.write_literal(ColorSpace::Bt601 as u32, 3); // color_space != sRGB
            w.write_bit(0); // color_range = Studio
            // Profile 1: ss_x, ss_y, reserved=0
            w.write_bit(1); // ss_x = 1
            w.write_bit(0); // ss_y = 0 → 4:2:2
            w.write_bit(0); // reserved
            // frame size
            w.write_literal(319, 16);
            w.write_literal(239, 16);
            w.write_bit(0); // no render size
            write_minimal_tail(w, 320);
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _) = parse_uncompressed_header(&buf, &NO_REFS).expect("valid Profile 1");
        assert_eq!(hdr.profile, Profile::Profile1);
        assert_eq!(hdr.bit_depth, BitDepth::Eight);
        assert_eq!(hdr.chroma, ChromaFormat::Yuv422);
        assert_eq!(hdr.color_range, ColorRange::Studio);
    }

    /// Profile 2 keyframe with bit_depth = 10: exercises the
    /// `Profile2 | Profile3` HBD branch reading 1 bit (`ten_or_twelve`)
    /// to pick between `BitDepth::Ten` (0) and `BitDepth::Twelve` (1).
    #[test]
    fn keyframe_profile2_bit_depth_10() {
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            // Profile 2: low=0, high=1 → raw = 2.
            w.write_bit(0);
            w.write_bit(1);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(1);
            w.write_bit(0);
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            w.write_bit(0); // ten_or_twelve = 0 → BitDepth::Ten
            w.write_literal(ColorSpace::Bt709 as u32, 3);
            w.write_bit(0); // color_range = Studio
            // Profile 2: no subsampling bits (forced to 4:2:0).
            w.write_literal(319, 16);
            w.write_literal(239, 16);
            w.write_bit(0);
            write_minimal_tail(w, 320);
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _) = parse_uncompressed_header(&buf, &NO_REFS).expect("valid Profile 2");
        assert_eq!(hdr.profile, Profile::Profile2);
        assert_eq!(hdr.bit_depth, BitDepth::Ten);
        assert_eq!(hdr.chroma, ChromaFormat::Yuv420);
    }

    /// Profile 3 keyframe with bit_depth = 12: hits the `ten_or_twelve
    /// = 1` arm. Profile 3 also reads subsampling bits since it's
    /// `Profile1 | Profile3` for that branch.
    #[test]
    fn keyframe_profile3_bit_depth_12_4_4_4() {
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            // Profile 3: low=1, high=1, extra=0 → raw=3.
            w.write_bit(1);
            w.write_bit(1);
            w.write_bit(0); // extra bit (only read when profile > 2)
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(1);
            w.write_bit(0);
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            w.write_bit(1); // ten_or_twelve = 1 → BitDepth::Twelve
            w.write_literal(ColorSpace::Bt709 as u32, 3);
            w.write_bit(1); // color_range = Full
            w.write_bit(0); // ss_x = 0
            w.write_bit(0); // ss_y = 0 → 4:4:4
            w.write_bit(0); // reserved
            w.write_literal(319, 16);
            w.write_literal(239, 16);
            w.write_bit(0);
            write_minimal_tail(w, 320);
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _) = parse_uncompressed_header(&buf, &NO_REFS).expect("valid Profile 3");
        assert_eq!(hdr.profile, Profile::Profile3);
        assert_eq!(hdr.bit_depth, BitDepth::Twelve);
        assert_eq!(hdr.chroma, ChromaFormat::Yuv444);
        assert_eq!(hdr.color_range, ColorRange::Full);
    }

    /// Profile 1 keyframe with `color_space = Srgb`: triggers the
    /// sRGB branch, which requires Profile 1/3 + 4:4:4 + reserved 0
    /// bit. Pins the Srgb true-arm path.
    #[test]
    fn keyframe_profile1_srgb_color_space() {
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            // Profile 1: low=1, high=0.
            w.write_bit(1);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(1);
            w.write_bit(0);
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            w.write_literal(ColorSpace::Srgb as u32, 3);
            // sRGB branch: 1 reserved bit = 0.
            w.write_bit(0);
            w.write_literal(319, 16);
            w.write_literal(239, 16);
            w.write_bit(0);
            write_minimal_tail(w, 320);
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _) = parse_uncompressed_header(&buf, &NO_REFS).expect("valid sRGB Profile 1");
        assert_eq!(hdr.color_space, ColorSpace::Srgb);
        assert_eq!(hdr.color_range, ColorRange::Full); // sRGB forces full
        assert_eq!(hdr.chroma, ChromaFormat::Yuv444); // sRGB forces 4:4:4
    }

    /// Profile 0 with `color_space = Srgb` is invalid — sRGB requires
    /// Profile 1/3. Surface as `InconsistentSrgb`.
    #[test]
    fn keyframe_profile0_srgb_rejected_as_inconsistent() {
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            // Profile 0.
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(1);
            w.write_bit(0);
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            w.write_literal(ColorSpace::Srgb as u32, 3);
            // No further color/range bits; the parser must reject
            // BEFORE reading the reserved bit.
            for _ in 0..16 {
                w.write_literal(0, 8);
            }
        });
        let err = parse_uncompressed_header(&buf, &NO_REFS).unwrap_err();
        assert_eq!(err, BitstreamError::InconsistentSrgb);
    }

    /// Profile 1 + sRGB with a non-zero reserved bit triggers
    /// `InconsistentSrgb`. Hits the second `InconsistentSrgb` arm.
    #[test]
    fn keyframe_profile1_srgb_reserved_bit_one_rejected() {
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(1); // low
            w.write_bit(0); // high → Profile 1
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(1);
            w.write_bit(0);
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            w.write_literal(ColorSpace::Srgb as u32, 3);
            w.write_bit(1); // reserved bit = 1 → reject
            for _ in 0..16 {
                w.write_literal(0, 8);
            }
        });
        let err = parse_uncompressed_header(&buf, &NO_REFS).unwrap_err();
        assert_eq!(err, BitstreamError::InconsistentSrgb);
    }

    /// Profile 1 non-sRGB with a non-zero reserved bit (in the
    /// subsampling block) triggers `InconsistentSrgb`. Pins the third
    /// `InconsistentSrgb` arm.
    #[test]
    fn keyframe_profile1_non_srgb_reserved_bit_one_rejected() {
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(1); // low
            w.write_bit(0); // high → Profile 1
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(1);
            w.write_bit(0);
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            w.write_literal(ColorSpace::Bt709 as u32, 3);
            w.write_bit(0); // color_range
            w.write_bit(0); // ss_x
            w.write_bit(0); // ss_y
            w.write_bit(1); // reserved → reject
            for _ in 0..16 {
                w.write_literal(0, 8);
            }
        });
        let err = parse_uncompressed_header(&buf, &NO_REFS).unwrap_err();
        assert_eq!(err, BitstreamError::InconsistentSrgb);
    }

    /// `error_resilient = 1` on a keyframe forces `(rfc = false,
    /// fpdm = true)` in the shared post-disposition tail. Exercises
    /// the `if error_resilient` true branch (the false branch is
    /// covered by every other keyframe test).
    #[test]
    fn keyframe_error_resilient_forces_fpdm_true() {
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(1);
            w.write_bit(1); // error_resilient = 1
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            w.write_literal(ColorSpace::Bt601 as u32, 3);
            w.write_bit(0);
            w.write_literal(319, 16);
            w.write_literal(239, 16);
            w.write_bit(0);
            // Skip the rfc/fpdm bits since error_resilient forces them.
            // frame_context_idx (2 bits)
            w.write_literal(0, 2);
            // loopfilter / quant / segmentation / tile / fp_size
            w.write_literal(0, 6);
            w.write_literal(0, 3);
            w.write_bit(0);
            w.write_literal(0, 8);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0); // log2_tile_rows = 0 (no col bits at 320)
            w.write_literal(1, 16);
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _) = parse_uncompressed_header(&buf, &NO_REFS).expect("valid keyframe");
        let tail = hdr.tail.expect("tail must be present");
        assert!(!tail.refresh_frame_context);
        assert!(tail.frame_parallel_decoding_mode);
        assert!(hdr.error_resilient);
    }

    /// `read_interp_filter` literal=0 → `InterpFilter::EightTapSmooth`.
    /// `read_interp_filter` literal=2 → `InterpFilter::EightTapSharp`.
    /// `read_interp_filter` literal=3 → `InterpFilter::Bilinear`.
    /// These pin the non-Switchable branch of `read_interp_filter`'s
    /// literal table.
    #[test]
    fn inter_frame_interp_filter_all_literals() {
        for (literal, expected) in [
            (0u32, InterpFilter::EightTapSmooth),
            (1u32, InterpFilter::EightTap),
            (2u32, InterpFilter::EightTapSharp),
            (3u32, InterpFilter::Bilinear),
        ] {
            let buf = build_header(|w| {
                w.write_literal(sync::FRAME_MARKER as u32, 2);
                w.write_bit(0); // Profile 0
                w.write_bit(0);
                w.write_bit(0);
                w.write_bit(1); // INTER
                w.write_bit(1); // show_frame
                w.write_bit(0); // !error_resilient
                w.write_literal(0, 2); // reset_frame_context
                w.write_literal(0, 8); // refresh_frame_flags
                for _ in 0..3 {
                    w.write_literal(0, 3); // ref_idx
                    w.write_bit(0); // sign_bias
                }
                w.write_bit(0); // size_in_found[0]
                w.write_bit(0); // [1]
                w.write_bit(0); // [2]
                w.write_literal(319, 16); // fresh width
                w.write_literal(239, 16);
                w.write_bit(0); // no render
                w.write_bit(0); // !allow_high_precision_mv
                w.write_bit(0); // not switchable
                w.write_literal(literal, 2); // interp literal
                write_minimal_tail(w, 320);
                for _ in 0..8 {
                    w.write_literal(0, 8);
                }
            });
            let (hdr, _) = parse_uncompressed_header(&buf, &NO_REFS).expect("valid inter");
            match hdr.kind {
                FrameKind::InterFrame { interp_filter, .. } => {
                    assert_eq!(interp_filter, expected, "literal {literal}");
                }
                _ => panic!("expected InterFrame"),
            }
        }
    }

    /// Round-trip a Profile 0 inter frame with `show_frame = false` —
    /// hits the `if !h.show_frame { wb.write_bit(0); }` arm of the
    /// inter writer (the show_frame=true case is the existing default).
    #[test]
    fn write_inter_frame_show_frame_false_round_trip() {
        let mut hdr = make_inter(InterpFilter::EightTap, false);
        hdr.show_frame = false;
        round_trip(&hdr, &NO_REFS, 7);
    }

    /// Round-trip an intra-only frame with `error_resilient = true` —
    /// the writer skips emitting the `reset_frame_context` bits.
    /// Also exercises the post-tail `error_resilient` arm that forces
    /// `fpdm = true`.
    #[test]
    fn write_intra_only_error_resilient_round_trip() {
        let hdr = UncompressedHeader {
            kind: FrameKind::IntraOnly {
                reset_frame_context: 0,
                refresh_frame_flags: 0x0F,
            },
            profile: Profile::Profile0,
            show_frame: false,
            error_resilient: true,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 320,
            height: 240,
            render_size: None,
            tail: Some(HeaderTail {
                refresh_frame_context: false,
                frame_parallel_decoding_mode: true, // forced when error_resilient
                ..default_tail()
            }),
        };
        round_trip(&hdr, &NO_REFS, 3);
    }

    /// Profile 2 intra-only frame: hits the `profile != Profile::Profile0`
    /// true branch in the intra-only color-config read (the parser reads
    /// a full color config instead of using the Profile 0 defaults).
    #[test]
    fn intra_only_profile2_reads_full_color_config() {
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            // Profile 2: low=0, high=1.
            w.write_bit(0);
            w.write_bit(1);
            w.write_bit(0); // show_existing
            w.write_bit(1); // frame_type INTER
            w.write_bit(0); // show_frame = 0
            w.write_bit(0); // error_resilient = 0
            w.write_bit(1); // intra_only = 1
            w.write_literal(0, 2); // reset_frame_context
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            // Profile 2 color config: ten_or_twelve bit + cs + range
            w.write_bit(1); // ten_or_twelve = 1 → 12-bit
            w.write_literal(ColorSpace::Bt709 as u32, 3);
            w.write_bit(1); // color_range = Full
            // Profile 2: no subsampling bits — forced to 4:2:0.
            w.write_literal(0x5A, 8); // refresh_frame_flags
            w.write_literal(319, 16);
            w.write_literal(239, 16);
            w.write_bit(0); // no render size
            write_minimal_tail(w, 320);
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _) =
            parse_uncompressed_header(&buf, &NO_REFS).expect("valid Profile 2 intra-only");
        assert_eq!(hdr.profile, Profile::Profile2);
        assert_eq!(hdr.bit_depth, BitDepth::Twelve);
        assert_eq!(hdr.color_space, ColorSpace::Bt709);
        assert_eq!(hdr.color_range, ColorRange::Full);
        assert_eq!(hdr.chroma, ChromaFormat::Yuv420);
        match hdr.kind {
            FrameKind::IntraOnly {
                refresh_frame_flags,
                ..
            } => {
                assert_eq!(refresh_frame_flags, 0x5A);
            }
            other => panic!("expected IntraOnly, got {other:?}"),
        }
    }

    /// Round-trip an InterFrame with `error_resilient = true`. Hits
    /// the `if !h.error_resilient` false arm in the InterFrame writer
    /// (the writer skips emitting `reset_frame_context` bits when
    /// error_resilient is set).
    #[test]
    fn write_inter_frame_error_resilient_round_trip() {
        let mut hdr = make_inter(InterpFilter::EightTap, false);
        hdr.error_resilient = true;
        // error_resilient forces fpdm=true on the parser side.
        if let Some(t) = hdr.tail.as_mut() {
            t.refresh_frame_context = false;
            t.frame_parallel_decoding_mode = true;
        }
        round_trip(&hdr, &NO_REFS, 5);
    }

    /// `write_uncompressed_header` returns `BufferFull` if the writer's
    /// underlying buffer overflows. Hits the `if wb.has_error()` true
    /// arm in `finalise`.
    #[test]
    fn write_uncompressed_header_buffer_full_returns_error() {
        let hdr = make_keyframe(320, 240);
        // Use a buffer too small to hold even the smallest valid header.
        let mut buf = vec![0u8; 4];
        let mut wb = RawBitWriter::new(&mut buf);
        let err = write_uncompressed_header(&mut wb, &hdr, 1).unwrap_err();
        assert!(matches!(err, BitstreamError::BufferFull));
    }

    /// Truncated keyframe (cut mid-frame-size) returns `Truncated`
    /// from `read_frame_size`'s `?`. Pins the `?` propagation arm of
    /// `read_frame_size`.
    #[test]
    fn keyframe_truncated_frame_size_returns_truncated() {
        // Stop the buffer right after the sync code (before the
        // 32-bit frame size). The parser's `read_literal(16)` for
        // width should fail.
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(0); // profile low
            w.write_bit(0); // profile high
            w.write_bit(0); // show_existing
            w.write_bit(0); // frame_type KEY
            w.write_bit(1); // show_frame
            w.write_bit(0); // error_resilient
            w.write_literal(sync::SYNC_CODE_0 as u32, 8);
            w.write_literal(sync::SYNC_CODE_1 as u32, 8);
            w.write_literal(sync::SYNC_CODE_2 as u32, 8);
            w.write_literal(ColorSpace::Bt601 as u32, 3);
            w.write_bit(0); // color_range
            // No width/height/render/tail — truncated.
        });
        // Force buffer truncation: don't add trailing padding bytes.
        let err = parse_uncompressed_header(&buf, &NO_REFS).unwrap_err();
        assert!(matches!(err, BitstreamError::Truncated { .. }));
    }

    /// `write_uncompressed_header` rejects non-Profile-0 inputs with
    /// `UnsupportedProfile`. Pins the early-return branch.
    #[test]
    fn write_uncompressed_header_rejects_non_profile_0() {
        let mut hdr = make_keyframe(320, 240);
        hdr.profile = Profile::Profile1;
        let mut buf = vec![0u8; 256];
        let mut wb = RawBitWriter::new(&mut buf);
        let err = write_uncompressed_header(&mut wb, &hdr, 1).unwrap_err();
        assert!(matches!(
            err,
            BitstreamError::UnsupportedProfile { profile: 1 }
        ));
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
