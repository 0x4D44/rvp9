//! VP9 uncompressed-header struct and parser.
//!
//! Mirrors libvpx `read_uncompressed_header`
//! (`vp9/decoder/vp9_decodeframe.c:2637-2861`) bit-for-bit. This module
//! consumes a byte slice (the start of a frame's payload) and produces a
//! populated [`UncompressedHeader`] plus the byte offset at which the
//! compressed header begins.
//!
//! ## Scope of this commit (Task 1c.1)
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
//! - intra-only frame decoding through the same `setup_frame_size`
//!   point (`:2710-2735`).
//!
//! **Not in this commit (deferred to Task 1d):**
//! - inter-frame ref-index and `setup_frame_size_with_refs` paths,
//! - loop filter / quantization / segmentation sub-parsers,
//! - tile-info parser,
//! - the trailing `first_partition_size` u16,
//! - the compressed-header boolean-coder driver.
//!
//! For now, encountering a non-key / non-show-existing / non-intra-only
//! frame returns [`UncompressedHeader::kind`] =
//! [`FrameKind::InterFrameTodo`] — a sentinel so callers can recognise
//! it and avoid feeding partial state to the rest of the decoder.

use super::error::BitstreamError;
use super::frame::{
    BitDepth, ChromaFormat, ColorRange, ColorSpace, FrameType, Profile, REF_FRAMES, sync,
};
use super::raw::RawBitReader;

/// Decoded uncompressed-header. One value per VP9 frame. Field set
/// depends on `kind`:
///
/// - `ShowExistingFrame`: only `kind` and `slot` (the rest are
///   inherited from the existing frame and left at default).
/// - `KeyFrame` / `IntraOnly`: full color/sampling/size populated.
/// - `InterFrameTodo`: parser bailed early; do not use field values.
#[derive(Clone, Debug)]
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
    /// — full inter-frame parsing is not yet implemented.
    InterFrameTodo,
}

impl FrameKind {
    /// Convenience: is this a KeyFrame or IntraOnly variant?
    #[must_use]
    pub const fn is_intra(self) -> bool {
        matches!(self, Self::KeyFrame | Self::IntraOnly { .. })
    }
}

/// Decode a VP9 uncompressed header from `data`. Returns the populated
/// header and the byte offset (one past the last header byte) at which
/// the compressed-header partition begins.
///
/// **Caveat**: while Task 1d is pending, callers that hit
/// `kind == FrameKind::InterFrameTodo` should treat the returned
/// header as opaque — only `profile` and the byte-stream-position
/// fields are valid.
///
/// Mirrors `read_uncompressed_header` (`vp9_decodeframe.c:2637-2861`).
/// The returned `usize` is the bit-reader's byte cursor at the point
/// where libvpx's parser would call `setup_loopfilter`.
pub fn parse_uncompressed_header(
    data: &[u8],
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
    // bit-depth / color / size. libvpx :2663-2679.
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
            },
            rb.bytes_consumed(),
        ));
    }

    // frame_type (1) + show_frame (1) + error_resilient (1).
    // libvpx :2681-2683.
    let frame_type = FrameType::from_bit(rb.read_bit()?);
    let show_frame = rb.read_bit()? != 0;
    let error_resilient = rb.read_bit()? != 0;

    if frame_type == FrameType::Key {
        // Keyframe path. libvpx :2685-2703.
        read_sync_code(&mut rb)?;
        let color = read_bitdepth_colorspace_sampling(&mut rb, profile)?;
        let (width, height) = read_frame_size(&mut rb)?;
        let render_size = read_render_size(&mut rb, width, height)?;
        return Ok((
            UncompressedHeader {
                kind: FrameKind::KeyFrame,
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
            },
            rb.bytes_consumed(),
        ));
    }

    // Inter-frame branch. libvpx :2704-2766.
    let intra_only = if show_frame {
        false
    } else {
        rb.read_bit()? != 0
    };
    let _reset_frame_context = if error_resilient {
        0
    } else {
        rb.read_literal(2)? as u8
    };

    if intra_only {
        // Intra-only inter frame: sync code + maybe color config +
        // refresh_frame_flags + frame_size. libvpx :2710-2735.
        read_sync_code(&mut rb)?;
        let color = if profile != Profile::Profile0 {
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
        let (width, height) = read_frame_size(&mut rb)?;
        let render_size = read_render_size(&mut rb, width, height)?;
        return Ok((
            UncompressedHeader {
                kind: FrameKind::IntraOnly {
                    reset_frame_context: _reset_frame_context,
                    refresh_frame_flags,
                },
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
            },
            rb.bytes_consumed(),
        ));
    }

    // Full inter-frame parse — Task 1d. Return sentinel.
    Ok((
        UncompressedHeader {
            kind: FrameKind::InterFrameTodo,
            profile,
            show_frame,
            error_resilient,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Unknown,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 0,
            height: 0,
            render_size: None,
        },
        rb.bytes_consumed(),
    ))
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

    #[test]
    fn invalid_frame_marker_rejected() {
        let buf = build_header(|w| {
            // Wrong marker — should be 0b10 = 2.
            w.write_literal(0b11, 2);
            // Pad enough bytes for the reader to attempt the next reads.
            w.write_literal(0, 8);
        });
        let err = parse_uncompressed_header(&buf).unwrap_err();
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
        let err = parse_uncompressed_header(&buf).unwrap_err();
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
        let (hdr, _) = parse_uncompressed_header(&buf).expect("valid header");
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
            // Pad.
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _cursor) = parse_uncompressed_header(&buf).expect("valid keyframe");
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
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _) = parse_uncompressed_header(&buf).expect("valid keyframe");
        assert_eq!(hdr.kind, FrameKind::KeyFrame);
        assert_eq!(hdr.width, 1920);
        assert_eq!(hdr.height, 1080);
        assert_eq!(hdr.color_space, ColorSpace::Bt709);
        assert_eq!(hdr.color_range, ColorRange::Full);
        assert_eq!(hdr.render_size, Some((640, 360)));
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
        let err = parse_uncompressed_header(&buf).unwrap_err();
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
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _) = parse_uncompressed_header(&buf).expect("valid intra-only");
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
    fn inter_frame_returns_todo_sentinel() {
        // frame_type = INTER (1), show_frame = 1 → intra_only forced to
        // 0; full inter parse not yet implemented.
        let buf = build_header(|w| {
            w.write_literal(sync::FRAME_MARKER as u32, 2);
            w.write_bit(0);
            w.write_bit(0);
            w.write_bit(0); // show_existing = 0
            w.write_bit(1); // frame_type = INTER
            w.write_bit(1); // show_frame = 1
            w.write_bit(0); // error_resilient = 0
            // reset_frame_context (2 bits) — not error_resilient → read.
            w.write_literal(0, 2);
            for _ in 0..8 {
                w.write_literal(0, 8);
            }
        });
        let (hdr, _) = parse_uncompressed_header(&buf).expect("valid header prefix");
        assert_eq!(hdr.kind, FrameKind::InterFrameTodo);
        assert_eq!(hdr.profile, Profile::Profile0);
    }
}
