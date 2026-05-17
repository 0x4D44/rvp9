//! VP9 frame-header types — enums and constants used by the uncompressed
//! and compressed header parsers.
//!
//! Mirrors libvpx's per-frame public type surface:
//!
//! - [`Profile`] mirrors `BITSTREAM_PROFILE` (`vp9/common/vp9_enums.h:36`).
//! - [`FrameType`] mirrors `FRAME_TYPE` (`vp9/common/vp9_blockd.h:33-38`).
//! - [`ColorSpace`] mirrors `vpx_color_space_t` (`vpx/vpx_image.h:54-63`).
//! - [`ColorRange`] mirrors `vpx_color_range_t` (`vpx/vpx_image.h:66-69`).
//! - [`BitDepth`] mirrors `vpx_bit_depth_t` (`vpx/vpx_codec.h:218-224`).
//!
//! These are pure data types — no I/O, no boolean coder dependencies.
//! Header parsing (the function that consumes a byte slice and produces
//! a populated header struct) lands in a subsequent commit alongside the
//! `UncompressedHeader` aggregate type.

/// VP9 bitstream profile. Profile 0 is the only profile rvp9 ships in
/// v1 (D-01 RESOLVED — 8-bit 4:2:0 only). Profiles 1/2/3 are kept in
/// the enum for spec faithfulness and so the parser can reject them
/// with a structured error rather than panicking.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum Profile {
    /// Profile 0 — 8-bit, 4:2:0 chroma sub-sampling. **The only profile
    /// rvp9 v1 supports.**
    Profile0 = 0,
    /// Profile 1 — 8-bit, 4:2:2 / 4:4:4 / 4:4:0 chroma. Rejected by
    /// the v1 decoder.
    Profile1 = 1,
    /// Profile 2 — 10-bit / 12-bit, 4:2:0 chroma. Rejected by the v1
    /// decoder (HBD out of scope per D-01).
    Profile2 = 2,
    /// Profile 3 — 10-bit / 12-bit, 4:2:2 / 4:4:4 / 4:4:0 chroma.
    /// Rejected.
    Profile3 = 3,
}

impl Profile {
    /// Profile 0 / 2 use 4:2:0 chroma; 1 / 3 use 4:2:2 / 4:4:4 / 4:4:0.
    /// Per VP9 spec table 7.1.
    #[inline]
    #[must_use]
    pub const fn uses_420_chroma(self) -> bool {
        matches!(self, Self::Profile0 | Self::Profile2)
    }

    /// Decode a 2-bit profile value as read from the uncompressed
    /// header. The profile field in the header is encoded as one of
    /// the four 2-bit values `0..=3`; values outside that range are
    /// rejected by the caller before reaching this function.
    #[inline]
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Profile0),
            1 => Some(Self::Profile1),
            2 => Some(Self::Profile2),
            3 => Some(Self::Profile3),
            _ => None,
        }
    }
}

/// VP9 frame type. Mirrors libvpx `FRAME_TYPE` (`vp9_blockd.h:33-38`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum FrameType {
    /// Keyframe — fully self-contained, starts a new GOP.
    Key = 0,
    /// Inter frame — references previously-decoded frames.
    Inter = 1,
}

impl FrameType {
    /// Decode the 1-bit `frame_type` field from the uncompressed header.
    #[inline]
    #[must_use]
    pub const fn from_bit(b: u8) -> Self {
        match b & 1 {
            0 => Self::Key,
            _ => Self::Inter,
        }
    }
}

/// Output bit depth as conveyed in the uncompressed header. 8 bits in
/// Profile 0/1; 10 or 12 bits in Profile 2/3. rvp9 v1 only emits
/// `Eight`; deeper bit-depths cause the decoder to reject the frame.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum BitDepth {
    /// 8-bit (Profile 0/1).
    Eight = 8,
    /// 10-bit (Profile 2/3, HBD — rejected by v1).
    Ten = 10,
    /// 12-bit (Profile 2/3, HBD — rejected by v1).
    Twelve = 12,
}

/// Color space as conveyed in the uncompressed header (key-frame /
/// intra-only frames only). Mirrors `vpx_color_space_t`
/// (`vpx/vpx_image.h:54-63`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum ColorSpace {
    /// Unknown / unspecified.
    Unknown = 0,
    /// ITU-R BT.601.
    Bt601 = 1,
    /// ITU-R BT.709.
    Bt709 = 2,
    /// SMPTE 170M (functionally identical to BT.601 for VP9).
    Smpte170 = 3,
    /// SMPTE 240M.
    Smpte240 = 4,
    /// ITU-R BT.2020.
    Bt2020 = 5,
    /// Reserved — a VP9 stream signalling this value is technically
    /// malformed but we accept it for robustness.
    Reserved = 6,
    /// sRGB. When this value appears the stream must be Profile 1 or 3
    /// and 4:4:4; the spec enforces a colour-range of full as well.
    Srgb = 7,
}

impl ColorSpace {
    /// Decode a 3-bit color-space value from the uncompressed header.
    #[inline]
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Unknown),
            1 => Some(Self::Bt601),
            2 => Some(Self::Bt709),
            3 => Some(Self::Smpte170),
            4 => Some(Self::Smpte240),
            5 => Some(Self::Bt2020),
            6 => Some(Self::Reserved),
            7 => Some(Self::Srgb),
            _ => None,
        }
    }
}

/// Color range as conveyed in the uncompressed header. Mirrors
/// `vpx_color_range_t` (`vpx/vpx_image.h:66-69`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
#[repr(u8)]
pub enum ColorRange {
    /// Studio / TV range: Y in `[16, 235]`, UV in `[16, 240]`.
    #[default]
    Studio = 0,
    /// Full range: YUV / RGB in `[0, 255]`.
    Full = 1,
}

impl ColorRange {
    /// Decode the 1-bit color-range field.
    #[inline]
    #[must_use]
    pub const fn from_bit(b: u8) -> Self {
        match b & 1 {
            0 => Self::Studio,
            _ => Self::Full,
        }
    }
}

/// Chroma sub-sampling, derived from `Profile` + the
/// `subsampling_x` / `subsampling_y` bits in the uncompressed header.
///
/// VP9 uses the sub-sampling factors `(sx, sy)` from §6.2:
///
/// - `(1, 1)` ⇒ 4:2:0 (chroma sub-sampled 2× horizontally and vertically).
/// - `(1, 0)` ⇒ 4:2:2 (chroma sub-sampled 2× horizontally only).
/// - `(0, 1)` ⇒ 4:4:0 (chroma sub-sampled 2× vertically only).
/// - `(0, 0)` ⇒ 4:4:4 (no chroma sub-sampling).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum ChromaFormat {
    /// 4:2:0 — Profile 0 (and Profile 2 in HBD). The only format rvp9
    /// v1 accepts.
    Yuv420 = 0,
    /// 4:2:2 — Profile 1 / 3.
    Yuv422 = 1,
    /// 4:4:0 — Profile 1 / 3.
    Yuv440 = 2,
    /// 4:4:4 — Profile 1 / 3.
    Yuv444 = 3,
}

impl ChromaFormat {
    /// Construct from raw `(subsampling_x, subsampling_y)` bits.
    #[inline]
    #[must_use]
    pub const fn from_subsampling(sx: bool, sy: bool) -> Self {
        match (sx, sy) {
            (true, true) => Self::Yuv420,
            (true, false) => Self::Yuv422,
            (false, true) => Self::Yuv440,
            (false, false) => Self::Yuv444,
        }
    }

    /// Horizontal chroma sub-sampling factor (1 → 4:2:x; 0 → 4:4:x).
    #[inline]
    #[must_use]
    pub const fn subsampling_x(self) -> bool {
        matches!(self, Self::Yuv420 | Self::Yuv422)
    }

    /// Vertical chroma sub-sampling factor.
    #[inline]
    #[must_use]
    pub const fn subsampling_y(self) -> bool {
        matches!(self, Self::Yuv420 | Self::Yuv440)
    }
}

/// Reference-frame index. VP9 maintains a slot pool of
/// `REF_FRAMES` (= 8) decoded frames; each slot is referenced by a
/// 3-bit index in the uncompressed header.
pub const REF_FRAMES: usize = 8;

/// Reference-frame index width (`log2(REF_FRAMES)`).
pub const REF_FRAMES_LOG2: u32 = 3;

/// Number of references an inter frame can use simultaneously. VP9
/// has three slots: LAST, GOLDEN, ALTREF. Mirrors libvpx's
/// `MAX_REF_FRAMES`.
pub const MAX_INTER_REFS: usize = 3;

/// VP9 sync-byte sequence at the start of every uncompressed header.
/// Profile 0/1 → `0x49 0x83 0x42`, Profile 2/3 → `0x49 0x83 0x43`.
/// Encoded as a 2-bit `profile_low_bit | profile_high_bit << 1` plus
/// a 24-bit sync field; libvpx wraps this in
/// `VP9_FRAME_MARKER` / `VP9_SYNC_CODE_*`.
pub mod sync {
    /// VP9 frame marker (top 2 bits of the first byte of every
    /// uncompressed header). Per spec §6.2 `frame_marker`.
    pub const FRAME_MARKER: u8 = 0x2;

    /// VP9 sync code byte 0 (keyframes / intra-only frames only).
    pub const SYNC_CODE_0: u8 = 0x49;

    /// VP9 sync code byte 1.
    pub const SYNC_CODE_1: u8 = 0x83;

    /// VP9 sync code byte 2.
    pub const SYNC_CODE_2: u8 = 0x42;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_round_trip() {
        for (raw, expected) in [
            (0u8, Profile::Profile0),
            (1, Profile::Profile1),
            (2, Profile::Profile2),
            (3, Profile::Profile3),
        ] {
            assert_eq!(Profile::from_u8(raw), Some(expected));
            assert_eq!(expected as u8, raw);
        }
        assert_eq!(Profile::from_u8(4), None);
    }

    #[test]
    fn profile_chroma_classification() {
        assert!(Profile::Profile0.uses_420_chroma());
        assert!(!Profile::Profile1.uses_420_chroma());
        assert!(Profile::Profile2.uses_420_chroma());
        assert!(!Profile::Profile3.uses_420_chroma());
    }

    #[test]
    fn frame_type_from_bit() {
        assert_eq!(FrameType::from_bit(0), FrameType::Key);
        assert_eq!(FrameType::from_bit(1), FrameType::Inter);
        // High bits ignored — only LSB used.
        assert_eq!(FrameType::from_bit(0b1110), FrameType::Key);
        assert_eq!(FrameType::from_bit(0b1111), FrameType::Inter);
    }

    #[test]
    fn color_space_round_trip() {
        for v in 0u8..=7 {
            let cs = ColorSpace::from_u8(v).expect("0..=7 are valid");
            assert_eq!(cs as u8, v);
        }
        assert_eq!(ColorSpace::from_u8(8), None);
    }

    #[test]
    fn color_range_from_bit_and_default() {
        assert_eq!(ColorRange::from_bit(0), ColorRange::Studio);
        assert_eq!(ColorRange::from_bit(1), ColorRange::Full);
        assert_eq!(ColorRange::default(), ColorRange::Studio);
    }

    #[test]
    fn chroma_format_from_subsampling() {
        assert_eq!(
            ChromaFormat::from_subsampling(true, true),
            ChromaFormat::Yuv420
        );
        assert_eq!(
            ChromaFormat::from_subsampling(true, false),
            ChromaFormat::Yuv422
        );
        assert_eq!(
            ChromaFormat::from_subsampling(false, true),
            ChromaFormat::Yuv440
        );
        assert_eq!(
            ChromaFormat::from_subsampling(false, false),
            ChromaFormat::Yuv444
        );
    }

    #[test]
    fn chroma_format_subsampling_round_trip() {
        for cf in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv440,
            ChromaFormat::Yuv444,
        ] {
            let sx = cf.subsampling_x();
            let sy = cf.subsampling_y();
            assert_eq!(ChromaFormat::from_subsampling(sx, sy), cf);
        }
    }

    #[test]
    fn ref_frame_constants() {
        assert_eq!(REF_FRAMES, 8);
        assert_eq!(1u32 << REF_FRAMES_LOG2, REF_FRAMES as u32);
        assert_eq!(MAX_INTER_REFS, 3);
    }

    #[test]
    fn sync_constants_are_libvpx_values() {
        assert_eq!(sync::FRAME_MARKER, 0x2);
        assert_eq!(sync::SYNC_CODE_0, 0x49);
        assert_eq!(sync::SYNC_CODE_1, 0x83);
        assert_eq!(sync::SYNC_CODE_2, 0x42);
    }
}
