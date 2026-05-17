//! Minimal container readers for the conformance harness.
//!
//! Two formats are supported, both used by libvpx's VP9 conformance
//! test vectors:
//!
//! - **IVF** (`*.ivf`): a 32-byte file header followed by 1-frame
//!   chunks each prefixed with a 12-byte `(size_le_u32, pts_u64)`
//!   record. One IVF chunk = one VP9 transport packet (which may be
//!   a superframe of 1..=8 frames; `rvp9-decoder::parse_packet`
//!   handles the split).
//!
//! - **WebM** (`*.webm`): an EBML/Matroska container. The conformance
//!   vectors all carry exactly one video track (VP9) with no audio.
//!   We walk EBML/Segment/Cluster elements and extract every
//!   `SimpleBlock` payload as a packet. (No `BlockGroup`s in the
//!   conformance corpus — confirmed empirically against the 13
//!   shipped vectors.)
//!
//! Both readers expose the same shape:
//!
//! ```ignore
//! let packets: Vec<&[u8]> = read_packets(&data)?;
//! ```
//!
//! Each `&[u8]` is one transport packet — i.e. exactly what gets fed
//! into [`rvp9_decoder::Decoder::parse_packet`].

use std::path::Path;

/// Errors from the container readers.
#[derive(Debug)]
pub enum ContainerError {
    /// File read failed at the filesystem layer.
    Io(std::io::Error),
    /// File extension is not `.ivf` or `.webm`.
    UnknownExtension(String),
    /// File too short to contain the expected fixed-size header.
    Truncated {
        /// What we were trying to read.
        what: &'static str,
        /// Actual file length.
        len: usize,
    },
    /// IVF magic bytes (`DKIF`) missing.
    BadIvfMagic,
    /// EBML / Matroska parse error.
    BadEbml(&'static str),
    /// A frame / block claimed more bytes than the file actually contains.
    BadFrameSize {
        /// 1-based packet index where the overflow happened.
        packet_index: u32,
        /// Declared size in the container.
        declared: usize,
        /// Available bytes remaining in the file.
        available: usize,
    },
}

impl std::fmt::Display for ContainerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::UnknownExtension(s) => write!(f, "unknown container extension: {s}"),
            Self::Truncated { what, len } => write!(f, "truncated {what} (len={len})"),
            Self::BadIvfMagic => f.write_str("not an IVF file (missing DKIF signature)"),
            Self::BadEbml(s) => write!(f, "ebml/matroska parse error: {s}"),
            Self::BadFrameSize {
                packet_index,
                declared,
                available,
            } => write!(
                f,
                "packet {packet_index}: declared {declared} bytes but only {available} available"
            ),
        }
    }
}

impl std::error::Error for ContainerError {}

impl From<std::io::Error> for ContainerError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// One transport packet's byte range within the parent buffer.
#[derive(Clone, Copy, Debug)]
pub struct PacketRange {
    /// Offset of the first byte (inclusive).
    pub start: usize,
    /// Offset just past the last byte (exclusive).
    pub end: usize,
}

impl PacketRange {
    /// Borrow the packet bytes from the parent buffer.
    #[must_use]
    pub fn slice<'a>(&self, data: &'a [u8]) -> &'a [u8] {
        &data[self.start..self.end]
    }

    /// Length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// `true` if the packet is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Read every transport packet from a container file. Dispatches on
/// the path extension (`.ivf` → [`read_ivf_packets`], `.webm` →
/// [`read_webm_packets`]).
///
/// Returns the raw file bytes and a list of `PacketRange`s into them.
///
/// # Errors
///
/// - [`ContainerError::Io`] on filesystem failures.
/// - [`ContainerError::UnknownExtension`] if the extension isn't
///   `.ivf` or `.webm`.
/// - Any error from the underlying reader.
pub fn read_packets(path: &Path) -> Result<(Vec<u8>, Vec<PacketRange>), ContainerError> {
    let data = std::fs::read(path)?;
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let packets = match ext.as_str() {
        "ivf" => read_ivf_packets(&data)?,
        "webm" => read_webm_packets(&data)?,
        other => return Err(ContainerError::UnknownExtension(other.to_string())),
    };
    Ok((data, packets))
}

/// Walk an IVF file in memory, returning the byte ranges of every
/// frame payload.
///
/// File layout (libvpx convention):
/// ```text
///   off 0:  "DKIF"               (4 bytes)
///   off 4:  version (le u16)
///   off 6:  header_size (le u16, usually 32)
///   ...     codec FourCC, width, height, framerate, etc.
///   off 32: per-frame records:
///             u32 size_le
///             u64 pts_le
///             <size bytes payload>
/// ```
///
/// # Errors
///
/// - [`ContainerError::Truncated`] if `data.len() < 32`.
/// - [`ContainerError::BadIvfMagic`] if the first 4 bytes aren't `DKIF`.
/// - [`ContainerError::BadFrameSize`] if a frame's declared size
///   extends past the end of the file.
pub fn read_ivf_packets(data: &[u8]) -> Result<Vec<PacketRange>, ContainerError> {
    if data.len() < 32 {
        return Err(ContainerError::Truncated {
            what: "IVF file header",
            len: data.len(),
        });
    }
    if &data[0..4] != b"DKIF" {
        return Err(ContainerError::BadIvfMagic);
    }
    let mut packets = Vec::new();
    let mut pos: usize = 32;
    let mut idx: u32 = 0;
    while pos + 12 <= data.len() {
        idx += 1;
        let size = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
            as usize;
        let payload_start = pos + 12;
        let payload_end = payload_start
            .checked_add(size)
            .ok_or(ContainerError::BadFrameSize {
                packet_index: idx,
                declared: usize::MAX,
                available: data.len(),
            })?;
        if payload_end > data.len() {
            return Err(ContainerError::BadFrameSize {
                packet_index: idx,
                declared: payload_end,
                available: data.len(),
            });
        }
        packets.push(PacketRange {
            start: payload_start,
            end: payload_end,
        });
        pos = payload_end;
    }
    Ok(packets)
}

// ============================================================================
// WebM (EBML / Matroska) reader
// ============================================================================
//
// Just enough EBML to walk the SimpleBlocks of the (single) video track in
// the conformance vectors. We do NOT validate codec ID, track number, or
// any of the metadata — the conformance corpus is curated and uniform.

/// Element IDs we recognize (full 4-byte IDs including the length-prefix bits).
mod ebml_id {
    pub const EBML: u32 = 0x1A45_DFA3;
    pub const SEGMENT: u32 = 0x1853_8067;
    pub const CLUSTER: u32 = 0x1F43_B675;
    pub const SIMPLE_BLOCK: u32 = 0xA3;
    pub const BLOCK_GROUP: u32 = 0xA0;
    pub const BLOCK: u32 = 0xA1;
}

/// Read a single variable-length EBML id at `data[pos..]`. Returns
/// `(id_with_marker_bit, length_in_bytes)`.
fn read_ebml_id(data: &[u8], pos: usize) -> Result<(u32, usize), ContainerError> {
    if pos >= data.len() {
        return Err(ContainerError::BadEbml("EOF while reading element id"));
    }
    let first = data[pos];
    if first == 0 {
        return Err(ContainerError::BadEbml("element id with no length marker"));
    }
    let len = (first.leading_zeros() as usize) + 1; // 1..=4
    if len > 4 {
        return Err(ContainerError::BadEbml("element id wider than 4 bytes"));
    }
    if pos + len > data.len() {
        return Err(ContainerError::BadEbml("EOF inside element id"));
    }
    let mut id: u32 = 0;
    for &b in &data[pos..pos + len] {
        id = (id << 8) | u32::from(b);
    }
    Ok((id, len))
}

/// Read a variable-length EBML size at `data[pos..]`. Returns
/// `(size_value, length_in_bytes)`. The leading marker bit is
/// stripped from the value. An all-ones value means "unknown size"
/// (encoded as `Some(u64::MAX)`).
fn read_ebml_vint(data: &[u8], pos: usize) -> Result<(u64, usize), ContainerError> {
    if pos >= data.len() {
        return Err(ContainerError::BadEbml("EOF while reading vint"));
    }
    let first = data[pos];
    if first == 0 {
        return Err(ContainerError::BadEbml("vint with no length marker"));
    }
    let len = (first.leading_zeros() as usize) + 1; // 1..=8
    if len > 8 {
        return Err(ContainerError::BadEbml("vint wider than 8 bytes"));
    }
    if pos + len > data.len() {
        return Err(ContainerError::BadEbml("EOF inside vint"));
    }
    // Clear the marker bit on the first byte, then accumulate.
    let mut value: u64 = u64::from(first & ((1u8 << (8 - len)) - 1));
    for &b in &data[pos + 1..pos + len] {
        value = (value << 8) | u64::from(b);
    }
    Ok((value, len))
}

/// Walk a WebM (Matroska/EBML) file, returning the payload byte
/// ranges of every video-track block (one per VP9 transport packet).
///
/// Implementation note: we recurse into top-level Segment and Cluster
/// containers, but treat any unrecognised element as "skip <size>
/// bytes". This is robust against tracks/seek-heads/cues/etc.
///
/// # Errors
///
/// - [`ContainerError::BadEbml`] on malformed EBML structure.
/// - [`ContainerError::BadFrameSize`] if a block claims more bytes
///   than remain in its parent.
pub fn read_webm_packets(data: &[u8]) -> Result<Vec<PacketRange>, ContainerError> {
    let mut packets: Vec<PacketRange> = Vec::new();
    walk_ebml(data, 0, data.len(), &mut packets, /* inside_cluster */ false)?;
    Ok(packets)
}

fn walk_ebml(
    data: &[u8],
    mut pos: usize,
    end: usize,
    out: &mut Vec<PacketRange>,
    inside_cluster: bool,
) -> Result<(), ContainerError> {
    while pos < end {
        let (id, id_len) = read_ebml_id(data, pos)?;
        pos += id_len;
        let (size, size_len) = read_ebml_vint(data, pos)?;
        pos += size_len;

        // EBML / Segment with "unknown size": the All-ones value means
        // "container runs to EOF" (Matroska Segment uses this). Treat
        // that as "remaining length of file".
        let unknown = is_unknown_vint(size, size_len);
        let payload_size = if unknown {
            end.saturating_sub(pos)
        } else {
            usize::try_from(size).map_err(|_| ContainerError::BadEbml("vint too large"))?
        };
        let payload_end = pos
            .checked_add(payload_size)
            .ok_or(ContainerError::BadEbml("payload size overflow"))?;
        if payload_end > end {
            return Err(ContainerError::BadEbml(
                "element payload runs past parent end",
            ));
        }

        match id {
            ebml_id::EBML | ebml_id::SEGMENT => {
                walk_ebml(data, pos, payload_end, out, false)?;
            }
            ebml_id::CLUSTER => {
                walk_ebml(data, pos, payload_end, out, true)?;
            }
            ebml_id::SIMPLE_BLOCK if inside_cluster => {
                if let Some(range) = parse_block_frame(data, pos, payload_size, out.len() + 1)? {
                    out.push(range);
                }
            }
            ebml_id::BLOCK_GROUP if inside_cluster => {
                // BlockGroup contains a Block + extras; recurse to find the Block.
                walk_ebml(data, pos, payload_end, out, true)?;
            }
            ebml_id::BLOCK if inside_cluster => {
                if let Some(range) = parse_block_frame(data, pos, payload_size, out.len() + 1)? {
                    out.push(range);
                }
            }
            _ => {
                // Unknown / unhandled element — skip its payload.
            }
        }
        pos = payload_end;
    }
    Ok(())
}

#[inline]
fn is_unknown_vint(value: u64, size_len: usize) -> bool {
    // All-ones vint of width `size_len` means "unknown size".
    let bits = (8 * size_len as u32) - size_len as u32; // (8 - 1) per first byte + 8 per subsequent
    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    value == mask
}

/// Parse one Block / SimpleBlock payload. The first 3+ bytes are
/// `track_number_vint, timecode_le_i16, flags_u8`. Following bytes
/// are the frame data (assuming `Lacing = 0`, which is the case for
/// every libvpx-emitted conformance vector — single frame per block,
/// no lacing).
///
/// Returns `Ok(Some(range))` for a valid single-frame block. Returns
/// `Ok(None)` if lacing is non-zero (we silently skip those; none of
/// the 13 webm conformance vectors use lacing). Returns an `Err` only
/// for malformed block headers.
fn parse_block_frame(
    data: &[u8],
    pos: usize,
    payload_size: usize,
    packet_idx_hint: usize,
) -> Result<Option<PacketRange>, ContainerError> {
    if payload_size < 4 {
        return Err(ContainerError::BadEbml("block payload too small"));
    }
    let block_end = pos + payload_size;
    // Track number is a vint.
    let (_track, tn_len) = read_ebml_vint(data, pos)?;
    let after_tn = pos + tn_len;
    // 2-byte timecode + 1 byte flags follow.
    if after_tn + 3 > block_end {
        return Err(ContainerError::BadEbml("block missing timecode/flags"));
    }
    let flags = data[after_tn + 2];
    let lacing = (flags >> 1) & 0x03;
    let frame_start = after_tn + 3;
    if frame_start > block_end {
        return Err(ContainerError::BadEbml("block frame_start past end"));
    }
    if lacing != 0 {
        // Lacing path: not used by the conformance corpus. Skip
        // gracefully so adding a laced vector someday doesn't crash —
        // the harness will simply see fewer packets.
        return Ok(None);
    }
    // Sanity: there must be at least one byte of frame data.
    if frame_start == block_end {
        return Err(ContainerError::BadFrameSize {
            packet_index: packet_idx_hint as u32,
            declared: 0,
            available: 0,
        });
    }
    Ok(Some(PacketRange {
        start: frame_start,
        end: block_end,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ivf_minimal_two_frames() {
        // 32-byte header (mostly zeros — content irrelevant for the
        // walker) + two 1-byte frames.
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"DKIF");
        // Frame 1: size=1, pts=0, payload=0xAA.
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.push(0xAA);
        // Frame 2: size=1, pts=1, payload=0xBB.
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.push(0xBB);

        let packets = read_ivf_packets(&data).unwrap();
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].slice(&data), &[0xAA]);
        assert_eq!(packets[1].slice(&data), &[0xBB]);
    }

    #[test]
    fn ivf_rejects_bad_magic() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"NOPE");
        assert!(matches!(
            read_ivf_packets(&data),
            Err(ContainerError::BadIvfMagic)
        ));
    }

    #[test]
    fn ivf_rejects_short_header() {
        let data = vec![0u8; 8];
        assert!(matches!(
            read_ivf_packets(&data),
            Err(ContainerError::Truncated { .. })
        ));
    }

    #[test]
    fn ivf_rejects_truncated_frame() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"DKIF");
        // Declare size=100 but only 5 bytes follow.
        data.extend_from_slice(&100u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&[0u8; 5]);
        assert!(matches!(
            read_ivf_packets(&data),
            Err(ContainerError::BadFrameSize { .. })
        ));
    }

    #[test]
    fn ebml_vint_examples() {
        // Spec examples (Matroska EBML):
        //   0x82 → length 1, value 2
        //   0x40 0x02 → length 2, value 2
        //   0x20 0x00 0x02 → length 3, value 2
        let (v, n) = read_ebml_vint(&[0x82], 0).unwrap();
        assert_eq!((v, n), (2, 1));
        let (v, n) = read_ebml_vint(&[0x40, 0x02], 0).unwrap();
        assert_eq!((v, n), (2, 2));
        let (v, n) = read_ebml_vint(&[0x20, 0x00, 0x02], 0).unwrap();
        assert_eq!((v, n), (2, 3));
        // Unknown-size single byte (all-ones).
        let (v, n) = read_ebml_vint(&[0xFF], 0).unwrap();
        assert_eq!(n, 1);
        assert!(is_unknown_vint(v, 1));
    }

    #[test]
    fn ebml_id_examples() {
        // EBML header element id 0x1A45_DFA3 is 4 bytes long.
        let id_bytes = [0x1A, 0x45, 0xDF, 0xA3];
        let (id, n) = read_ebml_id(&id_bytes, 0).unwrap();
        assert_eq!((id, n), (0x1A45_DFA3, 4));
        // Single-byte 0xA3 = SimpleBlock.
        let (id, n) = read_ebml_id(&[0xA3], 0).unwrap();
        assert_eq!((id, n), (0xA3, 1));
    }
}
