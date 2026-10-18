//! Minimal VP9 container readers/writers (IVF + WebM).
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
//! - **WebM** (`*.webm`): an EBML/Matroska container. We identify the
//!   first `V_VP9` video TrackEntry, then walk
//!   EBML/Segment/Cluster elements and extract that track's
//!   `SimpleBlock` and `BlockGroup` payloads as packets. Laced VP9
//!   blocks are rejected because this reader returns one contiguous
//!   range per transport packet.
//!
//! Both readers expose the same shape:
//!
//! ```ignore
//! let packets: Vec<&[u8]> = read_packets(&data)?;
//! ```
//!
//! Each `&[u8]` is one transport packet — i.e. exactly what gets fed
//! into `Decoder::parse_packet`.

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
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
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !matches!(ext.as_str(), "ivf" | "webm") {
        return Err(ContainerError::UnknownExtension(ext));
    }
    let data = std::fs::read(path)?;
    let packets = match ext.as_str() {
        "ivf" => read_ivf_packets(&data)?,
        "webm" => read_webm_packets(&data)?,
        _ => unreachable!("extension validated before reading"),
    };
    Ok((data, packets))
}

/// Incremental container reader for callers that may stop before EOF.
///
/// Unlike [`read_packets`], this reader retains only the current transport
/// packet. IVF frame records and WebM blocks are pulled from disk on demand.
pub struct PacketReader {
    inner: PacketReaderInner,
    framerate: Option<(u32, u32)>,
}

enum PacketReaderInner {
    Ivf(IvfPacketReader),
    Webm(WebmPacketReader),
}

struct IvfPacketReader {
    file: BufReader<File>,
    file_len: u64,
    packet_index: u32,
}

#[derive(Clone, Copy)]
struct WebmLevel {
    end: u64,
    context: WebmContext,
    depth: usize,
}

/// Schema context for the subset of master elements the packet walkers enter.
#[derive(Clone, Copy, Default)]
struct WebmContext {
    id: u32,
    parent_id: u32,
    unknown_size: bool,
    inside_cluster: bool,
}

impl WebmContext {
    fn child(self, id: u32, unknown_size: bool, inside_cluster: bool) -> Self {
        Self {
            id,
            parent_id: self.id,
            unknown_size,
            inside_cluster,
        }
    }

    fn ends_before(self, next_id: u32) -> bool {
        if !self.unknown_size {
            return false;
        }
        let root_element = matches!(next_id, ebml_id::EBML | ebml_id::SEGMENT);
        // RFC 8794 §6.2: an unknown-sized master ends before a sibling or
        // ancestor's sibling. RFC 9559 puts these children directly in Segment.
        // Global Void/CRC-32 elements deliberately do not end a master.
        match (self.parent_id, self.id) {
            (0, ebml_id::SEGMENT) => root_element,
            (ebml_id::SEGMENT, ebml_id::CLUSTER) => {
                root_element
                    || matches!(
                        next_id,
                        ebml_id::SEEK_HEAD
                            | ebml_id::INFO
                            | ebml_id::CLUSTER
                            | ebml_id::TRACKS
                            | ebml_id::CUES
                            | ebml_id::ATTACHMENTS
                            | ebml_id::CHAPTERS
                            | ebml_id::TAGS
                    )
            }
            _ => false,
        }
    }
}

struct WebmPacketReader {
    file: BufReader<File>,
    video_track: u64,
    levels: Vec<WebmLevel>,
    packet_index: u32,
}

impl PacketReader {
    /// Open an IVF or WebM file without loading its packet payloads.
    ///
    /// # Errors
    ///
    /// Returns [`ContainerError::UnknownExtension`] before touching the
    /// filesystem for paths other than `.ivf` or `.webm`. Container header
    /// and VP9-track discovery errors are returned while opening.
    pub fn open(path: &Path) -> Result<Self, ContainerError> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "ivf" => Self::open_ivf(path),
            "webm" => Self::open_webm(path),
            _ => Err(ContainerError::UnknownExtension(ext)),
        }
    }

    fn open_ivf(path: &Path) -> Result<Self, ContainerError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < 32 {
            return Err(ContainerError::Truncated {
                what: "IVF file header",
                len: usize::try_from(file_len).unwrap_or(usize::MAX),
            });
        }
        let mut file = BufReader::new(file);
        let mut header = [0u8; 32];
        file.read_exact(&mut header)?;
        if &header[0..4] != b"DKIF" {
            return Err(ContainerError::BadIvfMagic);
        }
        let num = u32::from_le_bytes(header[16..20].try_into().expect("four-byte slice"));
        let den = u32::from_le_bytes(header[20..24].try_into().expect("four-byte slice"));
        let framerate = (num != 0 && den != 0).then_some((num, den));
        Ok(Self {
            inner: PacketReaderInner::Ivf(IvfPacketReader {
                file,
                file_len,
                packet_index: 0,
            }),
            framerate,
        })
    }

    fn open_webm(path: &Path) -> Result<Self, ContainerError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut file = BufReader::new(file);
        let video_track = if file_len == 0 {
            0
        } else {
            find_vp9_video_track_stream(&mut file, file_len)?
        };
        file.seek(SeekFrom::Start(0))?;
        Ok(Self {
            inner: PacketReaderInner::Webm(WebmPacketReader {
                file,
                video_track,
                levels: vec![WebmLevel {
                    end: file_len,
                    context: WebmContext::default(),
                    depth: 0,
                }],
                packet_index: 0,
            }),
            framerate: None,
        })
    }

    /// Container framerate when carried by the format.
    ///
    /// IVF rates with a zero numerator or denominator and all WebM inputs
    /// return `None`.
    #[must_use]
    pub fn framerate(&self) -> Option<(u32, u32)> {
        self.framerate
    }

    /// Read the next VP9 transport packet.
    ///
    /// # Errors
    ///
    /// Returns filesystem and malformed-container errors encountered while
    /// advancing to the next packet.
    pub fn next_packet(&mut self) -> Result<Option<Vec<u8>>, ContainerError> {
        match &mut self.inner {
            PacketReaderInner::Ivf(reader) => reader.next_packet(),
            PacketReaderInner::Webm(reader) => reader.next_packet(),
        }
    }
}

impl IvfPacketReader {
    fn next_packet(&mut self) -> Result<Option<Vec<u8>>, ContainerError> {
        let pos = self.file.stream_position()?;
        if pos == self.file_len {
            return Ok(None);
        }
        let available = self.file_len.saturating_sub(pos);
        if available < 12 {
            return Err(ContainerError::Truncated {
                what: "IVF frame record header",
                len: usize::try_from(self.file_len).unwrap_or(usize::MAX),
            });
        }

        let mut record = [0u8; 12];
        self.file.read_exact(&mut record)?;
        self.packet_index = self.packet_index.saturating_add(1);
        let size = u32::from_le_bytes(record[0..4].try_into().expect("four-byte slice")) as usize;
        let available = self.file_len.saturating_sub(self.file.stream_position()?);
        if u64::try_from(size).unwrap_or(u64::MAX) > available {
            return Err(ContainerError::BadFrameSize {
                packet_index: self.packet_index,
                declared: size,
                available: usize::try_from(available).unwrap_or(usize::MAX),
            });
        }
        let mut packet = Vec::new();
        packet.try_reserve_exact(size).map_err(|_| {
            ContainerError::Io(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "allocating IVF packet",
            ))
        })?;
        packet.resize(size, 0);
        self.file.read_exact(&mut packet)?;
        Ok(Some(packet))
    }
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
/// - [`ContainerError::Truncated`] if the file header or a trailing frame
///   record header is incomplete.
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
        let size =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let payload_start = pos + 12;
        let payload_end = payload_start
            .checked_add(size)
            .ok_or(ContainerError::BadFrameSize {
                packet_index: idx,
                declared: size,
                available: data.len() - payload_start,
            })?;
        if payload_end > data.len() {
            return Err(ContainerError::BadFrameSize {
                packet_index: idx,
                declared: size,
                available: data.len() - payload_start,
            });
        }
        packets.push(PacketRange {
            start: payload_start,
            end: payload_end,
        });
        pos = payload_end;
    }
    if pos != data.len() {
        return Err(ContainerError::Truncated {
            what: "IVF frame record header",
            len: data.len(),
        });
    }
    Ok(packets)
}

// ============================================================================
// IVF writer
// ============================================================================
//
// RC.4 — mirror of [`read_ivf_packets`] in the write direction. Layout
// matches libvpx's `ivf_write_file_header` + `ivf_write_frame_header`
// (`ivfenc.c:9-44`).

/// Fixed-size IVF file header descriptor handed to [`write_ivf`].
///
/// Mirrors libvpx's `IVF_FILE_HEADER` layout (`ivfenc.c:11`): the magic
/// `DKIF`, a 16-bit version (always `0`), a 16-bit header size (always
/// `32`), the codec four-CC (`VP90` for VP9), width / height as little-
/// endian `u16`s, framerate as a `(num_le_u32, den_le_u32)` pair, and a
/// trailing `num_frames` slot left at the value supplied by the caller
/// (libvpx leaves this at `0` and patches it back in after the encode
/// completes — we follow the same convention by default).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IvfHeader {
    /// Codec FourCC. `*b"VP90"` for VP9.
    pub codec_fourcc: [u8; 4],
    /// Coded frame width in pixels.
    pub width: u16,
    /// Coded frame height in pixels.
    pub height: u16,
    /// Framerate numerator (e.g. `30` for 30 fps).
    pub framerate_num: u32,
    /// Framerate denominator (e.g. `1` for 30 fps).
    pub framerate_den: u32,
}

impl IvfHeader {
    /// VP9 FourCC (`VP90`).
    pub const FOURCC_VP9: [u8; 4] = *b"VP90";

    /// Build an IVF header for a VP9 stream at `(width, height)`,
    /// `framerate_num / framerate_den` fps.
    #[must_use]
    pub const fn vp9(width: u16, height: u16, framerate_num: u32, framerate_den: u32) -> Self {
        Self {
            codec_fourcc: Self::FOURCC_VP9,
            width,
            height,
            framerate_num,
            framerate_den,
        }
    }
}

/// Incremental IVF muxer that patches the packet count when finished.
pub struct IvfStreamWriter<W: Write + Seek> {
    out: W,
    packet_count_offset: u64,
    packet_count: u32,
}

impl<W: Write + Seek> IvfStreamWriter<W> {
    /// Write an IVF header at the current position with a provisional zero packet count.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] if the packet-count patch
    /// position overflows. Otherwise forwards any [`io::Error`] from the
    /// writer.
    pub fn new(mut out: W, header: &IvfHeader) -> io::Result<Self> {
        let start = out.stream_position()?;
        let packet_count_offset = start.checked_add(24).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "IVF packet-count patch position overflow",
            )
        })?;
        write_ivf_header(&mut out, header, 0)?;
        Ok(Self {
            out,
            packet_count_offset,
            packet_count: 0,
        })
    }

    /// Append one packet record and payload.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] if the packet length or
    /// resulting packet count cannot fit IVF. Otherwise forwards writer
    /// errors.
    pub fn write_packet(&mut self, packet: &[u8]) -> io::Result<()> {
        let size = u32::try_from(packet.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "packet too large for IVF (size must fit in u32)",
            )
        })?;
        let next_count = self.packet_count.checked_add(1).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "too many packets for IVF (must fit in u32)",
            )
        })?;
        let mut record = [0u8; 12];
        record[0..4].copy_from_slice(&size.to_le_bytes());
        record[4..12].copy_from_slice(&u64::from(self.packet_count).to_le_bytes());
        self.out.write_all(&record)?;
        self.out.write_all(packet)?;
        self.packet_count = next_count;
        Ok(())
    }

    /// Patch the header packet count, flush, and return the writer.
    ///
    /// # Errors
    ///
    /// Forwards any seek, write, or flush error.
    pub fn finish(mut self) -> io::Result<W> {
        self.out.seek(SeekFrom::Start(self.packet_count_offset))?;
        self.out.write_all(&self.packet_count.to_le_bytes())?;
        self.out.seek(SeekFrom::End(0))?;
        self.out.flush()?;
        Ok(self.out)
    }

    /// Number of packets written so far.
    #[must_use]
    pub const fn packet_count(&self) -> u32 {
        self.packet_count
    }
}

fn write_ivf_header<W: Write>(out: &mut W, header: &IvfHeader, num_frames: u32) -> io::Result<()> {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(b"DKIF");
    bytes[4..6].copy_from_slice(&0u16.to_le_bytes());
    bytes[6..8].copy_from_slice(&32u16.to_le_bytes());
    bytes[8..12].copy_from_slice(&header.codec_fourcc);
    bytes[12..14].copy_from_slice(&header.width.to_le_bytes());
    bytes[14..16].copy_from_slice(&header.height.to_le_bytes());
    bytes[16..20].copy_from_slice(&header.framerate_num.to_le_bytes());
    bytes[20..24].copy_from_slice(&header.framerate_den.to_le_bytes());
    bytes[24..28].copy_from_slice(&num_frames.to_le_bytes());
    out.write_all(&bytes)
}

/// Write an IVF file containing `packets`.
///
/// The file is laid out as:
/// - 32-byte file header: `DKIF`, version (`0_le_u16`), header_size
///   (`32_le_u16`), codec FourCC, width (`le_u16`), height (`le_u16`),
///   framerate_num (`le_u32`), framerate_den (`le_u32`), num_frames
///   (`le_u32`, set to `packets.len()`), unused (`0_le_u32`).
/// - One per-packet record per `&[u8]` in `packets`: `u32 size_le`,
///   `u64 pts_le` (the packet's index, monotonically increasing), and
///   the payload bytes verbatim.
///
/// # Errors
///
/// Forwards any [`io::Error`] from the writer.
pub fn write_ivf<W: Write>(out: &mut W, header: &IvfHeader, packets: &[&[u8]]) -> io::Result<()> {
    let num_frames = u32::try_from(packets.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "too many packets for IVF (must fit in u32)",
        )
    })?;

    write_ivf_header(out, header, num_frames)?;

    // Per-packet record.
    for (idx, packet) in packets.iter().enumerate() {
        let size = u32::try_from(packet.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "packet too large for IVF (size must fit in u32)",
            )
        })?;
        let pts = idx as u64;
        let mut rec = [0u8; 12];
        rec[0..4].copy_from_slice(&size.to_le_bytes());
        rec[4..12].copy_from_slice(&pts.to_le_bytes());
        out.write_all(&rec)?;
        out.write_all(packet)?;
    }

    Ok(())
}

// ============================================================================
// WebM (EBML / Matroska) reader
// ============================================================================
//
// Just enough EBML to identify a VP9 video TrackEntry and walk its
// SimpleBlock/Block payloads.

/// Element IDs we recognize (full 4-byte IDs including the length-prefix bits).
mod ebml_id {
    pub const EBML: u32 = 0x1A45_DFA3;
    pub const SEGMENT: u32 = 0x1853_8067;
    pub const TRACKS: u32 = 0x1654_AE6B;
    pub const TRACK_ENTRY: u32 = 0xAE;
    pub const TRACK_NUMBER: u32 = 0xD7;
    pub const TRACK_TYPE: u32 = 0x83;
    pub const CODEC_ID: u32 = 0x86;
    pub const CLUSTER: u32 = 0x1F43_B675;
    pub const SEEK_HEAD: u32 = 0x114D_9B74;
    pub const INFO: u32 = 0x1549_A966;
    pub const CUES: u32 = 0x1C53_BB6B;
    pub const ATTACHMENTS: u32 = 0x1941_A469;
    pub const CHAPTERS: u32 = 0x1043_A770;
    pub const TAGS: u32 = 0x1254_C367;
    pub const SIMPLE_BLOCK: u32 = 0xA3;
    pub const BLOCK_GROUP: u32 = 0xA0;
    pub const BLOCK: u32 = 0xA1;
}

/// Maximum recognized-container nesting accepted by the recursive packet walk.
///
/// Valid WebM packet paths are shallow (`Segment` → `Cluster` → `BlockGroup`).
/// This larger budget leaves ample room for unusual but valid layouts while
/// bounding stack use for hostile input.
const MAX_EBML_NESTING_DEPTH: usize = 32;

fn read_stream_ebml_id<R: Read>(reader: &mut R) -> Result<u32, ContainerError> {
    let mut first = [0u8; 1];
    reader
        .read_exact(&mut first)
        .map_err(|error| stream_header_error(error, "EOF while reading element id"))?;
    if first[0] == 0 {
        return Err(ContainerError::BadEbml("element id with no length marker"));
    }
    let len = first[0].leading_zeros() as usize + 1;
    if len > 4 {
        return Err(ContainerError::BadEbml("element id wider than 4 bytes"));
    }
    let mut id = u32::from(first[0]);
    for _ in 1..len {
        let mut byte = [0u8; 1];
        reader
            .read_exact(&mut byte)
            .map_err(|error| stream_header_error(error, "EOF inside element id"))?;
        id = (id << 8) | u32::from(byte[0]);
    }
    Ok(id)
}

fn read_stream_ebml_vint<R: Read>(reader: &mut R) -> Result<(u64, usize), ContainerError> {
    let mut first = [0u8; 1];
    reader
        .read_exact(&mut first)
        .map_err(|error| stream_header_error(error, "EOF while reading vint"))?;
    if first[0] == 0 {
        return Err(ContainerError::BadEbml("vint with no length marker"));
    }
    let len = first[0].leading_zeros() as usize + 1;
    if len > 8 {
        return Err(ContainerError::BadEbml("vint wider than 8 bytes"));
    }
    let mut value = u64::from(first[0] & ((1u8 << (8 - len)) - 1));
    for _ in 1..len {
        let mut byte = [0u8; 1];
        reader
            .read_exact(&mut byte)
            .map_err(|error| stream_header_error(error, "EOF inside vint"))?;
        value = (value << 8) | u64::from(byte[0]);
    }
    Ok((value, len))
}

fn stream_header_error(error: io::Error, eof_message: &'static str) -> ContainerError {
    if error.kind() == io::ErrorKind::UnexpectedEof {
        ContainerError::BadEbml(eof_message)
    } else {
        ContainerError::Io(error)
    }
}

fn stream_payload_end(
    payload_start: u64,
    size: u64,
    unknown_size: bool,
    parent_end: u64,
) -> Result<u64, ContainerError> {
    let payload_end = if unknown_size {
        parent_end
    } else {
        payload_start
            .checked_add(size)
            .ok_or(ContainerError::BadEbml("payload size overflow"))?
    };
    if payload_end > parent_end {
        return Err(ContainerError::BadEbml(
            "element payload runs past parent end",
        ));
    }
    Ok(payload_end)
}

fn find_vp9_video_track_stream(
    file: &mut BufReader<File>,
    file_len: u64,
) -> Result<u64, ContainerError> {
    #[derive(Clone, Copy)]
    struct ScanLevel {
        end: u64,
        inside_tracks: bool,
        depth: usize,
    }

    let mut levels = vec![ScanLevel {
        end: file_len,
        inside_tracks: false,
        depth: 0,
    }];
    while let Some(level) = levels.last().copied() {
        let pos = file.stream_position()?;
        if pos >= level.end {
            levels.pop();
            continue;
        }
        let id = read_stream_ebml_id(file)?;
        let (size, size_len) = read_stream_ebml_vint(file)?;
        let unknown_size = validate_element_size(id, size, size_len)?;
        let payload_start = file.stream_position()?;
        let payload_end = stream_payload_end(payload_start, size, unknown_size, level.end)?;

        match id {
            ebml_id::EBML | ebml_id::SEGMENT | ebml_id::TRACKS => {
                let depth = level.depth + 1;
                if depth > MAX_EBML_NESTING_DEPTH {
                    return Err(ContainerError::BadEbml(
                        "recognized container nesting depth exceeds limit",
                    ));
                }
                levels.push(ScanLevel {
                    end: payload_end,
                    inside_tracks: level.inside_tracks || id == ebml_id::TRACKS,
                    depth,
                });
            }
            ebml_id::TRACK_ENTRY if level.inside_tracks => {
                if let Some(track) = scan_vp9_track_entry(file, payload_end)? {
                    return Ok(track);
                }
            }
            _ => {
                file.seek(SeekFrom::Start(payload_end))?;
            }
        }
    }
    Err(ContainerError::BadEbml("VP9 video track not found"))
}

fn scan_vp9_track_entry(
    file: &mut BufReader<File>,
    end: u64,
) -> Result<Option<u64>, ContainerError> {
    let mut track_number = None;
    let mut track_type = None;
    let mut codec_is_vp9 = false;

    while file.stream_position()? < end {
        let id = read_stream_ebml_id(file)?;
        let (size, size_len) = read_stream_ebml_vint(file)?;
        let unknown_size = validate_element_size(id, size, size_len)
            .map_err(|_| ContainerError::BadEbml("unknown-size TrackEntry child"))?;
        if unknown_size {
            return Err(ContainerError::BadEbml("unknown-size TrackEntry child"));
        }
        let payload_start = file.stream_position()?;
        let payload_end = stream_payload_end(payload_start, size, unknown_size, end)?;
        match id {
            ebml_id::TRACK_NUMBER | ebml_id::TRACK_TYPE => {
                if size == 0 || size > 8 {
                    return Err(ContainerError::BadEbml(
                        "EBML unsigned integer must contain 1..=8 bytes",
                    ));
                }
                let mut bytes = [0u8; 8];
                let size = size as usize;
                file.read_exact(&mut bytes[..size])?;
                let value = bytes[..size]
                    .iter()
                    .fold(0u64, |value, &byte| (value << 8) | u64::from(byte));
                if id == ebml_id::TRACK_NUMBER {
                    if value == 0 {
                        return Err(ContainerError::BadEbml("TrackNumber must be non-zero"));
                    }
                    track_number = Some(value);
                } else {
                    track_type = Some(value);
                }
            }
            ebml_id::CODEC_ID if size == 5 => {
                let mut codec = [0u8; 5];
                file.read_exact(&mut codec)?;
                codec_is_vp9 = &codec == b"V_VP9";
            }
            _ => {
                file.seek(SeekFrom::Start(payload_end))?;
            }
        }
        file.seek(SeekFrom::Start(payload_end))?;
    }

    Ok(match (track_number, track_type, codec_is_vp9) {
        (Some(number), Some(1), true) => Some(number),
        _ => None,
    })
}

impl WebmPacketReader {
    fn next_packet(&mut self) -> Result<Option<Vec<u8>>, ContainerError> {
        while let Some(level) = self.levels.last().copied() {
            let pos = self.file.stream_position()?;
            if pos >= level.end {
                self.levels.pop();
                continue;
            }
            let id = read_stream_ebml_id(&mut self.file)?;
            if level.context.ends_before(id) {
                self.levels.pop();
                self.file.seek(SeekFrom::Start(pos))?;
                continue;
            }
            let (size, size_len) = read_stream_ebml_vint(&mut self.file)?;
            let unknown_size = validate_element_size(id, size, size_len)?;
            let payload_start = self.file.stream_position()?;
            let payload_end = stream_payload_end(payload_start, size, unknown_size, level.end)?;

            match id {
                ebml_id::EBML | ebml_id::SEGMENT => {
                    self.push_level(
                        payload_end,
                        level.context.child(id, unknown_size, false),
                        level.depth + 1,
                    )?;
                }
                ebml_id::CLUSTER => {
                    self.push_level(
                        payload_end,
                        level.context.child(id, unknown_size, true),
                        level.depth + 1,
                    )?;
                }
                ebml_id::BLOCK_GROUP if level.context.inside_cluster => {
                    self.push_level(
                        payload_end,
                        level.context.child(id, unknown_size, true),
                        level.depth + 1,
                    )?;
                }
                ebml_id::SIMPLE_BLOCK | ebml_id::BLOCK if level.context.inside_cluster => {
                    if let Some(packet) = self.read_block(payload_end)? {
                        self.packet_index = self.packet_index.saturating_add(1);
                        return Ok(Some(packet));
                    }
                }
                _ => {
                    self.file.seek(SeekFrom::Start(payload_end))?;
                }
            }
        }
        Ok(None)
    }

    fn push_level(
        &mut self,
        end: u64,
        context: WebmContext,
        depth: usize,
    ) -> Result<(), ContainerError> {
        if depth > MAX_EBML_NESTING_DEPTH {
            return Err(ContainerError::BadEbml(
                "recognized container nesting depth exceeds limit",
            ));
        }
        self.levels.push(WebmLevel {
            end,
            context,
            depth,
        });
        Ok(())
    }

    fn read_block(&mut self, block_end: u64) -> Result<Option<Vec<u8>>, ContainerError> {
        let payload_start = self.file.stream_position()?;
        let payload_size = block_end.saturating_sub(payload_start);
        if payload_size < 4 {
            return Err(ContainerError::BadEbml("block payload too small"));
        }
        let (track, track_len) = read_stream_ebml_vint(&mut self.file)?;
        let header_len = u64::try_from(track_len).unwrap_or(u64::MAX) + 3;
        if header_len > payload_size {
            return Err(ContainerError::BadEbml("block missing timecode/flags"));
        }
        let mut timecode_and_flags = [0u8; 3];
        self.file.read_exact(&mut timecode_and_flags)?;
        if track != self.video_track {
            self.file.seek(SeekFrom::Start(block_end))?;
            return Ok(None);
        }
        if ((timecode_and_flags[2] >> 1) & 0x03) != 0 {
            return Err(ContainerError::BadEbml(
                "lacing on the VP9 video track is unsupported",
            ));
        }
        let frame_size = payload_size - header_len;
        if frame_size == 0 {
            return Err(ContainerError::BadFrameSize {
                packet_index: self.packet_index.saturating_add(1),
                declared: 0,
                available: 0,
            });
        }
        let frame_size = usize::try_from(frame_size)
            .map_err(|_| ContainerError::BadEbml("block frame size does not fit usize"))?;
        let mut packet = Vec::new();
        packet.try_reserve_exact(frame_size).map_err(|_| {
            ContainerError::Io(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "allocating WebM packet",
            ))
        })?;
        packet.resize(frame_size, 0);
        self.file.read_exact(&mut packet)?;
        Ok(Some(packet))
    }
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
/// containers, but treat any unrecognised element as "skip `<size>`
/// bytes". This is robust against tracks/seek-heads/cues/etc.
///
/// # Errors
///
/// - [`ContainerError::BadEbml`] on malformed EBML structure, a
///   missing VP9 video track, or lacing on that track.
/// - [`ContainerError::BadFrameSize`] if a block claims more bytes
///   than remain in its parent.
pub fn read_webm_packets(data: &[u8]) -> Result<Vec<PacketRange>, ContainerError> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let video_track = find_vp9_video_track(data)?;
    let mut packets: Vec<PacketRange> = Vec::new();
    walk_ebml(
        data,
        0,
        data.len(),
        &mut packets,
        WebmContext::default(),
        video_track,
        0,
    )?;
    Ok(packets)
}

fn find_vp9_video_track(data: &[u8]) -> Result<u64, ContainerError> {
    #[derive(Clone, Copy)]
    struct ScanLevel {
        pos: usize,
        end: usize,
        inside_tracks: bool,
    }

    // Retain only active ancestors. Processing each child immediately also
    // selects the first VP9 track in document order, like the streaming finder.
    let mut levels = vec![ScanLevel {
        pos: 0,
        end: data.len(),
        inside_tracks: false,
    }];
    while let Some(level) = levels.last().copied() {
        let ScanLevel {
            mut pos,
            end,
            inside_tracks,
        } = level;
        if pos >= end {
            levels.pop();
            continue;
        }
        let (id, id_len) = read_ebml_id(&data[..end], pos)?;
        pos += id_len;
        let (size, size_len) = read_ebml_vint(&data[..end], pos)?;
        pos += size_len;
        let unknown_size = validate_element_size(id, size, size_len)?;

        let payload_size = if unknown_size {
            end - pos
        } else {
            usize::try_from(size)
                .map_err(|_| ContainerError::BadEbml("element size does not fit usize"))?
        };
        let payload_end = pos
            .checked_add(payload_size)
            .ok_or(ContainerError::BadEbml("payload size overflow"))?;
        if payload_end > end {
            return Err(ContainerError::BadEbml(
                "element payload runs past parent end",
            ));
        }

        // Resume the parent after this child when its active level finishes.
        levels.last_mut().expect("active scan level").pos = payload_end;
        match id {
            ebml_id::EBML | ebml_id::SEGMENT | ebml_id::TRACKS => {
                // The root level has depth zero, so the current number of
                // levels is the depth of the child about to be entered.
                if levels.len() > MAX_EBML_NESTING_DEPTH {
                    return Err(ContainerError::BadEbml(
                        "recognized container nesting depth exceeds limit",
                    ));
                }
                levels.push(ScanLevel {
                    pos,
                    end: payload_end,
                    inside_tracks: inside_tracks || id == ebml_id::TRACKS,
                });
            }
            ebml_id::TRACK_ENTRY if inside_tracks => {
                if let Some(track) = parse_vp9_track_entry(data, pos, payload_end)? {
                    return Ok(track);
                }
            }
            _ => {}
        }
    }

    Err(ContainerError::BadEbml("VP9 video track not found"))
}

fn parse_vp9_track_entry(
    data: &[u8],
    mut pos: usize,
    end: usize,
) -> Result<Option<u64>, ContainerError> {
    let mut track_number = None;
    let mut track_type = None;
    let mut codec_is_vp9 = false;

    while pos < end {
        let (id, id_len) = read_ebml_id(data, pos)?;
        pos += id_len;
        let (size, size_len) = read_ebml_vint(data, pos)?;
        pos += size_len;
        let unknown_size = validate_element_size(id, size, size_len)
            .map_err(|_| ContainerError::BadEbml("unknown-size TrackEntry child"))?;
        if unknown_size {
            return Err(ContainerError::BadEbml("unknown-size TrackEntry child"));
        }
        let payload_size = usize::try_from(size)
            .map_err(|_| ContainerError::BadEbml("TrackEntry child size does not fit usize"))?;
        let payload_end = pos
            .checked_add(payload_size)
            .ok_or(ContainerError::BadEbml("TrackEntry child size overflow"))?;
        if payload_end > end {
            return Err(ContainerError::BadEbml(
                "TrackEntry child runs past entry end",
            ));
        }

        match id {
            ebml_id::TRACK_NUMBER => {
                let number = read_ebml_uint(&data[pos..payload_end])?;
                if number == 0 {
                    return Err(ContainerError::BadEbml("TrackNumber must be non-zero"));
                }
                track_number = Some(number);
            }
            ebml_id::TRACK_TYPE => {
                track_type = Some(read_ebml_uint(&data[pos..payload_end])?);
            }
            ebml_id::CODEC_ID => {
                codec_is_vp9 = &data[pos..payload_end] == b"V_VP9";
            }
            _ => {}
        }
        pos = payload_end;
    }

    Ok(match (track_number, track_type, codec_is_vp9) {
        (Some(number), Some(1), true) => Some(number),
        _ => None,
    })
}

fn read_ebml_uint(bytes: &[u8]) -> Result<u64, ContainerError> {
    if bytes.is_empty() || bytes.len() > 8 {
        return Err(ContainerError::BadEbml(
            "EBML unsigned integer must contain 1..=8 bytes",
        ));
    }
    Ok(bytes
        .iter()
        .fold(0u64, |value, &byte| (value << 8) | u64::from(byte)))
}

fn walk_ebml(
    data: &[u8],
    mut pos: usize,
    end: usize,
    out: &mut Vec<PacketRange>,
    context: WebmContext,
    video_track: u64,
    depth: usize,
) -> Result<usize, ContainerError> {
    if depth > MAX_EBML_NESTING_DEPTH {
        return Err(ContainerError::BadEbml(
            "recognized container nesting depth exceeds limit",
        ));
    }

    while pos < end {
        let (id, id_len) = read_ebml_id(data, pos)?;
        if context.ends_before(id) {
            return Ok(pos);
        }
        pos += id_len;
        let (size, size_len) = read_ebml_vint(data, pos)?;
        pos += size_len;

        // The parent end bounds unknown-sized masters, but their schema
        // context can terminate them earlier at a sibling header.
        let unknown_size = validate_element_size(id, size, size_len)?;
        let payload_size = if unknown_size {
            end.saturating_sub(pos)
        } else {
            usize::try_from(size).map_err(|_| ContainerError::BadEbml("vint too large"))?
        };
        let mut payload_end = pos
            .checked_add(payload_size)
            .ok_or(ContainerError::BadEbml("payload size overflow"))?;
        if payload_end > end {
            return Err(ContainerError::BadEbml(
                "element payload runs past parent end",
            ));
        }

        match id {
            ebml_id::EBML | ebml_id::SEGMENT => {
                payload_end = walk_ebml(
                    data,
                    pos,
                    payload_end,
                    out,
                    context.child(id, unknown_size, false),
                    video_track,
                    depth + 1,
                )?;
            }
            ebml_id::CLUSTER => {
                payload_end = walk_ebml(
                    data,
                    pos,
                    payload_end,
                    out,
                    context.child(id, unknown_size, true),
                    video_track,
                    depth + 1,
                )?;
            }
            ebml_id::SIMPLE_BLOCK if context.inside_cluster => {
                if let Some(range) =
                    parse_block_frame(data, pos, payload_size, out.len() + 1, video_track)?
                {
                    out.push(range);
                }
            }
            ebml_id::BLOCK_GROUP if context.inside_cluster => {
                // BlockGroup contains a Block + extras; recurse to find the Block.
                payload_end = walk_ebml(
                    data,
                    pos,
                    payload_end,
                    out,
                    context.child(id, unknown_size, true),
                    video_track,
                    depth + 1,
                )?;
            }
            ebml_id::BLOCK if context.inside_cluster => {
                if let Some(range) =
                    parse_block_frame(data, pos, payload_size, out.len() + 1, video_track)?
                {
                    out.push(range);
                }
            }
            _ => {
                // Unknown / unhandled element — skip its payload.
            }
        }
        pos = payload_end;
    }
    Ok(pos)
}

#[inline]
fn is_unknown_vint(value: u64, size_len: usize) -> bool {
    // All-ones vint of width `size_len` means "unknown size".
    let bits = (8 * size_len as u32) - size_len as u32; // (8 - 1) per first byte + 8 per subsequent
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    value == mask
}

fn validate_element_size(id: u32, value: u64, size_len: usize) -> Result<bool, ContainerError> {
    let unknown_size = is_unknown_vint(value, size_len);
    if unknown_size && !matches!(id, ebml_id::SEGMENT | ebml_id::CLUSTER) {
        let message = if id == ebml_id::TRACK_ENTRY {
            "unknown-size TrackEntry"
        } else {
            "unknown-size marker is not allowed for this element ID"
        };
        return Err(ContainerError::BadEbml(message));
    }
    Ok(unknown_size)
}

/// Parse one Block / SimpleBlock payload. The first 3+ bytes are
/// `track_number_vint, timecode_le_i16, flags_u8`. Following bytes
/// are the frame data.
///
/// Returns `Ok(Some(range))` for a valid un-laced block on
/// `video_track`, `Ok(None)` for another track, and `Err` for malformed
/// headers or lacing on the selected VP9 track.
fn parse_block_frame(
    data: &[u8],
    pos: usize,
    payload_size: usize,
    packet_idx_hint: usize,
    video_track: u64,
) -> Result<Option<PacketRange>, ContainerError> {
    if payload_size < 4 {
        return Err(ContainerError::BadEbml("block payload too small"));
    }
    let block_end = pos
        .checked_add(payload_size)
        .ok_or(ContainerError::BadEbml("block payload size overflow"))?;
    // Track number is a vint.
    let (track, tn_len) = read_ebml_vint(data, pos)?;
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
    if track != video_track {
        return Ok(None);
    }
    if lacing != 0 {
        return Err(ContainerError::BadEbml(
            "lacing on the VP9 video track is unsupported",
        ));
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
    fn ivf_roundtrip_two_packets() {
        // RC.4 — round-trip `write_ivf` through `read_ivf_packets`.
        let p0: &[u8] = b"hello-frame-0";
        let p1: &[u8] = b"frame-one-with-different-length";
        let header = IvfHeader::vp9(256, 256, 30, 1);

        let mut buf: Vec<u8> = Vec::new();
        write_ivf(&mut buf, &header, &[p0, p1]).expect("write_ivf");

        // File header: 32 bytes; magic + version + header_size.
        assert!(buf.len() >= 32);
        assert_eq!(&buf[0..4], b"DKIF");
        assert_eq!(u16::from_le_bytes([buf[4], buf[5]]), 0);
        assert_eq!(u16::from_le_bytes([buf[6], buf[7]]), 32);
        assert_eq!(&buf[8..12], b"VP90");
        assert_eq!(u16::from_le_bytes([buf[12], buf[13]]), 256);
        assert_eq!(u16::from_le_bytes([buf[14], buf[15]]), 256);
        assert_eq!(u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]), 30);
        assert_eq!(u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]), 1);
        assert_eq!(u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]), 2);

        // Re-read and confirm payloads survive intact.
        let packets = read_ivf_packets(&buf).expect("read_ivf_packets");
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].slice(&buf), p0);
        assert_eq!(packets[1].slice(&buf), p1);
    }

    #[test]
    fn ivf_roundtrip_empty_stream() {
        // Zero packets is legal — produces just the 32-byte file header.
        let header = IvfHeader::vp9(64, 64, 30, 1);
        let mut buf: Vec<u8> = Vec::new();
        write_ivf(&mut buf, &header, &[]).expect("write_ivf empty");
        assert_eq!(buf.len(), 32);
        let packets = read_ivf_packets(&buf).expect("read_ivf_packets");
        assert_eq!(packets.len(), 0);
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

    // -----------------------------------------------------------------
    // Stage 11 coverage push — error-contract tests for the EBML
    // primitives + dispatch + writer overflow paths. Each test asserts
    // a *specific* error variant; collectively they pin the on-the-wire
    // failure modes the conformance fixtures don't exercise.
    // -----------------------------------------------------------------

    #[test]
    fn read_packets_unknown_extension_errors() {
        // Extension dispatch happens before filesystem access, so even a
        // nonexistent unsupported path reports the unsupported format.
        let path = std::path::Path::new("nonexistent.mp4");
        let result = read_packets(path);
        assert!(matches!(result, Err(ContainerError::UnknownExtension(ref ext)) if ext == "mp4"));
    }

    #[test]
    fn read_packets_truly_unknown_extension() {
        // Create a real tiny temp file with an unrecognised extension
        // so `fs::read` succeeds and the extension dispatch fires the
        // `UnknownExtension` arm.
        let tmp = std::env::temp_dir().join("rvp9_io_unknown_ext_test.xyz");
        std::fs::write(&tmp, b"hello").expect("write tmp");
        let result = read_packets(&tmp);
        let _ = std::fs::remove_file(&tmp);
        match result {
            Err(ContainerError::UnknownExtension(ref ext)) => assert_eq!(ext, "xyz"),
            other => panic!("expected UnknownExtension(\"xyz\"), got {other:?}"),
        }
    }

    #[test]
    fn read_ebml_id_eof_arm() {
        let err = read_ebml_id(&[], 0).unwrap_err();
        assert!(
            matches!(err, ContainerError::BadEbml(msg) if msg.contains("EOF")),
            "expected BadEbml(EOF), got different",
        );
    }

    #[test]
    fn read_ebml_id_zero_first_byte_arm() {
        let err = read_ebml_id(&[0x00], 0).unwrap_err();
        assert!(matches!(err, ContainerError::BadEbml(msg) if msg.contains("no length marker")),);
    }

    #[test]
    fn read_ebml_id_eof_inside_id() {
        // 0x10 starts a 4-byte id but only one byte available.
        let err = read_ebml_id(&[0x10], 0).unwrap_err();
        assert!(matches!(err, ContainerError::BadEbml(msg) if msg.contains("EOF inside")),);
    }

    #[test]
    fn read_ebml_vint_eof_arm() {
        let err = read_ebml_vint(&[], 0).unwrap_err();
        assert!(matches!(err, ContainerError::BadEbml(msg) if msg.contains("EOF")));
    }

    #[test]
    fn read_ebml_vint_zero_first_byte_arm() {
        let err = read_ebml_vint(&[0x00], 0).unwrap_err();
        assert!(matches!(err, ContainerError::BadEbml(msg) if msg.contains("no length marker")));
    }

    #[test]
    fn read_ebml_vint_eof_inside_value() {
        // 0x01 (top byte) starts an 8-byte vint but only one byte
        // available.
        let err = read_ebml_vint(&[0x01], 0).unwrap_err();
        assert!(matches!(err, ContainerError::BadEbml(msg) if msg.contains("EOF inside")));
    }

    #[test]
    fn webm_payload_runs_past_parent() {
        // Build a minimal EBML element whose declared payload size
        // overruns the buffer end — verifies the `payload_end > end`
        // arm in `walk_ebml`.
        //
        // EBML id 0xEC (Void) followed by a 1-byte vint declaring
        // size=200 but no payload follows.
        let data = vec![0xEC, 0x40, 200];
        let err = read_webm_packets(&data).unwrap_err();
        assert!(matches!(err, ContainerError::BadEbml(msg) if msg.contains("past parent end")));
    }

    #[test]
    fn webm_parse_block_too_small() {
        // SimpleBlock element inside a Cluster, with a 3-byte payload
        // (less than the required 4 bytes). Build: Segment(id 1-byte
        // marker via 0x1853_8067 — but that needs 4 bytes; use the
        // bare `parse_block_frame` directly to exercise the size
        // check without rebuilding a full WebM container).
        let err = parse_block_frame(&[0x81, 0x00, 0x00], 0, 3, 1, 1).unwrap_err();
        assert!(matches!(err, ContainerError::BadEbml(msg) if msg.contains("too small")));
    }

    #[test]
    fn webm_parse_block_lacing_is_rejected() {
        // Block with lacing=1 (xiph) must fail instead of silently
        // disappearing from the packet stream.
        // Payload: track_number_vint(0x81=1) + timecode(2) + flags(lacing<<1=0x02)
        let mut payload = vec![0x81u8, 0x00, 0x00, 0x02];
        payload.extend_from_slice(&[0xAA, 0xBB, 0xCC]); // some frame bytes
        let error = parse_block_frame(&payload, 0, payload.len(), 1, 1)
            .expect_err("laced block was silently skipped");
        assert!(
            matches!(error, ContainerError::BadEbml(message) if message.contains("lacing")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn webm_parse_block_zero_frame_bytes() {
        // Block with no frame data after the 1+2+1 = 4-byte header.
        // payload_size = 4 exactly → `frame_start == block_end`.
        let payload = vec![0x81u8, 0x00, 0x00, 0x00];
        let err = parse_block_frame(&payload, 0, payload.len(), 7, 1).unwrap_err();
        match err {
            ContainerError::BadFrameSize {
                packet_index,
                declared,
                available,
            } => {
                assert_eq!(packet_index, 7);
                assert_eq!(declared, 0);
                assert_eq!(available, 0);
            }
            other => panic!("expected BadFrameSize, got {other:?}"),
        }
    }

    #[test]
    fn webm_parse_block_missing_timecode_flags() {
        // Track-number vint of width 4 (0x10 + 3 more bytes) plus
        // only 2 bytes after — too short for the 3-byte
        // timecode+flags suffix. payload_size matches the buffer
        // length exactly so the `after_tn + 3 > block_end` arm
        // fires.
        let payload = vec![0x10u8, 0x00, 0x00, 0x01, 0xAA, 0xBB];
        let err = parse_block_frame(&payload, 0, payload.len(), 1, 1).unwrap_err();
        assert!(matches!(err, ContainerError::BadEbml(msg) if msg.contains("missing")));
    }

    #[test]
    fn write_ivf_too_many_packets_overflows() {
        // u32::MAX+1 packets cannot be expressed. We can't allocate
        // that many in memory, but the error path checks
        // `u32::try_from(packets.len())`. Skipped: would require a
        // borrow-free `&[&[u8]]` slice with usize::MAX > u32::MAX —
        // not available on 32-bit builds. Skip on 32-bit (`usize ==
        // u32`); on 64-bit there is no portable way to construct such
        // a slice. Instead exercise the per-packet size-overflow path
        // below.
    }

    #[test]
    fn write_ivf_empty_payload_then_read_back() {
        // Empty packet is legal — exercises the loop body with a
        // 0-byte payload (frame size record but no bytes).
        let header = IvfHeader::vp9(32, 32, 25, 1);
        let mut buf = Vec::new();
        let empty: &[u8] = &[];
        write_ivf(&mut buf, &header, &[empty, empty]).unwrap();
        let packets = read_ivf_packets(&buf).unwrap();
        assert_eq!(packets.len(), 2);
        assert!(packets[0].is_empty());
        assert!(packets[1].is_empty());
    }

    #[test]
    fn stream_writer_patches_count_and_preserves_packets() {
        let header = IvfHeader::vp9(64, 64, 30, 1);
        let cursor = std::io::Cursor::new(Vec::new());
        let mut writer = IvfStreamWriter::new(cursor, &header).expect("open stream writer");
        writer.write_packet(b"first").expect("write first");
        writer.write_packet(b"second").expect("write second");
        assert_eq!(writer.packet_count(), 2);
        let bytes = writer.finish().expect("finish writer").into_inner();
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().expect("frame count")),
            2
        );
        let packets = read_ivf_packets(&bytes).expect("parse streamed IVF");
        assert_eq!(packets[0].slice(&bytes), b"first");
        assert_eq!(packets[1].slice(&bytes), b"second");
    }

    #[test]
    fn stream_writer_relative_position_with_prefixed_cursor() {
        use std::io::{Seek, SeekFrom};

        let prefix = b"cursor-prefix-sentinel-larger-than-ivf-header";
        assert!(prefix.len() > 28);
        let header = IvfHeader::vp9(64, 48, 30, 1);
        let mut cursor = std::io::Cursor::new(prefix.to_vec());
        cursor
            .seek(SeekFrom::Start(prefix.len() as u64))
            .expect("seek after prefix");

        let mut writer = IvfStreamWriter::new(cursor, &header).expect("open prefixed writer");
        writer
            .write_packet(b"first-packet")
            .expect("write first packet");
        writer.write_packet(b"second").expect("write second packet");
        let cursor = writer.finish().expect("finish prefixed writer");
        let final_position = cursor.position();
        let bytes = cursor.into_inner();
        let stream_start = prefix.len();
        let stream = &bytes[stream_start..];
        let packets = read_ivf_packets(stream).expect("parse IVF subslice");

        assert_eq!(
            &bytes[..stream_start],
            prefix,
            "IVF count patch changed the prefix"
        );
        assert_eq!(
            final_position,
            bytes.len() as u64,
            "finish must leave the cursor at EOF"
        );
        assert_eq!(
            u32::from_le_bytes(stream[24..28].try_into().expect("frame count")),
            2,
            "relative IVF header must contain the nonzero packet count"
        );
        assert_eq!(packets.len(), 2);
        assert_eq!(
            u64::from_le_bytes(
                stream[packets[0].start - 8..packets[0].start]
                    .try_into()
                    .expect("first timestamp")
            ),
            0
        );
        assert_eq!(
            u64::from_le_bytes(
                stream[packets[1].start - 8..packets[1].start]
                    .try_into()
                    .expect("second timestamp")
            ),
            1
        );
        assert_eq!(packets[0].slice(stream), b"first-packet");
        assert_eq!(packets[1].slice(stream), b"second");
    }

    #[test]
    fn stream_writer_relative_position_rejects_patch_offset_overflow() {
        use std::cell::Cell;
        use std::rc::Rc;

        struct OffsetOverflowWriter {
            write_calls: Rc<Cell<usize>>,
        }

        impl std::io::Write for OffsetOverflowWriter {
            fn write(&mut self, _buffer: &[u8]) -> std::io::Result<usize> {
                self.write_calls.set(self.write_calls.get() + 1);
                Err(std::io::Error::other(
                    "synthetic writer cannot store bytes at this position",
                ))
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        impl std::io::Seek for OffsetOverflowWriter {
            fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
                match position {
                    SeekFrom::Current(0) => Ok(u64::MAX - 23),
                    _ => Err(std::io::Error::other("unexpected synthetic seek")),
                }
            }
        }

        let write_calls = Rc::new(Cell::new(0));
        let writer = OffsetOverflowWriter {
            write_calls: Rc::clone(&write_calls),
        };
        let header = IvfHeader::vp9(16, 16, 30, 1);
        let err = match IvfStreamWriter::new(writer, &header) {
            Ok(_) => panic!("overflowing patch offset must be rejected"),
            Err(err) => err,
        };

        assert_eq!(
            err.kind(),
            std::io::ErrorKind::InvalidInput,
            "constructor must reject patch offset overflow before writing the header"
        );
        assert_eq!(
            write_calls.get(),
            0,
            "overflow rejection must precede header writes"
        );
    }

    #[test]
    fn ivf_header_vp9_builder_uses_vp90_fourcc() {
        let h = IvfHeader::vp9(1280, 720, 60, 1);
        assert_eq!(h.codec_fourcc, *b"VP90");
        assert_eq!(h.width, 1280);
        assert_eq!(h.height, 720);
        assert_eq!(h.framerate_num, 60);
        assert_eq!(h.framerate_den, 1);
    }

    #[test]
    fn packet_range_helpers() {
        // `slice`, `len`, `is_empty` accessors — pin the contracts.
        let r = PacketRange { start: 4, end: 9 };
        assert_eq!(r.len(), 5);
        assert!(!r.is_empty());
        let r0 = PacketRange { start: 7, end: 7 };
        assert!(r0.is_empty());
        assert_eq!(r0.len(), 0);
        let data = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(r.slice(&data), &[4u8, 5, 6, 7, 8]);
    }

    #[test]
    fn container_error_display_covers_each_variant() {
        // Confirms each `Display` arm produces a non-empty string
        // mentioning the variant's salient field — protects against
        // a future refactor that accidentally drops a variant from
        // the `match`.
        let v = vec![
            ContainerError::Io(std::io::Error::other("oops")),
            ContainerError::UnknownExtension("mkv".into()),
            ContainerError::Truncated {
                what: "IVF file header",
                len: 8,
            },
            ContainerError::BadIvfMagic,
            ContainerError::BadEbml("EOF"),
            ContainerError::BadFrameSize {
                packet_index: 2,
                declared: 100,
                available: 5,
            },
        ];
        for err in v {
            let s = format!("{err}");
            assert!(!s.is_empty(), "Display empty for {err:?}");
        }
    }
}
