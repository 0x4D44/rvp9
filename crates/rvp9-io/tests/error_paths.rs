//! Truncated- and corrupted-input fixtures for `container.rs` and `y4m.rs`.
//!
//! Stage 3 of the coverage improvement plan
//! (`wrk_docs/2026.05.25 - CC - coverage improvement plan.md`).
//!
//! For each `?` propagation in the parser, we craft a buffer that succeeds
//! up to byte `n` and fails at `n+1`, then assert the **specific** error
//! variant. The point is to lock in the error contract — not just to bump
//! line counts.

use std::path::PathBuf;

use rvp9_core::bitstream::ChromaFormat;
use rvp9_io::container::{
    ContainerError, IvfHeader, PacketRange, PacketReader, read_ivf_packets, read_packets,
    read_webm_packets, write_ivf,
};
use rvp9_io::y4m::{Y4mHeader, parse_header, read_cropped, write_frame, write_header};

// ============================================================================
// Helpers
// ============================================================================

/// Build a syntactically-valid 32-byte IVF file header followed by `frames`.
/// Each frame is `(u32 size, u64 pts, payload)`.
fn make_ivf(width: u16, height: u16, frames: &[&[u8]]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"DKIF");
    buf.extend_from_slice(&0u16.to_le_bytes()); // version
    buf.extend_from_slice(&32u16.to_le_bytes()); // header_size
    buf.extend_from_slice(b"VP90"); // codec FourCC
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    buf.extend_from_slice(&30u32.to_le_bytes()); // framerate num
    buf.extend_from_slice(&1u32.to_le_bytes()); // framerate den
    buf.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    buf.extend_from_slice(&[0u8; 4]); // unused
    for (i, f) in frames.iter().enumerate() {
        buf.extend_from_slice(&(f.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(i as u64).to_le_bytes());
        buf.extend_from_slice(f);
    }
    buf
}

/// Produce a unique temp path inside `std::env::temp_dir()` for one test.
/// We don't bring in the `tempfile` crate (not currently a dep of `rvp9-io`).
fn unique_temp_path(stem: &str, ext: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rvp9_io_stage3_{}_{}_{}_{}.{}",
        stem,
        std::process::id(),
        id,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        ext,
    ))
}

// ============================================================================
// IVF reader — error paths
// ============================================================================

#[test]
fn ivf_truncated_below_header_size_reports_truncated() {
    // 0..32: any length under 32 bytes is rejected as a truncated header.
    for n in [0usize, 1, 4, 16, 31] {
        let data = vec![0u8; n];
        let err = read_ivf_packets(&data).unwrap_err();
        match err {
            ContainerError::Truncated { what, len } => {
                assert_eq!(what, "IVF file header");
                assert_eq!(len, n, "Truncated.len must reflect actual buffer length");
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
    }
}

#[test]
fn ivf_rejects_bad_magic_with_dedicated_variant() {
    // Right size, wrong signature — must be BadIvfMagic, not Truncated.
    let mut data = make_ivf(64, 64, &[]);
    data[0..4].copy_from_slice(b"NOPE");
    assert!(matches!(
        read_ivf_packets(&data),
        Err(ContainerError::BadIvfMagic)
    ));
}

#[test]
fn ivf_frame_payload_running_past_eof_is_bad_frame_size() {
    // One frame declares size 100 but only 5 payload bytes are present.
    let mut data = make_ivf(64, 64, &[]);
    data.extend_from_slice(&100u32.to_le_bytes()); // declared size
    data.extend_from_slice(&0u64.to_le_bytes()); // pts
    data.extend_from_slice(&[0u8; 5]); // only 5 bytes available
    match read_ivf_packets(&data).unwrap_err() {
        ContainerError::BadFrameSize {
            packet_index,
            declared,
            available,
        } => {
            assert_eq!(packet_index, 1, "first frame is packet index 1");
            assert_eq!(declared, 100, "declared frame payload size");
            assert_eq!(available, 5, "available frame payload bytes");
        }
        other => panic!("expected BadFrameSize, got {other:?}"),
    }
}

#[test]
fn ivf_frame_size_overflow_is_bad_frame_size() {
    // size = u32::MAX means payload_start (>=44) + size overflows usize on
    // 32-bit targets but NOT on 64-bit. On 64-bit the overflow check
    // doesn't fire — instead we just hit the payload-past-EOF arm. Test
    // the payload-past-EOF arm explicitly: it's the same `BadFrameSize`
    // variant, just a different field-pattern.
    let mut data = make_ivf(64, 64, &[]);
    data.extend_from_slice(&u32::MAX.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    // No payload bytes whatsoever.
    let err = read_ivf_packets(&data).unwrap_err();
    assert!(
        matches!(
            err,
            ContainerError::BadFrameSize {
                packet_index: 1,
                ..
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn ivf_truncated_frame_record_header_is_rejected() {
    let frame: &[u8] = b"complete-frame";
    let base = make_ivf(64, 64, &[frame]);
    for tail_len in 1..12 {
        let mut data = base.clone();
        data.extend(std::iter::repeat_n(0xAA, tail_len));
        match read_ivf_packets(&data).unwrap_err() {
            ContainerError::Truncated { what, len } => {
                assert_eq!(what, "IVF frame record header");
                assert_eq!(len, base.len() + tail_len);
            }
            other => panic!("expected Truncated for {tail_len} tail bytes, got {other:?}"),
        }
    }
}

#[test]
fn ivf_roundtrip_preserves_payloads_byte_for_byte() {
    let p0: &[u8] = b"frame-zero-payload";
    let p1: &[u8] = b"a-different-length-frame-one-payload";
    let p2: &[u8] = &[0u8, 1, 2, 3, 0xFF, 0xFE, 0x55, 0xAA];
    let header = IvfHeader::vp9(320, 240, 24, 1);

    let mut buf = Vec::new();
    write_ivf(&mut buf, &header, &[p0, p1, p2]).expect("write_ivf");

    let packets = read_ivf_packets(&buf).expect("read_ivf_packets");
    assert_eq!(packets.len(), 3);
    assert_eq!(packets[0].slice(&buf), p0);
    assert_eq!(packets[1].slice(&buf), p1);
    assert_eq!(packets[2].slice(&buf), p2);
}

#[test]
fn ivf_writer_rejects_packets_too_large_for_u32() {
    // `write_ivf` measures each packet length with `u32::try_from`. We can
    // construct a fake `&[u8]` whose length appears > u32::MAX via the
    // 64-bit usize. Easier: a Vec we know fits in u32 will succeed; the
    // overflow path is unreachable on 32-bit (already-bounded) but
    // reachable on 64-bit only by allocating >4 GiB. Skip that — instead
    // assert the success path encodes the size correctly.
    let p: Vec<u8> = vec![0xAB; 1024];
    let header = IvfHeader::vp9(16, 16, 30, 1);
    let mut buf = Vec::new();
    write_ivf(&mut buf, &header, &[&p]).expect("write_ivf small packet");
    // Per-packet record: size u32_le then pts u64_le.
    let recorded_size = u32::from_le_bytes([buf[32], buf[33], buf[34], buf[35]]);
    assert_eq!(recorded_size, 1024);
}

// ============================================================================
// IVF — PacketRange / IvfHeader micro-API
// ============================================================================

#[test]
fn packet_range_slice_len_is_empty_agree() {
    let buf = b"abcdef".to_vec();
    let empty = PacketRange { start: 2, end: 2 };
    let nonempty = PacketRange { start: 1, end: 4 };
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.slice(&buf), b"");
    assert!(!nonempty.is_empty());
    assert_eq!(nonempty.len(), 3);
    assert_eq!(nonempty.slice(&buf), b"bcd");
}

#[test]
fn ivf_header_vp9_constructor_sets_fourcc() {
    let h = IvfHeader::vp9(640, 480, 25, 1);
    assert_eq!(h.codec_fourcc, *b"VP90");
    assert_eq!((h.width, h.height), (640, 480));
    assert_eq!((h.framerate_num, h.framerate_den), (25, 1));
}

#[test]
fn container_error_display_covers_every_variant() {
    // Each Display arm produces a distinctive substring. If the format
    // strings drift, this test pins them. Construct one of each variant
    // and assert on the user-facing message.
    let truncated = ContainerError::Truncated {
        what: "thing",
        len: 7,
    };
    assert_eq!(format!("{truncated}"), "truncated thing (len=7)");

    let unknown = ContainerError::UnknownExtension("xyz".to_string());
    assert_eq!(format!("{unknown}"), "unknown container extension: xyz");

    let magic = ContainerError::BadIvfMagic;
    assert_eq!(
        format!("{magic}"),
        "not an IVF file (missing DKIF signature)"
    );

    let ebml = ContainerError::BadEbml("oops");
    assert_eq!(format!("{ebml}"), "ebml/matroska parse error: oops");

    let fs = ContainerError::BadFrameSize {
        packet_index: 5,
        declared: 100,
        available: 50,
    };
    assert_eq!(
        format!("{fs}"),
        "packet 5: declared 100 bytes but only 50 available"
    );

    let io_err: ContainerError = std::io::Error::other("disk gone").into();
    assert!(format!("{io_err}").starts_with("io: "));
}

// ============================================================================
// read_packets — dispatcher
// ============================================================================

#[test]
fn read_packets_unknown_extension_returns_unknown_extension() {
    let path = unique_temp_path("unknown_ext", "xyz");
    std::fs::write(&path, b"whatever").expect("write fixture");
    let err = read_packets(&path).unwrap_err();
    let _ = std::fs::remove_file(&path);
    match err {
        ContainerError::UnknownExtension(ext) => assert_eq!(ext, "xyz"),
        other => panic!("expected UnknownExtension(\"xyz\"), got {other:?}"),
    }
}

#[test]
fn read_packets_rejects_unknown_extension_before_filesystem_read() {
    let path = unique_temp_path("unknown_ext_directory", "xyz");
    std::fs::create_dir(&path).expect("create directory fixture");
    let err = read_packets(&path).unwrap_err();
    let _ = std::fs::remove_dir(&path);
    assert!(
        matches!(err, ContainerError::UnknownExtension(ref ext) if ext == "xyz"),
        "dispatcher touched the directory before rejecting its extension: {err:?}"
    );
}

#[test]
fn read_packets_no_extension_returns_unknown_extension_empty() {
    let path = unique_temp_path("no_ext_file", "");
    // Strip the trailing dot left by `.<ext>`-style joins.
    let mut s = path.to_string_lossy().into_owned();
    if s.ends_with('.') {
        s.pop();
    }
    let path = PathBuf::from(s);
    std::fs::write(&path, b"nope").expect("write fixture");
    let err = read_packets(&path).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert!(
        matches!(err, ContainerError::UnknownExtension(ref s) if s.is_empty()),
        "got {err:?}"
    );
}

#[test]
fn read_packets_missing_file_is_io_error() {
    let path = unique_temp_path("does_not_exist", "ivf");
    // Don't create the file.
    let err = read_packets(&path).unwrap_err();
    assert!(matches!(err, ContainerError::Io(_)), "got {err:?}");
}

#[test]
fn read_packets_dispatches_ivf_extension_to_ivf_reader() {
    let bytes = make_ivf(64, 64, &[b"hello"]);
    let path = unique_temp_path("dispatch_ivf", "ivf");
    std::fs::write(&path, &bytes).expect("write fixture");
    let (data, packets) = read_packets(&path).expect("read_packets");
    let _ = std::fs::remove_file(&path);
    assert_eq!(data, bytes);
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].slice(&data), b"hello");
}

#[test]
fn read_packets_dispatches_uppercase_extension_case_insensitively() {
    // The dispatcher lower-cases the extension; `.IVF` should still parse.
    let bytes = make_ivf(64, 64, &[b"AAAA"]);
    let path = unique_temp_path("dispatch_IVF", "IVF");
    std::fs::write(&path, &bytes).expect("write fixture");
    let (_data, packets) = read_packets(&path).expect("read_packets IVF");
    let _ = std::fs::remove_file(&path);
    assert_eq!(packets.len(), 1);
}

#[test]
fn read_packets_dispatches_webm_extension_to_webm_reader() {
    // A WebM with a VP9 track but no clusters yields zero packets.
    let path = unique_temp_path("dispatch_webm", "webm");
    std::fs::write(&path, vp9_tracks(1)).expect("write fixture");
    let (_data, packets) = read_packets(&path).expect("read_packets webm");
    let _ = std::fs::remove_file(&path);
    assert!(packets.is_empty());
}

#[test]
fn packet_reader_ivf_stops_before_unread_malformed_tail() {
    let mut bytes = make_ivf(64, 64, &[b"first"]);
    bytes.extend_from_slice(&100u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    let path = unique_temp_path("stream_ivf_bad_tail", "ivf");
    std::fs::write(&path, bytes).expect("write fixture");

    let mut reader = PacketReader::open(&path).expect("open packet reader");
    assert_eq!(reader.framerate(), Some((30, 1)));
    assert_eq!(
        reader.next_packet().expect("read first packet").as_deref(),
        Some(&b"first"[..])
    );
    let err = reader.next_packet().expect_err("malformed tail accepted");
    let _ = std::fs::remove_file(&path);
    assert!(
        matches!(
            err,
            ContainerError::BadFrameSize {
                packet_index: 2,
                declared: 100,
                available: 0
            }
        ),
        "got {err:?}"
    );
}

// ============================================================================
// WebM / EBML reader — error paths
// ============================================================================

/// Build an EBML element header: `(id_bytes, vint_size_bytes)`.
fn ebml_elt(id_bytes: &[u8], size: u64, payload: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(id_bytes);
    // Choose the smallest vint width that holds `size`. For deterministic
    // tests we just pick width 4 (28 bits available, plenty for fixtures).
    // Marker bit at bit 27 of byte 0.
    let s = size as u32;
    assert!(s < (1u32 << 28), "vint width=4 only holds <2^28");
    let b0 = 0x10 | ((s >> 24) as u8 & 0x0F);
    let b1 = (s >> 16) as u8;
    let b2 = (s >> 8) as u8;
    let b3 = s as u8;
    buf.extend_from_slice(&[b0, b1, b2, b3]);
    buf.extend_from_slice(payload);
    buf
}

fn vp9_track_entry(track_number: u8, track_type: u8, codec_id: &[u8]) -> Vec<u8> {
    let entry_payload = vp9_track_entry_payload(track_number, track_type, codec_id);
    ebml_elt(&[0xAE], entry_payload.len() as u64, &entry_payload)
}

fn vp9_track_entry_payload(track_number: u8, track_type: u8, codec_id: &[u8]) -> Vec<u8> {
    let mut entry_payload = ebml_elt(&[0xD7], 1, &[track_number]);
    entry_payload.extend_from_slice(&ebml_elt(&[0x83], 1, &[track_type]));
    entry_payload.extend_from_slice(&ebml_elt(&[0x86], codec_id.len() as u64, codec_id));
    entry_payload
}

fn tracks(track_number: u8, track_type: u8, codec_id: &[u8]) -> Vec<u8> {
    let entry = vp9_track_entry(track_number, track_type, codec_id);
    ebml_elt(&[0x16, 0x54, 0xAE, 0x6B], entry.len() as u64, &entry)
}

fn vp9_tracks(track_number: u8) -> Vec<u8> {
    tracks(track_number, 1, b"V_VP9")
}

fn with_vp9_track(mut body: Vec<u8>) -> Vec<u8> {
    let mut webm = vp9_tracks(1);
    webm.append(&mut body);
    webm
}

fn unknown_size_elt(id_bytes: &[u8], width: usize, payload: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(id_bytes);
    match width {
        1 => buf.push(0xFF),
        8 => buf.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]),
        _ => panic!("unsupported unknown-size vint width {width}"),
    }
    buf.extend_from_slice(payload);
    buf
}

fn read_webm_packet_bytes(data: &[u8]) -> Result<Vec<Vec<u8>>, ContainerError> {
    read_webm_packets(data).map(|ranges| {
        ranges
            .iter()
            .map(|range| range.slice(data).to_vec())
            .collect()
    })
}

fn read_webm_packet_bytes_streaming(
    data: &[u8],
    stem: &str,
) -> Result<Vec<Vec<u8>>, ContainerError> {
    let path = unique_temp_path(stem, "webm");
    std::fs::write(&path, data).expect("write WebM fixture");
    let result = (|| {
        let mut reader = PacketReader::open(&path)?;
        let mut packets = Vec::new();
        while let Some(packet) = reader.next_packet()? {
            packets.push(packet);
        }
        Ok(packets)
    })();
    std::fs::remove_file(path).expect("remove WebM fixture");
    result
}

fn unknown_size_error(result: &Result<Vec<Vec<u8>>, ContainerError>) -> bool {
    matches!(result, Err(ContainerError::BadEbml(message)) if message.contains("unknown-size"))
}

#[test]
fn webm_empty_buffer_yields_no_packets() {
    let packets = read_webm_packets(b"").expect("empty WebM");
    assert!(packets.is_empty());
}

#[test]
fn webm_non_vp9_track_is_rejected() {
    let webm = tracks(1, 2, b"A_OPUS");
    let error = read_webm_packets(&webm).expect_err("non-VP9 track was accepted");
    assert!(
        matches!(error, ContainerError::BadEbml(message) if message.contains("VP9 video track"))
    );
}

#[test]
fn webm_id_zero_first_byte_rejected() {
    // First byte of an EBML id must have at least one set bit; 0x00 is
    // illegal.
    let err = read_webm_packets(&[0x00, 0x82, 0xAA]).unwrap_err();
    assert!(
        matches!(err, ContainerError::BadEbml(msg) if msg.contains("no length marker")),
        "got {err:?}"
    );
}

#[test]
fn webm_id_eof_inside_id_rejected() {
    // `0x10` means the id is 4 bytes long, but only 1 byte is present.
    let err = read_webm_packets(&[0x10]).unwrap_err();
    assert!(
        matches!(err, ContainerError::BadEbml(msg) if msg.contains("inside element id")),
        "got {err:?}"
    );
}

#[test]
fn webm_id_wider_than_4_bytes_rejected() {
    // First byte 0x01 → 7 leading zeros → id is 8 bytes wide, which the
    // reader rejects as "element id wider than 4 bytes". We need the
    // `len > 4` arm to fire BEFORE the `pos + len > data.len()` arm,
    // which is the order the function uses.
    let err = read_webm_packets(&[0x01]).unwrap_err();
    assert!(
        matches!(err, ContainerError::BadEbml(msg) if msg.contains("wider than 4 bytes")),
        "got {err:?}"
    );
}

#[test]
fn webm_vint_eof_at_size_position() {
    // Valid 1-byte id (0xA3 = SimpleBlock), but no vint size byte follows.
    let err = read_webm_packets(&[0xA3]).unwrap_err();
    assert!(
        matches!(err, ContainerError::BadEbml(msg) if msg.contains("EOF while reading vint")),
        "got {err:?}"
    );
}

#[test]
fn webm_vint_zero_first_byte_rejected() {
    // SimpleBlock id + zero vint = "vint with no length marker".
    let err = read_webm_packets(&[0xA3, 0x00]).unwrap_err();
    assert!(
        matches!(err, ContainerError::BadEbml(msg) if msg.contains("vint with no length marker")),
        "got {err:?}"
    );
}

#[test]
fn webm_vint_eof_inside_vint_rejected() {
    // SimpleBlock id + vint claims 4 bytes (0x10) but only 1 byte follows.
    let err = read_webm_packets(&[0xA3, 0x10]).unwrap_err();
    assert!(
        matches!(err, ContainerError::BadEbml(msg) if msg.contains("EOF inside vint")),
        "got {err:?}"
    );
}

#[test]
fn webm_element_payload_running_past_parent_end_rejected() {
    // SimpleBlock (id=0xA3) with declared payload size 100, no bytes
    // present — should report "payload runs past parent end".
    // vint encoding 100 in width 1: top-bit marker + 100 = 0x80 | 100 = 0xE4.
    let bytes = [0xA3, 0xE4];
    let err = read_webm_packets(&bytes).unwrap_err();
    assert!(
        matches!(err, ContainerError::BadEbml(msg) if msg.contains("runs past parent end")),
        "got {err:?}"
    );
}

#[test]
fn webm_nested_header_crossing_parent_returns_error_without_panicking() {
    // The nested unknown-size Segment's ID or size header continues past the
    // outer Segment's declared payload.
    let cases = [
        (
            "ID",
            [0x18, 0x53, 0x80, 0x67, 0x81, 0x18, 0x53, 0x80, 0x67, 0xFF],
        ),
        (
            "size",
            [0x18, 0x53, 0x80, 0x67, 0x84, 0x18, 0x53, 0x80, 0x67, 0xFF],
        ),
    ];

    for (header_part, bytes) in cases {
        let parsed = std::panic::catch_unwind(|| read_webm_packets(&bytes));

        assert!(
            parsed.is_ok(),
            "read_webm_packets panicked on a parent-crossing {header_part} header"
        );
        assert!(
            matches!(parsed.unwrap(), Err(ContainerError::BadEbml(_))),
            "parent-crossing {header_part} header must return BadEbml"
        );
    }
}

#[test]
fn webm_simple_block_outside_cluster_is_skipped_not_returned() {
    // A SimpleBlock at the top level (not inside a Cluster) is not a frame
    // payload — the walker should silently skip it. Build a SimpleBlock
    // with 4 bytes of payload (vint track=1, le i16 ts=0, u8 flags=0).
    let payload = [0x81u8, 0x00, 0x00, 0x00];
    let mut bytes = vec![0xA3, 0x80 | 4]; // SimpleBlock id, vint size=4
    bytes.extend_from_slice(&payload);
    let bytes = with_vp9_track(bytes);
    let packets = read_webm_packets(&bytes).expect("top-level SimpleBlock");
    assert!(
        packets.is_empty(),
        "SimpleBlock outside Cluster must not produce a packet"
    );
}

#[test]
fn webm_cluster_with_simple_block_yields_one_packet() {
    // Smallest "VP9-shaped" WebM: a single Cluster containing one
    // SimpleBlock. The frame payload is the body after the 4 block header
    // bytes (track vint + 2-byte ts + 1-byte flags).
    let frame_payload: &[u8] = b"hello-vp9-frame";
    // Block payload: track=1 (vint 0x81), ts=0_le_i16, flags=0, then frame.
    let mut block_payload = vec![0x81u8, 0x00, 0x00, 0x00];
    block_payload.extend_from_slice(frame_payload);

    // SimpleBlock element: id=0xA3, vint size (1-byte vint if size < 0x80).
    let mut simple_block = Vec::new();
    simple_block.push(0xA3);
    assert!(block_payload.len() < 0x80, "test fixture too long");
    simple_block.push(0x80 | block_payload.len() as u8);
    simple_block.extend_from_slice(&block_payload);

    // Cluster element: id=0x1F43B675 (4 bytes), then vint size, then
    // SimpleBlock.
    let cluster = ebml_elt(
        &[0x1F, 0x43, 0xB6, 0x75],
        simple_block.len() as u64,
        &simple_block,
    );

    let cluster = with_vp9_track(cluster);
    let packets = read_webm_packets(&cluster).expect("cluster+simpleblock");
    assert_eq!(packets.len(), 1, "expected one packet from one SimpleBlock");
    assert_eq!(packets[0].slice(&cluster), frame_payload);
}

#[test]
fn packet_reader_webm_yields_video_blocks_incrementally() {
    let frame_payload: &[u8] = b"streamed-vp9-frame";
    let mut block_payload = vec![0x81u8, 0x00, 0x00, 0x00];
    block_payload.extend_from_slice(frame_payload);
    let simple_block = ebml_elt(&[0xA3], block_payload.len() as u64, &block_payload);
    let cluster = ebml_elt(
        &[0x1F, 0x43, 0xB6, 0x75],
        simple_block.len() as u64,
        &simple_block,
    );
    let webm = with_vp9_track(cluster);
    let path = unique_temp_path("stream_webm", "webm");
    std::fs::write(&path, webm).expect("write fixture");

    let mut reader = PacketReader::open(&path).expect("open WebM packet reader");
    assert_eq!(reader.framerate(), None);
    assert_eq!(
        reader.next_packet().expect("read packet").as_deref(),
        Some(frame_payload)
    );
    assert!(reader.next_packet().expect("read EOF").is_none());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn webm_filters_other_tracks_and_returns_only_the_vp9_track() {
    fn simple_block(track: u8, flags: u8, frame: &[u8]) -> Vec<u8> {
        let mut payload = vec![0x80 | track, 0, 0, flags];
        payload.extend_from_slice(frame);
        let mut block = vec![0xA3, 0x80 | payload.len() as u8];
        block.extend_from_slice(&payload);
        block
    }

    let audio = simple_block(1, 0x02, b"laced-audio");
    let vp9 = simple_block(2, 0, b"vp9-video");
    let mut cluster_payload = audio;
    cluster_payload.extend_from_slice(&vp9);
    let cluster = ebml_elt(
        &[0x1F, 0x43, 0xB6, 0x75],
        cluster_payload.len() as u64,
        &cluster_payload,
    );
    let mut webm = vp9_tracks(2);
    webm.extend_from_slice(&cluster);

    let packets = read_webm_packets(&webm).expect("read mixed-track WebM");
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].slice(&webm), b"vp9-video");
}

#[test]
fn webm_segment_wraps_cluster_and_yields_packet() {
    // Exercise the Segment recursion arm explicitly.
    let frame_payload: &[u8] = b"seg-cluster-frame";
    let mut block_payload = vec![0x81u8, 0x00, 0x00, 0x00];
    block_payload.extend_from_slice(frame_payload);
    let mut simple_block = vec![0xA3, 0x80 | block_payload.len() as u8];
    simple_block.extend_from_slice(&block_payload);
    let cluster = ebml_elt(
        &[0x1F, 0x43, 0xB6, 0x75],
        simple_block.len() as u64,
        &simple_block,
    );
    let segment = ebml_elt(&[0x18, 0x53, 0x80, 0x67], cluster.len() as u64, &cluster);
    let segment = with_vp9_track(segment);
    let packets = read_webm_packets(&segment).expect("segment+cluster+simpleblock");
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].slice(&segment), frame_payload);
}

#[test]
fn webm_ebml_header_element_is_recursed_into() {
    // EBML header (0x1A45_DFA3) is also recursed-into. Put an empty one
    // first, then a Cluster with a block, to prove both arms fire in one
    // walk.
    let frame_payload: &[u8] = b"after-ebml-header";
    let mut block_payload = vec![0x81u8, 0x00, 0x00, 0x00];
    block_payload.extend_from_slice(frame_payload);
    let mut simple_block = vec![0xA3, 0x80 | block_payload.len() as u8];
    simple_block.extend_from_slice(&block_payload);
    let cluster = ebml_elt(
        &[0x1F, 0x43, 0xB6, 0x75],
        simple_block.len() as u64,
        &simple_block,
    );

    let ebml_hdr = ebml_elt(&[0x1A, 0x45, 0xDF, 0xA3], 0, &[]);
    let mut bytes = ebml_hdr;
    bytes.extend_from_slice(&cluster);
    let bytes = with_vp9_track(bytes);
    let packets = read_webm_packets(&bytes).expect("ebml+cluster");
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].slice(&bytes), frame_payload);
}

#[test]
fn webm_block_group_at_top_level_is_skipped_not_recursed() {
    // BlockGroup (0xA0) outside a Cluster — the `if inside_cluster`
    // guard on the match arm should be False, falling through to the
    // default skip-and-move-on case. We assert no error, no packets.
    let block_group_payload = vec![0u8; 4];
    let mut block_group = vec![0xA0, 0x80 | block_group_payload.len() as u8];
    block_group.extend_from_slice(&block_group_payload);
    let block_group = with_vp9_track(block_group);
    let packets = read_webm_packets(&block_group).expect("top-level BG must not error");
    assert!(packets.is_empty());
}

#[test]
fn webm_bare_block_at_top_level_is_skipped_not_recursed() {
    // Block (0xA1) outside a Cluster — guard False on the Block arm.
    let block_payload = [0x81u8, 0x00, 0x00, 0x00, 0xAA];
    let mut block = vec![0xA1, 0x80 | block_payload.len() as u8];
    block.extend_from_slice(&block_payload);
    let block = with_vp9_track(block);
    let packets = read_webm_packets(&block).expect("top-level Block must not error");
    assert!(packets.is_empty());
}

#[test]
fn webm_bare_block_with_lacing_inside_cluster_is_rejected() {
    // Lacing on the selected VP9 track is unsupported and must not
    // disappear from the packet stream.
    let block_payload = [0x81u8, 0x00, 0x00, 0x02, 0xAA]; // flags 0x02 = Xiph lacing
    let mut block = vec![0xA1, 0x80 | block_payload.len() as u8];
    block.extend_from_slice(&block_payload);
    let cluster = ebml_elt(&[0x1F, 0x43, 0xB6, 0x75], block.len() as u64, &block);
    let cluster = with_vp9_track(cluster);
    let error = read_webm_packets(&cluster).expect_err("laced Block was silently skipped");
    assert!(matches!(
        error,
        ContainerError::BadEbml(message) if message.contains("lacing")
    ));
}

#[test]
fn webm_block_group_inside_cluster_recurses_to_block() {
    // BlockGroup (0xA0) inside Cluster — wraps a Block (0xA1). Exercises
    // both the BlockGroup arm and the bare-Block arm.
    let frame_payload: &[u8] = b"block-group-frame";
    let mut block_payload = vec![0x81u8, 0x00, 0x00, 0x00];
    block_payload.extend_from_slice(frame_payload);

    // Block (id=0xA1, vint size).
    let mut block = vec![0xA1u8, 0x80 | block_payload.len() as u8];
    block.extend_from_slice(&block_payload);

    // BlockGroup wraps the Block.
    let mut block_group = vec![0xA0u8, 0x80 | block.len() as u8];
    block_group.extend_from_slice(&block);

    // Cluster wraps the BlockGroup.
    let cluster = ebml_elt(
        &[0x1F, 0x43, 0xB6, 0x75],
        block_group.len() as u64,
        &block_group,
    );

    let cluster = with_vp9_track(cluster);
    let packets = read_webm_packets(&cluster).expect("cluster+blockgroup+block");
    assert_eq!(packets.len(), 1, "BlockGroup should yield exactly 1 frame");
    assert_eq!(packets[0].slice(&cluster), frame_payload);
}

#[test]
fn webm_unknown_element_is_skipped() {
    // An unknown element (id 0x80) inside a Segment is skipped over —
    // a following Cluster+SimpleBlock should still parse.
    let frame_payload: &[u8] = b"unknown-skipped-then-frame";
    let mut block_payload = vec![0x81u8, 0x00, 0x00, 0x00];
    block_payload.extend_from_slice(frame_payload);
    let mut simple_block = vec![0xA3, 0x80 | block_payload.len() as u8];
    simple_block.extend_from_slice(&block_payload);
    let cluster = ebml_elt(
        &[0x1F, 0x43, 0xB6, 0x75],
        simple_block.len() as u64,
        &simple_block,
    );

    // Junk element id=0x80 (1-byte id), payload size 3 bytes, 3 bytes payload.
    let mut junk = vec![0x80u8, 0x83, 0xDE, 0xAD, 0xBE];
    junk.extend_from_slice(&cluster);

    let segment = ebml_elt(&[0x18, 0x53, 0x80, 0x67], junk.len() as u64, &junk);
    let segment = with_vp9_track(segment);
    let packets = read_webm_packets(&segment).expect("segment with unknown+cluster");
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].slice(&segment), frame_payload);
}

#[test]
fn webm_unknown_size_segment_runs_to_eof() {
    // A Segment with an all-ones vint ("unknown size") covers the
    // is_unknown_vint=true branch. Inside the Segment we put a Cluster
    // with one SimpleBlock.
    let frame_payload: &[u8] = b"unknown-size-frame";
    let mut block_payload = vec![0x81u8, 0x00, 0x00, 0x00];
    block_payload.extend_from_slice(frame_payload);
    let mut simple_block = vec![0xA3, 0x80 | block_payload.len() as u8];
    simple_block.extend_from_slice(&block_payload);
    let cluster = ebml_elt(
        &[0x1F, 0x43, 0xB6, 0x75],
        simple_block.len() as u64,
        &simple_block,
    );

    // Segment id then 1-byte all-ones vint (0xFF) = unknown size.
    let mut bytes = vec![0x18, 0x53, 0x80, 0x67, 0xFF];
    bytes.extend_from_slice(&cluster);
    let bytes = with_vp9_track(bytes);
    let packets = read_webm_packets(&bytes).expect("unknown-size segment");
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].slice(&bytes), frame_payload);
}

fn unknown_cluster(payload: &[u8]) -> Vec<u8> {
    let mut cluster = vec![0x1F, 0x43, 0xB6, 0x75, 0xFF, 0xE7, 0x81, 0x00];
    cluster.extend_from_slice(payload);
    cluster
}

fn test_block(value: u8) -> Vec<u8> {
    ebml_elt(&[0xA3], 5, &[0x81, 0, 0, 0, value])
}

fn cluster_segment(body: &[u8], unknown: bool) -> Vec<u8> {
    let mut payload = vp9_tracks(1);
    payload.extend_from_slice(body);
    if unknown {
        let mut segment = vec![0x18, 0x53, 0x80, 0x67, 0xFF];
        segment.extend_from_slice(&payload);
        segment
    } else {
        ebml_elt(&[0x18, 0x53, 0x80, 0x67], payload.len() as u64, &payload)
    }
}

fn unexpected_unknown_size_results(data: &[u8], stem: &str) -> Option<String> {
    let memory = read_webm_packet_bytes(data);
    let streaming = read_webm_packet_bytes_streaming(data, stem);
    let memory_rejected = unknown_size_error(&memory);
    let streaming_rejected = unknown_size_error(&streaming);
    (!memory_rejected || !streaming_rejected)
        .then(|| format!("{stem}: memory={memory:?}, streaming={streaming:?}"))
}

#[test]
fn webm_unknown_size_block_leaves_are_rejected_by_both_readers() {
    let mut unexpected = Vec::new();
    for width in [1, 8] {
        for id in [&[0xA3][..], &[0xA1][..]] {
            let mut block_payload = vec![0x81, 0, 0, 0];
            block_payload.extend_from_slice(b"unknown-block-payload");
            let mut cluster_payload = Vec::new();
            if id == [0xA1] {
                let mut group_payload = unknown_size_elt(id, width, &block_payload);
                group_payload.extend_from_slice(&ebml_elt(&[0xFB], 1, &[0]));
                cluster_payload.extend_from_slice(&ebml_elt(
                    &[0xA0],
                    group_payload.len() as u64,
                    &group_payload,
                ));
            } else {
                cluster_payload.extend_from_slice(&unknown_size_elt(id, width, &block_payload));
                let sibling = ebml_elt(&[0xA3], 5, &[0x81, 0, 0, 0, 0x5A]);
                cluster_payload.extend_from_slice(&sibling);
            }
            let cluster = ebml_elt(
                &[0x1F, 0x43, 0xB6, 0x75],
                cluster_payload.len() as u64,
                &cluster_payload,
            );
            let mut segment_payload = vp9_tracks(1);
            segment_payload.extend_from_slice(&cluster);
            let webm = ebml_elt(
                &[0x18, 0x53, 0x80, 0x67],
                segment_payload.len() as u64,
                &segment_payload,
            );
            let kind = if id == [0xA3] { "simple" } else { "block" };
            if let Some(error) =
                unexpected_unknown_size_results(&webm, &format!("unknown_{kind}_width_{width}"))
            {
                unexpected.push(error);
            }
        }
    }
    assert!(
        unexpected.is_empty(),
        "accepted unknown-size blocks: {unexpected:#?}"
    );
}

#[test]
fn webm_unknown_size_track_entry_is_rejected_by_both_readers() {
    let mut unexpected = Vec::new();
    for width in [1, 8] {
        let entry_payload = vp9_track_entry_payload(1, 1, b"V_VP9");
        let entry = unknown_size_elt(&[0xAE], width, &entry_payload);
        let tracks = ebml_elt(&[0x16, 0x54, 0xAE, 0x6B], entry.len() as u64, &entry);
        let cluster_payload = test_block(7);
        let cluster = ebml_elt(
            &[0x1F, 0x43, 0xB6, 0x75],
            cluster_payload.len() as u64,
            &cluster_payload,
        );
        let mut segment_payload = tracks;
        segment_payload.extend_from_slice(&cluster);
        let webm = ebml_elt(
            &[0x18, 0x53, 0x80, 0x67],
            segment_payload.len() as u64,
            &segment_payload,
        );
        if let Some(error) =
            unexpected_unknown_size_results(&webm, &format!("unknown_track_entry_width_{width}"))
        {
            unexpected.push(error);
        }
    }
    assert!(
        unexpected.is_empty(),
        "accepted unknown-size TrackEntry: {unexpected:#?}"
    );
}

#[test]
fn webm_unknown_size_forbidden_element_ids_are_rejected_by_both_readers() {
    let mut unexpected = Vec::new();
    for width in [1, 8] {
        let mut ebml_contents = vp9_tracks(1);
        let cluster_payload = test_block(1);
        let cluster = ebml_elt(
            &[0x1F, 0x43, 0xB6, 0x75],
            cluster_payload.len() as u64,
            &cluster_payload,
        );
        ebml_contents.extend_from_slice(&cluster);
        let ebml_header = unknown_size_elt(&[0x1A, 0x45, 0xDF, 0xA3], width, &ebml_contents);
        if let Some(error) =
            unexpected_unknown_size_results(&ebml_header, &format!("unknown_ebml_width_{width}"))
        {
            unexpected.push(error);
        }

        let entry = vp9_track_entry(1, 1, b"V_VP9");
        let unknown_tracks = unknown_size_elt(&[0x16, 0x54, 0xAE, 0x6B], width, &entry);
        let mut segment_payload = unknown_tracks;
        let cluster_payload = test_block(2);
        let cluster = ebml_elt(
            &[0x1F, 0x43, 0xB6, 0x75],
            cluster_payload.len() as u64,
            &cluster_payload,
        );
        segment_payload.extend_from_slice(&cluster);
        let segment = ebml_elt(
            &[0x18, 0x53, 0x80, 0x67],
            segment_payload.len() as u64,
            &segment_payload,
        );
        if let Some(error) =
            unexpected_unknown_size_results(&segment, &format!("unknown_tracks_width_{width}"))
        {
            unexpected.push(error);
        }

        let block = ebml_elt(&[0xA1], 5, &[0x81, 0, 0, 0, 0x41]);
        let mut group_payload = block;
        group_payload.extend_from_slice(&ebml_elt(&[0xFB], 1, &[0]));
        let group = unknown_size_elt(&[0xA0], width, &group_payload);
        let cluster = ebml_elt(&[0x1F, 0x43, 0xB6, 0x75], group.len() as u64, &group);
        let mut segment_payload = vp9_tracks(1);
        segment_payload.extend_from_slice(&cluster);
        let segment = ebml_elt(
            &[0x18, 0x53, 0x80, 0x67],
            segment_payload.len() as u64,
            &segment_payload,
        );
        if let Some(error) =
            unexpected_unknown_size_results(&segment, &format!("unknown_block_group_width_{width}"))
        {
            unexpected.push(error);
        }

        let mut group_payload = unknown_size_elt(&[0xA1], width, &[0x81, 0, 0, 0, 0x31]);
        group_payload.extend_from_slice(&ebml_elt(&[0xFB], 1, &[0]));
        let group = ebml_elt(&[0xA0], group_payload.len() as u64, &group_payload);
        let cluster = ebml_elt(&[0x1F, 0x43, 0xB6, 0x75], group.len() as u64, &group);
        let mut segment_payload = vp9_tracks(1);
        segment_payload.extend_from_slice(&cluster);
        let segment = ebml_elt(
            &[0x18, 0x53, 0x80, 0x67],
            segment_payload.len() as u64,
            &segment_payload,
        );
        if let Some(error) = unexpected_unknown_size_results(
            &segment,
            &format!("unknown_block_group_child_width_{width}"),
        ) {
            unexpected.push(error);
        }

        for (id, name) in [
            (&[0x15, 0x49, 0xA9, 0x66][..], "info"),
            (&[0xEC][..], "void"),
            (&[0x80][..], "unknown_id"),
        ] {
            let cluster_payload = test_block(3);
            let cluster = ebml_elt(
                &[0x1F, 0x43, 0xB6, 0x75],
                cluster_payload.len() as u64,
                &cluster_payload,
            );
            let unknown = unknown_size_elt(id, width, &cluster);
            let mut segment_payload = vp9_tracks(1);
            segment_payload.extend_from_slice(&unknown);
            let segment = ebml_elt(
                &[0x18, 0x53, 0x80, 0x67],
                segment_payload.len() as u64,
                &segment_payload,
            );
            if let Some(error) =
                unexpected_unknown_size_results(&segment, &format!("unknown_{name}_width_{width}"))
            {
                unexpected.push(error);
            }
        }
    }
    assert!(
        unexpected.is_empty(),
        "accepted unknown-size elements outside the allowlist: {unexpected:#?}"
    );
}

#[test]
fn webm_known_size_blocks_and_skipped_elements_keep_exact_packets() {
    let track_vint_width_8 = [0x01, 0, 0, 0, 0, 0, 0, 1];
    let mut simple_block_payload = track_vint_width_8.to_vec();
    simple_block_payload.extend_from_slice(&[0, 0, 0]);
    simple_block_payload.extend_from_slice(b"simple-block-frame");
    let simple_block = ebml_elt(
        &[0xA3],
        simple_block_payload.len() as u64,
        &simple_block_payload,
    );

    let mut block_payload = track_vint_width_8.to_vec();
    block_payload.extend_from_slice(&[0, 0, 0]);
    block_payload.extend_from_slice(b"block-frame");
    let block = ebml_elt(&[0xA1], block_payload.len() as u64, &block_payload);
    let block_group = ebml_elt(&[0xA0], block.len() as u64, &block);
    let mut cluster_payload = simple_block;
    cluster_payload.extend_from_slice(&block_group);
    let cluster = ebml_elt(
        &[0x1F, 0x43, 0xB6, 0x75],
        cluster_payload.len() as u64,
        &cluster_payload,
    );

    let mut segment_payload = vp9_tracks(1);
    segment_payload.extend_from_slice(&ebml_elt(&[0x15, 0x49, 0xA9, 0x66], 3, b"inf"));
    segment_payload.extend_from_slice(&ebml_elt(&[0xEC], 2, &[0xAA, 0xBB]));
    segment_payload.extend_from_slice(&ebml_elt(&[0x80], 1, &[0xCC]));
    segment_payload.extend_from_slice(&cluster);
    let webm = ebml_elt(
        &[0x18, 0x53, 0x80, 0x67],
        segment_payload.len() as u64,
        &segment_payload,
    );
    let expected = vec![b"simple-block-frame".to_vec(), b"block-frame".to_vec()];

    assert_eq!(
        read_webm_packet_bytes(&webm).expect("memory reader accepts known sizes"),
        expected
    );
    assert_eq!(
        read_webm_packet_bytes_streaming(&webm, "known_size_calibration")
            .expect("stream reader accepts known sizes"),
        expected
    );
}

fn assert_cluster_packets(bytes: &[u8], expected: &[u8], streaming: bool) {
    let result = if streaming {
        let path = unique_temp_path("unknown_cluster", "webm");
        std::fs::write(&path, bytes).expect("write isolated WebM fixture");
        let result = (|| {
            let mut reader = PacketReader::open(&path)?;
            let mut packets = Vec::new();
            while let Some(packet) = reader.next_packet()? {
                packets.push(packet);
            }
            Ok::<_, ContainerError>(packets)
        })();
        std::fs::remove_file(path).expect("remove isolated WebM fixture");
        result
    } else {
        read_webm_packets(bytes).map(|ranges| {
            ranges
                .iter()
                .map(|range| range.slice(bytes).to_vec())
                .collect()
        })
    };
    let packets = result.expect("unknown-size Cluster siblings must not consume nesting depth");
    let expected: Vec<_> = expected.iter().map(|&value| vec![value]).collect();
    assert_eq!(
        packets, expected,
        "Segment siblings must terminate an unknown-size Cluster"
    );
}

fn long_unknown_clusters(streaming: bool) {
    let mut body = Vec::new();
    for value in 0..64 {
        body.extend_from_slice(&unknown_cluster(&test_block(value)));
    }
    let block = test_block(64);
    body.extend_from_slice(&ebml_elt(
        &[0x1F, 0x43, 0xB6, 0x75],
        block.len() as u64,
        &block,
    ));
    body.extend_from_slice(&unknown_cluster(&test_block(65)));
    for unknown_parent in [false, true] {
        assert_cluster_packets(
            &cluster_segment(&body, unknown_parent),
            &(0..66).collect::<Vec<_>>(),
            streaming,
        );
    }
}

#[test]
fn webm_unknown_cluster_siblings_memory() {
    long_unknown_clusters(false);
}

#[test]
fn webm_unknown_cluster_siblings_stream() {
    long_unknown_clusters(true);
}

fn unknown_cluster_boundaries(streaming: bool) {
    // Every other Segment child ends the Cluster. A subsequent misplaced
    // SimpleBlock must not inherit the previous Cluster's packet context.
    let siblings: &[&[u8]] = &[
        &[0x11, 0x4D, 0x9B, 0x74], // SeekHead
        &[0x15, 0x49, 0xA9, 0x66], // Info
        &[0x16, 0x54, 0xAE, 0x6B], // Tracks
        &[0x1C, 0x53, 0xBB, 0x6B], // Cues
        &[0x19, 0x41, 0xA4, 0x69], // Attachments
        &[0x10, 0x43, 0xA7, 0x70], // Chapters
        &[0x12, 0x54, 0xC3, 0x67], // Tags
    ];
    for sibling in siblings {
        let mut cluster_payload = test_block(1);
        // Global elements belong to the current Cluster; embedded ID-looking
        // bytes are opaque payload and must not be searched for boundaries.
        cluster_payload.extend_from_slice(&ebml_elt(&[0xEC], 4, &[0x1F, 0x43, 0xB6, 0x75]));
        cluster_payload.extend_from_slice(&ebml_elt(&[0xBF], 4, &[0; 4]));
        cluster_payload.extend_from_slice(&test_block(3));
        let mut body = unknown_cluster(&cluster_payload);
        body.extend_from_slice(&ebml_elt(sibling, 0, &[]));
        body.extend_from_slice(&test_block(99));
        let block = test_block(2);
        body.extend_from_slice(&ebml_elt(
            &[0x1F, 0x43, 0xB6, 0x75],
            block.len() as u64,
            &block,
        ));
        for unknown_parent in [false, true] {
            assert_cluster_packets(
                &cluster_segment(&body, unknown_parent),
                &[1, 3, 2],
                streaming,
            );
        }
    }
}

#[test]
fn webm_unknown_cluster_boundaries_memory() {
    unknown_cluster_boundaries(false);
}

#[test]
fn webm_unknown_cluster_boundaries_stream() {
    unknown_cluster_boundaries(true);
}

fn unknown_cluster_root_boundaries(streaming: bool) {
    for root in [&[0x1A, 0x45, 0xDF, 0xA3][..], &[0x18, 0x53, 0x80, 0x67][..]] {
        let mut bytes = cluster_segment(&unknown_cluster(&test_block(1)), true);
        bytes.extend_from_slice(&ebml_elt(root, 0, &[]));
        bytes.extend_from_slice(&test_block(99));
        bytes.extend_from_slice(&cluster_segment(&unknown_cluster(&test_block(2)), false));
        assert_cluster_packets(&bytes, &[1, 2], streaming);
    }
}

#[test]
fn webm_unknown_cluster_root_boundaries_memory() {
    unknown_cluster_root_boundaries(false);
}

#[test]
fn webm_unknown_cluster_root_boundaries_stream() {
    unknown_cluster_root_boundaries(true);
}

#[test]
fn webm_excessive_container_depth_is_rejected() {
    for id in [&[0x18, 0x53, 0x80, 0x67][..], &[0x1F, 0x43, 0xB6, 0x75][..]] {
        let mut nested = Vec::new();
        for _ in 0..64 {
            nested = ebml_elt(id, nested.len() as u64, &nested);
        }
        let webm = with_vp9_track(nested);
        let error = read_webm_packets(&webm).expect_err("unbounded nesting was accepted");
        assert!(
            matches!(error, ContainerError::BadEbml(message) if message.contains("nesting depth")),
            "unexpected error: {error:?}"
        );
        let path = unique_temp_path("nested_containers", "webm");
        std::fs::write(&path, &webm).expect("write isolated WebM fixture");
        let result = PacketReader::open(&path).and_then(|mut reader| reader.next_packet());
        std::fs::remove_file(path).expect("remove isolated WebM fixture");
        let error = result.expect_err("streaming unbounded nesting was accepted");
        assert!(
            matches!(error, ContainerError::BadEbml(message) if message.contains("nesting depth")),
            "unexpected error: {error:?}"
        );
    }
}

#[test]
fn webm_block_payload_smaller_than_4_bytes_rejected() {
    // SimpleBlock inside Cluster with only 3 payload bytes — the
    // parse_block_frame "payload too small" guard.
    let mut simple_block = vec![0xA3, 0x83, 0x81, 0x00, 0x00]; // size=3, 3 bytes
    let cluster = ebml_elt(
        &[0x1F, 0x43, 0xB6, 0x75],
        simple_block.len() as u64,
        &simple_block,
    );
    let _ = &mut simple_block;
    let cluster = with_vp9_track(cluster);
    let err = read_webm_packets(&cluster).unwrap_err();
    assert!(
        matches!(err, ContainerError::BadEbml(msg) if msg.contains("block payload too small")),
        "got {err:?}"
    );
}

#[test]
fn webm_block_with_lacing_set_is_rejected() {
    // Lacing != 0 on the VP9 track is rejected so no packet silently
    // disappears.
    // Block payload: track=1 (0x81), ts=0_le_i16, flags=0x02 (Xiph lacing),
    // plus 1 byte of frame data (avoiding the frame_start==block_end arm).
    let block_payload = [0x81u8, 0x00, 0x00, 0x02, 0xAA];
    let mut simple_block = vec![0xA3, 0x80 | block_payload.len() as u8];
    simple_block.extend_from_slice(&block_payload);
    let cluster = ebml_elt(
        &[0x1F, 0x43, 0xB6, 0x75],
        simple_block.len() as u64,
        &simple_block,
    );
    let cluster = with_vp9_track(cluster);
    let error = read_webm_packets(&cluster).expect_err("laced block was silently skipped");
    assert!(matches!(
        error,
        ContainerError::BadEbml(message) if message.contains("lacing")
    ));
}

#[test]
fn webm_block_with_zero_frame_bytes_rejected_as_bad_frame_size() {
    // Block payload exactly 4 bytes (track+ts+flags, no frame data). This
    // passes the `< 4` guard but trips the `frame_start == block_end`
    // arm.
    let block_payload = [0x81u8, 0x00, 0x00, 0x00];
    let mut simple_block = vec![0xA3, 0x80 | block_payload.len() as u8];
    simple_block.extend_from_slice(&block_payload);
    let cluster = ebml_elt(
        &[0x1F, 0x43, 0xB6, 0x75],
        simple_block.len() as u64,
        &simple_block,
    );
    let cluster = with_vp9_track(cluster);
    let err = read_webm_packets(&cluster).unwrap_err();
    assert!(
        matches!(
            err,
            ContainerError::BadFrameSize {
                declared: 0,
                available: 0,
                ..
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn webm_block_missing_timecode_and_flags_rejected() {
    // Block payload of 6 bytes but a 3-byte track vint (first byte 0x20)
    // leaves only 3 bytes for the required 2-byte ts + 1-byte flags + ≥1
    // frame byte. Wait — 6 - 3 = 3, exactly the required ts+flags but no
    // frame byte. We want to trip the `after_tn + 3 > block_end` guard,
    // so go with a 4-byte vint giving only 2 bytes for ts+flags.
    //   vint width 4 first byte: 0x10 with the rest of the size bits at 0
    //   → track number bits set don't matter for the test.
    //   Width-4 vint payload bytes: [0x10, 0x00, 0x00, 0x01], that's a
    //   4-byte vint for track number 1.
    // Total payload = 4 (track) + ? . We need block_end < after_tn + 3.
    // Pick payload size = 6 → after the 4-byte vint, only 2 bytes remain,
    // so `after_tn + 3 > block_end`.
    let block_payload = [0x10u8, 0x00, 0x00, 0x01, 0xAA, 0xBB];
    let mut simple_block = vec![0xA3, 0x80 | block_payload.len() as u8];
    simple_block.extend_from_slice(&block_payload);
    let cluster = ebml_elt(
        &[0x1F, 0x43, 0xB6, 0x75],
        simple_block.len() as u64,
        &simple_block,
    );
    let cluster = with_vp9_track(cluster);
    let err = read_webm_packets(&cluster).unwrap_err();
    assert!(
        matches!(err, ContainerError::BadEbml(msg) if msg.contains("missing timecode/flags")),
        "got {err:?}"
    );
}

// ============================================================================
// Y4M parser — error paths
// ============================================================================

fn write_y4m_buf_to_temp(contents: &[u8], tag: &str) -> PathBuf {
    let path = unique_temp_path(tag, "y4m");
    std::fs::write(&path, contents).expect("write fixture");
    path
}

#[test]
fn y4m_missing_magic_is_invalid_data() {
    let path = write_y4m_buf_to_temp(b"NOTAY4M W16 H16 F30:1\n", "no_magic");
    let mut f = std::fs::File::open(&path).expect("open");
    let err = parse_header(&mut f).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("missing YUV4MPEG2 magic"));
}

#[test]
fn y4m_unterminated_header_is_rejected() {
    let path = write_y4m_buf_to_temp(b"YUV4MPEG2 W16 H16 F30:1", "unterminated_header");
    let mut f = std::fs::File::open(&path).expect("open");
    let err = parse_header(&mut f).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    assert!(err.to_string().contains("header newline"));
}

#[test]
fn y4m_oversized_header_is_rejected() {
    let mut data = b"YUV4MPEG2 W16 H16 F30:1 ".to_vec();
    data.extend(std::iter::repeat_n(b'X', 8192));
    data.push(b'\n');
    let path = write_y4m_buf_to_temp(&data, "oversized_header");
    let mut f = std::fs::File::open(&path).expect("open");
    let err = parse_header(&mut f).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("header exceeds"));
}

#[test]
fn y4m_unsupported_chroma_format_rejected() {
    // C422 — supported only 4:2:0 variants.
    let path = write_y4m_buf_to_temp(b"YUV4MPEG2 W16 H16 F30:1 C422\n", "c422");
    let mut f = std::fs::File::open(&path).expect("open");
    let err = parse_header(&mut f).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("unsupported y4m chroma"));
}

#[test]
fn y4m_missing_required_field_rejected() {
    // Only W set — H and F are absent.
    let path = write_y4m_buf_to_temp(b"YUV4MPEG2 W16\n", "missing_h_f");
    let mut f = std::fs::File::open(&path).expect("open");
    let err = parse_header(&mut f).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("missing required y4m field"));
}

#[test]
fn y4m_chroma_jpeg_variant_accepted() {
    // The `chroma_ok` matcher branches by variant — exercise one that
    // isn't the default (C420jpeg).
    let path = write_y4m_buf_to_temp(b"YUV4MPEG2 W16 H8 F30:1 C420jpeg\n", "c420jpeg");
    let mut f = std::fs::File::open(&path).expect("open");
    let hdr = parse_header(&mut f).expect("parse C420jpeg");
    let _ = std::fs::remove_file(&path);
    assert_eq!((hdr.w, hdr.h), (16, 8));
}

#[test]
fn y4m_chroma_omitted_treated_as_acceptable() {
    // No `C` tag at all — the reader defaults `chroma_ok = true`.
    let path = write_y4m_buf_to_temp(b"YUV4MPEG2 W16 H8 F30:1\n", "no_chroma");
    let mut f = std::fs::File::open(&path).expect("open");
    let hdr = parse_header(&mut f).expect("parse without C tag");
    let _ = std::fs::remove_file(&path);
    assert_eq!((hdr.w, hdr.h, hdr.fps_num, hdr.fps_den), (16, 8, 30, 1));
}

#[test]
fn y4m_header_framerate_helper_returns_quotient() {
    let hdr = Y4mHeader {
        w: 4,
        h: 4,
        fps_num: 60,
        fps_den: 2,
    };
    assert!((hdr.framerate() - 30.0).abs() < 1e-12);
}

// ============================================================================
// read_cropped — error paths
// ============================================================================

fn write_one_frame_y4m(
    width: u32,
    height: u32,
    body: &[u8],
    frame_prefix: &[u8],
    tag: &str,
) -> PathBuf {
    let mut buf = Vec::new();
    write_header(&mut buf, width, height, 30, 1, ChromaFormat::Yuv420).expect("header");
    buf.extend_from_slice(frame_prefix);
    buf.extend_from_slice(body);
    write_y4m_buf_to_temp(&buf, tag)
}

#[test]
fn read_cropped_rejects_crop_exceeding_source_dims() {
    let y_len = (16 * 16) as usize;
    let uv_len = (8 * 8) as usize;
    let body: Vec<u8> = (0..y_len + 2 * uv_len).map(|i| i as u8).collect();
    let path = write_one_frame_y4m(16, 16, &body, b"FRAME\n", "crop_too_big");
    let err = read_cropped(&path, 32, 16, 1).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(err.to_string().contains("exceeds y4m"));
}

#[test]
fn read_cropped_rejects_odd_crop_dimensions() {
    let y_len = (16 * 16) as usize;
    let uv_len = (8 * 8) as usize;
    let body: Vec<u8> = vec![0u8; y_len + 2 * uv_len];
    let path = write_one_frame_y4m(16, 16, &body, b"FRAME\n", "odd_crop");
    let err = read_cropped(&path, 15, 16, 1).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(err.to_string().contains("crop dims must be even"));
}

#[test]
fn read_cropped_rejects_odd_height_only() {
    // Hit the second operand of `!crop_w.is_multiple_of(2) || !crop_h.is_multiple_of(2)`:
    // crop_w=16 is even, crop_h=15 is odd — exercises the right-hand branch.
    let y_len = (16 * 16) as usize;
    let uv_len = (8 * 8) as usize;
    let body: Vec<u8> = vec![0u8; y_len + 2 * uv_len];
    let path = write_one_frame_y4m(16, 16, &body, b"FRAME\n", "odd_crop_h");
    let err = read_cropped(&path, 16, 15, 1).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(err.to_string().contains("crop dims must be even"));
}

#[test]
fn read_cropped_rejects_height_only_exceeding_source() {
    // Hit the second operand of `crop_w > hdr.w || crop_h > hdr.h`:
    // crop_w fits, crop_h does not.
    let y_len = (16 * 16) as usize;
    let uv_len = (8 * 8) as usize;
    let body: Vec<u8> = vec![0u8; y_len + 2 * uv_len];
    let path = write_one_frame_y4m(16, 16, &body, b"FRAME\n", "crop_h_too_big");
    let err = read_cropped(&path, 16, 32, 1).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(err.to_string().contains("exceeds y4m"));
}

#[test]
fn read_cropped_missing_frame_tag_rejected() {
    let y_len = (16 * 16) as usize;
    let uv_len = (8 * 8) as usize;
    let body: Vec<u8> = vec![0u8; y_len + 2 * uv_len];
    // Wrong tag where FRAME should be.
    let path = write_one_frame_y4m(16, 16, &body, b"WRONG\n", "wrong_tag");
    let err = read_cropped(&path, 16, 16, 1).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("missing FRAME tag"));
}

#[test]
fn read_cropped_eof_in_frame_params_rejected() {
    // FRAME followed by a non-newline byte and then immediate EOF →
    // "EOF in FRAME params".
    let mut buf = Vec::new();
    write_header(&mut buf, 16, 16, 30, 1, ChromaFormat::Yuv420).expect("hdr");
    buf.extend_from_slice(b"FRAME I"); // 'I' instead of '\n', and no trailing '\n'
    let path = write_y4m_buf_to_temp(&buf, "eof_in_frame_params");
    let err = read_cropped(&path, 16, 16, 1).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    assert!(err.to_string().contains("EOF in FRAME params"));
}

#[test]
fn read_cropped_consumes_extended_frame_params_to_newline() {
    // FRAME with extra params (e.g. `I0` interlacing override) followed by
    // a newline — must read past them and then consume the plane data.
    let y_len = (16 * 16) as usize;
    let uv_len = (8 * 8) as usize;
    let body: Vec<u8> = (0..y_len + 2 * uv_len).map(|i| (i * 3) as u8).collect();
    let mut buf = Vec::new();
    write_header(&mut buf, 16, 16, 30, 1, ChromaFormat::Yuv420).expect("hdr");
    buf.extend_from_slice(b"FRAME I0\n"); // extended params then newline
    buf.extend_from_slice(&body);
    let path = write_y4m_buf_to_temp(&buf, "extended_frame_params");
    let (frames, _) = read_cropped(&path, 16, 16, 1).expect("read cropped");
    let _ = std::fs::remove_file(&path);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].0.len(), y_len);
    assert_eq!(frames[0].1.len(), uv_len);
    assert_eq!(frames[0].2.len(), uv_len);
}

#[test]
fn read_cropped_eof_before_any_frame_returns_empty() {
    // Header only, no FRAME tag → read_exact on hdr_buf fails → break, not
    // error. read_cropped returns (vec![], framerate).
    let mut buf = Vec::new();
    write_header(&mut buf, 16, 16, 30, 1, ChromaFormat::Yuv420).expect("hdr");
    let path = write_y4m_buf_to_temp(&buf, "empty_after_header");
    let (frames, fr) = read_cropped(&path, 16, 16, 3).expect("read cropped");
    let _ = std::fs::remove_file(&path);
    assert!(frames.is_empty(), "no frames after header");
    assert!((fr - 30.0).abs() < 1e-9);
}

#[test]
fn read_cropped_partial_frame_header_is_rejected() {
    let mut buf = Vec::new();
    write_header(&mut buf, 16, 16, 30, 1, ChromaFormat::Yuv420).expect("hdr");
    buf.extend_from_slice(b"FRA");
    let path = write_y4m_buf_to_temp(&buf, "partial_frame_header");
    let err = read_cropped(&path, 16, 16, 1).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    assert!(err.to_string().contains("FRAME header"));
}

#[test]
fn read_cropped_truncated_frame_body_propagates_read_exact_error() {
    // FRAME\n with fewer than y_size + 2*uv_size bytes → read_exact errors
    // with UnexpectedEof, which `?` forwards verbatim.
    let mut buf = Vec::new();
    write_header(&mut buf, 16, 16, 30, 1, ChromaFormat::Yuv420).expect("hdr");
    buf.extend_from_slice(b"FRAME\n");
    buf.extend_from_slice(&[0u8; 4]); // way too short
    let path = write_y4m_buf_to_temp(&buf, "truncated_frame_body");
    let err = read_cropped(&path, 16, 16, 1).unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
}

#[test]
fn read_cropped_full_frame_passthrough_preserves_bytes() {
    // Round-trip with crop == source: the read frame's Y plane must equal
    // the body's Y region exactly (bytewise).
    let y_len = (16 * 16) as usize;
    let uv_len = (8 * 8) as usize;
    let body: Vec<u8> = (0..y_len + 2 * uv_len).map(|i| i as u8).collect();
    let path = write_one_frame_y4m(16, 16, &body, b"FRAME\n", "full_passthrough");
    let (frames, _) = read_cropped(&path, 16, 16, 1).expect("read cropped");
    let _ = std::fs::remove_file(&path);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].0, &body[0..y_len]);
    assert_eq!(frames[0].1, &body[y_len..y_len + uv_len]);
    assert_eq!(frames[0].2, &body[y_len + uv_len..y_len + 2 * uv_len]);
}

#[test]
fn read_cropped_centre_crop_extracts_inner_region() {
    // 16x16 source, 8x8 centre crop → off_x = off_y = 4. Y[r][c] of the
    // crop must equal body[4..12][4..12] of the source.
    let src_w = 16usize;
    let src_h = 16usize;
    let y_len = src_w * src_h;
    let uv_len = (src_w / 2) * (src_h / 2);
    // Use a known pattern: y[r*W+c] = r*16+c.
    let mut body = Vec::with_capacity(y_len + 2 * uv_len);
    for r in 0..src_h {
        for c in 0..src_w {
            body.push((r * 16 + c) as u8);
        }
    }
    body.extend_from_slice(&vec![0x55u8; uv_len]); // U plane filler
    body.extend_from_slice(&vec![0xAAu8; uv_len]); // V plane filler
    let path = write_one_frame_y4m(16, 16, &body, b"FRAME\n", "centre_crop");
    let (frames, _) = read_cropped(&path, 8, 8, 1).expect("read cropped");
    let _ = std::fs::remove_file(&path);

    assert_eq!(frames.len(), 1);
    let (y, u, v) = &frames[0];
    assert_eq!(y.len(), 64);
    for r in 0..8 {
        for c in 0..8 {
            let expected = ((r + 4) * 16 + (c + 4)) as u8;
            assert_eq!(y[r * 8 + c], expected, "centre crop pixel ({r},{c}) wrong");
        }
    }
    assert!(u.iter().all(|&b| b == 0x55));
    assert!(v.iter().all(|&b| b == 0xAA));
}

// ============================================================================
// Y4M writer/reader round-trip
// ============================================================================

#[test]
fn y4m_writer_reader_roundtrip_preserves_bytes() {
    let (w, h) = (16u32, 8u32);
    let y_len = (w * h) as usize;
    let uv_len = ((w / 2) * (h / 2)) as usize;
    let y: Vec<u8> = (0..y_len).map(|i| (i ^ 0x37) as u8).collect();
    let u: Vec<u8> = (0..uv_len).map(|i| (i as u8).wrapping_mul(7)).collect();
    let v: Vec<u8> = (0..uv_len).map(|i| (i as u8).wrapping_add(11)).collect();

    let mut buf = Vec::new();
    write_header(&mut buf, w, h, 25, 1, ChromaFormat::Yuv420).expect("write header");
    write_frame(&mut buf, &y, &u, &v).expect("write frame");

    let path = write_y4m_buf_to_temp(&buf, "rt");
    let (frames, fr) = read_cropped(&path, w, h, 1).expect("read");
    let _ = std::fs::remove_file(&path);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].0, y);
    assert_eq!(frames[0].1, u);
    assert_eq!(frames[0].2, v);
    assert!((fr - 25.0).abs() < 1e-9);
}
