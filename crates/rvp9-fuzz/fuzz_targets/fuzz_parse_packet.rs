//! `fuzz_parse_packet` — structured-Arbitrary fuzz target for the
//! `rvp9-decoder` packet-parser entrypoint.
//!
//! HLD §G9 (load-bearing design decision): naive byte-mutation
//! libFuzzer on a VP9 stream wastes ~99% of its compute budget
//! producing garbage rejected in the first 50 bytes of
//! `parse_packet`. The structured `arbitrary::Arbitrary` input
//! shape below pre-shapes inputs into "plausibly-VP9" packets so
//! the fuzzer's bit budget gets spent on parser-reachable mutations.
//!
//! Success metric: no panic. `parse_packet` is allowed (and
//! expected) to return `Err(BitstreamError::...)` for most inputs;
//! the only failure this target catches is `panic!`, `abort`, ASan
//! / UBSan triggers, or other process death.
//!
//! ## Scope: parser-only
//!
//! Per HLD §G9, `parse_packet` flows into `parse_uncompressed_header`
//! which terminates at `first_partition_size` — tile / partition
//! bytes are NOT consumed by this target. The packet shape below
//! therefore models only the uncompressed-header bit layout plus an
//! opaque "tile tail" byte buffer (mutated freely by the fuzzer but
//! not interpreted here). The structured partition-tree strategy
//! belongs to `fuzz_decode_frame` (G9.1 deliverable), which lands
//! with Stage 4a once we can synthesise frames that actually round-
//! trip through the reconstruction path.
//
// TODO(G9-PERF): hoist Decoder::new() out of the per-iter loop
// (thread-local or round-robin) to lift libFuzzer exec/s. Tracked
// as a follow-up; deferred from the G9 squash to keep this change
// pure-correctness.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use rvp9_core::bitstream::RawBitWriter;
use rvp9_decoder::Decoder;

/// Top-level shape mirrored from HLD §G9. `arbitrary` will produce
/// any `u8` for `profile`; the byte encoder masks it to two bits to
/// stay within the 0..=3 range the spec defines.
#[derive(Arbitrary, Debug)]
struct PlausibleVp9Packet {
    profile: u8,
    show_frame: bool,
    show_existing_frame: bool,
    frame_kind: PlausibleFrameKind,
    width: u16,
    height: u16,
    tiles: Vec<PlausibleTile>,
}

#[derive(Arbitrary, Debug)]
enum PlausibleFrameKind {
    KeyFrame,
    IntraOnly,
    Inter,
}

/// Opaque per-tile tail bytes. Kept as a thin wrapper (rather than a
/// bare `Vec<u8>`) so that G9.1 — the `fuzz_decode_frame` target —
/// can extend this with a structured partition-tree strategy without
/// breaking the corpus shape too aggressively.
#[derive(Arbitrary, Debug)]
struct PlausibleTile {
    raw_tail: Vec<u8>,
}

impl PlausibleVp9Packet {
    /// Encode the structured packet into a byte stream that
    /// *resembles* a VP9 uncompressed header followed by opaque tail
    /// bytes. The output is intentionally approximate — many
    /// generated packets will fail mid-parse, and that's fine: the
    /// success metric is "no panic". The shape exists only to keep
    /// the fuzzer's bit budget out of the first-50-bytes-rejection
    /// trap.
    ///
    /// VP9 uncompressed header bit layout (read MSB-first):
    ///
    /// ```text
    /// frame_marker    : 2 bits (must be 0b10)
    /// profile_low     : 1 bit
    /// profile_high    : 1 bit
    /// show_existing   : 1 bit   ; if set: 3-bit slot + early return
    /// frame_type      : 1 bit   ; 0 = key, 1 = inter
    /// show_frame      : 1 bit
    /// error_resilient : 1 bit
    /// (keyframe)        24-bit sync 0x498342 + colour + size...
    /// ```
    fn encode(&self) -> Vec<u8> {
        // Pre-size: 1 header byte + 3 sync + 4 size bytes (worst
        // case) + tile-tail bytes. Over-allocate to avoid reallocs
        // while the RawBitWriter packs the header.
        let tail_bytes: usize = self.tiles.iter().map(|t| t.raw_tail.len()).sum();
        let mut bytes = vec![0u8; 16 + tail_bytes.min(4096)];
        let header_bytes_written;
        let frame_type_bit: u8 = match self.frame_kind {
            PlausibleFrameKind::KeyFrame => 0,
            _ => 1,
        };
        let show_existing_bit: u8 = u8::from(self.show_existing_frame);
        // The intra-only bit is read only when show_frame == 0
        // (libvpx vp9_decodeframe.c:2654-2658). Forcing show_frame = 0
        // for the IntraOnly variant is what makes the IntraOnly arm
        // reachable on the wire — otherwise IntraOnly + show_frame=1
        // is indistinguishable from Inter.
        let show_frame_bit: u8 = match self.frame_kind {
            PlausibleFrameKind::IntraOnly => 0,
            _ => u8::from(self.show_frame),
        };
        let profile_bits = self.profile & 0b11;

        {
            let mut w = RawBitWriter::new(&mut bytes);
            // frame_marker (2 bits, MSB-first, must be 0b10).
            w.write_literal(0b10, 2);
            // profile_low (bit 0 of profile), then profile_high
            // (bit 1) — matches libvpx `vp9_read_profile`
            // (`vp9_decodeframe.c:2945-2950`).
            w.write_bit(profile_bits & 1);
            w.write_bit((profile_bits >> 1) & 1);
            // show_existing_frame.
            w.write_bit(show_existing_bit);
            if self.show_existing_frame {
                // 3-bit reference slot; the parser returns immediately
                // after this on the show_existing_frame path.
                w.write_literal(0, 3);
            } else {
                // frame_type, show_frame, error_resilient.
                w.write_bit(frame_type_bit);
                w.write_bit(show_frame_bit);
                w.write_bit(0); // error_resilient
                if matches!(self.frame_kind, PlausibleFrameKind::KeyFrame) {
                    // 24-bit sync code (profile 0/1).
                    w.write_literal(u32::from(0x49u8), 8);
                    w.write_literal(u32::from(0x83u8), 8);
                    w.write_literal(u32::from(0x42u8), 8);
                    // Width / height are 16-bit MSB-first literals
                    // packed onto the bit cursor — not byte-aligned
                    // `to_le_bytes` writes. This matches the bit
                    // shape `read_frame_size` consumes (header.rs).
                    //
                    // NOTE: between the sync code and `read_frame_size`
                    // the parser actually reads
                    // `read_bitdepth_colorspace_sampling` (variable
                    // bits depending on profile / sRGB). This encoder
                    // does not model colour-config; most generated
                    // packets will be rejected at colour-config and
                    // the fuzzer mutates over the gap. Bit-packing
                    // the width/height anyway is strictly better than
                    // the previous `to_le_bytes` write because the
                    // mutator's neighbourhood is now bit-aligned.
                    w.write_literal(u32::from(self.width), 16);
                    w.write_literal(u32::from(self.height), 16);
                }
            }
            header_bytes_written = w.bytes_written();
        }
        // Truncate to the header we actually wrote, then append the
        // opaque per-tile tail bytes (which the parser does not
        // consume; they sit beyond `first_partition_size`).
        bytes.truncate(header_bytes_written);
        for tile in &self.tiles {
            bytes.extend_from_slice(&tile.raw_tail);
        }
        bytes
    }
}

fuzz_target!(|packet: PlausibleVp9Packet| {
    let bytes = packet.encode();
    // Pin to 1 worker thread — the fuzzer hits this constructor on
    // every iteration and `Decoder::new()` spawns
    // `available_parallelism()` threads (commonly 8-32). The fuzz
    // target only exercises `parse_packet`, which never submits any
    // tile-decode work, so the threads sit idle. See G5.1 TODO at
    // `Decoder::new`.
    let mut decoder = Decoder::with_threads(std::num::NonZeroUsize::new(1).expect("1 is non-zero"));
    // Result is irrelevant; we are only catching panics / UB.
    let _ = decoder.parse_packet(&bytes);
});
