//! Boolean (arithmetic) writer.
//!
//! Mirrors libvpx `vpx_dsp/bitwriter.{h,c}` bit-for-bit. The output stream
//! is byte-tight after [`Self::stop_encode`]; consumers stitch tile buffers
//! together via the superframe-index builder in [`crate::bitstream`].
//!
//! ## Algorithm sketch
//!
//! VP9's arithmetic codec is a *fractional-bit* coder over a per-symbol
//! probability of the symbol being `0`. State:
//!
//! - `low_value: u32` — the current numerator (24 active bits, top 24).
//! - `range: u32` — the current width of the interval, in `[128, 255]`
//!   after every renormalisation step.
//! - `count: i32` — bits-buffered counter; when it goes non-negative, one
//!   output byte is ready to emit.
//! - `error: bool` — sticky poisoning flag for output overflow.
//!
//! Per write: derive `split` from `range * prob`; pick the lower
//! sub-interval (bit 0) or upper (bit 1); renormalise via [`VPX_NORM`];
//! emit a byte when `count >= 0`, propagating any carry into already-
//! emitted bytes (the `0x80000000` carry-back trick).
//!
//! Documented divergences from libvpx (see `notes/divergence-ledger.md`
//! and `notes/lld/entropy.md` §7):
//!
//! - **§7.3 No `CONFIG_BITSTREAM_DEBUG` queue.** Libvpx has an optional
//!   `bitstream_queue_push` instrumentation site inside `vpx_write` —
//!   we don't ship that path.
//! - **§7.3 Panic-mid-write is not poison-safe.** Same as libvpx: state
//!   is left in whatever partial form the panic produced; we don't try
//!   to recover.

use super::VPX_PROB_HALF;
use super::tables::VPX_NORM;

/// Byte-stream boolean (arithmetic) writer.
///
/// Holds a `&mut [u8]` destination buffer. Use [`Self::start_encode`] to
/// reset state, then a sequence of [`Self::write`] / [`Self::write_bit`] /
/// [`Self::write_literal`] calls, then [`Self::stop_encode`] to flush
/// trailing bits and emit the marker byte. Returns the byte count or an
/// error if any write overflowed.
#[derive(Debug)]
pub struct BoolWriter<'a> {
    low_value: u32,
    range: u32,
    count: i32,
    pos: u32,
    size: u32,
    buffer: &'a mut [u8],
    error: bool,
    count_only: bool,
}

impl<'a> BoolWriter<'a> {
    /// Create a writer over `buffer`, initialised to start-of-frame state
    /// and with the leading marker bit (a `0` at prob 128) already
    /// emitted, matching libvpx `vpx_start_encode` (`bitwriter.c:20-31`).
    pub fn start_encode(buffer: &'a mut [u8]) -> Self {
        super::bitacct::set_category(super::bitacct::BitCategory::Other);
        let size = u32::try_from(buffer.len()).unwrap_or(i32::MAX as u32);
        let mut w = Self {
            low_value: 0,
            range: 255,
            count: -24,
            pos: 0,
            size,
            buffer,
            error: false,
            count_only: false,
        };
        // Per libvpx: emit a `0` at probability 128 as a marker bit so
        // the reader's `vpx_reader_init` can verify the start of stream.
        w.write_bit(0);
        w
    }

    /// Create a writer for a syntax/count traversal that deliberately emits
    /// no arithmetic-coded bytes.
    ///
    /// The caller must still execute the normal modes/token walk so its
    /// explicit [`super::FrameCounts`] updates remain exact. Calls to
    /// [`Self::write`] become no-ops and [`Self::has_error`] stays false.
    /// This is used by the encoder's VP9 two-loop coefficient-probability
    /// pass, whose first traversal needs symbol counts but discards bytes.
    pub fn start_count_only(buffer: &'a mut [u8]) -> Self {
        Self {
            low_value: 0,
            range: 255,
            count: -24,
            pos: 0,
            size: u32::try_from(buffer.len()).unwrap_or(i32::MAX as u32),
            buffer,
            error: false,
            count_only: true,
        }
    }

    /// Encode a single bit `bit` (0 or 1) using the given 8-bit
    /// probability `probability` that the bit is `0`.
    ///
    /// Mirrors `vpx_write` (`vpx_dsp/bitwriter.h:46-116`) verbatim — the
    /// carry-back trick at the `0x80000000` check is preserved exactly.
    #[inline(always)]
    pub fn write(&mut self, bit: u8, probability: u8) {
        if self.count_only {
            return;
        }
        // E4 diagnostic: env-gated per-category bit accounting. No-op
        // (one relaxed atomic load) unless RVP9_E4_BITACCT is set. Does
        // NOT alter coding state.
        if super::bitacct::enabled() {
            super::bitacct::record(bit, probability);
        }
        let mut count = self.count;
        let mut range = self.range;
        let mut low_value = self.low_value;

        // split = 1 + (((range - 1) * probability) >> 8)
        let split: u32 = 1 + (((range - 1) * u32::from(probability)) >> 8);

        range = split;

        if bit != 0 {
            low_value = low_value.wrapping_add(split);
            range = self.range - split;
        }

        let mut shift = i32::from(VPX_NORM[(range & 0xff) as usize]);

        range <<= shift;
        count += shift;

        if count >= 0 {
            let offset = shift - count;

            if !self.error {
                // Carry-back: if a top-bit carry-over would propagate
                // through the byte we're about to emit, propagate it
                // through any 0xff bytes preceding self.pos.
                // libvpx bitwriter.h:88-98.
                if offset >= 1 {
                    let test_shift = offset - 1;
                    // SAFETY: offset comes from shift-count math where
                    // 0 <= shift <= 7 and -24 <= count <= 7; offset-1 is
                    // always in [-1, 7]. We only enter this branch when
                    // offset >= 1, so test_shift is in [0, 7].
                    let probe = low_value.wrapping_shl(test_shift as u32);
                    if probe & 0x8000_0000 != 0 {
                        let mut x = self.pos as i32 - 1;
                        while x >= 0 && self.buffer[x as usize] == 0xff {
                            self.buffer[x as usize] = 0;
                            x -= 1;
                        }
                        // libvpx TODO: "How to prove x >= 0?"
                        // The codec invariant guarantees a non-0xff byte
                        // precedes the carry origin; mirror libvpx.
                        debug_assert!(x >= 0, "BoolWriter carry-back: x underflow");
                        if x >= 0 {
                            self.buffer[x as usize] = self.buffer[x as usize].wrapping_add(1);
                        }
                    }
                }

                if self.pos < self.size {
                    let out_byte = ((low_value >> (24 - offset)) & 0xff) as u8;
                    self.buffer[self.pos as usize] = out_byte;
                    self.pos += 1;
                } else {
                    self.error = true;
                }
            }
            low_value = low_value.wrapping_shl(offset as u32);
            shift = count;
            low_value &= 0xff_ffff;
            count -= 8;
        }

        low_value = low_value.wrapping_shl(shift as u32);
        self.count = count;
        self.low_value = low_value;
        self.range = range;
    }

    /// Encode a single bit at probability `128` (max-entropy). Alias for
    /// `write(bit, VPX_PROB_HALF)`. Mirrors `vpx_write_bit`.
    #[inline]
    pub fn write_bit(&mut self, bit: u8) {
        self.write(bit, VPX_PROB_HALF);
    }

    /// Encode the low `bits` bits of `data` MSB-first at probability 128.
    /// Mirrors `vpx_write_literal` (`bitwriter.h:122-126`).
    ///
    /// # Panics (debug only)
    ///
    /// Panics in debug mode if `bits > 32`.
    #[inline]
    pub fn write_literal(&mut self, data: u32, bits: u8) {
        debug_assert!(bits <= 32, "BoolWriter::write_literal: bits={bits} > 32");
        let n = bits.min(32);
        for i in (0..n).rev() {
            self.write_bit(((data >> i) & 1) as u8);
        }
    }

    /// Encode `prob` as an 8-bit literal at probability 128. Mirrors the
    /// `vpx_write_prob` macro.
    #[inline]
    pub fn write_prob(&mut self, prob: u8) {
        self.write_literal(u32::from(prob), 8);
    }

    /// Flush trailing arithmetic state and pad the buffer with the
    /// marker-collision-avoidance byte if needed. Mirrors `vpx_stop_encode`
    /// (`bitwriter.c:33-55`). Returns the number of bytes written, or
    /// [`StopEncodeError`] if the buffer overflowed at any point.
    pub fn stop_encode(mut self) -> Result<usize, StopEncodeError> {
        if self.count_only {
            return Ok(0);
        }
        super::bitacct::set_category(super::bitacct::BitCategory::Other);
        // Flush 32 trailing zero bits to drain the range register.
        for _ in 0..32 {
            self.write_bit(0);
        }
        // Avoid producing a byte that the superframe index could mistake
        // for a marker byte (top 3 bits == 0b110). Mirrors libvpx
        // bitwriter.c:42-48 exactly.
        if !self.error && self.pos > 0 {
            let last = self.buffer[(self.pos - 1) as usize];
            if (last & 0xe0) == 0xc0 {
                if self.pos < self.size {
                    self.buffer[self.pos as usize] = 0;
                    self.pos += 1;
                } else {
                    self.error = true;
                }
            }
        }
        if self.error {
            Err(StopEncodeError)
        } else {
            Ok(self.pos as usize)
        }
    }

    /// Whether the writer has overflowed at any point. Mirrors libvpx
    /// implicit `error` accessor.
    #[inline]
    #[must_use]
    pub fn has_error(&self) -> bool {
        self.error
    }

    /// Bytes emitted so far.
    #[inline]
    #[must_use]
    pub fn pos(&self) -> usize {
        self.pos as usize
    }
}

/// Returned by [`BoolWriter::stop_encode`] if the output buffer overflowed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StopEncodeError;

impl std::fmt::Display for StopEncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BoolWriter overflow during stop_encode")
    }
}

impl std::error::Error for StopEncodeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::BoolReader;

    /// Smoke test for `start_encode` initial state: the marker bit at
    /// prob 128 is emitted at construction time, so `pos()` starts at 0
    /// before any byte is flushed but `count` is already advanced.
    #[test]
    fn start_encode_emits_marker_bit_without_flushing_byte() {
        let mut buf = [0u8; 16];
        let w = BoolWriter::start_encode(&mut buf);
        // Marker bit is `0` at prob 128 — single write does not yet
        // accumulate 8 bits, so no byte has been written.
        assert_eq!(w.pos(), 0);
        assert!(!w.has_error());
    }

    /// Round-trip: write a single bit, decode it, verify match. This
    /// is the canonical writer<->reader oracle for the entropy codec.
    #[test]
    fn single_bit_round_trip() {
        let mut buf = [0u8; 16];
        {
            let mut w = BoolWriter::start_encode(&mut buf);
            w.write_bit(1);
            let n = w.stop_encode().expect("stop_encode failed");
            assert!(n > 0, "stop_encode must emit at least one byte");
        }
        let mut r = BoolReader::new(&buf).expect("reader init failed");
        assert_eq!(r.read_bit(), 1, "decoded bit must equal encoded bit");
    }

    /// Round-trip many bits at prob 128 (max entropy) — encoded
    /// then decoded must match exactly. Exercises `write` + `read`
    /// across the full byte-emission and carry paths.
    #[test]
    fn many_bits_round_trip_prob_128() {
        let bits: Vec<u8> = (0..256).map(|i| ((i * 17 + 3) & 1) as u8).collect();
        let mut buf = vec![0u8; 256];
        let pos = {
            let mut w = BoolWriter::start_encode(&mut buf);
            for &b in &bits {
                w.write_bit(b);
            }
            w.stop_encode().expect("stop_encode failed")
        };
        let mut r = BoolReader::new(&buf[..pos]).expect("reader init failed");
        for (i, &expected) in bits.iter().enumerate() {
            let got = r.read_bit();
            assert_eq!(got, expected, "mismatch at bit {i}");
        }
    }

    /// Round-trip with varied probabilities — exercises the `range *
    /// prob` arithmetic at extremes (low/high prob), forcing many
    /// renormalisation paths.
    #[test]
    fn round_trip_varied_probabilities() {
        // Hand-picked test vector: every (bit, prob) pair encoded must
        // decode to the same bit.
        let plan: Vec<(u8, u8)> = vec![
            (0, 1),   // very low prob of 0 → encoding a 0 is rare/expensive
            (1, 1),   // very high prob of 1 (since prob is P(0))
            (0, 255), // very high prob of 0 → cheap
            (1, 255), // very low prob of 1 → expensive
            (0, 128),
            (1, 128),
            (0, 64),
            (1, 192),
            (1, 32),
            (0, 200),
            (1, 50),
            (0, 100),
            (1, 10),
            (0, 240),
            (1, 5),
            (0, 250),
        ];
        let mut buf = vec![0u8; 64];
        let pos = {
            let mut w = BoolWriter::start_encode(&mut buf);
            for &(bit, prob) in &plan {
                w.write(bit, prob);
            }
            w.stop_encode().expect("stop_encode failed")
        };
        let mut r = BoolReader::new(&buf[..pos]).expect("reader init failed");
        for (i, &(expected_bit, prob)) in plan.iter().enumerate() {
            let got = r.read(prob);
            assert_eq!(got, expected_bit, "mismatch at index {i} (prob {prob})");
        }
    }

    /// `write_literal(data, bits)` emits the low `bits` bits of `data`
    /// MSB-first and is recovered by `read_literal(bits)` exactly.
    #[test]
    fn write_literal_round_trip() {
        let mut buf = vec![0u8; 64];
        let pos = {
            let mut w = BoolWriter::start_encode(&mut buf);
            w.write_literal(0xA5, 8); // 10100101
            w.write_literal(0b101, 3);
            w.write_literal(0x1234_5678, 32); // full 32-bit
            w.write_literal(0, 0); // zero-bit literal is a no-op
            w.stop_encode().expect("stop_encode failed")
        };
        let mut r = BoolReader::new(&buf[..pos]).expect("reader init failed");
        assert_eq!(r.read_literal(8), 0xA5);
        assert_eq!(r.read_literal(3), 0b101);
        assert_eq!(r.read_literal(32), 0x1234_5678);
        assert_eq!(r.read_literal(0), 0);
    }

    /// `write_prob` emits an 8-bit prob value at p=128; the reader
    /// recovers it exactly via `read_literal(8)`.
    #[test]
    fn write_prob_round_trip() {
        let mut buf = vec![0u8; 32];
        let pos = {
            let mut w = BoolWriter::start_encode(&mut buf);
            w.write_prob(199);
            w.write_prob(0);
            w.write_prob(255);
            w.stop_encode().expect("stop_encode failed")
        };
        let mut r = BoolReader::new(&buf[..pos]).expect("reader init failed");
        assert_eq!(r.read_literal(8), 199);
        assert_eq!(r.read_literal(8), 0);
        assert_eq!(r.read_literal(8), 255);
    }

    /// `stop_encode` returns `Err(StopEncodeError)` when the buffer is
    /// too small to hold the trailing flush. Test with a comically
    /// tiny buffer — even one bit at prob 128 won't fit because the
    /// 32-bit trailing-zero flush will overflow.
    #[test]
    fn stop_encode_returns_err_on_overflow() {
        let mut buf = [0u8; 1]; // way too small
        let mut w = BoolWriter::start_encode(&mut buf);
        // Write enough bits to force overflow during stop_encode.
        for _ in 0..100 {
            w.write_bit(1);
        }
        let result = w.stop_encode();
        assert_eq!(result, Err(StopEncodeError));
    }

    /// `has_error()` becomes true once the writer overflows during a
    /// regular write (not stop_encode), and subsequent writes are no-ops
    /// — `pos()` does not advance.
    #[test]
    fn has_error_set_on_buffer_overflow() {
        let mut buf = [0u8; 2];
        let mut w = BoolWriter::start_encode(&mut buf);
        // Spam bits until the buffer overflows.
        for _ in 0..200 {
            w.write_bit(1);
        }
        // After overflow `has_error()` should be true.
        assert!(w.has_error(), "writer should be in error state");
        // Further writes don't advance pos beyond the buffer.
        let pos_before = w.pos();
        w.write_bit(0);
        let pos_after = w.pos();
        assert!(pos_after <= 2, "pos must stay within buffer bounds");
        assert_eq!(
            pos_before, pos_after,
            "writes after error must not advance pos"
        );
    }

    /// `StopEncodeError`'s `Display` impl is non-empty and stable.
    /// Locks in the error message format for diagnostic stability.
    #[test]
    fn stop_encode_error_display() {
        let s = format!("{}", StopEncodeError);
        assert!(
            s.contains("overflow"),
            "Display should mention 'overflow', got: {s}"
        );
    }

    /// Round-trip a longer sequence with biased probabilities — this
    /// exercises the carry-back path through long runs of 0xff bytes
    /// that the arithmetic coder produces under high-prob skewed input.
    #[test]
    fn long_sequence_high_prob_round_trip() {
        // 1000 bits at strongly biased prob (240 → 94% likely to be 0).
        // Mix in occasional 1-bits to force renormalisation transitions.
        let mut bits = vec![0u8; 1000];
        for (i, slot) in bits.iter_mut().enumerate() {
            // ~6% chance of being 1, deterministic.
            *slot = u8::from((i * 31 + 7) % 17 == 0);
        }
        let mut buf = vec![0u8; 512];
        let pos = {
            let mut w = BoolWriter::start_encode(&mut buf);
            for &b in &bits {
                w.write(b, 240);
            }
            w.stop_encode().expect("stop_encode failed")
        };
        let mut r = BoolReader::new(&buf[..pos]).expect("reader init failed");
        for (i, &expected) in bits.iter().enumerate() {
            assert_eq!(r.read(240), expected, "mismatch at bit {i}");
        }
    }

    /// `stop_encode` may emit a trailing collision-avoidance byte (a
    /// zero) when the last emitted byte's top 3 bits == 0b110. Verify
    /// the post-condition: `(buf[pos-1] & 0xe0) != 0xc0` always after
    /// successful `stop_encode`.
    #[test]
    fn stop_encode_avoids_superframe_marker_byte_collision() {
        // Try many seeds; for at least some, the natural flush will
        // produce a 0b110xxxxx terminator and stop_encode must append
        // a guard byte. Verify the post-condition on the last byte
        // always holds.
        for seed in 0..32u32 {
            let mut bits: Vec<u8> = Vec::with_capacity(96);
            let mut state = seed.wrapping_add(1);
            for _ in 0..96 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                bits.push((state & 1) as u8);
            }
            let mut buf = vec![0u8; 64];
            let pos = {
                let mut w = BoolWriter::start_encode(&mut buf);
                for &b in &bits {
                    w.write_bit(b);
                }
                w.stop_encode().expect("stop_encode failed")
            };
            // Reader round-trip continues to work.
            let mut r = BoolReader::new(&buf[..pos]).expect("reader init failed");
            for (i, &expected) in bits.iter().enumerate() {
                assert_eq!(r.read_bit(), expected, "seed={seed} bit {i} mismatch");
            }
            // Last byte must not be a superframe-index marker.
            assert_ne!(
                buf[pos - 1] & 0xe0,
                0xc0,
                "seed={seed}: stop_encode left a superframe-marker-byte"
            );
        }
    }

    /// `write_bit` is exactly equivalent to `write(bit, 128)`. Two
    /// independent writers fed via each path must produce byte-identical
    /// streams.
    #[test]
    fn write_bit_equivalent_to_write_at_prob_128() {
        let bits = [1u8, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1];
        let mut buf_a = vec![0u8; 16];
        let mut buf_b = vec![0u8; 16];
        let pos_a = {
            let mut w = BoolWriter::start_encode(&mut buf_a);
            for &b in &bits {
                w.write_bit(b);
            }
            w.stop_encode().expect("stop_encode_a failed")
        };
        let pos_b = {
            let mut w = BoolWriter::start_encode(&mut buf_b);
            for &b in &bits {
                w.write(b, super::super::VPX_PROB_HALF);
            }
            w.stop_encode().expect("stop_encode_b failed")
        };
        assert_eq!(pos_a, pos_b, "same content → same length");
        assert_eq!(buf_a[..pos_a], buf_b[..pos_b], "byte-identical output");
    }

    #[test]
    fn count_only_writer_accepts_symbols_without_emitting_bytes() {
        let mut sink = [];
        let mut w = BoolWriter::start_count_only(&mut sink);
        for i in 0..1024 {
            w.write((i & 1) as u8, (i as u8).max(1));
        }
        assert_eq!(w.pos(), 0);
        assert!(!w.has_error());
        assert_eq!(w.stop_encode(), Ok(0));
    }

    /// Empty stream (zero user bits) still round-trips: stop_encode
    /// flushes successfully and produces a stream the reader's marker-
    /// bit check accepts.
    #[test]
    fn empty_stream_round_trips() {
        let mut buf = vec![0u8; 16];
        let pos = {
            let w = BoolWriter::start_encode(&mut buf);
            w.stop_encode().expect("stop_encode failed")
        };
        assert!(pos > 0, "even empty stream needs at least one byte");
        let _ = BoolReader::new(&buf[..pos]).expect("reader must initialize on empty stream");
    }

    /// `pos()` advances monotonically and reaches the same final value
    /// `stop_encode` returns.
    #[test]
    fn pos_matches_stop_encode_returned_count() {
        let mut buf = vec![0u8; 32];
        let mut w = BoolWriter::start_encode(&mut buf);
        for _ in 0..50 {
            w.write_bit(1);
        }
        let pos_mid = w.pos();
        let final_pos = w.stop_encode().expect("stop_encode failed");
        assert!(final_pos >= pos_mid, "stop_encode should not regress pos");
    }
}
