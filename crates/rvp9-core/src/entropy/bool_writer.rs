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
}

impl<'a> BoolWriter<'a> {
    /// Create a writer over `buffer`, initialised to start-of-frame state
    /// and with the leading marker bit (a `0` at prob 128) already
    /// emitted, matching libvpx `vpx_start_encode` (`bitwriter.c:20-31`).
    pub fn start_encode(buffer: &'a mut [u8]) -> Self {
        let size = u32::try_from(buffer.len()).unwrap_or(i32::MAX as u32);
        let mut w = Self {
            low_value: 0,
            range: 255,
            count: -24,
            pos: 0,
            size,
            buffer,
            error: false,
        };
        // Per libvpx: emit a `0` at probability 128 as a marker bit so
        // the reader's `vpx_reader_init` can verify the start of stream.
        w.write_bit(0);
        w
    }

    /// Encode a single bit `bit` (0 or 1) using the given 8-bit
    /// probability `probability` that the bit is `0`.
    ///
    /// Mirrors `vpx_write` (`vpx_dsp/bitwriter.h:46-116`) verbatim — the
    /// carry-back trick at the `0x80000000` check is preserved exactly.
    #[inline]
    pub fn write(&mut self, bit: u8, probability: u8) {
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
