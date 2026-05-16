//! MSB-first raw bit writer and reader.
//!
//! Mirrors libvpx `vpx_dsp/bitwriter_buffer.{h,c}` (writer) and
//! `bitreader_buffer.{h,c}` (reader) bit-for-bit. These are the byte-level
//! bit-packing primitives used by VP9's uncompressed frame header and
//! superframe index — the boolean-arithmetic coder for the compressed
//! header lives in [`crate::entropy`].
//!
//! ## Semantics summary
//!
//! - **Bit order:** MSB-first within each byte. Bit 0 of a byte is the
//!   *most* significant bit — see libvpx `bitwriter_buffer.c:38-49`.
//! - **Writer poisoning:** the first overflow sets the `error` flag and
//!   silently drops all subsequent writes, matching libvpx exactly. The
//!   accumulated state is recoverable via [`RawBitWriter::finish`].
//! - **Reader EOF:** libvpx returns `0` and calls a `setjmp`-driven
//!   error handler; we return [`BitstreamError::Truncated`] from each
//!   read method (D-09 RESOLVED — see `notes/divergence-ledger.md`).
//! - **`write_inv_signed_literal`** writes magnitude first, then sign bit
//!   (libvpx `bitwriter_buffer.c:58-62`). The reader's
//!   `read_inv_signed_literal` is aliased to `read_signed_literal` because
//!   libvpx's reader uses the same wire format for both
//!   (`bitreader_buffer.c:42-44`).

use super::error::BitstreamError;

/// MSB-first raw bit writer over a borrowed byte slice.
///
/// Mirrors libvpx `struct vpx_write_bit_buffer` (`vpx_dsp/bitwriter_buffer.h:20-31`).
/// The buffer is **never owned**; it is a `&mut [u8]` supplied by the
/// caller. On overflow the writer becomes poisoned and further writes
/// are silently dropped, matching libvpx semantics. Use
/// [`Self::finish`] to convert poisoned state into a `Result`.
#[derive(Debug)]
pub struct RawBitWriter<'a> {
    buffer: &'a mut [u8],
    bit_offset: usize,
    error: bool,
}

impl<'a> RawBitWriter<'a> {
    /// Constructs a writer at bit offset 0.
    ///
    /// Mirrors `vpx_wb_init` (`bitwriter_buffer.c:18-24`). The caller is
    /// responsible for ensuring `buffer` is zero-initialised — the
    /// writer relies on this for the `wb->bit_buffer[p] = bit << q`
    /// fast-path when `q == 7` (libvpx `bitwriter_buffer.c:44-49`).
    #[must_use]
    #[inline]
    pub fn new(buffer: &'a mut [u8]) -> Self {
        Self {
            buffer,
            bit_offset: 0,
            error: false,
        }
    }

    /// Writes a single bit (LSB of `bit`) at the current MSB-first
    /// position.
    ///
    /// On overflow, sets the poison flag and silently drops the bit
    /// (matching libvpx). The `debug_assert!` mirrors libvpx's
    /// `assert((wb->bit_buffer[p] & (1 << q)) == 0)` at line 47.
    #[inline]
    pub fn write_bit(&mut self, bit: u8) {
        if self.error {
            return;
        }
        let off = self.bit_offset;
        let p = off >> 3;
        let q = 7 - (off & 0x7) as u8;
        if p >= self.buffer.len() {
            self.error = true;
            return;
        }
        let masked = (bit & 1) << q;
        if q == 7 {
            self.buffer[p] = masked;
        } else {
            debug_assert!(
                self.buffer[p] & (1 << q) == 0,
                "rvp9: RawBitWriter double-write at byte {p}, bit {q}; \
                 caller forgot to zero-init the buffer"
            );
            self.buffer[p] |= masked;
        }
        self.bit_offset = off + 1;
    }

    /// Writes the low `bits` bits of `data`, MSB-first.
    ///
    /// Mirrors `vpx_wb_write_literal` (`bitwriter_buffer.c:53-56`).
    ///
    /// # Panics (debug only)
    ///
    /// Panics in debug mode if `bits > 32`. Release builds clamp the
    /// loop to 32 iterations.
    #[inline]
    pub fn write_literal(&mut self, data: u32, bits: u8) {
        debug_assert!(bits <= 32, "RawBitWriter::write_literal: bits={bits} > 32");
        let n = bits.min(32);
        for i in (0..n).rev() {
            self.write_bit(((data >> i) & 1) as u8);
        }
    }

    /// Writes an inverse-uniform-coded signed literal: magnitude first
    /// (as an unsigned `bits`-bit literal), then a single sign bit
    /// (`1` for negative). Mirrors `vpx_wb_write_inv_signed_literal`
    /// (`bitwriter_buffer.c:58-62`).
    #[inline]
    pub fn write_inv_signed_literal(&mut self, data: i32, bits: u8) {
        self.write_literal(data.unsigned_abs(), bits);
        self.write_bit(u8::from(data < 0));
    }

    /// Returns the current bit offset (MSB-first, from byte 0 bit 0).
    /// Used by the back-patch for `compressed_hdr_size` (LLD §5.1).
    #[inline]
    #[must_use]
    pub fn bit_offset(&self) -> usize {
        self.bit_offset
    }

    /// Whether the writer has overflowed its buffer at any point.
    #[inline]
    #[must_use]
    pub fn has_error(&self) -> bool {
        self.error
    }

    /// Bytes consumed so far, rounded up to the next byte boundary.
    /// Mirrors `vpx_wb_bytes_written` (`bitwriter_buffer.c:30-33`).
    ///
    /// libvpx asserts no error here; we follow the same contract.
    #[inline]
    #[must_use]
    pub fn bytes_written(&self) -> usize {
        debug_assert!(
            !self.error,
            "RawBitWriter::bytes_written called on poisoned writer"
        );
        (self.bit_offset / 8) + usize::from(!self.bit_offset.is_multiple_of(8))
    }

    /// Consumes the writer, returning the number of bytes written.
    /// Returns [`BitstreamError::BufferFull`] if any write overflowed.
    pub fn finish(self) -> Result<usize, BitstreamError> {
        if self.error {
            Err(BitstreamError::BufferFull)
        } else {
            Ok((self.bit_offset / 8) + usize::from(!self.bit_offset.is_multiple_of(8)))
        }
    }
}

/// MSB-first raw bit reader over a borrowed byte slice.
///
/// Mirrors libvpx `struct vpx_read_bit_buffer` and `vpx_rb_*`
/// (`vpx_dsp/bitreader_buffer.{h,c}`). Errors are surfaced via
/// [`BitstreamError::Truncated`] rather than libvpx's `setjmp`-driven
/// error-handler callback.
#[derive(Debug)]
pub struct RawBitReader<'a> {
    buffer: &'a [u8],
    bit_offset: usize,
}

impl<'a> RawBitReader<'a> {
    /// Constructs a reader at bit offset 0.
    #[must_use]
    #[inline]
    pub fn new(buffer: &'a [u8]) -> Self {
        Self {
            buffer,
            bit_offset: 0,
        }
    }

    /// Reads a single bit (MSB-first).
    ///
    /// Returns [`BitstreamError::Truncated`] if the buffer is exhausted.
    /// libvpx silently returns `0` here (`bitreader_buffer.c:26-28`); we
    /// surface the truncation explicitly per LLD §3.2 / D-09.
    #[inline]
    pub fn read_bit(&mut self) -> Result<u8, BitstreamError> {
        let off = self.bit_offset;
        let p = off >> 3;
        let q = 7 - (off & 0x7) as u8;
        if p >= self.buffer.len() {
            return Err(BitstreamError::Truncated {
                needed: p + 1,
                available: self.buffer.len(),
            });
        }
        let bit = (self.buffer[p] >> q) & 1;
        self.bit_offset = off + 1;
        Ok(bit)
    }

    /// Reads the next `bits` bits as an MSB-first unsigned value.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if `bits > 32`.
    #[inline]
    pub fn read_literal(&mut self, bits: u8) -> Result<u32, BitstreamError> {
        debug_assert!(bits <= 32, "RawBitReader::read_literal: bits={bits} > 32");
        let n = bits.min(32);
        let mut value: u32 = 0;
        for i in (0..n).rev() {
            let b = u32::from(self.read_bit()?);
            value |= b << i;
        }
        Ok(value)
    }

    /// Reads an inverse-uniform-coded signed literal: magnitude first
    /// (unsigned `bits`-bit literal), then sign bit (1 = negative).
    /// Mirrors `vpx_rb_read_signed_literal` and
    /// `vpx_rb_read_inv_signed_literal` (`bitreader_buffer.c:37-44`)
    /// — both libvpx readers use the same wire format.
    #[inline]
    pub fn read_inv_signed_literal(&mut self, bits: u8) -> Result<i32, BitstreamError> {
        let mag = self.read_literal(bits)? as i32;
        let sign = self.read_bit()?;
        if sign != 0 { Ok(-mag) } else { Ok(mag) }
    }

    /// Current bit offset (from byte 0 bit 0, MSB-first).
    #[inline]
    #[must_use]
    pub fn bit_offset(&self) -> usize {
        self.bit_offset
    }

    /// Bytes consumed so far, rounded up. Mirrors `vpx_rb_bytes_read`
    /// (`bitreader_buffer.c:13-15`).
    #[inline]
    #[must_use]
    pub fn bytes_consumed(&self) -> usize {
        (self.bit_offset + 7) >> 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writing one bit at offset 0 stores it at bit 7 of byte 0
    /// (MSB-first). Mirrors libvpx's behaviour at
    /// `bitwriter_buffer.c:44-49`.
    #[test]
    fn write_single_bit_msb_first() {
        let mut buf = [0u8; 2];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_bit(1);
        assert_eq!(w.bit_offset(), 1);
        assert_eq!(w.finish().unwrap(), 1);
        assert_eq!(buf[0], 0b1000_0000);
    }

    /// Writing 0b101 as a 3-bit literal lands in the top 3 bits of byte 0.
    #[test]
    fn write_literal_msb_first() {
        let mut buf = [0u8; 1];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_literal(0b101, 3);
        let n = w.finish().unwrap();
        assert_eq!(n, 1);
        assert_eq!(buf[0], 0b1010_0000);
    }

    /// Bytes_written rounds up after a non-byte-aligned write.
    #[test]
    fn bytes_written_rounds_up() {
        let mut buf = [0u8; 2];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_literal(0xF, 5); // 5 bits → bytes_written = 1
        assert_eq!(w.bytes_written(), 1);
        w.write_bit(1); // 6 bits total → still 1 byte
        assert_eq!(w.bytes_written(), 1);
        w.write_literal(0, 3); // 9 bits total → 2 bytes
        assert_eq!(w.bytes_written(), 2);
    }

    /// `write_inv_signed_literal(-5, 4)` writes the magnitude `5` as 4
    /// bits MSB-first, then a 1-bit sign. Total 5 bits = `0b01011`.
    #[test]
    fn write_inv_signed_negative() {
        let mut buf = [0u8; 1];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_inv_signed_literal(-5, 4);
        assert_eq!(w.finish().unwrap(), 1);
        // magnitude 5 = 0b0101, sign bit = 1 → upper 5 bits = 0b01011
        assert_eq!(buf[0], 0b0101_1000);
    }

    #[test]
    fn write_inv_signed_positive() {
        let mut buf = [0u8; 1];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_inv_signed_literal(5, 4);
        assert_eq!(w.finish().unwrap(), 1);
        // magnitude 5 = 0b0101, sign bit = 0 → upper 5 bits = 0b01010
        assert_eq!(buf[0], 0b0101_0000);
    }

    /// Overflowing the writer poisons it and subsequent writes are dropped.
    #[test]
    fn overflow_poisons_writer() {
        let mut buf = [0u8; 1];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_literal(0xFF, 8); // fills byte 0
        assert!(!w.has_error());
        w.write_bit(1); // overflows
        assert!(w.has_error());
        // Dropped writes after overflow:
        w.write_literal(0xF, 4);
        assert!(w.has_error());
        assert!(matches!(w.finish(), Err(BitstreamError::BufferFull)));
        assert_eq!(buf[0], 0xFF);
    }

    /// Reader round-trips a writer's output bit-for-bit.
    #[test]
    fn round_trip_literal() {
        let mut buf = [0u8; 4];
        {
            let mut w = RawBitWriter::new(&mut buf);
            w.write_literal(0xDEAD_BEEF, 32);
            let n = w.finish().unwrap();
            assert_eq!(n, 4);
        }
        let mut r = RawBitReader::new(&buf);
        assert_eq!(r.read_literal(32).unwrap(), 0xDEAD_BEEF);
        assert_eq!(r.bytes_consumed(), 4);
    }

    /// Reader returns Truncated on EOF (not 0 like libvpx).
    #[test]
    fn reader_eof_returns_truncated() {
        let buf = [0b1000_0000];
        let mut r = RawBitReader::new(&buf);
        assert_eq!(r.read_bit().unwrap(), 1);
        // 7 more bits available.
        for _ in 0..7 {
            r.read_bit().unwrap();
        }
        match r.read_bit() {
            Err(BitstreamError::Truncated { needed, available }) => {
                assert_eq!(needed, 2);
                assert_eq!(available, 1);
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    /// Inv-signed literal round-trips.
    #[test]
    fn round_trip_inv_signed() {
        for v in [-15, -1, 0, 1, 7, 15] {
            let mut buf = [0u8; 2];
            {
                let mut w = RawBitWriter::new(&mut buf);
                w.write_inv_signed_literal(v, 4);
                w.finish().unwrap();
            }
            let mut r = RawBitReader::new(&buf);
            let got = r.read_inv_signed_literal(4).unwrap();
            assert_eq!(got, v, "round-trip failed for {v}");
        }
    }

    /// Differential against the libvpx C reference behaviour for a
    /// specific sequence: write `[bit 1, literal 0b101_3, inv_signed(-5,4)]`
    /// and verify the resulting byte layout matches what libvpx would
    /// produce (computed by hand below).
    ///
    /// Sequence:
    ///   - bit 1       → bit 0 of byte 0 (MSB)
    ///   - 0b101 in 3  → bits 1..=3 of byte 0
    ///   - mag 5 (4 b) → bits 4..=7 of byte 0 = 0b0101
    ///   - sign bit 1  → bit 0 of byte 1
    ///
    /// Resulting bytes: byte0 = 0b1_101_0101 = 0xD5, byte1 = 0b1_0000000 = 0x80
    #[test]
    fn differential_known_sequence() {
        let mut buf = [0u8; 2];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_bit(1);
        w.write_literal(0b101, 3);
        w.write_inv_signed_literal(-5, 4);
        let n = w.finish().unwrap();
        assert_eq!(n, 2);
        assert_eq!(buf, [0xD5, 0x80]);
    }
}
