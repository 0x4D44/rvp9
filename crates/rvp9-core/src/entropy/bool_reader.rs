//! Boolean (arithmetic) reader.
//!
//! Mirrors libvpx `vpx_dsp/bitreader.{h,c}` for the **8-byte BD_VALUE**
//! variant (entropy LLD §7.1: 32-bit targets out of scope). No decryption
//! callback (entropy LLD §7.2 — `vpx_decrypt_cb` + `decrypt_state` are
//! not ported).
//!
//! ## EOF semantics
//!
//! libvpx uses a sentinel `LOTS_OF_BITS = 0x40000000` added to the
//! `count` register when the user buffer is exhausted, so that
//! subsequent reads succeed (returning whatever bits are still buffered)
//! and `vpx_reader_has_error` can detect "we read past EOF" *after the
//! fact*. We preserve this exactly (entropy LLD §7.6 — "LOTS_OF_BITS and
//! EOF semantics preserved").
//!
//! `has_error()` is therefore the right post-decode integrity check; a
//! `read()` that crosses EOF returns a valid-looking bit value but
//! `has_error()` will return `true` afterwards.

/// Per-LLD-§7.6: the magic sentinel added to `count` when the buffer is
/// drained. Any positive value larger than `BD_VALUE_SIZE` would work;
/// libvpx picked `0x40000000` because it fits in a single ARM immediate.
const LOTS_OF_BITS: i32 = 0x4000_0000;

/// Number of bits in the BD_VALUE register. We hardcode 64-bit per
/// entropy LLD §7.1 (`BD_VALUE = u64`, so `BD_VALUE_SIZE = 64`).
const BD_VALUE_SIZE: i32 = 64;

/// Byte-stream boolean (arithmetic) reader.
///
/// The decryption callback path (libvpx `bitreader.h:47-49` + `bitreader.c`
/// `decrypt_cb` branch) is omitted per entropy LLD §7.2.
#[derive(Debug)]
pub struct BoolReader<'a> {
    value: u64,
    range: u32,
    count: i32,
    /// Cursor within `buffer`.
    pos: usize,
    buffer: &'a [u8],
}

/// Returned by [`BoolReader::new`] when the marker bit is malformed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvalidMarkerBit;

impl std::fmt::Display for InvalidMarkerBit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BoolReader: stream did not begin with a 0-bit at prob 128")
    }
}

impl std::error::Error for InvalidMarkerBit {}

impl<'a> BoolReader<'a> {
    /// Initialise a reader over `buffer`. Mirrors `vpx_reader_init`
    /// (`bitreader.c:21-36`), including the trailing `read_bit() != 0`
    /// marker-bit check that returns `Err(InvalidMarkerBit)` if the
    /// leading bit isn't `0` at prob 128.
    pub fn new(buffer: &'a [u8]) -> Result<Self, InvalidMarkerBit> {
        let mut r = Self {
            value: 0,
            range: 255,
            count: -8,
            pos: 0,
            buffer,
        };
        r.fill();
        if r.read_bit() != 0 {
            Err(InvalidMarkerBit)
        } else {
            Ok(r)
        }
    }

    /// Decode one bit at probability `prob`. Mirrors `vpx_read`
    /// (`bitreader.h:77-136`) verbatim.
    #[inline]
    pub fn read(&mut self, prob: u8) -> u8 {
        // split = (range * prob + (256 - prob)) >> 8
        let split: u32 = (self.range * u32::from(prob) + (256 - u32::from(prob))) >> 8;

        if self.count < 0 {
            self.fill();
        }

        let mut value = self.value;
        let mut count = self.count;

        let bigsplit: u64 = (split as u64) << (BD_VALUE_SIZE - 8);

        let mut range = split;
        let mut bit: u8 = 0;

        if value >= bigsplit {
            range = self.range - split;
            value = value.wrapping_sub(bigsplit);
            bit = 1;
        }

        // Renormalise.
        let shift = u32::from(super::tables::VPX_NORM[(range & 0xff) as usize]);
        range <<= shift;
        value = value.wrapping_shl(shift);
        count -= shift as i32;

        self.value = value;
        self.count = count;
        self.range = range;
        bit
    }

    /// Decode one bit at probability 128 (max-entropy).
    #[inline]
    pub fn read_bit(&mut self) -> u8 {
        self.read(super::VPX_PROB_HALF)
    }

    /// Decode `bits` bits as an MSB-first unsigned literal.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if `bits > 32`.
    #[inline]
    pub fn read_literal(&mut self, bits: u8) -> u32 {
        debug_assert!(bits <= 32, "BoolReader::read_literal: bits={bits} > 32");
        let n = bits.min(32);
        let mut lit: u32 = 0;
        for i in (0..n).rev() {
            lit |= u32::from(self.read_bit()) << i;
        }
        lit
    }

    /// Decode a value by walking a libvpx-format probability tree.
    /// Mirrors libvpx `vpx_read_tree` (`vpx_dsp/bitreader.h:150-156`).
    ///
    /// `tree` is the flat libvpx tree representation: positive
    /// entries are next-node indices; negative entries `-leaf` are
    /// terminal leaves whose returned value is `leaf` (the negation
    /// recovers the positive enum value). `probs` are the
    /// non-leaf-node probabilities; `probs[i >> 1]` gates the
    /// branch decision at tree node `i`.
    ///
    /// Example (`vp9_partition_tree`):
    /// ```text
    /// tree = [-NONE, 2, -HORZ, 4, -VERT, -SPLIT]
    /// probs = [p_node0, p_node2, p_node4]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics on malformed trees (positive index out of bounds) in
    /// debug builds; release builds rely on the caller passing a
    /// libvpx-shaped tree.
    #[inline]
    pub fn read_tree(&mut self, tree: &[i8], probs: &[u8]) -> i8 {
        let mut i: i8 = 0;
        loop {
            let bit = self.read(probs[(i >> 1) as usize]);
            i = tree[i as usize + bit as usize];
            if i <= 0 {
                return -i;
            }
        }
    }

    /// Check whether the reader has crossed EOF. Returns `true` if **any
    /// `read` since `new` consumed bits beyond the user buffer**.
    ///
    /// libvpx's `vpx_reader_has_error` (`bitreader.h:59-75`) returns
    /// `true` iff `count > BD_VALUE_SIZE && count < LOTS_OF_BITS`.
    /// We preserve that contract exactly.
    #[inline]
    #[must_use]
    pub fn has_error(&self) -> bool {
        self.count > BD_VALUE_SIZE && self.count < LOTS_OF_BITS
    }

    /// Replenish the `value` register from `buffer[pos..]`. Mirrors
    /// `vpx_reader_fill` (`bitreader.c:38-91`), simplified per
    /// entropy LLD §7.2 (no decryption callback).
    pub fn fill(&mut self) {
        let bytes_left = self.buffer.len() - self.pos;
        let bits_left = (bytes_left as i32) * 8;
        let mut shift = BD_VALUE_SIZE - 8 - (self.count + 8);

        if bits_left > BD_VALUE_SIZE {
            // Fast path: at least 8 source bytes available.
            let bits = (shift & !0x7) + 8;
            let mut nv: u64 = 0;
            // Big-endian 8-byte load.
            for k in 0..8 {
                nv = (nv << 8) | u64::from(self.buffer[self.pos + k]);
            }
            nv >>= BD_VALUE_SIZE - bits;
            self.count += bits;
            self.pos += (bits >> 3) as usize;
            self.value |= nv << (shift & 0x7);
        } else {
            // Slow path: end of stream within reach. May set LOTS_OF_BITS.
            let bits_over = shift + 8 - bits_left;
            let mut loop_end = 0;
            if bits_over >= 0 {
                self.count += LOTS_OF_BITS;
                loop_end = bits_over;
            }
            if bits_over < 0 || bits_left > 0 {
                while shift >= loop_end {
                    self.count += 8;
                    self.value |= u64::from(self.buffer[self.pos]) << shift;
                    self.pos += 1;
                    shift -= 8;
                    if self.pos >= self.buffer.len() {
                        break;
                    }
                }
            }
        }
    }

    /// Number of bytes consumed from `buffer`.
    #[inline]
    #[must_use]
    pub fn bytes_consumed(&self) -> usize {
        self.pos
    }

    /// Diagnostic view of the bool-coder register: `(range, value, count, pos)`.
    /// Mirrors libvpx's internal `vpx_reader::{range,value,count,buffer}`. Used
    /// by the M.5 cursor-state instrumentation hook to checkpoint reader state
    /// at each step in `decode_block` (`RVP9_DEBUG_BOOLCURSOR=1`). Cheap
    /// (4 word copies); not on any hot path.
    #[inline]
    #[must_use]
    pub fn cursor_state(&self) -> (u32, u64, i32, usize) {
        (self.range, self.value, self.count, self.pos)
    }

    /// Diagnostic view of the underlying byte buffer. Used by the Stage 3
    /// stretch M.8 oracle-export hook to dump the bitstream slice that a
    /// captured snapshot was taken from, so an independent libvpx oracle
    /// can be initialised with the same `(range, value, count, pos)` and
    /// `buffer` pointer.
    #[inline]
    #[must_use]
    pub fn buffer(&self) -> &[u8] {
        self.buffer
    }

    /// Snapshot the full register state for later restoration. Mirrors a
    /// hand-rolled `memcpy(&saved, &reader, sizeof(reader))` pattern that
    /// libvpx debug builds use to rewind the reader for differential
    /// re-decode against an oracle. Used by the M.6 runtime-differential
    /// hook (`RVP9_DEBUG_REFCOEFS=1`) to replay `decode_coefs` against the
    /// libvpx reference port on identical input.
    #[inline]
    #[must_use]
    pub fn snapshot(&self) -> BoolReaderSnapshot {
        BoolReaderSnapshot {
            value: self.value,
            range: self.range,
            count: self.count,
            pos: self.pos,
        }
    }

    /// Restore a previously [`snapshot`]ed register state. The buffer
    /// reference is unchanged (snapshots are only valid against the same
    /// underlying buffer slice the snapshot was taken from).
    ///
    /// [`snapshot`]: BoolReader::snapshot
    #[inline]
    pub fn restore(&mut self, snap: &BoolReaderSnapshot) {
        self.value = snap.value;
        self.range = snap.range;
        self.count = snap.count;
        self.pos = snap.pos;
    }
}

/// Opaque register snapshot for [`BoolReader::snapshot`] /
/// [`BoolReader::restore`]. Holds the four words that fully determine
/// reader state relative to its buffer; rewinding to a snapshot reproduces
/// the same bit sequence on re-reads.
#[derive(Clone, Copy, Debug)]
pub struct BoolReaderSnapshot {
    value: u64,
    range: u32,
    count: i32,
    pos: usize,
}

impl BoolReaderSnapshot {
    /// Diagnostic view of the captured register: `(range, value, count, pos)`.
    /// Same shape as [`BoolReader::cursor_state`]; lets the M.6 differential
    /// hook dump the pre-decode state on a divergence panic.
    #[inline]
    #[must_use]
    pub fn cursor_state(&self) -> (u32, u64, i32, usize) {
        (self.range, self.value, self.count, self.pos)
    }

    /// Construct a snapshot from raw register values. Used by Stage 3
    /// stretch M.8's oracle diff harness to re-create a previously
    /// captured reader pre-state so we can replay `decode_coefs`
    /// against a libvpx C oracle on the same inputs.
    #[inline]
    #[must_use]
    pub fn from_parts(value: u64, range: u32, count: i32, pos: usize) -> Self {
        Self {
            value,
            range,
            count,
            pos,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::BoolWriter;

    /// Round-trip a sequence of equal-probability bits.
    #[test]
    fn round_trip_bits_p128() {
        let bits: Vec<u8> = (0..200).map(|i| (i * 37 + 11) as u8 & 1).collect();
        let mut buf = vec![0u8; 256];
        let n = {
            let mut w = BoolWriter::start_encode(&mut buf);
            for &b in &bits {
                w.write_bit(b);
            }
            w.stop_encode().unwrap()
        };
        let bytes = &buf[..n];
        let mut r = BoolReader::new(bytes).unwrap();
        for (i, &b) in bits.iter().enumerate() {
            assert_eq!(r.read_bit(), b, "bit {i} round-trip mismatch");
        }
        assert!(!r.has_error());
    }

    /// Round-trip a sequence of literals (8-bit prob writes).
    #[test]
    fn round_trip_literals() {
        let values: [u32; 5] = [0x12, 0x34, 0xFF, 0x00, 0xA5];
        let mut buf = vec![0u8; 64];
        let n = {
            let mut w = BoolWriter::start_encode(&mut buf);
            for &v in &values {
                w.write_literal(v, 8);
            }
            w.stop_encode().unwrap()
        };
        let bytes = &buf[..n];
        let mut r = BoolReader::new(bytes).unwrap();
        for &v in &values {
            assert_eq!(r.read_literal(8), v);
        }
        assert!(!r.has_error());
    }

    /// Round-trip with a non-trivial probability model (alternating 1/255
    /// to exercise the renormalisation path).
    #[test]
    fn round_trip_skewed_probabilities() {
        let bits: Vec<(u8, u8)> = (0..150)
            .map(|i| (((i * 19) & 1) as u8, if (i * 7) & 1 == 0 { 32 } else { 224 }))
            .collect();
        let mut buf = vec![0u8; 256];
        let n = {
            let mut w = BoolWriter::start_encode(&mut buf);
            for &(b, p) in &bits {
                w.write(b, p);
            }
            w.stop_encode().unwrap()
        };
        let bytes = &buf[..n];
        let mut r = BoolReader::new(bytes).unwrap();
        for (i, &(b, p)) in bits.iter().enumerate() {
            assert_eq!(r.read(p), b, "symbol {i} (prob={p}) round-trip mismatch");
        }
        assert!(!r.has_error());
    }

    /// Reading past the stream sets has_error. The bits returned past
    /// EOF are unspecified (libvpx semantics — see entropy LLD §7.6).
    #[test]
    fn over_read_sets_error() {
        let bits: Vec<u8> = (0..32).map(|_| 1).collect();
        let mut buf = vec![0u8; 16];
        let n = {
            let mut w = BoolWriter::start_encode(&mut buf);
            for &b in &bits {
                w.write_bit(b);
            }
            w.stop_encode().unwrap()
        };
        let bytes = &buf[..n];
        let mut r = BoolReader::new(bytes).unwrap();
        for _ in 0..bits.len() {
            let _ = r.read_bit();
        }
        // Now read past EOF for a while; has_error should eventually be true.
        for _ in 0..200 {
            let _ = r.read_bit();
        }
        assert!(r.has_error(), "expected over-read to set has_error");
    }
}
