//! Encoder-side prob-diff-update helper — the inverse of
//! [`super::diff_update::diff_update_prob`].
//!
//! Mirrors libvpx `vp9_subexp.c` (`vp9/encoder/vp9_subexp.c:35-115`).
//!
//! ## Algorithm
//!
//! For each probability slot the encoder wants to update:
//!
//! 1. Compute `delp = remap_prob(newp, oldp)`. Mirrors libvpx
//!    `remap_prob` (`vp9_subexp.c:44-77`).
//! 2. Emit `delp` via `encode_term_subexp`. Mirrors libvpx
//!    `encode_term_subexp` (`vp9_subexp.c:100-110`).
//!
//! Higher-level helpers in this module:
//!
//! - [`write_prob_diff_update`] writes the `delp` payload only —
//!   used after the caller emits the gate bit. Mirrors libvpx
//!   `vp9_write_prob_diff_update` (`vp9_subexp.c:112-115`).
//! - [`cond_prob_diff_update`] emits the 1-bit gate at
//!   [`DIFF_UPDATE_PROB`] AND the payload (when `newp != oldp`),
//!   matching the "force update on any diff" policy the rvp9
//!   encoder uses in v1. Updates `*oldp` in place so the caller
//!   can track the post-update value.
//!
//! libvpx's `vp9_cond_prob_diff_update` (`vp9_subexp.c:182-196`)
//! makes a cost-benefit decision against per-symbol counts; we
//! skip that — the encoder above us has already decided which
//! probs it wants to change.

use super::BoolWriter;
use super::diff_update::{DIFF_UPDATE_PROB, MAX_PROB};

/// Forward permutation — encoder-side analogue of
/// [`super::diff_update::INV_MAP_TABLE`]. `MAP_TABLE[i] = j` iff
/// `INV_MAP_TABLE[j] = i + 1`. Sourced verbatim from libvpx
/// `vp9_subexp.c:47-66`.
#[rustfmt::skip]
static MAP_TABLE: [u8; (MAX_PROB - 1) as usize] = [
    20,  21,  22,  23,  24,  25,  0,   26,  27,  28,  29,  30,  31,  32,  33,
    34,  35,  36,  37,  1,   38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
    48,  49,  2,   50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
    3,   62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  4,   74,
    75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  5,   86,  87,  88,
    89,  90,  91,  92,  93,  94,  95,  96,  97,  6,   98,  99,  100, 101, 102,
    103, 104, 105, 106, 107, 108, 109, 7,   110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 120, 121, 8,   122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 9,   134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    145, 10,  146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 11,
    158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 12,  170, 171,
    172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 13,  182, 183, 184, 185,
    186, 187, 188, 189, 190, 191, 192, 193, 14,  194, 195, 196, 197, 198, 199,
    200, 201, 202, 203, 204, 205, 15,  206, 207, 208, 209, 210, 211, 212, 213,
    214, 215, 216, 217, 16,  218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
    228, 229, 17,  230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
    18,  242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 19,
];

/// Encoder analogue of
/// [`super::diff_update::inv_recenter_nonneg`]. Mirrors libvpx
/// `recenter_nonneg` (`vp9_subexp.c:35-42`).
#[inline]
const fn recenter_nonneg(v: i32, m: i32) -> i32 {
    if v > (m << 1) {
        v
    } else if v >= m {
        (v - m) << 1
    } else {
        ((m - v) << 1) - 1
    }
}

/// Map a probability pair `(newp, oldp)` to the 8-bit `delp` value
/// the wire format carries. Mirrors libvpx `remap_prob`
/// (`vp9_subexp.c:44-77`).
///
/// Inverse of [`super::diff_update::inv_remap_prob`]: for any
/// `newp != oldp` with `newp` in `1..=255`, the decoder applied to
/// `(remap_prob(newp, oldp), oldp)` returns `newp`.
#[must_use]
pub fn remap_prob(newp: u8, oldp: u8) -> u8 {
    let v = newp as i32 - 1;
    let m = oldp as i32 - 1;
    let i = if (m << 1) <= MAX_PROB as i32 {
        recenter_nonneg(v, m) - 1
    } else {
        recenter_nonneg(MAX_PROB as i32 - 1 - v, MAX_PROB as i32 - 1 - m) - 1
    };
    debug_assert!(
        (0..MAP_TABLE.len() as i32).contains(&i),
        "remap_prob: pre-map index {i} out of range"
    );
    MAP_TABLE[i as usize]
}

/// Encode a uniform-ish value in `[0, 190]` ∪ `[64, 254]`-region.
/// Mirrors libvpx `encode_uniform` (`vp9_subexp.c:84-93`). Companion
/// of [`super::diff_update::decode_uniform`].
fn encode_uniform(w: &mut BoolWriter<'_>, v: i32) {
    const L: u8 = 8;
    const M: i32 = (1 << L) - 191;
    if v < M {
        w.write_literal(v as u32, L - 1);
    } else {
        w.write_literal((M + ((v - M) >> 1)) as u32, L - 1);
        w.write_literal(((v - M) & 1) as u32, 1);
    }
}

/// Encode a variable-length term-subexp value `delp` in `[0, 254]`.
/// Mirrors libvpx `encode_term_subexp` (`vp9_subexp.c:100-110`).
/// Companion of [`super::diff_update::decode_term_subexp`].
///
/// Wire prefix:
/// - `0` for `delp < 16` → 4-bit literal.
/// - `10` for `delp < 32` → 4-bit literal (`delp - 16`).
/// - `110` for `delp < 64` → 5-bit literal (`delp - 32`).
/// - `111` otherwise → `encode_uniform(delp - 64)` (which itself
///   emits 7 or 8 bits).
fn encode_term_subexp(w: &mut BoolWriter<'_>, word: i32) {
    debug_assert!((0..=254).contains(&word), "encode_term_subexp: word={word}");
    if word < 16 {
        w.write_literal(0, 1);
        w.write_literal(word as u32, 4);
    } else if word < 32 {
        w.write_literal(1, 1);
        w.write_literal(0, 1);
        w.write_literal((word - 16) as u32, 4);
    } else if word < 64 {
        w.write_literal(1, 1);
        w.write_literal(1, 1);
        w.write_literal(0, 1);
        w.write_literal((word - 32) as u32, 5);
    } else {
        w.write_literal(1, 1);
        w.write_literal(1, 1);
        w.write_literal(1, 1);
        encode_uniform(w, word - 64);
    }
}

/// Write the payload-only side of one prob-diff update: a remapped
/// `delp` term-subexp. Mirrors libvpx `vp9_write_prob_diff_update`
/// (`vp9_subexp.c:112-115`).
///
/// Callers must have already emitted the 1-bit gate at
/// [`DIFF_UPDATE_PROB`]; use [`cond_prob_diff_update`] for the
/// gate-and-payload combined version.
pub fn write_prob_diff_update(w: &mut BoolWriter<'_>, newp: u8, oldp: u8) {
    let delp = remap_prob(newp, oldp);
    encode_term_subexp(w, delp as i32);
}

/// Force-update conditional prob writer.
///
/// If `newp == *oldp` emit `gate=0`, otherwise emit `gate=1` followed
/// by `write_prob_diff_update(newp, *oldp)`. Updates `*oldp` to
/// `newp` after emitting the payload, so a sequence of calls
/// produces the same `oldp` trajectory the decoder will see.
///
/// Loosely mirrors libvpx `vp9_cond_prob_diff_update`
/// (`vp9_subexp.c:182-196`), but skips libvpx's rate-distortion
/// cost-benefit search — the encoder above us has already decided
/// which probs it wants to land on.
#[inline]
pub fn cond_prob_diff_update(w: &mut BoolWriter<'_>, oldp: &mut u8, newp: u8) {
    if *oldp == newp {
        w.write(0, DIFF_UPDATE_PROB);
    } else {
        w.write(1, DIFF_UPDATE_PROB);
        write_prob_diff_update(w, newp, *oldp);
        *oldp = newp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::BoolReader;
    use crate::entropy::diff_update::{diff_update_prob, inv_remap_prob};

    fn build_bitstream<F>(f: F) -> Vec<u8>
    where
        F: FnOnce(&mut BoolWriter),
    {
        let mut buf = vec![0u8; 256];
        let mut w = BoolWriter::start_encode(&mut buf);
        f(&mut w);
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);
        buf
    }

    /// `MAP_TABLE` and `INV_MAP_TABLE` form an inverse pair:
    /// `INV_MAP_TABLE[MAP_TABLE[i]] = i + 1` for every `i`.
    #[test]
    fn map_table_inverse_of_inv_map_table() {
        use crate::entropy::diff_update::inv_remap_prob;
        // Round-trip every (newp, oldp) pair with newp != oldp.
        for oldp in 1u16..=255 {
            for newp in 1u16..=255 {
                if newp == oldp {
                    continue;
                }
                let delp = remap_prob(newp as u8, oldp as u8);
                let recovered = inv_remap_prob(delp as i32, oldp as u8);
                assert_eq!(
                    recovered, newp as u8,
                    "round-trip failure: newp={newp}, oldp={oldp}, delp={delp}, got={recovered}"
                );
            }
        }
    }

    /// Encoder-side `recenter_nonneg` is the inverse of decoder
    /// `inv_recenter_nonneg` for arguments in range.
    #[test]
    fn recenter_nonneg_inverts_decoder() {
        use crate::entropy::diff_update::inv_remap_prob;
        // Spot-check via the higher-level round-trip already done.
        // Hand-check three points:
        // m=10, v=5 → v < m → ((10-5)<<1) - 1 = 9.
        assert_eq!(recenter_nonneg(5, 10), 9);
        // m=10, v=10 → v >= m → (0) << 1 = 0.
        assert_eq!(recenter_nonneg(10, 10), 0);
        // m=10, v=15 → v > 2m? 15>20? no; v>=m? yes → ((15-10)<<1) = 10.
        assert_eq!(recenter_nonneg(15, 10), 10);
        // m=10, v=21 → 21 > 20 → returns 21.
        assert_eq!(recenter_nonneg(21, 10), 21);
        let _ = inv_remap_prob; // silence unused import on certain builds
    }

    /// `encode_term_subexp` round-trips against the decoder for
    /// every `delp` value in `[0, 254]`.
    #[test]
    fn encode_term_subexp_round_trip_all_delp() {
        for delp in 0i32..=254 {
            let buf = build_bitstream(|w| {
                encode_term_subexp(w, delp);
                // Pad — the boolean reader needs enough trailing
                // bytes to renormalise past the last symbol.
                for _ in 0..16 {
                    w.write_bit(0);
                }
            });
            // We don't expose decode_term_subexp; round-trip via the
            // full diff_update wrapper instead by also writing the
            // gate bit and a known oldp; verify inv_remap_prob and
            // remap_prob agree on the round-trip.
            // Direct comparison: build a buffer with gate=1 +
            // term_subexp(delp), then feed to diff_update_prob with
            // a known oldp; the post-update prob must equal
            // inv_remap_prob(delp, oldp).
            let oldp = 100u8;
            let buf2 = build_bitstream(|w| {
                w.write(1, DIFF_UPDATE_PROB);
                encode_term_subexp(w, delp);
                for _ in 0..16 {
                    w.write_bit(0);
                }
            });
            let mut r = BoolReader::new(&buf2).expect("init");
            let mut p = oldp;
            let updated = diff_update_prob(&mut r, &mut p);
            assert!(updated, "gate=1 but diff_update_prob said no-update");
            assert_eq!(
                p,
                inv_remap_prob(delp, oldp),
                "term-subexp round-trip mismatch at delp={delp}"
            );
            // Silence unused.
            let _ = &buf;
        }
    }

    /// `cond_prob_diff_update`: when `newp == oldp` it emits one
    /// gate=0 bit at DIFF_UPDATE_PROB and the decoder leaves *p
    /// unchanged.
    #[test]
    fn cond_prob_diff_update_identity_is_gate_zero() {
        let buf = build_bitstream(|w| {
            let mut oldp = 100u8;
            cond_prob_diff_update(w, &mut oldp, 100);
            assert_eq!(oldp, 100);
            for _ in 0..16 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut p = 100u8;
        let updated = diff_update_prob(&mut r, &mut p);
        assert!(!updated);
        assert_eq!(p, 100);
    }

    /// `cond_prob_diff_update`: when `newp != oldp` the decoder
    /// recovers `newp` exactly and the caller's tracked `oldp`
    /// becomes `newp`.
    #[test]
    fn cond_prob_diff_update_diff_round_trip() {
        for (oldp, newp) in [(100u8, 50u8), (1, 255), (200, 199), (128, 1)] {
            let buf = build_bitstream(|w| {
                let mut tracked = oldp;
                cond_prob_diff_update(w, &mut tracked, newp);
                assert_eq!(tracked, newp);
                for _ in 0..16 {
                    w.write_bit(0);
                }
            });
            let mut r = BoolReader::new(&buf).expect("init");
            let mut p = oldp;
            let updated = diff_update_prob(&mut r, &mut p);
            assert!(updated, "gate should be 1 for oldp={oldp}, newp={newp}");
            assert_eq!(p, newp, "round-trip oldp={oldp}, newp={newp}");
        }
    }
}
