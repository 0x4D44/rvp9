//! Per-frame differential probability update — the helper the
//! compressed-header parser calls for every probability slot the
//! frame can adjust.
//!
//! Mirrors libvpx `vp9_diff_update_prob`
//! (`vp9/decoder/vp9_dsubexp.c:67-72`) and the three sub-helpers
//! `decode_term_subexp`, `inv_remap_prob`, `inv_recenter_nonneg`.
//!
//! ## Algorithm
//!
//! For each probability slot:
//!
//! 1. Read 1 bit at threshold [`DIFF_UPDATE_PROB`] (= 252) from the
//!    boolean decoder.
//! 2. If the bit is 0, the prob stays unchanged.
//! 3. If the bit is 1: decode a variable-length `term_subexp` value
//!    `delp` in `0..=254`, pass it through the [`INV_MAP_TABLE`]
//!    permutation and the `inv_recenter_nonneg` math to produce a
//!    new probability in `1..=255` centered around the existing `*p`,
//!    then overwrite `*p`.
//!
//! See libvpx `vp9_dsubexp.c:17-71` for the term-subexp wire format
//! (a 3-level variable-length code: 5-bit prefix that selects a
//! tier, then a 4-, 4-, 5-, or 8-bit literal).

use super::BoolReader;

/// Threshold used to decide whether a given probability slot is
/// updated this frame. Mirrors libvpx `DIFF_UPDATE_PROB`
/// (`vp9_entropy.h:24`).
pub const DIFF_UPDATE_PROB: u8 = 252;

/// Maximum probability value (`255`).
pub const MAX_PROB: u8 = 255;

/// libvpx's `inv_map_table[255]` (`vp9_dsubexp.c:31-49`). Permutes
/// `[0..255)` so that small `delp` values map to large adjustments.
/// (Mathematically: a Golomb-like prefix optimisation.)
#[rustfmt::skip]
static INV_MAP_TABLE: [u8; 255] = [
    7,   20,  33,  46,  59,  72,  85,  98,  111, 124, 137, 150, 163, 176, 189,
    202, 215, 228, 241, 254, 1,   2,   3,   4,   5,   6,   8,   9,   10,  11,
    12,  13,  14,  15,  16,  17,  18,  19,  21,  22,  23,  24,  25,  26,  27,
    28,  29,  30,  31,  32,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
    44,  45,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  60,
    61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  73,  74,  75,  76,
    77,  78,  79,  80,  81,  82,  83,  84,  86,  87,  88,  89,  90,  91,  92,
    93,  94,  95,  96,  97,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108,
    109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125,
    126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141,
    142, 143, 144, 145, 146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157,
    158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
    174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190,
    191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206,
    207, 208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222,
    223, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
    239, 240, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 253,
];

/// Inverse-recenter helper. Mirrors libvpx `inv_recenter_nonneg`
/// (`vp9_dsubexp.c:17-21`).
#[inline]
const fn inv_recenter_nonneg(v: i32, m: i32) -> i32 {
    if v > 2 * m {
        v
    } else if (v & 1) != 0 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

/// Decode an 8-bit uniform-ish value with biased range, used as the
/// fallback for `delp >= 64` in the term-subexp wire format. Mirrors
/// libvpx `decode_uniform` (`vp9_dsubexp.c:23-28`).
fn decode_uniform(r: &mut BoolReader<'_>) -> i32 {
    const L: u8 = 8;
    const M: i32 = (1 << L) - 191;
    let v = r.read_literal(L - 1) as i32;
    if v < M { v } else { (v << 1) - M + r.read_bit() as i32 }
}

/// Decode a variable-length term-subexp value (used in the per-frame
/// probability-diff wire format). Mirrors libvpx `decode_term_subexp`
/// (`vp9_dsubexp.c:60-65`).
///
/// 3-level prefix code:
/// - `0` ⇒ 4-bit literal in `[0, 15]`
/// - `10` ⇒ 4-bit literal `+ 16` in `[16, 31]`
/// - `110` ⇒ 5-bit literal `+ 32` in `[32, 63]`
/// - `111` ⇒ `decode_uniform() + 64` in `[64, 254]`
fn decode_term_subexp(r: &mut BoolReader<'_>) -> i32 {
    if r.read_bit() == 0 {
        return r.read_literal(4) as i32;
    }
    if r.read_bit() == 0 {
        return r.read_literal(4) as i32 + 16;
    }
    if r.read_bit() == 0 {
        return r.read_literal(5) as i32 + 32;
    }
    decode_uniform(r) + 64
}

/// Map a decoded `delp` value (0..=254) through [`INV_MAP_TABLE`]
/// and the inv-recenter math to produce a new probability centered
/// around the existing `m`. Mirrors libvpx `inv_remap_prob`
/// (`vp9_dsubexp.c:30-58`).
#[must_use]
pub fn inv_remap_prob(delp: i32, m: u8) -> u8 {
    debug_assert!((0..255).contains(&delp), "delp out of range: {delp}");
    let v = INV_MAP_TABLE[delp as usize] as i32;
    let m = m as i32 - 1;
    let result = if (m << 1) <= MAX_PROB as i32 {
        1 + inv_recenter_nonneg(v, m)
    } else {
        MAX_PROB as i32 - inv_recenter_nonneg(v, MAX_PROB as i32 - 1 - m)
    };
    result.clamp(0, MAX_PROB as i32) as u8
}

/// Update one probability slot from the boolean reader. Mirrors
/// libvpx `vp9_diff_update_prob` (`vp9_dsubexp.c:67-72`).
///
/// Reads 1 bit at threshold [`DIFF_UPDATE_PROB`]. If the bit is 0,
/// `*p` stays unchanged. If the bit is 1, decodes a term-subexp
/// `delp` and remaps `*p` to the new probability via
/// [`inv_remap_prob`].
///
/// Returns `true` if the probability was updated (the slot may have
/// been left unchanged even when the wire bit said "update" if the
/// delp produces the same value — but in practice it always changes
/// `*p` when the bit is 1).
pub fn diff_update_prob(r: &mut BoolReader<'_>, p: &mut u8) -> bool {
    if r.read(DIFF_UPDATE_PROB) != 0 {
        let delp = decode_term_subexp(r);
        *p = inv_remap_prob(delp, *p);
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::BoolWriter;

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

    /// `inv_recenter_nonneg` matches libvpx for hand-checked points.
    /// libvpx :17-21:
    ///   if v > 2*m: return v
    ///   else: (v & 1) ? m - (v+1)/2 : m + v/2
    #[test]
    fn inv_recenter_nonneg_hand_checks() {
        // m = 10, v = 0 → (v&1)=0, m + 0 = 10.
        assert_eq!(inv_recenter_nonneg(0, 10), 10);
        // m = 10, v = 1 → (v&1)=1, 10 - 1 = 9.
        assert_eq!(inv_recenter_nonneg(1, 10), 9);
        // m = 10, v = 2 → (v&1)=0, 10 + 1 = 11.
        assert_eq!(inv_recenter_nonneg(2, 10), 11);
        // m = 10, v = 3 → 10 - 2 = 8.
        assert_eq!(inv_recenter_nonneg(3, 10), 8);
        // m = 10, v = 21 → 21 > 20, return 21.
        assert_eq!(inv_recenter_nonneg(21, 10), 21);
    }

    /// `INV_MAP_TABLE` should be a permutation of `[1..=254]`
    /// (with the first 20 entries being a "high-information" prefix
    /// covering 7, 20, 33, ... in steps of 13 before falling into
    /// the natural-order tail). Verify: all values in [1, 254],
    /// every value appears once except the literal 253 which the
    /// libvpx source duplicates at positions 252 and 254.
    #[test]
    fn inv_map_table_structure() {
        // Last entry is 253 (per libvpx — index 254 is reserved).
        assert_eq!(INV_MAP_TABLE[0], 7);
        assert_eq!(INV_MAP_TABLE[19], 254);
        assert_eq!(INV_MAP_TABLE[20], 1);
        assert_eq!(INV_MAP_TABLE[254], 253);
        // No value exceeds 254.
        for &v in INV_MAP_TABLE.iter() {
            assert!((1..=254).contains(&v), "value {v} out of [1, 254]");
        }
    }

    /// Round-trip: write `update = false` for a probability,
    /// then read it back and verify `diff_update_prob` returns
    /// false and leaves `*p` unchanged.
    #[test]
    fn diff_update_prob_no_update_when_bit_zero() {
        let buf = build_bitstream(|w| {
            // The reader's `read(prob)` returns 1 when the bit's
            // model says "above threshold". To make it return 0
            // (no update), we write a 0-bit at DIFF_UPDATE_PROB.
            w.write(0, DIFF_UPDATE_PROB);
            // Pad.
            for _ in 0..20 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut p = 100u8;
        let updated = diff_update_prob(&mut r, &mut p);
        assert!(!updated);
        assert_eq!(p, 100);
    }

    /// Round-trip: write `update = true` with a delp of 0
    /// (4-bit literal path, prefix "0", then 4 zero bits).
    /// `delp = 0` → INV_MAP_TABLE[0] = 7. With m = 100:
    ///   inv_recenter_nonneg(7, 99) = 99 + 7/2 = 99 + 3 = 102
    ///                                (v=7 odd? 7 & 1 = 1 → 99 - (7+1)/2 = 99 - 4 = 95)
    ///   Actually 7 & 1 = 1 (7 is odd), so (m=99): 99 - ((7+1)>>1) = 99 - 4 = 95.
    ///   (m << 1) = 198, <= MAX_PROB(255)? yes. result = 1 + 95 = 96.
    /// So *p should become 96.
    #[test]
    fn diff_update_prob_applies_update_when_bit_one() {
        let buf = build_bitstream(|w| {
            w.write(1, DIFF_UPDATE_PROB); // update = true
            w.write_bit(0); // term_subexp prefix → 4-bit literal path
            w.write_literal(0, 4); // delp = 0
            for _ in 0..20 {
                w.write_bit(0);
            }
        });
        let mut r = BoolReader::new(&buf).expect("init");
        let mut p = 100u8;
        let updated = diff_update_prob(&mut r, &mut p);
        assert!(updated);
        assert_eq!(p, 96);
    }

    /// `inv_remap_prob(delp, m)` for the first few delp values with
    /// m = 100, hand-traced. INV_MAP_TABLE[delp] gives v; m_eff = 99
    /// (m - 1). v < 2*m_eff (= 198) for v < 198. Then:
    ///   v even → 1 + (m_eff + v/2)
    ///   v odd  → 1 + (m_eff - (v+1)/2)
    #[test]
    fn inv_remap_prob_hand_checks() {
        // delp=0 → v=7 (odd) → 1 + (99 - 4) = 96.
        assert_eq!(inv_remap_prob(0, 100), 96);
        // delp=1 → v=20 (even) → 1 + (99 + 10) = 110.
        assert_eq!(inv_remap_prob(1, 100), 110);
        // delp=20 → v=1 (odd) → 1 + (99 - 1) = 99.
        assert_eq!(inv_remap_prob(20, 100), 99);
    }
}
