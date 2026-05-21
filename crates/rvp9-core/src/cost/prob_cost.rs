//! Per-probability bit-cost lookup table.
//!
//! Mirrors libvpx `vp9/encoder/vp9_cost.{c,h}`. The 256-entry table
//! [`VP9_PROB_COST`] maps a probability `p ∈ [0, 256]` to the bit-cost
//! of encoding a `0` under that probability, in fixed-point units
//! (`VP9_PROB_COST_SHIFT = 9`, i.e. `512` = one bit). Tables are shipped
//! verbatim from libvpx — they were originally computed as
//! `round(-log2(p / 256.0) * (1 << 9))`. We keep them as a `static`
//! rather than computing at build time because the libvpx values are
//! the spec the rest of the codec is bit-exact against.

/// The factor that scales costs in bits to the [`VP9_PROB_COST`] fixed-point
/// units. Mirrors libvpx `VP9_PROB_COST_SHIFT` (`vp9_cost.h:24`).
pub const VP9_PROB_COST_SHIFT: u32 = 9;

/// `vp9_prob_cost[p]` is `round(-log2(p / 256.0) * 512)` for
/// `p ∈ [1, 255]`, with the `p = 0` slot zero-padded to a sentinel
/// value matching `p = 1`. Verbatim from libvpx
/// `vp9_cost.c:16-37`.
#[rustfmt::skip]
pub static VP9_PROB_COST: [u16; 256] = [
    4096, 4096, 3584, 3284, 3072, 2907, 2772, 2659, 2560, 2473, 2395, 2325, 2260,
    2201, 2147, 2096, 2048, 2003, 1961, 1921, 1883, 1847, 1813, 1780, 1748, 1718,
    1689, 1661, 1635, 1609, 1584, 1559, 1536, 1513, 1491, 1470, 1449, 1429, 1409,
    1390, 1371, 1353, 1335, 1318, 1301, 1284, 1268, 1252, 1236, 1221, 1206, 1192,
    1177, 1163, 1149, 1136, 1123, 1110, 1097, 1084, 1072, 1059, 1047, 1036, 1024,
    1013, 1001, 990,  979,  968,  958,  947,  937,  927,  917,  907,  897,  887,
    878,  868,  859,  850,  841,  832,  823,  814,  806,  797,  789,  780,  772,
    764,  756,  748,  740,  732,  724,  717,  709,  702,  694,  687,  680,  673,
    665,  658,  651,  644,  637,  631,  624,  617,  611,  604,  598,  591,  585,
    578,  572,  566,  560,  554,  547,  541,  535,  530,  524,  518,  512,  506,
    501,  495,  489,  484,  478,  473,  467,  462,  456,  451,  446,  441,  435,
    430,  425,  420,  415,  410,  405,  400,  395,  390,  385,  380,  375,  371,
    366,  361,  356,  352,  347,  343,  338,  333,  329,  324,  320,  316,  311,
    307,  302,  298,  294,  289,  285,  281,  277,  273,  268,  264,  260,  256,
    252,  248,  244,  240,  236,  232,  228,  224,  220,  216,  212,  209,  205,
    201,  197,  194,  190,  186,  182,  179,  175,  171,  168,  164,  161,  157,
    153,  150,  146,  143,  139,  136,  132,  129,  125,  122,  119,  115,  112,
    109,  105,  102,  99,   95,   92,   89,   86,   82,   79,   76,   73,   70,
    66,   63,   60,   57,   54,   51,   48,   45,   42,   38,   35,   32,   29,
    26,   23,   20,   18,   15,   12,   9,    6,    3,
];

/// Cost of encoding a `0` under probability `prob`. Mirrors libvpx
/// `vp9_cost_zero` (`vp9_cost.h:26`).
#[inline]
#[must_use]
pub fn vp9_cost_zero(prob: u8) -> u32 {
    u32::from(VP9_PROB_COST[prob as usize])
}

/// Cost of encoding a `1` under probability `prob`. Mirrors libvpx
/// `vp9_cost_one` (`vp9_cost.h:28`).
///
/// Equivalent to `vp9_cost_zero(256 - prob)`; the `prob == 0` slot of
/// the table is the sentinel value `4096`.
#[inline]
#[must_use]
pub fn vp9_cost_one(prob: u8) -> u32 {
    u32::from(VP9_PROB_COST[(256 - prob as u32) as usize])
}

/// Cost of encoding a single bit `bit ∈ {0, 1}` under probability
/// `prob`. Mirrors libvpx `vp9_cost_bit` (`vp9_cost.h:30`).
#[inline]
#[must_use]
pub fn vp9_cost_bit(prob: u8, bit: bool) -> u32 {
    if bit { vp9_cost_one(prob) } else { vp9_cost_zero(prob) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: probability 128 is "maximum entropy", so a single bit costs
    /// exactly 1 bit = `1 << VP9_PROB_COST_SHIFT = 512`.
    #[test]
    fn cost_at_half_is_one_bit() {
        assert_eq!(vp9_cost_zero(128), 512);
        assert_eq!(vp9_cost_one(128), 512);
        assert_eq!(vp9_cost_bit(128, false), 512);
        assert_eq!(vp9_cost_bit(128, true), 512);
    }

    /// Sanity: extremely high probability of `0` makes encoding a `0`
    /// nearly free and encoding a `1` very expensive.
    #[test]
    fn high_prob_zero_is_cheap_one_is_expensive() {
        let cost0 = vp9_cost_zero(255);
        let cost1 = vp9_cost_one(255);
        assert!(cost0 < 10, "cost_zero(255) = {cost0}");
        assert!(cost1 > 3000, "cost_one(255) = {cost1}");
    }

    /// Sanity: extremely low probability of `0` makes encoding a `1`
    /// nearly free and encoding a `0` very expensive.
    #[test]
    fn low_prob_zero_is_expensive_one_is_cheap() {
        let cost0 = vp9_cost_zero(1);
        let cost1 = vp9_cost_one(1);
        assert!(cost0 > 3000, "cost_zero(1) = {cost0}");
        assert!(cost1 < 10, "cost_one(1) = {cost1}");
    }

    /// Sanity: the table is monotonically non-increasing in `prob` —
    /// higher prob of `0` means lower cost to encode a `0`.
    #[test]
    fn cost_zero_monotone_in_prob() {
        for p in 1..255u8 {
            let here = vp9_cost_zero(p);
            let next = vp9_cost_zero(p + 1);
            assert!(next <= here, "cost_zero non-monotone at p={p}: {here} -> {next}");
        }
    }

    /// `vp9_cost_one(prob) == vp9_cost_zero(256 - prob)` for `prob > 0`.
    #[test]
    fn cost_one_is_symmetric_to_cost_zero() {
        for p in 1..=255u8 {
            assert_eq!(
                vp9_cost_one(p),
                vp9_cost_zero(((256 - p as u32) % 256) as u8),
                "asymmetric at p={p}"
            );
        }
    }

    /// Table size + sentinel slot must match libvpx.
    #[test]
    fn table_shape_matches_libvpx() {
        assert_eq!(VP9_PROB_COST.len(), 256);
        assert_eq!(VP9_PROB_COST[0], 4096); // sentinel for prob=0
        assert_eq!(VP9_PROB_COST[1], 4096); // log2(1/256) ≈ -8 → 8 * 512 = 4096
    }
}
