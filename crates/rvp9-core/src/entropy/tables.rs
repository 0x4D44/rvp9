//! Shared entropy-coder tables.
//!
//! `VPX_NORM[256]` is the renormalisation table used by both the arithmetic
//! writer ([`super::BoolWriter`]) and reader ([`super::BoolReader`]). The
//! value at index `r` is the number of leading-zero bits that must be
//! shifted out of the range register `r` to bring it back into the
//! `[128, 255]` interval. Mirrors libvpx `vpx_dsp/prob.c:13-24` verbatim.

/// The renormalisation table. See module-level docs for semantics.
///
/// Sourced verbatim from `third_party/libvpx/vpx_dsp/prob.c:13-24`.
#[rustfmt::skip]
pub static VPX_NORM: [u8; 256] = [
    0, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: the leading-zero count is consistent with the table.
    /// For r >= 128 the table is 0 (no normalisation needed).
    /// For r in [64, 127] it's 1. For r in [32, 63] it's 2. Etc.
    #[test]
    fn vpx_norm_matches_leading_zeros() {
        for r in 1u32..=255 {
            let expected = (r as u8).leading_zeros() as u8;
            assert_eq!(
                VPX_NORM[r as usize], expected,
                "VPX_NORM[{r}] = {} but leading_zeros = {expected}",
                VPX_NORM[r as usize]
            );
        }
    }

    /// VPX_NORM[0] = 0 by convention (the table is keyed on the byte
    /// value of `range`; `range = 0` is never produced by a well-formed
    /// codec, so this slot is filler).
    #[test]
    fn vpx_norm_zero_is_filler() {
        assert_eq!(VPX_NORM[0], 0);
    }

    /// The transition points must be at exact powers of two.
    #[test]
    fn vpx_norm_transitions() {
        assert_eq!(VPX_NORM[128], 0);
        assert_eq!(VPX_NORM[127], 1);
        assert_eq!(VPX_NORM[64], 1);
        assert_eq!(VPX_NORM[63], 2);
        assert_eq!(VPX_NORM[1], 7);
        assert_eq!(VPX_NORM[255], 0);
    }
}
