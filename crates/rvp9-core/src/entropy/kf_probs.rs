//! Keyframe-only mode probability tables (Task 2b.2).
//!
//! These tables are NOT carried in the `FRAME_CONTEXT` struct; they are
//! plain `static const` tables in libvpx that the keyframe entropy
//! decoder consults directly. Mirrors libvpx `vp9/common/vp9_entropymode.c`
//! lines 16-160 and 182-207.
//!
//! ## Tables
//!
//! - [`VP9_KF_Y_MODE_PROB`] — luma intra-mode probabilities for keyframes,
//!   conditioned on the `(above, left)` neighbour-mode pair.
//!   Shape: `[INTRA_MODES][INTRA_MODES][INTRA_MODES - 1]`
//!   = `10 × 10 × 9 = 900` entries.
//! - [`VP9_KF_UV_MODE_PROB`] — chroma intra-mode probabilities for
//!   keyframes, conditioned on the Y mode. Shape:
//!   `[INTRA_MODES][INTRA_MODES - 1]` = `10 × 9 = 90` entries.
//! - [`VP9_KF_PARTITION_PROBS`] — partition-tree probabilities for
//!   keyframes. Shape: `[PARTITION_CONTEXTS][PARTITION_TYPES - 1]`
//!   = `16 × 3 = 48` entries.

use super::default_probs::{INTRA_MODES, PARTITION_CONTEXTS, PARTITION_TYPES};

/// Keyframe luma intra-mode probabilities, indexed by
/// `[above_mode][left_mode][branch]`. 900 entries verbatim from
/// libvpx `vp9_kf_y_mode_prob` (`vp9_entropymode.c:16-147`).
#[rustfmt::skip]
pub static VP9_KF_Y_MODE_PROB: [[[u8; INTRA_MODES - 1]; INTRA_MODES]; INTRA_MODES] = [
    // above = dc
    [
        [137, 30, 42, 148, 151, 207, 70, 52, 91],   // left = dc
        [92, 45, 102, 136, 116, 180, 74, 90, 100],  // left = v
        [73, 32, 19, 187, 222, 215, 46, 34, 100],   // left = h
        [91, 30, 32, 116, 121, 186, 93, 86, 94],    // left = d45
        [72, 35, 36, 149, 68, 206, 68, 63, 105],    // left = d135
        [73, 31, 28, 138, 57, 124, 55, 122, 151],   // left = d117
        [67, 23, 21, 140, 126, 197, 40, 37, 171],   // left = d153
        [86, 27, 28, 128, 154, 212, 45, 43, 53],    // left = d207
        [74, 32, 27, 107, 86, 160, 63, 134, 102],   // left = d63
        [59, 67, 44, 140, 161, 202, 78, 67, 119],   // left = tm
    ],
    // above = v
    [
        [63, 36, 126, 146, 123, 158, 60, 90, 96],
        [43, 46, 168, 134, 107, 128, 69, 142, 92],
        [44, 29, 68, 159, 201, 177, 50, 57, 77],
        [58, 38, 76, 114, 97, 172, 78, 133, 92],
        [46, 41, 76, 140, 63, 184, 69, 112, 57],
        [38, 32, 85, 140, 46, 112, 54, 151, 133],
        [39, 27, 61, 131, 110, 175, 44, 75, 136],
        [52, 30, 74, 113, 130, 175, 51, 64, 58],
        [47, 35, 80, 100, 74, 143, 64, 163, 74],
        [36, 61, 116, 114, 128, 162, 80, 125, 82],
    ],
    // above = h
    [
        [82, 26, 26, 171, 208, 204, 44, 32, 105],
        [55, 44, 68, 166, 179, 192, 57, 57, 108],
        [42, 26, 11, 199, 241, 228, 23, 15, 85],
        [68, 42, 19, 131, 160, 199, 55, 52, 83],
        [58, 50, 25, 139, 115, 232, 39, 52, 118],
        [50, 35, 33, 153, 104, 162, 64, 59, 131],
        [44, 24, 16, 150, 177, 202, 33, 19, 156],
        [55, 27, 12, 153, 203, 218, 26, 27, 49],
        [53, 49, 21, 110, 116, 168, 59, 80, 76],
        [38, 72, 19, 168, 203, 212, 50, 50, 107],
    ],
    // above = d45
    [
        [103, 26, 36, 129, 132, 201, 83, 80, 93],
        [59, 38, 83, 112, 103, 162, 98, 136, 90],
        [62, 30, 23, 158, 200, 207, 59, 57, 50],
        [67, 30, 29, 84, 86, 191, 102, 91, 59],
        [60, 32, 33, 112, 71, 220, 64, 89, 104],
        [53, 26, 34, 130, 56, 149, 84, 120, 103],
        [53, 21, 23, 133, 109, 210, 56, 77, 172],
        [77, 19, 29, 112, 142, 228, 55, 66, 36],
        [61, 29, 29, 93, 97, 165, 83, 175, 162],
        [47, 47, 43, 114, 137, 181, 100, 99, 95],
    ],
    // above = d135
    [
        [69, 23, 29, 128, 83, 199, 46, 44, 101],
        [53, 40, 55, 139, 69, 183, 61, 80, 110],
        [40, 29, 19, 161, 180, 207, 43, 24, 91],
        [60, 34, 19, 105, 61, 198, 53, 64, 89],
        [52, 31, 22, 158, 40, 209, 58, 62, 89],
        [44, 31, 29, 147, 46, 158, 56, 102, 198],
        [35, 19, 12, 135, 87, 209, 41, 45, 167],
        [55, 25, 21, 118, 95, 215, 38, 39, 66],
        [51, 38, 25, 113, 58, 164, 70, 93, 97],
        [47, 54, 34, 146, 108, 203, 72, 103, 151],
    ],
    // above = d117
    [
        [64, 19, 37, 156, 66, 138, 49, 95, 133],
        [46, 27, 80, 150, 55, 124, 55, 121, 135],
        [36, 23, 27, 165, 149, 166, 54, 64, 118],
        [53, 21, 36, 131, 63, 163, 60, 109, 81],
        [40, 26, 35, 154, 40, 185, 51, 97, 123],
        [35, 19, 34, 179, 19, 97, 48, 129, 124],
        [36, 20, 26, 136, 62, 164, 33, 77, 154],
        [45, 18, 32, 130, 90, 157, 40, 79, 91],
        [45, 26, 28, 129, 45, 129, 49, 147, 123],
        [38, 44, 51, 136, 74, 162, 57, 97, 121],
    ],
    // above = d153
    [
        [75, 17, 22, 136, 138, 185, 32, 34, 166],
        [56, 39, 58, 133, 117, 173, 48, 53, 187],
        [35, 21, 12, 161, 212, 207, 20, 23, 145],
        [56, 29, 19, 117, 109, 181, 55, 68, 112],
        [47, 29, 17, 153, 64, 220, 59, 51, 114],
        [46, 16, 24, 136, 76, 147, 41, 64, 172],
        [34, 17, 11, 108, 152, 187, 13, 15, 209],
        [51, 24, 14, 115, 133, 209, 32, 26, 104],
        [55, 30, 18, 122, 79, 179, 44, 88, 116],
        [37, 49, 25, 129, 168, 164, 41, 54, 148],
    ],
    // above = d207
    [
        [82, 22, 32, 127, 143, 213, 39, 41, 70],
        [62, 44, 61, 123, 105, 189, 48, 57, 64],
        [47, 25, 17, 175, 222, 220, 24, 30, 86],
        [68, 36, 17, 106, 102, 206, 59, 74, 74],
        [57, 39, 23, 151, 68, 216, 55, 63, 58],
        [49, 30, 35, 141, 70, 168, 82, 40, 115],
        [51, 25, 15, 136, 129, 202, 38, 35, 139],
        [68, 26, 16, 111, 141, 215, 29, 28, 28],
        [59, 39, 19, 114, 75, 180, 77, 104, 42],
        [40, 61, 26, 126, 152, 206, 61, 59, 93],
    ],
    // above = d63
    [
        [78, 23, 39, 111, 117, 170, 74, 124, 94],
        [48, 34, 86, 101, 92, 146, 78, 179, 134],
        [47, 22, 24, 138, 187, 178, 68, 69, 59],
        [56, 25, 33, 105, 112, 187, 95, 177, 129],
        [48, 31, 27, 114, 63, 183, 82, 116, 56],
        [43, 28, 37, 121, 63, 123, 61, 192, 169],
        [42, 17, 24, 109, 97, 177, 56, 76, 122],
        [58, 18, 28, 105, 139, 182, 70, 92, 63],
        [46, 23, 32, 74, 86, 150, 67, 183, 88],
        [36, 38, 48, 92, 122, 165, 88, 137, 91],
    ],
    // above = tm
    [
        [65, 70, 60, 155, 159, 199, 61, 60, 81],
        [44, 78, 115, 132, 119, 173, 71, 112, 93],
        [39, 38, 21, 184, 227, 206, 42, 32, 64],
        [58, 47, 36, 124, 137, 193, 80, 82, 78],
        [49, 50, 35, 144, 95, 205, 63, 78, 59],
        [41, 53, 52, 148, 71, 142, 65, 128, 51],
        [40, 36, 28, 143, 143, 202, 40, 55, 137],
        [52, 34, 29, 129, 183, 227, 42, 35, 43],
        [42, 44, 44, 104, 105, 164, 64, 130, 80],
        [43, 81, 53, 140, 169, 204, 68, 84, 72],
    ],
];

/// Keyframe chroma intra-mode probabilities, indexed by `[y_mode][branch]`.
/// 90 entries verbatim from libvpx `vp9_kf_uv_mode_prob`
/// (`vp9_entropymode.c:149-160`).
#[rustfmt::skip]
pub static VP9_KF_UV_MODE_PROB: [[u8; INTRA_MODES - 1]; INTRA_MODES] = [
    [144, 11, 54, 157, 195, 130, 46, 58, 108],   // y = dc
    [118, 15, 123, 148, 131, 101, 44, 93, 131],  // y = v
    [113, 12, 23, 188, 226, 142, 26, 32, 125],   // y = h
    [120, 11, 50, 123, 163, 135, 64, 77, 103],   // y = d45
    [113, 9, 36, 155, 111, 157, 32, 44, 161],    // y = d135
    [116, 9, 55, 176, 76, 96, 37, 61, 149],      // y = d117
    [115, 9, 28, 141, 161, 167, 21, 25, 193],    // y = d153
    [120, 12, 32, 145, 195, 142, 32, 38, 86],    // y = d207
    [116, 12, 64, 120, 140, 125, 49, 115, 121],  // y = d63
    [102, 19, 66, 162, 182, 122, 35, 59, 128],   // y = tm
];

/// Keyframe partition-tree probabilities. 48 entries verbatim from libvpx
/// `vp9_kf_partition_probs` (`vp9_entropymode.c:182-207`).
#[rustfmt::skip]
pub static VP9_KF_PARTITION_PROBS:
    [[u8; PARTITION_TYPES - 1]; PARTITION_CONTEXTS] = [
    // 8x8 -> 4x4
    [158, 97, 94],
    [93, 24, 99],
    [85, 119, 44],
    [62, 59, 67],
    // 16x16 -> 8x8
    [149, 53, 53],
    [94, 20, 48],
    [83, 53, 24],
    [52, 18, 18],
    // 32x32 -> 16x16
    [150, 40, 39],
    [78, 12, 26],
    [67, 33, 11],
    [24, 7, 5],
    // 64x64 -> 32x32
    [174, 35, 49],
    [68, 11, 27],
    [57, 15, 9],
    [12, 3, 3],
];

#[cfg(test)]
mod tests {
    use super::*;

    /// All keyframe-mode probs are non-zero — libvpx never ships a 0.
    #[test]
    fn no_zero_probs() {
        for plane in VP9_KF_Y_MODE_PROB.iter() {
            for row in plane.iter() {
                for &p in row.iter() {
                    assert!(p > 0, "VP9_KF_Y_MODE_PROB contains 0");
                }
            }
        }
        for row in VP9_KF_UV_MODE_PROB.iter() {
            for &p in row.iter() {
                assert!(p > 0, "VP9_KF_UV_MODE_PROB contains 0");
            }
        }
        for row in VP9_KF_PARTITION_PROBS.iter() {
            for &p in row.iter() {
                assert!(p > 0, "VP9_KF_PARTITION_PROBS contains 0");
            }
        }
    }

    #[test]
    fn table_shapes_match_libvpx() {
        assert_eq!(VP9_KF_Y_MODE_PROB.len(), INTRA_MODES);
        assert_eq!(VP9_KF_Y_MODE_PROB[0].len(), INTRA_MODES);
        assert_eq!(VP9_KF_Y_MODE_PROB[0][0].len(), INTRA_MODES - 1);
        assert_eq!(VP9_KF_UV_MODE_PROB.len(), INTRA_MODES);
        assert_eq!(VP9_KF_UV_MODE_PROB[0].len(), INTRA_MODES - 1);
        assert_eq!(VP9_KF_PARTITION_PROBS.len(), PARTITION_CONTEXTS);
        assert_eq!(VP9_KF_PARTITION_PROBS[0].len(), PARTITION_TYPES - 1);

        // Total entry counts.
        let y_entries = INTRA_MODES * INTRA_MODES * (INTRA_MODES - 1);
        assert_eq!(y_entries, 900);
        let uv_entries = INTRA_MODES * (INTRA_MODES - 1);
        assert_eq!(uv_entries, 90);
        let part_entries = PARTITION_CONTEXTS * (PARTITION_TYPES - 1);
        assert_eq!(part_entries, 48);
    }

    /// Spec-quoted spot checks against libvpx's data.
    #[test]
    fn spec_spot_checks() {
        // above=dc, left=dc, first branch (libvpx :19).
        assert_eq!(VP9_KF_Y_MODE_PROB[0][0][0], 137);
        // above=tm, left=tm, last row (libvpx :145).
        assert_eq!(
            VP9_KF_Y_MODE_PROB[9][9],
            [43, 81, 53, 140, 169, 204, 68, 84, 72]
        );
        // UV y=dc, first branch (libvpx :150).
        assert_eq!(VP9_KF_UV_MODE_PROB[0][0], 144);
        // UV y=tm, last row (libvpx :159).
        assert_eq!(
            VP9_KF_UV_MODE_PROB[9],
            [102, 19, 66, 162, 182, 122, 35, 59, 128]
        );
        // Partition 64x64 -> 32x32, a/l both split (libvpx :206).
        assert_eq!(VP9_KF_PARTITION_PROBS[15], [12, 3, 3]);
    }
}
