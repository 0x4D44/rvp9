//! AArch64 NEON V, H, and full-neighbour DC intra predictors.
//!
//! These mirror pinned libvpx's fixed-width NEON predictors. VP9 only calls
//! these slots for square blocks with `bs` in `{4, 8, 16, 32}`.

use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
unsafe fn store_constant_block_neon_inner(dst: *mut u8, stride: usize, bs: usize, value: u8) {
    match bs {
        4 => {
            let row = u32::from_ne_bytes([value; 4]);
            for r in 0..4 {
                core::ptr::write_unaligned(dst.add(r * stride).cast::<u32>(), row);
            }
        }
        8 => {
            let row = vdup_n_u8(value);
            for r in 0..8 {
                vst1_u8(dst.add(r * stride), row);
            }
        }
        16 => {
            let row = vdupq_n_u8(value);
            for r in 0..16 {
                vst1q_u8(dst.add(r * stride), row);
            }
        }
        32 => {
            let row = vdupq_n_u8(value);
            for r in 0..32 {
                let dst_row = dst.add(r * stride);
                vst1q_u8(dst_row, row);
                vst1q_u8(dst_row.add(16), row);
            }
        }
        _ => unreachable!("validated VP9 intra block size"),
    }
}

#[target_feature(enable = "neon")]
unsafe fn v_predictor_neon_inner(dst: *mut u8, stride: usize, bs: usize, above: *const u8) {
    match bs {
        4 => {
            let row = core::ptr::read_unaligned(above.cast::<u32>());
            for r in 0..4 {
                core::ptr::write_unaligned(dst.add(r * stride).cast::<u32>(), row);
            }
        }
        8 => {
            let row = vld1_u8(above);
            for r in 0..8 {
                vst1_u8(dst.add(r * stride), row);
            }
        }
        16 => {
            let row = vld1q_u8(above);
            for r in 0..16 {
                vst1q_u8(dst.add(r * stride), row);
            }
        }
        32 => {
            let row_0 = vld1q_u8(above);
            let row_1 = vld1q_u8(above.add(16));
            for r in 0..32 {
                let dst_row = dst.add(r * stride);
                vst1q_u8(dst_row, row_0);
                vst1q_u8(dst_row.add(16), row_1);
            }
        }
        _ => unreachable!("validated VP9 intra block size"),
    }
}

/// Broadcast the above row through a VP9 intra block using fixed-width stores.
pub fn v_predictor_neon(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    crate::dsp::validate_intra_predictor_layout(dst.len(), stride, bs, above.len())
        .unwrap_or_else(|reason| panic!("v_predictor_neon: {reason}"));

    // SAFETY: dispatch installs this wrapper only on NEON-capable AArch64
    // hosts. The layout check proves every fixed-width load and store is in
    // bounds, and `bs` is one of the four VP9 transform sizes.
    unsafe { v_predictor_neon_inner(dst.as_mut_ptr(), stride, bs, above.as_ptr()) }
}

#[target_feature(enable = "neon")]
unsafe fn h_predictor_neon_inner(dst: *mut u8, stride: usize, bs: usize, left: *const u8) {
    match bs {
        4 => {
            for r in 0..4 {
                let value = *left.add(r);
                let row = u32::from_ne_bytes([value; 4]);
                core::ptr::write_unaligned(dst.add(r * stride).cast::<u32>(), row);
            }
        }
        8 => {
            for r in 0..8 {
                vst1_u8(dst.add(r * stride), vdup_n_u8(*left.add(r)));
            }
        }
        16 => {
            for r in 0..16 {
                vst1q_u8(dst.add(r * stride), vdupq_n_u8(*left.add(r)));
            }
        }
        32 => {
            for r in 0..32 {
                let row = vdupq_n_u8(*left.add(r));
                let dst_row = dst.add(r * stride);
                vst1q_u8(dst_row, row);
                vst1q_u8(dst_row.add(16), row);
            }
        }
        _ => unreachable!("validated VP9 intra block size"),
    }
}

/// Broadcast each left sample across its VP9 intra-block row using NEON.
pub fn h_predictor_neon(dst: &mut [u8], stride: usize, bs: usize, left: &[u8]) {
    crate::dsp::validate_intra_predictor_layout(dst.len(), stride, bs, left.len())
        .unwrap_or_else(|reason| panic!("h_predictor_neon: {reason}"));

    // SAFETY: see `v_predictor_neon`; the same checked geometry applies and
    // `left` contains one byte for every destination row.
    unsafe { h_predictor_neon_inner(dst.as_mut_ptr(), stride, bs, left.as_ptr()) }
}

/// Fill a VP9 intra block with the rounded mean of its above and left edges.
pub fn dc_predict_full_neon(dst: &mut [u8], stride: usize, bs: usize, above: &[u8], left: &[u8]) {
    crate::dsp::validate_intra_predictor_layout(dst.len(), stride, bs, above.len())
        .unwrap_or_else(|reason| panic!("dc_predict_full_neon: {reason}"));
    assert!(
        left.len() >= bs,
        "dc_predict_full_neon: left edge is shorter than block size"
    );

    let sum: u32 = above[..bs]
        .iter()
        .chain(&left[..bs])
        .map(|&value| u32::from(value))
        .sum();
    let dc = ((sum + bs as u32) / (2 * bs) as u32) as u8;

    // SAFETY: dispatch installs this wrapper only on NEON-capable AArch64
    // hosts. The checks above prove every fixed-width store is in bounds.
    unsafe { store_constant_block_neon_inner(dst.as_mut_ptr(), stride, bs, dc) }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::intra_pred::{PredCtx, dc_predictor, h_predictor, v_predictor};

    fn edge(seed: u8, bs: usize) -> Vec<u8> {
        (0..bs)
            .map(|index| seed.wrapping_add((index as u8).wrapping_mul(37)))
            .collect()
    }

    #[test]
    fn differential_all_sizes_and_strides() {
        for bs in [4, 8, 16, 32] {
            let above = edge(11, bs);
            let left = edge(193, bs);
            for jitter in 0..9 {
                let stride = bs + jitter;
                let len = (bs - 1) * stride + bs;
                let initial: Vec<u8> = (0..len).map(|index| index as u8).collect();

                let mut expected = initial.clone();
                let mut actual = initial.clone();
                v_predictor(
                    &mut expected,
                    stride,
                    bs,
                    PredCtx {
                        above: &above,
                        left: &[],
                        above_left: 0,
                    },
                );
                v_predictor_neon(&mut actual, stride, bs, &above);
                assert_eq!(actual, expected, "V mismatch at bs={bs}, jitter={jitter}");

                expected.clone_from(&initial);
                actual.clone_from(&initial);
                h_predictor(
                    &mut expected,
                    stride,
                    bs,
                    PredCtx {
                        above: &[],
                        left: &left,
                        above_left: 0,
                    },
                );
                h_predictor_neon(&mut actual, stride, bs, &left);
                assert_eq!(actual, expected, "H mismatch at bs={bs}, jitter={jitter}");

                expected.clone_from(&initial);
                actual.clone_from(&initial);
                dc_predictor(
                    &mut expected,
                    stride,
                    bs,
                    PredCtx {
                        above: &above,
                        left: &left,
                        above_left: 0,
                    },
                );
                dc_predict_full_neon(&mut actual, stride, bs, &above, &left);
                assert_eq!(actual, expected, "DC mismatch at bs={bs}, jitter={jitter}");
            }
        }
    }

    #[test]
    fn dispatch_routes_neon_and_scalar_caps() {
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::AArch64Caps;

        let neon = SimdDispatch::build(AArch64Caps {
            neon: true,
            dotprod: false,
        });
        let scalar = SimdDispatch::build(AArch64Caps::default());
        let neon_v: fn(&mut [u8], usize, usize, &[u8]) = v_predictor_neon;
        let neon_h: fn(&mut [u8], usize, usize, &[u8]) = h_predictor_neon;
        let neon_dc: fn(&mut [u8], usize, usize, &[u8], &[u8]) = dc_predict_full_neon;

        assert_eq!(neon.v_predictor() as usize, neon_v as usize);
        assert_eq!(neon.h_predictor() as usize, neon_h as usize);
        assert_eq!(neon.dc_predict_full() as usize, neon_dc as usize);
        assert_ne!(scalar.v_predictor() as usize, neon_v as usize);
        assert_ne!(scalar.h_predictor() as usize, neon_h as usize);
        assert_ne!(scalar.dc_predict_full() as usize, neon_dc as usize);
    }
}
