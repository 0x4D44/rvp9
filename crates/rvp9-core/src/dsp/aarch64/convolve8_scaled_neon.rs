//! AArch64 NEON scaled 8-tap interpolation.
//!
//! The scaled path is the full two-dimensional VP9 resampler used when the
//! reference and destination frames have different dimensions.  Unlike the
//! unscaled path, each output column and row selects its own filter phase.
//! This follows pinned libvpx's `vpx_scaled_2d_neon` structure while keeping
//! rvp9's scalar API convention: `src` is already shifted to the first tap.

use core::arch::aarch64::*;

use crate::inter::filter::{SUBPEL_MASK, SUBPEL_SHIFTS};

const FILTER_BITS: i32 = 7;
const FILTER_ROUNDING: i32 = 1 << (FILTER_BITS - 1);
const FILTER_TAPS: usize = 8;
const INTER_STRIDE: usize = 64;
const MAX_INTERMEDIATE_HEIGHT: usize = 135;

std::thread_local! {
    static SCALED_INTERMEDIATE: std::cell::RefCell<[u8; INTER_STRIDE * MAX_INTERMEDIATE_HEIGHT]> =
        const { std::cell::RefCell::new([0u8; INTER_STRIDE * MAX_INTERMEDIATE_HEIGHT]) };
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn filter_eight(samples: [uint8x8_t; 8], filter: &[i16; 8]) -> uint8x8_t {
    let mut sum_lo = vdupq_n_s32(FILTER_ROUNDING);
    let mut sum_hi = vdupq_n_s32(FILTER_ROUNDING);
    for tap in 0..FILTER_TAPS {
        let widened = vreinterpretq_s16_u16(vmovl_u8(samples[tap]));
        sum_lo = vmlal_n_s16(sum_lo, vget_low_s16(widened), filter[tap]);
        sum_hi = vmlal_n_s16(sum_hi, vget_high_s16(widened), filter[tap]);
    }
    let rounded = vcombine_s16(
        vshrn_n_s32::<FILTER_BITS>(sum_lo),
        vshrn_n_s32::<FILTER_BITS>(sum_hi),
    );
    vqmovun_s16(rounded)
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn filter_four(samples: [uint8x8_t; 8], filter: &[i16; 8]) -> uint8x8_t {
    let mut sum = vdupq_n_s32(FILTER_ROUNDING);
    for tap in 0..FILTER_TAPS {
        let widened = vreinterpretq_s16_u16(vmovl_u8(samples[tap]));
        sum = vmlal_n_s16(sum, vget_low_s16(widened), filter[tap]);
    }
    let rounded = vshrn_n_s32::<FILTER_BITS>(sum);
    vqmovun_s16(vcombine_s16(rounded, rounded))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn transpose_8x8(mut rows: [uint8x8_t; 8]) -> [uint8x8_t; 8] {
    let t0 = vtrn_u8(rows[0], rows[1]);
    let t1 = vtrn_u8(rows[2], rows[3]);
    let t2 = vtrn_u8(rows[4], rows[5]);
    let t3 = vtrn_u8(rows[6], rows[7]);

    let u0 = vtrn_u16(vreinterpret_u16_u8(t0.0), vreinterpret_u16_u8(t1.0));
    let u1 = vtrn_u16(vreinterpret_u16_u8(t0.1), vreinterpret_u16_u8(t1.1));
    let u2 = vtrn_u16(vreinterpret_u16_u8(t2.0), vreinterpret_u16_u8(t3.0));
    let u3 = vtrn_u16(vreinterpret_u16_u8(t2.1), vreinterpret_u16_u8(t3.1));

    let v0 = vtrn_u32(vreinterpret_u32_u16(u0.0), vreinterpret_u32_u16(u2.0));
    let v1 = vtrn_u32(vreinterpret_u32_u16(u1.0), vreinterpret_u32_u16(u3.0));
    let v2 = vtrn_u32(vreinterpret_u32_u16(u0.1), vreinterpret_u32_u16(u2.1));
    let v3 = vtrn_u32(vreinterpret_u32_u16(u1.1), vreinterpret_u32_u16(u3.1));

    rows[0] = vreinterpret_u8_u32(v0.0);
    rows[1] = vreinterpret_u8_u32(v1.0);
    rows[2] = vreinterpret_u8_u32(v2.0);
    rows[3] = vreinterpret_u8_u32(v3.0);
    rows[4] = vreinterpret_u8_u32(v0.1);
    rows[5] = vreinterpret_u8_u32(v1.1);
    rows[6] = vreinterpret_u8_u32(v2.1);
    rows[7] = vreinterpret_u8_u32(v3.1);
    rows
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn transpose_4x8(rows: [uint8x8_t; 4]) -> [uint8x8_t; 8] {
    let t0 = vtrn_u8(rows[0], rows[1]);
    let t1 = vtrn_u8(rows[2], rows[3]);
    let u0 = vtrn_u16(vreinterpret_u16_u8(t0.0), vreinterpret_u16_u8(t1.0));
    let u1 = vtrn_u16(vreinterpret_u16_u8(t0.1), vreinterpret_u16_u8(t1.1));

    // The first four bytes of each vector are one transposed column.  The
    // high halves are moved down for taps 4..7; `filter_four` only consumes
    // the low four lanes.
    [
        vreinterpret_u8_u16(u0.0),
        vreinterpret_u8_u16(u1.0),
        vreinterpret_u8_u16(u0.1),
        vreinterpret_u8_u16(u1.1),
        vext_u8::<4>(vreinterpret_u8_u16(u0.0), vreinterpret_u8_u16(u0.0)),
        vext_u8::<4>(vreinterpret_u8_u16(u1.0), vreinterpret_u8_u16(u1.0)),
        vext_u8::<4>(vreinterpret_u8_u16(u0.1), vreinterpret_u8_u16(u0.1)),
        vext_u8::<4>(vreinterpret_u8_u16(u1.1), vreinterpret_u8_u16(u1.1)),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_rows8(src: *const u8, src_stride: usize, row: usize, col: usize) -> [uint8x8_t; 8] {
    core::array::from_fn(|tap| vld1_u8(src.add((row + tap) * src_stride + col)))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_rows4(src: *const u8, src_stride: usize, row: usize, col: usize) -> [uint8x8_t; 4] {
    core::array::from_fn(|tap| vld1_u8(src.add((row + tap) * src_stride + col)))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_four(dst: *mut u8, value: uint8x8_t) {
    let packed = vget_lane_u32::<0>(vreinterpret_u32_u8(value));
    // SAFETY: callers validate four writable bytes at `dst`; the unaligned
    // store is deliberate because prediction rows need not be aligned.
    unsafe { dst.cast::<u32>().write_unaligned(packed) };
}

#[inline]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn horiz_block8(
    src: *const u8,
    src_stride: usize,
    intermediate: &mut [u8],
    filters: &[[i16; 8]; SUBPEL_SHIFTS],
    x0_q4: i32,
    x_step_q4: i32,
    row: usize,
    w: usize,
) {
    for x in (0..w).step_by(8) {
        let mut filtered = [vdup_n_u8(0); 8];
        for (column, value) in filtered.iter_mut().enumerate() {
            let x_q4 = x0_q4 + (x + column) as i32 * x_step_q4;
            let int_off = (x_q4 >> 4) as usize;
            let phase = (x_q4 & SUBPEL_MASK) as usize;
            let samples = transpose_8x8(load_rows8(src, src_stride, row, int_off));
            *value = if phase == 0 {
                // Phase zero is the identity row [0, 0, 0, 128, ...].
                samples[3]
            } else {
                filter_eight(samples, &filters[phase])
            };
        }
        let rows = transpose_8x8(filtered);
        for (output_row, value) in rows.into_iter().enumerate() {
            // SAFETY: validation guarantees eight writable bytes in every
            // intermediate row at this offset.
            unsafe {
                vst1_u8(
                    intermediate
                        .as_mut_ptr()
                        .add((row + output_row) * INTER_STRIDE + x),
                    value,
                );
            }
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn horiz_block4(
    src: *const u8,
    src_stride: usize,
    intermediate: &mut [u8],
    filters: &[[i16; 8]; SUBPEL_SHIFTS],
    x0_q4: i32,
    x_step_q4: i32,
    row: usize,
) {
    let mut filtered = [0u8; 16];
    for column in 0..4 {
        let x_q4 = x0_q4 + column as i32 * x_step_q4;
        let int_off = (x_q4 >> 4) as usize;
        let phase = (x_q4 & SUBPEL_MASK) as usize;
        let samples = transpose_4x8(load_rows4(src, src_stride, row, int_off));
        let value = if phase == 0 {
            samples[3]
        } else {
            filter_four(samples, &filters[phase])
        };
        // SAFETY: the temporary has sixteen bytes and `value` contributes
        // exactly its low four lanes.
        unsafe { store_four(filtered.as_mut_ptr().add(column * 4), value) };
    }
    for output_row in 0..4 {
        for column in 0..4 {
            intermediate[(row + output_row) * INTER_STRIDE + column] =
                filtered[column * 4 + output_row];
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn horiz_pass(
    src: &[u8],
    src_stride: usize,
    intermediate: &mut [u8],
    filters: &[[i16; 8]; SUBPEL_SHIFTS],
    x0_q4: i32,
    x_step_q4: i32,
    w: usize,
    intermediate_height: usize,
) {
    let src_ptr = src.as_ptr();
    let vector_rows = if w == 4 {
        intermediate_height & !3
    } else {
        intermediate_height & !7
    };
    let row_step = if w == 4 { 4 } else { 8 };
    for row in (0..vector_rows).step_by(row_step) {
        if w == 4 {
            horiz_block4(
                src_ptr,
                src_stride,
                intermediate,
                filters,
                x0_q4,
                x_step_q4,
                row,
            );
        } else {
            horiz_block8(
                src_ptr,
                src_stride,
                intermediate,
                filters,
                x0_q4,
                x_step_q4,
                row,
                w,
            );
        }
    }

    if vector_rows != intermediate_height {
        // The libvpx kernel rounds its horizontal height up to a transpose
        // block.  Use the scalar tail instead, so this wrapper never reads
        // beyond the source slice merely to fill unused transpose lanes.
        crate::inter::filter::convolve8_scaled_horiz(
            &src[vector_rows * src_stride..],
            src_stride,
            &mut intermediate[vector_rows * INTER_STRIDE..],
            INTER_STRIDE,
            filters,
            x0_q4,
            x_step_q4,
            w,
            intermediate_height - vector_rows,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn vert_pass(
    intermediate: &[u8],
    dst: &mut [u8],
    dst_stride: usize,
    filters: &[[i16; 8]; SUBPEL_SHIFTS],
    y0_q4: i32,
    y_step_q4: i32,
    w: usize,
    h: usize,
) {
    let intermediate_ptr = intermediate.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    for row in 0..h {
        let y_q4 = y0_q4 + row as i32 * y_step_q4;
        let int_row = (y_q4 >> 4) as usize;
        let phase = (y_q4 & SUBPEL_MASK) as usize;
        let dst_row = dst_ptr.add(row * dst_stride);
        if phase == 0 {
            let source = intermediate_ptr.add((int_row + 3) * INTER_STRIDE);
            if w == 4 {
                // SAFETY: validation guarantees four readable intermediate
                // bytes and four writable destination bytes.
                store_four(dst_row, vld1_u8(source));
            } else {
                for x in (0..w).step_by(8) {
                    // SAFETY: validation guarantees each eight-byte load and
                    // store is within the active intermediate/destination row.
                    vst1_u8(dst_row.add(x), vld1_u8(source.add(x)));
                }
            }
            continue;
        }

        let filter = &filters[phase];
        if w == 4 {
            let samples = core::array::from_fn(|tap| {
                vld1_u8(intermediate_ptr.add((int_row + tap) * INTER_STRIDE))
            });
            store_four(dst_row, filter_four(samples, filter));
        } else {
            for x in (0..w).step_by(8) {
                let samples = core::array::from_fn(|tap| {
                    vld1_u8(intermediate_ptr.add((int_row + tap) * INTER_STRIDE + x))
                });
                // SAFETY: validation proves the eight-byte output store is in
                // the requested destination row.
                vst1_u8(dst_row.add(x), filter_eight(samples, filter));
            }
        }
    }
}

#[inline]
fn intermediate_height(h: usize, y0_q4: i32, y_step_q4: i32) -> usize {
    (((h as i32 - 1) * y_step_q4 + y0_q4) >> 4) as usize + FILTER_TAPS
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn validate_scaled_footprint(
    src: &[u8],
    src_stride: usize,
    dst: &[u8],
    dst_stride: usize,
    x0_q4: i32,
    x_step_q4: i32,
    y0_q4: i32,
    y_step_q4: i32,
    w: usize,
    h: usize,
) -> usize {
    assert!((1..=64).contains(&w));
    assert!((1..=64).contains(&h));
    assert!(x0_q4 >= 0 && x_step_q4 >= 0 && y0_q4 >= 0 && y_step_q4 >= 0);
    let height = intermediate_height(h, y0_q4, y_step_q4);
    assert!(height <= MAX_INTERMEDIATE_HEIGHT);
    let max_x_q4 = x0_q4 + (w as i32 - 1) * x_step_q4;
    let max_x = (max_x_q4 >> 4) as usize;
    let src_required = (height - 1)
        .checked_mul(src_stride)
        .and_then(|offset| offset.checked_add(max_x + FILTER_TAPS))
        .expect("convolve8_scaled_neon: source footprint overflow");
    let dst_required = (h - 1)
        .checked_mul(dst_stride)
        .and_then(|offset| offset.checked_add(w))
        .expect("convolve8_scaled_neon: destination footprint overflow");
    assert!(
        src.len() >= src_required,
        "convolve8_scaled_neon: source slice is too short"
    );
    assert!(
        dst.len() >= dst_required,
        "convolve8_scaled_neon: destination slice is too short"
    );
    height
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn convolve8_scaled_neon_inner(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filters: &[[i16; 8]; SUBPEL_SHIFTS],
    x0_q4: i32,
    x_step_q4: i32,
    y0_q4: i32,
    y_step_q4: i32,
    w: usize,
    h: usize,
    intermediate_height: usize,
) {
    SCALED_INTERMEDIATE.with_borrow_mut(|intermediate| {
        // SAFETY: the public wrapper validates both source and destination
        // footprints, and `intermediate_height <= 135` matches this buffer.
        unsafe {
            horiz_pass(
                src,
                src_stride,
                intermediate,
                filters,
                x0_q4,
                x_step_q4,
                w,
                intermediate_height,
            );
            vert_pass(
                intermediate,
                dst,
                dst_stride,
                filters,
                y0_q4,
                y_step_q4,
                w,
                h,
            );
        }
    });
}

/// NEON full 2-D scaled 8-tap convolution.
///
/// The function pointer is installed only when AArch64 NEON is available.
/// Widths outside the fixed VP9 block-size set use the scalar implementation,
/// preserving the general scalar API for callers outside the normative path.
#[allow(clippy::too_many_arguments)]
pub fn convolve8_scaled_neon(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filters: &[[i16; 8]; SUBPEL_SHIFTS],
    x0_q4: i32,
    x_step_q4: i32,
    y0_q4: i32,
    y_step_q4: i32,
    w: usize,
    h: usize,
) {
    if !matches!(w, 4 | 8 | 16 | 32 | 64) {
        crate::inter::filter::convolve8_scaled(
            src, src_stride, dst, dst_stride, filters, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h,
        );
        return;
    }
    let height = validate_scaled_footprint(
        src, src_stride, dst, dst_stride, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h,
    );
    // SAFETY: AArch64 dispatch installs this wrapper only when NEON is
    // available; validation above proves every load and store footprint.
    unsafe {
        convolve8_scaled_neon_inner(
            src, src_stride, dst, dst_stride, filters, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h,
            height,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::convolve8_scaled_neon;
    use crate::inter::filter::{
        BILINEAR_KERNEL, EIGHTTAP_KERNEL, EIGHTTAP_SHARP_KERNEL, EIGHTTAP_SMOOTH_KERNEL,
        SUBPEL_MASK, convolve8_scaled,
    };

    struct Lcg(u32);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.0
        }

        fn byte(&mut self) -> u8 {
            (self.next() >> 16) as u8
        }
    }

    fn intermediate_height(h: usize, y0_q4: i32, y_step_q4: i32) -> usize {
        (((h as i32 - 1) * y_step_q4 + y0_q4) >> 4) as usize + 8
    }

    #[allow(clippy::too_many_arguments)]
    fn run_case(
        rng: &mut Lcg,
        kernel: &[[i16; 8]; 16],
        w: usize,
        h: usize,
        x0_q4: i32,
        x_step_q4: i32,
        y0_q4: i32,
        y_step_q4: i32,
        edge_pixels: bool,
    ) {
        let ih = intermediate_height(h, y0_q4, y_step_q4);
        if ih > 135 {
            return;
        }
        let max_x_q4 = x0_q4 + (w as i32 - 1) * x_step_q4;
        let src_width = (max_x_q4 >> 4) as usize + 8;
        let src_stride = src_width + (rng.byte() as usize & 15);
        let dst_stride = w + (rng.byte() as usize & 15);
        let src_len = (ih - 1) * src_stride + src_width;
        let dst_len = (h - 1) * dst_stride + w;
        let mut src = vec![0u8; src_len];
        for value in &mut src {
            *value = if edge_pixels { 0 } else { rng.byte() };
        }
        if edge_pixels {
            for (index, value) in src.iter_mut().enumerate() {
                *value = match index % 8 {
                    0 => 0,
                    1 => 255,
                    2 => 1,
                    3 => 254,
                    _ => rng.byte(),
                };
            }
        }
        let mut scalar: Vec<u8> = (0..dst_len).map(|_| rng.byte()).collect();
        let mut neon = scalar.clone();
        convolve8_scaled(
            &src,
            src_stride,
            &mut scalar,
            dst_stride,
            kernel,
            x0_q4,
            x_step_q4,
            y0_q4,
            y_step_q4,
            w,
            h,
        );
        convolve8_scaled_neon(
            &src, src_stride, &mut neon, dst_stride, kernel, x0_q4, x_step_q4, y0_q4, y_step_q4, w,
            h,
        );
        assert_eq!(
            neon, scalar,
            "scaled NEON mismatch {w}x{h} x0={x0_q4} xs={x_step_q4} y0={y0_q4} ys={y_step_q4} edge={edge_pixels}"
        );
    }

    #[test]
    fn differential_normative_width_height_steps_and_filters() {
        let widths = [4usize, 8, 16, 32, 64];
        let heights = [1usize, 4, 8, 16, 32, 64];
        let steps = [16i32, 20, 24, 32, 64];
        let phases = [0i32, 1, 5, 9, 15];
        let kernels = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        let mut rng = Lcg(0x5ca1_ed00);
        for (shape_index, &w) in widths.iter().enumerate() {
            for &h in &heights {
                for (step_index, &x_step) in steps.iter().enumerate() {
                    for &y_step in &steps {
                        if y_step == 64 && h > 32 {
                            continue;
                        }
                        let x0 = phases[(shape_index + step_index) % phases.len()];
                        let y0 = phases[(shape_index + h + step_index) % phases.len()];
                        for (kernel_index, kernel) in kernels.iter().enumerate() {
                            run_case(
                                &mut rng,
                                kernel,
                                w,
                                h,
                                x0,
                                x_step,
                                y0,
                                y_step,
                                kernel_index % 2 == 1,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn differential_all_q4_phases_and_edge_values() {
        let steps = [16i32, 20, 24, 32];
        let mut rng = Lcg(0xa17e_0001);
        for &x_step in &steps {
            for &y_step in &steps {
                for x0 in 0..=SUBPEL_MASK {
                    for y0 in 0..=SUBPEL_MASK {
                        run_case(
                            &mut rng,
                            &EIGHTTAP_KERNEL,
                            8,
                            8,
                            x0,
                            x_step,
                            y0,
                            y_step,
                            true,
                        );
                    }
                }
            }
        }
    }
}
