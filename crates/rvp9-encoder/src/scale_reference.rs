//! Reconstructed-reference scaling for spatial-SVC mode decision.
//!
//! Pinned libvpx scales a lower spatial layer's reconstructed reference to the
//! current layer with `vp9_scale_and_extend_frame(..., EIGHTTAP, 0)` before
//! motion search. This is deliberately separate from source-frame scaling,
//! which uses a different filter and phase.

#[cfg(any(test, not(target_arch = "aarch64")))]
use rvp9_core::inter::filter::SUBPEL_MASK;
use rvp9_core::inter::filter::{EIGHTTAP_KERNEL, FILTER_BITS, FILTER_TAPS};

use crate::scale_source::ScaledI420;

#[cfg(any(test, not(target_arch = "aarch64")))]
const SUBPEL_BITS: u32 = 4;
#[cfg(any(test, not(target_arch = "aarch64")))]
const UPSCALE_STEP_Q4: i32 = 8;

/// Reference-scaling input error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ScaleReferenceError {
    /// Canonical I420 2x scaling requires positive, even source dimensions.
    InvalidDimensions,
    /// One or more input planes do not match the dimensions exactly.
    InvalidPlaneLength,
}

/// Scale one tightly packed reconstructed I420 reference by exactly 2x.
///
/// This matches pinned libvpx's encoder-side scaled-reference image for the
/// canonical adjacent spatial-layer relationship: regular eight-tap filtering,
/// phase zero, an 8-bit rounded intermediate, and visible-edge replication.
pub(crate) fn scale_reference_i420_2x(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> Result<ScaledI420, ScaleReferenceError> {
    if width == 0
        || height == 0
        || !width.is_multiple_of(2)
        || !height.is_multiple_of(2)
        || width > (isize::MAX as usize) / 2
        || height > (isize::MAX as usize) / 2
    {
        return Err(ScaleReferenceError::InvalidDimensions);
    }
    let y_len = width
        .checked_mul(height)
        .ok_or(ScaleReferenceError::InvalidDimensions)?;
    let uv_width = width / 2;
    let uv_height = height / 2;
    let uv_len = uv_width
        .checked_mul(uv_height)
        .ok_or(ScaleReferenceError::InvalidDimensions)?;
    if y.len() != y_len || u.len() != uv_len || v.len() != uv_len {
        return Err(ScaleReferenceError::InvalidPlaneLength);
    }

    Ok(ScaledI420 {
        width: width * 2,
        height: height * 2,
        y: scale_plane_reference_2x(y, width, height)?,
        u: scale_plane_reference_2x(u, uv_width, uv_height)?,
        v: scale_plane_reference_2x(v, uv_width, uv_height)?,
    })
}

fn scale_plane_reference_2x(
    input: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, ScaleReferenceError> {
    #[cfg(target_arch = "aarch64")]
    {
        scale_plane_reference_2x_neon(input, width, height)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scale_plane_reference_2x_scalar(input, width, height)
    }
}

#[cfg(target_arch = "aarch64")]
fn scale_plane_reference_2x_neon(
    input: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, ScaleReferenceError> {
    let out_width = width
        .checked_mul(2)
        .ok_or(ScaleReferenceError::InvalidDimensions)?;
    let out_height = height
        .checked_mul(2)
        .ok_or(ScaleReferenceError::InvalidDimensions)?;
    let output_len = out_width
        .checked_mul(out_height)
        .ok_or(ScaleReferenceError::InvalidDimensions)?;
    let mut output = vec![0u8; output_len];

    // Horizontally expanded source rows are exactly the even output rows.
    // Keep them in their final locations instead of staging and then copying
    // a complete intermediate frame.
    for source_y in 0..height {
        let source_row = &input[source_y * width..(source_y + 1) * width];
        let row_start = 2 * source_y * out_width;
        let output_row = &mut output[row_start..row_start + out_width];
        // SAFETY: AArch64 always provides NEON. Both row slices cover the
        // exact widths passed to the inner routine, whose vector loop only
        // reads full eight-sample chunks inside the visible source row.
        unsafe { upscale_reference_row_neon(source_row, output_row) };
    }

    // Every vertical input is an already-written even row and every output is
    // a distinct odd row. The helper applies visible-edge replication while
    // loading those eight source rows.
    for source_y in 0..height {
        // SAFETY: all even rows were initialized above. The helper reads only
        // those rows and writes the disjoint odd row for `source_y`; its clamps
        // keep every tap inside the `height` visible source rows.
        unsafe {
            filter_reference_phase8_row_from_even_neon(
                output.as_mut_ptr(),
                out_width,
                height,
                source_y,
            );
        }
    }
    Ok(output)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn filter_reference_phase8_8_libvpx(
    base: *const u8,
    stride: usize,
) -> core::arch::aarch64::uint8x8_t {
    // SAFETY: callers prove that `base` starts an eight-row by eight-byte
    // readable footprint with the supplied row stride.
    unsafe {
        filter_reference_phase8_8_rows_libvpx([
            base,
            base.add(stride),
            base.add(2 * stride),
            base.add(3 * stride),
            base.add(4 * stride),
            base.add(5 * stride),
            base.add(6 * stride),
            base.add(7 * stride),
        ])
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn filter_reference_phase8_8_rows_libvpx(
    rows: [*const u8; FILTER_TAPS],
) -> core::arch::aarch64::uint8x8_t {
    use core::arch::aarch64::*;

    const COEFFICIENTS: [i16; 8] = [-1, 6, -19, 78, 78, -19, 6, -1];
    // SAFETY: callers prove eight readable samples at every one of the eight
    // tap offsets. The accumulation order is pinned libvpx's `convolve8_8`:
    // the six outer taps wrap in i16, then the two dominant positive taps are
    // added with saturation. Its valid VP9 filter range is scalar-exact.
    unsafe {
        let s0 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(rows[0])));
        let s1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(rows[1])));
        let s2 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(rows[2])));
        let s3 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(rows[3])));
        let s4 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(rows[4])));
        let s5 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(rows[5])));
        let s6 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(rows[6])));
        let s7 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(rows[7])));
        let mut sum = vmulq_n_s16(s0, COEFFICIENTS[0]);
        sum = vmlaq_n_s16(sum, s1, COEFFICIENTS[1]);
        sum = vmlaq_n_s16(sum, s2, COEFFICIENTS[2]);
        sum = vmlaq_n_s16(sum, s5, COEFFICIENTS[5]);
        sum = vmlaq_n_s16(sum, s6, COEFFICIENTS[6]);
        sum = vmlaq_n_s16(sum, s7, COEFFICIENTS[7]);
        sum = vqaddq_s16(sum, vmulq_n_s16(s3, COEFFICIENTS[3]));
        sum = vqaddq_s16(sum, vmulq_n_s16(s4, COEFFICIENTS[4]));
        vqrshrun_n_s16::<7>(sum)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn upscale_reference_row_neon(source: &[u8], output: &mut [u8]) {
    use core::arch::aarch64::*;

    let width = source.len();
    let mut source_x = 0;
    while source_x < 3.min(width) {
        output[2 * source_x] = source[source_x];
        output[2 * source_x + 1] = reference_phase8_sample(source, source_x);
        source_x += 1;
    }
    // SAFETY: the loop bound leaves three source samples before the first tap
    // and four after the last. Each pair of stores covers the corresponding
    // sixteen-byte destination span.
    unsafe {
        while source_x + 11 < width {
            let even = vld1_u8(source.as_ptr().add(source_x));
            let odd = filter_reference_phase8_8_libvpx(source.as_ptr().add(source_x - 3), 1);
            vst1_u8(output.as_mut_ptr().add(2 * source_x), vzip1_u8(even, odd));
            vst1_u8(
                output.as_mut_ptr().add(2 * source_x + 8),
                vzip2_u8(even, odd),
            );
            source_x += 8;
        }
    }
    while source_x < width {
        output[2 * source_x] = source[source_x];
        output[2 * source_x + 1] = reference_phase8_sample(source, source_x);
        source_x += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_reference_phase8_row_from_even_neon(
    output: *mut u8,
    width: usize,
    height: usize,
    source_y: usize,
) {
    let rows = std::array::from_fn(|tap| {
        let row = (source_y as isize + tap as isize - 3).clamp(0, height as isize - 1) as usize;
        // SAFETY: the caller provides a `2 * height * width` allocation, and
        // `row` is clamped to a valid source row before this even-row offset.
        unsafe { output.cast_const().add(2 * row * width) }
    });
    // SAFETY: `source_y < height`, so its odd-row destination lies inside the
    // caller-provided `2 * height * width` allocation.
    let destination = unsafe { output.add((2 * source_y + 1) * width) };
    let mut x = 0;
    // SAFETY: the caller proves every clamped even-row input and the odd-row
    // destination cover `width` bytes. Vector chunks and the scalar tail
    // partition that exact footprint.
    unsafe {
        while x + 8 <= width {
            core::arch::aarch64::vst1_u8(
                destination.add(x),
                filter_reference_phase8_8_rows_libvpx(rows.map(|row| row.add(x))),
            );
            x += 8;
        }
        const KERNEL: &[i16; FILTER_TAPS] = &EIGHTTAP_KERNEL[8];
        while x < width {
            let mut sum = 0i32;
            for (tap, &coefficient) in KERNEL.iter().enumerate() {
                sum += i32::from(*rows[tap].add(x)) * i32::from(coefficient);
            }
            *destination.add(x) = round_and_clip(sum);
            x += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn reference_phase8_sample(source: &[u8], source_x: usize) -> u8 {
    let mut sum = 0i32;
    for (tap, &coefficient) in EIGHTTAP_KERNEL[8].iter().enumerate() {
        let sample_x =
            (source_x as isize + tap as isize - 3).clamp(0, source.len() as isize - 1) as usize;
        sum += i32::from(source[sample_x]) * i32::from(coefficient);
    }
    round_and_clip(sum)
}

#[cfg(any(test, not(target_arch = "aarch64")))]
fn scale_plane_reference_2x_scalar(
    input: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, ScaleReferenceError> {
    let out_width = width
        .checked_mul(2)
        .ok_or(ScaleReferenceError::InvalidDimensions)?;
    let out_height = height
        .checked_mul(2)
        .ok_or(ScaleReferenceError::InvalidDimensions)?;
    let intermediate_height = height
        .checked_add(FILTER_TAPS - 1)
        .ok_or(ScaleReferenceError::InvalidDimensions)?;
    let intermediate_len = out_width
        .checked_mul(intermediate_height)
        .ok_or(ScaleReferenceError::InvalidDimensions)?;
    let mut intermediate = vec![0u8; intermediate_len];

    // Intermediate row zero represents source row -3, matching the source
    // pointer shift inside libvpx's separable `vpx_convolve8_c`.
    for intermediate_y in 0..intermediate_height {
        let source_y = (intermediate_y as isize - 3).clamp(0, height as isize - 1) as usize;
        let source_row = &input[source_y * width..(source_y + 1) * width];
        let output_row =
            &mut intermediate[intermediate_y * out_width..(intermediate_y + 1) * out_width];
        for (x, output) in output_row.iter_mut().enumerate() {
            let x_q4 = x as i32 * UPSCALE_STEP_Q4;
            let source_x = x_q4 >> SUBPEL_BITS;
            let kernel = &EIGHTTAP_KERNEL[(x_q4 & SUBPEL_MASK) as usize];
            let mut sum = 0i32;
            for (tap, &coefficient) in kernel.iter().enumerate() {
                let sample_x =
                    (source_x as isize + tap as isize - 3).clamp(0, width as isize - 1) as usize;
                sum += i32::from(source_row[sample_x]) * i32::from(coefficient);
            }
            *output = round_and_clip(sum);
        }
    }

    let output_len = out_width
        .checked_mul(out_height)
        .ok_or(ScaleReferenceError::InvalidDimensions)?;
    let mut output = vec![0u8; output_len];
    for y in 0..out_height {
        let y_q4 = y as i32 * UPSCALE_STEP_Q4;
        let first_intermediate_y = (y_q4 >> SUBPEL_BITS) as usize;
        let kernel = &EIGHTTAP_KERNEL[(y_q4 & SUBPEL_MASK) as usize];
        for x in 0..out_width {
            let mut sum = 0i32;
            for (tap, &coefficient) in kernel.iter().enumerate() {
                let sample = intermediate[(first_intermediate_y + tap) * out_width + x];
                sum += i32::from(sample) * i32::from(coefficient);
            }
            output[y * out_width + x] = round_and_clip(sum);
        }
    }
    Ok(output)
}

#[inline]
fn round_and_clip(sum: i32) -> u8 {
    ((sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS).clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;

    fn random_i420(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut state = 0x51ca_1e5eu32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        };
        let y = (0..width * height).map(|_| next()).collect();
        let uv_len = width * height / 4;
        let u = (0..uv_len).map(|_| next()).collect();
        let v = (0..uv_len).map(|_| next()).collect();
        (y, u, v)
    }

    fn packed_hash(frame: &ScaledI420) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&frame.y);
        hasher.update(&frame.u);
        hasher.update(&frame.v);
        format!("{:x}", hasher.finalize())
    }

    #[test]
    fn constant_planes_stay_constant() {
        let width = 32;
        let height = 16;
        let frame = scale_reference_i420_2x(
            &vec![17; width * height],
            &vec![91; width * height / 4],
            &vec![233; width * height / 4],
            width,
            height,
        )
        .unwrap();
        assert_eq!((frame.width, frame.height), (64, 32));
        assert!(frame.y.iter().all(|&sample| sample == 17));
        assert!(frame.u.iter().all(|&sample| sample == 91));
        assert!(frame.v.iter().all(|&sample| sample == 233));
    }

    #[test]
    fn optimized_reference_plane_matches_scalar_across_shapes() {
        let mut state = 0x7e5c_a1e5u32;
        for (width, height) in [(4, 2), (8, 8), (32, 18), (80, 45), (160, 90)] {
            let input = (0..width * height)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state as u8
                })
                .collect::<Vec<_>>();
            let scalar = scale_plane_reference_2x_scalar(&input, width, height).unwrap();
            let optimized = scale_plane_reference_2x(&input, width, height).unwrap();
            assert_eq!(optimized, scalar, "{width}x{height}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn libvpx_neon_phase8_schedule_matches_scalar_extremes() {
        use core::arch::aarch64::vget_lane_u8;

        for stride in [1usize, 2] {
            for mask in 0u16..256 {
                let mut source = [17u8; 22];
                for tap in 0..FILTER_TAPS {
                    source[tap * stride] = if mask & (1 << tap) == 0 { 0 } else { 255 };
                }
                // SAFETY: `source` covers the helper's eight-byte load at
                // every one of the eight tap offsets for both tested strides.
                let vector = unsafe { filter_reference_phase8_8_libvpx(source.as_ptr(), stride) };
                // SAFETY: lane zero is present in every uint8x8 vector.
                let actual = unsafe { vget_lane_u8::<0>(vector) };
                let expected = round_and_clip(
                    EIGHTTAP_KERNEL[8]
                        .iter()
                        .enumerate()
                        .map(|(tap, &coefficient)| {
                            i32::from(source[tap * stride]) * i32::from(coefficient)
                        })
                        .sum(),
                );
                assert_eq!(actual, expected, "stride={stride} mask={mask:#04x}");
            }
        }
    }

    #[test]
    fn validates_dimensions_and_planes() {
        assert_eq!(
            scale_reference_i420_2x(&[], &[], &[], 0, 2),
            Err(ScaleReferenceError::InvalidDimensions)
        );
        assert_eq!(
            scale_reference_i420_2x(&[], &[], &[], 3, 2),
            Err(ScaleReferenceError::InvalidDimensions)
        );
        assert_eq!(
            scale_reference_i420_2x(&[0; 64], &[0; 15], &[0; 16], 8, 8),
            Err(ScaleReferenceError::InvalidPlaneLength)
        );
    }

    #[test]
    fn random_reference_scale_matches_pinned_libvpx() {
        let (y, u, v) = random_i420(160, 90);
        let frame = scale_reference_i420_2x(&y, &u, &v, 160, 90).unwrap();
        assert_eq!(
            packed_hash(&frame),
            "f57c311767cb1fcc4d335f4a5f2e47e30cf924c8d1dd067a4be86f0efefaff4c"
        );
    }
}
