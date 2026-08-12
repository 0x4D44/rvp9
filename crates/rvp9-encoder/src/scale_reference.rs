//! Reconstructed-reference scaling for spatial-SVC mode decision.
//!
//! Pinned libvpx scales a lower spatial layer's reconstructed reference to the
//! current layer with `vp9_scale_and_extend_frame(..., EIGHTTAP, 0)` before
//! motion search. This is deliberately separate from source-frame scaling,
//! which uses a different filter and phase.

use rvp9_core::inter::filter::{EIGHTTAP_KERNEL, FILTER_BITS, FILTER_TAPS, SUBPEL_MASK};

use crate::scale_source::ScaledI420;

const SUBPEL_BITS: u32 = 4;
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
