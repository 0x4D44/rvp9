//! Source-frame scaling for the real-time spatial-SVC pipeline.
//!
//! Pinned libvpx uses `vp9_scale_and_extend_frame` with phase 8 for the
//! canonical 2:1 spatial steps. This module keeps that operation separate
//! from reference-frame scaling: it filters the uncompressed I420 source
//! before a layer is encoded.

use rvp9_core::bitstream::frame::InterpFilter;
use rvp9_core::inter::filter::{BILINEAR_KERNEL, EIGHTTAP_SMOOTH_KERNEL, FILTER_BITS};

const FILTER_TAPS: usize = 8;
const FILTER_CENTER: isize = 3;
const HALF_PIXEL_PHASE: usize = 8;

/// A tightly packed 8-bit I420 frame produced by [`scale_i420_2x`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScaledI420 {
    /// Luma width in pixels.
    pub width: usize,
    /// Luma height in pixels.
    pub height: usize,
    /// Tightly packed luma plane.
    pub y: Vec<u8>,
    /// Tightly packed 4:2:0 Cb plane.
    pub u: Vec<u8>,
    /// Tightly packed 4:2:0 Cr plane.
    pub v: Vec<u8>,
}

/// Source-scaling input error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ScaleSourceError {
    /// Canonical I420 2:1 scaling requires positive dimensions divisible by 4.
    InvalidDimensions,
    /// One or more input planes do not match the dimensions exactly.
    InvalidPlaneLength,
    /// Spatial SVC currently supports libvpx's bilinear and low-resolution
    /// eight-tap-smooth source filters.
    UnsupportedFilter,
}

impl std::fmt::Display for ScaleSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions => f.write_str(
                "I420 source dimensions must be positive and divisible by 4 for 2x scaling",
            ),
            Self::InvalidPlaneLength => {
                f.write_str("I420 source plane lengths do not match the frame dimensions")
            }
            Self::UnsupportedFilter => {
                f.write_str("source scaling supports only bilinear or eight-tap-smooth filtering")
            }
        }
    }
}

impl std::error::Error for ScaleSourceError {}

/// Downscale one tightly packed 8-bit I420 frame by exactly 2× in each axis.
///
/// This is the canonical step used by two-layer `1/2, 1/1` and three-layer
/// `1/4, 1/2, 1/1` spatial SVC. A 4× base layer is formed by calling this
/// function twice, matching pinned libvpx's `vp9_svc_twostage_scale` path.
/// The filter is evaluated at subpixel phase 8, which is libvpx's default
/// symmetric averaging phase for spatial layers.
///
/// # Errors
///
/// Returns [`ScaleSourceError::InvalidDimensions`] unless the luma dimensions
/// are positive multiples of four, [`ScaleSourceError::InvalidPlaneLength`]
/// unless all planes are tightly packed with their exact I420 lengths, or
/// [`ScaleSourceError::UnsupportedFilter`] for filters outside the v1 spatial
/// source-scaling surface.
pub fn scale_i420_2x(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
    filter: InterpFilter,
) -> Result<ScaledI420, ScaleSourceError> {
    if width == 0
        || height == 0
        || !width.is_multiple_of(4)
        || !height.is_multiple_of(4)
        || width > isize::MAX as usize
        || height > isize::MAX as usize
    {
        return Err(ScaleSourceError::InvalidDimensions);
    }
    let y_len = width
        .checked_mul(height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let uv_width = width / 2;
    let uv_height = height / 2;
    let uv_len = uv_width
        .checked_mul(uv_height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    if y.len() != y_len || u.len() != uv_len || v.len() != uv_len {
        return Err(ScaleSourceError::InvalidPlaneLength);
    }

    let kernel = match filter {
        InterpFilter::Bilinear => &BILINEAR_KERNEL[HALF_PIXEL_PHASE],
        InterpFilter::EightTapSmooth => &EIGHTTAP_SMOOTH_KERNEL[HALF_PIXEL_PHASE],
        _ => return Err(ScaleSourceError::UnsupportedFilter),
    };
    let out_width = width / 2;
    let out_height = height / 2;
    Ok(ScaledI420 {
        width: out_width,
        height: out_height,
        y: scale_plane_2x(y, width, height, kernel)?,
        u: scale_plane_2x(u, uv_width, uv_height, kernel)?,
        v: scale_plane_2x(v, uv_width, uv_height, kernel)?,
    })
}

fn scale_plane_2x(
    input: &[u8],
    width: usize,
    height: usize,
    kernel: &[i16; FILTER_TAPS],
) -> Result<Vec<u8>, ScaleSourceError> {
    let out_width = width / 2;
    let out_height = height / 2;

    // `vpx_convolve8_c` first filters horizontally into an 8-bit
    // intermediate, then filters that vertically. The vertical footprint for
    // output rows 0..out_height spans source rows -3..height+3. Materialise
    // exactly those rows with visible-edge replication, as libvpx's extended
    // YV12 source does.
    let intermediate_height = height
        .checked_add(FILTER_TAPS - 1)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let intermediate_len = out_width
        .checked_mul(intermediate_height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let mut intermediate = vec![0u8; intermediate_len];
    for intermediate_y in 0..intermediate_height {
        let source_y =
            (intermediate_y as isize - FILTER_CENTER).clamp(0, height as isize - 1) as usize;
        let source_row = &input[source_y * width..(source_y + 1) * width];
        let output_row =
            &mut intermediate[intermediate_y * out_width..(intermediate_y + 1) * out_width];
        for (x, output) in output_row.iter_mut().enumerate() {
            let center = (x * 2) as isize;
            let mut sum = 0i32;
            for (tap, &coefficient) in kernel.iter().enumerate() {
                let source_x =
                    (center + tap as isize - FILTER_CENTER).clamp(0, width as isize - 1) as usize;
                sum += i32::from(source_row[source_x]) * i32::from(coefficient);
            }
            *output = round_and_clip(sum);
        }
    }

    let output_len = out_width
        .checked_mul(out_height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let mut output = vec![0u8; output_len];
    for y in 0..out_height {
        for x in 0..out_width {
            let first_intermediate_y = y * 2;
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
        let mut state = 0x5ca1_ab1eu32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        };
        let y = (0..width * height).map(|_| next()).collect();
        let uv_len = (width / 2) * (height / 2);
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
        let width = 64;
        let height = 32;
        let y = vec![17; width * height];
        let u = vec![91; width * height / 4];
        let v = vec![233; width * height / 4];
        for filter in [InterpFilter::Bilinear, InterpFilter::EightTapSmooth] {
            let scaled = scale_i420_2x(&y, &u, &v, width, height, filter).unwrap();
            assert_eq!(scaled.width, 32);
            assert_eq!(scaled.height, 16);
            assert!(scaled.y.iter().all(|&sample| sample == 17));
            assert!(scaled.u.iter().all(|&sample| sample == 91));
            assert!(scaled.v.iter().all(|&sample| sample == 233));
        }
    }

    #[test]
    fn validates_dimensions_planes_and_filter() {
        assert_eq!(
            scale_i420_2x(&[], &[], &[], 0, 4, InterpFilter::Bilinear),
            Err(ScaleSourceError::InvalidDimensions)
        );
        assert_eq!(
            scale_i420_2x(&[], &[], &[], 6, 4, InterpFilter::Bilinear),
            Err(ScaleSourceError::InvalidDimensions)
        );
        assert_eq!(
            scale_i420_2x(&[0; 64], &[0; 15], &[0; 16], 8, 8, InterpFilter::Bilinear),
            Err(ScaleSourceError::InvalidPlaneLength)
        );
        assert_eq!(
            scale_i420_2x(&[0; 64], &[0; 16], &[0; 16], 8, 8, InterpFilter::EightTap),
            Err(ScaleSourceError::UnsupportedFilter)
        );
    }

    #[test]
    fn random_720p_half_scale_matches_pinned_libvpx() {
        let (y, u, v) = random_i420(1280, 720);
        let bilinear = scale_i420_2x(&y, &u, &v, 1280, 720, InterpFilter::Bilinear).unwrap();
        assert_eq!(
            packed_hash(&bilinear),
            "2f2f77757ac5c0383bec6d7a4e7f162213373f34c4cd5408d12071d3e2934741"
        );
        let smooth = scale_i420_2x(&y, &u, &v, 1280, 720, InterpFilter::EightTapSmooth).unwrap();
        assert_eq!(
            packed_hash(&smooth),
            "5616cf7b5089ad6af0c6e47b97986bdd7758663bdde50b9eaf43a39c2b133d97"
        );
    }

    #[test]
    fn random_720p_two_stage_quarter_scale_matches_pinned_libvpx() {
        let (y, u, v) = random_i420(1280, 720);
        for (filter, expected) in [
            (
                InterpFilter::Bilinear,
                "a3ead90551cdf43d97a531a7bb3c6cab61fdb3fd9cb81e3058b800b8c261e470",
            ),
            (
                InterpFilter::EightTapSmooth,
                "7c115cdcc64fbb9594d4c575cd371ddab5b1ff09f0c3c786aa70b191912adc26",
            ),
        ] {
            let half = scale_i420_2x(&y, &u, &v, 1280, 720, filter).unwrap();
            let quarter =
                scale_i420_2x(&half.y, &half.u, &half.v, half.width, half.height, filter).unwrap();
            assert_eq!((quarter.width, quarter.height), (320, 180));
            assert_eq!(packed_hash(&quarter), expected);
        }
    }
}
