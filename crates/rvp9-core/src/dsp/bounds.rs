//! Checked element extents for SIMD plane and loop-filter accesses.

/// Return the required element length through a zero-based final row.
///
/// Strides and widths count elements, not bytes. An overflowing footprint
/// must be rejected before a caller derives pointers from a slice.
#[cfg(any(target_arch = "aarch64", test))]
#[inline]
pub(crate) fn strided_len(last_row: usize, stride: usize, row_width: usize) -> usize {
    last_row
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(row_width))
        .expect("strided layout overflow")
}

/// Validate every byte loaded/stored by one or two adjacent SIMD strips.
/// Returns the second strip's start (the first start for a single strip).
#[inline]
pub(crate) fn validate_lpf(
    len: usize,
    index: usize,
    pitch: usize,
    radius: usize,
    vertical: bool,
    lanes: usize,
) -> usize {
    let (before, after_rows, after_cols) = if vertical {
        (radius, lanes - 1, radius)
    } else {
        (
            radius
                .checked_mul(pitch)
                .expect("loop-filter layout overflow"),
            radius - 1,
            lanes,
        )
    };
    assert!(index >= before, "loop-filter prefix too short");
    let end = after_rows
        .checked_mul(pitch)
        .and_then(|offset| index.checked_add(offset))
        .and_then(|offset| offset.checked_add(after_cols))
        .expect("loop-filter layout overflow");
    assert!(end <= len, "loop-filter slice too short");
    if lanes == 16 {
        let offset = if vertical {
            8usize
                .checked_mul(pitch)
                .expect("loop-filter layout overflow")
        } else {
            8
        };
        index
            .checked_add(offset)
            .expect("loop-filter layout overflow")
    } else {
        index
    }
}

#[cfg(test)]
mod tests {
    use super::{strided_len, validate_lpf};
    use std::panic::catch_unwind;

    #[test]
    fn strided_extents_reject_overflow_and_preserve_exact_tails() {
        // Shapes used by the recorded NEON copy, transforms, distortion,
        // integer-projection, subpixel and partition primitives.
        for (last_row, width) in [(3, 4), (7, 8), (15, 16), (31, 32), (63, 64), (64, 65)] {
            let stride = width + 5;
            assert_eq!(
                strided_len(last_row, stride, width),
                last_row * stride + width
            );
            assert!(
                catch_unwind(|| strided_len(last_row, usize::MAX, width)).is_err(),
                "accepted overflowing row offset: last_row={last_row} width={width}"
            );
        }
        assert_eq!(strided_len(0, usize::MAX, 8), 8);
        assert!(
            catch_unwind(|| strided_len(1, usize::MAX, 1)).is_err(),
            "accepted overflowing final row width"
        );
        assert!(
            catch_unwind(|| strided_len(7, usize::MAX / 7, 8)).is_err(),
            "accepted the recorded minmax wrapped extent"
        );
    }

    #[test]
    fn loop_filter_extents_cover_exact_windows_and_reject_overflow() {
        for vertical in [false, true] {
            for radius in [4, 8] {
                for lanes in [8, 16] {
                    let pitch = 32;
                    let index = if vertical { radius } else { radius * pitch };
                    let end = if vertical {
                        index + (lanes - 1) * pitch + radius
                    } else {
                        index + (radius - 1) * pitch + lanes
                    };
                    let second = validate_lpf(end, index, pitch, radius, vertical, lanes);
                    assert_eq!(
                        second,
                        index
                            + if lanes == 8 {
                                0
                            } else if vertical {
                                8 * pitch
                            } else {
                                8
                            }
                    );
                    assert!(
                        catch_unwind(|| validate_lpf(
                            end - 1,
                            index,
                            pitch,
                            radius,
                            vertical,
                            lanes
                        ))
                        .is_err()
                    );
                    assert!(
                        catch_unwind(|| validate_lpf(
                            usize::MAX,
                            usize::MAX,
                            usize::MAX,
                            radius,
                            vertical,
                            lanes
                        ))
                        .is_err(),
                        "accepted overflowing filter window"
                    );
                }
            }
        }
    }
}
