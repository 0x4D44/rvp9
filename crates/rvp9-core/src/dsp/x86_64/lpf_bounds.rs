//! Checked footprints for the x86 loop-filter wrappers.

/// Validate every byte loaded/stored by one or two adjacent SIMD strips.
/// Returns the second strip's start (the first start for a single strip).
#[inline]
pub(super) fn validate(
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
    use super::validate;
    use std::panic::catch_unwind;

    #[test]
    fn lpf_bounds_exact_footprints_and_rejections() {
        for vertical in [false, true] {
            for radius in [4, 8] {
                for lanes in [8, 16] {
                    let pitch = 32;
                    let index = if vertical { radius } else { radius * pitch };
                    let len = if vertical {
                        index + (lanes - 1) * pitch + radius
                    } else {
                        index + (radius - 1) * pitch + lanes
                    };
                    let second = validate(len, index, pitch, radius, vertical, lanes);
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
                        catch_unwind(|| validate(len - 1, index, pitch, radius, vertical, lanes))
                            .is_err()
                    );
                    assert!(
                        catch_unwind(|| validate(len, index - 1, pitch, radius, vertical, lanes))
                            .is_err()
                    );
                    assert!(
                        catch_unwind(|| validate(
                            usize::MAX,
                            index,
                            usize::MAX,
                            radius,
                            vertical,
                            lanes
                        ))
                        .is_err()
                    );
                    assert!(
                        catch_unwind(|| validate(
                            usize::MAX,
                            usize::MAX,
                            pitch,
                            radius,
                            vertical,
                            lanes
                        ))
                        .is_err()
                    );
                }
            }
        }
    }
    #[test]
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    fn lpf_bounds_wrappers_reject_before_memory_access() {
        use crate::dsp::x86_64 as x86;
        use std::panic::AssertUnwindSafe;
        type Single = fn(&mut [u8], usize, usize, u8, u8, u8);
        type Dual = fn(&mut [u8], usize, usize, u8, u8, u8, u8, u8, u8);
        let singles: [Single; 8] = [
            x86::lpf_vertical_4_avx2,
            x86::lpf_vertical_8_avx2,
            x86::lpf_vertical_16_avx2,
            x86::lpf_horizontal_4_avx2,
            x86::lpf_horizontal_8_avx2,
            x86::lpf_horizontal_16_avx2,
            x86::lpf_vertical_16_dual_avx2,
            x86::lpf_horizontal_16_dual_avx2,
        ];
        for call in singles {
            for (index, pitch) in [(0, 16), (usize::MAX, 16), (8, usize::MAX)] {
                assert!(
                    catch_unwind(AssertUnwindSafe(|| call(&mut [], index, pitch, 0, 0, 0)))
                        .is_err()
                );
            }
        }
        let duals: [Dual; 4] = [
            x86::lpf_vertical_4_dual_avx2,
            x86::lpf_vertical_8_dual_avx2,
            x86::lpf_horizontal_4_dual_avx2,
            x86::lpf_horizontal_8_dual_avx2,
        ];
        for call in duals {
            for (index, pitch) in [(0, 16), (usize::MAX, 16), (8, usize::MAX)] {
                assert!(
                    catch_unwind(AssertUnwindSafe(|| call(
                        &mut [],
                        index,
                        pitch,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    )))
                    .is_err()
                );
            }
        }
    }
}
