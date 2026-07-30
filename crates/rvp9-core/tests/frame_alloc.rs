//! Regression coverage for public frame allocation validation.

use rvp9_core::bitstream::ChromaFormat;
use rvp9_core::frame::{BorderUsage, ByteAlignment, FrameAllocError, Yv12Buffer};

#[test]
fn huge_aligned_custom_border_is_rejected() {
    let border = !31u32;
    let error = Yv12Buffer::alloc(
        8,
        8,
        ChromaFormat::Yuv420,
        BorderUsage::Custom(border),
        ByteAlignment::Legacy,
    )
    .unwrap_err();
    assert_eq!(
        error,
        FrameAllocError::AllocationTooLarge {
            width: 8,
            height: 8,
            border,
        }
    );
}

#[test]
fn custom_border_with_overlarge_total_allocation_is_rejected() {
    let border = 0x7fff_ffe0;
    let error = Yv12Buffer::alloc(
        8,
        8,
        ChromaFormat::Yuv420,
        BorderUsage::Custom(border),
        ByteAlignment::Legacy,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        FrameAllocError::AllocationTooLarge {
            width: 8,
            height: 8,
            border: 0x7fff_ffe0,
        }
    ));
}

#[test]
fn ordinary_custom_border_still_allocates() {
    let frame = Yv12Buffer::alloc(
        64,
        48,
        ChromaFormat::Yuv420,
        BorderUsage::Custom(64),
        ByteAlignment::V32,
    )
    .expect("ordinary custom border should remain valid");
    assert_eq!(frame.width(), 64);
    assert_eq!(frame.height(), 48);
    assert_eq!(frame.border(), 64);
}
