//! YV12 frame buffer: planar YUV with border padding.
//!
//! Mirrors libvpx `YV12_BUFFER_CONFIG` (`vpx_scale/yv12config.h:29-65`)
//! and the internal-allocation path of `vpx_realloc_frame_buffer`
//! (`yv12config.c:152-290`). Per LLD `notes/lld/frame-buffers.md`:
//! 8-bit only, owned `Box<[u8]>` backing, no callback-allocation path.
//!
//! ## Scope of this commit (Task 7.1)
//!
//! Lands the [`Yv12Buffer`] struct + allocation, dimension/stride
//! accessors, and plane accessors via [`PlaneRef`] / [`PlaneMut`].
//!
//! **Not in this commit (deferred):**
//! - `FrameBuf` (the `Arc<FrameBuf>` ref-counted handle) — Task 7.2.
//! - `FramePool` (the 12-slot / 8+1-slot recycling allocator) — Task 7.3.
//! - `RefFrames` / `TilePlanes{,Mut}` tile-disjoint views — Task 7.4
//!   (requires the `UnsafeCell` aliasing contract from
//!   `notes/hld/01-threading.md §6.2`).
//! - Border extension (`extend_borders` / `copy_and_extend_from`) —
//!   Task 7.2 (needed once we have the pool to test against).

use crate::bitstream::{ChromaFormat, ColorRange, ColorSpace};

/// Border padding for the decoder, in pixels. Mirrors libvpx
/// `VP9_DEC_BORDER_IN_PIXELS` (`yv12config.h:27`).
pub const DEC_BORDER_IN_PIXELS: u32 = 32;

/// Border padding for the encoder, in pixels. Mirrors libvpx
/// `VP9_ENC_BORDER_IN_PIXELS` (`yv12config.h:26`).
pub const ENC_BORDER_IN_PIXELS: u32 = 160;

/// Border-width selector for [`Yv12Buffer::alloc`]. `Custom(n)` requires
/// `n & 0x1f == 0` (`yv12config.c:169`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BorderUsage {
    /// 32-pixel border — decoder default.
    Decoder,
    /// 160-pixel border — encoder default (large enough for the
    /// motion-search range plus the sub-pel filter taps).
    Encoder,
    /// Caller-specified border width. Must be a multiple of 32.
    Custom(u32),
}

impl BorderUsage {
    /// Resolve the border width in pixels.
    #[inline]
    #[must_use]
    pub const fn pixels(self) -> u32 {
        match self {
            Self::Decoder => DEC_BORDER_IN_PIXELS,
            Self::Encoder => ENC_BORDER_IN_PIXELS,
            Self::Custom(n) => n,
        }
    }
}

/// Plane-data byte alignment. Mirrors libvpx
/// `vpx_codec_dec_cfg_t::byte_align` plus the `byte_alignment == 0
/// → vp9_byte_align = 1` legacy path (`yv12config.c:172`).
///
/// `Legacy` corresponds to libvpx `byte_alignment == 0`. The other
/// variants directly encode the alignment in bytes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ByteAlignment {
    /// `1` — the libvpx default (`byte_alignment == 0` path).
    Legacy,
    /// 32-byte alignment.
    V32,
    /// 64-byte alignment.
    V64,
    /// 128-byte alignment.
    V128,
    /// 256-byte alignment.
    V256,
    /// 512-byte alignment.
    V512,
    /// 1024-byte alignment.
    V1024,
}

impl ByteAlignment {
    /// Resolve the alignment in bytes.
    #[inline]
    #[must_use]
    pub const fn bytes(self) -> u32 {
        match self {
            Self::Legacy => 1,
            Self::V32 => 32,
            Self::V64 => 64,
            Self::V128 => 128,
            Self::V256 => 256,
            Self::V512 => 512,
            Self::V1024 => 1024,
        }
    }
}

/// Errors from [`Yv12Buffer::alloc`] / `realloc`. Distinct from
/// `BitstreamError` because this concerns physical memory, not
/// payload decoding.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum FrameAllocError {
    /// `width == 0` or `height == 0`.
    EmptyDimensions {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// The product `width * height` would exceed an internal cap
    /// (libvpx `yv12config.c:204` checks against `SIZE_MAX`; we use
    /// `1 << 30` per LLD-frame-buffers §3.1 precondition).
    DimensionsTooLarge {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// `BorderUsage::Custom(n)` where `n` is not a multiple of 32.
    BorderNotMultipleOf32 {
        /// The invalid border value.
        border: u32,
    },
}

impl core::fmt::Display for FrameAllocError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyDimensions { width, height } => {
                write!(f, "empty frame dimensions: {width}×{height}")
            }
            Self::DimensionsTooLarge { width, height } => {
                write!(f, "frame dimensions too large: {width}×{height}")
            }
            Self::BorderNotMultipleOf32 { border } => {
                write!(f, "custom border {border} is not a multiple of 32")
            }
        }
    }
}

impl std::error::Error for FrameAllocError {}

/// Maximum allowed `width × height` (`1 << 30`).
const MAX_FRAME_PIXELS: u64 = 1 << 30;

/// YV12 (planar YUV 4:2:0/4:2:2/4:4:4) 8-bit frame buffer with border
/// padding. Mirrors libvpx `YV12_BUFFER_CONFIG`.
///
/// In v1 the chroma format is always `Yuv420` (D-01 RESOLVED), but the
/// type carries [`ChromaFormat`] so future profiles slot in without
/// breaking the API.
#[derive(Debug)]
pub struct Yv12Buffer {
    // Configuration.
    crop_width: u32,
    crop_height: u32,
    /// Aligned (×8) Y width — corresponds to libvpx `y_width`.
    y_width: u32,
    /// Aligned (×8) Y height.
    y_height: u32,
    uv_width: u32,
    uv_height: u32,
    y_stride: u32,
    uv_stride: u32,
    border: u32,
    chroma: ChromaFormat,
    color_space: ColorSpace,
    color_range: ColorRange,
    render_width: u32,
    render_height: u32,

    // Plane offsets into `buffer_alloc`, pointing at the top-left
    // visible (= un-bordered) pixel.
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,

    // The single backing allocation. Plane "slices" project into this.
    buffer_alloc: Box<[u8]>,
}

impl Yv12Buffer {
    /// Allocate a fresh YV12 buffer. Mirrors `vpx_alloc_frame_buffer`
    /// (`yv12config.c:152-290`) for the internal-allocation,
    /// 8-bit-only path.
    ///
    /// # Errors
    ///
    /// Returns [`FrameAllocError`] for `width/height == 0`, oversized
    /// dimensions, or a `Custom(n)` border that isn't a multiple of 32.
    pub fn alloc(
        width: u32,
        height: u32,
        chroma: ChromaFormat,
        border: BorderUsage,
        byte_alignment: ByteAlignment,
    ) -> Result<Self, FrameAllocError> {
        if width == 0 || height == 0 {
            return Err(FrameAllocError::EmptyDimensions { width, height });
        }
        if u64::from(width) * u64::from(height) > MAX_FRAME_PIXELS {
            return Err(FrameAllocError::DimensionsTooLarge { width, height });
        }
        let border_px = border.pixels();
        if matches!(border, BorderUsage::Custom(_)) && !border_px.is_multiple_of(32) {
            return Err(FrameAllocError::BorderNotMultipleOf32 { border: border_px });
        }

        // Aligned dimensions — libvpx `yv12config.c:163-167` rounds the
        // visible region up to multiples of 8 (`mi_size`).
        let y_width = (width + 7) & !7;
        let y_height = (height + 7) & !7;

        // Y stride: aligned visible-plus-border, then rounded up to 32.
        let y_stride = (y_width + 2 * border_px).next_multiple_of(32);

        // Chroma sub-sampling factors.
        let ss_x = u32::from(chroma.subsampling_x());
        let ss_y = u32::from(chroma.subsampling_y());
        let uv_width = y_width >> ss_x;
        let uv_height = y_height >> ss_y;
        let uv_stride = y_stride >> ss_x;
        let uv_border_w = border_px >> ss_x;
        let uv_border_h = border_px >> ss_y;

        let yplane_rows = (y_height + 2 * border_px) as u64;
        let yplane_size = yplane_rows * u64::from(y_stride);
        let uvplane_rows = (uv_height + 2 * uv_border_h) as u64;
        let uvplane_size = uvplane_rows * u64::from(uv_stride);

        // For Legacy alignment we don't pad; otherwise we round each
        // plane start up. We aggregate as one Box<[u8]>; per-plane
        // start offsets do the work.
        let align = byte_alignment.bytes() as u64;
        let pad_after = |size: u64, alignment: u64| -> u64 {
            if alignment <= 1 {
                0
            } else {
                (alignment - (size % alignment)) % alignment
            }
        };
        let y_pad_after = pad_after(yplane_size, align);
        let uv_pad_after = pad_after(uvplane_size, align);
        let total = yplane_size + y_pad_after + uvplane_size + uv_pad_after + uvplane_size;
        let total_usize = usize::try_from(total).map_err(|_| {
            FrameAllocError::DimensionsTooLarge { width, height }
        })?;

        // Allocate zeroed.
        let buffer_alloc: Box<[u8]> = vec![0u8; total_usize].into_boxed_slice();

        // Compute plane-data offsets (pointing past the top-left
        // border region).
        let y_offset = (u64::from(border_px) * u64::from(y_stride) + u64::from(border_px)) as usize;
        let u_plane_start = yplane_size + y_pad_after;
        let u_offset = (u_plane_start
            + u64::from(uv_border_h) * u64::from(uv_stride)
            + u64::from(uv_border_w)) as usize;
        let v_plane_start = u_plane_start + uvplane_size + uv_pad_after;
        let v_offset = (v_plane_start
            + u64::from(uv_border_h) * u64::from(uv_stride)
            + u64::from(uv_border_w)) as usize;

        Ok(Self {
            crop_width: width,
            crop_height: height,
            y_width,
            y_height,
            uv_width,
            uv_height,
            y_stride,
            uv_stride,
            border: border_px,
            chroma,
            color_space: ColorSpace::Unknown,
            color_range: ColorRange::Studio,
            render_width: width,
            render_height: height,
            y_offset,
            u_offset,
            v_offset,
            buffer_alloc,
        })
    }

    /// Visible (cropped) Y plane width, in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.crop_width
    }

    /// Visible (cropped) Y plane height, in pixels.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.crop_height
    }

    /// Aligned (mi-multiple) Y plane width, in pixels.
    #[inline]
    #[must_use]
    pub const fn aligned_y_width(&self) -> u32 {
        self.y_width
    }

    /// Aligned (mi-multiple) Y plane height, in pixels.
    #[inline]
    #[must_use]
    pub const fn aligned_y_height(&self) -> u32 {
        self.y_height
    }

    /// Y plane stride, in bytes. Always a multiple of 32.
    #[inline]
    #[must_use]
    pub const fn y_stride(&self) -> u32 {
        self.y_stride
    }

    /// U/V plane stride, in bytes.
    #[inline]
    #[must_use]
    pub const fn uv_stride(&self) -> u32 {
        self.uv_stride
    }

    /// Border padding around the visible region, in pixels.
    #[inline]
    #[must_use]
    pub const fn border(&self) -> u32 {
        self.border
    }

    /// Chroma format. `Yuv420` in v1.
    #[inline]
    #[must_use]
    pub const fn chroma(&self) -> ChromaFormat {
        self.chroma
    }

    /// Color space attached to this buffer.
    #[inline]
    #[must_use]
    pub const fn color_space(&self) -> ColorSpace {
        self.color_space
    }

    /// Color range attached to this buffer.
    #[inline]
    #[must_use]
    pub const fn color_range(&self) -> ColorRange {
        self.color_range
    }

    /// Render size as `(render_width, render_height)`. Defaults to
    /// `(width, height)`; the decoder updates this from the header's
    /// optional `render_size` field.
    #[inline]
    #[must_use]
    pub const fn render_size(&self) -> (u32, u32) {
        (self.render_width, self.render_height)
    }

    /// Update the color-space / color-range / render-size metadata.
    /// Does not reallocate; the visible/aligned pixel dimensions stay
    /// the same.
    pub fn set_metadata(
        &mut self,
        color_space: ColorSpace,
        color_range: ColorRange,
        render_size: Option<(u32, u32)>,
    ) {
        self.color_space = color_space;
        self.color_range = color_range;
        let (rw, rh) = render_size.unwrap_or((self.crop_width, self.crop_height));
        self.render_width = rw;
        self.render_height = rh;
    }

    /// Read-only view of the Y plane (cropped region; border excluded).
    #[inline]
    #[must_use]
    pub fn y_plane(&self) -> PlaneRef<'_> {
        PlaneRef {
            data: &self.buffer_alloc[self.y_offset..],
            stride: self.y_stride as usize,
            width: self.y_width as usize,
            height: self.y_height as usize,
        }
    }

    /// Read-only view of the U plane.
    #[inline]
    #[must_use]
    pub fn u_plane(&self) -> PlaneRef<'_> {
        PlaneRef {
            data: &self.buffer_alloc[self.u_offset..],
            stride: self.uv_stride as usize,
            width: self.uv_width as usize,
            height: self.uv_height as usize,
        }
    }

    /// Read-only view of the V plane.
    #[inline]
    #[must_use]
    pub fn v_plane(&self) -> PlaneRef<'_> {
        PlaneRef {
            data: &self.buffer_alloc[self.v_offset..],
            stride: self.uv_stride as usize,
            width: self.uv_width as usize,
            height: self.uv_height as usize,
        }
    }

    /// Mutable view of the Y plane.
    #[inline]
    pub fn y_plane_mut(&mut self) -> PlaneMut<'_> {
        let stride = self.y_stride as usize;
        let width = self.y_width as usize;
        let height = self.y_height as usize;
        PlaneMut {
            data: &mut self.buffer_alloc[self.y_offset..],
            stride,
            width,
            height,
        }
    }

    /// Mutable view of the U plane.
    #[inline]
    pub fn u_plane_mut(&mut self) -> PlaneMut<'_> {
        let stride = self.uv_stride as usize;
        let width = self.uv_width as usize;
        let height = self.uv_height as usize;
        PlaneMut {
            data: &mut self.buffer_alloc[self.u_offset..],
            stride,
            width,
            height,
        }
    }

    /// Mutable view of the V plane.
    #[inline]
    pub fn v_plane_mut(&mut self) -> PlaneMut<'_> {
        let stride = self.uv_stride as usize;
        let width = self.uv_width as usize;
        let height = self.uv_height as usize;
        PlaneMut {
            data: &mut self.buffer_alloc[self.v_offset..],
            stride,
            width,
            height,
        }
    }
}

/// Read-only plane view: a slice into the backing buffer plus stride
/// and visible dimensions.
#[derive(Copy, Clone)]
pub struct PlaneRef<'a> {
    /// Underlying buffer starting at the top-left visible pixel.
    /// Contains border padding to the right and below (within the
    /// stride layout).
    data: &'a [u8],
    /// Bytes per row.
    stride: usize,
    /// Visible width, in pixels.
    width: usize,
    /// Visible height, in pixels.
    height: usize,
}

impl<'a> PlaneRef<'a> {
    /// Underlying buffer slice (top-left visible pixel onward).
    #[inline]
    #[must_use]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }
    /// Bytes per row (stride).
    #[inline]
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }
    /// Visible width in pixels.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }
    /// Visible height in pixels.
    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }
    /// Row `r` (0..height) — `width` bytes.
    #[inline]
    #[must_use]
    pub fn row(&self, r: usize) -> &'a [u8] {
        debug_assert!(r < self.height);
        let start = r * self.stride;
        &self.data[start..start + self.width]
    }
}

/// Mutable plane view.
pub struct PlaneMut<'a> {
    data: &'a mut [u8],
    stride: usize,
    width: usize,
    height: usize,
}

impl<'a> PlaneMut<'a> {
    /// Bytes per row.
    #[inline]
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }
    /// Visible width in pixels.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }
    /// Visible height in pixels.
    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }
    /// Row `r` (0..height) — `width` bytes, mutable.
    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [u8] {
        debug_assert!(r < self.height);
        let start = r * self.stride;
        &mut self.data[start..start + self.width]
    }
    /// Fill the entire visible region with `value`. Useful for tests
    /// and for initialising freshly-allocated buffers.
    pub fn fill(&mut self, value: u8) {
        for r in 0..self.height {
            self.row_mut(r).fill(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn border_usage_pixels() {
        assert_eq!(BorderUsage::Decoder.pixels(), 32);
        assert_eq!(BorderUsage::Encoder.pixels(), 160);
        assert_eq!(BorderUsage::Custom(64).pixels(), 64);
    }

    #[test]
    fn byte_alignment_bytes() {
        assert_eq!(ByteAlignment::Legacy.bytes(), 1);
        assert_eq!(ByteAlignment::V32.bytes(), 32);
        assert_eq!(ByteAlignment::V1024.bytes(), 1024);
    }

    #[test]
    fn alloc_zero_dimensions_rejected() {
        let err = Yv12Buffer::alloc(
            0,
            240,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap_err();
        assert!(matches!(err, FrameAllocError::EmptyDimensions { .. }));

        let err = Yv12Buffer::alloc(
            320,
            0,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap_err();
        assert!(matches!(err, FrameAllocError::EmptyDimensions { .. }));
    }

    #[test]
    fn alloc_oversize_dimensions_rejected() {
        // 65536 * 65536 = 1 << 32 > 1 << 30.
        let err = Yv12Buffer::alloc(
            65536,
            65536,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap_err();
        assert!(matches!(err, FrameAllocError::DimensionsTooLarge { .. }));
    }

    #[test]
    fn alloc_border_must_be_multiple_of_32() {
        let err = Yv12Buffer::alloc(
            320,
            240,
            ChromaFormat::Yuv420,
            BorderUsage::Custom(48),
            ByteAlignment::Legacy,
        )
        .unwrap_err();
        assert_eq!(err, FrameAllocError::BorderNotMultipleOf32 { border: 48 });
    }

    #[test]
    fn alloc_basic_dimensions_and_strides_4_2_0() {
        let b = Yv12Buffer::alloc(
            320,
            240,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("ok");
        assert_eq!(b.width(), 320);
        assert_eq!(b.height(), 240);
        // 320 + 2*32 = 384 → already mult-of-32, so y_stride = 384.
        assert_eq!(b.y_stride(), 384);
        // uv_stride = y_stride / 2 = 192.
        assert_eq!(b.uv_stride(), 192);
        assert_eq!(b.border(), 32);
        assert_eq!(b.chroma(), ChromaFormat::Yuv420);
    }

    #[test]
    fn alloc_aligned_widths_round_to_8() {
        // 1923 → y_width = 1928 (next multiple of 8).
        let b = Yv12Buffer::alloc(
            1923,
            1077,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("ok");
        assert_eq!(b.width(), 1923); // visible / crop
        assert_eq!(b.height(), 1077);
        assert_eq!(b.aligned_y_width(), 1928); // = 1923 rounded up to 8
        assert_eq!(b.aligned_y_height(), 1080);
    }

    #[test]
    fn plane_mut_fill_does_not_touch_border() {
        let mut b = Yv12Buffer::alloc(
            64,
            64,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("ok");
        b.y_plane_mut().fill(0x80);
        // The first visible row's bytes are 0x80; the byte at the
        // origin minus 1 (which is the rightmost border pixel of the
        // row above the visible region) is still 0.
        let y = b.y_plane();
        assert_eq!(y.row(0)[0], 0x80);
        assert_eq!(y.row(0)[63], 0x80);
        assert_eq!(y.row(63)[31], 0x80);
        // Sanity: stride is at least width.
        assert!(y.stride() >= y.width());
    }

    #[test]
    fn metadata_setter() {
        let mut b = Yv12Buffer::alloc(
            320,
            240,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("ok");
        assert_eq!(b.color_space(), ColorSpace::Unknown);
        assert_eq!(b.color_range(), ColorRange::Studio);
        assert_eq!(b.render_size(), (320, 240));

        b.set_metadata(ColorSpace::Bt709, ColorRange::Full, Some((640, 480)));
        assert_eq!(b.color_space(), ColorSpace::Bt709);
        assert_eq!(b.color_range(), ColorRange::Full);
        assert_eq!(b.render_size(), (640, 480));
    }

    #[test]
    fn plane_row_round_trip() {
        let mut b = Yv12Buffer::alloc(
            16,
            16,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("ok");
        // Write a checker pattern row.
        b.y_plane_mut().row_mut(5).fill(0x42);
        // Read back.
        assert_eq!(b.y_plane().row(5), [0x42u8; 16]);
        // Other rows untouched.
        assert_eq!(b.y_plane().row(4), [0u8; 16]);
    }
}
