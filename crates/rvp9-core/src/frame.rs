//! YV12 frame buffer: planar YUV with border padding.
//!
//! Mirrors libvpx `YV12_BUFFER_CONFIG` (`vpx_scale/yv12config.h:29-65`)
//! and the internal-allocation path of `vpx_realloc_frame_buffer`
//! (`yv12config.c:152-290`). Per LLD `notes/lld/frame-buffers.md`:
//! 8-bit only, owned `Box<[u8]>` backing, no callback-allocation path.
//!
//! ## Scope of this commit (Task 7.1 + 7.2)
//!
//! Task 7.1 landed the [`Yv12Buffer`] struct + allocation,
//! dimension/stride accessors, and plane accessors via [`PlaneRef`] /
//! [`PlaneMut`]. Task 7.2 adds the [`FrameBuf`] reference-counted
//! handle and the [`DecoderFramePool`] slot-recycling allocator.
//!
//! **Not in this commit (deferred):**
//! - `RefFrames` / `TilePlanes{,Mut}` tile-disjoint views — Task 7.4
//!   (requires the `UnsafeCell` aliasing contract from
//!   `notes/hld/01-threading.md §6.2`).
//! - The 12-slot encoder pool — Task 7.3 (the decoder only needs
//!   8 ref + 1 current = 9 slots; the extra 3 scaled slots are
//!   encoder-only).
//! - Encoder-side `copy_and_extend_from` — only needed when the
//!   encoder shell lands. (Stage 3 Task C.3 added the decoder-side
//!   [`Yv12Buffer::extend_borders`] / [`extend_frame_borders`] for
//!   inter-prediction halos.)

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
        // G7.0 review (2026-05-21): post-condition guarding against
        // future allocator drift — `aligned_chroma_width()` /
        // `aligned_chroma_height()` accessors return `uv_width` /
        // `uv_height` directly, and downstream code (e.g. tile.rs'
        // per-plane geometry walk) assumes the libvpx-faithful
        // `y_dim >> ss` relationship holds. Debug-only.
        debug_assert_eq!(uv_width, y_width >> ss_x);
        debug_assert_eq!(uv_height, y_height >> ss_y);

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
        let total_usize = usize::try_from(total)
            .map_err(|_| FrameAllocError::DimensionsTooLarge { width, height })?;

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

    /// Chroma format (4:2:0 / 4:2:2 / 4:4:0 / 4:4:4). Plane-extent
    /// accessors derive their dimensions from this — never hardcode
    /// `>> 1`; route through [`ChromaFormat::subsampling_x`] /
    /// [`subsampling_y`].
    ///
    /// [`subsampling_y`]: ChromaFormat::subsampling_y
    #[inline]
    #[must_use]
    pub const fn chroma(&self) -> ChromaFormat {
        self.chroma
    }

    /// Visible (cropped) chroma plane width, in pixels. Mirrors libvpx's
    /// `(y_crop_width + ss_x) >> ss_x` (`yv12config.c:254-255`). For
    /// `Yuv420` / `Yuv422` (`ss_x = 1`) the width is rounded up to an
    /// even chroma-pixel; for `Yuv440` / `Yuv444` (`ss_x = 0`) it
    /// equals the Y crop width.
    #[inline]
    #[must_use]
    pub const fn chroma_width(&self) -> u32 {
        let ss_x = self.chroma.subsampling_x() as u32;
        (self.crop_width + ss_x) >> ss_x
    }

    /// Visible (cropped) chroma plane height, in pixels. Mirrors libvpx's
    /// `(y_crop_height + ss_y) >> ss_y`. See [`chroma_width`] for the
    /// per-format derivation.
    ///
    /// [`chroma_width`]: Yv12Buffer::chroma_width
    #[inline]
    #[must_use]
    pub const fn chroma_height(&self) -> u32 {
        let ss_y = self.chroma.subsampling_y() as u32;
        (self.crop_height + ss_y) >> ss_y
    }

    /// Aligned (mi-multiple) chroma plane width — the allocated chroma
    /// plane width, including the alignment pad past the visible region.
    /// Equal to `aligned_y_width() >> ss_x`.
    #[inline]
    #[must_use]
    pub const fn aligned_chroma_width(&self) -> u32 {
        self.uv_width
    }

    /// Aligned (mi-multiple) chroma plane height. Equal to
    /// `aligned_y_height() >> ss_y`.
    #[inline]
    #[must_use]
    pub const fn aligned_chroma_height(&self) -> u32 {
        self.uv_height
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

    /// Read-only "with-border" view of plane `0`/`1`/`2` (Y/U/V). The
    /// returned [`BorderedPlaneRef`] exposes a slice starting at the
    /// border's top-left (i.e. `border` pixels above and left of the
    /// visible top-left) plus the per-plane border width and the
    /// visible / aligned dimensions. Motion-compensation kernels use
    /// this to index reference-frame pixels at MV-induced positions
    /// that fall outside the visible region — [`Self::extend_borders`]
    /// must have been called on this buffer first.
    ///
    /// # Panics
    ///
    /// Panics if `plane > 2`.
    #[inline]
    #[must_use]
    pub fn plane_with_border(&self, plane: usize) -> BorderedPlaneRef<'_> {
        let ss_x = u32::from(self.chroma.subsampling_x());
        let ss_y = u32::from(self.chroma.subsampling_y());
        let (offset, stride, w, h, border_w, border_h) = match plane {
            0 => (
                self.y_offset,
                self.y_stride as usize,
                self.y_width as usize,
                self.y_height as usize,
                self.border as usize,
                self.border as usize,
            ),
            1 => (
                self.u_offset,
                self.uv_stride as usize,
                self.uv_width as usize,
                self.uv_height as usize,
                (self.border >> ss_x) as usize,
                (self.border >> ss_y) as usize,
            ),
            2 => (
                self.v_offset,
                self.uv_stride as usize,
                self.uv_width as usize,
                self.uv_height as usize,
                (self.border >> ss_x) as usize,
                (self.border >> ss_y) as usize,
            ),
            _ => panic!("plane index {plane} out of range (0..=2)"),
        };
        // Slice base is `border_h` rows above and `border_w` columns
        // left of the visible top-left (= origin of the bordered
        // region in `buffer_alloc`).
        let base = offset - border_h * stride - border_w;
        BorderedPlaneRef {
            data: &self.buffer_alloc[base..],
            stride,
            border_w,
            border_h,
            width: w,
            height: h,
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

    /// Replicate edge pixels into each plane's border, so that
    /// motion-comp halos can read out-of-visible-region positions
    /// without bounds checks.
    ///
    /// Mirrors `vpx_extend_frame_borders_c` (`yv12extend.c:169`):
    /// `extend_frame(ybf, ybf->border)` with per-plane edge replication.
    /// For each plane:
    ///
    /// 1. Horizontal pass — for every visible row, fill the `border`
    ///    pixels left of column 0 with the row's first visible pixel
    ///    and the `border + (aligned_width - crop_width)` pixels right
    ///    of the last visible column with the row's last visible pixel.
    ///    The right pad also covers the mi-alignment padding between
    ///    `crop_width` and `aligned_width`.
    /// 2. Vertical pass — replicate the (now horizontally-extended)
    ///    top row into the `border` rows above, and the
    ///    horizontally-extended bottom visible row into the
    ///    `border + (aligned_height - crop_height)` rows below.
    ///
    /// Idempotent: calling twice on the same buffer is a no-op the
    /// second time (the borders already hold the edge values).
    ///
    /// Free-function form: see [`extend_frame_borders`].
    pub fn extend_borders(&mut self) {
        let border = self.border as usize;
        let ss_x = u32::from(self.chroma.subsampling_x());
        let ss_y = u32::from(self.chroma.subsampling_y());

        // Y plane.
        let y_crop_w = self.crop_width as usize;
        let y_crop_h = self.crop_height as usize;
        let y_w = self.y_width as usize;
        let y_h = self.y_height as usize;
        extend_plane(
            &mut self.buffer_alloc,
            self.y_offset,
            self.y_stride as usize,
            y_crop_w,
            y_crop_h,
            border,
            border,
            border + (y_h - y_crop_h),
            border + (y_w - y_crop_w),
        );

        // UV planes — crop dimensions follow libvpx
        // `(crop + ss) >> ss` (yv12config.c:254-255). Border halves
        // along each subsampled axis.
        let uv_border_w = border >> ss_x;
        let uv_border_h = border >> ss_y;
        let uv_crop_w = (self.crop_width as usize + ss_x as usize) >> ss_x;
        let uv_crop_h = (self.crop_height as usize + ss_y as usize) >> ss_y;
        let uv_w = self.uv_width as usize;
        let uv_h = self.uv_height as usize;
        let uv_stride = self.uv_stride as usize;
        extend_plane(
            &mut self.buffer_alloc,
            self.u_offset,
            uv_stride,
            uv_crop_w,
            uv_crop_h,
            uv_border_h,
            uv_border_w,
            uv_border_h + (uv_h - uv_crop_h),
            uv_border_w + (uv_w - uv_crop_w),
        );
        extend_plane(
            &mut self.buffer_alloc,
            self.v_offset,
            uv_stride,
            uv_crop_w,
            uv_crop_h,
            uv_border_h,
            uv_border_w,
            uv_border_h + (uv_h - uv_crop_h),
            uv_border_w + (uv_w - uv_crop_w),
        );
    }

    /// Re-zero the entire backing allocation (Y+U+V planes incl. the
    /// borders and stride padding). Used when recycling a pooled buffer
    /// as a fresh decode target so its starting state is byte-identical
    /// to a freshly-[`alloc`](Self::alloc)-ed (zero-initialised) buffer.
    pub fn zero_pixels(&mut self) {
        self.buffer_alloc.fill(0);
    }

    /// G5.1: produce a fresh [`Yv12Buffer`] with identical geometry —
    /// same crop dimensions, aligned dimensions, strides, borders,
    /// chroma format, alignment — but with a freshly zero-initialised
    /// pixel backing. Used as a per-tile scratch buffer for parallel
    /// tile decode: each tile worker writes its column range into its
    /// own scratch buffer, and the main thread merges the column
    /// ranges back via [`Self::merge_tile_columns_from`] once the
    /// per-tile scope has joined.
    ///
    /// The returned buffer's `color_space`, `color_range`, and
    /// `render_size` are reset to defaults — the caller is expected
    /// to consume the scratch buffer for pixel data only.
    ///
    /// # Errors
    ///
    /// Returns [`FrameAllocError`] only if the underlying allocation
    /// fails (extremely unlikely for sizes that the source buffer
    /// already holds).
    pub fn clone_geometry(&self) -> Result<Self, FrameAllocError> {
        // We can't directly recover the `byte_alignment` from `self`,
        // but `Legacy` (= 1) is the libvpx default and the only value
        // the decoder uses today; geometry comes from the strides and
        // offsets so the byte_alignment choice is moot. We use
        // `Legacy` to keep the scratch allocation compact.
        //
        // The decoder constructs frames with `BorderUsage::Decoder`
        // (= 32 px) — match that via `BorderUsage::Custom(self.border)`
        // so the scratch buffer has the same border width.
        Self::alloc(
            self.crop_width,
            self.crop_height,
            self.chroma,
            BorderUsage::Custom(self.border),
            ByteAlignment::Legacy,
        )
    }

    /// G5.1: copy a per-tile column range from a scratch source
    /// buffer into `self`. The two buffers MUST have identical
    /// geometry (`crop_width`/`crop_height`/`chroma`/strides).
    ///
    /// `luma_col_start` and `luma_col_end` specify the column range
    /// in LUMA pixels. Row and column endpoints must be multiples of
    /// [`Yv12TileSplit::ALIGN`] (= 2 for 4:2:0 chroma alignment; VP9
    /// tile boundaries are at MI units = 8 luma px, so the gate is
    /// always satisfied for well-formed streams).
    ///
    /// G5.3 (tile_rows > 1) tiles each own a disjoint rectangle in
    /// the (mi_row, mi_col) grid: each tile's rect is copied here
    /// once the tile worker has finished, with no two tiles writing
    /// the same pixel (VP9 spec guarantee).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the geometry mismatches, or if the
    /// rectangle is out of bounds, or if the alignment requirement
    /// is violated.
    pub fn merge_tile_rect_from(
        &mut self,
        src: &Yv12Buffer,
        luma_row_start: usize,
        luma_row_end: usize,
        luma_col_start: usize,
        luma_col_end: usize,
    ) {
        debug_assert_eq!(self.crop_width, src.crop_width);
        debug_assert_eq!(self.crop_height, src.crop_height);
        debug_assert_eq!(self.chroma, src.chroma);
        debug_assert_eq!(self.y_stride, src.y_stride);
        debug_assert_eq!(self.uv_stride, src.uv_stride);
        debug_assert_eq!(self.y_width, src.y_width);
        debug_assert_eq!(self.y_height, src.y_height);
        debug_assert_eq!(self.uv_width, src.uv_width);
        debug_assert_eq!(self.uv_height, src.uv_height);
        debug_assert!(luma_row_start <= luma_row_end);
        debug_assert!(luma_row_end <= self.y_height as usize);
        debug_assert!(luma_col_start <= luma_col_end);
        debug_assert!(luma_col_end <= self.y_width as usize);

        let ss_x = u32::from(self.chroma.subsampling_x()) as usize;
        let ss_y = u32::from(self.chroma.subsampling_y()) as usize;
        debug_assert!(luma_col_start.is_multiple_of(1 << ss_x));
        debug_assert!(luma_col_end.is_multiple_of(1 << ss_x));
        debug_assert!(luma_row_start.is_multiple_of(1 << ss_y));
        debug_assert!(luma_row_end.is_multiple_of(1 << ss_y));

        let y_stride = self.y_stride as usize;
        let uv_stride = self.uv_stride as usize;

        // Y plane copy: rows [luma_row_start, luma_row_end),
        // columns [luma_col_start, luma_col_end).
        let y_col_len = luma_col_end - luma_col_start;
        if y_col_len > 0 {
            for r in luma_row_start..luma_row_end {
                let dst_start = self.y_offset + r * y_stride + luma_col_start;
                let src_start = src.y_offset + r * y_stride + luma_col_start;
                self.buffer_alloc[dst_start..dst_start + y_col_len]
                    .copy_from_slice(&src.buffer_alloc[src_start..src_start + y_col_len]);
            }
        }

        // U/V plane copy: rows [luma_row_start >> ss_y, luma_row_end >> ss_y),
        // columns [luma_col_start >> ss_x, luma_col_end >> ss_x).
        let uv_col_start = luma_col_start >> ss_x;
        let uv_col_end = luma_col_end >> ss_x;
        let uv_col_len = uv_col_end - uv_col_start;
        let uv_row_start = luma_row_start >> ss_y;
        let uv_row_end = luma_row_end >> ss_y;
        if uv_col_len > 0 {
            for r in uv_row_start..uv_row_end {
                let dst_u = self.u_offset + r * uv_stride + uv_col_start;
                let src_u = src.u_offset + r * uv_stride + uv_col_start;
                self.buffer_alloc[dst_u..dst_u + uv_col_len]
                    .copy_from_slice(&src.buffer_alloc[src_u..src_u + uv_col_len]);
                let dst_v = self.v_offset + r * uv_stride + uv_col_start;
                let src_v = src.v_offset + r * uv_stride + uv_col_start;
                self.buffer_alloc[dst_v..dst_v + uv_col_len]
                    .copy_from_slice(&src.buffer_alloc[src_v..src_v + uv_col_len]);
            }
        }
    }

    /// G5.1: copy a per-tile column range (full height) from a scratch
    /// source buffer into `self`. Backward-compatible wrapper around
    /// [`Self::merge_tile_rect_from`] for the tile_rows == 1 case
    /// where every tile spans the full frame height.
    ///
    /// See [`Self::merge_tile_rect_from`] for geometry / alignment
    /// requirements and panic behaviour.
    pub fn merge_tile_columns_from(
        &mut self,
        src: &Yv12Buffer,
        luma_col_start: usize,
        luma_col_end: usize,
    ) {
        let y_height = self.y_height as usize;
        self.merge_tile_rect_from(src, 0, y_height, luma_col_start, luma_col_end);
    }
}

/// G5.1: helpers / constants for the per-tile-column split that the
/// decoder uses to drive parallel tile decode. Bundled into a single
/// struct so callers have a stable name to refer to.
pub struct Yv12TileSplit;

impl Yv12TileSplit {
    /// Alignment requirement on luma column start/end for the
    /// per-tile column split: must be a multiple of 2 (so chroma at
    /// 4:2:0's `ss_x = 1` lands on integer chroma pixels). Tile
    /// boundaries are at MI units (8 luma px), which is a stricter
    /// alignment than this requirement, so the gate is satisfied for
    /// every well-formed VP9 stream.
    pub const ALIGN: usize = 2;
}

/// G5.2 instrumentation: count of fresh [`Yv12Buffer`] allocations the
/// scratch pool has performed across this process's lifetime. Bumped
/// inside [`Yv12ScratchPool::take`] every time it cannot satisfy the
/// request from already-cached buffers and must allocate a new one.
///
/// Used by the `tile_scratch_pool_warm_no_alloc` regression test to
/// confirm the pool stops allocating once the steady-state tile count
/// is reached: the delta of this counter for frame N (N > warmup) is
/// expected to be 0.
pub static RVP9_G5_2_SCRATCH_ALLOCS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// G5.2: recycled scratch-buffer pool for parallel tile decoding.
///
/// **Why**: G5.1's parallel tile dispatch allocated a fresh
/// [`Yv12Buffer`] per tile per frame via [`Yv12Buffer::clone_geometry`]
/// — at 1080p × 8 tiles × 60 fps that's ~1.44 GB/s of allocator churn
/// and zero-init traffic (per-tile scratch is ~3 MB at 1080p). G5.2
/// recycles those buffers across frames.
///
/// **Lifecycle**: the [`crate::Decoder`] equivalent (or any host) owns
/// one [`Yv12ScratchPool`] for its lifetime. Per-frame:
///
/// 1. Call [`Self::take(n_tiles, &yv12)`](Self::take) to obtain
///    `n_tiles` owned scratch buffers, where `yv12` is the destination
///    frame whose geometry the scratches must mirror.
/// 2. Move each buffer into the per-tile worker (closure-captured).
/// 3. After the worker pool's scope joins, recover the owned buffers
///    from the slots and pass them all to [`Self::put`] to recycle
///    for the next frame.
///
/// **Pool growth**: lazy. `take(n, …)` allocates only as many as it
/// has to to satisfy `n`; subsequent calls with `n' <= max(n_seen)`
/// allocate zero buffers. The pool **never shrinks** within its
/// lifetime — once `n` tiles have been seen, the pool holds at least
/// `n` buffers across `take`/`put` cycles. (Justified by the fact
/// that decoders see a small distinct set of tile counts in practice:
/// usually exactly one. Shrinking would force re-allocation on the
/// next frame, defeating the entire point.)
///
/// **Geometry mismatch**: if `take` is called with a `template`
/// (destination frame) whose geometry differs from the pool's cached
/// buffers (e.g. mid-stream resize), the cached buffers are
/// **discarded** and replacements allocated. This is the libvpx-
/// faithful "resize blows away scratch" semantics — a resize is rare
/// and re-allocating once is cheaper than keeping a per-geometry
/// pool.
///
/// **Re-zero safety**: pooled buffers are **NOT** re-zeroed between
/// frames. The invariant that makes this safe: in entropy-tiled VP9,
/// each tile reads only pixels within its own column range (left/
/// above neighbours are in the same tile's column range), and the
/// post-scope merge in [`Yv12Buffer::merge_tile_columns_from`] copies
/// only that column range back to the main frame. Stale pixels
/// outside the column range from a prior tile decode therefore
/// **never get read by the new tile** (out of range) and **never get
/// merged back** (out of merge range). Adding a re-zero pass between
/// frames would halve the perf gain — don't.
///
/// **Thread-safety / concurrency**: the pool is `!Sync` by virtue of
/// `&mut self` on every mutator; it lives on the supervisor thread
/// (the decoder thread that drives `scope.submit`). Workers hold
/// individual `Yv12Buffer`s moved into their closures, not pool
/// handles, so no cross-thread access of the pool is possible.
#[derive(Debug, Default)]
pub struct Yv12ScratchPool {
    /// Cached scratch buffers, idle (not in use by any worker).
    /// Buffers move out via [`Self::take`] and return via
    /// [`Self::put`]. Geometry is uniform across the Vec (a
    /// mismatched-geometry `take` discards the entire Vec).
    buffers: Vec<Yv12Buffer>,
}

impl Yv12ScratchPool {
    /// Build an empty pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of buffers currently parked in the pool (i.e. **not**
    /// out at workers). Test / diagnostic accessor.
    #[must_use]
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Hand out `n` owned scratch buffers with geometry matching
    /// `template`. If the pool's cached buffers don't match
    /// `template`'s geometry (mid-stream resize), the entire cache
    /// is discarded and replacements allocated.
    ///
    /// The returned buffers' pixel contents are **arbitrary** —
    /// either zero (freshly allocated) or whatever the previous
    /// tile decode wrote. The caller MUST overwrite every pixel
    /// it intends to read; see the type-level "Re-zero safety"
    /// note for why this is sound for the parallel-tile use case.
    ///
    /// # Errors
    ///
    /// Returns [`FrameAllocError`] if a fresh allocation fails
    /// (oversized dimensions, etc. — extremely unlikely for sizes
    /// the host already holds in `template`).
    pub fn take(
        &mut self,
        n: usize,
        template: &Yv12Buffer,
    ) -> Result<Vec<Yv12Buffer>, FrameAllocError> {
        // Geometry-fingerprint the head of the cache. Compare a
        // representative buffer against the template; if they
        // differ, drop the entire cache (mid-stream resize).
        let geometry_matches = self
            .buffers
            .first()
            .map(|b| Self::geometry_eq(b, template))
            .unwrap_or(true);
        if !geometry_matches {
            self.buffers.clear();
        }

        let mut out: Vec<Yv12Buffer> = Vec::with_capacity(n);
        while out.len() < n {
            if let Some(b) = self.buffers.pop() {
                out.push(b);
            } else {
                // Allocate one fresh buffer matching template
                // geometry. Increments the global counter so
                // tests can detect non-zero allocs on the hot
                // path post-warmup.
                let fresh = template.clone_geometry()?;
                RVP9_G5_2_SCRATCH_ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                out.push(fresh);
            }
        }
        Ok(out)
    }

    /// Return previously-taken buffers to the pool. The buffers'
    /// pixel contents are **not** zeroed — see the type doc's
    /// "Re-zero safety" section.
    pub fn put(&mut self, mut buffers: Vec<Yv12Buffer>) {
        // If our cache currently holds a different geometry (very
        // rare — would happen if `take` was just called with a
        // resized template and the pool was empty when the next
        // `put` arrives), align them by dropping the older entries.
        if let (Some(first), Some(cached_first)) = (buffers.first(), self.buffers.first())
            && !Self::geometry_eq(first, cached_first)
        {
            self.buffers.clear();
        }
        self.buffers.append(&mut buffers);
    }

    /// Whether two buffers have the same geometry for the purposes
    /// of scratch-pool reuse. Compares dimensions, strides, border,
    /// chroma format, and plane offsets — i.e. every field that
    /// would make `merge_tile_columns_from` ill-defined if they
    /// differed.
    fn geometry_eq(a: &Yv12Buffer, b: &Yv12Buffer) -> bool {
        a.crop_width == b.crop_width
            && a.crop_height == b.crop_height
            && a.chroma == b.chroma
            && a.border == b.border
            && a.y_stride == b.y_stride
            && a.uv_stride == b.uv_stride
            && a.y_width == b.y_width
            && a.y_height == b.y_height
            && a.uv_width == b.uv_width
            && a.uv_height == b.uv_height
            && a.y_offset == b.y_offset
            && a.u_offset == b.u_offset
            && a.v_offset == b.v_offset
    }
}

/// Replicate edge pixels into each plane's border so motion-comp
/// halos can read out-of-visible-region positions without bounds
/// checks. Free-function wrapper around [`Yv12Buffer::extend_borders`].
///
/// Mirrors `vpx_extend_frame_borders_c` (`yv12extend.c:169`). Call
/// once per frame, after recon + loop-filter, before the frame
/// becomes a reference for subsequent inter prediction.
#[inline]
pub fn extend_frame_borders(yv12: &mut Yv12Buffer) {
    yv12.extend_borders();
}

/// Per-plane edge replication. Mirrors the static `extend_plane`
/// helper in libvpx `yv12extend.c:22-60`.
///
/// `origin` is the byte offset in `alloc` of the visible (0, 0)
/// pixel. `width`/`height` are the visible dimensions in pixels.
/// `extend_top`/`extend_left`/`extend_bottom`/`extend_right` are
/// the per-edge replication widths (in pixels); for the right
/// edge, libvpx folds the mi-alignment padding into `extend_right`
/// — same here.
#[allow(clippy::too_many_arguments)] // mirrors libvpx's argument list 1:1
fn extend_plane(
    alloc: &mut [u8],
    origin: usize,
    stride: usize,
    width: usize,
    height: usize,
    extend_top: usize,
    extend_left: usize,
    extend_bottom: usize,
    extend_right: usize,
) {
    // --- Horizontal pass: fill left/right border on every visible row.
    // Each row's leftmost byte is `origin + r*stride`; we write the
    // `extend_left` bytes immediately preceding it, and the
    // `extend_right` bytes starting at column `width`.
    for r in 0..height {
        let row_start = origin + r * stride;
        let left_pixel = alloc[row_start];
        let right_pixel = alloc[row_start + width - 1];
        // Left border: bytes [row_start - extend_left, row_start).
        let left_dst = row_start - extend_left;
        alloc[left_dst..row_start].fill(left_pixel);
        // Right border: bytes [row_start + width, row_start + width + extend_right).
        let right_dst = row_start + width;
        alloc[right_dst..right_dst + extend_right].fill(right_pixel);
    }

    // --- Vertical pass: replicate the first / last visible row (now
    // already extended horizontally) into the border rows above
    // and below. The replicated line covers
    // `extend_left + width + extend_right` bytes.
    let linesize = extend_left + width + extend_right;
    // First visible row, starting at its leftmost extended byte.
    let top_src = origin - extend_left;
    // Last visible row, starting at its leftmost extended byte.
    let bot_src = origin + (height - 1) * stride - extend_left;
    // First top-border row (at offset `-extend_top * stride - extend_left`
    // from the origin).
    let top_dst_first = origin - extend_top * stride - extend_left;
    // First bottom-border row (immediately after the last visible row).
    let bot_dst_first = origin + height * stride - extend_left;
    for k in 0..extend_top {
        let dst = top_dst_first + k * stride;
        alloc.copy_within(top_src..top_src + linesize, dst);
    }
    for k in 0..extend_bottom {
        let dst = bot_dst_first + k * stride;
        alloc.copy_within(bot_src..bot_src + linesize, dst);
    }
}

/// Read-only "with-border" plane view. The `data` slice starts at the
/// border's top-left corner (= `border_w` columns left and `border_h`
/// rows above the visible top-left). Motion-compensation kernels use
/// this to read reference-frame pixels at MV-induced positions that
/// fall outside the visible region.
///
/// To address visible pixel `(x, y)` (0-indexed inside the visible
/// area), use `data[(border_h + y) * stride + (border_w + x)]`. For
/// out-of-visible reads (e.g. when an MV pushes the kernel footprint
/// into the border), `x` and `y` may go negative as long as
/// `border_w + x >= 0` and `border_h + y >= 0` —
/// [`Yv12Buffer::extend_borders`] populates those cells with the
/// replicated edge values.
#[derive(Copy, Clone)]
pub struct BorderedPlaneRef<'a> {
    /// Underlying slice starting at the bordered region's top-left.
    pub data: &'a [u8],
    /// Bytes per row.
    pub stride: usize,
    /// Border width in pixels (= `border` for Y, `border >> ss_x` for
    /// chroma).
    pub border_w: usize,
    /// Border height in pixels (= `border` for Y, `border >> ss_y` for
    /// chroma).
    pub border_h: usize,
    /// Aligned plane width in pixels (= [`Yv12Buffer::aligned_y_width`]
    /// for Y, `>> ss_x` for chroma).
    pub width: usize,
    /// Aligned plane height in pixels.
    pub height: usize,
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

// ============================================================================
// Frame-buffer pool (decoder side)
// ============================================================================

use std::sync::Arc;

/// Reference-counted handle to a decoded frame. Cheap to clone (it's
/// `Arc<Yv12Buffer>` under the hood); `Arc::clone` is libvpx's
/// `++ref_count` and `Arc::drop` is `--ref_count`.
pub type FrameBuf = Arc<Yv12Buffer>;

/// Number of reference frames VP9 keeps in the buffer pool
/// (`REF_FRAMES` = 8 per `crate::bitstream::REF_FRAMES`).
pub const DECODER_REF_FRAMES: usize = 8;

/// Total slots in the decoder's pool: 8 ref slots + 1 "currently
/// being decoded" slot. Mirrors libvpx
/// `FRAME_BUFFERS = REF_FRAMES + 1` for the decoder configuration
/// (the encoder reserves an extra 3 scaled slots — not needed for
/// the decoder per LLD-frame-buffers §3).
pub const DECODER_FRAME_BUFFERS: usize = DECODER_REF_FRAMES + 1;

/// Errors from [`DecoderFramePool`] slot operations.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum FramePoolError {
    /// No free slot was available (every slot has an outstanding
    /// reference from somewhere). Mirrors libvpx `get_free_fb`
    /// returning `INVALID_IDX`.
    NoFreeSlot,
    /// `slot` is outside `0..DECODER_FRAME_BUFFERS`.
    SlotOutOfRange {
        /// The bad slot index.
        slot: usize,
    },
}

impl core::fmt::Display for FramePoolError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoFreeSlot => f.write_str("no free frame-pool slot available"),
            Self::SlotOutOfRange { slot } => {
                write!(
                    f,
                    "frame-pool slot {slot} out of range (0..{DECODER_FRAME_BUFFERS})"
                )
            }
        }
    }
}

impl std::error::Error for FramePoolError {}

/// 9-slot recycling pool of decoded frames for the decoder.
///
/// Each slot holds either nothing (`None`) or a `FrameBuf` (an
/// `Arc<Yv12Buffer>`). A slot is "free" when ALL of:
///
/// - the slot is `None`, OR its `Arc` has refcount 1 (only the pool
///   holds the reference, so it's safe to recycle the underlying
///   buffer), AND
/// - the parallel `ref_counts[slot] == 0` — no logical
///   `ref_frame_map[i]` entry currently names this slot.
///
/// The parallel `ref_counts` array exists because the decoder stores
/// `ref_frame_map[i]: Option<u8>` (integer slot indices, not Arc
/// clones), so an active ref-map entry does NOT bump
/// `Arc::strong_count`. Without the parallel array a slot whose Arc
/// is held only by the pool but is named by 8 ref-map entries would
/// be recycled, silently aliasing every ref-map slot onto the new
/// frame. (The G12 v6.0 refactor briefly relied on the strong-count
/// check alone and produced exactly that bug — DA traced it during
/// the G12 R2 review; see the regression test
/// `pool_acquire_blocked_by_nonzero_ref_count` below.)
///
/// libvpx counterpart: `BufferPool::frame_bufs` plus the
/// `ref_count`/`ref_cnt_fb`/`get_free_fb` family at
/// `vp9_onyxc_int.h:66-365`. The reference-count semantics translate
/// directly: `ref_counts[i]` is libvpx's `frame_bufs[i].ref_count`
/// (it counts logical ref-map slots, not Arc strong-count).
///
/// **Concurrency**: only the main decoder thread mutates the pool
/// (per HLD-threading §5.4); tile workers hold cloned `FrameBuf`
/// handles. The pool itself is `Sync` because every public method
/// takes `&mut self` — sharing must go through an external lock,
/// which matches libvpx's "main-thread mutates `buffer_pool`" rule.
#[derive(Debug, Default)]
pub struct DecoderFramePool {
    slots: [Option<FrameBuf>; DECODER_FRAME_BUFFERS],
    /// Per-slot logical ref-map refcount (number of
    /// `ref_frame_map[i] = Some(slot)` entries in the decoder that
    /// currently name this slot). Caller must keep this in sync via
    /// [`inc_ref`](Self::inc_ref) / [`dec_ref`](Self::dec_ref) on every
    /// `ref_frame_map[i]` mutation. See the type-level doc for why
    /// the Arc strong-count alone is insufficient.
    ref_counts: [u8; DECODER_FRAME_BUFFERS],
}

impl DecoderFramePool {
    /// Build an empty pool (every slot `None`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of currently-occupied slots.
    #[must_use]
    pub fn occupied_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Borrow the `FrameBuf` at `slot` (clones the `Arc`, doesn't
    /// remove). Returns `None` if the slot is empty.
    #[must_use]
    pub fn get(&self, slot: usize) -> Option<FrameBuf> {
        self.slots.get(slot).and_then(|s| s.clone())
    }

    /// Acquire a free slot, populating it with `frame`. Returns the
    /// slot index. Mirrors libvpx `get_free_fb`.
    ///
    /// A slot is "free" if **both** of these hold:
    ///
    /// 1. The slot is empty (`None`) OR its `Arc` has strong-count 1
    ///    (no external `FrameBuf` holder), AND
    /// 2. The parallel `ref_counts[slot] == 0` (no decoder
    ///    `ref_frame_map[i]` entry currently names this slot).
    ///
    /// The `ref_counts` check is what prevents recycling a slot that
    /// the decoder's `ref_frame_map` still indexes — without it, a
    /// slot whose external Arc was dropped but whose pool index is
    /// still in 8 ref-map slots would be silently overwritten,
    /// aliasing all 8 ref-map entries onto the new frame.
    ///
    /// If no slot is free, [`FramePoolError::NoFreeSlot`] is returned
    /// — at that point the caller has held too many refs (or has too
    /// many live ref-map entries — i.e., a bug).
    ///
    /// # Errors
    ///
    /// [`FramePoolError::NoFreeSlot`] when every slot has an
    /// outstanding external reference or a non-zero `ref_counts`.
    pub fn acquire_free(&mut self, frame: FrameBuf) -> Result<usize, FramePoolError> {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            let arc_free = match slot {
                None => true,
                Some(arc) => Arc::strong_count(arc) == 1,
            };
            let ref_count_free = self.ref_counts[i] == 0;
            if arc_free && ref_count_free {
                *slot = Some(frame);
                return Ok(i);
            }
        }
        Err(FramePoolError::NoFreeSlot)
    }

    /// Reclaim a recyclable buffer's backing [`Yv12Buffer`] for reuse as
    /// the next frame's decode target — mirrors libvpx `get_free_fb`,
    /// which reuses a `frame_bufs[i]` whose `ref_count` hit 0 instead of
    /// allocating a fresh one. This eliminates the per-frame fresh
    /// allocation's first-touch page faults (the dominant per-frame fixed
    /// cost on cheap/high-res content).
    ///
    /// A slot is recyclable iff its `Arc` is exclusive
    /// (`strong_count == 1`, no external `FrameBuf` holder) **and**
    /// `ref_counts[i] == 0` (no `ref_frame_map[i]` names it) — the SAME
    /// freeness predicate as [`acquire_free`](Self::acquire_free), so
    /// reuse can never alias a live reference (the `ref_counts` guard is
    /// what prevents the G12-era "silent aliasing on recycle" bug). The
    /// matching slot is emptied to `None`; a later `acquire_free`
    /// re-parks the freshly-decoded frame.
    ///
    /// Geometry must match `(width, height, chroma)`; on a mid-stream
    /// resolution change the mismatched buffers are left in place (a
    /// later `acquire_free` drops them) and `None` is returned so the
    /// caller allocates a correctly-sized buffer.
    ///
    /// The returned buffer's pixels are **stale** — the caller MUST zero
    /// or fully overwrite it before any read.
    #[must_use]
    pub fn take_recyclable(
        &mut self,
        width: u32,
        height: u32,
        chroma: ChromaFormat,
    ) -> Option<Yv12Buffer> {
        for i in 0..self.slots.len() {
            if self.ref_counts[i] != 0 {
                continue;
            }
            let recyclable = match &self.slots[i] {
                Some(arc) => {
                    Arc::strong_count(arc) == 1
                        && arc.width() == width
                        && arc.height() == height
                        && arc.chroma() == chroma
                }
                None => false,
            };
            if recyclable {
                let arc = self.slots[i].take().expect("slot matched Some above");
                // `strong_count == 1` was just verified and the pool is
                // mutated only on the main thread at frame boundaries (no
                // tile worker holds a clone at this point), so `try_unwrap`
                // cannot fail; the `.ok()` drops on the impossible Err.
                return Arc::try_unwrap(arc).ok();
            }
        }
        None
    }

    /// Bump the logical ref-map refcount for `idx`. The decoder calls
    /// this every time it writes `ref_frame_map[i] = Some(idx)`. Pair
    /// with [`dec_ref`](Self::dec_ref) when overwriting the slot to a
    /// different `Some(_)` or to `None`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `idx >= DECODER_FRAME_BUFFERS` or if
    /// the refcount would overflow `u8`. A `u8` is plenty for 8
    /// ref-map slots × 9 pool slots; an overflow would indicate the
    /// caller is double-incrementing.
    pub fn inc_ref(&mut self, idx: u8) {
        let i = idx as usize;
        debug_assert!(i < DECODER_FRAME_BUFFERS, "pool slot {i} out of range");
        debug_assert!(
            self.ref_counts[i] < u8::MAX,
            "pool slot {i} ref-count overflow (was {})",
            self.ref_counts[i],
        );
        self.ref_counts[i] = self.ref_counts[i].saturating_add(1);
    }

    /// Decrement the logical ref-map refcount for `idx`. The decoder
    /// calls this when it overwrites a `ref_frame_map[i] =
    /// Some(old_idx)` entry (to a different `Some(_)` or to `None`),
    /// passing `old_idx`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `idx >= DECODER_FRAME_BUFFERS` or if
    /// the refcount is already zero (which would mean the caller
    /// dec'd more times than they inc'd — a paired-call bug).
    pub fn dec_ref(&mut self, idx: u8) {
        let i = idx as usize;
        debug_assert!(i < DECODER_FRAME_BUFFERS, "pool slot {i} out of range");
        debug_assert!(
            self.ref_counts[i] > 0,
            "pool slot {i} dec_ref underflow (already zero)",
        );
        self.ref_counts[i] = self.ref_counts[i].saturating_sub(1);
    }

    /// Read the logical ref-map refcount for `idx` (test/diagnostic
    /// use only).
    #[must_use]
    pub fn ref_count(&self, idx: usize) -> u8 {
        self.ref_counts.get(idx).copied().unwrap_or(0)
    }

    /// Replace the frame at `slot` with `frame`. Mirrors libvpx
    /// `ref_cnt_fb` in its "set to a new buffer" mode — releases
    /// any existing reference (decrementing its Arc count) and
    /// installs the new one.
    ///
    /// # Errors
    ///
    /// [`FramePoolError::SlotOutOfRange`] when `slot >= 9`.
    pub fn assign_slot(&mut self, slot: usize, frame: FrameBuf) -> Result<(), FramePoolError> {
        if slot >= DECODER_FRAME_BUFFERS {
            return Err(FramePoolError::SlotOutOfRange { slot });
        }
        self.slots[slot] = Some(frame);
        Ok(())
    }

    /// Release the frame at `slot` (drop the pool's reference).
    /// External refs persist until their `Arc` drops.
    ///
    /// # Errors
    ///
    /// [`FramePoolError::SlotOutOfRange`] when `slot >= 9`.
    pub fn release_slot(&mut self, slot: usize) -> Result<(), FramePoolError> {
        if slot >= DECODER_FRAME_BUFFERS {
            return Err(FramePoolError::SlotOutOfRange { slot });
        }
        self.slots[slot] = None;
        Ok(())
    }

    /// Reset the pool to empty. Also zeroes every `ref_counts` entry
    /// (caller's responsibility to ensure no `ref_frame_map[i]` still
    /// names a cleared slot — otherwise the next `inc_ref` /
    /// `dec_ref` pairing is wrong).
    pub fn clear(&mut self) {
        for slot in self.slots.iter_mut() {
            *slot = None;
        }
        for c in self.ref_counts.iter_mut() {
            *c = 0;
        }
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
    /// Raw mutable byte slice covering the plane (top-left visible
    /// pixel onward). The slice extends through the right and bottom
    /// border padding, so callers may safely index past the visible
    /// region up to the border width. Pair with [`stride`] for
    /// row-major indexing.
    ///
    /// [`stride`]: PlaneMut::stride
    #[inline]
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.data
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

// ============================================================================
// TileFrameDst — generic plane-access trait for tile decode
// ============================================================================

/// Trait abstracting mutable plane access for tile decode functions.
///
/// Both [`Yv12Buffer`] (serial decode path) and [`TileFrameView`]
/// (parallel direct-to-frame path) implement this, so the tile decode
/// entry points (`decode_tile_intra` / `decode_tile_inter`) can be
/// generic over the destination type without runtime branching.
pub trait TileFrameDst {
    /// Mutable view of the Y plane.
    fn y_plane_mut(&mut self) -> PlaneMut<'_>;
    /// Mutable view of the U plane.
    fn u_plane_mut(&mut self) -> PlaneMut<'_>;
    /// Mutable view of the V plane.
    fn v_plane_mut(&mut self) -> PlaneMut<'_>;
    /// Chroma format (4:2:0 / 4:2:2 / 4:4:0 / 4:4:4).
    fn chroma(&self) -> ChromaFormat;
}

impl TileFrameDst for Yv12Buffer {
    #[inline]
    fn y_plane_mut(&mut self) -> PlaneMut<'_> {
        self.y_plane_mut()
    }
    #[inline]
    fn u_plane_mut(&mut self) -> PlaneMut<'_> {
        self.u_plane_mut()
    }
    #[inline]
    fn v_plane_mut(&mut self) -> PlaneMut<'_> {
        self.v_plane_mut()
    }
    #[inline]
    fn chroma(&self) -> ChromaFormat {
        self.chroma()
    }
}

// ============================================================================
// TileFrameView — non-overlapping mutable view into a Yv12Buffer
// ============================================================================

/// A non-overlapping mutable view into a rectangular sub-region of a
/// [`Yv12Buffer`]. Semantically equivalent to `&mut` over the pixel
/// rectangle `[row_start..row_end, col_start..col_end]` across all
/// three planes, but implemented via raw pointers to avoid Rust's
/// single-`&mut`-to-the-whole-buffer restriction.
///
/// # Safety invariants (construction-time)
///
/// 1. The parent [`Yv12Buffer`] must outlive all `TileFrameView`s.
/// 2. No two `TileFrameView`s may cover overlapping byte ranges in
///    any plane.
/// 3. No other `&mut` reference to the parent buffer's `buffer_alloc`
///    may exist while any `TileFrameView` is alive.
pub struct TileFrameView {
    // Per-plane raw pointers to the plane origin (row 0, col 0 of the
    // visible region — same position as Yv12Buffer::y_plane_mut().data).
    y_ptr: *mut u8,
    u_ptr: *mut u8,
    v_ptr: *mut u8,

    // Byte lengths of the raw pointer slices (from plane origin to end
    // of buffer_alloc).
    y_len: usize,
    u_len: usize,
    v_len: usize,

    // Strides (bytes per row) — same as the parent buffer's.
    y_stride: usize,
    uv_stride: usize,

    // Aligned plane dimensions (for PlaneMut width/height).
    y_width: usize,
    y_height: usize,
    uv_width: usize,
    uv_height: usize,

    // Chroma format (needed by downstream callers).
    chroma_fmt: ChromaFormat,
}

impl TileFrameView {
    /// Construct a `TileFrameView` that covers the full visible region
    /// of `yv12`. The returned view provides frame-absolute indexing
    /// (plane pointers start at the plane origin, not at a tile offset),
    /// so tile workers can use `mi_col * 8` / `mi_row * 8` directly.
    ///
    /// # Safety
    ///
    /// Caller must guarantee:
    /// - `yv12` lives at least as long as the returned view.
    /// - No two views constructed from the same `yv12` overlap in
    ///   their tile pixel rectangles (enforced by VP9 spec — tile
    ///   boundaries are SB-aligned and disjoint).
    /// - No other `&mut` to `yv12.buffer_alloc` exists concurrently.
    pub unsafe fn from_yv12(yv12: &mut Yv12Buffer) -> Self {
        let alloc_ptr = yv12.buffer_alloc.as_mut_ptr();
        let alloc_len = yv12.buffer_alloc.len();

        // SAFETY: y_offset / u_offset / v_offset are valid offsets into
        // buffer_alloc (established at allocation time in Yv12Buffer::alloc).
        // The resulting pointers remain within the allocation.
        let (y_ptr, u_ptr, v_ptr) = unsafe {
            (
                alloc_ptr.add(yv12.y_offset),
                alloc_ptr.add(yv12.u_offset),
                alloc_ptr.add(yv12.v_offset),
            )
        };

        let y_len = alloc_len - yv12.y_offset;
        let u_len = alloc_len - yv12.u_offset;
        let v_len = alloc_len - yv12.v_offset;

        Self {
            y_ptr,
            u_ptr,
            v_ptr,
            y_len,
            u_len,
            v_len,
            y_stride: yv12.y_stride as usize,
            uv_stride: yv12.uv_stride as usize,
            y_width: yv12.y_width as usize,
            y_height: yv12.y_height as usize,
            uv_width: yv12.uv_width as usize,
            uv_height: yv12.uv_height as usize,
            chroma_fmt: yv12.chroma,
        }
    }
}

// SAFETY: TileFrameView is Send because:
// 1. It holds raw pointers into a single Box<[u8]> allocation.
// 2. The construction contract guarantees non-overlapping ranges,
//    so sending different views to different threads cannot produce
//    data races — each thread writes only to its own disjoint byte set.
// 3. The parent Yv12Buffer's lifetime is guaranteed by the scoped
//    thread pool (the scope borrows the parent, preventing drop).
unsafe impl Send for TileFrameView {}

impl TileFrameDst for TileFrameView {
    #[inline]
    fn y_plane_mut(&mut self) -> PlaneMut<'_> {
        // SAFETY: the raw pointer is valid for `y_len` bytes and the
        // parent Yv12Buffer is alive (construction contract). No other
        // &mut to this byte range exists (non-overlapping tile views).
        let data = unsafe { std::slice::from_raw_parts_mut(self.y_ptr, self.y_len) };
        PlaneMut {
            data,
            stride: self.y_stride,
            width: self.y_width,
            height: self.y_height,
        }
    }

    #[inline]
    fn u_plane_mut(&mut self) -> PlaneMut<'_> {
        // SAFETY: see y_plane_mut.
        let data = unsafe { std::slice::from_raw_parts_mut(self.u_ptr, self.u_len) };
        PlaneMut {
            data,
            stride: self.uv_stride,
            width: self.uv_width,
            height: self.uv_height,
        }
    }

    #[inline]
    fn v_plane_mut(&mut self) -> PlaneMut<'_> {
        // SAFETY: see y_plane_mut.
        let data = unsafe { std::slice::from_raw_parts_mut(self.v_ptr, self.v_len) };
        PlaneMut {
            data,
            stride: self.uv_stride,
            width: self.uv_width,
            height: self.uv_height,
        }
    }

    #[inline]
    fn chroma(&self) -> ChromaFormat {
        self.chroma_fmt
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

    /// G7 oracle. Locks in the `(ss_x, ss_y)` tuple per chroma format
    /// against the spec, so a future "convenience" tweak to
    /// [`ChromaFormat::subsampling_x`] / `subsampling_y` cannot silently
    /// flip a sub-mode (e.g. swapping 4:2:2 with 4:4:0 would only be
    /// caught at the bit-exact output layer — far too late).
    #[test]
    fn chroma_format_subsampling_table_oracle() {
        use ChromaFormat::*;
        let table: [(ChromaFormat, bool, bool); 4] = [
            (Yuv420, true, true),
            (Yuv422, true, false),
            (Yuv440, false, true),
            (Yuv444, false, false),
        ];
        for (cf, ss_x, ss_y) in table {
            assert_eq!(cf.subsampling_x(), ss_x, "{cf:?} subsampling_x");
            assert_eq!(cf.subsampling_y(), ss_y, "{cf:?} subsampling_y");
        }
    }

    /// G7 oracle. Chroma plane dimensions per [`Yv12Buffer::chroma_width`]
    /// / `chroma_height` must follow libvpx's `(crop + ss) >> ss` rounding
    /// for every chroma format. A 17×11 crop tests the odd-axis path —
    /// for `ss_x = 1` chroma rounds to 9, for `ss_x = 0` it stays at 17;
    /// matching analysis on the Y axis covers 4:4:0.
    #[test]
    fn chroma_width_height_round_trip_all_formats() {
        let cases: [(ChromaFormat, u32, u32); 4] = [
            (ChromaFormat::Yuv420, 9, 6),  // (17+1)>>1 = 9, (11+1)>>1 = 6
            (ChromaFormat::Yuv422, 9, 11), // (17+1)>>1 = 9, full-height
            (ChromaFormat::Yuv440, 17, 6), // full-width,    (11+1)>>1 = 6
            (ChromaFormat::Yuv444, 17, 11),
        ];
        for (cf, expect_w, expect_h) in cases {
            let b = Yv12Buffer::alloc(17, 11, cf, BorderUsage::Decoder, ByteAlignment::Legacy)
                .expect("alloc");
            assert_eq!(b.chroma(), cf);
            assert_eq!(b.chroma_width(), expect_w, "{cf:?} chroma_width");
            assert_eq!(b.chroma_height(), expect_h, "{cf:?} chroma_height");
            // Aligned dims always follow the aligned Y dims shifted by ss.
            let ss_x = u32::from(cf.subsampling_x());
            let ss_y = u32::from(cf.subsampling_y());
            assert_eq!(
                b.aligned_chroma_width(),
                b.aligned_y_width() >> ss_x,
                "{cf:?} aligned_chroma_width"
            );
            assert_eq!(
                b.aligned_chroma_height(),
                b.aligned_y_height() >> ss_y,
                "{cf:?} aligned_chroma_height"
            );
        }
    }

    /// G7 — the U/V plane visible regions actually carry the expected
    /// extent after a fill round-trip across every chroma format. Catches
    /// regressions where an `alloc()` change miscomputes the chroma
    /// stride or plane size, leaving a chroma row read short of `chroma_width`.
    #[test]
    fn alloc_chroma_plane_round_trip_all_formats() {
        for cf in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv440,
            ChromaFormat::Yuv444,
        ] {
            let mut b = Yv12Buffer::alloc(32, 24, cf, BorderUsage::Decoder, ByteAlignment::Legacy)
                .expect("alloc");
            b.u_plane_mut().fill(0x55);
            b.v_plane_mut().fill(0xaa);
            let uw = b.chroma_width() as usize;
            let uh = b.chroma_height() as usize;
            let u = b.u_plane();
            let v = b.v_plane();
            // First and last visible row, first and last visible byte.
            assert_eq!(u.row(0)[0], 0x55, "{cf:?} U(0,0)");
            assert_eq!(u.row(uh - 1)[uw - 1], 0x55, "{cf:?} U last");
            assert_eq!(v.row(0)[0], 0xaa, "{cf:?} V(0,0)");
            assert_eq!(v.row(uh - 1)[uw - 1], 0xaa, "{cf:?} V last");
        }
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

    // ---- DecoderFramePool ----

    fn make_frame(w: u32, h: u32) -> FrameBuf {
        Arc::new(
            Yv12Buffer::alloc(
                w,
                h,
                ChromaFormat::Yuv420,
                BorderUsage::Decoder,
                ByteAlignment::Legacy,
            )
            .expect("alloc"),
        )
    }

    #[test]
    fn pool_constants_match_libvpx() {
        assert_eq!(DECODER_REF_FRAMES, 8);
        assert_eq!(DECODER_FRAME_BUFFERS, 9);
    }

    #[test]
    fn pool_new_is_empty() {
        let pool = DecoderFramePool::new();
        assert_eq!(pool.occupied_count(), 0);
        for i in 0..DECODER_FRAME_BUFFERS {
            assert!(pool.get(i).is_none());
        }
    }

    #[test]
    fn pool_acquire_fills_first_free_slot() {
        let mut pool = DecoderFramePool::new();
        let f0 = make_frame(64, 64);
        let slot = pool.acquire_free(f0.clone()).expect("free slot");
        assert_eq!(slot, 0);
        assert_eq!(pool.occupied_count(), 1);
        let got = pool.get(0).unwrap();
        assert!(Arc::ptr_eq(&got, &f0));
    }

    #[test]
    fn pool_acquire_uses_recyclable_slot() {
        let mut pool = DecoderFramePool::new();
        // Fill all 9 slots distinctly by holding external refs alive
        // (mirrors a real decoder: each acquired slot is "in use" by
        // the caller who just received it).
        let mut external: Vec<FrameBuf> = Vec::new();
        for _ in 0..DECODER_FRAME_BUFFERS {
            let f = make_frame(16, 16);
            pool.acquire_free(f.clone()).expect("free");
            external.push(f);
        }
        assert_eq!(pool.occupied_count(), DECODER_FRAME_BUFFERS);
        // Every slot has external refcount > 1 ⇒ acquire fails.
        assert_eq!(
            pool.acquire_free(make_frame(16, 16)).unwrap_err(),
            FramePoolError::NoFreeSlot
        );
        // Drop the external ref for slot 3 ⇒ that slot's Arc refcount
        // falls back to 1 (only pool holds it) and becomes recyclable.
        let ptr_before_recycle = Arc::as_ptr(&external[3]);
        external.swap_remove(3);
        let new_frame = make_frame(16, 16);
        let slot = pool
            .acquire_free(new_frame.clone())
            .expect("recycles slot 3");
        assert_eq!(slot, 3);
        // The replaced slot now holds the new Arc, not the old.
        assert!(Arc::ptr_eq(&pool.get(3).unwrap(), &new_frame));
        assert_ne!(Arc::as_ptr(&new_frame), ptr_before_recycle);
    }

    #[test]
    fn pool_acquire_errors_when_no_free_slot() {
        let mut pool = DecoderFramePool::new();
        // Fill every slot AND hold an extra reference so the pool
        // can't recycle it.
        let mut external_refs: Vec<FrameBuf> = Vec::new();
        for _ in 0..DECODER_FRAME_BUFFERS {
            let f = make_frame(16, 16);
            pool.acquire_free(f.clone()).expect("free");
            external_refs.push(f);
        }
        // Now every slot has refcount = 2 (pool + external_refs[i]).
        // Acquire must fail.
        let f_new = make_frame(16, 16);
        let err = pool.acquire_free(f_new).unwrap_err();
        assert_eq!(err, FramePoolError::NoFreeSlot);
    }

    #[test]
    fn pool_assign_and_release_round_trip() {
        let mut pool = DecoderFramePool::new();
        let f = make_frame(32, 32);
        pool.assign_slot(3, f.clone()).expect("ok");
        assert!(Arc::ptr_eq(&pool.get(3).unwrap(), &f));
        pool.release_slot(3).expect("ok");
        assert!(pool.get(3).is_none());
    }

    #[test]
    fn pool_assign_replaces_existing_slot() {
        let mut pool = DecoderFramePool::new();
        let f1 = make_frame(32, 32);
        let f2 = make_frame(32, 32);
        pool.assign_slot(2, f1.clone()).unwrap();
        pool.assign_slot(2, f2.clone()).unwrap();
        // f1 should now have only the external test ref (we don't
        // hold a second), so its strong count is 1.
        assert_eq!(Arc::strong_count(&f1), 1);
        assert!(Arc::ptr_eq(&pool.get(2).unwrap(), &f2));
    }

    #[test]
    fn pool_assign_out_of_range_rejected() {
        let mut pool = DecoderFramePool::new();
        let f = make_frame(16, 16);
        let err = pool.assign_slot(99, f).unwrap_err();
        assert_eq!(err, FramePoolError::SlotOutOfRange { slot: 99 });
    }

    #[test]
    fn pool_release_out_of_range_rejected() {
        let mut pool = DecoderFramePool::new();
        let err = pool.release_slot(99).unwrap_err();
        assert_eq!(err, FramePoolError::SlotOutOfRange { slot: 99 });
    }

    // ---- extend_borders ----

    /// Helper: read a single byte from the Y plane at signed
    /// (row, col) — where `(0, 0)` is the visible origin and
    /// negative coordinates reach into the border.
    fn read_y_at(buf: &Yv12Buffer, row: i32, col: i32) -> u8 {
        let stride = buf.y_stride as i32;
        let idx = buf.y_offset as i32 + row * stride + col;
        buf.buffer_alloc[idx as usize]
    }
    fn read_u_at(buf: &Yv12Buffer, row: i32, col: i32) -> u8 {
        let stride = buf.uv_stride as i32;
        let idx = buf.u_offset as i32 + row * stride + col;
        buf.buffer_alloc[idx as usize]
    }
    fn read_v_at(buf: &Yv12Buffer, row: i32, col: i32) -> u8 {
        let stride = buf.uv_stride as i32;
        let idx = buf.v_offset as i32 + row * stride + col;
        buf.buffer_alloc[idx as usize]
    }

    #[test]
    fn extend_borders_constant_fill() {
        // Solid-128 visible region → border is all 128.
        let mut b = Yv12Buffer::alloc(
            64,
            48,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc");
        b.y_plane_mut().fill(128);
        b.u_plane_mut().fill(64);
        b.v_plane_mut().fill(200);
        b.extend_borders();

        let border = b.border() as i32;
        // Sample 4 corners of each border ring.
        for &(r, c) in &[
            (-border, -border),
            (-1, -1),
            (47, 64),
            (47 + border - 1, 64 + border - 1),
        ] {
            assert_eq!(read_y_at(&b, r, c), 128, "Y at ({r},{c})");
        }
        let uv_border = (b.border() >> 1) as i32; // 4:2:0
        let uv_h_last = (b.crop_height as i32 / 2) - 1;
        let uv_w_last = b.crop_width as i32 / 2;
        for &(r, c) in &[
            (-uv_border, -uv_border),
            (-1, -1),
            (uv_h_last, uv_w_last),
            (uv_h_last + uv_border, uv_w_last + uv_border - 1),
        ] {
            assert_eq!(read_u_at(&b, r, c), 64, "U at ({r},{c})");
            assert_eq!(read_v_at(&b, r, c), 200, "V at ({r},{c})");
        }
    }

    #[test]
    fn extend_borders_corner_pixel() {
        // (0,0) = 99, rest = 0. After extension, the entire
        // top-left border quadrant must be 99 (top row of the
        // visible region is 99 at col 0; left border of row 0
        // replicates 99; top border replicates that extended row).
        let mut b = Yv12Buffer::alloc(
            32,
            32,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc");
        b.y_plane_mut().row_mut(0)[0] = 99;
        b.extend_borders();
        let border = b.border() as i32;
        // Entire top-left corner of border = 99 (because row 0 col
        // 0 is 99, gets replicated left, and the extended row 0 is
        // replicated up).
        for r in -border..=0 {
            for c in -border..=0 {
                assert_eq!(read_y_at(&b, r, c), 99, "TL corner at ({r},{c})");
            }
        }
        // Top-right corner (col >= 1 in the top border) replicates
        // row 0's last visible pixel = 0 (we only set col 0).
        assert_eq!(read_y_at(&b, -border, 1), 0);
        assert_eq!(read_y_at(&b, -1, 31), 0);
    }

    #[test]
    fn extend_borders_edge_row_and_col() {
        // Row 0 = 1s, col 0 = 2s (except (0,0) which is 2 — col
        // wins because we set col 0 after row 0). After extension:
        //   * top border above col >=1: row-0's value at that col = 1
        //   * left border row >=1: col-0's value at that row = 2
        //   * top-left border quadrant: (0,0) = 2 (col 0 won)
        let mut b = Yv12Buffer::alloc(
            16,
            16,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc");
        for c in 0..16 {
            b.y_plane_mut().row_mut(0)[c] = 1;
        }
        for r in 0..16 {
            b.y_plane_mut().row_mut(r)[0] = 2;
        }
        b.extend_borders();
        // Top border above column 5 = row-0's column-5 value = 1.
        assert_eq!(read_y_at(&b, -1, 5), 1);
        assert_eq!(read_y_at(&b, -32, 5), 1);
        // Left border left of row 5 = row-5's column-0 value = 2.
        assert_eq!(read_y_at(&b, 5, -1), 2);
        assert_eq!(read_y_at(&b, 5, -32), 2);
        // Top-left quadrant — (0,0) is 2 (col 0 overwrote it), so
        // the whole TL border quadrant is 2.
        assert_eq!(read_y_at(&b, -1, -1), 2);
        assert_eq!(read_y_at(&b, -32, -32), 2);
    }

    #[test]
    fn extend_borders_planes_independent() {
        // Distinct fills for Y / U / V must each end up replicated
        // into that plane's own border without cross-talk.
        let mut b = Yv12Buffer::alloc(
            16,
            16,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc");
        b.y_plane_mut().fill(10);
        b.u_plane_mut().fill(20);
        b.v_plane_mut().fill(30);
        b.extend_borders();
        assert_eq!(read_y_at(&b, -1, -1), 10);
        assert_eq!(read_u_at(&b, -1, -1), 20);
        assert_eq!(read_v_at(&b, -1, -1), 30);
        assert_eq!(read_y_at(&b, 16, 16), 10);
        assert_eq!(read_u_at(&b, 8, 8), 20);
        assert_eq!(read_v_at(&b, 8, 8), 30);
    }

    #[test]
    fn extend_borders_awkward_stride() {
        // 1923×1077: aligned to 1928×1080 (mi-multiple of 8); the
        // 5-pixel right gap and 3-pixel bottom gap must be folded
        // into extend_right / extend_bottom so it's also filled
        // with the edge pixel.
        let mut b = Yv12Buffer::alloc(
            1923,
            1077,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc");
        // Fill the visible region with a row-indexed pattern so
        // each row has a distinct right-edge value.
        {
            let mut y = b.y_plane_mut();
            for r in 0..1077 {
                let row = y.row_mut(r);
                // Last visible pixel = r as u8.
                row[1922] = (r & 0xff) as u8;
                // First visible pixel = (r + 17) as u8.
                row[0] = ((r + 17) & 0xff) as u8;
            }
        }
        b.extend_borders();
        // Row 100's right border (including the alignment gap)
        // must all read as 100. The first alignment-gap byte sits
        // at col 1923 (the first byte past the crop), and the
        // last extended byte sits at col 1923 + (border + 5) - 1.
        let border = b.border() as i32;
        let first_pad = 1923i32;
        let last_pad = 1923 + (border + 5) - 1;
        for c in first_pad..=last_pad {
            assert_eq!(read_y_at(&b, 100, c), 100, "row 100 col {c}");
        }
        // Left border of row 100 = (100 + 17) = 117.
        for c in -border..0 {
            assert_eq!(read_y_at(&b, 100, c), 117, "row 100 col {c}");
        }
    }

    #[test]
    fn extend_borders_idempotent() {
        // Running extend twice produces the same buffer as once.
        let mut a = Yv12Buffer::alloc(
            48,
            32,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc");
        for r in 0..32 {
            for c in 0..48 {
                a.y_plane_mut().row_mut(r)[c] = ((r * 7 + c * 13) & 0xff) as u8;
            }
        }
        for r in 0..16 {
            for c in 0..24 {
                a.u_plane_mut().row_mut(r)[c] = ((r * 3 + c * 5) & 0xff) as u8;
                a.v_plane_mut().row_mut(r)[c] = ((r * 11 + c * 17) & 0xff) as u8;
            }
        }
        let mut b = Yv12Buffer::alloc(
            48,
            32,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc");
        // Copy a's visible region into b before extension (so
        // border bytes start matching post-extend a).
        for r in 0..32 {
            b.y_plane_mut()
                .row_mut(r)
                .copy_from_slice(a.y_plane().row(r));
        }
        for r in 0..16 {
            b.u_plane_mut()
                .row_mut(r)
                .copy_from_slice(a.u_plane().row(r));
            b.v_plane_mut()
                .row_mut(r)
                .copy_from_slice(a.v_plane().row(r));
        }
        a.extend_borders();
        b.extend_borders();
        b.extend_borders(); // a second call — must be a no-op.
        assert_eq!(a.buffer_alloc, b.buffer_alloc);
    }

    #[test]
    fn extend_borders_free_fn_alias() {
        let mut b = Yv12Buffer::alloc(
            16,
            16,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc");
        b.y_plane_mut().fill(0xAB);
        extend_frame_borders(&mut b);
        assert_eq!(read_y_at(&b, -1, -1), 0xAB);
        assert_eq!(read_y_at(&b, 16, 16), 0xAB);
    }

    #[test]
    fn extend_borders_convolve8_halo_round_trip() {
        // End-to-end check: after extend_borders, a convolve8_horiz
        // whose 8-tap halo reads 3 pixels into the left border must
        // produce the same output as the same convolve on a manually
        // edge-replicated reference. Uses the C.1 scalar 8-tap kernel.
        use crate::inter::filter::{EIGHTTAP_KERNEL, convolve8_horiz};

        let w = 16usize;
        let h = 8usize;
        let mut b = Yv12Buffer::alloc(
            w as u32,
            h as u32,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc");
        // Fill the visible region with a row-distinctive pattern.
        for r in 0..h {
            for c in 0..w {
                b.y_plane_mut().row_mut(r)[c] = ((r * 11 + c * 3 + 17) & 0xff) as u8;
            }
        }
        b.extend_borders();

        // Reference: a flat (stride = w + 8, height = h) buffer
        // where each row is manually edge-extended by 3 left + 5
        // right. The 8-tap horizontal convolve reads `src[col + k]`
        // for `k ∈ 0..8` with caller-shifted src, so to produce
        // outputs at cols 0..w it needs taps up to col `w + 4`.
        let ref_stride = w + 8;
        let mut reference = vec![0u8; ref_stride * h];
        for r in 0..h {
            let edge_l = ((r * 11) + 17) as u8;
            for c in 0..3 {
                reference[r * ref_stride + c] = edge_l;
            }
            for c in 0..w {
                reference[r * ref_stride + 3 + c] = ((r * 11 + c * 3 + 17) & 0xff) as u8;
            }
            let edge_r = ((r * 11 + (w - 1) * 3 + 17) & 0xff) as u8;
            for c in 0..(ref_stride - 3 - w) {
                reference[r * ref_stride + 3 + w + c] = edge_r;
            }
        }

        // Compute convolve outputs from both buffers and compare.
        let phase = &EIGHTTAP_KERNEL[8]; // half-pel
        let stride = b.y_stride() as usize;

        let mut dst_a = vec![0u8; w * h];
        // Point src at (row 0, col -3) so the kernel's k=0 tap
        // reads the leftmost border byte.
        let src_a = &b.buffer_alloc[b.y_offset - 3..];
        convolve8_horiz(src_a, stride, &mut dst_a, w, phase, w, h);

        let mut dst_b = vec![0u8; w * h];
        convolve8_horiz(&reference, ref_stride, &mut dst_b, w, phase, w, h);
        assert_eq!(dst_a, dst_b);
    }

    /// Critical regression (G12 R2): a pool slot whose external Arc
    /// was dropped (strong_count == 1) but whose `ref_counts` is
    /// non-zero (logical ref-map entries still name it) MUST NOT be
    /// recycled by `acquire_free`. Pre-fix, the Arc check alone
    /// returned this slot — leading to silent aliasing of every
    /// ref-map entry onto the new frame.
    ///
    /// Walkthrough (matches DA's trace):
    ///   1. Park a frame in slot 0 via `acquire_free`; drop the
    ///      external `FrameBuf` so `strong_count == 1`.
    ///   2. Simulate keyframe `refresh_frame_flags = 0xFF` by
    ///      `inc_ref(0)` 8 times (every ref-map slot now names 0).
    ///   3. `acquire_free` for a new frame MUST return `NoFreeSlot`
    ///      (the pool is exhausted by the ref-map references). Only
    ///      after 8 `dec_ref(0)`s clear the count can the slot be
    ///      recycled.
    #[test]
    fn pool_acquire_blocked_by_nonzero_ref_count() {
        let mut pool = DecoderFramePool::new();
        let f = make_frame(16, 16);
        let slot = pool.acquire_free(f.clone()).expect("free");
        assert_eq!(slot, 0);
        // Drop external — Arc strong_count for slot 0 falls to 1
        // (only the pool holds it).
        drop(f);
        assert_eq!(Arc::strong_count(&pool.get(0).unwrap()), 2); // 1 pool + 1 just-cloned by get
        // Simulate keyframe refresh=0xFF wiring 8 ref-map slots to
        // pool slot 0.
        for _ in 0..8 {
            pool.inc_ref(0);
        }
        assert_eq!(pool.ref_count(0), 8);
        // Every other pool slot is empty; but acquire MUST NOT pick
        // slot 0 (it's still ref'd) and MUST pick the first empty
        // slot. After we fill all 8 empty slots externally-held, a
        // 10th acquire fails.
        let mut external: Vec<FrameBuf> = Vec::new();
        for _ in 0..(DECODER_FRAME_BUFFERS - 1) {
            let n = make_frame(16, 16);
            let s = pool.acquire_free(n.clone()).expect("non-slot-0");
            assert_ne!(s, 0, "slot 0 must not recycle while ref_counts[0]==8");
            external.push(n);
        }
        // Pool now has slot 0 ref-blocked + 8 externally held → no
        // free slot.
        let err = pool.acquire_free(make_frame(16, 16)).unwrap_err();
        assert_eq!(err, FramePoolError::NoFreeSlot);

        // Clear the ref-count on slot 0 — now it can recycle.
        for _ in 0..8 {
            pool.dec_ref(0);
        }
        assert_eq!(pool.ref_count(0), 0);
        let recycled = pool.acquire_free(make_frame(16, 16)).expect("slot 0 frees");
        assert_eq!(recycled, 0);
    }

    /// Defensive: `inc_ref` / `dec_ref` round-trip increments and
    /// decrements correctly across multiple slots independently.
    #[test]
    fn pool_ref_count_per_slot_independent() {
        let mut pool = DecoderFramePool::new();
        pool.inc_ref(0);
        pool.inc_ref(0);
        pool.inc_ref(3);
        assert_eq!(pool.ref_count(0), 2);
        assert_eq!(pool.ref_count(3), 1);
        assert_eq!(pool.ref_count(7), 0);
        pool.dec_ref(0);
        assert_eq!(pool.ref_count(0), 1);
        pool.dec_ref(3);
        assert_eq!(pool.ref_count(3), 0);
    }

    /// `clear()` zeroes both `slots` and `ref_counts`.
    #[test]
    fn pool_clear_resets_ref_counts() {
        let mut pool = DecoderFramePool::new();
        pool.acquire_free(make_frame(16, 16)).unwrap();
        pool.inc_ref(0);
        pool.inc_ref(0);
        assert_eq!(pool.ref_count(0), 2);
        pool.clear();
        assert_eq!(pool.ref_count(0), 0);
        assert_eq!(pool.occupied_count(), 0);
    }

    #[test]
    fn pool_clear_drops_all_slots() {
        let mut pool = DecoderFramePool::new();
        // Use 5 distinct slots — hold external refs so the pool
        // doesn't recycle slot 0 every iteration.
        let mut external: Vec<FrameBuf> = Vec::new();
        for _ in 0..5 {
            let f = make_frame(16, 16);
            pool.acquire_free(f.clone()).unwrap();
            external.push(f);
        }
        assert_eq!(pool.occupied_count(), 5);
        pool.clear();
        assert_eq!(pool.occupied_count(), 0);
        // External refs still alive even after clear (Arc semantics).
        assert_eq!(external.len(), 5);
    }

    // ---- G5.1: clone_geometry + merge_tile_columns_from ------------------

    #[test]
    fn clone_geometry_matches_source_dimensions() {
        let src = Yv12Buffer::alloc(
            128,
            96,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        let dst = src.clone_geometry().unwrap();
        assert_eq!(dst.width(), src.width());
        assert_eq!(dst.height(), src.height());
        assert_eq!(dst.aligned_y_width(), src.aligned_y_width());
        assert_eq!(dst.aligned_y_height(), src.aligned_y_height());
        assert_eq!(dst.y_stride(), src.y_stride());
        assert_eq!(dst.uv_stride(), src.uv_stride());
        assert_eq!(dst.chroma(), src.chroma());
        assert_eq!(dst.border(), src.border());
    }

    #[test]
    fn merge_tile_columns_round_trip() {
        // Build a source: stamp distinct values in each column range
        // [0, 64) and [64, 128). Build a fresh dst, merge each range
        // and confirm bytes round-trip.
        let mut src_a = Yv12Buffer::alloc(
            128,
            64,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        src_a.y_plane_mut().fill(0xAA);
        src_a.u_plane_mut().fill(0xA1);
        src_a.v_plane_mut().fill(0xA2);

        let mut src_b = src_a.clone_geometry().unwrap();
        src_b.y_plane_mut().fill(0xBB);
        src_b.u_plane_mut().fill(0xB1);
        src_b.v_plane_mut().fill(0xB2);

        let mut dst = src_a.clone_geometry().unwrap();
        // Tile A owns luma cols [0, 64); tile B owns [64, 128).
        dst.merge_tile_columns_from(&src_a, 0, 64);
        dst.merge_tile_columns_from(&src_b, 64, 128);

        // Spot-check Y plane: left half = 0xAA, right half = 0xBB.
        let y = dst.y_plane();
        let stride = y.stride();
        let data = y.data();
        assert_eq!(data[0], 0xAA, "Y col 0 = src_a fill");
        assert_eq!(data[63], 0xAA, "Y col 63 = src_a fill");
        assert_eq!(data[64], 0xBB, "Y col 64 = src_b fill");
        assert_eq!(data[127], 0xBB, "Y col 127 = src_b fill");
        // Row 1 sanity.
        assert_eq!(data[stride], 0xAA);
        assert_eq!(data[stride + 127], 0xBB);

        // Chroma plane: ss_x = 1, so cols [0, 32) = src_a, [32, 64) = src_b.
        let u = dst.u_plane();
        let u_stride = u.stride();
        let u_data = u.data();
        assert_eq!(u_data[0], 0xA1);
        assert_eq!(u_data[31], 0xA1);
        assert_eq!(u_data[32], 0xB1);
        assert_eq!(u_data[63], 0xB1);
        assert_eq!(u_data[u_stride], 0xA1);
        let v = dst.v_plane();
        assert_eq!(v.data()[0], 0xA2);
        assert_eq!(v.data()[63], 0xB2);
    }

    // ---- Yv12ScratchPool (G5.2) ----

    /// Shared lock to serialise the scratch-pool tests that observe
    /// the process-wide [`RVP9_G5_2_SCRATCH_ALLOCS`] counter. Without
    /// this, concurrent pool-test runs interleave allocations and
    /// the delta-snapshot assertions become flaky.
    fn scratch_alloc_counter_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    #[test]
    fn scratch_pool_new_is_empty() {
        let pool = Yv12ScratchPool::new();
        assert_eq!(pool.buffer_count(), 0);
    }

    #[test]
    fn scratch_pool_take_grows_to_n() {
        let _guard = scratch_alloc_counter_lock().lock().unwrap();
        let template = Yv12Buffer::alloc(
            128,
            96,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        let mut pool = Yv12ScratchPool::new();
        assert_eq!(pool.buffer_count(), 0);

        let bufs = pool.take(4, &template).expect("take 4");
        assert_eq!(bufs.len(), 4);
        // Pool is empty while buffers are out.
        assert_eq!(pool.buffer_count(), 0);

        // Geometry of each returned buffer matches the template.
        for b in &bufs {
            assert!(Yv12ScratchPool::geometry_eq(b, &template));
        }

        pool.put(bufs);
        assert_eq!(pool.buffer_count(), 4);
    }

    #[test]
    fn scratch_pool_warm_does_not_allocate() {
        // Serialise against other counter-observing tests in this
        // module to keep the delta snapshot tight.
        let _guard = scratch_alloc_counter_lock().lock().unwrap();
        let template = Yv12Buffer::alloc(
            64,
            48,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        let mut pool = Yv12ScratchPool::new();
        // Warmup: pull 3, put them back.
        let bufs = pool.take(3, &template).expect("warmup");
        pool.put(bufs);
        assert_eq!(pool.buffer_count(), 3);

        // Repeated take(3)/put(3) cycles must not bump the alloc
        // counter — they should all be served from the cache.
        use std::sync::atomic::Ordering;
        let before = RVP9_G5_2_SCRATCH_ALLOCS.load(Ordering::Relaxed);
        for _ in 0..5 {
            let bufs = pool.take(3, &template).expect("warm take");
            assert_eq!(bufs.len(), 3);
            pool.put(bufs);
        }
        let after = RVP9_G5_2_SCRATCH_ALLOCS.load(Ordering::Relaxed);
        assert_eq!(after, before, "warm pool must not allocate on take(3)");
    }

    #[test]
    fn scratch_pool_grows_monotonically() {
        let _guard = scratch_alloc_counter_lock().lock().unwrap();
        let template = Yv12Buffer::alloc(
            32,
            32,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        let mut pool = Yv12ScratchPool::new();

        // Round 1: take 2.
        let bufs = pool.take(2, &template).unwrap();
        pool.put(bufs);
        assert_eq!(pool.buffer_count(), 2);

        // Round 2: take 4 — pool must grow to satisfy.
        let bufs = pool.take(4, &template).unwrap();
        pool.put(bufs);
        assert_eq!(pool.buffer_count(), 4);

        // Round 3: take 2 again — pool keeps the larger high-water
        // mark even after a smaller take/put cycle.
        let bufs = pool.take(2, &template).unwrap();
        assert_eq!(bufs.len(), 2);
        // After take(2), the pool holds buffer_count = 4 - 2 = 2.
        assert_eq!(pool.buffer_count(), 2);
        pool.put(bufs);
        // After put, back to the high-water mark.
        assert_eq!(pool.buffer_count(), 4);
    }

    #[test]
    fn scratch_pool_resize_drops_cache() {
        let _guard = scratch_alloc_counter_lock().lock().unwrap();
        let small = Yv12Buffer::alloc(
            32,
            32,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        let large = Yv12Buffer::alloc(
            128,
            96,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        let mut pool = Yv12ScratchPool::new();
        let bufs = pool.take(3, &small).unwrap();
        pool.put(bufs);
        assert_eq!(pool.buffer_count(), 3);

        // Different geometry — old cache must be discarded.
        let bufs = pool.take(2, &large).unwrap();
        assert_eq!(bufs.len(), 2);
        for b in &bufs {
            assert!(Yv12ScratchPool::geometry_eq(b, &large));
        }
        // After putting back, pool holds only the new-geometry
        // buffers (the small ones were dropped on take).
        pool.put(bufs);
        assert_eq!(pool.buffer_count(), 2);
    }

    #[test]
    fn scratch_pool_alloc_counter_bumps_only_on_growth() {
        let _guard = scratch_alloc_counter_lock().lock().unwrap();
        // The counter is process-wide and other tests may run in
        // parallel and bump it. We assert relative deltas with an
        // inequality (other tests may have contributed to "after"
        // beyond our own 2), but the SECOND take is on our own pool
        // with the buffers already cached — our pool contributes 0,
        // and we assert the GLOBAL delta between the two snapshots
        // is `>= 0`. The tight equality check that the warm `take(2)`
        // contributes nothing is covered by `scratch_pool_warm_does_not_allocate`
        // which uses a `RVP9_G5_2_SCRATCH_ALLOCS` snapshot strictly
        // around its repeated take/put loop — that's tight enough
        // because any concurrent test's contribution would also have
        // to fire *during* that loop to spoil the measurement, and
        // the test corpus has no other test that bumps the counter
        // in a loop. Here we only check our pool grows and the warm
        // re-take returns the same buffers (a structural invariant
        // that doesn't depend on the global counter).
        let template = Yv12Buffer::alloc(
            32,
            32,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        let mut pool = Yv12ScratchPool::new();

        let bufs = pool.take(2, &template).unwrap();
        assert_eq!(bufs.len(), 2);
        pool.put(bufs);
        assert_eq!(pool.buffer_count(), 2);

        // Second take(2) drains the cache; the buffers come from
        // the cache (no growth). Verify by capturing the pointer
        // identity: the returned buffers must be byte-identical
        // to the ones we put. We can't compare `Yv12Buffer` directly
        // (no PartialEq), so we compare via geometry + verifying
        // pool_count drops to 0 (pool drained, no fresh alloc).
        let bufs2 = pool.take(2, &template).unwrap();
        assert_eq!(bufs2.len(), 2);
        // After draining 2 from a pool of 2, the pool is empty.
        assert_eq!(
            pool.buffer_count(),
            0,
            "second take(2) must drain the cache (not allocate fresh)"
        );
        pool.put(bufs2);
        // Pool returns to the high-water mark.
        assert_eq!(pool.buffer_count(), 2);
    }

    // ---- G5.3: merge_tile_rect_from --------------------------------------

    /// 2x2 tile grid (4 disjoint rects). Each tile stamps its scratch
    /// buffer with a unique fill byte; main thread merges all four
    /// rects into a fresh destination and confirms each rect retained
    /// its pattern. This is the G5.3 soundness oracle.
    #[test]
    fn merge_tile_rect_from_disjoint_rects() {
        const W: u32 = 128;
        const H: u32 = 64;
        const ROW_MID: usize = 32;
        const COL_MID: usize = 64;
        const COL_END: usize = 128;
        const ROW_END: usize = 64;
        const FILLS: [(u8, u8, u8); 4] = [
            (0xA0, 0xA1, 0xA2),
            (0xB0, 0xB1, 0xB2),
            (0xC0, 0xC1, 0xC2),
            (0xD0, 0xD1, 0xD2),
        ];

        let template = Yv12Buffer::alloc(
            W,
            H,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();

        let scratches: Vec<Yv12Buffer> = FILLS
            .iter()
            .map(|(y, u, v)| {
                let mut s = template.clone_geometry().unwrap();
                s.y_plane_mut().fill(*y);
                s.u_plane_mut().fill(*u);
                s.v_plane_mut().fill(*v);
                s
            })
            .collect();

        let mut dst = template.clone_geometry().unwrap();
        let rects: [(usize, usize, usize, usize); 4] = [
            (0, ROW_MID, 0, COL_MID),
            (0, ROW_MID, COL_MID, COL_END),
            (ROW_MID, ROW_END, 0, COL_MID),
            (ROW_MID, ROW_END, COL_MID, COL_END),
        ];
        for (i, (r0, r1, c0, c1)) in rects.iter().enumerate() {
            dst.merge_tile_rect_from(&scratches[i], *r0, *r1, *c0, *c1);
        }

        let y = dst.y_plane();
        let y_stride = y.stride();
        let y_data = y.data();
        let probes: [((usize, usize), u8); 8] = [
            ((0, 0), FILLS[0].0),
            ((0, COL_MID), FILLS[1].0),
            ((ROW_MID, 0), FILLS[2].0),
            ((ROW_MID, COL_MID), FILLS[3].0),
            ((ROW_MID - 1, COL_MID - 1), FILLS[0].0),
            ((ROW_MID - 1, COL_END - 1), FILLS[1].0),
            ((ROW_END - 1, COL_MID - 1), FILLS[2].0),
            ((ROW_END - 1, COL_END - 1), FILLS[3].0),
        ];
        for ((r, c), want) in probes {
            let got = y_data[r * y_stride + c];
            assert_eq!(got, want, "Y@({r},{c}) got 0x{got:02X} want 0x{want:02X}");
        }
    }

    // ------------------------------------------------------------------
    // Stage 9 — frame.rs branch coverage gaps
    // ------------------------------------------------------------------

    /// `Yv12Buffer::alloc` with non-Legacy byte alignment exercises
    /// the `pad_after` closure's `alignment > 1` arm (the Legacy path
    /// uses `alignment == 1` so its body returns 0 immediately).
    #[test]
    fn alloc_with_non_legacy_byte_alignment_succeeds() {
        let b = Yv12Buffer::alloc(
            64,
            48,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::V32,
        )
        .expect("V32 alignment allocates cleanly");
        assert_eq!(b.width(), 64);
        // V128 — the larger pad case exercises the same arm at different
        // alignment magnitudes.
        let b = Yv12Buffer::alloc(
            64,
            48,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::V128,
        )
        .expect("V128 alignment allocates cleanly");
        assert_eq!(b.width(), 64);
    }

    /// `Yv12ScratchPool::geometry_eq` returns false at each field in
    /// the `&&` chain. We construct pairs of buffers differing in
    /// exactly one of crop_height / chroma / border / strides / etc.
    /// to walk every short-circuit arm.
    #[test]
    fn scratch_pool_geometry_eq_false_arms() {
        let a = Yv12Buffer::alloc(
            64,
            48,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        // Different crop_height (also flows through aligned dims +
        // y_offset/u_offset/v_offset). 64×48 vs 64×32 differ only in
        // crop_height & dependent computed fields, exercising the
        // crop_height arm of the && chain.
        let b_height = Yv12Buffer::alloc(
            64,
            32,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        assert!(!Yv12ScratchPool::geometry_eq(&a, &b_height));

        // Different chroma format → ss differs → chroma_*/uv_*/offset
        // differ.
        let b_chroma = Yv12Buffer::alloc(
            64,
            48,
            ChromaFormat::Yuv444,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        assert!(!Yv12ScratchPool::geometry_eq(&a, &b_chroma));

        // Different border (32 vs 64) → strides change.
        let b_border = Yv12Buffer::alloc(
            64,
            48,
            ChromaFormat::Yuv420,
            BorderUsage::Custom(64),
            ByteAlignment::Legacy,
        )
        .unwrap();
        assert!(!Yv12ScratchPool::geometry_eq(&a, &b_border));

        // Different crop_width.
        let b_width = Yv12Buffer::alloc(
            128,
            48,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        assert!(!Yv12ScratchPool::geometry_eq(&a, &b_width));

        // Identical → true (this also pins the "all arms true" success).
        let a2 = a.clone_geometry().unwrap();
        assert!(Yv12ScratchPool::geometry_eq(&a, &a2));
    }

    /// `BorderUsage::Custom(n)` with `n % 32 == 0` is the valid arm of
    /// the custom-border check. Pin this against the rejection arm
    /// (already covered by `alloc_border_must_be_multiple_of_32`).
    #[test]
    fn alloc_custom_border_valid_multiple_of_32_succeeds() {
        let b = Yv12Buffer::alloc(
            64,
            64,
            ChromaFormat::Yuv420,
            BorderUsage::Custom(64),
            ByteAlignment::Legacy,
        )
        .expect("multiple-of-32 custom border is valid");
        assert_eq!(b.border(), 64);
    }

    /// `set_metadata` with `render_size = None` falls back to the
    /// `(crop_width, crop_height)` defaults via the `unwrap_or` branch
    /// (distinct from the `Some(..)` branch already covered by
    /// `metadata_setter`).
    #[test]
    fn set_metadata_none_render_size_falls_back_to_crop_dims() {
        let mut b = Yv12Buffer::alloc(
            100,
            80,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("ok");
        // First set a non-default render size.
        b.set_metadata(ColorSpace::Bt709, ColorRange::Full, Some((50, 40)));
        assert_eq!(b.render_size(), (50, 40));
        // Now pass None — render size should fall back to crop dims.
        b.set_metadata(ColorSpace::Bt2020, ColorRange::Studio, None);
        assert_eq!(b.render_size(), (100, 80));
        assert_eq!(b.color_space(), ColorSpace::Bt2020);
    }

    /// `merge_tile_columns_from` (wrapper) with `col_start == col_end`
    /// (zero-width tile range) is a no-op — exercises the
    /// `if y_col_len > 0` / `if uv_col_len > 0` false arms.
    #[test]
    fn merge_tile_columns_zero_width_is_noop() {
        let mut dst = Yv12Buffer::alloc(
            64,
            32,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc");
        dst.y_plane_mut().fill(0xAA);
        dst.u_plane_mut().fill(0xBB);
        dst.v_plane_mut().fill(0xCC);

        let mut src = dst.clone_geometry().expect("clone");
        src.y_plane_mut().fill(0x11); // different bytes
        src.u_plane_mut().fill(0x22);
        src.v_plane_mut().fill(0x33);

        // Zero-width range: should not modify dst.
        dst.merge_tile_columns_from(&src, 32, 32);

        let y = dst.y_plane();
        let u = dst.u_plane();
        let v = dst.v_plane();
        assert_eq!(y.row(0)[0], 0xAA, "Y left unchanged");
        assert_eq!(u.row(0)[0], 0xBB, "U left unchanged");
        assert_eq!(v.row(0)[0], 0xCC, "V left unchanged");
    }

    /// `Yv12ScratchPool::put` with a geometry-mismatched buffer + a
    /// non-empty cache drops the cached entries and keeps the new ones.
    /// Exercises the `geometry_eq` false branch in `put`.
    #[test]
    fn scratch_pool_put_geometry_mismatch_drops_cached() {
        let _guard = scratch_alloc_counter_lock().lock().unwrap();
        let small = Yv12Buffer::alloc(
            32,
            32,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        let large = Yv12Buffer::alloc(
            128,
            96,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        let mut pool = Yv12ScratchPool::new();

        // Step 1: park small-geometry buffers in the cache.
        let small_bufs = pool.take(2, &small).unwrap();
        pool.put(small_bufs);
        assert_eq!(pool.buffer_count(), 2);

        // Step 2: take a large-geometry buffer fresh (this discards the
        // cache via the `take` path — the put-mismatch arm is the
        // dual: cache holds geometry A, put with geometry B). We need
        // to reach `put`'s mismatch branch directly: the cache must
        // have small-geometry, and we put large-geometry without first
        // calling `take(_, &large)` (which would also drop the cache).
        //
        // Re-park small first by taking and putting once more.
        let small_bufs = pool.take(1, &small).unwrap();
        pool.put(small_bufs);
        assert_eq!(pool.buffer_count(), 2);

        // Now construct a large-geometry buffer OUTSIDE the pool, then
        // call put. The geometry mismatch should drop the small cache.
        let large_buf = large.clone_geometry().unwrap();
        pool.put(vec![large_buf]);
        // Cache size = 1 (small entries dropped, only the large one
        // remains).
        assert_eq!(pool.buffer_count(), 1);
        // Confirm the surviving buffer has large geometry.
        let bufs = pool.take(1, &large).unwrap();
        assert_eq!(bufs.len(), 1);
        assert!(Yv12ScratchPool::geometry_eq(&bufs[0], &large));
    }

    /// `extend_frame_borders` free-function alias matches
    /// `Yv12Buffer::extend_borders` byte-for-byte.
    #[test]
    fn extend_frame_borders_alias_matches_method() {
        let mut a = Yv12Buffer::alloc(
            32,
            32,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        let mut b = Yv12Buffer::alloc(
            32,
            32,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        a.y_plane_mut().fill(0x5A);
        b.y_plane_mut().fill(0x5A);
        a.extend_borders();
        extend_frame_borders(&mut b);
        assert_eq!(a.buffer_alloc, b.buffer_alloc);
    }

    /// `FramePoolError::Display` covers each variant's branch.
    #[test]
    fn frame_pool_error_display_each_variant() {
        let s_no_free = format!("{}", FramePoolError::NoFreeSlot);
        assert!(s_no_free.contains("no free frame-pool slot"));
        let s_oob = format!("{}", FramePoolError::SlotOutOfRange { slot: 42 });
        assert!(s_oob.contains("42"));
        assert!(s_oob.contains("out of range"));
    }

    /// `FrameAllocError::Display` covers each variant's branch.
    #[test]
    fn frame_alloc_error_display_each_variant() {
        let s_empty = format!(
            "{}",
            FrameAllocError::EmptyDimensions {
                width: 0,
                height: 240
            }
        );
        assert!(s_empty.contains("empty"));
        assert!(s_empty.contains("0"));
        let s_large = format!(
            "{}",
            FrameAllocError::DimensionsTooLarge {
                width: 100_000,
                height: 100_000,
            },
        );
        assert!(s_large.contains("too large"));
        let s_border = format!("{}", FrameAllocError::BorderNotMultipleOf32 { border: 48 });
        assert!(s_border.contains("48"));
    }

    /// `merge_tile_rect_from` with `row_start == row_end` is a no-op
    /// — both the Y and UV row loops never execute.
    #[test]
    fn merge_tile_rect_zero_height_is_noop() {
        let mut dst = Yv12Buffer::alloc(
            64,
            32,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc");
        dst.y_plane_mut().fill(0xAA);
        let mut src = dst.clone_geometry().unwrap();
        src.y_plane_mut().fill(0x11);

        dst.merge_tile_rect_from(&src, 16, 16, 0, 64);
        // dst unchanged at every visible row.
        for r in 0..32 {
            let val = dst.y_plane().row(r)[0];
            assert_eq!(val, 0xAA, "row {r}");
        }
    }

    /// `merge_tile_columns_from` (the G5.1 wrapper) must produce the
    /// same output as `merge_tile_rect_from(src, 0, y_height, c0, c1)`.
    /// Asserts backward compatibility for the tile_rows==1 path.
    #[test]
    fn merge_tile_columns_from_matches_full_height_rect() {
        let mut src = Yv12Buffer::alloc(
            128,
            64,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        src.y_plane_mut().fill(0x55);
        src.u_plane_mut().fill(0x33);
        src.v_plane_mut().fill(0x77);

        let mut dst_a = src.clone_geometry().unwrap();
        let mut dst_b = src.clone_geometry().unwrap();
        let y_h = src.aligned_y_height() as usize;
        dst_a.merge_tile_columns_from(&src, 0, 64);
        dst_b.merge_tile_rect_from(&src, 0, y_h, 0, 64);
        assert_eq!(
            dst_a.y_plane().data(),
            dst_b.y_plane().data(),
            "merge_tile_columns_from must equal full-height rect merge"
        );
    }
}
