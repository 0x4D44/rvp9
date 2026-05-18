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
                write!(f, "frame-pool slot {slot} out of range (0..{DECODER_FRAME_BUFFERS})")
            }
        }
    }
}

impl std::error::Error for FramePoolError {}

/// 9-slot recycling pool of decoded frames for the decoder.
///
/// Each slot holds either nothing (`None`) or a `FrameBuf` (an
/// `Arc<Yv12Buffer>`). A slot is "free" when either:
///
/// - the slot is `None`, or
/// - the slot's `Arc` has refcount 1 (only the pool holds the
///   reference, so it's safe to recycle the underlying buffer).
///
/// libvpx counterpart: `BufferPool::frame_bufs` plus the
/// `ref_count`/`ref_cnt_fb`/`get_free_fb` family at
/// `vp9_onyxc_int.h:66-365`. The reference-count semantics translate
/// directly: `Arc::strong_count` is `frame_bufs[i].ref_count`.
///
/// **Concurrency**: only the main decoder thread mutates the pool
/// (per HLD-threading §5.4); tile workers hold cloned `FrameBuf`
/// handles. The pool itself is `Sync` because every public method
/// takes `&mut self` — sharing must go through an external lock,
/// which matches libvpx's "main-thread mutates `buffer_pool`" rule.
#[derive(Debug, Default)]
pub struct DecoderFramePool {
    slots: [Option<FrameBuf>; DECODER_FRAME_BUFFERS],
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
    /// A slot is "free" if it's empty OR its `Arc` has refcount 1
    /// (only the pool holds it). If no slot is free,
    /// [`FramePoolError::NoFreeSlot`] is returned — at that point the
    /// caller has held too many refs (a bug).
    ///
    /// # Errors
    ///
    /// [`FramePoolError::NoFreeSlot`] when every slot has an
    /// outstanding external reference.
    pub fn acquire_free(&mut self, frame: FrameBuf) -> Result<usize, FramePoolError> {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            let is_free = match slot {
                None => true,
                Some(arc) => Arc::strong_count(arc) == 1,
            };
            if is_free {
                *slot = Some(frame);
                return Ok(i);
            }
        }
        Err(FramePoolError::NoFreeSlot)
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

    /// Reset the pool to empty.
    pub fn clear(&mut self) {
        for slot in self.slots.iter_mut() {
            *slot = None;
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
        let slot = pool.acquire_free(new_frame.clone()).expect("recycles slot 3");
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
        for &(r, c) in &[(-border, -border), (-1, -1), (47, 64), (47 + border - 1, 64 + border - 1)] {
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
            b.y_plane_mut().row_mut(r).copy_from_slice(a.y_plane().row(r));
        }
        for r in 0..16 {
            b.u_plane_mut().row_mut(r).copy_from_slice(a.u_plane().row(r));
            b.v_plane_mut().row_mut(r).copy_from_slice(a.v_plane().row(r));
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
        use crate::inter::filter::{convolve8_horiz, EIGHTTAP_KERNEL};

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
}
