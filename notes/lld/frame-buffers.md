# LLD: frame-buffers

> Stage 2 low-level design for `rvp9_core::frame`: the YV12 frame
> buffer, the 12-slot reference pool, the `Arc<FrameBuf>` ref-counted
> handle, frame padding/alignment, and the cross-thread ownership
> contract used by tile workers.
>
> Scope guardrails: **8-bit only** (`D:\language\rvp9-2\PLAN.md:75`,
> Q1 RESOLVED), **no altref** (PLAN.md D-13 RESOLVED at
> `D:\language\rvp9-2\PLAN.md:315`), **CYCLIC_REFRESH-only AQ** (D-05
> RESOLVED at `D:\language\rvp9-2\PLAN.md:316`). HBD branches in
> libvpx are cited but not ported.

---

## 1. Purpose

`rvp9_core::frame` owns the codec's pixel-storage substrate: the
`Yv12Buffer` (planar YUV 4:2:0, 8-bit, border-padded for SIMD
inter-prediction), the `FrameBuf` ref-counted handle, and the
`FramePool` recycling allocator (12 slots for the encoder: 8 ref +
1 current + 3 scaled; 9 slots for the decoder). Every pixel-touching
module reads or writes through this surface.

libvpx counterparts:
`third_party/libvpx/vpx_scale/yv12config.h:29-65` (`YV12_BUFFER_CONFIG`);
`vpx_scale/generic/yv12config.c:152-290` (alloc + strides);
`vp9/common/vp9_onyxc_int.h:66-97` (`RefCntBuffer` + `BufferPool`);
`vp9/encoder/vp9_encoder.c:6549-6554` (acquisition) and `:3292-3380`
(rotation).

Three commitments that ripple downstream:

1. Buffer storage is `Box<[u8]>`, not generic over `Pixel` (D-01 /
   8-bit only).
2. The handle is `Arc<FrameBuf>` (L-37 — workers cross threads).
3. Public API exposes the 8 reference slots only; the 3 scaled slots
   and the 1 current-frame slot are encoder-internal (D-08 / L-36 /
   A7 #22).

Cross-thread ownership transfer to tile workers uses a raw-pointer
aliasing contract mirroring HLD-threading §6.2 (strategy B) for the
current frame's recon planes; reference frames are pure
`Arc<FrameBuf>` reads under the §5.4-locked main-thread-only mutation
of the slot map.

---

## 2. libvpx origin

| Topic | libvpx file:line | Stage 1 reference |
|---|---|---|
| `YV12_BUFFER_CONFIG` struct | `third_party/libvpx/vpx_scale/yv12config.h:29-65` | `notes/vp9-analysis/01-architecture.md:2381-2389` |
| Frame allocator (encoder path) | `third_party/libvpx/vpx_scale/generic/yv12config.c:152-290` | `notes/vp9-analysis/01-architecture.md:1559-1560` |
| Encoder-only `vp8_yv12_alloc_frame_buffer` (legacy 32-byte alignment) | `third_party/libvpx/vpx_scale/generic/yv12config.c:50-130` | (legacy fallback) |
| `RefCntBuffer` + `BufferPool` | `third_party/libvpx/vp9/common/vp9_onyxc_int.h:66-97` | `notes/vp9-analysis/01-architecture.md:1331-1357` |
| `FRAME_BUFFERS = REF_FRAMES + 1 + REFS_PER_FRAME = 12` | `third_party/libvpx/vp9/common/vp9_onyxc_int.h:35-42` | `notes/vp9-analysis/01-architecture.md:1316-1325` |
| `REF_FRAMES = 8`, `REFS_PER_FRAME = 3` | `third_party/libvpx/vp9/common/vp9_onyxc_int.h:35-38` | `notes/vp9-analysis/01-architecture.md:2730-2731` |
| Border constants `VP9_ENC_BORDER_IN_PIXELS=160`, `VP9_DEC_BORDER_IN_PIXELS=32` | `third_party/libvpx/vpx_scale/yv12config.h:23-27` | (see §3.6) |
| Pool allocation site (encoder boot) | `third_party/libvpx/vp9/vp9_cx_iface.c:1126` | `notes/vp9-analysis/01-architecture.md:1532, 2370-2373` |
| `init_ref_frame_bufs` (slot map init) | `third_party/libvpx/vp9/encoder/vp9_encoder.c:2414-2415` | `notes/vp9-analysis/01-architecture.md:1559-1560` |
| `init_buffer_indices` (lst/gld/alt assignments) | `third_party/libvpx/vp9/encoder/vp9_encoder.c:1454-1463` | `notes/vp9-analysis/01-architecture.md:2397-2401` |
| `get_free_fb` (slot acquire) | `third_party/libvpx/vp9/common/vp9_onyxc_int.h:332-339` | `notes/vp9-analysis/01-architecture.md:2692-2710` |
| `ref_cnt_fb` (slot retain/release on rotation) | inline in `vp9_onyxc_int.h:332-365` | `notes/vp9-analysis/01-architecture.md:2675-2677` |
| New-frame slot acquisition per frame | `third_party/libvpx/vp9/encoder/vp9_encoder.c:6549-6554` | `notes/vp9-analysis/01-architecture.md:1727-1731, 2692-2702` |
| `update_ref_frames` (post-encode rotation, ref-count adjusts) | `third_party/libvpx/vp9/encoder/vp9_encoder.c:3292-3380` | `notes/vp9-analysis/01-architecture.md:2422-2459, 2605-2674` |
| `vp9_free_ref_frame_buffers` (teardown) | `third_party/libvpx/vp9/encoder/vp9_encoder.c:2910` | `notes/vp9-analysis/01-architecture.md:2363-2364` |
| Border extension (`vpx_extend_frame_borders_c`) | `third_party/libvpx/vpx_scale/generic/yv12extend.c:169` | (called from `vp9_scale_references` and `vp9_copy_and_extend_frame`) |
| Encoder copy-and-extend (`vp9_copy_and_extend_frame`) | `third_party/libvpx/vp9/encoder/vp9_extend.{h,c}` | — |
| Scaled-reference indirection | `third_party/libvpx/vp9/encoder/vp9_encoder.c:3454` (`vp9_scale_references`) | `notes/vp9-analysis/01-architecture.md:213, 791-799` |
| Cross-thread "no worker mutates buffer_pool" invariant | `third_party/libvpx/vp9/encoder/vp9_encoder.c:3292-3380` + frame loop | `notes/vp9-analysis/01-architecture.md:3057-3060`; `notes/vp9-analysis/00-overview.md:286` |

Primary Stage 1 specialist: **A1**
(`notes/vp9-analysis/01-architecture.md` §2.8, §3.9, §4.5, §7.1, §7.11,
§7.12). Secondary: **A6**
(`notes/vp9-analysis/06-threading-svc.md:1021` for the
tile-disjoint-write contract on `cm->buffer_pool->frame_bufs[..].buf`).

Downstream HLDs anchoring on this LLD:

- `notes/hld/00-architecture.md:290-334` (frame-buffer type layer; L-37
  Arc resolution).
- `notes/hld/01-threading.md:528, 5.9, §6.1, §6.2`
  (`buffer_pool.frame_bufs[12]` ownership; tile workers read
  `&FrameBuf` via `frame_refs[3]`; reconstructed pixel writes are
  tile-column-disjoint slices of `cur_frame.buf`).
- `notes/hld/04-svc.md:130-160, §3.1` (the slot map per layer; scaled
  source buffers separate from the pool).

---

## 3. Public API

The public API surfaces only the **8 reference slots** plus the
**current frame's reconstructed view** (read-only, after encode). The
3 scaled slots are an encoder-internal implementation detail of
`vp9_scale_references` (L-36 / D-08 / A7 #22) and are exposed only
through a `pub(crate)` interface to `rvp9-encoder`.

All public types and functions live in `rvp9_core::frame`. The module
graph mirrors `notes/hld/00-architecture.md:171`:

```text
rvp9_core::frame
├── pub struct Yv12Buffer        // §4.1
├── pub struct FrameBuf          // §4.2 (Arc-wrapped externally)
├── pub type   FrameRef          // = Arc<FrameBuf>
├── pub struct FramePool         // §4.3 (12 slots; encoder)
├── pub struct DecoderFramePool  // §4.3 (8+1 slots; decoder)
├── pub struct RefFrames         // §4.5 — the 3-ref snapshot per frame
├── pub struct TilePlanesMut     // §4.6 — tile-disjoint write view
└── pub struct TilePlanes        // §4.6 — tile-disjoint read view
```

### 3.1 `Yv12Buffer` lifecycle

```rust
impl Yv12Buffer {
    /// Mirrors `vpx_alloc_frame_buffer` (yv12config.c:292-307,
    /// internal-allocation path; callback variant at :208-230 not
    /// ported per DV-FB-5).
    pub fn alloc(
        width: u32, height: u32,
        border: BorderUsage, byte_alignment: ByteAlignment,
    ) -> Result<Self, FrameAllocError>;

    /// Mirrors `vpx_realloc_frame_buffer` (yv12config.c:152-290) —
    /// reuses backing alloc when `frame_size` fits.
    pub fn realloc(
        &mut self, width: u32, height: u32,
        border: BorderUsage, byte_alignment: ByteAlignment,
    ) -> Result<(), FrameAllocError>;

    pub fn width(&self) -> u32;            // y_crop_width
    pub fn height(&self) -> u32;           // y_crop_height
    pub fn y_stride(&self) -> i32;
    pub fn uv_stride(&self) -> i32;
    pub fn border(&self) -> u32;

    // Plane accessors return slices anchored at the cropped origin
    // (libvpx `y_buffer`, not `buffer_alloc`). Border padding lives
    // before/after this region; see §3.6.
    pub fn y_plane(&self) -> PlaneRef<'_>;
    pub fn u_plane(&self) -> PlaneRef<'_>;
    pub fn v_plane(&self) -> PlaneRef<'_>;
    pub fn y_plane_mut(&mut self) -> PlaneMut<'_>;
    pub fn u_plane_mut(&mut self) -> PlaneMut<'_>;
    pub fn v_plane_mut(&mut self) -> PlaneMut<'_>;

    /// Copies a YV12 source image into the cropped region. Does NOT
    /// extend borders. Mirrors the source-side copy inside
    /// `vp9_lookahead.c::vp9_lookahead_push`.
    pub fn copy_from_image(&mut self, src: &Yv12View<'_>) -> Result<(), FrameAllocError>;

    /// Replicates the cropped edge into the border. Mirrors
    /// `vpx_extend_frame_borders_c` (yv12extend.c:169).
    pub fn extend_borders(&mut self);

    /// Encoder-only one-pass copy + extend. Mirrors
    /// `vp9_copy_and_extend_frame` (vp9_extend.c:21-22).
    pub fn copy_and_extend_from(&mut self, src: &Yv12Buffer);
}
```

`BorderUsage = { Encoder=160, Decoder=32, Custom(n) }` —
`VP9_ENC_BORDER_IN_PIXELS` / `VP9_DEC_BORDER_IN_PIXELS`
(yv12config.h:26-27). `Custom(n)` requires `n & 0x1f == 0`
(yv12config.c:169).

`ByteAlignment = { Legacy=1, V32, V64, V128, V256, V512, V1024 }` —
`Legacy` is the libvpx `byte_alignment == 0` → `vp9_byte_align = 1`
path (yv12config.c:172). The encoder uses `Legacy`; the decoder
exposes the wider set per libvpx's `vpx_codec_dec_cfg_t::byte_align`.

**Preconditions**: `width >= 1`, `height >= 1`, `width * height <=
(1 << 30)` (libvpx `SIZE_MAX` guard at yv12config.c:204).
**Postconditions**: plane pointers non-null, non-overlapping,
`vp9_byte_align`-aligned (yv12config.c:276-284). **Panics**: none.
**Unsafety**: none in alloc; plane accessors project one
`Box<[u8]>` into non-overlapping sub-slices via offsets — safe code.

### 3.2 `FrameBuf` and the `Arc<FrameBuf>` handle

```rust
/// One ref-counted frame buffer. Wrap in `Arc` for the pool;
/// `Arc::clone` is the libvpx `ref_count++` and `Arc::drop` is
/// `ref_count--`.
pub struct FrameBuf {
    /// The YV12 buffer this slot owns. `UnsafeCell` because tile
    /// workers write disjoint slices of the **current** frame's recon
    /// plane during encode under the §3.6 aliasing contract; the
    /// `Arc<FrameBuf>` outer reference appears immutable to those
    /// workers.
    buf: UnsafeCell<Yv12Buffer>,

    /// Display order index (libvpx `RefCntBuffer.frame_index`,
    /// `vp9_onyxc_int.h:75`).
    frame_index: u32,

    /// Coding order index (libvpx `RefCntBuffer.frame_coding_index`,
    /// `vp9_onyxc_int.h:76`). Used for `ref_frame_sign_bias` in the
    /// frame loop.
    frame_coding_index: u32,

    /// The per-MV-block forward references stored with this frame.
    /// Allocated to `mi_rows * mi_cols` MV_REFs after `realloc`;
    /// libvpx counterpart `RefCntBuffer.mvs` (`vp9_onyxc_int.h:71`).
    /// Stored as `Box<[MvRef]>` here; access semantics in
    /// LLD-motion-estimation.
    mvs: UnsafeCell<Box<[MvRef]>>,

    /// `mi_rows` and `mi_cols` of the frame at the time the slot was
    /// last filled (`vp9_onyxc_int.h:72-73`).
    mi_rows: u32,
    mi_cols: u32,
}

// SAFETY: see §5.3 for the full justification. Briefly: the `buf`
// UnsafeCell is mutated through tile-disjoint slices under the
// HLD-threading §5.9 contract; the `mvs` UnsafeCell is mutated only on
// the main thread between frames and read by workers via `&FrameBuf`.
unsafe impl Sync for FrameBuf {}
unsafe impl Send for FrameBuf {}

pub type FrameRef = Arc<FrameBuf>;
```

`FrameBuf` is **not** `Clone` in the deep sense; clone the `Arc` to
duplicate the handle, never the buffer. The `Arc::strong_count`
discipline replaces libvpx's explicit `ref_count: int`
(`vp9_onyxc_int.h:69`): a slot is "free" iff `Arc::strong_count(slot)
== 1` (only the pool itself holds a reference).

### 3.3 `FramePool` (encoder, 12 slots) — public API

Field layout in §4.4; the constants:

```rust
pub const REF_FRAMES:     usize = 8;      // vp9_onyxc_int.h:37
pub const REFS_PER_FRAME: usize = 3;      // vp9_onyxc_int.h:35
pub const FRAME_BUFFERS:  usize = REF_FRAMES + 1 + REFS_PER_FRAME;  // 12

/// i8 preserves libvpx's "-1 means invalid" sentinel.
pub type SlotIndex = i8;
pub const INVALID_SLOT: SlotIndex = -1;   // libvpx INVALID_IDX (vp9_onyxc_int.h:48)
```

```rust
impl FramePool {
    /// Construct an empty 12-slot pool. Slots are allocated lazily on
    /// first acquire. Mirrors the pattern of `init_ref_frame_bufs`
    /// (vp9_encoder.c:2414-2415) followed by lazy `vpx_realloc_frame_buffer`
    /// inside `vp9_encode_frame_to_data_rate`.
    pub fn new(width: u32, height: u32, border: BorderUsage) -> Self;

    /// Acquire a free slot for the new frame. Returns the slot index
    /// and a mutable reference to the underlying buffer (the pool
    /// holds the only handle while the encoder writes recon pixels).
    ///
    /// Mirrors `get_free_fb(cm)` (`vp9_onyxc_int.h:332-339`) +
    /// `acquire_new_frame_slot` (notes/vp9-analysis/01-architecture.md:
    /// 2692-2702). On the libvpx side, `get_free_fb` linearly scans
    /// for `ref_count == 0`, sets `ref_count = 1`, returns the index.
    ///
    /// In rvp9, "ref_count == 0" maps to `Arc::strong_count(slot) == 1`
    /// (only the pool holds a reference). Returns
    /// `Err(PoolError::NoFreeSlot)` if all 12 slots are pinned (which
    /// should never happen at runtime — see §5 invariant SI-3).
    pub fn acquire_new(&mut self) -> Result<(SlotIndex, FrameRef), PoolError>;

    /// Look up the reference frame buffer for `which ∈ 0..REF_FRAMES`.
    /// Returns `None` if the slot is unmapped (initial state, or after
    /// a key frame that overwrote the slot).
    ///
    /// **This is the only public way to obtain a reference frame.**
    /// The 3 scaled-ref slots are *not* reachable here; they are
    /// reached via the `pub(crate)` `scaled_ref` accessor that
    /// rvp9-encoder uses internally.
    pub fn ref_frame(&self, which: RefFrameIndex) -> Option<FrameRef>;

    /// Slot map mutation, called by the encoder after a frame is
    /// encoded. Mirrors `ref_cnt_fb`
    /// (notes/vp9-analysis/01-architecture.md:2675-2677).
    ///
    /// Internally: `Arc::clone` the slot at `new` into
    /// `ref_frame_map[which]`, dropping the previous occupant. The
    /// previous occupant's refcount goes down by one; if that was the
    /// last external reference, the slot returns to the free pool.
    ///
    /// Pre: `new` was the value returned from `acquire_new` for this
    /// frame.
    /// Post: `ref_frame(which)` returns the frame just encoded.
    pub fn rotate(
        &mut self,
        which: RefFrameIndex,
        new: SlotIndex,
    );

    /// Release the new-frame slot after the encoder is done writing it
    /// and `rotate` has fanned it out to any reference slots that
    /// should retain it. Mirrors the `--ref_count` at
    /// `vp9_encoder.c:6549-6551`.
    ///
    /// In rvp9 this drops the encoder's strong reference; if all the
    /// slot's other holders also dropped it (e.g. it wasn't selected
    /// by any `rotate`), the slot is now free for the next
    /// `acquire_new`.
    pub fn release_new(&mut self);

    /// Resize all live slots to `(width, height)`. Mirrors libvpx's
    /// lazy realloc inside `vpx_realloc_frame_buffer`. Only safe
    /// between frames; the caller (encoder shell) is responsible for
    /// ordering. Asserts no slot has `Arc::strong_count > 1` (which
    /// would mean a worker is mid-encode).
    pub fn resize_all(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<(), FrameAllocError>;

    /// Drop every slot. Mirrors `vp9_free_ref_frame_buffers(pool)`
    /// (vp9_encoder.c:2910). Called from `Drop`.
    pub fn free_all(&mut self);
}

/// Index into the 8-slot reference map. Mirrors libvpx
/// `MV_REFERENCE_FRAME - LAST_FRAME` (`vp9/common/vp9_blockd.h:88-99`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RefFrameIndex {
    Last   = 0,
    Last2  = 1,
    Last3  = 2,
    Golden = 3,
    Bwd    = 4,
    Alt2   = 5,
    Alt    = 6,
    Slot7  = 7,
}
```

The library only ever publishes `Arc<FrameBuf>` handles. The user
cannot drop a slot prematurely (the pool always holds a strong
reference), cannot mutate a reference frame (only `&FrameBuf` is
exposed to non-pool holders; `FrameBuf::buf` is `UnsafeCell` so
`&FrameBuf` does not legally allow `&mut Yv12Buffer`), and cannot see
the 3 scaled slots.

### 3.4 `pub(crate)` scaled-slot API (encoder-internal)

Used by `rvp9_encoder::motion_estimation` and `rvp9_encoder::svc` to
manage `cpi->scaled_ref_idx[REFS_PER_FRAME]`
(`notes/vp9-analysis/01-architecture.md:790-798`). Encoder-internal
because the 3 scaled slots are an implementation detail of
`vp9_scale_references`
(`notes/vp9-analysis/01-architecture.md:213, 2508-2510`); exposing
them in the public API would (a) confuse downstream consumers who
expect `REF_FRAMES = 8` and (b) leak SVC behaviour that has nothing
to do with VP9's bitstream-visible reference semantics (A7 #22 /
D-08).

```rust
impl FramePool {
    /// Acquire one of the 3 scaled-ref slots for SVC inter-layer or
    /// dynamic-resize prediction. Returns the slot index inside the
    /// pool's `slots[]`. Mirrors the `vp9_scale_references` call site
    /// at `vp9_encoder.c:3454` and the `scaled_ref_idx[]` assignment.
    pub(crate) fn acquire_scaled(
        &mut self,
        ref_idx: u8,
    ) -> Result<(SlotIndex, FrameRef), PoolError>;

    /// Release one of the 3 scaled-ref slots. Mirrors
    /// `release_scaled_references` (vp9_encoder.c:5820-5822, called
    /// from frame loop when `!frame_is_intra_only` per
    /// `notes/vp9-analysis/01-architecture.md:1901`).
    pub(crate) fn release_scaled(&mut self, ref_idx: u8);

    /// Look up the slot index recorded for ref index `ref_idx` in
    /// `scaled_ref_idx[]`. Returns `INVALID_SLOT` if not scaled.
    pub(crate) fn scaled_ref(&self, ref_idx: u8) -> Option<FrameRef>;

    /// Reset `scaled_ref_idx[] = INVALID_SLOT` for the new frame
    /// (matches `vp9_encoder.c:6584`).
    pub(crate) fn reset_scaled_refs(&mut self);
}
```

### 3.5 `DecoderFramePool` — decoder pool

Decoder side allocates **9 slots** (`REF_FRAMES + 1`); the 3 scaled
slots have no decoder analogue (`notes/vp9-analysis/01-architecture.md:
1322-1323`). Same API shape as `FramePool` plus a `current_frame()`
display accessor, minus `acquire_scaled` / `release_scaled` /
`reset_scaled_refs`. Independent Rust struct (no generics — field
sets diverge); shared geometry helpers in `pool_common` (§5.5).

### 3.6 `RefFrames` — the per-frame 3-reference snapshot

Workers see only the 3 references the current frame actually uses
(`LAST`, `GOLDEN`, `ALTREF` — even though for spatial SVC the meaning
of `GOLDEN` and `ALTREF` differs per layer per
`notes/hld/04-svc.md:130-160`). This mirrors libvpx
`cm->frame_refs[REFS_PER_FRAME]` (`vp9_onyxc_int.h:1276` per
`notes/vp9-analysis/01-architecture.md:288`).

```rust
/// Per-frame 3-reference snapshot, immutable for the duration of one
/// frame encode. Workers receive `&RefFrames` and may not mutate it.
pub struct RefFrames {
    /// The 3 reference frames, each as an `Arc<FrameBuf>` handle. The
    /// `Arc`s are cloned out of the pool's `ref_frame_map` at frame
    /// start by the main thread.
    refs: [FrameRef; REFS_PER_FRAME],

    /// Per-reference sign bias: `true` if the reference frame's
    /// `frame_index > current_frame_index` (i.e. the reference is in
    /// the future; libvpx `cm->ref_frame_sign_bias[REFS_PER_FRAME]`).
    /// In our scope (no altref per D-13) this is always `[false; 3]`,
    /// but the field is kept for SVC where layer indexing can produce
    /// non-trivial values.
    sign_bias: [bool; REFS_PER_FRAME],
}

impl RefFrames {
    /// Build the snapshot from the encoder pool's current map and
    /// the per-frame `(lst_fb_idx, gld_fb_idx, alt_fb_idx)` indices.
    /// Called once on the main thread before workers spawn.
    pub fn snapshot(
        pool: &FramePool,
        lst: RefFrameIndex,
        gld: RefFrameIndex,
        alt: RefFrameIndex,
        current_frame_index: u32,
    ) -> Self;

    pub fn last(&self)   -> &FrameBuf { &*self.refs[0] }
    pub fn golden(&self) -> &FrameBuf { &*self.refs[1] }
    pub fn altref(&self) -> &FrameBuf { &*self.refs[2] }

    pub fn sign_bias(&self, i: u8) -> bool { self.sign_bias[i as usize] }
}
```

`RefFrames` is `Send + Sync` by composition (`FrameRef` is `Send +
Sync` per §3.2, scalar fields are `Sync`). Tile workers receive a
shared `&RefFrames`; no aliasing concerns because the underlying
plane buffers are read-only during encode (the §5.3 invariant).

### 3.7 `TilePlanesMut` / `TilePlanes` — cross-thread plane views

The current frame's recon plane is written tile-disjointly by
workers. The HLD-threading §6.2 strategy says: at frame start, the
main thread calls `FramePool::split_current_for_tiles(...)` which
returns one `TilePlanesMut` per tile column. Each worker receives its
own `&'frame mut TilePlanesMut`.

```rust
/// Tile-disjoint mutable view of one tile column's slice of the
/// current frame's recon planes. Granted to one worker per frame.
/// Lifetime tied to the `FrameScope<'frame>` in HLD-threading §3.2.
pub struct TilePlanesMut<'frame> {
    pub tile_col: u32,
    /// Pixel column range owned by this tile. Mirrors the
    /// `mi_col_start * MI_SIZE .. mi_col_end * MI_SIZE` slicing in
    /// `get_tile_offset` (vp9/common/vp9_tile_common.c:18-22).
    pub x_range: Range<u32>,
    /// Strides are frame-level (a tile is a contiguous slice of full
    /// scanlines).
    pub y_stride: i32,
    pub uv_stride: i32,
    /// Per-plane pointer into the cropped origin of the column slice.
    /// Raw pointers under §5.3 aliasing contract.
    y_ptr: NonNull<u8>,
    u_ptr: NonNull<u8>,
    v_ptr: NonNull<u8>,
    /// Frame-scope lifetime binding.
    _marker: PhantomData<&'frame mut FrameBuf>,
}

unsafe impl<'frame> Send for TilePlanesMut<'frame> {}
unsafe impl<'frame> Sync for TilePlanesMut<'frame> {}
```

Read-only counterpart for reference frames:

```rust
pub struct TilePlanes<'frame> {
    pub tile_col: u32,
    pub x_range: Range<u32>,
    pub y_stride: i32,
    pub uv_stride: i32,
    y_ptr: NonNull<u8>,
    u_ptr: NonNull<u8>,
    v_ptr: NonNull<u8>,
    _marker: PhantomData<&'frame FrameBuf>,
}
unsafe impl<'frame> Send for TilePlanes<'frame> {}
unsafe impl<'frame> Sync for TilePlanes<'frame> {}
```

Splitting API on the pool:

```rust
impl FramePool {
    /// Take one `TilePlanesMut` per tile column for the current
    /// frame. Returns `Err(PoolError::CurrentFrameNotAcquired)` if
    /// `acquire_new` hasn't been called this frame. The returned
    /// vector borrows the current frame for `'frame`; the pool
    /// statically prevents further mutation until the borrow ends.
    ///
    /// SAFETY contract is in §5.3.
    pub fn split_current_for_tiles<'frame>(
        &'frame mut self,
        log2_tile_cols: u8,
        mi_cols: u32,
    ) -> Result<Vec<TilePlanesMut<'frame>>, PoolError>;
}
```

Decoder counterpart on `DecoderFramePool::split_current_for_tiles`
is identical in shape.

### 3.8 Errors

```rust
#[derive(Debug, thiserror::Error)]
pub enum FrameAllocError {
    #[error("zero or negative dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    #[error("border must be a multiple of 32, got {0}")]
    InvalidBorder(u32),
    #[error("allocation failure")]
    OutOfMemory,
}

#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    #[error("no free slot in 12-slot pool")]
    NoFreeSlot,
    #[error("split_current_for_tiles called before acquire_new")]
    CurrentFrameNotAcquired,
    #[error("slot is still referenced by {refs} consumers")]
    SlotStillReferenced { refs: usize },
    #[error("alloc error")]
    Alloc(#[from] FrameAllocError),
}
```

Both error types are `Send + Sync + 'static`.

---

## 4. Types

### 4.1 `Yv12Buffer` — concrete layout

```rust
pub struct Yv12Buffer {
    /// 32-byte-aligned owning allocation (replaces libvpx
    /// `vpx_memalign(32, frame_size)`, yv12config.c:237).
    buffer_alloc:    Aligned32<Box<[u8]>>,
    buffer_alloc_sz: usize,

    /// Plane offsets into `buffer_alloc` (libvpx uses raw pointers
    /// instead; see DV-FB-4). y_offset points at the cropped origin
    /// (not the border start).
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,

    // Geometry — mirrors yv12config.h:30-65. Dropped fields per
    // DV-FB-3: alpha_*, bit_depth, flags, color_space, color_range,
    // corrupted.
    y_width:       u32,  y_height:      u32,
    y_crop_width:  u32,  y_crop_height: u32,
    y_stride:      i32,
    uv_width:      u32,  uv_height:     u32,
    uv_crop_width: u32,  uv_crop_height: u32,
    uv_stride:     i32,

    border:     u32,
    frame_size: usize,   // yplane + 2*uvplane (yv12config.c:190)
    subsampling_x: u8,   // fixed (1, 1) in v1 — 4:2:0 only
    subsampling_y: u8,
    byte_alignment: u16, // libvpx vp9_byte_align (yv12config.c:172)
    render_width:  u32,  // display knobs (yv12config.h:60-61)
    render_height: u32,
}

#[derive(Copy, Clone, Debug)]
pub enum BorderUsage {
    /// 160-pixel border, used for the encoder (matches
    /// VP9_ENC_BORDER_IN_PIXELS at yv12config.h:26). Required for
    /// sub-pel MC reach during ME and reconstruction.
    Encoder,
    /// 32-pixel border, used for the decoder (matches
    /// VP9_DEC_BORDER_IN_PIXELS at yv12config.h:27).
    Decoder,
    /// Caller-specified, must be a multiple of 32.
    Custom(u32),
}

#[derive(Copy, Clone, Debug)]
pub enum ByteAlignment {
    Legacy,
    V32,
    V64,
    V128,
    V256,
    V512,
    V1024,
}
```

`Yv12Buffer: Send + Sync` is auto-derived only if the `Aligned32` newtype
and `Box<[u8]>` are `Send + Sync` — both are. The `Yv12Buffer` exposes
no interior mutability of its own; the cross-thread-write contract is
expressed by the surrounding `UnsafeCell` inside `FrameBuf` (§3.2).

### 4.2 `Aligned32` — owned 32-byte-aligned allocation

Replaces libvpx's `vpx_memalign(32, ...)` (yv12config.c:237):

```rust
pub(crate) struct Aligned32<T> {
    inner: T,
    /// Offset within `inner` to the 32-aligned origin. 0..=31.
    offset: u8,
}

impl Aligned32<Box<[u8]>> {
    pub(crate) fn new(size: usize) -> Result<Self, FrameAllocError>;
    pub(crate) fn as_ptr(&self) -> *const u8;
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8;
    pub(crate) fn len(&self) -> usize;           // size, not size+31
    pub(crate) fn as_slice(&self) -> &[u8];
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8];
}
```

Implementation: `Box::new_zeroed_slice(size + 31)` over-allocates
by 31 bytes; `offset` is the byte distance from `Box::as_ptr()` to
the next 32-byte boundary. `as_ptr` returns the adjusted pointer.
`add(offset)` with `offset <= 31` is provably in-bounds, so no
`unsafe` is needed for the alignment arithmetic itself.

### 4.3 `FrameBuf` field invariants

Restating §3.2 with invariants spelled out:

- `buf: UnsafeCell<Yv12Buffer>` — interior mutability only because
  tile workers write disjoint slices via `TilePlanesMut<'frame>`
  raw-pointer views (§5.3). The `Yv12Buffer` itself is never moved
  or reallocated while any `FrameRef` exists outside the pool; the
  pool's `resize_all` asserts `Arc::strong_count == 1` (only the pool
  holds it) before any mutation that could invalidate worker pointers.
- `mvs: UnsafeCell<Box<[MvRef]>>` — written **only** on the main
  thread between frames (post-encode, when `cur_frame.mvs` is filled
  in for the next frame's MV-prediction lookups; see
  `notes/vp9-analysis/01-architecture.md:1336` —
  `RefCntBuffer.mvs`). Tile workers **read** through `&FrameBuf` and
  the `UnsafeCell::get()` accessor under a documented safety comment;
  no worker writes (HLD-threading §5.2 row "buffer_pool"). The
  read-only invariant during encode is the strong fence.
- `frame_index`, `frame_coding_index`, `mi_rows`, `mi_cols` —
  scalar, never mutated post-acquire. Set inside `acquire_new` and
  carried through.

### 4.4 `FramePool` field invariants

```rust
pub struct FramePool {
    slots: [Option<FrameRef>; FRAME_BUFFERS],
    ref_frame_map: [SlotIndex; REF_FRAMES],
    new_fb_idx: SlotIndex,
    pub(crate) scaled_ref_idx: [SlotIndex; REFS_PER_FRAME],

    // Allocator state.
    width: u32,
    height: u32,
    border: BorderUsage,
    byte_alignment: ByteAlignment,
}
```

Invariants:

- **SI-1 (refcount inverse)**: a slot is free iff `slots[i].is_none()`
  *or* `Arc::strong_count(slots[i].as_ref().unwrap()) == 1`. This is
  the rvp9 equivalent of libvpx's `ref_count == 0`
  (`vp9_onyxc_int.h:332-339`).
- **SI-2 (slot map disjointness)**: `ref_frame_map[i]` and
  `ref_frame_map[j]` may be equal (a slot can back multiple reference
  indices, as happens on a key frame — `notes/vp9-analysis/
  01-architecture.md:2432-2434` — golden and alt-ref both point at
  the same `new_fb_idx`). When that happens, the slot's refcount
  reflects the multiplicity.
- **SI-3 (capacity)**: at most 8 distinct ref-map entries + 1
  `new_fb_idx` + 3 `scaled_ref_idx[i]` = 12 distinct slot occupancies
  at any point. `acquire_new` never fails in practice because of this
  count
  (`notes/vp9-analysis/01-architecture.md:2705-2710`).
- **SI-4 (no worker mutation)**: `slots`, `ref_frame_map`, and
  `scaled_ref_idx` are mutated only on the main thread between frames
  (`notes/vp9-analysis/00-overview.md:286`,
  `notes/vp9-analysis/01-architecture.md:3057-3060`,
  `notes/hld/01-threading.md:762-765`). Workers borrow `&FramePool`
  read-only via `RefFrames` snapshots; the pool itself never crosses
  a worker thread boundary.

### 4.5 Constants

```rust
pub const REF_FRAMES:    usize = 8;        // vp9_onyxc_int.h:37
pub const REFS_PER_FRAME: usize = 3;       // vp9_onyxc_int.h:35
pub const FRAME_BUFFERS:  usize = 12;      // vp9_onyxc_int.h:42
pub const INVALID_SLOT: SlotIndex = -1;    // vp9_onyxc_int.h:48 (INVALID_IDX)

/// libvpx `VP9_ENC_BORDER_IN_PIXELS` (yv12config.h:26).
pub(crate) const ENC_BORDER_PIXELS: u32 = 160;
/// libvpx `VP9_DEC_BORDER_IN_PIXELS` (yv12config.h:27).
pub(crate) const DEC_BORDER_PIXELS: u32 = 32;
/// libvpx `VP9_INTERP_EXTEND` (yv12config.h:25). The sub-pel filter
/// reach beyond the cropped frame edge. Documented for SIMD predictor
/// LLDs; not a direct constraint on the allocator.
pub(crate) const INTERP_EXTEND: u32 = 4;
/// libvpx `VP9INNERBORDERINPIXELS` (yv12config.h:24). Inner cropping
/// margin used by some scalers. Documented for completeness.
pub(crate) const INNER_BORDER: u32 = 96;
```

### 4.6 Plane views

```rust
pub struct PlaneRef<'a> {
    pub width: u32,         // crop width
    pub height: u32,        // crop height
    pub stride: i32,
    /// Pointer to the cropped origin (row 0, col 0). The plane has
    /// `border` pixels of padding above, below, left, and right;
    /// negative row/col offsets reach into the padding.
    pub data: &'a [u8],
}

pub struct PlaneMut<'a> {
    pub width: u32,
    pub height: u32,
    pub stride: i32,
    pub data: &'a mut [u8],
}
```

The `data` slice spans from the cropped-origin row 0 to crop row
`height - 1` inclusive plus the border below; callers project
sub-rectangles via `chunks(stride as usize)` plus a column offset.
This is **not** the libvpx pattern (libvpx hands you a raw `uint8_t
*y_buffer`); we use slices in safe code paths and raw pointers
behind `TilePlanesMut` for the tile-disjoint inner loops.

---

## 5. Internal algorithm

### 5.1 Allocation arithmetic (mirrors yv12config.c:152-290)

For `Yv12Buffer::alloc(width, height, BorderUsage::Encoder,
ByteAlignment::Legacy)`:

```text
border = 160
ss_x = ss_y = 1                       # 4:2:0 fixed
vp9_byte_align = 1                    # Legacy maps to 1 (yv12config.c:172)
aligned_width  = (width  + 7) & !7    # yv12config.c:173
aligned_height = (height + 7) & !7    # yv12config.c:174
y_stride       = (aligned_width + 2*border + 31) & !31    # yv12config.c:175
yplane_size    = (aligned_height + 2*border) * y_stride + byte_alignment
                                                          # yv12config.c:176-177
uv_width       = aligned_width  >> ss_x                   # yv12config.c:178
uv_height      = aligned_height >> ss_y                   # yv12config.c:179
uv_stride      = y_stride       >> ss_x                   # yv12config.c:180
uv_border_w    = border         >> ss_x                   # yv12config.c:181
uv_border_h    = border         >> ss_y                   # yv12config.c:182
uvplane_size   = (uv_height + 2*uv_border_h) * uv_stride + byte_alignment
                                                          # yv12config.c:183-184
frame_size     = yplane_size + 2 * uvplane_size           # yv12config.c:190

allocate Aligned32<Box<[u8]>>(frame_size)                 # yv12config.c:237
memset(0, frame_size)                                     # yv12config.c:245

y_offset = align_up( border * y_stride + border,
                     vp9_byte_align )                     # yv12config.c:276
u_offset = align_up( yplane_size +
                     uv_border_h * uv_stride + uv_border_w,
                     vp9_byte_align )                     # yv12config.c:278
v_offset = align_up( yplane_size + uvplane_size +
                     uv_border_h * uv_stride + uv_border_w,
                     vp9_byte_align )                     # yv12config.c:281
```

The `align_up` helper is:

```rust
fn align_up(offset: usize, align: u32) -> usize {
    let align = align as usize;
    (offset + align - 1) & !(align - 1)
}
```

This mirrors the `yv12_align_addr` macro
(`yv12config.c:28-29`).

`y_offset` etc. are offsets into the 32-aligned sub-slice of
`buffer_alloc`. libvpx stores raw pointers (`ybf->y_buffer =
buffer_alloc + y_offset`); we store offsets to keep `Yv12Buffer`
relocatable and pointer-free in the public API surface
(see DV-FB-4). The legacy `vp8_yv12_*` allocator
(yv12config.c:50-130) is not ported (DV-FB-6).

### 5.2 Lifecycle: lazy allocation + reuse

The libvpx pool is allocated empty (just the `BufferPool` struct +
12 `RefCntBuffer` slots, each with `ref_count = 0` and a zero-sized
`YV12_BUFFER_CONFIG`). Each slot's underlying allocation is filled
lazily on first use, inside `vpx_realloc_frame_buffer` which is
called from `vp9_get_compressed_data` after `get_free_fb` returns
the slot index (`notes/vp9-analysis/01-architecture.md:1727-1731`).

rvp9 mirrors: `FramePool::new` returns 12 slots that are `None`.
`acquire_new` walks `slots[]` for the first index `i` such that
`slots[i].is_none() || strong_count(slots[i]) == 1`; if `is_none`,
construct a fresh `FrameBuf` (which triggers a `Yv12Buffer::alloc`);
if reusable, reuse and call `Yv12Buffer::realloc` if dimensions
changed.

Pseudocode (mirrors `get_free_fb` at `vp9_onyxc_int.h:332-339` +
the `--ref_count` at `vp9_encoder.c:6550-6551`):

```text
fn acquire_new(&mut self) -> Result<(SlotIndex, FrameRef), PoolError> {
    // Release the previous new frame slot (libvpx: --ref_count on
    // the old new_fb_idx, vp9_encoder.c:6550-6551).
    if self.new_fb_idx != INVALID_SLOT {
        // Drop our strong handle to the old new frame. If rotate() has
        // not fanned it out, this is the last strong reference and the
        // slot becomes free.
        // NOTE: we previously held a Self::clone() handle inside
        // `slots[i]` PLUS a "current frame" handle the encoder shell
        // returned to itself; here we drop the latter via release_new.
        let _ = self.new_fb_idx;
    }

    // Scan for a free slot (Arc::strong_count == 1 ↔ libvpx ref_count == 0).
    let i = self.find_free_slot()?;

    // Lazy alloc / reuse:
    let frame_ref = match self.slots[i].as_ref() {
        None => {
            let yv12 = Yv12Buffer::alloc(self.width, self.height,
                                          self.border, self.byte_alignment)?;
            Arc::new(FrameBuf::new(yv12))
        }
        Some(existing) => {
            // Reuse. Resize if needed.
            // SAFETY: SI-1 says strong_count == 1, so no other thread
            // holds a reference; we can mutate Yv12Buffer through the
            // UnsafeCell. The Arc itself stays alive via slots[i].
            let r = Arc::clone(existing);
            if r.buf_geometry() != (self.width, self.height) {
                unsafe { (*r.buf.get()).realloc(self.width, self.height,
                                                 self.border, self.byte_alignment)?; }
            }
            r
        }
    };

    self.slots[i] = Some(Arc::clone(&frame_ref));
    self.new_fb_idx = i as SlotIndex;
    Ok((i as SlotIndex, frame_ref))
}

fn find_free_slot(&self) -> Result<usize, PoolError> {
    self.slots.iter().enumerate().find_map(|(i, opt)| match opt {
        None => Some(i),
        Some(arc) if Arc::strong_count(arc) == 1 => Some(i),
        _ => None,
    }).ok_or(PoolError::NoFreeSlot)
}
```

### 5.3 Cross-thread plane access — the aliasing contract

This is the load-bearing aliasing contract that lets the encoder
write recon pixels into the current frame's plane from N worker
threads at once. Mirrors the libvpx invariant in
`notes/vp9-analysis/06-threading-svc.md:1021`:

> `cm->buffer_pool->frame_bufs[...].buf.y_buffer/u_buffer/v_buffer` —
> disjoint per tile column for the new frame; reference frames are
> read-only during tile encode.

And the rvp9 mapping per `notes/hld/01-threading.md:528, 5.9`:

> Workers read ref pixels via `&FrameBuf` projected from
> `frame_refs[3]` slots. `ref_count` is `AtomicI32`. … Workers write
> tile-disjoint slices of `frame_to_show.y_buffer/u_buffer/v_buffer`.

The `FrameBuf::buf` field is `UnsafeCell<Yv12Buffer>` (§3.2). The
**only** legal mutation routes are:

1. `FramePool::resize_all`, which asserts `Arc::strong_count == 1`
   for every live slot before mutating. Single-threaded, between
   frames.
2. `FramePool::split_current_for_tiles<'frame>(&'frame mut self,
   log2_tile_cols, mi_cols) -> Vec<TilePlanesMut<'frame>>`.

For route (2), the SAFETY contract is:

```rust
// SAFETY contract for `split_current_for_tiles`:
// - Provenance: the y/u/v pointers in each TilePlanesMut alias the
//   current new_fb's Yv12Buffer, which lives inside an
//   Arc<FrameBuf> held by self.slots[new_fb_idx]. The borrow
//   `&'frame mut self` proves that no other code path can mutate the
//   pool (and thus the slot) for the duration of `'frame`.
// - Bounds: each TilePlanesMut's `x_range` is computed by
//   `get_tile_offset` (vp9/common/vp9_tile_common.c:18-22) and
//   asserted (debug_assert!) to be within `[0, y_crop_width)`. Stride
//   indexing is bounded by `y_crop_height - 1`.
// - Alignment: Pixel = u8 in v1 (D-01); alignment is 1. SIMD callees
//   handle wider alignment themselves on top of `_mm_loadu_*` /
//   `vld1q_u8` (unaligned variants), matching libvpx's stride choice.
// - Aliasing: tile columns are pixel-disjoint by construction
//   (`get_tile_offset`'s output is monotonically increasing); the
//   `x_range`s in the returned Vec are non-overlapping. The
//   `_marker: PhantomData<&'frame mut FrameBuf>` makes overlapping
//   re-splits a borrow-checker error at the call site (FramePool
//   cannot return two `Vec<TilePlanesMut<'frame>>` simultaneously
//   without a borrow conflict).
// - Lifetime: `'frame` is tied to FramePool's exclusive borrow; when
//   the FrameScope joins, the lifetime ends, the Vec drops, and the
//   pool reacquires exclusive access.
// - CPU feature: n/a.
unsafe impl<'frame> Send for TilePlanesMut<'frame> { /* see above */ }
unsafe impl<'frame> Sync for TilePlanesMut<'frame> { /* see above */ }
```

Cross-row-MT mutation within a tile column is **not** governed by
`TilePlanesMut`; the row-MT happens-before edges are provided by
`RowSync` in HLD-threading §4.3 / §6.3. `TilePlanesMut` only proves
*outer* (tile-column-level) disjointness. The inner cell pattern is
handled by `LLD-tile-threading` and `LLD-mode-decision`; this LLD
exposes the necessary `NonNull<u8>` and trusts the caller to honour
the row-MT happens-before edges. The SAFETY comment at every callsite
of `TilePlanesMut::row_unchecked(...)` (defined in
`LLD-tile-threading`) restates the row-MT contract.

For **reference** frames (route through `RefFrames`), no
`UnsafeCell` aliasing is needed: `RefFrames` holds three
`Arc<FrameBuf>` and workers read through `&FrameBuf`. The
`UnsafeCell<Yv12Buffer>` interior is touched in read-only fashion via
`UnsafeCell::get()` cast to `*const Yv12Buffer` then dereferenced
under `unsafe`. The SAFETY argument is:

```text
// SAFETY: this thread holds `&FrameBuf` through an `Arc<FrameBuf>`
// projected via `RefFrames::last/golden/altref`. The §5.2 invariant
// guarantees no other thread holds a `&mut Yv12Buffer` to this slot
// during the encode (the slot is in `ref_frame_map[]`, mutated only
// on the main thread between frames). The read is therefore race-free.
```

### 5.4 Reference rotation (`rotate` / `release_new`)

Mirrors `update_ref_frames` in `vp9_encoder.c:3292-3380`
(`notes/vp9-analysis/01-architecture.md:2422-2459, 2605-2674`).
libvpx's `ref_cnt_fb(pool->frame_bufs, &cm->ref_frame_map[i], new_idx)`
decrements `frame_bufs[ref_frame_map[i]].ref_count`, then sets
`ref_frame_map[i] = new_idx` and increments
`frame_bufs[new_idx].ref_count`.

rvp9 represents the "+1 per ref-map entry" refcount as **parallel Arc
holders**, not a duplicated `slots[]` lookup:

```rust
struct FramePool {
    slots: [Option<FrameRef>; FRAME_BUFFERS],
    ref_frame_map: [SlotIndex; REF_FRAMES],
    /// Per-ref-index Arc retention — the rvp9 equivalent of libvpx's
    /// per-slot ref_count integer. `ref_holders[i]` is `Some(arc)`
    /// where `arc == slots[ref_frame_map[i] as usize].clone()`.
    ref_holders:    [Option<FrameRef>; REF_FRAMES],
    new_fb_holder:  Option<FrameRef>,
    scaled_holders: [Option<FrameRef>; REFS_PER_FRAME],
    // ...
}

fn rotate(&mut self, which: RefFrameIndex, new: SlotIndex) {
    let which = which as usize;
    if new != INVALID_SLOT {
        // Clone `new` (Arc::clone == libvpx ++ref_count on `new`).
        // The previous ref_holders[which] drops here (Arc::drop ==
        // libvpx --ref_count on `prev`).
        self.ref_holders[which] =
            Some(Arc::clone(self.slots[new as usize].as_ref().expect("acquired")));
        self.ref_frame_map[which] = new;
    } else {
        self.ref_holders[which] = None;
        self.ref_frame_map[which] = INVALID_SLOT;
    }
}

fn release_new(&mut self) {
    // Drop the encoder's +1 on new_fb_idx. If rotate() has not fanned
    // it out via ref_holders, this drops strong_count to 1 and the
    // slot becomes eligible for the next acquire_new.
    self.new_fb_holder = None;
    self.new_fb_idx    = INVALID_SLOT;
}
```

The `slots[]` array exists only to anchor the *one* place a fresh
allocation is made; `ref_holders`, `new_fb_holder`, and
`scaled_holders` are the rvp9 stand-ins for libvpx's per-slot
`ref_count: int`. On `acquire_new` the encoder receives a fresh
`FrameRef` clone; the pool keeps `slots[i]` plus `new_fb_holder` (two
strong refs). After `rotate(LAST, new_fb_idx)` + `release_new`, the
slot's strong count typically settles at 2 (the `slots[]` entry plus
`ref_holders[LAST]`), which mirrors libvpx's "this slot now has one
live reference" state.

### 5.5 Decoder pool — what's shared

Common helpers live in `rvp9_core::frame::pool_common`:

- `find_free_slot(slots: &[Option<FrameRef>]) -> Option<usize>` — pure
  function; works for any slot count.
- `pop_slot_or_alloc(slot: &mut Option<FrameRef>, w, h, border, ba) ->
  Result<FrameRef, FrameAllocError>` — the lazy alloc/reuse step.
- `align_up`, `compute_strides`, `compute_plane_offsets` — geometry
  helpers shared between encoder and decoder buffers.

`FramePool` and `DecoderFramePool` are independent structs that both
build on these helpers. We considered a generic `FramePool<const N:
usize>` but the field divergence (encoder has `scaled_ref_idx`;
decoder has none) makes generics noisier than the duplication.

### 5.6 Border extension and the SIMD invariant

VP9 inter-prediction reads `border` pixels beyond the cropped frame
edge to satisfy the sub-pel filter footprint
(`VP9_INTERP_EXTEND = 4`, yv12config.h:25). The encoder uses 160-pixel
borders so that the worst-case "MV points 64 pixels off the edge"
case has 96 pixels of slack inside the border. The decoder uses 32
pixels because it never searches.

`Yv12Buffer::extend_borders` replicates the cropped edge into the
border per `vpx_extend_frame_borders_c` (yv12extend.c:169). For the
encoder source frame (after `copy_from_image`), this is called once
per frame on the main thread before workers spawn. For reconstructed
output (the current new frame), the loop-filter pass implicitly
writes recon pixels inside the cropped region; the border is then
extended by `vp9_copy_and_extend_frame`
(vp9/encoder/vp9_extend.c:21) when the reconstructed frame is later
used as a reference for the next frame.

The **invariant** that LLD-intra-pred and LLD-motion-estimation rely
on: when a worker reads `&FrameBuf` for a reference frame, the
border bytes are already filled. The frame loop must call
`extend_borders` (via `vp9_copy_and_extend_frame` post-loopfilter)
before `rotate(…, new_fb_idx)` makes the frame available as a
reference. LLD-encoder-shell (Stage 2 follow-up) owns the exact call
ordering; this LLD documents the contract:

> **Border contract**: `FramePool::ref_frame(which)` returns a
> `FrameRef` whose underlying `Yv12Buffer` has fully extended
> borders. SIMD inter-prediction can read up to `border` pixels off
> any edge of the cropped region; intra-prediction reads at most 4
> pixels beyond any cropped edge (within the cropped block's
> available context).

### 5.7 Frame teardown

`FramePool::free_all` (called from `Drop`):

```text
for i in 0..REF_FRAMES { self.ref_holders[i] = None; }
self.new_fb_holder = None;
for i in 0..REFS_PER_FRAME { self.scaled_holders[i] = None; }
for i in 0..FRAME_BUFFERS { self.slots[i] = None; }
// The Arc::drop chain releases every Yv12Buffer; Aligned32 drops
// each Box<[u8]>; allocator returns the pages.
```

This matches `vp9_free_ref_frame_buffers(pool)`
(`vp9_encoder.c:2910`,
`notes/vp9-analysis/01-architecture.md:2363-2364`). The `BufferPool`
itself (libvpx) is freed by the codec destructor at
`cx_iface.c:2332` (`notes/vp9-analysis/01-architecture.md:2370-2373`);
in rvp9 the `FramePool` is dropped when the `Encoder` is dropped.

---

## 6. SIMD dispatch

This module contains **no SIMD kernels of its own**. The
border-extension kernel (`vpx_extend_frame_borders_c`) is the only
candidate; libvpx ships scalar-only for it
(`yv12extend.c:169`). HLD-simd `notes/hld/02-simd-dispatch.md:95`
lists "Frame-border extend / copy_y" as 3 kernels but rates them
low-priority. Plain `Box<[u8]>` byte copy with `copy_from_slice` plus
a row-replicate loop is fast enough; if profiling at Stage 5 shows
this in the top-10 we add SIMD then.

What this module **provides** to SIMD consumers:

- Plane pointers are 32-byte aligned at the start of the cropped
  region for both 8-bit luma (`y_buffer`) and chroma planes
  (yv12config.c:276-284), so SIMD readers can use aligned loads at the
  origin (libvpx uses `_mm_loadu_*` because per-row alignment isn't
  guaranteed inside the plane; we mirror that).
- Strides are multiples of 32 for luma (`y_stride = (aligned_width
  + 2*border + 31) & ~31`), so every row starts at a 32-aligned
  offset relative to the plane origin. This is the property the libvpx
  SAD/variance/predictor kernels rely on for `_mm_load_si128` at the
  start of each row.
- The chroma stride is `y_stride >> 1` (yv12config.c:180), which is
  always a multiple of 16 — sufficient for SSE2 chroma reads.

These guarantees go into HLD-simd's "alignment guarantees" section
and per-kernel preconditions for inter-pred / SAD / variance.

---

## 7. Deliberate divergences from libvpx

These feed `notes/divergence-ledger.md`. Each entry follows the
ledger format.

### DV-FB-1 — `FrameBuf` refcount uses `Arc::strong_count`, not an integer

- **Where**: `rvp9_core::frame::FrameBuf` vs libvpx
  `RefCntBuffer.ref_count` (vp9_onyxc_int.h:69).
- **libvpx**: `int ref_count` per slot, manually
  `++`/`--`'d in `get_free_fb` (vp9_onyxc_int.h:332-339) and
  `ref_cnt_fb`.
- **rvp9**: `Arc<FrameBuf>` clones in `ref_holders`, `new_fb_holder`,
  `scaled_holders`; slot occupancy = `Arc::strong_count(slot) - 1`.
- **Why**: idiomatic Rust; `Drop`-on-free for free; identical
  observable behaviour.
- **Byte-parity impact**: none — scan order matches (see DV-FB-8).
- **Compression-efficiency impact**: none.
- **Future plan**: durable.

### DV-FB-2 — 3 scaled slots hidden from public API

- **Where**: `FramePool::{acquire,release,reset}_scaled`,
  `scaled_ref_idx[]` are `pub(crate)`.
- **libvpx**: 12-slot `BufferPool` is exposed via the public
  `vpx_codec_*` ABI when EXT_RATECTRL / simple-encode is enabled.
- **rvp9**: public API is the 8-slot reference map only; scaled
  slots reachable only inside `rvp9-encoder`. Decision: D-08 / A7 #22.
- **Why**: scaled slots are SVC / dynamic-resize internal; not in
  VP9's reference model.
- **Byte-parity impact**: none.
- **Compression-efficiency impact**: none.
- **Future plan**: re-evaluate post-v1 if integrators request it.

### DV-FB-3 — `Yv12Buffer` drops alpha / HBD / color-meta / corrupted

- **Where**: `rvp9_core::frame::Yv12Buffer` vs `YV12_BUFFER_CONFIG`
  fields at yv12config.h:42-64.
- **libvpx**: carries `alpha_*`, `bit_depth`, `flags`
  (`YV12_FLAG_HIGHBITDEPTH`), `color_space`, `color_range`,
  `corrupted`.
- **rvp9**: drops all. Alpha unused; HBD is v2 (D-01); color metadata
  lives on `EncoderConfig`; `corrupted` replaced by `Result` returns.
- **Byte-parity impact**: none.
- **Compression-efficiency impact**: none.
- **Future plan**: re-add `bit_depth` at Stage 6+ for HBD.

### DV-FB-4 — Plane offsets stored as `usize`, not raw pointers

- **Where**: `Yv12Buffer::{y_offset, u_offset, v_offset}` vs libvpx
  `YV12_BUFFER_CONFIG::{y_buffer, u_buffer, v_buffer}` (raw, yv12config.h:46-48).
- **libvpx**: bakes plane base pointer at alloc time.
- **rvp9**: byte offsets into `buffer_alloc`; `PlaneRef`/`PlaneMut`
  compute slices on demand.
- **Why**: keeps `Yv12Buffer: Send` trivial; one extra add per
  accessor, below noise.
- **Byte-parity impact**: none.
- **Compression-efficiency impact**: none.
- **Future plan**: revisit if the extra add shows up in profiles
  (Stage 5).

### DV-FB-5 — External `vpx_codec_frame_buffer_t` callback path not implemented

- **Where**: `Yv12Buffer::alloc` is internal-only.
- **libvpx**: `vpx_realloc_frame_buffer` accepts a callback
  (yv12config.c:208-230); Chromium uses it for pinned shared memory.
- **rvp9**: `Box<[u8]>` only.
- **Why**: not in v1 scope; PLAN.md §8 defers C ABI / FFI bindings.
- **Byte-parity impact**: none.
- **Compression-efficiency impact**: none.
- **Future plan**: add a `FrameBufferAllocator` trait in v1.1/v2 when
  asked for.

### DV-FB-6 — Legacy `vp8_yv12_*` allocator not ported

- **Where**: `rvp9_core::frame` ships only the VP9 allocator.
- **libvpx**: also ships `vp8_yv12_*` (yv12config.c:50-130) with
  16-byte-aligned width and no `byte_alignment` knob; used by VP8.
- **rvp9**: VP9 path only.
- **Why**: no VP8 support in v1 (PLAN.md §8).
- **Byte-parity impact**: none.
- **Compression-efficiency impact**: none.
- **Future plan**: never.

### DV-FB-7 — `FrameBuf::mvs` owned, not pointer

- **Where**: `FrameBuf::mvs: UnsafeCell<Box<[MvRef]>>` vs libvpx
  `RefCntBuffer.mvs: MV_REF *` (vp9_onyxc_int.h:71).
- **libvpx**: pointer to externally-allocated `mi_rows * mi_cols`
  array; freed in `vp9_remove_common`.
- **rvp9**: `Box<[MvRef]>` owned by the slot; reallocated by
  `FrameBuf::resize_mvs`.
- **Byte-parity impact**: none (in-memory only).
- **Compression-efficiency impact**: none.
- **Future plan**: none.

### DV-FB-8 — Pool slot scan order **frozen** to match libvpx exactly

- **Where**: `FramePool::find_free_slot` (§5.2).
- **libvpx**: `for (i = 0; i < FRAME_BUFFERS; ++i)` first
  `ref_count == 0` (vp9_onyxc_int.h:332-339).
- **rvp9**: `slots.iter().enumerate().find_map(...)` — same scan
  order, lowest free index wins.
- **Why**: explicit *absence* of a divergence — recorded so future
  "optimise" passes don't break it. PLAN.md D8 byte-parity diagnostic
  needs `cm->new_fb_idx` values to match libvpx; changing scan order
  would break dynamic-resize parity tests.
- **Byte-parity impact**: zero (identical behaviour).
- **Compression-efficiency impact**: none.
- **Future plan**: keep stable until v2 frame-parallel encoding
  invalidates the single-thread assumption.

---

## 8. Open questions

None are blocking. Each below has a default this LLD adopts; the
"action" column says where the default is confirmed.

| # | Question | Default in this LLD | Action / owner |
|---|---|---|---|
| OQ-FB-1 | `FrameBuf::mvs` allocation site | allocated lazily inside `acquire_new` (mirrors libvpx `alloc_frame_mvs` called from `vp9_alloc_context_buffers`) | confirm with LLD-encoder-shell that no other writer exists |
| OQ-FB-2 | libvpx `RefCntBuffer.released: uint8_t` field (`vp9_onyxc_int.h:73`) | dropped — appears callback-only and DV-FB-5 drops callbacks | revert if Stage 4a finds an internal use |
| OQ-FB-3 | `RefFrames` field type: `[FrameRef; 3]` vs `[Option<FrameRef>; 3]` | `[FrameRef; 3]` plus a `is_intra_only: bool` companion — mirrors libvpx (`cm->frame_refs[i].buf` is always populated; call sites branch on `frame_type`) | confirm at LLD-encoder-shell |
| OQ-FB-4 | Encoder `byte_alignment` knob | `Legacy` (matches libvpx default). Any other value is supported by the allocator but not exercised by `rvp9-encoder`. | LLD-encoder-shell selects |
| OQ-FB-5 | Decoder `release_fb_cb` reintroduction | dropped in v1 (DV-FB-5); revisit when an integrator asks | LLD-decoder-shell owns |
| OQ-FB-6 | `RateControllerExt` exposure of pool internals | trait must **not** expose the pool. If WebRTC needs ref-slot stats, surface via a `RefFrameStats` view, never raw `FramePool`. | confirm at LLD-rate-control + HLD-svc |

---

## 9. Test strategy

### 9.1 Unit tests (in-crate, `rvp9-core/src/frame/tests.rs`)

- **alloc_stride_math**: for widths in `[1, 2, 7, 8, 16, 320, 640,
  1280, 1920, 3840]` × heights similarly, assert `y_stride`,
  `uv_stride`, `yplane_size`, `uvplane_size`, and per-plane offsets
  match the libvpx formulas at yv12config.c:173-184 byte-for-byte
  (spreadsheet test against §5.1).
- **alloc_validation**: `border = 33` returns
  `FrameAllocError::InvalidBorder(33)` (yv12config.c:169);
  `width = 0` returns `InvalidDimensions`.
- **plane_alignment**: at `(1920, 1080, 160, Legacy)`, every plane
  pointer satisfies `as_ptr() as usize % 32 == 0`.
- **realloc_reuse / realloc_shrink**: realloc same dims preserves
  capacity; growing increases `buffer_alloc_sz`; shrinking is a
  no-op on the backing allocation (yv12config.c:231-246).
- **pool_acquire_basic / with_refs / keyframe_fanout**: walk
  acquire / rotate(LAST) / rotate(LAST,GOLDEN,ALTREF) / release_new
  sequences; assert slot indices and `Arc::strong_count` move in
  lockstep with the libvpx model.
- **pool_12_slots_never_exhausts**: 100-frame encode sequence; assert
  no `NoFreeSlot` error.
- **pool_pinned_block_acquire**: pin 12 slots without releasing;
  assert the 13th `acquire_new` returns `Err(NoFreeSlot)`. Artificial
  guardrail — normal use can't reach 13 holders (8+1+3 = 12 max).
- **extend_borders_replicate**: fill cropped region with `0xAA`,
  call `extend_borders`, verify edge pixels and corner pixels are
  `0xAA`.
- **split_current_for_tiles_disjoint**: at
  `(1920, log2_tile_cols=2, mi_cols=240)`, the 4 returned
  `TilePlanesMut` have non-overlapping `x_range`s covering
  `[0, 1920)`.
- **split_current_for_tiles_compile_fail**: a `compile_fail` doc-test
  asserting that calling `split_current_for_tiles` twice without
  dropping the first result is a borrow-checker error.

### 9.2 Differential tests (against libvpx, `tests/differential/`)

- **diff_stride_math**: drive `vpx_realloc_frame_buffer` via the
  Stage 0 libvpx submodule build with the same inputs as
  `Yv12Buffer::alloc`; assert `y_stride`, `uv_stride`,
  `buffer_alloc_sz`, and `(plane_origin - buffer_alloc)` offsets
  all match. **Load-bearing**: any divergence here ripples into
  every SIMD kernel.
- **diff_slot_acquire_sequence**: drive a libvpx encoder for 30
  CBR single-tile frames; record `cm->new_fb_idx` sequence + final
  `cm->ref_frame_map[]`; replay the same `(rotate, release)` calls
  on rvp9's `FramePool`; assert slot indices match. Wired in Stage
  4a (depends on LLD-encoder-shell).

### 9.3 Property tests (`proptest`)

For random `(width: 1..=4096, height: 1..=4096, border ∈ {32, 64,
128, 160}, byte_alignment ∈ valid set)`, allocate and assert:
`y_stride >= aligned_width + 2*border`; `y_stride % 32 == 0`;
`uv_stride == y_stride / 2`; `(y_buffer_offset) % byte_alignment ==
0`; planes do not overlap.

For random pool-op sequences, run rvp9's pool side-by-side with a
small `MockPool` that re-implements libvpx semantics (slot scan +
explicit ref_count); assert state equivalence after every op.

### 9.4 Miri / Loom

- **Miri** (per `notes/hld/00-architecture.md:621`, `rvp9-core` outside
  SIMD must be Miri-clean): cover `Aligned32`'s offset math; cover
  `FramePool::split_current_for_tiles` with a small program that
  splits, writes each tile via `std::thread::scope`, and asserts no
  aliasing UB; cover `FrameBuf::mvs` reads through `UnsafeCell::get()`.
- **Loom** (`tests/loom/frame_pool_refcount.rs`): single small model
  exercising acquire / rotate / release_new under reordering; mostly
  a v2 future-safe guardrail because v1 keeps the pool on the main
  thread.

### 9.5 Conformance and bench

Conformance vectors flow through `DecoderFramePool` from
LLD-decoder-shell; no frame-buffer-specific vectors. A micro-bench in
`bench/frame_alloc.rs` covers `Yv12Buffer::alloc` and
`extend_borders` as a sanity check, not a release gate.

---

## Summary

Length: **~1362 lines** (over the 1200 target; surplus driven by 8
divergence-ledger entries and four load-bearing SAFETY contracts in
§3.7 + §5.3 + §5.4 + §5.6). Key **§7 entries**: DV-FB-1 (`Arc::
strong_count` replaces int `ref_count`), DV-FB-2 (scaled slots
`pub(crate)`), DV-FB-3 (drop alpha/HBD/color/`corrupted` fields),
DV-FB-4 (offsets, not pointers), DV-FB-5 (no callback allocator),
DV-FB-6 (no legacy `vp8_yv12_*`), DV-FB-7 (`mvs` owned, not pointer),
DV-FB-8 (slot scan order frozen). **§8 has no blocking questions**:
all six items have defaults this LLD adopts, each requiring only
confirmation in a downstream LLD (encoder-shell / rate-control /
decoder-shell) — no Stage 3 dependency.
