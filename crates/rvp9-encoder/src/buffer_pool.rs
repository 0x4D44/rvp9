//! W1.1 — VP9 frame-buffer pool and reference-frame management.
//!
//! Faithful port of the libvpx buffer-pool data structures and
//! refcounting helpers in `vp9/common/vp9_onyxc_int.h:332-358` and
//! `vp9_encoder.c::update_reference_frames`. The encoder allocates
//! every reconstructed frame into a shared pool of refcounted slots;
//! the `ref_frame_map[8]` array records which slots are currently
//! addressable by the bitstream's ref-frame indices.
//!
//! ## libvpx layout (`vp9/common/vp9_onyxc_int.h:35-93`)
//!
//! ```text
//! REF_FRAMES_LOG2 = 3
//! REF_FRAMES      = 8       (size of ref_frame_map[])
//! REFS_PER_FRAME  = 3       (LAST + GOLDEN + ALTREF)
//! FRAME_BUFFERS   = REF_FRAMES + 1 + REFS_PER_FRAME
//!                 = 8 + 1 + 3 = 12  (pool capacity)
//! INVALID_IDX     = -1
//! ```
//!
//! `cm->buffer_pool->frame_bufs[i]` is a refcounted slot. `cm->
//! ref_frame_map[i]` (for i in 0..8) holds a slot index, or
//! INVALID_IDX. `cm->new_fb_idx` is the slot holding the
//! currently-being-encoded frame. `cm->lst_fb_idx` / `cm->gld_fb_idx`
//! / `cm->alt_fb_idx` are slot indices for the three named refs.
//!
//! ## W1.1 scope
//!
//! - Carry the pool + ref-frame-map + slot indices as data structures
//!   the encoder can reason about.
//! - Implement the refcounting helpers (`get_free_fb`, `ref_cnt_fb`,
//!   `decrease_ref_count`) so allocation and slot rotation match
//!   libvpx semantics.
//! - Implement `update_reference_frames(refresh_frame_flags)` —
//!   the libvpx hook that walks the 8-bit refresh mask and
//!   rotates ref_frame_map[].
//!
//! Wave 4+ wires this into `StreamEncoder`'s frame loop; until then
//! the existing `last_recon_*` / `golden_recon_*` fields in
//! `StreamEncoder` continue to drive the encoder's reference reads,
//! and this module coexists as the W1 foundation that downstream
//! waves consume.

/// libvpx `REF_FRAMES_LOG2` (`vp9_onyxc_int.h:37`).
pub const REF_FRAMES_LOG2: u32 = 3;

/// libvpx `REF_FRAMES` (`vp9_onyxc_int.h:38`). Size of the bitstream's
/// ref-frame map. VP9 names 3 of these (LAST, GOLDEN, ALTREF); the
/// remaining 5 slots are reserved but writable via
/// `refresh_frame_flags`.
pub const REF_FRAMES: usize = 1usize << REF_FRAMES_LOG2;

/// libvpx `REFS_PER_FRAME` (`vp9_onyxc_int.h:35`). The number of
/// named refs any inter frame can use.
pub const REFS_PER_FRAME: usize = 3;

/// libvpx `FRAME_BUFFERS` (`vp9_onyxc_int.h:42`). Capacity of the
/// buffer pool: `REF_FRAMES + 1 + REFS_PER_FRAME` — enough for every
/// ref slot, the new-frame slot, and the scaled-ref slots.
pub const FRAME_BUFFERS: usize = REF_FRAMES + 1 + REFS_PER_FRAME;

/// libvpx `INVALID_IDX` (`vp9_alloccommon.h:14`). Sentinel for an
/// empty / freed buffer-pool slot.
pub const INVALID_IDX: i32 = -1;

pub(crate) type OwnedYuv420 = (Vec<u8>, Vec<u8>, Vec<u8>);

/// One refcounted slot in the buffer pool. Mirrors libvpx's
/// `RefCntBuffer` (`vp9_onyxc_int.h:64-91`) at field granularity that
/// the encoder hot path actually reads — frame data, slot metadata,
/// and the refcount. The actual pixel storage is owned externally
/// by the encoder (`StreamEncoder::last_recon_*` /
/// `golden_recon_*`); the slot tracks its refcount and a monotonic
/// frame ID so callers can identify which pool index holds which
/// encoded frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct RefCntBuffer {
    /// libvpx `ref_count`. `0` ⇒ the slot is free and reusable.
    /// `> 0` ⇒ the slot is held by at least one ref-frame-map entry
    /// or by the new-frame index.
    pub ref_count: u32,
    /// Monotonic encode-frame counter at the moment this slot was
    /// allocated by `get_free_fb`. `0` for the initial / never-used
    /// state.
    pub frame_index: u64,
}

/// libvpx `BufferPool` (`vp9_onyxc_int.h:88-105`). Pool of refcounted
/// frame buffers — encoder + decoder share the same shape (in
/// libvpx; rvp9 carries the encoder-side definition here and may
/// share it with the decoder later).
#[derive(Debug, Clone)]
pub struct BufferPool {
    /// `frame_bufs[FRAME_BUFFERS]` — the actual pool. Indexed by
    /// pool index (0..FRAME_BUFFERS).
    pub frame_bufs: [RefCntBuffer; FRAME_BUFFERS],
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferPool {
    /// Allocate a fresh empty pool — every slot has `ref_count = 0`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            frame_bufs: [RefCntBuffer {
                ref_count: 0,
                frame_index: 0,
            }; FRAME_BUFFERS],
        }
    }

    /// libvpx `get_free_fb` (`vp9_onyxc_int.h:332-347`). Walks the
    /// pool linearly for the first slot with `ref_count == 0`, sets
    /// its refcount to 1, and returns its index. Returns
    /// [`INVALID_IDX`] (`-1`) if no slot is free.
    ///
    /// `frame_index` is recorded on the allocated slot for diagnostic
    /// tracing (no libvpx equivalent — rvp9 carries this as a debug
    /// aid; downstream waves may surface it through a journal log).
    pub fn get_free_fb(&mut self, frame_index: u64) -> i32 {
        for i in 0..FRAME_BUFFERS {
            if self.frame_bufs[i].ref_count == 0 {
                self.frame_bufs[i].ref_count = 1;
                self.frame_bufs[i].frame_index = frame_index;
                return i as i32;
            }
        }
        INVALID_IDX
    }

    /// libvpx `ref_cnt_fb` (`vp9_onyxc_int.h:349-358`). Decrements
    /// the refcount of the slot at `*idx` (if it was a valid slot)
    /// and points `*idx` at `new_idx`, incrementing that slot's
    /// refcount.
    ///
    /// Used to walk `ref_frame_map[]`: when a refresh bit fires, the
    /// previous occupant of that slot is released (refcount goes
    /// down by 1) and the new frame is installed (refcount up by 1).
    pub fn ref_cnt_fb(&mut self, idx: &mut i32, new_idx: i32) {
        let ref_index = *idx;
        if ref_index >= 0 && self.frame_bufs[ref_index as usize].ref_count > 0 {
            self.frame_bufs[ref_index as usize].ref_count -= 1;
        }
        *idx = new_idx;
        if new_idx >= 0 {
            self.frame_bufs[new_idx as usize].ref_count += 1;
        }
    }

    /// libvpx `decrease_ref_count` (`vp9_decoder.c:33-45`). Decrement
    /// a single slot's refcount, saturating at zero. Used at
    /// end-of-frame to release the new-frame index after it's been
    /// promoted to a ref slot.
    pub fn decrease_ref_count(&mut self, idx: i32) {
        if idx >= 0 && self.frame_bufs[idx as usize].ref_count > 0 {
            self.frame_bufs[idx as usize].ref_count -= 1;
        }
    }
}

/// libvpx ref-frame management on top of [`BufferPool`]. Tracks the
/// 8-entry `ref_frame_map[]`, the current new-frame index, and the
/// three named-ref pool indices (LAST / GOLDEN / ALTREF).
///
/// libvpx layout: `vp9_onyxc_int.h:126` (`ref_frame_map[REF_FRAMES]`)
/// and `vp9_encoder.h` (`cpi->lst_fb_idx`, `cpi->gld_fb_idx`,
/// `cpi->alt_fb_idx`).
#[derive(Debug, Clone)]
pub struct RefFrameManager {
    /// `ref_frame_map[8]` — each entry holds a buffer-pool index or
    /// [`INVALID_IDX`]. The encoder's `refresh_frame_flags` (8-bit
    /// mask) selects which slots get the current frame's pool index.
    pub ref_frame_map: [i32; REF_FRAMES],
    /// `cm->new_fb_idx` — pool index of the currently-being-encoded
    /// frame.
    pub new_fb_idx: i32,
    /// `cpi->lst_fb_idx` — pool index that LASTFRAME currently
    /// resolves to.
    pub lst_fb_idx: i32,
    /// `cpi->gld_fb_idx` — pool index for GOLDEN_FRAME.
    pub gld_fb_idx: i32,
    /// `cpi->alt_fb_idx` — pool index for ALTREF_FRAME.
    pub alt_fb_idx: i32,
}

impl Default for RefFrameManager {
    fn default() -> Self {
        Self::new()
    }
}

impl RefFrameManager {
    /// libvpx `init_ref_frame_bufs` (`vp9_encoder.c:1014-1027`). All
    /// ref-frame slots start at [`INVALID_IDX`]; the first KF
    /// allocates a pool slot and installs it everywhere.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ref_frame_map: [INVALID_IDX; REF_FRAMES],
            new_fb_idx: INVALID_IDX,
            lst_fb_idx: INVALID_IDX,
            gld_fb_idx: INVALID_IDX,
            alt_fb_idx: INVALID_IDX,
        }
    }

    /// W1.3(vpxenc parity, A10 P2): KF-startup reset (libvpx's
    /// `vp9_setup_past_independence` clears all ref slots and
    /// allocates a fresh one for the KF). Acquires a free pool slot
    /// via `pool.get_free_fb`, installs it as `new_fb_idx`,
    /// `lst_fb_idx`, `gld_fb_idx`, `alt_fb_idx`, and broadcasts it
    /// across all 8 `ref_frame_map[]` entries.
    pub fn keyframe_reset(&mut self, pool: &mut BufferPool, frame_index: u64) {
        // Drop existing references first so the pool slots are freed
        // and `get_free_fb` can find one.
        for slot in &mut self.ref_frame_map {
            if *slot >= 0 {
                pool.decrease_ref_count(*slot);
                *slot = INVALID_IDX;
            }
        }
        if self.new_fb_idx >= 0 {
            pool.decrease_ref_count(self.new_fb_idx);
        }
        let idx = pool.get_free_fb(frame_index);
        self.new_fb_idx = idx;
        // libvpx broadcasts the KF index across every ref-map slot
        // via `vp9_setup_past_independence`. Each slot adds one
        // refcount; `get_free_fb` already set it to 1, so the total
        // ends up at 1 + 8 + 3 = 12 (3 for lst/gld/alt below).
        self.lst_fb_idx = idx;
        self.gld_fb_idx = idx;
        self.alt_fb_idx = idx;
        if idx >= 0 {
            // Refcount bookkeeping: lst/gld/alt each add one, plus
            // each of the 8 ref_frame_map slots adds one.
            for _ in 0..(3 + REF_FRAMES) {
                pool.frame_bufs[idx as usize].ref_count += 1;
            }
            for slot in &mut self.ref_frame_map {
                *slot = idx;
            }
        }
    }

    /// libvpx `update_reference_frames` (`vp9_encoder.c:3297-3349`).
    /// Walks the 8-bit `refresh_frame_flags`: for each set bit, the
    /// corresponding `ref_frame_map[i]` is updated to point at
    /// `new_fb_idx` via [`BufferPool::ref_cnt_fb`].
    ///
    /// The `lst_fb_idx` / `gld_fb_idx` / `alt_fb_idx` named-ref
    /// indices are NOT updated here — libvpx's encoder updates those
    /// in a separate post-encode step (`vp9_update_reference_frames`
    /// epilogue) once the GF/ARF rotation logic knows which slots
    /// the next frame will read from. rvp9 will land the named-ref
    /// rotation logic in Wave 3 (A10 P3) alongside ALTREF;
    /// `refresh_frame_flags` carries enough information to drive
    /// the ref_frame_map[] update on its own.
    pub fn update_ref_frame_map(&mut self, pool: &mut BufferPool, refresh_frame_flags: u8) {
        for i in 0..REF_FRAMES {
            if (refresh_frame_flags >> i) & 1 == 1 {
                self.ref_cnt_for_slot(pool, i, self.new_fb_idx);
            }
        }
    }

    /// Internal helper: update `ref_frame_map[slot]` to point at
    /// `new_idx`, decrementing the old occupant's refcount and
    /// incrementing the new one's. Borrows `&mut self` only on the
    /// `ref_frame_map[]` slot to avoid aliasing `self.new_fb_idx`.
    fn ref_cnt_for_slot(&mut self, pool: &mut BufferPool, slot: usize, new_idx: i32) {
        let old = self.ref_frame_map[slot];
        if old >= 0 && pool.frame_bufs[old as usize].ref_count > 0 {
            pool.frame_bufs[old as usize].ref_count -= 1;
        }
        self.ref_frame_map[slot] = new_idx;
        if new_idx >= 0 {
            pool.frame_bufs[new_idx as usize].ref_count += 1;
        }
    }
}

/// A10 — named-ref index into [`RefFramePool::frame_refs`] /
/// [`RefFramePool::ref_frame_map`]. These are the 0-based
/// `cm->frame_refs[REFS_PER_FRAME]` slots (libvpx
/// `vp9_onyxc_int.h:126` `frame_refs[REFS_PER_FRAME]`), NOT the
/// 1-based `MV_REFERENCE_FRAME` enum (LAST_FRAME=1 etc. in
/// `vp9_blockd.h`). The mapping is `MV_REF - LAST_FRAME`:
/// `LAST_FRAME(1) - 1 = 0`, `GOLDEN_FRAME(2) - 1 = 1`,
/// `ALTREF_FRAME(3) - 1 = 2` (libvpx `vp9_blockd.h:48-52`).
pub const LAST_REF: usize = 0;
/// A10 — GOLDEN named-ref index (`GOLDEN_FRAME - LAST_FRAME = 1`).
pub const GOLDEN_REF: usize = 1;
/// A10 — ALTREF named-ref index (`ALTREF_FRAME - LAST_FRAME = 2`).
pub const ALTREF_REF: usize = 2;

/// A10 — one physical reference-frame buffer slot's pixel storage.
///
/// libvpx's `RefCntBuffer::buf` is a `YV12_BUFFER_CONFIG`
/// (`vp9_onyxc_int.h:64-91`); rvp9 carries tight-stride Y/U/V plane
/// `Vec<u8>`s at coded dimensions (4:2:0, 8-bit). Stride == width per
/// plane (the encoder pins width/height to multiples of 64, so there is
/// no border padding in these buffers — the loop filter operates on the
/// tight reconstruction in place).
#[derive(Debug, Clone, Default)]
pub struct RefSlotPlanes {
    /// Visible luma width carried by this slot.
    width: usize,
    /// Visible luma height carried by this slot.
    height: usize,
    /// Luma plane, `width * height`.
    pub y: Vec<u8>,
    /// Cb plane, `(width / 2) * (height / 2)`.
    pub u: Vec<u8>,
    /// Cr plane, `(width / 2) * (height / 2)`.
    pub v: Vec<u8>,
}

impl RefSlotPlanes {
    /// Allocate a zero-filled slot at the given coded dimensions.
    #[must_use]
    pub fn zeros(width: usize, height: usize) -> Self {
        let uvw = width / 2;
        let uvh = height / 2;
        Self {
            width,
            height,
            y: vec![0u8; width * height],
            u: vec![0u8; uvw * uvh],
            v: vec![0u8; uvw * uvh],
        }
    }

    fn resize_for_dimensions(&mut self, width: usize, height: usize) {
        let y_len = width * height;
        let uv_len = (width / 2) * (height / 2);
        self.y.resize(y_len, 0);
        self.u.resize(uv_len, 0);
        self.v.resize(uv_len, 0);
        self.width = width;
        self.height = height;
    }
}

/// Borrowed reference slot with the dimensions needed for scaled prediction.
#[derive(Clone, Copy, Debug)]
pub struct RefSlotView<'a> {
    /// Visible luma width.
    pub width: usize,
    /// Visible luma height.
    pub height: usize,
    /// Tight-stride luma plane.
    pub y: &'a [u8],
    /// Tight-stride Cb plane.
    pub u: &'a [u8],
    /// Tight-stride Cr plane.
    pub v: &'a [u8],
}

/// A10 — the full VP9 8-slot reference-frame buffer pool with
/// `ref_frame_map[8]` indirection, owning the encoder's
/// residual-coding reference pixels.
///
/// This is the libvpx-faithful generalisation of the encoder's prior
/// ad-hoc `last_recon_* / golden_recon_* / altref_recon_*` fields. It
/// mirrors the parts of `cm` (`VP9_COMMON`) that drive reference
/// selection and recycling:
///
/// - `slots[REF_FRAMES]` — the physical buffer pool. libvpx's
///   `cm->buffer_pool->frame_bufs[FRAME_BUFFERS]`
///   (`vp9_onyxc_int.h:88-105`). rvp9 sizes this at exactly
///   [`REF_FRAMES`] (8) physical buffers — the encoder commits one
///   reconstruction per coded frame and never holds a scaled-ref
///   alias, so the `+1 + REFS_PER_FRAME` headroom libvpx reserves for
///   the new-frame index and scaled refs is not needed for owned
///   pixel storage (the new frame is reconstructed into a local
///   scratch and committed into a slot at end-of-frame).
/// - `ref_frame_map[REF_FRAMES]` — slot index for each of the 8 named
///   ref-map entries (libvpx `cm->ref_frame_map[REF_FRAMES]`,
///   `vp9_onyxc_int.h:126`). `refresh_frame_flags` rotates these.
/// - `frame_refs[REFS_PER_FRAME]` — resolved slot index for LAST /
///   GOLDEN / ALTREF this frame. The encoder's internal model uses the
///   identity selector `frame_refs[i] = ref_frame_map[i]` (LAST →
///   map[0], GOLDEN → map[1], ALTREF → map[2]); see
///   [`RefFramePool::frame_refs`]. libvpx's full indirection
///   `cm->frame_refs[i].idx = cm->ref_frame_map[ref_frame_idx[i]]`
///   (`vp9_decodeframe.c:1206`) collapses to the identity here because
///   the encoder pins each named ref to its own non-rotating map slot.
/// - `new_fb_idx` — the slot the current frame's reconstruction is
///   committed into (libvpx `cm->new_fb_idx`).
///
/// ## Default-path slot mapping (gate-safe refactor invariant)
///
/// The bitstream writer emits `ref_frame_indices = [0, 0, 0]` (default)
/// / `[0, 0, 2]` (ALTREF) and `refresh_frame_flags`: `0xFF` on keyframes
/// (all 8 slots), `0x01` on LAST-only inter frames, `0x03` on
/// GOLDEN-refresh inter frames, `0x04` on the non-show ARF. The encoder
/// reads named ref `i` from `ref_frame_map[i]`, reproducing the prior
/// ad-hoc `last_recon_* / golden_recon_* / altref_recon_*` separate
/// buffers exactly (LAST every frame, GOLDEN only on bit-1 refresh,
/// ALTREF only on bit-2 refresh) — the pool is a faithful re-indexing,
/// not a behaviour change. The cross-decoder gate
/// stays byte-exact.
#[derive(Debug, Clone)]
pub struct RefFramePool {
    /// Physical buffer slots, indexed `0..REF_FRAMES`.
    slots: [RefSlotPlanes; REF_FRAMES],
    /// `cm->ref_frame_map[8]` — slot index for each named ref-map
    /// entry. After [`Self::keyframe_reset`] all 8 point at the KF
    /// slot; `update_reference_frames` rotates them.
    ref_frame_map: [usize; REF_FRAMES],
    /// `cm->new_fb_idx` — slot holding the frame currently being
    /// committed.
    new_fb_idx: usize,
    /// Coded luma width (slot Y-plane stride).
    width: usize,
    /// Coded luma height.
    height: usize,
}

impl RefFramePool {
    /// Allocate a pool sized for `width × height` coded luma (4:2:0).
    /// All slots start zeroed; `ref_frame_map` and `new_fb_idx` start
    /// at 0. The first encoded frame is always a keyframe, which calls
    /// [`Self::keyframe_reset`] before any read.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            slots: core::array::from_fn(|_| RefSlotPlanes::zeros(width, height)),
            ref_frame_map: [0; REF_FRAMES],
            new_fb_idx: 0,
            width,
            height,
        }
    }

    /// Slot index each named ref (LAST/GOLDEN/ALTREF) resolves to this
    /// frame: `frame_refs[i] = ref_frame_map[i]`.
    ///
    /// ## Named-ref → map-slot mapping (encoder internal model)
    ///
    /// The encoder reads named ref `i` from `ref_frame_map[i]` — the
    /// identity selector LAST→map[0], GOLDEN→map[1], ALTREF→map[2].
    /// This reproduces the prior ad-hoc fields exactly: `last_recon_*`
    /// rotated every frame (map[0], refresh bit 0); `golden_recon_*`
    /// rotated only on GOLDEN-refresh (map[1], refresh bit 1);
    /// `altref_recon_*` rotated only on the ARF (map[2], refresh bit 2).
    ///
    /// NOTE: this is the encoder's INTERNAL reference-selection model,
    /// distinct from the bitstream's `ref_frame_indices[3]` header field
    /// (which the writer emits as `[0, 0, 0]` on the default path / `[0,
    /// 0, 2]` with ALTREF). The encoder keeps GOLDEN/ALTREF on their own
    /// non-rotating slots so the picker sees the same reference pixels
    /// the prior separate-buffer fields held — preserving byte-identity.
    /// libvpx's `cm->frame_refs[i].idx = cm->ref_frame_map[
    /// ref_frame_idx[i]]` (`vp9_decodeframe.c:1206`) collapses to the
    /// identity here because the encoder's per-frame `ref_frame_idx`
    /// model is the named-ref index itself.
    #[must_use]
    pub fn frame_refs(&self) -> [usize; REFS_PER_FRAME] {
        core::array::from_fn(|i| self.ref_frame_map[i])
    }

    /// Read-only view of `ref_frame_map[8]` (slot index per named-ref
    /// map entry). Mirrors libvpx `cm->ref_frame_map`.
    #[must_use]
    pub fn ref_frame_map(&self) -> [usize; REF_FRAMES] {
        self.ref_frame_map
    }

    /// `cm->new_fb_idx` — slot the next reconstruction commits into.
    #[must_use]
    pub fn new_fb_idx(&self) -> usize {
        self.new_fb_idx
    }

    /// Immutable Y/U/V planes for a named ref (LAST/GOLDEN/ALTREF),
    /// reading `&slots[ref_frame_map[which]]`. Multiple named refs may
    /// resolve to the same physical slot (e.g. right after a keyframe,
    /// where all three alias the KF slot) — that is sound because these
    /// are shared immutable borrows. See [`Self::frame_refs`] for the
    /// named-ref → map-slot mapping rationale.
    #[must_use]
    pub fn named_ref_planes(&self, which: usize) -> (&[u8], &[u8], &[u8]) {
        let view = self.named_ref_view(which);
        (view.y, view.u, view.v)
    }

    /// Immutable planes and visible dimensions for one named reference.
    ///
    /// Spatial SVC may keep a lower-layer reconstruction and a current-layer
    /// reconstruction in different map slots. Callers use these dimensions to
    /// select identity or scaled motion compensation without inferring geometry
    /// from a plane's byte length.
    #[must_use]
    pub fn named_ref_view(&self, which: usize) -> RefSlotView<'_> {
        let p = &self.slots[self.ref_frame_map[which]];
        RefSlotView {
            width: p.width,
            height: p.height,
            y: &p.y,
            u: &p.u,
            v: &p.v,
        }
    }

    /// libvpx `vp9_setup_past_independence` reference reset
    /// (`vp9_entropymode.c:424-437`, driven from
    /// `vp9_encoder.c::vp9_setup_past_independence`). A keyframe
    /// implicitly refreshes every reference slot (the encoder emits
    /// `refresh_frame_flags = 0xFF`): the freshly-reconstructed KF lands
    /// in `new_fb_idx` and all 8 `ref_frame_map[]` entries point at it,
    /// so LAST/GOLDEN/ALTREF all resolve to the KF reconstruction.
    ///
    /// Picks a fresh `new_fb_idx` distinct from the previous one (a
    /// minimal two-slot toggle is enough: keyframes broadcast a single
    /// slot across the whole map, so only one live slot is ever needed,
    /// but toggling makes the rotation observable in tests and matches
    /// libvpx's `get_free_fb` "never reuse the live slot" guarantee for
    /// the common single-KF-then-inters GOP).
    ///
    /// The caller writes the KF reconstruction into the new slot via
    /// [`Self::new_frame_planes_mut`] after this call returns.
    pub fn keyframe_reset(&mut self) {
        self.keyframe_reset_sized(self.width, self.height);
    }

    /// Spatial counterpart of [`Self::keyframe_reset`]. The broadcast slot is
    /// resized to the base layer before its reconstruction is written, while
    /// later enhancement components may commit different dimensions into
    /// other map entries.
    pub(crate) fn keyframe_reset_sized(&mut self, width: usize, height: usize) {
        validate_i420_dims(width, height);
        let new_idx = (self.new_fb_idx + 1) % REF_FRAMES;
        self.new_fb_idx = new_idx;
        self.slots[new_idx].resize_for_dimensions(width, height);
        self.ref_frame_map = [new_idx; REF_FRAMES];
    }

    /// Mutable Y/U/V planes for `new_fb_idx` — the slot the current
    /// frame reconstructs into. Used by the keyframe path (which writes
    /// the reconstruction directly into the committed slot). The inter
    /// path reconstructs into a local scratch and copies it in via
    /// [`Self::commit_new_frame`] so the previous LAST slot stays
    /// readable during the encode.
    #[must_use]
    pub fn new_frame_planes_mut(&mut self) -> (&mut [u8], &mut [u8], &mut [u8]) {
        let p = &mut self.slots[self.new_fb_idx];
        (&mut p.y, &mut p.u, &mut p.v)
    }

    /// Acquire a fresh `new_fb_idx` for an inter frame, copy the
    /// supplied reconstruction into it, and return the slot index. The
    /// new slot is chosen so it is not currently referenced by any
    /// `ref_frame_map[]` entry that this frame reads from
    /// (`frame_refs`), so the previous LAST/GOLDEN/ALTREF pixels stay
    /// intact for `update_reference_frames` bookkeeping. libvpx
    /// `get_free_fb` (`vp9_onyxc_int.h:332-347`) picks the first
    /// unreferenced pool slot; rvp9 picks the first slot not currently
    /// in `ref_frame_map[]`, falling back to round-robin if all 8 are
    /// live (only possible with 8 distinct refs, which the default path
    /// never produces).
    pub fn commit_new_frame(&mut self, recon: (&[u8], &[u8], &[u8])) -> usize {
        self.commit_new_frame_sized(self.width, self.height, recon)
    }

    /// Commit a reconstruction whose dimensions may differ from other live
    /// slots. This is the shared-map operation spatial SVC needs when a lower
    /// component and an enhancement component refresh different map entries.
    pub(crate) fn commit_new_frame_sized(
        &mut self,
        width: usize,
        height: usize,
        recon: (&[u8], &[u8], &[u8]),
    ) -> usize {
        validate_i420_planes(width, height, recon);
        let new_idx = self.get_free_slot();
        self.new_fb_idx = new_idx;
        let p = &mut self.slots[new_idx];
        p.resize_for_dimensions(width, height);
        p.y.copy_from_slice(recon.0);
        p.u.copy_from_slice(recon.1);
        p.v.copy_from_slice(recon.2);
        new_idx
    }

    /// Exchange owned tight-stride reconstruction planes with a fresh slot.
    ///
    /// The returned vectors are the displaced unreferenced slot storage and
    /// can be reused as next-frame scratch, so neither side is copied or
    /// deallocated.
    ///
    /// # Panics
    ///
    /// Panics if any plane length differs from the configured 4:2:0 size.
    pub(crate) fn exchange_new_frame(&mut self, recon: OwnedYuv420) -> (usize, OwnedYuv420) {
        self.exchange_new_frame_sized(self.width, self.height, recon)
    }

    fn exchange_new_frame_sized(
        &mut self,
        width: usize,
        height: usize,
        recon: OwnedYuv420,
    ) -> (usize, OwnedYuv420) {
        validate_i420_planes(width, height, (&recon.0, &recon.1, &recon.2));

        let new_idx = self.get_free_slot();
        self.new_fb_idx = new_idx;
        let p = &mut self.slots[new_idx];
        p.width = width;
        p.height = height;
        let mut displaced = (
            std::mem::replace(&mut p.y, recon.0),
            std::mem::replace(&mut p.u, recon.1),
            std::mem::replace(&mut p.v, recon.2),
        );
        // The returned storage is next-frame scratch. A heterogeneous slot
        // may have different lengths, so normalize only on a dimension
        // transition; the fixed-size production path remains allocation-free.
        displaced.0.resize(width * height, 0);
        let uv_len = (width / 2) * (height / 2);
        displaced.1.resize(uv_len, 0);
        displaced.2.resize(uv_len, 0);
        (new_idx, displaced)
    }

    /// libvpx `get_free_fb` analogue (`vp9_onyxc_int.h:332-347`): the
    /// first slot not currently held by any `ref_frame_map[]` entry.
    /// Falls back to `(new_fb_idx + 1) % REF_FRAMES` if every slot is
    /// referenced (not reachable on the default path, which never holds
    /// 8 distinct refs).
    fn get_free_slot(&self) -> usize {
        for cand in 0..REF_FRAMES {
            if !self.ref_frame_map.contains(&cand) {
                return cand;
            }
        }
        (self.new_fb_idx + 1) % REF_FRAMES
    }

    /// libvpx `vp9_update_reference_frames` ref-map rotation
    /// (`vp9_encoder.c:3382` entry; the `ref_cnt_fb(... &cm->
    /// ref_frame_map[...], cm->new_fb_idx)` rotation body is
    /// `vp9_encoder.c:3312-3367`). Walks the 8-bit `refresh_frame_flags`:
    /// every set bit `i` points `ref_frame_map[i]` at `new_fb_idx` (the
    /// slot just committed).
    /// Clear bits leave their slot untouched, so a `0x01` LAST-only
    /// refresh advances LAST while GOLDEN/ALTREF keep their prior slot.
    ///
    /// Must be called AFTER [`Self::commit_new_frame`] (which sets
    /// `new_fb_idx`). Named-ref reads then resolve through
    /// [`Self::named_ref_planes`] (`ref_frame_map[which]`).
    pub fn update_reference_frames(&mut self, refresh_frame_flags: u8) {
        for i in 0..REF_FRAMES {
            if (refresh_frame_flags >> i) & 1 == 1 {
                self.ref_frame_map[i] = self.new_fb_idx;
            }
        }
    }

    /// Coded luma dimensions this pool was allocated for.
    #[must_use]
    pub fn dims(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

fn validate_i420_dims(width: usize, height: usize) {
    assert!(
        width > 0 && height > 0,
        "reference dimensions must be non-zero"
    );
    assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "4:2:0 reference dimensions must be even",
    );
}

fn validate_i420_planes(width: usize, height: usize, planes: (&[u8], &[u8], &[u8])) {
    validate_i420_dims(width, height);
    let expected_y = width * height;
    let expected_uv = (width / 2) * (height / 2);
    assert_eq!(planes.0.len(), expected_y, "owned Y plane size mismatch");
    assert_eq!(planes.1.len(), expected_uv, "owned U plane size mismatch");
    assert_eq!(planes.2.len(), expected_uv, "owned V plane size mismatch");
}

#[cfg(test)]
mod tests {
    //! PROOF tests for W1.1 / W1.2 / W1.3 — the libvpx-faithful
    //! buffer pool + ref-frame manager.

    use super::*;

    /// W1.1 PROOF — constants match libvpx's
    /// `vp9_onyxc_int.h:35-42`.
    #[test]
    fn w1_1_constants_match_libvpx() {
        assert_eq!(REF_FRAMES_LOG2, 3);
        assert_eq!(REF_FRAMES, 8);
        assert_eq!(REFS_PER_FRAME, 3);
        assert_eq!(FRAME_BUFFERS, 12);
        assert_eq!(INVALID_IDX, -1);
    }

    /// W1.1 PROOF — fresh pool has all slots at refcount 0.
    #[test]
    fn w1_1_pool_starts_empty() {
        let pool = BufferPool::new();
        for slot in &pool.frame_bufs {
            assert_eq!(slot.ref_count, 0);
        }
    }

    /// W1.2 PROOF — `get_free_fb` returns the first free slot and
    /// sets its refcount to 1 (libvpx `vp9_onyxc_int.h:336-344`).
    /// Anti-G3: if the function ever silently returned the same
    /// slot twice without bumping refcount, this test catches it.
    #[test]
    fn w1_2_get_free_fb_returns_first_free_and_sets_refcount() {
        let mut pool = BufferPool::new();
        let a = pool.get_free_fb(/* frame_index = */ 0);
        assert_eq!(a, 0);
        assert_eq!(pool.frame_bufs[0].ref_count, 1);
        let b = pool.get_free_fb(1);
        assert_eq!(b, 1);
        assert_eq!(pool.frame_bufs[1].ref_count, 1);
    }

    /// W1.2 PROOF — `get_free_fb` returns INVALID_IDX when all 12
    /// slots are held.
    #[test]
    fn w1_2_get_free_fb_exhaustion_returns_invalid_idx() {
        let mut pool = BufferPool::new();
        for i in 0..FRAME_BUFFERS {
            assert_eq!(pool.get_free_fb(i as u64), i as i32);
        }
        assert_eq!(pool.get_free_fb(99), INVALID_IDX);
    }

    /// W1.2 PROOF — `ref_cnt_fb` matches libvpx semantics:
    /// decrement old, install new, increment new
    /// (`vp9_onyxc_int.h:349-358`).
    #[test]
    fn w1_2_ref_cnt_fb_moves_refcount_atomically() {
        let mut pool = BufferPool::new();
        let a = pool.get_free_fb(0); // pool[0].rc = 1
        let b = pool.get_free_fb(1); // pool[1].rc = 1
        let mut slot: i32 = a;
        pool.ref_cnt_fb(&mut slot, b);
        // After: old (pool[0]) dropped to 0, new (pool[1]) up to 2.
        assert_eq!(slot, b);
        assert_eq!(pool.frame_bufs[a as usize].ref_count, 0);
        assert_eq!(pool.frame_bufs[b as usize].ref_count, 2);
    }

    /// W1.2 PROOF — `decrease_ref_count` saturates at zero (libvpx
    /// `vp9_decoder.c:33-45`).
    #[test]
    fn w1_2_decrease_ref_count_saturates_at_zero() {
        let mut pool = BufferPool::new();
        let a = pool.get_free_fb(0);
        pool.decrease_ref_count(a);
        assert_eq!(pool.frame_bufs[a as usize].ref_count, 0);
        // Calling again must NOT underflow.
        pool.decrease_ref_count(a);
        assert_eq!(pool.frame_bufs[a as usize].ref_count, 0);
        // INVALID_IDX is a no-op.
        pool.decrease_ref_count(INVALID_IDX);
    }

    /// W1.3 PROOF — `keyframe_reset` allocates a fresh slot,
    /// broadcasts it across `ref_frame_map[]`, and points all named
    /// refs at it. Refcount bookkeeping: 1 (get_free_fb) + 3
    /// (lst/gld/alt) + 8 (ref_frame_map) = 12.
    #[test]
    fn w1_3_keyframe_reset_broadcasts_kf_pool_idx() {
        let mut pool = BufferPool::new();
        let mut mgr = RefFrameManager::new();
        mgr.keyframe_reset(&mut pool, /* frame_index = */ 0);
        let kf_idx = mgr.new_fb_idx;
        assert!(kf_idx >= 0);
        assert_eq!(mgr.lst_fb_idx, kf_idx);
        assert_eq!(mgr.gld_fb_idx, kf_idx);
        assert_eq!(mgr.alt_fb_idx, kf_idx);
        for slot in &mgr.ref_frame_map {
            assert_eq!(*slot, kf_idx);
        }
        // 1 + 3 + 8 = 12 references.
        assert_eq!(pool.frame_bufs[kf_idx as usize].ref_count, 12);
    }

    /// W1.1 PROOF — `update_ref_frame_map` rotates the 8-bit mask
    /// (libvpx `vp9_encoder.c:3297-3349`). Refreshing slot 0
    /// (LASTFRAME) replaces the old entry with the new-frame index.
    #[test]
    fn w1_1_update_ref_frame_map_refreshes_per_bit() {
        let mut pool = BufferPool::new();
        let mut mgr = RefFrameManager::new();
        mgr.keyframe_reset(&mut pool, 0);
        let kf = mgr.new_fb_idx;
        // Allocate a new frame buffer (simulating an inter frame).
        let inter = pool.get_free_fb(1);
        mgr.new_fb_idx = inter;
        // Refresh slot 0 only (bit 0 set = refresh LASTFRAME).
        mgr.update_ref_frame_map(&mut pool, 0x01);
        assert_eq!(mgr.ref_frame_map[0], inter);
        // Slots 1..8 still point at the KF index.
        for i in 1..REF_FRAMES {
            assert_eq!(mgr.ref_frame_map[i], kf);
        }
    }

    /// W1.1 PROOF — 30-frame ref-ring rotation. Simulates 30 inter
    /// frames refreshing LASTFRAME (bit 0) only. After the sequence
    /// the LAST slot must hold the most recent inter frame's pool
    /// index; GOLDEN/ALTREF must still point at the KF; no pool
    /// slot's refcount should underflow.
    #[test]
    fn w1_1_thirty_frame_ref_ring_rotation() {
        let mut pool = BufferPool::new();
        let mut mgr = RefFrameManager::new();
        mgr.keyframe_reset(&mut pool, 0);
        let kf = mgr.new_fb_idx;
        let mut last_inter = INVALID_IDX;
        for f in 1..=30u64 {
            // Drop the previous new_fb_idx if it wasn't promoted to a
            // ref slot (libvpx does this in
            // `vp9_release_scaled_references`); here we just
            // decrement the new-frame refcount once per inter frame
            // to mirror libvpx's net effect.
            if last_inter >= 0 {
                pool.decrease_ref_count(last_inter);
            }
            let new = pool.get_free_fb(f);
            assert!(
                new >= 0,
                "frame {f}: pool exhausted (rotation must release stale slots)"
            );
            mgr.new_fb_idx = new;
            mgr.update_ref_frame_map(&mut pool, 0x01);
            last_inter = new;
        }
        // After 30 inter frames refreshing slot 0, ref_frame_map[0]
        // == last_inter, and slots 1..8 still point at the KF.
        assert_eq!(mgr.ref_frame_map[0], last_inter);
        for i in 1..REF_FRAMES {
            assert_eq!(mgr.ref_frame_map[i], kf, "slot {i} drifted");
        }
        // Pool sanity: no refcount overflowed wildly. The KF slot
        // still has 1 (new) + 0 (lst no longer points at it) + 0
        // (gld/alt unchanged but unrefcounted by these helpers) + 7
        // (slots 1..8 of ref_frame_map). Actually because
        // keyframe_reset bumped gld/alt/lst (3) + ref_frame_map (8)
        // and lst was rotated off, the KF slot count is:
        //   1 (get_free_fb in keyframe_reset)
        //   + 3 (lst/gld/alt manual increments in keyframe_reset)
        //   + 8 (ref_frame_map increments)
        //   - 1 (slot 0 rotated off in first inter)
        // = 11.
        assert_eq!(pool.frame_bufs[kf as usize].ref_count, 11);
    }

    // -- A10 PROOF — `RefFramePool` (the 8-slot pixel-owning pool with
    //    `ref_frame_map[8]` indirection). --

    /// A10 PROOF — after a keyframe, all 8 `ref_frame_map[]` entries
    /// point at the single KF slot, so LAST/GOLDEN/ALTREF all resolve
    /// to it (`frame_refs = [kf, kf, kf]`). libvpx
    /// `vp9_setup_past_independence` (`refresh_frame_flags = 0xFF`).
    #[test]
    fn a10_keyframe_reset_broadcasts_one_slot() {
        let mut pool = RefFramePool::new(64, 64);
        pool.keyframe_reset();
        let kf = pool.new_fb_idx();
        for (i, &m) in pool.ref_frame_map().iter().enumerate() {
            assert_eq!(m, kf, "ref_frame_map[{i}] must point at the KF slot");
        }
        assert_eq!(pool.frame_refs(), [kf, kf, kf]);
    }

    /// A10 PROOF — `update_reference_frames` rotates per the 8-bit
    /// `refresh_frame_flags` mask. A `0x01` LAST-only refresh advances
    /// `ref_frame_map[0]` to the committed slot while GOLDEN/ALTREF map
    /// entries stay pinned to the KF slot. libvpx
    /// `update_reference_frames` (`vp9_encoder.c:3297-3349`).
    /// Anti-G3: the resolved `frame_refs` must change for LAST and stay
    /// for GOLDEN/ALTREF — a stubbed rotation that touched no slot, or
    /// one that touched all slots, fails this.
    #[test]
    fn a10_update_reference_frames_last_only_rotation() {
        let mut pool = RefFramePool::new(64, 64);
        pool.keyframe_reset();
        let kf = pool.new_fb_idx();
        // Commit an inter frame and refresh LAST only (0x01).
        let recon = vec![7u8; 64 * 64];
        let recon_uv = vec![7u8; 32 * 32];
        let new_slot = pool.commit_new_frame((&recon, &recon_uv, &recon_uv));
        assert_ne!(new_slot, kf, "inter commit must pick a fresh slot");
        pool.update_reference_frames(0x01);
        // LAST advanced; GOLDEN + ALTREF still at the KF slot.
        assert_eq!(pool.ref_frame_map()[LAST_REF], new_slot);
        assert_eq!(pool.frame_refs()[LAST_REF], new_slot);
        assert_eq!(pool.frame_refs()[GOLDEN_REF], kf);
        assert_eq!(pool.frame_refs()[ALTREF_REF], kf);
        // The LAST slot must now hold the committed pixels (7s).
        let (ly, _, _) = pool.named_ref_planes(LAST_REF);
        assert!(ly.iter().all(|&p| p == 7));
        // GOLDEN still holds the KF slot's pixels (0s — never written).
        let (gy, _, _) = pool.named_ref_planes(GOLDEN_REF);
        assert!(gy.iter().all(|&p| p == 0));
    }

    #[test]
    fn spatial_ref_map_preserves_heterogeneous_slot_dimensions() {
        let mut pool = RefFramePool::new(128, 64);
        pool.keyframe_reset_sized(64, 32);
        {
            let (y, u, v) = pool.new_frame_planes_mut();
            y.fill(3);
            u.fill(5);
            v.fill(7);
        }

        let base_slot = pool.new_fb_idx();
        let base = pool.named_ref_view(LAST_REF);
        assert_eq!((base.width, base.height), (64, 32));
        assert_eq!((base.y.len(), base.u.len(), base.v.len()), (2048, 512, 512));

        let enhancement_y = vec![11; 128 * 64];
        let enhancement_u = vec![13; 64 * 32];
        let enhancement_v = vec![17; 64 * 32];
        let enhancement_slot =
            pool.commit_new_frame_sized(128, 64, (&enhancement_y, &enhancement_u, &enhancement_v));
        pool.update_reference_frames(0x02);

        assert_ne!(enhancement_slot, base_slot);
        let last = pool.named_ref_view(LAST_REF);
        let golden = pool.named_ref_view(GOLDEN_REF);
        assert_eq!((last.width, last.height), (64, 32));
        assert_eq!((golden.width, golden.height), (128, 64));
        assert!(last.y.iter().all(|&pixel| pixel == 3));
        assert!(golden.y.iter().all(|&pixel| pixel == 11));
        assert_eq!(pool.ref_frame_map()[LAST_REF], base_slot);
        assert_eq!(pool.ref_frame_map()[GOLDEN_REF], enhancement_slot);
    }

    #[test]
    fn exchange_commit_reuses_both_plane_allocations() {
        let mut pool = RefFramePool::new(64, 64);
        let y = vec![1u8; 64 * 64];
        let u = vec![2u8; 32 * 32];
        let v = vec![3u8; 32 * 32];
        let input_pointers = (y.as_ptr(), u.as_ptr(), v.as_ptr());
        let slot = pool.get_free_slot();
        let displaced_pointers = (
            pool.slots[slot].y.as_ptr(),
            pool.slots[slot].u.as_ptr(),
            pool.slots[slot].v.as_ptr(),
        );

        let (committed, displaced) = pool.exchange_new_frame((y, u, v));

        assert_eq!(committed, slot);
        assert_eq!(pool.slots[slot].y.as_ptr(), input_pointers.0);
        assert_eq!(pool.slots[slot].u.as_ptr(), input_pointers.1);
        assert_eq!(pool.slots[slot].v.as_ptr(), input_pointers.2);
        assert_eq!(displaced.0.as_ptr(), displaced_pointers.0);
        assert_eq!(displaced.1.as_ptr(), displaced_pointers.1);
        assert_eq!(displaced.2.as_ptr(), displaced_pointers.2);
    }

    /// A10 PROOF — `0x03` (LAST + GOLDEN) refresh rotates both
    /// `ref_frame_map[0]` and `ref_frame_map[1]` to the committed slot,
    /// so GOLDEN now resolves to the same pixels as LAST (the
    /// GOLDEN-refresh path that subsumed the old
    /// `golden_recon.copy_from_slice(&last_recon)`).
    #[test]
    fn a10_update_reference_frames_golden_refresh() {
        let mut pool = RefFramePool::new(64, 64);
        pool.keyframe_reset();
        let recon = vec![9u8; 64 * 64];
        let recon_uv = vec![9u8; 32 * 32];
        let new_slot = pool.commit_new_frame((&recon, &recon_uv, &recon_uv));
        pool.update_reference_frames(0x03);
        assert_eq!(pool.ref_frame_map()[LAST_REF], new_slot);
        assert_eq!(pool.ref_frame_map()[GOLDEN_REF], new_slot);
        assert_eq!(pool.frame_refs()[LAST_REF], pool.frame_refs()[GOLDEN_REF]);
        let (gy, _, _) = pool.named_ref_planes(GOLDEN_REF);
        assert!(
            gy.iter().all(|&p| p == 9),
            "GOLDEN must hold refreshed pixels"
        );
    }

    /// A10 PROOF — a long LAST-only ref-ring rotation never exhausts the
    /// 8 physical slots (the free-slot picker recycles slots no longer
    /// in the map) and keeps GOLDEN/ALTREF pinned to the KF slot.
    #[test]
    fn a10_thirty_frame_last_ring_no_exhaustion() {
        let mut pool = RefFramePool::new(64, 64);
        pool.keyframe_reset();
        let kf = pool.new_fb_idx();
        let mut prev = kf;
        for f in 1..=30u8 {
            let recon = vec![f; 64 * 64];
            let recon_uv = vec![f; 32 * 32];
            let new_slot = pool.commit_new_frame((&recon, &recon_uv, &recon_uv));
            assert_ne!(new_slot, prev, "frame {f}: must not reuse the LAST slot");
            pool.update_reference_frames(0x01);
            prev = pool.ref_frame_map()[LAST_REF];
        }
        // GOLDEN + ALTREF still resolve to the KF slot (only LAST rotated).
        assert_eq!(pool.frame_refs()[GOLDEN_REF], kf);
        assert_eq!(pool.frame_refs()[ALTREF_REF], kf);
        // LAST holds the last frame's pixels (30).
        let (ly, _, _) = pool.named_ref_planes(LAST_REF);
        assert!(ly.iter().all(|&p| p == 30));
    }

    /// A10 PROOF — the ARF (`refresh_frame_flags = 0x04`, bit 2) rotates
    /// ONLY the ALTREF map entry. After the commit, `named_ref_planes(
    /// ALTREF_REF)` resolves through `ref_frame_map[2]` (the ARF slot),
    /// while LAST (map[0]) / GOLDEN (map[1]) keep their prior slot. This
    /// is the encoder's named-ref → map-slot identity model: ALTREF
    /// reads its own non-rotating slot, just like the prior
    /// `altref_recon_*` field. Anti-G3: a stubbed ARF commit that left
    /// `ref_frame_map[2]` at the KF slot would read 0s here, not 5s.
    #[test]
    fn a10_arf_commit_rotates_only_altref() {
        let mut pool = RefFramePool::new(64, 64);
        pool.keyframe_reset();
        let kf = pool.new_fb_idx();
        // Commit an ARF and refresh ALTREF only (0x04 = bit 2).
        let arf = vec![5u8; 64 * 64];
        let arf_uv = vec![5u8; 32 * 32];
        let arf_slot = pool.commit_new_frame((&arf, &arf_uv, &arf_uv));
        assert_ne!(arf_slot, kf, "ARF commit must pick a fresh slot");
        pool.update_reference_frames(0x04);
        assert_eq!(pool.ref_frame_map()[ALTREF_REF], arf_slot);
        // LAST + GOLDEN still resolve to the KF slot (0s, never written).
        assert_eq!(pool.frame_refs()[LAST_REF], kf);
        assert_eq!(pool.frame_refs()[GOLDEN_REF], kf);
        assert!(pool.named_ref_planes(LAST_REF).0.iter().all(|&p| p == 0));
        // ALTREF now resolves to the ARF slot (5s).
        assert_eq!(pool.frame_refs()[ALTREF_REF], arf_slot);
        assert!(pool.named_ref_planes(ALTREF_REF).0.iter().all(|&p| p == 5));
    }
}
