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
}
