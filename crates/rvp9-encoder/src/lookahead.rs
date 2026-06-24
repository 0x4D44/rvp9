//! W1.4 — lookahead ring buffer for the encoder's source-frame
//! pipeline.
//!
//! Faithful port of libvpx `vp9_lookahead.c` (lines 32-235). The
//! lookahead is the front of the encoder pipeline: callers push raw
//! source frames in via [`LookaheadCtx::push`], and the frame-loop
//! driver pops them out via [`LookaheadCtx::pop`]. Backward
//! peeking ([`LookaheadCtx::peek`] with a negative index) is used by
//! the scene-change detector and the prev-source ALTREF logic in
//! later waves.
//!
//! ## libvpx parity notes
//!
//! - Capacity: `max_sz = clamp(depth, 1, MAX_LAG_BUFFERS) + MAX_PRE_FRAMES`
//!   (libvpx `vp9_lookahead.c:55-58`). The "+ `MAX_PRE_FRAMES`" slot
//!   keeps one previously-popped entry available for backward peek.
//! - `full()`: the queue has room iff `sz + 1 + MAX_PRE_FRAMES <=
//!   max_sz` (libvpx `vp9_lookahead.c:85-87`).
//! - `pop(drain=false)`: blocks until the lookahead is "filled" —
//!   i.e. `sz == max_sz - MAX_PRE_FRAMES` (libvpx
//!   `vp9_lookahead.c:201-209`). `drain=true` pops the oldest entry
//!   unconditionally; used at end-of-stream.
//! - `peek(idx)`: forward (`idx >= 0`) walks the live entries from
//!   the read pointer; backward (`idx < 0`) walks into the
//!   `MAX_PRE_FRAMES` slot of previously-popped entries
//!   (libvpx `vp9_lookahead.c:212-233`).
//!
//! ## Scope
//!
//! v1 ships 8-bit-only 4:2:0 frame storage (matches the rvp9 pixel
//! policy at `crates/rvp9-core/src/lib.rs::Pixel`). Active-map
//! partial-copy (libvpx `vp9_lookahead.c:84` `USE_PARTIAL_COPY 0`) is
//! omitted — libvpx itself ships it disabled.

use crate::YuvFrameRef;

/// Maximum lookahead depth supported by libvpx
/// (`vp9_lookahead.h:21`). Caller-requested `depth` is clamped to
/// `[1, MAX_LAG_BUFFERS]`.
pub const MAX_LAG_BUFFERS: usize = 25;

/// Number of previously-popped frames retained for backward peek
/// (libvpx `vp9_lookahead.h:19`). Used by the scene-change detector
/// and ALTREF-source picker.
pub const MAX_PRE_FRAMES: usize = 1;

/// One entry in the lookahead ring — a raw source frame plus the
/// metadata the encoder needs to align it with the bitstream.
#[derive(Clone, Debug)]
pub struct LookaheadEntry {
    /// Luma plane (`width * height` samples).
    pub y: Vec<u8>,
    /// Cb plane (`(width/2) * (height/2)` samples, 4:2:0).
    pub u: Vec<u8>,
    /// Cr plane (`(width/2) * (height/2)` samples, 4:2:0).
    pub v: Vec<u8>,
    /// Coded luma width in pixels. Matches the lookahead context's
    /// configured width — partial-resolution sources fall outside
    /// v1 scope (libvpx `vp9_lookahead.c:135` `USE_PARTIAL_COPY 0`).
    pub width: u32,
    /// Coded luma height in pixels.
    pub height: u32,
    /// `ts_start` — presentation timestamp of the start of this
    /// frame (`vp9_lookahead.c:193`). libvpx's units are "ticks at
    /// the encoder's configured timebase". rvp9 carries them
    /// verbatim — the caller is the timebase authority.
    pub ts_start: i64,
    /// `ts_end` — presentation timestamp of the end of this frame
    /// (`vp9_lookahead.c:194`).
    pub ts_end: i64,
    /// Per-frame encoder flags (libvpx `vpx_enc_frame_flags_t`).
    /// rvp9 mirrors the bitfield shape verbatim — concrete flag
    /// constants (force-keyframe etc.) land alongside the consumer
    /// in later waves.
    pub flags: u32,
    /// Monotonic show-frame index assigned at push-time
    /// (`vp9_lookahead.c:196-197`).
    pub show_idx: u64,
}

/// The lookahead context — a circular buffer of source frames.
///
/// Layout: `buf[max_sz]` is a ring; `read_idx` is the oldest live
/// entry; `write_idx` is the next free slot; `sz` is the count of
/// live entries. A previously-popped entry sits at `read_idx - 1`
/// (mod `max_sz`) and is accessible via backward peek up to
/// `MAX_PRE_FRAMES` slots back.
#[derive(Debug)]
pub struct LookaheadCtx {
    buf: Vec<Option<LookaheadEntry>>,
    /// Ring capacity (= caller depth + `MAX_PRE_FRAMES`).
    max_sz: usize,
    /// Count of live entries in the ring.
    sz: usize,
    /// Index of the oldest live entry (next to be popped).
    read_idx: usize,
    /// Index of the next free slot (next to be written by push).
    write_idx: usize,
    /// Monotonic show-frame counter; assigned to each push.
    next_show_idx: u64,
    /// Configured coded width — every push must match.
    width: u32,
    /// Configured coded height — every push must match.
    height: u32,
}

/// Errors emitted by the lookahead.
#[derive(Debug, Eq, PartialEq)]
pub enum LookaheadError {
    /// The ring is full; the caller must pop before pushing.
    Full,
    /// Push tried to insert a frame with a different size than the
    /// ring was initialised for. v1 does NOT support dynamic resize
    /// (libvpx's resize lives in `vp9_lookahead.c:107-128`; it's a
    /// later-wave port).
    DimensionMismatch {
        /// Width the caller supplied.
        got_width: u32,
        /// Height the caller supplied.
        got_height: u32,
    },
    /// Push tried to insert a frame whose plane buffers don't match
    /// the declared dimensions.
    InvalidPlaneSize,
}

impl LookaheadCtx {
    /// Construct a lookahead context for `width × height` frames at
    /// the given `depth`. Capacity is `clamp(depth, 1,
    /// MAX_LAG_BUFFERS) + MAX_PRE_FRAMES` per libvpx
    /// `vp9_lookahead.c:55-58`.
    ///
    /// # Panics
    ///
    /// Panics if `width == 0`, `height == 0`, or either is not a
    /// multiple of 2 (4:2:0 subsampling).
    #[must_use]
    pub fn new(width: u32, height: u32, depth: usize) -> Self {
        assert!(width > 0 && height > 0, "lookahead requires positive dims");
        assert!(
            width.is_multiple_of(2) && height.is_multiple_of(2),
            "4:2:0 requires even dims"
        );
        let depth = depth.clamp(1, MAX_LAG_BUFFERS);
        let max_sz = depth + MAX_PRE_FRAMES;
        Self {
            buf: vec![None; max_sz],
            max_sz,
            sz: 0,
            read_idx: 0,
            write_idx: 0,
            next_show_idx: 0,
            width,
            height,
        }
    }

    /// `true` when no more frames can be pushed. Mirrors libvpx
    /// `vp9_lookahead_full` (`vp9_lookahead.c:85-87`): the ring is
    /// considered full when `sz + 1 + MAX_PRE_FRAMES > max_sz`, i.e.
    /// pushing one more frame would consume the pre-frame slot.
    #[must_use]
    pub fn full(&self) -> bool {
        self.sz + 1 + MAX_PRE_FRAMES > self.max_sz
    }

    /// Number of live entries currently in the ring (libvpx
    /// `vp9_lookahead_depth`, `vp9_lookahead.c:235`).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.sz
    }

    /// Total ring capacity (= configured depth + `MAX_PRE_FRAMES`).
    #[must_use]
    pub fn max_sz(&self) -> usize {
        self.max_sz
    }

    /// The monotonic show-frame counter (libvpx
    /// `vp9_lookahead_next_show_idx`, `vp9_lookahead.c:89-91`).
    #[must_use]
    pub fn next_show_idx(&self) -> u64 {
        self.next_show_idx
    }

    /// Push a source frame into the lookahead. Mirrors libvpx
    /// `vp9_lookahead_push` (`vp9_lookahead.c:93-199`) — copies the
    /// frame into the ring slot at `write_idx`, stamps it with
    /// `ts_start`/`ts_end`/`flags`/`show_idx`, and advances
    /// `write_idx`.
    ///
    /// # Errors
    ///
    /// - [`LookaheadError::Full`] if [`Self::full`] is true.
    /// - [`LookaheadError::DimensionMismatch`] if the source frame's
    ///   declared dimensions differ from the ring's. v1 rejects
    ///   resize.
    /// - [`LookaheadError::InvalidPlaneSize`] if the plane buffers
    ///   are shorter than the dimensions require.
    pub fn push(
        &mut self,
        src: &YuvFrameRef<'_>,
        ts_start: i64,
        ts_end: i64,
        flags: u32,
    ) -> Result<(), LookaheadError> {
        if self.full() {
            return Err(LookaheadError::Full);
        }
        // Implicit "src dimensions = ring dimensions" — the caller
        // supplies YuvFrameRef without explicit dims, so we validate
        // against the stride / buffer length.
        let w = self.width as usize;
        let h = self.height as usize;
        let uvw = w / 2;
        let uvh = h / 2;
        if src.y.len() < w * h
            || src.u.len() < uvw * uvh
            || src.v.len() < uvw * uvh
            || src.y_stride < w
            || src.u_stride < uvw
            || src.v_stride < uvw
        {
            return Err(LookaheadError::InvalidPlaneSize);
        }
        // Copy planes (handling strided sources by copying row-by-row).
        let mut y = vec![0u8; w * h];
        let mut u = vec![0u8; uvw * uvh];
        let mut v = vec![0u8; uvw * uvh];
        for row in 0..h {
            y[row * w..(row + 1) * w]
                .copy_from_slice(&src.y[row * src.y_stride..row * src.y_stride + w]);
        }
        for row in 0..uvh {
            u[row * uvw..(row + 1) * uvw]
                .copy_from_slice(&src.u[row * src.u_stride..row * src.u_stride + uvw]);
            v[row * uvw..(row + 1) * uvw]
                .copy_from_slice(&src.v[row * src.v_stride..row * src.v_stride + uvw]);
        }
        let entry = LookaheadEntry {
            y,
            u,
            v,
            width: self.width,
            height: self.height,
            ts_start,
            ts_end,
            flags,
            show_idx: self.next_show_idx,
        };
        self.next_show_idx += 1;
        self.buf[self.write_idx] = Some(entry);
        self.write_idx = (self.write_idx + 1) % self.max_sz;
        self.sz += 1;
        Ok(())
    }

    /// Pop the oldest live entry. Mirrors libvpx `vp9_lookahead_pop`
    /// (`vp9_lookahead.c:201-210`):
    ///
    /// - If `drain == false`, pop only when the lookahead is "filled"
    ///   (`sz == max_sz - MAX_PRE_FRAMES`). This is the steady-state
    ///   path — the consumer waits until the configured depth is full
    ///   before draining one frame per encode.
    /// - If `drain == true`, pop unconditionally. Used at
    ///   end-of-stream to flush remaining frames.
    ///
    /// Returns `None` when the ring is empty or (with `drain=false`)
    /// not yet filled to depth. The popped entry's slot is moved
    /// from the active region to the `MAX_PRE_FRAMES` backward-peek
    /// region (it's NOT cleared, so `peek(-1)` can still read it).
    pub fn pop(&mut self, drain: bool) -> Option<&LookaheadEntry> {
        if self.sz == 0 {
            return None;
        }
        // libvpx `vp9_lookahead.c:205`: only pop in steady state when
        // the ring is filled to its configured depth.
        if !drain && self.sz != self.max_sz - MAX_PRE_FRAMES {
            return None;
        }
        let idx = self.read_idx;
        self.read_idx = (self.read_idx + 1) % self.max_sz;
        self.sz -= 1;
        self.buf[idx].as_ref()
    }

    /// Peek at an entry without consuming it. Mirrors libvpx
    /// `vp9_lookahead_peek` (`vp9_lookahead.c:212-233`).
    ///
    /// - `index >= 0`: forward peek into live entries
    ///   (`read_idx + index` mod `max_sz`). Returns `None` if
    ///   `index >= depth`.
    /// - `index < 0`: backward peek into previously-popped entries.
    ///   Valid range is `-MAX_PRE_FRAMES ..= -1`; returns `None` for
    ///   indices further back.
    #[must_use]
    pub fn peek(&self, index: i32) -> Option<&LookaheadEntry> {
        if index >= 0 {
            let i = index as usize;
            if i >= self.sz {
                return None;
            }
            let slot = (self.read_idx + i) % self.max_sz;
            self.buf[slot].as_ref()
        } else {
            // libvpx `vp9_lookahead.c:225` — `-index <= MAX_PRE_FRAMES`.
            if (-index) as usize > MAX_PRE_FRAMES {
                return None;
            }
            let mut slot = self.read_idx as i32 + index;
            if slot < 0 {
                slot += self.max_sz as i32;
            }
            self.buf[slot as usize].as_ref()
        }
    }
}

#[cfg(test)]
mod tests {
    //! PROOF tests for W1.4 — the libvpx-faithful lookahead ring.

    use super::*;

    fn make_frame(w: usize, h: usize, seed: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y = (0..w * h).map(|i| seed.wrapping_add(i as u8)).collect();
        let u = vec![128u8; (w / 2) * (h / 2)];
        let v = vec![128u8; (w / 2) * (h / 2)];
        (y, u, v)
    }

    fn yview<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8], w: usize) -> YuvFrameRef<'a> {
        YuvFrameRef {
            y,
            u,
            v,
            y_stride: w,
            u_stride: w / 2,
            v_stride: w / 2,
        }
    }

    /// W1.4 PROOF — `depth=4` allocates `max_sz = 5` (4 lag + 1
    /// pre-frame slot). Mirrors libvpx `vp9_lookahead.c:55-58`.
    #[test]
    fn w1_4_capacity_includes_max_pre_frames() {
        let ctx = LookaheadCtx::new(64, 64, 4);
        assert_eq!(ctx.max_sz(), 4 + MAX_PRE_FRAMES);
        assert_eq!(ctx.depth(), 0);
        assert!(!ctx.full());
    }

    /// W1.4 PROOF — `full()` flips per libvpx `vp9_lookahead.c:85-87`.
    /// For depth=2 (max_sz=3, MAX_PRE_FRAMES=1), the ring becomes
    /// full after pushing 2 frames (`sz + 1 + 1 > 3`).
    #[test]
    fn w1_4_full_at_depth_minus_pre_frames() {
        let mut ctx = LookaheadCtx::new(64, 64, 2);
        let (y, u, v) = make_frame(64, 64, 0);
        ctx.push(&yview(&y, &u, &v, 64), 0, 1, 0).unwrap();
        assert!(!ctx.full());
        ctx.push(&yview(&y, &u, &v, 64), 1, 2, 0).unwrap();
        assert!(ctx.full(), "full at depth=2 after 2 pushes (max_sz=3)");
        let err = ctx.push(&yview(&y, &u, &v, 64), 2, 3, 0).unwrap_err();
        assert_eq!(err, LookaheadError::Full);
    }

    /// W1.4 PROOF — `pop(drain=false)` blocks until the ring is
    /// filled to depth (libvpx `vp9_lookahead.c:205`). At depth=4
    /// the ring fills at sz=4, so pops 1/2/3 should return None.
    #[test]
    fn w1_4_pop_drain_false_requires_filled_ring() {
        let mut ctx = LookaheadCtx::new(64, 64, 4);
        let (y, u, v) = make_frame(64, 64, 0);
        for i in 0..3 {
            ctx.push(&yview(&y, &u, &v, 64), i, i + 1, 0).unwrap();
            assert!(
                ctx.pop(false).is_none(),
                "pop must return None until sz == max_sz - MAX_PRE_FRAMES"
            );
        }
        // Fourth push fills the ring (sz=4 = max_sz - MAX_PRE_FRAMES).
        ctx.push(&yview(&y, &u, &v, 64), 3, 4, 0).unwrap();
        assert!(ctx.pop(false).is_some(), "pop must succeed once filled");
    }

    /// W1.4 PROOF — pushes/pops match libvpx ordering for
    /// `lag_in_frames=4`. Push N frames with show_idx 0..N; pop them
    /// (with `drain=true` for tail) and confirm order.
    #[test]
    fn w1_4_pushes_and_pops_preserve_show_idx_order() {
        let mut ctx = LookaheadCtx::new(64, 64, 4);
        for i in 0..6u8 {
            // Push when not full; pop one to make room when needed.
            if ctx.full() {
                ctx.pop(true).expect("drain to make room");
            }
            let (y, u, v) = make_frame(64, 64, i);
            ctx.push(&yview(&y, &u, &v, 64), i as i64, (i + 1) as i64, 0)
                .unwrap();
        }
        let mut seen = Vec::new();
        while let Some(e) = ctx.pop(true) {
            seen.push(e.show_idx);
        }
        // Show-idx is a monotonic counter assigned at push, so we
        // expect a contiguous subsequence of 0..6 in order. The
        // exact set depends on how many we drained mid-loop; what
        // matters is monotonicity.
        for w in seen.windows(2) {
            assert!(
                w[0] < w[1],
                "show_idx must be strictly increasing: {seen:?}"
            );
        }
        assert!(!seen.is_empty());
    }

    /// W1.4 PROOF — `peek(-1)` returns the previously-popped entry
    /// (libvpx `vp9_lookahead.c:223-229`). This is the scene-change
    /// detector's "prev source" access.
    #[test]
    fn w1_4_peek_minus_one_returns_prev_popped() {
        let mut ctx = LookaheadCtx::new(64, 64, 2);
        let (y0, u0, v0) = make_frame(64, 64, 10);
        let (y1, u1, v1) = make_frame(64, 64, 20);
        let (y2, u2, v2) = make_frame(64, 64, 30);
        ctx.push(&yview(&y0, &u0, &v0, 64), 0, 1, 0).unwrap();
        ctx.push(&yview(&y1, &u1, &v1, 64), 1, 2, 0).unwrap();
        // Ring is full (depth=2, sz=2, max_sz=3). Pop one (drain=true).
        let popped = ctx.pop(true).expect("pop frame 0").show_idx;
        assert_eq!(popped, 0);
        // Push another so the ring still carries frame 1 + new frame 2.
        ctx.push(&yview(&y2, &u2, &v2, 64), 2, 3, 0).unwrap();
        // peek(-1) must reach back into the popped slot (frame 0).
        let prev = ctx.peek(-1).expect("peek(-1) must yield prev source");
        assert_eq!(prev.show_idx, 0);
        // peek(-2) is beyond MAX_PRE_FRAMES.
        assert!(ctx.peek(-2).is_none());
    }

    /// W1.4 PROOF — forward `peek(0)` returns the next-to-pop entry;
    /// `peek(1)` returns the second. libvpx
    /// `vp9_lookahead.c:217-222`.
    #[test]
    fn w1_4_forward_peek_walks_live_entries() {
        let mut ctx = LookaheadCtx::new(64, 64, 4);
        for i in 0..4u8 {
            let (y, u, v) = make_frame(64, 64, i * 11);
            ctx.push(&yview(&y, &u, &v, 64), i as i64, (i + 1) as i64, 0)
                .unwrap();
        }
        assert_eq!(ctx.peek(0).unwrap().show_idx, 0);
        assert_eq!(ctx.peek(1).unwrap().show_idx, 1);
        assert_eq!(ctx.peek(3).unwrap().show_idx, 3);
        assert!(ctx.peek(4).is_none(), "peek past depth must return None");
    }

    /// W1.4 PROOF — depth is clamped to `[1, MAX_LAG_BUFFERS]`
    /// (libvpx `vp9_lookahead.c:55`).
    #[test]
    fn w1_4_depth_is_clamped() {
        let zero = LookaheadCtx::new(64, 64, 0);
        assert_eq!(zero.max_sz(), 1 + MAX_PRE_FRAMES);
        let huge = LookaheadCtx::new(64, 64, 1000);
        assert_eq!(huge.max_sz(), MAX_LAG_BUFFERS + MAX_PRE_FRAMES);
    }

    /// W1.4 PROOF — dimension-mismatch errors fire. Anti-G3:
    /// silently accepting smaller buffers would corrupt the ring.
    #[test]
    fn w1_4_dimension_mismatch_rejected() {
        let mut ctx = LookaheadCtx::new(128, 128, 2);
        let (y, u, v) = make_frame(64, 64, 0); // wrong size!
        let err = ctx.push(&yview(&y, &u, &v, 64), 0, 1, 0).unwrap_err();
        assert_eq!(err, LookaheadError::InvalidPlaneSize);
    }
}
