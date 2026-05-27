//! E9 — env-gated per-frame prediction-error accounting (diagnostic only).
//!
//! Companion to `skipstat`: that tracks the per-leaf skip decision; this
//! taps the Y-plane prediction residual (`source − prediction`, BEFORE
//! the forward-transform / quant / inverse-add) so the E9 quality-gap
//! harness can answer "how much of the ~9 dB matched-Q gap lives in
//! prediction quality vs transform/quant vs entropy?".
//!
//! The encoder builds prediction into the recon plane, then computes
//! `residual = input − pred` for each transform block. At that instant
//! `pred` is the pure prediction (intra or motion-compensated inter),
//! so summing `residual²` over the Y plane gives the per-frame
//! prediction SSE. The decoder-side reconstruction SSE is measured by
//! the harness from the decoded output, so the pair `(pred_sse,
//! recon_sse)` cleanly separates "how good is the prediction" from "how
//! much did residual coding recover".
//!
//! Changes NO coding logic — a single env-gated `record_y_residual`
//! call at each (intra + inter) residual-formation site feeds the
//! accumulator. No-op (one relaxed atomic load) when `RVP9_E9_PREDSTAT`
//! is unset.

use core::sync::atomic::{AtomicBool, Ordering};
use std::cell::RefCell;

static PREDSTAT_ON: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// Whole-frame Y prediction snapshot (tight-stride `frame_w × frame_h`).
    /// Filled by [`record_y_pred_block`] when prediction-capture is active;
    /// taken by [`take_pred_frame`]. Empty until the first frame populates
    /// it. This is the `diag_pred` backing store — the per-pixel companion
    /// to the per-block [`PredStats::y_pred_sse`] accumulator.
    static PRED_FRAME: RefCell<PredFrame> = const {
        RefCell::new(PredFrame { y: Vec::new(), w: 0, h: 0 })
    };
}

/// Whole-frame Y prediction snapshot. `y` is tight-stride `w × h`. Used
/// by the `stage_sse_diag` oracle to compute per-frame pred-SSE against
/// the source over the FULL frame (every pixel, including skip leaves) —
/// the per-pixel companion to the per-TX [`PredStats`] accumulator.
#[derive(Clone, Debug, Default)]
pub struct PredFrame {
    /// Tight-stride Y prediction pixels, `w * h` long (empty if no frame
    /// captured).
    pub y: Vec<u8>,
    /// Frame width in pixels.
    pub w: usize,
    /// Frame height in pixels.
    pub h: usize,
}

/// Record one Y-plane transform block's PREDICTION pixels into the
/// whole-frame snapshot. `pred` is the prediction plane (tight-stride
/// `frame_w`), `(px_x, px_y)` the block's top-left, `tx_side` the block
/// edge. `frame_w`/`frame_h` size the snapshot on first call per frame.
/// No-op unless enabled. Companion to [`record_y_residual`] — same call
/// sites, captures the pixels the residual is formed against.
#[inline]
pub fn record_y_pred_block(
    pred: &[u8],
    pred_stride: usize,
    px_x: usize,
    px_y: usize,
    tx_side: usize,
    frame_w: usize,
    frame_h: usize,
) {
    if !enabled() {
        return;
    }
    PRED_FRAME.with(|pf| {
        let mut pf = pf.borrow_mut();
        if pf.w != frame_w || pf.h != frame_h || pf.y.len() != frame_w * frame_h {
            pf.w = frame_w;
            pf.h = frame_h;
            pf.y = vec![0u8; frame_w * frame_h];
        }
        for r in 0..tx_side {
            let src_off = (px_y + r) * pred_stride + px_x;
            let dst_off = (px_y + r) * frame_w + px_x;
            pf.y[dst_off..dst_off + tx_side]
                .copy_from_slice(&pred[src_off..src_off + tx_side]);
        }
    });
}

/// Take and reset the whole-frame Y prediction snapshot for the current
/// thread. Returns an empty [`PredFrame`] if none was captured.
pub fn take_pred_frame() -> PredFrame {
    PRED_FRAME.with(|pf| {
        let mut pf = pf.borrow_mut();
        std::mem::take(&mut *pf)
    })
}

/// Per-frame prediction-error aggregate (Y plane only).
#[derive(Copy, Clone, Debug, Default)]
pub struct PredStats {
    /// Sum of squared (source − prediction) over every Y transform
    /// block coded this frame (non-skip residual-coded coverage).
    pub y_pred_sse: u64,
    /// Number of Y pixels covered by the above SSE (so the harness can
    /// report SSE-per-pixel and convert to an equivalent PSNR over the
    /// covered area).
    pub y_pixels: u64,
    /// Per-tx-size histogram of TX blocks coded this frame, indexed by
    /// `TxSize as usize` (0=4x4, 1=8x8, 2=16x16, 3=32x32). Y plane only.
    pub txsize_hist: [u64; 4],
}

thread_local! {
    static STATS: RefCell<PredStats> = const {
        RefCell::new(PredStats { y_pred_sse: 0, y_pixels: 0, txsize_hist: [0; 4] })
    };
}

/// Enable prediction-error accounting if `RVP9_E9_PREDSTAT` is set and
/// non-zero.
pub fn init_from_env() {
    let on = std::env::var_os("RVP9_E9_PREDSTAT").is_some_and(|v| v != "0");
    PREDSTAT_ON.store(on, Ordering::Relaxed);
}

/// Whether prediction-error accounting is active.
#[inline]
pub fn enabled() -> bool {
    PREDSTAT_ON.load(Ordering::Relaxed)
}

/// Record one Y-plane transform block's prediction residual. `residual`
/// is `source − prediction` row-major over `tx_side × tx_side`.
/// `txsize_idx` is `TxSize as usize`. No-op unless enabled.
#[inline]
pub fn record_y_residual(residual: &[i16], tx_side: usize, txsize_idx: usize) {
    if !enabled() {
        return;
    }
    let mut sse: u64 = 0;
    for &d in &residual[..tx_side * tx_side] {
        let dd = i64::from(d);
        sse += (dd * dd) as u64;
    }
    STATS.with(|s| {
        let mut s = s.borrow_mut();
        s.y_pred_sse += sse;
        s.y_pixels += (tx_side * tx_side) as u64;
        if txsize_idx < 4 {
            s.txsize_hist[txsize_idx] += 1;
        }
    });
}

/// Take and reset the per-frame stats for the current thread.
pub fn take_snapshot() -> PredStats {
    STATS.with(|s| {
        let mut s = s.borrow_mut();
        let snap = *s;
        *s = PredStats::default();
        snap
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single-test sweep that exercises every branch in this module
    /// in deterministic order. We share the process-wide
    /// `PREDSTAT_ON` atomic with other tests, but `STATS` is
    /// thread-local so the per-thread accumulator is private. We
    /// restore the atomic to its prior value on exit so the global
    /// state isn't perturbed for any concurrent test that might
    /// happen to be observing it.
    #[test]
    fn full_coverage_sweep() {
        // Save & restore the global atomic so we don't perturb
        // concurrent tests. Drop guard pattern in case of panic.
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                PREDSTAT_ON.store(self.0, Ordering::Relaxed);
            }
        }
        let _restore = Restore(PREDSTAT_ON.load(Ordering::Relaxed));

        // Ensure our thread starts with a clean accumulator
        // regardless of whatever earlier in-thread state existed.
        let _ = take_snapshot();

        // ---------------------------------------------------------
        // Branch 1: enabled() == false → record_y_residual is a no-op.
        // ---------------------------------------------------------
        PREDSTAT_ON.store(false, Ordering::Relaxed);
        assert!(!enabled(), "atomic should be off");

        // Use a non-trivial residual to make a missed early-return obvious.
        let residual_4x4: [i16; 16] = [
            10, -10, 10, -10, // row 0
            -5, 5, -5, 5, // row 1
            3, -3, 3, -3, // row 2
            -1, 1, -1, 1, // row 3
        ];
        // tx_side = 4, txsize_idx = 0 (4x4)
        record_y_residual(&residual_4x4, 4, 0);
        let snap_disabled = take_snapshot();
        assert_eq!(
            snap_disabled.y_pred_sse, 0,
            "record_y_residual must be a no-op when disabled",
        );
        assert_eq!(snap_disabled.y_pixels, 0);
        assert_eq!(snap_disabled.txsize_hist, [0; 4]);

        // ---------------------------------------------------------
        // Branch 2: enabled() == true with txsize_idx in-bounds
        // (txsize_idx < 4). Hand-compute the expected SSE.
        // ---------------------------------------------------------
        PREDSTAT_ON.store(true, Ordering::Relaxed);
        assert!(enabled());

        // Hand-computed SSE for `residual_4x4`:
        // 4 * (10^2) + 4 * (5^2) + 4 * (3^2) + 4 * (1^2)
        //   = 400 + 100 + 36 + 4 = 540
        let expected_sse: u64 = 540;

        // Record with txsize_idx = 2 (16x16). Note: tx_side and
        // txsize_idx are independent in this API — `tx_side` controls
        // how many residual entries we sum, while `txsize_idx`
        // selects the histogram bin. We deliberately use tx_side=4
        // (so the SSE math is hand-checked) with txsize_idx=2 (so
        // we hit the in-bounds bin in the middle).
        record_y_residual(&residual_4x4, 4, 2);
        let snap = take_snapshot();
        assert_eq!(snap.y_pred_sse, expected_sse);
        assert_eq!(snap.y_pixels, 16, "tx_side*tx_side = 4*4 = 16");
        assert_eq!(snap.txsize_hist, [0, 0, 1, 0]);

        // ---------------------------------------------------------
        // Branch 3: enabled() == true with txsize_idx >= 4
        // (out-of-bounds clamp). Histogram unchanged; sse/pixels
        // still accumulate.
        // ---------------------------------------------------------
        record_y_residual(&residual_4x4, 4, 99);
        let snap = take_snapshot();
        assert_eq!(snap.y_pred_sse, expected_sse);
        assert_eq!(snap.y_pixels, 16);
        assert_eq!(
            snap.txsize_hist,
            [0, 0, 0, 0],
            "out-of-bounds txsize_idx must NOT touch the histogram",
        );

        // ---------------------------------------------------------
        // Snapshot is reset by take_snapshot: re-take returns zeros.
        // ---------------------------------------------------------
        let snap = take_snapshot();
        assert_eq!(snap.y_pred_sse, 0);
        assert_eq!(snap.y_pixels, 0);
        assert_eq!(snap.txsize_hist, [0; 4]);
    }
}
