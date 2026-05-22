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
