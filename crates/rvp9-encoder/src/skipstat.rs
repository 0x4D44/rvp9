//! E4 — env-gated per-leaf skip / residual-energy accounting
//! (diagnostic only).
//!
//! Companion to `rvp9_core::entropy::bitacct`: that tracks bits by
//! syntax category; this tracks the encoder's per-inter-leaf skip
//! decision and the residual energy of non-skip leaves, so the E4
//! harness can answer "what fraction of inter blocks does rvp9 skip,
//! and how much residual do the non-skip ones actually carry?".
//!
//! Changes NO coding logic — a single env-gated `record_leaf` call at
//! the encoder's per-leaf skip-finalisation site feeds these counters.
//! No-op (one relaxed atomic load) when `RVP9_E4_SKIPSTAT` is unset.

use core::sync::atomic::{AtomicBool, Ordering};
use std::cell::RefCell;

static SKIPSTAT_ON: AtomicBool = AtomicBool::new(false);

/// Per-frame aggregate skip / residual statistics.
#[derive(Copy, Clone, Debug, Default)]
pub struct SkipStats {
    /// Number of coded inter leaves (any block size) this frame.
    pub leaves: u64,
    /// Number coded skip=1.
    pub skip_leaves: u64,
    /// Number coded skip=0 (carry at least one nonzero coeff).
    pub nonskip_leaves: u64,
    /// Sum of total-eob (nonzero-token count) over non-skip leaves.
    pub nonskip_eob_sum: u64,
    /// Number of non-skip leaves whose total eob is "tiny" (<= 4
    /// nonzero coefficients across the whole leaf — the classic
    /// "should-have-been-skipped" signature).
    pub nonskip_tiny: u64,
}

thread_local! {
    static STATS: RefCell<SkipStats> = const { RefCell::new(SkipStats {
        leaves: 0, skip_leaves: 0, nonskip_leaves: 0, nonskip_eob_sum: 0, nonskip_tiny: 0,
    }) };
}

/// Enable skip accounting if `RVP9_E4_SKIPSTAT` is set and non-zero.
pub fn init_from_env() {
    let on = std::env::var_os("RVP9_E4_SKIPSTAT").is_some_and(|v| v != "0");
    SKIPSTAT_ON.store(on, Ordering::Relaxed);
}

/// Whether skip accounting is active.
#[inline]
pub fn enabled() -> bool {
    SKIPSTAT_ON.load(Ordering::Relaxed)
}

/// Record one inter leaf's final skip decision and residual eob total.
#[inline]
pub fn record_leaf(skip: bool, leaf_total_eob: usize) {
    if !enabled() {
        return;
    }
    STATS.with(|s| {
        let mut s = s.borrow_mut();
        s.leaves += 1;
        if skip {
            s.skip_leaves += 1;
        } else {
            s.nonskip_leaves += 1;
            s.nonskip_eob_sum += leaf_total_eob as u64;
            if leaf_total_eob <= 4 {
                s.nonskip_tiny += 1;
            }
        }
    });
}

/// Take and reset the per-frame stats for the current thread.
pub fn take_snapshot() -> SkipStats {
    STATS.with(|s| {
        let mut s = s.borrow_mut();
        let snap = *s;
        *s = SkipStats::default();
        snap
    })
}
