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

#[cfg(test)]
mod tests {
    use super::*;

    /// Single-test sweep that covers every branch in this module
    /// in deterministic order. The accumulator (`STATS`) is
    /// thread-local; the on/off atomic is global, so we save and
    /// restore it around the test to be friendly to concurrent
    /// runners.
    #[test]
    fn full_coverage_sweep() {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                SKIPSTAT_ON.store(self.0, Ordering::Relaxed);
            }
        }
        let _restore = Restore(SKIPSTAT_ON.load(Ordering::Relaxed));

        // Make sure our thread starts fresh.
        let _ = take_snapshot();

        // ---------------------------------------------------------
        // Branch: enabled() == false → record_leaf is a no-op.
        // ---------------------------------------------------------
        SKIPSTAT_ON.store(false, Ordering::Relaxed);
        assert!(!enabled());
        record_leaf(true, 100);
        record_leaf(false, 7);
        let snap_disabled = take_snapshot();
        assert_eq!(snap_disabled.leaves, 0, "no-op when disabled");
        assert_eq!(snap_disabled.skip_leaves, 0);
        assert_eq!(snap_disabled.nonskip_leaves, 0);
        assert_eq!(snap_disabled.nonskip_eob_sum, 0);
        assert_eq!(snap_disabled.nonskip_tiny, 0);

        // ---------------------------------------------------------
        // Branch: skip == true. Only `skip_leaves` and `leaves`
        // bump; `nonskip_*` fields untouched regardless of eob.
        // (eob is ignored in the skip arm.)
        // ---------------------------------------------------------
        SKIPSTAT_ON.store(true, Ordering::Relaxed);
        record_leaf(true, 999);
        let snap = take_snapshot();
        assert_eq!(snap.leaves, 1);
        assert_eq!(snap.skip_leaves, 1);
        assert_eq!(snap.nonskip_leaves, 0);
        assert_eq!(snap.nonskip_eob_sum, 0);
        assert_eq!(snap.nonskip_tiny, 0);

        // ---------------------------------------------------------
        // Branch: skip == false && leaf_total_eob <= 4.
        // All three nonskip_* fields bump, including `tiny`.
        // ---------------------------------------------------------
        record_leaf(false, 2);
        let snap = take_snapshot();
        assert_eq!(snap.leaves, 1);
        assert_eq!(snap.skip_leaves, 0);
        assert_eq!(snap.nonskip_leaves, 1);
        assert_eq!(snap.nonskip_eob_sum, 2);
        assert_eq!(snap.nonskip_tiny, 1);

        // Boundary: eob == 4 still counts as tiny (<=).
        record_leaf(false, 4);
        let snap = take_snapshot();
        assert_eq!(snap.nonskip_leaves, 1);
        assert_eq!(snap.nonskip_eob_sum, 4);
        assert_eq!(snap.nonskip_tiny, 1, "eob == 4 should fall in tiny arm");

        // ---------------------------------------------------------
        // Branch: skip == false && leaf_total_eob > 4.
        // nonskip_leaves and nonskip_eob_sum bump; nonskip_tiny does NOT.
        // ---------------------------------------------------------
        record_leaf(false, 100);
        let snap = take_snapshot();
        assert_eq!(snap.leaves, 1);
        assert_eq!(snap.nonskip_leaves, 1);
        assert_eq!(snap.nonskip_eob_sum, 100);
        assert_eq!(
            snap.nonskip_tiny, 0,
            "eob > 4 must NOT bump the tiny counter",
        );

        // Boundary: eob == 5 is the first "not tiny" value.
        record_leaf(false, 5);
        let snap = take_snapshot();
        assert_eq!(snap.nonskip_eob_sum, 5);
        assert_eq!(snap.nonskip_tiny, 0, "eob == 5 must NOT be tiny");

        // ---------------------------------------------------------
        // Multiple records accumulate; take_snapshot resets to zero.
        // ---------------------------------------------------------
        record_leaf(true, 0);
        record_leaf(true, 0);
        record_leaf(false, 3);
        record_leaf(false, 50);
        let snap = take_snapshot();
        assert_eq!(snap.leaves, 4);
        assert_eq!(snap.skip_leaves, 2);
        assert_eq!(snap.nonskip_leaves, 2);
        assert_eq!(snap.nonskip_eob_sum, 53);
        assert_eq!(snap.nonskip_tiny, 1, "only the eob=3 leaf is tiny");

        // After taking the snapshot, accumulator is back to zero.
        let snap = take_snapshot();
        assert_eq!(snap.leaves, 0);
        assert_eq!(snap.skip_leaves, 0);
        assert_eq!(snap.nonskip_leaves, 0);
        assert_eq!(snap.nonskip_eob_sum, 0);
        assert_eq!(snap.nonskip_tiny, 0);
    }
}
