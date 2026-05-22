//! E4 — env-gated per-category bit accounting (diagnostic only).
//!
//! This module is a **measurement scaffold**, not part of the codec. It
//! lets the E4 overshoot-accounting harness tap the exact entropy cost
//! of every binary symbol the [`super::BoolWriter`] emits, bucketed by
//! syntax category. It changes NO coding logic: the only hook is a
//! single `if BITACCT_ON { record(...) }` at the top of `BoolWriter::
//! write`, which is a no-op (one relaxed atomic load) when the
//! `RVP9_E4_BITACCT` env var is unset.
//!
//! ## Cost model
//!
//! The VP9 arithmetic coder spends `-log2(p_used)` bits on a binary
//! symbol coded with probability-of-zero `prob/256`, where `p_used =
//! prob/256` for bit==0 and `(256-prob)/256` for bit==1. This is the
//! exact information content libvpx's own `vp9_cost_*` model uses (its
//! `vp9_prob_cost[]` table is a 256-entry quantisation of the same
//! `-log2`). Summed over a frame this matches the real coded bit count
//! to within the arithmetic-coder rounding (< 0.5 %).
//!
//! ## Usage
//!
//! 1. Set `RVP9_E4_BITACCT=1` before encoding (the harness does this).
//! 2. Before each leaf-category write burst, call [`set_category`].
//! 3. After a frame, read [`take_snapshot`] for per-category bits and
//!    reset for the next frame.

use core::sync::atomic::{AtomicBool, Ordering};
use std::cell::RefCell;

/// Syntax categories tracked by the E4 bit accounting harness. The
/// ordering matches the report table.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BitCategory {
    /// Partition-tree split bits.
    Partition = 0,
    /// Skip flag.
    Skip = 1,
    /// is-inter classification flag.
    IsInter = 2,
    /// tx_size signalling.
    TxSize = 3,
    /// Reference-frame + inter-mode + interp-filter syntax.
    ModeRef = 4,
    /// Motion-vector joint + component deltas.
    MvDelta = 5,
    /// Residual coefficient tokens.
    Coef = 6,
    /// Segment id (AQ).
    SegId = 7,
    /// Anything else (header, uncategorised).
    Other = 8,
}

/// Number of distinct [`BitCategory`] buckets.
pub const N_CATEGORIES: usize = 9;

/// Global on/off switch, set once from the `RVP9_E4_BITACCT` env var.
static BITACCT_ON: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// Per-category accumulated bits (fractional) for the current frame.
    static BITS: RefCell<[f64; N_CATEGORIES]> = const { RefCell::new([0.0; N_CATEGORIES]) };
    /// The category writes are currently attributed to.
    static CURRENT: RefCell<BitCategory> = const { RefCell::new(BitCategory::Other) };
}

/// Enable bit accounting if `RVP9_E4_BITACCT` is set and non-zero. Safe
/// to call repeatedly; idempotent.
pub fn init_from_env() {
    let on = std::env::var_os("RVP9_E4_BITACCT").is_some_and(|v| v != "0");
    BITACCT_ON.store(on, Ordering::Relaxed);
}

/// Whether accounting is active. One relaxed atomic load — cheap enough
/// to gate the `BoolWriter::write` hot path.
#[inline]
pub fn enabled() -> bool {
    BITACCT_ON.load(Ordering::Relaxed)
}

/// Set the category subsequent [`record`] calls attribute bits to.
#[inline]
pub fn set_category(cat: BitCategory) {
    if enabled() {
        CURRENT.with(|c| *c.borrow_mut() = cat);
    }
}

/// Record one binary symbol's entropy cost into the current category.
/// `bit` is the coded value (0/1); `prob` is the probability-of-zero in
/// 1/256 units (the value passed to `BoolWriter::write`).
#[inline]
pub fn record(bit: u8, prob: u8) {
    // p_used in (0, 1): probability the coder assigned to the symbol
    // actually emitted. prob is the prob of `0`.
    let p0 = f64::from(prob) / 256.0;
    let p_used = if bit == 0 { p0 } else { 1.0 - p0 };
    // Clamp away from 0 to avoid -inf on a degenerate prob (prob==0 or
    // 256 never happens via u8, but bit==1 with prob==255 gives p_used
    // = 1/256, which is fine).
    let p_used = p_used.max(1.0 / 512.0);
    let bits = -p_used.log2();
    let cat = CURRENT.with(|c| *c.borrow());
    BITS.with(|b| b.borrow_mut()[cat as usize] += bits);
}

/// Take and reset the per-category bit totals for the current thread.
/// Returns bits (fractional) indexed by [`BitCategory`].
pub fn take_snapshot() -> [f64; N_CATEGORIES] {
    BITS.with(|b| {
        let mut out = b.borrow_mut();
        let snap = *out;
        *out = [0.0; N_CATEGORIES];
        snap
    })
}
