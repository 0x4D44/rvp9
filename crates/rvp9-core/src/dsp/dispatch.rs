//! `SimdDispatch` — the per-arch kernel pointer table.
//!
//! Per HLD-simd §2.2 ("R-09 init invariant"): there is exactly one
//! `SimdDispatch` per running process, lazily constructed at the first
//! call to [`init`]/[`get`] and held in a `&'static OnceLock`. Workers
//! never observe a partially-initialised table: `OnceLock` linearises
//! concurrent constructors, and once a worker reads any kernel pointer
//! through the returned `&'static`, an Acquire memory barrier guarantees
//! it has seen every other slot too.
//!
//! ## R-09 enforcement
//!
//! `WorkerPool::new(&'static SimdDispatch)` (see `notes/lld/tile-threading.md`
//! §6.5) takes the dispatch reference as a constructor argument; you cannot
//! spawn a worker without first having called `init()` and obtained the
//! reference. This statically enforces the "RTCD initialised before worker
//! spawn" invariant.
//!
//! ## Scaffolding status
//!
//! Stage 3 Task 0 lands the **shape** of `SimdDispatch` and the init
//! machinery. The struct currently has no kernel pointer slots — they are
//! added one family at a time as Stage 3 tasks 1-7 land each LLD's content.
//! The struct stays `non_exhaustive` so adding a slot is non-breaking.

use std::sync::OnceLock;

use super::caps::{Caps, detect};

/// The runtime SIMD kernel dispatch table.
///
/// Populated once per process at [`init`]; immutable after that. Currently
/// empty — kernel pointer slots will land per Stage 3 task as the kernel
/// families come online:
///
/// | Task | LLD                    | Families added |
/// |------|------------------------|----------------|
/// | 1    | `bitstream.md`         | (none — pure scalar) |
/// | 2    | `entropy.md`           | (none — pure scalar) |
/// | 3    | `transforms.md`        | forward DCT/ADST + inverse DCT/ADST + Hadamard/SATD |
/// | 4    | `quant.md`             | quantise / quantise-32×32 |
/// | 5    | `intra-pred.md`        | 10 modes × 4 sizes + corner-variants |
/// | 6    | `loopfilter.md`        | vert/horiz × {16,16-dual,8,8-dual,4,4-dual} |
/// | 7    | `frame-buffers.md`     | scale-and-extend + frame-border helpers |
///
/// Encoder-only families (SAD/variance/sub-pel variance/etc.) land in
/// Stage 4a per `motion-estimation.md` and `mode-decision.md`.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SimdDispatch {
    /// The capability set the table was built against. Persisted for
    /// debug/instrumentation; the dispatch slots are already specialised
    /// to this caps state.
    pub caps: Caps,
}

impl SimdDispatch {
    /// Build a dispatch table for the given capability set. Used by
    /// [`init`] and by tests that want to drive a forced-caps scenario
    /// (e.g. `caps.avx2 = false` to exercise the SSE4.1 path on an AVX2
    /// CPU).
    ///
    /// Currently `caps` is the only field; this will fan out to one
    /// per-kernel selection clause as kernel families are added.
    #[must_use]
    pub fn build(caps: Caps) -> Self {
        Self { caps }
    }
}

/// The single process-wide dispatch table.
///
/// Initialised by [`init`] or implicitly by [`get`]. The `'static`
/// reference returned is the same for every caller.
static DISPATCH: OnceLock<SimdDispatch> = OnceLock::new();

/// Initialise (or retrieve) the process-wide dispatch table.
///
/// Idempotent: concurrent first-calls linearise via `OnceLock`. CPU caps
/// don't change at runtime, so the constructed table is deterministic.
///
/// **`Encoder::new` and `Decoder::new` MUST call this as their first
/// action** (HLD-simd §2.2 / R-09). The `&'static SimdDispatch` returned
/// is what gets passed to `WorkerPool::new` and is the only way to obtain
/// a worker pool — statically enforcing the init-before-spawn invariant.
#[inline]
pub fn init() -> &'static SimdDispatch {
    DISPATCH.get_or_init(|| SimdDispatch::build(detect()))
}

/// Get the process-wide dispatch table. Same as [`init`]; named
/// separately as documentation hook for "read-after-init" call sites
/// (e.g. inside an encoder method that knows `init` has already run).
#[inline]
pub fn get() -> &'static SimdDispatch {
    init()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `init()` returns the same `&'static` on every call.
    #[test]
    fn init_is_idempotent() {
        let a = init() as *const SimdDispatch;
        let b = init() as *const SimdDispatch;
        assert_eq!(a, b);
    }

    /// `get()` is an alias for `init()`.
    #[test]
    fn get_matches_init() {
        let a = init() as *const SimdDispatch;
        let b = get() as *const SimdDispatch;
        assert_eq!(a, b);
    }

    /// The dispatch table is `Send + Sync`: `&'static SimdDispatch` can
    /// safely move between threads. This is what the R-09 invariant
    /// relies on for worker pools.
    #[test]
    fn dispatch_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<&'static SimdDispatch>();
    }

    /// Building with explicit caps lets tests force a SIMD tier without
    /// changing the running CPU.
    #[test]
    fn explicit_build() {
        let caps = detect();
        let d = SimdDispatch::build(caps);
        assert_eq!(d.caps, caps);
    }
}
