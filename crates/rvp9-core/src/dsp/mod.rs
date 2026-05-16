//! Per-arch SIMD dispatch and DSP kernel families.
//!
//! This module is the RTCD-equivalent for rvp9. It owns:
//!
//! - [`caps`] — runtime CPU capability detection (`X86Caps`, `AArch64Caps`).
//! - [`dispatch`] — the `SimdDispatch` table struct + `OnceLock`-based
//!   `init()` / `get()` API enforcing the **R-09 init invariant** (the
//!   table is populated exactly once at `Encoder::new` / `Decoder::new`
//!   before any worker thread is spawned).
//!
//! Per `notes/hld/02-simd-dispatch.md` §2.2: workers always read through
//! a fully-populated `&'static SimdDispatch`. This is statically enforced
//! by `WorkerPool::new(&'static SimdDispatch)` (see `notes/lld/tile-threading.md`
//! §6.5).
//!
//! Per HLD-simd §1.4 (Q2 RESOLVED): the dispatch tables are hand-written
//! Rust, not generated from `vpx_dsp_rtcd_defs.pl`. Kernel families live
//! under `rvp9_core::dsp::*` and each declares both a scalar reference
//! and zero-or-more per-ISA implementations.
//!
//! ## Adding a new kernel family
//!
//! 1. Add a sub-module under `src/dsp/` with the scalar implementation.
//! 2. Add an `Option<fn>` slot to `SimdDispatch` per kernel size.
//! 3. Populate the slot in `SimdDispatch::build`.
//! 4. Add a differential test in the same file: random inputs through
//!    scalar + every-ISA variant, byte-exact equality.
//!
//! ## Status
//!
//! Stage 3 Task 0: scaffolding only — `SimdDispatch` is declared with the
//! per-family slots empty (`Option<fn>` currently `None` everywhere). Real
//! kernel bodies land per Stage 3 tasks 1-7 (one task per shared-infra
//! module: bitstream, entropy, transforms, quant, intra-pred, loopfilter,
//! frame-buffers).

pub mod caps;
pub mod dispatch;

pub use dispatch::{SimdDispatch, get, init};
