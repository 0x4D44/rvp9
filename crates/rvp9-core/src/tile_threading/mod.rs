//! Tile-threading primitive — worker pool + scoped job submission.
//!
//! Generic over the per-tile work. Decoder and encoder both wrap this
//! primitive — neither implementation depth lives here. The decoder's
//! wrap is `crates/rvp9-decoder/src/lib.rs::Decoder::decode_inter_frame`;
//! the encoder's wrap is forthcoming (see
//! `notes/lld/tile-threading.md`).
//!
//! ## Overview
//!
//! - [`WorkerPool`] — a long-lived pool of OS threads owned by a
//!   [`Decoder`] (or future `Encoder`). One construction per codec
//!   instance, joined on drop.
//! - [`FrameScope`] — a per-frame submission handle returned via
//!   [`WorkerPool::scope`]. The closure-form API (HLD V6 §G5
//!   K-3) makes the scope unforgettable: `mem::forget(scope)` is
//!   impossible because the caller never sees an owning handle, only
//!   a borrowed reference to the scope from inside the closure.
//!
//! ## Why closure-form, not Drop-based
//!
//! V3 of the HLD proposed a `FrameScope` value type with a Drop impl
//! that joined all pending jobs. V6 K-3 (`HLD §G5`, lines 837-895)
//! pointed out the soundness gap: `mem::forget(scope)` would skip the
//! Drop, leaving worker threads with closures whose lifetime-erased
//! captures reference stack data from a long-returned function. The
//! HRTB `for<'frame> FnOnce(&FrameScope<'pool, 'frame>) -> R` makes
//! `'frame` fresh per call, the caller never owns the scope, so
//! `mem::forget` is structurally impossible.
//!
//! ## SAFETY contract
//!
//! The `unsafe { mem::transmute }` inside [`FrameScope::submit`] erases
//! the closure's `'frame` lifetime to `'static` so the job can travel
//! across the `crossbeam_channel`. The transmute is sound because
//! [`WorkerPool::scope`] blocks on every submitted job before it
//! returns — see [`FrameScope::submit`]'s SAFETY comment and the
//! module-adjacent `SAFETY.md` for the full chain.
//!
//! [`Decoder`]: rvp9_decoder::Decoder

mod lf_pool;
mod pool;
mod scope;

pub use lf_pool::{LfBatchHandle, LfWorkerPool};
pub use pool::WorkerPool;
pub use scope::FrameScope;
