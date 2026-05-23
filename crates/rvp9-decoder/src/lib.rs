//! rvp9-decoder: VP9 video decoder.
//!
//! Implements the VP9 Bitstream & Decoding Process Specification.
//! Validated end-to-end against the official VP9 conformance test
//! vectors.
//!
//! ## Stage 3 status
//!
//! Stage 3 Task 8.1 lands the [`Decoder`] skeleton — the top-level
//! state container that owns the [`DecoderFramePool`], the per-frame
//! probability banks ([`FrameContext`] × 4), and the running
//! loop-filter delta state.
//!
//! What works today:
//!
//! - [`Decoder::parse_packet`] splits a superframe (1..=8 frames per
//!   transport packet) into individual frame slices via
//!   [`parse_superframe_index`], then parses each frame's
//!   uncompressed header via [`parse_uncompressed_header`]. Returns
//!   the [`PacketFrames`] iterator of `(header, byte_range)` per
//!   frame.
//!
//! - [`Decoder::frame_pool`] / `frame_pool_mut` accessors for the
//!   ref-counted frame buffer pool.
//!
//! What does NOT work yet (deferred to later Stage 3 tasks):
//!
//! - Compressed-header parsing (Task 1e — boolean-coder driver
//!   reading tx_mode, coef-prob updates, frame-ref-mode, intra/inter
//!   probs, MV probs).
//! - Tile-level coefficient decoding (Tasks 8.2+).
//! - Pixel reconstruction (intra-pred + transform + add + loopfilter)
//!   per superblock (Task 8.3).
//! - Conformance gate against vpxdec (Task 9).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
// Wave A G11 — workspace lint `clippy::print_stderr` is "deny" for
// library crates. The decoder's compressed-header / segmentation
// debug dumps + loop-filter mask probe + altref dump path all emit
// to stderr through env-var-gated breadcrumbs (`RVP9_DUMP_ALTREF`,
// `[rvp9_seg]`, `RVP9_TRACE_FC`, etc.). Scheduled for triage in the
// HLD §G11 deliverable 3 eprintln audit follow-up (~80% delete or
// `cfg!(rvp9_trace)`-gate, ~10% promote to `DecodeError`, ~10%
// convert to `log::info!`). Until that sweep lands, allow at the
// crate level for the decoder; new eprintln sites added by future
// commits should also be triaged at that time.
#![allow(clippy::print_stderr)]

pub mod detokenize;
pub mod reconstruct;
pub mod reconstruct_inter;
pub mod tile;
pub mod tile_ctx;
// R.1 — process-wide cache of RVP9_TRACE_* / RVP9_DEBUG_* env vars.
// Pub(crate) only; not part of the decoder's public surface.
mod trace_flags;

use rvp9_core::bitstream::{
    BitstreamError, ChromaFormat, CompressedHeaderOptions, CompressedHeaderResult, FrameKind,
    InterpFilter, Superframe, UncompressedHeader, parse_compressed_header, parse_superframe_index,
    parse_tile_buffers, parse_uncompressed_header,
};
use rvp9_core::block::BlockSize;
use rvp9_core::entropy::{FRAME_CONTEXTS, FrameContext};
use rvp9_core::frame::extend_frame_borders;
use rvp9_core::frame::{
    BorderUsage, ByteAlignment, DecoderFramePool, FrameBuf, Yv12Buffer, Yv12ScratchPool,
};
use rvp9_core::inter::mode_info::{InterModeReadParams, setup_compound_reference_mode};
use rvp9_core::inter::mv_ref::sign_bias_from_header;
use rvp9_core::loopfilter::{
    LoopFilterInfo, LoopFilterMask, ResolvedLfDeltas, SegmentationLfView, adjust_mask,
    resolve_filter_levels, setup_mask,
};
use rvp9_core::loopfilter::apply::loop_filter_frame_with_non420;
use rvp9_core::mode_info::ModeInfo;

use crate::tile::{
    FrameDecodeParams, InterFrameDecodeParams, TileDecodeError, decode_tile_inter,
    decode_tile_intra, tile_bounds,
};
use crate::tile_ctx::TileDecodeCtx;
use std::sync::Mutex;

/// G5.1: luma pixels per MI block (`MI_BLOCK_SIZE_LOG2 = 3` → 8 luma
/// pixels). Used by the parallel tile-dispatch path to translate
/// `TileBounds::mi_col_*` into luma pixel column offsets for
/// [`Yv12Buffer::merge_tile_columns_from`].
const MI_BLOCK_PIXELS: usize = 8;

/// D.20 — write the visible-region Y plane of a freshly-decoded
/// frame to `path` (raw bytes, row-major, no padding). Used by the
/// `RVP9_DUMP_ALTREF=<path>` env hook to capture our altref output
/// for diffing against libvpx's altref dump.
fn dump_visible_y_plane(yv12: &Yv12Buffer, path: &str) {
    let y = yv12.y_plane();
    let w = y.width();
    let h = y.height();
    let stride = y.stride();
    let mut buf = Vec::with_capacity(w * h);
    for r in 0..h {
        let row_start = r * stride;
        buf.extend_from_slice(&y.data()[row_start..row_start + w]);
    }
    if let Err(e) = std::fs::write(path, &buf) {
        eprintln!("RVP9_DUMP_ALTREF: failed to write {path}: {e}");
    } else {
        eprintln!(
            "RVP9_DUMP_ALTREF: wrote {} bytes ({}x{}) to {}",
            buf.len(),
            w,
            h,
            path
        );
    }
}

/// D.22 — global decode-frame counter, parallels libvpx's
/// `rvp9_trace_current_frame` (`vp9_decodeframe.c:55`). Incremented at
/// the top of [`Decoder::decode_frame`]; read by per-block tracers in
/// `tile.rs` to tag `RVP9_TRACE_INTERMI` lines with the same `f=N`
/// libvpx emits, so altref + visible-frame traces line up frame-by-
/// frame between the two implementations.
pub static RVP9_TRACE_FRAME_COUNTER: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// G5.1 instrumentation: count of tile-decode jobs the parallel
/// dispatch path submitted into the worker pool, summed across the
/// process lifetime. Used by the
/// `multi_tile_dispatch_uses_worker_pool` regression test (in this
/// crate's integration test suite) to confirm the parallel path
/// fires non-tautologically on multi-tile vectors. Bumped exactly
/// once per `scope.submit(...)` call in
/// [`Decoder::decode_intra_frame_inner`] and
/// [`Decoder::decode_inter_frame`].
pub static RVP9_G5_1_PARALLEL_JOBS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Top-level VP9 decoder state.
///
/// Mirrors the role of libvpx's `VP9Decoder` + `VP9_COMMON` — owns
/// every per-stream and per-frame-history data structure the decoder
/// needs.
///
/// Construction does not require a stream header — call
/// [`Decoder::new`], then feed packets via [`Decoder::parse_packet`].
///
/// # Thread safety
///
/// `Decoder` is both [`Send`] and [`Sync`]. The `&mut self` decode
/// API means concurrent `&Decoder` holders cannot mutate without a
/// lock, so `Sync` adds no requirement beyond what auto-derive
/// produces. Wave C G5's tile worker pool captures borrowed slices
/// in per-job closures and does not share `&Decoder` across threads.
/// The compile-time `static_assertions::assert_impl_all!(Decoder:
/// Send, Sync)` immediately below the struct catches accidental
/// drift if a future change introduces interior mutability.
///
/// # Panic / error safety
///
/// After [`Decoder::decode_frame`] returns `Err(_)`, the `Decoder`
/// is in a **well-defined state**: subsequent calls may succeed for
/// unrelated inputs. The poisoning contract is intentionally
/// non-transactional:
///
/// - `ref_frame_map`, the frame pool, and the four `FrameContext`
///   banks are **NOT reverted** on error. A mid-tile failure may
///   leave partial entropy adaptation in the bank that was being
///   updated.
/// - A subsequent keyframe / intra-only frame fully refreshes the
///   bank it selects (and any banks `reset_frame_context` /
///   `error_resilient` ask for), so a keyframe after an error
///   always restarts cleanly.
/// - `Decoder` does not implement `Clone`; the in-progress decode
///   state is too large and partly opaque (the per-frame mv-grid,
///   the four `FrameContext` banks, the prev-segment buffers, the
///   ref-frame-size cache, etc.). Consumers needing transactional
///   semantics (revert on error) should keep a fresh `Decoder` for
///   each independent stream rather than attempting to snapshot —
///   the public accessors do NOT cover every field, so a
///   "field-by-field copy" is not actually achievable today.
///
/// The integration tests in `tests/panic_safety.rs` pin this
/// contract: an early-error packet (uncompressed-header parser
/// `Err`) followed by a valid keyframe decodes successfully, a
/// mid-tile failure (truncated entropy stream) followed by a valid
/// keyframe decodes successfully, and a mid-tile failure followed
/// by a valid INTER frame (whose refs survived the failed decode)
/// also decodes — pinning that the surviving `ref_frame_map`
/// entries remain resolvable through the documented bank-poisoning.
///
/// # Memory and allocation
///
/// Frame allocation goes through [`Yv12Buffer::alloc`], which uses
/// `vec![0u8; N].into_boxed_slice()` internally. Two layers of
/// defence apply:
///
/// - `MAX_FRAME_PIXELS` (1 GiB worth of luma at 8-bit) rejects
///   oversized headers before any allocation attempt — see
///   `rvp9_core::frame` for the bound. Headers exceeding the bound
///   surface as [`DecodeError::FrameAlloc`] (no allocation tried).
/// - Allocations within the bound that fail panic per the global
///   allocator's default. A future `try_alloc` variant returning
///   `Result<Self, DecodeError::OutOfMemory>` is tracked but not in
///   Stage 3.5.
///
/// # Drop semantics
///
/// `Decoder` owns the frame pool and the ref-frame-map slot
/// indices. Both indirect into `Arc<Yv12Buffer>` storage. Any
/// [`FrameBuf`] handle returned from [`Decoder::decode_frame`] is an
/// `Arc`; **its backing buffer survives the decoder's drop** as
/// long as the caller holds the handle. The common
/// "decode-into-`Vec`" pattern is therefore safe:
///
/// ```ignore
/// let mut frames: Vec<rvp9_core::frame::FrameBuf> = Vec::new();
/// {
///     let mut dec = rvp9_decoder::Decoder::new();
///     // ... parse packets, push FrameBufs into `frames` ...
///     // `dec` drops at end of scope. Its frame_pool drops too.
/// }
/// // `frames[0]` is still valid here — the Arc-backed buffer
/// // outlives the Decoder because the Vec holds the only
/// // remaining strong reference.
/// drop(frames);
/// ```
///
/// Explicit drop order matters only if the caller wants the backing
/// allocations released eagerly: dropping the `Vec<FrameBuf>` first
/// releases the buffers, dropping the `Decoder` first leaves the
/// buffers alive until the `Vec` drops. Either order is sound.
///
/// # Pool exhaustion
///
/// The decoder's internal frame pool holds at most `N` buffers
/// (currently `9` — see [`rvp9_core::frame::DECODER_FRAME_BUFFERS`]).
/// Consumers that retain `N` or more decoded frames simultaneously
/// will receive [`DecodeError::FrameAlloc`] on the next decode —
/// `acquire_free` reports `NoFreeSlot` because every pool slot is
/// either externally held (`Arc::strong_count > 1`) or referenced
/// by the active `ref_frame_map`. The natural pattern
///
/// ```ignore
/// let frames: Vec<FrameBuf> = (0..M).map(|_| decode_one()).collect();
/// ```
///
/// is safe only for `M < N`. For longer streams, drain returned
/// frames (drop or process them) before requesting more — once an
/// external `FrameBuf` drops AND its slot is no longer named by any
/// `ref_frame_map` entry, the slot becomes recyclable.
pub struct Decoder {
    /// Worker-pool size requested at construction time. Currently a
    /// forward-compatibility plumb only — Wave A G12 stores it,
    /// Wave C G5 will consume it to build the tile-threading
    /// `WorkerPool`. Read by future `decode_inter_frame` /
    /// `decode_intra_frame_inner` to size their per-frame `scope`.
    ///
    /// Initialised by [`Decoder::new`] from
    /// `std::thread::available_parallelism`, defaulting to 1 if
    /// detection fails. Override via [`Decoder::with_threads`] for
    /// deterministic single-threaded test runs (conformance harness
    /// uses `with_threads(NonZeroUsize::new(1).unwrap())` to keep
    /// per-CPU-count variance out of the K-1 inline-worker math).
    num_threads: std::num::NonZeroUsize,
    /// 9-slot pool of decoded reference frames. Single source of
    /// truth for `Arc<Yv12Buffer>` storage — every `FrameBuf` lives
    /// here exactly once. [`Self::ref_frame_map`] indirects into this
    /// pool via `u8` slot indices (libvpx-faithful: see the
    /// `cm->ref_frame_map[i] -> cm->buffer_pool->frame_bufs[N]`
    /// chain in `vp9_decodeframe.c:2940`).
    ///
    /// G12 refactor (2026-05-21): prior to G12, `ref_frame_map` held
    /// `Option<FrameBuf>` and a parallel `frame_pool.assign_slot`
    /// mirror was written on every refresh. That doubled
    /// `Arc::strong_count` per slot and would have starved
    /// `acquire_free` as soon as the function was wired up (it
    /// wasn't yet, so the bug was latent). G12 collapses the two
    /// data structures to a single Arc-per-buffer model.
    ///
    // INVARIANT (frame_pool ↔ ref_frame_map): every
    // `Some(pool_idx)` entry in `ref_frame_map` MUST be paired with
    // a non-zero `frame_pool.ref_count(pool_idx)` AND index a
    // `Some(_)` entry in `frame_pool.slots` (i.e. an occupied pool
    // slot). `acquire_free`'s safety relies on
    // `ref_counts == 0 && Arc::strong_count == 1`: the Arc check
    // alone is NOT sufficient (the v6.0 refactor briefly attempted
    // that and produced silent aliasing on recycle — see the
    // `pool_acquire_blocked_by_nonzero_ref_count` regression in
    // `rvp9-core::frame`).
    //
    // This invariant is maintained by:
    //   - `decode_intra_frame_inner` / `decode_inter_frame`: write
    //     the pool index returned by `frame_pool.acquire_free(buf)`
    //     into every ref-map slot selected by `refresh_frame_flags`
    //     in one logical step (no half-applied refresh), via
    //     [`Self::write_ref_slot`] which handles the inc/dec_ref
    //     bookkeeping.
    //   - `decode_show_existing_frame`: reads via
    //     `ref_frame_buf(i)` which fails closed (`None` →
    //     `EmptyInterRef` / `EmptySlot`) if the pool slot is
    //     unexpectedly free, so a violated invariant surfaces as a
    //     clean error instead of UB.
    //   - `DecoderFramePool::acquire_free`: only recycles slots
    //     with `ref_counts[slot] == 0 AND Arc::strong_count == 1`,
    //     so an active `ref_frame_map[i] -> pool_idx` keeps that
    //     slot reserved via `ref_counts`.
    // Tests that synthesize ref-map state MUST use
    // [`Self::set_ref_slot`] (which performs the inc_ref under the
    // hood) AND pre-populate the matching pool slot via
    // [`Self::frame_pool_mut`] + [`DecoderFramePool::assign_slot`],
    // or the invariant breaks and `ref_frame_buf` returns `None`.
    frame_pool: DecoderFramePool,
    /// 8-slot reference-frame map. `ref_frame_map[i]` is the
    /// frame-pool slot index currently bound to logical ref slot
    /// `i` (`0..DECODER_FRAME_BUFFERS = 0..9`), or `None` if that
    /// slot has never been written. Mirrors libvpx's
    /// `cm->ref_frame_map[i]` (which is also a `int` index into the
    /// shared `BufferPool::frame_bufs` array — see
    /// `vp9_decodeframe.c:2940`'s `cm->frame_refs[i].buf =
    /// &cm->buffer_pool->frame_bufs[cm->ref_frame_map[ref_idx]].buf`).
    ///
    /// Resolve to a [`FrameBuf`] via
    /// [`Self::ref_frame_buf`]. Updated post-decode in
    /// [`Decoder::decode_keyframe`] (every slot per `refresh_frame_flags
    /// = 0xFF`) and [`Decoder::decode_inter_frame`] (per the parsed
    /// `refresh_frame_flags` mask).
    ref_frame_map: [Option<u8>; rvp9_core::bitstream::REF_FRAMES],
    /// Cached (width, height) of every frame currently held in the
    /// 8-slot reference-frame-map. `None` means the corresponding
    /// ref-map slot has no decoded frame yet. Mirrors the
    /// `cm->ref_frame_map[i] -> frame_bufs[i].buf->y_crop_{width,height}`
    /// lookup chain libvpx uses inside `setup_frame_size_with_refs`
    /// (`vp9_decodeframe.c:1559-1574`). Updated by `decode_keyframe`
    /// (which sets every slot per `refresh_frame_flags = 0xFF`).
    /// Decoupled from the frame_pool slot pool: this is the minimum
    /// state needed by the uncompressed-header parser, and does NOT
    /// require any actual pixels to be stored.
    ref_frame_sizes: [Option<(u32, u32)>; rvp9_core::bitstream::REF_FRAMES],
    /// Four independent FRAME_CONTEXT banks, selected per-frame by
    /// the 2-bit `frame_context_idx` header field. Each is reset to
    /// libvpx defaults on construction and updated by backward
    /// adaptation after each frame's residuals are decoded.
    frame_contexts: [FrameContext; FRAME_CONTEXTS],
    /// Accumulated motion-vector decode counts for the in-flight
    /// inter frame. Zeroed at the start of each inter frame; consumed
    /// by [`adapt_mv_probs`](rvp9_core::entropy::adapt::adapt_mv_probs)
    /// at end-of-frame to produce the next frame's `nmvc` table.
    /// Mirrors libvpx `cm->counts.mv` (`vp9_entropymode.h:78`).
    mv_counts: rvp9_core::entropy::mv_probs::NmvCounts,
    /// Accumulated coefficient decode counts for the in-flight frame.
    /// Zeroed at the start of every frame (libvpx clears `cm->counts`
    /// for any frame with `frame_parallel_decoding_mode == 0` —
    /// `vp9_decodeframe.c:2784`). Consumed by
    /// [`adapt_coef_probs`](rvp9_core::entropy::adapt::adapt_coef_probs)
    /// at end-of-frame to produce the next frame's `coef_probs` table.
    /// Both keyframes and inter frames adapt coefficient probabilities
    /// (only the `(count_sat, max_update_factor)` pair differs — see
    /// [`coef_adapt_params`](rvp9_core::entropy::adapt::coef_adapt_params)).
    /// Mirrors libvpx `cm->counts.coef` + `cm->counts.eob_branch`
    /// (`vp9_entropymode.h:66-68`).
    coef_counts: rvp9_core::entropy::adapt::CoefCountsBank,
    /// `true` iff the most recently decoded frame was a `KEY_FRAME`
    /// (libvpx `cm->last_frame_type`). Read by
    /// [`coef_adapt_params`](rvp9_core::entropy::adapt::coef_adapt_params)
    /// to pick the AFTER_KEY update factor on the first inter frame
    /// after a keyframe. Updated post-decode in
    /// [`Decoder::decode_keyframe`] (= `true`) and
    /// [`Decoder::decode_inter_frame`] (= `false`).
    last_frame_was_key: bool,
    /// Accumulated mode / partition / skip / inter / ref / tx
    /// counts for the in-flight inter frame. Zeroed at the start of
    /// each inter frame; consumed by
    /// [`adapt_mode_probs`](rvp9_core::entropy::adapt::adapt_mode_probs)
    /// at end-of-frame to produce the next frame's mode probabilities.
    /// Mirrors libvpx `cm->counts` (the y_mode / uv_mode / partition /
    /// inter_mode / intra_inter / comp_inter / comp_ref / single_ref /
    /// switchable_interp / skip / tx fields of `vp9_entropymode.h`
    /// `FRAME_COUNTS`). Keyframes do not call `adapt_mode_probs` (libvpx
    /// `vp9_decodeframe.c:3050`: gated on `!frame_is_intra_only(cm)`),
    /// so this field is only meaningfully populated on inter-frame
    /// decode.
    mode_counts: rvp9_core::entropy::adapt::FrameCounts,
    /// Current per-ref / per-mode loop-filter deltas. The
    /// uncompressed header overlays these via the `Option<i8>` slots
    /// in `LoopFilterParams`; we maintain the resolved running state
    /// here.
    lf_deltas: ResolvedLfDeltas,
    /// `true` once the decoder has seen a successful keyframe.
    /// `parse_packet` blocks inter-frame headers from being acted on
    /// before a keyframe (libvpx `pbi->need_resync`).
    have_resync: bool,
    /// Chroma subsampling established by the most recent key / intra-only
    /// frame's color config. Inter frames do NOT re-encode the color
    /// config (`vp9_decodeframe.c:2704-2766` never calls
    /// `read_bitdepth_colorspace_sampling` on the inter path), so they
    /// inherit `cm->subsampling_{x,y}` from the sequence. The parser
    /// hardcodes inter-frame chroma to `Yuv420` (it has no sequence
    /// state); `parse_packet` overlays this established value onto inter
    /// headers so the destination `Yv12Buffer` is allocated with the
    /// correct subsampling. Without this, the first inter frame of a
    /// 4:4:4 clip would mis-allocate chroma and silently corrupt.
    established_chroma: ChromaFormat,
    /// Per-ref-slot collocated-MV grid. Parallel to
    /// [`Decoder::ref_frame_map`]; `mv_grids[i]` holds the MV grid that
    /// was written by the frame currently bound at ref slot `i`. Used
    /// to populate the `prev_mv` parameter of
    /// [`read_inter_block_mode_info`](rvp9_core::inter::mode_info::read_inter_block_mode_info)
    /// when `use_prev_frame_mvs` fires.
    ///
    /// Mirrors libvpx `cm->cur_frame->mvs` + the
    /// `cm->prev_frame = cm->cur_frame` swap that happens after each
    /// frame finishes decoding (`vp9_decoder.c:478`). We store one grid
    /// per ref slot rather than a single "prev_frame" pointer because
    /// the libvpx model swaps before `refresh_frame_flags` is applied,
    /// so the prev frame is just whatever was at LAST_FRAME's slot
    /// last cycle.
    mv_grids: [Option<std::sync::Arc<Vec<rvp9_core::inter::mv_ref::PrevFrameMvRef>>>;
        rvp9_core::bitstream::REF_FRAMES],
    /// D.22 — most-recently-decoded frame's MV grid + (mi_rows,
    /// mi_cols) dims. Mirrors libvpx `cm->prev_frame->mvs` which is
    /// assigned via `cm->prev_frame = cm->cur_frame` at the end of
    /// every non-`show_existing_frame` decode (`vp9_decoder.c:478`).
    /// This is **independent of `refresh_frame_flags`** — it always
    /// tracks the truly-most-recent decode, regardless of which
    /// `ref_frame_map` slot was (or wasn't) refreshed. Replaces the
    /// previous `mv_grids[ref_frame_indices[0]]` lookup, which broke
    /// for altref frames whose `LAST_FRAME` index points at a slot
    /// that holds a different (older) frame's grid.
    prev_frame_mv_grid: Option<std::sync::Arc<Vec<rvp9_core::inter::mv_ref::PrevFrameMvRef>>>,
    /// `(mi_rows, mi_cols)` of [`Self::prev_frame_mv_grid`]. Used to
    /// guard against a stride mismatch when the next frame's size
    /// differs (libvpx's `use_prev_frame_mvs` gate catches that, but
    /// we double-check here as a defensive measure).
    prev_frame_mv_grid_dims: (usize, usize),
    /// Width / height of the most recently decoded frame. Used by
    /// `decode_inter_frame` to compute `use_prev_frame_mvs` per libvpx
    /// `vp9_decodeframe.c:2983`. `None` before any frame decodes.
    last_frame_size: Option<(u32, u32)>,
    /// `true` iff the most recently decoded frame was `intra_only`
    /// (libvpx `cm->last_intra_only`). Bypasses the prev-frame-MV
    /// salvage for the frame after an intra-only.
    last_intra_only: bool,
    /// `true` iff the most recently decoded frame was a shown frame
    /// (libvpx `cm->last_show_frame`). Always true for keyframes (which
    /// must be shown) and most inter frames; false for ALTREF /
    /// non-shown frames.
    last_show_frame: bool,
    /// Previous frame's per-`(mi_row, mi_col)` segment_id grid, row-
    /// major in 8×8 mi units at stride `last_frame_mi_cols`. Used by
    /// inter frames as the `predicted_segment_id` source for
    /// segmentation inheritance (libvpx
    /// `cm->last_frame_seg_map`, `vp9_decodemv.c:115-126`).
    ///
    /// D.10 (Wave 5) — populated at frame end by swapping the just-
    /// decoded `current_segment_ids` slab into this field. Keyframes
    /// and intra-only frames zero this field at frame end (mirroring
    /// libvpx's `vp9_setup_past_independence` at
    /// `vp9_entropymode.c:424-437`).
    ///
    /// Empty (`Vec::new`) before any frame has decoded; in that state
    /// the inter segment-id reader treats every cell as zero (matches
    /// libvpx `cm->last_frame_seg_map == NULL`).
    prev_segment_ids: Vec<u8>,
    /// `(mi_rows, mi_cols)` of `prev_segment_ids`. Used to detect
    /// resize / frame-dim mismatch — when the new frame's mi-dims
    /// differ, the prev-frame seg map is treated as all-zeros
    /// (matches libvpx, which reallocates `last_frame_seg_map` on
    /// resize and the new allocation is zeroed).
    prev_segment_ids_dims: (usize, usize),
    /// Persistent segmentation feature data across frames. libvpx's
    /// `setup_segmentation` (`vp9_decodeframe.c:1332-1383`) mutates
    /// `cm->seg` in place and only rewrites `feature_data` /
    /// `abs_delta` when the per-frame `update_data` bit is set; on
    /// frames with `update_data == 0` the feature data INHERITS from
    /// the previous frame. Our parser returns a fresh
    /// [`SegmentationParams`] per frame, so we maintain this running
    /// copy here and overlay it onto the parsed value when
    /// `update_data == false`. Needed for AQ-2 vectors where the
    /// keyframe writes the per-segment alt_Q (cyclic refresh) and
    /// every subsequent inter frame leaves the data untouched.
    seg_feature_data:
        [[Option<i16>; rvp9_core::bitstream::SEG_LVL_MAX]; rvp9_core::bitstream::MAX_SEGMENTS],
    /// `abs_delta` companion to [`Self::seg_feature_data`]. Same
    /// inheritance semantics.
    seg_abs_delta: bool,
    /// Stitched, frame-wide `Option<ModeInfo>` grid produced at the
    /// end of the most recent `decode_frame` call. Empty before any
    /// non-`ShowExistingFrame` frame has decoded. Valid only until
    /// the next `decode_frame` call (which overwrites it).
    ///
    /// Stored row-major at stride `last_frame_mi_cols`; a `None`
    /// cell means "outside the visible frame / not written by the
    /// partition walker" (libvpx's `mi_grid_visible` post-`vpx_calloc`
    /// state). Exposed via [`Self::last_frame_mi_grid`] +
    /// [`Self::last_frame_mi_dims`] for the bitstream-diff harness
    /// (`crates/rvp9-harness/src/bin/diff_bitstream.rs`); the encoder
    /// is expected to consume the same accessor for self-validation
    /// against this decoder.
    ///
    /// Strategy A from `notes/stage4-design/diff-bitstream-harness.md`
    /// §9 commit 4: smallest decoder-side change to expose the
    /// post-decode mi-grid without invasive callbacks. Already stitched
    /// internally for the LF apply driver; we just hold onto it.
    last_frame_mi_grid: Vec<Option<ModeInfo>>,
    /// `(mi_rows, mi_cols)` of [`Self::last_frame_mi_grid`]. `(0, 0)`
    /// before any non-`ShowExistingFrame` frame has decoded.
    last_frame_mi_dims: (u32, u32),
    /// G4.1 — Arc-shared SIMD dispatch table. Constructed once per
    /// `Decoder::new` via `rvp9_core::dsp::init()` (`OnceLock`-backed),
    /// then borrowed at frame entry and threaded down into
    /// `decode_tile_*` so the per-TX dispatch site picks up the
    /// AVX2 `idct8x8_add` (or the per-arch fallback) without
    /// per-call `OnceLock::get` overhead.
    ///
    /// G5 (tile threading) inherits this Arc — `WorkerPool::new` will
    /// take an `Arc<SimdDispatch>` clone so per-tile workers don't
    /// re-detect CPU caps.
    simd_dispatch: std::sync::Arc<rvp9_core::dsp::SimdDispatch>,
    /// G5 — tile-threaded decode worker pool. Sized at construction
    /// time from [`Self::num_threads`]; one OS thread per slot,
    /// joined on drop. The decoder convention (HLD V6 §G5 V5 line
    /// 1107: libvpx-faithful inline-worker pattern) is that for
    /// `num_workers = min(num_threads, num_tiles)`, ONE slot
    /// (`num_workers - 1`) runs inline on the caller thread and the
    /// other `num_workers - 1` workers receive a `memcpy` of
    /// `cm.counts` as a baseline. At `num_threads = 1` (the
    /// deterministic-test path), `num_workers = 1`, the inline slot
    /// is the only worker, no actual job is submitted, and the per-
    /// tile decode loop produces `cm.counts = baseline + delta`
    /// exactly as today — no code-path bifurcation.
    ///
    /// **G5 first-cut scope.** The pool is constructed and joined,
    /// the `pool.scope(...)` API is wired through inter / intra
    /// frame decode, and the all-inline path (every tile runs on
    /// the caller thread inside the scope closure) is the only
    /// execution mode. Per-worker `FrameCounts` memcpy + merge
    /// mechanics (V5 §G5 lines 1107-1162) land in the follow-up
    /// commit alongside the disjoint plane-view split (HLD §G5
    /// option (a) `Yv12Buffer::split_into_tile_views`). The G5
    /// primitive is what unlocks that follow-up; the inline path
    /// keeps conformance at 201/0/0/0/201 and inter-thread-count
    /// determinism at `decode(N=1) == decode(N=2)` byte-for-byte.
    worker_pool: rvp9_core::tile_threading::WorkerPool,
    /// G5.2 — recycled per-tile scratch-buffer pool. Replaces the
    /// G5.1 per-tile fresh-`clone_geometry()` allocation pattern,
    /// which at 1080p × 8 tiles × 60 fps was ~1.44 GB/s of allocator
    /// churn + zero-init traffic. The pool persists across frames
    /// and grows lazily to `max(n_tiles_seen)`; per-frame the
    /// parallel-dispatch path calls
    /// [`Yv12ScratchPool::take(n_tiles, &yv12)`](Yv12ScratchPool::take)
    /// before `scope.submit` and
    /// [`Yv12ScratchPool::put`](Yv12ScratchPool::put) after merge.
    ///
    /// **Re-zero safety**: cached buffers are NOT re-zeroed between
    /// uses. Each tile writes only its own column range, the merge
    /// copies only that column range back, and the tile's neighbour
    /// reads are also within the column range — so stale pixels
    /// outside the column range from a prior decode are unreachable.
    /// Re-zeroing here would halve the perf gain. See the
    /// [`Yv12ScratchPool`] type doc's "Re-zero safety" section.
    tile_scratch_pool: Yv12ScratchPool,
}

// Wave A G12 — compile-time check that auto-derive still produces
// `Send + Sync` for `Decoder`. Catches drift: if a future change adds
// an interior-mutability field (`Cell`/`RefCell`/`Rc`/`UnsafeCell`
// without an explicit `unsafe impl Sync`) the assertion fails to
// compile and the regression is caught at the offending PR rather
// than at the first multi-threaded consumer.
//
// Caveat (V6 §G12): this guards against accidental drift only. It
// does not prevent an intentional removal of `Sync` — if a future
// change genuinely needs `!Sync` (e.g., a per-frame cache requiring
// interior mutability), the assertion must be updated AND the
// `Decoder` API contract revisited. Sharing `&Decoder` across threads
// is currently a valid usage pattern (no consumer does it today, but
// it's permitted by the type).
static_assertions::assert_impl_all!(Decoder: Send, Sync);

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// Construct a fresh decoder. The frame pool starts empty; the
    /// four FRAME_CONTEXT banks are pre-populated with libvpx
    /// defaults; loop-filter deltas are all zero and disabled.
    ///
    /// The thread-pool size is determined by
    /// [`std::thread::available_parallelism`], defaulting to 1 if
    /// detection fails. Callers wanting a deterministic single-
    /// threaded decoder (e.g., conformance harness) should prefer
    /// [`Decoder::with_threads`] with `NonZeroUsize::new(1).unwrap()`.
    ///
    /// **G12 plumb status.** The thread-count parameter is currently
    /// stored but not yet consumed — Wave C G5 will wire it into the
    /// `WorkerPool` that drives per-tile parallelism. Callers can
    /// pre-emptively pin the count to keep behaviour stable across
    /// the G5 transition.
    // TODO(G5.1): consider lazy worker spawning — workers should be
    // created on first `scope()` invocation, not on `Decoder::new()`.
    // Today's eager spawn means each `Decoder::new()` in tests/fuzz
    // spawns N threads (`available_parallelism()` by default), which
    // amplifies the cost of fast unit tests by a factor of N. See the
    // review note in `notes/lld/core-tile-threading.md` §
    // "Performance/cost notes".
    #[must_use]
    pub fn new() -> Self {
        // V6 §G12 K6-2: derive the default thread count from the
        // host's reported parallelism, falling back to 1 if the OS
        // refuses to answer (sandboxes, exotic platforms). Keep
        // `Decoder::new()` zero-arg so the ~14 existing call sites
        // don't need to change.
        let n = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        // `n` is `>= 1` by construction (either the OS reported a
        // non-zero value, or we fell back to 1). The `unwrap` is
        // therefore infallible.
        Self::with_threads(std::num::NonZeroUsize::new(n).expect("n >= 1 by construction"))
    }

    /// Construct a decoder with an explicit thread-pool size.
    ///
    /// `n = NonZeroUsize::new(1).unwrap()` disables threading: when
    /// Wave C G5 lands, the decoder will run a single inline worker
    /// (no spawn) and behave identically to today's single-threaded
    /// pipeline. The deterministic-test path (conformance harness)
    /// pins this to 1 so K-1 inline-worker math is independent of
    /// host CPU count.
    ///
    /// **G12 plumb status.** The thread-count parameter is currently
    /// stored as `self.num_threads` but not yet consumed — there is
    /// no `WorkerPool` until Wave C G5 lands. `with_threads(N)` is
    /// equivalent to `new()` plus a stored `N` today; the divergence
    /// will surface once G5 wires the pool through `decode_inter_frame`.
    #[must_use]
    pub fn with_threads(n: std::num::NonZeroUsize) -> Self {
        // Default::default() doesn't work on a 4-array if FrameContext
        // isn't Copy (it isn't — it contains a large coef_probs table).
        // Build each bank explicitly.
        let frame_contexts = [
            FrameContext::new_default(),
            FrameContext::new_default(),
            FrameContext::new_default(),
            FrameContext::new_default(),
        ];
        // G5 — construct the tile-threading worker pool sized at
        // `n` threads. One pool lives for the decoder's lifetime;
        // `Drop` joins every worker. The pool spawns `n` OS threads
        // immediately — at `n = 1` (deterministic-test path) that's
        // exactly one worker. Uniform with multi-threaded shapes:
        // no code-path bifurcation per HLD V6 §G5.
        let worker_pool = rvp9_core::tile_threading::WorkerPool::new(n);
        Self {
            num_threads: n,
            frame_pool: DecoderFramePool::new(),
            ref_frame_map: Default::default(),
            ref_frame_sizes: [None; rvp9_core::bitstream::REF_FRAMES],
            frame_contexts,
            lf_deltas: ResolvedLfDeltas::default(),
            have_resync: true, // until we've seen any frame, treat as needing resync
            mv_counts: rvp9_core::entropy::mv_probs::NmvCounts::default(),
            coef_counts: rvp9_core::entropy::adapt::CoefCountsBank::default(),
            last_frame_was_key: false,
            // Default until the first key/intra-only frame sets it.
            established_chroma: ChromaFormat::Yuv420,
            mode_counts: rvp9_core::entropy::adapt::FrameCounts::default(),
            mv_grids: Default::default(),
            prev_frame_mv_grid: None,
            prev_frame_mv_grid_dims: (0, 0),
            last_frame_size: None,
            last_intra_only: false,
            last_show_frame: false,
            prev_segment_ids: Vec::new(),
            prev_segment_ids_dims: (0, 0),
            seg_feature_data: [[None; rvp9_core::bitstream::SEG_LVL_MAX];
                rvp9_core::bitstream::MAX_SEGMENTS],
            seg_abs_delta: false,
            last_frame_mi_grid: Vec::new(),
            last_frame_mi_dims: (0, 0),
            // G4.1 — populate the process-wide SimdDispatch once at
            // decoder construction. Subsequent calls to `dsp::init` are
            // free (Arc::clone of the OnceLock slot). HLD-simd §2.2
            // R-09: dispatch is fully populated before any decode
            // worker can spawn.
            simd_dispatch: rvp9_core::dsp::init(),
            worker_pool,
            tile_scratch_pool: Yv12ScratchPool::new(),
        }
    }

    /// Worker-pool size this decoder was constructed with.
    ///
    /// Returns the value passed to [`Decoder::with_threads`], or the
    /// auto-detected value from [`Decoder::new`]. Currently advisory
    /// only — Wave C G5 will consume it.
    #[must_use]
    pub fn num_threads(&self) -> std::num::NonZeroUsize {
        self.num_threads
    }

    /// Borrow the decoder's frame pool (immutable).
    #[must_use]
    pub fn frame_pool(&self) -> &DecoderFramePool {
        &self.frame_pool
    }

    /// Resolve a logical ref-frame-map index (`0..REF_FRAMES = 0..8`)
    /// to the [`FrameBuf`] currently bound there, or `None` if the
    /// slot has never been written.
    ///
    /// This is the libvpx-faithful indirection: `ref_frame_map[i]`
    /// holds a frame-pool slot index, not a direct `Arc`. Mirrors
    /// `cm->ref_frame_map[i] -> &cm->buffer_pool->frame_bufs[
    /// cm->ref_frame_map[i]].buf`.
    ///
    /// G12 refactor (2026-05-21). Replaces the pre-G12
    /// `self.ref_frame_map[i].clone()` direct-Arc pattern.
    #[must_use]
    pub fn ref_frame_buf(&self, ref_idx: usize) -> Option<FrameBuf> {
        self.ref_frame_map
            .get(ref_idx)
            .and_then(|slot| *slot)
            .and_then(|pool_slot| self.frame_pool.get(pool_slot as usize))
    }

    /// Borrow the decoder's frame pool (mutable). Used by tests and
    /// by the eventual pixel-reconstruction path.
    pub fn frame_pool_mut(&mut self) -> &mut DecoderFramePool {
        &mut self.frame_pool
    }

    /// G5.2 — number of scratch buffers currently parked in the
    /// per-tile scratch pool (i.e. **not** out at workers). Diagnostic
    /// accessor used by the `tile_scratch_pool_*` regression tests
    /// to verify pool growth + post-warmup steady-state.
    ///
    /// Outside of tests, this is informational only. The pool's
    /// behaviour is fully internal.
    #[must_use]
    pub fn tile_scratch_pool_buffer_count(&self) -> usize {
        self.tile_scratch_pool.buffer_count()
    }

    /// Internal helper: write `ref_frame_map[i] = Some(new_idx)`,
    /// keeping `frame_pool.ref_counts` in sync. If the slot already
    /// held `Some(old_idx)`, decrements that slot's ref-count first;
    /// then increments `ref_counts[new_idx]`.
    ///
    /// Every production write to `ref_frame_map[i]` MUST go through
    /// this helper (or [`Self::clear_ref_slot`] for the `= None`
    /// case) — direct field assignment skips the bookkeeping and
    /// breaks the [`Self::frame_pool`] invariant.
    fn write_ref_slot(&mut self, i: usize, new_idx: u8) {
        if let Some(old_idx) = self.ref_frame_map[i] {
            if old_idx == new_idx {
                // No-op: count stays the same (one dec + one inc).
                return;
            }
            self.frame_pool.dec_ref(old_idx);
        }
        self.ref_frame_map[i] = Some(new_idx);
        self.frame_pool.inc_ref(new_idx);
    }

    /// Test-only setter that writes a ref-map → pool-slot pointer.
    ///
    /// Replaces direct `self.ref_frame_map[i] = Some(pool_idx)` field
    /// writes from in-crate tests so that the structural invariant
    /// documented on [`Self::frame_pool`] is callable from one named
    /// site (greppable, attributable). Goes through
    /// [`Self::write_ref_slot`] so the `frame_pool.ref_counts`
    /// bookkeeping fires identically to a production refresh.
    ///
    /// Caller MUST have already populated `frame_pool.slots[pool_idx]`
    /// via [`Self::frame_pool_mut`] + [`DecoderFramePool::assign_slot`]
    /// before calling this; otherwise the next
    /// [`Self::ref_frame_buf`] lookup will return `None` and the
    /// downstream decode path will surface `EmptyInterRef` /
    /// `EmptySlot`.
    #[cfg(test)]
    pub(crate) fn set_ref_slot(&mut self, i: usize, pool_idx: u8) {
        self.write_ref_slot(i, pool_idx);
    }

    /// Borrow a specific frame-context bank (0..=3).
    ///
    /// # Panics
    ///
    /// If `idx >= FRAME_CONTEXTS`.
    #[must_use]
    pub fn frame_context(&self, idx: usize) -> &FrameContext {
        &self.frame_contexts[idx]
    }

    /// `true` if the decoder is waiting for a keyframe (the initial
    /// state, plus after a bitstream resync). Inter frames received
    /// in this state are recognised but their compressed payloads
    /// are not consumed.
    #[must_use]
    pub fn needs_resync(&self) -> bool {
        self.have_resync
    }

    /// Borrow the frame-wide `Option<ModeInfo>` grid produced by the
    /// most recent [`Self::decode_frame`] call. Row-major at stride
    /// [`Self::last_frame_mi_dims`]`.1`; `None` cells mean "outside
    /// the visible frame / not touched by the partition walker".
    ///
    /// **Lifetime:** valid until the next [`Self::decode_frame`] call
    /// (which overwrites the buffer in place).
    ///
    /// Returns an empty slice before any non-`ShowExistingFrame` frame
    /// has decoded.
    ///
    /// Consumed by the bitstream-diff harness
    /// (`crates/rvp9-harness/src/bin/diff_bitstream.rs`) at Level 4.
    /// Strategy A per `notes/stage4-design/diff-bitstream-harness.md`
    /// §9 commit 4.
    #[must_use]
    pub fn last_frame_mi_grid(&self) -> &[Option<ModeInfo>] {
        &self.last_frame_mi_grid
    }

    /// `(mi_rows, mi_cols)` of the grid returned by
    /// [`Self::last_frame_mi_grid`]. `(0, 0)` before any non-
    /// `ShowExistingFrame` frame has decoded.
    #[must_use]
    pub fn last_frame_mi_dims(&self) -> (u32, u32) {
        self.last_frame_mi_dims
    }

    /// Parse one transport packet (an IVF chunk, RTP packet, etc.)
    /// into its 1..=8 component VP9 frames.
    ///
    /// For each component frame, parses the uncompressed header and
    /// returns the result via [`PacketFrames`]. The compressed
    /// portion of each frame is NOT parsed yet (Task 1e); the
    /// `compressed_data` slice in the returned [`ParsedFrame`] is the
    /// raw bytes following the uncompressed header.
    ///
    /// # Errors
    ///
    /// Propagates any [`BitstreamError`] from the superframe-index
    /// parse or the per-frame header parse.
    pub fn parse_packet<'a>(&mut self, data: &'a [u8]) -> Result<PacketFrames<'a>, BitstreamError> {
        let sf = parse_superframe_index(data)?;
        let mut frame_slices: Vec<&'a [u8]> = Vec::new();

        match sf {
            Superframe::Single => {
                frame_slices.push(data);
            }
            Superframe::Multi {
                sizes,
                count,
                index_size,
            } => {
                let payload_end = data.len() - index_size;
                let mut offset = 0;
                for &sz in sizes.iter().take(count) {
                    let sz = sz as usize;
                    if offset + sz > payload_end {
                        return Err(BitstreamError::InvalidSuperframeIndex {
                            reason:
                                rvp9_core::bitstream::SuperframeIndexErrorReason::TruncatedIndex {
                                    declared: offset + sz,
                                    available: payload_end,
                                },
                        });
                    }
                    frame_slices.push(&data[offset..offset + sz]);
                    offset += sz;
                }
            }
        }

        // Pass the running per-slot ref-frame sizes to the parser so
        // it can resolve `setup_frame_size_with_refs` (`vp9_decodeframe.c:1559-1574`)
        // when the header inherits size from a previous frame.
        let mut parsed: Vec<ParsedFrame<'a>> = Vec::with_capacity(frame_slices.len());
        for frame_data in frame_slices {
            let (mut header, header_end) =
                parse_uncompressed_header(frame_data, &self.ref_frame_sizes)?;
            // Inter-frame chroma forwarding. Inter frames never re-read the
            // color config (`vp9_decodeframe.c:2704-2766`); they inherit the
            // sequence subsampling. The parser hardcodes their chroma to
            // `Yuv420` (no sequence state), so we overlay the value
            // established by the most recent key/intra-only frame and, in
            // turn, record any freshly-parsed color config. ShowExistingFrame
            // carries no color config and just replays a slot, so it neither
            // reads nor updates the established chroma.
            match header.kind {
                FrameKind::KeyFrame | FrameKind::IntraOnly { .. } => {
                    self.established_chroma = header.chroma;
                }
                FrameKind::InterFrame { .. } => {
                    header.chroma = self.established_chroma;
                }
                FrameKind::ShowExistingFrame { .. } => {}
            }
            // Keyframe and intra-only frames advertise a fresh width
            // and refresh some/all ref-frame-map slots; mirror libvpx's
            // `setup_frame_size` propagation (`vp9_decodeframe.c:2698`)
            // so subsequent inter frames in the same packet (or the
            // next packet) can resolve their sizes during parse.
            // Keyframes set every slot (`refresh_frame_flags = 0xFF` per
            // spec); intra-only carries an explicit 8-bit flag in the
            // header that we surface via `FrameKind::IntraOnly`.
            self.refresh_ref_frame_sizes(&header);
            parsed.push(ParsedFrame {
                header,
                header_end_byte: header_end,
                full_frame_data: frame_data,
            });
        }

        Ok(PacketFrames { frames: parsed })
    }

    /// Propagate the just-parsed header's frame size into the
    /// `ref_frame_sizes` cache, respecting the `refresh_frame_flags`
    /// mask. Mirrors the post-parse update libvpx performs at
    /// `vp9_decodeframe.c:2691` (keyframes set every slot) and the
    /// intra-only / inter mask handling at `:2730`.
    fn refresh_ref_frame_sizes(&mut self, header: &UncompressedHeader) {
        match header.kind {
            FrameKind::KeyFrame => {
                let size = Some((header.width, header.height));
                for slot in self.ref_frame_sizes.iter_mut() {
                    *slot = size;
                }
            }
            FrameKind::IntraOnly {
                refresh_frame_flags,
                ..
            } => {
                let size = (header.width, header.height);
                for (i, slot) in self.ref_frame_sizes.iter_mut().enumerate() {
                    if (refresh_frame_flags >> i) & 1 != 0 {
                        *slot = Some(size);
                    }
                }
            }
            FrameKind::InterFrame {
                refresh_frame_flags,
                ..
            } => {
                // Inter frames refresh slots per the mask. The width
                // and height stored here is the frame's decoded /
                // inherited (width, height) — see `parse_uncompressed
                // _header`. This lets subsequent inter frames in the
                // same packet resolve `setup_frame_size_with_refs`.
                let size = (header.width, header.height);
                for (i, slot) in self.ref_frame_sizes.iter_mut().enumerate() {
                    if (refresh_frame_flags >> i) & 1 != 0 {
                        *slot = Some(size);
                    }
                }
            }
            FrameKind::ShowExistingFrame { .. } => {}
        }
    }

    /// Update the running loop-filter deltas with a parsed
    /// `LoopFilterParams` from this frame's header. `None` slots in
    /// the params are interpreted as "keep previous"; `Some(v)`
    /// overlay onto the existing state.
    ///
    /// Mirrors libvpx's handling at `vp9_decodeframe.c:2819` (the
    /// `setup_loopfilter` block plus the running-state plumbing in
    /// `cm->lf`).
    pub fn merge_lf_params(&mut self, lf: &rvp9_core::bitstream::LoopFilterParams) {
        self.lf_deltas.enabled = lf.mode_ref_delta_enabled;
        if lf.mode_ref_delta_update {
            for (i, slot) in lf.ref_deltas.iter().enumerate() {
                if let Some(v) = slot {
                    self.lf_deltas.ref_deltas[i] = *v;
                }
            }
            for (i, slot) in lf.mode_deltas.iter().enumerate() {
                if let Some(v) = slot {
                    self.lf_deltas.mode_deltas[i] = *v;
                }
            }
        }
    }

    /// Read-only access to the running loop-filter delta state.
    /// Required by the per-frame filter-level resolver
    /// [`rvp9_core::loopfilter::resolve_filter_levels`].
    #[must_use]
    pub fn lf_deltas(&self) -> ResolvedLfDeltas {
        self.lf_deltas
    }

    /// Parse the compressed-header (first-partition) bytes of a
    /// non-show-existing frame into the decoder's selected
    /// `FrameContext` bank. Mirrors libvpx's call sequence at
    /// `vp9_decodeframe.c:2944-2948`:
    ///
    /// ```text
    /// read_uncompressed_header(...);
    /// // ...
    /// read_compressed_header(...);    // ← this method
    /// // ...
    /// decode_tiles(...);
    /// ```
    ///
    /// The selected bank is `frame_contexts[header.tail.frame_context_idx]`.
    /// The compressed-header byte slice is `frame.full_frame_data
    /// [header_end_byte .. header_end_byte + first_partition_size]`.
    ///
    /// The `compound_allowed` input to `parse_compressed_header` is
    /// conservatively set to `false` for now — proper POC-direction
    /// tracking lands with the reference-frame plumbing in Task 8.2+.
    /// This matches libvpx behaviour for keyframes / intra-only
    /// frames (where compound is always disallowed) but is
    /// conservative for inter frames.
    ///
    /// Returns `(tx_mode, reference_mode)` so the caller can wire
    /// them into per-block decode.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::WrongFrameKind`] if `frame.header.kind` is
    ///   `ShowExistingFrame` (those frames have no compressed
    ///   header).
    /// - [`DecodeError::TruncatedCompressedHeader`] if the declared
    ///   `first_partition_size` extends past the frame's data slice.
    /// - [`DecodeError::Bitstream`] for any error surfaced by
    ///   [`parse_compressed_header`] (marker-bit fail / read past end).
    pub fn parse_compressed_partition(
        &mut self,
        frame: &ParsedFrame<'_>,
    ) -> Result<CompressedHeaderResult, DecodeError> {
        let tail = frame
            .header
            .tail
            .as_ref()
            .ok_or(DecodeError::WrongFrameKind)?;

        // Slice out [header_end .. header_end + first_partition_size].
        let header_end = frame.header_end_byte;
        let fps = tail.first_partition_size as usize;
        let total = frame.full_frame_data.len();
        let needed_end =
            header_end
                .checked_add(fps)
                .ok_or(DecodeError::TruncatedCompressedHeader {
                    declared: usize::MAX,
                    available: total,
                })?;
        if needed_end > total {
            return Err(DecodeError::TruncatedCompressedHeader {
                declared: needed_end,
                available: total,
            });
        }
        let compressed = &frame.full_frame_data[header_end..needed_end];

        // Compute parser options from the uncompressed header.
        let lossless = tail.quantization.is_lossless();
        let intra_only = frame.header.kind.is_intra();
        let (interp_filter, allow_high_precision_mv) = match frame.header.kind {
            FrameKind::InterFrame {
                interp_filter,
                allow_high_precision_mv,
                ..
            } => (interp_filter, allow_high_precision_mv),
            // Intra-only / keyframe: these fields are unused — pass
            // any sane defaults. `parse_compressed_header` ignores
            // them when `intra_only == true`.
            _ => (InterpFilter::EightTap, false),
        };
        let opts = CompressedHeaderOptions {
            lossless,
            intra_only,
            interp_filter,
            // Conservative pending POC tracking (Task 8.2+).
            compound_allowed: false,
            allow_high_precision_mv,
        };

        // Drive the parser against the selected FrameContext bank.
        let idx = tail.frame_context_idx as usize;
        let fc = &mut self.frame_contexts[idx];
        parse_compressed_header(compressed, fc, opts).map_err(DecodeError::Bitstream)
    }

    /// Handle a `show_existing_frame` parsed header by resolving the
    /// 3-bit ref-frame-map index in the header to a [`FrameBuf`].
    /// This is the simplest possible decoder output path — no
    /// compressed-header / tile decoding required.
    ///
    /// Mirrors libvpx's `vp9_decodeframe.c:2664-2679` early-exit:
    /// after recording the show-existing buffer index, libvpx
    /// returns the buffer at `cm->ref_frame_map[i]`.
    ///
    /// G12 refactor (2026-05-21). Previously this method consulted
    /// the frame pool directly via the header's 3-bit `slot` field,
    /// which only worked because pre-G12 code mirror-wrote every
    /// refresh into both `ref_frame_map[i]` AND `frame_pool[i]`.
    /// G12 collapsed the mirror; now `decode_show_existing_frame`
    /// goes through `ref_frame_map` (libvpx-faithful).
    ///
    /// # Errors
    ///
    /// - [`DecodeError::WrongFrameKind`] if `header.kind` is not
    ///   `FrameKind::ShowExistingFrame`.
    /// - [`DecodeError::EmptySlot`] if the referenced slot is empty
    ///   (libvpx's "Buffer N does not contain a decoded frame"
    ///   rejection at `vp9_decodeframe.c:2667`).
    pub fn decode_show_existing_frame(
        &self,
        header: &UncompressedHeader,
    ) -> Result<FrameBuf, DecodeError> {
        let slot = match header.kind {
            FrameKind::ShowExistingFrame { slot } => slot,
            _ => return Err(DecodeError::WrongFrameKind),
        };
        self.ref_frame_buf(slot as usize)
            .ok_or(DecodeError::EmptySlot { slot })
    }

    /// Drive one parsed frame end-to-end and return the decoded
    /// [`FrameBuf`] that should be shown (or `Ok(None)` for a
    /// not-shown frame whose payload still references something).
    ///
    /// This is the single entry point the conformance harness uses
    /// to walk a packet's worth of frames. All four [`FrameKind`]
    /// arms are now plumbed through to real decode paths:
    /// `ShowExistingFrame`, `KeyFrame`, `IntraOnly` (Wave A G1),
    /// and `InterFrame` (with G3 covering the empty-inter-ref
    /// rejection). See the [`DecodeError`] enum for the canonical
    /// list of structured failures any of these paths may return.
    ///
    /// # Errors
    ///
    /// Returns the structured [`DecodeError`] variant appropriate to
    /// the failing path — see that enum's doc for the full set.
    pub fn decode_frame(&mut self, frame: &ParsedFrame<'_>) -> Result<FrameBuf, DecodeError> {
        // D.22 — global decode-frame counter, parallels libvpx's
        // `rvp9_trace_current_frame` (incremented at the top of
        // `vp9_decode_frame`, includes altref frames). Used by
        // RVP9_TRACE_INTERMI / RVP9_TRACE_PART traces to tag each line
        // with the same `f=N` libvpx emits, so per-frame diffs are
        // line-up-able.
        RVP9_TRACE_FRAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // G7.1 closed; all four chroma formats supported. The decode
        // chain (intra/inter reconstruction, loopfilter) is `ss`-aware
        // for every legal VP9 layout:
        //   - Yuv420 (ss=1,1): original 4:2:0 path.
        //   - Yuv444 (ss=0,0): chroma grid == luma grid; tile.rs / recon /
        //     loopfilter drive U/V through the luma-shaped code paths
        //     (`LF_PATH_444`, `uv_plane_tx_size((0,0)) == luma tx`).
        //   - Yuv422 (ss=1,0) / Yuv440 (ss=0,1): asymmetric — the
        //     loopfilter walks the MI grid directly via
        //     `filter_block_plane_non420` (libvpx `LF_PATH_SLOW`); the
        //     per-block recon already handles ss-aware geometry from
        //     G7.1a/b.
        //
        // Inter-frame chroma forwarding (G7.1c): inter frames inherit the
        // sequence chroma established by the most recent key/intra-only
        // frame — `parse_packet` overlays `self.established_chroma` onto
        // the inter header (the parser itself hardcodes inter chroma to
        // Yuv420). So `frame.header.chroma` here is already the correct
        // subsampling for every frame kind, and the destination buffer
        // allocates right. ShowExistingFrame still skips any gating
        // because it merely hands back an already-decoded slot.
        //
        // The `UnsupportedProfile` variant remains in `DecodeError` for
        // future Profiles 2/3 (high-bit-depth) — Profile 1 no longer
        // rejects here, but HBD profiles still will when they're wired.
        match frame.header.kind {
            FrameKind::ShowExistingFrame { .. } => self.decode_show_existing_frame(&frame.header),
            FrameKind::KeyFrame => self.decode_keyframe(frame),
            FrameKind::IntraOnly { .. } => self.decode_intra_only_frame(frame),
            FrameKind::InterFrame { .. } => self.decode_inter_frame(frame),
        }
    }

    /// Decode a keyframe end-to-end, returning the freshly-allocated
    /// [`FrameBuf`] that holds the reconstructed pixels.
    ///
    /// Mirrors libvpx's keyframe decode driver from
    /// `vp9/decoder/vp9_decodeframe.c:read_compressed_header` →
    /// `decode_tiles` for the `tile_cols == 1 && tile_rows == 1` path
    /// only (multi-tile vectors surface as
    /// [`DecodeError::TileDecode`]).
    ///
    /// Keyframes refresh every slot in `ref_frame_map` (libvpx implicit
    /// `refresh_frame_flags = 0xFF` at `vp9_decodeframe.c:2691`), so
    /// this method delegates to [`Decoder::decode_intra_frame_inner`]
    /// with `refresh_mask = 0xFF` and `intra_only = false`.
    fn decode_keyframe(&mut self, frame: &ParsedFrame<'_>) -> Result<FrameBuf, DecodeError> {
        self.decode_intra_frame_inner(frame, 0xFF, /* intra_only = */ false)
    }

    /// Decode an intra-only frame end-to-end, returning the freshly-
    /// allocated [`FrameBuf`] that holds the reconstructed pixels.
    ///
    /// IntraOnly frames are structurally identical to keyframes from
    /// the per-block reconstruction perspective — both go through
    /// `read_intra_frame_mode_info` (`vp9_decodemv.c:822`) and share
    /// the `vp9_kf_partition_probs` table (`vp9_onyxc_int.h:371`).
    /// They differ from keyframes only in three header-level details
    /// (G1, HLD V6 §G1):
    ///
    /// 1. **FrameContext reset semantics.** Keyframes always overwrite
    ///    every `frame_contexts[]` bank with libvpx defaults inside
    ///    `vp9_setup_past_independence` (`vp9_entropymode.c:452-459`).
    ///    Intra-only frames consult the 2-bit `reset_frame_context`
    ///    header field: `3` → all banks reset, `2` → only the selected
    ///    bank reset, `0`/`1` → keep existing banks.
    /// 2. **ref_frame_map refresh.** Keyframes splat every slot
    ///    (implicit `refresh_frame_flags = 0xFF`); intra-only honours
    ///    the explicit 8-bit `refresh_frame_flags` byte from the header.
    /// 3. **last_frame_type.** Libvpx sets `cm->last_frame_type =
    ///    cm->frame_type` (`vp9_decodeframe.c:2645`). Intra-only's
    ///    `frame_type == INTER_FRAME` (`cm->intra_only` is the
    ///    distinguishing flag), so the next frame's
    ///    `coef_adapt_params` picks the non-AFTER_KEY constants.
    ///
    /// The body itself is shared with [`Decoder::decode_keyframe`]
    /// via [`Decoder::decode_intra_frame_inner`].
    fn decode_intra_only_frame(
        &mut self,
        frame: &ParsedFrame<'_>,
    ) -> Result<FrameBuf, DecodeError> {
        let (reset_frame_context, refresh_frame_flags) = match frame.header.kind {
            FrameKind::IntraOnly {
                reset_frame_context,
                refresh_frame_flags,
            } => (reset_frame_context, refresh_frame_flags),
            _ => return Err(DecodeError::WrongFrameKind),
        };
        // Per `vp9_setup_past_independence` (`vp9_entropymode.c:452-459`):
        // for an intra-only frame (with `error_resilient == false`), the
        // bank-reset behaviour is gated on `reset_frame_context`:
        //   - 3 ⇒ reset all FRAME_CONTEXTS to defaults
        //   - 2 ⇒ reset only `cm->frame_contexts[cm->frame_context_idx]`
        //   - 0 or 1 ⇒ keep existing banks (the working copy `cm->fc`
        //              is loaded with defaults but the bank slots stay).
        // We materialise defaults directly via `FrameContext::new_default`.
        // Error-resilient frames force the all-banks reset path
        // (`vp9_entropymode.c:452`).
        let force_all_reset = frame.header.error_resilient || reset_frame_context == 3;
        let fc_idx = frame
            .header
            .tail
            .as_ref()
            .ok_or(DecodeError::WrongFrameKind)?
            .frame_context_idx as usize;
        if force_all_reset {
            for bank in self.frame_contexts.iter_mut() {
                *bank = FrameContext::new_default();
            }
        } else if reset_frame_context == 2 {
            self.frame_contexts[fc_idx] = FrameContext::new_default();
        }
        // `vp9_setup_past_independence` is also called whenever the
        // frame is intra-only, regardless of `reset_frame_context` —
        // it always wipes the seg map, sets default LF deltas, and
        // resets `ref_frame_sign_bias`. The seg-map reset is handled
        // by `resolve_segmentation(is_keyframe = true)` inside the
        // inner driver below, and the LF-delta reset is handled by
        // the unconditional `self.lf_deltas = ...keyframe defaults...`
        // assignment near the top of the LF-apply block. (Sign-bias
        // isn't tracked on `Decoder` — it's read from the inter-frame
        // header on demand.)
        self.decode_intra_frame_inner(frame, refresh_frame_flags, /* intra_only = */ true)
    }

    /// Shared intra-frame decode driver. Drives the full pipeline —
    /// compressed-header parse, tile walk, partition walk, per-block
    /// mode decode, plane reconstruction, loop-filter apply,
    /// ref-slot refresh, end-of-frame state mirror — for both the
    /// `FrameKind::KeyFrame` and `FrameKind::IntraOnly` paths.
    ///
    /// The two callers differ only in their entry-and-exit gymnastics:
    /// keyframes pass `refresh_mask = 0xFF` and `intra_only = false`;
    /// intra-only frames pass `tail.refresh_frame_flags` and
    /// `intra_only = true`. They also differ in the bank-reset
    /// performed BEFORE this method runs — keyframes implicitly reset
    /// nothing (the four banks already hold defaults at construction
    /// and any subsequent adapt is per-frame), while intra-only frames
    /// may consult `reset_frame_context` to selectively reset banks
    /// (handled in [`Decoder::decode_intra_only_frame`]).
    fn decode_intra_frame_inner(
        &mut self,
        frame: &ParsedFrame<'_>,
        refresh_mask: u8,
        intra_only: bool,
    ) -> Result<FrameBuf, DecodeError> {
        let tail = frame
            .header
            .tail
            .as_ref()
            .ok_or(DecodeError::WrongFrameKind)?;

        // G2 (decoder-hardening V6 §G2): lossless decode is wired
        // end-to-end through the inverse Walsh-Hadamard transform at
        // `crates/rvp9-core/src/transforms/inverse.rs::iwht4x4_{16,1}_add`.
        // The dispatch in `reconstruct::inverse_transform_add` selects
        // the WHT family whenever `lossless == true` at TX_4X4. VP9
        // restricts lossless mode to TX_4X4 (`vp9_decodeframe.c:280-281`),
        // so the rest of the keyframe driver is unchanged.

        // 1. Parse the compressed header into the selected FrameContext
        //    bank to get `tx_mode` (and the coef-prob updates etc.).
        //
        // **D.9 (cascade-fix preamble):** snapshot the bank *before* the
        // compressed-header diff-updates land, so the backward-adapt
        // pass below can use it as the `pre` baseline. libvpx's
        // `read_compressed_header` operates on a working copy `cm->fc`
        // (loaded from the bank at the top of `vp9_decode_frame`), and
        // `vp9_adapt_coef_probs` then reads `pre_fc = cm->frame_contexts
        // [frame_context_idx]` — the original bank value, *unaffected*
        // by the compressed-header deltas (`vp9_entropy.c:1058`). Our
        // [`parse_compressed_partition`] mutates the bank in place, so
        // without this snapshot the adapt pass would use the modified
        // bank as `pre`, causing the chained-merge baseline to drift
        // away from libvpx after every frame.
        let fc_idx = tail.frame_context_idx as usize;
        let pre_fc = self.frame_contexts[fc_idx].clone();
        let compressed_result = self.parse_compressed_partition(frame)?;

        // 2. Slice out the tile data (everything past the compressed
        //    header).
        let header_end = frame.header_end_byte;
        let fps = tail.first_partition_size as usize;
        let tile_data_start = header_end + fps;
        let total = frame.full_frame_data.len();
        if tile_data_start > total {
            return Err(DecodeError::TruncatedCompressedHeader {
                declared: tile_data_start,
                available: total,
            });
        }
        let tile_section = &frame.full_frame_data[tile_data_start..];

        // 3. Compute frame mi-dimensions + tile-count.
        let mi_cols = ((frame.header.width as usize) + 7) >> 3;
        let mi_rows = ((frame.header.height as usize) + 7) >> 3;
        let tile_cols = 1u32 << tail.tile_info.log2_tile_cols;
        let tile_rows = 1u32 << tail.tile_info.log2_tile_rows;

        // 4. Parse per-tile byte payloads. Stage 3 Task E.2 — wire up
        //    single-threaded multi-tile decode. libvpx splits the tile
        //    section by the 4-byte BE size prefix before every tile
        //    except the last (`get_tile_buffers`,
        //    `vp9_decodeframe.c:1649-1697`); we walk the resulting
        //    `Vec<TileBuffer>` in row-major order below.
        let buffers = parse_tile_buffers(tile_section, tile_cols, tile_rows)
            .map_err(DecodeError::Bitstream)?;
        if buffers.is_empty() {
            return Err(DecodeError::Bitstream(BitstreamError::Truncated {
                needed: 1,
                available: 0,
            }));
        }

        // 5. Allocate the destination FrameBuf. Chroma format flows
        //    from the parsed header — for Profile 0 always `Yuv420`,
        //    for Profile 1 one of the three non-420 sub-modes (gated
        //    upstream in `decode_frame` pending G7.1).
        let mut yv12 = Yv12Buffer::alloc(
            frame.header.width,
            frame.header.height,
            frame.header.chroma,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .map_err(|_| DecodeError::FrameAlloc)?;

        // 6. Per-frame state — drive every tile through a shared
        //    accumulator (counts) + shared output (`yv12`,
        //    `cur_segment_ids`). Each tile gets its own fresh
        //    `TileDecodeCtx` / `TileCoefCtx` / `BoolReader` allocated
        //    inside `decode_tile_intra`. This mirrors libvpx
        //    `decode_tiles` (`vp9_decodeframe.c:2200-2240`) which
        //    constructs a fresh `TileWorkerData` per tile but writes
        //    to the shared `cm->cur_frame->yv12` and accumulates
        //    counts into `cm->counts`.
        // D.10 — resolve the per-frame segmentation against the
        // running state, then use the RESOLVED view everywhere
        // downstream (per-segment alt_Q lookup + the loop-filter
        // segmentation view). Without this, AQ-2 inter frames lose
        // the keyframe's `feature_data` (encoder sets it once at the
        // keyframe with `update_data = 1`, then leaves it alone on
        // every inter frame).
        let resolved_seg =
            self.resolve_segmentation(&tail.segmentation, /* is_keyframe = */ true);
        // Per-segment alt-Q feature data: feature_data[seg][SEG_LVL_ALT_Q].
        let mut seg_alt_q = [None; rvp9_core::loopfilter::MAX_SEGMENTS];
        if resolved_seg.enabled {
            for (i, s) in resolved_seg.feature_data.iter().enumerate() {
                seg_alt_q[i] = s[0]; // SEG_LVL_ALT_Q == 0
            }
        }
        let params = FrameDecodeParams {
            mi_cols,
            mi_rows,
            tx_mode: compressed_result.tx_mode,
            segmentation_update_map: resolved_seg.enabled && resolved_seg.update_map,
            segmentation_enabled: resolved_seg.enabled,
            segmentation_temporal_update: resolved_seg.temporal_update,
            seg_tree_probs: resolved_seg.tree_probs,
            seg_pred_probs: resolved_seg.pred_probs,
            lossless: tail.quantization.is_lossless(),
            base_qindex: i32::from(tail.quantization.base_qindex),
            y_dc_delta_q: i32::from(tail.quantization.y_dc_delta_q),
            uv_dc_delta_q: i32::from(tail.quantization.uv_dc_delta_q),
            uv_ac_delta_q: i32::from(tail.quantization.uv_ac_delta_q),
            seg_alt_q,
            seg_abs_delta: resolved_seg.abs_delta,
        };
        // Stage 3 Task E.1.b — zero the coef counts at the start of
        // every frame and pass an accumulator into the tile decoder.
        // libvpx clears `cm->counts` in `read_uncompressed_header`
        // (`vp9_decodeframe.c:2784`) for every frame whose
        // `frame_parallel_decoding_mode == 0`. We zero unconditionally,
        // which is equivalent: the counts are consumed only by backward
        // adaptation, which is now correctly gated on
        // `!frame_parallel_decoding_mode` at end-of-frame — so on the
        // frames where libvpx skips the clear it also skips the
        // adaptation that would read them.
        self.coef_counts = rvp9_core::entropy::adapt::CoefCountsBank::default();
        // D.10 — allocate the current frame's segment_id slab. Keyframes
        // either write a fresh ID per block (when `update_map = true`)
        // or default every cell to 0. After decode finishes we wipe
        // `prev_segment_ids` (libvpx `vp9_setup_past_independence` at
        // `vp9_entropymode.c:424-437`) — keyframes RESET the seg map.
        let mut cur_segment_ids = vec![0u8; mi_rows * mi_cols];

        // Stage 3 Task E.2 — iterate `(tile_row, tile_col)` in raster
        // order, decoding each tile sequentially with a fresh per-tile
        // `TileDecodeCtx` + `TileCoefCtx` + `BoolReader` (allocated
        // inside `decode_tile_intra`). Per-tile state is intentionally
        // isolated — VP9 specifies that the `above_*` / `left_*`
        // entropy contexts reset at every tile boundary
        // (`vp9_decodeframe.c:2200-2240`). Keep the per-tile mi grids
        // so we can stitch them into a frame-wide mi grid for the LF
        // driver below.
        let mut tile_decode_ctxs: Vec<(tile::TileBounds, TileDecodeCtx)> =
            Vec::with_capacity(buffers.len());
        // G5.1 — real parallel tile dispatch. Eligibility:
        //   * `tile_rows == 1` — column-only split makes per-tile
        //     pixel rectangles disjoint by construction (VP9 tile-col
        //     boundaries align to MI = 8-luma-px, which is also
        //     chroma-aligned at 4:2:0).
        //   * `num_tiles > 1` — degenerate 1-tile frames stay on the
        //     inline path (no benefit to dispatching one job).
        //   * `num_threads > 1` — the worker-pool size knob; respects
        //     `Decoder::with_threads(1)` as the deterministic path.
        //
        // For ineligible frames (tile_rows > 1 OR single tile OR
        // single-threaded knob), we keep the inline path: every tile
        // runs sequentially on the caller thread, sharing the
        // decoder's `coef_counts` accumulator and the destination
        // `yv12`. This is the pre-G5.1 shape.
        //
        // For eligible frames we allocate per-tile scratch buffers,
        // fan-out into the pool, and merge results column-range-wise
        // back into `yv12` + sum per-tile `coef_counts` into the
        // decoder's running total. See `notes/lld/core-tile-
        // threading.md` § "G5.1 dispatch" for the design rationale
        // (especially the per-tile fresh-Yv12Buffer allocation cost
        // vs. unsafe slice-aliasing tradeoff).
        // G5.3: gate is "multi-tile + multi-thread"; the
        // tile_rows == 0 restriction of G5.1 is gone because
        // `merge_tile_rect_from` handles the 2D grid case.
        let parallel = buffers.len() > 1 && self.num_threads.get() > 1;

        let frame_contexts = &self.frame_contexts;
        let simd_dispatch = &self.simd_dispatch;
        if parallel {
            // Pre-compute per-tile bounds + allocate per-tile scratch
            // (Yv12Buffer + cur_segment_ids + CoefCountsBank). Each
            // worker writes its tile's column range into the scratch
            // YV12 and the per-tile counts bank; the main thread
            // merges after the scope closes.
            let num_tiles = buffers.len();
            let mut per_tile_bounds: Vec<tile::TileBounds> = Vec::with_capacity(num_tiles);
            for tile_buf in &buffers {
                per_tile_bounds.push(tile_bounds(
                    &tail.tile_info,
                    mi_cols,
                    mi_rows,
                    tile_buf.row,
                    tile_buf.col,
                ));
            }

            // Per-tile slots: each slot owns the scratch yv12, the
            // worker-local seg-id map, the worker-local counts, and
            // an `Option<TileDecodeCtx>` filled by the worker. The
            // workers consume the contained owned values via
            // `Mutex` interior mutability so each tile's outputs land
            // in the right slot regardless of completion order.
            //
            // `Mutex<Option<...>>` cleanly handles the "one worker
            // takes the contents, drops the mutex, then signals
            // completion" handoff. After scope joins, the main
            // thread `.take()`s each slot.
            //
            // G5.2: the scratch `Yv12Buffer`s are sourced from the
            // decoder-owned [`Yv12ScratchPool`] rather than via a
            // fresh `clone_geometry()` per tile per frame. The pool
            // recycles buffers across frames; see the
            // [`Yv12ScratchPool`] type doc for the lifecycle.
            struct TileSlot {
                scratch: Mutex<Option<Yv12Buffer>>,
                seg_ids: Mutex<Option<Vec<u8>>>,
                counts: Mutex<Option<Box<rvp9_core::entropy::adapt::CoefCountsBank>>>,
                result: Mutex<Option<Result<TileDecodeCtx, tile::TileDecodeError>>>,
            }
            let scratch_buffers = self
                .tile_scratch_pool
                .take(num_tiles, &yv12)
                .expect("scratch Yv12Buffer allocation");
            let slots: Vec<TileSlot> = scratch_buffers
                .into_iter()
                .map(|scratch| TileSlot {
                    scratch: Mutex::new(Some(scratch)),
                    seg_ids: Mutex::new(Some(vec![0u8; mi_rows * mi_cols])),
                    counts: Mutex::new(Some(Box::new(
                        rvp9_core::entropy::adapt::CoefCountsBank::default(),
                    ))),
                    result: Mutex::new(None),
                })
                .collect();

            // Borrow shared immutables.
            let fc = &frame_contexts[fc_idx];
            let params_ref = &params;
            self.worker_pool.scope(|scope| {
                for (tile_idx, tile_buf) in buffers.iter().enumerate() {
                    let bounds = per_tile_bounds[tile_idx];
                    let tile_data: &[u8] = tile_buf.data;
                    let slot = &slots[tile_idx];
                    scope.submit(move || {
                        RVP9_G5_1_PARALLEL_JOBS
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let mut scratch = slot
                            .scratch
                            .lock()
                            .expect("scratch mutex")
                            .take()
                            .expect("scratch present at job start");
                        let mut seg_ids = slot
                            .seg_ids
                            .lock()
                            .expect("seg_ids mutex")
                            .take()
                            .expect("seg_ids present at job start");
                        let mut counts = slot
                            .counts
                            .lock()
                            .expect("counts mutex")
                            .take()
                            .expect("counts present at job start");
                        let res = decode_tile_intra(
                            tile_data,
                            bounds,
                            params_ref,
                            fc,
                            &mut scratch,
                            Some(counts.as_mut()),
                            &mut seg_ids,
                            simd_dispatch,
                        );
                        // Restore the owned state so the main thread
                        // can merge.
                        *slot.scratch.lock().expect("scratch mutex") = Some(scratch);
                        *slot.seg_ids.lock().expect("seg_ids mutex") = Some(seg_ids);
                        *slot.counts.lock().expect("counts mutex") = Some(counts);
                        *slot.result.lock().expect("result mutex") = Some(res);
                    });
                }
            });
            // Post-scope merge: in submission order, collect TileDecodeCtx,
            // merge per-tile counts into `self.coef_counts`, and
            // merge per-tile pixels (column-range) into `yv12`.
            //
            // G5.2: recover the scratch `Yv12Buffer`s as we walk the
            // slots and return them to `self.tile_scratch_pool` at
            // the end of the merge so the next frame can reuse them.
            let mut recovered_scratch: Vec<Yv12Buffer> = Vec::with_capacity(slots.len());
            let mut merge_err: Option<DecodeError> = None;
            for (tile_idx, slot) in slots.into_iter().enumerate() {
                let scratch = slot
                    .scratch
                    .into_inner()
                    .expect("scratch mutex")
                    .expect("scratch present after job");
                // Reclaim the scratch buffer up-front so that even if
                // the per-tile result is an error (and we early-return
                // below), the pool gets the buffer back via the
                // post-loop `put`.
                recovered_scratch.push(scratch);

                if merge_err.is_some() {
                    // A prior tile errored; keep collecting scratch
                    // buffers but stop merging.
                    continue;
                }

                let res = slot
                    .result
                    .into_inner()
                    .expect("result mutex")
                    .expect("worker stored result");
                let decode_ctx = match res {
                    Ok(c) => c,
                    Err(e) => {
                        merge_err = Some(DecodeError::TileDecode(e));
                        continue;
                    }
                };
                let bounds = per_tile_bounds[tile_idx];
                let seg_ids = slot
                    .seg_ids
                    .into_inner()
                    .expect("seg_ids mutex")
                    .expect("seg_ids present after job");
                let counts = slot
                    .counts
                    .into_inner()
                    .expect("counts mutex")
                    .expect("counts present after job");

                // 1. Merge counts into the decoder's running coef_counts.
                self.coef_counts.add(&counts);

                // 2. Merge pixel rectangle (rows x cols) from scratch
                //    -> yv12. G5.3: row range is per-tile so
                //    tile_rows > 1 tiles only stamp their own rect.
                let luma_row_start = bounds.mi_row_start * MI_BLOCK_PIXELS;
                let luma_row_end = bounds.mi_row_end * MI_BLOCK_PIXELS;
                let luma_col_start = bounds.mi_col_start * MI_BLOCK_PIXELS;
                let luma_col_end = bounds.mi_col_end * MI_BLOCK_PIXELS;
                let scratch_ref = recovered_scratch
                    .last()
                    .expect("just pushed scratch above");
                yv12.merge_tile_rect_from(
                    scratch_ref,
                    luma_row_start,
                    luma_row_end,
                    luma_col_start,
                    luma_col_end,
                );

                // 3. Merge seg_id rectangle — each tile only writes
                //    its own (mi_row_start..mi_row_end,
                //    mi_col_start..mi_col_end) cells. G5.3: restrict
                //    the row loop to the tile's row range so multi-
                //    tile-row dispatch doesn't trample sibling-row
                //    tiles' seg-id writes.
                let mc_start = bounds.mi_col_start;
                let mc_end = bounds.mi_col_end;
                for mi_row in bounds.mi_row_start..bounds.mi_row_end {
                    let row_base = mi_row * mi_cols;
                    cur_segment_ids[row_base + mc_start..row_base + mc_end]
                        .copy_from_slice(&seg_ids[row_base + mc_start..row_base + mc_end]);
                }

                tile_decode_ctxs.push((bounds, decode_ctx));
            }
            // Return the scratch buffers to the pool (always, even on
            // error — the buffers are owned by us and stale content
            // is safe per the pool's Re-zero safety invariant).
            self.tile_scratch_pool.put(recovered_scratch);
            if let Some(e) = merge_err {
                return Err(e);
            }
        } else {
            // Inline path (single-threaded knob OR ineligible
            // geometry). Identical to the pre-G5.1 first-cut shape.
            let coef_counts = &mut self.coef_counts;
            self.worker_pool.scope(|_scope| {
                for tile_buf in &buffers {
                    let bounds = tile_bounds(
                        &tail.tile_info,
                        mi_cols,
                        mi_rows,
                        tile_buf.row,
                        tile_buf.col,
                    );
                    let fc = &frame_contexts[fc_idx];
                    let decode_ctx = decode_tile_intra(
                        tile_buf.data,
                        bounds,
                        &params,
                        fc,
                        &mut yv12,
                        Some(&mut *coef_counts),
                        &mut cur_segment_ids,
                        simd_dispatch,
                    )
                    .map_err(DecodeError::TileDecode)?;
                    tile_decode_ctxs.push((bounds, decode_ctx));
                }
                Ok::<(), DecodeError>(())
            })?;
        }

        // Backward-adapt `coef_probs` per `vp9_adapt_coef_probs`
        // (`vp9_entropy.c:1084-1100`). Gate matches libvpx
        // `vp9_decodeframe.c:3046-3077`: backward adaptation runs only
        // when `refresh_frame_context` is set AND the frame is neither
        // error-resilient NOR frame-parallel. A *default* vpxenc encode
        // sets `frame_parallel_decoding_mode = 1`, so libvpx skips
        // adaptation and writes the *un-adapted* (forward-updated)
        // context back; we mirror that by leaving the in-place
        // forward-updated bank untouched when the gate is false. (Omitting
        // the frame-parallel term was the foreman decoder-robustness bug —
        // Wave M M1: the corpus all uses `frame_parallel == 0`, so it
        // never exercised the miss.) For keyframes the kernel runs with
        // the `_KEY` constants — `intra_only = true` in `coef_adapt_params`.
        //
        // D.9 — use `pre_fc` (snapshot taken before
        // `parse_compressed_partition`) as the `pre` baseline. libvpx's
        // `pre_fc = cm->frame_contexts[fc_idx]` carries the *prior*
        // adapt result, unaffected by this frame's compressed-header
        // diff-updates (see snapshot comment above).
        if tail.refresh_frame_context
            && !frame.header.error_resilient
            && !tail.frame_parallel_decoding_mode
        {
            let (count_sat, max_update_factor) = rvp9_core::entropy::adapt::coef_adapt_params(
                /* intra_only = */ true,
                self.last_frame_was_key,
            );
            let new_coef_probs = rvp9_core::entropy::adapt::adapt_coef_probs(
                &pre_fc.coef_probs,
                &self.coef_counts,
                count_sat,
                max_update_factor,
            );
            self.frame_contexts[fc_idx].coef_probs = new_coef_probs;
        }

        // 8. K2.c — loop-filter apply driver. Skip when filter_level
        //    is zero (LF is a frame-level no-op — see libvpx
        //    `vp9_loopfilter.c:1422-1427`).
        //
        //    Keyframes invoke libvpx's `vp9_setup_past_independence`
        //    (`vp9_entropymode.c:424-469`), which resets the LF deltas
        //    to the keyframe defaults `{INTRA:+1, LAST:0, GOLDEN:-1,
        //    ALTREF:-1}` with `mode_ref_delta_enabled = true` BEFORE
        //    `setup_loopfilter` parses the header. We mirror that
        //    here so the header's "no update" path inherits the
        //    correct baseline rather than the all-zeros startup state.
        self.lf_deltas = ResolvedLfDeltas {
            ref_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
            enabled: true,
        };
        self.merge_lf_params(&tail.loop_filter);
        let filter_level = tail.loop_filter.filter_level;
        // Stage 3 stretch M.13 — `RVP9_DISABLE_LF=1` skips the loop-filter
        // apply driver, exposing pre-LF reconstruction for cascade
        // analysis (dump_frame_diff vs vpxdec's post-LF YUV).
        let lf_disabled = std::env::var("RVP9_DISABLE_LF").is_ok();
        // Stage 3 Task E.2 — stitch per-tile mi grids into a
        // frame-wide grid for the LF mask builder. Mirrors libvpx
        // `cm->mi_grid_visible` which spans the entire frame
        // regardless of tile count. We always stitch (not just when
        // LF fires) so the bitstream-diff harness's L4 accessor
        // [`Self::last_frame_mi_grid`] sees a populated grid even
        // when `filter_level == 0` (Stage 4 design Strategy A).
        let tile_grids: Vec<(tile::TileBounds, &[Option<ModeInfo>])> = tile_decode_ctxs
            .iter()
            .map(|(b, c)| (*b, c.mi_grid()))
            .collect();
        let frame_mi_grid = build_frame_mi_grid(mi_rows, mi_cols, &tile_grids);
        if filter_level > 0 && !lf_disabled {
            apply_loop_filter_to_frame(
                &mut yv12,
                &frame_mi_grid,
                mi_cols,
                mi_rows,
                mi_cols,
                filter_level,
                tail.loop_filter.sharpness_level,
                self.lf_deltas,
                SegmentationLfView::from_segmentation(&resolved_seg),
            );
        }

        // 9. Extend the frame's borders so future inter frames using
        //    this keyframe as a reference can read MC halos past the
        //    visible region (libvpx `vpx_extend_frame_borders_c`,
        //    `yv12extend.c:169`).
        extend_frame_borders(&mut yv12);

        let frame_buf: FrameBuf = std::sync::Arc::new(yv12);

        // Note: ref_frame_sizes is already populated by
        // `parse_packet -> refresh_ref_frame_sizes`; no work here.

        // Refresh `ref_frame_map` per `refresh_mask` (keyframes pass
        // `0xFF`, intra-only pass `tail.refresh_frame_flags`). Mirrors
        // libvpx `vp9_decodeframe.c:2691` (KEY) and `:2730` (intra-only).
        //
        // G12 refactor (2026-05-21). Pre-G12 stored an `Arc` clone in
        // every selected ref slot AND mirror-wrote `frame_pool[i]`.
        // The mirror doubled `Arc::strong_count` per slot and would
        // starve `acquire_free` once wired up. G12 reverts to the
        // libvpx-faithful single-Arc-per-buffer model: park the
        // FrameBuf in the pool via `acquire_free`, then store the
        // pool slot index in every selected ref_frame_map slot.
        //
        // `acquire_free` errors only when every pool slot has an
        // outstanding external ref (refcount > 1). At decode time
        // the freshly-built `frame_buf` has refcount 1 (only this
        // local owns it), so a free slot is always available
        // unless the caller is holding 9+ live FrameBufs from
        // previous decodes — that's a caller bug. Surface it via
        // `DecodeError::FrameAlloc` (reusing the existing
        // alloc-failure variant rather than adding a new one).
        let pool_slot = self
            .frame_pool
            .acquire_free(frame_buf.clone())
            .map_err(|_| DecodeError::FrameAlloc)?;
        // `pool_slot` returned by `acquire_free` is always
        // `< DECODER_FRAME_BUFFERS == 9`, so casting to u8 is safe.
        let pool_slot_u8 = pool_slot as u8;
        debug_assert!(pool_slot < rvp9_core::frame::DECODER_FRAME_BUFFERS);
        // Use `write_ref_slot` so `frame_pool.ref_counts` stays in
        // sync — this is what reserves the slot against
        // `acquire_free` recycling while the ref-map still names it.
        for i in 0..self.ref_frame_map.len() {
            if (refresh_mask >> i) & 1 != 0 {
                self.write_ref_slot(i, pool_slot_u8);
            }
        }
        // Intra frames don't populate the MV grid — libvpx never reads
        // `cm->cur_frame->mvs` for intra frames in `vp9_read_mode_info`
        // (the `frame_is_intra_only` early-return at
        // `vp9_decodemv.c:822`). Clear every ref-slot MV grid so the
        // first inter frame after this intra finds no salvageable
        // collocated MVs and falls through to the default zero-init
        // (matching libvpx `use_prev_frame_mvs == false` for
        // `cm->last_frame_type == KEY_FRAME` or `last_intra_only == 1`).
        for slot in &mut self.mv_grids {
            *slot = None;
        }
        // D.22 — keyframe also clears `cm->prev_frame->mvs` from the
        // decoder's perspective. The subsequent `use_prev_frame_mvs`
        // gate trips on `last_frame_type == KEY_FRAME` (or
        // `last_intra_only == 1` for intra-only) anyway, but null-ing
        // the field keeps state hygiene matching libvpx (intra frames
        // produce no MVs to carry forward).
        self.prev_frame_mv_grid = None;
        self.prev_frame_mv_grid_dims = (0, 0);

        self.have_resync = false;
        // Mirror libvpx `cm->last_frame_type = cm->frame_type`
        // (`vp9_decodeframe.c:2645`). For intra-only frames the
        // `frame_type` field is `INTER_FRAME` (`cm->intra_only` is the
        // distinguishing flag), so `last_frame_was_key` is `false` for
        // both intra-only and inter — only KeyFrames set it to `true`.
        // This is the right behaviour for `coef_adapt_params`'s
        // `last_was_key` argument in the *next* frame.
        self.last_frame_was_key = !intra_only;
        self.last_frame_size = Some((frame.header.width, frame.header.height));
        // Mirror libvpx `cm->last_intra_only = cm->intra_only`
        // (`vp9_decodeframe.c:2646`). Bypasses the prev-frame-MV
        // salvage for the frame after an intra-only.
        self.last_intra_only = intra_only;
        self.last_show_frame = frame.header.show_frame;
        // D.10 — promote the keyframe's just-decoded seg map into
        // `prev_segment_ids` so the next inter frame's
        // `read_inter_segment_id` inherits from it (when
        // `update_map = false`). This mirrors libvpx's per-frame
        // `vp9_swap_current_and_last_seg_map` at
        // `vp9_decoder.c:479`, which fires after every (shown) decode
        // whose `seg.enabled == true`. The keyframe's
        // `vp9_setup_past_independence` at frame START already zeroed
        // both maps; we then decoded into `cur_segment_ids`, so the
        // swap-equivalent is just a move into `prev_segment_ids`.
        if std::env::var("RVP9_TRACE_SEG").is_ok() {
            let nz = cur_segment_ids.iter().filter(|&&v| v != 0).count();
            eprintln!(
                "[rvp9_seg] keyframe end: mi={}x{} non-zero seg cells = {} / {}",
                mi_rows,
                mi_cols,
                nz,
                cur_segment_ids.len()
            );
            for r in 0..4.min(mi_rows) {
                let mut row = String::new();
                for c in 0..16.min(mi_cols) {
                    row.push_str(&format!(" {}", cur_segment_ids[r * mi_cols + c]));
                }
                eprintln!("[rvp9_seg] keyframe row{r}:{}", row);
            }
        }
        self.prev_segment_ids = cur_segment_ids;
        self.prev_segment_ids_dims = (mi_rows, mi_cols);
        // Strategy A (Stage 4 design §9 commit 4) — publish the
        // stitched frame-wide mi grid for the bitstream-diff harness.
        self.last_frame_mi_grid = frame_mi_grid;
        self.last_frame_mi_dims = (mi_rows as u32, mi_cols as u32);
        Ok(frame_buf)
    }

    /// Decode an inter frame end-to-end, returning the freshly-
    /// allocated [`FrameBuf`] that holds the reconstructed pixels.
    ///
    /// Mirrors libvpx's inter-frame decode driver:
    /// `read_compressed_header` → `decode_tiles` (`vp9_decodeframe.c`)
    /// for the single-tile path. Multi-tile vectors surface as
    /// [`DecodeError::TileDecode`].
    ///
    /// ## Stage D.3 status
    ///
    /// - **D.3.a (this commit)** wires the partition+tile walker, the
    ///   per-block ModeInfo reader, and (via D.2) the per-inter-block
    ///   reconstruction primitive. Intra-in-inter blocks consume their
    ///   bitstream bits but skip pixel reconstruction (deferred to
    ///   D.4). Lossless is handled by the WHT branch (G2). Any frame
    ///   referencing a `ref_frame_map` slot whose entry is `None`
    ///   surfaces as [`DecodeError::EmptyInterRef`] `{ slot }` (G3 —
    ///   the loop reports the slot index named by
    ///   `ref_frame_indices`, not the loop iteration index).
    /// - **D.3.b** rewires the per-block refs from "single-ref LAST
    ///   only" to the actual decoded `ref_frames` pair.
    /// - **D.3.c** wires `extend_frame_borders` + `refresh_frame_flags`
    ///   handling.
    fn decode_inter_frame(&mut self, frame: &ParsedFrame<'_>) -> Result<FrameBuf, DecodeError> {
        let tail = frame
            .header
            .tail
            .as_ref()
            .ok_or(DecodeError::WrongFrameKind)?;

        let (
            ref_frame_indices,
            ref_frame_sign_biases,
            interp_filter,
            allow_high_precision_mv,
            refresh_frame_flags,
        ) = match frame.header.kind {
            FrameKind::InterFrame {
                ref_frame_indices,
                ref_frame_sign_biases,
                interp_filter,
                allow_high_precision_mv,
                refresh_frame_flags,
                ..
            } => (
                ref_frame_indices,
                ref_frame_sign_biases,
                interp_filter,
                allow_high_precision_mv,
                refresh_frame_flags,
            ),
            _ => return Err(DecodeError::WrongFrameKind),
        };

        if std::env::var("RVP9_TRACE_INTER_MV").is_ok() {
            eprintln!(
                "[rvp9_inter_frame] indices={:?} sign_biases={:?} refresh=0x{:02x} last_key={} last_size={:?}",
                ref_frame_indices,
                ref_frame_sign_biases,
                refresh_frame_flags,
                self.last_frame_was_key,
                self.last_frame_size,
            );
            for i in 0..rvp9_core::bitstream::REF_FRAMES {
                let row0: Option<Vec<u8>> = self.ref_frame_buf(i).map(|b| {
                    let yp = b.y_plane();
                    let stride = yp.stride();
                    yp.data()[..stride.min(8)].to_vec()
                });
                eprintln!("  ref_frame_map[{i}] row0={:?}", row0);
            }
        }

        // G2 (decoder-hardening V6 §G2): lossless inter is wired
        // through `inverse_transform_add_dct`'s WHT branch (mirrors the
        // keyframe gating removed earlier in this commit). Inter blocks
        // resolve to `TxType::DctDct`, and at TX_4X4 the dispatch picks
        // `iwht4x4_{16,1}_add` whenever `inputs.lossless == true`.

        // 1. Resolve the three logical inter refs (LAST / GOLDEN /
        //    ALTREF) into actual FrameBuf handles via ref_frame_map.
        //    Empty slots surface as `EmptyInterRef` (G3 — V6 §G3):
        //    libvpx treats this as bitstream-corrupt, and in practice
        //    every real stream populates these via the prior keyframe.
        //    The sibling `EmptySlot` variant continues to cover the
        //    `decode_show_existing_frame` path.
        let mut refs: [Option<FrameBuf>; 3] = [None, None, None];
        for (i, &ref_idx) in ref_frame_indices.iter().enumerate() {
            let slot_idx = ref_idx as usize;
            // G12 refactor: `ref_frame_map[slot_idx]` is now an
            // `Option<u8>` pool slot index; resolve through the
            // `ref_frame_buf` helper that walks both indirections
            // (ref_frame_map → frame_pool). `None` from either step
            // (slot_idx out of range OR empty pool slot) is reported
            // identically as `EmptyInterRef` — libvpx treats both as
            // bitstream-corrupt.
            match self.ref_frame_buf(slot_idx) {
                Some(b) => refs[i] = Some(b),
                None => {
                    return Err(DecodeError::EmptyInterRef {
                        slot: slot_idx as u8,
                    });
                }
            }
        }

        // 2. Compute compound_allowed (libvpx
        //    `vp9_compound_reference_allowed`): true iff any two of
        //    the three sign biases differ.
        let compound_allowed = ref_frame_sign_biases[1] != ref_frame_sign_biases[0]
            || ref_frame_sign_biases[2] != ref_frame_sign_biases[0];

        // D.9 — snapshot the bank *before* compressed-header diff-updates
        // land, so the backward-adapt block below can use it as `pre_fc`
        // (libvpx `vp9_entropy.c:1058` / `vp9_entropymode.c:343-407`,
        // where `pre_fc = cm->frame_contexts[cm->frame_context_idx]`).
        // Our [`parse_compressed_partition_inter`] mutates the bank in
        // place — without the snapshot the adapt pass would chain-merge
        // from the bank's *post-diff* value, accumulating drift across
        // inter frames. This was the 32×32 "4 pass then cascade"
        // pattern: by frame 5 the probability tables had drifted enough
        // that the parent-mode bool-decode took a wrong branch, putting
        // the cursor out of sync and producing a wholly-wrong block at
        // mi=(1, 3).
        let fc_idx = tail.frame_context_idx as usize;
        let pre_fc = self.frame_contexts[fc_idx].clone();

        // 3. Parse compressed header into the selected FC bank.
        let compressed_result = self.parse_compressed_partition_inter(
            frame,
            compound_allowed,
            interp_filter,
            allow_high_precision_mv,
        )?;

        // D.16 — env-gated dump of FrameContext state immediately after
        // compressed-header parse, mirroring libvpx's RVP9_TRACE_FC
        // instrumentation in `vp9_decodeframe.c::read_compressed_header`.
        // Diff line-by-line vs the libvpx trace to localize any
        // bitstream-interpretation divergence at the per-frame FC
        // diff-update layer (separate from per-block decode bugs).
        if std::env::var("RVP9_TRACE_FC").ok().as_deref() == Some("1") {
            let fc = &self.frame_contexts[fc_idx];
            eprintln!("RVP9_TRACE_FC compressed_header_done kf=0 intra_only=0");
            eprintln!(
                "RVP9_TRACE_FC tx_mode={} ref_mode={}",
                compressed_result.tx_mode as u8, compressed_result.reference_mode as u8
            );
            eprintln!(
                "RVP9_TRACE_FC skip_probs=[{},{},{}]",
                fc.skip_probs[0], fc.skip_probs[1], fc.skip_probs[2]
            );
            eprintln!(
                "RVP9_TRACE_FC intra_inter_prob=[{},{},{},{}]",
                fc.intra_inter_prob[0],
                fc.intra_inter_prob[1],
                fc.intra_inter_prob[2],
                fc.intra_inter_prob[3]
            );
            eprintln!(
                "RVP9_TRACE_FC comp_inter_prob=[{},{},{},{},{}]",
                fc.comp_inter_prob[0],
                fc.comp_inter_prob[1],
                fc.comp_inter_prob[2],
                fc.comp_inter_prob[3],
                fc.comp_inter_prob[4]
            );
            for i in 0..5 {
                eprintln!(
                    "RVP9_TRACE_FC single_ref_prob[{i}]=[{},{}]",
                    fc.single_ref_prob[i][0], fc.single_ref_prob[i][1]
                );
            }
            eprintln!(
                "RVP9_TRACE_FC comp_ref_prob=[{},{},{},{},{}]",
                fc.comp_ref_prob[0],
                fc.comp_ref_prob[1],
                fc.comp_ref_prob[2],
                fc.comp_ref_prob[3],
                fc.comp_ref_prob[4]
            );
            for j in 0..7 {
                eprintln!(
                    "RVP9_TRACE_FC inter_mode_probs[{j}]=[{},{},{}]",
                    fc.inter_mode_probs[j][0], fc.inter_mode_probs[j][1], fc.inter_mode_probs[j][2]
                );
            }
            eprintln!(
                "RVP9_TRACE_FC nmvc.joints=[{},{},{}]",
                fc.nmvc.joints[0], fc.nmvc.joints[1], fc.nmvc.joints[2]
            );
            for i in 0..2 {
                let c = &fc.nmvc.comps[i];
                eprintln!(
                    "RVP9_TRACE_FC nmvc.comp[{i}].sign={} classes=[{},{},{},{},{},{},{},{},{},{}]",
                    c.sign,
                    c.classes[0],
                    c.classes[1],
                    c.classes[2],
                    c.classes[3],
                    c.classes[4],
                    c.classes[5],
                    c.classes[6],
                    c.classes[7],
                    c.classes[8],
                    c.classes[9]
                );
                eprintln!(
                    "RVP9_TRACE_FC nmvc.comp[{i}].class0=[{}] bits=[{},{},{},{},{},{},{},{},{},{}]",
                    c.class0[0],
                    c.bits[0],
                    c.bits[1],
                    c.bits[2],
                    c.bits[3],
                    c.bits[4],
                    c.bits[5],
                    c.bits[6],
                    c.bits[7],
                    c.bits[8],
                    c.bits[9]
                );
            }
        }

        // 4. Slice tile data (everything past the compressed header).
        let header_end = frame.header_end_byte;
        let fps = tail.first_partition_size as usize;
        let tile_data_start = header_end + fps;
        let total = frame.full_frame_data.len();
        if tile_data_start > total {
            return Err(DecodeError::TruncatedCompressedHeader {
                declared: tile_data_start,
                available: total,
            });
        }
        let tile_section = &frame.full_frame_data[tile_data_start..];

        let mi_cols = ((frame.header.width as usize) + 7) >> 3;
        let mi_rows = ((frame.header.height as usize) + 7) >> 3;
        let tile_cols = 1u32 << tail.tile_info.log2_tile_cols;
        let tile_rows = 1u32 << tail.tile_info.log2_tile_rows;
        // Stage 3 Task E.2 — wire up single-threaded multi-tile decode
        // for inter frames. Per-tile state (BoolReader / TileDecodeCtx
        // / TileCoefCtx) resets at every tile boundary; frame-level
        // accumulators (counts / cur_mv_grid / cur_segment_ids / yv12)
        // are shared across all tiles. Mirrors libvpx `decode_tiles`
        // (`vp9_decodeframe.c:2200-2240`).
        let buffers = parse_tile_buffers(tile_section, tile_cols, tile_rows)
            .map_err(DecodeError::Bitstream)?;
        if buffers.is_empty() {
            return Err(DecodeError::Bitstream(BitstreamError::Truncated {
                needed: 1,
                available: 0,
            }));
        }

        // 5. Allocate destination FrameBuf. Chroma format flows from
        //    the parsed header — see the matching keyframe / intra-only
        //    site for the Profile 1 gating context.
        let mut yv12 = Yv12Buffer::alloc(
            frame.header.width,
            frame.header.height,
            frame.header.chroma,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .map_err(|_| DecodeError::FrameAlloc)?;

        // D.24: pass the raw (crop) width / height to ScaleFactors, not
        // the aligned 8-multiple. Libvpx feeds `cm->width / cm->height`
        // (the un-aligned values) into `vp9_setup_scale_factors_for_frame`
        // at `vp9_decodeframe.c:2928-2935`. For frames whose dims are
        // already 8-multiples (most vectors) the two coincide; for resize
        // vectors with dims like 282×173 they don't, and using the
        // aligned dim produces a wrong Q14 ratio.
        let cur_y_width = frame.header.width;
        let cur_y_height = frame.header.height;

        // 6. Per-frame inter params.
        let sign_bias = sign_bias_from_header(&ref_frame_sign_biases);
        let (comp_fixed_ref, comp_var_ref) = setup_compound_reference_mode(&sign_bias);
        let inter_params = InterModeReadParams {
            allow_high_precision_mv,
            interp_filter,
            reference_mode: compressed_result.reference_mode,
            comp_fixed_ref,
            comp_var_ref,
            sign_bias,
        };

        // 7. D.10 — resolve per-frame segmentation against the running
        // state (inherits `feature_data` / `abs_delta` when the parsed
        // value's `update_data == false`).
        let resolved_seg =
            self.resolve_segmentation(&tail.segmentation, /* is_keyframe = */ false);

        // Per-segment alt-Q from the RESOLVED feature data.
        let mut seg_alt_q = [None; rvp9_core::loopfilter::MAX_SEGMENTS];
        if resolved_seg.enabled {
            for (i, s) in resolved_seg.feature_data.iter().enumerate() {
                seg_alt_q[i] = s[0];
            }
        }

        let base = FrameDecodeParams {
            mi_cols,
            mi_rows,
            tx_mode: compressed_result.tx_mode,
            segmentation_update_map: resolved_seg.enabled && resolved_seg.update_map,
            segmentation_enabled: resolved_seg.enabled,
            segmentation_temporal_update: resolved_seg.temporal_update,
            seg_tree_probs: resolved_seg.tree_probs,
            seg_pred_probs: resolved_seg.pred_probs,
            lossless: tail.quantization.is_lossless(),
            base_qindex: i32::from(tail.quantization.base_qindex),
            y_dc_delta_q: i32::from(tail.quantization.y_dc_delta_q),
            uv_dc_delta_q: i32::from(tail.quantization.uv_dc_delta_q),
            uv_ac_delta_q: i32::from(tail.quantization.uv_ac_delta_q),
            seg_alt_q,
            seg_abs_delta: resolved_seg.abs_delta,
        };

        // Borrow refs as &Yv12Buffer for InterFrameDecodeParams. The
        // FrameBufs we cloned above keep the underlying allocations
        // alive for the lifetime of this scope.
        let refs_borrowed: [Option<&Yv12Buffer>; 3] =
            [refs[0].as_deref(), refs[1].as_deref(), refs[2].as_deref()];
        let params = InterFrameDecodeParams {
            base,
            cur_y_width,
            cur_y_height,
            refs: refs_borrowed,
            inter_params,
        };

        // Zero MV + coef + mode counts at start of frame, then decode
        // the tile. After the tile completes the counts hold this
        // frame's NEWMV component tallies + per-(tx, plane, ref)
        // coef-token tallies + per-block mode/partition/skip/inter/ref/
        // tx tallies, which feed `adapt_mv_probs` / `adapt_coef_probs`
        // / `adapt_mode_probs` below.
        self.mv_counts = rvp9_core::entropy::mv_probs::NmvCounts::default();
        self.coef_counts = rvp9_core::entropy::adapt::CoefCountsBank::default();
        self.mode_counts = rvp9_core::entropy::adapt::FrameCounts::default();

        // D.7 — compute `use_prev_frame_mvs` per libvpx
        // `vp9_decodeframe.c:2983-2986`:
        //   !error_resilient && width == last_width && height ==
        //   last_height && !last_intra_only && last_show_frame &&
        //   last_frame_type != KEY_FRAME.
        //
        // If the gate fires, look up the LAST_FRAME ref slot's MV grid
        // and pass it to the tile decoder so `find_mv_refs` can salvage
        // collocated MV candidates. Otherwise pass `None` (matching
        // pre-D.7 behaviour).
        let use_prev_frame_mvs = !frame.header.error_resilient
            && self
                .last_frame_size
                .map(|(w, h)| w == frame.header.width && h == frame.header.height)
                .unwrap_or(false)
            && !self.last_intra_only
            && self.last_show_frame
            && !self.last_frame_was_key;
        // D.22 — `cm->prev_frame->mvs` mirror: pull the MV grid
        // captured at the *end of the previous decoded frame*
        // (regardless of which `ref_frame_map` slot was refreshed).
        // This matches libvpx's `cm->prev_frame = cm->cur_frame` swap
        // (`vp9_decoder.c:478`) — fires after every non-show-existing
        // decode, so prev_frame always points at the truly-most-recent
        // frame's buffer. The previous implementation pulled from
        // `mv_grids[ref_frame_indices[0]]` (the LAST_FRAME slot's
        // grid), which broke for altref frames whose LAST_FRAME index
        // points at a slot holding an older frame's grid.
        let prev_mv_grid: Option<&[rvp9_core::inter::mv_ref::PrevFrameMvRef]> =
            if use_prev_frame_mvs {
                self.prev_frame_mv_grid.as_ref().map(|g| g.as_slice())
            } else {
                None
            };
        // Verify the prev grid is correctly sized for this frame.
        let prev_mv_grid = prev_mv_grid.filter(|g| g.len() >= mi_rows * mi_cols);
        // Allocate fresh current-frame MV grid (zero-init = all
        // `(NoRefFrame, NoRefFrame)` entries with zero MVs).
        let mut cur_mv_grid =
            vec![rvp9_core::inter::mv_ref::PrevFrameMvRef::default(); mi_rows * mi_cols];

        // D.10 — populate `prev_segment_ids` view for segment-id
        // inheritance and allocate the current frame's seg-id slab.
        //
        // Libvpx's `read_inter_segment_id` reads `cm->last_frame_seg_map`;
        // when that pointer is NULL (first frame after past-independence
        // reset, or right after allocation) every cell behaves as zero
        // (`predicted_segment_id = cm->last_frame_seg_map ? ... : 0` at
        // `vp9_decodemv.c:156-159`). We model NULL as an empty slice.
        //
        // Resize check: if the prev seg map's dims don't match this
        // frame's mi-dims, treat it as missing — libvpx reallocates +
        // zero-fills `last_frame_seg_map` whenever mi_rows / mi_cols
        // change (`vp9_alloccommon.c:vp9_alloc_context_buffers`).
        let prev_seg_ids_slice: &[u8] = if self.prev_segment_ids_dims == (mi_rows, mi_cols)
            && self.prev_segment_ids.len() == mi_rows * mi_cols
        {
            &self.prev_segment_ids
        } else {
            &[]
        };
        let mut cur_segment_ids = vec![0u8; mi_rows * mi_cols];

        // Stage 3 Task E.2 — iterate `(tile_row, tile_col)` in raster
        // order. Each tile is decoded sequentially with its own fresh
        // `BoolReader` / `TileDecodeCtx` / `TileCoefCtx` (allocated
        // inside `decode_tile_inter`). Shared frame-level state
        // (`yv12`, `cur_mv_grid`, `cur_segment_ids`, counts) accumulates
        // across all tiles. Per-tile mi grids are collected here so
        // we can stitch them into a frame-wide mi grid for the LF
        // driver below.
        let mut tile_decode_ctxs: Vec<(tile::TileBounds, TileDecodeCtx)> =
            Vec::with_capacity(buffers.len());
        // G5.1 — parallel tile dispatch. See the matching block in
        // `decode_intra_frame_inner` for the rationale + eligibility
        // gate. The inter path additionally fans out the per-tile
        // mv-counts / mode-counts / cur_mv_grid / cur_segment_ids
        // worker-local buffers.
        // G5.3: gate is "multi-tile + multi-thread"; the
        // tile_rows == 0 restriction of G5.1 is gone because
        // `merge_tile_rect_from` handles the 2D grid case.
        let parallel = buffers.len() > 1 && self.num_threads.get() > 1;

        let frame_contexts = &self.frame_contexts;
        let simd_dispatch = &self.simd_dispatch;
        if parallel {
            let num_tiles = buffers.len();
            let mut per_tile_bounds: Vec<tile::TileBounds> = Vec::with_capacity(num_tiles);
            for tile_buf in &buffers {
                per_tile_bounds.push(tile_bounds(
                    &tail.tile_info,
                    mi_cols,
                    mi_rows,
                    tile_buf.row,
                    tile_buf.col,
                ));
            }

            struct InterTileSlot {
                scratch: Mutex<Option<Yv12Buffer>>,
                seg_ids: Mutex<Option<Vec<u8>>>,
                mv_grid: Mutex<Option<Vec<rvp9_core::inter::mv_ref::PrevFrameMvRef>>>,
                mv_counts: Mutex<Option<Box<rvp9_core::entropy::mv_probs::NmvCounts>>>,
                coef_counts: Mutex<Option<Box<rvp9_core::entropy::adapt::CoefCountsBank>>>,
                mode_counts: Mutex<Option<Box<rvp9_core::entropy::adapt::FrameCounts>>>,
                result: Mutex<Option<Result<TileDecodeCtx, tile::TileDecodeError>>>,
            }
            // G5.2: source the per-tile scratch buffers from the
            // decoder-owned [`Yv12ScratchPool`] (recycled across
            // frames) rather than per-tile `clone_geometry()`.
            let scratch_buffers = self
                .tile_scratch_pool
                .take(num_tiles, &yv12)
                .expect("scratch Yv12Buffer allocation");
            let slots: Vec<InterTileSlot> = scratch_buffers
                .into_iter()
                .map(|scratch| InterTileSlot {
                    scratch: Mutex::new(Some(scratch)),
                    seg_ids: Mutex::new(Some(vec![0u8; mi_rows * mi_cols])),
                    mv_grid: Mutex::new(Some(vec![
                        rvp9_core::inter::mv_ref::PrevFrameMvRef::default();
                        mi_rows * mi_cols
                    ])),
                    mv_counts: Mutex::new(Some(Box::new(
                        rvp9_core::entropy::mv_probs::NmvCounts::default(),
                    ))),
                    coef_counts: Mutex::new(Some(Box::new(
                        rvp9_core::entropy::adapt::CoefCountsBank::default(),
                    ))),
                    mode_counts: Mutex::new(Some(Box::new(
                        rvp9_core::entropy::adapt::FrameCounts::default(),
                    ))),
                    result: Mutex::new(None),
                })
                .collect();

            let fc = &frame_contexts[fc_idx];
            let params_ref = &params;
            let prev_mv_grid_ref = prev_mv_grid;
            let prev_seg_ref = prev_seg_ids_slice;
            self.worker_pool.scope(|scope| {
                for (tile_idx, tile_buf) in buffers.iter().enumerate() {
                    let bounds = per_tile_bounds[tile_idx];
                    let tile_data: &[u8] = tile_buf.data;
                    let slot = &slots[tile_idx];
                    scope.submit(move || {
                        RVP9_G5_1_PARALLEL_JOBS
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let mut scratch = slot
                            .scratch
                            .lock()
                            .expect("scratch mutex")
                            .take()
                            .expect("scratch present at job start");
                        let mut seg_ids = slot
                            .seg_ids
                            .lock()
                            .expect("seg_ids mutex")
                            .take()
                            .expect("seg_ids present at job start");
                        let mut mv_grid = slot
                            .mv_grid
                            .lock()
                            .expect("mv_grid mutex")
                            .take()
                            .expect("mv_grid present at job start");
                        let mut mv_counts = slot
                            .mv_counts
                            .lock()
                            .expect("mv_counts mutex")
                            .take()
                            .expect("mv_counts present at job start");
                        let mut coef_counts = slot
                            .coef_counts
                            .lock()
                            .expect("coef_counts mutex")
                            .take()
                            .expect("coef_counts present at job start");
                        let mut mode_counts = slot
                            .mode_counts
                            .lock()
                            .expect("mode_counts mutex")
                            .take()
                            .expect("mode_counts present at job start");
                        let res = decode_tile_inter(
                            tile_data,
                            bounds,
                            params_ref,
                            fc,
                            &mut scratch,
                            Some(mv_counts.as_mut()),
                            Some(coef_counts.as_mut()),
                            Some(mode_counts.as_mut()),
                            prev_mv_grid_ref,
                            &mut mv_grid,
                            prev_seg_ref,
                            &mut seg_ids,
                            simd_dispatch,
                        );
                        // Restore owned state to slot.
                        *slot.scratch.lock().expect("scratch mutex") = Some(scratch);
                        *slot.seg_ids.lock().expect("seg_ids mutex") = Some(seg_ids);
                        *slot.mv_grid.lock().expect("mv_grid mutex") = Some(mv_grid);
                        *slot.mv_counts.lock().expect("mv_counts mutex") = Some(mv_counts);
                        *slot.coef_counts.lock().expect("coef_counts mutex") = Some(coef_counts);
                        *slot.mode_counts.lock().expect("mode_counts mutex") = Some(mode_counts);
                        *slot.result.lock().expect("result mutex") = Some(res);
                    });
                }
            });
            // Post-scope merge in submission order.
            //
            // G5.2: reclaim each tile's scratch `Yv12Buffer` into
            // `recovered_scratch` and return them to
            // `self.tile_scratch_pool` after the merge so the next
            // frame can reuse them (rather than re-allocating).
            let mut recovered_scratch: Vec<Yv12Buffer> = Vec::with_capacity(slots.len());
            let mut merge_err: Option<DecodeError> = None;
            for (tile_idx, slot) in slots.into_iter().enumerate() {
                let scratch = slot
                    .scratch
                    .into_inner()
                    .expect("scratch mutex")
                    .expect("scratch present after job");
                recovered_scratch.push(scratch);
                if merge_err.is_some() {
                    continue;
                }

                let res = slot
                    .result
                    .into_inner()
                    .expect("result mutex")
                    .expect("worker stored result");
                let decode_ctx = match res {
                    Ok(c) => c,
                    Err(e) => {
                        merge_err = Some(DecodeError::TileDecode(e));
                        continue;
                    }
                };
                let bounds = per_tile_bounds[tile_idx];
                let seg_ids = slot
                    .seg_ids
                    .into_inner()
                    .expect("seg_ids mutex")
                    .expect("seg_ids present after job");
                let mv_grid = slot
                    .mv_grid
                    .into_inner()
                    .expect("mv_grid mutex")
                    .expect("mv_grid present after job");
                let mv_counts = slot
                    .mv_counts
                    .into_inner()
                    .expect("mv_counts mutex")
                    .expect("mv_counts present after job");
                let coef_counts = slot
                    .coef_counts
                    .into_inner()
                    .expect("coef_counts mutex")
                    .expect("coef_counts present after job");
                let mode_counts = slot
                    .mode_counts
                    .into_inner()
                    .expect("mode_counts mutex")
                    .expect("mode_counts present after job");

                // Merge counts (additive).
                self.mv_counts.add(&mv_counts);
                self.coef_counts.add(&coef_counts);
                self.mode_counts.add(&mode_counts);

                // Merge pixel rectangle (rows x cols) into yv12.
                // G5.3: row range is per-tile so tile_rows > 1 tiles
                // each stamp only their own rectangle.
                let luma_row_start = bounds.mi_row_start * MI_BLOCK_PIXELS;
                let luma_row_end = bounds.mi_row_end * MI_BLOCK_PIXELS;
                let luma_col_start = bounds.mi_col_start * MI_BLOCK_PIXELS;
                let luma_col_end = bounds.mi_col_end * MI_BLOCK_PIXELS;
                let scratch_ref = recovered_scratch
                    .last()
                    .expect("just pushed scratch above");
                yv12.merge_tile_rect_from(
                    scratch_ref,
                    luma_row_start,
                    luma_row_end,
                    luma_col_start,
                    luma_col_end,
                );

                // Merge seg_ids + mv_grid rectangles. G5.3: restrict
                // the row loop to the tile's row range to support
                // tile_rows > 1 without sibling-row tiles clobbering
                // each other.
                let mc_start = bounds.mi_col_start;
                let mc_end = bounds.mi_col_end;
                for mi_row in bounds.mi_row_start..bounds.mi_row_end {
                    let row_base = mi_row * mi_cols;
                    cur_segment_ids[row_base + mc_start..row_base + mc_end]
                        .copy_from_slice(&seg_ids[row_base + mc_start..row_base + mc_end]);
                    cur_mv_grid[row_base + mc_start..row_base + mc_end]
                        .copy_from_slice(&mv_grid[row_base + mc_start..row_base + mc_end]);
                }

                tile_decode_ctxs.push((bounds, decode_ctx));
            }
            // G5.2: return scratch buffers to the pool (always, even
            // on error — buffers are owned by us and stale content
            // is safe per the pool's Re-zero safety invariant).
            self.tile_scratch_pool.put(recovered_scratch);
            if let Some(e) = merge_err {
                return Err(e);
            }
        } else {
            let mv_counts = &mut self.mv_counts;
            let coef_counts = &mut self.coef_counts;
            let mode_counts = &mut self.mode_counts;
            self.worker_pool.scope(|_scope| {
                for tile_buf in &buffers {
                    let bounds = tile_bounds(
                        &tail.tile_info,
                        mi_cols,
                        mi_rows,
                        tile_buf.row,
                        tile_buf.col,
                    );
                    let fc = &frame_contexts[fc_idx];
                    let decode_ctx = decode_tile_inter(
                        tile_buf.data,
                        bounds,
                        &params,
                        fc,
                        &mut yv12,
                        Some(&mut *mv_counts),
                        Some(&mut *coef_counts),
                        Some(&mut *mode_counts),
                        prev_mv_grid,
                        &mut cur_mv_grid,
                        prev_seg_ids_slice,
                        &mut cur_segment_ids,
                        simd_dispatch,
                    )
                    .map_err(DecodeError::TileDecode)?;
                    tile_decode_ctxs.push((bounds, decode_ctx));
                }
                Ok::<(), DecodeError>(())
            })?;
        }

        // Backward-adapt the per-frame probability banks per
        // `vp9_adapt_*_probs` (`vp9_decodeframe.c:3046-3077`). Same gate
        // for all kernels: `refresh_frame_context && !error_resilient &&
        // !frame_parallel_decoding_mode`. When the gate is false (e.g. a
        // default vpxenc encode with `frame_parallel == 1`) the in-place
        // forward-updated bank is left as-is — matching libvpx, which
        // writes the un-adapted context back. `adapt_coef_probs` fires
        // here on inter frames (the keyframe path runs the kernel
        // separately with the `_KEY` constants); `adapt_mode_probs` /
        // `adapt_mv_probs` only fire on non-intra-only frames.
        //
        // D.9 — use `pre_fc` (snapshot taken before
        // `parse_compressed_partition_inter`) as the `pre` baseline.
        // libvpx's adapt kernels read `pre_fc = cm->frame_contexts
        // [frame_context_idx]` — the bank value at frame *start*,
        // *unaffected* by the compressed-header diff-updates that
        // `read_compressed_header` applies to its working copy `cm->fc`.
        // Because [`parse_compressed_partition_inter`] mutates the bank
        // in place, reading `&self.frame_contexts[fc_idx]` here would
        // give the bank *plus* this frame's diff updates — drifting
        // the chained-merge baseline away from libvpx each frame.
        if tail.refresh_frame_context
            && !frame.header.error_resilient
            && !tail.frame_parallel_decoding_mode
        {
            // mode_probs — libvpx `vp9_adapt_mode_probs`. Returns a
            // freshly built FrameContext with the mode / partition /
            // skip / tx / intra_inter / comp_inter / single_ref /
            // comp_ref / inter_mode / y_mode / uv_mode /
            // switchable_interp fields folded; `coef_probs` and `nmvc`
            // are copied through unchanged from `pre_fc` (see
            // `adapt_mode_probs` docs) — we re-install the coef / mv
            // adapts on top of this baseline below.
            let interp_switchable = matches!(
                params.inter_params.interp_filter,
                rvp9_core::bitstream::InterpFilter::Switchable
            );
            let tx_mode_select = matches!(
                params.base.tx_mode,
                rvp9_core::transforms::TxMode::TxModeSelect
            );
            let new_fc = rvp9_core::entropy::adapt::adapt_mode_probs(
                &pre_fc,
                &self.mode_counts,
                interp_switchable,
                tx_mode_select,
            );
            self.frame_contexts[fc_idx] = new_fc;

            // coef_probs — inter-frame adapt (`intra_only = false`).
            // `coef_adapt_params` picks AFTER_KEY constants on the first
            // inter frame after a keyframe, default constants otherwise.
            let (count_sat, max_update_factor) = rvp9_core::entropy::adapt::coef_adapt_params(
                /* intra_only = */ false,
                self.last_frame_was_key,
            );
            let new_coef_probs = rvp9_core::entropy::adapt::adapt_coef_probs(
                &pre_fc.coef_probs,
                &self.coef_counts,
                count_sat,
                max_update_factor,
            );
            self.frame_contexts[fc_idx].coef_probs = new_coef_probs;

            // nmvc — libvpx `vp9_adapt_mv_probs`. `allow_high_precision_mv`
            // gates the class0_hp / hp slots.
            let new_nmvc = rvp9_core::entropy::adapt::adapt_mv_probs(
                &pre_fc.nmvc,
                &self.mv_counts,
                allow_high_precision_mv,
            );
            self.frame_contexts[fc_idx].nmvc = new_nmvc;
        }

        // D.8 — loop-filter apply (inter path). Mirrors the keyframe
        // wireup at the top of `decode_keyframe`: build per-SB filter
        // masks via `setup_mask` + `adjust_mask`, then dispatch
        // `loop_filter_frame`. libvpx applies the LF for inter frames
        // through the same `vp9_loopfilter.c:vp9_loop_filter_frame`
        // entry point as keyframes (`vp9_decodeframe.c:2867`).
        //
        // Without this wireup the inter pre-LF reconstruction was
        // being MD5'd against libvpx's post-LF dump — a per-frame +5/-5
        // edge-smoothing diff cascading across the whole frame. Same
        // pattern as M.9's "filter_level > 0 keyframes fail" finding,
        // here for the inter path.
        self.merge_lf_params(&tail.loop_filter);
        let filter_level = tail.loop_filter.filter_level;
        let lf_disabled = std::env::var("RVP9_DISABLE_LF").is_ok();
        // Stage 3 Task E.2 — stitch per-tile mi grids into a
        // frame-wide grid for the LF mask builder. Mirrors libvpx
        // `cm->mi_grid_visible` which spans the entire frame
        // regardless of tile count. Always stitched (Stage 4 design
        // Strategy A) so the bitstream-diff harness's L4 accessor
        // [`Self::last_frame_mi_grid`] is populated for every
        // non-`ShowExistingFrame` frame.
        let tile_grids: Vec<(tile::TileBounds, &[Option<ModeInfo>])> = tile_decode_ctxs
            .iter()
            .map(|(b, c)| (*b, c.mi_grid()))
            .collect();
        let frame_mi_grid = build_frame_mi_grid(mi_rows, mi_cols, &tile_grids);
        if filter_level > 0 && !lf_disabled {
            apply_loop_filter_to_frame(
                &mut yv12,
                &frame_mi_grid,
                mi_cols,
                mi_rows,
                mi_cols,
                filter_level,
                tail.loop_filter.sharpness_level,
                self.lf_deltas,
                SegmentationLfView::from_segmentation(&resolved_seg),
            );
        }

        // D.3.c — frame finalisation:
        //
        // 1. Extend the frame's borders so future inter frames using
        //    this frame as a reference can read MC halos past the
        //    visible region (libvpx `vpx_extend_frame_borders_c`,
        //    `yv12extend.c:169`).
        extend_frame_borders(&mut yv12);

        // D.20 — env-gated dump of the post-recon, post-LF (if not
        // disabled), post-border-extend visible Y plane for any
        // non-shown frame (typically an altref). Writes
        // `${RVP9_DUMP_ALTREF}_y.bin` (raw Y bytes, row-major, no
        // padding) so we can diff against a libvpx altref dump.
        if !frame.header.show_frame
            && let Ok(path) = std::env::var("RVP9_DUMP_ALTREF")
        {
            dump_visible_y_plane(&yv12, &path);
        }

        let frame_buf: FrameBuf = std::sync::Arc::new(yv12);
        let cur_mv_grid_arc: std::sync::Arc<Vec<rvp9_core::inter::mv_ref::PrevFrameMvRef>> =
            std::sync::Arc::new(cur_mv_grid);

        // 2. Update `ref_frame_map` per `refresh_frame_flags` mask
        //    (libvpx `vp9_decodeframe.c:2914` — for each bit `i` set in
        //    the mask, `cm->ref_frame_map[i] = cm->new_fb_idx`). We
        //    park the freshly-decoded FrameBuf in the pool via
        //    `acquire_free` and write that pool slot index into
        //    every selected ref_frame_map slot. The parallel `mv_grids`
        //    array continues to hold `Arc<Vec<PrevFrameMvRef>>` clones
        //    (one per refreshed ref slot) — those don't go through the
        //    pool because they're not `Yv12Buffer`s.
        //
        // G12 refactor (2026-05-21). Pre-G12 stored an `Arc` clone in
        // every ref slot AND mirror-wrote `frame_pool[i]`, doubling
        // `Arc::strong_count` per buffer. G12 reverts to the libvpx-
        // faithful single-Arc-per-buffer model (`ref_frame_map[i]`
        // holds a `Some(u8)` pool slot index; the pool owns the only
        // strong ref to the buffer apart from the one we hand back
        // to the caller).
        let pool_slot = self
            .frame_pool
            .acquire_free(frame_buf.clone())
            .map_err(|_| DecodeError::FrameAlloc)?;
        let pool_slot_u8 = pool_slot as u8;
        debug_assert!(pool_slot < rvp9_core::frame::DECODER_FRAME_BUFFERS);
        // Use `write_ref_slot` so `frame_pool.ref_counts` stays in
        // sync (G12 R2 Critical fix — pre-fix the silent aliasing
        // bug fired when an inter refresh recycled a pool slot the
        // ref-map still pointed at).
        for i in 0..self.ref_frame_map.len() {
            if (refresh_frame_flags >> i) & 1 != 0 {
                self.write_ref_slot(i, pool_slot_u8);
                self.mv_grids[i] = Some(cur_mv_grid_arc.clone());
            }
        }

        // D.22 — `cm->prev_frame = cm->cur_frame` mirror. Save this
        // frame's MV grid as the "prev frame" for the next decode,
        // independent of which `ref_frame_map` slots were refreshed.
        // Matches libvpx `vp9_decoder.c:478` — fires for every non-
        // show-existing decode (we already early-return for that path).
        self.prev_frame_mv_grid = Some(cur_mv_grid_arc.clone());
        self.prev_frame_mv_grid_dims = (mi_rows, mi_cols);

        self.last_frame_was_key = false;
        self.last_frame_size = Some((frame.header.width, frame.header.height));
        self.last_intra_only = false;
        self.last_show_frame = frame.header.show_frame;
        // D.10 — swap the just-decoded segment_id slab into
        // `prev_segment_ids` for the next frame's inheritance reads.
        // Mirrors libvpx's seg-map double-buffer at end-of-frame
        // (`vp9_alloccommon.c:vp9_swap_current_and_last_seg_map`).
        if std::env::var("RVP9_TRACE_SEG").is_ok() {
            let nz = cur_segment_ids.iter().filter(|&&v| v != 0).count();
            eprintln!(
                "[rvp9_seg] inter end: mi={}x{} non-zero seg cells = {} / {}",
                mi_rows,
                mi_cols,
                nz,
                cur_segment_ids.len()
            );
            // D.20 — also dump first SB row of seg map (rows 0..4 cols 0..16)
            for r in 0..4.min(mi_rows) {
                let mut row = String::new();
                for c in 0..16.min(mi_cols) {
                    row.push_str(&format!(" {}", cur_segment_ids[r * mi_cols + c]));
                }
                eprintln!("[rvp9_seg] inter row{r}:{}", row);
            }
        }
        self.prev_segment_ids = cur_segment_ids;
        self.prev_segment_ids_dims = (mi_rows, mi_cols);
        // Strategy A (Stage 4 design §9 commit 4) — publish the
        // stitched frame-wide mi grid for the bitstream-diff harness.
        self.last_frame_mi_grid = frame_mi_grid;
        self.last_frame_mi_dims = (mi_rows as u32, mi_cols as u32);
        Ok(frame_buf)
    }

    /// D.10 — resolve the per-frame segmentation by overlaying the
    /// running `seg_feature_data` / `seg_abs_delta` onto the parsed
    /// `tail.segmentation` when this frame did NOT update those fields
    /// (`update_data == false`). Also refreshes the running copy when
    /// this frame DID update them. Mirrors libvpx's in-place mutation
    /// of `cm->seg` in `setup_segmentation`
    /// (`vp9_decodeframe.c:1332-1383`).
    ///
    /// Keyframes always start fresh: libvpx's
    /// `vp9_setup_past_independence` calls `vp9_clearall_segfeatures`
    /// before `setup_segmentation` runs, so any prior feature data is
    /// wiped. Callers should pass `is_keyframe = true` to mirror that.
    fn resolve_segmentation(
        &mut self,
        parsed: &rvp9_core::bitstream::SegmentationParams,
        is_keyframe: bool,
    ) -> rvp9_core::bitstream::SegmentationParams {
        if is_keyframe {
            self.seg_feature_data =
                [[None; rvp9_core::bitstream::SEG_LVL_MAX]; rvp9_core::bitstream::MAX_SEGMENTS];
            self.seg_abs_delta = false;
        }

        let mut resolved = parsed.clone();
        if resolved.enabled {
            if resolved.update_data {
                // This frame's parser already filled `feature_data` /
                // `abs_delta`; refresh our running copy from it.
                self.seg_feature_data = resolved.feature_data;
                self.seg_abs_delta = resolved.abs_delta;
            } else {
                // Inherit from the running state.
                resolved.feature_data = self.seg_feature_data;
                resolved.abs_delta = self.seg_abs_delta;
            }
        }
        resolved
    }

    /// Parse the compressed-header for an inter frame with the
    /// supplied `compound_allowed` / `interp_filter` /
    /// `allow_high_precision_mv` overrides. Companion to
    /// [`Decoder::parse_compressed_partition`] which conservatively
    /// pins `compound_allowed = false`.
    fn parse_compressed_partition_inter(
        &mut self,
        frame: &ParsedFrame<'_>,
        compound_allowed: bool,
        interp_filter: InterpFilter,
        allow_high_precision_mv: bool,
    ) -> Result<CompressedHeaderResult, DecodeError> {
        let tail = frame
            .header
            .tail
            .as_ref()
            .ok_or(DecodeError::WrongFrameKind)?;

        let header_end = frame.header_end_byte;
        let fps = tail.first_partition_size as usize;
        let total = frame.full_frame_data.len();
        let needed_end =
            header_end
                .checked_add(fps)
                .ok_or(DecodeError::TruncatedCompressedHeader {
                    declared: usize::MAX,
                    available: total,
                })?;
        if needed_end > total {
            return Err(DecodeError::TruncatedCompressedHeader {
                declared: needed_end,
                available: total,
            });
        }
        let compressed = &frame.full_frame_data[header_end..needed_end];

        let opts = CompressedHeaderOptions {
            lossless: tail.quantization.is_lossless(),
            intra_only: false,
            interp_filter,
            compound_allowed,
            allow_high_precision_mv,
        };

        let idx = tail.frame_context_idx as usize;
        let fc = &mut self.frame_contexts[idx];
        parse_compressed_header(compressed, fc, opts).map_err(DecodeError::Bitstream)
    }
}

/// Mode-info side length of a 64×64 superblock (mirrors libvpx
/// `MI_BLOCK_SIZE` at `vp9_enums.h:25`).
const SB_MI_SIZE: usize = 8;

/// Stitch per-tile mi grids into a single frame-level mi grid suitable
/// for downstream consumers (loop-filter mask builder, etc.).
///
/// `tile_grids` is a row-major list of `(bounds, tile_grid)` covering
/// the frame's `tile_rows × tile_cols` tiles. Each `tile_grid` is the
/// row-major `Option<ModeInfo>` slab returned by
/// [`tile_ctx::TileDecodeCtx::mi_grid`], sized
/// `(bounds.mi_row_end - bounds.mi_row_start) * (bounds.mi_col_end -
/// bounds.mi_col_start)` and addressed at stride =
/// `bounds.mi_col_end - bounds.mi_col_start`.
///
/// Returns a frame-wide `Vec<Option<ModeInfo>>` of length
/// `mi_rows * mi_cols` at stride `mi_cols`, where each cell's value
/// comes from the tile that covers `(mi_row, mi_col)`. Cells outside
/// any tile (shouldn't happen for valid `TileInfo`) stay `None`.
///
/// Mirrors libvpx's `cm->mi_grid_visible` (`vp9_onyxc_int.h:
/// mi_grid_visible`) which is allocated frame-wide and populated by
/// each tile's `vp9_decodeframe.c:decode_partition` calls writing
/// through the frame-wide stride. Our `TileDecodeCtx::write_block`
/// stores at tile-local stride; this stitcher merges them back to
/// frame stride after the tile loop completes.
fn build_frame_mi_grid(
    mi_rows: usize,
    mi_cols: usize,
    tile_grids: &[(tile::TileBounds, &[Option<ModeInfo>])],
) -> Vec<Option<ModeInfo>> {
    let mut frame: Vec<Option<ModeInfo>> = vec![None; mi_rows * mi_cols];
    for (bounds, tile_grid) in tile_grids {
        let tile_mi_cols = bounds.mi_col_end - bounds.mi_col_start;
        let tile_mi_rows = bounds.mi_row_end - bounds.mi_row_start;
        debug_assert_eq!(tile_grid.len(), tile_mi_cols * tile_mi_rows);
        for r in 0..tile_mi_rows {
            let frame_row = bounds.mi_row_start + r;
            if frame_row >= mi_rows {
                break;
            }
            let src_base = r * tile_mi_cols;
            let dst_base = frame_row * mi_cols + bounds.mi_col_start;
            let copy_cols = tile_mi_cols.min(mi_cols - bounds.mi_col_start);
            frame[dst_base..dst_base + copy_cols]
                .copy_from_slice(&tile_grid[src_base..src_base + copy_cols]);
        }
    }
    frame
}

/// Drive the per-SB loop-filter mask build + frame-wide apply. Mirrors
/// the libvpx sequence in `vp9_loop_filter_frame` →
/// `loop_filter_rows` (`vp9_loopfilter.c:1422-1483`): build the
/// `[seg][ref][mode]` filter-level cube once per frame, walk each
/// superblock to populate its [`LoopFilterMask`] via `setup_mask` +
/// `adjust_mask`, then dispatch [`loop_filter_frame`] in place on the
/// reconstructed planes.
///
/// `mi_grid` is the frame-level row-major `Option<ModeInfo>` slab
/// (stride = `mi_stride = mi_cols`). For single-tile frames this is
/// the tile's grid; for multi-tile frames it's stitched together by
/// [`build_frame_mi_grid`] across all tiles. Mirrors libvpx
/// `cm->mi_grid_visible` (`vp9_onyxc_int.h:mi_grid_visible`) which
/// spans the entire frame regardless of tile count.
#[allow(clippy::too_many_arguments)]
fn apply_loop_filter_to_frame(
    yv12: &mut Yv12Buffer,
    mi_grid: &[Option<ModeInfo>],
    mi_stride: usize,
    mi_rows: usize,
    mi_cols: usize,
    filter_level: u8,
    sharpness_level: u8,
    lf_deltas: ResolvedLfDeltas,
    seg_view: SegmentationLfView,
) {
    let levels = resolve_filter_levels(filter_level, lf_deltas, seg_view);
    let info = LoopFilterInfo::new(sharpness_level);

    let sb_rows = mi_rows.div_ceil(SB_MI_SIZE);
    let sb_cols = mi_cols.div_ceil(SB_MI_SIZE);

    // Materialise the full-frame ModeInfo grid (one entry per 8×8
    // luma cell, row-major with stride = mi_stride). `setup_mask`
    // expects a flat `&[ModeInfo]` indexed as `mi[r*stride + c]`,
    // mirroring libvpx's `cm->mi_grid_visible` pointer array
    // (`vp9_onyxc_int.h:mi_grid_visible`). For the single-tile path
    // the tile grid IS the frame grid.
    //
    // Cells the partition walker won't reach (past-frame quadrants,
    // bounded by `mi_rows` / `mi_cols`) get a placeholder; the
    // walker's `max_rows` / `max_cols` clamp guarantees those cells
    // are never read. Sub-leaf MI cells inside a >8×8 partition that
    // weren't explicitly written by the decoder (libvpx's "mi pointer
    // duplication" convention) likewise stay as the placeholder, but
    // `setup_mask` only reads the partition leaf's top-left MI at
    // each leaf boundary, so the duplication isn't required.
    let placeholder = mi_grid
        .iter()
        .find_map(|cell| cell.as_ref().cloned())
        .unwrap_or_else(default_padding_modeinfo);
    let mut frame_mi: Vec<ModeInfo> = Vec::with_capacity(mi_grid.len().max(1));
    for cell in mi_grid {
        match cell {
            Some(mi) => frame_mi.push(*mi),
            None => frame_mi.push(placeholder),
        }
    }

    let mut lfm_grid: Vec<LoopFilterMask> = vec![LoopFilterMask::default(); sb_rows * sb_cols];

    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let mi_row = sb_row * SB_MI_SIZE;
            let mi_col = sb_col * SB_MI_SIZE;

            let lfm = &mut lfm_grid[sb_row * sb_cols + sb_col];
            setup_mask(
                mi_row as u32,
                mi_col as u32,
                &frame_mi,
                mi_stride,
                mi_rows as u32,
                mi_cols as u32,
                &levels,
                lfm,
            );
            adjust_mask(
                mi_row as u32,
                mi_col as u32,
                mi_rows as u32,
                mi_cols as u32,
                lfm,
            );
            dump_lf_masks(lfm, &frame_mi, mi_stride, mi_rows, mi_cols, sb_row, sb_col);
        }
    }

    // G7.2: route through the non420-aware overload so 4:2:2 / 4:4:0
    // buffers get the MI slab + per-frame level cube
    // `filter_block_plane_non420` requires. The 420 / 444 dispatches
    // inside `loop_filter_frame_with_non420` ignore these extra inputs.
    loop_filter_frame_with_non420(
        yv12,
        &lfm_grid,
        sb_cols,
        sb_rows,
        sb_cols,
        mi_rows,
        mi_cols,
        &info,
        0,
        mi_rows,
        Some(rvp9_core::loopfilter::apply::Non420Inputs {
            mi_grid: &frame_mi,
            mi_stride,
            levels: &levels,
        }),
    );
}

/// Stage 3 stretch M.10 diagnostic: dump the per-SB
/// [`LoopFilterMask`] + the visible mi-grid block-size tile, gated by
/// `RVP9_DUMP_LF_MASKS=1`. Used to audit K2.c's wireup against
/// libvpx's `vp9_setup_mask` + `vp9_adjust_mask` output. Cheap when
/// the env var is unset (one `std::env::var` lookup per SB).
fn dump_lf_masks(
    lfm: &LoopFilterMask,
    frame_mi: &[ModeInfo],
    mi_stride: usize,
    mi_rows: usize,
    mi_cols: usize,
    sb_row: usize,
    sb_col: usize,
) {
    if std::env::var("RVP9_DUMP_LF_MASKS").is_err() {
        return;
    }
    let mi_row = sb_row * SB_MI_SIZE;
    let mi_col = sb_col * SB_MI_SIZE;
    eprintln!("SB ({sb_row},{sb_col}) mi=({mi_row},{mi_col}) mi_rows={mi_rows} mi_cols={mi_cols}");
    eprintln!(
        "  left_y    = [{:#018x}, {:#018x}, {:#018x}, {:#018x}]",
        lfm.left_y[0], lfm.left_y[1], lfm.left_y[2], lfm.left_y[3]
    );
    eprintln!(
        "  above_y   = [{:#018x}, {:#018x}, {:#018x}, {:#018x}]",
        lfm.above_y[0], lfm.above_y[1], lfm.above_y[2], lfm.above_y[3]
    );
    eprintln!("  int_4x4_y = {:#018x}", lfm.int_4x4_y);
    eprintln!(
        "  left_uv   = [{:#06x}, {:#06x}, {:#06x}, {:#06x}]",
        lfm.left_uv[0], lfm.left_uv[1], lfm.left_uv[2], lfm.left_uv[3]
    );
    eprintln!(
        "  above_uv  = [{:#06x}, {:#06x}, {:#06x}, {:#06x}]",
        lfm.above_uv[0], lfm.above_uv[1], lfm.above_uv[2], lfm.above_uv[3]
    );
    eprintln!("  int_4x4_uv= {:#06x}", lfm.int_4x4_uv);
    eprintln!("  lfl_y     = {:?}", lfm.lfl_y);
    eprintln!("  mi_grid block_size (visible only):");
    for r in 0..mi_rows {
        let mut row_str = String::new();
        for c in 0..mi_cols {
            let mi = &frame_mi[r * mi_stride + c];
            let tag = match mi.block_size {
                BlockSize::Block4x4 => "4x4 ",
                BlockSize::Block8x8 => "8x8 ",
                BlockSize::Block16x16 => "16x16",
                BlockSize::Block32x32 => "32x32",
                BlockSize::Block64x64 => "64x64",
                BlockSize::Block4x8 => "4x8 ",
                BlockSize::Block8x4 => "8x4 ",
                BlockSize::Block8x16 => "8x16",
                BlockSize::Block16x8 => "16x8",
                BlockSize::Block16x32 => "16x32",
                BlockSize::Block32x16 => "32x16",
                BlockSize::Block32x64 => "32x64",
                BlockSize::Block64x32 => "64x32",
            };
            row_str.push_str(&format!(" {tag}"));
        }
        eprintln!("    r{r}: {row_str}");
    }
}

/// Placeholder [`ModeInfo`] for the rare path where the entire mi
/// grid is empty (e.g. a degenerate tile that produced no leaves).
/// Mirrors libvpx's `vpx_calloc`-initialised `MODE_INFO` slot
/// (`vp9_onyxc_int.h:setup_mi`): an intra DC 8×8 block with
/// segment_id = 0. The partition walker in `setup_mask` only reads
/// this slot if `max_rows == 0 || max_cols == 0`, in which case the
/// walker early-exits before touching the data.
fn default_padding_modeinfo() -> ModeInfo {
    use rvp9_core::block::BlockSize;
    use rvp9_core::intra_pred::IntraMode;
    use rvp9_core::mode_info::BlockMode;
    use rvp9_core::transforms::TxSize;
    ModeInfo {
        block_size: BlockSize::Block8x8,
        tx_size: TxSize::Tx4x4,
        skip: false,
        segment_id: 0,
        seg_id_predicted: false,
        mode: BlockMode::Intra {
            y_mode: IntraMode::Dc,
            uv_mode: IntraMode::Dc,
        },
        bmi: Default::default(),
    }
}

/// Errors from the per-frame decode entry points.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DecodeError {
    /// Called against a header whose `kind` was inappropriate for
    /// the entry point. Examples:
    /// - `decode_show_existing_frame` on a non-`ShowExistingFrame`.
    /// - `parse_compressed_partition` on a `ShowExistingFrame`
    ///   (which has no `tail` / compressed header).
    WrongFrameKind,
    /// `decode_show_existing_frame` referenced an empty slot.
    /// Mirrors libvpx's `vp9_decodeframe.c:2667` corrupt-frame
    /// rejection ("Buffer N does not contain a decoded frame").
    EmptySlot {
        /// The 3-bit slot index that was requested.
        slot: u8,
    },
    /// An inter frame's `ref_frame_indices` named a `ref_frame_map`
    /// slot that holds no decoded frame. Sibling of [`Self::EmptySlot`]
    /// — `EmptySlot` covers the `decode_show_existing_frame` path,
    /// `EmptyInterRef` covers the inter-reference resolution path
    /// at [`Decoder::decode_inter_frame`]. In libvpx terms this is a
    /// bitstream-corrupt condition (no real stream ever names an
    /// empty slot — the prior keyframe always populates them), so
    /// callers should treat it as a malformed-bitstream error rather
    /// than an unimplemented-feature signal.
    EmptyInterRef {
        /// The 3-bit slot index that was requested.
        slot: u8,
    },
    /// `parse_compressed_partition` saw a `first_partition_size`
    /// that runs past the end of the frame's byte slice.
    TruncatedCompressedHeader {
        /// The declared end offset (`header_end + first_partition_size`).
        declared: usize,
        /// The frame's actual data length.
        available: usize,
    },
    /// A bitstream-layer error surfaced during compressed-header
    /// parsing. Carries the underlying `BitstreamError` for
    /// diagnostics.
    Bitstream(BitstreamError),
    /// A decode path that has not yet been implemented in this
    /// stage of the Stage 3 build-out. The `reason` describes which
    /// path. Used by [`Decoder::decode_frame`] for non-show-existing
    /// branches until the full pipeline lands.
    ///
    /// **Contrast with [`Self::UnsupportedProfile`].** `Unimplemented`
    /// means "the decoder reached a codepath whose implementation is
    /// not yet written — the missing work has NOT been catalogued as a
    /// specific follow-up task". `UnsupportedProfile` (and any sibling
    /// gated-feature variants in the future) means "the decoder
    /// DETECTED a known-unsupported feature and rejected it with a
    /// dedicated variant — the implementation work is deferred to a
    /// named follow-up". The conformance harness counts the two
    /// independently so a gated feature being silently re-routed
    /// through `Unimplemented` (K-5 regression) is impossible.
    Unimplemented {
        /// Short human-readable description of the missing path.
        reason: &'static str,
    },
    /// A feature explicitly gated until a follow-up task lands. The
    /// codepath has been DETECTED but its implementation is deferred.
    ///
    /// Sibling of [`Self::Unimplemented`] — see that variant's doc for
    /// the distinction. Today the only producer is the G7.0 gate at
    /// [`Decoder::decode_frame`] which rejects Profile 1 (non-`Yuv420`
    /// chroma) frames until the G7.1 tile/recon/loopfilter deep
    /// refactor lands. Lives as a dedicated variant (rather than
    /// reusing `Unimplemented`) so the conformance harness can bucket
    /// gated features independently from genuinely missing codepaths:
    /// pre-K-5 the silent skip-folding bug masked real failures behind
    /// the expected-skip count, and re-routing Profile 1 frames
    /// through `Unimplemented` would resurrect that exact failure
    /// mode the moment a Profile 1 vector entered the corpus.
    UnsupportedProfile {
        /// The 2-bit `profile` value parsed from the uncompressed
        /// header. Matches [`BitstreamError::UnsupportedProfile`]'s
        /// `profile: u8` shape.
        profile: u8,
    },
    /// A tile-decode-layer error surfaced during keyframe decode.
    /// Carries the underlying [`TileDecodeError`] for diagnostics.
    TileDecode(TileDecodeError),
    /// Failed to allocate the destination [`FrameBuf`]. Surfaces e.g.
    /// dimensions exceeding the safety cap in [`Yv12Buffer::alloc`].
    FrameAlloc,
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongFrameKind => f.write_str("frame kind is inappropriate for this entry point"),
            Self::EmptySlot { slot } => {
                write!(f, "frame pool slot {slot} contains no decoded frame")
            }
            Self::EmptyInterRef { slot } => {
                write!(f, "inter frame referenced empty ref_frame_map slot {slot}")
            }
            Self::TruncatedCompressedHeader {
                declared,
                available,
            } => write!(
                f,
                "compressed header declared {declared} bytes but only {available} available"
            ),
            Self::Bitstream(e) => write!(f, "bitstream error: {e}"),
            Self::Unimplemented { reason } => write!(f, "unimplemented: {reason}"),
            Self::UnsupportedProfile { profile } => {
                write!(
                    f,
                    "unsupported profile {profile} (feature gated, see DecodeError::UnsupportedProfile docs)"
                )
            }
            Self::TileDecode(e) => write!(f, "tile decode: {e}"),
            Self::FrameAlloc => f.write_str("frame allocation failed"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// One frame extracted from a transport packet by
/// [`Decoder::parse_packet`].
pub struct ParsedFrame<'a> {
    /// Parsed uncompressed-header struct.
    pub header: UncompressedHeader,
    /// Byte offset within `full_frame_data` where the uncompressed
    /// header ended and the compressed header begins. The
    /// `first_partition_size` field of `header.tail` (when present)
    /// tells you the size of the compressed-header partition.
    pub header_end_byte: usize,
    /// The frame's full byte slice (everything from `frame_marker` to
    /// the last tile byte). Tile data starts at
    /// `header_end_byte + header.tail.first_partition_size`.
    pub full_frame_data: &'a [u8],
}

/// Result of [`Decoder::parse_packet`]. Iterable over the 1..=8
/// frames found in the packet.
pub struct PacketFrames<'a> {
    frames: Vec<ParsedFrame<'a>>,
}

impl<'a> PacketFrames<'a> {
    /// Number of frames in the packet (`1..=8`).
    #[must_use]
    pub fn count(&self) -> usize {
        self.frames.len()
    }

    /// Borrow the parsed frames.
    #[must_use]
    pub fn frames(&self) -> &[ParsedFrame<'a>] {
        &self.frames
    }
}

impl<'a> IntoIterator for PacketFrames<'a> {
    type Item = ParsedFrame<'a>;
    type IntoIter = std::vec::IntoIter<ParsedFrame<'a>>;
    fn into_iter(self) -> Self::IntoIter {
        self.frames.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::bitstream::{ColorSpace, FrameKind, Profile, RawBitWriter, frame::sync};

    /// Build a minimal valid keyframe header buffer (returns the
    /// raw `&[u8]` payload). Mirrors the helper in
    /// `bitstream::header::tests`.
    fn build_minimal_keyframe() -> Vec<u8> {
        // 320 × 240 keyframe, profile 0, BT.601, studio.
        let mut buf = vec![0u8; 256];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_literal(sync::FRAME_MARKER as u32, 2);
        w.write_bit(0); // profile low
        w.write_bit(0); // profile high → Profile 0
        w.write_bit(0); // show_existing = 0
        w.write_bit(0); // frame_type = KEY
        w.write_bit(1); // show_frame = 1
        w.write_bit(0); // error_resilient = 0
        w.write_literal(sync::SYNC_CODE_0 as u32, 8);
        w.write_literal(sync::SYNC_CODE_1 as u32, 8);
        w.write_literal(sync::SYNC_CODE_2 as u32, 8);
        w.write_literal(ColorSpace::Bt601 as u32, 3);
        w.write_bit(0); // color_range = Studio
        w.write_literal(319, 16); // width-1
        w.write_literal(239, 16); // height-1
        w.write_bit(0); // no render size

        // Tail: refresh_frame_context (1), frame_parallel_decoding_mode (1),
        // frame_context_idx (2), loopfilter (10), quant (11),
        // segmentation (1), tile cols (0 — width 320 → mi_cols 40
        // → sb64_cols 5 → min=max=0), tile rows (1), first_partition_size (16=1).
        w.write_bit(0);
        w.write_bit(0);
        w.write_literal(0, 2);
        w.write_literal(0, 6);
        w.write_literal(0, 3);
        w.write_bit(0);
        w.write_literal(0, 8);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0); // log2_tile_rows = 0
        w.write_literal(1, 16); // first_partition_size = 1
        // Pad.
        for _ in 0..8 {
            w.write_literal(0, 8);
        }
        let used = w.bytes_written();
        buf.truncate(used + 8);
        buf
    }

    /// Build a minimal valid intra-only header buffer with the given
    /// `reset_frame_context` and `refresh_frame_flags`. The compressed
    /// header / tile section is `n` zero bytes (same shape as
    /// [`build_lossless_keyframe_with_compressed_size`]). Mirrors
    /// libvpx's intra-only header at `vp9_decodeframe.c:2705-2735`.
    ///
    /// `frame_context_idx` defaults to 0; use
    /// [`build_minimal_intra_only_with_fc_idx`] when a different bank
    /// selector is needed.
    fn build_minimal_intra_only(
        reset_frame_context: u8,
        refresh_frame_flags: u8,
        n: u16,
    ) -> Vec<u8> {
        build_minimal_intra_only_with_fc_idx(
            reset_frame_context,
            refresh_frame_flags,
            /* fc_idx = */ 0,
            n,
        )
    }

    /// As [`build_minimal_intra_only`] but with an explicit
    /// `frame_context_idx` (2 bits — value range `0..=3`).
    fn build_minimal_intra_only_with_fc_idx(
        reset_frame_context: u8,
        refresh_frame_flags: u8,
        fc_idx: u8,
        n: u16,
    ) -> Vec<u8> {
        build_minimal_intra_only_full(
            reset_frame_context,
            refresh_frame_flags,
            fc_idx,
            /* error_resilient = */ false,
            n,
        )
    }

    /// Most general intra-only fixture builder — adds an explicit
    /// `error_resilient` toggle on top of
    /// [`build_minimal_intra_only_with_fc_idx`]. Used by the error-
    /// resilient bank-reset test (the spec at
    /// `vp9_entropymode.c:452` short-circuits `force_all_reset` when
    /// `error_resilient == 1`, regardless of `reset_frame_context`).
    fn build_minimal_intra_only_full(
        reset_frame_context: u8,
        refresh_frame_flags: u8,
        fc_idx: u8,
        error_resilient: bool,
        n: u16,
    ) -> Vec<u8> {
        assert!(fc_idx < 4, "frame_context_idx is a 2-bit field");
        let mut buf = vec![0u8; 256];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_literal(sync::FRAME_MARKER as u32, 2);
        w.write_bit(0); // profile low
        w.write_bit(0); // profile high → Profile 0
        w.write_bit(0); // show_existing_frame = 0
        w.write_bit(1); // frame_type = INTER (= 1)
        w.write_bit(0); // show_frame = 0 (must be 0 for intra-only)
        w.write_bit(if error_resilient { 1 } else { 0 });
        w.write_bit(1); // intra_only = 1 (only read when !show_frame)
        // `reset_frame_context` is only present when !error_resilient
        // (libvpx forces it to 0 in that case; see header.rs:284-288).
        if !error_resilient {
            w.write_literal(u32::from(reset_frame_context), 2);
        }
        // Intra-only profile-0 path: sync code + (no color config —
        // Profile 0 inherits defaults per spec) + refresh_frame_flags +
        // frame_size + render_size flag.
        w.write_literal(sync::SYNC_CODE_0 as u32, 8);
        w.write_literal(sync::SYNC_CODE_1 as u32, 8);
        w.write_literal(sync::SYNC_CODE_2 as u32, 8);
        w.write_literal(u32::from(refresh_frame_flags), 8);
        w.write_literal(319, 16); // width - 1
        w.write_literal(239, 16); // height - 1
        w.write_bit(0); // no render size
        // Tail: refresh_frame_context (1), frame_parallel_decoding_mode
        // (1), frame_context_idx (2), loopfilter (10), quant (11),
        // segmentation (1), tile cols (0 — width 320 → mi_cols 40
        // → sb64_cols 5 → min=max=0), tile rows (1), first_partition_size.
        //
        // `refresh_frame_context` + `frame_parallel_decoding_mode` are
        // ONLY present in the bitstream when !error_resilient — libvpx
        // forces them to (false, true) otherwise (see header.rs:384-390).
        if !error_resilient {
            w.write_bit(0);
            w.write_bit(0);
        }
        w.write_literal(u32::from(fc_idx), 2);
        w.write_literal(0, 6);
        w.write_literal(0, 3);
        w.write_bit(0);
        // base_qindex = 100 (non-zero to bypass the lossless-only
        // gate at decode_intra_frame_inner; lossless is G2's gap, not
        // G1's). The actual value doesn't matter — only that
        // `is_lossless()` returns false.
        w.write_literal(100, 8);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        w.write_literal(u32::from(n), 16);
        let used = w.bytes_written();
        buf.truncate(used + n as usize + 4);
        buf
    }

    /// G1 routing test: an intra-only packet parses, classifies as
    /// `FrameKind::IntraOnly`, and dispatches through
    /// [`Decoder::decode_intra_only_frame`] — i.e. does NOT return
    /// `DecodeError::Unimplemented` (the pre-G1 behaviour). The
    /// downstream decode is expected to fail at the tile bool-coder
    /// (the test fixture has only zero bytes for tile data) but the
    /// error MUST be a tile / bitstream error, not the
    /// `Unimplemented` short-circuit.
    #[test]
    fn intra_only_dispatch_no_longer_unimplemented() {
        let buf = build_minimal_intra_only(
            /* reset_frame_context = */ 0, /* refresh_frame_flags = */ 0xFF, 16,
        );
        let mut dec = Decoder::new();
        let frames = dec.parse_packet(&buf).expect("valid intra-only");
        assert_eq!(frames.count(), 1);
        let header = frames.frames()[0].header.clone();
        assert_eq!(
            header.kind,
            FrameKind::IntraOnly {
                reset_frame_context: 0,
                refresh_frame_flags: 0xFF,
            }
        );
        let frame_data: Vec<u8> = frames.frames()[0].full_frame_data.to_vec();
        let header_end = frames.frames()[0].header_end_byte;
        drop(frames);
        let parsed = ParsedFrame {
            header,
            header_end_byte: header_end,
            full_frame_data: &frame_data,
        };
        // The fixture's tile section is zero-padded; downstream decode
        // may either succeed (zero bytes parse as all-zeros tokens) or
        // fail at a bool-coder boundary. EITHER outcome demonstrates
        // that the IntraOnly arm no longer short-circuits to
        // Unimplemented — the load-bearing assertion is the negative.
        let result = dec.decode_frame(&parsed);
        match &result {
            Err(DecodeError::Unimplemented { reason }) => {
                panic!("G1: intra-only frame still returns Unimplemented (reason: {reason})")
            }
            Ok(_) | Err(_) => { /* anything but Unimplemented is fine */ }
        }
        // Positive property: if the inner decode succeeded (the
        // synthetic zero-byte tile DOES parse as all-zero tokens for
        // this minimal fixture), the IntraOnly arm must have run the
        // `ref_frame_map` refresh per `refresh_frame_flags = 0xFF`,
        // populating all 8 slots with the freshly-decoded FrameBuf.
        // This complements the negative assertion above by exercising
        // a real downstream effect of the dispatch.
        //
        // If the decode failed (a future fixture change that causes
        // the bool-coder to trip), the ref-map refresh runs at the END
        // of `decode_intra_frame_inner` so the slots would remain
        // empty — in that case the assertion is skipped and only the
        // negative routing check is exercised.
        //
        // TODO(G1.1): expand with a pixel-correctness assertion when
        // an IntraOnly conformance vector is added to the corpus.
        //
        // Item 4 (devil's-advocate): the prior `if result.is_ok()` guard
        // silently downgraded this test to negative-only if the fixture
        // ever stopped decoding successfully. Hard-fail instead so
        // fixture drift surfaces immediately as a test failure rather
        // than as a silently-skipped positive check.
        assert!(
            result.is_ok(),
            "synthetic IntraOnly fixture should decode successfully; if \
             this assertion fires, either the fixture is broken or the \
             IntraOnly path regressed: {result:?}",
        );
        for (i, slot) in dec.ref_frame_map.iter().enumerate() {
            assert!(
                slot.is_some(),
                "G1: ref_frame_map[{i}] not populated after IntraOnly decode \
                 with refresh_frame_flags=0xFF",
            );
        }
    }

    /// G1: an intra-only frame with `reset_frame_context = 3` resets
    /// every FRAME_CONTEXTS bank to libvpx defaults before the inner
    /// decode driver runs. Verifies the `vp9_setup_past_independence`
    /// bank-reset mirror (`vp9_entropymode.c:452-455`).
    ///
    /// Strategy: poison every bank with a sentinel that differs from
    /// the default; build an intra-only packet with
    /// `reset_frame_context = 3`; invoke `decode_frame`; observe that
    /// every bank is restored to defaults regardless of how the
    /// downstream decode terminates.
    #[test]
    fn intra_only_reset_frame_context_3_resets_all_banks() {
        let default_bank = FrameContext::new_default();
        let mut dec = Decoder::new();
        // Poison every bank: zero its `skip_probs[0]` (default is
        // libvpx's 192). Verify the poison sticks pre-dispatch.
        for bank in dec.frame_contexts.iter_mut() {
            bank.skip_probs[0] = 0;
        }
        assert_ne!(
            dec.frame_contexts[0].skip_probs[0],
            default_bank.skip_probs[0]
        );

        let buf = build_minimal_intra_only(
            /* reset_frame_context = */ 3, /* refresh_frame_flags = */ 0xFF, 16,
        );
        let frames = dec.parse_packet(&buf).expect("valid intra-only");
        let header = frames.frames()[0].header.clone();
        let frame_data: Vec<u8> = frames.frames()[0].full_frame_data.to_vec();
        let header_end = frames.frames()[0].header_end_byte;
        drop(frames);
        let parsed = ParsedFrame {
            header,
            header_end_byte: header_end,
            full_frame_data: &frame_data,
        };
        // Decode is expected to fail downstream at the tile bool-coder
        // (zero tile bytes), but the bank reset happens *before* the
        // inner decode driver, so we check banks regardless of the
        // decode outcome.
        let _ = dec.decode_frame(&parsed);
        for (i, bank) in dec.frame_contexts.iter().enumerate() {
            assert_eq!(
                bank.skip_probs[0], default_bank.skip_probs[0],
                "bank[{i}] was not reset to defaults after reset_frame_context=3"
            );
        }
    }

    /// G1: an intra-only frame with `reset_frame_context = 2` resets
    /// ONLY the bank selected by `frame_context_idx` (= 0 in the
    /// minimal fixture). Banks 1..=3 keep the poison sentinel.
    /// Mirrors libvpx `vp9_entropymode.c:456-459`.
    #[test]
    fn intra_only_reset_frame_context_2_resets_selected_bank_only() {
        let default_bank = FrameContext::new_default();
        let mut dec = Decoder::new();
        for bank in dec.frame_contexts.iter_mut() {
            bank.skip_probs[0] = 0;
        }
        let buf = build_minimal_intra_only(
            /* reset_frame_context = */ 2, /* refresh_frame_flags = */ 0xFF, 16,
        );
        let frames = dec.parse_packet(&buf).expect("valid intra-only");
        let header = frames.frames()[0].header.clone();
        let frame_data: Vec<u8> = frames.frames()[0].full_frame_data.to_vec();
        let header_end = frames.frames()[0].header_end_byte;
        drop(frames);
        let parsed = ParsedFrame {
            header,
            header_end_byte: header_end,
            full_frame_data: &frame_data,
        };
        let _ = dec.decode_frame(&parsed);
        // Bank 0 reset; banks 1..=3 still poisoned.
        assert_eq!(
            dec.frame_contexts[0].skip_probs[0],
            default_bank.skip_probs[0]
        );
        for i in 1..FRAME_CONTEXTS {
            assert_eq!(
                dec.frame_contexts[i].skip_probs[0], 0,
                "bank[{i}] should be untouched when reset_frame_context=2"
            );
        }
    }

    /// G1: companion to
    /// [`intra_only_reset_frame_context_2_resets_selected_bank_only`]
    /// that exercises the bank-selection logic with a non-zero
    /// `frame_context_idx`. With `fc_idx = 2`, only bank 2 must be
    /// reset; banks 0, 1, 3 must retain their poison sentinel.
    ///
    /// Mirrors libvpx `vp9_entropymode.c:456-459` — the selection is
    /// `self.frame_contexts[fc_idx]`, so a single non-zero index
    /// suffices to demonstrate the indexing is honoured (rather than
    /// hardcoded to bank 0).
    #[test]
    fn intra_only_reset_frame_context_2_resets_fc_idx_2_only() {
        let default_bank = FrameContext::new_default();
        let mut dec = Decoder::new();
        for bank in dec.frame_contexts.iter_mut() {
            bank.skip_probs[0] = 0;
        }
        let buf = build_minimal_intra_only_with_fc_idx(
            /* reset_frame_context = */ 2, /* refresh_frame_flags = */ 0xFF,
            /* fc_idx = */ 2, 16,
        );
        let frames = dec.parse_packet(&buf).expect("valid intra-only");
        let header = frames.frames()[0].header.clone();
        // Sanity-check the fixture: the header tail's
        // `frame_context_idx` must round-trip as 2 (guards against a
        // future bit-layout drift in `build_minimal_intra_only_with_
        // fc_idx`).
        assert_eq!(
            header
                .tail
                .as_ref()
                .expect("tail present")
                .frame_context_idx,
            2,
        );
        let frame_data: Vec<u8> = frames.frames()[0].full_frame_data.to_vec();
        let header_end = frames.frames()[0].header_end_byte;
        drop(frames);
        let parsed = ParsedFrame {
            header,
            header_end_byte: header_end,
            full_frame_data: &frame_data,
        };
        let _ = dec.decode_frame(&parsed);
        // Bank 2 reset; banks 0, 1, 3 still poisoned.
        assert_eq!(
            dec.frame_contexts[2].skip_probs[0],
            default_bank.skip_probs[0]
        );
        for i in 0..FRAME_CONTEXTS {
            if i == 2 {
                continue;
            }
            assert_eq!(
                dec.frame_contexts[i].skip_probs[0], 0,
                "bank[{i}] should be untouched when reset_frame_context=2 and fc_idx=2"
            );
        }
    }

    /// G1 (DA Item 3): `reset_frame_context = 0` must NOT reset any
    /// FRAME_CONTEXTS bank. Per libvpx `vp9_entropymode.c:452-459` the
    /// switch only triggers a reset on `reset_frame_context == 2` or
    /// `== 3` (and on `error_resilient`); values 0 / 1 are no-ops for
    /// the bank pool. Poison every bank with a sentinel, decode an
    /// intra-only packet with `reset_frame_context = 0`, and verify
    /// every bank still carries the sentinel.
    #[test]
    fn intra_only_reset_frame_context_0_keeps_all_banks() {
        let default_bank = FrameContext::new_default();
        let mut dec = Decoder::new();
        for bank in dec.frame_contexts.iter_mut() {
            bank.skip_probs[0] = 0;
        }
        // Sanity: poison differs from the default we are testing
        // *against*.
        assert_ne!(0, default_bank.skip_probs[0]);

        let buf = build_minimal_intra_only_with_fc_idx(
            /* reset_frame_context = */ 0, /* refresh_frame_flags = */ 0xFF,
            /* fc_idx = */ 0, 16,
        );
        let frames = dec.parse_packet(&buf).expect("valid intra-only");
        let header = frames.frames()[0].header.clone();
        let frame_data: Vec<u8> = frames.frames()[0].full_frame_data.to_vec();
        let header_end = frames.frames()[0].header_end_byte;
        drop(frames);
        let parsed = ParsedFrame {
            header,
            header_end_byte: header_end,
            full_frame_data: &frame_data,
        };
        let _ = dec.decode_frame(&parsed);
        for i in 0..FRAME_CONTEXTS {
            assert_eq!(
                dec.frame_contexts[i].skip_probs[0], 0,
                "bank[{i}] should be untouched when reset_frame_context=0 \
                 (intra-only with reset_frame_context in 0..=1 is a no-op \
                  per vp9_entropymode.c:452-459)"
            );
        }
    }

    /// G1 (DA Item 3): `reset_frame_context = 1` is also a no-op for
    /// the bank pool per libvpx `vp9_entropymode.c:452-459` — only `2`
    /// and `3` trigger resets. Same poison-then-decode strategy as
    /// [`intra_only_reset_frame_context_0_keeps_all_banks`].
    #[test]
    fn intra_only_reset_frame_context_1_keeps_all_banks() {
        let mut dec = Decoder::new();
        for bank in dec.frame_contexts.iter_mut() {
            bank.skip_probs[0] = 0;
        }
        let buf = build_minimal_intra_only_with_fc_idx(
            /* reset_frame_context = */ 1, /* refresh_frame_flags = */ 0xFF,
            /* fc_idx = */ 0, 16,
        );
        let frames = dec.parse_packet(&buf).expect("valid intra-only");
        let header = frames.frames()[0].header.clone();
        let frame_data: Vec<u8> = frames.frames()[0].full_frame_data.to_vec();
        let header_end = frames.frames()[0].header_end_byte;
        drop(frames);
        let parsed = ParsedFrame {
            header,
            header_end_byte: header_end,
            full_frame_data: &frame_data,
        };
        let _ = dec.decode_frame(&parsed);
        for i in 0..FRAME_CONTEXTS {
            assert_eq!(
                dec.frame_contexts[i].skip_probs[0], 0,
                "bank[{i}] should be untouched when reset_frame_context=1 \
                 (libvpx treats 0/1 identically as the no-reset cases)"
            );
        }
    }

    /// G1 (DA Item 3): `error_resilient = 1` forces an all-banks reset
    /// regardless of `reset_frame_context`. Per libvpx
    /// `vp9_entropymode.c:452`, the `force_all_reset` predicate is
    /// `frame.error_resilient || reset_frame_context == 3`, so an
    /// error-resilient intra-only frame with `reset_frame_context = 0`
    /// must still wipe every bank back to defaults.
    #[test]
    fn intra_only_error_resilient_resets_all_banks() {
        let default_bank = FrameContext::new_default();
        let mut dec = Decoder::new();
        for bank in dec.frame_contexts.iter_mut() {
            bank.skip_probs[0] = 0;
        }
        assert_ne!(
            dec.frame_contexts[0].skip_probs[0],
            default_bank.skip_probs[0]
        );

        let buf = build_minimal_intra_only_full(
            /* reset_frame_context = */ 0, /* refresh_frame_flags = */ 0xFF,
            /* fc_idx = */ 0, /* error_resilient = */ true, 16,
        );
        let frames = dec.parse_packet(&buf).expect("valid intra-only");
        let header = frames.frames()[0].header.clone();
        // Sanity-check fixture round-trip: error_resilient must be true.
        assert!(
            header.error_resilient,
            "fixture builder failed to set error_resilient = 1"
        );
        let frame_data: Vec<u8> = frames.frames()[0].full_frame_data.to_vec();
        let header_end = frames.frames()[0].header_end_byte;
        drop(frames);
        let parsed = ParsedFrame {
            header,
            header_end_byte: header_end,
            full_frame_data: &frame_data,
        };
        let _ = dec.decode_frame(&parsed);
        for (i, bank) in dec.frame_contexts.iter().enumerate() {
            assert_eq!(
                bank.skip_probs[0], default_bank.skip_probs[0],
                "bank[{i}] should be reset to defaults under error_resilient=1 \
                 (same effect as reset_frame_context=3)"
            );
        }
    }

    /// G1 (DA Item 3) — refreshed for G12 (2026-05-21): two
    /// consecutive intra-only decodes each with `refresh_frame_flags
    /// = 0xFF` must each replace the FrameBuf currently bound to
    /// every ref_frame_map slot. The G12 refactor collapsed the
    /// pre-G12 parallel-mirror data structures (`ref_frame_map`
    /// held `Arc` clones and `frame_pool` held a mirror), so the
    /// "mirror in lockstep" assertion is gone; what remains is the
    /// libvpx-faithful chain `ref_frame_map[i] -> frame_pool[N]`,
    /// where after `refresh_frame_flags = 0xFF` every ref slot
    /// holds the SAME pool slot index, and the second decode
    /// installs a different FrameBuf at that (or some other) pool
    /// slot.
    #[test]
    fn intra_only_then_intra_only_overwrites_every_ref_slot() {
        let mut dec = Decoder::new();
        // First intra-only decode: populates every slot via the
        // same FrameBuf (refresh_frame_flags = 0xFF).
        let buf1 = build_minimal_intra_only(
            /* reset_frame_context = */ 0, /* refresh_frame_flags = */ 0xFF, 16,
        );
        let frames = dec.parse_packet(&buf1).expect("valid intra-only #1");
        let header = frames.frames()[0].header.clone();
        let frame_data: Vec<u8> = frames.frames()[0].full_frame_data.to_vec();
        let header_end = frames.frames()[0].header_end_byte;
        drop(frames);
        let parsed = ParsedFrame {
            header,
            header_end_byte: header_end,
            full_frame_data: &frame_data,
        };
        dec.decode_frame(&parsed).expect("intra-only #1 decode");

        // After decode #1: every ref_frame_map slot must point at
        // the same pool slot (refresh_frame_flags=0xFF splat).
        let pool_slot_after_1 =
            dec.ref_frame_map[0].expect("ref_frame_map[0] populated after first decode");
        for i in 0..rvp9_core::bitstream::REF_FRAMES {
            assert_eq!(
                dec.ref_frame_map[i],
                Some(pool_slot_after_1),
                "ref_frame_map[{i}] should point at the same pool slot as [0] \
                 after refresh_frame_flags=0xFF",
            );
        }
        // Snapshot the FrameBuf identity through the resolver
        // (uses Y-plane data pointer as a stable identity).
        let snap: Vec<*const u8> = (0..rvp9_core::bitstream::REF_FRAMES)
            .map(|i| {
                dec.ref_frame_buf(i)
                    .expect("slot populated after first decode")
                    .y_plane()
                    .data()
                    .as_ptr()
            })
            .collect();

        // Second intra-only decode (same fixture, fresh FrameBuf) — must
        // install a different FrameBuf at each resolved ref slot.
        let buf2 = build_minimal_intra_only(
            /* reset_frame_context = */ 0, /* refresh_frame_flags = */ 0xFF, 16,
        );
        let frames = dec.parse_packet(&buf2).expect("valid intra-only #2");
        let header = frames.frames()[0].header.clone();
        let frame_data: Vec<u8> = frames.frames()[0].full_frame_data.to_vec();
        let header_end = frames.frames()[0].header_end_byte;
        drop(frames);
        let parsed = ParsedFrame {
            header,
            header_end_byte: header_end,
            full_frame_data: &frame_data,
        };
        dec.decode_frame(&parsed).expect("intra-only #2 decode");

        // After decode #2: every ref_frame_map slot must again
        // point at the same pool slot (could be the same pool slot
        // recycled, or a different one — `acquire_free` recycles
        // when refcount == 1, which is the expected state because
        // the caller dropped the FrameBuf from #1).
        let pool_slot_after_2 =
            dec.ref_frame_map[0].expect("ref_frame_map[0] populated after second decode");
        for i in 0..rvp9_core::bitstream::REF_FRAMES {
            assert_eq!(
                dec.ref_frame_map[i],
                Some(pool_slot_after_2),
                "ref_frame_map[{i}] should point at the same pool slot as [0] \
                 after second refresh_frame_flags=0xFF",
            );
        }
        // The FrameBuf at every resolved ref slot must differ from
        // the snapshot — different `Arc`, different `Yv12Buffer`,
        // different Y-data pointer.
        for (i, &snap_ptr) in snap.iter().enumerate() {
            let new_ptr = dec
                .ref_frame_buf(i)
                .expect("ref_frame_map[i] still resolves after second decode")
                .y_plane()
                .data()
                .as_ptr();
            assert_ne!(
                new_ptr, snap_ptr,
                "FrameBuf at ref slot {i} was NOT replaced by the second \
                 IntraOnly decode (refresh_frame_flags=0xFF should install \
                 a fresh FrameBuf at every selected ref slot)",
            );
        }
    }

    #[test]
    fn decoder_construction() {
        let dec = Decoder::new();
        assert_eq!(dec.frame_pool().occupied_count(), 0);
        // Default lf deltas: disabled, all zeros.
        let lf = dec.lf_deltas();
        assert!(!lf.enabled);
        assert_eq!(lf.ref_deltas, [0; 4]);
        assert_eq!(lf.mode_deltas, [0; 2]);
        // Default FRAME_CONTEXT is the libvpx-initialised bank.
        for i in 0..FRAME_CONTEXTS {
            assert!(dec.frame_context(i).initialized);
        }
    }

    #[test]
    fn parse_packet_single_keyframe() {
        let buf = build_minimal_keyframe();
        let mut dec = Decoder::new();
        let frames = dec.parse_packet(&buf).expect("valid keyframe");
        assert_eq!(frames.count(), 1);
        let f = &frames.frames()[0];
        assert_eq!(f.header.kind, FrameKind::KeyFrame);
        assert_eq!(f.header.profile, Profile::Profile0);
        assert_eq!(f.header.width, 320);
        assert_eq!(f.header.height, 240);
        assert!(f.header_end_byte > 0);
        assert!(f.full_frame_data.len() >= f.header_end_byte);
    }

    /// Build a lossless keyframe with `first_partition_size = N`
    /// bytes of zero-padded compressed-header. The 1-byte minimal
    /// version produced by [`build_minimal_keyframe`] is too small
    /// for the bool reader to parse without crossing EOF; this
    /// helper gives the parser breathing room.
    fn build_lossless_keyframe_with_compressed_size(n: u16) -> Vec<u8> {
        let mut buf = vec![0u8; 256];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_literal(sync::FRAME_MARKER as u32, 2);
        w.write_bit(0); // profile low
        w.write_bit(0); // profile high → Profile 0
        w.write_bit(0); // show_existing = 0
        w.write_bit(0); // frame_type = KEY
        w.write_bit(1); // show_frame = 1
        w.write_bit(0); // error_resilient = 0
        w.write_literal(sync::SYNC_CODE_0 as u32, 8);
        w.write_literal(sync::SYNC_CODE_1 as u32, 8);
        w.write_literal(sync::SYNC_CODE_2 as u32, 8);
        w.write_literal(ColorSpace::Bt601 as u32, 3);
        w.write_bit(0); // color_range = Studio
        w.write_literal(319, 16);
        w.write_literal(239, 16);
        w.write_bit(0); // no render size
        // Tail.
        w.write_bit(0);
        w.write_bit(0);
        w.write_literal(0, 2);
        w.write_literal(0, 6);
        w.write_literal(0, 3);
        w.write_bit(0);
        w.write_literal(0, 8);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        w.write_literal(u32::from(n), 16);
        let used = w.bytes_written();
        buf.truncate(used + n as usize + 4);
        buf
    }

    /// `parse_compressed_partition` on a lossless keyframe with a
    /// 16-byte zero-padded compressed-header (plenty of room for
    /// the bool reader to issue all "no-update" gates):
    /// - Returns `tx_mode = OnlyTx4x4` (lossless forces this).
    /// - Returns `reference_mode = Single` (intra branch).
    /// - Leaves the selected FrameContext bank at its libvpx defaults.
    #[test]
    fn parse_compressed_partition_lossless_keyframe_no_updates() {
        use rvp9_core::bitstream::ReferenceMode;
        use rvp9_core::transforms::TxMode;

        let buf = build_lossless_keyframe_with_compressed_size(16);
        let mut dec = Decoder::new();

        let frames = dec.parse_packet(&buf).expect("valid keyframe");
        let header = frames.frames()[0].header.clone();
        let header_end = frames.frames()[0].header_end_byte;
        let frame_data: Vec<u8> = frames.frames()[0].full_frame_data.to_vec();
        drop(frames);

        let parsed = ParsedFrame {
            header,
            header_end_byte: header_end,
            full_frame_data: &frame_data,
        };

        let original = dec.frame_context(0).clone();
        let result = dec
            .parse_compressed_partition(&parsed)
            .expect("compressed header parse");

        assert_eq!(result.tx_mode, TxMode::OnlyTx4x4);
        assert_eq!(result.reference_mode, ReferenceMode::Single);
        assert_eq!(dec.frame_context(0), &original);
    }

    /// `parse_compressed_partition` on a `ShowExistingFrame` header
    /// fails with `WrongFrameKind` (those frames have no tail and
    /// no compressed header).
    #[test]
    fn parse_compressed_partition_show_existing_rejected() {
        // Build a tiny `show_existing_frame` packet: 1 byte with the
        // marker (0b10) + show_existing_flag (1) + slot (0..7=3 bits).
        let mut buf = vec![0u8; 16];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_literal(sync::FRAME_MARKER as u32, 2);
        w.write_bit(0); // profile low
        w.write_bit(0); // profile high → Profile 0
        w.write_bit(1); // show_existing = 1
        w.write_literal(2, 3); // slot = 2
        let used = w.bytes_written();
        buf.truncate(used);

        let mut dec = Decoder::new();
        let frames = dec.parse_packet(&buf).expect("valid show-existing");
        let header = frames.frames()[0].header.clone();
        let header_end = frames.frames()[0].header_end_byte;
        let frame_data: Vec<u8> = frames.frames()[0].full_frame_data.to_vec();
        drop(frames);

        let parsed = ParsedFrame {
            header,
            header_end_byte: header_end,
            full_frame_data: &frame_data,
        };
        let err = dec
            .parse_compressed_partition(&parsed)
            .expect_err("must fail");
        assert_eq!(err, DecodeError::WrongFrameKind);
    }

    #[test]
    fn parse_packet_superframe_splits_into_frames() {
        // Build a 2-frame superframe: two copies of the keyframe.
        let k = build_minimal_keyframe();
        // Strip trailing padding to a real frame size (the header
        // parser doesn't care about trailing bytes, but the
        // superframe builder needs accurate sizes).
        let frame_size = k.len();
        // Build superframe: frame A | frame B | trailing index.
        let mut packet = Vec::new();
        packet.extend_from_slice(&k);
        packet.extend_from_slice(&k);
        // mag = 2 (since frame_size > 255 for this header). Build
        // the trailing index manually.
        let mag = 2u8;
        let frames_count = 2u8;
        let marker = 0xc0 | ((mag - 1) << 3) | (frames_count - 1);
        packet.push(marker);
        // Size fields (little-endian, mag bytes each).
        for _ in 0..frames_count {
            packet.push((frame_size & 0xff) as u8);
            packet.push(((frame_size >> 8) & 0xff) as u8);
        }
        packet.push(marker);

        let mut dec = Decoder::new();
        let frames = dec.parse_packet(&packet).expect("valid superframe");
        assert_eq!(frames.count(), 2);
        for f in frames.frames() {
            assert_eq!(f.header.kind, FrameKind::KeyFrame);
            assert_eq!(f.header.width, 320);
            assert_eq!(f.header.height, 240);
        }
    }

    #[test]
    fn merge_lf_params_applies_only_some_slots() {
        let mut dec = Decoder::new();
        let lf = rvp9_core::bitstream::LoopFilterParams {
            filter_level: 30,
            sharpness_level: 0,
            mode_ref_delta_enabled: true,
            mode_ref_delta_update: true,
            ref_deltas: [Some(2), None, Some(-3), None],
            mode_deltas: [None, Some(1)],
        };
        dec.merge_lf_params(&lf);
        let deltas = dec.lf_deltas();
        assert!(deltas.enabled);
        assert_eq!(deltas.ref_deltas[0], 2);
        assert_eq!(deltas.ref_deltas[1], 0); // None → keep previous (0).
        assert_eq!(deltas.ref_deltas[2], -3);
        assert_eq!(deltas.ref_deltas[3], 0);
        assert_eq!(deltas.mode_deltas[0], 0);
        assert_eq!(deltas.mode_deltas[1], 1);
    }

    /// Build a single byte-buffer for a `show_existing_frame` header
    /// pointing at the given slot. Profile 0, marker correct,
    /// show_existing = 1, then 3 bits for the slot.
    fn build_show_existing_frame(slot: u8) -> Vec<u8> {
        let mut buf = vec![0u8; 16];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_literal(sync::FRAME_MARKER as u32, 2);
        w.write_bit(0); // profile low
        w.write_bit(0); // profile high → Profile 0
        w.write_bit(1); // show_existing_frame = 1
        w.write_literal(slot as u32, 3);
        // Pad.
        for _ in 0..4 {
            w.write_literal(0, 8);
        }
        let used = w.bytes_written();
        buf.truncate(used + 4);
        buf
    }

    #[test]
    fn decode_show_existing_frame_emits_slot() {
        use rvp9_core::bitstream::ChromaFormat;
        use rvp9_core::frame::{BorderUsage, ByteAlignment, FrameBuf, Yv12Buffer};
        use std::sync::Arc;

        let mut dec = Decoder::new();
        // G12 refactor: `ref_frame_map[i]` now holds an
        // `Option<u8>` pool slot index, not an `Arc` directly. To
        // wire a FrameBuf into `ref_frame_map[4]`:
        //   (1) park the FrameBuf in the frame_pool (any slot; we
        //       use pool slot 6 here to keep the test honest — i.e.,
        //       NOT relying on the historical invariant that the
        //       ref_frame_map slot index matches the pool slot index).
        //   (2) write that pool slot index into ref_frame_map[4].
        // `decode_show_existing_frame` then resolves
        // `ref_frame_map[slot=4] -> Some(6) -> frame_pool.get(6) ->
        // FrameBuf`. Pre-G12 the two indices were identical only
        // because of the mirror-write hack the refactor removed.
        let f: FrameBuf = Arc::new(
            Yv12Buffer::alloc(
                320,
                240,
                ChromaFormat::Yuv420,
                BorderUsage::Decoder,
                ByteAlignment::Legacy,
            )
            .unwrap(),
        );
        dec.frame_pool_mut().assign_slot(6, f.clone()).unwrap();
        dec.set_ref_slot(4, 6);

        // Build a packet pointing at ref slot 4.
        let buf = build_show_existing_frame(4);
        let parsed = dec.parse_packet(&buf).expect("valid");
        assert_eq!(parsed.count(), 1);
        let frame = &parsed.frames()[0];
        assert_eq!(frame.header.kind, FrameKind::ShowExistingFrame { slot: 4 });

        // Decode resolves through ref_frame_map → frame_pool and
        // emits the originally-installed FrameBuf.
        let out = dec.decode_show_existing_frame(&frame.header).unwrap();
        assert!(Arc::ptr_eq(&out, &f));
    }

    #[test]
    fn decode_show_existing_frame_empty_slot_rejected() {
        let mut dec = Decoder::new();
        // No slots populated.
        let buf = build_show_existing_frame(2);
        let parsed = dec.parse_packet(&buf).unwrap();
        let frame = &parsed.frames()[0];
        let err = dec.decode_show_existing_frame(&frame.header).unwrap_err();
        assert_eq!(err, DecodeError::EmptySlot { slot: 2 });
    }

    #[test]
    fn decode_show_existing_frame_rejects_non_show_existing() {
        let mut dec = Decoder::new();
        let buf = build_minimal_keyframe();
        let parsed = dec.parse_packet(&buf).unwrap();
        let kf = &parsed.frames()[0];
        let err = dec.decode_show_existing_frame(&kf.header).unwrap_err();
        assert_eq!(err, DecodeError::WrongFrameKind);
    }

    /// G3 (decoder-hardening V6 §G3): an inter frame whose
    /// `ref_frame_indices` point at an empty `ref_frame_map` slot
    /// surfaces as `DecodeError::EmptyInterRef { slot }`, not the
    /// pre-G3 misleading `Unimplemented` variant. The sibling
    /// variant `EmptySlot` continues to be used by the
    /// `decode_show_existing_frame` path (see
    /// `decode_show_existing_frame_empty_slot_rejected` above).
    ///
    /// The test constructs a `ParsedFrame` directly (no packet
    /// round-trip needed) — the ref-resolution loop runs before any
    /// compressed-header or tile-byte parsing, so the body bytes are
    /// irrelevant.
    #[test]
    fn decode_inter_frame_empty_ref_slot_rejected() {
        use rvp9_core::bitstream::header::HeaderTail;
        use rvp9_core::bitstream::sub_parsers::{
            LoopFilterParams, MAX_PROB, PREDICTION_PROBS, QuantizationParams, SEG_TREE_PROBS,
            SegmentationParams, TileInfo,
        };
        use rvp9_core::bitstream::{
            BitDepth, ChromaFormat, ColorRange, ColorSpace, InterpFilter, Profile,
            UncompressedHeader,
        };

        let mut dec = Decoder::new();
        // No keyframe decoded — every `ref_frame_map` slot is `None`.

        let tail = HeaderTail {
            refresh_frame_context: true,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams {
                filter_level: 0,
                sharpness_level: 0,
                mode_ref_delta_enabled: false,
                mode_ref_delta_update: false,
                ref_deltas: [None; 4],
                mode_deltas: [None; 2],
            },
            quantization: QuantizationParams {
                base_qindex: 50,
                y_dc_delta_q: 0,
                uv_dc_delta_q: 0,
                uv_ac_delta_q: 0,
            },
            segmentation: SegmentationParams {
                enabled: false,
                update_map: false,
                temporal_update: false,
                tree_probs: [MAX_PROB; SEG_TREE_PROBS],
                pred_probs: [MAX_PROB; PREDICTION_PROBS],
                update_data: false,
                abs_delta: false,
                feature_data: Default::default(),
            },
            tile_info: TileInfo {
                log2_tile_cols: 0,
                log2_tile_rows: 0,
            },
            first_partition_size: 0,
        };
        let header = UncompressedHeader {
            kind: FrameKind::InterFrame {
                reset_frame_context: 0,
                refresh_frame_flags: 0x00,
                // LAST/GOLDEN/ALTREF all point at empty slot 3.
                // The decoder reports the *first* unresolved slot
                // (LAST), so the expected payload is `slot: 3`.
                ref_frame_indices: [3, 3, 3],
                ref_frame_sign_biases: [false, false, false],
                allow_high_precision_mv: false,
                interp_filter: InterpFilter::EightTap,
            },
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(tail),
        };
        // `full_frame_data` is never read on this path — the
        // ref-resolution loop at lib.rs:1252 fires before any byte
        // of the compressed header / tiles is consumed. Use an
        // empty slice.
        let empty: [u8; 0] = [];
        let parsed = ParsedFrame {
            header,
            header_end_byte: 0,
            full_frame_data: &empty,
        };

        let err = dec.decode_frame(&parsed).unwrap_err();
        assert_eq!(err, DecodeError::EmptyInterRef { slot: 3 });
    }

    /// G3 (DA Item 4): the empty-ref-rejection loop must report the
    /// *slot index* named by `ref_frame_indices[i]`, not the loop
    /// iteration index `i`. Construct a case where slots 0 and 1 are
    /// populated but slot 5 is empty, and the inter frame's
    /// `ref_frame_indices = [0, 5, 0]`. The loop should resolve [0]
    /// (LAST = slot 0, OK), then hit [1] (GOLDEN = slot 5, empty),
    /// and return `EmptyInterRef { slot: 5 }` — proving the payload
    /// carries the ref-map slot, not the iteration counter `i = 1`.
    #[test]
    fn decode_inter_frame_empty_ref_slot_rejected_reports_correct_slot() {
        use rvp9_core::bitstream::header::HeaderTail;
        use rvp9_core::bitstream::sub_parsers::{
            LoopFilterParams, MAX_PROB, PREDICTION_PROBS, QuantizationParams, SEG_TREE_PROBS,
            SegmentationParams, TileInfo,
        };
        use rvp9_core::bitstream::{
            BitDepth, ChromaFormat, ColorRange, ColorSpace, InterpFilter, Profile,
            UncompressedHeader,
        };
        use rvp9_core::frame::{BorderUsage, ByteAlignment, FrameBuf, Yv12Buffer};
        use std::sync::Arc;

        let mut dec = Decoder::new();
        // Pre-populate ref_frame_map slots 0 and 1 with real FrameBufs
        // (slot 5 stays `None`). These tests share the crate; the
        // `set_ref_slot` accessor names the structural invariant the
        // raw `ref_frame_map[i] = Some(_)` write would have obscured
        // (see `Decoder::frame_pool` doc-comment).
        //
        // G12 refactor: `ref_frame_map[i]` holds an `Option<u8>`
        // pool slot index; park each FrameBuf in `frame_pool` and
        // wire the slot index into the ref_frame_map entry.
        let make_buf = || -> FrameBuf {
            Arc::new(
                Yv12Buffer::alloc(
                    64,
                    64,
                    ChromaFormat::Yuv420,
                    BorderUsage::Decoder,
                    ByteAlignment::Legacy,
                )
                .unwrap(),
            )
        };
        dec.frame_pool_mut().assign_slot(0, make_buf()).unwrap();
        dec.frame_pool_mut().assign_slot(1, make_buf()).unwrap();
        dec.set_ref_slot(0, 0);
        dec.set_ref_slot(1, 1);
        // dec.ref_frame_map[5] stays None — this is the empty slot.

        let tail = HeaderTail {
            refresh_frame_context: true,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams {
                filter_level: 0,
                sharpness_level: 0,
                mode_ref_delta_enabled: false,
                mode_ref_delta_update: false,
                ref_deltas: [None; 4],
                mode_deltas: [None; 2],
            },
            quantization: QuantizationParams {
                base_qindex: 50,
                y_dc_delta_q: 0,
                uv_dc_delta_q: 0,
                uv_ac_delta_q: 0,
            },
            segmentation: SegmentationParams {
                enabled: false,
                update_map: false,
                temporal_update: false,
                tree_probs: [MAX_PROB; SEG_TREE_PROBS],
                pred_probs: [MAX_PROB; PREDICTION_PROBS],
                update_data: false,
                abs_delta: false,
                feature_data: Default::default(),
            },
            tile_info: TileInfo {
                log2_tile_cols: 0,
                log2_tile_rows: 0,
            },
            first_partition_size: 0,
        };
        let header = UncompressedHeader {
            kind: FrameKind::InterFrame {
                reset_frame_context: 0,
                refresh_frame_flags: 0x00,
                // LAST = slot 0 (populated), GOLDEN = slot 5 (empty),
                // ALTREF = slot 0 (populated). The loop should fail at
                // iteration i=1 and report `slot: 5`, not `slot: 1`.
                ref_frame_indices: [0, 5, 0],
                ref_frame_sign_biases: [false, false, false],
                allow_high_precision_mv: false,
                interp_filter: InterpFilter::EightTap,
            },
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(tail),
        };
        let empty: [u8; 0] = [];
        let parsed = ParsedFrame {
            header,
            header_end_byte: 0,
            full_frame_data: &empty,
        };

        let err = dec.decode_frame(&parsed).unwrap_err();
        assert_eq!(err, DecodeError::EmptyInterRef { slot: 5 });
    }

    /// G3 (DA Item 5): regression guard against an accidental merge
    /// of [`DecodeError::EmptySlot`] and [`DecodeError::EmptyInterRef`].
    /// Even at identical slot indices the two variants must compare
    /// non-equal — they cover distinct decoder paths
    /// (show-existing-frame vs. inter-ref resolution) and any future
    /// refactor that collapses them would silently regress the
    /// diagnostic separation introduced by G3.
    #[test]
    fn empty_slot_and_empty_inter_ref_with_same_slot_are_not_equal() {
        assert_ne!(
            DecodeError::EmptySlot { slot: 3 },
            DecodeError::EmptyInterRef { slot: 3 },
        );
    }

    /// G7.0 review (2026-05-21): companion guard to
    /// `empty_slot_and_empty_inter_ref_with_same_slot_are_not_equal` —
    /// `UnsupportedProfile` and `Unimplemented` are siblings (both
    /// cover "this decoder cannot finish the frame") but must compare
    /// non-equal so the K-5 harness can bucket them independently.
    /// Any future refactor that collapses them would silently regress
    /// the diagnostic separation.
    #[test]
    fn unsupported_profile_and_unimplemented_are_not_equal() {
        assert_ne!(
            DecodeError::UnsupportedProfile { profile: 1 },
            DecodeError::Unimplemented {
                reason: "test reason",
            },
        );
    }

    /// Stage 3 stretch D.6: backward-adapt `nmvc` end-of-inter-frame
    /// shifts the bank away from defaults when counts accumulate.
    /// Verifies the plumbing: a synthetic count bump on the joint
    /// "NEWMV used horizontally" tally feeds `adapt_mv_probs` and
    /// pushes the `joints` probabilities toward libvpx's expected
    /// merged value.
    #[test]
    fn adapt_mv_probs_post_inter_frame_shifts_nmvc() {
        use rvp9_core::entropy::adapt::adapt_mv_probs;
        use rvp9_core::entropy::mv_probs::NmvCounts;
        let mut counts = NmvCounts::default();
        counts.joints[1] = 10; // 10 horizontal-only MV reads
        counts.joints[3] = 5; // 5 both-component reads
        let pre = rvp9_core::entropy::mv_probs::DEFAULT_NMV_CONTEXT;
        let post = adapt_mv_probs(&pre, &counts, /* allow_hp */ true);
        // Some prob in joints[] must differ — counts are non-zero so
        // mode-merge moves the prob from its default.
        assert!(
            pre.joints != post.joints,
            "expected adapt_mv_probs to shift joints with non-zero counts"
        );
    }

    /// Stage 3 Task E.1.b: a fresh [`Decoder`] starts with all-zero
    /// `coef_counts` and `last_frame_was_key == false`. Tests the
    /// Decoder-level state invariants the adapt pipeline relies on.
    #[test]
    fn decoder_new_initializes_coef_counts_zero() {
        use rvp9_core::entropy::adapt::CoefCountsBank;
        let dec = Decoder::new();
        // Every coef cell is zero on construction.
        assert_eq!(dec.coef_counts, CoefCountsBank::default());
        // Every eob_branch cell is zero too (CoefCountsBank::default
        // sets both axes to zero — pin the round-trip).
        for tx in dec.coef_counts.coef.iter() {
            for plane in tx.iter() {
                for ref_ty in plane.iter() {
                    for band in ref_ty.iter() {
                        for ctx in band.iter() {
                            assert_eq!(ctx, &[0u32; 4]);
                        }
                    }
                }
            }
        }
        for tx in dec.coef_counts.eob_branch.iter() {
            for plane in tx.iter() {
                for ref_ty in plane.iter() {
                    for band in ref_ty.iter() {
                        for &cell in band.iter() {
                            assert_eq!(cell, 0);
                        }
                    }
                }
            }
        }
        assert!(!dec.last_frame_was_key);
    }

    /// Stage 3 Task E.1.b: `adapt_coef_probs` fed a [`CoefCountsBank`]
    /// with non-zero entries produces a different bank than the
    /// default. Verifies the adapt kernel sees the count tally we
    /// route to it, independently of the per-frame plumbing.
    #[test]
    fn adapt_coef_probs_with_nonzero_counts_shifts_probs() {
        use rvp9_core::entropy::adapt::{CoefCountsBank, adapt_coef_probs, coef_adapt_params};
        let pre = FrameContext::new_default();
        let mut counts = CoefCountsBank::default();
        // Inject a single (tx=Tx4x4, plane=Y, ref=intra, band=1, ctx=0)
        // tally — n0=3, n1=1, n2=1, neob=5, eob_total=10 (same shape
        // as the adapt.rs hand-trace test).
        counts.coef[0][0][0][1][0][rvp9_core::entropy::ZERO_TOKEN] = 3;
        counts.coef[0][0][0][1][0][rvp9_core::entropy::ONE_TOKEN] = 1;
        counts.coef[0][0][0][1][0][rvp9_core::entropy::TWO_TOKEN] = 1;
        counts.coef[0][0][0][1][0][rvp9_core::entropy::EOB_MODEL_TOKEN] = 5;
        counts.eob_branch[0][0][0][1][0] = 10;

        let (cs, uf) = coef_adapt_params(true, false); // intra-only keyframe
        let post = adapt_coef_probs(&pre.coef_probs, &counts, cs, uf);
        // The injected slot must differ.
        assert_ne!(
            post[0][0][0][1][0], pre.coef_probs[0][0][0][1][0],
            "expected adapted (tx=4x4, plane=Y, intra, band=1, ctx=0) slot to differ"
        );
        // A slot we didn't touch should match the default.
        assert_eq!(
            post[3][1][1][5][5], pre.coef_probs[3][1][1][5][5],
            "untouched (tx=32x32, UV, inter, band=5, ctx=5) slot must be unchanged"
        );
    }

    /// Stage 3 Task E.1.b: synthetic counts fed directly through the
    /// adapt kernel + write-back sequence (mirroring the end-of-frame
    /// adapt step) shift the active FrameContext's `coef_probs` and
    /// leave the `coef_counts` field on the Decoder unchanged on a
    /// keyframe-success run. This verifies the integration handshake
    /// between the bank and the adapt entry point without depending
    /// on `build_minimal_keyframe` running a full conformance-class
    /// decode (the minimal-keyframe synthesis exists for header
    /// round-trips, not full tile execution).
    #[test]
    fn coef_counts_adapt_round_trip_via_decoder_fields() {
        use rvp9_core::entropy::adapt::{adapt_coef_probs, coef_adapt_params};
        let mut dec = Decoder::new();
        // Seed the bank — mirrors what the per-block plumbing would
        // do for a single (Tx4x4, Y, intra, band=2, ctx=1) decision
        // chain on a real keyframe. Use ZERO_TOKEN / ONE_TOKEN /
        // TWO_TOKEN / EOB_MODEL_TOKEN tallies on a non-default cell so
        // the adapt step can't return the input unchanged.
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::ZERO_TOKEN] = 7;
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::ONE_TOKEN] = 3;
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::TWO_TOKEN] = 1;
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::EOB_MODEL_TOKEN] = 2;
        dec.coef_counts.eob_branch[0][0][0][2][1] = 5;

        // Run the keyframe adapt step inline — matches the post-tile
        // adapt block in `decode_keyframe`.
        let (cs, uf) = coef_adapt_params(/* intra_only = */ true, dec.last_frame_was_key);
        let pre_defaults = FrameContext::new_default().coef_probs;
        let new_probs =
            adapt_coef_probs(&dec.frame_contexts[0].coef_probs, &dec.coef_counts, cs, uf);
        dec.frame_contexts[0].coef_probs = new_probs;
        dec.last_frame_was_key = true;

        // Adapt must have shifted the seeded slot.
        assert_ne!(
            dec.frame_contexts[0].coef_probs[0][0][0][2][1], pre_defaults[0][0][0][2][1],
            "the seeded (tx=4x4, Y, intra, band=2, ctx=1) slot must shift after adapt"
        );
        // And `last_frame_was_key` is true now (driver responsibility).
        assert!(dec.last_frame_was_key);
    }

    /// Stage 3 Task E.1.c.1: a fresh [`Decoder`] starts with all-zero
    /// `mode_counts`. Mirrors the analogous coef-counts init test.
    #[test]
    fn decoder_new_initializes_mode_counts_zero() {
        use rvp9_core::entropy::adapt::FrameCounts;
        let dec = Decoder::new();
        assert_eq!(dec.mode_counts, FrameCounts::default());
    }

    /// Stage 3 Task E.1.c.1: feed synthetic `mode_counts` through
    /// `adapt_mode_probs` + write-back and verify the active
    /// FrameContext's mode-related fields shift. Mirrors the
    /// `coef_counts_adapt_round_trip_via_decoder_fields` pattern for
    /// the third (and final) adapt kernel.
    #[test]
    fn mode_counts_adapt_round_trip_via_decoder_fields() {
        use rvp9_core::entropy::adapt::adapt_mode_probs;
        let mut dec = Decoder::new();
        // Inject a non-zero skip[0] tally — synthesises one block per
        // skip-context decision. `adapt_mode_probs` uses
        // `mode_mv_merge_probs` for skip; non-zero counts ⇒ shift.
        dec.mode_counts.skip[0] = [4, 6];
        // Inject a non-zero intra_inter[1] tally too — exercises a
        // second `mode_mv_merge_probs` slot.
        dec.mode_counts.intra_inter[1] = [5, 3];

        let pre = dec.frame_contexts[0].clone();
        let new_fc = adapt_mode_probs(
            &dec.frame_contexts[0],
            &dec.mode_counts,
            /* interp_switchable = */ false,
            /* tx_mode_select   = */ false,
        );
        dec.frame_contexts[0] = new_fc;

        assert_ne!(
            dec.frame_contexts[0].skip_probs[0], pre.skip_probs[0],
            "skip_probs[0] must shift after adapt with non-zero counts"
        );
        assert_ne!(
            dec.frame_contexts[0].intra_inter_prob[1], pre.intra_inter_prob[1],
            "intra_inter_prob[1] must shift after adapt with non-zero counts"
        );
        // coef_probs / nmvc must be unchanged — adapt_mode_probs copies
        // them through (separate kernels handle those).
        assert_eq!(dec.frame_contexts[0].coef_probs, pre.coef_probs);
        assert_eq!(dec.frame_contexts[0].nmvc, pre.nmvc);
    }

    /// Stage 3 stretch D.9: the backward-adapt `pre` baseline must be
    /// the FrameContext value *before* the compressed-header diff-updates
    /// land, not after. Mirrors libvpx's
    /// `pre_fc = cm->frame_contexts[cm->frame_context_idx]` in
    /// `adapt_coef_probs` (`vp9_entropy.c:1058`) — `pre_fc` reads from
    /// `cm->frame_contexts[fc_idx]` while `cm->fc` (the working copy
    /// that the compressed-header parser mutates) is separate. Without
    /// the pre-snapshot, our adapt would chain-merge from `bank +
    /// current_frame_diff_updates`, accumulating drift across inter
    /// frames and (in the 32×32 conformance vector) corrupting the
    /// bool-coder cursor by frame 5.
    ///
    /// This test exercises the snapshot-vs-mutated-bank distinction at
    /// the adapt kernel level: same counts, same write-back target,
    /// but two different `pre` inputs (snapshot vs. mutated bank)
    /// produce two distinguishable adapt results.
    #[test]
    fn adapt_pre_snapshot_diverges_from_mutated_bank() {
        use rvp9_core::entropy::adapt::{adapt_coef_probs, coef_adapt_params};
        let mut dec = Decoder::new();
        // Seed counts so adapt is forced to merge (zero counts would
        // produce `merge_probs(pre, _, _) == pre`, masking the
        // distinction this test wants to surface).
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::ZERO_TOKEN] = 6;
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::ONE_TOKEN] = 2;
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::TWO_TOKEN] = 1;
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::EOB_MODEL_TOKEN] = 3;
        dec.coef_counts.eob_branch[0][0][0][2][1] = 4;

        // Snapshot the bank BEFORE the simulated compressed-header
        // delta. This is what D.9's `pre_fc` captures.
        let pre_snapshot = dec.frame_contexts[0].clone();

        // Simulate the compressed-header diff-update: bump one
        // coef_probs cell by a small delta directly on the bank.
        // libvpx applies these to `cm->fc` (working copy); we apply
        // them in place on the bank, matching our actual decode path.
        let delta_cell = (0usize, 0, 0, 2, 1, 0);
        let original = dec.frame_contexts[0].coef_probs[delta_cell.0][delta_cell.1][delta_cell.2]
            [delta_cell.3][delta_cell.4][delta_cell.5];
        dec.frame_contexts[0].coef_probs[delta_cell.0][delta_cell.1][delta_cell.2][delta_cell.3]
            [delta_cell.4][delta_cell.5] = original.saturating_sub(20);

        let (cs, uf) = coef_adapt_params(
            /* intra_only = */ false, /* last_was_key = */ true,
        );

        // The D.9 codepath: adapt uses the pre-snapshot as the `pre`
        // baseline.
        let adapt_from_snapshot =
            adapt_coef_probs(&pre_snapshot.coef_probs, &dec.coef_counts, cs, uf);
        // The pre-D.9 (buggy) codepath: adapt would use the post-
        // compressed-header bank as the `pre` baseline.
        let adapt_from_mutated_bank =
            adapt_coef_probs(&dec.frame_contexts[0].coef_probs, &dec.coef_counts, cs, uf);

        // The two must differ — same counts, same merge kernel, but
        // different `pre` inputs ⇒ different merged outputs.
        assert_ne!(
            adapt_from_snapshot[delta_cell.0][delta_cell.1][delta_cell.2][delta_cell.3]
                [delta_cell.4][delta_cell.5],
            adapt_from_mutated_bank[delta_cell.0][delta_cell.1][delta_cell.2][delta_cell.3]
                [delta_cell.4][delta_cell.5],
            "adapt result must differ between snapshot-as-pre and \
             mutated-bank-as-pre — D.9's snapshot guards against the \
             compressed-header-diff drift cascade"
        );
    }

    #[test]
    fn merge_lf_params_disabled_means_no_carry() {
        let mut dec = Decoder::new();
        // First frame: enable + set deltas.
        dec.merge_lf_params(&rvp9_core::bitstream::LoopFilterParams {
            filter_level: 30,
            sharpness_level: 0,
            mode_ref_delta_enabled: true,
            mode_ref_delta_update: true,
            ref_deltas: [Some(5), Some(5), Some(5), Some(5)],
            mode_deltas: [Some(5), Some(5)],
        });
        assert_eq!(dec.lf_deltas().ref_deltas, [5, 5, 5, 5]);

        // Second frame: disabled. Existing deltas survive but
        // `enabled` flips.
        dec.merge_lf_params(&rvp9_core::bitstream::LoopFilterParams::default());
        let deltas = dec.lf_deltas();
        assert!(!deltas.enabled);
        // Persistent storage preserved.
        assert_eq!(deltas.ref_deltas, [5, 5, 5, 5]);
    }

    /// Stage 3 Task E.2: the multi-tile mi-grid stitcher must place
    /// each per-tile grid at the right `(mi_row_start, mi_col_start)`
    /// offset in the frame-wide grid, using the frame's `mi_cols` as
    /// the stride.
    ///
    /// Builds a 16×16 frame split into two 16×8 tiles (1 row × 2 cols),
    /// fills each tile's grid with a distinguishable `ModeInfo` tag,
    /// runs `build_frame_mi_grid`, and verifies that every frame cell
    /// resolves to the correct per-tile origin.
    #[test]
    fn build_frame_mi_grid_stitches_two_tiles_in_row() {
        use crate::tile::TileBounds;
        use rvp9_core::block::BlockSize;
        use rvp9_core::intra_pred::IntraMode;
        use rvp9_core::mode_info::{BlockMode, ModeInfo};
        use rvp9_core::transforms::TxSize;

        let mi_rows = 16usize;
        let mi_cols = 16usize;
        let tile_a_bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 16,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let tile_b_bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 16,
            mi_col_start: 8,
            mi_col_end: 16,
        };
        // Use `segment_id` as a tile tag (1 = tile A, 2 = tile B).
        let mk_mi = |seg: u8| ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: TxSize::Tx4x4,
            skip: false,
            segment_id: seg,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let tile_a_grid: Vec<Option<ModeInfo>> = vec![Some(mk_mi(1)); 16 * 8];
        let tile_b_grid: Vec<Option<ModeInfo>> = vec![Some(mk_mi(2)); 16 * 8];

        let frame = build_frame_mi_grid(
            mi_rows,
            mi_cols,
            &[
                (tile_a_bounds, tile_a_grid.as_slice()),
                (tile_b_bounds, tile_b_grid.as_slice()),
            ],
        );

        assert_eq!(frame.len(), mi_rows * mi_cols);
        // Spot-check: top-left of tile A.
        assert_eq!(frame[0].unwrap().segment_id, 1);
        // Top-right cell of tile A (col 7).
        assert_eq!(frame[7].unwrap().segment_id, 1);
        // First cell of tile B (col 8).
        assert_eq!(frame[8].unwrap().segment_id, 2);
        // Bottom row of tile B.
        assert_eq!(frame[15 * mi_cols + 15].unwrap().segment_id, 2);
        // Every cell on the tile A / tile B border respects the split.
        for r in 0..mi_rows {
            for c in 0..mi_cols {
                let expected_tag = if c < 8 { 1 } else { 2 };
                assert_eq!(
                    frame[r * mi_cols + c].unwrap().segment_id,
                    expected_tag,
                    "mismatch at (r={r}, c={c})"
                );
            }
        }
    }

    /// S4.10 — whole-loop synthesis: build a 64×64 keyframe IVF
    /// packet entirely from rvp9-core encoder primitives, feed it
    /// through `Decoder::parse_packet` + `Decoder::decode_frame`, and
    /// verify the decoder lands on the same `mi_grid` the writer
    /// emitted.
    ///
    /// This is the first end-to-end test exercising every encoder
    /// piece that has landed in Stage 4:
    /// `write_uncompressed_header` (S4.06), `write_compressed_header`
    /// (S4.07), `write_tile_buffers` (S4.09), and the new
    /// `write_tile_intra_keyframe` (S4.10). If this passes, we have a
    /// decodable bitstream produced wholly by our encoder pipeline
    /// from a synthetic `mi_grid`.
    #[test]
    fn s4_10_round_trip_synthetic_64x64_keyframe_through_decoder() {
        use rvp9_core::bitstream::frame::ReferenceMode;
        use rvp9_core::bitstream::header::HeaderTail;
        use rvp9_core::bitstream::sub_parsers::{
            LoopFilterParams, MAX_PROB, PREDICTION_PROBS, QuantizationParams, SEG_TREE_PROBS,
            SegmentationParams, TileInfo,
        };
        use rvp9_core::bitstream::{
            BitDepth, ChromaFormat, ColorRange, ColorSpace, CompressedHeaderOptions,
            CompressedHeaderResult, FrameKind, InterpFilter, ModesTileBounds, Profile,
            RawBitWriter, UncompressedHeader, write_compressed_header, write_tile_intra_keyframe,
            write_uncompressed_header,
        };
        use rvp9_core::block::BlockSize;
        use rvp9_core::entropy::{BoolWriter, FrameContext};
        use rvp9_core::intra_pred::IntraMode;
        use rvp9_core::mode_info::{BlockMode, ModeInfo};
        use rvp9_core::transforms::{TxMode, TxSize};

        // 1. Build the synthetic mi_grid: one 64×64 SB,
        //    PARTITION_NONE, DC_PRED on Y and UV, skip=true.
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let mi_grid: Vec<Option<ModeInfo>> = vec![Some(cell); (mi_rows * mi_cols) as usize];

        // 2. Build the compressed header. tx_mode = AllowTx32x32 so
        //    that the per-block tx_size is implicit (decoder derives
        //    Tx32x32 without consuming bits). pre == post so every
        //    diff-update flag is a single "0" bit.
        let pre_fc_template = FrameContext::new_default();
        let mut pre_fc = pre_fc_template.clone();
        let post_fc = pre_fc.clone();
        let opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: true,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let target = CompressedHeaderResult {
            tx_mode: TxMode::AllowTx32x32,
            reference_mode: ReferenceMode::Single,
        };

        let mut compressed_buf = vec![0u8; 16 * 1024];
        let compressed_len = {
            let mut bw = BoolWriter::start_encode(&mut compressed_buf);
            write_compressed_header(&mut bw, &mut pre_fc, &post_fc, opts, &target)
                .expect("write_compressed_header");
            bw.stop_encode().expect("stop_encode compressed header")
        };
        compressed_buf.truncate(compressed_len);

        // 3. Build the per-tile mode-info partition. Single tile
        //    covers the whole 64×64 frame.
        let tile_bounds = ModesTileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut tile_buf = vec![0u8; 4 * 1024];
        let tile_len = {
            let mut bw = BoolWriter::start_encode(&mut tile_buf);
            let tile_mi_cols =
                (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
            let mut coef_ctx = rvp9_core::coef_ctx::TileCoefCtx::new(tile_mi_cols);
            let coef_source = rvp9_core::bitstream::AllZeroIntraCoefSource::new();
            write_tile_intra_keyframe(
                &mut bw,
                &mi_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                &post_fc,
                None, // AQ.0: seg = None (no segmentation in this test)
                &mut coef_ctx,
                &coef_source,
            )
            .expect("write_tile_intra_keyframe");
            bw.stop_encode().expect("stop_encode tile partition")
        };
        tile_buf.truncate(tile_len);

        // 4. Build the uncompressed header. base_qindex = 50 (not
        //    lossless), segmentation disabled, single tile (log2 cols
        //    = log2 rows = 0). frame_context_idx = 0.
        let tail = HeaderTail {
            refresh_frame_context: true,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams {
                filter_level: 0,
                sharpness_level: 0,
                mode_ref_delta_enabled: false,
                mode_ref_delta_update: false,
                ref_deltas: [None; 4],
                mode_deltas: [None; 2],
            },
            quantization: QuantizationParams {
                base_qindex: 50,
                y_dc_delta_q: 0,
                uv_dc_delta_q: 0,
                uv_ac_delta_q: 0,
            },
            segmentation: SegmentationParams {
                enabled: false,
                update_map: false,
                temporal_update: false,
                tree_probs: [MAX_PROB; SEG_TREE_PROBS],
                pred_probs: [MAX_PROB; PREDICTION_PROBS],
                update_data: false,
                abs_delta: false,
                feature_data: Default::default(),
            },
            tile_info: TileInfo {
                log2_tile_cols: 0,
                log2_tile_rows: 0,
            },
            first_partition_size: compressed_len as u16,
        };
        let hdr = UncompressedHeader {
            kind: FrameKind::KeyFrame,
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(tail),
        };

        let mut hdr_buf = vec![0u8; 256];
        let hdr_len = {
            let mut wb = RawBitWriter::new(&mut hdr_buf);
            write_uncompressed_header(&mut wb, &hdr, compressed_len as u16)
                .expect("write_uncompressed_header");
            wb.finish().expect("RawBitWriter::finish")
        };
        hdr_buf.truncate(hdr_len);

        // 5. Assemble the full frame payload: uncompressed_header +
        //    compressed_header + tile_data. No superframe index for a
        //    single frame.
        let mut packet = Vec::new();
        packet.extend_from_slice(&hdr_buf);
        packet.extend_from_slice(&compressed_buf);
        packet.extend_from_slice(&tile_buf);

        // 6. Feed through the decoder.
        let mut dec = Decoder::new();
        let parsed_frame = {
            let frames = dec
                .parse_packet(&packet)
                .expect("parse_packet must accept synthetic keyframe");
            assert_eq!(frames.count(), 1, "single frame in packet");
            let pf = &frames.frames()[0];
            assert_eq!(pf.header.kind, FrameKind::KeyFrame);
            assert_eq!(pf.header.width, 64);
            assert_eq!(pf.header.height, 64);
            // Capture the header + spans we need; the `ParsedFrame`'s
            // `'a` borrow is on `packet`, which outlives `dec` here.
            (pf.header.clone(), pf.header_end_byte, pf.full_frame_data)
        };
        let (hdr_back, header_end_byte, full_frame_data) = parsed_frame;
        let parsed = ParsedFrame {
            header: hdr_back,
            header_end_byte,
            full_frame_data,
        };

        dec.decode_frame(&parsed)
            .expect("decode_frame must accept synthetic keyframe");

        // 7. Verify the decoder's recovered mi_grid matches what we
        //    emitted.
        let (got_rows, got_cols) = dec.last_frame_mi_dims();
        assert_eq!((got_rows, got_cols), (mi_rows, mi_cols));
        let got_grid = dec.last_frame_mi_grid();
        assert_eq!(got_grid.len(), (mi_rows * mi_cols) as usize);
        // Every cell must be a populated DC-skip 64×64 block. The
        // decoder splats the parent ModeInfo into all 8×8 sub-cells
        // (matches what the writer's `NeighbourCtx::write_block`
        // would do).
        for (i, cell_got) in got_grid.iter().enumerate() {
            let mi_row = i / (mi_cols as usize);
            let mi_col = i % (mi_cols as usize);
            let cell = cell_got.expect("every cell in the 1-SB frame must be populated");
            assert_eq!(
                cell.block_size,
                BlockSize::Block64x64,
                "block_size @ ({mi_row}, {mi_col})"
            );
            assert!(cell.skip, "skip @ ({mi_row}, {mi_col})");
            assert_eq!(cell.segment_id, 0, "segment_id @ ({mi_row}, {mi_col})");
            assert_eq!(
                cell.tx_size,
                TxSize::Tx32x32,
                "tx_size @ ({mi_row}, {mi_col}) — tx_mode = AllowTx32x32 forces Tx32x32"
            );
            match cell.mode {
                BlockMode::Intra { y_mode, uv_mode } => {
                    assert_eq!(y_mode, IntraMode::Dc, "y_mode @ ({mi_row}, {mi_col})");
                    assert_eq!(uv_mode, IntraMode::Dc, "uv_mode @ ({mi_row}, {mi_col})");
                }
                BlockMode::Inter { .. } => panic!("inter mode @ ({mi_row}, {mi_col})"),
            }
        }
    }

    /// S4.13 — whole-loop synthesis: build a 64x64 keyframe whose
    /// single SB is encoded with `PARTITION_SPLIT` into four 32x32
    /// `PARTITION_NONE` DC-skip leaves; feed through the full
    /// decoder; assert the recovered `mi_grid` has every 8x8 cell
    /// splatted with `block_size == Block32x32` and DC intra-skip.
    ///
    /// Exercises the recursive partition-tree writer landed in S4.13
    /// end-to-end against the live decoder, proving that the
    /// `write_tile_intra_keyframe` widening doesn't regress the
    /// every-PARTITION_NONE pipeline and DOES produce a valid
    /// bitstream for the SPLIT-then-NONE pattern.
    #[test]
    fn s4_13_round_trip_partition_split_64x64_through_decoder() {
        use rvp9_core::bitstream::frame::ReferenceMode;
        use rvp9_core::bitstream::header::HeaderTail;
        use rvp9_core::bitstream::sub_parsers::{
            LoopFilterParams, MAX_PROB, PREDICTION_PROBS, QuantizationParams, SEG_TREE_PROBS,
            SegmentationParams, TileInfo,
        };
        use rvp9_core::bitstream::{
            BitDepth, ChromaFormat, ColorRange, ColorSpace, CompressedHeaderOptions,
            CompressedHeaderResult, FrameKind, InterpFilter, ModesTileBounds, Profile,
            RawBitWriter, UncompressedHeader, write_compressed_header, write_tile_intra_keyframe,
            write_uncompressed_header,
        };
        use rvp9_core::block::BlockSize;
        use rvp9_core::entropy::{BoolWriter, FrameContext};
        use rvp9_core::intra_pred::IntraMode;
        use rvp9_core::mode_info::{BlockMode, ModeInfo};
        use rvp9_core::transforms::{TxMode, TxSize};

        // 1. Build the synthetic mi_grid: one 64x64 SB partitioned
        //    PARTITION_SPLIT into four 32x32 leaves (each DC-skip).
        //    The decoder's splat convention requires every 8x8 cell
        //    inside a 32x32 leaf to carry the same ModeInfo.
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let cell32 = ModeInfo {
            block_size: BlockSize::Block32x32,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let mut mi_grid: Vec<Option<ModeInfo>> = vec![None; (mi_rows * mi_cols) as usize];
        for &(r0, c0) in &[(0usize, 0usize), (0, 4), (4, 0), (4, 4)] {
            for dr in 0..4 {
                for dc in 0..4 {
                    mi_grid[(r0 + dr) * (mi_cols as usize) + (c0 + dc)] = Some(cell32);
                }
            }
        }

        // 2. Compressed header — same shape as the S4.10 round-trip
        //    test. `tx_mode = AllowTx32x32` so per-block tx_size is
        //    implicit; pre == post so every diff-update is a single
        //    "0" bit.
        let pre_fc_template = FrameContext::new_default();
        let mut pre_fc = pre_fc_template.clone();
        let post_fc = pre_fc.clone();
        let opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: true,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let target = CompressedHeaderResult {
            tx_mode: TxMode::AllowTx32x32,
            reference_mode: ReferenceMode::Single,
        };

        let mut compressed_buf = vec![0u8; 16 * 1024];
        let compressed_len = {
            let mut bw = BoolWriter::start_encode(&mut compressed_buf);
            write_compressed_header(&mut bw, &mut pre_fc, &post_fc, opts, &target)
                .expect("write_compressed_header");
            bw.stop_encode().expect("stop_encode compressed header")
        };
        compressed_buf.truncate(compressed_len);

        // 3. Tile data: single tile, the full 64x64 frame.
        let tile_bounds = ModesTileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut tile_buf = vec![0u8; 4 * 1024];
        let tile_len = {
            let mut bw = BoolWriter::start_encode(&mut tile_buf);
            let tile_mi_cols =
                (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
            let mut coef_ctx = rvp9_core::coef_ctx::TileCoefCtx::new(tile_mi_cols);
            let coef_source = rvp9_core::bitstream::AllZeroIntraCoefSource::new();
            write_tile_intra_keyframe(
                &mut bw,
                &mi_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                &post_fc,
                None, // AQ.0: seg = None (no segmentation in this test)
                &mut coef_ctx,
                &coef_source,
            )
            .expect("write_tile_intra_keyframe");
            bw.stop_encode().expect("stop_encode tile partition")
        };
        tile_buf.truncate(tile_len);

        // 4. Uncompressed header — same shape as the S4.10 round-trip.
        let tail = HeaderTail {
            refresh_frame_context: true,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams {
                filter_level: 0,
                sharpness_level: 0,
                mode_ref_delta_enabled: false,
                mode_ref_delta_update: false,
                ref_deltas: [None; 4],
                mode_deltas: [None; 2],
            },
            quantization: QuantizationParams {
                base_qindex: 50,
                y_dc_delta_q: 0,
                uv_dc_delta_q: 0,
                uv_ac_delta_q: 0,
            },
            segmentation: SegmentationParams {
                enabled: false,
                update_map: false,
                temporal_update: false,
                tree_probs: [MAX_PROB; SEG_TREE_PROBS],
                pred_probs: [MAX_PROB; PREDICTION_PROBS],
                update_data: false,
                abs_delta: false,
                feature_data: Default::default(),
            },
            tile_info: TileInfo {
                log2_tile_cols: 0,
                log2_tile_rows: 0,
            },
            first_partition_size: compressed_len as u16,
        };
        let hdr = UncompressedHeader {
            kind: FrameKind::KeyFrame,
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(tail),
        };

        let mut hdr_buf = vec![0u8; 256];
        let hdr_len = {
            let mut wb = RawBitWriter::new(&mut hdr_buf);
            write_uncompressed_header(&mut wb, &hdr, compressed_len as u16)
                .expect("write_uncompressed_header");
            wb.finish().expect("RawBitWriter::finish")
        };
        hdr_buf.truncate(hdr_len);

        // 5. Assemble the full frame payload.
        let mut packet = Vec::new();
        packet.extend_from_slice(&hdr_buf);
        packet.extend_from_slice(&compressed_buf);
        packet.extend_from_slice(&tile_buf);

        // 6. Feed through the decoder.
        let mut dec = Decoder::new();
        let parsed_frame = {
            let frames = dec
                .parse_packet(&packet)
                .expect("parse_packet must accept synthetic keyframe");
            assert_eq!(frames.count(), 1);
            let pf = &frames.frames()[0];
            assert_eq!(pf.header.kind, FrameKind::KeyFrame);
            (pf.header.clone(), pf.header_end_byte, pf.full_frame_data)
        };
        let (hdr_back, header_end_byte, full_frame_data) = parsed_frame;
        let parsed = ParsedFrame {
            header: hdr_back,
            header_end_byte,
            full_frame_data,
        };
        dec.decode_frame(&parsed)
            .expect("decode_frame must accept SPLIT-partition keyframe");

        // 7. Verify the decoder's recovered mi_grid matches what we
        //    emitted. Every 8x8 cell should carry block_size ==
        //    Block32x32 and intra DC-skip.
        let (got_rows, got_cols) = dec.last_frame_mi_dims();
        assert_eq!((got_rows, got_cols), (mi_rows, mi_cols));
        let got_grid = dec.last_frame_mi_grid();
        assert_eq!(got_grid.len(), (mi_rows * mi_cols) as usize);
        for (i, cell_got) in got_grid.iter().enumerate() {
            let mi_row = i / (mi_cols as usize);
            let mi_col = i % (mi_cols as usize);
            let cell = cell_got.expect("every cell of the SPLIT-partitioned SB must be populated");
            assert_eq!(
                cell.block_size,
                BlockSize::Block32x32,
                "block_size @ ({mi_row}, {mi_col}) — must be 32x32 leaf"
            );
            assert!(cell.skip, "skip @ ({mi_row}, {mi_col})");
            match cell.mode {
                BlockMode::Intra { y_mode, uv_mode } => {
                    assert_eq!(y_mode, IntraMode::Dc, "y_mode @ ({mi_row}, {mi_col})");
                    assert_eq!(uv_mode, IntraMode::Dc, "uv_mode @ ({mi_row}, {mi_col})");
                }
                BlockMode::Inter { .. } => panic!("inter mode @ ({mi_row}, {mi_col})"),
            }
        }
    }

    /// **S4.16d** — first end-to-end inter-frame round-trip through
    /// the live decoder. Builds a 64×64 keyframe (PARTITION_NONE,
    /// intra DC-skip) to prime pool slot 0, then a 64×64 inter frame
    /// referencing that keyframe as LAST (PARTITION_NONE, inter
    /// LASTFRAME ZEROMV, skip = true). Feeds both packets sequentially
    /// through the same `Decoder` and asserts the recovered inter
    /// mi_grid carries the expected `BlockMode::Inter` payload in
    /// every cell.
    ///
    /// Sibling of `s4_13_round_trip_partition_split_64x64_through_decoder`
    /// — that test covered the keyframe partition writer; this one
    /// proves the new `write_tile_inter` path (S4.16c) composes with
    /// the rest of the encoder primitives into a wire stream the
    /// decoder can consume.
    #[test]
    fn s4_16d_round_trip_inter_after_keyframe_through_decoder() {
        use rvp9_core::bitstream::frame::ReferenceMode;
        use rvp9_core::bitstream::header::HeaderTail;
        use rvp9_core::bitstream::sub_parsers::{
            LoopFilterParams, MAX_PROB, PREDICTION_PROBS, QuantizationParams, SEG_TREE_PROBS,
            SegmentationParams, TileInfo,
        };
        use rvp9_core::bitstream::{
            AllZeroCoefSource, BitDepth, ChromaFormat, ColorRange, ColorSpace,
            CompressedHeaderOptions, CompressedHeaderResult, FrameKind, InterpFilter,
            ModesTileBounds, Profile, RawBitWriter, UncompressedHeader, write_compressed_header,
            write_tile_inter, write_tile_intra_keyframe, write_uncompressed_header,
        };
        use rvp9_core::block::BlockSize;
        use rvp9_core::coef_ctx::TileCoefCtx;
        use rvp9_core::entropy::{BoolWriter, FrameContext};
        use rvp9_core::inter::{
            InterModeReadParams, setup_compound_reference_mode, sign_bias_from_header,
        };
        use rvp9_core::intra_pred::IntraMode;
        use rvp9_core::mode_info::{BlockMode, InterMode, ModeInfo, Mv, RefFrame};
        use rvp9_core::transforms::{TxMode, TxSize};

        // Shared geometry: 64×64 frame, mi_rows = mi_cols = 8.
        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let tile_bounds = ModesTileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };

        // Shared HeaderTail builder (identical shape for both frames;
        // the tail's per-field layout is independent of FrameKind —
        // see `write_uncompressed_header` :706-722).
        let make_tail = |compressed_len: usize| HeaderTail {
            refresh_frame_context: true,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams {
                filter_level: 0,
                sharpness_level: 0,
                mode_ref_delta_enabled: false,
                mode_ref_delta_update: false,
                ref_deltas: [None; 4],
                mode_deltas: [None; 2],
            },
            quantization: QuantizationParams {
                base_qindex: 50,
                y_dc_delta_q: 0,
                uv_dc_delta_q: 0,
                uv_ac_delta_q: 0,
            },
            segmentation: SegmentationParams {
                enabled: false,
                update_map: false,
                temporal_update: false,
                tree_probs: [MAX_PROB; SEG_TREE_PROBS],
                pred_probs: [MAX_PROB; PREDICTION_PROBS],
                update_data: false,
                abs_delta: false,
                feature_data: Default::default(),
            },
            tile_info: TileInfo {
                log2_tile_cols: 0,
                log2_tile_rows: 0,
            },
            first_partition_size: compressed_len as u16,
        };

        // ----------------------------------------------------------------
        // Frame 1 — keyframe (PARTITION_NONE, intra DC-skip 64×64 SB).
        // Primes the decoder's ref-frame pool so frame 2 can resolve
        // LAST against pool slot 0.
        // ----------------------------------------------------------------
        let kf_cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let kf_mi_grid: Vec<Option<ModeInfo>> = vec![Some(kf_cell); (mi_rows * mi_cols) as usize];
        // (Every 8×8 cell carries the same 64×64 splatted ModeInfo —
        // the decoder splat convention; see `tile_ctx.rs:411`.)

        let pre_fc_default = FrameContext::new_default();
        let mut kf_pre_fc = pre_fc_default.clone();
        let kf_post_fc = pre_fc_default.clone();
        let kf_opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: true,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let kf_target = CompressedHeaderResult {
            tx_mode: TxMode::AllowTx32x32,
            reference_mode: ReferenceMode::Single,
        };

        let mut kf_compressed_buf = vec![0u8; 16 * 1024];
        let kf_compressed_len = {
            let mut bw = BoolWriter::start_encode(&mut kf_compressed_buf);
            write_compressed_header(&mut bw, &mut kf_pre_fc, &kf_post_fc, kf_opts, &kf_target)
                .expect("write_compressed_header keyframe");
            bw.stop_encode().expect("stop_encode kf compressed")
        };
        kf_compressed_buf.truncate(kf_compressed_len);

        let mut kf_tile_buf = vec![0u8; 4 * 1024];
        let kf_tile_len = {
            let mut bw = BoolWriter::start_encode(&mut kf_tile_buf);
            let tile_mi_cols =
                (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
            let mut coef_ctx = rvp9_core::coef_ctx::TileCoefCtx::new(tile_mi_cols);
            let coef_source = rvp9_core::bitstream::AllZeroIntraCoefSource::new();
            write_tile_intra_keyframe(
                &mut bw,
                &kf_mi_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                &kf_post_fc,
                None, // AQ.0: seg = None (no segmentation in this test)
                &mut coef_ctx,
                &coef_source,
            )
            .expect("write_tile_intra_keyframe");
            bw.stop_encode().expect("stop_encode kf tile")
        };
        kf_tile_buf.truncate(kf_tile_len);

        let kf_tail = make_tail(kf_compressed_len);
        let kf_hdr = UncompressedHeader {
            kind: FrameKind::KeyFrame,
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(kf_tail),
        };

        let mut kf_hdr_buf = vec![0u8; 256];
        let kf_hdr_len = {
            let mut wb = RawBitWriter::new(&mut kf_hdr_buf);
            write_uncompressed_header(&mut wb, &kf_hdr, kf_compressed_len as u16)
                .expect("write_uncompressed_header keyframe");
            wb.finish().expect("RawBitWriter::finish kf")
        };
        kf_hdr_buf.truncate(kf_hdr_len);

        let mut kf_packet = Vec::new();
        kf_packet.extend_from_slice(&kf_hdr_buf);
        kf_packet.extend_from_slice(&kf_compressed_buf);
        kf_packet.extend_from_slice(&kf_tile_buf);

        let mut dec = Decoder::new();
        let kf_parsed = {
            let frames = dec
                .parse_packet(&kf_packet)
                .expect("parse_packet must accept synthetic keyframe");
            assert_eq!(frames.count(), 1);
            let pf = &frames.frames()[0];
            assert_eq!(pf.header.kind, FrameKind::KeyFrame);
            ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            }
        };
        dec.decode_frame(&kf_parsed)
            .expect("decode_frame must accept synthetic keyframe");

        // ----------------------------------------------------------------
        // Frame 2 — single-ref inter frame, LAST = pool slot 0
        // (just-decoded keyframe). PARTITION_NONE 64×64 inter
        // LASTFRAME ZEROMV with skip = true.
        // ----------------------------------------------------------------

        // Synthetic mi_grid: every 8×8 cell carries the same splatted
        // 64×64 inter ModeInfo.
        let inter_cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32, // min(largest_tx_size(AllowTx32x32), max_txsize(64×64))
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        let inter_mi_grid: Vec<Option<ModeInfo>> =
            vec![Some(inter_cell); (mi_rows * mi_cols) as usize];

        // The inter compressed-header parser mutates
        // `frame_contexts[fc_idx]` via diff updates. If we set
        // pre_fc == post_fc == the decoder's *current* FC bank, every
        // diff bit is "0" and the decoder leaves the bank untouched.
        // After the keyframe decode `dec.frame_context(0)` is the
        // post-adapt FC (coef_probs etc.); use that for both sides.
        let inter_post_fc = dec.frame_context(0).clone();
        let mut inter_pre_fc = inter_post_fc.clone();

        let inter_opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: false,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let inter_target = CompressedHeaderResult {
            tx_mode: TxMode::AllowTx32x32,
            reference_mode: ReferenceMode::Single,
        };

        let mut inter_compressed_buf = vec![0u8; 16 * 1024];
        let inter_compressed_len = {
            let mut bw = BoolWriter::start_encode(&mut inter_compressed_buf);
            write_compressed_header(
                &mut bw,
                &mut inter_pre_fc,
                &inter_post_fc,
                inter_opts,
                &inter_target,
            )
            .expect("write_compressed_header inter");
            bw.stop_encode().expect("stop_encode inter compressed")
        };
        inter_compressed_buf.truncate(inter_compressed_len);

        // InterModeReadParams: SINGLE ref mode, all-false sign-bias,
        // 8-tap filter, no HP. comp_fixed_ref / comp_var_ref must be
        // syntactically valid even though compound mode is off.
        let sign_bias = sign_bias_from_header(&[false, false, false]);
        let (comp_fixed_ref, comp_var_ref) = setup_compound_reference_mode(&sign_bias);
        let inter_params = InterModeReadParams {
            allow_high_precision_mv: false,
            interp_filter: InterpFilter::EightTap,
            reference_mode: ReferenceMode::Single,
            comp_fixed_ref,
            comp_var_ref,
            sign_bias,
        };

        let mut inter_tile_buf = vec![0u8; 4 * 1024];
        let inter_tile_len = {
            let mut bw = BoolWriter::start_encode(&mut inter_tile_buf);
            let mut coef_ctx =
                TileCoefCtx::new((tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1));
            let coef_source = AllZeroCoefSource::new();
            write_tile_inter(
                &mut bw,
                &inter_post_fc,
                &inter_params,
                TxMode::AllowTx32x32,
                &inter_mi_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                None, // AQ.0: seg = None (no segmentation in this test)
                None, // no prev-frame MV grid — keyframe just cleared it
                &mut coef_ctx,
                &coef_source,
            )
            .expect("write_tile_inter");
            bw.stop_encode().expect("stop_encode inter tile")
        };
        inter_tile_buf.truncate(inter_tile_len);

        let inter_tail = make_tail(inter_compressed_len);
        let inter_hdr = UncompressedHeader {
            kind: FrameKind::InterFrame {
                reset_frame_context: 0,
                refresh_frame_flags: 0x00,
                ref_frame_indices: [0, 0, 0],
                ref_frame_sign_biases: [false, false, false],
                allow_high_precision_mv: false,
                interp_filter: InterpFilter::EightTap,
            },
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(inter_tail),
        };

        let mut inter_hdr_buf = vec![0u8; 256];
        let inter_hdr_len = {
            let mut wb = RawBitWriter::new(&mut inter_hdr_buf);
            write_uncompressed_header(&mut wb, &inter_hdr, inter_compressed_len as u16)
                .expect("write_uncompressed_header inter");
            wb.finish().expect("RawBitWriter::finish inter")
        };
        inter_hdr_buf.truncate(inter_hdr_len);

        let mut inter_packet = Vec::new();
        inter_packet.extend_from_slice(&inter_hdr_buf);
        inter_packet.extend_from_slice(&inter_compressed_buf);
        inter_packet.extend_from_slice(&inter_tile_buf);

        let inter_parsed = {
            let frames = dec
                .parse_packet(&inter_packet)
                .expect("parse_packet must accept synthetic inter frame");
            assert_eq!(frames.count(), 1);
            let pf = &frames.frames()[0];
            assert!(matches!(pf.header.kind, FrameKind::InterFrame { .. }));
            ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            }
        };
        dec.decode_frame(&inter_parsed)
            .expect("decode_frame must accept synthetic inter frame");

        // Verify the decoder's recovered mi_grid matches what we
        // emitted. Every 8×8 cell of the 64×64 SB should carry a
        // splatted `BlockMode::Inter { ZEROMV, LAST, EightTap }`
        // payload with skip = true.
        let (got_rows, got_cols) = dec.last_frame_mi_dims();
        assert_eq!((got_rows, got_cols), (mi_rows, mi_cols));
        let got_grid = dec.last_frame_mi_grid();
        assert_eq!(got_grid.len(), (mi_rows * mi_cols) as usize);
        for (i, cell_got) in got_grid.iter().enumerate() {
            let mi_row = i / (mi_cols as usize);
            let mi_col = i % (mi_cols as usize);
            let cell =
                cell_got.expect("every cell of the PARTITION_NONE 64×64 SB must be populated");
            assert_eq!(
                cell.block_size,
                BlockSize::Block64x64,
                "block_size @ ({mi_row}, {mi_col}) — must be 64×64 leaf"
            );
            assert!(cell.skip, "skip @ ({mi_row}, {mi_col})");
            match cell.mode {
                BlockMode::Inter {
                    mode,
                    ref_frames,
                    mvs,
                    interp_filter,
                } => {
                    assert_eq!(mode, InterMode::ZeroMv, "inter_mode @ ({mi_row}, {mi_col})");
                    assert_eq!(
                        ref_frames,
                        [RefFrame::LastFrame, RefFrame::NoRefFrame],
                        "ref_frames @ ({mi_row}, {mi_col})"
                    );
                    assert_eq!(mvs, [Mv::ZERO; 2], "mvs @ ({mi_row}, {mi_col})");
                    assert_eq!(
                        interp_filter,
                        InterpFilter::EightTap,
                        "interp_filter @ ({mi_row}, {mi_col})"
                    );
                }
                BlockMode::Intra { .. } => panic!("intra mode @ ({mi_row}, {mi_col})"),
            }
        }
    }

    /// End-to-end inter-frame round-trip with **deep partition
    /// recursion** — 64×64 SB drilled all the way down to 64 BLOCK_8x8
    /// PARTITION_NONE LASTFRAME ZEROMV skip=true leaves. Counterpart
    /// to S4.16d for the "every leaf is BLOCK_8x8" case. Validates
    /// `write_tile_inter`'s recursive walker (S4.16c) plus the inter
    /// mode-info writer (S4.16b) handle 64 leaves per frame through
    /// the real decoder without partition-context drift.
    #[test]
    fn s4_16i_round_trip_inter_dense_8x8_grain_through_decoder() {
        use rvp9_core::bitstream::frame::ReferenceMode;
        use rvp9_core::bitstream::header::HeaderTail;
        use rvp9_core::bitstream::sub_parsers::{
            LoopFilterParams, MAX_PROB, PREDICTION_PROBS, QuantizationParams, SEG_TREE_PROBS,
            SegmentationParams, TileInfo,
        };
        use rvp9_core::bitstream::{
            AllZeroCoefSource, BitDepth, ChromaFormat, ColorRange, ColorSpace,
            CompressedHeaderOptions, CompressedHeaderResult, FrameKind, InterpFilter,
            ModesTileBounds, Profile, RawBitWriter, UncompressedHeader, write_compressed_header,
            write_tile_inter, write_tile_intra_keyframe, write_uncompressed_header,
        };
        use rvp9_core::block::BlockSize;
        use rvp9_core::coef_ctx::TileCoefCtx;
        use rvp9_core::entropy::{BoolWriter, FrameContext};
        use rvp9_core::inter::{
            InterModeReadParams, setup_compound_reference_mode, sign_bias_from_header,
        };
        use rvp9_core::intra_pred::IntraMode;
        use rvp9_core::mode_info::{BlockMode, InterMode, ModeInfo, Mv, RefFrame};
        use rvp9_core::transforms::{TxMode, TxSize};

        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let tile_bounds = ModesTileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };

        let make_tail = |compressed_len: usize| HeaderTail {
            refresh_frame_context: true,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams {
                filter_level: 0,
                sharpness_level: 0,
                mode_ref_delta_enabled: false,
                mode_ref_delta_update: false,
                ref_deltas: [None; 4],
                mode_deltas: [None; 2],
            },
            quantization: QuantizationParams {
                base_qindex: 50,
                y_dc_delta_q: 0,
                uv_dc_delta_q: 0,
                uv_ac_delta_q: 0,
            },
            segmentation: SegmentationParams {
                enabled: false,
                update_map: false,
                temporal_update: false,
                tree_probs: [MAX_PROB; SEG_TREE_PROBS],
                pred_probs: [MAX_PROB; PREDICTION_PROBS],
                update_data: false,
                abs_delta: false,
                feature_data: Default::default(),
            },
            tile_info: TileInfo {
                log2_tile_cols: 0,
                log2_tile_rows: 0,
            },
            first_partition_size: compressed_len as u16,
        };

        // -- Frame 1: keyframe (PARTITION_NONE 64x64 DC-skip) ------
        let kf_cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let kf_grid: Vec<Option<ModeInfo>> = vec![Some(kf_cell); (mi_rows * mi_cols) as usize];

        let mut kf_pre_fc = FrameContext::new_default();
        let kf_post_fc = kf_pre_fc.clone();
        let kf_opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: true,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let kf_target = CompressedHeaderResult {
            tx_mode: TxMode::AllowTx32x32,
            reference_mode: ReferenceMode::Single,
        };

        let mut kf_compressed_buf = vec![0u8; 16 * 1024];
        let kf_compressed_len = {
            let mut bw = BoolWriter::start_encode(&mut kf_compressed_buf);
            write_compressed_header(&mut bw, &mut kf_pre_fc, &kf_post_fc, kf_opts, &kf_target)
                .expect("kf compressed header");
            bw.stop_encode().expect("stop_encode kf compressed")
        };
        kf_compressed_buf.truncate(kf_compressed_len);

        let mut kf_tile_buf = vec![0u8; 4 * 1024];
        let kf_tile_len = {
            let mut bw = BoolWriter::start_encode(&mut kf_tile_buf);
            let tile_mi_cols =
                (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
            let mut coef_ctx = rvp9_core::coef_ctx::TileCoefCtx::new(tile_mi_cols);
            let coef_source = rvp9_core::bitstream::AllZeroIntraCoefSource::new();
            write_tile_intra_keyframe(
                &mut bw,
                &kf_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                &kf_post_fc,
                None, // AQ.0: seg = None (no segmentation in this test)
                &mut coef_ctx,
                &coef_source,
            )
            .expect("kf write_tile");
            bw.stop_encode().expect("stop_encode kf tile")
        };
        kf_tile_buf.truncate(kf_tile_len);

        let kf_tail = make_tail(kf_compressed_len);
        let kf_hdr = UncompressedHeader {
            kind: FrameKind::KeyFrame,
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(kf_tail),
        };
        let mut kf_hdr_buf = vec![0u8; 256];
        let kf_hdr_len = {
            let mut wb = RawBitWriter::new(&mut kf_hdr_buf);
            write_uncompressed_header(&mut wb, &kf_hdr, kf_compressed_len as u16)
                .expect("kf header");
            wb.finish().expect("kf finish")
        };
        kf_hdr_buf.truncate(kf_hdr_len);

        let mut kf_packet = Vec::new();
        kf_packet.extend_from_slice(&kf_hdr_buf);
        kf_packet.extend_from_slice(&kf_compressed_buf);
        kf_packet.extend_from_slice(&kf_tile_buf);

        // Feed the keyframe through the decoder.
        let mut dec = Decoder::new();
        let parsed = dec.parse_packet(&kf_packet).expect("parse kf");
        assert_eq!(parsed.count(), 1);
        let kf_parsed = ParsedFrame {
            header: parsed.frames()[0].header.clone(),
            header_end_byte: parsed.frames()[0].header_end_byte,
            full_frame_data: parsed.frames()[0].full_frame_data,
        };
        dec.decode_frame(&kf_parsed).expect("decode kf");

        // -- Frame 2: inter frame, 64 BLOCK_8x8 NONE leaves --------
        let inter_post_fc = dec.frame_context(0).clone();
        let inter_cell = ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: TxSize::Tx8x8,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        let inter_grid: Vec<Option<ModeInfo>> =
            vec![Some(inter_cell); (mi_rows * mi_cols) as usize];

        let mut inter_pre_fc = inter_post_fc.clone();
        let inter_target_fc = inter_post_fc.clone();
        let inter_opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: false,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let inter_target = CompressedHeaderResult {
            tx_mode: TxMode::AllowTx32x32,
            reference_mode: ReferenceMode::Single,
        };
        let mut inter_compressed_buf = vec![0u8; 16 * 1024];
        let inter_compressed_len = {
            let mut bw = BoolWriter::start_encode(&mut inter_compressed_buf);
            write_compressed_header(
                &mut bw,
                &mut inter_pre_fc,
                &inter_target_fc,
                inter_opts,
                &inter_target,
            )
            .expect("inter compressed header");
            bw.stop_encode().expect("stop_encode inter compressed")
        };
        inter_compressed_buf.truncate(inter_compressed_len);

        let sign_bias = sign_bias_from_header(&[false, false, false]);
        let (comp_fixed_ref, comp_var_ref) = setup_compound_reference_mode(&sign_bias);
        let inter_params = InterModeReadParams {
            allow_high_precision_mv: false,
            interp_filter: InterpFilter::EightTap,
            reference_mode: ReferenceMode::Single,
            comp_fixed_ref,
            comp_var_ref,
            sign_bias,
        };
        let mut inter_tile_buf = vec![0u8; 32 * 1024];
        let inter_tile_len = {
            let mut bw = BoolWriter::start_encode(&mut inter_tile_buf);
            let mut coef_ctx =
                TileCoefCtx::new((tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1));
            let coef_source = AllZeroCoefSource::new();
            write_tile_inter(
                &mut bw,
                &inter_target_fc,
                &inter_params,
                TxMode::AllowTx32x32,
                &inter_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                None, // AQ.0: seg = None (no segmentation in this test)
                None, // prev_mv_grid = None
                &mut coef_ctx,
                &coef_source,
            )
            .expect("inter write_tile");
            bw.stop_encode().expect("stop_encode inter tile")
        };
        inter_tile_buf.truncate(inter_tile_len);

        let inter_tail = make_tail(inter_compressed_len);
        let inter_hdr = UncompressedHeader {
            kind: FrameKind::InterFrame {
                reset_frame_context: 0,
                refresh_frame_flags: 0x00,
                ref_frame_indices: [0, 0, 0],
                ref_frame_sign_biases: [false, false, false],
                allow_high_precision_mv: false,
                interp_filter: InterpFilter::EightTap,
            },
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(inter_tail),
        };
        let mut inter_hdr_buf = vec![0u8; 256];
        let inter_hdr_len = {
            let mut wb = RawBitWriter::new(&mut inter_hdr_buf);
            write_uncompressed_header(&mut wb, &inter_hdr, inter_compressed_len as u16)
                .expect("inter header");
            wb.finish().expect("inter finish")
        };
        inter_hdr_buf.truncate(inter_hdr_len);

        let mut inter_packet = Vec::new();
        inter_packet.extend_from_slice(&inter_hdr_buf);
        inter_packet.extend_from_slice(&inter_compressed_buf);
        inter_packet.extend_from_slice(&inter_tile_buf);

        let inter_parsed_out = dec.parse_packet(&inter_packet).expect("parse inter");
        assert_eq!(inter_parsed_out.count(), 1);
        let inter_parsed = ParsedFrame {
            header: inter_parsed_out.frames()[0].header.clone(),
            header_end_byte: inter_parsed_out.frames()[0].header_end_byte,
            full_frame_data: inter_parsed_out.frames()[0].full_frame_data,
        };
        dec.decode_frame(&inter_parsed).expect("decode inter");

        // Verify all 64 mi-cells came back as BLOCK_8x8 inter
        // LASTFRAME ZEROMV skip=true.
        let (got_rows, got_cols) = dec.last_frame_mi_dims();
        assert_eq!((got_rows, got_cols), (mi_rows, mi_cols));
        let got = dec.last_frame_mi_grid();
        assert_eq!(got.len(), 64);
        for (i, cell_got) in got.iter().enumerate() {
            let mi_row = i / 8;
            let mi_col = i % 8;
            let cell = cell_got.expect("every mi-cell populated");
            assert_eq!(
                cell.block_size,
                BlockSize::Block8x8,
                "block_size @ ({mi_row}, {mi_col})"
            );
            assert!(cell.skip);
            match cell.mode {
                BlockMode::Inter {
                    mode, ref_frames, ..
                } => {
                    assert_eq!(mode, InterMode::ZeroMv);
                    assert_eq!(ref_frames, [RefFrame::LastFrame, RefFrame::NoRefFrame]);
                }
                BlockMode::Intra { .. } => panic!("intra @ ({mi_row}, {mi_col})"),
            }
        }
    }

    /// End-to-end inter-frame round-trip with **sub-8 partitioning**
    /// at one corner — 64×64 SB drilled to BLOCK_8x8 grain (63 cells)
    /// with cell (0, 0) further sub-partitioned into a BLOCK_4x4
    /// sub-8 inter leaf carrying 4× ZEROMV sub-blocks. Validates
    /// `write_sub8_bmi` (S4.16e) + the partition walker's `hbs == 0`
    /// sub-8 branch (S4.16f) compose through the **real decoder**'s
    /// `resolve_sub8_bmi` + per-bmi reconstruction (motion-compensated
    /// per 4×4 sub-block + chroma-MV averaging via
    /// `avg_split_mv_q4_4x4`).
    ///
    /// First proof-of-life that sub-8 inter leaves the encoder emits
    /// can be parsed AND reconstructed by the real decoder, not just
    /// round-tripped through the in-test replay walker.
    #[test]
    fn s4_16j_round_trip_inter_sub8_at_corner_through_decoder() {
        use rvp9_core::bitstream::frame::ReferenceMode;
        use rvp9_core::bitstream::header::HeaderTail;
        use rvp9_core::bitstream::sub_parsers::{
            LoopFilterParams, MAX_PROB, PREDICTION_PROBS, QuantizationParams, SEG_TREE_PROBS,
            SegmentationParams, TileInfo,
        };
        use rvp9_core::bitstream::{
            AllZeroCoefSource, BitDepth, ChromaFormat, ColorRange, ColorSpace,
            CompressedHeaderOptions, CompressedHeaderResult, FrameKind, InterpFilter,
            ModesTileBounds, Profile, RawBitWriter, UncompressedHeader, write_compressed_header,
            write_tile_inter, write_tile_intra_keyframe, write_uncompressed_header,
        };
        use rvp9_core::block::BlockSize;
        use rvp9_core::coef_ctx::TileCoefCtx;
        use rvp9_core::entropy::{BoolWriter, FrameContext};
        use rvp9_core::inter::{
            InterModeReadParams, setup_compound_reference_mode, sign_bias_from_header,
        };
        use rvp9_core::intra_pred::IntraMode;
        use rvp9_core::mode_info::{BlockMode, BmiInfo, InterMode, ModeInfo, Mv, RefFrame};
        use rvp9_core::transforms::{TxMode, TxSize};

        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let tile_bounds = ModesTileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let make_tail = |compressed_len: usize| HeaderTail {
            refresh_frame_context: true,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams {
                filter_level: 0,
                sharpness_level: 0,
                mode_ref_delta_enabled: false,
                mode_ref_delta_update: false,
                ref_deltas: [None; 4],
                mode_deltas: [None; 2],
            },
            quantization: QuantizationParams {
                base_qindex: 50,
                y_dc_delta_q: 0,
                uv_dc_delta_q: 0,
                uv_ac_delta_q: 0,
            },
            segmentation: SegmentationParams {
                enabled: false,
                update_map: false,
                temporal_update: false,
                tree_probs: [MAX_PROB; SEG_TREE_PROBS],
                pred_probs: [MAX_PROB; PREDICTION_PROBS],
                update_data: false,
                abs_delta: false,
                feature_data: Default::default(),
            },
            tile_info: TileInfo {
                log2_tile_cols: 0,
                log2_tile_rows: 0,
            },
            first_partition_size: compressed_len as u16,
        };

        // -- Frame 1: PARTITION_NONE 64x64 DC-skip keyframe --------
        let kf_cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let kf_grid: Vec<Option<ModeInfo>> = vec![Some(kf_cell); (mi_rows * mi_cols) as usize];
        let mut kf_pre_fc = FrameContext::new_default();
        let kf_post_fc = kf_pre_fc.clone();
        let kf_opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: true,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let kf_target = CompressedHeaderResult {
            tx_mode: TxMode::AllowTx32x32,
            reference_mode: ReferenceMode::Single,
        };
        let mut kf_compressed_buf = vec![0u8; 16 * 1024];
        let kf_compressed_len = {
            let mut bw = BoolWriter::start_encode(&mut kf_compressed_buf);
            write_compressed_header(&mut bw, &mut kf_pre_fc, &kf_post_fc, kf_opts, &kf_target)
                .expect("kf compressed");
            bw.stop_encode().expect("stop_encode kf compressed")
        };
        kf_compressed_buf.truncate(kf_compressed_len);
        let mut kf_tile_buf = vec![0u8; 4 * 1024];
        let kf_tile_len = {
            let mut bw = BoolWriter::start_encode(&mut kf_tile_buf);
            let tile_mi_cols =
                (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
            let mut coef_ctx = rvp9_core::coef_ctx::TileCoefCtx::new(tile_mi_cols);
            let coef_source = rvp9_core::bitstream::AllZeroIntraCoefSource::new();
            write_tile_intra_keyframe(
                &mut bw,
                &kf_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                &kf_post_fc,
                None, // AQ.0: seg = None (no segmentation in this test)
                &mut coef_ctx,
                &coef_source,
            )
            .expect("kf write_tile");
            bw.stop_encode().expect("stop_encode kf tile")
        };
        kf_tile_buf.truncate(kf_tile_len);
        let kf_tail = make_tail(kf_compressed_len);
        let kf_hdr = UncompressedHeader {
            kind: FrameKind::KeyFrame,
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(kf_tail),
        };
        let mut kf_hdr_buf = vec![0u8; 256];
        let kf_hdr_len = {
            let mut wb = RawBitWriter::new(&mut kf_hdr_buf);
            write_uncompressed_header(&mut wb, &kf_hdr, kf_compressed_len as u16)
                .expect("kf header");
            wb.finish().expect("kf finish")
        };
        kf_hdr_buf.truncate(kf_hdr_len);
        let mut kf_packet = Vec::new();
        kf_packet.extend_from_slice(&kf_hdr_buf);
        kf_packet.extend_from_slice(&kf_compressed_buf);
        kf_packet.extend_from_slice(&kf_tile_buf);
        let mut dec = Decoder::new();
        let parsed = dec.parse_packet(&kf_packet).expect("parse kf");
        let kf_parsed = ParsedFrame {
            header: parsed.frames()[0].header.clone(),
            header_end_byte: parsed.frames()[0].header_end_byte,
            full_frame_data: parsed.frames()[0].full_frame_data,
        };
        dec.decode_frame(&kf_parsed).expect("decode kf");

        // -- Frame 2: dense 8x8-grain inter with one BLOCK_4x4 leaf at (0,0)
        let inter_post_fc = dec.frame_context(0).clone();

        let plain_inter_cell = ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: TxSize::Tx8x8,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        let mut inter_grid: Vec<Option<ModeInfo>> =
            vec![Some(plain_inter_cell); (mi_rows * mi_cols) as usize];

        // Replace cell (0, 0) with a BLOCK_4x4 sub-8 leaf: 4× ZEROMV
        // sub-blocks. The writer's per-sub-block contract: each
        // bmi[j].as_mode carries InterMode::idx() (= 2 for ZEROMV),
        // each as_mv = [Mv::ZERO; 2]. Parent's `mode` = ZEROMV (matches
        // bmi[3].as_mode); parent's `mvs` = (0,0) (matches bmi[3].as_mv).
        let mut sub8_bmi = [BmiInfo::default(); 4];
        for slot in &mut sub8_bmi {
            slot.as_mode = InterMode::ZeroMv.idx() as u8;
            slot.as_mv = [Mv::ZERO; 2];
        }
        let sub8_cell = ModeInfo {
            block_size: BlockSize::Block4x4,
            tx_size: TxSize::Tx4x4,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: sub8_bmi,
        };
        inter_grid[0] = Some(sub8_cell);

        let mut inter_pre_fc = inter_post_fc.clone();
        let inter_target_fc = inter_post_fc.clone();
        let inter_opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: false,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let inter_target = CompressedHeaderResult {
            tx_mode: TxMode::AllowTx32x32,
            reference_mode: ReferenceMode::Single,
        };
        let mut inter_compressed_buf = vec![0u8; 16 * 1024];
        let inter_compressed_len = {
            let mut bw = BoolWriter::start_encode(&mut inter_compressed_buf);
            write_compressed_header(
                &mut bw,
                &mut inter_pre_fc,
                &inter_target_fc,
                inter_opts,
                &inter_target,
            )
            .expect("inter compressed");
            bw.stop_encode().expect("stop_encode inter compressed")
        };
        inter_compressed_buf.truncate(inter_compressed_len);

        let sign_bias = sign_bias_from_header(&[false, false, false]);
        let (comp_fixed_ref, comp_var_ref) = setup_compound_reference_mode(&sign_bias);
        let inter_params = InterModeReadParams {
            allow_high_precision_mv: false,
            interp_filter: InterpFilter::EightTap,
            reference_mode: ReferenceMode::Single,
            comp_fixed_ref,
            comp_var_ref,
            sign_bias,
        };
        let mut inter_tile_buf = vec![0u8; 32 * 1024];
        let inter_tile_len = {
            let mut bw = BoolWriter::start_encode(&mut inter_tile_buf);
            let mut coef_ctx =
                TileCoefCtx::new((tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1));
            let coef_source = AllZeroCoefSource::new();
            write_tile_inter(
                &mut bw,
                &inter_target_fc,
                &inter_params,
                TxMode::AllowTx32x32,
                &inter_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                None, // AQ.0: seg = None (no segmentation in this test)
                None, // prev_mv_grid = None
                &mut coef_ctx,
                &coef_source,
            )
            .expect("inter write_tile (with sub-8)");
            bw.stop_encode().expect("stop_encode inter tile")
        };
        inter_tile_buf.truncate(inter_tile_len);

        let inter_tail = make_tail(inter_compressed_len);
        let inter_hdr = UncompressedHeader {
            kind: FrameKind::InterFrame {
                reset_frame_context: 0,
                refresh_frame_flags: 0x00,
                ref_frame_indices: [0, 0, 0],
                ref_frame_sign_biases: [false, false, false],
                allow_high_precision_mv: false,
                interp_filter: InterpFilter::EightTap,
            },
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(inter_tail),
        };
        let mut inter_hdr_buf = vec![0u8; 256];
        let inter_hdr_len = {
            let mut wb = RawBitWriter::new(&mut inter_hdr_buf);
            write_uncompressed_header(&mut wb, &inter_hdr, inter_compressed_len as u16)
                .expect("inter header");
            wb.finish().expect("inter finish")
        };
        inter_hdr_buf.truncate(inter_hdr_len);
        let mut inter_packet = Vec::new();
        inter_packet.extend_from_slice(&inter_hdr_buf);
        inter_packet.extend_from_slice(&inter_compressed_buf);
        inter_packet.extend_from_slice(&inter_tile_buf);

        let inter_parsed_out = dec.parse_packet(&inter_packet).expect("parse inter");
        let inter_parsed = ParsedFrame {
            header: inter_parsed_out.frames()[0].header.clone(),
            header_end_byte: inter_parsed_out.frames()[0].header_end_byte,
            full_frame_data: inter_parsed_out.frames()[0].full_frame_data,
        };
        dec.decode_frame(&inter_parsed)
            .expect("decode inter (with sub-8 corner)");

        // Verify the sub-8 corner cell came back as Block4x4 inter
        // (decoder stores the sub-8 block_size in the mi-cell). The
        // surrounding cells should still be Block8x8.
        let got = dec.last_frame_mi_grid();
        let corner = got[0].expect("corner cell populated");
        assert_eq!(
            corner.block_size,
            BlockSize::Block4x4,
            "sub-8 corner leaf must round-trip as Block4x4"
        );
        match corner.mode {
            BlockMode::Inter {
                mode, ref_frames, ..
            } => {
                assert_eq!(mode, InterMode::ZeroMv);
                assert_eq!(ref_frames, [RefFrame::LastFrame, RefFrame::NoRefFrame]);
            }
            BlockMode::Intra { .. } => panic!("sub-8 corner came back as intra"),
        }
        // Surrounding cells stay Block8x8.
        let neighbour = got[1].expect("neighbour cell populated");
        assert_eq!(neighbour.block_size, BlockSize::Block8x8);
    }

    /// End-to-end inter-frame round-trip with **skip = false** + all-zero
    /// residual coefs through the **real Decoder**. Counterpart to S4.16d
    /// (the skip=true sibling) — same single-leaf 64×64 PARTITION_NONE
    /// LASTFRAME ZEROMV shape, but the leaf carries `skip = false` and
    /// the writer emits per-plane per-TX residual tokens via
    /// `write_block_tokens` (S4.16g phase 2) routed through
    /// `write_inter_block_residual` (S4.16g phase 3a).
    ///
    /// The decoder reads skip=0, walks per-plane per-TX (4 Y Tx32x32 +
    /// 1 U Tx32x32 + 1 V Tx32x32 = 6 TX blocks), decodes each token
    /// stream (`eob = 0` → terminal-EOB bit only), runs inverse transform
    /// of zero (= zero), adds to predictor (no change). Pixel output is
    /// identical to skip=true; the coef-decode + IT-add + writeback path
    /// **fires** for the first time end-to-end through the real Decoder.
    ///
    /// Proves: S4.16g phase 2 + phase 3a primitives compose correctly
    /// through `Decoder::decode_frame`. The encoder's all-zero coef
    /// emission is byte-identical with what `decode_block_tokens` consumes
    /// in the real decoder's inter-residual pipeline.
    #[test]
    fn s4_16k_round_trip_non_skip_all_zero_through_decoder() {
        use rvp9_core::bitstream::frame::ReferenceMode;
        use rvp9_core::bitstream::header::HeaderTail;
        use rvp9_core::bitstream::sub_parsers::{
            LoopFilterParams, MAX_PROB, PREDICTION_PROBS, QuantizationParams, SEG_TREE_PROBS,
            SegmentationParams, TileInfo,
        };
        use rvp9_core::bitstream::{
            AllZeroCoefSource, BitDepth, ChromaFormat, ColorRange, ColorSpace,
            CompressedHeaderOptions, CompressedHeaderResult, FrameKind, InterpFilter,
            ModesTileBounds, Profile, RawBitWriter, UncompressedHeader, write_compressed_header,
            write_tile_inter, write_tile_intra_keyframe, write_uncompressed_header,
        };
        use rvp9_core::block::BlockSize;
        use rvp9_core::coef_ctx::TileCoefCtx;
        use rvp9_core::entropy::{BoolWriter, FrameContext};
        use rvp9_core::inter::{
            InterModeReadParams, setup_compound_reference_mode, sign_bias_from_header,
        };
        use rvp9_core::intra_pred::IntraMode;
        use rvp9_core::mode_info::{BlockMode, InterMode, ModeInfo, Mv, RefFrame};
        use rvp9_core::transforms::{TxMode, TxSize};

        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let tile_bounds = ModesTileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };

        let make_tail = |compressed_len: usize| HeaderTail {
            refresh_frame_context: true,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams {
                filter_level: 0,
                sharpness_level: 0,
                mode_ref_delta_enabled: false,
                mode_ref_delta_update: false,
                ref_deltas: [None; 4],
                mode_deltas: [None; 2],
            },
            quantization: QuantizationParams {
                base_qindex: 50,
                y_dc_delta_q: 0,
                uv_dc_delta_q: 0,
                uv_ac_delta_q: 0,
            },
            segmentation: SegmentationParams {
                enabled: false,
                update_map: false,
                temporal_update: false,
                tree_probs: [MAX_PROB; SEG_TREE_PROBS],
                pred_probs: [MAX_PROB; PREDICTION_PROBS],
                update_data: false,
                abs_delta: false,
                feature_data: Default::default(),
            },
            tile_info: TileInfo {
                log2_tile_cols: 0,
                log2_tile_rows: 0,
            },
            first_partition_size: compressed_len as u16,
        };

        // ---- Frame 1: keyframe (primes ref pool slot 0) ----
        let kf_cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let kf_mi_grid: Vec<Option<ModeInfo>> = vec![Some(kf_cell); (mi_rows * mi_cols) as usize];

        let pre_fc_default = FrameContext::new_default();
        let mut kf_pre_fc = pre_fc_default.clone();
        let kf_post_fc = pre_fc_default.clone();
        let kf_opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: true,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let kf_target = CompressedHeaderResult {
            tx_mode: TxMode::AllowTx32x32,
            reference_mode: ReferenceMode::Single,
        };

        let mut kf_compressed_buf = vec![0u8; 16 * 1024];
        let kf_compressed_len = {
            let mut bw = BoolWriter::start_encode(&mut kf_compressed_buf);
            write_compressed_header(&mut bw, &mut kf_pre_fc, &kf_post_fc, kf_opts, &kf_target)
                .expect("write_compressed_header keyframe");
            bw.stop_encode().expect("stop_encode kf compressed")
        };
        kf_compressed_buf.truncate(kf_compressed_len);

        let mut kf_tile_buf = vec![0u8; 4 * 1024];
        let kf_tile_len = {
            let mut bw = BoolWriter::start_encode(&mut kf_tile_buf);
            let tile_mi_cols =
                (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
            let mut coef_ctx = rvp9_core::coef_ctx::TileCoefCtx::new(tile_mi_cols);
            let coef_source = rvp9_core::bitstream::AllZeroIntraCoefSource::new();
            write_tile_intra_keyframe(
                &mut bw,
                &kf_mi_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                &kf_post_fc,
                None, // AQ.0: seg = None (no segmentation in this test)
                &mut coef_ctx,
                &coef_source,
            )
            .expect("write_tile_intra_keyframe");
            bw.stop_encode().expect("stop_encode kf tile")
        };
        kf_tile_buf.truncate(kf_tile_len);

        let kf_tail = make_tail(kf_compressed_len);
        let kf_hdr = UncompressedHeader {
            kind: FrameKind::KeyFrame,
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(kf_tail),
        };

        let mut kf_hdr_buf = vec![0u8; 256];
        let kf_hdr_len = {
            let mut wb = RawBitWriter::new(&mut kf_hdr_buf);
            write_uncompressed_header(&mut wb, &kf_hdr, kf_compressed_len as u16)
                .expect("write_uncompressed_header keyframe");
            wb.finish().expect("RawBitWriter::finish kf")
        };
        kf_hdr_buf.truncate(kf_hdr_len);

        let mut kf_packet = Vec::new();
        kf_packet.extend_from_slice(&kf_hdr_buf);
        kf_packet.extend_from_slice(&kf_compressed_buf);
        kf_packet.extend_from_slice(&kf_tile_buf);

        let mut dec = Decoder::new();
        let kf_parsed = {
            let frames = dec.parse_packet(&kf_packet).expect("parse_packet kf");
            assert_eq!(frames.count(), 1);
            let pf = &frames.frames()[0];
            ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            }
        };
        dec.decode_frame(&kf_parsed).expect("decode_frame kf");

        // ---- Frame 2: inter frame, single 64×64 leaf, SKIP = FALSE ----
        let inter_cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: false, // <-- the only structural difference vs S4.16d
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        let inter_mi_grid: Vec<Option<ModeInfo>> =
            vec![Some(inter_cell); (mi_rows * mi_cols) as usize];

        let inter_post_fc = dec.frame_context(0).clone();
        let mut inter_pre_fc = inter_post_fc.clone();

        let inter_opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: false,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let inter_target = CompressedHeaderResult {
            tx_mode: TxMode::AllowTx32x32,
            reference_mode: ReferenceMode::Single,
        };

        let mut inter_compressed_buf = vec![0u8; 16 * 1024];
        let inter_compressed_len = {
            let mut bw = BoolWriter::start_encode(&mut inter_compressed_buf);
            write_compressed_header(
                &mut bw,
                &mut inter_pre_fc,
                &inter_post_fc,
                inter_opts,
                &inter_target,
            )
            .expect("write_compressed_header inter");
            bw.stop_encode().expect("stop_encode inter compressed")
        };
        inter_compressed_buf.truncate(inter_compressed_len);

        let sign_bias = sign_bias_from_header(&[false, false, false]);
        let (comp_fixed_ref, comp_var_ref) = setup_compound_reference_mode(&sign_bias);
        let inter_params = InterModeReadParams {
            allow_high_precision_mv: false,
            interp_filter: InterpFilter::EightTap,
            reference_mode: ReferenceMode::Single,
            comp_fixed_ref,
            comp_var_ref,
            sign_bias,
        };

        let mut inter_tile_buf = vec![0u8; 4 * 1024];
        let inter_tile_len = {
            let mut bw = BoolWriter::start_encode(&mut inter_tile_buf);
            let mut coef_ctx =
                TileCoefCtx::new((tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1));
            let coef_source = AllZeroCoefSource::new();
            write_tile_inter(
                &mut bw,
                &inter_post_fc,
                &inter_params,
                TxMode::AllowTx32x32,
                &inter_mi_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                None, // AQ.0: seg = None (no segmentation in this test)
                None, // prev_mv_grid = None
                &mut coef_ctx,
                &coef_source,
            )
            .expect("write_tile_inter");
            bw.stop_encode().expect("stop_encode inter tile")
        };
        inter_tile_buf.truncate(inter_tile_len);

        let inter_tail = make_tail(inter_compressed_len);
        let inter_hdr = UncompressedHeader {
            kind: FrameKind::InterFrame {
                reset_frame_context: 0,
                refresh_frame_flags: 0x00,
                ref_frame_indices: [0, 0, 0],
                ref_frame_sign_biases: [false, false, false],
                allow_high_precision_mv: false,
                interp_filter: InterpFilter::EightTap,
            },
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(inter_tail),
        };

        let mut inter_hdr_buf = vec![0u8; 256];
        let inter_hdr_len = {
            let mut wb = RawBitWriter::new(&mut inter_hdr_buf);
            write_uncompressed_header(&mut wb, &inter_hdr, inter_compressed_len as u16)
                .expect("write_uncompressed_header inter");
            wb.finish().expect("RawBitWriter::finish inter")
        };
        inter_hdr_buf.truncate(inter_hdr_len);

        let mut inter_packet = Vec::new();
        inter_packet.extend_from_slice(&inter_hdr_buf);
        inter_packet.extend_from_slice(&inter_compressed_buf);
        inter_packet.extend_from_slice(&inter_tile_buf);

        let inter_parsed = {
            let frames = dec
                .parse_packet(&inter_packet)
                .expect("parse_packet skip=false inter");
            assert_eq!(frames.count(), 1);
            let pf = &frames.frames()[0];
            assert!(matches!(pf.header.kind, FrameKind::InterFrame { .. }));
            ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            }
        };
        // **The critical assertion** — decode must succeed even though
        // skip=false drives the real decoder's coef-decode + IT-add
        // pipeline for every TX block.
        dec.decode_frame(&inter_parsed)
            .expect("decode_frame must accept skip=false inter with all-zero coefs");

        // Recover mi_grid, assert every cell came back as skip=false.
        let (got_rows, got_cols) = dec.last_frame_mi_dims();
        assert_eq!((got_rows, got_cols), (mi_rows, mi_cols));
        let got_grid = dec.last_frame_mi_grid();
        for (i, cell_got) in got_grid.iter().enumerate() {
            let mi_row = i / (mi_cols as usize);
            let mi_col = i % (mi_cols as usize);
            let cell = cell_got.expect("every cell of the 64×64 SB must be populated");
            assert_eq!(
                cell.block_size,
                BlockSize::Block64x64,
                "block_size @ ({mi_row}, {mi_col})"
            );
            assert!(
                !cell.skip,
                "skip flag must round-trip as FALSE @ ({mi_row}, {mi_col})"
            );
            match cell.mode {
                BlockMode::Inter {
                    mode,
                    ref_frames,
                    mvs,
                    interp_filter,
                } => {
                    assert_eq!(mode, InterMode::ZeroMv);
                    assert_eq!(ref_frames, [RefFrame::LastFrame, RefFrame::NoRefFrame]);
                    assert_eq!(mvs, [Mv::ZERO; 2]);
                    assert_eq!(interp_filter, InterpFilter::EightTap);
                }
                BlockMode::Intra { .. } => {
                    panic!("intra mode @ ({mi_row}, {mi_col})")
                }
            }
        }
    }

    /// End-to-end inter-frame round-trip with **skip = false** + ONE
    /// NON-ZERO Y coefficient per TX block (UV stay all-zero) through
    /// the real Decoder. Sibling of S4.16k (the trivial-coef sibling).
    ///
    /// S4.16k proved the coef-decode path fires for all-zero coefs
    /// (every TX block hits the terminal-EOB bit only — inverse
    /// transform of zero = identity = no pixel change). S4.16l proves
    /// the coef-decode AND inverse-transform AND add-to-predictor
    /// pipeline fires for actual non-zero residuals through the real
    /// `Decoder::decode_frame`.
    ///
    /// Encoder emits `raw_coefs[0] = 1, eob = 1` for every Y TX
    /// (4 × Tx32x32) and `eob = 0` for both UV TX blocks. Decoder
    /// reads these via `decode_block_tokens`, dequantizes (Y DC at
    /// qindex=50 → some non-zero value), runs `idct32x32_1024_add`
    /// per Y TX block, adds the resulting pattern to the predictor.
    ///
    /// Test asserts the decoder succeeds + mi_grid recovery matches.
    /// Pixel output WILL differ from S4.16k (non-zero residuals
    /// modify the predictor), but we don't check pixels here — the
    /// proof-of-life is that the entire pipeline runs without error.
    ///
    /// Together with S4.16k, this completes E2E coverage of every
    /// skip/coef combination through the real decoder.
    #[test]
    fn s4_16l_round_trip_non_skip_one_per_y_through_decoder() {
        use rvp9_core::bitstream::frame::ReferenceMode;
        use rvp9_core::bitstream::header::HeaderTail;
        use rvp9_core::bitstream::inter_coef_source::InterCoefSource;
        use rvp9_core::bitstream::sub_parsers::{
            LoopFilterParams, MAX_PROB, PREDICTION_PROBS, QuantizationParams, SEG_TREE_PROBS,
            SegmentationParams, TileInfo,
        };
        use rvp9_core::bitstream::{
            BitDepth, ChromaFormat, ColorRange, ColorSpace, CompressedHeaderOptions,
            CompressedHeaderResult, FrameKind, InterpFilter, ModesTileBounds, Profile,
            RawBitWriter, UncompressedHeader, write_compressed_header, write_tile_inter,
            write_tile_intra_keyframe, write_uncompressed_header,
        };
        use rvp9_core::block::BlockSize;
        use rvp9_core::coef_ctx::TileCoefCtx;
        use rvp9_core::entropy::{BoolWriter, FrameContext};
        use rvp9_core::inter::{
            InterModeReadParams, setup_compound_reference_mode, sign_bias_from_header,
        };
        use rvp9_core::intra_pred::IntraMode;
        use rvp9_core::mode_info::{BlockMode, InterMode, ModeInfo, Mv, RefFrame};
        use rvp9_core::transforms::{TxMode, TxSize};

        /// Per-Y-TX single non-zero coef source. Y plane gets
        /// `raw_coefs[0] = 1, eob = 1`; UV planes stay all-zero.
        /// Tile-scoped 'static buffers are safe because the test
        /// borrows immutably for one call at a time.
        struct OnePerYThenZero {
            y_buf: [i32; 1024],
            zero_buf: [i32; 1024],
        }
        impl OnePerYThenZero {
            fn new() -> Self {
                let mut y_buf = [0i32; 1024];
                y_buf[0] = 1;
                Self {
                    y_buf,
                    zero_buf: [0i32; 1024],
                }
            }
        }
        impl InterCoefSource for OnePerYThenZero {
            fn get_block_coefs(
                &self,
                _mi_row: usize,
                _mi_col: usize,
                plane: usize,
                _tx_x4_local: usize,
                _tx_y4_local: usize,
                tx_size: TxSize,
            ) -> (&[i32], usize) {
                let n = tx_size.n_coeffs();
                if plane == 0 {
                    (&self.y_buf[..n], 1)
                } else {
                    (&self.zero_buf[..n], 0)
                }
            }
        }

        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let tile_bounds = ModesTileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };

        let make_tail = |compressed_len: usize| HeaderTail {
            refresh_frame_context: true,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams {
                filter_level: 0,
                sharpness_level: 0,
                mode_ref_delta_enabled: false,
                mode_ref_delta_update: false,
                ref_deltas: [None; 4],
                mode_deltas: [None; 2],
            },
            quantization: QuantizationParams {
                base_qindex: 50,
                y_dc_delta_q: 0,
                uv_dc_delta_q: 0,
                uv_ac_delta_q: 0,
            },
            segmentation: SegmentationParams {
                enabled: false,
                update_map: false,
                temporal_update: false,
                tree_probs: [MAX_PROB; SEG_TREE_PROBS],
                pred_probs: [MAX_PROB; PREDICTION_PROBS],
                update_data: false,
                abs_delta: false,
                feature_data: Default::default(),
            },
            tile_info: TileInfo {
                log2_tile_cols: 0,
                log2_tile_rows: 0,
            },
            first_partition_size: compressed_len as u16,
        };

        // ---- Frame 1: keyframe (primes ref pool slot 0) ----
        let kf_cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let kf_mi_grid: Vec<Option<ModeInfo>> = vec![Some(kf_cell); (mi_rows * mi_cols) as usize];

        let pre_fc_default = FrameContext::new_default();
        let mut kf_pre_fc = pre_fc_default.clone();
        let kf_post_fc = pre_fc_default.clone();
        let kf_opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: true,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let kf_target = CompressedHeaderResult {
            tx_mode: TxMode::AllowTx32x32,
            reference_mode: ReferenceMode::Single,
        };

        let mut kf_compressed_buf = vec![0u8; 16 * 1024];
        let kf_compressed_len = {
            let mut bw = BoolWriter::start_encode(&mut kf_compressed_buf);
            write_compressed_header(&mut bw, &mut kf_pre_fc, &kf_post_fc, kf_opts, &kf_target)
                .expect("write_compressed_header kf");
            bw.stop_encode().expect("stop_encode kf compressed")
        };
        kf_compressed_buf.truncate(kf_compressed_len);

        let mut kf_tile_buf = vec![0u8; 4 * 1024];
        let kf_tile_len = {
            let mut bw = BoolWriter::start_encode(&mut kf_tile_buf);
            let tile_mi_cols =
                (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
            let mut coef_ctx = rvp9_core::coef_ctx::TileCoefCtx::new(tile_mi_cols);
            let coef_source = rvp9_core::bitstream::AllZeroIntraCoefSource::new();
            write_tile_intra_keyframe(
                &mut bw,
                &kf_mi_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                &kf_post_fc,
                None, // AQ.0: seg = None (no segmentation in this test)
                &mut coef_ctx,
                &coef_source,
            )
            .expect("write_tile_intra_keyframe");
            bw.stop_encode().expect("stop_encode kf tile")
        };
        kf_tile_buf.truncate(kf_tile_len);

        let kf_tail = make_tail(kf_compressed_len);
        let kf_hdr = UncompressedHeader {
            kind: FrameKind::KeyFrame,
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(kf_tail),
        };

        let mut kf_hdr_buf = vec![0u8; 256];
        let kf_hdr_len = {
            let mut wb = RawBitWriter::new(&mut kf_hdr_buf);
            write_uncompressed_header(&mut wb, &kf_hdr, kf_compressed_len as u16)
                .expect("write_uncompressed_header kf");
            wb.finish().expect("RawBitWriter::finish kf")
        };
        kf_hdr_buf.truncate(kf_hdr_len);

        let mut kf_packet = Vec::new();
        kf_packet.extend_from_slice(&kf_hdr_buf);
        kf_packet.extend_from_slice(&kf_compressed_buf);
        kf_packet.extend_from_slice(&kf_tile_buf);

        let mut dec = Decoder::new();
        let kf_parsed = {
            let frames = dec.parse_packet(&kf_packet).expect("parse_packet kf");
            let pf = &frames.frames()[0];
            ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            }
        };
        dec.decode_frame(&kf_parsed).expect("decode_frame kf");

        // ---- Frame 2: inter, skip=false, OnePerYThenZero coefs ----
        let inter_cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        let inter_mi_grid: Vec<Option<ModeInfo>> =
            vec![Some(inter_cell); (mi_rows * mi_cols) as usize];

        let inter_post_fc = dec.frame_context(0).clone();
        let mut inter_pre_fc = inter_post_fc.clone();

        let inter_opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: false,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let inter_target = CompressedHeaderResult {
            tx_mode: TxMode::AllowTx32x32,
            reference_mode: ReferenceMode::Single,
        };

        let mut inter_compressed_buf = vec![0u8; 16 * 1024];
        let inter_compressed_len = {
            let mut bw = BoolWriter::start_encode(&mut inter_compressed_buf);
            write_compressed_header(
                &mut bw,
                &mut inter_pre_fc,
                &inter_post_fc,
                inter_opts,
                &inter_target,
            )
            .expect("write_compressed_header inter");
            bw.stop_encode().expect("stop_encode inter compressed")
        };
        inter_compressed_buf.truncate(inter_compressed_len);

        let sign_bias = sign_bias_from_header(&[false, false, false]);
        let (comp_fixed_ref, comp_var_ref) = setup_compound_reference_mode(&sign_bias);
        let inter_params = InterModeReadParams {
            allow_high_precision_mv: false,
            interp_filter: InterpFilter::EightTap,
            reference_mode: ReferenceMode::Single,
            comp_fixed_ref,
            comp_var_ref,
            sign_bias,
        };

        let mut inter_tile_buf = vec![0u8; 4 * 1024];
        let inter_tile_len = {
            let mut bw = BoolWriter::start_encode(&mut inter_tile_buf);
            let mut coef_ctx =
                TileCoefCtx::new((tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1));
            let coef_source = OnePerYThenZero::new();
            write_tile_inter(
                &mut bw,
                &inter_post_fc,
                &inter_params,
                TxMode::AllowTx32x32,
                &inter_mi_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                None, // AQ.0: seg = None (no segmentation in this test)
                None, // prev_mv_grid = None
                &mut coef_ctx,
                &coef_source,
            )
            .expect("write_tile_inter");
            bw.stop_encode().expect("stop_encode inter tile")
        };
        inter_tile_buf.truncate(inter_tile_len);

        let inter_tail = make_tail(inter_compressed_len);
        let inter_hdr = UncompressedHeader {
            kind: FrameKind::InterFrame {
                reset_frame_context: 0,
                refresh_frame_flags: 0x00,
                ref_frame_indices: [0, 0, 0],
                ref_frame_sign_biases: [false, false, false],
                allow_high_precision_mv: false,
                interp_filter: InterpFilter::EightTap,
            },
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(inter_tail),
        };

        let mut inter_hdr_buf = vec![0u8; 256];
        let inter_hdr_len = {
            let mut wb = RawBitWriter::new(&mut inter_hdr_buf);
            write_uncompressed_header(&mut wb, &inter_hdr, inter_compressed_len as u16)
                .expect("write_uncompressed_header inter");
            wb.finish().expect("RawBitWriter::finish inter")
        };
        inter_hdr_buf.truncate(inter_hdr_len);

        let mut inter_packet = Vec::new();
        inter_packet.extend_from_slice(&inter_hdr_buf);
        inter_packet.extend_from_slice(&inter_compressed_buf);
        inter_packet.extend_from_slice(&inter_tile_buf);

        let inter_parsed = {
            let frames = dec
                .parse_packet(&inter_packet)
                .expect("parse_packet inter non-skip non-zero coefs");
            let pf = &frames.frames()[0];
            ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            }
        };

        // **The critical assertion** — decode must succeed even though
        // skip=false + non-zero Y coefs drive the real decoder's
        // coef-decode + dequant + idct32x32_add pipeline for every
        // Y TX block.
        dec.decode_frame(&inter_parsed)
            .expect("decode_frame must accept skip=false inter with non-zero Y coefs");

        // mi_grid recovery: every cell came back as skip=false inter
        // (same shape as S4.16k).
        let got_grid = dec.last_frame_mi_grid();
        for (i, cell_got) in got_grid.iter().enumerate() {
            let mi_row = i / (mi_cols as usize);
            let mi_col = i % (mi_cols as usize);
            let cell = cell_got.expect("every cell of the 64×64 SB must be populated");
            assert_eq!(cell.block_size, BlockSize::Block64x64);
            assert!(
                !cell.skip,
                "skip flag must round-trip as FALSE @ ({mi_row}, {mi_col})"
            );
            match cell.mode {
                BlockMode::Inter { mode, .. } => assert_eq!(mode, InterMode::ZeroMv),
                BlockMode::Intra { .. } => panic!("intra @ ({mi_row}, {mi_col})"),
            }
        }
    }

    /// End-to-end inter-frame round-trip with **64 BLOCK_8x8 leaves**
    /// all carrying `skip = false` + `AllZeroCoefSource`. Sibling of
    /// S4.16i (the skip=true variant) and S4.16k (single-leaf 64×64
    /// non-skip).
    ///
    /// This is the high-water mark for the coef-context propagation
    /// path: every Block8x8 inter leaf with `tx_mode=AllowTx32x32`
    /// uses `Tx8x8` (= 4 4×4 sub-cells per side for Y, 1 for UV).
    /// 64 leaves × 3 planes ≈ ~192 TX blocks. Each block's
    /// `decode_block_tokens` call consumes the running per-plane
    /// `TileCoefCtx` state (above row + left column) and writes back
    /// the `(eob > 0)` flag for downstream blocks. With all-zero
    /// coefs every flag is 0, so the entire frame's coef ctx stays
    /// zeroed — but the encoder must STILL traverse it in lock-step
    /// with the decoder. Any drift in OR-fold position, plane
    /// addressing, or `set_coef_flags` would surface here as a
    /// decode failure or mi_grid mismatch.
    ///
    /// Together with S4.16k (single-leaf non-skip) + S4.16l
    /// (non-zero Y coefs E2E), this provides the heaviest workload
    /// the encoder's wire-format toolbox can produce today and
    /// validates it composes correctly through `Decoder::decode_frame`.
    #[test]
    fn s4_16n_round_trip_dense_8x8_non_skip_through_decoder() {
        use rvp9_core::bitstream::frame::ReferenceMode;
        use rvp9_core::bitstream::header::HeaderTail;
        use rvp9_core::bitstream::sub_parsers::{
            LoopFilterParams, MAX_PROB, PREDICTION_PROBS, QuantizationParams, SEG_TREE_PROBS,
            SegmentationParams, TileInfo,
        };
        use rvp9_core::bitstream::{
            AllZeroCoefSource, BitDepth, ChromaFormat, ColorRange, ColorSpace,
            CompressedHeaderOptions, CompressedHeaderResult, FrameKind, InterpFilter,
            ModesTileBounds, Profile, RawBitWriter, UncompressedHeader, write_compressed_header,
            write_tile_inter, write_tile_intra_keyframe, write_uncompressed_header,
        };
        use rvp9_core::block::BlockSize;
        use rvp9_core::coef_ctx::TileCoefCtx;
        use rvp9_core::entropy::{BoolWriter, FrameContext};
        use rvp9_core::inter::{
            InterModeReadParams, setup_compound_reference_mode, sign_bias_from_header,
        };
        use rvp9_core::intra_pred::IntraMode;
        use rvp9_core::mode_info::{BlockMode, InterMode, ModeInfo, Mv, RefFrame};
        use rvp9_core::transforms::{TxMode, TxSize};

        let mi_rows = 8u32;
        let mi_cols = 8u32;
        let tile_bounds = ModesTileBounds {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };

        let make_tail = |compressed_len: usize| HeaderTail {
            refresh_frame_context: true,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams {
                filter_level: 0,
                sharpness_level: 0,
                mode_ref_delta_enabled: false,
                mode_ref_delta_update: false,
                ref_deltas: [None; 4],
                mode_deltas: [None; 2],
            },
            quantization: QuantizationParams {
                base_qindex: 50,
                y_dc_delta_q: 0,
                uv_dc_delta_q: 0,
                uv_ac_delta_q: 0,
            },
            segmentation: SegmentationParams {
                enabled: false,
                update_map: false,
                temporal_update: false,
                tree_probs: [MAX_PROB; SEG_TREE_PROBS],
                pred_probs: [MAX_PROB; PREDICTION_PROBS],
                update_data: false,
                abs_delta: false,
                feature_data: Default::default(),
            },
            tile_info: TileInfo {
                log2_tile_cols: 0,
                log2_tile_rows: 0,
            },
            first_partition_size: compressed_len as u16,
        };

        // -- Frame 1: keyframe (primes ref pool slot 0) ----------
        let kf_cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let kf_grid: Vec<Option<ModeInfo>> = vec![Some(kf_cell); (mi_rows * mi_cols) as usize];
        let mut kf_pre_fc = FrameContext::new_default();
        let kf_post_fc = kf_pre_fc.clone();
        let kf_opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: true,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let kf_target = CompressedHeaderResult {
            tx_mode: TxMode::AllowTx32x32,
            reference_mode: ReferenceMode::Single,
        };

        let mut kf_compressed_buf = vec![0u8; 16 * 1024];
        let kf_compressed_len = {
            let mut bw = BoolWriter::start_encode(&mut kf_compressed_buf);
            write_compressed_header(&mut bw, &mut kf_pre_fc, &kf_post_fc, kf_opts, &kf_target)
                .expect("kf compressed header");
            bw.stop_encode().expect("stop_encode kf compressed")
        };
        kf_compressed_buf.truncate(kf_compressed_len);

        let mut kf_tile_buf = vec![0u8; 4 * 1024];
        let kf_tile_len = {
            let mut bw = BoolWriter::start_encode(&mut kf_tile_buf);
            let tile_mi_cols =
                (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
            let mut coef_ctx = rvp9_core::coef_ctx::TileCoefCtx::new(tile_mi_cols);
            let coef_source = rvp9_core::bitstream::AllZeroIntraCoefSource::new();
            write_tile_intra_keyframe(
                &mut bw,
                &kf_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                &kf_post_fc,
                None, // AQ.0: seg = None (no segmentation in this test)
                &mut coef_ctx,
                &coef_source,
            )
            .expect("kf write_tile");
            bw.stop_encode().expect("stop_encode kf tile")
        };
        kf_tile_buf.truncate(kf_tile_len);

        let kf_tail = make_tail(kf_compressed_len);
        let kf_hdr = UncompressedHeader {
            kind: FrameKind::KeyFrame,
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(kf_tail),
        };
        let mut kf_hdr_buf = vec![0u8; 256];
        let kf_hdr_len = {
            let mut wb = RawBitWriter::new(&mut kf_hdr_buf);
            write_uncompressed_header(&mut wb, &kf_hdr, kf_compressed_len as u16)
                .expect("kf header");
            wb.finish().expect("kf finish")
        };
        kf_hdr_buf.truncate(kf_hdr_len);
        let mut kf_packet = Vec::new();
        kf_packet.extend_from_slice(&kf_hdr_buf);
        kf_packet.extend_from_slice(&kf_compressed_buf);
        kf_packet.extend_from_slice(&kf_tile_buf);

        let mut dec = Decoder::new();
        let parsed = dec.parse_packet(&kf_packet).expect("parse kf");
        let kf_parsed = ParsedFrame {
            header: parsed.frames()[0].header.clone(),
            header_end_byte: parsed.frames()[0].header_end_byte,
            full_frame_data: parsed.frames()[0].full_frame_data,
        };
        dec.decode_frame(&kf_parsed).expect("decode kf");

        // -- Frame 2: dense 8x8 grain inter with SKIP = FALSE -------
        let inter_post_fc = dec.frame_context(0).clone();
        let inter_cell = ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: TxSize::Tx8x8,
            skip: false, // <-- the structural difference vs S4.16i
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        let inter_grid: Vec<Option<ModeInfo>> =
            vec![Some(inter_cell); (mi_rows * mi_cols) as usize];

        let mut inter_pre_fc = inter_post_fc.clone();
        let inter_target_fc = inter_post_fc.clone();
        let inter_opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: false,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let inter_target = CompressedHeaderResult {
            tx_mode: TxMode::AllowTx32x32,
            reference_mode: ReferenceMode::Single,
        };

        let mut inter_compressed_buf = vec![0u8; 16 * 1024];
        let inter_compressed_len = {
            let mut bw = BoolWriter::start_encode(&mut inter_compressed_buf);
            write_compressed_header(
                &mut bw,
                &mut inter_pre_fc,
                &inter_target_fc,
                inter_opts,
                &inter_target,
            )
            .expect("inter compressed header");
            bw.stop_encode().expect("stop_encode inter compressed")
        };
        inter_compressed_buf.truncate(inter_compressed_len);

        let sign_bias = sign_bias_from_header(&[false, false, false]);
        let (comp_fixed_ref, comp_var_ref) = setup_compound_reference_mode(&sign_bias);
        let inter_params = InterModeReadParams {
            allow_high_precision_mv: false,
            interp_filter: InterpFilter::EightTap,
            reference_mode: ReferenceMode::Single,
            comp_fixed_ref,
            comp_var_ref,
            sign_bias,
        };

        let mut inter_tile_buf = vec![0u8; 64 * 1024];
        let inter_tile_len = {
            let mut bw = BoolWriter::start_encode(&mut inter_tile_buf);
            let mut coef_ctx =
                TileCoefCtx::new((tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1));
            let coef_source = AllZeroCoefSource::new();
            write_tile_inter(
                &mut bw,
                &inter_target_fc,
                &inter_params,
                TxMode::AllowTx32x32,
                &inter_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                None, // AQ.0: seg = None (no segmentation in this test)
                None, // prev_mv_grid = None
                &mut coef_ctx,
                &coef_source,
            )
            .expect("inter write_tile (dense 8x8 non-skip)");
            bw.stop_encode().expect("stop_encode inter tile")
        };
        inter_tile_buf.truncate(inter_tile_len);

        let inter_tail = make_tail(inter_compressed_len);
        let inter_hdr = UncompressedHeader {
            kind: FrameKind::InterFrame {
                reset_frame_context: 0,
                refresh_frame_flags: 0x00,
                ref_frame_indices: [0, 0, 0],
                ref_frame_sign_biases: [false, false, false],
                allow_high_precision_mv: false,
                interp_filter: InterpFilter::EightTap,
            },
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: 64,
            height: 64,
            render_size: None,
            tail: Some(inter_tail),
        };
        let mut inter_hdr_buf = vec![0u8; 256];
        let inter_hdr_len = {
            let mut wb = RawBitWriter::new(&mut inter_hdr_buf);
            write_uncompressed_header(&mut wb, &inter_hdr, inter_compressed_len as u16)
                .expect("inter header");
            wb.finish().expect("inter finish")
        };
        inter_hdr_buf.truncate(inter_hdr_len);

        let mut inter_packet = Vec::new();
        inter_packet.extend_from_slice(&inter_hdr_buf);
        inter_packet.extend_from_slice(&inter_compressed_buf);
        inter_packet.extend_from_slice(&inter_tile_buf);

        let inter_parsed_out = dec
            .parse_packet(&inter_packet)
            .expect("parse dense 8x8 non-skip inter");
        let inter_parsed = ParsedFrame {
            header: inter_parsed_out.frames()[0].header.clone(),
            header_end_byte: inter_parsed_out.frames()[0].header_end_byte,
            full_frame_data: inter_parsed_out.frames()[0].full_frame_data,
        };
        dec.decode_frame(&inter_parsed)
            .expect("decode dense 8x8 non-skip inter — coef_ctx propagation across 64 leaves");

        // Verify all 64 mi-cells came back as BLOCK_8x8 inter
        // LASTFRAME ZEROMV skip=false.
        let got = dec.last_frame_mi_grid();
        assert_eq!(got.len(), 64);
        for (i, cell_got) in got.iter().enumerate() {
            let mi_row = i / 8;
            let mi_col = i % 8;
            let cell = cell_got.expect("every mi-cell populated");
            assert_eq!(cell.block_size, BlockSize::Block8x8);
            assert!(
                !cell.skip,
                "skip must round-trip as FALSE @ ({mi_row}, {mi_col})"
            );
            match cell.mode {
                BlockMode::Inter {
                    mode, ref_frames, ..
                } => {
                    assert_eq!(mode, InterMode::ZeroMv);
                    assert_eq!(ref_frames, [RefFrame::LastFrame, RefFrame::NoRefFrame]);
                }
                BlockMode::Intra { .. } => panic!("intra @ ({mi_row}, {mi_col})"),
            }
        }
    }
}
