//! Anti-regression oracle for the per-frame decode-target buffer POOL
//! (the rvp9 analogue of libvpx `get_free_fb`).
//!
//! Byte-exactness of the recycle path is covered by `conformance` +
//! `real_world` (which drop each `FrameBuf` and check per-frame MD5).
//! This test guards the OPTIMIZATION itself: that the decoder actually
//! RECYCLES decode-target buffers across frames instead of allocating a
//! fresh — page-faulting — buffer every frame. A regression that made
//! `DecoderFramePool::take_recyclable` always miss would stay byte-exact
//! (every frame just `alloc`s fresh) yet silently lose the whole win;
//! only an efficacy oracle catches that.
//!
//! Mechanism: decode a 100-frame constant-resolution clip, dropping each
//! returned `FrameBuf` so its pool slot becomes recyclable, and assert
//! the count of FRESH allocations (`RVP9_DECODE_TARGET_ALLOCS`) plateaus
//! at the pool working-set size (≤ `DECODER_FRAME_BUFFERS`) rather than
//! growing once per frame.
//!
//! Lives in its OWN test binary so the process-global alloc counter is
//! not perturbed by other decode tests running concurrently in-process.

use rvp9_core::frame::DECODER_FRAME_BUFFERS;
use rvp9_decoder::{Decoder, RVP9_DECODE_TARGET_ALLOCS};
use rvp9_harness::container::read_packets;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

fn vector(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("tests")
        .join("conformance")
        .join("vectors")
        .join(name)
}

#[cfg_attr(
    not(rvp9_conformance_vectors),
    ignore = "requires tests/conformance/vectors; run scripts/fetch-vectors.sh"
)]
#[test]
fn decode_target_buffer_is_pooled_after_warmup() {
    // 100-frame constant-resolution clip — long enough that the pool
    // working set (≤ DECODER_FRAME_BUFFERS slots) fills within the first
    // handful of frames and every subsequent frame recycles.
    let path = vector("vp90-2-09-aq2.webm");
    let (data, packets) = read_packets(&path).expect("container parse");
    let mut decoder = Decoder::with_threads(NonZeroUsize::new(1).expect("1 != 0"));

    let before = RVP9_DECODE_TARGET_ALLOCS.load(Ordering::Relaxed);
    let mut n_decoded = 0usize;
    for range in &packets {
        let parsed = match decoder.parse_packet(range.slice(&data)) {
            Ok(p) => p,
            Err(_) => continue,
        };
        for frame in parsed {
            // Drop the returned FrameBuf immediately (a streaming consumer
            // / vpxdec does not retain decoded frames), making its pool
            // slot recyclable for the next frame.
            if decoder.decode_frame(&frame).is_ok() {
                n_decoded += 1;
            }
        }
    }
    let fresh_allocs = RVP9_DECODE_TARGET_ALLOCS
        .load(Ordering::Relaxed)
        .wrapping_sub(before);

    assert!(
        n_decoded >= 50,
        "needed a multi-frame clip; only {n_decoded} frames decoded",
    );
    // Never-recycling would give `fresh_allocs == n_decoded` (~100). With
    // recycling it converges to the working-set size, which can never
    // exceed the pool's `DECODER_FRAME_BUFFERS` slots.
    assert!(
        fresh_allocs <= DECODER_FRAME_BUFFERS as u64,
        "decode-target pool did not recycle: {fresh_allocs} fresh allocs over \
         {n_decoded} frames (expected <= {DECODER_FRAME_BUFFERS} = pool size). \
         A regression here means every frame re-allocates a fresh, \
         page-faulting decode buffer.",
    );
    eprintln!(
        "buffer-pool recycle: {fresh_allocs} fresh decode-target allocs over \
         {n_decoded} frames (pool size {DECODER_FRAME_BUFFERS})",
    );
}
