//! Wave A G12 §G12 step 5 — panic-safety contract integration tests.
//!
//! Pins the `Decoder` doc-comment guarantee that a `decode_frame`
//! `Err(_)` leaves the decoder in a well-defined state: a subsequent
//! valid keyframe still decodes successfully on the SAME `Decoder`.
//!
//! Two cases per V6 §G12:
//!
//! 1. **Early-error.** A packet that errors before any tile work
//!    starts (here: a keyframe whose `first_partition_size`
//!    overflows its payload → `DecodeError::Bitstream(_)` from the
//!    compressed-header parser). The trivial-poisoning baseline:
//!    no partial state mutation has occurred yet, so this case
//!    should always pass unless the entry point itself is broken.
//!
//! 2. **Mid-tile failure.** A structurally valid inter frame whose
//!    last tile is truncated mid-symbol (entropy reader runs out
//!    of bits inside the partition walk). This is the load-bearing
//!    test — a bug here would mean `ref_frame_map` or a
//!    `FrameContext` bank was half-mutated and the next decode
//!    trips on inconsistent state.
//!
//! Test 2 reuses the encoder's synthetic 2-frame builder rather
//! than re-deriving ~300 lines of inter-frame wire emission. The
//! circular dev-dep is documented in `Cargo.toml`.

use rvp9_core::bitstream::{ColorSpace, FrameKind, RawBitWriter, frame::sync};
use rvp9_decoder::{DecodeError, Decoder, ParsedFrame};
use rvp9_encoder::{Config, Encoder};
use std::num::NonZeroUsize;

/// Single-thread `Decoder` constructor for tests — avoids the
/// `Decoder::new()` thread amplification (see review M2 / G5.1
/// TODOs). Tests here don't exercise tile-threading; pinning to one
/// worker keeps wall-clock predictable and avoids host-CPU-dependent
/// scheduler noise.
fn single_thread_decoder() -> Decoder {
    Decoder::with_threads(NonZeroUsize::new(1).expect("1 is non-zero"))
}

/// The "valid keyframe" fixture both tests reach for after the
/// expected `Err`. `encode_synthetic_dc_skip_keyframe` produces a
/// real decodable keyframe (uncompressed header + compressed
/// header + tile bytes) via the encoder's full wire-format
/// pipeline. The minimal-bits hand-rolled keyframe from the
/// in-crate tests parses but does NOT fully decode (its compressed
/// header is all zeros, which the parser rejects), so we can't
/// reuse it here.
///
/// 64x64 single-SB frame keeps the encoder synthetic minimal and
/// matches the inter-frame fixture in
/// [`mid_tile_failure_then_valid_keyframe_decodes`].
fn build_decodable_keyframe_packet() -> Vec<u8> {
    let enc = Encoder::new(Config::new(64, 64, 50)).expect("valid encoder config");
    enc.encode_synthetic_dc_skip_keyframe()
        .expect("synthetic dc-skip keyframe builds")
}

/// Build a keyframe whose `first_partition_size` is 65000 — far
/// larger than the actual payload. `parse_packet` accepts this
/// (the header parses fine), but `decode_frame` short-circuits with
/// a `DecodeError::Bitstream(Truncated)` (or
/// `TruncatedCompressedHeader`) before touching tile decode or
/// any of the running-state fields (`ref_frame_map`, frame
/// contexts, mv counts). This is the trivial-poisoning case.
fn build_keyframe_overflowing_first_partition_size() -> Vec<u8> {
    let mut buf = vec![0u8; 256];
    let mut w = RawBitWriter::new(&mut buf);
    w.write_literal(u32::from(sync::FRAME_MARKER), 2);
    w.write_bit(0);
    w.write_bit(0);
    w.write_bit(0); // show_existing
    w.write_bit(0); // frame_type = KEY
    w.write_bit(1); // show_frame
    w.write_bit(0); // error_resilient
    w.write_literal(u32::from(sync::SYNC_CODE_0), 8);
    w.write_literal(u32::from(sync::SYNC_CODE_1), 8);
    w.write_literal(u32::from(sync::SYNC_CODE_2), 8);
    w.write_literal(ColorSpace::Bt601 as u32, 3);
    w.write_bit(0);
    w.write_literal(319, 16);
    w.write_literal(239, 16);
    w.write_bit(0);

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
    w.write_bit(0); // log2_tile_rows
    // first_partition_size = 65000 (16-bit field, fits in u16) but
    // the actual payload is ~32 bytes. Compressed-header parse will
    // see the declared size > available, and the parser returns
    // `Truncated` (or the decoder wraps as
    // `TruncatedCompressedHeader`). Either error variant is an
    // acceptable signal that decode_frame `Err`-ed before doing any
    // mutation.
    w.write_literal(65000, 16);
    for _ in 0..8 {
        w.write_literal(0, 8);
    }
    let used = w.bytes_written();
    buf.truncate(used + 8);
    buf
}

/// Test A: early-error case.
///
/// Feed a corrupt packet that fails before any tile work begins,
/// then a valid keyframe on the same `Decoder` — second decode must
/// succeed. Verifies the trivial poisoning case (`?`-propagation
/// from compressed-header parse before any running-state mutation).
#[test]
fn early_error_then_valid_keyframe_decodes() {
    let mut dec = single_thread_decoder();

    // Step 1: feed the broken packet.
    let bad = build_keyframe_overflowing_first_partition_size();
    let parsed = dec
        .parse_packet(&bad)
        .expect("uncompressed header parses; only the compressed-header size is wrong");
    assert_eq!(parsed.count(), 1, "single frame in packet");

    // Snapshot the parsed header / bytes before consuming the
    // borrow; we need them to build a fresh ParsedFrame that the
    // decoder can consume without re-borrowing `dec`.
    let header = parsed.frames()[0].header.clone();
    let header_end = parsed.frames()[0].header_end_byte;
    let body: Vec<u8> = parsed.frames()[0].full_frame_data.to_vec();
    drop(parsed);

    let frame = ParsedFrame {
        header,
        header_end_byte: header_end,
        full_frame_data: &body,
    };
    let err = dec
        .decode_frame(&frame)
        .expect_err("decode_frame must Err — first_partition_size overflows payload");
    // Pin the specific variant the corrupt fixture actually
    // produces today. Pre-G12-review this assertion was a `match`
    // accepting either `Bitstream(_)` or
    // `TruncatedCompressedHeader { .. }`, which tolerated silent
    // variant drift. The compressed-header bounds check at
    // `decode_frame` lib.rs:809 fires before the compressed-header
    // parser ever sees the truncated payload, so the variant is
    // always `TruncatedCompressedHeader { declared: 65014,
    // available: 30 }` for the current fixture. If the decoder is
    // refactored to surface a different variant (e.g. the parser
    // runs first and emits `Bitstream(_)`), this assertion will
    // fail loudly — that is the intent: the test catches the drift
    // rather than silently re-classifying it.
    assert!(
        matches!(err, DecodeError::TruncatedCompressedHeader { .. }),
        "expected TruncatedCompressedHeader for corrupt-header fixture, got {err:?}",
    );

    // Step 2: feed a valid keyframe on the SAME decoder. Must
    // succeed end-to-end, proving no poisoning occurred.
    let good = build_decodable_keyframe_packet();
    let parsed2 = dec.parse_packet(&good).expect("valid keyframe parses");
    let header2 = parsed2.frames()[0].header.clone();
    let header_end2 = parsed2.frames()[0].header_end_byte;
    let body2: Vec<u8> = parsed2.frames()[0].full_frame_data.to_vec();
    assert!(matches!(header2.kind, FrameKind::KeyFrame));
    drop(parsed2);

    let frame2 = ParsedFrame {
        header: header2,
        header_end_byte: header_end2,
        full_frame_data: &body2,
    };
    let out = dec
        .decode_frame(&frame2)
        .expect("valid keyframe must decode successfully after the prior early-error");
    // The synthetic keyframe is 64 × 64, so the returned FrameBuf
    // carries those dimensions. Defensive sanity check that we
    // actually got a real decode (not some zero-sized stub).
    assert_eq!(out.y_plane().width(), 64);
    assert_eq!(out.y_plane().height(), 64);
}

/// Test B: mid-tile failure case.
///
/// 1. Drive the synthetic 2-frame keyframe+inter sequence through
///    `Encoder::encode_synthetic_two_frame_kf_inter_non_skip` to
///    get a real superframe packet.
/// 2. Parse the packet, decode the keyframe successfully (this
///    populates the ref_frame_map).
/// 3. Truncate the inter frame's tile body by 16 bytes (BoolReader
///    runs out of bits mid-symbol — last 16 bytes of a non-skip
///    inter tile carry residual tokens, so the truncation hits
///    deep in `decode_tile_inter`'s coef walk).
/// 4. Feed the truncated inter to `decode_frame`; assert `Err`.
/// 5. Feed a FRESH valid keyframe on the same decoder; assert it
///    decodes successfully. This is the load-bearing assertion —
///    a bug means the truncated inter left `ref_frame_map` or a
///    `FrameContext` bank in a half-mutated state and the keyframe
///    trips on it.
#[test]
fn mid_tile_failure_then_valid_keyframe_decodes() {
    // 64x64 single-SB frame keeps the encoder synthetic minimal.
    let enc = Encoder::new(Config::new(64, 64, 50)).expect("valid encoder config");
    let packet = enc
        .encode_synthetic_two_frame_kf_inter_non_skip()
        .expect("synthetic 2-frame packet builds");

    let mut dec = single_thread_decoder();

    // Step 1: parse the packet into its 2 frames.
    let parsed = dec.parse_packet(&packet).expect("synthetic packet parses");
    assert_eq!(parsed.count(), 2, "kf + inter");
    let kf_header = parsed.frames()[0].header.clone();
    let kf_end = parsed.frames()[0].header_end_byte;
    let kf_bytes: Vec<u8> = parsed.frames()[0].full_frame_data.to_vec();
    let inter_header = parsed.frames()[1].header.clone();
    let inter_end = parsed.frames()[1].header_end_byte;
    let inter_bytes: Vec<u8> = parsed.frames()[1].full_frame_data.to_vec();
    drop(parsed);

    // Step 2: decode the keyframe (success).
    let kf_frame = ParsedFrame {
        header: kf_header,
        header_end_byte: kf_end,
        full_frame_data: &kf_bytes,
    };
    let _kf_buf = dec
        .decode_frame(&kf_frame)
        .expect("keyframe must decode (synthetic baseline)");

    // Step 3: truncate the inter frame's tile body by 16 bytes.
    // The inter payload contains: uncompressed header, compressed
    // header (first_partition_size bytes), then tile bytes. The
    // tail of the inter body is the last tile's entropy stream —
    // truncating from the end forces a mid-symbol BoolReader
    // failure inside `decode_tile_inter`.
    assert!(
        inter_bytes.len() > 16,
        "inter payload too short to truncate by 16 bytes (got {})",
        inter_bytes.len(),
    );
    let truncated: Vec<u8> = inter_bytes[..inter_bytes.len() - 16].to_vec();
    let bad_inter = ParsedFrame {
        header: inter_header,
        header_end_byte: inter_end,
        full_frame_data: &truncated,
    };
    let err = dec.decode_frame(&bad_inter).expect_err(
        "truncated inter must Err inside decode_tile_inter (BoolReader \
         runs out of bits mid-symbol on a non-skip residual token)",
    );
    // Any decode-layer error is fine — the contract is just that
    // we Err rather than silently produce garbage. Specifically
    // EmptyInterRef is NOT acceptable (would indicate the keyframe
    // didn't populate ref_frame_map, i.e. step 2 was a no-op).
    match &err {
        DecodeError::EmptyInterRef { .. } => panic!(
            "unexpected EmptyInterRef — the keyframe should have \
             populated every ref slot; got {err:?}",
        ),
        DecodeError::WrongFrameKind => panic!(
            "unexpected WrongFrameKind — inter header is well-formed; \
             got {err:?}",
        ),
        _ => {} // any other decode error variant is acceptable
    }

    // Step 4 — load-bearing: feed a fresh valid keyframe on the
    // SAME decoder. It must succeed. If `ref_frame_map` or a
    // `FrameContext` bank was half-mutated by step 3's failure,
    // this decode trips here.
    let good_kf = build_decodable_keyframe_packet();
    let parsed2 = dec
        .parse_packet(&good_kf)
        .expect("post-failure keyframe parses");
    let header2 = parsed2.frames()[0].header.clone();
    let end2 = parsed2.frames()[0].header_end_byte;
    let body2: Vec<u8> = parsed2.frames()[0].full_frame_data.to_vec();
    drop(parsed2);

    let frame2 = ParsedFrame {
        header: header2,
        header_end_byte: end2,
        full_frame_data: &body2,
    };
    let out = dec
        .decode_frame(&frame2)
        .expect("post-failure keyframe must decode — panic-safety contract");
    assert_eq!(out.y_plane().width(), 64);
    assert_eq!(out.y_plane().height(), 64);
}

/// Test C: mid-tile failure followed by a valid INTER frame.
///
/// The existing [`mid_tile_failure_then_valid_keyframe_decodes`]
/// only proves a subsequent KEYFRAME survives the failure —
/// keyframes do a hard reset (`vp9_setup_past_independence`,
/// `refresh_frame_flags = 0xFF`), so they paper over any bank
/// poisoning. The V6 §G12 contract acknowledges banks aren't
/// reverted on error, so the keyframe path is a weak assertion.
///
/// This test exercises the inter-recovery path: after the failed
/// inter, we feed ANOTHER inter frame (extracted from a fresh
/// synthetic kf+inter packet) on the same decoder. The new inter
/// names `ref_frame_indices = [0, 0, 0]` — i.e. it depends on the
/// keyframe that decoded in step 2, which lives in
/// `ref_frame_map[0]`. This pins:
///
/// 1. `ref_frame_map` is preserved across the failed inter (the
///    keyframe's slot is still resolvable — otherwise we get
///    `EmptyInterRef` and the test panics with a specific message).
/// 2. The actual end-state of the bank-poisoning. If the new inter
///    succeeds, banks are robust enough for an inter recovery —
///    great. If it fails with a non-`EmptyInterRef` decode error
///    (e.g. `Bitstream(_)` or `TileDecode`), that pins the
///    documented "banks may be poisoned" behavior as observable
///    rather than hypothetical. Either outcome is acceptable
///    per the contract — what we DON'T accept is `EmptyInterRef`
///    (which would mean the ref-map was corrupted, NOT the bank).
#[test]
fn mid_tile_failure_then_valid_inter_decodes() {
    let enc = Encoder::new(Config::new(64, 64, 50)).expect("valid encoder config");
    let packet1 = enc
        .encode_synthetic_two_frame_kf_inter_non_skip()
        .expect("synthetic 2-frame packet #1 builds");
    let packet2 = enc
        .encode_synthetic_two_frame_kf_inter_non_skip()
        .expect("synthetic 2-frame packet #2 builds");

    let mut dec = single_thread_decoder();

    // Step 1: decode the keyframe from packet #1 — populates
    // `ref_frame_map` (every slot points at the keyframe's pool
    // slot since keyframes do `refresh_frame_flags = 0xFF`).
    let parsed1 = dec.parse_packet(&packet1).expect("packet #1 parses");
    assert_eq!(parsed1.count(), 2, "kf + inter");
    let kf_header = parsed1.frames()[0].header.clone();
    let kf_end = parsed1.frames()[0].header_end_byte;
    let kf_bytes: Vec<u8> = parsed1.frames()[0].full_frame_data.to_vec();
    let inter1_header = parsed1.frames()[1].header.clone();
    let inter1_end = parsed1.frames()[1].header_end_byte;
    let inter1_bytes: Vec<u8> = parsed1.frames()[1].full_frame_data.to_vec();
    drop(parsed1);

    let kf_frame = ParsedFrame {
        header: kf_header,
        header_end_byte: kf_end,
        full_frame_data: &kf_bytes,
    };
    let _kf_buf = dec.decode_frame(&kf_frame).expect("keyframe decodes");

    // Step 2: truncate the inter from packet #1 and feed it; assert Err.
    assert!(inter1_bytes.len() > 16);
    let truncated: Vec<u8> = inter1_bytes[..inter1_bytes.len() - 16].to_vec();
    let bad_inter = ParsedFrame {
        header: inter1_header,
        header_end_byte: inter1_end,
        full_frame_data: &truncated,
    };
    let err = dec
        .decode_frame(&bad_inter)
        .expect_err("truncated inter must Err");
    assert!(
        !matches!(err, DecodeError::EmptyInterRef { .. }),
        "step 2 saw EmptyInterRef — the keyframe was not properly \
         installed into ref_frame_map; got {err:?}",
    );

    // Step 3 — load-bearing: extract the (untouched) inter from
    // packet #2 and feed it. This inter has
    // `ref_frame_indices = [0, 0, 0]` and depends on
    // `ref_frame_map[0]` — which still points at the keyframe from
    // step 1 if the ref-map survived step 2's failure.
    let parsed2 = dec.parse_packet(&packet2).expect("packet #2 parses");
    let inter2_header = parsed2.frames()[1].header.clone();
    let inter2_end = parsed2.frames()[1].header_end_byte;
    let inter2_bytes: Vec<u8> = parsed2.frames()[1].full_frame_data.to_vec();
    drop(parsed2);
    let good_inter = ParsedFrame {
        header: inter2_header,
        header_end_byte: inter2_end,
        full_frame_data: &inter2_bytes,
    };
    let result = dec.decode_frame(&good_inter);
    match result {
        Ok(buf) => {
            // Best case: ref-map survived AND bank poisoning didn't
            // unrecoverably break the inter path.
            assert_eq!(buf.y_plane().width(), 64);
            assert_eq!(buf.y_plane().height(), 64);
        }
        Err(DecodeError::EmptyInterRef { slot }) => {
            panic!(
                "ref_frame_map was corrupted by the failed inter: \
                 inter #2's ref_frame_indices = [0,0,0] but slot {slot} \
                 is unresolvable. This is the bug G12 R2's per-slot \
                 refcount was supposed to prevent.",
            );
        }
        Err(_) => {
            // Bank poisoning surfacing as a non-EmptyInterRef decode
            // error is documented-accepted behavior per the V6 §G12
            // contract. The test value is in distinguishing this case
            // from the EmptyInterRef case above — i.e. ref-map
            // preserved, banks poisoned. The arm intentionally does
            // nothing: the panic above pins the strong contract;
            // accepting Err here pins the weak (documented) contract.
        }
    }
}

/// Sanity check the assertion subjects of the two panic-safety
/// tests share a state-prep baseline: a fresh `Decoder` with no
/// prior frames is the correct starting point for both. This
/// pins the invariant that future refactors of `Decoder::new`
/// don't accidentally leave it in a partially-poisoned state
/// (`Default::default` regression).
#[test]
fn fresh_decoder_baseline() {
    let dec = single_thread_decoder();
    // No frames decoded yet → frame pool empty. (The test exists
    // to catch a regression where `Decoder::new` accidentally
    // started populating a slot.)
    assert_eq!(dec.frame_pool().occupied_count(), 0);
    // Auto-sized thread count must be >= 1 by NonZeroUsize
    // construction.
    assert!(dec.num_threads().get() >= 1);
}
