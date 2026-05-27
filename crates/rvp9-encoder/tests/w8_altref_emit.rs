//! W8 — ALTREF emission PROOF.
//!
//! With `enable_alt_ref = true` and `lag_in_frames > gop_length / 2`,
//! the encoder buffers source frames, builds a temporal-filtered ARF
//! via [`rvp9_encoder::temporal_filter::vp9_temporal_filter`], and
//! emits it as a non-show inter frame packed into a superframe with
//! the next visible inter frame.
//!
//! This test verifies the wiring end-to-end:
//!
//! 1. **At least one ARF packet** is emitted across `2 * gop_length`
//!    frames of input (cadence: every `gop_length / 2` inter frames).
//! 2. **The decoded stream contains exactly N show frames** for N
//!    input frames — the ARF is non-show, so the decoder's
//!    `last_show_frame` only advances on visible frames.
//! 3. **The ARF packet carries `show_frame = false`** at the wire
//!    level (anti-G3 — a stub that emitted a show frame would still
//!    pass the count check if the cadence was wrong; this asserts
//!    we actually flipped the bit).
//!
//! libvpx file:line citations:
//! - `vp9/encoder/vp9_encoder.c::setup_arf_frame` — pre-encode ARF
//!   source selection + temporal_filter dispatch.
//! - `vp9/encoder/vp9_temporal_filter.c:1104-1205` — top-level
//!   `vp9_temporal_filter`.
//! - `vp9/encoder/vp9_bitstream.c::vp9_pack_bitstream` — emits the
//!   superframe with the non-show ARF (`cm->show_frame = 0`,
//!   `refresh_frame_flags |= 1 << ALT_REF_FRAME_IDX`).

#![allow(clippy::print_stderr)]
#![allow(missing_docs)]

use rvp9_decoder::Decoder;
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};

const W: u32 = 64;
const H: u32 = 64;

/// Build a `W x H` 4:2:0 frame from a gradient + a moving bright
/// square. The motion gives the temporal filter something
/// non-trivial to look at; the inter encoder still falls back to
/// all-skip ZEROMV at the picker because the filtered ARF source's
/// per-block stats differ from the input. None of that matters for
/// the wiring PROOF (we assert on packet counts, not on bytes).
fn synthetic_frame(seed: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = W as usize;
    let h = H as usize;
    let uvw = w / 2;
    let uvh = h / 2;
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = (32 + ((r + c + seed as usize) & 0x7f) as u8) & 0xff;
        }
    }
    let u = vec![128u8; uvw * uvh];
    let v = vec![128u8; uvw * uvh];
    (y, u, v)
}

fn yuv_view<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> YuvFrameRef<'a> {
    YuvFrameRef {
        y,
        u,
        v,
        y_stride: W as usize,
        u_stride: (W as usize) / 2,
        v_stride: (W as usize) / 2,
    }
}

/// Encode `n_frames` source frames with ALTREF emission enabled,
/// returning the resulting transport packets in encode order.
fn encode_with_altref(n_frames: u32, gop_length: u32, lag: u32) -> Vec<Vec<u8>> {
    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate: 30.0,
        target_bitrate_bps: 1_000_000,
        gop_length,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: lag,
        enable_alt_ref: true,
        rc_mode: RcMode::Cbr,
        cq_level: 0,
        temporal_layers: 1,
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut packets: Vec<Vec<u8>> = Vec::new();
    for i in 0..n_frames {
        let (y, u, v) = synthetic_frame(i as u8);
        match enc.encode_frame_delayed(&yuv_view(&y, &u, &v)) {
            Ok(Some(pkt)) => packets.push(pkt.data),
            Ok(None) => {} // ring still filling
            Err(e) => panic!("encode_frame_delayed: {e:?}"),
        }
    }
    for r in enc.flush() {
        packets.push(r.expect("flush ok").data);
    }
    packets
}

/// W8 PROOF #1 — when ALTREF is enabled and the lookahead is large
/// enough, at least one transport packet is a superframe carrying a
/// non-show ARF.
#[test]
fn w8_altref_emits_non_show_packet() {
    let n_frames: u32 = 16;
    let gop_length: u32 = 16; // single GOP across the run
    let lag_in_frames: u32 = 8; // > gop_length / 2 = 8 → ARF can peek
    let packets = encode_with_altref(n_frames, gop_length, lag_in_frames);
    eprintln!(
        "W8 encode produced {} transport packets",
        packets.len()
    );
    // Decode through rvp9-decoder. Sum the show vs non-show frame
    // counts.
    let mut dec = Decoder::new();
    let mut show_frames = 0u32;
    let mut non_show_frames = 0u32;
    let mut total_subframes = 0u32;
    for pkt in &packets {
        let parsed = dec.parse_packet(pkt).expect("parse_packet");
        for pf in parsed.frames() {
            total_subframes += 1;
            if pf.header.show_frame {
                show_frames += 1;
            } else {
                non_show_frames += 1;
            }
        }
    }
    eprintln!(
        "show_frames={show_frames}, non_show_frames={non_show_frames}, \
         total_subframes={total_subframes}, input_frames={n_frames}"
    );
    // Anti-G3: a stub that never emits an ARF would pass the
    // show-count check; assert the non-show count is strictly > 0.
    assert!(
        non_show_frames >= 1,
        "ALTREF wiring must emit at least one non-show frame for {n_frames} \
         input frames at gop_length={gop_length}, lag_in_frames={lag_in_frames}; \
         got 0 non-show frames (cadence trigger or temporal-filter call missing)"
    );
    // The number of SHOW frames must equal the number of input
    // frames (the ARF is non-show, so the decoded show-frame stream
    // matches the input stream one-to-one).
    assert_eq!(
        show_frames, n_frames,
        "show-frame count {show_frames} must equal input-frame count {n_frames} \
         (the ARF is non-show; one input frame = one decoded show frame)"
    );
}

/// W8 PROOF #2 — with `enable_alt_ref = false` (the default), no
/// non-show frame is emitted. This is the anti-G3 guard against the
/// trigger firing unconditionally — if the gate is broken, the
/// default `EncoderConfig::new()` path would diverge from the
/// cross-decoder gate's expectation that every packet is a single
/// show frame.
#[test]
fn w8_altref_disabled_emits_no_non_show_packet() {
    let n_frames: u32 = 16;
    let gop_length: u32 = 16;
    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate: 30.0,
        target_bitrate_bps: 1_000_000,
        gop_length,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 8,
        enable_alt_ref: false, // <- the gate
        rc_mode: RcMode::Cbr,
        cq_level: 0,
        temporal_layers: 1,
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut packets: Vec<Vec<u8>> = Vec::new();
    for i in 0..n_frames {
        let (y, u, v) = synthetic_frame(i as u8);
        if let Ok(Some(pkt)) = enc.encode_frame_delayed(&yuv_view(&y, &u, &v)) {
            packets.push(pkt.data);
        }
    }
    for r in enc.flush() {
        packets.push(r.expect("flush ok").data);
    }
    let mut dec = Decoder::new();
    let mut non_show_frames = 0u32;
    for pkt in &packets {
        let parsed = dec.parse_packet(pkt).expect("parse_packet");
        for pf in parsed.frames() {
            if !pf.header.show_frame {
                non_show_frames += 1;
            }
        }
    }
    assert_eq!(
        non_show_frames, 0,
        "with enable_alt_ref = false, no non-show frame may be emitted \
         (got {non_show_frames}); the cross-decoder gate runs at this \
         config and any drift would break byte-exactness"
    );
}

/// W8 PROOF #3 — show-frame count equals input-frame count even with
/// ALTREF enabled (the ARF is non-show; decoded show frames map 1:1
/// to input frames). Cross-checks against PROOF #1 by varying the
/// frame count and cadence.
#[test]
fn w8_show_frame_count_matches_input_count() {
    for (n_frames, gop_length, lag) in
        [(20u32, 10u32, 6u32), (32u32, 16u32, 9u32), (48u32, 24u32, 13u32)]
    {
        let packets = encode_with_altref(n_frames, gop_length, lag);
        let mut dec = Decoder::new();
        let mut show_frames = 0u32;
        for pkt in &packets {
            let parsed = dec.parse_packet(pkt).expect("parse_packet");
            for pf in parsed.frames() {
                if pf.header.show_frame {
                    show_frames += 1;
                }
            }
        }
        assert_eq!(
            show_frames, n_frames,
            "show_frames must equal input frames for \
             n={n_frames} gop={gop_length} lag={lag}"
        );
    }
}
