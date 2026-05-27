// Integration test: stderr IS the output mechanism (diagnostic
// reporting), mirroring `cross_decoder_gate.rs` / `w_buffer_pool.rs`.
// The workspace's `print_stderr = "deny"` lint targets library code, not
// this binary.
#![allow(clippy::print_stderr)]
#![allow(missing_docs)]

//! GOP topology — PROOF: the libvpx-faithful `GfGroup` schedule
//! (`rvp9_encoder::gop`) is the SINGLE SOURCE OF TRUTH for each display
//! frame's update TYPE + wire `refresh_frame_flags`, and wiring it into
//! the frame loop is a **gate-safe, byte-identical refactor**.
//!
//! This replaces the encoder's ad-hoc `inter_frames_since_arf` /
//! `frames_since_golden_refresh` counter decisions with reads from the
//! `GfGroup` struct (modelled on libvpx's `GF_GROUP`,
//! `vp9_firstpass.h:80-100`). The emitted frame types + refresh flags
//! are IDENTICAL to the pre-refactor encoder.
//!
//! Oracles:
//!
//! 1. **Anti-G3 schedule fidelity (the critical anti-stub oracle).**
//!    Encode a representative clip, decode every emitted packet, and
//!    extract each frame's actual `(FrameKind, refresh_frame_flags,
//!    show_frame)` from its uncompressed header. Assert this EXACTLY
//!    matches the `GfGroup.schedule(frame_idx)` prediction for every
//!    frame: KF (`0xFF`) at offset 0 + every `gop_length`; GOLDEN
//!    refresh (`0x03`) at inter offsets that are multiples of
//!    `GOLDEN_REFRESH_PERIOD`; normal inter (`0x01`) elsewhere. If the
//!    `GfGroup` were stubbed (e.g. always `LfUpdate`) or diverged from
//!    the real emitted schedule, this fails.
//!
//! 2. **Byte-identity regression (the refactor oracle).** Encode a fixed
//!    synthetic clip and assert the concatenated packet bytes hash to a
//!    captured golden. Proves the refactor changed *nothing* about the
//!    emitted bitstream. Pins `RVP9_A4_RD_PARTITION_DISABLE` (mirroring
//!    `tests/w_buffer_pool.rs`) so the golden is stable against the A4
//!    partition descent.
//!
//! libvpx file:line citations:
//! - `vp9/encoder/vp9_firstpass.h:62-71` — `FRAME_UPDATE_TYPE` enum.
//! - `vp9/encoder/vp9_firstpass.h:80-100` — `GF_GROUP` struct
//!   (`update_type[]`, `arf_src_offset[]`, `frame_gop_index[]`,
//!   `gf_group_size`).
//! - `vp9/encoder/vp9_firstpass.c:2257-2302` — `define_gf_group_structure`.
//! - `vp9/encoder/vp9_firstpass.c:3426-3428` — KF handler
//!   (`update_type[0] = KF_UPDATE`).

use rvp9_core::bitstream::header::FrameKind as WireFrameKind;
use rvp9_decoder::Decoder;
use rvp9_encoder::gop::{FrameUpdateType, GfGroup, GOLDEN_REFRESH_PERIOD};
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};

const W: u32 = 128;
const H: u32 = 128;

fn yuv_view<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8], w: usize) -> YuvFrameRef<'a> {
    YuvFrameRef {
        y,
        u,
        v,
        y_stride: w,
        u_stride: w / 2,
        v_stride: w / 2,
    }
}

/// Deterministic moving-gradient source frame at index `idx` (same
/// generator as `tests/w_buffer_pool.rs` so the golden is comparable).
fn moving_gradient(idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = W as usize;
    let h = H as usize;
    let uvw = w / 2;
    let uvh = h / 2;
    let mut y = vec![0u8; w * h];
    let shift = (idx * 3) as usize;
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = ((r + c + shift) * 2) as u8;
        }
    }
    let mut u = vec![0u8; uvw * uvh];
    let mut v_pl = vec![0u8; uvw * uvh];
    for r in 0..uvh {
        for c in 0..uvw {
            u[r * uvw + c] = (128 + ((r + shift) as i32 % 40 - 20)) as u8;
            v_pl[r * uvw + c] = (128 + ((c + shift) as i32 % 40 - 20)) as u8;
        }
    }
    (y, u, v_pl)
}

/// FNV-1a 64-bit hash — dependency-free stable digest (same as
/// `tests/w_buffer_pool.rs`).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

const GOP_LENGTH: u32 = 16;
const N_FRAMES: u32 = 40; // > 2 GOPs so KF cadence + repeated GOLDEN refresh land

fn fixed_config() -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        framerate: 30.0,
        target_bitrate_bps: 1_000_000,
        gop_length: GOP_LENGTH,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        enable_alt_ref: false,
        rc_mode: RcMode::Cbr,
        cq_level: 0,
        temporal_layers: 1,
        min_qindex: 4,
        max_qindex: 255,
    }
}

/// Pin the pre-A4 partition path (mirrors `tests/w_buffer_pool.rs`) so
/// the byte-identity golden is stable. SAFETY: integration binary; all
/// tests here want the descent off, so set-only is race-free.
fn pin_a4() {
    unsafe {
        std::env::set_var("RVP9_A4_RD_PARTITION_DISABLE", "1");
    }
}

/// Encode the fixed clip and return the concatenated packet bytes.
fn encode_fixed_clip() -> Vec<u8> {
    pin_a4();
    let mut enc = StreamEncoder::new(fixed_config()).expect("StreamEncoder::new");
    let mut stream = Vec::new();
    for idx in 0..N_FRAMES {
        let (y, u, v) = moving_gradient(idx);
        let pkt = enc
            .encode_frame(&yuv_view(&y, &u, &v, W as usize))
            .expect("encode_frame");
        stream.extend_from_slice(&pkt.data);
    }
    stream
}

/// PROOF oracle 1 — ANTI-G3 SCHEDULE FIDELITY.
///
/// Decode each emitted packet's uncompressed header and assert the
/// actual `(frame_type, refresh_frame_flags)` matches the `GfGroup`
/// schedule prediction for that display index. The `GfGroup` must be the
/// real schedule the encoder followed — a stub or divergence fails here.
#[test]
fn gop_topology_schedule_matches_emitted_frames() {
    pin_a4();
    let cfg = fixed_config();
    // The independent oracle: the GfGroup built from the same inputs.
    let gf_group = GfGroup::new(cfg.gop_length, cfg.enable_alt_ref);

    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut dec = Decoder::new();

    let mut checked = 0u32;
    let mut kf_count = 0u32;
    let mut gf_count = 0u32;
    for idx in 0..N_FRAMES {
        let (y, u, v) = moving_gradient(idx);
        let pkt = enc
            .encode_frame(&yuv_view(&y, &u, &v, W as usize))
            .expect("encode_frame");
        let parsed = dec.parse_packet(&pkt.data).expect("parse_packet");

        // `enable_alt_ref = false` ⇒ no superframes ⇒ exactly one frame
        // per packet. (When ALTREF is wired the ARF would be a separate
        // non-show frame here; this config keeps the 1:1 mapping so the
        // packet index == display index.)
        assert_eq!(
            parsed.count(),
            1,
            "frame {idx}: expected a single (non-superframe) packet",
        );
        let header = parsed.frames()[0].header.clone();

        // Extract the ACTUAL emitted (frame_type, refresh_frame_flags).
        let (actual_is_kf, actual_refresh) = match header.kind {
            WireFrameKind::KeyFrame => (true, 0xFFu8),
            WireFrameKind::InterFrame {
                refresh_frame_flags,
                ..
            } => (false, refresh_frame_flags),
            other => panic!("frame {idx}: unexpected emitted kind {other:?}"),
        };
        // All emitted frames in this config are show frames.
        assert!(
            header.show_frame,
            "frame {idx}: expected a show frame",
        );

        // The PREDICTION from the GfGroup struct.
        let sched = gf_group.schedule(idx);
        let pred_is_kf = matches!(sched.update_type, FrameUpdateType::KfUpdate);

        eprintln!(
            "f{idx:>3} offset={:>2} pred update_type={:?} refresh={:#04x} | \
             emitted kf={} refresh={:#04x}",
            gf_group.gop_offset(idx),
            sched.update_type,
            sched.refresh_frame_flags,
            actual_is_kf,
            actual_refresh,
        );

        assert_eq!(
            pred_is_kf, actual_is_kf,
            "frame {idx}: GfGroup predicted KF={pred_is_kf} but emitted KF={actual_is_kf}",
        );
        assert_eq!(
            sched.refresh_frame_flags, actual_refresh,
            "frame {idx}: GfGroup predicted refresh_frame_flags={:#04x} but emitted {:#04x} \
             — the schedule diverges from the real encoder decision",
            sched.refresh_frame_flags, actual_refresh,
        );

        // Cross-check the update_type discriminant against the emitted
        // refresh mask (independent of `pred_is_kf`): GfUpdate ⇔ 0x03,
        // LfUpdate ⇔ 0x01, KfUpdate ⇔ 0xFF.
        match sched.update_type {
            FrameUpdateType::KfUpdate => {
                assert_eq!(actual_refresh, 0xFF, "f{idx} KfUpdate must emit 0xFF");
                kf_count += 1;
            }
            FrameUpdateType::GfUpdate => {
                assert_eq!(actual_refresh, 0x03, "f{idx} GfUpdate must emit 0x03");
                gf_count += 1;
            }
            FrameUpdateType::LfUpdate => {
                assert_eq!(actual_refresh, 0x01, "f{idx} LfUpdate must emit 0x01");
            }
            other => panic!("f{idx}: unexpected scheduled update_type {other:?}"),
        }
        checked += 1;
    }

    assert_eq!(checked, N_FRAMES, "every frame must be checked");
    // Representative-encode coverage assertions: across 40 frames at
    // gop=16, KFs land at 0/16/32 (3) and GOLDEN refresh at offset 8 of
    // each GOP — offsets 8, 24 (and 40 == idx 40 is out of range), i.e.
    // 2 GF_UPDATEs. These hard counts make the test FAIL if the schedule
    // collapses (e.g. a stub that never marks GF/KF).
    assert_eq!(kf_count, 3, "expected KFs at idx 0/16/32");
    assert_eq!(gf_count, 2, "expected GOLDEN refresh at idx 8/24");
    // Sanity: the constant the schedule keys off is the documented 8.
    assert_eq!(GOLDEN_REFRESH_PERIOD, 8);
}

/// PROOF oracle 2 — BYTE-IDENTITY REGRESSION.
///
/// The captured golden is the FNV-1a hash of the concatenated packet
/// bytes from `encode_fixed_clip`. The `GfGroup` refactor changed the
/// encoder's internal schedule bookkeeping (replacing the ad-hoc
/// counters), NOT the emitted bitstream — so the output MUST be
/// byte-identical. If this fails, the schedule wiring diverges from the
/// pre-refactor counter decisions. Do NOT update the golden to match a
/// changed stream — fix the schedule.
#[test]
fn gop_topology_byte_identity() {
    // Golden captured against the GfGroup-wired encoder AND confirmed
    // byte-identical to the PRE-refactor counter encoder: both produce
    // len = 26015, fnv1a = 0xc077248934d1d266 for this exact clip
    // (40 frames, gop=16, enable_alt_ref=false, A4 descent pinned off).
    // The pre-refactor capture was taken by stashing the GfGroup wiring
    // and re-running the identical encode — see the journal at
    // `wrk_journals/2026.05.27 - JRN - GOP topology landed.md`.
    const GOLDEN_HASH: u64 = 0xc077_2489_34d1_d266;
    const GOLDEN_LEN: usize = 26015;

    let stream = encode_fixed_clip();
    let hash = fnv1a(&stream);
    eprintln!(
        "GOP byte-identity: len = {}, fnv1a = {:#018x}",
        stream.len(),
        hash
    );
    assert_eq!(
        stream.len(),
        GOLDEN_LEN,
        "encoded stream length changed — the refactor altered the bitstream",
    );
    assert_eq!(
        hash, GOLDEN_HASH,
        "encoded stream bytes changed — the GfGroup refactor is NOT byte-identical. \
         The schedule must reproduce the exact pre-refactor counter decisions. \
         Do NOT update the golden to match a changed stream.",
    );
}
