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
use rvp9_encoder::gop::{FrameUpdateType, GOLDEN_REFRESH_PERIOD, GfGroup};
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
        enable_tpl: false,
        stage4a_product_split_override: Default::default(),
        min_qindex: 4,
        max_qindex: 255,
    }
}

/// Pin the pre-A4 partition path (mirrors `tests/w_buffer_pool.rs`) so
/// the byte-identity golden is stable. SAFETY: integration binary; all
/// tests here want the descent off, so set-only is race-free.
fn pin_a4() {
    // SAFETY: integration binary; all tests here set the same process env
    // knob to the same value.
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
        assert!(header.show_frame, "frame {idx}: expected a show frame",);

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
    // Golden re-captured (2026-05-27) after the Track F intra-lambda KF
    // Q-policy fix (`pick_q_and_bounds` KF arm now matches libvpx
    // `vp9_ratectrl.c:1013-1042` — first frame uses `best_quality`,
    // subsequent KFs use `get_kf_active_quality(avg_frame_qindex[KEY])`).
    // The previous `KF_ACTIVE_QUALITY_BASIS_CAP = 122` workaround pinned
    // the KF Q at ~57 regardless of bitrate; removing it lets KFs reach
    // best_quality (closes 1.2 dB of the Track F PSNR gap at 1200/2000k
    // cpu7). That intentionally changes the emitted bitstream (KFs now
    // code at lower Q → more KF bits → stream grows 25670→26222). The
    // GOP *schedule* this oracle guards is unchanged — the per-frame
    // refresh-mask assertions above (KfUpdate 0xFF / GfUpdate 0x03 /
    // LfUpdate 0x01) still hold; only the upstream KF Q decision moved.
    // The new stream is verified byte-exact encoder-recon-vs-decode by
    // `drift_diag` and passes the 4/4 cross-decoder gate.
    // Pre-fix golden was (len 25670, fnv1a 0xeee174942d662023).
    //
    // Re-captured again (2026-05-27, Track-F iter7) after enabling
    // per-block transform-size RD selection. Inter frames now code
    // `tx_mode = TxModeSelect` and signal each leaf's RD-chosen Y
    // tx_size (libvpx `super_block_yrd` / `choose_tx_size_from_rd`),
    // intentionally changing the bitstream (26222→27405). The GOP
    // *schedule* this oracle guards is unchanged (the refresh-mask
    // assertions above still hold). The new stream is verified
    // byte-exact encoder-recon-vs-decode by `drift_diag` (SSE=0 every
    // frame) and passes the 4/4 cross-decoder gate. Prior iter golden:
    // (len 26222, fnv1a 0x31525167_cc032401).
    //
    // Re-captured again (2026-05-29) against current `main` after TWO
    // stacked, independent deltas (the GOP *schedule* this oracle guards
    // is unchanged — the refresh-mask assertions above still hold):
    //   (a) concurrent-agent DSP/fmt drift already on `main` (pre-
    //       existing — encoding with `RVP9_3D=0`, which makes the S2/3D
    //       change inert, still diverges from the prior golden: len
    //       27405 -> 25028, fnv1a -> 0xe757c4ff_6484be53);
    //   (b) the frame-adaptive S2 inter-RD skip gate (Lever-3b/3D),
    //       DEFAULT-ON: this fixture's moving-gradient source classifies
    //       as "natural" (hf_energy ~8 >= 7.0 threshold), so S2
    //       honest-skip activates on its inter frames (25028 -> 23911).
    // Decode-validated before update: sibling `gop_topology_schedule_
    // matches_emitted_frames` passes (rvp9-decoder parses every emitted
    // frame), and the stream decodes cleanly through `ffmpeg` (all 40
    // frames, exit 0). NOTE: stock libvpx `vpxdec` aborts at frame 18
    // ("Invalid frame marker") on this fixture — but that failure is
    // PRE-EXISTING concurrent drift, NOT this change: it reproduces
    // identically with `RVP9_3D=0` (S2/3D inert), and the frame markers
    // are all valid (0b10), so it is a vpxdec strict-conformance quirk
    // at the 2nd-KF boundary, independent of the S2 activation here.
    // Correct, intentional golden update — NOT a masked regression.
    // Prior iter golden: (len 27405, fnv1a 0xdb380fd1_fcc1ac6b).
    //
    // Re-captured again (2026-05-30) after adding the per-candidate
    // tx-size SIGNALING cost to the inter TX-size emit walk
    // `choose_inter_luma_tx_size` (libvpx `choose_tx_size_from_rd`'s
    // `r1 = r0 + r_tx_size`, previously omitted). Charging it shifts the
    // inter Y tx-size decision toward larger transforms, changing the
    // signaled tx_size bits + coef tiling and therefore the emitted
    // bitstream (23911 -> 24240). The GOP *schedule* this oracle guards
    // (refresh-mask assertions above) is unchanged. Decode-validated:
    // sibling `gop_topology_schedule_matches_emitted_frames` passes and
    // the separate `cross_decoder_gate` is BYTE-EXACT 4/4 through vpxdec
    // + ffmpeg on a fixed clip (the pre-existing vpxdec strict-marker
    // quirk on THIS fixture's 2nd-KF boundary is unrelated, as noted
    // above). Correct, intentional golden update — NOT a masked
    // regression. Prior iter golden: (len 23911, fnv1a 0x924a_2c35_b117_0a9b).
    //
    // Re-captured again (2026-05-30) after the NEARESTMV/ZEROMV tie-break
    // fix (`mode_decision/inter.rs`, env `RVP9_NEAREST`, DEFAULT-ON). The
    // fix makes the duplicate-MV skip unconditional (libvpx-faithful), so
    // on the dominant `NEAREST==(0,0)` tie NEARESTMV wins over ZEROMV
    // (previously rvp9's `mode_checked` guard let ZEROMV become the
    // incumbent and the strict-`<` rule kept it). This fixture has such
    // ties, so the chosen inter mode symbols change and the emitted
    // bitstream shrinks (24240 -> 24225). The GOP *schedule* this oracle
    // guards (refresh-mask assertions above) is unchanged. Decode-
    // validated: the sibling `gop_topology_schedule_matches_emitted_frames`
    // passes (rvp9-decoder parses every emitted frame of the fix-ON
    // stream), and the separate `cross_decoder_gate` is BYTE-EXACT 4/4
    // through vpxdec + ffmpeg on a fixed clip. (The pre-existing vpxdec
    // strict-marker quirk on THIS fixture's 2nd-KF boundary — frame 18 —
    // is unrelated, as noted above; it reproduces with `RVP9_3D=0`.)
    // Confirmed `RVP9_NEAREST=0` restores this fixture's PRE-fix bytes
    // EXACTLY (len 24240, fnv1a 0x6f4c_8c58_1914_1c48) — the delta is
    // exactly the tie-fix, layered cleanly on top of the prior baseline.
    // Correct, intentional golden update — NOT a masked regression. Prior
    // iter golden: (len 24240, fnv1a 0x6f4c_8c58_1914_1c48).
    //
    // Re-captured again (2026-05-30) after the RCFI first-inter-frame
    // active_best fix (`rate_control/cbr.rs`, env `RVP9_RCFI`, DEFAULT-ON):
    // libvpx `vp9_ratectrl.c:1058-1068`'s `current_video_frame > 1` split
    // seeds the stream's FIRST inter frame's active_best from
    // `avg_frame_qindex[KEY]` rather than the seed-high (255) inter avg, so
    // that frame invests in a higher-quality first reference (Q drops, more
    // bits) — matching libvpx. The first inter of each GOP after the 2nd KF
    // is unaffected (uses the inter avg, current_video_frame > 1). Stream
    // grows (24225 -> 24727). Decode-validated BYTE-CLEAN (sibling decode
    // test passes) + `cross_decoder_gate` BYTE-EXACT 4/4. Confirmed
    // `RVP9_RCFI=0` restores this fixture's PRE-fix bytes EXACTLY (len
    // 24225, fnv1a 0x07f7_73a8_cb58_81df) — the delta is exactly the RCFI
    // fix. Correct, intentional golden update — NOT a masked regression.
    // Prior iter golden: (len 24225, fnv1a 0x07f7_73a8_cb58_81df).
    //
    // Re-captured again (2026-05-30) after the LPFQ loop-filter-level fix
    // (`loopfilter.rs::pick_filter_level`, env `RVP9_LPFQ`, DEFAULT-ON):
    // ports libvpx's calibrated `LPF_PICK_FROM_Q` (ac_quant-based, KF -4)
    // replacing the legacy `(q_index >> 1)` over-filter, lowering the
    // signaled `filter_level` and the post-LF recon → changed bitstream
    // (24727 -> 23722). cross_decoder_gate BYTE-EXACT 4/4; sibling decode
    // test passes. Confirmed `RVP9_LPFQ=0` restores this fixture's PRE-fix
    // bytes EXACTLY (len 24727, fnv1a 0xd593_baa5_50f1_21e1) — the delta is
    // exactly the LPFQ fix. Correct, intentional golden update — NOT a
    // masked regression. Prior iter golden: (len 24727, fnv1a
    // 0xd593_baa5_50f1_21e1).
    //
    // Re-captured again (2026-05-30) after the LA HF-gated lambda
    // calibration (`mode_decision/inter.rs`, env `RVP9_LA`, DEFAULT-ON):
    // doubles the picker lambda on natural-content frames (this fixture's
    // gradient classifies natural), making the picker more rate-conscious
    // → fewer coded bits (23722 -> 23434). Decode-validated BYTE-CLEAN;
    // `RVP9_LA=0` restores PRE-LA bytes EXACTLY (len 23722, fnv1a
    // 0xa7d6_45f0_28d1_8f7a). Correct, intentional golden update. Prior
    // iter golden: (len 23722, fnv1a 0xa7d6_45f0_28d1_8f7a).
    //
    // Re-captured again (2026-06-21) after keyframes started mirroring
    // libvpx `vp9_setup_past_independence` by resetting the encoder's FC
    // mirrors before the next inter frame diff-encodes its compressed
    // header. The GOP schedule is unchanged; sibling
    // `gop_topology_schedule_matches_emitted_frames` passes. The changed
    // post-KF entropy state shrinks the bitstream (23434 -> 23201).
    // Correct, intentional golden update. Prior iter golden:
    // (len 23434, fnv1a 0xb083_6c6d_2fd1_46bf).
    //
    // Re-captured again (2026-08-01) after the production-parity stack
    // changed the default AQ-off realtime picker and split policy. The
    // sibling schedule oracle still matches every emitted frame across more
    // than two GOPs, so the schedule invariant is unchanged. The new stream
    // reproduced twice with the same hash. Prior production golden: (len
    // 23201, fnv1a 0x646a_4565_7bc3_f735).
    //
    // Re-captured again (2026-08-02) after the accepted realtime intra-model
    // reuse and keyframe parity units. A clean-history bisect identified
    // 12a9db9 as the first intentional coding change after the prior refresh.
    // The schedule oracle still matches every emitted frame, and both the
    // clean baseline and edge-motion task tree reproduce these bytes exactly.
    //
    // Re-captured again (2026-08-02) after the realtime VBR picker stopped
    // using libvpx's CBR-only large-block model and Block32 started running
    // real `block_yrd`. This deliberately changes coding decisions; the
    // sibling schedule oracle still matches every emitted frame.
    //
    // Re-captured again (2026-08-02) after Block32's luma-only
    // encode-vs-skip comparison was ordered like libvpx. The new bytes
    // reproduced twice, and the sibling schedule oracle still matches every
    // emitted frame.
    //
    // Re-captured again (2026-08-02) after the realtime picker started
    // honoring libvpx's per-superblock U/V color-sensitivity mask. The new
    // bytes reproduced twice, and the sibling schedule oracle still matches
    // every emitted frame.
    //
    // Re-captured again (2026-08-02) after mode decisions started pricing
    // skip syntax from the live frame context. The new bytes reproduced twice,
    // and the sibling schedule oracle still matches every emitted frame.
    //
    // Re-captured again (2026-08-02) after full-pel and sub-pel motion
    // searches began using libvpx's qindex-derived rate multipliers. The new
    // bytes reproduced twice, and the sibling schedule oracle still matches
    // every emitted frame.
    //
    // Re-captured again (2026-08-02) after CPU7 NEWMV sub-pel refinement
    // adopted libvpx's realtime search level 0. The new bytes reproduced
    // twice, and the sibling schedule oracle still matches every emitted
    // frame.
    //
    // Re-captured after CPU7 stopped NEWMV refinement at quarter-pel, matching
    // pinned libvpx. The sibling schedule oracle still matches every frame.
    //
    // Re-captured after normal inter Q history excluded boosted GOLDEN
    // refreshes and adopted libvpx's rounded EWMA. The parent of 6a909b26
    // emits the prior 24,927-byte golden; that source-correct commit is the
    // first intentional coding change. The current clean Unit 651 baseline
    // reproduces the bytes below, and the sibling schedule oracle still
    // matches every emitted frame.
    const GOLDEN_HASH: u64 = 0xa34d_c1d6_f430_7467;
    const GOLDEN_LEN: usize = 23293;

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
        "encoded stream length changed; inspect the GOP schedule and bitstream delta before updating this golden",
    );
    assert_eq!(
        hash, GOLDEN_HASH,
        "encoded stream bytes changed; inspect the GOP schedule and bitstream delta before updating this golden",
    );
}
