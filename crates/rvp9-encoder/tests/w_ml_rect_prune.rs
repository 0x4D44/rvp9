// Integration test: stderr is the diagnostic mechanism; the workspace
// `print_stderr = "deny"` lint targets library code, not test binaries.
#![allow(clippy::print_stderr)]

//! A21-rect PROOF (2026-05-27) — ML rectangular-partition pruner wired
//! into the A4 recursive RD partition descent.
//!
//! Proves that libvpx's `ml_prune_rect_partition` rect-partition model
//! (`vp9/encoder/vp9_encodeframe.c:3334-3425` model; gate at `:4332-4339`)
//! is wired into rvp9's RD partition descent
//! (`rvp9_core::mode_decision::nonrd_partition`): after the descent has
//! evaluated PARTITION_NONE and PARTITION_SPLIT at a 16/32/64 block, the
//! real NN model is consulted with the block's DC quantiser + RD
//! features, and a negative vote on an arm forces `partition_horz_allowed`
//! / `partition_vert_allowed` to 0 — skipping that arm's picker
//! invocations (the rect HORZ/VERT evaluations A4-rect added).
//!
//! The pruner is gated behind the speed feature
//! `sf.rd_ml_partition.prune_rect_thresh` (libvpx
//! `vp9_speed_features.c:253-256, 274-276, 340-342`), which is ON (≥ 0)
//! at speed 0/1 of the **Good** deadline and OFF (all `-1`) at speed 2+
//! and across the **Realtime** preset (`:1012-1015`). The
//! cross_decoder_gate runs at speed 7 / Realtime where the thresholds are
//! all `-1` — so this PROOF runs at **speed 1 / Good** to exercise the
//! pruner. The gate's own validity check (vpxdec byte-exact) is reused
//! here on the speed-1 stream.
//!
//! Two assertions form the anti-cheat (anti-G3) gate:
//!
//! 1. **The pruner FIRES.** The global `ML_PARTITION_RECT_PRUNED` counter
//!    (one bump per arm the model removes) advances on a speed-1 Good
//!    encode (rect-pruning enabled) and stays at zero on a speed-7
//!    Realtime encode (all `prune_rect_thresh` entries `-1`). A stubbed
//!    pruner that never removed an arm — or wiring that computed the
//!    model then discarded its verdict — would leave the counter at zero
//!    on the speed-1 run and FAIL here.
//! 2. **The pruned stream is valid VP9.** The speed-1 stream (with the
//!    pruner active) is byte-exact decodable by BOTH rvp9-decoder AND
//!    libvpx `vpxdec` (the same oracle the cross_decoder_gate uses).
//!    Skipping a rect arm the model rejects changes which partitions are
//!    evaluated (and can change the chosen partition); this proves the
//!    resulting tree is still valid VP9.
//!
//! The input is a deterministic moving textured clip, keeping the proof
//! independent of optional local corpus fixtures.

use rvp9_decoder::Decoder;
use rvp9_encoder::rdo::inter_dispatch::ML_PARTITION_RECT_PRUNED;
use rvp9_encoder::{AqMode, Deadline, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::container::{IvfHeader, write_ivf};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;

const W: u32 = 256;
const H: u32 = 256;
const N_FRAMES: u32 = 8;
// Low target bitrate → high base_qindex so the rect-prune feature set
// (which keys on dc_q) sees a non-trivial quantiser, matching the
// breakout PROOF's corpus conditions.
const TARGET_BITRATE_BPS: u32 = 120_000;
type OwnedYuvFrame = (Vec<u8>, Vec<u8>, Vec<u8>);
type OwnedYuvClip = Vec<OwnedYuvFrame>;

fn yuv_view<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> YuvFrameRef<'a> {
    YuvFrameRef {
        y,
        u,
        v,
        y_stride: W as usize,
        u_stride: (W / 2) as usize,
        v_stride: (W / 2) as usize,
    }
}

/// Build a config at a given `(speed, deadline)`. Speed 1 / Good enables
/// `sf.rd_ml_partition.prune_rect_thresh = [-1, 225, 225, 225]`; speed 7
/// / Realtime leaves it all `-1` (off).
fn config(framerate: f64, speed: i32, deadline: Deadline) -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: TARGET_BITRATE_BPS,
        gop_length: N_FRAMES + 1,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed,
        deadline,
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

fn synthetic_motion_frames() -> (OwnedYuvClip, f64) {
    let w = W as usize;
    let h = H as usize;
    let uv_w = w / 2;
    let uv_h = h / 2;
    let mut frames = Vec::with_capacity(N_FRAMES as usize);

    for frame_idx in 0..N_FRAMES as usize {
        let x_shift = frame_idx * 5;
        let y_shift = frame_idx * 3;
        let mut y = vec![0u8; w * h];
        for row in 0..h {
            for col in 0..w {
                let sx = (col + x_shift) & 255;
                let sy = (row + y_shift) & 255;
                let v = sx
                    .wrapping_mul(37)
                    .wrapping_add(sy.wrapping_mul(19))
                    .wrapping_add((sx ^ sy).wrapping_mul(11))
                    .wrapping_add(((sx.wrapping_mul(sy)) >> 3) & 255);
                y[row * w + col] = v as u8;
            }
        }

        frames.push((y, vec![128u8; uv_w * uv_h], vec![128u8; uv_w * uv_h]));
    }

    (frames, 30.0)
}

/// Encode the clip, returning the per-frame packet bytes and the number
/// of rect arms the ML rect-prune model REMOVED during this encode.
fn encode(
    frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
    framerate: f64,
    speed: i32,
    deadline: Deadline,
) -> (Vec<Vec<u8>>, u64) {
    let before = ML_PARTITION_RECT_PRUNED.load(Ordering::Relaxed);
    let mut enc =
        StreamEncoder::new(config(framerate, speed, deadline)).expect("StreamEncoder::new");
    let mut packets = Vec::with_capacity(frames.len());
    for (y, u, v) in frames {
        let pkt = enc.encode_frame(&yuv_view(y, u, v)).expect("encode_frame");
        packets.push(pkt.data);
    }
    let after = ML_PARTITION_RECT_PRUNED.load(Ordering::Relaxed);
    (packets, after - before)
}

/// Decode every packet through rvp9-decoder into concatenated I420 bytes.
fn rvp9_decode_i420(packets: &[Vec<u8>]) -> Vec<u8> {
    let mut dec = Decoder::new();
    let w = W as usize;
    let h = H as usize;
    let uvw = w / 2;
    let uvh = h / 2;
    let mut out = Vec::new();
    for pkt in packets {
        let parsed = dec.parse_packet(pkt).expect("parse_packet");
        let owned: Vec<rvp9_decoder::ParsedFrame> = parsed
            .frames()
            .iter()
            .map(|pf| rvp9_decoder::ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            })
            .collect();
        for pf in &owned {
            let buf = dec.decode_frame(pf).expect("rvp9 decode_frame");
            let yp = buf.y_plane();
            let up = buf.u_plane();
            let vp = buf.v_plane();
            for r in 0..h {
                let off = r * yp.stride();
                out.extend_from_slice(&yp.data()[off..off + w]);
            }
            for r in 0..uvh {
                let off = r * up.stride();
                out.extend_from_slice(&up.data()[off..off + uvw]);
            }
            for r in 0..uvh {
                let off = r * vp.stride();
                out.extend_from_slice(&vp.data()[off..off + uvw]);
            }
        }
    }
    out
}

/// Probe `vpxdec.exe` at the canonical sibling path, then `vpxdec` on PATH.
fn vpxdec_path() -> Option<PathBuf> {
    let canonical = PathBuf::from(r"D:\language\libvpx-stock\vpxdec.exe");
    if canonical.is_file() {
        return Some(canonical);
    }
    if Command::new("vpxdec").arg("--help").output().is_ok() {
        return Some(PathBuf::from("vpxdec"));
    }
    None
}

/// Run `vpxdec --i420` on an IVF and return the raw I420 bytes.
fn run_vpxdec(vpxdec: &Path, ivf: &Path, out: &Path) -> std::io::Result<Vec<u8>> {
    let status = Command::new(vpxdec)
        .args(["--codec=vp9", "--i420", "-o"])
        .arg(out)
        .arg(ivf)
        .output()?;
    if !status.status.success() {
        return Err(std::io::Error::other(format!(
            "vpxdec failed: {}",
            String::from_utf8_lossy(&status.stderr)
        )));
    }
    std::fs::read(out)
}

/// Oracle leg: the pruned (speed-1) stream must be byte-exact decodable
/// by vpxdec, cross-checked against rvp9-decoder. Informational SKIP when
/// vpxdec is unavailable — the rvp9-decoder leg already ran by then.
fn verify_through_vpxdec(packets: &[Vec<u8>], framerate: f64) {
    let Some(vpxdec) = vpxdec_path() else {
        eprintln!("[vpxdec oracle SKIPPED] vpxdec not found; rvp9-decoder leg already green.");
        return;
    };
    let tmp = std::env::temp_dir();
    let ivf_path = tmp.join("rvp9_a21_ml_rect_prune.ivf");
    {
        let mut f = std::fs::File::create(&ivf_path).expect("create ivf");
        let header = IvfHeader::vp9(W as u16, H as u16, framerate.round() as u32, 1);
        let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        write_ivf(&mut f, &header, &refs).expect("write_ivf");
    }
    let vpx_out = tmp.join("rvp9_a21_ml_rect_prune_vpxdec.yuv");
    let vpx_yuv = run_vpxdec(&vpxdec, &ivf_path, &vpx_out)
        .expect("vpxdec must accept the A21-rect ML-pruned bitstream");
    let rvp9_yuv = rvp9_decode_i420(packets);
    assert_eq!(
        vpx_yuv.len(),
        rvp9_yuv.len(),
        "A21-rect stream: vpxdec/rvp9 size mismatch ({} vs {})",
        vpx_yuv.len(),
        rvp9_yuv.len()
    );
    assert_eq!(
        vpx_yuv, rvp9_yuv,
        "A21-rect ML-pruned stream: vpxdec and rvp9-decoder disagree — the \
         rect-pruned partition tree is not valid VP9"
    );
    eprintln!("vpxdec oracle: BYTE-EXACT on the A21-rect ML-pruned stream.");
    let _ = std::fs::remove_file(&ivf_path);
    let _ = std::fs::remove_file(&vpx_out);
}

#[test]
fn a21_ml_rect_prune_fires_and_decodes() {
    let (frames, framerate) = synthetic_motion_frames();

    // The A4 RD descent (which evaluates the rect arms the pruner gates)
    // must be ON (its default); clear the opt-outs.
    // SAFETY: this test runs in its own process (separate test binary).
    unsafe {
        std::env::remove_var("RVP9_A4_RD_PARTITION_DISABLE");
        std::env::remove_var("RVP9_F3_NONRD_PARTITION_ENABLE");
    }

    // ---- rect-pruning ENABLED: speed 1 / Good (libvpx :274-276 sets
    //      prune_rect_thresh = [-1, 225, 225, 225]) ---------------------
    let (on_packets, on_pruned) = encode(&frames, framerate, 1, Deadline::Good);

    // ---- rect-pruning DISABLED: speed 7 / Realtime (the RT preset
    //      leaves prune_rect_thresh all -1, libvpx :1012-1015) ----------
    let (_off_packets, off_pruned) = encode(&frames, framerate, 7, Deadline::Realtime);

    eprintln!("speed-1 Good (rect-prune ON) : model removed {on_pruned} rect arms");
    eprintln!("speed-7 RT  (rect-prune OFF) : model removed {off_pruned} rect arms");

    // Assertion #1 (anti-G3 — the pruner actually fires): the ML
    // rect-prune model removed at least one HORZ/VERT arm when the speed
    // feature enables it, and NEVER when the thresholds are all -1.
    // A stub that ignored the model, or wiring that discarded the verdict,
    // would leave `on_pruned` at zero and fail here.
    assert!(
        on_pruned > 0,
        "ML rect-prune model never removed a rect arm at speed 1 Good — the real \
         ml_prune_rect_partition model is not gating the HORZ/VERT evaluation \
         (stubbed/discarded verdict?)"
    );
    assert_eq!(
        off_pruned, 0,
        "ML rect-prune model removed an arm at speed 7 Realtime where every \
         sf.rd_ml_partition.prune_rect_thresh entry is -1 — the speed-feature gate leaked"
    );

    // Assertion #2 (oracle — valid VP9): both decoders agree byte-exact
    // on the speed-1 (pruner-active) stream.
    let on_decoded_frames = {
        let bytes = rvp9_decode_i420(&on_packets);
        bytes.len() / ((W as usize) * (H as usize) * 3 / 2)
    };
    assert_eq!(
        on_decoded_frames, N_FRAMES as usize,
        "pruned stream did not decode to {N_FRAMES} frames (got {on_decoded_frames})"
    );
    verify_through_vpxdec(&on_packets, framerate);

    eprintln!(
        "A21-rect PROOF PASS: ML rect-prune model removed {on_pruned} arms at speed 1 Good \
         (0 at speed 7 RT); pruned stream is byte-exact across decoders."
    );
}
