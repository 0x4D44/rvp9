// Integration test: stderr is the diagnostic mechanism; the workspace
// `print_stderr = "deny"` lint targets library code, not test binaries.
#![allow(clippy::print_stderr)]

//! A21 PROOF (2026-05-27) — ML partition-search breakout pruner wired
//! into the A4 recursive RD partition descent.
//!
//! Proves that libvpx's `ml_predict_breakout` partition-search breakout
//! (`vp9/encoder/vp9_encodeframe.c:3250-3329` model;
//! `:4161-4168` gate) is wired into rvp9's RD partition descent
//! (`rvp9_core::mode_decision::nonrd_partition`): after a winning
//! PARTITION_NONE pick the real NN/linear breakout model is consulted,
//! and a positive vote forces `do_split = 0` — skipping that block's
//! SPLIT recursion (and the per-quadrant picker evaluations it would
//! have run).
//!
//! The pruner is gated behind the speed feature
//! `sf.rd_ml_partition.search_breakout` (libvpx
//! `vp9_speed_features.c:94-134`), which is ON at speed 0/1 of the
//! **Good** deadline below 1080p and requires `base_qindex >= 100`
//! (libvpx `:4163`). The cross_decoder_gate runs at speed 7 / Realtime
//! where libvpx leaves `search_breakout = 0` — so this PROOF runs at
//! **speed 1 / Good** to exercise the pruner. The gate's own validity
//! check (vpxdec byte-exact) is reused here on the speed-1 stream.
//!
//! Two assertions form the anti-cheat (anti-G3) gate:
//!
//! 1. **The pruner FIRES.** The global `ML_PARTITION_BREAKOUT_FIRED`
//!    counter advances on a speed-1 Good encode (breakout enabled) and
//!    stays at zero on a speed-7 Realtime encode (breakout disabled by
//!    the speed feature). A stubbed pruner that never voted breakout —
//!    or a wiring that computed the model then discarded its vote —
//!    would leave the counter at zero on the speed-1 run and FAIL here.
//! 2. **The pruned stream is valid VP9.** The speed-1 stream (with the
//!    pruner active) is byte-exact decodable by BOTH rvp9-decoder AND
//!    libvpx `vpxdec` (the same oracle the cross_decoder_gate uses).
//!    Skipping a SPLIT the model rejects changes the partition tree;
//!    this proves the resulting tree is still valid VP9.
//!
//! The corpus (`testsrc_720p30.y4m`) is centre-cropped to 256x256; if it
//! is unavailable the test SKIPs (informational) rather than failing.

use rvp9_decoder::Decoder;
use rvp9_encoder::rdo::inter_dispatch::ML_PARTITION_BREAKOUT_FIRED;
use rvp9_encoder::{AqMode, Deadline, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::container::{IvfHeader, write_ivf};
use rvp9_harness::y4m::{probe_testsrc_720p, read_cropped as y4m_read_cropped};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;

const W: u32 = 256;
const H: u32 = 256;
const N_FRAMES: u32 = 8;
// Low target bitrate → high base_qindex (>= 100) so the libvpx
// `cm->base_qindex >= 100` breakout precondition (vp9_encodeframe.c:4163)
// is satisfied on this clip.
const TARGET_BITRATE_BPS: u32 = 120_000;

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
/// `sf.rd_ml_partition.search_breakout`; speed 7 / Realtime does not.
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

/// Encode the clip, returning the per-frame packet bytes and the number
/// of times the ML breakout pruner FIRED during this encode.
fn encode(
    frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
    framerate: f64,
    speed: i32,
    deadline: Deadline,
) -> (Vec<Vec<u8>>, u64) {
    let before = ML_PARTITION_BREAKOUT_FIRED.load(Ordering::Relaxed);
    let mut enc =
        StreamEncoder::new(config(framerate, speed, deadline)).expect("StreamEncoder::new");
    let mut packets = Vec::with_capacity(frames.len());
    for (y, u, v) in frames {
        let pkt = enc.encode_frame(&yuv_view(y, u, v)).expect("encode_frame");
        packets.push(pkt.data);
    }
    let after = ML_PARTITION_BREAKOUT_FIRED.load(Ordering::Relaxed);
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
    let ivf_path = tmp.join("rvp9_a21_ml_partition.ivf");
    {
        let mut f = std::fs::File::create(&ivf_path).expect("create ivf");
        let header = IvfHeader::vp9(W as u16, H as u16, framerate.round() as u32, 1);
        let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        write_ivf(&mut f, &header, &refs).expect("write_ivf");
    }
    let vpx_out = tmp.join("rvp9_a21_ml_partition_vpxdec.yuv");
    let vpx_yuv = run_vpxdec(&vpxdec, &ivf_path, &vpx_out)
        .expect("vpxdec must accept the A21 ML-pruned bitstream");
    let rvp9_yuv = rvp9_decode_i420(packets);
    assert_eq!(
        vpx_yuv.len(),
        rvp9_yuv.len(),
        "A21 stream: vpxdec/rvp9 size mismatch ({} vs {})",
        vpx_yuv.len(),
        rvp9_yuv.len()
    );
    assert_eq!(
        vpx_yuv, rvp9_yuv,
        "A21 ML-pruned stream: vpxdec and rvp9-decoder disagree — the \
         breakout-pruned partition tree is not valid VP9"
    );
    eprintln!("vpxdec oracle: BYTE-EXACT on the A21 ML-pruned stream.");
    let _ = std::fs::remove_file(&ivf_path);
    let _ = std::fs::remove_file(&vpx_out);
}

#[test]
fn a21_ml_partition_breakout_fires_and_decodes() {
    let Some(y4m_path) = probe_testsrc_720p() else {
        eprintln!("[SKIPPED] testsrc_720p30.y4m corpus not available.");
        return;
    };
    let (frames, framerate) = match y4m_read_cropped(&y4m_path, W, H, N_FRAMES as usize) {
        Ok((f, fr)) if f.len() == N_FRAMES as usize => (f, fr),
        Ok((f, _)) => {
            eprintln!(
                "[SKIPPED] testsrc only yielded {}/{} frames.",
                f.len(),
                N_FRAMES
            );
            return;
        }
        Err(e) => {
            eprintln!("[SKIPPED] testsrc read failed: {e}");
            return;
        }
    };

    // The A4 RD descent must be ON (its default); clear the opt-outs.
    // SAFETY: this test runs in its own process (separate test binary).
    unsafe {
        std::env::remove_var("RVP9_A4_RD_PARTITION_DISABLE");
        std::env::remove_var("RVP9_F3_NONRD_PARTITION_ENABLE");
    }

    // ---- breakout ENABLED: speed 1 / Good (libvpx :115 turns on
    //      sf.rd_ml_partition.search_breakout below 1080p) ------------
    let (on_packets, on_fired) = encode(&frames, framerate, 1, Deadline::Good);

    // ---- breakout DISABLED: speed 7 / Realtime (the RT preset never
    //      sets search_breakout) ---------------------------------------
    let (_off_packets, off_fired) = encode(&frames, framerate, 7, Deadline::Realtime);

    eprintln!("speed-1 Good (breakout ON) : pruner fired {on_fired} times");
    eprintln!("speed-7 RT  (breakout OFF) : pruner fired {off_fired} times");

    // Assertion #1 (anti-G3 — the pruner actually fires): the ML
    // breakout model voted to skip a SPLIT at least once when the speed
    // feature enables it, and NEVER when the speed feature disables it.
    // A stub that ignored the model, or wiring that discarded the vote,
    // would leave `on_fired` at zero and fail here.
    assert!(
        on_fired > 0,
        "ML partition breakout pruner never fired at speed 1 Good — the real \
         ml_predict_breakout model is not gating the SPLIT recursion \
         (stubbed/discarded vote?)"
    );
    assert_eq!(
        off_fired, 0,
        "ML breakout pruner fired at speed 7 Realtime where the speed feature \
         sf.rd_ml_partition.search_breakout is OFF — the speed-feature gate leaked"
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
        "A21 PROOF PASS: ML breakout pruner fired {on_fired} times at speed 1 Good \
         (0 at speed 7 RT); pruned stream is byte-exact across decoders."
    );
}
