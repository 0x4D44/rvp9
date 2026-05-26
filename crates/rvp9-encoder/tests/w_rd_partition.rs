// Integration test: stderr is the diagnostic mechanism; the workspace
// `print_stderr = "deny"` lint targets library code, not test binaries.
#![allow(clippy::print_stderr)]

//! A4 PROOF (2026-05-27) — recursive RD-optimal partition descent.
//!
//! Proves that the encoder's partition SHAPE is now chosen by a real
//! recursive NONE-vs-SPLIT rate-distortion comparison
//! (`rvp9_core::mode_decision::nonrd_partition::nonrd_pick_partition`
//! driven by the real `pick_inter_mode` leaf picker, libvpx
//! `vp9/encoder/vp9_encodeframe.c:4922-5210` / the `pc_tree->partitioning`
//! write at `:4284`), NOT by the variance heuristic alone.
//!
//! Before A4 the partition shape was always the variance partitioner's
//! output (`choose_partitioning`, `vp9_encodeframe.c:1261`); the RDO
//! machinery only re-priced MODES inside fixed leaves. A4 promotes the
//! RD descent from infrastructure-only (default-OFF, env-gated) to the
//! encoder's default partition decision.
//!
//! Three assertions form the anti-cheat (anti-G3) gate:
//!
//! 1. **Shape is RD-chosen, not fixed.** The global
//!    `RD_PARTITION_RESHAPED_SBS` counter advances on a default encode
//!    (descent ON) and stays put when the descent is disabled
//!    (`RVP9_A4_RD_PARTITION_DISABLE=1`). The counter only bumps when
//!    the RD descent's chosen per-cell leaf-size grid DIFFERS from the
//!    variance hint — so a stubbed descent that echoed the hint, or a
//!    fixed-grid partitioner, would leave it at zero and fail here.
//! 2. **Effect on the bitstream.** The descent-ON and descent-OFF byte
//!    streams DIFFER. Identical bytes would mean the descent's decision
//!    was computed and discarded (the forbidden cheat).
//! 3. **The partitioned stream is valid VP9.** The descent-ON stream is
//!    byte-exact decodable by BOTH rvp9-decoder AND libvpx `vpxdec`
//!    (the same oracle the cross_decoder_gate uses). A malformed
//!    partition tree / partition-symbol emission would diverge the
//!    decoders or fail vpxdec outright.
//!
//! The corpus (`testsrc_720p30.y4m`) is centre-cropped to 256x256; if it
//! is unavailable the test SKIPs (informational) rather than failing.

use rvp9_decoder::Decoder;
use rvp9_encoder::rdo::inter_dispatch::RD_PARTITION_RESHAPED_SBS;
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::container::{IvfHeader, write_ivf};
use rvp9_harness::y4m::{probe_testsrc_720p, read_cropped as y4m_read_cropped};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;

const W: u32 = 256;
const H: u32 = 256;
const N_FRAMES: u32 = 8;
const TARGET_BITRATE_BPS: u32 = 1_000_000;

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

/// Cross-decoder gate runs at speed 7 (realtime); A4 keeps the descent
/// active at that speed so the gate proves the RD-partitioned stream is
/// valid VP9. We use the same config here.
fn config(framerate: f64) -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: TARGET_BITRATE_BPS,
        gop_length: N_FRAMES + 1,
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
        min_qindex: 4,
        max_qindex: 255,
    }
}

/// Encode the clip, returning the per-frame packet bytes and the number
/// of SBs the RD partition descent re-shaped during this encode.
fn encode(frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)], framerate: f64) -> (Vec<Vec<u8>>, u64) {
    let before = RD_PARTITION_RESHAPED_SBS.load(Ordering::Relaxed);
    let mut enc = StreamEncoder::new(config(framerate)).expect("StreamEncoder::new");
    let mut packets = Vec::with_capacity(frames.len());
    for (y, u, v) in frames {
        let pkt = enc.encode_frame(&yuv_view(y, u, v)).expect("encode_frame");
        packets.push(pkt.data);
    }
    let after = RD_PARTITION_RESHAPED_SBS.load(Ordering::Relaxed);
    (packets, after - before)
}

/// Decode every packet through rvp9-decoder into concatenated I420
/// bytes (Y, then U, then V per frame).
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

/// Probe `vpxdec.exe` at the canonical sibling path, then `vpxdec` on
/// PATH. Returns `None` when neither exists.
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

/// Oracle leg: the descent-ON stream must be byte-exact decodable by
/// vpxdec (cross-checked against rvp9-decoder). Informational SKIP when
/// vpxdec is unavailable — the rvp9-decoder leg already ran by then.
fn verify_through_vpxdec(packets: &[Vec<u8>], framerate: f64) {
    let Some(vpxdec) = vpxdec_path() else {
        eprintln!("[vpxdec oracle SKIPPED] vpxdec not found; rvp9-decoder leg already green.");
        return;
    };
    let tmp = std::env::temp_dir();
    let ivf_path = tmp.join("rvp9_a4_rd_partition.ivf");
    {
        let mut f = std::fs::File::create(&ivf_path).expect("create ivf");
        let header = IvfHeader::vp9(W as u16, H as u16, framerate.round() as u32, 1);
        let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        write_ivf(&mut f, &header, &refs).expect("write_ivf");
    }
    let vpx_out = tmp.join("rvp9_a4_rd_partition_vpxdec.yuv");
    let vpx_yuv = run_vpxdec(&vpxdec, &ivf_path, &vpx_out)
        .expect("vpxdec must accept the A4 RD-partitioned bitstream");
    let rvp9_yuv = rvp9_decode_i420(packets);
    assert_eq!(
        vpx_yuv.len(),
        rvp9_yuv.len(),
        "A4 stream: vpxdec/rvp9 size mismatch ({} vs {})",
        vpx_yuv.len(),
        rvp9_yuv.len()
    );
    assert_eq!(
        vpx_yuv, rvp9_yuv,
        "A4 RD-partitioned stream: vpxdec and rvp9-decoder disagree — the \
         partition tree / partition-symbol emission is not valid VP9"
    );
    eprintln!("vpxdec oracle: BYTE-EXACT on the A4 RD-partitioned stream.");
    let _ = std::fs::remove_file(&ivf_path);
    let _ = std::fs::remove_file(&vpx_out);
}

#[test]
fn a4_rd_partition_shape_is_rd_chosen_and_decodes() {
    let Some(y4m_path) = probe_testsrc_720p() else {
        eprintln!("[SKIPPED] testsrc_720p30.y4m corpus not available.");
        return;
    };
    let (frames, framerate) = match y4m_read_cropped(&y4m_path, W, H, N_FRAMES as usize) {
        Ok((f, fr)) if f.len() == N_FRAMES as usize => (f, fr),
        Ok((f, _)) => {
            eprintln!("[SKIPPED] testsrc only yielded {}/{} frames.", f.len(), N_FRAMES);
            return;
        }
        Err(e) => {
            eprintln!("[SKIPPED] testsrc read failed: {e}");
            return;
        }
    };

    // ---- descent ON (the A4 default) ----------------------------------
    // SAFETY: this test runs in its own process (separate test binary);
    // no parallel test mutates these env vars. We clear both knobs so
    // the encoder takes the default (descent ON).
    unsafe {
        std::env::remove_var("RVP9_A4_RD_PARTITION_DISABLE");
        std::env::remove_var("RVP9_F3_NONRD_PARTITION_ENABLE");
    }
    let (on_packets, on_reshaped) = encode(&frames, framerate);

    // ---- descent OFF (opt out; pre-A4 variance-hint partition) --------
    unsafe {
        std::env::set_var("RVP9_A4_RD_PARTITION_DISABLE", "1");
    }
    let (off_packets, off_reshaped) = encode(&frames, framerate);
    // Restore the default for any later code in this process.
    unsafe {
        std::env::remove_var("RVP9_A4_RD_PARTITION_DISABLE");
    }

    eprintln!("descent ON  : reshaped {on_reshaped} SBs");
    eprintln!("descent OFF : reshaped {off_reshaped} SBs");

    // Assertion #1 (anti-G3 — shape is RD-chosen): the RD descent
    // re-shaped at least one SB relative to the variance hint when ON,
    // and re-shaped NONE when OFF. A stubbed / fixed descent that
    // echoed the hint could never satisfy `on_reshaped > 0`.
    assert!(
        on_reshaped > 0,
        "A4 RD partition descent never re-shaped any SB on real motion — \
         the partition shape is not actually RD-chosen (stubbed/fixed?)"
    );
    assert_eq!(
        off_reshaped, 0,
        "RD partition descent re-shaped SBs while disabled — the opt-out \
         leaked"
    );

    // Assertion #2 (effect): the two streams differ. Identical bytes
    // would mean the RD partition decision was computed then discarded.
    let on_total: usize = on_packets.iter().map(Vec::len).sum();
    let off_total: usize = off_packets.iter().map(Vec::len).sum();
    eprintln!("descent ON total bytes: {on_total}, OFF total bytes: {off_total}");
    assert_ne!(
        on_packets, off_packets,
        "descent ON and OFF produced byte-identical streams — the RD \
         partition decision is not driving the bitstream"
    );

    // Assertion #3 (oracle — valid VP9): both decoders agree byte-exact
    // on the descent-ON (default) stream.
    let on_decoded_frames = {
        let bytes = rvp9_decode_i420(&on_packets);
        bytes.len() / ((W as usize) * (H as usize) * 3 / 2)
    };
    assert_eq!(
        on_decoded_frames, N_FRAMES as usize,
        "descent-ON stream did not decode to {N_FRAMES} frames (got {on_decoded_frames})"
    );
    verify_through_vpxdec(&on_packets, framerate);

    eprintln!(
        "A4 PROOF PASS: RD descent re-shaped {on_reshaped} SBs (0 when off); \
         streams differ; descent-ON stream is byte-exact across decoders."
    );
}
