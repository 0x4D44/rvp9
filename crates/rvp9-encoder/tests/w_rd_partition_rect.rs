// Integration test: stderr is the diagnostic mechanism; the workspace
// `print_stderr = "deny"` lint targets library code, not test binaries.
#![allow(clippy::print_stderr)]

//! A4-rect PROOF (2026-05-27) — rectangular RD partitions
//! (`PARTITION_HORZ` / `PARTITION_VERT`).
//!
//! Extends the A4 square-only NONE/SPLIT descent
//! (`rvp9_core::mode_decision::nonrd_partition::nonrd_pick_partition`)
//! to additionally evaluate PARTITION_HORZ (two `w × h/2` blocks
//! stacked) and PARTITION_VERT (two `w/2 × h` blocks side by side),
//! coding each rect as two sub-blocks at the rect size and ranking the
//! summed RD against NONE and SPLIT. Mirrors libvpx `rd_pick_partition`
//! (`vp9/encoder/vp9_encodeframe.c:4342-4436`).
//!
//! Two assertions form the anti-cheat (anti-G3) gate:
//!
//! 1. **Rect is genuinely evaluated AND wins ≥ 1 block.** The global
//!    `RD_PARTITION_RECT_LEAVES` counter (bumped once per rectangular
//!    leaf the descent commits) advances on content engineered to
//!    favour a horizontal split — a translating horizontal-edge clip
//!    where the top band and bottom band move by DIFFERENT motion
//!    vectors, so a single MV (NONE) fits poorly, a 4-way SPLIT pays an
//!    extra partition bit for no gain, but a HORZ split lets each band
//!    take its own MV. A descent that never evaluated rect, or
//!    evaluated it but always discarded it for square, would leave this
//!    counter at zero and FAIL here.
//! 2. **The rect-containing stream is valid VP9.** The descent-ON stream
//!    is byte-exact decodable by BOTH rvp9-decoder AND libvpx `vpxdec`
//!    (the cross_decoder_gate oracle). A malformed rect partition-symbol
//!    emission or rect sub-block coding would diverge the two decoders
//!    or fail vpxdec outright — `infer_inter_partition` must map the
//!    rect leaf sizes to PARTITION_HORZ/VERT and emit the two halves in
//!    the layout the decoder's partition reader expects.
//!
//! The corpus is SYNTHETIC (constructed here), so this test never SKIPs
//! for a missing corpus; the vpxdec oracle leg SKIPs (informational)
//! only when vpxdec is unavailable, after the rvp9-decoder leg is green.

use rvp9_decoder::Decoder;
use rvp9_encoder::rdo::inter_dispatch::RD_PARTITION_RECT_LEAVES;
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::container::{IvfHeader, write_ivf};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;

const W: u32 = 128;
const H: u32 = 128;
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
        temporal_layers: 1,
        enable_tpl: false,
        stage4a_product_split_override: Default::default(),
        min_qindex: 4,
        max_qindex: 255,
    }
}

/// Build a synthetic frame whose TOP half scrolls left by `top_shift`
/// pixels and BOTTOM half scrolls left by `bottom_shift` pixels, over a
/// vertically-textured background (a column-index gradient). Because the
/// two bands move by different amounts, a single 64×64 MV (PARTITION_NONE)
/// fits one band but not the other, while a horizontal split (HORZ) lets
/// each band take its own MV — the RD descent should prefer HORZ on the
/// SB straddling the band boundary.
fn make_frame(top_shift: i32, bottom_shift: i32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = W as usize;
    let h = H as usize;
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        let shift = if row < h / 2 { top_shift } else { bottom_shift };
        for col in 0..w {
            // Strong vertical texture so motion is detectable: a sawtooth
            // on the (shifted) column index. Distinct per-band phase via
            // a row-band offset keeps the two bands visually separable.
            let band = if row < h / 2 { 0i32 } else { 96i32 };
            let src_col = (col as i32 - shift).rem_euclid(w as i32);
            let v = ((src_col * 5) % 200) as u8;
            y[row * w + col] = v.wrapping_add(band as u8).wrapping_add((row as u8) >> 3);
        }
    }
    // Flat chroma — the partition decision is luma-driven.
    let uvw = w / 2;
    let uvh = h / 2;
    let u = vec![128u8; uvw * uvh];
    let v = vec![128u8; uvw * uvh];
    (y, u, v)
}

fn synthetic_clip() -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    // Top band drifts left fast, bottom band drifts left slow — diverging
    // motion across the horizontal mid-line.
    (0..N_FRAMES)
        .map(|f| make_frame(2 * f as i32, -(f as i32)))
        .collect()
}

fn encode(frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)], framerate: f64) -> (Vec<Vec<u8>>, u64) {
    let before = RD_PARTITION_RECT_LEAVES.load(Ordering::Relaxed);
    let mut enc = StreamEncoder::new(config(framerate)).expect("StreamEncoder::new");
    let mut packets = Vec::with_capacity(frames.len());
    for (y, u, v) in frames {
        let pkt = enc.encode_frame(&yuv_view(y, u, v)).expect("encode_frame");
        packets.push(pkt.data);
    }
    let after = RD_PARTITION_RECT_LEAVES.load(Ordering::Relaxed);
    (packets, after - before)
}

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

fn verify_through_vpxdec(packets: &[Vec<u8>], framerate: f64) {
    let Some(vpxdec) = vpxdec_path() else {
        eprintln!("[vpxdec oracle SKIPPED] vpxdec not found; rvp9-decoder leg already green.");
        return;
    };
    let tmp = std::env::temp_dir();
    let ivf_path = tmp.join("rvp9_a4_rect_partition.ivf");
    {
        let mut f = std::fs::File::create(&ivf_path).expect("create ivf");
        let header = IvfHeader::vp9(W as u16, H as u16, framerate.round() as u32, 1);
        let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        write_ivf(&mut f, &header, &refs).expect("write_ivf");
    }
    let vpx_out = tmp.join("rvp9_a4_rect_partition_vpxdec.yuv");
    let vpx_yuv = run_vpxdec(&vpxdec, &ivf_path, &vpx_out)
        .expect("vpxdec must accept the A4-rect RD-partitioned bitstream");
    let rvp9_yuv = rvp9_decode_i420(packets);
    assert_eq!(
        vpx_yuv.len(),
        rvp9_yuv.len(),
        "A4-rect stream: vpxdec/rvp9 size mismatch ({} vs {})",
        vpx_yuv.len(),
        rvp9_yuv.len()
    );
    assert_eq!(
        vpx_yuv, rvp9_yuv,
        "A4-rect stream: vpxdec and rvp9-decoder disagree — the rect \
         partition-symbol emission / rect sub-block coding is not valid VP9"
    );
    eprintln!("vpxdec oracle: BYTE-EXACT on the A4-rect RD-partitioned stream.");
    let _ = std::fs::remove_file(&ivf_path);
    let _ = std::fs::remove_file(&vpx_out);
}

#[test]
fn a4_rect_partition_is_evaluated_and_decodes() {
    let framerate = 30.0;
    let frames = synthetic_clip();

    // Descent ON with full rectangular motion search. Disable the newer exact
    // variance tree so this focused legacy-A4 oracle reaches the recursive
    // descent. Production defaults use inherited/zero motion for rectangular
    // proxy candidates; this proof explicitly enables independent NEWMV
    // because its two bands require different motion vectors. This test runs
    // in its own process; no parallel test mutates these env vars.
    // SAFETY: this test owns these process env knobs for its test-binary run.
    unsafe {
        std::env::remove_var("RVP9_A4_RD_PARTITION_DISABLE");
        std::env::remove_var("RVP9_F3_NONRD_PARTITION_ENABLE");
        std::env::remove_var("RVP9_A4_RECT_DISABLE");
        std::env::set_var("RVP9_EXACT_VBP_TREE", "0");
        std::env::set_var("RVP9_A4_DESCENT_DISABLE_RECT_NEWMV", "0");
    }
    let (on_packets, on_rect_leaves) = encode(&frames, framerate);
    // SAFETY: restore the production default after the focused encode.
    unsafe {
        std::env::remove_var("RVP9_EXACT_VBP_TREE");
        std::env::remove_var("RVP9_A4_DESCENT_DISABLE_RECT_NEWMV");
    }

    eprintln!("descent ON: emitted {on_rect_leaves} rectangular leaves");

    // Assertion #1 (anti-G3 — rect genuinely evaluated and chosen): the
    // descent committed at least one rectangular leaf on the
    // diverging-horizontal-motion clip. A HORZ/VERT pick contributes 2
    // (two halves), so any rect decision yields >= 2 here. Zero would
    // mean rect was never evaluated, or never beat square.
    assert!(
        on_rect_leaves > 0,
        "A4-rect descent never emitted a rectangular leaf on \
         diverging-horizontal-motion content — PARTITION_HORZ/VERT is \
         not actually evaluated or never wins (stubbed/square-only?)"
    );

    // Assertion #2 (oracle — valid VP9): the rect-containing stream
    // decodes to N_FRAMES and is byte-exact across both decoders.
    let on_decoded_frames = {
        let bytes = rvp9_decode_i420(&on_packets);
        bytes.len() / ((W as usize) * (H as usize) * 3 / 2)
    };
    assert_eq!(
        on_decoded_frames, N_FRAMES as usize,
        "rect-containing stream did not decode to {N_FRAMES} frames (got {on_decoded_frames})"
    );
    verify_through_vpxdec(&on_packets, framerate);

    eprintln!(
        "A4-rect PROOF PASS: descent emitted {on_rect_leaves} rect leaves; \
         the rect-containing stream is byte-exact across decoders."
    );
}
