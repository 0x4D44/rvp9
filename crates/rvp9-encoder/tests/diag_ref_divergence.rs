#![allow(clippy::print_stderr)]

//! A11 — Encoder-side reference-buffer divergence diagnostic.
//!
//! A10 surfaced the hypothesis that the encoder's internal `last_recon_*`
//! buffer (used as LASTFRAME when picking + simulating inter frame N+1)
//! is NOT equal to the decoder's reconstruction of the same emitted
//! bitstream. Cross-decoder byte-exactness on the bitstream does NOT
//! imply parity on the reference buffer — both decoders (rvp9-decoder
//! and vpxdec) consume the bitstream identically, but the encoder's
//! internal buffer may differ from what every decoder reconstructs.
//!
//! The signature would be:
//! - frame 0 (KF): SSE(enc, dec) ≈ 0 (KF path is bit-exact by F5.1 oracle).
//! - frame N (inter, N >= 1): SSE(enc, dec) > 0 and **growing** with N.
//!
//! Run:
//! ```text
//! cargo test --release --test diag_ref_divergence -- --ignored --nocapture
//! ```

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{AqMode, EncoderConfig, StreamEncoder, YuvFrameRef};
use rvp9_harness::y4m::{probe_foreman_cif, probe_testsrc_720p, read_cropped as y4m_read_cropped};

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

fn decode_packet<'a>(dec: &mut Decoder, pkt: &'a [u8]) -> Vec<ParsedFrame<'a>> {
    let parsed = dec.parse_packet(pkt).expect("parse_packet");
    parsed
        .frames()
        .iter()
        .map(|pf| ParsedFrame {
            header: pf.header.clone(),
            header_end_byte: pf.header_end_byte,
            full_frame_data: pf.full_frame_data,
        })
        .collect()
}

/// Copy tight-strided plane out of a Yv12Buffer.
fn yv12_plane_tight(plane: &rvp9_core::frame::PlaneRef<'_>, w: usize, h: usize) -> Vec<u8> {
    let stride = plane.stride();
    let data = plane.data();
    let mut out = vec![0u8; w * h];
    for r in 0..h {
        out[r * w..r * w + w].copy_from_slice(&data[r * stride..r * stride + w]);
    }
    out
}

fn sse(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    let mut s: u64 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = i32::from(*x) - i32::from(*y);
        s += (d * d) as u64;
    }
    s
}

fn max_abs(a: &[u8], b: &[u8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (i32::from(*x) - i32::from(*y)).abs())
        .max()
        .unwrap_or(0)
}

/// Run the diagnostic on the given source frames. Reports per-frame
/// SSE / max_abs of `(encoder.last_recon_*, decoder.recon_*)`.
fn run_diagnostic(
    label: &str,
    frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
    w: u32,
    h: u32,
    framerate: f64,
    target_bitrate_bps: u32,
) {
    let w_us = w as usize;
    let h_us = h as usize;
    let uvw = w_us / 2;
    let uvh = h_us / 2;

    let cfg = EncoderConfig {
        width: w,
        height: h,
        framerate,
        target_bitrate_bps,
        gop_length: frames.len() as u32 + 1,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut dec = Decoder::new();

    eprintln!();
    eprintln!("=== {label} ===");
    eprintln!(
        "[A11 ref-divergence] {}x{} {} frames target {} bps",
        w, h, frames.len(), target_bitrate_bps
    );
    eprintln!(
        "  frame   kind  q   bytes    SSE(Y)      SSE(U)     SSE(V)   max|dY|  max|dU|  max|dV|"
    );

    for (idx, (y, u, v)) in frames.iter().enumerate() {
        let pkt = enc.encode_frame(&yuv_view(y, u, v, w_us)).expect("encode_frame");
        let parsed = decode_packet(&mut dec, &pkt.data);
        assert_eq!(parsed.len(), 1);
        let dec_buf = dec.decode_frame(&parsed[0]).expect("decode_frame");

        // Pull encoder's internal LASTFRAME and decoder's reconstruction
        // out as tight-strided planes for comparison.
        let enc_y = enc.diag_last_recon_y().to_vec();
        let enc_u = enc.diag_last_recon_u().to_vec();
        let enc_v = enc.diag_last_recon_v().to_vec();
        assert_eq!(enc_y.len(), w_us * h_us);
        assert_eq!(enc_u.len(), uvw * uvh);
        assert_eq!(enc_v.len(), uvw * uvh);

        let dec_y = yv12_plane_tight(&dec_buf.y_plane(), w_us, h_us);
        let dec_u = yv12_plane_tight(&dec_buf.u_plane(), uvw, uvh);
        let dec_v = yv12_plane_tight(&dec_buf.v_plane(), uvw, uvh);

        let sse_y = sse(&enc_y, &dec_y);
        let sse_u = sse(&enc_u, &dec_u);
        let sse_v = sse(&enc_v, &dec_v);
        let max_y = max_abs(&enc_y, &dec_y);
        let max_u = max_abs(&enc_u, &dec_u);
        let max_v = max_abs(&enc_v, &dec_v);

        eprintln!(
            "  {:>3}   {:>4?}  {:>3}  {:>6}  {:>9}  {:>9}  {:>9}     {:>3}      {:>3}      {:>3}",
            idx,
            pkt.frame_kind,
            pkt.q_index,
            pkt.data.len(),
            sse_y,
            sse_u,
            sse_v,
            max_y,
            max_u,
            max_v,
        );
    }
}

/// A11 — Run the diagnostic on foreman_cif 256×256 centre-crop at
/// 100 kbps, 12 frames. This is the corpus where libvpx-faithful PSNR
/// is locked at 16 dB.
#[test]
#[ignore = "A11 — encoder vs decoder LASTFRAME divergence diagnostic"]
fn diag_ref_divergence_foreman() {
    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 12;
    const TARGET_BITRATE_BPS: u32 = 100_000;

    let Some(y4m_path) = probe_foreman_cif() else {
        eprintln!("[SKIPPED] foreman_cif not available.");
        return;
    };
    let (frames, framerate) = y4m_read_cropped(&y4m_path, W, H, N_FRAMES as usize)
        .expect("y4m_read_cropped foreman");

    run_diagnostic("foreman_cif 256x256 @ 100 kbps", &frames, W, H, framerate, TARGET_BITRATE_BPS);
}

/// A11 — Run the diagnostic on testsrc 256×256 centre-crop at 1 Mbps,
/// 8 frames. testsrc has tiny inter residuals (synthetic motion) so
/// per-frame divergence should be much smaller than foreman, if any.
#[test]
#[ignore = "A11 — encoder vs decoder LASTFRAME divergence (testsrc)"]
fn diag_ref_divergence_testsrc() {
    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 8;
    const TARGET_BITRATE_BPS: u32 = 1_000_000;

    let Some(y4m_path) = probe_testsrc_720p() else {
        eprintln!("[SKIPPED] testsrc not available.");
        return;
    };
    let (frames, framerate) = y4m_read_cropped(&y4m_path, W, H, N_FRAMES as usize)
        .expect("y4m_read_cropped testsrc");

    run_diagnostic("testsrc 256x256 @ 1 Mbps", &frames, W, H, framerate, TARGET_BITRATE_BPS);
}
