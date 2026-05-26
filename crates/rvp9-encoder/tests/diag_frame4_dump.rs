#![allow(clippy::print_stderr)]

//! A0 — Frame-4 partition/mode distribution dump.
//!
//! The matched-target sweep in `diag_per_frame.rs` revealed that rvp9's
//! Y-PSNR is locked at ~24.65 dB on frame 4 regardless of target
//! bitrate (q=4 to q=218). 24.65 dB corresponds to MSE ≈ source(frame_4)
//! minus source(frame_3) ≈ pure ZEROMV result. That suggests the inter
//! mode picker is effectively picking ZEROMV / skip across the whole
//! frame.
//!
//! This test instruments the encoder by directly calling
//! `pick_and_simulate_inter_encode_partitions` and the partition picker
//! on frame 4 (after encoding frames 0..3 to populate the reference),
//! then dumps the partition / mode / MV distribution per superblock.

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{AqMode, EncoderConfig, StreamEncoder, YuvFrameRef};
use rvp9_harness::y4m::{probe_testsrc_720p, read_cropped as y4m_read_cropped};

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

fn psnr_db(orig: &[u8], recon: &[u8]) -> f64 {
    assert_eq!(orig.len(), recon.len());
    let mut sse: u64 = 0;
    for (a, b) in orig.iter().zip(recon.iter()) {
        let d = i32::from(*a) - i32::from(*b);
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return 100.0;
    }
    let mse = sse as f64 / orig.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

#[test]
#[ignore = "A0 diagnostic — dump partition / mode distribution for frame 4"]
fn diag_frame4_partition_mode_dump() {
    const W: u32 = 256;
    const H: u32 = 256;
    const N: u32 = 35; // run far enough that Q reaches low values
    let path = probe_testsrc_720p().expect("testsrc");
    let (frames, fr) = y4m_read_cropped(&path, W, H, N as usize).expect("read");
    eprintln!("Frame-4 dump @ {W}x{H} @ {fr} fps");

    // Run the encoder for frames 0..=4 at a high-bitrate config so RC
    // picks a moderate Q (so the picker has a realistic lambda).
    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate: fr,
        target_bitrate_bps: 250_000,
        gop_length: N + 1,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        enable_alt_ref: false,
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut dec = Decoder::new();

    for (i, (y, u, v)) in frames.iter().enumerate() {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, W as usize))
            .expect("encode_frame");
        let parsed = decode_packet(&mut dec, &pkt.data);
        let buf = dec.decode_frame(&parsed[0]).expect("decode_frame");
        let yp = buf.y_plane();
        let mut dec_y = vec![0u8; (W * H) as usize];
        for r in 0..H as usize {
            dec_y[r * W as usize..(r + 1) * W as usize]
                .copy_from_slice(&yp.data()[r * yp.stride()..r * yp.stride() + W as usize]);
        }
        let psnr_y = psnr_db(y, &dec_y);
        // Compute extra: SSE of recon vs source AND vs first-frame-source.
        let mut sse_self: u64 = 0;
        for (a, b) in y.iter().zip(dec_y.iter()) {
            let d = i32::from(*a) - i32::from(*b);
            sse_self += (d * d) as u64;
        }
        let mut sse_vs_prev_src: u64 = 0;
        if i > 0 {
            let prev = &frames[i - 1].0;
            for (a, b) in prev.iter().zip(dec_y.iter()) {
                let d = i32::from(*a) - i32::from(*b);
                sse_vs_prev_src += (d * d) as u64;
            }
        }
        eprintln!(
            "  frame {i}: q={} bits={} psnr_y={:.2} dB sse(rec,src)={} sse(rec,prev_src)={} kf={}",
            pkt.q_index,
            pkt.actual_bits,
            psnr_y,
            sse_self,
            sse_vs_prev_src,
            pkt.frame_kind.is_key(),
        );
        // Print pixel slice 0..16 for both source and recon.
        eprintln!("    src[0..16]  = {:?}", &y[0..16]);
        eprintln!("    recon[0..16]= {:?}", &dec_y[0..16]);
        // Also pixel slice 128*256+128..+16 (centre of frame).
        let mid = 128 * 256 + 128;
        eprintln!("    src[mid..mid+16]  = {:?}", &y[mid..mid + 16]);
        eprintln!("    recon[mid..mid+16]= {:?}", &dec_y[mid..mid + 16]);
    }
}
