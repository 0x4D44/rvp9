#![allow(clippy::print_stderr)]

//! A0' — Foreman KF intra-mode + per-plane PSNR diagnostic.
//!
//! We saw foreman_cif's KF lands at PSNR 19/16/24/25 dB even at q=124
//! (250 kbps target burns 37k bits on the KF). This test:
//!   - dumps per-plane (Y/U/V) PSNR for the KF
//!   - dumps the intra-mode distribution chosen by the picker for the
//!     KF (counts per mode across all leaves)
//!   - reports KF bit-spend
//!
//! If chroma DC-only is hurting (the encoder forces chroma intra = DC
//! regardless of luma mode), per-plane PSNR will show U/V much worse
//! than Y. If the picker is fine but the partition picker is splitting
//! too coarsely, we'll see most leaves at Block32x32+ on a textured
//! face.

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{AqMode, EncoderConfig, StreamEncoder, YuvFrameRef};
use rvp9_harness::y4m::{probe_foreman_cif, read_cropped as y4m_read_cropped};

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
#[ignore = "A0' KF intra diagnostic"]
fn diag_foreman_kf() {
    const W: u32 = 256;
    const H: u32 = 256;

    let Some(src) = probe_foreman_cif() else {
        eprintln!("    [skip] foreman_cif.y4m not found");
        return;
    };
    let (frames, framerate) = y4m_read_cropped(&src, W, H, 1).expect("read foreman_cif KF");

    for &tb in &[25_000u32, 50_000, 100_000, 250_000, 1_000_000] {
        eprintln!("\n=== target {} kbps (KF only) ===", tb / 1000);
        let cfg = EncoderConfig {
            width: W,
            height: H,
            framerate,
            target_bitrate_bps: tb,
            gop_length: 1,
            aq_mode: AqMode::Off,
            use_prev_frame_mvs: false,
            speed: 7,
            deadline: rvp9_encoder::Deadline::Realtime,
            noise_sensitivity: 0,
            bit_depth: rvp9_core::BitDepth::Bd8,
            min_qindex: 4,
            max_qindex: 255,
        };
        let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
        let mut dec = Decoder::new();
        let (y, u, v) = &frames[0];
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, W as usize))
            .expect("encode_frame");
        let parsed = decode_packet(&mut dec, &pkt.data);
        let buf = dec.decode_frame(&parsed[0]).expect("decode_frame");
        let yp = buf.y_plane();
        let up = buf.u_plane();
        let vp = buf.v_plane();
        let mut dec_y = vec![0u8; (W * H) as usize];
        let mut dec_u = vec![0u8; ((W / 2) * (H / 2)) as usize];
        let mut dec_v = vec![0u8; ((W / 2) * (H / 2)) as usize];
        for r in 0..H as usize {
            dec_y[r * W as usize..(r + 1) * W as usize]
                .copy_from_slice(&yp.data()[r * yp.stride()..r * yp.stride() + W as usize]);
        }
        for r in 0..(H / 2) as usize {
            dec_u[r * (W / 2) as usize..(r + 1) * (W / 2) as usize]
                .copy_from_slice(&up.data()[r * up.stride()..r * up.stride() + (W / 2) as usize]);
            dec_v[r * (W / 2) as usize..(r + 1) * (W / 2) as usize]
                .copy_from_slice(&vp.data()[r * vp.stride()..r * vp.stride() + (W / 2) as usize]);
        }
        let psnr_y = psnr_db(y, &dec_y);
        let psnr_u = psnr_db(u, &dec_u);
        let psnr_v = psnr_db(v, &dec_v);

        // Where is the error concentrated? Compute SSE on the centre
        // crop (where the face is) vs the periphery.
        let mut center_sse: u64 = 0;
        let mut edge_sse: u64 = 0;
        for r in 0..H as usize {
            for c in 0..W as usize {
                let d = i32::from(y[r * W as usize + c])
                    - i32::from(dec_y[r * W as usize + c]);
                let in_center = (64..192).contains(&r) && (64..192).contains(&c);
                if in_center {
                    center_sse += (d * d) as u64;
                } else {
                    edge_sse += (d * d) as u64;
                }
            }
        }
        let center_mse = center_sse as f64 / (128.0 * 128.0);
        let edge_mse = edge_sse as f64 / ((W * H) as f64 - 128.0 * 128.0);
        let center_psnr = if center_mse == 0.0 {
            100.0
        } else {
            10.0 * (255.0 * 255.0 / center_mse).log10()
        };
        let edge_psnr = if edge_mse == 0.0 {
            100.0
        } else {
            10.0 * (255.0 * 255.0 / edge_mse).log10()
        };

        eprintln!(
            "  q={} bits={} PSNR_Y={:.2} PSNR_U={:.2} PSNR_V={:.2}",
            pkt.q_index, pkt.actual_bits, psnr_y, psnr_u, psnr_v,
        );
        eprintln!(
            "  centre 128x128 PSNR_Y={:.2} dB  edge PSNR_Y={:.2} dB  (centre captures the face)",
            center_psnr, edge_psnr,
        );
    }
}
