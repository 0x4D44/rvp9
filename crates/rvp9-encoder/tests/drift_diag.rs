#![allow(clippy::print_stderr)]

//! DRIFT-DIAG — decisive inter-frame drift diagnostic.
//!
//! Encodes ~8 inter frames of foreman_320x256 at BOTH cpu-used 7 (RT
//! picker) and cpu-used 4 (RDO dispatch), then decodes the produced
//! bitstream with `rvp9-decoder` frame-by-frame. For each frame it
//! compares:
//!
//!   * the encoder's STORED (post-LF, committed-to-pool) reference vs the
//!     true decode — this is the buffer the next inter frame predicts
//!     from. ANY non-zero divergence here is the drift bug.
//!   * the encoder's PRE-LF reconstruction vs the true decode — isolates
//!     whether the divergence is in the residual/MC path (pre-LF) or in
//!     the loop filter (post-LF only).
//!
//! Run:
//! ```text
//! cargo test --release -p rvp9-encoder --test drift_diag -- --ignored --nocapture
//! ```

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::y4m::read_cropped as y4m_read_cropped;
use std::path::PathBuf;

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

fn corpus_path() -> Option<PathBuf> {
    let p = PathBuf::from("tests/corpus/foreman_320x256_60f.y4m");
    if p.is_file() {
        return Some(p);
    }
    // Fall back to repo-root-relative when run from crate dir.
    let p2 = PathBuf::from("../../tests/corpus/foreman_320x256_60f.y4m");
    if p2.is_file() {
        return Some(p2);
    }
    None
}

fn run_for_speed(
    speed: i32,
    frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
    w: u32,
    h: u32,
    framerate: f64,
) {
    let w_us = w as usize;
    let h_us = h as usize;
    let uvw = w_us / 2;
    let uvh = h_us / 2;

    let cfg = EncoderConfig {
        width: w,
        height: h,
        framerate,
        target_bitrate_bps: 700_000,
        gop_length: frames.len() as u32 + 1,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        enable_alt_ref: false,
        rc_mode: RcMode::Cbr,
        cq_level: 0,
        temporal_layers: 1,
        enable_tpl: false,
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut dec = Decoder::new();

    eprintln!();
    eprintln!("=== foreman 320x256 cpu-used {speed} (700 kbps) ===");
    eprintln!("  frame kind  q  bytes  |  STORED(post-LF) vs DECODE        |  PRE-LF vs DECODE");
    eprintln!(
        "                          SSE(Y)   SSE(U)   SSE(V) mx(Y/U/V) |  SSE(Y)   SSE(U)   SSE(V) mx(Y/U/V)"
    );

    for (idx, (y, u, v)) in frames.iter().enumerate() {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, w_us))
            .expect("encode_frame");
        let parsed = decode_packet(&mut dec, &pkt.data);
        assert_eq!(parsed.len(), 1);
        let dec_buf = dec.decode_frame(&parsed[0]).expect("decode_frame");

        let dec_y = yv12_plane_tight(&dec_buf.y_plane(), w_us, h_us);
        let dec_u = yv12_plane_tight(&dec_buf.u_plane(), uvw, uvh);
        let dec_v = yv12_plane_tight(&dec_buf.v_plane(), uvw, uvh);

        // Post-LF stored reference (what the NEXT frame predicts from).
        let enc_y = enc.diag_last_recon_y().to_vec();
        let enc_u = enc.diag_last_recon_u().to_vec();
        let enc_v = enc.diag_last_recon_v().to_vec();

        let s_post_y = sse(&enc_y, &dec_y);
        let s_post_u = sse(&enc_u, &dec_u);
        let s_post_v = sse(&enc_v, &dec_v);
        let m_post_y = max_abs(&enc_y, &dec_y);
        let m_post_u = max_abs(&enc_u, &dec_u);
        let m_post_v = max_abs(&enc_v, &dec_v);

        // Pre-LF reconstruction.
        let (py, pu, pv) = enc.diag_prelf_recon().expect("prelf snapshot present");
        let s_pre_y = sse(py, &dec_y);
        let s_pre_u = sse(pu, &dec_u);
        let s_pre_v = sse(pv, &dec_v);
        let m_pre_y = max_abs(py, &dec_y);
        let m_pre_u = max_abs(pu, &dec_u);
        let m_pre_v = max_abs(pv, &dec_v);

        eprintln!(
            "  {:>3}  {:>4?} {:>3} {:>5} | {:>8} {:>8} {:>8} {:>2}/{:>2}/{:>2} | {:>8} {:>8} {:>8} {:>2}/{:>2}/{:>2}",
            idx,
            pkt.frame_kind,
            pkt.q_index,
            pkt.data.len(),
            s_post_y,
            s_post_u,
            s_post_v,
            m_post_y,
            m_post_u,
            m_post_v,
            s_pre_y,
            s_pre_u,
            s_pre_v,
            m_pre_y,
            m_pre_u,
            m_pre_v,
        );
    }
}

#[test]
#[ignore = "DRIFT-DIAG — encoder stored-reference vs true-decode divergence"]
fn drift_diag_foreman() {
    const W: u32 = 320;
    const H: u32 = 256;
    const N_FRAMES: usize = 8;

    let Some(path) = corpus_path() else {
        eprintln!("[SKIPPED] foreman_320x256_60f.y4m corpus not available.");
        return;
    };
    let (frames, framerate) =
        y4m_read_cropped(&path, W, H, N_FRAMES).expect("y4m_read_cropped foreman");
    assert_eq!(frames.len(), N_FRAMES);

    run_for_speed(7, &frames, W, H, framerate);
    run_for_speed(4, &frames, W, H, framerate);

    // CROSS-CHECK: same corpus cropped to 256x256 (4x4 SB) — the known
    // KF-SSE-0 config from diag_ref_divergence. If THIS reports KF SSE 0
    // but 320x256 does not, the divergence is width-dependent.
    let (frames256, fr256) =
        y4m_read_cropped(&path, 256, 256, N_FRAMES).expect("y4m_read_cropped 256");
    run_for_speed(7, &frames256, 256, 256, fr256);
}

/// Localize the FIRST diverging 8x8 MI blocks on inter frame 2 at
/// 320x256 cpu7 (the residual-drift case after the GOLDEN ref-index
/// fix). Reports the decoded mode/mv/ref so we can identify which code
/// path still diverges.
#[test]
#[ignore = "DRIFT-DIAG — per-block localization of frame-2 residual divergence"]
fn drift_diag_localize_frame2() {
    use rvp9_core::mode_info::BlockMode;
    const W: u32 = 320;
    const H: u32 = 256;
    const N: usize = 3;

    let Some(path) = corpus_path() else {
        eprintln!("[SKIPPED] corpus missing");
        return;
    };
    let (frames, framerate) = y4m_read_cropped(&path, W, H, N).expect("read");

    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: 700_000,
        gop_length: N as u32 + 1,
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
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("enc");
    let mut dec = Decoder::new();
    let w_us = W as usize;

    for (idx, (y, u, v)) in frames.iter().enumerate() {
        let pkt = enc.encode_frame(&yuv_view(y, u, v, w_us)).expect("encode");
        let parsed = decode_packet(&mut dec, &pkt.data);
        let dec_buf = dec.decode_frame(&parsed[0]).expect("decode");
        if idx != 2 {
            continue;
        }
        let dec_y = yv12_plane_tight(&dec_buf.y_plane(), w_us, H as usize);
        let py = enc.diag_last_recon_y().to_vec();
        let grid = dec.last_frame_mi_grid();
        let (mi_rows, mi_cols) = dec.last_frame_mi_dims();
        let mi_cols_us = mi_cols as usize;

        eprintln!("=== frame 2 per-8x8 Y divergence (post-LF enc vs decode) ===");
        let mut shown = 0;
        let mut total_div_blocks = 0;
        let mut ref_hist = [0u32; 4]; // intra, last, golden, altref
        for mr in 0..mi_rows as usize {
            for mc in 0..mi_cols_us {
                let mut blk_sse: u64 = 0;
                let mut blk_max: i32 = 0;
                for r in 0..8 {
                    for c in 0..8 {
                        let px = mr * 8 + r;
                        let pxc = mc * 8 + c;
                        if px >= H as usize || pxc >= w_us {
                            continue;
                        }
                        let off = px * w_us + pxc;
                        let d = i32::from(py[off]) - i32::from(dec_y[off]);
                        blk_sse += (d * d) as u64;
                        blk_max = blk_max.max(d.abs());
                    }
                }
                if blk_sse == 0 {
                    continue;
                }
                total_div_blocks += 1;
                let cell = grid[mr * mi_cols_us + mc];
                if let Some(mi) = cell {
                    match mi.mode {
                        BlockMode::Inter { ref_frames, .. } => {
                            use rvp9_core::mode_info::RefFrame;
                            match ref_frames[0] {
                                RefFrame::LastFrame => ref_hist[1] += 1,
                                RefFrame::GoldenFrame => ref_hist[2] += 1,
                                RefFrame::AltrefFrame => ref_hist[3] += 1,
                                _ => {}
                            }
                        }
                        BlockMode::Intra { .. } => ref_hist[0] += 1,
                    }
                }
                if shown < 24 {
                    shown += 1;
                    let desc = match cell {
                        Some(mi) => match mi.mode {
                            BlockMode::Inter {
                                mode,
                                mvs,
                                ref_frames,
                                interp_filter,
                            } => format!(
                                "INTER {:?} mv=({},{}) ref={:?} interp={:?} bs={:?} sk={} tx={:?}",
                                mode,
                                mvs[0].row,
                                mvs[0].col,
                                ref_frames[0],
                                interp_filter,
                                mi.block_size,
                                u8::from(mi.skip),
                                mi.tx_size,
                            ),
                            BlockMode::Intra { y_mode, .. } => format!(
                                "INTRA y={:?} bs={:?} sk={} tx={:?}",
                                y_mode,
                                mi.block_size,
                                u8::from(mi.skip),
                                mi.tx_size,
                            ),
                        },
                        None => "None".to_string(),
                    };
                    eprintln!(
                        "  mi=({:>2},{:>2}) sse={:>6} max={:>3} | {}",
                        mr, mc, blk_sse, blk_max, desc
                    );
                }
            }
        }
        eprintln!("  (showed {shown} of {total_div_blocks} diverging blocks)");
        eprintln!(
            "  diverging-block ref histogram: intra={} last={} golden={} altref={}",
            ref_hist[0], ref_hist[1], ref_hist[2], ref_hist[3]
        );
    }
}

/// Localize the FIRST diverging 8x8 MI block on inter frame 1 at
/// 256x256 (the clean KF-SSE-0 case), reporting the decoded
/// mode/mv/ref/skip so we can identify which code path diverges.
#[test]
#[ignore = "DRIFT-DIAG — per-block localization of frame-1 divergence"]
fn drift_diag_localize() {
    use rvp9_core::mode_info::BlockMode;
    const W: u32 = 256;
    const H: u32 = 256;
    const N: usize = 2;

    let Some(path) = corpus_path() else {
        eprintln!("[SKIPPED] corpus missing");
        return;
    };
    let (frames, framerate) = y4m_read_cropped(&path, W, H, N).expect("read");

    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: 700_000,
        gop_length: N as u32 + 1,
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
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("enc");
    let mut dec = Decoder::new();
    let w_us = W as usize;

    for (idx, (y, u, v)) in frames.iter().enumerate() {
        let pkt = enc.encode_frame(&yuv_view(y, u, v, w_us)).expect("encode");
        let parsed = decode_packet(&mut dec, &pkt.data);
        let dec_buf = dec.decode_frame(&parsed[0]).expect("decode");
        if idx != 1 {
            continue;
        }
        let dec_y = yv12_plane_tight(&dec_buf.y_plane(), w_us, H as usize);
        // Compare POST-LF encoder stored reference vs POST-LF decode (both
        // post-LF — apples to apples).
        let py = enc.diag_last_recon_y().to_vec();
        let grid = dec.last_frame_mi_grid();
        let (mi_rows, mi_cols) = dec.last_frame_mi_dims();
        let mi_cols_us = mi_cols as usize;

        eprintln!("=== frame 1 per-8x8 Y divergence (pre-LF enc vs decode) ===");
        let mut shown = 0;
        for mr in 0..mi_rows as usize {
            for mc in 0..mi_cols_us {
                // SSE over this 8x8 block.
                let mut blk_sse: u64 = 0;
                let mut blk_max: i32 = 0;
                for r in 0..8 {
                    for c in 0..8 {
                        let px = mr * 8 + r;
                        let pxc = mc * 8 + c;
                        if px >= H as usize || pxc >= w_us {
                            continue;
                        }
                        let off = px * w_us + pxc;
                        let d = i32::from(py[off]) - i32::from(dec_y[off]);
                        blk_sse += (d * d) as u64;
                        blk_max = blk_max.max(d.abs());
                    }
                }
                if blk_sse > 0 && shown < 24 {
                    shown += 1;
                    let cell = grid[mr * mi_cols_us + mc];
                    let desc = match cell {
                        Some(mi) => match mi.mode {
                            BlockMode::Inter {
                                mode,
                                mvs,
                                ref_frames,
                                interp_filter,
                            } => format!(
                                "INTER {:?} mv=({},{}) ref={:?} interp={:?} bs={:?} sk={} tx={:?}",
                                mode,
                                mvs[0].row,
                                mvs[0].col,
                                ref_frames[0],
                                interp_filter,
                                mi.block_size,
                                u8::from(mi.skip),
                                mi.tx_size,
                            ),
                            BlockMode::Intra { y_mode, .. } => format!(
                                "INTRA y={:?} bs={:?} sk={} tx={:?}",
                                y_mode,
                                mi.block_size,
                                u8::from(mi.skip),
                                mi.tx_size,
                            ),
                        },
                        None => "None".to_string(),
                    };
                    eprintln!(
                        "  mi=({:>2},{:>2}) sse={:>6} max={:>3} | {}",
                        mr, mc, blk_sse, blk_max, desc
                    );
                }
            }
        }
        eprintln!("  (showed {shown} diverging blocks)");
    }
}
