//! A23 — TPL (Temporal-dependency Model) integration PROOF.
//!
//! Verifies the opt-in TPL subsystem end-to-end:
//!
//! 1. **Anti-G3 (counter).** With `enable_tpl = true` on content with
//!    motion (and a lookahead deep enough for a forward window), the
//!    single-GOP forward TPL pass produces non-trivial propagation stats
//!    AND the per-superblock rdmult is adjusted: the global diagnostic
//!    counter [`rvp9_encoder::tpl::TPL_RDMULT_ADJUSTED_BLOCKS`] advances
//!    strictly above zero. With `enable_tpl = false` it stays at zero.
//!
//!    This FAILS if the dispenser is stubbed: a stub that left every
//!    block's `mc_dep_cost == intra_cost` would yield `beta == r0 / r0 ==
//!    1` and `get_rdmult_delta` would return the base rdmult unchanged,
//!    so `cb_rdmult == orig` and the counter would never advance. Only a
//!    real forward motion search + real backward propagation grows
//!    `mc_dep_cost` above `intra_cost`, producing a `cb_rdmult != orig`.
//!
//! 2. **Oracle (byte-exact + differs).** The TPL-on stream is byte-exact
//!    through vpxdec (valid VP9) AND differs from the TPL-off stream
//!    (proving TPL changes the RDO lambda → changes mode/MV preference →
//!    changes the bitstream).
//!
//! libvpx file:line citations:
//! - `vp9/encoder/vp9_tpl_model.c:1138-1255` — `mc_flow_dispenser`.
//! - `vp9/encoder/vp9_tpl_model.c:571-727` — `mode_estimation`.
//! - `vp9/encoder/vp9_tpl_model.c:280-330` — `motion_compensated_prediction`.
//! - `vp9/encoder/vp9_encodeframe.c:3719-3768` — `get_rdmult_delta`.
//! - `vp9/encoder/vp9_encodeframe.c:2235-2242` — RDO consumer hook.
//! - `vp9/encoder/vp9_encodeframe.c:6151-6162` — `cpi->rd.r0`.

#![allow(clippy::print_stderr)]
#![allow(missing_docs)]

use rvp9_decoder::Decoder;
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;

const W: u32 = 128;
const H: u32 = 128;

/// Serialises every test in this file that touches the process-global
/// `TPL_RDMULT_ADJUSTED_BLOCKS` counter (cargo runs `#[test]`s in
/// parallel by default).
static COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A `W x H` 4:2:0 frame with a bright square that MOVES with `seed`,
/// over a static textured gradient background. The motion gives the TPL
/// motion search a real, non-zero MV to find (and gives downstream
/// frames a real dependency on the blocks the square passes through).
fn moving_frame(seed: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = W as usize;
    let h = H as usize;
    let uvw = w / 2;
    let uvh = h / 2;
    let mut y = vec![0u8; w * h];
    // Textured background.
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = (40 + (((r * 3) ^ (c * 5)) & 0x3f)) as u8;
        }
    }
    // Moving bright 24x24 square; advances ~6 px/frame diagonally.
    let off = (seed as usize * 6) % (w - 24);
    let sx = 8 + off;
    let sy = 8 + ((seed as usize * 6) % (h - 24));
    for r in sy..sy + 24 {
        for c in sx..sx + 24 {
            y[r * w + c] = 230;
        }
    }
    let u = vec![128u8; uvw * uvh];
    let v = vec![128u8; uvw * uvh];
    (y, u, v)
}

fn yuv_view<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> YuvFrameRef<'a> {
    YuvFrameRef {
        y,
        u,
        v,
        y_stride: W as usize,
        u_stride: (W as usize) / 2,
        v_stride: (W as usize) / 2,
    }
}

fn base_cfg(enable_tpl: bool, lag: u32) -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        framerate: 30.0,
        target_bitrate_bps: 1_000_000,
        gop_length: 16,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        // TPL is a Good/Best-quality (non-RT) feature in libvpx
        // (`cpi->sf.enable_tpl_model` is set only on the non-RT speed
        // presets). speed <= 4 routes inter leaves through the RDO
        // `InterDispatchLeaf` picker — the rd-multiplier path the TPL
        // `cb_rdmult` substitutes into. (At speed >= 5 the RT nonrd
        // picker is used and the TPL hook is inert, matching libvpx.)
        speed: 4,
        deadline: rvp9_encoder::Deadline::Good,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: lag,
        enable_alt_ref: false,
        rc_mode: RcMode::Cbr,
        cq_level: 0,
        temporal_layers: 1,
        enable_tpl,
        min_qindex: 4,
        max_qindex: 255,
    }
}

/// Encode `n_frames` moving-square frames with the given TPL config,
/// returning the transport packets in encode order. Uses the
/// delayed-output path when `lag > 0` (required for TPL's forward
/// window).
fn encode(enable_tpl: bool, lag: u32, n_frames: u32) -> Vec<Vec<u8>> {
    let cfg = base_cfg(enable_tpl, lag);
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut packets: Vec<Vec<u8>> = Vec::new();
    if lag == 0 {
        for i in 0..n_frames {
            let (y, u, v) = moving_frame(i as u8);
            let pkt = enc
                .encode_frame(&yuv_view(&y, &u, &v))
                .expect("encode_frame");
            packets.push(pkt.data);
        }
    } else {
        for i in 0..n_frames {
            let (y, u, v) = moving_frame(i as u8);
            match enc.encode_frame_delayed(&yuv_view(&y, &u, &v)) {
                Ok(Some(pkt)) => packets.push(pkt.data),
                Ok(None) => {}
                Err(e) => panic!("encode_frame_delayed: {e:?}"),
            }
        }
        for r in enc.flush() {
            packets.push(r.expect("flush ok").data);
        }
    }
    packets
}

/// A23 PROOF #1 — anti-G3 counter. Both phases live in one test (the
/// counter is a process global; combining avoids cross-test races).
#[test]
fn w_tpl_rdmult_adjusted_only_when_enabled() {
    let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let counter = &rvp9_encoder::tpl::TPL_RDMULT_ADJUSTED_BLOCKS;

    // Phase OFF — enable_tpl = false. The hook substitution is gated, so
    // the counter must NOT advance.
    counter.store(0, Ordering::Relaxed);
    let _ = encode(
        /* enable_tpl = */ false, /* lag = */ 8, /* n = */ 12,
    );
    let off = counter.load(Ordering::Relaxed);
    eprintln!("TPL OFF: rdmult-adjusted blocks = {off}");
    assert_eq!(off, 0, "TPL disabled must not adjust any block's rdmult");

    // Phase ON — enable_tpl = true with a forward window (lag = 8 ⇒ the
    // single-GOP forward TPL pass has downstream frames to propagate
    // from). The counter must advance strictly above zero.
    counter.store(0, Ordering::Relaxed);
    let _ = encode(
        /* enable_tpl = */ true, /* lag = */ 8, /* n = */ 12,
    );
    let on = counter.load(Ordering::Relaxed);
    eprintln!("TPL ON: rdmult-adjusted blocks = {on}");
    assert!(
        on > 0,
        "TPL enabled on moving content must adjust >= 1 superblock's \
         rdmult (real mc_dep_cost > intra_cost from forward propagation); \
         a stubbed dispenser leaves mc_dep_cost == intra_cost ⇒ beta == 1 \
         ⇒ cb_rdmult == orig ⇒ counter stays 0"
    );
}

/// A23 PROOF #2a — the TPL-on bitstream DIFFERS from the TPL-off
/// bitstream (TPL changes the RDO lambda ⇒ changes the stream). Runs
/// without any external decoder.
#[test]
fn w_tpl_on_stream_differs_from_off() {
    let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let n = 12u32;
    let off = encode(false, 8, n);
    let on = encode(true, 8, n);

    let off_bytes: usize = off.iter().map(Vec::len).sum();
    let on_bytes: usize = on.iter().map(Vec::len).sum();
    eprintln!(
        "TPL off: {} packets / {off_bytes} bytes; on: {} packets / {on_bytes} bytes",
        off.len(),
        on.len()
    );
    assert_eq!(
        off.len(),
        on.len(),
        "frame count must match (TPL changes bytes, not frame structure)"
    );
    // At least one packet must differ — TPL changes mode/MV preference.
    let differ = off.iter().zip(on.iter()).any(|(a, b)| a != b);
    assert!(
        differ,
        "TPL-on stream must differ from TPL-off (a no-op TPL pass would \
         produce byte-identical output, failing this anti-G3 check)"
    );

    // Both streams must still decode cleanly through rvp9-decoder.
    for (label, packets) in [("off", &off), ("on", &on)] {
        let mut dec = Decoder::new();
        let mut shown = 0u32;
        for pkt in packets.iter() {
            let parsed = dec.parse_packet(pkt).expect("parse_packet");
            let owned: Vec<_> = parsed
                .frames()
                .iter()
                .map(|pf| rvp9_decoder::ParsedFrame {
                    header: pf.header.clone(),
                    header_end_byte: pf.header_end_byte,
                    full_frame_data: pf.full_frame_data,
                })
                .collect();
            for pf in &owned {
                let _ = dec.decode_frame(pf).expect("rvp9 decode_frame");
                if pf.header.show_frame {
                    shown += 1;
                }
            }
        }
        eprintln!("TPL {label}: rvp9-decoder decoded {shown} show frames");
        assert_eq!(shown, n, "every input frame must produce one show frame");
    }
}

// ---------------------------------------------------------------------
// vpxdec oracle (byte-exact). #[ignore]-gated like cross_decoder_gate.
// ---------------------------------------------------------------------

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

fn yv12_to_i420(buf: &rvp9_core::frame::Yv12Buffer, w: u32, h: u32) -> Vec<u8> {
    let yp = buf.y_plane();
    let up = buf.u_plane();
    let vp = buf.v_plane();
    let (w, h) = (w as usize, h as usize);
    let (uvw, uvh) = (w / 2, h / 2);
    let mut out = Vec::with_capacity(w * h + 2 * uvw * uvh);
    for r in 0..h {
        let o = r * yp.stride();
        out.extend_from_slice(&yp.data()[o..o + w]);
    }
    for r in 0..uvh {
        let o = r * up.stride();
        out.extend_from_slice(&up.data()[o..o + uvw]);
    }
    for r in 0..uvh {
        let o = r * vp.stride();
        out.extend_from_slice(&vp.data()[o..o + uvw]);
    }
    out
}

/// A23 PROOF #2b — the TPL-on stream is byte-exact through vpxdec (valid
/// VP9) AND differs from the TPL-off stream. Skips if vpxdec is absent.
#[test]
#[ignore = "A23 TPL vpxdec oracle; run with `--ignored`"]
fn w_tpl_on_byte_exact_through_vpxdec_and_differs() {
    use rvp9_harness::container::{IvfHeader, write_ivf};

    let Some(vpxdec) = vpxdec_path() else {
        eprintln!("[SKIPPED] vpxdec not available.");
        return;
    };

    let n = 12u32;
    let off = encode(false, 8, n);
    let on = encode(true, 8, n);

    // Anti-G3: TPL must actually change the stream.
    assert!(
        off.iter().zip(on.iter()).any(|(a, b)| a != b),
        "TPL-on stream must differ from TPL-off"
    );

    // Decode the TPL-on stream through rvp9-decoder → reference YUV.
    let rvp9_yuv: Vec<u8> = {
        let mut dec = Decoder::new();
        let mut out = Vec::new();
        for pkt in &on {
            let parsed = dec.parse_packet(pkt).expect("parse_packet");
            let owned: Vec<_> = parsed
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
                out.extend_from_slice(&yv12_to_i420(&buf, W, H));
            }
        }
        out
    };

    // Wrap the TPL-on packets in an IVF and decode through vpxdec.
    let tmp = std::env::temp_dir();
    let ivf_path = tmp.join("rvp9_w_tpl_on.ivf");
    {
        let mut f = std::fs::File::create(&ivf_path).expect("create ivf");
        let header = IvfHeader::vp9(W as u16, H as u16, 30, 1);
        let refs: Vec<&[u8]> = on.iter().map(Vec::as_slice).collect();
        write_ivf(&mut f, &header, &refs).expect("write_ivf");
    }
    let vpx_out = tmp.join("rvp9_w_tpl_on_vpxdec.yuv");
    let vpx_yuv = run_vpxdec(&vpxdec, &ivf_path, &vpx_out).expect("vpxdec decode");

    eprintln!(
        "TPL-on: rvp9-decoder {} bytes, vpxdec {} bytes",
        rvp9_yuv.len(),
        vpx_yuv.len()
    );
    assert_eq!(
        vpx_yuv, rvp9_yuv,
        "TPL-on stream must be BYTE-EXACT through vpxdec (valid VP9)"
    );

    let _ = std::fs::remove_file(&ivf_path);
    let _ = std::fs::remove_file(&vpx_out);
    eprintln!("A23 vpxdec oracle: PASS (byte-exact + differs from TPL-off)");
}
