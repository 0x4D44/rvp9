// Integration test: stderr IS the diagnostic mechanism (per-decoder
// verdict reporting), mirroring `cross_decoder_gate.rs` and
// `w_altref_real_coding.rs`. The workspace `print_stderr = "deny"` lint
// targets library code, not this binary.
#![allow(clippy::print_stderr)]
#![allow(missing_docs)]

//! W-ALTREF picker iteration PROOF.
//!
//! Direct follow-on to commit 0ad7e7d (slot 2 became a real ALTREF
//! reconstruction). This wave makes the **show-frame inter-mode picker
//! iterate ALTREF (ref-pool slot 2) as a single-reference candidate**,
//! gated to `enable_alt_ref = true`. With ALTREF off (the cross-decoder
//! gate's config) the picker behaviour — and the bitstream — is
//! byte-identical to before.
//!
//! ## Oracles
//!
//! 1. **Anti-G3: at least one show-frame leaf selects ALTREF.** The
//!    encoder exposes the `RDO_DISPATCH_ALTREF_PICKS` counter, bumped once
//!    per leaf whose transform-domain RD winner resolved to
//!    `RefFrame::AltrefFrame`. On content where the temporally-filtered
//!    (denoised) ALTREF is a cleaner predictor than the previous noisy
//!    LAST reconstruction, the counter MUST advance. If the ALTREF
//!    iteration is stripped (the `altref` slot forced unavailable, or the
//!    gate inverted) the counter stays at zero and this assertion FAILS.
//!
//! 2. **The ALTREF-referencing stream is BYTE-EXACT decodable by vpxdec.**
//!    Encode the ALTREF stream, wrap in IVF, decode through both
//!    rvp9-decoder and libvpx-stock `vpxdec`. The two MUST agree
//!    byte-for-byte — a real-coded show frame that names ALTREF in its
//!    ref set must reconstruct identically on libvpx. Reuses the exact
//!    `vpxdec --i420` invocation from `cross_decoder_gate.rs`. `#[ignore]`
//!    -gated (needs the libvpx binary); SKIPs informationally when absent.
//!
//! Both run at `speed = 3` so the per-leaf decision routes through the
//! transform-domain RDO dispatch picker (`speed.abs() <= 4`), the path
//! that iterates the ALTREF single-ref candidate. (The realtime picker at
//! speed 6/7 is LAST/GOLDEN-only and is out of scope for this wave — see
//! the report's scope note.)
//!
//! libvpx file:line citations:
//! - `vp9/encoder/vp9_rdopt.c:3533`, `:3545`, `:4349` — the ref-candidate
//!   loop `for (ref_frame = LAST_FRAME; ref_frame <= ALTREF_FRAME; …)`
//!   iterates ALTREF as a first-class single-ref candidate.
//! - `vp9/encoder/vp9_rdopt.c:2511` `setup_buffer_inter` — wires the
//!   ALTREF reference buffer for the candidate's motion search.
//! - `vp9/encoder/vp9_encoder.c:5034-5049` `set_ref_sign_bias` — ALTREF
//!   is a backward (future display-order) reference → `sign_bias = true`.

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::rdo::inter_dispatch::RDO_DISPATCH_ALTREF_PICKS;
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::container::{IvfHeader, write_ivf};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;

const W: u32 = 128;
const H: u32 = 128;
const N_FRAMES: u32 = 16;

// ---------------------------------------------------------------------------
// Decoder probe + invocation — copied verbatim from cross_decoder_gate.rs.
// ---------------------------------------------------------------------------

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
            "vpxdec failed: stderr={}",
            String::from_utf8_lossy(&status.stderr)
        )));
    }
    std::fs::read(out)
}

// ---------------------------------------------------------------------------
// Synthetic content: a fixed structured base image plus per-frame additive
// pseudo-random noise. The temporal filter averages the noise across the
// ARF's neighbour window, so the ALTREF reconstruction is close to the
// clean base. A show frame's source is noisy; predicting from the clean
// (denoised) ALTREF leaves a smaller residual than predicting from the
// previous noisy LAST reconstruction — so the RD picker should select
// ALTREF for some leaves. This is REAL: ME against slot 2 + real residual
// RD, no synthetic model.
// ---------------------------------------------------------------------------

/// Deterministic per-(frame,x,y) noise via a small integer hash (no
/// external rng dependency). Range roughly ±12.
fn noise_at(frame: u32, x: usize, y: usize) -> i32 {
    let mut h = frame
        .wrapping_mul(2_654_435_761)
        .wrapping_add((x as u32).wrapping_mul(40_503))
        .wrapping_add((y as u32).wrapping_mul(2_246_822_519));
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    ((h % 25) as i32) - 12
}

/// Clean structured base luma sample at (x, y): smooth gradients plus a
/// few constant-luma rectangles (the kind of content where a denoised
/// predictor clearly beats a noisy one).
fn base_y(x: usize, y: usize) -> u8 {
    let grad = (x + y) / 2;
    let blocky = if (x / 24 + y / 24).is_multiple_of(2) {
        40
    } else {
        90
    };
    (((grad + blocky) % 200) + 28) as u8
}

fn base_uv(x: usize, y: usize) -> u8 {
    (128 + ((x / 16) as i32 - (y / 16) as i32) * 6).clamp(16, 240) as u8
}

fn make_frame(frame: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = W as usize;
    let h = H as usize;
    let uvw = w / 2;
    let uvh = h / 2;
    let mut yb = vec![0u8; w * h];
    let mut ub = vec![0u8; uvw * uvh];
    let mut vb = vec![0u8; uvw * uvh];
    for y in 0..h {
        for x in 0..w {
            let v = base_y(x, y) as i32 + noise_at(frame, x, y);
            yb[y * w + x] = v.clamp(0, 255) as u8;
        }
    }
    for y in 0..uvh {
        for x in 0..uvw {
            // Chroma carries lighter noise so it stays mostly flat.
            let n = noise_at(frame ^ 0x55, x, y) / 4;
            ub[y * uvw + x] = (base_uv(x, y) as i32 + n).clamp(0, 255) as u8;
            vb[y * uvw + x] = (base_uv(y, x) as i32 - n).clamp(0, 255) as u8;
        }
    }
    (yb, ub, vb)
}

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

fn config(enable_alt_ref: bool) -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        framerate: 30.0,
        target_bitrate_bps: 1_500_000,
        // gop_length 16 → arf_interval = max(8, 2) = 8.
        gop_length: 16,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        // speed <= 4 routes per-leaf decisions through the transform-domain
        // RDO dispatch picker, the path that iterates the ALTREF candidate.
        speed: 3,
        deadline: rvp9_encoder::Deadline::Good,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        // lag must exceed arf_interval so the lookahead can peek a
        // neighbour for the temporal filter when the cadence fires.
        lag_in_frames: 9,
        enable_alt_ref,
        rc_mode: RcMode::Cbr,
        cq_level: 0,
        temporal_layers: 1,
        enable_tpl: false,
        stage4a_product_split_override: Default::default(),
        min_qindex: 4,
        max_qindex: 255,
    }
}

fn decode_packet<'a>(dec: &mut Decoder, pkt: &'a [u8]) -> Vec<ParsedFrame<'a>> {
    dec.parse_packet(pkt)
        .expect("parse_packet")
        .frames()
        .iter()
        .map(|pf| ParsedFrame {
            header: pf.header.clone(),
            header_end_byte: pf.header_end_byte,
            full_frame_data: pf.full_frame_data,
        })
        .collect()
}

fn yv12_to_i420_bytes(buf: &rvp9_core::frame::Yv12Buffer, w: u32, h: u32) -> Vec<u8> {
    let yp = buf.y_plane();
    let up = buf.u_plane();
    let vp = buf.v_plane();
    let w_us = w as usize;
    let h_us = h as usize;
    let uvw_us = w_us / 2;
    let uvh_us = h_us / 2;
    let mut out = Vec::with_capacity(w_us * h_us + 2 * uvw_us * uvh_us);
    for r in 0..h_us {
        let off = r * yp.stride();
        out.extend_from_slice(&yp.data()[off..off + w_us]);
    }
    for r in 0..uvh_us {
        let off = r * up.stride();
        out.extend_from_slice(&up.data()[off..off + uvw_us]);
    }
    for r in 0..uvh_us {
        let off = r * vp.stride();
        out.extend_from_slice(&vp.data()[off..off + uvw_us]);
    }
    out
}

fn plane_diff_stats(a: &[u8], b: &[u8]) -> (i32, u64) {
    assert_eq!(a.len(), b.len());
    let mut max_abs: i32 = 0;
    let mut sse: u64 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = i32::from(*x) - i32::from(*y);
        if d.abs() > max_abs {
            max_abs = d.abs();
        }
        sse += (d * d) as u64;
    }
    (max_abs, sse)
}

/// Encode all `N_FRAMES` through the delayed (lag) path + flush, returning
/// the transport packets in encode order. ALTREF requires lookahead, so
/// the synchronous `encode_frame` path cannot fire an ARF.
fn encode_stream(enable_alt_ref: bool) -> Vec<Vec<u8>> {
    let mut enc = StreamEncoder::new(config(enable_alt_ref)).expect("StreamEncoder::new");
    let mut packets: Vec<Vec<u8>> = Vec::new();
    for f in 0..N_FRAMES {
        let (y, u, v) = make_frame(f);
        if let Some(pkt) = enc
            .encode_frame_delayed(&yuv_view(&y, &u, &v))
            .expect("encode_frame_delayed")
        {
            packets.push(pkt.data);
        }
    }
    for r in enc.flush() {
        packets.push(r.expect("flush ok").data);
    }
    packets
}

// ---------------------------------------------------------------------------
// PROOF 1 — anti-G3: at least one show-frame leaf selects ALTREF.
// ---------------------------------------------------------------------------

#[test]
fn show_frame_picker_selects_altref() {
    // Encode once with ALTREF OFF to capture the baseline ALTREF-pick
    // count delta (must be exactly zero — the gate keeps the picker
    // LAST/GOLDEN-only), then with ALTREF ON (must advance).
    let off_before = RDO_DISPATCH_ALTREF_PICKS.load(Ordering::Relaxed);
    let _off_packets = encode_stream(/* enable_alt_ref = */ false);
    let off_after = RDO_DISPATCH_ALTREF_PICKS.load(Ordering::Relaxed);
    let off_delta = off_after - off_before;
    eprintln!("ALTREF picks with enable_alt_ref=false: {off_delta}");
    assert_eq!(
        off_delta, 0,
        "with enable_alt_ref=false the picker MUST NOT iterate ALTREF \
         (gate-safety: behaviour byte-identical to the LAST/GOLDEN-only path)"
    );

    let on_before = RDO_DISPATCH_ALTREF_PICKS.load(Ordering::Relaxed);
    let _on_packets = encode_stream(/* enable_alt_ref = */ true);
    let on_after = RDO_DISPATCH_ALTREF_PICKS.load(Ordering::Relaxed);
    let on_delta = on_after - on_before;
    eprintln!("ALTREF picks with enable_alt_ref=true:  {on_delta}");
    assert!(
        on_delta >= 1,
        "with enable_alt_ref=true the show-frame picker must select \
         RefFrame::AltrefFrame for at least one leaf (the temporally-\
         filtered ALTREF is a cleaner predictor than the noisy LAST on \
         this content). Got {on_delta} ALTREF picks. If this is zero, the \
         ALTREF single-ref candidate is not being iterated."
    );
}

// ---------------------------------------------------------------------------
// PROOF 2 — the ALTREF-referencing stream is BYTE-EXACT decodable by vpxdec.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "W-ALTREF picker byte-exact vpxdec round-trip; run with `--ignored`"]
fn altref_picker_stream_byte_exact_vpxdec() {
    eprintln!("W-ALTREF picker — ALTREF-referencing stream byte-exact vpxdec");
    eprintln!("=============================================================");

    // Confirm a leaf actually referenced ALTREF in this stream (otherwise
    // the byte-exact check would pass trivially on a LAST/GOLDEN-only
    // stream).
    let before = RDO_DISPATCH_ALTREF_PICKS.load(Ordering::Relaxed);
    let packets = encode_stream(/* enable_alt_ref = */ true);
    let after = RDO_DISPATCH_ALTREF_PICKS.load(Ordering::Relaxed);
    let picks = after - before;
    eprintln!("ALTREF picks in this stream: {picks}");
    assert!(
        picks >= 1,
        "expected at least one ALTREF-referencing leaf; got {picks}"
    );
    eprintln!("Encoded {} transport packets", packets.len());

    // Confirm at least one non-show ARF + the expected show count at the
    // wire level.
    let mut dec_probe = Decoder::new();
    let mut non_show = 0u32;
    let mut show = 0u32;
    for pkt in &packets {
        for pf in dec_probe.parse_packet(pkt).expect("parse_packet").frames() {
            if pf.header.show_frame {
                show += 1;
            } else {
                non_show += 1;
            }
        }
    }
    eprintln!("wire: {show} show frames, {non_show} non-show (ARF) frames");
    assert!(non_show >= 1, "stream must carry at least one non-show ARF");
    assert_eq!(show, N_FRAMES, "show frames must equal input frames");

    // ---- Wrap in IVF ---------------------------------------------------
    let tmp_dir = std::env::temp_dir();
    let ivf_path = tmp_dir.join("rvp9_w_altref_picker.ivf");
    {
        let mut f = std::fs::File::create(&ivf_path).expect("create ivf");
        let header = IvfHeader::vp9(W as u16, H as u16, 30, 1);
        let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        write_ivf(&mut f, &header, &refs).expect("write_ivf");
    }
    eprintln!("IVF written to {}", ivf_path.display());

    // ---- rvp9-decoder baseline (show frames only) ----------------------
    let rvp9_yuv: Vec<u8> = {
        let mut dec = Decoder::new();
        let mut out: Vec<u8> = Vec::new();
        for pkt in &packets {
            for pf in &decode_packet(&mut dec, pkt) {
                let buf = dec.decode_frame(pf).expect("rvp9 decode_frame");
                if pf.header.show_frame {
                    out.extend_from_slice(&yv12_to_i420_bytes(&buf, W, H));
                }
            }
        }
        out
    };
    let frame_bytes = (W as usize) * (H as usize) * 3 / 2;
    assert_eq!(
        rvp9_yuv.len(),
        frame_bytes * (N_FRAMES as usize),
        "rvp9-decoder must emit exactly {N_FRAMES} visible frames"
    );

    // ---- vpxdec leg ----------------------------------------------------
    let Some(vpxdec) = vpxdec_path() else {
        eprintln!("[SKIPPED] vpxdec not found; ALTREF-picker byte-exact gate is informational.");
        let _ = std::fs::remove_file(&ivf_path);
        return;
    };
    eprintln!("vpxdec available at: {}", vpxdec.display());
    let vpx_yuv_path = tmp_dir.join("rvp9_w_altref_picker_vpxdec.yuv");
    let vpx_yuv = match run_vpxdec(&vpxdec, &ivf_path, &vpx_yuv_path) {
        Ok(b) => b,
        Err(e) => {
            let _ = std::fs::remove_file(&ivf_path);
            panic!(
                "BLOCKER: vpxdec REJECTED the ALTREF-referencing stream: {e}\n\
                 (a show frame naming ALTREF in its ref set is not decodable by libvpx)"
            );
        }
    };

    let verdict = if vpx_yuv.len() != rvp9_yuv.len() {
        format!(
            "FAIL (size mismatch: vpxdec={} bytes, rvp9={} bytes)",
            vpx_yuv.len(),
            rvp9_yuv.len()
        )
    } else if vpx_yuv == rvp9_yuv {
        "BYTE-EXACT".to_string()
    } else {
        let (max_abs, sse) = plane_diff_stats(&rvp9_yuv, &vpx_yuv);
        format!("FAIL (divergent: max_abs={max_abs}, sse={sse})")
    };
    eprintln!("rvp9-decoder vs vpxdec on ALTREF-referencing stream: {verdict}");

    let _ = std::fs::remove_file(&vpx_yuv_path);
    let _ = std::fs::remove_file(&ivf_path);

    assert_eq!(
        verdict, "BYTE-EXACT",
        "ALTREF-referencing stream must be byte-exact decodable by vpxdec; got: {verdict}"
    );
}
