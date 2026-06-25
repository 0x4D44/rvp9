//! Cross-process byte-identity oracle + speed harness for the P1-B SAD
//! SIMD work (HLD §8.2 / §8.3). NOT a production binary — an example used
//! by the PR2 validation to (a) prove the AVX2 SAD path moves no encoder
//! decision and (b) measure the foreman fps delta.
//!
//! Usage (single-threaded, RT/CBR, 64 frames of foreman 320×256):
//!
//! ```text
//! cargo run --release --example sad_simd_oracle
//! $env:RVP9_FORCE_SCALAR=1; cargo run --release --example sad_simd_oracle
//! ```
//!
//! It encodes the corpus, concatenates every transport packet's bytes
//! (the same bytes that would be written into the IVF body), and prints
//! the MD5 of that concatenation plus the wall-clock encode time. Two runs
//! (forced-scalar vs auto/AVX2) must print the SAME MD5 — that proves the
//! SAD SIMD produced byte-identical bitstream decisions. The printed
//! elapsed time gives the speed delta.
//!
//! stderr is the report channel; the workspace `print_stderr` lint targets
//! library code, not this example.
#![allow(clippy::print_stderr, clippy::print_stdout)]

use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::y4m::{probe_foreman_cif, read_cropped};
use rvp9_harness::{hex16, md5};
use std::time::Instant;

/// Default coded resolution: foreman cropped to 320×256 (both 64-multiples).
const W: u32 = 320;
const H: u32 = 256;
/// Default frames to encode (HLD §8.2/§8.3 prescribes 64).
const N_FRAMES: usize = 64;
/// CBR target bitrate (HLD: --target-bitrate=800 kbps).
const TARGET_BITRATE_BPS: u32 = 800_000;

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

fn main() {
    let forced_scalar = std::env::var_os("RVP9_FORCE_SCALAR").is_some_and(|v| v == "1");
    let leg = if forced_scalar {
        "FORCED-SCALAR"
    } else {
        "AUTO (AVX2 if present)"
    };

    // Optional second-clip override (PR3 byte-identity gate runs a second
    // clip too): `RVP9_ORACLE_CLIP` + `RVP9_ORACLE_W/H/FRAMES`. Defaults are
    // the foreman 320×256 §8.2/§8.3 operating point.
    let env_u32 = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let w = env_u32("RVP9_ORACLE_W", W);
    let h = env_u32("RVP9_ORACLE_H", H);
    let n_frames = env_u32("RVP9_ORACLE_FRAMES", N_FRAMES as u32) as usize;

    let path = if let Some(clip) = std::env::var_os("RVP9_ORACLE_CLIP") {
        std::path::PathBuf::from(clip)
    } else {
        // Prefer the dedicated 320×256 clip; fall back to cropping foreman_cif.
        let candidates = [
            std::path::PathBuf::from(r"D:\language\rvp9-2\tests\corpus\foreman_320x256_60f.y4m"),
            std::path::PathBuf::from("tests/corpus/foreman_320x256_60f.y4m"),
        ];
        candidates
            .into_iter()
            .find(|p| p.is_file())
            .or_else(probe_foreman_cif)
            .expect("foreman corpus not found")
    };

    // Request up to `n_frames`; readers clamp to the clip length.
    let (frames, framerate) = read_cropped(&path, w, h, n_frames).expect("read frames");
    assert!(!frames.is_empty(), "no frames read");
    let n = frames.len();

    let cfg = EncoderConfig {
        width: w,
        height: h,
        framerate,
        target_bitrate_bps: TARGET_BITRATE_BPS,
        gop_length: (n as u32) + 1,
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
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");

    let mut bitstream: Vec<u8> = Vec::new();
    let t0 = Instant::now();
    for (y, u, v) in &frames {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, w as usize))
            .expect("encode_frame");
        bitstream.extend_from_slice(&pkt.data);
    }
    let elapsed = t0.elapsed();

    let digest = hex16(md5(&bitstream));
    let fps = n as f64 / elapsed.as_secs_f64();

    eprintln!("sad_simd_oracle — leg={leg}");
    eprintln!("  source     : {}", path.display());
    eprintln!("  frames     : {n} @ {w}x{h} ({framerate:.3} fps src)");
    eprintln!("  bytes      : {}", bitstream.len());
    eprintln!("  elapsed    : {:.3} s", elapsed.as_secs_f64());
    eprintln!("  encode fps : {fps:.2}");
    // The MD5 line is the machine-readable oracle output.
    println!("MD5={digest}");
}
