//! DSP kernel micro-benchmark (Wave L L3).
//!
//! Lightweight scalar-vs-AVX2 timing harness for the motion-compensation
//! and directional-intra kernels. Exists to give the Wave L perf-polish
//! items (L4 MC, L5 D117/D153 intra) a measurement oracle: run before a
//! change to capture a baseline, run after to confirm the speedup.
//!
//! Methodology: warm up, then take the **minimum** per-call time over
//! several batches (min is the most stable estimator for a CPU-bound
//! microbench — least perturbed by scheduler interrupts / frequency
//! dips). All inputs and outputs pass through `black_box` so the
//! optimiser cannot hoist or elide the work. Not a statistical
//! benchmark — a repeatable relative ruler.
//!
//! Usage:
//!   cargo run -p rvp9-bench --release --bin dsp-microbench
//!   cargo run -p rvp9-bench --release --bin dsp-microbench -- bool_reader
//!   cargo run -p rvp9-bench --release --bin dsp-microbench -- subpel_variance
//!   cargo run -p rvp9-bench --release --bin dsp-microbench -- variance
//!   cargo run -p rvp9-bench --release --bin dsp-microbench -- quantize_fp
//!
//! The optional positional case-selector runs only the named case:
//!   - (no arg)      → DSP kernel suite (default — scalar vs AVX2).
//!   - `bool_reader` → Track T T-3 baseline: ns/call of
//!     `BoolReader::read` over a 1 MB synthetic stream with a
//!     coefficient-decode-shaped prob mix (50% p=128, 30% p=32, 20% p=224).
//!     The case prints detected `X86Caps` so the number is anchored to
//!     hardware.

use std::hint::black_box;
use std::time::Instant;

/// Minimum ns/call over `batches` batches of `iters` calls each, after a
/// warmup batch. `f` is called `iters` times per timed batch.
fn bench_min_ns<F: FnMut()>(batches: u32, iters: u32, mut f: F) -> f64 {
    // Warmup.
    for _ in 0..iters {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..batches {
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        let ns = t0.elapsed().as_nanos() as f64 / f64::from(iters);
        if ns < best {
            best = ns;
        }
    }
    best
}

fn lcg(state: &mut u32) -> u8 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state >> 24) as u8
}

fn filled(len: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    (0..len).map(|_| lcg(&mut s)).collect()
}

#[cfg(target_arch = "x86_64")]
fn report(name: &str, scalar_ns: f64, avx2_ns: f64) {
    let speedup = scalar_ns / avx2_ns;
    println!("{name:<34} scalar {scalar_ns:8.1} ns   avx2 {avx2_ns:8.1} ns   {speedup:5.2}x");
}

#[cfg(target_arch = "x86_64")]
fn run() {
    use rvp9_core::dsp::x86_64::{
        convolve8_avx2, convolve8_scaled_avx2, convolve8_vert_avx2, d117_predictor_avx2,
        d153_predictor_avx2,
    };
    use rvp9_core::inter::filter::{EIGHTTAP_KERNEL, convolve8, convolve8_scaled, convolve8_vert};
    use rvp9_core::intra_pred::{PredCtx, d117_predictor, d153_predictor};

    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("AVX2 not detected on this host — nothing to compare.");
        return;
    }

    let filt = &EIGHTTAP_KERNEL[8]; // half-pel (worst-case blur)
    let batches = 200;

    println!("== Motion compensation (8-tap) ==");
    for &(w, h) in &[(16usize, 16usize), (32, 32), (64, 64)] {
        let src = filled((h + 7) * (w + 8) + 64, 0x1111 ^ (w as u32));
        let stride = w + 8;
        let mut dst = vec![0u8; h * w + 64];
        let iters = if w >= 64 { 2000 } else { 8000 };

        let s = bench_min_ns(batches, iters, || {
            convolve8_vert(black_box(&src), stride, black_box(&mut dst), w, filt, w, h);
        });
        let a = bench_min_ns(batches, iters, || {
            convolve8_vert_avx2(black_box(&src), stride, black_box(&mut dst), w, filt, w, h);
        });
        report(&format!("convolve8_vert {w}x{h}"), s, a);

        let s = bench_min_ns(batches, iters, || {
            convolve8(
                black_box(&src),
                stride,
                black_box(&mut dst),
                w,
                filt,
                filt,
                w,
                h,
            );
        });
        let a = bench_min_ns(batches, iters, || {
            convolve8_avx2(
                black_box(&src),
                stride,
                black_box(&mut dst),
                w,
                filt,
                filt,
                w,
                h,
            );
        });
        report(&format!("convolve8 (2D) {w}x{h}"), s, a);
    }

    println!("== Scaled MC (Wave L L1) ==");
    {
        let kernel = &EIGHTTAP_KERNEL;
        let step = 20i32; // down-scale
        for &(w, h) in &[(16usize, 16usize), (32, 32), (64, 64)] {
            let ih = (((h as i32 - 1) * step) >> 4) as usize + 8;
            let src_w = (((w as i32 - 1) * step) >> 4) as usize + 8;
            let stride = src_w + 4;
            let src = filled((ih) * stride + src_w + 64, 0x2222 ^ (w as u32));
            let mut dst = vec![0u8; h * w + 64];
            let iters = if w >= 64 { 1500 } else { 6000 };
            let s = bench_min_ns(batches, iters, || {
                convolve8_scaled(
                    black_box(&src),
                    stride,
                    black_box(&mut dst),
                    w,
                    kernel,
                    0,
                    step,
                    0,
                    step,
                    w,
                    h,
                );
            });
            let a = bench_min_ns(batches, iters, || {
                convolve8_scaled_avx2(
                    black_box(&src),
                    stride,
                    black_box(&mut dst),
                    w,
                    kernel,
                    0,
                    step,
                    0,
                    step,
                    w,
                    h,
                );
            });
            report(&format!("convolve8_scaled {w}x{h}"), s, a);
        }
    }

    println!("== Directional intra (D117 / D153) ==");
    for &bs in &[16usize, 32] {
        let above = filled(2 * bs + 1, 0x3333 ^ bs as u32);
        let left = filled(2 * bs + 1, 0x4444 ^ bs as u32);
        let al = above[0] ^ left[0];
        let stride = bs + 4;
        let mut dst = vec![0u8; bs * stride + 64];
        let iters = 8000;
        let ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: al,
        };

        let s = bench_min_ns(batches, iters, || {
            d117_predictor(black_box(&mut dst), stride, bs, black_box(ctx));
        });
        let a = bench_min_ns(batches, iters, || {
            d117_predictor_avx2(black_box(&mut dst), stride, bs, &above, &left, al);
        });
        report(&format!("d117_predictor {bs}x{bs}"), s, a);

        let s = bench_min_ns(batches, iters, || {
            d153_predictor(black_box(&mut dst), stride, bs, black_box(ctx));
        });
        let a = bench_min_ns(batches, iters, || {
            d153_predictor_avx2(black_box(&mut dst), stride, bs, &above, &left, al);
        });
        report(&format!("d153_predictor {bs}x{bs}"), s, a);
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn run() {
    eprintln!("dsp-microbench is x86_64-only (compares scalar vs AVX2).");
}

/// Track T T-3 measurement slice.
///
/// Times `BoolReader::read` over a deterministic 1 MB encoded stream
/// shaped to mirror coefficient-decode prob distribution
/// (50% p=128, 30% p=32, 20% p=224). Reports the median ns/call across
/// `RUNS` independent runs and the implied per-call Mbits/sec.
///
/// **Bounds of the measurement.** This is a relative ruler, not a
/// statistical benchmark. Per the T-3 HLD it exists to anchor the
/// baseline against which any future Phase-2 attempt (BMI2 / inline-asm
/// on `read`) must clear. The HLD recommends NOT pursuing such a
/// change unless the microbench shows a positive, repeatable delta
/// large enough to escape the bench-noise floor. This binary publishes
/// the floor itself; it does not (and must not) modify `read`.
fn run_bool_reader() {
    use rvp9_core::entropy::{BoolReader, BoolWriter};

    // 1) Detect + print caps, so the published number is hardware-anchored.
    #[cfg(target_arch = "x86_64")]
    {
        use rvp9_core::dsp::caps::X86Caps;
        let caps = X86Caps::detect();
        println!("== Track T T-3: BoolReader::read baseline ==");
        println!(
            "X86Caps {{ sse2={}, ssse3={}, sse4_1={}, avx2={}, bmi2={} }}",
            caps.sse2, caps.ssse3, caps.sse4_1, caps.avx2, caps.bmi2
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("== Track T T-3: BoolReader::read baseline ==");
        println!("(non-x86_64 host — no X86Caps to report)");
    }

    // 2) Build the canonical prob mix as a fixed table indexable by an
    //    LCG state. 100-slot table: 50 × p=128, 30 × p=32, 20 × p=224.
    //    Reading slot `(lcg_state >> 24) % 100` gives the right
    //    long-run distribution without any branchy if/else.
    const PROB_MIX_LEN: usize = 100;
    let mut prob_mix = [0u8; PROB_MIX_LEN];
    for (i, p) in prob_mix.iter_mut().enumerate() {
        *p = if i < 50 {
            128
        } else if i < 80 {
            32
        } else {
            224
        };
    }

    // 3) Generate the deterministic (bit, prob) sequence and encode it.
    //    Target encoded buffer ≥ 1 MB. We over-shoot by ~20% so the
    //    decoder can comfortably read N bits per batch without crossing
    //    EOF (which would invoke `fill_slow` and skew the measurement).
    const TARGET_ENCODED_BYTES: usize = 1 << 20; // 1 MB.
    // Rough estimate: at the canonical prob mix the average bit consumes
    // ~0.6 bits of entropy, so the encoder packs ~10–13 source bits per
    // output byte. To guarantee ≥1 MB encoded we pre-generate 24 Mbit of
    // source bits (≈3 MB worth at 1 bit/byte upper bound). Then we trim
    // the decoded-bits count to whatever the encoder actually wrote.
    const SOURCE_BITS: usize = 24 * 1024 * 1024;
    let mut probs: Vec<u8> = Vec::with_capacity(SOURCE_BITS);
    let mut bits: Vec<u8> = Vec::with_capacity(SOURCE_BITS);
    let mut bit_seed: u32 = 0xC0FF_EE01;
    let mut prob_seed: u32 = 0xDEAD_BEEF;
    for _ in 0..SOURCE_BITS {
        // Pick a probability slot deterministically.
        let p = prob_mix[lcg(&mut prob_seed) as usize % PROB_MIX_LEN];
        // Draw a bit biased by `p` (probability the bit is 0).
        // `p` is the prob-of-zero in libvpx convention; treat the LCG
        // byte as a uniform u8 and compare.
        let b: u8 = if lcg(&mut bit_seed) < p { 0 } else { 1 };
        probs.push(p);
        bits.push(b);
    }

    // Reserve a generous encode buffer (worst case 1 bit/byte).
    let mut enc_buf = vec![0u8; SOURCE_BITS / 4 + 1024];
    let encoded_len = {
        let mut w = BoolWriter::start_encode(&mut enc_buf);
        for (&b, &p) in bits.iter().zip(probs.iter()) {
            w.write(b, p);
        }
        w.stop_encode().expect("BoolWriter stop_encode failed")
    };
    let encoded = &enc_buf[..encoded_len];
    assert!(
        encoded.len() >= TARGET_ENCODED_BYTES,
        "encoded stream {} bytes < 1 MB target — increase SOURCE_BITS",
        encoded.len()
    );

    // 4) Bench. Each batch creates a fresh BoolReader and reads
    //    `BITS_PER_BATCH` bits. We pick a count safely below `bits.len()`
    //    so we never cross EOF.
    const BITS_PER_BATCH: usize = 1_000_000;
    assert!(
        BITS_PER_BATCH < bits.len(),
        "BITS_PER_BATCH must be < source bit count to avoid EOF"
    );
    const BATCHES: u32 = 7;
    const RUNS: u32 = 5;

    // Warmup: one full batch, discarded.
    {
        let mut r = BoolReader::new(encoded).expect("marker bit");
        for &p in probs.iter().take(BITS_PER_BATCH) {
            let _ = black_box(r.read(black_box(p)));
        }
        assert!(!r.has_error(), "warmup over-read");
    }

    // Per the comment on `bench_min_ns`, `min` is the most stable
    // estimator for a CPU-bound microbench. We additionally take the
    // median across RUNS independent invocations so the headline number
    // isn't dominated by a single optimistic outlier; min/max bound the
    // dispersion.
    let mut run_ns: Vec<f64> = Vec::with_capacity(RUNS as usize);
    for _ in 0..RUNS {
        let mut best = f64::INFINITY;
        for _ in 0..BATCHES {
            let mut r = BoolReader::new(encoded).expect("marker bit");
            let t0 = Instant::now();
            for &p in probs.iter().take(BITS_PER_BATCH) {
                let _ = black_box(r.read(black_box(p)));
            }
            let ns = t0.elapsed().as_nanos() as f64 / BITS_PER_BATCH as f64;
            assert!(!r.has_error(), "bench batch over-read");
            if ns < best {
                best = ns;
            }
        }
        run_ns.push(best);
    }
    run_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = run_ns[run_ns.len() / 2];
    let min = *run_ns.first().unwrap();
    let max = *run_ns.last().unwrap();
    let mbits_per_sec = 1_000.0 / median;

    println!(
        "bool_reader_read    median {median:6.2} ns/call   min {min:6.2}   max {max:6.2}   ({mbits_per_sec:7.1} Mbits/s @ median)"
    );
    println!(
        "  source: encoded {} bytes (1 MB target), {BITS_PER_BATCH} bits/batch × {BATCHES} batches × {RUNS} runs",
        encoded.len()
    );
    println!("  prob mix: 50% p=128, 30% p=32, 20% p=224 (coefficient-decode-shaped)");
}

fn main() {
    let arg = std::env::args().nth(1);
    match arg.as_deref() {
        Some("bool_reader") => run_bool_reader(),
        Some("subpel_variance") => run_subpel_variance(),
        Some("variance") => run_variance(),
        Some("quantize_fp") => run_quantize_fp(),
        Some("vs_libvpx") => run_vs_libvpx(),
        Some(other) => {
            eprintln!(
                "unknown case '{other}' — valid: bool_reader, subpel_variance, variance, \
                 quantize_fp, vs_libvpx \
                 (no arg = DSP suite)"
            );
            std::process::exit(2);
        }
        None => run(),
    }
}

/// Production-dispatch ruler for the realtime fast quantizer.
fn run_quantize_fp() {
    use rvp9_core::quant::forward::{
        QuantizerDeltas, init_plane_quantizers, vp9_quantize_fp, vp9_quantize_fp_with_order,
    };
    use rvp9_core::transforms::scans::scan_for;
    use rvp9_core::transforms::{TxSize, TxType};

    let qp = init_plane_quantizers(120, QuantizerDeltas::Y);
    let batches = 50;

    println!("== Fast quantizer ==");
    for &(tx_size, iters) in &[
        (TxSize::Tx4x4, 20_000),
        (TxSize::Tx8x8, 10_000),
        (TxSize::Tx16x16, 4_000),
    ] {
        let n = tx_size.n_coeffs();
        let order = scan_for(tx_size, TxType::DctDct);
        let coeff: Vec<i32> = filled(n, 0x51A7 ^ n as u32)
            .into_iter()
            .map(|value| i32::from(value) - 128)
            .collect();
        let mut scalar_q = vec![0i16; n];
        let mut scalar_dq = vec![0i16; n];
        let mut dispatch_q = vec![0i16; n];
        let mut dispatch_dq = vec![0i16; n];

        let scalar_ns = bench_min_ns(batches, iters, || {
            black_box(vp9_quantize_fp(
                black_box(&coeff),
                n,
                false,
                &qp.round_fp,
                &qp.quant_fp,
                black_box(&mut scalar_q),
                black_box(&mut scalar_dq),
                &qp.dequant,
                order.scan,
            ));
        });
        let dispatch_ns = bench_min_ns(batches, iters, || {
            black_box(vp9_quantize_fp_with_order(
                black_box(&coeff),
                n,
                false,
                &qp.round_fp,
                &qp.quant_fp,
                black_box(&mut dispatch_q),
                black_box(&mut dispatch_dq),
                &qp.dequant,
                &order,
            ));
        });
        println!(
            "{:>2}x{:<2} scalar {scalar_ns:8.1} ns   dispatch {dispatch_ns:8.1} ns   {:5.2}x",
            tx_size.side(),
            tx_size.side(),
            scalar_ns / dispatch_ns
        );
    }
}

/// Production-dispatch ruler for the variance reductions in inter RD.
fn run_variance() {
    use rvp9_core::block::BlockSize;
    use rvp9_core::dsp::distortion::var_sum_sse;

    let dsp = rvp9_core::dsp::init();
    let batches = 50;

    println!("== Variance reduction ==");
    for &(bsize, width, height, iters) in &[
        (BlockSize::Block8x8, 8usize, 8usize, 20_000),
        (BlockSize::Block16x16, 16, 16, 10_000),
        (BlockSize::Block32x32, 32, 32, 4_000),
        (BlockSize::Block64x64, 64, 64, 1_000),
    ] {
        let src_stride = width + 17;
        let pred_stride = width + 11;
        let src = filled((height - 1) * src_stride + width, 0x5EED ^ width as u32);
        let pred = filled((height - 1) * pred_stride + width, 0xC0DE ^ height as u32);
        let dispatched = dsp.variance(bsize);

        let scalar_ns = bench_min_ns(batches, iters, || {
            black_box(var_sum_sse(
                black_box(&src),
                src_stride,
                black_box(&pred),
                pred_stride,
                width,
                height,
            ));
        });
        let dispatch_ns = bench_min_ns(batches, iters, || {
            black_box(dispatched(
                black_box(&src),
                src_stride,
                black_box(&pred),
                pred_stride,
            ));
        });
        println!(
            "{width:>2}x{height:<2} scalar {scalar_ns:8.1} ns   \
             dispatch {dispatch_ns:8.1} ns   {:5.2}x",
            scalar_ns / dispatch_ns
        );
    }
}

/// Production-dispatch ruler for the sub-pixel-variance kernel that drives
/// sub-pel MV refinement. The scalar result is timed separately so an ISA
/// implementation can be compared on the same deterministic buffers.
fn run_subpel_variance() {
    use rvp9_core::block::BlockSize;
    use rvp9_core::dsp::distortion::sub_pixel_variance;

    let dsp = rvp9_core::dsp::init();
    let batches = 50;

    println!("== Sub-pixel variance ==");
    for &(bsize, w, h, iters) in &[
        (BlockSize::Block16x16, 16usize, 16usize, 8_000),
        (BlockSize::Block32x32, 32, 32, 3_000),
        (BlockSize::Block64x64, 64, 64, 1_000),
    ] {
        let src_stride = w + 17;
        let ref_stride = w + 11;
        let src = filled((h + 1) * src_stride + w + 17, 0x5A17 ^ w as u32);
        let reference = filled((h - 1) * ref_stride + w, 0xC0DE ^ h as u32);
        let dispatched = dsp
            .subpel_variance(bsize)
            .expect("bench shape has a sub-pixel-variance kernel");

        for &(xoff, yoff) in &[(0usize, 0usize), (4, 4), (3, 5)] {
            let scalar_ns = bench_min_ns(batches, iters, || {
                black_box(sub_pixel_variance(
                    black_box(&src),
                    src_stride,
                    black_box(&reference),
                    ref_stride,
                    xoff,
                    yoff,
                    w,
                    h,
                ));
            });
            let dispatch_ns = bench_min_ns(batches, iters, || {
                black_box(dispatched(
                    black_box(&src),
                    src_stride,
                    xoff,
                    yoff,
                    black_box(&reference),
                    ref_stride,
                ));
            });
            println!(
                "{w:>2}x{h:<2} ({xoff},{yoff}) scalar {scalar_ns:8.1} ns   \
                 dispatch {dispatch_ns:8.1} ns   {:5.2}x",
                scalar_ns / dispatch_ns
            );
        }
    }
}

/// Head-to-head DSP kernel bench against libvpx (decoder kernel audit,
/// 2026-05-26).
///
/// Companion to `tools/libvpx_kernel_bench/main.c`. Generates identical
/// inputs via the same LCG seeds, calls the matching rvp9 kernel through
/// the production `Dsp` dispatch table (so we measure what the decoder
/// actually runs, including any ISA-tier gap), and prints CSV in the same
/// shape as the C harness:
///
///     side,kernel,size,ns_per_iter,input_fnv64,output_fnv64
///
/// The audit driver joins on `(kernel, size)`, asserts the input/output
/// FNV-1a 64-bit hashes match (bit-exactness), then prints the ratio.
fn run_vs_libvpx() {
    // Process-wide dispatch table — same one decode_frame touches.
    // `init()` is idempotent and runs caps detection once.
    let dsp = rvp9_core::dsp::init();
    let batches: u32 = 200;

    // IDCT family. Seeds + iters mirror tools/libvpx_kernel_bench/main.c
    // exactly so both rigs feed each kernel byte-identical coefficients.
    bench_idct_case(
        dsp.idct4x4_add(),
        "idct4x4_16_add",
        "4x4",
        4,
        0xAA10,
        16000,
        batches,
    );
    bench_idct_case(
        dsp.idct8x8_add(),
        "idct8x8_64_add",
        "8x8",
        8,
        0xAA11,
        12000,
        batches,
    );
    bench_idct_case(
        dsp.idct8x8_12_add(),
        "idct8x8_12_add",
        "8x8",
        8,
        0xAA12,
        12000,
        batches,
    );
    bench_idct_case(
        dsp.idct16x16_add(),
        "idct16x16_256_add",
        "16x16",
        16,
        0xAA13,
        8000,
        batches,
    );
    bench_idct_case(
        dsp.idct16x16_10_add(),
        "idct16x16_10_add",
        "16x16",
        16,
        0xAA14,
        8000,
        batches,
    );
    bench_idct_case(
        dsp.idct16x16_38_add(),
        "idct16x16_38_add",
        "16x16",
        16,
        0xAA01,
        8000,
        batches,
    );
    bench_idct_case(
        dsp.idct32x32_add(),
        "idct32x32_1024_add",
        "32x32",
        32,
        0xAA15,
        2000,
        batches,
    );
    bench_idct_case(
        dsp.idct32x32_34_add(),
        "idct32x32_34_add",
        "32x32",
        32,
        0xAA16,
        2000,
        batches,
    );
    bench_idct_case(
        dsp.idct32x32_135_add(),
        "idct32x32_135_add",
        "32x32",
        32,
        0xAA02,
        2000,
        batches,
    );

    // Convolve8 family. mode: 0 = 2D, 1 = horiz-only, 2 = vert-only.
    bench_conv_case(ConvCase {
        name: "convolve8_2d_halfpel",
        size: "8x8",
        w: 8,
        h: 8,
        mode: 0,
        seed: 0xAB08,
        iters: 16000,
        batches,
    });
    bench_conv_case(ConvCase {
        name: "convolve8_2d_halfpel",
        size: "16x16",
        w: 16,
        h: 16,
        mode: 0,
        seed: 0xAB10,
        iters: 8000,
        batches,
    });
    bench_conv_case(ConvCase {
        name: "convolve8_2d_halfpel",
        size: "32x32",
        w: 32,
        h: 32,
        mode: 0,
        seed: 0xAB20,
        iters: 3000,
        batches,
    });
    bench_conv_case(ConvCase {
        name: "convolve8_2d_halfpel",
        size: "64x64",
        w: 64,
        h: 64,
        mode: 0,
        seed: 0xAA03,
        iters: 2000,
        batches,
    });
    bench_conv_case(ConvCase {
        name: "convolve8_horiz_halfpel",
        size: "16x16",
        w: 16,
        h: 16,
        mode: 1,
        seed: 0xAC10,
        iters: 12000,
        batches,
    });
    bench_conv_case(ConvCase {
        name: "convolve8_horiz_halfpel",
        size: "64x64",
        w: 64,
        h: 64,
        mode: 1,
        seed: 0xAC40,
        iters: 3000,
        batches,
    });
    bench_conv_case(ConvCase {
        name: "convolve8_vert_halfpel",
        size: "16x16",
        w: 16,
        h: 16,
        mode: 2,
        seed: 0xAD10,
        iters: 12000,
        batches,
    });
    bench_conv_case(ConvCase {
        name: "convolve8_vert_halfpel",
        size: "64x64",
        w: 64,
        h: 64,
        mode: 2,
        seed: 0xAD40,
        iters: 3000,
        batches,
    });

    // Loopfilter (horizontal) family. getter_kind: 0 = lpf_horizontal_8,
    // 1 = lpf_horizontal_16, 2 = lpf_horizontal_4. Seeds + iters mirror
    // tools/libvpx_kernel_bench/main.c exactly so both rigs feed each
    // kernel byte-identical pixels and thresholds (blimit/limit/thresh =
    // 24/12/8, edge at row 16 of a 32×32 buffer).
    bench_lpf_case(&dsp, "lpf_horizontal_8", 0, 0xC008, 12000, batches);
    bench_lpf_case(&dsp, "lpf_horizontal_16", 1, 0xC016, 8000, batches);
    bench_lpf_case(&dsp, "lpf_horizontal_4", 2, 0xC004, 12000, batches);

    // Bool-reader (VP9 arithmetic decoder) head-to-head. Identical 16384-byte
    // LCG buffer (seed 0xB001) + 8000-prob sequence (seed 0xB002, clamped to
    // 1..=255). Times the full {re-init reader; 8000 read() calls} closure;
    // ns is reported per whole 8000-read closure (NOT per bit) so both sides'
    // numbers stay directly comparable. See tools/libvpx_kernel_bench/main.c.
    bench_boolreader_case(0xB001, 0xB002, 8000, batches);
}

/// Bench one IDCT-add kernel against its libvpx twin. `N` (coefficient
/// count) is inferred from the dispatch fn-pointer type; `dim` is the
/// block dimension. Hashes the `dim×dim` footprint (no stride padding)
/// so the C side can match without agreeing on padding layout.
fn bench_idct_case<const N: usize>(
    f: fn(&[i32; N], &mut [u8], usize),
    name: &str,
    size: &str,
    dim: usize,
    seed: u32,
    iters: u32,
    batches: u32,
) {
    let mut input = [0i32; N];
    fill_idct_input_seeded(&mut input, seed);
    let in_hash = fnv1a64_i32(&input);

    let stride = dim + 16; // same on the libvpx side
    let mut dest = vec![0u8; dim * stride + 64];

    // Verify pass: one call from a zero dest, hash the dim×dim footprint.
    f(&input, &mut dest, stride);
    let mut packed = vec![0u8; dim * dim];
    for r in 0..dim {
        packed[r * dim..r * dim + dim].copy_from_slice(&dest[r * stride..r * stride + dim]);
    }
    let out_hash = fnv1a64_bytes(&packed);

    dest.iter_mut().for_each(|b| *b = 0);
    let ns = bench_min_ns(batches, iters, || {
        f(black_box(&input), black_box(&mut dest), stride);
    });
    println!("rvp9,{name},{size},{ns:.2},{in_hash:016x},{out_hash:016x}");
}

/// Bench one convolve8 case against its libvpx twin. `mode`: 0 = 2D,
/// 1 = horiz-only, 2 = vert-only. The half-pel coefficients come from
/// `EIGHTTAP_KERNEL[8]` (= libvpx `vp9_sub_pel_filters_8[8]`). rvp9's
/// `src` argument is the filter footprint top-left for every mode; the
/// libvpx side offsets its `src` per-mode so its internal shift lands on
/// the same byte (see main.c `conv_thunk`).
struct ConvCase<'a> {
    name: &'a str,
    size: &'a str,
    w: usize,
    h: usize,
    mode: u8,
    seed: u32,
    iters: u32,
    batches: u32,
}

fn bench_conv_case(case: ConvCase<'_>) {
    use rvp9_core::inter::filter::EIGHTTAP_KERNEL;

    let ConvCase {
        name,
        size,
        w,
        h,
        mode,
        seed,
        iters,
        batches,
    } = case;

    let src_stride: usize = 80;
    let dst_stride: usize = 64;
    let mut src_buf = vec![0u8; 80 * 80];
    fill_bytes_seeded(&mut src_buf, seed);
    let in_hash = fnv1a64_bytes(&src_buf);
    let filt = &EIGHTTAP_KERNEL[8];
    let mut dst = vec![0u8; 64 * 64];

    let do_conv = |dst: &mut [u8]| {
        #[cfg(target_arch = "x86_64")]
        {
            use rvp9_core::dsp::x86_64::{
                convolve8_avx2, convolve8_horiz_avx2, convolve8_vert_avx2,
            };
            match mode {
                1 => convolve8_horiz_avx2(&src_buf, src_stride, dst, dst_stride, filt, w, h),
                2 => convolve8_vert_avx2(&src_buf, src_stride, dst, dst_stride, filt, w, h),
                _ => convolve8_avx2(&src_buf, src_stride, dst, dst_stride, filt, filt, w, h),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            use rvp9_core::inter::filter::{convolve8, convolve8_horiz, convolve8_vert};
            match mode {
                1 => convolve8_horiz(&src_buf, src_stride, dst, dst_stride, filt, w, h),
                2 => convolve8_vert(&src_buf, src_stride, dst, dst_stride, filt, w, h),
                _ => convolve8(&src_buf, src_stride, dst, dst_stride, filt, filt, w, h),
            }
        }
    };

    do_conv(dst.as_mut_slice());
    let mut packed = vec![0u8; w * h];
    for r in 0..h {
        packed[r * w..r * w + w].copy_from_slice(&dst[r * dst_stride..r * dst_stride + w]);
    }
    let out_hash = fnv1a64_bytes(&packed);

    dst.iter_mut().for_each(|b| *b = 0);
    let ns = bench_min_ns(batches, iters, || {
        do_conv(black_box(dst.as_mut_slice()));
    });
    println!("rvp9,{name},{size},{ns:.2},{in_hash:016x},{out_hash:016x}");
}

/// Bench one horizontal loopfilter kernel against its libvpx twin.
/// `getter_kind` selects the dispatch fn-pointer: 0 = `lpf_horizontal_8`,
/// 1 = `lpf_horizontal_16`, 2 = `lpf_horizontal_4`. All three share the
/// signature `fn(&mut [u8], usize, usize, u8, u8, u8)` =
/// (pixels, s_index, pitch, blimit, limit, thresh).
///
/// Geometry mirrors tools/libvpx_kernel_bench/main.c exactly: a 32×32
/// (pitch = 32, height = 32) 1024-byte buffer `orig` filled by the shared
/// LCG, the filtered horizontal edge at `s_index = 16 * 32 = 512` (row 16),
/// and thresholds blimit/limit/thresh = 24/12/8. The libvpx side offsets
/// its pointer by 512 and passes length-16 threshold arrays so its window
/// lands on the same bytes. Each timed call copies `orig` into a scratch
/// `work` buffer then filters in place; the copy is timed on both sides so
/// it cancels in the ratio.
fn bench_lpf_case(
    dsp: &rvp9_core::dsp::SimdDispatch,
    name: &str,
    getter_kind: u8,
    seed: u32,
    iters: u32,
    batches: u32,
) {
    let pitch: usize = 32;
    let rows: usize = 32;
    let s_index: usize = 16 * 32; // edge at row 16
    let (blimit, limit, thresh): (u8, u8, u8) = (24, 12, 8);

    let f: fn(&mut [u8], usize, usize, u8, u8, u8) = match getter_kind {
        0 => dsp.lpf_horizontal_8(),
        1 => dsp.lpf_horizontal_16(),
        _ => dsp.lpf_horizontal_4(),
    };

    let mut orig = vec![0u8; pitch * rows];
    fill_bytes_seeded(&mut orig, seed);
    let in_hash = fnv1a64_bytes(&orig);

    // Verify pass: copy a fresh buffer, one filter call, hash the full 1024.
    let mut verify = orig.clone();
    f(&mut verify, s_index, pitch, blimit, limit, thresh);
    let out_hash = fnv1a64_bytes(&verify);

    let mut work = vec![0u8; pitch * rows];
    let ns = bench_min_ns(batches, iters, || {
        // Reset each call so the in-place kernel always sees identical
        // input (the copy is in the libvpx timing too, so it cancels).
        work.copy_from_slice(&orig);
        f(black_box(&mut work), s_index, pitch, blimit, limit, thresh);
    });
    println!("rvp9,{name},edge,{ns:.2},{in_hash:016x},{out_hash:016x}");
}

/// Bench the VP9 arithmetic decoder (`BoolReader::read`) against its libvpx
/// twin (`vpx_read`). The apples-to-apples contract (Rust and C sides build
/// these byte-identically):
///
///   - BUF: 16384-byte buffer from the shared LCG seeded `seed_buf` (0xB001),
///     with the top bit of `BUF[0]` cleared (`BUF[0] &= 0x7F`) — see below.
///   - PROBS: `n` (=8000) u8 probabilities from the shared LCG seeded
///     `seed_probs` (0xB002), each clamped to 1..=255 (raw 0 → 1).
///   - Reader is initialised over BUF; `n` reads are performed.
///   - `stream_hash` folds every decoded bit with the FNV-ish recurrence
///     `h = h.wrapping_mul(1099511628211) ^ (bit as u64)` (start
///     0x14650FB0739D0383 = 1469598103934665603).
///   - `probs_hash` = FNV-1a-64 over the `n` PROBS bytes.
///
/// `stream_hash` and `probs_hash` MUST be identical rvp9-vs-libvpx (same
/// decoder, same inputs). Timing covers the full {re-init reader; `n` reads}
/// closure; `ns` is reported PER WHOLE CLOSURE (NOT divided by `n`) so both
/// sides' numbers stay directly comparable. The re-init is identical-cheap on
/// both sides, so it cancels in the ratio.
fn bench_boolreader_case(seed_buf: u32, seed_probs: u32, n: usize, batches: u32) {
    use rvp9_core::entropy::BoolReader;

    // BUF: 16384 bytes from the shared LCG (seed_buf).
    let mut buf = vec![0u8; 16384];
    fill_bytes_seeded(&mut buf, seed_buf);
    // Marker-bit normalisation. `BoolReader::new` (and libvpx `vpx_reader_init`)
    // decode one leading bit at prob 128 that must be 0; the Rust port turns a
    // 1 into a hard `Err(InvalidMarkerBit)`. A raw LCG fill at seed 0xB001 puts
    // a 1 in BUF[0]'s top bit, so clearing it makes the stream valid. The C
    // side MUST mirror this single op (`BUF[0] &= 0x7F`) so both decoders see a
    // byte-identical buffer and produce the same stream_hash.
    buf[0] &= 0x7F;

    // PROBS: `n` bytes from a second LCG (seed_probs), clamped to 1..=255.
    let mut probs = vec![0u8; n];
    fill_bytes_seeded(&mut probs, seed_probs);
    for p in probs.iter_mut() {
        if *p == 0 {
            *p = 1;
        }
    }
    let probs_hash = fnv1a64_bytes(&probs);

    // Closure: re-init the reader and decode `n` bits, folding into the
    // FNV-ish stream hash. Returned so the optimiser can't elide the reads.
    let decode_stream = || -> u64 {
        let mut r = BoolReader::new(&buf).expect("marker bit");
        let mut h: u64 = 1469598103934665603;
        for &p in probs.iter().take(n) {
            let bit = r.read(black_box(p));
            h = h.wrapping_mul(1099511628211) ^ u64::from(bit);
        }
        h
    };

    // Verify pass: capture the decoded-bit-stream hash (matched against libvpx).
    let stream_hash = decode_stream();

    // Time the full {re-init; n reads} closure. bench_min_ns calls `f` once
    // per "call", so each call is one whole 8000-read closure; ns is therefore
    // already per-whole-closure — do NOT divide by `n`.
    let ns = bench_min_ns(batches, 1, || {
        black_box(decode_stream());
    });
    println!("rvp9,boolreader_read,{n}reads,{ns:.2},{probs_hash:016x},{stream_hash:016x}");
}

// LCG mirrored bit-for-bit with the C harness:
//   state = state * 1664525 + 1013904223
//   byte  = (state >> 24) & 0xFF
fn lcg_step_byte(state: &mut u32) -> u8 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*state >> 24) & 0xFF) as u8
}

fn fill_idct_input_seeded(out: &mut [i32], seed: u32) {
    let mut s = seed;
    for v in out.iter_mut() {
        let lo = u32::from(lcg_step_byte(&mut s));
        let hi = u32::from(lcg_step_byte(&mut s));
        let bits = (lo | (hi << 8)) & 0x3FF;
        *v = bits as i32 - 0x200;
    }
}

fn fill_bytes_seeded(out: &mut [u8], seed: u32) {
    let mut s = seed;
    for b in out.iter_mut() {
        *b = lcg_step_byte(&mut s);
    }
}

fn fnv1a64_bytes(buf: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in buf {
        h ^= u64::from(b);
        // FNV-1a 64-bit prime = 0x100000001B3 = 2^40 + 0x1B3 — note the
        // 2^40 not 2^32. An earlier draft of this file had 0x10000_01B3
        // (2^32 + 0x1B3), which left the lower 32 bits of the hash
        // matching the C side (correct prime) while garbling the upper —
        // a multiplier-truncation bug with a very distinctive
        // signature.
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// Hash a `[i32]` as the underlying little-endian byte sequence — must
/// match the C harness's `fnv1a64(input, sizeof input)` byte view.
fn fnv1a64_i32(buf: &[i32]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &v in buf {
        let bytes = v.to_le_bytes();
        for &b in &bytes {
            h ^= u64::from(b);
            // FNV-1a 64-bit prime = 0x100000001B3 = 2^40 + 0x1B3 — note the
            // 2^40 not 2^32. An earlier draft of this file had 0x10000_01B3
            // (2^32 + 0x1B3), which left the lower 32 bits of the hash
            // matching the C side (correct prime) while garbling the upper —
            // a multiplier-truncation bug with a very distinctive
            // signature.
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}
