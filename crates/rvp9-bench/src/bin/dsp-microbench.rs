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
    use rvp9_core::inter::filter::{
        convolve8, convolve8_scaled, convolve8_vert, EIGHTTAP_KERNEL,
    };
    use rvp9_core::intra_pred::{d117_predictor, d153_predictor, PredCtx};

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
            convolve8(black_box(&src), stride, black_box(&mut dst), w, filt, filt, w, h);
        });
        let a = bench_min_ns(batches, iters, || {
            convolve8_avx2(black_box(&src), stride, black_box(&mut dst), w, filt, filt, w, h);
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
                    black_box(&src), stride, black_box(&mut dst), w, kernel, 0, step, 0, step, w, h,
                );
            });
            let a = bench_min_ns(batches, iters, || {
                convolve8_scaled_avx2(
                    black_box(&src), stride, black_box(&mut dst), w, kernel, 0, step, 0, step, w, h,
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
        let ctx = PredCtx { above: &above, left: &left, above_left: al };

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
        for i in 0..BITS_PER_BATCH {
            let p = probs[i];
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
            for i in 0..BITS_PER_BATCH {
                let p = probs[i];
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
    println!(
        "  prob mix: 50% p=128, 30% p=32, 20% p=224 (coefficient-decode-shaped)"
    );
}

fn main() {
    let arg = std::env::args().nth(1);
    match arg.as_deref() {
        Some("bool_reader") => run_bool_reader(),
        Some(other) => {
            eprintln!("unknown case '{other}' — valid: bool_reader (no arg = DSP suite)");
            std::process::exit(2);
        }
        None => run(),
    }
}
