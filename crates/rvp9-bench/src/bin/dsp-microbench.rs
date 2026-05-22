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
//! Usage: `cargo run -p rvp9-bench --release --bin dsp-microbench`

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

fn main() {
    run();
}
