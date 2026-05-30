// Standalone upper-bound microbench for the per-frame frame-buffer zeroing cost.
// Measures alloc+zero (vec![0u8;n]) and alloc+zero+full-overwrite for a 720p
// 4:2:0 decoder buffer, plus a "pool-recycle" model (reuse one Box, memset each
// iter) and a "pool-recycle, no-rezero" model (just overwrite). min-of-N, ns/iter.
//
// Build: rustc -O notes/zerobench.rs -o C:/Temp/zerobench.exe
// Run:   C:/Temp/zerobench.exe
use std::time::Instant;

#[inline(never)]
fn alloc_zero(n: usize) -> Box<[u8]> {
    // exactly what frame.rs:265 does
    vec![0u8; n].into_boxed_slice()
}

#[inline(never)]
fn touch(buf: &mut [u8], v: u8) {
    // simulate reconstruction overwriting every byte (sequential write)
    for b in buf.iter_mut() {
        *b = v;
    }
}

fn black_box<T>(x: T) -> T {
    unsafe {
        let r = std::ptr::read_volatile(&x);
        std::mem::forget(x);
        r
    }
}

fn bench<F: FnMut() -> u64>(name: &str, iters: usize, reps: usize, mut f: F) {
    let mut best = u64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..iters {
            acc = acc.wrapping_add(f());
        }
        let ns = t.elapsed().as_nanos() as u64 / iters as u64;
        best = best.min(ns);
        let _ = black_box(acc);
    }
    println!("{name:<34} {best:>8} ns/iter  ({:.3} ms)", best as f64 / 1e6);
}

fn main() {
    // 720p 4:2:0, border=32, matching frame.rs alloc geometry
    let y_w = 1280usize;
    let y_h = 720usize;
    let border = 32usize;
    let y_stride = (y_w + 2 * border + 31) & !31; // next_multiple_of(32)
    let yplane = (y_h + 2 * border) * y_stride;
    let uv_stride = y_stride >> 1;
    let uvplane = ((y_h / 2) + 2 * (border / 2)) * uv_stride;
    let total = yplane + uvplane + uvplane;
    println!("total buffer = {} bytes ({:.2} MB), y_stride={}", total, total as f64 / 1e6, y_stride);

    let iters = 200; // ~ a 200-frame run
    let reps = 30;

    // (A) fresh alloc+zero every iter (current frame.rs path, NO overwrite)
    bench("A fresh vec![0;n] (alloc+zero)", iters, reps, || {
        let b = alloc_zero(total);
        let s = b[total / 2] as u64; // force materialization
        black_box(b);
        s
    });

    // (B) fresh alloc+zero + full overwrite (current path + reconstruction)
    bench("B fresh alloc+zero + overwrite", iters, reps, || {
        let mut b = alloc_zero(total);
        touch(&mut b, 0xAB);
        let s = b[total / 2] as u64;
        black_box(b);
        s
    });

    // (C) pool recycle: one buffer, re-zero each iter + overwrite (keeps zero, saves alloc)
    {
        let mut pool = vec![0u8; total].into_boxed_slice();
        bench("C pool + rezero + overwrite", iters, reps, || {
            pool.fill(0);
            touch(&mut pool, 0xCD);
            pool[total / 2] as u64
        });
    }

    // (D) pool recycle, NO rezero, just overwrite (the proposed optimization)
    {
        let mut pool = vec![0u8; total].into_boxed_slice();
        bench("D pool + NO rezero + overwrite", iters, reps, || {
            touch(&mut pool, 0xEF);
            pool[total / 2] as u64
        });
    }

    // Delta interpretation:
    //   (B) = current per-frame buffer cost (alloc+zero+write)
    //   (D) = proposed (recycle, no zero, write)
    //   (B - D) ~= per-frame saving from pooling + skipping the zero
}
