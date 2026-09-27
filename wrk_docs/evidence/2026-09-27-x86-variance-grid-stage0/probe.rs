#![allow(unsafe_op_in_unsafe_fn)]
use std::arch::x86_64::*;
use std::hint::black_box;
use std::time::Instant;

#[target_feature(enable = "avx2")]
unsafe fn hsum256_epi32(v: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
    let s = _mm_add_epi32(s, _mm_srli_si128::<4>(s));
    _mm_cvtsi128_si32(s)
}

// Exact 8x8 specialization of variance_avx2.rs:var_sum_sse_inner.
#[target_feature(enable = "avx2")]
unsafe fn var_sum_sse_inner(src: &[u8], ss: usize, pred: &[u8], ps: usize) -> (i64, u64) {
    let src_ptr = src.as_ptr();
    let pred_ptr = pred.as_ptr();
    let ones16 = _mm256_set1_epi16(1);
    let mut sum_acc = _mm256_setzero_si256();
    let mut sse_acc = _mm256_setzero_si256();
    for r in 0..8 {
        let s = _mm_loadl_epi64(src_ptr.add(r * ss) as *const __m128i);
        let p = _mm_loadl_epi64(pred_ptr.add(r * ps) as *const __m128i);
        let sw = _mm256_cvtepu8_epi16(s);
        let pw = _mm256_cvtepu8_epi16(p);
        let diff = _mm256_sub_epi16(sw, pw);
        sum_acc = _mm256_add_epi32(sum_acc, _mm256_madd_epi16(diff, ones16));
        sse_acc = _mm256_add_epi32(sse_acc, _mm256_madd_epi16(diff, diff));
    }
    (hsum256_epi32(sum_acc) as i64, hsum256_epi32(sse_acc) as u32 as u64)
}

#[inline(never)]
fn cell(src: &[u8], ss: usize, pred: &[u8], ps: usize) -> (i32, u32) {
    let (sum, sse) = unsafe { var_sum_sse_inner(src, ss, pred, ps) };
    (sum as i32, sse as u32)
}

type Cell = fn(&[u8], usize, &[u8], usize) -> (i32, u32);
type Out = (u64, i64, [u32; 64], [i32; 64]);

// Exact loop and output shape of model_rd.rs:compute_variance_8x8.
#[inline(never)]
fn current(reduce: Cell, src: &[u8], ss: usize, pred: &[u8], ps: usize, bw: usize, bh: usize) -> Out {
    let nw = bw / 8;
    let nh = bh / 8;
    let mut sse8x8 = [0u32; 64];
    let mut sum8x8 = [0i32; 64];
    let mut total_sse = 0u64;
    let mut total_sum = 0i64;
    for i in 0..nh {
        for j in 0..nw {
            let k = i * nw + j;
            let s_off = (i * 8) * ss + j * 8;
            let p_off = (i * 8) * ps + j * 8;
            let (cs, csse) = reduce(&src[s_off..], ss, &pred[p_off..], ps);
            sse8x8[k] = csse;
            sum8x8[k] = cs;
            total_sse += csse as u64;
            total_sum += cs as i64;
        }
    }
    (total_sse, total_sum, sse8x8, sum8x8)
}

#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn grid_inner(src: &[u8], ss: usize, pred: &[u8], ps: usize, bw: usize, bh: usize,
                     sse: &mut [u32], sum: &mut [i32]) {
    let nw = bw / 8;
    for i in 0..bh / 8 {
        for j in 0..nw {
            let k = i * nw + j;
            let s_off = (i * 8) * ss + j * 8;
            let p_off = (i * 8) * ps + j * 8;
            let (cs, csse) = var_sum_sse_inner(&src[s_off..], ss, &pred[p_off..], ps);
            sum[k] = cs as i32;
            sse[k] = csse as u32;
        }
    }
}

#[inline(never)]
fn fused(src: &[u8], ss: usize, pred: &[u8], ps: usize, bw: usize, bh: usize) -> Out {
    let mut sse8x8 = [0u32; 64];
    let mut sum8x8 = [0i32; 64];
    let n = (bw / 8) * (bh / 8);
    unsafe { grid_inner(src, ss, pred, ps, bw, bh, &mut sse8x8[..n], &mut sum8x8[..n]); }
    let mut total_sse = 0u64;
    let mut total_sum = 0i64;
    for k in 0..n {
        total_sse += sse8x8[k] as u64;
        total_sum += sum8x8[k] as i64;
    }
    (total_sse, total_sum, sse8x8, sum8x8)
}

fn run(size: usize, cold: bool, rounds: usize) {
    let stride = size + 64;
    let banks = if cold { 32768 } else { 1 };
    let len = stride * size;
    let mut src = vec![0u8; len * banks];
    let mut pred = vec![0u8; len * banks];
    let mut x = 0x12345678u32;
    for k in 0..src.len() {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        src[k] = x as u8;
        pred[k] = (x >> 16) as u8;
    }
    let cell_fn = black_box(cell as Cell);
    for b in 0..banks.min(3) {
        let start = b * len;
        assert_eq!(
            current(cell_fn, &src[start..start + len], stride, &pred[start..start + len], stride, size, size),
            fused(&src[start..start + len], stride, &pred[start..start + len], stride, size, size)
        );
    }
    let iters = if cold { if size == 32 { 49_152 } else { 24_576 } } else { 200_000 };
    let mut measures = [Vec::new(), Vec::new()];
    let mut checksum = 0u64;
    for rep in 0..rounds {
        for order in 0..2 {
            let which = (rep + order) & 1;
            let t = Instant::now();
            for iter in 0..iters {
                let b = if cold { (iter * 73 + rep * 89) % banks } else { 0 };
                let start = b * len;
                let s = black_box(&src[start..start + len]);
                let p = black_box(&pred[start..start + len]);
                let out = if which == 0 {
                    current(cell_fn, s, stride, p, stride, size, size)
                } else {
                    fused(s, stride, p, stride, size, size)
                };
                checksum = checksum.wrapping_add(black_box(out.0));
                checksum = checksum.wrapping_add(black_box(out.1 as u64));
            }
            measures[which].push(t.elapsed().as_nanos() as f64 / iters as f64);
        }
    }
    for v in &mut measures { v.sort_by(|a,b| a.total_cmp(b)); }
    let mid = rounds / 2;
    println!("size={} cold={} current_ns={:.3} fused_ns={:.3} speedup={:.5} checksum={}",
             size, cold, measures[0][mid], measures[1][mid],
             measures[0][mid] / measures[1][mid], checksum);
    println!("raw_current={:?} raw_fused={:?}", measures[0], measures[1]);
}

fn main() {
    assert!(is_x86_feature_detected!("avx2"));
    let a = vec![1u8; 64*128];
    let b = vec![2u8; 64*128];
    for size in [32,64] {
        assert_eq!(current(cell, &a, 128, &b, 128, size, size), fused(&a, 128, &b, 128, size, size));
        run(size, false, 9);
        run(size, true, 9);
    }
}
