use std::arch::x86_64::*;
use std::arch::asm;
use std::hint::black_box;

#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn run(kind: u8, reps: usize) -> u64 {
    let mut bytes = [0i32; 16];
    let p = bytes.as_mut_ptr();
    let mut carry = _mm256_set1_epi32(1);
    let t0 = unsafe { _rdtsc() };
    for _ in 0..reps {
        let lo = carry;
        let hi = _mm256_add_epi32(carry, _mm256_set1_epi32(7));
        match kind {
            0 => {
                for j in 0..8 {
                    unsafe { std::ptr::write_volatile(p.add(j), _mm256_extract_epi32::<0>(lo) + j as i32) };
                    unsafe { std::ptr::write_volatile(p.add(j + 8), _mm256_extract_epi32::<0>(hi) + j as i32) };
                }
            }
            1 | 3 => {
                unsafe {
                    asm!(
                        "vmovdqu xmmword ptr [{p}], {lo0}",
                        "vmovdqu xmmword ptr [{p} + 16], {lo1}",
                        "vmovdqu xmmword ptr [{p} + 32], {hi0}",
                        "vmovdqu xmmword ptr [{p} + 48], {hi1}",
                        p = in(reg) p,
                        lo0 = in(xmm_reg) _mm256_castsi256_si128(lo),
                        lo1 = in(xmm_reg) _mm256_extracti128_si256::<1>(lo),
                        hi0 = in(xmm_reg) _mm256_castsi256_si128(hi),
                        hi1 = in(xmm_reg) _mm256_extracti128_si256::<1>(hi),
                        options(nostack)
                    );
                }
            }
            _ => unsafe {
                asm!(
                    "vmovdqu ymmword ptr [{p}], {lo}",
                    "vmovdqu ymmword ptr [{p} + 32], {hi}",
                    p = in(reg) p,
                    lo = in(ymm_reg) lo,
                    hi = in(ymm_reg) hi,
                    options(nostack)
                );
            },
        }
        black_box(&mut bytes);
        let (first, second) = if kind == 3 {
            let (a, b, c, d): (__m128i, __m128i, __m128i, __m128i);
            unsafe {
                asm!(
                    "vmovdqu {a}, xmmword ptr [{p}]",
                    "vmovdqu {b}, xmmword ptr [{p} + 16]",
                    "vmovdqu {c}, xmmword ptr [{p} + 32]",
                    "vmovdqu {d}, xmmword ptr [{p} + 48]",
                    p = in(reg) p,
                    a = lateout(xmm_reg) a,
                    b = lateout(xmm_reg) b,
                    c = lateout(xmm_reg) c,
                    d = lateout(xmm_reg) d,
                    options(nostack, readonly)
                );
            }
            (
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(a), b),
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(c), d),
            )
        } else {
            (unsafe { _mm256_loadu_si256(p.cast()) }, unsafe {
                _mm256_loadu_si256(p.add(8).cast())
            })
        };
        carry = _mm256_xor_si256(first, second);
        carry = black_box(carry);
    }
    let t1 = unsafe { _rdtsc() };
    black_box(carry);
    t1 - t0
}

fn main() {
    let reps = 10_000_000;
    for _ in 0..3 {
        for kind in [2, 1, 3, 0, 3, 1, 2, 0] {
            let cycles = unsafe { run(kind, reps) };
            println!("kind={kind} cycles/iter={:.2}", cycles as f64 / reps as f64);
        }
    }
}
