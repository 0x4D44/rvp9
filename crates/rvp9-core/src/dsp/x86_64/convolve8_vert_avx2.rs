//! 8-tap vertical sub-pel convolution — x86_64 AVX2 implementation
//! (Wave J J4).
//!
//! Sibling of [`crate::dsp::x86_64::convolve8_horiz_avx2`]: for every
//! output pixel, convolve eight vertically-adjacent source samples with
//! the resolved sub-pel filter row:
//!
//! ```text
//! dst[y][x] = clip255( ( Σ_{k=0..8} src[(y+k)][x] * filter_y[k] + 64 ) >> 7 )
//! ```
//!
//! Mirrors libvpx `convolve_vert` (`vpx_dsp/vpx_convolve.c:67-88`) with
//! `y_step_q4 == SUBPEL_SHIFTS`. The scalar reference is
//! [`crate::inter::filter::convolve8_vert`]; callers pre-shift `src` up
//! by three rows so `src[(y)*src_stride + x]` is the first of the eight
//! taps for output `(y, x)`.
//!
//! ## Algorithm
//!
//! Three column-width tiers, all using a **sliding-window** dataflow
//! (libvpx `vpx_filter_block1d{16,8}_v8_avx2` ported to Rust): seven
//! source rows are primed once and held in registers, and each inner
//! iteration loads only the new trailing row(s) instead of re-loading
//! all eight per output row.
//!
//! * `w - cg >= 32` — 256-bit fast path, 32 columns per group, used by
//!   the 32- and 64-wide VP9 block sizes (and pass 2 of the 2-D
//!   convolve). Holds the eight contributing source rows in eight
//!   256-bit registers `r[0..8]` (one full 256-bit row load each) and
//!   processes **two output rows per inner iteration**: reduce the
//!   current window, slide it down one row (`r[0..7] = r[1..8]`; load
//!   the new trailing row `r[7]`), reduce again, then slide once more
//!   for the next pair. Each new output row costs a single 256-bit row
//!   load instead of eight. Odd `h` is finished by a one-row tail that
//!   re-primes the trailing register.
//!
//! * `gw == 16` / `gw == 8` (the `cg < w` tier with `gw >= 8`) —
//!   256-bit **pair-row** path mirroring
//!   `vpx_filter_block1d{16,8}_v8_avx2`. Each 256-bit register pairs
//!   two adjacent source rows (`row k` in the low 128-bit lane,
//!   `row k+1` in the high lane — see `pair_rows`). Pre-`unpack`ing
//!   adjacent pairs builds the byte-interleaved tap-pair operands
//!   `ss_lo[0..2]` (+`ss_hi[0..2]` for the 16-wide high 8 columns);
//!   one `vconv256_pair` then yields **two output rows at once** —
//!   output row `y` in the low lane, `y+1` in the high lane. Each pair
//!   loads only two new rows and slides the `ss` window by two.
//!   `gw == 16` emits two full 16-byte rows via `packus(lo, hi)` +
//!   per-lane store; `gw == 8` uses the low lane only with `storel`.
//!   Odd `h` falls to a one-row 128-bit `vconv` tail.
//!
//! * `gw == 4` — the rare narrow tail keeps the original single-row
//!   128-bit path (4-byte loads + low-half reduction). Pair-row 4-byte
//!   paths add complexity for negligible win; the analogue is the
//!   horiz drive's `w >= 8` gate on its pair loop.
//!
//! All tiers feed the same verified reduction: the eight rows are
//! byte-interleaved into tap-pair operands (`unpacklo` for the low 8
//! columns, `unpackhi` for the high 8) and fed through 4× `maddubs`,
//! then reduced in libvpx's `(x0+x2) + (x1+x3) + 64` order with the
//! saturating final add and `srai 7`; `packus_epi16` clips to
//! `[0, 255]`. This reduction order is the bit-exact contract — see
//! [`combine`] / [`combine256`].
//!
//! ## Bounds
//!
//! Vertical column loads are exactly the group width (4 / 8 / 16 / 32
//! bytes), so there is no horizontal over-read — the kernel reads
//! `src` directly with no scratch. The vertical extent (`h + 7` rows)
//! is covered by the scalar `convolve8_vert` footprint
//! (`(h - 1 + 7) * src_stride + w`). The 256-bit sliding window only
//! ever loads rows `0..=h + 6` (its trailing load for the last output
//! row `h - 1` reaches exactly row `h + 6`), identical to the 128-bit
//! per-row path.
//!
//! ## Contract
//!
//! Sub-pel filter rows only (taps fit i8). The full-pel row (centre tap
//! 128) is routed to `convolve_copy` upstream — see the horizontal
//! kernel's module doc.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! All loads from `src` and stores to `dst` use unaligned intrinsics.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_adds_epi16, _mm_cvtsi32_si128, _mm_cvtsi128_si32,
    _mm_loadl_epi64, _mm_loadu_si128, _mm_maddubs_epi16, _mm_packus_epi16, _mm_set1_epi16,
    _mm_srai_epi16, _mm_storel_epi64, _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
    _mm256_add_epi16, _mm256_adds_epi16, _mm256_castsi128_si256, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_maddubs_epi16,
    _mm256_packus_epi16, _mm256_set1_epi16, _mm256_srai_epi16, _mm256_storeu_si256,
    _mm256_unpackhi_epi8, _mm256_unpacklo_epi8,
};

/// Pack two adjacent filter taps into the replicated i8-pair form
/// consumed by `_mm_maddubs_epi16`'s signed operand.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn tap_pair(lo: i16, hi: i16) -> __m128i {
    _mm_set1_epi16(i16::from_le_bytes([lo as i8 as u8, hi as i8 as u8]))
}

/// Load `gw` source bytes (one column-group of one row) into the low
/// lanes of an `__m128i`. `gw ∈ {4, 8, 16}`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_cols(ptr: *const u8, gw: usize) -> __m128i {
    match gw {
        4 => _mm_cvtsi32_si128(core::ptr::read_unaligned(ptr as *const i32)),
        8 => _mm_loadl_epi64(ptr as *const _),
        _ => _mm_loadu_si128(ptr as *const _),
    }
}

/// Reduce four `maddubs` partial sums to the rounded/shifted (pre-clip)
/// i16 result, matching libvpx `convolve8_16` ordering exactly.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn combine(x0: __m128i, x1: __m128i, x2: __m128i, x3: __m128i) -> __m128i {
    let k_64 = _mm_set1_epi16(1 << 6);
    let mut sum1 = _mm_add_epi16(x0, x2);
    let sum2 = _mm_add_epi16(x1, x3);
    sum1 = _mm_add_epi16(sum1, k_64);
    sum1 = _mm_adds_epi16(sum1, sum2);
    _mm_srai_epi16(sum1, 7)
}

/// 256-bit tap-pair: the i8 pair replicated across all sixteen 16-bit
/// lanes of an `__m256i` (the `_mm256_maddubs_epi16` signed operand).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn tap_pair256(lo: i16, hi: i16) -> __m256i {
    _mm256_set1_epi16(i16::from_le_bytes([lo as i8 as u8, hi as i8 as u8]))
}

/// 256-bit counterpart of [`combine`] — identical saturation order, all
/// sixteen 16-bit lanes at once.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn combine256(x0: __m256i, x1: __m256i, x2: __m256i, x3: __m256i) -> __m256i {
    let k_64 = _mm256_set1_epi16(1 << 6);
    let mut sum1 = _mm256_add_epi16(x0, x2);
    let sum2 = _mm256_add_epi16(x1, x3);
    sum1 = _mm256_add_epi16(sum1, k_64);
    sum1 = _mm256_adds_epi16(sum1, sum2);
    _mm256_srai_epi16(sum1, 7)
}

/// 256-bit vertical reduction over eight source rows for one half.
/// `_mm256_unpack{lo,hi}_epi8` interleave **within each 128-bit lane**,
/// and `_mm256_packus_epi16` packs the same way, so the lane structure
/// composes back to the natural column order 0..32 with no permute —
/// `unpacklo` carries cols {0–7, 16–23}, `unpackhi` cols {8–15, 24–31},
/// and `packus(lo, hi)` reassembles cols 0..15 / 16..31 per lane.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn vconv256(
    r: &[__m256i; 8],
    f01: __m256i,
    f23: __m256i,
    f45: __m256i,
    f67: __m256i,
    hi: bool,
) -> __m256i {
    let (u0, u1, u2, u3) = if hi {
        (
            _mm256_unpackhi_epi8(r[0], r[1]),
            _mm256_unpackhi_epi8(r[2], r[3]),
            _mm256_unpackhi_epi8(r[4], r[5]),
            _mm256_unpackhi_epi8(r[6], r[7]),
        )
    } else {
        (
            _mm256_unpacklo_epi8(r[0], r[1]),
            _mm256_unpacklo_epi8(r[2], r[3]),
            _mm256_unpacklo_epi8(r[4], r[5]),
            _mm256_unpacklo_epi8(r[6], r[7]),
        )
    };
    combine256(
        _mm256_maddubs_epi16(u0, f01),
        _mm256_maddubs_epi16(u1, f23),
        _mm256_maddubs_epi16(u2, f45),
        _mm256_maddubs_epi16(u3, f67),
    )
}

/// Pack a row-pair into a 256-bit register: `a` (a single output-block
/// row's `gw` source bytes) in the low 128-bit lane, `b` (the next
/// source row) in the high lane. Mirrors libvpx's
/// `s32b[k] = inserti128(row_k, row_{k+1})` used by the 8/16-wide
/// vertical paths.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pair_rows(a: __m128i, b: __m128i) -> __m256i {
    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(a), b)
}

/// 256-bit pair-row reduction (the 8/16-wide vertical paths). `ss`
/// holds the four byte-interleaved tap-pair operands for tap pairs
/// (0,1)..(6,7); each `ss[i]`'s low 128-bit lane feeds output row `y`
/// and its high lane output row `y+1` (libvpx's `s1`/`s2` topology —
/// see [`pair_rows`]). Identical reduction order to [`combine256`].
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn vconv256_pair(
    ss: &[__m256i; 4],
    f01: __m256i,
    f23: __m256i,
    f45: __m256i,
    f67: __m256i,
) -> __m256i {
    combine256(
        _mm256_maddubs_epi16(ss[0], f01),
        _mm256_maddubs_epi16(ss[1], f23),
        _mm256_maddubs_epi16(ss[2], f45),
        _mm256_maddubs_epi16(ss[3], f67),
    )
}

/// Vertical 8-tap reduction over eight source rows for one half
/// (low 8 columns if `!hi`, high 8 columns if `hi`).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn vconv(
    r: &[__m128i; 8],
    f01: __m128i,
    f23: __m128i,
    f45: __m128i,
    f67: __m128i,
    hi: bool,
) -> __m128i {
    let (u0, u1, u2, u3) = if hi {
        (
            _mm_unpackhi_epi8(r[0], r[1]),
            _mm_unpackhi_epi8(r[2], r[3]),
            _mm_unpackhi_epi8(r[4], r[5]),
            _mm_unpackhi_epi8(r[6], r[7]),
        )
    } else {
        (
            _mm_unpacklo_epi8(r[0], r[1]),
            _mm_unpacklo_epi8(r[2], r[3]),
            _mm_unpacklo_epi8(r[4], r[5]),
            _mm_unpacklo_epi8(r[6], r[7]),
        )
    };
    combine(
        _mm_maddubs_epi16(u0, f01),
        _mm_maddubs_epi16(u1, f23),
        _mm_maddubs_epi16(u2, f45),
        _mm_maddubs_epi16(u3, f67),
    )
}

/// AVX2 implementation of the 8-tap vertical sub-pel convolution.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`convolve8_vert_avx2`]) when `caps.avx2 == true`.
///
/// `src` must hold at least `(h - 1 + 7) * src_stride + w` valid bytes
/// (the scalar `convolve8_vert` footprint); `dst` at least
/// `(h - 1) * dst_stride + w`. `w` and `h` must be in `1..=64`.
// G4: loads from `src` use _mm_loadu_si128 / _mm_loadl_epi64 / read_unaligned (unaligned)
// G4: stores to `dst` use _mm_storeu_si128 / _mm_storel_epi64 / write_unaligned (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convolve8_vert_avx2_inner(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
) {
    debug_assert!((1..=64).contains(&w));
    debug_assert!((1..=64).contains(&h));
    debug_assert!(src.len() >= (h - 1 + 7) * src_stride + w);
    debug_assert!(dst.len() >= (h - 1) * dst_stride + w);
    debug_assert!(
        filter_y.iter().all(|&t| (-128..=127).contains(&t)),
        "convolve8_vert_avx2: tap out of i8 range (full-pel row must use convolve_copy)"
    );

    // The 128-bit tap pairs (used only by the narrow `gw == 4` tail and
    // the odd-h tail) are computed lazily at those sites — never for the
    // common VP9 sizes (even heights, w ∈ {8,16,32,64}), which take the
    // 256-bit pair-row path. This trims 4 broadcasts/call of fixed
    // per-call setup off every common convolve (see notes/convolve8-audit).

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    let mut cg = 0usize;

    // L4: 256-bit fast path — 32 columns per group (used by the 32- and
    // 64-wide VP9 block sizes, and pass 2 of the 2-D convolve). Same
    // maddubs reduction as the 128-bit path, twice as wide.
    //
    // Sliding-window dataflow: hoist the eight 256-bit row loads out of
    // the `for y` loop. Pre-prime `r[0..6]` from rows 0..6, then per
    // **pair** `(y, y+1)` load `r[6] = row y+6`, `r[7] = row y+7` and
    // produce two outputs. After each pair slide the window by 2 (the
    // prefix `r[0..6]` becomes the old `r[2..8]`). If `h` is odd, the
    // final row is handled by a 1-row tail. 8 row loads/output row →
    // 2/2 = 1 row load per output row.
    //
    // Bound: pair `(y, y+1)` loads rows `y..=y+7`. For the contract
    // (last loadable row `h+6`) the pair loop runs while
    // `y+7 <= h+6`, i.e., `y+1 < h`. The tail then handles `y == h-1`,
    // re-priming `r[7]` to row `y+7 = h+6` — exactly the last loadable
    // row (its window prefix is already populated from the prior iter).
    if w - cg >= 32 {
        let g01 = tap_pair256(filter_y[0], filter_y[1]);
        let g23 = tap_pair256(filter_y[2], filter_y[3]);
        let g45 = tap_pair256(filter_y[4], filter_y[5]);
        let g67 = tap_pair256(filter_y[6], filter_y[7]);
        while w - cg >= 32 {
            let col_ptr = src_ptr.add(cg);
            // Prime r[0..7] = rows 0..7 of this column-group. This is
            // the initial state for output row 0; pair `(0,1)` will
            // consume the entire prefix and then `r` gets slid.
            let mut r: [__m256i; 8] = core::array::from_fn(|k| {
                _mm256_loadu_si256(col_ptr.add(k * src_stride) as *const __m256i)
            });
            let mut y = 0usize;
            // 2-rows-per-iter pair loop. Invariant at loop top:
            // `r[k]` holds source row `y + k` for k ∈ 0..=7.
            // Condition `y + 1 < h` means we have two rows to produce
            // and rows `y..=y+7` are loadable (largest = `h+6`).
            while y + 1 < h {
                // Output row y from r[0..7] = rows y..=y+7.
                let lo_a = vconv256(&r, g01, g23, g45, g67, false);
                let hi_a = vconv256(&r, g01, g23, g45, g67, true);
                let res_a = _mm256_packus_epi16(lo_a, hi_a);
                _mm256_storeu_si256(dst_ptr.add(y * dst_stride + cg) as *mut __m256i, res_a);

                // Slide window by 1: r holds rows y+1..=y+7 + (new r[7]).
                // r[7] needs to be row y+8 for output row y+1. Only
                // load it if it's within the contract — for the very
                // last pair (y+1 == h-1, i.e., y+8 == h+6) this is
                // exactly the last loadable row.
                r[0] = r[1];
                r[1] = r[2];
                r[2] = r[3];
                r[3] = r[4];
                r[4] = r[5];
                r[5] = r[6];
                r[6] = r[7];
                r[7] = _mm256_loadu_si256(col_ptr.add((y + 8) * src_stride) as *const __m256i);

                // Output row y+1.
                let lo_b = vconv256(&r, g01, g23, g45, g67, false);
                let hi_b = vconv256(&r, g01, g23, g45, g67, true);
                let res_b = _mm256_packus_epi16(lo_b, hi_b);
                _mm256_storeu_si256(
                    dst_ptr.add((y + 1) * dst_stride + cg) as *mut __m256i,
                    res_b,
                );

                // Slide once more so the loop-top invariant holds for
                // the next pair `(y+2, y+3)`: r[k] must become row
                // y+2+k. The new trailing register r[7] = row y+9 is
                // only loaded when another pair iteration follows
                // (`y+3 < h`); otherwise row y+9 could exceed the
                // contract's `h+6` boundary, so we leave r[7] stale —
                // the odd-h tail below re-primes it before use.
                r[0] = r[1];
                r[1] = r[2];
                r[2] = r[3];
                r[3] = r[4];
                r[4] = r[5];
                r[5] = r[6];
                r[6] = r[7];
                if y + 3 < h {
                    r[7] = _mm256_loadu_si256(col_ptr.add((y + 9) * src_stride) as *const __m256i);
                }

                y += 2;
            }
            // Odd-h tail: one remaining output row at `y == h-1`. The
            // slides left r[0..6] = rows y..=y+6, but the trailing
            // r[7] may be stale (the `y+3 < h` guard above suppressed
            // its load on the final pair), so re-prime it to row y+7 =
            // h+6 — exactly the last row the contract guarantees.
            while y < h {
                r[7] = _mm256_loadu_si256(col_ptr.add((y + 7) * src_stride) as *const __m256i);
                let lo = vconv256(&r, g01, g23, g45, g67, false);
                let hi = vconv256(&r, g01, g23, g45, g67, true);
                let res = _mm256_packus_epi16(lo, hi);
                _mm256_storeu_si256(dst_ptr.add(y * dst_stride + cg) as *mut __m256i, res);
                y += 1;
            }
            cg += 32;
        }
    }

    while cg < w {
        let gw = (w - cg).min(16);
        let col_ptr = src_ptr.add(cg);

        if gw >= 8 {
            // 256-bit broadcast tap pairs for the pair-row path. Derived
            // here (not hoisted) so they cost nothing when only the
            // 32-col fast path or the `gw == 4` tail runs.
            let g01 = tap_pair256(filter_y[0], filter_y[1]);
            let g23 = tap_pair256(filter_y[2], filter_y[3]);
            let g45 = tap_pair256(filter_y[4], filter_y[5]);
            let g67 = tap_pair256(filter_y[6], filter_y[7]);

            // Pair-row sliding window mirroring libvpx
            // `vpx_filter_block1d{16,8}_v8_avx2`. Each 256-bit register
            // pairs two adjacent source rows (`row k` low lane,
            // `row k+1` high lane); a single `vconv256_pair` then
            // yields two output rows at once — output row `y` in the
            // low lane, `y+1` in the high lane. The window primes seven
            // rows once and slides by two per iteration, so each pair
            // loads only two new rows (8 loads/output-row → 1).
            //
            // `do_hi` selects the 16-wide path (compute the high 8
            // columns too) vs the 8-wide path (low lane only). For
            // VP9, `gw == 16` here implies `w == 16` (the 32-col fast
            // path already drained any 32-col groups), so `gw` is the
            // full group width.
            let do_hi = gw == 16;

            // Prime rows 0..6 as 128-bit `gw`-byte loads (always within
            // the contract for any `h >= 1`: rows `0..=h+6` are
            // loadable). `head` trails the last primed row.
            let s: [__m128i; 7] =
                core::array::from_fn(|k| load_cols(col_ptr.add(k * src_stride), gw));
            // Pre-pack tap-pair operands for taps (0,1)/(2,3)/(4,5).
            // Slot [3] (tap pair (6,7)) is filled fresh each iteration
            // before use; seed it with slot [2] to avoid an out-of-loop
            // dummy load.
            let r01 = pair_rows(s[0], s[1]);
            let r12 = pair_rows(s[1], s[2]);
            let r23 = pair_rows(s[2], s[3]);
            let r34 = pair_rows(s[3], s[4]);
            let r45 = pair_rows(s[4], s[5]);
            let r56 = pair_rows(s[5], s[6]);
            let ss_lo2 = _mm256_unpacklo_epi8(r45, r56);
            let mut ss_lo: [__m256i; 4] = [
                _mm256_unpacklo_epi8(r01, r12),
                _mm256_unpacklo_epi8(r23, r34),
                ss_lo2,
                ss_lo2,
            ];
            let mut ss_hi: [__m256i; 4] = if do_hi {
                let ss_hi2 = _mm256_unpackhi_epi8(r45, r56);
                [
                    _mm256_unpackhi_epi8(r01, r12),
                    _mm256_unpackhi_epi8(r23, r34),
                    ss_hi2,
                    ss_hi2,
                ]
            } else {
                ss_lo
            };
            // `head` = the last primed row (row y+6 at loop top).
            let mut head = s[6];

            let mut y = 0usize;
            // Pair loop. Invariant: `ss_{lo,hi}[0..3]` hold tap pairs
            // (0,1)/(2,3)/(4,5) for output rows `(y, y+1)`, and `head`
            // = source row `y+6`. `y + 1 < h` ⇒ two rows remain and the
            // largest read (`row y+8`) is `≤ h+6` (last loadable row).
            while y + 1 < h {
                let s7 = load_cols(col_ptr.add((y + 7) * src_stride), gw);
                let s8 = load_cols(col_ptr.add((y + 8) * src_stride), gw);
                let r6 = pair_rows(head, s7);
                let r7 = pair_rows(s7, s8);

                ss_lo[3] = _mm256_unpacklo_epi8(r6, r7);
                let lo = vconv256_pair(&ss_lo, g01, g23, g45, g67);

                let res = if do_hi {
                    ss_hi[3] = _mm256_unpackhi_epi8(r6, r7);
                    let hi = vconv256_pair(&ss_hi, g01, g23, g45, g67);
                    // packus per 128-bit lane: lane0 = full row y (16
                    // cols), lane1 = full row y+1.
                    _mm256_packus_epi16(lo, hi)
                } else {
                    // 8-wide: low lane only; packus(lo, lo) puts row y's
                    // 8 bytes in lane0[0..8] and row y+1's in
                    // lane1[0..8].
                    _mm256_packus_epi16(lo, lo)
                };

                let row_a = _mm256_castsi256_si128(res);
                let row_b = _mm256_extracti128_si256::<1>(res);
                let dst_a = dst_ptr.add(y * dst_stride + cg);
                let dst_b = dst_ptr.add((y + 1) * dst_stride + cg);
                if do_hi {
                    _mm_storeu_si128(dst_a as *mut _, row_a);
                    _mm_storeu_si128(dst_b as *mut _, row_b);
                } else {
                    _mm_storel_epi64(dst_a as *mut _, row_a);
                    _mm_storel_epi64(dst_b as *mut _, row_b);
                }

                // Slide the window down two rows.
                ss_lo[0] = ss_lo[1];
                ss_lo[1] = ss_lo[2];
                ss_lo[2] = ss_lo[3];
                if do_hi {
                    ss_hi[0] = ss_hi[1];
                    ss_hi[1] = ss_hi[2];
                    ss_hi[2] = ss_hi[3];
                }
                head = s8;
                y += 2;
            }

            // 128-bit tail operands = low lane of the pair-row taps (a
            // free `castsi256_si128` register alias). Only consumed by the
            // odd-h tail below, which never runs for even VP9 heights.
            let f01 = _mm256_castsi256_si128(g01);
            let f23 = _mm256_castsi256_si128(g23);
            let f45 = _mm256_castsi256_si128(g45);
            let f67 = _mm256_castsi256_si128(g67);
            // Odd-h tail: one output row at `y == h-1` via the 1-row
            // 128-bit path (reads rows `y..=y+7` = `h-1..=h+6`, all
            // within the contract).
            while y < h {
                let rr: [__m128i; 8] =
                    core::array::from_fn(|k| load_cols(col_ptr.add((y + k) * src_stride), gw));
                let dst_row = dst_ptr.add(y * dst_stride + cg);
                if do_hi {
                    let lo = vconv(&rr, f01, f23, f45, f67, false);
                    let hi = vconv(&rr, f01, f23, f45, f67, true);
                    _mm_storeu_si128(dst_row as *mut _, _mm_packus_epi16(lo, hi));
                } else {
                    let res16 = vconv(&rr, f01, f23, f45, f67, false);
                    _mm_storel_epi64(dst_row as *mut _, _mm_packus_epi16(res16, res16));
                }
                y += 1;
            }
        } else {
            // `gw == 4`: rare narrow tail, kept on the original
            // single-row 128-bit path (mirrors the horiz drive's
            // `w >= 8` gate on its pair loop).
            debug_assert_eq!(gw, 4);
            let f01 = tap_pair(filter_y[0], filter_y[1]);
            let f23 = tap_pair(filter_y[2], filter_y[3]);
            let f45 = tap_pair(filter_y[4], filter_y[5]);
            let f67 = tap_pair(filter_y[6], filter_y[7]);
            for y in 0..h {
                let rr: [__m128i; 8] =
                    core::array::from_fn(|k| load_cols(col_ptr.add((y + k) * src_stride), gw));
                let res16 = vconv(&rr, f01, f23, f45, f67, false);
                let res8 = _mm_packus_epi16(res16, res16);
                let dst_row = dst_ptr.add(y * dst_stride + cg);
                let four = _mm_cvtsi128_si32(res8) as u32;
                core::ptr::write_unaligned(dst_row as *mut u32, four);
            }
        }
        cg += 16;
    }
}

/// Safe wrapper for the AVX2 8-tap vertical sub-pel convolution.
///
/// The dispatch slot [`crate::dsp::SimdDispatch::convolve8_vert`] points
/// at this wrapper on AVX2 hosts and at the scalar reference
/// (`crate::inter::filter::convolve8_vert`) elsewhere.
#[cfg(target_arch = "x86_64")]
pub fn convolve8_vert_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
) {
    assert!(
        matches!(w, 4 | 8 | 16 | 32 | 64) && (1..=64).contains(&h),
        "unsupported convolution dimensions"
    );
    super::convolve_copy_avx2::validate_block(src.len(), src_stride, w, h + 7);
    super::convolve_copy_avx2::validate_block(dst.len(), dst_stride, w, h);

    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock init
    // time and are monotonic for the process lifetime. Test paths gate
    // every call on a runtime `is_x86_feature_detected!("avx2")` check.
    unsafe {
        convolve8_vert_avx2_inner(src, src_stride, dst, dst_stride, filter_y, w, h);
    }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `convolve8_vert_avx2`.
    use super::convolve8_vert_avx2;
    use crate::inter::filter::{
        BILINEAR_KERNEL, EIGHTTAP_KERNEL, EIGHTTAP_SHARP_KERNEL, EIGHTTAP_SMOOTH_KERNEL,
        convolve8_vert,
    };

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    fn lcg_next(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *state
    }

    /// Differential vs scalar across the VP9 (w, h) matrix, every
    /// sub-pel filter row (skipping the full-pel row 0 — routed to
    /// `convolve_copy` upstream), and all four kernel tables.
    #[test]
    fn differential_all_filters() {
        require_avx2!();
        let shapes: &[(usize, usize)] = &[
            (4, 4),
            (4, 8),
            (8, 4),
            (8, 8),
            (8, 16),
            (16, 8),
            (16, 16),
            (16, 32),
            (32, 16),
            (32, 32),
            (32, 64),
            (64, 32),
            (64, 64),
        ];
        let kernels = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        let mut state: u32 = 0x9E37_79B9;
        for &(w, h) in shapes {
            for kernel in kernels {
                for filter in kernel.iter().skip(1) {
                    for _iter in 0..16 {
                        let src_stride = w + (lcg_next(&mut state) as usize & 0x7);
                        let dst_stride = w + (lcg_next(&mut state) as usize & 0x7);
                        let src_len = (h - 1 + 7) * src_stride + w;
                        let dst_len = (h - 1) * dst_stride + w;
                        let mut src = vec![0u8; src_len];
                        for s in src.iter_mut() {
                            *s = (lcg_next(&mut state) & 0xFF) as u8;
                        }
                        let mut dst_scalar = vec![0u8; dst_len];
                        let mut dst_avx2 = vec![0u8; dst_len];
                        convolve8_vert(&src, src_stride, &mut dst_scalar, dst_stride, filter, w, h);
                        convolve8_vert_avx2(
                            &src,
                            src_stride,
                            &mut dst_avx2,
                            dst_stride,
                            filter,
                            w,
                            h,
                        );
                        assert_eq!(
                            dst_scalar, dst_avx2,
                            "mismatch w={w} h={h} filter={filter:?}"
                        );
                    }
                }
            }
        }
    }

    /// Shapes the head-to-head harness doesn't hit but production
    /// callers and the 2-D composite can: odd heights (force the
    /// 1-row pair tail), `h == 1` / `h == 2` corners (zero or one
    /// pair), and **tight-stride** cases (`src_stride == w`, no slack)
    /// that guard against a pair-row vertical over-read.
    ///
    /// The vertical analogue of the horiz `w == 4` OOB: the pair loop
    /// produces two output rows `(y, y+1)` and must only ever read
    /// source rows `y..=y+8` while `y + 1 < h`. The largest read is
    /// then row `(h - 2) + 8 = h + 6` — exactly the contract's last
    /// loadable row. A buggy slide that loaded row `h + 7` would
    /// overrun a tightly-sized `src` (no slack) and trip a Vec OOB
    /// under MIRI / read adjacent heap under release. Pinned seed so
    /// failures bisect cleanly.
    #[test]
    fn differential_odd_and_narrow_shapes() {
        require_avx2!();
        // (w, h, src_stride). `src_stride == 0` ⇒ randomized stride
        // with slack; `src_stride > 0` pins it to that exact tight
        // value.
        //
        // Shapes exercised:
        //   - (8, 8, 8) / (16, 16, 16) — headline widths at tight
        //     stride; the pair loop's last read must stop at `h + 6`.
        //   - (8, 7, 8) / (16, 15, 16) — odd h, forces the 1-row tail
        //     after the pair loop (tight stride).
        //   - (8, 1, 8) / (16, 1, 16) — h == 1: zero pairs, tail-only.
        //   - (8, 2, 8) / (16, 2, 16) — h == 2: exactly one pair, no
        //     tail.
        //   - (32, 33, 32) — odd h through the 256-bit fast path
        //     (tight stride) — exercises that path's pair + tail too.
        //   - (32, 32, 32) — even h, 256-bit fast path (tight stride):
        //     the `h + 6` boundary is reached via the slide-by-1
        //     `r[7] = row(y+8)` at `y == h-2` (no odd-h tail re-prime),
        //     covering the slide path the odd-h cases miss.
        //   - (64, 16, 64) — even h at the widest width (tight stride):
        //     two 256-bit column groups (`cg = 0` and `cg = 32`), each
        //     hitting `h + 6` via the same slide-by-1 path.
        //   - (8, 9, 0) / (16, 17, 0) — odd h with slack (sanity).
        //   - (4, 8, 4) — w == 4 stays on the single-row path (the
        //     pair loop is gated on `gw >= 8`); guards against
        //     accidentally routing the narrow width through a pair
        //     path.
        let shapes: &[(usize, usize, usize)] = &[
            (8, 8, 8),
            (16, 16, 16),
            (8, 7, 8),
            (16, 15, 16),
            (8, 1, 8),
            (16, 1, 16),
            (8, 2, 8),
            (16, 2, 16),
            (32, 33, 32),
            (32, 32, 32),
            (64, 16, 64),
            (8, 9, 0),
            (16, 17, 0),
            (4, 8, 4),
        ];
        let kernels = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        let mut state: u32 = 0xDEAD_BEEF;
        for &(w, h, fixed_stride) in shapes {
            for kernel in kernels {
                // A few sub-pel rows: every-other-tap-zero (1),
                // mid-range (4), half-pel symmetric (8),
                // asymmetric-high (12).
                for &r in &[1usize, 4, 8, 12] {
                    let filter = &kernel[r];
                    let src_stride = if fixed_stride > 0 {
                        fixed_stride
                    } else {
                        w + (lcg_next(&mut state) as usize & 0x7)
                    };
                    let dst_stride = w + (lcg_next(&mut state) as usize & 0x7);
                    // Exact scalar contract footprint — **no slack** —
                    // so any pair-row over-read past row `h + 6` trips
                    // a Vec OOB (MIRI) or reads adjacent heap (release).
                    let src_len = (h - 1 + 7) * src_stride + w;
                    let dst_len = (h - 1) * dst_stride + w;
                    let mut src = vec![0u8; src_len];
                    for s in src.iter_mut() {
                        *s = (lcg_next(&mut state) & 0xFF) as u8;
                    }
                    let mut dst_scalar = vec![0u8; dst_len];
                    let mut dst_avx2 = vec![0u8; dst_len];
                    convolve8_vert(&src, src_stride, &mut dst_scalar, dst_stride, filter, w, h);
                    convolve8_vert_avx2(&src, src_stride, &mut dst_avx2, dst_stride, filter, w, h);
                    assert_eq!(
                        dst_scalar, dst_avx2,
                        "mismatch w={w} h={h} src_stride={src_stride} r={r} filter={filter:?}"
                    );
                }
            }
        }
    }

    /// Tautology guard — AVX2 path exercised; a half-pel filter blurs.
    #[test]
    fn tautology_guard_avx2_path() {
        require_avx2!();
        assert!(is_x86_feature_detected!("avx2"));
        let filter = &EIGHTTAP_KERNEL[8];
        let w = 8;
        let h = 4;
        let stride = w;
        let src_len = (h - 1 + 7) * stride + w;
        let src: Vec<u8> = (0..src_len).map(|i| (i * 7 + 11) as u8).collect();
        let mut dst = vec![0u8; w * h];
        convolve8_vert_avx2(&src, stride, &mut dst, w, filter, w, h);
        // Output row 0 must not equal src row 3 (the integer-pel pass).
        assert_ne!(
            &dst[..w],
            &src[3 * stride..3 * stride + w],
            "half-pel must blur"
        );
    }

    /// Round-trip through SimdDispatch with forced-AVX2 / forced-scalar.
    #[test]
    fn dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps_avx2 = detect();
        caps_avx2.avx2 = true;
        let d_avx2 = SimdDispatch::build(caps_avx2);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let d_scalar = SimdDispatch::build(caps_scalar);

        let filter = &EIGHTTAP_SMOOTH_KERNEL[11];
        let mut state: u32 = 0x0BAD_F00D;
        for &(w, h) in &[(4usize, 4usize), (8, 8), (16, 16), (32, 32), (64, 64)] {
            let stride = w;
            let src_len = (h - 1 + 7) * stride + w;
            let mut src = vec![0u8; src_len];
            for s in src.iter_mut() {
                *s = (lcg_next(&mut state) & 0xFF) as u8;
            }
            let mut dst_avx2 = vec![0u8; h * w];
            let mut dst_scalar = vec![0u8; h * w];
            (d_avx2.convolve8_vert())(&src, stride, &mut dst_avx2, w, filter, w, h);
            (d_scalar.convolve8_vert())(&src, stride, &mut dst_scalar, w, filter, w, h);
            assert_eq!(
                dst_avx2, dst_scalar,
                "dispatch round-trip mismatch w={w} h={h}"
            );
        }
    }
}
