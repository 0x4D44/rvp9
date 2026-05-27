/* libvpx kernel head-to-head bench (rvp9 decoder kernel audit).
 *
 * Times a fixed set of libvpx DSP kernels on deterministic, LCG-generated
 * inputs that the rvp9 side reproduces byte-for-byte from the same seed.
 * Prints one CSV row per kernel:
 *
 *     side,kernel,size,ns_per_iter,input_fnv64,output_fnv64
 *
 * `side` is "libvpx"; the rvp9 side (`dsp-microbench vs_libvpx`) prints
 * "rvp9" with otherwise-identical columns. The audit driver
 * (scripts/run-kernel-audit.ps1) joins on (kernel,size), asserts
 * input/output hashes match (bit-exactness), then computes the ratio.
 *
 * Methodology mirrors `dsp-microbench`: minimum-of-batches after warmup.
 * Min is the most stable estimator for a CPU-bound microbench.
 *
 * Build: see Makefile. Requires mingw64 gcc + build/libvpx/libvpx.a.
 * Run pinned via scripts/bench-pin.ps1 or scripts/run-kernel-audit.ps1.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include "vpx_dsp_rtcd.h"

/* --------------------------- timing helpers ---------------------------- */

static double qpc_to_ns_scale(void) {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    return 1.0e9 / (double)f.QuadPart;
}
static int64_t qpc_now(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}

typedef void (*bench_fn)(void *ctx);
static double bench_min_ns(uint32_t batches, uint32_t iters, bench_fn f, void *ctx) {
    const double scale = qpc_to_ns_scale();
    for (uint32_t i = 0; i < iters; ++i) f(ctx);   /* warmup */
    double best = 1.0e30;
    for (uint32_t b = 0; b < batches; ++b) {
        int64_t t0 = qpc_now();
        for (uint32_t i = 0; i < iters; ++i) f(ctx);
        int64_t t1 = qpc_now();
        double ns = (double)(t1 - t0) * scale / (double)iters;
        if (ns < best) best = ns;
    }
    return best;
}

/* ---------------------------- LCG + FNV-1a ---------------------------- */
/* Mirrors the rvp9 side byte-for-byte. */

static uint32_t lcg_step(uint32_t *s) {
    *s = (*s) * 1664525u + 1013904223u;
    return (*s >> 24) & 0xFFu;
}
static uint64_t fnv1a64(const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)p[i];
        h *= 0x100000001B3ull;
    }
    return h;
}

/* IDCT input: N int32 values, low magnitude (post-quant-ish), range
 * [-512, 511]. */
static void fill_idct_input(int32_t *out, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        uint32_t lo = lcg_step(&s);
        uint32_t hi = lcg_step(&s);
        out[i] = (int32_t)((lo | (hi << 8)) & 0x3FFu) - 0x200;
    }
}
static void fill_bytes(uint8_t *out, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; ++i) out[i] = (uint8_t)lcg_step(&s);
}

#define BATCHES 200u

/* ------------------------------ IDCT family --------------------------- */
/* All inverse-DCT-add kernels share one signature:
 *   void f(const tran_low_t *input, uint8_t *dest, int stride);
 * tran_low_t == int32_t (libvpx built --enable-vp9-highbitdepth). */

typedef void (*idct_fn)(const tran_low_t *, uint8_t *, int);

typedef struct {
    idct_fn      fn;
    const int32_t *input;
    uint8_t      *dest;
    int           stride;
} idct_ctx;

static void idct_thunk(void *v) {
    idct_ctx *c = (idct_ctx *)v;
    c->fn(c->input, c->dest, c->stride);
}

/* NB: the libvpx kernels are runtime dispatch pointers (filled by
 * vpx_dsp_rtcd()), so they can't live in a static initializer. We pass
 * each kernel as a runtime argument instead of a const table. */
static void run_idct(idct_fn fn, const char *name, const char *size,
                     int n, int dim, uint32_t seed, uint32_t iters) {
    static int32_t input[1024];
    static uint8_t dest[32 * 48];        /* max: 32 rows, stride <= 48 */
    static uint8_t packed[32 * 32];
    const int stride = dim + 16;         /* same on rvp9 side */

    fill_idct_input(input, (size_t)n, seed);
    uint64_t in_hash = fnv1a64(input, (size_t)n * sizeof(int32_t));

    /* verify: zero dest, one call, hash the dim×dim footprint (no padding). */
    memset(dest, 0, sizeof dest);
    fn(input, dest, stride);
    for (int r = 0; r < dim; ++r)
        memcpy(packed + r * dim, dest + r * stride, (size_t)dim);
    uint64_t out_hash = fnv1a64(packed, (size_t)dim * dim);

    memset(dest, 0, sizeof dest);
    idct_ctx ctx = { fn, input, dest, stride };
    double ns = bench_min_ns(BATCHES, iters, idct_thunk, &ctx);
    printf("libvpx,%s,%s,%.2f,%016llx,%016llx\n",
           name, size, ns,
           (unsigned long long)in_hash, (unsigned long long)out_hash);
    fflush(stdout);
}

/* --------------------------- convolve8 family ------------------------- */
/* VP9 EIGHTTAP_REGULAR half-pel taps (= rvp9 EIGHTTAP_KERNEL[8]). */
static const int16_t kEighttapHalfPel[8] = { -1, 6, -19, 78, 78, -19, 6, -1 };

#define CONV_SRC_STRIDE 80

typedef struct {
    uint8_t        src_buf[CONV_SRC_STRIDE * CONV_SRC_STRIDE];
    uint8_t        dst[64 * 64];
    InterpKernel   filters;
    int            w, h;
    int            mode;   /* 0 = 2D, 1 = horiz, 2 = vert */
} conv_ctx;

static void conv_thunk(void *v) {
    conv_ctx *c = (conv_ctx *)v;
    /* libvpx shifts `src` to the filter footprint top-left internally; the
     * shift amount is mode-specific:
     *   2D    : convolve_horiz(src - 3*stride) then internal -3 cols  → -3,-3
     *   horiz : convolve_horiz only, internal -3 cols                  →  0,-3
     *   vert  : convolve_vert only,  internal -3 rows                  → -3, 0
     * We offset the caller `src` so libvpx's internal shift always lands
     * on src_buf[0] — the same footprint top-left the rvp9 side uses
     * (rvp9's `src` argument IS the footprint top-left, no internal shift). */
    const int s = CONV_SRC_STRIDE;
    const InterpKernel *f = (const InterpKernel *)&c->filters;
    switch (c->mode) {
        case 1: /* horiz: dst origin = footprint TL + 3 cols */
            vpx_convolve8_horiz(c->src_buf + 3, s, c->dst, 64, f,
                                0, 16, 0, 16, c->w, c->h);
            break;
        case 2: /* vert: dst origin = footprint TL + 3 rows */
            vpx_convolve8_vert(c->src_buf + 3 * s, s, c->dst, 64, f,
                               0, 16, 0, 16, c->w, c->h);
            break;
        default: /* 2D: dst origin = footprint TL + 3 rows + 3 cols */
            vpx_convolve8(c->src_buf + 3 * s + 3, s, c->dst, 64, f,
                          0, 16, 0, 16, c->w, c->h);
            break;
    }
}

typedef struct {
    const char *name;
    const char *size;
    int         w, h;
    int         mode;
    uint32_t    seed;
    uint32_t    iters;
} conv_case;

static void run_conv_case(const conv_case *cc) {
    static conv_ctx c;
    memset(&c, 0, sizeof c);
    c.w = cc->w; c.h = cc->h; c.mode = cc->mode;
    fill_bytes(c.src_buf, sizeof c.src_buf, cc->seed);
    memcpy(c.filters, kEighttapHalfPel, sizeof kEighttapHalfPel);
    uint64_t in_hash = fnv1a64(c.src_buf, sizeof c.src_buf);

    memset(c.dst, 0, sizeof c.dst);
    conv_thunk(&c);
    /* hash the w×h footprint (dst stride is exactly 64 here, but pack the
     * w×h region so non-64-wide blocks don't fold padding into the hash). */
    static uint8_t packed[64 * 64];
    for (int r = 0; r < cc->h; ++r) memcpy(packed + r * cc->w, c.dst + r * 64, (size_t)cc->w);
    uint64_t out_hash = fnv1a64(packed, (size_t)cc->w * cc->h);

    memset(c.dst, 0, sizeof c.dst);
    double ns = bench_min_ns(BATCHES, cc->iters, conv_thunk, &c);
    printf("libvpx,%s,%s,%.2f,%016llx,%016llx\n",
           cc->name, cc->size, ns,
           (unsigned long long)in_hash, (unsigned long long)out_hash);
    fflush(stdout);
}

static const conv_case kConvCases[] = {
    { "convolve8_2d_halfpel",    "8x8",   8,  8,  0, 0xAB08u, 16000u },
    { "convolve8_2d_halfpel",    "16x16", 16, 16, 0, 0xAB10u,  8000u },
    { "convolve8_2d_halfpel",    "32x32", 32, 32, 0, 0xAB20u,  3000u },
    { "convolve8_2d_halfpel",    "64x64", 64, 64, 0, 0xAA03u,  2000u },
    { "convolve8_horiz_halfpel", "16x16", 16, 16, 1, 0xAC10u, 12000u },
    { "convolve8_horiz_halfpel", "64x64", 64, 64, 1, 0xAC40u,  3000u },
    { "convolve8_vert_halfpel",  "16x16", 16, 16, 2, 0xAD10u, 12000u },
    { "convolve8_vert_halfpel",  "64x64", 64, 64, 2, 0xAD40u,  3000u },
};

/* ------------------------------- driver ------------------------------- */

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    /* Fill the libvpx dispatch table so the `vpx_*` pointers point at the
     * SSE2/AVX2 variant the host supports — what vpxdec runs at startup. */
    vpx_dsp_rtcd();

    run_idct(vpx_idct4x4_16_add,    "idct4x4_16_add",     "4x4",     16, 4,  0xAA10u, 16000u);
    run_idct(vpx_idct8x8_64_add,    "idct8x8_64_add",     "8x8",     64, 8,  0xAA11u, 12000u);
    run_idct(vpx_idct8x8_12_add,    "idct8x8_12_add",     "8x8",     64, 8,  0xAA12u, 12000u);
    run_idct(vpx_idct16x16_256_add, "idct16x16_256_add",  "16x16",  256, 16, 0xAA13u,  8000u);
    run_idct(vpx_idct16x16_10_add,  "idct16x16_10_add",   "16x16",  256, 16, 0xAA14u,  8000u);
    run_idct(vpx_idct16x16_38_add,  "idct16x16_38_add",   "16x16",  256, 16, 0xAA01u,  8000u);
    run_idct(vpx_idct32x32_1024_add,"idct32x32_1024_add", "32x32", 1024, 32, 0xAA15u,  2000u);
    run_idct(vpx_idct32x32_34_add,  "idct32x32_34_add",   "32x32", 1024, 32, 0xAA16u,  2000u);
    run_idct(vpx_idct32x32_135_add, "idct32x32_135_add",  "32x32", 1024, 32, 0xAA02u,  2000u);

    for (size_t i = 0; i < sizeof kConvCases / sizeof kConvCases[0]; ++i)
        run_conv_case(&kConvCases[i]);

    return 0;
}
