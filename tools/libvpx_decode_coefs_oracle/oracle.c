/*
 * Stage 3 stretch M.8 — libvpx decode_coefs oracle.
 *
 * Independent C harness that runs libvpx's exact `decode_coefs` on the
 * snapshot captured from `vp90-2-02-size-32x32.webm` frame 1 sub-1
 * (the M-series failing block). Output is compared against rvp9's
 * production output to localize the M.6 transcription bug.
 *
 * Build / run: see ./build.sh and ./README.md.
 *
 * decode_coefs body (and its read_bool / read_coeff helpers) is copied
 * VERBATIM from libvpx `vp9/decoder/vp9_detokenize.c:32-261` (commit
 * v1.14.1) — BSD-3 + WebM patent grant, see ../third_party/libvpx/LICENSE
 * and PATENTS. Original copyright preserved.
 *
 * Original copyright:
 *   Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *   Use of this source code is governed by a BSD-style license
 *   that can be found in the libvpx LICENSE file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

/* Pull in libvpx via libvpx-stock's include paths (see build.sh -I flags). */
#include "vpx_dsp/bitreader.h"
#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_entropy.h"
#include "vp9/common/vp9_entropymode.h"
#include "vp9/common/vp9_scan.h"

#include "snapshot.h"

/* ===================================================================
 * decode_coefs — verbatim copy from libvpx vp9_detokenize.c:32-261.
 * Renamed to `oracle_decode_coefs_inner` so we can use it standalone.
 * Counts path is kept identical (always-NULL in our harness, but the
 * branches stay so the bit consumption order matches exactly).
 * =================================================================== */

#define EOB_CONTEXT_NODE 0
#define ZERO_CONTEXT_NODE 1
#define ONE_CONTEXT_NODE 2

#define INCREMENT_COUNT(token)                   \
  do {                                           \
    if (counts) ++coef_counts[band][ctx][token]; \
  } while (0)

static INLINE int oracle_read_bool(vpx_reader *r, int prob, BD_VALUE *value,
                                   int *count, unsigned int *range) {
  const unsigned int split = (*range * prob + (256 - prob)) >> CHAR_BIT;
  const BD_VALUE bigsplit = (BD_VALUE)split << (BD_VALUE_SIZE - CHAR_BIT);

  if (*count < 0) {
    r->value = *value;
    r->count = *count;
    vpx_reader_fill(r);
    *value = r->value;
    *count = r->count;
  }

  if (*value >= bigsplit) {
    *range = *range - split;
    *value = *value - bigsplit;
    {
      const int shift = vpx_norm[*range];
      *range <<= shift;
      *value <<= shift;
      *count -= shift;
    }
    return 1;
  }
  *range = split;
  {
    const int shift = vpx_norm[*range];
    *range <<= shift;
    *value <<= shift;
    *count -= shift;
  }
  return 0;
}

static INLINE int oracle_read_coeff(vpx_reader *r, const vpx_prob *probs, int n,
                                    BD_VALUE *value, int *count, unsigned int *range) {
  int i, val = 0;
  for (i = 0; i < n; ++i)
    val = (val << 1) | oracle_read_bool(r, probs[i], value, count, range);
  return val;
}

static int oracle_decode_coefs_inner(const MACROBLOCKD *xd, PLANE_TYPE type,
                                     tran_low_t *dqcoeff, TX_SIZE tx_size,
                                     const int16_t *dq, int ctx,
                                     const int16_t *scan, const int16_t *nb,
                                     vpx_reader *r) {
  FRAME_COUNTS *counts = xd->counts;
  const int max_eob = 16 << (tx_size << 1);
  const FRAME_CONTEXT *const fc = xd->fc;
  const int ref = is_inter_block(xd->mi[0]);
  int band, c = 0;
  const vpx_prob(*coef_probs)[COEFF_CONTEXTS][UNCONSTRAINED_NODES] =
      fc->coef_probs[tx_size][type][ref];
  const vpx_prob *prob;
  unsigned int(*coef_counts)[COEFF_CONTEXTS][UNCONSTRAINED_NODES + 1];
  unsigned int(*eob_branch_count)[COEFF_CONTEXTS];
  uint8_t token_cache[32 * 32];
  const uint8_t *band_translate = get_band_translate(tx_size);
  const int dq_shift = (tx_size == TX_32X32);
  int v;
  int16_t dqv = dq[0];
  const uint8_t *const cat6_prob = vp9_cat6_prob;
  const int cat6_bits = 14;
  BD_VALUE value = r->value;
  unsigned int range = r->range;
  int count = r->count;

  if (counts) {
    coef_counts = counts->coef[tx_size][type][ref];
    eob_branch_count = counts->eob_branch[tx_size][type][ref];
  }
  /* silence unused warnings when counts == NULL (our path) */
  (void)coef_counts;
  (void)eob_branch_count;

  while (c < max_eob) {
    int val = -1;
    band = *band_translate++;
    prob = coef_probs[band][ctx];
    if (counts) ++eob_branch_count[band][ctx];
    if (!oracle_read_bool(r, prob[EOB_CONTEXT_NODE], &value, &count, &range)) {
      INCREMENT_COUNT(EOB_MODEL_TOKEN);
      break;
    }

    while (!oracle_read_bool(r, prob[ZERO_CONTEXT_NODE], &value, &count, &range)) {
      INCREMENT_COUNT(ZERO_TOKEN);
      dqv = dq[1];
      token_cache[scan[c]] = 0;
      ++c;
      if (c >= max_eob) {
        r->value = value;
        r->range = range;
        r->count = count;
        return c;
      }
      ctx = get_coef_context(nb, token_cache, c);
      band = *band_translate++;
      prob = coef_probs[band][ctx];
    }

    if (oracle_read_bool(r, prob[ONE_CONTEXT_NODE], &value, &count, &range)) {
      const vpx_prob *p = vp9_pareto8_full[prob[PIVOT_NODE] - 1];
      INCREMENT_COUNT(TWO_TOKEN);
      if (oracle_read_bool(r, p[0], &value, &count, &range)) {
        if (oracle_read_bool(r, p[3], &value, &count, &range)) {
          token_cache[scan[c]] = 5;
          if (oracle_read_bool(r, p[5], &value, &count, &range)) {
            if (oracle_read_bool(r, p[7], &value, &count, &range)) {
              val = CAT6_MIN_VAL +
                    oracle_read_coeff(r, cat6_prob, cat6_bits, &value, &count, &range);
            } else {
              val = CAT5_MIN_VAL +
                    oracle_read_coeff(r, vp9_cat5_prob, 5, &value, &count, &range);
            }
          } else if (oracle_read_bool(r, p[6], &value, &count, &range)) {
            val = CAT4_MIN_VAL +
                  oracle_read_coeff(r, vp9_cat4_prob, 4, &value, &count, &range);
          } else {
            val = CAT3_MIN_VAL +
                  oracle_read_coeff(r, vp9_cat3_prob, 3, &value, &count, &range);
          }
        } else {
          token_cache[scan[c]] = 4;
          if (oracle_read_bool(r, p[4], &value, &count, &range)) {
            val = CAT2_MIN_VAL +
                  oracle_read_coeff(r, vp9_cat2_prob, 2, &value, &count, &range);
          } else {
            val = CAT1_MIN_VAL +
                  oracle_read_coeff(r, vp9_cat1_prob, 1, &value, &count, &range);
          }
        }
        v = (val * dqv) >> dq_shift;
      } else {
        if (oracle_read_bool(r, p[1], &value, &count, &range)) {
          token_cache[scan[c]] = 3;
          v = ((3 + oracle_read_bool(r, p[2], &value, &count, &range)) * dqv) >>
              dq_shift;
        } else {
          token_cache[scan[c]] = 2;
          v = (2 * dqv) >> dq_shift;
        }
      }
    } else {
      INCREMENT_COUNT(ONE_TOKEN);
      token_cache[scan[c]] = 1;
      v = dqv >> dq_shift;
    }
    if (oracle_read_bool(r, 128, &value, &count, &range)) {
      dqcoeff[scan[c]] = (tran_low_t)-v;
    } else {
      dqcoeff[scan[c]] = (tran_low_t)v;
    }
    ++c;
    ctx = get_coef_context(nb, token_cache, c);
    dqv = dq[1];
  }

  r->value = value;
  r->range = range;
  r->count = count;
  return c;
}

/* ===================================================================
 * Oracle entry point — minimal MACROBLOCKD construction + vpx_reader
 * pre-state population from a captured snapshot.
 * =================================================================== */

typedef struct {
  int eob;
  int32_t dq_coefs[16];      /* sized for TX_4X4; harness path */
} OracleResult;

/* Run libvpx's decode_coefs on a snapshot. Mirrors the failing-block
 * inputs from M.6: TX_4X4, intra, plane=Y, DCT_ADST (→ col_scan_4x4),
 * initial_pt, dq=(177,231), captured reader pre-state, and the actual
 * compressed bitstream slice the production reader was reading from. */
static OracleResult oracle_decode_coefs(void) {
  OracleResult out;
  memset(&out, 0, sizeof(out));

  /* --- 1. Reader: directly populate registers + buffer pointers from
     the snapshot's pre-state. NOTE: libvpx's `r->buffer` is the
     CURRENT position pointer (advanced as bytes are consumed by
     vpx_reader_fill), so we set it to snap_buffer + SNAP_PRE_POS. */
  vpx_reader r;
  memset(&r, 0, sizeof(r));
  r.value       = SNAP_PRE_VALUE;
  r.range       = SNAP_PRE_RANGE;
  r.count       = SNAP_PRE_COUNT;
  r.buffer      = snap_buffer + SNAP_PRE_POS;
  r.buffer_end  = snap_buffer + SNAP_BUFFER_LEN;
  r.decrypt_cb  = NULL;
  r.decrypt_state = NULL;

  /* --- 2. MACROBLOCKD: only three fields are read by decode_coefs.
     `mi[0]->ref_frame[0]` (via is_inter_block), `fc->coef_probs`, and
     `counts` (NULL → no count updates).  */
  MODE_INFO mi_storage;
  memset(&mi_storage, 0, sizeof(mi_storage));
  mi_storage.ref_frame[0] = INTRA_FRAME;  /* 0 → is_inter_block returns 0 */
  mi_storage.ref_frame[1] = INTRA_FRAME;

  MODE_INFO *mi_ptr = &mi_storage;
  MODE_INFO **mi_grid = &mi_ptr;

  /* FRAME_CONTEXT: only `coef_probs[TX_4X4][PLANE_TYPE_Y][REF_INTRA]`
     is touched. Allocate the whole struct (zeroed) and fill the one
     slice that matters. */
  FRAME_CONTEXT *fc = (FRAME_CONTEXT *)calloc(1, sizeof(FRAME_CONTEXT));
  if (!fc) { fprintf(stderr, "calloc FRAME_CONTEXT failed\n"); exit(1); }
  memcpy(fc->coef_probs[TX_4X4][PLANE_TYPE_Y][/*ref=*/0],
         snap_coef_probs,
         sizeof(snap_coef_probs));

  MACROBLOCKD xd;
  memset(&xd, 0, sizeof(xd));
  xd.mi     = mi_grid;
  xd.fc     = fc;
  xd.counts = NULL;
  /* CONFIG_VP9_HIGHBITDEPTH=0 in libvpx-stock — no xd.bd field. */
  xd.lossless = 0;

  /* --- 3. Scan order: TX_4X4 + tx_type=DCT_ADST → col_scan_4x4. */
  const ScanOrder *sc = &vp9_scan_orders[TX_4X4][DCT_ADST];

  /* --- 4. Dequant array — [dc, ac]. */
  int16_t dq[2] = { SNAP_DQ_DC, SNAP_DQ_AC };

  /* --- 5. Output buffer (tran_low_t = int16_t when HBD is off; safe
     to also accept int32_t — we'll widen on copy-out). */
  tran_low_t dqcoeff[16];
  memset(dqcoeff, 0, sizeof(dqcoeff));

  /* --- 6. Run libvpx decode_coefs. */
  int eob = oracle_decode_coefs_inner(&xd, PLANE_TYPE_Y, dqcoeff,
                                      TX_4X4, dq, SNAP_INITIAL_PT,
                                      sc->scan, sc->neighbors, &r);

  out.eob = eob;
  for (int i = 0; i < 16; ++i) out.dq_coefs[i] = (int32_t)dqcoeff[i];

  free(fc);
  return out;
}

/* ===================================================================
 * main: run + diff.
 * =================================================================== */

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  printf("=== Stage 3 stretch M.8 oracle ===\n");
  printf("Snapshot: vp90-2-02-size-32x32.webm frame 1 sub-1 (Tx4x4, "
         "intra, Y plane, DCT_ADST → col_scan_4x4, "
         "initial_pt=%d, dq=(%d,%d))\n",
         SNAP_INITIAL_PT, SNAP_DQ_DC, SNAP_DQ_AC);
  printf("Pre-state: range=%u value=0x%016llx count=%d pos=%u\n",
         (unsigned)SNAP_PRE_RANGE,
         (unsigned long long)SNAP_PRE_VALUE,
         SNAP_PRE_COUNT,
         (unsigned)SNAP_PRE_POS);

  OracleResult libvpx_out = oracle_decode_coefs();

  printf("\nlibvpx decode_coefs:\n");
  printf("  eob      = %d\n", libvpx_out.eob);
  printf("  dq_coefs = [");
  for (int i = 0; i < 16; ++i) {
    printf("%s%d", i == 0 ? "" : ", ", libvpx_out.dq_coefs[i]);
  }
  printf("]\n");

  printf("\nrvp9 production:\n");
  printf("  eob      = %d\n", SNAP_PROD_EOB);
  printf("  dq_coefs = [");
  for (int i = 0; i < 16; ++i) {
    printf("%s%d", i == 0 ? "" : ", ", snap_prod_dq[i]);
  }
  printf("]\n");

  int match = (libvpx_out.eob == SNAP_PROD_EOB);
  for (int i = 0; i < 16 && match; ++i) {
    if (libvpx_out.dq_coefs[i] != snap_prod_dq[i]) match = 0;
  }

  printf("\n=== %s ===\n", match ? "MATCH" : "DIVERGE");

  if (!match) {
    printf("First diff (by index): ");
    for (int i = 0; i < 16; ++i) {
      if (libvpx_out.dq_coefs[i] != snap_prod_dq[i]) {
        printf("[%d] libvpx=%d rvp9=%d\n",
               i, libvpx_out.dq_coefs[i], snap_prod_dq[i]);
        break;
      }
    }
    if (libvpx_out.eob != SNAP_PROD_EOB) {
      printf("eob diff: libvpx=%d rvp9=%d\n", libvpx_out.eob, SNAP_PROD_EOB);
    }
  }

  /* Additional probe — Stage 3 stretch M.8: if decode_coefs MATCHES,
     the bug must be downstream. Run libvpx's vp9_iht4x4_16_add_c on
     the libvpx-decoded dq_coefs (DCT_ADST) into a zero dest, and print
     the resulting residual. That tells us what libvpx's iht4x4 would
     emit for this block — directly comparable to M.1's claim of
     `[+02, +02, -0e, -06]`. If libvpx's iht4x4 ALSO gives `[+0f, +09,
     -10, +03]`, then M.1's claim must have come from a different
     bitstream or M.1 is the source of confusion. If libvpx's iht4x4
     gives `[+02, +02, -0e, -06]`, then our iht4x4 has a hidden bug
     that M.3's differential missed. */
  {
    tran_low_t input[16];
    for (int i = 0; i < 16; ++i) input[i] = (tran_low_t)libvpx_out.dq_coefs[i];

    /* Print the input as libvpx sees it after `tran_low_t = int16_t`
       coercion (no overflow expected for our inputs, but document the
       fact). */
    printf("\nlibvpx input (tran_low_t = int16_t in libvpx-stock build):\n  [");
    for (int i = 0; i < 16; ++i)
      printf("%s%d", i == 0 ? "" : ", ", (int)input[i]);
    printf("]\n");

    /* Run the row pass standalone to expose the intermediate matrix —
       this is what libvpx writes into `tran_low_t out[16]` between row
       and column pass. If our Rust impl's intermediate differs from
       this, the bug is in our row-pass (idct4). If it agrees, the bug
       is in our column-pass (iadst4) or the residual-add. */
    extern void idct4_c(const tran_low_t *input, tran_low_t *output);
    extern void iadst4_c(const tran_low_t *input, tran_low_t *output);
    tran_low_t intermediate[16];
    for (int r = 0; r < 4; ++r) {
      idct4_c(&input[r * 4], &intermediate[r * 4]);  /* DctAdst: rows = idct4 */
    }
    printf("\nlibvpx intermediate after row-pass (idct4 × 4 rows):\n");
    for (int row = 0; row < 4; ++row) {
      printf("  row %d: [", row);
      for (int col = 0; col < 4; ++col) {
        printf("%s%6d", col == 0 ? "" : ", ", (int)intermediate[row * 4 + col]);
      }
      printf("]\n");
    }

    /* And the col-pass output BEFORE rounding+clip — for our diff. */
    printf("\nlibvpx col-pass output (iadst4 of each column, raw):\n");
    for (int i = 0; i < 4; ++i) {
      tran_low_t temp_in[4], temp_out[4];
      for (int j = 0; j < 4; ++j) temp_in[j] = intermediate[j * 4 + i];
      iadst4_c(temp_in, temp_out);
      printf("  col %d: in=[%d, %d, %d, %d]  out=[%d, %d, %d, %d]\n",
             i,
             (int)temp_in[0], (int)temp_in[1], (int)temp_in[2], (int)temp_in[3],
             (int)temp_out[0], (int)temp_out[1], (int)temp_out[2], (int)temp_out[3]);
    }

    /* Start dest at 128 — gives us residual = dest[i] - 128 (no clip
       issues for residuals in [-128, +127]). */
    uint8_t dest[4 * 4];
    for (int i = 0; i < 16; ++i) dest[i] = 128;

    extern void vp9_iht4x4_16_add_c(const tran_low_t *input, uint8_t *dest,
                                    int stride, int tx_type);
    vp9_iht4x4_16_add_c(input, dest, /*stride=*/4, /*tx_type=*/DCT_ADST);

    printf("\nlibvpx vp9_iht4x4_16_add_c(DctAdst, dq_coefs above, dest=128):\n");
    printf("  residual (dest - 128) =\n");
    for (int row = 0; row < 4; ++row) {
      printf("    [");
      for (int col = 0; col < 4; ++col) {
        int r = (int)dest[row * 4 + col] - 128;
        printf("%s%+4d", col == 0 ? "" : ", ", r);
      }
      printf("]\n");
    }

    /* Sanity check — apply col_pass output manually and compare to
       what vp9_iht4x4_16_add_c actually wrote. */
    printf("\n[debug] manual ROUND_POWER_OF_TWO of col_pass_out by 4:\n");
    for (int i = 0; i < 4; ++i) {
      tran_low_t temp_in[4], temp_out[4];
      for (int j = 0; j < 4; ++j) temp_in[j] = intermediate[j * 4 + i];
      iadst4_c(temp_in, temp_out);
      printf("  col %d: round = [", i);
      for (int j = 0; j < 4; ++j) {
        int rnd = (temp_out[j] + (1 << 3)) >> 4;
        printf("%s%+5d", j == 0 ? "" : ", ", rnd);
      }
      printf("]\n");
    }
  }

  /* ===================================================================
   * M.9 probe — run libvpx's pred[mode][TX_4X4] on the captured sub-1
   * inputs and print each row, then add the captured residual. This
   * tells us EXACTLY which mode libvpx is using for the sub-1 block.
   * Inputs captured 2026-05-18 from rvp9 trace for vp90-2-02-size-32x32
   * frame 1 sub-1: above_left=127, above=[127,127,127,127,127,127,127,127],
   * left=[34, 218, 103, 113]. The residual from the iht above is
   * [[+2,+2,-14,-3],[-42,-58,-13,-28],[+17,+26,-8,+21],[+33,+49,+60,+38]].
   * Expected reconstruction (from vpxdec): row0 = [36,36,20,28]
   *                                        row1 = [176,160,205,190]
   *                                        row2 = [120,129,95,124]
   *                                        row3 = [146,162,173,151]
   * So libvpx's pred row 0 must be [34,34,34,31] — and we need to find
   * which mode produces that.
   * =================================================================== */
  {
    extern void vpx_dc_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride,
                                       const uint8_t *above, const uint8_t *left);
    extern void vpx_dc_left_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride,
                                            const uint8_t *above, const uint8_t *left);
    extern void vpx_dc_top_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride,
                                           const uint8_t *above, const uint8_t *left);
    extern void vpx_dc_128_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride,
                                           const uint8_t *above, const uint8_t *left);
    extern void vpx_v_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride,
                                      const uint8_t *above, const uint8_t *left);
    extern void vpx_h_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride,
                                      const uint8_t *above, const uint8_t *left);
    extern void vpx_tm_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride,
                                       const uint8_t *above, const uint8_t *left);
    extern void vpx_d45_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride,
                                        const uint8_t *above, const uint8_t *left);
    extern void vpx_d135_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride,
                                         const uint8_t *above, const uint8_t *left);
    extern void vpx_d117_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride,
                                         const uint8_t *above, const uint8_t *left);
    extern void vpx_d153_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride,
                                         const uint8_t *above, const uint8_t *left);
    extern void vpx_d207_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride,
                                         const uint8_t *above, const uint8_t *left);
    extern void vpx_d63_predictor_4x4_c(uint8_t *dst, ptrdiff_t stride,
                                        const uint8_t *above, const uint8_t *left);

    /* The captured inputs from the rvp9 trace for sub-1.  The above_data
     * layout matches libvpx: above_data[16..] is the above row of length
     * 2*bs; above_data[15] is above_left.  We pass `above = above_data+16`. */
    uint8_t above_data[8 + 16];
    memset(above_data, 0, sizeof(above_data));
    above_data[15] = 127;  /* above_left */
    for (int i = 0; i < 8; ++i) above_data[16 + i] = 127;  /* above + above-right */
    const uint8_t *above = above_data + 16;
    uint8_t left[4] = {34, 218, 103, 113};

    struct { const char *name;
             void (*pred)(uint8_t *, ptrdiff_t, const uint8_t *, const uint8_t *); }
    modes[] = {
      {"DC",   vpx_dc_predictor_4x4_c},
      {"DC_L", vpx_dc_left_predictor_4x4_c},
      {"DC_T", vpx_dc_top_predictor_4x4_c},
      {"DC128",vpx_dc_128_predictor_4x4_c},
      {"V",    vpx_v_predictor_4x4_c},
      {"H",    vpx_h_predictor_4x4_c},
      {"TM",   vpx_tm_predictor_4x4_c},
      {"D45",  vpx_d45_predictor_4x4_c},
      {"D135", vpx_d135_predictor_4x4_c},
      {"D117", vpx_d117_predictor_4x4_c},
      {"D153", vpx_d153_predictor_4x4_c},
      {"D207", vpx_d207_predictor_4x4_c},
      {"D63",  vpx_d63_predictor_4x4_c},
    };

    int target_row0[4] = {34, 34, 34, 31};
    printf("\n=== M.9 probe — libvpx intra predictors for sub-1 ===\n");
    printf("  above_left=%u above=[%u,%u,%u,%u,%u,%u,%u,%u] left=[%u,%u,%u,%u]\n",
           above_data[15],
           above[0], above[1], above[2], above[3],
           above[4], above[5], above[6], above[7],
           left[0], left[1], left[2], left[3]);
    printf("  TARGET row 0 = [34, 34, 34, 31] (from libvpx_expected - residual)\n");
    for (size_t mi = 0; mi < sizeof(modes)/sizeof(modes[0]); ++mi) {
      uint8_t dst[16];
      memset(dst, 0xCC, sizeof(dst));
      modes[mi].pred(dst, /*stride=*/4, above, left);
      printf("  %-6s row 0=[%3u,%3u,%3u,%3u] %s\n",
             modes[mi].name, dst[0], dst[1], dst[2], dst[3],
             ((int)dst[0]==target_row0[0] && (int)dst[1]==target_row0[1] &&
              (int)dst[2]==target_row0[2] && (int)dst[3]==target_row0[3])
                 ? " <-- MATCHES TARGET" : "");
    }
  }

  /* ===================================================================
   * Stage 3 stretch M.13 probe — V block at MI(1, 3) sub-2 (= pixel
   * (24, 12)) on vp90-2-02-size-32x32.webm frame 1.
   *
   * Captured via `RVP9_DEBUG_DQCOEFS=all`:
   *   dq_coefs = [0, 231, -231, 231, -231, 0, 231, 0, 0, 0, 0, 0,
   *               231, 0, 0, 0]
   *   eob = 11, tx_type = ADST_DCT, dq = (177, 231).
   *
   * V mode prediction with above=[88, 96, 101, 109] replicates above
   * to every row of the 4×4 block. Adding libvpx's `vp9_iht4x4_16_add_c
   * (input, dest, 4, ADST_DCT)` yields the pre-LF reconstruction at
   * (col 24..27, row 12..15).
   *
   * rvp9's standalone iht test (`m13_v_block_24_12_iht4x4_probe` in
   * `crates/rvp9-core/src/transforms/inverse.rs`) produces row 3 =
   * [93, 112, 131, 79]. Final pre-LF at (col=27, row=15) = 79.
   *
   * The downstream cascade (block (24, 16) D63 reads above[3] = 79 →
   * its row 2 col 3 = above[3] = 79; expected post-LF = 93 implies
   * libvpx's above[3] for that block = 84). So this probe pins
   * libvpx's iht output at row 3 col 3 — if libvpx also outputs 79
   * here, the cascade is in our LF (not pre-LF). If libvpx outputs
   * 84, the cascade is in our iht4x4(AdstDct) on this exact pattern.
   * =================================================================== */
  {
    extern void vp9_iht4x4_16_add_c(const tran_low_t *input, uint8_t *dest,
                                    int stride, int tx_type);
    tran_low_t v_input[16] = {
        0, 231, -231, 231, -231, 0, 231, 0,
        0, 0, 0, 0, 231, 0, 0, 0
    };
    uint8_t v_dest[16] = {
        88, 96, 101, 109,
        88, 96, 101, 109,
        88, 96, 101, 109,
        88, 96, 101, 109,
    };
    vp9_iht4x4_16_add_c(v_input, v_dest, /*stride=*/4, /*tx_type=*/ADST_DCT);

    printf("\n=== M.13 V-block (24,12) iht4x4 probe ===\n");
    printf("Inputs: dq_coefs=[0,231,-231,231,-231,0,231,0,0,0,0,0,231,0,0,0]\n");
    printf("        above=[88,96,101,109] (V mode → replicate to all 4 rows)\n");
    printf("        tx_type=ADST_DCT\n");
    printf("libvpx vp9_iht4x4_16_add_c output:\n");
    for (int row = 0; row < 4; ++row) {
      printf("  row %d: [%3u, %3u, %3u, %3u]\n",
             row,
             v_dest[row * 4 + 0], v_dest[row * 4 + 1],
             v_dest[row * 4 + 2], v_dest[row * 4 + 3]);
    }
    printf("Compare row 3 col 3 (= pre-LF col=27, row=15):\n");
    printf("  libvpx output = %u\n", v_dest[15]);
    printf("  rvp9 output   = 79 (from m13_v_block_24_12_iht4x4_probe)\n");
    if (v_dest[15] == 79) {
      printf("  -> MATCH. Cascade is NOT in iht4x4(AdstDct).\n");
    } else {
      printf("  -> DIVERGE. Cascade IS in our iht4x4(AdstDct).\n");
    }
  }

  return match ? 0 : 1;
}
