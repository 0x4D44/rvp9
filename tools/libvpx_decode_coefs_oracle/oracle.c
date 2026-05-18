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

  return match ? 0 : 1;
}
