/* G-C ROI fixture generator.
 *
 * Mints a VP9 IVF that provably activates SEG_LVL_SKIP (roi.skip[seg])
 * and SEG_LVL_REF_FRAME (roi.ref_frame[seg]) via VP9E_SET_ROI_MAP,
 * mirroring third_party/libvpx/test/vp9_roi_test.cc RoiMaskBackgroundSkip:
 * realtime + cpu-used=7 + aq-mode=3, skip on segment 3, ref_frame on
 * segment 1. Writes <out.ivf>.
 *
 * Build (mingw64, against build_f02/libvpx.a):
 *   gcc gc_roi_fixture.c -I third_party/libvpx -I third_party/libvpx/build_f02 \
 *       third_party/libvpx/build_f02/libvpx.a -lm -o gc_roi_fixture.exe
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vpx/vpx_encoder.h"
#include "vpx/vp8cx.h"

#define W 160
#define H 120
#define NUM_FRAMES 60
#define FPS 30

static void mem_put_le16(uint8_t *p, unsigned v) {
  p[0] = (uint8_t)(v & 0xff);
  p[1] = (uint8_t)((v >> 8) & 0xff);
}
static void mem_put_le32(uint8_t *p, unsigned v) {
  p[0] = (uint8_t)(v & 0xff);
  p[1] = (uint8_t)((v >> 8) & 0xff);
  p[2] = (uint8_t)((v >> 16) & 0xff);
  p[3] = (uint8_t)((v >> 24) & 0xff);
}

static void write_ivf_file_header(FILE *f, unsigned w, unsigned h,
                                  unsigned num_frames) {
  uint8_t hdr[32];
  memcpy(hdr, "DKIF", 4);
  mem_put_le16(hdr + 4, 0);        /* version */
  mem_put_le16(hdr + 6, 32);       /* header size */
  memcpy(hdr + 8, "VP90", 4);      /* fourcc */
  mem_put_le16(hdr + 12, (unsigned short)w);
  mem_put_le16(hdr + 14, (unsigned short)h);
  mem_put_le32(hdr + 16, FPS);     /* timebase denom (rate) */
  mem_put_le32(hdr + 20, 1);       /* timebase numer (scale) */
  mem_put_le32(hdr + 24, num_frames);
  mem_put_le32(hdr + 28, 0);       /* unused */
  fwrite(hdr, 1, 32, f);
}

static void write_ivf_frame_header(FILE *f, int64_t pts, size_t sz) {
  uint8_t hdr[12];
  mem_put_le32(hdr, (unsigned)sz);
  mem_put_le32(hdr + 4, (unsigned)(pts & 0xffffffff));
  mem_put_le32(hdr + 8, (unsigned)(pts >> 32));
  fwrite(hdr, 1, 12, f);
}

/* Frame content: the LEFT third (segment 1, forced REF_FRAME) and the
 * RIGHT third (segment 3, forced SKIP) are STATIC across frames, so the
 * encoder's `content_state_sb <= kLowSadLowSumdiff` gate is satisfied and
 * it actually codes those superblocks on their forced segments (libvpx
 * forces seg=0 on superblocks with temporal change to avoid skip
 * artifacts, `vp9_encodeframe.c:5600`). Motion lives only in the center
 * third so the frame is genuinely inter-coded. */
static void fill_frame(vpx_image_t *img, int t) {
  unsigned int x, y;
  uint8_t *yp = img->planes[VPX_PLANE_Y];
  uint8_t *up = img->planes[VPX_PLANE_U];
  uint8_t *vp = img->planes[VPX_PLANE_V];
  const unsigned left_end = (unsigned)W / 3;
  const unsigned right_start = 2u * (unsigned)W / 3;
  for (y = 0; y < (unsigned)H; ++y) {
    for (x = 0; x < (unsigned)W; ++x) {
      if (x < left_end) {
        yp[x] = (uint8_t)((x + y) & 0xff);          /* static */
      } else if (x >= right_start) {
        yp[x] = (uint8_t)((x * 2 + y) & 0xff);        /* static */
      } else {
        yp[x] = (uint8_t)((x + y + t * 7) & 0xff);    /* moving */
      }
    }
    yp += img->stride[VPX_PLANE_Y];
  }
  for (y = 0; y < (unsigned)H / 2; ++y) {
    for (x = 0; x < (unsigned)W / 2; ++x) {
      /* keep chroma static so only luma center moves */
      up[x] = (uint8_t)((x * 2) & 0xff);
      vp[x] = (uint8_t)((y * 2) & 0xff);
    }
    up += img->stride[VPX_PLANE_U];
    vp += img->stride[VPX_PLANE_V];
  }
}

int main(int argc, char **argv) {
  const char *out = argc > 1 ? argv[1] : "gc_roi_fixture.ivf";
  vpx_codec_ctx_t codec;
  vpx_codec_enc_cfg_t cfg;
  vpx_image_t img;
  vpx_roi_map_t roi;
  vpx_codec_iface_t *iface = vpx_codec_vp9_cx();
  FILE *f;
  int frame, got = 0;

  if (vpx_codec_enc_config_default(iface, &cfg, 0)) {
    fprintf(stderr, "config_default failed\n");
    return 1;
  }
  cfg.g_w = W;
  cfg.g_h = H;
  cfg.g_timebase.num = 1;
  cfg.g_timebase.den = FPS;
  cfg.rc_target_bitrate = 200;
  cfg.g_lag_in_frames = 0;
  cfg.kf_max_dist = 9999;
  cfg.rc_end_usage = VPX_CBR;
  cfg.rc_min_quantizer = 0;
  cfg.rc_max_quantizer = 50;
  cfg.rc_buf_initial_sz = 500;
  cfg.rc_buf_optimal_sz = 500;
  cfg.rc_buf_sz = 1000;
  cfg.rc_dropframe_thresh = 0; /* never drop, so frame count is exact */

  if (!vpx_img_alloc(&img, VPX_IMG_FMT_I420, W, H, 1)) {
    fprintf(stderr, "img_alloc failed\n");
    return 1;
  }
  if (vpx_codec_enc_init(&codec, iface, &cfg, 0)) {
    fprintf(stderr, "enc_init failed: %s\n", vpx_codec_error(&codec));
    return 1;
  }

  /* ROI map: skip on segment 3, single-ref (LAST) on segment 1. */
  memset(&roi, 0, sizeof(roi));
  roi.enabled = 1;
  roi.rows = (cfg.g_h + 7) / 8;
  roi.cols = (cfg.g_w + 7) / 8;
  memset(roi.ref_frame, -1, sizeof(roi.ref_frame));
  /* Cyclic-refresh AQ (aq-mode=3) commandeers segments 0/1/2 for its
   * BASE/BOOST1/BOOST2 buckets, so put the forced REF_FRAME on a segment
   * it never touches (4) to keep it from being clobbered. SKIP stays on
   * the libvpx-required BACKGROUND_SEG_SKIP_ID == 3. */
  roi.ref_frame[4] = 1; /* LAST_FRAME on segment 4 (SEG_LVL_REF_FRAME) */
  roi.skip[3] = 1;      /* SEG_LVL_SKIP on segment 3 */
  roi.roi_map = (unsigned char *)calloc(roi.rows * roi.cols, 1);
  /* Assign segments across the frame: stripe segments 1 and 3 so both
   * features fire on real blocks. */
  {
    unsigned int i, j;
    for (i = 0; i < roi.rows; ++i) {
      for (j = 0; j < roi.cols; ++j) {
        unsigned idx = i * roi.cols + j;
        /* segment 1 (forced LAST) on the left half, segment 3 (skip)
         * on the right half, segment 0 elsewhere via a checker. */
        if (j < roi.cols / 3) {
          roi.roi_map[idx] = 4; /* static left third -> forced REF_FRAME */
        } else if (j > 2 * roi.cols / 3) {
          roi.roi_map[idx] = 3; /* static right third -> forced SKIP */
        } else {
          roi.roi_map[idx] = 0; /* moving center -> normal */
        }
      }
    }
  }

  f = fopen(out, "wb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", out);
    return 1;
  }
  write_ivf_file_header(f, W, H, NUM_FRAMES);

  for (frame = 0; frame < NUM_FRAMES; ++frame) {
    vpx_codec_iter_t iter = NULL;
    const vpx_codec_cx_pkt_t *pkt;
    fill_frame(&img, frame);

    if (frame == 0) {
      vpx_codec_control(&codec, VP8E_SET_CPUUSED, 7);
      vpx_codec_control(&codec, VP9E_SET_AQ_MODE, 3);
    }
    vpx_codec_control(&codec, VP9E_SET_ROI_MAP, &roi);

    if (vpx_codec_encode(&codec, &img, frame, 1, 0, VPX_DL_REALTIME)) {
      fprintf(stderr, "encode failed @%d: %s\n", frame,
              vpx_codec_error(&codec));
      return 1;
    }
    while ((pkt = vpx_codec_get_cx_data(&codec, &iter)) != NULL) {
      if (pkt->kind == VPX_CODEC_CX_FRAME_PKT) {
        write_ivf_frame_header(f, pkt->data.frame.pts, pkt->data.frame.sz);
        fwrite(pkt->data.frame.buf, 1, pkt->data.frame.sz, f);
        ++got;
      }
    }
  }
  /* flush */
  for (;;) {
    vpx_codec_iter_t iter = NULL;
    const vpx_codec_cx_pkt_t *pkt;
    if (vpx_codec_encode(&codec, NULL, frame, 1, 0, VPX_DL_REALTIME)) break;
    pkt = vpx_codec_get_cx_data(&codec, &iter);
    if (!pkt) break;
    {
      int any = 0;
      do {
        if (pkt->kind == VPX_CODEC_CX_FRAME_PKT) {
          write_ivf_frame_header(f, pkt->data.frame.pts, pkt->data.frame.sz);
          fwrite(pkt->data.frame.buf, 1, pkt->data.frame.sz, f);
          ++got;
          any = 1;
        }
        pkt = vpx_codec_get_cx_data(&codec, &iter);
      } while (pkt);
      if (!any) break;
    }
    ++frame;
  }

  fclose(f);
  vpx_codec_destroy(&codec);
  vpx_img_free(&img);
  free(roi.roi_map);
  fprintf(stderr, "wrote %d frames to %s\n", got, out);
  return 0;
}
