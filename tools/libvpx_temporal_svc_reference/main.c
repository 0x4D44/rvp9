/*
 * Pinned-libvpx temporal-SVC reference encoder for rvp9 acceptance tests.
 *
 * vpxenc does not expose VP9 temporal-SVC configuration on its CLI. This
 * adapter deliberately stays narrow: one-pass realtime VP9, one spatial
 * layer, and libvpx's built-in 0101 / 0212 temporal modes. It links the
 * repository's exact pinned libvpx archive and writes a normal IVF stream.
 */

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vpx/vp8cx.h"
#include "vpx/vpx_encoder.h"
#include "vpx/vpx_image.h"

#define EXPECTED_LIBVPX_REVISION "12f3a2ac603e8f10742105519e0cd03c3b8f71dd"
#define MAX_Y4M_HEADER 4096

typedef struct Options {
  const char *input_path;
  const char *output_path;
  unsigned int target_bitrate_kbps;
  unsigned int cpu_used;
  unsigned int threads;
  unsigned int tile_columns;
  unsigned int temporal_layers;
  unsigned int aq_mode;
  unsigned int noise_sensitivity;
  unsigned int limit;
  unsigned int keyframe_interval;
  enum vpx_rc_mode end_usage;
} Options;

typedef struct Y4mInfo {
  unsigned int width;
  unsigned int height;
  unsigned int fps_num;
  unsigned int fps_den;
} Y4mInfo;

static void usage(FILE *out, const char *program) {
  fprintf(out,
          "Usage: %s --input INPUT.y4m --output OUTPUT.ivf \\\n+\n"
          "  --target-bitrate KBPS --cpu-used {6,7,8} \\\n+\n"
          "  --temporal-layers {2,3} [options]\n\n"
          "Options:\n"
          "  --end-usage {cbr,vbr}       default: vbr\n"
          "  --threads N                 default: 1\n"
          "  --tile-columns N            default: 0\n"
          "  --aq-mode {0,3}             default: 0\n"
          "  --noise-sensitivity N       default: 0\n"
          "  --limit N                   default: all frames\n"
          "  --keyframe-interval N       default: 60\n"
          "  --version                   print pinned revision\n",
          program);
}

static void die(const char *message) {
  fprintf(stderr, "vpx_temporal_svc_reference: %s\n", message);
  exit(EXIT_FAILURE);
}

static void die_errno(const char *operation, const char *path) {
  fprintf(stderr, "vpx_temporal_svc_reference: %s %s: %s\n", operation,
          path, strerror(errno));
  exit(EXIT_FAILURE);
}

static unsigned int parse_uint(const char *flag, const char *value) {
  char *end = NULL;
  unsigned long parsed;
  errno = 0;
  parsed = strtoul(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed > UINT_MAX) {
    fprintf(stderr, "vpx_temporal_svc_reference: invalid %s: %s\n", flag,
            value);
    exit(EXIT_FAILURE);
  }
  return (unsigned int)parsed;
}

static const char *required_value(int argc, char **argv, int *index) {
  if (*index + 1 >= argc) {
    fprintf(stderr, "vpx_temporal_svc_reference: %s requires a value\n",
            argv[*index]);
    exit(EXIT_FAILURE);
  }
  ++*index;
  return argv[*index];
}

static Options parse_options(int argc, char **argv) {
  Options options;
  int i;
  memset(&options, 0, sizeof(options));
  options.cpu_used = 7;
  options.threads = 1;
  options.end_usage = VPX_VBR;
  options.limit = UINT_MAX;
  options.keyframe_interval = 60;

  for (i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
      usage(stdout, argv[0]);
      exit(EXIT_SUCCESS);
    } else if (strcmp(arg, "--version") == 0) {
      printf("libvpx %s\n", EXPECTED_LIBVPX_REVISION);
      exit(EXIT_SUCCESS);
    } else if (strcmp(arg, "--input") == 0) {
      options.input_path = required_value(argc, argv, &i);
    } else if (strcmp(arg, "--output") == 0 || strcmp(arg, "-o") == 0) {
      options.output_path = required_value(argc, argv, &i);
    } else if (strcmp(arg, "--target-bitrate") == 0) {
      options.target_bitrate_kbps =
          parse_uint(arg, required_value(argc, argv, &i));
    } else if (strcmp(arg, "--cpu-used") == 0) {
      options.cpu_used = parse_uint(arg, required_value(argc, argv, &i));
    } else if (strcmp(arg, "--threads") == 0) {
      options.threads = parse_uint(arg, required_value(argc, argv, &i));
    } else if (strcmp(arg, "--tile-columns") == 0) {
      options.tile_columns = parse_uint(arg, required_value(argc, argv, &i));
    } else if (strcmp(arg, "--temporal-layers") == 0) {
      options.temporal_layers =
          parse_uint(arg, required_value(argc, argv, &i));
    } else if (strcmp(arg, "--aq-mode") == 0) {
      options.aq_mode = parse_uint(arg, required_value(argc, argv, &i));
    } else if (strcmp(arg, "--noise-sensitivity") == 0) {
      options.noise_sensitivity =
          parse_uint(arg, required_value(argc, argv, &i));
    } else if (strcmp(arg, "--limit") == 0) {
      options.limit = parse_uint(arg, required_value(argc, argv, &i));
    } else if (strcmp(arg, "--keyframe-interval") == 0) {
      options.keyframe_interval =
          parse_uint(arg, required_value(argc, argv, &i));
    } else if (strcmp(arg, "--end-usage") == 0) {
      const char *value = required_value(argc, argv, &i);
      if (strcmp(value, "cbr") == 0) {
        options.end_usage = VPX_CBR;
      } else if (strcmp(value, "vbr") == 0) {
        options.end_usage = VPX_VBR;
      } else {
        die("--end-usage must be cbr or vbr");
      }
    } else {
      fprintf(stderr, "vpx_temporal_svc_reference: unknown argument: %s\n",
              arg);
      usage(stderr, argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  if (options.input_path == NULL || options.output_path == NULL ||
      options.target_bitrate_kbps == 0 || options.temporal_layers == 0) {
    usage(stderr, argv[0]);
    die("--input, --output, --target-bitrate, and --temporal-layers are required");
  }
  if (options.cpu_used < 6 || options.cpu_used > 8)
    die("--cpu-used must be 6, 7, or 8");
  if (options.threads == 0) die("--threads must be greater than zero");
  if (options.tile_columns > 6) die("--tile-columns must be in 0..=6");
  if (options.temporal_layers < 2 || options.temporal_layers > 3)
    die("--temporal-layers must be 2 or 3");
  if (options.aq_mode != 0 && options.aq_mode != 3)
    die("--aq-mode must be 0 or 3");
  if (options.noise_sensitivity > 6)
    die("--noise-sensitivity must be in 0..=6");
  if (options.keyframe_interval == 0)
    die("--keyframe-interval must be greater than zero");
  return options;
}

static Y4mInfo read_y4m_header(FILE *input) {
  char line[MAX_Y4M_HEADER];
  char *token;
  Y4mInfo info;
  int chroma_ok = 1;
  memset(&info, 0, sizeof(info));
  if (fgets(line, sizeof(line), input) == NULL) die("cannot read Y4M header");
  if (strchr(line, '\n') == NULL) die("Y4M header exceeds 4095 bytes");
  token = strtok(line, " \r\n");
  if (token == NULL || strcmp(token, "YUV4MPEG2") != 0)
    die("input is not a Y4M stream");
  while ((token = strtok(NULL, " \r\n")) != NULL) {
    if (token[0] == 'W') {
      info.width = parse_uint("Y4M width", token + 1);
    } else if (token[0] == 'H') {
      info.height = parse_uint("Y4M height", token + 1);
    } else if (token[0] == 'F') {
      char *colon = strchr(token + 1, ':');
      if (colon == NULL) die("invalid Y4M framerate");
      *colon = '\0';
      info.fps_num = parse_uint("Y4M fps numerator", token + 1);
      info.fps_den = parse_uint("Y4M fps denominator", colon + 1);
    } else if (token[0] == 'C') {
      chroma_ok = strcmp(token, "C420") == 0 ||
                  strcmp(token, "C420jpeg") == 0 ||
                  strcmp(token, "C420mpeg2") == 0 ||
                  strcmp(token, "C420paldv") == 0;
    }
  }
  if (info.width == 0 || info.height == 0 || info.fps_num == 0 ||
      info.fps_den == 0)
    die("Y4M header is missing W, H, or F");
  if ((info.width & 1U) != 0 || (info.height & 1U) != 0)
    die("Y4M dimensions must be even");
  if (!chroma_ok) die("only 8-bit 4:2:0 Y4M input is supported");
  return info;
}

static int read_plane(FILE *input, uint8_t *plane, int stride,
                      unsigned int width, unsigned int height) {
  unsigned int row;
  for (row = 0; row < height; ++row) {
    if (fread(plane + (size_t)row * (size_t)stride, 1, width, input) !=
        width)
      return 0;
  }
  return 1;
}

static int read_y4m_frame(FILE *input, vpx_image_t *image,
                          const Y4mInfo *info) {
  char frame_header[MAX_Y4M_HEADER];
  if (fgets(frame_header, sizeof(frame_header), input) == NULL) {
    if (feof(input)) return 0;
    die("failed while reading Y4M frame header");
  }
  if (strchr(frame_header, '\n') == NULL)
    die("Y4M frame header exceeds 4095 bytes");
  if (strncmp(frame_header, "FRAME", 5) != 0)
    die("missing Y4M FRAME marker");
  if (!read_plane(input, image->planes[VPX_PLANE_Y],
                  image->stride[VPX_PLANE_Y], info->width, info->height) ||
      !read_plane(input, image->planes[VPX_PLANE_U],
                  image->stride[VPX_PLANE_U], info->width / 2,
                  info->height / 2) ||
      !read_plane(input, image->planes[VPX_PLANE_V],
                  image->stride[VPX_PLANE_V], info->width / 2,
                  info->height / 2))
    die("truncated Y4M frame payload");
  return 1;
}

static void put_le16(uint8_t *dst, uint16_t value) {
  dst[0] = (uint8_t)value;
  dst[1] = (uint8_t)(value >> 8);
}

static void put_le32(uint8_t *dst, uint32_t value) {
  dst[0] = (uint8_t)value;
  dst[1] = (uint8_t)(value >> 8);
  dst[2] = (uint8_t)(value >> 16);
  dst[3] = (uint8_t)(value >> 24);
}

static void put_le64(uint8_t *dst, uint64_t value) {
  put_le32(dst, (uint32_t)value);
  put_le32(dst + 4, (uint32_t)(value >> 32));
}

static void write_ivf_header(FILE *output, const Y4mInfo *info,
                             unsigned int frame_count) {
  uint8_t header[32];
  memset(header, 0, sizeof(header));
  memcpy(header, "DKIF", 4);
  put_le16(header + 6, 32);
  memcpy(header + 8, "VP90", 4);
  put_le16(header + 12, (uint16_t)info->width);
  put_le16(header + 14, (uint16_t)info->height);
  put_le32(header + 16, info->fps_num);
  put_le32(header + 20, info->fps_den);
  put_le32(header + 24, frame_count);
  if (fwrite(header, 1, sizeof(header), output) != sizeof(header))
    die("failed to write IVF header");
}

static void write_ivf_packet(FILE *output, const vpx_codec_cx_pkt_t *packet) {
  uint8_t record[12];
  if (packet->data.frame.sz > UINT32_MAX) die("encoded packet is too large");
  put_le32(record, (uint32_t)packet->data.frame.sz);
  put_le64(record + 4, (uint64_t)packet->data.frame.pts);
  if (fwrite(record, 1, sizeof(record), output) != sizeof(record) ||
      fwrite(packet->data.frame.buf, 1, packet->data.frame.sz, output) !=
          packet->data.frame.sz)
    die("failed to write IVF packet");
}

static unsigned int temporal_id(unsigned int frame_index,
                                unsigned int temporal_layers) {
  if (temporal_layers == 2) return frame_index & 1U;
  switch (frame_index & 3U) {
    case 0: return 0;
    case 2: return 1;
    default: return 2;
  }
}

static void configure_temporal_layers(vpx_codec_enc_cfg_t *cfg,
                                      unsigned int temporal_layers,
                                      unsigned int target_bitrate_kbps) {
  cfg->ss_number_layers = 1;
  cfg->ss_target_bitrate[0] = target_bitrate_kbps;
  cfg->ts_number_layers = temporal_layers;
  if (temporal_layers == 2) {
    cfg->ts_periodicity = 2;
    cfg->ts_layer_id[0] = 0;
    cfg->ts_layer_id[1] = 1;
    cfg->ts_rate_decimator[0] = 2;
    cfg->ts_rate_decimator[1] = 1;
    cfg->ts_target_bitrate[0] = cfg->layer_target_bitrate[0] =
        2U * target_bitrate_kbps / 3U;
    cfg->ts_target_bitrate[1] = cfg->layer_target_bitrate[1] =
        target_bitrate_kbps;
    cfg->temporal_layering_mode = VP9E_TEMPORAL_LAYERING_MODE_0101;
  } else {
    cfg->ts_periodicity = 4;
    cfg->ts_layer_id[0] = 0;
    cfg->ts_layer_id[1] = 2;
    cfg->ts_layer_id[2] = 1;
    cfg->ts_layer_id[3] = 2;
    cfg->ts_rate_decimator[0] = 4;
    cfg->ts_rate_decimator[1] = 2;
    cfg->ts_rate_decimator[2] = 1;
    cfg->ts_target_bitrate[0] = cfg->layer_target_bitrate[0] =
        target_bitrate_kbps / 2U;
    cfg->ts_target_bitrate[1] = cfg->layer_target_bitrate[1] =
        3U * target_bitrate_kbps / 4U;
    cfg->ts_target_bitrate[2] = cfg->layer_target_bitrate[2] =
        target_bitrate_kbps;
    cfg->temporal_layering_mode = VP9E_TEMPORAL_LAYERING_MODE_0212;
  }
}

static void check_codec(vpx_codec_ctx_t *codec, vpx_codec_err_t result,
                        const char *operation) {
  if (result != VPX_CODEC_OK) {
    fprintf(stderr, "vpx_temporal_svc_reference: %s: %s\n", operation,
            vpx_codec_error(codec));
    if (vpx_codec_error_detail(codec) != NULL)
      fprintf(stderr, "  detail: %s\n", vpx_codec_error_detail(codec));
    exit(EXIT_FAILURE);
  }
}

static unsigned int drain_packets(vpx_codec_ctx_t *codec, FILE *output,
                                  uint64_t *payload_bytes) {
  vpx_codec_iter_t iter = NULL;
  const vpx_codec_cx_pkt_t *packet;
  unsigned int count = 0;
  while ((packet = vpx_codec_get_cx_data(codec, &iter)) != NULL) {
    if (packet->kind != VPX_CODEC_CX_FRAME_PKT) continue;
    write_ivf_packet(output, packet);
    *payload_bytes += packet->data.frame.sz;
    ++count;
  }
  return count;
}

int main(int argc, char **argv) {
  const Options options = parse_options(argc, argv);
  FILE *input = fopen(options.input_path, "rb");
  FILE *output;
  Y4mInfo info;
  vpx_codec_ctx_t codec;
  vpx_codec_enc_cfg_t cfg;
  vpx_svc_extra_cfg_t svc_parameters;
  vpx_image_t raw;
  unsigned int frame_count = 0;
  unsigned int packet_count = 0;
  uint64_t payload_bytes = 0;
  unsigned int layer;

  if (input == NULL) die_errno("opening", options.input_path);
  info = read_y4m_header(input);
  if (info.width > UINT16_MAX || info.height > UINT16_MAX)
    die("IVF dimensions must fit in 16 bits");
  output = fopen(options.output_path, "wb+");
  if (output == NULL) die_errno("creating", options.output_path);
  write_ivf_header(output, &info, 0);

  memset(&codec, 0, sizeof(codec));
  if (vpx_codec_enc_config_default(vpx_codec_vp9_cx(), &cfg, 0) !=
      VPX_CODEC_OK)
    die("cannot obtain the default VP9 encoder configuration");
  cfg.g_w = info.width;
  cfg.g_h = info.height;
  cfg.g_timebase.num = (int)info.fps_den;
  cfg.g_timebase.den = (int)info.fps_num;
  cfg.g_threads = options.threads;
  cfg.g_profile = 0;
  cfg.g_bit_depth = VPX_BITS_8;
  cfg.g_input_bit_depth = 8;
  cfg.g_error_resilient = VPX_ERROR_RESILIENT_DEFAULT;
  cfg.g_pass = VPX_RC_ONE_PASS;
  cfg.g_lag_in_frames = 0;
  cfg.rc_end_usage = options.end_usage;
  cfg.rc_target_bitrate = options.target_bitrate_kbps;
  cfg.rc_dropframe_thresh = 0;
  cfg.kf_mode = VPX_KF_AUTO;
  cfg.kf_min_dist = options.keyframe_interval;
  cfg.kf_max_dist = options.keyframe_interval;
  configure_temporal_layers(&cfg, options.temporal_layers,
                            options.target_bitrate_kbps);

  check_codec(&codec,
              vpx_codec_enc_init(&codec, vpx_codec_vp9_cx(), &cfg, 0),
              "initializing encoder");
  check_codec(&codec,
              vpx_codec_control(&codec, VP8E_SET_CPUUSED, options.cpu_used),
              "setting cpu-used");
  check_codec(&codec,
              vpx_codec_control(&codec, VP9E_SET_AQ_MODE, options.aq_mode),
              "setting AQ mode");
  check_codec(&codec,
              vpx_codec_control(&codec, VP9E_SET_NOISE_SENSITIVITY,
                                options.noise_sensitivity),
              "setting noise sensitivity");
  check_codec(&codec,
              vpx_codec_control(&codec, VP9E_SET_TUNE_CONTENT,
                                VP9E_CONTENT_DEFAULT),
              "setting tune-content");
  check_codec(&codec,
              vpx_codec_control(&codec, VP9E_SET_TILE_COLUMNS,
                                options.tile_columns),
              "setting tile columns");
  check_codec(&codec,
              vpx_codec_control(&codec, VP9E_SET_ROW_MT,
                                options.threads > 1 ? 1 : 0),
              "setting row-mt");
  check_codec(&codec,
              vpx_codec_control(&codec, VP9E_SET_FRAME_PARALLEL_DECODING, 0),
              "setting frame-parallel decoding");
  check_codec(&codec, vpx_codec_control(&codec, VP9E_SET_SVC, 1),
              "enabling SVC");

  memset(&svc_parameters, 0, sizeof(svc_parameters));
  svc_parameters.scaling_factor_num[0] = 1;
  svc_parameters.scaling_factor_den[0] = 1;
  svc_parameters.temporal_layering_mode = cfg.temporal_layering_mode;
  for (layer = 0; layer < options.temporal_layers; ++layer) {
    svc_parameters.min_quantizers[layer] = (int)cfg.rc_min_quantizer;
    svc_parameters.max_quantizers[layer] = (int)cfg.rc_max_quantizer;
    svc_parameters.speed_per_layer[layer] = (int)options.cpu_used;
  }
  check_codec(&codec,
              vpx_codec_control(&codec, VP9E_SET_SVC_PARAMETERS,
                                &svc_parameters),
              "setting SVC parameters");

  memset(&raw, 0, sizeof(raw));
  if (vpx_img_alloc(&raw, VPX_IMG_FMT_I420, info.width, info.height, 32) ==
      NULL)
    die("cannot allocate input image");

  while (frame_count < options.limit && read_y4m_frame(input, &raw, &info)) {
    vpx_svc_layer_id_t layer_id;
    const unsigned int expected_id =
        temporal_id(frame_count, options.temporal_layers);
    check_codec(&codec,
                vpx_codec_encode(&codec, &raw, (vpx_codec_pts_t)frame_count, 1,
                                 0, VPX_DL_REALTIME),
                "encoding frame");
    memset(&layer_id, 0, sizeof(layer_id));
    check_codec(&codec,
                vpx_codec_control(&codec, VP9E_GET_SVC_LAYER_ID, &layer_id),
                "querying temporal layer id");
    if ((unsigned int)layer_id.temporal_layer_id != expected_id) {
      fprintf(stderr,
              "vpx_temporal_svc_reference: frame %u has temporal id %d, "
              "expected %u\n",
              frame_count, layer_id.temporal_layer_id, expected_id);
      exit(EXIT_FAILURE);
    }
    packet_count += drain_packets(&codec, output, &payload_bytes);
    ++frame_count;
  }

  check_codec(&codec,
              vpx_codec_encode(&codec, NULL, (vpx_codec_pts_t)frame_count, 1,
                               0, VPX_DL_REALTIME),
              "flushing encoder");
  packet_count += drain_packets(&codec, output, &payload_bytes);
  if (packet_count != frame_count)
    die("lag-zero encode did not produce exactly one packet per input frame");

  if (fseek(output, 0, SEEK_SET) != 0) die("cannot seek to IVF header");
  write_ivf_header(output, &info, packet_count);
  if (fclose(output) != 0) die_errno("closing", options.output_path);
  fclose(input);
  vpx_img_free(&raw);
  check_codec(&codec, vpx_codec_destroy(&codec), "destroying encoder");

  fprintf(stderr,
          "libvpx_svc revision=%s frames=%u payload_bytes=%" PRIu64
          " layers=%u\n",
          EXPECTED_LIBVPX_REVISION, frame_count, payload_bytes,
          options.temporal_layers);
  return EXIT_SUCCESS;
}
