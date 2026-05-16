# Bitstream, entropy coder, and probability tables in libvpx VP9

Specialist: A4. Pinned at `third_party/libvpx/` v1.14.1 (commit 12f3a2a).
Every claim is cited to `third_party/libvpx/<file>:<line>`. All tables are
transcribed verbatim from the source.

Cross-references:
- A1 (architecture) owns `VP9_COMP`, `VP9_COMMON`, the tile-thread fanout,
  and the frame loop. This doc consumes those as inputs.
- A2 owns the mode-decision producers that *fill* the counts; A4 owns the
  sink (`pack_inter_mode_mvs`, `pack_mb_tokens`, etc.).
- A3 owns motion search; A4 owns the MV *coding* (`vp9_encode_mv`,
  `nmv_context`, `vp9_inc_mv`).

A note on file naming: the prompt referenced legacy `vp9_boolhuff.{c,h}` /
`vp9_reader.{c,h}` filenames. As of v1.14.1 the boolean coder has been
hoisted into the codec-agnostic `vpx_dsp/` layer and lives in
`vpx_dsp/bitwriter.{c,h}` (encoder) and `vpx_dsp/bitreader.{c,h}` (decoder).
This doc references the current location.

---

## 1. Scope

In scope:

- The complete VP9 bitstream layout produced by `vp9_pack_bitstream` —
  superframe index, uncompressed header, compressed header, tile data
  (`third_party/libvpx/vp9/encoder/vp9_bitstream.c:1396`).
- The two parallel coders used by VP9:
  - The **fixed-bit "raw" coder** (`vpx_write_bit_buffer` /
    `vpx_read_bit_buffer`, `vpx_dsp/bitwriter_buffer.{c,h}` and
    `vpx_dsp/bitreader_buffer.{c,h}`), used for the uncompressed header.
  - The **boolean (arithmetic) coder** (`vpx_writer` / `vpx_reader`,
    `vpx_dsp/bitwriter.{c,h}` and `vpx_dsp/bitreader.{c,h}`), used for
    the compressed header and tile data.
- Tree-based coding (`vp9_treewriter.{c,h}`, `vpx_read_tree` in
  `vpx_dsp/bitreader.h:150`).
- Every load-bearing probability table and tree shipped with libvpx, in
  full and verbatim:
  - Intra mode probabilities and trees (`vp9_entropymode.c`).
  - Inter mode probabilities (`vp9_entropymode.c`).
  - Partition probabilities (`vp9_entropymode.c`).
  - Skip / intra-inter / comp-inter / ref-frame / tx-size /
    switchable-interp probabilities (`vp9_entropymode.c`).
  - Coefficient model (`vp9_entropy.{c,h}`): the 4D `default_coef_probs_*`
    tables, the `vp9_pareto8_full` 255-row extension table, the
    `vp9_coef_tree` / `vp9_coef_con_tree`, all category extra-bit probs,
    band translation tables, energy class table.
  - MV probability model (`vp9_entropymv.{c,h}`): `default_nmv_context`,
    joint/class/class0/fp trees.
  - Segmentation tree (`vp9_seg_common.c`) and segmentation header
    probability slot conventions.
- Scan orders, inverse-scan ("iscan") tables, and neighbor tables for each
  transform size × transform type (`vp9_scan.c`).
- The coefficient context / band / coefficient-context-count machinery —
  how context is derived per token (`vp9_entropy.h`, `vp9_scan.h`).
- The probability-update protocols on top of the boolean coder:
  - Coefficient update (`update_coef_probs_common`,
    `vp9_bitstream.c:545`).
  - Sub-exponential probability diff coding (`vp9_subexp.c`,
    `vp9_dsubexp.c`).
  - `merge_probs` / `mode_mv_merge_probs` adaptation (`vpx_dsp/prob.h:69`).
- The bit-exact contract between encoder and decoder — what invariants
  must hold for byte-identical output (PLAN.md D8: byte parity is a
  *debugging* tool, not a release gate; BD-rate is the gate).

Out of scope (covered by the named specialist):
- The forward and inverse transforms themselves (Stage 3 LLDs).
- How the encoder *decides* mode / partition / MV (A2, A3).
- Rate control / AQ / denoiser (A5).
- Threading orchestration above the tile boundary (A6, though we
  document the bitstream-level effect of tile threading here).

---

## 2. Data structures

All structs reproduced verbatim from libvpx with file:line citations. Field
semantics are stated where they are not obvious from name.

### 2.1 The boolean coder writer — `vpx_writer`

`third_party/libvpx/vpx_dsp/bitwriter.h:28-40`:

```c
typedef struct vpx_writer {
  unsigned int lowvalue;
  unsigned int range;
  int count;
  // Whether there has been an error.
  int error;
  // We maintain the invariant that pos <= size, i.e., we never write beyond
  // the end of the buffer. If pos would be incremented to be greater than
  // size, leave pos unchanged and set error to 1.
  unsigned int pos;
  unsigned int size;
  uint8_t *buffer;
} vpx_writer;
```

Fields:

- `lowvalue` (32-bit) — bottom of the current interval, shifted left so
  the next output byte is in bits `[31..24]`.
- `range` (32-bit, but always in `[128, 255]` after normalization) — width
  of the current interval, stored as the upper byte of the implicit 16-bit
  range used by the arithmetic-coder math.
- `count` (signed) — number of bits in `lowvalue` waiting to be emitted.
  Initialized to `-24` so the first emitting condition triggers after 24
  more bits of work have accumulated (one byte then needs to come out).
  See `vpx_start_encode` (`vpx_dsp/bitwriter.c:20`).
- `error` — sticky once set, indicates buffer overrun.
- `pos`, `size` — byte position and buffer cap; `pos <= size` is invariant.
- `buffer` — the caller-supplied output byte buffer; not owned.

Lifetime: created and destroyed within `encode_tile_worker`
(`vp9/encoder/vp9_bitstream.c:942`) per tile, plus a separate one for the
compressed header (`vp9_bitstream.c:1322`). Not thread-shared.

### 2.2 The boolean coder reader — `vpx_reader`

`third_party/libvpx/vpx_dsp/bitreader.h:40-50`:

```c
typedef struct {
  // Be careful when reordering this struct, it may impact the cache negatively.
  BD_VALUE value;
  unsigned int range;
  int count;
  const uint8_t *buffer_end;
  const uint8_t *buffer;
  vpx_decrypt_cb decrypt_cb;
  void *decrypt_state;
  uint8_t clear_buffer[sizeof(BD_VALUE) + 1];
} vpx_reader;
```

`BD_VALUE` is `size_t` (`bitreader.h:31`), so this is either 32 or 64 bits.
`BD_VALUE_SIZE` is `sizeof(BD_VALUE) * CHAR_BIT` (`bitreader.h:33`).
`LOTS_OF_BITS` is `0x40000000` (`bitreader.h:38`), used as a sentinel to
mark "we have run past end of stream".

Field semantics:
- `value` — top `BD_VALUE_SIZE` bits of the current arithmetic-coder
  interval (a sliding window over the bitstream).
- `range` — same role as in the writer.
- `count` — number of valid bits in `value` minus 8. Negative means we
  need a refill; `>= BD_VALUE_SIZE` with `< LOTS_OF_BITS` means an error
  condition (`vpx_reader_has_error`, `bitreader.h:59`).
- `buffer_end` / `buffer` — current and end pointer into the input bytes.
- `decrypt_cb` / `decrypt_state` — optional in-place decryption hook
  (libvpx-internal; not used in standard VP9 conformance).
- `clear_buffer` — scratch buffer for the decryption path
  (`bitreader.c:48`); sized to hold one fill plus the marker byte.

### 2.3 The raw (fixed-bit) writer — `vpx_write_bit_buffer`

`third_party/libvpx/vpx_dsp/bitwriter_buffer.h:20-31`:

```c
struct vpx_write_bit_buffer {
  // Whether there has been an error.
  int error;
  // We maintain the invariant that bit_offset <= size * CHAR_BIT, i.e., we
  // never write beyond the end of bit_buffer. If bit_offset would be
  // incremented to be greater than size * CHAR_BIT, leave bit_offset unchanged
  // and set error to 1.
  size_t bit_offset;
  // Size of bit_buffer in bytes.
  size_t size;
  uint8_t *bit_buffer;
};
```

Used only for the uncompressed header (`write_uncompressed_header`,
`vp9_bitstream.c:1243`). Writes raw MSB-first bits into a byte buffer
without any entropy coding. Bit offset is in bits, not bytes.

### 2.4 The raw reader — `vpx_read_bit_buffer`

`third_party/libvpx/vpx_dsp/bitreader_buffer.h:24-31`:

```c
struct vpx_read_bit_buffer {
  const uint8_t *bit_buffer;
  const uint8_t *bit_buffer_end;
  size_t bit_offset;

  void *error_handler_data;
  vpx_rb_error_handler error_handler;
};
```

`vpx_rb_error_handler` is `void (*)(void *data)`
(`bitreader_buffer.h:22`).

### 2.5 The shared probability type and tree convention

`third_party/libvpx/vpx_dsp/prob.h:25-46`:

```c
typedef uint8_t vpx_prob;

#define MAX_PROB 255

#define vpx_prob_half ((vpx_prob)128)

typedef int8_t vpx_tree_index;

#define TREE_SIZE(leaf_count) (2 * (leaf_count)-2)

#define vpx_complement(x) (255 - (x))

#define MODE_MV_COUNT_SAT 20

/* We build coding trees compactly in arrays.
   Each node of the tree is a pair of vpx_tree_indices.
   Array index often references a corresponding probability table.
   Index <= 0 means done encoding/decoding and value = -Index,
   Index > 0 means need another bit, specification at index.
   Nonnegative indices are always even;  processing begins at node 0. */

typedef const vpx_tree_index vpx_tree[];
```

Critical: probabilities are 8-bit unsigned in `[1, 255]` (the encoder
returns `vpx_prob` clipped to `[1, 255]` from `get_prob`, `prob.h:48`;
zero is reserved). A probability `p` represents `P(bit==0) = p / 256`.
The pseudo-probability `128` is "fair coin"; it is what
`vpx_write_bit` and `vpx_read_bit` use (`bitwriter.h:118`,
`bitreader.h:138`).

Trees are arrays of pairs. Index 0 is the root. At each node, the
two children live at positions `tree[i]` and `tree[i+1]`. A non-positive
child encodes a leaf with value `-child`. The probability table
for tree `tree[]` is indexed by `i >> 1` (so node 0 uses probability
0, node 2 uses probability 1, etc.).

### 2.6 The frame-level entropy context — `FRAME_CONTEXT`

`third_party/libvpx/vp9/common/vp9_entropymode.h:44-60`:

```c
typedef struct frame_contexts {
  vpx_prob y_mode_prob[BLOCK_SIZE_GROUPS][INTRA_MODES - 1];
  vpx_prob uv_mode_prob[INTRA_MODES][INTRA_MODES - 1];
  vpx_prob partition_prob[PARTITION_CONTEXTS][PARTITION_TYPES - 1];
  vp9_coeff_probs_model coef_probs[TX_SIZES][PLANE_TYPES];
  vpx_prob switchable_interp_prob[SWITCHABLE_FILTER_CONTEXTS]
                                 [SWITCHABLE_FILTERS - 1];
  vpx_prob inter_mode_probs[INTER_MODE_CONTEXTS][INTER_MODES - 1];
  vpx_prob intra_inter_prob[INTRA_INTER_CONTEXTS];
  vpx_prob comp_inter_prob[COMP_INTER_CONTEXTS];
  vpx_prob single_ref_prob[REF_CONTEXTS][2];
  vpx_prob comp_ref_prob[REF_CONTEXTS];
  struct tx_probs tx_probs;
  vpx_prob skip_probs[SKIP_CONTEXTS];
  nmv_context nmvc;
  int initialized;
} FRAME_CONTEXT;
```

Constants (file:line each):
- `BLOCK_SIZE_GROUPS = 4` (`vp9_entropymode.h:23`).
- `TX_SIZE_CONTEXTS = 2` (`vp9_entropymode.h:25`).
- `INTRA_MODES = 10` (`vp9_enums.h:127`; `TM_PRED + 1`).
- `INTER_MODES = 4` (`vp9_enums.h:129`; `1 + NEWMV - NEARESTMV`).
- `INTER_MODE_CONTEXTS = 7` (`vp9_enums.h:132`).
- `INTRA_INTER_CONTEXTS = 4` (`vp9_enums.h:137`).
- `COMP_INTER_CONTEXTS = 5` (`vp9_enums.h:138`).
- `REF_CONTEXTS = 5` (`vp9_enums.h:139`).
- `SKIP_CONTEXTS = 3` (`vp9_enums.h:131`).
- `PARTITION_CONTEXTS = 16` (`vp9_enums.h:74`; `4 * PARTITION_PLOFFSET`).
- `PARTITION_TYPES = 4` (`vp9_enums.h:68`).
- `SWITCHABLE_FILTERS = 3` (`vp9_filter.h:26`).
- `SWITCHABLE_FILTER_CONTEXTS = 4` (`vp9_filter.h:31`; `SWITCHABLE_FILTERS + 1`).
- `TX_SIZES = 4` (`vp9_enums.h:82`).
- `PLANE_TYPES = 2` (`vp9_enums.h:108`).

The full coef probs nesting (`vp9_coeff_probs_model`,
`vp9_entropy.h:150-151`):

```c
typedef vpx_prob vp9_coeff_probs_model[REF_TYPES][COEF_BANDS][COEFF_CONTEXTS]
                                      [UNCONSTRAINED_NODES];
```

Where:
- `REF_TYPES = 2` (`vp9_entropy.h:86`; intra=0, inter=1).
- `COEF_BANDS = 6` (`vp9_entropy.h:89`).
- `COEFF_CONTEXTS = 6` (`vp9_entropy.h:107`), but
  `BAND_COEFF_CONTEXTS(band) = (band == 0 ? 3 : 6)`
  (`vp9_entropy.h:108`). Band 0 only uses 3 contexts (the 3 possible
  values of the "above + left both nonzero" sum); other bands use 6.
- `UNCONSTRAINED_NODES = 3` (`vp9_entropy.h:142`) — the first three
  nodes of the token tree (EOB?, ZERO?, ONE?) carry their own per-context
  probabilities; the rest are projected from the second one
  (`PIVOT_NODE = 2`, `vp9_entropy.h:144`) via the pareto8 model.

Tx-size sub-probabilities (`vp9_entropymode.h:31-35`):

```c
struct tx_probs {
  vpx_prob p32x32[TX_SIZE_CONTEXTS][TX_SIZES - 1];
  vpx_prob p16x16[TX_SIZE_CONTEXTS][TX_SIZES - 2];
  vpx_prob p8x8[TX_SIZE_CONTEXTS][TX_SIZES - 3];
};
```

Lifetime / ownership: `cm->fc` is the *current* frame context (one
instance, `VP9Common.fc`). `cm->frame_contexts[FRAME_CONTEXTS]` holds 4
saved contexts that signaled frames can select via `frame_context_idx`
(2-bit literal in the uncompressed header; see
`vp9_bitstream.c:1307`). `FRAME_CONTEXTS = 4`
(`vp9_onyxc_int.h:44-45`). Adaptation copies counts → probs
post-encode (`vp9_adapt_mode_probs`, `vp9_entropymode.c:340`). The
decoder runs the same `vp9_adapt_*` functions; this is the byte-parity
linchpin.

### 2.7 The per-frame entropy counts — `FRAME_COUNTS`

`third_party/libvpx/vp9/common/vp9_entropymode.h:62-79`:

```c
typedef struct FRAME_COUNTS {
  unsigned int y_mode[BLOCK_SIZE_GROUPS][INTRA_MODES];
  unsigned int uv_mode[INTRA_MODES][INTRA_MODES];
  unsigned int partition[PARTITION_CONTEXTS][PARTITION_TYPES];
  vp9_coeff_count_model coef[TX_SIZES][PLANE_TYPES];
  unsigned int eob_branch[TX_SIZES][PLANE_TYPES][REF_TYPES][COEF_BANDS]
                         [COEFF_CONTEXTS];
  unsigned int switchable_interp[SWITCHABLE_FILTER_CONTEXTS]
                                [SWITCHABLE_FILTERS];
  unsigned int inter_mode[INTER_MODE_CONTEXTS][INTER_MODES];
  unsigned int intra_inter[INTRA_INTER_CONTEXTS][2];
  unsigned int comp_inter[COMP_INTER_CONTEXTS][2];
  unsigned int single_ref[REF_CONTEXTS][2][2];
  unsigned int comp_ref[REF_CONTEXTS][2];
  struct tx_counts tx;
  unsigned int skip[SKIP_CONTEXTS][2];
  nmv_context_counts mv;
} FRAME_COUNTS;
```

`vp9_coeff_count_model` (`vp9_entropy.h:153-155`):

```c
typedef unsigned int vp9_coeff_count_model[REF_TYPES][COEF_BANDS]
                                          [COEFF_CONTEXTS]
                                          [UNCONSTRAINED_NODES + 1];
```

Note the `+ 1`: the extra slot is `EOB_MODEL_TOKEN` (= 3, see
`vp9_entropy.h:76`), used to count "above-EOB" tokens distinctly from
"non-EOB" so the EOB-branch probability can be merged
(`adapt_coef_probs`, `vp9_entropy.c:1055`).

`tx_counts` (`vp9_entropymode.h:37-42`):

```c
struct tx_counts {
  unsigned int p32x32[TX_SIZE_CONTEXTS][TX_SIZES];
  unsigned int p16x16[TX_SIZE_CONTEXTS][TX_SIZES - 1];
  unsigned int p8x8[TX_SIZE_CONTEXTS][TX_SIZES - 2];
  unsigned int tx_totals[TX_SIZES];
};
```

### 2.8 The MV context — `nmv_context` and `nmv_context_counts`

`third_party/libvpx/vp9/common/vp9_entropymv.h:88-102`:

```c
typedef struct {
  vpx_prob sign;
  vpx_prob classes[MV_CLASSES - 1];
  vpx_prob class0[CLASS0_SIZE - 1];
  vpx_prob bits[MV_OFFSET_BITS];
  vpx_prob class0_fp[CLASS0_SIZE][MV_FP_SIZE - 1];
  vpx_prob fp[MV_FP_SIZE - 1];
  vpx_prob class0_hp;
  vpx_prob hp;
} nmv_component;

typedef struct {
  vpx_prob joints[MV_JOINTS - 1];
  nmv_component comps[2];
} nmv_context;
```

`vp9_entropymv.h:114-128`:

```c
typedef struct {
  unsigned int sign[2];
  unsigned int classes[MV_CLASSES];
  unsigned int class0[CLASS0_SIZE];
  unsigned int bits[MV_OFFSET_BITS][2];
  unsigned int class0_fp[CLASS0_SIZE][MV_FP_SIZE];
  unsigned int fp[MV_FP_SIZE];
  unsigned int class0_hp[2];
  unsigned int hp[2];
} nmv_component_counts;

typedef struct {
  unsigned int joints[MV_JOINTS];
  nmv_component_counts comps[2];
} nmv_context_counts;
```

Constants (`vp9_entropymv.h:37-77`):
- `MV_UPDATE_PROB = 252`.
- `MV_JOINTS = 4` (zero / horiz-zero-vert-nonzero / vert-zero /
  both-nonzero).
- `MV_CLASSES = 11`.
- `CLASS0_BITS = 1`, `CLASS0_SIZE = 2`.
- `MV_OFFSET_BITS = MV_CLASSES + CLASS0_BITS - 2 = 10`.
- `MV_FP_SIZE = 4`.
- `MV_MAX_BITS = MV_CLASSES + CLASS0_BITS + 2 = 14`.
- `MV_MAX = (1 << 14) - 1 = 16383`.
- `MV_VALS = 2 * MV_MAX + 1 = 32767`.

### 2.9 The tokens that drive the coefficient coder — `TOKENEXTRA`

`third_party/libvpx/vp9/encoder/vp9_tokenize.h:36-40`:

```c
typedef struct {
  const vpx_prob *context_tree;
  int16_t token;
  EXTRABIT extra;
} TOKENEXTRA;
```

Where `EXTRABIT` is `int16_t` (`int32_t` for `CONFIG_VP9_HIGHBITDEPTH`),
`vp9_tokenize.h:25-29`. Each token carries a pointer to the probability
vector for its context (so the coder doesn't have to redo context
derivation in the writer). `EOSB_TOKEN = 127` (`vp9_tokenize.h:23`) is
an encoder-side sentinel that is *not* emitted; it just terminates the
per-superblock token stream.

`TOKENVALUE` (a token+extra pair without the context pointer, used in
cost computation):

```c
typedef struct {
  int16_t token;
  EXTRABIT extra;
} TOKENVALUE;
```

### 2.10 The bitstream worker data — `VP9BitstreamWorkerData`

`third_party/libvpx/vp9/encoder/vp9_bitstream.h:20-32`:

```c
typedef struct VP9BitstreamWorkerData {
  uint8_t *dest;
  int dest_size;
  vpx_writer bit_writer;
  int tile_idx;
  unsigned int max_mv_magnitude;
  // The size of interp_filter_selected in VP9_COMP is actually
  // MAX_REFERENCE_FRAMES x SWITCHABLE. But when encoding tiles, all we ever do
  // is increment the very first index (index 0) for the first dimension. Hence
  // this is sufficient.
  int interp_filter_selected[1][SWITCHABLE];
  DECLARE_ALIGNED(16, MACROBLOCKD, xd);
} VP9BitstreamWorkerData;
```

A4-relevant fields: every tile worker carries its own `vpx_writer`
(`bit_writer`) and its own output destination (`dest`, `dest_size`). The
per-worker buffer is allocated by `encode_tiles_buffer_alloc`
(`vp9_bitstream.c:975`). Worker 0 writes directly to the final output;
workers 1..N-1 write to private buffers and have their output `memcpy`'d
back after sync (`encode_tiles_mt`, `vp9_bitstream.c:1080`).

### 2.11 The scan-order descriptor — `ScanOrder`

`third_party/libvpx/vp9/common/vp9_scan.h:26-30`:

```c
typedef struct ScanOrder {
  const int16_t *scan;
  const int16_t *iscan;
  const int16_t *neighbors;
} ScanOrder;
```

- `scan[k]` is the position-in-raster of the `k`-th coefficient in scan
  order. Length is `1 << (2 * (tx_size_log2 + 2))` (16 / 64 / 256 / 1024).
- `iscan[p]` is the scan index that contains raster-position `p` *plus
  one* (so `iscan[p]` is the EOB position if the last nonzero coefficient
  lives at raster `p`). Same length as `scan`. See `vp9_scan.c:514` —
  "Add 1 to iscan values. This represents the EOB position instead of the
  index."
- `neighbors[2*k + {0,1}]` give the raster positions of the {top, left}
  neighbors of scan-position `k`, used to derive the coefficient context.
  Length is `(N + 1) * MAX_NEIGHBORS` where `N` is the number of
  coefficients; the trailing `(0, 0)` pair is a sentinel.
  `MAX_NEIGHBORS = 2` (`vp9_scan.h:24`).

The dispatch tables that select among per-tx-size, per-tx-type variants
are `vp9_default_scan_orders[TX_SIZES]` (`vp9_scan.c:691`) and
`vp9_scan_orders[TX_SIZES][TX_TYPES]` (`vp9_scan.c:698`). The
selector function is `get_scan` (`vp9_scan.h:42`): inter blocks and UV
blocks always use the default scan; intra Y blocks use the scan keyed by
the predicted intra mode via `intra_mode_to_tx_type_lookup`.

---

## 3. Control flow

### 3.1 Frame-level encode sequence

Top-level: `vp9_pack_bitstream` (`vp9_bitstream.c:1396-1451`). The
sequence per frame is:

```
vpx_wb_init(&wb, data, data_size)
write_uncompressed_header(cpi, &wb)              -- raw bits
  if (show_existing_frame) return early.

reserve 16 raw bits for compressed_hdr_size      -- saved_wb cursor
data += uncompressed_hdr_size
vpx_clear_system_state()
compressed_hdr_size = write_compressed_header(cpi, data, data_size)
  -- this opens a vpx_writer at data, writes:
  --   tx-size mode (2 bits raw via vpx_write_literal)
  --   per-tx-mode tx_probs updates           (encode_txfm_probs)
  --   coef_probs updates per tx_size         (update_coef_probs)
  --   skip_probs updates                     (update_skip_probs)
  --   inter-only:
  --     inter_mode_probs                     (prob_diff_update)
  --     switchable_interp_prob               (if SWITCHABLE)
  --     intra_inter_prob updates
  --     reference_mode bits + comp_inter updates
  --     single_ref / comp_ref updates
  --     y_mode_prob updates                  (per BLOCK_SIZE_GROUP)
  --     partition_prob updates
  --     vp9_write_nmv_probs                  (MV prob updates)
  -- vpx_stop_encode flushes the boolean coder.

backfill the 16-bit compressed_hdr_size into saved_wb.
data += compressed_hdr_size
encode_tiles(cpi, data, data_size)
  -- writes (tile_cols * tile_rows) tiles. All tiles except the last
  -- are preceded by a 4-byte big-endian tile size.
*size = total bytes written.
```

### 3.2 The uncompressed header

`write_uncompressed_header` (`vp9_bitstream.c:1243-1314`):

```
2 bits  : VP9_FRAME_MARKER (= 0x2)
2-3 bits: profile (per write_profile, vp9_bitstream.c:1206; PROFILE_3
          uses 3 bits, others 2)
1 bit   : show_existing_frame
  if show_existing_frame:
    3 bits: frame index in reference buffer pool (alt_fb_idx)
    return.
1 bit   : frame_type (0 = KEY_FRAME, 1 = inter)
1 bit   : show_frame
1 bit   : error_resilient_mode

if KEY_FRAME:
  24 bits: sync code (0x49, 0x83, 0x42; vp9_common.h:49-51)
  bit-depth / colorspace / sampling (write_bitdepth_colorspace_sampling)
  frame size (16+16 bits, render size optional)
else:
  if !show_frame: 1 bit intra_only
  if !error_resilient_mode: 2 bits reset_frame_context

  if intra_only:
    24 bits: sync code
    if profile > 0: write_bitdepth_colorspace_sampling
    8 bits  : refresh_frame_flags
    frame size
  else:
    8 bits  : refresh_frame_flags
    for each of 3 ref slots:
      3 bits: REF_FRAMES_LOG2 index into ref_frame_map
      1 bit : ref_frame_sign_bias
    frame size (possibly via reference)
    1 bit   : allow_high_precision_mv
    interp_filter (1 bit "is SWITCHABLE", + 2 bits if not)

if !error_resilient_mode:
  1 bit: refresh_frame_context
  1 bit: frame_parallel_decoding_mode

2 bits  : frame_context_idx (FRAME_CONTEXTS_LOG2)
loopfilter info (encode_loopfilter)
quantization info (encode_quantization)
segmentation info (encode_segmentation)
tile info (write_tile_info)
```

Tile info (`write_tile_info`, `vp9_bitstream.c:886`):

```
log2_tile_cols is encoded as a unary code with a 1-bit terminator:
  emit (log2_tile_cols - min_log2_tile_cols) '1' bits
  if log2_tile_cols < max_log2_tile_cols: emit '0'

log2_tile_rows:
  emit 1 bit (log2_tile_rows != 0)
  if nonzero: emit 1 bit (log2_tile_rows != 1)
  (so 0=>"0", 1=>"10", 2=>"11"; max is 2 rows-log2)
```

After the uncompressed header bits are packed (`vpx_wb_bytes_written`
rounds up to a byte boundary, `bitwriter_buffer.c:30`), the 16-bit
`compressed_hdr_size` literal follows. This is reserved at
`vp9_bitstream.c:1426` via `vpx_wb_write_literal(&wb, 0, 16)` then
back-patched at `vp9_bitstream.c:1445`.

### 3.3 The compressed header

`write_compressed_header` (`vp9_bitstream.c:1316-1394`). Opens a fresh
`vpx_writer` over the byte slice after the uncompressed header
(`vpx_start_encode`, `bitwriter.c:20`) and writes all probability *updates*
as boolean-coded literals. The first byte of the compressed header is
*not* an arithmetic-coded byte: `vpx_start_encode` writes a 0 bit as the
first event (`bitwriter.c:30`). This 0 bit is the spec-mandated "marker
bit"; the matching decoder side checks it returned 0 in
`vpx_reader_init` (`bitreader.c:34`).

The contents:

1. **Tx-size mode** (`encode_txfm_probs`, `vp9_bitstream.c:818`):
   - 2 raw bits: `min(tx_mode, ALLOW_32X32)` via `vpx_write_literal`.
   - If `tx_mode >= ALLOW_32X32`: 1 bit `tx_mode == TX_MODE_SELECT`.
   - If `TX_MODE_SELECT`: emit per-context tx-size prob updates via
     `vp9_cond_prob_diff_update`.

2. **Coefficient prob updates** (`update_coef_probs`,
   `vp9_bitstream.c:683`). Per `tx_size` from `TX_4X4` up to
   `max_tx_size`:
   - 1 bit "any updates this tx_size".
   - If yes: full traversal (`update_coef_probs_common`,
     `vp9_bitstream.c:545`) over the 4D `[PLANE_TYPES][REF_TYPES][COEF_BANDS]
     [BAND_COEFF_CONTEXTS(k)][UNCONSTRAINED_NODES]` model, with two
     possible loop structures controlled by
     `cpi->sf.use_fast_coef_updates`:
     - `TWO_LOOP` — dry run computes whether any update saves bits.
     - `ONE_LOOP_REDUCED` — single pass; trailing zeros omitted.
     For each `(i,j,k,l,t)` model node:
       1 arithmetic bit `u` against probability `DIFF_UPDATE_PROB = 252`
       (`vp9_entropy.h:24`) indicating "update this prob".
       If `u`: `vp9_write_prob_diff_update` (`vp9_subexp.c:112`) emits a
       sub-exponential code for the recentered prob difference. See
       §4.5 below for the codeword.

3. **Skip probs** (`update_skip_probs`, `vp9_bitstream.c:115`): 3
   conditional updates via `vp9_cond_prob_diff_update`.

4. (Inter-only) **Inter-mode probs**: 7 contexts × tree of 3 internal
   nodes; each `prob_diff_update` (`vp9_bitstream.c:74`) computes branch
   counts via `vp9_tree_probs_from_distribution` and runs
   `vp9_cond_prob_diff_update` per internal node.

5. (Inter-only) **Switchable interp probs**: if `interp_filter ==
   SWITCHABLE`, 4 contexts × tree of 2 internal nodes.

6. (Inter-only) **Intra/inter probs**: 4 contexts × 1
   `vp9_cond_prob_diff_update`.

7. (Inter-only) **Reference-mode signaling**:
   - 1 bit `use_compound_pred`.
   - If yes: 1 bit `use_hybrid_pred` and (if hybrid) 5 contexts ×
     `vp9_cond_prob_diff_update` for `comp_inter_prob`.

8. (Inter-only) **Single-ref / comp-ref probs** (5 contexts each).

9. (Inter-only) **Y-mode probs**: 4 block-size groups × tree of 9
   internal nodes.

10. (Inter-only) **Partition probs**: 16 contexts × tree of 3 internal
    nodes.

11. (Inter-only) **NMV probs** (`vp9_write_nmv_probs`,
    `vp9_encodemv.c:172`). See §4.6.

12. **Flush** via `vpx_stop_encode` (`bitwriter.c:33`). The flush emits 32
    "0" bits via the arithmetic coder so the decoder can drain its
    pipeline, then patches a final byte if the last byte would collide
    with the superframe-index marker pattern (`(buffer[pos-1] & 0xe0)
    == 0xc0`).

### 3.4 Tile data

`encode_tiles` (`vp9_bitstream.c:1092-1145`). For each tile in raster
order:

```
if (not last tile) reserve 4 bytes for the tile size (big-endian uint32).
vpx_start_encode(&residual_bc, data + offset, ...)
write_modes(cpi, xd, &tile_info, &residual_bc, ...)
vpx_stop_encode(&residual_bc)
if (not last tile) mem_put_be32(data + total_size, residual_bc.pos)
total_size += residual_bc.pos
```

`write_modes` walks superblocks within the tile (`vp9_bitstream.c:490`).
For each SB it walks the partition recursion (`write_modes_sb`,
`vp9_bitstream.c:427`), emitting partition decisions (`write_partition`,
`vp9_bitstream.c:405`) and per-block syntax (`write_modes_b`,
`vp9_bitstream.c:377`).

`write_modes_b` chooses between:
- `write_mb_modes_kf` (`vp9_bitstream.c:343`) for intra-only frames
  (keyframes and intra-only inter frames).
- `pack_inter_mode_mvs` (`vp9_bitstream.c:240`) for inter blocks.

Both then call `pack_mb_tokens` (`vp9_bitstream.c:132`) to dump the
coefficient stream.

**Multithreaded tile encode** (`encode_tiles_mt`, `vp9_bitstream.c:991`):
only used when `oxcf.mode == REALTIME && num_workers > 1 && tile_rows ==
1 && tile_cols > 1` (`vp9_bitstream.c:1107`). Each worker carries its own
`vpx_writer`; worker 0 writes in place, workers 1..N-1 buffer privately
and the main thread `memcpy`'s their output back after sync. Tile sizes
are written into the 4-byte prefix slots after sync.

### 3.5 Boolean-coder write state machine

The whole point: each `vpx_write(w, bit, prob)` updates the writer state
according to a simple recurrence. The recurrence in `vpx_write`
(`bitwriter.h:46-116`):

```
split = 1 + (((range - 1) * probability) >> 8)
range = split                              // if bit == 0
if (bit) {
  lowvalue += split
  range = old_range - split                // if bit == 1
}
shift = vpx_norm[range]                    // count leading-zero bits of range
range <<= shift
count += shift
if (count >= 0):                           // we have a byte to emit
  offset = shift - count
  // Carry propagation: if the about-to-emit byte's high bit is set,
  // walk backwards through any 0xff bytes resetting them and increment
  // the byte before.
  if ((lowvalue << (offset - 1)) & 0x80000000):
    x = pos - 1
    while (x >= 0 && buffer[x] == 0xff) { buffer[x] = 0; x-- }
    buffer[x] += 1
  buffer[pos++] = (lowvalue >> (24 - offset)) & 0xff
  lowvalue <<= offset
  shift = count
  lowvalue &= 0xffffff
  count -= 8
lowvalue <<= shift
```

Initial state from `vpx_start_encode` (`bitwriter.c:20`):
```
lowvalue = 0, range = 255, count = -24, pos = 0
then vpx_write_bit(br, 0)    // the spec "marker bit"
```

Termination from `vpx_stop_encode` (`bitwriter.c:33`):
```
for i in 0..32: vpx_write_bit(br, 0)        // drain
if !error && (buffer[pos-1] & 0xe0) == 0xc0:
  buffer[pos++] = 0                         // disambiguate from superframe index
```

The carry-propagation fixup is the only non-local thing in the encoder:
emitting a byte may require flipping bytes already emitted to handle a
"+1" that propagates through stored 0xff bytes. The encoder relies on the
invariant that any sequence of 0xff bytes in the output buffer is
preceded by a byte that can still absorb a "+1" carry (the carry never
escapes past `pos == 0` once the marker bit is written).

### 3.6 Boolean-coder read state machine

The dual in `vpx_read` (`bitreader.h:77-136`):

```
split = (range * prob + (256 - prob)) >> CHAR_BIT  // = ceil(range * prob / 256)
if (count < 0) vpx_reader_fill(r)            // refill value
bigsplit = (BD_VALUE)split << (BD_VALUE_SIZE - CHAR_BIT)
range = split
if (value >= bigsplit):
  range = old_range - split
  value -= bigsplit
  bit = 1
shift = vpx_norm[(uint8_t)range]
range <<= shift
value <<= shift
count -= shift
return bit
```

Note the asymmetric split formula: the encoder uses `1 + (((range - 1) *
prob) >> 8)`; the decoder uses `(range * prob + (256 - prob)) >> 8`. The
two are algebraically identical for `range, prob in [1, 255]`:

```
encoder: split = 1 + floor((range - 1) * prob / 256)
decoder: split = floor((range * prob + 256 - prob) / 256)
       = floor((range * prob - prob + 256) / 256)
       = floor(((range - 1) * prob + 256) / 256)
       = 1 + floor((range - 1) * prob / 256)
```

This identity is the single most important invariant for byte-parity. Any
Rust port must preserve it exactly (do NOT "simplify" either formula).

Initial state from `vpx_reader_init` (`bitreader.c:21`):
```
value = 0, count = -8, range = 255
vpx_reader_fill(r)                           // pull in first BD_VALUE_SIZE bits
assert vpx_read_bit(r) == 0                  // the marker bit must read 0
```

`vpx_reader_fill` (`bitreader.c:38`) pulls one big-endian
`BD_VALUE_SIZE`-bit chunk from the buffer, adjusts `count` to track
remaining bits, and handles end-of-stream by adding `LOTS_OF_BITS` to the
count once exhausted.

`vpx_norm[256]` (`vpx_dsp/prob.c:13-24`) is the byte-wise leading-zero
table:
```
vpx_norm[0..1]   = {0, 7}
vpx_norm[2..3]   = {6, 6}
vpx_norm[4..7]   = {5,5,5,5}
vpx_norm[8..15]  = {4,...}     // 8 entries
vpx_norm[16..31] = {3,...}     // 16 entries
vpx_norm[32..63] = {2,...}     // 32 entries
vpx_norm[64..127]= {1,...}     // 64 entries
vpx_norm[128..255]= {0,...}    // 128 entries
```
(Full bytes verbatim in §5.5.)

### 3.7 Tree-coded symbol — encode

`vp9_write_tree` (`vp9_treewriter.h:31-39`) and `vp9_write_token`
(`vp9_treewriter.h:41-45`). Given a `(value, len)` encoding for a leaf
(computed once at startup by `vp9_tokens_from_tree`,
`vp9_treewriter.c:13`) the encoder simply walks the bits MSB-first; each
bit is written with the probability table indexed by the current node.

```c
do {
  const int bit = (bits >> --len) & 1;
  vpx_write(w, bit, probs[i >> 1]);
  i = tree[i + bit];
} while (len);
```

### 3.8 Tree-coded symbol — decode

`vpx_read_tree` (`bitreader.h:150`):

```c
vpx_tree_index i = 0;
while ((i = tree[i + vpx_read(r, probs[i >> 1])]) > 0) continue;
return -i;
```

A single while-loop that follows `tree[]` until it hits a non-positive
child (the leaf value).

### 3.9 Coefficient encoding state machine

The token stream for a transform block is emitted by `pack_mb_tokens`
(`vp9_bitstream.c:132-195`). Each `TOKENEXTRA` carries:
- `context_tree` — pointer into `cm->fc->coef_probs[tx_size][...][...]`.
  Specifically `context_tree[0..2]` are the per-context probabilities for
  the first three nodes of `vp9_coef_tree` (EOB, ZERO, ONE);
  `context_tree[PIVOT_NODE] = context_tree[2]` is also used as the index
  into `vp9_pareto8_full` for the trailing nodes.
- `token` — one of `EOB_TOKEN`, `ZERO_TOKEN`, `ONE_TOKEN`,
  `TWO_TOKEN`..`CATEGORY6_TOKEN`, or the encoder-only sentinel
  `EOSB_TOKEN`.
- `extra` — the magnitude residual + sign (LSB) for category tokens.

The encoder walks the tokens in scan order:

```
for each TOKENEXTRA p:
  if p->token == EOSB_TOKEN: terminate this superblock.
  if p->token == EOB_TOKEN:
    vpx_write(w, 0, context_tree[0])       // 0 at EOB node => EOB
    continue (next block).
  vpx_write(w, 1, context_tree[0])         // 1 at EOB node => not EOB

  while p->token == ZERO_TOKEN:
    vpx_write(w, 0, context_tree[1])
    advance p.
    if p == stop or EOSB_TOKEN: return.

  // p is now in {ONE, TWO..FIVE, CAT1..CAT6}
  vpx_write(w, 1, context_tree[1])         // 1 at ZERO node => nonzero

  if p->token == ONE_TOKEN:
    vpx_write(w, 0, context_tree[2])       // 0 at ONE node => ONE
    vpx_write_bit(w, p->extra & 1)         // sign
  else:
    vpx_write(w, 1, context_tree[2])       // 1 at ONE node => higher
    // Walk vp9_coef_con_tree using pareto8 row indexed by context_tree[PIVOT_NODE] - 1.
    vp9_write_tree(w, vp9_coef_con_tree,
                   vp9_pareto8_full[context_tree[PIVOT_NODE] - 1],
                   t->value, t->len - UNCONSTRAINED_NODES, 0);
    // Emit category extra bits using vp9_extra_bits[token].prob.
    if p->token >= CATEGORY1_TOKEN:
      const vp9_extra_bit *b = &extra_bits[p->token];
      v = p->extra >> 1; n = b->len;
      while n > 0:
        vpx_write(w, (v >> --n) & 1, b->prob[<advance>]);
    vpx_write_bit(w, p->extra & 1)         // sign
```

The encoder-side context is computed once at tokenization time (in
`vp9_tokenize.c`, owned by A2's tokenizer logic) and stuffed into
`context_tree`. The decoder re-derives it on each token: see
`get_coef_context` (`vp9_scan.h:35`):

```c
static INLINE int get_coef_context(const int16_t *neighbors,
                                   const uint8_t *token_cache, int c) {
  return (1 + token_cache[neighbors[MAX_NEIGHBORS * c + 0]] +
          token_cache[neighbors[MAX_NEIGHBORS * c + 1]]) >> 1;
}
```

Where `token_cache` maps each *raster* coefficient position to a "pt
class" via `vp9_pt_energy_class` (`vp9_entropy.c:95`) lookup once a token
has been decoded for that position. The `+ 1` then `>> 1` rounds the
neighbor-class sum to {0,1,2}; for band 0 those are the only valid
contexts (hence `BAND_COEFF_CONTEXTS(0) == 3`).

The category-extra-bit prob arrays `vp9_catN_prob` are MSB-first arrays
of `N` 8-bit probabilities, one per bit of the residual; see §5.4.

---

## 4. Algorithms

### 4.1 The boolean encoder — pseudocode

```
init(buffer, size):
  lowvalue = 0
  range = 255
  count = -24
  error = 0
  pos = 0
  size = min(size, INT_MAX)
  buffer = buffer
  write(0, 128)                 // marker bit (vpx_prob_half)

write(bit, prob):
  split = 1 + (((range - 1) * prob) >> 8)
  if bit == 0:
    range = split
  else:
    lowvalue += split
    range = range - split
  shift = vpx_norm[range]
  range <<= shift
  count += shift
  if count >= 0:
    offset = shift - count
    // carry propagation
    if (lowvalue << (offset - 1)) has bit 31 set:
      x = pos - 1
      while x >= 0 && buffer[x] == 0xff:
        buffer[x] = 0
        x -= 1
      buffer[x] += 1
    buffer[pos++] = (lowvalue >> (24 - offset)) & 0xff
    lowvalue = (lowvalue << offset) & 0xffffff
    count -= 8
    shift = count          // re-used as the next left shift below
  lowvalue <<= shift

stop():
  for i in 0..32: write_bit(0)
  if (buffer[pos-1] & 0xe0) == 0xc0:
    buffer[pos++] = 0
```

### 4.2 The boolean decoder — pseudocode

```
init(buffer, size):
  buffer_end = buffer + size
  value = 0
  count = -8
  range = 255
  fill()
  if read_bit() != 0: ERROR              // marker bit must be 0

fill():
  // pulls one BD_VALUE_SIZE chunk from buffer in big-endian order,
  // shifts into value at the correct alignment, advances buffer,
  // bumps count by 8 per consumed byte. Past end-of-stream:
  // count += LOTS_OF_BITS so subsequent reads return 0 deterministically.

read(prob):
  split = (range * prob + (256 - prob)) >> 8
  if count < 0: fill()
  bigsplit = (size_t)split << (BD_VALUE_SIZE - 8)
  if value < bigsplit:
    range = split
    bit = 0
  else:
    range -= split
    value -= bigsplit
    bit = 1
  shift = vpx_norm[(uint8_t)range]
  range <<= shift
  value <<= shift
  count -= shift
  return bit
```

### 4.3 The tree writer

```
write_token(tree, probs, token):
  bits = token.value
  len = token.len
  i = 0
  do:
    bit = (bits >> --len) & 1
    write(bit, probs[i >> 1])
    i = tree[i + bit]
  while len > 0
```

`vp9_tokens_from_tree` (`vp9_treewriter.c:29`) flattens a tree into a
`vp9_token[leaf_count]` of `(value, len)` pairs by depth-first traversal.

### 4.4 The coefficient encoding flow

(Restated from §3.9 in algorithm form.)

For each transform block of size `tx_size`:
```
scan = get_scan(xd, tx_size, plane_type, block_idx)
                                          // selects default / row / col / default
token_cache[0..N-1] = 0
context = pt(above, left)                 // get_entropy_context(tx_size, a, l)
for c in 0..N-1:
  raster_pos = scan.scan[c]
  coef = qcoeff[raster_pos]
  band = (tx_size == TX_4X4 ? vp9_coefband_trans_4x4 : vp9_coefband_trans_8x8plus)[c]
  context_tree = coef_probs[ref_type][band][context]
                                          // 3 probs per (band, context)
  emit_token_for(coef, context_tree)      // EOB / ZERO / ONE / .. / CAT6
  if c < N-1:                             // update context for next pos
    next_context = get_coef_context(scan.neighbors, token_cache, c + 1)
    context = next_context
  if hit EOB: break
```

The encoder fills a `TOKENEXTRA` stream in `vp9_tokenize_sb`
(`vp9_tokenize.h:52`, body in `vp9_tokenize.c`) and consumes that stream
in `pack_mb_tokens` at bitstream write time.

### 4.5 Sub-exponential coding of probability updates

Both `vp9_subexp.c` (encoder) and `vp9_dsubexp.c` (decoder) define a
codec for an 8-bit `delp` ∈ `[0, 254]`. Used by
`vp9_write_prob_diff_update` (`vp9_subexp.c:112`).

Encode `encode_term_subexp` (`vp9_subexp.c:100`):
```
if word < 16:           emit "0" + write_literal(word, 4)         // 5 bits
elif word < 32:         emit "10" + write_literal(word - 16, 4)    // 6 bits
elif word < 64:         emit "110" + write_literal(word - 32, 5)   // 8 bits
else:                   emit "111" + encode_uniform(word - 64)     // 10 or 11 bits
```

Where `encode_uniform` (`vp9_subexp.c:84`):
```
const l = 8, m = (1 << l) - 191 = 65
if v < m:  write_literal(v, l - 1 = 7)
else:      write_literal(m + ((v - m) >> 1), 7)
           write_literal((v - m) & 1, 1)
```

The decoder side `decode_term_subexp` (`vp9_dsubexp.c:60`) is the
mirror:
```
if !read_bit(): return read_literal(4)
if !read_bit(): return read_literal(4) + 16
if !read_bit(): return read_literal(5) + 32
return decode_uniform() + 64
```

`decode_uniform` (`vp9_dsubexp.c:23`):
```
v = read_literal(7)
return v < 65 ? v : (v << 1) - 65 + read_bit()
```

The probability *recentering* maps the actual `(old, new)` pair into the
`delp` codeword. `remap_prob` (`vp9_subexp.c:44`) and its inverse
`inv_remap_prob` (`vp9_dsubexp.c:30`) round-trip via a 254-entry
`map_table` / `inv_map_table` (verbatim in §5.6).

`vp9_diff_update_prob` (`vp9_dsubexp.c:67`) is the decoder entry point:

```c
void vp9_diff_update_prob(vpx_reader *r, vpx_prob *p) {
  if (vpx_read(r, DIFF_UPDATE_PROB)) {
    const int delp = decode_term_subexp(r);
    *p = (vpx_prob)inv_remap_prob(delp, *p);
  }
}
```

The encoder side is `vp9_cond_prob_diff_update` (`vp9_subexp.c:182`):

```c
void vp9_cond_prob_diff_update(vpx_writer *w, vpx_prob *oldp,
                               const unsigned int ct[2]) {
  const vpx_prob upd = DIFF_UPDATE_PROB;   // 252
  vpx_prob newp = get_binary_prob(ct[0], ct[1]);
  const int64_t savings =
      vp9_prob_diff_update_savings_search(ct, *oldp, &newp, upd);
  if (savings > 0) {
    vpx_write(w, 1, upd);
    vp9_write_prob_diff_update(w, newp, *oldp);
    *oldp = newp;
  } else {
    vpx_write(w, 0, upd);
  }
}
```

### 4.6 MV probability updates

`vp9_write_nmv_probs` (`vp9_encodemv.c:172`). The MV probs use a
*different* update protocol (`MV_UPDATE_PROB = 252`) and a 7-bit literal
for the new probability (not sub-exponential):

```c
static int update_mv(vpx_writer *w, const unsigned int ct[2], vpx_prob *cur_p,
                     vpx_prob upd_p) {
  const vpx_prob new_p = get_binary_prob(ct[0], ct[1]) | 1;
  const int update = cost_branch256(ct, *cur_p) + vp9_cost_zero(upd_p) >
                     cost_branch256(ct, new_p) + vp9_cost_one(upd_p) +
                         (7 << VP9_PROB_COST_SHIFT);
  vpx_write(w, update, upd_p);
  if (update) {
    *cur_p = new_p;
    vpx_write_literal(w, new_p >> 1, 7);
  }
  return update;
}
```

So each MV probability that updates is encoded as one boolean bit
("update or not") followed by a fixed-bit 7-bit literal (the top 7 bits
of the new probability; the low bit is forced to 1 via `| 1`).

The full write sequence in `vp9_write_nmv_probs`:
```
joints: tree of MV_JOINTS=4, 3 update_mv calls.
per-component (×2):
  sign      : 1 update_mv
  classes   : tree of MV_CLASSES=11, 10 update_mv calls
  class0    : tree of CLASS0_SIZE=2, 1 update_mv call
  bits      : MV_OFFSET_BITS=10, 10 update_mv calls
per-component (×2):
  class0_fp[d] for d in 0..CLASS0_SIZE-1: tree of MV_FP_SIZE=4 each
  fp        : tree of MV_FP_SIZE=4, 3 update_mv calls
if usehp (per-component ×2):
  class0_hp : 1 update_mv
  hp        : 1 update_mv
```

### 4.7 Per-block coding of an MV

`vp9_encode_mv` (`vp9_encodemv.c:210`). The MV is coded as `(mv - ref)`,
so the value is always a difference:

```c
void vp9_encode_mv(VP9_COMP *cpi, vpx_writer *w, const MV *mv, const MV *ref,
                   const nmv_context *mvctx, int usehp,
                   unsigned int *const max_mv_magnitude) {
  const MV diff = { mv->row - ref->row, mv->col - ref->col };
  const MV_JOINT_TYPE j = vp9_get_mv_joint(&diff);
  usehp = usehp && use_mv_hp(ref);

  vp9_write_token(w, vp9_mv_joint_tree, mvctx->joints, &mv_joint_encodings[j]);
  if (mv_joint_vertical(j))
    encode_mv_component(w, diff.row, &mvctx->comps[0], usehp);

  if (mv_joint_horizontal(j))
    encode_mv_component(w, diff.col, &mvctx->comps[1], usehp);
  ...
}
```

`encode_mv_component` (`vp9_encodemv.c:31`):
```
sign = comp < 0
mag = |comp|
mv_class = vp9_get_mv_class(mag - 1, &offset)
                  // class in {MV_CLASS_0..MV_CLASS_10}; sets offset within class
d  = offset >> 3              // integer-pel data
fr = (offset >> 1) & 3        // fractional-pel data
hp = offset & 1               // high-precision (1/8 pel) bit

write(sign, mvcomp->sign)
write_token(vp9_mv_class_tree, mvcomp->classes, &mv_class_encodings[mv_class])

if mv_class == MV_CLASS_0:
  write(d, mvcomp->class0[0])              // 1 bit since CLASS0_BITS == 1
else:
  n = mv_class + CLASS0_BITS - 1
  for i in 0..n-1: write((d >> i) & 1, mvcomp->bits[i])

write_token(vp9_mv_fp_tree,
            mv_class == MV_CLASS_0 ? mvcomp->class0_fp[d] : mvcomp->fp,
            &mv_fp_encodings[fr])

if usehp:
  write(hp, mv_class == MV_CLASS_0 ? mvcomp->class0_hp : mvcomp->hp)
```

`vp9_get_mv_class` (`vp9_entropymv.c:103`) maps the unsigned magnitude
into one of 11 classes via the `log_in_base_2[1024]` table
(`vp9_entropymv.c:56-97`; reproduced verbatim in §5.7.2) and returns the
within-class offset.

### 4.8 Adaptation: `merge_probs` and `mode_mv_merge_probs`

The decoder and encoder must apply the *same* update to `cm->fc` after
each frame (post-reconstruction) so the next frame starts from the same
priors.

`merge_probs` (`vpx_dsp/prob.h:69`):

```c
static INLINE vpx_prob merge_probs(vpx_prob pre_prob, const unsigned int ct[2],
                                   unsigned int count_sat,
                                   unsigned int max_update_factor) {
  const vpx_prob prob = get_binary_prob(ct[0], ct[1]);
  const unsigned int count = VPXMIN(ct[0] + ct[1], count_sat);
  const unsigned int factor = max_update_factor * count / count_sat;
  return weighted_prob(pre_prob, prob, factor);
}
```

`mode_mv_merge_probs` (`vpx_dsp/prob.h:84`) is a specialization where
`count_sat = MODE_MV_COUNT_SAT = 20` and `max_update_factor = 128`,
implemented via a precomputed `count_to_update_factor[21]` table
(`vpx_dsp/prob.h:79`; reproduced in §5.5).

`weighted_prob` (`vpx_dsp/prob.h:65`):
```c
return ROUND_POWER_OF_TWO(prob1 * (256 - factor) + prob2 * factor, 8);
```

For coefficient probs, `adapt_coef_probs` (`vp9_entropy.c:1055`) uses
its own `count_sat` and `update_factor` (`vp9_entropy.c:1048-1053`):
```
COEF_COUNT_SAT                 = 24
COEF_MAX_UPDATE_FACTOR         = 112
COEF_COUNT_SAT_KEY             = 24
COEF_MAX_UPDATE_FACTOR_KEY     = 112
COEF_COUNT_SAT_AFTER_KEY       = 24
COEF_MAX_UPDATE_FACTOR_AFTER_KEY = 128
```

`vp9_adapt_coef_probs` (`vp9_entropy.c:1084`) selects between these based
on frame type: keyframe → KEY constants; first frame after keyframe →
AFTER_KEY constants; otherwise → vanilla constants.

`vp9_adapt_mode_probs` (`vp9_entropymode.c:340`) walks every per-context
slot of the FRAME_CONTEXT calling either `mode_mv_merge_probs` (for
binary contexts) or `vpx_tree_merge_probs` (for tree-coded contexts;
`vpx_dsp/prob.c:44`).

`vpx_tree_merge_probs` (`vpx_dsp/prob.c:26`) walks the tree recursively,
computing left/right counts from `counts[]` and calling
`mode_mv_merge_probs` at each internal node.

`vp9_adapt_mv_probs` (`vp9_entropymv.c:154`) does the same for the MV
context.

### 4.9 The bit-exact contract

For two implementations of this coder to produce byte-identical output
from identical input, they must agree on every one of:

1. **Probability quantization.** `get_prob` (`vpx_dsp/prob.h:48`):
   `p = ((uint64_t)num * 256 + (den >> 1)) / den`, then clipped to
   `[1, 255]`. The `(uint64_t)` cast matters: it stops 32-bit overflow
   when `num * 256` exceeds `UINT32_MAX`.

2. **The `split` formulas.** Encoder: `1 + (((range - 1) * prob) >> 8)`.
   Decoder: `(range * prob + (256 - prob)) >> 8`. These are identical for
   all `range, prob` in `[1, 255]` (proof in §3.6). Do not "optimize"
   either.

3. **The `vpx_norm` LUT** (§5.5) — bit-identical 256-byte table.

4. **The carry-propagation fixup** (`bitwriter.h:88`): when the
   to-be-emitted byte has its high bit set, walk back through 0xff bytes
   resetting them and increment the previous byte. The Rust port must
   handle the same byte sequence.

5. **The stop-encode tail**: 32 zero bits via the arithmetic coder, then
   the optional `0` byte if the last byte matches the superframe-index
   marker pattern (`0xc0..0xdf`).

6. **The probability merge formulas**: `mode_mv_merge_probs` rounded as
   `weighted_prob` rounds (round-to-nearest, ties up,
   `ROUND_POWER_OF_TWO`).

7. **The default tables** (every table in §5). Bit-identical.

8. **The scan orders, iscans, and neighbor tables** (§5.3).

9. **The pareto8 row index** used for trailing token nodes:
   `context_tree[PIVOT_NODE] - 1` (a 1-based index into the 255-row
   table; pivot=0 is illegal because probs are in `[1,255]`).

10. **The token encoding** in `vp9_coef_encodings`
    (`vp9_tokenize.c:306`) — the `(value, len)` flattening of
    `vp9_coef_tree` — must match.

Items 1, 2, 4 are the parts most easily broken by well-intentioned Rust
"cleanups". Items 3, 7, 8 are mechanical copy jobs. Items 5, 6, 9, 10
trip people who don't know they are decisions.

---

## 5. Tables and constants

This is the largest section. Every load-bearing constant in the
bitstream / entropy path is reproduced here verbatim, with file:line.
Tables are grouped by domain.

### 5.1 Sync codes, profile codes, marker constants

`third_party/libvpx/vp9/common/vp9_common.h:49-53`:
```c
#define VP9_SYNC_CODE_0 0x49
#define VP9_SYNC_CODE_1 0x83
#define VP9_SYNC_CODE_2 0x42

#define VP9_FRAME_MARKER 0x2
```

`third_party/libvpx/vp9/common/vp9_quant_common.h:24` and
`vp9_loopfilter.h:24`:
```c
#define QINDEX_BITS 8           // base_qindex is 8 bits
#define MAX_LOOP_FILTER 63
#define MAXQ 255
```

`vp9/common/vp9_onyxc_int.h:37-45`:
```c
#define REFS_PER_FRAME 3
#define REF_FRAMES_LOG2 3
#define REF_FRAMES (1 << REF_FRAMES_LOG2)        // = 8
#define FRAME_CONTEXTS_LOG2 2
#define FRAME_CONTEXTS (1 << FRAME_CONTEXTS_LOG2) // = 4
```

`vp9_loopfilter.h:29-30`:
```c
#define MAX_REF_LF_DELTAS 4
#define MAX_MODE_LF_DELTAS 2
```

`vp9_seg_common.h:20-26`:
```c
#define SEGMENT_DELTADATA 0
#define SEGMENT_ABSDATA 1
#define MAX_SEGMENTS 8
#define SEG_TREE_PROBS (MAX_SEGMENTS - 1)         // = 7
#define PREDICTION_PROBS 3
```

`vp9_filter.h:23-32`:
```c
#define EIGHTTAP 0
#define EIGHTTAP_SMOOTH 1
#define EIGHTTAP_SHARP 2
#define SWITCHABLE_FILTERS 3
#define BILINEAR 3
#define FOURTAP 4
#define SWITCHABLE_FILTER_CONTEXTS (SWITCHABLE_FILTERS + 1)   // = 4
#define SWITCHABLE 4
```

`vp9_entropy.h:24, 142-146`:
```c
#define DIFF_UPDATE_PROB 252
#define UNCONSTRAINED_NODES 3
#define PIVOT_NODE 2
#define MODEL_NODES (ENTROPY_NODES - UNCONSTRAINED_NODES)     // 8
```

`vp9_entropymv.h:35`:
```c
#define MV_UPDATE_PROB 252
```

### 5.2 The vpx_norm LUT (encoder + decoder)

`third_party/libvpx/vpx_dsp/prob.c:13-24` — used to compute the
post-renormalization shift after each boolean event. `vpx_norm[r]` =
`7 - floor(log2(r))` for `r >= 1`, and `vpx_norm[0] = 0`:

```c
const uint8_t vpx_norm[256] = {
  0, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
```

`vpx_dsp/prob.h:79-82` — used by `mode_mv_merge_probs`:

```c
static const int count_to_update_factor[MODE_MV_COUNT_SAT + 1] = {
  0,  6,  12, 19, 25, 32,  38,  44,  51,  57, 64,
  70, 76, 83, 89, 96, 102, 108, 115, 121, 128
};
```

### 5.3 Trees

#### 5.3.1 Intra mode tree

`vp9_entropymode.c:245-255`:
```c
const vpx_tree_index vp9_intra_mode_tree[TREE_SIZE(INTRA_MODES)] = {
  -DC_PRED,   2,          /* 0 = DC_NODE */
  -TM_PRED,   4,          /* 1 = TM_NODE */
  -V_PRED,    6,          /* 2 = V_NODE */
  8,          12,         /* 3 = COM_NODE */
  -H_PRED,    10,         /* 4 = H_NODE */
  -D135_PRED, -D117_PRED, /* 5 = D135_NODE */
  -D45_PRED,  14,         /* 6 = D45_NODE */
  -D63_PRED,  16,         /* 7 = D63_NODE */
  -D153_PRED, -D207_PRED  /* 8 = D153_NODE */
};
```

#### 5.3.2 Inter mode tree

`vp9_entropymode.c:257-260`:
```c
const vpx_tree_index vp9_inter_mode_tree[TREE_SIZE(INTER_MODES)] = {
  -INTER_OFFSET(ZEROMV), 2, -INTER_OFFSET(NEARESTMV), 4, -INTER_OFFSET(NEARMV),
  -INTER_OFFSET(NEWMV)
};
```

Recall `INTER_OFFSET(mode) = mode - NEARESTMV` (`vp9_entropymode.h:27`),
so `INTER_OFFSET(NEARESTMV)=0`, `(NEARMV)=1`, `(ZEROMV)=2`,
`(NEWMV)=3`.

#### 5.3.3 Partition tree

`vp9_entropymode.c:262-264`:
```c
const vpx_tree_index vp9_partition_tree[TREE_SIZE(PARTITION_TYPES)] = {
  -PARTITION_NONE, 2, -PARTITION_HORZ, 4, -PARTITION_VERT, -PARTITION_SPLIT
};
```

#### 5.3.4 Switchable interpolation filter tree

`vp9_entropymode.c:337-338`:
```c
const vpx_tree_index vp9_switchable_interp_tree[TREE_SIZE(SWITCHABLE_FILTERS)] =
    { -EIGHTTAP, 2, -EIGHTTAP_SMOOTH, -EIGHTTAP_SHARP };
```

#### 5.3.5 Coefficient tree

`vp9_tokenize.c:75-88`:
```c
const vpx_tree_index vp9_coef_tree[TREE_SIZE(ENTROPY_TOKENS)] = {
  -EOB_TOKEN, 2,                       // 0  = EOB
  -ZERO_TOKEN, 4,                      // 1  = ZERO
  -ONE_TOKEN, 6,                       // 2  = ONE
  8, 12,                               // 3  = LOW_VAL
  -TWO_TOKEN, 10,                      // 4  = TWO
  -THREE_TOKEN, -FOUR_TOKEN,           // 5  = THREE
  14, 16,                              // 6  = HIGH_LOW
  -CATEGORY1_TOKEN, -CATEGORY2_TOKEN,  // 7  = CAT_ONE
  18, 20,                              // 8  = CAT_THREEFOUR
  -CATEGORY3_TOKEN, -CATEGORY4_TOKEN,  // 9  = CAT_THREE
  -CATEGORY5_TOKEN, -CATEGORY6_TOKEN   // 10 = CAT_FIVE
};
```

#### 5.3.6 Unconstrained coefficient tree (used after PIVOT_NODE)

`vp9_entropy.c:20-29`:
```c
const vpx_tree_index vp9_coef_con_tree[TREE_SIZE(ENTROPY_TOKENS)] = {
  2, 6,                                // 0 = LOW_VAL
  -TWO_TOKEN, 4,                       // 1 = TWO
  -THREE_TOKEN, -FOUR_TOKEN,           // 2 = THREE
  8, 10,                               // 3 = HIGH_LOW
  -CATEGORY1_TOKEN, -CATEGORY2_TOKEN,  // 4 = CAT_ONE
  12, 14,                              // 5 = CAT_THREEFOUR
  -CATEGORY3_TOKEN, -CATEGORY4_TOKEN,  // 6 = CAT_THREE
  -CATEGORY5_TOKEN, -CATEGORY6_TOKEN   // 7 = CAT_FIVE
};
```

#### 5.3.7 MV joint, class, class0, fp trees

`vp9_entropymv.c:14-28`:
```c
const vpx_tree_index vp9_mv_joint_tree[TREE_SIZE(MV_JOINTS)] = {
  -MV_JOINT_ZERO, 2, -MV_JOINT_HNZVZ, 4, -MV_JOINT_HZVNZ, -MV_JOINT_HNZVNZ
};

const vpx_tree_index vp9_mv_class_tree[TREE_SIZE(MV_CLASSES)] = {
  -MV_CLASS_0, 2,           -MV_CLASS_1, 4,           6,
  8,           -MV_CLASS_2, -MV_CLASS_3, 10,          12,
  -MV_CLASS_4, -MV_CLASS_5, -MV_CLASS_6, 14,          16,
  18,          -MV_CLASS_7, -MV_CLASS_8, -MV_CLASS_9, -MV_CLASS_10,
};

const vpx_tree_index vp9_mv_class0_tree[TREE_SIZE(CLASS0_SIZE)] = { -0, -1 };

const vpx_tree_index vp9_mv_fp_tree[TREE_SIZE(MV_FP_SIZE)] = { -0, 2,  -1,
                                                               4,  -2, -3 };
```

#### 5.3.8 Segment tree

`vp9_seg_common.c:58-60`:
```c
const vpx_tree_index vp9_segment_tree[TREE_SIZE(MAX_SEGMENTS)] = {
  2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7
};
```

(Note the unusual `0` as a leaf — this encodes `segment_id == 0`. The
decoder side sees `-0` returned, then negates: `vpx_read_tree` returns
`-leaf`, but `-0 == 0`, so segment_id 0 falls out correctly.)

#### 5.3.9 Static token encodings for trees in `vp9_bitstream.c`

`vp9_bitstream.c:42-55`:
```c
static const struct vp9_token intra_mode_encodings[INTRA_MODES] = {
  { 0, 1 },  { 6, 3 },   { 28, 5 },  { 30, 5 }, { 58, 6 },
  { 59, 6 }, { 126, 7 }, { 127, 7 }, { 62, 6 }, { 2, 2 }
};
static const struct vp9_token
    switchable_interp_encodings[SWITCHABLE_FILTERS] = { { 0, 1 },
                                                        { 2, 2 },
                                                        { 3, 2 } };
static const struct vp9_token partition_encodings[PARTITION_TYPES] = {
  { 0, 1 }, { 2, 2 }, { 6, 3 }, { 7, 3 }
};
static const struct vp9_token inter_mode_encodings[INTER_MODES] = {
  { 2, 2 }, { 6, 3 }, { 0, 1 }, { 7, 3 }
};
```

`vp9_tokenize.c:306-309`:
```c
const struct vp9_token vp9_coef_encodings[ENTROPY_TOKENS] = {
  { 2, 2 },  { 6, 3 },   { 28, 5 },  { 58, 6 },  { 59, 6 },  { 60, 6 },
  { 61, 6 }, { 124, 7 }, { 125, 7 }, { 126, 7 }, { 127, 7 }, { 0, 1 }
};
```

These are precomputable from the tree definitions via
`vp9_tokens_from_tree`; they ship as static constants so the encoder
doesn't need to run that routine at init time.

---

## 6. RTCD / SIMD surfaces

> Note: this section appended by the Stage 1 driver to fix A4 truncation —
> the dispatched agent's transcript ended mid-§5 and §6/§7 were missing.
> Stage 1 review pass should re-dispatch A4 if this synthesis is
> insufficient for Stage 2 LLD work.

The entropy + bitstream subsystem is **almost entirely scalar** in libvpx.
The arithmetic coder is a tight sequential state machine (one bit at a
time with a serial dependency on `range`/`lowvalue`); SIMD does not help.
The same holds for tree-based encoding (`vp9_write_tree` is a branching
walk) and for header writes (`vpx_wb_write_literal` etc.).

The following kernels do have SIMD in libvpx and touch the entropy
pipeline indirectly:

| Kernel | Where used | ISAs (libvpx) | File:line |
|---|---|---|---|
| `vp9_quantize_fp` / `vp9_quantize_fp_32x32` | per-block quant before tokenization | scalar / SSE2 / AVX2 / NEON | `vp9/encoder/vp9_quantize.c`; SIMD under `vp9/encoder/x86/`, `arm/neon/` |
| `vpx_quantize_b` / `vpx_quantize_b_32x32` | adaptive quant | scalar / SSE2 / AVX / AVX2 / NEON | `vpx_dsp/quantize.c`; SIMD under `vpx_dsp/{x86,arm}/` |
| `vp9_block_error` / `vp9_block_error_fp` | RD cost — not on the cpu-used 6–8 path | scalar / SSE2 / AVX2 / NEON | `vp9/encoder/vp9_encodemb.c`; SIMD under `vp9/encoder/x86/`, `arm/neon/` |

Quantization owns scan-order traversal indirectly because the
post-quantize EOB scan is what determines which coefficients reach the
token writer. Strictly speaking these kernels are in A2/A5 territory
(quant lives in the encoder core's tokenize pipeline); listed here only
to mark the boundary.

**The boolean coder itself ships no SIMD** in libvpx. The Rust port
matches: scalar-only `vpx_writer` / `vpx_reader` equivalents, no
per-arch implementations needed for §2.1–§2.4.

Probability-table memcpy at frame-context reset (e.g.
`vp9_setup_past_independence`) uses `memcpy`, which libc may SIMD-vectorize
under the hood; not on our RTCD critical path.

---

## 7. Open questions for LLD

> Note: this section also a Stage 1-driver synthesis (A4 truncated before
> writing it). Items here are recovered from cross-references in §2–§5
> plus standard concerns. Devil's advocate (A7) should pressure-test.

1. **Encoder–decoder probability-table sharing.** §2.6 documents
   `FRAME_CONTEXT` and §5 transcribes the default probability tables.
   These tables are read by BOTH the encoder (to estimate bit cost) and
   the decoder (to update its state). Decision for the LLD: do both
   crates depend on `rvp9-core` for a single source of truth, or does
   each crate ship its own copy of the constants (risking drift)?
   Locked answer per PLAN.md §5: shared via `rvp9-core`.

2. **Backward probability update protocol.** §3 covers writing the
   compressed header. The decoder's backward-adaptation behaviour
   depends on the encoder writing the right update probabilities. The
   `vp9_coef_update_prob` table (verbatim in §5) is involved. What's the
   bit-exact contract here? Specifically: under what conditions does
   the encoder choose to update vs not update each coefficient
   probability?

3. **Trellis-quantization on the real-time path.** libvpx's
   `vp9_optimize_b` is a trellis search that re-tokenizes after quant
   to minimise rate-distortion cost. At cpu-used 6–8 this is typically
   skipped (`sf->optimize_coefficients = 0`). Confirm and document.

4. **Range-coder renormalization edge cases.** §3.5 documents the bool
   coder write state machine. What happens at frame-end when buffered
   bits don't fill a byte boundary? `vpx_stop_encode` handles the
   final flush — Rust port must mirror exactly for byte parity.

5. **Sign-bit and magnitude coding for coefficients.** The
   high-frequency coefficients use a Golomb-Rice tail in `WRITE_COEF_CONTINUE`
   (token codes 8–11, "categories 1-6"). Verify the exact RLE state
   machine in the LLD; this is one of the easiest places to silently
   diverge from libvpx.

6. **Tile-data byte alignment.** Each tile column starts on a byte
   boundary in the bitstream. The encoder pads with stuffing bits as
   needed. Document the exact alignment + size-prefix layout for tile
   columns in the LLD.

7. **Superframe index encoding.** When multiple frames are packed
   into one container payload (e.g. a hidden ALTREF frame followed by
   the visible frame referencing it), the superframe marker byte at the
   end specifies frame sizes. Layout per the spec; tested by the Stage 0
   conformance vectors that include `show-existing-frame` cases.

8. **High-bit-depth coefficient ranges (Q1 dependency).** If we ship
   HBD in v1, the coefficient encoding pathway changes (token values
   span a larger range, requiring extra raw bits). §5's tables are
   8-bit only — HBD tables live in separate `#if CONFIG_VP9_HIGHBITDEPTH`
   blocks not transcribed here. Decide HBD scope first; if in, re-extract
   the HBD tables.

9. **Probability-update damping during pre-encode tentative pack.** §3
   documents the encode sequence; the "tentative pack" that runs in
   `vp9_encodedframe_overshoot` (a major surprise flagged in A5)
   touches frame probabilities — does this leak into the
   final-pack bitstream or is it a pure cost-estimate side path? Verify.

10. **Show-existing-frame semantics.** Conformance vector
    `vp90-2-10-show-existing-frame.webm` exercises this path. The
    bitstream is essentially a single marker referencing one of the 8
    reference-frame slots. Document the exact wire format and the
    encoder triggers (only used by SVC and "hidden ALTREF" cases).
