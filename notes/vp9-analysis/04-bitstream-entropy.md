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

### 5.4 Intra-mode probability tables

#### 5.4.1 `vp9_kf_y_mode_prob` --- keyframe Y intra mode probabilities, conditioned on (above, left)

`vp9_entropymode.c:16-147` (indexed `[above][left][node]`,
9 internal nodes per (above, left) pair, 10 modes x 10 modes = 100 rows):

```c
const vpx_prob vp9_kf_y_mode_prob[INTRA_MODES][INTRA_MODES][INTRA_MODES - 1] = {
  {
      // above = dc
      { 137, 30, 42, 148, 151, 207, 70, 52, 91 },   // left = dc
      { 92, 45, 102, 136, 116, 180, 74, 90, 100 },  // left = v
      { 73, 32, 19, 187, 222, 215, 46, 34, 100 },   // left = h
      { 91, 30, 32, 116, 121, 186, 93, 86, 94 },    // left = d45
      { 72, 35, 36, 149, 68, 206, 68, 63, 105 },    // left = d135
      { 73, 31, 28, 138, 57, 124, 55, 122, 151 },   // left = d117
      { 67, 23, 21, 140, 126, 197, 40, 37, 171 },   // left = d153
      { 86, 27, 28, 128, 154, 212, 45, 43, 53 },    // left = d207
      { 74, 32, 27, 107, 86, 160, 63, 134, 102 },   // left = d63
      { 59, 67, 44, 140, 161, 202, 78, 67, 119 }    // left = tm
  },
  {
      // above = v
      { 63, 36, 126, 146, 123, 158, 60, 90, 96 },   // left = dc
      { 43, 46, 168, 134, 107, 128, 69, 142, 92 },  // left = v
      { 44, 29, 68, 159, 201, 177, 50, 57, 77 },    // left = h
      { 58, 38, 76, 114, 97, 172, 78, 133, 92 },    // left = d45
      { 46, 41, 76, 140, 63, 184, 69, 112, 57 },    // left = d135
      { 38, 32, 85, 140, 46, 112, 54, 151, 133 },   // left = d117
      { 39, 27, 61, 131, 110, 175, 44, 75, 136 },   // left = d153
      { 52, 30, 74, 113, 130, 175, 51, 64, 58 },    // left = d207
      { 47, 35, 80, 100, 74, 143, 64, 163, 74 },    // left = d63
      { 36, 61, 116, 114, 128, 162, 80, 125, 82 }   // left = tm
  },
  {
      // above = h
      { 82, 26, 26, 171, 208, 204, 44, 32, 105 },  // left = dc
      { 55, 44, 68, 166, 179, 192, 57, 57, 108 },  // left = v
      { 42, 26, 11, 199, 241, 228, 23, 15, 85 },   // left = h
      { 68, 42, 19, 131, 160, 199, 55, 52, 83 },   // left = d45
      { 58, 50, 25, 139, 115, 232, 39, 52, 118 },  // left = d135
      { 50, 35, 33, 153, 104, 162, 64, 59, 131 },  // left = d117
      { 44, 24, 16, 150, 177, 202, 33, 19, 156 },  // left = d153
      { 55, 27, 12, 153, 203, 218, 26, 27, 49 },   // left = d207
      { 53, 49, 21, 110, 116, 168, 59, 80, 76 },   // left = d63
      { 38, 72, 19, 168, 203, 212, 50, 50, 107 }   // left = tm
  },
  {
      // above = d45
      { 103, 26, 36, 129, 132, 201, 83, 80, 93 },  // left = dc
      { 59, 38, 83, 112, 103, 162, 98, 136, 90 },  // left = v
      { 62, 30, 23, 158, 200, 207, 59, 57, 50 },   // left = h
      { 67, 30, 29, 84, 86, 191, 102, 91, 59 },    // left = d45
      { 60, 32, 33, 112, 71, 220, 64, 89, 104 },   // left = d135
      { 53, 26, 34, 130, 56, 149, 84, 120, 103 },  // left = d117
      { 53, 21, 23, 133, 109, 210, 56, 77, 172 },  // left = d153
      { 77, 19, 29, 112, 142, 228, 55, 66, 36 },   // left = d207
      { 61, 29, 29, 93, 97, 165, 83, 175, 162 },   // left = d63
      { 47, 47, 43, 114, 137, 181, 100, 99, 95 }   // left = tm
  },
  {
      // above = d135
      { 69, 23, 29, 128, 83, 199, 46, 44, 101 },   // left = dc
      { 53, 40, 55, 139, 69, 183, 61, 80, 110 },   // left = v
      { 40, 29, 19, 161, 180, 207, 43, 24, 91 },   // left = h
      { 60, 34, 19, 105, 61, 198, 53, 64, 89 },    // left = d45
      { 52, 31, 22, 158, 40, 209, 58, 62, 89 },    // left = d135
      { 44, 31, 29, 147, 46, 158, 56, 102, 198 },  // left = d117
      { 35, 19, 12, 135, 87, 209, 41, 45, 167 },   // left = d153
      { 55, 25, 21, 118, 95, 215, 38, 39, 66 },    // left = d207
      { 51, 38, 25, 113, 58, 164, 70, 93, 97 },    // left = d63
      { 47, 54, 34, 146, 108, 203, 72, 103, 151 }  // left = tm
  },
  {
      // above = d117
      { 64, 19, 37, 156, 66, 138, 49, 95, 133 },   // left = dc
      { 46, 27, 80, 150, 55, 124, 55, 121, 135 },  // left = v
      { 36, 23, 27, 165, 149, 166, 54, 64, 118 },  // left = h
      { 53, 21, 36, 131, 63, 163, 60, 109, 81 },   // left = d45
      { 40, 26, 35, 154, 40, 185, 51, 97, 123 },   // left = d135
      { 35, 19, 34, 179, 19, 97, 48, 129, 124 },   // left = d117
      { 36, 20, 26, 136, 62, 164, 33, 77, 154 },   // left = d153
      { 45, 18, 32, 130, 90, 157, 40, 79, 91 },    // left = d207
      { 45, 26, 28, 129, 45, 129, 49, 147, 123 },  // left = d63
      { 38, 44, 51, 136, 74, 162, 57, 97, 121 }    // left = tm
  },
  {
      // above = d153
      { 75, 17, 22, 136, 138, 185, 32, 34, 166 },  // left = dc
      { 56, 39, 58, 133, 117, 173, 48, 53, 187 },  // left = v
      { 35, 21, 12, 161, 212, 207, 20, 23, 145 },  // left = h
      { 56, 29, 19, 117, 109, 181, 55, 68, 112 },  // left = d45
      { 47, 29, 17, 153, 64, 220, 59, 51, 114 },   // left = d135
      { 46, 16, 24, 136, 76, 147, 41, 64, 172 },   // left = d117
      { 34, 17, 11, 108, 152, 187, 13, 15, 209 },  // left = d153
      { 51, 24, 14, 115, 133, 209, 32, 26, 104 },  // left = d207
      { 55, 30, 18, 122, 79, 179, 44, 88, 116 },   // left = d63
      { 37, 49, 25, 129, 168, 164, 41, 54, 148 }   // left = tm
  },
  {
      // above = d207
      { 82, 22, 32, 127, 143, 213, 39, 41, 70 },   // left = dc
      { 62, 44, 61, 123, 105, 189, 48, 57, 64 },   // left = v
      { 47, 25, 17, 175, 222, 220, 24, 30, 86 },   // left = h
      { 68, 36, 17, 106, 102, 206, 59, 74, 74 },   // left = d45
      { 57, 39, 23, 151, 68, 216, 55, 63, 58 },    // left = d135
      { 49, 30, 35, 141, 70, 168, 82, 40, 115 },   // left = d117
      { 51, 25, 15, 136, 129, 202, 38, 35, 139 },  // left = d153
      { 68, 26, 16, 111, 141, 215, 29, 28, 28 },   // left = d207
      { 59, 39, 19, 114, 75, 180, 77, 104, 42 },   // left = d63
      { 40, 61, 26, 126, 152, 206, 61, 59, 93 }    // left = tm
  },
  {
      // above = d63
      { 78, 23, 39, 111, 117, 170, 74, 124, 94 },   // left = dc
      { 48, 34, 86, 101, 92, 146, 78, 179, 134 },   // left = v
      { 47, 22, 24, 138, 187, 178, 68, 69, 59 },    // left = h
      { 56, 25, 33, 105, 112, 187, 95, 177, 129 },  // left = d45
      { 48, 31, 27, 114, 63, 183, 82, 116, 56 },    // left = d135
      { 43, 28, 37, 121, 63, 123, 61, 192, 169 },   // left = d117
      { 42, 17, 24, 109, 97, 177, 56, 76, 122 },    // left = d153
      { 58, 18, 28, 105, 139, 182, 70, 92, 63 },    // left = d207
      { 46, 23, 32, 74, 86, 150, 67, 183, 88 },     // left = d63
      { 36, 38, 48, 92, 122, 165, 88, 137, 91 }     // left = tm
  },
  {
      // above = tm
      { 65, 70, 60, 155, 159, 199, 61, 60, 81 },    // left = dc
      { 44, 78, 115, 132, 119, 173, 71, 112, 93 },  // left = v
      { 39, 38, 21, 184, 227, 206, 42, 32, 64 },    // left = h
      { 58, 47, 36, 124, 137, 193, 80, 82, 78 },    // left = d45
      { 49, 50, 35, 144, 95, 205, 63, 78, 59 },     // left = d135
      { 41, 53, 52, 148, 71, 142, 65, 128, 51 },    // left = d117
      { 40, 36, 28, 143, 143, 202, 40, 55, 137 },   // left = d153
      { 52, 34, 29, 129, 183, 227, 42, 35, 43 },    // left = d207
      { 42, 44, 44, 104, 105, 164, 64, 130, 80 },   // left = d63
      { 43, 81, 53, 140, 169, 204, 68, 84, 72 }     // left = tm
  }
};
```

#### 5.4.2 `vp9_kf_uv_mode_prob` --- keyframe UV intra mode probabilities, conditioned on Y mode

`vp9_entropymode.c:149-160`:
```c
const vpx_prob vp9_kf_uv_mode_prob[INTRA_MODES][INTRA_MODES - 1] = {
  { 144, 11, 54, 157, 195, 130, 46, 58, 108 },   // y = dc
  { 118, 15, 123, 148, 131, 101, 44, 93, 131 },  // y = v
  { 113, 12, 23, 188, 226, 142, 26, 32, 125 },   // y = h
  { 120, 11, 50, 123, 163, 135, 64, 77, 103 },   // y = d45
  { 113, 9, 36, 155, 111, 157, 32, 44, 161 },    // y = d135
  { 116, 9, 55, 176, 76, 96, 37, 61, 149 },      // y = d117
  { 115, 9, 28, 141, 161, 167, 21, 25, 193 },    // y = d153
  { 120, 12, 32, 145, 195, 142, 32, 38, 86 },    // y = d207
  { 116, 12, 64, 120, 140, 125, 49, 115, 121 },  // y = d63
  { 102, 19, 66, 162, 182, 122, 35, 59, 128 }    // y = tm
};
```

#### 5.4.3 `default_if_y_probs` --- default non-keyframe Y intra-mode probabilities by block-size group

`vp9_entropymode.c:162-167`:
```c
static const vpx_prob default_if_y_probs[BLOCK_SIZE_GROUPS][INTRA_MODES - 1] = {
  { 65, 32, 18, 144, 162, 194, 41, 51, 98 },   // block_size < 8x8
  { 132, 68, 18, 165, 217, 196, 45, 40, 78 },  // block_size < 16x16
  { 173, 80, 19, 176, 240, 193, 64, 35, 46 },  // block_size < 32x32
  { 221, 135, 38, 194, 248, 121, 96, 85, 29 }  // block_size >= 32x32
};
```

#### 5.4.4 `default_if_uv_probs` --- default non-keyframe UV intra-mode probabilities by Y mode

`vp9_entropymode.c:169-180`:
```c
static const vpx_prob default_if_uv_probs[INTRA_MODES][INTRA_MODES - 1] = {
  { 120, 7, 76, 176, 208, 126, 28, 54, 103 },   // y = dc
  { 48, 12, 154, 155, 139, 90, 34, 117, 119 },  // y = v
  { 67, 6, 25, 204, 243, 158, 13, 21, 96 },     // y = h
  { 97, 5, 44, 131, 176, 139, 48, 68, 97 },     // y = d45
  { 83, 5, 42, 156, 111, 152, 26, 49, 152 },    // y = d135
  { 80, 5, 58, 178, 74, 83, 33, 62, 145 },      // y = d117
  { 86, 5, 32, 154, 192, 168, 14, 22, 163 },    // y = d153
  { 85, 5, 32, 156, 216, 148, 19, 29, 73 },     // y = d207
  { 77, 7, 64, 116, 132, 122, 37, 126, 120 },   // y = d63
  { 101, 21, 107, 181, 192, 103, 19, 67, 125 }  // y = tm
};
```

### 5.5 Partition probability tables

#### 5.5.1 `vp9_kf_partition_probs` --- keyframe partition probabilities

`vp9_entropymode.c:182-207`:
```c
const vpx_prob vp9_kf_partition_probs[PARTITION_CONTEXTS]
                                     [PARTITION_TYPES - 1] = {
                                       // 8x8 -> 4x4
                                       { 158, 97, 94 },  // a/l both not split
                                       { 93, 24, 99 },   // a split, l not split
                                       { 85, 119, 44 },  // l split, a not split
                                       { 62, 59, 67 },   // a/l both split

                                       // 16x16 -> 8x8
                                       { 149, 53, 53 },  // a/l both not split
                                       { 94, 20, 48 },   // a split, l not split
                                       { 83, 53, 24 },   // l split, a not split
                                       { 52, 18, 18 },   // a/l both split

                                       // 32x32 -> 16x16
                                       { 150, 40, 39 },  // a/l both not split
                                       { 78, 12, 26 },   // a split, l not split
                                       { 67, 33, 11 },   // l split, a not split
                                       { 24, 7, 5 },     // a/l both split

                                       // 64x64 -> 32x32
                                       { 174, 35, 49 },  // a/l both not split
                                       { 68, 11, 27 },   // a split, l not split
                                       { 57, 15, 9 },    // l split, a not split
                                       { 12, 3, 3 },     // a/l both split
                                     };
```

#### 5.5.2 `default_partition_probs` --- default non-keyframe partition probabilities

`vp9_entropymode.c:209-231`:
```c
static const vpx_prob
    default_partition_probs[PARTITION_CONTEXTS][PARTITION_TYPES - 1] = {
      // 8x8 -> 4x4
      { 199, 122, 141 },  // a/l both not split
      { 147, 63, 159 },   // a split, l not split
      { 148, 133, 118 },  // l split, a not split
      { 121, 104, 114 },  // a/l both split
      // 16x16 -> 8x8
      { 174, 73, 87 },  // a/l both not split
      { 92, 41, 83 },   // a split, l not split
      { 82, 99, 50 },   // l split, a not split
      { 53, 39, 39 },   // a/l both split
      // 32x32 -> 16x16
      { 177, 58, 59 },  // a/l both not split
      { 68, 26, 63 },   // a split, l not split
      { 52, 79, 25 },   // l split, a not split
      { 17, 14, 12 },   // a/l both split
      // 64x64 -> 32x32
      { 222, 34, 30 },  // a/l both not split
      { 72, 16, 44 },   // a split, l not split
      { 58, 32, 12 },   // l split, a not split
      { 10, 7, 6 },     // a/l both split
    };
```

### 5.6 Inter / reference / skip / tx-size / interp probability tables

#### 5.6.1 `default_inter_mode_probs`

`vp9_entropymode.c:233-242`:
```c
static const vpx_prob
    default_inter_mode_probs[INTER_MODE_CONTEXTS][INTER_MODES - 1] = {
      { 2, 173, 34 },  // 0 = both zero mv
      { 7, 145, 85 },  // 1 = one zero mv + one a predicted mv
      { 7, 166, 63 },  // 2 = two predicted mvs
      { 7, 94, 66 },   // 3 = one predicted/zero and one new mv
      { 8, 64, 46 },   // 4 = two new mvs
      { 17, 81, 31 },  // 5 = one intra neighbour + x
      { 25, 29, 30 },  // 6 = two intra neighbours
    };
```

#### 5.6.2 `default_intra_inter_p`

`vp9_entropymode.c:266-268`:
```c
static const vpx_prob default_intra_inter_p[INTRA_INTER_CONTEXTS] = { 9, 102,
                                                                      187,
                                                                      225 };
```

#### 5.6.3 `default_comp_inter_p`

`vp9_entropymode.c:270-272`:
```c
static const vpx_prob default_comp_inter_p[COMP_INTER_CONTEXTS] = { 239, 183,
                                                                    119, 96,
                                                                    41 };
```

#### 5.6.4 `default_comp_ref_p`

`vp9_entropymode.c:274-275`:
```c
static const vpx_prob default_comp_ref_p[REF_CONTEXTS] = { 50, 126, 123, 221,
                                                           226 };
```

#### 5.6.5 `default_single_ref_p`

`vp9_entropymode.c:277-279`:
```c
static const vpx_prob default_single_ref_p[REF_CONTEXTS][2] = {
  { 33, 16 }, { 77, 74 }, { 142, 142 }, { 172, 170 }, { 238, 247 }
};
```

#### 5.6.6 `default_tx_probs`

`vp9_entropymode.c:281-286`:
```c
static const struct tx_probs default_tx_probs = { { { 3, 136, 37 },
                                                    { 5, 52, 13 } },

                                                  { { 20, 152 }, { 15, 101 } },

                                                  { { 100 }, { 66 } } };
```

Decoded:
- `p32x32[2][3]` = `{{3, 136, 37}, {5, 52, 13}}`
- `p16x16[2][2]` = `{{20, 152}, {15, 101}}`
- `p8x8[2][1]`   = `{{100}, {66}}`

#### 5.6.7 `default_skip_probs`

`vp9_entropymode.c:313`:
```c
static const vpx_prob default_skip_probs[SKIP_CONTEXTS] = { 192, 128, 64 };
```

#### 5.6.8 `default_switchable_interp_prob`

`vp9_entropymode.c:315-321`:
```c
static const vpx_prob default_switchable_interp_prob[SWITCHABLE_FILTER_CONTEXTS]
                                                    [SWITCHABLE_FILTERS - 1] = {
                                                      { 235, 162 },
                                                      { 36, 255 },
                                                      { 34, 3 },
                                                      { 149, 144 },
                                                    };
```


### 5.7 Coefficient model and supporting tables

#### 5.7.1 Coefficient token alphabet

`vp9_entropy.h:27-40`:
```c
// Coefficient token alphabet
#define ZERO_TOKEN 0        // 0     Extra Bits 0+0
#define ONE_TOKEN 1         // 1     Extra Bits 0+1
#define TWO_TOKEN 2         // 2     Extra Bits 0+1
#define THREE_TOKEN 3       // 3     Extra Bits 0+1
#define FOUR_TOKEN 4        // 4     Extra Bits 0+1
#define CATEGORY1_TOKEN 5   // 5-6   Extra Bits 1+1
#define CATEGORY2_TOKEN 6   // 7-10  Extra Bits 2+1
#define CATEGORY3_TOKEN 7   // 11-18 Extra Bits 3+1
#define CATEGORY4_TOKEN 8   // 19-34 Extra Bits 4+1
#define CATEGORY5_TOKEN 9   // 35-66 Extra Bits 5+1
#define CATEGORY6_TOKEN 10  // 67+   Extra Bits 14+1
#define EOB_TOKEN 11        // EOB   Extra Bits 0+0

#define ENTROPY_TOKENS 12

#define ENTROPY_NODES 11
```

Category minimum values (`vp9_entropy.h:46-51`):
```c
#define CAT1_MIN_VAL 5
#define CAT2_MIN_VAL 7
#define CAT3_MIN_VAL 11
#define CAT4_MIN_VAL 19
#define CAT5_MIN_VAL 35
#define CAT6_MIN_VAL 67
```

#### 5.7.2 Coefficient energy-class lookup `vp9_pt_energy_class`

`vp9_entropy.c:95-96`:
```c
const uint8_t vp9_pt_energy_class[ENTROPY_TOKENS] = { 0, 1, 2, 3, 3, 4,
                                                      4, 5, 5, 5, 5, 5 };
```

Mapping (token -> energy class, used by `get_coef_context` via `token_cache`):

| token | name       | energy class |
|-------|------------|--------------|
| 0     | ZERO       | 0            |
| 1     | ONE        | 1            |
| 2     | TWO        | 2            |
| 3     | THREE      | 3            |
| 4     | FOUR       | 3            |
| 5     | CAT1       | 4            |
| 6     | CAT2       | 4            |
| 7     | CAT3       | 5            |
| 8     | CAT4       | 5            |
| 9     | CAT5       | 5            |
| 10    | CAT6       | 5            |
| 11    | EOB        | 5            |

#### 5.7.3 Coefficient band translation tables

`vp9_entropy.c:46-88` --- `vp9_coefband_trans_8x8plus[1024]`. The first 22
entries are non-trivial; everything from index 22 onward is `5`:

```c
const uint8_t vp9_coefband_trans_8x8plus[1024] = {
  0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5,
  // beyond MAXBAND_INDEX+1 all values are filled as 5
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  // ... [remaining 1002 entries all = 5] ...
};
```

(Verbatim: 1024 entries; positions 0..21 are
`[0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,5]` and positions 22..1023
are all `5`.)

`vp9_entropy.c:91-93`:
```c
const uint8_t vp9_coefband_trans_4x4[16] = {
  0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5,
};
```

#### 5.7.4 Category extra-bit probabilities

`vp9_entropy.c:32-38`:
```c
const vpx_prob vp9_cat1_prob[] = { 159 };
const vpx_prob vp9_cat2_prob[] = { 165, 145 };
const vpx_prob vp9_cat3_prob[] = { 173, 148, 140 };
const vpx_prob vp9_cat4_prob[] = { 176, 155, 140, 135 };
const vpx_prob vp9_cat5_prob[] = { 180, 157, 141, 134, 130 };
const vpx_prob vp9_cat6_prob[] = { 254, 254, 254, 252, 249, 243, 230,
                                   196, 177, 153, 140, 133, 130, 129 };
```

`vp9_entropy.c:39-43` (high-bit-depth, profile 2/3 only):
```c
#if CONFIG_VP9_HIGHBITDEPTH
const vpx_prob vp9_cat6_prob_high12[] = { 255, 255, 255, 255, 254, 254,
                                          254, 252, 249, 243, 230, 196,
                                          177, 153, 140, 133, 130, 129 };
#endif
```

For 10-bit the decoder uses `vp9_cat6_prob_high12 + 2` (i.e. the
first two `255`s are skipped) --- see `vp9/encoder/vp9_tokenize.c:287`:
```c
{ vp9_cat6_prob_high12 + 2, 16, CAT6_MIN_VAL, 0 },  // CAT6 at 10 bits
```
and `vp9_cat6_prob_high12, 18, CAT6_MIN_VAL, 0` for 12-bit
(`vp9_tokenize.c:301`).

#### 5.7.5 `vp9_extra_bits[ENTROPY_TOKENS]` dispatch

`vp9_tokenize.c:260-273`:
```c
const vp9_extra_bit vp9_extra_bits[ENTROPY_TOKENS] = {
  { 0, 0, 0, zero_cost },                         // ZERO_TOKEN
  { 0, 0, 1, sign_cost },                         // ONE_TOKEN
  { 0, 0, 2, sign_cost },                         // TWO_TOKEN
  { 0, 0, 3, sign_cost },                         // THREE_TOKEN
  { 0, 0, 4, sign_cost },                         // FOUR_TOKEN
  { vp9_cat1_prob, 1, CAT1_MIN_VAL, cat1_cost },  // CATEGORY1_TOKEN
  { vp9_cat2_prob, 2, CAT2_MIN_VAL, cat2_cost },  // CATEGORY2_TOKEN
  { vp9_cat3_prob, 3, CAT3_MIN_VAL, cat3_cost },  // CATEGORY3_TOKEN
  { vp9_cat4_prob, 4, CAT4_MIN_VAL, cat4_cost },  // CATEGORY4_TOKEN
  { vp9_cat5_prob, 5, CAT5_MIN_VAL, cat5_cost },  // CATEGORY5_TOKEN
  { vp9_cat6_prob, 14, CAT6_MIN_VAL, 0 },         // CATEGORY6_TOKEN
  { 0, 0, 0, zero_cost }                          // EOB_TOKEN
};
```

#### 5.7.6 `vp9_pareto8_full` --- 255-row extension table for trailing coefficient model nodes

`vp9_entropy.c:112-368`. For each `p in 1..255`, row `p-1` holds the
8 trailing probabilities used by `vp9_coef_con_tree` when the
context's PIVOT_NODE (one-prob, the model's third node) equals `p`. This
is THE main coefficient-coder lookup; every value below is bit-exact:

```c
const vpx_prob vp9_pareto8_full[COEFF_PROB_MODELS][MODEL_NODES] = {
  { 3, 86, 128, 6, 86, 23, 88, 29 },
  { 6, 86, 128, 11, 87, 42, 91, 52 },
  { 9, 86, 129, 17, 88, 61, 94, 76 },
  { 12, 86, 129, 22, 88, 77, 97, 93 },
  { 15, 87, 129, 28, 89, 93, 100, 110 },
  { 17, 87, 129, 33, 90, 105, 103, 123 },
  { 20, 88, 130, 38, 91, 118, 106, 136 },
  { 23, 88, 130, 43, 91, 128, 108, 146 },
  { 26, 89, 131, 48, 92, 139, 111, 156 },
  { 28, 89, 131, 53, 93, 147, 114, 163 },
  { 31, 90, 131, 58, 94, 156, 117, 171 },
  { 34, 90, 131, 62, 94, 163, 119, 177 },
  { 37, 90, 132, 66, 95, 171, 122, 184 },
  { 39, 90, 132, 70, 96, 177, 124, 189 },
  { 42, 91, 132, 75, 97, 183, 127, 194 },
  { 44, 91, 132, 79, 97, 188, 129, 198 },
  { 47, 92, 133, 83, 98, 193, 132, 202 },
  { 49, 92, 133, 86, 99, 197, 134, 205 },
  { 52, 93, 133, 90, 100, 201, 137, 208 },
  { 54, 93, 133, 94, 100, 204, 139, 211 },
  { 57, 94, 134, 98, 101, 208, 142, 214 },
  { 59, 94, 134, 101, 102, 211, 144, 216 },
  { 62, 94, 135, 105, 103, 214, 146, 218 },
  { 64, 94, 135, 108, 103, 216, 148, 220 },
  { 66, 95, 135, 111, 104, 219, 151, 222 },
  { 68, 95, 135, 114, 105, 221, 153, 223 },
  { 71, 96, 136, 117, 106, 224, 155, 225 },
  { 73, 96, 136, 120, 106, 225, 157, 226 },
  { 76, 97, 136, 123, 107, 227, 159, 228 },
  { 78, 97, 136, 126, 108, 229, 160, 229 },
  { 80, 98, 137, 129, 109, 231, 162, 231 },
  { 82, 98, 137, 131, 109, 232, 164, 232 },
  { 84, 98, 138, 134, 110, 234, 166, 233 },
  { 86, 98, 138, 137, 111, 235, 168, 234 },
  { 89, 99, 138, 140, 112, 236, 170, 235 },
  { 91, 99, 138, 142, 112, 237, 171, 235 },
  { 93, 100, 139, 145, 113, 238, 173, 236 },
  { 95, 100, 139, 147, 114, 239, 174, 237 },
  { 97, 101, 140, 149, 115, 240, 176, 238 },
  { 99, 101, 140, 151, 115, 241, 177, 238 },
  { 101, 102, 140, 154, 116, 242, 179, 239 },
  { 103, 102, 140, 156, 117, 242, 180, 239 },
  { 105, 103, 141, 158, 118, 243, 182, 240 },
  { 107, 103, 141, 160, 118, 243, 183, 240 },
  { 109, 104, 141, 162, 119, 244, 185, 241 },
  { 111, 104, 141, 164, 119, 244, 186, 241 },
  { 113, 104, 142, 166, 120, 245, 187, 242 },
  { 114, 104, 142, 168, 121, 245, 188, 242 },
  { 116, 105, 143, 170, 122, 246, 190, 243 },
  { 118, 105, 143, 171, 122, 246, 191, 243 },
  { 120, 106, 143, 173, 123, 247, 192, 244 },
  { 121, 106, 143, 175, 124, 247, 193, 244 },
  { 123, 107, 144, 177, 125, 248, 195, 244 },
  { 125, 107, 144, 178, 125, 248, 196, 244 },
  { 127, 108, 145, 180, 126, 249, 197, 245 },
  { 128, 108, 145, 181, 127, 249, 198, 245 },
  { 130, 109, 145, 183, 128, 249, 199, 245 },
  { 132, 109, 145, 184, 128, 249, 200, 245 },
  { 134, 110, 146, 186, 129, 250, 201, 246 },
  { 135, 110, 146, 187, 130, 250, 202, 246 },
  { 137, 111, 147, 189, 131, 251, 203, 246 },
  { 138, 111, 147, 190, 131, 251, 204, 246 },
  { 140, 112, 147, 192, 132, 251, 205, 247 },
  { 141, 112, 147, 193, 132, 251, 206, 247 },
  { 143, 113, 148, 194, 133, 251, 207, 247 },
  { 144, 113, 148, 195, 134, 251, 207, 247 },
  { 146, 114, 149, 197, 135, 252, 208, 248 },
  { 147, 114, 149, 198, 135, 252, 209, 248 },
  { 149, 115, 149, 199, 136, 252, 210, 248 },
  { 150, 115, 149, 200, 137, 252, 210, 248 },
  { 152, 115, 150, 201, 138, 252, 211, 248 },
  { 153, 115, 150, 202, 138, 252, 212, 248 },
  { 155, 116, 151, 204, 139, 253, 213, 249 },
  { 156, 116, 151, 205, 139, 253, 213, 249 },
  { 158, 117, 151, 206, 140, 253, 214, 249 },
  { 159, 117, 151, 207, 141, 253, 215, 249 },
  { 161, 118, 152, 208, 142, 253, 216, 249 },
  { 162, 118, 152, 209, 142, 253, 216, 249 },
  { 163, 119, 153, 210, 143, 253, 217, 249 },
  { 164, 119, 153, 211, 143, 253, 217, 249 },
  { 166, 120, 153, 212, 144, 254, 218, 250 },
  { 167, 120, 153, 212, 145, 254, 219, 250 },
  { 168, 121, 154, 213, 146, 254, 220, 250 },
  { 169, 121, 154, 214, 146, 254, 220, 250 },
  { 171, 122, 155, 215, 147, 254, 221, 250 },
  { 172, 122, 155, 216, 147, 254, 221, 250 },
  { 173, 123, 155, 217, 148, 254, 222, 250 },
  { 174, 123, 155, 217, 149, 254, 222, 250 },
  { 176, 124, 156, 218, 150, 254, 223, 250 },
  { 177, 124, 156, 219, 150, 254, 223, 250 },
  { 178, 125, 157, 220, 151, 254, 224, 251 },
  { 179, 125, 157, 220, 151, 254, 224, 251 },
  { 180, 126, 157, 221, 152, 254, 225, 251 },
  { 181, 126, 157, 221, 152, 254, 225, 251 },
  { 183, 127, 158, 222, 153, 254, 226, 251 },
  { 184, 127, 158, 223, 154, 254, 226, 251 },
  { 185, 128, 159, 224, 155, 255, 227, 251 },
  { 186, 128, 159, 224, 155, 255, 227, 251 },
  { 187, 129, 160, 225, 156, 255, 228, 251 },
  { 188, 130, 160, 225, 156, 255, 228, 251 },
  { 189, 131, 160, 226, 157, 255, 228, 251 },
  { 190, 131, 160, 226, 158, 255, 228, 251 },
  { 191, 132, 161, 227, 159, 255, 229, 251 },
  { 192, 132, 161, 227, 159, 255, 229, 251 },
  { 193, 133, 162, 228, 160, 255, 230, 252 },
  { 194, 133, 162, 229, 160, 255, 230, 252 },
  { 195, 134, 163, 230, 161, 255, 231, 252 },
  { 196, 134, 163, 230, 161, 255, 231, 252 },
  { 197, 135, 163, 231, 162, 255, 231, 252 },
  { 198, 135, 163, 231, 162, 255, 231, 252 },
  { 199, 136, 164, 232, 163, 255, 232, 252 },
  { 200, 136, 164, 232, 164, 255, 232, 252 },
  { 201, 137, 165, 233, 165, 255, 233, 252 },
  { 201, 137, 165, 233, 165, 255, 233, 252 },
  { 202, 138, 166, 233, 166, 255, 233, 252 },
  { 203, 138, 166, 233, 166, 255, 233, 252 },
  { 204, 139, 166, 234, 167, 255, 234, 252 },
  { 205, 139, 166, 234, 167, 255, 234, 252 },
  { 206, 140, 167, 235, 168, 255, 235, 252 },
  { 206, 140, 167, 235, 168, 255, 235, 252 },
  { 207, 141, 168, 236, 169, 255, 235, 252 },
  { 208, 141, 168, 236, 170, 255, 235, 252 },
  { 209, 142, 169, 237, 171, 255, 236, 252 },
  { 209, 143, 169, 237, 171, 255, 236, 252 },
  { 210, 144, 169, 237, 172, 255, 236, 252 },
  { 211, 144, 169, 237, 172, 255, 236, 252 },
  { 212, 145, 170, 238, 173, 255, 237, 252 },
  { 213, 145, 170, 238, 173, 255, 237, 252 },
  { 214, 146, 171, 239, 174, 255, 237, 253 },
  { 214, 146, 171, 239, 174, 255, 237, 253 },
  { 215, 147, 172, 240, 175, 255, 238, 253 },
  { 215, 147, 172, 240, 175, 255, 238, 253 },
  { 216, 148, 173, 240, 176, 255, 238, 253 },
  { 217, 148, 173, 240, 176, 255, 238, 253 },
  { 218, 149, 173, 241, 177, 255, 239, 253 },
  { 218, 149, 173, 241, 178, 255, 239, 253 },
  { 219, 150, 174, 241, 179, 255, 239, 253 },
  { 219, 151, 174, 241, 179, 255, 239, 253 },
  { 220, 152, 175, 242, 180, 255, 240, 253 },
  { 221, 152, 175, 242, 180, 255, 240, 253 },
  { 222, 153, 176, 242, 181, 255, 240, 253 },
  { 222, 153, 176, 242, 181, 255, 240, 253 },
  { 223, 154, 177, 243, 182, 255, 240, 253 },
  { 223, 154, 177, 243, 182, 255, 240, 253 },
  { 224, 155, 178, 244, 183, 255, 241, 253 },
  { 224, 155, 178, 244, 183, 255, 241, 253 },
  { 225, 156, 178, 244, 184, 255, 241, 253 },
  { 225, 157, 178, 244, 184, 255, 241, 253 },
  { 226, 158, 179, 244, 185, 255, 242, 253 },
  { 227, 158, 179, 244, 185, 255, 242, 253 },
  { 228, 159, 180, 245, 186, 255, 242, 253 },
  { 228, 159, 180, 245, 186, 255, 242, 253 },
  { 229, 160, 181, 245, 187, 255, 242, 253 },
  { 229, 160, 181, 245, 187, 255, 242, 253 },
  { 230, 161, 182, 246, 188, 255, 243, 253 },
  { 230, 162, 182, 246, 188, 255, 243, 253 },
  { 231, 163, 183, 246, 189, 255, 243, 253 },
  { 231, 163, 183, 246, 189, 255, 243, 253 },
  { 232, 164, 184, 247, 190, 255, 243, 253 },
  { 232, 164, 184, 247, 190, 255, 243, 253 },
  { 233, 165, 185, 247, 191, 255, 244, 253 },
  { 233, 165, 185, 247, 191, 255, 244, 253 },
  { 234, 166, 185, 247, 192, 255, 244, 253 },
  { 234, 167, 185, 247, 192, 255, 244, 253 },
  { 235, 168, 186, 248, 193, 255, 244, 253 },
  { 235, 168, 186, 248, 193, 255, 244, 253 },
  { 236, 169, 187, 248, 194, 255, 244, 253 },
  { 236, 169, 187, 248, 194, 255, 244, 253 },
  { 236, 170, 188, 248, 195, 255, 245, 253 },
  { 236, 170, 188, 248, 195, 255, 245, 253 },
  { 237, 171, 189, 249, 196, 255, 245, 254 },
  { 237, 172, 189, 249, 196, 255, 245, 254 },
  { 238, 173, 190, 249, 197, 255, 245, 254 },
  { 238, 173, 190, 249, 197, 255, 245, 254 },
  { 239, 174, 191, 249, 198, 255, 245, 254 },
  { 239, 174, 191, 249, 198, 255, 245, 254 },
  { 240, 175, 192, 249, 199, 255, 246, 254 },
  { 240, 176, 192, 249, 199, 255, 246, 254 },
  { 240, 177, 193, 250, 200, 255, 246, 254 },
  { 240, 177, 193, 250, 200, 255, 246, 254 },
  { 241, 178, 194, 250, 201, 255, 246, 254 },
  { 241, 178, 194, 250, 201, 255, 246, 254 },
  { 242, 179, 195, 250, 202, 255, 246, 254 },
  { 242, 180, 195, 250, 202, 255, 246, 254 },
  { 242, 181, 196, 250, 203, 255, 247, 254 },
  { 242, 181, 196, 250, 203, 255, 247, 254 },
  { 243, 182, 197, 251, 204, 255, 247, 254 },
  { 243, 183, 197, 251, 204, 255, 247, 254 },
  { 244, 184, 198, 251, 205, 255, 247, 254 },
  { 244, 184, 198, 251, 205, 255, 247, 254 },
  { 244, 185, 199, 251, 206, 255, 247, 254 },
  { 244, 185, 199, 251, 206, 255, 247, 254 },
  { 245, 186, 200, 251, 207, 255, 247, 254 },
  { 245, 187, 200, 251, 207, 255, 247, 254 },
  { 246, 188, 201, 252, 207, 255, 248, 254 },
  { 246, 188, 201, 252, 207, 255, 248, 254 },
  { 246, 189, 202, 252, 208, 255, 248, 254 },
  { 246, 190, 202, 252, 208, 255, 248, 254 },
  { 247, 191, 203, 252, 209, 255, 248, 254 },
  { 247, 191, 203, 252, 209, 255, 248, 254 },
  { 247, 192, 204, 252, 210, 255, 248, 254 },
  { 247, 193, 204, 252, 210, 255, 248, 254 },
  { 248, 194, 205, 252, 211, 255, 248, 254 },
  { 248, 194, 205, 252, 211, 255, 248, 254 },
  { 248, 195, 206, 252, 212, 255, 249, 254 },
  { 248, 196, 206, 252, 212, 255, 249, 254 },
  { 249, 197, 207, 253, 213, 255, 249, 254 },
  { 249, 197, 207, 253, 213, 255, 249, 254 },
  { 249, 198, 208, 253, 214, 255, 249, 254 },
  { 249, 199, 209, 253, 214, 255, 249, 254 },
  { 250, 200, 210, 253, 215, 255, 249, 254 },
  { 250, 200, 210, 253, 215, 255, 249, 254 },
  { 250, 201, 211, 253, 215, 255, 249, 254 },
  { 250, 202, 211, 253, 215, 255, 249, 254 },
  { 250, 203, 212, 253, 216, 255, 249, 254 },
  { 250, 203, 212, 253, 216, 255, 249, 254 },
  { 251, 204, 213, 253, 217, 255, 250, 254 },
  { 251, 205, 213, 253, 217, 255, 250, 254 },
  { 251, 206, 214, 254, 218, 255, 250, 254 },
  { 251, 206, 215, 254, 218, 255, 250, 254 },
  { 252, 207, 216, 254, 219, 255, 250, 254 },
  { 252, 208, 216, 254, 219, 255, 250, 254 },
  { 252, 209, 217, 254, 220, 255, 250, 254 },
  { 252, 210, 217, 254, 220, 255, 250, 254 },
  { 252, 211, 218, 254, 221, 255, 250, 254 },
  { 252, 212, 218, 254, 221, 255, 250, 254 },
  { 253, 213, 219, 254, 222, 255, 250, 254 },
  { 253, 213, 220, 254, 222, 255, 250, 254 },
  { 253, 214, 221, 254, 223, 255, 250, 254 },
  { 253, 215, 221, 254, 223, 255, 250, 254 },
  { 253, 216, 222, 254, 224, 255, 251, 254 },
  { 253, 217, 223, 254, 224, 255, 251, 254 },
  { 253, 218, 224, 254, 225, 255, 251, 254 },
  { 253, 219, 224, 254, 225, 255, 251, 254 },
  { 254, 220, 225, 254, 225, 255, 251, 254 },
  { 254, 221, 226, 254, 225, 255, 251, 254 },
  { 254, 222, 227, 255, 226, 255, 251, 254 },
  { 254, 223, 227, 255, 226, 255, 251, 254 },
  { 254, 224, 228, 255, 227, 255, 251, 254 },
  { 254, 225, 229, 255, 227, 255, 251, 254 },
  { 254, 226, 230, 255, 228, 255, 251, 254 },
  { 254, 227, 230, 255, 229, 255, 251, 254 },
  { 255, 228, 231, 255, 230, 255, 251, 254 },
  { 255, 229, 232, 255, 230, 255, 251, 254 },
  { 255, 230, 233, 255, 231, 255, 252, 254 },
  { 255, 231, 234, 255, 231, 255, 252, 254 },
  { 255, 232, 235, 255, 232, 255, 252, 254 },
  { 255, 233, 236, 255, 232, 255, 252, 254 },
  { 255, 235, 237, 255, 233, 255, 252, 254 },
  { 255, 236, 238, 255, 234, 255, 252, 254 },
  { 255, 238, 240, 255, 235, 255, 252, 255 },
  { 255, 239, 241, 255, 235, 255, 252, 254 },
  { 255, 241, 243, 255, 236, 255, 252, 254 },
  { 255, 243, 245, 255, 237, 255, 252, 254 },
  { 255, 246, 247, 255, 239, 255, 253, 255 },
};
```

(255 rows. The comment in libvpx notes that every odd row can be
reconstructed from its even neighbors via averaging:
`vp9_pareto8_full[l][node] = (vp9_pareto8_full[l-1][node] +
vp9_pareto8_full[l+1][node]) >> 1`, `vp9_entropy.c:109-111`.)


#### 5.7.7 `default_coef_probs_4x4` --- default coefficient probabilities for 4x4 transforms

`vp9_entropy.c:370-533`. Shape: `[PLANE_TYPES=2][REF_TYPES=2][COEF_BANDS=6][BAND_COEFF_CONTEXTS(band)][UNCONSTRAINED_NODES=3]`.

```c
static const vp9_coeff_probs_model default_coef_probs_4x4[PLANE_TYPES] = {
  {     // Y plane
    {   // Intra
      { // Band 0
        { 195, 29, 183 },
        { 84, 49, 136 },
        { 8, 42, 71 } },
      { // Band 1
        { 31, 107, 169 },
        { 35, 99, 159 },
        { 17, 82, 140 },
        { 8, 66, 114 },
        { 2, 44, 76 },
        { 1, 19, 32 } },
      { // Band 2
        { 40, 132, 201 },
        { 29, 114, 187 },
        { 13, 91, 157 },
        { 7, 75, 127 },
        { 3, 58, 95 },
        { 1, 28, 47 } },
      { // Band 3
        { 69, 142, 221 },
        { 42, 122, 201 },
        { 15, 91, 159 },
        { 6, 67, 121 },
        { 1, 42, 77 },
        { 1, 17, 31 } },
      { // Band 4
        { 102, 148, 228 },
        { 67, 117, 204 },
        { 17, 82, 154 },
        { 6, 59, 114 },
        { 2, 39, 75 },
        { 1, 15, 29 } },
      { // Band 5
        { 156, 57, 233 },
        { 119, 57, 212 },
        { 58, 48, 163 },
        { 29, 40, 124 },
        { 12, 30, 81 },
        { 3, 12, 31 } } },
    {   // Inter
      { // Band 0
        { 191, 107, 226 },
        { 124, 117, 204 },
        { 25, 99, 155 } },
      { // Band 1
        { 29, 148, 210 },
        { 37, 126, 194 },
        { 8, 93, 157 },
        { 2, 68, 118 },
        { 1, 39, 69 },
        { 1, 17, 33 } },
      { // Band 2
        { 41, 151, 213 },
        { 27, 123, 193 },
        { 3, 82, 144 },
        { 1, 58, 105 },
        { 1, 32, 60 },
        { 1, 13, 26 } },
      { // Band 3
        { 59, 159, 220 },
        { 23, 126, 198 },
        { 4, 88, 151 },
        { 1, 66, 114 },
        { 1, 38, 71 },
        { 1, 18, 34 } },
      { // Band 4
        { 114, 136, 232 },
        { 51, 114, 207 },
        { 11, 83, 155 },
        { 3, 56, 105 },
        { 1, 33, 65 },
        { 1, 17, 34 } },
      { // Band 5
        { 149, 65, 234 },
        { 121, 57, 215 },
        { 61, 49, 166 },
        { 28, 36, 114 },
        { 12, 25, 76 },
        { 3, 16, 42 } } } },
  {     // UV plane
    {   // Intra
      { // Band 0
        { 214, 49, 220 },
        { 132, 63, 188 },
        { 42, 65, 137 } },
      { // Band 1
        { 85, 137, 221 },
        { 104, 131, 216 },
        { 49, 111, 192 },
        { 21, 87, 155 },
        { 2, 49, 87 },
        { 1, 16, 28 } },
      { // Band 2
        { 89, 163, 230 },
        { 90, 137, 220 },
        { 29, 100, 183 },
        { 10, 70, 135 },
        { 2, 42, 81 },
        { 1, 17, 33 } },
      { // Band 3
        { 108, 167, 237 },
        { 55, 133, 222 },
        { 15, 97, 179 },
        { 4, 72, 135 },
        { 1, 45, 85 },
        { 1, 19, 38 } },
      { // Band 4
        { 124, 146, 240 },
        { 66, 124, 224 },
        { 17, 88, 175 },
        { 4, 58, 122 },
        { 1, 36, 75 },
        { 1, 18, 37 } },
      { //  Band 5
        { 141, 79, 241 },
        { 126, 70, 227 },
        { 66, 58, 182 },
        { 30, 44, 136 },
        { 12, 34, 96 },
        { 2, 20, 47 } } },
    {   // Inter
      { // Band 0
        { 229, 99, 249 },
        { 143, 111, 235 },
        { 46, 109, 192 } },
      { // Band 1
        { 82, 158, 236 },
        { 94, 146, 224 },
        { 25, 117, 191 },
        { 9, 87, 149 },
        { 3, 56, 99 },
        { 1, 33, 57 } },
      { // Band 2
        { 83, 167, 237 },
        { 68, 145, 222 },
        { 10, 103, 177 },
        { 2, 72, 131 },
        { 1, 41, 79 },
        { 1, 20, 39 } },
      { // Band 3
        { 99, 167, 239 },
        { 47, 141, 224 },
        { 10, 104, 178 },
        { 2, 73, 133 },
        { 1, 44, 85 },
        { 1, 22, 47 } },
      { // Band 4
        { 127, 145, 243 },
        { 71, 129, 228 },
        { 17, 93, 177 },
        { 3, 61, 124 },
        { 1, 41, 84 },
        { 1, 21, 52 } },
      { // Band 5
        { 157, 78, 244 },
        { 140, 72, 231 },
        { 69, 58, 184 },
        { 31, 44, 137 },
        { 14, 38, 105 },
        { 8, 23, 61 } } } }
};
```

#### 5.7.8 `default_coef_probs_8x8`

`vp9_entropy.c:535-698`.

```c
static const vp9_coeff_probs_model default_coef_probs_8x8[PLANE_TYPES] = {
  {     // Y plane
    {   // Intra
      { // Band 0
        { 125, 34, 187 },
        { 52, 41, 133 },
        { 6, 31, 56 } },
      { // Band 1
        { 37, 109, 153 },
        { 51, 102, 147 },
        { 23, 87, 128 },
        { 8, 67, 101 },
        { 1, 41, 63 },
        { 1, 19, 29 } },
      { // Band 2
        { 31, 154, 185 },
        { 17, 127, 175 },
        { 6, 96, 145 },
        { 2, 73, 114 },
        { 1, 51, 82 },
        { 1, 28, 45 } },
      { // Band 3
        { 23, 163, 200 },
        { 10, 131, 185 },
        { 2, 93, 148 },
        { 1, 67, 111 },
        { 1, 41, 69 },
        { 1, 14, 24 } },
      { // Band 4
        { 29, 176, 217 },
        { 12, 145, 201 },
        { 3, 101, 156 },
        { 1, 69, 111 },
        { 1, 39, 63 },
        { 1, 14, 23 } },
      { // Band 5
        { 57, 192, 233 },
        { 25, 154, 215 },
        { 6, 109, 167 },
        { 3, 78, 118 },
        { 1, 48, 69 },
        { 1, 21, 29 } } },
    {   // Inter
      { // Band 0
        { 202, 105, 245 },
        { 108, 106, 216 },
        { 18, 90, 144 } },
      { // Band 1
        { 33, 172, 219 },
        { 64, 149, 206 },
        { 14, 117, 177 },
        { 5, 90, 141 },
        { 2, 61, 95 },
        { 1, 37, 57 } },
      { // Band 2
        { 33, 179, 220 },
        { 11, 140, 198 },
        { 1, 89, 148 },
        { 1, 60, 104 },
        { 1, 33, 57 },
        { 1, 12, 21 } },
      { // Band 3
        { 30, 181, 221 },
        { 8, 141, 198 },
        { 1, 87, 145 },
        { 1, 58, 100 },
        { 1, 31, 55 },
        { 1, 12, 20 } },
      { // Band 4
        { 32, 186, 224 },
        { 7, 142, 198 },
        { 1, 86, 143 },
        { 1, 58, 100 },
        { 1, 31, 55 },
        { 1, 12, 22 } },
      { // Band 5
        { 57, 192, 227 },
        { 20, 143, 204 },
        { 3, 96, 154 },
        { 1, 68, 112 },
        { 1, 42, 69 },
        { 1, 19, 32 } } } },
  {     // UV plane
    {   // Intra
      { // Band 0
        { 212, 35, 215 },
        { 113, 47, 169 },
        { 29, 48, 105 } },
      { // Band 1
        { 74, 129, 203 },
        { 106, 120, 203 },
        { 49, 107, 178 },
        { 19, 84, 144 },
        { 4, 50, 84 },
        { 1, 15, 25 } },
      { // Band 2
        { 71, 172, 217 },
        { 44, 141, 209 },
        { 15, 102, 173 },
        { 6, 76, 133 },
        { 2, 51, 89 },
        { 1, 24, 42 } },
      { // Band 3
        { 64, 185, 231 },
        { 31, 148, 216 },
        { 8, 103, 175 },
        { 3, 74, 131 },
        { 1, 46, 81 },
        { 1, 18, 30 } },
      { // Band 4
        { 65, 196, 235 },
        { 25, 157, 221 },
        { 5, 105, 174 },
        { 1, 67, 120 },
        { 1, 38, 69 },
        { 1, 15, 30 } },
      { // Band 5
        { 65, 204, 238 },
        { 30, 156, 224 },
        { 7, 107, 177 },
        { 2, 70, 124 },
        { 1, 42, 73 },
        { 1, 18, 34 } } },
    {   // Inter
      { // Band 0
        { 225, 86, 251 },
        { 144, 104, 235 },
        { 42, 99, 181 } },
      { // Band 1
        { 85, 175, 239 },
        { 112, 165, 229 },
        { 29, 136, 200 },
        { 12, 103, 162 },
        { 6, 77, 123 },
        { 2, 53, 84 } },
      { // Band 2
        { 75, 183, 239 },
        { 30, 155, 221 },
        { 3, 106, 171 },
        { 1, 74, 128 },
        { 1, 44, 76 },
        { 1, 17, 28 } },
      { // Band 3
        { 73, 185, 240 },
        { 27, 159, 222 },
        { 2, 107, 172 },
        { 1, 75, 127 },
        { 1, 42, 73 },
        { 1, 17, 29 } },
      { // Band 4
        { 62, 190, 238 },
        { 21, 159, 222 },
        { 2, 107, 172 },
        { 1, 72, 122 },
        { 1, 40, 71 },
        { 1, 18, 32 } },
      { // Band 5
        { 61, 199, 240 },
        { 27, 161, 226 },
        { 4, 113, 180 },
        { 1, 76, 129 },
        { 1, 46, 80 },
        { 1, 23, 41 } } } }
};
```

#### 5.7.9 `default_coef_probs_16x16`

`vp9_entropy.c:700-863`.

```c
static const vp9_coeff_probs_model default_coef_probs_16x16[PLANE_TYPES] = {
  {     // Y plane
    {   // Intra
      { // Band 0
        { 7, 27, 153 },
        { 5, 30, 95 },
        { 1, 16, 30 } },
      { // Band 1
        { 50, 75, 127 },
        { 57, 75, 124 },
        { 27, 67, 108 },
        { 10, 54, 86 },
        { 1, 33, 52 },
        { 1, 12, 18 } },
      { // Band 2
        { 43, 125, 151 },
        { 26, 108, 148 },
        { 7, 83, 122 },
        { 2, 59, 89 },
        { 1, 38, 60 },
        { 1, 17, 27 } },
      { // Band 3
        { 23, 144, 163 },
        { 13, 112, 154 },
        { 2, 75, 117 },
        { 1, 50, 81 },
        { 1, 31, 51 },
        { 1, 14, 23 } },
      { // Band 4
        { 18, 162, 185 },
        { 6, 123, 171 },
        { 1, 78, 125 },
        { 1, 51, 86 },
        { 1, 31, 54 },
        { 1, 14, 23 } },
      { // Band 5
        { 15, 199, 227 },
        { 3, 150, 204 },
        { 1, 91, 146 },
        { 1, 55, 95 },
        { 1, 30, 53 },
        { 1, 11, 20 } } },
    {   // Inter
      { // Band 0
        { 19, 55, 240 },
        { 19, 59, 196 },
        { 3, 52, 105 } },
      { // Band 1
        { 41, 166, 207 },
        { 104, 153, 199 },
        { 31, 123, 181 },
        { 14, 101, 152 },
        { 5, 72, 106 },
        { 1, 36, 52 } },
      { // Band 2
        { 35, 176, 211 },
        { 12, 131, 190 },
        { 2, 88, 144 },
        { 1, 60, 101 },
        { 1, 36, 60 },
        { 1, 16, 28 } },
      { // Band 3
        { 28, 183, 213 },
        { 8, 134, 191 },
        { 1, 86, 142 },
        { 1, 56, 96 },
        { 1, 30, 53 },
        { 1, 12, 20 } },
      { // Band 4
        { 20, 190, 215 },
        { 4, 135, 192 },
        { 1, 84, 139 },
        { 1, 53, 91 },
        { 1, 28, 49 },
        { 1, 11, 20 } },
      { // Band 5
        { 13, 196, 216 },
        { 2, 137, 192 },
        { 1, 86, 143 },
        { 1, 57, 99 },
        { 1, 32, 56 },
        { 1, 13, 24 } } } },
  {     // UV plane
    {   // Intra
      { // Band 0
        { 211, 29, 217 },
        { 96, 47, 156 },
        { 22, 43, 87 } },
      { // Band 1
        { 78, 120, 193 },
        { 111, 116, 186 },
        { 46, 102, 164 },
        { 15, 80, 128 },
        { 2, 49, 76 },
        { 1, 18, 28 } },
      { // Band 2
        { 71, 161, 203 },
        { 42, 132, 192 },
        { 10, 98, 150 },
        { 3, 69, 109 },
        { 1, 44, 70 },
        { 1, 18, 29 } },
      { // Band 3
        { 57, 186, 211 },
        { 30, 140, 196 },
        { 4, 93, 146 },
        { 1, 62, 102 },
        { 1, 38, 65 },
        { 1, 16, 27 } },
      { // Band 4
        { 47, 199, 217 },
        { 14, 145, 196 },
        { 1, 88, 142 },
        { 1, 57, 98 },
        { 1, 36, 62 },
        { 1, 15, 26 } },
      { // Band 5
        { 26, 219, 229 },
        { 5, 155, 207 },
        { 1, 94, 151 },
        { 1, 60, 104 },
        { 1, 36, 62 },
        { 1, 16, 28 } } },
    {   // Inter
      { // Band 0
        { 233, 29, 248 },
        { 146, 47, 220 },
        { 43, 52, 140 } },
      { // Band 1
        { 100, 163, 232 },
        { 179, 161, 222 },
        { 63, 142, 204 },
        { 37, 113, 174 },
        { 26, 89, 137 },
        { 18, 68, 97 } },
      { // Band 2
        { 85, 181, 230 },
        { 32, 146, 209 },
        { 7, 100, 164 },
        { 3, 71, 121 },
        { 1, 45, 77 },
        { 1, 18, 30 } },
      { // Band 3
        { 65, 187, 230 },
        { 20, 148, 207 },
        { 2, 97, 159 },
        { 1, 68, 116 },
        { 1, 40, 70 },
        { 1, 14, 29 } },
      { // Band 4
        { 40, 194, 227 },
        { 8, 147, 204 },
        { 1, 94, 155 },
        { 1, 65, 112 },
        { 1, 39, 66 },
        { 1, 14, 26 } },
      { // Band 5
        { 16, 208, 228 },
        { 3, 151, 207 },
        { 1, 98, 160 },
        { 1, 67, 117 },
        { 1, 41, 74 },
        { 1, 17, 31 } } } }
};
```

#### 5.7.10 `default_coef_probs_32x32`

`vp9_entropy.c:865-1028`.

```c
static const vp9_coeff_probs_model default_coef_probs_32x32[PLANE_TYPES] = {
  {     // Y plane
    {   // Intra
      { // Band 0
        { 17, 38, 140 },
        { 7, 34, 80 },
        { 1, 17, 29 } },
      { // Band 1
        { 37, 75, 128 },
        { 41, 76, 128 },
        { 26, 66, 116 },
        { 12, 52, 94 },
        { 2, 32, 55 },
        { 1, 10, 16 } },
      { // Band 2
        { 50, 127, 154 },
        { 37, 109, 152 },
        { 16, 82, 121 },
        { 5, 59, 85 },
        { 1, 35, 54 },
        { 1, 13, 20 } },
      { // Band 3
        { 40, 142, 167 },
        { 17, 110, 157 },
        { 2, 71, 112 },
        { 1, 44, 72 },
        { 1, 27, 45 },
        { 1, 11, 17 } },
      { // Band 4
        { 30, 175, 188 },
        { 9, 124, 169 },
        { 1, 74, 116 },
        { 1, 48, 78 },
        { 1, 30, 49 },
        { 1, 11, 18 } },
      { // Band 5
        { 10, 222, 223 },
        { 2, 150, 194 },
        { 1, 83, 128 },
        { 1, 48, 79 },
        { 1, 27, 45 },
        { 1, 11, 17 } } },
    {   // Inter
      { // Band 0
        { 36, 41, 235 },
        { 29, 36, 193 },
        { 10, 27, 111 } },
      { // Band 1
        { 85, 165, 222 },
        { 177, 162, 215 },
        { 110, 135, 195 },
        { 57, 113, 168 },
        { 23, 83, 120 },
        { 10, 49, 61 } },
      { // Band 2
        { 85, 190, 223 },
        { 36, 139, 200 },
        { 5, 90, 146 },
        { 1, 60, 103 },
        { 1, 38, 65 },
        { 1, 18, 30 } },
      { // Band 3
        { 72, 202, 223 },
        { 23, 141, 199 },
        { 2, 86, 140 },
        { 1, 56, 97 },
        { 1, 36, 61 },
        { 1, 16, 27 } },
      { // Band 4
        { 55, 218, 225 },
        { 13, 145, 200 },
        { 1, 86, 141 },
        { 1, 57, 99 },
        { 1, 35, 61 },
        { 1, 13, 22 } },
      { // Band 5
        { 15, 235, 212 },
        { 1, 132, 184 },
        { 1, 84, 139 },
        { 1, 57, 97 },
        { 1, 34, 56 },
        { 1, 14, 23 } } } },
  {     // UV plane
    {   // Intra
      { // Band 0
        { 181, 21, 201 },
        { 61, 37, 123 },
        { 10, 38, 71 } },
      { // Band 1
        { 47, 106, 172 },
        { 95, 104, 173 },
        { 42, 93, 159 },
        { 18, 77, 131 },
        { 4, 50, 81 },
        { 1, 17, 23 } },
      { // Band 2
        { 62, 147, 199 },
        { 44, 130, 189 },
        { 28, 102, 154 },
        { 18, 75, 115 },
        { 2, 44, 65 },
        { 1, 12, 19 } },
      { // Band 3
        { 55, 153, 210 },
        { 24, 130, 194 },
        { 3, 93, 146 },
        { 1, 61, 97 },
        { 1, 31, 50 },
        { 1, 10, 16 } },
      { // Band 4
        { 49, 186, 223 },
        { 17, 148, 204 },
        { 1, 96, 142 },
        { 1, 53, 83 },
        { 1, 26, 44 },
        { 1, 11, 17 } },
      { // Band 5
        { 13, 217, 212 },
        { 2, 136, 180 },
        { 1, 78, 124 },
        { 1, 50, 83 },
        { 1, 29, 49 },
        { 1, 14, 23 } } },
    {   // Inter
      { // Band 0
        { 197, 13, 247 },
        { 82, 17, 222 },
        { 25, 17, 162 } },
      { // Band 1
        { 126, 186, 247 },
        { 234, 191, 243 },
        { 176, 177, 234 },
        { 104, 158, 220 },
        { 66, 128, 186 },
        { 55, 90, 137 } },
      { // Band 2
        { 111, 197, 242 },
        { 46, 158, 219 },
        { 9, 104, 171 },
        { 2, 65, 125 },
        { 1, 44, 80 },
        { 1, 17, 91 } },
      { // Band 3
        { 104, 208, 245 },
        { 39, 168, 224 },
        { 3, 109, 162 },
        { 1, 79, 124 },
        { 1, 50, 102 },
        { 1, 43, 102 } },
      { // Band 4
        { 84, 220, 246 },
        { 31, 177, 231 },
        { 2, 115, 180 },
        { 1, 79, 134 },
        { 1, 55, 77 },
        { 1, 60, 79 } },
      { // Band 5
        { 43, 243, 240 },
        { 8, 180, 217 },
        { 1, 115, 166 },
        { 1, 84, 121 },
        { 1, 51, 67 },
        { 1, 16, 6 } } } }
};
```

#### 5.7.11 Coefficient-probability adaptation constants

`vp9_entropy.c:1048-1053`:
```c
#define COEF_COUNT_SAT 24
#define COEF_MAX_UPDATE_FACTOR 112
#define COEF_COUNT_SAT_KEY 24
#define COEF_MAX_UPDATE_FACTOR_KEY 112
#define COEF_COUNT_SAT_AFTER_KEY 24
#define COEF_MAX_UPDATE_FACTOR_AFTER_KEY 128
```

Used by `vp9_adapt_coef_probs` (`vp9_entropy.c:1084`) and
`adapt_coef_probs` (`vp9_entropy.c:1055`).


### 5.8 MV probability defaults and helper tables

#### 5.8.1 `default_nmv_context`

`vp9_entropymv.c:30-54`:
```c
static const nmv_context default_nmv_context = {
  { 32, 64, 96 },
  { {
        // Vertical component
        128,                                                   // sign
        { 224, 144, 192, 168, 192, 176, 192, 198, 198, 245 },  // class
        { 216 },                                               // class0
        { 136, 140, 148, 160, 176, 192, 224, 234, 234, 240 },  // bits
        { { 128, 128, 64 }, { 96, 112, 64 } },                 // class0_fp
        { 64, 96, 64 },                                        // fp
        160,                                                   // class0_hp bit
        128,                                                   // hp
    },
    {
        // Horizontal component
        128,                                                   // sign
        { 216, 128, 176, 160, 176, 176, 192, 198, 198, 208 },  // class
        { 208 },                                               // class0
        { 136, 140, 148, 160, 176, 192, 224, 234, 234, 240 },  // bits
        { { 128, 128, 64 }, { 96, 112, 64 } },                 // class0_fp
        { 64, 96, 64 },                                        // fp
        160,                                                   // class0_hp bit
        128,                                                   // hp
    } },
};
```

The top-level `{ 32, 64, 96 }` is `joints[MV_JOINTS - 1]`. Each component
then has: 1 sign prob, 10 class probs, 1 class0 prob, 10 offset-bits
probs, 2x3 class0_fp probs, 3 fp probs, 1 class0_hp prob, 1 hp prob.
Total per-component slots: 30.

#### 5.8.2 `log_in_base_2[1024]` --- magnitude-to-class lookup

`vp9_entropymv.c:56-97`. Used by `vp9_get_mv_class`
(`vp9_entropymv.c:103`): `class = log_in_base_2[z >> 3]` for `z`
in range. Verbatim, all 1024 entries (the table is a step function
with a few transitions at powers of 2):

```c
static const uint8_t log_in_base_2[] = {
  0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10
};
```

### 5.9 Sub-exponential prob-update remap tables

#### 5.9.1 Encoder `update_bits[255]`

`vp9_subexp.c:17-32`. Number of bits the subexp codeword takes for each
recentered delta:
```c
static const uint8_t update_bits[255] = {
  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,
  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  8,  8,  8,  8,  8,  8,
  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
  8,  8,  8,  8,  8,  8,  8,  8,  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
  11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
  11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
  11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
  11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
  11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
  11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
  11, 11, 11, 11, 11, 11, 11, 0,
};
#define MIN_DELP_BITS 5
```

#### 5.9.2 Encoder `map_table[MAX_PROB - 1]`

`vp9_subexp.c:46-66`. 254 entries; maps `(v, m)` pair into the subexp
codeword domain:
```c
static const uint8_t map_table[MAX_PROB - 1] = {
  // generated by:
  //   map_table[j] = split_index(j, MAX_PROB - 1, MODULUS_PARAM);
  20,  21,  22,  23,  24,  25,  0,   26,  27,  28,  29,  30,  31,  32,  33,
  34,  35,  36,  37,  1,   38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
  48,  49,  2,   50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
  3,   62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  4,   74,
  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  5,   86,  87,  88,
  89,  90,  91,  92,  93,  94,  95,  96,  97,  6,   98,  99,  100, 101, 102,
  103, 104, 105, 106, 107, 108, 109, 7,   110, 111, 112, 113, 114, 115, 116,
  117, 118, 119, 120, 121, 8,   122, 123, 124, 125, 126, 127, 128, 129, 130,
  131, 132, 133, 9,   134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
  145, 10,  146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 11,
  158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 12,  170, 171,
  172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 13,  182, 183, 184, 185,
  186, 187, 188, 189, 190, 191, 192, 193, 14,  194, 195, 196, 197, 198, 199,
  200, 201, 202, 203, 204, 205, 15,  206, 207, 208, 209, 210, 211, 212, 213,
  214, 215, 216, 217, 16,  218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
  228, 229, 17,  230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
  18,  242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 19,
};
```

`SUBEXP_PARAM = 4`, `MODULUS_PARAM = 13` (`vp9_entropy.h:117-118`).

#### 5.9.3 Decoder `inv_map_table[MAX_PROB]`

`vp9_dsubexp.c:30-49`. 255 entries; inverse of `map_table`:
```c
static uint8_t inv_map_table[MAX_PROB] = {
  7,   20,  33,  46,  59,  72,  85,  98,  111, 124, 137, 150, 163, 176, 189,
  202, 215, 228, 241, 254, 1,   2,   3,   4,   5,   6,   8,   9,   10,  11,
  12,  13,  14,  15,  16,  17,  18,  19,  21,  22,  23,  24,  25,  26,  27,
  28,  29,  30,  31,  32,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
  44,  45,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  60,
  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  73,  74,  75,  76,
  77,  78,  79,  80,  81,  82,  83,  84,  86,  87,  88,  89,  90,  91,  92,
  93,  94,  95,  96,  97,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108,
  109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125,
  126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141,
  142, 143, 144, 145, 146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157,
  158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
  174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190,
  191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206,
  207, 208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222,
  223, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
  239, 240, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 253
};
```


### 5.10 Scan orders, inverse scans, and neighbor tables

All tables in this section live in
`third_party/libvpx/vp9/common/vp9_scan.c`. The dispatch arrays at the
bottom of the file (`vp9_scan.c:691-725`) pull these together; the
selector is `get_scan` (`vp9_scan.h:42`).

Reminder of structure (`vp9_scan.h:26-30`):
```c
typedef struct ScanOrder {
  const int16_t *scan;
  const int16_t *iscan;
  const int16_t *neighbors;
} ScanOrder;
```

For each transform size we have a default scan (zigzag), a row scan (for
intra modes with horizontal energy concentration), and a column scan
(for intra modes with vertical energy concentration). 32x32 uses the
default scan for all four `TX_TYPE`s (see `vp9_scan_orders` dispatch).

`MAX_NEIGHBORS = 2` (`vp9_scan.h:24`).

#### 5.10.1 4x4 scan orders

##### `default_scan_4x4[16]`

```c
DECLARE_ALIGNED(16, static const int16_t, default_scan_4x4[16]) = {
  0, 4, 1, 5, 8, 2, 12, 9, 3, 6, 13, 10, 7, 14, 11, 15,
};
```

##### `col_scan_4x4[16]`

```c
DECLARE_ALIGNED(16, static const int16_t, col_scan_4x4[16]) = {
  0, 4, 8, 1, 12, 5, 9, 2, 13, 6, 10, 3, 7, 14, 11, 15,
};
```

##### `row_scan_4x4[16]`

```c
DECLARE_ALIGNED(16, static const int16_t, row_scan_4x4[16]) = {
  0, 1, 4, 2, 5, 3, 6, 8, 9, 7, 12, 10, 13, 11, 14, 15,
};
```

#### 5.10.2 8x8 scan orders

##### `default_scan_8x8[64]`

```c
DECLARE_ALIGNED(16, static const int16_t, default_scan_8x8[64]) = {
  0,  8,  1,  16, 9,  2,  17, 24, 10, 3,  18, 25, 32, 11, 4,  26,
  33, 19, 40, 12, 34, 27, 5,  41, 20, 48, 13, 35, 42, 28, 21, 6,
  49, 56, 36, 43, 29, 7,  14, 50, 57, 44, 22, 37, 15, 51, 58, 30,
  45, 23, 52, 59, 38, 31, 60, 53, 46, 39, 61, 54, 47, 62, 55, 63,
};
```

##### `col_scan_8x8[64]`

```c
DECLARE_ALIGNED(16, static const int16_t, col_scan_8x8[64]) = {
  0,  8,  16, 1,  24, 9,  32, 17, 2,  40, 25, 10, 33, 18, 48, 3,
  26, 41, 11, 56, 19, 34, 4,  49, 27, 42, 12, 35, 20, 57, 50, 28,
  5,  43, 13, 36, 58, 51, 21, 44, 6,  29, 59, 37, 14, 52, 22, 7,
  45, 60, 30, 15, 38, 53, 23, 46, 31, 61, 39, 54, 47, 62, 55, 63,
};
```

##### `row_scan_8x8[64]`

```c
DECLARE_ALIGNED(16, static const int16_t, row_scan_8x8[64]) = {
  0,  1,  2,  8,  9,  3,  16, 10, 4,  17, 11, 24, 5,  18, 25, 12,
  19, 26, 32, 6,  13, 20, 33, 27, 7,  34, 40, 21, 28, 41, 14, 35,
  48, 42, 29, 36, 49, 22, 43, 15, 56, 37, 50, 44, 30, 57, 23, 51,
  58, 45, 38, 52, 31, 59, 53, 46, 60, 39, 61, 47, 54, 55, 62, 63,
};
```

#### 5.10.3 16x16 scan orders

##### `default_scan_16x16[256]`

```c
DECLARE_ALIGNED(16, static const int16_t, default_scan_16x16[256]) = {
  0,   16,  1,   32,  17,  2,   48,  33,  18,  3,   64,  34,  49,  19,  65,
  80,  50,  4,   35,  66,  20,  81,  96,  51,  5,   36,  82,  97,  67,  112,
  21,  52,  98,  37,  83,  113, 6,   68,  128, 53,  22,  99,  114, 84,  7,
  129, 38,  69,  100, 115, 144, 130, 85,  54,  23,  8,   145, 39,  70,  116,
  101, 131, 160, 146, 55,  86,  24,  71,  132, 117, 161, 40,  9,   102, 147,
  176, 162, 87,  56,  25,  133, 118, 177, 148, 72,  103, 41,  163, 10,  192,
  178, 88,  57,  134, 149, 119, 26,  164, 73,  104, 193, 42,  179, 208, 11,
  135, 89,  165, 120, 150, 58,  194, 180, 27,  74,  209, 105, 151, 136, 43,
  90,  224, 166, 195, 181, 121, 210, 59,  12,  152, 106, 167, 196, 75,  137,
  225, 211, 240, 182, 122, 91,  28,  197, 13,  226, 168, 183, 153, 44,  212,
  138, 107, 241, 60,  29,  123, 198, 184, 227, 169, 242, 76,  213, 154, 45,
  92,  14,  199, 139, 61,  228, 214, 170, 185, 243, 108, 77,  155, 30,  15,
  200, 229, 124, 215, 244, 93,  46,  186, 171, 201, 109, 140, 230, 62,  216,
  245, 31,  125, 78,  156, 231, 47,  187, 202, 217, 94,  246, 141, 63,  232,
  172, 110, 247, 157, 79,  218, 203, 126, 233, 188, 248, 95,  173, 142, 219,
  111, 249, 234, 158, 127, 189, 204, 250, 235, 143, 174, 220, 205, 159, 251,
  190, 221, 175, 236, 237, 191, 206, 252, 222, 253, 207, 238, 223, 254, 239,
  255,
};
```

##### `col_scan_16x16[256]`

```c
DECLARE_ALIGNED(16, static const int16_t, col_scan_16x16[256]) = {
  0,   16,  32,  48,  1,   64,  17,  80,  33,  96,  49,  2,   65,  112, 18,
  81,  34,  128, 50,  97,  3,   66,  144, 19,  113, 35,  82,  160, 98,  51,
  129, 4,   67,  176, 20,  114, 145, 83,  36,  99,  130, 52,  192, 5,   161,
  68,  115, 21,  146, 84,  208, 177, 37,  131, 100, 53,  162, 224, 69,  6,
  116, 193, 147, 85,  22,  240, 132, 38,  178, 101, 163, 54,  209, 117, 70,
  7,   148, 194, 86,  179, 225, 23,  133, 39,  164, 8,   102, 210, 241, 55,
  195, 118, 149, 71,  180, 24,  87,  226, 134, 165, 211, 40,  103, 56,  72,
  150, 196, 242, 119, 9,   181, 227, 88,  166, 25,  135, 41,  104, 212, 57,
  151, 197, 120, 73,  243, 182, 136, 167, 213, 89,  10,  228, 105, 152, 198,
  26,  42,  121, 183, 244, 168, 58,  137, 229, 74,  214, 90,  153, 199, 184,
  11,  106, 245, 27,  122, 230, 169, 43,  215, 59,  200, 138, 185, 246, 75,
  12,  91,  154, 216, 231, 107, 28,  44,  201, 123, 170, 60,  247, 232, 76,
  139, 13,  92,  217, 186, 248, 155, 108, 29,  124, 45,  202, 233, 171, 61,
  14,  77,  140, 15,  249, 93,  30,  187, 156, 218, 46,  109, 125, 62,  172,
  78,  203, 31,  141, 234, 94,  47,  188, 63,  157, 110, 250, 219, 79,  126,
  204, 173, 142, 95,  189, 111, 235, 158, 220, 251, 127, 174, 143, 205, 236,
  159, 190, 221, 252, 175, 206, 237, 191, 253, 222, 238, 207, 254, 223, 239,
  255,
};
```

##### `row_scan_16x16[256]`

```c
DECLARE_ALIGNED(16, static const int16_t, row_scan_16x16[256]) = {
  0,   1,   2,   16,  3,   17,  4,   18,  32,  5,   33,  19,  6,   34,  48,
  20,  49,  7,   35,  21,  50,  64,  8,   36,  65,  22,  51,  37,  80,  9,
  66,  52,  23,  38,  81,  67,  10,  53,  24,  82,  68,  96,  39,  11,  54,
  83,  97,  69,  25,  98,  84,  40,  112, 55,  12,  70,  99,  113, 85,  26,
  41,  56,  114, 100, 13,  71,  128, 86,  27,  115, 101, 129, 42,  57,  72,
  116, 14,  87,  130, 102, 144, 73,  131, 117, 28,  58,  15,  88,  43,  145,
  103, 132, 146, 118, 74,  160, 89,  133, 104, 29,  59,  147, 119, 44,  161,
  148, 90,  105, 134, 162, 120, 176, 75,  135, 149, 30,  60,  163, 177, 45,
  121, 91,  106, 164, 178, 150, 192, 136, 165, 179, 31,  151, 193, 76,  122,
  61,  137, 194, 107, 152, 180, 208, 46,  166, 167, 195, 92,  181, 138, 209,
  123, 153, 224, 196, 77,  168, 210, 182, 240, 108, 197, 62,  154, 225, 183,
  169, 211, 47,  139, 93,  184, 226, 212, 241, 198, 170, 124, 155, 199, 78,
  213, 185, 109, 227, 200, 63,  228, 242, 140, 214, 171, 186, 156, 229, 243,
  125, 94,  201, 244, 215, 216, 230, 141, 187, 202, 79,  172, 110, 157, 245,
  217, 231, 95,  246, 232, 126, 203, 247, 233, 173, 218, 142, 111, 158, 188,
  248, 127, 234, 219, 249, 189, 204, 143, 174, 159, 250, 235, 205, 220, 175,
  190, 251, 221, 191, 206, 236, 207, 237, 252, 222, 253, 223, 238, 239, 254,
  255,
};
```

#### 5.10.4 32x32 default scan

##### `default_scan_32x32[1024]`

```c
DECLARE_ALIGNED(16, static const int16_t, default_scan_32x32[1024]) = {
  0,    32,   1,    64,  33,   2,    96,   65,   34,   128,  3,    97,   66,
  160,  129,  35,   98,  4,    67,   130,  161,  192,  36,   99,   224,  5,
  162,  193,  68,   131, 37,   100,  225,  194,  256,  163,  69,   132,  6,
  226,  257,  288,  195, 101,  164,  38,   258,  7,    227,  289,  133,  320,
  70,   196,  165,  290, 259,  228,  39,   321,  102,  352,  8,    197,  71,
  134,  322,  291,  260, 353,  384,  229,  166,  103,  40,   354,  323,  292,
  135,  385,  198,  261, 72,   9,    416,  167,  386,  355,  230,  324,  104,
  293,  41,   417,  199, 136,  262,  387,  448,  325,  356,  10,   73,   418,
  231,  168,  449,  294, 388,  105,  419,  263,  42,   200,  357,  450,  137,
  480,  74,   326,  232, 11,   389,  169,  295,  420,  106,  451,  481,  358,
  264,  327,  201,  43,  138,  512,  482,  390,  296,  233,  170,  421,  75,
  452,  359,  12,   513, 265,  483,  328,  107,  202,  514,  544,  422,  391,
  453,  139,  44,   234, 484,  297,  360,  171,  76,   515,  545,  266,  329,
  454,  13,   423,  203, 108,  546,  485,  576,  298,  235,  140,  361,  330,
  172,  547,  45,   455, 267,  577,  486,  77,   204,  362,  608,  14,   299,
  578,  109,  236,  487, 609,  331,  141,  579,  46,   15,   173,  610,  363,
  78,   205,  16,   110, 237,  611,  142,  47,   174,  79,   206,  17,   111,
  238,  48,   143,  80,  175,  112,  207,  49,   18,   239,  81,   113,  19,
  50,   82,   114,  51,  83,   115,  640,  516,  392,  268,  144,  20,   672,
  641,  548,  517,  424, 393,  300,  269,  176,  145,  52,   21,   704,  673,
  642,  580,  549,  518, 456,  425,  394,  332,  301,  270,  208,  177,  146,
  84,   53,   22,   736, 705,  674,  643,  612,  581,  550,  519,  488,  457,
  426,  395,  364,  333, 302,  271,  240,  209,  178,  147,  116,  85,   54,
  23,   737,  706,  675, 613,  582,  551,  489,  458,  427,  365,  334,  303,
  241,  210,  179,  117, 86,   55,   738,  707,  614,  583,  490,  459,  366,
  335,  242,  211,  118, 87,   739,  615,  491,  367,  243,  119,  768,  644,
  520,  396,  272,  148, 24,   800,  769,  676,  645,  552,  521,  428,  397,
  304,  273,  180,  149, 56,   25,   832,  801,  770,  708,  677,  646,  584,
  553,  522,  460,  429, 398,  336,  305,  274,  212,  181,  150,  88,   57,
  26,   864,  833,  802, 771,  740,  709,  678,  647,  616,  585,  554,  523,
  492,  461,  430,  399, 368,  337,  306,  275,  244,  213,  182,  151,  120,
  89,   58,   27,   865, 834,  803,  741,  710,  679,  617,  586,  555,  493,
  462,  431,  369,  338, 307,  245,  214,  183,  121,  90,   59,   866,  835,
  742,  711,  618,  587, 494,  463,  370,  339,  246,  215,  122,  91,   867,
  743,  619,  495,  371, 247,  123,  896,  772,  648,  524,  400,  276,  152,
  28,   928,  897,  804, 773,  680,  649,  556,  525,  432,  401,  308,  277,
  184,  153,  60,   29,  960,  929,  898,  836,  805,  774,  712,  681,  650,
  588,  557,  526,  464, 433,  402,  340,  309,  278,  216,  185,  154,  92,
  61,   30,   992,  961, 930,  899,  868,  837,  806,  775,  744,  713,  682,
  651,  620,  589,  558, 527,  496,  465,  434,  403,  372,  341,  310,  279,
  248,  217,  186,  155, 124,  93,   62,   31,   993,  962,  931,  869,  838,
  807,  745,  714,  683, 621,  590,  559,  497,  466,  435,  373,  342,  311,
  249,  218,  187,  125, 94,   63,   994,  963,  870,  839,  746,  715,  622,
  591,  498,  467,  374, 343,  250,  219,  126,  95,   995,  871,  747,  623,
  499,  375,  251,  127, 900,  776,  652,  528,  404,  280,  156,  932,  901,
  808,  777,  684,  653, 560,  529,  436,  405,  312,  281,  188,  157,  964,
  933,  902,  840,  809, 778,  716,  685,  654,  592,  561,  530,  468,  437,
  406,  344,  313,  282, 220,  189,  158,  996,  965,  934,  903,  872,  841,
  810,  779,  748,  717, 686,  655,  624,  593,  562,  531,  500,  469,  438,
  407,  376,  345,  314, 283,  252,  221,  190,  159,  997,  966,  935,  873,
  842,  811,  749,  718, 687,  625,  594,  563,  501,  470,  439,  377,  346,
  315,  253,  222,  191, 998,  967,  874,  843,  750,  719,  626,  595,  502,
  471,  378,  347,  254, 223,  999,  875,  751,  627,  503,  379,  255,  904,
  780,  656,  532,  408, 284,  936,  905,  812,  781,  688,  657,  564,  533,
  440,  409,  316,  285, 968,  937,  906,  844,  813,  782,  720,  689,  658,
  596,  565,  534,  472, 441,  410,  348,  317,  286,  1000, 969,  938,  907,
  876,  845,  814,  783, 752,  721,  690,  659,  628,  597,  566,  535,  504,
  473,  442,  411,  380, 349,  318,  287,  1001, 970,  939,  877,  846,  815,
  753,  722,  691,  629, 598,  567,  505,  474,  443,  381,  350,  319,  1002,
  971,  878,  847,  754, 723,  630,  599,  506,  475,  382,  351,  1003, 879,
  755,  631,  507,  383, 908,  784,  660,  536,  412,  940,  909,  816,  785,
  692,  661,  568,  537, 444,  413,  972,  941,  910,  848,  817,  786,  724,
  693,  662,  600,  569, 538,  476,  445,  414,  1004, 973,  942,  911,  880,
  849,  818,  787,  756, 725,  694,  663,  632,  601,  570,  539,  508,  477,
  446,  415,  1005, 974, 943,  881,  850,  819,  757,  726,  695,  633,  602,
  571,  509,  478,  447, 1006, 975,  882,  851,  758,  727,  634,  603,  510,
  479,  1007, 883,  759, 635,  511,  912,  788,  664,  540,  944,  913,  820,
  789,  696,  665,  572, 541,  976,  945,  914,  852,  821,  790,  728,  697,
  666,  604,  573,  542, 1008, 977,  946,  915,  884,  853,  822,  791,  760,
  729,  698,  667,  636, 605,  574,  543,  1009, 978,  947,  885,  854,  823,
  761,  730,  699,  637, 606,  575,  1010, 979,  886,  855,  762,  731,  638,
  607,  1011, 887,  763, 639,  916,  792,  668,  948,  917,  824,  793,  700,
  669,  980,  949,  918, 856,  825,  794,  732,  701,  670,  1012, 981,  950,
  919,  888,  857,  826, 795,  764,  733,  702,  671,  1013, 982,  951,  889,
  858,  827,  765,  734, 703,  1014, 983,  890,  859,  766,  735,  1015, 891,
  767,  920,  796,  952, 921,  828,  797,  984,  953,  922,  860,  829,  798,
  1016, 985,  954,  923, 892,  861,  830,  799,  1017, 986,  955,  893,  862,
  831,  1018, 987,  894, 863,  1019, 895,  924,  956,  925,  988,  957,  926,
  1020, 989,  958,  927, 1021, 990,  959,  1022, 991,  1023,
};
```

#### 5.10.5 Neighbor tables


##### `default_scan_4x4_neighbors`

```c
                default_scan_4x4_neighbors[17 * MAX_NEIGHBORS]) = {
  0, 0, 0, 0, 0,  0, 1, 4, 4, 4,  1,  1, 8,  8,  5,  8, 2,
  2, 2, 5, 9, 12, 6, 9, 3, 6, 10, 13, 7, 10, 11, 14, 0, 0,
};
```

##### `col_scan_4x4_neighbors`

```c
                col_scan_4x4_neighbors[17 * MAX_NEIGHBORS]) = {
  0, 0, 0, 0, 4, 4, 0, 0, 8, 8,  1,  1, 5, 5,  1,  1, 9,
  9, 2, 2, 6, 6, 2, 2, 3, 3, 10, 10, 7, 7, 11, 11, 0, 0,
};
```

##### `row_scan_4x4_neighbors`

```c
                row_scan_4x4_neighbors[17 * MAX_NEIGHBORS]) = {
  0, 0, 0, 0, 0, 0, 1, 1,  4,  4,  2,  2,  5,  5,  4,  4, 8,
  8, 6, 6, 8, 8, 9, 9, 12, 12, 10, 10, 13, 13, 14, 14, 0, 0,
};
```

##### `col_scan_8x8_neighbors`

```c
                col_scan_8x8_neighbors[65 * MAX_NEIGHBORS]) = {
  0,  0,  0,  0,  8,  8,  0,  0,  16, 16, 1,  1,  24, 24, 9,  9,  1,  1,  32,
  32, 17, 17, 2,  2,  25, 25, 10, 10, 40, 40, 2,  2,  18, 18, 33, 33, 3,  3,
  48, 48, 11, 11, 26, 26, 3,  3,  41, 41, 19, 19, 34, 34, 4,  4,  27, 27, 12,
  12, 49, 49, 42, 42, 20, 20, 4,  4,  35, 35, 5,  5,  28, 28, 50, 50, 43, 43,
  13, 13, 36, 36, 5,  5,  21, 21, 51, 51, 29, 29, 6,  6,  44, 44, 14, 14, 6,
  6,  37, 37, 52, 52, 22, 22, 7,  7,  30, 30, 45, 45, 15, 15, 38, 38, 23, 23,
  53, 53, 31, 31, 46, 46, 39, 39, 54, 54, 47, 47, 55, 55, 0,  0,
};
```

##### `row_scan_8x8_neighbors`

```c
                row_scan_8x8_neighbors[65 * MAX_NEIGHBORS]) = {
  0,  0,  0,  0,  1,  1,  0,  0,  8,  8,  2,  2,  8,  8,  9,  9,  3,  3,  16,
  16, 10, 10, 16, 16, 4,  4,  17, 17, 24, 24, 11, 11, 18, 18, 25, 25, 24, 24,
  5,  5,  12, 12, 19, 19, 32, 32, 26, 26, 6,  6,  33, 33, 32, 32, 20, 20, 27,
  27, 40, 40, 13, 13, 34, 34, 40, 40, 41, 41, 28, 28, 35, 35, 48, 48, 21, 21,
  42, 42, 14, 14, 48, 48, 36, 36, 49, 49, 43, 43, 29, 29, 56, 56, 22, 22, 50,
  50, 57, 57, 44, 44, 37, 37, 51, 51, 30, 30, 58, 58, 52, 52, 45, 45, 59, 59,
  38, 38, 60, 60, 46, 46, 53, 53, 54, 54, 61, 61, 62, 62, 0,  0,
};
```

##### `default_scan_8x8_neighbors`

```c
                default_scan_8x8_neighbors[65 * MAX_NEIGHBORS]) = {
  0,  0,  0,  0,  0,  0,  8,  8,  1,  8,  1,  1,  9,  16, 16, 16, 2,  9,  2,
  2,  10, 17, 17, 24, 24, 24, 3,  10, 3,  3,  18, 25, 25, 32, 11, 18, 32, 32,
  4,  11, 26, 33, 19, 26, 4,  4,  33, 40, 12, 19, 40, 40, 5,  12, 27, 34, 34,
  41, 20, 27, 13, 20, 5,  5,  41, 48, 48, 48, 28, 35, 35, 42, 21, 28, 6,  6,
  6,  13, 42, 49, 49, 56, 36, 43, 14, 21, 29, 36, 7,  14, 43, 50, 50, 57, 22,
  29, 37, 44, 15, 22, 44, 51, 51, 58, 30, 37, 23, 30, 52, 59, 45, 52, 38, 45,
  31, 38, 53, 60, 46, 53, 39, 46, 54, 61, 47, 54, 55, 62, 0,  0,
};
```

##### `col_scan_16x16_neighbors`

```c
                col_scan_16x16_neighbors[257 * MAX_NEIGHBORS]) = {
  0,   0,   0,   0,   16,  16,  32,  32,  0,   0,   48,  48,  1,   1,   64,
  64,  17,  17,  80,  80,  33,  33,  1,   1,   49,  49,  96,  96,  2,   2,
  65,  65,  18,  18,  112, 112, 34,  34,  81,  81,  2,   2,   50,  50,  128,
  128, 3,   3,   97,  97,  19,  19,  66,  66,  144, 144, 82,  82,  35,  35,
  113, 113, 3,   3,   51,  51,  160, 160, 4,   4,   98,  98,  129, 129, 67,
  67,  20,  20,  83,  83,  114, 114, 36,  36,  176, 176, 4,   4,   145, 145,
  52,  52,  99,  99,  5,   5,   130, 130, 68,  68,  192, 192, 161, 161, 21,
  21,  115, 115, 84,  84,  37,  37,  146, 146, 208, 208, 53,  53,  5,   5,
  100, 100, 177, 177, 131, 131, 69,  69,  6,   6,   224, 224, 116, 116, 22,
  22,  162, 162, 85,  85,  147, 147, 38,  38,  193, 193, 101, 101, 54,  54,
  6,   6,   132, 132, 178, 178, 70,  70,  163, 163, 209, 209, 7,   7,   117,
  117, 23,  23,  148, 148, 7,   7,   86,  86,  194, 194, 225, 225, 39,  39,
  179, 179, 102, 102, 133, 133, 55,  55,  164, 164, 8,   8,   71,  71,  210,
  210, 118, 118, 149, 149, 195, 195, 24,  24,  87,  87,  40,  40,  56,  56,
  134, 134, 180, 180, 226, 226, 103, 103, 8,   8,   165, 165, 211, 211, 72,
  72,  150, 150, 9,   9,   119, 119, 25,  25,  88,  88,  196, 196, 41,  41,
  135, 135, 181, 181, 104, 104, 57,  57,  227, 227, 166, 166, 120, 120, 151,
  151, 197, 197, 73,  73,  9,   9,   212, 212, 89,  89,  136, 136, 182, 182,
  10,  10,  26,  26,  105, 105, 167, 167, 228, 228, 152, 152, 42,  42,  121,
  121, 213, 213, 58,  58,  198, 198, 74,  74,  137, 137, 183, 183, 168, 168,
  10,  10,  90,  90,  229, 229, 11,  11,  106, 106, 214, 214, 153, 153, 27,
  27,  199, 199, 43,  43,  184, 184, 122, 122, 169, 169, 230, 230, 59,  59,
  11,  11,  75,  75,  138, 138, 200, 200, 215, 215, 91,  91,  12,  12,  28,
  28,  185, 185, 107, 107, 154, 154, 44,  44,  231, 231, 216, 216, 60,  60,
  123, 123, 12,  12,  76,  76,  201, 201, 170, 170, 232, 232, 139, 139, 92,
  92,  13,  13,  108, 108, 29,  29,  186, 186, 217, 217, 155, 155, 45,  45,
  13,  13,  61,  61,  124, 124, 14,  14,  233, 233, 77,  77,  14,  14,  171,
  171, 140, 140, 202, 202, 30,  30,  93,  93,  109, 109, 46,  46,  156, 156,
  62,  62,  187, 187, 15,  15,  125, 125, 218, 218, 78,  78,  31,  31,  172,
  172, 47,  47,  141, 141, 94,  94,  234, 234, 203, 203, 63,  63,  110, 110,
  188, 188, 157, 157, 126, 126, 79,  79,  173, 173, 95,  95,  219, 219, 142,
  142, 204, 204, 235, 235, 111, 111, 158, 158, 127, 127, 189, 189, 220, 220,
  143, 143, 174, 174, 205, 205, 236, 236, 159, 159, 190, 190, 221, 221, 175,
  175, 237, 237, 206, 206, 222, 222, 191, 191, 238, 238, 207, 207, 223, 223,
  239, 239, 0,   0,
};
```

##### `row_scan_16x16_neighbors`

```c
                row_scan_16x16_neighbors[257 * MAX_NEIGHBORS]) = {
  0,   0,   0,   0,   1,   1,   0,   0,   2,   2,   16,  16,  3,   3,   17,
  17,  16,  16,  4,   4,   32,  32,  18,  18,  5,   5,   33,  33,  32,  32,
  19,  19,  48,  48,  6,   6,   34,  34,  20,  20,  49,  49,  48,  48,  7,
  7,   35,  35,  64,  64,  21,  21,  50,  50,  36,  36,  64,  64,  8,   8,
  65,  65,  51,  51,  22,  22,  37,  37,  80,  80,  66,  66,  9,   9,   52,
  52,  23,  23,  81,  81,  67,  67,  80,  80,  38,  38,  10,  10,  53,  53,
  82,  82,  96,  96,  68,  68,  24,  24,  97,  97,  83,  83,  39,  39,  96,
  96,  54,  54,  11,  11,  69,  69,  98,  98,  112, 112, 84,  84,  25,  25,
  40,  40,  55,  55,  113, 113, 99,  99,  12,  12,  70,  70,  112, 112, 85,
  85,  26,  26,  114, 114, 100, 100, 128, 128, 41,  41,  56,  56,  71,  71,
  115, 115, 13,  13,  86,  86,  129, 129, 101, 101, 128, 128, 72,  72,  130,
  130, 116, 116, 27,  27,  57,  57,  14,  14,  87,  87,  42,  42,  144, 144,
  102, 102, 131, 131, 145, 145, 117, 117, 73,  73,  144, 144, 88,  88,  132,
  132, 103, 103, 28,  28,  58,  58,  146, 146, 118, 118, 43,  43,  160, 160,
  147, 147, 89,  89,  104, 104, 133, 133, 161, 161, 119, 119, 160, 160, 74,
  74,  134, 134, 148, 148, 29,  29,  59,  59,  162, 162, 176, 176, 44,  44,
  120, 120, 90,  90,  105, 105, 163, 163, 177, 177, 149, 149, 176, 176, 135,
  135, 164, 164, 178, 178, 30,  30,  150, 150, 192, 192, 75,  75,  121, 121,
  60,  60,  136, 136, 193, 193, 106, 106, 151, 151, 179, 179, 192, 192, 45,
  45,  165, 165, 166, 166, 194, 194, 91,  91,  180, 180, 137, 137, 208, 208,
  122, 122, 152, 152, 208, 208, 195, 195, 76,  76,  167, 167, 209, 209, 181,
  181, 224, 224, 107, 107, 196, 196, 61,  61,  153, 153, 224, 224, 182, 182,
  168, 168, 210, 210, 46,  46,  138, 138, 92,  92,  183, 183, 225, 225, 211,
  211, 240, 240, 197, 197, 169, 169, 123, 123, 154, 154, 198, 198, 77,  77,
  212, 212, 184, 184, 108, 108, 226, 226, 199, 199, 62,  62,  227, 227, 241,
  241, 139, 139, 213, 213, 170, 170, 185, 185, 155, 155, 228, 228, 242, 242,
  124, 124, 93,  93,  200, 200, 243, 243, 214, 214, 215, 215, 229, 229, 140,
  140, 186, 186, 201, 201, 78,  78,  171, 171, 109, 109, 156, 156, 244, 244,
  216, 216, 230, 230, 94,  94,  245, 245, 231, 231, 125, 125, 202, 202, 246,
  246, 232, 232, 172, 172, 217, 217, 141, 141, 110, 110, 157, 157, 187, 187,
  247, 247, 126, 126, 233, 233, 218, 218, 248, 248, 188, 188, 203, 203, 142,
  142, 173, 173, 158, 158, 249, 249, 234, 234, 204, 204, 219, 219, 174, 174,
  189, 189, 250, 250, 220, 220, 190, 190, 205, 205, 235, 235, 206, 206, 236,
  236, 251, 251, 221, 221, 252, 252, 222, 222, 237, 237, 238, 238, 253, 253,
  254, 254, 0,   0,
};
```

##### `default_scan_16x16_neighbors`

```c
                default_scan_16x16_neighbors[257 * MAX_NEIGHBORS]) = {
  0,   0,   0,   0,   0,   0,   16,  16,  1,   16,  1,   1,   32,  32,  17,
  32,  2,   17,  2,   2,   48,  48,  18,  33,  33,  48,  3,   18,  49,  64,
  64,  64,  34,  49,  3,   3,   19,  34,  50,  65,  4,   19,  65,  80,  80,
  80,  35,  50,  4,   4,   20,  35,  66,  81,  81,  96,  51,  66,  96,  96,
  5,   20,  36,  51,  82,  97,  21,  36,  67,  82,  97,  112, 5,   5,   52,
  67,  112, 112, 37,  52,  6,   21,  83,  98,  98,  113, 68,  83,  6,   6,
  113, 128, 22,  37,  53,  68,  84,  99,  99,  114, 128, 128, 114, 129, 69,
  84,  38,  53,  7,   22,  7,   7,   129, 144, 23,  38,  54,  69,  100, 115,
  85,  100, 115, 130, 144, 144, 130, 145, 39,  54,  70,  85,  8,   23,  55,
  70,  116, 131, 101, 116, 145, 160, 24,  39,  8,   8,   86,  101, 131, 146,
  160, 160, 146, 161, 71,  86,  40,  55,  9,   24,  117, 132, 102, 117, 161,
  176, 132, 147, 56,  71,  87,  102, 25,  40,  147, 162, 9,   9,   176, 176,
  162, 177, 72,  87,  41,  56,  118, 133, 133, 148, 103, 118, 10,  25,  148,
  163, 57,  72,  88,  103, 177, 192, 26,  41,  163, 178, 192, 192, 10,  10,
  119, 134, 73,  88,  149, 164, 104, 119, 134, 149, 42,  57,  178, 193, 164,
  179, 11,  26,  58,  73,  193, 208, 89,  104, 135, 150, 120, 135, 27,  42,
  74,  89,  208, 208, 150, 165, 179, 194, 165, 180, 105, 120, 194, 209, 43,
  58,  11,  11,  136, 151, 90,  105, 151, 166, 180, 195, 59,  74,  121, 136,
  209, 224, 195, 210, 224, 224, 166, 181, 106, 121, 75,  90,  12,  27,  181,
  196, 12,  12,  210, 225, 152, 167, 167, 182, 137, 152, 28,  43,  196, 211,
  122, 137, 91,  106, 225, 240, 44,  59,  13,  28,  107, 122, 182, 197, 168,
  183, 211, 226, 153, 168, 226, 241, 60,  75,  197, 212, 138, 153, 29,  44,
  76,  91,  13,  13,  183, 198, 123, 138, 45,  60,  212, 227, 198, 213, 154,
  169, 169, 184, 227, 242, 92,  107, 61,  76,  139, 154, 14,  29,  14,  14,
  184, 199, 213, 228, 108, 123, 199, 214, 228, 243, 77,  92,  30,  45,  170,
  185, 155, 170, 185, 200, 93,  108, 124, 139, 214, 229, 46,  61,  200, 215,
  229, 244, 15,  30,  109, 124, 62,  77,  140, 155, 215, 230, 31,  46,  171,
  186, 186, 201, 201, 216, 78,  93,  230, 245, 125, 140, 47,  62,  216, 231,
  156, 171, 94,  109, 231, 246, 141, 156, 63,  78,  202, 217, 187, 202, 110,
  125, 217, 232, 172, 187, 232, 247, 79,  94,  157, 172, 126, 141, 203, 218,
  95,  110, 233, 248, 218, 233, 142, 157, 111, 126, 173, 188, 188, 203, 234,
  249, 219, 234, 127, 142, 158, 173, 204, 219, 189, 204, 143, 158, 235, 250,
  174, 189, 205, 220, 159, 174, 220, 235, 221, 236, 175, 190, 190, 205, 236,
  251, 206, 221, 237, 252, 191, 206, 222, 237, 207, 222, 238, 253, 223, 238,
  239, 254, 0,   0,
};
```

##### `default_scan_32x32_neighbors`

```c
                default_scan_32x32_neighbors[1025 * MAX_NEIGHBORS]) = {
  0,   0,    0,   0,    0,   0,    32,  32,   1,   32,  1,   1,    64,  64,
  33,  64,   2,   33,   96,  96,   2,   2,    65,  96,  34,  65,   128, 128,
  97,  128,  3,   34,   66,  97,   3,   3,    35,  66,  98,  129,  129, 160,
  160, 160,  4,   35,   67,  98,   192, 192,  4,   4,   130, 161,  161, 192,
  36,  67,   99,  130,  5,   36,   68,  99,   193, 224, 162, 193,  224, 224,
  131, 162,  37,  68,   100, 131,  5,   5,    194, 225, 225, 256,  256, 256,
  163, 194,  69,  100,  132, 163,  6,   37,   226, 257, 6,   6,    195, 226,
  257, 288,  101, 132,  288, 288,  38,  69,   164, 195, 133, 164,  258, 289,
  227, 258,  196, 227,  7,   38,   289, 320,  70,  101, 320, 320,  7,   7,
  165, 196,  39,  70,   102, 133,  290, 321,  259, 290, 228, 259,  321, 352,
  352, 352,  197, 228,  134, 165,  71,  102,  8,   39,  322, 353,  291, 322,
  260, 291,  103, 134,  353, 384,  166, 197,  229, 260, 40,  71,   8,   8,
  384, 384,  135, 166,  354, 385,  323, 354,  198, 229, 292, 323,  72,  103,
  261, 292,  9,   40,   385, 416,  167, 198,  104, 135, 230, 261,  355, 386,
  416, 416,  293, 324,  324, 355,  9,   9,    41,  72,  386, 417,  199, 230,
  136, 167,  417, 448,  262, 293,  356, 387,  73,  104, 387, 418,  231, 262,
  10,  41,   168, 199,  325, 356,  418, 449,  105, 136, 448, 448,  42,  73,
  294, 325,  200, 231,  10,  10,   357, 388,  137, 168, 263, 294,  388, 419,
  74,  105,  419, 450,  449, 480,  326, 357,  232, 263, 295, 326,  169, 200,
  11,  42,   106, 137,  480, 480,  450, 481,  358, 389, 264, 295,  201, 232,
  138, 169,  389, 420,  43,  74,   420, 451,  327, 358, 11,  11,   481, 512,
  233, 264,  451, 482,  296, 327,  75,  106,  170, 201, 482, 513,  512, 512,
  390, 421,  359, 390,  421, 452,  107, 138,  12,  43,  202, 233,  452, 483,
  265, 296,  328, 359,  139, 170,  44,  75,   483, 514, 513, 544,  234, 265,
  297, 328,  422, 453,  12,  12,   391, 422,  171, 202, 76,  107,  514, 545,
  453, 484,  544, 544,  266, 297,  203, 234,  108, 139, 329, 360,  298, 329,
  140, 171,  515, 546,  13,  44,   423, 454,  235, 266, 545, 576,  454, 485,
  45,  76,   172, 203,  330, 361,  576, 576,  13,  13,  267, 298,  546, 577,
  77,  108,  204, 235,  455, 486,  577, 608,  299, 330, 109, 140,  547, 578,
  14,  45,   14,  14,   141, 172,  578, 609,  331, 362, 46,  77,   173, 204,
  15,  15,   78,  109,  205, 236,  579, 610,  110, 141, 15,  46,   142, 173,
  47,  78,   174, 205,  16,  16,   79,  110,  206, 237, 16,  47,   111, 142,
  48,  79,   143, 174,  80,  111,  175, 206,  17,  48,  17,  17,   207, 238,
  49,  80,   81,  112,  18,  18,   18,  49,   50,  81,  82,  113,  19,  50,
  51,  82,   83,  114,  608, 608,  484, 515,  360, 391, 236, 267,  112, 143,
  19,  19,   640, 640,  609, 640,  516, 547,  485, 516, 392, 423,  361, 392,
  268, 299,  237, 268,  144, 175,  113, 144,  20,  51,  20,  20,   672, 672,
  641, 672,  610, 641,  548, 579,  517, 548,  486, 517, 424, 455,  393, 424,
  362, 393,  300, 331,  269, 300,  238, 269,  176, 207, 145, 176,  114, 145,
  52,  83,   21,  52,   21,  21,   704, 704,  673, 704, 642, 673,  611, 642,
  580, 611,  549, 580,  518, 549,  487, 518,  456, 487, 425, 456,  394, 425,
  363, 394,  332, 363,  301, 332,  270, 301,  239, 270, 208, 239,  177, 208,
  146, 177,  115, 146,  84,  115,  53,  84,   22,  53,  22,  22,   705, 736,
  674, 705,  643, 674,  581, 612,  550, 581,  519, 550, 457, 488,  426, 457,
  395, 426,  333, 364,  302, 333,  271, 302,  209, 240, 178, 209,  147, 178,
  85,  116,  54,  85,   23,  54,   706, 737,  675, 706, 582, 613,  551, 582,
  458, 489,  427, 458,  334, 365,  303, 334,  210, 241, 179, 210,  86,  117,
  55,  86,   707, 738,  583, 614,  459, 490,  335, 366, 211, 242,  87,  118,
  736, 736,  612, 643,  488, 519,  364, 395,  240, 271, 116, 147,  23,  23,
  768, 768,  737, 768,  644, 675,  613, 644,  520, 551, 489, 520,  396, 427,
  365, 396,  272, 303,  241, 272,  148, 179,  117, 148, 24,  55,   24,  24,
  800, 800,  769, 800,  738, 769,  676, 707,  645, 676, 614, 645,  552, 583,
  521, 552,  490, 521,  428, 459,  397, 428,  366, 397, 304, 335,  273, 304,
  242, 273,  180, 211,  149, 180,  118, 149,  56,  87,  25,  56,   25,  25,
  832, 832,  801, 832,  770, 801,  739, 770,  708, 739, 677, 708,  646, 677,
  615, 646,  584, 615,  553, 584,  522, 553,  491, 522, 460, 491,  429, 460,
  398, 429,  367, 398,  336, 367,  305, 336,  274, 305, 243, 274,  212, 243,
  181, 212,  150, 181,  119, 150,  88,  119,  57,  88,  26,  57,   26,  26,
  833, 864,  802, 833,  771, 802,  709, 740,  678, 709, 647, 678,  585, 616,
  554, 585,  523, 554,  461, 492,  430, 461,  399, 430, 337, 368,  306, 337,
  275, 306,  213, 244,  182, 213,  151, 182,  89,  120, 58,  89,   27,  58,
  834, 865,  803, 834,  710, 741,  679, 710,  586, 617, 555, 586,  462, 493,
  431, 462,  338, 369,  307, 338,  214, 245,  183, 214, 90,  121,  59,  90,
  835, 866,  711, 742,  587, 618,  463, 494,  339, 370, 215, 246,  91,  122,
  864, 864,  740, 771,  616, 647,  492, 523,  368, 399, 244, 275,  120, 151,
  27,  27,   896, 896,  865, 896,  772, 803,  741, 772, 648, 679,  617, 648,
  524, 555,  493, 524,  400, 431,  369, 400,  276, 307, 245, 276,  152, 183,
  121, 152,  28,  59,   28,  28,   928, 928,  897, 928, 866, 897,  804, 835,
  773, 804,  742, 773,  680, 711,  649, 680,  618, 649, 556, 587,  525, 556,
  494, 525,  432, 463,  401, 432,  370, 401,  308, 339, 277, 308,  246, 277,
  184, 215,  153, 184,  122, 153,  60,  91,   29,  60,  29,  29,   960, 960,
  929, 960,  898, 929,  867, 898,  836, 867,  805, 836, 774, 805,  743, 774,
  712, 743,  681, 712,  650, 681,  619, 650,  588, 619, 557, 588,  526, 557,
  495, 526,  464, 495,  433, 464,  402, 433,  371, 402, 340, 371,  309, 340,
  278, 309,  247, 278,  216, 247,  185, 216,  154, 185, 123, 154,  92,  123,
  61,  92,   30,  61,   30,  30,   961, 992,  930, 961, 899, 930,  837, 868,
  806, 837,  775, 806,  713, 744,  682, 713,  651, 682, 589, 620,  558, 589,
  527, 558,  465, 496,  434, 465,  403, 434,  341, 372, 310, 341,  279, 310,
  217, 248,  186, 217,  155, 186,  93,  124,  62,  93,  31,  62,   962, 993,
  931, 962,  838, 869,  807, 838,  714, 745,  683, 714, 590, 621,  559, 590,
  466, 497,  435, 466,  342, 373,  311, 342,  218, 249, 187, 218,  94,  125,
  63,  94,   963, 994,  839, 870,  715, 746,  591, 622, 467, 498,  343, 374,
  219, 250,  95,  126,  868, 899,  744, 775,  620, 651, 496, 527,  372, 403,
  248, 279,  124, 155,  900, 931,  869, 900,  776, 807, 745, 776,  652, 683,
  621, 652,  528, 559,  497, 528,  404, 435,  373, 404, 280, 311,  249, 280,
  156, 187,  125, 156,  932, 963,  901, 932,  870, 901, 808, 839,  777, 808,
  746, 777,  684, 715,  653, 684,  622, 653,  560, 591, 529, 560,  498, 529,
  436, 467,  405, 436,  374, 405,  312, 343,  281, 312, 250, 281,  188, 219,
  157, 188,  126, 157,  964, 995,  933, 964,  902, 933, 871, 902,  840, 871,
  809, 840,  778, 809,  747, 778,  716, 747,  685, 716, 654, 685,  623, 654,
  592, 623,  561, 592,  530, 561,  499, 530,  468, 499, 437, 468,  406, 437,
  375, 406,  344, 375,  313, 344,  282, 313,  251, 282, 220, 251,  189, 220,
  158, 189,  127, 158,  965, 996,  934, 965,  903, 934, 841, 872,  810, 841,
  779, 810,  717, 748,  686, 717,  655, 686,  593, 624, 562, 593,  531, 562,
  469, 500,  438, 469,  407, 438,  345, 376,  314, 345, 283, 314,  221, 252,
  190, 221,  159, 190,  966, 997,  935, 966,  842, 873, 811, 842,  718, 749,
  687, 718,  594, 625,  563, 594,  470, 501,  439, 470, 346, 377,  315, 346,
  222, 253,  191, 222,  967, 998,  843, 874,  719, 750, 595, 626,  471, 502,
  347, 378,  223, 254,  872, 903,  748, 779,  624, 655, 500, 531,  376, 407,
  252, 283,  904, 935,  873, 904,  780, 811,  749, 780, 656, 687,  625, 656,
  532, 563,  501, 532,  408, 439,  377, 408,  284, 315, 253, 284,  936, 967,
  905, 936,  874, 905,  812, 843,  781, 812,  750, 781, 688, 719,  657, 688,
  626, 657,  564, 595,  533, 564,  502, 533,  440, 471, 409, 440,  378, 409,
  316, 347,  285, 316,  254, 285,  968, 999,  937, 968, 906, 937,  875, 906,
  844, 875,  813, 844,  782, 813,  751, 782,  720, 751, 689, 720,  658, 689,
  627, 658,  596, 627,  565, 596,  534, 565,  503, 534, 472, 503,  441, 472,
  410, 441,  379, 410,  348, 379,  317, 348,  286, 317, 255, 286,  969, 1000,
  938, 969,  907, 938,  845, 876,  814, 845,  783, 814, 721, 752,  690, 721,
  659, 690,  597, 628,  566, 597,  535, 566,  473, 504, 442, 473,  411, 442,
  349, 380,  318, 349,  287, 318,  970, 1001, 939, 970, 846, 877,  815, 846,
  722, 753,  691, 722,  598, 629,  567, 598,  474, 505, 443, 474,  350, 381,
  319, 350,  971, 1002, 847, 878,  723, 754,  599, 630, 475, 506,  351, 382,
  876, 907,  752, 783,  628, 659,  504, 535,  380, 411, 908, 939,  877, 908,
  784, 815,  753, 784,  660, 691,  629, 660,  536, 567, 505, 536,  412, 443,
  381, 412,  940, 971,  909, 940,  878, 909,  816, 847, 785, 816,  754, 785,
  692, 723,  661, 692,  630, 661,  568, 599,  537, 568, 506, 537,  444, 475,
  413, 444,  382, 413,  972, 1003, 941, 972,  910, 941, 879, 910,  848, 879,
  817, 848,  786, 817,  755, 786,  724, 755,  693, 724, 662, 693,  631, 662,
  600, 631,  569, 600,  538, 569,  507, 538,  476, 507, 445, 476,  414, 445,
  383, 414,  973, 1004, 942, 973,  911, 942,  849, 880, 818, 849,  787, 818,
  725, 756,  694, 725,  663, 694,  601, 632,  570, 601, 539, 570,  477, 508,
  446, 477,  415, 446,  974, 1005, 943, 974,  850, 881, 819, 850,  726, 757,
  695, 726,  602, 633,  571, 602,  478, 509,  447, 478, 975, 1006, 851, 882,
  727, 758,  603, 634,  479, 510,  880, 911,  756, 787, 632, 663,  508, 539,
  912, 943,  881, 912,  788, 819,  757, 788,  664, 695, 633, 664,  540, 571,
  509, 540,  944, 975,  913, 944,  882, 913,  820, 851, 789, 820,  758, 789,
  696, 727,  665, 696,  634, 665,  572, 603,  541, 572, 510, 541,  976, 1007,
  945, 976,  914, 945,  883, 914,  852, 883,  821, 852, 790, 821,  759, 790,
  728, 759,  697, 728,  666, 697,  635, 666,  604, 635, 573, 604,  542, 573,
  511, 542,  977, 1008, 946, 977,  915, 946,  853, 884, 822, 853,  791, 822,
  729, 760,  698, 729,  667, 698,  605, 636,  574, 605, 543, 574,  978, 1009,
  947, 978,  854, 885,  823, 854,  730, 761,  699, 730, 606, 637,  575, 606,
  979, 1010, 855, 886,  731, 762,  607, 638,  884, 915, 760, 791,  636, 667,
  916, 947,  885, 916,  792, 823,  761, 792,  668, 699, 637, 668,  948, 979,
  917, 948,  886, 917,  824, 855,  793, 824,  762, 793, 700, 731,  669, 700,
  638, 669,  980, 1011, 949, 980,  918, 949,  887, 918, 856, 887,  825, 856,
  794, 825,  763, 794,  732, 763,  701, 732,  670, 701, 639, 670,  981, 1012,
  950, 981,  919, 950,  857, 888,  826, 857,  795, 826, 733, 764,  702, 733,
  671, 702,  982, 1013, 951, 982,  858, 889,  827, 858, 734, 765,  703, 734,
  983, 1014, 859, 890,  735, 766,  888, 919,  764, 795, 920, 951,  889, 920,
  796, 827,  765, 796,  952, 983,  921, 952,  890, 921, 828, 859,  797, 828,
  766, 797,  984, 1015, 953, 984,  922, 953,  891, 922, 860, 891,  829, 860,
  798, 829,  767, 798,  985, 1016, 954, 985,  923, 954, 861, 892,  830, 861,
  799, 830,  986, 1017, 955, 986,  862, 893,  831, 862, 987, 1018, 863, 894,
  892, 923,  924, 955,  893, 924,  956, 987,  925, 956, 894, 925,  988, 1019,
  957, 988,  926, 957,  895, 926,  989, 1020, 958, 989, 927, 958,  990, 1021,
  959, 990,  991, 1022, 0,   0,
};
```

#### 5.10.6 Inverse-scan tables (iscan)

Note: iscan values are `EOB position` (i.e. scan index + 1), not
raw scan index. See `vp9_scan.c:514`.


##### `vp9_default_iscan_4x4`

```c
DECLARE_ALIGNED(16, static const int16_t, vp9_default_iscan_4x4[16]) = {
  1, 3, 6, 9, 2, 4, 10, 13, 5, 8, 12, 15, 7, 11, 14, 16,
};
```

##### `vp9_col_iscan_4x4`

```c
DECLARE_ALIGNED(16, static const int16_t, vp9_col_iscan_4x4[16]) = {
  1, 4, 8, 12, 2, 6, 10, 13, 3, 7, 11, 15, 5, 9, 14, 16,
};
```

##### `vp9_row_iscan_4x4`

```c
DECLARE_ALIGNED(16, static const int16_t, vp9_row_iscan_4x4[16]) = {
  1, 2, 4, 6, 3, 5, 7, 10, 8, 9, 12, 14, 11, 13, 15, 16,
};
```

##### `vp9_col_iscan_8x8`

```c
DECLARE_ALIGNED(16, static const int16_t, vp9_col_iscan_8x8[64]) = {
  1,  4,  9,  16, 23, 33, 41, 48, 2,  6,  12, 19, 27, 35, 45, 52,
  3,  8,  14, 21, 29, 39, 47, 55, 5,  11, 17, 25, 32, 42, 51, 57,
  7,  13, 22, 28, 36, 44, 53, 59, 10, 18, 26, 34, 40, 49, 56, 61,
  15, 24, 31, 38, 46, 54, 60, 63, 20, 30, 37, 43, 50, 58, 62, 64,
};
```

##### `vp9_row_iscan_8x8`

```c
DECLARE_ALIGNED(16, static const int16_t, vp9_row_iscan_8x8[64]) = {
  1,  2,  3,  6,  9,  13, 20, 25, 4,  5,  8,  11, 16, 21, 31, 40,
  7,  10, 14, 17, 22, 28, 38, 47, 12, 15, 18, 24, 29, 35, 45, 53,
  19, 23, 26, 32, 36, 42, 51, 58, 27, 30, 34, 39, 44, 50, 56, 60,
  33, 37, 43, 48, 52, 55, 61, 62, 41, 46, 49, 54, 57, 59, 63, 64,
};
```

##### `vp9_default_iscan_8x8`

```c
DECLARE_ALIGNED(16, static const int16_t, vp9_default_iscan_8x8[64]) = {
  1,  3,  6,  10, 15, 23, 32, 38, 2,  5,  9,  14, 20, 27, 39, 45,
  4,  7,  11, 18, 25, 31, 43, 50, 8,  12, 16, 22, 30, 37, 48, 54,
  13, 17, 21, 28, 35, 44, 53, 58, 19, 24, 29, 36, 42, 49, 57, 61,
  26, 33, 40, 46, 51, 56, 60, 63, 34, 41, 47, 52, 55, 59, 62, 64,
};
```

##### `vp9_col_iscan_16x16`

```c
DECLARE_ALIGNED(16, static const int16_t, vp9_col_iscan_16x16[256]) = {
  1,  5,  12,  21,  32,  44,  60,  76,  86,  110, 131, 151, 166, 182, 196, 199,
  2,  7,  15,  24,  35,  48,  65,  82,  96,  115, 136, 154, 172, 189, 202, 213,
  3,  9,  17,  26,  39,  53,  68,  84,  102, 117, 137, 158, 173, 191, 206, 217,
  4,  11, 19,  30,  42,  56,  72,  90,  104, 120, 142, 160, 177, 195, 209, 219,
  6,  13, 22,  33,  46,  59,  75,  94,  105, 124, 145, 165, 180, 197, 211, 224,
  8,  16, 27,  38,  50,  64,  79,  97,  113, 130, 147, 167, 183, 201, 216, 229,
  10, 20, 29,  40,  55,  70,  87,  103, 118, 133, 152, 171, 188, 207, 221, 231,
  14, 25, 36,  47,  61,  74,  92,  109, 123, 138, 155, 175, 190, 208, 225, 236,
  18, 31, 41,  54,  67,  83,  99,  116, 127, 143, 162, 181, 198, 214, 228, 238,
  23, 37, 49,  63,  77,  93,  106, 121, 134, 148, 168, 187, 204, 220, 233, 241,
  28, 45, 57,  71,  85,  100, 114, 128, 141, 157, 176, 194, 210, 227, 237, 245,
  34, 52, 69,  80,  95,  111, 126, 139, 150, 163, 185, 203, 218, 230, 242, 248,
  43, 62, 78,  91,  107, 122, 135, 149, 161, 174, 192, 212, 226, 239, 246, 252,
  51, 73, 88,  101, 119, 129, 146, 159, 169, 184, 205, 223, 234, 243, 250, 254,
  58, 81, 98,  112, 132, 144, 156, 170, 179, 193, 215, 232, 240, 247, 251, 255,
  66, 89, 108, 125, 140, 153, 164, 178, 186, 200, 222, 235, 244, 249, 253, 256,
};
```

##### `vp9_row_iscan_16x16`

```c
DECLARE_ALIGNED(16, static const int16_t, vp9_row_iscan_16x16[256]) = {
  1,   2,   3,   5,   7,   10,  13,  18,  23,  30,  37,  44,  55,  65,  77,
  87,  4,   6,   8,   12,  16,  20,  26,  33,  39,  49,  60,  69,  85,  100,
  116, 131, 9,   11,  14,  19,  24,  28,  34,  43,  52,  61,  73,  89,  104,
  120, 143, 168, 15,  17,  21,  27,  32,  38,  45,  54,  62,  74,  86,  101,
  117, 136, 162, 186, 22,  25,  31,  36,  41,  48,  56,  66,  75,  82,  95,
  113, 134, 155, 180, 206, 29,  35,  40,  46,  51,  59,  68,  78,  88,  97,
  107, 122, 147, 170, 197, 213, 42,  47,  50,  57,  64,  71,  80,  91,  99,
  108, 123, 139, 160, 183, 208, 223, 53,  58,  63,  70,  76,  84,  94,  103,
  111, 121, 135, 151, 177, 196, 216, 227, 67,  72,  79,  83,  92,  98,  109,
  114, 128, 137, 149, 169, 189, 203, 222, 233, 81,  90,  93,  102, 106, 115,
  126, 132, 140, 152, 163, 178, 193, 209, 224, 235, 96,  105, 110, 118, 124,
  129, 144, 145, 156, 166, 176, 191, 207, 220, 234, 240, 112, 119, 125, 130,
  141, 148, 158, 165, 171, 182, 192, 204, 225, 231, 241, 244, 127, 133, 138,
  146, 154, 161, 175, 179, 185, 198, 205, 217, 232, 238, 245, 247, 142, 150,
  157, 167, 173, 181, 190, 200, 201, 211, 221, 229, 239, 243, 250, 252, 153,
  164, 172, 184, 187, 194, 202, 212, 215, 219, 228, 237, 246, 248, 253, 254,
  159, 174, 188, 195, 199, 210, 214, 218, 226, 230, 236, 242, 249, 251, 255,
  256,
};
```

##### `vp9_default_iscan_16x16`

```c
DECLARE_ALIGNED(16, static const int16_t, vp9_default_iscan_16x16[256]) = {
  1,   3,   6,   10,  18,  25,  37,  45,  56,  73,  89,  105, 129, 144, 167,
  180, 2,   5,   9,   14,  21,  31,  41,  55,  67,  80,  97,  114, 142, 155,
  179, 197, 4,   8,   12,  19,  26,  34,  47,  58,  72,  87,  102, 120, 149,
  165, 187, 202, 7,   13,  17,  24,  32,  40,  54,  65,  79,  93,  111, 128,
  154, 170, 194, 209, 11,  15,  20,  29,  38,  48,  59,  68,  85,  99,  115,
  134, 162, 177, 199, 215, 16,  22,  27,  35,  44,  53,  66,  78,  92,  107,
  121, 141, 166, 186, 206, 222, 23,  28,  33,  42,  49,  61,  74,  86,  100,
  117, 131, 152, 176, 191, 212, 226, 30,  36,  43,  50,  60,  70,  82,  96,
  109, 126, 140, 156, 183, 198, 218, 230, 39,  46,  52,  62,  69,  81,  94,
  106, 119, 135, 151, 169, 192, 208, 224, 235, 51,  57,  64,  75,  84,  95,
  110, 118, 130, 148, 164, 178, 200, 214, 229, 239, 63,  71,  77,  88,  98,
  108, 123, 132, 146, 160, 173, 189, 211, 223, 236, 243, 76,  83,  91,  103,
  113, 125, 139, 147, 158, 174, 188, 203, 220, 231, 241, 246, 90,  101, 112,
  124, 133, 143, 157, 168, 181, 190, 204, 217, 232, 238, 247, 251, 104, 116,
  127, 137, 150, 163, 172, 184, 195, 205, 216, 225, 237, 242, 249, 253, 122,
  136, 145, 159, 171, 182, 193, 201, 210, 219, 228, 234, 244, 245, 252, 255,
  138, 153, 161, 175, 185, 196, 207, 213, 221, 227, 233, 240, 248, 250, 254,
  256,
};
```

##### `vp9_default_iscan_32x32`

```c
DECLARE_ALIGNED(16, static const int16_t, vp9_default_iscan_32x32[1024]) = {
  1,    3,    6,    11,   18,   26,   39,   48,   63,   84,   102,  122,  146,
  171,  194,  205,  211,  220,  230,  234,  246,  258,  276,  300,  343,  357,
  378,  406,  456,  472,  496,  528,  2,    5,    9,    16,   23,   31,   46,
  59,   75,   93,   113,  134,  159,  185,  204,  216,  223,  229,  235,  238,
  257,  275,  299,  318,  356,  377,  405,  427,  471,  495,  527,  552,  4,
  8,    13,   19,   29,   37,   53,   65,   83,   103,  119,  143,  165,  190,
  209,  218,  225,  232,  236,  239,  274,  298,  317,  330,  376,  404,  426,
  441,  494,  526,  551,  568,  7,    12,   17,   24,   32,   44,   61,   74,
  91,   110,  127,  151,  174,  197,  212,  221,  227,  233,  237,  240,  297,
  316,  329,  336,  403,  425,  440,  448,  525,  550,  567,  576,  10,   15,
  20,   30,   38,   51,   66,   79,   96,   117,  135,  158,  180,  202,  215,
  224,  245,  256,  273,  296,  342,  355,  375,  402,  455,  470,  493,  524,
  583,  597,  618,  646,  14,   21,   27,   36,   45,   55,   73,   86,   106,
  124,  141,  164,  183,  206,  217,  226,  255,  272,  295,  315,  354,  374,
  401,  424,  469,  492,  523,  549,  596,  617,  645,  667,  22,   28,   34,
  43,   54,   64,   81,   95,   114,  133,  152,  173,  191,  210,  219,  228,
  271,  294,  314,  328,  373,  400,  423,  439,  491,  522,  548,  566,  616,
  644,  666,  681,  25,   33,   40,   49,   58,   72,   89,   105,  121,  140,
  160,  179,  198,  213,  222,  231,  293,  313,  327,  335,  399,  422,  438,
  447,  521,  547,  565,  575,  643,  665,  680,  688,  35,   41,   47,   57,
  69,   82,   97,   112,  131,  148,  168,  187,  244,  254,  270,  292,  341,
  353,  372,  398,  454,  468,  490,  520,  582,  595,  615,  642,  694,  706,
  724,  748,  42,   50,   56,   68,   78,   92,   108,  125,  139,  162,  178,
  195,  253,  269,  291,  312,  352,  371,  397,  421,  467,  489,  519,  546,
  594,  614,  641,  664,  705,  723,  747,  766,  52,   60,   67,   77,   90,
  100,  120,  132,  150,  169,  182,  201,  268,  290,  311,  326,  370,  396,
  420,  437,  488,  518,  545,  564,  613,  640,  663,  679,  722,  746,  765,
  778,  62,   70,   76,   88,   101,  115,  130,  145,  163,  181,  192,  208,
  289,  310,  325,  334,  395,  419,  436,  446,  517,  544,  563,  574,  639,
  662,  678,  687,  745,  764,  777,  784,  71,   80,   87,   98,   109,  123,
  138,  156,  243,  252,  267,  288,  340,  351,  369,  394,  453,  466,  487,
  516,  581,  593,  612,  638,  693,  704,  721,  744,  789,  799,  814,  834,
  85,   94,   104,  111,  126,  142,  155,  172,  251,  266,  287,  309,  350,
  368,  393,  418,  465,  486,  515,  543,  592,  611,  637,  661,  703,  720,
  743,  763,  798,  813,  833,  849,  99,   107,  116,  128,  144,  157,  170,
  186,  265,  286,  308,  324,  367,  392,  417,  435,  485,  514,  542,  562,
  610,  636,  660,  677,  719,  742,  762,  776,  812,  832,  848,  859,  118,
  129,  137,  149,  161,  176,  189,  199,  285,  307,  323,  333,  391,  416,
  434,  445,  513,  541,  561,  573,  635,  659,  676,  686,  741,  761,  775,
  783,  831,  847,  858,  864,  136,  147,  153,  166,  242,  250,  264,  284,
  339,  349,  366,  390,  452,  464,  484,  512,  580,  591,  609,  634,  692,
  702,  718,  740,  788,  797,  811,  830,  868,  876,  888,  904,  154,  167,
  175,  184,  249,  263,  283,  306,  348,  365,  389,  415,  463,  483,  511,
  540,  590,  608,  633,  658,  701,  717,  739,  760,  796,  810,  829,  846,
  875,  887,  903,  916,  177,  188,  196,  203,  262,  282,  305,  322,  364,
  388,  414,  433,  482,  510,  539,  560,  607,  632,  657,  675,  716,  738,
  759,  774,  809,  828,  845,  857,  886,  902,  915,  924,  193,  200,  207,
  214,  281,  304,  321,  332,  387,  413,  432,  444,  509,  538,  559,  572,
  631,  656,  674,  685,  737,  758,  773,  782,  827,  844,  856,  863,  901,
  914,  923,  928,  241,  248,  261,  280,  338,  347,  363,  386,  451,  462,
  481,  508,  579,  589,  606,  630,  691,  700,  715,  736,  787,  795,  808,
  826,  867,  874,  885,  900,  931,  937,  946,  958,  247,  260,  279,  303,
  346,  362,  385,  412,  461,  480,  507,  537,  588,  605,  629,  655,  699,
  714,  735,  757,  794,  807,  825,  843,  873,  884,  899,  913,  936,  945,
  957,  967,  259,  278,  302,  320,  361,  384,  411,  431,  479,  506,  536,
  558,  604,  628,  654,  673,  713,  734,  756,  772,  806,  824,  842,  855,
  883,  898,  912,  922,  944,  956,  966,  973,  277,  301,  319,  331,  383,
  410,  430,  443,  505,  535,  557,  571,  627,  653,  672,  684,  733,  755,
  771,  781,  823,  841,  854,  862,  897,  911,  921,  927,  955,  965,  972,
  976,  337,  345,  360,  382,  450,  460,  478,  504,  578,  587,  603,  626,
  690,  698,  712,  732,  786,  793,  805,  822,  866,  872,  882,  896,  930,
  935,  943,  954,  978,  982,  988,  996,  344,  359,  381,  409,  459,  477,
  503,  534,  586,  602,  625,  652,  697,  711,  731,  754,  792,  804,  821,
  840,  871,  881,  895,  910,  934,  942,  953,  964,  981,  987,  995,  1002,
  358,  380,  408,  429,  476,  502,  533,  556,  601,  624,  651,  671,  710,
  730,  753,  770,  803,  820,  839,  853,  880,  894,  909,  920,  941,  952,
  963,  971,  986,  994,  1001, 1006, 379,  407,  428,  442,  501,  532,  555,
  570,  623,  650,  670,  683,  729,  752,  769,  780,  819,  838,  852,  861,
  893,  908,  919,  926,  951,  962,  970,  975,  993,  1000, 1005, 1008, 449,
  458,  475,  500,  577,  585,  600,  622,  689,  696,  709,  728,  785,  791,
  802,  818,  865,  870,  879,  892,  929,  933,  940,  950,  977,  980,  985,
  992,  1009, 1011, 1014, 1018, 457,  474,  499,  531,  584,  599,  621,  649,
  695,  708,  727,  751,  790,  801,  817,  837,  869,  878,  891,  907,  932,
  939,  949,  961,  979,  984,  991,  999,  1010, 1013, 1017, 1021, 473,  498,
  530,  554,  598,  620,  648,  669,  707,  726,  750,  768,  800,  816,  836,
  851,  877,  890,  906,  918,  938,  948,  960,  969,  983,  990,  998,  1004,
  1012, 1016, 1020, 1023, 497,  529,  553,  569,  619,  647,  668,  682,  725,
  749,  767,  779,  815,  835,  850,  860,  889,  905,  917,  925,  947,  959,
  968,  974,  989,  997,  1003, 1007, 1015, 1019, 1022, 1024,
};
```

#### 5.10.7 Scan-order dispatch tables

`vp9_scan.c:691-725`:

```c
const ScanOrder vp9_default_scan_orders[TX_SIZES] = {
  { default_scan_4x4, vp9_default_iscan_4x4, default_scan_4x4_neighbors },
  { default_scan_8x8, vp9_default_iscan_8x8, default_scan_8x8_neighbors },
  { default_scan_16x16, vp9_default_iscan_16x16, default_scan_16x16_neighbors },
  { default_scan_32x32, vp9_default_iscan_32x32, default_scan_32x32_neighbors },
};

const ScanOrder vp9_scan_orders[TX_SIZES][TX_TYPES] = {
  { // TX_4X4
    { default_scan_4x4, vp9_default_iscan_4x4, default_scan_4x4_neighbors },
    { row_scan_4x4, vp9_row_iscan_4x4, row_scan_4x4_neighbors },
    { col_scan_4x4, vp9_col_iscan_4x4, col_scan_4x4_neighbors },
    { default_scan_4x4, vp9_default_iscan_4x4, default_scan_4x4_neighbors } },
  { // TX_8X8
    { default_scan_8x8, vp9_default_iscan_8x8, default_scan_8x8_neighbors },
    { row_scan_8x8, vp9_row_iscan_8x8, row_scan_8x8_neighbors },
    { col_scan_8x8, vp9_col_iscan_8x8, col_scan_8x8_neighbors },
    { default_scan_8x8, vp9_default_iscan_8x8, default_scan_8x8_neighbors } },
  { // TX_16X16
    { default_scan_16x16, vp9_default_iscan_16x16,
      default_scan_16x16_neighbors },
    { row_scan_16x16, vp9_row_iscan_16x16, row_scan_16x16_neighbors },
    { col_scan_16x16, vp9_col_iscan_16x16, col_scan_16x16_neighbors },
    { default_scan_16x16, vp9_default_iscan_16x16,
      default_scan_16x16_neighbors } },
  { // TX_32X32
    { default_scan_32x32, vp9_default_iscan_32x32,
      default_scan_32x32_neighbors },
    { default_scan_32x32, vp9_default_iscan_32x32,
      default_scan_32x32_neighbors },
    { default_scan_32x32, vp9_default_iscan_32x32,
      default_scan_32x32_neighbors },
    { default_scan_32x32, vp9_default_iscan_32x32,
      default_scan_32x32_neighbors } }
};
```


### 5.11 Tokenizer cost tables (encoder-only)

These are precomputed bit-cost tables used by mode decision to estimate
the cost of writing a token *without* actually invoking the boolean
coder. They are not part of the bitstream proper, but they are
load-bearing for rate-distortion decisions in the encoder.
`VP9_PROB_COST_SHIFT = 9` (see `vp9_cost.h` --- A2 owns the cost
framework; cited here for table-grouping purposes only).

#### 5.11.1 `dct_cat_lt_10_value_tokens` --- token+extra for small magnitudes

`vp9_tokenize.c:26-46`. Index domain is `[-CAT6_MIN_VAL+1 ..
CAT6_MIN_VAL-1]` = `[-66 .. 66]`. The exported pointer
`vp9_dct_cat_lt_10_value_tokens` points to the middle of the table so
that signed values can be indexed directly.

```c
static const TOKENVALUE dct_cat_lt_10_value_tokens[] = {
  { 9, 63 }, { 9, 61 }, { 9, 59 }, { 9, 57 }, { 9, 55 }, { 9, 53 }, { 9, 51 },
  { 9, 49 }, { 9, 47 }, { 9, 45 }, { 9, 43 }, { 9, 41 }, { 9, 39 }, { 9, 37 },
  { 9, 35 }, { 9, 33 }, { 9, 31 }, { 9, 29 }, { 9, 27 }, { 9, 25 }, { 9, 23 },
  { 9, 21 }, { 9, 19 }, { 9, 17 }, { 9, 15 }, { 9, 13 }, { 9, 11 }, { 9, 9 },
  { 9, 7 },  { 9, 5 },  { 9, 3 },  { 9, 1 },  { 8, 31 }, { 8, 29 }, { 8, 27 },
  { 8, 25 }, { 8, 23 }, { 8, 21 }, { 8, 19 }, { 8, 17 }, { 8, 15 }, { 8, 13 },
  { 8, 11 }, { 8, 9 },  { 8, 7 },  { 8, 5 },  { 8, 3 },  { 8, 1 },  { 7, 15 },
  { 7, 13 }, { 7, 11 }, { 7, 9 },  { 7, 7 },  { 7, 5 },  { 7, 3 },  { 7, 1 },
  { 6, 7 },  { 6, 5 },  { 6, 3 },  { 6, 1 },  { 5, 3 },  { 5, 1 },  { 4, 1 },
  { 3, 1 },  { 2, 1 },  { 1, 1 },  { 0, 0 },  { 1, 0 },  { 2, 0 },  { 3, 0 },
  { 4, 0 },  { 5, 0 },  { 5, 2 },  { 6, 0 },  { 6, 2 },  { 6, 4 },  { 6, 6 },
  { 7, 0 },  { 7, 2 },  { 7, 4 },  { 7, 6 },  { 7, 8 },  { 7, 10 }, { 7, 12 },
  { 7, 14 }, { 8, 0 },  { 8, 2 },  { 8, 4 },  { 8, 6 },  { 8, 8 },  { 8, 10 },
  { 8, 12 }, { 8, 14 }, { 8, 16 }, { 8, 18 }, { 8, 20 }, { 8, 22 }, { 8, 24 },
  { 8, 26 }, { 8, 28 }, { 8, 30 }, { 9, 0 },  { 9, 2 },  { 9, 4 },  { 9, 6 },
  { 9, 8 },  { 9, 10 }, { 9, 12 }, { 9, 14 }, { 9, 16 }, { 9, 18 }, { 9, 20 },
  { 9, 22 }, { 9, 24 }, { 9, 26 }, { 9, 28 }, { 9, 30 }, { 9, 32 }, { 9, 34 },
  { 9, 36 }, { 9, 38 }, { 9, 40 }, { 9, 42 }, { 9, 44 }, { 9, 46 }, { 9, 48 },
  { 9, 50 }, { 9, 52 }, { 9, 54 }, { 9, 56 }, { 9, 58 }, { 9, 60 }, { 9, 62 }
};
```

#### 5.11.2 `dct_cat_lt_10_value_cost`

`vp9_tokenize.c:56-68`. Costs in the same index domain as above:
```c
static const int dct_cat_lt_10_value_cost[] = {
  3773, 3750, 3704, 3681, 3623, 3600, 3554, 3531, 3432, 3409, 3363, 3340, 3282,
  3259, 3213, 3190, 3136, 3113, 3067, 3044, 2986, 2963, 2917, 2894, 2795, 2772,
  2726, 2703, 2645, 2622, 2576, 2553, 3197, 3116, 3058, 2977, 2881, 2800, 2742,
  2661, 2615, 2534, 2476, 2395, 2299, 2218, 2160, 2079, 2566, 2427, 2334, 2195,
  2023, 1884, 1791, 1652, 1893, 1696, 1453, 1256, 1229, 864,  512,  512,  512,
  512,  0,    512,  512,  512,  512,  864,  1229, 1256, 1453, 1696, 1893, 1652,
  1791, 1884, 2023, 2195, 2334, 2427, 2566, 2079, 2160, 2218, 2299, 2395, 2476,
  2534, 2615, 2661, 2742, 2800, 2881, 2977, 3058, 3116, 3197, 2553, 2576, 2622,
  2645, 2703, 2726, 2772, 2795, 2894, 2917, 2963, 2986, 3044, 3067, 3113, 3136,
  3190, 3213, 3259, 3282, 3340, 3363, 3409, 3432, 3531, 3554, 3600, 3623, 3681,
  3704, 3750, 3773,
};
```

#### 5.11.3 Category cost tables

`vp9_tokenize.c:90-103`:
```c
static const int16_t zero_cost[] = { 0 };
static const int16_t sign_cost[1] = { 512 };
static const int16_t cat1_cost[1 << 1] = { 864, 1229 };
static const int16_t cat2_cost[1 << 2] = { 1256, 1453, 1696, 1893 };
static const int16_t cat3_cost[1 << 3] = { 1652, 1791, 1884, 2023,
                                           2195, 2334, 2427, 2566 };
static const int16_t cat4_cost[1 << 4] = { 2079, 2160, 2218, 2299, 2395, 2476,
                                           2534, 2615, 2661, 2742, 2800, 2881,
                                           2977, 3058, 3116, 3197 };
static const int16_t cat5_cost[1 << 5] = {
  2553, 2576, 2622, 2645, 2703, 2726, 2772, 2795, 2894, 2917, 2963,
  2986, 3044, 3067, 3113, 3136, 3190, 3213, 3259, 3282, 3340, 3363,
  3409, 3432, 3531, 3554, 3600, 3623, 3681, 3704, 3750, 3773
};
```

#### 5.11.4 `vp9_cat6_low_cost[256]`

`vp9_tokenize.c:104-125`:
```c
const int16_t vp9_cat6_low_cost[256] = {
  3378, 3390, 3401, 3413, 3435, 3447, 3458, 3470, 3517, 3529, 3540, 3552, 3574,
  3586, 3597, 3609, 3671, 3683, 3694, 3706, 3728, 3740, 3751, 3763, 3810, 3822,
  3833, 3845, 3867, 3879, 3890, 3902, 3973, 3985, 3996, 4008, 4030, 4042, 4053,
  4065, 4112, 4124, 4135, 4147, 4169, 4181, 4192, 4204, 4266, 4278, 4289, 4301,
  4323, 4335, 4346, 4358, 4405, 4417, 4428, 4440, 4462, 4474, 4485, 4497, 4253,
  4265, 4276, 4288, 4310, 4322, 4333, 4345, 4392, 4404, 4415, 4427, 4449, 4461,
  4472, 4484, 4546, 4558, 4569, 4581, 4603, 4615, 4626, 4638, 4685, 4697, 4708,
  4720, 4742, 4754, 4765, 4777, 4848, 4860, 4871, 4883, 4905, 4917, 4928, 4940,
  4987, 4999, 5010, 5022, 5044, 5056, 5067, 5079, 5141, 5153, 5164, 5176, 5198,
  5210, 5221, 5233, 5280, 5292, 5303, 5315, 5337, 5349, 5360, 5372, 4988, 5000,
  5011, 5023, 5045, 5057, 5068, 5080, 5127, 5139, 5150, 5162, 5184, 5196, 5207,
  5219, 5281, 5293, 5304, 5316, 5338, 5350, 5361, 5373, 5420, 5432, 5443, 5455,
  5477, 5489, 5500, 5512, 5583, 5595, 5606, 5618, 5640, 5652, 5663, 5675, 5722,
  5734, 5745, 5757, 5779, 5791, 5802, 5814, 5876, 5888, 5899, 5911, 5933, 5945,
  5956, 5968, 6015, 6027, 6038, 6050, 6072, 6084, 6095, 6107, 5863, 5875, 5886,
  5898, 5920, 5932, 5943, 5955, 6002, 6014, 6025, 6037, 6059, 6071, 6082, 6094,
  6156, 6168, 6179, 6191, 6213, 6225, 6236, 6248, 6295, 6307, 6318, 6330, 6352,
  6364, 6375, 6387, 6458, 6470, 6481, 6493, 6515, 6527, 6538, 6550, 6597, 6609,
  6620, 6632, 6654, 6666, 6677, 6689, 6751, 6763, 6774, 6786, 6808, 6820, 6831,
  6843, 6890, 6902, 6913, 6925, 6947, 6959, 6970, 6982
};
```

#### 5.11.5 `vp9_cat6_high_cost[64]` (8-bit)

`vp9_tokenize.c:126-133`:
```c
const uint16_t vp9_cat6_high_cost[64] = {
  88,    2251,  2727,  4890,  3148,  5311,  5787,  7950,  3666,  5829,  6305,
  8468,  6726,  8889,  9365,  11528, 3666,  5829,  6305,  8468,  6726,  8889,
  9365,  11528, 7244,  9407,  9883,  12046, 10304, 12467, 12943, 15106, 3666,
  5829,  6305,  8468,  6726,  8889,  9365,  11528, 7244,  9407,  9883,  12046,
  10304, 12467, 12943, 15106, 7244,  9407,  9883,  12046, 10304, 12467, 12943,
  15106, 10822, 12985, 13461, 15624, 13882, 16045, 16521, 18684
};
```

The 10-bit (`vp9_cat6_high10_high_cost[256]`) and 12-bit
(`vp9_cat6_high12_high_cost[1024]`) tables also live in
`vp9_tokenize.c` (lines 136-161 and 162-257 respectively) under
`CONFIG_VP9_HIGHBITDEPTH`. They are large; cited by line range rather
than transcribed here. The Q1 high-bit-depth decision (Stage 2) will
determine whether the Rust port needs them.

---

## 6. RTCD / SIMD surfaces

The entropy coder and bitstream paths are **entirely scalar** in libvpx.
There are no `add_proto` entries in `vpx_dsp/vpx_dsp_rtcd_defs.pl` for
`bitwriter`, `bitreader`, or `prob` (confirmed by Grep of the RTCD perl
file; zero matches). The boolean coder is implemented as an
`INLINE`-able function in C headers (`vpx_dsp/bitwriter.h:46`,
`vpx_dsp/bitreader.h:77`); the compiler is expected to inline it at each
call site (especially inside `pack_mb_tokens` and the various
mode-writing functions where it is called in tight inner loops).

There are *some* alignment annotations on tables that hint at SIMD
expectations:
- `DECLARE_ALIGNED(16, ...)` is applied to scan tables, neighbor tables,
  iscan tables, and the `vpx_norm`, `vp9_pareto8_full`, and `vp9_catN_prob`
  arrays. These are consumed by SIMD-accelerated quantization / tokenize
  helpers in `vp9_dsp` and `vp9_quantize_*.asm`, but the consumers live
  outside A4's scope (transforms/quant are owned by Stage 3 LLDs).

What this means for the Rust port:

| Surface                       | SIMD in libvpx? | Rust port complexity |
|-------------------------------|-----------------|----------------------|
| `vpx_write` / `vpx_read`      | No              | Scalar; ensure compiler inlines |
| `vpx_norm` lookup             | No              | const array          |
| Tree write / tree read        | No              | Scalar               |
| `pack_mb_tokens` loop         | No              | Scalar; perf-sensitive due to per-coef work |
| Carry-propagation walk        | No              | Scalar, branchy      |
| Subexp encode/decode          | No              | Scalar               |
| `vp9_pareto8_full` lookup     | No              | const 2D array       |
| Scan/iscan/neighbor tables    | No (alignment only) | const arrays    |
| MV encode                     | No              | Scalar               |

For the Rust port (PLAN.md D5, Stage 3) this section requires only the
scalar reference. The `core::arch` intrinsics design (Q2 in Stage 2)
will affect quant / transforms but *not* the boolean coder. No `unsafe`
beyond what the bare-metal numeric ops imply.

One micro-optimization worth flagging: the encoder's
carry-propagation while loop (`bitwriter.h:91-94`) walks backwards through
0xff bytes. In production libvpx-cpu-used-7 traces the bytes preceding
the cursor are 0xff only at low rates; the loop is short. Don't try to
SIMD-vectorize it.

---

## 7. Open questions for LLD

These are decisions the Stage 2 LLD writer must resolve. Most are
deliberate forks where libvpx made a choice that is not obviously right
for a Rust port. Cross-cite to A1 and A6 where threading is involved.

### Q4.1 --- Byte-parity threshold for the boolean coder

PLAN.md D8: BD-rate parity is the gate; byte parity is a debug tool. The
boolean coder is the most byte-parity-sensitive subsystem in the whole
codec; *all* downstream divergence ultimately surfaces here. Question:
does the LLD spec a hard "byte-identical to libvpx for the same
`(probs, bits)` input" unit-test gate on the coder itself, or only on
the higher-level entry points (`vp9_pack_bitstream` output)? Lean: hard
gate on the coder, because if the coder is wrong, nothing higher up can
be trusted.

### Q4.2 --- Inlining strategy

`vpx_write` (~70 lines in C) is *the* hot function in real-time encode.
libvpx makes it `static INLINE` in a header. The Rust equivalent:
`#[inline]` on a pub-crate fn? `#[inline(always)]`? A macro? In real-time
mode the call count is in the hundreds of millions per second of video,
so this matters. Recommend a Criterion micro-bench in Stage 3 to settle
the question; do *not* let the LLD pre-commit to a representation.

### Q4.3 --- Reader buffer abstraction

`BD_VALUE` is `size_t` in libvpx, so the reader's internal accumulator
is 32-bit on 32-bit platforms and 64-bit on 64-bit platforms. This
changes the refill cadence (`vpx_reader_fill`, `bitreader.c:38`)
non-trivially. PLAN.md D7 says "mirror integer widths exactly"; do we
mirror by hardcoding 64-bit (since x86_64 / aarch64 are the only targets
that meet the performance budget), or by genuinely abstracting over
`usize`-sized values? Recommend hardcoding 64-bit and documenting in the
divergence ledger that 32-bit decode is not supported in v1.

### Q4.4 --- `vpx_writer` carry-propagation correctness under panic

The carry-propagation loop assumes `buffer` is uniquely owned for the
lifetime of the writer. In Rust with `&mut [u8]` this is fine, but if a
panic mid-coder leaves the buffer in a half-flushed state, the bytes are
*not* a valid VP9 substream. Question: do we offer
`vpx_writer::stop` as a `Result`-returning op (forwarding overflow
errors) and document that a panic mid-write yields an undefined buffer?
Or do we make the writer poison-on-panic?

### Q4.5 --- 32-bit vs 16-bit `EXTRABIT`

`EXTRABIT` is `int16_t` for 8-bit and `int32_t` for high-bit-depth.
If we commit to 10/12-bit in v1 (Q1), `EXTRABIT` must be `i32`; if we
don't, it can be `i16` and save 50% on `TOKENEXTRA` size (cache pressure
in the encoder is real). Defer to Q1 resolution.

### Q4.6 --- Per-tile writer buffers

`encode_tiles_mt` (`vp9_bitstream.c:991`) allocates a private buffer per
tile worker; worker 0 writes in place. The Rust equivalent needs to
either (a) hand each thread a `Vec<u8>` and concatenate after sync, or
(b) hand each thread a unique non-overlapping `&mut [u8]` slice into a
pre-allocated arena. Option (b) is closer to libvpx and avoids a memcpy
in the merge step, but requires careful slice arithmetic. A6 owns the
overall threading design; this is the bitstream-layer touchpoint.

### Q4.7 --- The 0x80000000 carry check on 64-bit `lowvalue`

`bitwriter.h:88` checks `(lowvalue << (offset - 1)) & 0x80000000`. On
64-bit builds, `lowvalue` is `unsigned int` (32-bit), so this is fine.
But in Rust we'd naturally use `u32` for lowvalue --- does
`u32::wrapping_shl(offset - 1) & 0x80000000` give the same result? Yes,
provided `offset - 1` is in `[0, 31]` (which it is, by precondition of
the surrounding code). Recommend an `assert!(offset >= 1 && offset <=
8)` at entry to the byte-emit branch, and an explicit `wrapping_shl`.

### Q4.8 --- Sub-exponential update on every probability

libvpx emits `DIFF_UPDATE_PROB`-coded updates for *every* slot in
`FRAME_CONTEXT` that is potentially adaptive (per-tx-size coefficient
probs, partition, intra mode, inter mode, single ref, comp ref, skip,
intra-inter, comp-inter, switchable interp, tx-size). For a real-time
encoder at cpu-used 7/8, many of these updates save fewer bits than they
cost; libvpx's `prob_diff_update_savings_search` (`vp9_subexp.c:117`)
gates this. The LLD should not change *what* gets updated, but Stage 4a
may want to short-circuit the per-update savings search for cpu-used
7/8 by skipping updates entirely for low-count contexts. Flag for A2.

### Q4.9 --- Compressed-header size limit

`vp9_bitstream.c:1441-1443`:
```c
if (compressed_hdr_size > UINT16_MAX) {
  vpx_internal_error(&cm->error, VPX_CODEC_ERROR,
                     "compressed_hdr_size > 16 bits");
}
```
This is a 16-bit field. In the worst case, the compressed header could
in theory exceed 64 KiB; libvpx errors out hard. The Rust port must
preserve this check exactly (returning an error to the caller, not
panicking).

### Q4.10 --- Decoder marker-bit on init

`vpx_reader_init` (`bitreader.c:34`) returns `vpx_read_bit(r) != 0` as
the "did we fail" code, where the marker bit must be 0. A caller that
ignores the return value silently desyncs. In Rust this should be a
`Result<vpx_reader, BitstreamError>`. Defer the error-type design to the
LLD (Stage 2's error-model spec).

### Q4.11 --- Probability `0` is forbidden but the model nodes are unbounded

`vp9_pareto8_full[p - 1]` (`vp9_bitstream.c:178`) indexes the table at
position `p - 1`, requiring `p >= 1`. Every default coef prob in
`default_coef_probs_*` has `[2]` (the pivot) >= 1, but adaptation runs
through `merge_probs` and could in theory drift the pivot to 0. Is there
a guard? `get_prob` clamps to `[1, 255]` via the bit hack
`p | ((255 - p) >> 23) | (p == 0)` (`vpx_dsp/prob.h:53`). So no, the
pivot cannot reach 0 from `get_binary_prob`. The LLD should *assert*
this invariant at the boundary and document it.

### Q4.12 --- Adapter for `vp9_norm` on 32-bit targets

`vpx_norm` is u8[256], 256 bytes, aligned to 16. Trivial. The only
question is whether we want the array static or a `const` (compile-time
embedded). Recommend `const` to maximize LLVM's ability to constant-fold
the boolean coder's normalization step.

### Q4.13 --- Test-vector strategy

For the boolean coder, the simplest oracle is "run the libvpx coder and
ours over the same stream of `(bit, prob)` pairs and assert
byte-identical output". The stream can come from instrumenting libvpx
itself. The LLD should specify:
1. A capture format for `(bit, prob)` pairs (one row per call;
   pack `bit:1, prob:8` and disambiguate the marker bit, the EOB-byte,
   and the stop-encode tail).
2. A replay harness that drives our Rust coder with the captured stream.
3. A second harness for the *reader* that consumes the libvpx-produced
   bitstream and asserts every bit it reads matches the captured
   ground truth.

This is the highest-leverage test we can write, because every higher-up
divergence ultimately bottoms out in the coder.

### Q4.14 --- Where to put the cost tables

`vp9_dct_cat_lt_10_value_cost` (and friends, §5.11) are precomputed but
the underlying probability model is fixed. Do we put them in
`rvp9-core` (shared with decoder, even though decoder doesn't need
them) or in `rvp9-encoder` (where they actually live)? Recommend
`rvp9-encoder`: the decoder does not compute costs and the data flow is
asymmetric.

### Q4.15 --- Sub-exponential update strategy under cpu-used 7/8

libvpx still runs the full `update_coef_probs_common` traversal at
cpu-used 7 (per `set_rt_speed_features`, A6 will document). For our
target performance budget, profiling may reveal that this is too
expensive at 1080p60 even with `ONE_LOOP_REDUCED`. Question: do we have
a flag to suppress *all* coef prob updates and rely on the default
table? Yes --- `vpx_write_bit(w, 0)` for each tx_size suppresses that
size's update. Decision point: should we expose a "no adaptive coef
updates" flag in Stage 4a? Flag for A2 (mode decision speed features) /
A5 (rate control).

---

## Appendix A. File:line cross-reference index

For convenience when reviewing this doc against libvpx:

| Topic                                  | File:line |
|----------------------------------------|-----------|
| `vpx_writer` struct                    | `vpx_dsp/bitwriter.h:28-40` |
| `vpx_write` inline                     | `vpx_dsp/bitwriter.h:46-116` |
| `vpx_start_encode`                     | `vpx_dsp/bitwriter.c:20-31` |
| `vpx_stop_encode`                      | `vpx_dsp/bitwriter.c:33-55` |
| `vpx_reader` struct                    | `vpx_dsp/bitreader.h:40-50` |
| `vpx_read` inline                      | `vpx_dsp/bitreader.h:77-136` |
| `vpx_read_tree`                        | `vpx_dsp/bitreader.h:150-157` |
| `vpx_reader_init`                      | `vpx_dsp/bitreader.c:21-36` |
| `vpx_reader_fill`                      | `vpx_dsp/bitreader.c:38-91` |
| `vpx_write_bit_buffer` struct          | `vpx_dsp/bitwriter_buffer.h:20-31` |
| `vpx_wb_write_bit`                     | `vpx_dsp/bitwriter_buffer.c:35-51` |
| `vpx_rb_read_bit`                      | `vpx_dsp/bitreader_buffer.c:17-29` |
| `vpx_prob` typedef + `MAX_PROB`        | `vpx_dsp/prob.h:25-46` |
| `get_prob` quantizer                   | `vpx_dsp/prob.h:48-56` |
| `weighted_prob` / `merge_probs`        | `vpx_dsp/prob.h:65-75` |
| `mode_mv_merge_probs`                  | `vpx_dsp/prob.h:84-95` |
| `vpx_tree_merge_probs`                 | `vpx_dsp/prob.c:44-47` |
| `vpx_norm[256]`                        | `vpx_dsp/prob.c:13-24` |
| `vp9_pack_bitstream`                   | `vp9_bitstream.c:1396-1451` |
| `write_uncompressed_header`            | `vp9_bitstream.c:1243-1314` |
| `write_compressed_header`              | `vp9_bitstream.c:1316-1394` |
| `encode_tiles`                         | `vp9_bitstream.c:1092-1145` |
| `encode_tiles_mt`                      | `vp9_bitstream.c:991-1090` |
| `encode_tile_worker`                   | `vp9_bitstream.c:942-952` |
| `write_modes`                          | `vp9_bitstream.c:490-516` |
| `write_modes_sb`                       | `vp9_bitstream.c:427-488` |
| `write_modes_b`                        | `vp9_bitstream.c:377-403` |
| `write_partition`                      | `vp9_bitstream.c:405-425` |
| `write_mb_modes_kf` (intra-only)       | `vp9_bitstream.c:343-375` |
| `pack_inter_mode_mvs`                  | `vp9_bitstream.c:240-341` |
| `pack_mb_tokens`                       | `vp9_bitstream.c:132-195` |
| `write_segment_id`                     | `vp9_bitstream.c:197-201` |
| `write_ref_frames`                     | `vp9_bitstream.c:203-238` |
| `write_selected_tx_size`               | `vp9_bitstream.c:89-102` |
| `update_coef_probs`                    | `vp9_bitstream.c:683-699` |
| `update_coef_probs_common`             | `vp9_bitstream.c:545-681` |
| `update_skip_probs`                    | `vp9_bitstream.c:115-121` |
| `encode_loopfilter`                    | `vp9_bitstream.c:701-739` |
| `encode_quantization`                  | `vp9_bitstream.c:751-757` |
| `encode_segmentation`                  | `vp9_bitstream.c:759-816` |
| `encode_txfm_probs`                    | `vp9_bitstream.c:818-852` |
| `write_tile_info`                      | `vp9_bitstream.c:886-900` |
| `intra_mode_encodings` static          | `vp9_bitstream.c:42-45` |
| `partition_encodings` static           | `vp9_bitstream.c:50-52` |
| `inter_mode_encodings` static          | `vp9_bitstream.c:53-55` |
| `switchable_interp_encodings`          | `vp9_bitstream.c:46-49` |
| `vp9_coef_tree`                        | `vp9_tokenize.c:75-88` |
| `vp9_coef_encodings`                   | `vp9_tokenize.c:306-309` |
| `vp9_coef_con_tree`                    | `vp9_entropy.c:20-29` |
| `vp9_pareto8_full[255][8]`             | `vp9_entropy.c:112-368` |
| `default_coef_probs_4x4`               | `vp9_entropy.c:370-533` |
| `default_coef_probs_8x8`               | `vp9_entropy.c:535-698` |
| `default_coef_probs_16x16`             | `vp9_entropy.c:700-863` |
| `default_coef_probs_32x32`             | `vp9_entropy.c:865-1028` |
| `vp9_default_coef_probs` (init)        | `vp9_entropy.c:1041-1046` |
| `vp9_adapt_coef_probs`                 | `vp9_entropy.c:1084-1100` |
| `vp9_pt_energy_class`                  | `vp9_entropy.c:95-96` |
| `vp9_coefband_trans_8x8plus`           | `vp9_entropy.c:46-88` |
| `vp9_coefband_trans_4x4`               | `vp9_entropy.c:91-93` |
| `vp9_cat1_prob` .. `vp9_cat6_prob`     | `vp9_entropy.c:32-38` |
| `vp9_cat6_prob_high12`                 | `vp9_entropy.c:40-42` |
| `vp9_kf_y_mode_prob`                   | `vp9_entropymode.c:16-147` |
| `vp9_kf_uv_mode_prob`                  | `vp9_entropymode.c:149-160` |
| `default_if_y_probs`                   | `vp9_entropymode.c:162-167` |
| `default_if_uv_probs`                  | `vp9_entropymode.c:169-180` |
| `vp9_kf_partition_probs`               | `vp9_entropymode.c:182-207` |
| `default_partition_probs`              | `vp9_entropymode.c:209-231` |
| `default_inter_mode_probs`             | `vp9_entropymode.c:233-242` |
| `vp9_intra_mode_tree`                  | `vp9_entropymode.c:245-255` |
| `vp9_inter_mode_tree`                  | `vp9_entropymode.c:257-260` |
| `vp9_partition_tree`                   | `vp9_entropymode.c:262-264` |
| `default_intra_inter_p`                | `vp9_entropymode.c:266-268` |
| `default_comp_inter_p`                 | `vp9_entropymode.c:270-272` |
| `default_comp_ref_p`                   | `vp9_entropymode.c:274-275` |
| `default_single_ref_p`                 | `vp9_entropymode.c:277-279` |
| `default_tx_probs`                     | `vp9_entropymode.c:281-286` |
| `default_skip_probs`                   | `vp9_entropymode.c:313` |
| `default_switchable_interp_prob`       | `vp9_entropymode.c:315-321` |
| `vp9_switchable_interp_tree`           | `vp9_entropymode.c:337-338` |
| `vp9_adapt_mode_probs`                 | `vp9_entropymode.c:340-409` |
| `tx_counts_to_branch_counts_*`         | `vp9_entropymode.c:288-311` |
| `init_mode_probs`                      | `vp9_entropymode.c:323-335` |
| `vp9_setup_past_independence`          | `vp9_entropymode.c:424-469` |
| `default_nmv_context`                  | `vp9_entropymv.c:30-54` |
| `log_in_base_2[1024]`                  | `vp9_entropymv.c:56-97` |
| `vp9_get_mv_class`                     | `vp9_entropymv.c:103-109` |
| `vp9_mv_joint_tree`                    | `vp9_entropymv.c:14-16` |
| `vp9_mv_class_tree`                    | `vp9_entropymv.c:18-23` |
| `vp9_mv_class0_tree`                   | `vp9_entropymv.c:25` |
| `vp9_mv_fp_tree`                       | `vp9_entropymv.c:27-28` |
| `vp9_inc_mv`                           | `vp9_entropymv.c:139-152` |
| `vp9_init_mv_probs`                    | `vp9_entropymv.c:191` |
| `vp9_adapt_mv_probs`                   | `vp9_entropymv.c:154-189` |
| `vp9_encode_mv`                        | `vp9_encodemv.c:210-230` |
| `encode_mv_component`                  | `vp9_encodemv.c:31-67` |
| `vp9_write_nmv_probs`                  | `vp9_encodemv.c:172-208` |
| `update_mv` (encoder)                  | `vp9_encodemv.c:143-155` |
| `vp9_segment_tree`                     | `vp9_seg_common.c:58-60` |
| `vp9_default_scan_orders`              | `vp9_scan.c:691-696` |
| `vp9_scan_orders` (per-tx-type)        | `vp9_scan.c:698-725` |
| `get_scan` selector                    | `vp9_scan.h:42-52` |
| `get_coef_context`                     | `vp9_scan.h:35-40` |
| `vp9_tree_probs_from_distribution`     | `vp9_treewriter.c:54-58` |
| `vp9_tokens_from_tree`                 | `vp9_treewriter.c:29-32` |
| `vp9_write_tree` inline                | `vp9_treewriter.h:31-39` |
| `vp9_write_token` inline               | `vp9_treewriter.h:41-45` |
| `vp9_write_prob_diff_update`           | `vp9_subexp.c:112-115` |
| `vp9_cond_prob_diff_update`            | `vp9_subexp.c:182-196` |
| `encode_term_subexp`                   | `vp9_subexp.c:100-110` |
| `encode_uniform`                       | `vp9_subexp.c:84-93` |
| `remap_prob`                           | `vp9_subexp.c:44-77` |
| `vp9_diff_update_prob` (decoder)       | `vp9_dsubexp.c:67-72` |
| `decode_term_subexp`                   | `vp9_dsubexp.c:60-65` |
| `decode_uniform`                       | `vp9_dsubexp.c:23-28` |
| `inv_remap_prob`                       | `vp9_dsubexp.c:30-58` |
| Superframe index encoder               | `vp9_cx_iface.c:1216-1271` |
| Superframe index decoder               | `vp9_decoder.c:527-585` |
