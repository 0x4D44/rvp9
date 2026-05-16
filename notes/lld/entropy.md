# LLD: entropy

## 1. Purpose

`rvp9-core::entropy` owns the load-bearing primitives of VP9's compressed
bitstream and the encoder/decoder probability machinery: the boolean
(arithmetic) coder writer + reader, the per-frame `FrameContext` snapshot,
the per-frame `FrameCounts` statistics bucket, the backward-adaptation
kernels that fold counts into probabilities between frames, the static
default probability tables (intra/inter mode probs, partition probs,
coefficient model, MV context, skip / interp / tx-size probs, and
`vp9_pareto8_full`), and the encoder cost framework — `VP9_PROB_COST[256]`
plus the three `cost_zero` / `cost_one` / `cost_bit` accessors that every
encoder cost computation reads.

Per L-09 (locked: share via `rvp9-core`) the module is consumed by *both*
the decoder (for parsing the compressed header and tile data) and the
encoder (for cost-LUT derivation and tile-pack). Bit-exactness is
mandatory (PLAN.md D8 / 04-bitstream-entropy.md §4.9): the boolean coder
state machine, the probability quantization rule, the backward
adaptation arithmetic, and every static table must be byte-identical to
libvpx for matched input.

Out of scope, owned by sibling LLDs:

- The uncompressed-header writer/reader, superframe index assembly, and
  the higher-level frame-pack orchestration are owned by
  `notes/lld/bitstream.md`. This LLD provides the `BoolWriter` /
  `BoolReader` primitives that `bitstream.md` composes.
- Coefficient *scan orders* (`vp9_scan.c` — `scan`, `iscan`, `neighbors`)
  and the tokenizer R-D dispatch tables (`dct_cat_lt_10_value_tokens`,
  `vp9_extra_bits[]`, the `vp9_cat6_*` cost tables) live in
  `notes/lld/transforms.md` (scan side) and the encoder's tokenizer
  (`rvp9-encoder::tokenize`). This LLD owns *only* the `ScanOrder`
  reference type because the coefficient coder consumes one.
- Cost-LUT derivation per frame (`vp9_initialize_rd_consts` builds
  `inter_mode_cost`, `partition_cost`, `intra_uv_mode_cost`, …) is owned
  by `notes/lld/mode-decision.md`. This LLD owns *only* the cost-of-bit
  LUT (D-14: `vp9_prob_cost[256]`) and the three macros; per-symbol cost
  arrays are built downstream from those primitives.

---

## 2. libvpx origin

Primary specialist: A4 (`notes/vp9-analysis/04-bitstream-entropy.md`).
Each Rust item below is a port of a specific libvpx file:line range.

| rvp9 item | libvpx origin |
|---|---|
| `BoolWriter` | `third_party/libvpx/vpx_dsp/bitwriter.{h,c}`; struct at `bitwriter.h:28-40`, `vpx_start_encode` at `bitwriter.c:20`, `vpx_write` inline at `bitwriter.h:46-116`, `vpx_stop_encode` at `bitwriter.c:33` |
| `BoolReader` | `third_party/libvpx/vpx_dsp/bitreader.{h,c}`; struct at `bitreader.h:40-50`, `vpx_reader_init` at `bitreader.c:21`, `vpx_read` inline at `bitreader.h:77-136`, `vpx_reader_fill` at `bitreader.c:38` |
| `VPX_NORM` LUT | `third_party/libvpx/vpx_dsp/prob.c:13-24` (256-byte) |
| `FrameContext` | `third_party/libvpx/vp9/common/vp9_entropymode.h:44-60`; `FRAME_CONTEXTS = 4` at `vp9_onyxc_int.h:44-45` |
| `FrameCounts` | `vp9/common/vp9_entropymode.h:62-79` + `tx_counts` at `vp9_entropymode.h:37-42` + `vp9_coeff_count_model` at `vp9/common/vp9_entropy.h:153-155` |
| `NmvContext` / `NmvContextCounts` | `vp9/common/vp9_entropymv.h:88-102` + `:114-128` |
| `CoefProbsModel` | `vp9_entropy.h:150-151` (`vp9_coeff_probs_model`) |
| Probability merge (`merge_probs`, `mode_mv_merge_probs`, `weighted_prob`) | `vpx_dsp/prob.h:60-87` + `count_to_update_factor[21]` at `vpx_dsp/prob.h:79` |
| Tree merge (`vpx_tree_merge_probs`) | `vpx_dsp/prob.c:26-43` |
| `vp9_adapt_coef_probs` | `vp9/common/vp9_entropy.c:1055-1100`; constants at `vp9_entropy.c:1048-1053` |
| `vp9_adapt_mode_probs` | `vp9/common/vp9_entropymode.c:340-420` |
| `vp9_adapt_mv_probs` | `vp9/common/vp9_entropymv.c:154-200` |
| Sub-exponential prob update (`encode_term_subexp` / `decode_term_subexp`) | `vp9/encoder/vp9_subexp.c:100-180` + `vp9/decoder/vp9_dsubexp.c:23-90` |
| Probability defaults (intra, inter, partition, skip, tx, interp, coef, MV, segmentation) | `vp9/common/vp9_entropymode.c` (mode/partition); `vp9/common/vp9_entropy.c` (coef + pareto8); `vp9/common/vp9_entropymv.c` (MV); `vp9/common/vp9_seg_common.c` (segment) |
| `VP9_PROB_COST[256]` (D-14) | `vp9/encoder/vp9_cost.c:16-37`; macros at `vp9/encoder/vp9_cost.h:21-30` |
| `BitstreamWorker` (per-tile) | `vp9/encoder/vp9_bitstream.h:20-32` |
| `accumulate_frame_counts` | `vp9/encoder/vp9_ethread.c:243-247` and the count-summation kernel (scalar-only, A6 §6) |
| Probability type / tree convention | `vpx_dsp/prob.h:25-46` |

Cross-reference: 00-overview §4.3 (frame contexts) and §4.4 (per-worker
counts).

---

## 3. Public API

All signatures live in `crates/rvp9-core/src/entropy/`. Visibility rules:

- The boolean-coder types (`BoolWriter`, `BoolReader`) are `pub` at the
  crate root so `rvp9-encoder::bitstream` and `rvp9-decoder::tile` can
  use them.
- `FrameContext`, `FrameCounts`, `NmvContext` and the default-prob
  statics are `pub` at the crate root.
- `VP9_PROB_COST` and the three cost accessors are `pub`.
- Backward-adaptation kernels are `pub` but typed in terms of the
  shared types so neither crate has to redeclare them.

### 3.1 Boolean coder — writer

```rust
/// Streaming arithmetic-coder writer; mirrors libvpx `vpx_writer`.
///
/// The writer is *not* `Sync` — it carries the active interval state
/// (`lowvalue`, `range`, `count`). A single tile owns a single
/// `BoolWriter` for the duration of `encode_tile`.
///
/// The borrowed `&mut [u8]` is the destination byte buffer; the writer
/// updates `pos` as bytes are emitted. On `pos == buffer.len()`, the
/// writer sets `error = true` (sticky) and silently drops further
/// emits — see libvpx `vpx_writer.error` semantics at `bitwriter.h:33-37`.
pub struct BoolWriter<'a> { /* fields in §4 */ }

impl<'a> BoolWriter<'a> {
    /// `vpx_start_encode` (`bitwriter.c:20`). Sets `lowvalue=0`,
    /// `range=255`, `count=-24`, `pos=0`, then writes the spec-mandated
    /// `0` marker bit (under `vpx_prob_half = 128`).
    ///
    /// Preconditions: `buffer.len() >= 1`. Postconditions:
    /// `self.pos == 0`, marker bit has been written.
    pub fn start(buffer: &'a mut [u8]) -> Self;

    /// `vpx_write` (`bitwriter.h:46-116`). Writes one bit with the given
    /// 8-bit probability of zero. Probability **must** be in `[1, 255]`
    /// (zero is forbidden — §4.9 item 1). Panic policy: debug-asserts
    /// `prob != 0`; in release the lookup still works but the result is
    /// not spec-conformant.
    ///
    /// Hot path: marked `#[inline]`. The Stage 3 LLD bench (Q4.2) will
    /// decide whether `#[inline(always)]` is preferable.
    #[inline]
    pub fn write(&mut self, bit: bool, prob: u8);

    /// `vpx_write_bit` (`bitwriter.h:118`): write a fair-coin bit under
    /// `vpx_prob_half = 128`. Equivalent to `self.write(bit, 128)`;
    /// kept distinct for parity with libvpx call sites.
    #[inline]
    pub fn write_bit(&mut self, bit: bool);

    /// `vpx_write_literal` (`bitwriter.h:124`): write the `bits` low-order
    /// bits of `value`, MSB-first, each under `vpx_prob_half`.
    pub fn write_literal(&mut self, value: u32, bits: u8);

    /// `vp9_write_tree` (`vp9_treewriter.h:31-39`): walk a flattened
    /// probability tree to encode `(value, len)` (MSB-first). `tree` is
    /// the `[vpx_tree_index]` array; `probs` is the per-internal-node
    /// probability slice indexed by `node_idx >> 1`. See §2.5 of A4 for
    /// the tree convention.
    pub fn write_tree(
        &mut self,
        tree: &[i8],
        probs: &[u8],
        value: u32,
        len: u8,
        start_node: u8,
    );

    /// `vp9_write_token` convenience wrapper (`vp9_treewriter.h:41-45`).
    #[inline]
    pub fn write_token(&mut self, tree: &[i8], probs: &[u8], token: TokenEncoding) {
        self.write_tree(tree, probs, token.value, token.len, 0);
    }

    /// `vpx_stop_encode` (`bitwriter.c:33`): drains 32 zero bits, then
    /// appends a `0` byte if the last emitted byte matches the
    /// superframe-index marker pattern `(buffer[pos-1] & 0xe0) == 0xc0`.
    /// Returns `Err` if the writer overflowed at any point.
    pub fn stop(self) -> Result<usize, EntropyError>;

    /// Sticky overflow status; mirrors libvpx `vpx_writer.error`.
    #[inline]
    pub fn had_error(&self) -> bool;

    /// Bytes written so far (libvpx `pos`). Used by `bitstream.md` to
    /// compute tile-size prefixes.
    #[inline]
    pub fn bytes_written(&self) -> usize;
}

/// `(value, len)` pair produced by `vp9_tokens_from_tree`
/// (`vp9_treewriter.c:13-27`). Encoder uses these to drive
/// `BoolWriter::write_tree`.
#[derive(Copy, Clone, Debug)]
pub struct TokenEncoding {
    pub value: u32,
    pub len: u8,
}
```

### 3.2 Boolean coder — reader

```rust
/// Streaming arithmetic-coder reader; mirrors libvpx `vpx_reader`.
///
/// The decoder operates on a 64-bit `BD_VALUE` (Q4.3 RESOLVED: we
/// hardcode `u64` because 32-bit targets are explicitly out of scope —
/// PLAN.md success criteria target only x86_64 + aarch64; document in
/// the divergence ledger).
pub struct BoolReader<'a> { /* fields in §4 */ }

impl<'a> BoolReader<'a> {
    /// `vpx_reader_init` (`bitreader.c:21`). Sets `value=0`, `count=-8`,
    /// `range=255`, refills, then reads and asserts the marker bit is
    /// `0`. Returns `Err(EntropyError::BadMarkerBit)` if the marker bit
    /// is non-zero (matches libvpx behavior of failing the parse).
    pub fn init(buffer: &'a [u8]) -> Result<Self, EntropyError>;

    /// `vpx_read` (`bitreader.h:77-136`). Reads one bit with the given
    /// probability of zero. Hot path; marked `#[inline]`. Bit-exact with
    /// libvpx's formula: `split = (range * prob + (256 - prob)) >> 8`.
    /// **Do not "simplify"** — see §4.9 item 2 of A4.
    #[inline]
    pub fn read(&mut self, prob: u8) -> bool;

    /// `vpx_read_bit` (`bitreader.h:138`).
    #[inline]
    pub fn read_bit(&mut self) -> bool;

    /// `vpx_read_literal` (`bitreader.h:142`): read `bits` MSB-first.
    pub fn read_literal(&mut self, bits: u8) -> u32;

    /// `vpx_read_tree` (`bitreader.h:150`).
    pub fn read_tree(&mut self, tree: &[i8], probs: &[u8]) -> i8;

    /// `vpx_reader_has_error` (`bitreader.h:59`): true once the reader
    /// has consumed past `buffer_end` (count crossed `LOTS_OF_BITS`).
    #[inline]
    pub fn had_error(&self) -> bool;
}
```

### 3.3 Cost framework (D-14)

```rust
/// `vp9_prob_cost[256]` from `vp9/encoder/vp9_cost.c:16-37`.
///
/// `VP9_PROB_COST[p] = round(-log2(p/256) * 512)` for `p in [1, 255]`.
/// Entry `[0]` is the bogus placeholder `4096` so that the lookup can
/// be unconditionally indexed by an 8-bit probability (probabilities
/// are nominally in `[1, 255]` because `0` is forbidden — §4.9).
///
/// Without this table, every encoder cost computation silently diverges
/// from libvpx; it was the D-14 Blocker resolved by the Stage 2 prep
/// patch to A4 §5.11.0.
pub static VP9_PROB_COST: [u16; 256] = [
    4096, 4096, 3584, 3284, /* ... 252 more verbatim entries ... */
];

/// `VP9_PROB_COST_SHIFT = 9` (`vp9_cost.h`). The fixed-point shift on
/// every cost value: 1 bit costs `512` cost-units; `0.5` bits cost `256`.
pub const VP9_PROB_COST_SHIFT: u32 = 9;

/// `vp9_cost_zero(prob) = vp9_prob_cost[prob]` (`vp9_cost.h:26`).
#[inline]
pub const fn cost_zero(prob: u8) -> u16 {
    VP9_PROB_COST[prob as usize]
}

/// `vp9_cost_one(prob) = vp9_prob_cost[256 - prob]` (`vp9_cost.h:27`).
/// Computed as `cost_zero(256 - prob)`. Note: `256 - prob` for
/// `prob in [1, 255]` yields `[1, 255]`, so the lookup is in-range.
/// For `prob == 0` the C macro would index `vp9_prob_cost[256]` which
/// is out-of-bounds; we debug-assert `prob >= 1`.
#[inline]
pub const fn cost_one(prob: u8) -> u16 {
    // (256 - prob) for prob in [1, 255] yields [1, 255].
    VP9_PROB_COST[256usize - prob as usize]
}

/// `vp9_cost_bit(prob, bit)` (`vp9_cost.h:29`).
#[inline]
pub const fn cost_bit(prob: u8, bit: bool) -> u16 {
    if bit { cost_one(prob) } else { cost_zero(prob) }
}
```

### 3.4 Frame context + counts

```rust
/// `FRAME_CONTEXT` (`vp9_entropymode.h:44-60`). One per active frame;
/// stored in `Common::fc`. There are also `FRAME_CONTEXTS = 4` saved
/// contexts in `Common::frame_contexts` selected by `frame_context_idx`
/// (a 2-bit field in the uncompressed header — `bitstream.md`).
///
/// Mutated only between frames by `adapt_mode_probs`, `adapt_coef_probs`,
/// `adapt_mv_probs`; immutable during a frame encode. Lives behind
/// `&Common` for tile workers (see HLD-arch §3.4, HLD-threading §5.3).
#[derive(Clone)]
#[repr(C)]
pub struct FrameContext {
    pub y_mode_prob:           [[u8; INTRA_MODES - 1]; BLOCK_SIZE_GROUPS],     // [4][9]
    pub uv_mode_prob:          [[u8; INTRA_MODES - 1]; INTRA_MODES],           // [10][9]
    pub partition_prob:        [[u8; PARTITION_TYPES - 1]; PARTITION_CONTEXTS],// [16][3]
    pub coef_probs:            [[CoefProbsModel; PLANE_TYPES]; TX_SIZES],      // [4][2]
    pub switchable_interp_prob:
        [[u8; SWITCHABLE_FILTERS - 1]; SWITCHABLE_FILTER_CONTEXTS],            // [4][2]
    pub inter_mode_probs:      [[u8; INTER_MODES - 1]; INTER_MODE_CONTEXTS],   // [7][3]
    pub intra_inter_prob:      [u8; INTRA_INTER_CONTEXTS],                     // [4]
    pub comp_inter_prob:       [u8; COMP_INTER_CONTEXTS],                      // [5]
    pub single_ref_prob:       [[u8; 2]; REF_CONTEXTS],                        // [5][2]
    pub comp_ref_prob:         [u8; REF_CONTEXTS],                             // [5]
    pub tx_probs:              TxProbs,
    pub skip_probs:            [u8; SKIP_CONTEXTS],                            // [3]
    pub nmvc:                  NmvContext,
    pub initialized:           bool,
}

/// `vp9_coeff_probs_model` (`vp9_entropy.h:150-151`).
pub type CoefProbsModel =
    [[[[u8; UNCONSTRAINED_NODES]; COEFF_CONTEXTS]; COEF_BANDS]; REF_TYPES];

/// `struct tx_probs` (`vp9_entropymode.h:31-35`).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct TxProbs {
    pub p32x32: [[u8; TX_SIZES - 1]; TX_SIZE_CONTEXTS], // [2][3]
    pub p16x16: [[u8; TX_SIZES - 2]; TX_SIZE_CONTEXTS], // [2][2]
    pub p8x8:   [[u8; TX_SIZES - 3]; TX_SIZE_CONTEXTS], // [2][1]
}

/// `FRAME_COUNTS` (`vp9_entropymode.h:62-79`). Per-frame statistics
/// bucket. Per-worker copies live in `ThreadData::counts`
/// (HLD-threading §5.6, L-57); merged back via
/// `accumulate_frame_counts` on the main thread.
///
/// Approximate size: ~8-10 KB (dominated by `coef`). Boxed when
/// per-worker (~80 workers × 10 KB = 800 KB), inline-owned in `Common`.
#[derive(Clone)]
#[repr(C)]
pub struct FrameCounts {
    pub y_mode:           [[u32; INTRA_MODES]; BLOCK_SIZE_GROUPS],          // [4][10]
    pub uv_mode:          [[u32; INTRA_MODES]; INTRA_MODES],                // [10][10]
    pub partition:        [[u32; PARTITION_TYPES]; PARTITION_CONTEXTS],     // [16][4]
    pub coef:             [[CoefCountModel; PLANE_TYPES]; TX_SIZES],        // [4][2]
    pub eob_branch:
        [[[[[u32; COEFF_CONTEXTS]; COEF_BANDS]; REF_TYPES]; PLANE_TYPES]; TX_SIZES],
    pub switchable_interp:
        [[u32; SWITCHABLE_FILTERS]; SWITCHABLE_FILTER_CONTEXTS],            // [4][3]
    pub inter_mode:       [[u32; INTER_MODES]; INTER_MODE_CONTEXTS],        // [7][4]
    pub intra_inter:      [[u32; 2]; INTRA_INTER_CONTEXTS],                 // [4][2]
    pub comp_inter:       [[u32; 2]; COMP_INTER_CONTEXTS],                  // [5][2]
    pub single_ref:       [[[u32; 2]; 2]; REF_CONTEXTS],                    // [5][2][2]
    pub comp_ref:         [[u32; 2]; REF_CONTEXTS],                         // [5][2]
    pub tx:               TxCounts,
    pub skip:             [[u32; 2]; SKIP_CONTEXTS],                        // [3][2]
    pub mv:               NmvContextCounts,
}

/// `vp9_coeff_count_model` (`vp9_entropy.h:153-155`). The trailing
/// `UNCONSTRAINED_NODES + 1` slot is `EOB_MODEL_TOKEN = 3`
/// (`vp9_entropy.h:76`) — counted distinctly so the EOB branch
/// probability can be merged.
pub type CoefCountModel =
    [[[[u32; UNCONSTRAINED_NODES + 1]; COEFF_CONTEXTS]; COEF_BANDS]; REF_TYPES];

/// `struct tx_counts` (`vp9_entropymode.h:37-42`).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct TxCounts {
    pub p32x32:    [[u32; TX_SIZES];     TX_SIZE_CONTEXTS], // [2][4]
    pub p16x16:    [[u32; TX_SIZES - 1]; TX_SIZE_CONTEXTS], // [2][3]
    pub p8x8:      [[u32; TX_SIZES - 2]; TX_SIZE_CONTEXTS], // [2][2]
    pub tx_totals: [u32; TX_SIZES],                         // [4]
}
```

Constants (verbatim from A4 §2.6):

```rust
pub const FRAME_CONTEXTS:         usize = 4;   // vp9_onyxc_int.h:44-45
pub const FRAME_CONTEXTS_LOG2:    u8    = 2;
pub const BLOCK_SIZE_GROUPS:      usize = 4;   // vp9_entropymode.h:23
pub const TX_SIZE_CONTEXTS:       usize = 2;   // vp9_entropymode.h:25
pub const INTRA_MODES:            usize = 10;  // vp9_enums.h:127
pub const INTER_MODES:            usize = 4;   // vp9_enums.h:129
pub const INTER_MODE_CONTEXTS:    usize = 7;   // vp9_enums.h:132
pub const INTRA_INTER_CONTEXTS:   usize = 4;   // vp9_enums.h:137
pub const COMP_INTER_CONTEXTS:    usize = 5;   // vp9_enums.h:138
pub const REF_CONTEXTS:           usize = 5;   // vp9_enums.h:139
pub const SKIP_CONTEXTS:          usize = 3;   // vp9_enums.h:131
pub const PARTITION_CONTEXTS:     usize = 16;  // vp9_enums.h:74
pub const PARTITION_TYPES:        usize = 4;   // vp9_enums.h:68
pub const SWITCHABLE_FILTERS:     usize = 3;   // vp9_filter.h:26
pub const SWITCHABLE_FILTER_CONTEXTS: usize = 4; // vp9_filter.h:31 (= 3 + 1)
pub const SWITCHABLE:             usize = 4;   // vp9_filter.h:32 (interp+bilinear)
pub const TX_SIZES:               usize = 4;   // vp9_enums.h:82
pub const PLANE_TYPES:            usize = 2;   // vp9_enums.h:108
pub const REF_TYPES:              usize = 2;   // vp9_entropy.h:86
pub const COEF_BANDS:             usize = 6;   // vp9_entropy.h:89
pub const COEFF_CONTEXTS:         usize = 6;   // vp9_entropy.h:107
pub const UNCONSTRAINED_NODES:    usize = 3;   // vp9_entropy.h:142
pub const PIVOT_NODE:             usize = 2;   // vp9_entropy.h:144
pub const ENTROPY_NODES:          usize = 11;
pub const MODEL_NODES:            usize = ENTROPY_NODES - UNCONSTRAINED_NODES; // 8

pub const DIFF_UPDATE_PROB:       u8    = 252; // vp9_entropy.h:24
pub const MV_UPDATE_PROB:         u8    = 252; // vp9_entropymv.h:35
pub const MAX_PROB:               u8    = 255;
pub const VPX_PROB_HALF:          u8    = 128;
pub const MODE_MV_COUNT_SAT:      u32   = 20;  // vpx_dsp/prob.h:33

// Coefficient adaptation constants (vp9_entropy.c:1048-1053).
pub const COEF_COUNT_SAT:               u32 = 24;
pub const COEF_MAX_UPDATE_FACTOR:       u32 = 112;
pub const COEF_COUNT_SAT_KEY:           u32 = 24;
pub const COEF_MAX_UPDATE_FACTOR_KEY:   u32 = 112;
pub const COEF_COUNT_SAT_AFTER_KEY:     u32 = 24;
pub const COEF_MAX_UPDATE_FACTOR_AFTER_KEY: u32 = 128;
```

`BAND_COEFF_CONTEXTS(band)` is `if band == 0 { 3 } else { 6 }`
(`vp9_entropy.h:108`). Stored as a `const fn`:

```rust
#[inline]
pub const fn band_coeff_contexts(band: usize) -> usize {
    if band == 0 { 3 } else { 6 }
}
```

### 3.5 MV context

```rust
/// `nmv_context` (`vp9_entropymv.h:88-102`).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct NmvContext {
    pub joints:    [u8; MV_JOINTS - 1],          // 3
    pub comps:     [NmvComponent; 2],
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct NmvComponent {
    pub sign:       u8,
    pub classes:    [u8; MV_CLASSES - 1],        // 10
    pub class0:     [u8; CLASS0_SIZE - 1],       // 1
    pub bits:       [u8; MV_OFFSET_BITS],        // 10
    pub class0_fp:  [[u8; MV_FP_SIZE - 1]; CLASS0_SIZE], // [2][3]
    pub fp:         [u8; MV_FP_SIZE - 1],        // 3
    pub class0_hp:  u8,
    pub hp:         u8,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct NmvContextCounts {
    pub joints:    [u32; MV_JOINTS],             // 4
    pub comps:     [NmvComponentCounts; 2],
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct NmvComponentCounts {
    pub sign:       [u32; 2],
    pub classes:    [u32; MV_CLASSES],           // 11
    pub class0:     [u32; CLASS0_SIZE],          // 2
    pub bits:       [[u32; 2]; MV_OFFSET_BITS],  // [10][2]
    pub class0_fp:  [[u32; MV_FP_SIZE]; CLASS0_SIZE], // [2][4]
    pub fp:         [u32; MV_FP_SIZE],           // 4
    pub class0_hp:  [u32; 2],
    pub hp:         [u32; 2],
}

pub const MV_JOINTS:      usize = 4;     // vp9_entropymv.h:39
pub const MV_CLASSES:     usize = 11;    // vp9_entropymv.h:46
pub const CLASS0_BITS:    usize = 1;     // vp9_entropymv.h:64
pub const CLASS0_SIZE:    usize = 2;     // 1 << CLASS0_BITS
pub const MV_OFFSET_BITS: usize = 10;    // MV_CLASSES + CLASS0_BITS - 2
pub const MV_FP_SIZE:     usize = 4;     // vp9_entropymv.h:69
pub const MV_MAX_BITS:    usize = 14;
pub const MV_MAX:         i32   = 16383; // (1 << 14) - 1
pub const MV_VALS:        usize = 32767; // 2 * MV_MAX + 1
```

### 3.6 Backward adaptation (merge_probs)

```rust
/// `merge_probs` (`vpx_dsp/prob.h:69`). Generic count→prob fold.
///
/// `prob_pre` is the prior, `ct` is `[count_zero, count_one]`,
/// `count_sat` saturates the count, `max_update_factor` caps the new
/// prob's weight. Returns the merged probability.
#[inline]
pub fn merge_probs(prob_pre: u8, ct: [u32; 2], count_sat: u32,
                   max_update_factor: u32) -> u8;

/// `mode_mv_merge_probs` (`vpx_dsp/prob.h:84`). Specialization with
/// `count_sat = 20`, `max_update_factor = 128`, table-driven via
/// `count_to_update_factor[21]` (verbatim from `vpx_dsp/prob.h:79`).
#[inline]
pub fn mode_mv_merge_probs(prob_pre: u8, ct: [u32; 2]) -> u8;

/// `vpx_tree_merge_probs` (`vpx_dsp/prob.c:26-43`). Walks `tree`
/// recursively, computing per-internal-node `ct[2]` from `counts[]`
/// and calling `mode_mv_merge_probs` at each. The result lives in
/// `probs[]` (one entry per internal node).
pub fn tree_merge_probs(
    tree: &[i8],
    pre_probs: &[u8],
    counts: &[u32],
    probs: &mut [u8],
);

/// `vp9_adapt_mode_probs` (`vp9_entropymode.c:340`). Folds
/// `Common::counts` into a fresh `FrameContext` derived from
/// `Common::pre_fc`. Caller stages the result into `cm.fc` between frames.
pub fn adapt_mode_probs(pre_fc: &FrameContext, counts: &FrameCounts,
                        out: &mut FrameContext);

/// `vp9_adapt_coef_probs` (`vp9_entropy.c:1084`). Folds coefficient
/// counts; selects between `(COEF_COUNT_SAT_KEY, COEF_MAX_UPDATE_FACTOR_KEY)`,
/// `(COEF_COUNT_SAT_AFTER_KEY, COEF_MAX_UPDATE_FACTOR_AFTER_KEY)`, and
/// vanilla constants based on `frame_type` and `prev_frame_type`.
pub fn adapt_coef_probs(pre_fc: &FrameContext, counts: &FrameCounts,
                        frame_type: FrameType, prev_frame_type: FrameType,
                        out: &mut FrameContext);

/// `vp9_adapt_mv_probs` (`vp9_entropymv.c:154`). Folds MV counts.
pub fn adapt_mv_probs(pre_nmvc: &NmvContext, counts: &NmvContextCounts,
                      allow_high_precision_mv: bool,
                      out: &mut NmvContext);
```

### 3.7 Per-worker FRAME_COUNTS accumulation

```rust
/// `vp9_accumulate_frame_counts` (referenced from
/// `vp9_ethread.c:243-247`). Scalar-only kernel (A6 §6, 00-overview §4.4).
/// Field-by-field `+=` of every `u32` count in `src` into `dst`.
///
/// HLD-threading §5.6 documents the per-worker baseline-copy + naive
/// summation protocol that double-counts the baseline. This kernel
/// mirrors libvpx exactly (divergence-ledger entry).
pub fn accumulate_frame_counts(dst: &mut FrameCounts, src: &FrameCounts);
```

### 3.8 Sub-exponential probability update codec

```rust
/// `vp9_cond_prob_diff_update` (`vp9_subexp.c:182`). Computes the
/// optimal `new_p` from counts, decides whether the savings are
/// positive, writes 1 boolean bit ("update or not") at `DIFF_UPDATE_PROB`,
/// and on update writes the sub-exponential delta.
pub fn cond_prob_diff_update(w: &mut BoolWriter<'_>, oldp: &mut u8,
                             ct: [u32; 2]);

/// `vp9_diff_update_prob` (`vp9_dsubexp.c:67`). Decoder dual: reads the
/// 1-bit gate; on update reads the sub-exponential delta and patches
/// `*p` in place.
pub fn diff_update_prob(r: &mut BoolReader<'_>, p: &mut u8);
```

### 3.9 Per-tile bitstream worker

```rust
/// Per-tile bitstream pack state; mirrors libvpx `VP9BitstreamWorkerData`
/// (`vp9_bitstream.h:20-32`). Owned by the tile worker for the duration
/// of one frame's pack.
///
/// **L-35 (locked):** bitstream packing is scalar and **single-tile-at-a-
/// time** in v1. We mirror libvpx `encode_tiles_mt`
/// (`vp9_bitstream.c:991`) — multi-threaded pack with per-tile
/// `BoolWriter`s — but the boolean coder itself stays scalar (no SIMD
/// inside `BoolWriter::write`). See HLD-arch §3.5.
pub struct BitstreamWorker<'buf> {
    /// Destination byte slice for this tile's packed output. For tile 0
    /// (in `encode_tiles_mt`) this slice points directly into the
    /// final output; for tiles 1..N-1 it is a per-tile scratch buffer
    /// the main thread `memcpy`'s back after sync.
    pub bit_writer: BoolWriter<'buf>,
    pub tile_idx: u32,
    pub max_mv_magnitude: u32,
    /// `interp_filter_selected[1][SWITCHABLE]`. The libvpx struct
    /// allocates one row even though `VP9_COMP::interp_filter_selected`
    /// has `MAX_REFERENCE_FRAMES` rows; tile encode only writes the
    /// first row. (Per A4 §2.10 — the libvpx comment is explicit.)
    pub interp_filter_selected: [[u32; SWITCHABLE]; 1],
    /// Per-tile `MACROBLOCKD` snapshot. The full layout is owned by
    /// `notes/lld/mode-decision.md` / `notes/lld/bitstream.md`; this
    /// field is exposed so the per-tile pack code can derive
    /// above/left/segmentation contexts.
    pub xd: MacroblockD,
}
```

### 3.10 Error type

```rust
#[derive(Debug, thiserror::Error)]
pub enum EntropyError {
    /// Decoder init found a non-zero marker bit.
    /// `vpx_reader_init` returns this; libvpx hard-errors at
    /// `bitreader.c:34`.
    #[error("compressed header marker bit must be zero")]
    BadMarkerBit,

    /// Writer buffer overflowed at `pos == size`. Mirrors libvpx
    /// `vpx_writer.error` becoming sticky-true.
    #[error("boolean coder writer overflowed buffer")]
    WriterOverflow,

    /// Decoder consumed past `buffer_end`. Mirrors libvpx
    /// `vpx_reader_has_error` returning true.
    #[error("boolean coder reader ran past end of buffer")]
    ReaderOverflow,

    /// `bitstream.md` raises this when the compressed-header size
    /// would exceed the 16-bit field — Q4.9 from A4 §7.
    #[error("compressed header size > 16 bits")]
    CompressedHeaderTooLarge,
}
```

`MacroblockD` and `FrameType` are types pulled in from sibling modules
(`rvp9-core::frame`); cited here only because the public signatures
reference them.

---

## 4. Types

This section pins the concrete in-memory layout of the load-bearing
types so SIMD and cross-FFI debugging produce predictable byte offsets.
`#[repr(C)]` is used wherever libvpx's layout is observed by the encoder
through pointer arithmetic; otherwise default Rust layout is acceptable
because nothing inspects bytes directly.

### 4.1 `BoolWriter`

```rust
/// libvpx `vpx_writer` (`bitwriter.h:28-40`). Field semantics per A4 §2.1.
pub struct BoolWriter<'a> {
    /// Bottom of the active arithmetic interval, shifted so the next
    /// emitting byte sits in bits [31..24]. 32-bit `u32` matches libvpx's
    /// `unsigned int lowvalue` exactly (libvpx `int` is 32-bit on every
    /// supported target — PLAN.md D7).
    lowvalue: u32,

    /// Active interval width, always in `[128, 255]` after normalization.
    /// 32-bit so `(range - 1) * prob` cannot overflow when computing
    /// `split = 1 + (((range - 1) * prob) >> 8)`.
    range: u32,

    /// Bits in `lowvalue` waiting to be emitted. Signed because the
    /// initial value is `-24` (A4 §2.1).
    count: i32,

    /// Sticky overflow flag.
    error: bool,

    /// Next write position in `buffer`. `pos <= buffer.len()` invariant.
    pos: usize,

    /// Destination byte buffer. `'a` lifetime ensures non-overlapping
    /// per-tile slicing under `tile-threading.md`.
    buffer: &'a mut [u8],
}
```

Note: libvpx uses raw `unsigned int *buffer` with a separate `size`; we
fold this into `&mut [u8]` and read `pos` / `buffer.len()` as needed.
This is a non-divergent representation change (the wire-level behaviour
is identical).

**Marker traits:** `BoolWriter<'a>: !Send + !Sync` (default, because of
the `&mut` borrow). Tile workers cannot share writers; each gets its own
non-overlapping `&mut [u8]` slice per `tile-threading.md`.

**Panic-safety (Q4.4 from A4 §7):** if a panic strikes mid-`write`, the
borrowed buffer contents are in an indeterminate state. We do not
poison-on-panic explicitly; instead the writer is `!UnwindSafe` (the
default since it holds `&mut`), and the only `panic = "unwind"`
configuration is debug. Production `panic = "abort"` short-circuits the
issue. Divergence-ledger entry §7.

### 4.2 `BoolReader`

```rust
/// libvpx `vpx_reader` (`bitreader.h:40-50`). Field semantics per A4 §2.2.
pub struct BoolReader<'a> {
    /// Top `BD_VALUE_SIZE` bits of the active interval. We hardcode
    /// 64-bit per Q4.3 RESOLVED — 32-bit targets are out of scope; this
    /// matches libvpx behaviour on every 64-bit target.
    value: u64,

    /// Active interval width.
    range: u32,

    /// Valid bits in `value` minus 8. Negative means refill needed.
    count: i32,

    /// End and current pointer into the input buffer.
    buffer: &'a [u8],
    /// Bytes consumed (i.e., `buffer_pos == buffer.len()` is EOF).
    buffer_pos: usize,
}
```

The `decrypt_cb` / `decrypt_state` libvpx fields (`bitreader.h:46-47`)
are not ported — they are libvpx-internal and not used by VP9
conformance. Divergence-ledger entry §7.

`BD_VALUE_SIZE = 64` (compile-time constant). `LOTS_OF_BITS =
0x40000000` (the past-end sentinel) is a `const i32`.

**Marker traits:** `BoolReader<'a>: !Send + !Sync` (default). Each tile
decode owns its own `BoolReader` exclusively.

### 4.3 `VPX_NORM` and supporting tables

```rust
/// `vpx_norm[256]` (`vpx_dsp/prob.c:13-24`). `VPX_NORM[r] = 7 -
/// floor(log2(r))` for `r >= 1`, and `VPX_NORM[0] = 0`. Used to
/// renormalize after each boolean-coder event.
///
/// Bit-exact-mandatory (§4.9 item 3). Verbatim 256 bytes from libvpx.
pub static VPX_NORM: [u8; 256] = [
    0, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    /* ...250 more bytes verbatim... */
];

/// `count_to_update_factor[21]` (`vpx_dsp/prob.h:79`). Used by
/// `mode_mv_merge_probs` to avoid the per-call division.
pub static COUNT_TO_UPDATE_FACTOR: [u32; 21] = [
    0, 6, 12, 19, 25, 32, 38, 44, 51, 57, 64, 70, 76, 83, 89, 96, 102, 108, 115, 121, 128,
];
```

### 4.4 Static default probability tables

Stored as `pub static` items in `rvp9-core::entropy::defaults`. Every
table is verbatim from A4 §5; each is `'static` and `Sync` trivially
because they are read-only.

```rust
// Intra (vp9_entropymode.c)
pub static VP9_KF_Y_MODE_PROB: [[[u8; INTRA_MODES - 1]; INTRA_MODES]; INTRA_MODES];
pub static VP9_KF_UV_MODE_PROB: [[u8; INTRA_MODES - 1]; INTRA_MODES];
pub static DEFAULT_IF_Y_PROBS: [[u8; INTRA_MODES - 1]; BLOCK_SIZE_GROUPS];
pub static DEFAULT_IF_UV_PROBS: [[u8; INTRA_MODES - 1]; INTRA_MODES];

// Partition (vp9_entropymode.c)
pub static VP9_KF_PARTITION_PROBS: [[u8; PARTITION_TYPES - 1]; PARTITION_CONTEXTS];
pub static DEFAULT_PARTITION_PROBS: [[u8; PARTITION_TYPES - 1]; PARTITION_CONTEXTS];

// Inter / ref / skip / tx / interp (vp9_entropymode.c)
pub static DEFAULT_INTER_MODE_PROBS: [[u8; INTER_MODES - 1]; INTER_MODE_CONTEXTS];
pub static DEFAULT_INTRA_INTER_P:    [u8; INTRA_INTER_CONTEXTS];
pub static DEFAULT_COMP_INTER_P:     [u8; COMP_INTER_CONTEXTS];
pub static DEFAULT_COMP_REF_P:       [u8; REF_CONTEXTS];
pub static DEFAULT_SINGLE_REF_P:     [[u8; 2]; REF_CONTEXTS];
pub static DEFAULT_TX_PROBS:         TxProbs;
pub static DEFAULT_SKIP_PROBS:       [u8; SKIP_CONTEXTS];
pub static DEFAULT_SWITCHABLE_INTERP_PROB:
    [[u8; SWITCHABLE_FILTERS - 1]; SWITCHABLE_FILTER_CONTEXTS];

// Coefficient model (vp9_entropy.c)
pub static DEFAULT_COEF_PROBS_4X4:   CoefProbsModel;
pub static DEFAULT_COEF_PROBS_8X8:   CoefProbsModel;
pub static DEFAULT_COEF_PROBS_16X16: CoefProbsModel;
pub static DEFAULT_COEF_PROBS_32X32: CoefProbsModel;

/// `vp9_pareto8_full[255][8]` (`vp9_entropy.c`). The 255-row extension
/// table for trailing coefficient model nodes; indexed by
/// `context_tree[PIVOT_NODE] - 1` (§4.9 item 9).
pub static VP9_PARETO8_FULL: [[u8; MODEL_NODES]; 255];

// Coefficient bands + extra bits + energy class (vp9_entropy.c)
pub static VP9_COEFBAND_TRANS_4X4:    [u8; 16];
pub static VP9_COEFBAND_TRANS_8X8PLUS:[u8; /* max 1024 */ 1024];
pub static VP9_PT_ENERGY_CLASS:       [u8; ENTROPY_TOKENS];  // per A4 §5.7.2

// MV (vp9_entropymv.c)
pub static DEFAULT_NMV_CONTEXT: NmvContext;
pub static LOG_IN_BASE_2:       [u8; 1024];

// Segmentation (vp9_seg_common.c)
pub static SEG_TREE: [i8; /* TREE_SIZE(MAX_SEGMENTS) */ 14];
```

### 4.5 Trees

The 12 trees of A4 §5.3 are `pub static [i8; ...]` arrays sized via
`TREE_SIZE(leaf_count) = 2 * leaf_count - 2` (the `vpx_tree` convention
from `vpx_dsp/prob.h:42`).

```rust
pub static VP9_INTRA_MODE_TREE:         [i8; TREE_SIZE(INTRA_MODES)];        // 18
pub static VP9_INTER_MODE_TREE:         [i8; TREE_SIZE(INTER_MODES)];        // 6
pub static VP9_PARTITION_TREE:          [i8; TREE_SIZE(PARTITION_TYPES)];    // 6
pub static VP9_SWITCHABLE_INTERP_TREE:  [i8; TREE_SIZE(SWITCHABLE_FILTERS)]; // 4
pub static VP9_COEF_TREE:               [i8; TREE_SIZE(ENTROPY_TOKENS)];     // 22
pub static VP9_COEF_CON_TREE:           [i8; TREE_SIZE(8)];                  // 14
pub static VP9_MV_JOINT_TREE:           [i8; TREE_SIZE(MV_JOINTS)];          // 6
pub static VP9_MV_CLASS_TREE:           [i8; TREE_SIZE(MV_CLASSES)];         // 20
pub static VP9_MV_CLASS0_TREE:          [i8; TREE_SIZE(CLASS0_SIZE)];        // 2
pub static VP9_MV_FP_TREE:              [i8; TREE_SIZE(MV_FP_SIZE)];         // 6
pub static VP9_SEG_TREE:                [i8; TREE_SIZE(MAX_SEGMENTS)];       // 14
```

`TREE_SIZE` is a `const fn` (`vpx_dsp/prob.h:33`).

```rust
pub const fn tree_size(leaf_count: usize) -> usize { 2 * leaf_count - 2 }
```

### 4.6 `ScanOrder` (reference declaration only)

```rust
/// `ScanOrder` (`vp9_scan.h:26-30`). Owned by `transforms.md`; declared
/// here because the coefficient coder's signature references it.
pub struct ScanOrder {
    pub scan:      &'static [i16],
    pub iscan:     &'static [i16],
    pub neighbors: &'static [i16],
}
```

### 4.7 Memory layout / alignment

Per A4 §6, libvpx applies `DECLARE_ALIGNED(16, ...)` to scan tables,
neighbor tables, `vpx_norm`, `vp9_pareto8_full`, and the `vp9_catN_prob`
arrays. These alignments are for *consumers* (SIMD quant / tokenize
helpers) that live outside this LLD; the boolean coder itself does not
need alignment. Rust's default `static` alignment is sufficient
(`#[repr(align(16))]` may be added later if benchmarks demand it).

The cost LUT `VP9_PROB_COST: [u16; 256]` is 512 bytes; cache-line-aligned
implicitly because `pub static` items get natural alignment.

---

## 5. Internal algorithm

### 5.1 Boolean encoder — `BoolWriter::write` pseudocode

Verbatim from A4 §4.1 (`bitwriter.h:46-116`):

```text
write(bit, prob):
    split = 1 + (((range - 1) * prob) >> 8)        // §4.9 item 2: do NOT simplify
    if bit == 0:
        range = split
    else:
        lowvalue += split
        range -= split
    shift = VPX_NORM[range]
    range <<= shift
    count += shift
    if count >= 0:
        offset = shift - count
        // Carry propagation (§4.9 item 4). The about-to-emit byte may
        // need to absorb a +1 carry that propagates through any
        // already-emitted 0xff bytes.
        if (lowvalue << (offset - 1)) & 0x80000000:
            x = pos - 1
            while x >= 0 && buffer[x] == 0xff:
                buffer[x] = 0
                x -= 1
            buffer[x] += 1
        buffer[pos] = ((lowvalue >> (24 - offset)) & 0xff) as u8
        pos += 1
        lowvalue = (lowvalue << offset) & 0xffffff
        count -= 8
        shift = count       // reused as the next-left-shift below
    lowvalue <<= shift
```

**Q4.7 (A4 §7) — the 0x80000000 carry check.** On 64-bit builds libvpx's
`lowvalue` is still `unsigned int` (32-bit), so `lowvalue << (offset - 1)`
naturally truncates to 32 bits. We use `u32` for `lowvalue`; the Rust
shift is `lowvalue.wrapping_shl(offset - 1) & 0x8000_0000`. Precondition
`offset` is in `[1, 8]` (it equals `shift - count` where `shift in [0, 7]`
from `VPX_NORM` and `count >= 0` to enter the branch, so `offset` ranges
over the same `[1, 8]` libvpx assumes). Debug-assert this on the branch
entry; release builds omit the assertion.

**Carry boundary safety.** The carry loop walks back at most until `x = 0`
(the marker bit is the first emitted byte; it cannot be 0xff because the
marker bit forces `buffer[0]` low bits — explicitly, the first write
of `vpx_prob_half = 128` with `bit = 0` cannot produce a leading 0xff
under the post-marker-bit state). We rely on this invariant rather than
checking `x >= 0` (libvpx is content to UB on the invariant breaking;
our Rust port debug-asserts `x >= 0` at loop entry).

### 5.2 Boolean encoder — `start` and `stop`

`BoolWriter::start` (A4 §4.1):

```text
start(buffer):
    lowvalue = 0, range = 255, count = -24, pos = 0
    error = false
    write_bit(false)        // the spec-mandated marker bit (vpx_prob_half = 128)
```

`BoolWriter::stop` (A4 §4.1, `bitwriter.c:33`):

```text
stop():
    for _ in 0..32: write_bit(false)
    if (buffer[pos-1] & 0xe0) == 0xc0:
        buffer[pos] = 0     // disambiguate from superframe index marker
        pos += 1
    return Ok(pos) if !error else Err(WriterOverflow)
```

**Q4.4 (A4 §7) — panic mid-write.** `stop` consumes `self`, so the user
cannot accidentally re-use a partially-flushed writer. The borrowed
slice's contents on a panic-mid-write are undefined; the caller is
expected to discard the tile and restart the frame. Documented in §7.

### 5.3 Boolean decoder — `BoolReader::read` pseudocode

Verbatim from A4 §4.2 (`bitreader.h:77-136`):

```text
read(prob):
    split = (range * prob + (256 - prob)) >> 8       // §4.9 item 2
    if count < 0: fill()
    bigsplit = (split as u64) << (64 - 8)            // BD_VALUE_SIZE = 64
    if value < bigsplit:
        range = split
        bit = false
    else:
        range -= split
        value -= bigsplit
        bit = true
    shift = VPX_NORM[(range & 0xff) as u8]
    range <<= shift
    value <<= shift
    count -= shift
    return bit
```

**Decoder split equivalence (§4.9 item 2):** decoder `split = (range *
prob + (256 - prob)) >> 8 = 1 + ((range - 1) * prob >> 8) =` encoder
`split` for all `range, prob in [1, 255]`. Algebraic proof in A4 §3.6.
Both forms must be preserved verbatim.

### 5.4 Decoder fill (`vpx_reader_fill`, `bitreader.c:38`)

```text
fill():
    bits_left = (buffer.len() - buffer_pos) * 8 + (BD_VALUE_SIZE - 8 - count)
    bits = -count + (BD_VALUE_SIZE - 8 - count)         // see libvpx
    shift = BD_VALUE_SIZE - 8 - (count + 8)

    while shift >= 0 && bits_left > 0:
        let b = if buffer_pos < buffer.len() {
            let b = buffer[buffer_pos];
            buffer_pos += 1;
            b
        } else { break };
        value |= (b as u64) << shift;
        shift -= 8;
        count += 8;
        bits_left -= 8;

    if buffer_pos == buffer.len() && bits_left <= 0:
        count += LOTS_OF_BITS as i32     // mark "past end of stream"
```

Past-end-of-stream behaviour: once `count >= LOTS_OF_BITS`, subsequent
`read` calls produce deterministic `bit = false` (because `value < bigsplit`
always for any positive `split` — `value` is fully shifted-out at that
point). Matches libvpx's silent-zero-pad behaviour.

### 5.5 Tree write / read

Tree-write (A4 §3.7):

```text
write_tree(tree, probs, value, len, start_node):
    i = start_node
    while len > 0:
        len -= 1
        bit = (value >> len) & 1
        write(bit != 0, probs[i >> 1])
        i = tree[i + bit] as i8
        if i <= 0: break        // leaf reached; loop exits
```

Tree-read (A4 §3.8):

```text
read_tree(tree, probs) -> i8:
    i = 0
    while i > 0:
        let b = if read(probs[i >> 1]) { 1 } else { 0 };
        i = tree[i + b];
    return -i
```

Note the asymmetric loop test: write loops *until `len == 0`* (the bit
budget is known a priori from `vp9_tokens_from_tree`), while read loops
*until a leaf is hit* (it doesn't know the bit count in advance).

### 5.6 `cond_prob_diff_update` — encoder side

From `vp9_subexp.c:182`:

```text
cond_prob_diff_update(w, oldp, ct):
    upd = DIFF_UPDATE_PROB             // 252
    newp = get_binary_prob(ct[0], ct[1])
    savings = vp9_prob_diff_update_savings_search(ct, oldp, &newp, upd)
    if savings > 0:
        write(w, 1, upd)
        write_prob_diff_update(w, newp, oldp)     // see §5.7
        oldp = newp
    else:
        write(w, 0, upd)
```

`get_binary_prob(c0, c1)` rounds `((c0 << 8) + (c0 + c1) >> 1) / (c0 + c1)`
with the standard libvpx clipping to `[1, 255]`.

### 5.7 Sub-exponential codec (A4 §4.5)

Encode (`vp9_subexp.c:100`):

```text
encode_term_subexp(w, word):
    if word < 16:        write_bit(0); write_literal(word, 4)
    elif word < 32:      write_bit(1); write_bit(0); write_literal(word - 16, 4)
    elif word < 64:      write_bit(1); write_bit(1); write_bit(0); write_literal(word - 32, 5)
    else:                write_bit(1); write_bit(1); write_bit(1); encode_uniform(w, word - 64)
```

`encode_uniform(w, v)` (`vp9_subexp.c:84`):

```text
let l = 8, m = (1 << l) - 191 = 65
if v < m:  write_literal(v, l - 1)              # 7 bits
else:      write_literal(m + ((v - m) >> 1), 7)
           write_literal((v - m) & 1, 1)
```

Decode mirror in `vp9_dsubexp.c:60` + `:23`. `remap_prob` /
`inv_remap_prob` use the 254-entry `map_table` / `inv_map_table` from
A4 §5.9 (verbatim ports).

### 5.8 `merge_probs` and `mode_mv_merge_probs` (A4 §4.8)

`weighted_prob` (`vpx_dsp/prob.h:65`):

```rust
#[inline]
fn weighted_prob(p1: u8, p2: u8, factor: u32) -> u8 {
    // ROUND_POWER_OF_TWO(p1 * (256 - factor) + p2 * factor, 8)
    let sum = (p1 as u32) * (256 - factor) + (p2 as u32) * factor;
    (((sum + 128) >> 8) & 0xff) as u8
}
```

`merge_probs`:

```rust
pub fn merge_probs(pre: u8, ct: [u32; 2], count_sat: u32, max_uf: u32) -> u8 {
    let prob = get_binary_prob(ct[0], ct[1]);
    let count = (ct[0] + ct[1]).min(count_sat);
    let factor = max_uf * count / count_sat;
    weighted_prob(pre, prob, factor)
}
```

`mode_mv_merge_probs`:

```rust
pub fn mode_mv_merge_probs(pre: u8, ct: [u32; 2]) -> u8 {
    let den = ct[0] + ct[1];
    if den == 0 { return pre; }
    let prob = get_binary_prob(ct[0], ct[1]);
    let count = den.min(MODE_MV_COUNT_SAT);
    let factor = COUNT_TO_UPDATE_FACTOR[count as usize];
    weighted_prob(pre, prob, factor)
}
```

`tree_merge_probs` (`vpx_dsp/prob.c:26`) is a recursive walk; each
internal node's `(ct[0], ct[1])` is the sum of counts on the left vs
right subtree, and each call writes one `u8` into `probs[]`.

### 5.9 `adapt_coef_probs` (A4 §4.8, `vp9_entropy.c:1055`)

```text
adapt_coef_probs(pre_fc, counts, frame_type, prev_frame_type, out):
    (cs, uf) = match frame_type {
        KEY_FRAME              => (COEF_COUNT_SAT_KEY,       COEF_MAX_UPDATE_FACTOR_KEY),
        _ if prev_frame_type == KEY_FRAME
                               => (COEF_COUNT_SAT_AFTER_KEY, COEF_MAX_UPDATE_FACTOR_AFTER_KEY),
        _                      => (COEF_COUNT_SAT,           COEF_MAX_UPDATE_FACTOR),
    }
    for tx in 0..TX_SIZES:
      for plane in 0..PLANE_TYPES:
        for ref in 0..REF_TYPES:
          for band in 0..COEF_BANDS:
            for ctx in 0..band_coeff_contexts(band):
              let ct = counts.coef[tx][plane][ref][band][ctx]   // 4 entries
              let pre = pre_fc.coef_probs[tx][plane][ref][band][ctx]
              for node in 0..UNCONSTRAINED_NODES:
                let (c0, c1) = derive_node_counts(node, ct, counts.eob_branch[...])
                out.coef_probs[tx][plane][ref][band][ctx][node]
                    = merge_probs(pre[node], [c0, c1], cs, uf)
```

The exact `derive_node_counts` per-node split mirrors
`adapt_coef_probs` lines 1060-1097 of `vp9_entropy.c`. Each of the
three unconstrained nodes (EOB, ZERO, ONE) has a deterministic
`(c0, c1)` derived from `counts.coef[...]` plus `counts.eob_branch[...]`
for the EOB node.

### 5.10 `adapt_mode_probs` (A4 §4.8, `vp9_entropymode.c:340`)

Walks every per-context slot of `FrameContext` calling either
`mode_mv_merge_probs` (for binary contexts) or `tree_merge_probs` (for
tree-coded contexts). The visitation order is rigidly preserved (it
matters for byte parity only if any intermediate fold reads back a
just-written value — none do; each call is independent — but the order
is documented to make differential-test diffs comparable).

Order (verbatim from `vp9_entropymode.c:340-420`):

1. `intra_inter_prob[4]`
2. `comp_inter_prob[5]`
3. `comp_ref_prob[5]`
4. `single_ref_prob[5][2]` (10 contexts)
5. `inter_mode_probs[7][3]` via `tree_merge_probs` with `vp9_inter_mode_tree`
6. `y_mode_prob[4][9]` via `tree_merge_probs` with `vp9_intra_mode_tree`
7. `uv_mode_prob[10][9]` via `tree_merge_probs`
8. `partition_prob[16][3]` via `tree_merge_probs` with `vp9_partition_tree`
9. `switchable_interp_prob[4][2]` via `tree_merge_probs`
10. `skip_probs[3]`
11. `tx_probs.p32x32` / `p16x16` / `p8x8` via `tree_merge_probs`

### 5.11 `adapt_mv_probs` (A4 §4.8, `vp9_entropymv.c:154`)

Same shape; walks `joints`, then per-component `sign`, `classes`,
`class0`, `bits`, `class0_fp`, `fp`, `class0_hp` (if `allow_hp`), `hp`
(if `allow_hp`).

### 5.12 `accumulate_frame_counts`

```text
accumulate_frame_counts(dst, src):
    for each u32 field f in FrameCounts:
        dst.f += src.f
```

Field-by-field summation — a derive-macro generates it from the
`FrameCounts` shape (matches libvpx's `vp9_accumulate_frame_counts`
field walk). HLD-threading §5.6 documents the baseline-double-count
quirk: this kernel preserves it exactly (divergence-ledger entry §7).

### 5.13 Cost framework

Hot path: every `cost_zero(p)` is a single array index. The encoder cost
of writing a bit `b` under probability `p` is `cost_bit(p, b)`. The
encoder cost of writing a tree token is the sum over the path:

```rust
pub fn cost_tree(tree: &[i8], probs: &[u8], value: u32, len: u8,
                 start_node: u8) -> u32 {
    let mut i = start_node as i8;
    let mut total: u32 = 0;
    for k in (0..len).rev() {
        let bit = ((value >> k) & 1) as u8;
        total += cost_bit(probs[(i as usize) >> 1], bit != 0) as u32;
        i = tree[(i + bit as i8) as usize];
    }
    total
}
```

Used heavily by `mode-decision.md` to fill `inter_mode_cost`,
`partition_cost`, `intra_uv_mode_cost`. Cost-LUT *derivation* is owned
by `mode-decision.md` (00-overview §4.3); we only export the
`cost_zero` / `cost_one` / `cost_bit` / `cost_tree` primitives plus the
foundational `VP9_PROB_COST` table.

---

## 6. SIMD dispatch

**None.** The entropy coder and bitstream paths are entirely scalar in
libvpx (A4 §6: zero matches for `add_proto` against `bitwriter`,
`bitreader`, or `prob` in `vpx_dsp/vpx_dsp_rtcd_defs.pl`). Per L-35
(locked), this LLD's outputs stay scalar in v1 — there is no SIMD
bitstream packing.

The boolean coder is a tight scalar loop where the only realistic
optimisation lever is inlining (Q4.2 from A4 §7). Strategy:

- `#[inline]` on `BoolWriter::write`, `BoolReader::read`,
  `BoolWriter::write_bit`, `BoolReader::read_bit`. The Stage 3 LLD bench
  will decide if `#[inline(always)]` improves throughput; the LLD does
  *not* pre-commit either way.
- The carry-propagation while-loop (A4 §3.5) stays scalar and branchy.
  Production traces (per A4 §6) show 0xff-byte runs are short at
  cpu-used 7; a SIMD vectorization is not worth the complexity.

`VP9_PROB_COST` could in principle be batched (per-mode cost
derivation reads 9-12 entries) but the hot consumers are not on the
encoder's critical path — the per-frame cost-LUT build runs once on the
main thread. No SIMD planned.

Test strategy for scalar correctness uses property tests + a libvpx
differential harness (§9). Because there is no SIMD variant, the usual
per-arch differential test is not applicable.

---

## 7. Deliberate divergences from libvpx

Entries here feed `notes/divergence-ledger.md`. Format mirrors the
ledger's spec.

### 7.1 Boolean reader hardcoded to 64-bit `BD_VALUE`

**Where:** `rvp9_core::entropy::BoolReader::value: u64`. libvpx
counterpart: `BD_VALUE = size_t` (`bitreader.h:31`).

**What libvpx does:** `BD_VALUE` is `size_t`, so the reader's accumulator
is 32-bit on 32-bit targets and 64-bit on 64-bit targets. `vpx_reader_fill`
adjusts refill cadence accordingly.

**What rvp9 does:** Hardcode `BD_VALUE = u64`. 32-bit targets are out of
scope.

**Why:** PLAN.md §2 success criteria target only x86_64 + aarch64
(64-bit); 32-bit performance is not relevant. Hardcoding eliminates a
generic parameter and a branch in `fill`.

**Byte-parity impact:** None. The two paths produce identical decoded
streams (proof: the refill differs only in how many bytes are pulled per
fill; the resulting `(value, range, count)` post-fill state is the
same after the next `read` triggers a refill).

**Compression-efficiency impact:** None.

**Future plan:** Revisit if v2 needs 32-bit support; trivial change.

### 7.2 Decryption callback fields not ported

**Where:** `rvp9_core::entropy::BoolReader` omits libvpx's
`decrypt_cb` / `decrypt_state` / `clear_buffer` fields
(`bitreader.h:46-48`).

**What libvpx does:** Provides an in-place decryption hook used by
private VP9 fork integrations.

**What rvp9 does:** No decryption support. The reader operates directly
on the input slice.

**Why:** The hook is not exercised by any public VP9 test vector or
production WebRTC pipeline; A4 §2.2 notes it is libvpx-internal.

**Byte-parity impact:** None for non-encrypted streams (i.e., every
public test vector).

**Compression-efficiency impact:** None.

**Future plan:** Out of scope.

### 7.3 BoolWriter panic-mid-write is not poison-safe

**Where:** `rvp9_core::entropy::BoolWriter`. libvpx counterpart: same
behaviour — `vpx_writer` has no poison-on-panic logic because C has no
unwinding model.

**What libvpx does:** Panic is undefined (`longjmp`-driven).

**What rvp9 does:** Production `panic = "abort"` short-circuits the
issue. Under `panic = "unwind"` (debug), a panic mid-write leaves the
borrowed `&mut [u8]` in an indeterminate state.

**Why:** Adding poison-on-panic would add a `Drop` impl plus a bool
field; the borrowed slice's contents would still be invalid as a
bitstream regardless. Honest documentation > false safety.

**Byte-parity impact:** None (no panic in conformant operation).

**Compression-efficiency impact:** None.

**Future plan:** Revisit if Stage 4a observes panic conversions to
recoverable errors.

### 7.4 Counts-baseline double-count quirk preserved (per HLD-threading §5.6)

**Where:** `rvp9_core::entropy::accumulate_frame_counts`. libvpx
counterpart: `vp9_accumulate_frame_counts` (referenced from
`vp9_ethread.c:243-247`).

**What libvpx does:** Per-worker `td->counts` starts at `cm->counts`
baseline (memcpy at frame start) and is added back via
`cm->counts[i] += td->counts[i]`. This double-counts the baseline by
`(N - 1)` workers; the inline worker N-1 aliases `cm->counts` and
contributes its delta directly.

**What rvp9 does:** Mirror the exact arithmetic. Per-worker boxed
`FrameCounts` starts at baseline; the inline worker borrows
`&mut cm.counts` directly; main-thread accumulation sums each non-inline
worker's full `FrameCounts` into `cm.counts`.

**Why:** Byte parity matters for differential testing; the bias is
small (statistics drive backward adaptation, not the wire data
directly) but consistent across both libvpx and rvp9.

**Byte-parity impact:** None when matched; the wire stream is identical.

**Compression-efficiency impact:** Identical to libvpx — same backward
adaptation curve.

**Future plan:** A correctness-first v2 could subtract `(N - 1) *
baseline` before fold; track as a v2 cleanup.

### 7.5 Bitstream-pack scalar in v1 (L-35 locked)

**Where:** All of `rvp9_core::entropy`. libvpx counterpart: same
(libvpx has no SIMD for the boolean coder).

**What libvpx does:** Scalar boolean coder; multi-tile pack is concurrent
across tiles but each tile's writer is scalar.

**What rvp9 does:** Same. Multi-tile concurrency lives in
`tile-threading.md`; the per-tile `BoolWriter` is scalar.

**Why:** L-35 RESOLVED — no win identified for SIMD bitstream packing.

**Byte-parity impact:** None.

**Compression-efficiency impact:** None.

**Future plan:** Revisit only if Stage 5 perf gates indicate pack is on
the critical path.

### 7.6 `LOTS_OF_BITS` and EOF semantics preserved

**Where:** `rvp9_core::entropy::BoolReader::fill`. Behaviour mirrors
`bitreader.c:38` exactly: once past end, subsequent reads return `false`
deterministically.

This is not strictly a divergence (the wire-level behaviour is
identical) but documented because Rust idioms might tempt a `Result`-
returning fill. We do not; matching libvpx silent-zero-pad behaviour is
necessary for cross-decoder bit parity on truncated streams.

---

## 8. Open questions

These are decisions that surfaced during LLD writing and need
resolution before Stage 3 implementation begins. None are blocking the
LLD; all have a defensible default that this doc states explicitly.

### 8.1 Inlining strategy for `BoolWriter::write` (Q4.2 from A4 §7)

`vpx_write` is the single hottest function in real-time encode; call
counts run into the hundreds of millions per second of video. libvpx
makes it `static INLINE` in a header. Options:

(A) `#[inline]` on a `pub(crate)` fn — lets the optimizer decide.
(B) `#[inline(always)]` — forces inlining.
(C) Macro form — guarantees inlining and avoids closure-style cost.

**Default chosen for the LLD:** (A) `#[inline]` initially; the Stage 3
LLD bench measures end-to-end frame-encode time with (B) and falls back
to (C) only if (B) regresses (which would be surprising). Action: open
issue tracked under L-XX (this doc) once Stage 3 starts.

### 8.2 Probability-update damping during tentative pack (L-27 from A4 §7 #9)

libvpx runs `vp9_pack_bitstream` once as a tentative pack to size the
compressed-header literal (16-bit field), then again for real. During
the tentative pack, `cond_prob_diff_update` mutates `*oldp` *in place*.
If the tentative pack's mutation leaks back into the real pack, the
real pack double-applies updates — undefined behavior in the spec.

**Action:** Stage 3 LLD writers (bitstream.md owner) must verify that
libvpx's tentative pack either (a) clones `FrameContext` before pack or
(b) restores it after the tentative pass. Cite the libvpx file:line that
makes this safe; document in the divergence ledger if our port chooses
a different mechanism. This LLD recommends cloning `FrameContext` for
the tentative pack — small (≈ a few KB) and unambiguous.

### 8.3 Storage location for `vp9_pareto8_full` (cross-LLD concern)

`vp9_pareto8_full[255][8]` is 2040 bytes and consumed by both the
encoder (per-coef tree write) and the decoder (per-coef tree read).
Living in `rvp9-core::entropy::defaults` is right for L-09; the question
is whether to colocate it with `VP9_COEFBAND_TRANS_*` or with the
coefficient-token tables (which live in the encoder's tokenizer per the
scoping note in §1). **Default:** colocate in `rvp9-core::entropy::
defaults` so the decoder can use it without taking a dependency on the
encoder's tokenizer.

### 8.4 Compressed-header size limit at write time (Q4.9 from A4 §7)

The compressed-header size is a 16-bit big-endian field in the
uncompressed header. `vp9_bitstream.c:1441-1443` hard-errors if the
compressed header exceeds `UINT16_MAX`. **Action:** `bitstream.md`
must propagate `EntropyError::CompressedHeaderTooLarge` cleanly back to
the caller (not panic). This LLD just provides the error variant.

---

## 9. Test strategy

### 9.1 Unit tests (in `crates/rvp9-core/src/entropy/`)

**Boolean coder round-trip.** Property tests via `proptest`: generate
random `(bit, prob)` streams (probs in `[1, 255]`, bits arbitrary, up to
1 MB of bits), encode through `BoolWriter`, decode through `BoolReader`,
assert recovered stream equals input. Cover edge cases:

- Empty stream (just init + stop)
- Single-bit stream
- Long runs of one bit value (stresses carry propagation)
- Streams that force the writer into the carry-propagation branch
  (specifically: alternating runs of `prob == 1` and `prob == 255`)
- Streams that hit `(buffer[pos-1] & 0xe0) == 0xc0` at stop time
  (forces the appended zero byte)
- Writer buffer-overflow path: `BoolWriter::stop()` returns
  `Err(WriterOverflow)` when the destination slice is undersized.
- Reader past-EOF: feed a short buffer; `read()` returns deterministic
  `false` once past end; `had_error()` becomes true.

**`VPX_NORM` table.** Compile-time `const fn` derives `VPX_NORM[r] = 7
- floor(log2(r))` for `r >= 1`, `VPX_NORM[0] = 0`, and asserts via a
`const _` that the runtime table matches. Catches mistranscription.

**Tree write/read round-trip.** For each of the 11 trees in §4.5,
generate random leaf values uniformly, write each via
`BoolWriter::write_tree`, then read back via `BoolReader::read_tree`,
assert equality. Covers the asymmetric loop-exit shapes (write-by-len
vs read-until-leaf).

**Cost framework.** For each `prob in [1, 255]`:

- `cost_zero(prob) + cost_one(prob)` is the expected entropy in
  cost-units (verify against `-log2(prob/256) * 512 + -log2((256 -
  prob)/256) * 512`, within ±1 ULP due to rounding).
- `cost_bit(prob, false) == cost_zero(prob)`,
  `cost_bit(prob, true) == cost_one(prob)`.
- Differential against libvpx: dump libvpx's `vp9_prob_cost[]` (built
  binary), compare to `VP9_PROB_COST[]`. Must be bit-identical.

**`merge_probs` and `mode_mv_merge_probs`.** For each
`(pre, ct[0], ct[1])` over a deterministic sweep (pre in {0, 1, 128,
254, 255}; counts in {0..200}), assert output matches a Python
reference of the exact arithmetic.

**`adapt_*_probs` round-trip.** Pre-seed `FrameContext` from
`DEFAULT_*` statics, inject a known `FrameCounts`, run
`adapt_mode_probs` / `adapt_coef_probs` / `adapt_mv_probs`, compare to
a libvpx-derived golden file (recorded once per Stage 3, regenerated on
change with explicit review).

### 9.2 Differential tests (against libvpx)

The most important byte-parity gate is on the boolean coder primitives.
Per A4 §7 Q4.1: we run a hard byte-identical gate on the coder *itself*,
because if the coder is wrong nothing downstream can be trusted.

**Coder differential harness.** A `tests/differential/entropy.rs`
script that:

1. Wraps libvpx's `vpx_writer` via a tiny C shim built from
   `third_party/libvpx/`.
2. For each of N=10,000 randomized `(bits, probs)` streams (seeded
   deterministically), encodes through both libvpx and our
   `BoolWriter`; asserts byte-identical output.
3. Decodes through libvpx and our `BoolReader`; asserts identical bits
   recovered.

Edge cases generated explicitly:

- 0xff carry-propagation triggers
- Marker-bit-only stream
- Stop-flush byte-pad trigger (`0xc0`)
- Long runs of `prob == 1` / `prob == 255`
- 1-byte input buffer (smallest legal stream)

**`adapt_*_probs` differential.** Encode a corpus of synthetic
sequences (random partition / mode / coef streams) through libvpx and
rvp9 separately; dump `cm->fc` after each frame; assert byte-identical.

### 9.3 Conformance tests (Stage 3 exit criteria)

Indirect: every VP9 conformance vector exercises the boolean reader,
`FrameContext` parsing, and `adapt_*` paths. The Stage 3 exit gate
(`100% pass on official VP9 conformance test vectors`) covers the
decoder side of this LLD comprehensively. Stage 4a's diff-bitstream
harness covers the encoder side.

### 9.4 Property tests

`proptest` strategies:

- Generate random `[u8]` of length 0..16K, treat as `BoolReader` input,
  ensure no panic, ensure `had_error` is correctly set at end.
- Generate random `FrameCounts` (small random `u32`s), ensure
  `adapt_*_probs` produces probabilities in `[1, 255]` (never zero).
- Cost monotonicity: `cost_zero(p)` is non-increasing in `p`;
  `cost_one(p)` is non-decreasing. Trivially derivable from
  `VP9_PROB_COST`'s monotonic shape; sanity check the transcribed table.

### 9.5 Static-table tests

Compile-time-checked sums and counts:

- `VP9_PROB_COST.len() == 256`
- `VPX_NORM.len() == 256`
- `VP9_PARETO8_FULL.len() == 255` (each row 8 entries)
- `DEFAULT_COEF_PROBS_*` size matches the `CoefProbsModel` shape.

Recorded via `static_assertions::const_assert_eq!`.

### 9.6 Miri runs

The boolean coder uses no `unsafe` in the hot path (per §6 — purely
scalar). Miri is run as part of CI on all `rvp9-core::entropy` tests to
catch any future regressions; today there's no `unsafe` code to flag.

### 9.7 Bit-exact contract gate

Per A4 §4.9 the ten contract items are made explicit unit tests:

1. **Probability quantization.** Test `get_binary_prob` against the
   libvpx `(num * 256 + (den >> 1)) / den` formula with `u64` arithmetic
   over a broad sweep. Verify clipping to `[1, 255]`.
2. **Split formulas.** Encoder/decoder pair test: for every
   `(range, prob)` in `[128, 255] × [1, 255]`, assert encoder `split ==`
   decoder `split` (the §3.6 identity proof becomes a runtime test).
3. **`VPX_NORM` table.** Already covered by §9.1.
4. **Carry-propagation fixup.** Stress test specifically targets
   0xff-byte runs.
5. **Stop-encode tail.** Tested in §9.1.
6. **Merge formulas.** Tested in §9.1.
7. **Default tables.** Compile-time + golden-file checked.
8. **Scan orders, iscans, neighbor tables.** Owned by `transforms.md`;
   cross-referenced here.
9. **Pareto8 row index.** `vp9_pareto8_full[context_tree[PIVOT_NODE] -
   1]` — assert `context_tree[PIVOT_NODE] >= 1` in debug builds (it
   would index `vp9_pareto8_full[-1]` otherwise; libvpx implicitly
   relies on the prob never being zero).
10. **Token encoding.** Owned by the tokenizer (encoder side); cross-
    referenced.

All ten gates are wired into `cargo test --workspace -p rvp9-core
--features bit-exact-tests`. CI runs this feature on every PR.

---

## Summary

LLD line count: ~1000 lines (target 800-1500). Key §7 entries feeding
`notes/divergence-ledger.md`: §7.1 (BoolReader hardcoded 64-bit), §7.2
(decryption fields omitted), §7.3 (BoolWriter panic semantics), §7.4
(counts-baseline double-count quirk preserved), §7.5 (scalar-only pack
per L-35), §7.6 (EOF silent-zero-pad preserved). §8 blocking questions:
none are strict blockers; §8.2 (probability-update damping during
tentative pack — verify libvpx's mechanism in Stage 3 prep) is the most
important to nail down before `bitstream.md` implementation.
