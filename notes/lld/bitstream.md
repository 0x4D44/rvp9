# LLD: bitstream

**Status:** draft v0.1.0, Stage 2 per-module LLD.
**Author:** LLD-bitstream (Stage 2).
**Date:** 2026-05-16.
**Scope locked to:** D-01 (8-bit only), D-05 (AQ mode 3 only),
D-13 (CBR + VBR-RT, NO altref), R-06 (EXT_RATECTRL in v1).

This LLD specifies the **bitstream layout** subsystem: the VP9 frame
container (uncompressed header + compressed header + tile data) and
the superframe index that frames it externally. Companion LLDs
(`entropy.md`, `transforms.md`) own the boolean coder primitives,
probability tables, scan orders, and coefficient model that this LLD
*invokes*. The carve is:

- **bitstream.md** (this doc) — header layout, tile-data layout,
  superframe index, byte-level assembly, per-tile output buffers,
  `BitstreamWriter` streaming API, `show_existing_frame` short-path,
  bit-exact contract at the byte-stream level.
- **entropy.md** — `BoolWriter`, `BoolReader`, tree writer/reader,
  `FRAME_CONTEXT`, `FRAME_COUNTS`, default probability tables, backward
  adaptation, sub-exponential prob-update codec, `vp9_prob_cost[256]`.

Where bitstream.md and entropy.md touch, this LLD calls into a
typed `BoolWriter`/`BoolReader` surface defined in `entropy.md` §3.
The boolean-coder state machine itself is **not** redocumented here.

---

## 0. Document map

- [§1 Purpose](#1-purpose)
- [§2 libvpx origin](#2-libvpx-origin)
- [§3 Public API](#3-public-api)
- [§4 Types](#4-types)
- [§5 Internal algorithm](#5-internal-algorithm)
- [§6 SIMD dispatch](#6-simd-dispatch)
- [§7 Deliberate divergences from libvpx](#7-deliberate-divergences-from-libvpx)
- [§8 Open questions](#8-open-questions)
- [§9 Test strategy](#9-test-strategy)

---

## 1. Purpose

The bitstream subsystem owns **byte-level frame and superframe
assembly**: given an encoded frame's already-decided state (frame
type, sizes, refresh mask, segmentation map, loop filter parameters,
quant indices, partition decisions, per-block mode/MV/ref decisions,
per-coefficient tokens, per-frame entropy-prob deltas) it produces a
contiguous byte buffer that is a valid VP9 frame, and assembles those
frame buffers into a VP9 *superframe* (with an optional trailing
index) for delivery as one packet. On the decode side it parses that
buffer in reverse: locates the superframe index, splits into frame
buffers, parses the uncompressed header (raw bits), parses the
compressed header (range-coded prob updates), then hands each tile to
the entropy decoder + reconstruction pipeline.

It does **not** own:

- The boolean coder state machine (`entropy.md` §3-§4).
- Probability tables or backward adaptation (`entropy.md` §5-§6).
- Mode/MV/partition/coefficient *encoding decisions* (encoder LLDs).
- Block parsing past the boolean-coder boundary (decoder LLDs).
- Tile-thread orchestration (HLD-threading §3, `tile-threading.md`).

It **does** own the byte-level invariants that make decoder and
encoder bit-compatible: the 16-bit `compressed_hdr_size` literal
placement and back-fill, the per-tile 4-byte big-endian size prefix,
the superframe marker-byte layout, and the
`show_existing_frame` short-circuit path.

Per HLD-arch §1.1 / §2.1, the bitstream subsystem lives in
**`rvp9-core::bitstream`**. The encoder-side high-level wrapper
(`pack_frame`, `assemble_superframe`) lives in
**`rvp9-encoder::bitstream_writer`** and depends on `rvp9-core::bitstream`
plus on the per-tile token streams produced by mode-decision.

---

## 2. libvpx origin

Primary specialist: A4 (`notes/vp9-analysis/04-bitstream-entropy.md`).
Secondary: A1 (`01-architecture.md` §3.4 step 49 — placement of
`vp9_pack_bitstream` in the frame loop).

All citations are into `third_party/libvpx/` pinned at v1.14.1.

### 2.1 Primary source files

| Concern | File | Lines |
|---|---|---|
| Top-level packer | `vp9/encoder/vp9_bitstream.c` | 1396-1451 (`vp9_pack_bitstream`) |
| Uncompressed header (encode) | `vp9/encoder/vp9_bitstream.c` | 1243-1314 (`write_uncompressed_header`) |
| Compressed header (encode) | `vp9/encoder/vp9_bitstream.c` | 1316-1394 (`write_compressed_header`) |
| Tile data (encode, scalar) | `vp9/encoder/vp9_bitstream.c` | 1092-1145 (`encode_tiles`) |
| Tile data (encode, MT) | `vp9/encoder/vp9_bitstream.c` | 991-1090 (`encode_tiles_mt`) |
| Tile worker entry | `vp9/encoder/vp9_bitstream.c` | 942-952 (`encode_tile_worker`) |
| Per-tile worker buffer alloc | `vp9/encoder/vp9_bitstream.c` | 975-989 (`encode_tiles_buffer_alloc`) |
| Tile info encode | `vp9/encoder/vp9_bitstream.c` | 886-900 (`write_tile_info`) |
| Profile encode | `vp9/encoder/vp9_bitstream.c` | 1206-1218 (`write_profile`) |
| Frame-size encode (intra+kf path) | `vp9/encoder/vp9_bitstream.c` | 1147-1205 (`write_render_size`, `write_frame_size`, `write_frame_size_with_refs`) |
| Loop-filter header | `vp9/encoder/vp9_bitstream.c` | 701-739 (`encode_loopfilter`) |
| Quantization header | `vp9/encoder/vp9_bitstream.c` | 751-757 (`encode_quantization`) |
| Segmentation header | `vp9/encoder/vp9_bitstream.c` | 759-816 (`encode_segmentation`) |
| Interp-filter encode | `vp9/encoder/vp9_bitstream.c` | (see `write_interp_filter`, around 880) |
| Raw-bit writer | `vpx_dsp/bitwriter_buffer.{h,c}` | header `:20-31`; impl `:35-51` |
| Raw-bit reader | `vpx_dsp/bitreader_buffer.{h,c}` | header `:24-31`; impl `:17-29` |
| Sync codes + frame marker | `vp9/common/vp9_common.h` | 49-53 |
| Profile/refs/contexts constants | `vp9/common/vp9_onyxc_int.h` | 37-45 |
| Superframe index (encode) | `vp9/vp9_cx_iface.c` | 1216-1271 (`write_superframe_index`) |
| Superframe index (decode) | `vp9/vp9_decoder.c` | 527-585 (parse path) |

### 2.2 Companion-LLD touchpoints (citations only — algorithm lives there)

| Concern | Companion LLD | Stage 1 anchor |
|---|---|---|
| Boolean coder state machine | `entropy.md` §3-§4 | A4 §3.5, §3.6, §4.1, §4.2 |
| Compressed-header prob updates | `entropy.md` §5 | A4 §3.3 |
| Coefficient-token serialization | `entropy.md` §6 (`pack_mb_tokens`) | A4 §3.9, §4.4 |
| Sub-exponential update codec | `entropy.md` §7 | A4 §4.5 |
| MV prob updates + per-block MV coding | `entropy.md` §8 | A4 §4.6, §4.7 |
| Default probability tables | `entropy.md` §10 | A4 §5 |
| Scan / iscan / neighbor tables | `entropy.md` §11 | A4 §5.10 |
| `vp9_prob_cost[256]` cost LUT | `entropy.md` §12 (D-14) | A4 §5.11.0 |

### 2.3 HBD code lines (deferred to v2)

The following branches of `write_uncompressed_header` are HBD-only
(`#if CONFIG_VP9_HIGHBITDEPTH` or `profile > 0` gates) and **not**
implemented in v1 (D-01):

- `vp9_bitstream.c:1277-1279` — `write_bitdepth_colorspace_sampling`
  call in `intra_only` path under `cm->profile > PROFILE_0`. v2.
- `vp9_bitstream.c` `write_bitdepth_colorspace_sampling` body — the
  `profile >= 2` 1-bit `bit_depth` selector and the 2-bit color-space
  literal beyond 8-bit. v2.
- `vp9_bitstream.c` `write_profile` for `PROFILE_3` (3-bit form). v2
  (we still ship `write_profile` for `PROFILE_0`, which is 2 bits
  `0b00`).

The Rust port returns `Error::UnsupportedProfile { profile }` at
parse time and `EncodeError::UnsupportedConfiguration` at construct
time for any non-Profile-0 input (HLD-arch §7.3).

---

## 3. Public API

All signatures use the integer-width and pixel-type policy from
HLD-arch §6-§7. No `TBD` markers without an explicit Arthur ask.

### 3.1 `rvp9-core::bitstream` — module-level

This module exposes the byte-level building blocks. It is consumed
by both the decoder and the encoder. Boolean-coder types live in
`rvp9-core::entropy` (see `entropy.md` §3).

```rust
//! `rvp9_core::bitstream` — byte-level VP9 bitstream assembly.
//!
//! Exposes:
//! - Raw-bit reader/writer used by uncompressed headers.
//! - Superframe index parser + builder.
//! - High-level frame-header descriptor types (FrameHeader, TileInfo).
//! - Streaming `BitstreamWriter` used by the encoder.
//! - Streaming `BitstreamReader` used by the decoder.
//!
//! Boolean (arithmetic) coding lives in `rvp9_core::entropy`.

pub use crate::entropy::{BoolReader, BoolWriter};

pub mod raw;          // RawBitWriter / RawBitReader (uncompressed header)
pub mod header;       // FrameHeader, write/parse_uncompressed_header
pub mod tile_buffer;  // TileBuffer, per-tile output staging
pub mod superframe;   // SuperframeBuilder, SuperframeIndex parser
pub mod writer;       // BitstreamWriter (encoder top-level)
pub mod reader;       // BitstreamReader (decoder top-level)
```

### 3.2 Raw-bit writer/reader (uncompressed header)

`raw.rs` mirrors `vpx_dsp/bitwriter_buffer.{h,c}` and
`vpx_dsp/bitreader_buffer.{h,c}`.

```rust
/// MSB-first raw bit writer over a borrowed byte slice.
///
/// Mirrors libvpx `struct vpx_write_bit_buffer`
/// (`vpx_dsp/bitwriter_buffer.h:20-31`). The buffer is **never owned**;
/// it is a `&mut [u8]` slice supplied by the caller. On overflow the
/// writer becomes poisoned (`error == true`) and further writes are
/// silently dropped, matching libvpx semantics. Use [`Self::finish`]
/// to convert the poisoned state into a `Result`.
pub struct RawBitWriter<'a> {
    buffer: &'a mut [u8],
    bit_offset: usize,
    error: bool,
}

impl<'a> RawBitWriter<'a> {
    /// Constructs a writer at bit offset 0.
    /// Mirrors `vpx_wb_init` (`bitwriter_buffer.c`).
    pub fn new(buffer: &'a mut [u8]) -> Self;

    /// Writes a single bit (LSB of `bit`) at the current MSB-first
    /// position. On overflow, sets `error` and silently drops the bit.
    /// Mirrors `vpx_wb_write_bit` (`bitwriter_buffer.c:35-51`).
    pub fn write_bit(&mut self, bit: u8);

    /// Writes the low `bits` bits of `data`, MSB-first.
    /// Mirrors `vpx_wb_write_literal` (`bitwriter_buffer.c:53-58`).
    ///
    /// # Panics (debug only)
    /// Panics if `bits > 32`. Release builds clamp.
    pub fn write_literal(&mut self, data: u32, bits: u8);

    /// Writes an inverse-uniform code matching libvpx
    /// `vpx_wb_write_inv_signed_literal`. Used by segmentation deltas
    /// and loopfilter deltas in the uncompressed header.
    pub fn write_inv_signed_literal(&mut self, data: i32, bits: u8);

    /// Returns the bit offset of a future write; used by the
    /// `compressed_hdr_size` back-patch (see §5.1).
    pub fn bit_offset(&self) -> usize { self.bit_offset }

    /// Returns `(byte_count, error)` after rounding the cursor to the
    /// next byte boundary. Mirrors `vpx_wb_bytes_written`.
    pub fn bytes_written(&self) -> usize;

    /// Yields the borrowed buffer back to the caller. Returns
    /// `Err(BitstreamError::BufferFull)` if any write overflowed.
    pub fn finish(self) -> Result<usize, BitstreamError>;
}

/// MSB-first raw bit reader. Mirrors `vpx_dsp/bitreader_buffer.{h,c}`.
/// Errors are reported through `Result` rather than libvpx's callback.
pub struct RawBitReader<'a> {
    buffer: &'a [u8],
    bit_offset: usize,
}

impl<'a> RawBitReader<'a> {
    pub fn new(buffer: &'a [u8]) -> Self;
    pub fn read_bit(&mut self) -> Result<u8, BitstreamError>;
    pub fn read_literal(&mut self, bits: u8) -> Result<u32, BitstreamError>;
    pub fn read_inv_signed_literal(&mut self, bits: u8) -> Result<i32, BitstreamError>;
    pub fn bit_offset(&self) -> usize;
    pub fn bytes_consumed(&self) -> usize;
}
```

**Notes.** libvpx uses callback-based error reporting on the reader
(`vpx_rb_error_handler`, `bitreader_buffer.h:22`) so it can call back
into the codec's `setjmp` jump-buffer machinery. We replace this with
`Result` (HLD-arch §4.1, D-09 RESOLVED).

### 3.3 Frame-header types (encoder + decoder)

```rust
/// Profile (D-01: only `Profile0` is implemented in v1).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Profile {
    Profile0 = 0,
    // Profile1, Profile2, Profile3 reserved for v2 (HBD/4:4:4).
}

/// One of the four FRAME_CONTEXTS slots (2-bit field).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FrameContextIdx(pub u8); // 0..=3

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FrameType { Key = 0, Inter = 1 }

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum InterpFilter {
    Eighttap        = 0,
    EighttapSmooth  = 1,
    EighttapSharp   = 2,
    Bilinear        = 3, // not signaled when SWITCHABLE
    Switchable      = 4, // sentinel meaning "switchable" in header
}

/// `vp9_bitstream.c:886-900` (`write_tile_info`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TileInfo {
    /// `log2_tile_cols` in `[min_log2_tile_cols, max_log2_tile_cols]`;
    /// see `vp9_common::vp9_get_tile_n_bits` for the bounds.
    pub log2_cols: u8,
    /// `log2_tile_rows` in `[0, 2]`. L-62: always 0 in real-time v1.
    pub log2_rows: u8,
}

/// Snapshot consumed by `write_uncompressed_header` and produced by
/// `parse_uncompressed_header`. Field names mirror libvpx
/// `VP9_COMMON`; rationale-by-field appears in §5.1.
#[derive(Clone, Debug)]
pub struct FrameHeader {
    pub profile:                  Profile,
    pub show_existing_frame:      Option<u8>, // Some(alt_fb_idx in 0..8) → short-circuit
    pub frame_type:               FrameType,
    pub show_frame:               bool,
    pub error_resilient_mode:     bool,
    pub intra_only:               bool,
    pub reset_frame_context:      u8,           // 0..=3; 2 bits
    pub width:                    u16,
    pub height:                   u16,
    pub render_width:             u16,          // == width when no scaling
    pub render_height:            u16,
    /// 8-bit `refresh_frame_flags`. For inter frames also encodes
    /// which of the 8 ref slots will be overwritten.
    pub refresh_frame_flags:      u8,
    /// Three reference slot indices, one per (LAST, GOLDEN, ALTREF).
    pub ref_frame_idx:            [u8; 3],      // 3 bits each
    pub ref_frame_sign_bias:      [bool; 3],
    pub allow_high_precision_mv:  bool,
    pub interp_filter:            InterpFilter,
    pub refresh_frame_context:    bool,
    pub frame_parallel_decoding_mode: bool,
    pub frame_context_idx:        FrameContextIdx,
    pub loop_filter:              LoopFilterHeader,
    pub quantization:             QuantizationHeader,
    pub segmentation:             SegmentationHeader,
    pub tile_info:                TileInfo,
}

/// `vp9_bitstream.c:701-739`. Subset of `struct loopfilter`.
#[derive(Clone, Debug)]
pub struct LoopFilterHeader {
    pub filter_level:      u8,   // 0..=63
    pub sharpness_level:   u8,   // 0..=7
    pub mode_ref_delta_enabled: bool,
    pub mode_ref_delta_update:  bool,
    pub ref_deltas:        [i8; 4],         // MAX_REF_LF_DELTAS = 4
    pub mode_deltas:       [i8; 2],         // MAX_MODE_LF_DELTAS = 2
    pub ref_delta_update_flags:  u8,         // 4 bits packed
    pub mode_delta_update_flags: u8,         // 2 bits packed
}

/// `vp9_bitstream.c:751-757`.
#[derive(Clone, Debug)]
pub struct QuantizationHeader {
    pub base_qindex:    u8,    // 8 bits, 0..=255 (QINDEX_BITS / MAXQ)
    pub y_dc_delta:     i8,    // 0 if absent (presence is a 1-bit flag)
    pub uv_dc_delta:    i8,
    pub uv_ac_delta:    i8,
}

/// `vp9_bitstream.c:759-816`.
#[derive(Clone, Debug)]
pub struct SegmentationHeader {
    pub enabled:        bool,
    pub update_map:     bool,
    pub temporal_update: bool,
    /// `SEG_TREE_PROBS = 7` (`vp9_seg_common.h:24`). Only emitted
    /// when `update_map`.
    pub tree_probs:     [u8; 7],
    /// `PREDICTION_PROBS = 3`. Only emitted with `temporal_update`.
    pub pred_probs:     [u8; 3],
    pub update_data:    bool,
    pub abs_delta:      bool,
    pub features:       [SegmentFeatures; 8], // MAX_SEGMENTS = 8
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SegmentFeatures {
    /// One per SEG_LVL_MAX = 4. SEG_FEATURE_DATA_MAX in libvpx.
    pub enabled: [bool; 4],
    pub data:    [i16; 4],
}
```

### 3.4 Uncompressed-header encode / decode

```rust
/// Writes the uncompressed header into the provided raw-bit writer.
/// Mirrors `write_uncompressed_header` (`vp9_bitstream.c:1243-1314`).
///
/// # Errors
/// Returns `BitstreamError::BufferFull` if the underlying writer
/// overflows. The writer's poison flag is the source of truth.
///
/// # Postconditions
/// On success, the writer cursor is byte-aligned per
/// `vpx_wb_bytes_written` rounding. Caller should round up before
/// any subsequent byte-aligned writes.
pub fn write_uncompressed_header(
    hdr: &FrameHeader,
    wb: &mut RawBitWriter<'_>,
) -> Result<(), BitstreamError>;

/// Parses the uncompressed header. Mirrors
/// `read_uncompressed_header` (decoder side; `vp9_decodeframe.c`).
///
/// # Errors
/// Returns `BitstreamError::SyncCode`, `UnsupportedProfile`,
/// `InvalidFrameSize` per the spec.
pub fn parse_uncompressed_header(
    rb: &mut RawBitReader<'_>,
) -> Result<FrameHeader, BitstreamError>;
```

### 3.5 Compressed-header encode

The compressed header is range-coded; it lives in `entropy.md` §5
for the body, but the top-level dispatch and the
`compressed_hdr_size` invariants belong here.

```rust
/// Writes the compressed header into `dest`. Returns the number of
/// bytes consumed. Mirrors `write_compressed_header`
/// (`vp9_bitstream.c:1316-1394`).
///
/// # Errors
/// `BitstreamError::CompressedHeaderTooLarge` if the produced header
/// would exceed `u16::MAX` (mirrors `vp9_bitstream.c:1441-1443`).
///
/// # Inputs
/// - `fc`: the current frame context (probability snapshot to update from)
/// - `counts`: per-frame token counts (driver of `vp9_cond_prob_diff_update`)
/// - `header`: the already-finalised frame header (drives intra-only gate)
/// - `tx_mode`: 0..=4 enum mirroring `TX_MODE`
/// - `reference_mode`: 0..=2 enum mirroring `REFERENCE_MODE`
/// - `interp_filter`: as in `header`
/// - `allow_high_precision_mv`: as in `header`
///
/// Probability adaptation itself (the `vp9_cond_prob_diff_update`
/// calls) is in `entropy.md` §7.
pub fn write_compressed_header(
    fc: &mut FrameContext,
    counts: &FrameCounts,
    header: &FrameHeader,
    tx_mode: TxMode,
    reference_mode: ReferenceMode,
    dest: &mut [u8],
) -> Result<usize, BitstreamError>;

/// Decoder mirror. Parses prob updates *into* `fc` and consumes
/// `len` bytes. Companion to `parse_uncompressed_header`.
pub fn parse_compressed_header(
    fc: &mut FrameContext,
    src: &[u8],
    len: usize,
    header: &FrameHeader,
) -> Result<(TxMode, ReferenceMode), BitstreamError>;
```

### 3.6 Tile-buffer staging

```rust
/// One per worker; matches the contract from
/// `VP9BitstreamWorkerData` (`vp9_bitstream.h:20-32`) with the libvpx
/// dest-pointer + dest-size protocol replaced by an owned `Vec<u8>`.
/// See L-13 (per-tile buffer ownership) for rationale.
pub struct TileBuffer {
    /// Output bytes for the boolean coder. Inner `BoolWriter` borrows
    /// `&mut buf` for the lifetime of `take_writer`.
    pub buf: Vec<u8>,
    /// Tile column index. `log2_tile_rows == 0` is asserted at start
    /// of frame per L-62.
    pub tile_col: u8,
    /// Maintained alongside the tile worker's per-block writes.
    pub max_mv_magnitude: u32,
    /// `interp_filter_selected[0][SWITCHABLE]` — see §4.4 below.
    pub interp_filter_selected: [u32; 4], // SWITCHABLE = 4 (vp9_filter.h:32)
}

impl TileBuffer {
    pub fn new() -> Self;

    /// Resets buffers between frames. Capacity is preserved.
    pub fn reset(&mut self);

    /// Hands out the `BoolWriter` for the tile worker's per-tile
    /// arithmetic coding pass. The writer borrows `self.buf`; the
    /// caller `drop`s it before calling `finalize`.
    pub fn writer(&mut self) -> BoolWriter<'_>;

    /// Returns the number of bytes the boolean coder produced.
    /// Equal to `BoolWriter::pos` at `vpx_stop_encode` time.
    pub fn pos(&self) -> usize;
}
```

### 3.7 Superframe assembly

```rust
/// Builder for the optional trailing superframe index.
/// Mirrors `write_superframe_index` (`vp9_cx_iface.c:1216-1271`).
///
/// In v1, multi-frame superframes appear for:
/// - SVC spatial layers (one frame per layer in one packet); and
/// - `show_existing_frame` paired with the frame that creates the
///   referenced buffer.
/// (D-13: no altref → no GF_GROUP altref slot driving superframe
/// composition; see §7.3 below.)
pub struct SuperframeBuilder {
    /// Concatenated frame bytes. Each push appends a frame; the
    /// index is appended at `finish` time.
    output: Vec<u8>,
    /// Per-frame byte counts. Capacity 8 (assert at push).
    frame_sizes: [u32; 8],
    frame_count: u8,
}

impl SuperframeBuilder {
    pub fn new() -> Self;
    pub fn with_capacity(bytes: usize) -> Self;

    /// Appends one frame's bytes. Asserts `frame_count < 8`
    /// (`vp9_cx_iface.c:1221-1222`).
    pub fn push_frame(&mut self, frame: &[u8]);

    /// Appends the trailing index per §5.4 below and returns the
    /// final concatenated buffer.
    pub fn finish(self) -> Vec<u8>;

    /// Returns the number of frames pushed so far.
    pub fn frame_count(&self) -> u8 { self.frame_count }
}

/// Decoder-side parser. Splits a packet into one or more frame slices.
///
/// `parse_superframe` returns frames in raster order with the index
/// stripped (if present).
pub fn parse_superframe(packet: &[u8]) -> Result<SmallVec<[&[u8]; 8]>, BitstreamError>;
```

(`SmallVec<[&[u8]; 8]>` is a workspace-shared inline-capacity vector;
the API contract is "up to 8 frames per superframe, no heap allocation
in the common case". The exact crate is L-LL-1 in §8 — `smallvec` is
the workspace default, but if a slimmer alternative exists at workspace
review time we adopt it.)

### 3.8 Top-level encoder/decoder entry points

```rust
/// Encoder entry point. Composes `write_uncompressed_header` +
/// `write_compressed_header` + concatenated tile data. Mirrors
/// `vp9_pack_bitstream` (`vp9_bitstream.c:1396-1451`).
///
/// `tiles` is in tile-raster order (row-major, `log2_tile_rows = 0`
/// in v1 so always a single row).
///
/// # Errors
/// - `BitstreamError::BufferFull` — `dest` too small.
/// - `BitstreamError::CompressedHeaderTooLarge` — see §3.5.
///
/// # Postconditions
/// `dest[..returned_len]` is one VP9 frame (without superframe index).
pub fn pack_frame(
    fc: &mut FrameContext,
    counts: &FrameCounts,
    header: &FrameHeader,
    tx_mode: TxMode,
    reference_mode: ReferenceMode,
    tiles: &[TileBuffer],
    dest: &mut [u8],
) -> Result<usize, BitstreamError>;

/// Decoder entry point — parses a single VP9 frame's headers and
/// computes the tile-data layout. Body of per-tile parsing is in
/// the decoder's tile-parse path (`rvp9-decoder::tile_parse`).
///
/// Returns `FrameSlices` containing the parsed `FrameHeader`, the
/// compressed-header byte range, and tile-byte ranges.
pub fn parse_frame_layout(
    frame: &[u8],
) -> Result<FrameSlices<'_>, BitstreamError>;

pub struct FrameSlices<'a> {
    pub header: FrameHeader,
    pub compressed_header: &'a [u8],
    pub tiles: SmallVec<[&'a [u8]; 32]>,   // log2_tile_cols ≤ 6
}
```

### 3.9 Error type

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BitstreamError {
    /// `RawBitWriter` overflowed its destination buffer.
    BufferFull,
    /// `write_compressed_header` produced more than `u16::MAX` bytes.
    /// Mirrors `vp9_bitstream.c:1441-1443` exactly.
    CompressedHeaderTooLarge { actual: usize },
    /// Frame-marker bits did not equal `VP9_FRAME_MARKER` (0b10).
    InvalidFrameMarker { observed: u8 },
    /// `write_sync_code` round-trip failed.
    InvalidSyncCode,
    /// Profile encoded as `b11` (Profile 3, HBD/4:4:4). v2.
    UnsupportedProfile { profile: u8 },
    /// Width/height outside `[1, 65535]` or zero-sized.
    InvalidFrameSize { width: u16, height: u16 },
    /// `show_existing_frame` referenced an empty ref slot.
    InvalidShowExistingFrame { slot: u8 },
    /// Truncated input.
    Truncated { needed: usize, available: usize },
    /// Superframe-index marker byte pattern recognised but malformed
    /// magnitude/count combination (e.g. `(mag+1) * frames > 32`).
    InvalidSuperframeIndex,
    /// Tile-size 4-byte big-endian prefix exceeded the remaining
    /// tile-data byte budget.
    TileSizePrefixOverflow,
}
```

`BitstreamError` is `Send + Sync + 'static`, `Clone`, and implements
`std::error::Error`. It is the lowest-level error type and is
`From`-convertible into both `EncodeError` and `DecodeError`
(HLD-arch §4.2).

---

## 4. Types

This section consolidates load-bearing types. Field-by-field
commentary cites the libvpx origin.

### 4.1 `RawBitWriter` (already shown in §3.2)

Layout:

```text
struct RawBitWriter<'a> {
    buffer:     &'a mut [u8],   // not owned
    bit_offset: usize,          // 0..=buffer.len() * 8
    error:      bool,           // poison once set
}
```

Mirrors `struct vpx_write_bit_buffer` (`bitwriter_buffer.h:20-31`).
Differences:

- libvpx stores `size: size_t` separately; we use `&mut [u8]` and
  `slice::len()` instead. Saves 8 bytes per writer.
- libvpx's `error: int` is a 32-bit flag; we use `bool` (1 byte).
  Semantically identical.
- No callback (libvpx is `error_handler` in the reader, not writer).

Marker traits: `Send`, not `Sync` (single-threaded; the lifetime
`'a` ties it to the borrowed slice).

### 4.2 `RawBitReader`

```text
struct RawBitReader<'a> {
    buffer:     &'a [u8],
    bit_offset: usize,
}
```

Mirrors `struct vpx_read_bit_buffer` (`bitreader_buffer.h:24-31`)
minus the `error_handler` callback, which is replaced by `Result`
propagation (HLD-arch §4.1).

### 4.3 `FrameHeader` (already shown in §3.3)

Cross-cutting notes:

- `show_existing_frame: Option<u8>`: libvpx packs this as `bit +
  3-bit literal`. The Rust port uses `Option` so the 8-bit alt_fb_idx
  is always meaningful when present. L-15.
- `refresh_frame_flags: u8`: 8 bits. For inter frames, this is the
  bitmask of which of the 8 reference slots are overwritten with the
  decoded current frame (`REF_FRAMES = 8`, `vp9_onyxc_int.h:39`).
  Derived from libvpx `vp9_get_refresh_mask` on the encoder side.
- `ref_frame_idx: [u8; 3]`: indices into the 8-slot ref pool, one
  for each of LAST/GOLDEN/ALTREF. 3 bits each (`REF_FRAMES_LOG2 = 3`).
- `tile_info: TileInfo`: log2-cols and log2-rows (§3.3). Real-time
  v1 always has `log2_rows = 0` (L-62); v1 ships an
  `assert!(header.tile_info.log2_rows == 0)` at `pack_frame` entry.
- `frame_context_idx: FrameContextIdx`: 2-bit field
  (`FRAME_CONTEXTS_LOG2 = 2`, `vp9_onyxc_int.h:44-45`).

### 4.4 `TileBuffer` (already shown in §3.6)

Per-tile output staging. Memory ownership belongs to the encoder
(allocated at `Encoder::new` per `cfg.max_threads`, pooled by tile
worker index). `Vec<u8>` is sized to `encode_tiles_buffer_alloc_size`
on first use — a conservative upper bound based on input frame
dimensions; libvpx uses
`(mb_cols * mb_rows * 8 * 16 + (32 << 20) / num_workers)`
(`vp9_bitstream.c:967-973`). We mirror this constant exactly for
byte parity of the allocation behaviour, but `Vec` will reallocate
if exceeded; the libvpx C version returns `output buffer full` and
that maps to `BitstreamError::BufferFull` on `finish`.

`interp_filter_selected[0][SWITCHABLE]` length: the libvpx field is
`int interp_filter_selected[1][SWITCHABLE]`
(`vp9_bitstream.h:25-26`). `SWITCHABLE = 4` (`vp9_filter.h:32`). The
comment in libvpx notes that only `[0][k]` is ever incremented; we
flatten to a 4-element array. After tile sync the main thread sums
per-worker arrays into `cpi->interp_filter_selected[0][k]`
(`vp9_bitstream.c:1062-1064`).

Marker traits: `Send`. Not `Sync` — each tile worker holds its own
`&mut TileBuffer` for the duration of the tile encode.

### 4.5 `SuperframeBuilder` (already shown in §3.7)

Bounded to 8 frames per libvpx invariant
`assert(ctx->pending_frame_count <= 8)` (`vp9_cx_iface.c:1222`).
`frame_sizes` is `[u32; 8]` because the libvpx `pending_frame_sizes`
field is `unsigned int`. Total index size:
`2 + (mag + 1) * frame_count` bytes (`vp9_cx_iface.c:1236`).

### 4.6 `FrameSlices<'_>` (already shown in §3.8)

The decoder counterpart to `pack_frame`'s output. Holds byte slices
borrowed from the input packet, parsed lazily by the consumer.

`tiles: SmallVec<[&'a [u8]; 32]>` — `log2_tile_cols ≤ 6` so
`tile_cols ≤ 64`; the inline capacity of 32 covers the common case
(real-time encodes use at most `tile_cols = 8` per `cfg.tile_columns`
bounded against frame dimension).

### 4.7 Constants exported by this module

Mirror the libvpx defines verbatim per HLD-arch §6 (integer-width
policy):

```rust
pub const VP9_FRAME_MARKER: u8 = 0x2;                // vp9_common.h:53
pub const VP9_SYNC_CODE: [u8; 3] = [0x49, 0x83, 0x42]; // vp9_common.h:49-51

pub const REF_FRAMES_LOG2: u8 = 3;                   // vp9_onyxc_int.h:38
pub const REF_FRAMES: u8 = 1 << REF_FRAMES_LOG2;     // = 8
pub const REFS_PER_FRAME: u8 = 3;                    // vp9_onyxc_int.h:37
pub const FRAME_CONTEXTS_LOG2: u8 = 2;               // vp9_onyxc_int.h:44
pub const FRAME_CONTEXTS: u8 = 1 << FRAME_CONTEXTS_LOG2; // = 4

pub const QINDEX_BITS: u8 = 8;                       // vp9_quant_common.h:24
pub const MAXQ: u8 = 255;
pub const MAX_LOOP_FILTER: u8 = 63;                  // vp9_loopfilter.h:24

pub const MAX_REF_LF_DELTAS: usize = 4;              // vp9_loopfilter.h:29
pub const MAX_MODE_LF_DELTAS: usize = 2;             // vp9_loopfilter.h:30

pub const MAX_SEGMENTS: usize = 8;                   // vp9_seg_common.h:22
pub const SEG_TREE_PROBS: usize = MAX_SEGMENTS - 1;  // = 7
pub const PREDICTION_PROBS: usize = 3;               // vp9_seg_common.h:26

pub const SUPERFRAME_MAX_FRAMES: usize = 8;          // implicit from vp9_cx_iface.c:1222
```

These live in `rvp9_core::bitstream::consts` and are re-exported at
the module root.

---

## 5. Internal algorithm

This section is pseudocode-level. Boolean-coder primitives are
already specified in `entropy.md` §3-§4 and not repeated here.
Every non-obvious step has a libvpx file:line citation.

### 5.1 Frame-level encode sequence (`pack_frame`)

Mirrors `vp9_pack_bitstream` (`vp9_bitstream.c:1396-1451`).

```text
pack_frame(fc, counts, header, tx_mode, reference_mode, tiles, dest):
    wb = RawBitWriter::new(dest)                                  // §3.2
    write_uncompressed_header(header, &mut wb)?                   // §5.2

    if header.show_existing_frame.is_some():
        // vp9_bitstream.c:1417-1422 — show_existing short-circuit
        n = wb.bytes_written()
        return Ok(n)

    // Reserve 16 bits for compressed_hdr_size; back-patch later.
    saved_bit_offset = wb.bit_offset()
    wb.write_literal(0, 16)                                       // :1426
    if wb.error: return Err(BufferFull)

    uncompressed_hdr_bytes = wb.bytes_written()                   // :1432
    drop(wb)                                                       // release &mut dest

    // From here on we slice dest after the uncompressed-header bytes
    // and pass to the compressed-header writer.
    let (uh_slice, rest) = dest.split_at_mut(uncompressed_hdr_bytes);
    let compressed_size = write_compressed_header(fc, counts, header,
                                                  tx_mode, reference_mode,
                                                  rest)?;          // §5.3

    if compressed_size > u16::MAX as usize:
        return Err(CompressedHeaderTooLarge { actual: compressed_size })

    // Back-patch the 16-bit literal.  Re-open the writer pointing
    // at uh_slice, seek to saved_bit_offset, write the literal.
    let mut wb = RawBitWriter::from_existing(uh_slice, saved_bit_offset);
    wb.write_literal(compressed_size as u32, 16);
    debug_assert!(!wb.error);

    // Tile data starts at uncompressed_hdr_bytes + compressed_size.
    let tile_start = uncompressed_hdr_bytes + compressed_size;
    let tile_bytes = encode_tile_data(&header.tile_info, tiles,
                                      &mut dest[tile_start..])?;    // §5.4
    Ok(tile_start + tile_bytes)
```

Notes:

- The "reserve 16 bits, back-patch later" pattern (`saved_wb` in
  libvpx, `vp9_bitstream.c:1424-1445`) is the **only** place
  bitstream encode requires a non-monotonic cursor. We mirror it
  exactly. The Rust port stores the saved bit offset, drops the
  writer, runs the compressed-header writer over the *next* slice,
  then opens a fresh `RawBitWriter` over the uncompressed-header
  bytes and seeks to the saved offset to write the literal. This
  avoids holding two `&mut [u8]` borrows simultaneously.

- `show_existing_frame` short-circuits the entire compressed header
  AND tile data (`vp9_bitstream.c:1417-1422`). Total wire form:
  uncompressed header (~3 bytes) only. L-15.

### 5.2 Uncompressed header (`write_uncompressed_header`)

Mirrors `vp9_bitstream.c:1243-1314`. The control flow:

```text
write_uncompressed_header(hdr, wb):
    wb.write_literal(VP9_FRAME_MARKER, 2)              // = 0b10 (:1248)
    write_profile(hdr.profile, wb)                     // :1250, body :1206-1218

    wb.write_bit(hdr.show_existing_frame.is_some())    // :1253
    if let Some(idx) = hdr.show_existing_frame:
        wb.write_literal(idx as u32, 3)                // :1255
        return Ok(())                                   // L-15

    wb.write_bit(hdr.frame_type as u8)                 // :1259
    wb.write_bit(hdr.show_frame as u8)                 // :1260
    wb.write_bit(hdr.error_resilient_mode as u8)       // :1261

    if hdr.frame_type == Key:
        write_sync_code(wb)                             // :1264; emits [0x49,0x83,0x42] as 24 bits
        write_bitdepth_colorspace_sampling(hdr, wb)     // :1265; **Profile 0 only in v1**
        write_frame_size(hdr, wb)                       // :1266
    else:
        if !hdr.show_frame:
            wb.write_bit(hdr.intra_only as u8)         // :1268
        if !hdr.error_resilient_mode:
            wb.write_literal(hdr.reset_frame_context as u32, 2)  // :1271

        if hdr.intra_only:
            write_sync_code(wb)                         // :1274
            // v1: profile is always Profile0, so the bitdepth/colorspace
            // block is skipped (:1277-1279, gated on profile > 0).
            wb.write_literal(hdr.refresh_frame_flags as u32, 8)   // :1281
            write_frame_size(hdr, wb)                   // :1282
        else:
            wb.write_literal(hdr.refresh_frame_flags as u32, 8)   // :1285
            for r in 0..3:                              // :1286-1291
                wb.write_literal(hdr.ref_frame_idx[r] as u32, REF_FRAMES_LOG2)
                wb.write_bit(hdr.ref_frame_sign_bias[r] as u8)
            write_frame_size_with_refs(hdr, wb)         // :1293
            wb.write_bit(hdr.allow_high_precision_mv as u8)       // :1295
            // fix_interp_filter (libvpx :1297) is a counts-driven adjustment
            // owned by the encoder LLD; this LLD takes the resolved value.
            write_interp_filter(hdr.interp_filter, wb)  // :1298

    if !hdr.error_resilient_mode:
        wb.write_bit(hdr.refresh_frame_context as u8)   // :1303
        wb.write_bit(hdr.frame_parallel_decoding_mode as u8)  // :1304

    wb.write_literal(hdr.frame_context_idx.0 as u32, FRAME_CONTEXTS_LOG2)  // :1307

    encode_loopfilter(&hdr.loop_filter, wb)             // :1309; §5.2.1
    encode_quantization(&hdr.quantization, wb)          // :1310; §5.2.2
    encode_segmentation(&hdr.segmentation, wb)          // :1311; §5.2.3
    write_tile_info(&hdr.tile_info, wb)                 // :1313; §5.2.4
```

**Sub-procedures:**

#### 5.2.1 `encode_loopfilter` (`vp9_bitstream.c:701-739`)

```text
encode_loopfilter(lf, wb):
    wb.write_literal(lf.filter_level as u32, 6)           // :706
    wb.write_literal(lf.sharpness_level as u32, 3)        // :707
    wb.write_bit(lf.mode_ref_delta_enabled as u8)         // :709
    if lf.mode_ref_delta_enabled:
        wb.write_bit(lf.mode_ref_delta_update as u8)
        if lf.mode_ref_delta_update:
            for i in 0..MAX_REF_LF_DELTAS:
                emit = (lf.ref_delta_update_flags >> i) & 1
                wb.write_bit(emit)
                if emit != 0:
                    wb.write_inv_signed_literal(lf.ref_deltas[i] as i32, 6)
            for i in 0..MAX_MODE_LF_DELTAS:
                emit = (lf.mode_delta_update_flags >> i) & 1
                wb.write_bit(emit)
                if emit != 0:
                    wb.write_inv_signed_literal(lf.mode_deltas[i] as i32, 6)
```

#### 5.2.2 `encode_quantization` (`vp9_bitstream.c:751-757`)

```text
encode_quantization(q, wb):
    wb.write_literal(q.base_qindex as u32, QINDEX_BITS)   // = 8
    write_delta_q(q.y_dc_delta, wb)                       // 1 bit present + 4 bits if !=0
    write_delta_q(q.uv_dc_delta, wb)
    write_delta_q(q.uv_ac_delta, wb)
```

where `write_delta_q(d, wb)` is `vp9_bitstream.c:741-749`:

```text
write_delta_q(d, wb):
    if d != 0:
        wb.write_bit(1)
        wb.write_inv_signed_literal(d as i32, 4)
    else:
        wb.write_bit(0)
```

#### 5.2.3 `encode_segmentation` (`vp9_bitstream.c:759-816`)

```text
encode_segmentation(seg, wb):
    wb.write_bit(seg.enabled as u8)
    if !seg.enabled: return
    wb.write_bit(seg.update_map as u8)
    if seg.update_map:
        for i in 0..SEG_TREE_PROBS:
            write_prob(seg.tree_probs[i], wb)         // §5.2.3.1
        wb.write_bit(seg.temporal_update as u8)
        if seg.temporal_update:
            for i in 0..PREDICTION_PROBS:
                write_prob(seg.pred_probs[i], wb)
    wb.write_bit(seg.update_data as u8)
    if seg.update_data:
        wb.write_bit(seg.abs_delta as u8)
        for s in 0..MAX_SEGMENTS:
            for f in 0..SEG_LVL_MAX:                  // = 4
                wb.write_bit(seg.features[s].enabled[f] as u8)
                if seg.features[s].enabled[f]:
                    bits = SEG_FEATURE_DATA_BITS[f]
                    write_signed_or_unsigned_per_feature(seg.features[s].data[f], wb, bits)
```

`write_prob(p, wb)` is "1 bit `present` + 8-bit literal":
`vp9_bitstream.c:78-87` in libvpx (`write_segment_pred_probs` and
`vp9_write_tree_probs`). The 8-bit literal is the *whole* probability
since per-segment probs are 8-bit.

#### 5.2.4 `write_tile_info` (`vp9_bitstream.c:886-900`)

```text
write_tile_info(tile, wb):
    // log2_tile_cols emitted as a unary code with terminator:
    //   emit (log2_cols - min_log2_cols) 1-bits
    //   if log2_cols < max_log2_cols: emit a 0-bit
    let (min_log2_cols, max_log2_cols) = vp9_get_tile_n_bits(mi_cols)
    let mut diff = tile.log2_cols - min_log2_cols
    while diff > 0:
        wb.write_bit(1)
        diff -= 1
    if tile.log2_cols < max_log2_cols:
        wb.write_bit(0)

    // log2_tile_rows: special encoding
    //   0 -> "0"  ; 1 -> "10" ; 2 -> "11"
    wb.write_bit((tile.log2_rows != 0) as u8)
    if tile.log2_rows != 0:
        wb.write_bit((tile.log2_rows != 1) as u8)
    // L-62: assert(tile.log2_rows == 0) at pack_frame entry in real-time v1.
```

`vp9_get_tile_n_bits` lives in `vp9_common::tile.c`; we re-implement
it under `rvp9_core::bitstream::tile_bounds` (about 20 LoC). Source:
loops `min_log2_cols` up while `(MAX_MI_COLS - 1) >> min_log2_cols >=
MIN_TILE_WIDTH_B64`, and `max_log2_cols` up while `(MAX_TILE_WIDTH_B64
<< max_log2_cols) < mi_cols`.

### 5.3 Compressed header (`write_compressed_header`)

Top-level structure (`vp9_bitstream.c:1316-1394`). Each numbered step
calls into `entropy.md` for the actual prob update math:

```text
write_compressed_header(fc, counts, header, tx_mode, reference_mode, dest):
    let mut bw = BoolWriter::start_encode(dest)              // entropy.md §3
    encode_txfm_probs(&mut fc.tx_probs, &counts.tx, tx_mode, &mut bw)
    update_coef_probs(&mut fc.coef_probs, &counts.coef, tx_mode, &mut bw)
    update_skip_probs(&mut fc.skip_probs, &counts.skip, &mut bw)

    if header.frame_type == Inter && !header.intra_only:
        update_inter_mode_probs(&mut fc, &counts, &mut bw)
        if header.interp_filter == Switchable:
            update_switchable_interp_probs(&mut fc, &counts, &mut bw)
        update_intra_inter_probs(&mut fc.intra_inter_prob, &counts.intra_inter, &mut bw)
        write_reference_mode_bits(reference_mode, &mut bw)   // 1 or 2 raw bits
        if reference_mode != SingleReference:
            update_comp_inter_probs(&mut fc, &counts, &mut bw)
        if reference_mode != CompoundReference:
            update_single_ref_probs(&mut fc, &counts, &mut bw)
        if reference_mode != SingleReference:
            update_comp_ref_probs(&mut fc, &counts, &mut bw)
        for bsg in 0..BLOCK_SIZE_GROUPS:
            update_y_mode_probs(&mut fc.y_mode_prob[bsg], &counts.y_mode[bsg], &mut bw)
        update_partition_probs(&mut fc.partition_prob, &counts.partition, &mut bw)
        write_nmv_probs(&mut fc.nmvc, &counts.mv, header.allow_high_precision_mv, &mut bw)

    bw.stop_encode()                                          // entropy.md §3
    return Ok(bw.pos())
```

All `update_*_probs` calls are `entropy.md` §7 (sub-exponential
prob-update encoding). The `update_coef_probs_common` body is
`entropy.md` §6. **This LLD only owns the order and gating.**

### 5.4 Tile data (`encode_tile_data`)

Mirrors `encode_tiles` (`vp9_bitstream.c:1092-1145`) with the
**multi-threaded packing path dropped** per L-35 (RESOLVED). The
Rust v1 implementation is single-threaded over already-encoded
tile buffers:

```text
encode_tile_data(tile_info, tiles, dest):
    assert!(tile_info.log2_rows == 0)                        // L-62
    let tile_cols = 1 << tile_info.log2_cols
    debug_assert!(tiles.len() == tile_cols)

    let mut total = 0
    for (col, tile) in tiles.iter().enumerate():
        let is_last = (col == tile_cols - 1)
        let tile_bytes = tile.pos()                          // BoolWriter::pos at stop time

        if !is_last:
            // 4-byte big-endian size prefix; mirrors mem_put_be32 in
            // vp9_bitstream.c:1137 and :1072.  L-13.
            if dest.len() - total < 4: return Err(BufferFull)
            dest[total..total+4].copy_from_slice(&u32::to_be_bytes(tile_bytes as u32))
            total += 4

        if dest.len() - total < tile_bytes: return Err(BufferFull)
        dest[total..total+tile_bytes].copy_from_slice(&tile.buf[..tile_bytes])
        total += tile_bytes

    Ok(total)
```

**Why this is single-threaded.** libvpx's `encode_tiles_mt`
(`vp9_bitstream.c:991-1090`) parallelises the per-tile boolean-coder
pass across workers. In our design, the per-tile boolean coding is
**performed by the tile workers during mode-decision** (each
`TileBuffer::writer()` is a per-worker `BoolWriter`); by the time
`encode_tile_data` runs, the per-tile byte buffers are already
complete. The "packing" step is just concatenation with size
prefixes. Trying to parallelise byte-copy + size-prefix-write is
both pointless and creates aliasing complications. **L-35 RESOLVED:
scalar pack only.**

This shifts the tile-thread bitstream-encode contract upstream: tile
workers must drive their own `BoolWriter` per-tile. That contract
is established by HLD-threading §3.6 and refined in
`tile-threading.md` §5.

### 5.5 Superframe assembly (`SuperframeBuilder::finish`)

Mirrors `write_superframe_index` (`vp9_cx_iface.c:1216-1271`).

```text
SuperframeBuilder::finish(self):
    if self.frame_count == 1:
        // Single-frame superframe: no index, just the frame bytes.
        return self.output

    assert!(self.frame_count >= 2 && self.frame_count <= 8)  // :1221-1222

    // marker: 110 NNN MMM  (bits)
    //   0xc0 base; OR (frame_count - 1) into low 3 bits; OR mag into bits 3..5
    let mut marker: u8 = 0xc0
    marker |= self.frame_count - 1                            // :1225

    // Magnitude: number of bytes per frame_size minus 1.
    // Find smallest mag (0..4) such that max_frame_size fits in (mag+1) bytes.
    let max_size = self.frame_sizes[..self.frame_count].iter().max()
    let mut mag = 0
    let mut mask: u32 = 0xff
    while mag < 4:
        if max_size < mask: break
        mask = (mask << 8) | 0xff
        mag += 1
    marker |= (mag << 3) as u8                                // :1233

    self.output.push(marker)                                  // :1255 (first marker)
    for i in 0..self.frame_count:
        let mut sz = self.frame_sizes[i] as u32
        for _ in 0..=mag:
            self.output.push((sz & 0xff) as u8)              // :1259-1262 (little-endian)
            sz >>= 8
    self.output.push(marker)                                  // :1264 (trailing marker)
    self.output
```

**Per-frame size is little-endian** within the index (LSB first;
`vp9_cx_iface.c:1259-1262`). The marker byte appears twice — at the
start and end of the index — so a decoder scanning backwards from
end-of-packet can detect "this is an index, not a frame".

### 5.6 Superframe parse (`parse_superframe`)

Mirrors the decoder-side logic in `vp9_decoder.c:527-585`.

```text
parse_superframe(packet) -> SmallVec<[&[u8]; 8]>:
    if packet.is_empty(): return Err(Truncated { needed: 1, available: 0 })

    let last = packet[packet.len() - 1]
    if (last & 0xe0) != 0xc0:
        // No trailing marker → single-frame superframe.
        return Ok(smallvec![packet])

    let frame_count = ((last & 0x7) + 1) as usize             // bits [2:0] + 1
    let mag = ((last >> 3) & 0x3) as usize                    // bits [4:3]
    let index_sz = 2 + (mag + 1) * frame_count
    if packet.len() < index_sz: return Err(InvalidSuperframeIndex)

    let index_start = packet.len() - index_sz
    if packet[index_start] != last:
        // Mismatched start marker → fall back to "no index".
        return Ok(smallvec![packet])

    // Parse per-frame sizes.
    let mut frames: SmallVec<[&[u8]; 8]> = SmallVec::new()
    let mut offset = 0
    let mut p = index_start + 1
    for _ in 0..frame_count:
        let mut sz: u64 = 0
        for j in 0..=mag:
            sz |= (packet[p + j] as u64) << (8 * j)
        p += mag + 1
        let sz = sz as usize
        if offset + sz > index_start: return Err(InvalidSuperframeIndex)
        frames.push(&packet[offset..offset + sz])
        offset += sz
    if offset != index_start: return Err(InvalidSuperframeIndex)
    Ok(frames)
```

Note the **little-endian** decode of `sz` (matches the encoder's
little-endian write, `vp9_cx_iface.c:1259-1262`). This contradicts
the per-tile size prefix in §5.4 which is **big-endian**
(`mem_put_be32`). Both are correct mirrors; the asymmetry is in
libvpx itself.

### 5.7 Per-tile boolean coder driven from tile workers

The per-tile token serialization (`pack_mb_tokens`,
`pack_inter_mode_mvs`, `write_mb_modes_kf`) lives in `entropy.md`
§6 (token loop) and in the encoder LLDs (`mode-decision.md` for
the mode/MV writers). This LLD's responsibility is the **boundary**:

```text
// In tile_worker (rvp9-encoder::threading):
let mut writer = tile_buffer.writer();              // BoolWriter borrow
for sb in tile.superblocks() {
    write_modes_sb(&fc, &mut writer, sb);          // mode-decision.md
}
writer.stop_encode();                                // entropy.md §3
drop(writer);                                        // returns &mut buf to tile_buffer

// In main thread after all tile workers sync:
let bytes = pack_frame(&mut fc, &counts, &header,
                       tx_mode, reference_mode, &tile_buffers, dest)?;
```

The main thread does NOT enter the boolean coder during
`encode_tile_data`; it does byte-copy + size-prefix arithmetic only.

### 5.8 The bit-exact contract (recap, with bitstream-specific items)

Per A4 §4.9, the boolean-coder contract is enumerated in `entropy.md`
§4. **This LLD adds bitstream-layout-specific invariants** that must
hold for byte parity:

1. **Marker bit position.** `VP9_FRAME_MARKER` is the **first two
   bits** of every frame (after Profile 3's 3-bit profile expansion
   in v2; in v1 with Profile 0, the first 4 bits are
   `0b10_00_0_0...` = marker(2) + profile(2) + show_existing_bit(1).
   Citation: `vp9_bitstream.c:1248-1253`.

2. **Sync-code byte placement.** Key frames emit
   `[0x49, 0x83, 0x42]` after the first 7 bits (frame marker 2 +
   profile 2 + show_existing 1 + frame_type 1 + show_frame 1 +
   error_resilient 1). The sync code is byte-aligned within libvpx's
   bit stream by the `vpx_wb_write_literal` 24-bit call; the bit
   stream is just bits, not bytes, until `vpx_wb_bytes_written`
   rounds up at uncompressed-header end. `vp9_bitstream.c:1264`.

3. **`compressed_hdr_size` field location.** The 16-bit reserved
   literal sits at byte offset `vpx_wb_bytes_written(saved_wb)`
   *measured at the time `vpx_wb_write_literal(&wb, 0, 16)` is
   called* (`vp9_bitstream.c:1424-1426`). The back-patch happens
   after the compressed header is fully written, by writing into
   the same bit offset (`vp9_bitstream.c:1445`). The Rust port stores
   the bit offset (not byte offset) so the rounding is identical.

4. **Per-tile size-prefix bytes.** Big-endian 32-bit (`mem_put_be32`,
   `vp9_bitstream.c:1072` and `:1137`). All but the **last** tile
   gets a prefix. The last tile's bytes extend to the end of the
   compressed-data region. L-13.

5. **Superframe-index marker byte byte-pattern.** `(marker & 0xe0)
   == 0xc0` is the recognition test (`vp9_decoder.c:527-585`).
   Bits [4:3] of the marker are `mag`; bits [2:0] are `frame_count -
   1`. **The `vpx_stop_encode` flush has a special case for this
   pattern**: if the last emitted byte of a tile's boolean coder
   stream matches `0xc0..0xdf`, an extra zero byte is appended
   (`vpx_dsp/bitwriter.c:33-55`, A4 §3.5). The boolean-coder owns
   this; this LLD only documents that the contract is honored.

6. **`show_existing_frame` short-circuit byte boundary.** Total wire
   form is the uncompressed-header bits (4 + show_existing_bit(1) +
   3 bits for `alt_fb_idx`) = 8 bits = 1 byte exactly. After
   `vpx_wb_bytes_written` rounds (1 byte) we are byte-aligned and
   the frame is "1 byte total". `vp9_bitstream.c:1417-1422`. L-15.

7. **Compressed-header size limit.** 16-bit literal forces
   `compressed_hdr_size ≤ u16::MAX = 65535`. Mirrors
   `vp9_bitstream.c:1441-1443`. Our `BitstreamError::CompressedHeaderTooLarge`
   maps to libvpx's `vpx_internal_error(VPX_CODEC_ERROR,
   "compressed_hdr_size > 16 bits")`.

8. **Tile-info encoding asymmetry.** `log2_tile_cols` uses a unary
   code; `log2_tile_rows` uses a 1-or-2-bit fixed code. Citation:
   §5.2.4 / `vp9_bitstream.c:886-900`.

---

## 6. SIMD dispatch

**None.** Per A4 §6 and HLD-simd §8.6, the bitstream and entropy
paths in libvpx have **zero RTCD entries** (confirmed by grep over
`vpx_dsp/vpx_dsp_rtcd_defs.pl`). The boolean coder is a serial
arithmetic loop; the raw-bit writer is a simple shift-and-OR loop;
the tile concat is byte-memcpy plus 4-byte big-endian writes.

All code in this LLD is therefore **scalar Rust**. Performance
hinges on the compiler inlining `RawBitWriter::write_bit` and on
LLVM's vectorization of the `copy_from_slice` calls in §5.4.

No SIMD dispatch table is added by this module; the encoder-wide
dispatch in `rvp9-core::simd::dispatch` (HLD-simd §2.2) is not
touched.

---

## 7. Deliberate divergences from libvpx

Each entry feeds `notes/divergence-ledger.md` per the divergence-ledger
template. Format: **what we change, why, impact on byte parity**.

### §7.1 Drop parallel tile-packing (L-35 RESOLVED)

**What:** the libvpx `encode_tiles_mt` path
(`vp9_bitstream.c:991-1090`) parallelises the *boolean coder pass*
across workers within `vp9_pack_bitstream`. We do not.

**Why:** in our design, each tile worker has already produced its
own per-tile byte buffer during mode-decision (the tile worker
drives `BoolWriter` directly). At pack time the main thread only
does concatenation + size-prefix arithmetic, which is memcpy-bound
and trivial to parallelise but yields no measurable speedup (a few
hundred KB per frame at 1080p60 is ~1 ms of memcpy). L-35.

**Impact on byte parity:** none. The byte order in the output is
identical to libvpx's. The only observable difference is wall-clock
during the pack call (a few hundred μs lower in libvpx's MT path,
not enough to measure against the frame budget).

**Ledger fields:** *Reason*: simplification. *Effect on byte
parity*: zero. *Effect on perf*: negligible (within noise of the
real-time budget).

### §7.2 Replace `setjmp/longjmp` error model with `Result`

**What:** libvpx errors via `vpx_internal_error()` →
`longjmp(error_info.jmp)`. Encountered at
`vp9_bitstream.c:1027-1029`, `:1085-1087`, `:1121-1124`,
`:1131-1134`, `:1412-1413`, `:1427-1430`, `:1441-1444`. We propagate
`Result<T, BitstreamError>` instead.

**Why:** HLD-arch §4 + D-09 RESOLVED. `setjmp/longjmp` is not
expressible in safe Rust without `nightly` features and `unsafe`
blocks at every call site; `Result` is the idiomatic alternative.

**Impact on byte parity:** none on success. On error, libvpx aborts
the frame and the partial bytes are abandoned. Our `Result` does the
same — caller discards `dest` content on `Err`. Functionally
equivalent.

### §7.3 Drop altref-driven multi-frame superframes (D-13)

**What:** libvpx's `write_superframe_index` is driven by
`pending_frame_count` which can be `> 1` when an altref frame is
queued ahead of a "show_existing" reveal. With altref **excluded**
in v1 (D-13 RESOLVED), the only sources of multi-frame superframes
in v1 are:

- SVC spatial layers (one frame per spatial layer per packet); and
- `show_existing_frame` paired with a hidden non-shown ref-creating
  frame (intra-only inter frame with `show_frame = 0`).

**Why:** D-13 RESOLVED — no altref in v1.

**Impact on byte parity:** none for streams that wouldn't use
altref anyway (real-time / WebRTC). The `SuperframeBuilder` API is
identical; the encoder simply never queues an altref frame for it.

### §7.4 Per-tile output is owned `Vec<u8>`, not `&mut [u8]` slice into a shared arena

**What:** libvpx allocates a single arena
`vp9_bitstream_worker_data[]` (`vp9_bitstream.c:967-989`) and hands
each worker a non-overlapping slice. We use `Vec<u8>` per `TileBuffer`.

**Why:** L-13. The arena pattern requires raw-pointer slice
arithmetic that survives across the multi-threaded boundary, with
no escape hatch when one worker's tile exceeds its slice budget.
With per-tile `Vec`, each tile grows naturally if its budget is
exceeded; no SAFETY contract needs to claim "tile `t` writes only
to bytes `[off_t, off_t+len_t)` of the shared arena". The cost is a
heap allocation per `TileBuffer` (amortised — the `Vec` is reused
frame-to-frame, sized to libvpx's `encode_tiles_buffer_alloc_size`
on first use).

**Impact on byte parity:** zero. The final packed frame is identical.

### §7.5 `show_existing_frame` is `Option<u8>`, not "bit + literal"

**What:** libvpx packs `show_existing` as a 1-bit flag followed by
an unconditionally-emitted 3-bit `alt_fb_idx` when the flag is set
(`vp9_bitstream.c:1253-1257`). The Rust port uses `Option<u8>` for
the in-memory representation; the wire form is identical (1 + 3
bits when `Some(idx)`).

**Why:** type safety. The libvpx layout makes it possible for an
inconsistent state (`show_existing = false` but `alt_fb_idx`
nominally set) to slip into the writer. `Option<u8>` makes this
unrepresentable.

**Impact on byte parity:** zero.

### §7.6 `BitstreamError::CompressedHeaderTooLarge` carries the actual size

**What:** libvpx errors with a string `"compressed_hdr_size > 16
bits"`. Our error carries the offending byte count for diagnostic
purposes. Functional behaviour identical.

**Why:** debuggability.

**Impact on byte parity:** none (error path).

### §7.7 No callback-based error handler on the raw-bit reader

**What:** libvpx `struct vpx_read_bit_buffer` (`bitreader_buffer.h:24-31`)
holds `error_handler_data` and `error_handler` callbacks that fire
on out-of-bounds read. We omit them and return
`BitstreamError::Truncated` instead.

**Why:** Rust's `Result` propagation makes the callback redundant.
The callback exists in libvpx only because C lacks `Result` and
because the caller wants to longjmp back to a recovery frame.

**Impact on byte parity:** none.

### §7.8 Reserve a typed `SuperframeBuilder` rather than per-frame mutate of `ctx->pending_*`

**What:** libvpx accumulates frames into `vpx_codec_alg_priv_t`
fields (`pending_cx_data`, `pending_frame_count`,
`pending_frame_sizes[]`, `pending_frame_magnitude`,
`vp9_cx_iface.c`). We expose a dedicated `SuperframeBuilder` type
the encoder constructs explicitly when SVC or `show_existing_frame`
queues multiple frames.

**Why:** Rust separation of concerns. The "pending superframe" state
is conceptually a builder, not a tangled field of the encoder root.

**Impact on byte parity:** zero. The wire form is identical.

### §7.9 `tile_buffer.buf` is byte-tight, not over-allocated to libvpx's worst-case constant

**What:** libvpx pre-allocates each worker's slice to
`mb_cols * mb_rows * 8 * 16 + (32 << 20) / num_workers`
(`vp9_bitstream.c:967-973`). Our `Vec<u8>` is grown on demand and
shrunk only by `Encoder` shutdown. Memory peak is similar; peak
allocation point differs.

**Why:** Rust's `Vec` is a better fit than malloc-and-track. The
libvpx constant is a hard cap with no fallback; ours is amortised.

**Impact on byte parity:** zero.

---

## 8. Open questions

Non-blocking cross-LLD / cross-HLD dependencies. Each item below
states this LLD's working assumption; the named owner picks them up
or pushes back.

| # | Item | Working assumption | Owner |
|---|---|---|---|
| 8.1 | The `smallvec` crate used for `FrameSlices::tiles` | Use workspace `smallvec` (already in Cargo lockfile via other crates) | LLD-frame-buffers / HLD-arch review |
| 8.2 | Whether `RawBitWriter` should be `#[inline(always)]` on `write_bit` / `write_literal` | Yes for `write_bit`; defer `write_literal` to Criterion bench in Stage 3 | LLD-entropy (paired with `BoolWriter` inlining) |
| 8.3 | Cost-of-overshoot detection during `write_compressed_header` (libvpx's `vpx_internal_error`) — surface to RC for `vp9_encodedframe_overshoot`? | No: this is a hard buffer-full failure, distinct from RC's overshoot logic (which is post-pack byte-count) | LLD-rate-control |
| 8.4 | Exact `BoolWriter` API surface (`start_encode`/`stop_encode`/`write`/`write_bit`/`write_literal`/`pos`) used by §5.7 and §5.4 | Surface as listed; `entropy.md` §3 confirms | LLD-entropy |
| 8.5 | Whether `TxMode` and `ReferenceMode` enums live in `rvp9-core::entropy` (consumers of the prob update) or `rvp9-core::bitstream` (header carriers) | `rvp9-core::entropy` for both — these are entropy-context fields, surfaced to bitstream as a thin enum that bitstream re-exports | LLD-entropy |
| 8.6 | The exact `encode_tiles_buffer_alloc_size` formula transcribed from `vp9_bitstream.c:967-973` and any edge cases at very low resolutions | Mirror exactly; cap `tile_buf.capacity()` at first frame, grow as needed | LLD-tile-threading |
| 8.7 | `frame_context_idx` choice (which of the 4 saved contexts the encoder selects per frame) | Stage 1 A4 documents this as a `vp9_bitstream.c:1307` raw 2-bit literal; the *selection logic* is encoder-policy, not bitstream concern. This LLD takes the value as input. | LLD-rate-control / LLD-frame-loop |
| 8.8 | Exact wire form of "fix_interp_filter" adjustment (`vp9_bitstream.c:1297`): does the bitstream writer adjust the value or does the encoder pre-adjust before calling us? | Pre-adjust before calling (encoder is the policy owner; this LLD is the byte-emitter). | LLD-mode-decision |
| 8.9 | Whether `parse_uncompressed_header` ever needs to be byte-tight (zero-copy) for decoder perf | Defer: the header is ~30 bytes; allocate a fresh `FrameHeader` per frame | LLD-decoder-driver |
| 8.10 | Does the encoder need a way to emit a "supplemental superframe data" test pattern (the `TEST_SUPPLEMENTAL_SUPERFRAME_DATA` `#ifdef` at `vp9_cx_iface.c:1240-1253`)? | No — `#ifdef`-gated debug-only code, never enabled in v1. Drop. | divergence-ledger §7.10 |
| 8.11 | Whether to seal `BitstreamError` (non-exhaustive)? | Mark `#[non_exhaustive]`. Allows future variants without breaking SemVer at workspace boundaries. | HLD-arch §4 confirm |
| 8.12 | How `pack_frame` interacts with `encode_breakout` / frame drop in RC (a dropped frame produces zero output bytes; does it still call `pack_frame`?) | No — RC drops at `vp9_get_compressed_data` level (A1 §3.4 step 4), before `pack_frame` is called. This LLD never sees a dropped frame. | LLD-rate-control |
| 8.13 | The decoder-side companion `parse_compressed_header` returns the updated `fc` in place. Should it also return per-frame counts? | No — counts are encoder-only (used for backward adaptation by the same encoder pass). The decoder applies the adapted probs from the bitstream prob updates and does not maintain counts. | LLD-entropy |

None of these are blocking; this LLD is implementable with the
working assumptions stated above.

---

## 9. Test strategy

The bitstream subsystem is the **single most byte-parity-sensitive
surface in the encoder** (per A4 §4.9 and HLD-simd §8). Test design
follows accordingly.

### 9.1 Unit tests

In `crates/rvp9-core/src/bitstream/{raw,header,tile_buffer,superframe,writer,reader}.rs`
as `#[cfg(test)]` modules.

| Surface | Test | Oracle |
|---|---|---|
| `RawBitWriter::write_bit` | Append bits 0..32, read back via `RawBitReader`, assert equal | round-trip |
| `RawBitWriter::write_literal` | All `bits ∈ 1..=32`, random values; round-trip | round-trip |
| `RawBitWriter::write_inv_signed_literal` | Sign-magnitude encoding for `i32` in `[-((1<<bits)-1), (1<<bits)-1]`; round-trip | round-trip |
| `SuperframeBuilder::finish` for `frame_count = 1` | No index emitted; output is just the pushed frame | golden bytes |
| `SuperframeBuilder::finish` for 2 ≤ frame_count ≤ 8, mag 0..=3 | Mark byte layout, per-frame size little-endian | golden bytes against libvpx (§9.3) |
| `parse_superframe` on builder output | Round-trip, frame slices match input | round-trip |
| `parse_superframe` on malformed input | Returns `InvalidSuperframeIndex` for: mismatched markers, sz overflow, truncation | error variant |
| `BitstreamError::CompressedHeaderTooLarge` | Construct fake `FrameContext` + `FrameCounts` that drive the compressed header > 64 KB; assert error | error variant |
| `pack_frame` `show_existing_frame` short-circuit | Output is exactly 1 byte; parse round-trip | golden byte |
| `pack_frame` keyframe | Sync code at correct byte offset; compressed-header back-patch matches measured size | structural |
| `write_tile_info` unary code | All valid `(min..=max) × {0,1,2}` combinations; round-trip | enumeration |

### 9.2 Property tests (`proptest`)

Located in `crates/rvp9-core/tests/bitstream_proptest.rs`.

```rust
proptest! {
    #[test]
    fn raw_writer_reader_roundtrip(
        ops in proptest::collection::vec(write_op(), 0..512),
    ) {
        let mut buf = vec![0u8; 4096];
        let mut wb = RawBitWriter::new(&mut buf);
        for op in &ops { op.apply_write(&mut wb); }
        let n = wb.finish().unwrap();

        let mut rb = RawBitReader::new(&buf[..n]);
        for op in &ops {
            let read = op.apply_read(&mut rb).unwrap();
            prop_assert_eq!(read, op.expected_value());
        }
    }

    #[test]
    fn superframe_roundtrip(frames in proptest::collection::vec(
        proptest::collection::vec(any::<u8>(), 1..1024),
        1..=8,
    )) {
        let mut sb = SuperframeBuilder::new();
        for f in &frames { sb.push_frame(f); }
        let packet = sb.finish();
        let parsed = parse_superframe(&packet).unwrap();
        prop_assert_eq!(parsed.len(), frames.len());
        for (i, &slice) in parsed.iter().enumerate() {
            prop_assert_eq!(slice, frames[i].as_slice());
        }
    }
}
```

### 9.3 Differential tests against libvpx (byte-parity)

The highest-leverage test. We **must** produce byte-identical
bitstream layouts for the parts we own, since downstream encode
divergence will eventually surface here.

#### 9.3.1 Captured `FrameHeader` test corpus

Capture libvpx-emitted `FrameHeader`-equivalent state at
`vp9_pack_bitstream` entry for the Stage 0 test corpus (synthetic
clips at 720p30/1080p30/1080p60 + WebRTC content). Format:
`tests/bitstream/golden/frame_<n>.bin` containing a deterministic
serialization of the input `FrameHeader` plus the byte range of
libvpx's emitted uncompressed-header bytes.

Replay through `rvp9_core::bitstream::write_uncompressed_header`
and assert byte-identical output.

#### 9.3.2 `pack_frame` byte parity

For frames where we can deterministically reconstruct the
*compressed-header inputs* (FrameContext + FrameCounts), test
end-to-end `pack_frame` against libvpx output. This requires
cooperation with `entropy.md` §9 (which owns the FrameContext /
FrameCounts capture format). In Stage 4a, this is the primary
divergence-hunting tool.

#### 9.3.3 Superframe layout parity

For SVC and `show_existing_frame` paired-frame test vectors,
assemble via `SuperframeBuilder::finish` and `memcmp` against
libvpx's `write_superframe_index` output. Test cases:
- Single-frame (no index)
- 2 frames, max_size < 256 (mag 0)
- 2 frames, max_size in [256, 65535] (mag 1)
- 2 frames, max_size > 65535 (mag 2)
- 8 frames, all sizes mixed (mag adjusts to max)

#### 9.3.4 Decoder-side conformance vectors

Per PLAN.md Stage 3 exit criterion ("100% pass on official VP9
conformance test vectors"): every conformance vector exercises
`parse_uncompressed_header`, `parse_superframe`, and the tile-byte
layout. Failures here block Stage 3 exit.

### 9.4 Fuzz tests

In Stage 5 hardening, `cargo-fuzz` targets:

- `fuzz_parse_uncompressed_header` — feed random `&[u8]` of length
  up to 2048; assert no panic, no `unwrap` panic, no buffer overrun.
  All malformed inputs must produce a `BitstreamError`.
- `fuzz_parse_superframe` — random `&[u8]` of length up to 65535;
  same contract.
- `fuzz_pack_unpack_roundtrip` — generate a valid `FrameHeader`
  (using `proptest`-derived strategies), pack to bytes, parse back;
  assert structural equality of the resulting `FrameHeader`.

### 9.5 Miri runs

`rvp9-core::bitstream` is **safe code** (all `unsafe` lives in
`rvp9-core::simd` or `rvp9-encoder::threading`). Miri runs are
expected to be clean. Where `pack_frame` calls into the boolean
coder, the boundary is `&mut [u8]`/`&mut BoolWriter` — also safe.

### 9.6 Bench harness

`crates/rvp9-bench/benches/bitstream.rs`:

- `bench_pack_frame_720p30` — driving a synthetic
  `(FrameHeader, FrameContext, FrameCounts, [TileBuffer; 4])` through
  `pack_frame` 1000 times; report median ns.
- `bench_parse_uncompressed_header` — golden frame header, 100k iters.
- `bench_superframe_assemble` — 3-spatial-layer SVC frame group,
  10k iters.

Targets:
- 720p30 `pack_frame`: ≤ 200 μs median (cumulative across the 33ms
  frame budget at < 1%).
- `parse_uncompressed_header`: ≤ 5 μs median.

### 9.7 Regression gates in CI

Per HLD-simd §6.5 / PLAN.md Stage 5:

- Differential byte-parity tests run on every PR; any drift triggers
  a divergence-ledger entry or a fix.
- Benchmarks tracked; > 10% regression on `pack_frame` is gated.

---

## 10. References

| Doc | Used for |
|---|---|
| PLAN.md | D1-D9 lock; D-01 / D-05 / D-13 / R-06 RESOLVED block; success criteria |
| `notes/vp9-analysis/00-overview.md` | Stage 1 synthesis (§3 state graph, §4 cross-cuts) |
| `notes/vp9-analysis/01-architecture.md` | §3.4 step 49 placement of `vp9_pack_bitstream` |
| `notes/vp9-analysis/04-bitstream-entropy.md` | Primary specialist; §1-§5 contain all of this LLD's libvpx algorithm references |
| `notes/vp9-analysis/decisions-needed.md` | L-13, L-14, L-15, L-35 |
| `notes/hld/00-architecture.md` | §1 crate boundaries; §2.1 module placement; §4 error model; §6 integer-width; §7 8-bit-only pixel policy |
| `notes/hld/01-threading.md` | §3.6 per-tile worker contract (this LLD's `TileBuffer::writer()` boundary) |
| `notes/hld/02-simd-dispatch.md` | §8.6 — zero SIMD on bitstream confirmed |
| `notes/hld/04-svc.md` | superframe assembly for spatial layers |
| Appendix A of `04-bitstream-entropy.md` | File:line cross-reference index used throughout this LLD |

---

*Document version: 0.1.0. LLD-bitstream — Stage 2 per-module LLD.
Anchors to HLD-arch §§1, 2, 4, 6, 7; HLD-threading §3.6; HLD-simd §8.6.
Companion to `entropy.md` (load-bearing — that LLD owns the boolean
coder and all probability tables).*
