# HLD-04: Scalable Video Coding (SVC) + EXT_RATECTRL plug-in

**Stage:** 2 (HLD)
**Owner:** SVC HLD writer (Claude, dispatched by Arthur).
**Status:** Draft for Arthur review.
**Locked PLAN.md decisions in scope:**

- **D3** — Full SVC (temporal + spatial) in v1.
- **D-13** — RESOLVED 2026-05-16: **CBR + VBR-RT, NO altref.**
- **R-06** — RESOLVED 2026-05-16: **EXT_RATECTRL plug-in IN v1** (libwebrtc
  dependency).

**Stage 1 sources cited throughout:**

- `notes/vp9-analysis/00-overview.md` (A8 synthesis)
- `notes/vp9-analysis/06-threading-svc.md` (A6, primary SVC source)
- `notes/vp9-analysis/05-rate-control.md` (A5, per-layer RC)
- `notes/vp9-analysis/04-bitstream-entropy.md` (A4, superframe index)
- `notes/vp9-analysis/decisions-needed.md` (open L-NN / D-NN / R-NN)

All `file:line` citations are into `third_party/libvpx/` at v1.14.1 commit
`12f3a2a` unless explicitly noted otherwise.

Cross-HLD assumptions (we do not block on peer HLD output; see §10):

- **HLD-00 (arch)**: encoder lives in `crates/rvp9-encoder`; public type is
  `Encoder` owning a single `EncoderState` (Rust analogue of `VP9_COMP`).
- **HLD-01 (threading)**: tile-MT + row-MT primitives use `std::thread::scope`
  with hand-rolled per-tile-column job queues (A6 §7.3 recommendation; L-56).
- **HLD-02 (SIMD)**: down-sampling filters for source scaling (bilinear,
  EIGHTTAP_SMOOTH, EIGHTTAP) have SIMD entries in the shared RTCD dispatch
  table populated before any worker spawns (R-09).

---

## 1. Goals and non-goals

### 1.1 Goals

1. **Bitstream-output equivalence to libvpx** for fixed temporal patterns
   (MODE_0101, MODE_0212) and for `MODE_BYPASS` with `vpx_svc_ref_frame_config_t`
   driven by the application. BD-rate per layer within 0.5% of libvpx
   (PLAN.md Stage 4d + 4e exit criterion).
2. **WebRTC viability**: 3-spatial × 3-temporal layers at 720p30 stays within
   the real-time budget (≤33 ms median, ≤50 ms p99 — PLAN.md §2).
3. **WebRTC integration path**: a thin Rust analogue of libvpx's
   `VP9RateControlRTC` is exposed so libwebrtc can drive QP and frame-drop
   decisions externally (R-06 / EXT_RATECTRL).
4. **Safe layer-context restore/save**: spatial layer iteration mutates
   `EncoderState` in libvpx-equivalent ways without taking `&mut` to overlapping
   state.

### 1.2 Non-goals (v1)

1. **Altref / temporal filtering inside SVC.** D-13 RESOLVED to no altref;
   PLAN.md §2 commits to `lag_in_frames=0`. This collapses several SVC
   code paths — see §9.
2. **2-pass SVC** (`vp9_rc_get_svc_params` two-pass branches at
   `vp9_ratectrl.c:2279-2491`). 1-pass only per PLAN.md.
3. **TPL model for SVC** (`enable_tpl_model`). Off in all RT cascades
   (`vp9_speed_features.c:474`).
4. **Spatial scale factors other than 1:1, 1:2, 1:4.** `use_base_mv` /
   `use_partition_reuse` only work for these in libvpx
   (`vp9_svc_layercontext.c:936-961`); rvp9 v1 will assert the same constraint
   in the configuration step.
5. **Mid-stream layer count changes.** Layer count is fixed at
   `Encoder::new` time; only the *enabled* layer range (`first_spatial_layer_to_encode`
   ..`number_spatial_layers`) can shift mid-stream.
6. **Spatial layer counts > 3.** PLAN.md §2 commits to "2- and 3-layer"
   spatial SVC. libvpx supports up to `VPX_SS_MAX_LAYERS = 5`
   (`vpx/vpx_encoder.h:46`); we cap at 3 in v1 and reject configs with more
   at the API boundary.
7. **`max_consec_drop_ms` mode.** `vp9_ratectrl.c:447-512` references a
   newer-API consecutive-drop cap by milliseconds rather than by frame count
   (`max_consec_drop_ms_set` path); v1 mirrors only the frame-count variant
   (`max_consec_drop`, used by `VP9RateControlRTC`).

---

## 2. Temporal SVC

### 2.1 Layering modes (mirror of libvpx)

Three temporal patterns are in scope, dispatched by
`VP9E_TEMPORAL_LAYERING_MODE` (`vpx/vp8cx.h:790`; A6 §5.3):

| Mode | Pattern | Period | Dispatch (libvpx) |
|---|---|---|---|
| `NoLayering` | `tl = 0, 0, 0, ...` | 1 | `set_flags_and_fb_idx_for_temporal_mode_noLayering` (`vp9_svc_layercontext.c:687`) |
| `Mode_0101` (2-layer) | `tl = 0, 1, 0, 1, ...` | 2 | `set_flags_and_fb_idx_for_temporal_mode2` (`vp9_svc_layercontext.c:626`) |
| `Mode_0212` (3-layer) | `tl = 0, 2, 1, 2, ...` | 4 | `set_flags_and_fb_idx_for_temporal_mode3` (`vp9_svc_layercontext.c:517`) |
| `Bypass` | application-driven | n/a | `set_flags_and_fb_idx_bypass_via_set_ref_frame_config` (`vp9_svc_layercontext.c:722`) |

Rust enum (in `rvp9-encoder::svc`):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalLayering {
    NoLayering,
    /// 2-layer hierarchy, period 2. tl pattern: 0,1,0,1,...
    Mode0101,
    /// 3-layer hierarchy, period 4. tl pattern: 0,2,1,2,0,2,1,2,...
    Mode0212,
    /// Application drives layer + ref-frame selection per frame via
    /// `RefFrameConfig` on each `Encoder::encode` call.
    Bypass,
}
```

Pattern computation, mirroring `vp9_svc_layercontext.c:519-528` (mode_0212):

```rust
fn temporal_id_in_pattern(mode: TemporalLayering, frame_in_layer: u32) -> u8 {
    match mode {
        TemporalLayering::NoLayering => 0,
        TemporalLayering::Mode0101 => (frame_in_layer & 1) as u8,
        TemporalLayering::Mode0212 => {
            // sequence: frame_num%4 = 0→tl0, 1→tl2, 2→tl1, 3→tl2
            let n = frame_in_layer & 3;
            if n & 1 != 0 { 2 } else { (n >> 1) as u8 }
        }
        TemporalLayering::Bypass => {
            // unreachable: caller supplies temporal_id via RefFrameConfig
            unreachable!("temporal_id is application-driven in Bypass mode")
        }
    }
}
```

### 2.2 Reference-frame slot map per layer (verbatim)

A6 §5.4.1 (`MODE_0101`) and §5.4.2 (`MODE_0212`) gave the exhaustive
`(temporal_id, spatial_id) → (ref_frame_flags, refresh_last, refresh_golden,
refresh_alt, lst_fb_idx, gld_fb_idx, alt_fb_idx)` tables. The HLD restates
the design rules (rvp9 will table-drive these, mirroring libvpx exactly):

- **`tl == 0` (base temporal)**: refreshes LAST slot `sl` for `sl == 0`; for
  `sl > 0` either refreshes GOLDEN (when base spatial is a key frame, `K`) or
  refreshes LAST and uses GOLDEN as the inter-layer reference (otherwise).
  GOLDEN ↔ LAST swap is what enables inter-layer prediction (sl-1 → sl).
- **`tl == 1` (mid-temporal)**: refreshes ALTREF in slot `NSL + sl`; reads from
  LAST (own spatial layer's TL0) + GOLDEN (spatial layer sl-1's TL1 / ALTREF).
- **`tl == 2` (top-temporal, in `MODE_0212`)**: never refreshes a reference
  used by a future TL0 frame. Two variants depending on which TL2 slot in the
  period (`frame_num%4 == 1` vs `frame_num%4 == 3`).

**Buffer-pool slot allocation** (mirrors `vp9_svc_layercontext.c:69-73,
83-86`):

| Slot | Use |
|---|---|
| 0..NSL-1 | Per-spatial-layer LAST (TL0) refresh |
| NSL..2*NSL-1 | Per-spatial-layer ALTREF (TL1) refresh |
| 6, 7 | Long-term GF temporal reference (`use_gf_temporal_ref`), only for top 1–2 spatial layers |

`NSL = number_spatial_layers ≤ 3`. With NSL=3, slots 0..5 are used by the
periodic temporal/spatial refresh scheme, and 6+7 are the long-term refs.
This fits in VP9's 8-slot reference pool (`REF_FRAMES = 8`). For NSL=2,
slots 0..3 are used and 4..7 are spare; libvpx still places long-term refs at
6+7 if `use_gf_temporal_ref` is enabled.

**Rust layout** for the SVC layer-context table (per Stage 1 §2.8–2.9):

```rust
/// Per (spatial, temporal) cell. Layer-array index =
/// sl * number_temporal_layers + tl (mirrors LAYER_IDS_TO_IDX,
/// vp9_encoder.h:1495).
pub struct LayerContext {
    pub rc: RateControl,                        // A5: full per-layer RC
    pub target_bandwidth: i64,                  // bits/sec for this cell
    pub spatial_layer_target_bandwidth: i64,
    pub framerate: f64,
    pub avg_frame_size: i32,
    pub max_q: u8, pub min_q: u8,
    pub scaling_factor_num: u16, pub scaling_factor_den: u16,
    pub current_video_frame_in_layer: u32,
    pub is_key_frame: bool, pub frames_from_key_frame: u32,
    pub last_frame_type: FrameType,
    pub layer_size: usize,
    // CYCLIC_REFRESH state (D-05 RESOLVED: mode 3 only), see L-67:
    pub cr_sb_index: u32,
    pub cr_map:              Option<Box<[i8]>>,
    pub cr_last_coded_q_map: Option<Box<[u8]>>,
    pub cr_consec_zero_mv:   Option<Box<[u8]>>,
    pub cr_actual_num_seg1_blocks: i32,
    pub cr_actual_num_seg2_blocks: i32,
    pub counter_encode_maxq_scene_change: i32,
    pub qindex_delta: [i32; 3],
    pub speed: u8, pub loopfilter_ctrl: u8,
    pub frame_qp: i32, pub mbs: u32,
}
```

Absent vs libvpx's `LAYER_CONTEXT` (`vp9_svc_layercontext.h:40`):
`twopass`, `rc_twopass_stats_in`, `alt_ref_source`, `alt_ref_idx`,
`gold_ref_idx`, `has_alt_frame` — **D-13 RESOLVED: no altref, no 2-pass.**
See §9.

### 2.3 Layer iteration control flow

Per A6 §3.7, the libvpx superframe loop is:

```text
for sl in first_spatial_layer_to_encode..number_spatial_layers:
    cpi->svc.spatial_layer_to_encode = sl
    vp9_one_pass_svc_start_layer(cpi)   // set fb_idx, ref_frame_flags, scale
    vp9_restore_layer_context(cpi)       // cpi->rc = lc->rc; swap CR maps
    encode_frame_to_data_rate(cpi)
    vp9_save_layer_context(cpi)
    vp9_inc_frame_in_layer(cpi)
```

The rvp9 mapping is a method on `Encoder` driving a `SuperframeContext`:

```rust
impl Encoder {
    pub fn encode(&mut self, frame: &SourceFrame, opts: &FrameOptions)
        -> Result<EncodedSuperframe, EncodeError>
    {
        let mut sf_out = EncodedSuperframe::empty(self.svc.number_spatial_layers);

        for sl in self.svc.first_spatial_layer..self.svc.number_spatial_layers {
            self.svc.spatial_layer_id = sl;
            self.start_layer(opts)?;            // §2.4
            self.restore_layer_context();        // L-58
            let layer_out = self.encode_one_layer(frame)?;
            self.save_layer_context();
            self.inc_frame_in_layer();
            sf_out.push_layer(layer_out);
        }
        sf_out.assemble_superframe_index();      // §5
        Ok(sf_out)
    }
}
```

`Encoder` owns `svc: Svc` (the top-level SVC state, mirroring A6 §2.8) and
`layers: Box<[LayerContext]>` of length
`number_spatial_layers * number_temporal_layers`. Layer iteration is
strictly sequential; tile workers (HLD-01) parallelize *within* one layer's
`encode_one_layer` call.

### 2.4 `start_layer` responsibilities

`start_layer` (rvp9 analogue of `vp9_one_pass_svc_start_layer`,
`vp9_svc_layercontext.c:758-1024`) does, in order:

1. Compute `temporal_layer_id` from `temporal_layering_mode` +
   `lc.current_video_frame_in_layer` (§2.1). For `Bypass`, read it from the
   caller-supplied `RefFrameConfig`.
2. **Simulcast detection** (`vp9_svc_layercontext.c:765-775`): if
   `disable_inter_layer_pred == InterLayerPred::Off` AND the base spatial
   layer is a key frame, mark `svc.simulcast_mode = true`. In simulcast mode,
   every spatial layer is encoded as a key frame independently.
3. **Reference-slot dispatch**: dispatch to `set_flags_..._mode0101` /
   `set_flags_..._mode0212` / `set_flags_..._noLayering` /
   `set_flags_..._bypass_via_set_ref_frame_config`. The dispatch fills:
   - `lst_fb_idx[sl]`, `gld_fb_idx[sl]`, `alt_fb_idx[sl]`
   - `ext_refresh_last_frame`, `ext_refresh_golden_frame`,
     `ext_refresh_alt_ref_frame`
   - `ref_frame_flags` (bitmask of `VP9_LAST_FLAG | VP9_GOLD_FLAG |
     VP9_ALT_FLAG`)
4. **Source scaling** (`vp9_svc_layercontext.c:854-925`): compute
   `(width, height)` for this layer via
   `get_layer_resolution(oxcf.width, oxcf.height, scaling_num, scaling_den)`.
   Switch the active source-scaling filter:
   - `width * height ≤ 320 * 240` → `EIGHTTAP_SMOOTH`
     (`vp9_svc_layercontext.c:928-930`)
   - `scaling_factor_num > (3 * scaling_factor_den) >> 2` (i.e. > 0.75x) →
     `downsample_filter_phase[sl] = 0`
   - Default: `Bilinear`, `phase = 8`.
5. **Inter-layer-prediction constraint** (`vp9_svc_constrain_inter_layer_pred`,
   `vp9_svc_layercontext.c:1076-1147`): clear ref flags that would require a
   scaled (cross-resolution) reference if `InterLayerPred::Off` or
   `OffNonKey`, or if `InterLayerPredOnConstrained` and the immediately-prior
   spatial layer didn't refresh the slot we're about to read from.
6. **Long-term GF temporal ref** (`vp9_svc_layercontext.c:851-860`): enable
   `use_gf_temporal_ref_current_layer` if all six gates pass (A6 §5.4.4).
7. **`use_base_mv` / `use_partition_reuse` validation**
   (`vp9_svc_layercontext.c:941-961`): disable if any spatial scale factor
   isn't 1:2 or 1:4. If previous spatial layer dropped, force these off.
8. **Drop-state propagation**
   (`vp9_svc_layercontext.c:979-1024`): if the same-spatial-layer frame from
   the previous superframe was dropped, redirect `lst_fb_idx[sl]` to
   `fb_idx_upd_tl0[sl]` so we don't reference a frame that doesn't exist.
   See §4.

The output of `start_layer` is a fully-prepared `LayerEncodeParams` that
feeds the frame loop in §3 (spatial-SVC down-scale) and §6 (per-layer Q
assignment).

### 2.5 Non-reference-frame marking

`svc.non_reference_frame` (`vp9_svc_layercontext.c:963-972`) is set to 1 when
neither LAST, GOLDEN, nor ALTREF will be refreshed by this frame. For
`MODE_0212` this is true for the second TL2 frame at the top spatial layer
(`sl == NSL-1`, `frame_num % 4 == 3`). The flag is used by the speed-feature
cascade (`vp9_speed_features.c:712`: `SUBPEL_TREE_PRUNED_EVENMORE` for
non-ref enhancement layers at cpu-used 7+) and the bitstream packer (to set
`show_existing_frame = 0` and skip backward prob updates).

Rust struct field: `svc.non_reference_frame: bool` per layer, computed in
`start_layer` step 6/7.

---

## 3. Spatial SVC

### 3.1 Multi-resolution pipeline

For each superframe input, the source frame is **down-scaled per spatial
layer** before encode. libvpx maintains pre-allocated scaled-source buffers
in `cpi->svc.scaled_frames[MAX_LAG_BUFFERS]` and `cpi->svc.scaled_temp`
(`vp9_svc_layercontext.h:91-94`). With D-13 RESOLVED (no altref,
`lag_in_frames = 0`), `MAX_LAG_BUFFERS` is effectively 1 in our build; we
only need `scaled_frames[0]` and `scaled_temp` per superframe.

**Rust mapping**:

```rust
pub struct SvcSourceScaler {
    // One pre-allocated YUV buffer per spatial layer, owned by the encoder
    // and reused superframe-to-superframe.
    scaled: Box<[Yv12Buffer]>,             // len == number_spatial_layers
    temp:   Yv12Buffer,                    // 2-stage downsample scratch
    // Filter kernel selection updated per layer in start_layer step 4.
    filter_type:  [InterpFilter; MAX_SPATIAL_LAYERS],
    filter_phase: [u8;            MAX_SPATIAL_LAYERS],
}
```

The down-scale call site (libvpx: `vp9_scale_if_required`,
`vp9_encoder.c:3990-4016`) becomes a per-spatial-layer method:

```rust
fn scale_source_for_layer(&mut self, src: &SourceFrame, sl: u8) -> &Yv12Buffer {
    let lc = &self.layers[sl as usize * self.svc.number_temporal_layers as usize];
    let (w, h) = get_layer_resolution(
        self.cfg.frame_width, self.cfg.frame_height,
        lc.scaling_factor_num, lc.scaling_factor_den,
    );
    if w == src.width() && h == src.height() {
        // 1:1 — caller can use src directly.
        return src.as_yv12();
    }
    // Two-stage if scale factor is 1:4 to keep filter quality.
    vp9_scale_and_extend_frame_nonnormative(
        src,
        &mut self.svc.scaler.temp,
        &mut self.svc.scaler.scaled[sl as usize],
        self.svc.scaler.filter_type[sl as usize],
        self.svc.scaler.filter_phase[sl as usize],
    );
    &self.svc.scaler.scaled[sl as usize]
}
```

Spatial scale factor constraints (v1):

- 2 spatial layers: 1:2 or 1:1 ratios (e.g. 360p+720p, 540p+1080p).
- 3 spatial layers: 1:4 / 1:2 / 1:1 (e.g. 270p + 540p + 1080p) or
  1:2 / 1:2 / 1:1 (illegal in libvpx for use_base_mv but legal for plain
  inter-layer prediction).
- Other ratios are accepted but disable `use_base_mv` and
  `use_partition_reuse` (A6 §5.5; `vp9_svc_layercontext.c:941-953`).

### 3.2 Inter-layer texture prediction

VP9 has no dedicated inter-layer prediction syntax; instead it reuses the
existing **inter-frame prediction** machinery with the lower-layer
reconstruction stored in a *scaled* reference buffer. The "magic" is in the
reference-slot scheme (§2.2):

- Base spatial layer (`sl == 0`) is a regular VP9 frame. After encode, its
  reconstruction is stored in slot 0 (and slot 6 / 7 if long-term GF
  temporal ref is enabled).
- Enhancement spatial layer (`sl > 0`) at the same temporal position
  reads from GOLDEN slot which has been remapped to point at slot `sl - 1`
  (`gld_fb_idx[sl] = sl - 1`). The decoder uses VP9's standard inter-frame
  prediction with a scale-factor `ref_scale[GOLDEN] ≠ 1`, which triggers the
  scaled-prediction code path (`vp9_reconinter.c` scaled-MC kernels).
- The encoder side mirrors this: when computing prediction error for an
  enhancement-layer SB, the lower-layer reconstruction is treated as a
  scaled reference (handled by `vp9/encoder/vp9_pickmode.c` via the same
  `scaled_ref_idx[]` indirection that handles dynamic resize).

**Key Rust invariant**: the buffer-pool slot for `sl-1`'s reconstruction
must be *fully written* before `sl`'s encode begins. Since spatial layers
iterate sequentially in `Encoder::encode` (§2.3), this is guaranteed by the
single-thread layer loop; inside one layer, tile workers operate on
disjoint slices and don't read across the slot boundary.

### 3.3 Inter-layer motion prediction (`use_base_mv`)

Real-time spatial SVC at 1:2 scale uses **motion-vector reuse**: the
enhancement layer's MV search seeds from the doubled MV of the
co-located base-layer block. This is `use_base_mv` (A6 §5.5).

libvpx implementation:

- `svc.use_base_mv = 1` is set when all spatial ratios are 1:2 (or 1:4
  for `sl == 0` of a 3-layer config).
- `vp9_pickmode.c` consults `cpi->svc.use_base_mv` and, when set, scales the
  base-layer MV by 2 and uses it as the starting point of the
  enhancement-layer ME loop.

rvp9 design: store the per-superblock MV grid of the base spatial layer
after encode. When encoding the enhancement layer, the ME module reads
`(2*mv_base.x, 2*mv_base.y)` as the seed for `mi_row*2, mi_col*2`
co-located SB. Disabled if the previous spatial layer dropped
(`vp9_svc_layercontext.c:956-960`).

### 3.4 Partition reuse (`use_partition_reuse`)

Similar to MV reuse, partition decisions from the lower layer can seed the
upper layer's variance-based partition search. Stored in
`svc.prev_partition_svc: Box<[BlockSize]>` of length
`(mi_cols / 8 + 1) * (mi_rows / 8 + 1)` per spatial layer
(`vp9_svc_layercontext.h:115`).

rvp9 design: a single `prev_partition_svc` buffer per spatial layer,
allocated once at `Encoder::new` and reused per superframe. Hooks into the
`VAR_BASED_PARTITION` path in HLD-partition.

Speed-feature gate: `sf->svc_use_lowres_part` is set at cpu-used 7 only for
3-spatial-layer configs with `temporal_layer_id > 0` and resolution > 640×480
(`vp9_speed_features.c:735-739`).

### 3.5 Inter-layer prediction policy (`InterLayerPred`)

Matches `vp9_svc_layercontext.h:22-28`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterLayerPred {
    /// Default. Inter-layer prediction enabled on every spatial layer
    /// beyond the base.
    On,
    /// Inter-layer prediction fully disabled; every spatial layer is
    /// independently encoded (simulcast).
    Off,
    /// Inter-layer prediction enabled only on key superframes; non-key
    /// frames are independent.
    OffNonKey,
    /// Inter-layer prediction enabled only when the immediately-previous
    /// spatial layer in the same superframe actually refreshed the slot
    /// the current layer wants to reference.
    OnConstrained,
}
```

The constraint is enforced in `start_layer` step 5 (§2.4); rvp9 mirrors
`vp9_svc_constrain_inter_layer_pred` line-for-line because the slot-refresh
predicate is non-obvious and divergence here corrupts reference state.

---

## 4. Cross-layer rate control

This section resolves **L-77** (temporal_layering × per-layer RC) and
**L-68** (scene-detect per-layer behaviour).

### 4.1 Per-layer `RateControl` state

Per A5 §3.4 and §2.1, each `LayerContext` carries its own full
`RATE_CONTROL` struct (~70 fields). The top-level `cpi->rc` is a *scratch*
that's swapped with the active layer's `LayerContext.rc` at start/end of
each layer encode (A6 §2.9).

Rust mapping (L-58 RESOLVED):

```rust
impl Encoder {
    fn restore_layer_context(&mut self) {
        let idx = self.layer_idx(self.svc.spatial_layer_id, self.svc.temporal_layer_id);
        // value-swap, not pointer-swap, to match libvpx semantics
        std::mem::swap(&mut self.rc, &mut self.layers[idx].rc);
        // For CYCLIC_REFRESH (D-05): swap the segmentation map ownership.
        // L-67 makes ownership unambiguous: maps live in LayerContext;
        // the active layer's pointers are temporarily aliased into self.cr.
        if self.cfg.aq_mode == AqMode::CyclicRefresh
            && self.svc.number_spatial_layers > 1
            && self.svc.temporal_layer_id == 0
        {
            std::mem::swap(&mut self.cr.map, &mut self.layers[idx].cr_map);
            std::mem::swap(&mut self.cr.last_coded_q_map, &mut self.layers[idx].cr_last_coded_q_map);
            std::mem::swap(&mut self.cr.consec_zero_mv,    &mut self.layers[idx].cr_consec_zero_mv);
        }
    }
    fn save_layer_context(&mut self) { /* reverse of restore */ }
}
```

The swap is single-threaded, between frame-encode invocations (tile workers
have already joined), so no aliasing issue.

### 4.2 Per-layer Q assignment

`vp9_rc_get_svc_params` (`vp9_ratectrl.c:2279-2491`) is the SVC entry point
that fans out to:

- `vp9_calc_pframe_target_size_one_pass_cbr` per layer (A5 §4.6, line 900:
  `target = layer.avg_frame_size`).
- Per-temporal-layer buffer-level pre-pump in
  `vp9_update_buffer_level_svc_preencode` (`vp9_ratectrl.c:263-294`).
- Per-spatial-layer `active_worst_quality` / `active_best_quality` via the
  CBR/VBR pick paths but with `LAYER_CONTEXT.rc` swapped in.

**Cross-layer Q allocation rule** (libvpx baseline, mirror exactly):

- The base spatial layer (`sl == 0`) gets the most aggressive Q range
  (low Q = high quality). `LayerContext.max_q` / `min_q` per layer constrains
  the per-layer pick.
- Upper layers inherit a `lower_layer_qindex` floor from the just-encoded
  lower layer (`vp9_svc_layercontext.c:1066-1074`,
  `svc->lower_layer_qindex = cm->base_qindex`). This ensures the upper layer
  never picks a Q *lower* than the layer it predicts from.
- Each spatial layer's *target bitrate* drives its `LayerContext.rc.avg_frame_bandwidth`;
  the `vp9_rc_regulate_q` Q search (A5 §4.2) honours that per-layer target.

Rust struct field: `svc.base_qindex: [u8; MAX_SPATIAL_LAYERS]`. Written
after each layer encode in `save_layer_context`; read by the next layer's
`pick_q` to enforce the floor.

### 4.3 L-77 RESOLUTION: temporal_layering × per-layer RC

A7 (#42) flagged a gap: A5 and A6 didn't trace the interaction between
`temporal_layering_mode` and per-layer RC. Resolution for the rvp9 HLD:

1. **One `LayerContext.rc` per (spatial, temporal) cell**, not per spatial
   layer. With NSL=3, NTL=3 we have 9 `LayerContext` instances; each has
   its own `RateControl`. This matches libvpx
   (`layer_context[sl * NTL + tl]`, `vp9_encoder.h:1495`).
2. **Cross-temporal RC fanout**: at the start of each TL=0 superframe,
   `vp9_rc_get_svc_params` pre-pumps the buffer level of TL=1 and TL=2
   layers proportionally to their bandwidth fraction. Specifically
   (mirroring `vp9_ratectrl.c:263-294`):

   ```text
   for sl in 0..NSL:
       for tl in 0..NTL:
           lc = layers[sl*NTL + tl]
           lc.rc.bits_off_target += lc.rc.avg_frame_bandwidth
           lc.rc.bits_off_target = min(lc.rc.bits_off_target,
                                        lc.rc.maximum_buffer_size)
           lc.rc.buffer_level = lc.rc.bits_off_target
   ```

3. **Post-encode RC update** debits *only* the layer that was actually
   encoded (and any TL>= layer that this frame is *displayed-by*; in
   `MODE_0212` a TL=0 frame is displayed by TL=1 and TL=2 as well, so all
   three buffers debit). This is the "temporal fan-out" that A5
   `update_layer_buffer_level_postencode` (`vp9_ratectrl.c:298-313`) handles:

   ```text
   fn post_encode_update_layers(sl: u8, tl_encoded: u8, encoded_size: i64):
       for tl in tl_encoded..NTL:
           let lc = &mut layers[sl as usize * NTL + tl as usize];
           lc.rc.bits_off_target -= encoded_size;
           lc.rc.bits_off_target = min(lc.rc.bits_off_target, lc.rc.maximum_buffer_size);
           lc.rc.buffer_level = lc.rc.bits_off_target;
   ```

4. **AQ-CR in SVC**: `CYCLIC_REFRESH_AQ` only operates on the base
   temporal layer (`vp9_aq_cyclicrefresh.c:491`). This means the
   CR segmentation map updates happen on `tl == 0` frames; TL>0 frames
   inherit the segmentation map but don't update it. The HLD therefore
   stores the CR map *only in the TL=0 LayerContext per spatial layer*
   (one CR map per spatial layer, 3 maps for NSL=3). This matches
   `vp9_svc_layercontext.c:158-179` ("allocated only when
   `ss_number_layers > 1 && tl == 0`"). Memory: ~`mi_rows * mi_cols / 16`
   bytes per map per spatial layer.

5. **Per-layer scene-change reset (L-68)**: `vp9_scene_detection_onepass`
   runs on the **base spatial, base temporal** frame only (A5 §3.4 implicit
   gating). When a scene change is detected, the reset propagates to all
   spatial layers via `svc.high_source_sad_superframe = 1`. This drives:
   - `rc.high_source_sad = 1` on all per-layer RCs for this superframe.
   - `rc.reset_high_source_sad = 1`, which triggers `active_worst =
     worst_quality` in the next inter frame's `calc_active_worst_quality`
     (A5 §4.4 line 772).
   - For TL>0 layers, `rc.high_source_sad` is read but not separately
     computed — they inherit the base-layer detection. **rvp9 v1: mirror
     exactly. Per-layer scene detection is out of scope.**

### 4.4 Denoiser × SVC interaction

A5 §3.4 + denoiser §3 (`vp9_denoiser.c:625-631`):

- The denoiser only runs on the top 1–2 spatial layers (controlled by
  `svc.first_layer_denoise`).
- Denoiser allocates `mc_running_avg_y[]` per spatial layer
  (`vp9_denoiser.h:30`, `SVC_REF_FRAMES = 9` slots per layer).
- Inter-layer reference state (the lower spatial layer's reconstruction
  stored in GOLDEN slot) is *not* denoised — only LAST.

rvp9 v1 design: a `Denoiser` instance is owned per top-2 spatial layers.
The `Encoder::encode_one_layer` call gates denoiser invocation on
`sl >= self.svc.first_layer_denoise`. The denoiser's per-layer state is
stored in `LayerContext.denoiser: Option<Denoiser>` (None for layers not
covered).

This is HLD-denoiser's territory; we only restate that the per-layer
ownership lives here.

---

## 5. Bitstream layout

### 5.1 Superframe index

The VP9 spec defines a **superframe index** as the trailing bytes of every
encoded packet that contains more than one VP9 frame. This is the only
SVC-aware bitstream construct; the per-frame data inside the superframe is
plain VP9 — see A4 §3.4 for tile data and §3.3 for the uncompressed-header
flush invariant.

The libvpx encoder side is `write_superframe_index` in
`vp9_cx_iface.c:1216-1271`. Wire format:

```text
   ┌──────────┬──────────┬──────────┬──────────┬──────────┐
   │  marker  │ size[0]  │ size[1]  │   ...    │  marker  │
   └──────────┴──────────┴──────────┴──────────┴──────────┘
       1 B     (mag+1) B  (mag+1) B  ...         1 B
```

- `marker` byte: `0b110_MMM_NN`
  - `0b110_xxxxx` — magic prefix (`0xc0..0xdf`).
  - `MMM` (bits 3–5): `magnitude = (sizeof(this_sz)) - 1`, value 0..3
    (i.e. each per-frame size is encoded in 1..4 bytes, little-endian).
  - `NN` (bits 0–2): `pending_frame_count - 1`, value 0..7 (up to 8 frames
    per superframe).
- Each `size[i]` is the byte length of the i-th frame in the superframe,
  little-endian, `mag+1` bytes wide.
- Marker byte is repeated at the start AND end of the index so a decoder
  scanning backwards from the packet end can find it.

```rust
fn write_superframe_index(
    out: &mut Vec<u8>,
    frame_sizes: &[u32], // up to 8 frames
) {
    assert!(!frame_sizes.is_empty() && frame_sizes.len() <= 8);
    let max_size = *frame_sizes.iter().max().unwrap();
    let mag = (0..4).find(|&m| max_size < (1u64 << ((m + 1) * 8)) as u32)
                    .unwrap_or(3) as u8;
    let marker = 0xc0u8 | (mag << 3) | (frame_sizes.len() as u8 - 1);
    out.push(marker);
    for &sz in frame_sizes {
        for b in 0..=mag {
            out.push((sz >> (b * 8)) as u8);
        }
    }
    out.push(marker);
}
```

**Critical edge case** (A4 §3.5, line 692-693): when the last byte of any
inner frame's bitstream is in the range `0xc0..0xdf`, `vpx_stop_encode`
appends a `0` byte to disambiguate from the superframe-index marker. rvp9
mirrors this exactly in the encoder bitstream-flush — see HLD-arch / A4
§3.3 step 12.

### 5.2 Per-frame bitstream invariants

Each spatial layer encodes as a complete VP9 frame (uncompressed header +
compressed header + tile data; A4 §3). Only the first frame of a superframe
has `show_frame = 1`; for `MODE_0101` and `MODE_0212`, every spatial layer's
TL=0 frame in the same superframe has `show_frame` set by the caller's
layer pattern — typically only the *top* spatial layer's frame is shown,
lower spatial layers are `show_frame = 0` (their pixels are only used for
inter-layer prediction).

For simulcast mode (`disable_inter_layer_pred = Off` and base spatial is
key, §2.4 step 2), every spatial layer is `show_frame = 1` because each
layer is an independently displayable stream.

### 5.3 Layer-ID metadata is OUT-OF-BAND

VP9's bitstream proper does **not** carry `(spatial_layer_id,
temporal_layer_id)`. WebRTC/RTP-level carriers (RFC 9628) add a payload
descriptor for this. The libvpx encoder exposes the layer IDs via
`vpx_codec_cx_pkt_t` callback fields and `vpx_svc_layer_id_t`
(`vp8cx.h:868`); the bitstream itself is layer-agnostic.

rvp9 mirrors this in the public API: `EncodedFrame` carries `spatial_id`
and `temporal_id` as Rust fields next to the encoded byte buffer, and
caller (e.g. WebRTC adapter) embeds them in RTP descriptors.

### 5.4 Dropped-frame semantics in the wire format

When a single spatial layer drops within a superframe (LAYER_DROP /
CONSTRAINED_LAYER_DROP), its byte size in the index is `0` and the frame
data is absent. Decoders parse the index, see `size[i] == 0`, and skip the
frame. The reference-slot state for that slot is **not refreshed**, so
upper layers in the same superframe see whatever was in the slot from a
prior superframe.

When the **entire** superframe drops (FULL_SUPERFRAME_DROP, §6.2), no
superframe is written at all — the packet is empty / not emitted.

---

## 6. SVC frame-drop semantics

Resolves **L-69** (cross-layer frame-drop) and **L-75** (frame-drop chain).

### 6.1 Drop modes

Mirroring `SVC_LAYER_DROP_MODE` (`vpx/vp8cx.h:929-936`):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDropMode {
    /// Default. If a spatial layer drops, all upper layers in the same
    /// superframe are constrained to drop too.
    ConstrainedLayerDrop,
    /// Any spatial layer can drop independently; upper layers continue.
    LayerDrop,
    /// Only the whole superframe (all layers) drops, or none.
    FullSuperframeDrop,
    /// If a spatial layer drops, all *lower* layers must drop too
    /// (rare, used for some bandwidth-allocation strategies).
    ConstrainedFromAboveDrop,
}
```

Libvpx default is `ConstrainedLayerDrop`
(`vp9_svc_layercontext.c:51`). `VP9RateControlRTC` (the WebRTC interface)
forces `FullSuperframeDrop` (`ratectrl_rtc.cc:181`).

### 6.2 Drop decision (per-frame, per-layer)

The libvpx flow (A5 §4.8 + `vp9_ratectrl.c:614-664`) is:

```text
fn rc_drop_frame(svc, sl) -> bool:
    let prev_dropped = sl > 0 && svc.drop_spatial_layer[sl - 1] != 0

    // Mode-specific cross-layer constraint
    if prev_dropped {
        match svc.framedrop_mode {
            ConstrainedLayerDrop => return true,    // upper drops if lower did
            FullSuperframeDrop   => return true,
            LayerDrop            => /* no constraint */ {}
            ConstrainedFromAboveDrop => /* lower-drops-from-upper handled
                                          on a second pass; see step 4 */ {}
        }
    }

    if svc.framedrop_thresh[sl] == 0 { return false; }
    if rc.buffer_level < 0           { return true;  }

    // Decimation-based drop (A5 §4.8 step 1)
    let drop_mark = svc.framedrop_thresh[sl]
                   * rc.optimal_buffer_level / 100;
    if rc.buffer_level <= drop_mark {
        rc.decimation_factor = max(rc.decimation_factor, 1);
    } else if rc.buffer_level > drop_mark {
        rc.decimation_factor = rc.decimation_factor.saturating_sub(1);
    }
    if rc.decimation_factor > 0 {
        rc.decimation_count += 1;
        if rc.decimation_count >= rc.decimation_factor + 1 {
            rc.decimation_count = 0;
            return true;
        }
    }
    false
}
```

When a layer drops, the encoder calls
`vp9_rc_postencode_update_drop_frame` (`vp9_ratectrl.c:1998-2017`) which:

- Increments `current_video_frame`, `frames_since_key`; decrements
  `frames_to_key`.
- Clears `rc_1_frame`/`rc_2_frame` (overshoot/undershoot oscillation
  tracker).
- Sets `last_q[INTER_FRAME] = base_qindex` (so the next frame's Q baseline
  is preserved).
- For SVC non-`LAYER_DROP`: if `buffer_level > optimal`, clip back to
  optimal so the upper-layer buffer doesn't grow unboundedly when the whole
  superframe drops.

### 6.3 Drop bookkeeping

Two flags per layer (mirroring `vp9_svc_layercontext.h`):

- `svc.drop_spatial_layer[sl]: bool` — set when this superframe's layer
  `sl` dropped. Read by upper layers in the same superframe.
- `svc.last_layer_dropped[sl]: bool` — set when the *previous* superframe's
  layer `sl` dropped. Read by the next superframe's `start_layer` to
  redirect references (§6.4).

Plus a global cap: `svc.max_consec_drop` (default `INT_MAX` for libvpx,
explicitly set per-config for WebRTC). Tracked via
`svc.drop_count[sl]` per spatial layer; if `drop_count[sl] >= max_consec_drop`,
force-encode (no drop) regardless of buffer.

### 6.4 Reference-state after a drop

This is the **L-75 RESOLUTION** (drop chain semantics):

1. **Same-superframe upper layer**: when `sl-1` drops and we're in
   `ConstrainedLayerDrop` or `FullSuperframeDrop` mode, `sl` is forced to
   drop too. No reference-state changes.

2. **Same-superframe upper layer in `LayerDrop` mode**: `sl-1` dropped but
   `sl` continues. `start_layer` step 5 (inter-layer prediction constraint)
   clears any ref flag that points at a slot the dropped `sl-1` would have
   refreshed. For typical TL=0 inter frames, GOLDEN points at slot `sl-1`
   for inter-layer prediction (§2.2); since slot `sl-1` was not refreshed
   this superframe, GOLDEN now points at slot `sl-1`'s *previous*
   superframe content. libvpx accepts this (the slot was last refreshed by
   an older valid frame); rvp9 mirrors.

3. **Next-superframe same-layer**: `svc.last_layer_dropped[sl] = true`.
   In `start_layer` for the next superframe (`vp9_svc_layercontext.c:979-1024`):
   - If `last_layer_dropped[sl]` AND `fb_idx_upd_tl0[sl] != -1` (we know
     which slot the last successfully-encoded TL0 of layer `sl` lives in),
     redirect `lst_fb_idx[sl] = fb_idx_upd_tl0[sl]`. This skips over the
     dropped frame back to the last valid same-spatial-layer reference.
   - Disable `use_base_mv` and `use_partition_reuse` for the current
     spatial layer (`vp9_svc_layercontext.c:956-960`).
   - If this layer's `LayerContext.is_key_frame` is set for the new
     superframe, force a fresh keyframe (not a reference jump).

4. **GoP-level effects (D-13 simplification)**: since we have no altref,
   the drop-chain is a single linear chain per spatial layer (TL0 →
   recovers TL1 → recovers TL2). With altref this would also need to track
   recovery through `alt_ref_source` per layer. Removing altref reduces the
   state-machine surface materially — see §9.

### 6.5 `FullSuperframeDrop` × WebRTC

The libwebrtc adapter forces `FullSuperframeDrop` because RTP/RTCP relies
on superframes being atomic units. rvp9's EXT_RATECTRL surface (§7) reports
drop decisions at the *superframe* granularity in this mode: a single
`FrameDrop::Drop` from `compute_qp` discards all spatial layers for the
current input.

For the encoder-internal modes (`ConstrainedLayerDrop`, `LayerDrop`,
`ConstrainedFromAboveDrop`), the drop decision is *per spatial layer*; the
EXT_RATECTRL surface returns a per-spatial-layer drop verdict.

---

## 7. EXT_RATECTRL plug-in interface (R-06)

The libvpx world has **two** external-RC interfaces:

1. **`vpx_ext_ratectrl.h`** (`vpx/vpx_ext_ratectrl.h:511-552`) — generic
   plug-in for libvpx's 2-pass + RDO encoder. Targets offline tooling, not
   WebRTC.
2. **`VP9RateControlRTC`** (`vp9/ratectrl_rtc.h:87-111`) — the
   *standalone* RC engine used by libwebrtc. It mirrors libvpx's CBR
   real-time RC logic but is decoupled from the encoder pipeline so the
   caller can use the encoder's RC decisions without instantiating the
   encoder.

PLAN.md §2 / R-06 RESOLVED commit to **EXT_RATECTRL plug-in IN v1** for
libwebrtc integration. The relevant target is **interface (2)**, because
that's what libwebrtc actually links against. We also provide a thin
in-process "callback" trait modelled after interface (1) so the same
mechanism can drive offline tooling later.

### 7.1 `RateControlRtc` standalone RC engine

Rust struct mirroring `VP9RateControlRTC` (`vp9/ratectrl_rtc.h:87-111`):

```rust
// crates/rvp9-encoder/src/ratectrl/rtc.rs
pub struct RateControlRtc { inner: Box<EncoderState> }
//   ^ a stripped-down EncoderState with only RateControl + SVC +
//     CyclicRefresh fields populated; no buffer pool, no tile workers,
//     no actual encode.

#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum FrameDropDecision { Ok, Drop }
#[derive(Debug, Clone, Copy)] pub enum RcFrameType { KeyFrame, InterFrame }

#[derive(Debug, Clone)] pub struct VP9FrameParamsQpRtc {
    pub frame_type:       RcFrameType,
    pub spatial_layer_id: u8,
    pub temporal_layer_id: u8,
}

// Mirror of VpxRateControlRtcConfig + VP9RateControlRtcConfig
// (vpx/internal/vpx_ratectrl_rtc.h:25-72, vp9/ratectrl_rtc.h:27-52).
#[derive(Debug, Clone)] pub struct VP9RateControlRtcConfig {
    pub width: u32, pub height: u32,
    pub max_quantizer: u8, pub min_quantizer: u8,
    pub target_bandwidth_kbps: u32,
    pub buf_initial_sz_ms: u32, pub buf_optimal_sz_ms: u32, pub buf_sz_ms: u32,
    pub undershoot_pct: u8, pub overshoot_pct: u8,
    pub max_intra_bitrate_pct: u8, pub max_inter_bitrate_pct: u8,
    pub framerate: f64,
    pub ts_number_layers: u8,   // 1..=3
    pub ss_number_layers: u8,   // 1..=3
    pub layer_target_bitrate_kbps: [u32; MAX_LAYERS],
    pub ts_rate_decimator:         [u8;  MAX_TEMPORAL_LAYERS],
    pub max_quantizers:            [u8;  MAX_LAYERS],
    pub min_quantizers:            [u8;  MAX_LAYERS],
    pub scaling_factor_num:        [u8;  MAX_SPATIAL_LAYERS],
    pub scaling_factor_den:        [u8;  MAX_SPATIAL_LAYERS],
    pub rc_mode: RcMode,            // CBR | VBR (no CQ in the RTC API)
    pub aq_mode: AqMode,            // None | CyclicRefresh
    pub frame_drop_thresh: u32,
    pub max_consec_drop:   u32,
    pub is_screen: bool,
}

impl RateControlRtc {
    pub fn create(cfg: &VP9RateControlRtcConfig) -> Option<Self>;
    pub fn update_rate_control(&mut self, cfg: &VP9RateControlRtcConfig) -> bool;
    /// May return Drop, in which case caller MUST NOT call get_qp /
    /// post_encode_update — engine has already run drop bookkeeping.
    pub fn compute_qp(&mut self, params: &VP9FrameParamsQpRtc) -> FrameDropDecision;
    pub fn get_qp(&self) -> u8;
    pub fn get_loopfilter_level(&self) -> u8;
    pub fn get_segmentation_data(&self) -> Option<SegmentationData<'_>>;
    pub fn post_encode_update(&mut self, encoded_bits: u64,
                              params: &VP9FrameParamsQpRtc);
}

pub struct SegmentationData<'a> {
    pub segmentation_map: &'a [u8],
    pub delta_q:          &'a [i8],
}
```

Semantics 1:1 with `VP9RateControlRTC::ComputeQP` / `PostEncodeUpdate`
(`vp9/ratectrl_rtc.cc:233-285`). Internal state reuses the same
`RateControl` + `SVC` + `CyclicRefresh` structs as the full encoder; the
standalone engine just skips the actual frame encode.

### 7.2 In-process `ExternalRateCtrl` trait

For deeper integration (e.g. an offline experiment that wants to override
the encoder's per-frame Q decisions during a real encode), we offer:

```rust
// crates/rvp9-encoder/src/ratectrl/ext.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum RcStatus { Ok, Error }
#[derive(Debug, Clone, Copy)] pub enum RcType { Qp, Gop, Rdmult, GopQp, GopQpRdmult }

#[derive(Debug, Clone)] pub struct EncodeFrameInfo {
    pub frame_type:    FrameType,            // Key / Inter / Altref(unused, v1)
    pub show_index:    u32,
    pub coding_index:  u32,
    pub gop_index:     u32,
    pub ref_frame_coding_indexes: [u32; 3],  // LAST / GOLDEN / ALTREF
    pub ref_frame_valid_list:     [bool; 3],
    pub gop_size:      u32,
    pub use_alt_ref:   bool,                  // always false (D-13)
    // rvp9 additions over libvpx's vpx_rc_encodeframe_info_t:
    pub spatial_layer_id:  u8,
    pub temporal_layer_id: u8,
}

#[derive(Debug, Clone, Copy)] pub struct EncodeFrameDecision {
    pub q_index:        Option<u8>,            // None → use encoder default
    pub max_frame_size: MaxFrameSize,
}

#[derive(Debug, Clone, Copy)] pub enum MaxFrameSize {
    NoLimit, VP9Default, Bits(u32),
    // ^^^ map to libvpx's max_frame_size == 0 / -1 / N respectively.
}

#[derive(Debug, Clone)] pub struct EncodeFrameResult {
    pub sse: i64, pub bit_count: i64, pub pixel_count: i64,
    pub actual_encoding_qindex: u8,
}

/// Mirrors vpx_rc_funcs_t (vpx/vpx_ext_ratectrl.h:511-552) using a Rust
/// trait instead of raw fn-pointer table.
pub trait ExternalRateCtrl: Send + 'static {
    fn rc_type(&self) -> RcType;
    fn on_init(&mut self, cfg: &RcConfig) -> RcStatus;
    /// q_index = None → fall back to encoder's internal RC pick.
    fn get_encode_frame_decision(&mut self, info: &EncodeFrameInfo)
        -> EncodeFrameDecision;
    fn update_encode_frame_result(&mut self, result: &EncodeFrameResult);
    fn on_teardown(&mut self) {}
}

// First-pass stats, TPL stats, and GOP-decision callbacks from
// vpx_ext_ratectrl.h are NOT in v1 (2-pass + altref features; D-13 OOS).
```

### 7.3 How SVC layer info flows through

The crucial difference from libvpx's `vpx_rc_encodeframe_info_t` is that
rvp9's `EncodeFrameInfo` carries `(spatial_layer_id, temporal_layer_id)`.
This addresses the L-77 gap: an external RC model needs to know which
layer it's picking Q for.

Calling sequence per spatial layer (extending §2.3):

```rust
fn encode_one_layer(&mut self, frame: &SourceFrame) -> Result<EncodedFrame, _> {
    let info = self.build_encode_frame_info();         // §7.2

    // Internal RC pick (mirrors set_size_dependent_vars / pick_q):
    let internal_q = self.rc.pick_q_and_bounds(&self.svc);

    // External RC consult, if installed:
    let final_q = if let Some(ext) = &mut self.ext_rc {
        let decision = ext.get_encode_frame_decision(&info);
        match decision.q_index {
            None    => internal_q,
            Some(q) => q,
        }
    } else {
        internal_q
    };
    // (max_frame_size honoured via the recode-loop budget — but D-13
    // implies no recode in real-time anyway; we ignore Bits(_) in v1
    // and document this in the divergence ledger.)

    let encoded = self.encode_frame_with_q(frame, final_q)?;

    if let Some(ext) = &mut self.ext_rc {
        ext.update_encode_frame_result(&EncodeFrameResult {
            sse:           encoded.sse,
            bit_count:     (encoded.bytes.len() * 8) as i64,
            pixel_count:   (self.cm.width * self.cm.height * 3 / 2) as i64,
            actual_encoding_qindex: final_q,
        });
    }

    Ok(encoded)
}
```

### 7.4 Frame-drop decisions in EXT_RATECTRL

The libvpx `vpx_rc_encodeframe_decision_t` does **not** have a "drop"
flag. Frame-drop is the encoder's call, based on internal RC; the external
model only influences Q and (in offline mode) GOP structure.

By contrast, `VP9RateControlRTC::ComputeQP` *can* return `kDrop`
(`vp9/ratectrl_rtc.h:102`). rvp9 v1 follows `VP9RateControlRTC` for the
standalone engine (§7.1) and follows `vpx_ext_ratectrl` for the in-process
trait (§7.2): external models can't override drop in the trait path
because that introduces a re-entrancy hole into the RC state machine
(`vp9_rc_postencode_update_drop_frame` mutates buffer state and must run
before the next frame's `pick_q`).

If a libwebrtc-style caller needs drop control, they use
`RateControlRtc::compute_qp` (§7.1) and the encoder's `Encoder::encode`
accepts an explicit `FrameDropDecision` override (see §8).

### 7.5 Threading

EXT_RATECTRL callbacks run on the **main thread**, between frame encodes,
*not* inside tile workers. The `ExternalRateCtrl` trait does not need to be
`Sync` (only `Send`), which simplifies caller integration. This matches
libvpx — `vpx_rc_funcs_t::get_encodeframe_decision` is called from
`vp9_encoder.c` outside the worker dispatch.

For `RateControlRtc` (§7.1), the engine is fully sequential by design —
no worker pool, no aliasing.

### 7.6 Versioning and ABI

libvpx exposes `VPX_EXT_RATECTRL_ABI_VERSION` (currently `7`,
`vpx/vpx_ext_ratectrl.h:29`). rvp9 ships Rust traits with the usual
`#[non_exhaustive]` annotations on `EncodeFrameInfo`, `RcConfig`, etc. so
that new fields can be added without breaking downstream `impl`s. Semantic
versioning of `rvp9-encoder` covers ABI evolution; no separate ABI version
constant is needed.

---

## 8. Encoder API surface for SVC

### 8.1 Configuration

`Encoder::new` accepts an `EncoderConfig` whose SVC fields are:

```rust
pub struct SvcConfig {
    pub number_spatial_layers:  u8,    // 1..=3
    pub number_temporal_layers: u8,    // 1..=3
    pub temporal_layering:      TemporalLayering,    // §2.1
    pub inter_layer_pred:       InterLayerPred,      // §3.5
    pub frame_drop_mode:        FrameDropMode,       // §6.1
    pub use_gf_temporal_ref:    bool,                // long-term GF ref (§2.2)
    pub spatial_scaling: [(u8, u8); MAX_SPATIAL_LAYERS], // (num, den) per layer
    pub layer_target_kbps: [u32; MAX_LAYERS],
    pub layer_min_q:  [u8; MAX_LAYERS],
    pub layer_max_q:  [u8; MAX_LAYERS],
    pub framedrop_thresh: [u8; MAX_SPATIAL_LAYERS],   // 0..=100
    pub max_consec_drop:  u32,
    pub first_spatial_layer_to_encode: u8,
    pub speed_per_layer: [u8; MAX_SPATIAL_LAYERS],    // cpu-used 5..=8 (D1)
}
```

Validation at `Encoder::new`:

- `number_spatial_layers ∈ 1..=3` (PLAN.md §2 + §1.2 above).
- `number_temporal_layers ∈ 1..=3`.
- `(temporal_layering, number_temporal_layers)` pair is consistent
  (`NoLayering ↔ ntl == 1`, `Mode0101 ↔ ntl == 2`, `Mode0212 ↔ ntl == 3`).
- `spatial_scaling[0..number_spatial_layers]` strictly increasing toward
  1:1, all denominators in `{1, 2, 4}`.
- `layer_target_kbps` non-zero for each enabled (sl, tl).
- `framedrop_thresh[sl] ∈ 0..=100`.

### 8.2 Per-frame call

```rust
pub struct FrameOptions {
    pub pts: i64, pub duration: u32,
    pub force_keyframe: bool, pub end_of_stream: bool,
    /// REQUIRED in Bypass temporal-layering mode; None in fixed modes
    /// (0101 / 0212 / NoLayering).
    pub ref_frame_config: Option<RefFrameConfig>,
    /// External drop override (e.g. libwebrtc bandwidth probing).
    pub forced_drop: Option<ForcedDrop>,
    /// Bandwidth-constrained fast path (§8.4).
    pub skip_enhancement_layers: bool,
}

pub struct RefFrameConfig {
    pub temporal_layer_id: u8,
    pub reference_last:    [bool; MAX_SPATIAL_LAYERS],
    pub reference_golden:  [bool; MAX_SPATIAL_LAYERS],
    pub reference_altref:  [bool; MAX_SPATIAL_LAYERS],
    pub lst_fb_idx:        [u8;   MAX_SPATIAL_LAYERS],
    pub gld_fb_idx:        [u8;   MAX_SPATIAL_LAYERS],
    pub alt_fb_idx:        [u8;   MAX_SPATIAL_LAYERS],
    pub update_buffer_slot:[u8;   MAX_SPATIAL_LAYERS], // 8-slot bitmask
    pub duration:          [i64;  MAX_SPATIAL_LAYERS],
}

pub enum ForcedDrop {
    DropEntireSuperframe,
    DropSpatialLayers(u8),    // bitmask
}

pub fn encode(&mut self, frame: &SourceFrame, opts: &FrameOptions)
    -> Result<EncodedSuperframe, EncodeError>;
```

`EncodedSuperframe` exposes the packed superframe bytes plus per-layer
metadata:

```rust
pub struct EncodedSuperframe {
    pub bytes:    Vec<u8>,
    pub layers:   Vec<EncodedLayerInfo>,
    pub pts:      i64,
    pub duration: u32,
}

pub struct EncodedLayerInfo {
    pub spatial_id:  u8,
    pub temporal_id: u8,
    pub was_dropped: bool,
    pub frame_type:  FrameType,         // Key | Inter | IntraOnly
    pub qindex:      u8,
    pub size_bytes:  u32,                // 0 if dropped
    pub byte_range:  std::ops::Range<usize>, // slice into bytes[]
}
```

### 8.3 Frame-by-frame layer-selection hooks

Three integration models, in increasing flexibility:

1. **Fixed pattern**: caller sets `TemporalLayering::Mode0101 |
   Mode0212` once at `Encoder::new`. Each `encode()` call automatically
   walks all spatial layers and emits per-layer frames per the pattern.
   `FrameOptions::ref_frame_config = None`.

2. **Bypass mode**: caller sets `TemporalLayering::Bypass`. Each
   `encode()` call must include `FrameOptions::ref_frame_config = Some(_)`.
   The encoder uses the caller's `lst_fb_idx / gld_fb_idx / alt_fb_idx /
   reference_* / update_buffer_slot / temporal_layer_id` verbatim. This is
   the WebRTC integration mode.

3. **External RC plug-in**: caller installs an `impl ExternalRateCtrl`
   via `Encoder::set_external_rate_ctrl(...)`. The encoder calls
   `get_encode_frame_decision` per spatial layer per encode; in fixed-pattern
   mode the layer assignment is still automatic. This is the offline-tooling
   model.

Models 2 and 3 can be combined.

### 8.4 Skip-enhancement-layer fast path

libvpx exposes `svc.skip_enhancement_layer`
(`vp9_svc_layercontext.h:111`) which, when set, skips encoding all spatial
layers above the base for one superframe. rvp9 mirrors this via the
`FrameOptions::skip_enhancement_layers` flag above — useful for
bandwidth-constrained moments without going through the full drop
mechanism.

---

## 9. No-altref simplification (D-13 RESOLVED)

D-13 RESOLVED to **CBR + VBR-RT, NO altref**. This collapses several SVC
code paths in libvpx that we won't port:

### 9.1 Removed: `vp9_temporal_filter_*` integration

libvpx's `vp9_temporal_filter.c` (~1200 LoC) runs only when
`lag_in_frames > 0` and produces the synthetic altref frame. With
`lag_in_frames = 0` enforced in our config, this whole module is unreachable
and is dropped from v1.

Effect on SVC: `svc.scaled_frames[MAX_LAG_BUFFERS]`
(`vp9_svc_layercontext.h:91-93`) was used to hold scaled source frames for
the temporal filter's accumulator. With no altref, only the *current*
superframe needs scaled sources, so `scaled_frames` becomes a 1-element
buffer (just `scaled_frames[0]`) plus `scaled_temp` for 2-stage
downsampling.

### 9.2 Removed: `vp9_one_pass_svc_alt_ref` insertion

libvpx inserts a synthetic altref between SVC superframes for
`MODE_0212` configurations with lag (`vp9_svc_layercontext.c:817-831` —
the `VPX_RC_QP` plus `lag_in_frames > 0` branch). Our path is the
`lag_in_frames == 0` branch only — never insert altref.

Consequence: `LayerContext.has_alt_frame`, `LayerContext.alt_ref_source`,
`LayerContext.alt_ref_idx` (`vp9_svc_layercontext.h:55-58`) are all
**removed** from rvp9's `LayerContext` representation (§2.2). Slots that
libvpx uses for ALTREF refresh (slot 2 for `alt_fb_idx[0] = 2` default;
slots `NSL..2*NSL-1` for `MODE_0212` TL=1 ALTREF refresh, §2.2) are still
used — but they hold **regular inter frames**, not synthetic-altref
output. The reference-slot dispatch tables in §2.2 still allocate
`alt_fb_idx` and the bitstream still writes `refresh_alt_ref_frame = 1`
when the pattern requires it; we just don't synthesize anything special
for those frames.

### 9.3 Removed: `use_altref_onepass` and `use_compound_nonrd_pickmode`

`SPEED_FEATURES::use_altref_onepass` (`vp9_speed_features.h:589`) is set in
the cpu-used 4–6 cascades only when `VBR + lag_in_frames > 0`
(`vp9_speed_features.c:559-561, 649-651, 663-665`). With D-13 RESOLVED
these branches never fire; rvp9 sets `use_altref_onepass = false` at
speed-feature init and never updates it. Document in divergence ledger as
an *unreachable branch*, not a behavioural divergence.

`use_compound_nonrd_pickmode` has identical gating
(`vp9_speed_features.c:561, 651, 665`); compound prediction in non-RD mode
is gated on altref availability. A2's **L-46** ("compound mode handling
at speed 6+") gets a concrete answer here: **compound is removed from the
non-RD path in v1.**

### 9.4 Removed: `is_src_frame_alt_ref` per-layer state

`LAYER_CONTEXT.alt_ref_source` and the `rc.is_src_frame_alt_ref` toggle
disappear; downstream branches on `rc.is_src_frame_alt_ref` (e.g.
`vp9_speed_features.c:651` for `VAR_BASED_PARTITION` override,
`vp9_ratectrl.c:1227-1233` for the `{0.50,1.0,...}` 8-frame GF rate
pattern) reduce to the `== false` branch.

### 9.5 SVC simplification summary

The no-altref decision removes ~1500 LoC of libvpx surface area. Logged in
the divergence ledger as a scope reduction, not a behavioural divergence
(libvpx's no-lag, no-altref real-time configurations produce identical
output to ours by construction).

---

## 10. Open questions and cross-HLD dependencies

Items where this HLD takes a position but defers detailed resolution to
the LLDs, or where a peer HLD must align:

### 10.1 Cross-HLD assumptions (best guesses, await peer review)

| Assumption | Owner if wrong |
|---|---|
| Tile threading uses `std::thread::scope` (A6 §7.3 recommendation; L-56) | HLD-01 (threading) |
| Per-worker `SPEED_FEATURES` snapshot via memcpy at frame start (A6 §7.1; D-10) | HLD-01 |
| Source scaling kernel dispatch lives in shared `rvp9-core` RTCD | HLD-02 (SIMD) |
| `EncoderState` is a single owning struct, not multiple-Arc | HLD-00 (arch) |
| `RateControl` and `LayerContext.rc` are plain types, value-copied on swap (no internal `Box`/`Arc`) | HLD-rc (LLD) |
| Buffer-pool is `Arc<FrameBuf>` per slot, with `Arc::strong_count` mirroring libvpx ref-count semantics | HLD-00 + L-37 |

If any of these assumptions are wrong, the SVC implementation can adapt:
the `Encoder` method names + state-transition design in §2.3 don't depend
on the threading model or the SIMD layout. The only hard dependency is
that **layer iteration is single-threaded** (§2.3) — peer HLDs must
preserve that.

### 10.2 LLD-bound questions (deferred to `notes/lld/svc.md` and friends)

| ID | Topic | LLD |
|---|---|---|
| L-58 | Layer-context swap exact field ordering (which `RateControl` fields are *NOT* swapped?) | LLD-rc + LLD-svc |
| L-67 | Cyclic-refresh segmentation map ownership: per-spatial-layer-base-tl0 vs encoder-global | LLD-aq + LLD-svc |
| L-68 | `vp9_scene_detection_onepass` per-layer behaviour (§4.3 §5 position: base-spatial-base-temporal only) | LLD-rc |
| L-69 | SVC cross-layer frame-drop interactions (§6 position locked) | LLD-svc + LLD-rc |
| L-75 | SVC frame-drop chain semantics (§6.4 position locked; finer-grained recovery rules deferred) | LLD-svc |
| L-77 | `temporal_layering × per-layer RC` (§4.3 position locked) | LLD-rc + LLD-svc |
| L-36 | 8-slot reference pool layout across spatial+temporal layers (§2.2 position locked) | LLD-frame-buffers |
| L-61 | Worker pool sized to top-spatial-layer max tile cols (A6 §7.11) | HLD-01 |

### 10.3 Open questions for Arthur

1. **Bypass mode validation strictness**: should rvp9 *strictly* validate
   the caller's `RefFrameConfig` (reject configs where `lst_fb_idx[sl]`
   points at a slot that wasn't refreshed in a prior superframe) or
   accept silently like libvpx? **Position**: strict-by-default with an
   opt-out for "I know what I'm doing" mode; defaults to safer behaviour
   for libwebrtc integration.

2. **`MAX_LAYERS` constants**: PLAN.md says 2- and 3-layer. Should the
   compile-time constant be `3` (tight) or `5` (matches libvpx)? **Position**:
   tighten to 3 in v1 (`MAX_SPATIAL_LAYERS = 3`, `MAX_TEMPORAL_LAYERS = 3`,
   `MAX_LAYERS = 9`). Reject configs beyond this. Costs ~zero memory, gains
   compile-time bounds checks. Document in divergence ledger.

3. **`compute_qp` re-entrancy from `EXT_RATECTRL`**: should an
   `ExternalRateCtrl` implementor be allowed to call back into the
   encoder during `get_encode_frame_decision`? **Position**: no.
   `&mut self` discipline in Rust prevents this anyway, but document
   explicitly. The standalone `RateControlRtc` (§7.1) does not have an
   encoder, so this is moot for the WebRTC integration path.

4. **External Q clamping**: if `ExternalRateCtrl` returns `q_index = Some(0)`
   (lossless) for a CBR stream, do we honour it or clamp to
   `LayerContext.min_q..max_q`? **Position**: clamp to layer min/max,
   warn via a callback or `tracing::warn!`. Out-of-range Q breaks buffer
   model assumptions.

---

## 11. Test strategy

### 11.1 Conformance (decoder side)

Stage 3 work, not Stage 4d/4e (this HLD's territory). The VP9 conformance
vectors include SVC test cases (`vp90-2-22-svc_*` family in the AOM test
set). rvp9's decoder MUST decode those identically to libvpx; this is the
default-pass criterion before any encoder SVC tests are meaningful.

### 11.2 Per-layer extraction harness

`tests/svc/extract-layers.rs`: take an `EncodedSuperframe`, parse the
superframe index (A4 §3.4 parser), and for each `sl ∈ 0..NSL` extract a
sub-bitstream containing only frames with `spatial_id ≤ sl`. Pass through
libvpx's `vpxdec`, assert decode success, capture per-frame MD5, compare
against a layer extracted from a libvpx-encoded reference at matched
config. This is the **primary BD-rate-per-layer test fixture** (Stage 4d
/ 4e exit criterion).

### 11.3 BD-rate per layer

`bench/bdrate-svc.rs`: encode the test corpus at a rate-sweep
(200..2000 kbps across layers proportionally). For each (operating point,
spatial layer) pair, extract the per-layer sub-bitstream (§11.2), decode
to YUV, compare against the original at the layer's spatial resolution,
and compute BD-rate per layer vs the libvpx reference encode. Pass: BD-rate
within 0.5% per layer.

### 11.4 Differential: rvp9-vs-libvpx superframe bytes

`tests/differential/diff-svc-bitstream.rs`:

- Encode the same input with both rvp9 and libvpx at matched config.
- Diff the superframe-index bytes byte-by-byte. These should match
  exactly (the index is purely structural: marker, magnitude, per-frame
  size LE32; no entropy state involved).
- Diff the inner per-frame bitstreams. Divergence here is acceptable
  (RC differences are documented divergences from libvpx); diff is logged
  as a *diagnostic*, not a gate.

### 11.5 Frame-drop chain regression

`tests/svc/drop-chain.rs`:

- For each of the four `FrameDropMode` values, encode a clip with a
  forced bandwidth squeeze halfway through.
- Assert:
  - `ConstrainedLayerDrop`: when layer 0 drops, layers 1 and 2 also drop
    in the same superframe.
  - `LayerDrop`: layer 0 dropping does not prevent layer 1 from being
    encoded (subject to ref-availability constraints).
  - `FullSuperframeDrop`: every drop is whole-superframe; per-layer
    selective drop never happens.
  - `ConstrainedFromAboveDrop`: when layer 2 drops, layers 0 and 1 also
    drop in the same superframe.
- For each mode, assert that the next superframe's reference state is
  consistent: no frame references a slot that has never been refreshed.

### 11.6 EXT_RATECTRL conformance

`tests/svc/ext-ratectrl.rs`: install a stub `impl ExternalRateCtrl` that
records every `EncodeFrameInfo` and returns `q_index = None`. Encode a
3-layer SVC clip. Assert one callback per (spatial, temporal) per frame,
`spatial_layer_id`/`temporal_layer_id` consistent with the pattern, and
`ref_frame_coding_indexes` matching chosen `lst/gld/alt_fb_idx`.

`tests/svc/ratectrl-rtc.rs`: drive `RateControlRtc` through a sequence of
`compute_qp` / `post_encode_update` calls. Run libvpx's `VP9RateControlRTC`
(via FFI) through the same sequence. Assert: for every
`(frame_type, sl, tl, encoded_size)` input, the QP / loopfilter level /
drop decision matches **byte-for-byte**. This is a hard byte-parity test
— the RTC API is the libwebrtc contract; any divergence breaks WebRTC
compatibility.

### 11.7 Cross-layer reference-slot integrity

`tests/svc/ref-slot-integrity.rs`:

- For each `(TemporalLayering, NSL)` pair in scope, encode 8 superframes
  (= 2 full 4-frame periods for `Mode0212`).
- For each frame, parse the uncompressed header's `ref_frame_idx[]` and
  assert that the slot pointed to was either:
  - the initial keyframe slot (slot 0 first refreshed at superframe 0), or
  - refreshed in a prior superframe at a (spatial, temporal) cell that
    matches §2.2.
- Catches off-by-one and Mode0212-second-TL2 confusion early.

### 11.8 Speed-feature golden file for SVC paths

Per A6 §7.14 (L-64), capture `SPEED_FEATURES` from libvpx for SVC-enabled
configurations specifically:

- `(cpu_used, frame_type, (w,h), use_svc=true, sl, tl, ntl, nsl, content_type, rc_mode)`
- Assert rvp9's speed-feature setter produces byte-equal output.

The SVC paths exercise these speed-feature toggles that don't appear in
non-SVC paths:

- `sf->reference_masking = (svc->number_spatial_layers == 1)`
  (`vp9_speed_features.c:521`).
- `sf->nonrd_keyframe = (svc->number_spatial_layers > 1 && !svc->simulcast_mode) ? 0 : 1`
  (cpu-used 8, `vp9_speed_features.c:761-766`).
- `sf->mv.subpel_search_method = SUBPEL_TREE_PRUNED_EVENMORE` for SVC
  non-ref enhancement layers (cpu-used 7, `vp9_speed_features.c:710-712`).
- `sf->svc_use_lowres_part = 1` for 3-spatial-layer + TL>0 + >640×480
  (cpu-used 7, `vp9_speed_features.c:735-739`).
- `sf->base_mv_aggressive = 1` for SVC TL>0 (cpu-used 6,
  `vp9_speed_features.c:686-689`).

---

## 12. Summary and exit-criteria mapping

### 12.1 What this HLD locks down

- **§2** — Temporal SVC: three fixed modes + Bypass, libvpx-faithful
  reference-slot mapping per (sl, tl), no-altref simplification.
- **§3** — Spatial SVC: per-layer down-scale pipeline, inter-layer texture
  via the GOLDEN-slot remap trick, MV+partition reuse for 1:2 scales.
- **§4** — Cross-layer RC: per-(sl, tl) `RateControl`, single CR map per
  spatial layer at TL=0, scene-detect only on base spatial + base
  temporal. **L-77 RESOLVED.**
- **§5** — Bitstream: superframe index format, per-frame is plain VP9,
  layer IDs are out-of-band.
- **§6** — Frame-drop: four modes; ref-state recovery via
  `fb_idx_upd_tl0[]`; chain semantics codified. **L-69, L-75 RESOLVED.**
- **§7** — EXT_RATECTRL: two surfaces (`RateControlRtc` standalone +
  `ExternalRateCtrl` trait), SVC layer IDs surfaced in `EncodeFrameInfo`.
  **R-06 RESOLVED.**
- **§8** — Public API: `EncoderConfig.svc`, per-frame `RefFrameConfig`,
  bypass + fixed + external integration models.
- **§9** — No-altref simplification documented; ~1500 LoC of libvpx skipped.
- **§11** — Test strategy: per-layer extraction, BD-rate, byte-parity
  for `RateControlRtc`.

### 12.2 Mapping to PLAN.md Stage 4d / 4e exit criteria

Stage 4d: temporal-layer extract → §5.1 + §11.2; BD-rate/layer ≤0.5% → §4 + §11.3; 720p30 + 1080p30 RT with 3-TL → §4.3 + §2. Stage 4e: spatial-layer extract → §3 + §11.2; BD-rate/layer ≤0.5% → §3.2 + §3.3 + §3.4; full-SVC RT → §3.1 + §10.1.

### 12.3 Stage 4 work breakdown driven by this HLD

The LLDs that consume this HLD:

- `notes/lld/svc.md` — full per-module LLD. Will inherit §2, §3, §6 and
  enumerate the exact per-`(sl, tl)` slot map for all three temporal
  modes. Plus the `start_layer` step-by-step, mirroring
  `vp9_one_pass_svc_start_layer` line for line.
- `notes/lld/rate-control.md` — per-layer `RateControl` state machine; the
  `restore_layer_context` / `save_layer_context` field-by-field swap;
  the L-77 fan-out.
- `notes/lld/aq.md` — cyclic-refresh's per-spatial-layer map and the
  TL=0-only update gate.
- `notes/lld/denoiser.md` — `first_layer_denoise` per-layer ownership.
- `notes/lld/ext-ratectrl.md` (new) — the `ExternalRateCtrl` trait,
  `RateControlRtc` standalone engine, byte-parity test plan.
- `notes/lld/bitstream.md` — superframe index packer; verified against
  the A4 §3.4 + `write_superframe_index` source.

---

