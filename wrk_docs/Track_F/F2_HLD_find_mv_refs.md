# F.2 — HLD: Faithful `find_mv_refs` port + flip `use_prev_frame_mvs` default ON

| Field | Value |
|---|---|
| Track | F (nonrd_pickmode port) |
| Sub-track | F.2 (`find_mv_refs` salvage + A18 default-ON) |
| Author | F.2 HLD-writer agent |
| Status | LANDED — infrastructure-only (see §12 ship-state addendum, 2026-05-23) |
| Last revised | 2026-05-23 |
| Repo HEAD at writing | `f1a8327` (post-F.0 foundations; F.1 implementer running in parallel) |
| libvpx HEAD at writing | `12f3a2ac6` (`v1.14.1`) at `D:\language\libvpx` |
| Dependencies | F.0.3 (prev_frame_mvs plumbing, signed-off, code landed), F.1 (`block_yrd`, signed-off, implementer in flight) |
| Blocks | F.3 (motion-aware partition), F.4 (tuning + byte-exact re-establishment) |

**Sign-off:** 2026-05-23 — Track F team-lead approved this HLD with revisions 1–3. Q1–Q5 resolved (see §9).

> **Headline finding from the source survey** (see §3): rvp9's
> `dec_find_mv_refs` at `crates\rvp9-core\src\inter\mv_ref.rs:482-605`
> **already mirrors libvpx's `find_mv_refs_idx`
> (`vp9/common/vp9_mvref_common.c:16-131`) bullet-for-bullet** including
> both salvage passes (matching-ref at libvpx :73-79 = rvp9 :542-553;
> different-ref-with-sign-bias at libvpx :99-122 = rvp9 :586-602). The
> per-leaf lookup wiring at `crates\rvp9-encoder\src\lib.rs:2845-2905`
> is also already correct. **F.2 is therefore primarily a
> default-flip + validation + targeted tests, not a real port.** The
> "port" framing in the kickoff plan is misleading — the real work was
> done before Track F started.

---

## 1. Problem statement & motivation

After F.1 (`block_yrd`) makes coefficient bits visible to the picker's
RD, NEWMV becomes RD-competitive on blocks where its prediction is
genuinely better. But the picker's NEAREST/NEAR anchors still come
from `find_mv_refs`, which today walks only the in-progress mi-grid's
**spatial** neighbours.

After a keyframe every neighbour in the first inter frame carries the
default `(ZEROMV, LASTFRAME)` — nothing has been picked yet — so
`find_mv_refs` returns `(0, 0)` for every candidate. NEAREST/NEAR/NEW
all evaluate against the same all-zero anchor; the strict-`<` RD argmin
falls to NEARESTMV (the cheapest); the picker emits `LASTFRAME + ZEROMV`
everywhere; the **second** inter frame inherits the degeneracy because
its top-left leaf's neighbours still carry `(NEAREST, ZERO)` from frame
1. This is the **chicken-and-egg degeneracy** E9 §4 (`wrk_docs/E9_quality_gap_decomposition.md:54-67`)
names by file and line, and F.0.3's chain-test
(`crates\rvp9-encoder\tests\a18_prev_frame_mvs_fires.rs`) confirms in
running code.

libvpx's escape hatch is the **previous frame's coded MV grid**
(`cm->prev_frame->mvs`, libvpx `vp9_mvref_common.c:25-28`): when no
spatial neighbour carries the target ref, `find_mv_refs` salvages a
candidate from the co-located prev-frame cell, then runs a second
"different-ref with sign-bias scale" pass for refs that mismatch even
the prev-grid cell. F.0.3 plumbed the grid all the way to
`find_mv_refs`, but defaulted activation **OFF** because pre-F.1
activation regressed testsrc ~240 pp BD-rate (the picker preferred
rate-free NEARMV over a worse-prediction-but-better refined NEWMV when
NEARMV's anchor became non-zero — see F.0.3 HLD §2.5 and
`notes\divergence-ledger.md:676-697`).

F.2 is the **second lever** after F.1 in the kickoff journal's plan
(F.1 → F.2 → F.3 → F.4). It is the change that collapses the
chicken-and-egg degeneracy from end to end. Without F.1, F.2 alone
makes the picker pick the wrong mode (NEARMV instead of NEWMV); without
F.2, F.1's residual-savings RD signal is only legible on frames where a
spatial neighbour happens to carry a real MV — i.e. it only takes
effect deep into a GOP, not on the first inter frame after a KF.

---

## 2. libvpx reference walkthrough (verified against v1.14.1 `12f3a2ac6`)

All paths absolute. Line numbers verified against `D:\language\libvpx`
HEAD `12f3a2ac6` (v1.14.1) — see §3.2 for the verification commands.

### 2.1 `find_mv_refs` entry point

`D:\language\libvpx\vp9\common\vp9_mvref_common.c:133-139`:

```c
void vp9_find_mv_refs(const VP9_COMMON *cm, const MACROBLOCKD *xd,
                      MODE_INFO *mi, MV_REFERENCE_FRAME ref_frame,
                      int_mv *mv_ref_list, int mi_row, int mi_col,
                      uint8_t *mode_context) {
  find_mv_refs_idx(cm, xd, mi, ref_frame, mv_ref_list, -1, mi_row, mi_col,
                   mode_context);
}
```

A thin wrapper over `find_mv_refs_idx` with `block = -1`. The
`block != -1` variant is `vp9_append_sub8x8_mvs_for_idx` (lines
153-199); rvp9 already mirrors both via the `block` parameter on
`dec_find_mv_refs` (`crates\rvp9-core\src\inter\mv_ref.rs:482`) and
the public `find_mv_refs` / `append_sub8x8_mvs_for_idx` wrappers
(`mv_ref.rs:674`, `mv_ref.rs:750`).

### 2.2 `find_mv_refs_idx` body — the meat

`vp9_mvref_common.c:16-131`. Step-by-step:

**Step 0 — prev_frame_mvs pointer** (lines 25-28). Computes
`prev_frame_mvs = cm->prev_frame->mvs + mi_row * cm->mi_cols + mi_col`
if `cm->use_prev_frame_mvs` is set; else `NULL`. The grid is row-major
at stride `cm->mi_cols`. **rvp9 mirror:**
`crates\rvp9-encoder\src\lib.rs:2859-2862` —
`prev_mv_grid.and_then(|g| g.get(leaf_mi_row * mi_cols_us + leaf_mi_col).copied())`.
Identical math; the lookup returns `Option<PrevFrameMvRef>` so the
`NULL` case becomes `None`.

**Step 1 — `memset` mv_ref_list to zero** (line 32). **rvp9 mirror:**
`mv_ref.rs:494` — `let mut list = [Mv::ZERO; MAX_MV_REF_CANDIDATES];`.

**Step 2 — First neighbour pair scan with sub-8 awareness + mode-count
accumulation** (lines 37-53). Iterates `i ∈ {0, 1}` over the first
two `mv_ref_blocks` positions, accumulates `context_counter +=
mode_2_counter[candidate_mi->mode]`, sets `different_ref_found = 1` on
every hit, then `ADD_MV_REF_LIST` with `get_sub_block_mv` (which
returns the sub-8 BMI MV when `block >= 0`, the parent MV otherwise).
**rvp9 mirror:** `mv_ref.rs:498-520`. **Difference (non-load-bearing):**
rvp9 does **not** accumulate `context_counter` because it does not fill
`mode_context` (`mv_ref.rs:18-25` doc-comment) — see §2.4 below.

**Step 3 — Remaining neighbour scan** (lines 57-70). Iterates `i ∈
{2..MVREF_NEIGHBOURS}` over the remaining positions, no sub-8
awareness. **rvp9 mirror:** `mv_ref.rs:525-540`. Identical.

**Step 4 — Prev-frame "matching-ref" salvage** (lines 72-79):

```c
if (cm->use_prev_frame_mvs) {
  if (prev_frame_mvs->ref_frame[0] == ref_frame) {
    ADD_MV_REF_LIST(prev_frame_mvs->mv[0], refmv_count, mv_ref_list, Done);
  } else if (prev_frame_mvs->ref_frame[1] == ref_frame) {
    ADD_MV_REF_LIST(prev_frame_mvs->mv[1], refmv_count, mv_ref_list, Done);
  }
}
```

**rvp9 mirror:** `mv_ref.rs:542-553`:

```rust
if let Some(prev) = prev_frame_mv {
    if prev.ref_frames[0] == ref_frame {
        if add_mv_ref_list_eb(prev.mvs[0], &mut list, &mut count, eb) {
            return finish(list, count, mode, bounds);
        }
    } else if prev.ref_frames[1] == ref_frame
        && add_mv_ref_list_eb(prev.mvs[1], &mut list, &mut count, eb)
    {
        return finish(list, count, mode, bounds);
    }
}
```

**Verdict: byte-equivalent**, modulo the `if/else if` short-circuit
shape (libvpx falls through to the next block on miss; rvp9 returns to
the caller on a `Done` hit — same semantics).

**Step 5 — Different-ref second walk (with sign-bias scale)**
(lines 81-96). Re-walks all `MVREF_NEIGHBOURS` positions; on every
inter neighbour whose ref differs from `ref_frame`, runs
`IF_DIFF_REF_FRAME_ADD_MV` which calls `scale_mv` (`vp9_mvref_common.h:234-243`)
and adds. Guarded by `different_ref_found`. **rvp9 mirror:**
`mv_ref.rs:557-583`. Identical.

**Step 6 — Prev-frame "different-ref with sign-bias scale" salvage**
(lines 99-122):

```c
if (cm->use_prev_frame_mvs) {
  if (prev_frame_mvs->ref_frame[0] != ref_frame &&
      prev_frame_mvs->ref_frame[0] > INTRA_FRAME) {
    int_mv mv = prev_frame_mvs->mv[0];
    if (ref_sign_bias[prev_frame_mvs->ref_frame[0]] !=
        ref_sign_bias[ref_frame]) {
      mv.as_mv.row *= -1;
      mv.as_mv.col *= -1;
    }
    ADD_MV_REF_LIST(mv, refmv_count, mv_ref_list, Done);
  }
  if (prev_frame_mvs->ref_frame[1] > INTRA_FRAME &&
      prev_frame_mvs->ref_frame[1] != ref_frame &&
      prev_frame_mvs->mv[1].as_int != prev_frame_mvs->mv[0].as_int) {
    /* same shape, slot 1 */
  }
}
```

**rvp9 mirror:** `mv_ref.rs:586-602`:

```rust
if let Some(prev) = prev_frame_mv {
    if prev.ref_frames[0] != ref_frame && prev.ref_frames[0].is_inter() {
        let mv = scale_mv(prev.ref_frames[0], ref_frame, sign_bias, prev.mvs[0]);
        if add_mv_ref_list_eb(mv, &mut list, &mut count, eb) {
            return finish(list, count, mode, bounds);
        }
    }
    if prev.ref_frames[1].is_inter()
        && prev.ref_frames[1] != ref_frame
        && prev.mvs[1] != prev.mvs[0]
    {
        let mv = scale_mv(prev.ref_frames[1], ref_frame, sign_bias, prev.mvs[1]);
        if add_mv_ref_list_eb(mv, &mut list, &mut count, eb) {
            return finish(list, count, mode, bounds);
        }
    }
}
```

**Verdict: byte-equivalent**, with the inline sign-bias `*= -1` math
factored into `scale_mv` (`mv_ref.rs:326-342`) — see §2.3.

**Step 7 — Done label** (line 124-130). `mode_context[ref_frame] =
counter_to_context[context_counter]` then `clamp_mv_ref` over the two
output slots. **rvp9 mirror:** `mv_ref.rs:610-640` (`finish`).
`mode_context` write skipped per §2.4. `clamp_mv_ref` happens
identically.

### 2.3 Sign-bias scaling math

libvpx `scale_mv` is **inlined** in `find_mv_refs_idx` at lines
103-107 / 115-119 (it's just `mv.as_mv.row *= -1; mv.as_mv.col *= -1`
when the two sign_bias entries differ). The same factored-out helper
lives at `vp9_mvref_common.h:234-243`.

rvp9 `scale_mv` (`mv_ref.rs:326-342`):

```rust
fn scale_mv(candidate_ref: RefFrame, this_ref: RefFrame,
            sign_bias: &[bool; 4], mv: Mv) -> Mv {
    let cand = ref_sign_bias_at(sign_bias, candidate_ref);
    let this_ = ref_sign_bias_at(sign_bias, this_ref);
    if cand != this_ {
        Mv { row: mv.row.wrapping_neg(), col: mv.col.wrapping_neg() }
    } else {
        mv
    }
}
```

`wrapping_neg` is the i16 equivalent of libvpx's `*= -1` on a signed
16-bit MV component. **Faithful.** The 4-slot lookup table
(`ref_sign_bias_at`) translates libvpx's `[LAST=0, GOLDEN=1,
ALTREF=2]` 3-slot indexing into the 4-slot `[INTRA, LAST, GOLDEN,
ALTREF]` discriminant-order rvp9 uses internally — done in
`sign_bias_from_header` (`mv_ref.rs:367-370`). No semantic divergence.

### 2.4 `mode_context` output

libvpx fills `mode_context[ref_frame]` at line 126 from a
running `context_counter` accumulated only in Step 2 (the first-two
neighbour scan). The picker reads this as the **NEARESTMV/NEARMV
inter-mode context** for entropy-coding the chosen mode
(`vp9_pickmode.c:2780-2800` adds `cost_mv_mode_context[ctx][...]` to
the rate).

**rvp9 status:** the inter-mode probability lookup is currently
context-agnostic. `mv_ref.rs:40-43` doc-comments out the `mode_context`
fill ("the decoder builds it via the separate `get_mode_context`
pre-pass, not via `find_mv_refs`") and the picker at `lib.rs:2871-2904`
calls `find_mv_refs` without consuming a context array.

**Impact for F.2:** the `mode_context` machinery interacts with
**inter-mode probability cost** (rate side). It does **not** affect
which MV candidates `find_mv_refs` returns. The picker's
NEAREST/NEAR/NEW MV anchors are unaffected. The rate-cost effect is a
small percentage-point on mode RD and is the **same kind of cost-model
divergence** F.1's HLD §10 R2 documents as "not in scope until F.4
tuning" (DV-F1-2). **F.2 explicitly defers `mode_context`** — see §4.3.

### 2.5 `mv_refs_rt` (the alternative path)

libvpx's nonrd picker has a **second** MV-ref function,
`mv_refs_rt` at `vp9_pickmode.c:59-152`. It is a stripped-down
spatial-only walker that:
- only checks `ref_frame[0]` on candidates (no slot-1 check);
- early-exits the second loop on first hit (`!refmv_count` guard);
- has no prev-frame-mvs path at all;
- adds an SVC base-layer MV path (lines 127-141).

The picker's dispatch (`vp9_pickmode.c:1275-1283`):

```c
if (cm->use_prev_frame_mvs || comp_pred_allowed) {
  vp9_find_mv_refs(cm, xd, xd->mi[0], ref_frame, candidates, mi_row, mi_col,
                   x->mbmi_ext->mode_context);
} else {
  const_motion[ref_frame] =
      mv_refs_rt(cpi, cm, x, xd, tile_info, xd->mi[0], ref_frame,
                 candidates, &frame_mv[NEWMV][ref_frame], mi_row, mi_col,
                 (int)(cpi->svc.use_base_mv && cpi->svc.spatial_layer_id));
}
```

**Two distinct code paths in libvpx**: the full `vp9_find_mv_refs`
when the gate is open, the stripped `mv_refs_rt` otherwise. rvp9 only
has the former. **F.2 does not port `mv_refs_rt`** — it is a
single-layer-only realtime fast-path optimisation and out of scope
under the kickoff journal's plan (F.2 is "make A18 fire correctly";
performance tuning is F.4).

---

## 3. rvp9 current state

### 3.1 `find_mv_refs` implementation

`D:\language\rvp9-2\crates\rvp9-core\src\inter\mv_ref.rs:482-605`
(`dec_find_mv_refs`). **Already libvpx-faithful** for all six
semantic blocks:

| libvpx (`vp9_mvref_common.c`) | rvp9 (`mv_ref.rs`) | Verdict |
|---|---|---|
| 32 (memset) | 494 (`[Mv::ZERO; 2]`) | Identical |
| 37-53 (first-2 sub-8-aware scan) | 498-520 | Identical (no mode_context) |
| 57-70 (rest of neighbours) | 525-540 | Identical |
| 73-79 (matching-ref prev salvage) | 542-553 | Identical |
| 81-96 (different-ref second walk) | 557-583 | Identical |
| 99-122 (different-ref prev salvage) | 586-602 | Identical |
| 124-130 (clamp + mode_context fill) | 610-640 (`finish`) | Identical except `mode_context` skipped (§2.4) |

The F.0.3 HLD §2.3 claim — *"Both 'matching-ref' and 'different-ref'
salvage passes are in place and reachable"* — is **verified line-by-line**
against the libvpx v1.14.1 source. There is no real porting work for
F.2 in `mv_ref.rs`.

### 3.2 Picker call site

`D:\language\rvp9-2\crates\rvp9-encoder\src\lib.rs:2845-2913`. The
per-leaf prev-MV lookup runs at lines 2859-2862:

```rust
let leaf_prev_mv: Option<PrevFrameMvRef> = prev_mv_grid
    .and_then(|g| {
        g.get(leaf_mi_row * mi_cols_us + leaf_mi_col).copied()
    });
```

Threaded into **both** the NEWMV/NEAREST `find_mv_refs` walk
(line 2878) **and** the separate NEARMV walk (line 2894). Two calls
per ref frame (LAST, GOLDEN) ⇒ 4 `find_mv_refs` invocations per leaf,
each carrying the prev-MV. **The picker does not short-circuit on the
first spatial match** — it calls `find_mv_refs` twice (once each for
the NEW/NEAREST `early_break = true` path and the NEARMV
`early_break = false` path), which is libvpx-faithful per
`mv_ref.rs:493` (`early_break(mode)`).

### 3.3 Writer-side `find_mv_refs`

Per F.0.3 HLD §2.3, `crates\rvp9-core\src\bitstream\modes_writer.rs:1472-1505`
threads the same `prev_mv_grid` into the writer's per-block
`find_mv_refs` calls (needed for NEWMV delta encoding's
`best_ref_mv`). The writer and picker compute the same anchors, so
cross-decode stays byte-exact as long as both sides see the same
gate state.

### 3.4 Storage + capture

Per F.0.3 HLD §2.1 and §2.2. Already correct: keyframe path rebuilds
to all-`(IntraFrame, ZERO)`; inter path captures `cur_mv_grid` into
`self.prev_mv_grid`. **No changes needed in F.2.**

### 3.5 Why A18 activation regressed testsrc pre-F.1

`notes\divergence-ledger.md:676-697`, re-stated in F.0.3 HLD §2.5.
With non-zero prev-frame MVs propagating into `mvc.near`, the picker's
RD math favoured NEARMV (rate-free; no MV-delta bits) even when its
prediction was worse than refined-NEWMV's. The picker's strict-`<`
argmin then kept NEARMV. Cross-decoder remained byte-exact (encoder
and decoder agreed on the wrong picks); the regression was BD-rate,
not correctness.

The fix is **not** in `find_mv_refs` — it is in the picker's RD
ranking, and that is exactly what F.1's `block_yrd` provides:
coefficient-bit-aware RD makes NEWMV win against rate-free NEARMV
whenever NEWMV's prediction is genuinely better (its smaller residual
is now visible to the RD comparator). With F.1 in place, the
regression mechanism is closed.

---

## 4. Proposed F.2 design

### 4.1 What changes in `find_mv_refs`?

**Nothing** in the body of `dec_find_mv_refs`. The salvage passes are
in place and verified line-by-line against libvpx. The implementer
re-reads §2.2 against the live code to convince themselves before
landing the default-flip — that re-read is the "port verification"
deliverable.

### 4.2 The default flip — single-line edit

`D:\language\rvp9-2\crates\rvp9-encoder\src\frame_loop.rs:157` and
`:173`:

```rust
// Before (today):
//   pub use_prev_frame_mvs: bool,
// ...
//   use_prev_frame_mvs: false,

// After (F.2):
//   pub use_prev_frame_mvs: bool,
// ...
//   use_prev_frame_mvs: true,
```

And the doc-comment at `frame_loop.rs:141-157` updated to record the
F.2 flip:

```rust
/// F.0.3 — Enable the previous-frame MV salvage path
/// (libvpx `cm->use_prev_frame_mvs`).
///
/// **Default `true`** since F.2 (2026-05-2x). F.0.3 landed this field
/// default-`false` because activating before F.1 regressed testsrc
/// ~240pp BD-rate (picker preferred rate-free NEARMV over refined
/// NEWMV when NEARMV's anchor was non-zero — see F.0.3 HLD §2.5 and
/// `notes/divergence-ledger.md` 2026-05-… entry). With F.1's
/// `block_yrd` in place, coefficient-bit-aware RD makes NEWMV win
/// against rate-free NEARMV when its prediction is actually better,
/// and the regression mechanism is closed. F.2 flips the default ON.
///
/// When `true`, the encoder evaluates the libvpx
/// `use_prev_frame_mvs` predicate (size match + previous frame was
/// not a KF) per inter frame and threads the captured prev-frame
/// grid into both the picker's and the writer's `find_mv_refs` calls.
pub use_prev_frame_mvs: bool,
```

**The ~50 test-only initialisers** at `frame_loop.rs:1914, 1945, ...,
3906` (per the grep in §6 of the prep work) that pass
`use_prev_frame_mvs: false` are **left as-is**. They are
test-fixtures that pin the pre-F.2 baseline behaviour for byte-exact
tests on synthetic / KF-only / skip-only paths, so they continue to
exercise the gated-off branch. Tests that want F.2 behaviour will
explicitly opt in (`use_prev_frame_mvs: true`) or use a new helper
constructor — implementer's choice.

**Optional:** an implementer hygiene step is to add a
`EncoderConfig::with_track_f_defaults()` constructor and migrate
production call sites (the BD-rate harness, the realtime gate, the
F.0.2 oracle) to it. Out of F.2's minimum scope; implementer's
discretion.

### 4.3 `mode_context` — explicitly out of scope

Per §2.4. F.2 does not fill the `mode_context` array. The
NEAREST/NEAR/NEW MV candidate set is what F.2 needs to lift; the
inter-mode probability rate is F.4 tuning.

**Implementer note:** if F.0.2's per-block oracle starts surfacing
`Δ mode_bits` as material after F.2 lands (e.g. > 50 prob-cost units
on a population mean), elevate `mode_context` from "F.4 tuning" to
"F.2.1 follow-up". The oracle's `mode_bits_est` column is already live
per F.0.2's liveness matrix.

### 4.4 No new probability tables

`find_mv_refs` consumes no probability tables. The sign-bias array is
already plumbed (`sign_bias_from_header` at `mv_ref.rs:367` builds it
from the frame header's `[bool; 3]` ref-sign-bias fields). **No new
data tables, no new entropy state.**

### 4.5 A19 frame-resize interaction — verified closed

F.0.3 HLD §4.5 documented this is closed by the size-match clause in
`gate_open` (`frame_loop.rs:927-928`):

```rust
let gate_open = !self.last_frame_was_key
    && self.prev_mv_grid_dims == (mi_rows_us, mi_cols_us);
```

This matches libvpx (`vp9_encodeframe.c:6108-6110` — `cm->width ==
cm->last_width && cm->height == cm->last_height`). When the frame
size changes mid-stream, `prev_mv_grid_dims` mismatches `(mi_rows_us,
mi_cols_us)`, the gate closes, the picker falls back to spatial-only
seeding for one frame, and the next inter frame's write-back
re-establishes the grid at the new dims. **Self-correcting.** F.2
inherits this; **no further work required**.

### 4.6 No public API breakage

`EncoderConfig.use_prev_frame_mvs` is already a public field
(`frame_loop.rs:157`). Changing its default value is a behavioural
change but not a signature change. The
`pick_and_simulate_inter_encode_partitions` and
`build_inter_packet_with_residuals` APIs already accept the grid; the
`find_mv_refs` and `append_sub8x8_mvs_for_idx` signatures are
unchanged.

---

## 5. Interface changes

**None at the Rust type-system level.** The change is one
literal-constant edit (`false` → `true`) and an accompanying doc-comment
update.

If the implementer chooses the `EncoderConfig::with_track_f_defaults`
hygiene route (§4.2), that adds one method but no required call-site
changes — existing constructors keep working.

---

## 6. Test plan

### 6.1 Salvage-path unit tests (`mv_ref.rs::tests`)

Already extensively covered. `mv_ref.rs:949-1033` exercises
`dec_find_mv_refs` over multiple block sizes and ref frames; the
existing tests cover:
- spatial-only path (no `prev_frame_mv`),
- matching-ref prev salvage (`prev_frame_mv` slot 0 matches `ref_frame`),
- different-ref prev salvage (`prev_frame_mv` slots don't match).

**F.2 adds three new tests** to make the salvage logic firing
auditable from the test report:

| Test | Coverage |
|---|---|
| `prev_salvage_matching_ref_emits_temporal_candidate` | All spatial neighbours intra; `prev_frame_mv.ref_frames[0] == LASTFRAME`, `prev_frame_mv.mvs[0] == (12, 24)`. Assert `RefMvList.mvs[0] == clamp(scale=identity, (12, 24))`. |
| `prev_salvage_different_ref_with_sign_bias_flip` | All spatial neighbours intra; `prev_frame_mv.ref_frames[0] == GOLDENFRAME`, `prev_frame_mv.mvs[0] == (10, 20)`; sign-bias differs between LAST and GOLDEN. Query for `ref_frame = LASTFRAME`. Assert `RefMvList.mvs[0] == (-10, -20)`. |
| `prev_salvage_dedup_against_spatial` | One spatial neighbour carries `(LASTFRAME, (5, 5))`; `prev_frame_mv.ref_frames[0] == LASTFRAME`, `prev_frame_mv.mvs[0] == (5, 5)`. Assert `RefMvList.count == 1` (the prev candidate dedups against the spatial via `add_mv_ref_list_eb`'s != check). |

These tests verify the salvage logic was wired correctly — not that
the picker uses the result (that is F.2's smoke test at §6.2). The
existing 9 `mv_ref.rs` unit tests stay green.

### 6.2 Smoke test — `f2_find_mv_refs_temporal_visible.rs`

F.0.3 §6.3b explicitly handed this off to F.2:

> *6.3b — Deferred: temporal candidates visible in `MvCandidates`.
> Not landed with F.0.3. … this assertion is owned by F.2's HLD
> (which has to test it anyway). F.0.3's responsibility ends at "the
> grid is supplied to `find_mv_refs`" — what `find_mv_refs` does
> with it is F.2's surface.*

**New file:**
`D:\language\rvp9-2\crates\rvp9-encoder\tests\f2_find_mv_refs_temporal_visible.rs`.

Encode a 2-frame KF+inter sequence on a small synthetic source (use
the `encode_synthetic_two_frame_kf_inter_non_skip` helper from S4.16m,
cited in F.1 HLD §7.4) where the inter frame is the KF shifted by a
known full-pel offset (e.g. `(0, +8)` luma pixels = `(0, 64)` in 1/8-pel
MV units). Configure `use_prev_frame_mvs: true`. Add a
`#[cfg(test)] pub(crate) fn` peeker to expose the picker's
`MvCandidates` for one chosen leaf on frame 2.

**Assertions:**

1. `mvc.nearest != Mv::ZERO` on at least one leaf (the prev-frame
   salvage actually produced a non-zero candidate). This is the
   *direct* signal that the salvage path fires end-to-end.
2. `mvc.nearest` is consistent with the KF→inter motion field
   (within sign and a small tolerance — implementer rounds the
   expected value through `clamp_mv_ref` and `lower_mv_precision`).
3. With `use_prev_frame_mvs: false`, the same encode produces
   `mvc.nearest == Mv::ZERO` on every leaf (regression-guards the
   F.0.3 default-OFF behaviour).

### 6.3 F.0.2 oracle — pre/post F.2 diff

**Primary diagnostic.** Run the F.0.2 per-block oracle on foreman_cif
at the Track F target bitrates pre-F.2 (post-F.1) and post-F.2:

| Metric | Expectation |
|---|---|
| `mode_match%` on matched-partition leaves | Materially higher than post-F.1 (NEAREST/NEW anchors now carry real motion → NEW picks become more libvpx-like). |
| Fraction of leaves with non-zero `mvc.nearest` on first inter after KF | Rises from ~0% (F.0.3 baseline: spatial-only after KF is all-zero) to a material fraction (depends on how many leaves have a populated prev-grid cell from frame N-1's picks). |
| `mean Δ mv_bits` (rvp9 − libvpx) | Narrows. With NEAREST anchored on a real MV, NEWMV's delta encodes a smaller residual MV → fewer bits per chosen NEWMV. |
| `partition_match%` | Unchanged (F.2 doesn't touch partitioning; F.3 does). |

Per F.0.2 HLD §3.4 the oracle schema's mv-bits and mode-bits columns
are live by F.2 landing. The `coef_bits_est` column is live as of
F.1.

### 6.4 PSNR-delta gate (F.0.1 primary gate) — pre/post F.2

Run `cargo test -p rvp9-encoder --release --test quality_gate
quality_gate_stage4a_realcontent -- --ignored --nocapture` pre-F.2
(post-F.1) and post-F.2.

**Expectation:** foreman mean PSNR-delta drops further from F.1's
post-baseline. **The post-F.2 threshold to land at is a measurement,
not a prediction** (per F.0.1 HLD §3.2: the threshold is a literal
constant the team-lead updates per milestone after measurement).
F.2's implementer measures the new mean Δ and reports it; the
team-lead updates `TRACK_F_PSNR_DELTA_THRESHOLD_DB` in
`crates/rvp9-encoder/tests/quality_gate.rs`.

The kickoff journal's loose forecast (F.1 → ~+6-7 dB; F.2 → further
drop) is a best-guess; treat the printed number as the canonical
signal.

### 6.5 Byte-exact regression — INTENTIONALLY BROKEN on foreman

Per the kickoff journal and F.1 HLD §7.5, byte-exact cross-decode on
foreman is intentionally broken during F.1-F.3 and re-established as
the F.4 exit criterion. F.2 inherits this — the foreman byte-exact
test stays `#[ignore]` with the F.4 comment F.1 added.

**Other byte-exact tests stay green.** F.2 only changes behaviour
when `use_prev_frame_mvs == true` AND `gate_open == true` (size
match, not the first inter after a KF). Tests on synthetic / KF-only
/ skip-only paths either don't activate the gate (KF) or don't have a
populated prev-grid (the synthetic tests pass
`use_prev_frame_mvs: false` in their `EncoderConfig`, per the ~50
init sites at `frame_loop.rs:1914-3906`). The 94 encoder unit-lib
tests must stay green.

### 6.6 Realtime budget — re-run `realtime_gate_720p30`

`cargo test -p rvp9-encoder --release --test realtime_gate -- --ignored --nocapture`.

**Must stay ≤33 ms median.** Current 5.34 ms median / 33 ms budget
(6× headroom). F.2's added cost is *less than* F.0.3's measurement
(§7 below: ~40 µs/frame on foreman). Critical gate is unchanged.

### 6.7 Clippy + workspace unit-test gate

`cargo clippy --workspace --all-targets -- -D warnings` clean. The
~94 encoder unit tests stay green. The 9 `mv_ref.rs` unit tests stay
green + the 3 new tests in §6.1.

---

## 7. Realtime-budget impact

`find_mv_refs` runs once per (leaf, ref_frame) — 2 ref frames (LAST,
GOLDEN) × 2 modes (NEWMV-style early-break + NEARMV no-early-break)
= 4 invocations per leaf. The prev-grid lookup is **one cache-line
read** (`PrevFrameMvRef` is 16 bytes, aligned to one cache line). The
matching-ref salvage adds **at most 2 array probes** per call
(slot 0 / slot 1). The different-ref salvage adds at most 2 more
probes + one `scale_mv` call (a sign-bias compare + two
`wrapping_neg`).

**Per-leaf added cost: ~20 ns × 4 invocations = ~80 ns.**

**Per-frame at foreman_cif (256×256, ~256 leaves)**: ~80 ns × 256 =
**~20 µs / frame**.

**Per-frame at 720p**: picker pipeline is gated off for `w*h >
307_200` (`lib.rs:2330,2340,2359` — A4.rt-recovery from night-7), so
**zero cost** at 720p+.

This is **substantially below F.1's `block_yrd` budget impact**
(F.1 HLD §8.5 estimates ~0.6 ms / frame at foreman_cif). F.2's cost
is in the noise. **F.2 cannot regress the 33 ms RT gate.**

---

## 8. Coupling

### 8.1 F.2 depends on F.1

**Hard dependency.** Without F.1, flipping `use_prev_frame_mvs` to
`true` regresses testsrc ~240 pp BD-rate (per F.0.3 HLD §2.5 and
`notes/divergence-ledger.md:676-697`). F.1 must land first.

F.2's implementer task is therefore sequenced **after** F.1's
implementer task — same ordering the kickoff journal §"Plan" already
specifies.

### 8.2 F.2 unblocks F.3

Once NEAREST anchors carry real motion (after F.2), the partitioner
can sense whether motion-comp RD justifies larger blocks — the
current variance-of-ZEROMV-residual partitioner over-splits to 8×8
because ZEROMV's residual is always large (F.1 HLD §9, footnote on
the foreman partitioning histogram).

F.3's HLD (separate, not yet written) will pick up the motion-aware
partition lever once F.2's RD signal is in place.

---

## 9. Risks / open questions

### Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | F.1 lands at a quality regime where flipping A18 still regresses testsrc. The F.0.3 ledger entry documents the regression mechanism (NEARMV winning over refined NEWMV), and F.1's `block_yrd` is *expected* to fix it — but the verification gate is empirical. | See R1 mitigation paragraph below. |

> **R1 mitigation (team-lead, 2026-05-23):** F.2's implementer captures the testsrc mean PSNR-delta from F.1's implementer report as the F.1 baseline. F.2's post-flip testsrc mean PSNR-delta must stay **≤ F.1_testsrc_baseline + 3 dB**. If the post-flip number exceeds this threshold, the F.2 implementer **HALTS and reports** — the regression mechanism is not fully closed by F.1 and we revisit F.1's tuning before activating A18 by default. Pin both numbers (F.1 baseline + post-F.2 measurement) in the F.2 PR description.
| R2 | `mode_context` skipped — the inter-mode rate is slightly miscalibrated. | Out of scope per §4.3. F.0.2 oracle's `Δ mode_bits` is the monitor. Elevate if > 50 prob-cost units population mean post-F.2. |
| R3 | F.2's three new unit tests in §6.1 assert against a `scale_mv` semantic that depends on the rvp9 4-slot sign-bias layout (§2.3). If a future refactor changes `RefFrame` discriminants, the tests still pass because they go through `sign_bias_from_header`, but the assertions encode the libvpx-faithful behaviour and would need to be updated alongside. | Document the dependency in the test's preamble doc-comment. |
| R4 | Picker `mvc.nearest` is consumed by **both** the picker's RD evaluation **and** the writer's `best_ref_mv` derivation for NEWMV delta encoding. If F.2's flip somehow desynchronises the two (e.g. one site forgets the new default), cross-decode breaks on tests that *do* opt into `use_prev_frame_mvs: true` and aren't on the foreman intentional-break list. | F.0.3 HLD §2.3 verified both sites already accept the grid. F.2 changes only one literal; the wiring is unchanged. Smoke test §6.2 with `use_prev_frame_mvs: false` regression-guards this. |
| R5 | F.0.2 oracle's libvpx leg requires the patched `vpxenc` per the F.0.2 HLD §3.3 instrumentation patch. If the patch is not yet built when F.2 lands, §6.3 cannot run. | F.0.2 implementer task sequenced before F.2 implementer per the TaskList in the kickoff journal §"Plan". |

### Open questions for the team-lead

All five open questions resolved by team-lead 2026-05-23. Decisions
recorded inline below.

- **Q1 (verification) → RESOLVED 2026-05-23:** F.2 = default-flip +
  tests + verification re-read. The §3 line-by-line walkthrough **is**
  the verification — no paranoia rewrite of the salvage passes.
- **Q2 (smoke-test peeker) → RESOLVED 2026-05-23:** Peeker mechanism
  is implementer's choice. Lean toward the F.0.2 sink if it already
  exposes `mvc.nearest`; otherwise a `#[cfg(test)] pub(crate) fn`
  peeker is fine.
- **Q3 (libvpx fixture differential for `scale_mv`) → RESOLVED
  2026-05-23:** Skip for F.2. The §2.3 line-by-line comparison plus
  the F.0.2 oracle's `mode_match%` is sufficient evidence. F.4 may
  revisit if needed.
- **Q4 (testsrc halt threshold) → RESOLVED 2026-05-23:** Concrete
  threshold approved — post-F.2 testsrc mean PSNR-delta must stay
  **≤ F.1_testsrc_baseline + 3 dB**. If higher, the F.2 implementer
  halts and reports. Both numbers (F.1 baseline captured from F.1's
  implementer report + post-F.2 measurement) are pinned in the F.2 PR
  description. See §9 R1 mitigation paragraph and §11 item 6.
- **Q5 (byte-exact confirm) → RESOLVED 2026-05-23:** Confirmed — only
  the foreman byte-exact test stays `#[ignore]` (F.4 re-establishes).
  The testsrc / synthetic / KF-only / skip-only byte-exact tests stay
  green.

---

## 10. Out of scope (explicit)

- **F.3 motion-aware partition decision** — separate HLD, sequenced
  after F.2.
- **`vp9_NEWMV_diff_bias` and `rd_thresh_freq_fact` machinery** — per
  F.1 HLD §10 Q2, deferred to F.4 tuning.
- **`mv_refs_rt` (libvpx's spatial-only fast-path)** — §2.5.
  Single-layer realtime optimisation, not on the quality path. F.4
  may revisit for the cpu-used-8 lane.
- **Compound prediction** — picker is single-ref today; comp_pred is
  out of v1 scope per `PLAN.md` §2 (D9 / D8 envelope).
- **GOLDEN / ALTREF references** — GOLDEN is already plumbed
  (`build_mvc` builds both LAST and GOLDEN `MvCandidates`); ALTREF is
  out of v1 (no altref per PLAN.md §2 D-13 RESOLVED).
- **`mode_context` fill** — §2.4 / §4.3 / R2. Inter-mode rate
  correction is F.4.
- **SIMD for `find_mv_refs`** — the function is ~50 lines of
  branchy logic; SIMD doesn't apply. Scalar is fast enough (§7).
- **Decoder-side changes** — the decoder already has the full
  `find_mv_refs` and reads the prev-frame MV grid identically
  (`crates\rvp9-decoder\src\tile.rs:2021-2022`). F.2 is encoder-only.

---

## 11. Implementer-facing summary

Strict task order:

1. **Read libvpx `vp9_mvref_common.c:16-131` and rvp9
   `mv_ref.rs:482-605` side by side.** Convince yourself the §3.1
   table holds. If you find any mismatch the F.2 HLD missed, **halt
   and report** to the team-lead before continuing — the whole F.2
   plan rests on this verification.
2. **Add the three salvage-path unit tests in §6.1** to
   `crates/rvp9-core/src/inter/mv_ref.rs::tests`. Run them; confirm
   green. (These tests don't need any other changes to land.)
3. **Add the smoke test
   `crates/rvp9-encoder/tests/f2_find_mv_refs_temporal_visible.rs`**
   per §6.2. The peeker per Q2 (lean F.0.2 sink). Run with
   `use_prev_frame_mvs: true`; confirm `mvc.nearest != ZERO` on the
   chosen leaf. Run with `use_prev_frame_mvs: false`; confirm
   `mvc.nearest == ZERO`. **Both must pass before flipping the
   default.**
4. **Flip the default** at `crates/rvp9-encoder/src/frame_loop.rs:173`:
   `use_prev_frame_mvs: true,`. **Update the doc-comment at
   `:141-157`** per §4.2.
5. **Run the F.0.2 oracle on foreman_cif** pre-F.2 (capture from the
   F.1 implementer's checkpoint commit) and post-F.2 (this PR).
   Record `mode_match%`, fraction-non-zero-nearest, `mean Δ mv_bits`
   per §6.3.
6. **Run `quality_gate_stage4a_realcontent` foreman pre/post**.
   Record the mean Δ. **Run `quality_gate_stage4a` testsrc pre/post.
   If testsrc mean Δ rises by more than +3 dB above the F.1 baseline
   (captured from F.1's implementer report), halt and report.** Both
   numbers (F.1 baseline + F.2 measurement) appear in the PR
   description.
7. **Update the `TRACK_F_PSNR_DELTA_THRESHOLD_DB` constant** in
   `crates/rvp9-encoder/tests/quality_gate.rs` to the post-F.2 mean
   (matched-bitrate, in-range subset) per F.0.1 HLD §3.2.
8. **Run `realtime_gate_720p30`**. Confirm ≤33 ms median (expected:
   unchanged from F.1 — F.2 is sub-microsecond).
9. **Run `cargo clippy --workspace --all-targets -- -D warnings`.**
   Must stay clean. **Run the workspace unit-test suite.** All 94
   encoder library tests + the 12 `mv_ref.rs` unit tests must stay
   green.
10. **Update `notes/divergence-ledger.md`** with the F.2 entry:
    A18 default flipped to ON; testsrc / foreman PSNR-delta pre/post
    numbers; cross-decode foreman test remains `#[ignore]` until F.4.
11. **Run `/code-reviewer`** on the PR. **Run
    `/code-review-apply-feedback`** if it surfaces material items.
12. **Report in the PR description:**
    - Foreman PSNR-delta mean pre/post (number).
    - Testsrc PSNR-delta mean pre/post (number; must be below the
      halt threshold).
    - F.0.2 oracle `mode_match%` pre/post (numbers).
    - F.0.2 oracle fraction-non-zero-nearest pre/post (numbers).
    - 720p realtime median pre/post (numbers; both ≤ 33 ms).
    - All test outputs verbatim.

Total expected diff size: **~10 LoC of production code** (the literal
flip + doc-comment update) + **~120 LoC of test code** (three unit
tests + the smoke test). The smaller-than-F.1 LoC count reflects that
F.2 is structurally a default-flip, not a feature port — see Q1 above
for the framing question.

---

## 12. Ship-state addendum (team-lead, 2026-05-23)

F.1 shipped INFRASTRUCTURE-ONLY (see F.1 HLD §14 / divergence-ledger
2026-05-23 F.1 ship-state entry). Empirical measurement showed F.1's
`coef_rate` SATD-to-prob-cost scale is mis-calibrated against rvp9's
lambda: activating `coef_rate` regressed foreman PSNR-delta from
+9.07 to +10.55 dB (mean) with the 250 kbps point catastrophically
dropping to 20.91 dB (−6 dB). Both `coef_rate` and the libvpx
skip-flag tie-break are gated behind env vars, default OFF.

**Without F.1's `coef_rate` active, F.2's planned A18 default-flip
re-introduces the F.0.3 §2.5 regression mechanism**: the picker
prefers rate-free NEARMV over refined NEWMV when NEARMV's anchor
becomes non-zero (testsrc −240 pp BD-rate). The chicken-and-egg fix
requires BOTH `RVP9_F1_COEF_RATE_ENABLE=1` AND lambda recalibration
— and the calibration is F.4's primary work-item (per F.1 §14
"F.4 inheritance" and the divergence-ledger F.4 inherited work-items
list).

**Team-lead decision (Arthur, 2026-05-23): F.2 ships
INFRASTRUCTURE-ONLY**, mirroring F.1's pattern:

- The literal `use_prev_frame_mvs: false` at
  `crates/rvp9-encoder/src/frame_loop.rs:173` **stays as-is**. No
  default-flip.
- The doc-comment on `EncoderConfig.use_prev_frame_mvs`
  (`frame_loop.rs:141-157`) is updated to record the deferral
  rationale and the F.4 ownership transfer.
- The three salvage-path unit tests (HLD §6.1) land in
  `crates/rvp9-core/src/inter/mv_ref.rs::tests`. These test the
  salvage code-path directly, independent of picker-RD activation.
- The smoke test (HLD §6.2,
  `crates/rvp9-encoder/tests/f2_find_mv_refs_temporal_visible.rs`)
  lands with `use_prev_frame_mvs: true` set explicitly per test
  (NOT via default).
- The F.2 HLD §11 implementer steps that depended on the default
  flip (steps 5–8 — F.0.2 oracle pre/post, testsrc halt-threshold
  gate, `TRACK_F_PSNR_DELTA_THRESHOLD_DB` update, realtime gate
  re-run) are **NOT executed** at F.2 sign-off. The picker hot path
  is unchanged at defaults, so the realtime gate cannot regress.

**What does land:**

- `find_mv_refs` line-by-line verification against libvpx v1.14.1
  `vp9_mvref_common.c:16-131` (the F.2 HLD §3.1 table holds
  verbatim against the live source — see workflow step 2 in the
  implementer brief).
- Three salvage-path unit tests in `mv_ref.rs::tests`:
  - `prev_salvage_matching_ref_emits_temporal_candidate`
  - `prev_salvage_different_ref_with_sign_bias_flip`
  - `prev_salvage_dedup_against_spatial`
- Smoke test
  `crates/rvp9-encoder/tests/f2_find_mv_refs_temporal_visible.rs`
  with `use_prev_frame_mvs: true` opted in per test. Asserts the
  matching-ref salvage lifts a known non-zero `PrevFrameMvRef` into
  `mvc.nearest` when active and stays Mv::ZERO when gated off.
- Doc-comment update on `EncoderConfig.use_prev_frame_mvs` citing
  the F.1 lambda-calibration reason and the F.4 ownership transfer.
- Divergence-ledger entry for F.2 ship-state.

**What does NOT land:**

- The literal `use_prev_frame_mvs: false` → `true` default flip
  (HLD §4.2 / step 4 in the implementer brief). Deferred to F.4.
- The post-flip oracle / quality-gate / realtime re-runs (HLD §11
  steps 5–8). Not meaningful at defaults.
- Any update to `TRACK_F_PSNR_DELTA_THRESHOLD_DB`
  (`crates/rvp9-encoder/tests/quality_gate.rs`). Stays at 9.5 dB —
  F.2 doesn't change picker behaviour at defaults.

**F.4 inheritance:**

F.4 owns flipping `EncoderConfig.use_prev_frame_mvs` to `true` once:

1. The SATD-to-bits scale is recalibrated so
   `RVP9_F1_COEF_RATE_ENABLE=1` does not regress PSNR-delta (per
   F.1 ship-state F.4 work-items).
2. The post-recalibration A/B sweep confirms `use_prev_frame_mvs:
   true` does NOT regress testsrc BD-rate beyond the F.1
   baseline + 3 dB halt threshold (HLD §9 R1 mitigation).
3. The two activation knobs (`RVP9_F1_COEF_RATE_ENABLE` and
   `RVP9_F1_SKIP_TIE_BREAK_ENABLE`) are removed (or default-ON).

**Why this is the right call:**

F.2's verification + tests land on this infrastructure. The salvage
code-path is exercised by the new unit tests + smoke test even at
defaults, so a regression in `find_mv_refs` is caught CI-wide. F.3
(motion-aware partition) lands on the same infrastructure. F.4
flips the default in a single literal-constant edit once its
recalibration work-items close.

**Cross-references:**

- Decision recorded in `notes/divergence-ledger.md` F.2 ship-state
  entry (2026-05-23).
- F.1 ship-state addendum (`F1_HLD_block_yrd_port.md` §14) — the
  precedent this addendum mirrors.
- F.4 task inherits the default-flip work-item.
