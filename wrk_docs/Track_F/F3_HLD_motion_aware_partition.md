# F.3 — HLD: Replace variance-of-ZEROMV-residual partitioner with libvpx's motion-aware (RD-driven) partition decision

| Field | Value |
|---|---|
| Track | F (nonrd_pickmode port) |
| Sub-track | F.3 (motion-aware partition — final coupled piece) |
| Author | F.3 HLD-writer agent |
| Status | SIGNED-OFF — pending F.1 + F.2 implementer landing |
| Last revised | 2026-05-23 |
| Repo HEAD at writing | `f1a8327` (post-F.0 foundations; F.1 + F.2 HLDs signed off, implementers in flight) |
| libvpx HEAD at writing | `12f3a2ac6` (`v1.14.1`) at `D:\language\libvpx` |
| Dependencies | **F.1 (`block_yrd`)** — must land first, F.3 has no RD signal without it. **F.2 (`find_mv_refs` proper + A18 default-ON)** — must land first, motion-aware decisions need real MV anchors. |
| Blocks | F.4 (tuning + byte-exact re-establishment) — F.3 is the FINAL coupled piece; after F.3 the F.1+F.2+F.3 cluster closes. |

**Sign-off:** 2026-05-23 — Track F team-lead approved this HLD with revisions 1–5. Q1–Q5 resolved (see §8.2).

> rvp9's current partitioner (`crates/rvp9-core/src/mode_decision/partition.rs:344` —
> `choose_partitioning`) is a faithful port of libvpx's **variance fast-path**
> (`vp9_encodeframe.c:1261 choose_partitioning`). What libvpx ALSO has — and rvp9
> does NOT — is the **motion-aware RD descent**
> (`vp9_encodeframe.c:4922 nonrd_pick_partition`) that evaluates inter modes at
> the larger block first and splits only when the bigger-block RD does not
> justify itself. F.3 ports that descent.

---

## 1. Problem statement & motivation

### 1.1 The 85% Block8x8 ceiling on F.1

F.1 HLD §9 (`wrk_docs/Track_F/F1_HLD_block_yrd_port.md:817`) records the
hard ceiling on F.1's impact in foreman:

> Per E1's block_count histogram, foreman's current variance-partitioner
> descends to Block8x8 on ~85% of leaves. F.1's `block_yrd` only runs on
> Block16x16+ (the picker bails to ZERO+skip at 8×8 per
> `lib.rs:2660-2711`). So F.1's RD signal only reaches ~15% of foreman's
> leaves until F.3 (motion-aware partition) lifts the typical leaf size.

The variance partitioner sees the **ZEROMV residual** (`src - last_recon` at
the same coordinates). On natural motion content every block has a large
ZEROMV residual — the variance threshold is exceeded everywhere and the
descent reaches 8×8 on >85% of leaves. The partitioner has no notion of
"would motion compensation make this residual small at the larger
block?" — because it has not done motion compensation yet.

The result: F.1's coefficient-domain rate signal is starved of leaves it
can act on. The whole cluster's PSNR-delta payoff is gated on F.3
lifting the typical leaf size.

### 1.2 E11 / E14 history — partition-cap previously REFUTED, but the test was broken

E11 (`wrk_docs/E11_partition_size_probe.md`) and E14
(`wrk_docs/E14_partition_size_retest.md`) tested a crude "force Block16x16
cap" and found NEWMV win-share stayed at 0.0%, prediction PSNR was flat,
and recon PSNR moved < 0.1 dB. The probes concluded partition-cap was not
a standalone lever.

**Those experiments ran under broken search/RD**: F.1 had not landed
(coefficient bits invisible to RD), and the picker bailed to
ZERO+skip at Block8x8. With F.1 making residual rate RD-visible and F.2
giving NEAREST a non-zero anchor, the lever finally has signal. The E11
/ E14 verdict was correct *at that point in the codebase* but does not
generalise to a post-F.1+F.2 baseline. F.3 must measure freshly.

### 1.3 The goal

After F.1+F.2+F.3 land, the F.0.2 oracle's `partition_match%` against
libvpx should converge (today rvp9 picks Block8x8 where libvpx picks
Block16x16/32x32 on natural-motion regions). Foreman PSNR-delta should
fall measurably from the current +9.07 dB baseline. The exact
threshold is a measurement, not a prediction — see §6.

---

## 2. libvpx reference — minimal walkthrough

### 2.1 Two-tier structure

libvpx's realtime path has **two** partition mechanisms that run together,
not one. They are best understood as the fast-path (variance) plus the
quality-path (RD) layered on top:

1. **Variance fast-path — `choose_partitioning`**
   (`D:\language\libvpx\vp9\encoder\vp9_encodeframe.c:1261`). Walks the SB
   top-down (64→32→16→8), splitting where per-pixel variance against the
   ZEROMV reference exceeds a q-conditional threshold. **This is the
   function rvp9 already ports** (`partition.rs:344 choose_partitioning`).
   It produces an initial partition tree as a hint, not a final
   decision. Cheap (~tens of µs/SB).

2. **Motion-aware RD descent — `nonrd_pick_partition`**
   (`D:\language\libvpx\vp9\encoder\vp9_encodeframe.c:4922`). Recursive
   descent that at each `bsize` evaluates `PARTITION_NONE` (one
   `nonrd_pick_sb_modes` call at this size), then optionally evaluates
   `PARTITION_SPLIT` (four recursive calls at the child size), and keeps
   whichever has the lower `rd_cost`. The variance tree from
   `choose_partitioning` informs `x->min_partition_size` /
   `x->max_partition_size` (`vp9_encodeframe.c:4968-4978`) which bound
   the descent but do not replace it.

The call relationship at `vp9_encodeframe.c:5633-5677` shows them
running back-to-back at the SB level: variance hint first, then
RD descent walks within those bounds.

### 2.2 `nonrd_pick_partition` skeleton (cited not enumerated)

`vp9_encodeframe.c:4922-5210` (~280 lines). The algorithmic skeleton:

```
nonrd_pick_partition(bsize):
    if partition_none_allowed:                  // :5000
        nonrd_pick_sb_modes(bsize)              // :5001  — full mode search at this size
        this_rdc.rate += partition_cost[NONE]
        if this_rdc.rdcost < best_rdc.rdcost:
            best = this_rdc
            // breakout: if rate & dist are both far below thresholds, do_split = 0
    if do_split:                                // :5037
        for each of 4 children:
            nonrd_pick_partition(subsize)        // :5049 — recurse
            sum_rdc += child_rdc
        if sum_rdc.rdcost < best_rdc.rdcost:
            best = sum_rdc; pc_tree->partitioning = SPLIT
    // Optionally PARTITION_HORZ / PARTITION_VERT for speed < 5    (skipped for rvp9 F.3)
    // Write back best partition + modes into pc_tree
```

The mode search inside `nonrd_pick_sb_modes` is the F.1 / F.2 territory —
F.3 just calls into it once per candidate partition size.

The function is large (~280 lines) because of edge-handling, breakout
heuristics, and PARTITION_HORZ/VERT branches. **F.3's first cut takes
ONLY the square-only NONE-vs-SPLIT skeleton** and defers HORZ/VERT to a
follow-up — libvpx itself does the same at `speed >= 5`
(`vp9_encodeframe.c:4957`: `if (cpi->oxcf.speed >= 5) do_rect = 0`).

### 2.3 What we are NOT porting in F.3

- `PARTITION_HORZ` / `PARTITION_VERT` — disabled at libvpx speed ≥ 5,
  defer to F.4.
- `ML_BASED_PARTITION` (`vp9_encodeframe.c:4951,4984`) — speed-feature off
  at rvp9's target profile; defer indefinitely.
- The `simple_motion_search`-driven `auto_min_max_partition_size`
  refinement of variance bounds (`vp9_encodeframe.c:4968-4978`) — F.3
  uses the variance hint as-is.
- `PC_TREE` ownership and `pred_pixel_ready` plumbing — that's libvpx's
  buffer-reuse pipeline; rvp9 already materialises per-leaf state
  differently. Use rvp9's existing per-leaf data flow.

---

## 3. rvp9 current state

### 3.1 Variance partitioner (the function F.3 augments, not replaces)

- **Production code**: `crates/rvp9-core/src/mode_decision/partition.rs:344`
  (`choose_partitioning`). Top-level `descend` at line 401 implements the
  64→32→16→(8) recursion; `allow_split_to_8x8` at line 461 is the
  inter-vs-KF gate. Returns `PartitionDecision { cells: [[BlockSize; 8]; 8] }`.
- **SB call site**: `crates/rvp9-encoder/src/frame_loop.rs:1200`. One call
  per 64×64 SB during the inter-frame partition pre-pass. The result is
  fed to `leaves_from_decision` (`frame_loop.rs:1233`) to flatten into a
  `Vec<PartitionLeafDesc>` per SB.
- **Picker consumes**: the per-SB leaf list is then passed to
  `pick_and_simulate_inter_encode_partitions`
  (`crates/rvp9-encoder/src/lib.rs:2292`) as `sb_partitions:
  &mut [Vec<InterPartitionLeafDesc>]`. The picker iterates the leaves and
  for each leaf runs the per-leaf mode picker
  (`pick_inter_mode_luma` at `mode_decision/inter.rs:545+`).

### 3.2 The coupling that makes F.3 invasive

Today the call ordering is:

```
frame_loop.rs:1200  choose_partitioning(SB)  →  sb_partitions
                                                       ↓
lib.rs:2292         pick_and_simulate_inter_encode_partitions(sb_partitions, ...)
                                                       ↓
                    for each leaf:
                        pick_inter_mode_luma(leaf)     ← mode search
```

The partition is decided BEFORE any mode search runs. F.3 has to
**invert** this so mode search runs at the larger block sizes first and
informs whether to split. The cleanest invariant: keep `frame_loop.rs`
unchanged but have the picker treat `sb_partitions` as a *hint* rather
than a *commitment*. See §4.3.

---

## 4. Proposed F.3 design

### 4.1 What changes, what is preserved

**Preserved**:
- `choose_partitioning` stays exactly as it is today. It becomes the
  **variance hint** path — same as libvpx's two-tier model.
- `frame_loop.rs:1200` continues to call `choose_partitioning` per SB and
  produces the variance-tree leaf list.
- F.0.x infrastructure (oracle, gates) — unchanged.
- The variance leaf list is used to derive **min/max partition size
  bounds** per SB (libvpx's `x->{min,max}_partition_size`,
  `vp9_encodeframe.c:4968-4978`).

**Changes**:
- A new `nonrd_pick_partition`-shaped descent runs **inside**
  `pick_and_simulate_inter_encode_partitions`
  (`crates/rvp9-encoder/src/lib.rs:2292`). For each SB, it recursively
  evaluates `PARTITION_NONE` (mode search at the current bsize) vs
  `PARTITION_SPLIT` (recurse into four children) and picks the
  lower-RD-cost option.
- The output is the same `Vec<InterPartitionLeafDesc>` shape that the
  rest of the pipeline (bitstream writer, simulator) consumes — but
  the contents reflect the motion-aware decision, not the variance hint.

### 4.2 Recommended architecture: keep variance as fast-path

**Recommendation: KEEP `choose_partitioning` as the upstream hint, do
not rip it out.**

Rationale:
- libvpx does the same — `vp9_encodeframe.c:5633-5677` shows variance
  and RD descent both running.
- The variance hint shrinks the F.3 search space via min/max partition
  bounds, which is the realtime budget escape hatch (§7).
- Removing the variance pass means F.3 must always start at Block64x64
  and walk all the way down — a much bigger realtime hit than the
  bounded descent libvpx uses.
- The variance partitioner is already battle-tested in rvp9 (F8.2 +
  E1/E11/E14 stress runs). Keeping it leaves a known-good fallback if
  F.3 needs to be env-gated off mid-bringup.

### 4.3 Recursion-depth cap: 16×16 (first cut), not 8×8

**Recommendation: F.3's first cut caps recursion at Block16x16.**

Rationale:
- The existing picker already bails to ZERO+skip at Block8x8
  (`crates/rvp9-encoder/src/lib.rs:2660-2711`, per F.1 HLD §9). F.3
  inherits that — running mode search at 8×8 would require unblocking
  that bail, which is out of scope.
- libvpx itself excludes Block8x8 from `nonrd_pick_partition` in some
  paths (`vp9_encodeframe.c:5662` — "nonrd_pick_partition does not
  support 4x4 partition"). The 8×8 leaf falls through to the variance
  hint's decision.
- The whole point of F.3 is to LIFT the typical leaf size from 8×8 to
  16×16+. Searching at 8×8 inside the F.3 descent works against that
  goal.

**Behaviour**: when the F.3 descent reaches `bsize == Block16x16`, it
evaluates PARTITION_NONE only (no further split). The four child 8×8
cells inherit whatever the variance hint decided for them — typically
they will all be flagged 8×8 (the existing degenerate path), but the
F.3 RD decision at 16×16 will usually beat that and emit a single
16×16 leaf instead.

### 4.4 The new module shape (file-level only — no signatures)

- **New module**: `crates/rvp9-core/src/mode_decision/nonrd_partition.rs`
  (or similar). Houses the recursive descent function. Mode search
  remains in `mode_decision/inter.rs`.
- **Edited**: `crates/rvp9-encoder/src/lib.rs` around line 2292 —
  `pick_and_simulate_inter_encode_partitions` invokes the new descent
  per SB instead of iterating the variance leaf list directly. The
  variance leaf list is converted into min/max bounds and passed in.
- **Edited**: `crates/rvp9-core/src/mode_decision/mod.rs` — re-export
  the new descent function.

### 4.5 RD ranking — what makes one partition "win"

Each candidate partition's RD cost is computed via the existing
`rd_cost` helper (`mode_decision/mod.rs:55`) with:
- `rate` = sum of per-leaf F.1 rate (`mode_rate + mv_rate +
  ref_frame_rate_cost + coef_rate` from F.1's `BlockYrdResult`) PLUS the
  partition-tree wire cost. Partition tree cost is `partition_cost[pl]
  [PARTITION_NONE]` or `[PARTITION_SPLIT]` per
  `vp9_encodeframe.c:5009-5011, 5039`.
- `dist` = sum of per-leaf F.1 distortion.

Partition wire cost is part of the cost table already in rvp9
(`crates/rvp9-core/src/cost/symbol_costs.rs` — verify location during
implementation). F.3 just looks it up; it does not need new entropy
tables.

### 4.6 Breakout heuristic (cheap and important)

libvpx's `vp9_encodeframe.c:5017-5028` short-circuits split when the
NONE rdcost is far below `partition_search_breakout_thr`. F.3 should
adopt this (single integer compare, no extra cost) — it preserves the
realtime budget when a large block has a clearly-good motion-comp fit.

---

## 5. Coupling with F.1 / F.2 / F.4

| Dep | Why F.3 needs it |
|---|---|
| F.1 (`block_yrd`) | F.3's per-candidate-partition RD compares rate+dist sums. Without `coef_rate`, every candidate at every partition size has the same constant rate term — F.3 cannot rank them. **HARD dependency.** |
| F.2 (`find_mv_refs` proper + A18 default-ON) | At Block32x32/64x64, NEAREST/NEAR anchors must come from non-degenerate neighbours. Without F.2, NEAREST is `(0,0)` everywhere and NEWMV cannot win at larger blocks → mode search at the larger block reports a bad RD → F.3 always splits → no behaviour change vs today. **HARD dependency.** |
| F.4 (tuning + byte-exact) | F.4 lambda-recal, simple-yrd shortcut, and the libvpx-exact byte-stream re-establishment are tuning that follows F.3. F.3 ships with byte-exactness intentionally broken (same as F.1 / F.2). |

**F.3 is the final coupled piece.** After F.3 lands and F.4 closes
byte-exactness + tuning, the Track F cluster is closed.

---

## 6. Test plan

### 6.1 Unit tests (synthetic, in-tree)

- **Flat block picks Block64x64**. Source = ref everywhere → motion-comp
  RD at 64×64 is near zero → F.3 must emit `PARTITION_NONE` at 64×64.
  Sanity: matches existing variance-partitioner test
  (`partition.rs:587 choose_partitioning_flat_block_picks_64x64`).
- **Synthetic translating block** — 32×32 region of unique texture
  shifted by `(2, 0)` against the reference. F.3 must pick Block32x32
  with NEWMV `(2, 0)` (not split to 4× Block16x16 with ZEROMV). This
  is the headline behavioural test — today's variance partitioner
  splits to 8×8 because ZEROMV residual is large.
- **Mixed-motion SB** — half static, half textured-shifted. F.3 must
  emit asymmetric splits (one 32×32 leaf + four 16×16 leaves). Verifies
  recursion correctness.
- **Recursion-depth cap** — verify F.3 never emits Block8x8 leaves
  driven by the F.3 descent itself; any 8×8 leaves must come from the
  variance hint.

### 6.2 F.0.2 oracle — the headline signal

`partition_match%` between rvp9's picked partition tree and libvpx's
picked partition tree (per SB, per frame). Today this is dominated by
"rvp9 picked 8×8, libvpx picked 32×32". After F.3 lands, the
`partition_match%` on foreman should converge with libvpx within a
sign-off threshold (specific number is a measurement decision at F.3
review time — propose ≥ 60% match on foreman as a working target).

### 6.3 F.0.1 quality gate — PSNR-delta drop

`wrk_docs/Track_F/F0.1_HLD_psnr_delta_primary_gate.md` is the primary
gate. F.3's expected behaviour: foreman PSNR-delta at matched bitrate
drops measurably from the post-F.2 number. Concrete threshold is a
measurement — set at F.3 implementation review based on what F.1+F.2
have already delivered.

### 6.4 Byte-exact: still broken (intentionally)

Same as F.1, F.2. F.3 changes the partition decision, which changes
mode-symbol order, MV-symbol values, EOB positions — every bitstream
byte. F.4 re-establishes byte-exactness across the whole cluster.

### 6.5 Realtime gate — THE major risk

`realtime_gate_720p30` runs at 1280×720. The picker is already gated
off above 307_200 pixels (`crates/rvp9-encoder/src/lib.rs:2401
inter_picker_enabled = w * h <= 307_200`), so F.3 cost on the 720p
gate itself is **zero** — F.3's descent only runs when the picker is
on. F.0.x gates pass automatically.

**But** the small-frame corpora (foreman_cif 352×288, testsrc
256×256) where the picker IS on get the full F.3 hit. See §7.

### 6.6 Partition histogram telemetry

`RVP9_PART_HIST` env-gated histogram already exists
(`frame_loop.rs:1203`). F.3 must extend it to dump the **post-descent**
partition histogram, not the variance-hint one. Compare against
libvpx's histogram for the same content.

---

## 7. Realtime-budget impact — the major risk

### 7.1 Worst-case per-SB cost multiplier

Variance hint today: ~1 µs/SB (just per-pixel-variance computation +
threshold compare).

F.3 descent: at each non-leaf bsize, runs `nonrd_pick_sb_modes` (which
calls `pick_inter_mode_luma` with all of {ZERO, NEAREST, NEAR, NEW} ×
{LAST, GOLDEN} = 8 candidates). Per F.1 HLD §8.2, that's ~38 µs at
Block64x64, ~10 µs at Block32x32, ~2.4 µs at Block16x16.

Worst case (every SB walks fully): 38 + 4×10 + 16×2.4 ≈ 116 µs per SB
where today's pipeline is ~2.4 µs × ~16 leaves = ~38 µs per SB
(when 16x16-dominant) or 0.3 µs × ~64 leaves = ~19 µs (when 8x8-dominant
with picker bail). **Worst-case multiplier: ~3× to ~6× per SB.**

### 7.2 Per-frame projection — foreman_cif 256×256

Today's F.1 projection (F.1 HLD §8.3): ~0.6 ms/frame at Block16x16
average. F.3 worst case: ~16 SBs × 116 µs ≈ **1.9 ms/frame**.

That's well under the 33 ms 720p30 budget — but combined with F.1+F.2
and the higher absolute leaf count at small block sizes, the F.3
implementer must measure the realised number, not trust this estimate.

### 7.3 Per-frame projection — 720p picker disabled

Zero impact. F.3's descent only runs when `inter_picker_enabled` is
true, and that's already gated off for w*h > 307_200.

### 7.4 Mitigations available without redesign

1. **Variance hint bounds the descent.** If variance says
   `min_partition_size = 16` for a region, F.3 stops splitting at
   16×16 — the typical case. This is the libvpx mechanism and the
   single most important budget control. Implement it from day one,
   not as an optimisation.
2. **Breakout heuristic (§4.6).** Short-circuits split when NONE is
   clearly good.
3. **`avg_partition_rdcost`-driven early-out.** Track per-frame mean
   RD cost; if a partition's rdcost at NONE is below
   0.5 × frame_mean, skip the split branch entirely. Libvpx has a
   similar mechanism in the speed features.
4. **Fallback env gate.** Ship F.3 with `RVP9_F3_NONRD_PARTITION_OFF`
   env var that reverts to the pre-F.3 variance-only behaviour, in
   case the realtime gate regresses.

### 7.5 Verdict and budget thresholds

F.3 is **the** realtime risk in the Track F cluster, but the risk is bounded:

- **`realtime_gate_720p30` (the hard 33 ms gate)**: F.3 cost is zero because the picker is disabled above 307_200 pixels. The 33 ms gate is automatically preserved.
- **Foreman_cif 256×256 (the quality testbench)**: F.3's worst-case projection is ~1.9 ms/frame against an aggregate post-F.1 baseline of ~5-7 ms/frame at this resolution. **Working ceiling: 8 ms median on foreman_cif.** If F.3's measured cost crosses 8 ms, fall back to mitigation 3 (avg-RD early-out) as F.3.1 — this is a self-imposed engineering ceiling for the small-frame bench, not a project-level hard gate.

Implementer's sign-off must include:
- 720p realtime median (must remain ≤ 33 ms — expected zero impact).
- Foreman_cif median frame time before / after.
- Foreman_cif 95th-percentile frame time before / after.
- 720p picker-on stress run (manually flip `inter_picker_enabled = true` at 720p) to validate the variance-hint-bounded descent doesn't explode when forced on at larger resolutions. **This is a one-off informational measurement, not a gate** — it tells F.4 whether the picker can be re-enabled at higher resolutions without redesigning F.3.

---

## 8. Risks / open questions

### 8.1 Risks (R-numbered)

| # | Risk | Mitigation |
|---|---|---|
| R1 | **F.3 doesn't move PSNR-delta despite landing cleanly.** Possible if the picker's mode-search RD is still skewed by some F.1/F.2 residual mismatch (lambda, ref-frame cost, etc.). | F.3's headline signal is the F.0.2 oracle's `partition_match%` — that converges with libvpx independently of any PSNR effect. PSNR-delta gate is a follow-on measurement, not a sign-off blocker for F.3 in isolation. |
| R2 | **Realtime budget regression on small-frame corpora.** F.3 worst case is ~3-6× per-SB; foreman_cif could see ~1.9 ms/frame. | §7.4 mitigations 1–4. Env-gate the F.3 behaviour for safe rollback. |
| R3 | **Variance hint min/max bounds are wrong / nonsensical at high q.** rvp9's variance partitioner already runs at q ≥ 200 with thresholds that push it to 8×8 everywhere; converting that into `max_partition_size = 8` would defeat F.3 entirely. | Compute min/max bounds from the variance leaf set with a floor at 16×16 (never let `max_partition_size` drop below 16×16). The variance hint is then a "suggestion of where to split", not a hard cap. |
| R4 | **Partition-tree wire cost not in rvp9's cost tables.** F.3 adds `partition_cost[pl][PARTITION_NONE/SPLIT]` to the RD sum, but rvp9's `cost/symbol_costs.rs` may not expose these (the encoder currently does not need them because the picker doesn't choose partitions). | **R4 mitigation (team-lead Q5 resolved, 2026-05-23):** if `partition_cost` is absent from `crates/rvp9-core/src/cost/symbol_costs.rs`, the F.3 implementer ports the small helper from libvpx (`vp9_cost_tokens(c, partition_probs[pl], vp9_partition_tree)` from `vp9_encodeframe.c:1235`-ish). This is part of F.3 scope, not a separate sub-track. |
| R5 | **E11/E14 verdict re-applies.** The probes measured no PSNR movement under partition-cap. If F.3's mechanism is fundamentally the same as the cap (just better-targeted), the result may not improve much. | F.1+F.2 close the broken-search/broken-RD assumption. F.3's design uses RD ranking *with* coefficient bits visible, not blind capping. If F.3 still doesn't move signal, that's a finding worth its own retrospective. |

### 8.2 Open questions for team-lead

1. **F.3 first-cut depth cap.** Resolved (team-lead, 2026-05-23): APPROVED — recursion caps at Block16x16; no F.3-driven Block8x8 mode search.
2. **Variance hint role.** Resolved (team-lead, 2026-05-23): APPROVED — KEEP `choose_partitioning` as fast-path / bound-source (libvpx does the same; variance hint shrinks F.3 search space).
3. **PARTITION_HORZ / PARTITION_VERT.** Resolved (team-lead, 2026-05-23): APPROVED — deferred to F.4 (matches libvpx speed ≥ 5).
4. **Foreman `partition_match%` sign-off threshold.** Resolved (team-lead, 2026-05-23): APPROVED as a *working target* (≥ 60% on foreman) — the actual sign-off threshold is a measurement at F.3 implementation review, set after F.1+F.2 deliver their own numbers (F.0.1 "measurement, not prediction" pattern).
5. **Partition-cost-table sourcing (R4).** Resolved (team-lead, 2026-05-23): APPROVED — if `crates/rvp9-core/src/cost/symbol_costs.rs` doesn't already expose `partition_cost[ctx][PARTITION_NONE/SPLIT]`, a small port from libvpx (`vp9_cost_tokens` + `vp9_partition_tree`) is acceptable within F.3 scope, NOT a separate sub-track.

---

## 9. Out of scope

The following are explicitly NOT part of F.3:

- **F.4 tuning** — lambda recalibration, simple-yrd shortcut,
  partition_search_breakout threshold tuning. F.3 ships with
  conservative defaults; F.4 owns the tuning sweep.
- **Compound prediction** (libvpx's PRED_INTERP_FILTER ×
  COMPOUND_INTER combinations). The picker today is single-ref
  per-candidate; compound prediction is a separate track.
- **Intra-in-inter mode evaluation.** libvpx's nonrd_pick_sb_modes
  evaluates intra candidates inside inter frames at some bsizes; F.3
  inherits whatever rvp9's `pick_inter_mode_luma` does today (inter-only).
- **PARTITION_HORZ / PARTITION_VERT** (rectangular partitions).
  Defer to F.4 or beyond. The libvpx speed >= 5 path also defers
  these (`vp9_encodeframe.c:4957`).
- **`PC_TREE` adoption.** rvp9 has its own per-leaf data flow; do not
  port libvpx's tree-of-mode-contexts machinery. The F.3 descent can
  carry its working state on the stack.
- **ML-based partitioning** (`ml_predict_var_partitioning`,
  `vp9_encodeframe.c:4988`). Off in libvpx's realtime profile at
  rvp9's target speed setting; not relevant.
- **Byte-exact re-establishment.** Track F as a whole intentionally
  breaks byte-exactness; F.4 re-establishes it across the cluster.

---

## 10. Implementer-facing summary

- **Port target**: `nonrd_pick_partition` at
  `D:\language\libvpx\vp9\encoder\vp9_encodeframe.c:4922-5210`.
  Take the square-only NONE-vs-SPLIT skeleton; drop HORZ/VERT, ML,
  pred_pixel_ready, PC_TREE.
- **Keep**: `crates/rvp9-core/src/mode_decision/partition.rs:344
  choose_partitioning` as upstream variance hint.
- **Edit**: `crates/rvp9-encoder/src/lib.rs:2292
  pick_and_simulate_inter_encode_partitions` — invoke the new descent
  per SB; convert variance leaf list into min/max bounds; descent
  produces the final `Vec<InterPartitionLeafDesc>`.
- **Edit**: `crates/rvp9-encoder/src/frame_loop.rs:1200` — unchanged in
  behaviour, but its output is now interpreted as a hint, not a
  commitment. Document this.
- **New module**: `crates/rvp9-core/src/mode_decision/nonrd_partition.rs`
  for the descent function.
- **Recursion cap**: stops at Block16x16 (no F.3-driven 8×8 mode
  search). Block8x8 leaves come from the variance hint, not the
  descent.
- **RD cost**: existing `rd_cost(lambda, rate, dist)` with rate =
  Σ per-leaf F.1 rate + partition-tree wire cost; dist = Σ per-leaf
  F.1 dist.
- **Breakout heuristic**: adopt libvpx's
  `partition_search_breakout_thr` cheap short-circuit (single integer
  compare).
- **Env gate**: `RVP9_F3_NONRD_PARTITION_OFF` reverts to pre-F.3
  behaviour for safe rollback.
- **Test gates**:
  - F.0.2 oracle `partition_match%` ≥ working target (measure after
    F.1+F.2 land).
  - F.0.1 PSNR-delta gate — measurable drop from the post-F.2 baseline
    (threshold is a measurement, not a prediction).
  - **Realtime gate 720p30**: ≤ 33 ms median (expected: zero F.3 impact — picker disabled).
  - **Foreman_cif self-imposed ceiling**: 8 ms median / quality testbench, not a hard project gate. If exceeded, fall back to F.3.1 (avg-RD early-out).
  - **Foreman_cif 95th-percentile**: report number, no gate (signal for F.4 tuning).
  - Byte-exact: still broken (F.4 territory).
- **Dependencies**: F.1 (block_yrd) AND F.2 (find_mv_refs proper +
  A18 default-ON) MUST land first. F.3 has no signal without them.
- **Status after F.3**: the F.1+F.2+F.3 cluster is closed; F.4 takes
  over for tuning + byte-exact re-establishment.

---

## Appendix A — File:line citation table

| Subject | Path | Lines |
|---|---|---|
| libvpx variance fast-path | `D:\language\libvpx\vp9\encoder\vp9_encodeframe.c` | `1261` |
| libvpx motion-aware RD descent (**F.3 port target**) | `D:\language\libvpx\vp9\encoder\vp9_encodeframe.c` | `4922-5210` |
| libvpx two-tier SB call site (variance + RD side-by-side) | `D:\language\libvpx\vp9\encoder\vp9_encodeframe.c` | `5633-5677` |
| libvpx min/max partition size machinery | `D:\language\libvpx\vp9\encoder\vp9_encodeframe.c` | `4968-4978` |
| libvpx breakout heuristic | `D:\language\libvpx\vp9\encoder\vp9_encodeframe.c` | `5017-5028` |
| libvpx partition-cost (PARTITION_NONE / SPLIT) | `D:\language\libvpx\vp9\encoder\vp9_encodeframe.c` | `5009-5011, 5039` |
| rvp9 variance partitioner | `D:\language\rvp9-2\crates\rvp9-core\src\mode_decision\partition.rs` | `344` (entry), `401` (descend), `461` (KF gate) |
| rvp9 SB-level partitioner call site | `D:\language\rvp9-2\crates\rvp9-encoder\src\frame_loop.rs` | `1200` |
| rvp9 inter picker entry | `D:\language\rvp9-2\crates\rvp9-encoder\src\lib.rs` | `2292` (`pick_and_simulate_inter_encode_partitions`) |
| rvp9 Block8x8 picker bail | `D:\language\rvp9-2\crates\rvp9-encoder\src\lib.rs` | `2660-2711` |
| rvp9 720p picker gate | `D:\language\rvp9-2\crates\rvp9-encoder\src\lib.rs` | `2401` (`inter_picker_enabled`) |
| F.1 HLD foreman 85% Block8x8 quote | `D:\language\rvp9-2\wrk_docs\Track_F\F1_HLD_block_yrd_port.md` | `817` |
| F.2 HLD F.3-blocks-on-F.2 note | `D:\language\rvp9-2\wrk_docs\Track_F\F2_HLD_find_mv_refs.md` | §1 + §8.2 |
| E11 partition-cap refuted | `D:\language\rvp9-2\wrk_docs\E11_partition_size_probe.md` | §1, §3 |
| E14 partition-cap retest | `D:\language\rvp9-2\wrk_docs\E14_partition_size_retest.md` | (full) |
