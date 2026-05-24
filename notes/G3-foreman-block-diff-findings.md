# G.3 — foreman per-block diff findings (F.0.2 oracle)

**Date:** 2026-05-24
**Source data:** `target/track_f_oracle/run_6a123fe6/` (100 kbps) and
`run_6a1240b5/` (25, 50, 250 kbps). Both produced by
`f02-block-oracle` against `tests/corpus/foreman_cif.y4m` at 64 frames.
**rvp9 commit at measurement:** main HEAD = `5a0f8aa` (post-F.4-2 +
F.4-4 landed: `coef_rate` active, `use_prev_frame_mvs=true`).

## Headline

**rvp9's `find_mv_refs` returns `Mv::ZERO` for the NEAREST candidate
much more often than libvpx's**, and **returns `Mv::ZERO` for the NEAR
candidate any time at all** (libvpx never does). At 100 kbps:

| metric | libvpx | rvp9 (inter) |
|---|---:|---:|
| NEARESTMV with mv = (0,0) | 17% | **38%** |
| NEARMV with mv = (0,0) | 0% | **27%** |

This poisons motion compensation on roughly **16 % of all inter
blocks**, directly accounting for a multi-dB PSNR loss on
motion-heavy content like foreman.

The same divergence holds across all four bitrate targets:

| kbps | libvpx NEAREST=0 | rvp9 NEAREST=0 | libvpx NEAR=0 | rvp9 NEAR=0 |
|---:|---:|---:|---:|---:|
| 25  | 48% | 51% | 0% | 35% |
| 50  | 34% | 51% | 0% | 35% |
| 100 | 17% | 38% | 0% | 27% |
| 250 | 11% | 19% | 0% | 13% |

The NEAR-side gap is the cleanest signal: **libvpx is at 0% across
every bitrate**, rvp9 is 13-35%. This is not noise.

## Supporting findings

### F1 — Over-partitioning to 8×8 (100 kbps)

| bsize | libvpx | rvp9 |
|---|---:|---:|
| 8×8 | 13,148 (41%) | **19,536 (64%)** |
| 16×16 | 18,073 (56%) | 10,912 (36%) |
| 32×32 | 897 (3%) | 83 (0.3%) |

rvp9 splits 23 pp more often to 8×8 — partly cause and partly
consequence of poor motion compensation (variance partitioner sees
high-residual blocks → splits → 8×8 leaf inherits the same broken
MVs).

### F2 — Mode distribution (100 kbps)

| mode | libvpx | rvp9 inter |
|---|---:|---:|
| NEARESTMV | 66% | 64% |
| NEWMV | **18%** | **8%** |
| NEARMV | 11% | 12% |
| ZEROMV | **5%** | **15%** |

NEWMV is picked 2.3× less often. ZEROMV is picked 3× more often. The
+10 pp ZEROMV inflation directly mirrors the -10 pp NEWMV deficit.

### F3 — NEWMV motion vectors look like overshoot

| metric (1/8-pel units) | libvpx | rvp9 |
|---|---:|---:|
| count | 5,658 | 2,501 |
| mean magnitude | 17.8 | **107.5** |
| median magnitude | 14 | **34** |
| max magnitude | 260 | **1,610** |

rvp9's NEWMV picks are 6× larger mean magnitude. The max of 1,610 (=
201 pels of motion on a 352×288 frame) is genuinely a search bug —
foreman has small head-and-shoulders motion, not 200-pel translations.
Independent of the `find_mv_refs` issue.

### F4 — Per-bsize×mode (8×8 stratification)

| 8×8 mode | libvpx % of 8×8 | rvp9 % of 8×8 |
|---|---:|---:|
| NEARESTMV | 59% | 64% |
| NEWMV | **18%** | **10%** |
| NEARMV | 15% | 13% |
| ZEROMV | **5%** | **13%** |

The picker IS running on 8×8 leaves on foreman
(`block8x8_picker_enabled = w*h ≤ 307_200` and foreman is 101,376).
The picker is just losing to ZEROMV more often there than libvpx does.

### F5 — Ref-frame distribution

| ref | libvpx | rvp9 |
|---|---:|---:|
| LAST | 97% | 98% |
| INTRA (in inter frames) | 1.5% | 1.7% |
| GOLDEN | 0.6% | **0%** |

rvp9 never picks GOLDEN. Numerically small (~200 blocks) but
symptomatic — the GOLDEN ref isn't reaching the picker.

### F6 — Skip rate

| skip | libvpx | rvp9 |
|---|---:|---:|
| true | 82% | 89% |
| false | 18% | 11% |

rvp9 emits 7 pp more skips → 7 pp fewer residual-encoded blocks at
matched bitrate → that much less of the bit budget reconstructs the
textured content correctly.

## Root cause synthesis

These are not independent bugs. The dominant load-bearing item is the
zero-MV inflation in `find_mv_refs`, which compounds across frames:

1. Frame 1 (first inter): `use_prev_frame_mvs` gate is closed
   (`last_frame_was_key`), so `find_mv_refs` has only spatial
   neighbours to read. The first leaves of the SB have NO inter
   neighbours yet (mi_grid is fresh). `find_mv_refs` returns Mv::ZERO.
2. With NEAREST=NEAR=Mv::ZERO, the picker compares
   {ZEROMV, NEARESTMV-at-zero, NEARMV-at-zero, NEWMV-from-diamond}.
   The first three produce identical predictions; only NEWMV finds
   real motion.
3. If NEWMV's diamond doesn't find a competitive MV (because the
   `int_pro` seed is small, or because `compute_rd_mult`'s lambda at
   high q over-charges the NEWMV bits), the picker commits ZEROMV or
   NEARESTMV-at-zero.
4. That block's mi_grid cell is now Mv::ZERO.
5. The NEXT block's `find_mv_refs` reads this neighbour and contributes
   the (0,0) MV. Frame's mi_grid degenerates toward all-zero.
6. Frame 2's `prev_frame_mvs` salvage reads frame 1's mi_grid. It's
   full of (0,0)s. Frame 2 inherits the poisoned candidate pool.
7. Verified empirically: frame 2 NEAREST=0 drops from frame 1's 40%
   to 20% (salvage kicks in once), but frames 3+ climb back to 25-49%
   (the salvage now reads frame 2's polluted mi_grid).

libvpx breaks this cycle because its picker is MUCH more competitive
at finding NEWMV (Findings F2, F4) when NEAREST is zero — populating
the mi_grid with real motion that propagates to subsequent frames.

## Lever candidates (ranked by expected payoff and tractability)

### L1 — "When NEAREST is Mv::ZERO, fall back to int_pro SB seed"

When `find_mv_refs(...)` returns `mvs[0] == Mv::ZERO`, substitute the
SB-level `int_pro` motion-search MV (already computed once per SB at
`lib.rs:2654`) as the NEAREST candidate. The substitution would:

- Inject a non-zero NEAREST when the spatial+temporal walk fails,
  giving the picker something to evaluate at a low mode-rate cost.
- Break the chicken-and-egg cycle on frame 1 (where no neighbours
  exist).
- Self-bootstrap the mi_grid into a useful state.

**Expected impact:** large — addresses the 21 pp NEAREST gap and the
27 pp NEAR gap directly. Likely +1 to +3 dB on foreman.

**Risk:** the int_pro seed is per-SB, not per-leaf. Using a single
SB-level MV on all leaves of an SB is coarse; libvpx's spatial
neighbour walk is per-leaf. The substitution would be more aggressive
than libvpx's behaviour on the SBs where it fires. testsrc may
regress (testsrc has near-zero motion; int_pro returns Mv::ZERO most
of the time anyway, so the substitution is a no-op there — low risk).

**Cost:** ~50 LOC in `lib.rs::pick_and_simulate_inter_encode_partitions`,
adding a fallback at the `build_mvc` callback site. Bench-driven.

### L2 — "Fix the NEWMV diamond overshoot" (F3 finding)

Investigate why rvp9's NEWMV median is 34 vs libvpx's 14. Possibilities:

- Wrong sub-pel search step pattern
- Wrong diamond shape / shrinkage schedule
- Sub-pel refinement not running (or running with wrong filter precision)

**Expected impact:** medium — would improve NEWMV quality when it
fires, but not the rate at which it fires. ~+0.5 to +1.5 dB.

**Risk:** the existing diamond code has many subtleties; touching it
without understanding which specific bug is in play risks regression.
Needs targeted instrumentation (which leaves overshoot, at what
qindex).

**Cost:** unknown — depends on which diamond step is wrong. Could be
~20 LOC or could be a multi-day audit.

### L3 — "Activate the int_pro fallback within `find_mv_refs` itself"

Variant of L1: instead of substituting at the call site, plumb the
int_pro seed THROUGH `find_mv_refs` and let it populate `list.mvs[0]`
when the spatial+temporal+prev walks produce zero non-zero entries.

**Expected impact:** same as L1.

**Risk:** modifies a core decoder/encoder-shared function. Must keep
the decoder path byte-exact (decoder doesn't have int_pro). Means
the substitution has to be guarded by an encoder-only flag — adding
complexity to a shared module.

**Verdict:** L1 is cleaner; L3 isn't worth the cross-module change.

### L4 — "Enable GOLDEN reference picking"

Investigate why rvp9 never picks GOLDEN (Finding F5). Small numerical
impact (~200 blocks) but suggests the GOLDEN candidate is being
silently rejected somewhere in the picker dispatch.

**Expected impact:** small — ~+0.05 to +0.2 dB.

**Verdict:** worth a quick look but not the priority lever.

## Recommendation

Ship **L1 first**. It's the most direct attack on the dominant
divergence, the cost is bounded, and the diagnostic gives us the
exact criterion for success (rvp9 NEAREST-zero rate should drop
toward libvpx's). Bench against:
- `quality_gate_stage4a_realcontent` (foreman PSNR-delta should drop
  from +9.00 dB toward 0).
- `quality_gate_stage4a_testsrc` (must stay within the halt threshold).
- `realtime_gate_720p30` (must stay ≤ 33 ms).
- Cross-decoder gate (must stay green — bitstream must remain valid).
- Encoder lib tests (must stay 100/100).

If L1 ships net-positive, follow up with L2 (NEWMV overshoot fix)
guided by the same oracle. L4 is a cleanup item.

## Forward-looking items NOT addressed by L1

- The `coef_bits_est` columns are 0 on both legs at this milestone
  (the libvpx patch doesn't emit them; rvp9 emits 0). Per HLD §3.4.1
  this is documented; can be wired in later if a Track needs to compare
  rate-component splits.
- The patch only instruments `vp9_pick_inter_mode` — keyframe / intra
  blocks aren't emitted on the libvpx side. Fine for measuring the
  inter-frame quality gap which is where the 9 dB lives.

## How to re-run

```text
# (one-time) build patched libvpx
$env:PATH = "C:\msys64\usr\bin;C:\msys64\mingw64\bin;" + $env:PATH
& "D:\language\rvp9-2\scripts\build_f02_libvpx.ps1"

# the harness
cargo run -p rvp9-harness --release --bin f02-block-oracle -- \
    --foreman tests/corpus/foreman_cif.y4m \
    --vpxenc third_party/libvpx/build_f02/vpxenc.exe \
    --frames 64 --kbps 25,50,100,250 --out target/track_f_oracle/
```

Then re-run the jq queries in the journal for this session.
