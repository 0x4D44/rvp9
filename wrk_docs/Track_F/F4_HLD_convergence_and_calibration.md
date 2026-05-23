# F.4 — Convergence + lambda recalibration + activation flips + byte-exact re-establishment (HLD)

**Status**: SIGNED-OFF — ready for implementer dispatch (2026-05-23).
**Owner**: Track F lead (Arthur) — implementer assigned post-sign-off.

**Sign-off**: 2026-05-23 — Track F team-lead approved this HLD as written. No revisions; sweep methodology is appropriately flexible within §3.3's structure; R1 escalation path is explicit.
**Depends on**: F.0.1 (PSNR-delta gate), F.0.2 (per-block oracle), F.0.3 (prev-frame MVs), F.1 (block_yrd, infrastructure-only), F.2 (find_mv_refs, infrastructure-only), F.3 (motion-aware partition, infrastructure-only).
**Blocks**: Track F sign-off ("rvp9 quality matches vpxenc within curve-overlap").

---

## 1. Problem statement & motivation

### 1.1 What we inherited

F.1, F.2, and F.3 all shipped **infrastructure-only** with all behavioural toggles defaulting **OFF**:

| Sub-track | Behavioural switch | Default at ship | Effect when ON |
|---|---|---|---|
| F.1 | `RVP9_F1_COEF_RATE_ENABLE=1` | OFF | Adds `coef_rate` to picker's RD rate sum (per F.1 HLD §5.5). |
| F.1 | `RVP9_F1_SKIP_TIE_BREAK_ENABLE=1` | OFF | Runs the libvpx skip-flag tie-break (`vp9_pickmode.c:2355-2362`). |
| F.2 | `EncoderConfig.use_prev_frame_mvs` (literal const in `frame_loop.rs:173`) | `false` | Enables the `find_mv_refs` previous-frame MV salvage. |
| F.3 | `RVP9_F3_NONRD_PARTITION_ENABLE=1` (or equivalent gate) | OFF | Replaces the variance-only partition with motion-aware RD descent. |

At defaults, the picker is byte-identical to pre-Track-F. Foreman PSNR-delta is unchanged at **+9.07 dB mean**. The Track F goal — match vpxenc quality within curve-overlap (PSNR-delta ≤ +1 dB where BD-rate becomes measurable) — is **entirely on F.4**.

### 1.2 Why the activations are gated OFF

F.1's empirical measurement (divergence-ledger 2026-05-23 F.1 ship-state entry):

| Configuration | Mean Δ | 250 kbps PSNR |
|---|---:|---:|
| Pre-F.1 baseline (shipped at default) | +9.07 dB | 26.96 dB |
| F.1 `coef_rate` ON, tie-break OFF | +10.55 dB | **20.91 dB** (−6 dB) |
| F.1 both mechanisms ON | +12.22 dB | 28.61 dB but 100 kbps collapses |

The libvpx `<< (2 + VP9_PROB_COST_SHIFT)` SATD-to-prob-cost scale is tuned against libvpx's own `compute_rd_mult` and against libvpx's choice of `dist` (coefficient-domain `block_error_fp >> 2`). rvp9's `compute_rd_mult` is line-for-line a port of libvpx's, but rvp9's picker uses **pixel-SSE-of-recon** as `dist` (DV-F1-3) and **DCT qcoeffs** rather than Hadamard for SATD (DV-F1-1). Both of those choices shift the rate/distortion balance away from what the `<< 2` scale was tuned for, so turning on `coef_rate` overprices candidates → picker falls back to worse modes at higher bitrates → PSNR collapses (most visibly the 250 kbps target dropping 6 dB).

### 1.3 What F.4 must deliver

1. **Find a calibration setpoint** (combination of: SATD-to-bits scale exponent, lambda multiplier, `dist` units) where `RVP9_F1_COEF_RATE_ENABLE=1` does not regress foreman PSNR-delta below the +9.07 dB baseline, and ideally improves it.
2. **Activate F.1 mechanisms** in a measured sequence: `coef_rate`, then `skip_tie_break`. Each independently checked against F.0.1 + F.0.2.
3. **Activate F.2** — flip `EncoderConfig.use_prev_frame_mvs` default `false → true`. Verify testsrc stays inside the **F.1_testsrc_baseline + 3 dB halt threshold** (per F.2 HLD §9 R1 mitigation).
4. **Activate F.3** — flip `RVP9_F3_NONRD_PARTITION_ENABLE` default ON (or remove the gate). Verify F.0.2 oracle `partition_match%` against libvpx.
5. **Re-establish byte-exactness discipline** at the cluster's new equilibrium: cross-decode valid VP9; synthetic byte-exact tests untouched; document expected non-byte-exact vs. libvpx (rvp9 ≠ libvpx is fine; rvp9 must be valid VP9).
6. **Tighten `TRACK_F_PSNR_DELTA_THRESHOLD_DB`** in `quality_gate.rs` to the post-F.4 measured value at each activation milestone — never as a prediction, always as a measurement (F.0.1 discipline).

### 1.4 Goal & gate

Reduce foreman PSNR-delta mean from **+9.07 dB → ≤ +1 dB** (the curve-overlap threshold at which BD-rate becomes measurable per F.0.1 §3). +1 dB is the *Track F success goal*; F.4's *immediate* gate is "no regression at each activation step and measurable forward movement of ≥ 0.5 dB across the cluster". If the calibration sweep terminates without forward movement, F.4 escalates to Arthur for goal reassessment (Risk R1).

---

## 2. Inherited work-items (verbatim from F.1/F.2/F.3 ship-state addenda)

### 2.1 From F.1 (HLD §14 / divergence-ledger 2026-05-23 F.1 entry)

1. **Recalibrate** `compute_rd_mult` OR the SATD-to-bits scale (the `<< 2` exponent in `block_yrd.rs:157`) OR the `dist` units (DV-F1-3: pixel-SSE-recon vs coef_dist) so that `RVP9_F1_COEF_RATE_ENABLE=1` does not regress foreman PSNR-delta.
2. **Then** activate `RVP9_F1_SKIP_TIE_BREAK_ENABLE=1` and verify the tie-break improves (or doesn't regress) the metric. The follow-up reviewer note (divergence-ledger 2026-05-23 follow-up) flagged a likely domain mismatch in the tie-break: libvpx compares `RDCOST(rate, dist_coef)` vs `RDCOST(0, sse_pred << 4)`; rvp9 ports it with pixel-SSE on both sides, which can collapse to "argmin sse_after_pred" at high q. F.4's tie-break activation must address this — most likely by switching the encode-side `dist` to `coef_dist` and/or rescaling the skip side.
3. **Candidates listed for the calibration sweep**: SATD scalar correction (the `<< 2`); port `vpx_hadamard_{8,16}` (DV-F1-1); tune the skip-prob context (currently `DEFAULT_SKIP_PROBS[1] = 128`); thread the real per-block skip context (`vp9_get_pred_context_skip`); switch `dist` to `coef_dist` (DV-F1-3 / Q3 resolution revisit).

### 2.2 From F.2 (HLD §12 / divergence-ledger 2026-05-23 F.2 entry)

After F.1 is calibrated, flip `EncoderConfig.use_prev_frame_mvs` default from `false` to `true` at `crates/rvp9-encoder/src/frame_loop.rs:173`. The flip is gated on:
1. F.1 `coef_rate` active (so NEWMV's coefficient-bit savings are visible to RD — closes the F.0.3 §2.5 NEARMV-anchor regression on testsrc).
2. Post-recalibration A/B sweep confirms `use_prev_frame_mvs: true` does NOT regress testsrc PSNR-delta beyond the **F.1 testsrc baseline + 3 dB halt threshold** (per F.2 HLD §9 R1 mitigation).

The doc-comment on `EncoderConfig.use_prev_frame_mvs` (currently records the F.4 deferral) is updated to reflect the new default.

### 2.3 From F.3 (HLD §10)

Activate `RVP9_F3_NONRD_PARTITION_ENABLE=1` (or remove the env-gate). F.3's headline signal is the F.0.2 oracle `partition_match%` (the working target was ≥ 60% on foreman, with the actual sign-off threshold set as a measurement). F.3 also expects:
- Foreman_cif self-imposed realtime ceiling of ~8 ms median (not a hard project gate; if exceeded, fall back to F.3.1 avg-RD early-out per F.3 HLD §7.4 mitigations).
- Foreman_cif 95th-percentile reported but not gated (signal for F.4 tuning).
- 720p30 realtime gate: still ≤ 33 ms median.

### 2.4 Removal of env-gates

Once each activation is verified, F.4 **removes** the env-var gates entirely (not just default-flips them). The codebase ends up with a single behavioural mode, not a maintenance burden of "two configurations to test forever". Default-flipping with the env still readable would invite drift; the env-vars existed solely to ship the infrastructure safely.

---

## 3. Calibration sweep methodology

### 3.1 Sweep variables (in priority order)

F.4 explores a **small number of named knobs**, not a free-form grid. The order reflects the strength of the diagnostic evidence pointing at each one:

| Knob | Location | Range | Notes |
|---|---|---|---|
| **K1**: SATD-to-bits scale exponent | `block_yrd.rs:157` — `(rate_satd as u64) << (2 + VP9_PROB_COST_SHIFT)` | 0..=4 (the `2` constant) | Direct match to libvpx's `<< 2`. Lowering it underweights `coef_rate`; raising it overweights. |
| **K2**: encode-side `dist` units | `inter.rs::evaluate_candidate` | {pixel_SSE_recon (current), coef_dist (DV-F1-3 alt), coef_dist with `>> 2` rescale} | Reviewer hypothesis (2026-05-23 follow-up) flagged this as the likely tie-break root cause; also affects `coef_rate`-active comparisons because `dist` magnitude sets the lambda*rate balance point. |
| **K3**: skip-side `sse` rescale for tie-break | `inter.rs::evaluate_candidate` tie-break branch | {`pixel_SSE_pred` (current), `pixel_SSE_pred << 4`, `pixel_SSE_pred >> 2`} | Mirrors libvpx's `(*sse << 4)`. Only meaningful when K2 picks coef_dist on the encode side. |
| **K4**: lambda multiplier on `compute_rd_mult` | `cost/rd_mult.rs` | scale ∈ {0.5×, 0.75×, 1.0× (current), 1.5×, 2.0×} | Last resort. If K1–K3 cannot find a setpoint, lambda itself is suspect. |

**K1 is the primary knob**: it directly answers "did F.1 overprice rate?". K2/K3 are secondary because they restructure the comparison rather than rescaling it. K4 is the escape hatch.

### 3.2 Measurement signals

**Headline (gate)**: F.0.1 foreman PSNR-delta mean. The sweep terminates successfully when this is ≤ pre-sweep baseline.

**Granular (oracle)**: F.0.2 per-block oracle, with these columns sampled per setpoint:
- `mode_match%` — fraction of SB-leaves where rvp9 picks the same `(ref_frame, mv_mode)` as libvpx.
- `coef_bits_est` mean delta vs. libvpx.
- Fraction of NEWMV picks (sanity — should not collapse to NEARMV-only or zero-mv-only).
- Per-leaf bsize distribution (don't drift to all-Block8x8 at sweep limits).

**Halt sanity**: F.0.1 testsrc PSNR-delta. Must stay below `F.1_testsrc_baseline + 3 dB`. This is the F.2 halt threshold; F.4 inherits it.

**Realtime sanity**: 720p30 median (`realtime_gate_720p30`) must stay ≤ 33 ms throughout.

### 3.3 Search procedure

**Phase A — single-variable bisection on K1.** With K2/K3/K4 held at current values, sweep K1 ∈ {0, 1, 2 (current), 3, 4}. Five setpoints; one foreman + testsrc quality run + one F.0.2 oracle run per setpoint. Pick the K1 value that minimises foreman mean Δ. If the minimum is *worse* than the K1=current=2 baseline (which IS the F.1 infra-only +9.07 dB number), K1 alone is not the answer; proceed to Phase B.

**Phase B — K2 + K3 paired sweep.** K2 has three values; K3 has three values; 3×3 = 9 setpoints, run at the K1 chosen in Phase A. The reviewer's hypothesis (2026-05-23 follow-up) makes (K2=coef_dist, K3=`<< 4`) the favoured setpoint a priori, but the sweep confirms.

**Phase C — K4 fallback** if Phase B doesn't find a non-regressing setpoint. K4 has five values × Phase A/B winning K1/K2/K3 — five additional runs. K4 changes are global and affect every encode (not just the F.1 mechanisms), so K4 movement requires extra justification and a full quality gate re-run.

**Total run budget**: ~5 (Phase A) + ~9 (Phase B) + ~5 (Phase C, if reached) = up to ~19 foreman quality_gate runs + 19 oracle runs + 19 realtime spot-checks. At ~3–5 minutes per quality_gate, this is a ~2-hour interactive sweep or an overnight unattended sweep with a small driver script.

**Stopping rule**: bisection terminates when (a) a setpoint passes the F.0.1 gate at +9.07 dB or below AND (b) one of its neighbours fails. If no setpoint passes after Phase C, escalate to Arthur (Risk R1).

### 3.4 Bookkeeping

Each sweep setpoint produces:
- A row in a sweep results table (markdown in `notes/F4-calibration-sweep.md`): setpoint id, K1/K2/K3/K4 values, foreman mean Δ, max Δ, min Δ, 250 kbps point, testsrc mean Δ, F.0.2 mode_match%, 720p ms median.
- The raw quality_gate output saved per setpoint (artifact dir).

The sweep is reproducible (deterministic encode given a fixed config), so a setpoint can be re-run from its config-id alone.

---

## 4. Activation sequencing

F.4 lands as a series of independently-measured activations. Each step lands as a separate commit (or commit cluster) with its own quality_gate + oracle + realtime numbers. The order is **forced by dependencies**: F.2 depends on F.1's coef_rate; F.3 depends on F.1's block_yrd RD (already infra-only present); F.2's testsrc halt threshold is set off F.1's testsrc value.

### 4.1 Step F.4-1: K-calibration sweep

Per §3.3. Output: one chosen setpoint `(K1*, K2*, K3*, K4*)`.

**No code lands at this step** beyond the sweep-driver scaffolding (if any is needed beyond manual env-var twiddling). The sweep informs Steps F.4-2 through F.4-5.

### 4.2 Step F.4-2: Activate F.1 `coef_rate`

- Apply the K1*/K2*/K3*/K4* values as in-code constants.
- Remove the `RVP9_F1_COEF_RATE_ENABLE` env-gate (the `coef_rate` term is now unconditionally in the rate sum).
- Update `TRACK_F_PSNR_DELTA_THRESHOLD_DB` to the new measured foreman mean Δ (rounded up to nearest 0.1).
- Run: foreman quality_gate, testsrc quality_gate, F.0.2 oracle, 720p realtime gate.
- Verify: foreman ≤ new threshold, testsrc within halt-threshold range, oracle mode_match% improved or stable, realtime ≤ 33 ms.

If any of these fail: roll back the activation, re-enter sweep (Phase B or C).

### 4.3 Step F.4-3: Activate F.1 skip-flag tie-break

- Remove the `RVP9_F1_SKIP_TIE_BREAK_ENABLE` env-gate.
- Apply any K2/K3 changes deferred to this step (per the reviewer follow-up, the tie-break is the mechanism most sensitive to K2/K3).
- Run: same gates as 4.2.
- Verify: foreman strictly ≤ post-4.2 threshold, no testsrc/halt regression.

If the tie-break **regresses** but `coef_rate` alone passed: leave the tie-break gated OFF (keep the env-var as a documented dormant feature) and document as a follow-up. F.1 HLD §10 Q5 anticipated this outcome.

### 4.4 Step F.4-4: Flip F.2 default

- Edit `crates/rvp9-encoder/src/frame_loop.rs:173`: `use_prev_frame_mvs: false → true`.
- Update doc-comment on `EncoderConfig.use_prev_frame_mvs` to reflect new default and remove the F.4-deferral language.
- Run: same gates as 4.2 + F.0.2 oracle's `fraction-non-zero-nearest` should rise on motion content.
- Verify: testsrc strictly ≤ `F.1_testsrc_baseline + 3 dB`. This is the load-bearing measurement for the flip.

If testsrc regresses past the halt-threshold: revert the literal, escalate. F.4 may need a finer split (e.g. flip A18 default but keep an off-switch for testsrc-class content) — that's a separate sub-track if it happens, not in F.4's scope.

### 4.5 Step F.4-5: Activate F.3 motion-aware partition

- Remove the `RVP9_F3_NONRD_PARTITION_ENABLE` env-gate (or default-flip + remove the gate in the same edit).
- Run: same gates as 4.2, with an extra emphasis on F.0.2 oracle `partition_match%` and the foreman_cif self-imposed ~8 ms median ceiling.
- Verify: 720p30 ≤ 33 ms; foreman_cif median ≤ 8 ms or fall back to F.3.1 avg-RD early-out.

### 4.6 Step F.4-6: Final threshold + cleanup

- Set `TRACK_F_PSNR_DELTA_THRESHOLD_DB` to the post-cluster measured value (the convergence floor of the cluster).
- Sweep `notes/divergence-ledger.md` for resolved-by-F.4 entries (DV-F1-1, DV-F1-2, DV-F1-3) and either close them or update them with the F.4 disposition.
- Delete the dead env-gate branches (left over from infra-only ship state).

### 4.7 Activation matrix

| Step | F.1 coef_rate | F.1 skip-tie-break | F.2 default | F.3 descent | Gate update | Expected foreman Δ |
|---|---|---|---|---|---|---|
| Pre-F.4 (ship state) | OFF | OFF | false | OFF | 9.5 dB | +9.07 dB |
| F.4-2 | **ON** | OFF | false | OFF | TBM* | < +9.07 dB |
| F.4-3 | ON | **ON** | false | OFF | TBM | < F.4-2 |
| F.4-4 | ON | ON | **true** | OFF | TBM | < F.4-3 (motion) |
| F.4-5 | ON | ON | true | **ON** | TBM | < F.4-4 |
| F.4-6 (final) | ON | ON | true | ON | final | target ≤ +1 dB |

*TBM = "threshold by measurement" (F.0.1 discipline).

---

## 5. Byte-exact re-establishment

### 5.1 The F.1/F.2/F.3 implementer notes were anticipatory

Several F.1–F.3 commits carry comments like "byte-exact intentionally broken on foreman during F.1-F.3; F.4 re-establishes". Read against the ship-state reality, those comments are **anticipatory** — the sub-tracks were *planned* to break byte-exact, but **landed infrastructure-only**, so there is no current break to re-establish. The picker is byte-identical to pre-Track-F at defaults.

### 5.2 What F.4 will break (deliberately)

The F.4 activations **will** change rvp9-encoded bitstreams: F.1's `coef_rate` changes mode picks; F.2's prev-frame MVs changes refs; F.3's descent changes partitions. After F.4:

- **rvp9 ≠ libvpx**: rvp9-encoder produces different bitstreams from libvpx-encoder for the same input. This was never the goal — the goal is matched **quality**, not matched **bytes**.
- **rvp9 cross-decodes**: rvp9-encoder bitstreams must still be valid VP9 — `vpxdec` and `rvp9-decoder` must both decode them losslessly.
- **rvp9 self-decode round-trip**: rvp9-encoder → rvp9-decoder must produce the pixels rvp9-encoder thinks it encoded (post-recon equality, the existing invariant).

### 5.3 Byte-exact tests that MUST stay green through F.4

- All **synthetic** byte-exact tests in `crates/rvp9-encoder/tests/` (the KF-only, skip-only, single-block, two-frame fixtures). These are testing the bitstream **emitter**, not the picker — F.4 changes the picker, not the emitter. They must not regress.
- **Decoder round-trip** tests: rvp9-decoder vs vpxdec on a corpus of rvp9-encoded fixtures.
- **Self-consistency**: rvp9-encoder's prediction matches rvp9-decoder's reconstruction (`recon == prediction + dequant_idct(coeffs)`).

### 5.4 Byte-exact tests that F.4 is allowed to retire / mark ignored

- The `foreman` cross-decode byte-exact `#[ignore]`d during F.1 (per F.1 §13 step 7). F.4 **leaves it ignored** and updates the explanatory comment: "F.4 changed picker behaviour; rvp9 and libvpx are no longer byte-identical on foreman, but both produce valid VP9 (verified by cross-decode test below)". A *cross-decode* test (rvp9-encoded → vpxdec returns the expected pixel data) takes its place and **is green**.

### 5.5 What F.4 does NOT add

- **No new "byte-exact rvp9 vs libvpx encoder" gate.** That's a fundamentally different (and much harder) project. rvp9 is libvpx-faithful in *behaviour*, not in *byte output* — the picker has multiple legitimate divergences (DV-F1-1 through DV-F1-3 etc.) that F.4 is not removing.

### 5.6 New test added by F.4

A **cross-decoder valid-bitstream gate** in `crates/rvp9-encoder/tests/`:
- Encode foreman_cif with rvp9-encoder at each post-F.4 bitrate target.
- Pipe to `vpxdec`; assert exit 0 and decoded pixel count matches.
- Pipe to `rvp9-decoder`; assert exit 0 and decoded pixels match `vpxdec`'s output bit-for-bit.

This gate is the **byte-exact discipline at the cluster's new equilibrium**: not "rvp9 == libvpx encoder" but "rvp9 produces standards-conformant VP9 that round-trips through every decoder we test".

---

## 6. Test plan

### 6.1 Per activation step (steps F.4-2 through F.4-5)

| Test | Tool | Pass criterion | Source of truth |
|---|---|---|---|
| Foreman PSNR-delta | `quality_gate_stage4a_realcontent` | mean Δ ≤ `TRACK_F_PSNR_DELTA_THRESHOLD_DB` | F.0.1 |
| Testsrc PSNR-delta | `quality_gate_stage4a_testsrc` | mean Δ ≤ `TRACK_F_PSNR_DELTA_THRESHOLD_TESTSRC_DB` (50 dB; this is the coherence sanity, NOT the halt) | F.0.1 |
| Testsrc halt threshold | derived | mean Δ ≤ F.1_testsrc_baseline + 3 dB | F.2 HLD §9 R1 |
| F.0.2 oracle diff | `per_block_oracle` | mode_match% ↑ or stable; coef_bits_est mean Δ ↓ | F.0.2 |
| F.0.2 partition_match% (F.4-5 only) | per_block_oracle | ≥ working target (set as measurement) | F.3 HLD §10 |
| 720p30 realtime | `realtime_gate_720p30` | median ≤ 33 ms | F.3 HLD §10 |
| Foreman_cif realtime (F.4-5 only) | `realtime_gate_foreman_cif` | median ≤ 8 ms (self-imposed) | F.3 HLD §10 |
| Encoder lib tests | `cargo test -p rvp9-encoder --lib` | all 100+ tests pass | repo invariant |
| Synthetic byte-exact | `cargo test -p rvp9-encoder --tests` (the byte-exact tests) | all pass | repo invariant |
| Cross-decoder valid bitstream | new test in §5.6 | vpxdec + rvp9-decoder both decode all post-F.4 fixtures | F.4 §5.6 |

### 6.2 Final cluster sign-off (F.4-6)

In addition to the per-step suite:
- Run the **full** Stage 4a quality_gate on foreman + testsrc + akiyo_qcif (3 corpora).
- Capture the **post-cluster** F.0.2 oracle row table for the divergence-ledger F.4 ship-state entry.
- Run all 9 cross-decoder validity scenarios from §5.6.
- Update `TRACK_F_PSNR_DELTA_THRESHOLD_DB` to the measured foreman mean Δ + 0.1 dB headroom.

### 6.3 No new fixtures

F.4 reuses F.0.1/F.0.2 fixtures (foreman_cif, testsrc, akiyo_qcif) and the existing libvpx reference curves at the existing bitrate operating points. F.4 does not introduce a new corpus — that would be a separate measurement track.

---

## 7. Realtime-budget budget

### 7.1 Inherited components

| Component | Mechanism | Worst-case per-SB cost added |
|---|---|---|
| F.1 block_yrd | DCT + quantize + dequantize + IDCT + SATD per 4×4 sub-block | ~Σ tile cost (already in path; only the SATD scan is "new" overhead — small) |
| F.1 skip-tie-break | One extra RDCOST comparison per candidate | ≪ 1 µs |
| F.2 prev-frame MV salvage | One scan over the prev-frame MV slot per SB | ~1 µs per SB on motion content |
| F.3 motion-aware descent | NONE vs SPLIT RD descent through Block16x16 | F.3 HLD §7.4 estimated ~1.9 ms/frame worst-case on foreman_cif (3-6× per-SB) |

### 7.2 Cumulative budget on foreman_cif

Pre-F.4 baseline median (per F.3 HLD §7.4): well within the 720p30 ≤ 33 ms gate. F.4's worst-case addition is dominated by F.3 (~1.9 ms/frame on foreman_cif). Other contributions are sub-millisecond.

**Hard gate**: 720p30 median ≤ 33 ms throughout F.4. This is non-negotiable.

**Soft self-imposed ceiling**: foreman_cif median ≤ 8 ms (F.3 working target). If exceeded:
1. First mitigation: tune F.3's `partition_search_breakout_thr` to short-circuit more aggressively (F.3 HLD §10 / R2 mitigation 1).
2. Second mitigation: enable F.1's simple-yrd shortcut for high-q blocks (F.1 HLD §10 — currently scalar reference only).
3. Third mitigation: pull back F.3 (leave gated OFF in the final ship, keep F.1+F.2). The cluster still moves PSNR; F.3's incremental gain is the smallest of the three.

### 7.3 Measurement points

Realtime gate runs at every step F.4-2..F.4-5. The Step F.4-5 gate is the load-bearing one for budget. Step F.4-6 runs the gate one more time as final sign-off.

---

## 8. Risks

| ID | Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|---|
| R1 | **Calibration sweep doesn't find a value that improves PSNR.** K1–K4 are exhausted; foreman mean Δ doesn't drop below +9.07 dB. | Medium | High | Escalate to Arthur for Track F goal reassessment. F.4 may need to expand scope into additional libvpx-faithfulness investigations (Hadamard port, quantize_fp port, skip-prob context wiring). Document the negative result thoroughly — the sweep findings remain useful even if F.4 cannot ship the convergence. |
| R2 | **Activations interact non-monotonically** — F.1+F.2 worse than F.1 alone; F.1+F.2+F.3 worse than F.1+F.2; etc. | Medium | Medium | Bisect the activations. The sequencing (§4) is already structured to land each step independently with its own gate. If step F.4-N regresses past F.4-(N-1), defer F.4-N as a sub-issue (keep its activation gated OFF, document, move on). F.4 ships partial activation rather than refusing to ship. |
| R3 | **Realtime budget regression** — foreman_cif blows past the ~8 ms soft ceiling or 720p30 blows past 33 ms. | Medium | Medium | Per §7.2: tune breakout, enable simple-yrd, ultimately pull back F.3. The 720p30 hard gate is non-negotiable; the foreman_cif soft ceiling is. |
| R4 | **testsrc regression beyond halt threshold** at F.4-4 (the F.2 default flip). | Medium | Medium | F.2 HLD §9 R1 mitigation already specifies this case. Revert the literal at frame_loop.rs:173, keep F.4-2/4-3 active, document deferral. testsrc is a content-artefact corner case (per Track_F_baseline_numbers.md); a partial cluster ship is acceptable. |
| R5 | **Cross-decoder gate fails** — rvp9-encoder produces a bitstream that vpxdec rejects. | Low | High | F.4 must NOT ship in this state. Investigate as a Track F bug, not an F.4 tuning issue. The most likely cause would be an emitter bug exposed by a new combination of partition + mv_ref states (F.3 + F.2 interaction). Bisect the activations to find the offending combination. |
| R6 | **F.0.2 oracle becomes a moving target.** The per-block oracle compares rvp9 to libvpx per-leaf; as rvp9 converges, the libvpx leg becomes the constant and the rvp9 leg the variable. If libvpx itself is rebuilt or its instrumentation drifts, F.4 measurements become noisy. | Low | Medium | Pin libvpx at the v1.14.1 build verified by F.0.2. Re-validate the oracle reference at F.4-1. Document the libvpx commit + build flags in the sweep record. |
| R7 | **The sweep finds a setpoint that wins on foreman but loses on testsrc/akiyo.** Calibration is corpus-specific. | Medium | Medium | The sweep's halt-threshold gate (testsrc + 3 dB) catches this. If foreman improves but testsrc/akiyo cross the threshold, the setpoint is rejected and the sweep continues. Worst case: no single setpoint passes all corpora simultaneously, falls into R1. |

---

## 9. Out of scope

- **Major architectural changes to the picker.** F.1/F.2/F.3 are already in place. F.4 *configures* them; it does not replace them.
- **Compound prediction** (libvpx's compound inter modes). Separate track.
- **Intra-in-inter mode evaluation.** rvp9's picker is inter-only inside inter frames; libvpx's `nonrd_pick_sb_modes` mixes intra. Out of scope for F.4 (F.3 HLD §9 also lists this OOS).
- **New SIMD ports.** Scalar `block_yrd` stays. The Hadamard port (DV-F1-1) is listed as a candidate sweep variable but only as a *value substitution* in the existing scalar path — no new SIMD lands at F.4. (A SIMD port would be its own follow-up task once the picker proves it cares.)
- **Variance partitioner lambda recalibration.** The variance partition path (KF + the pre-F.3 inter path) is not touched. F.4 recalibrates only the F.1 RD path (the inter mode-decision picker). The variance path's existing thresholds are taken as ground truth — they're tuned on a different signal (luma variance), not RD.
- **`thresh_freq_fact` adaptive mode-skip** (F.1 HLD §11 OOS). F.4 may *propose* it as a follow-up if the cluster underperforms, but doesn't ship it.
- **Per-frame-type lambda tables.** `compute_rd_mult` already varies by frame type (Key/Inter/ArfGolden); F.4 doesn't add a finer split (e.g. per-q, per-bsize, per-ref).
- **Encoder API changes.** No new public surface area. F.4 is entirely internal.

---

## 10. Implementer-facing summary

The F.4 implementer's task, in order:

1. **Read the F.1/F.2/F.3 ship-state addenda** (HLD §14 / §12 / §10 respectively) and the divergence-ledger F.1/F.2 entries (2026-05-23, ~line 2025+). Verify the F.4 inheritance list against the latest ledger.

2. **Establish the sweep baselines** (§3.2). Run `quality_gate_stage4a_realcontent`, `quality_gate_stage4a_testsrc`, F.0.2 oracle, and `realtime_gate_720p30` at HEAD (defaults). These are the numbers the sweep is improving on. Capture them in `notes/F4-calibration-sweep.md`.

3. **Phase A sweep**: K1 ∈ {0, 1, 2, 3, 4}. Five runs of the foreman quality_gate with the K1 constant patched. Pick the winner. (If you have to manually patch a constant per run, that's fine; if you can env-var it via a temporary scaffold, even better.)

4. **Phase B sweep** if needed: 3×3 K2/K3 grid at the Phase A winner's K1. Nine runs.

5. **Phase C sweep** if needed: 5-value K4 at the Phase B winner's K1/K2/K3. Five runs.

6. **Step F.4-2** (activate `coef_rate`): apply chosen constants in-code, remove `RVP9_F1_COEF_RATE_ENABLE` env-gate, update `TRACK_F_PSNR_DELTA_THRESHOLD_DB` to measured value, run full per-step suite (§6.1). Commit. Update divergence-ledger.

7. **Step F.4-3** (activate skip-tie-break): remove `RVP9_F1_SKIP_TIE_BREAK_ENABLE` env-gate (and apply any K2/K3 changes deferred to this step). Run suite. Commit.

8. **Step F.4-4** (flip F.2 default): edit `crates/rvp9-encoder/src/frame_loop.rs:173` `use_prev_frame_mvs: false → true`. Update doc-comment. Run suite, paying special attention to the testsrc halt threshold. Commit.

9. **Step F.4-5** (activate F.3): remove `RVP9_F3_NONRD_PARTITION_ENABLE` env-gate (default-flip if you prefer to do it in two commits — either way the env-var goes away in F.4-6). Run suite, paying special attention to F.0.2 `partition_match%` and foreman_cif realtime median. Commit.

10. **Step F.4-6** (cleanup + final gate): set `TRACK_F_PSNR_DELTA_THRESHOLD_DB` to the post-cluster measured value. Delete remaining dead env-gate branches. Add the cross-decoder valid-bitstream test (§5.6). Run the full sign-off suite (§6.2). Update divergence-ledger DV-F1-1/2/3 + add an F.4 ship-state entry.

11. **Run `/code-reviewer` and `/code-review-apply-feedback`** per the Track F workflow.

12. **Report in the F.4 ship comment / PR**:
    - Sweep results table (every setpoint, every metric).
    - Per-step gate results (steps F.4-2 through F.4-5).
    - Final cluster numbers (foreman, testsrc, akiyo_qcif mean Δ).
    - Final `TRACK_F_PSNR_DELTA_THRESHOLD_DB` value.
    - Realtime median across all corpora.
    - Cross-decoder gate results.
    - Any deferred activations (e.g. tie-break stayed OFF, F.3 stayed OFF).
    - Divergence-ledger disposition for DV-F1-1/2/3.

**If at any point R1 or R5 fires**: stop the implementation, escalate to Arthur with the sweep data and the diagnostic question, do not unilaterally expand scope. F.4 is allowed to ship a *partial* cluster (per R2/R4 mitigations) but is not allowed to ship a *negative* cluster.

---

## 11. Appendix A — file:line citation table

| Subject | Path | Lines |
|---|---|---|
| F.1 SATD-to-bits scale (K1 target) | `crates/rvp9-core/src/mode_decision/block_yrd.rs` | `157` (the `<< (2 + VP9_PROB_COST_SHIFT)`) |
| F.1 `coef_rate` activation gate | `crates/rvp9-core/src/mode_decision/inter.rs::evaluate_candidate` | (env-var `RVP9_F1_COEF_RATE_ENABLE`) |
| F.1 skip-tie-break activation gate | `crates/rvp9-core/src/mode_decision/inter.rs::evaluate_candidate` | (env-var `RVP9_F1_SKIP_TIE_BREAK_ENABLE`) |
| F.2 `use_prev_frame_mvs` literal | `crates/rvp9-encoder/src/frame_loop.rs` | `173` |
| F.2 `use_prev_frame_mvs` doc-comment | `crates/rvp9-encoder/src/frame_loop.rs` | `141-157` |
| F.3 activation gate | `crates/rvp9-core/src/mode_decision/nonrd_partition.rs` (or equivalent F.3-introduced module) | env-var `RVP9_F3_NONRD_PARTITION_ENABLE` |
| `compute_rd_mult` (K4 target) | `crates/rvp9-core/src/cost/rd_mult.rs` | `60-79` |
| `TRACK_F_PSNR_DELTA_THRESHOLD_DB` constant | `crates/rvp9-encoder/tests/quality_gate.rs` | `64` |
| Stage 4a quality gate runner | `crates/rvp9-encoder/tests/quality_gate.rs` | `786` (`run_stage4a_gate` invocation) |
| F.0.2 oracle | `crates/rvp9-encoder/tests/per_block_oracle.rs` (per F.0.2 HLD) | — |
| 720p realtime gate | `crates/rvp9-encoder/tests/realtime_gate_720p30.rs` (per F.3 HLD §10) | — |
| F.1 ship-state addendum | `wrk_docs/Track_F/F1_HLD_block_yrd_port.md` | §14 (lines 960–1023) |
| F.2 ship-state addendum | `wrk_docs/Track_F/F2_HLD_find_mv_refs.md` | §12 (lines 815–913) |
| F.3 implementer summary | `wrk_docs/Track_F/F3_HLD_motion_aware_partition.md` | §10 (lines 483–523) |
| Divergence-ledger F.1 ship-state entry | `notes/divergence-ledger.md` | ~2025–2079 |
| Divergence-ledger F.2 ship-state entry | `notes/divergence-ledger.md` | ~2112–2161 |
| Divergence-ledger F.1 reviewer follow-up | `notes/divergence-ledger.md` | ~2081–2109 |
| libvpx SATD-to-bits scale (reference) | `D:\language\libvpx\vp9\encoder\vp9_pickmode.c` | (`block_yrd` epilogue, the `<< 2` + `<< VP9_PROB_COST_SHIFT`) |
| libvpx skip-flag tie-break (reference) | `D:\language\libvpx\vp9\encoder\vp9_pickmode.c` | `2355-2362` |
| libvpx `compute_rd_mult_based_on_qindex` (reference) | `D:\language\libvpx\vp9\encoder\vp9_rd.c` | `241-276` |

---

## 12. Acceptance — HLD signed off

- [x] Team-lead reviewed §1-§11.
- [x] No open questions registered at draft; sweep methodology is per-implementer judgement within §3.3's structure.
- [x] HLD signed off — F.4 ready for implementer dispatch.
- [ ] Implementer task created with explicit references to §3 (sweep), §4 (sequencing), §5 (byte-exact discipline), §6 (test plan), §7 (realtime budget), §8 (risks), and §10 (implementer-facing summary).
- [ ] F.4 implementer dispatched after F.3 lands on `main` (sequenced; F.4-1 sweep needs F.3 infrastructure-only landed so K-knob measurements include F.3's mode-eval cost).
