export const meta = {
  name: 'hld-composite-8x8-review',
  description: 'Diverse-lens review panel for the 2-D composite 8x8 fusion HLD',
  phases: [
    { title: 'Review', detail: 'five independent lenses incl. devils advocate' },
  ],
}

// args = { docPath: string, round: number }
const docPath = args.docPath
const round = args.round

const GROUND_TRUTH = `
VERIFIED GROUND TRUTH (read directly from the repo at d:\\language\\rvp9-2 — trust this over the doc):

1. compute_variance_8x8 (crates/rvp9-core/src/mode_decision/model_rd.rs:578) signature TODAY:
   fn compute_variance_8x8(
       reduce: fn(&[u8], usize, &[u8], usize) -> (i32, u32),   // bare fn-ptr type, NOT a named alias
       src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize,
       bw: usize, bh: usize,
   ) -> (u64, i64, [u32; 64], [i32; 64], [u32; 64])
   It loops i in 0..nh, j in 0..nw, calls reduce() per cell, then runs the abs-diff >>6 finisher
   AND total_sse/total_sum accumulation in the SAME loop.

2. CellVarFn (dispatch.rs:1354) ALREADY EXISTS = fn(&[u8],usize,&[u8],usize)->(i32,u32).
   Scalar get_var_sse_8x8 (distortion.rs:160) returns (i32,u32). Per-cell AVX2 get_var_sse_8x8_avx2
   (variance_avx2.rs:269) returns (i32,u32).

3. var_sum_sse_inner (variance_avx2.rs:125) — the existing (w,h)-generic reducer:
   - Holds TWO accumulators: sum_acc, sse_acc (both __m256i).
   - Per row: process w in 16-byte chunks via _mm_loadu_si128 -> _mm256_cvtepu8_epi16 (widens 16 bytes -> 16 i16)
     -> _mm256_sub_epi16 -> accumulate_diff.
   - accumulate_diff: sum_acc += _mm256_madd_epi16(diff, ones16);  sse_acc += _mm256_madd_epi16(diff, diff).
   - Tail: w=8 uses _mm_loadl_epi64 (loads 8 bytes, zero-extends to 128b); w=4 reads a u32.
   - At the END it runs hsum256_epi32 TWICE: once on sum_acc, once on sse_acc. So the per-cell
     get_var_sse_8x8_avx2 ALREADY does exactly 2 hsum256 per cell.

4. _mm256_cvtepu8_epi16 widens 16 input bytes to 16 i16 lanes (lane i = byte i, contiguous).
   _mm256_madd_epi16 folds adjacent i16 pairs into i32: result i32 lane k = a[2k]*b[2k] + a[2k+1]*b[2k+1],
   operating within each 128-bit half independently (no cross-lane).

5. Call sites: the ONLY consumer of the get_var_sse_8x8 dispatch slot is model_rd.rs:706
   (let reduce_8x8 = ...get_var_sse_8x8();). Unit tests at model_rd.rs:1545 and 1564 call the
   scalar crate::dsp::distortion::get_var_sse_8x8 directly (not via the slot accessor).

6. Existing variance_avx2.rs tests: differential_random_all_sizes (512 iters/shape, strides w+0..16),
   differential_edge_content (uniform corner cases), differential_8x8_cell (2048 iters),
   dispatch_round_trip (forced caps.avx2 true vs false). SimdDispatch::build is pub(crate).

7. Hot-path block sizes that actually reach model_rd_for_sb_y_large: bw,bh in {32, 64}.
   So nw,nh in {4, 8}; max grid = 64 cells (64x64 block, Tx8x8).

8. Recent repo history: a Phase-2 "option-A" commit just landed that reuses the partition-descent
   NEWMV search. The speed campaign's stated big lever is Phase-2 (52.7% partition double-pick),
   NOT this PR5. This HLD (PR5) is explicitly the "last safe Phase-1 byte-identity tightening".
`

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['reviewer', 'findings', 'overall_assessment', 'ship_recommendation'],
  properties: {
    reviewer: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'severity', 'title', 'location', 'problem', 'proposed_solution', 'confidence'],
        properties: {
          id: { type: 'string', description: 'short stable id, e.g. SIMD-1' },
          severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
          title: { type: 'string' },
          location: { type: 'string', description: 'section/line of the doc this targets' },
          problem: { type: 'string', description: 'what is wrong or risky, with technical reasoning' },
          proposed_solution: { type: 'string', description: 'concrete fix to apply to the doc/design' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
      },
    },
    overall_assessment: { type: 'string' },
    ship_recommendation: { type: 'string', enum: ['ship_as_is', 'revise', 'do_not_ship'] },
  },
}

const COMMON = `
You are reviewing a High-Level Design (HLD) doc for a video encoder (rvp9, a VP9 encoder in Rust).

THE FILE UNDER REVIEW IS EXACTLY THIS PATH — review ONLY this file:
    ${docPath}
MANDATORY FIRST STEP: Read THAT EXACT PATH in full. Confirm the file's Status/header references "V4"
(it is the "decision-first" revision with the kernel spec in "Appendix A"); if what you opened does not
match, you have the WRONG file — re-Read the exact path above. DO NOT Glob/search for HLD files and do
NOT open any other file in wrk_docs: several SIMILARLY NAMED siblings exist ("2-D composite 8x8 fusion
(speed)", "... V1", "... V2", "... V3") and they are ALL OBSOLETE/WITHDRAWN/SUPERSEDED. Ignore them
entirely. Cite section and line numbers from the file at the exact path above ONLY.

NOTE: This V4 deliberately DEMOTES the kernel design into "Appendix A — Contingent Stage-1 kernel spec"
and leads with a recommendation to DEFER the PR and only run a cheap measurement spike (Stage 0). That
demotion is an intentional, already-reviewed decision — do NOT re-raise "the kernel is over-specified /
should be an appendix / should be deferred" as a new finding; it is already done. Focus on whether any
CRITICAL technical error, false claim, or correctness/safety hazard remains in the doc as written.

You MAY Read/Grep the referenced Rust source files under
d:\\language\\rvp9-2\\crates\\rvp9-core\\src to verify claims (encouraged), and you may Read the
predecessor profiling HLD if you need to check a profiling number — but the DESIGN you review and score
is the one at the exact path above.

${GROUND_TRUTH}

This is review ROUND ${round}. Be specific and technical. Cite doc sections and source lines.
Severity guidance:
- critical = a correctness bug, a false premise that invalidates the design, a safety/UB hazard,
  or something that makes the stated goal unachievable. MUST be fixed before the design can proceed.
- major = a significant weakness, an overstated claim, an unnecessary complexity, or a missing
  consideration that materially affects the design quality.
- minor = polish, clarity, small omissions.

QUALITY PHILOSOPHY YOU MUST APPLY (from the project's CLAUDE.md):
- Do NOT overengineer. Solve the problem at hand. Simplicity and cleanliness matter most.
- This is a dev sandbox: NO migration, NO rollback, NO backwards-compat needed. Do not reward the doc
  for adding those; flag them as unnecessary if present.
- Minimize environment variables; AVOID feature flags. Flag any that are gratuitous.
- The cheapest code is code you didn't write. Eliminating a requirement collapses complexity.
- Be honest about whether this PR is even worth doing given its own stated ~2-4% (and >=+3% ship gate).

Return ONLY the structured output. Do not propose vague "consider X" — give a concrete change the
supervisor can apply to the doc.
`

phase('Review')

const lenses = [
  {
    label: 'simd-microarch',
    prompt: `${COMMON}

YOUR LENS: AVX2 micro-architecture & performance realism. Scrutinize:
- The claim (sec 4.1) that the composite gives "one hsum256_epi32 per cell (down from two per cell today)".
  Is that true? The composite must still reduce BOTH a sum and an sse accumulator per cell. Work out the
  actual hsum count for the composite vs the per-cell kernel and call out any error.
- Register pressure (sec 4.1): the doc claims "up to 8 sum + 8 sse = 16 ymm registers fits AVX2's 16 ymm
  budget without spill". AVX2 has only 16 ymm registers TOTAL. If all 16 hold accumulators, what holds the
  loaded data, the diff, ones16, masks, temporaries? Will the nw=8 (64-wide) case spill? Quantify.
- Option A (per-cell 8-byte loads, sec 4.2) vs Option B (32-byte shared row loads). Which option actually
  delivers the "cross-cell row reuse" benefit claimed in sec 1 cost #3? Does Option A reuse any loads at all?
  If Option A does per-cell 8-byte loads, does the cell-row-interleaved accumulator scheme buy anything for
  Option A, or is it pointless register pressure?
- cost #1 "function-pointer indirection": a single repeated indirect-call target is near-perfectly predicted
  by modern branch predictors. Is the real cost the misprediction, or the non-inlining of the callee body?
  Is the doc's framing accurate?
- Whether the honest +2-4% expectation is plausible given what Option A actually saves.`,
  },
  {
    label: 'correctness-byteid',
    prompt: `${COMMON}

YOUR LENS: byte-identity correctness, integer exactness, memory safety / UB. Scrutinize:
- Does the composite kernel provably produce the SAME per-cell (sum, sse) i32/u32 as nw*nh calls to the
  per-cell reducer? Walk the lane math for both Option A and Option B.
- Overflow (R2): are the per-cell accumulators guaranteed within i32/u32 at the cell boundary? Is the doc's
  bound (|sum|<=16320, sse<=4.16M) correct for an 8x8 cell of 8-bit pixels? Show the arithmetic.
- Memory safety: the AVX2 kernel reads via raw pointers. For Option B's 32-byte loads, what happens at the
  right edge of a block whose width is not a multiple of 32 (e.g. bw=24, nw=3)? Could it read past the
  src/pred buffer? For Option A's _mm_loadl_epi64 (8 bytes), is it safe? What buffer-length precondition
  must the kernel document/debug_assert? Does the slice passed (&mut sse8x8[..nw*nh]) and the src/pred
  full-buffer-with-offsets pattern create any bounds hazard the per-cell &src[s_off..] slice did not?
- The out-param contract: kernel writes sse_out[ri*nw+j]. Is the length precondition asserted? What if
  sse_out.len() != sum_out.len() != nw*nh?
- Does the scalar reference get_var_sse_8x8_grid_scalar (sec 4.4) exactly reproduce the current loop body?
  Note the current finisher loop ALSO computes var8x8 and totals; the scalar grid ref only fills sse/sum.
  Is that split clean and correct?`,
  },
  {
    label: 'rust-integration',
    prompt: `${COMMON}

YOUR LENS: Rust API design, dispatch integration, test adequacy, conformance to existing conventions.
Scrutinize:
- The slot wiring (sec 4.4): does it match the EXISTING get_var_sse_8x8 wiring exactly (CellVarFn pattern,
  cfg arms, accessor)? The doc introduces CellVarGridFn — is the signature/style consistent with CellVarFn
  and VarTable conventions? Any naming nit (the slot is named ..._grid but the kernel fn is ..._grid_inner
  in sec 4.1 but ..._grid_avx2 in sec 4.4 — inconsistent)?
- compute_variance_8x8 signature change: today the first param is a bare fn-ptr type, not CellVarFn. The doc
  changes it to CellVarGridFn. Is that the minimal change? The two unit tests (1545,1564) pass the scalar
  fn directly — they must be re-pointed to get_var_sse_8x8_grid_scalar. Does the doc's test re-pointing
  actually compile given those tests assert sse8x8/sum8x8/var8x8/totals?
- Test plan (sec 5.1): 64 shapes x 256 iters. Is testing all 64 shapes justified, or overkill given only
  {32,64}^2 are reachable (anti-overengineering)? Is the "alternating bands" cross-cell-isolation case the
  RIGHT oracle to catch a lane/cell-swap bug? Is there redundancy with the existing per-cell tests?
- Does keeping BOTH get_var_sse_8x8 AND get_var_sse_8x8_grid slots add ongoing maintenance/duplication? Is
  the scalar grid ref calling the scalar per-cell ref (sec 4.4) clean reuse or needless indirection?
- LOC budget (sec 6, ~370 LOC): realistic? Is ~160 LOC for the kernel a sign it's more complex than "a
  mechanical port"?`,
  },
  {
    label: 'pragmatist-quality',
    prompt: `${COMMON}

YOUR LENS: quality philosophy & pragmatism (the project's "don't overengineer, simplicity wins" ethos).
Scrutinize the WHOLE doc for whether this PR should exist as designed:
- Cost/benefit: ~370 LOC + a second permanent dispatch slot + a second kernel to maintain forever, for a
  stated ~2-4% with a >=+3% ship gate. Is that a good trade? The doc itself says "hold and report" below
  +3%. Given the SIMD reality (Option A saves only call overhead), is the expected value below the gate?
- Is there a SIMPLER design that captures most of the win? E.g. instead of a 2-D interleaved-accumulator
  kernel, just an inlined loop-over-cells that calls the existing var_sum_sse_inner(8,8) body with no
  fn-pointer indirection — fewer registers, no lane masking, trivially byte-identical, far less code. Would
  that get most of the benefit (kill the indirection + per-call epilogue) at a fraction of the complexity
  and risk? Argue it concretely.
- Sequencing: the doc admits Phase-2 (52.7%) is the real lever and a Phase-2 commit just landed. Is spending
  effort on a 2-4% Phase-1 tightening the right priority RIGHT NOW, or should this be deferred/dropped?
- Any gratuitous complexity, premature generality (the {8,16,24,...,64} genericity when only {32,64} is
  used), env vars, or feature flags to cut?
- Does the doc's own kill-criterion (sec 5.3 / R4) make the whole thing low-risk-but-also-low-value?`,
  },
  {
    label: 'devils-advocate',
    prompt: `${COMMON}

YOUR LENS: DEVIL'S ADVOCATE. Your job is to surface what everyone else missed and the IMPLICIT ASSUMPTIONS
the design treats as given. Be relentless. Consider:
- Is the foundational premise even valid? The doc says compute_variance_8x8 is "37.4% per-final-leaf cost"
  and "10-15% of pickmode on >=32x32 leaves". Are those numbers sourced/current, or inherited assumptions?
  A Phase-2 commit just changed the partition descent — does PR5's profiling baseline still hold, or is the
  hot path now different?
- Hidden assumption: that fusing helps at all. What if compute_variance_8x8 is memory-bandwidth bound (it
  reads src+pred for the whole block), not compute/overhead bound? Then removing call overhead changes
  nothing. Does the doc establish it's overhead-bound?
- Hidden assumption: the per-cell (sum,sse) is read back identically. After the kernel writes sse8x8/sum8x8
  arrays to the stack, the finisher reads them. Is there any case where the OLD code's interleaving (compute
  + finish in one pass, hot in registers) was actually FASTER than write-all-cells-then-read-all-cells (two
  passes over a 64-entry array, more L1 traffic, no reuse)? Could the "fusion" be a pessimization?
- Hidden assumption: byte-identity is sufficient proof of no-harm. Is there a perf cliff (the new code is
  slower) that byte-identity tests will NOT catch? The ship-gate covers it, but is the measurement plan
  (foreman + testsrc, median of 3) statistically robust to a 2-4% effect, given run-to-run noise often
  exceeds that?
- What about the interaction with the kept get_var_sse_8x8 slot — dead code risk, two sources of truth?
- Anything about correctness of Option B's lane masks that the "pinned by the differential test" hand-wave
  is papering over? "The exact lane discipline is a micro-architectural choice... pinned by the test" — is
  that a design or a TODO?
- Is "byte-identical" even the right goal, or is it a self-imposed constraint that forecloses a bigger win?
List every implicit assumption you find as a separate finding.`,
  },
]

const results = await parallel(
  lenses.map((l) => () =>
    agent(l.prompt, { label: l.label, phase: 'Review', schema: SCHEMA })
  )
)

return results.filter(Boolean)
