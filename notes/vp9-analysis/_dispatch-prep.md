# Stage 1 dispatch-prep (scratchpad)

In-flight working doc for the Stage 1 driver. Not a deliverable — captures
the downstream prompts and review checklist so they can be fired the moment
A1-A6 land.

The leading underscore on the filename is the convention for "scratchpad,
do not promote to a Stage 1 deliverable."

---

## Review checklist — applied to each of A1-A6 as it lands

For each of `01-architecture.md` through `06-threading-svc.md`:

1. **Structure compliance** (per `notes/vp9-analysis/README.md` "Required
   output structure"):
   - [ ] §1 Scope present
   - [ ] §2 Data structures present, struct defs verbatim
   - [ ] §3 Control flow with call graphs / sequence diagrams
   - [ ] §4 Algorithms with pseudocode + file:line citations
   - [ ] §5 Tables and constants verbatim
   - [ ] §6 RTCD / SIMD surfaces enumerated
   - [ ] §7 Open questions for LLD

2. **Citation discipline:**
   - [ ] Every algorithmic claim has a `third_party/libvpx/<file>:<line>` citation
   - [ ] No "see libvpx for details" hand-waves
   - [ ] Citations resolve (line numbers reasonable for v1.14.1)

3. **Verbatim transcription:**
   - [ ] Load-bearing tables (probability, scan order, speed-features,
         quantizer, search-pattern) are COPIED, not summarised
   - [ ] Struct definitions are copied with original field types
   - [ ] If the doc says "transcribed from foo.c:42-180", a `grep` of the
         line range should match the doc

4. **Scope adherence:**
   - [ ] No bleed into another specialist's territory (cross-references OK,
         duplicated analysis is not)
   - [ ] No Rust code or implementation suggestions

5. **Re-dispatch decision:**
   - [ ] PASS → mark task done, move on
   - [ ] PARTIAL → log gaps in `decisions-needed.md`, re-dispatch with
         narrow scope
   - [ ] FAIL → re-dispatch in full

---

## A7 — Devil's advocate prompt (ready to fire when A1-A6 all pass review)

```
You are agent A7, the devil's advocate, of the Stage 1 deep-dive of libvpx
VP9's real-time encoder.

Working directory: D:\language\rvp9-2. libvpx at third_party/libvpx/ pinned
at v1.14.1 (commit 12f3a2a). Read:
- PLAN.md (locked design decisions D1-D9)
- notes/vp9-analysis/README.md (your role + the quality bar)
- notes/vp9-analysis/01-architecture.md  (agent A1's output)
- notes/vp9-analysis/02-encoder-core.md  (agent A2's output)
- notes/vp9-analysis/03-motion-estimation.md  (agent A3's output)
- notes/vp9-analysis/04-bitstream-entropy.md  (agent A4's output)
- notes/vp9-analysis/05-rate-control.md  (agent A5's output)
- notes/vp9-analysis/06-threading-svc.md  (agent A6's output)

YOUR ROLE: read the six specialist docs as a hostile reviewer. Your goal is
to make the Rust port FAIL in Stage 4 by finding everything the specialists
missed, hand-waved, or got wrong NOW so we can fix it before LLD writing
begins. Be specific. Be cruel.

OUTPUT: notes/vp9-analysis/07-devils-advocate.md with at MINIMUM 20 concrete
challenges. Each challenge:

  ### N. <one-sentence statement of the gap or risk>
  **Affects:** <which specialist doc(s), with §section>
  **libvpx evidence:** <file:line citation showing what's actually there>
  **Why it bites us:** <how this hurts BD-rate parity, byte parity, or
                       real-time performance in Stage 4>
  **Severity:** Blocker / High / Medium / Low
  **Resolution:** <re-dispatch A_x with scope: ... | answer in Stage 2 LLD |
                  accept and document in divergence-ledger.md>

CATEGORIES TO HUNT FOR EXPLICITLY (do not limit yourself to these):

1. **Tables not transcribed verbatim** — paraphrased probability tables,
   summarised scan orders, "see file X" instead of copying the data.
2. **Citation gaps** — claims without file:line; line numbers that don't
   resolve to what's claimed.
3. **Scope gaps** — subsystems that should be in some specialist's scope
   but were skipped (e.g. did A4 miss the segmentation probability tables?
   Did A6 miss row-mt decoder threading?).
4. **Hidden state** — global / static / file-scoped state that doesn't
   appear in any struct definition (libvpx uses some).
5. **Build-time configuration** — `CONFIG_*` macros that change behaviour
   (CONFIG_VP9_HIGHBITDEPTH, CONFIG_VP9_TEMPORAL_DENOISING, CONFIG_*_ENCODER,
   CONFIG_BETTER_HW_COMPATIBILITY, etc.) — which ones do the specialists
   need to handle, and did they?
6. **Threading invariants** — implicit aliasing contracts that won't survive
   Rust's borrow checker. Where will the Rust port need `UnsafeCell` /
   raw pointers? Did A6 catalogue all of them?
7. **RTCD coverage** — did each specialist correctly identify all the SIMD
   kernels in their domain, or did some slip through?
8. **Bug-for-bug compatibility** — are there known libvpx bugs (search
   commit log / TODOs in the source) that affect our domains?
9. **Real-time vs slow-path leakage** — did any specialist over-document
   slow-path RDO code that we explicitly DO NOT need at cpu-used 6-8?
10. **SVC interactions** — every other subsystem has SVC-specific paths;
    did the specialists in A1-A5 acknowledge these or did A6 carry the
    whole burden?

QUALITY BAR: every challenge needs a file:line citation. "I have a feeling
this is wrong" is not a challenge. "Line 1234 of vp9_pickmode.c calls a
function not documented in 02-encoder-core.md" is a challenge.

Do not write Rust code. Do not write outside notes/vp9-analysis/.

Aim for 30-40 challenges if the specialists were thorough; 50+ if they
weren't. Quality and specificity matter more than count beyond the minimum
20.
```

---

## A8 — Synthesizer prompt (ready to fire when A7 lands)

```
You are agent A8, the synthesizer, finalising Stage 1 of the libvpx VP9
real-time encoder deep-dive.

Working directory: D:\language\rvp9-2. libvpx at third_party/libvpx/ pinned
at v1.14.1.

INPUTS to read in this order:
1. PLAN.md
2. notes/vp9-analysis/README.md
3. notes/vp9-analysis/01-architecture.md through 06-threading-svc.md
4. notes/vp9-analysis/07-devils-advocate.md (A7's challenges)

YOUR ROLE: produce notes/vp9-analysis/00-overview.md — a cross-cutting
synthesis that ties the six specialists together. For each devil's-advocate
challenge in A7, you must explicitly state ONE of:

  (a) ACCEPTED — the challenge is valid; fix is in <which doc, what
      change>. If fix requires re-dispatch, list scope.
  (b) DEFERRED — the challenge is valid but is properly Stage 2 LLD work,
      not Stage 1 archaeology. Move to decisions-needed.md with explicit
      framing.
  (c) REJECTED — the challenge is not valid; rationale (with file:line
      citations) why the specialist was correct as written.

OUTPUT STRUCTURE for 00-overview.md:

## 1. Architecture in one diagram
A single ASCII diagram of the libvpx VP9 real-time encoder showing how the
six specialist domains interconnect at runtime. Frame-loop oriented.

## 2. Per-frame timeline
For one inter frame at cpu-used 7: list every subsystem touched, in order,
with the specialist owning each step. This is the "execution skeleton"
the Stage 4a MVP must implement.

## 3. State graph
What state lives where: per-frame mutable, per-encoder persistent, shared
across tile workers, per-thread, etc. Critical for Stage 2's threading LLD.

## 4. Cross-cutting concerns
Things that touch multiple specialists' domains and need a unified
treatment in Stage 2:
- Speed features (A6) consumed by A2, A3, A5 — list the cross-references.
- SVC layer context (A6) consumed by A5 (per-layer RC) and A1 (frame loop) —
  list the cross-references.
- Frame contexts / probability updates (A4) consumed by A2 and A6 — list.

## 5. Devil's advocate disposition
A table: every A7 challenge with one-line ACCEPTED/DEFERRED/REJECTED + the
fix or rationale.

## 6. Verified Stage 0 assumptions
List things the Stage 0 plan assumed about libvpx that the deep-dive has
confirmed (or contradicted). Examples:
- "Real-time path is largely separate from RDO" — confirmed/contradicted?
- "SVC adds ~30% complexity on top of base encoder" — accurate?
- "Threading mirrors a pthread tile-worker model" — accurate?

## 7. Stage 2 entry briefing
A single page summary of what the LLD writers in Stage 2 need to know,
referencing the six specialist docs for detail.

QUALITY BAR: every claim cited to either a specialist doc (with section
reference) or a file:line in third_party/libvpx/. No new analysis — your
job is synthesis, not original research.

Do not write Rust code. Do not write outside notes/vp9-analysis/.
```

---

## decisions-needed.md compilation strategy

After A8 lands:

1. Walk each specialist doc's §7 (Open questions for LLD)
2. Walk A7's challenges marked DEFERRED in A8
3. Walk A8's "cross-cutting concerns" that didn't get resolved inline
4. Merge into `notes/vp9-analysis/decisions-needed.md`, grouping by:
   - **D — Decision required before any Stage 3 implementation**
     (e.g. high-bit-depth support Q1; RTCD generation strategy Q2;
      tile-threading primitives Q3; frame-parallel encode Q4)
   - **L — Decision required during Stage 2 LLD writing**
     (per-LLD detail; e.g. "how do we represent partition decision tree
      in Rust given byte-parity goal?")
   - **R — Risk register** (no decision needed yet, but flagged for later
     stage gates)

Format: numbered (D-01, D-02, ..., L-01, L-02, ..., R-01, ...) so we can
cross-reference from LLDs and PLAN.md.

---

## Status updates as work progresses

- [ ] A1 lands → review → mark task 1 done OR re-dispatch
- [ ] A2 lands → review → mark task 2 done OR re-dispatch
- [ ] A3 lands → review → mark task 3 done OR re-dispatch
- [ ] A4 lands → review → mark task 4 done OR re-dispatch
- [ ] A5 lands → review → mark task 5 done OR re-dispatch
- [ ] A6 lands → review → mark task 6 done OR re-dispatch
- [ ] All six pass → fire A7 with the prompt above
- [ ] A7 lands → review → fire A8 with the prompt above (passing A7 as input)
- [ ] A8 lands → review → compile decisions-needed.md
- [ ] Final commit → update PLAN.md Stage 1 completion log → hand to Arthur
