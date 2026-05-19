# Stage 4 drive ledger

Mirror of `notes/overnight-2026-05-17-stage3.md`. One row per agent dispatch
or supervisor decision. Newest at top; commits referenced by short SHA.

## Goal

Drive Stage 4a (encoder MVP) toward exit. Stage 4a exit gate (PLAN.md §Stage 4a):

1. 720p30 real-time (≤33 ms/frame median on a single x86_64 core)
2. BD-rate within 2% of libvpx at `--cpu-used 7 --rt --tune-content default`
3. All produced bitstreams decode under `vpxdec`, `ffmpeg`, and our decoder
4. Every divergence either fixed or recorded in `notes/divergence-ledger.md`

## Sequencing (supervisor decision, 2026-05-19)

Stage 4a's gate is **continuous** (BD-rate + latency + decode-cleanness),
unlike Stage 3's binary MD5 oracle. Building the encoder without first
building the oracle infrastructure is the trap CLAUDE.md warns about:
"testing oracles encode 'what correct looks like' in a form machines can
check at scale. They are the critical multiplier — without them, compute
hours are worthless."

**Therefore, build oracles before encoder modules:**

1. **S4.O1** — Structured bitstream diff harness (`diff_bitstream` rewrite).
   Inputs: two `.ivf` files. Output: per-frame, per-syntax-element diff
   (uncompressed header, compressed header, tile data, residual coefs).
   Becomes the per-commit oracle for every encoder task.

2. **S4.O2** — Encoder smoke harness. Inputs: raw YUV + config. Output:
   `.ivf` round-tripped through our own decoder, byte-identical pixel
   compare against input where lossless, PSNR where lossy. Even an empty
   encoder skeleton produces SOMETHING that this harness validates.

3. **S4.O3** — `bdrate` + `realtime` audit. Confirm both bench bins
   actually do what their names imply or stub-out cleanly; wire them
   into a CI-skipped feature so they exist for manual gate runs.

After oracles: **task ordering per PLAN.md Stage 4a tasks 1–11** but
gated by S4.O1 verdict on each commit. First module: `bitstream_writer`
(LLD `notes/lld/bitstream.md`), round-tripped via S4.O2.

## Holds / open items

- Lossless q00 WHT path (decoder-side; Stage 3 carry-over). Out of Stage 4
  critical path unless we want lossless encode coverage.
- Pre-existing doctest failure on `Decoder::parse_compressed_partition`
  (commit `20bc60d`). Conformance gate dodges via `--lib --bins --tests`.
- Stage 3 secondary gates (SIMD audit, throughput bench, unsafe-comment
  audit) still pending — revisit before formal Stage 3 sign-off.
- libvpx-stock instrumentation tree (D:/language/libvpx-stock/)
  uncommitted; outside this workspace per scope rules. Useful for
  encoder oracle work too — Arthur to decide on commit-and-tag.

## Trajectory

| # | Date | Sub-task | Agent | Result | Commit |
|---|------|----------|-------|--------|--------|
| — | 2026-05-19 | Stage 3 closeout (PLAN.md ticks, scratch cleanup, journal commits) | supervisor | done | `92fbc4c` |
| S4.01 | 2026-05-19 | Read `notes/lld/bitstream.md` end-to-end; produce structured-diff harness design note | S4.01 | done — design note at `notes/stage4-design/diff-bitstream-harness.md` (656 lines, 4 commits planned) | — |
| — | 2026-05-19 | Supervisor decisions on Q5/Q6/Q7: defer Level 5, `packet=N frame=M` labels, ledger via post-processor (not harness) | supervisor | done | — |
| S4.02 | 2026-05-19 | Implement commit #1 from design: CLI restructure on `diff_bitstream.rs`; preserve Level 0 byte diff exactly; add output-formatter trait; tests + clippy clean | (next dispatch) | pending | — |

## Supervisor notes

- Pattern same as Stage 3 overnight: supervisor (me) dispatches agents;
  agents do contained, well-bounded work; supervisor reviews + commits.
- Agents must NEVER commit; that's the supervisor's job. This keeps
  attribution clean and lets me reject hallucinated artefacts.
- Journal: `wrk_journals/2026.05.19 - JRN - stage4 kickoff.md` (and
  successors). One journal per drive day, prepend new entries.
- Hold for Arthur on architectural calls only. Routine implementation
  per LLD spec is supervisor-decidable.
