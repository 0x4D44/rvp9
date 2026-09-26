# NSTEP winner-variance reuse: exact, slower

The production CPU6 `NSTEP` search ranks its winning full-pel MV with a raw
variance. The following full-tree subpixel search evaluates that same center
again. Candidate commit `0651eea1` carried the raw variance into the subpixel
center and recomputed only its rate term. An out-of-plane or unavailable value
fell back to the original evaluator. The patch is archived here but **not
integrated**: the equal-clip paired process-CPU ratio was 1.021802×, slower.

## Evidence

- Base: integrated `02498a1da801f7a3b8a664459a000bd1b9098d91`.
- Exact candidate: local commit `0651eea1a416e300e723142cb73e986fc0613abf`,
  reproduced on Crucible with the three changed Git blob hashes pinned in
  `build-pgo-center-crucible.sh`.
- Regression oracle: the new test compared cached raw variance with an
  independent scalar variance and compared complete subpixel results at an
  ordinary block, a shifted block, and a frame-edge block. A deliberate +1
  mutation in the cached value made the test fail; restoring it made it pass.
- Both AArch64 and x86-64 passed 22 full-pixel, 19 subpixel, and 95 inter-mode
  focused tests, plus an encoder check. Formatting and diff checks passed.
- Crucible PGO build used the pinned 28-row, 60-frame production training
  plan; candidate EXE SHA-256
  `1a7a50b150b937fe54bc1a56039d15cd9e0aaef41a1fc376a3e1f98e69cf2dfa`.
  The integrated NSTEP control EXE SHA-256 was
  `3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee`.
- Five alternating pairs per clip, after one warm-up encode per arm. Both arms
  produced the pinned, identical IVF SHA-256 for every encode. Candidate/control
  process-CPU geometric means: Ducks **1.049037×**, Tree **1.009547×**, Town
  **1.007355×**. Equal-clip geometric mean: **1.021802×**.

`raw.csv`, `summary.csv`, `identity.csv`, and `manifest.txt` hold the measured
values and binary/input identities. `candidate.patch.gz`,
`build-pgo-center-crucible.sh`, and `screen-center.ps1` reproduce the
candidate and screen. The scripts refer to the original Crucible worktree and
would need a new task path if replayed elsewhere.

The duplicate center kernel was real, but carrying the extra state and PGO
layout effects cost more than the saved work. Do not integrate it without a
different implementation and a new paired screen.
