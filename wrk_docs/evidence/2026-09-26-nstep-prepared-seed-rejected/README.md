# NSTEP prepared seed reuse: exact, slower on production PGO

Decision: reject the source patch. The candidate reused the initial clamped MV,
source view, SAD dispatch, and augmented seed SAD across NSTEP diamond restarts.
The restart census found 1,274,418 extra starts on Ducks, but eliminating that
repeated setup did not reduce whole-encoder CPU. Do not land this source change.

The candidate patch is `seed-reuse.patch` against integrated `84d5e0fe`. It was
committed only on an unpushed local prototype branch (`625e016b`), not on trunk.
The source was exact in the focused `rvp9-core` full-pixel suite on both
AArch64 and x86_64 (22 tests), with `cargo fmt --all --check`, AArch64
`cargo check --locked -p rvp9-encoder`, and `git diff --check` green. The new
differential test went red when its reused initial SAD was deliberately set to
zero, then passed after restoration. All 36 paired-screen encodes exited zero
and matched the pinned IVF hashes in `raw.csv`.

Crucible built the candidate with the same 28-row, 60-frame production PGO
training plan as the integrated-code control. The candidate EXE SHA-256 is
`6f1e0a916c24a49e6a6b1e9b0c56f032470e926cfdf740c264ca132ec6975eba`;
the control is
`3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee`.
`pgo-manifest.txt` records the toolchain and training identity. The screen ran
under a `deltic quiet` lease with zero heavy builds, using five alternating
equal-clip pairs plus a warmup per arm. CPU is process time, not wall time.

| Clip | Candidate / current CPU geomean | Change |
| --- | ---: | ---: |
| Ducks | 1.068456× | 6.85% slower |
| Tree | 1.012258× | 1.23% slower |
| Town | 1.029428× | 2.94% slower |
| Equal-clip combined | 1.036449× | 3.64% slower |

Every Ducks pair was slower, so another production screen would not be a
proportionate retention check. No new candidate/libvpx parity number is
claimed. This agrees directionally with the older, narrower seed-SAD cache
rejection in `wrk_journals/2026.07.30 - JRN - production parity drive.md`
(`Unit 1282`), although that used a plain-release 720p corpus.

Artifacts: `identity.csv`, `raw.csv`, and `summary.csv` are the screen outputs;
`seed-reuse.patch`, `build-pgo-seed-crucible.sh`, and `screen-seed.ps1` reproduce
the source and runner setup. The scripts name the historical Crucible scratch
paths, which may be removed after this evidence is archived.
