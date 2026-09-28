# Dual-filter luma-prediction equality: no-go

Only **4 of 89,085 Ducks** two-filter selector pairs produced identical active luma predictions. Reusing the second filter's luma model on byte-equal predictions could therefore remove at most four model calls in the canonical 60-frame Ducks encode. This is not a credible parity lever. The diagnostic did not change encoder decisions, and all three 60-frame IVFs matched the integrated hashes. The latest uninstrumented integrated production-PGO comparison remains **1.057638×** pinned libvpx equal-clip process CPU against the ≤1.05× target; these diagnostic runs are not speed measurements.

| Clip | Evaluated two-filter pairs | Equal active predictions | Equal share |
| --- | ---: | ---: | ---: |
| Ducks | 89,085 | 4 | 0.0045% |
| Tree | 50,860 | 173 | 0.34% |
| Town | 47,431 | 436 | 0.92% |

The opt-in diagnostic commit is `f83cd772` on the unintegrated local branch `task/20260928-TSK-HUM-count-equal-dual-filter-luma-predictions`; [its patch](0001-diag-enc-count-equal-dual-filter-luma-predictions.patch) is retained here. Crucible applied the same patch as commit `9a6784d0` in its isolated task worktree. The native x86 release diagnostic EXE SHA-256 was `7112030ad56fdc671c532f2e794d5588333fe7d8eccb7b07a19d7cf84f14379c`. It was a normal release build, not a production-PGO build. The [raw census](census.csv) records every input, output and EXE identity, exit status and count.

The probe runs only when `RVP9_DIAG_DUAL_FILTER_EQUALITY` is set. It compares the two completed predictor slots over active `bw×bh` pixels, respecting the output-backed slot's row stride; scoring, filter rates and selection stay unchanged. The existing selector calls the luma model once per filter, so an exact-prediction reuse could skip only the second call, not prediction construction. The earlier filter-stage census independently counted 178,170 Ducks filter trials, exactly twice this run's pair count.

On AArch64, the new slot-equality test was deliberately made red by inverting its assertion, then restored and passed; 96 inter-picker tests, focused clippy, CLI compile, formatting, and a tiny gate-off/on IVF comparison passed. The lead re-ran the focused test. Native Crucible x86 ran that same focused test successfully. The 60-frame CPU6/CBR1260 single-thread commands used the pinned canonical Ducks, Tree and Town Y4M inputs with `--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf` and the diagnostic environment variable; all three exit codes were zero and output SHA-256s matched the integrated baseline. The run script, stderr reports and IVFs remain under the Crucible diagnostic worktree's `.scratch/dual-filter-equality-60f/`; the source branch is intentionally not integrated.

The next optimization needs a different operation-level redundancy or a measured expensive stall. This count rules out equal-prediction model reuse on the current canonical clips; it does not rule out a different filter-search algorithm, which would have to preserve exact output and pass a separate CPU screen.
