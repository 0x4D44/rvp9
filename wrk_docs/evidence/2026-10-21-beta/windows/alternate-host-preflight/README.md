# Alternate Windows host timing eligibility preflights

NMI is an Intel Core i7-14700KF host, SSH login `ano`, with 28 logical processors. No checkout, build, transfer or timing workload was started.

The bounded 60-second guard sample observed **20.2966 busy logical cores**, mean 72.4879% CPU, against the unchanged 0.75 limit. NMI is not currently eligible for release timing. This is an eligibility sample, not timing gate evidence. Discovery saw no active quiet lease, zero counted heavy processes and 23 low-priority heavy processes. Read-only fixture hashing overlapped the sample; the large sustained load prevents a usable idle-host conclusion.

Available paths:

- Public-origin main clone: `C:\language\rvp9`, HEAD `fb5c0c917038db6c640b3dbc7c2b84b567233333`.
- Existing qualification lane: `C:\worktrees\rvp9\20261017-TSK-HUM@NMI-final-candidate-windows-qualification`.
- Reference executables: lane `build\libvpx\vpxenc.exe` and `vpxdec.exe`; SHA256 hashes exactly match the CRUCIBLE reference pair. The encoder help executes successfully and identifies libvpx v1.14.1.
- Canonical three 720p50 Y4Ms: lane `tests\corpus`, each 691,203,035 bytes, canonical SHA256 identities verified.
- Git Bash: `C:\Program Files\Git\bin\bash.exe`; Git Bash can resolve `shasum`, `cargo`, `deltic`, and `ffmpeg`.
- ffmpeg: `C:\ProgramData\chocolatey\bin\ffmpeg.exe`.
- Rust MSVC: installed 1.95.0 plus default 1.98.1.

The pinned libvpx standalone source is uninitialized in the existing lane: Git queries resolve to its enclosing RVP9 checkout. No reference configuration header was found at `build\libvpx\vpx_config.h`. These prerequisites would need repair or an explicit pinned-source override before unchanged qualification script execution, after host eligibility is established.

Raw discovery, fixture/binary hashes, tool identities, quiet state and sample values are retained beside this report.

ANVIL read-only preflight: 10.60885 busy logical cores across 28 logical processors, mean 37.8887%, against 0.75. Quiet discovery showed zero counted heavy processes and 15 low-priority heavy processes. Public clone `D:\language\rvp9-postfix-main` is at `62c7e13a`; Rust 1.95, Git Bash and ffmpeg exist, but no RVP9 corpus/reference/build directories or registered scratch were found. No staging or builds occurred.

Further Edinburgh desktop eligibility samples, with no builds/transfers: ANNEAL 1.31610 busy logical cores / 24; QUENCH 2.51761 / 24; TEMPER 9.62171 / 24. Every sample exceeds 0.75; full 12-value samples and quiet states are retained. No campaign was staged on these hosts.

KILN read-only preflight also refused 8.17742 busy logical cores / 24 despite zero counted heavy and low-priority processes. Its legacy clones and older references were inspected read-only; no canonical three-scene corpus was present at the inspected main-clone paths. No staging or builds occurred. All alternate-host samples are summarized without splicing campaigns in `eligibility-census.json`.

ANNEAL repeated eligibility refused 3.72270 busy logical cores / 24. It has no inspected RVP9 clone/fixtures or ffmpeg on PATH. No transfers or builds were performed. CRUCIBLE remains the prepared-host retry target after its fair quiet cooldown.

The stale NOMAD host card could not be resolved by DNS; SSH exited 255 before any remote command. The exact bounded probe error is retained. No staging occurred.

ANNEAL quiet-window attempt also refused 6.96045 busy logical cores after a bounded three-minute drain. Counted heavy work stayed zero, while low-priority work rose from two to four during the existing workload interval. The exact 60-second guard and all 12 samples are retained; the lease was released immediately. No staging or builds occurred there.

Evening authoritative refresh (21:03–21:05 local) resampled all six alternate hosts rather than assuming morning jobs remained live. All refused: ANNEAL 1.78781; ANVIL 2.48152; QUENCH 5.18011; KILN 8.09306; TEMPER 8.61446; NMI 27.60781 busy logical cores. ANNEAL/KILN had zero counted heavy and low-priority work; NMI showed two cargo processes and 52 low-priority heavy processes. See evening-census.json and unchanged raw samples. No staging or builds occurred.
