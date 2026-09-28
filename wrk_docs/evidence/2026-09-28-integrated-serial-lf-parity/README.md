# Integrated serial loop-filter parity audit (2026-09-28)

The packet-overlap serial loop-filter change landed on `origin/main` at
`56ad4100413a885709ebdafc4761c4a1b9a946a7`. A fresh production-PGO
build of that exact commit on Crucible produced `rvp9enc.exe` SHA-256
`43e3c6708310b612909c8252862aa38b424ac6b9c0d13194f6f89b9f1d557fea`.
The pinned libvpx `vpxenc.exe` SHA-256 was
`88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d`.
The build used the checked-in 28-row, 60-frame PGO recipe and 31 raw profiles;
see `pgo-manifest.txt`.

Two independent 30-pair-per-clip screens alternated arm and clip order after
one warmup per arm and clip. Both used the same 60-frame Ducks, Tree and Town
inputs, with input hashes in `identity.csv`. The runner checked exit code and
the expected IVF SHA-256 after every encode. Its process CPU time, not wall
time, is the measured quantity. All 372 encodes exited successfully and
matched the expected hashes (360 paired observations plus 12 warmups).

| Clip | Screen 1 RVP9/libvpx | Screen 2 | Pooled 60 pairs |
| --- | ---: | ---: | ---: |
| Ducks | 1.173932 | 1.174937 | 1.174434 |
| Tree | 0.993196 | 0.988384 | 0.990787 |
| Town | 0.940087 | 0.929466 | 0.934761 |
| Equal-clip geometric mean | **1.031055** | **1.025789** | **1.028419** |

The measured integrated-trunk equal-clip result is below the 1.05× retention
bar in both groups. Ducks remains 17.4% slower than libvpx; this is a suite
average result, not clip-by-clip parity. The preceding candidate/control
screens showed a reproducible 1.058% equal-clip CPU saving from the change,
while Ducks regressed 1.28% against the prior RVP9 trunk. Those screens and
the exact mixed-inter AArch64/x86 checks are retained in
`../2026-09-28-serial-loopfilter-pgo/README.md`.

Treat the remaining margin with care. The identical earlier RVP9/libvpx
images measured 1.057638× in a 60-pair host regime and 1.028672× in a later
30-pair regime. Absolute CPU moved for both images, more for libvpx. These
fresh results prove the current integrated build passed this audit; they do
not prove that every future Crucible regime will stay below 1.05×. Continue
watching Ducks and repeat the paired direct audit if the host regime changes.

`run-integrated.ps1` reuses the checked-in direct runner with only the
integrated image path and pinned hash substituted. `integrated-direct-01/`
and `integrated-direct-02/` contain the raw per-encode records, identities,
and per-clip paired geometric means.
