# Unit 1407 screen A — fdct16 butterfly inlining

This is the first independent Crucible x86-64 timing screen for commit
`857c80af` (`#[inline(always)]` on `fdct16_butterfly`). The control is the
integrated trunk at `539f1c5e`. Both binaries were built with Rust 1.95.0 on
Crucible and produced identical IVF bytes for every sample.

The run used the pinned 1920x1080@30/1, 60-frame Ducks, In To Tree, and Old
Town clips from the acceptance corpus. It used CPU-used 6, CBR 1260 kb/s, one
thread, tile-columns 0, AQ0, noise sensitivity 0, one warm-up, five paired
samples, and two reversed-order rotations. The relaxed policy omitted
`--max-load1`; the recorded complete-run load was 3.871797 busy logical cores.

| clip | paired candidate/control |
| --- | ---: |
| ducks | 0.957541x |
| intree | 0.989604x |
| oldtown | 0.986698x |
| corpus | **0.977840x** |

Candidate SHA-256: `5c44fc0980c8ea899f85b6b96ab6e09a1f47d149525937b4bc142db2515758bc`.
Control SHA-256: `cfe921b51a2d5a25d20204b37916a9d1920ca8ac787015d4a58c63fa13b64aff`.
The raw CSV and harness manifest beside this file are the authoritative record.
