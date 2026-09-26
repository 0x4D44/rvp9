# Exact integrated sub-pixel pass profile — 2026-09-26

Fresh Crucible WPR captures of integrated trunk `07edb682280a448c728fc6487fdb133048c47459`
separate the AVX2 bilinear first pass from the rest of sub-pixel variance.
They refine the [same-search reuse census](../2026-09-26-subpel-horizontal-reuse-census/README.md);
they do **not** measure a cache or update the paired RVP9/libvpx performance ratio.

| 60-frame clip | Target-process sampled rows | Retained RVP9 EXE rows | AVX2 sub-pixel-family rows | Full two-pass horizontal first-pass rows | Share of process samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tree | 1,611 | 1,375 | 219 | 45 | 2.79% |
| Town | 1,457 | 1,232 | 192 | 40 | 2.75% |

The first-pass rows are only a *gross sampled self-work target*. A one-entry
horizontal buffer could find a previous matching key in 13.25% of Tree and
12.93% of Town two-dimensional calls. Those call shares cannot simply be
multiplied by sample shares: block shapes, costs, and call paths differ. Lookup,
invalidation, and cache lifetime also cost CPU. This profile does not establish
whether a cache would be net faster. Out-of-line `variance_avx2` samples cannot
be assigned to sub-pixel callers without stacks, so the function-family shares
are not full sub-pixel-operation shares.

The exact PE EXE SHA-256 is
`c0db2195fa6d2ab722fbbd9828c4c6541942f42080e65231ea861714f8540b56`;
its paired PDB SHA-256 is
`8a3ba06d40bd9855cb0ccefb10784f7bc265779289411136111596ba927798aa`.
The PE preferred base is `0x140000000`; the live image base for both captures
was `0x00007ff648520000`. The PDB RSDS GUID/age matched the PE, and DbgHelp
loaded that path. The binary and PDB were hash-checked again after capture.
The remote source worktree was clean at the commit above.

Both encodes used the pinned 60-frame CPU6/CBR command from the earlier
[Unit1434 capture](../2026-09-25-unit1434-three-clip-ip-attribution/capture-clip.ps1)
with exact input hashes. Tree IVF SHA-256 was
`5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41`;
Town was `4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f`.
Both reproduce the pinned integrated outputs. Both WPR captures ended normally;
`xperf -a tracestats` reported **zero lost events and buffers**. Tree ETL
SHA-256 was `8e02c8707df37d2cb7e595bcb5dda93783068513fa413d4f42a65a69327c82f6`;
Town was `cca6b722ba1682f06bd6645aeec8fb0a317b95b2f671e60f4853cfd9f4bcbafe`.
Unpaired WPR encode CPU times are not parity measurements.

## Exact-image instruction classification

The source-line map alone cannot separate the inlined loops. GNU objdump 2.42
disassembled the hash-matched EXE, and every sampled RVA in the
`subpel_variance_avx2` symbol family was matched to an instruction. The table
includes inlined wrapper samples; the five ambiguous Tree samples remain
unassigned.

| Machine-code stage | Tree samples (unique RVAs) | Town samples (unique RVAs) |
| --- | ---: | ---: |
| Full two-pass horizontal first pass | 45 (29) | 40 (17) |
| Horizontal-only shortcut, `yoffset=0` | 30 (19) | 31 (19) |
| Full two-pass vertical second pass | 42 (22) | 38 (21) |
| Vertical-only shortcut, `xoffset=0` | 37 (15) | 25 (13) |
| Variance | 60 (33) | 58 (27) |
| Ambiguous setup | 5 (5) | 0 |
| **Total** | **219 (123)** | **192 (97)** |

The generic inlined routine's RVA intervals are: `0x6a3c0–0x6a712` full
horizontal, `0x6a713–0x6aa2f` vertical-only, `0x6aa30–0x6acb7` full vertical,
`0x6acb8–0x6af8f` horizontal-only, and `0x6af90–0x6c419` variance.
Wrapper RVAs were classified individually; the exact allocations are in
`wrapper-rvas.md`. Horizontal-only writes directly to the byte temp and would
not benefit automatically from caching the widened two-pass first-pass buffer.

The canonical EXE/PDB/ETLs and full capture logs remain in Crucible task
`D:\worktrees\rvp9-2\20260926-TSK-HUM-profile-integrated-subpel-pass\.scratch`.
The checked-in per-clip `ip-symbols.csv`, `samples.csv`, and `xperf-tracestats.txt`
retain the sampled addresses and zero-loss evidence if that scratch is later
reaped. They do not replace the exact EXE for re-disassembling instructions.
The latest **paired** RVP9/libvpx process-CPU ratio remains **1.182520×**.
