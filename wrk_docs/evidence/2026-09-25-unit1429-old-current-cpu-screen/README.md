# Unit 1429 — old/current x86 process-CPU comparison

This screen tests the historical 12–14% old/current slowdown reported in
Unit 1423. It compares the *retained, hash-verified* old `5b6822c0` encoder
with the integrated-code `e588d13f` encoder on Crucible. The two source
revisions differ in shipped code only by the FDCT16 AVX2 first pass. The
executable and PDB hashes, exact paths, and three Y4M hashes are in
`identity.csv`. Pinned libvpx was hashed but **not run** in this screen.

The scratch-only `run-ab.ps1` ran the same CPU6, CBR 1260 kb/s, 60-frame,
single-thread settings as Unit 1426. It used one warmup per arm and clip,
then 15 serial old/current pairs per clip, alternating arm order and
reversing clip order on even pairs. Process CPU was the primary metric;
wall time and host-load samples were secondary. Each encode had a 90-second
limit and the whole run a 1200-second limit. No row was excluded.

| Clip | Current/old process CPU | Current/old wall |
|---|---:|---:|
| Ducks | 1.005914× | 1.026326× |
| In-to-tree | 1.005358× | 1.008382× |
| Old-town-cross | 1.010800× | 1.011316× |
| Equal-clip corpus | **1.007354×** | **1.015311×** |

I independently recomputed all 45 paired CPU and wall ratios from
`raw.csv`; they match `summary.csv`. All 96 encodes (six warmups plus 90
measured) exited zero without timeout. Every output matched the expected
Unit 1426 IVF SHA-256, and each clip/arm had one stable hash across 16
outputs. The 192 captured stdout/stderr files were empty. Two Ducks CPU
ratios are descriptive Tukey outliers, **retained** in the results.

The host counter returned 102 samples without query errors, but only 33
arrived before their encode exited; 69 returned too late. Short Tree and
Town runs often have no during-run load sample. Process CPU is complete,
but it still depends on host frequency, cache and memory contention, and
SMT neighbours; it does not make this a laboratory-isolated comparison.
The screen **does not reproduce** a uniform 12–14% current-code slowdown.
It does not prove the small 0.7% difference is intrinsic, and it does not
revise the provisional RVP9/libvpx parity ratio. That needs a direct
current-versus-pinned-libvpx comparison under this protocol.

`raw.csv`, `load.csv`, `summary.csv`, `output-stability.csv`, `identity.csv`,
`manifest.txt`, and the exact runner are retained here. The runner SHA-256
is `ceed28f5b69cd744d5400fd21e0eb2e8a71278f94a800a06e93586e4a3ae18e4`.
The checked-in raw CSV SHA-256 is
`fc73a9bdcd75538d5553cedfb151383fdd98b72a71ecd7d57e827a8d37ac3668`.
CSV and manifest files were normalized from Windows CRLF to repository LF;
their remote bytes matched the local transfer before that line-ending-only
change. The original remote raw CSV SHA-256 was
`5d0c48c4a68b42972e1ad8a3e54ee407f2729915dee4c18ca75455241e371272`.
The remote run remains in registered Crucible scratch
`u1429-cpu-ab-d2814479`; its seven transferred evidence files matched
their remote SHA-256 values before copying into this worktree.
