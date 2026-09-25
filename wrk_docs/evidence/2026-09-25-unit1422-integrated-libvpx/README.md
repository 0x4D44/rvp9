# Unit 1422 — integrated x86 encoder versus pinned libvpx

Crucible built integrated `8759316a` with Rust 1.95.0, native target CPU,
release debug symbols, and no PGO. The encoder SHA-256 was
`d36846d3c80e8ed5c2fff8243b660224b3b7f360e5754045b4f24c68b9d1b4cb`.
The pinned libvpx `vpxenc.exe` SHA-256 remained
`88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d`.

Both screens used the same three 1920×1080, 60-frame clips, CPU6, CBR
1260 kb/s, one thread, one warmup, seven paired iterations, two reversed
rotations, and no load ceiling. Every arm emitted one stable hash per clip;
the RVP9 hashes also matched prior integrated runs. The full commands,
input hashes, timings, and output hashes are in each screen's manifest,
raw CSV, and summary CSV.

Screen A reported 1.671545× paired RVP9/libvpx corpus runtime (1.772176×
separate medians). Its RVP9 rows contained extreme multi-second stalls: for
In-to-tree, 14 runs ranged from 1.36 to 18.46 seconds, while most libvpx
runs were near one second. A read-only host check just after the run showed
86–97% total CPU, with Defender alone using about six to eight logical
cores. Deltic health also warned of 6.6 GB nonpaged and 7.6 GB paged pool,
including a rising nonpaged trend. The timing and host-state observations
make Screen A unsuitable as a current parity estimate; they do not prove
which process caused each stall.

After the CPU spike cleared, Screen B used the same warmed binaries and
reported 1.400201× paired corpus runtime (1.392218× separate medians):
Ducks 1.637907×, In-to-tree 1.340160×, Old-town-cross 1.250619× paired.
Two five-second spot checks during this screen found 3–20% total CPU and
low Defender activity. There were still isolated outliers, so 1.400× is a
provisional current measurement, not a precise regression against Unit
1415's 1.230937× result under different host conditions. It does clearly
leave the <=1.05× target unmet. At 1.400×, the illustrative runtime gap is
25% of current RVP9 time if the reference stays fixed.

The read-only hotspot audit found TX4 batching to be the only candidate with
a plausible >1% whole-encode ceiling. Its invariant-bearing design remains
an explicit Arthur decision in `wrk_docs/2026.09.25 - HLD - TX4 batching
diagnostic.md`; no prototype was started in this unit.
