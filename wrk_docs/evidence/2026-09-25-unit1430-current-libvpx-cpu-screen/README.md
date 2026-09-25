# Units 1430–1431 — current x86 encoder versus pinned libvpx, process CPU

These two independent Crucible screens compare the hash-verified integrated
`e588d13f` encoder with pinned libvpx v1.14.1 (`vpxenc.exe` SHA-256
`88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d`).
The current encoder SHA-256 is
`c371d472690baba94dc1374b9ce4b65dddd3f3ac2381feca362efa25e55f0d98`.
Each screen independently rehashed both executables, the encoder PDB, and
the same three pinned 60-frame 1080p Y4Ms. Each used CPU6, CBR 1260 kb/s,
one thread, tile 0, AQ 0, noise 0, lag 0, FPS 30/1, KF 60, one warmup per
arm/clip, then 15 serial paired iterations per clip. Arm order alternated
and clip order reversed on even pairs. The command matches Unit 1422's
libvpx anchor and Unit 1426's current-encoder command exactly.

The primary metric is each process's total CPU time. Wall time and sampled
host load are secondary. No pair was excluded. The same runner SHA-256
`adf6514686f544541129eccc9d5e33bffde5d20e071a1578f50add0d425bf72e`
was used in both fresh registered C: scratches. A separate Ducks smoke
verified both arms' distinct expected IVF hashes before screen A. Each
encode had a 90-second limit; each full screen had a 1200-second limit.

| Clip | Screen A CPU current/libvpx | Screen B CPU current/libvpx | Screen A/B wall current/libvpx |
|---|---:|---:|---:|
| Ducks | 1.344927× | 1.364240× | 1.354751× / 1.367467× |
| In-to-tree | 1.179724× | 1.213582× | 1.192681× / 1.222061× |
| Old-town-cross | 1.104130× | 1.099310× | 1.111487× / 1.104897× |
| Equal-clip corpus | **1.205498×** | **1.220937×** | **1.215522× / 1.226810×** |

I independently recomputed all 45 process-CPU and wall ratios from each
screen's `raw.csv`; the results match both `summary.csv` files. Each screen
has 96 successful encodes (six warmups and 90 measured), zero timeouts or
hash mismatches, and one stable, expected IVF hash per clip and arm. The
RVP9 and libvpx IVF hashes differ as expected; each is checked against its
own prior pinned hash. The screens' equal-weight geometric midpoint is
**1.213193× process CPU** and **1.221153× wall time**. These are descriptive
three-clip CPU6 figures, not a confidence interval or a claim about every
production mode.

Host-load sampling is incomplete: A recorded 40 during-encode and 63 late
counter returns; B recorded 33 during-encode and 65 late, with no query
errors. Short Tree and Town runs often finish before a counter returns.
Sampled total-CPU readings spanned 6.19–99.95% in A and 5.18–71.43% in B.
The relaxed load policy kept every run, including the descriptive Tukey
outliers (A: 1/2/3, B: 2/2/1 on Ducks/Tree/Town). Process CPU avoids
descheduling and I/O waits, but frequency, SMT, cache, and memory contention
can still affect it. The two screen results differ by about 1.3% relative
and are consistent enough to replace Unit 1422's **provisional** 1.400201×
wall estimate for this exact protocol, not to erase that historical result.

The <=1.05× target remains unmet. Relative to the 1.213193× CPU midpoint,
RVP9 would need about **13.5% less process CPU** with libvpx held fixed.
No codec or benchmark code changed and no optimization was retained here.
The next kernel candidate still needs source-level attribution, exact-output
differentials with a deliberate red test, both-architecture validation,
and a repeatable gain against this direct pinned reference.

Each `screen-*/` directory holds the exact runner, raw rows, load rows,
summary, output-stability report, asset identities, and manifest. The
checked-in LF-normalized `raw.csv` SHA-256 values are
`0180d63bb9f0fe9ac9a48cad9937d5817f05e8e1a1c4ac42f9ce6b09fca115c4`
(A) and `f2a3e12b20114d0b97c197cebfceb231e5733a59c3e0e90055e5e6cecff341d9`
(B). Before line-ending normalization, local transfer hashes matched the
remote originals: `8449aebf3ccebe154720d31c39b77e1372d1e5f3414d1ccb3a71de9dd77c4be8`
(A) and `b85e427e2dd46ce7d12dbb4365c1b1148b6b9b8b1a32398bd72cd53d8aa7037c`
(B). The registered remote scratches are `u1430-cpu-ab-95116d02` and
`u1431-cpu-ab-30815c9c`; the seven artifacts from each matched their
transferred SHA-256 values.
