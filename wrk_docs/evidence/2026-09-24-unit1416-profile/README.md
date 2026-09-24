# Unit 1416 — fresh exact x86 motion-search profile

Crucible profiled integrated `5b6822c05c8d7ff3005902715d618f0ecbfbfd2c`
with 20 serial 60-frame 1920×1080 Ducks encodes: CPU-used 6, CBR
1260 kb/s, one thread, tile 0, AQ0, noise 0, lag 0, KF60. The executable
was a plain native release build with `CARGO_PROFILE_RELEASE_DEBUG=2`, not PGO.
All 20 IVF files were 529,682 bytes and had SHA-256
`8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`.

The input, executable, PDB, ETL, and xperf report SHA-256 values were,
respectively:

- `6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7`
- `8d175c1bffba875f8381461a7be447335ad8ec77847ef83bb07b965474b29f6e`
- `5dca6f9dca6cbc1ecbf924b090231b5fd86a149f590dec3985c3234d41014f2e`
- `e7e4c331f99de00b532905e509f5c04b041baf3dbb70d72535ee544a86e9c0a8`
- `967a5ea885d6212122767f1fa33f2fd80456968dcc1b46e55d686a835ff0cb58`

The 20-encode function report has 40,988,965 weighted `rvp9enc.exe`
samples. `sad4d_at_from_source` has 1,304,011 (3.18%), its
`diamond_search_sad_from_step` caller has 1,756,889 (4.29%), and
`vpx_sad16x16x4d_avx2` has 2,080,005 (5.07%). These are self weights
from distinct symbols, not additive call-tree shares.

DbgHelp resolved the bounded raw-IP extraction against that exact PDB and
module base `0x7ff7bcd10000`; all selected target IPs mapped. The retained
extract spans timestamp ranges `0..5,000,000` and
`5,000,000..60,000,000`, with 40,989 direct rows and 20 image events over
19 unique PIDs. The top direct rows were:

| Symbol | Source line | Direct rows |
|---|---:|---:|
| `sad4d_at_from_source` | `full_pixel.rs:387` | 429 |
| `sad4d_at_from_source` | `full_pixel.rs:412` | 364 |
| `diamond_search_sad_from_step` | `full_pixel.rs:665` | 509 |
| `vpx_sad16x16x4d_avx2` | `sad4d_avx2.rs:229` | 1,991 |

Line 229 is the generated wrapper call into the inlined kernel, so this
mapping does not prove which instruction *inside* the AVX2 arithmetic is
hot. The line CSVs and summaries beside this file preserve the direct-IP
evidence. Neither the WPR console nor xperf report reported lost, dropped,
overflow, or discarded events; this supports "no reported loss," not an
independent zero-loss proof.

## Equal-stride 16×16 SAD4D experiment

The candidate shares a row offset between source and four references when
their strides match; unequal strides retain the original kernel. Its focused
new differential failed deliberately with lane 0 off by one (`21518` versus
`21519`), then passed after restoring the correct assertion. Crucible passed
the five SAD4D and 20 full-pixel tests; the AArch64 core suite passed 1,719
tests, with two fixture-dependent tests ignored.

Screen A compared the exact integrated profile executable (SHA-256
`8d175c1bffba875f8381461a7be447335ad8ec77847ef83bb07b965474b29f6e`)
with the candidate build (SHA-256
`1b777c0ae62c9539e8cb7715ab37144d953bcda19fb98eb26205260fe0c41c60`).
Both used the same native release/debug=2 flags. Three 1080p clips at the
profile operating point used one warmup, three pairs, and two rotations.
Every IVF hash matched control and was stable within each arm. Paired
candidate/control ratios were Ducks 1.003237, in-to-tree 0.998893, and
old-town-cross 0.994702; equal-clip corpus 0.998938. Load samples ranged
from 1.67 to 6.69 busy logical cores of 32, with no ceiling.

That approximately 0.1% apparent corpus gain is smaller than the
between-pair variation and mixed across clips. Screen B repeated the same
protocol with five pairs per clip. It measured Ducks 1.007265,
in-to-tree 1.001365, and old-town-cross 0.999263; equal-clip corpus
1.002625. Its load samples ranged from 0.81 to 2.18 busy logical cores.
All IVF hashes again matched control and stayed stable. The two screens
do not establish a speed gain, and the larger, quieter repeat favours the
control. Reject the candidate.

The raw rows, summaries, and manifests are `screen-a-*` and `screen-b-*`
beside this file. The exact rejected source diff is archived as
`unit1416-equal-stride.diff.gz`, with hashes in
`unit1416-equal-stride-diff.sha256`. No candidate source change is retained.
