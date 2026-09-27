# Guarded Hadamard metrics: exact, locally faster, not retained

The source-only prototype is committed locally as `133ea611` (guarded SSSE3/SSE4.1 metrics) plus `aa917fda` (out-of-line helper) on `task/20260927-TSK-HUM-prototype-bounded-hadamard-metrics-fast`. Neither commit is integrated. The integrated-code production-PGO comparison remains **1.099305× pinned libvpx equal-clip process CPU**. The prototype did not produce a repeatable overall gain, so it must not change that baseline.

The one-time guard admits all 256 live Y DC/AC quantizer parameter sets and retains the old SSE2 path for arbitrary parameters. The new x86 differential covers 64- and 256-coefficient tiles, `i16::MIN`/`MAX`, deterministic mixed coefficients, all live indices, and guard-edge/fallback cases. Deliberately replacing the new error subtraction with addition made the test fail; restoring it made it pass. AArch64 `cargo check --locked -p rvp9-encoder`, x86-64 Rosetta and native Crucible focused tests, rustfmt, and diff checks passed. Scratch Rosetta microbenchmarks with live q-indices 30/127/220 found a faster metrics loop and faster Hadamard-plus-metrics chains, but those short synthetic warm-cache samples are not production evidence.

Both candidates used the same 28-row, 60-frame production-PGO recipe as the integrated control. The control EXE SHA-256 was `3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee`; the inlined candidate was `717d85837ca270aa8627a58ff8723902174f362b0efe685d11ca4a644f56e193`, and the out-of-line candidate was `392607587171b1c7c1930837b6af85dc1db6567dd0a68d75b1cd9cc9ce6e2c0a`. The `pgo-*-manifest.txt` files preserve the training identities. Exact candidate PDB inspection showed six nested bounded-helper inline sites in large encoder callers; `#[inline(never)]` yielded one out-of-line helper, zero inline sites, and six direct calls. This proves the intended code-shape change, **not** that layout caused Ducks' loss.

Each screen alternated control/candidate order over 15 measured pairs per clip, plus one warmup per arm. The pinned 60-frame 1080p Ducks, In-to-Tree, and Old-Town Y4Ms used CPU6/CBR1260, one thread/no tiles, AQ0/noise0/lag0/KF60. All **384 encodes** exited zero and matched each clip's canonical IVF SHA-256. Every row is retained in `*-raw.csv`, with no load- or stall-based exclusions. The process-CPU ratio is candidate/control; smaller is better. The equal-clip number is the geometric mean of the three clip ratios, not a direct candidate/libvpx run.

| Candidate and screen | Ducks | Tree | Town | Equal-clip CPU |
| --- | ---: | ---: | ---: | ---: |
| Inlined 01 | 1.052796× | 0.987612× | 0.968003× | **1.002157×** |
| Inlined 02 | 1.106869× | 0.971065× | 1.020585× | **1.031331×** |
| Out-of-line 01 | 1.042947× | 0.971128× | 0.984743× | **0.999126×** |
| Out-of-line 02 | 1.074414× | 0.996300× | 1.007702× | **1.025569×** |

Ducks was slower in 13/15, 15/15, 15/15, and 15/15 measured pairs respectively. The out-of-line variant reduced the first-screen penalty but did not reproduce a net gain. Several wall-time stalls appeared under the relaxed Crucible load policy; their process-CPU observations remain included. This is a retention rejection, not a correctness rejection. The likely next path is a different exact-output kernel or a measured broader Ducks hotspot, not another assumption that this microbenchmark gain transfers.

`screen-bounded-metrics.ps1` and `screen-bounded-metrics-noinline.ps1` preserve the paired commands. Each `*-identity.csv`, `*-raw.csv`, and `*-summary.csv` preserves binary/input identities, process CPU and wall times, output hashes, and the full paired ratios. CSV line endings were normalized for Git without changing values. The candidate local and Crucible worktrees remain unintegrated for forensic reproduction.
