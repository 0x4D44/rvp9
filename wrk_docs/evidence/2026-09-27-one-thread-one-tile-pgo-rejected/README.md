# One-thread, one-tile PGO training: rejected

The production PGO recipe's 28 training rows were unchanged except for
`--threads=4 --tile-columns=2` becoming `--threads=1 --tile-columns=0`.
This matched the CPU6 parity screen's topology and tested whether a less
concurrent profile would produce a faster, more stable binary. It did not.

Crucible built the candidate from `da116a7f` plus local prototype commit
`9b0bf9f89`. The candidate binary SHA-256 was
`812d054e2199ef7a391ed3ae600c8080f881f887220569adb8bd9bfc63c8cb5a`.
The paired control was the integrated production-PGO binary SHA-256
`3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`.

The direct screen used three 60-frame, 1080p, CPU6 CBR clips with one thread,
zero tile columns, alternating arm and clip order. One warmup per clip/arm was
excluded. All 36 encodes had the expected, identical arm-specific IVF hashes.
Five measured pairs per clip yielded candidate/control process-CPU geometric
means:

| Clip | Candidate / control CPU |
| --- | ---: |
| Ducks | 1.100412 |
| Tree | 1.045882 |
| Town | 1.023415 |
| Equal-clip geometric mean | 1.056079 |

The recipe was slower on every clip and misses the retention criterion. Do not
integrate this build-script change. The five-pair screen was enough to reject;
no second build or longer screen was run. Host-load telemetry was not recorded,
so use process CPU and the uniform direction of the paired results, not wall
time, as the evidence. The source-identical integrated binary's separate
60-pair/clip screen remains 1.057638× pinned libvpx; this candidate was not
directly paired against libvpx.

`raw.csv` has each measured CPU time and output hash. `summary.csv` has the
per-clip ratios. The build manifest is retained as `manifest.txt`.
