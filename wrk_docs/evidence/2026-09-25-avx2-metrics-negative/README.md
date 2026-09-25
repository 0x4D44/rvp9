# AVX2 Hadamard full-metrics prototype: not retained

The Unit 1432 exact Crucible IP map put 142 direct samples at the inlined
full-metrics callsite in `crates/rvp9-core/src/mode_decision/inter.rs:5590`.
Instruction attribution favored vector squared-error work, but the native
binary used EVEX-encoded XMM operations, so an AVX2 gain was unproven.

The prototype in the unpushed local worktree
`/Users/md/worktrees/rvp9-2/20260925-TSK-HUM-prototype-avx2-hadamard-quant-metrics`
replaces the 8-coefficient SSE2 full-metrics loop with a 16-coefficient AVX2
loop on native AVX2 builds. Other x86-64 builds retain SSE2; AArch64, the
rate-only path, and 32×32 metrics are unchanged. The exact source SHA-256
was `695a2b631516bea760c7571165e94a99153b9a6c3a99b2ad7ea00768b90dbc1b`.
The warning-clean Crucible release EXE SHA-256 was
`213cfd83b5d583792f41af2fcab08ead5f7f84624fa93db75959cb38f31d6d1d`;
the PDB was `0adf36f0e65a5ee862a819e4081fb6accfb4e3b2ba5af2c25450fbf2a6fc8f7b`.
The paired control EXE was the unchanged integrated source at `e588d13f`,
SHA-256 `c371d472690baba94dc1374b9ce4b65dddd3f3ac2381feca362efa25e55f0d98`.
There were no shipped-source changes between that revision and the prototype's
base `f8e05e2b`.

The focused AVX2/SSE2 differential covers 64 and 256 coefficients, extreme
i16 coefficients, negative parameters, and live quantizers. It deliberately
failed when AVX2 `mulhi` was changed to `mullo`, then passed after restoration
(one passed, 2,060 filtered out). AArch64 check, Clippy, formatting, and the
warning-clean Crucible release build passed. The three-clip smoke produced
identical IVF hashes for both arms.

Each independent screen used one warmup per arm per clip and 15 alternating
paired runs per clip: CPU6 CBR1260, single thread, 60 frames, Ducks, In-to-tree,
and Old-town-cross. Process CPU is the primary metric; all 96 encodes per
screen completed and matched the expected clip hash. Ratios below are
AVX2/current geometric means from *all* measured rows, with no exclusions.

| Clip | Screen A (15 pairs) | Screen B (15 pairs) | Combined (30 pairs) |
| --- | ---: | ---: | ---: |
| Ducks | 1.000368 | 1.015704 | 1.008007 |
| In-to-tree | 0.995205 | 0.989622 | 0.992409 |
| Old-town-cross | 0.988689 | 0.996080 | 0.992378 |
| Equal-clip corpus | 0.994743 | 1.000407 | 0.997571 |

Screen A suggested a 0.53% CPU gain; screen B did not repeat it (0.04%
regression). The combined 0.24% advantage is below a defensible retention
signal, while Ducks regressed 0.80% across both screens. No code was integrated
or committed as a production gain. The previous current/libvpx comparison was
1.213193× process CPU; no fresh pinned-libvpx run was warranted for this
negative candidate.

Host-load samples averaged 21.59% in A and 23.17% in B; peaks were 99.95%
and 84.68%. Short Tree/Town encodes often ended before a during-run load
sample, but their child process CPU was recorded. One B Tree run took 15.57 s
wall time while using 1.20 s process CPU; no row was removed. The saved
`raw.csv`, `load.csv`, `identity.csv`, `output-stability.csv`, and manifest files
permit independent recalculation. The CSV/text captures were LF-normalized
after transfer; `screen-a/summary.csv` still has an inherited `Tx4` column
label, but its `avx2` arm and ratios were independently checked from `raw.csv`.
