# Inclusive loop-filter stack replay

## Conclusion

The existing ETLs are sufficient to answer the attribution question. Every
target-process sample joined a captured stack, and every unique RVP9 stack
address resolved with the exact paired PDB. The serial fallback call path is a
large enough signal to justify a small prototype, especially for Tree, but the
sampled fractions are not a promised wall-clock gain.

The strongest safe conclusion is an observed sampled-share range, not a speedup
claim. The lower bound on a removable gain remains 0% until an A/B benchmark
proves that an alternate path preserves output and reduces CPU. The observed
resolved share of the complete fallback call block is 6.007% (Ducks), 12.456%
(Tree), and 9.006% (Town) of all target-process samples. Source-resolved
allocation plus copy-in/copy-back samples account for 5.410%, 11.270%, and
8.228%, respectively; this is a candidate removable share, not a guaranteed
saving or a formal bound on hidden runtime work.

With no assumption about source-line-unresolved frames, the conservative bound
from these captures is 0% to 100% of the target-process denominator. The useful
resolved-stack figures above narrow the observed signal but do not replace an
A/B measurement: at least one EXE frame lacks source-line info in 97.845% of
Ducks samples, 97.257% of Tree samples, and 96.606% of Town samples. Those
unresolved frames are not evidence of loop-filter work; they are why the hard
upper bound remains trivial.

## Method and identity

[`replay_loop_filter_stack.py`](replay_loop_filter_stack.py) runs on Crucible in
two passes:

1. `xperf.exe -i <capture.etl> -a dumper -add_fieldnames` is parsed with the
   reviewed `extract-ip.py` parser to retain all target `SampledProfile` rows.
2. The target `(timestamp, TID)` keys are joined to `Stack` rows. Only unique
   `rvp9enc.exe` addresses inside image range `0x7ff626790000..0x7ff6269f3000`
   are sent to DbgHelp with the paired PDB.

The target process denominator includes direct EXE, CRT, kernel, and other
target-process samples. Direct-EXE counts are shown for identity checks only;
module totals are never used as loop-filter cost.

| Clip | PID | Target samples | Direct EXE | Stack keys | Unique RVP9 addresses | Source-line-resolved unique | Median depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ducks | 64104 | 20,092 | 18,127 | 20,092 / 20,092 | 7,953 / 7,953 | 7,932 / 7,953 | 14 |
| Tree | 45240 | 9,698 | 8,195 | 9,698 / 9,698 | 4,316 / 4,316 | 4,303 / 4,316 | 14 |
| Town | 66556 | 10,282 | 8,629 | 10,282 / 10,282 | 4,759 / 4,759 | 4,747 / 4,759 | 14 |

The exact EXE SHA-256 is
`3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`; the
paired PDB SHA-256 is
`c41525ee2ccd56d6bda7e4e9e0880452437d5ac7981bc2af4f1b7c34a71943f8`. The ETL
hashes and per-clip TIDs are recorded in the JSON results. Capture diagnostics
reported zero lost buffers and zero lost events.

## Inclusive attribution

Counts below are target-process sample rows whose joined stack contains the
`frame_loop.rs:5005` fallback call block and a resolved source line in the
specified `loopfilter.rs` range. Categories are disjoint and sum to the
resolved fallback-chain count. Percentages use all target-process samples as
the denominator, not direct EXE rows.

| Clip | Fallback call block | Resolved fallback chain | Alloc/zero `229-236` | Copy-in `238-262` | Filter `264-313` | Copy-back `314-321` | Alloc + copies |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ducks | 1,207 (6.007%) | 1,158 (5.763%) | 241 (1.199%) | 822 (4.091%) | 71 (0.353%) | 24 (0.119%) | 1,087 (5.410%) |
| Tree | 1,208 (12.456%) | 1,143 (11.786%) | 192 (1.980%) | 828 (8.538%) | 50 (0.516%) | 73 (0.753%) | 1,093 (11.270%) |
| Town | 926 (9.006%) | 884 (8.598%) | 230 (2.237%) | 582 (5.660%) | 38 (0.370%) | 34 (0.331%) | 846 (8.228%) |

The exact fallback callsite and call-block counts agree for Ducks and Tree. Town
has one additional call-block sample mapped to an argument line. Allocation
source-line samples also had a runtime frame (`ucrtbase.dll`, `vcruntime140.dll`,
`ntdll.dll`, or `kernelbase.dll`) in 241/241 Ducks, 191/192 Tree, and 230/230
Town rows. That is supporting call-chain evidence, not a runtime-module cost
estimate.

Line information resolved 91.7–91.9% of repeated EXE stack-frame rows; 12–21
unique addresses lacked source lines. The reported category counts therefore
exclude line-unresolved frames and should not be treated as exact operation
budgets. All unique EXE addresses did resolve to symbols, and stack-key coverage
was 100% in every clip.

## Reproduction and artifacts

The remote replay used the existing Crucible task scratch root:

```text
D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s\.scratch\fresh-integrated-pgo-ip\
```

For each clip, the bounded command was:

```text
python D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s\.scratch\fresh-integrated-pgo-ip\replay_loop_filter_stack.py --repo-root D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s --etl-root D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s\.scratch\fresh-integrated-pgo-ip --out-root D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s\.scratch\fresh-integrated-pgo-ip\inclusive-stack --out D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s\.scratch\fresh-integrated-pgo-ip\inclusive-stack\results-<clip>.json --clip <clip>
```

Results are retained as [`results-ducks.json`](results-ducks.json),
[`results-tree.json`](results-tree.json), and [`results-town.json`](results-town.json).
The raw dumper files remain only in Crucible `.scratch` under
`inclusive-stack/<clip>/xperf-dumper.txt`.

Go/no-go: proceed with a narrow allocation/copy-elision prototype and paired
CPU/correctness validation. Do not infer a production gain from these sampled
fractions alone; the next acceptance gate is an isolated microbenchmark or
fresh paired CPU run with output comparison.
