<!-- deltic:quality-projection:v1 -->
<!-- schema: 1 -->
<!-- projection-digest: sha256:f01fd6ec29648ca2b51a4eef4e1b7ec38e9330f6408fbb7b0fb411a570f67829 -->
<!-- view: board -->

# Quality work board

- **Agent selection:** none
- **Human selection:** programme:activate

| Order | Work key | Variant / Action | Route state | Rank | Evidence gap | Next probe |
|---:|---|---|---|---|---|---|
| 1 | `programme:activate` | Programme activation | Human: runnable | class 0 | human_activation_confirmation, target_date | — |
| 2 | `area:bounded-reliability:stakeholder:target` | Stakeholder / None | Human: runnable | block=true gap=0 importance=High impact=High area=`bounded-reliability` |  | If the confirmed boundary requires production-mode smoke, rerun bounded CPU-used 6–8, denoiser, or spatial workloads; keep the long-duration deferrals explicit. |
| 3 | `area:codec-correctness:stakeholder:target` | Stakeholder / None | Human: runnable | block=true gap=1 importance=High impact=High area=`codec-correctness` |  | Reconcile the missing Mac scalar path and decide whether full kernel-by-kernel SIMD/scalar differential coverage is a release gate; then repeat the accepted exact-source correctness matrix. |
| 4 | `area:compression-quality:stakeholder:target` | Stakeholder / None | Human: runnable | block=true gap=2 importance=High impact=High area=`compression-quality` |  | After the quality oracle and scope are accepted, run the complete exact-source production CBR census and cross-decoder checks on both native platforms. |
| 5 | `area:encoder-throughput:stakeholder:target` | Stakeholder / None | Human: runnable | block=true gap=2 importance=High impact=High area=`encoder-throughput` |  | Once the timing gate and quiet-host route are ready, run all confirmed production speed groups with paired repeats and retained host evidence. |
| 6 | `area:frame-latency:stakeholder:target` | Stakeholder / None | Human: runnable | block=true gap=2 importance=High impact=High area=`frame-latency` |  | Settle the median limits and run the full production per-frame latency matrix on both native platforms, preserving paired runs and p99 evidence. |
| 7 | `area:memory-safety:stakeholder:target` | Stakeholder / None | Human: runnable | block=true gap=2 importance=High impact=High area=`memory-safety` |  | Resolve the shared-infrastructure unsafe-audit scope, close owner-assigned safety findings, then run the missing applicable Miri, native ASan, and parser/full-frame campaigns. |
| 8 | `area:package-consumption:stakeholder:target` | Stakeholder / None | Human: runnable | block=true gap=2 importance=High impact=High area=`package-consumption` |  | Resolve the external archive-test dependency/checksum obstacle, then run the locked tests from each finalized extracted package on both native platforms. |
| 9 | `area:api-and-local-gates:stakeholder:target` | Stakeholder / None | Human: runnable | block=true gap=1 importance=Medium impact=Medium area=`api-and-local-gates` |  | Reduce or explicitly disposition the public rustdoc warnings, and replay each documented local gate on the confirmed candidate and platforms. |
| 10 | `area:corpus-representativeness:stakeholder:target` | Stakeholder / None | Human: runnable | block=false gap=2 importance=High impact=High area=`corpus-representativeness` |  | Choose the required source families or approve a documented narrower boundary; record clip provenance, content coverage, and rights before the quality census. |
| 11 | `area:feature-boundary:stakeholder:target` | Stakeholder / None | Human: runnable | block=false gap=2 importance=High impact=High area=`feature-boundary` |  | Resolve whether frame dropping and EXT_RATECTRL are included; if included, define implementation acceptance and requalify the affected modes. |
| 12 | `area:decoder-throughput:stakeholder:target` | Stakeholder / None | Human: runnable | block=false gap=2 importance=Medium impact=Medium area=`decoder-throughput` |  | Decide whether the 2× comparison is a release gate; if so, define fixtures, aggregation, and platform scope before measuring. |

Hold and Park Areas stay visible in `QUALITY-MAP.md` and `READINESS-CURRENT.md`; they are not runnable work units.
