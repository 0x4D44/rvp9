# Emit-side TX selector on eventual skip leaves — opportunity census

This is an output-neutral **work census, not a speed comparison**. Diagnostic source commit `9ec53e65` on `task/20260927-TSK-HUM-measure-dead-emit-side-luma-tx-selector` timed each `choose_inter_luma_tx_size` call and classified it after chroma emission by whether the luma transform was skipped and whether the final leaf was coded. The complete source commit is archived as `0001-diag-enc-measure-selector-work-on-eventual-skip-leav.patch.gz`; it is not integrated. The last direct integrated-code parity result remains **1.099305× pinned libvpx equal-clip process CPU**.

| 59 inter frames | Selector calls | Luma skipped, final skipped | Share of calls | All selector wall time | That bucket's wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ducks | 439,604 | 300,732 | 68.41% | 345.107 ms | 222.680 ms |
| In-to-Tree | 43,817 | 41,858 | 95.53% | 42.250 ms | 39.970 ms |
| Old-Town | 41,550 | 7,509 | 18.07% | 39.806 ms | 6.677 ms |

The other three buckets, in the order luma-emitted/final-coded, luma-emitted/final-skipped, and luma-skipped/final-coded, were Ducks `115630/19916/3326`, Tree `695/0/1264`, and Town `17602/16217/222`. Each stderr file has exactly 59 `[DEAD_TX_SELECTOR]` rows; the per-frame values, not just these sums, are retained. The `*_ns` values time the selector inside an instrumented plain-release build. They are wall durations subject to timer overhead and scheduling, **not removable process CPU**. The plain-release process CPU recorded by this diagnostic run was Ducks 2.250 s, Tree 1.219 s, Town 1.063 s; these are unpaired and must not update parity.

`run-dead-selector.ps1` records the pinned 60-frame CPU6/CBR1260 one-thread command, input and EXE SHA-256, output SHA-256 and process CPU. All three runs exited zero and matched canonical integrated IVF hashes. The input, binary and output identities are in `identity.csv` and `results.csv`. The diagnostic's bucket-merging test was deliberately red once and green after restoration; the lead reran it on AArch64 and Crucible x86. AArch64 `cargo check --locked -p rvp9-encoder --features runtime-profiling`, rustfmt, and diff hygiene passed. Native x86 emitted two pre-existing dead-code warnings in untouched functions. The local three-frame diagnostic on/off encode was also hash-identical.

The possible saving is real work exposure, not a safe shortcut yet. On a luma-skipped leaf, chroma can be nonzero, and its transform size depends on the selected luma size. Any bypass must establish zero chroma EOB under both possible chroma sizes before treating the selector result as dead; it must retain exact prediction/reconstruction, final skip context, coefficient recording, diagnostic-oracle behavior, edge geometry and trellis fallback. An exact-output prototype and paired production-PGO CPU screen are still required before any code integration. Prior selector-to-emission coefficient caches and chroma zero certificates were neutral for different mechanisms; neither measures this proposed deferral.

## Cheap zero-residual certificate: rejected

Follow-up diagnostic commit `adea2112`, archived as `0001-diag-enc-count-exact-chroma-residual-free-skip-leave.patch.gz`, counted whether both chroma source planes equal the final prediction over every visible pixel on a luma-skipped, final-skipped leaf. Such equality would imply zero coefficients under either chroma TX size without an early transform. It found **zero** exact-pixel leaves among all `300,732 / 41,858 / 7,509` eligible Ducks/Tree/Town leaves. This rejects a source-equals-prediction fast path on these clips; it does not prove an early quantization-based path would be slow. Visible-pixel equality alone also does not cover padded TX read footprints at frame edges.

The `uv/` files retain every counter row, binary/input identity, canonical output hash, and process CPU; `run-dead-selector-uv.ps1` records the second command. All three output hashes again matched the integrated baseline. The counter-merge test was deliberately red once, then green; the compact-window pixel check, AArch64 and native x86 focused tests, rustfmt, and runtime-profiling check passed. Both diagnostic source commits remain unintegrated. The next possible prototype would have to build chroma prediction early and prove exact EOB0 under every chroma TX size that the selector could choose, or use an equally exact certificate; that is a separate emit-path ownership change, not an assumed saving from this census.

## Exact alternative-TX quantization: high coverage, unpriced cost

Follow-up diagnostic commit `3e66e8f0`, archived in `alt/0001-diag-enc-count-alternate-chroma-TX-zero-on-skip-leav.patch.gz`, replayed the existing padded chroma residual/fast-quant operation at the **other** possible chroma TX size after normal emission. It ran only on selector-called luma-skip leaves that finished skip with total EOB0, with TX4 enabled, UV capable of TX8, and trellis off. The chosen size was already proven EOB0 by normal emission; the probe tested the remaining TX4-or-TX8 possibility. It read final recon, which equals the MC prediction on these zero-EOB leaves, and did not alter recon or entropy state.

| 59 inter frames | Eligible tested | Other TX EOB0 | Other TX nonzero | EOB0 share |
| --- | ---: | ---: | ---: | ---: |
| Ducks | 300,732 | 298,591 | 2,141 | 99.29% |
| In-to-Tree | 41,858 | 40,054 | 1,804 | 95.69% |
| Old-Town | 7,509 | 7,188 | 321 | 95.73% |

Every clip again produced the canonical IVF SHA-256 with zero exit status. `alt/` holds 59 raw counter lines per clip and the matching binary/input/output identities; `run-dead-selector-alt.ps1` records the exact command. The new AArch64 test was deliberately red before restoration; all three focused diagnostic tests passed on AArch64 and native Crucible x86. The local three-frame diagnostic on/off IVF was byte-identical. This is a strong *eligibility* result for a dual-TX chroma proof, not a process-CPU gain: an early production probe must also pay chroma prediction, TX8/TX4 work, fallback and ownership costs. The three diagnostic encodes' unpaired process CPU is not a speed comparison and does not change 1.099305× parity.

## Timed alternative-TX probe: partial cost only

Diagnostic commit `c9336c5a`, archived in `timed/0001-diag-enc-time-alternate-chroma-TX-probe.patch.gz`, added a timer around the *post-emit* alternative-TX quant probe. Two runs of the same plain-release diagnostic binary on Crucible gave these sums across 59 inter frames per clip:

| Clip | Run 1 dead-selector wall time | Run 1 alt-probe wall time | Run 2 dead-selector wall time | Run 2 alt-probe wall time |
| --- | ---: | ---: | ---: | ---: |
| Ducks | 295.102 ms | 163.323 ms | 219.762 ms | 117.885 ms |
| In-to-Tree | 50.493 ms | 20.794 ms | 39.603 ms | 16.247 ms |
| Old-Town | 7.110 ms | 3.324 ms | 6.645 ms | 3.036 ms |

The probe cost was about 54–55% of the dead-selector timer on Ducks, 41% on Tree, and 45–47% on Town. The eligible and zero-EOB counts were identical in both runs to the table above. All six outputs matched the canonical IVF hashes. `timed/run-01/` and `timed/run-02/` retain raw stderr, identities and results; `timed/run-02.ps1` is the second run's exact command. Unpaired process CPU varied substantially (Ducks 3.547 vs 2.359 s), so neither run is a speed comparison. These instrumented wall timers exclude early chroma prediction, the chosen-TX proof, fallback and scratch ownership. They price only one component of a proposed path; no product CPU gain or new parity number has been measured.
