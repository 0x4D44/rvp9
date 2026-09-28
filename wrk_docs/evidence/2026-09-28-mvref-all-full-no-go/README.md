# Realtime MV-reference all-full entry guard: no-go census

The cached scaled-neighbour pass in `crates/rvp9-core/src/inter/mv_ref.rs` can be skipped if every active MV-reference list is already full after the direct-neighbour and previous-frame passes. The exact-output census shows that state is rare, and the existing per-list checks already prevent all scaled-MV add attempts in it. Do not prototype the entry guard for the current parity drive: even a free guard would save only 1.1% of cached-neighbour visits on Ducks, and far less on Tree and Town. This is a work count, **not a CPU gain**.

| 60-frame clip | Scaled-pass entries | All lists full | All-full share | Cached-neighbour visits | Visits avoidable | Avoidable share | Scaled-add attempts avoidable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ducks | 502,630 | 5,483 | 1.091% | 3,944,143 | 43,808 | 1.111% | 0 |
| Tree | 143,679 | 183 | 0.127% | 1,127,626 | 1,462 | 0.130% | 0 |
| Town | 131,975 | 275 | 0.208% | 1,037,923 | 2,200 | 0.212% | 0 |

All entries above used the two-reference resolver; the one-reference resolver reported zero entries on these clips. The diagnostic counts a cached neighbour visit before checking whether it is an inter block, so the avoidable-visit figure is a generous upper bound on useful work. The older exact-IP profile had only 14 Ducks iterator samples in this area ([prior analysis](../../../wrk_journals/2026.09.26%20-%20JRN%20-%20exact%20IP%20no-gos%20and%20w32%20reducer%20screen.md)); multiplying that self-sample share by the rare all-full state would not be a valid CPU prediction. No production PGO candidate was built.

The count-only source is local diagnostic commit `dc126710` on base `d5eb3ac2`, not integrated. Only `mv_ref.rs` (SHA-256 `470bf75bddbf3583b866901e859ba21d308513000c843fa73d34fc7fa07b8c7c`) and `rvp9enc.rs` (SHA-256 `531a48f955860057e71383e2e9a7c467a5871e25da98c16b4c21877f036c3a24`) changed. The native x86 release EXE SHA-256 was `1377258997d072f468862f68cc243de6dc12488ec994c067b53800fd98cd6e2f`. The three inputs and six output hashes are in [identity.txt](identity.txt); gate-on and gate-off each matched the canonical integrated IVF for every clip. A first aborted output capture independently produced the same Ducks counters. The opt-in gate prints after successful encode and changes no search decision.

The focused count test went red when the all-full predicate was deliberately changed to any-full, then green after restoration. AArch64 and native x86 each passed all 57 `inter::mv_ref::tests`; AArch64 CLI check, rustfmt check, and diff check passed. The native release build passed. [The runner](run-mvref-census.ps1) and the [gate-on stderr](ducks-on.stderr.txt) (with Tree and Town siblings) retain the exact invocation and raw counter lines. The large IVFs remain only in the Crucible task worktree's `.scratch/mvref-full-census-20260928-c`.

The separate paired integrated production-PGO result remains [1.057638× pinned libvpx](../2026-09-27-integrated-pgo-direct-repeat/README.md) process CPU at equal clip weight. This no-go does not change that result.
