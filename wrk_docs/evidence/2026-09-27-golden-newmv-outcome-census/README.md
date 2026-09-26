# Ducks GOLDEN NEWMV search outcomes

This is an exact-output diagnostic, **not a production speed result**. It extends the earlier [paired full-pel entry census](../2026-09-26-paired-fullpel-entry-census/README.md) with exclusive outcomes for fresh GOLDEN NEWMV searches made in the final RVP9 picker. The same pinned 60-frame 1080p Ducks CPU6/CBR1260 input and options were used on x86-64 Crucible. Its input SHA-256 was `6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7`, and the instrumented IVF reproduced canonical SHA-256 `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`.

| Fresh GOLDEN NEWMV outcome, 59 inter frames | Count | Share |
| --- | ---: | ---: |
| Rejected by post-search gate | 45,719 | 19.36% |
| Deduplicated after search | 1,710 | 0.72% |
| Evaluated, lost/tied against another inter mode | 184,447 | 78.10% |
| Won inter loop, then replaced by intra fallback | 926 | 0.39% |
| **Chosen by final picker** | **3,364** | **1.42%** |
| **Total** | **236,166** | **100%** |

Every fresh search has exactly one terminal outcome, and 236,166 equals the earlier count of GOLDEN full-pel entries. All 59 per-frame rows report `reconciled=true` and `matches_entries=true`; independent CSV summation reproduces the table. The diagnostic separately recorded **zero** full cached picker results and **zero** consumed GOLDEN NEWMV overrides in this run. The final-picker result flows into the encoded leaf description, but this count is taken at the picker boundary; it is not an independent decoded-bitstream tally.

The result closes the "perhaps these are speculative partition searches" hypothesis: these fresh entries occur in final picker calls. It also shows that a blanket GOLDEN skip would change the 3,364 picker winners and cannot be called exact. The 232,802 non-winners are known only **after** search or candidate scoring. Their count alone supplies no safe pre-search condition and no CPU saving; a candidate needs a conservative bound or reusable result available before the expensive work. Pinned libvpx's CPU6 CBR GOLDEN path uses integer projection and fractional refinement rather than the counted full-pel wrapper, so the cross-codec wrapper-entry gap is not a total motion-estimation-work ratio.

The opt-in source remains unintegrated: local diagnostic commit `fde00bf3` (after base diagnostic `e8433538`) and code-equivalent clean Crucible commit `85e272c6`, tree `f5e1e21565de192c378218322f517f51c77d66a5`. The [base64-encoded incremental patch](outcome-source.patch.b64) decodes to SHA-256 `f095ce774378d4925dd3e0add1582e5d474333b6bcdb00b501fcb08f158eb651`; the earlier base patch is in the linked census. Decode on macOS with `base64 -d -i FILE > PATCH`. The x86 diagnostic EXE SHA-256 was `1f149651c8b50e0c9065e95ea28c00fc8b579a849170684270d12bccdad706e9`. [Identity and exact commands](identity.txt), [runner](run.ps1), [per-frame CSV](per-frame.csv), [aggregate CSV](aggregate.csv) and [raw stderr](ducks-stderr.txt) preserve the run. The native x86 release census and outcome tests, release encoder build, AArch64 focused tests and encoder check, formatting, and diff hygiene passed. The new counter test was deliberately made red before restoration; an AArch64 eight-frame on/off encode was byte-identical. Neither architecture's instrumented CPU time is used for a speed claim.

The last uninstrumented direct integrated-code RVP9/libvpx result remains [1.099305× equal-clip process CPU](../2026-09-26-nstep-duplicate-variance-pgo/README.md), above the ≤1.05× target. A next performance candidate needs a proof-backed pre-search bound or an exact reusable result, then paired production-PGO Crucible CPU screens. This census does not warrant integrating the diagnostic or changing GOLDEN policy.
