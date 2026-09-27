# NSTEP final-batch reuse across diamond/refinement: no-go

The [earlier NSTEP SAD4D census](../2026-09-26-nstep-sad4d-batch-reuse-census/README.md) counted repeated `(step, center)` keys inside diamond restarts. It did not match a winning diamond's final unit-step batch to the first refinement batch, where the four candidate positions can recur under a different lane order. A [base64-encoded opt-in diagnostic patch](diagnostic.patch.b64) counted exact position matches and whether both sides actually invoked SAD4D; it left search results unchanged.

Crucible built the diagnostic from local commit `4bd06a98` on integrated `b35a34b9` as equivalent remote commit `cfcb35d37323daeadb5238454be466fab68422fc`. The x86 release EXE SHA-256 was `a601386dec51038112629ed31b569c7c402020ca2861b462514b637884135204`. It encoded the three pinned 60-frame 1080p CPU6/CBR1260 inputs with one thread, zero tile columns, AQ/noise disabled, lag zero, KF60. All IVFs matched their canonical integrated hashes. Its process CPU is not performance evidence. The focused lane-permutation/bounds test was deliberately red then green on AArch64 and passed on x86; the AArch64 CLI check, 22 full-pixel tests, x86 release build, formatting, and diff checks passed.

| Clip | NSTEP calls | First refinement SAD4D batches | Position matches | Eligible cross-phase reuse |
| --- | ---: | ---: | ---: | ---: |
| Ducks | 629,530 | 475,652 | 97,071 | 95,690 |
| Tree | 131,265 | 67,313 | 11,162 | 10,908 |
| Town | 125,089 | 42,254 | 5,546 | 5,294 |

Ducks has 95,690 eligible matches, versus 5,814,834 eligible diamond batches in the previous census plus 475,652 first-refinement SAD4D batches here. This added cross-phase reuse is about **1.52% of those batches**, before lookup/bookkeeping cost. Attributing all of the current image's 1,053/18,127 Ducks SAD4D-helper samples to those batches would suggest under 0.1% Ducks CPU saving, but that is **not a rigorous bound** because the sample attribution and per-batch costs may differ. This does not justify a production cache or paired PGO screen. The prior diamond-key cache and this cross-phase seam are both no-go at their observed counts.

[Totals](counts.csv), [raw diagnostic output](ducks.stderr.txt), [runner](run-nstep-census.ps1), and [build identity](identity.txt) retain the observation. The diagnostic branch remains unintegrated.
