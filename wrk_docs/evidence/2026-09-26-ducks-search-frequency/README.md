# Ducks fractional-search origin: coded leaf count is not the cause

This is a structural discriminator, **not** a performance comparison or a
proposal to skip searches. The earlier paired work census measured 530,902
RVP9 16×16 full-tree search entries on canonical 60-frame Ducks. Pinned
libvpx made 3,494,010 fractional 16×16 SVF calls, or about 349,401 searches
at its observed ten probes per search. The inferred excess is about 181,501
16×16 searches. The libvpx entry count by shape is inferred, not directly
instrumented.

The existing `e1_block_stats` harness decoded both canonical Ducks IVFs with
the same RVP9 decoder. The IVFs came from the prior integrated three-gain
parity run on Crucible at
`D:\worktrees\rvp9-2\20260926-TSK-HUM-measure-combined-integrated-parity-after\.scratch\integrated-three-gains-libvpx-01\`.
RVP9 `ducks-00-candidate.ivf` SHA-256 was
`8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`;
libvpx `ducks-00-control.ivf` SHA-256 was
`b24234d8db46c22e03585f1973b27c2e62315f0464da392a2ecd2769d2124fa8`.
Both match the committed canonical manifest in
`../2026-09-26-combined-integrated-parity/combined-integrated-libvpx.ps1`.

The harness came from code-equivalent integrated source commit `48cab42d`.
Crucible release `e1_block_stats.exe` SHA-256 was
`47b13bd00c16b2e786f6b7363e185e3ab77e80b5ef29eb537a165bff226dee26`.
It ran `--rvp9-ivf` and `--libvpx-ivf` on those files with `--frames 60`,
`--fps 30`, and JSON output. The run exited zero; the JSON reports 60 frames,
59 inter frames, and zero decode failures for each stream. The complete JSON
and Markdown reports are retained here.

| Coded inter-frame measure | RVP9 | libvpx |
| --- | ---: | ---: |
| Distinct 16×16 leaves | 453,190 | 454,966 |
| 16×16 area | 94.829% | 95.201% |
| NEWMV area | 13.273% | 10.585% |
| GOLDEN-reference area | 1.422% | 0.533% |

RVP9 has **1,776 fewer** coded 16×16 leaves, not 181,501 more. Thus the
extra 16×16 fractional-search entries cannot be explained by a larger
*final coded* 16×16 leaf count. That does not prove any search is redundant:
the encoder can search trial blocks that never become coded leaves, and it
can make different LAST/GOLDEN NEWMV attempts within a leaf. The higher coded
NEWMV and GOLDEN area is suggestive, not attribution of the internal calls.

Next discriminator: count actual post-gate subpixel search entries by
`(origin, block size, reference)` on RVP9, separating trial/partition
analysis from final leaf picking. Compare corresponding libvpx counts where
the reference can be instrumented, or use a parity trace. Only then consider
an exact policy or cost reduction. The last *uninstrumented* pinned
RVP9/libvpx process-CPU ratio remains 1.134647×; this run did not remeasure it.
