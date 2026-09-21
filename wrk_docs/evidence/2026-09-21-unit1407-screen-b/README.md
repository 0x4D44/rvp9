# Unit 1407 screen B — fdct16 butterfly inlining repeat

This is the independent repeat of the Unit 1407 Crucible x86-64 screen. It
uses the same commit, control, clips, encoder settings, and relaxed load policy
as screen A, but a fresh work directory and three paired samples per clip.
Every candidate/control IVF hash is identical.

The complete-run load was 6.320805 busy logical cores. The paired ratios are:

| clip | paired candidate/control |
| --- | ---: |
| ducks | 0.963371x |
| intree | 0.986927x |
| oldtown | 0.972697x |
| corpus | **0.974284x** |

The raw CSV and manifest beside this file preserve all six samples per clip.
