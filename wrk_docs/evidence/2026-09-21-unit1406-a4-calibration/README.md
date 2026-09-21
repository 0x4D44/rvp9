# Unit 1406 A4 calibration-cache screen

Unit 1406 tested a small encoder setup-cache reuse against integrated control
`1ae7c8f6`. The candidate selected the already-computed baseline or boosted
`InterRdCalibration` for final leaves. The values already include the
frame-level `model_only_large` adjustment, and the output remained byte-exact.

## Protocol

- Crucible x86-64 release builds with Rust 1.95.0.
- Canonical 1280x720@30/1 clips, 60 frames each.
- CPU-used 6, CBR 1260 kbps, one encoder thread, tile-columns 0, AQ0, and
  noise sensitivity 0.
- One warmup, alternating paired samples, two reversed clip-order rotations.
- The relaxed load policy omits the load ceiling. Each manifest retains the
  60-second busy-core measurements.
- `paired_ratio` is candidate time divided by control time; values below 1.0
  are faster.

## Results

| screen | pairs | ducks | intree | oldtown | corpus |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 3 | 1.010137x | 1.022502x | 1.024791x | 1.015899x |
| B | 3 | 0.999381x | 1.003214x | 0.996074x | 0.999552x |
| C | 15 | 1.005829x | 1.000044x | 1.005986x | 1.003949x |

The long screen C is the decision run and shows a 0.395% regression. Every
candidate/control IVF stream was byte-identical. The candidate was therefore
reverted; no libvpx comparison was warranted for a slower candidate.

Raw CSVs and manifests for all three screens are in the sibling `screen-a`,
`screen-b`, and `screen-c` directories. The build hashes and focused-test
evidence are in `build-manifest.txt`.
