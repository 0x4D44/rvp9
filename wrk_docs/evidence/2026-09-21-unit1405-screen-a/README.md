# Unit 1405 x86 fdct16 second-pass screen

This is the first timing screen for a bounded SSE2 batch of the x86 16x16
forward-DCT second pass. The implementation was tested and then removed after
the screen showed a repeatable regression. The branch therefore records a
rejected experiment only; shipped source remains identical to integrated
commit `8bf671ad`.

## Protocol

- Crucible x86-64 release builds with Rust 1.95.0.
- Canonical 1280x720@30/1 clips, 60 frames each.
- CPU-used 6, CBR 1260 kbps, one encoder thread, tile-columns 0, AQ0, and
  noise sensitivity 0.
- One warmup, three paired samples, two reversed clip-order rotations.
- The relaxed load policy omits the load ceiling. The manifest retains the
  60-second busy-core measurements.
- `paired_ratio` is candidate time divided by control time; values below 1.0
  are faster.

## Results

| clip | separate ratio | paired ratio |
| --- | ---: | ---: |
| ducks | 1.010489x | 1.010137x |
| intree | 1.012648x | 1.022502x |
| oldtown | 1.023118x | 1.024791x |
| corpus | 1.015403x | 1.019123x |

All candidate/control IVF hashes were byte-identical for every clip. The
candidate was slower on every clip, so no repeated long screen or libvpx
comparison was warranted. The raw CSV, summary, manifest, and console output
are retained beside this README.
