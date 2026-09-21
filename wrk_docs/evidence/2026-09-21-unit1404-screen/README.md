# Unit 1404 timing evidence

This directory records the Crucible timing screens for commit `84f85dec`
(`perf(core): vectorize AVX2 quantizer AC tail`). The candidate is compared
with integrated control `5b1d653e` and with pinned libvpx revision
`12f3a2ac-libvpx`.

## Protocol

- Crucible x86-64 release builds, Rust 1.95.0, one encoder thread.
- Canonical 1280x720@30/1 clips: 60 frames each, CBR 1260 kbps, CPU-used 6,
  tile-columns 0, AQ0, noise sensitivity 0.
- One warm-up, three paired samples, two rotations, alternating arm order.
- The relaxed Crucible policy omits the load ceiling; manifests retain the
  60-second busy-core measurements.
- `paired_ratio` is candidate time divided by the anchor time; values below
  1.0 are faster.

## Results

| comparison | screen | ducks | intree | oldtown | corpus |
| --- | --- | ---: | ---: | ---: | ---: |
| candidate / control | A | 0.987245x | 0.995030x | 0.988016x | 0.990091x |
| candidate / control | B | 0.986332x | 0.996244x | 0.996933x | 0.993158x |
| candidate / pinned libvpx | fresh | 1.670395x | 1.638377x | 1.535727x | 1.613798x |

Both RVP9 screens passed the 1.05x limit. Every candidate/control IVF stream
was byte-identical, with one distinct hash per clip and rotation. The pinned
libvpx stream is a separate reference comparison, so its expected differing
hashes do not indicate a regression.

The source and harness hashes are in the three manifest files. The canonical
input hashes are also recorded there.
