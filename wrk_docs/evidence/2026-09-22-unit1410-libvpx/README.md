# Unit 1410 — fresh pinned-libvpx comparison

Crucible compared candidate `2ebca99e` against the pinned libvpx `vpxenc`
binary (SHA-256 `88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d`).
The three 1920×1080 60-frame clips used CPU-used 6, CBR 1260 kb/s,
30 fps, one thread, zero tiles, AQ0, noise sensitivity 0, one warmup,
three pairs, two rotations, and no `--max-load1` ceiling.

| Clip | Paired RVP9/libvpx time |
| --- | ---: |
| Ducks | 1.402598× |
| In To Tree | 1.210165× |
| Old Town | 1.090273× |
| Corpus | **1.227734×** |

Both arms had one stable IVF hash per clip. Their bitstreams differ as
expected; RVP9 candidate/control hashes match in the sibling A/B/C screens.
The `1.05×` parity check failed (harness exit 1). At this measured corpus
ratio, reaching 1.05× still requires roughly 14.5% less RVP9 runtime.
This is a fresh screen, not an inferred adjustment to an older libvpx result.

The exact binary/input hashes, commands, load samples, and raw timings are
in `manifest.txt`, `summary.csv`, and `raw.csv` beside this file.
