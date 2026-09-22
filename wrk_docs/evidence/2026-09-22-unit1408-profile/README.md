# Unit 1408 profile — current x86-64 encoder hotspots

This is a fresh Crucible profile of integrated trunk `9e882a46`. It ran twenty
serial 60-frame 1920x1080@30 CBR encodes at CPU-used 6, one thread, tile 0,
AQ0, noise sensitivity 0, lag 0, and a 60-frame keyframe interval. The input
was `ducks_take_off_420_1080p30_60f.y4m` from the pinned acceptance corpus.

The WPR report contained 39,747,732 weighted `rvp9enc.exe` module samples.
The UTF-16 report hash is recorded in `manifest.txt`; `aggregate.txt` is the
deterministic symbol aggregation used to select the next bounded candidate.

The fixed 16x16 variance kernel accounted for 4.213347% of samples. It was a
reasonable bounded experiment, but the legal plain `#[inline]` form did not
produce a reproducible end-to-end gain and was rejected by the timing screen.
