# Unit 1411 profile — integrated x86-64 encoder

This is a fresh Crucible CPU profile of integrated trunk `12ed1adf`, after the
Unit 1410 AVX2 EOB change. It used twenty serial 60-frame 1920x1080@30 CBR
encodes at CPU-used 6, one thread, tile 0, AQ0, noise sensitivity 0, lag 0,
and a 60-frame keyframe interval. The input was the pinned
`ducks_take_off_420_1080p30_60f.y4m` acceptance clip.

The report has 38,993,365 weighted `rvp9enc.exe` symbol samples. The picker
callback remains the largest exclusive symbol at 11.078698%; diamond search
accounts for 4.734162%. These are profile shares, not projected speedups.
`aggregate.txt` holds the leading symbols and `manifest.txt` identifies the
uncommitted raw ETL and symbol report by SHA-256. No new encoder change or
paired performance result follows from this measurement alone.

The next bounded experiment should first attribute instructions inside the
picker callback to source lines using this exact executable and PDB. A proposed
in-plane diamond-search shortcut was not taken forward: it substantially
overlaps the Unit 1401 direct-slice experiment, which was slower in two paired
screens. The last measured RVP9/libvpx ratio remains 1.227734x from Unit 1410.
