# Pinned libvpx temporal-SVC reference

`vpxenc` does not expose VP9 temporal-SVC configuration on its command line.
This adapter links the repository's pinned libvpx v1.14.1 archive and exposes
only the reference surface required by rvp9's Stage 4d acceptance gate:

- one-pass realtime VP9;
- one spatial layer;
- built-in `0101` and `0212` temporal modes;
- matched CPU6–8, CBR/VBR, AQ, denoiser, thread, and tile controls;
- no lookahead or frame dropping.

Build the pinned reference and adapter:

```sh
./scripts/build-libvpx.sh
./tools/libvpx_temporal_svc_reference/build.sh
```

Example:

```sh
build/libvpx/vpx_temporal_svc_reference \
  --input tests/corpus/clip.y4m \
  --output /tmp/libvpx-t3.ivf \
  --target-bitrate 1500 \
  --cpu-used 7 \
  --end-usage vbr \
  --threads 1 \
  --tile-columns 0 \
  --temporal-layers 3 \
  --aq-mode 0 \
  --trace /tmp/libvpx-t3-trace.csv \
  --limit 60
```

The binary validates libvpx's reported temporal ID against the fixed pattern
for every encoded frame. `--trace` emits the corresponding internal Q-index
and packet bits without changing encoder state. Per-operating-point extraction
and quality analysis live in `rvp9-bench` so both encoders pass through the
same container and PSNR code.
