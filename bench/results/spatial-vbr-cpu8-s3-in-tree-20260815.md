# Spatial VBR CPU8/S3 probe — 2026-08-15

This is the first measured VBR spatial-SVC curve after the Stage 4e evidence
audit. It is a diagnostic slice, not the complete acceptance matrix.

- RVP9 harness commit: `2ee29ebfa27e2dd93f3f846dcee599d71e7e98a8`
- libvpx revision: `12f3a2ac603e8f10742105519e0cd03c3b8f71dd`
- Input: In To Tree, 1280x720, 30 fps, 60 frames, SHA-256
  `3053ddad95a66bb970685412d58ee807813319668a57cd5f8a6d00c07415a895`
- Configuration: one-pass realtime VBR, CPU8, three spatial layers,
  inter-layer prediction 0, AQ0, denoiser 0, one thread, no tiles, keyframe
  interval 60, targets `420,630,840,1260,2100,3360` kbps.
- Production-PGO RVP9 SHA-256:
  `60b546a13bd6218aa6bbca00645da8e49332647dab13471632267cc16407a0e1`
- Spatial libvpx anchor SHA-256:
  `dc73d7779a6e62ffdfa4857c459bcee15e14df799f2564e94c8dc2f868e04caf`
- `bdrate-svc` SHA-256:
  `9a67d1582ce6f99b44469dedd1167c93787fcf82c53580a15d40e825ea883c22`

## Result

| Layer | PSNR BD-rate | Gate | Matched-bitrate anchor − test PSNR |
| --- | ---: | --- | ---: |
| SL0 | `-6.915707%` | pass | mean `-0.2527 dB` |
| SL1 | `-5.072550%` | pass | mean `-0.1057 dB` |
| SL2 | `+1.256547%` | **fail** | mean `+0.0532 dB` |

The miss is confined to the cumulative top operating point. The matched-rate
top-layer PSNR difference is small, while RVP9 emits substantially fewer bits
than the anchor at most requested targets. That points first to spatial VBR
rate allocation or top-layer rate-control calibration, not a broad prediction
quality failure.

All 24 RVP9 full/extracted streams decoded successfully through rvp9dec,
pinned vpxdec, and FFmpeg. Rvp9dec and vpxdec produced identical I420 SHA-256
for every stream: 72 decoder runs, zero failures. FFmpeg is treated only as a
decode-success oracle.

The host load exceeded 20 during this probe. The wall-time columns in the CSV
are therefore not speed evidence and must not be used for acceptance.

## Evidence

- RD points: `spatial-vbr-cpu8-s3-in-tree-20260815.csv`, SHA-256
  `a963ebe77c11136fea69ac4c5f5b633bd886fe0e7054725218c9a680f6c939c9`
- Cross-decoder rows:
  `spatial-vbr-cpu8-s3-in-tree-20260815-cross-decoder.csv`, SHA-256
  `1c41f950ae8c910b651c488793f70633f1e8702774d7e11531efe031e0512f1b`

Command shape:

```bash
target/release/bdrate-svc \
  --anchor-encoder build/libvpx/examples/vp9_spatial_svc_encoder \
  --test-encoder target/encoder-pgo/optimized/release/rvp9enc \
  --decoder build/libvpx/vpxdec \
  --input in-tree-720p30-60f.y4m \
  --cpu-used 8 --bitrates 420,630,840,1260,2100,3360 \
  --end-usage vbr --svc-kind spatial --spatial-layers 3 \
  --inter-layer-pred 0 --temporal-layers 1 \
  --threads 1 --tile-columns 0 --aq-mode 0 --noise-sensitivity 0 \
  --limit 60 --keyframe-interval 60 --max-bdrate-pct 0.5 \
  --work-dir "$work_dir" --csv "$csv_path"
```
