#!/usr/bin/env bash
# cpu4_gap_measure.sh — clean 4-way RD measurement on the QUALITY path.
# rvp9 cpu7(RT) / rvp9 cpu4(good) vs vpxenc cpu7(RT) / vpxenc cpu4(good).
# lag-in-frames=0 EVERYWHERE so ffmpeg frame-by-frame PSNR stays aligned
# (vpxenc good-mode default lag=25 reorders alt-ref frames -> misalignment).
set -u
SRC="${SRC:-tests/corpus/foreman_320x256_60f.y4m}"
W="${W:-320}"; H="${H:-256}"; FPS="${FPS:-30000/1001}"
E=target/release/rvp9enc.exe
VPX="D:/language/libvpx-stock/vpxenc.exe"
OUT="${OUT:-build/cpu4probe}"; mkdir -p "$OUT"
BITRATES="${BITRATES:-100 250 500 1000}"

ypsnr() { ffmpeg -hide_banner -loglevel info -i "$1" -i "$SRC" \
  -lavfi "[0:v][1:v]psnr" -f null - 2>&1 | grep -o "PSNR y:[0-9.]*" | grep -o "[0-9.]*$"; }
kbps() { python -c "import os,sys;print('%.1f'%(os.path.getsize(sys.argv[1])*8*30000/1001/60/1000))" "$1"; }

printf "%-6s | %-20s | %-20s | %-20s | %-20s\n" "kbps" "rvp9_cpu7(RT)" "rvp9_cpu4(good)" "vpx_cpu7(RT)" "vpx_cpu4(good)"
for B in $BITRATES; do
  "$E" --ivf --rt   --cpu-used 7 --end-usage cbr --target-bitrate $B -o "$OUT/r7_$B.ivf" "$SRC" 2>/dev/null
  "$E" --ivf --good --cpu-used 4 --end-usage cbr --target-bitrate $B -o "$OUT/r4_$B.ivf" "$SRC" 2>/dev/null
  # NOTE: with y4m input, do NOT pass --width/--height/--fps — vpxenc then
  # reads the y4m header as raw pixels and produces garbage (flat ~27 dB).
  # Let vpxenc auto-detect geometry from the y4m header.
  "$VPX" --codec=vp9 --ivf --rt   --cpu-used=7 --tune-content=default --passes=1 --end-usage=cbr \
     --lag-in-frames=0 --target-bitrate=$B --aq-mode=0 -o "$OUT/v7_$B.ivf" "$SRC" 2>/dev/null
  "$VPX" --codec=vp9 --ivf --good --cpu-used=4 --tune-content=default --passes=1 --end-usage=cbr \
     --lag-in-frames=0 --target-bitrate=$B --aq-mode=0 -o "$OUT/v4_$B.ivf" "$SRC" 2>/dev/null
  printf "%-6s | %6s dB @%-9s | %6s dB @%-9s | %6s dB @%-9s | %6s dB @%-9s\n" "$B" \
    "$(ypsnr $OUT/r7_$B.ivf)" "$(kbps $OUT/r7_$B.ivf)" \
    "$(ypsnr $OUT/r4_$B.ivf)" "$(kbps $OUT/r4_$B.ivf)" \
    "$(ypsnr $OUT/v7_$B.ivf)" "$(kbps $OUT/v7_$B.ivf)" \
    "$(ypsnr $OUT/v4_$B.ivf)" "$(kbps $OUT/v4_$B.ivf)"
done
