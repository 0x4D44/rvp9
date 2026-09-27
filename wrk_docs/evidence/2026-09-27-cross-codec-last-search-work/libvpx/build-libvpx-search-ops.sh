#!/usr/bin/env bash
set -euo pipefail

root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/libvpx-search-ops-census-f50573b0
source_dir="$root/source"
build_dir="$root/build"

test "$(git -C "$source_dir" rev-parse HEAD)" = 12f3a2ac603e8f10742105519e0cd03c3b8f71dd
git -C "$source_dir" diff --check
test ! -e "$build_dir"
mkdir -p "$build_dir"
cd "$build_dir"

printf 'source=%s\n' "$(git -C "$source_dir" rev-parse HEAD)"
printf 'gcc=%s\n' "$(gcc --version | head -1)"
printf 'gmake=%s\n' "$(gmake --version | head -1)"
CFLAGS='-O3 -fomit-frame-pointer -DRVP9_DIAG_LIBVPX_FULLPEL_CENSUS=1' \
  ../source/configure \
  --target=x86_64-win64-gcc \
  --enable-vp9 --disable-vp8 --enable-vp9-highbitdepth \
  --enable-vp9-temporal-denoising --enable-pic \
  --disable-docs --disable-unit-tests
gmake -j8 target=libs libvpx.a
gmake -j8 target=examples
sha256sum vpxenc.exe
