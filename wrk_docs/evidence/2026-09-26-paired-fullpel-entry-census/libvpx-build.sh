#!/usr/bin/env bash
set -euo pipefail

root=/d/worktrees/rvp9-2/20260926-TSK-CDX@CRUCIBLE-libvpx-fullpel-entry-census-x86
source_dir="$root/third_party/libvpx"
build_dir="$root/.scratch/libvpx-fullpel-build-relative"
patch="$root/.scratch/libvpx-fullpel-entry-census.patch"

test "$(git -C "$source_dir" rev-parse HEAD)" = 12f3a2ac603e8f10742105519e0cd03c3b8f71dd
test "$(sha256sum "$patch" | cut -d' ' -f1)" = dd5df8c9c7de79d3f3e8257032a135272551d786c4d1de79da0aecebd74343ff
git -C "$source_dir" diff --check
test ! -e "$build_dir"
mkdir -p "$build_dir"
cd "$build_dir"

printf 'source=%s\npatch=%s\n' "$(git -C "$source_dir" rev-parse HEAD)" "$(sha256sum "$patch" | cut -d' ' -f1)"
printf 'gcc=%s\n' "$(gcc --version | head -1)"
printf 'gmake=%s\n' "$(gmake --version | head -1)"
CFLAGS='-O3 -fomit-frame-pointer -DRVP9_DIAG_LIBVPX_FULLPEL_CENSUS=1' \
  ../../third_party/libvpx/configure \
  --target=x86_64-win64-gcc \
  --enable-vp9 --disable-vp8 --enable-vp9-highbitdepth \
  --enable-vp9-temporal-denoising --enable-pic \
  --disable-docs --disable-unit-tests
gmake -j8 target=libs libvpx.a
gmake -j8 target=examples
sha256sum vpxenc.exe
