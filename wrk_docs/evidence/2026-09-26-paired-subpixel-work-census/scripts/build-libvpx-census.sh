#!/usr/bin/env bash
set -euo pipefail

root=/d/worktrees/rvp9-2/20260926-TSK-HUM-crucible-profile-after-zero-eob/.scratch/libvpx-work-census
source_dir="$root/instrumented-src"
build_dir="$root/instrumented-build"
test "$(git -C "$source_dir" rev-parse HEAD)" = 12f3a2ac603e8f10742105519e0cd03c3b8f71dd
git -C "$source_dir" diff --check
test ! -e "$build_dir"
mkdir -p "$build_dir"
cd "$build_dir"

CFLAGS='-O3 -fomit-frame-pointer -DRVP9_DIAG_LIBVPX_SVF_CENSUS=1' \
  ../instrumented-src/configure \
  --target=x86_64-win64-gcc \
  --enable-vp9 --disable-vp8 --enable-vp9-highbitdepth \
  --enable-vp9-temporal-denoising --enable-pic \
  --disable-docs --disable-unit-tests
gmake -j8 target=libs libvpx.a
gmake -j8 target=examples
sha256sum vpxenc.exe
