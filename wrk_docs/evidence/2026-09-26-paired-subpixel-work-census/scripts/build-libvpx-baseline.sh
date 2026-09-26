#!/usr/bin/env bash
set -euo pipefail

worktree=/d/worktrees/rvp9-2/20260926-TSK-HUM-crucible-profile-after-zero-eob
source_dir="$worktree/third_party/libvpx"
build_dir="$worktree/.scratch/libvpx-work-census/baseline-build-rel"
expected_head=12f3a2ac603e8f10742105519e0cd03c3b8f71dd

test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$source_dir" status --porcelain)"
test ! -e "$build_dir"
mkdir -p "$build_dir"
cd "$build_dir"

CFLAGS='-O3 -fomit-frame-pointer' ../../../third_party/libvpx/configure \
  --target=x86_64-win64-gcc \
  --enable-vp9 --disable-vp8 --enable-vp9-highbitdepth \
  --enable-vp9-temporal-denoising --enable-pic \
  --disable-docs --disable-unit-tests
gmake -j8 target=libs libvpx.a
gmake -j8 target=examples
sha256sum vpxenc.exe
