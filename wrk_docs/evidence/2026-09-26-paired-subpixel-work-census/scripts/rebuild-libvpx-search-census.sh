#!/usr/bin/env bash
set -euo pipefail

root=/d/worktrees/rvp9-2/20260926-TSK-HUM-crucible-profile-after-zero-eob/.scratch/libvpx-work-census
source_dir="$root/instrumented-src"
build_dir="$root/instrumented-build"
test "$(git -C "$source_dir" rev-parse HEAD)" = 12f3a2ac603e8f10742105519e0cd03c3b8f71dd
git -C "$source_dir" diff --check
grep -F -- '-DRVP9_DIAG_LIBVPX_SVF_CENSUS=1' "$build_dir/examples-x86_64-win64-gcc.mk" >/dev/null
cd "$build_dir"
gmake -j8 target=libs libvpx.a
gmake -j8 target=examples
sha256sum vpxenc.exe
