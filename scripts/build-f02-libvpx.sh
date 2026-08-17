#!/usr/bin/env bash
# Build pinned libvpx v1.14.1 vpxenc with the output-only F.0.2 oracle.
#
# The pinned checkout stays clean. A temporary shared clone receives the
# instrumentation patch; build output lands in build/libvpx-f02/.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
source_dir="$repo_root/third_party/libvpx"
patch_file="$repo_root/third_party/libvpx-patches/f02_oracle.patch"
coeff_patch_file="$repo_root/third_party/libvpx-patches/f02_coeff_vectors.patch"
build_dir="$repo_root/build/libvpx-f02"
expected_commit="12f3a2ac603e8f10742105519e0cd03c3b8f71dd"
actual_commit="$(git -C "$source_dir" rev-parse HEAD)"

if [ "$actual_commit" != "$expected_commit" ]; then
    echo "libvpx pin mismatch: expected $expected_commit, got $actual_commit" >&2
    exit 1
fi
if ! git -C "$source_dir" diff --quiet || ! git -C "$source_dir" diff --cached --quiet; then
    echo "third_party/libvpx is dirty; refusing an ambiguous oracle build" >&2
    exit 1
fi
git -C "$source_dir" apply --unidiff-zero --check "$patch_file"

scratch_root="$(mktemp -d "${TMPDIR:-/tmp}/rvp9-f02-libvpx.XXXXXX")"
scratch_source="$scratch_root/libvpx"
git clone --quiet --shared "$source_dir" "$scratch_source"
git -C "$scratch_source" checkout --quiet --detach "$expected_commit"
git -C "$scratch_source" apply --unidiff-zero "$patch_file"
git -C "$scratch_source" apply --check "$coeff_patch_file"
git -C "$scratch_source" apply "$coeff_patch_file"

case "$(uname -s):$(uname -m)" in
    Darwin:arm64) target="arm64-darwin23-gcc" ;;
    Darwin:x86_64) target="x86_64-darwin-gcc" ;;
    Linux:aarch64|Linux:arm64) target="arm64-linux-gcc" ;;
    Linux:x86_64) target="x86_64-linux-gcc" ;;
    MINGW*|MSYS*|CYGWIN*) target="x86_64-win64-gcc" ;;
    *) target="generic-gnu" ;;
esac

mkdir -p "$build_dir"
cd "$build_dir"
CFLAGS="-O3 -fomit-frame-pointer" \
    "$scratch_source/configure" \
    --target="$target" \
    --enable-vp9 \
    --disable-vp8 \
    --enable-vp9-highbitdepth \
    --enable-pic \
    --disable-docs \
    --disable-unit-tests

jobs="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
make -j"$jobs" target=libs libvpx.a
make -j"$jobs" target=examples vpxenc examples/vp9_spatial_svc_encoder

echo "F.0.2 oracle vpxenc: $build_dir/vpxenc"
echo "F.0.2 spatial oracle: $build_dir/examples/vp9_spatial_svc_encoder"
echo "Patched temporary source retained for audit: $scratch_source"
