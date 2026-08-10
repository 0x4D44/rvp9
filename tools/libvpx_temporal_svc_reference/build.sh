#!/usr/bin/env bash
# Build the narrow temporal-SVC reference adapter against the repository's
# exact pinned libvpx archive. The pinned checkout remains unmodified.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
source_dir="$repo_root/third_party/libvpx"
build_dir="${RVP9_LIBVPX_BUILD_DIR:-$repo_root/build/libvpx}"
source_file="$repo_root/tools/libvpx_temporal_svc_reference/main.c"
expected_commit="12f3a2ac603e8f10742105519e0cd03c3b8f71dd"
actual_commit="$(git -C "$source_dir" rev-parse HEAD)"

if [ "$actual_commit" != "$expected_commit" ]; then
    echo "libvpx pin mismatch: expected $expected_commit, got $actual_commit" >&2
    exit 1
fi
if ! git -C "$source_dir" diff --quiet || ! git -C "$source_dir" diff --cached --quiet; then
    echo "third_party/libvpx is dirty; refusing an ambiguous reference build" >&2
    exit 1
fi
if [ ! -f "$build_dir/libvpx.a" ]; then
    echo "missing $build_dir/libvpx.a; run ./scripts/build-libvpx.sh first" >&2
    exit 1
fi

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) executable="$build_dir/vpx_temporal_svc_reference.exe" ;;
    *) executable="$build_dir/vpx_temporal_svc_reference" ;;
esac

"${CC:-cc}" \
    -std=c11 -O3 -fomit-frame-pointer -Wall -Wextra -Werror \
    -I"$build_dir" -I"$source_dir" \
    "$source_file" "$build_dir/libvpx.a" \
    -lm -lpthread \
    -o "$executable"

"$executable" --version
echo "Temporal-SVC reference: $executable"
