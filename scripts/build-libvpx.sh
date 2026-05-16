#!/usr/bin/env bash
# Build libvpx from the submodule at third_party/libvpx.
#
# Targets: vpxenc, vpxdec, libvpx.a. Output: third_party/libvpx/build/.
#
# Windows: invoke from Git Bash or MSYS2 mingw64 shell. Requires gcc, make,
# yasm, perl on PATH. (Git Bash + MSYS2 mingw64 supplies all of these via
# /c/msys64/mingw64/bin and /c/msys64/usr/bin.)
#
# Linux / macOS: requires the same toolchain. macOS users should brew-install
# yasm; ./configure auto-detects clang.

set -euo pipefail

# On Windows / Git Bash, MSYS2's `make` lives in /c/msys64/usr/bin which is
# not on the default Git Bash PATH. Prepend MSYS2 paths if they exist.
case "$(uname -s)" in
    MINGW*|MSYS*)
        for p in /c/msys64/usr/bin /c/msys64/mingw64/bin; do
            if [ -d "$p" ] && [[ ":$PATH:" != *":$p:"* ]]; then
                PATH="$p:$PATH"
            fi
        done
        export PATH
        # libvpx's configure invokes gcc with hardcoded /tmp/... paths.
        # On Windows the inherited TMP/TEMP may point at C:\windows\ which
        # is read-only for normal users, breaking every compile probe.
        # Force a writable temp.
        TMPDIR="${TMPDIR:-/tmp}"
        TMP="$TMPDIR"
        TEMP="$TMPDIR"
        mkdir -p "$TMPDIR"
        export TMPDIR TMP TEMP
        ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
LIBVPX_SRC="$REPO_ROOT/third_party/libvpx"
BUILD_DIR="$REPO_ROOT/build/libvpx"

# Known-issue on Windows / Git Bash: libvpx's configure + Makefile assume
# a Linux-like environment and intermittently mis-handle paths and TMPDIR
# inheritance. If a prebuilt v1.14.1 vpxenc/vpxdec is already present at
# RVP9_LIBVPX_PREBUILT_DIR (env var) or the conventional D:/language/
# libvpx-stock location, copy them in instead of attempting to build.
#
# Followup work tracked in PLAN.md Stage 0 follow-ups.
case "$(uname -s)" in
    MINGW*|MSYS*)
        for cand in "${RVP9_LIBVPX_PREBUILT_DIR:-}" \
                    "/d/language/libvpx-stock" \
                    "/d/language/libvpx-build-release"; do
            if [ -n "$cand" ] && [ -x "$cand/vpxenc.exe" ] && [ -x "$cand/vpxdec.exe" ]; then
                echo "Using prebuilt libvpx binaries from: $cand"
                mkdir -p "$BUILD_DIR"
                cp "$cand/vpxenc.exe" "$cand/vpxdec.exe" "$BUILD_DIR/"
                ls -lh "$BUILD_DIR"
                exit 0
            fi
        done
        echo "No prebuilt libvpx found; attempting build from submodule." >&2
        echo "If this fails, build vpxenc/vpxdec manually under MSYS2"      >&2
        echo "mingw64 shell and copy them to $BUILD_DIR."                   >&2
        ;;
esac

if [ ! -e "$LIBVPX_SRC/.git" ]; then
    echo "libvpx submodule not initialised. Run:" >&2
    echo "  git submodule update --init --recursive" >&2
    exit 1
fi

# Pin to v1.14.1 to match PLAN.md (and Arthur's existing local clone).
PINNED_TAG="v1.14.1"
( cd "$LIBVPX_SRC" && \
    if [ "$(git describe --tags --exact-match 2>/dev/null || true)" != "$PINNED_TAG" ]; then
        echo "Checking out $PINNED_TAG ..."
        git fetch --tags origin
        git checkout "$PINNED_TAG"
    fi
)

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Detect target. libvpx's configure understands these triples.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) TARGET="x86_64-win64-gcc" ;;
    Linux)                TARGET="x86_64-linux-gcc" ;;
    Darwin)               TARGET="x86_64-darwin-gcc" ;;
    *)                    TARGET="generic-gnu" ;;
esac

# Real-time encoder + decoder, both 8-bit and 10-bit profiles. High-bit-depth
# (Q1 in PLAN.md §4) is left enabled here because it's harmless to build it
# in the reference, even if we don't ship it in v1.
echo "Configuring libvpx for $TARGET ..."
CFLAGS="-O3 -fomit-frame-pointer" \
"$LIBVPX_SRC/configure" \
    --target="$TARGET" \
    --enable-vp9 \
    --disable-vp8 \
    --enable-vp9-highbitdepth \
    --enable-pic \
    --disable-docs \
    --disable-tools \
    --disable-unit-tests
# Note: NOT passing --disable-examples; vpxenc + vpxdec live under
# examples/ in libvpx and are exactly what we need.

JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
echo "Building with -j$JOBS ..."
make -j"$JOBS" vpxenc vpxdec

echo
echo "Build complete. Artifacts:"
for f in vpxenc vpxdec vpxenc.exe vpxdec.exe; do
    [ -f "$BUILD_DIR/$f" ] && ls -lh "$BUILD_DIR/$f"
done
