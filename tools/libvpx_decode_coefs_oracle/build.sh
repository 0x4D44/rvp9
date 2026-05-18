#!/usr/bin/env bash
# Build the Stage 3 stretch M.8 libvpx decode_coefs oracle.
#
# Toolchain:
#   - MinGW-w64 gcc (msys2), or any gcc with libvpx already built locally
#   - Pre-built libvpx static archive (libvpx.a) — defaults to the
#     Stage 0 build at D:/language/libvpx-stock/, which is the same
#     v1.14.1 tree that rvp9 conforms against. Override via
#     LIBVPX_DIR env var if needed.
#
# Output: ./oracle.exe (Windows) or ./oracle (Linux/macOS).
#
# Run:    ./oracle.exe  → prints libvpx output + diff vs rvp9 snapshot.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LIBVPX_DIR="${LIBVPX_DIR:-/d/language/libvpx-stock}"

if [[ ! -f "$LIBVPX_DIR/libvpx.a" ]]; then
    echo "error: libvpx.a not found at $LIBVPX_DIR/libvpx.a" >&2
    echo "        Set LIBVPX_DIR to the libvpx source+build tree." >&2
    exit 1
fi
if [[ ! -f "$LIBVPX_DIR/vpx_config.h" ]]; then
    echo "error: vpx_config.h not found at $LIBVPX_DIR/vpx_config.h" >&2
    echo "        Need a configured libvpx tree (the build dir doubles" >&2
    echo "        as the include root for libvpx's in-tree builds)." >&2
    exit 1
fi

# Detect output binary name. Windows MSYS2 produces a .exe automatically.
OUT="$HERE/oracle"
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) OUT="$HERE/oracle.exe" ;;
esac

CC="${CC:-gcc}"

# -I order matches libvpx's own build:
#   - $LIBVPX_DIR (root)   → vpx_config.h, ./vpx_dsp/..., ./vp9/...
#   - $HERE                → snapshot.h
# Treat libvpx headers as system to silence -Wunused-* in third-party.
"$CC" \
    -O2 -g -Wall -Wextra \
    -isystem "$LIBVPX_DIR" \
    -I "$HERE" \
    "$HERE/oracle.c" \
    "$LIBVPX_DIR/libvpx.a" \
    -o "$OUT"

echo "Built: $OUT"
