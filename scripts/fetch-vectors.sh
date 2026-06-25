#!/usr/bin/env bash
# Download official VP9 conformance test vectors into tests/conformance/vectors/.
#
# Source: WebM Project's public test data bucket.
# Each vector ships with a .md5 file containing the expected decoded YUV MD5.
#
# Stage 0 seeds a small starter set (~50 MB) enough to validate the harness.
# Stage 3 will extend this list to the full upstream conformance set.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
DEST="$REPO_ROOT/tests/conformance/vectors"
mkdir -p "$DEST"

BASE_URL="https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx"

# libvpx uses .webm containers for VP9 test vectors (not raw .ivf). Each
# vector has a paired .md5 file containing the MD5 of the decoded YUV.
# Full canonical list lives at third_party/libvpx/test/test-data.sha1 (~430
# VP9 vectors). Stage 0 seeds a small representative set.
VECTORS=(
    "vp90-2-00-quantizer-00.webm"
    "vp90-2-00-quantizer-26.webm"
    "vp90-2-00-quantizer-63.webm"
    "vp90-2-01-sharpness-1.webm"
    "vp90-2-01-sharpness-7.webm"
    "vp90-2-02-size-08x08.webm"
    "vp90-2-02-size-32x32.webm"
    "vp90-2-03-size-202x202.webm"
    "vp90-2-03-deltaq.webm"
    "vp90-2-05-resize.ivf"
    "vp90-2-08-tile_1x2.webm"
    "vp90-2-08-tile_1x4.webm"
    "vp90-2-08-tile_1x8.webm"
    "vp90-2-09-aq2.webm"
    "vp90-2-10-show-existing-frame.webm"
)

fetched=0
skipped=0
failed=0

for v in "${VECTORS[@]}"; do
    for suffix in "" ".md5"; do
        target="$DEST/$v$suffix"
        if [ -s "$target" ]; then
            skipped=$((skipped + 1))
            continue
        fi
        printf "  fetch %s%s ... " "$v" "$suffix"
        if curl -fsSL --retry 3 -o "$target" "$BASE_URL/$v$suffix"; then
            echo "ok"
            fetched=$((fetched + 1))
        else
            echo "FAILED"
            rm -f "$target"
            failed=$((failed + 1))
        fi
    done
done

echo
echo "Vectors: $fetched fetched, $skipped already present, $failed failed."
echo "Stored in: $DEST"
if [ "$failed" -gt 0 ]; then
    exit 1
fi
