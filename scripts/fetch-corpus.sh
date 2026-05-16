#!/usr/bin/env bash
# Download the encoder test corpus into tests/corpus/.
#
# Used by the real-time latency benchmark and the BD-rate benchmark. These
# are well-known, freely-licensed test clips from Xiph.org and the AOM
# Common Test Conditions set.
#
# Stage 0 seeds 720p clips. Larger clips (1080p, 4K) added when Stage 4
# needs them.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
DEST="$REPO_ROOT/tests/corpus"
mkdir -p "$DEST"

XIPH_BASE="https://media.xiph.org/video/derf/y4m"

# Each line: <source relative path>:<local filename>
CLIPS=(
    "parkrun_ter_720p50.y4m"
    "in_to_tree_420_720p50.y4m"
    "old_town_cross_420_720p50.y4m"
    "ducks_take_off_420_720p50.y4m"
)

fetched=0
skipped=0
failed=0

for clip in "${CLIPS[@]}"; do
    target="$DEST/$clip"
    if [ -s "$target" ]; then
        skipped=$((skipped + 1))
        continue
    fi
    size_estimate="(several hundred MB; large download)"
    echo "[fetch] $clip $size_estimate"
    if curl -fL --retry 3 -o "$target" "$XIPH_BASE/$clip"; then
        echo "  done."
        fetched=$((fetched + 1))
    else
        echo "  FAILED."
        rm -f "$target"
        failed=$((failed + 1))
    fi
done

echo
echo "Corpus: $fetched fetched, $skipped already present, $failed failed."
echo "Stored in: $DEST"
if [ "$failed" -gt 0 ]; then
    exit 1
fi
