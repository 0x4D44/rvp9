#!/usr/bin/env python3
"""Spatial distribution analysis of `dump_frame_diff`'s per-4×4-block
mismatch grid. Used during D.23 to localize tile_1x4 frame 3 divergence
to SB row 16 (partial-bottom). Reusable for future multi-tile bisects.

Usage:
    python3 diff_y_dump.py <vector> <expected.yuv> <frame_idx>

Example:
    python3 diff_y_dump.py \
        tests/conformance/vectors/vp90-2-08-tile_1x4.webm \
        tile_1x4_f3.yuv 3

Output: per-SB(64×64) mismatch totals, sorted, plus marginal totals by
sb_row and sb_col. Filters totals > 50 to suppress noise.
"""
import os
import re
import subprocess
import sys
from collections import Counter

GRID_PATTERN = re.compile(r"^r\s*(\d+)\(\s*(\d+)\):(.*)$")


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    vector, expected, frame_idx = sys.argv[1], sys.argv[2], sys.argv[3]

    out = subprocess.run(
        [
            os.path.join(".", "target", "release", "dump_frame_diff.exe"),
            "--vector", vector,
            "--expected", expected,
            "--frame", str(frame_idx),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    nonzero = []  # list of (px_x, px_y, count)
    for line in out.stdout.splitlines():
        m = GRID_PATTERN.match(line)
        if not m:
            continue
        block_row = int(m.group(1))
        rest = m.group(3)
        # Cells are " %3d " or "  .  " — fixed-width 5 chars.
        cells = []
        i = 0
        while i < len(rest):
            chunk = rest[i:i + 5]
            if not chunk.strip():
                i += 5
                continue
            cells.append(chunk.strip())
            i += 5
        for bc, val in enumerate(cells):
            if val == ".":
                continue
            try:
                cnt = int(val)
            except ValueError:
                continue
            if cnt > 0:
                nonzero.append((bc * 4, block_row * 4, cnt))

    print(f"Total nonzero 4x4 blocks: {len(nonzero)}")

    sb_band: Counter = Counter()
    for px_x, px_y, cnt in nonzero:
        sb_band[(px_y // 64, px_x // 64)] += cnt
    print(f"Distinct (sb_row, sb_col) bands with diffs: {len(sb_band)}")
    print("Top 20 SB bands by total diff:")
    for (sr, sc), cnt in sorted(sb_band.items(), key=lambda x: -x[1])[:20]:
        print(
            f"  SB(row={sr}, col={sc}) @ px ({sc*64}..{sc*64+64}, "
            f"{sr*64}..{sr*64+64}): {cnt} mismatches"
        )

    print("\nSB band totals by sb_col (where total > 50):")
    col_totals: Counter = Counter()
    for (_, sc), cnt in sb_band.items():
        col_totals[sc] += cnt
    for sc, cnt in sorted(col_totals.items()):
        if cnt > 50:
            print(f"  sb_col={sc} (px {sc*64}..{sc*64+64}): {cnt} mismatches")

    print("\nSB band totals by sb_row (where total > 50):")
    row_totals: Counter = Counter()
    for (sr, _), cnt in sb_band.items():
        row_totals[sr] += cnt
    for sr, cnt in sorted(row_totals.items()):
        if cnt > 50:
            print(f"  sb_row={sr} (px {sr*64}..{sr*64+64}): {cnt} mismatches")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
