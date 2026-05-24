#!/usr/bin/env python3
"""T-5 per-tile-dispatch profile analysis.

Reads the CSV emitted by the `decode-profile-tiles` bench binary and
prints a per-phase summary table: median, p95, and percentage share of
total frame time. Confirms or rejects the hypotheses from the T-5 HLD.

Usage:
    python scripts/t5-analyze.py bench/results/t5-tile-profile.tile_1x4.t4.csv [--cpu-mhz N]
"""

import csv
import sys
import statistics
from pathlib import Path


def percentile(data: list[float], pct: float) -> float:
    """Nearest-rank percentile."""
    if not data:
        return 0.0
    k = max(0, min(len(data) - 1, int(len(data) * pct / 100.0)))
    return data[k]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    cpu_mhz = None
    if "--cpu-mhz" in sys.argv:
        idx = sys.argv.index("--cpu-mhz")
        cpu_mhz = float(sys.argv[idx + 1])

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No data rows in CSV.")
        sys.exit(1)

    # Validate schema
    schema = int(rows[0].get("schema", "0"))
    if schema != 1:
        print(f"WARNING: unexpected schema version {schema} (expected 1)")

    # Collect per-phase cycles arrays
    phases = ["setup", "submit", "join", "merge", "grid_stitch", "loopfilter", "border"]
    phase_data: dict[str, list[float]] = {p: [] for p in phases}
    wall_us_data: list[float] = []

    # Per-tile decode cycles (aggregate across all tiles in each frame)
    tile_decode_data: list[float] = []
    tile_lockin_data: list[float] = []
    tile_lockout_data: list[float] = []

    for row in rows:
        for p in phases:
            val = float(row.get(p, 0))
            phase_data[p].append(val)
        wall_us_data.append(float(row.get("wall_us", 0)))

        # Sum per-tile decode across all tiles present
        num_tiles = int(row.get("num_tiles", 0))
        tile_decode_sum = 0.0
        tile_lockin_sum = 0.0
        tile_lockout_sum = 0.0
        for t in range(num_tiles):
            tile_decode_sum += float(row.get(f"tile{t}_decode", 0))
            tile_lockin_sum += float(row.get(f"tile{t}_lockin", 0))
            tile_lockout_sum += float(row.get(f"tile{t}_lockout", 0))
        tile_decode_data.append(tile_decode_sum)
        tile_lockin_data.append(tile_lockin_sum)
        tile_lockout_data.append(tile_lockout_sum)

    n = len(rows)
    print(f"\nT-5 Phase Analysis — {n} rows from {csv_path.name}\n")

    # Sort arrays for percentile computation
    for p in phases:
        phase_data[p].sort()
    tile_decode_data.sort()
    tile_lockin_data.sort()
    tile_lockout_data.sort()
    wall_us_data.sort()

    # Compute total cycles per frame = sum of all phases
    # To get a share, we compute per-frame totals then take medians
    frame_totals: list[float] = []
    for i, row in enumerate(rows):
        total = sum(float(row.get(p, 0)) for p in phases)
        frame_totals.append(total)

    frame_totals_sorted = sorted(frame_totals)
    median_total = statistics.median(frame_totals_sorted) if frame_totals_sorted else 1.0

    # Conversion factor
    conv_label = "cycles"
    conv = 1.0
    if cpu_mhz:
        conv = 1.0 / cpu_mhz  # cycles -> us
        conv_label = "us"

    # Print summary table
    header = f"{'Phase':<16} {'Median':>12} {'P95':>12} {'Share':>8}"
    print(header)
    print("-" * len(header))

    serial_phases = ["setup", "submit", "join", "merge", "grid_stitch", "loopfilter", "border"]
    serial_median_sum = 0.0
    for p in serial_phases:
        d = phase_data[p]
        med = statistics.median(d) if d else 0.0
        p95 = percentile(d, 95)
        share = (med / median_total * 100.0) if median_total > 0 else 0.0
        serial_median_sum += med
        med_disp = med * conv
        p95_disp = p95 * conv
        print(f"{p:<16} {med_disp:>12.0f} {p95_disp:>12.0f} {share:>7.1f}%")

    # Tile worker phases
    for name, d in [("tile_decode", tile_decode_data), ("tile_lockin", tile_lockin_data), ("tile_lockout", tile_lockout_data)]:
        med = statistics.median(d) if d else 0.0
        p95 = percentile(d, 95)
        share = (med / median_total * 100.0) if median_total > 0 else 0.0
        med_disp = med * conv
        p95_disp = p95 * conv
        print(f"{name:<16} {med_disp:>12.0f} {p95_disp:>12.0f} {share:>7.1f}%")

    print()
    med_wall = statistics.median(wall_us_data) if wall_us_data else 0.0
    print(f"Median wall time: {med_wall:.1f} us ({med_wall / 1000.0:.2f} ms)")
    serial_frac = serial_median_sum / median_total if median_total > 0 else 0.0
    print(f"Serial fraction (setup+submit+join+merge+grid+LF+border): {serial_frac*100:.1f}%")
    amdahl_4 = 1.0 / (serial_frac + (1.0 - serial_frac) / 4.0) if serial_frac < 1.0 else 1.0
    print(f"Amdahl predicted speedup (4 cores): {amdahl_4:.2f}x")
    print()

    # Hypothesis verdicts
    lf_share = statistics.median(phase_data["loopfilter"]) / median_total * 100 if median_total else 0
    merge_share = statistics.median(phase_data["merge"]) / median_total * 100 if median_total else 0
    grid_share = statistics.median(phase_data["grid_stitch"]) / median_total * 100 if median_total else 0
    border_share = statistics.median(phase_data["border"]) / median_total * 100 if median_total else 0
    setup_share = statistics.median(phase_data["setup"]) / median_total * 100 if median_total else 0
    serial_total = lf_share + merge_share + grid_share + border_share + setup_share

    print("Hypothesis verdicts:")
    if serial_total >= 30:
        print(f"  H1 CONFIRMED — serial work = {serial_total:.1f}% of frame total (threshold >= 30%)")
    else:
        print(f"  H1 rejected — serial work = {serial_total:.1f}% (below 30% threshold)")

    lockin_med = statistics.median(tile_lockin_data) if tile_lockin_data else 0
    decode_med = statistics.median(tile_decode_data) if tile_decode_data else 0
    if decode_med > 0 and (lockin_med / decode_med) * 100 > 5:
        print(f"  H2 CONFIRMED — lock overhead = {lockin_med/decode_med*100:.1f}% of decode (threshold > 5%)")
    else:
        print(f"  H2 rejected — lock overhead = {lockin_med/decode_med*100:.1f}% of decode" if decode_med > 0 else "  H2 rejected — no decode data")

    if setup_share > 5:
        print(f"  H3 likely — setup = {setup_share:.1f}% (may include alloc churn)")
    else:
        print(f"  H3 rejected — setup = {setup_share:.1f}%")

    if lf_share > 40 * serial_frac:
        print(f"  H1a: loopfilter is dominant serial component ({lf_share:.1f}%)")
    elif merge_share > lf_share:
        print(f"  H1b: merge dominates over loopfilter ({merge_share:.1f}% vs {lf_share:.1f}%)")

    print()


if __name__ == "__main__":
    main()
