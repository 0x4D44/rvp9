"""Validate and summarize the paired RVP9/libvpx subpixel work census.

The expected input and IVF output hashes match the clips in
wrk_docs/evidence/2026-09-26-combined-integrated-parity/combined-integrated-libvpx.ps1.
Counts describe observed work, not equivalent search policy or elapsed time.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


CLIPS = ("ducks", "tree", "town")
ENCODERS = ("rvp9", "libvpx")
INPUT_HASHES = {
    "ducks": "6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7",
    "tree": "ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791",
    "town": "b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909",
}
OUTPUT_HASHES = {
    "rvp9": {
        "ducks": "8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089",
        "tree": "5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41",
        "town": "4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f",
    },
    "libvpx": {
        "ducks": "b24234d8db46c22e03585f1973b27c2e62315f0464da392a2ecd2769d2124fa8",
        "tree": "b930800c5bac04a67aa0a25b3344031c5aebe6f3efe4ba569d354301bf238964",
        "town": "e05daf970ff4f9888a64b6edc2f4a2cd7151cf0ee2bc0b1de4170cce5f1b24bb",
    },
}
SHAPES = {
    (4, 4), (4, 8), (8, 4), (8, 8), (8, 16), (16, 8), (16, 16),
    (16, 32), (32, 16), (32, 32), (32, 64), (64, 32), (64, 64),
}
MANIFEST_HEADER = ["Clip", "InputSha256", "OutputSha256", "BinarySha256", "CensusRows"]
ENTRY_MANIFEST_HEADER = MANIFEST_HEADER + ["Searches"]
CAPTURE_HEADERS = {
    "rvp9": ["subpel_census", "family", "width", "height", "xoff", "yoff", "calls", "pixels"],
    "libvpx": ["LIBVPX_SVF_CENSUS", "shape", "x_phase", "y_phase", "calls", "pixels"],
}
SUMMARY_HEADER = ["clip", "encoder", "caller_family", "shape", "phase_class", "calls",
                  "pixels", "comparison_scope", "searches", "fractional_probes_per_search",
                  "fractional_pixels_per_search"]
FRACTIONAL_SCOPE = "descriptive_not_like_for_like_time_or_search_policy"
COMPARISON_NOTE = (
    "Fractional svf call and pixel totals are descriptive only, not like-for-like "
    "time or search-policy equivalence. RVP9 (0,0) center work is shown "
    "separately and excluded only from fractional totals; libvpx svf has no center row. "
    "Per-search figures use explicit full-tree search entries. RVP9 has per-family/shape "
    "denominators; libvpx has only a clip-wide denominator."
)
HASH_CHECK_NOTE = (
    "Manifest-recorded input and IVF output SHA-256 values match pinned script constants. "
    "The source media, IVF files, and binaries are not included here and were not rehashed."
)
HASH_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
SHAPE_RE = re.compile(r"([0-9]+)x([0-9]+)\Z")
FAMILY_RE = re.compile(r"[a-z][a-z0-9_]*\Z")


def decimal(value: str, label: str, location: str) -> int:
    if not value.isdecimal():
        raise ValueError(f"{location}: {label} must be a non-negative decimal integer")
    return int(value)


def check_hash(value: str, label: str, location: str) -> str:
    if not HASH_RE.fullmatch(value):
        raise ValueError(f"{location}: {label} must be a 64-digit SHA-256 hex string")
    return value.lower()


def read_manifest(path: Path, encoder: str, with_searches: bool = False) -> dict:
    entries = {}
    binary_hash = None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_header = ENTRY_MANIFEST_HEADER if with_searches else MANIFEST_HEADER
        if reader.fieldnames != expected_header:
            raise ValueError(f"{path}: manifest schema must be {expected_header}")
        for line, entry in enumerate(reader, 2):
            location = f"{path}:{line}"
            if None in entry or any(value is None for value in entry.values()):
                raise ValueError(f"{location}: manifest row schema mismatch")
            clip = entry["Clip"]
            if clip not in CLIPS or clip in entries:
                raise ValueError(f"{location}: unknown or duplicate clip {clip!r}")
            input_hash = check_hash(entry["InputSha256"], "input hash", location)
            output_hash = check_hash(entry["OutputSha256"], "output hash", location)
            current_binary_hash = check_hash(entry["BinarySha256"], "binary hash", location)
            if input_hash != INPUT_HASHES[clip]:
                raise ValueError(f"{location}: input hash differs from canonical {clip} input")
            if output_hash != OUTPUT_HASHES[encoder][clip]:
                raise ValueError(f"{location}: output hash differs from canonical {encoder} IVF")
            if binary_hash is not None and current_binary_hash != binary_hash:
                raise ValueError(f"{location}: binary hash differs across {encoder} clips")
            binary_hash = current_binary_hash
            census_rows = decimal(entry["CensusRows"], "CensusRows", location)
            if census_rows == 0:
                raise ValueError(f"{location}: CensusRows must be positive")
            entries[clip] = {
                "input_sha256": input_hash,
                "output_sha256": output_hash,
                "expected_output_sha256": OUTPUT_HASHES[encoder][clip],
                "binary_sha256": current_binary_hash,
                "census_rows": census_rows,
            }
            if with_searches:
                searches = decimal(entry["Searches"], "Searches", location)
                if searches == 0:
                    raise ValueError(f"{location}: Searches must be positive")
                entries[clip]["searches"] = searches
    if set(entries) != set(CLIPS):
        raise ValueError(f"{path}: manifest must contain exactly {', '.join(CLIPS)}")
    return entries


def read_capture(path: Path, encoder: str) -> list:
    header = CAPTURE_HEADERS[encoder]
    prefix = header[0]
    rows = []
    seen = set()
    header_seen = False
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or not line.startswith(prefix):
                continue
            location = f"{path}:{line_number}"
            fields = next(csv.reader([line], strict=True))
            if fields[0] != prefix:
                raise ValueError(f"{location}: invalid row {fields[0]!r}")
            if fields == header:
                if header_seen:
                    raise ValueError(f"{location}: duplicate header schema")
                header_seen = True
                continue
            if len(fields) > 1 and fields[1] in ("family", "shape"):
                raise ValueError(f"{location}: capture header schema mismatch")
            if not header_seen:
                raise ValueError(f"{location}: census data precedes header schema")
            if len(fields) != len(header):
                raise ValueError(f"{location}: capture row schema mismatch")
            if encoder == "rvp9":
                family = fields[1]
                if not FAMILY_RE.fullmatch(family):
                    raise ValueError(f"{location}: invalid caller family {family!r}")
                width = decimal(fields[2], "width", location)
                height = decimal(fields[3], "height", location)
                x_phase = decimal(fields[4], "x phase", location)
                y_phase = decimal(fields[5], "y phase", location)
                calls = decimal(fields[6], "calls", location)
                pixels = decimal(fields[7], "pixels", location)
                key = (family, width, height, x_phase, y_phase)
            else:
                family = "svf_dispatch_unattributed"
                match = SHAPE_RE.fullmatch(fields[1])
                if match is None:
                    raise ValueError(f"{location}: invalid block shape {fields[1]!r}")
                width, height = (int(value) for value in match.groups())
                x_phase = decimal(fields[2], "x phase", location)
                y_phase = decimal(fields[3], "y phase", location)
                calls = decimal(fields[4], "calls", location)
                pixels = decimal(fields[5], "pixels", location)
                key = (width, height, x_phase, y_phase)
            if (width, height) not in SHAPES:
                raise ValueError(f"{location}: unsupported VP9 block shape {width}x{height}")
            if not (0 <= x_phase <= 7 and 0 <= y_phase <= 7):
                raise ValueError(f"{location}: phase must be in 0..7")
            if encoder == "libvpx" and x_phase == y_phase == 0:
                raise ValueError(f"{location}: libvpx svf unexpectedly has a center phase")
            if calls == 0:
                raise ValueError(f"{location}: calls must be positive")
            if pixels != calls * width * height:
                raise ValueError(f"{location}: pixels must equal calls * width * height")
            if key in seen:
                raise ValueError(f"{location}: duplicate key {key}")
            seen.add(key)
            phase_class = ("center" if x_phase == y_phase == 0 else
                           "one_axis" if x_phase == 0 or y_phase == 0 else "two_axis")
            rows.append((family, width, height, x_phase, y_phase, phase_class, calls, pixels))
    if not header_seen or not rows:
        raise ValueError(f"{path}: missing capture header schema or census rows")
    return rows


def read_libvpx_searches(path: Path) -> int:
    searches = None
    prefix = "LIBVPX_SUBPEL_SEARCHES"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line.startswith(prefix):
                continue
            location = f"{path}:{line_number}"
            fields = next(csv.reader([line], strict=True))
            if len(fields) != 3 or fields[:2] != [prefix, "total"] or searches is not None:
                raise ValueError(f"{location}: invalid or duplicate Searches row schema")
            searches = decimal(fields[2], "Searches", location)
            if searches == 0:
                raise ValueError(f"{location}: Searches must be positive")
    if searches is None:
        raise ValueError(f"{path}: missing Searches row")
    return searches


def read_rvp9_searches(path: Path) -> dict:
    header = ["subpel_searches", "family", "width", "height", "searches"]
    counts = {}
    header_seen = False
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line.startswith(header[0]):
                continue
            location = f"{path}:{line_number}"
            fields = next(csv.reader([line], strict=True))
            if fields[0] != header[0]:
                raise ValueError(f"{location}: invalid search row {fields[0]!r}")
            if fields == header:
                if header_seen:
                    raise ValueError(f"{location}: duplicate search header schema")
                header_seen = True
                continue
            if len(fields) > 1 and fields[1] == "family":
                raise ValueError(f"{location}: search header schema mismatch")
            if not header_seen or len(fields) != len(header):
                raise ValueError(f"{location}: search row schema mismatch")
            family = fields[1]
            if not FAMILY_RE.fullmatch(family):
                raise ValueError(f"{location}: invalid caller family {family!r}")
            width = decimal(fields[2], "width", location)
            height = decimal(fields[3], "height", location)
            if (width, height) not in SHAPES:
                raise ValueError(f"{location}: unsupported VP9 block shape {width}x{height}")
            searches = decimal(fields[4], "searches", location)
            if searches == 0:
                raise ValueError(f"{location}: searches must be positive")
            key = (family, width, height)
            if key in counts:
                raise ValueError(f"{location}: duplicate search key {key}")
            counts[key] = searches
    if not header_seen or not counts:
        raise ValueError(f"{path}: missing search header schema or rows")
    return counts


def aggregate(captures: dict, searches: dict, shape_searches: dict) -> list:
    grouped = defaultdict(lambda: [0, 0])
    totals = defaultdict(lambda: [0, 0])
    fractional = defaultdict(lambda: [0, 0])
    for (clip, encoder), rows in captures.items():
        for family, width, height, _, _, phase_class, calls, pixels in rows:
            group = grouped[(clip, encoder, family, width, height, phase_class)]
            group[0] += calls
            group[1] += pixels
            totals[(clip, encoder)][0] += calls
            totals[(clip, encoder)][1] += pixels
            if phase_class != "center":
                fractional[(clip, encoder)][0] += calls
                fractional[(clip, encoder)][1] += pixels

    result = []
    class_order = {"center": 0, "one_axis": 1, "two_axis": 2}
    for clip in CLIPS:
        for encoder in ENCODERS:
            by_shape = defaultdict(list)
            for key in grouped:
                if key[:2] == (clip, encoder):
                    by_shape[key[2:5]].append(key)
            shape_keys = sorted(by_shape, key=lambda key: (key[1] * key[2], key[1], key[2], key[0]))
            for family, width, height in shape_keys:
                shape_count = shape_searches.get((clip, encoder, family, width, height))
                shape_calls = shape_pixels = 0
                for key in sorted(by_shape[(family, width, height)],
                                  key=lambda row: class_order[row[5]]):
                    phase_class = key[5]
                    calls, pixels = grouped[key]
                    result.append([clip, encoder, family, f"{width}x{height}", phase_class,
                                   calls, pixels, "within_rvp9_only" if phase_class == "center"
                                   else FRACTIONAL_SCOPE,
                                   shape_count if shape_count is not None else "",
                                   f"{calls / shape_count:.6f}" if shape_count and phase_class != "center"
                                   else "",
                                   f"{pixels / shape_count:.6f}" if shape_count and phase_class != "center"
                                   else ""])
                    if phase_class != "center":
                        shape_calls += calls
                        shape_pixels += pixels
                result.append([clip, encoder, family, f"{width}x{height}", "fractional_total",
                               shape_calls, shape_pixels, FRACTIONAL_SCOPE,
                               shape_count if shape_count is not None else "",
                               f"{shape_calls / shape_count:.6f}" if shape_count else "",
                               f"{shape_pixels / shape_count:.6f}" if shape_count else ""])
            calls, pixels = fractional[(clip, encoder)]
            search_count = searches.get((clip, encoder))
            result.append([clip, encoder, "all", "ALL", "fractional_total", calls, pixels,
                           FRACTIONAL_SCOPE, search_count if search_count is not None else "",
                           f"{calls / search_count:.6f}" if search_count else "",
                           f"{pixels / search_count:.6f}" if search_count else ""])
            calls, pixels = totals[(clip, encoder)]
            result.append([clip, encoder, "all", "ALL", "all_recorded_total", calls,
                           pixels, "within_encoder_only", "", "", ""])
    return result


def write_json(path: Path, report: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"valid": False, "errors": [], "comparison_note": COMPARISON_NOTE,
              "hash_check_note": HASH_CHECK_NOTE,
              "clips": {clip: {} for clip in CLIPS}}
    try:
        manifests = {encoder: read_manifest(args.input_dir / encoder / "manifest.csv", encoder)
                     for encoder in ENCODERS}
        libvpx_entry_manifest = read_manifest(args.input_dir / "libvpx-entry" / "manifest.csv",
                                              "libvpx", with_searches=True)
        rvp9_entry_manifest = read_manifest(args.input_dir / "rvp9-entry" / "manifest.csv",
                                            "rvp9")
        captures = {}
        searches = {}
        shape_searches = {}
        for clip in CLIPS:
            for encoder in ENCODERS:
                rows = read_capture(args.input_dir / encoder / f"{clip}.stderr.txt", encoder)
                manifest = manifests[encoder][clip]
                if len(rows) != manifest["census_rows"]:
                    raise ValueError(f"{encoder}/{clip}: CensusRows {manifest['census_rows']} "
                                     f"does not match {len(rows)} captured rows")
                captures[(clip, encoder)] = rows
                report["clips"][clip][encoder] = manifest
            entry_path = args.input_dir / "libvpx-entry" / f"{clip}.stderr.txt"
            entry_rows = read_capture(entry_path, "libvpx")
            entry = libvpx_entry_manifest[clip]
            if len(entry_rows) != entry["census_rows"]:
                raise ValueError(f"libvpx-entry/{clip}: CensusRows {entry['census_rows']} "
                                 f"does not match {len(entry_rows)} captured rows")
            if sorted(entry_rows) != sorted(captures[(clip, "libvpx")]):
                raise ValueError(f"libvpx-entry/{clip}: SVF rows differ from original capture")
            search_count = read_libvpx_searches(entry_path)
            if search_count != entry["searches"]:
                raise ValueError(f"libvpx-entry/{clip}: Searches {search_count} "
                                 f"does not match manifest {entry['searches']}")
            searches[(clip, "libvpx")] = search_count
            report["clips"][clip]["libvpx"].update({
                "searches": search_count,
                "search_entry_binary_sha256": entry["binary_sha256"],
                "svf_rows_match_original": True,
            })
            rvp9_entry_path = args.input_dir / "rvp9-entry" / f"{clip}.stderr.txt"
            rvp9_entry_rows = read_capture(rvp9_entry_path, "rvp9")
            rvp9_entry = rvp9_entry_manifest[clip]
            if len(rvp9_entry_rows) != rvp9_entry["census_rows"]:
                raise ValueError(f"rvp9-entry/{clip}: CensusRows {rvp9_entry['census_rows']} "
                                 f"does not match {len(rvp9_entry_rows)} captured rows")
            if sorted(rvp9_entry_rows) != sorted(captures[(clip, "rvp9")]):
                raise ValueError(f"rvp9-entry/{clip}: SVF rows differ from original capture")
            rvp9_searches = read_rvp9_searches(rvp9_entry_path)
            captured_shapes = {(row[0], row[1], row[2]) for row in captures[(clip, "rvp9")]}
            if set(rvp9_searches) != captured_shapes:
                raise ValueError(f"rvp9-entry/{clip}: search keys differ from census family/shapes")
            for (family, width, height), count in rvp9_searches.items():
                shape_searches[(clip, "rvp9", family, width, height)] = count
            searches[(clip, "rvp9")] = sum(rvp9_searches.values())
            report["clips"][clip]["rvp9"].update({
                "searches": searches[(clip, "rvp9")],
                "search_rows": len(rvp9_searches),
                "search_entry_binary_sha256": rvp9_entry["binary_sha256"],
                "svf_rows_match_original": True,
            })
        summary = aggregate(captures, searches, shape_searches)
        with (args.out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(SUMMARY_HEADER)
            writer.writerows(summary)
        report["valid"] = True
        report["summary_rows"] = len(summary)
    except (OSError, ValueError, csv.Error) as error:
        report["errors"].append(str(error))
    write_json(args.out_dir / "validation.json", report)
    if not report["valid"]:
        print(report["errors"][0], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
