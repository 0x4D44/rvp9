#!/usr/bin/env python3
"""Map one xperf CPU trace to function symbols in the exact pinned vpxenc PE.

No PDB or source-line information is assumed. The COFF type-20 table omits
assembly functions in this binary, so nm T/t symbols provide the boundaries.
COFF type-20 confirms which lowercase t symbols are functions rather than
assembly-local labels. Symbol starts have no sizes; each reported interval
ends at the next function start (an upper bound for attribution).
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import hashlib
import json
import locale
import ntpath
import re
import struct
import subprocess
import sys
from pathlib import Path


EXPECTED_EXE_SHA256 = "88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d"
FUNCTION_ROW = re.compile(
    r"^\[\s*\d+\]\(sec\s+(\d+)\).*?\(ty\s+20\).*?0x([0-9a-fA-F]+)\s+(\S+)\s*$"
)
NM_ROW = re.compile(r"^([0-9a-fA-F]+)\s+([A-Za-z])\s+(\S+)\s*$")
PID_NAME = re.compile(r"^(.*?)\s*\(\s*(\d+)\s*\)\s*$")
HEX_AT_END = re.compile(r"0x([0-9a-fA-F]+)\s*$")
LOSS_ROW = re.compile(r"^\s*(?:Total\s+)?#\s+Lost\s+(Buffers|Events)\s*:\s*(\d+)\s*$", re.I)
TRACE_DURATION_ROW = re.compile(
    r"^End time \(UTC\).*?\(\+\s*(\d+):(\d+):(\d+):(\d+)\.(\d+)\)"
)


def fail(message: str) -> None:
    raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pe(path: Path) -> tuple[int, int, dict[int, dict[str, int | str]]]:
    """Return preferred image base, SizeOfImage, and one-based PE sections."""
    with path.open("rb") as stream:
        stream.seek(0x3C)
        pe_offset = struct.unpack("<I", stream.read(4))[0]
        stream.seek(pe_offset)
        header = stream.read(24)
        if len(header) != 24 or header[:4] != b"PE\0\0":
            fail("invalid PE header")
        section_count = struct.unpack_from("<H", header, 6)[0]
        optional_size = struct.unpack_from("<H", header, 20)[0]
        if not 1 <= section_count <= 96:
            fail(f"implausible PE section count: {section_count}")
        optional = stream.read(optional_size)
        if len(optional) != optional_size:
            fail("truncated PE optional header")
        magic = struct.unpack_from("<H", optional)[0]
        if magic == 0x20B:
            preferred_base = struct.unpack_from("<Q", optional, 24)[0]
        elif magic == 0x10B:
            preferred_base = struct.unpack_from("<I", optional, 28)[0]
        else:
            fail(f"unsupported PE optional-header magic {magic:#x}")
        image_size = struct.unpack_from("<I", optional, 56)[0]
        sections: dict[int, dict[str, int | str]] = {}
        for index in range(1, section_count + 1):
            raw = stream.read(40)
            if len(raw) != 40:
                fail("truncated PE section table")
            name = raw[:8].split(b"\0", 1)[0].decode("ascii", "replace")
            virtual_size, virtual_address, raw_size = struct.unpack_from("<III", raw, 8)
            characteristics = struct.unpack_from("<I", raw, 36)[0]
            sections[index] = {
                "name": name,
                "rva": virtual_address,
                "size": max(virtual_size, raw_size),
                "executable": int(bool(characteristics & 0x20000000)),
            }
    if not image_size or not preferred_base:
        fail("invalid PE image base or SizeOfImage")
    return preferred_base, image_size, sections


def run_tool(command: list[str], output: Path, timeout: int) -> None:
    with output.open("wb") as stream:
        try:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            fail(f"{command[0]} timed out after {timeout}s: {error}")
    if result.returncode:
        fail(f"{command[0]} exited {result.returncode}; see {output}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="mbcs" if sys.platform == "win32" else locale.getpreferredencoding(False), errors="replace")


def function_table(
    objdump_text: str,
    nm_text: str,
    sections: dict[int, dict[str, int | str]],
    preferred_base: int,
) -> tuple[list[int], dict[int, dict[str, object]], dict[str, int]]:
    grouped: dict[int, set[str]] = collections.defaultdict(set)
    objdump_named: list[tuple[str, int]] = []
    for line in objdump_text.splitlines():
        match = FUNCTION_ROW.match(line)
        if not match:
            continue
        section = sections.get(int(match.group(1)))
        if section is None or not section["executable"]:
            continue
        offset = int(match.group(2), 16)
        rva = int(section["rva"]) + offset
        section_start = int(section["rva"])
        section_end = section_start + int(section["size"])
        if not section_start <= rva < section_end:
            fail(f"function {match.group(3)} lies outside {section['name']}")
        name = match.group(3)
        objdump_named.append((name, preferred_base + rva))
    if len(objdump_named) < 1000:
        fail(f"only {len(objdump_named)} executable function symbols found")

    typed_starts = set(objdump_named)
    nm_addresses: dict[str, set[int]] = collections.defaultdict(set)
    nm_total = 0
    nm_text_rows = 0
    skipped_section_markers = 0
    skipped_local_labels: list[tuple[str, int]] = []
    added_global_assembly_rows = 0
    for line in nm_text.splitlines():
        match = NM_ROW.match(line)
        if match:
            nm_total += 1
            if match.group(2) in "Tt":
                nm_text_rows += 1
                va = int(match.group(1), 16)
                kind = match.group(2)
                name = match.group(3)
                nm_addresses[name].add(va)
                if name.startswith(".text"):
                    skipped_section_markers += 1
                    continue
                if kind == "t" and (name, va) not in typed_starts:
                    skipped_local_labels.append((name, va))
                    continue
                rva = va - preferred_base
                if not any(
                    section["executable"] and int(section["rva"]) <= rva < int(section["rva"]) + int(section["size"])
                    for section in sections.values()
                ):
                    fail(f"nm text symbol {name} at {va:#x} is outside executable PE sections")
                grouped[rva].add(name)
                if kind == "T" and (name, va) not in typed_starts:
                    added_global_assembly_rows += 1
    checked = [(name, va) for name, va in objdump_named if name in nm_addresses]
    mismatch = [(name, va) for name, va in checked if va not in nm_addresses[name]]
    missing_typed = [(name, va) for name, va in objdump_named if name not in nm_addresses]
    if len(checked) < 1000 or mismatch or missing_typed:
        fail(
            f"objdump/nm function address reconciliation failed: checked={len(checked)}, "
            f"mismatches={len(mismatch)}, missing={len(missing_typed)}, "
            f"first={(mismatch + missing_typed)[:3]}"
        )
    included_names = {name for names in grouped.values() for name in names}
    unexplained_labels = [
        (name, va) for name, va in skipped_local_labels
        if not (name.startswith(".") or name in {"one", "limit"}
                or "." in name and name.rsplit(".", 1)[0] in included_names)
    ]
    if unexplained_labels:
        fail(f"unclassified nm local text labels; inspect before dropping boundaries: {unexplained_labels[:5]}")
    starts = sorted(grouped)
    functions: dict[int, dict[str, object]] = {}
    for index, start in enumerate(starts):
        section = next(
            item for item in sections.values()
            if item["executable"] and int(item["rva"]) <= start < int(item["rva"]) + int(item["size"])
        )
        section_end = int(section["rva"]) + int(section["size"])
        next_start = starts[index + 1] if index + 1 < len(starts) else section_end
        end = min(next_start, section_end)
        names = sorted(grouped[start], key=lambda name: (
            name.startswith("."),
            (name, preferred_base + start) not in typed_starts,
            name.startswith("__"), len(name), name,
        ))
        source = "nm+coff-type20" if any((name, preferred_base + start) in typed_starts for name in names) else "nm-global-text"
        functions[start] = {"name": names[0], "aliases": names, "end_rva": end, "source": source}
    return starts, functions, {
        "objdump_function_rows": len(objdump_named),
        "unique_function_starts": len(starts),
        "nm_defined_symbols": nm_total,
        "nm_text_rows": nm_text_rows,
        "nm_section_markers_excluded": skipped_section_markers,
        "nm_local_labels_excluded": len(skipped_local_labels),
        "nm_global_assembly_rows_added": added_global_assembly_rows,
        "nm_unclassified_local_labels": len(unexplained_labels),
        "reconciled_function_rows": len(checked),
        "reconciliation_mismatches": len(mismatch),
    }


def assert_pinned_assembly_boundaries(functions: dict[int, dict[str, object]]) -> None:
    """Guard the exact-EXE boundaries that the old type-20-only map missed."""
    expected = {
        0x128BA0: ("vpx_clear_system_state", 0x128BB0),
        0x132F60: ("vpx_sad16x16x4d_sse2", 0x1333E0),
        0x13A220: ("vpx_sad16x16_sse2", 0x13A290),
    }
    for start, (name, end) in expected.items():
        entry = functions.get(start)
        if entry is None or entry["name"] != name or entry["end_rva"] != end:
            fail(f"pinned assembly boundary changed at RVA {start:#x}: expected {name}..{end:#x}, got {entry}")


def field(value: str) -> str:
    return value.strip().strip('"').strip()


def process_identity(value: str) -> tuple[str, int | None]:
    match = PID_NAME.match(field(value))
    if match:
        return ntpath.basename(match.group(1)).casefold(), int(match.group(2))
    return ntpath.basename(field(value)).casefold(), None


def parse_dumper(
    raw_path: Path, exe: Path, pid: int, image_size: int
) -> tuple[int, int, int, list[dict[str, int]], dict[str, object]]:
    images: dict[int, dict[str, object]] = {}
    samples: list[dict[str, int]] = []
    all_process_count = 0
    exe_count = 0
    other_module_count = 0
    malformed_other_rows = 0
    encoding = "mbcs" if sys.platform == "win32" else locale.getpreferredencoding(False)
    with raw_path.open("r", encoding=encoding, errors="replace", newline="") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            try:
                fields = next(csv.reader([raw_line], skipinitialspace=True, strict=True))
            except csv.Error as error:
                if raw_line.lstrip().startswith("SampledProfile") and exe.name.casefold() in raw_line.casefold():
                    fail(f"malformed target dumper row {line_number}: {error}")
                malformed_other_rows += 1
                continue
            if not fields:
                continue
            event = fields[0].strip().lstrip("\ufeff")
            if event in ("I-Start", "ImageId"):
                if len(fields) < 5:
                    continue
                name, image_pid = process_identity(fields[2])
                if name != exe.name.casefold() or image_pid != pid:
                    continue
                image_field = fields[8] if event == "I-Start" and len(fields) > 8 else fields[-1]
                image_path = field(image_field)
                if ntpath.basename(image_path).casefold() != exe.name.casefold():
                    continue
                if re.match(r"^[A-Za-z]:[\\/]", image_path) and ntpath.normcase(ntpath.normpath(image_path)) != ntpath.normcase(ntpath.normpath(str(exe))):
                    fail(f"target image path differs from pinned EXE: {image_path}")
                base = int(field(fields[3]), 16)
                extent = int(field(fields[4]), 16)
                size = extent - base if event == "I-Start" else extent
                if size != image_size:
                    fail(f"xperf image size {size:#x} differs from PE SizeOfImage {image_size:#x}")
                found = {"base": base, "size": size, "event": event, "path": image_path}
                if pid in images and (images[pid]["base"], images[pid]["size"]) != (base, size):
                    fail("conflicting loaded image ranges for target PID")
                images[pid] = found
                continue
            if event != "SampledProfile":
                continue
            if len(fields) < 3:
                continue
            name, sample_pid = process_identity(fields[2])
            if name != exe.name.casefold() or sample_pid != pid:
                continue
            if len(fields) < 10:
                fail(f"target SampledProfile row {line_number} has {len(fields)} fields")
            timestamp = int(field(fields[1]).replace(",", ""), 10)
            tid = int(field(fields[3]).replace(",", ""), 10)
            ip = int(field(fields[4]), 16)
            count = int(field(fields[8]).replace(",", ""), 10)
            if count <= 0:
                fail(f"non-positive target sample count at row {line_number}")
            module_text = field(fields[7])
            module_name, separator, address_text = module_text.partition("!")
            match = HEX_AT_END.search(address_text) if separator else None
            if match is None or int(match.group(1), 16) != ip:
                fail(f"target sample IP text mismatch at row {line_number}: {module_text}")
            all_process_count += count
            if ntpath.basename(module_name).casefold() == exe.name.casefold():
                exe_count += count
                samples.append({"timestamp": timestamp, "tid": tid, "ip": ip, "count": count})
            else:
                other_module_count += count
    if pid not in images or not samples or not all_process_count:
        fail(f"missing image range or sampled EXE IPs for target PID {pid}")
    base = int(images[pid]["base"])
    for sample in samples:
        if not base <= sample["ip"] < base + image_size:
            fail(f"sample IP {sample['ip']:#x} outside loaded image")
    return base, exe_count, all_process_count, samples, {
        "image": images[pid],
        "other_module_direct_count": other_module_count,
        "malformed_non_target_rows": malformed_other_rows,
    }


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etl", required=True, type=Path)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--cpu-seconds", required=True, type=float)
    parser.add_argument("--wall-seconds", required=True, type=float)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--replay-from", type=Path,
                        help="read prior xperf/nm/objdump artifacts; write results only to a new output directory")
    parser.add_argument("--xperf", default="xperf.exe")
    parser.add_argument("--nm", default=r"C:\Strawberry\c\bin\nm.exe")
    parser.add_argument("--objdump", default=r"C:\Strawberry\c\bin\objdump.exe")
    args = parser.parse_args()
    if args.pid <= 0 or args.cpu_seconds <= 0 or args.wall_seconds <= 0:
        fail("PID, process CPU seconds, and wall seconds must be positive")
    if not args.etl.is_file() or not args.exe.is_file():
        fail("existing ETL and EXE are required")
    if args.replay_from is None and not args.outdir.is_dir():
        fail("live extraction needs an existing output directory")
    if args.replay_from is not None:
        if not args.replay_from.is_dir():
            fail(f"replay artifact directory does not exist: {args.replay_from}")
        if args.outdir.exists() or not args.outdir.parent.is_dir():
            fail("replay output must be a new directory under an existing parent")
    exe_sha = sha256(args.exe)
    if exe_sha != EXPECTED_EXE_SHA256:
        fail(f"pinned vpxenc SHA-256 mismatch: {exe_sha}")
    preferred_base, image_size, sections = read_pe(args.exe)
    artifact_root = args.replay_from or args.outdir
    paths = {
        "tracestats": artifact_root / "xperf-tracestats.txt",
        "profile": artifact_root / "xperf-profile.txt",
        "dumper": artifact_root / "xperf-dumper.txt",
        "nm": artifact_root / "nm-symbols.txt",
        "objdump": artifact_root / "objdump-symbols.txt",
    }
    if args.replay_from is None:
        for path in paths.values():
            if path.exists():
                fail(f"refusing to overwrite analysis artifact: {path}")
        run_tool([args.xperf, "-i", str(args.etl), "-a", "tracestats"], paths["tracestats"], 120)
    else:
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            fail(f"replay artifacts are missing: {missing}")
        args.outdir.mkdir()
    stats_text = read_text(paths["tracestats"])
    loss_rows = [(kind.lower(), int(value)) for line in stats_text.splitlines()
                 if (match := LOSS_ROW.match(line)) for kind, value in [match.groups()]]
    totals = {kind: value for kind, value in loss_rows[:2]}
    if set(totals) != {"buffers", "events"} or any(value != 0 for _, value in loss_rows):
        fail(f"ETL loss is nonzero or unverifiable: {loss_rows}")
    duration_match = next((match for line in stats_text.splitlines()
                           if (match := TRACE_DURATION_ROW.match(line))), None)
    if duration_match is None:
        fail("xperf tracestats did not report ETL duration")
    days, hours, minutes, seconds, fraction = duration_match.groups()
    trace_seconds = (int(days) * 86400 + int(hours) * 3600 + int(minutes) * 60
                     + int(seconds) + float("0." + fraction))
    if trace_seconds <= 0 or trace_seconds + 0.001 < args.wall_seconds:
        fail(f"ETL duration {trace_seconds}s does not cover encoder wall time {args.wall_seconds}s")
    if args.replay_from is None:
        run_tool([args.xperf, "-i", str(args.etl), "-a", "profile"], paths["profile"], 120)
        run_tool([args.xperf, "-i", str(args.etl), "-a", "dumper", "-add_fieldnames"], paths["dumper"], 600)
        run_tool([args.nm, "-n", "--defined-only", str(args.exe)], paths["nm"], 120)
        run_tool([args.objdump, "-t", str(args.exe)], paths["objdump"], 120)
    if sha256(args.exe) != EXPECTED_EXE_SHA256:
        fail("pinned executable changed during extraction")
    starts, functions, symbol_counts = function_table(
        read_text(paths["objdump"]), read_text(paths["nm"]), sections, preferred_base
    )
    assert_pinned_assembly_boundaries(functions)
    base, exe_count, process_count, samples, parse_details = parse_dumper(
        paths["dumper"], args.exe, args.pid, image_size
    )
    sample_rows: list[list[object]] = []
    by_function: collections.Counter[tuple[int | None, str]] = collections.Counter()
    unresolved_count = 0
    for sample in samples:
        rva = sample["ip"] - base
        index = bisect.bisect_right(starts, rva) - 1
        start = starts[index] if index >= 0 else None
        function = functions[start] if start is not None else None
        if function is None or rva >= int(function["end_rva"]):
            function = None
            start = None
            unresolved_count += sample["count"]
        name = str(function["name"]) if function else "<unresolved>"
        offset = rva - start if start is not None else ""
        interval_end = f"0x{int(function['end_rva']):x}" if function else ""
        start_source = str(function["source"]) if function else ""
        sample_rows.append([
            sample["timestamp"], args.pid, sample["tid"], f"0x{sample['ip']:x}",
            sample["count"], f"0x{base:x}", f"0x{rva:x}", name,
            f"0x{start:x}" if start is not None else "", offset, interval_end, start_source,
        ])
        by_function[(start, name)] += sample["count"]
    sample_path = args.outdir / "sampled-ips.csv"
    function_path = args.outdir / "summary-by-function.csv"
    for path in (sample_path, function_path, args.outdir / "analysis.json"):
        if path.exists():
            fail(f"refusing to overwrite analysis artifact: {path}")
    write_csv(sample_path,
              ["timestamp", "pid", "tid", "ip", "direct_count", "loaded_image_base", "rva",
               "function", "function_start_rva", "offset_from_start", "symbol_interval_end_rva",
               "function_start_source"],
              sample_rows)
    function_rows: list[list[object]] = []
    for (start, name), count in by_function.most_common():
        estimated_cpu_ms = count / process_count * args.cpu_seconds * 1000.0
        function_rows.append([
            name, f"0x{start:x}" if start is not None else "", count,
            f"{count / exe_count:.8f}", f"{count / process_count:.8f}",
            f"{estimated_cpu_ms:.3f}", f"{estimated_cpu_ms / 60.0:.4f}",
            "|".join(functions[start]["aliases"]) if start is not None else "",
            functions[start]["source"] if start is not None else "",
        ])
    write_csv(function_path,
              ["function", "start_rva", "direct_count", "share_of_exe_samples",
               "share_of_process_samples", "estimated_cpu_ms_per_encode",
               "estimated_cpu_ms_per_frame", "aliases", "function_start_source"], function_rows)
    artifact_sha = {name: sha256(path) for name, path in paths.items()}
    artifact_sha.update({"sampled_ips": sha256(sample_path), "summary_by_function": sha256(function_path)})
    summary = {
        "method": "xperf SampledProfile direct IP -> PID-specific image base -> RVA -> exact PE nm T/t function start",
        "interval_limit": "nm gives starts without sizes; next function start is an attribution upper bound",
        "label_policy": "nm global T text starts are retained, including assembly functions omitted by COFF type 20; lowercase t starts require COFF type 20; section markers and classified assembly-local labels are excluded",
        "source_lines": "not available for libvpx functions; no DWARF line claims made",
        "replay_from": str(args.replay_from) if args.replay_from else None,
        "exe_path": str(args.exe), "exe_sha256": exe_sha,
        "etl_path": str(args.etl), "etl_sha256": sha256(args.etl),
        "pid": args.pid, "preferred_image_base": f"0x{preferred_base:x}",
        "loaded_image_base": f"0x{base:x}", "image_size": f"0x{image_size:x}",
        "image_record": parse_details["image"],
        "lost_buffers": totals["buffers"], "lost_events": totals["events"],
        "all_loss_rows": loss_rows,
        "process_cpu_seconds": args.cpu_seconds, "encoder_wall_seconds": args.wall_seconds,
        "etl_trace_seconds": trace_seconds,
        "process_cpu_over_wall": args.cpu_seconds / args.wall_seconds,
        "frames": 60, "sampled_exe_rows": len(samples),
        "sampled_exe_direct_count": exe_count,
        "sampled_process_direct_count": process_count,
        "process_samples_per_cpu_second": process_count / args.cpu_seconds,
        "sampled_other_modules_direct_count": parse_details["other_module_direct_count"],
        "unresolved_exe_direct_count": unresolved_count,
        "malformed_non_target_rows": parse_details["malformed_non_target_rows"],
        "symbol_counts": symbol_counts,
        "normalization": "estimated function CPU = function direct count / all target-process direct count * measured process CPU; sampling estimate, not timed function duration",
        "artifacts_sha256": artifact_sha,
    }
    analysis_path = args.outdir / "analysis.json"
    analysis_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"exe_samples={exe_count} process_samples={process_count} unresolved={unresolved_count}")
    print(f"lost_buffers={totals['buffers']} lost_events={totals['events']}")
    print(f"function_summary={function_path}")
    print(f"analysis={analysis_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, struct.error, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
