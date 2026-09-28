#!/usr/bin/env python3
"""Replay the three fresh PGO ETLs and attribute inclusive loop-filter stacks.

This is deliberately a two-pass replay.  The first pass uses the reviewed
``extract-ip.py`` parser to identify every target-process SampledProfile row
and its timestamp/TID key.  The second pass retains only Stack rows for those
keys, so the parser never treats module totals as loop-filter cost.  RVP9
addresses are resolved with the exact paired PDB through the same DbgHelp
resolver used by the committed IP evidence.

The script is intended to run on Crucible.  Its dumper text and JSON output
must live under the existing task .scratch directory; no ETL or generated
artifact belongs in the repository.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable


IMAGE_BASE = 0x00007FF626790000
IMAGE_SIZE = 0x263000
IMAGE_END = IMAGE_BASE + IMAGE_SIZE
EXE_NAME = "rvp9enc.exe"
EXE_SHA = "3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad"
PDB_SHA = "c41525ee2ccd56d6bda7e4e9e0880452437d5ac7981bc2af4f1b7c34a71943f8"

CLIPS: dict[str, dict[str, object]] = {
    "ducks": {
        "pid": 64104,
        "expected_rows": 20092,
        "expected_direct_rows": 18127,
        "etl_sha": "1039581496d25e222c214fb4a31d7aab58f9238b2052adb2c0747ea5fd54668d",
    },
    "tree": {
        "pid": 45240,
        "expected_rows": 9698,
        "expected_direct_rows": 8195,
        "etl_sha": "ff0ec450487f7f6dd2f4ffdb35150c131db7333b43771c8586658920a7c118ad",
    },
    "town": {
        "pid": 66556,
        "expected_rows": 10282,
        "expected_direct_rows": 8629,
        "etl_sha": "b7be117a62e8571f7d60d028214472bba94bb55e84d6e7434a549eefe60b0de8",
    },
}


@dataclass(frozen=True)
class StackFrame:
    depth: int
    address: int
    module: str
    image_text: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ip_parser(repo_root: Path):
    path = repo_root / "wrk_docs" / "evidence" / "2026-09-25-unit1434-three-clip-ip-attribution" / "extract-ip.py"
    spec = importlib.util.spec_from_file_location("reviewed_extract_ip", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reviewed parser: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fields(raw_line: str) -> list[str]:
    return next(csv.reader([raw_line], skipinitialspace=True, strict=False))


def decimal(value: str) -> int:
    return int(value.strip().strip('"'), 10)


def address(value: str) -> int:
    return int(value.strip().strip('"'), 16)


def module_name(image_text: str) -> str:
    value = image_text.strip().strip('"')
    return value.split("!", 1)[0].strip().casefold()


def run_dumper(xperf: str, etl: Path, dump: Path, timeout: int, reuse: bool) -> None:
    if reuse and dump.is_file() and dump.stat().st_size > 0:
        return
    command = [xperf, "-i", os.fspath(etl), "-a", "dumper", "-add_fieldnames"]
    dump.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dump.open("w", encoding="utf-8", newline="") as output:
            completed = subprocess.run(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"xperf dumper exceeded {timeout}s: {' '.join(command)}") from error
    if completed.returncode != 0:
        raise RuntimeError(f"xperf dumper exited {completed.returncode}: {' '.join(command)}")


def parse_target_samples(ip_parser, dump: Path, pid: int) -> tuple[object, list[object]]:
    fallback = ip_parser.ImageRange(pid, IMAGE_BASE, IMAGE_SIZE, "I-Start")
    with dump.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        parsed = ip_parser.parse_dumper_rows(
            stream,
            EXE_NAME,
            pid_filter=pid,
            fallback_image=fallback,
        )
    samples = list(parsed.samples) + list(parsed.other_module_samples)
    return parsed, samples


def parse_target_stacks(dump: Path, sample_keys: set[tuple[int, int]]) -> tuple[dict[tuple[int, int], list[StackFrame]], int]:
    frames: dict[tuple[int, int], list[StackFrame]] = collections.defaultdict(list)
    stack_rows = 0
    with dump.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        for raw_line in stream:
            if not raw_line.lstrip().startswith("Stack,"):
                continue
            try:
                row = fields(raw_line.rstrip("\r\n"))
                if len(row) < 6 or row[0].strip() != "Stack":
                    continue
                timestamp = decimal(row[1])
                tid = decimal(row[2])
                depth = decimal(row[3])
                frame_address = address(row[4])
            except (ValueError, csv.Error):
                continue
            stack_rows += 1
            key = (timestamp, tid)
            if key not in sample_keys:
                continue
            image_text = ",".join(row[5:]).strip()
            frames[key].append(
                StackFrame(depth, frame_address, module_name(image_text), image_text)
            )
    for key in frames:
        frames[key].sort(key=lambda frame: frame.depth)
    return dict(frames), stack_rows


def source_key(info: object) -> tuple[str, int, str, bool, bool]:
    filename = str(getattr(info, "file", ""))
    line = int(getattr(info, "line", 0))
    symbol = str(getattr(info, "symbol", ""))
    return (
        filename.casefold(),
        line,
        symbol,
        bool(getattr(info, "symbol_ok", False)),
        bool(getattr(info, "line_ok", False)),
    )


def is_file_line(info: object, basename: str, first: int, last: int) -> bool:
    filename, line, _symbol, _symbol_ok, line_ok = source_key(info)
    return line_ok and filename.replace("/", "\\").endswith("\\" + basename.casefold()) and first <= line <= last


def resolve_stack_addresses(ip_parser, repo_root: Path, exe: Path, pdb: Path, pid: int, frames: Iterable[StackFrame]) -> dict[int, object]:
    addresses = sorted(
        {
            frame.address
            for frame in frames
            if frame.module == EXE_NAME.casefold() and IMAGE_BASE <= frame.address < IMAGE_END
        }
    )
    fake_samples = [
        ip_parser.Sample(
            pid=pid,
            timestamp=0,
            tid=0,
            ip=frame_address,
            raw_count=1,
            module_text=EXE_NAME,
            module_ip_text=frame_address,
            raw_fields=(),
            raw_line="",
        )
        for frame_address in addresses
    ]
    image = ip_parser.ImageRange(pid, IMAGE_BASE, IMAGE_SIZE, "I-Start")
    resolved = ip_parser.resolve_symbols(exe, pdb, {pid: image}, fake_samples)
    return {frame_address: resolved[(pid, frame_address)] for frame_address in addresses}


def classify(samples: list[object], frames_by_key: dict[tuple[int, int], list[StackFrame]], symbols: dict[int, object]) -> dict[str, object]:
    counters: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    top_lines: collections.Counter[tuple[str, int, str]] = collections.Counter()
    joined_rows = 0
    joined_raw = 0
    depths: list[int] = []
    resolved_frame_rows = 0
    unresolved_exe_frame_rows = 0
    line_resolved_frame_rows = 0
    line_unresolved_frame_rows = 0
    line_unresolved_sample_rows = 0
    line_unresolved_outside_call_block_rows = 0
    runtime_modules = {"ucrtbase.dll", "vcruntime140.dll", "ntdll.dll", "kernelbase.dll"}

    for sample in samples:
        key = (int(sample.timestamp), int(sample.tid))
        stack = frames_by_key.get(key)
        if not stack:
            continue
        joined_rows += 1
        weight = int(sample.raw_count)
        joined_raw += weight
        depths.append(len(stack))
        infos: list[object] = []
        sample_has_line_unresolved = False
        for frame in stack:
            if frame.module != EXE_NAME.casefold() or not (IMAGE_BASE <= frame.address < IMAGE_END):
                continue
            info = symbols.get(frame.address)
            if info is None:
                unresolved_exe_frame_rows += 1
                continue
            resolved_frame_rows += 1
            if bool(getattr(info, "line_ok", False)):
                line_resolved_frame_rows += 1
            else:
                line_unresolved_frame_rows += 1
                sample_has_line_unresolved = True
            infos.append(info)

        has_exact_callsite = any(
            is_file_line(info, "frame_loop.rs", 5005, 5005) for info in infos
        )
        has_call_block = any(
            is_file_line(info, "frame_loop.rs", 5003, 5018) for info in infos
        )
        if sample_has_line_unresolved:
            line_unresolved_sample_rows += 1
            if not has_call_block:
                line_unresolved_outside_call_block_rows += 1
        block_infos = [
            info
            for info in infos
            if is_file_line(info, "loopfilter.rs", 229, 321)
        ]
        has_any_lf = bool(block_infos)
        strict_exact = has_exact_callsite and has_any_lf
        strict_block = has_call_block and has_any_lf

        def add(name: str) -> None:
            counters[name]["rows"] += 1
            counters[name]["raw_count"] += weight

        if has_exact_callsite:
            add("frame_loop_5005_exact")
        if has_call_block:
            add("frame_loop_call_block")
        if has_any_lf:
            add("loopfilter_block_any_callsite")
        if strict_exact:
            add("fallback_chain_exact_5005")
        if strict_block:
            add("fallback_chain_call_block")

        if strict_block:
            category_hits: set[str] = set()
            for info in block_infos:
                _filename, line, symbol, _symbol_ok, _line_ok = source_key(info)
                if 229 <= line <= 236:
                    category_hits.add("alloc_or_zero_source")
                elif 238 <= line <= 262:
                    category_hits.add("copy_in_source")
                elif 264 <= line <= 313:
                    category_hits.add("filter_source")
                elif 314 <= line <= 321:
                    category_hits.add("copy_back_source")
                top_lines[(_filename, line, symbol)] += 1
            for category in category_hits:
                add(category)
            if any(frame.module in runtime_modules for frame in stack) and "alloc_or_zero_source" in category_hits:
                add("alloc_or_zero_with_runtime_frame")
        elif has_any_lf:
            for info in block_infos:
                filename, line, symbol, _symbol_ok, _line_ok = source_key(info)
                top_lines[(filename, line, symbol)] += 1

    return {
        "joined_sample_rows": joined_rows,
        "joined_raw_count": joined_raw,
        "stack_depth_min": min(depths) if depths else 0,
        "stack_depth_median": median(depths) if depths else 0,
        "stack_depth_max": max(depths) if depths else 0,
        "resolved_exe_frame_rows": resolved_frame_rows,
        "unresolved_exe_frame_rows": unresolved_exe_frame_rows,
        "line_resolved_exe_frame_rows": line_resolved_frame_rows,
        "line_unresolved_exe_frame_rows": line_unresolved_frame_rows,
        "line_unresolved_sample_rows": line_unresolved_sample_rows,
        "line_unresolved_outside_call_block_rows": line_unresolved_outside_call_block_rows,
        "counts": {name: dict(counts) for name, counts in sorted(counters.items())},
        "top_resolved_loopfilter_lines": [
            {
                "file": filename,
                "line": line,
                "symbol": symbol,
                "joined_sample_rows": count,
            }
            for (filename, line, symbol), count in top_lines.most_common(20)
        ],
    }


def run_clip(name: str, spec: dict[str, object], args: argparse.Namespace, ip_parser) -> dict[str, object]:
    pid = int(spec["pid"])
    etl = args.etl_root / name / "capture.etl"
    outdir = args.out_root / name
    dump = outdir / "xperf-dumper.txt"
    exe = args.repo_root / ".scratch" / "fresh-integrated-pgo" / "optimized" / "release" / EXE_NAME
    pdb = exe.with_suffix(".pdb")
    observed = {
        "clip": name,
        "pid": pid,
        "etl": os.fspath(etl),
        "exe": os.fspath(exe),
        "pdb": os.fspath(pdb),
        "expected_target_sample_rows": int(spec["expected_rows"]),
        "expected_direct_sample_rows": int(spec["expected_direct_rows"]),
        "etl_sha256": sha256(etl),
        "exe_sha256": sha256(exe),
        "pdb_sha256": sha256(pdb),
    }
    if observed["etl_sha256"] != spec["etl_sha"]:
        raise RuntimeError(f"{name}: ETL SHA mismatch: {observed['etl_sha256']} != {spec['etl_sha']}")
    if observed["exe_sha256"] != EXE_SHA:
        raise RuntimeError(f"{name}: EXE SHA mismatch: {observed['exe_sha256']} != {EXE_SHA}")
    if observed["pdb_sha256"] != PDB_SHA:
        raise RuntimeError(f"{name}: PDB SHA mismatch: {observed['pdb_sha256']} != {PDB_SHA}")

    run_dumper(args.xperf, etl, dump, args.timeout, args.reuse_dump)
    parsed, samples = parse_target_samples(ip_parser, dump, pid)
    direct_samples = list(parsed.samples)
    observed["target_process_rows"] = int(parsed.target_process_rows)
    observed["target_process_raw_count"] = sum(int(sample.raw_count) for sample in samples)
    observed["direct_exe_rows"] = len(direct_samples)
    observed["direct_exe_raw_count"] = sum(int(sample.raw_count) for sample in direct_samples)
    observed["target_tids"] = sorted({int(sample.tid) for sample in samples})
    observed["sample_keys"] = len({(int(sample.timestamp), int(sample.tid)) for sample in samples})
    if observed["target_process_rows"] != spec["expected_rows"]:
        raise RuntimeError(f"{name}: target sample rows mismatch: {observed['target_process_rows']} != {spec['expected_rows']}")
    if observed["direct_exe_rows"] != spec["expected_direct_rows"]:
        raise RuntimeError(f"{name}: direct EXE rows mismatch: {observed['direct_exe_rows']} != {spec['expected_direct_rows']}")

    sample_keys = {(int(sample.timestamp), int(sample.tid)) for sample in samples}
    frames_by_key, stack_rows = parse_target_stacks(dump, sample_keys)
    observed["stack_rows_total"] = stack_rows
    observed["stack_keys_joined"] = len(frames_by_key)
    observed["stack_key_coverage"] = (
        len(frames_by_key) / len(sample_keys) if sample_keys else 0.0
    )
    all_frames = [frame for stack in frames_by_key.values() for frame in stack]
    observed["unique_rvp9_stack_addresses"] = len(
        {
            frame.address
            for frame in all_frames
            if frame.module == EXE_NAME.casefold() and IMAGE_BASE <= frame.address < IMAGE_END
        }
    )
    symbols = resolve_stack_addresses(ip_parser, args.repo_root, exe, pdb, pid, all_frames)
    observed["resolved_unique_rvp9_stack_addresses"] = len(symbols)
    observed["unresolved_unique_rvp9_stack_addresses"] = observed["unique_rvp9_stack_addresses"] - len(symbols)
    observed["line_resolved_unique_rvp9_stack_addresses"] = sum(
        bool(getattr(info, "line_ok", False)) for info in symbols.values()
    )
    observed["line_unresolved_unique_rvp9_stack_addresses"] = (
        observed["resolved_unique_rvp9_stack_addresses"]
        - observed["line_resolved_unique_rvp9_stack_addresses"]
    )
    observed["attribution"] = classify(samples, frames_by_key, symbols)
    observed["stack_join_complete"] = observed["stack_keys_joined"] == observed["sample_keys"]
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--etl-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--xperf", default="xperf.exe")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--reuse-dump", action="store_true")
    parser.add_argument("--clip", choices=tuple(CLIPS), action="append")
    args = parser.parse_args()

    ip_parser = load_ip_parser(args.repo_root)
    clips = args.clip or list(CLIPS)
    result: dict[str, object] = {
        "method": "two-pass xperf dumper replay; target SampledProfile keys joined to Stack timestamp/TID; paired PDB via DbgHelp",
        "xperf_command": [args.xperf, "-i", "<clip>\\capture.etl", "-a", "dumper", "-add_fieldnames"],
        "image": {"base": hex(IMAGE_BASE), "size": hex(IMAGE_SIZE), "end_exclusive": hex(IMAGE_END)},
        "exe_name": EXE_NAME,
        "exe_sha256": EXE_SHA,
        "pdb_sha256": PDB_SHA,
        "loopfilter_source_ranges": {
            "alloc_or_zero_source": "loopfilter.rs:229-236",
            "copy_in_source": "loopfilter.rs:238-262",
            "filter_source": "loopfilter.rs:264-313",
            "copy_back_source": "loopfilter.rs:314-321",
            "exact_fallback_callsite": "frame_loop.rs:5005",
            "fallback_call_block": "frame_loop.rs:5003-5018",
        },
        "clips": {},
    }
    try:
        for clip in clips:
            print(f"replay {clip}", flush=True)
            result["clips"][clip] = run_clip(clip, CLIPS[clip], args, ip_parser)  # type: ignore[index]
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(result["error"], file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
