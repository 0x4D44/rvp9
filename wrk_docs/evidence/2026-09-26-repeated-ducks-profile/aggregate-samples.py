#!/usr/bin/env python3
"""Aggregate repeated, exact-image Ducks IP extractions without losing any run.

Example (the directories must already contain extract-ip.py outputs)::

    python aggregate-samples.py --expected-runs 3 \
      --run run-01=D:\\...\\run-01\\extracted \
      --run run-02=D:\\...\\run-02\\extracted \
      --run run-03=D:\\...\\run-03\\extracted \
      --out-dir D:\\...\\aggregate

Use --start-at 4 with run-04 onward after an interrupted capture batch. The
output directory must exist and be empty. A source line groups potentially
many instructions; only an RVA names an exact instruction location here.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PINNED_IDENTITY = {
    "exe_sha256": "c4b24b2c5b16add984a2dd6d32e5a7080037290e58b254aff838a0dcf26d63c1",
    "pdb_sha256": "085c7c7cc7f724ed7674797dc88bf27011e6662d27d7661e9758677004bd8f9c",
    "pdb_guid": "8a90f52a-43f5-cf3e-4c4c-44205044422e",
    "pdb_age": 1,
    "pe_size_of_image": 0x1DA000,
}
RUN_ID = re.compile(r"run-(\d{2})\Z")
LOSS_ROW = re.compile(r"^[ \t]*(Total[ \t]+)?#[ \t]*Lost[ \t]+(Buffers|Events)[ \t]*:[ \t]*(\d+)[ \t]*$", re.I)
LOSS_LABEL = re.compile(r"(?:Total[ \t]+)?#[ \t]*Lost[ \t]+(?:Buffers|Events)", re.I)
MAP_FIELDS = {"pid", "ip", "module_base", "rva", "raw_count", "sample_rows", "symbol", "file", "line", "symbol_ok", "line_ok"}
SAMPLE_FIELDS = {"pid", "ip", "module_base", "rva", "raw_count", "symbol", "file", "line", "symbol_ok", "line_ok"}


class AggregateError(ValueError):
    """An input or evidence contract failed."""


@dataclass(frozen=True)
class IpRow:
    run: str
    pid: int
    ip: int
    base: int
    rva: int
    raw_count: int
    sample_rows: int
    symbol: str
    file: str
    line: int
    symbol_ok: int
    line_ok: int

    @property
    def source_key(self) -> tuple[str, str, int]:
        return self.symbol, self.file, self.line

    @property
    def mapping(self) -> tuple[str, str, int, int, int]:
        return self.symbol, self.file, self.line, self.symbol_ok, self.line_ok


@dataclass(frozen=True)
class Run:
    name: str
    sample_rows: int
    direct_count: int
    unique_ips: int
    rows: tuple[IpRow, ...]


def required_object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise AggregateError(f"{label} must be an object")
    return value


def nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise AggregateError(f"{label} is not an integer: {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise AggregateError(f"{label} is not an integer: {value!r}") from error
    if number < 0 or str(value).strip() != str(number):
        raise AggregateError(f"{label} is not a nonnegative decimal integer: {value!r}")
    return number


def hex_int(value: object, label: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"0x[0-9a-fA-F]+", value):
        raise AggregateError(f"{label} is not a hexadecimal address: {value!r}")
    return int(value, 16)


def read_csv(path: Path, fields: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise AggregateError(f"missing extraction file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if len(header) != len(set(header)) or not fields.issubset(header):
            raise AggregateError(f"missing or duplicate CSV headers in {path}: {sorted(fields - set(header))}")
        rows = list(reader)
    if not rows or any(None in row.values() or None in row for row in rows):
        raise AggregateError(f"empty or malformed CSV rows: {path}")
    return rows


def assert_zero_loss(path: Path) -> None:
    if not path.is_file():
        raise AggregateError(f"missing xperf tracestats: {path}")
    totals = collections.Counter()
    traces = collections.Counter()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not LOSS_LABEL.search(line):
            continue
        match = LOSS_ROW.fullmatch(line)
        if match is None:
            raise AggregateError(f"unparseable loss row in {path}: {line!r}")
        kind = match.group(2).lower()
        (totals if match.group(1) else traces)[kind] += 1
        if int(match.group(3)) != 0:
            raise AggregateError(f"nonzero xperf loss in {path}: {line!r}")
    if totals != {"buffers": 1, "events": 1} or traces["buffers"] < 1 or traces["buffers"] != traces["events"]:
        raise AggregateError(f"missing or duplicate xperf loss rows in {path}: total={dict(totals)} trace={dict(traces)}")


def load_run(name: str, directory: Path) -> Run:
    if not directory.is_dir():
        raise AggregateError(f"missing extracted run directory for {name}: {directory}")
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise AggregateError(f"missing extraction file: {summary_path}")
    try:
        summary = required_object(json.loads(summary_path.read_text(encoding="utf-8-sig")), str(summary_path))
        pairing = required_object(summary["pdb_pairing"], "pdb_pairing")
        pe = required_object(summary["pe"], "pe")
        counts = required_object(summary["counts"], "counts")
        xperf = required_object(summary["xperf"], "xperf")
        reports = required_object(xperf["reports"], "xperf.reports")
        tracestats = required_object(reports["tracestats"], "xperf.reports.tracestats")
        dumper = required_object(xperf["dumper"], "xperf.dumper")
    except (KeyError, json.JSONDecodeError) as error:
        raise AggregateError(f"incomplete or invalid summary for {name}: {error}") from error
    identity = {
        "exe_sha256": str(summary.get("exe_sha256", "")).lower(),
        "pdb_sha256": str(summary.get("pdb_sha256", "")).lower(),
        "pdb_guid": str(pairing.get("guid", "")).lower(),
        "pdb_age": nonnegative_int(pairing.get("age"), "pdb age"),
        "pe_size_of_image": hex_int(pe.get("size_of_image"), "PE image size"),
    }
    if identity != PINNED_IDENTITY or pairing.get("embedded_pdb_name", "").lower() != "rvp9enc.pdb":
        raise AggregateError(f"{name} PE/PDB identity differs from pinned 48cab42d build: {identity}")
    if summary.get("source_mode") != "etl" or dumper.get("exit_code") != 0 or tracestats.get("exit_code") != 0 or tracestats.get("report") != "xperf-tracestats.txt":
        raise AggregateError(f"{name} is not a successful ETL extraction with xperf tracestats")
    assert_zero_loss(directory / "xperf-tracestats.txt")

    sample_csv = read_csv(directory / "samples.csv", SAMPLE_FIELDS)
    map_csv = read_csv(directory / "ip-symbols.csv", MAP_FIELDS)
    indexed: dict[tuple[int, int], IpRow] = {}
    for line_number, row in enumerate(map_csv, 2):
        label = f"{name} ip-symbols.csv:{line_number}"
        parsed = IpRow(
            name, nonnegative_int(row["pid"], label + " pid"), hex_int(row["ip"], label + " ip"),
            hex_int(row["module_base"], label + " module_base"), hex_int(row["rva"], label + " rva"),
            nonnegative_int(row["raw_count"], label + " raw_count"),
            nonnegative_int(row["sample_rows"], label + " sample_rows"), row["symbol"], row["file"],
            nonnegative_int(row["line"], label + " line"),
            nonnegative_int(row["symbol_ok"], label + " symbol_ok"),
            nonnegative_int(row["line_ok"], label + " line_ok"),
        )
        if parsed.raw_count == 0 or parsed.sample_rows == 0 or parsed.rva >= PINNED_IDENTITY["pe_size_of_image"] or parsed.ip - parsed.base != parsed.rva:
            raise AggregateError(f"invalid mapped IP, RVA or count at {label}")
        if parsed.symbol_ok not in (0, 1) or parsed.line_ok not in (0, 1):
            raise AggregateError(f"invalid symbol/line status at {label}")
        key = parsed.pid, parsed.ip
        if key in indexed:
            raise AggregateError(f"duplicate PID/full IP at {label}")
        indexed[key] = parsed

    observed: dict[tuple[int, int], list[int]] = collections.defaultdict(lambda: [0, 0])
    for line_number, row in enumerate(sample_csv, 2):
        label = f"{name} samples.csv:{line_number}"
        key = nonnegative_int(row["pid"], label + " pid"), hex_int(row["ip"], label + " ip")
        mapped = indexed.get(key)
        if mapped is None:
            raise AggregateError(f"sample has no full-IP mapping at {label}")
        if (hex_int(row["module_base"], label + " module_base"), hex_int(row["rva"], label + " rva"),
            row["symbol"], row["file"], nonnegative_int(row["line"], label + " line"),
            nonnegative_int(row["symbol_ok"], label + " symbol_ok"),
            nonnegative_int(row["line_ok"], label + " line_ok")) != (
            mapped.base, mapped.rva, *mapped.mapping
        ):
            raise AggregateError(f"sample/full-IP mapping differs at {label}")
        count = nonnegative_int(row["raw_count"], label + " raw_count")
        if count == 0:
            raise AggregateError(f"zero direct sample count at {label}")
        observed[key][0] += count
        observed[key][1] += 1
    for key, mapped in indexed.items():
        if observed[key] != [mapped.raw_count, mapped.sample_rows]:
            raise AggregateError(f"sample/map counts differ for {name} PID/full IP {key}")
    direct_count = sum(row.raw_count for row in indexed.values())
    expected_counts = {
        "retained_target_executable_rows": len(sample_csv),
        "retained_direct_count": direct_count,
        "unique_ips": len({row.ip for row in indexed.values()}),
        "unique_pid_ip_pairs": len(indexed),
        "nonzero_source_line_direct_count": sum(row.raw_count for row in indexed.values() if row.line_ok),
    }
    for key, expected in expected_counts.items():
        if nonnegative_int(counts.get(key), f"{name} summary {key}") != expected:
            raise AggregateError(f"{name} summary {key} disagrees with CSV: expected {expected}")
    return Run(name, len(sample_csv), direct_count, expected_counts["unique_ips"],
               tuple(sorted(indexed.values(), key=lambda row: (row.pid, row.ip))))


def variability(per_run: dict[str, int], totals: dict[str, int]) -> tuple[float, float]:
    shares_pp = [100.0 * per_run.get(name, 0) / total for name, total in totals.items()]
    mean = sum(shares_pp) / len(shares_pp)
    return max(shares_pp) - min(shares_pp), sum((share - mean) ** 2 for share in shares_pp) / len(shares_pp)


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def aggregate(runs: list[Run], out_dir: Path) -> None:
    if not out_dir.is_dir() or out_dir.is_symlink() or any(out_dir.iterdir()):
        raise AggregateError(f"output directory must already exist, be plain, and be empty: {out_dir}")
    names = [run.name for run in runs]
    totals = {run.name: run.direct_count for run in runs}
    by_rva: dict[int, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    metadata: dict[int, tuple[str, str, int, int, int]] = {}
    full_ip_rows: list[list[object]] = []
    for run in runs:
        for row in run.rows:
            prior = metadata.setdefault(row.rva, row.mapping)
            if prior != row.mapping:
                raise AggregateError(f"same pinned RVA has inconsistent symbol/source mapping: 0x{row.rva:x}")
            by_rva[row.rva][run.name] += row.raw_count
            full_ip_rows.append([run.name, row.pid, f"0x{row.ip:016x}", f"0x{row.base:016x}",
                                 f"0x{row.rva:x}", row.raw_count, row.sample_rows, row.symbol,
                                 row.file, row.line, row.symbol_ok, row.line_ok])

    line_counts: dict[tuple[str, str, int], dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    line_rvas: dict[tuple[str, str, int], set[int]] = collections.defaultdict(set)
    rva_rows: list[list[object]] = []
    rva_variance: list[dict[str, object]] = []
    for rva in sorted(by_rva):
        symbol, file, line, symbol_ok, line_ok = metadata[rva]
        counts = by_rva[rva]
        spread, variance = variability(counts, totals)
        rva_rows.append([f"0x{rva:x}", symbol, file, line, symbol_ok, line_ok,
                         sum(counts.values()), f"{spread:.8f}", f"{variance:.8f}",
                         *(counts.get(name, 0) for name in names)])
        rva_variance.append({"rva": f"0x{rva:x}", "symbol": symbol, "file": file, "line": line,
                             "total_direct_count": sum(counts.values()),
                             "per_run_direct_count": {name: counts.get(name, 0) for name in names},
                             "range_share_percentage_points": round(spread, 8),
                             "population_variance_share_pp_squared": round(variance, 8)})
        source_key = symbol, file, line
        line_rvas[source_key].add(rva)
        for name, count in counts.items():
            line_counts[source_key][name] += count

    line_rows: list[list[object]] = []
    line_variance: list[dict[str, object]] = []
    for symbol, file, line in sorted(line_counts):
        key = symbol, file, line
        counts = line_counts[key]
        spread, variance = variability(counts, totals)
        line_rows.append([symbol, file, line, len(line_rvas[key]), sum(counts.values()),
                          f"{spread:.8f}", f"{variance:.8f}", *(counts.get(name, 0) for name in names)])
        line_variance.append({"symbol": symbol, "file": file, "line": line,
                              "unique_rvas": len(line_rvas[key]), "total_direct_count": sum(counts.values()),
                              "per_run_direct_count": {name: counts.get(name, 0) for name in names},
                              "range_share_percentage_points": round(spread, 8),
                              "population_variance_share_pp_squared": round(variance, 8)})

    rva_variance.sort(key=lambda row: (-row["range_share_percentage_points"], -row["total_direct_count"], int(row["rva"], 16)))
    line_variance.sort(key=lambda row: (-row["range_share_percentage_points"], -row["total_direct_count"], row["symbol"], row["file"], row["line"]))
    write_csv(out_dir / "runs.csv", ["run", "sample_rows", "direct_count", "unique_full_ips", "unique_rvas"],
              [[run.name, run.sample_rows, run.direct_count, run.unique_ips, len({row.rva for row in run.rows})] for run in runs])
    write_csv(out_dir / "full-ip-counts.csv", ["run", "pid", "ip", "module_base", "rva", "raw_count", "sample_rows", "symbol", "file", "line", "symbol_ok", "line_ok"], full_ip_rows)
    write_csv(out_dir / "rva-counts.csv", ["rva", "symbol", "file", "line", "symbol_ok", "line_ok", "total_direct_count", "range_share_pp", "population_variance_share_pp_squared", *(f"{name}_direct_count" for name in names)], rva_rows)
    write_csv(out_dir / "source-lines.csv", ["symbol", "file", "line", "unique_rvas", "total_direct_count", "range_share_pp", "population_variance_share_pp_squared", *(f"{name}_direct_count" for name in names)], line_rows)
    summary = {
        "identity": PINNED_IDENTITY,
        "interpretation": "Full IP is run-specific; RVA is the exact ASLR-stable instruction location. Source lines are broad attribution groups, not operations or disassembled instructions. Variability is descriptive, not a significance test.",
        "run_ids": names,
        "runs": [{"run": run.name, "sample_rows": run.sample_rows, "direct_count": run.direct_count,
                  "unique_full_ips": run.unique_ips, "unique_rvas": len({row.rva for row in run.rows}),
                  "lost_buffers": 0, "lost_events": 0} for run in runs],
        "total_direct_count": sum(totals.values()),
        "unique_rvas": len(by_rva),
        "top_rva_variability": rva_variance[:20],
        "top_source_line_variability": line_variance[:20],
        "outputs": ["runs.csv", "full-ip-counts.csv", "rva-counts.csv", "source-lines.csv", "summary.json"],
    }
    with (out_dir / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="run-NN=EXTRACTED_DIR",
                        help="one extracted directory per captured run; repeat for all N runs")
    parser.add_argument("--expected-runs", type=int, required=True, metavar="N",
                        help="number of consecutive runs required")
    parser.add_argument("--start-at", type=int, default=1, metavar="N",
                        help="first run number, default 1 (for interrupted batches)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="existing empty directory for deterministic CSV/JSON outputs")
    args = parser.parse_args(argv)
    if not 1 <= args.expected_runs <= 99 or not 1 <= args.start_at <= 99 or args.start_at + args.expected_runs - 1 > 99:
        parser.error("run numbers must be in run-01 through run-99")
    declared: dict[str, Path] = {}
    for entry in args.run:
        name, sep, path = entry.partition("=")
        if not sep or not RUN_ID.fullmatch(name) or not path or name in declared:
            parser.error(f"invalid or duplicate --run {entry!r}; expected run-NN=EXTRACTED_DIR")
        declared[name] = Path(path)
    expected = [f"run-{number:02d}" for number in range(args.start_at, args.start_at + args.expected_runs)]
    if set(declared) != set(expected):
        raise AggregateError(f"missing or extra runs: expected {expected}, supplied {sorted(declared)}")
    if len({path.resolve() for path in declared.values()}) != len(declared):
        raise AggregateError("two run IDs refer to the same extracted directory")
    runs = [load_run(name, declared[name]) for name in expected]
    aggregate(runs, args.out_dir)
    print(f"aggregated_runs={len(runs)} total_direct_count={sum(run.direct_count for run in runs)} output={args.out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AggregateError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
