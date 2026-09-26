"""Small, independent fixtures for the repeated-Ducks IP aggregator."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).with_name("aggregate-samples.py")
HEADER = ["pid", "ip", "module_base", "rva", "raw_count", "sample_rows", "symbol", "file", "line", "symbol_ok", "line_ok"]
EXE_SHA256 = "c4b24b2c5b16add984a2dd6d32e5a7080037290e58b254aff838a0dcf26d63c1"
PDB_SHA256 = "085c7c7cc7f724ed7674797dc88bf27011e6662d27d7661e9758677004bd8f9c"
PDB_GUID = "8a90f52a-43f5-cf3e-4c4c-44205044422e"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_run(root: Path, name: str, base: int, samples: list[tuple[int, int]]) -> Path:
    directory = root / name / "extracted"
    directory.mkdir(parents=True)
    pid = 42
    sample_rows = []
    for rva, count in samples:
        sample_rows.append({"pid": pid, "ip": f"0x{base + rva:016x}",
                            "module_base": f"0x{base:016x}", "rva": f"0x{rva:x}",
                            "raw_count": count, "sample_rows": 1, "symbol": "kernel",
                            "file": "D:\\src\\kernel.rs", "line": 42, "symbol_ok": 1, "line_ok": 1})
    grouped = Counter(rva for rva, _ in samples)
    counts = Counter()
    for rva, count in samples:
        counts[rva] += count
    ip_map = []
    for rva in sorted(grouped):
        row = next(row for row in sample_rows if row["rva"] == f"0x{rva:x}").copy()
        row["raw_count"] = counts[rva]
        row["sample_rows"] = grouped[rva]
        ip_map.append(row)
    write_csv(directory / "samples.csv", sample_rows)
    write_csv(directory / "ip-symbols.csv", ip_map)
    (directory / "xperf-tracestats.txt").write_text(
        "Total # Lost Buffers : 0\nTotal # Lost Events  : 0\n"
        "    # Lost Buffers       : 0\n    # Lost Events        : 0\n", encoding="utf-8"
    )
    summary = {
        "source_mode": "etl", "exe_sha256": EXE_SHA256, "pdb_sha256": PDB_SHA256,
        "pdb_pairing": {"embedded_pdb_name": "rvp9enc.pdb", "guid": PDB_GUID, "age": 1},
        "pe": {"size_of_image": "0x1da000"},
        "counts": {"retained_target_executable_rows": len(samples),
                   "retained_direct_count": sum(count for _, count in samples),
                   "unique_ips": len(grouped), "unique_pid_ip_pairs": len(grouped),
                   "nonzero_source_line_direct_count": sum(count for _, count in samples)},
        "xperf": {"dumper": {"exit_code": 0},
                  "reports": {"tracestats": {"exit_code": 0, "report": "xperf-tracestats.txt"}}},
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return directory


class AggregateSamplesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.first = make_run(self.root, "run-01", 0x100000, [(0x10, 2), (0x10, 3), (0x20, 1)])
        self.second = make_run(self.root, "run-02", 0x200000, [(0x10, 4), (0x20, 2)])
        self.out = self.root / "out"
        self.out.mkdir()

    def invoke(self, *runs: tuple[str, Path], out: Path | None = None) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(SCRIPT), "--expected-runs", "2", "--out-dir", str(out or self.out)]
        for name, path in runs:
            command += ["--run", f"{name}={path}"]
        return subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)

    def test_duplicate_full_ip_aggregates_by_rva_without_merging_operations(self) -> None:
        result = self.invoke(("run-01", self.first), ("run-02", self.second))
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.out / "rva-counts.csv").open(newline="", encoding="utf-8") as stream:
            rows = {row["rva"]: row for row in csv.DictReader(stream)}
        self.assertEqual(set(rows), {"0x10", "0x20"})
        self.assertEqual(rows["0x10"]["total_direct_count"], "9")
        self.assertEqual(rows["0x10"]["run-01_direct_count"], "5")
        self.assertEqual(rows["0x10"]["run-02_direct_count"], "4")
        with (self.out / "full-ip-counts.csv").open(newline="", encoding="utf-8") as stream:
            ip_rows = list(csv.DictReader(stream))
        self.assertEqual(len(ip_rows), 4)
        self.assertEqual(ip_rows[0]["sample_rows"], "2")
        self.assertNotEqual(ip_rows[0]["ip"], ip_rows[2]["ip"])
        with (self.out / "source-lines.csv").open(newline="", encoding="utf-8") as stream:
            line_rows = list(csv.DictReader(stream))
        self.assertEqual(len(line_rows), 1)
        self.assertEqual(line_rows[0]["unique_rvas"], "2")
        self.assertEqual(line_rows[0]["total_direct_count"], "12")
        summary = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["run_ids"], ["run-01", "run-02"])
        self.assertEqual(summary["runs"][0]["direct_count"], 6)
        self.assertEqual(summary["runs"][1]["direct_count"], 6)
        self.assertEqual(summary["top_rva_variability"][0]["rva"], "0x10")
        self.assertEqual(summary["top_rva_variability"][0]["range_share_percentage_points"], 16.66666667)
        self.assertIn("not operations", summary["interpretation"])
        other_out = self.root / "other-out"
        other_out.mkdir()
        reversed_result = self.invoke(("run-02", self.second), ("run-01", self.first), out=other_out)
        self.assertEqual(reversed_result.returncode, 0, reversed_result.stderr)
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.out.iterdir()},
            {path.name: path.read_bytes() for path in other_out.iterdir()},
        )

    def test_rejects_mismatched_pdb_identity(self) -> None:
        path = self.second / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["pdb_sha256"] = "0" * 64
        path.write_text(json.dumps(summary), encoding="utf-8")
        result = self.invoke(("run-01", self.first), ("run-02", self.second))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PE/PDB identity differs", result.stderr)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_rejects_missing_declared_run(self) -> None:
        result = self.invoke(("run-01", self.first))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing or extra runs", result.stderr)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_rejects_nonzero_xperf_loss(self) -> None:
        stats = self.second / "xperf-tracestats.txt"
        stats.write_text(stats.read_text(encoding="utf-8").replace("Total # Lost Events  : 0", "Total # Lost Events  : 1"), encoding="utf-8")
        result = self.invoke(("run-01", self.first), ("run-02", self.second))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nonzero xperf loss", result.stderr)
        self.assertEqual(list(self.out.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
