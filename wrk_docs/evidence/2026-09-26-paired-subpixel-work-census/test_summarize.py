"""Synthetic, end-to-end checks for the paired subpixel census summarizer."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("summarize.py")
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
INPUT_HASHES = {
    "ducks": "6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7",
    "tree": "ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791",
    "town": "b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909",
}
HEADERS = {
    "rvp9": "subpel_census,family,width,height,xoff,yoff,calls,pixels",
    "libvpx": "LIBVPX_SVF_CENSUS,shape,x_phase,y_phase,calls,pixels",
}
ROWS = {
    "rvp9": {
        "ducks": [
            "subpel_census,full_tree,8,8,0,0,2,128",
            "subpel_census,full_tree,8,8,0,2,3,192",
            "subpel_census,full_tree,8,8,2,2,1,64",
        ],
        "tree": ["subpel_census,full_tree,16,16,4,0,2,512"],
        "town": ["subpel_census,full_tree,8,8,0,4,1,64"],
    },
    "libvpx": {
        "ducks": [
            "LIBVPX_SVF_CENSUS,8x8,0,2,5,320",
            "LIBVPX_SVF_CENSUS,8x8,2,2,7,448",
        ],
        "tree": ["LIBVPX_SVF_CENSUS,16x16,4,0,3,768"],
        "town": ["LIBVPX_SVF_CENSUS,8x8,0,4,4,256"],
    },
}
ENTRY_SEARCHES = {"ducks": 10, "tree": 2, "town": 3}
RVP9_SEARCHES = {"ducks": (8, 8, 2), "tree": (16, 16, 2), "town": (8, 8, 1)}


class SummarizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        for encoder in ("rvp9", "libvpx"):
            directory = self.root / encoder
            directory.mkdir()
            with (directory / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Clip", "InputSha256", "OutputSha256", "BinarySha256", "CensusRows"])
                for clip in ("ducks", "tree", "town"):
                    writer.writerow([clip, INPUT_HASHES[clip], OUTPUT_HASHES[encoder][clip],
                                     "f" * 64, len(ROWS[encoder][clip])])
            for clip in ("ducks", "tree", "town"):
                (directory / f"{clip}.stderr.txt").write_text(
                    "unrelated log line\n" + HEADERS[encoder] + "\n"
                    + "\n".join(ROWS[encoder][clip]) + "\n", encoding="utf-8"
                )
        entry_dir = self.root / "libvpx-entry"
        entry_dir.mkdir()
        with (entry_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Clip", "InputSha256", "OutputSha256", "BinarySha256",
                             "CensusRows", "Searches"])
            for clip in ("ducks", "tree", "town"):
                writer.writerow([clip, INPUT_HASHES[clip], OUTPUT_HASHES["libvpx"][clip],
                                 "e" * 64, len(ROWS["libvpx"][clip]), ENTRY_SEARCHES[clip]])
                (entry_dir / f"{clip}.stderr.txt").write_text(
                    f"LIBVPX_SUBPEL_SEARCHES,total,{ENTRY_SEARCHES[clip]}\n"
                    + HEADERS["libvpx"] + "\n" + "\n".join(ROWS["libvpx"][clip]) + "\n",
                    encoding="utf-8",
                )
        rvp9_entry = self.root / "rvp9-entry"
        rvp9_entry.mkdir()
        with (rvp9_entry / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Clip", "InputSha256", "OutputSha256", "BinarySha256", "CensusRows"])
            for clip in ("ducks", "tree", "town"):
                writer.writerow([clip, INPUT_HASHES[clip], OUTPUT_HASHES["rvp9"][clip],
                                 "d" * 64, len(ROWS["rvp9"][clip])])
                width, height, searches = RVP9_SEARCHES[clip]
                (rvp9_entry / f"{clip}.stderr.txt").write_text(
                    HEADERS["rvp9"] + "\n" + "\n".join(ROWS["rvp9"][clip]) + "\n"
                    + "subpel_searches,family,width,height,searches\n"
                    + f"subpel_searches,full_tree,{width},{height},{searches}\n",
                    encoding="utf-8",
                )

    def run_summary(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--input-dir", str(self.root),
             "--out-dir", str(self.output)],
            capture_output=True, text=True, timeout=10, check=False,
        )

    def assert_invalid(self, expected_error: str) -> None:
        (self.output / "validation.json").unlink(missing_ok=True)
        result = self.run_summary()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse((self.output / "summary.csv").exists())
        report = json.loads((self.output / "validation.json").read_text(encoding="utf-8"))
        self.assertFalse(report["valid"])
        self.assertIn(expected_error, " ".join(report["errors"]))

    def test_center_is_retained_but_excluded_from_fractional_total(self) -> None:
        result = self.run_summary()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.output / "validation.json").read_text(encoding="utf-8"))
        self.assertTrue(report["valid"])
        self.assertEqual(report["clips"]["ducks"]["libvpx"]["output_sha256"],
                         OUTPUT_HASHES["libvpx"]["ducks"])
        self.assertIn("not like-for-like", report["comparison_note"])
        self.assertIn("not rehashed", report["hash_check_note"])
        with (self.output / "summary.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_key = {(r["clip"], r["encoder"], r["shape"], r["phase_class"]): r for r in rows}
        self.assertEqual(by_key[("ducks", "rvp9", "8x8", "center")]["calls"], "2")
        self.assertEqual(by_key[("ducks", "rvp9", "8x8", "one_axis")]["pixels"], "192")
        self.assertEqual(by_key[("ducks", "rvp9", "8x8", "one_axis")]["searches"], "2")
        self.assertEqual(by_key[("ducks", "rvp9", "8x8", "one_axis")]
                         ["fractional_probes_per_search"], "1.500000")
        self.assertEqual(by_key[("ducks", "rvp9", "8x8", "fractional_total")]["calls"], "4")
        self.assertEqual(by_key[("ducks", "rvp9", "8x8", "fractional_total")]
                         ["fractional_probes_per_search"], "2.000000")
        self.assertEqual(by_key[("ducks", "rvp9", "ALL", "fractional_total")]["calls"], "4")
        self.assertEqual(by_key[("ducks", "rvp9", "ALL", "fractional_total")]["searches"], "2")
        self.assertEqual(by_key[("ducks", "rvp9", "ALL", "fractional_total")]
                         ["fractional_probes_per_search"], "2.000000")
        self.assertEqual(report["clips"]["ducks"]["rvp9"]["searches"], 2)
        self.assertTrue(report["clips"]["ducks"]["rvp9"]["svf_rows_match_original"])
        self.assertEqual(by_key[("ducks", "rvp9", "ALL", "all_recorded_total")]["calls"], "6")
        self.assertEqual(by_key[("ducks", "libvpx", "ALL", "fractional_total")]["calls"], "12")
        self.assertEqual(by_key[("ducks", "libvpx", "ALL", "fractional_total")]["pixels"], "768")
        self.assertEqual(by_key[("ducks", "libvpx", "ALL", "fractional_total")]["searches"], "10")
        self.assertEqual(by_key[("ducks", "libvpx", "ALL", "fractional_total")]
                         ["fractional_probes_per_search"], "1.200000")
        self.assertEqual(report["clips"]["ducks"]["libvpx"]["searches"], 10)
        self.assertTrue(report["clips"]["ducks"]["libvpx"]["svf_rows_match_original"])
        self.assertEqual(
            by_key[("ducks", "libvpx", "ALL", "fractional_total")]["comparison_scope"],
            "descriptive_not_like_for_like_time_or_search_policy",
        )
        self.assertNotIn(("ducks", "libvpx", "8x8", "center"), by_key)

    def test_duplicate_key_fails_even_if_manifest_count_is_unchanged(self) -> None:
        path = self.root / "libvpx" / "ducks.stderr.txt"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace(ROWS["libvpx"]["ducks"][1], ROWS["libvpx"]["ducks"][0]),
                        encoding="utf-8")
        self.assert_invalid("duplicate key")

    def test_bad_phase_and_pixel_count_fail(self) -> None:
        path = self.root / "rvp9" / "ducks.stderr.txt"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace(",0,2,3,192", ",0,8,3,192"), encoding="utf-8")
        self.assert_invalid("phase")

        path.write_text(text.replace(",0,2,3,192", ",0,2,3,193"), encoding="utf-8")
        self.assert_invalid("pixels")

    def test_zero_calls_and_wrong_input_hash_fail(self) -> None:
        path = self.root / "libvpx" / "ducks.stderr.txt"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace(",0,2,5,320", ",0,2,0,0"), encoding="utf-8")
        self.assert_invalid("calls must be positive")

        path.write_text(text, encoding="utf-8")
        manifest = self.root / "libvpx" / "manifest.csv"
        original = manifest.read_text(encoding="utf-8-sig")
        manifest.write_text(original.replace(INPUT_HASHES["ducks"], "0" * 64),
                            encoding="utf-8-sig")
        self.assert_invalid("input hash")

    def test_invalid_row_and_schema_fail(self) -> None:
        path = self.root / "libvpx" / "ducks.stderr.txt"
        text = path.read_text(encoding="utf-8")
        path.write_text(text + "LIBVPX_SVF_CENSUS_INVALID,8x8,1\n", encoding="utf-8")
        self.assert_invalid("invalid row")

        path.write_text(text.replace(HEADERS["libvpx"], "LIBVPX_SVF_CENSUS,shape,x_phase"),
                        encoding="utf-8")
        self.assert_invalid("schema")

    def test_manifest_output_hash_and_row_count_are_checked(self) -> None:
        path = self.root / "rvp9" / "manifest.csv"
        text = path.read_text(encoding="utf-8-sig")
        path.write_text(text.replace(OUTPUT_HASHES["rvp9"]["ducks"], "0" * 64),
                        encoding="utf-8-sig")
        self.assert_invalid("output hash")

        path.write_text(text.replace(",3\n", ",9\n", 1), encoding="utf-8-sig")
        self.assert_invalid("CensusRows")

    def test_entry_capture_must_match_original_svf_rows_and_search_count(self) -> None:
        path = self.root / "libvpx-entry" / "ducks.stderr.txt"
        original = path.read_text(encoding="utf-8")
        path.write_text(original.replace(",0,2,5,320", ",0,2,6,384"), encoding="utf-8")
        self.assert_invalid("SVF rows differ")

        path.write_text(original.replace("SEARCHES,total,10", "SEARCHES,total,11"),
                        encoding="utf-8")
        self.assert_invalid("Searches")

    def test_rvp9_entry_rows_and_search_keys_are_validated(self) -> None:
        path = self.root / "rvp9-entry" / "ducks.stderr.txt"
        original = path.read_text(encoding="utf-8")
        path.write_text(original.replace(",0,2,3,192", ",0,2,4,256"), encoding="utf-8")
        self.assert_invalid("SVF rows differ")

        path.write_text(original + "subpel_searches,full_tree,8,8,2\n", encoding="utf-8")
        self.assert_invalid("duplicate search key")


if __name__ == "__main__":
    unittest.main()
