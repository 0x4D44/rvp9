"""Regression checks for decoded-file size handling in qualification scripts.

The tests execute the production ``run_row`` and ``file_size`` functions with
fixture codec commands and a fake ``stat``. They do not build or run codecs.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHA_SCRIPT = REPO_ROOT / "scripts/qualify-alpha-encoder.sh"
REFERENCE_SCRIPT = REPO_ROOT / "scripts/qualify-reference-decoder.sh"
ALPHA_BYTES = 1280 * 720 * 3 // 2
REFERENCE_BYTES = 128 * 64 * 3 // 2


def extract_function(script: Path, name: str) -> str | None:
    lines = script.read_text(encoding="utf-8").splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == f"{name}() {{"),
        None,
    )
    if start is None:
        return None
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index] == "}"),
        None,
    )
    if end is None:
        raise AssertionError(f"could not find the end of {name}() in {script}")
    return "\n".join(lines[start : end + 1])


class QualificationFileSizeTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = REPO_ROOT / ".scratch"
        scratch.mkdir(exist_ok=True)
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="qualification-file-size-", dir=scratch
        )
        self.addCleanup(temporary_directory.cleanup)
        self.fixture = Path(temporary_directory.name)
        self.bin_directory = self.fixture / "bin"
        self.bin_directory.mkdir()
        self.stat_log = self.fixture / "stat.log"
        self._write_executable(
            self.bin_directory / "stat",
            """#!/bin/sh
set -eu
printf '%s\\n' "$1" >> "$FAKE_STAT_LOG"
case "${FAKE_STAT_MODE}:$1" in
    bsd:-f)
        [ -z "${FAKE_STAT_VALUE-}" ] || printf '%s\\n' "$FAKE_STAT_VALUE"
        exit "${FAKE_STAT_F_STATUS:-0}"
        ;;
    bsd:-c)
        exit 93
        ;;
    gnu:-f)
        printf 'Filesystem root: /fixture\\n'
        exit 0
        ;;
    gnu:-c)
        [ -z "${FAKE_STAT_VALUE-}" ] || printf '%s\\n' "$FAKE_STAT_VALUE"
        exit "${FAKE_STAT_C_STATUS:-0}"
        ;;
    failure:-f|failure:-c)
        exit 1
        ;;
    empty:-f|empty:-c)
        exit 0
        ;;
    nonnumeric:-f|nonnumeric:-c)
        printf 'not-a-byte-count\\n'
        exit 0
        ;;
    *)
        printf 'unexpected stat mode/format: %s %s\\n' "$FAKE_STAT_MODE" "$1" >&2
        exit 94
        ;;
esac
""",
        )
        self._write_executable(
            self.bin_directory / "fake-codec",
            """#!/bin/sh
set -eu
output=
last=
while [ "$#" -gt 0 ]; do
    if [ "$1" = -o ]; then
        shift
        output=$1
    fi
    last=$1
    shift
done
[ -n "$output" ] || output=$last
printf 'fixture output\\n' > "$output"
""",
        )

    @staticmethod
    def _write_executable(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    def _run_row(
        self,
        script: Path,
        *,
        mode: str,
        value: str = "",
        f_status: int = 0,
        c_status: int = 0,
    ) -> tuple[int, str, str, list[str]]:
        self.stat_log.unlink(missing_ok=True)
        (self.fixture / "qualification.txt").unlink(missing_ok=True)
        source_text = script.read_text(encoding="utf-8")
        self.assertIn(
            'source "$repo_root/scripts/qualification-file-size.sh"', source_text
        )
        run_row = extract_function(script, "run_row")
        self.assertIsNotNone(run_row, f"missing production run_row() in {script}")
        file_size = extract_function(script, "file_size")
        harness = [
            "set -euo pipefail",
            'repo_root="$TEST_REPO_ROOT"',
            'source "$repo_root/scripts/qualification-file-size.sh"',
        ]
        if file_size is not None:
            harness.append(file_size)

        harness.extend(
            [
                'work_dir="$TEST_FIXTURE"',
                'manifest="$TEST_FIXTURE/qualification.txt"',
                'rows=(probe)',
                'frames=1',
                'rvp9enc="$TEST_FIXTURE/bin/fake-codec"',
                'rvp9dec="$TEST_FIXTURE/bin/fake-codec"',
                'vpxdec="$TEST_FIXTURE/bin/fake-codec"',
                'vpxenc="$TEST_FIXTURE/bin/fake-codec"',
                'ffmpeg_bin="$TEST_FIXTURE/bin/fake-codec"',
                'source_for_key() { printf "%s\\n" "$TEST_FIXTURE/source.y4m"; }',
                'source_y4m="$TEST_FIXTURE/source.y4m"',
                "WIDTH=128",
                "HEIGHT=64",
                f"expected_bytes={REFERENCE_BYTES}",
                "target_bitrate=512",
                "kf_max_dist=12",
                "cleanup_rows=1",
                'sha256_file() { shasum -a 256 "$1" | awk \'{print $1}\'; }',
                run_row,
            ]
        )
        row = (
            "probe|720-in-tree|30/1|6|vbr|0|0|1|1500|1|0"
            if script == ALPHA_SCRIPT
            else "probe|6|cbr|0|0"
        )
        harness.append(
            f'if run_row 1 "{row}"; then printf "RUN_STATUS=0\\n"; '
            'else status=$?; printf "RUN_STATUS=%s\\n" "$status"; fi'
        )

        env = {
            "PATH": f"{self.bin_directory}:/usr/bin:/bin:/usr/sbin:/sbin",
            "LC_ALL": "C",
            "TEST_FIXTURE": str(self.fixture),
            "TEST_REPO_ROOT": str(REPO_ROOT),
            "FAKE_STAT_LOG": str(self.stat_log),
            "FAKE_STAT_MODE": mode,
            "FAKE_STAT_VALUE": value,
            "FAKE_STAT_F_STATUS": str(f_status),
            "FAKE_STAT_C_STATUS": str(c_status),
        }
        result = subprocess.run(
            ["/bin/bash", "-c", "\n".join(harness)],
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        manifest_path = self.fixture / "qualification.txt"
        manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""
        stat_calls = self.stat_log.read_text(encoding="utf-8").splitlines() if self.stat_log.exists() else []
        match = re.search(r"^RUN_STATUS=(\d+)$", result.stdout, re.MULTILINE)
        self.assertIsNotNone(
            match,
            f"run_row harness did not report a status; stdout={result.stdout!r}, stderr={result.stderr!r}",
        )
        return int(match.group(1)), manifest, result.stderr, stat_calls

    def test_missing_or_malformed_size_never_emits_pass(self) -> None:
        for script in (ALPHA_SCRIPT, REFERENCE_SCRIPT):
            for mode in ("failure", "empty", "nonnumeric"):
                with self.subTest(script=script.name, stat_mode=mode):
                    status, manifest, _stderr, _calls = self._run_row(
                        script, mode=mode
                    )
                    self.assertNotEqual(
                        status,
                        0,
                        f"{script.name} accepted missing/nonnumeric decoded size ({mode})",
                    )
                    self.assertNotIn("|PASS|", manifest)

    def test_bsd_stat_size_is_accepted_exactly(self) -> None:
        for script, expected in (
            (ALPHA_SCRIPT, ALPHA_BYTES),
            (REFERENCE_SCRIPT, REFERENCE_BYTES),
        ):
            with self.subTest(script=script.name):
                status, manifest, _stderr, calls = self._run_row(
                    script, mode="bsd", value=str(expected)
                )
                self.assertEqual(status, 0)
                self.assertIn("|PASS|", manifest)
                expected_calls = 3 if script == ALPHA_SCRIPT else 2
                self.assertEqual(calls, ["-f"] * expected_calls)

    def test_leading_zero_size_is_canonicalized_before_exact_comparison(self) -> None:
        for script, expected in (
            (ALPHA_SCRIPT, ALPHA_BYTES),
            (REFERENCE_SCRIPT, REFERENCE_BYTES),
        ):
            with self.subTest(script=script.name):
                status, manifest, _stderr, _calls = self._run_row(
                    script, mode="bsd", value=f"000{expected}"
                )
                self.assertEqual(status, 0)
                self.assertIn("|PASS|", manifest)

    def test_gnu_stat_filesystem_text_falls_back_to_exact_byte_count(self) -> None:
        for script, expected in (
            (ALPHA_SCRIPT, ALPHA_BYTES),
            (REFERENCE_SCRIPT, REFERENCE_BYTES),
        ):
            with self.subTest(script=script.name):
                status, manifest, _stderr, calls = self._run_row(
                    script, mode="gnu", value=str(expected)
                )
                self.assertEqual(status, 0)
                self.assertIn("|PASS|", manifest)
                expected_calls = 3 if script == ALPHA_SCRIPT else 2
                self.assertEqual(calls, ["-f", "-c"] * expected_calls)

    def test_numeric_stdout_with_failed_stat_status_is_rejected(self) -> None:
        for script, expected in (
            (ALPHA_SCRIPT, ALPHA_BYTES),
            (REFERENCE_SCRIPT, REFERENCE_BYTES),
        ):
            for mode, kwargs in (
                ("bsd", {"f_status": 1}),
                ("gnu", {"c_status": 1}),
            ):
                with self.subTest(script=script.name, stat_mode=mode):
                    status, manifest, _stderr, _calls = self._run_row(
                        script, mode=mode, value=str(expected), **kwargs
                    )
                    self.assertNotEqual(status, 0)
                    self.assertNotIn("|PASS|", manifest)

    def test_mismatched_or_oversized_decimal_size_is_rejected(self) -> None:
        values = ("17", "999999999999999999999999999999999999999999999999999999999999")
        for script in (ALPHA_SCRIPT, REFERENCE_SCRIPT):
            for value in values:
                with self.subTest(script=script.name, value=value):
                    status, manifest, stderr, _calls = self._run_row(
                        script, mode="bsd", value=value
                    )
                    self.assertNotEqual(status, 0)
                    self.assertNotIn("|PASS|", manifest)
                    self.assertNotIn("integer expression expected", stderr)


if __name__ == "__main__":
    unittest.main()
