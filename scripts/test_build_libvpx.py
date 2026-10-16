"""Exit-status and artifact checks for build-libvpx.sh using Unix shell fixtures.

Run with Python 3 on macOS or Linux with Bash available at /bin/bash.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


BUILD_SCRIPT = Path(__file__).with_name("build-libvpx.sh")
EXPECTED_GIT_COMMANDS = [
    "rev-parse --show-toplevel",
    "describe --tags --exact-match",
]
EXPECTED_MAKE_COMMANDS = [
    "-j1 target=libs libvpx.a",
    "-j1 target=examples vpxenc vpxdec examples/vp9_spatial_svc_encoder",
]
UNIX_ARTIFACTS = (
    "libvpx.a",
    "vpxenc",
    "vpxdec",
    "examples/vp9_spatial_svc_encoder",
)


class BuildLibvpxScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="test-build-libvpx-"
        )
        self.addCleanup(self.temporary_directory.cleanup)
        self.fixture = Path(self.temporary_directory.name)

        self.bin_directory = self.fixture / "bin"
        self.bin_directory.mkdir()
        self.fake_repo = self.fixture / "fake-repo"
        self.libvpx_source = self.fake_repo / "third_party" / "libvpx"
        self.libvpx_source.mkdir(parents=True)
        (self.libvpx_source / ".git").mkdir()
        self.build_directory = self.fixture / "output" / "libvpx"
        self.child_tmpdir = self.fixture / "tmp"
        self.child_tmpdir.mkdir()

        self.git_log = self.fixture / "git.log"
        self.configure_log = self.fixture / "configure.log"
        self.make_log = self.fixture / "make.log"

        self._write_executable(
            self.bin_directory / "git",
            """#!/bin/sh
printf '%s\\n' "$*" >> "$FAKE_GIT_LOG"
case "$*" in
    'rev-parse --show-toplevel') printf '%s\\n' "$FAKE_GIT_ROOT" ;;
    'describe --tags --exact-match') printf 'v1.14.1\\n' ;;
    *) printf 'unexpected git command: %s\\n' "$*" >&2; exit 91 ;;
esac
""",
        )
        self._write_executable(
            self.bin_directory / "uname",
            """#!/bin/sh
case "${1-}" in
    -s) printf 'Darwin\\n' ;;
    -m) printf 'arm64\\n' ;;
    *) printf 'unexpected uname argument: %s\\n' "${1-}" >&2; exit 92 ;;
esac
""",
        )
        self._write_executable(
            self.bin_directory / "nproc",
            """#!/bin/sh
printf '1\\n'
""",
        )
        self._write_executable(
            self.bin_directory / "make",
            """#!/bin/sh
set -eu
printf '%s\\n' "$*" >> "$FAKE_MAKE_LOG"
case "$*" in
    '-j1 target=libs libvpx.a')
        printf 'synthetic library marker\\n' > libvpx.a
        ;;
    '-j1 target=examples vpxenc vpxdec examples/vp9_spatial_svc_encoder')
        if [ -n "${FAKE_MAKE_STATUS:-}" ]; then
            exit "$FAKE_MAKE_STATUS"
        fi
        mkdir -p examples
        for artifact in vpxenc vpxdec examples/vp9_spatial_svc_encoder; do
            printf 'synthetic executable marker\\n' > "$artifact"
        done
        ;;
    *)
        printf 'unexpected make command: %s\\n' "$*" >&2
        exit 93
        ;;
esac
""",
        )
        self._write_executable(
            self.libvpx_source / "configure",
            """#!/bin/sh
set -eu
printf '%s\\n' "$*" >> "$FAKE_CONFIGURE_LOG"
exit "${FAKE_CONFIGURE_STATUS:-0}"
""",
        )

    def _write_executable(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

    def _lines(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8").splitlines()

    def _run_script(
        self, *, configure_status: int = 0, make_status: int = 0
    ) -> subprocess.CompletedProcess[str]:
        env = {
            "PATH": f"{self.bin_directory}:/usr/bin:/bin:/usr/sbin:/sbin",
            "LC_ALL": "C",
            "TMPDIR": str(self.child_tmpdir),
            "RVP9_LIBVPX_BUILD_DIR": str(self.build_directory),
            "FAKE_GIT_ROOT": str(self.fake_repo),
            "FAKE_GIT_LOG": str(self.git_log),
            "FAKE_CONFIGURE_LOG": str(self.configure_log),
            "FAKE_CONFIGURE_STATUS": str(configure_status),
            "FAKE_MAKE_LOG": str(self.make_log),
            "FAKE_MAKE_STATUS": str(make_status) if make_status else "",
        }
        return subprocess.run(
            ["/bin/bash", str(BUILD_SCRIPT)],
            cwd=self.fake_repo,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )

    def _result_context(self, result: subprocess.CompletedProcess[str]) -> str:
        return (
            f"exit status: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            f"git calls: {self._lines(self.git_log)!r}\n"
            f"configure calls: {self._lines(self.configure_log)!r}\n"
            f"make calls: {self._lines(self.make_log)!r}"
        )

    def test_successful_build_reports_unix_artifacts(self) -> None:
        result = self._run_script()

        for artifact in UNIX_ARTIFACTS:
            artifact_path = self.build_directory / artifact
            self.assertTrue(artifact_path.is_file(), artifact)
            self.assertGreater(artifact_path.stat().st_size, 0, artifact)
        self.assertIn("Build complete. Artifacts:", result.stdout)
        for artifact in UNIX_ARTIFACTS[1:]:
            self.assertIn(artifact, result.stdout)
        self.assertNotIn(".exe", result.stdout)
        self.assertEqual(self._lines(self.git_log), EXPECTED_GIT_COMMANDS)
        configure_calls = self._lines(self.configure_log)
        self.assertEqual(len(configure_calls), 1)
        self.assertIn("--target=arm64-darwin23-gcc", configure_calls[0])
        self.assertEqual(self._lines(self.make_log), EXPECTED_MAKE_COMMANDS)
        self.assertFalse((self.build_directory / "vpxenc.exe").exists())
        self.assertFalse((self.build_directory / "vpxdec.exe").exists())
        self.assertFalse(
            (self.build_directory / "examples/vp9_spatial_svc_encoder.exe").exists()
        )
        self.assertEqual(result.returncode, 0, self._result_context(result))

    def test_configure_failure_preserves_exit_status(self) -> None:
        result = self._run_script(configure_status=17)

        self.assertEqual(result.returncode, 17, self._result_context(result))
        self.assertEqual(self._lines(self.git_log), EXPECTED_GIT_COMMANDS)
        self.assertEqual(len(self._lines(self.configure_log)), 1)
        self.assertEqual(self._lines(self.make_log), [])
        self.assertNotIn("Build complete. Artifacts:", result.stdout)

    def test_make_failure_preserves_exit_status(self) -> None:
        result = self._run_script(make_status=23)

        self.assertEqual(result.returncode, 23, self._result_context(result))
        self.assertEqual(self._lines(self.git_log), EXPECTED_GIT_COMMANDS)
        self.assertTrue((self.build_directory / "libvpx.a").is_file())
        self.assertEqual(self._lines(self.make_log), EXPECTED_MAKE_COMMANDS)
        self.assertNotIn("Build complete. Artifacts:", result.stdout)
        self.assertFalse((self.build_directory / "vpxenc").exists())


if __name__ == "__main__":
    unittest.main()
