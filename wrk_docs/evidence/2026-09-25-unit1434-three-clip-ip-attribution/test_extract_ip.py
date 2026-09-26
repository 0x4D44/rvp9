"""Focused xperf dumper parsing regressions for exact-image attribution."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("extract-ip.py")
SPEC = importlib.util.spec_from_file_location("extract_ip", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extract_ip = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extract_ip
SPEC.loader.exec_module(extract_ip)

PID = 51484
BASE = 0x00007FF654B00000
ENCODER_IP = BASE + 0x12650
ENCODER_ROW = (
    f"SampledProfile, 2987450, rvp9enc.exe ({PID}), 6224, "
    f"0x{ENCODER_IP:016x}, 16, rvp9enc.exe!0x{ENCODER_IP:016x}, "
    f"rvp9enc.exe!0x{ENCODER_IP:016x}, 1, Unbatched"
)
KERNEL_IP = 0xFFFFF803A136A2D9
KERNEL_ROW = (
    f"SampledProfile, 2987451, rvp9enc.exe ({PID}), 6224, "
    f"0x{KERNEL_IP:016x}, 16, rvp9enc.exe!0x{ENCODER_IP:016x}, "
    f'"Unknown"!0x{KERNEL_IP:016x}, 1, Unbatched'
)


class DumperParsingTests(unittest.TestCase):
    def parse(self, extra_row: str):
        image = extract_ip.ImageRange(PID, BASE, 0x211000, "fixture")
        return extract_ip.parse_dumper_rows(
            [ENCODER_ROW, extra_row], "rvp9enc.exe", PID, image
        )

    def test_malformed_quoted_kernel_module_is_not_encoder_ip(self):
        parsed = self.parse(KERNEL_ROW)
        self.assertEqual([sample.ip for sample in parsed.samples], [ENCODER_IP])
        self.assertEqual([sample.ip for sample in parsed.other_module_samples], [KERNEL_IP])
        self.assertEqual(parsed.sampled_profile_rows, 2)

    def test_malformed_quoted_encoder_module_stays_fatal(self):
        malformed = ENCODER_ROW.replace(
            f"rvp9enc.exe!0x{ENCODER_IP:016x}, 1, Unbatched",
            f'"rvp9enc.exe"!0x{ENCODER_IP:016x}, 1, Unbatched',
        )
        with self.assertRaisesRegex(extract_ip.ExtractError, "malformed target CSV"):
            self.parse(malformed)


if __name__ == "__main__":
    unittest.main()
