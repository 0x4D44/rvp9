#!/usr/bin/env python3
"""W6.2 helper: add `bit_depth: rvp9_core::BitDepth::Bd8` to every
EncoderConfig struct literal. Anchors on the `noise_sensitivity:` field,
which is unique to EncoderConfig (RateControlConfig has min/max_qindex
but NOT noise_sensitivity), so RC literals are not touched.

Idempotent: skips literals that already have a `bit_depth:` line in the
3 lines following the anchor.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# frame_loop.rs already has the field on the struct def + its own
# constructor; skip it (the agent committed those by hand).
SKIP = {"crates/rvp9-encoder/src/frame_loop.rs"}

# Internal files (rvp9-encoder/src/lib.rs internal tests) can use the
# crate path; everything else uses the fully-qualified rvp9_core path.
PAT = re.compile(r"^(?P<indent>[ \t]+)noise_sensitivity:[^\n]*,\n", re.MULTILINE)


def transform(text: str) -> tuple[str, int]:
    out = []
    last = 0
    count = 0
    for m in PAT.finditer(text):
        tail = text[m.end(): m.end() + 120]
        if re.search(r"^\s*bit_depth\s*:", tail, re.MULTILINE):
            continue
        out.append(text[last:m.end()])
        indent = m.group("indent")
        out.append(f"{indent}bit_depth: rvp9_core::BitDepth::Bd8,\n")
        last = m.end()
        count += 1
    out.append(text[last:])
    return "".join(out), count


def main() -> None:
    total = 0
    for path in ROOT.rglob("*.rs"):
        s = str(path).replace("\\", "/")
        if "third_party" in s or "/target/" in s:
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if rel in SKIP:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new_text, count = transform(text)
        if count:
            path.write_text(new_text, encoding="utf-8")
            print(f"  +{count}: {rel}")
            total += count
    print(f"total: {total}")


if __name__ == "__main__":
    main()
