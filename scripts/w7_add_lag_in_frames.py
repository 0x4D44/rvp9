#!/usr/bin/env python3
"""W7 helper: add `lag_in_frames: 0,` to every EncoderConfig struct literal.

Anchors on the EncoderConfig-unique `bit_depth: <…>::BitDepth::Bd<…>,`
pattern (added at ca69ff1 — W6.2). The `Bd` variant is exclusive to
EncoderConfig: the wire-format `UncompressedHeader` uses
`BitDepth::Eight|Ten|Twelve` (and `WireBitDepth::Eight|…`); the
rate-control / kernel sites don't have a `bit_depth` field at all. So
this anchor matches all and only the EncoderConfig literals we need
to touch.

Idempotent: skips literals that already have a `lag_in_frames:` line
in the 5 lines following the anchor.

The struct-def site at `frame_loop.rs` is committed by hand (the
`pub lag_in_frames: u32,` field and the `EncoderConfig::new`
constructor default), so it's listed in SKIP.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# frame_loop.rs has the field definition + constructor default added
# by hand; let the script touch its test-module literals only.
# (The struct def's `bit_depth: BitDepth,` is a type signature, not a
# value, so it won't match the anchor — see PAT below.)
SKIP: set[str] = set()

# EncoderConfig literals: `bit_depth: BitDepth::Bd8,` or
# `bit_depth: rvp9_core::BitDepth::Bd8,`. The `Bd` variant prefix is
# the EncoderConfig-unique signal — wire-format uses Eight/Ten/Twelve.
PAT = re.compile(
    r"^(?P<indent>[ \t]+)bit_depth:\s*(?:rvp9_core::)?BitDepth::Bd\w+\s*,\n",
    re.MULTILINE,
)


def transform(text: str) -> tuple[str, int]:
    out = []
    last = 0
    count = 0
    for m in PAT.finditer(text):
        # Skip if `lag_in_frames:` already follows within ~5 lines.
        tail = text[m.end(): m.end() + 200]
        if re.search(r"^\s*lag_in_frames\s*:", tail, re.MULTILINE):
            continue
        out.append(text[last:m.end()])
        indent = m.group("indent")
        out.append(f"{indent}lag_in_frames: 0,\n")
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
