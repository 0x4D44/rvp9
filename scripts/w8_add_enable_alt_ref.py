#!/usr/bin/env python3
"""W8 helper: add `enable_alt_ref: false,` to every EncoderConfig literal.

Mirrors `scripts/w7_add_lag_in_frames.py`: anchors on the
EncoderConfig-unique `lag_in_frames: <N>,` line (added at eeae080 — W7).
That field is exclusive to EncoderConfig in the rvp9 source tree, so
this anchor matches all and only the literals we need to touch.

Idempotent: skips literals that already have an `enable_alt_ref:` line
within the 5 lines following the anchor.

The struct-def site at `frame_loop.rs` (the `pub enable_alt_ref: bool,`
field) and the `EncoderConfig::new` constructor default are committed
by hand.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP: set[str] = set()

# `lag_in_frames: 0,` or `lag_in_frames: LAG,` etc — anchor on the
# value expression by allowing any non-comma run after the colon.
PAT = re.compile(
    r"^(?P<indent>[ \t]+)lag_in_frames:\s*[^,\n]+,\n",
    re.MULTILINE,
)


def transform(text: str) -> tuple[str, int]:
    out = []
    last = 0
    count = 0
    for m in PAT.finditer(text):
        # Skip if `enable_alt_ref:` already follows within ~5 lines.
        tail = text[m.end(): m.end() + 200]
        if re.search(r"^\s*enable_alt_ref\s*:", tail, re.MULTILINE):
            continue
        out.append(text[last:m.end()])
        indent = m.group("indent")
        out.append(f"{indent}enable_alt_ref: false,\n")
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
