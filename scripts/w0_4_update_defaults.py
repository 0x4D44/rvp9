#!/usr/bin/env python3
"""W0.4 helper: update the speed/deadline defaults inserted by W0.2 to
match rvp9's actual op point (cpu-used=7, Realtime) instead of vpxenc's
default (cpu-used=0, Good). The W0.3 dispatcher's output at (7, Realtime)
reproduces the §11 hardcodes the dispatcher replaces; this keeps the
existing test suite bit-identical when W0.4 lands the wiring.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Patterns: handle both qualifier styles
#   - internal (rvp9-encoder/src/{frame_loop,lib}.rs): `Deadline::Good`
#   - external                                       : `rvp9_encoder::Deadline::Good`
# Match the exact two-line pair that W0.2 inserted, leaving the
# surrounding context untouched.
PATTERNS = [
    (
        re.compile(r"(?P<indent>[ \t]+)speed:\s*0,\s*\n[ \t]+deadline:\s*Deadline::Good,\s*\n"),
        "{indent}speed: 7,\n{indent}deadline: Deadline::Realtime,\n",
    ),
    (
        re.compile(
            r"(?P<indent>[ \t]+)speed:\s*0,\s*\n[ \t]+deadline:\s*rvp9_encoder::Deadline::Good,\s*\n"
        ),
        "{indent}speed: 7,\n{indent}deadline: rvp9_encoder::Deadline::Realtime,\n",
    ),
]


def main() -> None:
    total = 0
    for path in ROOT.rglob("*.rs"):
        # Skip libvpx and target
        s = str(path)
        if "third_party" in s or "\\target\\" in s or "/target/" in s:
            continue
        text = path.read_text()
        new_text = text
        count = 0
        for pat, repl in PATTERNS:
            def sub(m: "re.Match[str]") -> str:
                nonlocal count
                count += 1
                return repl.format(indent=m.group("indent"))
            new_text = pat.sub(sub, new_text)
        if count > 0:
            path.write_text(new_text)
            print(f"  +{count} sites:   {path.relative_to(ROOT)}")
            total += count
    print(f"total rewrites: {total}")


if __name__ == "__main__":
    main()
