#!/usr/bin/env python3
"""W0.2 helper: rewrite EncoderConfig struct literals to include the two
new fields (`speed`, `deadline`) introduced by the vpxenc parity port.

Run from the repo root. Idempotent — skips literals that already mention
`speed:` or `deadline:`.

The pattern: every existing literal ends with `use_prev_frame_mvs: ...,`.
We insert the two new fields immediately after that line, matching the
file's indent. The qualifier on `Deadline::Good` differs by file:
  - frame_loop.rs (internal): `Deadline::Good`
  - everything else:           `rvp9_encoder::Deadline::Good`
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files to rewrite (collected from `cargo test --no-run` error output).
TARGETS = [
    "crates/rvp9-bench/src/bin/soak.rs",
    "crates/rvp9-cli/src/bin/rvp9enc.rs",
    "crates/rvp9-encoder/src/frame_loop.rs",
    "crates/rvp9-encoder/src/lib.rs",
    "crates/rvp9-encoder/tests/a18_picker_reads_prev_grid.rs",
    "crates/rvp9-encoder/tests/a18_prev_frame_mvs_fires.rs",
    "crates/rvp9-encoder/tests/cross_decoder_gate.rs",
    "crates/rvp9-encoder/tests/diag_foreman_kf.rs",
    "crates/rvp9-encoder/tests/diag_foreman_per_frame.rs",
    "crates/rvp9-encoder/tests/diag_frame4_dump.rs",
    "crates/rvp9-encoder/tests/diag_per_frame.rs",
    "crates/rvp9-encoder/tests/diag_ref_divergence.rs",
    "crates/rvp9-encoder/tests/e12_oracle.rs",
    "crates/rvp9-encoder/tests/e13_diamond.rs",
    "crates/rvp9-encoder/tests/e3_mvdiag.rs",
    "crates/rvp9-encoder/tests/e4_overshoot_accounting.rs",
    "crates/rvp9-encoder/tests/e9_quality_gap.rs",
    "crates/rvp9-encoder/tests/f2_find_mv_refs_temporal_visible.rs",
    "crates/rvp9-encoder/tests/quality_gate.rs",
    "crates/rvp9-harness/src/bin/e1_block_stats.rs",
    "crates/rvp9-harness/src/bin/f02_block_oracle.rs",
    "crates/rvp9-harness/tests/f02_oracle_smoke.rs",
]

INTERNAL = {"crates/rvp9-encoder/src/frame_loop.rs", "crates/rvp9-encoder/src/lib.rs"}

# Match `<indent>use_prev_frame_mvs: <expr>,\n` and capture indent + line.
PAT = re.compile(r"^(?P<indent>[ \t]+)use_prev_frame_mvs:[^\n]*?,\n", re.MULTILINE)


def transform(text: str, deadline_path: str) -> tuple[str, int]:
    """Insert speed/deadline lines after each matching line.

    Skip a match if the literal already contains a `speed:` or
    `deadline:` field within the next 5 lines (idempotency guard).
    """
    out = []
    last = 0
    count = 0
    for m in PAT.finditer(text):
        # Idempotency: look ahead a few lines for an existing speed/deadline.
        tail_window = text[m.end(): m.end() + 200]
        if re.search(r"^\s*(?:speed|deadline)\s*:", tail_window, re.MULTILINE):
            continue
        out.append(text[last:m.end()])
        indent = m.group("indent")
        out.append(f"{indent}speed: 0,\n{indent}deadline: {deadline_path}::Good,\n")
        last = m.end()
        count += 1
    out.append(text[last:])
    return "".join(out), count


def main() -> int:
    total = 0
    for rel in TARGETS:
        p = ROOT / rel
        if not p.exists():
            print(f"  skip (not found): {rel}", file=sys.stderr)
            continue
        text = p.read_text()
        deadline_path = (
            "Deadline" if rel.replace("\\", "/") in INTERNAL
            else "rvp9_encoder::Deadline"
        )
        new_text, count = transform(text, deadline_path)
        if count == 0:
            print(f"  no changes:    {rel}")
            continue
        p.write_text(new_text)
        print(f"  +{count} sites:   {rel}")
        total += count
    print(f"total inserts: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
