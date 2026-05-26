#!/usr/bin/env python3
"""W5.1 helper: rewrite EncoderConfig struct literals (and the
`EncoderConfig::new` constant constructor) to include the new
`noise_sensitivity: u8` field introduced by the vpxenc parity port.

Run from the repo root. Idempotent — skips literals that already mention
`noise_sensitivity:`.

The pattern: every existing literal ends with `deadline: <expr>,\n`. We
insert the new field immediately after that line, matching the file's
indent. Default value is `0` (denoiser off) everywhere — matches
vpxenc's CLI default and preserves existing test behaviour.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files to rewrite (collected from `cargo test --no-run` error output of
# the W0.2 sweep — denoiser pickup is a superset of that list, plus the
# realtime_gate test that landed after W0.2).
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
    "crates/rvp9-encoder/tests/realtime_gate.rs",
    "crates/rvp9-harness/src/bin/e1_block_stats.rs",
    "crates/rvp9-harness/src/bin/f02_block_oracle.rs",
    "crates/rvp9-harness/tests/f02_oracle_smoke.rs",
]

# Match the `speed: <expr>,\n<indent>deadline: <expr>,\n` *pair* and
# capture the trailing indent. The `(speed, deadline)` pair was
# introduced by W0.2 *only* inside EncoderConfig literals — no other
# struct or function-arg list in the workspace uses both adjacent
# fields, so this is a robust EncoderConfig-only anchor. Anchoring on
# `deadline:` alone caught `EncOpts { deadline: None, ... }` and
# function-arg lists `(deadline: <Type>, ...)`, both unrelated.
PAT = re.compile(
    r"^[ \t]+speed:[^\n]*?,\n"
    r"^(?P<indent>[ \t]+)deadline:(?P<value>[^\n]*?),\n",
    re.MULTILINE,
)


def transform(text: str) -> tuple[str, int]:
    """Insert noise_sensitivity line after each matching line.

    Skip a match if the literal already contains a `noise_sensitivity:`
    field within the next ~5 lines (idempotency guard). Also skip
    matches that look like a function-argument-list (where the next
    non-whitespace token is `)` or another typed param `<ident>: <Type>`
    rather than another struct field literal `<ident>: <expr>,` or a
    closing brace).
    """
    out = []
    last = 0
    count = 0
    for m in PAT.finditer(text):
        # Type-declaration guard: `deadline: <Type>,` (e.g.
        # `Option<Deadline>` in a struct/fn-arg declaration). Real
        # struct-literal values for `deadline:` are simple expressions
        # like `Deadline::Realtime` or `opts.deadline.unwrap_or(...)`.
        # If the value contains a `<` token it is almost certainly a
        # generic-type bracket; skip.
        value = m.group("value")
        if "<" in value:
            continue
        tail_window = text[m.end(): m.end() + 200]
        # Idempotency: skip if already present.
        if re.search(r"^\s*noise_sensitivity\s*:", tail_window, re.MULTILINE):
            continue
        # Function-arg-list guard: in a function-arg list the next
        # token after `deadline: <Type>,` is the next typed param
        # (`<ident>: <Type>,`) and the block terminates with `) ->`
        # or `)` instead of `}`. Struct literals terminate with `}`
        # (or `};` at statement position) within a few lines. Require
        # at least one `}` to appear before a `) ->` or `) {` in the
        # near window — otherwise this is a function arg list.
        close_brace = tail_window.find('}')
        close_paren_arrow = tail_window.find(') ->')
        close_paren_brace = tail_window.find(') {')
        # Pick the nearest closer; if none exists, treat as struct
        # (idempotency window will have skipped anyway). If a `)`
        # closer comes before the `}`, it's a function-arg list.
        candidates = [c for c in (close_paren_arrow, close_paren_brace) if c >= 0]
        nearest_paren = min(candidates) if candidates else -1
        if nearest_paren >= 0 and (close_brace < 0 or nearest_paren < close_brace):
            continue
        out.append(text[last:m.end()])
        indent = m.group("indent")
        # rvp9 default = vpxenc default: denoiser disabled. Every
        # existing call site predates W5 and was written against an
        # encoder with no denoiser — preserve that by defaulting to 0.
        out.append(f"{indent}noise_sensitivity: 0,\n")
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
        text = p.read_text(encoding="utf-8")
        new_text, count = transform(text)
        if count == 0:
            print(f"  no changes:    {rel}")
            continue
        p.write_text(new_text, encoding="utf-8")
        print(f"  +{count} sites:   {rel}")
        total += count
    print(f"total inserts: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
