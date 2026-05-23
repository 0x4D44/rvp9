#!/usr/bin/env python3
"""Diff rvp9 vs libvpx keyframe partition cursor traces.

Each side emits per-partition pre/post lines.  We normalise both to a list
of (mi_row, mi_col, bsl, state, partition, range, value, count) records and
diff in order.

state ∈ {"pre", "post"}.  Cursor (range, value, count) must match between
sides at every event.  Print the first divergence.
"""
import re, sys

rvp9_log, libvpx_log = sys.argv[1], sys.argv[2]

# rvp9 lines (matches both pre and post):
re_rvp9 = re.compile(
    r"^\[rvp9_kf_partition_(pre|post)\] mi=\((\d+),(\d+)\) bsize=(\w+) "
    r"(?:ctx=(\d+)|partition=(\w+))"
    r".*range=(\d+) value=0x([0-9a-f]+) count=(-?\d+) pos=(\d+)"
)
re_libvpx = re.compile(
    r"^\[libvpx_kf_part_(pre|post)\] mi=\((\d+),(\d+)\) bsl=(\d+) "
    r"(?:ctx=(\d+)|partition=(\d+))"
    r".*range=(\d+) value=0x([0-9a-f]+) count=(-?\d+) pos=(\d+)"
)

# bsize_name -> bsl (16x16=1, 32x32=2, 64x64=3, 8x8=0)
BSL_FROM_BSIZE = {"Block8x8": 0, "Block16x16": 1, "Block32x32": 2, "Block64x64": 3}

def parse_rvp9(path):
    events = []
    with open(path) as f:
        for line in f:
            m = re_rvp9.match(line)
            if not m:
                continue
            kind, mr, mc, bs, ctx, part, rng, val, cnt, pos = m.groups()
            bsl = BSL_FROM_BSIZE.get(bs, -1)
            events.append({
                "kind": kind,
                "mr": int(mr), "mc": int(mc), "bsl": bsl,
                "ctx": int(ctx) if ctx else None,
                "partition": part,
                "range": int(rng), "value": int(val, 16), "count": int(cnt),
                "pos": int(pos),
            })
    return events

def parse_libvpx(path):
    events = []
    with open(path) as f:
        for line in f:
            m = re_libvpx.match(line)
            if not m:
                continue
            kind, mr, mc, bsl, ctx, part, rng, val, cnt, pos = m.groups()
            events.append({
                "kind": kind,
                "mr": int(mr), "mc": int(mc), "bsl": int(bsl),
                "ctx": int(ctx) if ctx else None,
                "partition": part,
                "range": int(rng), "value": int(val, 16), "count": int(cnt),
                "pos": int(pos),
            })
    return events

A = parse_rvp9(rvp9_log)
B = parse_libvpx(libvpx_log)

# Filter both to keyframe frame 0 only.  rvp9 trace is per-frame and we
# can't easily distinguish frames in the log; but libvpx (with --limit=1)
# only emits one frame's worth.  So use libvpx's count as a soft anchor.
# Actually: take the first N rvp9 records where N = len(libvpx).
A = A[:len(B)]

print(f"rvp9 events: {len(A)}, libvpx events: {len(B)}")
print(f"(comparing first {min(len(A), len(B))} pairs)")

first_diff = None
for i, (a, b) in enumerate(zip(A, B)):
    # Identity: same (mr, mc, bsl, kind) tag.
    if (a["mr"], a["mc"], a["bsl"], a["kind"]) != (b["mr"], b["mc"], b["bsl"], b["kind"]):
        print(f"\nEVENT-TAG DIVERGENCE at i={i}:")
        print(f"  rvp9:   mi=({a['mr']},{a['mc']}) bsl={a['bsl']} kind={a['kind']}")
        print(f"  libvpx: mi=({b['mr']},{b['mc']}) bsl={b['bsl']} kind={b['kind']}")
        first_diff = i
        break
    # Cursor must match exactly.
    if (a["range"], a["value"], a["count"]) != (b["range"], b["value"], b["count"]):
        print(f"\nCURSOR DIVERGENCE at i={i} mi=({a['mr']},{a['mc']}) bsl={a['bsl']} kind={a['kind']}:")
        print(f"  rvp9:   range={a['range']:>3} value=0x{a['value']:016x} count={a['count']:>3}")
        print(f"  libvpx: range={b['range']:>3} value=0x{b['value']:016x} count={b['count']:>3}")
        first_diff = i
        # Show 3 events before for context
        print(f"\n  Last 3 matching events before divergence:")
        for j in range(max(0, i-3), i):
            print(f"    [{j}] mi=({A[j]['mr']},{A[j]['mc']}) bsl={A[j]['bsl']} {A[j]['kind']} range={A[j]['range']} value=0x{A[j]['value']:016x} count={A[j]['count']}")
        break

if first_diff is None:
    print(f"\nAll {min(len(A), len(B))} partition events matched on (mi, bsl, kind, range, value, count).")
    if len(A) != len(B):
        print(f"  But event counts differ: rvp9={len(A)} vs libvpx={len(B)}")
