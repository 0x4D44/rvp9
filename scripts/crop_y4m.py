#!/usr/bin/env python3
"""Crop a 4:2:0 Y4M to a top-left WxH window and optionally limit frames.

Produces an apples-to-apples encoder-comparison source: both encoders read
the SAME cropped/limited file, so PSNR is computed over identical input
pixels. Needed because rvp9enc requires coded dimensions that are multiples
of 64, while vpxenc accepts any size.

Usage:
  python scripts/crop_y4m.py <in.y4m> <out.y4m> <W> <H> [<frame_limit>]
"""
import sys


def main() -> int:
    if len(sys.argv) < 5:
        print(__doc__)
        return 2
    src, dst, W, H = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else None

    with open(src, "rb") as f:
        data = f.read()

    nl = data.index(b"\n")
    header = data[:nl].decode("ascii")
    tokens = header.split()
    assert tokens[0] == "YUV4MPEG2", f"not a y4m: {tokens[0]!r}"
    sw = sh = None
    for t in tokens[1:]:
        if t[0] == "W":
            sw = int(t[1:])
        elif t[0] == "H":
            sh = int(t[1:])
    assert sw and sh, "missing W/H in header"
    assert W <= sw and H <= sh, f"crop {W}x{H} exceeds source {sw}x{sh}"

    # Rebuild header with new W/H, preserving every other token verbatim.
    out_tokens = ["YUV4MPEG2"]
    for t in tokens[1:]:
        if t[0] == "W":
            out_tokens.append(f"W{W}")
        elif t[0] == "H":
            out_tokens.append(f"H{H}")
        else:
            out_tokens.append(t)
    out_header = (" ".join(out_tokens) + "\n").encode("ascii")

    s_y, s_c = sw * sh, (sw // 2) * (sh // 2)
    s_frame = s_y + 2 * s_c
    pos = nl + 1
    out = bytearray(out_header)
    n = 0
    while pos < len(data):
        if data[pos:pos + 5] != b"FRAME":
            raise SystemExit(f"expected FRAME at {pos}, got {data[pos:pos+8]!r}")
        fnl = data.index(b"\n", pos)
        out += data[pos:fnl + 1]            # copy FRAME header line verbatim
        base = fnl + 1
        yb = base
        ub = base + s_y
        vb = base + s_y + s_c
        # Y plane: top-left HxW window, source stride sw.
        for r in range(H):
            o = yb + r * sw
            out += data[o:o + W]
        # U/V planes: (H/2)x(W/2) window, source stride sw/2.
        for (cb, scw) in ((ub, sw // 2), (vb, sw // 2)):
            for r in range(H // 2):
                o = cb + r * scw
                out += data[o:o + (W // 2)]
        pos = base + s_frame
        n += 1
        if limit is not None and n >= limit:
            break

    with open(dst, "wb") as f:
        f.write(out)
    print(f"wrote {dst}: {W}x{H}, {n} frames ({len(out)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
