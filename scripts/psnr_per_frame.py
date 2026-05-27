#!/usr/bin/env python3
"""Per-frame Y-PSNR between two 4:2:0 Y4M files (same dims/frame count).

Usage: python scripts/psnr_per_frame.py <source.y4m> <decoded.y4m> [max_frames]
"""
import math
import sys


def read_y4m(path):
    with open(path, "rb") as f:
        data = f.read()
    nl = data.index(b"\n")
    toks = data[:nl].split()
    w = h = None
    for t in toks[1:]:
        if t[:1] == b"W":
            w = int(t[1:])
        elif t[:1] == b"H":
            h = int(t[1:])
    y, c = w * h, (w // 2) * (h // 2)
    frame = y + 2 * c
    pos = nl + 1
    frames = []
    while pos < len(data):
        fnl = data.index(b"\n", pos)
        base = fnl + 1
        frames.append(data[base:base + y])  # Y plane only
        pos = base + frame
    return w, h, frames


def psnr(a, b):
    n = len(a)
    se = 0
    for i in range(n):
        d = a[i] - b[i]
        se += d * d
    mse = se / n
    if mse == 0:
        return math.inf
    return 10.0 * math.log10(255.0 * 255.0 / mse)


def main():
    src, dec = sys.argv[1], sys.argv[2]
    maxf = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    ws, hs, fs = read_y4m(src)
    wd, hd, fd = read_y4m(dec)
    print(f"src {ws}x{hs} {len(fs)}f | dec {wd}x{hd} {len(fd)}f")
    n = min(len(fs), len(fd), maxf)
    for i in range(n):
        print(f"  frame {i:3d}  Y-PSNR {psnr(fs[i], fd[i]):6.2f} dB")


if __name__ == "__main__":
    main()
