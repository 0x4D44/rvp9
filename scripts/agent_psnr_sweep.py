#!/usr/bin/env python3
"""Per-frame PSNR sweep — rvp9enc vs vpxenc on a single y4m at multiple bitrates / cpus.

Used by the Track-F #2 SATD-iter agent. Encodes via both encoders, decodes both
.ivf via vpxdec, then computes per-frame Y-PSNR vs the original y4m. Prints
mean / min / KF PSNR.

Usage:
    python scripts/agent_psnr_sweep.py <tag>
        tag is e.g. "BEFORE" / "AFTER" — used in the output prefix only.
"""
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "tests/corpus/foreman_320x256_60f.y4m"
VPXENC = REPO / "build/libvpx/vpxenc.exe"
VPXDEC = REPO / "build/libvpx/vpxdec.exe"
RVP9ENC = REPO / "target/release/rvp9enc.exe"
OUTDIR = REPO / "build/agent_psnr_sweep"

BITRATES = [400, 1200, 2000]
CPUS = [7, 4]


def read_y4m_y_planes(path):
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
    y_sz = w * h
    c_sz = (w // 2) * (h // 2)
    frame_sz = y_sz + 2 * c_sz
    pos = nl + 1
    frames = []
    while pos < len(data):
        fnl = data.index(b"\n", pos)
        base = fnl + 1
        frames.append(data[base:base + y_sz])
        pos = base + frame_sz
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


def encode_rvp9(out_ivf, bitrate_kbps, cpu_used):
    # Match the bdrate harness flags: --rt --end-usage=cbr --lag-in-frames=0.
    cmd = [
        str(RVP9ENC),
        "--codec=vp9",
        "--passes=1",
        "--rt",
        f"--cpu-used={cpu_used}",
        "--end-usage=cbr",
        f"--target-bitrate={bitrate_kbps}",
        "--lag-in-frames=0",
        "--ivf",
        "-o", str(out_ivf),
        str(SRC),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise RuntimeError(f"rvp9enc failed for {out_ivf}")


def encode_vpx(out_ivf, bitrate_kbps, cpu_used):
    cmd = [
        str(VPXENC),
        "--codec=vp9",
        "--passes=1",
        "--rt",
        f"--cpu-used={cpu_used}",
        "--end-usage=cbr",
        f"--target-bitrate={bitrate_kbps}",
        "--lag-in-frames=0",
        "--ivf",
        "-o", str(out_ivf),
        str(SRC),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise RuntimeError(f"vpxenc failed for {out_ivf}")


def decode_ivf(ivf, out_y4m):
    # Omit --i420 so vpxdec emits proper Y4M with header; -o foo.y4m
    # triggers the y4m muxer.
    cmd = [str(VPXDEC), "--codec=vp9", "-o", str(out_y4m), str(ivf)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    # vpxdec returns non-zero on partial-frame corruption but still writes
    # decoded frames up to the failure point. We accept that and measure
    # PSNR over the frames that decoded successfully.
    if not os.path.exists(out_y4m):
        sys.stderr.write(r.stderr)
        raise RuntimeError(f"vpxdec produced no output for {ivf}")


def measure_psnr(src_path, dec_y4m):
    _, _, src_frames = read_y4m_y_planes(src_path)
    _, _, dec_frames = read_y4m_y_planes(dec_y4m)
    n = min(len(src_frames), len(dec_frames))
    psnrs = [psnr(src_frames[i], dec_frames[i]) for i in range(n)]
    return psnrs


def ivf_size(p):
    return os.path.getsize(p)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "RUN"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Per-frame PSNR sweep ({tag}) — foreman_320x256_60f ===")
    print(f"{'enc':<6} {'cpu':<4} {'br':<5} {'size_KiB':<9} {'mean':<7} {'min':<7} {'kf':<7}")
    for cpu in CPUS:
        for br in BITRATES:
            # rvp9
            rvp9_ivf = OUTDIR / f"rvp9_cpu{cpu}_{br}k.ivf"
            rvp9_y4m = OUTDIR / f"rvp9_cpu{cpu}_{br}k.y4m"
            encode_rvp9(rvp9_ivf, br, cpu)
            decode_ivf(rvp9_ivf, rvp9_y4m)
            p = measure_psnr(SRC, rvp9_y4m)
            mean = sum(p) / len(p)
            mn = min(p)
            kf = p[0]
            print(f"{'rvp9':<6} {cpu:<4} {br:<5} {ivf_size(rvp9_ivf)/1024:<9.1f} {mean:<7.2f} {mn:<7.2f} {kf:<7.2f}")
            # vpx
            vpx_ivf = OUTDIR / f"vpx_cpu{cpu}_{br}k.ivf"
            vpx_y4m = OUTDIR / f"vpx_cpu{cpu}_{br}k.y4m"
            encode_vpx(vpx_ivf, br, cpu)
            decode_ivf(vpx_ivf, vpx_y4m)
            p = measure_psnr(SRC, vpx_y4m)
            mean = sum(p) / len(p)
            mn = min(p)
            kf = p[0]
            print(f"{'vpx':<6} {cpu:<4} {br:<5} {ivf_size(vpx_ivf)/1024:<9.1f} {mean:<7.2f} {mn:<7.2f} {kf:<7.2f}")


if __name__ == "__main__":
    main()
