#!/usr/bin/env python3
"""S2 inter-RD reconciliation RCA — 4-config x 2-clip PSNR + mode-mix harness.

Encodes a clip at a given bitrate with rvp9enc (cpu7 --rt --threads=1),
decodes via vpxdec, computes mean Y-PSNR vs a CENTER-CROPPED reference that
matches the encoder's `read_cropped` (off = (src-crop)/2). Reports mean Y-PSNR
and encoded bytes.

Configs are produced by env toggles on a SINGLE S2 binary so the only thing
that varies is the S2 lever(s):
  baseline : RVP9_S2_GATE=0 RVP9_S2_ACDC_RT=0   (HEAD picker behaviour)
  gate     : RVP9_S2_GATE=1 RVP9_S2_ACDC_RT=0
  acdc     : RVP9_S2_GATE=0 RVP9_S2_ACDC_RT=1
  full     : RVP9_S2_GATE=1 RVP9_S2_ACDC_RT=1   (default S2)

Usage:
  python notes/s2-rca-measure.py <enc_exe> <clip.y4m> <coded_w> <coded_h> <limit> <br1,br2,...>
"""
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VPXDEC = REPO / "build/libvpx/vpxdec.exe"
OUTDIR = REPO / "build/s2_rca"

CONFIGS = {
    "baseline": {"RVP9_S2_GATE": "0", "RVP9_S2_ACDC_RT": "0"},
    "gate":     {"RVP9_S2_GATE": "1", "RVP9_S2_ACDC_RT": "0"},
    "acdc":     {"RVP9_S2_GATE": "0", "RVP9_S2_ACDC_RT": "1"},
    "full":     {"RVP9_S2_GATE": "1", "RVP9_S2_ACDC_RT": "1"},
}


def parse_y4m_dims(path):
    with open(path, "rb") as f:
        head = f.read(120)
    nl = head.index(b"\n")
    toks = head[:nl].split()
    w = h = None
    for t in toks[1:]:
        if t[:1] == b"W":
            w = int(t[1:])
        elif t[:1] == b"H":
            h = int(t[1:])
    return w, h


def read_y4m_y_planes(path, limit=None):
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
        if limit is not None and len(frames) >= limit:
            break
    return w, h, frames


def center_crop_y(src_path, cw, ch, limit):
    """Center-crop the Y plane to (cw,ch) matching encoder read_cropped:
    off_x=(sw-cw)//2, off_y=(sh-ch)//2. Returns list of bytes (Y planes)."""
    sw, sh, frames = read_y4m_y_planes(src_path, limit)
    if (sw, sh) == (cw, ch):
        return frames
    off_x = (sw - cw) // 2
    off_y = (sh - ch) // 2
    out = []
    for fy in frames:
        cropped = bytearray(cw * ch)
        for r in range(ch):
            so = (off_y + r) * sw + off_x
            cropped[r * cw:(r + 1) * cw] = fy[so:so + cw]
        out.append(bytes(cropped))
    return out


def psnr(a, b):
    n = min(len(a), len(b))
    se = 0
    for i in range(n):
        d = a[i] - b[i]
        se += d * d
    mse = se / n
    if mse == 0:
        return math.inf
    return 10.0 * math.log10(255.0 * 255.0 / mse)


def encode(enc, src, out_ivf, br, cw, ch, limit, env):
    cmd = [
        str(enc), "--codec=vp9", "--passes=1", "--rt",
        "--cpu-used=7", "--end-usage=cbr", f"--target-bitrate={br}",
        "--lag-in-frames=0", "--threads=1",
        "-w", str(cw), "-h", str(ch), "--limit", str(limit),
        "--ivf", "-o", str(out_ivf), str(src),
    ]
    e = dict(os.environ)
    e.update(env)
    r = subprocess.run(cmd, capture_output=True, text=True, env=e)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:])
        raise RuntimeError(f"encode failed: {out_ivf}")


def decode(ivf, out_y4m):
    cmd = [str(VPXDEC), "--codec=vp9", "-o", str(out_y4m), str(ivf)]
    subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(out_y4m):
        raise RuntimeError(f"decode produced no output: {ivf}")


def main():
    enc, clip, cw, ch, limit, brs = sys.argv[1:7]
    cw, ch, limit = int(cw), int(ch), int(limit)
    brs = [int(x) for x in brs.split(",")]
    clip_path = REPO / clip if not os.path.isabs(clip) else Path(clip)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    name = Path(clip).stem

    # Reference: center-cropped Y planes, frame-limited.
    ref_y = center_crop_y(clip_path, cw, ch, limit)

    print(f"=== {name}  coded={cw}x{ch} limit={limit} enc={Path(enc).name} ===")
    print(f"{'br':<6} {'config':<10} {'bytes':<10} {'mean_psnr':<10}")
    results = {}
    for br in brs:
        for cfg, env in CONFIGS.items():
            tag = f"{name}_{br}k_{cfg}"
            ivf = OUTDIR / f"{tag}.ivf"
            y4m = OUTDIR / f"{tag}.y4m"
            encode(enc, clip_path, ivf, br, cw, ch, limit, env)
            decode(ivf, y4m)
            _, _, dec_y = read_y4m_y_planes(y4m, limit)
            n = min(len(ref_y), len(dec_y))
            ps = [psnr(ref_y[i], dec_y[i]) for i in range(n)]
            finite = [p for p in ps if math.isfinite(p)]
            mean = sum(finite) / len(finite) if finite else float("inf")
            nbytes = os.path.getsize(ivf)
            results[(br, cfg)] = (nbytes, mean, n)
            print(f"{br:<6} {cfg:<10} {nbytes:<10} {mean:<10.3f}  (frames={n})")
    return results


if __name__ == "__main__":
    main()
