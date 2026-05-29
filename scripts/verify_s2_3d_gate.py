#!/usr/bin/env python3
"""A/B verify the frame-adaptive S2 (3D) gate: default-ON vs baseline (RVP9_3D=0).

For each (clip, bitrate) the script encodes twice with rvp9enc:
  - BASELINE: env RVP9_3D=0 (classifier off -> global S2 gate off -> 1806f7a baseline)
  - DEFAULT : no env (classifier ON by default -> S2 active on natural frames)
Both .ivf are decoded with the libvpx reference decoder (vpxdec) and per-frame
Y-PSNR is computed against the source y4m. Reports mean PSNR for each, the delta,
matched encoded size, and the frame-0 hf_energy classification (natural/synthetic).

Usage:
    python scripts/verify_s2_3d_gate.py
"""
import math
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VPXDEC = REPO / "third_party/libvpx/build_f02/vpxdec.exe"
RVP9ENC = REPO / "target/release/rvp9enc.exe"
OUTDIR = REPO / "build/verify_s2_3d"

# (label, src y4m, coded_w, coded_h, [bitrates_kbps], limit)
CLIPS = [
    ("foreman_320x256", "tests/corpus/foreman_320x256_60f.y4m", 320, 256, [100, 250, 1000], 60),
    ("testsrc_720p_crop", "tests/corpus/testsrc_720p30.y4m", 1280, 704, [1000, 2000], 60),
    ("foreman_cif_crop", "tests/corpus/foreman_cif.y4m", 320, 256, [250, 500], 60),
]


def read_y4m_y_planes(path, want_w=None, want_h=None):
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
        y = data[base:base + y_sz]
        # Centre-crop the source luma to the coded (want_w x want_h)
        # window so it matches the encoder (rvp9-io `read_cropped` uses
        # `off = (src - crop) / 2`); the decoded frame is the coded size.
        if want_w is not None and (want_w != w or want_h != h):
            off_x = (w - want_w) // 2
            off_y = (h - want_h) // 2
            cropped = bytearray(want_w * want_h)
            for r in range(want_h):
                s = (r + off_y) * w + off_x
                cropped[r * want_w:(r + 1) * want_w] = y[s:s + want_w]
            y = bytes(cropped)
        frames.append(y)
        pos = base + frame_sz
    return (want_w or w), (want_h or h), frames


def psnr_pair(a, b):
    n = min(len(a), len(b))
    se = 0
    for i in range(n):
        d = a[i] - b[i]
        se += d * d
    mse = se / n
    if mse == 0:
        return math.inf
    return 10.0 * math.log10(255.0 * 255.0 / mse)


def encode(out_ivf, src, br, w, h, limit, env3d, dump=False):
    cmd = [
        str(RVP9ENC), "--codec=vp9", "--passes=1", "--rt", "--cpu-used=7",
        "--end-usage=cbr", f"--target-bitrate={br}", "--lag-in-frames=0",
        f"--limit={limit}", "-w", str(w), "-h", str(h),
        "--ivf", "-o", str(out_ivf), str(src),
    ]
    env = dict(os.environ)
    if env3d is not None:
        env["RVP9_3D"] = env3d
    if dump:
        env["RVP9_3D_DUMP"] = "1"
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise RuntimeError(f"rvp9enc failed for {out_ivf}")
    return r.stderr


def decode_ivf(ivf, out_y4m):
    cmd = [str(VPXDEC), "--codec=vp9", "-o", str(out_y4m), str(ivf)]
    subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(out_y4m):
        raise RuntimeError(f"vpxdec produced no output for {ivf}")


def mean_psnr(src, dec_y4m, w, h):
    _, _, sf = read_y4m_y_planes(src, w, h)
    _, _, df = read_y4m_y_planes(dec_y4m)
    n = min(len(sf), len(df))
    if n == 0:
        return None, 0
    ps = [psnr_pair(sf[i], df[i]) for i in range(n)]
    finite = [p for p in ps if math.isfinite(p)]
    if not finite:
        return math.inf, n
    return sum(finite) / len(finite), n


HF_RE = re.compile(r"hf_energy=([0-9.]+)")


def first_hf(stderr_text):
    m = HF_RE.search(stderr_text)
    return float(m.group(1)) if m else None


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"{'clip':<20} {'br':>5} {'base_dB':>9} {'def_dB':>9} {'delta':>8} "
          f"{'base_KiB':>9} {'def_KiB':>9} {'hf':>6} {'class':>10}")
    any_regress = False
    for label, rel, w, h, brs, limit in CLIPS:
        src = REPO / rel
        for br in brs:
            base_ivf = OUTDIR / f"{label}_{br}k_base.ivf"
            def_ivf = OUTDIR / f"{label}_{br}k_def.ivf"
            base_y4m = OUTDIR / f"{label}_{br}k_base.y4m"
            def_y4m = OUTDIR / f"{label}_{br}k_def.y4m"
            encode(base_ivf, src, br, w, h, limit, env3d="0")
            stderr_def = encode(def_ivf, src, br, w, h, limit, env3d=None, dump=True)
            decode_ivf(base_ivf, base_y4m)
            decode_ivf(def_ivf, def_y4m)
            base_db, _ = mean_psnr(src, base_y4m, w, h)
            def_db, _ = mean_psnr(src, def_y4m, w, h)
            hf = first_hf(stderr_def)
            cls = "natural" if (hf is not None and hf >= 7.0) else "synthetic"
            delta = def_db - base_db
            if delta < -0.001:
                any_regress = True
            bsz = os.path.getsize(base_ivf) / 1024
            dsz = os.path.getsize(def_ivf) / 1024
            print(f"{label:<20} {br:>5} {base_db:>9.3f} {def_db:>9.3f} {delta:>+8.3f} "
                  f"{bsz:>9.1f} {dsz:>9.1f} {hf:>6.2f} {cls:>10}")
    print()
    print("REGRESSION DETECTED" if any_regress else "NO REGRESSION (all delta >= 0)")


if __name__ == "__main__":
    main()
