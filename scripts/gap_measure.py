#!/usr/bin/env python3
"""Matched-bitrate Y-PSNR gap: rvp9enc vs libvpx (vpxenc), independent PSNR.

Drives both CLI encoders at matched flags + target bitrates, decodes each IVF
with vpxdec, computes Y-PSNR vs the SOURCE y4m with ffmpeg (decoder-independent).
Reports per (clip, preset, bitrate): measured kbps + Y-PSNR for each encoder and
the dB gap, plus a cubic Bjontegaard BD-rate when the PSNR ranges overlap.

Read-only diagnosis; writes only temp IVF/y4m under TMP.
"""
import os, subprocess, sys, tempfile, math, json, re

ROOT = r"D:\language\rvp9-2"
RVP9ENC = os.path.join(ROOT, r"target\release\rvp9enc.exe")
VPXENC = os.path.join(ROOT, r"third_party\libvpx\build_f02\vpxenc.exe")
VPXDEC = os.path.join(ROOT, r"third_party\libvpx\build_f02\vpxdec.exe")
FFMPEG = "ffmpeg"
TMP = tempfile.gettempdir()

def y4m_dims(path):
    with open(path, "rb") as f:
        hdr = f.readline().decode("latin1")
    w = int(re.search(r"W(\d+)", hdr).group(1))
    h = int(re.search(r"H(\d+)", hdr).group(1))
    m = re.search(r"F(\d+):(\d+)", hdr)
    fps = (int(m.group(1)) / int(m.group(2))) if m else 30.0
    # frame count
    fsz = w * h * 3 // 2
    total = os.path.getsize(path)
    nframes = (total - len(hdr)) // (fsz + 6)
    return w, h, fps, nframes

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def encode_rvp9(src, w, h, fps, kbps, preset, out):
    deadline = "--rt" if preset == "cpu7" else "--good"
    cpu = "7" if preset == "cpu7" else "4"
    cmd = [RVP9ENC, "--codec=vp9", "--ivf", deadline, f"--cpu-used={cpu}",
           "--passes=1", "--end-usage=cbr", f"--target-bitrate={kbps}",
           "--lag-in-frames=0", f"-w", str(w), f"-h", str(h),
           "-o", out, src]
    r = run(cmd)
    if r.returncode != 0:
        sys.stderr.write(f"[rvp9enc FAIL {preset} {kbps}k] {r.stderr.decode('latin1')[-400:]}\n")
        return False
    return True

def encode_vpx(src, w, h, fps, kbps, preset, out):
    deadline = "--rt" if preset == "cpu7" else "--good"
    cpu = "7" if preset == "cpu7" else "4"
    cmd = [VPXENC, "--codec=vp9", "--ivf", deadline, f"--cpu-used={cpu}",
           "--passes=1", "--end-usage=cbr", f"--target-bitrate={kbps}",
           "--lag-in-frames=0", "--aq-mode=0",
           "-o", out, src]
    r = run(cmd)
    if r.returncode != 0:
        sys.stderr.write(f"[vpxenc FAIL {preset} {kbps}k] {r.stderr.decode('latin1')[-400:]}\n")
        return False
    return True

def decode_to_y4m(ivf, out):
    cmd = [VPXDEC, "--codec=vp9", "-o", out, ivf]
    r = run(cmd)
    return r.returncode == 0

def ypsnr(src, dec):
    # ffmpeg psnr filter: dec as test, src as ref. CRITICAL: normalize
    # timebase + reindex frames by N so a fps/timebase mismatch between the
    # source y4m (e.g. 30000/1001) and the vpxdec output doesn't misalign
    # frames (which silently produces a bogus flat-PSNR curve).
    cmd = [FFMPEG, "-i", dec, "-i", src, "-lavfi",
           "[0:v]settb=AVTB,setpts=N[a];[1:v]settb=AVTB,setpts=N[b];[a][b]psnr",
           "-f", "null", "-"]
    r = run(cmd)
    txt = r.stderr.decode("latin1")
    # Summary line: "PSNR y:27.125253 u:... v:... average:... min:... max:..."
    m = re.search(r"PSNR y:([0-9.]+|inf)", txt)
    if m:
        v = m.group(1)
        return float("inf") if v == "inf" else float(v)
    return None

def measured_kbps(ivf, nframes, fps):
    bytes_ = os.path.getsize(ivf)
    dur = nframes / fps
    return (bytes_ * 8 / 1000.0) / dur

def polyfit3(xs, ys):
    # least squares cubic: y = c0 + c1 x + c2 x^2 + c3 x^3
    import numpy as np
    A = np.vstack([xs**0, xs, xs**2, xs**3]).T
    c, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return c

def bdrate(anchor, test):
    # anchor/test = list of (kbps, psnr). Returns BD-rate % (positive = test costs more bits).
    import numpy as np
    if len(anchor) < 4 or len(test) < 4:
        return None
    a = sorted(anchor, key=lambda p: p[1]); t = sorted(test, key=lambda p: p[1])
    ax = np.array([p[1] for p in a]); ay = np.log(np.array([p[0] for p in a]))
    tx = np.array([p[1] for p in t]); ty = np.log(np.array([p[0] for p in t]))
    ca = polyfit3(ax, ay); ct = polyfit3(tx, ty)
    lo = max(ax.min(), tx.min()); hi = min(ax.max(), tx.max())
    if hi <= lo:
        return None
    def integ(c):
        f = lambda x: c[0]*x + c[1]*x*x/2 + c[2]*x**3/3 + c[3]*x**4/4
        return f(hi) - f(lo)
    avg = (integ(ct) - integ(ca)) / (hi - lo)
    return (math.exp(avg) - 1.0) * 100.0

def main():
    clips = [
        ("foreman", os.path.join(ROOT, r"tests\corpus\foreman_320x256_60f.y4m")),
        ("testsrc", os.path.join(ROOT, r"notes\testsrc_1280x704_60f.y4m")),
    ]
    bitrates = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [100,250,500,1000,2000]
    presets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["cpu7","cpu4"]

    for name, src in clips:
        w, h, fps, nf = y4m_dims(src)
        print(f"\n========== {name}  {w}x{h} @ {fps:.2f} fps, {nf} frames ==========")
        for preset in presets:
            print(f"\n--- preset {preset} ---")
            print(f"{'tgt':>6} | {'rvp9 kbps':>9} {'rvp9 dB':>8} | {'vpx kbps':>9} {'vpx dB':>8} | {'gap dB':>7}")
            rvp9_rd = []; vpx_rd = []
            for kbps in bitrates:
                tag = f"{name}_{preset}_{kbps}"
                r_ivf = os.path.join(TMP, f"r_{tag}.ivf"); v_ivf = os.path.join(TMP, f"v_{tag}.ivf")
                r_y4m = os.path.join(TMP, f"r_{tag}.y4m"); v_y4m = os.path.join(TMP, f"v_{tag}.y4m")
                ok_r = encode_rvp9(src, w, h, fps, kbps, preset, r_ivf)
                ok_v = encode_vpx(src, w, h, fps, kbps, preset, v_ivf)
                rp = vp = None; rk = vk = float('nan')
                if ok_r and decode_to_y4m(r_ivf, r_y4m):
                    rp = ypsnr(src, r_y4m); rk = measured_kbps(r_ivf, nf, fps)
                if ok_v and decode_to_y4m(v_ivf, v_y4m):
                    vp = ypsnr(src, v_y4m); vk = measured_kbps(v_ivf, nf, fps)
                gap = (vp - rp) if (rp is not None and vp is not None and math.isfinite(rp) and math.isfinite(vp)) else float('nan')
                print(f"{kbps:>6} | {rk:>9.1f} {rp if rp is not None else float('nan'):>8.2f} | {vk:>9.1f} {vp if vp is not None else float('nan'):>8.2f} | {gap:>7.2f}")
                if rp is not None and math.isfinite(rp): rvp9_rd.append((rk, rp))
                if vp is not None and math.isfinite(vp): vpx_rd.append((vk, vp))
                for p in (r_ivf, v_ivf, r_y4m, v_y4m):
                    try: os.remove(p)
                    except OSError: pass
            bd = bdrate(vpx_rd, rvp9_rd)
            if bd is not None:
                print(f"BD-rate (vpx anchor -> rvp9 test, +=rvp9 costs more bits): {bd:+.2f}%")
            else:
                print("BD-rate: not computable (no PSNR overlap or <4 points)")

if __name__ == "__main__":
    main()
