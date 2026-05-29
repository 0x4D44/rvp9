#!/usr/bin/env python3
"""Phase-2 option-A measurement harness."""
import math, os, subprocess, sys, time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
SHARED = Path(r"D:/language/rvp9-2")
FOREMAN = SHARED / "tests/corpus/foreman_320x256_60f.y4m"
FOREMAN_CIF = SHARED / "tests/corpus/foreman_cif.y4m"
VPXDEC = SHARED / "build/libvpx/vpxdec.exe"
OUTDIR = WORKTREE / "build/phase2_measure"
BITRATES = [100, 250, 1000]
CPUS = [7, 4]
BEST_OF = 3
CIF_FRAMES = 60

def read_y4m(path):
    with open(path, "rb") as f:
        data = f.read()
    nl = data.index(b"\n"); toks = data[:nl].split(); w = h = None
    for t in toks[1:]:
        if t[:1] == b"W": w = int(t[1:])
        elif t[:1] == b"H": h = int(t[1:])
    y_sz = w*h; c_sz = (w//2)*(h//2); frame_sz = y_sz+2*c_sz; pos = nl+1; frames = []
    while pos < len(data):
        fnl = data.index(b"\n", pos); base = fnl+1
        frames.append((data[base:base+y_sz], data[base+y_sz:base+y_sz+c_sz], data[base+y_sz+c_sz:base+frame_sz]))
        pos = base+frame_sz
    return w, h, frames

def crop_center_y4m(src, dw, dh, n, out):
    w, h, frames = read_y4m(src); x0 = (w-dw)//2; y0 = (h-dh)//2; cw, ch = dw//2, dh//2
    with open(out, "wb") as f:
        f.write(b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420jpeg\n" % (dw, dh))
        for i in range(min(n, len(frames))):
            y, u, v = frames[i]; f.write(b"FRAME\n")
            for r in range(dh):
                so = (y0+r)*w+x0; f.write(y[so:so+dw])
            for r in range(ch):
                so = (y0//2+r)*(w//2)+x0//2; f.write(u[so:so+cw])
            for r in range(ch):
                so = (y0//2+r)*(w//2)+x0//2; f.write(v[so:so+cw])

def read_y_planes(path):
    w, h, frames = read_y4m(path); return w, h, [fr[0] for fr in frames]

def psnr(a, b):
    n = min(len(a), len(b)); se = 0
    for i in range(n):
        d = a[i]-b[i]; se += d*d
    mse = se/n
    return math.inf if mse == 0 else 10.0*math.log10(255.0*255.0/mse)

def encode(enc, out_ivf, src, br, cpu):
    cmd = [str(enc), "--codec=vp9", "--passes=1", "--rt", f"--cpu-used={cpu}",
           "--end-usage=cbr", f"--target-bitrate={br}", "--lag-in-frames=0",
           "--ivf", "-o", str(out_ivf), str(src)]
    t0 = time.perf_counter(); r = subprocess.run(cmd, capture_output=True, text=True); dt = time.perf_counter()-t0
    if r.returncode != 0:
        sys.stderr.write(r.stderr); raise RuntimeError(f"rvp9enc failed for {out_ivf}")
    return dt

def decode_ivf(ivf, out_y4m):
    cmd = [str(VPXDEC), "--codec=vp9", "-o", str(out_y4m), str(ivf)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return os.path.exists(out_y4m), r.returncode

def measure(src_path, dec_y4m, n_src):
    _, _, sf = read_y_planes(src_path); _, _, df = read_y_planes(dec_y4m)
    n = min(len(sf), len(df), n_src)
    if n == 0: return None
    return [psnr(sf[i], df[i]) for i in range(n)], n

def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "RUN"
    enc = Path(sys.argv[2]) if len(sys.argv) > 2 else (WORKTREE / "target/release/rvp9enc.exe")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tcrop = OUTDIR / "foremancif256.y4m"
    crop_center_y4m(FOREMAN_CIF, 256, 256, CIF_FRAMES, tcrop)
    _, _, tsf = read_y_planes(tcrop); _, _, fmf = read_y_planes(FOREMAN)
    clips = [("foreman", FOREMAN, len(fmf)), ("foremancif256", tcrop, len(tsf))]
    print(f"=== Phase-2 measure ({tag}) enc={enc} ===")
    print(f"{'clip':<11} {'cpu':<4} {'br':<5} {'size_KiB':<9} {'mean':<8} {'min':<7} {'kf':<7} {'fps':<7} {'enc_s':<7} {'decok':<6} {'decrc'}")
    for cn, src, nf in clips:
        for cpu in CPUS:
            for br in BITRATES:
                ivf = OUTDIR / f"{tag}_{cn}_cpu{cpu}_{br}k.ivf"; best = None
                for _ in range(BEST_OF):
                    dt = encode(enc, ivf, src, br, cpu)
                    if best is None or dt < best: best = dt
                y4m = OUTDIR / f"{tag}_{cn}_cpu{cpu}_{br}k.y4m"
                if y4m.exists(): y4m.unlink()
                decok, decrc = decode_ivf(ivf, y4m)
                res = measure(src, y4m, nf) if decok else None
                if res is None:
                    print(f"{cn:<11} {cpu:<4} {br:<5} {os.path.getsize(ivf)/1024:<9.1f} {'DECFAIL':<8} {'':<7} {'':<7} {nf/best:<7.2f} {best:<7.3f} {str(decok):<6} {decrc}")
                    continue
                p, n = res; mean = sum(p)/len(p); mn = min(p); kf = p[0]; fps = nf/best
                print(f"{cn:<11} {cpu:<4} {br:<5} {os.path.getsize(ivf)/1024:<9.1f} {mean:<8.4f} {mn:<7.3f} {kf:<7.3f} {fps:<7.2f} {best:<7.3f} {str(decok):<6} {decrc}")

if __name__ == "__main__":
    main()
