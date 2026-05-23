#!/usr/bin/env python3
"""Compare rvp9 KFLEAF trace to libvpx mi-grid dump for frame 0.

Finds the FIRST mi position where the two decoders disagree on
{sb_type, segment_id, skip, tx_size, y_mode, uv_mode}.

Usage:
  python scripts/compare_kfleaf_to_libvpx.py \
      notes/foreman_c440_kfleaf_frame0.log libvpx_mi_dump.txt
"""
import re, sys

# rvp9 enum -> libvpx int.  Order is the libvpx BLOCK_SIZE / TX_SIZE / etc.
BSIZE = {
    "Block4x4": 0, "Block4x8": 1, "Block8x4": 2, "Block8x8": 3,
    "Block8x16": 4, "Block16x8": 5, "Block16x16": 6, "Block16x32": 7,
    "Block32x16": 8, "Block32x32": 9, "Block32x64": 10, "Block64x32": 11,
    "Block64x64": 12,
}
TXSIZE = {"Tx4x4": 0, "Tx8x8": 1, "Tx16x16": 2, "Tx32x32": 3}
INTRAMODE = {  # rvp9 enum names -> libvpx int (PREDICTION_MODE)
    "Dc": 0, "V": 1, "H": 2, "D45": 3, "D135": 4,
    "D117": 5, "D153": 6, "D207": 7, "D63": 8, "Tm": 9,
}

rvp9_log, libvpx_log = sys.argv[1], sys.argv[2]

# --- Parse libvpx mi grid (frame 0 only) -------------------------------
libvpx = {}
in_frame_zero = False
with open(libvpx_log) as f:
    for line in f:
        if line.startswith("FRAME "):
            in_frame_zero = "FRAME 0 " in line
            continue
        if not in_frame_zero or not line.startswith("MI,"):
            continue
        parts = line.strip().split(",")
        # MI,mi_row,mi_col,sb_type,segment_id,skip,tx_size,mode,uv_mode,r0,r1
        mr, mc, sb, seg, sk, tx, mo, uv, r0, r1 = map(int, parts[1:])
        libvpx[(mr, mc)] = dict(sb=sb, seg=seg, sk=sk, tx=tx, mo=mo, uv=uv)

# --- Parse rvp9 KFLEAF stream (frame 0 only) ----------------------------
# A "block" record spans several lines: enter ... after_segment_id ...
# skip_ctx=..skip=... after_skip ... tx_size=... after_tx_size ...
# y_mode=... bmi=[..] after_keyframe_y_mode ... uv_mode=... after_intra_mode(uv_mode)
re_enter = re.compile(
    r"^RVP9_DBG kfleaf mi=\((\d+),(\d+)\) bsize=(\S+) enter "
)
re_segid = re.compile(
    r"^RVP9_DBG kfleaf mi=\((\d+),(\d+)\) after read_segment_id "
)
re_skip  = re.compile(
    r"^RVP9_DBG kfleaf mi=\((\d+),(\d+)\) skip_ctx=\d+ skip=(\w+)"
)
re_tx    = re.compile(
    r"^RVP9_DBG kfleaf mi=\((\d+),(\d+)\) max_tx=\S+ tx_ctx=\d+ tx_size=(\S+)"
)
re_y     = re.compile(
    r"^RVP9_DBG kfleaf mi=\((\d+),(\d+)\) y_mode=(\S+) bmi=\[(\d+),(\d+),(\d+),(\d+)\]"
)
re_uv    = re.compile(
    r"^RVP9_DBG kfleaf mi=\((\d+),(\d+)\) uv_mode=(\S+)"
)

# We only want frame 0.  Heuristically: the first frame's KFLEAF records are
# contiguous starting at line 1 — they end when something else starts.  Use a
# crude marker: the partition_pre trace for inter frames begins much later.
# In practice the file is dominated by frame 0; just take ALL KFLEAF records.
rvp9 = {}            # (mi_row, mi_col) -> {bsize, sb_type, sk, tx, mo, uv, bmi}
last_block = None    # (mi_row, mi_col)
with open(rvp9_log) as f:
    for line in f:
        m = re_enter.match(line)
        if m:
            mr, mc = int(m.group(1)), int(m.group(2))
            bs = m.group(3)
            last_block = (mr, mc)
            rvp9.setdefault(last_block, {})
            rvp9[last_block]["bsize_name"] = bs
            rvp9[last_block]["sb"] = BSIZE.get(bs, -1)
            continue
        m = re_skip.match(line)
        if m and last_block == (int(m.group(1)), int(m.group(2))):
            rvp9[last_block]["sk"] = 1 if m.group(3) == "true" else 0
            continue
        m = re_tx.match(line)
        if m and last_block == (int(m.group(1)), int(m.group(2))):
            rvp9[last_block]["tx"] = TXSIZE.get(m.group(3), -1)
            continue
        m = re_y.match(line)
        if m and last_block == (int(m.group(1)), int(m.group(2))):
            ym = m.group(3)
            bmi3 = int(m.group(7))
            # For Block4x4: libvpx mi->mode = bmi[3]; for others: mi->mode = y_mode.
            if rvp9[last_block]["bsize_name"] == "Block4x4":
                rvp9[last_block]["mo"] = bmi3
            elif rvp9[last_block]["bsize_name"] == "Block4x8":
                # bmi[1].as_mode → mi->mode (libvpx code)
                rvp9[last_block]["mo"] = int(m.group(5))
            elif rvp9[last_block]["bsize_name"] == "Block8x4":
                # bmi[2].as_mode → mi->mode (libvpx code)
                rvp9[last_block]["mo"] = int(m.group(6))
            else:
                rvp9[last_block]["mo"] = INTRAMODE.get(ym, -1)
            continue
        m = re_uv.match(line)
        if m and last_block == (int(m.group(1)), int(m.group(2))):
            rvp9[last_block]["uv"] = INTRAMODE.get(m.group(3), -1)
            continue

# --- Compare in raster order (mi_row, mi_col).  Report first divergence.
diffs = []
for mr in range(36):  # foreman_c440 = 352x288 → mi grid 36 rows × 44 cols
    for mc in range(44):
        if (mr, mc) not in libvpx:
            continue
        L = libvpx[(mr, mc)]
        R = rvp9.get((mr, mc))
        if R is None:
            # mi cell not directly decoded as the "owner" of its block —
            # libvpx replicates mode info into every covered mi cell.  Our
            # KFLEAF trace only fires on the owner.  Skip these cells.
            continue
        for k in ("sb", "sk", "tx", "mo", "uv"):
            if L[k] != R.get(k):
                diffs.append(((mr, mc), k, L[k], R.get(k), R.get("bsize_name")))

if not diffs:
    print("PERFECT MATCH on all owner cells. (sanity check failed — this should not happen if there's a bug.)")
else:
    print(f"FIRST DIVERGENCE: mi={diffs[0][0]} field={diffs[0][1]} libvpx={diffs[0][2]} rvp9={diffs[0][3]} (rvp9 bsize={diffs[0][4]})")
    print(f"Total diff cells: {len(diffs)}")
    print("First 15 diffs:")
    for d in diffs[:15]:
        (mr, mc), k, L, R, bs = d
        print(f"  mi=({mr:2d},{mc:2d})  {k}: libvpx={L:>3}  rvp9={R}  (rvp9 sb_name={bs})")
