"""
M.7 default-probs audit: compare every libvpx default probability byte
against the Rust transcription in crates/rvp9-core/src/entropy/.

Strategy: parse all-integer braced sequences out of both C and Rust
source files (after the relevant array's declaration), flatten to a
list of ints, and assert equality.

Walks:
- `default_coef_probs_4x4..32x32` vs DEFAULT_COEF_PROBS_*
- `default_if_y_probs` / `default_if_uv_probs` vs DEFAULT_IF_Y_PROBS / DEFAULT_IF_UV_PROBS
- `default_partition_probs` vs DEFAULT_PARTITION_PROBS
- `default_inter_mode_probs` vs DEFAULT_INTER_MODE_PROBS
- `default_intra_inter_p` / `default_comp_inter_p` / `default_comp_ref_p` / `default_single_ref_p`
- `default_skip_probs` / `default_switchable_interp_prob` / `default_tx_probs`
- `vp9_kf_y_mode_prob` / `vp9_kf_uv_mode_prob` / `vp9_kf_partition_probs`
- `default_nmv_context`
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIBVPX_COMMON = ROOT / "third_party" / "libvpx" / "vp9" / "common"
ENTROPY = ROOT / "crates" / "rvp9-core" / "src" / "entropy"

INT_RE = re.compile(r"\b(\d+)\b")


def extract_after(text: str, anchor: str, end: str = ";") -> str:
    """Return the substring strictly between the first `=` after the
    first `anchor` and the next `end`. Used to slice out the
    initialiser body for one declared array — skipping the type
    decl (which contains `[INTRA_MODES-1]` etc that would otherwise
    leak constants into the ints scan)."""
    i = text.index(anchor)
    eq = text.index("=", i)
    j = text.index(end, eq)
    return text[eq + 1 : j]


def ints_in(body: str) -> list[int]:
    """All integer literals in a slice, in source order."""
    out = []
    for m in INT_RE.finditer(body):
        out.append(int(m.group(1)))
    return out


def strip_c_comments(text: str) -> str:
    """Remove // line comments and /* ... */ block comments."""
    # Block comments first (non-greedy).
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    # Line comments.
    text = re.sub(r"//[^\n]*", "", text)
    return text


def load(path: Path) -> str:
    return strip_c_comments(path.read_text(encoding="utf-8"))


def cmp(name: str, libvpx_ints: list[int], rust_ints: list[int]) -> list[str]:
    errs = []
    if len(libvpx_ints) != len(rust_ints):
        errs.append(
            f"{name}: LENGTH MISMATCH libvpx={len(libvpx_ints)} rust={len(rust_ints)}"
        )
        # Show first divergence anyway.
    for i, (a, b) in enumerate(zip(libvpx_ints, rust_ints)):
        if a != b:
            errs.append(f"{name}: byte {i}: libvpx={a} rust={b}")
            if len(errs) > 5:
                errs.append(f"{name}: (further mismatches truncated)")
                break
    return errs


def main() -> int:
    entropy_c = load(LIBVPX_COMMON / "vp9_entropy.c")
    entropymode_c = load(LIBVPX_COMMON / "vp9_entropymode.c")
    entropymv_c = load(LIBVPX_COMMON / "vp9_entropymv.c")

    coef_rs = load(ENTROPY / "coef_probs.rs")
    default_rs = load(ENTROPY / "default_probs.rs")
    kf_rs = load(ENTROPY / "kf_probs.rs")
    mv_rs = load(ENTROPY / "mv_probs.rs")

    errs: list[str] = []

    # --- coef probs (four tables) ---
    coef_specs = [
        ("default_coef_probs_4x4", "DEFAULT_COEF_PROBS_4X4"),
        ("default_coef_probs_8x8", "DEFAULT_COEF_PROBS_8X8"),
        ("default_coef_probs_16x16", "DEFAULT_COEF_PROBS_16X16"),
        ("default_coef_probs_32x32", "DEFAULT_COEF_PROBS_32X32"),
    ]
    for c_name, rs_name in coef_specs:
        c_body = extract_after(entropy_c, c_name, "};")
        rs_body = extract_after(coef_rs, rs_name, "];")
        # Coef tables in C have band-0 implicit zero-padding (only 3
        # rows specified, but type is [6]). Our Rust mirrors with
        # explicit [0,0,0] zeros. So Rust will have MORE ints. We need
        # to expand the C side: for each of plane × ref_type, band 0
        # only has 3 ctx rows; the C code omits the next 3. Add 9 zeros
        # per (plane,ref_type) immediately after the band-0 rows.
        # Easier: walk and expand.
        # We'll parse the C body as 432 floats by reinserting the zeros.
        # Logic: extract ints; we expect 3*3 + 6*3*5 = 9 + 90 = 99 per
        # (plane, ref_type); 4 of those = 396 ints. The Rust side has
        # 432 with 36 extra zeros.
        c_ints = ints_in(c_body)
        rs_ints = ints_in(rs_body)
        # Reshape C: 4 (plane,ref) sub-blocks of 99 ints each.
        if len(c_ints) != 4 * 99:
            errs.append(
                f"{c_name}: expected 4*99=396 ints in libvpx body, got {len(c_ints)}"
            )
            continue
        expanded = []
        for k in range(4):
            chunk = c_ints[k * 99 : (k + 1) * 99]
            # First 9 ints = band 0 (3 ctx × 3 nodes); insert 9 zeros after.
            expanded.extend(chunk[:9])
            expanded.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])
            expanded.extend(chunk[9:])
        errs.extend(cmp(f"{c_name} vs {rs_name}", expanded, rs_ints))

    # --- small mode probs ---
    pairs = [
        ("default_if_y_probs", "DEFAULT_IF_Y_PROBS"),
        ("default_if_uv_probs", "DEFAULT_IF_UV_PROBS"),
        ("default_partition_probs", "DEFAULT_PARTITION_PROBS"),
        ("default_inter_mode_probs", "DEFAULT_INTER_MODE_PROBS"),
        ("default_intra_inter_p", "DEFAULT_INTRA_INTER_PROB"),
        ("default_comp_inter_p", "DEFAULT_COMP_INTER_PROB"),
        ("default_comp_ref_p", "DEFAULT_COMP_REF_PROB"),
        ("default_single_ref_p", "DEFAULT_SINGLE_REF_PROB"),
        ("default_skip_probs", "DEFAULT_SKIP_PROBS"),
        ("default_switchable_interp_prob", "DEFAULT_SWITCHABLE_INTERP_PROB"),
        ("default_tx_probs", "DEFAULT_TX_PROBS"),
    ]
    for c_name, rs_name in pairs:
        c_end = "};" if "tx_probs" in c_name else ";"
        rs_end = "};" if rs_name == "DEFAULT_TX_PROBS" else "];"
        # Some 1-D Rust statics end with `;` (`[u8; N] = [..];`).
        # extract_after takes the substring from `=` to first `end`.
        c_body = extract_after(entropymode_c, c_name, c_end)
        rs_body = extract_after(default_rs, rs_name, rs_end)
        c_ints = ints_in(c_body)
        rs_ints = ints_in(rs_body)
        errs.extend(cmp(f"{c_name} vs {rs_name}", c_ints, rs_ints))

    # --- keyframe mode probs ---
    kf_pairs = [
        ("vp9_kf_y_mode_prob", "VP9_KF_Y_MODE_PROB"),
        ("vp9_kf_uv_mode_prob", "VP9_KF_UV_MODE_PROB"),
        ("vp9_kf_partition_probs", "VP9_KF_PARTITION_PROBS"),
    ]
    for c_name, rs_name in kf_pairs:
        c_body = extract_after(entropymode_c, c_name, "};")
        rs_body = extract_after(kf_rs, rs_name, "];")
        errs.extend(cmp(f"{c_name} vs {rs_name}", ints_in(c_body), ints_in(rs_body)))

    # --- CAT probs + coefband_trans tables (consumed inside decode_coefs) ---
    detokenize_rs = load(ENTROPY / "detokenize.rs")
    cat_pairs = [
        ("vp9_cat1_prob", "VP9_CAT1_PROB"),
        ("vp9_cat2_prob", "VP9_CAT2_PROB"),
        ("vp9_cat3_prob", "VP9_CAT3_PROB"),
        ("vp9_cat4_prob", "VP9_CAT4_PROB"),
        ("vp9_cat5_prob", "VP9_CAT5_PROB"),
        ("vp9_cat6_prob", "VP9_CAT6_PROB"),
        ("vp9_coefband_trans_4x4", "VP9_COEFBAND_TRANS_4X4"),
    ]
    for c_name, rs_name in cat_pairs:
        c_body = extract_after(entropy_c, c_name, ";")
        rs_body = extract_after(detokenize_rs, rs_name, "];")
        errs.extend(cmp(f"{c_name} vs {rs_name}", ints_in(c_body), ints_in(rs_body)))

    # --- nmv defaults ---
    c_body = extract_after(entropymv_c, "default_nmv_context", "};")
    rs_body = extract_after(mv_rs, "DEFAULT_NMV_CONTEXT", "};")
    # Strip any "MV_OFFSET_BITS" / "CLASS0_SIZE" / "MV_FP_SIZE" / "MV_JOINTS"
    # / "MV_CLASSES" symbols that appear as integer-like in the Rust
    # type position. Easiest: strip all-uppercase identifiers via regex
    # before extracting ints.
    rs_body_clean = re.sub(r"\b[A-Z][A-Z0-9_]+\b", "", rs_body)
    errs.extend(cmp("default_nmv_context vs DEFAULT_NMV_CONTEXT", ints_in(c_body), ints_in(rs_body_clean)))

    if errs:
        print(f"FOUND {len(errs)} MISMATCH(ES):")
        for e in errs:
            print(f"  {e}")
        return 1
    else:
        print("All default probability tables match libvpx exactly.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
