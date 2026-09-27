"""Exact-rational arithmetic checks for the proposed 8-bit TX4/TX8 bound."""

from fractions import Fraction as F
from pathlib import Path
import math
import re

tables = Path("crates/rvp9-core/src/quant/tables.rs").read_text()
constants = Path("crates/rvp9-core/src/transforms/constants.rs").read_text()


def table(name):
    body = re.search(r"pub static " + name + r": \[i16; 256\] = \[(.*?)\];", tables, re.S).group(1)
    values = [int(x) for x in re.findall(r"\d+", body)]
    assert len(values) == 256
    return values


def cosine(index):
    return int(re.search(r"pub const COSPI_" + str(index) + r"_64: i32 = (\d+);", constants).group(1))


DC, AC = table("DC_QLOOKUP"), table("AC_QLOOKUP")
C4, C8, C12, C16, C20, C24, C28 = (cosine(i) for i in (4, 8, 12, 16, 20, 24, 28))
Q = F(1, 1 << 14)


def lin8(a):
    s = [a[0] + a[7], a[1] + a[6], a[2] + a[5], a[3] + a[4], a[3] - a[4], a[2] - a[5], a[1] - a[6], a[0] - a[7]]
    x0, x1, x2, x3 = s[0] + s[3], s[1] + s[2], s[1] - s[2], s[0] - s[3]
    t2, t3 = (s[6] - s[5]) * C16 * Q, (s[6] + s[5]) * C16 * Q
    u0, u1, u2, u3 = s[4] + t2, s[4] - t2, s[7] - t3, s[7] + t3
    return [(x0 + x1) * C16 * Q, (u0 * C28 + u3 * C4) * Q, (x2 * C24 + x3 * C8) * Q, (u2 * C12 - u1 * C20) * Q,
            (x0 - x1) * C16 * Q, (u1 * C12 + u2 * C20) * Q, (-x2 * C8 + x3 * C24) * Q, (u3 * C28 - u0 * C4) * Q]


def lin4(a):
    s0, s1, s2, s3 = a[0] + a[3], a[1] + a[2], a[1] - a[2], a[0] - a[3]
    return [(s0 + s1) * C16 * Q, (s2 * C24 + s3 * C8) * Q,
            (s0 - s1) * C16 * Q, (-s2 * C8 + s3 * C24) * Q]


def rows(transform, n):
    columns = [transform([F(int(i == j)) for i in range(n)]) for j in range(n)]
    return [[columns[j][k] for j in range(n)] for k in range(n)]


D8, D4 = rows(lin8, 8), rows(lin4, 4)
n2 = lambda row: sum(x * x for x in row)
n1 = lambda row: sum(abs(x) for x in row)
N8, N4, half = max(map(n2, D8)), max(map(n2, D4)), F(1, 2)
odd_outer = half + F(C28 + C4, 1 << 15)
odd_inner = half + F(C12 + C20, 1 << 15)
eta8 = [half, odd_outer, half, odd_inner, half, odd_inner, half, odd_outer]
B8 = max(eta8[k] * n1(D8[m]) + eta8[m] for k in range(8) for m in range(8)) / 2
B4 = (max(abs(D4[k][0] * D4[m][0]) for k in range(4) for m in range(4))
      + max(half * n1(row) + half for row in D4) + 2) / 4
assert 4 * N8 * N8 <= 65 and 16 * N4 * N4 <= 65 and B8 <= 4 and B4 <= 4


def threshold(quant, factor):
    qfp = (1 << 16) // quant
    rounding = (factor * quant) >> 7
    result = -((-65536) // qfp) - rounding
    assert all(((min(a + rounding, 32767)) * qfp) >> 16 == 0 for a in range(result))
    assert ((min(result + rounding, 32767)) * qfp) >> 16 != 0 and result <= 32767
    return result


same_q_thresholds = [min(threshold(DC[q], 64 if q == 0 else 48),
                         threshold(AC[q], 64 if q == 0 else 42)) for q in range(256)]
# DC and AC header deltas clamp independently; check the largest min(Tdc, Tac)
# across their complete 8-bit table ranges, for base q=0 and base q>0.
max_independent_threshold = max(
    min(max(threshold(value, dc_factor) for value in DC),
        max(threshold(value, ac_factor) for value in AC))
    for dc_factor, ac_factor in ((64, 64), (48, 42))
)
assert max_independent_threshold == max(same_q_thresholds) == 837

max_sse = ((max_independent_threshold - 4) ** 2 - 1) // 65
max_residual = math.isqrt(max_sse)
up = lambda z: -((-z) // 16384) + 1


def chain8(a):
    s = 2 * a
    x = 2 * s
    xx = 2 * x
    t = up(x * C16)
    u = s + t
    return max(xx, u), max(up(xx * C16), up(x * (C24 + C8)),
                        up(u * (C28 + C4)), up(u * (C12 + C20)))


i0, o0 = chain8(4 * max_residual)
i1, o1 = chain8(o0)
a = 16 * max_residual + 1
p0 = max(up(4 * a * C16), up(2 * a * (C8 + C24)))
p1 = max(up(4 * p0 * C16), up(2 * p0 * (C8 + C24)))
assert max(i0, o0, i1, o1, p0, p1) <= 32767 and o0 <= 4095

print("N8", N8, "N4", N4, "B8", float(B8), "B4", float(B4))
print("Tmax", max_independent_threshold, "Smax", max_sse, "xmax", max_residual,
      "8x8 ranges", i0, o0, i1, o1, "4x4 ranges", p0, p1)
print("ALL CHECKS PASS")
