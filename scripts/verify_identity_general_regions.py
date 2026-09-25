"""Exact (rational) test of P_H = P_V = P1P2 + P3P4 on arbitrary regions of Z^2,
by direct enumeration of domino tilings -- independent of any Kasteleyn matrix."""
from fractions import Fraction
from functools import lru_cache
import random

def count(cells):
    """Number of domino tilings of a finite set of cells (row-major DP with memo)."""
    cells = frozenset(cells)
    order = sorted(cells)
    @lru_cache(None)
    def f(covered):
        # first uncovered cell in row-major order
        for c in order:
            if c not in covered:
                break
        else:
            return 1
        i, j = c
        tot = 0
        for d in ((i, j + 1), (i + 1, j)):
            if d in cells and d not in covered:
                tot += f(covered | {c, d})
        return tot
    return f(frozenset())

def check(cells, label):
    cells = set(cells); M = count(cells)
    worst, nsq = 0, 0
    for (i, j) in cells:
        sq = [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]
        if not all(c in cells for c in sq):
            continue
        a, b, c, d = sq   # a=(i,j) b=(i,j+1) c=(i+1,j) d=(i+1,j+1)
        P = lambda *rm: Fraction(count(cells - set(rm)), M)
        P1, P2 = P(a, b), P(c, d)          # two horizontal edges
        P3, P4 = P(a, c), P(b, d)          # two vertical edges
        PH, PV = P(a, b, c, d), P(a, b, c, d)   # same cell set; see note below
        # P_H and P_V both equal M(G - square)/M(G) because removing the four
        # cells leaves the same region: this IS the flip bijection H <-> V.
        assert PH == P1 * P2 + P3 * P4, (label, (i, j), PH, P1*P2 + P3*P4)
        nsq += 1
    print(f"{label:38s} tilings={M:>8d}  squares checked={nsq:3d}  identity exact: OK")

check([(i, j) for i in range(4) for j in range(4)], "4x4 chessboard")
check([(i, j) for i in range(6) for j in range(6)], "6x6 chessboard")
check([(i, j) for i in range(4) for j in range(7)], "4x7 rectangle")
check([(i, j) for i in range(6) for j in range(6) if (i, j) not in {(2, 2), (2, 3), (3, 2), (3, 3)}],
      "6x6 with central 2x2 hole (annulus)")
random.seed(3)
for t in range(3):
    while True:
        R = {(i, j) for i in range(6) for j in range(6) if random.random() < 0.85}
        if count(R) > 0: break
    check(R, f"random region #{t+1} ({len(R)} cells)")
