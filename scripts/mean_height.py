"""
mean_height.py
--------------
Height-function check used in Sec. IV of the paper.

  1. Expected central height <h(0,0)> of AZ(n) under the uniform measure,
     computed exactly from single-edge probabilities (inverse Kasteleyn matrix)
     and compared with Eq. (13), <h(0,0)> = (-1)^(n+1) n.
  2. Residue r_n = h(0,0) mod 4 (from the all-horizontal tiling) and the
     offset (<h(0,0)> - r_n) mod 4.
  3. Mean heights of the four basketweave placements on a large patch,
     measured from the residue of the vertex (0,0).

Height convention (Thurston): h = 0 at the west corner (-n, 0).  A unit step
with a black cell on its left changes h by +1 along a domino boundary and by
-3 across a domino; with a white cell on its left the signs are reversed.
Cell (i, j) = [j, j+1] x [i, i+1]; vertex (x, y) = (j, i); black if i + j even.

Usage:  python mean_height.py
"""
from collections import deque

import numpy as np

from tables import aztec, kasteleyn, edge_prob

STEPS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def colour(c):
    return 1 if (c[0] + c[1]) % 2 == 0 else -1


def left_right(v, d):
    """Cells to the left and right of the unit step v -> v + d."""
    x, y = v
    if d == (1, 0):
        return (y, x), (y - 1, x)
    if d == (-1, 0):
        return (y - 1, x - 1), (y, x - 1)
    if d == (0, 1):
        return (y, x - 1), (y, x)
    return (y - 1, x), (y - 1, x - 1)


def integrate(start, dh, inside):
    """Breadth-first sum of step increments dh(v, d); returns heights and the
    largest closed-loop discrepancy."""
    H = {start: 0.0}
    q = deque([start])
    err = 0.0
    while q:
        v = q.popleft()
        for d in STEPS:
            u = (v[0] + d[0], v[1] + d[1])
            if not inside(v, d, u):
                continue
            hv = H[v] + dh(v, d)
            if u in H:
                err = max(err, abs(H[u] - hv))
            else:
                H[u] = hv
                q.append(u)
    return H, err


def aztec_heights(n, crossing):
    """Heights on AZ(n); crossing(L, R) = probability (or indicator) that a
    domino covers the two cells on either side of the step."""
    S = set(aztec(n))

    def inside(v, d, u):
        L, R = left_right(v, d)
        return L in S or R in S

    def dh(v, d):
        L, R = left_right(v, d)
        p = crossing(L, R) if (L in S and R in S) else 0.0
        return colour(L) * (1 - 4 * p)

    return integrate((-n, 0), dh, inside)


def mean_central_height(n):
    cells = aztec(n)
    K, Ki, bm, wm = kasteleyn(cells)
    H, err = aztec_heights(n, lambda L, R: edge_prob(L, R, K, Ki, bm, wm))
    return H[(0, 0)], err


def central_residue(n):
    """h(0,0) mod 4 from the all-horizontal tiling (every row has even length)."""
    def partner(c):
        i, j = c
        j0 = int(-(n - abs(i + 0.5) + 0.5))
        return (i, j + 1 if (j - j0) % 2 == 0 else j - 1)
    H, _ = aztec_heights(n, lambda L, R: 1.0 if partner(L) == R else 0.0)
    return int(round(H[(0, 0)])) % 4


def basketweave_mean(p, q, swap=False, W=60):
    """Mean height of the basketweave whose 2x2 blocks have corner cells
    (a, b) = (p, q) mod 2, on the patch |x|, |y| <= W, with h(0,0) = 0."""
    def partner(c):
        i, j = c
        a, b = i - (i - p) % 2, j - (j - q) % 2
        horiz = ((((a - p) // 2) + ((b - q) // 2)) % 2 == 0) ^ swap
        return (i, b + 1 if j == b else b) if horiz else (a + 1 if i == a else a, j)

    def dh(v, d):
        L, R = left_right(v, d)
        return colour(L) * (-3 if partner(L) == R else 1)

    def inside(v, d, u):
        return max(abs(u[0]), abs(u[1])) <= W

    H, _ = integrate((0, 0), dh, inside)
    return float(np.mean(list(H.values())))


if __name__ == '__main__':
    print('Central mean height of AZ(n)')
    print(f"{'n':>3} {'n%4':>4} {'<h(0,0)>':>12} {'(-1)^(n+1)n':>12} {'r_n':>4} "
          f"{'offset':>7} {'loop err':>9}")
    for n in range(4, 23):
        h0, err = mean_central_height(n)
        r = central_residue(n)
        print(f"{n:>3} {n % 4:>4} {h0:12.8f} {(-1) ** (n + 1) * n:>12} {r:>4} "
              f"{(round(h0) - r) % 4:>7} {err:9.1e}")
    print()
    print('Basketweave placements: mean height mod 4 (from residue at (0,0))')
    for (p, q) in [(0, 0), (1, 1), (1, 0), (0, 1)]:
        cls = 'black' if (p + q) % 2 == 0 else 'white'
        vals = [basketweave_mean(p, q, s) % 4 for s in (False, True)]
        print(f"  corner (a,b) = ({p},{q}) mod 2  [{cls}-anchored]: "
              f"{vals[0]:.3f}, swapped {vals[1]:.3f}")
