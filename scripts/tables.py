"""
tables.py
---------
Reproduce the numbers in Tables I-III of the paper.

  Table I   : central edge probability P_e(n) and central P_H = 2 P_e(n)^2,
              exact (Schoenfelder's formula) and from the inverse Kasteleyn matrix.
  Table II  : maxima of P_H over black-anchored and white-anchored 2x2 squares
              of AZ(n), 4 <= n <= 17.
  Table III : sublattice contrast Delta (mean P_H over black-anchored squares
              within distance 3 of the centre minus the same mean over
              white-anchored squares) for AZ(h) and the 2h x 2h chessboard.

Conventions (as in all scripts): cell (i, j) = [j, j+1] x [i, i+1], i = row;
black if i + j even; K[b, w] = +1 (right), -1 (left), +i (up), -i (down).

Usage:  python tables.py            (all tables; takes about a minute)
        python tables.py 1          (only Table I; likewise 2 or 3)
"""

import sys
from fractions import Fraction
from math import comb

import numpy as np


# ── lattice regions ─────────────────────────────────────────────────────────

def aztec(n):
    return [(i, j) for i in range(-n, n) for j in range(-n, n)
            if abs(i + 0.5) + abs(j + 0.5) <= n]


def board(L):
    return [(i, j) for i in range(L) for j in range(L)]


# ── Kasteleyn machinery ─────────────────────────────────────────────────────

def kasteleyn(cells):
    B = [c for c in cells if (c[0] + c[1]) % 2 == 0]
    W = [c for c in cells if (c[0] + c[1]) % 2 == 1]
    bm = {b: k for k, b in enumerate(B)}
    wm = {w: k for k, w in enumerate(W)}
    K = np.zeros((len(B), len(W)), complex)
    for b, bi in bm.items():
        for (di, dj), x in [((0, 1), 1), ((0, -1), -1), ((1, 0), 1j), ((-1, 0), -1j)]:
            w = (b[0] + di, b[1] + dj)
            if w in wm:
                K[bi, wm[w]] = x
    return K, np.linalg.inv(K), bm, wm


def _bw(c1, c2):
    return (c1, c2) if (c1[0] + c1[1]) % 2 == 0 else (c2, c1)


def edge_prob(c1, c2, K, Ki, bm, wm):
    b, w = _bw(c1, c2)
    return float((K[bm[b], wm[w]] * Ki[wm[w], bm[b]]).real)


def p_h(anchor, K, Ki, bm, wm):
    """P_H of the 2x2 square with corner cell `anchor`, from the two-edge determinant."""
    i, j = anchor
    pairs = [_bw((i, j), (i, j + 1)), _bw((i + 1, j), (i + 1, j + 1))]
    bs = [p[0] for p in pairs]; ws = [p[1] for p in pairs]
    if any(x not in bm for x in bs) or any(x not in wm for x in ws):
        return None
    pref = np.prod([K[bm[b], wm[w]] for b, w in pairs])
    M = np.array([[Ki[wm[w], bm[b]] for b in bs] for w in ws])
    return float((pref * np.linalg.det(M)).real)


def all_squares(cells):
    K, Ki, bm, wm = kasteleyn(cells)
    cs = set(cells)
    out = {}
    for (i, j) in cells:
        if all(c in cs for c in [(i, j + 1), (i + 1, j), (i + 1, j + 1)]):
            out[(i, j)] = p_h((i, j), K, Ki, bm, wm)
    return out, (K, Ki, bm, wm)


# ── Table I ─────────────────────────────────────────────────────────────────

def exact_pe(n):
    p, a = divmod(n, 4)
    c = Fraction(comb(2 * p - 1, p) ** 2, 2 ** (4 * p)) if p >= 1 else Fraction(0)
    return {0: Fraction(1, 4) - c, 1: Fraction(1, 4) + c,
            2: Fraction(1, 4), 3: Fraction(1, 4)}[a]


def table1():
    print('Table I: central edge and central 2x2 probabilities')
    print(f"{'n':>3} {'n%4':>4} {'P_e exact':>12} {'P_H exact':>14} {'|P_e(Kast)-exact|':>18}")
    for n in [4, 5, 6, 7, 8, 9, 12, 13]:
        K, Ki, bm, wm = kasteleyn(aztec(n))
        pe_num = edge_prob((-1, -1), (-1, 0), K, Ki, bm, wm)
        pe = exact_pe(n)
        print(f"{n:>3} {n % 4:>4} {str(pe):>12} {str(2 * pe * pe):>14} {abs(pe_num - float(pe)):>18.1e}")


# ── Table II ────────────────────────────────────────────────────────────────

def table2(ns=range(4, 18), tol=1e-10):
    print('Table II: maxima of P_H by sublattice (representative = smallest corner cell)')
    print(f"{'n':>3} {'n%4':>4} | {'black max':>11} {'at':>9} {'#':>2} | "
          f"{'white max':>11} {'at':>9} {'#':>2} | {'central':>11}")
    for n in ns:
        sq, _ = all_squares(aztec(n))
        rows = []
        for parity in (0, 1):
            vals = {a: v for a, v in sq.items() if (a[0] + a[1]) % 2 == parity}
            m = max(vals.values())
            at = sorted(a for a, v in vals.items() if abs(v - m) < tol)
            rows.append((m, at[0], len(at)))
        (mb, ab, kb), (mw, aw, kw) = rows
        print(f"{n:>3} {n % 4:>4} | {mb:11.9f} {str(ab):>9} {kb:>2} | "
              f"{mw:11.9f} {str(aw):>9} {kw:>2} | {sq[(-1, -1)]:11.9f}")


# ── Table III ───────────────────────────────────────────────────────────────

def contrast(cells, centre, radius=3.0):
    sq, _ = all_squares(cells)
    B, W = [], []
    for (i, j), v in sq.items():
        d = np.hypot(i + 1 - centre[0], j + 1 - centre[1])   # square centre = vertex (i+1, j+1)
        if d <= radius:
            (B if (i + j) % 2 == 0 else W).append(v)
    return np.mean(B) - np.mean(W)


def table3(hs=range(4, 16)):
    print('Table III: sublattice contrast Delta (radius 3)')
    print(f"{'h':>3} {'Delta AZ(h)':>12} {'Delta 2h x 2h':>14}")
    for h in hs:
        da = contrast(aztec(h), (0, 0))
        dc = contrast(board(2 * h), (h, h))
        print(f"{h:>3} {da:+12.4f} {dc:+14.4f}")


if __name__ == '__main__':
    which = sys.argv[1:] or ['1', '2', '3']
    for t in which:
        {'1': table1, '2': table2, '3': table3}[t]()
        print()
