"""
gen_fig2_background.py
----------------------
Generate fig2_background.pdf: the notation and the known results of Section 2.

  (a) The two sublattices of elementary 2x2 squares in AZ(4): the center of
      every square is marked by a filled circle (black-anchored, corner cell
      (a,b) with a+b even) or an open square (white-anchored).  The central
      square is outlined.
  (b) Notation for one black-anchored square: cells b1, w1, w2, b2, the four
      edge probabilities P1..P4 and the Kasteleyn phases K[b,w]; the H pair is
      {b1w1, b2w2} (horizontal), the V pair {b1w2, b2w1} (vertical).
  (c) Central edge probability P_e(n) - 1/4 for 2 <= n <= 24 from the inverse
      Kasteleyn matrix (markers), grouped by n mod 4, with Schoenfelder's exact
      formula (crosses).
  (d) Central P_H = 2 P_e(n)^2 with the reference value 1/8
      (exact for n = 2, 3 mod 4: Ciucu).

Conventions as in all scripts: cell (i,j) = [j,j+1] x [i,i+1]; black if i+j even;
K[b,w] = +1 (right), -1 (left), +i (up), -i (down).

Output: fig2_background.pdf
"""

import numpy as np
from math import comb
import matplotlib
matplotlib.use('Agg')
import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection


def aztec_cells(n):
    return [(i, j) for i in range(-n, n) for j in range(-n, n)
            if abs(i + 0.5) + abs(j + 0.5) <= n]


def boundary_segs(cells_set):
    segs = []
    for (i, j) in cells_set:
        if (i, j + 1) not in cells_set: segs.append([(j + 1, i), (j + 1, i + 1)])
        if (i, j - 1) not in cells_set: segs.append([(j, i), (j, i + 1)])
        if (i + 1, j) not in cells_set: segs.append([(j, i + 1), (j + 1, i + 1)])
        if (i - 1, j) not in cells_set: segs.append([(j, i), (j + 1, i)])
    return segs


def central_pe(n):
    cells = aztec_cells(n)
    B = [c for c in cells if sum(c) % 2 == 0]; W = [c for c in cells if sum(c) % 2]
    bm = {b: k for k, b in enumerate(B)}; wm = {w: k for k, w in enumerate(W)}
    K = np.zeros((len(B), len(W)), complex)
    for b, bi in bm.items():
        for (di, dj), x in [((0, 1), 1), ((0, -1), -1), ((1, 0), 1j), ((-1, 0), -1j)]:
            w = (b[0] + di, b[1] + dj)
            if w in wm: K[bi, wm[w]] = x
    Ki = np.linalg.inv(K)
    b, w = (-1, -1), (-1, 0)
    return float((K[bm[b], wm[w]] * Ki[wm[w], bm[b]]).real)


def exact_pe(n):
    p, a = divmod(n, 4)
    c = 2.0 ** (-4 * p) * comb(2 * p - 1, p) ** 2 if p >= 1 else 0.0
    return {0: 0.25 - c, 1: 0.25 + c, 2: 0.25, 3: 0.25}[a]


COL = {0: '#c62828', 1: '#1565c0', 2: '#2e7d32', 3: '#ef6c00'}
MRK = {0: 's', 1: 'o', 2: '^', 3: 'D'}
C_BLACK_CELL, C_WHITE_CELL = '#d9d9d9', '#ffffff'

fig = plt.figure(figsize=(12.0, 9.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.45, wspace=0.28)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

# ── (a) sublattices on AZ(4) ────────────────────────────────────────────────
n = 4
cells = set(aztec_cells(n))
for (i, j) in cells:
    axA.add_patch(Rectangle((j, i), 1, 1, lw=0.4, edgecolor='0.75',
                            facecolor=C_BLACK_CELL if (i + j) % 2 == 0 else C_WHITE_CELL))
axA.add_collection(LineCollection(boundary_segs(cells), colors='k', linewidths=1.2))
for i in range(-n, n):
    for j in range(-n, n):
        if all(c in cells for c in [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]):
            x, y = j + 1, i + 1
            if (i + j) % 2 == 0:
                axA.scatter(x, y, s=46, marker='o', color='k', zorder=4)
            else:
                axA.scatter(x, y, s=46, marker='s', facecolor='white', edgecolor='k',
                            linewidth=1.3, zorder=4)
axA.add_patch(Rectangle((-1, -1), 2, 2, fill=False, edgecolor='#1976d2', lw=2.4, zorder=5))
axA.scatter([], [], s=46, marker='o', color='k', label='black-anchored square (center)')
axA.scatter([], [], s=46, marker='s', facecolor='white', edgecolor='k', linewidth=1.3,
            label='white-anchored square (center)')
axA.plot([], [], color='#1976d2', lw=2.4, label='central square')
axA.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5,
           frameon=False, columnspacing=1.2)
axA.set_xlim(-n - 0.4, n + 0.4); axA.set_ylim(-n - 0.4, n + 0.4); axA.set_aspect('equal')
axA.set_xticks(range(-n, n + 1, 2)); axA.set_yticks(range(-n, n + 1, 2))
axA.set_xlabel('$j$'); axA.set_ylabel('$i$')
axA.set_title(r'(a) Two sublattices of $2\times2$ squares, $\mathrm{AZ}(4)$', fontsize=11)

# ── (b) notation for one square ─────────────────────────────────────────────
ax = axB
lab = {(0, 0): ('$b_1$', C_BLACK_CELL), (0, 1): ('$w_1$', C_WHITE_CELL),
       (1, 0): ('$w_2$', C_WHITE_CELL), (1, 1): ('$b_2$', C_BLACK_CELL)}
for (r, c), (t, fc) in lab.items():          # r = row from bottom, c = column
    ax.add_patch(Rectangle((c, r), 1, 1, facecolor=fc, edgecolor='0.4', lw=1.0))
    # label near the outer corner of the cell, away from the edges drawn below
    ax.text(c + (0.2 if c == 0 else 0.8), r + (0.2 if r == 0 else 0.8), t,
            ha='center', va='center', fontsize=15)
cH, cV = '#d32f2f', '#1976d2'
edges = [  # (x0,y0,x1,y1, color, label, label position)
    (0.5, 0.5, 1.5, 0.5, cH, r'$P_1,\ K=+1$', (1.0, 0.30)),
    (0.5, 1.5, 1.5, 1.5, cH, r'$P_2,\ K=-1$', (1.0, 1.70)),
    (0.5, 0.5, 0.5, 1.5, cV, r'$P_3$' + '\n' + r'$K=+i$', (0.22, 1.0)),
    (1.5, 0.5, 1.5, 1.5, cV, r'$P_4$' + '\n' + r'$K=-i$', (1.78, 1.0)),
]
for x0, y0, x1, y1, col, t, (tx, ty) in edges:
    ax.plot([x0, x1], [y0, y1], color=col, lw=4.0, solid_capstyle='round', zorder=3,
            alpha=0.85)
    ax.text(tx, ty, t, ha='center', va='center', fontsize=10.5, color=col,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none',
                      alpha=0.9), zorder=4)
ax.text(1.0, -0.28, r'$H=\{b_1w_1,\,b_2w_2\}$ (red),  $V=\{b_1w_2,\,b_2w_1\}$ (blue)',
        ha='center', va='top', fontsize=10.5)
ax.text(1.0, -0.58, r'$P_H=P_V=P_1P_2+P_3P_4$', ha='center', va='top', fontsize=12)
ax.set_xlim(-0.35, 2.35); ax.set_ylim(-0.85, 2.15); ax.set_aspect('equal'); ax.axis('off')
ax.set_title(r'(b) Notation for a black-anchored square', fontsize=11)

# ── (c) central edge probability ────────────────────────────────────────────
ns = list(range(2, 25))
pe = [central_pe(k) for k in ns]
axC.axhline(0, color='0.4', ls='--', lw=1.0)
for m in range(4):
    xs = [k for k in ns if k % 4 == m]
    axC.scatter(xs, [pe[ns.index(k)] - 0.25 for k in xs], s=48, marker=MRK[m],
                color=COL[m], edgecolor='white', linewidth=0.5, zorder=3,
                label=rf'$n\equiv{m}\ (\mathrm{{mod}}\ 4)$')
axC.scatter(ns, [exact_pe(k) - 0.25 for k in ns], s=26, marker='x', color='k',
            linewidth=0.9, zorder=4, label='exact formula [Schönfelder]')
axC.set_xlabel('$n$'); axC.set_ylabel(r'$P_e(n)-1/4$')
axC.set_xticks(range(2, 25, 2)); axC.grid(True, alpha=0.25); axC.set_ylim(-0.072, 0.115)
axC.legend(fontsize=8, loc='upper right', ncol=1, framealpha=0.92)
axC.set_title(r'(c) Central edge: the mod-4 structure', fontsize=11)

# ── (d) central P_H ─────────────────────────────────────────────────────────
ph = [2 * p ** 2 for p in pe]
axD.axhline(0.125, color='0.4', ls='--', lw=1.0,
            label=r'$1/8$ (exact for $n\equiv2,3$ [Ciucu])')
for m in range(4):
    xs = [k for k in ns if k % 4 == m]
    axD.scatter(xs, [ph[ns.index(k)] for k in xs], s=48, marker=MRK[m], color=COL[m],
                edgecolor='white', linewidth=0.5, zorder=3,
                label=rf'$n\equiv{m}\ (\mathrm{{mod}}\ 4)$')
axD.set_xlabel('$n$'); axD.set_ylabel(r'central $P_H=2P_e(n)^2$')
axD.set_xticks(range(2, 25, 2)); axD.grid(True, alpha=0.25)
axD.legend(fontsize=8, loc='upper right', framealpha=0.92)
axD.set_title(r'(d) Central $2\times2$ square', fontsize=11)

plt.savefig(os.path.join(FIGDIR, 'fig2_background.pdf'), bbox_inches='tight')
plt.close()
print('Saved fig2_background.pdf')
print('max |Kasteleyn - exact| for P_e, n=2..24:',
      max(abs(p - exact_pe(k)) for k, p in zip(ns, pe)))
