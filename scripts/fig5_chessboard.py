"""
gen_fig5_chessboard.py
----------------------
Generate fig5_chessboard.pdf: comparison with the square L x L chessboard.

  (a) Central edge probability of the even L x L board, scaled as L (P_e - 1/4),
      grouped by the parity of m = L/2 (i.e. L mod 4).  For odd L the board has
      no tiling; with the central cell removed ("central monomer") the radial
      edge probability next to the hole, P_rad, is shown on the right axis.
  (b), (c) P_H over all elementary 2x2 squares of the 20x20 (m even) and
      22x22 (m odd) boards, drawn in the style of Figure 4(a).

Kasteleyn weights: 1 for horizontal (j) steps, i for vertical (i) steps, as in
the other scripts.  For the holed odd boards the sign of every horizontal edge
crossing the vertical lattice line x = k+1 above the hole is flipped, which
restores the Kasteleyn condition on the 8-cycle face around the hole
(verified against exact enumeration for L = 3, 5, 7, 9).

Output: fig5_chessboard.pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection


def build(cells, hole=None):
    cells = list(cells)
    B = [c for c in cells if (c[0] + c[1]) % 2 == 0]
    W = [c for c in cells if (c[0] + c[1]) % 2 == 1]
    bm = {b: n for n, b in enumerate(B)}; wm = {w: n for n, w in enumerate(W)}
    K = np.zeros((len(B), len(W)), complex)
    for b, bi in bm.items():
        for (di, dj), x in [((0, 1), 1), ((0, -1), -1), ((1, 0), 1j), ((-1, 0), -1j)]:
            w = (b[0] + di, b[1] + dj)
            if w in wm:
                if hole is not None and di == 0 and b[0] > hole[0] \
                        and {b[1], w[1]} == {hole[1], hole[1] + 1}:
                    x = -x
                K[bi, wm[w]] = x
    return K, np.linalg.inv(K), bm, wm


def _bw(c1, c2):
    return (c1, c2) if (c1[0] + c1[1]) % 2 == 0 else (c2, c1)


def edge_prob(c1, c2, K, Ki, bm, wm):
    b, w = _bw(c1, c2)
    return float((K[bm[b], wm[w]] * Ki[wm[w], bm[b]]).real)


def ph(anchor, K, Ki, bm, wm):
    i, j = anchor
    pairs = [_bw((i, j), (i, j + 1)), _bw((i + 1, j), (i + 1, j + 1))]
    bs = [p[0] for p in pairs]; ws = [p[1] for p in pairs]
    if any(x not in bm for x in bs) or any(x not in wm for x in ws):
        return None
    pref = np.prod([K[bm[b], wm[w]] for b, w in pairs])
    M = np.array([[Ki[wm[w], bm[b]] for b in bs] for w in ws])
    return float((pref * np.linalg.det(M)).real)


def square(L):
    return [(i, j) for i in range(L) for j in range(L)]


def boundary_segs(L):
    return [[(0, 0), (L, 0)], [(L, 0), (L, L)], [(L, L), (0, L)], [(0, L), (0, 0)]]


C_CENTER, C_MAX = '#1976d2', '#d32f2f'
S = 0.94

fig = plt.figure(figsize=(13.2, 5.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.55)
axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1]); axC = fig.add_subplot(gs[2])
cax = axC.inset_axes([1.05, 0.0, 0.05, 1.0])

# ── (a) central edge ─────────────────────────────────────────────────────────
Ls_even = list(range(4, 61, 2))
dev = []
for L in Ls_even:
    K, Ki, bm, wm = build(square(L)); m = L // 2
    dev.append(L * (edge_prob((m - 1, m - 1), (m - 1, m), K, Ki, bm, wm) - 0.25))
dev = np.array(dev)
odd_m = np.array([(L // 2) % 2 == 1 for L in Ls_even])
axA.axhline(0, color='0.35', ls='--', lw=1.0)
axA.plot(np.array(Ls_even)[odd_m], dev[odd_m], 'o-', color='#c62828', ms=5, lw=1.2,
         label=r'$L\equiv2\ (\mathrm{mod}\ 4)$, $m$ odd')
axA.plot(np.array(Ls_even)[~odd_m], dev[~odd_m], 's-', color='#1565c0', ms=5, lw=1.2,
         label=r'$L\equiv0\ (\mathrm{mod}\ 4)$, $m$ even')
axA.set_xlabel(r'board size $L$')
axA.set_ylabel(r'$L\,(P_e-1/4)$, central edge')
axA.set_ylim(-0.75, 0.75)
axA.grid(True, alpha=0.25)

axA2 = axA.twinx()
Ls_odd = list(range(5, 42, 2))
prad = []
for L in Ls_odd:
    m = L // 2
    R = [c for c in square(L) if c != (m, m)]
    K, Ki, bm, wm = build(R, hole=(m, m))
    prad.append(edge_prob((m + 1, m), (m + 2, m), K, Ki, bm, wm))
prad = np.array(prad)
even_m = np.array([(L // 2) % 2 == 0 for L in Ls_odd])
axA2.plot(np.array(Ls_odd)[even_m], prad[even_m], '^:', color='#ef6c00', ms=4.5, lw=1.0,
          label=r'odd $L$, central hole: $P_{\mathrm{rad}}$, $L\equiv1\ (\mathrm{mod}\ 4)$')
axA2.plot(np.array(Ls_odd)[~even_m], prad[~even_m], 'v:', color='#2e7d32', ms=4.5, lw=1.0,
          label=r'odd $L$, central hole: $P_{\mathrm{rad}}$, $L\equiv3\ (\mathrm{mod}\ 4)$')
axA2.set_ylabel(r'$P_{\mathrm{rad}}$ (odd $L$, central cell removed)')
axA2.set_ylim(0.2, 0.62)
h1, l1 = axA.get_legend_handles_labels(); h2, l2 = axA2.get_legend_handles_labels()
axA.legend(h1 + h2, l1 + l2, fontsize=7.6, loc='upper center', ncol=2,
           bbox_to_anchor=(0.5, -0.16), frameon=False, columnspacing=0.8)
axA.set_title(r'(a) Central statistics of the $L\times L$ board', fontsize=11)

# ── (b), (c) P_H maps ────────────────────────────────────────────────────────
maps = []
for L in (20, 22):
    K, Ki, bm, wm = build(square(L))
    vals = {(i, j): ph((i, j), K, Ki, bm, wm) for i in range(L - 1) for j in range(L - 1)}
    maps.append((L, vals))
# Clip the color scale at P_H = 0.20: only the four corner squares of each board
# (P_H ~ 0.30) exceed it, and they would otherwise wash out the interior.
vmax = 0.20
vmin = min(min(v.values()) for _, v in maps)
norm = Normalize(vmin=vmin, vmax=vmax); cmap = plt.cm.viridis
for ax, (L, vals), lab in zip((axB, axC), maps, ('b', 'c')):
    m = L // 2
    for (i, j), v in vals.items():
        ax.add_patch(Rectangle((j + 1 - S / 2, i + 1 - S / 2), S, S,
                               facecolor=cmap(norm(v)), edgecolor='none'))
    gmax = max(vals.values())
    for (i, j), v in vals.items():
        if abs(v - gmax) < 1e-10:
            ax.add_patch(Rectangle((j + 1 - S / 2, i + 1 - S / 2), S, S, fill=False,
                                   edgecolor=C_MAX, lw=2.0, zorder=5))
    ax.add_patch(Rectangle((m - S / 2, m - S / 2), S, S, fill=False,
                           edgecolor=C_CENTER, lw=2.2, zorder=5))
    ax.add_collection(LineCollection(boundary_segs(L), colors='k', linewidths=1.1, zorder=6))
    ax.set_xlim(-0.6, L + 0.6); ax.set_ylim(-0.6, L + 0.6); ax.set_aspect('equal')
    ax.set_xticks(range(0, L + 1, 5)); ax.set_yticks(range(0, L + 1, 5))
    ax.set_xlabel('$j$'); ax.set_ylabel('$i$')
    par = 'even' if m % 2 == 0 else 'odd'
    ax.set_title(rf'({lab}) $P_H$, ${L}\times{L}$ board ($m={m}$, {par})' + '\n'
                 + rf'center $P_H={vals[(m - 1, m - 1)]:.4f}$', fontsize=10.5)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend='max')
cb.set_label(r'$P_H$')

fig.legend(handles=[
    plt.Line2D([0], [0], color=C_CENTER, lw=2.2, label='central square'),
    plt.Line2D([0], [0], color=C_MAX, lw=2.0, label='global maxima (corner squares)')],
    loc='lower center', ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.64, 0.0))
fig.subplots_adjust(left=0.06, right=0.93, bottom=0.24, top=0.87)
plt.savefig(os.path.join(FIGDIR, 'fig5_chessboard.pdf'), bbox_inches='tight')
plt.close()
print('Saved fig5_chessboard.pdf')
print('even L, L(Pe-1/4):', dict(zip(Ls_even[-4:], np.round(dev[-4:], 4))))
print('odd L, P_rad:', dict(zip(Ls_odd[-4:], np.round(prad[-4:], 4))))
