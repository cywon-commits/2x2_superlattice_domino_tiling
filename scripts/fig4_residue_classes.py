"""Generate the main-text Figure 4: spatial structure of P_H across the four
residue classes n mod 4, using ALL elementary 2x2 squares (lower-left cell black
or white).  Each square's value is drawn as a unit square at the square's center,
so no information is hidden by overlapping blocks."""

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


def build_aztec(n):
    cells = [(i, j) for i in range(-n, n) for j in range(-n, n)
             if abs(i + 0.5) + abs(j + 0.5) <= n]
    black = [c for c in cells if (c[0] + c[1]) % 2 == 0]
    white = [c for c in cells if (c[0] + c[1]) % 2 == 1]
    bmap = {b: k for k, b in enumerate(black)}
    wmap = {w: k for k, w in enumerate(white)}
    K = np.zeros((len(black), len(white)), dtype=complex)
    for b, bi in bmap.items():
        i, j = b
        for (di, dj), weight in [
            ((0, +1), 1), ((0, -1), -1),
            ((+1, 0), 1j), ((-1, 0), -1j)
        ]:
            w = (i + di, j + dj)
            if w in wmap:
                K[bi, wmap[w]] = weight
    return K, np.linalg.inv(K), bmap, wmap


def aztec_boundary_segs(cells_set):
    """Unit cell-edge segments forming the true (staircase) boundary of AZ(n)."""
    segs = []
    for (i, j) in cells_set:
        if (i, j + 1) not in cells_set: segs.append([(j + 1, i), (j + 1, i + 1)])
        if (i, j - 1) not in cells_set: segs.append([(j, i), (j, i + 1)])
        if (i + 1, j) not in cells_set: segs.append([(j, i + 1), (j + 1, i + 1)])
        if (i - 1, j) not in cells_set: segs.append([(j, i), (j + 1, i)])
    return segs


def edge_prob(b, w, K, Kinv, bmap, wmap):
    if b not in bmap or w not in wmap:
        return None
    return float((K[bmap[b], wmap[w]] * Kinv[wmap[w], bmap[b]]).real)


def _bw(c1, c2):
    """Order an adjacent pair of cells as (black, white)."""
    return (c1, c2) if (c1[0] + c1[1]) % 2 == 0 else (c2, c1)


def pair_prob(pairs, K, Kinv, bmap, wmap):
    """Joint probability of a set of (black, white) edges via the Kasteleyn determinant."""
    bs = [p[0] for p in pairs]; ws = [p[1] for p in pairs]
    if any(x not in bmap for x in bs) or any(x not in wmap for x in ws):
        return None
    pref = np.prod([K[bmap[bb], wmap[ww]] for bb, ww in pairs])
    M = np.array([[Kinv[wmap[ww], bmap[bb]] for bb in bs] for ww in ws])
    return float((pref * np.linalg.det(M)).real)


def ph_block(anchor, K, Kinv, bmap, wmap):
    """P_H for the 2x2 square with lower-left cell `anchor` (either color)."""
    i, j = anchor
    H = [_bw((i, j), (i, j + 1)), _bw((i + 1, j), (i + 1, j + 1))]
    return pair_prob(H, K, Kinv, bmap, wmap)



C_CENTER, C_MAX, C_GMAX = '#1976d2', '#d32f2f', '#ff00ff'
S = 0.94
TOL = 1e-10

fig, axes = plt.subplots(2, 2, figsize=(13.5, 12.8))
# (no figure title: APS figures carry their title in the caption)

for ax, n, lab in zip(axes.flat, [9, 10, 11, 12], 'abcd'):
    K, Kinv, bmap, wmap = build_aztec(n)
    blocks_all = []
    for i in range(-n, n):
        for j in range(-n, n):
            ph = ph_block((i, j), K, Kinv, bmap, wmap)
            if ph is not None:
                blocks_all.append(((i, j), ph))
    black = [(a, v) for a, v in blocks_all if (a[0] + a[1]) % 2 == 0]
    bmax = max(v for _, v in black)
    gmax = max(v for _, v in blocks_all)
    bmax_at = [a for a, v in black if abs(v - bmax) < TOL]
    gmax_at = [a for a, v in blocks_all if abs(v - gmax) < TOL]
    center = (-1, -1)
    cval = dict(blocks_all)[center]

    norm = Normalize(vmin=min(v for _, v in blocks_all), vmax=gmax)
    cmap = plt.cm.viridis
    for (i, j), v in blocks_all:
        ax.add_patch(Rectangle((j + 1 - S/2, i + 1 - S/2), S, S,
                               facecolor=cmap(norm(v)), edgecolor='none'))
    marks = [(center, C_CENTER, 2.4)]
    if gmax > bmax + 1e-12:
        marks += [(a, C_MAX, 2.0) for a in bmax_at] + [(a, C_GMAX, 2.2) for a in gmax_at]
    else:                                     # global max is black-anchored
        marks += [(a, C_MAX, 2.2) for a in gmax_at if a != center]
        if center in gmax_at:
            marks[0] = (center, C_CENTER, 3.0)
    for (i, j), col, lw in marks:
        ax.add_patch(Rectangle((j + 1 - S/2, i + 1 - S/2), S, S, fill=False,
                               edgecolor=col, linewidth=lw, zorder=5))

    ax.add_collection(LineCollection(aztec_boundary_segs(set(bmap) | set(wmap)),
                                     colors='k', linewidths=1.1, zorder=6))
    r = n / np.sqrt(2); th = np.linspace(0, 2*np.pi, 400)
    ax.plot(r*np.cos(th), r*np.sin(th), color='0.6', ls='--', lw=1.1, zorder=4)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 fraction=0.046, pad=0.03, label=r'$P_H$')
    ax.set_xlim(-n-0.5, n+0.5); ax.set_ylim(-n-0.5, n+0.5); ax.set_aspect('equal')
    ax.set_xlabel('$j$'); ax.set_ylabel('$i$')
    where = 'white-anchored' if (gmax_at[0][0] + gmax_at[0][1]) % 2 else 'black-anchored'
    ax.set_title(rf'({lab}) $\mathrm{{AZ}}({n})$, $n\equiv{n % 4}\ \mathrm{{mod}}\ 4$' + '\n'
                 + rf'center $P_H={cval:.5f}$;  max $P_H={gmax:.5f}$ '
                 + f'({len(gmax_at)}x, {where})', fontsize=11.5)
    print(n, n % 4, f'center {cval:.6f}', f'black max {bmax:.6f} x{len(bmax_at)} {bmax_at}',
          f'global max {gmax:.6f} x{len(gmax_at)} {where} {gmax_at}')

fig.legend(handles=[
    plt.Line2D([0], [0], color=C_CENTER, lw=2.4, label='central square'),
    plt.Line2D([0], [0], color=C_MAX, lw=2.2, label='maxima among black-anchored squares'),
    plt.Line2D([0], [0], color=C_GMAX, lw=2.2, label='maxima among all squares (if different)')],
    loc='lower center', ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(rect=[0, 0.03, 1, 0.965])
plt.savefig(os.path.join(FIGDIR, 'fig4_residue_classes.pdf'), bbox_inches='tight')
plt.close()
