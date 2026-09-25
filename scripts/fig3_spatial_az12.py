"""Generate the main-text Figure 3: spatial structure for AZ(12)."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle


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


n = 12
K, Kinv, bmap, wmap = build_aztec(n)

# All 2x2 squares: lower-left cell black ("black-anchored") or white ("white-anchored").
blocks_all = []
for i in range(-n, n):
    for j in range(-n, n):
        ph = ph_block((i, j), K, Kinv, bmap, wmap)
        if ph is not None:
            blocks_all.append(((i, j), ph))
blocks = [(a, v) for a, v in blocks_all if (a[0] + a[1]) % 2 == 0]   # black-anchored only

# Maxima among black-anchored squares (the convention of Definition 3.1 / Table 2)
max_ph = max(v for _, v in blocks)
max_blocks = [a for a, v in blocks if abs(v - max_ph) < 1e-10]
# Global maxima over all 2x2 squares
gmax_ph = max(v for _, v in blocks_all)
gmax_blocks = [a for a, v in blocks_all if abs(v - gmax_ph) < 1e-10]
print(f"black-anchored max {max_ph:.9f} at {max_blocks}")
print(f"global max         {gmax_ph:.9f} at {gmax_blocks}")
center = (-1, -1)
center_ph = dict(blocks)[center]

PANEL_A_MODE = 'centered'   # 'centered': one panel, each 2x2 block shown as a unit
                            #             square at the block center (no overlap)
                            # 'split'   : two panels, anchors (2a,2b) and (2a+1,2b+1),
                            #             each a non-overlapping tiling by full 2x2 blocks

from matplotlib.collections import LineCollection
bsegs = aztec_boundary_segs(set(bmap) | set(wmap))
r = n / np.sqrt(2)
theta = np.linspace(0, 2*np.pi, 500)

vals = [v for _, v in blocks_all]
norm = Normalize(vmin=min(vals), vmax=max(vals))
cmap = plt.cm.viridis
C_CENTER, C_MAX, C_GMAX = '#1976d2', '#d32f2f', '#ff00ff'


def finish_A(ax, title):
    ax.add_collection(LineCollection(bsegs, colors='k', linewidths=1.1, zorder=6))
    ax.plot(r*np.cos(theta), r*np.sin(theta), color='0.6', ls='--', lw=1.1, zorder=4)
    ax.set_xlim(-n-0.5, n+0.5); ax.set_ylim(-n-0.5, n+0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('$j$'); ax.set_ylabel('$i$')
    ax.set_title(title)


if PANEL_A_MODE == 'centered':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.8))
    S = 0.94                                   # side of the drawn square (1 = touching corners)
    for (i, j), val in blocks_all:            # every 2x2 square, no gaps
        xc, yc = j + 1, i + 1                  # center of the 2x2 block
        ax1.add_patch(Rectangle((xc - S/2, yc - S/2), S, S,
                                facecolor=cmap(norm(val)), edgecolor='none'))
    marks = [(center, C_CENTER, 2.2)] + [(b, C_MAX, 2.0) for b in max_blocks]
    if gmax_ph > max_ph + 1e-12:
        marks += [(b, C_GMAX, 2.0) for b in gmax_blocks]
    for (i, j), col, lw in marks:
        ax1.add_patch(Rectangle((j + 1 - S/2, i + 1 - S/2), S, S, fill=False,
                                edgecolor=col, linewidth=lw, zorder=5))
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1,
                 fraction=0.045, pad=0.02, label=r'$P_H$')
    ax1.legend(handles=[
        plt.Line2D([0], [0], color=C_CENTER, lw=2.2, label='central block'),
        plt.Line2D([0], [0], color=C_MAX, lw=2.0, label='maxima, black-anchored squares')]
        + ([plt.Line2D([0], [0], color=C_GMAX, lw=2.0, label='maxima, all squares')]
           if gmax_ph > max_ph + 1e-12 else []),
        fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.135), ncol=3,
        framealpha=0.9, handlelength=1.4, columnspacing=1.0)
    finish_A(ax1, r'(a) $P_H$ at the center of each $2\times2$ block')
else:
    fig, (ax1a, ax1b, ax2) = plt.subplots(1, 3, figsize=(17.5, 5.8))
    sub = {0: ax1a, 1: ax1b}                   # anchor parity: (even,even) / (odd,odd)
    for (i, j), val in blocks:
        sub[i % 2].add_patch(Rectangle((j, i), 2, 2, facecolor=cmap(norm(val)),
                                       edgecolor='white', linewidth=0.3))
    for (i, j), col, lw in [(center, C_CENTER, 2.2)] + [(b, C_MAX, 2.0) for b in max_blocks]:
        sub[i % 2].add_patch(Rectangle((j, i), 2, 2, fill=False, edgecolor=col,
                                       linewidth=lw, zorder=5))
    for ax in (ax1a, ax1b):
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     fraction=0.045, pad=0.02, label=r'$P_H$')
    ax1a.legend(handles=[
        plt.Line2D([0], [0], color=C_CENTER, lw=2.2, label='central block'),
        plt.Line2D([0], [0], color=C_MAX, lw=2.0, label='four numerical maxima')],
        fontsize=8, loc='upper right', framealpha=0.9)
    finish_A(ax1a, r'(A1) $P_H$, blocks anchored at $(2a,2b)$')
    finish_A(ax1b, r'(A2) $P_H$, blocks anchored at $(2a{+}1,2b{+}1)$')

# (no figure title: APS figures carry their title in the caption)


# Panel B: edge deviations from 1/4, drawn as dimer bonds (black-cell center
# -> white-cell center), i.e. along the direction joining the two cells of the
# domino.
from matplotlib.colors import LinearSegmentedColormap

SHRINK = 0.72              # fraction of the center-to-center distance drawn
DEV_CLIP = 0.125           # color scale spans P_e-1/4 in [-DEV_CLIP, +DEV_CLIP]

segs, devs = [], []
for b in bmap:
    i, j = b
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        w = (i + di, j + dj)
        if w in wmap:
            p = edge_prob(b, w, K, Kinv, bmap, wmap)
            xm, ym = j + 0.5 + dj / 2, i + 0.5 + di / 2      # edge midpoint
            hx, hy = SHRINK * dj / 2, SHRINK * di / 2          # half-length vector
            segs.append([(xm - hx, ym - hy), (xm + hx, ym + hy)])
            devs.append(p - 0.25)
devs = np.array(devs)

# Diverging red-white-blue map: favoured edges (P_e > 1/4) red,
# disfavoured edges (P_e < 1/4) blue, P_e = 1/4 white.
cmap2 = plt.cm.RdBu_r
norm2 = TwoSlopeNorm(vmin=-DEV_CLIP, vcenter=0.0, vmax=DEV_CLIP)

order = np.argsort(np.abs(devs))             # draw strongly deviating bonds on top
lc = LineCollection([segs[k] for k in order], array=devs[order], cmap=cmap2,
                    norm=norm2, linewidths=2.3, capstyle='butt', zorder=3)
ax2.add_collection(lc)
cbar2 = plt.colorbar(lc, ax=ax2, fraction=0.045, pad=0.02, extend='both',
                     label=r'$P_e-1/4$')
cbar2.set_ticks([-0.125, -0.0625, 0, 0.0625, 0.125])
cbar2.set_ticklabels([r'$\leq -1/8$', r'$-1/16$', r'$0$', r'$+1/16$', r'$\geq 1/8$'])
ax2.add_collection(LineCollection(bsegs, colors='k', linewidths=1.1, zorder=6))
ax2.plot(r*np.cos(theta), r*np.sin(theta), color='0.25', ls='--', lw=1.1, alpha=0.8)
ax2.set_xlim(-n-0.5, n+0.5); ax2.set_ylim(-n-0.5, n+0.5)
ax2.set_aspect('equal')
ax2.set_xlabel('$j$'); ax2.set_ylabel('$i$')
ax2.set_title(r'(b) Edge-probability deviation $P_e-1/4$')

fig.text(0.5, 0.005,
         rf'Center: $P_H={center_ph:.6f}$;  max (black-anchored): $P_H={max_ph:.6f}$;  max (all squares): $P_H={gmax_ph:.6f}$',
         ha='center', fontsize=9.5)
plt.tight_layout(rect=[0, 0.075, 1, 0.96])
plt.savefig(os.path.join(FIGDIR, 'fig3_spatial.pdf'), bbox_inches='tight')
plt.close()
