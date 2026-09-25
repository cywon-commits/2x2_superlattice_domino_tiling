"""
gen_fig1_arctic_final.py
------------------------
Generate fig1_arctic_circle.pdf: introductory three-panel figure.

  (a) The Aztec diamond AZ(4) with its checkerboard coloring and one domino
      tiling; each domino is drawn together with the dimer edge (black-cell
      center -> white-cell center) that it represents.
  (b) A uniformly random domino tiling of AZ(N_RANDOM), sampled exactly with
      the domino-shuffling algorithm (Elkies-Kuperberg-Larsen-Propp; Propp 2003),
      colored by the four domino types N, S, E, W.  The frozen corners and the
      disordered interior bounded by the Arctic circle are clearly visible.
  (c) Qualitative frozen/liquid map of AZ(12) computed from the inverse
      Kasteleyn matrix: each cell is colored by the type of its most likely
      domino when that probability exceeds P_FROZEN, and grey otherwise.

Conventions (identical to the other figure scripts):
  cells (i, j) with |i+0.5| + |j+0.5| <= n, plotted as [j, j+1] x [i, i+1];
  black cells: (i+j) even.  Arctic circle: center (0, 0), radius n/sqrt(2).
  Domino type of a domino with anchor cell (i, j) (left cell of a horizontal
  domino, lower cell of a vertical one) in AZ(n):  p = (i + j + n) mod 2,
    horizontal: p = 0 -> N,  p = 1 -> S;   vertical: p = 0 -> E,  p = 1 -> W.
  With this rule the top frozen corner of AZ(n) consists of N dominoes, the
  bottom of S, the right of E and the left of W.

Style matches Figure 4: staircase boundary, dashed Arctic circle, legend below.

Output: fig1_arctic_circle.pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection, PatchCollection

SEED = 20260923
N_EXAMPLE = 4
N_RANDOM = 60
N_KAST = 12
P_FROZEN = 0.85

TYPE_COLOR = {'N': '#e57373', 'S': '#64b5f6', 'E': '#81c784', 'W': '#ffd54f'}
LIQUID_COLOR = '#e0e0e0'


# ── Geometry helpers ─────────────────────────────────────────────────────────

def aztec_cells(n):
    return [(i, j) for i in range(-n, n) for j in range(-n, n)
            if abs(i + 0.5) + abs(j + 0.5) <= n]


def aztec_boundary_segs(cells_set):
    """Unit cell-edge segments forming the true (staircase) boundary of AZ(n)."""
    segs = []
    for (i, j) in cells_set:
        if (i, j + 1) not in cells_set: segs.append([(j + 1, i), (j + 1, i + 1)])
        if (i, j - 1) not in cells_set: segs.append([(j, i), (j, i + 1)])
        if (i + 1, j) not in cells_set: segs.append([(j, i + 1), (j + 1, i + 1)])
        if (i - 1, j) not in cells_set: segs.append([(j, i), (j + 1, i)])
    return segs


def domino_type(anchor, horizontal, n):
    p = (anchor[0] + anchor[1] + n) % 2
    if horizontal:
        return 'N' if p == 0 else 'S'
    return 'E' if p == 0 else 'W'


def finish(ax, n, circle_color='0.25', lw_boundary=1.1):
    ax.add_collection(LineCollection(aztec_boundary_segs(set(aztec_cells(n))),
                                     colors='k', linewidths=lw_boundary, zorder=6))
    r = n / np.sqrt(2)
    th = np.linspace(0, 2 * np.pi, 600)
    ax.plot(r * np.cos(th), r * np.sin(th), color=circle_color, ls='--',
            lw=1.1, zorder=7)
    pad = 0.04 * n + 0.3
    ax.set_xlim(-n - pad, n + pad); ax.set_ylim(-n - pad, n + pad)
    ax.set_aspect('equal')


# ── Domino shuffling (exact uniform sampler) ─────────────────────────────────
# A tiling is a dict: anchor cell -> 'h' or 'v'.  Dominoes move one step per
# growth n -> n+1 in the direction of their type: N up, S down, E right, W left.

MOVE = {'N': (1, 0), 'S': (-1, 0), 'E': (0, 1), 'W': (0, -1)}


def shuffle_step(tiling, n, rng):
    """Map a uniform tiling of AZ(n) to a uniform tiling of AZ(n+1)."""
    types = {a: domino_type(a, o == 'h', n) for a, o in tiling.items()}

    # 1. Destruction: remove colliding pairs inside a 2x2 block
    #    (S above N, or E to the left of W).
    removed = set()
    for a, t in types.items():
        i, j = a
        if t == 'N' and tiling[a] == 'h':
            above = (i + 1, j)
            if tiling.get(above) == 'h' and types.get(above) == 'S':
                removed.update([a, above])
        if t == 'E' and tiling[a] == 'v':
            right = (i, j + 1)
            if tiling.get(right) == 'v' and types.get(right) == 'W':
                removed.update([a, right])

    # 2. Sliding.
    new = {}
    occupied = set()
    for a, o in tiling.items():
        if a in removed:
            continue
        di, dj = MOVE[types[a]]
        b = (a[0] + di, a[1] + dj)
        new[b] = o
        occupied.update([b, (b[0], b[1] + 1)] if o == 'h' else [b, (b[0] + 1, b[1])])

    # 3. Creation: the empty cells of AZ(n+1) split into 2x2 blocks; scanning
    #    top-to-bottom and left-to-right, the first empty cell is the top-left
    #    cell of such a block.  Fill it with two horizontal or two vertical
    #    dominoes with probability 1/2 each.
    m = n + 1
    cells = set(aztec_cells(m))
    for i in range(m - 1, -m - 1, -1):
        for j in range(-m, m):
            c = (i, j)
            if c not in cells or c in occupied:
                continue
            block = [(i, j), (i, j + 1), (i - 1, j), (i - 1, j + 1)]
            assert all(x in cells and x not in occupied for x in block), \
                f'creation failed at {c} for n={m}'
            if rng.random() < 0.5:
                new[(i, j)] = 'h'; new[(i - 1, j)] = 'h'
            else:
                new[(i - 1, j)] = 'v'; new[(i - 1, j + 1)] = 'v'
            occupied.update(block)
    assert occupied == cells
    return new


def random_tiling(n, rng):
    t = {}
    for k in range(n):
        t = shuffle_step(t, k, rng)
    return t


# ── Kasteleyn helpers (for panel C) ──────────────────────────────────────────

def build_aztec(n):
    cells = aztec_cells(n)
    black = [c for c in cells if (c[0] + c[1]) % 2 == 0]
    white = [c for c in cells if (c[0] + c[1]) % 2 == 1]
    bmap = {b: k for k, b in enumerate(black)}
    wmap = {w: k for k, w in enumerate(white)}
    K = np.zeros((len(black), len(white)), dtype=complex)
    for b, bi in bmap.items():
        i, j = b
        for (di, dj), weight in [((0, 1), 1), ((0, -1), -1), ((1, 0), 1j), ((-1, 0), -1j)]:
            w = (i + di, j + dj)
            if w in wmap:
                K[bi, wmap[w]] = weight
    return K, np.linalg.inv(K), bmap, wmap


def frozen_map_color(i, j, n, K, Kinv, bmap, wmap):
    """Color of cell (i,j): type of its most likely domino if P > P_FROZEN."""
    best, best_p = None, -1.0
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nb = (i + di, j + dj)
        b, w = ((i, j), nb) if (i + j) % 2 == 0 else (nb, (i, j))
        if b not in bmap or w not in wmap:
            continue
        p = (K[bmap[b], wmap[w]] * Kinv[wmap[w], bmap[b]]).real
        if p > best_p:
            best_p, best = p, nb
    if best_p <= P_FROZEN:
        return LIQUID_COLOR
    horizontal = best[0] == i
    anchor = (i, min(j, best[1])) if horizontal else (min(i, best[0]), j)
    return TYPE_COLOR[domino_type(anchor, horizontal, n)]


# ── Panels ───────────────────────────────────────────────────────────────────

def domino_rect(a, o):
    i, j = a
    return (j, i, 2, 1) if o == 'h' else (j, i, 1, 2)


def panel_example(ax, rng):
    n = N_EXAMPLE
    for (i, j) in aztec_cells(n):                      # faint unit-cell grid
        ax.add_patch(mpatches.Rectangle((j, i), 1, 1, facecolor='white',
                                        edgecolor='0.8', lw=0.6))
    tiling = random_tiling(n, rng)
    for a, o in tiling.items():
        x, y, w, h = domino_rect(a, o)
        t = domino_type(a, o == 'h', n)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x + 0.08, y + 0.08), w - 0.16, h - 0.16,
            boxstyle='round,pad=0,rounding_size=0.12',
            facecolor=TYPE_COLOR[t], alpha=0.45, edgecolor='k', lw=1.0, zorder=3))
        # dimer edge between the two cell centers
        if o == 'h':
            ax.plot([x + 0.5, x + 1.5], [y + 0.5, y + 0.5], color='k', lw=1.4, zorder=4)
        else:
            ax.plot([x + 0.5, x + 0.5], [y + 0.5, y + 1.5], color='k', lw=1.4, zorder=4)
        for (ci, cj) in ([(a[0], a[1]), (a[0], a[1] + 1)] if o == 'h'
                         else [(a[0], a[1]), (a[0] + 1, a[1])]):
            black = (ci + cj) % 2 == 0
            ax.scatter(cj + 0.5, ci + 0.5, s=16, zorder=5, linewidths=0.9,
                       facecolor='k' if black else 'white', edgecolor='k')
    finish(ax, n, circle_color='0.35')
    ax.set_xticks(range(-n, n + 1, 2)); ax.set_yticks(range(-n, n + 1, 2))
    ax.set_xlabel('$j$'); ax.set_ylabel('$i$')
    ax.set_title(rf'(a) A domino tiling of $\mathrm{{AZ}}({n})$', fontsize=11)


def panel_random(ax, rng):
    n = N_RANDOM
    tiling = random_tiling(n, rng)
    patches, colors = [], []
    for a, o in tiling.items():
        x, y, w, h = domino_rect(a, o)
        patches.append(mpatches.Rectangle((x, y), w, h))
        colors.append(TYPE_COLOR[domino_type(a, o == 'h', n)])
    # No per-domino outlines at this size (they wash out when printed);
    # rasterise the 3660 rectangles so the PDF stays small and renders reliably.
    ax.add_collection(PatchCollection(patches, facecolor=colors,
                                      edgecolor='none', rasterized=True))
    finish(ax, n, circle_color='k', lw_boundary=0.8)
    ax.set_xticks([-n, -n // 2, 0, n // 2, n]); ax.set_yticks([-n, -n // 2, 0, n // 2, n])
    ax.set_xlabel('$j$'); ax.set_ylabel('$i$')
    ax.set_title(rf'(b) Uniformly random tiling of $\mathrm{{AZ}}({n})$', fontsize=11)
    return tiling


def panel_kasteleyn(ax):
    n = N_KAST
    K, Kinv, bmap, wmap = build_aztec(n)
    for (i, j) in aztec_cells(n):
        ax.add_patch(mpatches.Rectangle(
            (j, i), 1, 1, edgecolor='white', linewidth=0.3,
            facecolor=frozen_map_color(i, j, n, K, Kinv, bmap, wmap)))
    finish(ax, n, circle_color='0.25')
    ax.set_xticks(range(-n, n + 1, 4)); ax.set_yticks(range(-n, n + 1, 4))
    ax.set_xlabel('$j$'); ax.set_ylabel('$i$')
    ax.set_title(rf'(c) Frozen/liquid map of $\mathrm{{AZ}}({n})$', fontsize=11)


# ── Main ─────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.5))
panel_example(axes[0], rng)
tiling = panel_random(axes[1], rng)
panel_kasteleyn(axes[2])

blank = mpatches.Patch(facecolor='none', edgecolor='none', label=' ')
legend_handles = [      # filled column by column, two rows
    mpatches.Patch(facecolor=TYPE_COLOR['N'], label='N domino'),
    mpatches.Patch(facecolor=TYPE_COLOR['S'], label='S domino'),
    mpatches.Patch(facecolor=TYPE_COLOR['E'], label='E domino'),
    mpatches.Patch(facecolor=TYPE_COLOR['W'], label='W domino'),
    plt.Line2D([0], [0], marker='o', ls='', markerfacecolor='k', markeredgecolor='k',
               markersize=5, label='black cell (a)'),
    plt.Line2D([0], [0], marker='o', ls='', markerfacecolor='white', markeredgecolor='k',
               markersize=5, label='white cell (a)'),
    mpatches.Patch(facecolor=LIQUID_COLOR, label=rf'no domino with $P>{P_FROZEN}$ (c)'),
    blank,
    plt.Line2D([0], [0], color='k', lw=1.1, label='boundary of $\\mathrm{AZ}(n)$'),
    plt.Line2D([0], [0], color='0.25', lw=1.1, ls='--',
               label=r'Arctic circle, $r=n/\sqrt{2}$'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=5, fontsize=8.8,
           frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.6,
           columnspacing=1.4)
plt.tight_layout(rect=[0, 0.11, 1, 1])
out = os.path.join(FIGDIR, 'fig1_aztec_tilings.pdf')
plt.savefig(out, bbox_inches='tight', dpi=400)
plt.close()
print(f'Saved {out}  (AZ({N_RANDOM}) sample: {len(tiling)} dominoes)')
