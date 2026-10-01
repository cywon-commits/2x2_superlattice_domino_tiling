"""
fig6_height_plateaus.py
-----------------------
Generate fig6_height_plateaus.pdf: the four placements of the basketweave
tiling near the center, labelled by the corner-cell class (a,b) mod 2 of their
2x2 blocks and by their mean height modulo 4, measured from the residue of the
central vertex (computed, not typed in: see mean_height.py).

  (a) (1,1): black-anchored, contains the central square      -> mean h = 2
  (b) (0,0): black-anchored, contains its four diagonal neighbors -> 0
  (c) (0,1): white-anchored -> 1        (d) (1,0): white-anchored -> 3

Conventions as in all scripts: cell (i,j) = [j,j+1] x [i,i+1]; black if i+j even.
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mean_height import basketweave_mean

R = 4                      # window: cells with -R <= i, j < R
C_BLACK_CELL = '#d9d9d9'
FILL = {'black': '#cfe0f3', 'white': '#fbe0c8'}
C_CENTER, C_DIAG = '#1976d2', '#c62828'

plt.rcParams.update({'font.size': 8, 'mathtext.fontset': 'cm'})


def blocks(p, q):
    """Corner cells and orientation (True = horizontal pair) of the blocks."""
    out = []
    for a in range(-R - 2, R + 2):
        for b in range(-R - 2, R + 2):
            if (a - p) % 2 or (b - q) % 2:
                continue
            horiz = (((a - p) // 2 + (b - q) // 2) % 2 == 0)
            out.append((a, b, horiz))
    return out


NOTE = {(1, 1): r'$n\equiv1,2$ (mod 4)', (0, 0): r'$n\equiv0,3$ (mod 4)'}


def panel(ax, p, q, label):
    cls = 'black' if (p + q) % 2 == 0 else 'white'
    for i in range(-R, R):
        for j in range(-R, R):
            ax.add_patch(Rectangle((j, i), 1, 1, lw=0,
                         facecolor=C_BLACK_CELL if (i + j) % 2 == 0 else 'white'))
    g = 0.09
    for a, b, horiz in blocks(p, q):
        doms = ([(b, a, 2, 1), (b, a + 1, 2, 1)] if horiz
                else [(b, a, 1, 2), (b + 1, a, 1, 2)])
        for x, y, w, h in doms:
            ax.add_patch(FancyBboxPatch((x + g, y + g), w - 2 * g, h - 2 * g,
                         boxstyle='round,pad=0,rounding_size=0.12',
                         facecolor=FILL[cls], edgecolor='k', lw=0.7, alpha=0.92))
    if (p, q) == (0, 0):
        for (a, b) in [(-2, -2), (-2, 0), (0, -2), (0, 0)]:
            ax.add_patch(Rectangle((b, a), 2, 2, fill=False, edgecolor=C_DIAG,
                                   lw=1.6, ls=(0, (3, 1.5)), zorder=5))
    ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, edgecolor=C_CENTER,
                           lw=1.8, zorder=6))
    ax.plot(0, 0, 'o', ms=3.2, color='k', zorder=7)
    hbar = basketweave_mean(p, q) % 4
    ax.set_title(label + rf' {cls}-anchored, $\bar h\equiv{round(hbar)}$',
                 fontsize=8.5, pad=3)
    sub = rf'$(a,b)\equiv({p},{q})$'
    if (p, q) in NOTE:
        sub += '  ·  ' + NOTE[(p, q)]
    ax.text(0.5, -0.035, sub, transform=ax.transAxes, ha='center', va='top',
            fontsize=7.5, color='0.2')
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color('0.5')
    return hbar


fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.9))
for ax, (p, q), lab in zip(axs.flat, [(1, 1), (0, 0), (0, 1), (1, 0)],
                           ['(a)', '(b)', '(c)', '(d)']):
    print(lab, (p, q), 'mean h mod 4 =', round(panel(ax, p, q, lab) , 3))
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.06,
                    wspace=0.06, hspace=0.3)
plt.savefig(os.path.join(HERE, '..', 'figures', 'fig6_height_plateaus.pdf'), bbox_inches='tight')
