# 2x2_superlattice_domino_tiling

Code accompanying the manuscript

> C. Won, *Parity-dependent sublattice structure of local 2×2 correlations in
> domino tilings of the Aztec diamond* (in preparation).

The scripts compute exact finite-size dimer statistics of uniformly random domino
tilings of the Aztec diamond AZ(n) and of the L×L chessboard from the inverse
Kasteleyn matrix, and reproduce every figure and table of the paper.

## Contents

| Script | Output | Content |
|---|---|---|
| `scripts/fig1_aztec_tilings.py` | `figures/fig1_aztec_tilings.pdf` | Fig. 1: a tiling of AZ(4), an exact uniform sample of AZ(60) by domino shuffling, and the frozen/liquid map of AZ(12) |
| `scripts/fig2_background.py` | `figures/fig2_background.pdf` | Fig. 2: the two sublattices of 2×2 squares, notation, central edge probability and central P_H |
| `scripts/fig3_spatial_az12.py` | `figures/fig3_spatial.pdf` | Fig. 3: P_H on all 2×2 squares and single-edge deviations in AZ(12) |
| `scripts/fig4_residue_classes.py` | `figures/fig4_residue_classes.pdf` | Fig. 4: P_H for AZ(9)–AZ(12), one per residue class of n mod 4 |
| `scripts/fig5_chessboard.py` | `figures/fig5_chessboard.pdf` | Fig. 5: comparison with the L×L chessboard (even boards, odd boards with a central hole) |
| `scripts/fig6_height_plateaus.py` | `figures/fig6_height_plateaus.pdf` | Fig. 6: the four basketweave placements and their mean heights mod 4 |
| `scripts/tables.py` | printed | Tables I–III: central values, maxima of P_H by sublattice, sublattice contrast |
| `scripts/mean_height.py` | printed | Sec. IV: exact central mean height ⟨h(0,0)⟩ = (−1)^(n+1) n, its offset mod 4, and the basketweave plateau heights |
| `scripts/verify_identity_general_regions.py` | printed | Exact enumeration check of P_H = P_V = P1 P2 + P3 P4 on regions other than the Aztec diamond, independent of any Kasteleyn matrix |

## Requirements

Python ≥ 3.9 with

```
pip install -r requirements.txt
```

(NumPy and Matplotlib; tested with NumPy 2.4 and Matplotlib 3.10.)

## Usage

```
python scripts/fig1_aztec_tilings.py      # likewise for fig2 ... fig5
python scripts/tables.py                  # all tables; `python scripts/tables.py 2` for Table II only, `2x` for n = 18–22
python scripts/verify_identity_general_regions.py
```

Figures are written to `figures/`.  Each figure script runs in well under a
minute on a laptop; `tables.py` takes about a minute.

## Conventions

* Cell `(i, j)` is the unit square `[j, j+1] × [i, i+1]`; `i` labels rows
  (increasing upward), `j` labels columns.  AZ(n) is the set of cells with
  `|i + 1/2| + |j + 1/2| <= n`.
* A cell is black if `i + j` is even.
* Kasteleyn phases: `K[b, w] = +1` (w to the right of b), `-1` (left),
  `+i` (up), `-i` (down).
* The elementary square with corner cell `(a, b)` is `{a, a+1} × {b, b+1}`; it is
  *black-anchored* if `a + b` is even and *white-anchored* otherwise.
* For odd chessboards with the central cell removed, the sign of every horizontal
  edge crossing the lattice line `x = h + 1` above the hole is flipped to restore
  the Kasteleyn condition (checked against exact enumeration for L = 3, 5, 7, 9).

## License

MIT License; see `LICENSE`.
