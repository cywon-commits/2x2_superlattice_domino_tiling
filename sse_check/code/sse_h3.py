"""
SSE for the spin-1/2 Heisenberg AF  H = J sum_<ij> S_i.S_j  on an arbitrary graph.
Directed loops with heat-bath exit probabilities (ergodic also on non-bipartite
graphs, unlike the deterministic switch-reverse loop, which cannot change the
parity of the number of off-diagonal operators).
Configurations are sampled with |W|; sign = (-1)^(# off-diagonal operators).

Bond operator  H'_b = C - S^z_i S^z_j - 1/2 (S+S- + h.c.),  C = 1/4 + eps
  vertex weights: diag antiparallel 1/2+eps, diag parallel eps, off-diag 1/2
"""
import numpy as np
from numba import njit

EPS = 0.25


@njit(cache=True)
def wdiag(si, sj, C, hi, hj):
    return C - 0.25 * si * sj + 0.5 * (hi * si + hj * sj)


@njit(cache=True)
def vweight(s0, s1, s2, s3, C, hi, hj):
    if s0 == s2 and s1 == s3:
        return wdiag(s0, s1, C, hi, hj)
    if s2 == -s0 and s3 == -s1 and s0 != s1:
        return 0.5
    return 0.0


@njit(cache=True)
def seed(s):
    np.random.seed(s)


@njit(cache=True)
def sweep(spin, ops, bsites, beta, nb, n_ops, Cb, hb, nloops, cap, tr):
    N = spin.shape[0]
    M = ops.shape[0]
    # ---------------- diagonal update
    for p in range(M):
        op = ops[p]
        if op == -1:
            b = np.random.randint(nb)
            i, j = bsites[b, 0], bsites[b, 1]
            w = wdiag(spin[i], spin[j], Cb[b], hb[b, 0], hb[b, 1])
            if np.random.random() * (M - n_ops) < beta * nb * w:
                ops[p] = 2 * b
                n_ops += 1
        elif op % 2 == 0:
            b = op // 2
            i, j = bsites[b, 0], bsites[b, 1]
            w = wdiag(spin[i], spin[j], Cb[b], hb[b, 0], hb[b, 1])
            if np.random.random() * beta * nb * w < (M - n_ops + 1):
                ops[p] = -1
                n_ops -= 1
        else:
            b = op // 2
            spin[bsites[b, 0]] *= -1
            spin[bsites[b, 1]] *= -1
    # ---------------- build vertex list with leg spins
    link = np.full(4 * M, -1, np.int64)
    leg = np.zeros(4 * M, np.int64)
    first = np.full(N, -1, np.int64)
    last = np.full(N, -1, np.int64)
    cur = spin.copy()
    for p in range(M):
        op = ops[p]
        if op == -1:
            continue
        b = op // 2
        i, j = bsites[b, 0], bsites[b, 1]
        leg[4 * p] = cur[i]
        leg[4 * p + 1] = cur[j]
        if op % 2 == 1:
            cur[i] *= -1
            cur[j] *= -1
        leg[4 * p + 2] = cur[i]
        leg[4 * p + 3] = cur[j]
        for k in range(2):
            s = bsites[b, k]
            vl = 4 * p + k
            vu = 4 * p + 2 + k
            if last[s] != -1:
                link[last[s]] = vl
                link[vl] = last[s]
            else:
                first[s] = vl
            last[s] = vu
    for s in range(N):
        if first[s] != -1:
            link[first[s]] = last[s]
            link[last[s]] = first[s]
    # ---------------- directed loops
    if n_ops > 0:
        probs = np.zeros(4)
        for _ in range(nloops):
            j0 = np.random.randint(4 * M)
            while ops[j0 // 4] == -1:
                j0 = np.random.randint(4 * M)
            j = j0
            steps = 0
            while True:
                p = j // 4
                le = j % 4
                base = 4 * p
                bb = ops[p] // 2
                leg[base + le] *= -1          # entrance flip
                tot = 0.0
                for x in range(4):
                    leg[base + x] *= -1
                    w = vweight(leg[base], leg[base + 1], leg[base + 2], leg[base + 3], Cb[bb], hb[bb, 0], hb[bb, 1])
                    leg[base + x] *= -1
                    probs[x] = w
                    tot += w
                r = np.random.random() * tot
                x = 0
                acc = probs[0]
                while acc < r:
                    x += 1
                    acc += probs[x]
                leg[base + x] *= -1           # exit flip
                j = link[base + x]
                steps += 1
                if j == j0:
                    break
                if steps > 100 * M and steps <= 100 * M + 1:
                    tr[0] += 1          # loop exceeded the original cap
                if steps > cap:
                    tr[1] += 1          # loop actually truncated
                    break
            if steps > tr[2]:
                tr[2] = steps
        # write back operator types and spins
        for p in range(M):
            op = ops[p]
            if op == -1:
                continue
            b = op // 2
            base = 4 * p
            if leg[base] == leg[base + 2] and leg[base + 1] == leg[base + 3]:
                ops[p] = 2 * b
            else:
                ops[p] = 2 * b + 1
    for s in range(N):
        if first[s] != -1:
            spin[s] = leg[first[s]]
        elif np.random.random() < 0.5:
            spin[s] *= -1
    return n_ops


@njit(cache=True)
def measure(spin, ops):
    nod = 0
    n = 0
    for p in range(ops.shape[0]):
        if ops[p] != -1:
            n += 1
            if ops[p] % 2 == 1:
                nod += 1
    mz = 0
    for s in range(spin.shape[0]):
        mz += spin[s]
    return 1 - 2 * (nod % 2), 0.5 * mz, n


def run(bonds, N, beta, ntherm=2000, nmeas=20000, seed_=0, eps=EPS, nloops=None, init_spin=None, h=0.0):
    b = np.ascontiguousarray(bonds, dtype=np.int64)
    nb = len(b)
    z = np.bincount(b.ravel(), minlength=N).astype(float)
    hb = np.ascontiguousarray(np.stack([h / z[b[:, 0]], h / z[b[:, 1]]], 1))
    Cb = 0.25 + 0.5 * (hb[:, 0] + hb[:, 1]) + eps
    np.random.seed(seed_)
    seed(seed_)
    spin = (np.where(np.random.random(N) < 0.5, 1, -1) if init_spin is None else init_spin).astype(np.int64)
    M = max(40, int(beta * nb * 0.3))
    ops = np.full(M, -1, np.int64)
    n = 0
    nl = 10 if nloops is None else nloops
    for t in range(ntherm):
        n = sweep(spin, ops, b, beta, nb, n, Cb, hb, nl)
        newM = int(1.3 * n) + 40
        if newM > M:
            new = np.full(newM, -1, np.int64)
            pos = np.sort(np.random.choice(newM, M, replace=False))
            new[pos] = ops
            ops = new
            M = newM
        if nloops is None and t > 20:
            nl = max(10, int(2 * n / 10))   # crude: ~ loops covering the operators
    sg = np.empty(nmeas); m2 = np.empty(nmeas); nn = np.empty(nmeas)
    for t in range(nmeas):
        n = sweep(spin, ops, b, beta, nb, n, Cb, hb, nl)
        sg[t], m2[t], nn[t] = measure(spin, ops)
    E = -nn / beta + Cb.sum()
    return sg, m2, E
