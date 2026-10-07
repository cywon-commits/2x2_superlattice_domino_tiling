"""Collect runs/*.json (seeds merged per graph and beta) and print the QMC/LSWT comparison tables (markdown)."""
import json, glob, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
ESQ_QMC=-0.669437      # Sandvik 1997, thermodynamic limit
ESQ_LSWT=-0.6579474209435209
runs={}
for f in glob.glob(os.path.join(HERE,'runs','*.json')):
    r=json.load(open(f)); runs.setdefault((r['graph'],r['beta']),[]).append(r)
M={}
for (g,b),rs in runs.items():
    w=np.array([1/r['E_err']**2 for r in rs]); E=np.sum(w*[r['E'] for r in rs])/w.sum(); s=1/np.sqrt(w.sum())
    M[(g,b)]=dict(E=E,s=s,L=rs[0]['E_LSWT'],N=rs[0]['N'],seeds=len(rs),bins=sum(r['n_bins'] for r in rs),
                  sweeps=sum(r['n_bins']*r['sweeps_per_bin'] for r in rs),tau=None)
print('## Raw energies\n\n| graph | beta | N | seeds | sweeps | E_QMC | E_LSWT |\n|---|---|---|---|---|---|---|')
for (g,b),m in sorted(M.items()): print(f"| {g} | {b} | {m['N']} | {m['seeds']} | {m['sweeps']} | {m['E']:.5f} ± {m['s']:.5f} | {m['L']:.5f} |")
def get(g,b): return M.get((g,b))
print('\n## A. zero-field quintet: D(N) = E(2band) - E(1band) = 3*eps5 + dC/sqrt(N)\n')
print('| N | beta | D_QMC | D_LSWT | eps5(N) QMC | eps5(N) LSWT | ratio |\n|---|---|---|---|---|---|---|')
Dq={};Dl={}
for N in (164,656):
    for b in (300,600):
        a1=get(f'facet_1band_N{N}',b); a2=get(f'facet_2band_N{N}',b)
        if a1 and a2:
            d=a2['E']-a1['E']; s=np.hypot(a1['s'],a2['s']); dl=a2['L']-a1['L']; Dq[(N,b)]=(d,s); Dl[N]=dl
            print(f'| {N} | {b} | {d:.4f} ± {s:.4f} | {dl:.4f} | {d/3:.4f} ± {s/3:.4f} | {dl/3:.4f} | {d/dl:.3f} ± {s/dl:.3f} |')
if (164,300) in Dq and (656,300) in Dq:
    x=1/np.sqrt(np.array([164.,656.]))
    def solve(y): A=np.stack([3*np.ones(2),x],1); return np.linalg.solve(A,np.array(y))
    q=solve([Dq[(164,300)][0],Dq[(656,300)][0]]); l=solve([Dl[164],Dl[656]])
    # error propagation
    A=np.stack([3*np.ones(2),x],1); Ai=np.linalg.inv(A); sq=np.sqrt((Ai[0,0]*Dq[(164,300)][1])**2+(Ai[0,1]*Dq[(656,300)][1])**2)
    print(f'\nCasimir-separated eps5 (two-point solve, beta=300): QMC {q[0]:.4f} ± {sq:.4f} J, LSWT {l[0]:.4f} J, ratio {q[0]/l[0]:.2f} ± {sq/l[0]:.2f};  dC: QMC {q[1]:.3f}, LSWT {l[1]:.3f}')
    print('(LSWT eps5 from the full 5-size fit was 0.0347 J)')
print('\n## B. crystals: eps per rhombus = (E - N e_sq)/(L^2 R_cell)\n\n| graph | eps QMC | eps LSWT | ratio |\n|---|---|---|---|')
for (g,b),m in sorted(M.items()):
    if g.startswith('z5crystal') or g.startswith('dice'):
        L=int(g.split('_L')[1].split('_')[0]); Rc=6 if g.startswith('z5') else 8; Rt=Rc*L*L
        eq=(m['E']-m['N']*ESQ_QMC)/Rt; el=(m['L']-m['N']*ESQ_LSWT)/Rt
        print(f"| {g} (beta {b}) | {eq:.5f} ± {m['s']/Rt:.5f} | {el:.5f} | {eq/el:.3f} |")
print('\n## D. ribbon turns (N = 783): eps_t = [E(k3)-E(k9)]/12 ; J1 = [E(k1)-E(k3)-36 eps_t]/54\n')
k={kk:get(f'ribbon_L27_k{kk}_N783',300) for kk in (1,3,9)}
if k[3] and k[9]:
    et=(k[3]['E']-k[9]['E'])/12; set_=np.hypot(k[3]['s'],k[9]['s'])/12; etl=(k[3]['L']-k[9]['L'])/12
    print(f'eps_t: QMC {et:.4f} ± {set_:.4f}, LSWT {etl:.4f}, ratio {et/etl:.2f} ± {set_/etl:.2f}')
    if k[1]:
        j=(k[1]['E']-k[3]['E']-36*et)/54; jl=(k[1]['L']-k[3]['L']-36*etl)/54
        sj=np.sqrt(k[1]['s']**2+k[3]['s']**2+(36*set_)**2)/54
        print(f'J1:    QMC {j:.4f} ± {sj:.4f}, LSWT {jl:.4f}, ratio {j/jl:.2f} ± {sj/jl:.2f}')
