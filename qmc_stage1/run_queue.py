"""Resumable, parallel SSE job queue.  Usage:  python run_queue.py -j 4  [--only A|B|D] [--extra-seeds 2]
Each finished job is written to runs/<graph>_b<beta>_s<seed>.json and skipped on restart."""
import json, os, sys, time, argparse
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'sse'))
IN=os.path.join(HERE,'qmc_inputs'); OUT=os.path.join(HERE,'runs'); os.makedirs(OUT,exist_ok=True)
# (group, graph, beta, n_bins, sweeps_per_bin)   ~ wall time on one core (numba): see TASK.md
JOBS=[('D','ribbon_L27_k9_N783',300,40,500),('D','ribbon_L27_k1_N783',300,40,500),
      ('A','facet_1band_N164',600,40,500),('A','facet_2band_N164',600,40,500),
      ('A','facet_1band_N656',300,40,1000),('A','facet_2band_N656',300,40,1000),
      ('B','z5crystal_L6_N540',300,40,300),('B','dice_L6_N432',300,40,300)]
def run(job):
    grp,name,beta,nb,spb,seed=job
    fn=os.path.join(OUT,f'{name}_b{beta}_s{seed}.json')
    if os.path.exists(fn): return f'skip {os.path.basename(fn)}'
    from sse_det import simulate
    g=json.load(open(os.path.join(IN,name+'.json'))); t=time.time()
    r=simulate(g['N'],g['bonds'],float(beta),0.0,1000,nb,spb,seed,verbose=False)
    r.update(graph=name,group=grp,E_LSWT=g['E_LSWT'],wall_s=round(time.time()-t,1))
    json.dump(r,open(fn+'.tmp','w'),indent=1); os.replace(fn+'.tmp',fn)
    return f"{name} beta={beta} seed={seed}: E={r['E']:.5f}+-{r['E_err']:.5f} (LSWT {g['E_LSWT']:.5f}) {r['wall_s']}s"
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('-j',type=int,default=1); ap.add_argument('--only',default=None)
    ap.add_argument('--extra-seeds',type=int,default=0,help='additional independent seeds per job (for more statistics)')
    a=ap.parse_args()
    jobs=[(g,n,b,nb,s,seed) for g,n,b,nb,s in JOBS if a.only is None or g in a.only for seed in range(2,3+a.extra_seeds)]
    jobs.sort(key=lambda j:-j[2]*int(j[1].split('_N')[-1]))      # longest first
    with Pool(a.j) as p:
        for msg in p.imap_unordered(run,jobs): print(msg,flush=True)
