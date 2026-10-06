import sys, json, os
B='/tmp/claude-0/-home-user-2x2-superlattice-domino-tiling/11418d45-3125-5b14-813d-71e8685c9abc/scratchpad/'
sys.path.insert(0,B+'z5/sse'); from sse_det import simulate
name,beta,mode,h,seed=sys.argv[1],float(sys.argv[2]),sys.argv[3],float(sys.argv[4]),int(sys.argv[5])
out=B+f'run/det/res_{name}_b{int(beta)}_{mode}_s{seed}.json'
if os.path.exists(out): sys.exit()
g=json.load(open(B+f'run/det/{name}.json'))
r=simulate(g['N'],g['bonds'],beta,h if mode=='field' else 0.0,2000,100,500,seed,verbose=False,fixM=(g['S_lieb'] if mode=='fix' else None))
r['name']=name; r['mode']=mode; r['S_lieb']=g['S_lieb']; r['h_pub']=h
r['e_total']=(r['E_J']-h*(g['S_lieb'] if mode=='fix' else r['M']))/g['N']    # (E_J - h M)/N, same quantity as published e
r['e_total_err']=(r['E_J_err'] if mode=='fix' else r['E_err'])/g['N']
r.pop('bins_E'); json.dump(r,open(out,'w'))
print(name,beta,mode,seed,r['e_total'],r['e_total_err'],r['M'],r['sweeps_per_s'],flush=True)
