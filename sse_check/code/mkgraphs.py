import sys, pickle, json, numpy as np
B='/tmp/claude-0/-home-user-2x2-superlattice-domino-tiling/11418d45-3125-5b14-813d-71e8685c9abc/scratchpad/'
sys.path.insert(0,B+'z/bigcell_task/code')
from dice_string import Tiling
from qmc_tiling import lieb
for k in (36,18,12,9):
    T=Tiling(36); T.removed=pickle.load(open(B+f'z/bigcell_task/ref/big_m{k}_L36.pkl','rb'))
    b=np.array(T.bonds()).tolist(); S=lieb(T)
    json.dump(dict(name=f'big_m{k}_L36',N=int(T.N),bonds=b,S_lieb=S),open(B+f'run/det/big_m{k}_L36.json','w')); print(k,T.N,len(b),S)
