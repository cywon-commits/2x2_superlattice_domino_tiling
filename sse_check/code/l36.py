"""Re-run a published L=36 crystal with loop-cap instrumentation. usage: l36.py pkl h seed capmode(orig|nocap) ntherm nmeas tag"""
import sys, json, time, pickle, os, numpy as np
B='/tmp/claude-0/-home-user-2x2-superlattice-domino-tiling/11418d45-3125-5b14-813d-71e8685c9abc/scratchpad/'
sys.path.insert(0,B+'run'); sys.path.insert(0,B+'z/bigcell_task/code')
from dice_string import Tiling
from qmc_tiling import lieb
import sse_h3
from sse_h3 import sweep, measure, seed, EPS
pk,h,sd,mode,ntherm,nmeas,tag=sys.argv[1],float(sys.argv[2]),int(sys.argv[3]),sys.argv[4],int(sys.argv[5]),int(sys.argv[6]),sys.argv[7]
L=36; beta=36.0
ck=B+f'run/ck_{tag}.pkl'
T=Tiling(L); T.removed=pickle.load(open(B+'z/bigcell_task/ref/'+pk,'rb')); S=lieb(T); N=T.N
b=np.ascontiguousarray(T.bonds(),np.int64); nb=len(b)
z=np.bincount(b.ravel(),minlength=N).astype(float)
hb=np.ascontiguousarray(np.stack([h/z[b[:,0]],h/z[b[:,1]]],1)); Cb=0.25+0.5*(hb[:,0]+hb[:,1])+EPS
tr=np.zeros(3,np.int64)
if os.path.exists(ck):
    st=pickle.load(open(ck,'rb')); spin,ops,n,nl,t,Es,Ms,Sg,tr=st['spin'],st['ops'],st['n'],st['nl'],st['t'],st['E'],st['M'],st['S'],st['tr']
    seed(sd*1000+t); np.random.seed(sd*1000+t)
else:
    np.random.seed(sd); seed(sd); spin=np.where(np.random.random(N)<0.5,1,-1).astype(np.int64)
    ops=np.full(max(40,int(beta*nb*0.3)),-1,np.int64); n=0; nl=10; t=0; Es=[]; Ms=[]; Sg=[]
t0=time.time(); last=t0
while t<ntherm+nmeas:
    cap = 100*len(ops) if mode=='orig' else 10**12
    n=sweep(spin,ops,b,beta,nb,n,Cb,hb,nl,cap,tr)
    if t<ntherm:
        M=len(ops); newM=int(1.3*n)+40
        if newM>M:
            new=np.full(newM,-1,np.int64); pos=np.sort(np.random.choice(newM,M,replace=False)); new[pos]=ops; ops=new
        if t>20: nl=max(10,int(2*n/10))
    else:
        sg,mz,nn=measure(spin,ops); Es.append(-nn/beta+Cb.sum()); Ms.append(mz); Sg.append(sg)
    t+=1
    if time.time()-last>90:
        pickle.dump(dict(spin=spin,ops=ops,n=n,nl=nl,t=t,E=Es,M=Ms,S=Sg,tr=tr),open(ck+'.tmp','wb')); os.replace(ck+'.tmp',ck); last=time.time()
        print(f"t={t}/{ntherm+nmeas} tr={tr.tolist()}",flush=True)
E=np.array(Es); Mz=np.array(Ms); nbk=20
Eb=E[:len(E)//nbk*nbk].reshape(nbk,-1).mean(1)/N; Mb=Mz[:len(Mz)//nbk*nbk].reshape(nbk,-1).mean(1)/N
print(json.dumps(dict(tag=tag,pkl=pk,mode=mode,beta=beta,h=h,seed=sd,ntherm=ntherm,nmeas=nmeas,S_lieb=S,e=float(Eb.mean()),e_err=float(Eb.std()/np.sqrt(nbk)),m=float(np.abs(Mb).mean()),sign_mean=float(np.mean(Sg)),sign_min=int(np.min(Sg)),loops_over_cap=int(tr[0]),loops_truncated=int(tr[1]),max_loop_steps=int(tr[2]),maxM=len(ops))),flush=True)
