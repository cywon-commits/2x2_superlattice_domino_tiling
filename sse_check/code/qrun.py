import sys, json, time, numpy as np
B='/tmp/claude-0/-home-user-2x2-superlattice-domino-tiling/11418d45-3125-5b14-813d-71e8685c9abc/scratchpad/'
sys.path.insert(0, B+'run')
import sse_h2 as sse_h   # sse_h.py with the loop-length cap removed (cap truncated loops -> sign violations at beta>=30)
name, beta, ntherm, nmeas, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
d = json.load(open(B+f'z4/qmc_inputs/{name}.json'))
t = time.time()
sg, m2, E = sse_h.run(np.array(d['bonds']), d['N'], beta, ntherm, nmeas, seed)
def tau_int(x):
    x = x - x.mean(); n = len(x)
    f = np.fft.rfft(x, 2*n); ac = np.fft.irfft(f*np.conj(f))[:n]; ac /= ac[0]
    tau = 0.5
    for M in range(1, n):
        tau += ac[M]
        if M >= 6*tau: break
    return max(tau, 0.5)
tau = tau_int(E)
err = float(E.std(ddof=1)*np.sqrt(2*tau/len(E)))
print(json.dumps(dict(name=name, N=d['N'], beta=beta, nmeas=nmeas, ntherm=ntherm, seed=seed, E=float(E.mean()), err=err, tau_int=float(tau), n_eff=float(len(E)/(2*tau)), E_LSWT=d['E_LSWT'], sign=float(sg.mean()), t=time.time()-t)), flush=True)
