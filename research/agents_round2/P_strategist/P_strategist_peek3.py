import sys; sys.path.insert(0,'.')
import numpy as np
from lib import data, sim
S=sim.slot_of
g=data.load_grid("NQ",include_holdout=True)
ds=[str(d)[:10] for d in g.dates]; i=ds.index("2026-11-11")
O,H,L,C,V=g.O[i],g.H[i],g.L[i],g.C[i],g.V[i]
r0=S("09:30")
tp=(H+L+C)/3
for a in range(S("10:30"),S("14:30"),5):
    e=a+5
    vv=V[r0:e]; t=tp[r0:e]; m=np.isfinite(t)&np.isfinite(vv)
    vw=np.sum(t[m]*vv[m])/np.sum(vv[m]); sd=np.sqrt(np.sum(vv[m]*(t[m]-vw)**2)/np.sum(vv[m]))
    hh=np.nanmax(H[a:a+5]); ll=np.nanmin(L[a:a+5])
    # 12-bar range before
    RH=np.nanmax(H[a-60:a]); RL=np.nanmin(L[a-60:a])
    print(f"{(a-r0)//5*5:4d} O{O[a]:.2f} H{hh:.2f} L{ll:.2f} C{C[a+4]:.2f} V{np.nansum(V[a:a+5]):.0f} vwap{vw:.1f} sd{sd:.1f} U2{vw+2*sd:.1f} RH{RH:.2f} RL{RL:.2f}")
