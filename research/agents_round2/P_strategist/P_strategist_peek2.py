import sys; sys.path.insert(0,'.')
import numpy as np
from lib import data, features, sim
S=sim.slot_of
for sym in ["NQ","ES"]:
    g=data.load_grid(sym,include_holdout=True)
    ds=[str(d)[:10] for d in g.dates]; i=ds.index("2026-11-11")
    H,L,C,O=g.H[i],g.L[i],g.C[i],g.O[i]
    print(sym)
    for a in range(S("09:00"),S("11:05"),5):
        print(" ", a-S("09:30"), np.nanmax(H[a:a+5]), np.nanmin(L[a:a+5]), C[a+4])
    if sym=="NQ":
        # NWOG: last friday close and monday open
        k=ds.index("2026-11-09"); f=ds.index("2026-11-06")
        cf=g.C[f]; lastf=np.where(~np.isnan(cf))[0][-1]
        om=g.O[k]; firstm=np.where(~np.isnan(om))[0][0]
        print("Fri close slot",lastf,cf[lastf],"Mon open slot",firstm,om[firstm])
        # ADR10
        rh=g.rth(g.H); rl=g.rth(g.L)
        print("ADR10", np.mean([np.nanmax(rh[j])-np.nanmin(rl[j]) for j in range(i-10,i)]))
        # 15m bars 02:00-09:30 closes
        for a in range(S("00:00"),S("09:30"),15):
            print(" 15m", a, np.nanmax(H[a:a+15]), np.nanmin(L[a:a+15]), O[a], C[a+14])
        for a in range(S("12:00"),S("16:00"),15):
            print(" 15mPM", a-S("09:30"), O[a], np.nanmax(H[a:a+15]), np.nanmin(L[a:a+15]), C[a+14])
