import sys; sys.path.insert(0,'.')
import numpy as np
from lib import data, features, sim
g=data.load_grid("NQ",include_holdout=True)
lv=features.daily_levels(g)
S=sim.slot_of
ds=[str(d)[:10] for d in g.dates]
i=ds.index("2026-11-11"); j=ds.index("2026-11-10")
O,H,L,C,V=g.O[i],g.H[i],g.L[i],g.C[i],g.V[i]
def rng(a,b): return np.nanmax(H[S(a):S(b)+1]), np.nanmin(L[S(a):S(b)+1])
print(lv.iloc[i][["pdh","pdl","pdc","vah","val","poc","onh","onl","open_0930"]].to_dict())
for nm,(a,b) in {"asia18-24":("18:00","23:59"),"asia20-24":("20:00","23:59"),"lon02-05":("02:00","04:59"),"pre0830-0929":("08:30","09:29"),"or5":("09:30","09:34"),"or15":("09:30","09:44"),"or30":("09:30","09:59"),"ib":("09:30","10:29"),"10-11":("10:00","10:59"),"rth":("09:30","15:59")}.items():
    print(nm, rng(a,b))
print("open18",O[0],"mid open",O[S("00:00")],"0830",O[S("08:30")],"close1559",C[S("15:59")])
print("prev 1659 close?", g.C[j][-1] if len(g.C[j])>1380 else None, "prev C last", g.C[j][np.where(~np.isnan(g.C[j]))[0][-1]])
# minute of rth low/high
r0=S("09:30"); r1=S("15:59")
lo=np.nanargmin(L[r0:r1+1]); hi=np.nanargmax(H[r0:r1+1])
print("rth low at +",lo, L[r0+lo], "high at +",hi, H[r0+hi])
# 5-min bars 09:30-11:30
for a in range(S("09:30"),S("11:30"),5):
    o=O[a];h=np.nanmax(H[a:a+5]);l=np.nanmin(L[a:a+5]);c=C[a+4]
    print(sim.__dict__.get('hhmm',lambda x:x)(a) if False else a-S("09:30"), round(o,2),round(h,2),round(l,2),round(c,2))
# vwap
tp=(H+L+C)/3; m=np.isfinite(tp)
cv=np.nancumsum(np.where(m,tp*V,0))[r0:]-0; 
vv=V[r0:r1+1]; t=tp[r0:r1+1]
cvw=np.nancumsum(t*vv)/np.nancumsum(vv)
print("vwap at 10:00,10:30,11:00,12:00", cvw[30],cvw[60],cvw[90],cvw[150])
# prior days high/lows of week
for k in range(1,8):
    jj=i-k
    rh=g.rth(g.H)[jj]; rl=g.rth(g.L)[jj]
    print(ds[jj], np.nanmax(rh), np.nanmin(rl))
print("----")
def t(s): h=(18*60+s)//60%24; m=(18*60+s)%60; return f"{h:02d}:{m:02d}"
fv=np.where(np.isfinite(O))[0]; print("first valid", t(fv[0]), O[fv[0]], "n valid", len(fv))
for hh in ["00:00","00:05","08:30","03:00","10:00","14:00"]:
    s=S(hh); k=s
    while np.isnan(O[k]): k+=1
    print(hh, t(k), O[k])
on=slice(0,S("09:29")+1)
print("ONH at", t(np.nanargmax(H[on])), "ONL at", t(np.nanargmin(L[on])))
print("LON low at", t(S("02:00")+np.nanargmin(L[S("02:00"):S("05:00")])))
# 1m FVG list 09:31-11:00 (bull: L[k]>H[k-2])
for k in range(S("09:32"),S("11:00")):
    if L[k]>H[k-2]: print("bullFVG closes", t(k), "gap", H[k-2], L[k])
    if H[k]<L[k-2]: print("bearFVG closes", t(k), "gap", H[k], L[k-2])
