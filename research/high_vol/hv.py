import pandas as pd, numpy as np, itertools
d=pd.read_pickle("full_clean.pkl"); vm=pd.read_csv("daily_vm.csv",index_col=0)
ROLL={'2024-03-12','2024-06-18','2024-09-17','2024-12-17','2025-03-18','2025-06-17','2025-09-16','2025-12-16','2026-03-17','2026-06-16','2026-09-15'}
def sim(m,o,h,l,c,i,side,e,sl,tp):
    for j in range(i,len(m)):
        if m[j]>955: return side*(c[j-1]-e)
        if (l[j]<=sl if side==1 else h[j]>=sl): return side*(sl-e)
        if tp is not None and (h[j]>=tp if side==1 else l[j]<=tp): return side*(tp-e)
    return side*(c[-1]-e)
def b5(m,o,h,l,c,start,end):
    """5m clock bars whose END minute (=next bar start) is in [start,end]; yields (end_m, open, high, low, close)"""
    k=(m-570)//5 if start>=570 else None
    out=[]
    g=pd.DataFrame({"k":np.floor((m-570)/5).astype(int),"o":o,"h":h,"l":l,"c":c,"m":m})
    for kk,x in g.groupby("k",sort=True):
        em=570+5*(kk+1)
        if em<start or em>end: continue
        out.append((em,x.o.iloc[0],x.h.max(),x.l.min(),x.c.iloc[-1]))
    return out
res=[]
for td,x in d.groupby("td",sort=True):
    if td not in vm.index or td in ROLL or np.isnan(vm.loc[td,"adr10"]): continue
    adr=vm.loc[td,"adr10"]; v=vm.loc[td,"vm"]
    gopen=x.open.iloc[0]
    xr=x[(x.m>=570)&(x.m<=959)]
    if len(xr)<300 or xr.m.iloc[0]!=570: continue
    m,o,h,l,c,vol=[xr[f].to_numpy() for f in ["m","open","high","low","close","volume"]]
    idx={mm:i for i,mm in enumerate(m)}
    def add(s,p,side,ei,sl,risk,tps):
        e=o[ei]
        for lab,mult in tps:
            tp=None if mult is None else e+side*mult*risk
            res.append(dict(td=td,vm=v,strat=s,param=p,exit=lab,side=side,entry=e,et=int(m[ei]),risk=risk,pnl=sim(m,o,h,l,c,ei,side,e,sl,tp)*20-48))
    # S1
    for R in (15,30):
        orh=h[m<570+R].max(); orl=l[m<570+R].min()
        for em,bo,bh,bl,bc in b5(m,o,h,l,c,570+R+5,690):
            side=1 if bc>orh else -1 if bc<orl else 0
            if side and em in idx:
                ei=idx[em]; e=o[ei]; sl=orl if side==1 else orh
                if side*(e-sl)>0.35*adr or side*(e-sl)<=0: sl=e-side*0.35*adr
                add("S1",f"R{R}",side,ei,sl,abs(e-sl),[("1R",1),("2R",2),("EOD",None)]); break
    # S2
    mv=c[idx[574]]-o[0] if 574 in idx else 0
    if abs(mv)>=0.15*adr and 575 in idx:
        side=int(np.sign(mv)); ei=idx[575]; e=o[ei]; add("S2","-",side,ei,e-side*0.25*adr,0.25*adr,[("1R",1),("2R",2),("EOD",None)])
    # S3 (extreme measured from Globex open, including overnight)
    xa=x[x.m.notna()]; allb=pd.DataFrame({"k":np.floor((xa.t-xa.t.iloc[0]).dt.total_seconds()/300).astype(int),"o":xa.open.values,"h":xa.high.values,"l":xa.low.values,"c":xa.close.values,"m":xa.m.values})
    # use ET-clock 5m bars: key by floor(minute-of-day/5) with day rollover handled by order
    xa2=xa.assign(key=((xa.t.dt.tz_convert("America/New_York").dt.floor("5min"))))
    bars=xa2.groupby("key",sort=True).agg(h=("high","max"),l=("low","min"),c=("close","last"),m0=("m","first"))
    for k in (0.5,0.7):
        for side in (1,-1):
            ext=None; extbar=None; done=False
            for key,bb in bars.iterrows():
                endm=key.hour*60+key.minute+5; rth_end= key.date()==pd.Timestamp(td).date() and 575<=endm<=780
                if side==1:
                    if ext is None or bb.l<ext: ext=bb.l; extbar=bb.h; continue
                    trig=(gopen-ext)>=k*adr and bb.c>extbar
                else:
                    if ext is None or bb.h>ext: ext=bb.h; extbar=bb.l; continue
                    trig=(ext-gopen)>=k*adr and bb.c<extbar
                if trig and rth_end and endm in idx:
                    ei=idx[endm]; e=o[ei]; sl=ext-side*0.05*adr; risk=side*(e-sl)
                    if risk>0: add("S3",f"k{k}",side,ei,sl,risk,[("2R",2),("3R",3),("EOD",None)])
                    break
                if key.date()==pd.Timestamp(td).date() and endm>780: break
    # S4
    if 629 in idx and 630 in idx:
        i=idx[629]; mv=c[i]-o[0]; tp_=(h[:i+1]+l[:i+1]+c[:i+1])/3; vw=(tp_*np.maximum(vol[:i+1],1)).sum()/np.maximum(vol[:i+1],1).sum()
        if abs(mv)>=0.4*adr and np.sign(c[i]-vw)==np.sign(mv):
            side=int(np.sign(mv)); ei=idx[630]; e=o[ei]; add("S4","-",side,ei,e-side*0.25*adr,0.25*adr,[("1R",1),("2R",2),("EOD",None)])
r=pd.DataFrame(res); r.to_csv("hv_trades.csv",index=False); print(len(r))
